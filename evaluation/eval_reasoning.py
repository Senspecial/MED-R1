"""
Evaluate reasoning process quality using PRM (Process Reward Model).

Metrics:
  1. Answer Accuracy  — is the final answer correct?
  2. Step Scores      — PRM score for each reasoning step
  3. Reasoning Quality — average step score across all problems
  4. Correlation       — do higher PRM scores predict correct answers?

Usage:
    # 8-GPU parallel — evaluate GRPO model
    python3 evaluation/eval_reasoning.py \
        --model_name /path/to/grpo_model \
        --prm_path /path/to/prm_model \
        --eval_file evaluation/data/eval_data.json \
        --num_gpus 8 \
        --max_new_tokens 1200

    # Compare with SFT baseline
    python3 evaluation/eval_reasoning.py \
        --model_name /path/to/sft_model \
        --prm_path /path/to/prm_model \
        --eval_file evaluation/data/eval_data.json \
        --num_gpus 8

    # Single GPU (quick test)
    python3 evaluation/eval_reasoning.py \
        --model_name /path/to/grpo_model \
        --prm_path /path/to/prm_model \
        --eval_file evaluation/data/eval_data.json \
        --limit 100 --gpu_id 0
"""

import argparse
import json
import os
import subprocess
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template
from scorer import match_choice, get_results


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred


class PRMScorer:
    """Load PRM and score reasoning steps with single forward pass."""
    def __init__(self, prm_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(prm_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            prm_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
        base_model.eval()

        hidden_size = base_model.config.hidden_size
        self.base_model = base_model
        self.reward_head = torch.nn.Linear(hidden_size, 1).to(device)

        head_path = os.path.join(prm_path, "reward_head.pt")
        if os.path.exists(head_path):
            self.reward_head.load_state_dict(torch.load(head_path, map_location=device))
            self.reward_head.eval()
        else:
            print(f"Warning: reward_head.pt not found at {prm_path}, using random head")

        self.step_sep = '\n\n'
        self.step_sep_ids = self.tokenizer.encode(self.step_sep, add_special_tokens=False)

    def _find_step_boundary_indices(self, input_ids):
        """Find token indices where each step ends (matching \\n\\n boundaries)."""
        ids = input_ids.tolist()
        sep = self.step_sep_ids
        sep_len = len(sep)
        boundaries = []
        for i in range(len(ids) - sep_len + 1):
            if ids[i:i+sep_len] == sep:
                boundaries.append(i + sep_len - 1)
        if not boundaries or boundaries[-1] != len(ids) - 1:
            boundaries.append(len(ids) - 1)
        return boundaries

    @torch.no_grad()
    def score_steps(self, question, reasoning_text):
        steps = [s.strip() for s in reasoning_text.split(self.step_sep) if s.strip()]
        if not steps:
            return [], 0.0

        full_text = question + self.step_sep + (self.step_sep).join(steps)
        inputs = self.tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=4096
        ).to(self.device)

        outputs = self.base_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1].squeeze(0)

        boundaries = self._find_step_boundary_indices(inputs.input_ids.squeeze(0))

        question_ids = self.tokenizer.encode(question, add_special_tokens=False)
        q_len = len(question_ids)
        step_boundaries = [b for b in boundaries if b >= q_len]

        if not step_boundaries:
            step_boundaries = boundaries[-len(steps):] if len(boundaries) >= len(steps) else boundaries

        step_boundaries = step_boundaries[:len(steps)]

        boundary_hidden = hidden_states[step_boundaries]
        scores = torch.sigmoid(self.reward_head(boundary_hidden)).squeeze(-1)
        step_scores = scores.cpu().tolist()

        avg_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        return step_scores, avg_score


def extract_reasoning_and_answer(text):
    """Split model output into reasoning part and final answer part."""
    if '## Final Response' in text:
        parts = text.split('## Final Response')
        reasoning = parts[0].replace('## Thinking', '').strip()
        answer = parts[1].strip() if len(parts) > 1 else ''
    elif '</think>' in text:
        parts = text.split('</think>')
        reasoning = parts[0].replace('<think>', '').strip()
        answer = parts[1].strip() if len(parts) > 1 else ''
    else:
        reasoning = text
        answer = text
    return reasoning, answer


def run_shard(args):
    """Run evaluation on a single GPU shard."""
    device = torch.device(f"cuda:0")

    print(f"[GPU {args.gpu_id}] Loading generation model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = Template(tokenizer.chat_template) if tokenizer.chat_template else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print(f"[GPU {args.gpu_id}] Loading PRM: {args.prm_path}")
    prm = PRMScorer(args.prm_path, device)

    with open(args.eval_file) as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {'normal': data}
    input_data = []
    for k, v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)

    if args.limit > 0:
        input_data = input_data[:args.limit]

    shard_size = (len(input_data) + args.num_gpus - 1) // args.num_gpus
    shard = input_data[args.gpu_id * shard_size : (args.gpu_id + 1) * shard_size]
    print(f"[GPU {args.gpu_id}] Processing {len(shard)}/{len(input_data)} samples")

    if args.strict_prompt:
        query_prompt = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"

    for item in shard:
        item['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in item['options'].items()])
        item['input_str'] = query_prompt.format_map(item)

    local_batch_size = 4
    results = []

    for i in tqdm(range(0, len(shard), local_batch_size),
                  desc=f"GPU {args.gpu_id}", position=args.gpu_id):
        batch = shard[i:i+local_batch_size]
        prompts = [item['input_str'] for item in batch]

        if template:
            prompts = [template.render(
                messages=[{"role": "user", "content": p}],
                bos_token=tokenizer.bos_token,
                add_generation_prompt=True
            ) for p in prompts]

        if args.max_tokens > 0:
            truncated = []
            for p in prompts:
                ids = tokenizer.encode(p, add_special_tokens=False)
                if len(ids) > args.max_tokens:
                    truncated.append(tokenizer.decode(ids[:args.max_tokens]))
                else:
                    truncated.append(p)
            prompts = truncated

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                temperature=0.1, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        for j, item in enumerate(batch):
            full_output = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            full_output = postprocess_output(full_output)
            if not full_output:
                continue

            reasoning, answer_text = extract_reasoning_and_answer(full_output)
            ans, ans_type = match_choice(full_output, item['options'])
            is_correct = ans[0].lower() == item['answer_idx'].lower()
            step_scores, avg_prm = prm.score_steps(item['input_str'], reasoning)

            results.append({
                'question': item['question'],
                'answer_idx': item['answer_idx'],
                'predicted': ans[0],
                'correct': is_correct,
                'output': full_output,
                'reasoning': reasoning[:500],
                'step_scores': step_scores,
                'avg_prm_score': avg_prm,
                'num_steps': len(step_scores),
                'source': item.get('source', 'unknown'),
                'options': item['options'],
            })

    shard_path = f"/tmp/eval_reasoning_shard_{args.gpu_id}.json"
    with open(shard_path, 'w') as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)
    print(f"[GPU {args.gpu_id}] Done. Saved {len(results)} results to {shard_path}")


def aggregate_and_report(num_gpus, model_name, eval_file, task='api', strict_prompt=False):
    """Merge shard results and compute final metrics."""
    all_results = []
    for gid in range(num_gpus):
        shard_path = f"/tmp/eval_reasoning_shard_{gid}.json"
        if os.path.exists(shard_path):
            with open(shard_path) as f:
                all_results.extend(json.load(f))
            os.remove(shard_path)

    n = len(all_results)
    if n == 0:
        print("No results collected!")
        return

    correct_count = sum(1 for r in all_results if r['correct'])
    all_prm = [r['avg_prm_score'] for r in all_results]
    correct_prm = [r['avg_prm_score'] for r in all_results if r['correct']]
    wrong_prm = [r['avg_prm_score'] for r in all_results if not r['correct']]
    all_steps = [r['num_steps'] for r in all_results]

    accuracy = correct_count / n
    avg_prm = sum(all_prm) / n
    avg_correct_prm = sum(correct_prm) / len(correct_prm) if correct_prm else 0
    avg_wrong_prm = sum(wrong_prm) / len(wrong_prm) if wrong_prm else 0
    avg_steps = sum(all_steps) / n

    by_source = {}
    for r in all_results:
        src = r['source']
        if src not in by_source:
            by_source[src] = {'correct': 0, 'total': 0, 'prm_sum': 0.0}
        by_source[src]['total'] += 1
        by_source[src]['prm_sum'] += r['avg_prm_score']
        if r['correct']:
            by_source[src]['correct'] += 1

    print("\n" + "=" * 60)
    print(f"Reasoning Evaluation Results ({n} samples)")
    print("=" * 60)
    print(f"  Answer Accuracy:           {accuracy:.4f} ({correct_count}/{n})")
    print(f"  Avg Reasoning Steps:       {avg_steps:.1f}")
    print(f"  Avg PRM Score (all):       {avg_prm:.4f}")
    print(f"  Avg PRM Score (correct):   {avg_correct_prm:.4f}")
    print(f"  Avg PRM Score (wrong):     {avg_wrong_prm:.4f}")
    print(f"  PRM Score Gap:             {avg_correct_prm - avg_wrong_prm:+.4f}")
    print("-" * 60)
    print("  Per-source breakdown:")
    for src, stats in sorted(by_source.items()):
        src_acc = stats['correct'] / stats['total']
        src_prm = stats['prm_sum'] / stats['total']
        print(f"    {src:30s}  acc={src_acc:.4f}  prm={src_prm:.4f}  n={stats['total']}")
    print("=" * 60)

    if avg_correct_prm > avg_wrong_prm:
        print("  PRM is discriminative: correct answers have higher process scores")
    else:
        print("  Warning: PRM may need more training")

    task_name = os.path.split(model_name)[-1]
    task_name = task_name + os.path.basename(eval_file).replace('.json', '') + f'_{task}' + ('_strict-prompt' if strict_prompt else '')
    save_path = f'{task_name}_reasoning.json'

    with open(save_path, 'w') as fw:
        json.dump({
            'metrics': {
                'accuracy': accuracy,
                'avg_steps': avg_steps,
                'avg_prm_score': avg_prm,
                'avg_prm_correct': avg_correct_prm,
                'avg_prm_wrong': avg_wrong_prm,
                'prm_gap': avg_correct_prm - avg_wrong_prm,
                'total_samples': n,
                'per_source': {
                    src: {
                        'accuracy': s['correct'] / s['total'],
                        'avg_prm': s['prm_sum'] / s['total'],
                        'count': s['total']
                    } for src, s in by_source.items()
                }
            },
            'details': all_results
        }, fw, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {save_path}")

    compatible_results = [r for r in all_results if 'output' in r]
    if compatible_results:
        compat_path = f'{task_name}.json'
        with open(compat_path, 'w') as fw:
            json.dump(compatible_results, fw, ensure_ascii=False, indent=2)
        print(f"\nscorer-compatible results saved to {compat_path}")
        get_results(compat_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                        help="Path to the model to evaluate (GRPO model or SFT model)")
    parser.add_argument('--prm_path', type=str, required=True, help="Path to trained PRM model")
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=1200,
                        help="Max generation tokens (recommended 1200~1500)")
    parser.add_argument('--max_tokens', type=int, default=-1,
                        help="Max input tokens (-1 for no truncation)")
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=-1, help="Internal: set automatically for shards")
    parser.add_argument('--limit', type=int, default=-1, help="Limit number of samples (-1 for all)")
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str, default='api')
    args = parser.parse_args()

    if args.gpu_id >= 0:
        run_shard(args)
        return

    if args.num_gpus > 1:
        print(f"Launching {args.num_gpus}-GPU parallel reasoning evaluation...")
        procs = []
        for gid in range(args.num_gpus):
            cmd = [
                sys.executable, __file__,
                '--model_name', args.model_name,
                '--prm_path', args.prm_path,
                '--eval_file', args.eval_file,
                '--max_new_tokens', str(args.max_new_tokens),
                '--max_tokens', str(args.max_tokens),
                '--num_gpus', str(args.num_gpus),
                '--gpu_id', str(gid),
                '--limit', str(args.limit),
                '--task', args.task,
            ]
            if args.strict_prompt:
                cmd.append('--strict_prompt')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gid)
            procs.append(subprocess.Popen(cmd, env=env))

        for p in procs:
            p.wait()

        aggregate_and_report(args.num_gpus, args.model_name, args.eval_file,
                             args.task, args.strict_prompt)
    else:
        args.gpu_id = 0
        args.num_gpus = 1
        run_shard(args)
        aggregate_and_report(1, args.model_name, args.eval_file,
                             args.task, args.strict_prompt)


if __name__ == "__main__":
    main()
