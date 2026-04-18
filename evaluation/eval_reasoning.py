"""
Evaluate reasoning process quality using PRM (Process Reward Model).

Metrics:
  1. Answer Accuracy  — is the final answer correct?
  2. Step Scores      — PRM score for each reasoning step
  3. Reasoning Quality — average step score across all problems
  4. Correlation       — do higher PRM scores predict correct answers?

Usage:
    # 8-GPU parallel — evaluate GRPO model (HuggingFace format)
    python3 evaluation/eval_reasoning.py \
        --model_name /path/to/grpo_model \
        --prm_path /path/to/prm_model \
        --eval_file evaluation/data/eval_data.json \
        --num_gpus 8 \
        --max_new_tokens 1200

    # Directly evaluate FSDP checkpoint (auto-merge in memory, no disk save)
    python3 evaluation/eval_reasoning.py \
        --model_name ./ckpts/dapo/global_step_150/actor \
        --prm_path /path/to/prm_model \
        --eval_file evaluation/data/eval_data.json \
        --num_gpus 8

    # Single GPU (quick test)
    python3 evaluation/eval_reasoning.py \
        --model_name ./ckpts/dapo/global_step_150/actor \
        --prm_path /path/to/prm_model \
        --eval_file evaluation/data/eval_data.json \
        --limit 100
"""

import argparse
import json
import os
import subprocess
import sys
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from jinja2 import Template
from scorer import match_choice, get_results


def load_fsdp_model_to_hf(actor_dir, device):
    """
    Load FSDP sharded checkpoint directly into a HuggingFace model in memory.
    No files are saved to disk.

    Args:
        actor_dir: Path to the actor directory containing FSDP shards
                   (e.g., ./ckpts/dapo/global_step_150/actor)
        device: Target torch device

    Returns:
        model: HuggingFace model with merged weights
        tokenizer: Corresponding tokenizer
    """
    from torch.distributed._tensor import Shard as DTShard

    actor_dir = Path(actor_dir)
    hf_dir = actor_dir / "huggingface"

    # 1. Load config and tokenizer from huggingface subdir
    print(f"  Loading config & tokenizer from {hf_dir}")
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Read FSDP config
    fsdp_cfg_path = actor_dir / "fsdp_config.json"
    with open(fsdp_cfg_path) as f:
        fsdp_cfg = json.load(f)
    world_size = fsdp_cfg["world_size"]
    print(f"  FSDP world_size={world_size}, loading {world_size} shards...")

    # 3. Load all shards in parallel
    shard_list = [None] * world_size

    def _load_shard(rank):
        p = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        sd = torch.load(p, map_location="cpu", weights_only=False)
        shard_list[rank] = sd

    with ThreadPoolExecutor(max_workers=min(world_size, os.cpu_count() or 4)) as pool:
        futs = [pool.submit(_load_shard, r) for r in range(world_size)]
        for fut in tqdm(futs, desc="Loading FSDP shards", total=world_size):
            fut.result()

    # 4. Detect mesh info from rank-0 shard
    from torch.distributed._tensor import DTensor
    pivot_key = sorted(shard_list[0].keys())[0]
    sample = shard_list[0][pivot_key]
    if isinstance(sample, DTensor):
        mesh = sample.device_mesh.mesh
        mesh_dim_names = sample.device_mesh.mesh_dim_names
    else:
        mesh = np.array([world_size], dtype=np.int64)
        mesh_dim_names = ("fsdp",)
    total_shards = int(np.prod(mesh))

    # 5. Merge state dict
    print(f"  Merging {total_shards} shards (mesh={mesh}, dims={mesh_dim_names})...")
    merged = {}
    param_placements = {}

    for key in set(shard_list[0].keys()):
        merged[key] = []
        for sd in shard_list:
            tensor = sd.pop(key)
            if isinstance(tensor, DTensor):
                merged[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] in ("dp", "ddp"):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
            else:
                merged[key].append(tensor.bfloat16())

    del shard_list

    for key in sorted(merged):
        if not isinstance(merged[key], list):
            continue
        if key in param_placements:
            placements = param_placements[key]
            assert len(placements) == 1, "FSDP+TP not supported"
            p = placements[0]
            if isinstance(p, DTShard):
                merged[key] = torch.cat(merged[key], dim=p.dim)
            else:
                # Replicated
                merged[key] = merged[key][0]
        else:
            merged[key] = torch.cat(merged[key], dim=0)

    # 6. Build empty model and load state dict
    print(f"  Loading merged weights into model...")
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.load_state_dict(merged, strict=True)
    del merged

    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully!")
    return model, tokenizer


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred


class PRMScorer:
    """Load PRM and score reasoning steps with single forward pass.

    Matches the training-time scoring logic (reward_server.py / grpo_trainer.py):
      - Input format:  "Question: {q}\\n\\nReasoning:\\n\\n{step1}\\n\\n{step2}..."
      - Boundary detection via offset_mapping (char→token)
      - Aggregation: min / mean / last / weighted_mean  (default: min)
    """
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
        # 确保 reward_head 和 base_model 使用相同的 dtype
        self.reward_head = self.reward_head.to(torch.bfloat16)

        self.step_sep = '\n\n'

    @staticmethod
    def _aggregate(scores, agg="min"):
        """Aggregate step scores — same logic as reward_server._aggregate."""
        if not scores:
            return 0.0
        if agg == "min":
            return min(scores)
        if agg == "mean":
            return sum(scores) / len(scores)
        if agg == "last":
            return scores[-1]
        if agg == "weighted_mean":
            weights = list(range(1, len(scores) + 1))
            return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return min(scores)  # fallback

    @torch.no_grad()
    def score_steps(self, question, reasoning_text, agg="min"):
        """Score each reasoning step and return (step_scores, aggregated_score).

        Uses the same input format and offset_mapping boundary detection as
        reward_server.py ``_score_single`` to ensure train/eval consistency.
        """
        steps = [s.strip() for s in reasoning_text.split(self.step_sep) if s.strip()]
        if not steps:
            return [], 0.0

        # ── Build input text with the same prefix as training ──
        prefix = f"Question: {question}\n\nReasoning:\n\n"
        full_text = prefix + "\n\n".join(steps)

        encoding = self.tokenizer(
            full_text, add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=True, max_length=4096,
        )
        full_tokens = encoding["input_ids"]
        offsets = encoding.get("offset_mapping")

        # ── Locate step-end token positions via offset_mapping ──
        if offsets is not None:
            step_end_chars = []
            char_pos = len(prefix)
            for i, step in enumerate(steps):
                char_pos += len(step)
                step_end_chars.append(char_pos - 1)
                if i < len(steps) - 1:
                    char_pos += 2  # len("\n\n")

            step_end_positions = []
            for target in step_end_chars:
                pos = len(full_tokens) - 1  # fallback to last token
                for tok_idx in range(len(offsets)):
                    start, end = offsets[tok_idx]
                    if start <= target < end:
                        pos = tok_idx
                        break
                step_end_positions.append(pos)
        else:
            # Fallback: re-tokenize partial sequences
            step_end_positions = []
            for i in range(len(steps)):
                partial = prefix + "\n\n".join(steps[:i + 1])
                partial_tokens = self.tokenizer.encode(partial, add_special_tokens=True)
                step_end_positions.append(len(partial_tokens) - 1)

        # ── Forward pass ──
        input_ids = torch.LongTensor([full_tokens]).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden)
        seq_len = hidden_states.size(1)

        # ── Extract hidden states at step boundaries and score ──
        step_scores = []
        for pos in step_end_positions:
            pos = min(pos, seq_len - 1)
            h = hidden_states[0, pos]
            s = torch.sigmoid(self.reward_head(h)).item()
            step_scores.append(s)

        aggregated = self._aggregate(step_scores, agg)
        return step_scores, aggregated


def extract_reasoning_and_answer(text):
    """Split model output into reasoning part and final answer part.

    Handles mixed formats where models may output both
    <think>...</think> AND ## Thinking / ## Final Response tags.
    """
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

    # Clean up residual <think>/<\/think> tags (models may output both
    # <think>...</think> and ## Thinking / ## Final Response simultaneously)
    reasoning = reasoning.replace('<think>', '').replace('</think>', '').strip()

    return reasoning, answer


def _is_fsdp_checkpoint(path):
    """Check if the path points to an FSDP sharded checkpoint (actor directory)."""
    p = Path(path)
    fsdp_cfg = p / "fsdp_config.json"
    hf_dir = p / "huggingface"
    has_shards = any(p.glob("model_world_size_*_rank_*.pt"))
    # It's FSDP if: has fsdp_config.json + shard files, and huggingface/ lacks weight files
    if fsdp_cfg.exists() and has_shards:
        hf_weights = list(hf_dir.glob("*.safetensors")) + list(hf_dir.glob("pytorch_model*.bin"))
        if not hf_weights:
            return True
    return False


def run_shard(args):
    """Run evaluation on a single GPU shard."""
    device = torch.device(f"cuda:0")

    print(f"[GPU {args.gpu_id}] Loading generation model: {args.model_name}")

    if _is_fsdp_checkpoint(args.model_name):
        print(f"[GPU {args.gpu_id}] Detected FSDP checkpoint, merging shards in memory (no disk save)...")
        model, tokenizer = load_fsdp_model_to_hf(args.model_name, device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
        model.eval()

    template = Template(tokenizer.chat_template) if tokenizer.chat_template else None

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
                add_generation_prompt=True,
                enable_thinking=False,
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
            step_scores, avg_prm = prm.score_steps(item['input_str'], reasoning, agg=args.prm_agg)

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
    # 结果保存到 evaluation/ 目录下
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(eval_dir, f'{task_name}_reasoning.json')

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
        compat_path = os.path.join(eval_dir, f'{task_name}.json')
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
    parser.add_argument('--prm_agg', type=str, default='mean',
                        choices=['min', 'mean', 'last', 'weighted_mean'],
                        help="PRM score aggregation strategy (default: min, same as training)")
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
                '--prm_agg', args.prm_agg,
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
