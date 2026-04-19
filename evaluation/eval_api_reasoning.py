"""
Evaluate API models (GPT-4, Claude, DeepSeek, etc.) with PRM scoring.

Output format is identical to eval_reasoning.py, enabling direct comparison
between local RL-trained models and API baselines.

Supports any OpenAI-compatible API endpoint.

Usage:
    # GPT-4o evaluation (20 samples quick test)
    python3 evaluation/eval_api_reasoning.py \
        --api_base https://api.openai.com/v1 \
        --api_key sk-xxx \
        --api_model gpt-4o \
        --prm_path /tmp/prm_qwen3_4b \
        --eval_file evaluation/data/eval_data.json \
        --limit 20

    # DeepSeek evaluation (full dataset, 8 concurrent)
    python3 evaluation/eval_api_reasoning.py \
        --api_base https://api.deepseek.com/v1 \
        --api_key sk-xxx \
        --api_model deepseek-chat \
        --prm_path /tmp/prm_qwen3_4b \
        --eval_file evaluation/data/eval_data.json \
        --concurrency 8

    # Local vLLM / Ollama server
    python3 evaluation/eval_api_reasoning.py \
        --api_base http://127.0.0.1:8000/v1 \
        --api_key EMPTY \
        --api_model default \
        --prm_path /tmp/prm_qwen3_4b \
        --eval_file evaluation/data/eval_data.json

    # Skip PRM scoring (accuracy-only comparison)
    python3 evaluation/eval_api_reasoning.py \
        --api_base https://api.openai.com/v1 \
        --api_key sk-xxx \
        --api_model gpt-4o \
        --eval_file evaluation/data/eval_data.json \
        --no_prm
"""

# ═══════════════════════════════════════════════════════════════════
#  Default system prompt — aligns API model output with PRM format
#
#  Training format (SFT_stage1.py / medical_reward.py):
#    ## Thinking
#    
#    {step1}
#    
#    {step2}
#    
#    ...
#    
#    ## Final Response
#    
#    {answer}
#
#  PRM scores each step at "\n\n" boundaries inside the Thinking
#  section, so this system prompt instructs API models to produce
#  the same structure for fair comparison.
# ═══════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = """You are a medical expert. Please structure your response in the following format:

## Thinking

{your step-by-step reasoning here, with each reasoning step separated by a blank line}

## Final Response

{your final answer here}

Important formatting rules:
- Start with "## Thinking" followed by a blank line.
- Write your reasoning as multiple separate steps. Each step should be a short paragraph focusing on one aspect of the analysis.
- Separate each reasoning step with exactly one blank line (i.e., two newlines).
- After all reasoning steps, write "## Final Response" preceded by a blank line.
- In the Final Response section, provide your final answer clearly.
- For multiple-choice questions, state the answer in the format: "The answer is X." where X is the option letter."""

import argparse
import json
import os
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# scorer.py is in the same directory
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer import match_choice, get_results


# ═══════════════════════════════════════════════════════════════════
#  API Client — wraps OpenAI-compatible endpoints
# ═══════════════════════════════════════════════════════════════════

class APIClient:
    """Thin wrapper around OpenAI-compatible chat completions API."""

    def __init__(self, api_base, api_key, model, max_new_tokens=1200,
                 temperature=0.1, top_p=0.9, max_retries=3, retry_delay=2):
        import openai
        self.client = openai.OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(self, prompt, system_prompt=None):
        """Send a single chat request with retry logic."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"  API error (attempt {attempt+1}/{self.max_retries}): {e}")
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  API failed after {self.max_retries} attempts: {e}")
                    return ""


# ═══════════════════════════════════════════════════════════════════
#  PRM Scorer — identical to eval_reasoning.py PRMScorer
# ═══════════════════════════════════════════════════════════════════

class PRMScorer:
    """Load PRM and score reasoning steps (same logic as eval_reasoning.py)."""

    def __init__(self, prm_path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
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
        self.reward_head = self.reward_head.to(torch.bfloat16)
        self.step_sep = '\n\n'

    @staticmethod
    def _aggregate(scores, agg="min"):
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
        return min(scores)

    @torch.no_grad()
    def score_steps(self, question, reasoning_text, agg="min"):
        steps = [s.strip() for s in reasoning_text.split(self.step_sep) if s.strip()]
        if not steps:
            return [], 0.0

        prefix = f"Question: {question}\n\nReasoning:\n\n"
        full_text = prefix + "\n\n".join(steps)

        encoding = self.tokenizer(
            full_text, add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=True, max_length=4096,
        )
        full_tokens = encoding["input_ids"]
        offsets = encoding.get("offset_mapping")

        if offsets is not None:
            step_end_chars = []
            char_pos = len(prefix)
            for i, step in enumerate(steps):
                char_pos += len(step)
                step_end_chars.append(char_pos - 1)
                if i < len(steps) - 1:
                    char_pos += 2

            step_end_positions = []
            for target in step_end_chars:
                pos = len(full_tokens) - 1
                for tok_idx in range(len(offsets)):
                    start, end = offsets[tok_idx]
                    if start <= target < end:
                        pos = tok_idx
                        break
                step_end_positions.append(pos)
        else:
            step_end_positions = []
            for i in range(len(steps)):
                partial = prefix + "\n\n".join(steps[:i + 1])
                partial_tokens = self.tokenizer.encode(partial, add_special_tokens=True)
                step_end_positions.append(len(partial_tokens) - 1)

        input_ids = torch.LongTensor([full_tokens]).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        seq_len = hidden_states.size(1)

        step_scores = []
        for pos in step_end_positions:
            pos = min(pos, seq_len - 1)
            h = hidden_states[0, pos]
            s = torch.sigmoid(self.reward_head(h)).item()
            step_scores.append(s)

        aggregated = self._aggregate(step_scores, agg)
        return step_scores, aggregated


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def extract_reasoning_and_answer(text):
    """Split model output into reasoning part and final answer part.
    
    Handles mixed formats where API models may output both
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

    # Clean up residual <think>/<\/think> tags (API models may output both
    # <think>...</think> and ## Thinking / ## Final Response simultaneously)
    reasoning = reasoning.replace('<think>', '').replace('</think>', '').strip()

    return reasoning, answer


def build_query(item, strict_prompt=False):
    """Build query string for a single evaluation item."""
    item['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in item['options'].items()])
    if strict_prompt:
        query_prompt = ("Please answer the following multiple-choice questions. "
                        "Please answer the following multiple-choice questions, "
                        "ensuring your response concludes with the correct option "
                        "in the format: 'The answer is A.'.\n{question}\n{option_str}")
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"
    item['input_str'] = query_prompt.format_map(item)
    return item


# ═══════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ═══════════════════════════════════════════════════════════════════

def evaluate(args):
    # ── Load data ──
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

    for item in input_data:
        build_query(item, args.strict_prompt)

    print(f"Loaded {len(input_data)} samples from {args.eval_file}")

    # ── Init API client ──
    api = APIClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.api_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"API: {args.api_model} @ {args.api_base}")

    # ── Init PRM (optional) ──
    prm = None
    if not args.no_prm:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading PRM: {args.prm_path} → {device}")
        prm = PRMScorer(args.prm_path, device)
    else:
        print("PRM scoring disabled (--no_prm)")

    # ── System prompt (aligned with PRM training format by default) ──
    if args.no_system_prompt:
        system_prompt = None
        print("System prompt: DISABLED (raw model output)")
    else:
        system_prompt = args.system_prompt
        print(f"System prompt: {'[default PRM-aligned]' if system_prompt == DEFAULT_SYSTEM_PROMPT else '[custom]'}")
        print(f"  (first 100 chars: {system_prompt[:100]}...)")

    # ── Call API concurrently ──
    results = [None] * len(input_data)

    def _process_one(idx):
        item = input_data[idx]
        output = api.chat(item['input_str'], system_prompt=system_prompt)
        return idx, output

    print(f"Running API calls (concurrency={args.concurrency})...")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process_one, i): i for i in range(len(input_data))}
        for future in tqdm(as_completed(futures), total=len(input_data), desc="API calls"):
            idx, output = future.result()
            results[idx] = output

    # ── Score with PRM and collect results ──
    print("Processing results and scoring...")
    final_results = []
    for i, item in enumerate(tqdm(input_data, desc="PRM scoring")):
        full_output = results[i]
        if not full_output:
            continue

        reasoning, answer_text = extract_reasoning_and_answer(full_output)
        ans, ans_type = match_choice(full_output, item['options'])
        is_correct = ans[0].lower() == item['answer_idx'].lower()

        step_scores, avg_prm = [], 0.0
        if prm is not None and reasoning.strip():
            step_scores, avg_prm = prm.score_steps(
                item['input_str'], reasoning, agg=args.prm_agg)

        final_results.append({
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

    # ── Aggregate and report ──
    n = len(final_results)
    if n == 0:
        print("No results collected!")
        return

    correct_count = sum(1 for r in final_results if r['correct'])
    all_prm = [r['avg_prm_score'] for r in final_results]
    correct_prm = [r['avg_prm_score'] for r in final_results if r['correct']]
    wrong_prm = [r['avg_prm_score'] for r in final_results if not r['correct']]
    all_steps = [r['num_steps'] for r in final_results]

    accuracy = correct_count / n
    avg_prm_all = sum(all_prm) / n if n else 0
    avg_correct_prm = sum(correct_prm) / len(correct_prm) if correct_prm else 0
    avg_wrong_prm = sum(wrong_prm) / len(wrong_prm) if wrong_prm else 0
    avg_steps = sum(all_steps) / n if n else 0

    by_source = {}
    for r in final_results:
        src = r['source']
        if src not in by_source:
            by_source[src] = {'correct': 0, 'total': 0, 'prm_sum': 0.0}
        by_source[src]['total'] += 1
        by_source[src]['prm_sum'] += r['avg_prm_score']
        if r['correct']:
            by_source[src]['correct'] += 1

    print("\n" + "=" * 60)
    print(f"API Model Reasoning Evaluation ({n} samples)")
    print(f"Model: {args.api_model}")
    print("=" * 60)
    print(f"  Answer Accuracy:           {accuracy:.4f} ({correct_count}/{n})")
    print(f"  Avg Reasoning Steps:       {avg_steps:.1f}")
    if not args.no_prm:
        print(f"  Avg PRM Score (all):       {avg_prm_all:.4f}")
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

    if not args.no_prm:
        if avg_correct_prm > avg_wrong_prm:
            print("  ✓ PRM is discriminative: correct answers have higher process scores")
        else:
            print("  ⚠ PRM may not align well with this model's reasoning style")

    # ── Save results ──
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    safe_model_name = args.api_model.replace("/", "_").replace(":", "_")
    task_name = (safe_model_name
                 + os.path.basename(args.eval_file).replace('.json', '')
                 + f'_{args.task}'
                 + ('_strict-prompt' if args.strict_prompt else ''))

    save_path = os.path.join(eval_dir, f'{task_name}_reasoning.json')
    with open(save_path, 'w') as fw:
        json.dump({
            'model': args.api_model,
            'api_base': args.api_base,
            'prm_agg': args.prm_agg,
            'metrics': {
                'accuracy': accuracy,
                'avg_steps': avg_steps,
                'avg_prm_score': avg_prm_all,
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
            'details': final_results
        }, fw, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {save_path}")

    # scorer-compatible output
    compat_path = os.path.join(eval_dir, f'{task_name}.json')
    with open(compat_path, 'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)
    print(f"scorer-compatible results saved to {compat_path}")
    get_results(compat_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate API models with PRM scoring for comparison")

    # API settings
    parser.add_argument('--api_base', type=str, required=True,
                        help="OpenAI-compatible API base URL (e.g. https://api.openai.com/v1)")
    parser.add_argument('--api_key', type=str, default=None,
                        help="API key (can also set via OPENAI_API_KEY env var)")
    parser.add_argument('--api_model', type=str, required=True,
                        help="Model name (e.g. gpt-4o, deepseek-chat, claude-3-opus)")
    parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt for the API model. Default instructs the model "
                             "to output in ## Thinking / ## Final Response format for PRM "
                             "compatibility. Use --no_system_prompt to disable.")
    parser.add_argument('--no_system_prompt', action='store_true',
                        help="Disable system prompt entirely (raw model output)")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--concurrency', type=int, default=4,
                        help="Number of concurrent API requests")

    # PRM settings
    parser.add_argument('--prm_path', type=str, default=None,
                        help="Path to trained PRM model (required unless --no_prm)")
    parser.add_argument('--prm_agg', type=str, default='min',
                        choices=['min', 'mean', 'last', 'weighted_mean'],
                        help="PRM score aggregation strategy (default: min)")
    parser.add_argument('--no_prm', action='store_true',
                        help="Skip PRM scoring (accuracy-only evaluation)")

    # Eval settings
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=1200)
    parser.add_argument('--limit', type=int, default=-1,
                        help="Limit number of samples (-1 for all)")
    parser.add_argument('--strict_prompt', action='store_true')
    parser.add_argument('--task', type=str, default='api')

    args = parser.parse_args()

    # API key fallback to env var
    if args.api_key is None:
        args.api_key = os.environ.get('OPENAI_API_KEY', '')
        if not args.api_key:
            parser.error("--api_key is required (or set OPENAI_API_KEY env var)")

    # PRM path required unless --no_prm
    if not args.no_prm and not args.prm_path:
        parser.error("--prm_path is required (or use --no_prm to skip PRM scoring)")

    evaluate(args)


if __name__ == "__main__":
    main()
