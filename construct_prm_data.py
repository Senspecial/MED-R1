"""
Construct PRM (Process Reward Model) training data via LLM-as-Judge step annotation.

Supports three execution modes:
  A) Two-step split (recommended for GPU containers with no internet):
     Step 1 (GPU container):  --generate_only  -> saves chains to save_dir/
     Step 2 (any machine):    --judge_only     -> reads chains, calls DeepSeek judge
  B) Full pipeline (when both GPU and internet are available)
  C) All-API mode (DeepSeek for both generation and judging, no GPU needed)

Usage:
    # === Two-step mode (GPU container has no internet) ===

    # Step 1: Generate chains in GPU container (8-GPU parallel)
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python3 construct_prm_data.py \\
        --generate_only --local --model_path /tmp/sft_model \\
        --data_path data/medical_o1_verifiable_problem.json \\
        --num_chains 10 --start_idx $((i*125)) --limit_num 125 \\
        > prm_gen_gpu${i}.log 2>&1 &
    done

    # Step 2: Judge chains outside container (has internet)
    python3 construct_prm_data.py \\
        --judge_only \\
        --data_path data/medical_o1_verifiable_problem.json \\
        --save_dir output_data/prm_construction \\
        --judge_model deepseek-chat \\
        --judge_api_url https://api.deepseek.com/v1/chat/completions \\
        --judge_api_key  \\
        --output_path data/prm_train_data.json

    # === All-API mode (no GPU needed) ===
    python3 construct_prm_data.py \\
        --model_name deepseek-chat \\
        --api_url https://api.deepseek.com/v1/chat/completions \\
        --api_key  \\
        --data_path data/medical_o1_verifiable_problem.json \\
        --num_chains 10 --num_workers 8 \\
        --output_path data/prm_train_data.json
"""
import os
import re
import json
import time
import random
import logging
import argparse
import traceback
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible API)
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, model_name, api_url, api_key, max_concurrent=32, timeout=120):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_concurrent,
            pool_maxsize=max_concurrent,
            max_retries=3,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        logger.info("LLM client: model=%s, api=%s", model_name, api_url)

    def generate(self, prompt, temperature=0.7, max_tokens=4096, n=1):
        for attempt in range(3):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": n,
                }
                resp = self.session.post(
                    self.api_url, headers=headers, json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
                if "error" in data:
                    raise ValueError(f"API Error: {data}")
                if n == 1:
                    return data["choices"][0]["message"]["content"]
                return [c["message"]["content"] for c in data["choices"]]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)


# ---------------------------------------------------------------------------
# Local LLM client (transformers, no API server needed)
# ---------------------------------------------------------------------------
class LocalLLMClient:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading tokenizer from %s ...", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Loading model from %s ...", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype="bfloat16", device_map="auto",
        )
        self.model.eval()
        self._lock = threading.Lock()
        logger.info("Local model loaded on %s", self.device)

    def generate(self, prompt, temperature=0.7, max_tokens=4096, n=1):
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        with self._lock, torch.no_grad():
            input_ids = encoded["input_ids"].to(self.model.device)
            attn_mask = encoded["attention_mask"].to(self.model.device)
            if n > 1:
                input_ids = input_ids.expand(n, -1)
                attn_mask = attn_mask.expand(n, -1)
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)

        prompt_len = input_ids.shape[-1]
        results = [self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in outputs]
        return results[0] if n == 1 else results


# ---------------------------------------------------------------------------
# Step-level judge (calls DeepSeek to score each reasoning step)
# ---------------------------------------------------------------------------
class StepJudge:
    """Use a strong LLM to judge whether each reasoning step is correct."""

    JUDGE_PROMPT = (
        "You are a medical reasoning evaluator. Your task is to judge whether "
        "a single reasoning step is logically correct.\n\n"
        "## Question\n{question}\n\n"
        "## Reference Answer\n{answer}\n\n"
        "## Previous Reasoning Steps\n{previous_steps}\n\n"
        "## Current Step to Evaluate\n{current_step}\n\n"
        "Is this reasoning step logically sound, medically accurate, and "
        "consistent with the previous steps? Does it contribute toward "
        "reaching the correct answer?\n\n"
        'Reply with ONLY "1" (correct) or "0" (incorrect).'
    )

    def __init__(self, model_name, api_url, api_key, max_concurrent=32, timeout=60):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_concurrent,
            pool_maxsize=max_concurrent,
            max_retries=3,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)
        logger.info("Step judge: model=%s, api=%s", model_name, api_url)

    def _call_api(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 8,
                }
                resp = self.session.post(
                    self.api_url, headers=headers, json=payload, timeout=self.timeout,
                )
                data = resp.json()
                if "error" in data:
                    raise ValueError(f"Judge API Error: {data}")
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    def judge_one(self, question: str, answer: str,
                  previous_steps: list[str], current_step: str) -> float:
        """Judge a single step. Returns 1.0 (correct) or 0.0 (incorrect)."""
        if previous_steps:
            prev_text = "\n\n".join(
                f"Step {i+1}: {s}" for i, s in enumerate(previous_steps)
            )
        else:
            prev_text = "(This is the first step)"

        prompt = self.JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            previous_steps=prev_text,
            current_step=current_step,
        )
        try:
            reply = self._call_api(prompt)
            return 1.0 if "1" in reply else 0.0
        except Exception:
            traceback.print_exc()
            return 0.0

    def judge_chain(self, question: str, answer: str,
                    steps: list[str]) -> list[float]:
        """Judge all steps in a chain concurrently. Returns list of scores."""
        futures = []
        for i, step in enumerate(steps):
            fut = self._pool.submit(
                self.judge_one, question, answer, steps[:i], step,
            )
            futures.append(fut)
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Parse chain output (compatible with SFT model format)
# ---------------------------------------------------------------------------
THINKING_PATTERN = re.compile(r"## Thinking\n\n(.*?)(?=\n\n## Final Response)", re.S)
RESPONSE_PATTERN = re.compile(r"## Final Response\n\n(.*)", re.S)


def parse_chain(chain_text: str) -> tuple[list[str], str | None]:
    """Parse a chain into (thinking_steps, final_response).

    Handles both SFT model format (## Thinking / ## Final Response)
    and free-form text (DeepSeek output).
    """
    thinking_match = THINKING_PATTERN.search(chain_text)
    final_resp_match = RESPONSE_PATTERN.search(chain_text)

    if thinking_match:
        thinking_text = thinking_match.group(1)
        final_resp = final_resp_match.group(1).strip() if final_resp_match else None
    else:
        paragraphs = [p.strip() for p in chain_text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 2:
            thinking_text = "\n\n".join(paragraphs[:-1])
            final_resp = paragraphs[-1]
        else:
            thinking_text = chain_text
            final_resp = None

    steps = [s.strip() for s in thinking_text.split("\n\n") if s.strip()]
    return steps, final_resp


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def generate_chains(item, llm, args):
    """Step 1: Generate N chains for a question (GPU, no network needed)."""
    question = item["Open-ended Verifiable Question"]
    answer = item["Ground-True Answer"]

    try:
        chains_raw = llm.generate(
            question, temperature=0.7, max_tokens=args.max_tokens,
            n=args.num_chains,
        )
        if isinstance(chains_raw, str):
            chains_raw = [chains_raw]
    except Exception:
        traceback.print_exc()
        return None

    if not chains_raw:
        return None

    parsed_chains = []
    for chain_text in chains_raw:
        steps, final_resp = parse_chain(chain_text)
        if len(steps) < 2:
            continue
        parsed_chains.append({
            "steps": [{"step_idx": i, "text": s} for i, s in enumerate(steps)],
            "final_response": final_resp,
        })

    if not parsed_chains:
        return None
    return {"question": question, "answer": answer, "chains": parsed_chains}


def judge_chains(result, judge):
    """Step 2: Call DeepSeek to judge every step (network, no GPU needed).

    All steps across all chains are submitted concurrently to maximise API
    throughput instead of processing chains one-by-one sequentially.
    """
    question = result["question"]
    answer = result["answer"]

    futures = []
    for chain in result["chains"]:
        steps_text = [s["text"] for s in chain["steps"]]
        chain_futures = []
        for i, step in enumerate(steps_text):
            fut = judge._pool.submit(
                judge.judge_one, question, answer, steps_text[:i], step,
            )
            chain_futures.append(fut)
        futures.append((chain, chain_futures))

    for chain, chain_futures in futures:
        scores = [f.result() for f in chain_futures]
        for s, score in zip(chain["steps"], scores):
            s["score"] = score
        chain["chain_correct"] = all(sc == 1.0 for sc in scores)
    return result


def process_single(item, llm, judge, args):
    """Full pipeline: generate + judge (only when both are available)."""
    result = generate_chains(item, llm, args)
    if result is None:
        return None
    return judge_chains(result, judge)


def main():
    parser = argparse.ArgumentParser(
        description="Construct PRM data via LLM-as-Judge step annotation",
    )
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/prm_train_data.json")
    parser.add_argument("--save_dir", type=str, default="output_data/prm_construction")
    parser.add_argument("--limit_num", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in the data (for multi-GPU splitting)")
    # Local mode
    parser.add_argument("--generate_only", action="store_true",
                        help="Step 1: generate chains only (GPU, no network)")
    parser.add_argument("--judge_only", action="store_true",
                        help="Step 2: judge saved chains only (network, no GPU)")
    parser.add_argument("--local", action="store_true",
                        help="Load SFT model locally (requires GPU)")
    parser.add_argument("--model_path", type=str, default=None)
    # API mode (generation)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--api_url", type=str,
                        default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--api_key", type=str, default="token-placeholder")
    # Judge (defaults to generation model if not specified)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--judge_api_url", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    # Sampling
    parser.add_argument("--num_chains", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)
    # Parallelism
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Parallel questions (1 for --local)")
    parser.add_argument("--judge_workers", type=int, default=32,
                        help="Parallel step judging threads")
    # Misc
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.generate_only and args.judge_only:
        parser.error("Cannot use --generate_only and --judge_only together")

    # --------------------------------------------------------------------------
    #  JUDGE-ONLY MODE: read saved chains, call DeepSeek, output final data
    # --------------------------------------------------------------------------
    if args.judge_only:
        logger.info("JUDGE-ONLY mode: scoring saved chains via API")
        judge = StepJudge(
            model_name=args.judge_model or "deepseek-chat",
            api_url=args.judge_api_url or "https://api.deepseek.com/v1/chat/completions",
            api_key=args.judge_api_key or args.api_key,
            max_concurrent=args.judge_workers,
            timeout=args.timeout,
        )

        chain_files = sorted(
            [f for f in os.listdir(args.save_dir) if f.endswith(".json")],
            key=lambda x: int(x.replace(".json", "")),
        )
        logger.info("Found %d chain files in %s", len(chain_files), args.save_dir)

        t0 = time.time()
        n_success = 0
        n_fail = 0
        judged_dir = args.save_dir + "_judged"
        os.makedirs(judged_dir, exist_ok=True)

        already_judged = set(f for f in os.listdir(judged_dir) if f.endswith(".json"))

        def _judge_worker(fname):
            if fname in already_judged:
                with open(os.path.join(judged_dir, fname)) as f:
                    return fname, json.load(f)
            with open(os.path.join(args.save_dir, fname)) as f:
                result = json.load(f)
            try:
                result = judge_chains(result, judge)
                with open(os.path.join(judged_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                return fname, result
            except Exception:
                traceback.print_exc()
                return fname, None

        results = {}
        with ThreadPoolExecutor(max_workers=args.num_workers or 8) as pool:
            futures = {pool.submit(_judge_worker, f): f for f in chain_files}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Judging steps")
            for fut in pbar:
                fname, result = fut.result()
                idx = int(fname.replace(".json", ""))
                if result:
                    results[idx] = result
                    n_success += 1
                else:
                    n_fail += 1
                pbar.set_postfix(ok=n_success, fail=n_fail)

        elapsed_total = time.time() - t0
        ordered_results = [results[i] for i in sorted(results.keys())]
        _write_final_output(ordered_results, args, len(chain_files),
                            n_success, n_fail, elapsed_total)
        return

    # --------------------------------------------------------------------------
    #  GENERATE mode (--generate_only) or FULL mode
    # --------------------------------------------------------------------------
    if args.local:
        if not args.model_path:
            parser.error("--model_path required for --local")
        args.num_workers = 1
        logger.info("LOCAL mode: transformers inference")
    else:
        if not args.model_name:
            parser.error("--model_name required for API mode")

    with open(args.data_path) as f:
        data = json.load(f)
    if args.start_idx > 0:
        data = data[args.start_idx:]
    if args.limit_num:
        data = data[: args.limit_num]

    required_keys = {"Open-ended Verifiable Question", "Ground-True Answer"}
    valid_data = [d for d in data if required_keys.issubset(d.keys())]
    if len(valid_data) < len(data):
        logger.warning("Dropped %d items missing required keys", len(data) - len(valid_data))
        data = valid_data
    logger.info("Loaded %d questions from %s", len(data), args.data_path)

    # Init LLM client
    if args.local:
        llm = LocalLLMClient(args.model_path)
    else:
        llm = LLMClient(
            args.model_name, args.api_url, args.api_key,
            max_concurrent=args.num_workers, timeout=args.timeout,
        )

    # Init judge (only for full mode)
    judge = None
    if not args.generate_only:
        judge = StepJudge(
            model_name=args.judge_model or args.model_name or "deepseek-chat",
            api_url=args.judge_api_url or args.api_url,
            api_key=args.judge_api_key or args.api_key,
            max_concurrent=args.judge_workers,
            timeout=args.timeout,
        )

    # Resume
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    processed_ids = set()
    for fname in os.listdir(args.save_dir):
        if fname.endswith(".json"):
            processed_ids.add(fname.replace(".json", ""))

    results: dict[int, dict] = {}
    pending = []
    for idx, item in enumerate(data, start=args.start_idx):
        str_idx = str(idx)
        save_path = os.path.join(args.save_dir, f"{str_idx}.json")
        if str_idx in processed_ids:
            try:
                with open(save_path) as f:
                    results[idx] = json.load(f)
            except Exception:
                pending.append((idx, item))
        else:
            pending.append((idx, item))

    logger.info("Resumed %d, remaining %d", len(results), len(pending))

    t0 = time.time()
    n_success = 0
    n_fail = 0

    def _worker(idx_item):
        idx, item = idx_item
        save_path = os.path.join(args.save_dir, f"{idx}.json")
        try:
            if args.generate_only:
                result = generate_chains(item, llm, args)
            else:
                result = process_single(item, llm, judge, args)
            if result and result["chains"]:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                return idx, result
        except Exception:
            traceback.print_exc()
        return idx, None

    if pending:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(_worker, item): item for item in pending}
            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="Generating chains" if args.generate_only
                        else "Building PRM data")
            for fut in pbar:
                idx, result = fut.result()
                if result:
                    results[idx] = result
                    n_success += 1
                else:
                    n_fail += 1
                elapsed = time.time() - t0
                rate = (n_success + n_fail) / elapsed if elapsed > 0 else 0
                pbar.set_postfix(ok=n_success, fail=n_fail, qps=f"{rate:.1f}")

    elapsed_total = time.time() - t0

    if args.generate_only:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  Chain Generation Complete (--generate_only)")
        logger.info("=" * 60)
        logger.info("Questions : %d (ok=%d, fail=%d)", len(data), n_success, n_fail)
        logger.info("Saved to  : %s/", args.save_dir)
        logger.info("Time      : %.0fs (%.1f min)", elapsed_total, elapsed_total / 60)
        logger.info("Next step : run with --judge_only --save_dir %s", args.save_dir)
        logger.info("=" * 60)
        return

    # Full mode: write final output
    ordered_results = [results[i] for i in sorted(results.keys())]
    _write_final_output(ordered_results, args, len(data),
                        n_success, n_fail, elapsed_total)


def _write_final_output(ordered_results, args, total_q, n_success, n_fail,
                        elapsed_total):
    """Flatten results to train_prm.py format and write output."""
    train_data = []
    total_chains = 0
    total_correct_chains = 0
    total_steps = 0
    total_correct_steps = 0

    for r in ordered_results:
        for chain in r["chains"]:
            total_chains += 1
            if chain.get("chain_correct"):
                total_correct_chains += 1
            for step in chain["steps"]:
                total_steps += 1
                if step.get("score", 0) == 1.0:
                    total_correct_steps += 1
                train_data.append({
                    "question": r["question"],
                    "answer": r["answer"],
                    "step_idx": step["step_idx"],
                    "step_text": step["text"],
                    "prefix_steps": [
                        s["text"] for s in chain["steps"][: step["step_idx"] + 1]
                    ],
                    "score": step.get("score", 0),
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PRM Data Construction Complete")
    logger.info("=" * 60)
    logger.info("Questions : %d total, %d processed (ok=%d, fail=%d)",
                total_q, n_success + n_fail, n_success, n_fail)
    logger.info("Chains    : %d total, %d all-correct (%.1f%%)",
                total_chains, total_correct_chains,
                100 * total_correct_chains / max(total_chains, 1))
    logger.info("Steps     : %d total, %d correct (%.1f%%)",
                total_steps, total_correct_steps,
                100 * total_correct_steps / max(total_steps, 1))
    logger.info("Train samples: %d", len(train_data))
    logger.info("Time      : %.0fs (%.1f min)", elapsed_total, elapsed_total / 60)
    logger.info("Output    : %s", args.output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
