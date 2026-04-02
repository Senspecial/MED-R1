"""
Construct PRM (Process Reward Model) training data via Monte Carlo sampling.

Pipeline:
1. Load verifiable problems (question + ground-truth answer)
2. Use SFT model (via vLLM / OpenAI-compatible API) to generate reasoning chains
3. Split each chain into steps by \n\n
4. For each step prefix, sample N completions and check correctness with ORM
5. step_score = correct_completions / N

Usage:
    python construct_prm_data.py \
        --data_path data/medical_o1_verifiable_problem.json \
        --model_name <your-sft-model> \
        --api_url http://localhost:8000/v1/chat/completions \
        --api_key token-placeholder \
        --verifier_model_path FreedomIntelligence/medical_o1_verifier_3B \
        --num_chains 4 \
        --num_completions 16 \
        --num_workers 8 \
        --output_path data/prm_train_data.json
"""

import os
import re
import json
import random
import argparse
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from retrying import retry

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible API, works with vLLM / sglang / etc.)
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"LLM client using model: {self.model_name}")

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def generate(self, prompt, temperature=0.7, max_tokens=4096):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(self.api_url, headers=headers, json=payload)
        data = resp.json()
        if "error" in data:
            raise ValueError(f"API Error: {data}")
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# ORM verifier (reuses the existing medical_o1_verifier logic)
# ---------------------------------------------------------------------------
class ORMVerifier:
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.device = device

    VERIFY_TEMPLATE = (
        "<Model Response>\n{response}\n</Model Response>\n\n"
        "<Reference Answer>\n{answer}\n</Reference Answer>\n\n"
        "Your task is to evaluate the model response by comparing it to the "
        'reference answer. If the model response is correct and aligns with '
        'the reference answer, output "True". If it is incorrect or fails to '
        'select the correct option (if options are provided), output "False". '
        "{eos}"
    )

    @torch.no_grad()
    def verify_batch(self, responses: list[str], answers: list[str]) -> list[bool]:
        texts = []
        for resp, ans in zip(responses, answers):
            texts.append(self.VERIFY_TEMPLATE.format(
                response=resp, answer=ans, eos=self.tokenizer.eos_token
            ))
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=4000, add_special_tokens=False,
        ).to(self.device)
        logits = self.model(**inputs, return_dict=True).logits
        probs = F.softmax(logits, dim=-1)
        return [p[1].item() > 0.4 for p in probs]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
THINKING_PATTERN = re.compile(r"## Thinking\n\n(.*?)(?=\n\n## Final Response)", re.S)
RESPONSE_PATTERN = re.compile(r"## Final Response\n\n(.*)", re.S)


def split_thinking_steps(thinking_text: str) -> list[str]:
    return [s.strip() for s in thinking_text.split("\n\n") if s.strip()]


def extract_final_response(full_output: str) -> str | None:
    m = RESPONSE_PATTERN.search(full_output)
    return m.group(1).strip() if m else None


def build_partial_prompt(question: str, steps: list[str], up_to: int) -> str:
    """Build a prompt that contains the question + first `up_to` steps,
    asking the model to continue reasoning."""
    partial_thinking = "\n\n".join(steps[:up_to])
    return (
        f"{question}\n\n"
        f"Please continue your reasoning from where you left off. "
        f"Your previous thinking:\n\n## Thinking\n\n{partial_thinking}\n\n"
        f"Continue your thinking and provide ## Final Response at the end."
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_single(item, llm, verifier, args):
    """Process one question: generate chains, compute per-step MC scores."""
    question = item["Open-ended Verifiable Question"]
    answer = item["Ground-True Answer"]

    result = {
        "question": question,
        "answer": answer,
        "chains": [],
    }

    # Step 1: generate multiple reasoning chains
    prompt = question
    chains_raw = []
    for _ in range(args.num_chains):
        try:
            output = llm.generate(prompt, temperature=0.7, max_tokens=args.max_tokens)
            chains_raw.append(output)
        except Exception:
            traceback.print_exc()

    if not chains_raw:
        return None

    # Step 2: for each chain, split steps and compute MC scores
    for chain_text in chains_raw:
        thinking_match = THINKING_PATTERN.search(chain_text)
        final_resp = extract_final_response(chain_text)

        if thinking_match is None:
            thinking_text = chain_text
        else:
            thinking_text = thinking_match.group(1)

        steps = split_thinking_steps(thinking_text)
        if len(steps) < 2:
            continue

        # Verify the chain's own final answer
        chain_correct = False
        if final_resp:
            chain_correct = verifier.verify_batch([final_resp], [answer])[0]

        # Step 3: MC sampling for each step prefix
        step_scores = []
        for step_idx in range(len(steps)):
            partial_prompt = build_partial_prompt(question, steps, step_idx + 1)

            completions_correct = 0
            completions_total = 0
            completion_batch_resps = []
            for _ in range(args.num_completions):
                try:
                    completion = llm.generate(
                        partial_prompt, temperature=0.7, max_tokens=args.max_tokens
                    )
                    comp_resp = extract_final_response(completion)
                    if comp_resp is None:
                        comp_resp = completion[-500:]
                    completion_batch_resps.append(comp_resp)
                    completions_total += 1
                except Exception:
                    traceback.print_exc()

            # Batch verify all completions for this step
            if completion_batch_resps:
                verdicts = verifier.verify_batch(
                    completion_batch_resps, [answer] * len(completion_batch_resps)
                )
                completions_correct = sum(verdicts)

            score = completions_correct / max(completions_total, 1)
            step_scores.append({
                "step_idx": step_idx,
                "text": steps[step_idx],
                "score": round(score, 4),
                "num_correct": completions_correct,
                "num_total": completions_total,
            })

        result["chains"].append({
            "steps": step_scores,
            "final_response": final_resp,
            "chain_correct": chain_correct,
        })

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--api_key", type=str, default="token-placeholder")
    parser.add_argument("--verifier_model_path", type=str, default="FreedomIntelligence/medical_o1_verifier_3B")
    parser.add_argument("--verifier_device", type=str, default="cuda:0")
    parser.add_argument("--num_chains", type=int, default=4, help="Chains per question")
    parser.add_argument("--num_completions", type=int, default=16, help="MC samples per step")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit_num", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="data/prm_train_data.json")
    parser.add_argument("--save_dir", type=str, default="output_data/prm_construction")
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)
    if args.limit_num:
        data = data[: args.limit_num]
    print(f"Loaded {len(data)} questions")

    llm = LLMClient(args.model_name, args.api_url, args.api_key)
    verifier = ORMVerifier(args.verifier_model_path, device=args.verifier_device)

    os.makedirs(args.save_dir, exist_ok=True)

    # Check previously processed
    processed_ids = set()
    for fname in os.listdir(args.save_dir):
        if fname.endswith(".json"):
            processed_ids.add(fname.replace(".json", ""))

    results = []
    for idx, item in enumerate(tqdm(data, desc="Building PRM data")):
        str_idx = str(idx)
        save_path = os.path.join(args.save_dir, f"{str_idx}.json")
        if str_idx in processed_ids:
            with open(save_path) as f:
                results.append(json.load(f))
            continue

        try:
            result = process_single(item, llm, verifier, args)
            if result and result["chains"]:
                results.append(result)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            traceback.print_exc()

    # Flatten to per-step training format
    train_data = []
    for r in results:
        for chain in r["chains"]:
            for step in chain["steps"]:
                train_data.append({
                    "question": r["question"],
                    "answer": r["answer"],
                    "step_idx": step["step_idx"],
                    "step_text": step["text"],
                    "prefix_steps": [s["text"] for s in chain["steps"][: step["step_idx"] + 1]],
                    "score": step["score"],
                })

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(train_data)} step-level samples to {args.output_path}")
    pos = sum(1 for d in train_data if d["score"] > 0.5)
    print(f"Positive (score>0.5): {pos}, Negative: {len(train_data) - pos}")


if __name__ == "__main__":
    main()
