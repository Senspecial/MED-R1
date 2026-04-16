"""
Combined PRM + Judge HTTP server on a single GPU.

Loads both models into one process to share CUDA context and memory,
freeing an extra GPU for verl training.

Memory footprint (bf16):
    PRM   (Qwen3-4B)  ~  8 GB
    Judge (SFT 7B)    ~ 14 GB
    Total             ~ 22 GB  (fits easily on H20 96GB or A100-80G)

API — PRM:
    POST /score
        Body: {"question": str, "thinking_text": str, "agg": "min"|...}
        Response: {"step_scores": [...], "aggregated": float}

    POST /score_batch
        Body: {"samples": [...]}
        Response: {"results": [...]}

API — Judge:
    POST /judge
        Body: {"question": str, "model_answer": str, "ground_truth": str}
        Response: {"score": float, "raw_response": str}

    POST /judge_batch
        Body: {"samples": [...]}
        Response: {"results": [...]}

    GET /health

Usage:
    CUDA_VISIBLE_DEVICES=7 python verl_reward/reward_server.py \
        --prm_path /tmp/prm_qwen3_4b \
        --judge_path /tmp/sft_model \
        --port 8100
"""

import argparse
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  PRM model
# ═══════════════════════════════════════════════════════════════════════════

class PRMScorer(nn.Module):
    def __init__(self, backbone, reward_head):
        super().__init__()
        self.backbone = backbone
        self.reward_head = reward_head

    @torch.no_grad()
    def score_steps(self, input_ids, attention_mask, step_end_positions):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        seq_len = hidden.size(1)
        scores = []
        for pos in step_end_positions:
            pos = min(pos, seq_len - 1)
            h = hidden[0, pos]
            s = torch.sigmoid(self.reward_head(h)).item()
            scores.append(s)
        return scores


# ═══════════════════════════════════════════════════════════════════════════
#  Schemas
# ═══════════════════════════════════════════════════════════════════════════

# -- PRM --
class ScoreRequest(BaseModel):
    question: str
    thinking_text: str
    agg: str = Field(default="min")

class ScoreResponse(BaseModel):
    step_scores: List[float]
    aggregated: float

class BatchScoreRequest(BaseModel):
    samples: List[ScoreRequest]

class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]

# -- Judge --
class JudgeRequest(BaseModel):
    question: str
    model_answer: str
    ground_truth: str

class JudgeResponse(BaseModel):
    score: float
    raw_response: str

class BatchJudgeRequest(BaseModel):
    samples: List[JudgeRequest]

class BatchJudgeResponse(BaseModel):
    results: List[JudgeResponse]


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="MED-R1 Reward Server (PRM + Judge)")

_prm_model: Optional[PRMScorer] = None
_prm_tokenizer = None
_judge_model = None
_judge_tokenizer = None
_device: str = "cuda"
_prm_path: str = ""
_judge_path: str = ""

_judge_yes_token_id: int = -1
_judge_no_token_id: int = -1
_judge_logit_verified = {"done": False, "fallback": False}

_JUDGE_TEMPLATE = (
    "Judge whether the student's answer is correct by comparing it with the "
    "reference answer. Consider medical synonyms (e.g. 'heart attack' = "
    "'myocardial infarction') and abbreviations (e.g. 'MI' = 'myocardial "
    "infarction'). The student may include extra explanation; focus on whether "
    "the core medical conclusion is equivalent.\n\n"
    "Question: {question}\n"
    "Reference answer: {ground_truth}\n"
    "Student's answer: {model_answer}\n\n"
    "Respond with ONLY one word: Yes or No."
)


# ── PRM helpers ───────────────────────────────────────────────────────────

def _aggregate(scores: List[float], agg: str) -> float:
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


def _score_single(question: str, thinking_text: str, agg: str) -> ScoreResponse:
    steps = [s.strip() for s in thinking_text.split("\n\n") if s.strip()]
    if not steps:
        return ScoreResponse(step_scores=[], aggregated=0.0)

    prefix = f"Question: {question}\n\nReasoning:\n\n"
    full_text = prefix + "\n\n".join(steps)

    encoding = _prm_tokenizer(
        full_text, add_special_tokens=True, return_offsets_mapping=True,
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
            partial = prefix + "\n\n".join(steps[: i + 1])
            partial_tokens = _prm_tokenizer.encode(partial, add_special_tokens=True)
            step_end_positions.append(len(partial_tokens) - 1)

    input_ids = torch.LongTensor([full_tokens]).to(_device)
    attention_mask = torch.ones_like(input_ids)
    step_scores = _prm_model.score_steps(input_ids, attention_mask, step_end_positions)
    aggregated = _aggregate(step_scores, agg)
    return ScoreResponse(step_scores=step_scores, aggregated=aggregated)


# ── Judge helpers ─────────────────────────────────────────────────────────

def _build_judge_prompt(question: str, model_answer: str, ground_truth: str) -> str:
    content = _JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer[:500],
    )
    messages = [{"role": "user", "content": content}]
    try:
        return _judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return _judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def _judge_single_generate(question: str, model_answer: str, ground_truth: str) -> JudgeResponse:
    """Old generate-based judge, kept for verification only."""
    if not model_answer or not ground_truth:
        return JudgeResponse(score=0.0, raw_response="")

    prompt = _build_judge_prompt(question, model_answer, ground_truth)
    inputs = _judge_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(_device)

    outputs = _judge_model.generate(
        input_ids, max_new_tokens=3, do_sample=False,
        pad_token_id=_judge_tokenizer.pad_token_id or _judge_tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_ids.shape[1]:]
    raw = _judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    response_lower = raw.lower()

    if response_lower.startswith("yes"):
        score = 1.0
    elif response_lower.startswith("no"):
        score = 0.0
    else:
        score = -1.0

    return JudgeResponse(score=score, raw_response=raw)


@torch.no_grad()
def _judge_single(question: str, model_answer: str, ground_truth: str) -> JudgeResponse:
    """Logit-based judge: single forward pass, compare Yes vs No logits."""
    if not model_answer or not ground_truth:
        return JudgeResponse(score=0.0, raw_response="")

    if _judge_logit_verified.get("fallback"):
        return _judge_single_generate(question, model_answer, ground_truth)

    prompt = _build_judge_prompt(question, model_answer, ground_truth)
    inputs = _judge_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(_device)

    outputs = _judge_model(input_ids=input_ids)
    logits = outputs.logits[0, -1]

    yes_logit = logits[_judge_yes_token_id].item()
    no_logit = logits[_judge_no_token_id].item()

    if yes_logit >= no_logit:
        return JudgeResponse(score=1.0, raw_response="Yes")
    else:
        return JudgeResponse(score=0.0, raw_response="No")


# ── Batch PRM ─────────────────────────────────────────────────────────────

_PRM_BATCH_SIZE = 8
_prm_batch_verified = {"done": False, "fallback": False}


def _score_batch_impl(samples: List[ScoreRequest]) -> List[ScoreResponse]:
    # If verification failed, fall back to sequential (still one HTTP call)
    if _prm_batch_verified.get("fallback"):
        return [_score_single(s.question, s.thinking_text, s.agg) for s in samples]

    """True batch PRM: pad + single forward pass per mini-batch."""
    if not samples:
        return []

    per_sample_ids: List[torch.Tensor] = []
    per_sample_positions: List[List[int]] = []
    per_sample_agg: List[str] = []
    empty_set: set = set()

    for idx, sample in enumerate(samples):
        steps = [s.strip() for s in sample.thinking_text.split("\n\n") if s.strip()]
        if not steps:
            empty_set.add(idx)
            per_sample_ids.append(torch.zeros(0, dtype=torch.long))
            per_sample_positions.append([])
            per_sample_agg.append(sample.agg)
            continue

        prefix = f"Question: {sample.question}\n\nReasoning:\n\n"
        full_text = prefix + "\n\n".join(steps)
        encoding = _prm_tokenizer(
            full_text, add_special_tokens=True, return_offsets_mapping=True,
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
                partial = prefix + "\n\n".join(steps[: i + 1])
                partial_tokens = _prm_tokenizer.encode(partial, add_special_tokens=True)
                step_end_positions.append(len(partial_tokens) - 1)

        per_sample_ids.append(torch.LongTensor(full_tokens))
        per_sample_positions.append(step_end_positions)
        per_sample_agg.append(sample.agg)

    results: List[Optional[ScoreResponse]] = [None] * len(samples)
    for idx in empty_set:
        results[idx] = ScoreResponse(step_scores=[], aggregated=0.0)

    non_empty = [(i, per_sample_ids[i]) for i in range(len(samples)) if i not in empty_set]
    if not non_empty:
        return results

    # Sort by token length to minimise padding waste within each mini-batch
    non_empty.sort(key=lambda x: x[1].shape[0])

    pad_id = _prm_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = _prm_tokenizer.eos_token_id or 0

    for batch_start in range(0, len(non_empty), _PRM_BATCH_SIZE):
        batch_items = non_empty[batch_start : batch_start + _PRM_BATCH_SIZE]
        batch_indices = [item[0] for item in batch_items]
        batch_ids = [item[1] for item in batch_items]
        max_len = max(ids.shape[0] for ids in batch_ids)

        padded = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=_device)
        attn = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=_device)
        pad_amounts = []
        for j, ids in enumerate(batch_ids):
            pad = max_len - ids.shape[0]
            padded[j, pad:] = ids.to(_device)
            attn[j, pad:] = 1
            pad_amounts.append(pad)

        with torch.no_grad():
            outputs = _prm_model.backbone(
                input_ids=padded, attention_mask=attn, output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]

        for j, orig_idx in enumerate(batch_indices):
            pad = pad_amounts[j]
            step_scores = []
            for pos in per_sample_positions[orig_idx]:
                adj = min(pos + pad, hidden.size(1) - 1)
                h = hidden[j, adj]
                s = torch.sigmoid(_prm_model.reward_head(h)).item()
                step_scores.append(s)
            agg = _aggregate(step_scores, per_sample_agg[orig_idx])
            results[orig_idx] = ScoreResponse(step_scores=step_scores, aggregated=agg)

    # ── Verification: compare first non-empty sample batch vs single ──
    if not _prm_batch_verified["done"] and non_empty:
        verify_idx = non_empty[0][0]
        single_result = _score_single(
            samples[verify_idx].question,
            samples[verify_idx].thinking_text,
            samples[verify_idx].agg,
        )
        batch_agg = results[verify_idx].aggregated
        single_agg = single_result.aggregated
        diff = abs(batch_agg - single_agg)
        _prm_batch_verified["done"] = True
        if diff > 0.02:
            print(f"[PRM] WARNING: batch vs single mismatch! "
                  f"batch={batch_agg:.4f} single={single_agg:.4f} diff={diff:.4f}")
            print(f"[PRM] Falling back to sequential scoring for safety.")
            _prm_batch_verified["fallback"] = True
            for item_idx, _ in non_empty:
                results[item_idx] = _score_single(
                    samples[item_idx].question,
                    samples[item_idx].thinking_text,
                    samples[item_idx].agg,
                )
        else:
            print(f"[PRM] Batch verification passed: "
                  f"batch={batch_agg:.4f} single={single_agg:.4f} diff={diff:.4f}")

    return results


# ── Batch Judge ───────────────────────────────────────────────────────────

_JUDGE_BATCH_SIZE = 32


@torch.no_grad()
def _judge_batch_impl(samples: List[JudgeRequest]) -> List[JudgeResponse]:
    """Batch Judge via logit classification: forward pass + compare Yes/No logits."""
    if not samples:
        return []

    if _judge_logit_verified.get("fallback"):
        return [_judge_single_generate(s.question, s.model_answer, s.ground_truth)
                for s in samples]

    prompts: List[str] = []
    skip_set: set = set()
    for idx, s in enumerate(samples):
        if not s.model_answer or not s.ground_truth:
            skip_set.add(idx)
            prompts.append("")
            continue
        prompts.append(_build_judge_prompt(s.question, s.model_answer, s.ground_truth))

    results: List[Optional[JudgeResponse]] = [None] * len(samples)
    for idx in skip_set:
        results[idx] = JudgeResponse(score=0.0, raw_response="")

    non_empty = [(i, prompts[i]) for i in range(len(samples)) if i not in skip_set]
    if not non_empty:
        return results

    orig_padding_side = _judge_tokenizer.padding_side
    _judge_tokenizer.padding_side = "left"
    if _judge_tokenizer.pad_token_id is None:
        _judge_tokenizer.pad_token = _judge_tokenizer.eos_token

    for batch_start in range(0, len(non_empty), _JUDGE_BATCH_SIZE):
        batch_items = non_empty[batch_start : batch_start + _JUDGE_BATCH_SIZE]
        batch_indices = [item[0] for item in batch_items]
        batch_prompts = [item[1] for item in batch_items]

        inputs = _judge_tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024, add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(_device)
        attention_mask = inputs["attention_mask"].to(_device)

        outputs = _judge_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]

        yes_logits = logits[:, _judge_yes_token_id]
        no_logits = logits[:, _judge_no_token_id]

        for j, orig_idx in enumerate(batch_indices):
            if yes_logits[j].item() >= no_logits[j].item():
                results[orig_idx] = JudgeResponse(score=1.0, raw_response="Yes")
            else:
                results[orig_idx] = JudgeResponse(score=0.0, raw_response="No")

    _judge_tokenizer.padding_side = orig_padding_side

    # ── Verification: compare logit vs generate on first call ──
    if not _judge_logit_verified["done"] and non_empty:
        _judge_logit_verified["done"] = True
        n_verify = min(5, len(non_empty))
        mismatch = 0
        for k in range(n_verify):
            vi = non_empty[k][0]
            s = samples[vi]
            gen_result = _judge_single_generate(s.question, s.model_answer, s.ground_truth)
            logit_score = results[vi].score
            gen_score = gen_result.score
            if gen_score < 0:
                continue
            if logit_score != gen_score:
                mismatch += 1
                print(f"[Judge] Verify sample {k}: logit={logit_score} generate={gen_score} "
                      f"(gen_raw='{gen_result.raw_response}')")
        if mismatch > n_verify // 2:
            print(f"[Judge] WARNING: {mismatch}/{n_verify} mismatches! "
                  f"Falling back to generate mode.")
            _judge_logit_verified["fallback"] = True
            for item_idx, _ in non_empty:
                s = samples[item_idx]
                results[item_idx] = _judge_single_generate(
                    s.question, s.model_answer, s.ground_truth)
        else:
            print(f"[Judge] Logit verification passed: {mismatch}/{n_verify} mismatches")

    return results


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": _device,
        "prm_model": _prm_path,
        "judge_model": _judge_path,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    return _score_single(req.question, req.thinking_text, req.agg)


@app.post("/score_batch", response_model=BatchScoreResponse)
def score_batch(req: BatchScoreRequest):
    return BatchScoreResponse(results=_score_batch_impl(req.samples))


@app.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest):
    return _judge_single(req.question, req.model_answer, req.ground_truth)


@app.post("/judge_batch", response_model=BatchJudgeResponse)
def judge_batch(req: BatchJudgeRequest):
    return BatchJudgeResponse(results=_judge_batch_impl(req.samples))


# ── Model loading ─────────────────────────────────────────────────────────

def load_models(prm_path: str, judge_path: str, device: str = "cuda"):
    global _prm_model, _prm_tokenizer, _judge_model, _judge_tokenizer
    global _device, _prm_path, _judge_path
    _device = device
    _prm_path = prm_path
    _judge_path = judge_path

    t0 = time.time()

    # ── PRM ──
    print(f"Loading PRM from {prm_path} on {device} ...")
    backbone = AutoModelForCausalLM.from_pretrained(
        prm_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()

    hidden_size = backbone.config.hidden_size
    reward_head = nn.Linear(hidden_size, 1)
    head_path = os.path.join(prm_path, "reward_head.pt")
    if os.path.exists(head_path):
        reward_head.load_state_dict(
            torch.load(head_path, map_location=device, weights_only=True)
        )
        print(f"  reward_head loaded from {head_path}")
    else:
        print(f"  WARNING: {head_path} not found, using random reward_head")
    reward_head = reward_head.to(device=device, dtype=torch.bfloat16)

    _prm_tokenizer = AutoTokenizer.from_pretrained(prm_path)
    _prm_model = PRMScorer(backbone, reward_head).eval()

    prm_mem = torch.cuda.memory_allocated(device) / 1e9
    print(f"  PRM loaded ({prm_mem:.1f} GB)")

    # ── Judge ──
    print(f"Loading Judge from {judge_path} on {device} ...")
    _judge_model = AutoModelForCausalLM.from_pretrained(
        judge_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    _judge_tokenizer = AutoTokenizer.from_pretrained(judge_path)

    global _judge_yes_token_id, _judge_no_token_id
    _judge_yes_token_id = _judge_tokenizer.encode("Yes", add_special_tokens=False)[0]
    _judge_no_token_id = _judge_tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"  Judge logit token IDs: Yes={_judge_yes_token_id}, No={_judge_no_token_id}")

    total_mem = torch.cuda.memory_allocated(device) / 1e9
    print(f"  Judge loaded ({total_mem - prm_mem:.1f} GB)")

    print(f"Reward server ready on {device} — "
          f"total GPU memory: {total_mem:.1f} GB ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="MED-R1 Reward Server (PRM + Judge)")
    parser.add_argument("--prm_path", type=str, required=True,
                        help="PRM checkpoint path")
    parser.add_argument("--judge_path", type=str, required=True,
                        help="Judge LLM path (typically SFT model)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    load_models(args.prm_path, args.judge_path, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
