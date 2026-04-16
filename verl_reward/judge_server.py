"""
Standalone LLM Judge HTTP server.

Deploys an LLM (typically the SFT model) as a judge service on a dedicated GPU.
The judge evaluates whether a student answer matches the reference answer,
handling medical synonyms and abbreviations.

API:
    POST /judge
        Body: {"question": str, "model_answer": str, "ground_truth": str}
        Response: {"score": float, "raw_response": str}

    POST /judge_batch
        Body: {"samples": [{"question": str, "model_answer": str, "ground_truth": str}, ...]}
        Response: {"results": [{"score": float, "raw_response": str}, ...]}

    GET /health
        Response: {"status": "ok", "model": "...", "device": "..."}

Usage:
    CUDA_VISIBLE_DEVICES=7 python verl_reward/judge_server.py \
        --model_path /tmp/sft_model \
        --port 8200
"""

import argparse
import time
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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


# ── Request / Response schemas ────────────────────────────────────────────

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


# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(title="MED-R1 Judge Server")

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_device: str = "cuda"
_model_path: str = ""


@torch.no_grad()
def _judge_single(question: str, model_answer: str, ground_truth: str) -> JudgeResponse:
    """Run the judge on a single sample."""
    if not model_answer or not ground_truth:
        return JudgeResponse(score=0.0, raw_response="")

    judge_content = _JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer[:500],
    )
    messages = [{"role": "user", "content": judge_content}]

    try:
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = _tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(_device)

    outputs = _model.generate(
        input_ids,
        max_new_tokens=3,
        temperature=0.0,
        do_sample=False,
    )

    new_tokens = outputs[0][input_ids.shape[1]:]
    raw = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    response_lower = raw.lower()

    if response_lower.startswith("yes"):
        score = 1.0
    elif response_lower.startswith("no"):
        score = 0.0
    else:
        score = -1.0   # sentinel: caller should use keyword fallback

    return JudgeResponse(score=score, raw_response=raw)


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_path, "device": _device}


@app.post("/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest):
    return _judge_single(req.question, req.model_answer, req.ground_truth)


@app.post("/judge_batch", response_model=BatchJudgeResponse)
async def judge_batch(req: BatchJudgeRequest):
    results = [
        _judge_single(s.question, s.model_answer, s.ground_truth)
        for s in req.samples
    ]
    return BatchJudgeResponse(results=results)


def load_model(model_path: str, device: str = "cuda"):
    global _model, _tokenizer, _device, _model_path
    _device = device
    _model_path = model_path

    print(f"Loading Judge LLM from {model_path} ...")
    t0 = time.time()

    _model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Judge server ready on {device} ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="MED-R1 Judge HTTP Server")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    load_model(args.model_path, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
