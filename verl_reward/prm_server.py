"""
Standalone PRM (Process Reward Model) HTTP server.

Deploys the PRM from train_prm.py as a FastAPI service on a dedicated GPU,
used by verl's reward function during DAPO training via HTTP calls.

API:
    POST /score
        Body: {"question": str, "thinking_text": str, "agg": "min"|"mean"|"last"|"weighted_mean"}
        Response: {"step_scores": [float, ...], "aggregated": float}

    POST /score_batch
        Body: {"samples": [{"question": str, "thinking_text": str, "agg": str}, ...]}
        Response: {"results": [{"step_scores": [...], "aggregated": float}, ...]}

    GET /health
        Response: {"status": "ok", "model": "...", "device": "..."}

Usage:
    CUDA_VISIBLE_DEVICES=6 python verl_reward/prm_server.py \
        --prm_path /tmp/prm_qwen3_4b \
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


# ── Request / Response schemas ────────────────────────────────────────────

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


# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(title="MED-R1 PRM Server")

_model: Optional[PRMScorer] = None
_tokenizer = None
_device: str = "cuda"
_model_path: str = ""


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
    """Score a single sample (shared by /score and /score_batch)."""
    steps = [s.strip() for s in thinking_text.split("\n\n") if s.strip()]
    if not steps:
        return ScoreResponse(step_scores=[], aggregated=0.0)

    prefix = f"Question: {question}\n\nReasoning:\n\n"
    full_text = prefix + "\n\n".join(steps)

    encoding = _tokenizer(
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
            partial_tokens = _tokenizer.encode(partial, add_special_tokens=True)
            step_end_positions.append(len(partial_tokens) - 1)

    input_ids = torch.LongTensor([full_tokens]).to(_device)
    attention_mask = torch.ones_like(input_ids)

    step_scores = _model.score_steps(input_ids, attention_mask, step_end_positions)
    aggregated = _aggregate(step_scores, agg)

    return ScoreResponse(step_scores=step_scores, aggregated=aggregated)


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_path, "device": _device}


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    return _score_single(req.question, req.thinking_text, req.agg)


@app.post("/score_batch", response_model=BatchScoreResponse)
async def score_batch(req: BatchScoreRequest):
    results = [
        _score_single(s.question, s.thinking_text, s.agg)
        for s in req.samples
    ]
    return BatchScoreResponse(results=results)


def load_model(prm_path: str, device: str = "cuda"):
    global _model, _tokenizer, _device, _model_path
    _device = device
    _model_path = prm_path

    print(f"Loading PRM backbone from {prm_path} ...")
    t0 = time.time()

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
        print(f"Loaded reward_head from {head_path}")
    else:
        print(f"WARNING: {head_path} not found, using random reward_head")
    reward_head = reward_head.to(device=device, dtype=torch.bfloat16)

    _tokenizer = AutoTokenizer.from_pretrained(prm_path)
    _model = PRMScorer(backbone, reward_head).eval()
    print(f"PRM server ready on {device} ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="MED-R1 PRM HTTP Server")
    parser.add_argument("--prm_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    load_model(args.prm_path, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
