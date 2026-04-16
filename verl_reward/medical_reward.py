"""
Reward function for verl DAPO training (verl 0.5.0 API).

Combines:
  1. PRM (Process Reward Model)  — reward_server.py  POST /score_batch
  2. LLM-as-Judge answer check   — reward_server.py  POST /judge_batch
  3. Format penalty for malformed responses

Performance:
  Monkey-patches NaiveRewardManager to send ALL samples via batch HTTP
  calls instead of per-sample sequential calls.
  896 samples: ~256s (old) → ~30s (new).

Reward scaling:
  correct answer  → +1.0  + PRM bonus  →  [+1.0, +1.3]
  wrong   answer  → -1.0  + PRM bonus  →  [-1.0, -0.7]
  format  error   → -1.0

Environment variables:
    REWARD_SERVICE_URL - reward server URL  (default: http://localhost:8100)
    PRM_AGG            - PRM aggregation    (default: min)
    PRM_WEIGHT         - PRM reward weight  (default: 0.3)
    ANSWER_WEIGHT      - Answer weight      (default: 1.0)
"""

import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import requests

# ── Patterns ──────────────────────────────────────────────────────────────

THINKING_PATTERN = re.compile(r"## Thinking\n\n(.*?)(?=\n\n## Final Response)", re.S)
RESPONSE_PATTERN = re.compile(r"## Final Response\n\n(.*)", re.S)

# ── Weights & config ─────────────────────────────────────────────────────

FORMAT_PENALTY = -1.0
PRM_WEIGHT     = float(os.environ.get("PRM_WEIGHT",    "0.3"))
ANSWER_WEIGHT  = float(os.environ.get("ANSWER_WEIGHT", "1.0"))
PRM_AGG        = os.environ.get("PRM_AGG", "min")

# ── Service URL ───────────────────────────────────────────────────────────

REWARD_SERVICE_URL = os.environ.get("REWARD_SERVICE_URL", "http://localhost:8100")

# ── HTTP session with connection pooling ──────────────────────────────────

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})

_executor = ThreadPoolExecutor(max_workers=2)
_warned = {"prm": False, "judge": False}


# ── Response parsing ──────────────────────────────────────────────────────

def _extract(text):
    t = THINKING_PATTERN.search(text)
    r = RESPONSE_PATTERN.search(text)
    return (t.group(1) if t else None), (r.group(1).strip() if r else None)


# ── Keyword fallback ──────────────────────────────────────────────────────

def _verify_answer_keyword(model_answer: str, ground_truth: str) -> float:
    if not model_answer or not ground_truth:
        return 0.0
    ans = model_answer.lower().strip()
    gt  = ground_truth.lower().strip()
    if gt in ans:
        return 1.0
    words = [w for w in gt.split() if len(w) > 2]
    if words and all(w in ans for w in words):
        return 1.0
    return 0.0


# ── Single-sample service calls (fallback) ────────────────────────────────

def _call_prm(question: str, thinking: str) -> tuple:
    try:
        r = _session.post(
            f"{REWARD_SERVICE_URL}/score",
            json={"question": question, "thinking_text": thinking, "agg": PRM_AGG},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["aggregated"], True
    except Exception as e:
        if not _warned["prm"]:
            _warned["prm"] = True
            print(f"[reward] PRM service unavailable ({type(e).__name__}: {e})")
        return 0.5, False


def _call_judge(question: str, model_answer: str, ground_truth: str) -> tuple:
    try:
        r = _session.post(
            f"{REWARD_SERVICE_URL}/judge",
            json={"question": question, "model_answer": model_answer[:500],
                  "ground_truth": ground_truth},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        score = data["score"]
        if score < 0:
            return _verify_answer_keyword(model_answer, ground_truth), False
        return score, True
    except Exception as e:
        if not _warned["judge"]:
            _warned["judge"] = True
            print(f"[reward] Judge service unavailable ({type(e).__name__}: {e})")
        return _verify_answer_keyword(model_answer, ground_truth), False


def _compute_reward(ans_reward: float, prm_reward: float) -> float:
    """Shift ans_reward: correct(1)→+1, wrong(0)→-1."""
    ans_shifted = ans_reward * 2.0 - 1.0
    return ANSWER_WEIGHT * ans_shifted + PRM_WEIGHT * prm_reward


# ── verl entry point (per-sample, kept for compatibility) ─────────────────

def compute_score(data_source: str,
                  solution_str: str,
                  ground_truth: str,
                  extra_info: Optional[dict] = None,
                  **kwargs) -> Union[float, dict]:
    question = ""
    if extra_info:
        question = extra_info.get("question", "")

    thinking, final_resp = _extract(solution_str)
    if thinking is None or final_resp is None:
        return {"score": FORMAT_PENALTY, "ans_reward": 0.0, "prm_reward": 0.0,
                "format_ok": False, "judge_used": False, "prm_used": False}

    prm_future  = _executor.submit(_call_prm, question, thinking)
    judge_future = _executor.submit(_call_judge, question, final_resp, ground_truth)

    prm_reward, prm_used  = prm_future.result()
    ans_reward, judge_used = judge_future.result()

    total = _compute_reward(ans_reward, prm_reward)
    return {
        "score":      total,
        "ans_reward":  ans_reward,
        "prm_reward":  prm_reward,
        "format_ok":   True,
        "judge_used":  judge_used,
        "prm_used":    prm_used,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Batch processing — called by the monkey-patched NaiveRewardManager
# ═══════════════════════════════════════════════════════════════════════════

def _batch_call_prm(items: List[dict]) -> List[float]:
    """Batch PRM via /score_batch. Returns list of aggregated scores."""
    if not items:
        return []
    try:
        r = _session.post(
            f"{REWARD_SERVICE_URL}/score_batch",
            json={"samples": items},
            timeout=600,
        )
        r.raise_for_status()
        return [res["aggregated"] for res in r.json()["results"]]
    except Exception as e:
        print(f"[reward] Batch PRM failed ({type(e).__name__}: {e}); fallback 0.5")
        return [0.5] * len(items)


def _batch_call_judge(items: List[dict]) -> List[float]:
    """Batch Judge via /judge_batch. Returns list of scores."""
    if not items:
        return []
    try:
        r = _session.post(
            f"{REWARD_SERVICE_URL}/judge_batch",
            json={"samples": items},
            timeout=600,
        )
        r.raise_for_status()
        results = []
        for res in r.json()["results"]:
            score = res["score"]
            results.append(score if score >= 0 else 0.0)
        return results
    except Exception as e:
        print(f"[reward] Batch Judge failed ({type(e).__name__}: {e}); fallback 0.0")
        return [0.0] * len(items)


def _batch_compute_all(samples: List[dict]) -> List[dict]:
    """
    Process all samples in batch. Each sample dict has keys:
      question, response_str, ground_truth
    Returns list of score dicts.
    """
    parsed = [_extract(s["response_str"]) for s in samples]

    prm_items = []
    judge_items = []
    format_ok_indices = []

    for i, (thinking, final_resp) in enumerate(parsed):
        if thinking is None or final_resp is None:
            continue
        format_ok_indices.append(i)
        q = samples[i].get("question", "")
        prm_items.append({"question": q, "thinking_text": thinking, "agg": PRM_AGG})
        judge_items.append({
            "question": q,
            "model_answer": final_resp[:500],
            "ground_truth": samples[i]["ground_truth"],
        })

    prm_future   = _executor.submit(_batch_call_prm, prm_items)
    judge_future  = _executor.submit(_batch_call_judge, judge_items)
    prm_scores  = prm_future.result()
    judge_scores = judge_future.result()

    all_scores: List[dict] = [None] * len(samples)
    ok_idx = 0
    for i in range(len(samples)):
        thinking, final_resp = parsed[i]
        if thinking is None or final_resp is None:
            all_scores[i] = {
                "score": FORMAT_PENALTY, "ans_reward": 0.0, "prm_reward": 0.0,
                "format_ok": False, "judge_used": False, "prm_used": False,
            }
        else:
            pr = prm_scores[ok_idx]
            ar = judge_scores[ok_idx]
            total = _compute_reward(ar, pr)
            all_scores[i] = {
                "score": total, "ans_reward": ar, "prm_reward": pr,
                "format_ok": True, "judge_used": True, "prm_used": True,
            }
            ok_idx += 1
    return all_scores


# ═══════════════════════════════════════════════════════════════════════════
#  Monkey-patch NaiveRewardManager for batch processing
# ═══════════════════════════════════════════════════════════════════════════

def _monkey_patch_reward_manager():
    try:
        import torch as _torch
        from verl import DataProto
        from verl.workers.reward_manager.naive import NaiveRewardManager
    except ImportError:
        return

    def _batched_call(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = _torch.zeros_like(data.batch["responses"], dtype=_torch.float32)
        reward_extra_info = defaultdict(list)

        t0 = time.time()
        n = len(data)

        samples = []
        valid_lengths = []
        data_sources = []
        prompt_strs = []
        response_strs = []
        ground_truths = []

        for i in range(n):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            vpl = item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-vpl:]

            response_ids = item.batch["responses"]
            vrl = item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:vrl]

            ps = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            rs = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            gt = item.non_tensor_batch["reward_model"]["ground_truth"]
            ds = item.non_tensor_batch[self.reward_fn_key]
            ei = item.non_tensor_batch.get("extra_info", {})

            question = ei.get("question", "") if ei else ""
            samples.append({
                "question": question,
                "response_str": rs,
                "ground_truth": gt,
            })
            valid_lengths.append(vrl.item())
            data_sources.append(ds)
            prompt_strs.append(ps)
            response_strs.append(rs)
            ground_truths.append(gt)

        scores = _batch_compute_all(samples)

        already_printed = {}
        for i, score in enumerate(scores):
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            reward_tensor[i, valid_lengths[i] - 1] = reward

            ds = data_sources[i]
            if ds not in already_printed:
                already_printed[ds] = 0
            if already_printed[ds] < self.num_examine:
                already_printed[ds] += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_strs[i])
                print("[ground_truth]", ground_truths[i])
                if isinstance(score, dict):
                    for k, v in score.items():
                        print(f"[{k}]", v)
                else:
                    print("[score]", score)

        elapsed = time.time() - t0
        print(f"[reward] Batch processed {n} samples in {elapsed:.1f}s "
              f"({elapsed/max(n,1)*1000:.0f}ms/sample)")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    NaiveRewardManager.__call__ = _batched_call
    print("[reward] NaiveRewardManager patched → batch HTTP mode")


_monkey_patch_reward_manager()
