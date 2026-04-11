"""
DAPO (Decoupled Alignment Policy Optimization) Trainer with PRM + answer rewards.

Core algorithm (differences from GRPO marked with [DAPO]):
    For each question in the batch:
        1. Sample K responses from the policy (group_size = K)
        2. Score each response (PRM + answer verification)
        3. [DAPO] Dynamic sampling: skip groups where all rewards are identical
        4. [DAPO] Overlong shaping: truncated responses get discounted reward
        5. Compute group-relative advantages: A_i = (r_i - mean) / std
        6. [DAPO] Token-level clipped surrogate with asymmetric clipping:
              ratio = exp(new_logp - old_logp)
              clipped = clamp(ratio, 1-ε_low, 1+ε_high)
              loss = -min(ratio * A, clipped * A)
        7. [DAPO] No KL penalty — policy constrained by clipping alone
"""

import gc
import os
import re
import time
import math
import random
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerControl,
    TrainerState,
)


# ---------------------------------------------------------------------------
# Inline replacements for trl utilities 
# ---------------------------------------------------------------------------
@contextmanager
def unwrap_model_for_generation(model, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    is_zero3 = (
        getattr(accelerator.state, "deepspeed_plugin", None) is not None
        and accelerator.state.deepspeed_plugin.zero_stage == 3
    )

    original_use_cache = unwrapped.config.use_cache
    had_gradient_checkpointing = getattr(unwrapped, "is_gradient_checkpointing", False)

    unwrapped.config.use_cache = True
    if had_gradient_checkpointing:
        unwrapped.gradient_checkpointing_disable()

    accelerator.wait_for_everyone()

    if is_zero3:
        import deepspeed
        with deepspeed.zero.GatheredParameters(unwrapped.parameters()):
            unwrapped.eval()
            yield unwrapped
            unwrapped.train()
    else:
        unwrapped.eval()
        yield unwrapped
        unwrapped.train()

    accelerator.wait_for_everyone()

    unwrapped.config.use_cache = original_use_cache
    if had_gradient_checkpointing:
        unwrapped.gradient_checkpointing_enable()
        if hasattr(unwrapped, "enable_input_require_grads"):
            unwrapped.enable_input_require_grads()


class OnlineTrainerState(TrainerState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode = 0


def disable_dropout_in_model(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0


def first_true_indices(bools, dtype=torch.long):
    row_len = bools.size(-1)
    indices = torch.where(
        bools,
        torch.arange(row_len, device=bools.device).expand_as(bools),
        row_len,
    )
    return indices.min(dim=-1).values.to(dtype)


def truncate_response(eos_token_id, pad_token_id, responses):
    eos_mask = responses == eos_token_id
    first_eos = first_true_indices(eos_mask)
    seq_len = responses.size(1)
    pos = torch.arange(seq_len, device=responses.device).unsqueeze(0).expand_as(responses)
    keep = pos <= first_eos.unsqueeze(1)
    return torch.where(keep, responses, pad_token_id)


# ---------------------------------------------------------------------------
# PRM wrapper for inference during training
# ---------------------------------------------------------------------------
class PRMScorer(nn.Module):
    def __init__(self, backbone, reward_head):
        super().__init__()
        self.backbone = backbone
        self.reward_head = reward_head

    @torch.no_grad()
    def score_steps(self, input_ids, attention_mask, step_end_positions):
        """
        Score multiple steps in a single forward pass.
        step_end_positions: list of token indices where each step ends.
        Returns: list of float scores.
        """
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1].clone()  # (1, L, H)
        del outputs
        seq_len = hidden.size(1)
        scores = []
        for pos in step_end_positions:
            pos = min(pos, seq_len - 1)
            h = hidden[0, pos]
            s = torch.sigmoid(self.reward_head(h)).item()
            scores.append(s)
        del hidden
        return scores


# ---------------------------------------------------------------------------
# Response parsing 
# ---------------------------------------------------------------------------
THINKING_PATTERN = re.compile(r"## Thinking\n\n(.*?)(?=\n\n## Final Response)", re.S)
RESPONSE_PATTERN = re.compile(r"## Final Response\n\n(.*)", re.S)


def extract_thinking_and_response(text):
    thinking_match = THINKING_PATTERN.search(text)
    response_match = RESPONSE_PATTERN.search(text)
    thinking = thinking_match.group(1) if thinking_match else None
    response = response_match.group(1).strip() if response_match else None
    return thinking, response


# ---------------------------------------------------------------------------
# Answer verification
# ---------------------------------------------------------------------------
def verify_answer(model_answer, ground_truth):
    """
    Check if the model's final answer matches the ground truth.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    if not model_answer or not ground_truth:
        return 0.0

    ans_lower = model_answer.lower().strip()
    gt_lower = ground_truth.lower().strip()

    if gt_lower in ans_lower:
        return 1.0

    gt_words = [w for w in gt_lower.split() if len(w) > 2]
    if gt_words and all(w in ans_lower for w in gt_words):
        return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# LLM-as-Judge answer verification via vLLM server
# ---------------------------------------------------------------------------
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


def verify_answer_llm(question, model_answer, ground_truth,
                       vllm_base_url, vllm_model_name, processing_class):
    """Use the LLM served by vLLM to judge answer correctness.

    Falls back to keyword matching on any failure.
    """
    import requests as _requests

    if not model_answer or not ground_truth:
        return 0.0

    judge_content = _JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer[:500],
    )

    message = [{"role": "user", "content": judge_content}]
    prompt_text = processing_class.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    try:
        resp = _requests.post(
            f"{vllm_base_url}/v1/completions",
            json={
                "model": vllm_model_name,
                "prompt": prompt_text,
                "max_tokens": 3,
                "temperature": 0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["text"].strip().lower()
        if content.startswith("yes"):
            return 1.0
        if content.startswith("no"):
            return 0.0
        return verify_answer(model_answer, ground_truth)
    except Exception:
        return verify_answer(model_answer, ground_truth)


def batch_verify_answers_llm(question, model_answers, ground_truth,
                              vllm_base_url, vllm_model_name, processing_class):
    """Judge multiple answers for the same question concurrently."""
    from concurrent.futures import ThreadPoolExecutor

    def _judge_one(ans):
        return verify_answer_llm(
            question, ans, ground_truth,
            vllm_base_url, vllm_model_name, processing_class,
        )

    with ThreadPoolExecutor(max_workers=len(model_answers)) as pool:
        return list(pool.map(_judge_one, model_answers))


# ---------------------------------------------------------------------------
# PRM step scoring 
# ---------------------------------------------------------------------------
def _aggregate_scores(step_scores, agg):
    if not step_scores:
        return 0.0
    if agg == "min":
        return min(step_scores)
    elif agg == "mean":
        return sum(step_scores) / len(step_scores)
    elif agg == "last":
        return step_scores[-1]
    elif agg == "weighted_mean":
        weights = list(range(1, len(step_scores) + 1))
        return sum(s * w for s, w in zip(step_scores, weights)) / sum(weights)
    return min(step_scores)


def compute_prm_step_scores(prm_model, prm_tokenizer, question, steps, device):
    """Run PRM on the reasoning steps, return per-step scores."""
    prefix = f"Question: {question}\n\nReasoning:\n\n"
    full_text = prefix + "\n\n".join(steps)

    encoding = prm_tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
    full_tokens = encoding["input_ids"]
    offsets = encoding.get("offset_mapping")

    if offsets is not None:
        step_end_chars = []
        char_pos = len(prefix)
        for i, step in enumerate(steps):
            char_pos += len(step)
            step_end_chars.append(char_pos - 1)
            if i < len(steps) - 1:
                char_pos += 2  # "\n\n"

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
            partial_text = prefix + "\n\n".join(steps[: i + 1])
            partial_tokens = prm_tokenizer.encode(partial_text, add_special_tokens=True)
            step_end_positions.append(len(partial_tokens) - 1)

    input_ids = torch.LongTensor([full_tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)
    return prm_model.score_steps(input_ids, attention_mask, step_end_positions)


# ---------------------------------------------------------------------------
# Combined reward:  answer verification + PRM process reward + format check
# ---------------------------------------------------------------------------
def compute_reward(prm_model, prm_tokenizer, question, response_text, ground_truth,
                   config, device, ans_reward_override=None):
    """
    Compute the total reward for a single response.
    Returns (total_reward, info_dict).

    If *ans_reward_override* is provided (from LLM judge), it is used
    instead of keyword matching.
    """
    thinking, final_resp = extract_thinking_and_response(response_text)

    if thinking is None or final_resp is None:
        return config.format_penalty, {"answer": 0.0, "prm": 0.0, "format_ok": False}

    if ans_reward_override is not None:
        ans_reward = ans_reward_override
    else:
        ans_reward = verify_answer(final_resp, ground_truth)

    steps = [s.strip() for s in thinking.split("\n\n") if s.strip()]
    if not steps:
        return config.format_penalty, {"answer": ans_reward, "prm": 0.0, "format_ok": False}

    step_scores = compute_prm_step_scores(prm_model, prm_tokenizer, question, steps, device)
    prm_reward = _aggregate_scores(step_scores, config.prm_agg)

    total = config.answer_reward_weight * ans_reward + config.prm_reward_weight * prm_reward
    return total, {"answer": ans_reward, "prm": prm_reward, "format_ok": True}


# ---------------------------------------------------------------------------
# DAPO Trainer
# ---------------------------------------------------------------------------
class GRPOTrainer(Trainer):
    _tag_names = ["trl", "dapo"]

    def __init__(
        self,
        config,
        processing_class: PreTrainedTokenizerBase,
        policy: nn.Module,
        ref_policy: nn.Module,
        prm_model: PRMScorer,
        prm_tokenizer: PreTrainedTokenizerBase,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        callbacks=None,
    ):
        self.args = config
        self.processing_class = processing_class
        self.policy = policy
        self.ref_policy = ref_policy
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.optimizer = None
        self.lr_scheduler = None
        self.optimizer_cls_and_kwargs = None

        args = config
        accelerator = Accelerator()
        self.accelerator = accelerator

        args.world_size = accelerator.num_processes
        args.local_process_index = accelerator.local_process_index
        args.process_index = accelerator.process_index
        args.batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * args.world_size
        args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)
        self.local_seed = args.seed + accelerator.process_index * 100003

        for module in [self.policy, self.ref_policy]:
            if module is not None:
                disable_dropout_in_model(module)

        if args.gradient_checkpointing:
            self.policy.gradient_checkpointing_enable()
            if hasattr(self.policy, "enable_input_require_grads"):
                self.policy.enable_input_require_grads()

        self.model = self.policy
        self.model.config = self.policy.config

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )
        from transformers import get_cosine_schedule_with_warmup
        warmup_steps = int(args.num_total_batches * getattr(args, "warmup_ratio", 0.05))
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.num_total_batches,
        )

        self.control = TrainerControl()
        self.state = OnlineTrainerState()
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None

        if args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.tb_writer = None
        self.hub_model_id = None

        # Dataloader
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.lr_scheduler
        )
        torch.manual_seed(self.local_seed)

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.per_device_batch_size,
                collate_fn=self.data_collator,
                drop_last=True,
            )
            self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.ref_policy is not None:
            if self.is_deepspeed_enabled:
                import deepspeed
                ds_config = {
                    "train_micro_batch_size_per_gpu": args.per_device_batch_size,
                    "bf16": {"enabled": getattr(args, "bf16", True)},
                    "fp16": {"enabled": False},
                    "zero_optimization": {
                        "stage": 3,
                        "offload_param": {"device": "none"},
                    },
                }
                self.ref_policy, *_ = deepspeed.initialize(
                    model=self.ref_policy, config=ds_config
                )
                self.ref_policy.eval()
            else:
                self.ref_policy = self.ref_policy.to(self.accelerator.device)

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        self.tb_writer = None
        if self.accelerator.is_main_process and args.report_to == "tensorboard":
            log_dir = args.logging_dir or os.path.join(args.output_dir, "tb_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            self.accelerator.print(f"TensorBoard logging to {log_dir}")

        if args.use_vllm or args.use_llm_judge:
            import requests as _requests
            try:
                r = _requests.get(f"{args.vllm_base_url}/v1/models", timeout=10)
                r.raise_for_status()
                model_ids = [m["id"] for m in r.json().get("data", [])]
                accelerator.print(f"vLLM server connected at {args.vllm_base_url}, models: {model_ids}")
                if args.use_vllm:
                    accelerator.print("  → vLLM used for generation (off-policy)")
                else:
                    accelerator.print("  → Generation uses training model (on-policy)")
                if args.use_llm_judge:
                    accelerator.print("  → LLM Judge enabled for answer verification")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot reach vLLM server at {args.vllm_base_url}: {e}\n"
                    f"Start it first: bash run/start_vllm_server.sh {args.model_name_or_path}"
                )

    def log(self, logs, start_time=None):
        if self.accelerator.is_main_process:
            step = self.state.global_step
            msg = f"[Step {step}] " + "  ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
            self.accelerator.print(msg)
            if self.tb_writer is not None:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(k, v, step)
                self.tb_writer.flush()

    def get_train_dataloader(self):
        return self.dataloader

    def get_eval_dataloader(self):
        return self.eval_dataloader

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)

        if hasattr(unwrapped, "save_pretrained") and hasattr(unwrapped, "peft_config"):
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(output_dir)
                self.processing_class.save_pretrained(output_dir)
        else:
            backup = self.model
            self.model = self.policy
            Trainer.save_model(self, output_dir, _internal_call)
            self.model = backup

        if self.accelerator.is_main_process:
            import json
            state = {
                "global_step": self.state.global_step,
                "episode": self.state.episode,
                "accumulate_rewards": getattr(self, "_accumulate_rewards", []),
                "accumulate_answer_acc": getattr(self, "_accumulate_answer_acc", []),
                "skipped_groups": getattr(self, "_skipped_groups", 0),
            }
            with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
                json.dump(state, f)

    def _save(self, output_dir=None, state_dict=None):
        if self.is_deepspeed_enabled and state_dict is not None:
            state_dict = {
                name.removeprefix("policy."): param
                for name, param in state_dict.items()
                if name.startswith("policy.")
            }
        super()._save(output_dir, state_dict)

    # ------------------------------------------------------------------
    # vLLM generation helpers
    # ------------------------------------------------------------------
    def _resolve_vllm_model_name(self):
        """Resolve vLLM model name once and cache it."""
        if getattr(self, "_vllm_model_name_resolved", None):
            return self._vllm_model_name_resolved

        import requests as _requests
        if self.args.vllm_model_name:
            self._vllm_model_name_resolved = self.args.vllm_model_name
        else:
            try:
                r = _requests.get(f"{self.args.vllm_base_url}/v1/models", timeout=10)
                r.raise_for_status()
                models = r.json().get("data", [])
                if models:
                    self._vllm_model_name_resolved = models[0]["id"]
                else:
                    self._vllm_model_name_resolved = self.args.model_name_or_path
            except Exception:
                self._vllm_model_name_resolved = self.args.model_name_or_path
        return self._vllm_model_name_resolved

    def _generate_vllm(self, prompt_text, n, max_retries=3):
        """Generate *n* completions for a single prompt via the vLLM server."""
        import requests as _requests

        url = f"{self.args.vllm_base_url}/v1/completions"
        model_name = self._resolve_vllm_model_name()
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "n": n,
            "max_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature + 1e-7,
            "top_p": self.args.top_p,
        }
        for attempt in range(max_retries):
            try:
                resp = _requests.post(url, json=payload, timeout=600)
                resp.raise_for_status()
                choices = sorted(resp.json()["choices"], key=lambda x: x["index"])
                texts = [c["text"] for c in choices]
                finish_reasons = [c.get("finish_reason", "length") for c in choices]
                return texts, finish_reasons
            except Exception as e:
                if attempt == max_retries - 1:
                    self.accelerator.print(
                        f"[vLLM ERROR] Failed after {max_retries} attempts: {e}"
                    )
                    return [""] * n, ["length"] * n
                self.accelerator.print(
                    f"[vLLM WARN] Attempt {attempt+1}/{max_retries} failed: {e}, retrying..."
                )
                time.sleep(2 ** attempt)
        return [""] * n, ["length"] * n

    def _texts_to_response_tensor(self, response_texts, finish_reasons, max_len, device):
        """Convert generated texts to a padded response token-ID tensor."""
        pad_id = self.processing_class.pad_token_id
        eos_id = self.processing_class.eos_token_id

        all_ids = []
        for text, reason in zip(response_texts, finish_reasons):
            ids = self.processing_class.encode(text, add_special_tokens=False)
            if not ids:
                ids = [pad_id]
            if reason == "stop" and eos_id is not None and ids[-1] != eos_id:
                ids.append(eos_id)
            ids = ids[:max_len]
            all_ids.append(ids)

        max_resp_len = max(len(ids) for ids in all_ids)
        padded = [ids + [pad_id] * (max_resp_len - len(ids)) for ids in all_ids]
        return torch.LongTensor(padded).to(device)

    # ------------------------------------------------------------------
    # DAPO token-level clipped surrogate loss
    # ------------------------------------------------------------------
    @staticmethod
    def _dapo_policy_loss(
        new_token_lp,      # (K, L) new policy log-probs (with grad)
        old_token_lp,      # (K, L) old policy log-probs (detached)
        advantages,        # (K,)   group-relative advantages
        response_mask,     # (K, L) 1 for real tokens, 0 for padding
        clip_low,          # float  lower clip bound (e.g. 0.2)
        clip_high,         # float  upper clip bound (e.g. 0.28)
    ):
        """
        Token-level PPO clip loss with DAPO's asymmetric clipping.
        Larger clip_high allows bigger probability increases → encourages exploration.
        """
        ratio = torch.exp(new_token_lp - old_token_lp)  # (K, L)

        # Expand advantages to token level
        token_adv = advantages.unsqueeze(1) * response_mask  # (K, L)

        surr1 = ratio * token_adv
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
        surr2 = clipped_ratio * token_adv

        # min for positive advantage, effectively max for negative
        loss = -torch.min(surr1, surr2)

        # Average over non-padding tokens
        num_tokens = response_mask.sum().clamp(min=1.0)
        return loss.sum() / num_tokens

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        processing_class = self.processing_class
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from self.dataloader
        iter_dataloader = iter(repeat_generator())

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature + 1e-7,
            top_p=args.top_p,
            do_sample=True,
        )

        from tqdm import tqdm

        accelerator.print("=== DAPO Training with PRM + Answer Verification ===")
        accelerator.print(f"    clip: [{args.clip_range_low}, {args.clip_range_high}]  "
                          f"dynamic_sampling={args.dynamic_sampling}  "
                          f"overlong_factor={args.overlong_factor}")
        accelerator.print(f"    total_batches={args.num_total_batches}  lr={args.learning_rate}")
        start_time = time.time()
        model.train()

        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len

        resume_step = 0
        if getattr(args, "resume_from_checkpoint", ""):
            import json
            ckpt_dir = args.resume_from_checkpoint
            state_file = os.path.join(ckpt_dir, "trainer_state.json")
            if os.path.exists(state_file):
                with open(state_file) as f:
                    saved = json.load(f)
                resume_step = saved["global_step"]
                self.state.global_step = resume_step
                self.state.episode = saved["episode"]
                for _ in range(resume_step):
                    self.lr_scheduler.step()
                self._resume_accumulate_rewards = saved.get("accumulate_rewards", [])
                self._resume_accumulate_answer_acc = saved.get("accumulate_answer_acc", [])
                self._resume_skipped_groups = saved.get("skipped_groups", 0)
                accelerator.print(f"  Resuming from step {resume_step}, episode {self.state.episode}")
                accelerator.print(f"  LR scheduler fast-forwarded to step {resume_step}, "
                                  f"current lr={self.lr_scheduler.get_last_lr()[0]:.6e}")
                accelerator.print(f"  Restored running avg from {len(self._resume_accumulate_rewards)} samples")

        if args.logging_steps and args.logging_steps >= 1:
            self.state.logging_steps = args.logging_steps
        if args.eval_steps and args.eval_steps >= 1:
            self.state.eval_steps = args.eval_steps
        if args.save_steps and args.save_steps >= 1:
            self.state.save_steps = args.save_steps

        accumulate_rewards = getattr(self, "_resume_accumulate_rewards", []).copy()
        accumulate_answer_acc = getattr(self, "_resume_accumulate_answer_acc", []).copy()
        skipped_groups = getattr(self, "_resume_skipped_groups", 0)

        gas = args.gradient_accumulation_steps
        pbar = tqdm(range(1, args.num_total_batches + 1), desc="DAPO", disable=not accelerator.is_main_process,
                    initial=resume_step, total=args.num_total_batches)
        for update in pbar:
            if update <= resume_step:
                for _ in range(gas):
                    next(iter_dataloader)
                continue

            self.state.episode += args.batch_size

            # ===== Accumulate gradients across micro-batches =====
            optimizer.zero_grad()
            step_pg_loss = 0.0
            step_total_valid = 0
            step_all_rewards = []
            step_all_answer_accs = []
            step_sample_resp = None
            step_sample_q = None
            step_sample_gt = None
            step_sample_reward = None
            step_sample_acc = None

            for micro_step in range(gas):
                data = next(iter_dataloader)

                with torch.no_grad():
                    queries = data["input_ids"].to(device)
                    questions = data["question"]
                    answers = data["answer"]
                    context_length = queries.shape[1]
                    B = queries.shape[0]

                    micro_responses = []
                    micro_old_log_probs = []
                    micro_rewards = []
                    micro_advantages = []
                    micro_answer_accs = []
                    micro_response_masks = []
                    micro_valid_indices = []
                    micro_queries = queries

                    for q_idx in range(B):
                        query = queries[q_idx: q_idx + 1]

                        torch.cuda.empty_cache()

                        if args.use_vllm:
                            message = [{"role": "user", "content": questions[q_idx]}]
                            prompt_text = processing_class.apply_chat_template(
                                message, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False,
                            )
                            resp_texts, finish_reasons = self._generate_vllm(
                                prompt_text, args.group_size,
                            )
                            responses = self._texts_to_response_tensor(
                                resp_texts, finish_reasons,
                                args.max_new_tokens, device,
                            )
                        else:
                            query_repeated = query.repeat(args.group_size, 1)
                            query_attn_mask = (query_repeated != processing_class.pad_token_id).long()
                            with unwrap_model_for_generation(model, accelerator) as unwrapped:
                                gen_output = unwrapped.generate(
                                    query_repeated,
                                    attention_mask=query_attn_mask,
                                    generation_config=generation_config,
                                    pad_token_id=processing_class.pad_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=False,
                                )
                            query_responses = gen_output.sequences
                            del gen_output
                            responses = query_responses[:, context_length:]
                            del query_responses, query_repeated, query_attn_mask

                            if processing_class.eos_token_id is not None:
                                responses = truncate_response(
                                    processing_class.eos_token_id,
                                    processing_class.pad_token_id,
                                    responses,
                                )

                        full_seqs = torch.cat([query.repeat(args.group_size, 1), responses], dim=1)
                        attn_mask = (full_seqs != processing_class.pad_token_id).long()

                        torch.cuda.empty_cache()

                        _unwrapped = accelerator.unwrap_model(model)
                        _had_gc = getattr(_unwrapped, "is_gradient_checkpointing", False)
                        if _had_gc:
                            _unwrapped.gradient_checkpointing_disable()

                        all_out = model(full_seqs, attention_mask=attn_mask, use_cache=False)
                        all_logits = all_out.logits[:, context_length - 1: -1]
                        all_logits = all_logits / (args.temperature + 1e-7)
                        lp_flat = -F.cross_entropy(
                            all_logits.reshape(-1, all_logits.size(-1)),
                            responses.reshape(-1),
                            reduction="none",
                        )
                        old_token_lp = lp_flat.reshape(responses.shape)
                        del all_out, all_logits, lp_flat

                        if _had_gc:
                            _unwrapped.gradient_checkpointing_enable()
                            if hasattr(_unwrapped, "enable_input_require_grads"):
                                _unwrapped.enable_input_require_grads()
                        torch.cuda.empty_cache()

                        response_mask = (responses != processing_class.pad_token_id).float()
                        old_token_lp = old_token_lp * response_mask

                        contains_eos = torch.any(
                            responses == processing_class.eos_token_id, dim=-1
                        )

                        prm_device = self.prm_model.backbone.device

                        resp_texts_decoded = [
                            processing_class.decode(responses[k], skip_special_tokens=True)
                            for k in range(args.group_size)
                        ]

                        llm_judge_scores = [None] * args.group_size
                        if args.use_llm_judge:
                            final_answers = []
                            for rt in resp_texts_decoded:
                                _, fa = extract_thinking_and_response(rt)
                                final_answers.append(fa or "")
                            llm_judge_scores = batch_verify_answers_llm(
                                questions[q_idx], final_answers, answers[q_idx],
                                args.vllm_base_url,
                                self._resolve_vllm_model_name(),
                                processing_class,
                            )
                            if accelerator.is_main_process and random.random() < 0.05:
                                n_yes = sum(1 for s in llm_judge_scores if s == 1.0)
                                accelerator.print(
                                    f"  [Judge] {n_yes}/{len(llm_judge_scores)} correct  "
                                    f"GT={answers[q_idx][:60]}  "
                                    f"sample_ans={final_answers[0][:60]}"
                                )

                        group_rewards = []
                        group_answer_acc = []
                        for k in range(args.group_size):
                            reward, info = compute_reward(
                                self.prm_model, self.prm_tokenizer,
                                questions[q_idx], resp_texts_decoded[k], answers[q_idx],
                                args, prm_device,
                                ans_reward_override=llm_judge_scores[k],
                            )
                            if not contains_eos[k].item():
                                if reward > 0:
                                    reward = reward * args.overlong_factor
                                else:
                                    reward = reward - 1.0
                            group_rewards.append(reward)
                            group_answer_acc.append(info["answer"])
                            torch.cuda.empty_cache()

                        rewards_tensor = torch.tensor(group_rewards, device=device, dtype=torch.float32)
                        rewards_tensor = torch.clamp(rewards_tensor, min=-10.0, max=10.0)

                        if args.dynamic_sampling and rewards_tensor.std() < 1e-6:
                            skipped_groups += 1
                            accumulate_rewards.append(rewards_tensor.mean().item())
                            accumulate_answer_acc.append(sum(group_answer_acc) / len(group_answer_acc))
                            torch.cuda.empty_cache()
                            continue

                        mean_r = rewards_tensor.mean()
                        std_r = rewards_tensor.std() + 1e-4
                        advantages = torch.clamp((rewards_tensor - mean_r) / std_r, -5.0, 5.0)

                        micro_valid_indices.append(q_idx)
                        micro_responses.append(responses)
                        micro_old_log_probs.append(old_token_lp)
                        micro_rewards.append(rewards_tensor)
                        micro_advantages.append(advantages)
                        micro_response_masks.append(response_mask)
                        micro_answer_accs.append(sum(group_answer_acc) / len(group_answer_acc))

                        accumulate_rewards.append(mean_r.item())
                        accumulate_answer_acc.append(micro_answer_accs[-1])

                        torch.cuda.empty_cache()

                # ----- Backward for this micro-batch (with grad) -----
                # Per-response forward+backward to avoid OOM from batching
                # all group_size responses at once (activations for 8 seqs is huge).
                n_micro_valid = len(micro_valid_indices)
                if n_micro_valid > 0:
                    for i, q_idx in enumerate(micro_valid_indices):
                        responses = micro_responses[i]
                        old_token_lp = micro_old_log_probs[i]
                        adv = micro_advantages[i]
                        response_mask = micro_response_masks[i]

                        total_resp_tokens = response_mask.sum().clamp(min=1.0)
                        scale = 1.0 / (n_micro_valid * gas)
                        group_pg_loss = 0.0

                        for k in range(args.group_size):
                            query_k = micro_queries[q_idx: q_idx + 1]
                            resp_k = responses[k: k + 1]
                            full_k = torch.cat([query_k, resp_k], dim=1)
                            mask_k = (full_k != processing_class.pad_token_id).long()

                            out_k = model(full_k, attention_mask=mask_k, use_cache=False)
                            logits_k = out_k.logits[:, context_length - 1: -1]
                            logits_k = logits_k / (args.temperature + 1e-7)
                            lp_k = F.log_softmax(logits_k, dim=-1)
                            new_lp_k = torch.gather(lp_k, 2, resp_k.unsqueeze(-1)).squeeze(-1)
                            rmask_k = response_mask[k: k + 1]
                            new_lp_k = new_lp_k * rmask_k

                            old_lp_k = old_token_lp[k: k + 1].detach()
                            adv_k = adv[k: k + 1].detach()

                            ratio_k = torch.exp(new_lp_k - old_lp_k)
                            token_adv_k = adv_k.unsqueeze(1) * rmask_k
                            surr1 = ratio_k * token_adv_k
                            clipped_ratio_k = torch.clamp(ratio_k, 1.0 - args.clip_range_low, 1.0 + args.clip_range_high)
                            surr2 = clipped_ratio_k * token_adv_k
                            loss_k = (-torch.min(surr1, surr2)).sum() / total_resp_tokens

                            accelerator.backward(loss_k * scale)
                            group_pg_loss += loss_k.detach().item()

                            del out_k, logits_k, lp_k, new_lp_k, ratio_k, loss_k
                            torch.cuda.empty_cache()

                        step_pg_loss += group_pg_loss

                    step_total_valid += n_micro_valid
                    step_all_rewards.extend(micro_rewards)
                    step_all_answer_accs.extend(micro_answer_accs)

                    if step_sample_resp is None:
                        step_sample_resp = processing_class.decode(micro_responses[0][0], skip_special_tokens=True)
                        step_sample_q = questions[micro_valid_indices[0]]
                        step_sample_gt = answers[micro_valid_indices[0]]
                        step_sample_reward = micro_rewards[0][0].item()
                        step_sample_acc = micro_answer_accs[0]

                del micro_responses, micro_old_log_probs, micro_rewards, micro_advantages, micro_response_masks
                torch.cuda.empty_cache()

            # ===== Optimizer step (once per macro-batch) =====
            if step_total_valid == 0:
                optimizer.zero_grad()
                self.lr_scheduler.step()
                self.state.global_step += 1
                continue

            for name, p in model.named_parameters():
                if p.requires_grad and ("embed" in name or "lm_head" in name or "reward" in name):
                    if torch.isnan(p).any():
                        accelerator.print(f"[NaN DETECTED] param={name} at step {self.state.global_step}")
                        break

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if math.isnan(grad_norm_val) or math.isinf(grad_norm_val) or grad_norm_val > 100.0:
                accelerator.print(f"[WARNING] grad_norm={grad_norm_val:.4f} at step {self.state.global_step}, skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
            self.lr_scheduler.step()

            # Logging
            with torch.no_grad():
                avg_reward = sum(r.mean().item() for r in step_all_rewards) / max(step_total_valid, 1)
                avg_answer_acc = sum(step_all_answer_accs) / max(step_total_valid, 1)
                recent_n = min(50, len(accumulate_rewards))
                metrics = {
                    "loss/pg": step_pg_loss / max(step_total_valid, 1),
                    "reward/mean": avg_reward,
                    "reward/running_avg": sum(accumulate_rewards[-recent_n:]) / max(recent_n, 1),
                    "reward/answer_acc": avg_answer_acc,
                    "reward/answer_acc_running": sum(accumulate_answer_acc[-recent_n:]) / max(recent_n, 1),
                    "dapo/skipped_groups_total": skipped_groups,
                    "dapo/valid_groups_this_step": step_total_valid,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "episode": self.state.episode,
                }
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1
                self.log(metrics)
                pbar.set_postfix(loss=step_pg_loss / max(step_total_valid, 1), reward=avg_reward, acc=avg_answer_acc)

                if random.random() < 0.1 and step_sample_resp is not None:
                    accelerator.print(
                        f"\n[Step {self.state.global_step}] "
                        f"reward={step_sample_reward:.3f} "
                        f"ans_acc={step_sample_acc:.2f} "
                        f"skipped={skipped_groups}\n"
                        f"Q: {step_sample_q[:100]}...\n"
                        f"GT: {step_sample_gt[:100]}\n"
                        f"A: {step_sample_resp[:]}\n"
                    )

            _MAX_RUNNING_HISTORY = 500
            if len(accumulate_rewards) > _MAX_RUNNING_HISTORY * 2:
                accumulate_rewards = accumulate_rewards[-_MAX_RUNNING_HISTORY:]
                accumulate_answer_acc = accumulate_answer_acc[-_MAX_RUNNING_HISTORY:]

            if args.save_steps and self.state.global_step % args.save_steps == 0:
                self._accumulate_rewards = accumulate_rewards
                self._accumulate_answer_acc = accumulate_answer_acc
                self._skipped_groups = skipped_groups
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{self.state.global_step}")
                self.save_model(ckpt_dir)
                accelerator.print(f"  Saved checkpoint at step {self.state.global_step}")

                if accelerator.is_main_process and getattr(args, "save_total_limit", 0) > 0:
                    import glob, shutil
                    ckpt_dirs = sorted(
                        glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                        key=lambda d: int(d.rsplit("-", 1)[-1]),
                    )
                    while len(ckpt_dirs) > args.save_total_limit:
                        old = ckpt_dirs.pop(0)
                        shutil.rmtree(old, ignore_errors=True)
                        accelerator.print(f"  Removed old checkpoint: {old}")

            del step_all_rewards, step_all_answer_accs
            torch.cuda.empty_cache()
            gc.collect()

        self._accumulate_rewards = accumulate_rewards
        self._accumulate_answer_acc = accumulate_answer_acc
        self._skipped_groups = skipped_groups
        self.save_model(os.path.join(args.output_dir, "final"))
        if self.tb_writer is not None:
            self.tb_writer.close()
        accelerator.print(f"Training complete. Final model saved to {args.output_dir}/final")
