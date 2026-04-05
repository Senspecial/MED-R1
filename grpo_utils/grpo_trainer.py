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
from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)

from contextlib import contextmanager
from transformers import TrainerState


# ---------------------------------------------------------------------------
# Inline replacements for trl utilities (torch 2.3.1 compatible)
# ---------------------------------------------------------------------------
@contextmanager
def unwrap_model_for_generation(model, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    is_zero3 = (
        getattr(accelerator.state, "deepspeed_plugin", None) is not None
        and accelerator.state.deepspeed_plugin.zero_stage == 3
    )
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

if is_wandb_available():
    import wandb


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
        hidden = outputs.hidden_states[-1]  # (1, L, H)
        scores = []
        for pos in step_end_positions:
            h = hidden[0, pos]
            s = torch.sigmoid(self.reward_head(h)).item()
            scores.append(s)
        return scores


# ---------------------------------------------------------------------------
# Response parsing — aligned with SFT format from SFT_stage1.py
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
# Answer verification — lightweight keyword matching for medical QA
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
# PRM step scoring — aligned with train_prm.py format
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
    full_text = f"Question: {question}\n\nReasoning:\n\n" + "\n\n".join(steps)
    full_tokens = prm_tokenizer.encode(full_text, add_special_tokens=True)

    step_end_positions = []
    for i in range(len(steps)):
        partial_text = f"Question: {question}\n\nReasoning:\n\n" + "\n\n".join(steps[: i + 1])
        partial_tokens = prm_tokenizer.encode(partial_text, add_special_tokens=True)
        step_end_positions.append(len(partial_tokens) - 1)

    input_ids = torch.LongTensor([full_tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)
    return prm_model.score_steps(input_ids, attention_mask, step_end_positions)


# ---------------------------------------------------------------------------
# Combined reward:  answer verification + PRM process reward + format check
# ---------------------------------------------------------------------------
def compute_reward(prm_model, prm_tokenizer, question, response_text, ground_truth, config, device):
    """
    Compute the total reward for a single response.
    Returns (total_reward, info_dict).
    """
    thinking, final_resp = extract_thinking_and_response(response_text)

    if thinking is None or final_resp is None:
        return config.format_penalty, {"answer": 0.0, "prm": 0.0, "format_ok": False}

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
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
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
        if accelerator.is_main_process:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

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
        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
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
                accelerator.print(f"  Resuming from step {resume_step}, episode {self.state.episode}")

        if args.logging_steps and args.logging_steps >= 1:
            self.state.logging_steps = args.logging_steps
        if args.eval_steps and args.eval_steps >= 1:
            self.state.eval_steps = args.eval_steps
        if args.save_steps and args.save_steps >= 1:
            self.state.save_steps = args.save_steps

        accumulate_rewards = []
        accumulate_answer_acc = []
        skipped_groups = 0

        pbar = tqdm(range(1, args.num_total_batches + 1), desc="DAPO", disable=not accelerator.is_main_process,
                    initial=resume_step, total=args.num_total_batches)
        for update in pbar:
            self.state.episode += args.batch_size
            data = next(iter_dataloader)

            if update <= resume_step:
                continue

            with torch.no_grad():
                queries = data["input_ids"].to(device)
                questions = data["question"]
                answers = data["answer"]
                context_length = queries.shape[1]
                B = queries.shape[0]

                # ----- Step 1: Generate K responses per query -----
                all_responses = []
                all_old_log_probs = []
                all_rewards = []
                all_advantages = []
                all_answer_accs = []
                all_response_masks = []
                valid_indices = []

                for q_idx in range(B):
                    query = queries[q_idx: q_idx + 1]
                    query_repeated = query.repeat(args.group_size, 1)

                    with unwrap_model_for_generation(model, accelerator) as unwrapped:
                        gen_output = unwrapped.generate(
                            query_repeated,
                            generation_config=generation_config,
                            pad_token_id=processing_class.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=False,
                        )
                    query_responses = gen_output.sequences
                    responses = query_responses[:, context_length:]

                    if processing_class.eos_token_id is not None:
                        responses = truncate_response(
                            processing_class.eos_token_id,
                            processing_class.pad_token_id,
                            responses,
                        )

                    full_seqs = torch.cat([query.repeat(args.group_size, 1), responses], dim=1)
                    attn_mask = (full_seqs != processing_class.pad_token_id).long()

                    # Old policy log-probs (no grad, for ratio computation)
                    policy_out = model(full_seqs, attention_mask=attn_mask)
                    logits = policy_out.logits[:, context_length - 1: -1]
                    logits = logits / (args.temperature + 1e-7)
                    log_probs = F.log_softmax(logits, dim=-1)
                    old_token_lp = torch.gather(log_probs, 2, responses.unsqueeze(-1)).squeeze(-1)

                    response_mask = (responses != processing_class.pad_token_id).float()
                    old_token_lp = old_token_lp * response_mask

                    # Check which responses were truncated (no EOS)
                    contains_eos = torch.any(
                        responses == processing_class.eos_token_id, dim=-1
                    )  # (K,)

                    # ----- Step 2: Combined reward (PRM + answer) -----
                    prm_device = self.prm_model.backbone.device
                    group_rewards = []
                    group_answer_acc = []
                    for k in range(args.group_size):
                        resp_text = processing_class.decode(responses[k], skip_special_tokens=True)
                        reward, info = compute_reward(
                            self.prm_model, self.prm_tokenizer,
                            questions[q_idx], resp_text, answers[q_idx],
                            args, prm_device,
                        )
                        # [DAPO] Overlong shaping: discount reward for truncated responses
                        if not contains_eos[k].item():
                            reward = reward * args.overlong_factor
                        group_rewards.append(reward)
                        group_answer_acc.append(info["answer"])

                    rewards_tensor = torch.tensor(group_rewards, device=device, dtype=torch.float32)

                    # [DAPO] Dynamic sampling: skip groups with no reward variance
                    if args.dynamic_sampling and rewards_tensor.std() < 1e-6:
                        skipped_groups += 1
                        accumulate_rewards.append(rewards_tensor.mean().item())
                        accumulate_answer_acc.append(sum(group_answer_acc) / len(group_answer_acc))
                        del policy_out, logits, log_probs
                        torch.cuda.empty_cache()
                        continue

                    # Group-relative advantage
                    mean_r = rewards_tensor.mean()
                    std_r = rewards_tensor.std() + 1e-8
                    advantages = (rewards_tensor - mean_r) / std_r

                    valid_indices.append(q_idx)
                    all_responses.append(responses)
                    all_old_log_probs.append(old_token_lp)
                    all_rewards.append(rewards_tensor)
                    all_advantages.append(advantages)
                    all_response_masks.append(response_mask)
                    all_answer_accs.append(sum(group_answer_acc) / len(group_answer_acc))

                    accumulate_rewards.append(mean_r.item())
                    accumulate_answer_acc.append(all_answer_accs[-1])

                    del policy_out, logits, log_probs
                    torch.cuda.empty_cache()

            # ----- Step 3: DAPO policy gradient update -----
            n_valid = len(valid_indices)
            if n_valid == 0:
                # All groups were skipped — no gradient this step
                optimizer.zero_grad()
                self.state.global_step += 1
                continue

            total_pg_loss = torch.tensor(0.0, device=device)

            for i, q_idx in enumerate(valid_indices):
                responses = all_responses[i]
                old_token_lp = all_old_log_probs[i]
                advantages = all_advantages[i]
                response_mask = all_response_masks[i]

                query = queries[q_idx: q_idx + 1].repeat(args.group_size, 1)
                full_seqs = torch.cat([query, responses], dim=1)
                attn_mask = (full_seqs != processing_class.pad_token_id).long()

                # New policy log-probs (with grad)
                policy_out = model(full_seqs, attention_mask=attn_mask)
                new_logits = policy_out.logits[:, context_length - 1: -1]
                new_logits = new_logits / (args.temperature + 1e-7)
                new_log_probs = F.log_softmax(new_logits, dim=-1)
                new_token_lp = torch.gather(new_log_probs, 2, responses.unsqueeze(-1)).squeeze(-1)
                new_token_lp = new_token_lp * response_mask

                # [DAPO] Token-level clipped surrogate with asymmetric clipping
                pg_loss = self._dapo_policy_loss(
                    new_token_lp, old_token_lp.detach(),
                    advantages.detach(), response_mask,
                    args.clip_range_low, args.clip_range_high,
                )

                # [DAPO] No KL penalty — clipping alone constrains the policy
                accelerator.backward(pg_loss / n_valid)
                total_pg_loss += pg_loss.detach()

                del policy_out, new_logits, new_log_probs, new_token_lp
                torch.cuda.empty_cache()

            optimizer.step()
            self.lr_scheduler.step()
            optimizer.zero_grad()

            # Logging
            with torch.no_grad():
                avg_reward = sum(r.mean().item() for r in all_rewards) / max(n_valid, 1)
                avg_answer_acc = sum(all_answer_accs) / max(n_valid, 1)
                recent_n = min(50, len(accumulate_rewards))
                metrics = {
                    "loss/pg": total_pg_loss.item() / max(n_valid, 1),
                    "reward/mean": avg_reward,
                    "reward/running_avg": sum(accumulate_rewards[-recent_n:]) / max(recent_n, 1),
                    "reward/answer_acc": avg_answer_acc,
                    "reward/answer_acc_running": sum(accumulate_answer_acc[-recent_n:]) / max(recent_n, 1),
                    "dapo/skipped_groups_total": skipped_groups,
                    "dapo/valid_groups_this_step": n_valid,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "episode": self.state.episode,
                }
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1
                self.log(metrics)
                pbar.set_postfix(loss=total_pg_loss.item() / max(n_valid, 1), reward=avg_reward, acc=avg_answer_acc)

                if random.random() < 0.05 and n_valid > 0:
                    sample_resp = processing_class.decode(all_responses[0][0], skip_special_tokens=True)
                    accelerator.print(
                        f"\n[Step {self.state.global_step}] "
                        f"reward={all_rewards[0][0].item():.3f} "
                        f"ans_acc={all_answer_accs[0]:.2f} "
                        f"skipped={skipped_groups}\n"
                        f"Q: {questions[valid_indices[0]][:100]}...\n"
                        f"GT: {answers[valid_indices[0]][:100]}\n"
                        f"A: {sample_resp[:300]}...\n"
                    )

            if args.save_steps and self.state.global_step % args.save_steps == 0:
                self.save_model(os.path.join(args.output_dir, f"checkpoint-{self.state.global_step}"))
                accelerator.print(f"  Saved checkpoint at step {self.state.global_step}")

            del all_responses, all_old_log_probs, all_rewards, all_advantages, all_response_masks
            torch.cuda.empty_cache()
            gc.collect()

        self.save_model(os.path.join(args.output_dir, "final"))
        if self.tb_writer is not None:
            self.tb_writer.close()
        accelerator.print(f"Training complete. Final model saved to {args.output_dir}/final")
