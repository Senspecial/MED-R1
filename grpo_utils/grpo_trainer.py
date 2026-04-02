"""
GRPO (Group Relative Policy Optimization) Trainer with PRM rewards.

Core algorithm:
    For each question in the batch:
        1. Sample K responses from the policy (group_size = K)
        2. Score each response with PRM (per-step) and aggregate
        3. Compute group-relative advantages: A_i = (r_i - mean) / std
        4. Policy gradient: loss = -E[A_i * log pi(y_i|x)]
        5. KL penalty: loss += kl_coef * KL(pi || pi_ref)

This replaces PPO entirely — no value model needed.
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
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    OnlineTrainerState,
    disable_dropout_in_model,
    first_true_indices,
    truncate_response,
)

if is_wandb_available():
    import wandb


# ---------------------------------------------------------------------------
# PRM wrapper for inference during GRPO training
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
        step_end_positions: list of token indices where each step ends
        Returns: list of float scores
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


THINKING_PATTERN = re.compile(r"## Thinking\n\n(.*?)(?=\n\n## Final Response)", re.S)
RESPONSE_PATTERN = re.compile(r"## Final Response\n\n(.*)", re.S)


def extract_thinking_and_response(text):
    thinking_match = THINKING_PATTERN.search(text)
    response_match = RESPONSE_PATTERN.search(text)
    thinking = thinking_match.group(1) if thinking_match else None
    response = response_match.group(1).strip() if response_match else None
    return thinking, response


def compute_prm_reward(
    prm_model, prm_tokenizer, question, response_text, agg="min", device="cuda"
):
    """
    Score a single response with PRM, returning an aggregated scalar reward.
    """
    thinking, final_resp = extract_thinking_and_response(response_text)

    # Format penalty: if the model doesn't follow ## Thinking / ## Final Response
    if thinking is None or final_resp is None:
        return 0.0

    steps = [s.strip() for s in thinking.split("\n\n") if s.strip()]
    if not steps:
        return 0.0

    # Tokenize progressively and find step-end positions
    full_text = f"Question: {question}\n\nReasoning:\n\n" + "\n\n".join(steps)
    full_tokens = prm_tokenizer.encode(full_text, add_special_tokens=True)

    step_end_positions = []
    for i in range(len(steps)):
        partial_text = f"Question: {question}\n\nReasoning:\n\n" + "\n\n".join(steps[: i + 1])
        partial_tokens = prm_tokenizer.encode(partial_text, add_special_tokens=True)
        step_end_positions.append(len(partial_tokens) - 1)

    input_ids = torch.LongTensor([full_tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)

    step_scores = prm_model.score_steps(input_ids, attention_mask, step_end_positions)

    # Aggregate
    if agg == "min":
        return min(step_scores)
    elif agg == "mean":
        return sum(step_scores) / len(step_scores)
    elif agg == "last":
        return step_scores[-1]
    elif agg == "weighted_mean":
        weights = list(range(1, len(step_scores) + 1))
        return sum(s * w for s, w in zip(step_scores, weights)) / sum(weights)
    else:
        return min(step_scores)


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------
class GRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

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
        args.batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * args.world_size

        args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)
        self.local_seed = args.seed + accelerator.process_index * 100003

        for module in [self.policy, self.ref_policy]:
            if module is not None:
                disable_dropout_in_model(module)

        self.model = self.policy
        self.model.config = self.policy.config
        self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        # Callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

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

        if self.is_deepspeed_enabled:
            from trl.trainer.utils import prepare_deepspeed
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_batch_size, False, args.bf16
            )
            self.deepspeed = self.model
            self.model_wrapped = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)

    def get_train_dataloader(self):
        return self.dataloader

    def get_eval_dataloader(self):
        return self.eval_dataloader

    def save_model(self, output_dir=None, _internal_call=False):
        backup = self.model
        self.model = self.policy
        Trainer.save_model(self, output_dir, _internal_call)
        self.model = backup

    def _save(self, output_dir=None, state_dict=None):
        if self.is_deepspeed_enabled and state_dict is not None:
            state_dict = {
                name.removeprefix("policy."): param
                for name, param in state_dict.items()
                if name.startswith("policy.")
            }
        super()._save(output_dir, state_dict)

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

        accelerator.print("=== GRPO Training with PRM ===")
        start_time = time.time()
        model.train()

        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len

        if args.logging_steps and args.logging_steps >= 1:
            self.state.logging_steps = args.logging_steps
        if args.eval_steps and args.eval_steps >= 1:
            self.state.eval_steps = args.eval_steps
        if args.save_steps and args.save_steps >= 1:
            self.state.save_steps = args.save_steps

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        accumulate_rewards = []

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += args.batch_size
            data = next(iter_dataloader)

            with torch.no_grad():
                queries = data["input_ids"].to(device)
                questions = data["question"]
                answers = data["answer"]
                context_length = queries.shape[1]
                B = queries.shape[0]

                # ----- Step 1: Generate K responses per query -----
                all_responses = []      # (B, K, L)
                all_log_probs = []      # (B, K, L)
                all_ref_log_probs = []  # (B, K, L)
                all_rewards = []        # (B, K)

                for q_idx in range(B):
                    query = queries[q_idx: q_idx + 1]  # (1, ctx_len)
                    query_repeated = query.repeat(args.group_size, 1)  # (K, ctx_len)

                    with unwrap_model_for_generation(model, accelerator) as unwrapped:
                        gen_output = unwrapped.generate(
                            query_repeated,
                            generation_config=generation_config,
                            pad_token_id=processing_class.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=False,
                        )
                    query_responses = gen_output.sequences  # (K, ctx_len + gen_len)
                    responses = query_responses[:, context_length:]  # (K, gen_len)

                    # Truncate at stop token
                    if processing_class.eos_token_id is not None:
                        responses = truncate_response(
                            processing_class.eos_token_id,
                            processing_class.pad_token_id,
                            responses,
                        )

                    # Compute log probs for policy
                    full_seqs = torch.cat([query.repeat(args.group_size, 1), responses], dim=1)
                    policy_out = model(full_seqs, attention_mask=(full_seqs != processing_class.pad_token_id).long())
                    logits = policy_out.logits[:, context_length - 1: -1]
                    logits = logits / (args.temperature + 1e-7)
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_log_probs = torch.gather(log_probs, 2, responses.unsqueeze(-1)).squeeze(-1)

                    # Compute ref log probs
                    ref_out = ref_policy(full_seqs, attention_mask=(full_seqs != processing_class.pad_token_id).long())
                    ref_logits = ref_out.logits[:, context_length - 1: -1]
                    ref_logits = ref_logits / (args.temperature + 1e-7)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = torch.gather(ref_log_probs, 2, responses.unsqueeze(-1)).squeeze(-1)

                    # Mask padding
                    response_mask = (responses != processing_class.pad_token_id).float()
                    token_log_probs = token_log_probs * response_mask
                    ref_token_log_probs = ref_token_log_probs * response_mask

                    # ----- Step 2: PRM scoring -----
                    group_rewards = []
                    for k in range(args.group_size):
                        resp_text = processing_class.decode(responses[k], skip_special_tokens=True)
                        reward = compute_prm_reward(
                            self.prm_model, self.prm_tokenizer,
                            questions[q_idx], resp_text,
                            agg=args.prm_agg, device=self.prm_model.backbone.device,
                        )
                        group_rewards.append(reward)

                    # ----- Step 3: Group-relative advantage -----
                    rewards_tensor = torch.tensor(group_rewards, device=device, dtype=torch.float32)
                    mean_r = rewards_tensor.mean()
                    std_r = rewards_tensor.std() + 1e-8
                    advantages = (rewards_tensor - mean_r) / std_r  # (K,)

                    all_responses.append(responses)
                    all_log_probs.append(token_log_probs)
                    all_ref_log_probs.append(ref_token_log_probs)
                    all_rewards.append(rewards_tensor)

                    accumulate_rewards.append(mean_r.item())

                    del policy_out, ref_out, logits, ref_logits, log_probs, ref_log_probs
                    torch.cuda.empty_cache()

            # ----- Step 4: Policy gradient update -----
            total_pg_loss = torch.tensor(0.0, device=device)
            total_kl = torch.tensor(0.0, device=device)
            num_tokens = 0

            for q_idx in range(B):
                responses = all_responses[q_idx]
                token_lp = all_log_probs[q_idx]
                ref_token_lp = all_ref_log_probs[q_idx]
                response_mask = (responses != processing_class.pad_token_id).float()

                rewards_t = all_rewards[q_idx]
                mean_r = rewards_t.mean()
                std_r = rewards_t.std() + 1e-8
                advantages = (rewards_t - mean_r) / std_r

                query = queries[q_idx: q_idx + 1].repeat(args.group_size, 1)
                full_seqs = torch.cat([query, responses], dim=1)

                # Recompute policy logprobs (with grad)
                policy_out = model(full_seqs, attention_mask=(full_seqs != processing_class.pad_token_id).long())
                new_logits = policy_out.logits[:, context_length - 1: -1]
                new_logits = new_logits / (args.temperature + 1e-7)
                new_log_probs = F.log_softmax(new_logits, dim=-1)
                new_token_lp = torch.gather(new_log_probs, 2, responses.unsqueeze(-1)).squeeze(-1)
                new_token_lp = new_token_lp * response_mask

                # Per-sequence log prob
                seq_log_probs = new_token_lp.sum(dim=1)  # (K,)

                # Policy gradient loss: -advantage * log_prob
                pg_loss = -(advantages.detach() * seq_log_probs).mean()

                # KL penalty (token-level)
                kl = (new_token_lp - ref_token_lp.detach()) * response_mask
                kl_loss = args.kl_coef * kl.sum(dim=1).mean()

                loss = pg_loss + kl_loss

                # Accumulate gradients
                accelerator.backward(loss / B)

                total_pg_loss += pg_loss.detach()
                total_kl += kl.sum(dim=1).mean().detach()
                num_tokens += response_mask.sum().item()

                del policy_out, new_logits, new_log_probs, new_token_lp
                torch.cuda.empty_cache()

            # Optimizer step
            optimizer.step()
            self.lr_scheduler.step()
            optimizer.zero_grad()

            # Logging
            with torch.no_grad():
                avg_reward = sum(r.mean().item() for r in all_rewards) / B
                metrics = {
                    "loss/pg": total_pg_loss.item() / B,
                    "loss/kl": total_kl.item() / B,
                    "reward/mean": avg_reward,
                    "reward/running_avg": sum(accumulate_rewards[-50:]) / max(len(accumulate_rewards[-50:]), 1),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "episode": self.state.episode,
                }
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1
                self.log(metrics)

                if random.random() < 0.05:
                    sample_resp = processing_class.decode(all_responses[0][0], skip_special_tokens=True)
                    accelerator.print(
                        f"\n[Step {self.state.global_step}] reward={all_rewards[0][0].item():.3f}\n"
                        f"Q: {questions[0][:100]}...\nA: {sample_resp[:300]}...\n"
                    )

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            del all_responses, all_log_probs, all_ref_log_probs, all_rewards
            torch.cuda.empty_cache()
            gc.collect()

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
