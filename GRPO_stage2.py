"""
GRPO + PRM training entry point.

Replaces RL_stage2.py (PPO + ORM) with:
  - GRPO (no value model needed)
  - PRM (per-step reward instead of outcome-only reward)

Usage:
    accelerate launch \
        --num_processes 8 \
        --num_machines 1 \
        --machine_rank 0 \
        --config_file ./configs/deepspeed_zero3.yaml \
        --deepspeed_multinode_launcher standard \
        GRPO_stage2.py \
        --model_name_or_path [FreedomIntelligence/HuatuoGPT-o1-8B] \
        --prm_model_path [ckpts/prm_3b/final] \
        --dataset_path [data/medical_o1_verifiable_problem.json] \
        --output_dir ./ckpts/grpo \
        --group_size 8 \
        --temperature 0.7 \
        --kl_coef 0.04 \
        --learning_rate 5e-7 \
        --total_episodes 20000 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --save_steps 50 \
        --prm_agg min
"""

import os
import json
import random

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from grpo_utils.grpo_config import GRPOConfig
from grpo_utils.grpo_trainer import GRPOTrainer, PRMScorer

os.environ["WANDB_MODE"] = "offline"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


# ---------------------------------------------------------------------------
# Dataset (same interface as the PPO version)
# ---------------------------------------------------------------------------
class GRPODataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=1000, debug=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug

        self.data = [
            {"question": d["Open-ended Verifiable Question"], "answer": d["Ground-True Answer"]}
            for d in data
            if d.get("Open-ended Verifiable Question") and d.get("Ground-True Answer")
        ]
        print(f"Dataset: {len(data)} -> {len(self.data)} (after filtering)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _tokenize(self, da):
        message = [{"role": "user", "content": da["question"]}]
        prompt = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(prompt, padding=False, truncation=False, add_special_tokens=False)
        da["input_ids"] = tokens["input_ids"]
        return da

    def collate_fn(self, batch):
        data = [self._tokenize(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]

        max_len = min(max(len(x) for x in input_ids), self.max_length)
        input_ids = [
            [self.tokenizer.pad_token_id] * (max_len - len(item)) + item[:max_len]
            for item in input_ids
        ]

        if self.debug > 0:
            print("[input_ids]", self.tokenizer.decode(input_ids[-1]))
            print("[question]", questions[-1])
            print("[answer]", answers[-1])
            self.debug -= 1

        return {
            "input_ids": torch.LongTensor(input_ids),
            "question": questions,
            "answer": answers,
        }


# ---------------------------------------------------------------------------
# Load PRM
# ---------------------------------------------------------------------------
def load_prm(prm_path, device="cuda:0"):
    """Load a trained PRM (backbone + reward_head)."""
    from transformers import AutoModelForCausalLM

    backbone = AutoModelForCausalLM.from_pretrained(
        prm_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(device).eval()

    reward_head = nn.Linear(backbone.config.hidden_size, 1)
    head_path = os.path.join(prm_path, "reward_head.pt")
    if os.path.exists(head_path):
        reward_head.load_state_dict(torch.load(head_path, map_location=device))
    reward_head = reward_head.to(device=device, dtype=torch.bfloat16)

    prm_tokenizer = AutoTokenizer.from_pretrained(prm_path)
    prm_model = PRMScorer(backbone, reward_head).eval()

    print(f"Loaded PRM from {prm_path}")
    return prm_model, prm_tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(GRPOConfig)
    (config,) = parser.parse_args_into_dataclasses()

    output_dir = config.output_dir
    if config.run_name not in output_dir:
        output_dir = os.path.join(output_dir, config.run_name)
        config.output_dir = output_dir

    # Policy & ref policy
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    policy = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, attn_implementation="flash_attention_2"
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, attn_implementation="flash_attention_2"
    )

    # Handle pad token
    if "<|eot_id|>" in tokenizer.vocab:
        assert "<|end_of_text|>" in tokenizer.vocab
        tokenizer.pad_token = "<|end_of_text|>"
        tokenizer.pad_token_id = tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    assert tokenizer.pad_token_id != tokenizer.eos_token_id

    # PRM
    prm_device = "cuda:0"
    prm_model, prm_tokenizer = load_prm(config.prm_model_path, device=prm_device)

    # Data
    with open(config.dataset_path) as f:
        data = json.load(f)
    random.shuffle(data)

    eval_num = min(int(len(data) * config.eval_ratio), config.eval_max_num)
    train_dataset = GRPODataset(data[eval_num:], tokenizer, debug=1)
    eval_dataset = GRPODataset(data[:eval_num], tokenizer)

    # Trainer
    trainer = GRPOTrainer(
        config=config,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        prm_model=prm_model,
        prm_tokenizer=prm_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
    )
    trainer.train()
