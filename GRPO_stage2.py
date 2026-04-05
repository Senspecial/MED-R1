"""
DAPO + PRM training entry point (Stage 2).

DAPO (Decoupled Alignment Policy Optimization) improves on GRPO with:
  - Asymmetric clipping (clip_low=0.2, clip_high=0.28) to encourage exploration
  - No KL penalty — clipping alone constrains the policy
  - Dynamic sampling — skip groups with identical rewards (no learning signal)
  - Overlong reward shaping — truncated responses get discounted reward
  - Token-level clipped surrogate loss instead of sequence-level

Prerequisites:
  1. SFT model from SFT_stage1.py   → ckpts/sft_stage1/checkpoint-1-3168
  2. PRM  model from train_prm.py   → ckpts/prm_qwen3_4b/final/

Usage:
    # Step 1: copy models to local disk for faster I/O
    cp -r /apdcephfs_cq11/share_303693288/hunyuan/jiansensong/MED-R1/ckpts/sft_stage1/checkpoint-1-3168/tfmr /tmp/sft_model
    cp -r ckpts/prm_qwen3_4b/final /tmp/prm_qwen3_4b

    # Step 2: launch training
    accelerate launch \
        --config_file configs/deepspeed_zero2.yaml \
        --num_processes 8 \
        --num_machines 1 \
        --machine_rank 0 \
        --deepspeed_multinode_launcher standard \
        GRPO_stage2.py \
        --model_name_or_path /tmp/sft_model \
        --prm_model_path /tmp/prm_qwen3_4b \
        --dataset_path data/medical_o1_verifiable_problem.json \
        --output_dir ./ckpts/dapo \
        --group_size 8 \
        --temperature 0.7 \
        --clip_range_low 0.2 \
        --clip_range_high 0.28 \
        --dynamic_sampling True \
        --learning_rate 5e-7 \
        --total_episodes 20000 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --save_steps 50 \
        --prm_agg min

    # With QLoRA (much less memory — single GPU possible):
    accelerate launch \
        --config_file configs/deepspeed_zero2.yaml \
        --num_processes 8 \
        GRPO_stage2.py \
        --model_name_or_path /tmp/sft_model \
        --prm_model_path /tmp/prm_qwen3_4b \
        --use_qlora True \
        --lora_r 64 \
        --lora_alpha 16 \
        --output_dir ./ckpts/dapo_qlora \
        --group_size 8 \
        --per_device_batch_size 1
"""

import os
import json
import random

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from grpo_utils.grpo_config import GRPOConfig
from grpo_utils.grpo_trainer import GRPOTrainer, PRMScorer

os.environ["WANDB_MODE"] = "offline"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


# ---------------------------------------------------------------------------
# Dataset
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
    backbone = AutoModelForCausalLM.from_pretrained(
        prm_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    reward_head = nn.Linear(backbone.config.hidden_size, 1)
    head_path = os.path.join(prm_path, "reward_head.pt")
    if os.path.exists(head_path):
        reward_head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
    reward_head = reward_head.to(device=device, dtype=torch.bfloat16)

    prm_tokenizer = AutoTokenizer.from_pretrained(prm_path)
    prm_model = PRMScorer(backbone, reward_head).eval()

    print(f"Loaded PRM from {prm_path}")
    return prm_model, prm_tokenizer


# ---------------------------------------------------------------------------
# Pad token setup — works for Qwen3, Llama, and other models
# ---------------------------------------------------------------------------
def setup_pad_token(tokenizer):
    """Ensure pad_token_id is set and differs from eos_token_id."""
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        return

    # Model-specific candidates that are NOT the eos token
    candidates = ["<|endoftext|>", "<|end_of_text|>"]
    vocab = tokenizer.get_vocab()
    for candidate in candidates:
        if candidate in vocab:
            cid = tokenizer.convert_tokens_to_ids(candidate)
            if cid != tokenizer.eos_token_id:
                tokenizer.pad_token = candidate
                tokenizer.pad_token_id = cid
                return

    # Fallback: add a dedicated pad token
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


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

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    if config.use_qlora:
        # QLoRA: 4-bit quantized base + LoRA adapters
        # Memory: ~5GB (vs ~32GB for two bf16 copies)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        policy = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2",
        )
        policy = prepare_model_for_kbit_training(
            policy, use_gradient_checkpointing=config.gradient_checkpointing
        )

        target_modules = config.lora_target_modules
        if "," in target_modules:
            target_modules = [m.strip() for m in target_modules.split(",")]

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_config)
        policy.print_trainable_parameters()

        # DAPO has no KL penalty, so ref_policy is not needed
        ref_policy = None
        # gradient_checkpointing already handled by prepare_model_for_kbit_training
        config.gradient_checkpointing = False
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        if config.lora_r > 0:
            target_modules = config.lora_target_modules
            if "," in target_modules:
                target_modules = [m.strip() for m in target_modules.split(",")]
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            policy = get_peft_model(policy, lora_config)
            policy.print_trainable_parameters()

        # DAPO has no KL penalty, ref_policy not needed
        ref_policy = None

    setup_pad_token(tokenizer)
    config.stop_token_id = tokenizer.eos_token_id
    print(f"pad_token={tokenizer.pad_token!r} (id={tokenizer.pad_token_id}), "
          f"eos_token={tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    # PRM — each process loads onto its own GPU via LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    prm_device = f"cuda:{local_rank}"
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
