"""
Train a Process Reward Model (PRM).

The PRM predicts a scalar quality score for each reasoning step. We place
a reward head on top of a causal LM backbone (e.g. Qwen3-4B) and train
it to predict the Monte-Carlo step scores produced by construct_prm_data.py.

Input format (from construct_prm_data.py --output_path):
[
  {
    "question": "...",
    "prefix_steps": ["step0 text", "step1 text", ...],
    "score": 0.875
  },
  ...
]

Usage:
    accelerate launch --config_file configs/deepspeed_zero2.yaml \
        --num_processes 8 train_prm.py \
        --model_path /tmp/Qwen3-4B \
        --data_path data/prm_train_data.json \
        --output_dir ckpts/prm_qwen3_4b \
        --epochs 3 \
        --lr 2e-5 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --max_seq_len 4096
"""

import os
import json
import math
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


# ---------------------------------------------------------------------------
# PRM model: backbone + scalar reward head
# ---------------------------------------------------------------------------
class ProcessRewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids, attention_mask, score_positions):
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            score_positions: (B,) index of the token where we extract the reward
        Returns:
            scores: (B,) predicted step scores in [0, 1]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # (B, L, H)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        step_hidden = hidden[batch_idx, score_positions]  # (B, H)
        scores = self.reward_head(step_hidden).squeeze(-1)  # (B,)
        return torch.sigmoid(scores)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
STEP_SEP = "\n\n"

class PRMDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids_list = []
        score_positions = []
        labels = []

        for item in batch:
            q = item["question"]
            steps = item["prefix_steps"]
            score = item["score"]

            text = f"Question: {q}\n\nReasoning:\n\n{STEP_SEP.join(steps)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            tokens = tokens[-self.max_seq_len:]

            input_ids_list.append(tokens)
            score_positions.append(len(tokens) - 1)
            labels.append(score)

        max_len = max(len(t) for t in input_ids_list)
        padded_ids = []
        attention_masks = []
        adjusted_positions = []

        for i, tokens in enumerate(input_ids_list):
            pad_len = max_len - len(tokens)
            padded_ids.append([self.tokenizer.pad_token_id] * pad_len + tokens)
            attention_masks.append([0] * pad_len + [1] * len(tokens))
            adjusted_positions.append(score_positions[i] + pad_len)

        return {
            "input_ids": torch.LongTensor(padded_ids),
            "attention_mask": torch.LongTensor(attention_masks),
            "score_positions": torch.LongTensor(adjusted_positions),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="ckpts/prm_qwen3_4b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Find a pad token that differs from eos (needed for Qwen3, Llama, etc.)
        for candidate in ["<|endoftext|>", "<|end_of_text|>"]:
            vocab = tokenizer.get_vocab()
            if candidate in vocab:
                cid = tokenizer.convert_tokens_to_ids(candidate)
                if cid != tokenizer.eos_token_id:
                    tokenizer.pad_token = candidate
                    tokenizer.pad_token_id = cid
                    break
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    accelerator.print(f"pad_token={tokenizer.pad_token!r} (id={tokenizer.pad_token_id}), "
                      f"eos_token={tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = ProcessRewardModel(backbone)

    with open(args.data_path) as f:
        all_data = json.load(f)
    random.shuffle(all_data)

    eval_size = max(int(len(all_data) * args.eval_ratio), 100)
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]
    accelerator.print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    train_dataset = PRMDataset(train_data, tokenizer, args.max_seq_len)
    eval_dataset = PRMDataset(eval_data, tokenizer, args.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=eval_dataset.collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    loss_fn = nn.MSELoss()
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                preds = model(
                    batch["input_ids"], batch["attention_mask"], batch["score_positions"]
                )
                loss = loss_fn(preds, batch["labels"])
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                if global_step % 50 == 0:
                    avg_loss = total_loss / (step + 1)
                    accelerator.print(
                        f"Epoch {epoch+1} Step {global_step} Loss: {avg_loss:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"step_{global_step}")
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped.backbone.save_pretrained(save_path)
                        torch.save(unwrapped.reward_head.state_dict(),
                                   os.path.join(save_path, "reward_head.pt"))
                        tokenizer.save_pretrained(save_path)

        # Eval
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in eval_loader:
                preds = model(
                    batch["input_ids"], batch["attention_mask"], batch["score_positions"]
                )
                loss = loss_fn(preds, batch["labels"])
                eval_loss += loss.item()
                pred_binary = (preds > 0.5).float()
                label_binary = (batch["labels"] > 0.5).float()
                eval_correct += (pred_binary == label_binary).sum().item()
                eval_total += len(preds)

        avg_eval_loss = eval_loss / max(len(eval_loader), 1)
        accuracy = eval_correct / max(eval_total, 1)
        accelerator.print(
            f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f} Accuracy: {accuracy:.4f}"
        )

    # Final save
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        unwrapped.backbone.save_pretrained(final_path)
        torch.save(unwrapped.reward_head.state_dict(),
                   os.path.join(final_path, "reward_head.pt"))
        tokenizer.save_pretrained(final_path)
        accelerator.print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
