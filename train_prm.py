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
    nohup accelerate launch --config_file configs/deepspeed_zero2.yaml \
        --num_processes 8 train_prm.py \
        --model_path /tmp/Qwen3-4B \
        --data_path data/prm_train_data.json \
        --output_dir ckpts/prm_qwen3_4b \
        --use_lora \
        --epochs 2 \
        --lr 2e-5 \
        > prm_train.log 2>&1 &
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
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_cosine_schedule_with_warmup
from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

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
        self.reward_head = nn.Linear(hidden_size, 1, dtype=backbone.dtype)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids, attention_mask, score_positions):
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            score_positions: (B,) index of the token where we extract the reward
        Returns:
            logits: (B,) raw logits (apply sigmoid for probabilities)
        """
        base_model = self.backbone.model if not hasattr(self.backbone, 'base_model') \
            else self.backbone.base_model.model.model
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (B, L, H)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        step_hidden = hidden[batch_idx, score_positions]  # (B, H)
        logits = self.reward_head(step_hidden).squeeze(-1)  # (B,)
        return logits


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

    def get_lengths(self):
        """Estimate token lengths for all samples (char-based approximation)."""
        lengths = []
        for item in self.data:
            text = f"Question: {item['question']}\n\nReasoning:\n\n{STEP_SEP.join(item['prefix_steps'])}"
            lengths.append(len(text) // 3)  # rough char-to-token ratio
        return lengths

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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=2300)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--logging_dir", type=str, default=None,
                        help="TensorBoard log dir (default: <output_dir>/tb_logs)")
    args = parser.parse_args()

    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.output_dir, "tb_logs")

    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.logging_dir,
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
    if args.use_lora:
        assert PEFT_AVAILABLE, "pip install peft  is required for --use_lora"
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        backbone = get_peft_model(backbone, lora_config)
        trainable, total = backbone.get_nb_trainable_parameters()
        accelerator.print(f"LoRA enabled: trainable {trainable:,} / {total:,} params "
                          f"({100*trainable/total:.2f}%)")

    model = ProcessRewardModel(backbone)

    with open(args.data_path) as f:
        all_data = json.load(f)

    questions_to_samples = {}
    for item in all_data:
        q = item["question"]
        questions_to_samples.setdefault(q, []).append(item)
    unique_questions = list(questions_to_samples.keys())
    random.shuffle(unique_questions)

    eval_q_count = max(int(len(unique_questions) * args.eval_ratio), 10)
    eval_questions = set(unique_questions[:eval_q_count])
    eval_data = [s for q in eval_questions for s in questions_to_samples[q]]
    train_data = [s for q in unique_questions[eval_q_count:] for s in questions_to_samples[q]]
    random.shuffle(train_data)
    random.shuffle(eval_data)
    accelerator.print(f"Questions: {len(unique_questions)} total, {eval_q_count} eval, "
                      f"{len(unique_questions)-eval_q_count} train")
    accelerator.print(f"Samples: Train {len(train_data)}, Eval {len(eval_data)}")
    pos_train = sum(1 for d in train_data if d["score"] > 0.5)
    pos_eval = sum(1 for d in eval_data if d["score"] > 0.5)
    accelerator.print(f"Train positive ratio: {pos_train}/{len(train_data)} "
                      f"({100*pos_train/len(train_data):.1f}%)")
    accelerator.print(f"Eval  positive ratio: {pos_eval}/{len(eval_data)} "
                      f"({100*pos_eval/len(eval_data):.1f}%)")

    train_dataset = PRMDataset(train_data, tokenizer, args.max_seq_len)
    eval_dataset = PRMDataset(eval_data, tokenizer, args.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn, drop_last=True,
        num_workers=4, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=4, pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(f"Optimizer trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    accelerator.init_trackers("prm_training", config=vars(args))

    loss_fn = nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=not accelerator.is_main_process)
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                preds = model(
                    batch["input_ids"], batch["attention_mask"], batch["score_positions"]
                )
                loss = loss_fn(preds, batch["labels"].to(preds.dtype))
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                avg_loss = total_loss / (step + 1)
                cur_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}",
                                 lr=f"{cur_lr:.2e}",
                                 step=global_step)
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/loss_avg": avg_loss,
                    "train/lr": cur_lr,
                }, step=global_step)
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
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                logits = model(
                    batch["input_ids"], batch["attention_mask"], batch["score_positions"]
                )
                loss = loss_fn(logits, batch["labels"].to(logits.dtype))
                eval_loss += loss.item()
                probs = torch.sigmoid(logits)
                pred_binary = (probs > 0.5).float()
                label_binary = (batch["labels"] > 0.5).float()
                eval_correct += (pred_binary == label_binary).sum().item()
                eval_total += len(logits)
                all_probs.extend(probs.float().cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        avg_eval_loss = eval_loss / max(len(eval_loader), 1)
        accuracy = eval_correct / max(eval_total, 1)
        auc_str = ""
        auc_val = 0.0
        try:
            from sklearn.metrics import roc_auc_score
            binary_labels = [1 if l > 0.5 else 0 for l in all_labels]
            if len(set(binary_labels)) > 1:
                auc_val = roc_auc_score(binary_labels, all_probs)
                auc_str = f" AUC: {auc_val:.4f}"
        except ImportError:
            pass
        accelerator.print(
            f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f} "
            f"Accuracy: {accuracy:.4f}{auc_str}"
        )
        accelerator.log({
            "eval/loss": avg_eval_loss,
            "eval/accuracy": accuracy,
            "eval/auc": auc_val,
        }, step=global_step)

    accelerator.end_training()
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


def merge_lora():
    """Merge LoRA adapter weights into base model and save full PRM checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the original base model (e.g. /tmp/Qwen3-4B)")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint (e.g. ckpts/prm_qwen3_4b/step_2300)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save merged full model")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model_path}")
    from peft import PeftModel
    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter from {args.lora_checkpoint}")
    backbone = PeftModel.from_pretrained(backbone, args.lora_checkpoint)

    print("Merging LoRA weights into base model...")
    backbone = backbone.merge_and_unload()

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving merged model to {args.output_path}")
    backbone.save_pretrained(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.lora_checkpoint)
    tokenizer.save_pretrained(args.output_path)

    head_src = os.path.join(args.lora_checkpoint, "reward_head.pt")
    if os.path.exists(head_src):
        import shutil
        shutil.copy2(head_src, os.path.join(args.output_path, "reward_head.pt"))
        print("Copied reward_head.pt")

    print("Done! Merged PRM saved to", args.output_path)


if __name__ == "__main__":
    import sys
    if "--merge_lora" in sys.argv:
        sys.argv.remove("--merge_lora")
        merge_lora()
    else:
        main()
