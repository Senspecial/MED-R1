import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRPOConfig:
    """Configuration for GRPO + PRM training."""

    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    prm_model_path: str = "ckpts/prm_3b/final"

    # Generation
    group_size: int = 8
    temperature: float = 0.7
    max_new_tokens: int = 2048
    top_p: float = 1.0

    # PRM reward
    prm_agg: str = "min"  # "min", "mean", "last", "weighted_mean"
    format_reward_weight: float = 0.1

    # KL
    kl_coef: float = 0.04
    clip_range: float = 0.2

    # Training
    learning_rate: float = 5e-7
    total_episodes: int = 20000
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging & saving
    output_dir: str = "./ckpts/grpo"
    run_name: str = "grpo_prm"
    save_steps: int = 50
    save_total_limit: int = 3
    eval_steps: int = 50
    logging_steps: int = 10
    report_to: str = "wandb"

    # Data
    dataset_path: str = "data/medical_o1_verifiable_problem.json"
    eval_ratio: float = 0.05
    eval_max_num: int = 200

    # System
    seed: int = 42
    num_workers: int = 4

    @property
    def world_batch_size(self):
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def generation_batch_size(self):
        return self.per_device_batch_size * self.group_size
