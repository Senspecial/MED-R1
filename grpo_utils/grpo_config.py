import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRPOConfig:
    """
    Configuration for DAPO (Decoupled Alignment Policy Optimization) + PRM.

    Key differences from GRPO:
      1. Asymmetric clipping (clip_range_low < clip_range_high) to encourage exploration
      2. No KL penalty — policy constrained by clipping alone
      3. Dynamic sampling — skip groups where all rewards are identical (no signal)
      4. Overlong reward shaping — truncated responses get discounted reward
      5. Token-level clipped surrogate loss instead of sequence-level
    """

    # Model — default to SFT checkpoint + PRM from train_prm.py
    model_name_or_path: str = "/tmp/sft_stage1/checkpoint-1-3168"
    prm_model_path: str = "/tmp/prm_qwen3_4b"

    # Generation
    group_size: int = 8
    temperature: float = 0.7
    max_new_tokens: int = 1024
    top_p: float = 1.0

    # Reward composition:
    #   total = answer_reward_weight * answer + prm_reward_weight * prm
    #   (bad format → format_penalty)
    prm_agg: str = "min"  # "min", "mean", "last", "weighted_mean"
    prm_reward_weight: float = 0.5
    answer_reward_weight: float = 1.0
    format_penalty: float = -0.5

    # DAPO: asymmetric clipping — larger upper bound to encourage exploration
    clip_range_low: float = 0.2
    clip_range_high: float = 0.28

    # DAPO: dynamic sampling — skip groups with zero reward variance
    dynamic_sampling: bool = True

    # DAPO: overlong shaping — discount factor for truncated responses (no EOS)
    overlong_factor: float = 0.5

    # QLoRA — 4-bit quantized base + LoRA adapters (saves ~80% memory)
    use_qlora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    total_episodes: int = 40000
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.05
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Logging & saving
    output_dir: str = "./ckpts/dapo"
    run_name: str = "dapo_prm"
    save_steps: int = 200
    save_total_limit: int = 2
    eval_steps: int = 50
    logging_steps: int = 10
    report_to: str = "tensorboard"
    logging_dir: str = ""  # default: <output_dir>/tb_logs
    disable_tqdm: bool = False

    # Data
    dataset_path: str = "data/medical_o1_verifiable_problem.json"
    eval_ratio: float = 0.05
    eval_max_num: int = 200

    # System
    seed: int = 42
    num_workers: int = 4
    push_to_hub: bool = False
    save_on_each_node: bool = False
    local_process_index: int = 0
    process_index: int = 0

    @property
    def world_batch_size(self):
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def generation_batch_size(self):
        return self.per_device_batch_size * self.group_size

    @property
    def should_save(self):
        return self.local_process_index == 0
