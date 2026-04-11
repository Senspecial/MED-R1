"""
Convert MED-R1 medical_o1_verifiable_problem.json to verl-compatible parquet files.

verl 0.5.0 expects parquet with columns:
  - data_source (str): dataset identifier
  - prompt (list[dict]): chat messages [{"role": "user", "content": ...}]
  - reward_model (dict): must contain "ground_truth" key
  - extra_info (dict): forwarded to compute_score as extra_info arg

Usage:
    python convert_data_to_parquet.py \
        --input data/medical_o1_verifiable_problem.json \
        --output_dir data/verl \
        --eval_ratio 0.05 \
        --eval_max_num 200
"""

import argparse
import json
import os
import random

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/medical_o1_verifiable_problem.json")
    parser.add_argument("--output_dir", default="data/verl")
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--eval_max_num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    rows = []
    for d in data:
        q = d.get("Open-ended Verifiable Question")
        a = d.get("Ground-True Answer")
        if not q or not a:
            continue
        rows.append({
            "data_source": "medical_o1",
            "prompt": [{"role": "user", "content": q}],
            "reward_model": {"ground_truth": a},
            "extra_info": {"question": q},
        })

    n_eval = min(int(len(rows) * args.eval_ratio), args.eval_max_num)
    train_rows = rows[n_eval:]
    eval_rows = rows[:n_eval]

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "medical_o1_train.parquet")
    eval_path = os.path.join(args.output_dir, "medical_o1_eval.parquet")

    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(eval_rows).to_parquet(eval_path)

    print(f"Train: {len(train_rows)} samples -> {train_path}")
    print(f"Eval:  {len(eval_rows)} samples -> {eval_path}")


if __name__ == "__main__":
    main()
