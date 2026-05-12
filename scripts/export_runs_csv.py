from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
    "hellaswag",
]

CSV_COLUMNS = [
    "run_name",
    "code_version",
    "model_name",
    "target_modules",
    "n_frequency",
    "candidate_size",
    "grad_store_steps",
    "ratio",
    "low",
    "up",
    "threshold_mode",
    "dropout",
    "quantile_lr",
    "quantile_alpha",
    "lr",
    "batch_size_per_gpu",
    "num_gpus",
    "global_batch_size",
    "num_epochs",
    "max_length",
    "val_set_size",
    "eval_steps",
    "save_steps",
    "precision",
    "seed",
    "run_time",
    "train_loss_final",
    "eval_loss_best",
    "best_step",
    "boolq_acc",
    "piqa_acc",
    "social_i_qa_acc",
    "winogrande_acc",
    "ARC-Challenge_acc",
    "ARC-Easy_acc",
    "openbookqa_acc",
    "hellaswag_acc",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact run records to a CSV file.")
    parser.add_argument("--output_root", default="/root/code/DSS/output")
    parser.add_argument("--csv_path", default="/root/code/DSS/experiments/commonsense/runs.csv")
    return parser.parse_args()


def load_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_record(record: dict) -> dict[str, str]:
    training = record.get("training", {})
    evals = record.get("evals", {})
    row: dict[str, str] = {
        "run_name": str(record.get("run_name", "")),
        "code_version": str(record.get("code_version", "")),
        "notes": "；".join(record.get("notes", [])),
    }
    for key in [
        "model_name",
        "target_modules",
        "n_frequency",
        "candidate_size",
        "grad_store_steps",
        "ratio",
        "low",
        "up",
        "threshold_mode",
        "dropout",
        "quantile_lr",
        "quantile_alpha",
        "lr",
        "batch_size_per_gpu",
        "num_gpus",
        "global_batch_size",
        "num_epochs",
        "max_length",
        "val_set_size",
        "eval_steps",
        "save_steps",
        "precision",
        "seed",
        "run_time",
        "train_loss_final",
        "eval_loss_best",
        "best_step",
    ]:
        value = training.get(key, "")
        row[key] = "" if value is None else str(value)

    for dataset in DATASETS:
        eval_row = evals.get(dataset, {})
        row[f"{dataset}_acc"] = ""
        if eval_row.get("is_full_eval"):
            acc = eval_row.get("acc", "")
            row[f"{dataset}_acc"] = f"{float(acc):.4f}" if acc != "" else ""
    return row


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    csv_path = Path(args.csv_path)
    rows = []
    for record_path in sorted(output_root.glob("*/record.json")):
        rows.append(flatten_record(load_record(record_path)))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    print(f"[record] exported csv: {csv_path}")


if __name__ == "__main__":
    main()
