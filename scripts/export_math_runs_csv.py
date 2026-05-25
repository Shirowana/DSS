from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ["gsm8k", "svamp", "aqua", "mawps"]

CSV_COLUMNS = [
    "run_name",
    "code_version",
    "model_name",
    "method",
    "target_modules",
    "n_frequency",
    "candidate_size",
    "grad_store_steps",
    "ratio",
    "low",
    "up",
    "threshold_mode",
    "score_method",
    "dropout",
    "lr",
    "batch_size_per_gpu",
    "num_gpus",
    "global_batch_size",
    "num_epochs",
    "max_length",
    "max_prompt_length",
    "val_set_size",
    "eval_steps",
    "save_steps",
    "precision",
    "seed",
    "run_time",
    "train_loss_final",
    "eval_loss_best",
    "best_step",
    "gsm8k_acc",
    "svamp_acc",
    "aqua_acc",
    "mawps_acc",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export math run records to CSV.")
    parser.add_argument("--output_root", default="/root/code/DSS/output_math")
    parser.add_argument("--csv_path", default="/root/code/DSS/experiments/math/runs.csv")
    return parser.parse_args()


def load_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_record(record: dict) -> dict[str, str]:
    training = record.get("training", {})
    evals = record.get("evals", {})
    row = {
        "run_name": str(record.get("run_name", "")),
        "code_version": str(record.get("code_version", "")),
        "notes": "；".join(record.get("notes", [])),
    }
    for key in CSV_COLUMNS:
        if key in row or key.endswith("_acc"):
            continue
        value = training.get(key, "")
        row[key] = "" if value is None else str(value)
    for dataset in DATASETS:
        eval_row = evals.get(dataset, {})
        value = ""
        if eval_row.get("is_full_eval"):
            acc = eval_row.get("acc", "")
            value = f"{float(acc):.4f}" if acc != "" else ""
        row[f"{dataset}_acc"] = value
    return row


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    csv_path = Path(args.csv_path)
    rows = [flatten_record(load_record(path)) for path in sorted(output_root.glob("*/record.json"))]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    print(f"[record] exported csv: {csv_path}")


if __name__ == "__main__":
    main()
