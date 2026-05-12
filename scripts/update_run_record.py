from __future__ import annotations

import argparse
import json
import re
import subprocess
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a compact experiment record JSON.")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--eval_output_dir", default=None)
    parser.add_argument("--dataset", default=None, choices=DATASETS)
    parser.add_argument("--is_full_eval", type=int, default=None, choices=[0, 1])
    parser.add_argument("--data_dir", default="/root/datasets/evaluate")
    parser.add_argument("--project_root", default="/root/code/DSS")
    parser.add_argument("--notes", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def append_note(record: dict, note: str | None) -> None:
    if not note:
        return
    note = note.strip()
    if not note:
        return
    notes = record.setdefault("notes", [])
    if note not in notes:
        notes.append(note)


def git_code_version(project_root: Path) -> str:
    branch = (
        subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    )
    commit = subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(project_root), "status", "--short"], text=True).strip()
    suffix = "+dirty" if dirty else ""
    return f"{branch}@{commit}{suffix}"


def parse_training_log(log_file: Path) -> dict:
    text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    lines = text.splitlines()
    result: dict[str, str] = {"training_completed": ""}
    for line in lines:
        if line.startswith("NUM_GPUS="):
            result["num_gpus"] = line.split("=", 1)[1].strip()
        elif line.startswith("elapsed="):
            result["run_time"] = line.split("=", 1)[1].strip()
            result["training_completed"] = "1"
    train_summary_re = re.compile(
        r"\{'train_runtime': [^}]*'train_loss': (?P<train_loss>[-+0-9.eE]+), 'epoch': (?P<epoch>[-+0-9.eE]+)\}"
    )
    if result.get("training_completed") == "1":
        for line in reversed(lines):
            match = train_summary_re.search(line)
            if match:
                result["train_loss_final"] = match.group("train_loss")
                break
    return result


def parse_trainer_state(output_dir: Path) -> dict:
    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    state_path = checkpoint_dirs[-1] / "trainer_state.json" if checkpoint_dirs else None
    state = load_json(state_path) if state_path else {}
    best_model_checkpoint = state.get("best_model_checkpoint", "")
    best_step = state.get("best_global_step", "")
    if not best_step and best_model_checkpoint:
        match = re.search(r"checkpoint-(\d+)$", best_model_checkpoint)
        if match:
            best_step = int(match.group(1))
    final_checkpoint = checkpoint_dirs[-1].name if checkpoint_dirs else ""
    return {
        "eval_loss_best": state.get("best_metric", ""),
        "best_step": best_step,
        "best_model_checkpoint": best_model_checkpoint,
        "final_checkpoint": final_checkpoint,
    }


def expected_total(data_dir: Path, dataset: str) -> int | None:
    path = data_dir / dataset / "test.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return len(json.load(handle))


def update_training_record(record: dict, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    training_args = load_json(output_dir / "training_args.json")
    log_summary = parse_training_log(Path(args.log_file)) if args.log_file else {}
    trainer_state = parse_trainer_state(output_dir)

    num_gpus = int(log_summary.get("num_gpus") or 1)
    batch_size_per_gpu = int(training_args.get("batch_size") or 0)
    grad_accum = int(training_args.get("gradient_accumulation_steps") or 1)

    record["run_name"] = args.run_name
    record["code_version"] = git_code_version(Path(args.project_root))
    training_completed = log_summary.get("training_completed") == "1"
    record["training"] = {
        "model_name": training_args.get("model_name", ""),
        "target_modules": training_args.get("target_modules", ""),
        "n_frequency": training_args.get("n_frequency", ""),
        "candidate_size": training_args.get("candidate_size", ""),
        "grad_store_steps": training_args.get("grad_store_steps", ""),
        "ratio": training_args.get("ratio", ""),
        "low": training_args.get("low", ""),
        "up": training_args.get("up", ""),
        "threshold_mode": training_args.get("threshold_mode", ""),
        "dropout": training_args.get("dropout", ""),
        "quantile_lr": training_args.get("quantile_lr", ""),
        "quantile_alpha": training_args.get("quantile_alpha", ""),
        "lr": training_args.get("lr", ""),
        "batch_size_per_gpu": batch_size_per_gpu,
        "num_gpus": num_gpus,
        "global_batch_size": batch_size_per_gpu * grad_accum * num_gpus,
        "num_epochs": training_args.get("num_epochs", ""),
        "max_length": training_args.get("max_length", ""),
        "val_set_size": training_args.get("val_set_size", ""),
        "eval_steps": training_args.get("eval_steps", ""),
        "save_steps": training_args.get("save_steps", ""),
        "precision": training_args.get("precision", ""),
        "seed": training_args.get("seed", ""),
        "run_time": log_summary.get("run_time", "") if training_completed else "",
        "train_loss_final": log_summary.get("train_loss_final", "") if training_completed else "",
        "eval_loss_best": trainer_state.get("eval_loss_best", "") if training_completed else "",
        "best_step": trainer_state.get("best_step", "") if training_completed else "",
    }
    append_note(record, args.notes)


def update_eval_record(record: dict, args: argparse.Namespace) -> None:
    eval_output_dir = Path(args.eval_output_dir)
    summary = load_json(eval_output_dir / "summary.json")
    datasets = summary.get("datasets", {})
    data_dir = Path(args.data_dir)
    evals = record.setdefault("evals", {})

    dataset_names = [args.dataset] if args.dataset else [name for name in DATASETS if name in datasets]
    for dataset in dataset_names:
        stats = datasets.get(dataset)
        if not stats:
            continue
        total = int(stats.get("total", 0))
        expected = expected_total(data_dir, dataset)
        explicit_full = args.is_full_eval == 1
        inferred_full = args.is_full_eval is None and expected is not None and total == expected
        is_full_eval = explicit_full or inferred_full
        if is_full_eval and expected is not None and total != expected:
            is_full_eval = False

        if is_full_eval:
            evals[dataset] = {
                "acc": float(stats.get("accuracy", 0.0)),
                "total": total,
                "is_full_eval": True,
            }
        else:
            append_note(record, f"中文：{dataset} 仅做过 {total} 条非完整评测，未记入正式结果")

    append_note(record, args.notes)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    record_path = output_dir / "record.json"
    record = load_json(record_path)

    if args.eval_output_dir:
        update_eval_record(record, args)
    else:
        update_training_record(record, args)

    save_json(record_path, record)
    print(f"[record] updated: {record_path}")


if __name__ == "__main__":
    main()
