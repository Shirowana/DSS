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

INDEX_COLUMNS = [
    "run_name",
    "date",
    "model",
    "target_modules",
    "basis",
    "lr",
    "scheduler",
    "warmup_steps",
    "steps",
    "epochs",
    "final_loss",
    "final_lr",
    "elapsed",
    "status",
    "eval_status",
    "eval_avg",
    *[f"eval_{dataset}" for dataset in DATASETS],
    "adapter_path",
    "log_file",
    "eval_output_dir",
    "eval_log_file",
    "md_path",
    "notes",
]

EVAL_BEGIN = "<!-- EVAL_RESULTS_BEGIN -->"
EVAL_END = "<!-- EVAL_RESULTS_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append commonsense eval results to an experiment record.")
    parser.add_argument("--experiment_md", required=True)
    parser.add_argument("--eval_output_dir", required=True)
    parser.add_argument("--eval_log_file", default=None)
    parser.add_argument("--experiment_root", default="/root/code/DSS/experiments/commonsense")
    return parser.parse_args()


def load_eval_results(eval_output_dir: Path) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for dataset in DATASETS:
        path = eval_output_dir / f"{dataset}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        total = len(rows)
        correct = sum(1 for row in rows if bool(row.get("flag")))
        accuracy = correct / max(total, 1)
        results[dataset] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
    return results


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_eval_block(
    results: dict[str, dict[str, float | int]],
    eval_output_dir: Path,
    eval_log_file: Path | None,
) -> str:
    rows = []
    accuracies = []
    for dataset in DATASETS:
        stats = results.get(dataset)
        if not stats:
            continue
        accuracy = float(stats["accuracy"])
        accuracies.append(accuracy)
        rows.append(
            [
                dataset,
                str(stats["correct"]),
                str(stats["total"]),
                f"{accuracy:.4f}",
            ]
        )
    avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
    if rows:
        rows.append(["average", "", "", f"{avg:.4f}"])
        content = markdown_table(["dataset", "correct", "total", "accuracy"], rows)
    else:
        content = "未找到 eval JSON 结果。"
    return "\n".join(
        [
            EVAL_BEGIN,
            "## Eval 结果",
            "",
            content,
            "",
            f"- eval_output_dir: `{eval_output_dir}`",
            f"- eval_log_file: `{eval_log_file}`" if eval_log_file is not None else "- eval_log_file: `(not recorded)`",
            EVAL_END,
        ]
    )


def replace_eval_block(markdown: str, eval_block: str) -> str:
    begin = markdown.find(EVAL_BEGIN)
    end = markdown.find(EVAL_END)
    if begin == -1 or end == -1 or end < begin:
        return markdown.rstrip() + "\n\n" + eval_block + "\n"
    end += len(EVAL_END)
    return markdown[:begin].rstrip() + "\n\n" + eval_block + "\n" + markdown[end:].lstrip()


def load_index(index_path: Path) -> list[dict[str, str]]:
    if not index_path.exists():
        return []
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_index(index_path: Path, rows: list[dict[str, str]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in INDEX_COLUMNS})


def update_index(
    index_path: Path,
    run_name: str,
    eval_output_dir: Path,
    eval_log_file: Path | None,
    results: dict[str, dict[str, float | int]],
) -> None:
    rows = load_index(index_path)
    row = None
    for existing in rows:
        if existing.get("run_name") == run_name:
            row = existing
            break
    if row is None:
        row = {"run_name": run_name}
        rows.append(row)

    accuracies = []
    for dataset, stats in results.items():
        accuracy = float(stats["accuracy"])
        accuracies.append(accuracy)
        row[f"eval_{dataset}"] = f"{accuracy:.4f}"
    row["eval_avg"] = f"{(sum(accuracies) / len(accuracies)):.4f}" if accuracies else ""
    row["eval_status"] = "completed" if accuracies else "missing"
    row["eval_output_dir"] = str(eval_output_dir)
    if eval_log_file is not None:
        row["eval_log_file"] = str(eval_log_file)
    write_index(index_path, rows)


def main() -> None:
    args = parse_args()
    experiment_md = Path(args.experiment_md)
    eval_output_dir = Path(args.eval_output_dir)
    eval_log_file = Path(args.eval_log_file) if args.eval_log_file else None
    experiment_root = Path(args.experiment_root)
    run_name = experiment_md.stem

    results = load_eval_results(eval_output_dir)
    markdown = experiment_md.read_text(encoding="utf-8") if experiment_md.exists() else f"# {run_name}\n"
    eval_block = build_eval_block(results, eval_output_dir, eval_log_file)
    experiment_md.parent.mkdir(parents=True, exist_ok=True)
    experiment_md.write_text(replace_eval_block(markdown, eval_block), encoding="utf-8")
    update_index(experiment_root / "index.csv", run_name, eval_output_dir, eval_log_file, results)
    print(f"[experiment] appended eval results to: {experiment_md}")
    print(f"[experiment] updated index: {experiment_root / 'index.csv'}")


if __name__ == "__main__":
    main()
