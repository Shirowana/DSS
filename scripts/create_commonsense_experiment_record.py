from __future__ import annotations

import argparse
import csv
import json
import re
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
    parser = argparse.ArgumentParser(description="Create or update a commonsense experiment record.")
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_root", default="/data/home/7250091/date/DSS/experiments/commonsense")
    parser.add_argument("--experiment_md", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_log(log_file: Path) -> dict:
    text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    lines = text.splitlines()
    progress_re = re.compile(
        r"\|\s*(?P<step>\d+)/(?:\s*)?(?P<total>\d+).*?loss=(?P<loss>[-+0-9.eE]+), lr=(?P<lr>[-+0-9.eE]+)"
    )
    health_re = re.compile(
        r"\[dss-health\].*?group=(?P<group>\S+).*?"
        r"active_slots=(?P<active_slots>\d+).*?"
        r"delta_base_ratio=(?P<ratio>[-+0-9.eE]+).*?"
        r"coeff_abs_max=(?P<coeff_abs_max>[-+0-9.eE]+).*?"
        r"coeff_rms=(?P<coeff_rms>[-+0-9.eE]+)"
    )
    elapsed_re = re.compile(r"\[train\] done: .* elapsed=(?P<elapsed>.+)$")

    summary: dict = {
        "steps": "",
        "total_steps": "",
        "final_loss": "",
        "final_lr": "",
        "elapsed": "",
        "bad_status": "ok",
        "health": {},
    }
    for line in lines:
        match = progress_re.search(line)
        if match:
            summary["steps"] = match.group("step")
            summary["total_steps"] = match.group("total")
            summary["final_loss"] = match.group("loss")
            summary["final_lr"] = match.group("lr")
        health = health_re.search(line)
        if health:
            group = health.group("group")
            summary["health"][group] = {
                "active_slots": health.group("active_slots"),
                "delta_base_ratio": health.group("ratio"),
                "coeff_abs_max": health.group("coeff_abs_max"),
                "coeff_rms": health.group("coeff_rms"),
            }
        elapsed = elapsed_re.search(line)
        if elapsed:
            summary["elapsed"] = elapsed.group("elapsed").strip()
        if re.search(r"\b(nan|inf)\b|non-finite|FloatingPointError", line, flags=re.IGNORECASE):
            summary["bad_status"] = "needs_check"
    return summary


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "".join(str(item) for item in value)
    return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(
    *,
    run_name: str,
    training_args: dict,
    log_summary: dict,
    output_dir: Path,
    log_file: Path,
    md_path: Path,
) -> str:
    checkpoints = sorted(path.name for path in output_dir.glob("checkpoint-*") if path.is_dir())
    config_rows = [
        ["model", format_value(training_args.get("model_name"))],
        ["target_modules", format_value(training_args.get("target_modules"))],
        ["shared_basis_path", format_value(training_args.get("shared_basis_path"))],
        ["n_frequency", format_value(training_args.get("n_frequency"))],
        ["candidate_size", format_value(training_args.get("candidate_size"))],
        ["grad_store_steps", format_value(training_args.get("grad_store_steps"))],
        ["lr", format_value(training_args.get("lr"))],
        ["scheduler", format_value(training_args.get("scheduler"))],
        ["warmup_steps", format_value(training_args.get("warmup_steps"))],
        ["num_epochs", format_value(training_args.get("num_epochs"))],
        ["max_steps", format_value(training_args.get("max_steps"))],
        ["stage2_enabled", format_value(training_args.get("stage2_enabled"))],
        ["steady_stage_ratio", format_value(training_args.get("steady_stage_ratio"))],
        ["update_interval", format_value(training_args.get("update_interval"))],
        ["update_counts", format_value(training_args.get("update_counts"))],
    ]
    result_rows = [
        ["steps", format_value(log_summary.get("steps") or log_summary.get("total_steps"))],
        ["total_steps", format_value(log_summary.get("total_steps"))],
        ["epochs", format_value(training_args.get("num_epochs"))],
        ["final_loss", format_value(log_summary.get("final_loss"))],
        ["final_lr", format_value(log_summary.get("final_lr"))],
        ["elapsed", format_value(log_summary.get("elapsed"))],
        ["status", "completed" if log_summary.get("bad_status") == "ok" else "needs_check"],
        ["checkpoints", ", ".join(checkpoints) if checkpoints else "(none found)"],
    ]
    health_rows = []
    for group in ["q_proj", "k_proj", "v_proj"]:
        stats = log_summary.get("health", {}).get(group, {})
        health_rows.append(
            [
                group,
                format_value(stats.get("active_slots")),
                format_value(stats.get("delta_base_ratio")),
                format_value(stats.get("coeff_abs_max")),
                format_value(stats.get("coeff_rms")),
            ]
        )

    eval_placeholder = "\n".join(
        [
            EVAL_BEGIN,
            "## Eval 结果",
            "",
            "等待评测。",
            EVAL_END,
        ]
    )

    return "\n\n".join(
        [
            f"# {run_name}",
            "## 训练配置\n\n" + markdown_table(["key", "value"], config_rows),
            "## 训练结果\n\n" + markdown_table(["key", "value"], result_rows),
            "## 路径\n\n"
            + markdown_table(
                ["key", "path"],
                [
                    ["adapter", f"`{output_dir}`"],
                    ["log", f"`{log_file}`"],
                    ["record", f"`{md_path}`"],
                ],
            ),
            "## DSS health 摘要\n\n"
            + markdown_table(
                ["group", "active_slots", "delta_base_ratio", "coeff_abs_max", "coeff_rms"],
                health_rows,
            ),
            eval_placeholder,
            "",
        ]
    )


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


def upsert_index(index_path: Path, row: dict[str, str]) -> None:
    rows = load_index(index_path)
    replaced = False
    for idx, existing in enumerate(rows):
        if existing.get("run_name") == row["run_name"]:
            merged = dict(existing)
            merged.update({key: value for key, value in row.items() if value != ""})
            rows[idx] = merged
            replaced = True
            break
    if not replaced:
        rows.append(row)
    write_index(index_path, rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_file = Path(args.log_file)
    experiment_root = Path(args.experiment_root)
    run_name = output_dir.name
    md_path = Path(args.experiment_md) if args.experiment_md else experiment_root / f"{run_name}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    training_args = read_json(output_dir / "training_args.json")
    log_summary = parse_log(log_file)
    md_path.write_text(
        build_markdown(
            run_name=run_name,
            training_args=training_args,
            log_summary=log_summary,
            output_dir=output_dir,
            log_file=log_file,
            md_path=md_path,
        ),
        encoding="utf-8",
    )

    row = {
        "run_name": run_name,
        "date": run_name[-15:] if re.search(r"\d{8}_\d{6}$", run_name) else "",
        "model": format_value(training_args.get("model_name")),
        "target_modules": format_value(training_args.get("target_modules")),
        "basis": Path(format_value(training_args.get("shared_basis_path"))).name,
        "lr": format_value(training_args.get("lr")),
        "scheduler": format_value(training_args.get("scheduler")),
        "warmup_steps": format_value(training_args.get("warmup_steps")),
        "steps": format_value(log_summary.get("steps") or log_summary.get("total_steps")),
        "epochs": format_value(training_args.get("num_epochs")),
        "final_loss": format_value(log_summary.get("final_loss")),
        "final_lr": format_value(log_summary.get("final_lr")),
        "elapsed": format_value(log_summary.get("elapsed")),
        "status": "completed" if log_summary.get("bad_status") == "ok" else "needs_check",
        "adapter_path": str(output_dir),
        "log_file": str(log_file),
        "md_path": str(md_path),
        "notes": "auto-created from training log",
    }
    upsert_index(experiment_root / "index.csv", row)
    print(f"[experiment] wrote markdown: {md_path}")
    print(f"[experiment] updated index: {experiment_root / 'index.csv'}")


if __name__ == "__main__":
    main()
