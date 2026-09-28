#!/usr/bin/env python3
"""Run task-specific evaluation workers on two GPUs and append one summary block."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "commonsense": {
        "module": "eval_commonsense.py",
        "counts": {
            "boolq": 3270, "piqa": 1838, "social_i_qa": 1954,
            "hellaswag": 10042, "winogrande": 1267,
            "ARC-Challenge": 1172, "ARC-Easy": 2376, "openbookqa": 500,
        },
    },
    "math": {
        "module": "eval_math.py",
        "counts": {"gsm8k": 1319, "aqua": 254, "mawps": 238, "svamp": 1000},
    },
    "code": {
        "module": "eval_code.py",
        "counts": {"humaneval": 164, "mbpp": 378},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--train_log", type=Path)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def balanced_partition(counts: dict[str, int], workers: int) -> list[list[str]]:
    groups: list[list[str]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        index = min(range(workers), key=loads.__getitem__)
        groups[index].append(name)
        loads[index] += count
    return groups


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_fraction(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def aggregate(task: str, worker_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for payload in worker_summaries:
        datasets.update(payload.get("datasets", {}))
    expected = TASKS[task]["counts"]
    missing = sorted(set(expected) - set(datasets))
    if missing:
        raise RuntimeError(f"Missing completed datasets: {', '.join(missing)}")
    summary: dict[str, Any] = {"task": task, "datasets": datasets}
    for name in expected:
        if int(datasets[name].get("total", expected[name])) != expected[name]:
            raise RuntimeError(
                f"{name}: aggregate total {datasets[name].get('total')} != expected {expected[name]}"
            )
    if task == "commonsense":
        values = [as_fraction(float(datasets[name]["accuracy"])) for name in expected]
        summary["average_accuracy"] = sum(values) / len(values)
        summary["metric"] = "unweighted_mean_of_eight_task_accuracies"
    elif task == "math":
        values = [as_fraction(float(datasets[name]["accuracy"])) for name in expected]
        summary["average_accuracy"] = sum(values) / len(values)
        summary["metric"] = "unweighted_mean_of_four_task_accuracies"
    else:
        keys = ("HumanEval", "HumanEval+", "MBPP", "MBPP+")
        aliases = {
            "humaneval": ("HumanEval", "HumanEval+"),
            "mbpp": ("MBPP", "MBPP+"),
        }
        for name, value in datasets.items():
            if name in aliases:
                base_key, plus_key = aliases[name]
                summary[base_key] = as_fraction(float(value["base_pass_at_1"]))
                summary[plus_key] = as_fraction(float(value["plus_pass_at_1"]))
        for payload in worker_summaries:
            summary.update(
                {key: as_fraction(float(payload[key])) for key in keys if key in payload}
            )
        summary["average_accuracy"] = sum(float(summary[key]) for key in keys) / len(keys)
        summary["metric"] = "unweighted_mean_of_four_pass_at_1_metrics"
    return summary


def append_summary(path: Path, summary: dict[str, Any], assignments: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n========== EVALUATION SUMMARY ==========\n")
        for worker, names in enumerate(assignments):
            handle.write(f"[summary] gpu{worker} datasets={','.join(names)}\n")
        for name in TASKS[summary["task"]]["counts"]:
            value = summary["datasets"][name]
            if "accuracy" in value:
                accuracy = as_fraction(float(value["accuracy"]))
                handle.write(f"[summary] {name:<14} accuracy={accuracy:.4f}\n")
            elif "plus_pass_at_1" in value:
                handle.write(
                    f"[summary] {name:<14} base={as_fraction(float(value['base_pass_at_1'])):.4f} "
                    f"plus={as_fraction(float(value['plus_pass_at_1'])):.4f}\n"
                )
        handle.write(f"[summary] average_accuracy={summary['average_accuracy']:.4f}\n")
        handle.write(f"[summary] metric={summary['metric']}\n")


def main() -> None:
    args = parse_args()
    config = TASKS[args.task]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpus) < 2:
        raise ValueError("Distributed evaluation requires at least two GPU ids")
    assignments = balanced_partition(config["counts"], len(gpus))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] task={args.task} assignments={assignments}", flush=True)
    loads = [sum(config["counts"][name] for name in names) for names in assignments]
    print(f"[eval] worker_sample_counts={loads}", flush=True)
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    start = time.monotonic()
    for worker, (gpu, names) in enumerate(zip(gpus, assignments)):
        worker_result = args.result_dir / f"worker{worker}"
        worker_log = args.log_dir / f"eval_gpu{gpu}.log"
        command = [
            args.python_bin, str(ROOT / config["module"]),
            "--model_path", args.model_path,
            "--result_dir", str(worker_result),
            "--datasets", *names,
            "--seed", str(args.seed),
            "--worker", str(worker),
            "--debug_first_n", "5",
        ]
        if args.data_root is not None:
            command += ["--data_root", str(args.data_root)]
        if args.resume:
            command.append("--resume")
        else:
            command.append("--no-resume")
        if args.task == "code":
            command += ["--score"]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log_handle = worker_log.open("a", encoding="utf-8")
        log_handle.write(f"[worker] gpu={gpu} datasets={','.join(names)}\n")
        log_handle.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        processes.append((worker, process))
    failures = []
    for worker, process in processes:
        code = process.wait()
        if code:
            failures.append(f"worker{worker}=exit{code}")
    if failures:
        raise SystemExit("Evaluation worker failure: " + ", ".join(failures))
    payloads = []
    for worker in range(len(assignments)):
        path = args.result_dir / f"worker{worker}" / "summary.json"
        if not path.exists():
            path = args.result_dir / f"worker{worker}" / "summary.partial.json"
        payloads.append(read_json(path))
    summary = aggregate(args.task, payloads)
    summary["assignments"] = assignments
    summary["elapsed_seconds"] = time.monotonic() - start
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.train_log:
        append_summary(args.train_log, summary, assignments)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
