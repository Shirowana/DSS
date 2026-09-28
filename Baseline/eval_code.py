from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from data_processing.common import sha256_file
from evaluation_core import (
    EvaluationLogger,
    GenerationSettings,
    Generator,
    atomic_write_json,
    atomic_write_jsonl,
    generation_manifest,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/home/7250091/date/datasets/code_generation/official/evalplus")
SANDBOX = ROOT / "tools" / "code_eval_sandbox" / "run_evalplus_sandbox.sh"
DATASET_SPECS = {
    "humaneval": {
        "filename": "HumanEvalPlus-v0.1.10.jsonl.gz",
        "prefix": "HumanEval/",
        "count": 164,
    },
    "mbpp": {"filename": "MbppPlus-v0.2.0.jsonl.gz", "prefix": "Mbpp/", "count": 378},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and sandbox-score EvalPlus 0.3.1.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_SPECS), default=list(DATASET_SPECS)
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush_every_batches", type=int, default=10)
    parser.add_argument("--verify_batch_equivalence", type=int, default=0, metavar="N")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation")
    parser.add_argument("--debug_first_n", type=int, default=5)
    parser.add_argument("--worker", type=int)
    return parser.parse_args()


def chat_prompts(tokenizer, rows: list[dict[str, Any]]) -> list[str]:
    if tokenizer.chat_template is None:
        raise ValueError("Code evaluation requires the Llama3-Instruct chat template")
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def terminators(tokenizer) -> list[int]:
    values = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and 0 <= eot_id < len(tokenizer) and eot_id not in values:
        values.append(eot_id)
    return values


def validate_canonical(rows: list[dict[str, Any]], dataset_name: str) -> None:
    spec = DATASET_SPECS[dataset_name]
    if len(rows) != spec["count"]:
        raise ValueError(f"{dataset_name}: expected {spec['count']} tasks, found {len(rows)}")
    task_ids = [row.get("task_id") for row in rows]
    if len(set(task_ids)) != len(task_ids) or not all(
        isinstance(task_id, str) and task_id.startswith(spec["prefix"]) for task_id in task_ids
    ):
        raise ValueError(f"{dataset_name}: invalid or duplicate canonical task IDs")


def validate_resume(completed: list[dict[str, Any]], source: list[dict[str, Any]]) -> None:
    expected = [row["task_id"] for row in source[: len(completed)]]
    actual = [row.get("task_id") for row in completed]
    if len(completed) > len(source) or actual != expected:
        raise ValueError("Existing samples are not an exact canonical prefix")
    if any(set(row) != {"task_id", "solution"} for row in completed):
        raise ValueError("Existing samples must contain exactly task_id and solution")


def score_result(path: Path, expected_count: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("eval", {})
    if len(results) != expected_count:
        raise ValueError(f"EvalPlus result has {len(results)} tasks, expected {expected_count}")
    base_pass = plus_pass = 0
    for task_id, samples in results.items():
        if len(samples) != 1:
            raise ValueError(f"{task_id}: expected exactly one completion")
        sample = samples[0]
        base_ok = sample.get("base_status") == "pass"
        plus_ok = sample.get("plus_status") == "pass"
        base_pass += int(base_ok)
        plus_pass += int(base_ok and plus_ok)
    return {
        "total": expected_count,
        "base_correct": base_pass,
        "plus_correct": plus_pass,
        "base_pass_at_1": 100.0 * base_pass / expected_count,
        "plus_pass_at_1": 100.0 * plus_pass / expected_count,
    }


def run_sandbox(dataset_name: str, sample_path: Path) -> Path:
    subprocess.run([str(SANDBOX), "sanitize", str(sample_path)], check=True)
    sanitized = sample_path.with_name(sample_path.name.replace(".jsonl", "-sanitized.jsonl"))
    if not sanitized.is_file():
        raise FileNotFoundError(f"EvalPlus did not create {sanitized}")
    subprocess.run(
        [
            str(SANDBOX),
            "evaluate",
            dataset_name,
            str(sanitized),
            "--i_just_wanna_run",
        ],
        check=True,
    )
    result = sanitized.with_name(sanitized.name.replace(".jsonl", "_eval_results.json"))
    if not result.is_file():
        raise FileNotFoundError(f"EvalPlus did not create {result}")
    return result


def main() -> None:
    args = parse_args()
    if (args.batch_size, args.max_new_tokens) != (16, 1024):
        raise ValueError("Frozen Code evaluation requires batch=16 and max_new_tokens=1024")
    if args.score and (not SANDBOX.is_file() or not SANDBOX.stat().st_mode & 0o111):
        raise FileNotFoundError(f"Missing executable EvalPlus sandbox: {SANDBOX}")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    logger = EvaluationLogger(task="code", worker=args.worker, debug_first_n=args.debug_first_n)
    logger.section("CODE EVAL")
    logger.log(f"[eval] model_path={args.model_path}")
    logger.log(f"[eval] result_dir={args.result_dir}")
    logger.log(f"[eval] datasets={','.join(args.datasets)}")
    sources = {}
    for dataset_name in args.datasets:
        path = args.data_root / DATASET_SPECS[dataset_name]["filename"]
        sources[dataset_name] = read_jsonl(path)
        validate_canonical(sources[dataset_name], dataset_name)

    generator = Generator(
        args.model_path,
        GenerationSettings(args.batch_size, 1, args.max_new_tokens, args.seed),
        args.trust_remote_code,
        args.attn_implementation,
    )
    logger.run_info(generator)
    stop_ids = terminators(generator.tokenizer)
    manifest = {
        **generation_manifest(generator, "code"),
        "prompt": {
            "messages": [{"role": "user", "content": "canonical EvalPlus prompt"}],
            "system_message": None,
            "add_generation_prompt": True,
            "add_special_tokens": False,
        },
        "terminator_ids": stop_ids,
        "evalplus": {"version": "0.3.1", "sandbox": str(SANDBOX)},
        "datasets": {
            name: {
                "path": str((args.data_root / DATASET_SPECS[name]["filename"]).resolve()),
                "sha256": sha256_file(args.data_root / DATASET_SPECS[name]["filename"]),
                "expected_count": DATASET_SPECS[name]["count"],
            }
            for name in args.datasets
        },
    }
    if args.verify_batch_equivalence:
        verification_rows = []
        for dataset_name in args.datasets:
            verification_rows.extend(sources[dataset_name])
        verification_rows = verification_rows[: args.verify_batch_equivalence]
        prompts = chat_prompts(generator.tokenizer, verification_rows)
        manifest["batch_equivalence"] = generator.verify_batch_equivalence(
            prompts, add_special_tokens=False, terminators=stop_ids
        )
    atomic_write_json(args.result_dir / "generation_manifest.json", manifest)

    for dataset_name in args.datasets:
        source = sources[dataset_name]
        output_path = args.result_dir / f"{dataset_name}_samples.jsonl"
        completed = read_jsonl(output_path) if args.resume and output_path.exists() else []
        validate_resume(completed, source)
        logger.dataset_start(dataset_name, args.data_root / DATASET_SPECS[dataset_name]["filename"], len(source), len(completed))
        chunk_size = args.batch_size * args.flush_every_batches
        for start in range(len(completed), len(source), chunk_size):
            rows = source[start : start + chunk_size]
            prompts = chat_prompts(generator.tokenizer, rows)
            outputs, token_batches = generator.generate(
                prompts,
                add_special_tokens=False,
                terminators=stop_ids,
                progress_label=dataset_name,
            )
            for offset, (row, output, token_ids) in enumerate(zip(rows, outputs, token_batches)):
                completed.append({"task_id": row["task_id"], "solution": output})
                logger.sample_debug(
                    dataset=dataset_name, sample_index=start + offset, gold_raw=None, gold=None,
                    pred=None, correct=False, prompt=prompts[offset], output=output,
                    token_ids=token_ids, decoded_tokens=generator.decode_tokens(token_ids),
                )
            atomic_write_jsonl(output_path, completed)
        validate_resume(completed, source)

    if not args.score:
        return
    scored = {}
    for dataset_name in args.datasets:
        result_path = run_sandbox(dataset_name, args.result_dir / f"{dataset_name}_samples.jsonl")
        scored[dataset_name] = score_result(result_path, DATASET_SPECS[dataset_name]["count"])
        score = scored[dataset_name]
        logger.log(
            f"[summary] {dataset_name:<14} base_pass_at_1={score['base_pass_at_1'] / 100.0:.4f} "
            f"plus_pass_at_1={score['plus_pass_at_1'] / 100.0:.4f}"
        )
    summary: dict[str, Any] = {"datasets": scored}
    if set(args.datasets) == set(DATASET_SPECS):
        summary.update(
            {
                "HumanEval": scored["humaneval"]["base_pass_at_1"],
                "HumanEval+": scored["humaneval"]["plus_pass_at_1"],
                "MBPP": scored["mbpp"]["base_pass_at_1"],
                "MBPP+": scored["mbpp"]["plus_pass_at_1"],
            }
        )
        summary["average_accuracy"] = sum(
            summary[key] for key in ("HumanEval", "HumanEval+", "MBPP", "MBPP+")
        ) / 4
        summary["metric"] = "greedy_pass_at_1_percent"
        atomic_write_json(args.result_dir / "summary.json", summary)
    else:
        atomic_write_json(args.result_dir / "summary.partial.json", summary)
    logger.done()


if __name__ == "__main__":
    main()
