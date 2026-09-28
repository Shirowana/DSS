from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from data_processing.common import alpaca_prompt, sha256_file
from evaluation_core import (
    GenerationSettings,
    Generator,
    EvaluationLogger,
    atomic_write_json,
    generation_manifest,
    read_json,
)


DATASETS = (
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
)
EXPECTED_COUNTS = {
    "boolq": 3270,
    "piqa": 1838,
    "social_i_qa": 1954,
    "hellaswag": 10042,
    "winogrande": 1267,
    "ARC-Challenge": 1172,
    "ARC-Easy": 2376,
    "openbookqa": 500,
}
LABEL_SPECS = {
    "piqa": ("solution", ("solution", "choice"), 2),
    "social_i_qa": ("answer", ("answer", "choice"), 3),
    "hellaswag": ("ending", ("ending", "choice"), 4),
    "winogrande": ("option", ("option", "choice"), 2),
    "ARC-Challenge": ("answer", ("answer", "choice"), 4),
    "ARC-Easy": ("answer", ("answer", "choice"), 5),
    "openbookqa": ("answer", ("answer", "choice"), 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen eight-task Commonsense protocol.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument(
        "--data_root", type=Path, default=Path("/data/home/7250091/date/datasets/evaluate")
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush_every", type=int, default=100)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation")
    parser.add_argument("--debug_first_n", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--worker", type=int)
    return parser.parse_args()


def actual_choice_count(row: dict[str, Any], maximum: int) -> int:
    instruction = str(row.get("instruction", ""))
    choices = [int(value) for value in re.findall(r"\bAnswer([1-9])\s*:", instruction)]
    return min(max(choices, default=maximum), maximum)


def normalize_label(dataset: str, text: str | None, row: dict[str, Any] | None = None) -> str:
    if not text:
        return ""
    if dataset == "boolq":
        matches = re.findall(r"\b(true|false)\b", text, flags=re.IGNORECASE)
        return matches[-1].lower() if matches else ""
    prefix, aliases, maximum = LABEL_SPECS[dataset]
    if dataset == "ARC-Easy" and row is not None:
        maximum = actual_choice_count(row, maximum)
    choices = "".join(str(number) for number in range(1, maximum + 1))
    alias_pattern = "|".join(aliases)
    patterns = [rf"\b(?:{alias_pattern})\s*([{choices}])\b", rf"\b([{choices}])\b"]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return f"{prefix}{matches[-1]}".lower()
    return ""


def validate_resume(existing: list[dict[str, Any]], source: list[dict[str, Any]]) -> None:
    if len(existing) > len(source):
        raise ValueError("Resume output is longer than the source dataset")
    if [row.get("source_index") for row in existing] != list(range(len(existing))):
        raise ValueError("Resume output is not an exact source-index prefix")


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Frozen Commonsense protocol requires --batch_size 1")
    if args.max_new_tokens != 32:
        raise ValueError("Frozen Commonsense protocol requires --max_new_tokens 32")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    logger = EvaluationLogger(task="commonsense", worker=args.worker, debug_first_n=args.debug_first_n)
    logger.section("COMMONSENSE EVAL")
    logger.log(f"[eval] model_path={args.model_path}")
    logger.log(f"[eval] result_dir={args.result_dir}")
    logger.log(f"[eval] datasets={','.join(args.datasets)}")
    logger.log(f"[eval] batch_size={args.batch_size} max_new_tokens={args.max_new_tokens} seed={args.seed}")
    generator = Generator(
        args.model_path,
        GenerationSettings(args.batch_size, 1, args.max_new_tokens, args.seed),
        args.trust_remote_code,
        args.attn_implementation,
    )
    logger.run_info(generator)
    atomic_write_json(
        args.result_dir / "generation_manifest.json",
        {
            **generation_manifest(generator, "commonsense"),
            "prompt": "alpaca",
            "parser": "last_valid_label",
            "datasets": {
                name: {
                    "path": str((args.data_root / name / "test.json").resolve()),
                    "sha256": sha256_file(args.data_root / name / "test.json"),
                    "expected_count": EXPECTED_COUNTS[name],
                }
                for name in args.datasets
            },
        },
    )
    summary: dict[str, Any] = {"datasets": {}}
    for dataset_name in args.datasets:
        source_path = args.data_root / dataset_name / "test.json"
        source = read_json(source_path)
        if len(source) != EXPECTED_COUNTS[dataset_name]:
            raise ValueError(
                f"{dataset_name}: expected {EXPECTED_COUNTS[dataset_name]}, found {len(source)}"
            )
        output_path = args.result_dir / f"{dataset_name}.json"
        completed = read_json(output_path) if args.resume and output_path.exists() else []
        validate_resume(completed, source)
        logger.dataset_start(dataset_name, source_path, len(source), len(completed))
        correct_so_far = sum(bool(row.get("correct")) for row in completed)
        for start in range(len(completed), len(source), args.flush_every):
            rows = source[start : start + args.flush_every]
            prompts = [alpaca_prompt(row["instruction"], row.get("input", "")) for row in rows]
            outputs, token_batches = generator.generate(prompts, progress_label=dataset_name)
            for offset, (row, prompt, output, token_ids) in enumerate(zip(rows, prompts, outputs, token_batches)):
                gold = normalize_label(dataset_name, row.get("answer"), row)
                pred = normalize_label(dataset_name, output, row)
                is_correct = bool(gold and gold == pred)
                completed.append(
                    {
                        "source_index": start + offset,
                        "instruction": row.get("instruction", ""),
                        "input": row.get("input", ""),
                        "prompt": prompt,
                        "gold": gold,
                        "output_pred": output,
                        "pred": pred,
                        "correct": is_correct,
                    }
                )
                correct_so_far += int(is_correct)
                sample_index = start + offset
                logger.sample_debug(
                    dataset=dataset_name, sample_index=sample_index, gold_raw=row.get("answer"),
                    gold=gold, pred=pred, correct=is_correct, prompt=prompt, output=output,
                    token_ids=token_ids, decoded_tokens=generator.decode_tokens(token_ids),
                )
                if args.log_every > 0 and (sample_index + 1) % args.log_every == 0:
                    logger.running_accuracy(sample_index + 1, len(source), correct_so_far)
            atomic_write_json(output_path, completed)
        correct = sum(row["correct"] for row in completed)
        summary["datasets"][dataset_name] = {
            "correct": correct,
            "total": len(completed),
            "accuracy": correct / len(completed),
        }
        logger.dataset_done(dataset_name, correct, len(completed))
        atomic_write_json(args.result_dir / "summary.partial.json", summary)
    if set(args.datasets) == set(DATASETS):
        summary["average_accuracy"] = sum(
            summary["datasets"][name]["accuracy"] for name in DATASETS
        ) / len(DATASETS)
        summary["metric"] = "unweighted_mean_of_eight_task_accuracies"
        atomic_write_json(args.result_dir / "summary.json", summary)
    else:
        atomic_write_json(args.result_dir / "summary.partial.json", summary)
    logger.done()


if __name__ == "__main__":
    main()
