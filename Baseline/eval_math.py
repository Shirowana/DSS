from __future__ import annotations

import argparse
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from data_processing.common import alpaca_prompt, sha256_file
from evaluation_core import (
    EvaluationLogger,
    GenerationSettings,
    Generator,
    atomic_write_json,
    generation_manifest,
    read_json,
)


DATASETS = ("gsm8k", "aqua", "mawps", "svamp")
DIRECTORIES = {"gsm8k": "gsm8k", "aqua": "AQuA", "mawps": "mawps", "svamp": "SVAMP"}
EXPECTED_COUNTS = {"gsm8k": 1319, "aqua": 254, "mawps": 238, "svamp": 1000}
RESPONSE_PREFIX = "Let's think step by step."
NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen four-task Math protocol.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument(
        "--data_root", type=Path, default=Path("/data/home/7250091/date/datasets/evaluate")
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush_every", type=int, default=100)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation")
    parser.add_argument("--debug_first_n", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--worker", type=int)
    return parser.parse_args()


def math_prompt(row: dict[str, Any]) -> str:
    return alpaca_prompt(row["instruction"], row.get("input", "")).rstrip() + " " + RESPONSE_PREFIX + "\n"


def normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    value = value.rstrip(".")
    if value.startswith("="):
        value = value[1:].strip()
    return value


def as_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(normalize_numeric_text(text))
    except InvalidOperation:
        return None


def numeric_equal(gold: str, pred: str) -> bool:
    gold_value, pred_value = as_decimal(gold), as_decimal(pred)
    if gold_value is None or pred_value is None:
        return normalize_numeric_text(gold) == normalize_numeric_text(pred)
    return abs(gold_value - pred_value) <= Decimal("1e-6")


def extract_numeric_answer(output: str) -> str:
    patterns = [
        rf"the final answer is\s*:?\s*({NUMBER})",
        rf"the answer is\s*:?\s*({NUMBER})",
        rf"####\s*({NUMBER})",
        rf"\b({NUMBER})\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            return normalize_numeric_text(matches[-1])
    return ""


def extract_choice_answer(output: str) -> str:
    patterns = [
        r"answer\s*:?\s*\(?([ABCDE])\)?",
        r"the answer is\s*:?\s*\(?([ABCDE])\)?",
        r"\(([ABCDE])\)",
        r"\b([ABCDE])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return ""


def validate_resume(existing: list[dict[str, Any]], source: list[dict[str, Any]]) -> None:
    if len(existing) > len(source) or [row.get("source_index") for row in existing] != list(
        range(len(existing))
    ):
        raise ValueError("Resume output is not an exact source-index prefix")


def main() -> None:
    args = parse_args()
    if (args.batch_size, args.num_beams, args.max_new_tokens) != (4, 4, 512):
        raise ValueError("Frozen Math evaluation requires batch=4, beams=4, max_new_tokens=512")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    logger = EvaluationLogger(task="math", worker=args.worker, debug_first_n=args.debug_first_n)
    logger.section("MATH EVAL")
    logger.log(f"[eval] model_path={args.model_path}")
    logger.log(f"[eval] result_dir={args.result_dir}")
    logger.log(f"[eval] datasets={','.join(args.datasets)}")
    logger.log(f"[eval] batch_size={args.batch_size} num_beams={args.num_beams} max_new_tokens={args.max_new_tokens} seed={args.seed}")
    generator = Generator(
        args.model_path,
        GenerationSettings(args.batch_size, args.num_beams, args.max_new_tokens, args.seed),
        args.trust_remote_code,
        args.attn_implementation,
    )
    logger.run_info(generator)
    atomic_write_json(
        args.result_dir / "generation_manifest.json",
        {
            **generation_manifest(generator, "math"),
            "prompt": "alpaca_with_lets_think_step_by_step_prefix",
            "judge": "deterministic_numeric_or_choice",
            "datasets": {
                name: {
                    "path": str(
                        (args.data_root / DIRECTORIES[name] / "test.json").resolve()
                    ),
                    "sha256": sha256_file(
                        args.data_root / DIRECTORIES[name] / "test.json"
                    ),
                    "expected_count": EXPECTED_COUNTS[name],
                }
                for name in args.datasets
            },
        },
    )
    summary: dict[str, Any] = {"datasets": {}}
    for dataset_name in args.datasets:
        source = read_json(args.data_root / DIRECTORIES[dataset_name] / "test.json")
        if len(source) != EXPECTED_COUNTS[dataset_name]:
            raise ValueError(f"{dataset_name}: wrong test count {len(source)}")
        output_path = args.result_dir / f"{dataset_name}.json"
        completed = read_json(output_path) if args.resume and output_path.exists() else []
        validate_resume(completed, source)
        logger.dataset_start(dataset_name, args.data_root / DIRECTORIES[dataset_name] / "test.json", len(source), len(completed))
        correct_so_far = sum(bool(row.get("correct")) for row in completed)
        for start in range(len(completed), len(source), args.flush_every):
            rows = source[start : start + args.flush_every]
            prompts = [math_prompt(row) for row in rows]
            outputs, token_batches = generator.generate(prompts, progress_label=dataset_name)
            for offset, (row, prompt, output, token_ids) in enumerate(zip(rows, prompts, outputs, token_batches)):
                gold = str(row.get("answer", "")).strip()
                if dataset_name == "aqua":
                    pred = extract_choice_answer(output)
                    correct = pred == gold.upper()
                else:
                    pred = extract_numeric_answer(output)
                    correct = bool(pred) and numeric_equal(gold, pred)
                completed.append(
                    {
                        "source_index": start + offset,
                        "instruction": row.get("instruction", ""),
                        "input": row.get("input", ""),
                        "prompt": prompt,
                        "gold": gold,
                        "output_pred": output,
                        "pred": pred,
                        "correct": correct,
                    }
                )
                correct_so_far += int(correct)
                sample_index = start + offset
                logger.sample_debug(
                    dataset=dataset_name, sample_index=sample_index, gold_raw=row.get("answer"),
                    gold=gold, pred=pred, correct=correct, prompt=prompt, output=output,
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
        values = [summary["datasets"][name] for name in DATASETS]
        summary["average_accuracy"] = sum(value["accuracy"] for value in values) / 4
        summary["4_average_accuracy"] = summary["average_accuracy"]
        summary["weighted_accuracy"] = sum(value["correct"] for value in values) / sum(
            value["total"] for value in values
        )
        summary["num_beams"] = 4
        summary["judge"] = "deterministic_numeric_or_symbolic_equivalence"
        summary["api_judge"] = False
        atomic_write_json(args.result_dir / "summary.json", summary)
    logger.done()


if __name__ == "__main__":
    main()
