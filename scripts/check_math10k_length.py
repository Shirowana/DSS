#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


REMOTE_DATA_ROOT = Path("/root/datasets")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")
MODEL_MAP = {
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
    "Qwen2.5-7B": str(REMOTE_MODEL_ROOT / "Qwen2.5-7B"),
}
RESPONSE_PREFIX = "Let's think step by step."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Math10K token lengths and answer truncation risk.")
    parser.add_argument("--data_path", default=str(REMOTE_DATA_ROOT / "ft-training_set" / "math_10k.json"))
    parser.add_argument("--model_name", default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_cache_dir", default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--short_length", type=int, default=512)
    parser.add_argument("--long_length", type=int, default=768)
    parser.add_argument("--example_index", type=int, default=-1)
    return parser.parse_args()


def instruction_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response: {RESPONSE_PREFIX}\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response: {RESPONSE_PREFIX}\n"
    )


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return rows


def answer_visible(answer: str, decoded_response: str) -> bool:
    answer = answer.strip()
    if not answer:
        return True
    candidates = {answer, answer.rstrip(".0"), answer.replace(",", "")}
    return any(candidate and candidate in decoded_response for candidate in candidates)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path or MODEL_MAP[args.model_name], cache_dir=args.model_cache_dir)
    rows = load_rows(Path(args.data_path))

    records: list[dict] = []
    for index, row in enumerate(rows):
        prompt = instruction_prompt(str(row.get("instruction") or ""), str(row.get("input") or ""))
        response = str(row.get("output") or "").strip()
        prompt_ids_raw = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        prompt_truncated = len(prompt_ids_raw) > args.max_prompt_length
        prompt_ids = prompt_ids_raw[-args.max_prompt_length:] if prompt_truncated else prompt_ids_raw
        total = len(prompt_ids) + len(response_ids) + 1
        available_short = args.short_length - len(prompt_ids) - 1
        truncated_short_ids = response_ids[: max(available_short, 0)]
        truncated_short_text = tokenizer.decode(truncated_short_ids, skip_special_tokens=True)
        records.append(
            {
                "index": index,
                "row": row,
                "prompt_tokens_raw": len(prompt_ids_raw),
                "prompt_tokens_used": len(prompt_ids),
                "prompt_truncated": prompt_truncated,
                "response_tokens": len(response_ids),
                "total_tokens": total,
                "over_short": total > args.short_length,
                "over_long": total > args.long_length,
                "between": args.short_length < total <= args.long_length,
                "answer_visible_after_short_trunc": answer_visible(str(row.get("answer") or ""), truncated_short_text),
                "truncated_short_text": truncated_short_text,
            }
        )

    over_short = [record for record in records if record["over_short"]]
    over_long = [record for record in records if record["over_long"]]
    between = [record for record in records if record["between"]]
    between_answer_lost = [record for record in between if not record["answer_visible_after_short_trunc"]]
    prompt_truncated = [record for record in records if record["prompt_truncated"]]

    print("========== Math10K Length Check ==========")
    print(f"data_path={args.data_path}")
    print(f"model_name={args.model_name}")
    print(f"max_prompt_length={args.max_prompt_length}")
    print(f"short_length={args.short_length}")
    print(f"long_length={args.long_length}")
    print(f"total_examples={len(records)}")
    print(f"prompt_raw_gt_{args.max_prompt_length}={len(prompt_truncated)}")
    print(f"total_gt_{args.short_length}={len(over_short)}")
    print(f"total_gt_{args.long_length}={len(over_long)}")
    print(f"{args.short_length}_lt_total_le_{args.long_length}={len(between)}")
    print(f"between_answer_lost_if_truncate_to_{args.short_length}={len(between_answer_lost)}")

    example = None
    if args.example_index >= 0:
        example = next((record for record in records if record["index"] == args.example_index), None)
    if example is None and between_answer_lost:
        example = between_answer_lost[0]
    if example is None and between:
        example = between[0]
    if example is None and over_short:
        example = over_short[0]

    if example:
        row = example["row"]
        full_response = str(row.get("output") or "").strip()
        print("\n========== Example ==========")
        print(f"index={example['index']}")
        print(f"prompt_tokens_raw={example['prompt_tokens_raw']}")
        print(f"prompt_tokens_used={example['prompt_tokens_used']}")
        print(f"response_tokens={example['response_tokens']}")
        print(f"total_tokens={example['total_tokens']}")
        print(f"answer={row.get('answer')}")
        print(f"answer_visible_after_short_trunc={example['answer_visible_after_short_trunc']}")
        print("\n[instruction]")
        print(str(row.get("instruction") or "").strip()[:1200])
        print("\n[full response tail]")
        print(full_response[-1200:])
        print(f"\n[response after {args.short_length} truncation tail]")
        print(example["truncated_short_text"][-1200:])


if __name__ == "__main__":
    main()
