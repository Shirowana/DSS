"""
Preprocess Math10K for the DSS math reasoning protocol.

This creates a Hugging Face disk dataset with prompt tokens masked out from
the loss and a fixed validation split stored alongside metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


REMOTE_DATA_ROOT = Path("/root/datasets")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")

MODEL_MAP = {
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
    "Qwen2.5-7B": str(REMOTE_MODEL_ROOT / "Qwen2.5-7B"),
}

RESPONSE_PREFIX = "Let's think step by step."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Math10K for DSS math fine-tuning.")
    parser.add_argument("--data_path", default=str(REMOTE_DATA_ROOT / "ft-training_set" / "math_10k.json"))
    parser.add_argument("--output_dir", default=str(REMOTE_DATA_ROOT / "math_reasoning" / "processed"))
    parser.add_argument("--model_name", default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--model_cache_dir", default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--validation_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload)!r}")
    return payload


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


def response_parts(row: dict) -> tuple[str, str]:
    output = str(row.get("output") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not answer:
        return output, ""
    return output, f"The final answer is: {answer}"


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_root = Path(args.output_dir)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)

    model_path = args.model_path or MODEL_MAP[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.model_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rows = load_json_list(data_path)
    dataset = Dataset.from_list(rows).shuffle(seed=args.seed)
    if args.validation_samples <= 0 or args.validation_samples >= len(dataset):
        raise ValueError("validation_samples must be positive and smaller than the Math10K dataset size.")
    split = dataset.train_test_split(test_size=args.validation_samples, shuffle=False)

    truncation_stats = {
        "prompt_truncated": 0,
        "reasoning_truncated": 0,
        "response_truncated": 0,
        "final_answer_preserved": 0,
        "total": 0,
    }

    def tokenize_row(row: dict) -> dict:
        prompt = instruction_prompt(str(row.get("instruction") or ""), str(row.get("input") or ""))
        reasoning, final_answer = response_parts(row)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        reasoning_ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"]
        final_answer_ids = []
        if final_answer:
            final_answer_ids = tokenizer(f"\n\n{final_answer}", add_special_tokens=False)["input_ids"]

        if len(prompt_ids) > args.max_prompt_length:
            prompt_ids = prompt_ids[-args.max_prompt_length :]
            truncation_stats["prompt_truncated"] += 1

        available_response = args.max_length - len(prompt_ids) - 1
        if available_response < 1:
            prompt_ids = prompt_ids[-(args.max_length - 2) :]
            available_response = 1
            truncation_stats["prompt_truncated"] += 1

        if final_answer_ids and len(final_answer_ids) >= available_response:
            final_answer_ids = final_answer_ids[:available_response]
            reasoning_ids = []
            truncation_stats["response_truncated"] += 1
        else:
            available_reasoning = available_response - len(final_answer_ids)
            if len(reasoning_ids) > available_reasoning:
                reasoning_ids = reasoning_ids[:available_reasoning]
                truncation_stats["reasoning_truncated"] += 1
                truncation_stats["response_truncated"] += 1

        response_ids = reasoning_ids + final_answer_ids
        if final_answer_ids:
            truncation_stats["final_answer_preserved"] += 1
        if len(response_ids) > available_response:
            response_ids = response_ids[:available_response]
            truncation_stats["response_truncated"] += 1

        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        input_ids = prompt_ids + response_ids + eos
        labels = [-100] * len(prompt_ids) + response_ids + eos
        truncation_stats["total"] += 1
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "answer": str(row.get("answer") or ""),
        }

    tokenized = DatasetDict(
        {
            "train": split["train"].map(tokenize_row, remove_columns=split["train"].column_names),
            "validation": split["test"].map(tokenize_row, remove_columns=split["test"].column_names),
        }
    )

    save_path = output_root / f"math10k_{args.max_length}_prompt{args.max_prompt_length}_OnlyOutput_{args.model_name}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(save_path))

    metadata = {
        "dataset_name": "math10k",
        "source": "LLM-Adapters math_10k.json local copy",
        "original_file": str(data_path),
        "processed_dir": str(save_path),
        "num_examples": len(dataset),
        "num_train": len(tokenized["train"]),
        "num_validation": len(tokenized["validation"]),
        "sha256": sha256_file(data_path),
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "seed": args.seed,
        "truncation": truncation_stats,
        "notes": (
            "Prompt tokens are label-masked; prompts use `### Response: Let's think step by step.`; "
            "reasoning is truncated before the standardized final-answer suffix."
        ),
    }
    metadata_dir = output_root.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / f"math10k_{args.model_name}_{args.max_length}.metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"[math10k] saved dataset: {save_path}")
    print(f"[math10k] train={len(tokenized['train'])} validation={len(tokenized['validation'])}")
    print(f"[math10k] metadata: {metadata_dir}")


if __name__ == "__main__":
    main()
