"""
Preprocess commonsense_170k.json for DSS commonsense SFT.

Example:
    python process_commonsense.py \
        --data_path /root/datasets/ft-training_set/commonsense_170k.json \
        --output_dir /root/datasets/commonsense_new \
        --model_name Llama3-8B \
        --max_length 512
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


REMOTE_DATA_ROOT = Path("/root/datasets")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")
COMMONSENSE_EVAL_ROOT = REMOTE_DATA_ROOT / "evaluate"
COMMONSENSE_TASKS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]

MODEL_MAP = {
    "Llama2-7B": str(REMOTE_MODEL_ROOT / "Llama-2-7b-hf"),
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess commonsense data for DSS fine-tuning.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(REMOTE_DATA_ROOT / "ft-training_set" / "commonsense_170k.json"),
        help="Path to local commonsense_170k.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REMOTE_DATA_ROOT / "commonsense_new"),
        help="Root directory for processed data.",
    )
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token, or set HF_TOKEN.")
    parser.add_argument("--num_proc", type=int, default=32)
    return parser.parse_args()


def example_signature(data_point: dict) -> tuple[str, str, str, str]:
    return (
        data_point.get("instruction", ""),
        data_point.get("input", ""),
        data_point.get("output", ""),
        data_point.get("answer", ""),
    )


def build_task_lookup() -> dict[tuple[str, str, str, str], str]:
    task_lookup: dict[tuple[str, str, str, str], str] = {}
    for task_name in COMMONSENSE_TASKS:
        task_path = COMMONSENSE_EVAL_ROOT / task_name / "train.json"
        if not task_path.exists():
            raise FileNotFoundError(f"Missing task train split for balanced init labeling: {task_path}")
        with task_path.open("r", encoding="utf-8") as handle:
            task_rows = json.load(handle)
        for row in task_rows:
            signature = example_signature(row)
            if signature in task_lookup:
                previous_task = task_lookup[signature]
                if previous_task != task_name:
                    raise ValueError(
                        f"Duplicate commonsense example signature appears in both {previous_task!r} and {task_name!r}."
                    )
                continue
            task_lookup[signature] = task_name
    return task_lookup


def generate_prompt(data_point: dict) -> str:
    if data_point.get("input"):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""

    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def main() -> None:
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)

    model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR")
    model_path = args.model_path or MODEL_MAP[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenize(prompt: str, add_eos_token: bool = True) -> dict:
        result = tokenizer(prompt, truncation=True, max_length=args.max_length, padding=False, return_tensors=None)
        if (
            add_eos_token
            and result["input_ids"]
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.max_length
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    task_lookup = build_task_lookup()

    def add_task_name(data_point: dict) -> dict:
        signature = example_signature(data_point)
        if signature not in task_lookup:
            raise KeyError("Failed to map raw commonsense example to a known task split.")
        return {**data_point, "task_name": task_lookup[signature]}

    def generate_and_tokenize_prompt(data_point: dict) -> dict:
        full_prompt = generate_prompt(data_point)
        tokenized_full = tokenize(full_prompt)

        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user["input_ids"])
        tokenized_full["labels"] = [-100] * user_prompt_len + tokenized_full["labels"][user_prompt_len:]

        result = {key: torch.tensor(value) for key, value in tokenized_full.items()}
        result["task_name"] = data_point["task_name"]
        return result

    data = load_dataset("json", data_files=args.data_path)
    train_data = data["train"].map(add_task_name, num_proc=args.num_proc)
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt, num_proc=args.num_proc)

    missing_columns = {"task_name"} - set(train_data.column_names)
    if missing_columns:
        raise RuntimeError(f"Processed dataset is missing required task columns: {sorted(missing_columns)}")

    task_counts: dict[str, int] = {}
    for task_name in train_data["task_name"]:
        task_counts[task_name] = task_counts.get(task_name, 0) + 1

    save_path = Path(args.output_dir) / f"train_all_{args.max_length}_OnlyOutput_{args.model_name}"
    train_data.save_to_disk(str(save_path))
    print(f"Dataset saved to {save_path}")
    print("Task counts:", json.dumps(task_counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
