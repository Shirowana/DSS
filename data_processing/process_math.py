"""
Preprocess Math10K instruction data for DSS/PEFT SFT.

Example:
    python process_math.py \
        --data_path /root/datasets/ft-training_set/math_10k.json \
        --output_dir /root/datasets/math10k_processed \
        --model_name Llama3-8B \
        --max_length 512
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


REMOTE_DATA_ROOT = Path("/root/datasets")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")

MODEL_MAP = {
    "Llama2-7B": str(REMOTE_MODEL_ROOT / "Llama-2-7b-hf"),
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Math10K data for DSS fine-tuning.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(REMOTE_DATA_ROOT / "ft-training_set" / "math_10k.json"),
        help="Path to local math_10k.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REMOTE_DATA_ROOT / "math10k_processed"),
        help="Root directory for processed data.",
    )
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token, or set HF_TOKEN.")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    return parser.parse_args()


def generate_prompt(data_point: dict) -> str:
    instruction = data_point.get("instruction", "")
    input_text = data_point.get("input", "")
    output_text = data_point.get("output", "")
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""

    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output_text}"""


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

    def generate_and_tokenize_prompt(data_point: dict) -> dict:
        full_prompt = generate_prompt(data_point)
        tokenized_full = tokenize(full_prompt)

        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user["input_ids"])
        tokenized_full["labels"] = [-100] * user_prompt_len + tokenized_full["labels"][user_prompt_len:]

        result = {key: torch.tensor(value) for key, value in tokenized_full.items()}
        result["answer"] = data_point.get("answer", "")
        return result

    data = load_dataset("json", data_files=args.data_path)
    train_data = data["train"].shuffle(seed=args.shuffle_seed).map(generate_and_tokenize_prompt, num_proc=args.num_proc)

    save_path = Path(args.output_dir) / f"train_math10k_{args.max_length}_OnlyOutput_{args.model_name}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.save_to_disk(str(save_path))
    print(f"Dataset saved to {save_path}")
    print(f"Train examples: {len(train_data)}")


if __name__ == "__main__":
    main()
