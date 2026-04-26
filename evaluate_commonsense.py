"""
Evaluate a DSS PEFT adapter on commonsense reasoning benchmarks.

The script follows the DiaBlo commonsense evaluation format:
load ``dataset_commonsense/{dataset}/test.json``, generate short answers, extract
the option token with dataset-specific regex rules, and save per-example results.

Example:
    python evaluate_commonsense.py \
        --model_name Llama3-8B \
        --dataset boolq \
        --adapter_path /data/home/7250091/date/DSS/output/... \
        --data_dir /data/home/7250091/date/datasets/evaluate \
        --output_dir /data/home/7250091/date/DSS/results_commonsense/...
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires `peft` to load DSS adapters.") from exc


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dss  # noqa: F401,E402 - importing registers the DSS PEFT method


SUPPORTED_DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]

REMOTE_PROJECT_ROOT = Path("/data/home/7250091/date/DSS")
REMOTE_DATA_ROOT = Path("/data/home/7250091/date/datasets")
REMOTE_MODEL_ROOT = Path("/data/home/7250091/date/hf_cache_models/models")
REMOTE_RESULTS_ROOT = REMOTE_PROJECT_ROOT / "results_commonsense"

MODEL_MAP = {
    "Llama2-7B": str(REMOTE_MODEL_ROOT / "Llama2-7B"),
    "Llama2-13B": str(REMOTE_MODEL_ROOT / "Llama2-13B"),
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Llama3-8B"),
    "Llama3-3B": str(REMOTE_MODEL_ROOT / "Llama3-3B"),
    "Mistral-7B": str(REMOTE_MODEL_ROOT / "Mistral-7B"),
    "Qwen2.5-7B": str(REMOTE_MODEL_ROOT / "Qwen2.5-7B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSS on commonsense reasoning benchmarks.")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--adapter_path", type=str, default=None, help="DSS adapter directory. Omit for base-model zero-shot.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=str(REMOTE_DATA_ROOT / "evaluate"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--merge_adapter", action="store_true", help="Merge DSS adapter into the base model before evaluation.")
    return parser.parse_args()


def resolve_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def generate_prompt(instruction: str, input_text: str | None = None) -> str:
    if input_text:
        return f"{instruction}\n{input_text}"
    return f"{instruction}\n"


def load_data(args: argparse.Namespace) -> list[dict]:
    file_path = Path(args.data_dir) / "dataset_commonsense" / args.dataset / "test.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model(args: argparse.Namespace) -> tuple[AutoTokenizer, torch.nn.Module]:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)

    model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR")
    model_id = args.model_path or MODEL_MAP[args.model_name]
    torch_dtype = resolve_dtype(args.precision)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_dir, torch_dtype=torch_dtype)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        if args.merge_adapter:
            model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model


def extract_answer(dataset: str, sentence: str) -> str:
    sentence = sentence.strip().lower()
    if dataset == "boolq":
        pred = re.findall(r"true|false", sentence)
    elif dataset == "piqa":
        pred = re.findall(r"solution1|solution2", sentence)
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        pred = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence)
    elif dataset == "hellaswag":
        pred = re.findall(r"ending1|ending2|ending3|ending4", sentence)
    elif dataset == "winogrande":
        pred = re.findall(r"option1|option2", sentence)
    else:
        pred = []
    return pred[0] if pred else ""


def main() -> None:
    args = parse_args()
    dataset = load_data(args)
    tokenizer, model = load_model(args)
    device = next(model.parameters()).device

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_name = Path(args.adapter_path).name if args.adapter_path else "base"
    run_name = args.run_name or f"eval_commonsense_{args.model_name}_{adapter_name}_{timestamp}"
    output_dir = Path(args.output_dir) if args.output_dir else REMOTE_RESULTS_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / f"{args.dataset}.json"

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    batches = [dataset[i : i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]
    correct = 0
    seen = 0
    output_data: list[dict] = []

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Evaluating {args.dataset}")):
        instructions = [item.get("instruction", "") for item in batch]
        inputs = [item.get("input") for item in batch]
        prompts = [generate_prompt(instruction, input_text) for instruction, input_text in zip(instructions, inputs)]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.no_grad():
            generation_output = model.generate(
                **tokenized,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_new_tokens,
            )

        decoded = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        decoded = [text.split("the correct answer is")[-1].strip() for text in decoded]

        for item, output in zip(batch, decoded):
            label = str(item.get("answer", "")).lower()
            predict = extract_answer(args.dataset, output)
            flag = label == predict
            correct += int(flag)
            seen += 1

            row = copy.deepcopy(item)
            row["output_pred"] = output
            row["pred"] = predict
            row["flag"] = flag
            output_data.append(row)

        accuracy = correct / max(seen, 1)
        print(f"  {batch_idx + 1}/{len(batches)} | accuracy: {correct}/{seen} = {accuracy:.4f}")

        with save_file.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=4, ensure_ascii=False)

    final_accuracy = correct / max(seen, 1)
    print(f"\nFinal accuracy: {correct}/{seen} = {final_accuracy:.4f}")
    print(f"Saved predictions to {save_file}")


if __name__ == "__main__":
    main()
