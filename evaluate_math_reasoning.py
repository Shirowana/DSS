from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from peft import PeftModel
from peft.tuners.dss import DSSConfig  # noqa: F401 - register DSS PEFT type


REMOTE_DATA_ROOT = Path("/root/datasets")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")

DATASETS = ["gsm8k", "svamp", "aqua", "mawps"]
RESPONSE_PREFIX = "Let's think step by step."
DATASET_DIR_NAMES = {
    "gsm8k": "gsm8k",
    "svamp": "SVAMP",
    "aqua": "AQuA",
    "mawps": "mawps",
}

MODEL_MAP = {
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
    "Qwen2.5-7B": str(REMOTE_MODEL_ROOT / "Qwen2.5-7B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSS adapters on math reasoning benchmarks.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model_name", default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default=str(REMOTE_DATA_ROOT / "evaluate"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--model_cache_dir", default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--max_examples", type=int, default=0)
    return parser.parse_args()


def resolve_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def generate_prompt(dataset: str, instruction: str, input_text: str = "") -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response: {RESPONSE_PREFIX}\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response: {RESPONSE_PREFIX}\n"
    )


def load_data(args: argparse.Namespace) -> list[dict]:
    data_dir = Path(args.data_dir)
    path = data_dir / DATASET_DIR_NAMES[args.dataset] / "test.json"
    if not path.exists():
        path = data_dir / args.dataset / "test.json"
    if not path.exists() and data_dir.name in {args.dataset, DATASET_DIR_NAMES[args.dataset]}:
        path = data_dir / "test.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    return rows


def resolve_adapter_dir(adapter_path: str | None) -> str | None:
    if not adapter_path:
        return None
    path = Path(adapter_path)
    if path.is_file():
        if path.name not in {"adapter_model.safetensors", "adapter_model.bin"}:
            raise ValueError(f"Unsupported adapter file path: {adapter_path}. Please pass an adapter directory.")
        path = path.parent
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter directory does not contain adapter_config.json: {path}")
    return str(path)


def load_model(args: argparse.Namespace):
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)

    model_path = args.model_path or MODEL_MAP[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.model_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.model_cache_dir,
        torch_dtype=resolve_precision(args.precision),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None and len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    adapter_dir = resolve_adapter_dir(args.adapter_path)
    print(f"[eval] base_model={model_path}")
    if adapter_dir:
        print(f"[eval] adapter_dir={adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    else:
        print("[eval] adapter_dir=(none; zero-shot base model)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    value = value.rstrip(".")
    if value.startswith("="):
        value = value[1:].strip()
    return value


def as_decimal(text: str | None) -> Decimal | None:
    value = normalize_numeric_text(text)
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def numeric_equal(gold: str, pred: str) -> bool:
    gold_value = as_decimal(gold)
    pred_value = as_decimal(pred)
    if gold_value is None or pred_value is None:
        return normalize_numeric_text(gold) == normalize_numeric_text(pred)
    return abs(gold_value - pred_value) <= Decimal("1e-6")


def extract_numeric_answer(output: str) -> str:
    patterns = [
        r"the final answer is\s*:?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"the answer is\s*:?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\b([-+]?\d[\d,]*(?:\.\d+)?)\b",
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
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].upper()
    return ""


def load_existing_summary(output_dir: Path) -> dict:
    path = output_dir / "summary.json"
    if not path.exists():
        return {"datasets": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(output_dir: Path, dataset: str, correct: int, total: int, run_name: str | None) -> None:
    summary = load_existing_summary(output_dir)
    summary["run_name"] = run_name or summary.get("run_name", "")
    summary.setdefault("datasets", {})[dataset] = {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    rows = load_data(args)
    print(f"[eval] dataset={args.dataset} size={len(rows)}")
    tokenizer, model, device = load_model(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / f"{args.dataset}.json"
    generation_config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_beams=args.num_beams,
        do_sample=False,
    )

    correct = 0
    seen = 0
    output_rows: list[dict] = []
    batches = [rows[i : i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    for batch_index, batch in enumerate(tqdm(batches, desc=f"Evaluating {args.dataset}")):
        prompts = [generate_prompt(args.dataset, item["instruction"], item.get("input") or "") for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_width = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                generation_config=generation_config,
                max_new_tokens=args.max_new_tokens,
            )
        outputs = [
            tokenizer.decode(seq[input_width:], skip_special_tokens=True).strip()
            for seq in generated
        ]

        for item, output in zip(batch, outputs):
            gold = str(item.get("answer") or "").strip()
            if args.dataset == "aqua":
                pred = extract_choice_answer(output)
                flag = pred == gold.upper()
            else:
                pred = extract_numeric_answer(output)
                flag = numeric_equal(gold, pred)
            correct += int(flag)
            seen += 1
            output_rows.append(
                {
                    "id": item.get("id", ""),
                    "instruction": item.get("instruction", ""),
                    "answer": gold,
                    "output_pred": output,
                    "pred": pred,
                    "flag": flag,
                }
            )

        print(f"  {batch_index + 1}/{len(batches)} | accuracy: {correct}/{seen} = {correct / max(seen, 1):.4f}", flush=True)
        with save_file.open("w", encoding="utf-8") as handle:
            json.dump(output_rows, handle, ensure_ascii=False, indent=2)

    write_summary(output_dir, args.dataset, correct, seen, args.run_name)
    print(f"\nFinal accuracy: {correct}/{seen} = {correct / max(seen, 1):.4f}")


if __name__ == "__main__":
    main()
