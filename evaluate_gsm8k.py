from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from peft import PeftModel
from peft.tuners.dss import DSSConfig  # noqa: F401 - ensure DSS PEFT type is registered


REMOTE_DATA_ROOT = Path("/root/datasets")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")

MODEL_MAP = {
    "Llama2-7B": str(REMOTE_MODEL_ROOT / "Llama-2-7b-hf"),
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DSS adapter on GSM8K.")
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to a saved PEFT adapter directory.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=str(REMOTE_DATA_ROOT / "official" / "gsm8k" / "test.jsonl"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=0, help="Evaluate at most this many examples; 0 means full dataset.")
    return parser.parse_args()


def resolve_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def generate_prompt(question: str) -> str:
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "Solve the following grade-school math problem. Show your reasoning briefly, and end with a final line in the form 'The answer is <number>'.\n\n"
        "### Input:\n"
        f"{question}\n\n"
        "### Response:\n"
    )


def load_data(args: argparse.Namespace) -> list[dict]:
    file_path = Path(args.data_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    rows: list[dict] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.max_examples and args.max_examples > 0:
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
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter directory does not contain adapter_config.json: {path}")
    return str(path)


def load_model(args: argparse.Namespace):
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)

    model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR")
    load_name = args.model_path or MODEL_MAP[args.model_name]
    torch_dtype = resolve_precision(args.precision)

    tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=torch_dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None and len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    adapter_dir = resolve_adapter_dir(args.adapter_path)
    print(f"[eval] base_model={load_name}")
    print(f"[eval] model_cache_dir={model_dir}")
    if adapter_dir:
        print(f"[eval] adapter_dir={adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        print("[eval] merged adapter into base model for inference")
    else:
        print("[eval] adapter_dir=(none; zero-shot base model)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def extract_gold_answer(answer_text: str) -> str:
    if "####" not in answer_text:
        raise ValueError(f"GSM8K answer is missing #### delimiter: {answer_text!r}")
    return normalize_numeric_text(answer_text.split("####")[-1])


def normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = text.strip()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("%", "")
    normalized = normalized.rstrip(".")
    if normalized.startswith("="):
        normalized = normalized[1:].strip()
    return normalized


def extract_pred_answer(output: str) -> str:
    patterns = [
        r"the answer is\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\b([-+]?\d[\d,]*(?:\.\d+)?)\b\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            return normalize_numeric_text(matches[-1])
    return ""


def write_summary(output_dir: Path, correct: int, total: int, run_name: str | None) -> None:
    summary_path = output_dir / "summary.json"
    accuracy = correct / max(total, 1)
    payload = {
        "run_name": run_name or "",
        "dataset": "gsm8k",
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    dataset = load_data(args)
    print(f"[eval] dataset_file={Path(args.data_file)}")
    print(f"[eval] dataset_size={len(dataset)}")
    if args.max_examples and args.max_examples > 0:
        print(f"[eval] max_examples={args.max_examples}")

    tokenizer, model, device = load_model(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / "gsm8k.json"

    batches = [dataset[i : i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]
    generation_config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_beams=args.num_beams,
        do_sample=False,
    )

    correct = 0
    seen = 0
    output_data: list[dict] = []

    for idx, batch in enumerate(tqdm(batches, desc="Evaluating gsm8k")):
        prompts = [generate_prompt(item["question"]) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                generation_config=generation_config,
                max_new_tokens=args.max_new_tokens,
            )

        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        outputs = []
        for i, seq in enumerate(generation_output):
            gen_ids = seq[prompt_lens[i] :]
            outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        for data, output in zip(batch, outputs):
            gold = extract_gold_answer(data["answer"])
            pred = extract_pred_answer(output)
            flag = gold == pred
            if flag:
                correct += 1
            seen += 1

            output_data.append(
                {
                    "question": data["question"],
                    "answer": gold,
                    "output_pred": output,
                    "pred": pred,
                    "flag": flag,
                }
            )

        accuracy = correct / max(seen, 1)
        print(f"  {idx + 1}/{len(batches)} | accuracy: {correct}/{seen} = {accuracy:.4f}", flush=True)
        with save_file.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2, ensure_ascii=False)

    final_accuracy = correct / max(seen, 1)
    write_summary(output_dir, correct, seen, args.run_name)
    print(f"\nFinal accuracy: {correct}/{seen} = {final_accuracy:.4f}")


if __name__ == "__main__":
    main()
