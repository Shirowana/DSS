"""
LLM-Adapters-style commonsense evaluator adapted for DSS experiments.

This keeps the upstream prompt/eval template while adding:
- Qwen-family compatibility
- DoRA, DSS, and SHiRA compatibility through standard PEFT loading
- configurable data/output paths
- configurable beam size and precision
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.dss import DSSConfig  # noqa: F401 - register DSS before loading its adapter config
from peft.tuners.shira import ShiraConfig  # noqa: F401 - register SHiRA before loading its adapter config
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-Adapters-style commonsense evaluation.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_name", choices=["lora", "dora", "dss", "shira"], required=True)
    parser.add_argument("--adapter_weights", required=True)
    parser.add_argument("--shira_r", type=int, default=None)
    parser.add_argument("--shira_mask_type", choices=["random", "snip"], default=None)
    parser.add_argument("--shira_random_seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def generate_prompt(instruction: str, input_text: str | None = None) -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def load_data(args: argparse.Namespace) -> list[dict]:
    file_path = Path(args.data_dir) / args.dataset / "test.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_batch(dataset: list[dict], batch_size: int) -> list[list[dict]]:
    return [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]


def extract_answer(dataset: str, sentence: str) -> str:
    sentence = sentence.strip().lower()
    if dataset == "boolq":
        matches = re.findall(r"true|false", sentence)
        return matches[0] if matches else ""
    if dataset == "piqa":
        matches = re.findall(r"solution1|solution2", sentence)
        return matches[0] if matches else ""
    if dataset in {"social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"}:
        matches = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence)
        return matches[0] if matches else ""
    if dataset == "hellaswag":
        matches = re.findall(r"ending1|ending2|ending3|ending4", sentence)
        return matches[0] if matches else ""
    if dataset == "winogrande":
        matches = re.findall(r"option1|option2", sentence)
        return matches[0] if matches else ""
    return ""


def write_summary(output_dir: Path, dataset: str, correct: int, total: int) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {"datasets": {}}

    accuracy = correct / max(total, 1)
    summary["datasets"][dataset] = {"correct": correct, "total": total, "accuracy": accuracy}
    values = [float(stats["accuracy"]) for stats in summary["datasets"].values()]
    summary["average_accuracy"] = sum(values) / len(values) if values else 0.0

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_weights, torch_dtype=torch_dtype)
    if args.adapter_name == "shira":
        shira_config = next(iter(model.peft_config.values()))
        expected = {
            "r": args.shira_r,
            "mask_type": args.shira_mask_type,
            "random_seed": args.shira_random_seed,
        }
        for name, expected_value in expected.items():
            if expected_value is not None and getattr(shira_config, name) != expected_value:
                raise ValueError(
                    f"SHiRA {name} mismatch: CLI requested {expected_value!r}, "
                    f"adapter config contains {getattr(shira_config, name)!r}."
                )
    model.to(device)
    model.eval()

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=args.num_beams,
        do_sample=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / f"{args.dataset}.json"

    correct = 0
    current = 0
    output_data: list[dict] = []

    for idx, batch in enumerate(tqdm(batches, desc=f"Evaluating {args.dataset}")):
        current += len(batch)
        instructions = [data.get("instruction") for data in batch]
        input_texts = [data.get("input") or None for data in batch]
        prompts = [generate_prompt(instruction, input_text) for instruction, input_text in zip(instructions, input_texts)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=args.max_new_tokens,
            )

        outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        outputs = [output.split("### Response:")[-1].strip() for output in outputs]

        for data, output in zip(batch, outputs):
            label = data.get("answer", "")
            predict = extract_answer(args.dataset, output)
            flag = label == predict
            if flag:
                correct += 1

            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)

        print("---------------", flush=True)
        print(f"test:{idx + 1}/{len(batches)} | accuracy {correct}  {correct / max(current, 1):.4f}", flush=True)
        print("---------------", flush=True)
        with save_file.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2, ensure_ascii=False)

    final_accuracy = correct / max(current, 1)
    write_summary(output_dir, args.dataset, correct, current)
    print(f"\nFinal accuracy: {correct}/{current} = {final_accuracy:.4f}", flush=True)
    print("test finished", flush=True)


if __name__ == "__main__":
    main()
