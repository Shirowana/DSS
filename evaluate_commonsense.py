from __future__ import annotations

import argparse
import copy
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

MODEL_MAP = {
    "Llama2-7B": str(REMOTE_MODEL_ROOT / "Llama-2-7b-hf"),
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DSS adapter on commonsense reasoning benchmarks.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to a saved PEFT adapter directory.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=str(REMOTE_DATA_ROOT / "evaluate"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=0, help="Evaluate at most this many examples; 0 means full dataset.")
    parser.add_argument("--debug_eval", action="store_true")
    parser.add_argument("--debug_first_n", type=int, default=10)
    parser.add_argument("--debug_max_failures", type=int, default=20)
    return parser.parse_args()


def resolve_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


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
        dataset = json.load(handle)
    if args.max_examples and args.max_examples > 0:
        dataset = dataset[: args.max_examples]
    return dataset


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


def normalize_label(dataset: str, text: str | None) -> str:
    if text is None:
        return ""
    raw = text.strip()
    s = raw.lower()
    if dataset == "boolq":
        matches = re.findall(r"\b(true|false)\b", raw, flags=re.IGNORECASE)
        matches = [match.lower() for match in matches]
        return matches[-1] if matches else ""

    patterns = {
        "piqa": [r"\b(solution|choice)\s*([12])\b", r"\b([12])\b"],
        "social_i_qa": [r"\b(answer|choice)\s*([123])\b", r"\b([123])\b"],
        "hellaswag": [r"\b(ending|choice)\s*([1234])\b", r"\b([1234])\b"],
        "winogrande": [r"\b(option|choice)\s*([12])\b", r"\b([12])\b"],
        "ARC-Challenge": [r"\b(answer|choice)\s*([12345])\b", r"\b([12345])\b"],
        "ARC-Easy": [r"\b(answer|choice)\s*([12345])\b", r"\b([12345])\b"],
        "openbookqa": [r"\b(answer|choice)\s*([12345])\b", r"\b([12345])\b"],
    }

    for pattern in patterns.get(dataset, []):
        matches = re.findall(pattern, s)
        if matches:
            last = matches[-1]
            if isinstance(last, tuple):
                return f"choice{last[1]}"
            return f"choice{last}"
    return ""


def extract_answer(dataset: str, sentence: str) -> str:
    return normalize_label(dataset, sentence)


def debug_candidate_variants(dataset: str) -> dict[str, list[str]]:
    if dataset == "boolq":
        return {
            "true": ["true", " true", "\ntrue", "True", " True", "\nTrue"],
            "false": ["false", " false", "\nfalse", "False", " False", "\nFalse"],
        }
    if dataset in {"piqa", "winogrande"}:
        return {
            "1": ["1", " 1", "\n1"],
            "2": ["2", " 2", "\n2"],
        }
    if dataset == "social_i_qa":
        return {
            "1": ["1", " 1", "\n1"],
            "2": ["2", " 2", "\n2"],
            "3": ["3", " 3", "\n3"],
        }
    if dataset == "hellaswag":
        return {
            "1": ["1", " 1", "\n1"],
            "2": ["2", " 2", "\n2"],
            "3": ["3", " 3", "\n3"],
            "4": ["4", " 4", "\n4"],
        }
    return {
        "1": ["1", " 1", "\n1"],
        "2": ["2", " 2", "\n2"],
        "3": ["3", " 3", "\n3"],
        "4": ["4", " 4", "\n4"],
        "5": ["5", " 5", "\n5"],
    }


def build_debug_logits(tokenizer, dataset: str, first_step_scores: torch.Tensor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, variants in debug_candidate_variants(dataset).items():
        best = None
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue
            token_id = token_ids[0]
            logit = float(first_step_scores[token_id].item())
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            candidate = {
                "label": label,
                "variant": variant,
                "token_id": token_id,
                "decoded_token": decoded,
                "logit": logit,
            }
            if best is None or logit > best["logit"]:
                best = candidate
        if best is not None:
            rows.append(best)
    return rows


def build_topk_debug(tokenizer, first_step_scores: torch.Tensor, topk: int = 10) -> list[dict[str, object]]:
    values, indices = torch.topk(first_step_scores, k=min(topk, first_step_scores.shape[-1]))
    rows: list[dict[str, object]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        rows.append(
            {
                "token_id": int(index),
                "decoded_token": tokenizer.decode([int(index)], skip_special_tokens=False),
                "logit": float(value),
            }
        )
    return rows


def write_summary(output_dir: Path, dataset: str, correct: int, total: int, run_name: str | None) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {"run_name": run_name or "", "datasets": {}}

    accuracy = correct / max(total, 1)
    summary["run_name"] = run_name or summary.get("run_name", "")
    summary["datasets"][dataset] = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }
    accuracies = [float(stats["accuracy"]) for stats in summary["datasets"].values()]
    summary["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    dataset = load_data(args)
    print(f"[eval] dataset_file={Path(args.data_dir) / args.dataset / 'test.json'}")
    print(f"[eval] dataset_size={len(dataset)}")
    if args.max_examples and args.max_examples > 0:
        print(f"[eval] max_examples={args.max_examples}")
    tokenizer, model, device = load_model(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / f"{args.dataset}.json"

    batches = [dataset[i : i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=args.num_beams,
        do_sample=False,
    )

    correct = 0
    current = 0
    failures = 0
    debug_printed = 0
    output_data: list[dict] = []

    for idx, batch in enumerate(tqdm(batches, desc=f"Evaluating {args.dataset}")):
        current += len(batch)
        instructions = [data.get("instruction", "") for data in batch]
        input_texts = [data.get("input") or "" for data in batch]
        prompts = [generate_prompt(inst, input_text if input_text else None) for inst, input_text in zip(instructions, input_texts)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=args.debug_eval,
                max_new_tokens=args.max_new_tokens,
            )

        sequences = generation_output.sequences
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        outputs = []
        for i, seq in enumerate(sequences):
            gen_ids = seq[prompt_lens[i] :]
            outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        first_step_scores = None
        if args.debug_eval and generation_output.scores:
            first_step_scores = generation_output.scores[0]

        for item_idx, (data, output) in enumerate(zip(batch, outputs)):
            label = normalize_label(args.dataset, data.get("answer"))
            predict = extract_answer(args.dataset, output)
            flag = label == predict
            if flag:
                correct += 1
            elif args.debug_eval and failures < args.debug_max_failures:
                failures += 1

            if args.debug_eval and debug_printed < args.debug_first_n:
                sample_scores = None
                if first_step_scores is not None:
                    beam_row = item_idx * max(args.num_beams, 1)
                    beam_row = min(beam_row, first_step_scores.shape[0] - 1)
                    sample_scores = first_step_scores[beam_row].detach().float().cpu()

                print("\n========== DEBUG SAMPLE ==========", flush=True)
                print(f"[debug] dataset={args.dataset}", flush=True)
                print(f"[debug] sample_index={debug_printed}", flush=True)
                print(f"[debug] gold_raw={data.get('answer')!r}", flush=True)
                print(f"[debug] gold_norm={label!r}", flush=True)
                print(f"[debug] pred_norm={predict!r}", flush=True)
                print(f"[debug] flag={flag}", flush=True)
                print("[debug] prompt:", flush=True)
                print(prompts[item_idx], flush=True)
                print("[debug] raw_output:", flush=True)
                print(output, flush=True)
                if sample_scores is not None:
                    print("[debug] candidate_first_token_logits:", flush=True)
                    print(
                        json.dumps(
                            build_debug_logits(tokenizer, args.dataset, sample_scores),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        flush=True,
                    )
                    print("[debug] top10_first_token_logits:", flush=True)
                    print(
                        json.dumps(
                            build_topk_debug(tokenizer, sample_scores, topk=10),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        flush=True,
                    )
                debug_printed += 1

            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)

        accuracy = correct / max(current, 1)
        print(f"  {idx + 1}/{len(batches)} | accuracy: {correct}/{current} = {accuracy:.4f}", flush=True)
        with save_file.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2, ensure_ascii=False)

    final_accuracy = correct / max(current, 1)
    write_summary(output_dir, args.dataset, correct, current, args.run_name)
    print(f"\nFinal accuracy: {correct}/{current} = {final_accuracy:.4f}")


if __name__ == "__main__":
    main()
