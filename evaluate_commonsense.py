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
import math
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

import peft.tuners.dss  # noqa: F401,E402 - importing registers the DSS PEFT method


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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--debug_eval", action="store_true", help="Enable low-noise evaluation debug logging.")
    parser.add_argument("--debug_first_n", type=int, default=5, help="Always print debug details for the first N examples.")
    parser.add_argument("--debug_print_failures", action="store_true", default=True, help="Print debug details for failed examples.")
    parser.add_argument("--debug_max_failures", type=int, default=20, help="Maximum number of failed examples to print.")
    return parser.parse_args()


def resolve_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def generate_prompt(instruction: str, input_text: str | None = None) -> str:
    return f"{instruction}\n"


def load_data(args: argparse.Namespace) -> list[dict]:
    file_path = Path(args.data_dir) / args.dataset / "test.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    print(f"[eval] loading dataset: {file_path}")
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
    model.config.use_cache = True

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        dss_layers_before = count_dss_layers(model)
        print(f"[eval] loaded adapter: {args.adapter_path}")
        print(f"[eval] DSS layers before merge: {dss_layers_before}")
        merge_debug_summary = None
        if args.debug_eval:
            merge_debug_summary = collect_merge_debug_summary(model)
            print_merge_debug_summary(merge_debug_summary)
        print("[eval] merging DSS adapter into base weights...")
        model = model.merge_and_unload()
        print(f"[eval] DSS layers after merge: {count_dss_layers(model)}")
        if args.debug_eval and merge_debug_summary is not None:
            validate_merged_weights(model, merge_debug_summary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[eval] model.config.use_cache={model.config.use_cache}")
    return tokenizer, model


def count_dss_layers(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if module.__class__.__name__ in {"DSSLayer", "DSSLinear"})


def normalize_label(dataset: str, text: str | None) -> str:
    if text is None:
        return ""
    sentence = text.strip().lower()
    if dataset == "boolq":
        if re.search(r"\btrue\b", sentence):
            return "true"
        if re.search(r"\bfalse\b", sentence):
            return "false"
        return ""

    matches = re.findall(r"\b(?:solution|answer|option|ending)\s*([1-5])\b", sentence)
    if matches:
        return f"choice{matches[-1]}"
    return ""


def extract_answer(dataset: str, sentence: str) -> str:
    return normalize_label(dataset, sentence)


def tensor_rms(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return tensor.detach().float().square().mean().sqrt().item()


def tensor_abs_max(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return tensor.detach().float().abs().max().item()


def find_module_by_suffix(model: torch.nn.Module, suffix: str) -> tuple[str, torch.nn.Module] | tuple[None, None]:
    for name, module in model.named_modules():
        if name == suffix or name.endswith(f".{suffix}"):
            return name, module
    return None, None


def adapter_keyword_present(dataset: str, text: str) -> bool:
    sentence = text.strip().lower()
    if dataset == "boolq":
        return bool(re.search(r"\btrue\b|\bfalse\b", sentence))
    return bool(re.search(r"\b(?:solution|answer|option|ending)\s*[1-5]\b", sentence))


def collect_merge_debug_summary(model: torch.nn.Module) -> dict[str, object]:
    suffixes = (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    )
    layer_summaries: list[dict[str, object]] = []
    active_counts: list[int] = []
    coeff_abs_max_values: list[float] = []
    coeff_rms_values: list[float] = []

    for _name, module in model.named_modules():
        if module.__class__.__name__ not in {"DSSLayer", "DSSLinear"}:
            continue
        for adapter_name in getattr(module, "active_adapters", []):
            if adapter_name not in getattr(module, "coefficient", {}):
                continue
            runtime = module.runtime[adapter_name]
            curr_count = int(runtime.curr_count)
            active_counts.append(curr_count)
            if curr_count > 0:
                coeff = module.coefficient[adapter_name][:curr_count]
                coeff_abs_max_values.append(tensor_abs_max(coeff))
                coeff_rms_values.append(tensor_rms(coeff))

    for suffix in suffixes:
        name, module = find_module_by_suffix(model, suffix)
        if module is None or module.__class__.__name__ not in {"DSSLayer", "DSSLinear"}:
            continue
        adapter_names = [adapter for adapter in getattr(module, "active_adapters", []) if adapter in module.coefficient]
        if not adapter_names:
            continue
        adapter_name = adapter_names[0]
        runtime = module.runtime[adapter_name]
        curr_count = int(runtime.curr_count)
        coeff = module.coefficient[adapter_name][:curr_count]
        delta_weight = module.get_delta_weight(adapter_name).detach().float()
        base_weight = module.get_base_layer().weight.detach().float()
        delta_rms = tensor_rms(delta_weight)
        base_rms = tensor_rms(base_weight)
        layer_summaries.append(
            {
                "name": name,
                "curr_count": curr_count,
                "coefficient_abs_max": tensor_abs_max(coeff),
                "coefficient_rms": tensor_rms(coeff),
                "delta_abs_max": tensor_abs_max(delta_weight),
                "delta_rms": delta_rms,
                "base_abs_max": tensor_abs_max(base_weight),
                "base_rms": base_rms,
                "delta_over_base_rms": delta_rms / max(base_rms, 1e-12),
                "premerge_weight": base_weight.detach().cpu().clone(),
            }
        )

    num_layers = len(active_counts)
    total_active_slots = int(sum(active_counts))
    mean_active_slots = float(total_active_slots / num_layers) if num_layers else 0.0
    max_active_slots = int(max(active_counts)) if active_counts else 0
    global_coeff_abs_max = max(coeff_abs_max_values) if coeff_abs_max_values else 0.0
    global_coeff_rms = math.sqrt(
        sum(value * value for value in coeff_rms_values) / len(coeff_rms_values)
    ) if coeff_rms_values else 0.0

    return {
        "num_dss_layers": num_layers,
        "total_active_slots": total_active_slots,
        "mean_active_slots": mean_active_slots,
        "max_active_slots": max_active_slots,
        "global_coefficient_abs_max": global_coeff_abs_max,
        "global_coefficient_rms": global_coeff_rms,
        "layers": layer_summaries,
    }


def print_merge_debug_summary(summary: dict[str, object]) -> None:
    print(
        "[eval-debug][adapter-summary] "
        f"num_dss_layers={summary['num_dss_layers']} "
        f"total_active_slots={summary['total_active_slots']} "
        f"mean_active_slots={summary['mean_active_slots']:.2f} "
        f"max_active_slots={summary['max_active_slots']} "
        f"global_coefficient_abs_max={summary['global_coefficient_abs_max']:.4e} "
        f"global_coefficient_rms={summary['global_coefficient_rms']:.4e}"
    )
    for layer in summary["layers"]:
        print(
            "[eval-debug][adapter-layer] "
            f"name={layer['name']} "
            f"curr_count={layer['curr_count']} "
            f"coefficient_abs_max={layer['coefficient_abs_max']:.4e} "
            f"coefficient_rms={layer['coefficient_rms']:.4e} "
            f"delta_abs_max={layer['delta_abs_max']:.4e} "
            f"delta_rms={layer['delta_rms']:.4e} "
            f"base_abs_max={layer['base_abs_max']:.4e} "
            f"base_rms={layer['base_rms']:.4e} "
            f"delta_over_base_rms={layer['delta_over_base_rms']:.4e}"
        )


def validate_merged_weights(model: torch.nn.Module, summary: dict[str, object]) -> None:
    for layer in summary["layers"]:
        suffix = str(layer["name"])
        _name, merged_module = find_module_by_suffix(model, suffix)
        if merged_module is None or not hasattr(merged_module, "weight"):
            print(f"[eval-debug][merge-check] missing merged module for {suffix}")
            continue
        merged_weight = merged_module.weight.detach().float()
        premerge_weight = layer["premerge_weight"]
        diff = merged_weight - premerge_weight
        print(
            "[eval-debug][merge-check] "
            f"name={suffix} "
            f"merged_diff_abs_max={tensor_abs_max(diff):.4e} "
            f"merged_diff_rms={tensor_rms(diff):.4e} "
            f"delta_abs_max={layer['delta_abs_max']:.4e} "
            f"delta_rms={layer['delta_rms']:.4e}"
        )


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
        num_beams=args.num_beams,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    batches = [dataset[i : i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]
    correct = 0
    seen = 0
    output_data: list[dict] = []
    debug_failure_count = 0
    generated_token_lengths: list[int] = []
    hitting_max_new_tokens = 0
    empty_raw_output_count = 0
    empty_prediction_count = 0
    keyword_present_count = 0

    def should_print_debug(example_index: int, flag: bool) -> bool:
        nonlocal debug_failure_count
        if not args.debug_eval:
            return False
        if example_index < args.debug_first_n:
            return True
        if args.debug_print_failures and not flag and debug_failure_count < args.debug_max_failures:
            debug_failure_count += 1
            return True
        return False

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Evaluating {args.dataset}")):
        instructions = [item.get("instruction", "") for item in batch]
        prompts = [generate_prompt(instruction) for instruction in instructions]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.inference_mode():
            generation_output = model.generate(
                **tokenized,
                generation_config=generation_config,
                max_new_tokens=args.max_new_tokens,
            )

        prompt_lens = tokenized["attention_mask"].sum(dim=1).tolist()
        decoded = []
        decoded_token_lengths = []
        for seq, prompt_len in zip(generation_output, prompt_lens):
            gen_ids = seq[int(prompt_len) :]
            decoded_token_lengths.append(int(gen_ids.numel()))
            decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        for item, output, raw_token_len in zip(batch, decoded, decoded_token_lengths):
            label_raw = item.get("answer")
            label = normalize_label(args.dataset, label_raw)
            predict = extract_answer(args.dataset, output)
            flag = label == predict
            correct += int(flag)
            seen += 1
            generated_token_lengths.append(raw_token_len)
            hitting_max_new_tokens += int(raw_token_len >= args.max_new_tokens)
            empty_raw_output_count += int(output.strip() == "")
            empty_prediction_count += int(predict == "")
            keyword_present_count += int(adapter_keyword_present(args.dataset, output))

            row = copy.deepcopy(item)
            row["output_pred"] = output
            row["pred"] = predict
            row["flag"] = flag
            if args.debug_eval:
                row["debug"] = {
                    "gold_answer_raw": label_raw,
                    "gold_answer_normalized": label,
                    "raw_decoded_text": output,
                    "effective_answer": predict,
                    "raw_decoded_token_length": raw_token_len,
                }
            output_data.append(row)

            if should_print_debug(seen - 1, flag):
                print(
                    "[eval-debug][sample] "
                    f"index={seen - 1} "
                    f"flag={flag} "
                    f"gold_raw={json.dumps(label_raw, ensure_ascii=False)} "
                    f"gold_norm={label!r} "
                    f"effective_answer={predict!r} "
                    f"raw_token_len={raw_token_len}"
                )
                print(f"[eval-debug][instruction] {item.get('instruction', '')}")
                print(f"[eval-debug][raw_output] {output}")

        accuracy = correct / max(seen, 1)
        print(f"  {batch_idx + 1}/{len(batches)} | accuracy: {correct}/{seen} = {accuracy:.4f}")

        with save_file.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=4, ensure_ascii=False)

    final_accuracy = correct / max(seen, 1)
    with save_file.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=4, ensure_ascii=False)
    if args.debug_eval:
        avg_generated_len = sum(generated_token_lengths) / max(len(generated_token_lengths), 1)
        raw_empty_fraction = empty_raw_output_count / max(seen, 1)
        empty_fraction = empty_prediction_count / max(seen, 1)
        max_token_fraction = hitting_max_new_tokens / max(seen, 1)
        keyword_fraction = keyword_present_count / max(seen, 1)
        print(
            "[eval-debug][generation-summary] "
            f"average_generated_token_length={avg_generated_len:.2f} "
            f"fraction_hitting_max_new_tokens={max_token_fraction:.4f} "
            f"fraction_empty_raw_output={raw_empty_fraction:.4f} "
            f"fraction_extract_answer_empty={empty_fraction:.4f} "
            f"fraction_keyword_present={keyword_fraction:.4f}"
        )
    print(f"\nFinal accuracy: {correct}/{seen} = {final_accuracy:.4f}")
    print(f"Saved predictions to {save_file}")


if __name__ == "__main__":
    main()
