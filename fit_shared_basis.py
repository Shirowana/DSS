"""
Fit a DSS SharedBasisPack from a local pretrained causal LM.

This is the offline stage before DSS fine-tuning. It reads local model weights,
groups target Linear modules by suffix (q/k/v/up/down by default), fits one
shared basis per group, and saves a torch-serialized SharedBasisPack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from dss.shared_basis import SharedBasisPack, fit_shared_basis_for_group


REMOTE_MODEL_ROOT = Path("/data/home/7250091/date/hf_cache_models/models")
REMOTE_PROJECT_ROOT = Path("/data/home/7250091/date/DSS")

MODEL_MAP = {
    "Llama2-7B": str(REMOTE_MODEL_ROOT / "Llama2-7B"),
    "Llama2-13B": str(REMOTE_MODEL_ROOT / "Llama2-13B"),
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Llama3-8B"),
    "Llama3-3B": str(REMOTE_MODEL_ROOT / "Llama3-3B"),
    "Mistral-7B": str(REMOTE_MODEL_ROOT / "Mistral-7B"),
    "Qwen2.5-7B": str(REMOTE_MODEL_ROOT / "Qwen2.5-7B"),
}

MODULE_MAP = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "u": "up_proj",
    "d": "down_proj",
    "o": "o_proj",
    "g": "gate_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit DSS shared basis from local model weights.")
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--output_path", type=str, default=str(REMOTE_PROJECT_ROOT / "basis" / "llama3_8b_dss_basis.pt"))
    parser.add_argument("--target_modules", type=str, default="qkvud")
    parser.add_argument("--basis_offset", type=int, default=0)
    parser.add_argument("--basis_lr", type=float, default=0.01)
    parser.add_argument("--basis_iters", type=int, default=1000)
    parser.add_argument("--basis_device", type=str, default="cuda")
    parser.add_argument(
        "--max_layers_per_group",
        type=int,
        default=0,
        help="Debug/smoke-test limit. 0 means use all matched layers.",
    )
    return parser.parse_args()


def module_suffixes(target_modules: str) -> list[str]:
    suffixes = [MODULE_MAP[key] for key in target_modules if key in MODULE_MAP]
    if not suffixes:
        raise ValueError("`--target_modules` did not resolve to any known module suffixes.")
    return suffixes


def collect_group_weights(
    model: torch.nn.Module,
    suffix: str,
    device: torch.device,
    max_layers: int,
) -> list[torch.Tensor]:
    weights: list[torch.Tensor] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name != suffix and not name.endswith(f".{suffix}"):
            continue
        weights.append(module.weight.detach().to(device=device, dtype=torch.float32))
        if max_layers > 0 and len(weights) >= max_layers:
            break
    if not weights:
        raise ValueError(f"No Linear weights found for suffix {suffix!r}.")
    return weights


def main() -> None:
    args = parse_args()
    model_path = args.model_path or MODEL_MAP[args.model_name]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.basis_device if torch.cuda.is_available() or args.basis_device == "cpu" else "cpu")
    print(f"[basis] loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.model_cache_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    entries = {}
    for suffix in module_suffixes(args.target_modules):
        weights = collect_group_weights(model, suffix, device=device, max_layers=args.max_layers_per_group)
        entry = fit_shared_basis_for_group(
            group_name=suffix,
            weights=weights,
            offset=args.basis_offset,
            lr=args.basis_lr,
            max_iters=args.basis_iters,
        )
        entries[suffix] = entry
        del weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    SharedBasisPack(entries).save(output_path)
    print(f"[basis] saved shared basis to: {output_path}")


if __name__ == "__main__":
    main()
