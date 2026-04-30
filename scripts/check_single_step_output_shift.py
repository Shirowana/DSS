from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires `peft` to load DSS adapters.") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import peft.tuners.dss  # noqa: F401,E402
from evaluate_commonsense import MODEL_MAP, REMOTE_DATA_ROOT, REMOTE_MODEL_ROOT, REMOTE_PROJECT_ROOT


REMOTE_RESULTS_ROOT = REMOTE_PROJECT_ROOT / "results_commonsense"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump DeltaW/DeltaLambda/A_inv/B_inv matrices before and after one step.")
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"))
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument("--adapter_path", type=str, default=str(REMOTE_PROJECT_ROOT / "output" / "identity"))
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=str(REMOTE_DATA_ROOT / "commonsense_new" / "train_all_256_OnlyOutput_Llama3-8B"),
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr_override", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target_layer",
        type=str,
        default="model.layers.0.self_attn.q_proj",
        help="Canonical suffix for the target DSS layer, e.g. model.layers.0.self_attn.q_proj",
    )
    parser.add_argument("--output_pt", type=str, default=None)
    return parser.parse_args()


def resolve_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_train_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask = [torch.as_tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    labels = [torch.as_tensor(item["labels"], dtype=torch.long) for item in batch]
    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }


def mark_dss_coefficients_trainable(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if module.__class__.__name__ not in {"DSSLayer", "DSSLinear"}:
            continue
        for adapter_name in getattr(module, "active_adapters", []):
            if adapter_name in getattr(module, "coefficient", {}):
                module.coefficient[adapter_name].requires_grad_(True)


def resolve_learning_rate(adapter_path: Path, lr_override: float | None) -> float:
    if lr_override is not None:
        return float(lr_override)
    training_args_path = adapter_path / "training_args.json"
    if training_args_path.exists():
        try:
            payload = torch.load(training_args_path, map_location="cpu")
            if isinstance(payload, dict) and payload.get("lr") is not None:
                return float(payload["lr"])
        except Exception:
            pass
        try:
            import json

            with training_args_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("lr") is not None:
                return float(payload["lr"])
        except Exception:
            pass
    return 1e-4


def find_module_by_suffix(model: torch.nn.Module, suffix: str):
    for name, module in model.named_modules():
        if name == suffix or name.endswith(f".{suffix}"):
            return name, module
    return None, None


def load_model_and_tokenizer(args: argparse.Namespace, adapter_path: Path, device: torch.device):
    model_id = args.model_path or MODEL_MAP[args.model_name]
    torch_dtype = resolve_dtype(args.precision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.model_cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=args.model_cache_dir, torch_dtype=torch_dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device)
    mark_dss_coefficients_trainable(model)
    return tokenizer, model


def dense_core_from_layer(module, adapter_name: str) -> torch.Tensor:
    curr_count = int(module.runtime[adapter_name].curr_count)
    out_features = int(module.out_features)
    in_features = int(module.in_features)
    flat = torch.zeros(out_features * in_features, device=module.coefficient[adapter_name].device, dtype=torch.float32)
    if curr_count > 0:
        coeff = module.coefficient[adapter_name][:curr_count].detach().float()
        indices = module.coefficient_indices[adapter_name][:curr_count].long()
        flat = flat.scatter_add(0, indices, coeff)
    return flat.view(out_features, in_features)


def capture_layer_state(module, adapter_name: str) -> dict[str, object]:
    return {
        "delta_lambda": dense_core_from_layer(module, adapter_name).cpu(),
        "delta_w": module.get_delta_weight(adapter_name).detach().float().cpu().clone(),
    }


def matrix_summary(matrix: torch.Tensor) -> dict[str, float]:
    x = matrix.detach().float().cpu()
    abs_x = x.abs()
    return {
        "min": float(x.min().item()) if x.numel() else 0.0,
        "max": float(x.max().item()) if x.numel() else 0.0,
        "mean_abs": float(abs_x.mean().item()) if x.numel() else 0.0,
        "rms": float(torch.sqrt((x * x).mean()).item()) if x.numel() else 0.0,
    }


def run_single_step(model, batch: dict[str, torch.Tensor], lr: float) -> float:
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr)
    batch = {key: value.to(next(model.parameters()).device) for key, value in batch.items()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("Expected model loss during training step, got None.")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().item())


def build_output_path(output_pt: str | None) -> Path:
    if output_pt:
        return Path(output_pt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REMOTE_RESULTS_ROOT / f"matrix_dump_{timestamp}.pt"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_path = Path(args.adapter_path)
    tokenizer, model = load_model_and_tokenizer(args, adapter_path, device)

    layer_name, module = find_module_by_suffix(model, args.target_layer)
    if module is None or module.__class__.__name__ not in {"DSSLayer", "DSSLinear"}:
        raise ValueError(f"Could not find DSS target layer matching {args.target_layer!r}.")

    adapter_names = [adapter for adapter in getattr(module, "active_adapters", []) if adapter in module.coefficient]
    if not adapter_names:
        raise ValueError(f"Target layer {layer_name} does not expose an active DSS adapter.")
    adapter_name = adapter_names[0]

    train_dataset = load_from_disk(args.train_dataset_path)
    train_batch_raw = [train_dataset[i] for i in range(min(args.train_batch_size, len(train_dataset)))]
    if not train_batch_raw:
        raise ValueError("Training dataset is empty; cannot run single-step update experiment.")
    train_batch = collate_train_batch(train_batch_raw, tokenizer.pad_token_id)

    before = capture_layer_state(module, adapter_name)
    basis_state = {
        "A_inv": module.basis_A_inv[adapter_name].detach().float().cpu().clone(),
        "B_inv": module.basis_B_inv[adapter_name].detach().float().cpu().clone(),
    }
    lr = resolve_learning_rate(adapter_path, args.lr_override)
    train_step_loss = run_single_step(model, train_batch, lr)
    after = capture_layer_state(module, adapter_name)

    payload = {
        "model_path": args.model_path,
        "adapter_path": str(adapter_path),
        "layer_name": layer_name,
        "adapter_name": adapter_name,
        "learning_rate": lr,
        "train_step_loss": train_step_loss,
        "basis": basis_state,
        "before": before,
        "after": after,
        "stats": {
            "A_inv": matrix_summary(basis_state["A_inv"]),
            "B_inv": matrix_summary(basis_state["B_inv"]),
            "before_delta_lambda": matrix_summary(before["delta_lambda"]),
            "after_delta_lambda": matrix_summary(after["delta_lambda"]),
            "before_delta_w": matrix_summary(before["delta_w"]),
            "after_delta_w": matrix_summary(after["delta_w"]),
        },
    }

    output_path = build_output_path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved matrix dump to {output_path}")
    print(f"layer_name={layer_name}")
    print(f"adapter_name={adapter_name}")
    print(f"train_step_loss={train_step_loss:.6f}")
    print(f"A_inv shape={tuple(basis_state['A_inv'].shape)} stats={payload['stats']['A_inv']}")
    print(f"B_inv shape={tuple(basis_state['B_inv'].shape)} stats={payload['stats']['B_inv']}")
    print(f"before_delta_lambda shape={tuple(before['delta_lambda'].shape)} stats={payload['stats']['before_delta_lambda']}")
    print(f"after_delta_lambda shape={tuple(after['delta_lambda'].shape)} stats={payload['stats']['after_delta_lambda']}")
    print(f"before_delta_w shape={tuple(before['delta_w'].shape)} stats={payload['stats']['before_delta_w']}")
    print(f"after_delta_w shape={tuple(after['delta_w'].shape)} stats={payload['stats']['after_delta_w']}")


if __name__ == "__main__":
    main()
