"""
Diagnose DSS shared-basis scales on a local pretrained causal LM.

This script is intended for the remote container. It reads an existing DSS
shared-basis .pt file, loads base model weights, and reports:

- W0 scale for target modules.
- A W0 B scale in core space.
- A/B/A_inv/B_inv matrix scales and approximate spectral amplification.
- Probe Delta W = A_inv @ lambda @ B_inv scales for sparse lambda probes.

It does not modify model files or training outputs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM


REMOTE_PROJECT_ROOT = Path("/data/home/7250091/date/DSS")
REMOTE_MODEL_ROOT = Path("/data/home/7250091/date/hf_cache_models/models")

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent
for candidate_root in (LOCAL_PROJECT_ROOT, REMOTE_PROJECT_ROOT):
    if candidate_root.exists() and str(candidate_root) not in sys.path:
        sys.path.insert(0, str(candidate_root))

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
    parser = argparse.ArgumentParser(description="Diagnose DSS shared-basis numerical scales.")
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory.")
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))
    parser.add_argument(
        "--shared_basis_path",
        type=str,
        default=str(REMOTE_PROJECT_ROOT / "basis" / "llama3_8b_dss_basis.pt"),
    )
    parser.add_argument("--target_modules", type=str, default="qkv")
    parser.add_argument(
        "--max_layers_per_group",
        type=int,
        default=1,
        help="How many matched layers per group to run W0/AW0B diagnostics on. 0 means all layers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for matrix products and spectral power iterations.",
    )
    parser.add_argument("--power_iters", type=int, default=40)
    parser.add_argument("--probe_slots", type=int, default=128)
    parser.add_argument(
        "--lambda_scales",
        type=str,
        default="1,1e-2,1e-4",
        help="Comma-separated coefficient scales for random sparse lambda probes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(REMOTE_PROJECT_ROOT / "logs_commonsense"),
        help="Directory for the diagnosis log file.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional explicit log file path. Defaults to log_dir/dss_basis_scale_<timestamp>.log.",
    )
    return parser.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_logging(args: argparse.Namespace):
    log_path = Path(args.log_file) if args.log_file else Path(args.log_dir) / f"dss_basis_scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, handle)
    sys.stderr = Tee(sys.__stderr__, handle)
    print(f"[diagnose] log_file={log_path}")
    return handle


def module_suffixes(target_modules: str) -> list[str]:
    suffixes = [MODULE_MAP[key] for key in target_modules if key in MODULE_MAP]
    if not suffixes:
        raise ValueError("`--target_modules` did not resolve to any known module suffixes.")
    return suffixes


def load_basis_entries(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Shared basis file must contain a dictionary-like payload.")
    entries = payload.get("entries", payload.get("groups", payload))
    if not isinstance(entries, dict):
        raise ValueError("Shared basis payload must contain entries/groups.")
    return entries


def matched_linear_weights(
    model: torch.nn.Module,
    suffix: str,
    max_layers: int,
) -> list[tuple[str, torch.Tensor]]:
    weights: list[tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name != suffix and not name.endswith(f".{suffix}"):
            continue
        weights.append((name, module.weight.detach().cpu().float()))
        if max_layers > 0 and len(weights) >= max_layers:
            break
    if not weights:
        raise ValueError(f"No Linear weights found for suffix {suffix!r}.")
    return weights


def tensor_stats(label: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    finite = torch.isfinite(x)
    finite_count = int(finite.sum().item())
    total = x.numel()
    if finite_count == 0:
        print(f"  {label:16s} finite=0/{total}")
        return {"abs_max": float("nan"), "rms": float("nan")}

    values = x[finite]
    abs_values = values.abs()
    qs = torch.quantile(abs_values.flatten(), torch.tensor([0.5, 0.9, 0.99, 0.999]))
    rms = torch.sqrt((values * values).mean())
    stats = {
        "abs_mean": abs_values.mean().item(),
        "rms": rms.item(),
        "p50": qs[0].item(),
        "p90": qs[1].item(),
        "p99": qs[2].item(),
        "p999": qs[3].item(),
        "abs_max": abs_values.max().item(),
        "fro": torch.linalg.norm(values).item(),
    }
    print(
        f"  {label:16s} shape={tuple(x.shape)} finite={finite_count}/{total} "
        f"abs_mean={stats['abs_mean']:.4e} rms={stats['rms']:.4e} "
        f"p50={stats['p50']:.4e} p90={stats['p90']:.4e} "
        f"p99={stats['p99']:.4e} p999={stats['p999']:.4e} "
        f"abs_max={stats['abs_max']:.4e} fro={stats['fro']:.4e}"
    )
    return stats


@torch.no_grad()
def spectral_norm_power(x: torch.Tensor, device: torch.device, iters: int) -> float:
    x = x.detach().float().to(device)
    v = torch.randn(x.shape[1], device=device)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(iters):
        u = x @ v
        u = u / u.norm().clamp_min(1e-12)
        v = x.T @ u
        v = v / v.norm().clamp_min(1e-12)
    return (x @ v).norm().item()


def print_basis_stats(group: str, raw: dict, device: torch.device, power_iters: int) -> tuple[torch.Tensor, ...]:
    A = raw["A"].detach().float()
    B = raw["B"].detach().float()
    A_inv = raw.get("A_inv")
    B_inv = raw.get("B_inv")
    A_inv = torch.linalg.inv(A) if A_inv is None else A_inv.detach().float()
    B_inv = torch.linalg.inv(B) if B_inv is None else B_inv.detach().float()
    shape = tuple(raw.get("shape", tuple(A.shape)))
    p, n = shape
    A_inv_target_fro = float(p) ** 0.5
    B_inv_target_fro = float(n) ** 0.5
    inverse_normalization = raw.get("inverse_normalization", "<none>")
    A_inv_scale = raw.get("A_inv_scale", None)
    B_inv_scale = raw.get("B_inv_scale", None)

    print(f"\n[group] {group}")
    print(
        f"  {'metadata':16s} shape={shape} inverse_normalization={inverse_normalization} "
        f"A_inv_scale={A_inv_scale} B_inv_scale={B_inv_scale}"
    )
    tensor_stats("A", A)
    tensor_stats("B", B)
    A_inv_stats = tensor_stats("A_inv", A_inv)
    B_inv_stats = tensor_stats("B_inv", B_inv)
    print(
        f"  {'identity_fro':16s} A_inv_fro/target={A_inv_stats['fro']:.4e}/{A_inv_target_fro:.4e} "
        f"ratio={A_inv_stats['fro'] / max(A_inv_target_fro, 1e-30):.4e}"
    )
    print(
        f"  {'identity_fro':16s} B_inv_fro/target={B_inv_stats['fro']:.4e}/{B_inv_target_fro:.4e} "
        f"ratio={B_inv_stats['fro'] / max(B_inv_target_fro, 1e-30):.4e}"
    )

    A_norm = spectral_norm_power(A, device, power_iters)
    A_inv_norm = spectral_norm_power(A_inv, device, power_iters)
    B_norm = spectral_norm_power(B, device, power_iters)
    B_inv_norm = spectral_norm_power(B_inv, device, power_iters)
    print(f"  {'spectral':16s} ||A||2≈{A_norm:.4e}, ||A_inv||2≈{A_inv_norm:.4e}, cond(A)≈{A_norm * A_inv_norm:.4e}")
    print(f"  {'spectral':16s} ||B||2≈{B_norm:.4e}, ||B_inv||2≈{B_inv_norm:.4e}, cond(B)≈{B_norm * B_inv_norm:.4e}")
    return A, B, A_inv, B_inv


def print_w0_and_core_stats(
    weights: Iterable[tuple[str, torch.Tensor]],
    A: torch.Tensor,
    B: torch.Tensor,
    device: torch.device,
) -> float:
    A_dev = A.to(device)
    B_dev = B.to(device)
    w0_rms_values: list[float] = []
    for name, W0 in weights:
        print(f"\n  [layer] {name}")
        w0_stats = tensor_stats("W0", W0)
        w0_rms_values.append(w0_stats["rms"])
        W_dev = W0.to(device)
        core = A_dev @ W_dev @ B_dev
        core_stats = tensor_stats("A W0 B", core.cpu())
        ratio = core_stats["rms"] / max(w0_stats["rms"], 1e-30)
        print(f"  {'core/W0 rms':16s} {ratio:.4e}")
        del W_dev, core
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not w0_rms_values:
        return float("nan")
    reference_rms = sum(w0_rms_values) / len(w0_rms_values)
    print(f"  {'W0 rms ref':16s} {reference_rms:.4e}")
    return reference_rms


def make_sparse_probe_pattern(
    out_features: int,
    in_features: int,
    slots: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_numel = out_features * in_features
    slots = min(slots, dense_numel)
    if slots <= 0:
        return torch.empty(0, device=device, dtype=torch.long), torch.empty(0, device=device)

    flat = torch.randperm(dense_numel, device=device)[:slots]
    signs = torch.randint(0, 2, (slots,), device=device, dtype=torch.float32).mul_(2).sub_(1)
    return flat, signs


def sparse_delta_probe(
    A_inv: torch.Tensor,
    B_inv: torch.Tensor,
    flat_indices: torch.Tensor,
    signs: torch.Tensor,
    coeff_scale: float,
    reference_w0_rms: float,
    device: torch.device,
) -> None:
    in_features = B_inv.shape[1]
    if flat_indices.numel() == 0:
        return

    rows = torch.div(flat_indices, in_features, rounding_mode="floor")
    cols = flat_indices.remainder(in_features)
    coeff = signs * float(coeff_scale)

    A_cols = A_inv.to(device)[:, rows]
    B_rows = B_inv.to(device)[cols, :]
    delta_w = torch.einsum("k,ok,ki->oi", coeff, A_cols, B_rows)
    print(f"\n  [probe lambda] random sparse slots={flat_indices.numel()}, coeff_abs={coeff_scale:.4e}")
    delta_stats = tensor_stats("Delta W", delta_w.cpu())
    print(f"  {'DeltaW/coeff rms':16s} {delta_stats['rms'] / max(abs(coeff_scale), 1e-30):.4e}")
    if reference_w0_rms == reference_w0_rms and reference_w0_rms > 0:
        print(f"  {'DeltaW/W0 rms':16s} {delta_stats['rms'] / reference_w0_rms:.4e}")
    del rows, cols, coeff, A_cols, B_rows, delta_w
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    _log_handle = setup_logging(args)
    torch.manual_seed(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device

    model_path = args.model_path or MODEL_MAP[args.model_name]
    suffixes = module_suffixes(args.target_modules)
    lambda_scales = [float(item) for item in args.lambda_scales.split(",") if item.strip()]

    print(f"[diagnose] model_path={model_path}")
    print(f"[diagnose] shared_basis_path={args.shared_basis_path}")
    print(f"[diagnose] target_modules={suffixes}")
    print(f"[diagnose] max_layers_per_group={args.max_layers_per_group}")
    print(f"[diagnose] device={device}")

    entries = load_basis_entries(args.shared_basis_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.model_cache_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    for suffix in suffixes:
        if suffix not in entries:
            raise KeyError(f"Basis group {suffix!r} not found. Available groups: {sorted(entries)}")
        A, B, A_inv, B_inv = print_basis_stats(suffix, entries[suffix], device, args.power_iters)
        weights = matched_linear_weights(model, suffix, args.max_layers_per_group)
        reference_w0_rms = print_w0_and_core_stats(weights, A, B, device)
        flat_indices, signs = make_sparse_probe_pattern(
            out_features=A_inv.shape[0],
            in_features=B_inv.shape[1],
            slots=args.probe_slots,
            device=device,
        )
        for scale in lambda_scales:
            sparse_delta_probe(A_inv, B_inv, flat_indices, signs, scale, reference_w0_rms, device)


if __name__ == "__main__":
    main()
