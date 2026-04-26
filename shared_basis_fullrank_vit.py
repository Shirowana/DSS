"""
Full-rank shared-basis structural hypothesis test on ViT attention q weights.

This script is an offline structural analysis tool, not a training script.
Its purpose is to test a new PEFT hypothesis on pretrained ViT weights:

    W_i ~= U (Lambda_i + Delta_i) V^T

where:
- U, V are shared full-rank bases across same-type layers
- Lambda_i is the main per-layer structure, ideally close to diagonal
- Delta_i is a sparse non-diagonal correction

The current experiment focuses on pretrained attention q weights only.
It does not use datasets and it does not analyze fine-tuned Delta W yet.

Why `diagonalization.py` matters
--------------------------------
The existing `diagonalization.py` learns shared transforms A, B so that a list
of same-shaped matrices all become near-diagonal or near-banded under

    A @ W_i @ B.

Its loss is an outside-band / inside-band energy ratio controlled by `offset`,
so it is not a
reconstruction program; it is a numerical search for a common coordinate system
that makes many matrices simple at the same time. That idea directly inspires
this script: we also look for one shared coordinate system in which multiple
same-type matrices become jointly simple.

Why this script skips low-rank rank sweep
-----------------------------------------
The current target is not "low-rank shared subspace + dense core". The target
method is closer to "full-rank shared basis + structured core". Therefore the
main question is no longer:

    do low-dimensional shared subspaces exist?

Instead, the question is:

    under a shared full-rank basis, do the cores become simple enough that
    diagonal structure plus a small sparse off-diagonal correction is useful?

Why full-rank basis still needs optimization
--------------------------------------------
With full-rank U, V, any matrix family can be represented exactly by

    S_i = U^T W_i V.

So the question is not whether a full-rank basis exists, but whether there
exists a *good* full-rank basis whose cores are simple. This is non-trivial
because full-rank bases have an internal rotation ambiguity:

    U S_i V^T = (U R_u) (R_u^T S_i R_v) (V R_v)^T

for orthogonal R_u, R_v. Different shared bases inside the same full-rank space
can turn the same cores from dense-looking to near-diagonal-looking. Therefore
we explicitly optimize U, V so the cores jointly become simpler.

What this script measures
-------------------------
After learning shared U, V for the ViT q weights, the script computes

    S_i = U^T W_i V

and evaluates:
1. Main-band approximation
   - when `offset=0`, this is pure diagonal
   - when `offset>0`, this is the main diagonal plus nearby sub/super diagonals
2. Main-band + top-k off-band sparse correction
   - this is the most relevant metric for the proposed PEFT parameterization
   - if a small number of non-diagonal corrections sharply reduces error, the
     method looks promising
3. Off-band energy ratio
   - asks whether most energy is already concentrated inside the allowed band
4. Top-k sparse correction gain curve
   - asks how much error is recovered by only a few off-diagonal entries

Current scope
-------------
This version runs on pretrained ViT same-type weights selected by
`--weight-type`, currently supporting:
- q
- k
- v
- o
- up
- down

The code still leaves a natural extension point for future Delta W analysis,
but intentionally does not expand that mode yet. The present goal is to
validate the geometric hypothesis first, using fixed pretrained weights in an
offline setting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MATPLOTLIB = False
    plt = None


Tensor = torch.Tensor


@dataclass
class FitResult:
    U: Tensor
    V: Tensor
    cores: Tensor
    transformed: Tensor
    loss_history: List[float]
    init_mean_offdiag_ratio: float
    final_mean_offdiag_ratio: float
    init_diagonalization_ratio: float
    final_diagonalization_ratio: float
    init_mean_offdiag_norm: float
    final_mean_offdiag_norm: float
    init_mean_diag_norm: float
    final_mean_diag_norm: float


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stack_weights(weights: Sequence[Tensor], device: Optional[torch.device] = None) -> Tensor:
    if not weights:
        raise ValueError("weights must contain at least one matrix")
    shape = weights[0].shape
    for idx, weight in enumerate(weights):
        if weight.ndim != 2:
            raise ValueError(f"weights[{idx}] must be 2D, got {tuple(weight.shape)}")
        if weight.shape != shape:
            raise ValueError(
                f"all matrices must share one shape, got {tuple(weight.shape)} vs {tuple(shape)}"
            )
    target_device = device or weights[0].device
    return torch.stack([w.to(target_device, dtype=torch.float32) for w in weights], dim=0)


def orthogonalize_square(matrix: Tensor) -> Tensor:
    q, _ = torch.linalg.qr(matrix)
    return q


def diagonal_mask(rows: int, cols: int, offset: int = 0, device: Optional[torch.device] = None) -> Tensor:
    i = torch.arange(rows, device=device).view(-1, 1)
    j = torch.arange(cols, device=device).view(1, -1)
    return torch.abs(i - j) <= offset


def block_diagonal_mask(rows: int, cols: int, block_size: int, device: Optional[torch.device] = None) -> Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    i = torch.arange(rows, device=device).view(-1, 1)
    j = torch.arange(cols, device=device).view(1, -1)
    return (i // block_size) == (j // block_size)


def compute_cores(weights: Tensor, U: Tensor, V: Tensor) -> Tensor:
    return torch.matmul(U.transpose(0, 1), torch.matmul(weights, V))


def project_diagonal(core: Tensor) -> Tensor:
    mask = diagonal_mask(core.shape[0], core.shape[1], offset=0, device=core.device)
    return core * mask


def project_main_band(core: Tensor, offset: int) -> Tensor:
    mask = diagonal_mask(core.shape[0], core.shape[1], offset=offset, device=core.device)
    return core * mask


def project_block_diagonal(core: Tensor, block_size: int) -> Tensor:
    mask = block_diagonal_mask(core.shape[0], core.shape[1], block_size=block_size, device=core.device)
    return core * mask


def project_mainband_plus_topk(core: Tensor, offset: int, topk: int) -> Tuple[Tensor, Tensor]:
    rows, cols = core.shape
    band_keep = diagonal_mask(rows, cols, offset=offset, device=core.device)
    off_band = core.masked_fill(band_keep, 0.0)
    k = min(max(topk, 0), off_band.numel())
    if k == 0:
        keep_mask = band_keep
        return core * keep_mask, keep_mask
    flat_abs = off_band.abs().reshape(-1)
    topk_indices = torch.topk(flat_abs, k=k, largest=True).indices
    sparse_mask = torch.zeros_like(flat_abs, dtype=torch.bool)
    sparse_mask[topk_indices] = True
    sparse_mask = sparse_mask.view(rows, cols)
    keep_mask = band_keep | sparse_mask
    return core * keep_mask, keep_mask


def relative_matrix_error(original: Tensor, approximation: Tensor, eps: float = 1e-12) -> float:
    return float(
        (
            torch.linalg.matrix_norm(original - approximation, ord="fro")
            / torch.linalg.matrix_norm(original, ord="fro").clamp_min(eps)
        ).item()
    )


def retained_energy_ratio(original: Tensor, approximation: Tensor, eps: float = 1e-12) -> float:
    return float((approximation.pow(2).sum() / original.pow(2).sum().clamp_min(eps)).item())


def off_band_energy_ratio(core: Tensor, offset: int, eps: float = 1e-12) -> float:
    band = project_main_band(core, offset)
    off_band = core - band
    return float((off_band.pow(2).sum() / core.pow(2).sum().clamp_min(eps)).item())


def diagonalization_style_ratio(
    matrices: Tensor,
    offset: int,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Match diagonalization.py exactly in spirit:
    - off part: norm of entries outside the |i-j| <= offset mask
    - diag part: norm of torch.diag(matrix), even when offset > 0

    That means this is a ratio of means:
        mean(off-band norm) / mean(main diagonal norm)
    not a mean of per-matrix ratios.
    """
    rows, cols = matrices.shape[-2:]
    mask = diagonal_mask(rows, cols, offset=offset, device=matrices.device).float()
    off_norms = torch.stack([(matrix - matrix * mask).norm() for matrix in matrices])
    diag_norms = torch.stack([torch.diag(matrix).norm() for matrix in matrices])
    off_mean = off_norms.mean()
    diag_mean = diag_norms.mean()
    ratio = off_mean / diag_mean.clamp_min(eps)
    return float(ratio.item()), float(off_mean.item()), float(diag_mean.item())


def fit_shared_fullrank_basis(
    weights: Tensor,
    offset: int,
    lr: float,
    max_iters: int,
    seed: int,
    device: torch.device,
    log_every: int = 50,
) -> FitResult:
    """
    Learn one shared full-rank orthogonal basis U, V so that the transformed
    cores U^T W_i V become jointly simple.

    The loss mirrors diagonalization.py exactly in spirit:
    - define a band mask by |i-j| <= offset
    - minimize average outside-band energy / inside-band energy

    This does not test whether a full-rank basis exists; that is trivial.
    It tests whether there exists a *good* common coordinate system in which the
    ViT q-weight cores become simple.
    """
    set_seed(seed)
    dev = device
    weights = weights.to(dev)
    _, rows, cols = weights.shape
    if rows != cols:
        raise ValueError(f"full-rank square basis requires square matrices, got {(rows, cols)}")

    U_raw = torch.randn(rows, rows, device=dev, requires_grad=True)
    V_raw = torch.randn(cols, cols, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([U_raw, V_raw], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    history: List[float] = []

    mask = diagonal_mask(rows, cols, offset=offset, device=dev).float()

    with torch.no_grad():
        U0 = orthogonalize_square(U_raw)
        V0 = orthogonalize_square(V_raw)
        init_cores = compute_cores(weights, U0, V0)
        init_offdiag = sum(off_band_energy_ratio(core, offset=offset) for core in init_cores) / init_cores.shape[0]
        init_diag_ratio, init_off_mean, init_diag_mean = diagonalization_style_ratio(init_cores, offset=offset)

    for step in range(max_iters):
        optimizer.zero_grad()
        U = orthogonalize_square(U_raw)
        V = orthogonalize_square(V_raw)
        cores = compute_cores(weights, U, V)
        band_part = cores * mask
        offband_part = cores - band_part
        per_matrix_ratio = offband_part.pow(2).sum(dim=(-2, -1)) / band_part.pow(2).sum(dim=(-2, -1)).clamp_min(1e-12)
        loss = per_matrix_ratio.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == max_iters:
            current_ratio, current_off_mean, current_diag_mean = diagonalization_style_ratio(
                cores.detach(), offset=offset
            )
            print(
                f"[fit] step={step + 1:4d}/{max_iters} "
                f"offset={offset} "
                f"objective={loss_value:.6f} "
                f"current_a1_over_a2={current_ratio:.6f} "
                f"current_a1={current_off_mean:.6f} "
                f"current_a2={current_diag_mean:.6f}"
            )

    with torch.no_grad():
        U = orthogonalize_square(U_raw).detach()
        V = orthogonalize_square(V_raw).detach()
        cores = compute_cores(weights, U, V).detach()
        transformed = torch.matmul(U.transpose(0, 1), torch.matmul(weights, V)).detach()
        final_offdiag = sum(off_band_energy_ratio(core, offset=offset) for core in cores) / cores.shape[0]
        final_diag_ratio, final_off_mean, final_diag_mean = diagonalization_style_ratio(cores, offset=offset)

    return FitResult(
        U=U.cpu(),
        V=V.cpu(),
        cores=cores.cpu(),
        transformed=transformed.cpu(),
        loss_history=history,
        init_mean_offdiag_ratio=float(init_offdiag),
        final_mean_offdiag_ratio=float(final_offdiag),
        init_diagonalization_ratio=float(init_diag_ratio),
        final_diagonalization_ratio=float(final_diag_ratio),
        init_mean_offdiag_norm=float(init_off_mean),
        final_mean_offdiag_norm=float(final_off_mean),
        init_mean_diag_norm=float(init_diag_mean),
        final_mean_diag_norm=float(final_diag_mean),
    )


def analyze_structures(
    weights: Tensor,
    U: Tensor,
    V: Tensor,
    offset: int,
    topk_list: Sequence[int],
    block_size: int,
) -> Dict[str, Any]:
    weights_cpu = weights.cpu()
    U_cpu = U.cpu()
    V_cpu = V.cpu()
    cores = compute_cores(weights_cpu, U_cpu, V_cpu)
    per_layer: List[Dict[str, Any]] = []
    aggregate_topk: Dict[int, Dict[str, float]] = {
        int(k): {
            "core_relative_error_mean": 0.0,
            "weight_relative_error_mean": 0.0,
            "retained_core_energy_ratio_mean": 0.0,
        }
        for k in topk_list
    }

    for idx, core in enumerate(cores):
        layer_report: Dict[str, Any] = {
            "layer_index": idx,
            "off_band_energy_ratio": off_band_energy_ratio(core, offset=offset),
        }

        mainband_core = project_main_band(core, offset=offset)
        mainband_weight = U_cpu @ mainband_core @ V_cpu.transpose(0, 1)
        layer_report["main_band"] = {
            "offset": int(offset),
            "core_relative_error": relative_matrix_error(core, mainband_core),
            "weight_relative_error": relative_matrix_error(weights_cpu[idx], mainband_weight),
            "retained_core_energy_ratio": retained_energy_ratio(core, mainband_core),
        }

        topk_report: Dict[str, Dict[str, float]] = {}
        for k in topk_list:
            diag_topk_core, keep_mask = project_mainband_plus_topk(core, offset=offset, topk=int(k))
            diag_topk_weight = U_cpu @ diag_topk_core @ V_cpu.transpose(0, 1)
            metrics = {
                "core_relative_error": relative_matrix_error(core, diag_topk_core),
                "weight_relative_error": relative_matrix_error(weights_cpu[idx], diag_topk_weight),
                "retained_core_energy_ratio": retained_energy_ratio(core, diag_topk_core),
                "selected_off_band_entries": int(
                    keep_mask.logical_and(~diagonal_mask(core.shape[0], core.shape[1], offset=offset)).sum().item()
                ),
            }
            topk_report[str(int(k))] = metrics
            aggregate_topk[int(k)]["core_relative_error_mean"] += metrics["core_relative_error"]
            aggregate_topk[int(k)]["weight_relative_error_mean"] += metrics["weight_relative_error"]
            aggregate_topk[int(k)]["retained_core_energy_ratio_mean"] += metrics["retained_core_energy_ratio"]
        layer_report["diag_plus_topk"] = topk_report

        block_core = project_block_diagonal(core, block_size)
        block_weight = U_cpu @ block_core @ V_cpu.transpose(0, 1)
        layer_report["block_diagonal"] = {
            "block_size": int(block_size),
            "core_relative_error": relative_matrix_error(core, block_core),
            "weight_relative_error": relative_matrix_error(weights_cpu[idx], block_weight),
            "retained_core_energy_ratio": retained_energy_ratio(core, block_core),
        }

        per_layer.append(layer_report)

    num_layers = len(per_layer)
    for k in aggregate_topk:
        for metric_name in list(aggregate_topk[k].keys()):
            aggregate_topk[k][metric_name] /= num_layers

    summary = {
        "mean_off_band_energy_ratio": sum(layer["off_band_energy_ratio"] for layer in per_layer) / num_layers,
        "mean_main_band_core_relative_error": sum(layer["main_band"]["core_relative_error"] for layer in per_layer) / num_layers,
        "mean_main_band_weight_relative_error": sum(layer["main_band"]["weight_relative_error"] for layer in per_layer) / num_layers,
        "topk_gain_curve": aggregate_topk,
    }

    return {
        "cores": cores,
        "per_layer": per_layer,
        "summary": summary,
    }


def choose_representative_layers(per_layer: Sequence[Dict[str, Any]], vis_count: int) -> List[int]:
    ordered = sorted(per_layer, key=lambda item: item["off_band_energy_ratio"])
    selected: List[int] = []

    for item in ordered[:vis_count]:
        selected.append(int(item["layer_index"]))
    for item in ordered[-vis_count:]:
        if int(item["layer_index"]) not in selected:
            selected.append(int(item["layer_index"]))

    middle_start = max(0, len(ordered) // 2 - vis_count // 2)
    middle_end = min(len(ordered), middle_start + vis_count)
    for item in ordered[middle_start:middle_end]:
        if int(item["layer_index"]) not in selected:
            selected.append(int(item["layer_index"]))

    return selected


def save_heatmap(matrix: Tensor, title: str, path: Path, cmap: str = "coolwarm") -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.cpu().numpy(), cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_mask_plot(mask: Tensor, title: str, path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(mask.cpu().numpy(), cmap="gray_r", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_topk_curve(topk_curve: Dict[int, Dict[str, float]], path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    ks = sorted(topk_curve.keys())
    core_err = [topk_curve[k]["core_relative_error_mean"] for k in ks]
    weight_err = [topk_curve[k]["weight_relative_error_mean"] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, core_err, marker="o", label="core relative error")
    ax.plot(ks, weight_err, marker="s", label="weight relative error")
    ax.set_xlabel("top-k off-diagonal corrections")
    ax.set_ylabel("mean relative error")
    ax.set_title("Main-band + top-k sparse correction gain curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_visualizations(
    figures_dir: Path,
    cores: Tensor,
    per_layer: Sequence[Dict[str, Any]],
    offset: int,
    topk_list: Sequence[int],
    vis_count: int,
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib is not available; skipping figure generation.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    representative_indices = choose_representative_layers(per_layer, vis_count)
    representative_topk = int(max(topk_list)) if topk_list else 0

    for idx in representative_indices:
        core = cores[idx]
        diagonal_core = project_main_band(core, offset=offset)
        diag_topk_core, keep_mask = project_mainband_plus_topk(core, offset=offset, topk=representative_topk)
        residual = core - diagonal_core
        offdiag_keep_mask = keep_mask.logical_and(
            ~diagonal_mask(core.shape[0], core.shape[1], offset=offset, device=keep_mask.device)
        )

        prefix = figures_dir / f"layer_{idx:02d}"
        save_heatmap(core, f"Layer {idx} core", prefix.with_name(prefix.name + "_core.png"))
        save_heatmap(
            diagonal_core,
            f"Layer {idx} main-band approximation (offset={offset})",
            prefix.with_name(prefix.name + "_mainband.png"),
        )
        save_heatmap(
            diag_topk_core,
            f"Layer {idx} main-band + top-{representative_topk}",
            prefix.with_name(prefix.name + f"_diag_topk_{representative_topk}.png"),
        )
        save_heatmap(
            residual,
            f"Layer {idx} off-band residual (offset={offset})",
            prefix.with_name(prefix.name + "_offdiag_residual.png"),
        )
        save_mask_plot(
            offdiag_keep_mask.float(),
            f"Layer {idx} top-{representative_topk} off-band mask",
            prefix.with_name(prefix.name + f"_topk_mask_{representative_topk}.png"),
        )


def parse_topk_list(topk_str: str) -> List[int]:
    values = []
    for chunk in topk_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("topk list must contain at least one integer")
    return values


def resolve_model_loader(model_path: str) -> Tuple[Any, str]:
    attempted: List[str] = []
    try:
        from transformers import AutoModel

        attempted.append("transformers.AutoModel.from_pretrained")
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        return model, attempted[-1]
    except Exception as exc_auto:
        attempted.append(f"AutoModel failed: {exc_auto}")

    try:
        from transformers import ViTModel

        attempted.append("transformers.ViTModel.from_pretrained")
        model = ViTModel.from_pretrained(model_path, local_files_only=True)
        return model, attempted[-1]
    except Exception as exc_vit:
        attempted.append(f"ViTModel failed: {exc_vit}")

    message = "Unable to load ViT model from local path. Attempts:\n- " + "\n- ".join(attempted)
    raise RuntimeError(message)


def split_qkv_weight(weight: Tensor, which: str) -> Tensor:
    if weight.ndim != 2:
        raise ValueError(f"qkv weight must be 2D, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if out_features % 3 != 0:
        raise ValueError(f"cannot split qkv weight with out_features={out_features}")
    chunk = out_features // 3
    index_map = {"q": 0, "k": 1, "v": 2}
    if which not in index_map:
        raise ValueError(f"qkv split only supports q/k/v, got {which}")
    start = index_map[which] * chunk
    end = start + chunk
    return weight[start:end, :]


def get_weight_type_patterns(weight_type: str) -> Dict[str, Any]:
    if weight_type == "q":
        return {
            "module_tokens": ("query", "q_proj", ".q", "to_q"),
            "parameter_tokens": ("query.weight", "q_proj.weight", ".q.weight", "to_q.weight"),
            "allow_qkv_split": True,
            "qkv_component": "q",
        }
    if weight_type == "k":
        return {
            "module_tokens": ("key", "k_proj", ".k", "to_k"),
            "parameter_tokens": ("key.weight", "k_proj.weight", ".k.weight", "to_k.weight"),
            "allow_qkv_split": True,
            "qkv_component": "k",
        }
    if weight_type == "v":
        return {
            "module_tokens": ("value", "v_proj", ".v", "to_v"),
            "parameter_tokens": ("value.weight", "v_proj.weight", ".v.weight", "to_v.weight"),
            "allow_qkv_split": True,
            "qkv_component": "v",
        }
    if weight_type == "o":
        return {
            "module_tokens": ("output.dense", "out_proj", "proj", "to_out"),
            "parameter_tokens": ("output.dense.weight", "out_proj.weight", "proj.weight", "to_out.weight"),
            "allow_qkv_split": False,
            "qkv_component": None,
        }
    if weight_type == "up":
        return {
            "module_tokens": ("intermediate.dense", "fc1", "up_proj", "gate_proj"),
            "parameter_tokens": ("intermediate.dense.weight", "fc1.weight", "up_proj.weight", "gate_proj.weight"),
            "allow_qkv_split": False,
            "qkv_component": None,
        }
    if weight_type == "down":
        return {
            "module_tokens": ("output.dense", "fc2", "down_proj"),
            "parameter_tokens": ("output.dense.weight", "fc2.weight", "down_proj.weight"),
            "allow_qkv_split": False,
            "qkv_component": None,
        }
    raise ValueError(f"unsupported weight_type: {weight_type}")


def extract_named_weights_from_vit(
    model: Any,
    weight_type: str,
    max_layers: Optional[int] = None,
) -> Tuple[List[Tensor], List[str], List[str]]:
    patterns = get_weight_type_patterns(weight_type)
    named_modules = list(model.named_modules())
    named_parameters = dict(model.named_parameters())
    entries: List[Tuple[int, str, Tensor]] = []
    candidates: List[str] = []

    for name, module in named_modules:
        lower_name = name.lower()
        if not lower_name:
            continue

        if any(token in lower_name for token in patterns["module_tokens"]):
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                entries.append((len(entries), name, weight.detach().cpu()))
                candidates.append(name)
        elif patterns["allow_qkv_split"] and "qkv" in lower_name:
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                component = patterns["qkv_component"]
                entries.append(
                    (
                        len(entries),
                        f"{name}[{component}-from-qkv]",
                        split_qkv_weight(weight.detach().cpu(), component),
                    )
                )
                candidates.append(name)
        elif any(token in lower_name for token in ("attention", "attn", "intermediate", "output", "mlp")):
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                candidates.append(name)

    if not entries:
        # Parameter fallback for implementations that do not expose Linear modules cleanly.
        for name, parameter in named_parameters.items():
            lower_name = name.lower()
            if parameter.ndim != 2:
                continue
            if any(token in lower_name for token in patterns["parameter_tokens"]):
                entries.append((len(entries), name, parameter.detach().cpu()))
            elif patterns["allow_qkv_split"] and "qkv.weight" in lower_name:
                component = patterns["qkv_component"]
                entries.append(
                    (
                        len(entries),
                        f"{name}[{component}-from-qkv]",
                        split_qkv_weight(parameter.detach().cpu(), component),
                    )
                )
            elif any(token in lower_name for token in ("attention", "attn", "intermediate", "output", "mlp")):
                candidates.append(name)

    if not entries:
        raise RuntimeError(
            f"Failed to locate weight_type={weight_type}. Candidate modules/params:\n- "
            + "\n- ".join(sorted(set(candidates))[:80])
        )

    # Keep insertion order, then truncate if requested.
    names = [entry[1] for entry in entries]
    weights = [entry[2] for entry in entries]
    if max_layers is not None:
        names = names[:max_layers]
        weights = weights[:max_layers]

    return weights, names, sorted(set(candidates))


def print_weight_inventory(weight_type: str, weights: Sequence[Tensor], names: Sequence[str]) -> None:
    print(f"Found {len(weights)} matrices for weight_type={weight_type}.")
    for name, weight in zip(names, weights):
        print(f"  {name}: shape={tuple(weight.shape)}")


def summarize_results(analysis: Dict[str, Any], offset: int, topk_list: Sequence[int]) -> None:
    summary = analysis["summary"]
    print("\n=== Summary ===")
    print(f"mean off-band energy ratio: {summary['mean_off_band_energy_ratio']:.6f}")
    print(f"mean main-band(offset={offset}) core relative error: {summary['mean_main_band_core_relative_error']:.6f}")
    print(f"mean main-band(offset={offset}) weight relative error: {summary['mean_main_band_weight_relative_error']:.6f}")
    print("main-band + top-k sparse correction gain curve:")
    for k in topk_list:
        metrics = summary["topk_gain_curve"][int(k)]
        print(
            f"  k={int(k):>3d}: core_rel_err={metrics['core_relative_error_mean']:.6f}, "
            f"weight_rel_err={metrics['weight_relative_error_mean']:.6f}, "
            f"energy_kept={metrics['retained_core_energy_ratio_mean']:.6f}"
        )


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def save_outputs(
    output_dir: Path,
    config: Dict[str, Any],
    layer_names: Sequence[str],
    weights: Tensor,
    fit: FitResult,
    analysis: Dict[str, Any],
    topk_list: Sequence[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    structured_approxs: Dict[str, Any] = {
        "main_band": [],
        "main_band_plus_topk": {str(int(k)): [] for k in topk_list},
    }

    for core in analysis["cores"]:
        structured_approxs["main_band"].append(project_main_band(core, offset=int(config["offset"])))
        for k in topk_list:
            approx_core, _ = project_mainband_plus_topk(core, offset=int(config["offset"]), topk=int(k))
            structured_approxs["main_band_plus_topk"][str(int(k))].append(approx_core)

    structured_approxs["main_band"] = torch.stack(structured_approxs["main_band"], dim=0)
    for key, mats in structured_approxs["main_band_plus_topk"].items():
        structured_approxs["main_band_plus_topk"][key] = torch.stack(mats, dim=0)

    artifact_payload = {
        "config": config,
        "layer_names": list(layer_names),
        "original_weights": weights.cpu(),
        "U": fit.U,
        "V": fit.V,
        "cores": analysis["cores"],
        "loss_history": fit.loss_history,
        "structured_approximations": structured_approxs,
        "per_layer_metrics": analysis["per_layer"],
        "summary": analysis["summary"],
    }
    torch.save(artifact_payload, output_dir / "artifacts.pt")

    json_payload = {
        "config": config,
        "layer_names": list(layer_names),
        "fit_summary": {
            "init_mean_offdiag_ratio": fit.init_mean_offdiag_ratio,
            "final_mean_offdiag_ratio": fit.final_mean_offdiag_ratio,
            "init_diagonalization_ratio": fit.init_diagonalization_ratio,
            "final_diagonalization_ratio": fit.final_diagonalization_ratio,
            "init_mean_offdiag_norm": fit.init_mean_offdiag_norm,
            "final_mean_offdiag_norm": fit.final_mean_offdiag_norm,
            "init_mean_diag_norm": fit.init_mean_diag_norm,
            "final_mean_diag_norm": fit.final_mean_diag_norm,
            "loss_history": fit.loss_history,
        },
        "summary": analysis["summary"],
        "per_layer": analysis["per_layer"],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(json_payload), handle, indent=2, ensure_ascii=False)

    save_visualizations(
        figures_dir,
        analysis["cores"],
        analysis["per_layer"],
        offset=int(config["offset"]),
        topk_list=topk_list,
        vis_count=int(config["vis_count"]),
    )
    save_topk_curve(analysis["summary"]["topk_gain_curve"], figures_dir / "topk_gain_curve.png")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-rank shared-basis diagonal structural hypothesis test on ViT weights."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/home/7250091/date/hf_cache_models/models/vit-base-patch16-224",
        help="Local ViT model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="shared_basis_fullrank_vit_outputs",
        help="Directory for .pt/.json/figure outputs.",
    )
    parser.add_argument("--max-layers", type=int, default=None, help="Optional debug limit on number of layers.")
    parser.add_argument(
        "--weight-type",
        type=str,
        choices=["q", "k", "v", "o", "up", "down"],
        default="q",
        help="Which same-type ViT weight group to analyze.",
    )
    parser.add_argument(
        "--topk-list",
        type=str,
        default="4,8,16,32",
        help="Comma-separated top-k values for diagonal + sparse correction analysis.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Band half-width used exactly like diagonalization.py. 0 means pure diagonal.",
    )
    parser.add_argument("--vis-count", type=int, default=2, help="Representative layers per category for visualization.")
    parser.add_argument("--device", type=str, default=None, help="Explicit torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate for basis optimization.")
    parser.add_argument("--max-iters", type=int, default=400, help="Optimization iterations.")
    parser.add_argument("--block-size", type=int, default=16, help="Optional block-diagonal analysis size.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    topk_list = parse_topk_list(args.topk_list)
    device = get_device(args.device)
    set_seed(args.seed)

    print(f"Using device: {device}")
    print(f"Loading ViT model from: {args.model_path}")
    model, loader_name = resolve_model_loader(args.model_path)
    print(f"Model loaded via: {loader_name}")
    model.eval()

    weights_list, layer_names, candidates = extract_named_weights_from_vit(
        model,
        weight_type=args.weight_type,
        max_layers=args.max_layers,
    )
    print_weight_inventory(args.weight_type, weights_list, layer_names)
    if candidates:
        print("\nCandidate module paths discovered during probing:")
        for name in candidates[:30]:
            print(f"  {name}")
        if len(candidates) > 30:
            print(f"  ... and {len(candidates) - 30} more")

    weights = stack_weights(weights_list, device=device)
    print(
        f"\nAnalyzing {weights.shape[0]} matrices for weight_type={args.weight_type} "
        f"with shape {tuple(weights.shape[-2:])}"
    )

    fit = fit_shared_fullrank_basis(
        weights=weights,
        offset=args.offset,
        lr=args.lr,
        max_iters=args.max_iters,
        seed=args.seed,
        device=device,
    )

    print("Average off-diagonal norm:")
    print(f"  after_ratio  (a1/a2): {fit.final_diagonalization_ratio}")
    print(f"  before_ratio (b1/b2): {fit.init_diagonalization_ratio}")
    print(f"  after_offdiag_mean_norm (a1): {fit.final_mean_offdiag_norm}")
    print(f"  after_main_diagonal_mean_norm (a2): {fit.final_mean_diag_norm}")
    print(f"  before_offdiag_mean_norm (b1): {fit.init_mean_offdiag_norm}")
    print(f"  before_main_diagonal_mean_norm (b2): {fit.init_mean_diag_norm}")

    analysis = analyze_structures(
        weights=weights.cpu(),
        U=fit.U,
        V=fit.V,
        offset=args.offset,
        topk_list=topk_list,
        block_size=args.block_size,
    )
    summarize_results(analysis, args.offset, topk_list)

    output_dir = Path(args.output_dir)
    config = {
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "max_layers": args.max_layers,
        "weight_type": args.weight_type,
        "offset": args.offset,
        "topk_list": topk_list,
        "vis_count": args.vis_count,
        "device": str(device),
        "seed": args.seed,
        "lr": args.lr,
        "max_iters": args.max_iters,
        "block_size": args.block_size,
        "loader_name": loader_name,
    }
    save_outputs(output_dir, config, layer_names, weights.cpu(), fit, analysis, topk_list)
    print(f"\nSaved artifacts to: {output_dir.resolve()}")
    if not HAS_MATPLOTLIB:
        print("matplotlib unavailable; figures were skipped, but numerical outputs were saved.")


if __name__ == "__main__":
    main()
