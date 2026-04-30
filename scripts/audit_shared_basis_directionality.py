from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import torch


REMOTE_PROJECT_ROOT = Path("/data/home/7250091/date/DSS")

GROUP_MAP = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "u": "up_proj",
    "d": "down_proj",
    "o": "o_proj",
    "g": "gate_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print simple A/B and inverse statistics from a saved shared-basis file.")
    parser.add_argument(
        "--shared_basis_path",
        type=str,
        default=str(REMOTE_PROJECT_ROOT / "basis" / "llama3_8b_dss_basis_identity_fro.pt"),
    )
    parser.add_argument("--groups", type=str, default="q", help="Short group keys such as q, k, v, qkv.")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def resolve_groups(groups: str) -> list[str]:
    resolved = [GROUP_MAP[key] for key in groups if key in GROUP_MAP]
    if not resolved:
        raise ValueError("`--groups` did not resolve to any known basis groups.")
    return resolved


def load_entries(path: str | Path) -> dict[str, dict]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Shared basis file must contain a dictionary-like payload.")
    entries = payload.get("entries", payload.get("groups", payload))
    if not isinstance(entries, dict):
        raise ValueError("Shared basis payload must contain an `entries` or `groups` dictionary.")
    return entries


def matrix_stats(matrix: torch.Tensor) -> dict[str, float]:
    x = matrix.detach().float().cpu()
    abs_x = x.abs()
    return {
        "mean_abs": float(abs_x.mean().item()) if x.numel() else 0.0,
        "rms": float(torch.sqrt((x * x).mean()).item()) if x.numel() else 0.0,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("inf")
    return float(numerator / denominator)


def normalize_inverse_basis_to_identity_fro(
    a_inv: torch.Tensor,
    b_inv: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    p, n = tuple(shape)
    a_norm = a_inv.detach().float().norm().clamp_min(1e-12)
    b_norm = b_inv.detach().float().norm().clamp_min(1e-12)
    a_scale = float(sqrt(float(p)) / a_norm.item())
    b_scale = float(sqrt(float(n)) / b_norm.item())
    return a_inv * a_scale, b_inv * b_scale


def inverse_match_stats(saved_inv: torch.Tensor, recomputed_inv: torch.Tensor) -> dict[str, float]:
    diff = (saved_inv.detach().float().cpu() - recomputed_inv.detach().float().cpu()).abs()
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def audit_group(group_name: str, raw_entry: dict) -> dict[str, object]:
    a = raw_entry["A"].detach().float().cpu()
    b = raw_entry["B"].detach().float().cpu()
    shape = tuple(a.shape)

    saved_a_inv = raw_entry.get("A_inv")
    saved_b_inv = raw_entry.get("B_inv")
    if saved_a_inv is None or saved_b_inv is None:
        raise ValueError(f"Group {group_name!r} does not contain saved A_inv/B_inv in the basis file.")
    saved_a_inv = saved_a_inv.detach().float().cpu()
    saved_b_inv = saved_b_inv.detach().float().cpu()

    recomputed_a_inv = torch.linalg.inv(a)
    recomputed_b_inv = torch.linalg.inv(b)
    recomputed_a_inv_scaled, recomputed_b_inv_scaled = normalize_inverse_basis_to_identity_fro(
        recomputed_a_inv,
        recomputed_b_inv,
        shape,
    )

    return {
        "group_name": group_name,
        "shape": list(shape),
        "A": matrix_stats(a),
        "B": matrix_stats(b),
        "A_inv_raw": matrix_stats(recomputed_a_inv),
        "B_inv_raw": matrix_stats(recomputed_b_inv),
        "A_inv_scaled_saved": matrix_stats(saved_a_inv),
        "B_inv_scaled_saved": matrix_stats(saved_b_inv),
        "A_inv_match": inverse_match_stats(saved_a_inv, recomputed_a_inv_scaled),
        "B_inv_match": inverse_match_stats(saved_b_inv, recomputed_b_inv_scaled),
    }


def print_group_result(result: dict[str, object]) -> None:
    print(f"\n[group] {result['group_name']} shape={tuple(result['shape'])}")
    print(f"A.mean_abs={result['A']['mean_abs']:.6f}")
    print(f"A.rms={result['A']['rms']:.6f}")
    print(f"B.mean_abs={result['B']['mean_abs']:.6f}")
    print(f"B.rms={result['B']['rms']:.6f}")
    print(f"A_inv_raw.mean_abs={result['A_inv_raw']['mean_abs']:.6f}")
    print(f"A_inv_raw.rms={result['A_inv_raw']['rms']:.6f}")
    print(f"A_inv_scaled(saved).mean_abs={result['A_inv_scaled_saved']['mean_abs']:.6f}")
    print(f"A_inv_scaled(saved).rms={result['A_inv_scaled_saved']['rms']:.6f}")
    print(f"B_inv_raw.mean_abs={result['B_inv_raw']['mean_abs']:.6f}")
    print(f"B_inv_raw.rms={result['B_inv_raw']['rms']:.6f}")
    print(f"B_inv_scaled(saved).mean_abs={result['B_inv_scaled_saved']['mean_abs']:.6f}")
    print(f"B_inv_scaled(saved).rms={result['B_inv_scaled_saved']['rms']:.6f}")
    print(
        f"A_inv_raw / A: mean_abs_ratio={safe_ratio(result['A_inv_raw']['mean_abs'], result['A']['mean_abs']):.4f} "
        f"rms_ratio={safe_ratio(result['A_inv_raw']['rms'], result['A']['rms']):.4f}"
    )
    print(
        f"A_inv_scaled(saved) / A: mean_abs_ratio={safe_ratio(result['A_inv_scaled_saved']['mean_abs'], result['A']['mean_abs']):.4f} "
        f"rms_ratio={safe_ratio(result['A_inv_scaled_saved']['rms'], result['A']['rms']):.4f}"
    )
    print(
        f"A_inv_raw / A_inv_scaled(saved): mean_abs_ratio={safe_ratio(result['A_inv_raw']['mean_abs'], result['A_inv_scaled_saved']['mean_abs']):.4f} "
        f"rms_ratio={safe_ratio(result['A_inv_raw']['rms'], result['A_inv_scaled_saved']['rms']):.4f}"
    )
    print(
        f"B_inv_raw / B: mean_abs_ratio={safe_ratio(result['B_inv_raw']['mean_abs'], result['B']['mean_abs']):.4f} "
        f"rms_ratio={safe_ratio(result['B_inv_raw']['rms'], result['B']['rms']):.4f}"
    )
    print(
        f"B_inv_scaled(saved) / B: mean_abs_ratio={safe_ratio(result['B_inv_scaled_saved']['mean_abs'], result['B']['mean_abs']):.4f} "
        f"rms_ratio={safe_ratio(result['B_inv_scaled_saved']['rms'], result['B']['rms']):.4f}"
    )
    print(
        f"B_inv_raw / B_inv_scaled(saved): mean_abs_ratio={safe_ratio(result['B_inv_raw']['mean_abs'], result['B_inv_scaled_saved']['mean_abs']):.4f} "
        f"rms_ratio={safe_ratio(result['B_inv_raw']['rms'], result['B_inv_scaled_saved']['rms']):.4f}"
    )
    print(
        f"A_inv(saved vs recomputed+scaled) max_abs_diff={result['A_inv_match']['max_abs_diff']:.6e} "
        f"mean_abs_diff={result['A_inv_match']['mean_abs_diff']:.6e}"
    )
    print(
        f"B_inv(saved vs recomputed+scaled) max_abs_diff={result['B_inv_match']['max_abs_diff']:.6e} "
        f"mean_abs_diff={result['B_inv_match']['mean_abs_diff']:.6e}"
    )


def main() -> None:
    args = parse_args()
    groups = resolve_groups(args.groups)
    entries = load_entries(args.shared_basis_path)

    results = {
        "shared_basis_path": args.shared_basis_path,
        "groups": groups,
        "results": {},
    }

    for group_name in groups:
        if group_name not in entries:
            raise KeyError(f"Group {group_name!r} was not found in shared basis file {args.shared_basis_path}.")
        result = audit_group(group_name, entries[group_name])
        results["results"][group_name] = result
        print_group_result(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\nSaved A/B inverse audit to {output_path}")


if __name__ == "__main__":
    main()
