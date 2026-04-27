"""
Convert an existing DSS SharedBasisPack to identity-Frobenius-normalized inverse bases.

This does not refit A or B. It reads an existing basis .pt file, recomputes or
loads the raw inverse matrices, scales them as:

    C = c * A_inv,  ||C||_F = sqrt(p)
    D = d * B_inv,  ||D||_F = sqrt(n)

for W in R^{p x n}, then saves a new basis .pt file.
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import torch


REMOTE_PROJECT_ROOT = Path("/data/home/7250091/date/DSS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize DSS inverse bases by same-shape identity Frobenius norm.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(REMOTE_PROJECT_ROOT / "basis" / "llama3_8b_dss_basis.pt"),
        help="Existing shared-basis .pt path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(REMOTE_PROJECT_ROOT / "basis" / "llama3_8b_dss_basis_identity_fro.pt"),
        help="Output path for the normalized shared-basis .pt.",
    )
    return parser.parse_args()


def fro(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().item())


def normalize_inverse_basis_to_identity_fro(
    A_inv: torch.Tensor,
    B_inv: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    p, n = tuple(shape)
    A_norm = A_inv.detach().float().norm().clamp_min(1e-12)
    B_norm = B_inv.detach().float().norm().clamp_min(1e-12)
    A_inv_scale = float(sqrt(float(p)) / A_norm.item())
    B_inv_scale = float(sqrt(float(n)) / B_norm.item())
    return A_inv * A_inv_scale, B_inv * B_inv_scale, A_inv_scale, B_inv_scale


def load_entries(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Shared basis file must contain a dictionary payload.")
    entries = payload.get("entries", payload.get("groups", payload))
    if not isinstance(entries, dict):
        raise ValueError("Shared basis payload must contain entries/groups.")
    return entries


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_entries(input_path)
    normalized_entries: dict[str, dict] = {}

    print(f"[normalize] input={input_path}")
    print(f"[normalize] output={output_path}")
    for group_name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry for group {group_name!r}; expected a dictionary.")
        if entry.get("inverse_normalization") == "identity_fro":
            raise ValueError(
                f"Basis group {group_name!r} is already identity_fro normalized. "
                "Use the original unnormalized basis as --input_path to avoid double scaling."
            )
        A = entry["A"].detach().float()
        B = entry["B"].detach().float()
        A_inv_raw = entry.get("A_inv")
        B_inv_raw = entry.get("B_inv")
        A_inv_raw = torch.linalg.inv(A) if A_inv_raw is None else A_inv_raw.detach().float()
        B_inv_raw = torch.linalg.inv(B) if B_inv_raw is None else B_inv_raw.detach().float()
        shape = tuple(entry.get("shape", tuple(A.shape)))
        A_inv, B_inv, A_inv_scale, B_inv_scale = normalize_inverse_basis_to_identity_fro(
            A_inv_raw,
            B_inv_raw,
            shape,
        )

        p, n = shape
        print(f"\n[group] {group_name} shape={shape}")
        print(f"  A_inv_scale={A_inv_scale:.8e}, B_inv_scale={B_inv_scale:.8e}, product={A_inv_scale * B_inv_scale:.8e}")
        print(f"  A_inv fro: before={fro(A_inv_raw):.8e}, after={fro(A_inv):.8e}, target={p ** 0.5:.8e}")
        print(f"  B_inv fro: before={fro(B_inv_raw):.8e}, after={fro(B_inv):.8e}, target={n ** 0.5:.8e}")

        normalized_entry = dict(entry)
        normalized_entry.update(
            {
                "group_name": entry.get("group_name", group_name),
                "A": entry["A"].detach().cpu(),
                "B": entry["B"].detach().cpu(),
                "A_inv": A_inv.detach().cpu(),
                "B_inv": B_inv.detach().cpu(),
                "shape": shape,
                "offset": int(entry.get("offset", 0)),
                "A_inv_scale": A_inv_scale,
                "B_inv_scale": B_inv_scale,
                "inverse_normalization": "identity_fro",
            }
        )
        normalized_entries[group_name] = normalized_entry

    torch.save({"entries": normalized_entries}, output_path)
    print(f"\n[normalize] saved normalized basis to: {output_path}")


if __name__ == "__main__":
    main()
