from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch


Tensor = torch.Tensor


def diagonalization(
    weights: list[Tensor],
    offset: int = 0,
    lr: float = 0.01,
    max_iters: int = 1000,
    init_A: Optional[Tensor] = None,
    init_B: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Fit shared transforms A and B such that A @ W_i @ B becomes approximately diagonal/banded.

    This intentionally mirrors `E:\\code\\new\\diagonalization.py` as closely as possible so the same
    empirical objective can be reused without changing the method definition.
    """
    if not weights:
        raise ValueError("`weights` must contain at least one matrix.")

    p, q = weights[0].shape
    for weight in weights:
        if weight.shape != (p, q):
            raise ValueError(f"Expected all weights to have shape {(p, q)}, got {tuple(weight.shape)}.")

    device = weights[0].device
    dtype = weights[0].dtype
    A = torch.randn(p, p, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(q, q, device=device, dtype=dtype, requires_grad=True)
    if init_A is not None:
        A = init_A.clone().to(device=device, dtype=dtype).requires_grad_(True)
    if init_B is not None:
        B = init_B.clone().to(device=device, dtype=dtype).requires_grad_(True)

    mask = (
        torch.abs(torch.arange(p, device=device).view(-1, 1) - torch.arange(q, device=device).view(1, -1)) <= offset
    )

    weight_norms = [weight.norm().clamp_min(1e-12) for weight in weights]

    optimizer = torch.optim.Adam([A, B], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    for iter_idx in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        A_norm = A / A.norm().clamp_min(1e-12)
        B_norm = B / B.norm().clamp_min(1e-12)
        total_loss = torch.zeros((), device=device, dtype=dtype)
        for weight, weight_norm in zip(weights, weight_norms):
            normalized_weight = weight / weight_norm
            transformed = A_norm @ normalized_weight @ B_norm
            diag = transformed.masked_fill(~mask, 0)
            off_diag = transformed.masked_fill(mask, 0)
            total_loss = total_loss + off_diag.norm() ** 2 / diag.norm().clamp_min(1e-12) ** 2
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    A = (A / A.norm().clamp_min(1e-12)).detach()
    B = (B / B.norm().clamp_min(1e-12)).detach()
    return A, B


def to_core_space(weight: Tensor, A: Tensor, B: Tensor) -> Tensor:
    return A @ weight @ B


def from_core_space(core: Tensor, A_inv: Tensor, B_inv: Tensor) -> Tensor:
    return A_inv @ core @ B_inv


def dense_core_grad_from_weight_grad(grad_w: Tensor, A_inv: Tensor, B_inv: Tensor) -> Tensor:
    return A_inv.transpose(0, 1) @ grad_w @ B_inv.transpose(0, 1)


@dataclass
class SharedBasisEntry:
    """Frozen shared transforms for one functional module group."""

    group_name: str
    A: Tensor
    B: Tensor
    A_inv: Tensor
    B_inv: Tensor
    shape: tuple[int, int]
    offset: int = 0

    def validate(self, expected_shape: tuple[int, int]) -> None:
        if tuple(expected_shape) != tuple(self.shape):
            raise ValueError(
                f"Shared basis group {self.group_name!r} has shape {self.shape}, expected {tuple(expected_shape)}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "A": self.A,
            "B": self.B,
            "A_inv": self.A_inv,
            "B_inv": self.B_inv,
            "shape": tuple(self.shape),
            "offset": self.offset,
        }


class SharedBasisPack:
    """Collection of named shared-basis entries loaded from disk."""

    def __init__(self, entries: dict[str, SharedBasisEntry]) -> None:
        self.entries = entries

    def get(self, group_name: str) -> SharedBasisEntry:
        if group_name not in self.entries:
            raise KeyError(f"Shared basis group {group_name!r} was not found in the basis pack.")
        return self.entries[group_name]

    def validate(self, group_name: str, expected_shape: tuple[int, int]) -> SharedBasisEntry:
        entry = self.get(group_name)
        entry.validate(expected_shape)
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {"entries": {name: entry.to_dict() for name, entry in self.entries.items()}}

    def save(self, path: str | Path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path | None) -> SharedBasisPack:
        if path is None:
            raise ValueError("`shared_basis_path` must be set for DSS v0.")
        payload = torch.load(Path(path), map_location="cpu")
        if isinstance(payload, SharedBasisPack):
            return payload

        if not isinstance(payload, dict):
            raise ValueError("Shared basis file must contain a dictionary-like payload.")

        raw_entries = payload.get("entries", payload.get("groups", payload))
        if not isinstance(raw_entries, dict):
            raise ValueError("Shared basis payload must contain an `entries` or `groups` dictionary.")

        entries: dict[str, SharedBasisEntry] = {}
        for group_name, raw_entry in raw_entries.items():
            entries[group_name] = cls._coerce_entry(group_name, raw_entry)
        return cls(entries)

    @staticmethod
    def _coerce_entry(group_name: str, raw_entry: Any) -> SharedBasisEntry:
        if isinstance(raw_entry, SharedBasisEntry):
            return raw_entry
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Invalid shared basis entry for group {group_name!r}.")

        A = raw_entry["A"]
        B = raw_entry["B"]
        A_inv = raw_entry.get("A_inv")
        B_inv = raw_entry.get("B_inv")
        if A_inv is None:
            A_inv = torch.linalg.inv(A)
        if B_inv is None:
            B_inv = torch.linalg.inv(B)
        shape = tuple(raw_entry.get("shape", tuple(A.shape)))
        offset = int(raw_entry.get("offset", 0))
        return SharedBasisEntry(
            group_name=raw_entry.get("group_name", group_name),
            A=A,
            B=B,
            A_inv=A_inv,
            B_inv=B_inv,
            shape=shape,
            offset=offset,
        )


def fit_shared_basis_for_group(
    group_name: str,
    weights: Iterable[Tensor],
    offset: int = 0,
    lr: float = 0.01,
    max_iters: int = 1000,
    init_A: Optional[Tensor] = None,
    init_B: Optional[Tensor] = None,
) -> SharedBasisEntry:
    """Fit one shared basis entry from a list of pretrained weight matrices.

    This is an offline preprocessing step. Its Adam optimizer and learning rate
    belong only to basis fitting, not to the online quantile estimator and not
    to the outer DSSAdamW used during fine-tuning.
    """
    weight_list = list(weights)
    if not weight_list:
        raise ValueError("`weights` must contain at least one matrix.")
    A, B = diagonalization(
        weight_list,
        offset=offset,
        lr=lr,
        max_iters=max_iters,
        init_A=init_A,
        init_B=init_B,
    )
    A_inv = torch.linalg.inv(A)
    B_inv = torch.linalg.inv(B)
    return SharedBasisEntry(
        group_name=group_name,
        A=A.detach().cpu(),
        B=B.detach().cpu(),
        A_inv=A_inv.detach().cpu(),
        B_inv=B_inv.detach().cpu(),
        shape=tuple(weight_list[0].shape),
        offset=offset,
    )
