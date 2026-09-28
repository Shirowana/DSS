from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SIFT_ROOT = ROOT / "SIFT"


class SIFTAdapter:
    """工程适配层：保留官方 SIFT 选择/更新逻辑，补齐 DDP 和导出。"""

    def __init__(
        self,
        model: nn.Module,
        sparse_rate: float,
        sparse_modules: list[str],
        grad_acc: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        if not 0.0 < sparse_rate <= 1.0:
            raise ValueError(f"sparse_rate must be in (0, 1], found {sparse_rate}")
        if not sparse_modules:
            raise ValueError("sparse_modules must not be empty")
        if grad_acc < 1:
            raise ValueError(f"grad_acc must be >= 1, found {grad_acc}")
        if str(SIFT_ROOT) not in sys.path:
            sys.path.insert(0, str(SIFT_ROOT))
        from sift import SIFT  # official implementation from SIFT/sift/sift.py

        self.model = model
        self.sift = SIFT(
            model,
            sparse_rate=sparse_rate,
            sparse_module=sparse_modules,
            grad_acc=grad_acc,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._indices_synchronized = False
        # DDP must not reduce dense base-parameter gradients. SIFT's sparse
        # parameters remain visible to DDP and are synchronized explicitly.
        self.model._ddp_params_and_buffers_to_ignore = [
            name for name, _ in self.model.named_parameters() if not name.endswith("_sparse")
        ]
        self.manifest: dict[str, Any] = {
            "implementation": "official_SIFT_with_baseline_DDP_adapter",
            "selection": "official_first_backward_gradient_topk_then_fixed_indices",
            "sparse_rate": sparse_rate,
            "sparse_modules": list(sparse_modules),
            "gradient_accumulation": grad_acc,
            "gradient_checkpointing": gradient_checkpointing,
            "distributed_index_rule": "rank0_first_backward_topk_broadcast_before_second_backward",
            "distributed_gradient_rule": "explicit_sparse_allreduce; dense_gradients_ignored_by_ddp",
            "export": "merge_sparse_delta_into_dense_parameters_and_remove_sparse_parameters",
        }

    @property
    def sparse_mapping(self):
        return self.sift.sparse_mapping

    @property
    def if_get_idx(self):
        return self.sift.if_get_idx

    def get_trainable_num(self) -> int:
        return self.sift.get_trainable_num()

    def parameters_in_optimizer(self):
        return self.sift.parameters_in_optimizer()

    def named_parameters_in_optimizer(self):
        return self.sift.named_parameters_in_optimizer()

    def set_trainer(self, trainer) -> None:
        self.sift.set_trainer(trainer)

    def _selection_payload(self) -> dict[str, list[list[int]]]:
        return {
            name: np.asarray(value.idx).tolist()
            for name, value in self.sparse_mapping.items()
        }

    def _selection_hash(self) -> str:
        payload = json.dumps(self._selection_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _broadcast_indices(self) -> None:
        for sparse_parameter in self.parameters_in_optimizer():
            indices = torch.as_tensor(
                sparse_parameter.idx,
                dtype=torch.long,
                device=sparse_parameter.device,
            )
            dist.broadcast(indices, src=0)
            sparse_parameter.idx = indices.cpu().numpy()

    def sync_gradients(self) -> None:
        """Synchronize official SIFT state after backward, before optimizer.step."""
        if not self._indices_synchronized:
            if not all(self.if_get_idx.values()):
                missing = [name for name, ready in self.if_get_idx.items() if not ready]
                raise RuntimeError(f"Official SIFT did not select indices for {len(missing)} parameters")
            if dist.is_available() and dist.is_initialized():
                self._broadcast_indices()
                hashes: list[str | None] = [None] * dist.get_world_size()
                dist.all_gather_object(hashes, self._selection_hash())
                if len(set(hashes)) != 1:
                    raise RuntimeError(f"SIFT index broadcast verification failed: {hashes}")
            self._indices_synchronized = True

        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for sparse_parameter in self.parameters_in_optimizer():
            if sparse_parameter.grad is not None:
                dist.all_reduce(sparse_parameter.grad, op=dist.ReduceOp.SUM)
                sparse_parameter.grad.div_(world_size)

    def clear_dense_gradients(self) -> None:
        for name, parameter in self.model.named_parameters():
            if not name.endswith("_sparse"):
                parameter.grad = None

    def apply_delta(self) -> None:
        named_parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, sparse_parameter in self.sparse_mapping.items():
                parameter = named_parameters[name]
                indices = tuple(
                    torch.as_tensor(axis, dtype=torch.long, device=parameter.device)
                    for axis in sparse_parameter.idx
                )
                parameter[indices] += sparse_parameter.to(parameter.dtype)
                sparse_parameter.zero_()

    def save_candidate(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        named_parameters = dict(self.model.named_parameters())
        state = {}
        for name, sparse_parameter in self.sparse_mapping.items():
            parameter = named_parameters[name]
            indices = tuple(
                torch.as_tensor(axis, dtype=torch.long, device=parameter.device)
                for axis in sparse_parameter.idx
            )
            state[name] = {
                "indices": torch.as_tensor(sparse_parameter.idx, dtype=torch.long),
                "values": parameter.detach()[indices].cpu(),
            }
        torch.save(state, path)

    def load_candidate(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        named_parameters = dict(self.model.named_parameters())
        if set(state) != set(self.sparse_mapping):
            raise ValueError("Saved SIFT candidate does not match configured sparse parameters")
        with torch.no_grad():
            for name, item in state.items():
                parameter = named_parameters[name]
                indices = tuple(axis.to(parameter.device) for axis in item["indices"])
                parameter[indices] = item["values"].to(parameter)
                self.sparse_mapping[name].idx = item["indices"].cpu().numpy()

    def prepare_dense_export(self) -> None:
        self.apply_delta()
        for name in list(self.sparse_mapping):
            registered_name = name.replace(".", "_") + "_sparse"
            if hasattr(self.model, registered_name):
                delattr(self.model, registered_name)

