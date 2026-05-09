from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers.pytorch_utils import Conv1D

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


@dataclass
class SearchRuntime:
    curr_count: int = 0


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


class GPUQuantileEstimator:
    def __init__(
        self,
        target_quantile: float,
        initial_value: float = 0.0,
        lr: float = 0.01,
        alpha: float = 0.0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.q = target_quantile
        self.device = torch.device(device)
        self.quantile = nn.Parameter(torch.tensor(initial_value, device=self.device, dtype=torch.float32))
        self.base_lr = lr
        self.alpha = alpha
        self.step_count = 0
        self.optimizer = optim.SGD([self.quantile], lr=lr)
        self.quantile.requires_grad_(True)

    def _pinball_loss(self, x: torch.Tensor) -> torch.Tensor:
        error = x - self.quantile
        loss = torch.where(error > 0, self.q * error, (self.q - 1) * error)
        return loss.mean()

    def update(self, new_data: torch.Tensor) -> None:
        if new_data.numel() == 0:
            return
        self.step_count += 1
        current_scale = abs(self.quantile.item()) + 1e-8
        real_lr = self.base_lr * current_scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = real_lr
        loss = self._pinball_loss(new_data.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_quantile(self) -> torch.Tensor:
        return self.quantile.data

    def reset(self) -> None:
        self.quantile.data.fill_(0.0)
        self.step_count = 0
        self.optimizer = optim.SGD([self.quantile], lr=self.base_lr)


class DSSLayer(BaseTunerLayer):
    adapter_layer_names = ("coefficient",)
    other_param_names = ("coefficient_indices", "candidate_indices", "elite_bitset", "dropout_layer")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.coefficient = nn.ParameterDict({})
        self.coefficient_indices = BufferDict({})
        self.candidate_indices = BufferDict({})
        self.elite_bitset = BufferDict({})
        self.dropout_layer = nn.ModuleDict({})
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        self.runtime: dict[str, SearchRuntime] = {}
        self.grad_cache: dict[str, Optional[torch.Tensor]] = {}
        self.grad_count: dict[str, int] = {}
        self.search_quantile_estimator: dict[str, GPUQuantileEstimator] = {}
        self.grad_store_steps: dict[str, int] = {}
        self.candidate_size: dict[str, int] = {}
        self.low: dict[str, int] = {}
        self.up: dict[str, int] = {}
        self.threshold_mode: dict[str, str] = {}
        self.module_name: dict[str, str] = {}

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def _adapter_device(self, adapter_name: str) -> torch.device:
        return self.coefficient[adapter_name].device

    def clear_candidate_state(self, adapter_name: str) -> None:
        device = self._adapter_device(adapter_name)
        self.candidate_indices[adapter_name] = torch.empty(0, device=device, dtype=torch.long)
        self.grad_cache[adapter_name] = None
        self.grad_count[adapter_name] = 0

    @torch.no_grad()
    def export_sparse_checkpoint(self, adapter_name: str) -> dict[str, torch.Tensor]:
        curr_count = self.runtime[adapter_name].curr_count
        return {
            "coefficient": self.coefficient[adapter_name][:curr_count].detach().cpu().clone(),
            "coefficient_indices": self.coefficient_indices[adapter_name][:curr_count].detach().cpu().clone(),
        }

    @torch.no_grad()
    def restore_sparse_checkpoint(
        self,
        adapter_name: str,
        coefficient_values: torch.Tensor,
        coefficient_indices: torch.Tensor,
    ) -> None:
        coefficient_values = coefficient_values.reshape(-1).to(dtype=torch.float32)
        coefficient_indices = coefficient_indices.reshape(-1).to(dtype=torch.long)
        if coefficient_values.numel() != coefficient_indices.numel():
            raise ValueError(
                f"Sparse checkpoint for adapter {adapter_name!r} has mismatched values/indices lengths: "
                f"{coefficient_values.numel()} vs {coefficient_indices.numel()}."
            )

        coeff_param = self.coefficient[adapter_name]
        index_buffer = self.coefficient_indices[adapter_name]
        max_slots = coeff_param.numel()
        curr_count = int(coefficient_values.numel())
        if curr_count > max_slots:
            raise ValueError(
                f"Sparse checkpoint for adapter {adapter_name!r} has {curr_count} active slots, "
                f"but layer capacity is only {max_slots}."
            )

        coeff_param.data.zero_()
        index_buffer.zero_()
        if curr_count > 0:
            coeff_param.data[:curr_count].copy_(coefficient_values.to(device=coeff_param.device, dtype=coeff_param.dtype))
            index_buffer[:curr_count].copy_(coefficient_indices.to(device=index_buffer.device, dtype=index_buffer.dtype))

        elite_bitset = self.elite_bitset[adapter_name]
        elite_bitset.zero_()
        if curr_count > 0:
            elite_bitset[index_buffer[:curr_count].long()] = True

        self.runtime[adapter_name].curr_count = curr_count
        if adapter_name in self.search_quantile_estimator:
            self.search_quantile_estimator[adapter_name].reset()
        self.clear_candidate_state(adapter_name)

    @staticmethod
    def _resolve_bound(value: int | float, capacity: int) -> int:
        if isinstance(value, float):
            if value >= 1.0 and float(value).is_integer():
                return int(value)
            if value < 1.0:
                return int(capacity * value)
            return int(value)
        return int(value)

    def refresh_candidate_batch(self, adapter_name: str) -> None:
        device = self._adapter_device(adapter_name)
        dense_numel = self.out_features * self.in_features
        elite_bitset = self.elite_bitset[adapter_name]
        available = dense_numel - int(elite_bitset.sum().item())
        target_k = min(self.candidate_size[adapter_name], max(available, 0))
        if target_k <= 0:
            self.clear_candidate_state(adapter_name)
            return

        if _dist_rank() == 0:
            sampled = torch.empty(0, device=device, dtype=torch.long)
            sample_k = max(target_k * 4, target_k)
            while sampled.numel() < target_k:
                proposal = torch.randint(0, dense_numel, (sample_k,), device=device)
                proposal = torch.unique(proposal)
                proposal = proposal[~elite_bitset[proposal]]
                sampled = torch.unique(torch.cat([sampled, proposal], dim=0))
                if sampled.numel() >= target_k or sampled.numel() >= available:
                    break
                sample_k = min(dense_numel, max(sample_k * 2, target_k))
            sampled = sampled[:target_k].contiguous()
        else:
            sampled = torch.empty(target_k, device=device, dtype=torch.long)

        if _dist_ready():
            dist.broadcast(sampled, src=0)

        self.candidate_indices[adapter_name] = sampled[:target_k]
        self.grad_cache[adapter_name] = None
        self.grad_count[adapter_name] = 0

    def sync_stage1_state(self, adapter_name: str) -> None:
        if not _dist_ready():
            return

        grad_cache = self.grad_cache.get(adapter_name)
        if grad_cache is not None and grad_cache.numel() > 0:
            dist.all_reduce(grad_cache, op=dist.ReduceOp.SUM)

        count_tensor = torch.tensor(
            [self.grad_count.get(adapter_name, 0)],
            device=self._adapter_device(adapter_name),
            dtype=torch.long,
        )
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        self.grad_count[adapter_name] = int(count_tensor.item())

    def compute_x_mean(self, adapter_name: str) -> torch.Tensor:
        grad_cache = self.grad_cache[adapter_name]
        grad_count = self.grad_count.get(adapter_name, 0)
        if grad_cache is None or grad_cache.numel() == 0 or grad_count <= 0:
            return torch.empty(0, device=self.coefficient_indices[adapter_name].device, dtype=torch.float32)
        return (grad_cache / float(grad_count)).abs()

    def update_distribution(self, adapter_name: str, x_mean: torch.Tensor) -> None:
        if x_mean.numel() == 0:
            return
        candidates_data = x_mean.reshape(-1).float()
        search_estimator = self.search_quantile_estimator[adapter_name]
        if self.threshold_mode[adapter_name] == "oracle":
            if candidates_data.max() > 0:
                search_estimator.quantile.data.fill_(torch.quantile(candidates_data, search_estimator.q))
            return

        if search_estimator.quantile.item() == 0.0 and candidates_data.max() > 0:
            search_estimator.quantile.data.fill_(torch.quantile(candidates_data, search_estimator.q))

        batch_size = 1000
        total_samples = candidates_data.shape[0]
        perm = torch.randperm(total_samples, device=candidates_data.device)
        shuffled = candidates_data[perm]
        for start in range(0, total_samples, batch_size):
            search_estimator.update(shuffled[start : start + batch_size])

    def select_location(self, adapter_name: str, x_mean: torch.Tensor, remaining_budget: int) -> torch.Tensor:
        if x_mean.numel() == 0 or remaining_budget <= 0:
            return torch.empty(0, device=x_mean.device, dtype=torch.long)

        threshold = self.search_quantile_estimator[adapter_name].get_quantile()
        numel = x_mean.numel()
        real_up = min(self.up[adapter_name], remaining_budget, numel)
        real_low = min(self.low[adapter_name], remaining_budget, numel)
        candidate_mask = x_mean.float() > threshold
        num_candidates = int(candidate_mask.sum().item())

        if num_candidates > real_up:
            return torch.argsort(x_mean.float(), descending=True)[:real_up]
        if num_candidates < real_low:
            if real_low == 0:
                return torch.empty(0, device=x_mean.device, dtype=torch.long)
            return torch.argsort(x_mean.float(), descending=True)[:real_low]
        return torch.nonzero(candidate_mask, as_tuple=True)[0]

    def apply_stage1_promotions(self, adapter_name: str, selected: torch.Tensor) -> int:
        if selected.numel() == 0:
            return 0

        runtime = self.runtime[adapter_name]
        start = runtime.curr_count
        end = start + int(selected.numel())
        new_indices = self.candidate_indices[adapter_name][selected]
        self.coefficient_indices[adapter_name][start:end] = new_indices
        self.coefficient[adapter_name].data[start:end].zero_()
        self.elite_bitset[adapter_name][new_indices] = True
        runtime.curr_count = end
        return int(selected.numel())

    def maybe_refresh_stage1(self, adapter_name: str) -> int:
        runtime = self.runtime[adapter_name]
        total_budget = int(self.coefficient[adapter_name].numel())
        if runtime.curr_count >= total_budget:
            self.clear_candidate_state(adapter_name)
            return 0

        if self.candidate_indices[adapter_name].numel() == 0:
            self.refresh_candidate_batch(adapter_name)

        if self.grad_count.get(adapter_name, 0) < self.grad_store_steps[adapter_name]:
            return 0

        self.sync_stage1_state(adapter_name)
        remaining_budget = total_budget - runtime.curr_count
        x_mean = self.compute_x_mean(adapter_name)
        self.update_distribution(adapter_name, x_mean)
        selected = self.select_location(adapter_name, x_mean, remaining_budget)
        promoted = self.apply_stage1_promotions(adapter_name, selected)
        self.clear_candidate_state(adapter_name)
        if runtime.curr_count < total_budget:
            self.refresh_candidate_batch(adapter_name)
        return promoted

    @torch.no_grad()
    def collect_candidate_grads(self, adapter_name: str, candidate_grads: torch.Tensor) -> None:
        if candidate_grads.numel() == 0:
            return
        device = self._adapter_device(adapter_name)
        candidate_grads = candidate_grads.detach().to(device=device, dtype=torch.float32)
        grad_cache = self.grad_cache.get(adapter_name)
        if grad_cache is None:
            self.grad_cache[adapter_name] = torch.zeros_like(candidate_grads, device=device, dtype=torch.float32)
            self.grad_count[adapter_name] = 0
        elif grad_cache.numel() != candidate_grads.numel():
            raise RuntimeError(
                f"Stage1 grad cache size mismatch for adapter {adapter_name!r}: "
                f"expected {grad_cache.numel()}, got {candidate_grads.numel()}."
            )
        self.grad_cache[adapter_name].add_(candidate_grads)
        self.grad_count[adapter_name] += 1

    def get_candidate_grad_hook(self, adapter_name: str, candidate_indices: torch.Tensor):
        def hook(grad_delta: torch.Tensor) -> torch.Tensor:
            candidate_grads = grad_delta.reshape(-1).index_select(0, candidate_indices.long())
            self.collect_candidate_grads(adapter_name, candidate_grads)
            return grad_delta

        return hook

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        curr_count = self.runtime[adapter_name].curr_count
        delta_flat = self.coefficient[adapter_name].new_zeros(self.out_features * self.in_features)
        if curr_count > 0:
            elite_values = self.coefficient[adapter_name][:curr_count]
            elite_indices = self.coefficient_indices[adapter_name][:curr_count].long()
            delta_flat = delta_flat.scatter_add(0, elite_indices, elite_values)
        return delta_flat.view(self.out_features, self.in_features)

    def merge_(self, adapter_name: str) -> None:
        delta_weight = self.get_delta_weight(adapter_name).to(self.get_base_layer().weight.dtype)
        self.get_base_layer().weight.data.add_(delta_weight)


class DSSLinear(nn.Module, DSSLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        n_frequency: int,
        candidate_size: int,
        grad_store_steps: int,
        low: int | float,
        up: int | float,
        ratio: float,
        threshold_mode: str = "oracle",
        dropout: float = 0.0,
        quantile_lr: float = 0.01,
        quantile_alpha: float = 0.0,
        fan_in_fan_out: bool = False,
        module_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        DSSLayer.__init__(self, base_layer, fan_in_fan_out=fan_in_fan_out, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            n_frequency=n_frequency,
            candidate_size=candidate_size,
            grad_store_steps=grad_store_steps,
            low=low,
            up=up,
            ratio=ratio,
            threshold_mode=threshold_mode,
            dropout=dropout,
            quantile_lr=quantile_lr,
            quantile_alpha=quantile_alpha,
            module_name=module_name,
        )

    def update_layer(
        self,
        adapter_name: str,
        n_frequency: int,
        candidate_size: int,
        grad_store_steps: int,
        low: int | float,
        up: int | float,
        ratio: float,
        threshold_mode: str = "oracle",
        dropout: float = 0.0,
        quantile_lr: float = 0.01,
        quantile_alpha: float = 0.0,
        module_name: Optional[str] = None,
    ) -> None:
        resolved_low = self._resolve_bound(low, n_frequency)
        resolved_up = self._resolve_bound(up, n_frequency)
        if resolved_low < 0 or resolved_up < 0:
            raise ValueError("`low` and `up` must be non-negative after conversion.")
        if resolved_low > resolved_up:
            raise ValueError("`low` cannot exceed `up` after conversion.")
        if threshold_mode not in {"oracle", "sgd"}:
            raise ValueError("`threshold_mode` must be `oracle` or `sgd`.")

        device = self.get_base_layer().weight.device
        self.coefficient[adapter_name] = nn.Parameter(torch.zeros(n_frequency, device=device, dtype=torch.float32))
        self.coefficient_indices[adapter_name] = torch.zeros(n_frequency, device=device, dtype=torch.long)
        self.candidate_indices[adapter_name] = torch.empty(0, device=device, dtype=torch.long)
        self.elite_bitset[adapter_name] = torch.zeros(self.out_features * self.in_features, device=device, dtype=torch.bool)
        self.dropout_layer[adapter_name] = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.runtime[adapter_name] = SearchRuntime()
        self.grad_store_steps[adapter_name] = grad_store_steps
        self.candidate_size[adapter_name] = candidate_size
        self.low[adapter_name] = resolved_low
        self.up[adapter_name] = resolved_up
        self.threshold_mode[adapter_name] = threshold_mode
        self.module_name[adapter_name] = module_name or ""
        self.search_quantile_estimator[adapter_name] = GPUQuantileEstimator(
            target_quantile=1.0 - ratio,
            lr=quantile_lr,
            alpha=quantile_alpha,
            device=device,
        )
        self.clear_candidate_state(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.coefficient_indices[adapter_name] = self.coefficient_indices[adapter_name].to(device=device, dtype=torch.long)
        self.candidate_indices[adapter_name] = self.candidate_indices[adapter_name].to(device=device, dtype=torch.long)
        self.elite_bitset[adapter_name] = self.elite_bitset[adapter_name].to(device=device, dtype=torch.bool)
        self.set_adapter(self.active_adapters)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.merged:
                self.unmerge()
        else:
            if not self.merged:
                self.merge()
        return self

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter not in self.coefficient:
                continue
            delta_weight = self.get_delta_weight(active_adapter).to(self.get_base_layer().weight.dtype)
            base_layer = self.get_base_layer()
            if safe_merge:
                merged = base_layer.weight.data.clone()
                merged.add_(delta_weight)
                if not torch.isfinite(merged).all():
                    raise ValueError(f"NaNs detected while merging adapter {active_adapter}.")
                base_layer.weight.data = merged
            else:
                base_layer.weight.data.add_(delta_weight)
            if active_adapter not in self.merged_adapters:
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.coefficient:
                continue
            delta_weight = self.get_delta_weight(active_adapter).to(self.get_base_layer().weight.dtype)
            self.get_base_layer().weight.data.add_(delta_weight, alpha=-1.0)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs).to(previous_dtype)

        if self.merged:
            return self.base_layer(x, *args, **kwargs).to(previous_dtype)

        if self.training:
            for active_adapter in self.active_adapters:
                if active_adapter in self.coefficient:
                    self.maybe_refresh_stage1(active_adapter)

        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.coefficient:
                continue

            curr_count = self.runtime[active_adapter].curr_count
            candidate_indices = self.candidate_indices[active_adapter]
            if curr_count == 0 and candidate_indices.numel() == 0:
                continue

            delta_flat = x.new_zeros(self.out_features * self.in_features, dtype=self.get_base_layer().weight.dtype)
            if curr_count > 0:
                elite_values = self.dropout_layer[active_adapter](self.coefficient[active_adapter][:curr_count])
                elite_values = elite_values.to(dtype=delta_flat.dtype)
                elite_indices = self.coefficient_indices[active_adapter][:curr_count].long()
                delta_flat = delta_flat.scatter_add(0, elite_indices, elite_values)
            delta_weight = delta_flat.view(self.out_features, self.in_features)

            if self.training and candidate_indices.numel() > 0:
                if not delta_weight.requires_grad:
                    delta_weight.requires_grad_(True)
                delta_weight.register_hook(self.get_candidate_grad_hook(active_adapter, candidate_indices))

            x_cast = self._cast_input_dtype(x, delta_weight.dtype)
            result = result + F.linear(x_cast, delta_weight, bias=None).to(dtype=result.dtype)

        return result.to(previous_dtype)

    def __repr__(self) -> str:
        return "dss." + super().__repr__()
