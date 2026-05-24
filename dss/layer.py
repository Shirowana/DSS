from __future__ import annotations

import os
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
from .scoring import ScoreStats, compute_score, required_stats_for


@dataclass
class SearchRuntime:
    curr_count: int = 0
    total_steps: int = 0
    refresh_rounds: int = 0
    last_logged_step: int = -1
    init_phase: bool = False
    init_done: bool = False
    pending_init_threshold: Optional[float] = None


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


def _ddp_sampling_sync_enabled() -> bool:
    return os.environ.get("DSS_DISABLE_DDP_CANDIDATE_SYNC", "0") != "1"


def _elite_bitset_enabled() -> bool:
    """Debug switch for isolating dense bitset memory from the rest of DSS.

    The bitset normally prevents future candidate batches from re-sampling
    positions that already entered the elite pool. Disabling it is only meant
    for short memory probes: duplicate elite positions may be selected, so the
    resulting run should not be treated as a valid training experiment.
    """

    return os.environ.get("DSS_DISABLE_ELITE_BITSET", "0") != "1"


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
        self.threshold_log_every_steps: dict[str, int] = {}
        self.score_method: dict[str, str] = {}
        self.score_eps: dict[str, float] = {}
        self.module_name: dict[str, str] = {}
        self.init_enabled: dict[str, bool] = {}
        self.init_steps: dict[str, int] = {}
        self.init_candidate_ratio: dict[str, float] = {}
        self.init_seed_mode: dict[str, str] = {}
        self.abs_grad_sum: dict[str, Optional[torch.Tensor]] = {}
        self.grad_sq_sum: dict[str, Optional[torch.Tensor]] = {}
        self.theta0_abs_cache: dict[str, Optional[torch.Tensor]] = {}

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
        self.abs_grad_sum[adapter_name] = None
        self.grad_sq_sum[adapter_name] = None
        self.theta0_abs_cache[adapter_name] = None

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

        if _elite_bitset_enabled():
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

    def _broadcast_indices(self, sampled: torch.Tensor, device: torch.device) -> torch.Tensor:
        if _dist_world_size() == 1 or not _ddp_sampling_sync_enabled():
            return sampled

        if _dist_rank() == 0:
            size_tensor = torch.tensor([int(sampled.numel())], device=device, dtype=torch.long)
        else:
            size_tensor = torch.zeros(1, device=device, dtype=torch.long)
        dist.broadcast(size_tensor, src=0)
        target_size = int(size_tensor.item())

        if _dist_rank() != 0:
            sampled = torch.empty(target_size, device=device, dtype=torch.long)
        elif sampled.numel() != target_size:
            sampled = sampled[:target_size]

        if target_size > 0:
            dist.broadcast(sampled, src=0)
        return sampled

    def _sync_selected_locations(self, selected: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Keep DDP search state identical by promoting rank-0 selections everywhere.

        Candidate batches are already broadcast from rank 0, but the scores are
        computed from each rank's local batch. If each rank promotes its own
        selected positions, `curr_count` and `elite_bitset` can drift apart and
        later collective broadcasts can be called for different modules. Reusing
        the same compact broadcast helper here keeps all sparse-search state in
        lockstep while leaving the threshold/scoring logic unchanged.
        """

        return self._broadcast_indices(selected.to(device=device, dtype=torch.long), device)

    def _score_requires_theta0_abs(self, adapter_name: str) -> bool:
        return "theta0_abs" in required_stats_for(self.score_method[adapter_name])

    def _cache_theta0_abs_for_candidates(self, adapter_name: str, sampled: torch.Tensor, device: torch.device) -> None:
        if sampled.numel() == 0 or not self._score_requires_theta0_abs(adapter_name):
            self.theta0_abs_cache[adapter_name] = None
            return
        base_weight_flat = self.get_base_layer().weight.detach().reshape(-1)
        theta0 = base_weight_flat.index_select(0, sampled.long())
        self.theta0_abs_cache[adapter_name] = theta0.abs().to(
            device=device, dtype=torch.float32
        )

    def refresh_candidate_batch(self, adapter_name: str) -> None:
        device = self._adapter_device(adapter_name)
        dense_numel = self.out_features * self.in_features
        elite_bitset = self.elite_bitset[adapter_name] if _elite_bitset_enabled() else None
        runtime = self.runtime[adapter_name]
        available = dense_numel - runtime.curr_count if elite_bitset is not None else dense_numel
        target_k = min(self.candidate_size[adapter_name], max(available, 0))
        if target_k <= 0:
            self.clear_candidate_state(adapter_name)
            return

        sampled = torch.empty(0, device=device, dtype=torch.long)
        if _dist_rank() == 0:
            sample_k = max(target_k * 4, target_k)
            proposal_chunks = []
            while True:
                proposal = torch.randint(0, dense_numel, (sample_k,), device=device)
                proposal = torch.unique(proposal)
                if elite_bitset is not None:
                    proposal = proposal[~elite_bitset[proposal]]
                if proposal.numel() > 0:
                    proposal_chunks.append(proposal)
                buffered = sum(chunk.numel() for chunk in proposal_chunks)
                if buffered >= target_k or buffered >= available or sample_k >= dense_numel:
                    if proposal_chunks:
                        sampled = torch.unique(torch.cat(proposal_chunks, dim=0))
                    break
                if buffered == 0 and sample_k >= dense_numel:
                    break
                sample_k = min(dense_numel, max(sample_k * 2, target_k))
            sampled = sampled[:target_k]

        sampled = self._broadcast_indices(sampled, device)

        self.candidate_indices[adapter_name] = sampled
        self.grad_cache[adapter_name] = None
        self.grad_count[adapter_name] = 0
        self._cache_theta0_abs_for_candidates(adapter_name, sampled, device)

    def refresh_init_candidate_batch(self, adapter_name: str) -> None:
        device = self._adapter_device(adapter_name)
        rows, cols = self.out_features, self.in_features
        ratio = self.init_candidate_ratio[adapter_name]
        block_size = 64
        sampled = torch.empty(0, device=device, dtype=torch.long)
        if _dist_rank() == 0:
            if rows % block_size == 0 and cols % block_size == 0:
                block_rows = rows // block_size
                block_cols = cols // block_size
                num_blocks = block_rows * block_cols
                block_numel = block_size * block_size
                take = max(1, int(round(block_numel * ratio)))

                local_perm = torch.randperm(block_numel, device=device)[:take]
                local_rows = local_perm // block_size
                local_cols = local_perm % block_size

                block_row_ids = torch.arange(block_rows, device=device, dtype=torch.long)
                block_col_ids = torch.arange(block_cols, device=device, dtype=torch.long)
                grid_rows, grid_cols = torch.meshgrid(block_row_ids, block_col_ids, indexing="ij")
                row_offsets = (grid_rows.reshape(-1) * block_size).unsqueeze(1)
                col_offsets = (grid_cols.reshape(-1) * block_size).unsqueeze(1)

                sampled = ((row_offsets + local_rows.unsqueeze(0)) * cols + (col_offsets + local_cols.unsqueeze(0))).reshape(-1)
                if sampled.numel() != num_blocks * take:
                    raise RuntimeError(
                        f"Vectorized init sampling produced {sampled.numel()} indices, "
                        f"expected {num_blocks * take} for {self.module_name.get(adapter_name, '')}."
                    )
            else:
                blocks: list[torch.Tensor] = []
                for row_start in range(0, rows, block_size):
                    block_rows_local = min(block_size, rows - row_start)
                    for col_start in range(0, cols, block_size):
                        block_cols_local = min(block_size, cols - col_start)
                        block_numel = block_rows_local * block_cols_local
                        take = max(1, int(round(block_numel * ratio)))
                        local_perm = torch.randperm(block_numel, device=device)[:take]
                        local_rows = local_perm // block_cols_local
                        local_cols = local_perm % block_cols_local
                        flat = (row_start + local_rows) * cols + (col_start + local_cols)
                        blocks.append(flat.to(dtype=torch.long))
                sampled = torch.cat(blocks, dim=0) if blocks else torch.empty(0, device=device, dtype=torch.long)

        sampled = self._broadcast_indices(sampled, device)
        self.candidate_indices[adapter_name] = sampled
        self.grad_cache[adapter_name] = None
        self.grad_count[adapter_name] = 0
        self._cache_theta0_abs_for_candidates(adapter_name, sampled, device)

    def compute_x_mean(self, adapter_name: str) -> torch.Tensor:
        stats = ScoreStats(
            grad_sum=self.grad_cache.get(adapter_name),
            abs_grad_sum=self.abs_grad_sum.get(adapter_name),
            grad_sq_sum=self.grad_sq_sum.get(adapter_name),
            count=self.grad_count.get(adapter_name, 0),
        )
        # NOTE: `x_mean` is a legacy name kept for compatibility with the
        # existing refresh pipeline. It now represents the selected importance
        # score vector, not necessarily `abs(mean(g))`.
        return compute_score(
            method=self.score_method[adapter_name],
            stats=stats,
            theta0_abs=self.theta0_abs_cache.get(adapter_name),
            eps=self.score_eps[adapter_name],
        )

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

    def select_location_with_threshold(
        self,
        adapter_name: str,
        x_mean: torch.Tensor,
        remaining_budget: int,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        if x_mean.numel() == 0 or remaining_budget <= 0:
            return torch.empty(0, device=x_mean.device, dtype=torch.long)

        numel = x_mean.numel()
        real_up = min(self.up[adapter_name], remaining_budget, numel)
        real_low = min(self.low[adapter_name], remaining_budget, numel)
        candidate_mask = x_mean.float() > threshold
        num_candidates = int(candidate_mask.sum().item())

        if num_candidates > real_up:
            return torch.topk(x_mean.float(), k=real_up, largest=True, sorted=False).indices
        if num_candidates < real_low:
            if real_low == 0:
                return torch.empty(0, device=x_mean.device, dtype=torch.long)
            return torch.topk(x_mean.float(), k=real_low, largest=True, sorted=False).indices
        return torch.nonzero(candidate_mask, as_tuple=True)[0]

    def select_location(self, adapter_name: str, x_mean: torch.Tensor, remaining_budget: int) -> torch.Tensor:
        threshold = self.search_quantile_estimator[adapter_name].get_quantile()
        return self.select_location_with_threshold(adapter_name, x_mean, remaining_budget, threshold)

    def select_init_seed_locations(self, x_mean: torch.Tensor, remaining_budget: int, threshold: torch.Tensor) -> torch.Tensor:
        if x_mean.numel() == 0 or remaining_budget <= 0:
            return torch.empty(0, device=x_mean.device, dtype=torch.long)
        selected = torch.nonzero(x_mean.float() > threshold, as_tuple=True)[0]
        if selected.numel() > remaining_budget:
            selected = torch.topk(x_mean.float(), k=remaining_budget, largest=True, sorted=False).indices
        return selected

    @torch.no_grad()
    def maybe_log_threshold_health(
        self,
        adapter_name: str,
        x_mean: torch.Tensor,
        threshold: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        if _dist_rank() != 0 or x_mean.numel() == 0:
            return

        runtime = self.runtime[adapter_name]
        log_every = self.threshold_log_every_steps.get(adapter_name, 100)
        step = int(runtime.total_steps)
        if runtime.last_logged_step >= 0 and step - runtime.last_logged_step < log_every:
            return

        scores = x_mean.reshape(-1).float()
        threshold_value = float(threshold.detach().float().item())
        threshold_selected = int((scores > threshold).sum().item())
        module_name = self.module_name.get(adapter_name, "") or "<unknown>"
        print(
            f"[DSS threshold] module={module_name} "
            f"score_method={self.score_method[adapter_name]} "
            f"step={step} "
            f"refresh={runtime.refresh_rounds + 1} "
            f"threshold={threshold_value:.8f} "
            f"score_min={float(scores.min().item()):.8f} "
            f"score_mean={float(scores.mean().item()):.8f} "
            f"score_max={float(scores.max().item()):.8f} "
            f"threshold_selected={threshold_selected} "
            f"final_selected={int(selected.numel())}",
            flush=True,
        )
        runtime.last_logged_step = step

    def apply_stage1_promotions(self, adapter_name: str, selected: torch.Tensor) -> int:
        if selected.numel() == 0:
            return 0

        runtime = self.runtime[adapter_name]
        start = runtime.curr_count
        end = start + int(selected.numel())
        new_indices = self.candidate_indices[adapter_name][selected]
        self.coefficient_indices[adapter_name][start:end] = new_indices
        self.coefficient[adapter_name].data[start:end].zero_()
        if _elite_bitset_enabled():
            self.elite_bitset[adapter_name][new_indices] = True
        runtime.curr_count = end
        return int(selected.numel())

    def maybe_refresh_stage1(self, adapter_name: str) -> int:
        runtime = self.runtime[adapter_name]
        total_budget = int(self.coefficient[adapter_name].numel())
        if runtime.curr_count >= total_budget:
            self.clear_candidate_state(adapter_name)
            return 0

        if runtime.init_phase:
            if self.candidate_indices[adapter_name].numel() == 0:
                self.refresh_init_candidate_batch(adapter_name)
            if self.grad_count.get(adapter_name, 0) < self.init_steps[adapter_name]:
                return 0

            x_mean = self.compute_x_mean(adapter_name)
            candidates_data = x_mean.reshape(-1).float()
            threshold = torch.tensor(0.0, device=x_mean.device, dtype=torch.float32)
            if candidates_data.numel() > 0 and candidates_data.max() > 0:
                threshold = torch.quantile(candidates_data, self.search_quantile_estimator[adapter_name].q)

            selected = torch.empty(0, device=x_mean.device, dtype=torch.long)
            if self.init_seed_mode[adapter_name] == "seed_elite":
                remaining_budget = total_budget - runtime.curr_count
                selected = self.select_init_seed_locations(x_mean, remaining_budget, threshold)
            selected = self._sync_selected_locations(selected, x_mean.device)
            if self.init_seed_mode[adapter_name] == "seed_elite":
                self.apply_stage1_promotions(adapter_name, selected)

            if self.threshold_mode[adapter_name] == "sgd":
                self.search_quantile_estimator[adapter_name].quantile.data.fill_(float(threshold.item()))
            else:
                runtime.pending_init_threshold = float(threshold.item())

            runtime.init_phase = False
            runtime.init_done = True
            self.clear_candidate_state(adapter_name)
            return int(selected.numel())

        if self.candidate_indices[adapter_name].numel() == 0:
            self.refresh_candidate_batch(adapter_name)

        if self.grad_count.get(adapter_name, 0) < self.grad_store_steps[adapter_name]:
            return 0

        remaining_budget = total_budget - runtime.curr_count
        x_mean = self.compute_x_mean(adapter_name)
        if runtime.pending_init_threshold is not None and self.threshold_mode[adapter_name] == "oracle" and runtime.refresh_rounds == 0:
            threshold = torch.tensor(runtime.pending_init_threshold, device=x_mean.device, dtype=torch.float32)
            runtime.pending_init_threshold = None
        else:
            self.update_distribution(adapter_name, x_mean)
            threshold = self.search_quantile_estimator[adapter_name].get_quantile()
        selected = self.select_location_with_threshold(adapter_name, x_mean, remaining_budget, threshold)
        self.maybe_log_threshold_health(adapter_name, x_mean, threshold, selected)
        selected = self._sync_selected_locations(selected, x_mean.device)
        promoted = self.apply_stage1_promotions(adapter_name, selected)
        runtime.refresh_rounds += 1
        self.clear_candidate_state(adapter_name)
        if runtime.curr_count < total_budget:
            self.refresh_candidate_batch(adapter_name)
        return promoted

    @torch.no_grad()
    def collect_candidate_grads(self, adapter_name: str, candidate_grads: torch.Tensor) -> None:
        if candidate_grads.numel() == 0:
            return
        cache_device = self._adapter_device(adapter_name)
        candidate_grads = candidate_grads.detach().to(dtype=torch.float32)
        grad_cache = self.grad_cache.get(adapter_name)
        if grad_cache is None:
            self.grad_cache[adapter_name] = torch.zeros_like(candidate_grads, device=cache_device, dtype=torch.float32)
            self.grad_count[adapter_name] = 0
        elif grad_cache.numel() != candidate_grads.numel():
            raise RuntimeError(
                f"Stage1 grad cache size mismatch for adapter {adapter_name!r}: "
                f"expected {grad_cache.numel()}, got {candidate_grads.numel()}."
            )
        candidate_grads = candidate_grads.to(device=cache_device, dtype=torch.float32)
        self.grad_cache[adapter_name].add_(candidate_grads)
        if "grad_sq_sum" in required_stats_for(self.score_method[adapter_name]):
            if self.grad_sq_sum[adapter_name] is None:
                self.grad_sq_sum[adapter_name] = torch.zeros_like(candidate_grads, device=cache_device, dtype=torch.float32)
            self.grad_sq_sum[adapter_name].add_(candidate_grads.square())

        if "abs_grad_sum" in required_stats_for(self.score_method[adapter_name]):
            abs_candidate_grads = candidate_grads.abs()
            if self.abs_grad_sum[adapter_name] is None:
                self.abs_grad_sum[adapter_name] = torch.zeros_like(abs_candidate_grads, device=cache_device, dtype=torch.float32)
            self.abs_grad_sum[adapter_name].add_(abs_candidate_grads)
        self.grad_count[adapter_name] += 1
        self.runtime[adapter_name].total_steps += 1

    def get_candidate_grad_hook(self, adapter_name: str, candidate_indices: torch.Tensor):
        def hook(grad_merged_weight: torch.Tensor) -> torch.Tensor:
            candidate_indices_device = candidate_indices.to(device=grad_merged_weight.device, dtype=torch.long, non_blocking=True)
            candidate_grads = grad_merged_weight.reshape(-1).index_select(0, candidate_indices_device)
            self.collect_candidate_grads(adapter_name, candidate_grads)
            return grad_merged_weight

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

    def _build_merged_weight_for_forward(self, adapter_names: list[str], dtype: torch.dtype) -> torch.Tensor:
        """Build a dense merged weight for the hot forward path.

        This path favors speed/memory and uses the base weight dtype; an fp32
        merge would preserve tiny updates better but costs more memory/time.
        """

        base_weight = self.get_base_layer().weight.detach().to(dtype=dtype)
        base_flat = base_weight.reshape(-1)
        merged_flat = base_flat.clone()
        for adapter_name in adapter_names:
            curr_count = self.runtime[adapter_name].curr_count
            if curr_count == 0:
                continue
            elite_indices = self.coefficient_indices[adapter_name][:curr_count].long()
            elite_values = self.dropout_layer[adapter_name](self.coefficient[adapter_name][:curr_count])
            elite_values = elite_values.to(dtype=base_flat.dtype)
            merged_flat = merged_flat.scatter_add(0, elite_indices, elite_values)
        return merged_flat.view_as(base_weight)


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
        score_method: str = "abs_mean",
        score_eps: float = 1e-8,
        dropout: float = 0.0,
        quantile_lr: float = 0.01,
        quantile_alpha: float = 0.0,
        threshold_log_every_steps: int = 100,
        init_enabled: bool = False,
        init_steps: int = 10,
        init_candidate_ratio: float = 0.05,
        init_seed_mode: str = "threshold_only",
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
            score_method=score_method,
            score_eps=score_eps,
            dropout=dropout,
            quantile_lr=quantile_lr,
            quantile_alpha=quantile_alpha,
            threshold_log_every_steps=threshold_log_every_steps,
            init_enabled=init_enabled,
            init_steps=init_steps,
            init_candidate_ratio=init_candidate_ratio,
            init_seed_mode=init_seed_mode,
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
        score_method: str = "abs_mean",
        score_eps: float = 1e-8,
        dropout: float = 0.0,
        quantile_lr: float = 0.01,
        quantile_alpha: float = 0.0,
        threshold_log_every_steps: int = 100,
        init_enabled: bool = False,
        init_steps: int = 10,
        init_candidate_ratio: float = 0.05,
        init_seed_mode: str = "threshold_only",
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
        if score_method not in {
            "mean_abs",
            "abs_mean",
            "mean_square",
            "rms_over_param",
            "abs_mean_over_param",
            "snr",
            "newton_like",
        }:
            raise ValueError(f"Unsupported DSS score method: {score_method!r}.")
        if score_eps <= 0.0:
            raise ValueError("`score_eps` must be positive.")

        device = self.get_base_layer().weight.device
        self.coefficient[adapter_name] = nn.Parameter(torch.zeros(n_frequency, device=device, dtype=torch.float32))
        self.coefficient_indices[adapter_name] = torch.zeros(n_frequency, device=device, dtype=torch.long)
        self.candidate_indices[adapter_name] = torch.empty(0, device=device, dtype=torch.long)
        elite_bitset_size = self.out_features * self.in_features if _elite_bitset_enabled() else 0
        self.elite_bitset[adapter_name] = torch.zeros(elite_bitset_size, device=device, dtype=torch.bool)
        self.dropout_layer[adapter_name] = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.runtime[adapter_name] = SearchRuntime()
        self.grad_store_steps[adapter_name] = grad_store_steps
        self.candidate_size[adapter_name] = candidate_size
        self.low[adapter_name] = resolved_low
        self.up[adapter_name] = resolved_up
        self.threshold_mode[adapter_name] = threshold_mode
        self.threshold_log_every_steps[adapter_name] = int(threshold_log_every_steps)
        self.score_method[adapter_name] = score_method
        self.score_eps[adapter_name] = score_eps
        self.module_name[adapter_name] = module_name or ""
        self.init_enabled[adapter_name] = init_enabled
        self.init_steps[adapter_name] = init_steps
        self.init_candidate_ratio[adapter_name] = init_candidate_ratio
        self.init_seed_mode[adapter_name] = init_seed_mode
        self.search_quantile_estimator[adapter_name] = GPUQuantileEstimator(
            target_quantile=1.0 - ratio,
            lr=quantile_lr,
            alpha=quantile_alpha,
            device=device,
        )
        self.runtime[adapter_name].init_phase = init_enabled
        self.runtime[adapter_name].init_done = not init_enabled
        self.runtime[adapter_name].pending_init_threshold = None
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

        active_dss_adapters: list[str] = []
        for active_adapter in self.active_adapters:
            if active_adapter not in self.coefficient:
                continue
            curr_count = self.runtime[active_adapter].curr_count
            candidate_indices = self.candidate_indices[active_adapter]
            if curr_count > 0 or candidate_indices.numel() > 0:
                active_dss_adapters.append(active_adapter)

        if not active_dss_adapters:
            return self.base_layer(x, *args, **kwargs).to(previous_dtype)

        base_layer = self.get_base_layer()
        weight_dtype = base_layer.weight.dtype
        merged_weight = self._build_merged_weight_for_forward(active_dss_adapters, dtype=weight_dtype)
        candidate_probe_present = False
        has_active_elite = False

        for active_adapter in active_dss_adapters:
            curr_count = self.runtime[active_adapter].curr_count
            candidate_indices = self.candidate_indices[active_adapter]
            candidate_probe_present = candidate_probe_present or candidate_indices.numel() > 0
            has_active_elite = has_active_elite or curr_count > 0

        if not has_active_elite and candidate_probe_present:
            zero_dep = self.coefficient[active_dss_adapters[0]][:0].sum().to(dtype=merged_weight.dtype) * 0.0
            merged_weight = merged_weight + zero_dep

        if self.training:
            for active_adapter in active_dss_adapters:
                candidate_indices = self.candidate_indices[active_adapter]
                if candidate_indices.numel() > 0:
                    merged_weight.register_hook(self.get_candidate_grad_hook(active_adapter, candidate_indices))

        x_cast = self._cast_input_dtype(x, merged_weight.dtype)
        if isinstance(base_layer, nn.Linear):
            result = F.linear(x_cast, merged_weight, bias=base_layer.bias)
        elif isinstance(base_layer, Conv1D):
            # Conv1D uses the native HF layout: weight shape is [in_features, out_features].
            size_out = x_cast.size()[:-1] + (base_layer.nf,)
            bias = base_layer.bias
            if bias is None:
                bias = torch.zeros(base_layer.nf, device=x_cast.device, dtype=merged_weight.dtype)
            result = torch.addmm(bias, x_cast.reshape(-1, x_cast.size(-1)), merged_weight)
            result = result.view(size_out)
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        return result.to(previous_dtype)

    def __repr__(self) -> str:
        return "dss." + super().__repr__()
