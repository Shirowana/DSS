from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from math import ceil
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers.pytorch_utils import Conv1D

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .optimizer import get_optimizer_betas, require_dss_optimizer, rewrite_slot_state
from .shared_basis import SharedBasisEntry


@dataclass
class AdapterRuntime:
    phase: str = "stage1"
    curr_count: int = 0  #当前精英个数
    steady_phase: int = 0
    current_step: int = 0   #位于当前update_interval中的第几步
    update_rounds: int = 0  #update_interval已进行了多少次
    update_flag: bool = False  #当前这一步是否正在 stage2 的统计区间里
    stage2_start_step: int = -1  #真正开始 stage2 更新时的全局步数


@dataclass
class AdapterHyperParams:
    grad_store_steps: int
    candidate_size: int
    stage2_enabled: bool
    steady_stage_ratio: float  #steady_phase的占总步数的占比
    update_interval: int
    update_counts: int
    update_margin: float
    low: int
    up: int


@dataclass
class StageReport:
    refreshed: bool = False
    updated: bool = False
    promoted_slots: int = 0
    pruned_slots: int = 0
    grown_slots: int = 0
    skipped_slots: int = 0


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
    other_param_names = (
        "coefficient_indices",
        "candidate_indices",
        "elite_bitset",
        "basis_A_inv",
        "basis_B_inv",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.coefficient = nn.ParameterDict({})
        self.coefficient_indices = BufferDict({})
        self.candidate_indices = BufferDict({})
        self.elite_bitset = BufferDict({})
        self.basis_A_inv = BufferDict({})
        self.basis_B_inv = BufferDict({})
        self.group_scale_log_param: dict[str, nn.Parameter] = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        self.quantile_lr = 0.01
        self.quantile_alpha = 0.0
        self.quantile_mode = os.environ.get("DSS_QUANTILE_MODE", "sgd").strip().lower()
        if self.quantile_mode not in {"sgd", "oracle"}:
            raise ValueError(f"Unsupported DSS quantile mode: {self.quantile_mode!r}. Use 'sgd' or 'oracle'.")
        self.delta_scale = float(os.environ.get("DSS_DELTA_SCALE", "1.0"))

        self.runtime: dict[str, AdapterRuntime] = {}
        self.hparams: dict[str, AdapterHyperParams] = {}
        self.grad_cache: dict[str, Optional[torch.Tensor]] = {}
        self.grad_count: dict[str, int] = {}
        self.search_quantile_estimator: dict[str, GPUQuantileEstimator] = {}
        self.candidate_grad_sums: dict[str, Optional[torch.Tensor]] = {}
        self.candidate_grad_sq_sums: dict[str, Optional[torch.Tensor]] = {}
        self.last_promoted_slot_positions: dict[str, torch.Tensor] = {}
        self.last_promoted_flat_indices: dict[str, torch.Tensor] = {}
        self.basis_group_name: dict[str, str] = {}
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

    def _adapter_label(self, adapter_name: str) -> str:
        return str(adapter_name)

    def _group_scale(self, adapter_name: str, dtype: torch.dtype) -> torch.Tensor:
        return self.group_scale_log_param[adapter_name].exp().to(dtype=dtype)

    #把候选池全部清空
    def clear_candidate_state(self, adapter_name: str) -> None:
        device = self._adapter_device(adapter_name)
        self.candidate_indices[adapter_name] = torch.empty(0, device=device, dtype=torch.long)
        self.grad_cache[adapter_name] = None
        self.grad_count[adapter_name] = 0
        self.candidate_grad_sums[adapter_name] = None
        self.candidate_grad_sq_sums[adapter_name] = None

    @torch.no_grad()
    def export_sparse_checkpoint(self, adapter_name: str) -> dict[str, torch.Tensor]:
        runtime = self.runtime[adapter_name]
        curr_count = runtime.curr_count
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

        runtime = self.runtime[adapter_name]
        runtime.phase = "stage1"
        runtime.curr_count = curr_count
        runtime.steady_phase = 0
        runtime.current_step = 0
        runtime.update_rounds = 0
        runtime.update_flag = False
        runtime.stage2_start_step = -1

        if adapter_name in self.search_quantile_estimator:
            self.search_quantile_estimator[adapter_name].reset()
        self.clear_candidate_state(adapter_name)
        self.last_promoted_slot_positions.pop(adapter_name, None)
        self.last_promoted_flat_indices.pop(adapter_name, None)

    #生成一批新的 candidate，并初始化它们的统计缓存
    def refresh_candidate_batch(self, adapter_name: str) -> None:
        hparams = self.hparams[adapter_name]
        device = self._adapter_device(adapter_name)
        dense_numel = self.out_features * self.in_features
        elite_bitset = self.elite_bitset[adapter_name]
        available = dense_numel - int(elite_bitset.sum().item())
        target_k = min(hparams.candidate_size, max(available, 0))
        sampled = torch.empty(0, device=device, dtype=torch.long)
        sample_k = max(target_k * 4, target_k)
        while sampled.numel() < target_k and target_k > 0:
            new_indices = torch.randint(0, dense_numel, (sample_k,), device=device)
            new_indices = torch.unique(new_indices)
            new_indices = new_indices[~elite_bitset[new_indices]]
            sampled = torch.unique(torch.cat([sampled, new_indices], dim=0))
            sample_k = min(dense_numel, max(sample_k * 2, target_k))
            if sampled.numel() >= available:
                break
        sampled = sampled[:target_k]
        self.candidate_indices[adapter_name] = sampled
        self.grad_cache[adapter_name] = None
        self.grad_count[adapter_name] = 0
        self.candidate_grad_sums[adapter_name] = None
        self.candidate_grad_sq_sums[adapter_name] = None

    # Scheduling
    #判断什么时候结束steady_phase,进入stage2
    def advance_update_state(self, adapter_name: str) -> bool:
        runtime = self.runtime[adapter_name]
        hparams = self.hparams[adapter_name]
        runtime.update_flag = False
        window_size = min(hparams.grad_store_steps, hparams.update_interval)
        #本周期里，从第几步开始进入候选统计区间
        stage2_window_start = max(1, hparams.update_interval - window_size + 1)

        if runtime.phase != "stage2" or not hparams.stage2_enabled:
            return False
        if runtime.steady_phase > 0:
            runtime.steady_phase -= 1
            runtime.current_step = 0
            self.clear_candidate_state(adapter_name)
            return False

        next_step = runtime.current_step + 1  #当前步数的下一步
        if next_step > hparams.update_interval:
            next_step = 1
        runtime.current_step = next_step
        #如果当前正好走到  stage2 候选统计区间的起点，就刷新一批新的候选
        if next_step == stage2_window_start:
            self.refresh_candidate_batch(adapter_name)

        runtime.update_flag = next_step >= stage2_window_start
        if not runtime.update_flag:
            self.clear_candidate_state(adapter_name)
        return runtime.update_flag

    # Stage decisions

    #计算平均梯度绝对值
    def compute_x_mean(self, adapter_name: str) -> torch.Tensor:
        grad_cache = self.grad_cache[adapter_name]
        grad_count = self.grad_count.get(adapter_name, 0)
        if grad_cache is None or grad_cache.numel() == 0 or grad_count <= 0:
            return torch.empty(0, device=self.coefficient_indices[adapter_name].device, dtype=torch.float32)
        return (grad_cache / float(grad_count)).abs()
    
    #根据mean info更新阈值估计器
    def update_distribution(self, adapter_name: str, x_mean: torch.Tensor) -> None:
        if x_mean.numel() == 0:
            return
        candidates_data = x_mean.reshape(-1).float()
        search_estimator = self.search_quantile_estimator[adapter_name]
        if self.quantile_mode == "oracle":
            if candidates_data.max() > 0:
                search_estimator.quantile.data.fill_(torch.quantile(candidates_data, search_estimator.q))
            return

        if search_estimator.quantile.item() == 0.0 and candidates_data.max() > 0:
            init_val = torch.quantile(candidates_data, search_estimator.q)
            search_estimator.quantile.data.fill_(init_val)

        batch_size = 1000
        total_samples = candidates_data.shape[0]
        perm = torch.randperm(total_samples, device=candidates_data.device)
        shuffled = candidates_data[perm]
        for start in range(0, total_samples, batch_size):
            search_estimator.update(shuffled[start : start + batch_size])


    def select_location(self, adapter_name: str, x_mean: torch.Tensor, remaining_budget: int) -> torch.Tensor:
        if x_mean.numel() == 0 or remaining_budget <= 0:
            return torch.empty(0, device=x_mean.device, dtype=torch.long)
        hparams = self.hparams[adapter_name]
        threshold = self.search_quantile_estimator[adapter_name].get_quantile()
        numel = x_mean.numel()
        real_up = min(hparams.up, remaining_budget, numel)
        real_low = min(hparams.low, remaining_budget, numel)
        candidate_mask = x_mean.float() > threshold
        num_candidates = int(candidate_mask.sum().item())

        if num_candidates > real_up:
            selected = torch.argsort(x_mean.float(), descending=True)[:real_up]
        elif num_candidates < real_low:
            if real_low == 0:
                selected = torch.empty(0, device=x_mean.device, dtype=torch.long)
            else:
                selected = torch.argsort(x_mean.float(), descending=True)[:real_low]
        else:
            selected = torch.nonzero(candidate_mask, as_tuple=True)[0]
        return selected

    #算出当前这一轮，prune/grow的个数与具体位置
    def select_stage2_slots(
        self,
        adapter_name: str,
        total_steps: int,
        avg_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        runtime = self.runtime[adapter_name]
        hparams = self.hparams[adapter_name]
        current_round = runtime.update_rounds + 1
        if runtime.stage2_start_step < 0:
            raise RuntimeError(f"Adapter {adapter_name!r} has no valid `stage2_start_step` recorded.")
        remaining_stage2_steps = max(1, total_steps - runtime.stage2_start_step)
        total_rounds = max(1, ceil(remaining_stage2_steps / hparams.update_interval))
        progress = min(current_round, total_rounds) / total_rounds
        k_t = hparams.update_counts if current_round <= 1 else max(0, int(hparams.update_counts * (1.0 - progress)))
        requested = min(k_t, runtime.curr_count, int(self.candidate_indices[adapter_name].numel()))
        if requested <= 0:
            empty = torch.empty(0, device=avg_grad.device, dtype=torch.long)
            return empty, empty, 0

        elite_scores = self.coefficient[adapter_name][: runtime.curr_count].abs()
        candidate_scores = avg_grad.abs()
        _, prune_slots = torch.topk(
            elite_scores,
            k=min(requested, elite_scores.numel()),
            largest=False,
            sorted=True,
        )
        grow_scores, grow_slots = torch.topk(
            candidate_scores.float(),
            k=min(requested, candidate_scores.numel()),
            largest=True,
            sorted=True,
        )
        if prune_slots.numel() != requested or grow_slots.numel() != requested:
            raise RuntimeError(
                f"Stage2 slot selection size mismatch for adapter {adapter_name!r}: "
                f"requested {requested}, got prune={prune_slots.numel()} and grow={grow_slots.numel()}."
            )

        prune_flat = self.coefficient_indices[adapter_name][prune_slots].long()
        grow_flat = self.candidate_indices[adapter_name][grow_slots].long()
        overlap = torch.isin(grow_flat, prune_flat)
        if overlap.any():
            raise RuntimeError(
                f"Stage2 selection overlap detected for adapter {adapter_name!r}; "
                "candidate sampling or elite membership is inconsistent."
            )

        return prune_slots, grow_slots, requested

    # State mutation and optimizer interaction
    
    #stage1 时：把 candidate 梯度累计成 grad_cache
    #stage2 时：把 candidate 梯度累计成 sum / sq_sum / count
    @torch.no_grad()
    def collect_probe_grads(
        self,
        adapter_name: str,
        candidate_grads: torch.Tensor,
        collect_mode: str,
    ) -> None:
        if candidate_grads.numel() == 0:
            return
        device = self._adapter_device(adapter_name)

        if collect_mode == "stage1":
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
            elif grad_cache.device != device:
                self.grad_cache[adapter_name] = grad_cache.to(device=device, dtype=torch.float32)
            self.grad_cache[adapter_name].add_(candidate_grads)
            self.grad_count[adapter_name] += 1
        elif collect_mode == "stage2":
            candidate_grads = candidate_grads.detach().to(device=device, dtype=torch.float32)
            grad_sums = self.candidate_grad_sums.get(adapter_name)
            grad_sq_sums = self.candidate_grad_sq_sums.get(adapter_name)
            if grad_sums is None:
                grad_sums = torch.zeros_like(candidate_grads, device=device, dtype=torch.float32)
                self.candidate_grad_sums[adapter_name] = grad_sums
            elif grad_sums.numel() != candidate_grads.numel():
                raise RuntimeError(
                    f"Stage2 grad-sum size mismatch for adapter {adapter_name!r}: "
                    f"expected {grad_sums.numel()}, got {candidate_grads.numel()}."
                )
            elif grad_sums.device != device:
                grad_sums = grad_sums.to(device=device, dtype=torch.float32)
                self.candidate_grad_sums[adapter_name] = grad_sums
            if grad_sq_sums is None:
                grad_sq_sums = torch.zeros_like(candidate_grads, device=device, dtype=torch.float32)
                self.candidate_grad_sq_sums[adapter_name] = grad_sq_sums
            elif grad_sq_sums.numel() != candidate_grads.numel():
                raise RuntimeError(
                    f"Stage2 grad-sq-sum size mismatch for adapter {adapter_name!r}: "
                    f"expected {grad_sq_sums.numel()}, got {candidate_grads.numel()}."
                )
            elif grad_sq_sums.device != device:
                grad_sq_sums = grad_sq_sums.to(device=device, dtype=torch.float32)
                self.candidate_grad_sq_sums[adapter_name] = grad_sq_sums
            self.candidate_grad_sums[adapter_name].add_(candidate_grads)
            self.candidate_grad_sq_sums[adapter_name].addcmul_(
                candidate_grads,
                candidate_grads,
            )

    def get_core_grad_hook(
        self,
        adapter_name: str,
        candidate_indices: torch.Tensor,
        collect_mode: str,
    ):
        def hook(grad_core: torch.Tensor) -> torch.Tensor:
            candidate_grads = grad_core.reshape(-1).index_select(0, candidate_indices.long())
            self.collect_probe_grads(adapter_name, candidate_grads, collect_mode)
            return grad_core
        '''
        hook 拿到梯度矩阵 grad_core
        从里面按 candidate_indices 抽出 candidate 对应位置的梯度
        送进 collect_probe_grads(...)
        '''
        return hook

    #将晋升的参数写入
    def apply_stage1_promotions(
        self,
        adapter_name: str,
        selected: torch.Tensor,
    ) -> int:
        if selected.numel() == 0:
            device = self._adapter_device(adapter_name)
            self.last_promoted_slot_positions[adapter_name] = torch.empty(0, device=device, dtype=torch.long)
            self.last_promoted_flat_indices[adapter_name] = torch.empty(0, device=device, dtype=torch.long)
            return 0

        runtime = self.runtime[adapter_name]
        start = runtime.curr_count
        end = start + int(selected.numel())
        new_indices = self.candidate_indices[adapter_name][selected]
        self.coefficient_indices[adapter_name][start:end] = new_indices
        self.coefficient[adapter_name].data[start:end].zero_()
        self.elite_bitset[adapter_name][new_indices] = True
        self.last_promoted_slot_positions[adapter_name] = torch.arange(
            start,
            end,
            device=new_indices.device,
            dtype=torch.long,
        )
        self.last_promoted_flat_indices[adapter_name] = new_indices.clone()
        runtime.curr_count = end
        return int(selected.numel())

    #各种阶段的判断，状态机
    def check_reinitiate(
        self,
        adapter_name: str,
        total_steps: int,
        global_step: int,
    ) -> StageReport:
        report = StageReport()
        runtime = self.runtime[adapter_name]
        adapter_label = self._adapter_label(adapter_name)
        total_budget = int(self.coefficient[adapter_name].numel())
        if runtime.phase != "stage1" or self.grad_count.get(adapter_name, 0) < self.hparams[adapter_name].grad_store_steps:
            return report

        remaining_budget = total_budget - runtime.curr_count
        if remaining_budget <= 0:
            runtime.phase = "stage2"
            runtime.steady_phase = ceil(self.hparams[adapter_name].steady_stage_ratio * total_steps)
            runtime.current_step = 0
            runtime.update_flag = False
            runtime.stage2_start_step = global_step + runtime.steady_phase
            self.clear_candidate_state(adapter_name)
            print(f"{adapter_label} 已切换到 stage2，steady={runtime.steady_phase}")
            return report

        x_mean = self.compute_x_mean(adapter_name)
        self.update_distribution(adapter_name, x_mean)
        selected = self.select_location(adapter_name, x_mean, remaining_budget)
        promoted_slots = self.apply_stage1_promotions(adapter_name, selected)
        if promoted_slots > 0:
            print(f"{adapter_label} stage1 晋升 {promoted_slots} 个参数，当前已激活 {runtime.curr_count} / {total_budget}")

        if runtime.curr_count >= total_budget:
            runtime.phase = "stage2"
            runtime.steady_phase = ceil(self.hparams[adapter_name].steady_stage_ratio * total_steps)
            runtime.current_step = 0
            runtime.update_flag = False
            runtime.stage2_start_step = global_step + runtime.steady_phase
            self.clear_candidate_state(adapter_name)
            print(f"{adapter_label} 已切换到 stage2，steady={runtime.steady_phase}")
        else:
            self.refresh_candidate_batch(adapter_name)

        report = StageReport(refreshed=True, promoted_slots=promoted_slots)
        return report


    #prune/grow的写入
    def apply_stage2_replacements(
        self,
        adapter_name: str,
        prune_slots: torch.Tensor,
        grow_slots: torch.Tensor,
        avg_grad: torch.Tensor,
        avg_grad_sq: torch.Tensor,
        optimizer=None,
        grad_accumulation_steps: int = 1,
    ) -> StageReport:
        # Stage2 replacement is a strict one-to-one swap:
        # prune the selected elite slots, write in the chosen grow candidates,
        # zero the new coefficient values, then rewrite optimizer state if needed.
        if prune_slots.numel() == 0 or grow_slots.numel() == 0:
            return StageReport()

        if prune_slots.numel() != grow_slots.numel():
            raise RuntimeError(
                f"Stage2 replacement size mismatch for adapter {adapter_name!r}: "
                f"prune={prune_slots.numel()} vs grow={grow_slots.numel()}."
            )

        incoming_indices = self.candidate_indices[adapter_name][grow_slots]
        old_indices = self.coefficient_indices[adapter_name][prune_slots].clone()
        incoming_avg_grad = avg_grad[grow_slots].clone()
        incoming_avg_grad_sq = avg_grad_sq[grow_slots].clone()
        probe_steps = float(min(self.hparams[adapter_name].grad_store_steps, self.hparams[adapter_name].update_interval))
        incoming_ages = torch.full(
            (prune_slots.numel(),),
            probe_steps / max(float(grad_accumulation_steps), 1.0),
            device=incoming_avg_grad.device,
            dtype=torch.float32,
        )

        betas = get_optimizer_betas(optimizer, self.coefficient[adapter_name]) if optimizer is not None else None
        if betas is not None and incoming_ages.numel() > 0:
            beta1, beta2 = betas
            incoming_avg_grad *= 1.0 - torch.pow(
                torch.full_like(incoming_ages, beta1),
                incoming_ages,
            )
            incoming_avg_grad_sq *= 1.0 - torch.pow(
                torch.full_like(incoming_ages, beta2),
                incoming_ages,
            )

        self.elite_bitset[adapter_name][old_indices[: prune_slots.numel()]] = False
        self.coefficient_indices[adapter_name][prune_slots] = incoming_indices[: prune_slots.numel()]
        self.coefficient[adapter_name].data[prune_slots] = 0.0
        self.elite_bitset[adapter_name][incoming_indices[: prune_slots.numel()]] = True

        if optimizer is not None:
            rewrite_slot_state(
                optimizer,
                self.coefficient[adapter_name],
                prune_slots,
                age=incoming_ages[: prune_slots.numel()],
                exp_avg=incoming_avg_grad[: prune_slots.numel()],
                exp_avg_sq=incoming_avg_grad_sq[: prune_slots.numel()],
            )

        return StageReport(
            updated=True,
            pruned_slots=int(prune_slots.numel()),
            grown_slots=int(prune_slots.numel()),
        )

    #stage2的总入口函数，类似check reinitiate
    def run_stage2_update(
        self,
        adapter_name: str,
        total_steps: int,
        optimizer=None,
        grad_accumulation_steps: int = 1,
    ) -> StageReport:
        runtime = self.runtime[adapter_name]
        hparams = self.hparams[adapter_name]
        adapter_label = self._adapter_label(adapter_name)
        if runtime.phase != "stage2":
            return StageReport()
        if not hparams.stage2_enabled:
            return StageReport()
        if runtime.steady_phase > 0:
            return StageReport()
        if runtime.current_step != hparams.update_interval:
            return StageReport()

        probe_steps = min(hparams.grad_store_steps, hparams.update_interval)
        grad_sums = self.candidate_grad_sums.get(adapter_name)
        grad_sq_sums = self.candidate_grad_sq_sums.get(adapter_name)
        if probe_steps <= 0 or grad_sums is None or grad_sq_sums is None or grad_sums.numel() == 0:
            return StageReport()
        if optimizer is not None:
            require_dss_optimizer(optimizer)

        avg_grad = grad_sums / float(probe_steps)
        avg_grad_sq = grad_sq_sums / float(probe_steps)
        prune_slots, grow_slots, requested = self.select_stage2_slots(
            adapter_name,
            total_steps,
            avg_grad,
        )

        if prune_slots.numel() == 0 or grow_slots.numel() == 0:
            report = StageReport(updated=True, skipped_slots=max(requested, 0))
            if requested > 0:
                print(f"{adapter_label} stage2 到更新点，但本轮未替换参数")
        else:
            report = self.apply_stage2_replacements(
                adapter_name,
                prune_slots,
                grow_slots,
                avg_grad,
                avg_grad_sq,
                optimizer=optimizer,
                grad_accumulation_steps=grad_accumulation_steps,
            )
            report.skipped_slots = max(requested - report.grown_slots, 0)
            print(
                f"{adapter_label} stage2 替换：prune={report.pruned_slots}，"
                f"grow={report.grown_slots}，skip={report.skipped_slots}"
            )

        self.clear_candidate_state(adapter_name)
        runtime.update_flag = False
        runtime.current_step = 0
        runtime.update_rounds += 1
        return report

    # 参考loca
    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        curr_count = self.runtime[adapter_name].curr_count
        slot_values = self.coefficient[adapter_name][:curr_count]
        flat_indices = self.coefficient_indices[adapter_name][:curr_count]
        A_inv = self.basis_A_inv[adapter_name]
        B_inv = self.basis_B_inv[adapter_name]
        group_scale = self._group_scale(adapter_name, A_inv.dtype)

        if slot_values.numel() == 0:
            return torch.zeros(
                (A_inv.shape[0], B_inv.shape[1]),
                device=A_inv.device,
                dtype=A_inv.dtype,
            )

        flat_indices = flat_indices.long()
        rows = torch.div(flat_indices, B_inv.shape[1], rounding_mode="floor")
        cols = flat_indices.remainder(B_inv.shape[1])
        A_cols = A_inv[:, rows]
        B_rows = B_inv[cols, :]
        delta_weight = torch.einsum("k,ok,ki->oi", slot_values.to(dtype=A_inv.dtype), A_cols, B_rows)
        return group_scale * delta_weight
        '''目前这个方法已经节省了大部分开销，无法避免的是生成一张完整的 dense delta_weight
        ，然后如果放在训练前向里，还要再做一次 x @ delta_w.T'''

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

        stage2_enabled: bool,
        steady_stage_ratio: float,
        update_interval: int,
        update_counts: int,
        update_margin: float,

        basis_group_name: str,
        shared_basis: SharedBasisEntry,
        group_scale_log: nn.Parameter,

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
            stage2_enabled=stage2_enabled,
            steady_stage_ratio=steady_stage_ratio,
            update_interval=update_interval,
            update_counts=update_counts,
            update_margin=update_margin,
            basis_group_name=basis_group_name,
            shared_basis=shared_basis,
            group_scale_log=group_scale_log,
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
        stage2_enabled: bool,
        steady_stage_ratio: float,
        update_interval: int,
        update_counts: int,
        update_margin: float,
        basis_group_name: str,
        shared_basis: SharedBasisEntry,
        group_scale_log: nn.Parameter,
        module_name: Optional[str] = None,
    ) -> None:
        core_shape = tuple(shared_basis.shape)
        shared_basis.validate(core_shape)
        base_shape = tuple(self.get_base_layer().weight.shape)
        if core_shape != base_shape:
            raise ValueError(
                f"Shared basis group {basis_group_name!r} expects shape {core_shape}, but base layer has shape {base_shape}."
            )

        if isinstance(low, float) or isinstance(up, float):
            raise ValueError("`low` and `up` must be integers in the current DSS implementation.")

        resolved_low = int(low)
        resolved_up = int(up)
        if resolved_low < 0 or resolved_up < 0:
            raise ValueError("`low` and `up` must be non-negative integers.")
        if resolved_low > resolved_up:
            raise ValueError("`low` cannot exceed `up` after conversion.")

        device = self.get_base_layer().weight.device
        self.coefficient[adapter_name] = nn.Parameter(torch.zeros(n_frequency, device=device, dtype=torch.float32))
        self.coefficient_indices[adapter_name] = torch.zeros(n_frequency, device=device, dtype=torch.long)
        self.elite_bitset[adapter_name] = torch.zeros(self.out_features * self.in_features, device=device, dtype=torch.bool)
        self.basis_A_inv[adapter_name] = shared_basis.A_inv.to(device=device, dtype=torch.float32)
        self.basis_B_inv[adapter_name] = shared_basis.B_inv.to(device=device, dtype=torch.float32)
        self.group_scale_log_param[adapter_name] = group_scale_log
        self.basis_group_name[adapter_name] = basis_group_name
        self.module_name[adapter_name] = module_name or ""

        self.hparams[adapter_name] = AdapterHyperParams(
            grad_store_steps=grad_store_steps,
            candidate_size=candidate_size,
            stage2_enabled=stage2_enabled,
            steady_stage_ratio=steady_stage_ratio,
            update_interval=update_interval,
            update_counts=update_counts,
            update_margin=update_margin,
            low=resolved_low,
            up=resolved_up,
        )
        self.runtime[adapter_name] = AdapterRuntime()
        self.search_quantile_estimator[adapter_name] = GPUQuantileEstimator(
            target_quantile=1.0 - ratio,
            lr=self.quantile_lr,
            alpha=self.quantile_alpha,
            device=device,
        )
        self.clear_candidate_state(adapter_name)
        self.set_adapter(self.active_adapters)

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

        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.coefficient:
                continue

            runtime = self.runtime[active_adapter]
            if runtime.phase == "stage1" and self.candidate_indices[active_adapter].numel() == 0:
                self.refresh_candidate_batch(active_adapter)
            candidate_indices = self.candidate_indices[active_adapter]
            if runtime.curr_count == 0 and candidate_indices.numel() == 0:
                continue

            B_inv = self.basis_B_inv[active_adapter]
            A_inv = self.basis_A_inv[active_adapter]
            group_scale = self._group_scale(active_adapter, A_inv.dtype)
            x_basis = torch.matmul(self._cast_input_dtype(x, B_inv.dtype), B_inv.transpose(0, 1))
            #根据当前阶段决定是否收集 candidate probe
            if runtime.phase == "stage1":
                collect_mode = "stage1"
            elif runtime.phase == "stage2" and runtime.update_flag:
                collect_mode = "stage2"
            else:
                collect_mode = "off"

            lambda_flat = x_basis.new_zeros(self.out_features * self.in_features)
            if runtime.curr_count > 0:
                elite_values = self.coefficient[active_adapter][: runtime.curr_count].to(dtype=x_basis.dtype)
                elite_indices = self.coefficient_indices[active_adapter][: runtime.curr_count].long()
                lambda_flat = lambda_flat.scatter_add(0, elite_indices, elite_values)
            lambda_dense = lambda_flat.view(self.out_features, self.in_features)
            if collect_mode != "off" and candidate_indices.numel() > 0:
                if not lambda_dense.requires_grad:
                    lambda_dense.requires_grad_(True)
                #如果当前需要 probe candidate，就给 lambda_dense 注册 hook
                lambda_dense.register_hook(self.get_core_grad_hook(active_adapter, candidate_indices, collect_mode))

            y_core = F.linear(x_basis, lambda_dense)
            y_delta = torch.matmul(y_core, A_inv.transpose(0, 1))
            y_delta = group_scale * y_delta
            result = result + (self.delta_scale * y_delta).to(dtype=result.dtype)
            '''
            base:   y_base = base_layer(x)

            DSS:    x_basis = x @ B_inv.T
                    lambda_dense = scatter(coefficient, coefficient_indices)
                    y_core = x_basis @ lambda_dense.T
                    y_delta = y_core @ A_inv.T

            output: y = y_base + y_delta
            '''

        return result.to(previous_dtype)

    def __repr__(self) -> str:
        return "dss." + super().__repr__()
