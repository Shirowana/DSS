from __future__ import annotations
# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import dropout
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
import weakref


class GPUQuantileEstimator:
    def __init__(self, target_quantile, initial_value=0.0, lr=0.01, alpha=0.0, device="cuda"):
        """
        一个完全在 GPU 上运行的在线分位数估计器。
        :param target_quantile (float): 目标分位数 (例如 0.95)。
        :param lr: Base LR。
        :param alpha: 衰减系数 (在此版本中已禁用衰减)。
        :param device: PyTorch device。
        """
        self.q = target_quantile
        self.device = device
        self.quantile = nn.Parameter(torch.tensor(initial_value, device=self.device, dtype=torch.float32))
        self.base_lr = lr
        self.alpha = alpha
        self.step_count = 0
        self.optimizer = optim.SGD([self.quantile], lr=lr)
        self.quantile.requires_grad_(True)

    def _pinball_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算弹球损失（Pinball Loss）
        """
        error = x - self.quantile
        loss = torch.where(error > 0, self.q * error, (self.q - 1) * error)
        return loss.mean()

    def update(self, new_data: torch.Tensor):
        self.step_count += 1
        # 使用当前阈值的绝对值作为 Scale，实现自适应
        current_scale = abs(self.quantile.item()) + 1e-8
        # [修改] 移除时间衰减，保持估算器敏锐，适应非平稳分布
        time_decay = 1.0
        # Final LR
        real_lr = self.base_lr * current_scale * time_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = real_lr
        loss = self._pinball_loss(new_data.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_quantile(self) -> torch.Tensor:
        return self.quantile.data

    def reset(self):
        self.quantile.data.fill_(0.0)
        self.step_count = 0
        self.optimizer = optim.SGD([self.quantile], lr=self.base_lr)


# 为一个坐标张量 l 填充随机且唯一的坐标 (CPU fallback)
def fill_new_coords_old(M, N, l, k):
    n = l.shape[1]
    filled = k
    existing = set(tuple(l[:, i].tolist()) for i in range(k))
    new_coords = []
    while len(new_coords) < n - k:
        batch = torch.stack([
            torch.randint(0, M, (n - filled,)),
            torch.randint(0, N, (n - filled,))
        ], dim=1)
        for coord in batch:
            coord_tuple = tuple(coord.tolist())
            if coord_tuple not in existing:
                new_coords.append(coord_tuple)
                existing.add(coord_tuple)
                filled += 1
                if len(new_coords) == n - k:
                    break
    l[:, k:] = torch.tensor(new_coords).T
    return l


def fill_new_coords(M, N, l, k):
    """
    全 GPU 版本的坐标填充函数。
    """
    device = l.device
    dtype = l.dtype
    n = l.shape[1]
    n_needed = n - k
    if n_needed == 0:
        return l
    existing_indices = l[0, :k] * N + l[1, :k]
    try:
        all_indices_mask = torch.ones(M * N, dtype=torch.bool, device=device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"警告：无法在 GPU 上分配 {M * N} 的索引掩码 ({e})。回退到 CPU 实现。")
        return fill_new_coords_old(M, N, l, k)
    all_indices_mask[existing_indices] = False
    candidate_indices = torch.where(all_indices_mask)[0]
    n_available = candidate_indices.shape[0]
    if n_needed > n_available:
        raise ValueError(f"无法找到 {n_needed} 个新的唯一坐标。")
    perm = torch.randperm(n_available, device=device)
    selected_new_indices = candidate_indices[perm[:n_needed]]
    new_rows = selected_new_indices // N
    new_cols = selected_new_indices % N
    l[0, k:] = new_rows.to(dtype)
    l[1, k:] = new_cols.to(dtype)
    return l


def coeff_to_dense(coefs, L, M, N):
    return torch.sparse_coo_tensor(L, coefs, torch.Size([M, N])).to_dense()


class DSSMakeDeltaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, updates, locations, out_features, in_features, layer, first_k, isFull, adapter):
        ctx.save_for_backward(updates, locations)
        ctx.first_k = first_k
        ctx.isFull = isFull
        ctx.adapter = adapter
        ctx._layer_ref = weakref.ref(layer)
        delta_w_sparse = torch.sparse_coo_tensor(
            locations,
            updates,
            size=(out_features, in_features)
        )
        return delta_w_sparse

    @staticmethod
    def backward(ctx, grad_delta_w):
        updates, locations = ctx.saved_tensors
        grad_updates = None
        if ctx.needs_input_grad[0]:
            row_indices = locations[0]
            col_indices = locations[1]
            if grad_delta_w.is_sparse:
                grad_delta_w = grad_delta_w.to_dense()
                #从 M X N的大梯度矩阵中，只提取 locations 指定位置的数值。
            grad_updates = grad_delta_w[row_indices, col_indices].float()
            layer_ref = getattr(ctx, '_layer_ref', None)
            layer = layer_ref() if layer_ref else None

            if layer is not None:
                if not ctx.isFull:
                    # 动态筛选阶段：grad_cache 只存储候选参数的梯度
                    layer.grad_cache[ctx.adapter].append(grad_updates[ctx.first_k:].detach().clone())
                    #估算器稍后会计算这批数据的平均值和分位数，来筛选新的参数
                else:
                    # 稳定训练阶段
                    layer.mean_Info[ctx.adapter].append(torch.mean(grad_updates.abs()).item())
                    #用这个平均值来判断模型是否收敛。
        if hasattr(ctx, '_layer_ref'): del ctx._layer_ref
        return grad_updates, None, None, None, None, None, None, None


class DSSLayer(BaseTunerLayer):
    adapter_layer_names = ("coefficient",)
    other_param_names = (
        "n_frequency", "dropout_layer", "_disable_adapters", "coefficient_indices",
        "merged_adapters", "grad_cache", "mean_Info", "grad_store_steps",
        "select_pars", "train_location_mode",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        #前向传播时，输入会先经过这个 base_layer 得到基础输出，然后再叠加上 DSS 的稀疏增量
        self.base_layer = base_layer
        self.n_frequency = {}
        self.coefficient = nn.ParameterDict({})
        self.coefficient_indices = {}
        self.dropout_layer = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        self.curr_index = {}  #已选定的精英参数
        self.dss_make_weight = DSSMakeDeltaWeight.apply
        self.grad_cache = {}
        self.mean_Info = {}
        self.grad_store_steps = {}
        self.hook_handle = {}
        self.search_quantile_estimator = {}
        self.reinit_quantile_estimator = {}
        self.quantile_lr = {}
        self.select_pars = {}
        self.train_location_mode = {}
        self.fixed_threshold = {}  #重置阈值
        #探测base layer形状
        base_layer = self.get_base_layer()
        # 情况 1：最常见的线性层
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        # 情况 2：Conv1D 层
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, n_frequency, dropout, grad_store_steps, low, up, ratio, reinitiate_ratio,
                     ratioAdd=False, quantile_lr=0.01, quantile_alpha=0.0):
        #为一个新的 Adapter（比如名为 "dss"）分配内存、设置预算、初始化超参数，并启动那两个关键的分位数估算器
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency
        self.curr_index[adapter_name] = 0
        self.hook_handle[adapter_name] = None
        self.grad_store_steps[adapter_name] = grad_store_steps
        self.grad_cache[adapter_name] = []
        self.mean_Info[adapter_name] = []
        # 将 up（每轮最多选多少个）从比例转为整数
        if type(up) == float:
            up = int(n_frequency * up)
        # 将 low（每轮最少选多少个）从比例转为整数
        if type(low) == float:
            low = int(n_frequency * low)
        self.quantile_lr[adapter_name] = quantile_lr
        if ratioAdd:
            ratio += n_frequency / (self.in_features * self.out_features)
        else:
            ratio *= n_frequency / (self.in_features * self.out_features)
        # 打包保存这些参数，方便 select_location 函数调用
        self.select_pars[adapter_name] = [low, up, ratio, reinitiate_ratio]
        self.train_location_mode[adapter_name] = True
        # 1. 创建数值向量 (coefficient)
        self.coefficient[adapter_name] = nn.Parameter(torch.zeros(n_frequency), requires_grad=True)
        # 2. 创建坐标矩阵 (coefficient_indices)
        self.coefficient_indices[adapter_name] = torch.zeros(2, n_frequency, dtype=torch.long, requires_grad=False)
        # 调用 fill_new_coords 函数
        # k=0 表示目前没有精英参数，所有的位置都要重新随机生成。
        self.coefficient_indices[adapter_name] = fill_new_coords(
            self.out_features, self.in_features, self.coefficient_indices[adapter_name], k=0
        )
        nn.init.zeros_(self.coefficient[adapter_name])
        device = self.coefficient[adapter_name].device
        search_q = 1.0 - ratio
        self.search_quantile_estimator[adapter_name] = GPUQuantileEstimator(
            target_quantile=search_q, lr=quantile_lr, alpha=quantile_alpha, device=device)
        reinit_q = reinitiate_ratio
        self.reinit_quantile_estimator[adapter_name] = GPUQuantileEstimator(
            target_quantile=reinit_q, lr=quantile_lr, alpha=quantile_alpha, device=device)
        #配置 Dropout 层
        if dropout > 0.0:
            self.dropout_layer[adapter_name] = nn.Dropout(p=dropout)
        else:
            self.dropout_layer[adapter_name] = nn.Identity()

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def select_location(self, adapter, x_mean):
        #每隔 grad_store_steps 步调用一次。  用于筛选精英参数
        low, up, ratio, reinitiate_ratio = self.select_pars[adapter]
        if x_mean.numel() == 0: #这是从 grad_cache 里算出来的梯度均值（仅包含候选参数）
            return
        candidates_data = x_mean.float()
        # 用SGD估计，获取当前最新的动态阈值。
        estimated_threshold_tensor = self.search_quantile_estimator[adapter].get_quantile()
        # 生成 Mask
        candidates_mask = candidates_data > estimated_threshold_tensor
        num_candidates = candidates_mask.sum().item()
        #self.curr_index：当前已经有多少个参数是精英参数。   remaining_budget：还剩多少个空位可以选
        remaining_budget = self.n_frequency[adapter] - self.curr_index[adapter]
        real_up = min(up, remaining_budget)
        real_low = min(low, remaining_budget)
        if self.coefficient_indices[adapter].device != x_mean.device:
            self.coefficient_indices[adapter] = self.coefficient_indices[adapter].to(x_mean.device)
        start_idx = self.curr_index[adapter]
        #把所有参与考核的参数的 (Row, Col) 坐标 拿出来了，准备根据考核结果进行搬运。
        candidate_coords = self.coefficient_indices[adapter][:, start_idx:]
        #1.多了
        if num_candidates > real_up:
            # 按梯度大小降序排列，拿到排名索引
            indices = torch.argsort(candidates_data, descending=True)
            selected_coords = candidate_coords[:, indices[:real_up]]  #只取前 real up个
            start = self.curr_index[adapter]
            end = start + real_up
            #这里是“原地覆盖”，把新的精英参数集中到了分界线的左侧
            self.coefficient_indices[adapter][:, start:end] = selected_coords
            self.curr_index[adapter] += real_up
        #2.少了
        elif num_candidates < real_low:
            indices = torch.argsort(candidates_data, descending=True)
            if candidates_data.numel() < real_low:
                actual_take = candidates_data.numel()
            else:
                actual_take = real_low

            selected_coords = candidate_coords[:, indices[:actual_take]]
            start = self.curr_index[adapter]
            end = start + actual_take
            self.coefficient_indices[adapter][:, start:end] = selected_coords
            self.curr_index[adapter] += actual_take
        #3.适中
        else:
            # 1. 直接筛选：不用sort，直接用nonzero找出 Mask 为 True 的索引。  这里的 selected_indices 就是所有及格考生的编号。
            selected_indices = torch.nonzero(candidates_mask, as_tuple=True)[0]
            indices_num = num_candidates
            self.coefficient_indices[adapter][:, start_idx: start_idx + indices_num] = \
                candidate_coords[:, selected_indices]
            self.curr_index[adapter] += indices_num

    def clear_cache(self, adapter):
        #在 check_reinitiate 的最后一步，筛选完毕后，及时清理
        # 循环弹出缓存中的每一个张量
        while self.grad_cache[adapter]:
            grad = self.grad_cache[adapter].pop()
            if torch.is_tensor(grad):
                grad.detach_()
                grad = None
        self.grad_cache[adapter].clear()

    def update_distribution(self, adapter, x_mean):
        # 被 check_reinitiate 调用。也就是当 grad_cache 攒满了 grad_store_steps的梯度后
        if x_mean.numel() == 0:  # x_mean 是候选参数的这几步梯度的平均值
            return
        candidates_data = x_mean.reshape(-1)
        # 内循环估算器更新
        search_estimator = self.search_quantile_estimator[adapter]
        # Warm Start (热启动)
        # 刚开始训练时，估算器内部的阈值默认为 0.0。直接算出当前这一批数据的真实分位数，强制赋值给它、直接从正确的位置开始微调。
        if search_estimator.quantile.item() == 0.0 and candidates_data.max() > 0:
            init_val = torch.quantile(candidates_data, search_estimator.q)
            search_estimator.quantile.data.fill_(init_val)

        #外循环估算器更新，逻辑完全一样：如果还是 0 就热启动，然后用全部数据更新一次。
        reinit_estimator = self.reinit_quantile_estimator[adapter]
        if reinit_estimator.quantile.item() == 0.0 and candidates_data.max() > 0:
            init_val_reinit = torch.quantile(candidates_data, reinit_estimator.q)
            reinit_estimator.quantile.data.fill_(init_val_reinit)

        # 分批 SGD 更新
        batch_size = 1000
        total_samples = candidates_data.shape[0]
        perm = torch.randperm(total_samples, device=candidates_data.device)
        candidates_data = candidates_data[perm]
        #est_val = self.search_quantile_estimator[adapter].get_quantile().item()
        # 2. 计算真实值 (Ground Truth)
        # 注意：ratio 是 Top-K，所以分位数是 1 - ratio
        #target_q = self.search_quantile_estimator[adapter].q
        #true_val = torch.quantile(candidates_data, target_q).item()
        # 3. 打印偏差
        #diff = abs(est_val - true_val)
        #rel_err = diff / (true_val + 1e-9) * 100
        #print(f"\n[🔍 Estimator Check] Adapter: {adapter}")
        #print(f"  > SGD Est:  {est_val:.8f}")
        #print(f"  > True Val: {true_val:.8f}")
        #print(f"  > Error:    {rel_err:.2f}%")

        for i in range(0, total_samples, batch_size):
            batch = candidates_data[i: i + batch_size]
            search_estimator.update(batch)
            reinit_estimator.update(batch)

    def check_reinitiate(self, adapter):
        #每次前向都检查是否到达grad store steps，若达到，开始下列一系列操作
        if len(self.grad_cache[adapter]) >= self.grad_store_steps[adapter]:
            # 1.计算每个候选参数的平均梯度
            x_mean = torch.stack(self.grad_cache[adapter]).mean(dim=0).abs()
            # 2.根据x mean，更新两个分位数估计器
            self.update_distribution(adapter, x_mean)
            # 3.根据更新的阈值，选择晋升的参数
            self.select_location(adapter, x_mean)
            # 4.没晋升的参数被覆盖，作为新的候选参数
            self.coefficient_indices[adapter] = fill_new_coords(self.out_features, self.in_features,
                                                                self.coefficient_indices[adapter],
                                                                self.curr_index[adapter])
            self.clear_cache(adapter)

    def genGradientHook(self, adapter):
        #梯度传给优化器之前，强行把那些“候选参数”的梯度置为 0
        def mask_gradient(grad):
            k = self.curr_index[adapter]   # k (curr_index)分界线： 左边是精英参数，右边是试用期参数
            if k < grad.shape[0]:
                grad[k:] = 0.0
            return grad
        # 返回这个函数给 register_hook 使用
        return mask_gradient
    #把稀疏存储的 coefficient（值）和 indices（坐标）还原成一个完整的稠密矩阵 ΔW
    def get_delta_weight(self, adapter):
        #调用者：extrenal_reset
        updates = self.coefficient[adapter]
        locations = self.coefficient_indices[adapter].to(updates.device)
        return torch.sparse_coo_tensor(locations, updates, torch.Size([self.out_features, self.in_features])).to_dense()

    #阶段：稳定训练期（isFull 为 True 时）。
    def external_reset(self, adapter):
        # 1. 检查数据积攒是否足够
        if len(self.mean_Info[adapter]) < self.grad_store_steps[adapter]:
            return False
        current_mean_val = torch.tensor(self.mean_Info[adapter]).mean().item()
        decay_ratio = self.select_pars[adapter][3]
        if self.fixed_threshold.get(adapter) is None:
            self.fixed_threshold[adapter] = current_mean_val
            print(f"[🔒 BASELINE INIT] Adapter: {adapter} | Init: {current_mean_val:.8f}")
            self.mean_Info[adapter] = []
            return False
        baseline = self.fixed_threshold[adapter]
        #  如果当前梯度反超 Baseline，说明之前的 Baseline 定低了，或者是噪声导致的波谷
        # 我们承认这个新高点，以此为新基准。这样可以防止 Baseline 被锁死在一个低位。
        if current_mean_val > baseline:
            # 可以选择直接更新，或者平滑更新（例如：0.9*旧 + 0.1*新）
            # 这里是直接更新
            self.fixed_threshold[adapter] = current_mean_val
            self.mean_Info[adapter] = []
            return False
        # -------------------------------------------------------------
        # C. 正常的衰减检查
        # 此时 baseline 已经是历史最高点，threshold 是它的 decay_ratio 倍
        threshold_val = baseline * decay_ratio
        triggered = False
        # 打印调试
        print(f"[DEBUG] {adapter}: Curr {current_mean_val:.6f} vs Thr {threshold_val:.6f} (Base: {baseline:.6f})")
        if current_mean_val < threshold_val:
            print(f"\n[🔄 RESET TRIGGERED] Adapter: {adapter}")
            print(f"  > Reason: Activity dropped to {decay_ratio * 100}% of Peak Baseline")
            print(f"  > Peak: {baseline:.8f} -> Current: {current_mean_val:.8f}")
            # 1. 执行融合
            self.merge_(adapter)
            # 2. 清零参数
            nn.init.zeros_(self.coefficient[adapter])
            self.coefficient_indices[adapter].zero_()
            self.search_quantile_estimator[adapter].reset()
            self.reinit_quantile_estimator[adapter].reset()
            self.grad_cache[adapter] = []
            self.curr_index[adapter] = 0
            # 3. 重置后，Baseline 也要清空，等待下一轮重新寻找高峰
            self.fixed_threshold[adapter] = None
            triggered = True
        # 无论是否触发，都要清空统计缓存
        self.mean_Info[adapter] = []
        return triggered

    def merge_(self, adapter):
        updates = self.coefficient[adapter]
        locations = self.coefficient_indices[adapter].to(updates.device)
        # 构建稀疏增量矩阵，根据l位置填充a
        sparse_delta_w = torch.sparse_coo_tensor(
            locations,
            updates,
            torch.Size([self.out_features, self.in_features])
        )
        #原地相加 (In-place Add)  W_base = W_base + 1.0 * ΔW
        self.get_base_layer().weight.data.add_(
            sparse_delta_w,
            alpha=1.0
        )

class DSSLinear(nn.Module, DSSLayer):
    #把模型中指定的线性层（如 q_proj, v_proj）替换成这个。 原来的线性层并没有被扔掉，而是变成了它的 self.base_layer。
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            n_frequency: int = 1000,
            fan_in_fan_out: bool = False,
            dropout: float = 0.0,
            grad_store_steps=3,
            low=1,
            up=20,
            ratio=0.05,
            reinitiate_ratio=0.05,
            ratioAdd=False,
            quantile_lr=0.01,
            quantile_alpha=0.0,
            **kwargs,
    ) -> None:
        super().__init__()
        # 调用DSSLayer 的初始化
        DSSLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        #调用各种初始化
        self.update_layer(adapter_name, n_frequency, dropout, grad_store_steps, low, up, ratio, reinitiate_ratio,
                          ratioAdd, quantile_lr, quantile_alpha)
    #用于在“训练模式”和“推理（评估）模式”之间切换
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.merged:
                self.unmerge()
        else:
            if not self.merged:
                self.merge()

    #临时性融合（加速推理）——在推理阶段直接把adapter加到模型层中
    #模型就变成了一个普通的稠密线性层，直接矩阵乘法即可，速度最快
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        # 如果没指定 adapter_names，默认合并所有激活的 adapter。
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.coefficient.keys():
                # 1. 安全合并   慢，占显存，但安全（检查 NaN）
                if safe_merge:
                    base_layer = self.get_base_layer()
                    updates = self.coefficient[active_adapter]
                    # 确保坐标和数值在同一个设备上
                    locations = self.coefficient_indices[active_adapter].to(updates.device)
                    # 利用坐标 (locations) 和数值 (updates) 还原出和权重矩阵一样大的稀疏张量
                    sparse_delta = torch.sparse_coo_tensor(
                        locations,
                        updates,
                        base_layer.weight.shape
                    )
                    orig_weights = base_layer.weight.data.clone()
                    # 执行加法：W_clone = W_clone + ΔW
                    orig_weights.add_(sparse_delta)
                    # 检查是否溢出或出现非数字 (NaN)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f"NaNs detected in {active_adapter}")
                    # 将检查通过的权重赋回给基座
                    base_layer.weight.data = orig_weights
                # 2. 快速合并   不占额外显存，In-place 操作
                else:
                    # 它执行的是：base_layer.weight.data.add_(sparse_delta)
                    self.merge_(active_adapter)
                # 将该 Adapter 标记为“已合并”，防止重复合并导致权重加了两次
                if active_adapter not in self.merged_adapters:
                    self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.coefficient.keys():
                base_layer = self.get_base_layer()
                updates = self.coefficient[active_adapter]
                locations = self.coefficient_indices[active_adapter].to(updates.device)
                sparse_delta = torch.sparse_coo_tensor(
                    locations,
                    updates,
                    base_layer.weight.shape
                )
                # alpha=-1.0 表示加负的更新量，即减法。
                base_layer.weight.data.add_(sparse_delta, alpha=-1.0)

    #forward 充当了状态机的角色。每次数据流过，它都会检查当前处于什么阶段，并触发相应的操作
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        '''
            1.筛选期 (Search Phase)：
        状态：参数还没招满 (not isFull)。
        动作：调用 check_reinitiate。
        目的：检查 grad_cache 里的梯度攒够了吗？够了就筛选一批新人晋升。
            2.稳定期 (Stable Phase)：
        状态：参数招满了 (isFull)。
        动作：调用 check_merge_clear。
        目的：检查这批精英是不是学不动了（收敛了）？如果是，就触发 Merge & Reset。
            3.总结：DSS 的“动态性”不是由外部循环控制的，而是寄生在每一次前向传播里自动发生的。'''
        previous_dtype = x.dtype  #记录下“原来的格式/精度”
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            #拿到基座模型权重的引用。这个权重通常是冻结的
            base_weight = self.base_layer.weight
            combined_weight = base_weight
            weight_modified = False
            for active_adapter in self.active_adapters:
                if active_adapter not in self.coefficient.keys():
                    continue
                updates = self.coefficient[active_adapter]
                locations = self.coefficient_indices[active_adapter].to(updates.device)
                #判断当前是处于 搜索 还是 训练
                isFull = (self.curr_index[active_adapter] == self.n_frequency[active_adapter])
                if self.training:
                    # 这里的 Dropout 是对这 18w 个值随机置零一部分。
                    updates = self.dropout_layer[active_adapter](updates)
                    # 条件：not isFull 没搜索完 且 train_location_mode 为真 (允许调整位置)
                    if (not isFull) and (self.train_location_mode[active_adapter]):
                        # 调用 check_reinitiate ->检查 grad_cache 里的梯度攒够了没 -> 如果攒够了，就调用 select_location 从候选中挑选精英晋升。
                        self.check_reinitiate(active_adapter)
                    # 条件：isFull 搜索阶段结束 且 允许调整位置
                    if (isFull) and (self.train_location_mode[active_adapter]):
                        #调用 check_merge_clear -> 计算当前参数的梯度均值 (mean_Info)  -> 判断是否触发 Merge & Reset (重置)。
                        #self.check_merge_clear(active_adapter)
                        pass
                #模型当前正处于 训练阶段 ，且在 构建反向传播计算图 时
                if updates.requires_grad and torch.is_grad_enabled():
                    #清理旧钩子
                    if self.hook_handle.get(active_adapter) is not None:
                        self.hook_handle[active_adapter].remove()
                    # 注册新钩子 当计算出 updates 的梯度时，  先把梯度传给 self.genGradientHook(...) 处理一下，再存起来。
                    self.hook_handle[active_adapter] = updates.register_hook(self.genGradientHook(active_adapter))
                #最后一步——构建计算图与前向计算
                updates_cast = updates.to(dtype=base_weight.dtype)
                # 构建出稀疏矩阵 delta_w。
                #当反向传播经过 delta_w 时，PyTorch 会自动调用 DSSMakeDeltaWeight.backward。
                #backward 会负责把 delta_w 的梯度“偷”出来存进 grad_cache。
                delta_w = self.dss_make_weight(
                    updates_cast,
                    locations,
                    self.out_features,
                    self.in_features,
                    self,
                    self.curr_index[active_adapter],
                    isFull,
                    active_adapter
                )
                # 2. 权重融合
                if not weight_modified:
                    # 第一次循环：创建新张量  base_weight.data +  delta_w
                    combined_weight = base_weight.data + delta_w
                    weight_modified = True
                else:
                    # 后续循环（如果有多个 Adapter）：原地累加
                    combined_weight.add_(delta_w)
            # 3. 线性运算  数学公式：Y = X * W_combined^T + b
            x = x.to(combined_weight.dtype)
            result = F.linear(x, combined_weight, bias=self.base_layer.bias)
        # 4. 还原数据类型
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "dss." + rep