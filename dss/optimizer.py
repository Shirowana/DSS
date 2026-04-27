from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch

try:
    from accelerate.optimizer import AcceleratedOptimizer
except ImportError:  # pragma: no cover
    AcceleratedOptimizer = None


def unwrap_optimizer(optimizer):
    if AcceleratedOptimizer is not None and isinstance(optimizer, AcceleratedOptimizer):
        return optimizer.optimizer
    return optimizer
    '''如果外面用了 accelerate，optimizer 可能被包成 AcceleratedOptimizer。
    这个函数把它拆回真实 optimizer'''

    #检查当前 optimizer 必须是 DSSAdamW
def require_dss_optimizer(optimizer):
    optimizer = unwrap_optimizer(optimizer)
    if not isinstance(optimizer, DSSAdamW):
        raise TypeError(
            "DSS stage-wise optimizer state management requires `DSSAdamW`. "
            "Create the optimizer with `create_dss_optimizer(...)`."
        )
    return optimizer

    #检查某个参数是否有完整的state
def ensure_optimizer_state(optimizer, param: torch.nn.Parameter) -> dict[str, torch.Tensor]:
    optimizer = require_dss_optimizer(optimizer)
    state = optimizer.state[param]
    if len(state) == 0:
        dtype = optimizer.momentum_dtype
        state["age"] = torch.ones_like(param, dtype=dtype)
        state["exp_avg"] = torch.zeros_like(param, dtype=dtype)
        state["exp_avg_sq"] = torch.zeros_like(param, dtype=dtype)
    return state

    #stage1  简单初始化
def init_slot_state(optimizer, param: torch.nn.Parameter, slot_indices: torch.Tensor) -> None:
    if optimizer is None or slot_indices.numel() == 0:
        return
    state = ensure_optimizer_state(optimizer, param)
    slot_count = slot_indices.numel()
    rewrite_slot_state(
        optimizer,
        param,
        slot_indices,
        age=torch.ones(slot_count, device=slot_indices.device, dtype=torch.float32),
        exp_avg=torch.zeros(slot_count, device=slot_indices.device, dtype=torch.float32),
        exp_avg_sq=torch.zeros(slot_count, device=slot_indices.device, dtype=torch.float32),
    )

    #对一批指定的 slot，局部重写 optimizer state
def rewrite_slot_state(
    optimizer,
    param: torch.nn.Parameter,
    slot_indices: torch.Tensor,
    *,
    age: Optional[torch.Tensor] = None,
    exp_avg: Optional[torch.Tensor] = None,
    exp_avg_sq: Optional[torch.Tensor] = None,
) -> None:
    if optimizer is None or slot_indices.numel() == 0:
        return

    state = ensure_optimizer_state(optimizer, param)
    updates = {"age": age, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    for state_name, init_value in updates.items():
        state_tensor = state[state_name]
        if init_value is None:
            state_tensor[slot_indices] = 0.0
            continue
        state_tensor[slot_indices] = init_value.to(device=state_tensor.device, dtype=state_tensor.dtype)

    #计算β，用于热启动估计m，v
def get_optimizer_betas(optimizer, param: torch.nn.Parameter) -> tuple[float, float] | None:
    optimizer = unwrap_optimizer(optimizer)
    if optimizer is None:
        return None

    for group in optimizer.param_groups:
        if not any(group_param is param for group_param in group.get("params", ())):
            continue
        betas = group.get("betas")
        if betas is None:
            return None
        return float(betas[0]), float(betas[1])
    return None


class DSSAdamW(torch.optim.Optimizer):
    """AdamW variant with per-slot age and elementwise bias correction."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        momentum_dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.momentum_dtype = momentum_dtype or torch.float32

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            correct_bias = group["correct_bias"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("DSSAdamW does not support sparse gradients.")

                state = ensure_optimizer_state(self, param)
                age = state["age"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = grad.to(dtype=self.momentum_dtype)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                if correct_bias:
                    age_fp32 = age.to(dtype=torch.float32)
                    bias1 = 1.0 - beta1**age_fp32
                    bias2 = 1.0 - beta2**age_fp32
                    denom.mul_(bias1.to(dtype=denom.dtype))
                    denom.div_(torch.sqrt(bias2).to(dtype=denom.dtype))

                param.addcdiv_(exp_avg, denom, value=-lr)
                if weight_decay > 0.0:
                    param.add_(param, alpha=-lr * weight_decay)

                age.add_(1)

        return loss


def create_dss_optimizer(
    model,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    correct_bias: bool = True,
    momentum_dtype: torch.dtype = torch.float32,
):
    """Create the default outer optimizer for trainable DSS coefficients only."""
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = DSSAdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        correct_bias=correct_bias,
        momentum_dtype=momentum_dtype,
    )
    return optimizer
