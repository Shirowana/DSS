from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


def _default_basis_group_map() -> dict[str, list[str]]:
    return {
        "q_proj": ["q_proj"],
        "k_proj": ["k_proj"],
        "v_proj": ["v_proj"],
        "up_proj": ["up_proj"],
        "down_proj": ["down_proj"],
    }


@dataclass
class DSSConfig(PeftConfig):
    shared_basis_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a torch-serialized shared basis pack."},
    )
    basis_group_map: dict[str, list[str]] = field(
        default_factory=_default_basis_group_map,
        metadata={"help": "Mapping from basis group names to target-module suffix aliases."},
    )
    group_scale_init: float = field(
        default=1.0,
        metadata={"help": "Initial positive scale c_g used in DeltaW = c_g * A_inv * DeltaLambda * B_inv."},
    )
    n_frequency: int = field(
        default=8,
        metadata={"help": "Total elite-slot budget per adapted layer. This is the final size of the trainable elite pool after stage-1 fill completes."},
    )
    candidate_size: int = field(
        default=32,
        metadata={"help": "Candidate-pool budget per adapted layer. Stage-1 first samples this many candidate coordinates, then candidate probing and gradient-statistics accumulation happen on the same set together."},
    )
    grad_store_steps: int = field(
        default=10,
        metadata={"help": "Accumulate candidate gradients for this many optimizer steps, then compute x_mean, update the threshold estimator, and move above-threshold candidates into the elite pool."},
    )
    init_candidate_mode: str = field(
        default="random_unique",
        metadata={"help": "Candidate initialization mode. currently supports only `random_unique`."},
    )
    low: Union[int, float] = field(
        default=1,
        metadata={"help": "Minimum number of candidate promotions during one stage-1 refresh."},
    )
    up: Union[int, float] = field(
        default=4,
        metadata={"help": "Maximum number of candidate promotions during one stage-1 refresh."},
    )
    ratio: float = field(
        default=0.1,
        metadata={"help": "Stage-1 quantile ratio used by the pinball-loss threshold estimator."},
    )
    score_metric: str = field(
        default="core_grad_abs",
        metadata={"help": "Candidate scoring metric. v0 currently supports only core-space gradient magnitude."},
    )
    stage2_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable stage-2 prune/grow hooks. False means fixing elite slots after fill; True means entering periodic stage-2 updates."},
    )
    steady_stage_ratio: float = field(
        default=0.0,
        metadata={"help": "Stable-training ratio applied after stage-1 fill completes. Stage-2 stays inactive for ceil(steady_stage_ratio * total_steps) optimizer steps before prune/grow starts."},
    )
    update_interval: int = field(
        default=100,
        metadata={"help": "Shared stage-2 update interval for prune/grow placeholders."},
    )
    update_counts: int = field(
        default=1,
        metadata={"help": "Initial replacement budget used by stage-2 linear decay scheduling."},
    )
    update_margin: float = field(
        default=0.0,
        metadata={"help": "Unused stage-2 prune-vs-grow comparison margin placeholder for future gate logic."},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if the target layer stores weights as (fan_in, fan_out)."},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for DSS. Can be `none`, `all`, or `dss_only`."},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Target module names or regex pattern for DSS injection."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Modules to exclude from DSS injection."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Additional trainable modules to save alongside DSS adapters."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "Optional layer indices to transform."},
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Layer pattern used together with `layers_to_transform`."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.peft_type = PeftType.DSS
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a regex string.")
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a regex string.")
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")

        if self.n_frequency <= 0:
            raise ValueError("`n_frequency` must be a positive integer.")
        if self.candidate_size < 0:
            raise ValueError("`candidate_size` must be non-negative.")
        if self.grad_store_steps <= 0:
            raise ValueError("`grad_store_steps` must be a positive integer.")
        if not isinstance(self.low, (int, float)) or self.low < 0:
            raise ValueError("`low` must be a non-negative int or float.")
        if not isinstance(self.up, (int, float)) or self.up <= 0:
            raise ValueError("`up` must be a positive int or float.")
        if not 0.0 < self.ratio < 1.0:
            raise ValueError("`ratio` must be in the open interval (0, 1).")
        if not 0.0 <= self.steady_stage_ratio <= 1.0:
            raise ValueError("`steady_stage_ratio` must be in the closed interval [0, 1].")
        if self.update_interval <= 0:
            raise ValueError("`update_interval` must be a positive integer.")
        if self.update_counts < 0:
            raise ValueError("`update_counts` must be non-negative.")
        if self.init_candidate_mode != "random_unique":
            raise ValueError("v0 only supports `init_candidate_mode='random_unique'`.")
        if self.score_metric != "core_grad_abs":
            raise ValueError("v0 only supports `score_metric='core_grad_abs'`.")

        if not isinstance(self.basis_group_map, dict) or not self.basis_group_map:
            raise ValueError("`basis_group_map` must be a non-empty dictionary.")
        if self.group_scale_init <= 0:
            raise ValueError("`group_scale_init` must be strictly positive.")
        normalized_group_map: dict[str, list[str]] = {}
        for group_name, aliases in self.basis_group_map.items():
            if not isinstance(group_name, str) or not group_name:
                raise ValueError("Every `basis_group_map` key must be a non-empty string.")
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = list(aliases)
            if not aliases or not all(isinstance(alias, str) and alias for alias in aliases):
                raise ValueError(f"`basis_group_map[{group_name!r}]` must contain one or more non-empty strings.")
            normalized_group_map[group_name] = aliases
        self.basis_group_map = normalized_group_map
