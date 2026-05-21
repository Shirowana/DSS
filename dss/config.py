from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class DSSConfig(PeftConfig):
    n_frequency: int = field(
        default=8,
        metadata={"help": "Final elite-slot budget per adapted layer."},
    )
    candidate_size: int = field(
        default=32,
        metadata={"help": "How many candidate coordinates to probe in each stage1 refresh round."},
    )
    grad_store_steps: int = field(
        default=10,
        metadata={"help": "Accumulate candidate gradients for this many optimizer steps before refreshing stage1."},
    )
    low: Union[int, float] = field(
        default=1,
        metadata={"help": "Minimum promotions per stage1 refresh. Floats are interpreted as a ratio of n_frequency."},
    )
    up: Union[int, float] = field(
        default=4,
        metadata={"help": "Maximum promotions per stage1 refresh. Floats are interpreted as a ratio of n_frequency."},
    )
    ratio: float = field(
        default=0.1,
        metadata={"help": "Target top-k ratio used by the stage1 threshold estimator."},
    )
    threshold_mode: str = field(
        default="oracle",
        metadata={"help": "Threshold estimator mode. Supported values: `oracle`, `sgd`."},
    )
    score_method: str = field(
        default="abs_mean",
        metadata={
            "help": (
                "Candidate importance score used before thresholding. Supported values: "
                "`mean_abs`, `abs_mean`, `mean_square`, `rms_over_param`, "
                "`abs_mean_over_param`, `snr`, `newton_like`."
            )
        },
    )
    score_eps: float = field(
        default=1e-8,
        metadata={"help": "Numerical stability constant used by score methods with a denominator."},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "Training-time dropout applied to elite DSS coefficients."},
    )
    quantile_lr: float = field(
        default=0.01,
        metadata={"help": "Learning rate for the pinball-loss quantile estimator when threshold_mode='sgd'."},
    )
    quantile_alpha: float = field(
        default=0.0,
        metadata={"help": "Reserved alpha parameter for the online quantile estimator."},
    )
    threshold_log_every_steps: int = field(
        default=100,
        metadata={"help": "Print DSS threshold debug info on first refresh and then every N optimizer steps per layer."},
    )
    init_enabled: bool = field(
        default=False,
        metadata={"help": "Enable the initial gradient-observation window before normal stage1 refresh begins."},
    )
    init_steps: int = field(
        default=10,
        metadata={"help": "How many early optimizer steps to use for the initialization observation window."},
    )
    init_candidate_ratio: float = field(
        default=0.05,
        metadata={"help": "Per-matrix candidate ratio used during initialization block sampling."},
    )
    init_seed_mode: str = field(
        default="threshold_only",
        metadata={"help": "Initialization behavior. Supported values: `threshold_only`, `seed_elite`."},
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
        if self.threshold_mode not in {"oracle", "sgd"}:
            raise ValueError("`threshold_mode` must be either `oracle` or `sgd`.")
        if self.score_method not in {
            "mean_abs",
            "abs_mean",
            "mean_square",
            "rms_over_param",
            "abs_mean_over_param",
            "snr",
            "newton_like",
        }:
            raise ValueError(
                "`score_method` must be one of: "
                "`mean_abs`, `abs_mean`, `mean_square`, `rms_over_param`, "
                "`abs_mean_over_param`, `snr`, `newton_like`."
            )
        if self.score_eps <= 0.0:
            raise ValueError("`score_eps` must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("`dropout` must be in the interval [0, 1).")
        if self.quantile_lr <= 0.0:
            raise ValueError("`quantile_lr` must be positive.")
        if self.threshold_log_every_steps <= 0:
            raise ValueError("`threshold_log_every_steps` must be a positive integer.")
        if self.init_steps <= 0:
            raise ValueError("`init_steps` must be a positive integer.")
        if not 0.0 < self.init_candidate_ratio <= 1.0:
            raise ValueError("`init_candidate_ratio` must be in the interval (0, 1].")
        if self.init_seed_mode not in {"threshold_only", "seed_elite"}:
            raise ValueError("`init_seed_mode` must be `threshold_only` or `seed_elite`.")
