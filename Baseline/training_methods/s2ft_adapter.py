from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
S2FT_ROOT = ROOT / "S2FT"
S2_UTILS_PATH = S2FT_ROOT / "experiments" / "utils" / "s2_utils.py"


@dataclass(frozen=True)
class S2FTRatios:
    v: float
    o: float
    u: float
    d: float

    def as_dict(self) -> dict[str, float]:
        return {"v_ratio": self.v, "o_ratio": self.o, "u_ratio": self.u, "d_ratio": self.d}


OFFICIAL_RATIOS = {
    "llama2": S2FTRatios(v=0.0, o=0.052, u=0.0, d=0.02),
    "llama3": S2FTRatios(v=0.0, o=0.0, u=0.0, d=0.03),
}


def official_ratios(model_family: str) -> S2FTRatios:
    try:
        return OFFICIAL_RATIOS[model_family]
    except KeyError as error:
        raise ValueError(f"S2FT has no official ratio configuration for {model_family!r}") from error


def _load_official_utils() -> ModuleType:
    root = str(S2FT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    module_name = "baseline_official_s2ft_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, S2_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official S2FT utilities from {S2_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"S2FT {name} must be in [0, 1], found {value}")


def _empty_by_layer(layer_count: int) -> dict[int, list[int]]:
    return {index: [] for index in range(layer_count)}


def _sample_by_layer(
    rng: random.Random,
    layer_count: int,
    width: int,
    ratio: float,
) -> tuple[dict[int, list[int]], list[int]]:
    count = int(layer_count * width * ratio)
    selected = sorted(rng.sample(range(layer_count * width), count))
    by_layer = _empty_by_layer(layer_count)
    for flat_index in selected:
        by_layer[flat_index // width].append(flat_index % width)
    return by_layer, selected


def _selection_hash(selection: dict[str, list[int]]) -> str:
    payload = json.dumps(selection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class S2FTAdapter:
    """Thin adapter around the authors' S2FT layers and conversion utilities."""

    def __init__(
        self,
        model: nn.Module,
        model_family: str,
        seed: int,
        ratios: S2FTRatios | None = None,
    ) -> None:
        self.model = model
        self.model_family = model_family
        self.seed = seed
        self.ratios = ratios or official_ratios(model_family)
        for name, value in self.ratios.as_dict().items():
            _validate_ratio(name, value)

        self._utils = _load_official_utils()
        self.selected_parameters, flat_selection = self._build_selection()
        self.selection_hash = _selection_hash(flat_selection)
        self._configure_model()
        self.manifest: dict[str, Any] = {
            "implementation": "official_S2FT_layers_and_conversion_utils",
            "selection_strategy": "S2FT-R_fixed_random",
            "model_family": model_family,
            "seed": seed,
            "ratios": self.ratios.as_dict(),
            "selected_counts": {name: len(values) for name, values in flat_selection.items()},
            "selected_flat_indices": flat_selection,
            "selection_sha256": self.selection_hash,
            "distributed_selection_rule": "identical_seed_then_all_rank_hash_verification",
            "export": "fuse_s2_then_replace_with_torch_nn_Linear",
        }

    def _build_selection(self):
        config = self.model.config
        layer_count = config.num_hidden_layers
        rng = random.Random(self.seed)

        v_by_layer, v_flat = _sample_by_layer(
            rng, layer_count, config.num_attention_heads, self.ratios.v
        )
        o_by_layer, o_flat = _sample_by_layer(
            rng, layer_count, config.num_attention_heads, self.ratios.o
        )
        u_by_layer, u_flat = _sample_by_layer(
            rng, layer_count, config.intermediate_size, self.ratios.u
        )
        d_by_layer, d_flat = _sample_by_layer(
            rng, layer_count, config.intermediate_size, self.ratios.d
        )
        selected_parameters = {
            "mha": {"v_proj": v_by_layer, "o_proj": o_by_layer},
            "ffn": {"up_proj": u_by_layer, "down_proj": d_by_layer},
        }
        flat_selection = {
            "v_proj": v_flat,
            "o_proj": o_flat,
            "up_proj": u_flat,
            "down_proj": d_flat,
        }
        return selected_parameters, flat_selection

    def _configure_model(self) -> None:
        if self.ratios.v > 0 or self.ratios.o > 0:
            if self.model.config.num_key_value_heads != self.model.config.num_attention_heads:
                raise ValueError(
                    "Official S2FT MHA conversion requires equal attention and key/value head "
                    "counts; the official Llama3 configuration must keep v_ratio=o_ratio=0"
                )
            self._utils.convert_mha_layer_to_s2(self.model, self.selected_parameters["mha"])
        if self.ratios.u > 0 or self.ratios.d > 0:
            self._utils.convert_ffn_layer_to_s2(self.model, self.selected_parameters["ffn"])
        self._utils.only_optimize_s2_parameters(self.model)

    def optimizer_named_parameters(self):
        return [(name, value) for name, value in self.model.named_parameters() if value.requires_grad]

    def verify_distributed_selection(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, self.selection_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"S2FT selection differs across distributed ranks: {hashes}")

    def save_candidate(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: parameter.detach().cpu()
            for name, parameter in self.model.named_parameters()
            if name.endswith(".s2")
        }
        torch.save(state, path)

    def load_candidate(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        named_parameters = dict(self.model.named_parameters())
        if set(state) != {name for name in named_parameters if name.endswith(".s2")}:
            raise ValueError("Saved S2FT state does not match the configured S2 parameters")
        with torch.no_grad():
            for name, value in state.items():
                named_parameters[name].copy_(value.to(named_parameters[name]))

    def densify(self) -> int:
        from s2ft import S2ColumnLinear, S2RowLinear

        replaced = 0

        def replace_children(parent: nn.Module) -> None:
            nonlocal replaced
            for name, child in list(parent.named_children()):
                if isinstance(child, (S2ColumnLinear, S2RowLinear)):
                    child.fuse_s2_weight()
                    dense = nn.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        dense.weight.copy_(child.weight)
                        if child.bias is not None:
                            dense.bias.copy_(child.bias)
                    dense.train(child.training)
                    setattr(parent, name, dense)
                    replaced += 1
                else:
                    replace_children(child)

        replace_children(self.model)
        if replaced == 0:
            raise RuntimeError("No S2FT modules were found during dense export")
        return replaced
