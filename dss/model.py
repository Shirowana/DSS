from __future__ import annotations

import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional

import torch
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_DSS_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .config import DSSConfig
from .layer import DSSLayer, DSSLinear
from .shared_basis import SharedBasisPack


class DSSModel(BaseTuner):
    prefix: str = "dss_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False, state_dict=None) -> None:
        primary_config = config[adapter_name] if isinstance(config, dict) else config
        self.shared_basis_pack = SharedBasisPack.load(primary_config.shared_basis_path)
        super().__init__(
            model,
            config,
            adapter_name,
            low_cpu_mem_usage=low_cpu_mem_usage,
            state_dict=state_dict,
        )

    def _check_new_adapter_config(self, config: DSSConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(dss_config, key):
        return check_target_module_exists(dss_config, key)

    def _resolve_basis_group(self, dss_config: DSSConfig, current_key: str) -> str:
        for group_name, aliases in dss_config.basis_group_map.items():
            for alias in aliases:
                if current_key == alias or current_key.endswith(f".{alias}"):
                    return group_name
        raise ValueError(
            f"Could not resolve a shared-basis group for module {current_key!r}. "
            f"Available groups: {sorted(dss_config.basis_group_map.keys())}."
        )

    def _create_and_replace(
        self,
        dss_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current key should not be None.")

        basis_group_name = self._resolve_basis_group(dss_config, current_key)
        shared_basis = self.shared_basis_pack.get(basis_group_name)
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "n_frequency": dss_config.n_frequency,
            "candidate_size": dss_config.candidate_size,
            "grad_store_steps": dss_config.grad_store_steps,
            "low": dss_config.low,
            "up": dss_config.up,
            "ratio": dss_config.ratio,
            "stage2_enabled": dss_config.stage2_enabled,
            "steady_stage_ratio": dss_config.steady_stage_ratio,
            "update_interval": dss_config.update_interval,
            "update_counts": dss_config.update_counts,
            "update_margin": dss_config.update_margin,
            "basis_group_name": basis_group_name,
            "shared_basis": shared_basis,
            "module_name": current_key,
            "fan_in_fan_out": dss_config.fan_in_fan_out,
            "bias": bias,
        }
        if isinstance(target, DSSLayer):
            target.update_layer(
                adapter_name=adapter_name,
                n_frequency=dss_config.n_frequency,
                candidate_size=dss_config.candidate_size,
                grad_store_steps=dss_config.grad_store_steps,
                low=dss_config.low,
                up=dss_config.up,
                ratio=dss_config.ratio,
                stage2_enabled=dss_config.stage2_enabled,
                steady_stage_ratio=dss_config.steady_stage_ratio,
                update_interval=dss_config.update_interval,
                update_counts=dss_config.update_counts,
                update_margin=dss_config.update_margin,
                basis_group_name=basis_group_name,
                shared_basis=shared_basis,
                module_name=current_key,
            )
        else:
            new_module = self._create_new_module(dss_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        if not any(parameter.device == meta for parameter in new_module.parameters()):
            new_module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for module in model.modules():
            if not isinstance(module, DSSLayer):
                continue
            for active_adapter in module.active_adapters:
                if active_adapter in module.coefficient:
                    module.coefficient[active_adapter].requires_grad_(True)

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for name, parameter in model.named_parameters():
                    if "bias" in name:
                        parameter.requires_grad = True
            elif bias == "dss_only":
                for module in model.modules():
                    if isinstance(module, DSSLayer) and getattr(module, "bias", None) is not None:
                        module.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias {bias!r} is not implemented.")

    @staticmethod
    def _create_new_module(dss_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = dss_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = dss_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently only `torch.nn.Linear` and `Conv1D` are supported."
            )

        return DSSLinear(target, adapter_name, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        for active_adapter in self.active_adapters:
            if self.peft_config[active_adapter].bias != "none":
                warnings.warn(
                    "Disabling DSS adapters with trainable bias does not restore the exact base-model output."
                )
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        self.set_auxiliary_adapters(adapter_name)
        for module in self.model.modules():
            if isinstance(module, DSSLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def active_dss_layers(self):
        for module_name, module in self.model.named_modules():
            if not isinstance(module, DSSLayer):
                continue
            for adapter_name in module.active_adapters:
                if adapter_name in module.coefficient:
                    yield module_name, module, adapter_name

    def refresh_dss_layout(
        self,
        global_step: int,
        total_steps: int,
        optimizer=None,
        grad_accumulation_steps: int = 1,
    ) -> dict[str, int]:
        summary = {
            "stage1_refreshed_layers": 0,
            "stage2_updated_layers": 0,
            "promoted_slots": 0,
            "pruned_slots": 0,
            "grown_slots": 0,
        }

        for _module_name, layer, adapter_name in self.active_dss_layers():
            report = layer.check_reinitiate(
                adapter_name,
                total_steps=total_steps,
                global_step=global_step,
            )
            if report.refreshed:
                summary["stage1_refreshed_layers"] += 1
                summary["promoted_slots"] += int(report.promoted_slots)

        for _module_name, layer, adapter_name in self.active_dss_layers():
            report = layer.run_stage2_update(
                adapter_name,
                total_steps,
                optimizer=optimizer,
                grad_accumulation_steps=grad_accumulation_steps,
            )
            if report.updated:
                summary["stage2_updated_layers"] += 1
                summary["pruned_slots"] += int(report.pruned_slots)
                summary["grown_slots"] += int(report.grown_slots)

        for _module_name, layer, adapter_name in self.active_dss_layers():
            layer.advance_update_state(adapter_name)

        return summary

        '''用法：
        loss.backward()
        optimizer.step()
        layout_summary = model.refresh_dss_layout(
            global_step=global_step,
            total_steps=total_train_steps,
            optimizer=optimizer,
            grad_accumulation_steps=grad_accumulation_steps,
        )
        optimizer.zero_grad()
        '''
        
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_DSS_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`.")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_DSS_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        return self.model

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> torch.nn.Module:
        return self._unload_and_optionally_merge(
            progressbar=progressbar,
            safe_merge=safe_merge,
            adapter_names=adapter_names,
        )

    def unload(self) -> torch.nn.Module:
        return self._unload_and_optionally_merge(merge=False)
