from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "peft" not in sys.modules:
    peft_mod = types.ModuleType("peft")
    peft_utils = types.ModuleType("peft.utils")
    peft_config = types.ModuleType("peft.config")
    peft_tuners = types.ModuleType("peft.tuners")
    peft_tuners_dss = types.ModuleType("peft.tuners.dss")
    peft_buffer_dict = types.ModuleType("peft.tuners._buffer_dict")
    peft_tuners_utils = types.ModuleType("peft.tuners.tuners_utils")

    class PeftConfig:
        def __post_init__(self):
            return None

    class PeftType:
        DSS = "DSS"

    class BufferDict(dict):
        pass

    class BaseTunerLayer:
        def __init__(self, *args, **kwargs):
            self._active_adapter = []
            self._disable_adapters = False
            self.merged_adapters = []

        @property
        def active_adapters(self):
            if isinstance(self._active_adapter, str):
                return [self._active_adapter]
            return list(self._active_adapter)

        def set_adapter(self, adapter_name):
            if isinstance(adapter_name, str):
                self._active_adapter = [adapter_name]
            else:
                self._active_adapter = list(adapter_name)

        def enable_adapters(self, enabled=True):
            self._disable_adapters = not enabled

        @property
        def disable_adapters(self):
            return self._disable_adapters

        @property
        def merged(self):
            return len(self.merged_adapters) > 0

        def get_base_layer(self):
            return self.base_layer

        def _cast_input_dtype(self, x, dtype):
            return x.to(dtype=dtype) if x.dtype != dtype else x

    class BaseTuner(torch.nn.Module):
        def __init__(self, model=None, config=None, adapter_name=None, **kwargs):
            super().__init__()
            self.model = model
            self.peft_config = config if isinstance(config, dict) else ({adapter_name: config} if config is not None else {})
            self.active_adapter = adapter_name
            self.active_adapters = [adapter_name] if adapter_name is not None else []

        def set_auxiliary_adapters(self, adapter_name):
            if isinstance(adapter_name, str):
                self.active_adapters = [adapter_name]
            else:
                self.active_adapters = list(adapter_name)

    class ModulesToSaveWrapper(torch.nn.Module):
        pass

    def register_peft_method(**kwargs):
        return None

    def check_target_module_exists(config, key):
        return True

    def check_adapters_to_merge(module, adapter_names):
        return adapter_names or module.active_adapters

    def _get_submodules(model, key):
        raise AttributeError

    def create_dss_optimizer(*args, **kwargs):
        raise NotImplementedError("test stub")

    peft_utils.register_peft_method = register_peft_method
    peft_utils.PeftType = PeftType
    peft_utils.TRANSFORMERS_MODELS_TO_DSS_TARGET_MODULES_MAPPING = {}
    peft_utils.ModulesToSaveWrapper = ModulesToSaveWrapper
    peft_utils._get_submodules = _get_submodules
    peft_config.PeftConfig = PeftConfig
    peft_buffer_dict.BufferDict = BufferDict
    peft_tuners_utils.BaseTunerLayer = BaseTunerLayer
    peft_tuners_utils.BaseTuner = BaseTuner
    peft_tuners_utils.check_target_module_exists = check_target_module_exists
    peft_tuners_utils.check_adapters_to_merge = check_adapters_to_merge
    peft_tuners_dss.create_dss_optimizer = create_dss_optimizer

    sys.modules["peft"] = peft_mod
    sys.modules["peft.utils"] = peft_utils
    sys.modules["peft.config"] = peft_config
    sys.modules["peft.tuners"] = peft_tuners
    sys.modules["peft.tuners.dss"] = peft_tuners_dss
    sys.modules["peft.tuners._buffer_dict"] = peft_buffer_dict
    sys.modules["peft.tuners.tuners_utils"] = peft_tuners_utils

if "transformers.pytorch_utils" not in sys.modules:
    transformers_mod = types.ModuleType("transformers")
    trainer_utils_mod = types.ModuleType("transformers.trainer_utils")
    pytorch_utils_mod = types.ModuleType("transformers.pytorch_utils")

    class Trainer:
        pass

    class TrainOutput:
        def __init__(self, global_step, training_loss, metrics):
            self.global_step = global_step
            self.training_loss = training_loss
            self.metrics = metrics

    class Conv1D(torch.nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(nx, nf))

    transformers_mod.Trainer = Trainer
    trainer_utils_mod.TrainOutput = TrainOutput
    pytorch_utils_mod.Conv1D = Conv1D
    sys.modules["transformers"] = transformers_mod
    sys.modules["transformers.trainer_utils"] = trainer_utils_mod
    sys.modules["transformers.pytorch_utils"] = pytorch_utils_mod

from new.dss.layer import DSSLinear
from new.dss.shared_basis import SharedBasisEntry, SharedBasisPack, normalize_inverse_basis_to_identity_fro


def make_identity_basis(out_features: int, in_features: int) -> SharedBasisEntry:
    return SharedBasisEntry(
        group_name="test",
        A=torch.eye(out_features),
        B=torch.eye(in_features),
        A_inv=torch.eye(out_features),
        B_inv=torch.eye(in_features),
        shape=(out_features, in_features),
        offset=0,
    )


class SharedBasisNormalizationTests(unittest.TestCase):
    def test_identity_fro_normalization_matches_identity_norms(self):
        A_inv = torch.diag(torch.tensor([10.0, 2.0, 0.5]))
        B_inv = torch.diag(torch.tensor([4.0, 3.0]))

        C, D, c, d = normalize_inverse_basis_to_identity_fro(A_inv, B_inv, shape=(3, 2))

        self.assertAlmostEqual(float(C.norm().item()), 3**0.5, places=6)
        self.assertAlmostEqual(float(D.norm().item()), 2**0.5, places=6)
        self.assertAlmostEqual(c, (3**0.5) / float(A_inv.norm().item()), places=6)
        self.assertAlmostEqual(d, (2**0.5) / float(B_inv.norm().item()), places=6)

        delta_lambda = torch.randn(3, 2)
        delta_w = C @ delta_lambda @ D
        self.assertEqual(tuple(delta_w.shape), (3, 2))

    def test_old_basis_payload_without_metadata_still_loads(self):
        raw = {
            "entries": {
                "test": {
                    "A": torch.eye(2),
                    "B": torch.eye(3),
                    "A_inv": torch.eye(2),
                    "B_inv": torch.eye(3),
                    "shape": (2, 3),
                    "offset": 0,
                }
            }
        }

        entry = SharedBasisPack._coerce_entry("test", raw["entries"]["test"])

        self.assertEqual(entry.inverse_normalization, None)
        self.assertEqual(entry.A_inv_scale, None)
        self.assertEqual(entry.B_inv_scale, None)
        self.assertTrue(torch.equal(entry.A_inv, torch.eye(2)))
        self.assertTrue(torch.equal(entry.B_inv, torch.eye(3)))


def dense_core_from_slots(
    coefficient: torch.Tensor,
    coefficient_indices: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    flat = coefficient.new_zeros(out_features * in_features)
    if coefficient.numel() > 0:
        flat = flat.scatter_add(0, coefficient_indices.long(), coefficient)
    return flat.view(out_features, in_features)


def slot_grads_reference(
    input_basis: torch.Tensor,
    grad_core_out: torch.Tensor,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    if flat_indices.numel() == 0:
        return torch.empty(0, device=input_basis.device, dtype=input_basis.dtype)
    input_2d = input_basis.reshape(-1, input_basis.size(-1))
    grad_2d = grad_core_out.reshape(-1, grad_core_out.size(-1))
    rows = torch.div(flat_indices.long(), input_basis.size(-1), rounding_mode="floor")
    cols = flat_indices.long().remainder(input_basis.size(-1))
    return (grad_2d.index_select(1, rows) * input_2d.to(grad_2d.dtype).index_select(1, cols)).sum(0)


class DSSLayerStageTests(unittest.TestCase):
    def build_layer(self, *, stage2_enabled: bool) -> DSSLinear:
        base = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.zeros_(base.weight)
        layer = DSSLinear(
            base,
            adapter_name="default",
            n_frequency=4,
            candidate_size=3,
            grad_store_steps=2,
            low=1,
            up=1,
            ratio=0.5,
            stage2_enabled=stage2_enabled,
            steady_stage_ratio=0.0,
            update_interval=2,
            update_counts=1,
            update_margin=0.0,
            basis_group_name="test",
            shared_basis=make_identity_basis(4, 4),
        )
        layer.set_adapter("default")
        return layer

    def test_forward_matches_dense_core_reference(self):
        layer = self.build_layer(stage2_enabled=False)
        runtime = layer.runtime["default"]
        runtime.curr_count = 3
        layer.coefficient_indices["default"][:3] = torch.tensor([0, 6, 11], dtype=torch.long)
        layer.coefficient["default"].data[:3] = torch.tensor([0.5, -1.0, 2.0])
        layer.candidate_indices["default"] = torch.empty(0, dtype=torch.long)

        x = torch.randn(2, 4)
        out = layer(x)
        lambda_dense = dense_core_from_slots(
            layer.coefficient["default"][:3],
            layer.coefficient_indices["default"][:3],
            4,
            4,
        )
        expected = torch.nn.functional.linear(x, lambda_dense)

        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))

    def test_elite_coefficient_grad_matches_dense_reference(self):
        layer = self.build_layer(stage2_enabled=False)
        runtime = layer.runtime["default"]
        runtime.curr_count = 3
        layer.coefficient_indices["default"][:3] = torch.tensor([0, 6, 11], dtype=torch.long)
        layer.coefficient["default"].data[:3] = torch.tensor([0.5, -1.0, 2.0])
        layer.candidate_indices["default"] = torch.empty(0, dtype=torch.long)

        x = torch.randn(2, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        expected = slot_grads_reference(
            x.detach(),
            torch.ones_like(out.detach()),
            layer.coefficient_indices["default"][:3],
        )
        self.assertTrue(torch.allclose(layer.coefficient["default"].grad[:3], expected.float(), atol=1e-5, rtol=1e-5))

    def test_stage1_collects_candidate_probe_grads(self):
        layer = self.build_layer(stage2_enabled=False)
        layer.candidate_indices["default"] = torch.tensor([0, 6], dtype=torch.long)
        layer.grad_cache["default"] = torch.zeros(2, dtype=torch.float32)
        layer.grad_count["default"] = 0

        x = torch.randn(2, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        self.assertEqual(layer.grad_count["default"], 1)
        expected = slot_grads_reference(x.detach(), torch.ones_like(out.detach()), layer.candidate_indices["default"])
        self.assertTrue(torch.allclose(layer.grad_cache["default"], expected.float(), atol=1e-5, rtol=1e-5))

    def test_check_reinitiate_promotes_and_resamples(self):
        layer = self.build_layer(stage2_enabled=False)
        layer.candidate_indices["default"] = torch.tensor([3, 7], dtype=torch.long)
        layer.grad_cache["default"] = torch.tensor([8.0, 2.0], dtype=torch.float32)
        layer.grad_count["default"] = 2
        layer.search_quantile_estimator["default"].quantile.data.zero_()

        report = layer.check_reinitiate("default", total_steps=10, global_step=4)

        self.assertTrue(report.refreshed)
        self.assertEqual(report.promoted_slots, 1)
        self.assertEqual(layer.runtime["default"].curr_count, 1)
        self.assertEqual(int(layer.coefficient_indices["default"][0].item()), 3)
        self.assertEqual(int(layer.coefficient["default"][0].item()), 0)
        self.assertTrue(bool(layer.elite_bitset["default"][3].item()))

    def test_stage2_collects_running_stats(self):
        layer = self.build_layer(stage2_enabled=True)
        runtime = layer.runtime["default"]
        runtime.phase = "stage2"
        runtime.curr_count = 1
        runtime.update_flag = True
        layer.coefficient_indices["default"][0] = 5
        layer.coefficient["default"].data[0] = 0.3
        layer.elite_bitset["default"][5] = True
        layer.candidate_indices["default"] = torch.tensor([0, 6], dtype=torch.long)
        layer.candidate_grad_sums["default"] = torch.zeros(2)
        layer.candidate_grad_sq_sums["default"] = torch.zeros(2)

        x = torch.randn(2, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        expected = slot_grads_reference(x.detach(), torch.ones_like(out.detach()), layer.candidate_indices["default"])
        self.assertTrue(torch.allclose(layer.candidate_grad_sums["default"], expected.float(), atol=1e-5, rtol=1e-5))

    def test_stage2_update_prunes_and_grows(self):
        layer = self.build_layer(stage2_enabled=True)
        runtime = layer.runtime["default"]
        runtime.phase = "stage2"
        runtime.curr_count = 2
        runtime.current_step = 2
        runtime.steady_phase = 0
        runtime.stage2_start_step = 4
        layer.coefficient_indices["default"][:2] = torch.tensor([0, 1], dtype=torch.long)
        layer.coefficient["default"].data[:2] = torch.tensor([0.1, 1.0])
        layer.elite_bitset["default"][0] = True
        layer.elite_bitset["default"][1] = True
        layer.candidate_indices["default"] = torch.tensor([2, 3], dtype=torch.long)
        layer.candidate_grad_sums["default"] = torch.tensor([4.0, 1.0])
        layer.candidate_grad_sq_sums["default"] = torch.tensor([16.0, 1.0])

        report = layer.run_stage2_update("default", total_steps=10, optimizer=None, grad_accumulation_steps=1)

        self.assertIsNotNone(report)
        self.assertTrue(report.updated)
        self.assertEqual(report.pruned_slots, 1)
        self.assertEqual(report.grown_slots, 1)
        self.assertEqual(int(layer.coefficient_indices["default"][0].item()), 2)
        self.assertEqual(float(layer.coefficient["default"][0].item()), 0.0)
        self.assertFalse(bool(layer.elite_bitset["default"][0].item()))
        self.assertTrue(bool(layer.elite_bitset["default"][2].item()))

    def test_merge_and_unmerge_restore_base_weight(self):
        layer = self.build_layer(stage2_enabled=False)
        runtime = layer.runtime["default"]
        runtime.curr_count = 2
        layer.coefficient_indices["default"][:2] = torch.tensor([0, 6], dtype=torch.long)
        layer.coefficient["default"].data[:2] = torch.tensor([0.25, -0.75])

        base_weight = layer.get_base_layer().weight.detach().clone()
        delta_weight = layer.get_delta_weight("default").to(base_weight.dtype)

        layer.merge()
        self.assertTrue(torch.allclose(layer.get_base_layer().weight, base_weight + delta_weight))

        layer.unmerge()
        self.assertTrue(torch.allclose(layer.get_base_layer().weight, base_weight))

    def test_sparse_checkpoint_export_is_compact_prefix(self):
        layer = self.build_layer(stage2_enabled=True)
        runtime = layer.runtime["default"]
        runtime.phase = "stage2"
        runtime.curr_count = 2
        runtime.current_step = 1
        runtime.update_rounds = 3
        layer.coefficient["default"].data[:2] = torch.tensor([0.25, -0.5])
        layer.coefficient_indices["default"][:2] = torch.tensor([1, 7], dtype=torch.long)

        exported = layer.export_sparse_checkpoint("default")

        self.assertEqual(tuple(exported["coefficient"].shape), (2,))
        self.assertEqual(tuple(exported["coefficient_indices"].shape), (2,))
        self.assertTrue(torch.equal(exported["coefficient"], torch.tensor([0.25, -0.5])))
        self.assertTrue(torch.equal(exported["coefficient_indices"], torch.tensor([1, 7], dtype=torch.long)))

    def test_sparse_checkpoint_restore_rebuilds_active_state(self):
        source = self.build_layer(stage2_enabled=True)
        source.runtime["default"].curr_count = 2
        source.coefficient["default"].data[:2] = torch.tensor([0.25, -0.5])
        source.coefficient_indices["default"][:2] = torch.tensor([1, 7], dtype=torch.long)
        exported = source.export_sparse_checkpoint("default")

        restored = self.build_layer(stage2_enabled=True)
        restored.runtime["default"].phase = "stage2"
        restored.runtime["default"].current_step = 2
        restored.runtime["default"].update_rounds = 4
        restored.candidate_indices["default"] = torch.tensor([2, 3], dtype=torch.long)
        restored.grad_cache["default"] = torch.ones(2)
        restored.grad_count["default"] = 2
        restored.last_promoted_slot_positions["default"] = torch.tensor([0], dtype=torch.long)
        restored.last_promoted_flat_indices["default"] = torch.tensor([1], dtype=torch.long)

        restored.restore_sparse_checkpoint("default", exported["coefficient"], exported["coefficient_indices"])

        self.assertEqual(restored.runtime["default"].curr_count, 2)
        self.assertEqual(restored.runtime["default"].phase, "stage1")
        self.assertEqual(restored.runtime["default"].current_step, 0)
        self.assertEqual(restored.runtime["default"].update_rounds, 0)
        self.assertTrue(torch.equal(restored.coefficient_indices["default"][:2], torch.tensor([1, 7], dtype=torch.long)))
        self.assertTrue(torch.allclose(restored.coefficient["default"][:2], torch.tensor([0.25, -0.5])))
        self.assertTrue(bool(restored.elite_bitset["default"][1].item()))
        self.assertTrue(bool(restored.elite_bitset["default"][7].item()))
        self.assertEqual(restored.candidate_indices["default"].numel(), 0)
        self.assertIsNone(restored.grad_cache["default"])
        self.assertEqual(restored.grad_count["default"], 0)
        self.assertNotIn("default", restored.last_promoted_slot_positions)
        self.assertNotIn("default", restored.last_promoted_flat_indices)


if __name__ == "__main__":
    unittest.main()
