from __future__ import annotations

import importlib.util
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

        def _move_adapter_to_device_of_base_layer(self, adapter_name, device=None):
            return None

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

    sys.modules["peft"] = peft_mod
    sys.modules["peft.utils"] = peft_utils
    sys.modules["peft.config"] = peft_config
    sys.modules["peft.tuners"] = peft_tuners
    peft_tuners_dss.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["peft.tuners.dss"] = peft_tuners_dss
    sys.modules["peft.tuners._buffer_dict"] = peft_buffer_dict
    sys.modules["peft.tuners.tuners_utils"] = peft_tuners_utils

if "transformers.pytorch_utils" not in sys.modules:
    transformers_mod = types.ModuleType("transformers")
    pytorch_utils_mod = types.ModuleType("transformers.pytorch_utils")

    class Conv1D(torch.nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(nx, nf))

    pytorch_utils_mod.Conv1D = Conv1D
    sys.modules["transformers"] = transformers_mod
    sys.modules["transformers.pytorch_utils"] = pytorch_utils_mod

LAYER_PATH = Path(__file__).with_name("layer.py")
spec = importlib.util.spec_from_file_location("peft.tuners.dss.layer", LAYER_PATH)
layer_module = importlib.util.module_from_spec(spec)
sys.modules["peft.tuners.dss.layer"] = layer_module
assert spec.loader is not None
spec.loader.exec_module(layer_module)
DSSLinear = layer_module.DSSLinear

SCORING_PATH = Path(__file__).with_name("scoring.py")
scoring_spec = importlib.util.spec_from_file_location("peft.tuners.dss.scoring", SCORING_PATH)
scoring_module = importlib.util.module_from_spec(scoring_spec)
sys.modules["peft.tuners.dss.scoring"] = scoring_module
assert scoring_spec.loader is not None
scoring_spec.loader.exec_module(scoring_module)
ScoreStats = scoring_module.ScoreStats
compute_score = scoring_module.compute_score


def dense_reference(layer: DSSLinear, adapter_name: str, x: torch.Tensor) -> torch.Tensor:
    delta = layer.get_delta_weight(adapter_name).to(layer.get_base_layer().weight.dtype)
    return torch.nn.functional.linear(x.to(delta.dtype), delta, bias=None).to(x.dtype)


class DSSLayerStageTests(unittest.TestCase):
    def build_layer(self, *, threshold_mode: str = "oracle", dropout: float = 0.0) -> DSSLinear:
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
            threshold_mode=threshold_mode,
            dropout=dropout,
            quantile_lr=0.01,
            quantile_alpha=0.0,
        )
        layer.set_adapter("default")
        return layer

    def test_forward_matches_dense_reference(self):
        layer = self.build_layer()
        layer.runtime["default"].curr_count = 3
        layer.coefficient_indices["default"][:3] = torch.tensor([0, 6, 11], dtype=torch.long)
        layer.coefficient["default"].data[:3] = torch.tensor([0.5, -1.0, 2.0])
        layer.candidate_indices["default"] = torch.empty(0, dtype=torch.long)

        x = torch.randn(2, 4)
        actual = layer(x)
        expected = dense_reference(layer, "default", x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_candidate_grads_are_collected_without_elites(self):
        layer = self.build_layer()
        layer.train()
        layer.runtime["default"].curr_count = 0
        layer.candidate_indices["default"] = torch.tensor([0, 5, 10], dtype=torch.long)

        x = torch.randn(2, 4, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()

        self.assertEqual(layer.grad_count["default"], 1)
        self.assertIsNotNone(layer.grad_cache["default"])
        self.assertEqual(layer.grad_cache["default"].numel(), 3)
        self.assertIsNone(layer.coefficient["default"].grad)

    def test_stage1_refresh_promotes_top_candidate(self):
        layer = self.build_layer()
        layer.runtime["default"].curr_count = 0
        layer.candidate_indices["default"] = torch.tensor([2, 7, 11], dtype=torch.long)
        layer.grad_cache["default"] = torch.tensor([0.2, 0.8, 0.4], dtype=torch.float32)
        layer.grad_count["default"] = 2

        promoted = layer.maybe_refresh_stage1("default")

        self.assertEqual(promoted, 1)
        self.assertEqual(layer.runtime["default"].curr_count, 1)
        self.assertEqual(int(layer.coefficient_indices["default"][0].item()), 7)
        self.assertEqual(layer.grad_count["default"], 0)

    def test_sgd_threshold_mode_refreshes_without_error(self):
        layer = self.build_layer(threshold_mode="sgd")
        layer.runtime["default"].curr_count = 0
        layer.candidate_indices["default"] = torch.tensor([1, 4, 8], dtype=torch.long)
        layer.grad_cache["default"] = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        layer.grad_count["default"] = 2

        promoted = layer.maybe_refresh_stage1("default")

        self.assertGreaterEqual(promoted, 1)
        self.assertGreater(layer.search_quantile_estimator["default"].get_quantile().item(), 0.0)

    def test_export_restore_sparse_checkpoint(self):
        source = self.build_layer()
        source.runtime["default"].curr_count = 2
        source.coefficient["default"].data[:2] = torch.tensor([1.25, -0.75], dtype=torch.float32)
        source.coefficient_indices["default"][:2] = torch.tensor([3, 10], dtype=torch.long)
        source.elite_bitset["default"][torch.tensor([3, 10], dtype=torch.long)] = True

        exported = source.export_sparse_checkpoint("default")

        restored = self.build_layer()
        restored.restore_sparse_checkpoint("default", exported["coefficient"], exported["coefficient_indices"])

        self.assertEqual(restored.runtime["default"].curr_count, 2)
        self.assertTrue(torch.equal(restored.coefficient["default"][:2].detach(), exported["coefficient"]))
        self.assertTrue(torch.equal(restored.coefficient_indices["default"][:2], exported["coefficient_indices"]))
        self.assertEqual(int(restored.elite_bitset["default"].sum().item()), 2)
        self.assertEqual(restored.grad_count["default"], 0)

    def test_default_abs_mean_matches_legacy_formula(self):
        layer = self.build_layer()
        layer.grad_cache["default"] = torch.tensor([0.2, -0.8, 0.4], dtype=torch.float32)
        layer.grad_count["default"] = 2

        actual = layer.compute_x_mean("default")
        expected = (layer.grad_cache["default"] / 2.0).abs()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))


class DSSScoringTests(unittest.TestCase):
    def setUp(self):
        self.stats = ScoreStats(
            grad_sum=torch.tensor([2.0, -4.0], dtype=torch.float32),
            abs_grad_sum=torch.tensor([4.0, 4.0], dtype=torch.float32),
            grad_sq_sum=torch.tensor([6.0, 12.0], dtype=torch.float32),
            count=2,
        )
        self.theta0_abs = torch.tensor([2.0, 4.0], dtype=torch.float32)

    def test_score_methods_match_hand_computation(self):
        mean_abs = compute_score("mean_abs", self.stats, self.theta0_abs)
        abs_mean = compute_score("abs_mean", self.stats, self.theta0_abs)
        mean_square = compute_score("mean_square", self.stats, self.theta0_abs)
        rms_over_param = compute_score("rms_over_param", self.stats, self.theta0_abs)
        abs_mean_over_param = compute_score("abs_mean_over_param", self.stats, self.theta0_abs)
        snr = compute_score("snr", self.stats, self.theta0_abs)
        newton_like = compute_score("newton_like", self.stats, self.theta0_abs)

        self.assertTrue(torch.allclose(mean_abs, torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.allclose(abs_mean, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(mean_square, torch.tensor([3.0, 6.0])))
        self.assertTrue(torch.allclose(rms_over_param, torch.tensor([0.8660254, 0.6123724]), atol=1e-6))
        self.assertTrue(torch.allclose(abs_mean_over_param, torch.tensor([0.5, 0.5])))
        self.assertTrue(torch.allclose(snr, torch.tensor([0.70710677, 1.4142135]), atol=1e-6))
        self.assertTrue(torch.allclose(newton_like, torch.tensor([0.33333334, 0.33333334]), atol=1e-6))

    def test_snr_is_stable_for_zero_variance(self):
        stats = ScoreStats(
            grad_sum=torch.tensor([2.0], dtype=torch.float32),
            abs_grad_sum=torch.tensor([2.0], dtype=torch.float32),
            grad_sq_sum=torch.tensor([2.0], dtype=torch.float32),
            count=2,
        )
        score = compute_score("snr", stats, torch.tensor([1.0]))
        self.assertTrue(torch.isfinite(score).all())
        self.assertGreater(score.item(), 0.0)

    def test_denominator_based_scores_are_stable_for_zero_theta(self):
        theta0_abs = torch.tensor([0.0, 0.0], dtype=torch.float32)
        for method in ("rms_over_param", "abs_mean_over_param", "newton_like"):
            score = compute_score(method, self.stats, theta0_abs, eps=1e-8)
            self.assertTrue(torch.isfinite(score).all(), msg=method)


if __name__ == "__main__":
    unittest.main()
