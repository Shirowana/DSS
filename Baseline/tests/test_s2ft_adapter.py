import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from training_methods.s2ft_adapter import S2FTAdapter, S2FTRatios, official_ratios


class ToyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, inputs):
        return self.o_proj(self.v_proj(inputs))


class ToyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, inputs):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = ToyAttention(hidden_size)
        self.mlp = ToyMLP(hidden_size, intermediate_size)

    def forward(self, inputs):
        return self.self_attn(inputs) + self.mlp(inputs)


class ToyLlama(nn.Module):
    def __init__(self, layers=2, hidden_size=8, intermediate_size=10, heads=4):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=heads,
            num_key_value_heads=heads,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [ToyLayer(hidden_size, intermediate_size) for _ in range(layers)]
        )

    def forward(self, inputs):
        for layer in self.model.layers:
            inputs = layer(inputs)
        return inputs


class S2FTAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = torch.randn(2, 3, 8)
        self.ratios = S2FTRatios(v=0.0, o=0.25, u=0.0, d=0.2)

    def test_official_model_ratios(self):
        self.assertEqual(official_ratios("llama2"), S2FTRatios(0.0, 0.052, 0.0, 0.02))
        self.assertEqual(official_ratios("llama3"), S2FTRatios(0.0, 0.0, 0.0, 0.03))

    def test_zero_delta_conversion_preserves_output_and_selects_only_s2(self):
        model = ToyLlama()
        expected = model(self.inputs)
        adapter = S2FTAdapter(model, "llama2", seed=42, ratios=self.ratios)
        actual = model(self.inputs)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.endswith(".s2") for name in trainable))
        self.assertEqual(adapter.manifest["selected_counts"]["o_proj"], 2)
        self.assertEqual(adapter.manifest["selected_counts"]["down_proj"], 4)

    def test_dense_export_preserves_nonzero_s2_output(self):
        model = ToyLlama()
        adapter = S2FTAdapter(model, "llama2", seed=42, ratios=self.ratios)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(".s2"):
                    parameter.normal_(mean=0.0, std=0.05)
        expected = model(self.inputs)
        s2_module_count = sum(1 for name, _ in model.named_parameters() if name.endswith(".s2"))

        replaced = adapter.densify()
        actual = model(self.inputs)

        self.assertEqual(replaced, s2_module_count)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertFalse(any(name.endswith(".s2") for name, _ in model.named_parameters()))
        self.assertTrue(
            all(type(layer.self_attn.o_proj) is nn.Linear for layer in model.model.layers)
        )
        self.assertTrue(all(type(layer.mlp.down_proj) is nn.Linear for layer in model.model.layers))

    def test_candidate_round_trip(self):
        model = ToyLlama()
        adapter = S2FTAdapter(model, "llama2", seed=42, ratios=self.ratios)
        original = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.endswith(".s2")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            adapter.save_candidate(path)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith(".s2"):
                        parameter.add_(1)
            adapter.load_candidate(path)
        for name, parameter in model.named_parameters():
            if name.endswith(".s2"):
                torch.testing.assert_close(parameter, original[name])

    def test_selection_is_seed_deterministic(self):
        first = S2FTAdapter(ToyLlama(), "llama2", seed=42, ratios=self.ratios)
        second = S2FTAdapter(ToyLlama(), "llama2", seed=42, ratios=self.ratios)
        self.assertEqual(first.selection_hash, second.selection_hash)
        self.assertEqual(first.selected_parameters, second.selected_parameters)

    def test_dense_hugging_face_export_reloads_without_s2ft(self):
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        model = LlamaForCausalLM(config).eval()
        adapter = S2FTAdapter(model, "llama2", seed=42, ratios=self.ratios)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(".s2"):
                    parameter.normal_(mean=0.0, std=0.05)
        adapter.densify()
        input_ids = torch.tensor([[1, 4, 7, 2]])
        with torch.no_grad():
            expected = model(input_ids=input_ids).logits

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            reloaded = LlamaForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                actual = reloaded(input_ids=input_ids).logits

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(any(name.endswith(".s2") for name, _ in reloaded.named_parameters()))


if __name__ == "__main__":
    unittest.main()
