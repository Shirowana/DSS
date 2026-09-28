import unittest

import torch
from torch import nn

from training_methods.sift_adapter import SIFTAdapter


class TinySIFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)

    def forward(self, inputs):
        return self.down_proj(self.q_proj(inputs))


class SIFTAdapterTests(unittest.TestCase):
    def test_official_sparse_logic_and_dense_export(self):
        torch.manual_seed(42)
        model = TinySIFTModel()
        adapter = SIFTAdapter(
            model,
            sparse_rate=0.1,
            sparse_modules=["q_proj", "down_proj"],
            grad_acc=1,
        )
        optimizer = torch.optim.AdamW(adapter.parameters_in_optimizer(), lr=1e-3)

        for _ in range(3):
            loss = model(torch.randn(2, 8)).pow(2).mean()
            loss.backward()
            adapter.sync_gradients()
            adapter.clear_dense_gradients()
            optimizer.step()
            adapter.apply_delta()
            optimizer.zero_grad()

        self.assertTrue(all(adapter.if_get_idx.values()))
        self.assertEqual(adapter.get_trainable_num(), 14)
        self.assertIsNone(model.other.weight.grad)

        adapter.prepare_dense_export()
        self.assertFalse(any(name.endswith("_sparse") for name, _ in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
