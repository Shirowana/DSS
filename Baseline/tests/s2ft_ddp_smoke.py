"""Two-GPU S2FT integration smoke; run with torch.distributed.run."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_methods.s2ft_adapter import S2FTAdapter, S2FTRatios


def assert_all_ranks_equal(tensor: torch.Tensor, label: str) -> None:
    maximum = tensor.detach().float().clone()
    minimum = tensor.detach().float().clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    difference = (maximum - minimum).abs().max().item()
    if difference != 0:
        raise AssertionError(f"{label} differs across ranks: max_abs_diff={difference}")


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.manual_seed(123)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config).to(device=local_rank, dtype=torch.bfloat16)
    adapter = S2FTAdapter(
        model,
        "llama2",
        seed=42,
        ratios=S2FTRatios(v=0.0, o=0.25, u=0.0, d=0.25),
    )
    adapter.verify_distributed_selection()

    wrapped = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in adapter.optimizer_named_parameters()], lr=1e-3
    )
    generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(1000 + rank)
    input_ids = torch.randint(0, config.vocab_size, (2, 12), generator=generator, device=local_rank)
    labels = input_ids.clone()
    loss = wrapped(input_ids=input_ids, labels=labels).loss
    loss.backward()
    optimizer.step()

    for name, parameter in model.named_parameters():
        if name.endswith(".s2"):
            assert_all_ranks_equal(parameter, name)

    adapter.densify()
    if any(name.endswith(".s2") for name, _ in model.named_parameters()):
        raise AssertionError("Dense export retained S2 parameters")
    for name, parameter in model.named_parameters():
        assert_all_ranks_equal(parameter, name)

    if rank == 0:
        print("S2FT two-GPU DDP smoke passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
