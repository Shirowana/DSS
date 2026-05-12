from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ROOT = Path("/root/hf_cache_models/models")
DATA_ROOT = Path("/root/datasets/commonsense_new")

MODEL_MAP = {
    "Llama2-7B": str(MODEL_ROOT / "Llama-2-7b-hf"),
    "Llama3-8B": str(MODEL_ROOT / "Meta-Llama-3-8B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe base Llama layers without PEFT.")
    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_cache_dir", type=str, default=str(MODEL_ROOT))
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=2)
    return parser.parse_args()


def resolve_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_train_dataset(dataset_obj) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if isinstance(dataset_obj, DatasetDict):
        if "train" not in dataset_obj:
            raise ValueError("DatasetDict must contain a `train` split.")
        return dataset_obj["train"]
    if hasattr(dataset_obj, "keys") and "train" in dataset_obj:
        return dataset_obj["train"]
    raise ValueError(f"Unsupported dataset object loaded from disk: {type(dataset_obj)!r}")


def collate_batch(batch, pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask = [torch.as_tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    labels = [torch.as_tensor(item["labels"], dtype=torch.long) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def summarize_tensor(tensor: torch.Tensor) -> str:
    view = tensor.detach().float()
    zero_count = int((view == 0).sum().item())
    return (
        f"shape={tuple(view.shape)} "
        f"dtype={tensor.dtype} "
        f"abs_mean={float(view.abs().mean().item()):.8f} "
        f"min={float(view.min().item()):.8f} "
        f"max={float(view.max().item()):.8f} "
        f"zero_count={zero_count}/{view.numel()}"
    )


def register_probe_hooks(model):
    hook_specs = {
        "model.layers.10.post_attention_layernorm_input": ("pre", model.model.layers[10].post_attention_layernorm),
        "model.layers.10.post_attention_layernorm_output": ("post", model.model.layers[10].post_attention_layernorm),
        "model.layers.10.self_attn.o_proj_output": ("post", model.model.layers[10].self_attn.o_proj),
        "model.layers.10.mlp.up_proj_input": ("pre", model.model.layers[10].mlp.up_proj),
        "model.layers.10.mlp.up_proj_output": ("post", model.model.layers[10].mlp.up_proj),
        "model.layers.10.mlp.down_proj_input": ("pre", model.model.layers[10].mlp.down_proj),
        "model.layers.10.mlp.down_proj_output": ("post", model.model.layers[10].mlp.down_proj),
        "model.layers.11.post_attention_layernorm_input": ("pre", model.model.layers[11].post_attention_layernorm),
        "model.layers.11.post_attention_layernorm_output": ("post", model.model.layers[11].post_attention_layernorm),
        "model.layers.11.self_attn.o_proj_output": ("post", model.model.layers[11].self_attn.o_proj),
        "model.layers.11.mlp.up_proj_input": ("pre", model.model.layers[11].mlp.up_proj),
        "model.layers.11.mlp.up_proj_output": ("post", model.model.layers[11].mlp.up_proj),
        "model.layers.11.mlp.down_proj_input": ("pre", model.model.layers[11].mlp.down_proj),
        "model.layers.11.mlp.down_proj_output": ("post", model.model.layers[11].mlp.down_proj),
        "model.layers.12.post_attention_layernorm_input": ("pre", model.model.layers[12].post_attention_layernorm),
        "model.layers.12.post_attention_layernorm_output": ("post", model.model.layers[12].post_attention_layernorm),
        "model.layers.12.self_attn.o_proj_output": ("post", model.model.layers[12].self_attn.o_proj),
        "model.layers.12.mlp.up_proj_input": ("pre", model.model.layers[12].mlp.up_proj),
        "model.layers.12.mlp.up_proj_output": ("post", model.model.layers[12].mlp.up_proj),
        "model.layers.12.mlp.down_proj_input": ("pre", model.model.layers[12].mlp.down_proj),
        "model.layers.12.mlp.down_proj_output": ("post", model.model.layers[12].mlp.down_proj),
    }
    stats: dict[str, str] = {}
    hooks = []

    def make_pre_hook(name: str):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                stats[name] = summarize_tensor(inputs[0])

        return hook

    def make_post_hook(name: str):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                stats[name] = summarize_tensor(output)

        return hook

    for name, (hook_type, module) in hook_specs.items():
        if hook_type == "pre":
            hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
        else:
            hooks.append(module.register_forward_hook(make_post_hook(name)))
    return hooks, stats


def print_norm_weight_stats(model) -> None:
    for layer_idx in (10, 11, 12):
        weight = model.model.layers[layer_idx].post_attention_layernorm.weight
        print(
            f"[BASE weight] module=model.layers.{layer_idx}.post_attention_layernorm.weight "
            f"requires_grad={weight.requires_grad} {summarize_tensor(weight)}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_precision(args.precision)
    load_name = args.model_path or MODEL_MAP[args.model_name]
    dataset_path = args.dataset_path or str(DATA_ROOT / f"train_all_256_OnlyOutput_{args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=args.model_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=args.model_cache_dir, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    dataset = resolve_train_dataset(load_from_disk(dataset_path))
    take = min(max(args.batch_size, args.max_samples), len(dataset))
    batch = [dataset[i] for i in range(take)]
    tensors = collate_batch(batch, tokenizer.pad_token_id)
    batch_inputs = {
        "input_ids": tensors["input_ids"].to(device),
        "attention_mask": tensors["attention_mask"].to(device),
        "labels": tensors["labels"].to(device),
    }

    print(
        f"[BASE probe] model={args.model_name} precision={args.precision} "
        f"dataset_path={dataset_path} batch_size={take}",
        flush=True,
    )
    print_norm_weight_stats(model)

    hooks, stats = register_probe_hooks(model)
    with torch.no_grad():
        _ = model(**batch_inputs)
    for hook in hooks:
        hook.remove()

    for name in sorted(stats):
        print(f"[BASE probe] module={name} {stats[name]}", flush=True)


if __name__ == "__main__":
    main()
