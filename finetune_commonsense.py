"""
Fine-tune Llama with DSS on the preprocessed commonsense dataset.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from packaging.version import Version
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

try:
    from peft import get_peft_model
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires `peft` with DSS registration support.") from exc

from peft.tuners.dss import DSSConfig  # noqa: F401 - importing peft.tuners.dss registers the PEFT method


REMOTE_PROJECT_ROOT = Path("/root/code/DSS")
REMOTE_DATA_ROOT = Path("/root/datasets")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")
REMOTE_OUTPUT_ROOT = REMOTE_PROJECT_ROOT / "output"

MODEL_MAP = {
    "Llama2-7B": str(REMOTE_MODEL_ROOT / "Llama-2-7b-hf"),
    "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
}

MODULE_MAP = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "u": "up_proj",
    "d": "down_proj",
    "o": "o_proj",
    "g": "gate_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune commonsense with DSS.")

    parser.add_argument("--model_name", type=str, default="Llama3-8B", choices=sorted(MODEL_MAP))
    parser.add_argument("--model_path", type=str, default=None, help="Explicit local model directory; overrides --model_name.")
    parser.add_argument("--model_cache_dir", type=str, default=str(REMOTE_MODEL_ROOT))

    parser.add_argument("--data_dir", type=str, default=str(REMOTE_DATA_ROOT / "commonsense_new"))
    parser.add_argument("--dataset_path", type=str, default=None, help="Full HuggingFace disk dataset path; overrides --data_dir.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_set_size", type=int, default=0)

    parser.add_argument("--target_modules", type=str, default="qkvud")
    parser.add_argument("--n_frequency", type=int, default=8)
    parser.add_argument("--candidate_size", type=int, default=32)
    parser.add_argument("--grad_store_steps", type=int, default=10)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--up", type=float, default=4.0)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--threshold_mode", type=str, default="oracle", choices=["oracle", "sgd"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--quantile_lr", type=float, default=0.01)
    parser.add_argument("--quantile_alpha", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def build_run_name(args: argparse.Namespace, timestamp: str) -> str:
    return (
        f"commonsense_{args.model_name}_dss_"
        f"nf{args.n_frequency}_cand{args.candidate_size}_gs{args.grad_store_steps}_{timestamp}"
    )


def resolve_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def count_parameters(model) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


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


def torch_supports_safe_checkpoint_resume() -> bool:
    version = torch.__version__.split("+", 1)[0]
    return Version(version) >= Version("2.6.0")


@contextmanager
def maybe_hide_unsafe_resume_state(checkpoint_dir: str | None):
    if not checkpoint_dir or torch_supports_safe_checkpoint_resume():
        yield
        return

    checkpoint_path = Path(checkpoint_dir)
    hidden_paths: list[tuple[Path, Path]] = []
    for filename in ("optimizer.pt", "scheduler.pt"):
        original = checkpoint_path / filename
        if not original.exists():
            continue
        hidden = checkpoint_path / f"{filename}.unsafe-hidden"
        if hidden.exists():
            hidden.unlink()
        original.rename(hidden)
        hidden_paths.append((original, hidden))

    try:
        yield
    finally:
        for original, hidden in hidden_paths:
            if hidden.exists():
                hidden.rename(original)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and world_size == 1:
        raise ValueError(
            "Multiple visible GPUs detected without a distributed launcher. "
            "Use torchrun/DDP instead of single-process replicated training."
        )

    if args.val_set_size < 0:
        raise ValueError("`--val_set_size` must be non-negative.")
    if args.load_best_model_at_end and args.val_set_size <= 0:
        raise ValueError("`--load_best_model_at_end` requires `--val_set_size > 0`.")
    if args.val_set_size > 0 and args.eval_steps <= 0:
        raise ValueError("`--eval_steps` must be positive when using a validation split.")

    model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR")
    torch_dtype = resolve_precision(args.precision)
    load_name = args.model_path or MODEL_MAP[args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=torch_dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None and len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = [MODULE_MAP[key] for key in args.target_modules if key in MODULE_MAP]
    if not target_modules:
        raise ValueError("`--target_modules` did not resolve to any known module names.")

    dss_config = DSSConfig(
        target_modules=target_modules,
        n_frequency=args.n_frequency,
        candidate_size=args.candidate_size,
        grad_store_steps=args.grad_store_steps,
        low=args.low,
        up=args.up,
        ratio=args.ratio,
        threshold_mode=args.threshold_mode,
        dropout=args.dropout,
        quantile_lr=args.quantile_lr,
        quantile_alpha=args.quantile_alpha,
        bias="none",
    )
    model = get_peft_model(model, dss_config)

    if torch.cuda.is_available():
        if local_rank >= 0:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.train()

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = Path(args.data_dir) / f"train_all_{args.max_length}_OnlyOutput_{args.model_name}"
    full_train_dataset = resolve_train_dataset(load_from_disk(str(dataset_path)))

    if args.val_set_size > 0:
        if args.val_set_size >= len(full_train_dataset):
            raise ValueError(
                f"`--val_set_size` ({args.val_set_size}) must be smaller than the dataset size ({len(full_train_dataset)})."
            )
        split = full_train_dataset.train_test_split(test_size=args.val_set_size, shuffle=True, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = full_train_dataset
        eval_dataset = None

    def collate_fn(batch):
        input_ids = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_mask = [torch.as_tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        labels = [torch.as_tensor(item["labels"], dtype=torch.long) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or build_run_name(args, timestamp)
    output_dir = Path(args.output_dir) if args.output_dir else REMOTE_OUTPUT_ROOT / run_name
    args.run_name = run_name
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    has_eval = eval_dataset is not None
    eval_strategy = "steps" if has_eval else "no"
    save_strategy = "steps" if (args.save_steps > 0 or has_eval) else "no"
    resolved_eval_steps = max(args.eval_steps, 1) if has_eval else None
    resolved_save_steps = max(args.save_steps, 1) if args.save_steps > 0 else (resolved_eval_steps or 1)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=resolved_eval_steps,
        save_strategy=save_strategy,
        save_steps=resolved_save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=[] if args.report_to.lower() in {"none", "no", "false", "0"} else [args.report_to],
        run_name=run_name,
        disable_tqdm=False,
        ddp_find_unused_parameters=True if world_size > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    if args.resume_from_checkpoint and not torch_supports_safe_checkpoint_resume():
        print(
            "Resume fallback: torch<2.6 blocks optimizer/scheduler torch.load in Transformers; "
            "restoring adapter weights and trainer state without binary optimizer/scheduler state."
        )
    with maybe_hide_unsafe_resume_state(args.resume_from_checkpoint):
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir))
    print(f"Training complete. Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
