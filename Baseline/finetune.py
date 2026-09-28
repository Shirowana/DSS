from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import accelerate
import datasets
import transformers
from datasets import load_from_disk
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from data_processing.common import sha256_file
from training_methods.s2ft_adapter import S2FTAdapter, S2FTRatios, official_ratios
from training_methods.sift_adapter import SIFTAdapter


ROOT = Path(__file__).resolve().parent
IGNORE_INDEX = -100


@dataclass(frozen=True)
class TaskConfig:
    name: str
    train_rows: int
    validation_rows: int
    max_length: int
    epochs: int
    global_batch_size: int
    expected_steps: int | None
    scheduler: str
    warmup_ratio: float
    default_micro_batch: int
    default_gradient_accumulation: int


TASKS = {
    "commonsense": TaskConfig(
        "commonsense", 170_420, 0, 256, 3, 32, None, "linear", 0.0, 4, 4
    ),
    "math": TaskConfig("math", 9_419, 500, 512, 3, 8, 3_534, "linear", 0.03, 4, 1),
    "code": TaskConfig("code", 111_183, 0, 3072, 1, 128, 869, "cosine", 0.03, 4, 16),
}
MATH_CHECKPOINT_STEPS = (2651, 2828, 3004, 3181, 3358, 3534)
SIFT_DEFAULT_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(default_task: str = "commonsense") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared S2FT/SIFT trainer. finetune.py defaults to Commonsense."
    )
    parser.add_argument("--task", choices=TASKS, default=default_task)
    parser.add_argument("--method", choices=("s2ft", "sift"), required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--prepared_data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--log_dir", type=Path)
    parser.add_argument("--run_id")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    # Checkpointing is a memory-saving option, not part of S2FT/SIFT itself.
    # Keep it opt-in so the standard runs use the user's normal non-checkpointed
    # training setup; pass --gradient_checkpointing when memory requires it.
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--s2_v_ratio", type=float)
    parser.add_argument("--s2_o_ratio", type=float)
    parser.add_argument("--s2_u_ratio", type=float)
    parser.add_argument("--s2_d_ratio", type=float)
    parser.add_argument("--sift_rate", type=float, default=0.01)
    parser.add_argument("--sift_modules", nargs="+", default=list(SIFT_DEFAULT_MODULES))
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    return parser.parse_args()


def model_family(path: str) -> str:
    lowered = path.lower()
    if "llama-2" in lowered or "llama2" in lowered:
        return "llama2"
    if "llama-3" in lowered or "llama3" in lowered:
        return "llama3"
    raise ValueError("Cannot infer Llama family from --model_name_or_path")


def default_learning_rate(task: str, family: str) -> float:
    values = {
        ("commonsense", "llama2"): 1.5e-4,
        ("commonsense", "llama3"): 8e-5,
        ("math", "llama2"): 4e-4,
        ("math", "llama3"): 5e-5,
        ("code", "llama3"): 1.5e-4,
    }
    try:
        return values[(task, family)]
    except KeyError as error:
        raise ValueError(f"No protocol learning rate for task={task}, family={family}") from error


def validate_model_task(task: str, model_path: str) -> None:
    lowered = model_path.lower()
    instruct = "instruct" in lowered or "chat" in lowered
    if task == "code" and not instruct:
        raise ValueError("Code requires Meta-Llama-3-8B-Instruct")
    if task != "code" and instruct:
        raise ValueError(f"{task} requires a base model, not an Instruct/chat model")


def load_prepared(path: Path, task: TaskConfig):
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != task.name:
        raise ValueError(f"Prepared task is {manifest.get('task')!r}, expected {task.name!r}")
    if manifest.get("preprocessing", {}).get("max_length") != task.max_length:
        raise ValueError("Prepared max_length conflicts with the frozen task protocol")
    train = load_from_disk(str(path / "train"))
    if len(train) != task.train_rows:
        raise ValueError(f"Expected {task.train_rows} training rows, found {len(train)}")
    validation = None
    if task.validation_rows:
        validation = load_from_disk(str(path / "validation"))
        if len(validation) != task.validation_rows:
            raise ValueError(
                f"Expected {task.validation_rows} validation rows, found {len(validation)}"
            )
    return manifest, train, validation


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in features]
        labels = [torch.tensor(row["labels"], dtype=torch.long) for row in features]
        padded_ids = nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        padded_labels = nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return {
            "input_ids": padded_ids,
            "labels": padded_labels,
            "attention_mask": padded_ids.ne(self.pad_token_id),
        }


class MethodState:
    def __init__(self, method: str, model, args: argparse.Namespace, family: str):
        self.method = method
        self.model = model
        self.s2ft = None
        self.sift = None
        if method == "s2ft":
            defaults = official_ratios(family)
            ratios = S2FTRatios(
                v=defaults.v if args.s2_v_ratio is None else args.s2_v_ratio,
                o=defaults.o if args.s2_o_ratio is None else args.s2_o_ratio,
                u=defaults.u if args.s2_u_ratio is None else args.s2_u_ratio,
                d=defaults.d if args.s2_d_ratio is None else args.s2_d_ratio,
            )
            self.s2ft = S2FTAdapter(model, family, args.seed, ratios)
            self.manifest = self.s2ft.manifest
        else:
            self.sift = SIFTAdapter(
                model,
                sparse_rate=args.sift_rate,
                sparse_modules=args.sift_modules,
                grad_acc=args.gradient_accumulation_steps,
                gradient_checkpointing=args.gradient_checkpointing,
            )
            self.manifest = {
                **self.sift.manifest,
                "trainable_sparse_values": self.sift.get_trainable_num(),
            }

    def optimizer_named_parameters(self):
        if self.s2ft is not None:
            return self.s2ft.optimizer_named_parameters()
        if self.sift is not None:
            return list(self.sift.named_parameters_in_optimizer())
        return [(name, value) for name, value in self.model.named_parameters() if value.requires_grad]

    def set_trainer(self, trainer: Trainer) -> None:
        if self.s2ft is not None:
            self.s2ft.verify_distributed_selection()
        if self.sift is not None:
            self.sift.set_trainer(trainer)

    def sync_sift_gradients(self) -> None:
        if self.sift is None:
            return
        self.sift.sync_gradients()

    def apply_sift_delta(self) -> None:
        if self.sift is None:
            return
        self.sift.apply_delta()

    def clear_sift_dense_gradients(self) -> None:
        if self.sift is None:
            return
        self.sift.clear_dense_gradients()

    def save_candidate(self, path: Path) -> None:
        if self.s2ft is not None:
            self.s2ft.save_candidate(path)
            return
        self.sift.save_candidate(path)

    def load_candidate(self, path: Path) -> None:
        if self.s2ft is not None:
            self.s2ft.load_candidate(path)
            return
        self.sift.load_candidate(path)

    def prepare_dense_export(self) -> None:
        if self.s2ft is not None:
            self.apply_sift_delta()
            self.s2ft.densify()
        elif self.sift is not None:
            self.sift.prepare_dense_export()


class MethodCallback(TrainerCallback):
    def __init__(self, method_state: MethodState):
        self.method_state = method_state

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.method_state.sync_sift_gradients()
        if self.method_state.sift is not None:
            self.method_state.clear_sift_dense_gradients()
            torch.nn.utils.clip_grad_norm_(
                list(self.method_state.sift.parameters_in_optimizer()), max_norm=1.0
            )
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.method_state.apply_sift_delta()
        return control


class MathCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: Path, method_state: MethodState):
        self.output_dir = output_dir
        self.method_state = method_state
        self.records: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in MATH_CHECKPOINT_STEPS:
            control.should_evaluate = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.global_step not in MATH_CHECKPOINT_STEPS:
            return control
        if not metrics or "eval_loss" not in metrics:
            raise RuntimeError(f"No eval_loss at eligible Math step {state.global_step}")
        record = {"step": state.global_step, "eval_loss": float(metrics["eval_loss"])}
        if state.is_world_process_zero:
            candidate_dir = self.output_dir / "candidates" / f"step-{state.global_step}"
            self.method_state.save_candidate(candidate_dir / "method_state.pt")
            atomic_write_json(candidate_dir / "metrics.json", record)
        self.records.append(record)
        if self.best is None or record["eval_loss"] < self.best["eval_loss"]:
            self.best = record
        return control


def build_optimizer(method_state: MethodState, args: argparse.Namespace):
    no_decay = ("bias", "layernorm.weight", "layer_norm.weight", "norm.weight")
    named = method_state.optimizer_named_parameters()
    groups = [
        {
            "params": [p for name, p in named if not any(key in name.lower() for key in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for name, p in named if any(key in name.lower() for key in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    groups = [group for group in groups if group["params"]]
    beta2 = args.adam_beta2
    if beta2 is None:
        beta2 = 0.95 if args.method == "s2ft" else 0.999
    return torch.optim.AdamW(
        groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, beta2),
        eps=args.adam_epsilon,
        weight_decay=0.0,
    ), beta2


def run(default_task: str = "commonsense") -> None:
    args = parse_args(default_task)
    task = TASKS[args.task]
    validate_model_task(task.name, args.model_name_or_path)
    family = model_family(args.model_name_or_path)
    if args.task == "code" and family != "llama3":
        raise ValueError("Code protocol supports Llama3-8B-Instruct only")
    args.learning_rate = args.learning_rate or default_learning_rate(task.name, family)
    args.per_device_train_batch_size = (
        args.per_device_train_batch_size or task.default_micro_batch
    )
    args.gradient_accumulation_steps = (
        args.gradient_accumulation_steps or task.default_gradient_accumulation
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    if global_batch != task.global_batch_size:
        raise ValueError(
            f"Protocol global batch is {task.global_batch_size}, but microbatch "
            f"{args.per_device_train_batch_size} * accumulation "
            f"{args.gradient_accumulation_steps} * world_size {world_size} = {global_batch}"
        )
    if task.name == "math" and world_size != 2:
        raise ValueError("Math protocol requires exactly two distributed GPU processes")

    manifest, train_dataset, eval_dataset = load_prepared(args.prepared_data, task)
    updates_per_epoch = math.ceil(len(train_dataset) / global_batch)
    calculated_steps = updates_per_epoch * task.epochs
    if task.expected_steps is not None and calculated_steps != task.expected_steps:
        raise ValueError(
            f"Calculated {calculated_steps} steps, expected {task.expected_steps}; check batching"
        )
    expected_steps = task.expected_steps or calculated_steps

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.prepared_data / "tokenizer", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError(
            f"Prepared tokenizer has {len(tokenizer)} tokens but model expects "
            f"{model.config.vocab_size}"
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_seed42")
    model_slug = Path(args.model_name_or_path.rstrip("/")).name
    output_dir = args.output_dir or ROOT / "outputs" / args.method / task.name / model_slug / run_id
    output_dir = output_dir.resolve()
    log_dir = (args.log_dir or ROOT / "logs" / run_id).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    method_state = MethodState(args.method, model, args, family)
    # Match the conventional report used by the reference runs: total params
    # includes the method's added trainable state. In DDP, print only once.
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for _, parameter in method_state.optimizer_named_parameters())
    trainable_pct = 100.0 * trainable / total_params if total_params else 0.0
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Total params: {total_params:,} | "
            f"Trainable: {trainable:,} ({trainable_pct:.4f}%)",
            flush=True,
        )
    optimizer, beta2 = build_optimizer(method_state, args)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_state"),
        overwrite_output_dir=args.overwrite_output_dir,
        max_steps=expected_steps,
        num_train_epochs=task.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        adam_beta1=args.adam_beta1,
        adam_beta2=beta2,
        adam_epsilon=args.adam_epsilon,
        # SIFT gradients are synchronized and clipped by MethodCallback because
        # its optimizer parameters are not part of the forward graph.
        max_grad_norm=0.0 if args.method == "sift" else 1.0,
        lr_scheduler_type=task.scheduler,
        warmup_ratio=task.warmup_ratio,
        bf16=True,
        bf16_full_eval=True,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_dir=str(log_dir),
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=False,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        ddp_find_unused_parameters=(args.method == "sift"),
    )
    callbacks: list[TrainerCallback] = [MethodCallback(method_state)]
    math_callback = None
    if task.name == "math":
        math_callback = MathCheckpointCallback(output_dir, method_state)
        callbacks.append(math_callback)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )
    method_state.set_trainer(trainer)

    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "task": asdict(task),
        "method": args.method,
        "method_config": method_state.manifest,
        "model_name_or_path": args.model_name_or_path,
        "model_family": family,
        "prepared_data": str(args.prepared_data.resolve()),
        "log_dir": str(log_dir),
        "output_dir": str(output_dir),
        "preprocessing_source_sha256": manifest["source"]["sha256"],
        "preprocessing_manifest_sha256": sha256_file(args.prepared_data / "manifest.json"),
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
        },
        "world_size": world_size,
        "global_batch_size": global_batch,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "betas": [args.adam_beta1, beta2],
            "epsilon": args.adam_epsilon,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "scheduler": task.scheduler,
        "warmup_ratio": task.warmup_ratio,
        "expected_optimizer_steps": expected_steps,
        "total_model_parameters": total_params,
        "trainable_optimizer_values": trainable,
        "trainable_parameter_percent": trainable_pct,
        "precision": "bf16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "checkpoint_policy": (
            {
                "eligible_steps": list(MATH_CHECKPOINT_STEPS),
                "selection": "minimum_eval_loss",
                "early_stopping": False,
            }
            if task.name == "math"
            else {"selection": "final_state", "early_stopping": False}
        ),
    }
    if trainer.is_world_process_zero():
        atomic_write_json(output_dir / "run_manifest.json", run_manifest)

    train_result = trainer.train()
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(
            f"Training ended at step {trainer.state.global_step}, expected {expected_steps}"
        )
    selected_checkpoint = "final_state"
    if math_callback is not None:
        if [item["step"] for item in math_callback.records] != list(MATH_CHECKPOINT_STEPS):
            raise RuntimeError(f"Incomplete Math checkpoint evaluations: {math_callback.records}")
        assert math_callback.best is not None
        best_step = math_callback.best["step"]
        selected_checkpoint = f"candidate-step-{best_step}"
        method_state.load_candidate(
            output_dir / "candidates" / f"step-{best_step}" / "method_state.pt"
        )
        if trainer.is_world_process_zero():
            atomic_write_json(
                output_dir / "checkpoint_selection.json",
                {
                    "eligible": math_callback.records,
                    "selected": math_callback.best,
                    "rule": "minimum_eval_loss; ties keep earliest eligible step",
                },
            )

    method_state.prepare_dense_export()
    model.config.use_cache = True
    final_dir = output_dir / "final_model"
    if trainer.is_world_process_zero():
        model.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
        atomic_write_json(
            output_dir / "train_summary.json",
            {
                "global_step": trainer.state.global_step,
                "training_loss": train_result.training_loss,
                "selected_checkpoint": selected_checkpoint,
                "final_model": str(final_dir),
            },
        )
        atomic_write_json(log_dir / "trainer_log_history.json", trainer.state.log_history)
        atomic_write_json(log_dir / "run_manifest.json", run_manifest)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run("commonsense")
