"""
Fine-tune Llama with DSS on the preprocessed commonsense dataset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_from_disk
from packaging.version import Version
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires `peft` with DSS registration support.") from exc

from peft.tuners.dss import DSSConfig, DSSLayer  # noqa: F401 - importing peft.tuners.dss registers the PEFT method


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

TARGET_MODULE_ALIASES = {
    "qkvud": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    "qkvupdown": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    "qkvod": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"],
}

COMMONSENSE_TASKS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]


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
    parser.add_argument("--peft_method", type=str, default="dss", choices=["dss", "lora", "dora"])
    parser.add_argument("--n_frequency", type=int, default=8)
    parser.add_argument("--candidate_size", type=int, default=32)
    parser.add_argument("--grad_store_steps", type=int, default=10)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--up", type=float, default=4.0)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--threshold_mode", type=str, default="oracle", choices=["oracle", "sgd"])
    parser.add_argument(
        "--score_method",
        type=str,
        default="abs_mean",
        choices=[
            "mean_abs",
            "abs_mean",
            "mean_square",
            "rms_over_param",
            "abs_mean_over_param",
            "snr",
            "newton_like",
        ],
    )
    parser.add_argument("--score_eps", type=float, default=1e-8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--quantile_lr", type=float, default=0.01)
    parser.add_argument("--quantile_alpha", type=float, default=0.0)
    parser.add_argument("--threshold_log_every_steps", type=int, default=100)
    parser.add_argument("--init_enabled", action="store_true")
    parser.add_argument("--init_steps", type=int, default=10)
    parser.add_argument("--init_candidate_ratio", type=float, default=0.05)
    parser.add_argument("--init_seed_mode", type=str, default="threshold_only", choices=["threshold_only", "seed_elite"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_rslora", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
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
    if args.peft_method == "dss":
        return (
            f"commonsense_{args.model_name}_dss_"
            f"nf{args.n_frequency}_cand{args.candidate_size}_gs{args.grad_store_steps}_{timestamp}"
        )
    return (
        f"commonsense_{args.model_name}_{args.peft_method}_"
        f"r{args.lora_r}_a{args.lora_alpha}_lr{args.lr}_{timestamp}"
    )


def build_peft_model(model, args: argparse.Namespace, target_modules: list[str]):
    if args.peft_method == "dss":
        peft_config = DSSConfig(
            target_modules=target_modules,
            n_frequency=args.n_frequency,
            candidate_size=args.candidate_size,
            grad_store_steps=args.grad_store_steps,
            low=args.low,
            up=args.up,
            ratio=args.ratio,
            threshold_mode=args.threshold_mode,
            score_method=args.score_method,
            score_eps=args.score_eps,
            dropout=args.dropout,
            quantile_lr=args.quantile_lr,
            quantile_alpha=args.quantile_alpha,
            threshold_log_every_steps=args.threshold_log_every_steps,
            init_enabled=args.init_enabled,
            init_steps=args.init_steps,
            init_candidate_ratio=args.init_candidate_ratio,
            init_seed_mode=args.init_seed_mode,
            bias="none",
        )
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=args.use_rslora,
            use_dora=(args.peft_method == "dora"),
        )
    return get_peft_model(model, peft_config)


def resolve_target_modules(spec: str) -> list[str]:
    normalized = (spec or "").strip().lower()
    if not normalized:
        return []
    if normalized in TARGET_MODULE_ALIASES:
        return TARGET_MODULE_ALIASES[normalized]

    resolved: list[str] = []
    i = 0
    while i < len(normalized):
        if normalized.startswith("down", i):
            key = "d"
            i += 4
        elif normalized.startswith("up", i):
            key = "u"
            i += 2
        elif normalized.startswith("gate", i):
            key = "g"
            i += 4
        else:
            key = normalized[i]
            i += 1

        module_name = MODULE_MAP.get(key)
        if module_name and module_name not in resolved:
            resolved.append(module_name)
    return resolved


def resolve_precision(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


@contextmanager
def maybe_autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        yield
        return
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


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


def resolve_train_eval_datasets(dataset_obj, val_set_size: int, seed: int) -> tuple[Dataset, Dataset | None]:
    if isinstance(dataset_obj, DatasetDict) and "train" in dataset_obj and "validation" in dataset_obj:
        return dataset_obj["train"], dataset_obj["validation"]
    if hasattr(dataset_obj, "keys") and "train" in dataset_obj and "validation" in dataset_obj:
        return dataset_obj["train"], dataset_obj["validation"]

    full_train_dataset = resolve_train_dataset(dataset_obj)
    if val_set_size > 0:
        if val_set_size >= len(full_train_dataset):
            raise ValueError(
                f"`--val_set_size` ({val_set_size}) must be smaller than the dataset size ({len(full_train_dataset)})."
            )
        split = full_train_dataset.train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
        return split["train"], split["test"]
    return full_train_dataset, None


def latest_checkpoint_dir(output_dir: Path) -> Path | None:
    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    return checkpoint_dirs[-1] if checkpoint_dirs else None


def build_balanced_init_dataset(train_dataset: Dataset, batch_size: int, init_steps: int, seed: int) -> Dataset:
    if "task_name" not in train_dataset.column_names:
        raise ValueError(
            "Balanced init requires the training dataset to include a `task_name` column. "
            "Please re-run data_processing/process_commonsense.py to regenerate the dataset."
        )

    total_examples = batch_size * init_steps
    base = total_examples // len(COMMONSENSE_TASKS)
    remainder = total_examples % len(COMMONSENSE_TASKS)
    target_counts = {task: base for task in COMMONSENSE_TASKS}
    for task_name in COMMONSENSE_TASKS[:remainder]:
        target_counts[task_name] += 1

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, task_name in enumerate(train_dataset["task_name"]):
        grouped_indices[task_name].append(index)

    missing_tasks = [task for task in COMMONSENSE_TASKS if task not in grouped_indices]
    if missing_tasks:
        raise ValueError(f"Balanced init is missing task_name groups: {missing_tasks}")

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for task_name in COMMONSENSE_TASKS:
        task_indices = grouped_indices[task_name]
        need = target_counts[task_name]
        if len(task_indices) < need:
            raise ValueError(
                f"Task {task_name!r} only has {len(task_indices)} train samples, but balanced init needs {need}."
            )
        chosen = rng.sample(task_indices, need)
        selected_indices.extend(chosen)

    rng.shuffle(selected_indices)
    if len(selected_indices) != total_examples:
        raise RuntimeError(
            f"Balanced init selected {len(selected_indices)} examples, expected exactly {total_examples}."
        )
    return train_dataset.select(selected_indices)


def run_init_observation(
    model,
    init_dataset: Dataset,
    collate_fn,
    device: torch.device,
    precision: str,
    batch_size: int,
    init_steps: int,
) -> float:
    model.train()
    init_loader = DataLoader(init_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    init_start = datetime.now()
    observed_steps = 0
    model.zero_grad(set_to_none=True)
    for batch in init_loader:
        next_step = observed_steps + 1
        print(f"[DSS init-step] begin step={next_step}/{init_steps}", flush=True)
        batch = {
            key: value.to(device=device, non_blocking=(device.type == "cuda"))
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        model.zero_grad(set_to_none=True)
        print(f"[DSS init-step] forward step={next_step}/{init_steps}", flush=True)
        with maybe_autocast(device, precision):
            outputs = model(**batch)
            loss = outputs.loss
        print(f"[DSS init-step] backward step={next_step}/{init_steps}", flush=True)
        loss.backward()
        observed_steps += 1
        elapsed = (datetime.now() - init_start).total_seconds()
        print(
            f"[DSS init-step] done step={observed_steps}/{init_steps} "
            f"loss={float(loss.detach().item()):.6f} "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if observed_steps >= init_steps:
            break
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if observed_steps != init_steps:
        raise RuntimeError(f"Init observation finished with {observed_steps} steps, expected {init_steps}.")
    return (datetime.now() - init_start).total_seconds()


def ddp_barrier(local_rank: int) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available() and local_rank >= 0:
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def finalize_dss_init(model) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    start = time.perf_counter()
    print("[DSS init-finalize] start", flush=True)
    family_elapsed: dict[str, float] = defaultdict(float)
    layer_elapsed: dict[str, float] = defaultdict(float)
    finalized_modules = 0
    current_layer: str | None = None
    layer_start: float | None = None

    for module in model.modules():
        if not isinstance(module, DSSLayer):
            continue
        for active_adapter in module.active_adapters:
            runtime = module.runtime.get(active_adapter)
            if runtime is None or not runtime.init_phase:
                continue
            module_name = module.module_name.get(active_adapter, "")
            parts = module_name.split(".")
            layer_id = parts[2] if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers" else "unknown"
            family = parts[-1] if parts else "unknown"

            if layer_id != current_layer:
                if current_layer is not None and layer_start is not None:
                    elapsed = time.perf_counter() - layer_start
                    print(f"[DSS init-finalize] layer_done layer={current_layer} elapsed={elapsed:.2f}s", flush=True)
                current_layer = layer_id
                layer_start = time.perf_counter()
                print(f"[DSS init-finalize] layer_start layer={current_layer}", flush=True)

            module_start = time.perf_counter()
            module.maybe_refresh_stage1(active_adapter)
            elapsed = time.perf_counter() - module_start
            family_elapsed[family] += elapsed
            layer_elapsed[layer_id] += elapsed
            finalized_modules += 1

            if finalized_modules % 10 == 0:
                total_elapsed = time.perf_counter() - start
                print(
                    f"[DSS init-finalize] progress modules={finalized_modules}/160 elapsed={total_elapsed:.2f}s "
                    f"last_module={module_name} module_elapsed={elapsed:.2f}s",
                    flush=True,
                )

    if current_layer is not None and layer_start is not None:
        elapsed = time.perf_counter() - layer_start
        print(f"[DSS init-finalize] layer_done layer={current_layer} elapsed={elapsed:.2f}s", flush=True)

    total_elapsed = time.perf_counter() - start
    top_layers = sorted(layer_elapsed.items(), key=lambda item: item[1], reverse=True)[:5]
    top_families = sorted(family_elapsed.items(), key=lambda item: item[1], reverse=True)
    print(
        f"[DSS init-finalize] all_thresholds_done modules={finalized_modules} elapsed={total_elapsed:.2f}s "
        f"top_layers={top_layers} family_elapsed={top_families}",
        flush=True,
    )


def prepare_dss_init_candidates(model) -> float:
    start = datetime.now()
    prepared = 0
    for module in model.modules():
        if not isinstance(module, DSSLayer):
            continue
        for active_adapter in module.active_adapters:
            runtime = module.runtime.get(active_adapter)
            if runtime is None or not runtime.init_phase:
                continue
            if module.candidate_indices[active_adapter].numel() == 0:
                module.refresh_init_candidate_batch(active_adapter)
                prepared += 1
    elapsed = (datetime.now() - start).total_seconds()
    print(f"[DSS init-prepare] prepared_modules={prepared} elapsed={elapsed:.2f}s", flush=True)
    return elapsed


def sync_dss_init_state(model, src_rank: int = 0) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return

    rank = dist.get_rank()
    for module in model.modules():
        if not isinstance(module, DSSLayer):
            continue
        for active_adapter in module.active_adapters:
            device = module.coefficient[active_adapter].device
            runtime = module.runtime[active_adapter]

            meta = torch.zeros(6, device=device, dtype=torch.long)
            floats = torch.zeros(2, device=device, dtype=torch.float32)
            coeff = torch.empty(0, device=device, dtype=torch.float32)
            indices = torch.empty(0, device=device, dtype=torch.long)

            if rank == src_rank:
                exported = module.export_sparse_checkpoint(active_adapter)
                coeff = exported["coefficient"].to(device=device, dtype=torch.float32)
                indices = exported["coefficient_indices"].to(device=device, dtype=torch.long)
                meta[0] = runtime.curr_count
                meta[1] = runtime.total_steps
                meta[2] = runtime.refresh_rounds
                meta[3] = runtime.last_logged_step
                meta[4] = 1 if runtime.init_phase else 0
                meta[5] = 1 if runtime.init_done else 0
                floats[0] = float("nan") if runtime.pending_init_threshold is None else float(runtime.pending_init_threshold)
                floats[1] = float(module.search_quantile_estimator[active_adapter].get_quantile().item())

            dist.broadcast(meta, src=src_rank)
            dist.broadcast(floats, src=src_rank)

            curr_count = int(meta[0].item())
            if rank != src_rank:
                coeff = torch.empty(curr_count, device=device, dtype=torch.float32)
                indices = torch.empty(curr_count, device=device, dtype=torch.long)
            if curr_count > 0:
                dist.broadcast(coeff, src=src_rank)
                dist.broadcast(indices, src=src_rank)

            if rank != src_rank:
                module.restore_sparse_checkpoint(active_adapter, coeff, indices)
            runtime = module.runtime[active_adapter]
            runtime.curr_count = curr_count
            runtime.total_steps = int(meta[1].item())
            runtime.refresh_rounds = int(meta[2].item())
            runtime.last_logged_step = int(meta[3].item())
            runtime.init_phase = bool(meta[4].item())
            runtime.init_done = bool(meta[5].item())
            pending = float(floats[0].item())
            runtime.pending_init_threshold = None if torch.isnan(floats[0]).item() else pending
            module.search_quantile_estimator[active_adapter].quantile.data.fill_(float(floats[1].item()))


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
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("`--warmup_ratio` must be within [0, 1].")
    if args.load_best_model_at_end and args.val_set_size <= 0:
        raise ValueError("`--load_best_model_at_end` requires `--val_set_size > 0`.")
    if args.val_set_size > 0 and args.eval_steps <= 0:
        raise ValueError("`--eval_steps` must be positive when using a validation split.")
    if args.peft_method != "dss" and args.init_enabled:
        raise ValueError("`--init_enabled` is only supported for `--peft_method dss`.")
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

    target_modules = resolve_target_modules(args.target_modules)
    if not target_modules:
        raise ValueError("`--target_modules` did not resolve to any known module names.")

    if args.peft_method != "dss":
        print(
            "[PEFT] Using "
            f"{args.peft_method}; DSS-only args are inactive: "
            "n_frequency, candidate_size, grad_store_steps, low, up, ratio, "
            "threshold_mode, score_method, score_eps, quantile_lr, quantile_alpha, "
            "threshold_log_every_steps, init_*",
            flush=True,
        )
    model = build_peft_model(model, args, target_modules)

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
    train_dataset, eval_dataset = resolve_train_eval_datasets(
        load_from_disk(str(dataset_path)),
        val_set_size=args.val_set_size,
        seed=args.seed,
    )

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
        warmup_ratio=args.warmup_ratio,
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
        ddp_find_unused_parameters=(True if (world_size > 1 and args.peft_method == "dss") else False if world_size > 1 else None),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    if args.init_enabled:
        if args.resume_from_checkpoint:
            raise ValueError("`--init_enabled` is not supported together with `--resume_from_checkpoint` yet.")
        if world_size > 1:
            if local_rank in (-1, 0):
                init_dataset = build_balanced_init_dataset(
                    train_dataset=train_dataset,
                    batch_size=args.batch_size,
                    init_steps=args.init_steps,
                    seed=args.seed,
                )
                print(
                    f"[DSS init] start balanced-task observation (rank0): "
                    f"steps={args.init_steps} batch_size={args.batch_size} total_examples={len(init_dataset)}"
                )
                os.environ["DSS_DISABLE_DDP_CANDIDATE_SYNC"] = "1"
                try:
                    prepare_dss_init_candidates(model)
                    init_runtime = run_init_observation(
                        model=model,
                        init_dataset=init_dataset,
                        collate_fn=collate_fn,
                        device=device,
                        precision=args.precision,
                        batch_size=args.batch_size,
                        init_steps=args.init_steps,
                    )
                finally:
                    os.environ.pop("DSS_DISABLE_DDP_CANDIDATE_SYNC", None)
                finalize_dss_init(model)
                print(f"[DSS init] observation complete (rank0): runtime={init_runtime:.2f}s")
            ddp_barrier(local_rank)
            if local_rank in (-1, 0):
                print("[DSS init-sync] start", flush=True)
            sync_dss_init_state(model, src_rank=0)
            ddp_barrier(local_rank)
            if local_rank in (-1, 0):
                print("[DSS init-sync] done", flush=True)
                print("[DSS init] synchronized init state across DDP ranks")
        else:
            init_dataset = build_balanced_init_dataset(
                train_dataset=train_dataset,
                batch_size=args.batch_size,
                init_steps=args.init_steps,
                seed=args.seed,
            )
            print(
                f"[DSS init] start balanced-task observation: "
                f"steps={args.init_steps} batch_size={args.batch_size} total_examples={len(init_dataset)}"
            )
            prepare_dss_init_candidates(model)
            init_runtime = run_init_observation(
                model=model,
                init_dataset=init_dataset,
                collate_fn=collate_fn,
                device=device,
                precision=args.precision,
                batch_size=args.batch_size,
                init_steps=args.init_steps,
            )
            finalize_dss_init(model)
            print(f"[DSS init] observation complete: runtime={init_runtime:.2f}s")

    if args.resume_from_checkpoint and not torch_supports_safe_checkpoint_resume():
        print(
            "Resume fallback: torch<2.6 blocks optimizer/scheduler torch.load in Transformers; "
            "restoring adapter weights and trainer state without binary optimizer/scheduler state."
        )
    with maybe_hide_unsafe_resume_state(args.resume_from_checkpoint):
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    if has_eval:
        best_checkpoint = trainer.state.best_model_checkpoint or ""
        final_checkpoint = latest_checkpoint_dir(output_dir)
        final_checkpoint_step = trainer.state.global_step

        if final_checkpoint is not None:
            print(f"Loading final checkpoint for validation: {final_checkpoint}")
            trainer._load_from_checkpoint(str(final_checkpoint))

        final_eval_metrics = trainer.evaluate(metric_key_prefix="final_eval")
        final_eval_metrics["final_eval_step"] = final_checkpoint_step
        final_eval_metrics["final_eval_checkpoint"] = str(final_checkpoint) if final_checkpoint is not None else ""
        with (output_dir / "final_eval_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(final_eval_metrics, handle, indent=2, ensure_ascii=False)
        print(
            "Final-checkpoint validation complete: "
            f"step={final_checkpoint_step} "
            f"checkpoint={final_eval_metrics.get('final_eval_checkpoint')} "
            f"final_eval_loss={final_eval_metrics.get('final_eval_loss')}"
        )
        if args.load_best_model_at_end and best_checkpoint:
            print(f"Reloading best checkpoint for final adapter export: {best_checkpoint}")
            trainer._load_from_checkpoint(best_checkpoint)
    trainer.save_model(str(output_dir))
    print(f"Training complete. Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
