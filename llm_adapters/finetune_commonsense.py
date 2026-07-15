"""
LLM-Adapters-style commonsense finetuning entrypoint adapted for DSS experiments.

This keeps the core online prompt/tokenize flow of AGI-Edgerunners/LLM-Adapters
while adding:
- Qwen-family compatibility
- LoRA, DoRA, DSS, and SHiRA support
- bf16/fp16 precision selection
- configurable local data/model paths
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from peft import LoraConfig, ShiraConfig, get_peft_model
from peft.tuners.dss import DSSConfig  # Registers the DSS PEFT type for save/load.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-Adapters-style commonsense finetuning.")

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_name", choices=["lora", "dora", "dss", "shira"], default="lora")
    parser.add_argument("--load_8bit", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size.")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Per-device micro batch size.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--val_set_size", type=int, default=500)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--eval_step", type=int, default=10000)
    parser.add_argument("--save_step", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--shira_r", type=int, default=32)
    parser.add_argument("--shira_mask_type", choices=["random", "snip"], default="random")
    parser.add_argument("--shira_random_seed", type=int, default=42)
    parser.add_argument(
        "--shira_calibration_samples",
        type=int,
        default=16,
        help="Examples in the one-shot SNIP calibration batch. Ignored for random SHiRA.",
    )
    parser.add_argument(
        "--shira_init_weights",
        type=int,
        choices=[0, 1],
        default=1,
        help="Initialize SHiRA sparse deltas to zero (1) or random values (0).",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,up_proj,down_proj",
        help="Comma-separated target module names.",
    )

    # DSS defaults reproduce the 87.2 Commonsense baseline configuration.
    parser.add_argument("--n_frequency", type=int, default=180000)
    parser.add_argument("--candidate_size", type=int, default=10000)
    parser.add_argument("--grad_store_steps", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--low", type=float, default=100.0)
    parser.add_argument("--up", type=float, default=1000.0)
    parser.add_argument("--threshold_mode", choices=["oracle", "sgd"], default="oracle")
    parser.add_argument("--score_method", default="snr")
    parser.add_argument("--score_eps", type=float, default=1e-8)
    parser.add_argument("--dss_dropout", type=float, default=0.05)
    parser.add_argument("--quantile_lr", type=float, default=0.01)
    parser.add_argument("--quantile_alpha", type=float, default=0.0)
    parser.add_argument("--threshold_log_every_steps", type=int, default=100)

    parser.add_argument("--train_on_inputs", type=int, choices=[0, 1], default=1)
    parser.add_argument("--group_by_length", action="store_true", default=False)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def generate_prompt(data_point: dict) -> str:
    if data_point.get("input"):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{data_point['instruction']}\n\n"
            f"### Input:\n{data_point['input']}\n\n"
            f"### Response:\n{data_point['output']}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{data_point['instruction']}\n\n"
        f"### Response:\n{data_point['output']}"
    )


def parse_target_modules(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def prepare_shira_snip_gradients(
    model: nn.Module,
    calibration_data,
    data_collator: DataCollatorForSeq2Seq,
    device: torch.device,
    target_modules: list[str],
) -> dict[str, object]:
    """Populate target base-layer gradients for one-shot SNIP mask construction."""

    targets = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target_modules
    }
    if not targets:
        raise ValueError(f"No nn.Linear calibration targets found for {target_modules!r}.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in targets.values():
        module.weight.requires_grad_(True)

    batch = data_collator([calibration_data[index] for index in range(len(calibration_data))])
    batch = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()

    missing = [name for name, module in targets.items() if module.weight.grad is None]
    if missing:
        raise RuntimeError(f"SNIP calibration produced no gradients for: {', '.join(missing[:5])}")

    return {
        "loss": float(loss.detach().float().item()),
        "samples": len(calibration_data),
        "target_layer_count": len(targets),
        "target_layers": sorted(targets),
    }


def main() -> None:
    args = parse_args()
    transformers.set_seed(args.seed)

    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive.")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by micro_batch_size ({args.micro_batch_size})."
        )
    if args.adapter_name == "shira" and args.shira_r <= 0:
        raise ValueError("--shira_r must be positive when --adapter_name shira.")
    if args.adapter_name == "shira" and args.shira_mask_type == "snip" and args.shira_calibration_samples <= 0:
        raise ValueError("--shira_calibration_samples must be positive for SHiRA SNIP.")

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if torch.cuda.is_available() else {"": "cpu"}

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch_dtype,
        device_map=device_map if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tokenize(prompt: str, add_eos_token: bool = True) -> dict:
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            add_eos_token
            and result["input_ids"]
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point: dict) -> dict:
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not bool(args.train_on_inputs):
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    if args.data_path.endswith(".json"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    raw_columns = data["train"].column_names

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=args.seed)
        train_data = train_val["train"].shuffle(seed=args.seed).map(
            generate_and_tokenize_prompt,
            remove_columns=raw_columns,
        )
        val_data = train_val["test"].shuffle(seed=args.seed).map(
            generate_and_tokenize_prompt,
            remove_columns=raw_columns,
        )
    else:
        train_data = data["train"].shuffle(seed=args.seed).map(
            generate_and_tokenize_prompt,
            remove_columns=raw_columns,
        )
        val_data = None

    target_modules = parse_target_modules(args.target_modules)
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    shira_calibration_metadata = None
    if args.adapter_name == "shira" and args.shira_mask_type == "snip":
        calibration_count = min(args.shira_calibration_samples, len(train_data))
        calibration_data = train_data.select(range(calibration_count))
        shira_calibration_metadata = prepare_shira_snip_gradients(
            model=model,
            calibration_data=calibration_data,
            data_collator=data_collator,
            device=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu"),
            target_modules=target_modules,
        )
        shira_calibration_metadata["mask_type"] = "snip"
        shira_calibration_metadata["rank"] = args.shira_r
        shira_calibration_metadata["train_data_indices"] = list(range(calibration_count))
        print(f"[SHiRA SNIP] calibration={shira_calibration_metadata}", flush=True)

    if args.adapter_name == "dss":
        config = DSSConfig(
            target_modules=target_modules,
            n_frequency=args.n_frequency,
            candidate_size=args.candidate_size,
            grad_store_steps=args.grad_store_steps,
            ratio=args.ratio,
            low=args.low,
            up=args.up,
            threshold_mode=args.threshold_mode,
            score_method=args.score_method,
            score_eps=args.score_eps,
            dropout=args.dss_dropout,
            quantile_lr=args.quantile_lr,
            quantile_alpha=args.quantile_alpha,
            threshold_log_every_steps=args.threshold_log_every_steps,
            bias="none",
        )
    elif args.adapter_name == "shira":
        config = ShiraConfig(
            r=args.shira_r,
            mask_type=args.shira_mask_type,
            random_seed=args.shira_random_seed,
            target_modules=target_modules,
            init_weights=bool(args.shira_init_weights),
        )
    else:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=(args.adapter_name == "dora"),
        )
    model = get_peft_model(model, config)
    model.zero_grad(set_to_none=True)
    model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "llm_adapters_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    if shira_calibration_metadata is not None:
        with (output_dir / "shira_calibration.json").open("w", encoding="utf-8") as handle:
            json.dump(shira_calibration_metadata, handle, indent=2, ensure_ascii=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        logging_steps=args.logging_steps,
        optim="adamw_torch",
        eval_strategy="steps" if args.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_step if args.val_set_size > 0 else None,
        save_steps=args.save_step,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if args.val_set_size > 0 else False,
        # DSS progressively activates sparse coefficients, so DDP must tolerate unused parameters.
        ddp_find_unused_parameters=True if ddp and args.adapter_name == "dss" else False if ddp else None,
        group_by_length=args.group_by_length,
        report_to=[] if args.report_to.lower() in {"none", "no", "false", "0"} else [args.report_to],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("\nTraining complete.")


if __name__ == "__main__":
    main()
