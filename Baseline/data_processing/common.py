from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset
import datasets
import transformers
from transformers import AutoTokenizer


IGNORE_INDEX = -100
ALPACA_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    "### Response:\n"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


@dataclass(frozen=True)
class PrepareSpec:
    task: str
    source: Path
    output_dir: Path
    model_name_or_path: str
    max_length: int
    expected_rows: int
    seed: int = 42
    num_proc: int = 1
    overwrite: bool = False
    trust_remote_code: bool = False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list/JSONL of objects in {path}")
    return rows


def alpaca_prompt(instruction: str, input_text: str = "") -> str:
    values = {"instruction": instruction.strip(), "input": input_text.strip()}
    template = ALPACA_PROMPT_INPUT if values["input"] else ALPACA_PROMPT_NO_INPUT
    return template.format_map(values)


def load_tokenizer(spec: PrepareSpec):
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_name_or_path,
        use_fast=True,
        trust_remote_code=spec.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = spec.max_length
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_alpaca_row(
    row: dict[str, Any], tokenizer, max_length: int, preserve_response: bool = False
) -> dict[str, Any]:
    source = alpaca_prompt(str(row["instruction"]), str(row.get("input", "")))
    response = str(row["output"]).replace(tokenizer.eos_token or "", "").rstrip()
    source_ids = tokenizer(source, add_special_tokens=True, truncation=False)["input_ids"]
    target_ids = tokenizer(response, add_special_tokens=False, truncation=False)["input_ids"]
    target_ids.append(tokenizer.eos_token_id)
    full_length = len(source_ids) + len(target_ids)
    if preserve_response:
        source_ids, target_ids = truncate_alpaca_tokens(source_ids, target_ids, max_length)
    else:
        source_ids, target_ids = source_ids[:max_length], target_ids[: max(0, max_length - len(source_ids))]
    input_ids = source_ids + target_ids
    supervised = len(target_ids)
    labels = [IGNORE_INDEX] * len(source_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "length": len(input_ids),
        "source_length": len(source_ids),
        "target_length": len(target_ids),
        "supervised_tokens": supervised,
        "was_truncated": full_length > max_length,
    }


def truncate_alpaca_tokens(
    source_ids: list[int], target_ids: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    """Keep the response (including EOS) and right-truncate the Alpaca prompt."""
    if len(source_ids) + len(target_ids) <= max_length:
        return source_ids, target_ids
    if len(target_ids) > max_length:
        # Preserve the terminator when an unusually long answer fills the window.
        target_ids = target_ids[: max_length - 1] + [target_ids[-1]]
        return [], target_ids
    source_budget = max_length - len(target_ids)
    return source_ids[:source_budget], target_ids


def tokenize_code_row(row: dict[str, Any], tokenizer, max_length: int) -> dict[str, Any]:
    if tokenizer.chat_template is None:
        raise ValueError("Code preprocessing requires the model tokenizer's chat_template")
    messages = [{"role": "user", "content": str(row["instruction"])}]
    source_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(source_ids, "tolist"):
        source_ids = source_ids.tolist()
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if not isinstance(eot_id, int) or eot_id < 0 or eot_id >= len(tokenizer):
        raise ValueError("Code preprocessing requires a valid <|eot_id|> token")
    response = str(row["response"]).replace("<|eot_id|>", "").rstrip()
    target_ids = tokenizer(response, add_special_tokens=False, truncation=False)["input_ids"]
    target_ids.append(eot_id)
    original_source_length = len(source_ids)
    original_target_length = len(target_ids)
    full_length = original_source_length + original_target_length
    source_ids, target_ids = truncate_code_tokens(source_ids, target_ids, max_length)
    input_ids = source_ids + target_ids
    supervised = len(target_ids)
    labels = [IGNORE_INDEX] * len(source_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "length": len(input_ids),
        "source_length": original_source_length,
        "target_length": original_target_length,
        "supervised_tokens": supervised,
        "was_truncated": full_length > max_length,
        "source_was_truncated": len(source_ids) < original_source_length,
        "target_was_truncated": len(target_ids) < original_target_length,
    }


def truncate_code_tokens(
    source_ids: list[int], target_ids: list[int], max_length: int, min_source_tokens: int = 64
) -> tuple[list[int], list[int]]:
    if len(source_ids) + len(target_ids) <= max_length:
        return source_ids, target_ids
    source_floor = min(len(source_ids), min_source_tokens)
    target_budget = min(len(target_ids), max_length - source_floor)
    source_budget = min(len(source_ids), max_length - target_budget)
    target_budget = min(len(target_ids), max_length - source_budget)
    if len(source_ids) > source_budget:
        head = (source_budget + 1) // 2
        tail = source_budget - head
        source_ids = source_ids[:head] + (source_ids[-tail:] if tail else [])
    if len(target_ids) > target_budget:
        target_ids = target_ids[: target_budget - 1] + [target_ids[-1]]
    return source_ids, target_ids


def tokenize_code_batch(
    batch: dict[str, list[Any]], tokenizer, max_length: int
) -> dict[str, list[Any]]:
    if tokenizer.chat_template is None:
        raise ValueError("Code preprocessing requires the model tokenizer's chat_template")
    conversations = [
        [{"role": "user", "content": str(instruction)}]
        for instruction in batch["instruction"]
    ]
    rendered = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )
    source_batch = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if not isinstance(eot_id, int) or eot_id < 0 or eot_id >= len(tokenizer):
        raise ValueError("Code preprocessing requires a valid <|eot_id|> token")
    responses = [str(value).replace("<|eot_id|>", "").rstrip() for value in batch["response"]]
    target_batch = tokenizer(
        responses,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    output: dict[str, list[Any]] = {
        key: []
        for key in (
            "input_ids",
            "labels",
            "length",
            "source_length",
            "target_length",
            "supervised_tokens",
            "was_truncated",
            "source_was_truncated",
            "target_was_truncated",
        )
    }
    for source_ids, response_ids in zip(source_batch, target_batch):
        target_ids = response_ids + [eot_id]
        original_source_length = len(source_ids)
        original_target_length = len(target_ids)
        full_length = original_source_length + original_target_length
        source_ids, target_ids = truncate_code_tokens(source_ids, target_ids, max_length)
        input_ids = source_ids + target_ids
        supervised = len(target_ids)
        values = {
            "input_ids": input_ids,
            "labels": [IGNORE_INDEX] * len(source_ids) + target_ids,
            "length": len(input_ids),
            "source_length": original_source_length,
            "target_length": original_target_length,
            "supervised_tokens": supervised,
            "was_truncated": full_length > max_length,
            "source_was_truncated": len(source_ids) < original_source_length,
            "target_was_truncated": len(target_ids) < original_target_length,
        }
        for key, value in values.items():
            output[key].append(value)
    return output


def dataset_from_rows(
    rows: list[dict[str, Any]], tokenizer, max_length: int, kind: str, num_proc: int
) -> Dataset:
    raw = Dataset.from_list(rows)
    def process(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if kind == "code":
            return tokenize_code_batch(batch, tokenizer, max_length)
        keys = list(batch)
        output: dict[str, list[Any]] = {}
        for index in range(len(batch[keys[0]])):
            row = {key: batch[key][index] for key in keys}
            # Commonsense follows the plain tokenizer contract: concatenate the
            # Alpaca prompt and response, then right-truncate the resulting
            # sequence to max_length.  Do not reserve a response budget or
            # otherwise special-case long prompts.  Other Alpaca tasks retain
            # their existing behavior.
            tokenized = tokenize_alpaca_row(
                row, tokenizer, max_length, preserve_response=False
            )
            for key, value in tokenized.items():
                output.setdefault(key, []).append(value)
        return output

    return raw.map(
        process,
        batched=True,
        batch_size=64 if kind == "code" else 256,
        num_proc=num_proc,
        remove_columns=raw.column_names,
        desc=f"Tokenizing {kind}",
    )


def tokenization_stats(dataset: Dataset) -> dict[str, int]:
    truncated = sum(bool(value) for value in dataset["was_truncated"])
    zero_supervision = sum(value == 0 for value in dataset["supervised_tokens"])
    if zero_supervision:
        warnings.warn(
            f"{zero_supervision} rows contain no supervised response token after truncation; "
            "retaining them to match concatenated right-truncation preprocessing",
            stacklevel=2,
        )
    stats = {
        "rows": len(dataset),
        "truncated_rows": truncated,
        "zero_supervision_rows": zero_supervision,
        "max_observed_length": max(dataset["length"], default=0),
        "supervised_tokens": sum(dataset["supervised_tokens"]),
    }
    if "source_was_truncated" in dataset.column_names:
        stats["source_truncated_rows"] = sum(dataset["source_was_truncated"])
    if "target_was_truncated" in dataset.column_names:
        stats["target_truncated_rows"] = sum(dataset["target_was_truncated"])
    return stats


def training_columns(dataset: Dataset) -> Dataset:
    keep = {"input_ids", "labels", "length"}
    return dataset.remove_columns([name for name in dataset.column_names if name not in keep])


def base_manifest(spec: PrepareSpec, tokenizer, prompt_contract: dict[str, Any]) -> dict[str, Any]:
    tokenizer_files = {}
    tokenizer_dir = spec.output_dir / "tokenizer"
    for path in sorted(tokenizer_dir.iterdir()):
        if path.is_file():
            tokenizer_files[path.name] = sha256_file(path)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": spec.task,
        "source": {
            "path": str(spec.source.resolve()),
            "sha256": sha256_file(spec.source),
            "expected_rows": spec.expected_rows,
        },
        "tokenizer": {
            "source": spec.model_name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "files_sha256": tokenizer_files,
            "chat_template_sha256": (
                hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
                if tokenizer.chat_template
                else None
            ),
        },
        "preprocessing": {
            "max_length": spec.max_length,
            "num_proc": spec.num_proc,
            "truncation": "right_on_concatenated_prompt_and_response",
            "padding": "dynamic_at_training_time",
            "label_mask": "prompt_tokens_are_-100; response_tokens_only",
            "packing": False,
            "prompt_contract": prompt_contract,
        },
        "seed": spec.seed,
        "software": {
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
        },
    }


def deterministic_math_split(row_count: int, seed: int, validation_size: int = 500):
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    validation = sorted(indices[:validation_size])
    validation_set = set(validation)
    train = [index for index in range(row_count) if index not in validation_set]
    return train, validation


def validate_fields(rows: Iterable[dict[str, Any]], required: set[str]) -> None:
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
