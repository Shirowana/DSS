from __future__ import annotations

import gzip
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent


class EvaluationLogger:
    """Small, line-oriented logger shared by the evaluation entry points."""

    def __init__(self, *, task: str, worker: int | None = None, debug_first_n: int = 5) -> None:
        self.task = task
        self.worker = worker
        self.debug_first_n = debug_first_n
        self.started = time.monotonic()

    def log(self, message: str = "") -> None:
        print(message, flush=True)

    def section(self, title: str) -> None:
        self.log(f"========== {title} ==========")

    def run_info(self, generator: "Generator") -> None:
        metadata = generator.metadata()
        self.log(f"[eval] worker={self.worker}")
        self.log(f"[eval] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        for key in (
            "model_class", "tokenizer_class", "transformers_version", "torch_version",
            "torch_dtype", "attention_implementation", "batch_size", "num_beams",
            "do_sample", "max_new_tokens", "seed", "padding_side", "pad_token_id",
        ):
            self.log(f"[eval] {key}={metadata[key]}")

    def dataset_start(self, name: str, path: Path, total: int, completed: int) -> None:
        self.section(f"DATASET: {name}")
        self.log(f"[eval] dataset_file={path}")
        self.log(f"[eval] dataset_size={total}")
        self.log(f"[eval] completed_before_resume={completed}")
        self.log(f"[eval] remaining={total - completed}")

    def sample_debug(
        self,
        *,
        dataset: str,
        sample_index: int,
        gold_raw: Any,
        gold: Any,
        pred: Any,
        correct: bool,
        prompt: str,
        output: str,
        token_ids: Sequence[int],
        decoded_tokens: Sequence[str],
    ) -> None:
        if sample_index >= self.debug_first_n:
            return
        self.section("DEBUG SAMPLE")
        self.log(f"[debug] dataset={dataset}")
        self.log(f"[debug] sample_index={sample_index}")
        self.log(f"[debug] gold_raw={gold_raw!r}")
        self.log(f"[debug] gold_norm={gold!r}")
        self.log(f"[debug] pred_norm={pred!r}")
        self.log(f"[debug] correct={correct}")
        self.log("[debug] prompt:")
        self.log(prompt)
        self.log("[debug] raw_output:")
        self.log(output)
        self.log(f"[debug] generated_token_ids={list(token_ids)!r}")
        self.log(f"[debug] generated_tokens={list(decoded_tokens)!r}")
        self.log(f"[debug] generated_token_count={len(token_ids)}")

    def running_accuracy(self, index: int, total: int, correct: int) -> None:
        self.log(f"  {index}/{total} | accuracy: {correct}/{index} = {correct / index:.4f}")

    def dataset_done(self, name: str, correct: int, total: int) -> None:
        accuracy = 100.0 * correct / total if total else 0.0
        self.log(f"Final accuracy: {correct}/{total} = {accuracy / 100.0:.4f}")
        self.log(f"[summary] {name:<14} correct={correct} total={total} accuracy={accuracy / 100.0:.4f}")

    def done(self) -> None:
        self.log(f"[eval] elapsed_seconds={time.monotonic() - self.started:.1f}")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def batches(values: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@dataclass(frozen=True)
class GenerationSettings:
    batch_size: int
    num_beams: int
    max_new_tokens: int
    seed: int = 42


class Generator:
    def __init__(
        self,
        model_path: str,
        settings: GenerationSettings,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Evaluation requires a CUDA GPU")
        self.model_path = model_path
        self.settings = settings
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        torch.cuda.manual_seed_all(settings.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.num_beams = settings.num_beams
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": str(Path(self.model_path).resolve()),
            "model_class": self.model.__class__.__name__,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "torch_dtype": "bfloat16",
            "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
            "batch_size": self.settings.batch_size,
            "num_beams": self.settings.num_beams,
            "do_sample": False,
            "max_new_tokens": self.settings.max_new_tokens,
            "seed": self.settings.seed,
            "padding_side": "left",
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def _trim_generated(self, token_ids: list[int], terminators: set[int]) -> list[int]:
        result = []
        for token_id in token_ids:
            if token_id == self.tokenizer.pad_token_id and result and result[-1] in terminators:
                break
            result.append(token_id)
            if token_id in terminators:
                break
        return result

    def decode_tokens(self, token_ids: Sequence[int]) -> list[str]:
        return [self.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        add_special_tokens: bool = True,
        terminators: Sequence[int] | None = None,
    ) -> tuple[list[str], list[list[int]]]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=add_special_tokens,
        )
        input_width = encoded["input_ids"].shape[1]
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if model_limit and input_width + self.settings.max_new_tokens > model_limit:
            raise ValueError(
                f"Prompt width {input_width} + max_new_tokens {self.settings.max_new_tokens} "
                f"exceeds model context {model_limit}"
            )
        stop_ids = list(terminators or [self.tokenizer.eos_token_id])
        generated = self.model.generate(
            input_ids=encoded["input_ids"].to(self.model.device),
            attention_mask=encoded["attention_mask"].to(self.model.device),
            do_sample=False,
            num_beams=self.settings.num_beams,
            max_new_tokens=self.settings.max_new_tokens,
            eos_token_id=stop_ids if len(stop_ids) > 1 else stop_ids[0],
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        continuation = generated[:, input_width:].detach().cpu().tolist()
        normalized = [self._trim_generated(ids, set(stop_ids)) for ids in continuation]
        decoded = self.tokenizer.batch_decode(normalized, skip_special_tokens=True)
        return decoded, normalized

    def generate(
        self,
        prompts: Sequence[str],
        *,
        add_special_tokens: bool = True,
        terminators: Sequence[int] | None = None,
        progress_label: str = "Generating",
    ) -> tuple[list[str], list[list[int]]]:
        all_text: list[str] = []
        all_tokens: list[list[int]] = []
        total_batches = (len(prompts) + self.settings.batch_size - 1) // self.settings.batch_size
        for batch_index, prompt_batch in enumerate(
            batches(prompts, self.settings.batch_size), start=1
        ):
            text, tokens = self.generate_batch(
                prompt_batch,
                add_special_tokens=add_special_tokens,
                terminators=terminators,
            )
            all_text.extend(text)
            all_tokens.extend(tokens)
            print(f"{progress_label}: batch {batch_index}/{total_batches}", flush=True)
        return all_text, all_tokens

    def verify_batch_equivalence(
        self,
        prompts: Sequence[str],
        *,
        add_special_tokens: bool,
        terminators: Sequence[int],
    ) -> dict[str, Any]:
        if not prompts:
            return {"checked": 0, "equivalent": True}
        _, batched_tokens = self.generate_batch(
            prompts, add_special_tokens=add_special_tokens, terminators=terminators
        )
        single_tokens = []
        for prompt in prompts:
            _, token_ids = self.generate_batch(
                [prompt], add_special_tokens=add_special_tokens, terminators=terminators
            )
            single_tokens.append(token_ids[0])
        mismatches = [
            index
            for index, (batched, single) in enumerate(zip(batched_tokens, single_tokens))
            if batched != single
        ]
        report = {"checked": len(prompts), "equivalent": not mismatches, "mismatches": mismatches}
        if mismatches:
            raise RuntimeError(f"Batch generation differs from batch size 1 at indices {mismatches}")
        return report


def generation_manifest(generator: Generator, task: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "generation": generator.metadata(),
    }
