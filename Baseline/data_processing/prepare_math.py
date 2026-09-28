from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.common import (
    ALPACA_PROMPT_INPUT,
    ALPACA_PROMPT_NO_INPUT,
    PrepareSpec,
    atomic_write_json,
    base_manifest,
    dataset_from_rows,
    deterministic_math_split,
    load_json_or_jsonl,
    load_tokenizer,
    prepare_output_dir,
    sha256_json,
    tokenization_stats,
    training_columns,
    validate_fields,
)


DEFAULT_SOURCE = Path("/data/home/7250091/date/datasets/ft-training_set/math_10k.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the frozen Math10K 9419/500 split.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = PrepareSpec(
        task="math",
        source=args.source,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_length=512,
        expected_rows=9_919,
        seed=args.seed,
        num_proc=args.num_proc,
        overwrite=args.overwrite,
        trust_remote_code=args.trust_remote_code,
    )
    rows = load_json_or_jsonl(spec.source)
    if len(rows) != spec.expected_rows:
        raise ValueError(f"Expected {spec.expected_rows} rows, found {len(rows)}")
    validate_fields(rows, {"instruction", "output"})
    train_indices, validation_indices = deterministic_math_split(len(rows), spec.seed)
    split_manifest = {
        "algorithm": "python_random.Random(seed).shuffle(range(n)); first_500_validation; source_order_within_splits",
        "seed": spec.seed,
        "source_rows": len(rows),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    prepare_output_dir(spec.output_dir, spec.overwrite)
    tokenizer = load_tokenizer(spec)
    tokenizer.save_pretrained(spec.output_dir / "tokenizer")
    train = dataset_from_rows([rows[index] for index in train_indices], tokenizer, 512, "math", spec.num_proc)
    validation = dataset_from_rows(
        [rows[index] for index in validation_indices], tokenizer, 512, "math", spec.num_proc
    )
    train_stats = tokenization_stats(train)
    validation_stats = tokenization_stats(validation)
    training_columns(train).save_to_disk(spec.output_dir / "train")
    training_columns(validation).save_to_disk(spec.output_dir / "validation")
    atomic_write_json(spec.output_dir / "split_manifest.json", split_manifest)
    manifest = base_manifest(
        spec,
        tokenizer,
        {
            "type": "alpaca",
            "prompt_input": ALPACA_PROMPT_INPUT,
            "prompt_no_input": ALPACA_PROMPT_NO_INPUT,
            "response_terminator": "tokenizer.eos_token_id",
        },
    )
    manifest["source"]["provenance_note"] = (
        "Local LLM-Adapters Math10K candidate; the target paper did not publish exact sample IDs."
    )
    manifest["splits"] = {"train": train_stats, "validation": validation_stats}
    manifest["split_manifest_sha256"] = sha256_json(split_manifest)
    manifest["eligible_checkpoint_steps"] = [2651, 2828, 3004, 3181, 3358, 3534]
    manifest["eligible_step_rounding"] = "ceil(total_optimizer_steps * fraction)"
    atomic_write_json(spec.output_dir / "manifest.json", manifest)
    print(f"Prepared {len(train):,} train + {len(validation):,} validation rows at {spec.output_dir}")


if __name__ == "__main__":
    main()
