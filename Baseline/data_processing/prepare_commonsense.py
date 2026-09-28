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
    load_json_or_jsonl,
    load_tokenizer,
    prepare_output_dir,
    tokenization_stats,
    training_columns,
    validate_fields,
)


DEFAULT_SOURCE = Path("/data/home/7250091/date/datasets/ft-training_set/commonsense_170k.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen Commonsense170K tokens.")
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
        task="commonsense",
        source=args.source,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_length=256,
        expected_rows=170_420,
        seed=args.seed,
        num_proc=args.num_proc,
        overwrite=args.overwrite,
        trust_remote_code=args.trust_remote_code,
    )
    rows = load_json_or_jsonl(spec.source)
    if len(rows) != spec.expected_rows:
        raise ValueError(f"Expected {spec.expected_rows} rows, found {len(rows)}")
    validate_fields(rows, {"instruction", "output"})
    prepare_output_dir(spec.output_dir, spec.overwrite)
    tokenizer = load_tokenizer(spec)
    tokenizer.save_pretrained(spec.output_dir / "tokenizer")
    dataset = dataset_from_rows(rows, tokenizer, spec.max_length, "commonsense", spec.num_proc)
    stats = tokenization_stats(dataset)
    training_columns(dataset).save_to_disk(spec.output_dir / "train")
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
    manifest["splits"] = {"train": stats, "validation": None}
    manifest["preprocessing"]["truncation"] = "plain_right_truncate_full_sequence"
    manifest["checkpoint_selection"] = "final_state"
    atomic_write_json(spec.output_dir / "manifest.json", manifest)
    print(f"Prepared {len(dataset):,} Commonsense rows at {spec.output_dir}")


if __name__ == "__main__":
    main()
