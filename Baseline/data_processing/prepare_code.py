from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.common import (
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


DEFAULT_SOURCE = Path(
    "/data/home/7250091/date/datasets/Magicoder-Evol-Instruct-110K/"
    "data-evol_instruct-decontaminated.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen Magicoder training tokens.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = PrepareSpec(
        task="code",
        source=args.source,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_length=3072,
        expected_rows=111_183,
        seed=args.seed,
        num_proc=args.num_proc,
        overwrite=args.overwrite,
        trust_remote_code=args.trust_remote_code,
    )
    rows = load_json_or_jsonl(spec.source)
    if len(rows) != spec.expected_rows:
        raise ValueError(f"Expected {spec.expected_rows} rows, found {len(rows)}")
    validate_fields(rows, {"instruction", "response"})
    prepare_output_dir(spec.output_dir, spec.overwrite)
    tokenizer = load_tokenizer(spec)
    tokenizer.save_pretrained(spec.output_dir / "tokenizer")
    dataset = dataset_from_rows(rows, tokenizer, 3072, "code", spec.num_proc)
    stats = tokenization_stats(dataset)
    training_columns(dataset).save_to_disk(spec.output_dir / "train")
    manifest = base_manifest(
        spec,
        tokenizer,
        {
            "type": "llama3_chat_template",
            "messages": [{"role": "user", "content": "{instruction}"}],
            "system_message": None,
            "add_generation_prompt": True,
            "response_terminator": "<|eot_id|>",
        },
    )
    manifest["preprocessing"]["truncation"] = (
        "response_priority; keep_full_response_when_possible; keep_at_least_64_prompt_tokens; "
        "middle_truncate_prompt_to_preserve_chat_prefix_and_assistant_header; "
        "right_truncate_response_body_only_if_it_exceeds_remaining_budget; preserve_eot"
    )
    manifest["splits"] = {"train": stats, "validation": None}
    manifest["checkpoint_selection"] = "final_state_after_optimizer_step_869"
    atomic_write_json(spec.output_dir / "manifest.json", manifest)
    print(f"Prepared {len(dataset):,} Magicoder rows at {spec.output_dir}")


if __name__ == "__main__":
    main()
