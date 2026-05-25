"""
Normalize math reasoning evaluation datasets into raw_eval/<dataset>/test.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REMOTE_DATA_ROOT = Path("/root/datasets")

DEFAULT_SOURCES = {
    "gsm8k": REMOTE_DATA_ROOT / "evaluate" / "gsm8k" / "test.json",
    "svamp": REMOTE_DATA_ROOT / "evaluate" / "SVAMP" / "test.json",
    "aqua": REMOTE_DATA_ROOT / "evaluate" / "AQuA" / "test.json",
    "mawps": REMOTE_DATA_ROOT / "evaluate" / "mawps" / "test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized raw eval files for math reasoning.")
    parser.add_argument("--output_root", default=str(REMOTE_DATA_ROOT / "math_reasoning"))
    parser.add_argument("--gsm8k_path", default=str(DEFAULT_SOURCES["gsm8k"]))
    parser.add_argument("--svamp_path", default=str(DEFAULT_SOURCES["svamp"]))
    parser.add_argument("--aqua_path", default=str(DEFAULT_SOURCES["aqua"]))
    parser.add_argument("--mawps_path", default=str(DEFAULT_SOURCES["mawps"]))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload)!r}")
    return payload


def normalize_row(dataset_name: str, row: dict, index: int) -> dict:
    instruction = str(row.get("instruction") or row.get("question") or "").strip()
    input_text = str(row.get("input") or "").strip()
    answer = str(row.get("answer") or row.get("label") or "").strip()
    output = str(row.get("output") or "").strip()
    normalized = {
        "id": row.get("id", f"{dataset_name}-{index:05d}"),
        "dataset": dataset_name,
        "instruction": instruction,
        "input": input_text,
        "answer": answer,
    }
    if output:
        normalized["reference_output"] = output
    if dataset_name == "aqua":
        normalized["answer_type"] = "choice"
        normalized["choices"] = row.get("choices", [])
    else:
        normalized["answer_type"] = "number"
    return normalized


def prepare_one(dataset_name: str, source_path: Path, output_root: Path) -> None:
    rows = load_json_list(source_path)
    raw_dir = output_root / "raw_eval" / dataset_name
    metadata_dir = output_root / "metadata" / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    original_copy = raw_dir / source_path.name
    if source_path.resolve() != original_copy.resolve():
        shutil.copy2(source_path, original_copy)

    normalized = [normalize_row(dataset_name, row, i) for i, row in enumerate(rows)]
    normalized_file = raw_dir / "test.json"
    with normalized_file.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2)

    metadata = {
        "dataset_name": dataset_name,
        "source": str(source_path),
        "original_file": str(original_copy),
        "normalized_file": str(normalized_file),
        "num_examples": len(normalized),
        "sha256": sha256_file(source_path),
        "notes": "Original local file is copied unchanged; test.json is normalized for evaluate_math_reasoning.py.",
    }
    with (metadata_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"[prepare-eval] {dataset_name}: {len(normalized)} examples -> {normalized_file}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    sources = {
        "gsm8k": Path(args.gsm8k_path),
        "svamp": Path(args.svamp_path),
        "aqua": Path(args.aqua_path),
        "mawps": Path(args.mawps_path),
    }
    for dataset_name, source_path in sources.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file for {dataset_name}: {source_path}")
        prepare_one(dataset_name, source_path, output_root)


if __name__ == "__main__":
    main()
