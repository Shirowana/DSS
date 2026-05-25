"""Fine-tune DSS on the Math10K protocol.

The training loop is intentionally shared with finetune_commonsense.py; this
wrapper only swaps project defaults and supported model names for math runs.
"""

from __future__ import annotations

from pathlib import Path

import finetune_commonsense as commonsense


REMOTE_PROJECT_ROOT = Path("/root/code/DSS")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")


def build_run_name(args, timestamp: str) -> str:
    return (
        f"math_{args.model_name}_dss_nobasis_"
        f"nf{args.n_frequency}_cand{args.candidate_size}_gs{args.grad_store_steps}_{timestamp}"
    )


def main() -> None:
    commonsense.REMOTE_OUTPUT_ROOT = REMOTE_PROJECT_ROOT / "output_math"
    commonsense.MODEL_MAP = {
        "Llama3-8B": str(REMOTE_MODEL_ROOT / "Meta-Llama-3-8B"),
        "Qwen2.5-7B": str(REMOTE_MODEL_ROOT / "Qwen2.5-7B"),
    }
    commonsense.build_run_name = build_run_name
    commonsense.main()


if __name__ == "__main__":
    main()
