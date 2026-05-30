from __future__ import annotations

from pathlib import Path

import finetune_commonsense as commonsense


REMOTE_PROJECT_ROOT = Path("/root/code/DSS")
REMOTE_MODEL_ROOT = Path("/root/hf_cache_models/models")


def build_run_name(args, timestamp: str) -> str:
    if args.peft_method == "dss":
        return (
            f"math_{args.model_name}_dss_nobasis_"
            f"nf{args.n_frequency}_cand{args.candidate_size}_gs{args.grad_store_steps}_{timestamp}"
        )
    return (
        f"math_{args.model_name}_{args.peft_method}_"
        f"r{args.lora_r}_a{args.lora_alpha}_lr{args.lr}_{timestamp}"
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
