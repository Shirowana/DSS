# LLM-Adapters Commonsense Protocol

This directory contains a compatibility branch for training DSS and SHiRA with
the data preparation and evaluation conventions used by LLM-Adapters.

## Entrypoints

- Training and eight-task evaluation: `scripts/train_eval_llm_adapters.sh`
- Training implementation: `llm_adapters/finetune_commonsense.py`
- Evaluation implementation: `llm_adapters/commonsense_evaluate.py`

The root-level `llm_adapters_*.py` and `train_eval_llm_adapters.sh` files are
compatibility wrappers for existing launch commands.

## SHiRA-SNIP

Set `ADAPTER_NAME=shira` and `SHIRA_MASK_TYPE=snip`. Before adapter injection,
the training program runs one calibration backward pass over
`SHIRA_CALIBRATION_SAMPLES` examples and selects each target layer's top
`r * (in_features + out_features)` coordinates by `abs(W * grad)`.

The selected coordinates are stored with the adapter. They remain int64 during
device moves so bf16 model weights cannot corrupt sparse indices.

## Default Llama3-8B Run

The current SHiRA-SNIP baseline uses `r=32`, `lr=1e-4`, global batch size 32,
three epochs, maximum length 256, q/k/v/up/down targets, and beam size 1 for
evaluation. This produces 56,623,104 trainable parameters (0.7002% of
Llama3-8B).
