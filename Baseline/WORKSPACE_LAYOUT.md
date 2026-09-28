# Baseline workspace layout

This workspace shares task plumbing across S2FT and SIFT. No StreamSelect,
DSS, or QUEST training source is used.

```text
baseline/
|-- S2FT/                       # Upstream S2FT method implementation
|-- SIFT/                       # Upstream SIFT method implementation
|-- finetune.py                 # Shared trainer; Commonsense default
|-- finetune_math.py            # Thin Math training entrypoint
|-- finetune_code.py            # Thin Code training entrypoint
|-- evaluation_core.py          # Shared model loading and generation
|-- eval_commonsense.py         # Commonsense prompt/parser/metrics
|-- eval_math.py                # Math prompt/parser/metrics
|-- eval_code.py                # EvalPlus generation and sandbox scoring
|-- data_processing/            # Frozen, tokenizer-specific data preparation
|-- training_methods/           # Thin adapters around upstream method implementations
|-- experiments/                # Cross-method CSV/table aggregation only
|-- scripts/                    # Optional launchers (not method-owned)
|-- third_party/                # Pinned evaluator helpers
|-- tools/                      # Pinned external evaluation runtimes
|-- logs/                       # Console logs and operational diagnostics
|-- outputs/                    # Training checkpoints and exported model artifacts
`-- results/                    # Evaluation samples, scores, and summaries
```

## Run identity

Every official run must use one immutable `run_id` in all three artifact trees:

```text
<method>/<task>/<model>/<run_id>
```

For example:

```text
logs/S2FT/code/Llama3-8B-Instruct/20260920_210000_seed42/
outputs/S2FT/code/Llama3-8B-Instruct/20260920_210000_seed42/
results/S2FT/code/Llama3-8B-Instruct/20260920_210000_seed42/
```

Training code must never write reportable evaluation results under `outputs`.
Evaluation code must never modify the exported model under `outputs`.

## Method directories

The method repositories retain their public sparse-layer implementations and
upstream provenance. Shared preprocessing and training orchestration live at
the workspace root. A frozen preprocessing artifact is reused unchanged by
both methods; no method-specific split is permitted.

## Evaluation

The three public evaluators consume `outputs/.../final_model`, which is a dense
Hugging Face artifact for both S2FT and SIFT. They do not import either method's
training integration. Model loading and generation are shared; prompt, parser,
and scoring rules remain task-specific.

`experiments/` is not a run directory. It reads immutable summaries below
`results/` and builds cross-method CSV/table views.
