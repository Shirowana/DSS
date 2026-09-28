# S2FT/SIFT baseline runbook

Run commands from `/data/home/7250091/baseline` with:

```bash
PYTHON=/data/home/7250091/date/conda_env/quest/bin/python
BASE_MODEL=/data/home/7250091/date/hf_cache_models/models/Meta-Llama-3-8B
INSTRUCT_MODEL=/data/home/7250091/date/hf_cache_models/models/Meta-Llama-3-8B-Instruct
```

## Prepare data once per tokenizer

The prepared Arrow datasets are method-independent. Reuse the same directory
for S2FT and SIFT runs using the same tokenizer.

```bash
$PYTHON data_processing/prepare_commonsense.py \
  --model_name_or_path "$BASE_MODEL" \
  --output_dir /data/home/7250091/date/datasets/commonsense_new/Llama3-8B

$PYTHON data_processing/prepare_math.py \
  --model_name_or_path "$BASE_MODEL" \
  --output_dir /data/home/7250091/date/datasets/math_new/Llama3-8B

$PYTHON data_processing/prepare_code.py \
  --model_name_or_path "$INSTRUCT_MODEL" \
  --output_dir /data/home/7250091/date/datasets/code_new/Llama3-8B-Instruct
```

Each artifact includes `manifest.json`, a frozen tokenizer, tokenized splits,
source hashes, truncation statistics, and the exact Math split indices.
Code preprocessing preserves the full response whenever possible. For an
overlong prompt it keeps both chat boundaries and removes prompt tokens from
the middle, so every one of the 111,183 rows retains supervised response
tokens within the 3,072-token limit.

## Train

The protocol batch sizes assume two GPU processes. Use the same entrypoint for
either method and change only `--method` plus method-specific arguments.

```bash
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=2 finetune.py \
  --method s2ft \
  --model_name_or_path "$BASE_MODEL" \
  --prepared_data /data/home/7250091/date/datasets/commonsense_new/Llama3-8B

$PYTHON -m torch.distributed.run --standalone --nproc_per_node=2 finetune_math.py \
  --method sift \
  --model_name_or_path "$BASE_MODEL" \
  --prepared_data /data/home/7250091/date/datasets/math_new/Llama3-8B \
  --sift_rate 0.01

$PYTHON -m torch.distributed.run --standalone --nproc_per_node=2 finetune_code.py \
  --method s2ft \
  --model_name_or_path "$INSTRUCT_MODEL" \
  --prepared_data /data/home/7250091/date/datasets/code_new/Llama3-8B-Instruct
```

S2FT defaults to the authors' model-specific S2FT-R allocation: Llama2 uses
`o=0.052, d=0.02`; Llama3 uses `d=0.03`. The optional
`--s2_{v,o,u,d}_ratio` arguments are only for explicitly labeled ablations.

Unless `--output_dir` is supplied, a run writes:

```text
logs/<run_id>/
outputs/<method>/<task>/<model>/<run_id>/final_model/
```

The run log directory keeps `train.log`, `eval_gpu0.log`, `eval_gpu1.log`,
`trainer_log_history.json`, and `run_manifest.json` together. Use
`scripts/eval_distributed.py` to split complete datasets between two GPUs and
append the final task summary to `train.log`.

Both methods export a full dense Hugging Face model. Math additionally retains
small method-state candidates at the six eligible steps and records the
validation-loss selection in `checkpoint_selection.json`.

## Evaluate

Evaluation consumes the exported `final_model/` and is method-independent.

```bash
$PYTHON eval_commonsense.py \
  --model_path outputs/s2ft/commonsense/Meta-Llama-3-8B/<run_id>/final_model \
  --result_dir results/s2ft/commonsense/Meta-Llama-3-8B/<run_id>

$PYTHON eval_math.py \
  --model_path outputs/sift/math/Meta-Llama-3-8B/<run_id>/final_model \
  --result_dir results/sift/math/Meta-Llama-3-8B/<run_id>

$PYTHON eval_code.py \
  --model_path outputs/s2ft/code/Meta-Llama-3-8B-Instruct/<run_id>/final_model \
  --result_dir results/s2ft/code/Meta-Llama-3-8B-Instruct/<run_id> \
  --verify_batch_equivalence 16
```

Code generation uses batch 16. `--verify_batch_equivalence 16` checks the
first 16 outputs token-for-token against batch size 1 before the full run.
Generated programs are sanitized and executed only by the pinned EvalPlus
0.3.1 bubblewrap sandbox.

## Aggregate

```bash
$PYTHON experiments/collect_results.py
```

Only complete `results/.../summary.json` files enter
`experiments/results.csv`; partial runs are ignored.
