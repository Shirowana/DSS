# DSS

DSS is a sparse parameter-efficient fine-tuning method for causal language models. It keeps the base model frozen, samples candidate coordinates in selected linear layers, estimates their gradient importance online, and promotes selected coordinates to trainable sparse coefficients.

The current DSS implementation is **no-basis, stage1-only**:

- Target modules default to `q_proj`, `k_proj`, `v_proj`, `up_proj`, and `down_proj` (`qkvupdown`).
- Candidate gradients are accumulated for `grad_store_steps` steps.
- `oracle` uses the exact candidate-score quantile; `sgd` uses an online pinball-loss quantile estimator.
- After the sparse budget is filled, training continues only on the selected coefficients.

## Environment

The scripts expect the `quest` Conda environment, local PEFT source, datasets, and model cache:

```bash
conda activate quest
export PYTHONPATH=/root/code/quest/peft/src:/root/code/DSS:${PYTHONPATH}
```

Default locations can be overridden with `REMOTE_PROJECT_ROOT`, `REMOTE_PEFT_SRC`, `REMOTE_DATA_ROOT`, and `REMOTE_MODEL_ROOT`.

## Commonsense

Prepare the tokenized commonsense training data before training:

```bash
python data_processing/process_commonsense.py --help
```

Run DSS training only:

```bash
bash run_dss.sh
```

Run two-GPU DDP training followed by the eight-task evaluation. Evaluation uses one GPU by default:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 EVAL_CUDA_VISIBLE_DEVICES=0 bash train_eval.sh
```

Useful overrides are environment variables, for example:

```bash
N_FREQUENCY=360000 CANDIDATE_SIZE=30000 GRAD_STORE_STEPS=10 \
THRESHOLD_MODE=oracle SCORE_METHOD=snr LR=8e-5 \
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash train_eval.sh
```

The main training entrypoint is `finetune_commonsense.py`; the evaluator is `evaluate_commonsense.py`.

## Math Reasoning

Prepare Math10K data:

```bash
python data_processing/process_math10k.py --help
```

Run two-GPU training only or training followed by evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run_dss_math.sh
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash train_eval_math.sh
```

The math protocol and dataset details are in [math_task.md](math_task.md). The standalone evaluator is `scripts/eval_math_reasoning.sh`.

## Outputs And Records

- Commonsense checkpoints: `output/`
- Commonsense evaluation: `results_commonsense/`
- Commonsense logs and records: `logs_commonsense/`, `experiments/commonsense/`
- Math checkpoints: `output_math/`
- Math evaluation: `results_math/`
- Math logs and records: `logs_math/`, `experiments/math/`

Use `RESUME_FROM_CHECKPOINT=/path/to/checkpoint` to resume an adapter checkpoint. DSS restores sparse adapter state, but the training optimizer, scheduler, and in-progress candidate statistics are not guaranteed to reproduce an uninterrupted run exactly; use resumed runs as separate experiments.

## DSS Implementation

The authoritative DSS tuner is maintained in the local PEFT checkout:

- `/root/code/quest/peft/src/peft/tuners/dss/config.py`
- `/root/code/quest/peft/src/peft/tuners/dss/model.py`
- `/root/code/quest/peft/src/peft/tuners/dss/layer.py`
- `/root/code/quest/peft/src/peft/tuners/dss/scoring.py`
