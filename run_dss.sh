#!/bin/bash
# DSS no-basis, stage1-only training entrypoint for the current ablation round.
#
# Usage:
#   bash /root/code/DSS/run_dss.sh
#
# Notes:
# - This script reuses the same remote paths and environment layout as
#   `scripts/train_commonsense.sh`.
# - It runs the refactored no-basis DSS path with the official Hugging Face
#   Trainer and a fixed held-out validation split.

set -euo pipefail

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/root/code/DSS"}
REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC:-"/root/code/quest/peft/src"}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-"/root/datasets"}
REMOTE_MODEL_ROOT=${REMOTE_MODEL_ROOT:-"/root/hf_cache_models/models"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_commonsense"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REMOTE_PROJECT_ROOT}/output"}
RESULT_ROOT=${RESULT_ROOT:-"${REMOTE_PROJECT_ROOT}/results_commonsense"}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-"${REMOTE_PROJECT_ROOT}/experiments/commonsense"}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

MODEL_NAME=${MODEL_NAME:-"Llama3-8B"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"${REMOTE_MODEL_ROOT}"}
if [[ -z "${MODEL_PATH:-}" ]]; then
    case "${MODEL_NAME}" in
        "Llama3-8B")
            MODEL_PATH="${REMOTE_MODEL_ROOT}/Meta-Llama-3-8B"
            ;;
        "Llama2-7B")
            MODEL_PATH="${REMOTE_MODEL_ROOT}/Llama-2-7b-hf"
            ;;
        *)
            echo "Unsupported MODEL_NAME=${MODEL_NAME}. Use Llama3-8B or Llama2-7B." >&2
            exit 1
            ;;
    esac
fi

DATA_DIR=${DATA_DIR:-"${REMOTE_DATA_ROOT}/commonsense_new"}
MAX_LENGTH=${MAX_LENGTH:-256}
DATASET_PATH=${DATASET_PATH:-"${DATA_DIR}/train_all_${MAX_LENGTH}_OnlyOutput_${MODEL_NAME}"}
VAL_SET_SIZE=${VAL_SET_SIZE:-120}

TARGET_MODULES=${TARGET_MODULES:-"qkvud"}
N_FREQUENCY=${N_FREQUENCY:-180000}
CANDIDATE_SIZE=${CANDIDATE_SIZE:-30000}
GRAD_STORE_STEPS=${GRAD_STORE_STEPS:-3}
LOW=${LOW:-500}
UP=${UP:-4000}
RATIO=${RATIO:-0.10}
THRESHOLD_MODE=${THRESHOLD_MODE:-"oracle"}
DSS_DROPOUT=${DSS_DROPOUT:-0.05}
QUANTILE_LR=${QUANTILE_LR:-0.01}
QUANTILE_ALPHA=${QUANTILE_ALPHA:-0.0}

BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:--1}
PRECISION=${PRECISION:-"bf16"}
LR=${LR:-"8e-5"}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-10000}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
REPORT_TO=${REPORT_TO:-"wandb"}
SEED=${SEED:-42}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED:-1}
EXPERIMENT_MD=${EXPERIMENT_MD:-}

RUN_NAME=${RUN_NAME:-"commonsense_${MODEL_NAME}_dss_nobasis_nf${N_FREQUENCY}_cand${CANDIDATE_SIZE}_gs${GRAD_STORE_STEPS}_${TIMESTAMP}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${RUN_NAME}"}
LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${TIMESTAMP}_dss_nobasis.log"}

format_elapsed() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    if [[ ${hours} -gt 0 ]]; then
        printf "%dh %dm %ds" "${hours}" "${minutes}" "${seconds}"
    elif [[ ${minutes} -gt 0 ]]; then
        printf "%dm %ds" "${minutes}" "${seconds}"
    else
        printf "%ds" "${seconds}"
    fi
}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT=${WANDB_PROJECT:-"dss_commonsense"}
export WANDB_NAME=${WANDB_NAME:-"${RUN_NAME}"}

if ! python - <<'PY' >/dev/null 2>&1
import peft  # noqa: F401
PY
then
    echo "Failed to import local peft after setting PYTHONPATH=${PYTHONPATH}" >&2
    echo "Please check REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC} and the local PEFT source tree." >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${OUTPUT_DIR}" "${EXPERIMENT_ROOT}"
cd "${REMOTE_PROJECT_ROOT}"

{
    echo "========== DSS NO-BASIS RUN =========="
    echo "[config] start: $(date)"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "DATASET_PATH=${DATASET_PATH}"
    echo "MAX_LENGTH=${MAX_LENGTH}"
    echo "VAL_SET_SIZE=${VAL_SET_SIZE}"
    echo "TARGET_MODULES=${TARGET_MODULES}"
    echo "N_FREQUENCY=${N_FREQUENCY}"
    echo "CANDIDATE_SIZE=${CANDIDATE_SIZE}"
    echo "GRAD_STORE_STEPS=${GRAD_STORE_STEPS}"
    echo "LOW=${LOW}"
    echo "UP=${UP}"
    echo "RATIO=${RATIO}"
    echo "THRESHOLD_MODE=${THRESHOLD_MODE}"
    echo "DSS_DROPOUT=${DSS_DROPOUT}"
    echo "QUANTILE_LR=${QUANTILE_LR}"
    echo "QUANTILE_ALPHA=${QUANTILE_ALPHA}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "MAX_STEPS=${MAX_STEPS}"
    echo "PRECISION=${PRECISION}"
    echo "LR=${LR}"
    echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
    echo "WARMUP_STEPS=${WARMUP_STEPS}"
    echo "EVAL_STEPS=${EVAL_STEPS}"
    echo "SAVE_STEPS=${SAVE_STEPS}"
    echo "LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END}"
    echo "REPORT_TO=${REPORT_TO}"
    echo "SEED=${SEED}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED}"
    if [[ -n "${EXPERIMENT_MD}" ]]; then
        echo "EXPERIMENT_MD=${EXPERIMENT_MD}"
    fi
    if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
        echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
    fi
    echo
} | tee "${LOG_FILE}"

cmd=(
    python finetune_commonsense.py
    --model_name "${MODEL_NAME}"
    --model_path "${MODEL_PATH}"
    --model_cache_dir "${MODEL_CACHE_DIR}"
    --data_dir "${DATA_DIR}"
    --dataset_path "${DATASET_PATH}"
    --max_length "${MAX_LENGTH}"
    --val_set_size "${VAL_SET_SIZE}"
    --target_modules "${TARGET_MODULES}"
    --n_frequency "${N_FREQUENCY}"
    --candidate_size "${CANDIDATE_SIZE}"
    --grad_store_steps "${GRAD_STORE_STEPS}"
    --low "${LOW}"
    --up "${UP}"
    --ratio "${RATIO}"
    --threshold_mode "${THRESHOLD_MODE}"
    --dropout "${DSS_DROPOUT}"
    --quantile_lr "${QUANTILE_LR}"
    --quantile_alpha "${QUANTILE_ALPHA}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --num_epochs "${NUM_EPOCHS}"
    --max_steps "${MAX_STEPS}"
    --precision "${PRECISION}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_steps "${WARMUP_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --num_workers "${NUM_WORKERS}"
    --report_to "${REPORT_TO}"
    --output_dir "${OUTPUT_DIR}"
    --run_name "${RUN_NAME}"
    --seed "${SEED}"
)

if [[ "${LOAD_BEST_MODEL_AT_END}" == "1" ]]; then
    cmd+=(--load_best_model_at_end)
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

{
    echo "========== DSS TRAINING =========="
    echo "[train] start: $(date)"
} | tee -a "${LOG_FILE}"
train_start=$(date +%s)

"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"

train_end=$(date +%s)
elapsed=$((train_end - train_start))
{
    echo
    echo "========== DSS TRAINING DONE =========="
    echo "[train] end: $(date)"
    echo "elapsed=$(format_elapsed "${elapsed}")"
    echo "output_dir=${OUTPUT_DIR}"
} | tee -a "${LOG_FILE}"
