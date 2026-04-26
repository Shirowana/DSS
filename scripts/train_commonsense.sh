#!/bin/bash
# Fine-tune Llama3-8B on commonsense with DSS in the remote container.
#
# Usage:
#   bash scripts/train_commonsense.sh
#
# The script has two explicit phases:
#   1. Fit shared basis if SHARED_BASIS_PATH does not exist, or if FIT_BASIS=1.
#   2. Fine-tune DSS on the commonsense dataset.

set -euo pipefail

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/data/home/7250091/date/DSS"}
REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC:-"/data/home/7250091/date/quest20260313/peft/src"}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-"/data/home/7250091/date/datasets"}
REMOTE_MODEL_ROOT=${REMOTE_MODEL_ROOT:-"/data/home/7250091/date/hf_cache_models/models"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_commonsense"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REMOTE_PROJECT_ROOT}/output"}
RESULT_ROOT=${RESULT_ROOT:-"${REMOTE_PROJECT_ROOT}/results_commonsense"}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

MODEL_NAME=${MODEL_NAME:-"Llama3-8B"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"${REMOTE_MODEL_ROOT}"}
MODEL_PATH=${MODEL_PATH:-"${REMOTE_MODEL_ROOT}/${MODEL_NAME}"}

DATA_DIR=${DATA_DIR:-"${REMOTE_DATA_ROOT}/commonsense_new"}
MAX_LENGTH=${MAX_LENGTH:-256}
DATASET_PATH=${DATASET_PATH:-"${DATA_DIR}/train_all_${MAX_LENGTH}_OnlyOutput_${MODEL_NAME}"}

TARGET_MODULES=${TARGET_MODULES:-"qkvud"}
N_FREQUENCY=${N_FREQUENCY:-180000}
CANDIDATE_SIZE=${CANDIDATE_SIZE:-10000}
GRAD_STORE_STEPS=${GRAD_STORE_STEPS:-10}
LOW=${LOW:-200}
UP=${UP:-2000}
RATIO=${RATIO:-0.10}

STAGE2_ENABLED=${STAGE2_ENABLED:-1}
STEADY_STAGE_RATIO=${STEADY_STAGE_RATIO:-0.10}
UPDATE_INTERVAL=${UPDATE_INTERVAL:-100}
UPDATE_COUNTS=${UPDATE_COUNTS:-5000}

BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:--1}
PRECISION=${PRECISION:-"bf16"}
LR=${LR:-"1e-4"}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-5000}
NUM_WORKERS=${NUM_WORKERS:-0}

SHARED_BASIS_PATH=${SHARED_BASIS_PATH:-"${REMOTE_PROJECT_ROOT}/basis/llama3_8b_dss_basis.pt"}
BASIS_OFFSET=${BASIS_OFFSET:-0}
BASIS_LR=${BASIS_LR:-0.01}
BASIS_ITERS=${BASIS_ITERS:-1000}
BASIS_DEVICE=${BASIS_DEVICE:-"cuda"}
MAX_LAYERS_PER_GROUP=${MAX_LAYERS_PER_GROUP:-0}
FIT_BASIS=${FIT_BASIS:-"auto"}

RUN_NAME=${RUN_NAME:-"commonsense_${MODEL_NAME}_dss_nf${N_FREQUENCY}_cand${CANDIDATE_SIZE}_gs${GRAD_STORE_STEPS}_${TIMESTAMP}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${RUN_NAME}"}
LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${TIMESTAMP}.log"}
SHARED_BASIS_SCRIPT=${SHARED_BASIS_SCRIPT:-"${REMOTE_PEFT_SRC}/peft/tuners/dss/shared_basis.py"}

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

export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT=${WANDB_PROJECT:-"dss_commonsense"}
export WANDB_NAME=${WANDB_NAME:-"${RUN_NAME}"}

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${OUTPUT_DIR}"
cd "${REMOTE_PROJECT_ROOT}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "SHARED_BASIS_PATH=${SHARED_BASIS_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

if [[ "${FIT_BASIS}" == "1" || ! -f "${SHARED_BASIS_PATH}" ]]; then
    if [[ ! -f "${SHARED_BASIS_SCRIPT}" ]]; then
        echo "Missing shared-basis script: ${SHARED_BASIS_SCRIPT}" | tee "${LOG_FILE}"
        echo "Please check /data/home/7250091/date/quest20260313/peft/src/peft/tuners/dss/shared_basis.py, or set SHARED_BASIS_SCRIPT." | tee -a "${LOG_FILE}"
        exit 1
    fi
    {
        echo "========== STAGE 1/2: shared basis =========="
        echo "[basis] start: $(date)"
    } | tee "${LOG_FILE}"
    basis_start=$(date +%s)
    basis_cmd=(
        python "${SHARED_BASIS_SCRIPT}"
        --model_name "${MODEL_NAME}"
        --model_path "${MODEL_PATH}"
        --model_cache_dir "${MODEL_CACHE_DIR}"
        --output_path "${SHARED_BASIS_PATH}"
        --target_modules "${TARGET_MODULES}"
        --basis_offset "${BASIS_OFFSET}"
        --basis_lr "${BASIS_LR}"
        --basis_iters "${BASIS_ITERS}"
        --basis_device "${BASIS_DEVICE}"
        --max_layers_per_group "${MAX_LAYERS_PER_GROUP}"
    )
    "${basis_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
    basis_end=$(date +%s)
    {
        echo "[basis] done: $(date), elapsed=$(format_elapsed $((basis_end - basis_start)))"
        echo
    } | tee -a "${LOG_FILE}"
else
    {
        echo "========== STAGE 1/2: shared basis =========="
        echo "[basis] skip existing file: ${SHARED_BASIS_PATH}"
        echo
    } | tee "${LOG_FILE}"
fi

cmd=(
    python finetune_commonsense.py
    --model_name "${MODEL_NAME}"
    --model_path "${MODEL_PATH}"
    --model_cache_dir "${MODEL_CACHE_DIR}"
    --data_dir "${DATA_DIR}"
    --dataset_path "${DATASET_PATH}"
    --max_length "${MAX_LENGTH}"
    --shared_basis_path "${SHARED_BASIS_PATH}"
    --target_modules "${TARGET_MODULES}"
    --n_frequency "${N_FREQUENCY}"
    --candidate_size "${CANDIDATE_SIZE}"
    --grad_store_steps "${GRAD_STORE_STEPS}"
    --low "${LOW}"
    --up "${UP}"
    --ratio "${RATIO}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --num_epochs "${NUM_EPOCHS}"
    --max_steps "${MAX_STEPS}"
    --precision "${PRECISION}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_steps "${WARMUP_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --num_workers "${NUM_WORKERS}"
    --output_dir "${OUTPUT_DIR}"
    --run_name "${RUN_NAME}"
)

if [[ "${STAGE2_ENABLED}" == "1" ]]; then
    cmd+=(
        --stage2_enabled
        --steady_stage_ratio "${STEADY_STAGE_RATIO}"
        --update_interval "${UPDATE_INTERVAL}"
        --update_counts "${UPDATE_COUNTS}"
    )
fi

{
    echo "========== STAGE 2/2: DSS commonsense training =========="
    echo "[train] start: $(date)"
} | tee -a "${LOG_FILE}"
train_start=$(date +%s)
"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
train_end=$(date +%s)
echo "[train] done: $(date), elapsed=$(format_elapsed $((train_end - train_start)))" | tee -a "${LOG_FILE}"
