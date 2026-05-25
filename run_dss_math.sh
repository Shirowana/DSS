#!/bin/bash
# DSS no-basis, stage1-only training entrypoint for Math10K.

set -euo pipefail

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/root/code/DSS"}
REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC:-"/root/code/quest/peft/src"}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-"/root/datasets"}
REMOTE_MODEL_ROOT=${REMOTE_MODEL_ROOT:-"/root/hf_cache_models/models"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_math"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REMOTE_PROJECT_ROOT}/output_math"}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-"${REMOTE_PROJECT_ROOT}/experiments/math"}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

MODEL_NAME=${MODEL_NAME:-"Llama3-8B"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"${REMOTE_MODEL_ROOT}"}
if [[ -z "${MODEL_PATH:-}" ]]; then
    case "${MODEL_NAME}" in
        "Llama3-8B") MODEL_PATH="${REMOTE_MODEL_ROOT}/Meta-Llama-3-8B" ;;
        "Qwen2.5-7B") MODEL_PATH="${REMOTE_MODEL_ROOT}/Qwen2.5-7B" ;;
        *) echo "Unsupported MODEL_NAME=${MODEL_NAME}. Use Llama3-8B or Qwen2.5-7B." >&2; exit 1 ;;
    esac
fi

MAX_LENGTH=${MAX_LENGTH:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
DATA_DIR=${DATA_DIR:-"${REMOTE_DATA_ROOT}/math_reasoning/processed"}
DATASET_PATH=${DATASET_PATH:-"${DATA_DIR}/math10k_${MAX_LENGTH}_prompt${MAX_PROMPT_LENGTH}_OnlyOutput_${MODEL_NAME}"}
VAL_SET_SIZE=${VAL_SET_SIZE:-500}

TARGET_MODULES=${TARGET_MODULES:-"qvud"}
N_FREQUENCY=${N_FREQUENCY:-180000}
CANDIDATE_SIZE=${CANDIDATE_SIZE:-30000}
GRAD_STORE_STEPS=${GRAD_STORE_STEPS:-3}
LOW=${LOW:-500}
UP=${UP:-4000}
RATIO=${RATIO:-0.1}
THRESHOLD_MODE=${THRESHOLD_MODE:-"oracle"}
SCORE_METHOD=${SCORE_METHOD:-"abs_mean"}
SCORE_EPS=${SCORE_EPS:-"1e-8"}
DSS_DROPOUT=${DSS_DROPOUT:-0.05}
QUANTILE_LR=${QUANTILE_LR:-0.01}
QUANTILE_ALPHA=${QUANTILE_ALPHA:-0.0}
THRESHOLD_LOG_EVERY_STEPS=${THRESHOLD_LOG_EVERY_STEPS:-100}

BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:--1}
PRECISION=${PRECISION:-"bf16"}
LR=${LR:-"1e-4"}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-0}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-500}
SAVE_STEPS=${SAVE_STEPS:-500}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
REPORT_TO=${REPORT_TO:-"none"}
SEED=${SEED:-42}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED:-1}

RUN_MODE=${RUN_MODE:-"${THRESHOLD_MODE}"}
RUN_NAME=${RUN_NAME:-"math_${MODEL_NAME}_dss_nobasis_${RUN_MODE}_nf${N_FREQUENCY}_cand${CANDIDATE_SIZE}_gs${GRAD_STORE_STEPS}_lr${LR}_${TIMESTAMP}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${RUN_NAME}"}
LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${TIMESTAMP}_math_train_${RUN_MODE}.log"}

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

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi
export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT=${WANDB_PROJECT:-"dss_math"}
export WANDB_NAME=${WANDB_NAME:-"${RUN_NAME}"}

for path in "${MODEL_PATH}" "${DATASET_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path missing: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${EXPERIMENT_ROOT}"
cd "${REMOTE_PROJECT_ROOT}"

{
    echo "========== DSS MATH TRAIN =========="
    echo "[config] start: $(date)"
    echo "RUN_NAME=${RUN_NAME}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS=${NUM_GPUS}"
    echo "DATASET_PATH=${DATASET_PATH}"
    echo "MAX_LENGTH=${MAX_LENGTH}"
    echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
    echo "VAL_SET_SIZE=${VAL_SET_SIZE}"
    echo "TARGET_MODULES=${TARGET_MODULES}"
    echo "N_FREQUENCY=${N_FREQUENCY}"
    echo "CANDIDATE_SIZE=${CANDIDATE_SIZE}"
    echo "GRAD_STORE_STEPS=${GRAD_STORE_STEPS}"
    echo "LOW=${LOW}"
    echo "UP=${UP}"
    echo "RATIO=${RATIO}"
    echo "THRESHOLD_MODE=${THRESHOLD_MODE}"
    echo "SCORE_METHOD=${SCORE_METHOD}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "LR=${LR}"
    echo "EVAL_STEPS=${EVAL_STEPS}"
    echo "SAVE_STEPS=${SAVE_STEPS}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
    echo
} | tee "${LOG_FILE}"

train_cmd=(
    finetune_math.py
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
    --score_method "${SCORE_METHOD}"
    --score_eps "${SCORE_EPS}"
    --dropout "${DSS_DROPOUT}"
    --quantile_lr "${QUANTILE_LR}"
    --quantile_alpha "${QUANTILE_ALPHA}"
    --threshold_log_every_steps "${THRESHOLD_LOG_EVERY_STEPS}"
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
    train_cmd+=(--load_best_model_at_end)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    train_cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

train_start=$(date +%s)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
    python -m torch.distributed.run --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" "${train_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
    python "${train_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
train_end=$(date +%s)
elapsed=$((train_end - train_start))

{
    echo
    echo "========== DSS MATH TRAIN DONE =========="
    echo "[train] end: $(date)"
    echo "elapsed=$(format_elapsed "${elapsed}")"
    echo "output_dir=${OUTPUT_DIR}"
} | tee -a "${LOG_FILE}"

if [[ "${EXPERIMENT_RECORD_ENABLED}" == "1" ]]; then
    python scripts/update_math_run_record.py \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_file "${LOG_FILE}"
    python scripts/export_math_runs_csv.py
fi
