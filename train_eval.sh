#!/bin/bash
# Single-GPU DSS train + full commonsense eval entrypoint.
#
# Usage examples:
#   CUDA_VISIBLE_DEVICES=0 THRESHOLD_MODE=oracle tmux new-session -d -s train_oracle \
#     "cd /root/code/DSS && bash /root/code/DSS/train_eval.sh"
#   CUDA_VISIBLE_DEVICES=1 THRESHOLD_MODE=sgd QUANTILE_LR=0.03 tmux new-session -d -s train_sgd \
#     "cd /root/code/DSS && bash /root/code/DSS/train_eval.sh"

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
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"${REMOTE_DATA_ROOT}/evaluate"}
MAX_LENGTH=${MAX_LENGTH:-256}
DATASET_PATH=${DATASET_PATH:-"${DATA_DIR}/train_all_${MAX_LENGTH}_OnlyOutput_${MODEL_NAME}"}
VAL_SET_SIZE=${VAL_SET_SIZE:-500}

TARGET_MODULES=${TARGET_MODULES:-"qkvud"}
PEFT_METHOD=${PEFT_METHOD:-"dss"}
N_FREQUENCY=${N_FREQUENCY:-180000}
CANDIDATE_SIZE=${CANDIDATE_SIZE:-10000}
GRAD_STORE_STEPS=${GRAD_STORE_STEPS:-10}
LOW=${LOW:-100}
UP=${UP:-1000}
RATIO=${RATIO:-0.05}
THRESHOLD_MODE=${THRESHOLD_MODE:-"oracle"}
SCORE_METHOD=${SCORE_METHOD:-"abs_mean"}
SCORE_EPS=${SCORE_EPS:-"1e-8"}
DSS_DROPOUT=${DSS_DROPOUT:-0.05}
QUANTILE_LR=${QUANTILE_LR:-0.01}
QUANTILE_ALPHA=${QUANTILE_ALPHA:-0.0}
THRESHOLD_LOG_EVERY_STEPS=${THRESHOLD_LOG_EVERY_STEPS:-100}
INIT_ENABLED=${INIT_ENABLED:-0}
INIT_STEPS=${INIT_STEPS:-10}
INIT_CANDIDATE_RATIO=${INIT_CANDIDATE_RATIO:-0.05}
INIT_SEED_MODE=${INIT_SEED_MODE:-"threshold_only"}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
USE_RSLORA=${USE_RSLORA:-0}

BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:--1}
PRECISION=${PRECISION:-"bf16"}
LR=${LR:-"8e-5"}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-10000}
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
REPORT_TO=${REPORT_TO:-"none"}
SEED=${SEED:-42}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
NUM_GPUS=${NUM_GPUS:-}
MASTER_PORT=${MASTER_PORT:-29500}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-32}
EVAL_NUM_BEAMS=${EVAL_NUM_BEAMS:-4}
EVAL_DEBUG_EVAL=${EVAL_DEBUG_EVAL:-1}
EVAL_DEBUG_FIRST_N=${EVAL_DEBUG_FIRST_N:-10}
EVAL_DEBUG_MAX_FAILURES=${EVAL_DEBUG_MAX_FAILURES:-20}
EVAL_MAX_EXAMPLES=${EVAL_MAX_EXAMPLES:-0}

EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED:-1}

RUN_MODE=${RUN_MODE:-"${THRESHOLD_MODE}"}
if [[ -z "${RUN_NAME:-}" ]]; then
    if [[ "${PEFT_METHOD}" == "dss" ]]; then
        RUN_NAME="commonsense_${MODEL_NAME}_dss_nobasis_${RUN_MODE}_nf${N_FREQUENCY}_cand${CANDIDATE_SIZE}_gs${GRAD_STORE_STEPS}_${TIMESTAMP}"
    else
        RUN_NAME="commonsense_${MODEL_NAME}_${PEFT_METHOD}_r${LORA_R}_a${LORA_ALPHA}_lr${LR}_${TIMESTAMP}"
    fi
fi
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${RUN_NAME}"}
LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${TIMESTAMP}_train_eval_${RUN_MODE}.log"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"${RESULT_ROOT}/${RUN_NAME}_all_eval"}

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

visible_gpu_count=1
if [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
    visible_gpu_count=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
fi
if [[ -z "${NUM_GPUS}" ]]; then
    NUM_GPUS=${visible_gpu_count}
fi
if [[ "${NUM_GPUS}" -gt "${visible_gpu_count}" ]]; then
    echo "NUM_GPUS=${NUM_GPUS} exceeds visible GPUs (${CUDA_VISIBLE_DEVICES})" >&2
    exit 1
fi
export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_PROJECT=${WANDB_PROJECT:-"dss_commonsense"}
export WANDB_NAME=${WANDB_NAME:-"${RUN_NAME}"}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}

if [[ "${THRESHOLD_MODE}" != "oracle" && "${THRESHOLD_MODE}" != "sgd" ]]; then
    echo "THRESHOLD_MODE must be oracle or sgd; got ${THRESHOLD_MODE}" >&2
    exit 1
fi

for path in "${MODEL_PATH}" "${DATASET_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path missing: ${path}" >&2
        exit 1
    fi
done

for dataset in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    if [[ ! -f "${EVAL_DATA_DIR}/${dataset}/test.json" ]]; then
        echo "Missing eval dataset file: ${EVAL_DATA_DIR}/${dataset}/test.json" >&2
        exit 1
    fi
done

if ! python - <<'PY' >/dev/null 2>&1
import peft  # noqa: F401
PY
then
    echo "Failed to import local peft after setting PYTHONPATH=${PYTHONPATH}" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${OUTPUT_DIR}" "${EXPERIMENT_ROOT}" "${EVAL_OUTPUT_DIR}"
cd "${REMOTE_PROJECT_ROOT}"

{
    echo "========== DSS TRAIN+EVAL RUN =========="
    echo "[config] start: $(date)"
    echo "RUN_MODE=${RUN_MODE}"
    echo "RUN_NAME=${RUN_NAME}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS=${NUM_GPUS}"
    echo "DATASET_PATH=${DATASET_PATH}"
    echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
    echo "MAX_LENGTH=${MAX_LENGTH}"
    echo "VAL_SET_SIZE=${VAL_SET_SIZE}"
    echo "TARGET_MODULES=${TARGET_MODULES}"
    echo "PEFT_METHOD=${PEFT_METHOD}"
    echo "N_FREQUENCY=${N_FREQUENCY}"
    echo "CANDIDATE_SIZE=${CANDIDATE_SIZE}"
    echo "GRAD_STORE_STEPS=${GRAD_STORE_STEPS}"
    echo "LOW=${LOW}"
    echo "UP=${UP}"
    echo "RATIO=${RATIO}"
    echo "THRESHOLD_MODE=${THRESHOLD_MODE}"
    echo "SCORE_METHOD=${SCORE_METHOD}"
    echo "SCORE_EPS=${SCORE_EPS}"
    echo "DSS_DROPOUT=${DSS_DROPOUT}"
    echo "QUANTILE_LR=${QUANTILE_LR}"
    echo "QUANTILE_ALPHA=${QUANTILE_ALPHA}"
    echo "THRESHOLD_LOG_EVERY_STEPS=${THRESHOLD_LOG_EVERY_STEPS}"
    echo "INIT_ENABLED=${INIT_ENABLED}"
    echo "INIT_STEPS=${INIT_STEPS}"
    echo "INIT_CANDIDATE_RATIO=${INIT_CANDIDATE_RATIO}"
    echo "INIT_SEED_MODE=${INIT_SEED_MODE}"
    echo "LORA_R=${LORA_R}"
    echo "LORA_ALPHA=${LORA_ALPHA}"
    echo "LORA_DROPOUT=${LORA_DROPOUT}"
    echo "USE_RSLORA=${USE_RSLORA}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "MAX_STEPS=${MAX_STEPS}"
    echo "PRECISION=${PRECISION}"
    echo "LR=${LR}"
    echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
    echo "WARMUP_RATIO=${WARMUP_RATIO}"
    echo "EVAL_STEPS=${EVAL_STEPS}"
    echo "SAVE_STEPS=${SAVE_STEPS}"
    echo "LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END}"
    echo "REPORT_TO=${REPORT_TO}"
    echo "SEED=${SEED}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
    echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
    echo "EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS}"
    echo "EVAL_NUM_BEAMS=${EVAL_NUM_BEAMS}"
    echo "EVAL_DEBUG_EVAL=${EVAL_DEBUG_EVAL}"
    echo "EVAL_DEBUG_FIRST_N=${EVAL_DEBUG_FIRST_N}"
    echo "EVAL_DEBUG_MAX_FAILURES=${EVAL_DEBUG_MAX_FAILURES}"
    echo "EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED}"
    if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
        echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
    fi
    echo
} | tee "${LOG_FILE}"

train_cmd=(
    finetune_commonsense.py
    --model_name "${MODEL_NAME}"
    --model_path "${MODEL_PATH}"
    --model_cache_dir "${MODEL_CACHE_DIR}"
    --data_dir "${DATA_DIR}"
    --dataset_path "${DATASET_PATH}"
    --max_length "${MAX_LENGTH}"
    --val_set_size "${VAL_SET_SIZE}"
    --target_modules "${TARGET_MODULES}"
    --peft_method "${PEFT_METHOD}"
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
    --init_steps "${INIT_STEPS}"
    --init_candidate_ratio "${INIT_CANDIDATE_RATIO}"
    --init_seed_mode "${INIT_SEED_MODE}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --num_epochs "${NUM_EPOCHS}"
    --max_steps "${MAX_STEPS}"
    --precision "${PRECISION}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_ratio "${WARMUP_RATIO}"
    --logging_steps "${LOGGING_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --num_workers "${NUM_WORKERS}"
    --report_to "${REPORT_TO}"
    --output_dir "${OUTPUT_DIR}"
    --run_name "${RUN_NAME}"
    --seed "${SEED}"
)

if [[ "${INIT_ENABLED}" == "1" ]]; then
    train_cmd+=(--init_enabled)
fi
if [[ "${USE_RSLORA}" == "1" ]]; then
    train_cmd+=(--use_rslora)
fi

if [[ "${LOAD_BEST_MODEL_AT_END}" == "1" ]]; then
    train_cmd+=(--load_best_model_at_end)
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    train_cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

{
    echo "========== DSS TRAINING =========="
    echo "[train] start: $(date)"
} | tee -a "${LOG_FILE}"
train_start=$(date +%s)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    launch_cmd=(
        python
        -m
        torch.distributed.run
        --nproc_per_node "${NUM_GPUS}"
        --master_port "${MASTER_PORT}"
        "${train_cmd[@]}"
    )
else
    launch_cmd=(
        python
        "${train_cmd[@]}"
    )
fi

"${launch_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"

train_end=$(date +%s)
elapsed=$((train_end - train_start))
{
    echo
    echo "========== DSS TRAINING DONE =========="
    echo "[train] end: $(date)"
    echo "elapsed=$(format_elapsed "${elapsed}")"
    echo "output_dir=${OUTPUT_DIR}"
} | tee -a "${LOG_FILE}"

for artifact in adapter_config.json adapter_model.safetensors training_args.json; do
    if [[ ! -f "${OUTPUT_DIR}/${artifact}" ]]; then
        echo "Training artifact missing: ${OUTPUT_DIR}/${artifact}" | tee -a "${LOG_FILE}"
        echo "训练产物不完整，未开始评测" | tee -a "${LOG_FILE}"
        exit 1
    fi
done

if [[ "${EXPERIMENT_RECORD_ENABLED}" == "1" ]]; then
    python scripts/update_run_record.py \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_file "${LOG_FILE}"
    python scripts/export_runs_csv.py
fi

{
    echo
    echo "========== COMMONSENSE EVAL =========="
    echo "[eval] start: $(date)"
    echo "adapter_dir=${OUTPUT_DIR}"
    echo "eval_output_dir=${EVAL_OUTPUT_DIR}"
} | tee -a "${LOG_FILE}"

MODEL_NAME="${MODEL_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${EVAL_DATA_DIR}" \
PRECISION="${PRECISION}" \
BATCH_SIZE="${EVAL_BATCH_SIZE}" \
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}" \
NUM_BEAMS="${EVAL_NUM_BEAMS}" \
MAX_EXAMPLES="${EVAL_MAX_EXAMPLES}" \
DEBUG_EVAL="${EVAL_DEBUG_EVAL}" \
DEBUG_FIRST_N="${EVAL_DEBUG_FIRST_N}" \
DEBUG_MAX_FAILURES="${EVAL_DEBUG_MAX_FAILURES}" \
RUN_NAME="${RUN_NAME}" \
LOG_FILE="${LOG_FILE}" \
LOG_APPEND=1 \
EXPERIMENT_RECORD_ENABLED="${EXPERIMENT_RECORD_ENABLED}" \
bash "${REMOTE_PROJECT_ROOT}/scripts/eval_commonsense.sh" "${OUTPUT_DIR}" all "${EVAL_OUTPUT_DIR}"

{
    echo
    echo "========== TRAIN+EVAL DONE =========="
    echo "[done] finish: $(date)"
    echo "log_file=${LOG_FILE}"
    echo "output_dir=${OUTPUT_DIR}"
    echo "eval_output_dir=${EVAL_OUTPUT_DIR}"
} | tee -a "${LOG_FILE}"
