#!/bin/bash
# LLM-Adapters-style commonsense train + 8-task eval entrypoint.

set -euo pipefail

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/root/code/DSS"}
REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC:-"/root/code/quest/peft/src"}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-"/root/datasets"}
REMOTE_MODEL_ROOT=${REMOTE_MODEL_ROOT:-"/root/hf_cache_models/models"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_commonsense"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REMOTE_PROJECT_ROOT}/output_llm_adapters"}
RESULT_ROOT=${RESULT_ROOT:-"${REMOTE_PROJECT_ROOT}/results_commonsense_llm_adapters"}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

MODEL_NAME=${MODEL_NAME:-"Qwen3-8B"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"${REMOTE_MODEL_ROOT}"}
if [[ -z "${MODEL_PATH:-}" ]]; then
    case "${MODEL_NAME}" in
        "Qwen3-8B")
            MODEL_PATH="${REMOTE_MODEL_ROOT}/Qwen3-8B"
            ;;
        "Llama3-8B")
            MODEL_PATH="${REMOTE_MODEL_ROOT}/Meta-Llama-3-8B"
            ;;
        *)
            echo "Unsupported MODEL_NAME=${MODEL_NAME}. Use Qwen3-8B or Llama3-8B." >&2
            exit 1
            ;;
    esac
fi

TRAIN_JSON=${TRAIN_JSON:-"${REMOTE_DATA_ROOT}/ft-training_set/commonsense_170k.json"}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"${REMOTE_DATA_ROOT}/evaluate"}

ADAPTER_NAME=${ADAPTER_NAME:-"lora"}
TARGET_MODULES=${TARGET_MODULES:-"q_proj,k_proj,v_proj,up_proj,down_proj"}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
SHIRA_R=${SHIRA_R:-32}
SHIRA_MASK_TYPE=${SHIRA_MASK_TYPE:-"random"}
SHIRA_RANDOM_SEED=${SHIRA_RANDOM_SEED:-42}
SHIRA_INIT_WEIGHTS=${SHIRA_INIT_WEIGHTS:-1}
SHIRA_CALIBRATION_SAMPLES=${SHIRA_CALIBRATION_SAMPLES:-32}

# Match the 87.2 DSS baseline: global batch 32 = 16 samples/GPU x 2 GPUs.
BATCH_SIZE=${BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-"8e-5"}
MAX_LENGTH=${MAX_LENGTH:-256}
VAL_SET_SIZE=${VAL_SET_SIZE:-120}
WARMUP_STEPS=${WARMUP_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-10000}
PRECISION=${PRECISION:-"bf16"}
SEED=${SEED:-42}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-0}
NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29501}

# DSS defaults reproduce the 87.2 Commonsense baseline configuration.
N_FREQUENCY=${N_FREQUENCY:-180000}
CANDIDATE_SIZE=${CANDIDATE_SIZE:-10000}
GRAD_STORE_STEPS=${GRAD_STORE_STEPS:-10}
RATIO=${RATIO:-0.05}
LOW=${LOW:-100}
UP=${UP:-1000}
THRESHOLD_MODE=${THRESHOLD_MODE:-"oracle"}
SCORE_METHOD=${SCORE_METHOD:-"snr"}
SCORE_EPS=${SCORE_EPS:-"1e-8"}
DSS_DROPOUT=${DSS_DROPOUT:-0.05}
QUANTILE_LR=${QUANTILE_LR:-0.01}
QUANTILE_ALPHA=${QUANTILE_ALPHA:-0.0}
THRESHOLD_LOG_EVERY_STEPS=${THRESHOLD_LOG_EVERY_STEPS:-100}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_NUM_BEAMS=${EVAL_NUM_BEAMS:-1}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-32}

if [[ $((BATCH_SIZE % (MICRO_BATCH_SIZE * NUM_GPUS))) -ne 0 ]]; then
    echo "BATCH_SIZE=${BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE * NUM_GPUS (${MICRO_BATCH_SIZE} * ${NUM_GPUS})" >&2
    exit 1
fi

GRAD_ACCUM_STEPS=$((BATCH_SIZE / (MICRO_BATCH_SIZE * NUM_GPUS)))
if [[ -z "${RUN_NAME:-}" && "${ADAPTER_NAME}" == "dss" ]]; then
    RUN_NAME="commonsense_${MODEL_NAME}_llm_adapter_dss_rawjson_snr_nf${N_FREQUENCY}_cand${CANDIDATE_SIZE}_gs${GRAD_STORE_STEPS}_lr${LR}_${TIMESTAMP}"
elif [[ -z "${RUN_NAME:-}" && "${ADAPTER_NAME}" == "shira" ]]; then
    RUN_NAME="commonsense_${MODEL_NAME}_llm_adapter_shira_${SHIRA_MASK_TYPE}_r${SHIRA_R}_seed${SHIRA_RANDOM_SEED}_lr${LR}_bs${MICRO_BATCH_SIZE}x${GRAD_ACCUM_STEPS}_${TIMESTAMP}"
else
    RUN_NAME=${RUN_NAME:-"commonsense_${MODEL_NAME}_llm_adapter_${ADAPTER_NAME}_r${LORA_R}_a${LORA_ALPHA}_lr${LR}_bs${MICRO_BATCH_SIZE}x${GRAD_ACCUM_STEPS}_${TIMESTAMP}"}
fi
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${RUN_NAME}"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"${RESULT_ROOT}/${RUN_NAME}_all_eval"}
LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${TIMESTAMP}_train_eval_llm_adapters_${ADAPTER_NAME}.log"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
fi
EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES%%,*}}

export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"

for path in "${MODEL_PATH}" "${TRAIN_JSON}"; do
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

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
cd "${REMOTE_PROJECT_ROOT}"

{
    echo "========== LLM-ADAPTERS TRAIN+EVAL RUN =========="
    echo "[config] start: $(date)"
    echo "RUN_NAME=${RUN_NAME}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "TRAIN_JSON=${TRAIN_JSON}"
    echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS=${NUM_GPUS}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES}"
    echo "ADAPTER_NAME=${ADAPTER_NAME}"
    echo "TARGET_MODULES=${TARGET_MODULES}"
    echo "LORA_R=${LORA_R}"
    echo "LORA_ALPHA=${LORA_ALPHA}"
    echo "LORA_DROPOUT=${LORA_DROPOUT}"
    if [[ "${ADAPTER_NAME}" == "shira" ]]; then
        echo "SHIRA_R=${SHIRA_R}"
        echo "SHIRA_MASK_TYPE=${SHIRA_MASK_TYPE}"
        echo "SHIRA_RANDOM_SEED=${SHIRA_RANDOM_SEED}"
        echo "SHIRA_INIT_WEIGHTS=${SHIRA_INIT_WEIGHTS}"
        echo "SHIRA_CALIBRATION_SAMPLES=${SHIRA_CALIBRATION_SAMPLES}"
    fi
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "LR=${LR}"
    echo "MAX_LENGTH=${MAX_LENGTH}"
    echo "VAL_SET_SIZE=${VAL_SET_SIZE}"
    echo "WARMUP_STEPS=${WARMUP_STEPS}"
    echo "LOGGING_STEPS=${LOGGING_STEPS}"
    echo "EVAL_STEPS=${EVAL_STEPS}"
    echo "SAVE_STEPS=${SAVE_STEPS}"
    echo "PRECISION=${PRECISION}"
    echo "SEED=${SEED}"
    echo "USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
    if [[ "${ADAPTER_NAME}" == "dss" ]]; then
        echo "N_FREQUENCY=${N_FREQUENCY}"
        echo "CANDIDATE_SIZE=${CANDIDATE_SIZE}"
        echo "GRAD_STORE_STEPS=${GRAD_STORE_STEPS}"
        echo "RATIO=${RATIO}"
        echo "LOW=${LOW}"
        echo "UP=${UP}"
        echo "THRESHOLD_MODE=${THRESHOLD_MODE}"
        echo "SCORE_METHOD=${SCORE_METHOD}"
        echo "SCORE_EPS=${SCORE_EPS}"
        echo "DSS_DROPOUT=${DSS_DROPOUT}"
        echo "QUANTILE_LR=${QUANTILE_LR}"
        echo "QUANTILE_ALPHA=${QUANTILE_ALPHA}"
        echo "THRESHOLD_LOG_EVERY_STEPS=${THRESHOLD_LOG_EVERY_STEPS}"
    fi
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
    echo "EVAL_NUM_BEAMS=${EVAL_NUM_BEAMS}"
    echo "EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS}"
    echo
} | tee "${LOG_FILE}"

train_cmd=(
    torchrun
    --standalone
    --nproc_per_node "${NUM_GPUS}"
    --master_port "${MASTER_PORT}"
    "${REMOTE_PROJECT_ROOT}/llm_adapters_finetune.py"
    --base_model "${MODEL_PATH}"
    --data_path "${TRAIN_JSON}"
    --output_dir "${OUTPUT_DIR}"
    --adapter_name "${ADAPTER_NAME}"
    --batch_size "${BATCH_SIZE}"
    --micro_batch_size "${MICRO_BATCH_SIZE}"
    --num_epochs "${NUM_EPOCHS}"
    --learning_rate "${LR}"
    --cutoff_len "${MAX_LENGTH}"
    --val_set_size "${VAL_SET_SIZE}"
    --eval_step "${EVAL_STEPS}"
    --save_step "${SAVE_STEPS}"
    --warmup_steps "${WARMUP_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --target_modules "${TARGET_MODULES}"
    --precision "${PRECISION}"
    --seed "${SEED}"
    --report_to none
)

if [[ "${USE_GRADIENT_CHECKPOINTING}" == "1" ]]; then
    train_cmd+=(--use_gradient_checkpointing)
fi

if [[ "${ADAPTER_NAME}" == "dss" ]]; then
    train_cmd+=(
        --n_frequency "${N_FREQUENCY}"
        --candidate_size "${CANDIDATE_SIZE}"
        --grad_store_steps "${GRAD_STORE_STEPS}"
        --ratio "${RATIO}"
        --low "${LOW}"
        --up "${UP}"
        --threshold_mode "${THRESHOLD_MODE}"
        --score_method "${SCORE_METHOD}"
        --score_eps "${SCORE_EPS}"
        --dss_dropout "${DSS_DROPOUT}"
        --quantile_lr "${QUANTILE_LR}"
        --quantile_alpha "${QUANTILE_ALPHA}"
        --threshold_log_every_steps "${THRESHOLD_LOG_EVERY_STEPS}"
    )
fi

if [[ "${ADAPTER_NAME}" == "shira" ]]; then
    if [[ "${SHIRA_MASK_TYPE}" != "random" && "${SHIRA_MASK_TYPE}" != "snip" ]]; then
        echo "SHIRA_MASK_TYPE must be random or snip, got: ${SHIRA_MASK_TYPE}" >&2
        exit 2
    fi
    train_cmd+=(
        --shira_r "${SHIRA_R}"
        --shira_mask_type "${SHIRA_MASK_TYPE}"
        --shira_random_seed "${SHIRA_RANDOM_SEED}"
        --shira_init_weights "${SHIRA_INIT_WEIGHTS}"
        --shira_calibration_samples "${SHIRA_CALIBRATION_SAMPLES}"
    )
fi

{
    echo "========== LLM-ADAPTERS TRAINING =========="
    echo "[train] start: $(date)"
} | tee -a "${LOG_FILE}"

"${train_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"

{
    echo
    echo "========== LLM-ADAPTERS EVAL =========="
    echo "[eval] start: $(date)"
} | tee -a "${LOG_FILE}"

eval_extra_args=()
if [[ "${ADAPTER_NAME}" == "shira" ]]; then
    # The saved adapter config is authoritative; these values make mismatches fail loudly.
    eval_extra_args=(
        --shira_r "${SHIRA_R}"
        --shira_mask_type "${SHIRA_MASK_TYPE}"
        --shira_random_seed "${SHIRA_RANDOM_SEED}"
    )
fi

for dataset in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    echo "========== DATASET: ${dataset} ==========" | tee -a "${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" python "${REMOTE_PROJECT_ROOT}/llm_adapters_commonsense_evaluate.py" \
        --dataset "${dataset}" \
        --base_model "${MODEL_PATH}" \
        --adapter_name "${ADAPTER_NAME}" \
        --adapter_weights "${OUTPUT_DIR}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_beams "${EVAL_NUM_BEAMS}" \
        --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
        --precision "${PRECISION}" \
        --data_dir "${EVAL_DATA_DIR}" \
        --output_dir "${EVAL_OUTPUT_DIR}" \
        "${eval_extra_args[@]}" \
        2>&1 | tee -a "${LOG_FILE}"
done

{
    echo
    echo "========== LLM-ADAPTERS TRAIN+EVAL DONE =========="
    echo "[done] finish: $(date)"
    echo "log_file=${LOG_FILE}"
    echo "output_dir=${OUTPUT_DIR}"
    echo "eval_output_dir=${EVAL_OUTPUT_DIR}"
} | tee -a "${LOG_FILE}"
