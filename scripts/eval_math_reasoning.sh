#!/bin/bash
# Evaluate a DSS adapter on GSM8K, SVAMP, AQuA, and MAWPS.

set -euo pipefail

ADAPTER_PATH=${1:?"Usage: bash scripts/eval_math_reasoning.sh <adapter_path> [dataset|all] [output_dir]"}
DATASET_ARG=${2:-all}

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/root/code/DSS"}
REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC:-"/root/code/quest/peft/src"}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-"/root/datasets"}
REMOTE_MODEL_ROOT=${REMOTE_MODEL_ROOT:-"/root/hf_cache_models/models"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_math"}
RESULT_ROOT=${RESULT_ROOT:-"${REMOTE_PROJECT_ROOT}/results_math"}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
ADAPTER_NAME="$(basename "${ADAPTER_PATH}")"
RUN_NAME=${RUN_NAME:-"eval_math_${MODEL_NAME:-Llama3-8B}_${ADAPTER_NAME}_${TIMESTAMP}"}
OUTPUT_DIR=${3:-"${RESULT_ROOT}/${RUN_NAME}"}
LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${RUN_NAME}.log"}
EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED:-1}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest

export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${REMOTE_PROJECT_ROOT}"

MODEL_NAME=${MODEL_NAME:-"Llama3-8B"}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-"${REMOTE_DATA_ROOT}/math_reasoning/raw_eval"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"${REMOTE_MODEL_ROOT}"}
PRECISION=${PRECISION:-"bf16"}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
NUM_BEAMS=${NUM_BEAMS:-1}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
ALL_DATASETS="gsm8k svamp aqua mawps"

if [[ "${DATASET_ARG}" == "all" ]]; then
    DATASETS=${DATASETS:-"${ALL_DATASETS}"}
else
    if [[ " ${ALL_DATASETS} " != *" ${DATASET_ARG} "* ]]; then
        echo "Unknown dataset '${DATASET_ARG}'. Use one of: ${ALL_DATASETS}, or all." >&2
        exit 1
    fi
    DATASETS="${DATASET_ARG}"
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}" "${RESULT_ROOT}"

{
    echo "========== MATH REASONING EVAL =========="
    echo "[eval] start: $(date)"
    echo "ADAPTER_PATH=${ADAPTER_PATH}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "DATASETS=${DATASETS}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
    echo "NUM_BEAMS=${NUM_BEAMS}"
    echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
    echo "PRECISION=${PRECISION}"
    echo
} | tee "${LOG_FILE}"

for dataset in ${DATASETS}; do
    echo "========== DATASET: ${dataset} ==========" | tee -a "${LOG_FILE}"
    cmd=(
        python evaluate_math_reasoning.py
        --model_name "${MODEL_NAME}"
        --dataset "${dataset}"
        --adapter_path "${ADAPTER_PATH}"
        --output_dir "${OUTPUT_DIR}"
        --data_dir "${DATA_DIR}"
        --model_cache_dir "${MODEL_CACHE_DIR}"
        --precision "${PRECISION}"
        --batch_size "${BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --num_beams "${NUM_BEAMS}"
        --max_examples "${MAX_EXAMPLES}"
        --run_name "${RUN_NAME}"
    )
    if [[ -n "${MODEL_PATH}" ]]; then
        cmd+=(--model_path "${MODEL_PATH}")
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
        cmd+=(--hf_token "${HF_TOKEN}")
    fi
    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "[eval] done: $(date)" | tee -a "${LOG_FILE}"

if [[ "${EXPERIMENT_RECORD_ENABLED}" == "1" ]]; then
    FULL_EVAL_FLAG=1
    if [[ "${MAX_EXAMPLES}" != "0" ]]; then
        FULL_EVAL_FLAG=0
    fi
    for dataset in ${DATASETS}; do
        python scripts/update_math_run_record.py \
            --run_name "${ADAPTER_NAME}" \
            --output_dir "${ADAPTER_PATH}" \
            --eval_output_dir "${OUTPUT_DIR}" \
            --dataset "${dataset}" \
            --is_full_eval "${FULL_EVAL_FLAG}"
    done
    python scripts/export_math_runs_csv.py
fi
