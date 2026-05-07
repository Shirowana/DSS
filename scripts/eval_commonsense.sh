#!/bin/bash
# Evaluate a DSS adapter on commonsense reasoning benchmarks.
#
# Usage:
#   bash scripts/eval_commonsense.sh <adapter_path> [dataset|all] [output_dir]
#
# Optional environment overrides:
#   MODEL_NAME=Llama3-8B
#   MODEL_PATH=/root/hf_cache_models/models/Meta-Llama-3-8B
#   DATA_DIR=/root/datasets/evaluate
#   PRECISION=bf16
#   BATCH_SIZE=1
#   MAX_NEW_TOKENS=32
#   NUM_BEAMS=4
#   DEBUG_EVAL=0
#   DEBUG_FIRST_N=5
#   DEBUG_MAX_FAILURES=20
#   DATASETS="boolq piqa ..."

set -euo pipefail

ADAPTER_PATH=${1:?"Usage: bash scripts/eval_commonsense.sh <adapter_path> [dataset|all] [output_dir]"}
DATASET_ARG=${2:-all}

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/root/code/DSS"}
REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC:-"/root/code/quest/peft/src"}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-"/root/datasets"}
REMOTE_MODEL_ROOT=${REMOTE_MODEL_ROOT:-"/root/hf_cache_models/models"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_commonsense"}
RESULT_ROOT=${RESULT_ROOT:-"${REMOTE_PROJECT_ROOT}/results_commonsense"}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-"${REMOTE_PROJECT_ROOT}/experiments/commonsense"}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
ADAPTER_NAME="$(basename "${ADAPTER_PATH}")"
RUN_NAME=${RUN_NAME:-"eval_commonsense_${MODEL_NAME:-Llama3-8B}_${ADAPTER_NAME}_${TIMESTAMP}"}
OUTPUT_DIR=${3:-"${RESULT_ROOT}/${RUN_NAME}"}
LOG_FILE=${LOG_FILE:-"${LOG_ROOT}/${RUN_NAME}.log"}
EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED:-1}
EXPERIMENT_MD=${EXPERIMENT_MD:-"${EXPERIMENT_ROOT}/${ADAPTER_NAME}.md"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest

export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${REMOTE_PROJECT_ROOT}"

MODEL_NAME=${MODEL_NAME:-"Llama3-8B"}
MODEL_PATH=${MODEL_PATH:-"${REMOTE_MODEL_ROOT}/Meta-Llama-3-8B"}
DATA_DIR=${DATA_DIR:-"${REMOTE_DATA_ROOT}/evaluate"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"${REMOTE_MODEL_ROOT}"}
PRECISION=${PRECISION:-"bf16"}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
NUM_BEAMS=${NUM_BEAMS:-4}
DEBUG_EVAL=${DEBUG_EVAL:-0}
DEBUG_FIRST_N=${DEBUG_FIRST_N:-5}
DEBUG_MAX_FAILURES=${DEBUG_MAX_FAILURES:-20}
ALL_DATASETS="boolq piqa social_i_qa winogrande ARC-Challenge ARC-Easy openbookqa hellaswag"
if [[ "${DATASET_ARG}" == "all" ]]; then
    DATASETS=${DATASETS:-"${ALL_DATASETS}"}
else
    if [[ " ${ALL_DATASETS} " != *" ${DATASET_ARG} "* ]]; then
        echo "Unknown dataset '${DATASET_ARG}'. Use one of: ${ALL_DATASETS}, or all."
        exit 1
    fi
    DATASETS="${DATASET_ARG}"
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}" "${RESULT_ROOT}" "${EXPERIMENT_ROOT}"

{
    echo "========== COMMONSENSE EVAL =========="
    echo "[eval] start: $(date)"
    echo "ADAPTER_PATH=${ADAPTER_PATH}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "MODEL_PATH=${MODEL_PATH:-}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "DATASETS=${DATASETS}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
    echo "NUM_BEAMS=${NUM_BEAMS}"
    echo "PRECISION=${PRECISION}"
    echo "DEBUG_EVAL=${DEBUG_EVAL}"
    echo "DEBUG_FIRST_N=${DEBUG_FIRST_N}"
    echo "DEBUG_MAX_FAILURES=${DEBUG_MAX_FAILURES}"
    echo "EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED}"
    echo "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
    echo "EXPERIMENT_MD=${EXPERIMENT_MD}"
    echo
} | tee "${LOG_FILE}"

for dataset in ${DATASETS}; do
    echo "========== DATASET: ${dataset} ==========" | tee -a "${LOG_FILE}"
    echo "Evaluating DSS on ${dataset}..." | tee -a "${LOG_FILE}"

    cmd=(
        python evaluate_commonsense.py
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
        --run_name "${RUN_NAME}"
    )

    if [[ -n "${MODEL_PATH:-}" ]]; then
        cmd+=(--model_path "${MODEL_PATH}")
    fi

    if [[ -n "${HF_TOKEN:-}" ]]; then
        cmd+=(--hf_token "${HF_TOKEN}")
    fi

    if [[ "${DEBUG_EVAL}" == "1" ]]; then
        cmd+=(
            --debug_eval
            --debug_first_n "${DEBUG_FIRST_N}"
            --debug_max_failures "${DEBUG_MAX_FAILURES}"
        )
    fi

    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "[eval] done: $(date)" | tee -a "${LOG_FILE}"

if [[ "${EXPERIMENT_RECORD_ENABLED}" == "1" ]]; then
    python scripts/append_commonsense_eval_record.py \
        --experiment_md "${EXPERIMENT_MD}" \
        --eval_output_dir "${OUTPUT_DIR}" \
        --eval_log_file "${LOG_FILE}" \
        --experiment_root "${EXPERIMENT_ROOT}"
fi
