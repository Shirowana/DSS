#!/bin/bash
# Evaluate a DSS adapter on commonsense reasoning benchmarks.
#
# Usage:
#   bash scripts/eval_commonsense.sh <adapter_path> [output_dir]
#
# Optional environment overrides:
#   MODEL_NAME=Llama3-8B
#   MODEL_PATH=/data/home/7250091/date/hf_cache_models/models/...
#   DATA_DIR=/data/home/7250091/date/datasets/evaluate
#   HF_TOKEN=...
#   PRECISION=bf16
#   BATCH_SIZE=4
#   MAX_NEW_TOKENS=32
#   MERGE_ADAPTER=0
#   DATASETS="boolq piqa ..."

set -e

ADAPTER_PATH=${1:?"Usage: bash scripts/eval_commonsense.sh <adapter_path> [output_dir]"}

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/data/home/7250091/date/DSS"}
REMOTE_PEFT_SRC=${REMOTE_PEFT_SRC:-"/data/home/7250091/date/quest20260313/peft/src"}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-"/data/home/7250091/date/datasets"}
REMOTE_MODEL_ROOT=${REMOTE_MODEL_ROOT:-"/data/home/7250091/date/hf_cache_models/models"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_commonsense"}
RESULT_ROOT=${RESULT_ROOT:-"${REMOTE_PROJECT_ROOT}/results_commonsense"}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
ADAPTER_NAME="$(basename "${ADAPTER_PATH}")"
RUN_NAME=${RUN_NAME:-"eval_commonsense_${MODEL_NAME:-Llama3-8B}_${ADAPTER_NAME}_${TIMESTAMP}"}
OUTPUT_DIR=${2:-"${RESULT_ROOT}/${RUN_NAME}"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest

export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH}"
cd "${REMOTE_PROJECT_ROOT}"

MODEL_NAME=${MODEL_NAME:-"Llama3-8B"}
DATA_DIR=${DATA_DIR:-"${REMOTE_DATA_ROOT}/evaluate"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"${REMOTE_MODEL_ROOT}"}
PRECISION=${PRECISION:-"bf16"}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
MERGE_ADAPTER=${MERGE_ADAPTER:-0}
DATASETS=${DATASETS:-"boolq piqa social_i_qa winogrande ARC-Challenge ARC-Easy openbookqa hellaswag"}

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}" "${RESULT_ROOT}"

for dataset in ${DATASETS}; do
    echo "Evaluating DSS on ${dataset}..."

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
        --run_name "${RUN_NAME}"
    )

    if [[ -n "${MODEL_PATH}" ]]; then
        cmd+=(--model_path "${MODEL_PATH}")
    fi

    if [[ -n "${HF_TOKEN}" ]]; then
        cmd+=(--hf_token "${HF_TOKEN}")
    fi

    if [[ "${MERGE_ADAPTER}" == "1" ]]; then
        cmd+=(--merge_adapter)
    fi

    "${cmd[@]}" 2>&1 | tee "${LOG_ROOT}/eval_${dataset}_${TIMESTAMP}.log"
done
