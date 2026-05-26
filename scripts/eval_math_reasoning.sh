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
LOG_APPEND=${LOG_APPEND:-0}
EXPERIMENT_RECORD_ENABLED=${EXPERIMENT_RECORD_ENABLED:-1}
EVAL_PARALLEL_2GPU=${EVAL_PARALLEL_2GPU:-1}
GPU0_DEVICE=${GPU0_DEVICE:-0}
GPU1_DEVICE=${GPU1_DEVICE:-1}
GPU0_DATASETS=${GPU0_DATASETS:-"gsm8k"}
GPU1_DATASETS=${GPU1_DATASETS:-"svamp aqua mawps"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest

export PYTHONPATH="${REMOTE_PEFT_SRC}:${REMOTE_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${REMOTE_PROJECT_ROOT}"

MODEL_NAME=${MODEL_NAME:-"Llama3-8B"}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-"${REMOTE_DATA_ROOT}/evaluate"}
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

if [[ "${LOG_APPEND}" == "1" ]]; then
    log_tee_args=(-a)
else
    log_tee_args=()
fi

if [[ "${DATASET_ARG}" == "all" && "${EVAL_PARALLEL_2GPU}" == "1" ]]; then
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
        echo "EVAL_PARALLEL_2GPU=1"
        echo "GPU0_DEVICE=${GPU0_DEVICE}"
        echo "GPU0_DATASETS=${GPU0_DATASETS}"
        echo "GPU1_DEVICE=${GPU1_DEVICE}"
        echo "GPU1_DATASETS=${GPU1_DATASETS}"
        echo "BATCH_SIZE=${BATCH_SIZE}"
        echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
        echo "NUM_BEAMS=${NUM_BEAMS}"
        echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
        echo "PRECISION=${PRECISION}"
        echo
    } | tee "${log_tee_args[@]}" "${LOG_FILE}"

    worker0_dir="${OUTPUT_DIR}/worker_gpu${GPU0_DEVICE}"
    worker1_dir="${OUTPUT_DIR}/worker_gpu${GPU1_DEVICE}"
    worker0_log="${OUTPUT_DIR}/worker_gpu${GPU0_DEVICE}.log"
    worker1_log="${OUTPUT_DIR}/worker_gpu${GPU1_DEVICE}.log"

    run_worker() {
        local device=$1
        local datasets=$2
        local worker_dir=$3
        local worker_log=$4

        CUDA_VISIBLE_DEVICES="${device}" \
        DATASETS="${datasets}" \
        MODEL_NAME="${MODEL_NAME}" \
        MODEL_PATH="${MODEL_PATH}" \
        DATA_DIR="${DATA_DIR}" \
        MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
        PRECISION="${PRECISION}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
        NUM_BEAMS="${NUM_BEAMS}" \
        MAX_EXAMPLES="${MAX_EXAMPLES}" \
        RUN_NAME="${RUN_NAME}" \
        LOG_FILE="${worker_log}" \
        EVAL_PARALLEL_2GPU=0 \
        EXPERIMENT_RECORD_ENABLED=0 \
        bash "${REMOTE_PROJECT_ROOT}/scripts/eval_math_reasoning.sh" "${ADAPTER_PATH}" all "${worker_dir}"
    }

    run_worker "${GPU0_DEVICE}" "${GPU0_DATASETS}" "${worker0_dir}" "${worker0_log}" &
    pid0=$!
    run_worker "${GPU1_DEVICE}" "${GPU1_DATASETS}" "${worker1_dir}" "${worker1_log}" &
    pid1=$!

    status=0
    if ! wait "${pid0}"; then
        echo "[eval] GPU${GPU0_DEVICE} worker failed. See ${worker0_log}" | tee -a "${LOG_FILE}"
        status=1
    fi
    if ! wait "${pid1}"; then
        echo "[eval] GPU${GPU1_DEVICE} worker failed. See ${worker1_log}" | tee -a "${LOG_FILE}"
        status=1
    fi
    if [[ "${status}" != "0" ]]; then
        exit "${status}"
    fi

    python - "${OUTPUT_DIR}" "${worker0_dir}" "${worker1_dir}" "${RUN_NAME}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
worker_dirs = [Path(sys.argv[2]), Path(sys.argv[3])]
run_name = sys.argv[4]
summary = {"run_name": run_name, "datasets": {}}

for worker_dir in worker_dirs:
    worker_summary_path = worker_dir / "summary.json"
    if worker_summary_path.exists():
        worker_summary = json.loads(worker_summary_path.read_text(encoding="utf-8"))
        summary["datasets"].update(worker_summary.get("datasets", {}))
    for result_file in worker_dir.glob("*.json"):
        if result_file.name == "summary.json":
            continue
        shutil.copy2(result_file, output_dir / result_file.name)

(output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

    {
        echo "[eval] workers done: $(date)"
        echo "[eval] worker logs:"
        echo "  ${worker0_log}"
        echo "  ${worker1_log}"
        echo "[eval] merged summary: ${OUTPUT_DIR}/summary.json"
    } | tee -a "${LOG_FILE}"

    if [[ "${EXPERIMENT_RECORD_ENABLED}" == "1" ]]; then
        FULL_EVAL_FLAG=1
        if [[ "${MAX_EXAMPLES}" != "0" ]]; then
            FULL_EVAL_FLAG=0
        fi
        for dataset in ${ALL_DATASETS}; do
            python scripts/update_math_run_record.py \
                --run_name "${ADAPTER_NAME}" \
                --output_dir "${ADAPTER_PATH}" \
                --eval_output_dir "${OUTPUT_DIR}" \
                --dataset "${dataset}" \
                --is_full_eval "${FULL_EVAL_FLAG}"
        done
        python scripts/export_math_runs_csv.py
    fi

    echo "[eval] done: $(date)" | tee -a "${LOG_FILE}"
    exit 0
fi

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
} | tee "${log_tee_args[@]}" "${LOG_FILE}"

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
