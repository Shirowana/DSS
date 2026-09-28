#!/usr/bin/env bash
set -euo pipefail

# End-to-end Commonsense S2FT run.  Override any variable before invoking, e.g.
# MODEL_FAMILY=llama3 RUN_ID=llama3_cs_s2ft bash scripts/train_eval_commonsense.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/7250091/date/conda_env/quest/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MODEL_FAMILY="${MODEL_FAMILY:-llama2}"
METHOD="${METHOD:-s2ft}"
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
RUN_ID="${RUN_ID:-commonsense_${MODEL_FAMILY}_${METHOD}_$(date -u +%Y%m%d_%H%M%S)_seed${SEED}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

case "${MODEL_FAMILY}" in
  llama2)
    MODEL_PATH="${MODEL_PATH:-/data/home/7250091/date/hf_cache_models/models/Llama-2-7b-hf}"
    PREPARED_DATA="${PREPARED_DATA:-/data/home/7250091/date/datasets/commonsense_new/Llama2-7B}"
    ;;
  llama3)
    MODEL_PATH="${MODEL_PATH:-/data/home/7250091/date/hf_cache_models/models/Meta-Llama-3-8B}"
    PREPARED_DATA="${PREPARED_DATA:-/data/home/7250091/date/datasets/commonsense_new/Llama3-8B}"
    ;;
  *)
    echo "MODEL_FAMILY must be llama2 or llama3, got: ${MODEL_FAMILY}" >&2
    exit 2
    ;;
esac

MODEL_SLUG="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${METHOD}/commonsense/${MODEL_SLUG}/${RUN_ID}}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/${METHOD}/commonsense/${MODEL_SLUG}/${RUN_ID}}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "========== TRAIN+EVAL RUN =========="
echo "[config] run_id=${RUN_ID}"
echo "[config] log_dir=${LOG_DIR}"
echo "[config] log_file=${LOG_FILE}"
echo "[train] model=${MODEL_PATH} data=${PREPARED_DATA} run_id=${RUN_ID}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  finetune.py \
  --task commonsense \
  --method "${METHOD}" \
  --model_name_or_path "${MODEL_PATH}" \
  --prepared_data "${PREPARED_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --run_id "${RUN_ID}" \
  --learning_rate "${LEARNING_RATE}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --seed "${SEED}"

FINAL_MODEL="${OUTPUT_DIR}/final_model"
if [[ ! -f "${FINAL_MODEL}/config.json" ]]; then
  echo "Training completed but final HF model is missing: ${FINAL_MODEL}" >&2
  exit 1
fi

echo "[eval] model=${FINAL_MODEL} results=${RESULT_DIR}"
"${PYTHON_BIN}" scripts/eval_distributed.py \
  --task commonsense \
  --model_path "${FINAL_MODEL}" \
  --result_dir "${RESULT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --train_log "${LOG_FILE}" \
  --gpus "${EVAL_GPUS:-0,1}" \
  --python "${PYTHON_BIN}" \
  --seed "${SEED}"

echo "Completed. Summary: ${RESULT_DIR}/summary.json"
