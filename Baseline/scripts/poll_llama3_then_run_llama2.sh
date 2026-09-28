#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA3_RUN_ID="commonsense_llama3_s2ft_20260923_seed42"
LLAMA3_MODEL_SLUG="Meta-Llama-3-8B"
LLAMA2_RUN_ID="commonsense_llama2_s2ft_20260923_lr1e4_seed42"
LLAMA3_LOG_DIR="${ROOT_DIR}/logs/${LLAMA3_RUN_ID}"
LLAMA3_TRAIN_SUMMARY="${ROOT_DIR}/outputs/s2ft/commonsense/${LLAMA3_MODEL_SLUG}/${LLAMA3_RUN_ID}/train_summary.json"
LLAMA3_FINAL_CONFIG="${ROOT_DIR}/outputs/s2ft/commonsense/${LLAMA3_MODEL_SLUG}/${LLAMA3_RUN_ID}/final_model/config.json"
LLAMA3_EVAL_SUMMARY="${ROOT_DIR}/results/s2ft/commonsense/${LLAMA3_MODEL_SLUG}/${LLAMA3_RUN_ID}/summary.json"
MONITOR_LOG="${ROOT_DIR}/logs/launch/poll_llama3_then_run_llama2.log"

mkdir -p "${ROOT_DIR}/logs/launch"
exec >>"${MONITOR_LOG}" 2>&1
echo "[$(date -u +%FT%TZ)] monitor started; waiting for complete Llama-3 train + eval"

while :; do
  if [[ -s "${LLAMA3_TRAIN_SUMMARY}" && -s "${LLAMA3_FINAL_CONFIG}" && -s "${LLAMA3_EVAL_SUMMARY}" ]]; then
    echo "[$(date -u +%FT%TZ)] Llama-3 S2FT train and evaluation are complete"
    break
  fi

  progress="unknown"
  if [[ -s "${LLAMA3_LOG_DIR}/trainer_log_history.json" ]]; then
    progress="trainer log saved"
  elif [[ -s "${LLAMA3_LOG_DIR}/train.log" ]]; then
    progress="$(tail -n 1 "${LLAMA3_LOG_DIR}/train.log" | tr '\n' ' ' | cut -c1-240)"
  fi
  echo "[$(date -u +%FT%TZ)] Llama-3 incomplete; ${progress}; checking again in 10 minutes"
  sleep 600
done

LLAMA2_RESULT="${ROOT_DIR}/results/s2ft/commonsense/Llama-2-7b-hf/${LLAMA2_RUN_ID}/summary.json"
LLAMA2_TRAIN_SUMMARY="${ROOT_DIR}/outputs/s2ft/commonsense/Llama-2-7b-hf/${LLAMA2_RUN_ID}/train_summary.json"
if [[ -s "${LLAMA2_TRAIN_SUMMARY}" && -s "${LLAMA2_RESULT}" ]]; then
  echo "[$(date -u +%FT%TZ)] Llama-2 run already has complete train and eval artifacts; nothing to launch"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] launching Llama-2 S2FT at lr=1e-4 with full train + eval"
cd "${ROOT_DIR}"
MODEL_FAMILY=llama2 \
METHOD=s2ft \
LEARNING_RATE=1e-4 \
NPROC_PER_NODE=2 \
SEED=42 \
RUN_ID="${LLAMA2_RUN_ID}" \
bash scripts/train_eval_commonsense.sh
echo "[$(date -u +%FT%TZ)] Llama-2 train + eval script completed"
