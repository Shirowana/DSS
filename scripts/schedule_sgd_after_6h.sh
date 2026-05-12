#!/bin/bash

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/root/code/DSS"}
WAIT_HOURS=${WAIT_HOURS:-6}
POLL_SECONDS=${POLL_SECONDS:-600}
NUM_GPUS=${NUM_GPUS:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
MASTER_PORT=${MASTER_PORT:-29502}
LOG_ROOT=${LOG_ROOT:-"${PROJECT_ROOT}/logs_commonsense"}
SCHED_LOG=${SCHED_LOG:-"${LOG_ROOT}/schedule_sgd_after_6h.log"}
TRAIN_SESSION=${TRAIN_SESSION:-"train_sgd"}
QUANTILE_LR=${QUANTILE_LR:-0.03}
QUANTILE_ALPHA=${QUANTILE_ALPHA:-0.0}

mkdir -p "${LOG_ROOT}"

is_gpu_busy() {
    local gpu_apps
    gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    if [[ -n "${gpu_apps}" ]]; then
        return 0
    fi
    if pgrep -af "python .*finetune_commonsense.py|python -m torch.distributed.run .*finetune_commonsense.py" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "${SCHED_LOG}"
}

launch_sgd_run() {
    local timestamp
    timestamp=$(date -u '+%Y%m%d_%H%M%S')
    local run_name="commonsense_Llama3-8B_dss_sgd_qlr${QUANTILE_LR}_nf180000_cand30000_gs3_${timestamp}"
    log "GPUs are idle. Launching SGD-threshold experiment in tmux session '${TRAIN_SESSION}'."
    tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null && tmux kill-session -t "${TRAIN_SESSION}" || true
    tmux new-session -d -s "${TRAIN_SESSION}" \
        "cd ${PROJECT_ROOT} && \
         NUM_GPUS=${NUM_GPUS} \
         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
         NCCL_P2P_DISABLE=0 \
         MASTER_PORT=${MASTER_PORT} \
         THRESHOLD_MODE=sgd \
         QUANTILE_LR=${QUANTILE_LR} \
         QUANTILE_ALPHA=${QUANTILE_ALPHA} \
         RUN_NAME=${run_name} \
         bash ${PROJECT_ROOT}/run_dss.sh"
    log "Started tmux session '${TRAIN_SESSION}' with RUN_NAME=${run_name}."
}

log "Scheduler armed. Waiting ${WAIT_HOURS} hours before checking GPU availability."
log "Current time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
log "Target time: $(date -u -d "+${WAIT_HOURS} hours" '+%Y-%m-%d %H:%M:%S UTC')"
log "Planned SGD threshold config: THRESHOLD_MODE=sgd, QUANTILE_LR=${QUANTILE_LR}, QUANTILE_ALPHA=${QUANTILE_ALPHA}"

sleep "$((WAIT_HOURS * 3600))"

log "Initial wait finished. Checking GPU availability."
while true; do
    if is_gpu_busy; then
        log "GPU or training process still busy. Sleeping ${POLL_SECONDS} seconds before retry."
        sleep "${POLL_SECONDS}"
        continue
    fi
    launch_sgd_run
    break
done
