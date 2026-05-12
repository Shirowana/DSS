#!/bin/bash
# Train-only wrapper for DSS experiments.
#
# This script is intentionally thin: it performs a couple of prechecks and
# then delegates to run_dss.sh, which already contains the actual training
# entrypoint and DDP launch logic.

set -euo pipefail

REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-"/root/code/DSS"}
DATASET_PATH=${DATASET_PATH:-"/root/datasets/commonsense_new/train_all_256_OnlyOutput_Llama3-8B"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_PROJECT_ROOT}/logs_commonsense"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Missing DATASET_PATH=${DATASET_PATH}" >&2
    exit 1
fi

python - <<'PY' "${DATASET_PATH}"
import sys
from datasets import load_from_disk

path = sys.argv[1]
ds = load_from_disk(path)
if "task_name" not in ds.column_names:
    raise SystemExit(f"Dataset at {path} is missing required task_name column for init experiments.")
print(f"[train_only] dataset ok: {path}")
PY

mkdir -p "${LOG_ROOT}"
cd "${REMOTE_PROJECT_ROOT}"
exec bash "${REMOTE_PROJECT_ROOT}/run_dss.sh"
