#!/bin/bash

# Deprecated compatibility entrypoint.
# The active commonsense training path is now /root/code/DSS/run_dss.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

echo "[deprecated] scripts/train_commonsense.sh now forwards to ${PROJECT_ROOT}/run_dss.sh" >&2
exec bash "${PROJECT_ROOT}/run_dss.sh" "$@"
