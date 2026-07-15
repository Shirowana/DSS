#!/bin/bash
# Backward-compatible shell entrypoint. The maintained script lives in scripts/.

set -euo pipefail
exec "$(dirname "$0")/scripts/train_eval_llm_adapters.sh" "$@"
