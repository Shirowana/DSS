#!/bin/bash
# Download and verify the official GSM8K train/test JSONL files.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"/root/datasets/official/gsm8k"}

TRAIN_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
TEST_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

TRAIN_SHA256="17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
TEST_SHA256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"

mkdir -p "${DATA_ROOT}"

curl -L --fail -o "${DATA_ROOT}/train.jsonl" "${TRAIN_URL}"
curl -L --fail -o "${DATA_ROOT}/test.jsonl" "${TEST_URL}"

cat > "${DATA_ROOT}/SHA256SUMS" <<EOF
${TRAIN_SHA256}  train.jsonl
${TEST_SHA256}  test.jsonl
EOF

(
    cd "${DATA_ROOT}"
    sha256sum -c SHA256SUMS
)

wc -l "${DATA_ROOT}/train.jsonl" "${DATA_ROOT}/test.jsonl"
echo "Official GSM8K downloaded and verified at ${DATA_ROOT}"
