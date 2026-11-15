#!/usr/bin/env bash
# Reproduce everything from raw downloads. Stops at the first failing step.
# Usage: bash run_all.sh   (set PYTHON=... to use a specific interpreter)
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

SCRIPTS=(
  01_download_claims.py
  02_download_policies.py
  03_download_storms.py
  04_clean_claims.py
  05_match_narratives.py
  06_embed_text.py
  07_train_models.py
  08_explain.py
  09_frequency.py
  10_shortfall.py
  11_ian_replay.py
)

for script in "${SCRIPTS[@]}"; do
  echo "==> src/${script}"
  "$PYTHON" "src/${script}"
done

echo "==> All steps finished."
