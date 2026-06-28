#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/data
python scripts/download_dataset.py --repo-id Sahibnoor1/gi-endoscopy-megabank-stage2 --local-dir /workspace/data/gi-endoscopy-megabank-stage2
