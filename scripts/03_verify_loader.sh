#!/usr/bin/env bash
set -euo pipefail
python -m src.data.verify_loader --data-root /workspace/data/gi-endoscopy-megabank-stage2 --split-csv metadata/hyperkvasir_23class_official_70_15_15_split.csv --n 24
