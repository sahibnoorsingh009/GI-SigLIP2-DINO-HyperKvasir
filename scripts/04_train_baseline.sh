#!/usr/bin/env bash
set -euo pipefail
python -m src.train_finetune --config configs/baseline_siglip2.yaml
