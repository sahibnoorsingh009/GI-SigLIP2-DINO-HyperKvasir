#!/usr/bin/env bash
set -euo pipefail
python -m src.train_finetune --config configs/hierarchy_siglip2.yaml
