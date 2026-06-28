#!/usr/bin/env bash
set -euo pipefail
python -m src.eval_classification --config configs/eval.yaml
