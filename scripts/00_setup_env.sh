#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
PY
