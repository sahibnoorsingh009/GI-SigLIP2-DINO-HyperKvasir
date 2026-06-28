#!/usr/bin/env bash
set -euo pipefail
python -m src.data.prepare_metadata --data-root /workspace/data/gi-endoscopy-megabank-stage2 --metadata-csv /workspace/data/gi-endoscopy-megabank-stage2/metadata/metadata_gi_endoscopy_megabank_stage2_all_sharded.csv
