# GI-SigLIP2-DINO-HyperKvasir++

Runnable project scaffold for the GI-Endoscopy MegaBank Stage 2 plan.

First goal: build the official HyperKvasir 23-class benchmark pipeline correctly, then train `google/siglip2-base-patch16-224` with imbalance-aware fine-tuning. SSL is kept as the later stage; do **not** spend GPU money on SSL until metadata, tar loading, split, metrics, and supervised baseline work.

## Quick start on RunPod

```bash
cd /workspace
unzip GI-SigLIP2-DINO-HyperKvasir.zip
cd GI-SigLIP2-DINO-HyperKvasir
bash scripts/00_setup_env.sh
huggingface-cli login
bash scripts/01_download_dataset.sh
bash scripts/02_prepare_metadata.sh
bash scripts/03_verify_loader.sh
bash scripts/04_train_baseline.sh
bash scripts/05_train_hierarchy.sh
bash scripts/06_eval_best.sh
```

Expected HF dataset: `Sahibnoor1/gi-endoscopy-megabank-stage2`.

Expected metadata: `metadata/metadata_gi_endoscopy_megabank_stage2_all_sharded.csv`.

Official benchmark subset must use only original labelled HyperKvasir still images: 10,619 images, 23 classes.

## Outputs

- `metadata/hyperkvasir_23class_official_70_15_15_split.csv`
- `results/split_summary.csv`
- `results/per_class_split_counts.csv`
- `results/label_list.txt`
- `results/official_hk_23_class_counts.csv`
- `checkpoints/*/best.pt`
- `results/final_eval/*`
