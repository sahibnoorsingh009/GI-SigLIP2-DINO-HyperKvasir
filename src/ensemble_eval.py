import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ShardedEndoscopyDataset, load_labels
from src.models.siglip_classifier import SigLIPClassifier
from src.utils.metrics import save_reports


def load_model(ckpt_path, cfg, labels, ds, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SigLIPClassifier(
        cfg["model"]["name"],
        len(labels),
        False,
        cfg["model"].get("use_hierarchy", False),
        len(ds.organ_to_id),
        len(ds.category_to_id),
        len(ds.family_to_id),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpts", nargs="+", required=True)
    args = parser.parse_args()

    cfg = {
        "paths": {
            "data_root": "/workspace/data/gi-endoscopy-megabank-stage2",
            "split_csv": "metadata/hyperkvasir_23class_official_70_15_15_split.csv",
            "label_list": "results/label_list.txt",
        },
        "model": {
            "name": "google/siglip2-base-patch16-224",
            "image_size": 224,
            "use_hierarchy": False,
        },
    }

    labels, label_to_id = load_labels(cfg["paths"]["label_list"])

    ds = ShardedEndoscopyDataset(
        cfg["paths"]["split_csv"],
        cfg["paths"]["data_root"],
        label_to_id,
        args.split,
        False,
        cfg["model"]["image_size"],
        False,
    )

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        load_model(ckpt, cfg, labels, ds, device)
        for ckpt in args.ckpts
    ]

    ys, ps, probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ensemble-{args.split}"):
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)

            logits_sum = None

            for model in models:
                logits = model(x)["logits"].float()

                if args.tta:
                    logits_flip = model(torch.flip(x, dims=[3]))["logits"].float()
                    logits = (logits + logits_flip) / 2.0

                logits_sum = logits if logits_sum is None else logits_sum + logits

            logits_avg = logits_sum / len(models)
            pr = torch.softmax(logits_avg, dim=-1)

            ys.extend(y.cpu().tolist())
            ps.extend(pr.argmax(-1).cpu().tolist())
            probs.append(pr.cpu())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = save_reports(
        ys,
        ps,
        torch.cat(probs).numpy(),
        labels,
        out_dir,
        args.split,
    )

    print(metrics)


if __name__ == "__main__":
    main()
