import argparse, io, json, tarfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)


class TarImageDataset(Dataset):
    def __init__(self, split_csv, data_root, split, label_to_id, image_size=224):
        self.df = pd.read_csv(split_csv)
        self.df = self.df[self.df["split"].eq(split)].reset_index(drop=True)
        self.data_root = Path(data_root)
        self.label_to_id = label_to_id
        self.tar_cache = {}

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def _resolve_shard(self, shard_file):
        p = Path(str(shard_file))
        candidates = [
            self.data_root / p,
            self.data_root / "data" / p,
            self.data_root / "data" / "loose_image_shards" / p.name,
            self.data_root / "data" / "kvasir_capsule_video_frame_shards" / p.name,
            self.data_root / p.name,
        ]
        for c in candidates:
            if c.exists():
                return c
        hits = list(self.data_root.rglob(p.name))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"Shard not found: {shard_file}")

    def _get_tar(self, shard_path):
        key = str(shard_path)
        cached = self.tar_cache.get(key)
        if cached is None:
            tf = tarfile.open(key, "r")
            members = tf.getnames()
            basename_map = {Path(x).name: x for x in members}
            cached = (tf, set(members), basename_map)
            self.tar_cache[key] = cached
        return cached

    def _load_image(self, row):
        shard = self._resolve_shard(row["shard_file"])
        member = str(row["path_in_shard"])
        tf, members, basename_map = self._get_tar(shard)

        candidates = [member, Path(member).name, "./" + member]
        if Path(member).name in basename_map:
            candidates.append(basename_map[Path(member).name])

        for cand in candidates:
            if cand in members:
                f = tf.extractfile(cand)
                return Image.open(io.BytesIO(f.read())).convert("RGB")

        base = Path(member).name
        for name in members:
            if name.endswith(base):
                f = tf.extractfile(name)
                return Image.open(io.BytesIO(f.read())).convert("RGB")

        raise KeyError(f"Missing member: {member}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row)
        x = self.tf(img)
        y = self.label_to_id[row["label"]]
        image_id = row["image_id"] if "image_id" in row else str(idx)
        return x, torch.tensor(y).long(), str(image_id)


class SSLFineTuneClassifier(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.vision_model = self.backbone.vision_model if hasattr(self.backbone, "vision_model") else self.backbone
        hidden = getattr(self.vision_model.config, "hidden_size", 768)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out = self.vision_model(pixel_values=x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state.mean(dim=1)
        return self.head(feat)


def compute_metrics(y_true, y_pred):
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def load_model(ckpt_path, model_name, n_classes, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SSLFineTuneClassifier(model_name, n_classes)

    if "model" not in ckpt:
        raise ValueError(f"Checkpoint does not contain 'model': {ckpt_path}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="metadata/hyperkvasir_23class_official_70_15_15_split.csv")
    ap.add_argument("--data_root", default="/workspace/data/gi-endoscopy-megabank-stage2")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--model_name", default="google/siglip2-base-patch16-224")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    labels = sorted(df["label"].dropna().unique().tolist())
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    ds = TarImageDataset(args.split_csv, args.data_root, args.split, label_to_id, args.image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        load_model(p, args.model_name, len(labels), device)
        for p in args.checkpoints
    ]

    y_true, y_pred, image_ids = [], [], []

    for x, y, ids in tqdm(loader, desc=f"ensemble {args.split}"):
        x = x.to(device, non_blocking=True)

        logits_sum = None
        for model in models:
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        logits_mean = logits_sum / len(models)
        pred = logits_mean.argmax(dim=1).cpu().numpy().tolist()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred)
        image_ids.extend(list(ids))

    metrics = compute_metrics(y_true, y_pred)

    with open(out_dir / f"{args.split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(out_dir / f"{args.split}_metrics.csv", index=False)

    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "y_true_id": y_true,
        "y_pred_id": y_pred,
        "y_true": [id_to_label[i] for i in y_true],
        "y_pred": [id_to_label[i] for i in y_pred],
    })
    pred_df.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(out_dir / f"{args.split}_per_class_metrics.csv")

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"{args.split}_confusion_matrix.csv")

    print(metrics)


if __name__ == "__main__":
    main()
