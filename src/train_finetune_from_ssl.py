import argparse, io, tarfile, json, random
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)


class TarImageDataset(Dataset):
    def __init__(self, split_csv, data_root, split, label_to_id, image_size=224, train=False):
        self.df = pd.read_csv(split_csv)
        self.df = self.df[self.df["split"].eq(split)].reset_index(drop=True)
        self.data_root = Path(data_root)
        self.label_to_id = label_to_id
        self.train = train
        self.tar_cache = {}

        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.015),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
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

        candidates = [
            member,
            Path(member).name,
            "./" + member,
        ]

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
        return x, torch.tensor(y).long()


class SSLFineTuneClassifier(nn.Module):
    def __init__(self, model_name, n_classes, ssl_ckpt=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.vision_model = self.backbone.vision_model if hasattr(self.backbone, "vision_model") else self.backbone

        hidden = getattr(self.vision_model.config, "hidden_size", 768)
        self.head = nn.Linear(hidden, n_classes)

        if ssl_ckpt is not None:
            ckpt = torch.load(ssl_ckpt, map_location="cpu")
            sd = ckpt["student"]

            vision_sd = {}
            for k, v in sd.items():
                if k.startswith("vision_model."):
                    vision_sd[k.replace("vision_model.", "")] = v
                elif k.startswith("backbone.vision_model."):
                    vision_sd[k.replace("backbone.vision_model.", "")] = v

            missing, unexpected = self.vision_model.load_state_dict(vision_sd, strict=False)
            print("Loaded SSL vision weights")
            print("Missing keys:", len(missing))
            print("Unexpected keys:", len(unexpected))

    def forward(self, x):
        out = self.vision_model(pixel_values=x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state.mean(dim=1)
        return self.head(feat)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "mcc": mcc,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="eval"):
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        ys.extend(y.numpy().tolist())
        ps.extend(pred)
    return compute_metrics(ys, ps), ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="metadata/hyperkvasir_23class_official_70_15_15_split.csv")
    ap.add_argument("--data_root", default="/workspace/data/gi-endoscopy-megabank-stage2")
    ap.add_argument("--ssl_ckpt", required=True)
    ap.add_argument("--out_dir", default="checkpoints/finetune_from_ssl_10k")
    ap.add_argument("--model_name", default="google/siglip2-base-patch16-224")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr_encoder", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    labels = sorted(df["label"].dropna().unique().tolist())
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    with open(out_dir / "label_to_id.json", "w") as f:
        json.dump(label_to_id, f, indent=2)

    train_ds = TarImageDataset(args.split_csv, args.data_root, "train", label_to_id, args.image_size, train=True)
    val_ds = TarImageDataset(args.split_csv, args.data_root, "val", label_to_id, args.image_size, train=False)

    train_labels = train_ds.df["label"].map(label_to_id).values
    counts = np.bincount(train_labels, minlength=len(labels))
    class_weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SSLFineTuneClassifier(
        args.model_name,
        n_classes=len(labels),
        ssl_ckpt=args.ssl_ckpt,
    ).to(device)

    ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW([
        {"params": model.vision_model.parameters(), "lr": args.lr_encoder},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.04)

    best_macro = -1
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y in tqdm(train_loader, desc=f"train epoch {epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = loss_fn(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = float(np.mean(losses))

        print("VAL:", val_metrics)

        with open(out_dir / "last_val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            best_metrics = val_metrics
            torch.save({
                "model": model.state_dict(),
                "labels": labels,
                "label_to_id": label_to_id,
                "ssl_ckpt": args.ssl_ckpt,
                "val_metrics": val_metrics,
            }, out_dir / "best.pt")

            with open(out_dir / "best_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)

            print("NEW BEST:", best_metrics)

    print("DONE")
    print("BEST:", best_metrics)


if __name__ == "__main__":
    main()
