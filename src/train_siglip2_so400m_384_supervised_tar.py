import os
import io
import json
import math
import tarfile
import random
import argparse
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from transformers import AutoModel, get_cosine_schedule_with_warmup

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

from tqdm import tqdm


class TarLRUCache:
    def __init__(self, max_open=4):
        self.max_open = max_open
        self.cache = OrderedDict()

    def get(self, tar_path):
        if tar_path in self.cache:
            tar = self.cache.pop(tar_path)
            self.cache[tar_path] = tar
            return tar

        if len(self.cache) >= self.max_open:
            _, old_tar = self.cache.popitem(last=False)
            old_tar.close()

        tar = tarfile.open(tar_path, "r")
        self.cache[tar_path] = tar
        return tar

    def close(self):
        for _, tar in self.cache.items():
            tar.close()
        self.cache.clear()


class HyperKvasirTarDataset(Dataset):
    def __init__(self, df, data_root, label2id, image_size=384, split="train"):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.label2id = label2id
        self.split = split
        self.tar_cache = None

        self.shard_root_candidates = [
            os.path.join(data_root, "data", "loose_image_shards"),
            os.path.join(data_root, "data"),
            data_root,
        ]

        self.transform = self._build_transform(image_size, split)

    def _build_transform(self, image_size, split):
        if split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.90, 1.10),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=10,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    fill=0,
                ),
                transforms.ColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.06,
                    hue=0.02,
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])

        return transforms.Compose([
            transforms.Resize(
                int(image_size * 1.15),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def _resolve_tar_path(self, shard_file):
        shard_file = str(shard_file)

        if os.path.isabs(shard_file) and os.path.exists(shard_file):
            return shard_file

        for root in self.shard_root_candidates:
            p = os.path.join(root, shard_file)
            if os.path.exists(p):
                return p

        raise FileNotFoundError(
            f"Could not find shard {shard_file}. Tried: "
            + ", ".join(os.path.join(r, shard_file) for r in self.shard_root_candidates)
        )

    def __getitem__(self, idx):
        if self.tar_cache is None:
            self.tar_cache = TarLRUCache(max_open=4)

        row = self.df.iloc[idx]

        shard_file = row["shard_file"]
        path_in_shard = row["path_in_shard"]
        label_name = str(row["label"])
        image_id = str(row["image_id"])

        tar_path = self._resolve_tar_path(shard_file)
        tar = self.tar_cache.get(tar_path)

        try:
            f = tar.extractfile(str(path_in_shard))
            if f is None:
                raise FileNotFoundError(path_in_shard)
            img_bytes = f.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed loading image_id={image_id}, shard={tar_path}, path={path_in_shard}: {e}"
            )

        image = self.transform(image)
        label = self.label2id[label_name]

        return image, torch.tensor(label, dtype=torch.long), image_id


class SigLIP2Classifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.vision_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pixel_values):
        out = self.backbone.vision_model(pixel_values=pixel_values)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state[:, 0]

        return self.classifier(feats)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model)
        self.decay = decay
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()

        for k in esd.keys():
            if k not in msd:
                continue
            if esd[k].dtype.is_floating_point:
                esd[k].copy_(esd[k] * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                esd[k].copy_(msd[k])

    def to(self, device):
        self.ema.to(device)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []

    for i in range(num_classes):
        c = counts[i]
        w = total / (num_classes * max(c, 1))
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = torch.clamp(weights, min=0.25, max=5.0)
    return weights


def build_sampler(labels):
    counts = Counter(labels)
    sample_weights = [1.0 / math.sqrt(counts[y]) for y in labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_param_groups(model, backbone_lr, head_lr, weight_decay):
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(p)
        else:
            backbone_params.append(p)

    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


@torch.no_grad()
def evaluate(model, loader, device, id2label, use_amp=True, out_dir=None, prefix="val"):
    model.eval()

    all_logits = []
    all_targets = []
    all_ids = []

    for images, targets, image_ids in tqdm(loader, desc=f"Evaluating {prefix}", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)

        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
        all_ids.extend(list(image_ids))

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "micro_f1": float(f1_score(targets, preds, average="micro")),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, preds)),
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        labels_sorted = list(range(len(id2label)))
        target_names = [id2label[i] for i in labels_sorted]

        report = classification_report(
            targets,
            preds,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )

        with open(os.path.join(out_dir, f"{prefix}_classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        cm = confusion_matrix(targets, preds, labels=labels_sorted)
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
            os.path.join(out_dir, f"{prefix}_confusion_matrix.csv")
        )

        top3 = np.argsort(-probs, axis=1)[:, :3]
        rows = []

        for i in range(len(targets)):
            rows.append({
                "image_id": all_ids[i],
                "true_id": int(targets[i]),
                "true_label": id2label[int(targets[i])],
                "pred_id": int(preds[i]),
                "pred_label": id2label[int(preds[i])],
                "correct": bool(preds[i] == targets[i]),
                "top1_label": id2label[int(top3[i, 0])],
                "top1_prob": float(probs[i, top3[i, 0]]),
                "top2_label": id2label[int(top3[i, 1])],
                "top2_prob": float(probs[i, top3[i, 1]]),
                "top3_label": id2label[int(top3[i, 2])],
                "top3_prob": float(probs[i, top3[i, 2]]),
            })

        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"{prefix}_predictions.csv"),
            index=False,
        )

    return metrics


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, best_metric, label2id, id2label, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "label2id": label2id,
        "id2label": id2label,
        "args": vars(args),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    if ema is not None:
        ckpt["ema_model"] = ema.ema.state_dict()

    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--loss", default="weighted_ce", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=1.5)

    parser.add_argument("--use_class_balanced_sampler", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--save_metric", default="micro_f1", choices=["micro_f1", "macro_f1", "mcc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.split_csv)

    required_cols = ["split", "label", "shard_file", "path_in_shard", "image_id"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
    val_df = df[df["split"].astype(str).str.lower() == "val"].copy()

    labels = sorted(train_df["label"].astype(str).unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Classes:", len(labels))
    print(labels)

    with open(os.path.join(args.out_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    with open(os.path.join(args.out_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=2)

    train_targets = [label2id[str(x)] for x in train_df["label"].tolist()]
    val_targets = [label2id[str(x)] for x in val_df["label"].tolist()]

    train_ds = HyperKvasirTarDataset(
        train_df,
        data_root=args.data_root,
        label2id=label2id,
        image_size=args.image_size,
        split="train",
    )

    val_ds = HyperKvasirTarDataset(
        val_df,
        data_root=args.data_root,
        label2id=label2id,
        image_size=args.image_size,
        split="val",
    )

    sampler = None
    shuffle = True

    if args.use_class_balanced_sampler:
        sampler = build_sampler(train_targets)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
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

    model = SigLIP2Classifier(args.model_name, num_classes=len(labels), dropout=0.1)
    model.to(device)

    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay)
        ema.to(device)

    class_weights = None
    if args.loss in ["weighted_ce", "focal"]:
        class_weights = compute_class_weights(train_targets, len(labels)).to(device)
        print("Class weights:", class_weights.detach().cpu().tolist())

    if args.loss in ["ce", "weighted_ce"]:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights if args.loss == "weighted_ce" else None,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = FocalLoss(
            weight=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )

    optimizer = torch.optim.AdamW(
        make_param_groups(model, args.backbone_lr, args.head_lr, args.weight_decay)
    )

    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = args.precision == "bf16"
    best_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, (images, targets, _) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
                loss = loss / args.grad_accum

            loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

            running += loss.item() * args.grad_accum

            pbar.set_postfix(
                loss=running / step,
                lr=optimizer.param_groups[0]["lr"],
            )

        eval_model = ema.ema if ema is not None else model

        val_metrics = evaluate(
            eval_model,
            val_loader,
            device,
            id2label,
            use_amp=use_amp,
            out_dir=args.out_dir,
            prefix=f"val_epoch_{epoch}",
        )

        record = {
            "epoch": epoch,
            "train_loss": float(running / len(train_loader)),
            **val_metrics,
        }

        history.append(record)

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(json.dumps(record, indent=2))

        selected = val_metrics[args.save_metric]

        if selected > best_metric:
            best_metric = selected

            save_checkpoint(
                os.path.join(args.out_dir, "best.pt"),
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                label2id,
                id2label,
                args,
            )

            with open(os.path.join(args.out_dir, "best_metrics.json"), "w") as f:
                json.dump(record, f, indent=2)

            print(f"Saved best checkpoint: {args.save_metric}={best_metric:.6f}")

        save_checkpoint(
            os.path.join(args.out_dir, "last.pt"),
            model,
            ema,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            label2id,
            id2label,
            args,
        )

    print("Done.")
    print("Best", args.save_metric, best_metric)


if __name__ == "__main__":
    main()
