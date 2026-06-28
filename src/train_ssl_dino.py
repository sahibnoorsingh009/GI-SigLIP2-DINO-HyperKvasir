import argparse
import copy
import io
import json
import math
import random
import tarfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import AutoModel


class SSLTarDataset(Dataset):
    def __init__(self, meta_csv, data_root, image_size=224, max_rows=None):
        self.df = pd.read_csv(meta_csv, low_memory=False)

        need = ["shard_file", "path_in_shard"]
        for c in need:
            if c not in self.df.columns:
                raise ValueError(f"Missing column {c}. Columns={self.df.columns.tolist()}")

        self.df = self.df[self.df["shard_file"].notna() & self.df["path_in_shard"].notna()].copy()
        self.df = self.df.reset_index(drop=True)

        if max_rows is not None and max_rows > 0 and len(self.df) > max_rows:
            self.df = self.df.sample(max_rows, random_state=42).reset_index(drop=True)

        self.data_root = Path(data_root)
        self.tar_cache = {}

        self.t1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.45, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.t2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.2))], p=0.20),
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
        shard_path = str(shard_path)
        tf = self.tar_cache.get(shard_path)
        if tf is None:
            tf = tarfile.open(shard_path, "r")
            self.tar_cache[shard_path] = tf
        return tf

    def _load_image(self, row):
        shard = self._resolve_shard(row["shard_file"])
        member = str(row["path_in_shard"])

        tf = self._get_tar(shard)

        try:
            f = tf.extractfile(member)
        except KeyError:
            # fallback: try basename
            f = tf.extractfile(Path(member).name)

        img = Image.open(io.BytesIO(f.read())).convert("RGB")
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row)

        return {
            "view1": self.t1(img),
            "view2": self.t2(img),
        }


class DINOHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        return self.net(x)


class SigLIPVisionDINO(nn.Module):
    def __init__(self, model_name, out_dim=4096):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        if hasattr(self.backbone, "vision_model"):
            self.vision_model = self.backbone.vision_model
        else:
            self.vision_model = self.backbone

        hidden = 768
        if hasattr(self.vision_model, "config"):
            hidden = getattr(self.vision_model.config, "hidden_size", 768)

        self.head = DINOHead(hidden, 2048, out_dim)

    def features(self, x):
        out = self.vision_model(pixel_values=x)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state[:, 0]
        else:
            feat = out[0][:, 0]

        return feat

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outs, teacher_outs):
        student_outs = [s / self.student_temp for s in student_outs]

        with torch.no_grad():
            teacher_probs = [
                F.softmax((t - self.center) / self.teacher_temp, dim=-1)
                for t in teacher_outs
            ]

        total_loss = 0.0
        n_terms = 0

        for iq, q in enumerate(teacher_probs):
            for v, s in enumerate(student_outs):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1).mean()
                total_loss += loss
                n_terms += 1

        total_loss = total_loss / max(n_terms, 1)

        with torch.no_grad():
            batch_center = torch.cat(teacher_outs, dim=0).mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total_loss


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def cosine_momentum(step, max_steps, base=0.996, final=1.0):
    return final - (final - base) * (math.cos(math.pi * step / max_steps) + 1) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_csv", default="metadata/metadata_gi_endoscopy_megabank_stage2_all_sharded.csv")
    parser.add_argument("--data_root", default="/workspace/data/gi-endoscopy-megabank-stage2")
    parser.add_argument("--model_name", default="google/siglip2-base-patch16-224")
    parser.add_argument("--out_dir", default="checkpoints/dino_siglip2_224_ssl")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--save_every", type=int, default=2500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SSLTarDataset(
        args.meta_csv,
        args.data_root,
        image_size=args.image_size,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    student = SigLIPVisionDINO(args.model_name).to(device)
    teacher = copy.deepcopy(student).to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    criterion = DINOLoss(out_dim=4096).to(device)

    opt = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    step = 0
    best_loss = 999999.0

    pbar = tqdm(total=args.max_steps, desc="DINO-SSL")

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            x1 = batch["view1"].to(device, non_blocking=True)
            x2 = batch["view2"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                s1 = student(x1)
                s2 = student(x2)

                with torch.no_grad():
                    t1 = teacher(x1)
                    t2 = teacher(x2)

                loss = criterion([s1, s2], [t1, t2])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            mom = cosine_momentum(step, args.max_steps)
            update_teacher(student, teacher, mom)

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.item()), mom=round(mom, 6))

            if loss.item() < best_loss:
                best_loss = float(loss.item())
                torch.save(
                    {
                        "step": step,
                        "student": student.state_dict(),
                        "teacher": teacher.state_dict(),
                        "best_loss": best_loss,
                        "model_name": args.model_name,
                    },
                    out_dir / "best_ssl.pt",
                )

            if step % args.save_every == 0:
                torch.save(
                    {
                        "step": step,
                        "student": student.state_dict(),
                        "teacher": teacher.state_dict(),
                        "loss": float(loss.item()),
                        "model_name": args.model_name,
                    },
                    out_dir / f"ssl_step_{step}.pt",
                )

                with open(out_dir / "progress.json", "w") as f:
                    json.dump({"step": step, "loss": float(loss.item()), "best_loss": best_loss}, f, indent=2)

    pbar.close()

    torch.save(
        {
            "step": step,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "best_loss": best_loss,
            "model_name": args.model_name,
        },
        out_dir / "final_ssl.pt",
    )

    print("DONE SSL")
    print("best_loss:", best_loss)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
