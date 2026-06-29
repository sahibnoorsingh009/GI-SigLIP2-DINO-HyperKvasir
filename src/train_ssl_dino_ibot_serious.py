import argparse, copy, io, json, math, tarfile, random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoModel


class SeriousMegaBankSSL(Dataset):
    def __init__(self, meta_csv, data_root, image_size=224):
        self.df = pd.read_csv(meta_csv, low_memory=False)
        self.df = self.df[self.df["shard_file"].notna() & self.df["path_in_shard"].notna()].copy()
        self.df = self.df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.tar_cache = {}

        if "origin" not in self.df.columns:
            self.df["origin"] = "unknown"

        group_counts = self.df["origin"].value_counts().to_dict()
        self.sample_weights = self.df["origin"].map(lambda x: 1.0 / group_counts.get(x, 1)).values

        self.global_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.40, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.015),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.local_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 0.40)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.015),
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
        key = str(shard_path)
        cached = self.tar_cache.get(key)
        if cached is None:
            tf = tarfile.open(key, "r")
            members = tf.getnames()
            basename_map = {}
            for name in members:
                basename_map[Path(name).name] = name
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
            "images/" + Path(member).name,
            "data/" + Path(member).name,
        ]

        # If tar stores paths like folder/KCIMG_xxx.jpg, use basename map.
        if Path(member).name in basename_map:
            candidates.append(basename_map[Path(member).name])

        for cand in candidates:
            if cand in members:
                f = tf.extractfile(cand)
                return Image.open(io.BytesIO(f.read())).convert("RGB")

        # Last fallback: suffix search in this tar.
        base = Path(member).name
        for name in members:
            if name.endswith(base):
                f = tf.extractfile(name)
                return Image.open(io.BytesIO(f.read())).convert("RGB")

        raise KeyError(f"Member not found in shard={shard}: {member}")

    def __getitem__(self, idx):
        # SSL can safely skip rare corrupted/mismatched rows.
        # Try the requested row first, then random alternatives.
        last_error = None
        for attempt in range(20):
            try:
                real_idx = idx if attempt == 0 else random.randint(0, len(self.df) - 1)
                row = self.df.iloc[real_idx]
                img = self._load_image(row)
                return {
                    "g1": self.global_tf(img),
                    "g2": self.global_tf(img),
                    "l1": self.local_tf(img),
                    "l2": self.local_tf(img),
                }
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"Failed to load image after 20 attempts. Last error: {last_error}")


class MLPHead(nn.Module):
    def __init__(self, in_dim=768, hidden=2048, out_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(F.normalize(x, dim=-1))


class SigLIPDINOiBOT(nn.Module):
    def __init__(self, model_name, dino_dim=4096, patch_dim=768):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.vision_model = self.backbone.vision_model if hasattr(self.backbone, "vision_model") else self.backbone

        hidden = 768
        if hasattr(self.vision_model, "config"):
            hidden = getattr(self.vision_model.config, "hidden_size", 768)

        self.dino_head = MLPHead(hidden, 2048, dino_dim)
        self.patch_head = nn.Linear(hidden, patch_dim)

    def forward(self, x, return_patches=False):
        out = self.vision_model(pixel_values=x, output_hidden_states=False)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state.mean(dim=1)

        dino = self.dino_head(pooled)

        if return_patches:
            patches = out.last_hidden_state
            patches = self.patch_head(patches)
            return dino, patches

        return dino


class DINOLoss(nn.Module):
    def __init__(self, dim=4096, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, dim))

    def forward(self, student_outs, teacher_outs):
        student_outs = [s / self.student_temp for s in student_outs]
        with torch.no_grad():
            teacher_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1) for t in teacher_outs]

        total, n = 0.0, 0
        for iq, q in enumerate(teacher_probs):
            for iv, s in enumerate(student_outs):
                if iv == iq:
                    continue
                total = total + torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1).mean()
                n += 1

        with torch.no_grad():
            batch_center = torch.cat(teacher_outs, dim=0).mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total / max(n, 1)


def ibot_patch_loss(student_patches, teacher_patches, mask_ratio=0.30):
    # iBOT-like masked patch distillation on global views.
    # We distill normalized teacher patch features to student patch features on random masked positions.
    B, N, D = student_patches.shape
    device = student_patches.device
    mask = torch.rand(B, N, device=device) < mask_ratio

    s = F.normalize(student_patches, dim=-1)
    t = F.normalize(teacher_patches.detach(), dim=-1)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    return F.mse_loss(s[mask], t[mask])


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def cosine_momentum(step, max_steps, base=0.996, final=1.0):
    return final - (final - base) * (math.cos(math.pi * step / max_steps) + 1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="checkpoints/dino_ibot_siglip2_224_SERIOUS")
    ap.add_argument("--model_name", default="google/siglip2-base-patch16-224")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.04)
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--ibot_weight", type=float, default=1.0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SeriousMegaBankSSL(args.meta_csv, args.data_root, args.image_size)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(ds.sample_weights),
        num_samples=len(ds),
        replacement=True,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    student = SigLIPDINOiBOT(args.model_name).to(device)
    teacher = copy.deepcopy(student).to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss_fn = DINOLoss(dim=4096).to(device)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    best_loss = 1e9

    if args.resume is not None:
        print("Resuming from:", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(ckpt["student"], strict=True)
        teacher.load_state_dict(ckpt["teacher"], strict=True)
        step = int(ckpt.get("step", 0))
        best_loss = float(ckpt.get("best_loss", ckpt.get("loss", 1e9)))
        print("Resumed step:", step)
        print("Resumed best_loss:", best_loss)

    pbar = tqdm(total=args.max_steps, initial=step, desc="SERIOUS-DINO-iBOT")

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            g1 = batch["g1"].to(device, non_blocking=True)
            g2 = batch["g2"].to(device, non_blocking=True)
            l1 = batch["l1"].to(device, non_blocking=True)
            l2 = batch["l2"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                s_g1, sp_g1 = student(g1, return_patches=True)
                s_g2, sp_g2 = student(g2, return_patches=True)
                s_l1 = student(l1, return_patches=False)
                s_l2 = student(l2, return_patches=False)

                with torch.no_grad():
                    t_g1, tp_g1 = teacher(g1, return_patches=True)
                    t_g2, tp_g2 = teacher(g2, return_patches=True)

                loss_dino = dino_loss_fn([s_g1, s_g2, s_l1, s_l2], [t_g1, t_g2])
                loss_ibot = 0.5 * (
                    ibot_patch_loss(sp_g1, tp_g1, 0.30) +
                    ibot_patch_loss(sp_g2, tp_g2, 0.30)
                )
                loss = loss_dino + args.ibot_weight * loss_ibot

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            mom = cosine_momentum(step, args.max_steps)
            update_teacher(student, teacher, mom)

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                total=float(loss.item()),
                dino=float(loss_dino.item()),
                ibot=float(loss_ibot.item()),
                mom=round(mom, 6),
            )

            if loss.item() < best_loss:
                best_loss = float(loss.item())
                torch.save({
                    "step": step,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "best_loss": best_loss,
                    "model_name": args.model_name,
                    "type": "dino_ibot_siglip2",
                }, out_dir / "best_ssl.pt")

            if step % args.save_every == 0:
                torch.save({
                    "step": step,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "loss": float(loss.item()),
                    "loss_dino": float(loss_dino.item()),
                    "loss_ibot": float(loss_ibot.item()),
                    "model_name": args.model_name,
                    "type": "dino_ibot_siglip2",
                }, out_dir / f"ssl_step_{step}.pt")

                with open(out_dir / "progress.json", "w") as f:
                    json.dump({
                        "step": step,
                        "loss": float(loss.item()),
                        "loss_dino": float(loss_dino.item()),
                        "loss_ibot": float(loss_ibot.item()),
                        "best_loss": best_loss,
                    }, f, indent=2)

    pbar.close()

    torch.save({
        "step": step,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "best_loss": best_loss,
        "model_name": args.model_name,
        "type": "dino_ibot_siglip2",
    }, out_dir / "final_ssl.pt")

    print("DONE SERIOUS DINO/iBOT SSL")
    print("saved:", out_dir)
    print("best_loss:", best_loss)


if __name__ == "__main__":
    main()
