# almond_density_mnv3_small.py
# ------------------------------
# This is the main training script for the CNN-based almond counting model.
# 
#

import os
import json
import math
import csv
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from tqdm import tqdm

# -----------------------------
# Constants / Hyperparameters
# -----------------------------
@dataclass
class HYPERPARAMS:
    IMG_SIZE: int = 256
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LR: float = 1e-4
    WEIGHT_DECAY: float = 0.0
    
    SIGMA_PX: Optional[float] = None 
    SIGMA_MIN: float = 2.0
    SIGMA_MAX: float = 12.0
    SIGMA_SPACING_RATIO: float = 0.3 

    AUG_HFLIP_P: float = 0.5
    AUG_VFLIP_P: float = 0.2

    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    LOG_CSV: str = "training_log.csv"
    CKPT_DIR: str = "models"
    BEST_NAME: str = "almond_density_best.pth"
    LAST_NAME: str = "almond_density_last.pth"

HP = HYPERPARAMS()

# -----------------------------
# Utility: density map creation
# -----------------------------

def estimate_sigma_from_points(points: List[Tuple[float, float]], fallback: float = 8.0, ratio: float = HP.SIGMA_SPACING_RATIO) -> float:
    if len(points) < 2:
        return fallback
    P = np.array(points, dtype=np.float32)
    dists = []
    for i in range(len(P)):
        diffs = P - P[i]
        r2 = (diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        r2[i] = np.inf
        dists.append(float(np.sqrt(r2.min())))
    med = float(np.median(dists))
    sigma = max(HP.SIGMA_MIN, min(HP.SIGMA_MAX, ratio * med))
    return sigma


def gaussian2d_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian kernel whose discrete sum is 1.0."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    ksum = kernel.sum()
    if ksum > 0:
        kernel /= ksum
    return kernel.astype(np.float32)


def make_density_map(h: int, w: int, points: List[Tuple[float, float]], sigma_px: float) -> np.ndarray:
    """Place a normalized Gaussian at each point; ensures total sum == len(points)."""
    if len(points) == 0:
        return np.zeros((h, w), dtype=np.float32)

    k = max(3, int(math.ceil(sigma_px * 6)))
    if k % 2 == 0:
        k += 1
    g = gaussian2d_kernel(k, sigma_px)  # sum == 1.0

    dm = np.zeros((h, w), dtype=np.float32)
    r = k // 2

    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        x1 = max(0, xi - r); x2 = min(w, xi + r + 1)
        y1 = max(0, yi - r); y2 = min(h, yi + r + 1)

        gx1 = r - (xi - x1)
        gy1 = r - (yi - y1)
        gx2 = gx1 + (x2 - x1)
        gy2 = gy1 + (y2 - y1)

        dm[y1:y2, x1:x2] += g[gy1:gy2, gx1:gx2]

    s = dm.sum()
    if s > 0:
        dm *= (len(points) / s)
    return dm

# -----------------------------
# Dataset
# -----------------------------
class AlmondDensityDataset(Dataset):
    def __init__(self, images_dir: str, annotations_json: str, img_size: int = HP.IMG_SIZE,
                 aug: bool = True):
        self.images_dir = Path(images_dir)
        self.data: Dict[str, List[List[float]]] = json.load(open(annotations_json, "r"))
        self.fnames = [f for f in self.data.keys() if (self.images_dir / f).exists()]
        self.fnames.sort()
        self.img_size = img_size
        self.aug = aug
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.fnames)

    def _apply_flips(self, img: Image.Image, pts: List[Tuple[float, float]]):
        w, h = img.size
        # Horizontal flip
        if self.aug and random.random() < HP.AUG_HFLIP_P:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            pts = [(w - x, y) for (x, y) in pts]
        # Vertical flip
        if self.aug and random.random() < HP.AUG_VFLIP_P:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            pts = [(x, h - y) for (x, y) in pts]
        return img, pts

    def __getitem__(self, idx: int):
        fname = self.fnames[idx]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        pts = [(float(x), float(y)) for (x, y) in self.data.get(fname, [])]

        img, pts = self._apply_flips(img, pts)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        sx = self.img_size / w0
        sy = self.img_size / h0
        pts_resized = [(x * sx, y * sy) for (x, y) in pts]

        if HP.SIGMA_PX is None:
            sigma = estimate_sigma_from_points(pts_resized)
        else:
            sigma = float(HP.SIGMA_PX)

        dm = make_density_map(self.img_size, self.img_size, pts_resized, sigma_px=sigma)

        img_t = self.to_tensor(img)
        dm_t = torch.from_numpy(dm).unsqueeze(0)
        count = float(len(pts))
        return img_t, dm_t, count, fname

# -----------------------------
# Model: MobileNetV3-Small backbone + lightweight decoder
# -----------------------------
class DensityRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.conv_out = nn.Conv2d(backbone.features[-1].out_channels, 256, kernel_size=1, bias=False)

        def up(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.up1 = up(256, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)
        self.up4 = up(32, 16)
        self.up5 = up(16, 8)
        self.head = nn.Conv2d(8, 1, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.head.weight, a=1)

    def forward(self, x):
        f = self.features(x)
        y = self.conv_out(f)
        y = self.up1(y)
        y = self.up2(y)
        y = self.up3(y)
        y = self.up4(y)
        y = self.up5(y)
        y = self.head(y)
        y = F.relu(y)
        return y 

# -----------------------------
# Training / Evaluation
# -----------------------------

def mae_counts(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [B,1,H,W]"""
    pred_c = pred.sum(dim=[1, 2, 3])
    gt_c = gt.sum(dim=[1, 2, 3])
    return (pred_c - gt_c).abs().mean()


def split_indices(n: int, val_split: float = 0.15, seed: int = 42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    v = int(round(n * val_split))
    val_idx = idx[:v]
    train_idx = idx[v:]
    return train_idx, val_idx


def collate(batch):
    imgs, dms, counts, fnames = zip(*batch)
    return torch.stack(imgs), torch.stack(dms), torch.tensor(counts, dtype=torch.float32), list(fnames)


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_csv_row(path: str, row: Dict[str, float | int | str]):
    exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def train_loop(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    ds_full = AlmondDensityDataset(args.images, args.annotations, img_size=HP.IMG_SIZE, aug=True)
    n = len(ds_full)
    if n == 0:
        raise RuntimeError("No images found that match annotations.")
    train_idx, val_idx = split_indices(n, val_split=args.val_split, seed=args.seed)

    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val = torch.utils.data.Subset(AlmondDensityDataset(args.images, args.annotations, img_size=HP.IMG_SIZE, aug=False), val_idx)

    dl_train = DataLoader(ds_train, batch_size=HP.BATCH_SIZE, shuffle=True, num_workers=HP.NUM_WORKERS,
                          pin_memory=HP.PIN_MEMORY, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=HP.BATCH_SIZE, shuffle=False, num_workers=HP.NUM_WORKERS,
                        pin_memory=HP.PIN_MEMORY, collate_fn=collate)

    model = DensityRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HP.LR, weight_decay=HP.WEIGHT_DECAY)
    crit = nn.MSELoss(reduction="mean")

    ensure_dir(HP.CKPT_DIR)
    best_mae = float("inf")

    # Resume Code
    last_path = Path(HP.CKPT_DIR) / HP.LAST_NAME
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"]) if "opt" in state else None
        print(f"[resume] Loaded {last_path}")

    for epoch in range(1, HP.EPOCHS + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for imgs, dms, counts, _ in tqdm(dl_train, desc=f"Epoch {epoch}/{HP.EPOCHS}"):
            imgs = imgs.to(device)
            dms = dms.to(device)
            opt.zero_grad()
            pred = model(imgs)

            # --- pixelwise loss ---
            mse_loss = crit(pred, dms)

            # --- count loss ---
            pred_counts = pred.sum(dim=[1, 2, 3])
            gt_counts = dms.sum(dim=[1, 2, 3])
            count_loss = F.l1_loss(pred_counts, gt_counts)

            # --- total loss ---
            loss = mse_loss + 10.0 * count_loss

            loss.backward()
            opt.step()
            if random.random() < 0.01:  # randomly print examples to check performance.
                print(f"[debug] pred_count={pred_counts.mean().item():.2f}, gt_count={gt_counts.mean().item():.2f}")

            run_loss += float(loss.item()) * imgs.size(0)

        train_loss = run_loss / len(dl_train.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            abs_err = 0.0
            for imgs, dms, counts, _ in dl_val:
                imgs = imgs.to(device)
                dms = dms.to(device)
                pred = model(imgs)
                abs_err += float((pred.sum(dim=[1, 2, 3]) - dms.sum(dim=[1, 2, 3])).abs().sum().item())
            val_mae = abs_err / len(dl_val.dataset) if len(dl_val.dataset) > 0 else 0.0

        dt = time.time() - t0
        print(f"Epoch {epoch}/{HP.EPOCHS} - Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.3f} | {dt:.1f}s")

        save_csv_row(HP.LOG_CSV, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "time_sec": dt,
        })

        # Save latest checkpoint
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
        }, Path(HP.CKPT_DIR) / HP.LAST_NAME)

        # Save best checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"model": model.state_dict(), "epoch": epoch}, Path(HP.CKPT_DIR) / HP.BEST_NAME)
            print(f"[best] New best MAE: {best_mae:.3f}")


# -----------------------------
# Inference utilities
# -----------------------------

def load_model_for_infer(device):
    model = DensityRegressor().to(device)
    best_path = Path(HP.CKPT_DIR) / HP.BEST_NAME
    last_path = Path(HP.CKPT_DIR) / HP.LAST_NAME
    ckpt_path = best_path if best_path.exists() else last_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {best_path} or {last_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])  # strict
    model.eval()
    print(f"[infer] Loaded checkpoint: {ckpt_path}")
    return model


def preprocess_image(img_path: str, img_size: int = HP.IMG_SIZE) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return transforms.ToTensor()(img).unsqueeze(0)  # [1,3,H,W]


def count_image(model: nn.Module, device, img_path: str) -> float:
    x = preprocess_image(img_path).to(device)
    with torch.no_grad():
        d = model(x)
        c = float(d.sum().item())
    return c


def save_density_overlay(img_path: str, density: np.ndarray, out_path: str):
    import matplotlib.pyplot as plt
    img = Image.open(img_path).convert("RGB").resize((HP.IMG_SIZE, HP.IMG_SIZE), Image.BILINEAR)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.imshow(density, alpha=0.45)
    plt.axis('off')
    ensure_dir(Path(out_path).parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def infer_folder(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_for_infer(device)
    images = sorted([p for p in Path(args.images).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])

    with open(args.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["image", "predicted_count"])  
        for p in tqdm(images, desc="Infer"):
            x = preprocess_image(str(p)).to(device)
            with torch.no_grad():
                d = model(x)
                c = float(d.sum().item())
            w.writerow([p.name, c])

            if args.viz_inference:
                save_density_overlay(str(p), d.squeeze(0).squeeze(0).cpu().numpy(), str(Path(args.viz_dir) / f"{p.stem}_overlay.png"))

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Almond counting via density-map regression (MobileNetV3-Small)")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--images", type=str, required=True, help="Folder with images")
    parser.add_argument("--annotations", type=str, help="annotations.json for training")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    # Inference-only
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--viz_inference", action="store_true")
    parser.add_argument("--viz_dir", type=str, default="viz")

    args = parser.parse_args()

    if HP.IMG_SIZE % 32 != 0:
        raise ValueError("IMG_SIZE must be divisible by 32.")

    if args.mode == "train":
        if not args.annotations:
            raise ValueError("--annotations is required for training")
        train_loop(args)
    else:
        infer_folder(args)
