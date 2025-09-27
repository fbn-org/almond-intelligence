#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Almond density-based counter using PyTorch, with train-time data augmentation.

Augmentations (train only):
- RandomHorizontalFlip / RandomVerticalFlip
- RandomRotation in [-max_rotate, max_rotate] degrees about image center (coordinates updated)
- ColorJitter (brightness/contrast/saturation/hue)
- Optional Gaussian blur
- Optional Gaussian noise (applied to tensor)

Usage:
    python almond_density_counter.py --images path/to/images \
        --annotations path/to/annotations.json \
        --epochs 50 --batch_size 4 --sigma 15 \
        --resize 1024,1024 \
        --use_gpu \
        --use_aug \
        --hflip_p 0.5 --vflip_p 0.0 --rotate_p 0.5 --max_rotate 15 \
        --jitter_brightness 0.2 --jitter_contrast 0.2 --jitter_saturation 0.2 --jitter_hue 0.05 \
        --blur_p 0.2 --blur_min 0.1 --blur_max 1.5 \
        --noise_p 0.2 --noise_std 0.02

Note:
- Augmentations are applied after optional resize, then density is synthesized from transformed points.
- If you don’t want augmentation, omit --use_aug (validation never augments).
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

# ------------------------- Utilities -------------------------

def parse_wh(hs: Optional[str]) -> Optional[Tuple[int, int]]:
    if hs is None:
        return None
    try:
        w_str, h_str = hs.split(",")
        return (int(w_str), int(h_str))
    except Exception:
        raise ValueError("--resize must be 'width,height' (e.g. '512,512')")

def rotate_point(x: float, y: float, cx: float, cy: float, theta_deg: float) -> Tuple[float, float]:
    """Rotate (x,y) around center (cx,cy) by theta degrees (CCW)."""
    theta = math.radians(theta_deg)
    dx, dy = x - cx, y - cy
    xr =  dx * math.cos(theta) - dy * math.sin(theta) + cx
    yr =  dx * math.sin(theta) + dy * math.cos(theta) + cy
    return xr, yr

def clamp_point(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    return max(0.0, min(x, w - 1)), max(0.0, min(y, h - 1))

# ------------------------- Dataset with Augmentation -------------------------

class AlmondDataset(Dataset):
    """Dataset that can apply geometric/photometric augmentation consistently with coordinates."""

    def __init__(
        self,
        img_dir: str,
        annotation_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        sigma: float = 15.0,
        use_aug: bool = False,
        # geometric aug params
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        rotate_p: float = 0.5,
        max_rotate: float = 15.0,
        # photometric aug params
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.05,
        blur_p: float = 0.2,
        blur_min: float = 0.1,
        blur_max: float = 1.5,
        noise_p: float = 0.2,
        noise_std: float = 0.02,
        subset_names: Optional[Sequence[str]] = None,
        rng_seed: int = 42,
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        with open(annotation_path, "r", encoding="utf-8") as f:
            self.annotations: Dict[str, List[List[float]]] = json.load(f)

        all_names = list(self.annotations.keys())
        self.img_names = list(subset_names) if subset_names is not None else all_names

        self.target_size = target_size
        self.sigma = sigma

        self.use_aug = use_aug
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_p = rotate_p
        self.max_rotate = max_rotate

        self.blur_p = blur_p
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.noise_p = noise_p
        self.noise_std = noise_std

        # Build photometric transform objects (operate on PIL)
        self.color_jitter = T.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=jitter_hue,
        )

        # For deterministic-ish randomness per sample if desired
        self._rng = random.Random(rng_seed)

    def __len__(self) -> int:
        return len(self.img_names)

    def _maybe_augment_geom(self, img: Image.Image, pts: List[Tuple[float, float]]) -> Tuple[Image.Image, List[Tuple[float, float]]]:
        """Apply random flips/rotation to both image and point coordinates."""
        w, h = img.size
        cx, cy = w / 2.0, h / 2.0
        pts_out = [tuple(p) for p in pts]

        # Horizontal flip
        if self._rng.random() < self.hflip_p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            pts_out = [(w - 1 - x, y) for (x, y) in pts_out]

        # Vertical flip
        if self._rng.random() < self.vflip_p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            pts_out = [(x, h - 1 - y) for (x, y) in pts_out]

        # Small rotation about center
        if self._rng.random() < self.rotate_p and self.max_rotate > 0:
            angle = self._rng.uniform(-self.max_rotate, self.max_rotate)
            # expand=False keeps size constant; fillcolor black
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
            new_pts = []
            for (x, y) in pts_out:
                xr, yr = rotate_point(x, y, cx, cy, angle)
                # Keep only points that remain within bounds after rotation
                if 0 <= xr < w and 0 <= yr < h:
                    new_pts.append((xr, yr))
            pts_out = new_pts

        return img, pts_out

    def _maybe_augment_photo(self, img: Image.Image) -> Image.Image:
        """Color jitter and optional Gaussian blur (PIL-level)."""
        img = self.color_jitter(img)
        if self._rng.random() < self.blur_p:
            # PIL GaussianBlur radius
            radius = self._rng.uniform(self.blur_min, self.blur_max)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def _maybe_add_noise(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor in [0,1]."""
        if self._rng.random() < self.noise_p:
            noise = torch.randn_like(tensor_img) * self.noise_std
            tensor_img = torch.clamp(tensor_img + noise, 0.0, 1.0)
        return tensor_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image {img_path} not found.")
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize first (simplifies coord math)
        if self.target_size is not None:
            target_w, target_h = self.target_size
            img = img.resize((target_w, target_h), Image.LANCZOS)
            w_ratio = target_w / orig_w
            h_ratio = target_h / orig_h
        else:
            target_w, target_h = orig_w, orig_h
            w_ratio = h_ratio = 1.0

        # Scale coords to resized space
        pts = []
        for (x, y) in self.annotations[img_name]:
            xs = max(0.0, min(x * w_ratio, target_w - 1))
            ys = max(0.0, min(y * h_ratio, target_h - 1))
            pts.append((xs, ys))

        # Geometric augmentation (train only)
        if self.use_aug:
            img, pts = self._maybe_augment_geom(img, pts)
            img = self._maybe_augment_photo(img)

        # Convert image to tensor in [0,1]
        img_array = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,C)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C,H,W)

        # Optional noise (train only)
        if self.use_aug:
            img_tensor = self._maybe_add_noise(img_tensor)

        # Build density map from (possibly augmented) points
        density = np.zeros((target_h, target_w), dtype=np.float32)
        for (x, y) in pts:
            xi = int(round(x)); yi = int(round(y))
            if 0 <= xi < target_w and 0 <= yi < target_h:
                density[yi, xi] += 1.0
        if density.sum() > 0:
            density = gaussian_filter(density, sigma=self.sigma, mode="constant")

        density_tensor = torch.from_numpy(density).unsqueeze(0)  # (1,H,W)
        return img_tensor, density_tensor

# ------------------------- Model -------------------------

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x

# ------------------------- Train / Eval -------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
) -> None:
    model.to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, densities in train_loader:
            images = images.to(device, dtype=torch.float32)
            densities = densities.to(device, dtype=torch.float32)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, densities)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.6f}")

        if val_loader is not None:
            model.eval()
            abs_error = 0.0
            with torch.no_grad():
                for images, densities in val_loader:
                    images = images.to(device, dtype=torch.float32)
                    densities = densities.to(device, dtype=torch.float32)
                    outputs = model(images)
                    pred_counts = outputs.sum(dim=[1, 2, 3])
                    gt_counts = densities.sum(dim=[1, 2, 3])
                    abs_error += torch.abs(pred_counts - gt_counts).sum().item()
            mae = abs_error / len(val_loader.dataset)
            print(f"           Validation MAE (count): {mae:.4f}")

# ------------------------- Main -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train an almond density estimator using PyTorch (with augmentation).")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--resize", type=str, default=None, help="'W,H' (e.g. 1024,1024)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--use_gpu", action="store_true")

    # Augmentation flags
    parser.add_argument("--use_aug", action="store_true", help="Enable train-time data augmentation")
    parser.add_argument("--hflip_p", type=float, default=0.5)
    parser.add_argument("--vflip_p", type=float, default=0.0)
    parser.add_argument("--rotate_p", type=float, default=0.5)
    parser.add_argument("--max_rotate", type=float, default=15.0)
    parser.add_argument("--jitter_brightness", type=float, default=0.2)
    parser.add_argument("--jitter_contrast", type=float, default=0.2)
    parser.add_argument("--jitter_saturation", type=float, default=0.2)
    parser.add_argument("--jitter_hue", type=float, default=0.05)
    parser.add_argument("--blur_p", type=float, default=0.2)
    parser.add_argument("--blur_min", type=float, default=0.1)
    parser.add_argument("--blur_max", type=float, default=1.5)
    parser.add_argument("--noise_p", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    target_size = parse_wh(args.resize)

    # Device
    if args.use_gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Repro split
    rng = np.random.default_rng(args.seed)

    # Load annotations to decide split
    with open(args.annotations, "r", encoding="utf-8") as f:
        anno = json.load(f)
    all_names = list(anno.keys())
    n_total = len(all_names)
    n_val = int(round(n_total * max(0.0, min(args.val_split, 0.99))))
    perm = rng.permutation(n_total).tolist()
    val_idx = set(perm[:n_val])
    train_names = [all_names[i] for i in range(n_total) if i not in val_idx]
    val_names = [all_names[i] for i in range(n_total) if i in val_idx]

    # Datasets
    train_dataset = AlmondDataset(
        img_dir=args.images,
        annotation_path=args.annotations,
        target_size=target_size,
        sigma=args.sigma,
        use_aug=args.use_aug,
        hflip_p=args.hflip_p,
        vflip_p=args.vflip_p,
        rotate_p=args.rotate_p,
        max_rotate=args.max_rotate,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
        jitter_saturation=args.jitter_saturation,
        jitter_hue=args.jitter_hue,
        blur_p=args.blur_p,
        blur_min=args.blur_min,
        blur_max=args.blur_max,
        noise_p=args.noise_p,
        noise_std=args.noise_std,
        subset_names=train_names,
        rng_seed=args.seed,
    )

    val_loader = None
    if n_val > 0:
        val_dataset = AlmondDataset(
            img_dir=args.images,
            annotation_path=args.annotations,
            target_size=target_size,
            sigma=args.sigma,
            use_aug=False,  # NEVER augment validation
            subset_names=val_names,
            rng_seed=args.seed,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model + train
    model = SimpleCNN()
    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    # Save
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "almond_density_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
