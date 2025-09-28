# allows user to use check a single image for almond amounts- used for the final web app.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference: count almonds in a single image using a trained density model.
- Prompts a file picker if --image is not provided.
- Loads models/almond_density_best.pth (or ..._last.pth).
- Prints estimated count and optionally saves an overlay.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from PIL import Image

# -----------------------------
# Constants (match your training)
# -----------------------------
IMG_SIZE = 256                 # set to the value you used in training (256 if you followed our tweak)
CKPT_DIR = "models"
BEST_NAME = "almond_density_best.pth"
LAST_NAME = "almond_density_last.pth"

# -----------------------------
# Model (must match training)
# -----------------------------
class DensityRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features  # stride 32
        self.conv_out = nn.Conv2d(self.features[-1].out_channels, 256, kernel_size=1, bias=False)

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

        # Inference will load weights; init here is irrelevant but keep sane
        nn.init.kaiming_uniform_(self.head.weight, a=1)

    def forward(self, x):
        f = self.features(x)     # [B, C, H/32, W/32]
        y = self.conv_out(f)
        y = self.up1(y)
        y = self.up2(y)
        y = self.up3(y)
        y = self.up4(y)
        y = self.up5(y)
        y = self.head(y)
        y = F.relu(y)            # non-negative density
        return y                 # [B,1,H,W]

# -----------------------------
# Utilities
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(device: torch.device) -> nn.Module:
    model = DensityRegressor().to(device)
    best = Path(CKPT_DIR) / BEST_NAME
    last = Path(CKPT_DIR) / LAST_NAME
    ckpt = best if best.exists() else last
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found at {best} or {last}")
    state = torch.load(ckpt, map_location=device)
    # training saved {"model": state_dict, ...}
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[info] Loaded checkpoint: {ckpt}")
    return model

def pick_image_with_dialog() -> str:
    # Lazy import so CLI-only users don't need Tkinter everywhere
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return path

def preprocess(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return transforms.ToTensor()(img).unsqueeze(0)  # [1,3,H,W], float32 in [0,1]

def run_inference(model: nn.Module, device: torch.device, img_path: str) -> float:
    x = preprocess(img_path).to(device)
    with torch.no_grad():
        d = model(x)        # [1,1,H,W]
        count = float(d.sum().item())
    return count

def save_overlay(img_path: str, model: nn.Module, device: torch.device, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # forward to get density
    x = preprocess(img_path).to(device)
    with torch.no_grad():
        d = model(x).squeeze(0).squeeze(0).cpu().numpy()

    # prepare overlay
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(d, alpha=0.45)
    plt.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[info] Saved overlay to {out_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Select an image and estimate almond count.")
    parser.add_argument("--image", type=str, help="Path to an image file; if omitted a file picker will open.")
    parser.add_argument("--save_overlay", action="store_true", help="Save a heatmap overlay PNG next to the image.")
    parser.add_argument("--overlay_dir", type=str, default="viz", help="Directory to write overlays if --save_overlay is set.")
    args = parser.parse_args()

    device = get_device()
    print(f"[info] Using device: {device}")

    model = load_model(device)

    img_path = args.image
    if not img_path:
        img_path = pick_image_with_dialog()
        if not img_path:
            print("[error] No image selected. Exiting.")
            return
    if not Path(img_path).exists():
        print(f"[error] Image not found: {img_path}")
        return

    count = run_inference(model, device, img_path)
    print(f"[result] {Path(img_path).name}: estimated almonds = {count:.3f}")

    if args.save_overlay:
        stem = Path(img_path).stem
        out_path = Path(args.overlay_dir) / f"{stem}_overlay.png"
        save_overlay(img_path, model, device, str(out_path))

if __name__ == "__main__":
    main()
