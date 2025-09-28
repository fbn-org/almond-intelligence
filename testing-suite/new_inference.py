#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation: read almond_counts.csv (expects 'image_name'), predict presence+count
for each image in IMAGE_DIR using a trained density model, and write results back to the CSV.

- Hard-coded model, resize, calibration, and paths.
- Uses MobileNetV3-based DensityRegressor (must match training).
"""

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

# ------ Config ------
CKPT_DIR = "model"
BEST_NAME = "almond_density_best.pth"
LAST_NAME = "almond_density_last.pth"
MODEL_CKPT = str(Path(CKPT_DIR) / BEST_NAME)
MODEL_FALLBACK_CKPT = str(Path(CKPT_DIR) / LAST_NAME)

IMAGE_DIR = "image_dir"  
INPUT_CSV = "almond_counts.csv"
OUTPUT_CSV = "almond_counts.csv"

RESIZE = (256, 256)
PRESENCE_THRESHOLD = 0.5

CALIBRATION_SCALE = 1 #unused
CALIBRATION_BIAS  = 0 #unused (but technically 1 in final app)

FORCE_CPU = False


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

def get_device():
    if FORCE_CPU:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(device: torch.device) -> nn.Module:
    model = DensityRegressor().to(device)

    ckpt_path = Path(MODEL_CKPT)
    if not ckpt_path.exists():
        ckpt_path = Path(MODEL_FALLBACK_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CKPT} or {MODEL_FALLBACK_CKPT}")

    state = torch.load(str(ckpt_path), map_location=device)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[info] Loaded checkpoint: {ckpt_path}")
    return model

_TO_TENSOR = transforms.ToTensor()

def preprocess(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB").resize(RESIZE, Image.BILINEAR)
    return _TO_TENSOR(img).unsqueeze(0)

@torch.no_grad()
def predict_count(image_path: str,
                  model: nn.Module,
                  device: torch.device):
    """Return (raw_count, calibrated_count, density_np) for a single image."""
    x = preprocess(image_path).to(device)
    d = model(x)  # [1,1,H,W]
    raw = float(d.sum().item())
    cal = max(0.0, raw * CALIBRATION_SCALE + CALIBRATION_BIAS)
    dens = d.squeeze(0).squeeze(0).detach().cpu().numpy()
    return raw, cal, dens

def test_directory(directory_name: str):
    device = get_device()
    print(f"[info] Using device: {device}")

    model = load_model(device)

    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if "image_name" not in df.columns:
        raise ValueError(f"CSV must contain a column named 'image_name'. Columns found: {list(df.columns)}")

    has_almonds_counts = []
    almond_counts = []

    it = tqdm(df.itertuples(index=False), total=len(df), desc="Evaluating")
    for row in it:
        filename = getattr(row, "image_name")
        file_path = os.path.join(directory_name, filename)

        if not Path(file_path).exists():
            print(f"[warn] Missing image, skipping: {file_path}")
            has_almonds_counts.append(False)
            almond_counts.append(0)
            continue

        raw, cal, _ = predict_count(
            image_path=file_path,
            model=model,
            device=device,
        )

        contains = (cal > PRESENCE_THRESHOLD)
        has_almonds_counts.append(bool(contains))
        almond_counts.append(int(round(cal)))

        it.set_postfix_str(f"{filename}: contains={'true' if contains else 'false'} | count={int(round(cal))}")
        print(f"[result] image={filename} | raw={raw:.3f} | cal={cal:.3f} | contains_almond={'true' if contains else 'false'} | count={int(round(cal))}")

    df["contains_almond"] = has_almonds_counts
    df["predicted_count"] = almond_counts
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[info] Saved predictions to {OUTPUT_CSV}")

if __name__ == "__main__":
    test_directory(IMAGE_DIR)
