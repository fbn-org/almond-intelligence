#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Almond density evaluation with visualization.

Examples
--------
CSV only:
    python evaluation.py --images path/to/imgs --model weights.pth --csv results.csv

CSV + overlays for every image:
    python evaluation.py --images imgs --model weights.pth --csv results.csv \
        --viz_dir viz --save_overlays

CSV + a single PDF report (recommended):
    python evaluation.py --images imgs --model weights.pth --annotations ann.json \
        --csv results.csv --viz_dir viz --report_pdf viz/report.pdf --save_overlays --verbose

Notes
-----
- Overlays are saved into <viz_dir>/overlays/*.png
- Temporary density maps are stored as .npy in <viz_dir>/.tmp_dens (auto-created).
- Top-K error mosaic goes into the PDF; adjust with --topk_error_mosaic.
"""

import argparse, os, json, glob, sys, time, math, csv
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ------------------ Model (must match training) ------------------
class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
    def forward(self, x):
        x = self.features(x)
        return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)


# ------------------ Helpers ------------------
def parse_wh(hs: Optional[str]) -> Optional[Tuple[int, int]]:
    if hs is None:
        return None
    try:
        w_str, h_str = hs.split(",")
        return (int(w_str), int(h_str))
    except Exception:
        raise SystemExit(f"--resize must be 'W,H' (e.g. 512,512). Got: {hs}")


def load_image_as_tensor(path: str, target_size: Optional[Tuple[int,int]] = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW


def _normalize_density(d: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """Normalize density to [0,1] with percentile clipping for robust display."""
    d = np.asarray(d, dtype=np.float32)
    hi = np.percentile(d, pct) if pct is not None else d.max()
    hi = max(hi, 1e-6)
    return np.clip(d / hi, 0.0, 1.0)


def save_overlay(image_path: str, density_2d: np.ndarray, out_path: str,
                 alpha: float = 0.5, pct: float = 99.0, title: Optional[str] = None,
                 cmap: Optional[str] = None, gt_points: Optional[List[Tuple[float,float]]] = None):
    """
    Overlay density heatmap on the original (resized) image and optionally draw GT points.
    """
    base = Image.open(image_path).convert("RGB")
    H, W = density_2d.shape
    if base.size != (W, H):
        base = base.resize((W, H), Image.BILINEAR)

    # Optionally draw GT dots on a copy so we preserve the photo
    if gt_points:
        draw = ImageDraw.Draw(base)
        r = max(2, int(0.004 * (W + H)))  # radius scales with image size
        for (x, y) in gt_points:
            # Ensure points are within bounds if original annot was for original size
            # (We assume coords already match resized input; if not, pre-scale before calling.)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 0), width=max(2, r//2))

    d_norm = _normalize_density(density_2d, pct=pct)

    plt.figure()
    plt.imshow(base)
    if cmap:
        plt.imshow(d_norm, alpha=alpha, cmap=cmap)
    else:
        plt.imshow(d_norm, alpha=alpha)
    if title:
        plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=160)
    plt.close()


def save_mosaic(rows: int, cols: int, items: List[Dict], out_path: str,
                alpha: float = 0.5, pct: float = 99.0, cmap: Optional[str] = None):
    """
    items: list of dicts
      {'image_path': str, 'density': np.ndarray(H,W), 'title': str, 'gt_points': Optional[List[(x,y)]]}
    """
    n = min(len(items), rows * cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        im_path = items[i]['image_path']
        dens = items[i]['density']
        title = items[i].get('title', None)
        gt_points = items[i].get('gt_points', None)

        base = Image.open(im_path).convert("RGB")
        H, W = dens.shape
        if base.size != (W, H):
            base = base.resize((W, H), Image.BILINEAR)

        if gt_points:
            draw = ImageDraw.Draw(base)
            r = max(2, int(0.004 * (W + H)))
            for (x, y) in gt_points:
                draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 0), width=max(2, r//2))

        ax.imshow(base)
        if cmap:
            ax.imshow(_normalize_density(dens, pct=pct), alpha=alpha, cmap=cmap)
        else:
            ax.imshow(_normalize_density(dens, pct=pct), alpha=alpha)
        if title:
            ax.set_title(title, fontsize=9)
        ax.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def scatter_with_identity(x, y, title, xlabel, ylabel, note=None):
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, s=15, alpha=0.7)
    lim = [0, max(max(x), max(y)) * 1.05 if len(x) and len(y) else 1.0]
    plt.plot(lim, lim, linestyle='--', linewidth=1)
    plt.xlim(lim); plt.ylim(lim)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if note:
        plt.figtext(0.01, -0.08, note, fontsize=8, ha='left', va='top')
    plt.tight_layout()


def hist_plot(values, title, xlabel):
    plt.figure(figsize=(6,4))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser("Evaluate or infer with almond density model (with visualization)")
    ap.add_argument("--images", required=True, help="Folder of images")
    ap.add_argument("--model", required=True, help="Path to .pth/.pt weights")
    ap.add_argument("--resize", default=None, help="W,H to resize inputs (e.g., 1024,768)")
    ap.add_argument("--use_gpu", action="store_true", help="Try CUDA/MPS if available")
    ap.add_argument("--annotations", default=None, help="JSON mapping filename -> [[x,y], ...]")
    ap.add_argument("--csv", default=None, help="Optional CSV output path")
    # Visualization options
    ap.add_argument("--viz_dir", default=None, help="Directory to write visualization artifacts")
    ap.add_argument("--save_overlays", action="store_true", help="Save per-image heatmap overlays")
    ap.add_argument("--report_pdf", default=None, help="Write a single multi-page PDF report")
    ap.add_argument("--alpha_heatmap", type=float, default=0.5, help="Overlay alpha (0-1)")
    ap.add_argument("--pct_clip", type=float, default=99.0, help="Percentile clip for heatmap")
    ap.add_argument("--cmap", default=None, help="Matplotlib colormap for heatmap (e.g., 'magma')")
    ap.add_argument("--topk_error_mosaic", type=int, default=12, help="Top-K calibrated error examples in report")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("[info] Starting evaluator")
    print(f"[info] args: {args}")

    # Paths
    if not os.path.isdir(args.images):
        raise SystemExit(f"[error] --images directory not found: {args.images}")
    if not os.path.isfile(args.model):
        raise SystemExit(f"[error] --model file not found: {args.model}")

    target_size = parse_wh(args.resize)

    # Device
    if args.use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[info] Using device: {device}")

    # Model
    model = SimpleCNN().to(device)
    print(f"[info] Loading weights: {args.model}")
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("[info] Model ready")

    # Images
    exts = ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG")
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(args.images, e))
    img_paths.sort()
    print(f"[info] Found {len(img_paths)} images in {args.images}")
    if not img_paths:
        print("[warn] No images matched. Check extensions and path.")
        return

    # Annotations (optional)
    anno = None
    if args.annotations:
        if not os.path.isfile(args.annotations):
            raise SystemExit(f"[error] annotations file not found: {args.annotations}")
        with open(args.annotations, "r", encoding="utf-8") as f:
            anno = json.load(f)
        print(f"[info] Loaded annotations for {len(anno)} files")

    # Visualization dirs
    want_viz = bool(args.viz_dir or args.save_overlays or args.report_pdf)
    viz_dir = args.viz_dir if args.viz_dir else (os.path.join(os.getcwd(), "viz") if (args.save_overlays or args.report_pdf) else None)
    overlays_dir = None
    tmp_dens_dir = None
    if want_viz:
        os.makedirs(viz_dir, exist_ok=True)
        overlays_dir = os.path.join(viz_dir, "overlays")
        os.makedirs(overlays_dir, exist_ok=True)
        tmp_dens_dir = os.path.join(viz_dir, ".tmp_dens")
        os.makedirs(tmp_dens_dir, exist_ok=True)

    # Pass 1: inference & (optional) store density maps
    rows = []
    abs_err_sum = 0.0
    n_with_gt = 0

    t0 = time.time()
    for i, p in enumerate(img_paths, 1):
        name = os.path.basename(p)
        if args.verbose:
            print(f"[info] ({i}/{len(img_paths)}) reading {name}")
        x = load_image_as_tensor(p, target_size).to(device, dtype=torch.float32)
        with torch.no_grad():
            y = model(x)  # 1x1xHxW (H,W == resized image)
        dens = y[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_count = float(dens.sum())

        # Save density map if we'll need it later
        dens_path = None
        if want_viz:
            dens_path = os.path.join(tmp_dens_dir, f"{name}.npy")
            np.save(dens_path, dens)

        row = {
            "image": name,
            "image_path": p,
            "predicted_count": float(pred_count),
            "_dens_path": dens_path  # internal, for later viz
        }

        if anno is not None and name in anno:
            gt_count = float(len(anno[name]))
            row["gt_count"] = gt_count
            row["abs_error"] = abs(pred_count - gt_count)
            abs_err_sum += row["abs_error"]
            n_with_gt += 1
            if args.verbose:
                print(f"{name}: pred {pred_count:.3f} | gt {gt_count:.3f} | abs_err {row['abs_error']:.3f}")
        else:
            if args.verbose:
                print(f"{name}: pred {pred_count:.3f}")

        rows.append(row)

    dt = time.time() - t0
    print(f"[info] Inference done in {dt:.2f}s on {len(rows)} images")

    # Calibration
    ratios = []
    for r in rows:
        if "gt_count" in r and r["predicted_count"] > 1e-12:
            ratios.append(r["gt_count"] / r["predicted_count"])
    if len(ratios) == 0:
        scale = 1.0
        print("[warn] No labeled images available for calibration; using scale=1.0")
    else:
        scale = float(np.median(np.array(ratios)))
        print(f"[info] Calibration scale (median gt/pred over {len(ratios)} images): {scale:.6f}")

    cal_abs_err_sum = 0.0
    cal_n = 0
    for r in rows:
        calibrated = scale * r["predicted_count"]
        r["calibrated_count"] = float(calibrated)
        if "gt_count" in r:
            cae = abs(calibrated - r["gt_count"])
            r["calibrated_abs_error"] = float(cae)
            cal_abs_err_sum += cae
            cal_n += 1

    # Metrics print
    if n_with_gt > 0:
        mae_raw = abs_err_sum / n_with_gt
        print(f"[info] Raw Test MAE over {n_with_gt} labeled images: {mae_raw:.4f}")
    else:
        mae_raw = None
    if cal_n > 0:
        mae_cal = cal_abs_err_sum / cal_n
        print(f"[info] Calibrated Test MAE over {cal_n} labeled images: {mae_cal:.4f}")
    else:
        mae_cal = None

    # CSV
    if args.csv:
        fieldnames = ["image", "predicted_count", "calibrated_count"]
        if any("gt_count" in r for r in rows):
            fieldnames += ["gt_count", "abs_error", "calibrated_abs_error"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"[info] Wrote CSV: {args.csv}")

    # ---------------- Visualization ----------------
    # 1) Save per-image overlays (optional)
    if want_viz and args.save_overlays:
        print(f"[info] Saving overlays to: {overlays_dir}")
        for r in rows:
            dens = np.load(r["_dens_path"])
            title = None
            if "gt_count" in r:
                title = f"{r['image']} | pred {r['predicted_count']:.2f} → cal {r['calibrated_count']:.2f} | gt {r['gt_count']:.2f}"
            else:
                title = f"{r['image']} | pred {r['predicted_count']:.2f} → cal {r['calibrated_count']:.2f}"
            gt_pts = None
            if anno is not None and r["image"] in anno:
                # IMPORTANT: If your annotations were for ORIGINAL size and you used --resize,
                # you should pre-scale the coords. Here we assume they already match the resized inputs.
                gt_pts = [tuple(pt) for pt in anno[r["image"]]]
            out_png = os.path.join(overlays_dir, os.path.splitext(r["image"])[0] + ".png")
            save_overlay(
                r["image_path"], dens, out_png,
                alpha=args.alpha_heatmap, pct=args.pct_clip, title=title, cmap=args.cmap,
                gt_points=gt_pts
            )

    # 2) PDF report (optional)
    if want_viz and args.report_pdf:
        print(f"[info] Building PDF report: {args.report_pdf}")
        with PdfPages(args.report_pdf) as pdf:
            # Page 1: Summary
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            txt = []
            txt.append("Almond Density Evaluation Report")
            txt.append("")
            txt.append(f"Images evaluated: {len(rows)}")
            txt.append(f"Resize: {args.resize if args.resize else 'None'}")
            txt.append(f"Device: {device}")
            if mae_raw is not None:
                txt.append(f"Raw MAE (labeled subset): {mae_raw:.4f}")
            if mae_cal is not None:
                txt.append(f"Calibrated MAE (labeled subset): {mae_cal:.4f}")
            txt.append(f"Calibration scale: {scale:.6f}")
            txt.append("")
            txt.append(f"Overlays saved: {'Yes' if args.save_overlays else 'No'}")
            txt.append(f"Percentile clip for heatmap: {args.pct_clip}")
            txt.append(f"Heatmap alpha: {args.alpha_heatmap}")
            if args.cmap:
                txt.append(f"Colormap: {args.cmap}")

            plt.figtext(0.1, 0.9, "\n".join(txt), ha='left', va='top', fontsize=12)
            pdf.savefig(); plt.close()

            # Page 2: Histogram of predicted counts (raw & calibrated)
            preds = [r["predicted_count"] for r in rows]
            cals  = [r["calibrated_count"] for r in rows]
            plt.figure(figsize=(8,5))
            plt.hist(preds, bins=30, alpha=0.6, label="Predicted (raw)")
            plt.hist(cals, bins=30, alpha=0.6, label="Predicted (calibrated)")
            plt.title("Distribution of Predicted Counts")
            plt.xlabel("Count"); plt.ylabel("Frequency"); plt.legend()
            plt.tight_layout(); pdf.savefig(); plt.close()

            # If GT exists: scatter & error hist
            have_gt = [r for r in rows if "gt_count" in r]
            if have_gt:
                gt = [r["gt_count"] for r in have_gt]
                pr = [r["predicted_count"] for r in have_gt]
                ca = [r["calibrated_count"] for r in have_gt]

                # Page 3: Scatter raw
                scatter_with_identity(gt, pr, "Raw Predictions vs Ground Truth", "Ground Truth", "Predicted (raw)",
                                      note=(f"Raw MAE: {mae_raw:.4f}" if mae_raw is not None else None))
                pdf.savefig(); plt.close()

                # Page 4: Scatter calibrated
                scatter_with_identity(gt, ca, "Calibrated Predictions vs Ground Truth", "Ground Truth", "Predicted (calibrated)",
                                      note=(f"Cal MAE: {mae_cal:.4f}" if mae_cal is not None else None))
                pdf.savefig(); plt.close()

                # Page 5: Error histogram (calibrated if available, else raw)
                if mae_cal is not None:
                    errs = [abs(r["calibrated_count"] - r["gt_count"]) for r in have_gt]
                    hist_plot(errs, "Calibrated Absolute Error", "Absolute Error")
                else:
                    errs = [abs(r["predicted_count"] - r["gt_count"]) for r in have_gt]
                    hist_plot(errs, "Raw Absolute Error", "Absolute Error")
                pdf.savefig(); plt.close()

                # Page 6: Mosaic of top-K calibrated errors
                K = min(args.topk_error_mosaic, len(have_gt))
                # choose calibrated error if available
                key_err = "calibrated_abs_error" if "calibrated_abs_error" in have_gt[0] else "abs_error"
                topk = sorted(have_gt, key=lambda r: r.get(key_err, 0.0), reverse=True)[:K]

                # Build items with loaded densities
                items = []
                for r in topk:
                    dens = np.load(r["_dens_path"])
                    title = f"{r['image']} | pred {r['predicted_count']:.2f} → cal {r['calibrated_count']:.2f} | gt {r['gt_count']:.2f} | err {r.get(key_err,0):.2f}"
                    gt_pts = None
                    if anno is not None and r["image"] in anno:
                        gt_pts = [tuple(pt) for pt in anno[r["image"]]]
                    items.append({
                        "image_path": r["image_path"],
                        "density": dens,
                        "title": title,
                        "gt_points": gt_pts
                    })

                # Save a temporary mosaic image and embed it
                mosaic_path = os.path.join(viz_dir, "top_errors_mosaic.png")
                rows_grid = int(math.ceil(math.sqrt(K)))
                cols_grid = int(math.ceil(K / rows_grid))
                save_mosaic(rows_grid, cols_grid, items, mosaic_path,
                            alpha=args.alpha_heatmap, pct=args.pct_clip, cmap=args.cmap)

                img = Image.open(mosaic_path).convert("RGB")
                plt.figure(figsize=(10, 10 * img.height / img.width))
                plt.imshow(img); plt.axis('off'); plt.title("Top-K Error Overlays")
                pdf.savefig(); plt.close()

        print(f"[info] Wrote PDF: {args.report_pdf}")

    print("[info] Done.")


if __name__ == "__main__":
    main()
