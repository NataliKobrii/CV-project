"""Metrics and evaluation harnesses for RQ3.

PSNR and SSIM are implemented directly, following the same policy as
eval_detect and eval_pose: the metric itself should be easy to inspect. The
intrinsic evaluation degrades aerial frames with known ground truth and
scores every restoration method on PSNR, SSIM and runtime.

The extrinsic evaluation is our addition. The same degraded and restored
frames go through the detector (AP@0.5 from eval_detect) and the pose
estimator (PCK from eval_pose). This shows whether restoration recovers
task performance and not only pixel quality.
"""
import csv
import time
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DIR, TABLES_DIR, VISDRONE_PERSON  # noqa: E402
from restore import (deblur_methods, defocus_psf, degrade, denoise_methods,  # noqa: E402
                     motion_psf)


def psnr(a, b):
    """Return the peak signal-to-noise ratio in dB between two uint8
    images."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a, b, sigma=1.5):
    """Return the single-scale SSIM (Wang et al. 2004) on grayscale,
    computed with the standard 11x11 Gaussian window and constants."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)

    def blur(z):
        return cv2.GaussianBlur(z, (11, 11), sigma)
    # The local statistics come from Gaussian-weighted windows.
    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * cxy + C2)) / (
        (mx * mx + my * my + C1) * (vx + vy + C2))
    return float(s.mean())


def conditions():
    """Return the fixed degradation suite. Each entry holds a name, a PSF, a
    noise level, a salt-and-pepper amount, and the method family that
    applies to it."""
    return [
        dict(name="motion15_n2", psf=motion_psf(15, 45), noise_sigma=2.0,
             sp=0.0, methods="deblur"),
        dict(name="motion25_n5", psf=motion_psf(25, 20), noise_sigma=5.0,
             sp=0.0, methods="deblur"),
        dict(name="defocus7_n2", psf=defocus_psf(7), noise_sigma=2.0,
             sp=0.0, methods="deblur"),
        dict(name="noise15", psf=None, noise_sigma=15.0, sp=0.0,
             methods="denoise"),
        dict(name="noise25", psf=None, noise_sigma=25.0, sp=0.0,
             methods="denoise"),
        dict(name="saltpepper4", psf=None, noise_sigma=0.0, sp=0.04,
             methods="denoise"),
    ]


def methods_for(cond):
    """Return the restoration methods that apply to the given condition."""
    if cond["methods"] == "deblur":
        return deblur_methods(cond["psf"], cond["noise_sigma"])
    return denoise_methods(cond["noise_sigma"], salt_pepper=cond["sp"] > 0)


def degrade_by(cond, img, seed=0):
    """Degrade one image according to the given condition."""
    return degrade(img, cond["psf"], cond["noise_sigma"], cond["sp"], seed=seed)


def run_intrinsic(images, conds=None, out_csv=None, verbose=True):
    """Return the mean PSNR, SSIM and runtime for every condition and method
    over a list of BGR images."""
    conds = conds or conditions()
    rows = []
    for cond in conds:
        methods = methods_for(cond)
        agg = {m: [] for m in methods}
        for i, clean in enumerate(images):
            # A different seed per image keeps the noise patterns independent.
            deg = degrade_by(cond, clean, seed=i)
            for name, fn in methods.items():
                t0 = time.perf_counter()
                rest = fn(deg)
                ms = (time.perf_counter() - t0) * 1000.0
                agg[name].append((psnr(rest, clean), ssim(rest, clean), ms))
        for name, vals in agg.items():
            v = np.asarray(vals)
            rows.append({"condition": cond["name"], "method": name,
                         "PSNR": round(float(v[:, 0].mean()), 2),
                         "SSIM": round(float(v[:, 1].mean()), 4),
                         "ms": round(float(v[:, 2].mean()), 1),
                         "n_images": len(images)})
            if verbose:
                print(rows[-1])
    if out_csv:
        write_csv(rows, out_csv)
    return rows


def write_csv(rows, path):
    """Write a list of result dictionaries to a CSV file."""
    path = Path(path)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                            extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("Saved ->", path)


def example_grid(clean, cond, out_path, crop=420, label_h=26):
    """Save a qualitative strip for one image: the clean frame, the degraded
    frame and every restoration, center-cropped, with the PSNR printed
    under each panel."""
    methods = methods_for(cond)
    deg = degrade_by(cond, clean, seed=0)
    panels = [("clean", clean)]
    panels += [(n if n != "none" else "degraded", fn(deg))
               for n, fn in methods.items()]
    h, w = clean.shape[:2]
    y0, x0 = max(0, h // 2 - crop // 2), max(0, w // 2 - crop // 2)
    tiles = []
    for name, img in panels:
        tile = img[y0:y0 + crop, x0:x0 + crop].copy()
        bar = np.full((label_h, tile.shape[1], 3), 30, np.uint8)
        txt = name if name == "clean" else f"{name}  {psnr(img, clean):.1f}dB"
        cv2.putText(bar, txt, (6, label_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([tile, bar]))
    grid = np.hstack(tiles)
    cv2.imwrite(str(out_path), grid)
    print("Saved ->", out_path)


def detection_under_degradation(weights, cond, method_names, max_images=60,
                                imgsz=1280, conf=0.01, device=None,
                                max_width=None):
    """Compute AP@0.5 on the VisDrone-person validation set for clean,
    degraded and restored frames.

    The function reuses ap50 and load_yolo_gt from eval_detect, so the
    numbers are directly comparable with the detection baseline table.
    """
    from detect import PersonDetector
    from eval_detect import ap50, load_yolo_gt

    # One detector instance scores every variant, so the AP values are
    # directly comparable.
    det = PersonDetector(weights=weights, conf=conf, imgsz=imgsz, device=device)
    methods = methods_for(cond)
    variants = ["clean", "none"] + [m for m in method_names if m != "none"]
    img_dir = VISDRONE_PERSON / "images" / "val"
    lbl_dir = VISDRONE_PERSON / "labels" / "val"
    images = sorted(img_dir.glob("*.jpg"))[:max_images]
    preds = {v: [] for v in variants}
    gts = []
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if max_width and img.shape[1] > max_width:
            s = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s)
        h, w = img.shape[:2]
        gts.append(load_yolo_gt(lbl_dir / (img_path.stem + ".txt"), w, h))
        deg = degrade_by(cond, img, seed=i)
        for v in variants:
            x = img if v == "clean" else deg if v == "none" else methods[v](deg)
            preds[v].append(det(x))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(images)} images")
    rows = []
    for v in variants:
        m = ap50(preds[v], gts)
        rows.append({"condition": cond["name"],
                     "variant": "degraded" if v == "none" else v,
                     "AP50": round(m["AP50"], 4),
                     "recall": round(m["recall"], 4),
                     "n_pred": m["n_pred"], "n_images": len(gts)})
        print(rows[-1])
    return rows


def pose_under_degradation(cond, method_names, max_persons=200, device="cpu"):
    """Compute PCK on the UAV-Human ground-truth keypoints for clean,
    degraded and restored frames.

    The degradations change only pixel values, not geometry, so the
    ground-truth keypoints stay valid.
    """
    from data.uavhuman import iter_dataset
    from eval_pose import pck
    from pose import PoseEstimator

    # The ground-truth boxes are reused for every variant, so only the pose
    # quality varies.
    est = PoseEstimator(device=device)
    methods = methods_for(cond)
    variants = ["clean", "none"] + [m for m in method_names if m != "none"]
    preds = {v: [] for v in variants}
    gts, viss, boxes = [], [], []
    n = 0
    for img_path, persons in iter_dataset(RAW_DIR / "uavhuman_pose"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        deg = degrade_by(cond, img, seed=n)
        bxs = np.stack([p["box"] for p in persons])
        for v in variants:
            x = img if v == "clean" else deg if v == "none" else methods[v](deg)
            kp, _ = est(x, bxs)
            preds[v].extend(kp)
        for p in persons:
            gts.append(p["kpts"]); viss.append(p["vis"]); boxes.append(p["box"])
        n += len(persons)
        if n >= max_persons:
            break
    rows = []
    for v in variants:
        m = pck(np.stack(preds[v]), np.stack(gts), np.stack(viss),
                np.stack(boxes))
        rows.append({"condition": cond["name"],
                     "variant": "degraded" if v == "none" else v,
                     "PCK": round(m["PCK"], 4),
                     "PCK_h50-100": round(m["PCK_h50-100"], 4)
                     if np.isfinite(m["PCK_h50-100"]) else "nan",
                     "PCK_h100-inf": round(m["PCK_h100-inf"], 4),
                     "n_persons": m["n_persons"]})
        print(rows[-1])
    return rows


if __name__ == "__main__":
    # A quick self-check on three validation frames, intrinsic table only.
    imgs = []
    for p in sorted((VISDRONE_PERSON / "images" / "val").glob("*.jpg"))[:3]:
        im = cv2.imread(str(p))
        if im is not None:
            s = min(1.0, 1280 / im.shape[1])
            imgs.append(cv2.resize(im, None, fx=s, fy=s))
    run_intrinsic(imgs, out_csv=TABLES_DIR / "restoration_intrinsic_quick.csv")
