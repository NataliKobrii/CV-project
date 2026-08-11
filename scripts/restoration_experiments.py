"""The local small-scale experiments for RQ3, run on the laptop CPU and
MPS (the Apple GPU backend).

The script produces:
  tables/restoration_intrinsic.csv             PSNR, SSIM and runtime per method
  figures/restoration_grid_<cond>.png          qualitative strips
  figures/restoration_wiener_k.png             the Wiener K sweep and the PSF mismatch check
  tables/restoration_extrinsic_detection.csv   AP50 on clean, degraded and restored frames
  tables/restoration_extrinsic_pose.csv        PCK on clean, degraded and restored frames

The full-scale versions are provided in notebook 05.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from config import FIGURES_DIR, MODELS_DIR, TABLES_DIR, VISDRONE_PERSON  # noqa: E402
from eval_restore import (conditions, degrade_by, detection_under_degradation,  # noqa: E402
                          example_grid, pose_under_degradation, psnr,
                          run_intrinsic, write_csv)
from restore import motion_psf, wiener_filter  # noqa: E402


def load_val_images(n, max_width=1280):
    """Load up to n validation images, resized to the given width."""
    out = []
    for p in sorted((VISDRONE_PERSON / "images" / "val").glob("*.jpg"))[:n]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = min(1.0, max_width / im.shape[1])
        out.append(cv2.resize(im, None, fx=s, fy=s) if s < 1.0 else im)
    return out


def wiener_k_sweep(img, out_png):
    """Plot PSNR against K on a logarithmic sweep, and measure how much a
    slightly wrong PSF costs at the best K."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond = next(c for c in conditions() if c["name"] == "motion25_n5")
    deg = degrade_by(cond, img, seed=0)
    Ks = np.logspace(-4, 0, 13)
    vals = [psnr(wiener_filter(deg, cond["psf"], K), img) for K in Ks]
    bestK = float(Ks[int(np.argmax(vals))])
    mismatch = {
        "exact_psf": psnr(wiener_filter(deg, cond["psf"], bestK), img),
        "angle+10deg": psnr(wiener_filter(deg, motion_psf(25, 30), bestK), img),
        "len_19": psnr(wiener_filter(deg, motion_psf(19, 20), bestK), img),
        "len_31": psnr(wiener_filter(deg, motion_psf(31, 20), bestK), img),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(Ks, vals, marker="o")
    ax.axhline(psnr(deg, img), ls="--", c="gray",
               label="Degraded (no restoration)")
    ax.set_xlabel("Wiener K (noise-to-signal ratio)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Wiener K sensitivity – motion25_n5")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved ->", out_png)
    print(f"Best K {bestK:.4g} | PSF mismatch:",
          {k: round(v, 2) for k, v in mismatch.items()})
    return bestK, mismatch


def main(args):
    """Run the intrinsic evaluation, the K sweep and, unless skipped, the
    extrinsic evaluations."""
    imgs = load_val_images(args.n_images)
    print(f"[rq3] {len(imgs)} VisDrone validation frames (width at most 1280)")

    run_intrinsic(imgs, out_csv=TABLES_DIR / "restoration_intrinsic.csv")
    for cname in ("motion25_n5", "noise25"):
        cond = next(c for c in conditions() if c["name"] == cname)
        example_grid(imgs[0], cond,
                     FIGURES_DIR / f"restoration_grid_{cname}.png")
    wiener_k_sweep(imgs[0], FIGURES_DIR / "restoration_wiener_k.png")

    if args.skip_extrinsic:
        return
    import torch
    ft = MODELS_DIR / "yolo11n_visdrone_ft" / "weights" / "best.pt"
    weights = str(ft) if ft.exists() else "yolo11n.pt"
    device = "mps" if torch.backends.mps.is_available() else None
    conds = {c["name"]: c for c in conditions()}
    print(f"[rq3] Extrinsic detection with {weights} on {device or 'cpu'}")
    det_rows = detection_under_degradation(
        weights, conds["motion25_n5"], ["wiener", "rl20", "unsharp"],
        max_images=args.det_images, device=device)
    det_rows += detection_under_degradation(
        weights, conds["noise25"], ["median", "nlm", "butterworth"],
        max_images=args.det_images, device=device)
    write_csv(det_rows, TABLES_DIR / "restoration_extrinsic_detection.csv")

    print("[rq3] Extrinsic pose (UAV-Human, RTMPose on CPU)")
    pose_rows = pose_under_degradation(conds["motion15_n2"], ["wiener"],
                                       max_persons=args.pose_persons)
    pose_rows += pose_under_degradation(conds["noise25"],
                                        ["median", "butterworth"],
                                        max_persons=args.pose_persons)
    write_csv(pose_rows, TABLES_DIR / "restoration_extrinsic_pose.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--det-images", type=int, default=60)
    ap.add_argument("--pose-persons", type=int, default=200)
    ap.add_argument("--skip-extrinsic", action="store_true")
    main(ap.parse_args())
