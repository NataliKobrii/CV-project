"""The local small-scale experiments for RQ4.

The script produces:
  models/tiny_desc.pt                  the learned descriptor, trained locally
  tables/matching_local.csv            metrics per method and condition
  tables/matching_pr_local.csv         precision and recall against the ratio threshold
  figures/matching_mma_local.png       matching accuracy against the pixel threshold
  figures/matching_pr_local.png        precision-recall curves for the viewpoint pairs
  figures/registration_mosaic_*.png    the mosaic of the drone sweep
  figures/victim_map_*.png             the triage states on one map
  tables/registration_stats_*.json

The benchmark uses synthetic warps of VisDrone validation frames with
controlled viewpoint, scale and illumination changes. The descriptor trains
on frames from the training split, so there is no image overlap. The full
HPatches protocol is provided in notebook 06.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from config import FIGURES_DIR, MODELS_DIR, OKUTAMA_DIR, TABLES_DIR, VISDRONE_PERSON  # noqa: E402
from eval_match import aggregate, evaluate_matcher, pr_by_ratio, synthetic_pairs  # noqa: E402
from eval_restore import write_csv  # noqa: E402
from match import build_features, patch_pairs_from_image, train_descriptor  # noqa: E402
import register  # noqa: E402


def load_images(split, n, max_width=1280, offset=0):
    """Load up to n images from a VisDrone split, resized to the given width."""
    out = []
    for p in sorted((VISDRONE_PERSON / "images" / split)
                    .glob("*.jpg"))[offset:offset + n]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = min(1.0, max_width / im.shape[1])
        out.append(cv2.resize(im, None, fx=s, fy=s) if s < 1.0 else im)
    return out


def build_benchmark(images, n_per_image=3, seed=0):
    """Build the synthetic benchmark pairs for all three conditions."""
    pairs = []
    for cond in ("viewpoint", "scale", "illumination"):
        pairs += synthetic_pairs(images, cond, n_per_image=n_per_image,
                                 seed=seed)
    return pairs


def train_tinydesc(device, n_images=40, pairs_per_image=32, rounds=4,
                   epochs=6):
    """Train TinyDescNet on self-supervised patch pairs and save the weights."""
    rng = np.random.default_rng(0)
    imgs = load_images("train", n_images)
    pa, pb = [], []
    for _ in range(rounds):
        for im in imgs:
            a, b = patch_pairs_from_image(im, rng, n=pairs_per_image)
            if len(a):
                pa.append(a)
                pb.append(b)
    pa, pb = np.concatenate(pa), np.concatenate(pb)
    print(f"[rq4] Training TinyDescNet on {len(pa)} patch pairs "
          f"({len(imgs)} train frames), device={device}")
    net = train_descriptor(pa, pb, epochs=epochs, device=device)
    out = MODELS_DIR / "tiny_desc.pt"
    torch.save(net.state_dict(), out)
    print("Saved ->", out)


def make_figures(rows_all, pr_rows):
    """Plot the matching-accuracy curves and the precision-recall curves."""
    display = {"sift": "SIFT", "orb": "ORB", "tinydesc": "TinyDesc"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ths = (1, 3, 5, 10)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in sorted({r["method"] for r in rows_all}):
        rs = [r for r in rows_all if r["method"] == m]
        ax.plot(ths, [np.mean([r[f"MMA@{t}"] for r in rs]) for t in ths],
                marker="o", label=display.get(m, m))
    ax.set_xlabel("Reprojection threshold (px)")
    ax.set_ylabel("MMA")
    ax.set_title("Mean matching accuracy – synthetic aerial benchmark")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matching_mma_local.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in sorted({r["method"] for r in pr_rows}):
        rs = sorted((r for r in pr_rows if r["method"] == m),
                    key=lambda r: r["ratio"])
        ax.plot([r["recall@3"] for r in rs], [r["precision@3"] for r in rs],
                marker="o", label=display.get(m, m))
    ax.set_xlabel("Recall at 3 px")
    ax.set_ylabel("Precision at 3 px (MMA)")
    ax.set_title("PR as the ratio-test threshold sweeps 0.6 – 0.95")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matching_pr_local.png", dpi=150)
    plt.close(fig)
    print("Saved -> matching_mma_local.png, matching_pr_local.png")


def main(args):
    """Run the descriptor benchmark and, unless skipped, the registration
    demonstration."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bench = build_benchmark(load_images("val", args.bench_images),
                            n_per_image=args.warps)
    print(f"[rq4] Benchmark: {len(bench)} pairs")

    if args.retrain or not (MODELS_DIR / "tiny_desc.pt").exists():
        train_tinydesc(device)

    rows_all, pr_rows = [], []
    for kind in ("sift", "orb", "tinydesc"):
        feats = build_features(kind, device=device)
        rows_all += evaluate_matcher(feats, bench, kind)
        pr_rows += pr_by_ratio(
            feats, [p for p in bench if p[3] == "viewpoint"], kind)
        print(f"[rq4] Evaluated {kind}")
    agg = aggregate(rows_all)
    for r in agg:
        print(r)
    write_csv(agg, TABLES_DIR / "matching_local.csv")
    write_csv(pr_rows, TABLES_DIR / "matching_pr_local.csv")
    make_figures(rows_all, pr_rows)

    if args.skip_registration:
        return
    ft = MODELS_DIR / "yolo11n_visdrone_ft" / "weights" / "best.pt"
    weights = str(ft) if ft.exists() else "yolo11n.pt"
    register.render_map(
        OKUTAMA_DIR / "1.1.1.mov", step=args.reg_step,
        max_frames=args.reg_frames, weights=weights,
        device="mps" if device == "mps" else None, pose_device="cpu",
        gt_labels=(str(OKUTAMA_DIR / "1.1.1.txt") if args.gt_states else None),
        caption="EECS 4422 - SAR sweep map (Okutama sample, chained homographies)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-images", type=int, default=10)
    ap.add_argument("--warps", type=int, default=3)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-registration", action="store_true")
    ap.add_argument("--gt-states", action="store_true",
                    help="Victim-map states from the Okutama ground truth instead of the pipeline.")
    ap.add_argument("--reg-step", type=int, default=12)
    ap.add_argument("--reg-frames", type=int, default=288)
    main(ap.parse_args())
