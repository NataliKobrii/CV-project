"""Evaluation for RQ4: matching accuracy under a known homography.

The protocol follows the common mean matching accuracy setup. We detect
and describe both images, match with the ratio test, and project the
matched reference keypoints through the ground-truth homography. A match
counts as correct if its reprojection error is under a pixel threshold.

We report MMA@t (the fraction of matches that are correct at threshold t),
recall@3 (correct matches over covisible reference keypoints), the corner
error of the RANSAC homography against the ground truth, and the RANSAC
inlier count, which is what registration actually consumes.

The same code runs on two data sources: the real HPatches sequences
(downloaded in notebook 06) and synthetic warps of our own aerial frames,
which give a local benchmark with controlled viewpoint, scale and
illumination axes and no large download.
"""
import time
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import HPATCHES_DIR, VISDRONE_PERSON  # noqa: E402
from match import (build_features, estimate_homography, match_descriptors,  # noqa: E402
                   photometric_jitter, random_homography)


def project(xy, H):
    """Apply the homography H to an array of points [N, 2]."""
    if len(xy) == 0:
        return xy.astype(np.float32)
    p = cv2.perspectiveTransform(xy.reshape(-1, 1, 2).astype(np.float64), H)
    return p.reshape(-1, 2).astype(np.float32)


def corner_error(H_est, H_gt, shape):
    """Return the mean distance between the image corners mapped by the
    estimated homography and by the ground-truth homography."""
    h, w = shape[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return float(np.linalg.norm(project(c, H_est) - project(c, H_gt),
                                axis=1).mean())


def pair_data(feats, img1, img2):
    """Detect and describe both images once, so that a ratio sweep does not
    have to repeat the detection."""
    t0 = time.perf_counter()
    xy1, d1, _ = feats.detect_describe(img1)
    xy2, d2, _ = feats.detect_describe(img2)
    return {"xy1": xy1, "d1": d1, "xy2": xy2, "d2": d2, "norm": feats.norm,
            "ms_desc": (time.perf_counter() - t0) * 1000.0}


def score_pair(pd, H_gt, shape1, shape2, thresholds=(1, 3, 5, 10), ratio=0.8):
    """Match the precomputed descriptors of one pair and score the matches
    against the ground-truth homography."""
    t0 = time.perf_counter()
    idx, _ = match_descriptors(pd["d1"], pd["d2"], pd["norm"], ratio)
    ms_match = (time.perf_counter() - t0) * 1000.0
    h2, w2 = shape2[:2]
    proj_all = project(pd["xy1"], H_gt)
    # A reference keypoint is covisible when its projection falls inside
    # the target image.
    covis = ((proj_all[:, 0] > -3) & (proj_all[:, 0] < w2 + 3)
             & (proj_all[:, 1] > -3) & (proj_all[:, 1] < h2 + 3))
    res = {"n_kp1": len(pd["xy1"]), "n_kp2": len(pd["xy2"]),
           "n_covis": int(covis.sum()), "n_matches": int(len(idx)),
           "ratio": ratio, "ms": round(pd["ms_desc"] + ms_match, 1)}
    if len(idx):
        # The reprojection error is the distance between the projected
        # reference keypoint and its matched target keypoint.
        err = np.linalg.norm(proj_all[idx[:, 0]] - pd["xy2"][idx[:, 1]],
                             axis=1)
    else:
        err = np.zeros(0, np.float32)
    for t in thresholds:
        res[f"MMA@{t}"] = float((err <= t).mean()) if len(err) else 0.0
    res["recall@3"] = float((err <= 3).sum() / max(int(covis.sum()), 1))
    if len(idx) >= 4:
        H_est, inl = estimate_homography(pd["xy1"][idx[:, 0]],
                                         pd["xy2"][idx[:, 1]])
        res["n_inliers"] = int(inl.sum())
        res["H_corner_err"] = (round(corner_error(H_est, H_gt, shape1), 2)
                               if H_est is not None else float("nan"))
    else:
        res["n_inliers"], res["H_corner_err"] = 0, float("nan")
    return res


def evaluate_pair(feats, img1, img2, H_gt, thresholds=(1, 3, 5, 10),
                  ratio=0.8):
    """Detect, describe, match and score one image pair."""
    return score_pair(pair_data(feats, img1, img2), H_gt, img1.shape,
                      img2.shape, thresholds, ratio)


def evaluate_matcher(feats, pairs, method_name, ratio=0.8, verbose=False):
    """Evaluate one method on an iterable of (img1, img2, H_gt, condition)
    pairs. Returns one result row per pair."""
    rows = []
    for img1, img2, H_gt, cond in pairs:
        r = evaluate_pair(feats, img1, img2, H_gt, ratio=ratio)
        r.update({"method": method_name, "condition": cond})
        rows.append(r)
        if verbose:
            print({k: r[k] for k in ("method", "condition", "n_matches",
                                     "MMA@3", "H_corner_err")})
    return rows


def pr_by_ratio(feats, pairs, method_name,
                ratios=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    """Compute precision and recall at 3 pixels while the ratio-test
    threshold sweeps. Features are computed once per pair and only the
    matching is redone for each ratio."""
    cached = [(pair_data(feats, i1, i2), H, i1.shape, i2.shape)
              for i1, i2, H, _ in pairs]
    rows = []
    for r in ratios:
        scored = [score_pair(pd, H, s1, s2, ratio=r)
                  for pd, H, s1, s2 in cached]
        rows.append({"method": method_name, "ratio": r,
                     "precision@3": round(float(np.mean(
                         [s["MMA@3"] for s in scored])), 4),
                     "recall@3": round(float(np.mean(
                         [s["recall@3"] for s in scored])), 4),
                     "n_matches": round(float(np.mean(
                         [s["n_matches"] for s in scored])), 1)})
    return rows


def aggregate(rows, by=("method", "condition")):
    """Group the rows by the given keys and average every numeric field."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        agg = dict(zip(by, key))
        for f in rs[0]:
            if f in by or not isinstance(rs[0][f], (int, float)):
                continue
            vals = [r[f] for r in rs
                    if isinstance(r[f], (int, float)) and np.isfinite(r[f])]
            if vals:
                agg[f] = round(float(np.mean(vals)), 4)
        agg["n_pairs"] = len(rs)
        out.append(agg)
    return out


def synthetic_pairs(images, condition, n_per_image=3, seed=0):
    """Build (reference, warped, H_gt, condition) tuples from aerial frames.

    The viewpoint condition applies rotation and perspective jitter, the
    scale condition zooms about the center, and the illumination condition
    changes only the photometry, with the identity homography. This matches
    the HPatches i-sequences.
    """
    rng = np.random.default_rng(seed)
    out = []
    for img in images:
        h, w = img.shape[:2]
        for _ in range(n_per_image):
            if condition == "viewpoint":
                H = random_homography(img.shape, rng, max_rot=30,
                                      max_persp=0.15, scale=(0.85, 1.2))
                warped = cv2.warpPerspective(img, H, (w, h))
            elif condition == "scale":
                s = float(rng.choice([0.5, 0.65, 1.5, 1.9]))
                H = np.diag([s, s, 1.0])
                H[:2, 2] = [(1 - s) * w / 2, (1 - s) * h / 2]
                warped = cv2.warpPerspective(img, H, (w, h))
            elif condition == "illumination":
                H = np.eye(3)
                warped = photometric_jitter(img, rng)
            else:
                raise ValueError(condition)
            out.append((img, warped, H, condition))
    return out


def hpatches_sequences(root=HPATCHES_DIR, max_seqs=None, resize_width=None):
    """Yield (reference, target, H_gt, condition, sequence name, target
    index) over an HPatches sequences folder.

    Every sequence directory holds the images 1.ppm to 6.ppm and the
    homographies H_1_2 to H_1_6. Sequences with the v_ prefix vary the
    viewpoint and those with i_ vary the illumination. resize_width shrinks
    both images and rescales the homography to match.
    """
    seqs = sorted(p for p in Path(root).iterdir() if p.is_dir())
    if max_seqs:
        seqs = seqs[:max_seqs]
    for seq in seqs:
        ref = cv2.imread(str(seq / "1.ppm"))
        if ref is None:
            continue
        cond = "viewpoint" if seq.name.startswith("v_") else "illumination"
        for t in range(2, 7):
            tgt = cv2.imread(str(seq / f"{t}.ppm"))
            hfile = seq / f"H_1_{t}"
            if tgt is None or not hfile.exists():
                continue
            H = np.loadtxt(str(hfile)).reshape(3, 3)
            r, g = ref, tgt
            if resize_width:
                s1 = min(1.0, resize_width / r.shape[1])
                s2 = min(1.0, resize_width / g.shape[1])
                r = cv2.resize(r, None, fx=s1, fy=s1)
                g = cv2.resize(g, None, fx=s2, fy=s2)
                H = np.diag([s2, s2, 1.0]) @ H @ np.diag([1 / s1, 1 / s1, 1.0])
            yield r, g, H, cond, seq.name, t


if __name__ == "__main__":
    # A quick self-check: SIFT against ORB on synthetic viewpoint warps.
    imgs = []
    for p in sorted((VISDRONE_PERSON / "images" / "val").glob("*.jpg"))[:3]:
        im = cv2.imread(str(p))
        if im is not None:
            s = min(1.0, 1280 / im.shape[1])
            imgs.append(cv2.resize(im, None, fx=s, fy=s))
    pairs = synthetic_pairs(imgs, "viewpoint", n_per_image=2)
    for kind in ("sift", "orb"):
        rows = evaluate_matcher(build_features(kind), pairs, kind)
        print(aggregate(rows))
