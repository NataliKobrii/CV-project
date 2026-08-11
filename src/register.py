"""Frame registration for RQ4: from feature matches to a search-area mosaic
and a victim map.

Every Nth frame is registered to the previous one with feature matches and a
RANSAC homography. This assumes the scene is roughly planar, which is
reasonable for high-altitude, near-nadir footage over flat ground. Chaining
the pairwise homographies maps every frame back to the first one, so all
frames warp into a single mosaic. Each track's foot point (the bottom center
of its box) is projected the same way and drawn in its triage color, which
turns the per-frame triage overlay into one map of the whole sweep.

A known limitation is that chained homographies drift on long sweeps.
Loop closure or bundle adjustment would address this and is left as future
work.
"""
import argparse
import bisect
import json
from collections import defaultdict, deque
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import (FIGURES_DIR, STATE_COLORS, STATE_PRIORITY, TABLES_DIR,  # noqa: E402
                    WINDOW)
from match import build_features, estimate_homography, match_descriptors  # noqa: E402


def pairwise_homography(feats, img_src, img_dst, ratio=0.8):
    """Estimate the homography that maps img_src pixels into img_dst.
    Returns the homography (or None) and the RANSAC inlier count."""
    xy1, d1, _ = feats.detect_describe(img_src)
    xy2, d2, _ = feats.detect_describe(img_dst)
    idx, _ = match_descriptors(d1, d2, feats.norm, ratio)
    if len(idx) < 8:
        return None, 0
    H, inl = estimate_homography(xy1[idx[:, 0]], xy2[idx[:, 1]])
    return H, int(inl.sum())


def chain_to_first(hs):
    """Turn the pairwise list [H 1->0, H 2->1, ...] into the cumulative
    transforms [I, H 1->0, H 2->0, ...] that map every frame to frame 0."""
    # Multiplying the successive pairwise homographies accumulates them into
    # a single transform from each frame back to the first frame. This is why
    # small per-pair errors slowly build up (drift) over a long sequence.
    out = [np.eye(3)]
    for H in hs:
        out.append(out[-1] @ H)
    return out


def mosaic_transform(shapes, chains, mosaic_width=2200):
    """Return a global scale-and-translation transform A that places every
    warped frame onto one canvas, together with the canvas size."""
    corners = []
    for (h, w), C in zip(shapes, chains):
        c = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, C))
    # The bounding box of all warped frame corners gives the size of the
    # canvas needed to hold the whole mosaic.
    allc = np.concatenate(corners).reshape(-1, 2)
    x0, y0 = allc.min(0)
    x1, y1 = allc.max(0)
    # A single scale and translation map that bounding box into a canvas of
    # the requested width.
    s = mosaic_width / max(x1 - x0, 1.0)
    A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1.0]])
    return A, (int(np.ceil((x1 - x0) * s)), int(np.ceil((y1 - y0) * s)))


def render_map(video_path, out_dir=FIGURES_DIR, step=12, max_frames=288,
               proc_width=1920, weights="yolo11n.pt", imgsz=1280, device=None,
               pose_device="cpu", gt_labels=None, feature_kind="sift",
               mosaic_width=2200, obs_every=3, caption=None):
    """Build the mosaic and the victim map for one video. The states come
    from the live pipeline (tracking, pose and classification) or, if
    gt_labels is given, from the Okutama ground truth."""
    video_path = Path(video_path)
    feats = build_features(feature_kind)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    gt = None
    if gt_labels:
        from data.okutama import parse_annotations
        gt = parse_annotations(gt_labels)
    else:
        from pose import PoseEstimator
        from render_demo import StateClassifier
        from track import PersonTracker
        tracker = PersonTracker(weights=weights, imgsz=imgsz, device=device)
        pose = PoseEstimator(device=pose_device)
        classifier = StateClassifier()
        hist = defaultdict(lambda: deque(maxlen=WINDOW))
        print(f"[register] States from the pipeline "
              f"(pose = {pose.backend}, classifier = {classifier.kind})")

    reg_frames, obs = [], []
    track_state = {}
    scale, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        if scale is None:
            scale = min(1.0, proc_width / frame.shape[1])
        small = (cv2.resize(frame, None, fx=scale, fy=scale)
                 if scale < 1.0 else frame)
        if n % step == 0:
            reg_frames.append((n, small.copy()))
        if gt is not None:
            if n in gt and n % obs_every == 0:
                for b in gt[n]:
                    x1, y1, x2, y2 = b.box
                    obs.append((n, b.track_id,
                                ((x1 + x2) / 2 * scale, y2 * scale), b.state))
                    track_state[b.track_id] = b.state
        else:
            tracks = tracker.update(small)
            boxes = np.array([t[1] for t in tracks],
                             np.float32).reshape(-1, 4)
            kpts, scores = pose(small, boxes)
            for (tid, box, _c), kp, sc in zip(tracks, kpts, scores):
                hist[tid].append({"kpts": kp, "scores": sc, "box": box})
                st = classifier(hist[tid])
                track_state[tid] = st
                if n % obs_every == 0:
                    obs.append((n, tid, ((box[0] + box[2]) / 2, box[3]), st))
        n += 1
        if n % 60 == 0:
            print(f"  Frame {n}: {len(obs)} observations, "
                  f"{len(reg_frames)} registration frames")
    cap.release()

    # Register every step-th frame to the previous one, then chain the
    # homographies back to frame 0.
    hs, inliers = [], []
    for (_, prev), (_, cur) in zip(reg_frames[:-1], reg_frames[1:]):
        H, ninl = pairwise_homography(feats, cur, prev)
        if H is None:
            H = np.eye(3)
        hs.append(H)
        inliers.append(ninl)
    chains = chain_to_first(hs)
    shapes = [f.shape[:2] for _, f in reg_frames]
    A, size = mosaic_transform(shapes, chains, mosaic_width)
    mosaic = np.zeros((size[1], size[0], 3), np.uint8)
    for (_, f), C in zip(reg_frames, chains):
        warped = cv2.warpPerspective(f, A @ C, size)
        mask = warped.any(-1).astype(np.uint8)
        # Erode so the interpolated frame borders do not stripe the mosaic.
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
        mask &= ~mosaic.any(-1)                   # The first frame is kept.
        mosaic[mask] = warped[mask]

    # Project every observation through the homography of the nearest
    # registered frame at or before it.
    vmap = mosaic.copy()
    keys = [nf for nf, _ in reg_frames]
    for nf, _tid, (fx, fy), st in obs:
        k = bisect.bisect_right(keys, nf) - 1
        M = A @ chains[k]
        p = cv2.perspectiveTransform(
            np.array([[[fx, fy]]], np.float64), M).ravel()
        cv2.circle(vmap, (int(p[0]), int(p[1])), 4,
                   STATE_COLORS.get(st, STATE_COLORS["unknown"]), -1,
                   cv2.LINE_AA)

    counts = defaultdict(int)
    for st in track_state.values():
        counts[st] += 1
    order = sorted(counts, key=lambda s: STATE_PRIORITY.get(s, 9))
    box_h = 34 + 26 * (len(order) + 1)
    over = vmap.copy()
    cv2.rectangle(over, (10, 10), (360, 10 + box_h), (0, 0, 0), -1)
    vmap = cv2.addWeighted(over, 0.65, vmap, 0.35, 0)
    cv2.putText(vmap, "VICTIM MAP  (tracks by state)", (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    for i, st in enumerate(order):
        y = 38 + 26 * (i + 1)
        cv2.circle(vmap, (30, y - 5), 6,
                   STATE_COLORS.get(st, STATE_COLORS["unknown"]), -1,
                   cv2.LINE_AA)
        cv2.putText(vmap, f"{st.capitalize()}: {counts[st]}", (46, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    if caption:
        (tw, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(vmap, caption, ((size[0] - tw) // 2, size[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic_path = out_dir / f"registration_mosaic_{video_path.stem}.png"
    map_path = out_dir / f"victim_map_{video_path.stem}.png"
    cv2.imwrite(str(mosaic_path), mosaic)
    cv2.imwrite(str(map_path), vmap)
    stats = {"video": video_path.name, "frames": n,
             "reg_frames": len(reg_frames), "step": step,
             "features": feature_kind,
             "mean_inliers": round(float(np.mean(inliers)), 1) if inliers else 0,
             "min_inliers": int(min(inliers)) if inliers else 0,
             "canvas": list(size),
             "tracks_by_state": dict(sorted(counts.items())),
             "state_source": "gt-labels" if gt is not None else "pipeline"}
    (TABLES_DIR / f"registration_stats_{video_path.stem}.json").write_text(
        json.dumps(stats, indent=2))
    print(f"[register] Wrote {mosaic_path.name}, {map_path.name}")
    print("[register] Stats:", stats)
    return mosaic_path, map_path, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=288)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--gt-labels", default=None,
                    help="Okutama .txt to use ground-truth states instead of the pipeline")
    ap.add_argument("--features", default="sift",
                    choices=["sift", "orb", "tinydesc"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--pose-device", default="cpu")
    ap.add_argument("--caption", default=None)
    args = ap.parse_args()
    render_map(args.video, step=args.step, max_frames=args.max_frames,
               weights=args.weights, gt_labels=args.gt_labels,
               feature_kind=args.features, device=args.device,
               pose_device=args.pose_device, caption=args.caption)
