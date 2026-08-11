"""The local small-scale RQ2 experiment on the Okutama sample video.

The pipeline takes the ground-truth boxes and action labels from 1.1.1.txt,
runs RTMPose on them, builds track windows, and compares the PoseMLP on
pose features against the AppearanceCNN on raw crops. The split is by track
id (70 to 30) to avoid leakage.

This is deliberately small: one video. The full experiment runs in
notebooks/03_state_classification.ipynb on the complete Okutama set. The
script saves results/tables/rq2_local.csv, the confusion figures and
models/pose_mlp.pt.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import MODELS_DIR, OKUTAMA_DIR, TABLES_DIR, WINDOW  # noqa: E402
from data.okutama import parse_annotations  # noqa: E402
from classify import (AppearanceCNN, PoseMLP, evaluate, predict,  # noqa: E402
                      train_classifier)
from features import windows_from_track  # noqa: E402
from pose import PoseEstimator  # noqa: E402


def extract(video_path, frames, step=2, max_frames=None, crop_size=64,
            pose_device="cpu"):
    """Decode the video sequentially, run pose on the ground-truth boxes,
    and collect the per-track data."""
    est = PoseEstimator(device=pose_device)
    print(f"[extract] Pose backend: {est.backend}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    tracks = defaultdict(list)   # Maps tid -> [{frame, kpts, scores, box, state}].
    crops = {}                   # Maps (tid, frame) -> a 64x64 uint8 crop.
    wanted = {f for f in frames if f % step == 0}
    if max_frames:
        wanted = {f for f in wanted if f < max_frames}
    last = max(wanted, default=-1)
    idx, used = 0, 0
    while idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        if idx in wanted:
            boxes_meta = [b for b in frames[idx] if b.state != "unknown"]
            if boxes_meta:
                bxs = np.array([b.box for b in boxes_meta], np.float32)
                h, w = img.shape[:2]
                bxs[:, [0, 2]] = bxs[:, [0, 2]].clip(0, w - 1)
                bxs[:, [1, 3]] = bxs[:, [1, 3]].clip(0, h - 1)
                kpts, scores = est(img, bxs)
                for b, kp, sc, bx in zip(boxes_meta, kpts, scores, bxs):
                    x1, y1, x2, y2 = map(int, bx)
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    crop = cv2.resize(img[y1:y2, x1:x2], (crop_size, crop_size))
                    tracks[b.track_id].append(
                        {"frame": idx, "kpts": kp, "scores": sc,
                         "box": bx.copy(), "state": b.state})
                    crops[(b.track_id, idx)] = crop
                used += 1
                if used % 100 == 0:
                    print(f"  Processed {used} labeled frames (video frame {idx})")
        idx += 1
    cap.release()
    print(f"[extract] {used} frames, {len(tracks)} tracks, {len(crops)} crops")
    return dict(tracks), crops


def build_dataset(tracks, crops, window=WINDOW):
    """Turn the track windows into pose features X, crop tensors C, labels y
    and track ids."""
    X, C, y, tids = [], [], [], []
    for tid, seq in tracks.items():
        feats, labels = windows_from_track(seq, window=window)
        stride = max(window // 3, 1)
        for w_i, (f, lab) in enumerate(zip(feats, labels)):
            if lab is None:
                continue
            # The crop from the middle frame represents the window for the
            # appearance model.
            mid = seq[min(w_i * stride + window // 2, len(seq) - 1)]
            crop = crops.get((tid, mid["frame"]))
            if crop is None:
                continue
            X.append(f)
            C.append(crop)
            y.append(lab)
            tids.append(tid)
    X = np.stack(X); C = np.stack(C)
    classes = sorted(set(y))
    y_idx = np.array([classes.index(v) for v in y])
    print(f"[dataset] {len(X)} windows, classes: "
          f"{ {c: int((y_idx == i).sum()) for i, c in enumerate(classes)} }")
    return X, C, y_idx, np.array(tids), classes


def split_by_track(tids, val_frac=0.3, seed=0):
    """Split the windows into training and validation sets by track identity."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(tids.tolist())))
    rng.shuffle(unique)
    n_val = max(int(len(unique) * val_frac), 1)
    val_set = set(unique[:n_val].tolist())
    val_mask = np.array([t in val_set for t in tids])
    return ~val_mask, val_mask


def main(max_frames=None, epochs=40, step=2, use_cache=True):
    """Extract or load the dataset, train both classifiers and write the
    results table."""
    cache = OKUTAMA_DIR.parent / "rq2_cache.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, C, y, tids = z["X"], z["C"], z["y"], z["tids"]
        classes = list(z["classes"])
        print(f"[cache] Loaded {len(X)} windows from {cache}")
    else:
        txt = OKUTAMA_DIR / "1.1.1.txt"
        video = OKUTAMA_DIR / "1.1.1.mov"
        frames = parse_annotations(txt)
        tracks, crops = extract(video, frames, step=step, max_frames=max_frames)
        X, C, y, tids, classes = build_dataset(tracks, crops)
        np.savez_compressed(cache, X=X, C=C, y=y, tids=tids,
                            classes=np.array(classes))
        print(f"[cache] Saved -> {cache}")
    tr, va = split_by_track(tids)
    print(f"[split] Training windows {tr.sum()}, validation windows {va.sum()} "
          f"({len(set(tids[va].tolist()))} held-out tracks)")
    # The class weights balance the loss across the unequal state counts.
    counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1) / len(classes)

    rows = []
    # The pose-based model.
    pose_model, _ = train_classifier(PoseMLP(len(classes)), X[tr], y[tr],
                                     X[va], y[va], epochs=epochs,
                                     class_weights=weights)
    m = evaluate(y[va], predict(pose_model, X[va]), classes, "pose_mlp_local")
    rows.append({"model": "PoseMLP (pose features)", **flatten(m, classes)})
    import torch
    torch.save(pose_model.state_dict(), MODELS_DIR / "pose_mlp.pt")
    (MODELS_DIR / "pose_mlp_classes.json").write_text(json.dumps(classes))
    print("Saved", MODELS_DIR / "pose_mlp.pt")

    # The appearance-based model.
    Cn = np.ascontiguousarray(
        (C.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))  # Channels move to the second axis.
    cnn, _ = train_classifier(AppearanceCNN(len(classes)), Cn[tr], y[tr],
                              Cn[va], y[va], epochs=epochs,
                              class_weights=weights)
    m = evaluate(y[va], predict(cnn, Cn[va]), classes, "appearance_cnn_local")
    rows.append({"model": "AppearanceCNN (raw crops)", **flatten(m, classes)})

    out = TABLES_DIR / "rq2_local.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print("Wrote", out)


def flatten(m, classes):
    """Flatten a metrics dictionary into one row for the results table."""
    row = {"macro_f1": round(m["macro_f1"], 4), "accuracy": round(m["accuracy"], 4)}
    for c in classes:
        row[f"f1_{c}"] = round(m["per_class_f1"][c], 4)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap on the video frames for quick runs.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()
    main(max_frames=args.max_frames, epochs=args.epochs, step=args.step)
