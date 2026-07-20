"""Pose -> feature vectors for state classification (RQ2).

Static features are computed per frame from 17 COCO keypoints + the person
box; temporal features are computed over a short track window and capture
what separates the triage states: overall body-axis orientation (lying vs
upright), limb elevation (waving), and motion energy (mobile vs stationary).
"""
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import WINDOW  # noqa: E402

# 34 (normalized kpts) + 7 (geometry) static, meaned over the window, + 6 temporal
FEATURE_DIM = 34 + 7 + 6

L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, NOSE = 5, 6, 9, 10, 11, 12, 0


def _mid(kpts, a, b):
    return (kpts[a] + kpts[b]) / 2.0


def normalize_kpts(kpts, box):
    """Center on mid-hip, scale by torso length (fallback: box diagonal)."""
    mid_hip = _mid(kpts, L_HIP, R_HIP)
    mid_sh = _mid(kpts, L_SH, R_SH)
    torso = float(np.linalg.norm(mid_sh - mid_hip))
    if torso < 1e-3:
        x1, y1, x2, y2 = box
        torso = max(float(np.hypot(x2 - x1, y2 - y1)) / 3.0, 1e-3)
    return ((kpts - mid_hip) / torso).reshape(-1)  # 34


def static_features(kpts, scores, box):
    """7 geometric features for one frame."""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    mid_hip = _mid(kpts, L_HIP, R_HIP)
    mid_sh = _mid(kpts, L_SH, R_SH)
    axis = mid_sh - mid_hip  # points from hips toward shoulders
    # angle of the body axis from vertical: 0 = upright, 90 = horizontal (lying)
    angle = np.degrees(np.arctan2(abs(float(axis[0])), abs(float(axis[1])) + 1e-6))
    aspect = h / w
    # note: image y grows downward, so "above" means smaller y
    l_wr_up_sh = float(kpts[L_WR][1] < mid_sh[1])
    r_wr_up_sh = float(kpts[R_WR][1] < mid_sh[1])
    l_wr_up_nose = float(kpts[L_WR][1] < kpts[NOSE][1])
    r_wr_up_nose = float(kpts[R_WR][1] < kpts[NOSE][1])
    return np.array([
        angle / 90.0, min(aspect, 5.0) / 5.0,
        l_wr_up_sh, r_wr_up_sh, l_wr_up_nose, r_wr_up_nose,
        float(np.mean(scores)),
    ], np.float32)


def temporal_features(kpts_seq, boxes_seq):
    """6 motion features over a window. Distances normalized by box size."""
    boxes = np.asarray(boxes_seq, np.float32)
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    scale = float(np.mean(sizes)) + 1e-3
    kseq = np.asarray(kpts_seq, np.float32)  # [T,17,2]
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                        (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    # wrist positions relative to the box center (removes global motion)
    rel = kseq[:, [L_WR, R_WR], :] - centers[:, None, :]
    wrist_y_std = np.std(rel[:, :, 1], axis=0) / scale            # 2
    wrist_speed = (np.mean(np.abs(np.diff(rel, axis=0)), axis=(0, 2)) / scale
                   if len(kseq) > 1 else np.zeros(2))             # 2
    center_speed = (float(np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))) / scale
                    if len(centers) > 1 else 0.0)                 # 1
    size_change = float(np.std(sizes)) / scale                    # 1
    return np.concatenate([wrist_y_std, wrist_speed,
                           [center_speed, size_change]]).astype(np.float32)


def window_feature(kpts_seq, scores_seq, boxes_seq):
    """One FEATURE_DIM vector for a track window (lists of per-frame arrays)."""
    stat = np.mean(
        [np.concatenate([normalize_kpts(k, b), static_features(k, s, b)])
         for k, s, b in zip(kpts_seq, scores_seq, boxes_seq)], axis=0)
    return np.concatenate([stat, temporal_features(kpts_seq, boxes_seq)]).astype(np.float32)


def windows_from_track(frames_data, window=WINDOW, stride=None):
    """Slice a track's per-frame (kpts, scores, box, state) into feature windows.

    frames_data: list of dicts with keys kpts, scores, box, state (state may be None).
    Returns (features [M,FEATURE_DIM], majority_states [M]).
    """
    stride = stride or max(window // 3, 1)
    feats, labels = [], []
    for start in range(0, max(len(frames_data) - window + 1, 1), stride):
        chunk = frames_data[start:start + window]
        if len(chunk) < min(window, len(frames_data)):
            continue
        feats.append(window_feature(
            [c["kpts"] for c in chunk],
            [c["scores"] for c in chunk],
            [c["box"] for c in chunk]))
        states = [c.get("state") for c in chunk if c.get("state")]
        labels.append(max(set(states), key=states.count) if states else None)
    return np.array(feats, np.float32), labels
