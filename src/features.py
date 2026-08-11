"""Pose keypoints turned into feature vectors for state classification (RQ2).

Static features are computed per frame from the 17 COCO keypoints and the
person box. Temporal features are computed over a short track window. They
capture what separates the triage states: the overall body-axis orientation
(lying against upright), limb elevation (waving), and motion energy (mobile
against stationary).
"""
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import WINDOW  # noqa: E402

# 34 normalized keypoints plus 7 geometric features, averaged over the
# window, plus 6 temporal features.
FEATURE_DIM = 34 + 7 + 6

# Indices of the COCO keypoints we rely on: the shoulders, wrists, hips
# and nose. Naming them avoids magic numbers later in the file.
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, NOSE = 5, 6, 9, 10, 11, 12, 0


def _mid(kpts, a, b):
    """Return the midpoint of two keypoints."""
    return (kpts[a] + kpts[b]) / 2.0


def normalize_kpts(kpts, box):
    """Center the keypoints on the mid-hip and scale by the torso length.
    The box diagonal is the fallback scale."""
    # We center on the mid-hip and divide by the torso length so that the
    # keypoints become invariant to the person's position and size in the
    # image. A person then looks the same to the classifier whether they are
    # near the camera or far away.
    mid_hip = _mid(kpts, L_HIP, R_HIP)
    mid_sh = _mid(kpts, L_SH, R_SH)
    torso = float(np.linalg.norm(mid_sh - mid_hip))
    # When the torso collapses to almost nothing (a near-top-down view, or
    # missing hip and shoulder keypoints) we fall back to the box diagonal.
    if torso < 1e-3:
        x1, y1, x2, y2 = box
        torso = max(float(np.hypot(x2 - x1, y2 - y1)) / 3.0, 1e-3)
    return ((kpts - mid_hip) / torso).reshape(-1)  # 34


def static_features(kpts, scores, box):
    """Return the 7 geometric features for one frame."""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    mid_hip = _mid(kpts, L_HIP, R_HIP)
    mid_sh = _mid(kpts, L_SH, R_SH)
    axis = mid_sh - mid_hip  # Points from the hips toward the shoulders.
    # The angle of the body axis from vertical: 0 is upright, 90 is lying.
    angle = np.degrees(np.arctan2(abs(float(axis[0])), abs(float(axis[1])) + 1e-6))
    # A lying person tends to have a wide, short box, so the aspect ratio
    # complements the body-axis angle.
    aspect = h / w
    # Note that image y grows downward, so "above" means a smaller y.
    # These four flags detect raised hands (a wrist above the shoulders or
    # above the nose), which is the signature of a waving or signaling person.
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
    """Return the 6 motion features over a window. Distances are normalized
    by the box size."""
    boxes = np.asarray(boxes_seq, np.float32)
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    # All motion distances are divided by this scale so that the features
    # measure motion relative to the person's own size, not in raw pixels.
    scale = float(np.mean(sizes)) + 1e-3
    kseq = np.asarray(kpts_seq, np.float32)  # [T,17,2]
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                        (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    # Wrist positions relative to the box center, which removes global motion.
    # Subtracting the box center removes the person's overall movement, so
    # what remains is the motion of the wrists relative to the body. This
    # separates arm gestures from walking.
    rel = kseq[:, [L_WR, R_WR], :] - centers[:, None, :]
    # The vertical spread of the wrists over the window is high when the
    # arms move up and down, as in waving.
    wrist_y_std = np.std(rel[:, :, 1], axis=0) / scale            # 2
    wrist_speed = (np.mean(np.abs(np.diff(rel, axis=0)), axis=(0, 2)) / scale
                   if len(kseq) > 1 else np.zeros(2))             # 2
    # The speed of the box center is the main cue that separates a moving
    # person from a stationary one.
    center_speed = (float(np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))) / scale
                    if len(centers) > 1 else 0.0)                 # 1
    size_change = float(np.std(sizes)) / scale                    # 1
    return np.concatenate([wrist_y_std, wrist_speed,
                           [center_speed, size_change]]).astype(np.float32)


def window_feature(kpts_seq, scores_seq, boxes_seq):
    """Return one FEATURE_DIM vector for a track window, given lists of
    per-frame arrays."""
    stat = np.mean(
        [np.concatenate([normalize_kpts(k, b), static_features(k, s, b)])
         for k, s, b in zip(kpts_seq, scores_seq, boxes_seq)], axis=0)
    return np.concatenate([stat, temporal_features(kpts_seq, boxes_seq)]).astype(np.float32)


def windows_from_track(frames_data, window=WINDOW, stride=None):
    """Slice the per-frame data of one track into feature windows.

    frames_data is a list of dictionaries with the keys kpts, scores, box
    and state, where state may be None. Returns the features
    [M, FEATURE_DIM] and the majority state of every window.
    """
    # Overlapping windows (the stride is smaller than the window) give more
    # training examples per track and smooth the per-frame noise.
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
        # The window label is the most frequent state among its frames, which
        # is robust to a few mislabeled or missing frames.
        states = [c.get("state") for c in chunk if c.get("state")]
        labels.append(max(set(states), key=states.count) if states else None)
    return np.array(feats, np.float32), labels
