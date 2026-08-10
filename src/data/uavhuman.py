"""Loader for the UAV-Human pose-estimation subset (Label Studio JSON exports).

Each annotations/*.json holds one frame; keypoints are stored as percentages
of the original image size, one entry per keypoint, grouped per person by
`from_name` ("kp-1", "kp-2", ...). Keypoint names are camelCase; we map them
to COCO-17 order. Person boxes are not annotated — we derive the standard
tight box around labeled keypoints with padding (usual top-down eval protocol).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME_TO_COCO = {
    "nose": 0, "leftEye": 1, "rightEye": 2, "leftEar": 3, "rightEar": 4,
    "leftShoulder": 5, "rightShoulder": 6, "leftElbow": 7, "rightElbow": 8,
    "leftWrist": 9, "rightWrist": 10, "leftHip": 11, "rightHip": 12,
    "leftKnee": 13, "rightKnee": 14, "leftAnkle": 15, "rightAnkle": 16,
}


def parse_annotation(json_path, pad=0.15, min_kpts=6):
    """-> list of dicts {kpts[17,2] px, vis[17], box[4] xyxy} for one frame."""
    d = json.loads(Path(json_path).read_text())
    completions = d.get("completions") or d.get("annotations") or []
    if not completions:
        return []
    persons = defaultdict(lambda: (np.zeros((17, 2), np.float32),
                                   np.zeros(17, np.float32)))
    W = H = None
    for r in completions[0].get("result", []):
        if r.get("type") != "keypointlabels":
            continue
        v = r.get("value", {})
        labels = v.get("keypointlabels", [])
        if not labels or labels[0] not in NAME_TO_COCO:
            continue

        # Some keypoint annotation records may not contain coordinates
        if "x" not in v or "y" not in v:
            print("Skipping keypoint with missing coordinates:", r)
            continue

        W = r.get("original_width")
        H = r.get("original_height")

        if W is None or H is None:
            print("Skipping keypoint with missing image dimensions:", r)
            continue

        k = NAME_TO_COCO[labels[0]]
        kpts, vis = persons[r.get("from_name", "kp-1")]

        kpts[k] = (
            v["x"] / 100.0 * W,
            v["y"] / 100.0 * H
        )
        vis[k] = 1.0
    out = []
    for kpts, vis in persons.values():
        if vis.sum() < min_kpts:
            continue
        pts = kpts[vis > 0]
        x1, y1 = pts.min(0)
        x2, y2 = pts.max(0)
        bw, bh = max(x2 - x1, 4.0), max(y2 - y1, 4.0)
        box = np.array([
            max(x1 - pad * bw, 0), max(y1 - pad * bh, 0),
            min(x2 + pad * bw, (W or 1e9)), min(y2 + pad * bh, (H or 1e9)),
        ], np.float32)
        out.append({"kpts": kpts, "vis": vis, "box": box})
    return out


def iter_dataset(root, limit=None, seed=0):
    """Yield (frame_path, persons) over the subset; shuffled, optional cap."""
    root = Path(root)
    anns = sorted((root / "annotations").glob("*.json"))
    rng = np.random.default_rng(seed)
    rng.shuffle(anns)
    n = 0
    for ann in anns:
        img = root / "frames" / (ann.stem + ".jpg")
        if not img.exists():
            continue
        persons = parse_annotation(ann)
        if not persons:
            continue
        yield img, persons
        n += 1
        if limit and n >= limit:
            return
