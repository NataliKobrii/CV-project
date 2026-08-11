"""The RQ1 metric: PCK stratified by person pixel height.

Under PCK@a, a predicted keypoint is correct if its distance to the
ground-truth keypoint is at most a * max(box width, box height). We
normalize by the box size rather than the torso or head length deliberately:
at aerial scales the torso can project to a few pixels, which makes
torso-normalized thresholds degenerate.

The main block validates the whole evaluator on COCO val2017 persons, which
are ground-level images with annotations that can be fetched without
registration. This proves the RQ1 tooling before any aerial data is
needed.
"""
import json
from collections import defaultdict
from pathlib import Path
import sys
import urllib.request

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DIR, TABLES_DIR  # noqa: E402

HEIGHT_BINS = [(0, 25), (25, 50), (50, 100), (100, 10_000)]


def pck(pred_kpts, gt_kpts, gt_vis, boxes, alpha=0.1):
    """Compute the overall PCK and the PCK per height bin.

    pred_kpts and gt_kpts have shape [N, 17, 2]. gt_vis has shape [N, 17],
    where a value above zero means labeled and visible. The boxes hold the
    corner coordinates (x1, y1, x2, y2). Returns a dictionary.
    """
    pred = np.asarray(pred_kpts, np.float32)
    gt = np.asarray(gt_kpts, np.float32)
    vis = np.asarray(gt_vis) > 0
    boxes = np.asarray(boxes, np.float32)
    # We normalize the error threshold by the longer side of the box rather
    # than by the torso or head length. At aerial scale the torso can be only
    # a few pixels, which would make a torso-normalized threshold meaningless.
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    heights = boxes[:, 3] - boxes[:, 1]
    # The Euclidean distance in pixels between each predicted keypoint and
    # its ground-truth position.
    dist = np.linalg.norm(pred - gt, axis=2)  # [N,17]
    # A keypoint is correct within a fraction alpha of the box size.
    correct = dist <= (alpha * sizes)[:, None]

    # The correctness rate over a chosen subset of persons, counting only
    # the keypoints that are labeled and visible.
    def rate(mask_persons):
        m = vis & mask_persons[:, None]
        return float(correct[m].mean()) if m.any() else float("nan")

    out = {"PCK": rate(np.ones(len(pred), bool)),
           "alpha": alpha, "n_persons": int(len(pred)),
           "n_kpts": int(vis.sum())}
    # Reporting the PCK separately per height bin is the core of RQ1: it
    # reveals that accuracy collapses for small (distant) people.
    for lo, hi in HEIGHT_BINS:
        out[f"PCK_h{lo}-{hi if hi < 10_000 else 'inf'}"] = rate(
            (heights >= lo) & (heights < hi))
    return out


# COCO validation part: proves the evaluator on ground-level images

def load_coco_persons(ann_path, min_kpts=8, max_persons=150):
    """Pick well-annotated persons from COCO val2017.

    Returns a list of dictionaries that are later grouped by image, so each
    image is downloaded only once.
    """
    data = json.loads(Path(ann_path).read_text())
    images = {im["id"]: im for im in data["images"]}
    persons = []
    for ann in data["annotations"]:
        if ann.get("iscrowd") or ann["num_keypoints"] < min_kpts:
            continue
        k = np.array(ann["keypoints"], np.float32).reshape(17, 3)
        x, y, w, h = ann["bbox"]
        if w < 40 or h < 60:  # Keep clearly visible people for the sanity check.
            continue
        im = images[ann["image_id"]]
        persons.append({
            "url": im["coco_url"], "file_name": im["file_name"],
            "box": np.array([x, y, x + w, y + h], np.float32),
            "kpts": k[:, :2], "vis": k[:, 2],
        })
        if len(persons) >= max_persons:
            break
    return persons


def validate_on_coco(max_persons=100, alpha=0.1, device="cpu"):
    """Evaluate the pose estimator on ground-level COCO persons and save the
    reference PCK."""
    import cv2
    from pose import PoseEstimator

    ann = RAW_DIR / "annotations" / "person_keypoints_val2017.json"
    persons = load_coco_persons(ann, max_persons=max_persons)
    by_image = defaultdict(list)
    for p in persons:
        by_image[p["file_name"]].append(p)

    img_cache = RAW_DIR / "coco_val_subset"
    img_cache.mkdir(exist_ok=True)
    est = PoseEstimator(device=device)
    preds, gts, viss, boxes = [], [], [], []
    for fname, plist in by_image.items():
        local = img_cache / fname
        if not local.exists():
            urllib.request.urlretrieve(plist[0]["url"], local)
        img = cv2.imread(str(local))
        if img is None:
            continue
        bxs = np.stack([p["box"] for p in plist])
        kp, _sc = est(img, bxs)
        for i, p in enumerate(plist):
            preds.append(kp[i]); gts.append(p["kpts"])
            viss.append(p["vis"]); boxes.append(p["box"])
    m = pck(np.stack(preds), np.stack(gts), np.stack(viss), np.stack(boxes),
            alpha=alpha)
    m["backend"] = est.backend
    m["dataset"] = "COCO-val2017-subset (ground-level sanity check)"
    print(m)
    out = TABLES_DIR / "pck_coco_sanity.json"
    out.write_text(json.dumps(m, indent=2))
    print("Saved", out)
    return m


if __name__ == "__main__":
    validate_on_coco()
