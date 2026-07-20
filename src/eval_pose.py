"""RQ1 metric: PCK stratified by person pixel height.

PCK@a: a predicted keypoint is correct if its distance to the ground-truth
keypoint is <= a * max(box_w, box_h). Box-size normalization (rather than
torso/head length) is deliberate: at aerial scales the torso can project to
a couple of pixels, making torso-normalized thresholds degenerate.

The __main__ block validates the whole evaluator on COCO val2017 persons
(ground-level images, annotations fetched without registration) so the RQ1
tooling is proven before UAV-Human access is granted.
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
    """Overall + per-height-bin PCK.

    pred_kpts/gt_kpts: [N,17,2]; gt_vis: [N,17] (>0 = labeled&visible);
    boxes: [N,4] xyxy. Returns dict.
    """
    pred = np.asarray(pred_kpts, np.float32)
    gt = np.asarray(gt_kpts, np.float32)
    vis = np.asarray(gt_vis) > 0
    boxes = np.asarray(boxes, np.float32)
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    heights = boxes[:, 3] - boxes[:, 1]
    dist = np.linalg.norm(pred - gt, axis=2)  # [N,17]
    correct = dist <= (alpha * sizes)[:, None]

    def rate(mask_persons):
        m = vis & mask_persons[:, None]
        return float(correct[m].mean()) if m.any() else float("nan")

    out = {"PCK": rate(np.ones(len(pred), bool)),
           "alpha": alpha, "n_persons": int(len(pred)),
           "n_kpts": int(vis.sum())}
    for lo, hi in HEIGHT_BINS:
        out[f"PCK_h{lo}-{hi if hi < 10_000 else 'inf'}"] = rate(
            (heights >= lo) & (heights < hi))
    return out


# --------------------------- COCO validation ---------------------------

def load_coco_persons(ann_path, min_kpts=8, max_persons=150):
    """Pick well-annotated persons from COCO val2017.

    Returns list of dicts {image_url, file_name, box, kpts, vis} grouped later
    by image so each image is downloaded once.
    """
    data = json.loads(Path(ann_path).read_text())
    images = {im["id"]: im for im in data["images"]}
    persons = []
    for ann in data["annotations"]:
        if ann.get("iscrowd") or ann["num_keypoints"] < min_kpts:
            continue
        k = np.array(ann["keypoints"], np.float32).reshape(17, 3)
        x, y, w, h = ann["bbox"]
        if w < 40 or h < 60:  # keep clearly visible people for the sanity check
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
    print("saved", out)
    return m


if __name__ == "__main__":
    validate_on_coco()
