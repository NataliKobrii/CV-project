"""Person-detection evaluation with a transparent AP@0.5 implementation.

The metric is implemented directly instead of calling ultralytics.val, so
the zero-shot COCO model and the fine-tuned single-class model are scored
by exactly the same code, and the metric itself is easy to inspect.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import TABLES_DIR, VISDRONE_PERSON  # noqa: E402


def iou_matrix(a, b):
    """Return the IoU between two box sets given as corner coordinates
    (x1, y1, x2, y2), with shape [N, M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    # Broadcasting compares every box in a against every box in b at once.
    # The intersection rectangle runs from the larger top-left corner to the
    # smaller bottom-right corner.
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def ap50(all_preds, all_gts, iou_thr=0.5):
    """Compute AP@0.5.

    all_preds holds one (boxes, confidences) pair per image and all_gts
    holds the ground-truth boxes per image. Returns a dictionary with AP50,
    the precision and recall at the end of the score sweep, and the counts.
    """
    records = []  # Pairs of (confidence, whether the prediction was correct).
    n_gt = 0
    for (pboxes, pconfs), gboxes in zip(all_preds, all_gts):
        n_gt += len(gboxes)
        if len(pboxes) == 0:
            continue
        # Within an image we consider the most confident predictions first,
        # so that a high-confidence box claims a ground-truth match before a
        # weaker overlapping box can.
        order = np.argsort(-pconfs)
        pboxes, pconfs = pboxes[order], pconfs[order]
        ious = iou_matrix(pboxes, gboxes)
        # Each ground-truth box may be matched at most once.
        taken = np.zeros(len(gboxes), bool)
        for i in range(len(pboxes)):
            j = int(np.argmax(ious[i])) if len(gboxes) else -1
            if j >= 0 and ious[i, j] >= iou_thr and not taken[j]:
                taken[j] = True
                records.append((float(pconfs[i]), 1))
            else:
                records.append((float(pconfs[i]), 0))
    if not records or n_gt == 0:
        return {"AP50": 0.0, "precision": 0.0, "recall": 0.0,
                "n_gt": n_gt, "n_pred": len(records)}
    # Predictions are ranked by confidence, as the definition of average
    # precision requires.
    records.sort(key=lambda r: -r[0])
    # Sweeping the confidence threshold from high to low, the cumulative
    # counts of true and false positives trace the precision-recall curve.
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    recall = tp / n_gt
    precision = tp / (tp + fp)
    # The 101-point interpolated AP, as in the COCO protocol.
    # Average precision is the mean of the precision values interpolated at
    # 101 evenly spaced recall levels, which is the standard COCO definition.
    ap = float(np.mean([np.max(precision[recall >= t], initial=0.0)
                        for t in np.linspace(0, 1, 101)]))
    return {"AP50": ap, "precision": float(precision[-1]),
            "recall": float(recall[-1]), "n_gt": n_gt, "n_pred": len(records)}


def load_yolo_gt(label_path, w, h):
    """Read a YOLO label file and return boxes as corner coordinates
    (x1, y1, x2, y2) in pixels."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, np.float32).reshape(-1, 4)


def evaluate_on_visdrone(weights, split="val", imgsz=1280, conf=0.01,
                         max_images=None, tag=None, device=None):
    """Run the detector over a VisDrone split and compute AP@0.5 against the
    ground truth."""
    import cv2
    from detect import PersonDetector

    det = PersonDetector(weights=weights, conf=conf, imgsz=imgsz, device=device)
    img_dir = VISDRONE_PERSON / "images" / split
    lbl_dir = VISDRONE_PERSON / "labels" / split
    images = sorted(img_dir.glob("*.jpg"))
    if max_images:
        images = images[:max_images]
    preds, gts = [], []
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        preds.append(det(img))
        gts.append(load_yolo_gt(lbl_dir / (img_path.stem + ".txt"), w, h))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} images")
    m = ap50(preds, gts)
    m.update({"model": tag or str(weights), "split": split,
              "imgsz": imgsz, "n_images": len(images)})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    return m


def append_csv(row, csv_path=TABLES_DIR / "detection_baseline.csv"):
    """Append one result row to the detection table, writing the header when
    the file is new."""
    fields = ["model", "split", "n_images", "imgsz", "AP50",
              "precision", "recall", "n_gt", "n_pred"]
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            wr.writeheader()
        wr.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in row.items()})
    print("Appended ->", csv_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    row = evaluate_on_visdrone(args.weights, args.split, args.imgsz,
                               max_images=args.max_images, tag=args.tag,
                               device=args.device)
    append_csv(row)
