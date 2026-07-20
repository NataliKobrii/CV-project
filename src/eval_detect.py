"""Person-detection evaluation: transparent AP@0.5 implementation.

Written from scratch (rather than calling ultralytics.val) so the zero-shot
COCO model and the fine-tuned single-class model are scored by the exact
same code, and so the metric itself is inspectable for the report.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import TABLES_DIR, VISDRONE_PERSON  # noqa: E402


def iou_matrix(a, b):
    """IoU between two box sets, xyxy. a:[N,4] b:[M,4] -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def ap50(all_preds, all_gts, iou_thr=0.5):
    """all_preds: list per image of (boxes[N,4], confs[N]); all_gts: list of boxes[M,4].

    Returns dict with AP50, precision/recall at the score threshold sweep end,
    and counts.
    """
    records = []  # (conf, is_tp)
    n_gt = 0
    for (pboxes, pconfs), gboxes in zip(all_preds, all_gts):
        n_gt += len(gboxes)
        if len(pboxes) == 0:
            continue
        order = np.argsort(-pconfs)
        pboxes, pconfs = pboxes[order], pconfs[order]
        ious = iou_matrix(pboxes, gboxes)
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
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    recall = tp / n_gt
    precision = tp / (tp + fp)
    # 101-point interpolated AP (COCO style)
    ap = float(np.mean([np.max(precision[recall >= t], initial=0.0)
                        for t in np.linspace(0, 1, 101)]))
    return {"AP50": ap, "precision": float(precision[-1]),
            "recall": float(recall[-1]), "n_gt": n_gt, "n_pred": len(records)}


def load_yolo_gt(label_path, w, h):
    """YOLO txt -> xyxy boxes in pixels."""
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
    fields = ["model", "split", "n_images", "imgsz", "AP50",
              "precision", "recall", "n_gt", "n_pred"]
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            wr.writeheader()
        wr.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in row.items()})
    print("appended ->", csv_path)


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
