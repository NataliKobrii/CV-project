"""RQ1: zero-shot aerial PCK of COCO-pretrained RTMPose on UAV-Human.

The ground-level reference (same model, same evaluator, same alpha) is
results/tables/pck_coco_sanity.json — the difference IS the domain gap.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import RAW_DIR, TABLES_DIR  # noqa: E402
from data.uavhuman import iter_dataset  # noqa: E402
from eval_pose import pck  # noqa: E402
from pose import PoseEstimator  # noqa: E402


def main(limit_frames=800, device="cpu"):
    root = RAW_DIR / "uavhuman_pose"
    est = PoseEstimator(device=device)
    preds, gts, viss, boxes = [], [], [], []
    n_img = 0
    for img_path, persons in iter_dataset(root, limit=limit_frames):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        bxs = np.stack([p["box"] for p in persons])
        kp, _sc = est(img, bxs)
        for p, k in zip(persons, kp):
            preds.append(k); gts.append(p["kpts"])
            viss.append(p["vis"]); boxes.append(p["box"])
        n_img += 1
        if n_img % 100 == 0:
            print(f"  {n_img} frames, {len(preds)} persons")
    m = pck(np.stack(preds), np.stack(gts), np.stack(viss), np.stack(boxes))
    m["backend"] = est.backend
    m["dataset"] = f"UAV-Human pose subset (aerial), {n_img} frames"
    print(m)
    out = TABLES_DIR / "pck_uavhuman_zeroshot.json"
    out.write_text(json.dumps(m, indent=2))
    print("saved", out)

    ref_path = TABLES_DIR / "pck_coco_sanity.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        print(f"\nRQ1 domain gap (PCK@{m['alpha']}): "
              f"ground-level {ref['PCK']:.3f} -> aerial {m['PCK']:.3f} "
              f"({(m['PCK'] - ref['PCK']) * 100:+.1f} pts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-frames", type=int, default=800)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    main(limit_frames=args.limit_frames, device=args.device)
