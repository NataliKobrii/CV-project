"""End-to-end demo renderer: detect -> track -> pose -> state -> triage overlay.

Produces the deliverable demo video: boxes colored by triage state, skeleton
overlays, per-track IDs, and a live priority counter panel.

State classification uses the trained PoseMLP if models/pose_mlp.pt exists;
otherwise a transparent rule-based fallback (body-axis angle + motion) so the
demo runs before any training has happened.
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import (COCO_SKELETON, MODELS_DIR, POSE_KP_CONF, VIDEOS_DIR,  # noqa: E402
                    WINDOW)
from features import static_features, temporal_features, window_feature  # noqa: E402
from pose import PoseEstimator  # noqa: E402
from track import PersonTracker  # noqa: E402
from triage import color_for, summarize  # noqa: E402


class StateClassifier:
    """Trained PoseMLP if available, else rule-based fallback."""

    def __init__(self, model_path=MODELS_DIR / "pose_mlp.pt",
                 classes_path=MODELS_DIR / "pose_mlp_classes.json"):
        self.model, self.classes = None, None
        if Path(model_path).exists() and Path(classes_path).exists():
            import torch
            from classify import PoseMLP
            self.classes = json.loads(Path(classes_path).read_text())
            self.model = PoseMLP(len(self.classes))
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.kind = "pose-mlp"
        else:
            self.kind = "rule-based"

    def __call__(self, hist):
        """hist: deque of {kpts, scores, box} (newest last) -> state str."""
        if len(hist) < 3:
            return "unknown"
        kpts = [h["kpts"] for h in hist]
        scores = [h["scores"] for h in hist]
        boxes = [h["box"] for h in hist]
        if self.model is not None:
            import torch
            feat = window_feature(kpts, scores, boxes)
            with torch.no_grad():
                idx = int(self.model(torch.tensor(feat)[None]).argmax(1))
            return self.classes[idx]
        # rule-based: mean body-axis angle + normalized center speed
        stat = np.mean([static_features(k, s, b)
                        for k, s, b in zip(kpts, scores, boxes)], axis=0)
        temp = temporal_features(kpts, boxes)
        angle_norm, center_speed = float(stat[0]), float(temp[4])
        if angle_norm > 0.6:          # body axis > ~55 deg from vertical
            return "motionless"
        if center_speed > 0.04:
            return "mobile"
        return "stationary"


def draw_person(frame, box, tid, state, kpts, scores, thickness=2):
    color = color_for(state)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"#{tid} {state}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if kpts is not None:
        for a, b in COCO_SKELETON:
            if scores[a] > POSE_KP_CONF and scores[b] > POSE_KP_CONF:
                cv2.line(frame, tuple(map(int, kpts[a])), tuple(map(int, kpts[b])),
                         color, 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(kpts):
            if scores[j] > POSE_KP_CONF:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)


def draw_panel(frame, states, caption, classifier_kind):
    h, w = frame.shape[:2]
    summary = summarize(states.values())
    lines = [f"SAR TRIAGE  |  {summary}",
             f"tracks: {len(states)}   state model: {classifier_kind}"]
    pad, lh = 12, 30
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                for t in lines) + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + lh * len(lines) + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10 + pad, 10 + pad + lh * i + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if caption:
        (tw, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, caption, ((w - tw) // 2, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def render(video_path, out_path=None, weights="yolo11n.pt", imgsz=1280,
           max_frames=None, out_width=1600, caption=None, device=None,
           pose_device="cpu", start_frame=0):
    video_path = Path(video_path)
    out_path = Path(out_path or VIDEOS_DIR / f"demo_{video_path.stem}.mp4")
    tracker = PersonTracker(weights=weights, imgsz=imgsz, device=device)
    pose = PoseEstimator(device=pose_device)
    classifier = StateClassifier()
    print(f"[demo] pose backend={pose.backend}, state model={classifier.kind}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer, size = None, None
    hist = defaultdict(lambda: deque(maxlen=WINDOW))
    states = {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        tracks = tracker.update(frame)
        boxes = np.array([t[1] for t in tracks], np.float32).reshape(-1, 4)
        kpts, scores = pose(frame, boxes)
        seen = set()
        for (tid, box, _conf), kp, sc in zip(tracks, kpts, scores):
            hist[tid].append({"kpts": kp, "scores": sc, "box": box})
            states[tid] = classifier(hist[tid])
            seen.add(tid)
        states = {t: s for t, s in states.items() if t in seen}
        for (tid, box, _conf), kp, sc in zip(tracks, kpts, scores):
            draw_person(frame, box, tid, states[tid], kp, sc)
        draw_panel(frame, states, caption, classifier.kind)
        if writer is None:
            scale = out_width / frame.shape[1]
            size = (out_width, int(frame.shape[0] * scale))
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        writer.write(cv2.resize(frame, size))
        n += 1
        if n % 30 == 0:
            print(f"  frame {n}: {summarize(states.values())}")
    cap.release()
    if writer:
        writer.release()
    print(f"[demo] wrote {n} frames -> {out_path}")
    return out_path, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--caption", default="EECS 4422 - SAR drone pipeline (held-out footage)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    render(args.video, args.out, weights=args.weights, imgsz=args.imgsz,
           max_frames=args.max_frames, caption=args.caption,
           device=args.device, start_frame=args.start_frame)
