"""Person tracking: ByteTrack association on top of YOLO detections."""
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent))
from config import DET_CONF, DET_WEIGHTS  # noqa: E402


class PersonTracker:
    """Frame-by-frame tracker; call .update(frame) in video order."""

    def __init__(self, weights=DET_WEIGHTS, conf=DET_CONF, imgsz=1280,
                 tracker="bytetrack.yaml", device=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device

    def update(self, image_bgr):
        """Return list of (track_id:int, box xyxy np[4], conf:float)."""
        r = self.model.track(
            image_bgr, persist=True, conf=self.conf, imgsz=self.imgsz,
            classes=[0], tracker=self.tracker, device=self.device, verbose=False,
        )[0]
        out = []
        if r.boxes is None or r.boxes.id is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(np.float32)
        for tid, box, c in zip(ids, boxes, confs):
            out.append((int(tid), box, float(c)))
        return out

    def reset(self):
        """Start a fresh video (clears track state)."""
        if getattr(self.model.predictor, "trackers", None):
            self.model.predictor.trackers[0].reset()
