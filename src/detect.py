"""Person detection wrapper around Ultralytics YOLO."""
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent))
from config import DET_CONF, DET_WEIGHTS  # noqa: E402


class PersonDetector:
    """YOLO detector filtered to the person class.

    Works with COCO-pretrained weights (person = class 0 of 80) and with
    our fine-tuned single-class VisDrone-person weights (class 0 of 1).
    """

    def __init__(self, weights=DET_WEIGHTS, conf=DET_CONF, imgsz=1280, device=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        # class 0 is "person" in both COCO and our fine-tuned dataset
        self.classes = [0]

    def __call__(self, image_bgr):
        """Return (boxes [N,4] xyxy float32, confs [N])."""
        r = self.model.predict(
            image_bgr, conf=self.conf, imgsz=self.imgsz, classes=self.classes,
            device=self.device, verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return (
            r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.conf.cpu().numpy().astype(np.float32),
        )
