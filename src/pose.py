"""2D pose estimation on person boxes.

The primary backend is RTMPose through rtmlib and ONNX, which is fast on
the CPU and predicts the 17 COCO keypoints for given person boxes. The
fallback backend is YOLO11-pose run on one crop per person; it is used
automatically if the RTMPose model cannot be downloaded or loaded.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

# The COCO-17 body model, SimCC RTMPose-m, with input size 192 x 256.
RTMPOSE_M_COCO17 = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


class PoseEstimator:
    """Calling this with a frame and boxes [N, 4] returns the keypoints
    [N, 17, 2] in absolute pixels and the scores [N, 17]."""

    def __init__(self, device="cpu", backend=None):
        self.backend = None
        if backend in (None, "rtmpose"):
            try:
                from rtmlib import RTMPose
                self._rtm = RTMPose(
                    onnx_model=RTMPOSE_M_COCO17,
                    model_input_size=(192, 256),
                    backend="onnxruntime",
                    device=device if device in ("cpu", "cuda") else "cpu",
                )
                self.backend = "rtmpose"
            except Exception as e:  # noqa: BLE001 – fall back to YOLO-pose
                print(f"[pose] RTMPose unavailable ({e}); falling back to YOLO-pose")
        if self.backend is None:
            from ultralytics import YOLO
            self._yolo = YOLO("yolo11n-pose.pt")
            self._yolo_device = device
            self.backend = "yolo-pose"

    def __call__(self, image_bgr, boxes):
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        if self.backend == "rtmpose":
            kpts, scores = self._rtm(image_bgr, bboxes=boxes)
            return np.asarray(kpts, np.float32), np.asarray(scores, np.float32)
        return self._yolo_pose(image_bgr, boxes)

    def _yolo_pose(self, image_bgr, boxes, pad=0.15):
        """Estimate the pose of each box with YOLO-pose run on a padded crop."""
        h, w = image_bgr.shape[:2]
        kpts_out = np.zeros((len(boxes), 17, 2), np.float32)
        scores_out = np.zeros((len(boxes), 17), np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            bw, bh = x2 - x1, y2 - y1
            cx1 = int(max(0, x1 - pad * bw)); cy1 = int(max(0, y1 - pad * bh))
            cx2 = int(min(w, x2 + pad * bw)); cy2 = int(min(h, y2 + pad * bh))
            crop = image_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            r = self._yolo.predict(crop, imgsz=256, device=self._yolo_device,
                                   verbose=False, conf=0.1)[0]
            if r.keypoints is None or len(r.keypoints) == 0:
                continue
            j = int(np.argmax(r.boxes.conf.cpu().numpy()))
            xy = r.keypoints.xy[j].cpu().numpy()
            sc = (r.keypoints.conf[j].cpu().numpy()
                  if r.keypoints.conf is not None else np.ones(17, np.float32))
            kpts_out[i] = xy + np.array([cx1, cy1], np.float32)
            scores_out[i] = sc
        return kpts_out, scores_out
