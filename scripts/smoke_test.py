"""End-to-end smoke test: every pipeline stage on real local data, pass/fail."""
import sys
import traceback
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

RESULTS = []


def stage(name):
    def deco(fn):
        def run():
            try:
                out = fn()
                RESULTS.append((name, True, out))
                print(f"PASS  {name}: {out}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, str(e)))
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
        return run
    return deco


@stage("imports + device")
def s1():
    import torch
    from config import PROJECT_ROOT  # noqa: F401
    return f"torch {torch.__version__}, mps={torch.backends.mps.is_available()}"


@stage("okutama parser")
def s2():
    from config import OKUTAMA_DIR
    from data.okutama import parse_annotations, tracks_from_frames
    frames = parse_annotations(OKUTAMA_DIR / "1.1.1.txt")
    tracks = tracks_from_frames(frames)
    assert len(frames) > 1000 and len(tracks) > 50
    return f"{len(frames)} frames, {len(tracks)} tracks"


@stage("detection on VisDrone val image")
def s3():
    import cv2
    from config import VISDRONE_PERSON
    from detect import PersonDetector
    imgs = sorted((VISDRONE_PERSON / "images" / "val").glob("*.jpg"))
    img = cv2.imread(str(imgs[0]))
    det = PersonDetector()
    boxes, confs = det(img)
    assert len(boxes) > 0, "no people detected on a crowded VisDrone image"
    return f"{len(boxes)} persons, max conf {confs.max():.2f}"


@stage("tracking over 30 Okutama frames")
def s4():
    import cv2
    from config import OKUTAMA_DIR
    from track import PersonTracker
    cap = cv2.VideoCapture(str(OKUTAMA_DIR / "1.1.1.mov"))
    trk = PersonTracker()
    ids_seen = set()
    per_frame = []
    for _ in range(30):
        ok, frame = cap.read()
        assert ok, "video decode failed"
        tracks = trk.update(frame)
        per_frame.append(len(tracks))
        ids_seen.update(t[0] for t in tracks)
    cap.release()
    assert max(per_frame) > 0, "no tracks at all"
    return f"{max(per_frame)} people/frame peak, {len(ids_seen)} unique ids over 30 frames"


@stage("pose on real GT boxes")
def s5():
    import cv2
    from config import OKUTAMA_DIR
    from data.okutama import parse_annotations
    from pose import PoseEstimator
    frames = parse_annotations(OKUTAMA_DIR / "1.1.1.txt")
    f0 = min(frames)
    cap = cv2.VideoCapture(str(OKUTAMA_DIR / "1.1.1.mov"))
    ok, img = cap.read()
    cap.release()
    assert ok
    boxes = np.array([b.box for b in frames[f0]], np.float32)
    est = PoseEstimator()
    kpts, scores = est(img, boxes)
    assert kpts.shape == (len(boxes), 17, 2)
    assert (scores > 0.3).any(), "all keypoints low-confidence"
    return f"backend={est.backend}, {len(boxes)} persons, mean kp conf {scores.mean():.2f}"


@stage("feature extraction")
def s6():
    from features import FEATURE_DIM, window_feature
    rng = np.random.default_rng(0)
    T = 15
    kpts = [rng.uniform(0, 100, (17, 2)).astype(np.float32) for _ in range(T)]
    scores = [np.ones(17, np.float32) for _ in range(T)]
    boxes = [np.array([0, 0, 50, 100], np.float32) for _ in range(T)]
    f = window_feature(kpts, scores, boxes)
    assert f.shape == (FEATURE_DIM,) and np.isfinite(f).all()
    return f"dim={FEATURE_DIM}, finite"


@stage("classifier mechanics (1 epoch, synthetic)")
def s7():
    from classify import PoseMLP, train_classifier, predict
    from features import FEATURE_DIM
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, FEATURE_DIM)).astype(np.float32)
    y = rng.integers(0, 3, 64)
    model, _ = train_classifier(PoseMLP(3), X, y, X, y, epochs=1,
                                device="cpu", verbose=False)
    assert predict(model, X, device="cpu").shape == (64,)
    return "train+predict ok"


@stage("triage summary")
def s8():
    from triage import priority_list, summarize
    s = summarize(["mobile", "motionless", "mobile", "stationary"])
    assert s.startswith("1 motionless"), s
    pl = priority_list({1: "mobile", 2: "motionless"})
    assert pl[0] == (2, "motionless")
    return s


@stage("demo render (60 frames)")
def s9():
    import cv2
    from config import OKUTAMA_DIR, VIDEOS_DIR
    from render_demo import render
    out, n = render(OKUTAMA_DIR / "1.1.1.mov",
                    VIDEOS_DIR / "smoke_demo.mp4",
                    max_frames=60, caption="smoke test")
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frames > 0, "written video has no frames"
    return f"{n} frames rendered -> {out.name} ({frames} readable)"


if __name__ == "__main__":
    for s in (s1, s2, s3, s4, s5, s6, s7, s8, s9):
        s()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} stages passed")
    sys.exit(1 if n_fail else 0)
