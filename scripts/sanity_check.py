"""An end-to-end sanity check that runs every pipeline stage on real
local data and reports pass or fail."""
import sys
import traceback
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

RESULTS = []


def stage(name):
    """Wrap a stage function so that its outcome is recorded and printed as
    pass or fail."""
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


@stage("Imports + device")
def s1():
    """Import the core libraries and report the torch version and MPS
    availability."""
    import torch
    from config import PROJECT_ROOT  # noqa: F401
    return f"torch {torch.__version__}, mps = {torch.backends.mps.is_available()}"


@stage("Okutama parser")
def s2():
    """Parse the Okutama annotations and require a plausible number of frames
    and tracks."""
    from config import OKUTAMA_DIR
    from data.okutama import parse_annotations, tracks_from_frames
    frames = parse_annotations(OKUTAMA_DIR / "1.1.1.txt")
    tracks = tracks_from_frames(frames)
    assert len(frames) > 1000 and len(tracks) > 50
    return f"{len(frames)} frames, {len(tracks)} tracks"


@stage("Detection on a VisDrone validation image")
def s3():
    """Detect people on one validation image and require at least one
    detection."""
    import cv2
    from config import VISDRONE_PERSON
    from detect import PersonDetector
    imgs = sorted((VISDRONE_PERSON / "images" / "val").glob("*.jpg"))
    img = cv2.imread(str(imgs[0]))
    det = PersonDetector()
    boxes, confs = det(img)
    assert len(boxes) > 0, "No people detected on a crowded VisDrone image"
    return f"{len(boxes)} persons, maximum confidence {confs.max():.2f}"


@stage("Tracking over 30 Okutama frames")
def s4():
    """Track people over 30 video frames and require at least one track."""
    import cv2
    from config import OKUTAMA_DIR
    from track import PersonTracker
    cap = cv2.VideoCapture(str(OKUTAMA_DIR / "1.1.1.mov"))
    trk = PersonTracker()
    ids_seen = set()
    per_frame = []
    for _ in range(30):
        ok, frame = cap.read()
        assert ok, "Video decode failed"
        tracks = trk.update(frame)
        per_frame.append(len(tracks))
        ids_seen.update(t[0] for t in tracks)
    cap.release()
    assert max(per_frame) > 0, "No tracks were detected"
    return f"{max(per_frame)} people/frame peak, {len(ids_seen)} unique ids over 30 frames"


@stage("Pose on real ground-truth boxes")
def s5():
    """Estimate the pose on real ground-truth boxes and require confident
    keypoints."""
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
    assert (scores > 0.3).any(), "All keypoints have low confidence"
    return f"backend = {est.backend}, {len(boxes)} persons, mean keypoint confidence {scores.mean():.2f}"


@stage("Feature extraction")
def s6():
    """Build one feature window from synthetic data and require a finite
    vector."""
    from features import FEATURE_DIM, window_feature
    rng = np.random.default_rng(0)
    T = 15
    kpts = [rng.uniform(0, 100, (17, 2)).astype(np.float32) for _ in range(T)]
    scores = [np.ones(17, np.float32) for _ in range(T)]
    boxes = [np.array([0, 0, 50, 100], np.float32) for _ in range(T)]
    f = window_feature(kpts, scores, boxes)
    assert f.shape == (FEATURE_DIM,) and np.isfinite(f).all()
    return f"dim = {FEATURE_DIM}, finite"


@stage("Classifier mechanics (1 epoch, synthetic)")
def s7():
    """Train the classifier for one epoch on synthetic data and run a
    prediction."""
    from classify import PoseMLP, train_classifier, predict
    from features import FEATURE_DIM
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, FEATURE_DIM)).astype(np.float32)
    y = rng.integers(0, 3, 64)
    model, _ = train_classifier(PoseMLP(3), X, y, X, y, epochs=1,
                                device="cpu", verbose=False)
    assert predict(model, X, device="cpu").shape == (64,)
    return "train+predict ok"


@stage("Triage summary")
def s8():
    """Check the triage summary line and the priority ordering."""
    from triage import priority_list, summarize
    s = summarize(["mobile", "motionless", "mobile", "stationary"])
    assert s.startswith("1 motionless"), s
    pl = priority_list({1: "mobile", 2: "motionless"})
    assert pl[0] == (2, "motionless")
    return s


@stage("Demo render (60 frames)")
def s9():
    """Render 60 demo frames and require a readable output video."""
    import cv2
    from config import OKUTAMA_DIR, VIDEOS_DIR
    from render_demo import render
    out, n = render(OKUTAMA_DIR / "1.1.1.mov",
                    VIDEOS_DIR / "sanity_demo.mp4",
                    max_frames=60, caption="sanity check")
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frames > 0, "The written video has no frames"
    return f"{n} frames rendered -> {out.name} ({frames} readable)"


@stage("RQ3 restoration (Wiener round-trip)")
def s10():
    """Degrade one frame, restore it with the Wiener filter and require a
    PSNR gain."""
    import cv2
    from config import VISDRONE_PERSON
    from eval_restore import psnr
    from restore import degrade, motion_psf, wiener_K_for, wiener_filter
    imgs = sorted((VISDRONE_PERSON / "images" / "val").glob("*.jpg"))
    img = cv2.imread(str(imgs[0]))
    s = 960 / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s)
    psf = motion_psf(15, 45)
    deg = degrade(img, psf, noise_sigma=3, seed=0)
    rest = wiener_filter(deg, psf, wiener_K_for(3))
    p_deg, p_rest = psnr(deg, img), psnr(rest, img)
    assert p_rest > p_deg + 1.0, f"No PSNR gain: {p_deg:.2f} -> {p_rest:.2f}"
    return f"PSNR {p_deg:.1f} -> {p_rest:.1f} dB"


@stage("RQ4 matching on a warped aerial frame")
def s11():
    """Match a warped aerial frame with SIFT and require accurate matches and
    an accurate homography."""
    import cv2
    import numpy as np
    from config import VISDRONE_PERSON
    from eval_match import evaluate_pair, synthetic_pairs
    from match import build_features
    imgs = sorted((VISDRONE_PERSON / "images" / "val").glob("*.jpg"))
    img = cv2.imread(str(imgs[1]))
    s = 960 / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s)
    (i1, i2, H, _), = synthetic_pairs([img], "viewpoint", n_per_image=1,
                                      seed=1)
    r = evaluate_pair(build_features("sift"), i1, i2, H)
    assert r["n_matches"] >= 30 and r["MMA@3"] >= 0.5, r
    assert np.isfinite(r["H_corner_err"]) and r["H_corner_err"] < 5, r
    return (f"{r['n_matches']} matches, MMA@3 {r['MMA@3']:.2f}, "
            f"homography error {r['H_corner_err']:.2f} px")


@stage("RQ4 two-frame registration (Okutama)")
def s12():
    """Register two Okutama frames and require a homography with enough
    inliers."""
    import cv2
    from config import OKUTAMA_DIR
    from match import build_features
    from register import pairwise_homography
    cap = cv2.VideoCapture(str(OKUTAMA_DIR / "1.1.1.mov"))
    frames = []
    for i in range(13):
        ok, f = cap.read()
        assert ok, "Video decode failed"
        if i in (0, 12):
            s = 1280 / f.shape[1]
            frames.append(cv2.resize(f, None, fx=s, fy=s))
    cap.release()
    H, ninl = pairwise_homography(build_features("sift"), frames[1], frames[0])
    assert H is not None and ninl >= 30, f"Registration failed ({ninl} inliers)"
    return f"H estimated with {ninl} inliers over a 12-frame gap"


if __name__ == "__main__":
    for s in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12):
        s()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} stages passed")
    sys.exit(1 if n_fail else 0)
