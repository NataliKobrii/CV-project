"""Central configuration: paths, triage taxonomy, colors, model choices."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
VIDEOS_DIR = RESULTS_DIR / "videos"

VISDRONE_TRAIN = RAW_DIR / "VisDrone2019-DET-train"
VISDRONE_VAL = RAW_DIR / "VisDrone2019-DET-val"
VISDRONE_PERSON = DATA_DIR / "visdrone_person"  # YOLO-format person-only version
OKUTAMA_DIR = RAW_DIR / "okutama"

# ---------------------------------------------------------------------------
# Triage taxonomy (RQ2).
# Okutama-Action's 12 atomic actions are grouped into SAR-relevant states.
# Verified against real labels (1.1.1.txt): Okutama has NO "Waving" class,
# so "signaling" only materializes if extra data (e.g. UAV-Gesture) is added;
# the code discovers present states dynamically. Boxes can carry several
# actions at once ("Standing" + "Carrying") — resolve_state() applies triage
# precedence: motionless > signaling > mobile > stationary.
# ---------------------------------------------------------------------------
ACTION_TO_STATE = {
    "Lying": "motionless",
    "Sitting": "stationary",
    "Standing": "stationary",
    "Reading": "stationary",
    "Drinking": "stationary",
    "Calling": "stationary",
    "Hand Shaking": "stationary",
    "Handshaking": "stationary",
    "Hugging": "stationary",
    "Waving": "signaling",
    "Walking": "mobile",
    "Running": "mobile",
    "Carrying": "mobile",
    "Pushing/Pulling": "mobile",
}

STATE_PRECEDENCE = ["motionless", "signaling", "mobile", "stationary"]


def resolve_state(actions):
    """Map a list of Okutama action strings to one triage state."""
    states = {ACTION_TO_STATE[a] for a in actions if a in ACTION_TO_STATE}
    for s in STATE_PRECEDENCE:
        if s in states:
            return s
    return "unknown"

# Lower number = higher rescue priority.
STATE_PRIORITY = {"motionless": 0, "signaling": 1, "stationary": 2, "mobile": 3}

# BGR colors for OpenCV overlays.
STATE_COLORS = {
    "motionless": (0, 0, 220),     # red
    "signaling": (0, 165, 255),    # orange
    "stationary": (220, 130, 0),   # blue
    "mobile": (0, 180, 0),         # green
    "unknown": (160, 160, 160),    # gray
}

# COCO-17 keypoint skeleton (pairs of keypoint indices to connect).
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 5), (0, 6),                           # head-shoulders
]
KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# Model choices (small variants — local machine is a 16GB M1 Pro).
DET_WEIGHTS = "yolo11n.pt"           # COCO-pretrained; person = class 0
DET_CONF = 0.25
POSE_KP_CONF = 0.3                   # keypoint visibility threshold for drawing

# Track-window settings for temporal pose features (Okutama is 30 fps).
WINDOW = 15
WINDOW_STRIDE = 5

for _d in (TABLES_DIR, FIGURES_DIR, VIDEOS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
