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

# The triage taxonomy for RQ2.
# The 12 atomic actions of Okutama-Action are grouped into states that
# matter for search and rescue. We verified on the real labels (1.1.1.txt)
# that Okutama has no "Waving" class, so "signaling" only appears if extra
# data such as UAV-Gesture is added; the code discovers the present states
# dynamically. A box can carry several actions at once ("Standing" and
# "Carrying"), so resolve_state() applies the triage precedence:
# motionless > signaling > mobile > stationary.
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

# A lower number means a higher rescue priority.
STATE_PRIORITY = {"motionless": 0, "signaling": 1, "stationary": 2, "mobile": 3}

# BGR colors for OpenCV overlays.
STATE_COLORS = {
    "motionless": (0, 0, 220),     # Red
    "signaling": (0, 165, 255),    # Orange
    "stationary": (220, 130, 0),   # Blue
    "mobile": (0, 180, 0),         # Green
    "unknown": (160, 160, 160),    # Gray
}

# The COCO-17 keypoint skeleton: pairs of keypoint indices to connect.
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # Arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
    (0, 5), (0, 6),                           # Head to shoulders
]

# Model choices. We use the small variants because the local machine is a
# 16 GB M1 Pro.
DET_WEIGHTS = "yolo11n.pt"           # COCO-pretrained; person = class 0
DET_CONF = 0.25
POSE_KP_CONF = 0.3                   # The keypoint score needed for drawing.

# RQ4 feature matching. HPatches is stored here when notebook 06 downloads it.
HPATCHES_DIR = RAW_DIR / "hpatches-sequences-release"
PATCH_SIZE = 32                      # The learned-descriptor patch size in pixels.
DESC_DIM = 128                       # The learned-descriptor output dimension.

# The track-window length for the temporal pose features. Okutama runs
# at 30 frames per second.
WINDOW = 15

for _d in (TABLES_DIR, FIGURES_DIR, VIDEOS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
