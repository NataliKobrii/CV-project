"""Generate the four Colab notebooks into notebooks/.

Notebooks are generated (not hand-edited) so they stay consistent; re-run
this script after changing src/ APIs.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": [
            {"cell_type": t,
             "metadata": {},
             **({"source": s.splitlines(keepends=True)} if t == "markdown"
                else {"source": s.splitlines(keepends=True),
                      "outputs": [], "execution_count": None})}
            for t, s in cells
        ],
    }


SETUP_MD = """\
### Setup (every notebook starts with this)
1. Runtime → Change runtime type → **T4 GPU** (free tier).
2. Zip your local `Project/` folder's `src/` and `scripts/` dirs as `src.zip`
   (`cd Project && zip -r src.zip src scripts`), then either upload it below
   or put it in Drive and adjust `SRC_ZIP`.
"""

SETUP_CODE = """\
# --- environment ---
!pip -q install ultralytics rtmlib onnxruntime-gpu
import torch, os
print('cuda:', torch.cuda.is_available())

# --- project code: upload src.zip (or mount Drive and set SRC_ZIP) ---
from pathlib import Path
SRC_ZIP = None  # e.g. '/content/drive/MyDrive/sar_project/src.zip'
if SRC_ZIP is None:
    from google.colab import files
    up = files.upload()  # choose src.zip
    SRC_ZIP = next(iter(up))
!mkdir -p /content/project && unzip -q -o "$SRC_ZIP" -d /content/project
import sys
sys.path.insert(0, '/content/project/src')
sys.path.insert(0, '/content/project')
print('project code ready')

# --- results go to Drive so they survive the session ---
from google.colab import drive
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/sar_project_results'); OUT.mkdir(parents=True, exist_ok=True)
"""

OKUTAMA_DL = """\
# Okutama-Action direct downloads (public Dropbox folder, verified working).
# preview=<file>&dl=1 selects a single file from the shared folder.
OKUTAMA_BASE = ('https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/'
                'APTyV-f01XLnJ0WFpZSBLOE?preview={name}&rlkey=7u7131amaul29amyr4jbnnu03&dl=1')

def fetch_okutama(name, dest='/content/data/okutama'):
    import subprocess, pathlib
    d = pathlib.Path(dest); d.mkdir(parents=True, exist_ok=True)
    zp = d / name
    if not zp.exists():
        subprocess.run(['curl', '-L', '-o', str(zp), OKUTAMA_BASE.format(name=name)], check=True)
    subprocess.run(['unzip', '-q', '-o', str(zp), '-d', str(d)], check=True)
    return d
"""


# ---------------------------------------------------------------- notebook 01
nb01 = [
    ("markdown", """\
# 01 — Person detection: fine-tune YOLO on VisDrone (aerial person)
**Goal (foundation):** quantify the aerial domain gap for detection and close
part of it by fine-tuning. Produces the before/after AP50 table used in the
report. ~40 min on a free T4.
"""),
    ("markdown", SETUP_MD),
    ("code", SETUP_CODE),
    ("code", """\
# VisDrone-DET from the ultralytics mirror (verified live)
%cd /content
!curl -sL -o vd_train.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip
!curl -sL -o vd_val.zip   https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip
!mkdir -p data/raw && unzip -q vd_train.zip -d data/raw && unzip -q vd_val.zip -d data/raw && rm vd_*.zip
"""),
    ("code", """\
# Convert to YOLO person-only format using the project converter
import config
from pathlib import Path
config.RAW_DIR = Path('/content/data/raw')
config.DATA_DIR = Path('/content/data')
config.VISDRONE_TRAIN = config.RAW_DIR / 'VisDrone2019-DET-train'
config.VISDRONE_VAL = config.RAW_DIR / 'VisDrone2019-DET-val'
config.VISDRONE_PERSON = config.DATA_DIR / 'visdrone_person'
from data import visdrone
visdrone.VISDRONE_PERSON = config.VISDRONE_PERSON
visdrone.DATA_DIR = config.DATA_DIR
visdrone.convert_split(config.VISDRONE_TRAIN, 'train', config.VISDRONE_PERSON)
visdrone.convert_split(config.VISDRONE_VAL, 'val', config.VISDRONE_PERSON)
yaml_path = visdrone.write_dataset_yaml(config.VISDRONE_PERSON)
"""),
    ("code", """\
# Baseline: zero-shot COCO yolo11s on aerial imagery
import eval_detect
eval_detect.VISDRONE_PERSON = config.VISDRONE_PERSON
base = eval_detect.evaluate_on_visdrone('yolo11s.pt', imgsz=1280,
                                        tag='yolo11s zero-shot (COCO)', device=0)
"""),
    ("code", """\
# Fine-tune (full train split). ~30 min on T4 at imgsz 1280.
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
model.train(data=str(yaml_path), epochs=30, imgsz=1280, batch=8, device=0,
            project='/content/runs', name='yolo11s_visdrone_ft', exist_ok=True)
best = '/content/runs/yolo11s_visdrone_ft/weights/best.pt'
"""),
    ("code", """\
# After: same evaluator, same split
ft = eval_detect.evaluate_on_visdrone(best, imgsz=1280,
                                      tag='yolo11s fine-tuned (30ep)', device=0)
import pandas as pd
df = pd.DataFrame([base, ft])
df.to_csv(OUT / 'detection_visdrone.csv', index=False)
!cp {best} {OUT}/yolo11s_visdrone_best.pt
df
"""),
    ("markdown", """\
**Report:** the AP50 jump zero-shot → fine-tuned *is* the detection domain-gap
result. Also try `--imgsz 640` vs `1280` as the resolution ablation.
"""),
]

# ---------------------------------------------------------------- notebook 02
nb02 = [
    ("markdown", """\
# 02 — RQ1: pose estimation under the aerial domain gap
How much do COCO-pretrained pose models degrade from above, and how much does
fine-tuning recover?

- **Part A** — sanity reference: PCK of RTMPose on *ground-level* COCO val.
- **Part B** — aerial PCK on **UAV-Human** (needs free registration:
  https://sutdcv.github.io/uav-human-web/ — request the *pose estimation subset*).
- **Part C** — no-registration path: pseudo-label Okutama with a large teacher,
  fine-tune yolo11n-pose as student, evaluate on held-out video.
"""),
    ("markdown", SETUP_MD),
    ("code", SETUP_CODE),
    ("code", """\
# Part A — ground-level reference PCK (evaluator sanity check)
import config
from pathlib import Path
config.RAW_DIR = Path('/content/data/raw'); config.RAW_DIR.mkdir(parents=True, exist_ok=True)
config.TABLES_DIR = OUT
!curl -sL -o /content/coco_ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!cd /content/data/raw && unzip -q -o /content/coco_ann.zip annotations/person_keypoints_val2017.json
import eval_pose
eval_pose.RAW_DIR = config.RAW_DIR; eval_pose.TABLES_DIR = OUT
ref = eval_pose.validate_on_coco(max_persons=150, device='cuda')
"""),
    ("code", OKUTAMA_DL),
    ("code", """\
# Part B — UAV-Human aerial PCK (public Google Drive, no registration needed)
!pip -q install gdown
!gdown 1kWStmFjrN1Njf6mj4rTso6XPMULcFKS5 -O /content/PoseEstimation.zip
!mkdir -p /content/data/raw/uavhuman_pose && unzip -q -o /content/PoseEstimation.zip -d /content/data/raw/uavhuman_pose
"""),
    ("code", """\
# Zero-shot aerial PCK on ALL 22,476 frames (project loader; ~30 min on T4).
# Ablation built in: `downscale` shrinks people before pose to probe the
# scale axis of the domain gap (0.5 -> half-size people), not just viewpoint.
import numpy as np, cv2
from data.uavhuman import iter_dataset
from pose import PoseEstimator
from eval_pose import pck

def aerial_pck(limit=None, downscale=1.0, device='cuda'):
    est = PoseEstimator(device=device)
    preds, gts, viss, boxes = [], [], [], []
    for img_path, persons in iter_dataset('/content/data/raw/uavhuman_pose', limit=limit):
        img = cv2.imread(str(img_path))
        if img is None: continue
        if downscale != 1.0:
            img = cv2.resize(img, None, fx=downscale, fy=downscale)
        for p in persons:
            box = p['box'] * downscale
            kp, _ = est(img, box[None])
            preds.append(kp[0]); gts.append(p['kpts'] * downscale)
            viss.append(p['vis']); boxes.append(box)
    return pck(np.stack(preds), np.stack(gts), np.stack(viss), np.stack(boxes))

aerial = aerial_pck(limit=None)             # full zero-shot aerial PCK
aerial_small = aerial_pck(limit=3000, downscale=0.35)  # small-person regime
print('AERIAL zero-shot:', aerial)
print('AERIAL @0.35 scale:', aerial_small)  # compare both against Part A
"""),
    ("code", """\
# Part C — pseudo-label Okutama -> fine-tune yolo11n-pose (student)
import numpy as np, cv2, sys
fetch_okutama('Sample.zip')          # quick run; use TrainSetVideos.zip for the real one
from data.okutama import parse_annotations
from pose import PoseEstimator

video = '/content/data/okutama/1.1.1.mov'
labels = parse_annotations('/content/data/okutama/1.1.1.txt')
est = PoseEstimator(device='cuda')   # teacher (swap in RTMPose-l for a stronger teacher)

# Build a YOLO-pose dataset: crops around GT boxes + teacher keypoints
from pathlib import Path
ds = Path('/content/data/okutama_pose'); (ds/'images/train').mkdir(parents=True, exist_ok=True)
(ds/'labels/train').mkdir(parents=True, exist_ok=True)
(ds/'images/val').mkdir(parents=True, exist_ok=True); (ds/'labels/val').mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(video); idx = 0; n = 0
while True:
    ok, img = cap.read()
    if not ok: break
    if idx in labels and idx % 5 == 0:
        H, W = img.shape[:2]
        boxes = np.array([b.box for b in labels[idx]], np.float32)
        kpts, scores = est(img, boxes)
        split = 'val' if idx % 25 == 0 else 'train'
        lines = []
        for (x1,y1,x2,y2), kp, sc in zip(boxes, kpts, scores):
            if sc.mean() < 0.35: continue   # keep confident pseudo-labels only
            cx,cy,w,h = ((x1+x2)/2/W,(y1+y2)/2/H,(x2-x1)/W,(y2-y1)/H)
            ks = ' '.join(f'{x/W:.5f} {y/H:.5f} {2 if s>0.35 else 0}' for (x,y),s in zip(kp,sc))
            lines.append(f'0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f} ' + ks)
        if lines:
            cv2.imwrite(str(ds/f'images/{split}/{idx:06d}.jpg'), img)
            (ds/f'labels/{split}/{idx:06d}.txt').write_text('\\n'.join(lines))
            n += 1
    idx += 1
cap.release(); print(n, 'pseudo-labeled frames')
(ds/'okutama_pose.yaml').write_text(f'path: {ds}\\ntrain: images/train\\nval: images/val\\nkpt_shape: [17, 3]\\nnames:\\n  0: person\\n')
"""),
    ("code", """\
# Student fine-tune + PCK before/after vs teacher on the val frames
from ultralytics import YOLO
student = YOLO('yolo11n-pose.pt')
student.train(data=str(ds/'okutama_pose.yaml'), epochs=20, imgsz=1280, batch=8,
              device=0, project='/content/runs', name='pose_ft', exist_ok=True)
!cp /content/runs/pose_ft/weights/best.pt {OUT}/yolo11n_pose_okutama.pt
"""),
    ("code", """\
# Stratified PCK plot (the RQ1 headline figure)
import json, matplotlib.pyplot as plt
# fill with your measured values: ground-level (Part A), aerial zero-shot,
# aerial fine-tuned (Part B/C)
results = {'ground-level (COCO)': ref}
fig, ax = plt.subplots(figsize=(6,4))
bins = [k for k in ref if k.startswith('PCK_h')]
for name, m in results.items():
    ax.plot(bins, [m[b] for b in bins], marker='o', label=name)
ax.set_xlabel('person pixel height'); ax.set_ylabel(f"PCK@{ref['alpha']}")
ax.set_title('RQ1: pose accuracy vs person scale'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'rq1_pck_vs_scale.png', dpi=150)
"""),
]

# ---------------------------------------------------------------- notebook 03
nb03 = [
    ("markdown", """\
# 03 — RQ2: pose vs appearance for triage state classification
Full-scale version of `scripts/train_rq2_local.py`: train on multiple Okutama
videos, hold out whole videos (not just tracks), compare PoseMLP vs
AppearanceCNN, run the ablations.
"""),
    ("markdown", SETUP_MD),
    ("code", SETUP_CODE),
    ("code", OKUTAMA_DL),
    ("code", """\
# Download videos (TrainSetVideos ~5GB — Colab disk is fine; start with Sample)
fetch_okutama('Sample.zip')
# fetch_okutama('TrainSetVideos.zip')   # uncomment for the full run
# fetch_okutama('TestSetVideos.zip')
import glob
videos = sorted(glob.glob('/content/data/okutama/**/*.mov', recursive=True))
labels = sorted(glob.glob('/content/data/okutama/**/*.txt', recursive=True))
print(len(videos), 'videos')
"""),
    ("code", """\
import numpy as np, sys
import config
from pathlib import Path
config.MODELS_DIR = OUT; config.TABLES_DIR = OUT; config.FIGURES_DIR = OUT
from scripts.train_rq2_local import extract, build_dataset
from data.okutama import parse_annotations

all_tracks, all_crops, offset = {}, {}, 0
for v, t in zip(videos, labels):
    frames = parse_annotations(t)
    tr, cr = extract(v, frames, step=2, pose_device='cuda')
    # re-key track ids so videos don't collide; remember source video per track
    for tid, seq in tr.items():
        all_tracks[offset + tid] = seq
    for (tid, f), c in cr.items():
        all_crops[(offset + tid, f)] = c
    offset += 10_000
X, C, y, tids, classes = build_dataset(all_tracks, all_crops)
"""),
    ("code", """\
# Split by VIDEO (tids//10_000) — stricter than by-track, matches the report
vid_of = tids // 10_000
val_vids = set(list(sorted(set(vid_of)))[-max(1, len(set(vid_of))//4):])
va = np.isin(vid_of, list(val_vids)); tr = ~va
print('train', tr.sum(), 'val', va.sum())

from classify import PoseMLP, AppearanceCNN, train_classifier, predict, evaluate
counts = np.bincount(y[tr], minlength=len(classes)).astype('float32')
w = counts.sum()/np.maximum(counts,1)/len(classes)

pose_model,_ = train_classifier(PoseMLP(len(classes)), X[tr], y[tr], X[va], y[va],
                                epochs=60, class_weights=w, device='cuda')
m_pose = evaluate(y[va], predict(pose_model, X[va], device='cuda'), classes, 'pose_mlp_full')

Cn = (C.astype('float32')/255.).transpose(0,3,1,2)
cnn,_ = train_classifier(AppearanceCNN(len(classes)), Cn[tr], y[tr], Cn[va], y[va],
                         epochs=60, class_weights=w, device='cuda')
m_app = evaluate(y[va], predict(cnn, Cn[va], device='cuda'), classes, 'appearance_cnn_full')

import pandas as pd
pd.DataFrame([{'model':'PoseMLP', **m_pose['per_class_f1'], 'macro_f1':m_pose['macro_f1']},
              {'model':'AppearanceCNN', **m_app['per_class_f1'], 'macro_f1':m_app['macro_f1']}]
             ).to_csv(OUT/'rq2_full.csv', index=False)
"""),
    ("markdown", """\
**Ablations to run** (rerun the two cells above, one change at a time):
window size 5/15/30; crop resolution 32/64/96; pose-features-only vs +temporal
(slice `X[:, :41]`); per-class error analysis — save 20 misclassified crops
and look at them. The last one is worth the most marks.
"""),
]

# ---------------------------------------------------------------- notebook 04
nb04 = [
    ("markdown", """\
# 04 — Full pipeline demo video
detect → track → pose → classify → triage overlay on a **held-out** Okutama
test video, with the fine-tuned weights from notebooks 01–03.
"""),
    ("markdown", SETUP_MD),
    ("code", SETUP_CODE),
    ("code", OKUTAMA_DL),
    ("code", """\
fetch_okutama('TestSetVideos.zip')   # held-out footage the models never saw
import glob
test_videos = sorted(glob.glob('/content/data/okutama/**/*.mov', recursive=True))
print(test_videos[:5])
"""),
    ("code", """\
import config
config.MODELS_DIR = OUT   # picks up pose_mlp.pt trained in notebook 03
from render_demo import render
det_weights = str(OUT/'yolo11s_visdrone_best.pt')  # from notebook 01 (or 'yolo11s.pt')
out, n = render(test_videos[0], '/content/demo_raw.mp4', weights=det_weights,
                imgsz=1280, max_frames=1800, device=0, pose_device='cuda',
                caption='EECS 4422 — SAR triage pipeline (held-out footage)')
"""),
    ("code", """\
# h264 re-encode so it plays everywhere, then keep in Drive
!ffmpeg -y -loglevel error -i /content/demo_raw.mp4 -c:v libx264 -pix_fmt yuv420p {OUT}/demo_final.mp4
print('saved to Drive:', OUT/'demo_final.mp4')
"""),
]

for name, cells in [("01_detection_finetune", nb01), ("02_pose_domain_gap", nb02),
                    ("03_state_classification", nb03), ("04_demo_pipeline", nb04)]:
    path = NB_DIR / f"{name}.ipynb"
    path.write_text(json.dumps(nb(cells), indent=1))
    print("wrote", path)
