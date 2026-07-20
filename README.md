# Is Anyone Down There Moving?
**Pose-based human state recognition for search-and-rescue drones**
EECS 4422 Computer Vision · Summer 2026 · York University

A drone sweeping a disaster area should not just *find* people — it should tell
a person lying motionless from one walking around, so rescuers get a priority
list instead of a dot map. Pipeline: **detect → track → pose → state → triage**.

Two research questions:
- **RQ1 (domain gap):** how much do COCO-pretrained 2D pose estimators degrade
  under aerial viewpoints/scales, and how much does fine-tuning recover?
- **RQ2 (does pose matter?):** does an explicit pose representation beat an
  appearance-only baseline for classifying *motionless / stationary / mobile*?

## Repo layout

```
src/                 pipeline modules (importable, used by scripts AND notebooks)
  config.py          paths, triage taxonomy (action→state map), colors
  detect.py          YOLO person detection wrapper
  track.py           ByteTrack tracking wrapper
  pose.py            RTMPose (rtmlib/ONNX) top-down pose; YOLO-pose fallback
  features.py        pose → feature vectors (geometry + temporal motion)
  classify.py        PoseMLP vs AppearanceCNN + shared train/eval loop
  triage.py          state → priority, colors, operator summary line
  eval_detect.py     from-scratch AP@0.5 evaluator (person detection)
  eval_pose.py       PCK@α stratified by person pixel height (RQ1 metric)
  render_demo.py     demo video renderer (overlay + live triage counter)
  data/visdrone.py   VisDrone → YOLO person-only converter
  data/okutama.py    Okutama-Action VATIC label parser (+ state resolution)
scripts/
  smoke_test.py      end-to-end pipeline check on real local data (9 stages)
  train_rq2_local.py smoke-scale RQ2 experiment on the Okutama sample video
  detection_experiments.py  zero-shot baseline → mini fine-tune → re-eval
  make_notebooks.py  regenerates the notebooks below
notebooks/           Colab notebooks for the FULL experiments (free T4 GPU)
  01_detection_finetune.ipynb    VisDrone person fine-tune + before/after AP50
  02_pose_domain_gap.ipynb       RQ1: PCK vs scale; UAV-Human + pseudo-labels
  03_state_classification.ipynb  RQ2: pose vs appearance on full Okutama
  04_demo_pipeline.ipynb         final demo video on held-out test footage
data/                datasets (gitignore-worthy; see below)
models/              trained weights (pose_mlp.pt, fine-tuned YOLO runs)
results/             tables/ figures/ videos/ — everything the report cites
report/              proposal.pdf, pitch, report skeleton
```

## Setup (local)

```bash
cd Project
/opt/homebrew/bin/python3.11 -m venv .venv        # any Python 3.10–3.12
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py             # should print 9/9 PASS
```

## Data

| Dataset | What for | Access |
|---|---|---|
| **VisDrone-DET** | person detection (foundation) | auto-download: `curl -L -o v.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip` (same pattern for `-train`); then `python src/data/visdrone.py` |
| **Okutama-Action** | RQ2 states + demo footage | public Dropbox; single files via `https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/APTyV-f01XLnJ0WFpZSBLOE?preview=<FILE>&dl=1&rlkey=7u7131amaul29amyr4jbnnu03` where `<FILE>` ∈ `Sample.zip, TrainSetVideos.zip, TestSetVideos.zip` → unzip into `data/raw/okutama/` |
| **UAV-Human (pose subset)** | RQ1 aerial keypoint GT | public Google Drive (no form needed): folder https://drive.google.com/drive/folders/1QeYXeM_pbWBSSmpRr_rKHurMpI2TxAKs → `PoseEstimation/PoseEstimation.zip`, or `pip install gdown && gdown 1kWStmFjrN1Njf6mj4rTso6XPMULcFKS5` → unzip into `data/raw/uavhuman_pose/` |
| **SARD** | optional extra detection data | free IEEE account: https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard — or the no-login Roboflow mirror: https://universe.roboflow.com/dataset-ay6sw/sard-peykp |
| **HIT-UAV** | optional thermal extension | https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset (git clone) |
| **COCO val2017 keypoints** | PCK evaluator sanity check | auto-fetched by `src/eval_pose.py` |

Note: Okutama has **no "Waving" action** (verified on real labels) — the triage
taxonomy is *motionless / stationary / mobile*; a "signaling" class would need
extra data (e.g. UAV-Gesture) and is documented as future work.

## Reproduce the local results

```bash
.venv/bin/python scripts/detection_experiments.py   # AP50 zero-shot vs fine-tuned
.venv/bin/python scripts/train_rq2_local.py         # RQ2 smoke-scale, saves pose_mlp.pt
.venv/bin/python src/eval_pose.py                   # PCK sanity on COCO val subset
.venv/bin/python src/render_demo.py data/raw/okutama/1.1.1.mov --max-frames 300
```

Outputs land in `results/tables/`, `results/figures/`, `results/videos/`.

## Full experiments (Colab, free T4)

1. `cd Project && zip -r src.zip src scripts`
2. Open each notebook in `notebooks/` on Colab (GPU runtime), upload `src.zip`
   when prompted; results are written to your Google Drive.
3. Order: 01 → 02 → 03 → 04 (04 consumes weights from 01 and 03).

## Team split

One partner owns detection/tracking + notebook 01 + the demo (04); the other
owns pose + both research questions (02, 03). `src/` is shared.
