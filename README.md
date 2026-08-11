# Is Anyone Down There Moving?
**Pose-based human state recognition for search-and-rescue drones**
EECS 4422 Computer Vision · Summer 2026 · York University

A drone that sweeps a disaster area should not just find people. It should
distinguish a person who is lying motionless from one who is walking around, so
that rescuers get a priority list instead of a dot map. The pipeline detects
people, tracks them, estimates their pose, classifies their state and
produces the triage output. RQ3 adds an optional restoration front-end, and
RQ4 adds a registration back-end that draws a victim map.

We study four research questions:

- **RQ1 (domain gap).** How much do 2D pose estimators pretrained on
  ground-level images degrade under aerial viewpoints and scales, and how
  much does fine-tuning recover?
- **RQ2 (pose against appearance).** Does an explicit pose representation outperform an
  appearance-only baseline at classifying people as motionless, stationary
  or mobile?
- **RQ3 (restoration).** Can frequency-domain restoration (the inverse
  filter, the Wiener filter and Richardson-Lucy deconvolution) undo motion
  blur and sensor noise on aerial frames, and does it recover detection AP50
  and pose PCK rather than only PSNR and SSIM?
- **RQ4 (matching and mapping).** How do the classical descriptors (SIFT and
  ORB) and a learned patch descriptor compare under viewpoint, scale and
  illumination changes, and are the resulting homographies good enough to
  register drone frames and place every track on one victim map of the
  sweep?

## Repository layout

```
src/                 The pipeline modules, imported by the scripts and notebooks.
  config.py          Paths, the triage taxonomy, colors and model choices.
  detect.py          The YOLO person-detection wrapper.
  track.py           The ByteTrack tracking wrapper.
  pose.py            RTMPose top-down pose with a YOLO-pose fallback.
  features.py        Pose keypoints turned into feature vectors.
  classify.py        The PoseMLP and the AppearanceCNN with a shared training loop.
  triage.py          The mapping from state to priority, colors and the summary line.
  eval_detect.py     The AP@0.5 evaluator, written from scratch.
  eval_pose.py       PCK stratified by person pixel height (the RQ1 metric).
  restore.py         RQ3: the degradations and the restoration filters.
  eval_restore.py    PSNR and SSIM from scratch, plus the RQ3 harnesses.
  match.py           RQ4: SIFT, ORB and the TinyDescNet patch descriptor.
  eval_match.py      Matching accuracy under known homographies.
  register.py        Frame registration into a mosaic and the victim map.
  render_demo.py     The demo video renderer with the triage overlay.
  data/visdrone.py   Converts VisDrone to a YOLO person-only dataset.
  data/okutama.py    Parses the Okutama-Action labels and resolves the states.
  data/uavhuman.py   Loads the UAV-Human pose subset.
scripts/
  sanity_check.py             An end-to-end check of all 12 pipeline stages.
  detection_experiments.py    The local detection baseline and a short fine-tune.
  train_rq2_local.py          The local RQ2 experiment on the sample video.
  rq1_uavhuman.py             The local zero-shot aerial PCK for RQ1.
  restoration_experiments.py  The local RQ3 experiments.
  matching_experiments.py     The local RQ4 experiments.
notebooks/           The Colab notebooks for the full experiments (free T4 GPU).
  01_detection_finetune.ipynb    Fine-tunes the detector on VisDrone.
  02_pose_domain_gap.ipynb       RQ1: the aerial pose domain gap.
  03_state_classification.ipynb  RQ2: pose against appearance on full Okutama.
  04_demo_pipeline.ipynb         The final demo video on held-out footage.
  05_restoration.ipynb           RQ3 at full scale, with optional real blur.
  06_matching_hpatches.ipynb     RQ4 on the HPatches benchmark.
data/                The datasets (not committed; see below).
models/              Trained weights, such as pose_mlp.pt and the YOLO runs.
results/             The tables, figures and videos that the report cites.
report/              The proposal, the pitch and the report skeleton.
```

## Setup (local)

```bash
cd Project
/opt/homebrew/bin/python3.11 -m venv .venv        # Any Python 3.10-3.12 works.
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/sanity_check.py           # This should print 12/12 stages passed.
```

## Data

| Dataset | Used for | Access |
|---|---|---|
| **VisDrone-DET** | person detection (the foundation) | Auto-download: `curl -L -o v.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip` (the same pattern works for `-train`); then run `python src/data/visdrone.py`. |
| **Okutama-Action** | the RQ2 states and the demo footage | Public Dropbox folder; single files via `https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/APTyV-f01XLnJ0WFpZSBLOE?preview=<FILE>&dl=1&rlkey=7u7131amaul29amyr4jbnnu03`, where `<FILE>` is one of `Sample.zip`, `TrainSetVideos.zip`, `TestSetVideos.zip`; unzip into `data/raw/okutama/`. |
| **UAV-Human (pose subset)** | the RQ1 aerial keypoint ground truth | Public Google Drive, no registration required: folder https://drive.google.com/drive/folders/1QeYXeM_pbWBSSmpRr_rKHurMpI2TxAKs (`PoseEstimation/PoseEstimation.zip`), or `pip install gdown && gdown 1kWStmFjrN1Njf6mj4rTso6XPMULcFKS5`; unzip into `data/raw/uavhuman_pose/`. |
| **SARD** | optional extra detection data | Free IEEE account: https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard, or the no-login Roboflow mirror: https://universe.roboflow.com/dataset-ay6sw/sard-peykp. |
| **HIT-UAV** | an optional thermal extension | Clone https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset. |
| **COCO val2017 keypoints** | a sanity check for the PCK evaluator | Fetched automatically by `src/eval_pose.py`. |
| **HPatches sequences** | the RQ4 matching benchmark | Downloaded automatically by notebook 06 (`http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz`, about 4.2 GB). The local benchmark needs no download, because it uses synthetic warps of VisDrone frames. |
| **GoPro deblurring** | an optional real-blur check for RQ3 | Notebook 05, Part D (optional). |

Okutama has no "Waving" action; we verified this on the real labels. The
triage taxonomy is therefore motionless, stationary and mobile. A
"signaling" class would need extra data such as UAV-Gesture and is
documented as future work.

## Reproduce the local results

```bash
.venv/bin/python scripts/detection_experiments.py   # AP50 of the zero-shot and the fine-tuned detector.
.venv/bin/python scripts/train_rq2_local.py         # The small-scale RQ2 run; saves pose_mlp.pt.
.venv/bin/python src/eval_pose.py                   # The PCK sanity check on COCO.
.venv/bin/python scripts/rq1_uavhuman.py            # The zero-shot aerial PCK for RQ1.
.venv/bin/python scripts/restoration_experiments.py # The RQ3 tables and figures.
.venv/bin/python scripts/matching_experiments.py    # The RQ4 benchmark and the victim map.
.venv/bin/python src/render_demo.py data/raw/okutama/1.1.1.mov --max-frames 300
```

The outputs are written to `results/tables/`, `results/figures/` and
`results/videos/`.

## Full experiments (Colab, free T4)

The notebooks contain only the experiments. To run them:

1. Zip the project code once: `cd Project && zip -r src.zip src scripts`.
2. Open a notebook in Colab and switch it to a GPU: open the Runtime menu,
   click "Change runtime type" and pick T4 GPU (free).
3. Run the first code cell. It installs the packages, asks you to upload
   `src.zip`, and mounts Google Drive. Instead of uploading every time, you
   can put `src.zip` in Drive once and set the `SRC_ZIP` path in that cell.
4. Run the remaining cells top to bottom. Every table, figure and weight is
   saved to `sar_project_results/` in your Drive, so nothing is lost if the
   session disconnects. Copy the outputs into `results/` and `models/` here afterwards.
5. Run 01 first, then 02, 03 and 04 (04 needs the weights from 01 and 03).
   Notebooks 05 and 06 can run at any point; 05 reuses the weights from 01
   when they are in Drive. The longest run is notebook 01 at about 40
   minutes on the T4.

## Division of responsibilities

One partner is responsible for detection and tracking, notebook 01 and
the demo (notebook 04). The other is responsible for pose and both
original research questions (notebooks 02 and 03). The `src/` code is
shared. The added questions are assigned the same way: RQ3 and notebook 05
belong to the first partner, because restoration feeds detection, and RQ4
and notebook 06 to the second.
