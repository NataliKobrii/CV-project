# Local results (smoke scale) — 2026-07-18

Everything below ran locally (M1 Pro, MPS/CPU) on real data. These are the
*smoke-scale* versions of the experiments; the full-scale runs live in
`notebooks/01–04` on Colab GPUs. Numbers are honest but small-scale — treat
them as existence proofs and trend indicators, not final report numbers.

## Foundation — person detection (VisDrone-person val, 548 images, 13,969 GT boxes)

Custom AP@0.5 evaluator (`src/eval_detect.py`), identical for both rows.
Source: `tables/detection_baseline.csv`.

| model | AP50 | recall | notes |
|---|---|---|---|
| YOLO11n zero-shot (COCO) | 0.351 | 0.523 | ground-level model on aerial imagery |
| YOLO11n fine-tuned (2 ep, 25% train) | **0.446** | **0.716** | +9.5 AP50 from a *tiny* fine-tune |

**Takeaway:** the aerial domain gap for detection is real and cheaply narrowed.
Full 30-epoch yolo11s fine-tune (notebook 01) should land far higher.

## RQ1 — the aerial pose gap is a SMALL-PERSON problem, not a viewpoint problem

Same model (RTMPose-m), same evaluator (PCK@0.1, height-stratified), two domains:

| domain | PCK@0.1 | PCK h50–100px | PCK h100px+ |
|---|---|---|---|
| ground-level (COCO val subset, 100 persons) | 0.946 | 0.90 | 0.956 |
| **aerial (UAV-Human, 800 persons, 11,878 kpts)** | 0.936 | **0.469** | 0.940 |

Sources: `tables/pck_coco_sanity.json`, `tables/pck_uavhuman_zeroshot.json`.

**Finding:** at UAV-Human's typical person size (median 359 px) the aerial
*viewpoint* costs only ~1 PCK point — but in the 50–100 px bin accuracy
craters to 0.47 (caveat: small support locally; the full 22,476-frame run in
notebook 02 firms this up, and its downscale ablation traces the whole
scale-degradation curve). Consistent proxy on Okutama (30–80 px people):
mean keypoint confidence drops to ~0.50. This reframes RQ1 — and explains
RQ2's result below: pose features lose exactly where keypoints get noisy.

## RQ2 — pose vs appearance, Okutama sample video (held-out tracks)

1,313 track windows from 99 tracks (70/30 split by track id, no leakage);
identical training loop/splits for both models. Source: `tables/rq2_local.csv`,
figures: `figures/confusion_pose_mlp_local.png`, `figures/confusion_appearance_cnn_local.png`.

| model | macro-F1 | acc | F1 motionless | F1 stationary | F1 mobile |
|---|---|---|---|---|---|
| PoseMLP (pose features) | 0.62 | 0.67 | 0.51 (R=0.81) | 0.72 | 0.64 |
| AppearanceCNN (raw crops) | **0.72** | **0.74** | **0.71** (R=0.94) | 0.80 | 0.64 |

**Honest finding:** at single-video scale, appearance wins. Working hypotheses
for the report (to be tested in notebook 03 at full scale with *video-level*
splits): (1) RTMPose keypoints are noisy on ~30–80 px people — RQ1's domain
gap propagating into RQ2; (2) the CNN can exploit scene-specific background
context inside crops (lying people on distinctive ground patches) that won't
generalize across videos. Either outcome at full scale is a reportable result.

## Pipeline verification

- `scripts/smoke_test.py`: **9/9 stages pass** (imports/MPS, parser, detection,
  tracking, pose, features, classifier mechanics, triage, demo render).
- Demo videos in `videos/`: `smoke_demo.mp4` (rule-based states) and
  `demo_final_local.mp4` (fine-tuned detector + trained PoseMLP, frames
  450–900 of the sample video where all three states co-occur).

## Dataset facts worth citing

- Okutama-Action has **no Waving class** (verified on labels) → taxonomy is
  motionless/stationary/mobile; "signaling" needs UAV-Gesture (future work).
- Sample-video state distribution: 8,433 stationary / 5,559 mobile /
  1,337 motionless / 205 unknown boxes across 2,272 labeled frames.
