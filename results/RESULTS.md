# Full-scale results

These are the final results used in the report. Notebooks 01-02 ran on Colab GPUs
(the detector and the student pose model); notebooks 03-06 and the two
detector-dependent restoration/demo artifacts ran locally on the M1 Pro
(MPS plus CPU) at full scale. Every number below traces to one detector
checkpoint, `models/yolo11s_visdrone_best.pt` (AP50 0.709).

## Detection – VisDrone-person validation (548 images, 13,969 boxes)

Source `detection/tables/detection_visdrone.csv`, our own AP@0.5 evaluator.

| model | AP50 | recall |
|---|---|---|
| yolo11s zero-shot (COCO) | 0.437 | 0.596 |
| **yolo11s fine-tuned (30 epochs)** | **0.709** | **0.859** |

The 30-epoch fine-tune increases AP50 by 0.27 over the ground-level model,
confirming that the aerial gap narrows substantially with training.
The full training curves and confusion matrices are in
`detection/runs/yolo11s_visdrone_ft/`.

## RQ1 – the aerial pose gap as a function of person size (notebook 02)

The ground-level reference measurement (COCO subset, 150 persons) gives
PCK@0.1 = 0.95 and already exhibits the size effect (0.958 above 100 px against
0.90 at 50-100 px). The full aerial evaluation applies RTMPose-m zero-shot over
all UAV-Human frames (22,319 persons, 334,217 keypoints). Sources `pose_domain_gap/tables/pck_coco_sanity.json`
and `pose_domain_gap/tables/pck_uavhuman_zeroshot.json`.

| domain | PCK@0.1 | h25-50 px | h50-100 px | h100+ px |
|---|---|---|---|---|
| ground-level (COCO, 150 persons) | 0.95 | – | 0.90 | 0.958 |
| **aerial (UAV-Human, 22,319 persons)** | 0.942 | **0.091** | **0.440** | 0.944 |

The overall aerial PCK is within one point of the ground-level value, but when
stratified by person height the accuracy declines sharply: it remains high above
100 px (0.944), is approximately halved in the 50-100 px bin (0.440), and
approaches total failure below 50 px (0.091). The aerial pose gap is therefore
primarily a function of person size rather than of viewpoint. It also accounts for the RQ2
result below, since the pose features become unreliable precisely where the
keypoints do. The downscale ablation (persons reduced to 35 % of their size)
isolates scale as the underlying cause.

We also train the domain-adapted student pose model (yolo11n-pose on Okutama
pseudo-labels, 20 epochs): pose mAP@0.5 = 0.508, mAP@0.5:0.95 = 0.307
(`pose_domain_gap/runs/pose_ft/`).

## RQ2 – pose against appearance, full scale with video-level splits (notebook 03)

Whole videos are held out, which is a strict test of generalization across
scenes. Source `tables/rq2_full.csv`, figures
`figures/confusion_pose_mlp_full.png` and `figures/confusion_appearance_cnn_full.png`.

| model | macro-F1 | F1 mobile | F1 stationary | F1 motionless |
|---|---|---|---|---|
| PoseMLP (pose features) | 0.363 | 0.631 | 0.458 | 0.000 |
| **AppearanceCNN (raw crops)** | **0.399** | 0.611 | 0.588 | 0.000 |

Two findings emerge. First, both models fail on the rare motionless class
(F1 = 0), whereas at single-video scale the same models reached 0.5-0.7;
video-level generalization is considerably harder than track-level
generalization, and the
motionless class (persons lying down) is the most affected. Second, the
appearance model still outperforms the pose model, but the margin narrows to
0.036, which is consistent with the RQ1 finding: the pose features are limited by
unreliable keypoints on persons 30-80 px tall, while the appearance model loses
the track-level background cue it could otherwise exploit once whole videos are
held out.

## RQ3 – restoration, full scale (notebook 05)

Intrinsic quality over 200 VisDrone-val frames, best method per condition in
PSNR dB (`tables/restoration_intrinsic_full.csv`):

| condition | degraded | best deblur | best denoise |
|---|---|---|---|
| motion15_n2 | 23.8 | **rl20 26.7**, wiener 25.9; inverse 8.4 (fails) | – |
| motion25_n5 | 22.5 | **rl20 24.1**, wiener 23.5 | – |
| defocus7_n2 | 23.3 | **wiener 25.5**, rl20 25.2 | – |
| noise25 | 20.5 | – | **bilateral 27.3**, nlm 26.7 |
| saltpepper4 | 19.1 | – | **median 27.8** |

Frequency-domain methods perform best on the blur conditions (the Wiener filter
matches Richardson-Lucy at a fraction of the cost), and nonlinear spatial filters
perform best on the noise conditions, with each noise type requiring a different
filter. The Wiener K sweep over 20 frames peaks at K = 0.046
(`figures/restoration_wiener_k_full.png`).

Extrinsic evaluation — whether restoration recovers detection accuracy — over all
548 images, with the canonical detector
(`tables/restoration_extrinsic_detection_full.csv`):

| condition | clean | degraded | best restore | others |
|---|---|---|---|---|
| motion25_n5 | 0.709 | 0.025 | **wiener 0.091** | rl20 0.079, unsharp 0.011 |
| noise25 | 0.709 | 0.460 | **median 0.496** | butterworth 0.428, nlm 0.387 |

The principal finding of this section holds at full scale: PSNR does not measure
task utility. On noise, non-local means attains high PSNR yet **reduces**
detection accuracy (0.387, below the 0.460 degraded baseline) by smoothing away
small persons, whereas the median filter is the only method that improves it. At
blur comparable to the person size, detection accuracy is not recoverable (0.025;
the Wiener filter reaches only 0.091), because the information is destroyed rather
than merely obscured. A restoration method must therefore be validated on the
task, not on PSNR alone.

## RQ4 – matching on HPatches, all 116 sequences (notebook 06)

Source `tables/matching_hpatches.csv`, 285 illumination and 295 viewpoint pairs.

| method | MMA@3 illum | MMA@3 vp | MMA@1 vp | corner err vp (px) | ms/pair |
|---|---|---|---|---|---|
| **SIFT** | **0.700** | **0.710** | **0.466** | **51.7** | 148-251 |
| ORB | 0.675 | 0.670 | 0.283 | 119.9 | **52-79** |
| TinyDescNet | 0.646 | 0.601 | 0.383 | 93.5 | ~2050 |

SIFT attains the highest accuracy and geometric precision, particularly under
viewpoint change. ORB exchanges accuracy for a three-to-five-fold speedup and
degrades sharply at the strict 1-pixel threshold under viewpoint change
(MMA@1 0.28). The self-supervised TinyDescNet nearly matches SIFT under
illumination change (MMA@1 0.505 against 0.522) but performs worse under
viewpoint change, which is consistent with training only on aerial appearance
perturbations rather than on geometric warps. The full HPatches sequences are
challenging, so the absolute values are lower than on simpler synthetic warps,
but the ranking is clear.

## Demo

`videos/demo_final.mp4` — the full pipeline (canonical detector, ByteTrack,
RTMPose, PoseMLP states, triage overlay) over 1,800 frames of Okutama 1.1.1,
h264, rendered with the canonical detector.


## Pipeline verification

- `scripts/sanity_check.py` passes 12 of 12 stages: imports and MPS, the
  parser, detection, tracking, pose, features, classifier mechanics, triage,
  the demo render, the RQ3 Wiener round trip (+2.6 dB), RQ4 matching (MMA@3
  of 0.96 and a homography error of 0.13 pixels) and RQ4 two-frame registration
  (478 inliers).
- The demonstration video is `videos/demo_final.mp4`: the full pipeline with the
  fine-tuned detector, ByteTrack, RTMPose, and the trained PoseMLP over held-out
  Okutama frames, coloured by triage state.

## Dataset facts for the report

- Okutama-Action has no Waving class; we verified this on the labels. The
  taxonomy is therefore motionless, stationary and mobile, and a signaling
  class would need UAV-Gesture (future work).
- The state distribution of the sample video is 8,433 stationary, 5,559
  mobile, 1,337 motionless and 205 unknown boxes across 2,272 labeled
  frames.
