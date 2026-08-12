# Full-scale results – 2026-08-12

These are the final, report-grade numbers. Notebooks 01-02 ran on Colab GPUs
(the detector and the student pose model); notebooks 03-06 and the two
detector-dependent restoration/demo artifacts ran locally on the M1 Pro
(MPS plus CPU) at full scale. Every number below traces to one detector
checkpoint, `models/yolo11s_visdrone_best.pt` (AP50 0.709). The small-scale
log from July is kept below as a record and as the source of the predictions
these runs confirm.

## Detection – VisDrone-person validation (548 images, 13,969 boxes)

Source `detection/tables/detection_visdrone.csv`, our own AP@0.5 evaluator.

| model | AP50 | recall |
|---|---|---|
| yolo11s zero-shot (COCO) | 0.437 | 0.596 |
| **yolo11s fine-tuned (30 epochs)** | **0.709** | **0.859** |

The 30-epoch fine-tune lifts AP50 by 0.27 over the ground-level model, confirming
the small-scale prediction that the aerial gap narrows substantially with training.
The full training curves and confusion matrices are in
`detection/runs/yolo11s_visdrone_ft/`.

## RQ1 – the aerial pose gap is a small-person problem (notebook 02)

Ground-level sanity check (COCO subset, 150 persons): PCK@0.1 = 0.95, already
showing the size effect (0.958 above 100 px vs 0.90 at 50-100 px). The full
aerial evaluation runs RTMPose-m zero-shot over all UAV-Human frames — 22,319
persons, 334,217 keypoints. Sources `pose_domain_gap/tables/pck_coco_sanity.json`
and `pose_domain_gap/tables/pck_uavhuman_zeroshot.json`.

| domain | PCK@0.1 | h25-50 px | h50-100 px | h100+ px |
|---|---|---|---|---|
| ground-level (COCO, 150 persons) | 0.95 | – | 0.90 | 0.958 |
| **aerial (UAV-Human, 22,319 persons)** | 0.942 | **0.091** | **0.440** | 0.944 |

The overall aerial PCK is within a point of ground-level, but stratified by
person height it collapses: fine above 100 px (0.944), halved in the 50-100 px
bin (0.440), and near-total failure below 50 px (0.091). This confirms and
sharpens the small-scale finding at 150x the support: **the aerial pose gap is
a small-person problem, not a viewpoint problem**, and it explains the RQ2 result
below — pose features fail exactly where the keypoints become unreliable. The
built-in downscale ablation (people shrunk to 35 %) isolates scale as the cause.

We also train the domain-adapted student pose model (yolo11n-pose on Okutama
pseudo-labels, 20 epochs): pose mAP@0.5 = 0.508, mAP@0.5:0.95 = 0.307
(`pose_domain_gap/runs/pose_ft/`).

## RQ2 – pose against appearance, full scale with video-level splits (notebook 03)

Whole videos are held out, which is stricter than the small-scale track-level
split. Source `tables/rq2_full.csv`, figures
`figures/confusion_pose_mlp_full.png` and `figures/confusion_appearance_cnn_full.png`.

| model | macro-F1 | F1 mobile | F1 stationary | F1 motionless |
|---|---|---|---|---|
| PoseMLP (pose features) | 0.363 | 0.631 | 0.458 | 0.000 |
| **AppearanceCNN (raw crops)** | **0.399** | 0.611 | 0.588 | 0.000 |

Two findings. First, both models collapse on the rare motionless class (F1 = 0),
where the small-scale run still reached 0.5-0.7 — video-level generalization is
much harder than track-level, and motionless (people lying down) is the class
that suffers most. Second, appearance still edges out pose, but the margin
shrinks to 0.036, consistent with the RQ1 story: the pose features are limited
by unreliable keypoints on 30-80 px people, while the appearance model loses its
track-level background shortcut once whole videos are held out.

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

Frequency-domain methods win the blur cases (Wiener matches Richardson-Lucy at a
fraction of the cost), and nonlinear spatial filters win the noise cases, each
noise type needing a different filter. The Wiener K sweep over 20 frames peaks at
K = 0.046 (`figures/restoration_wiener_k_full.png`).

Extrinsic — does restoration recover detection? Over all 548 images with the
canonical detector (`tables/restoration_extrinsic_detection_full.csv`):

| condition | clean | degraded | best restore | others |
|---|---|---|---|---|
| motion25_n5 | 0.709 | 0.025 | **wiener 0.091** | rl20 0.079, unsharp 0.011 |
| noise25 | 0.709 | 0.460 | **median 0.496** | butterworth 0.428, nlm 0.387 |

The headline result of the section holds at full scale: PSNR is not task utility.
On noise, non-local means reaches good PSNR yet **lowers** detection (0.387 below
the 0.460 degraded baseline) by smoothing away small people, while the simple
median filter is the only method that improves it. At blur comparable to the
person size, detection is unrecoverable (0.025, Wiener only to 0.091) — the
information is destroyed, not hidden. A restoration method must be validated on
the task, not on PSNR.

## RQ4 – matching on HPatches, all 116 sequences (notebook 06)

Source `tables/matching_hpatches.csv`, 285 illumination and 295 viewpoint pairs.

| method | MMA@3 illum | MMA@3 vp | MMA@1 vp | corner err vp (px) | ms/pair |
|---|---|---|---|---|---|
| **SIFT** | **0.700** | **0.710** | **0.466** | **51.7** | 148-251 |
| ORB | 0.675 | 0.670 | 0.283 | 119.9 | **52-79** |
| TinyDescNet | 0.646 | 0.601 | 0.383 | 93.5 | ~2050 |

SIFT is the accuracy and geometric-precision leader, especially under viewpoint
change. ORB trades accuracy for a 3-5x speedup and collapses at the strict
1-pixel threshold under viewpoint (MMA@1 0.28). Our self-supervised TinyDescNet
nearly matches SIFT under illumination (MMA@1 0.505 vs 0.522) but lags under
viewpoint, consistent with training only on aerial appearance jitter rather than
geometric warps. The real HPatches sequences are harder than the small-scale
synthetic warps below, so the absolute numbers are lower but the ranking is the
same.

## Demo

`videos/demo_final.mp4` — the full pipeline (canonical detector, ByteTrack,
RTMPose, PoseMLP states, triage overlay) over 1,800 frames of Okutama 1.1.1,
h264, rendered with the canonical detector.

---

# Local results (small scale) – 2026-07-18, RQ3 and RQ4 added 2026-07-25

Everything below ran locally on the M1 Pro (MPS, the Apple GPU backend, and the CPU) on real data. These
are the small-scale versions of the experiments; the full-scale runs are in
notebooks 01-06 on Colab GPUs. The numbers are small-scale, so we treat
them as preliminary indicators rather than final report numbers.

## Foundation – person detection (VisDrone-person validation, 548 images, 13,969 boxes)

Both rows use our own AP@0.5 evaluator from `src/eval_detect.py`, so they
are directly comparable. The source is `tables/detection_baseline.csv`.

| model | AP50 | recall | notes |
|---|---|---|---|
| YOLO11n zero-shot (COCO) | 0.351 | 0.523 | A ground-level model on aerial images. |
| YOLO11n fine-tuned (2 epochs, 25% of the training split) | **0.446** | **0.716** | +9.5 AP50 from a very small fine-tune. |

The aerial domain gap for detection is substantial, and even a small fine-tune
narrows it. The full 30-epoch yolo11s fine-tune in notebook 01 should
improve this result substantially.

## RQ1 – the aerial pose gap is a small-person problem, not a viewpoint problem

We evaluate the same model (RTMPose-m) with the same evaluator (PCK@0.1,
stratified by person height) on two domains:

| domain | PCK@0.1 | PCK h50-100px | PCK h100px+ |
|---|---|---|---|
| ground-level (COCO validation subset, 100 persons) | 0.946 | 0.90 | 0.956 |
| **aerial (UAV-Human, 800 persons, 11,878 keypoints)** | 0.936 | **0.469** | 0.940 |

The sources are `tables/pck_coco_sanity.json` and
`tables/pck_uavhuman_zeroshot.json`.

At the typical UAV-Human person size (the median is 359 pixels), the aerial
viewpoint costs only about 1 PCK point. In the 50-100 pixel bin, however,
accuracy drops to 0.47. The local support for that bin is small; the full
22,476-frame run in notebook 02 confirms this, and its downscale ablation
traces the whole scale-degradation curve. A consistent proxy on Okutama,
where people are 30-80 pixels tall, is that the mean keypoint confidence drops
to about 0.50. This reframes RQ1 and explains the RQ2 result below: pose
features fail exactly where the keypoints become unreliable.

## RQ2 – pose against appearance on the Okutama sample video (held-out tracks)

We build 1,313 track windows from 99 tracks and split them 70 to 30 by track
id, so no track leaks between the splits. Both models use the identical
training loop and splits. The source is `tables/rq2_local.csv`; the figures
are `figures/confusion_pose_mlp_local.png` and
`figures/confusion_appearance_cnn_local.png`.

| model | macro-F1 | accuracy | F1 motionless | F1 stationary | F1 mobile |
|---|---|---|---|---|---|
| PoseMLP (pose features) | 0.62 | 0.67 | 0.51 (R = 0.81) | 0.72 | 0.64 |
| AppearanceCNN (raw crops) | **0.72** | **0.74** | **0.71** (R = 0.94) | 0.80 | 0.64 |

At single-video scale, the appearance model performs better. We have two working hypotheses to
test in notebook 03 at full scale with video-level splits. First, the
RTMPose keypoints are noisy on 30-80 pixel people, which is the RQ1 domain gap
propagating into RQ2. Second, the CNN can exploit scene-specific background
context inside the crops, such as lying people on distinctive ground
patches, and that context will not generalize across videos. Either outcome
at full scale is a reportable result.

## RQ3 – frequency-domain restoration: pixel quality recovers, task performance only sometimes

We use six degradation conditions with known ground truth on 12 VisDrone
validation frames (width at most 1280). PSNR and SSIM are our own
implementations from `src/eval_restore.py`, and the runtimes are measured on
the M1 Pro CPU. The sources are `tables/restoration_intrinsic.csv`, the
qualitative strips `figures/restoration_grid_*.png` and the K sweep
`figures/restoration_wiener_k.png`.

The intrinsic results in PSNR dB (selected rows):

| condition | degraded | inverse | wiener | rl20 | best spatial |
|---|---|---|---|---|---|
| motion15_n2 | 22.8 | 8.4 | **25.6** (80 ms) | 25.7 (2.3 s) | 22.8 (gaussian) |
| motion25_n5 | 21.7 | 6.0 | 22.9 | **23.4** | 21.9 |
| defocus7_n2 | 22.4 | 6.3 | **24.9** | 24.1 | 22.5 |
| noise25 | 20.5 | – | – | – | **26.9 (bilateral)** |
| saltpepper4 | 19.0 | – | – | – | **26.5 (median)**; bilateral 18.9, nlm 21.1 |

- For deblurring, the frequency-domain methods perform best: the Wiener
  filter matches Richardson-Lucy at 1/29th of the runtime, and the inverse
  filter shows the expected noise amplification.
- For denoising, the nonlinear spatial filters perform best. The
  Butterworth low-pass, our frequency-domain method, is weak on Gaussian
  noise and worse than the median filter on impulse noise. Each noise type
  therefore requires a different filter.
- In the Wiener K sweep, the best K is about 0.02 at sigma 5. A slightly
  wrong PSF costs about 2 dB (with the angle deviating by 10 degrees, PSNR falls
  from 21.6 to 19.5), which supports blind PSF estimation as future work.

The extrinsic evaluation is our addition to the topic: does restoration
recover pipeline performance? We use the fine-tuned yolo11n on 60 validation
images with the same AP evaluator, and RTMPose PCK on 200 UAV-Human persons.
The sources are `tables/restoration_extrinsic_detection.csv` and
`tables/restoration_extrinsic_pose.csv`.

| metric, condition | clean | degraded | wiener | others |
|---|---|---|---|---|
| AP50, motion15_n2 | 0.536 | 0.068 | **0.405** | rl20 0.200 |
| AP50, motion25_n5 | 0.536 | 0.010 | 0.058 | rl20 0.039, unsharp 0.007 |
| AP50, noise25 | 0.536 | 0.374 | – | **median 0.405**, nlm 0.268, butterworth 0.305 |
| PCK, motion15_n2 | 0.958 | 0.931 | **0.952** | (h50-100 bin: 0.25 to 0.67) |
| PCK, noise25 | 0.958 | 0.937 | – | median 0.948, butterworth 0.945 |

We draw three conclusions. First, at moderate blur the Wiener filter
recovers about 76% of the lost detection AP (from 0.068 back to 0.405) and
nearly all of the pose PCK. Second, at blur comparable to the person size
(25 pixels), detection is unrecoverable, because the information is destroyed
rather than hidden. Third, PSNR does not equal task utility: non-local
means and Butterworth reach reasonable PSNR yet degrade detection (from
0.374 down to 0.27 and 0.30) by smoothing away small people, while the
simple median filter improves it. A restoration method must be validated on the task.

## RQ4 – matching: SIFT, ORB and a learned descriptor, and the victim map

The benchmark is 90 synthetic-warp pairs over 10 validation frames with
controlled viewpoint, scale and illumination changes and known homographies,
evaluated with the mean-matching-accuracy protocol in `src/eval_match.py`.
TinyDescNet is a shallow CNN in the style of L2-Net, trained locally in
about a minute on 5,095 self-supervised aerial patch pairs from
training-split frames, so there is no image overlap with the benchmark. The
training uses the HardNet loss with the hardest negative in the batch, and
the final loss fell from 0.98 to 0.69. The sources are
`tables/matching_local.csv` and `tables/matching_pr_local.csv`; the figures
are `matching_mma_local.png` and `matching_pr_local.png`.

| method | MMA@3 viewpoint | MMA@3 scale | MMA@3 illumination | homography corner error (viewpoint) | ms per pair |
|---|---|---|---|---|---|
| SIFT | **0.942** | **0.909** | 0.977 | 0.22 pixels | 141 |
| ORB | 0.911 | 0.887 | **0.983** | 2.23 pixels | **57** |
| TinyDescNet | 0.929 | 0.837 | 0.975 | **0.20 pixels** | 400 |

- SIFT performs best overall. ORB is 2.5 times faster but imprecise below
  3 pixels (its MMA@1 is 0.52 against 0.90 for SIFT), which is sufficient for
  coarse alignment but worse for map building.
- The learned descriptor reaches SIFT-level accuracy on viewpoint and
  illumination after 6 epochs on 5,000 pairs. Scale is its weakest condition
  (0.84), and the Easy, Hard and Tough splits in notebook 06 examine
  exactly this case.

The applied result is the registration and the victim map from
`src/register.py`. We take 288 frames of the Okutama sample, register every
12th frame with SIFT and RANSAC (905 inliers per link on average, 610 at
minimum), chain the homographies back to the first frame, build the mosaic,
and project all 15 pipeline tracks by their foot points. The pipeline here
is the fine-tuned detector, ByteTrack, RTMPose and the PoseMLP states. The
outputs are `figures/registration_mosaic_1.1.1.png`,
`figures/victim_map_1.1.1.png` and `tables/registration_stats_1.1.1.json`.
The per-frame triage overlay becomes one operator map of the whole sweep.

## Pipeline verification

- `scripts/sanity_check.py` passes 12 of 12 stages: imports and MPS, the
  parser, detection, tracking, pose, features, classifier mechanics, triage,
  the demo render, the RQ3 Wiener round trip (+2.6 dB), RQ4 matching (MMA@3
  of 0.96 and a homography error of 0.13 pixels) and RQ4 two-frame registration
  (478 inliers).
- The demo videos are in `videos/`: `sanity_demo.mp4` uses the rule-based
  states, and `demo_final_local.mp4` uses the fine-tuned detector and the
  trained PoseMLP on frames 450-900 of the sample video, where all three
  states co-occur.

## Dataset facts for the report

- Okutama-Action has no Waving class; we verified this on the labels. The
  taxonomy is therefore motionless, stationary and mobile, and a signaling
  class would need UAV-Gesture (future work).
- The state distribution of the sample video is 8,433 stationary, 5,559
  mobile, 1,337 motionless and 205 unknown boxes across 2,272 labeled
  frames.
