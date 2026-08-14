# Presentation — Pose-Based Human State Recognition for Search-and-Rescue Drones

Sixteen slides for a fifteen-minute recorded presentation. Each slide lists its
assigned speaker and the figure to display. Figures are in `results/`. The
detailed narration and timing are in `video_script.md`.

---

## Slide 1 — Title (all members on camera)
**Pose-Based Human State Recognition for Search-and-Rescue Drones**
EECS 4422 Computer Vision — Summer 2026 — York University
Nataliia Kobrii · Daniel Vinitski · Nityam Goyal
*Each member appears on camera and states their name.*

---

## Slide 2 — Motivation (Nataliia)
- Drones sweep a disaster area far faster than ground teams, but a human
  operator must still locate people and judge their condition.
- The operator is the bottleneck.
- Detection alone yields a map of dots; adding each person's state yields a
  triage priority list.
- The key judgement — motionless versus signalling — is a human-pose problem.

---

## Slide 3 — Research questions (Nataliia)
- **RQ1:** How much does a ground-trained pose estimator degrade on aerial
  imagery, and how much does fine-tuning recover?
- **RQ2:** Does an explicit pose representation outperform an appearance-only
  baseline for state classification?
- **RQ3:** Does frequency-domain restoration recover pixel quality and task
  performance on degraded aerial frames?
- **RQ4:** How do SIFT, ORB, and a learned descriptor compare, and can
  registration build a single victim map?

---

## Slide 4 — Pipeline architecture (Nataliia)
- Detection, tracking, pose estimation, features, classification, and the triage
  overlay.
- YOLO11s detector, ByteTrack tracker, RTMPose top-down pose.
- A 47-dimensional feature vector per track window (keypoints, geometry,
  motion).
- Two classifiers compared under identical splits: PoseMLP and AppearanceCNN.

---

## Slide 5 — Data and taxonomy (Daniel)
- VisDrone-person (detection), COCO and UAV-Human (pose), Okutama-Action
  (state, demo), HPatches (matching).
- Okutama has no *Waving* class; the taxonomy is motionless, stationary, mobile.
- A precedence rule resolves multi-action boxes: motionless first.
- Synthetic degradations with known ground truth support the restoration study.

---

## Slide 6 — Detection foundation (Daniel)
- Our own AP@0.5 evaluator on 548 VisDrone-person validation images.
- Zero-shot COCO model: AP50 0.437.
- Fine-tuned for 30 epochs: **AP50 0.709**, recall 0.859.
- *Figure: results/detection/runs/yolo11s_visdrone_ft/results.png*

---

## Slide 7 — RQ1: the aerial pose gap (Daniel)
- Ground-level reference PCK@0.1 = 0.95.
- Aerial evaluation over 22,319 UAV-Human persons.
- Overall PCK is within one point of ground level, but by size it collapses:
  **0.944 above 100 px, 0.440 at 50–100 px, and 0.091 below 50 px.**
- The gap is a small-person problem, not a viewpoint problem.
- *Figure: results/pose_domain_gap/figures/rq1_pck_vs_scale.png*

---

## Slide 8 — RQ1: recovery (Daniel)
- Label-free recovery: pseudo-label Okutama with a teacher, fine-tune a
  yolo11n-pose student.
- Student pose mAP@0.5 = 0.508, mAP@0.5:0.95 = 0.307.
- A full before-and-after PCK quantification remains future work.

---

## Slide 9 — RQ2: pose against appearance (Nityam)
- Whole videos held out — a strict cross-video split.
- PoseMLP macro-F1 0.363; AppearanceCNN macro-F1 0.399.
- Both models fail on the rare motionless class (F1 = 0).
- The difficulty lies in the small-person regime identified in RQ1.
- *Figures: results/figures/confusion_pose_mlp_full.png and confusion_appearance_cnn_full.png*

---

## Slide 10 — RQ3: restoration (Nataliia)
- Six synthetic degradations; frequency-domain against spatial filters.
- Pixel quality: Wiener matches Richardson–Lucy on blur; spatial filters win on
  noise.
- **Task performance: PSNR does not equal utility.** On noise, non-local means
  raises PSNR but lowers detection AP (0.387) below the degraded baseline
  (0.460); the median filter improves it (0.496).
- *Figure: results/figures/restoration_wiener_k_full.png*

---

## Slide 11 — RQ4: descriptor comparison (Nityam)
- 116 HPatches sequences, matching accuracy and homography error.
- **SIFT** — highest accuracy and precision. **ORB** — three-to-five times
  faster, weaker at strict thresholds. **TinyDescNet** — near SIFT under
  illumination, weaker under viewpoint.
- A restoration- and viewpoint-aware descriptor choice matters for mapping.

---

## Slide 12 — RQ4: the victim map (Nityam)
- SIFT and RANSAC register every twelfth frame of a 288-frame sweep
  (mean 905 inliers per link).
- Chained homographies build a mosaic; foot points project each track's state.
- The per-frame overlay becomes one operator-facing map.
- *Figures: results/figures/registration_mosaic_1.1.1.png and victim_map_1.1.1.png*

---

## Slide 13 — Demonstration (Nataliia)
- Full pipeline over 1,800 frames of held-out Okutama footage.
- Detection, tracking, pose, state classification, and the triage overlay.
- *Play: results/videos/demo_final.mp4 (about 45–60 seconds).*

---

## Slide 14 — Status: functional, partial, assumptions (all, by area)
- **Fully functional:** the end-to-end pipeline, detection fine-tuning, the
  full-scale RQ1, RQ2, RQ3, and RQ4 evaluations, and the demonstration.
- **Partially functional:** the RQ1 recovery quantification and the RQ2
  ablations are not completed at full scale; registration is shown on one video.
- **Assumptions:** three-state taxonomy, box-normalised PCK, known point spread
  function, and a planar scene for registration.

---

## Slide 15 — Lessons and conclusions (Nataliia)
- The aerial gap is driven by person size; small people break both pose and
  detection.
- Restoration must be evaluated on the downstream task, not on PSNR.
- Classical descriptors remain strong baselines; a small learned descriptor is
  competitive only within its training regime.
- The pipeline gives a realistic account of where aerial triage succeeds and
  fails.

---

## Slide 16 — Contributions and closing (all)
- **Daniel Vinitski:** data preparation, detection fine-tuning, and the aerial
  pose-gap study (notebooks 01 and 02).
- **Nataliia Kobrii:** state classification (RQ2) and the matching, registration,
  and victim map (RQ4) (notebooks 03 and 06).
- **Nityam Goyal:** pipeline integration, restoration (RQ3), the demonstration,
  and the report (notebooks 04 and 05).
- Thank you. Repository and demonstration links are in the submission.
