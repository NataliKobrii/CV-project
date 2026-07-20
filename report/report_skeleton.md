# Final Report Skeleton — fill sections as results land

**Title:** Is Anyone Down There Moving? Pose-Based Human State Recognition for
Search-and-Rescue Drones
**Course:** EECS 4422 Computer Vision, Summer 2026, York University
**Team:** Natali Kobrii, ⟨teammate⟩

## 1. Introduction & Motivation (~0.75 p)
- SAR context: drones sweep faster than ground teams; the operator bottleneck.
- Triage argument: detection gives a dot map; states give a priority list
  (use the "3 motionless · 5 waving · 192 walking" framing from the proposal).
- State the two RQs verbatim from the proposal.

## 2. Related Work (~0.5 p, short)
- Aerial person detection (VisDrone line of work), top-down pose (RTMPose/ViTPose),
  skeleton-based action recognition (ST-GCN family), SAR-specific efforts (SARD, Okutama).

## 3. Data (~0.75 p)
- Table of datasets used + sizes. Include the verified finding: Okutama has no
  Waving class → taxonomy motionless/stationary/mobile (limitation → future work).
- Action→state mapping table + precedence rule (from `src/config.py`).
- State distribution table (real numbers from parser: e.g. sample video
  8,433 stationary / 5,559 mobile / 1,337 motionless boxes).

## 4. Method (~1.5 p)
- Pipeline figure: detect → track → pose → features → classify → triage.
- Detection/tracking: YOLO11 + ByteTrack (cite), fine-tuning setup.
- Pose: RTMPose top-down on tracked boxes; why top-down at aerial scales.
- Features: normalized keypoints + body-axis angle + temporal wrist/center
  motion (list from `src/features.py`), window size W=15.
- Classifiers: PoseMLP vs AppearanceCNN — stress identical splits/loop.
- Metrics: AP50 (own implementation), PCK@0.1·boxsize stratified by pixel
  height (justify box-size norm at aerial scale), macro-F1.

## 5. Experiments & Results
### 5.1 Detection foundation
- results/tables/detection_baseline.csv → before/after table (local smoke) +
  notebook-01 full numbers. Resolution ablation 640 vs 1280.
### 5.2 RQ1 — aerial domain gap for pose
- Ground-level reference PCK (COCO sanity, results/tables/pck_coco_sanity.json).
- Aerial zero-shot PCK vs height bins (UAV-Human) → THE domain-gap figure
  (rq1_pck_vs_scale.png). Fine-tuned recovery (pseudo-label student and/or
  UAV-Human fine-tune).
### 5.3 RQ2 — pose vs appearance
- results/tables/rq2_local.csv (smoke) + rq2_full.csv (Colab, video-level split).
- Confusion matrices (results/figures/confusion_*.png). Per-class discussion:
  which states does pose win on, and why (angle feature ≈ lying).
- Ablations: window size, crop resolution, ±temporal features.
### 5.4 Qualitative & failure analysis (high-value section)
- 6–8 frames from the demo; 10–20 misclassified crops with categorized causes
  (tiny person, occlusion, pose collapse, ambiguous GT).

## 6. Demo
- One paragraph + link/frame of results/videos/ final demo (held-out footage,
  fine-tuned weights, triage overlay).

## 7. Limitations & Future Work
- No Waving class (UAV-Gesture as the fix); no aerial 3D GT (lifting is
  qualitative only); geolocation needs camera metadata (flat-ground demo only);
  single-video smoke results vs full-set Colab results.

## 8. Contributions statement
- Who did what (required).

## Appendix
- Reproducibility: exact commands (README), seeds, hardware, runtimes.
