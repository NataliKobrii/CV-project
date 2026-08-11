# Final Report Skeleton – fill the sections as results become available

**Title:** Is Anyone Down There Moving? Pose-Based Human State Recognition for
Search-and-Rescue Drones
**Course:** EECS 4422 Computer Vision, Summer 2026, York University
**Team:** Natali Kobrii, ⟨teammate⟩

## 1. Introduction and Motivation (about 0.75 pages)
- Describe the search-and-rescue context: drones sweep an area faster than
  ground teams, and the operator is the bottleneck.
- Make the triage argument: detection alone gives a dot map, while states
  give a priority list (use the "3 motionless · 5 waving · 192 walking"
  framing from the proposal).
- State the two research questions verbatim from the proposal, then the two
  added ones:
  - **RQ3 (restoration):** frequency-domain deblurring and denoising of
    aerial frames, evaluated on PSNR, SSIM and runtime, and on the recovered
    detection AP and pose PCK.
  - **RQ4 (matching and mapping):** SIFT against ORB against a learned
    descriptor under viewpoint, scale and illumination change, and a
    homography registration that turns the per-frame triage into one victim
    map.
- Add one sentence on why RQ3 and RQ4 follow from the pipeline: drones shake
  and fly in low light (RQ3), and operators need a map rather than per-frame
  pixels (RQ4).

## 2. Related Work (about half a page)
- Aerial person detection (the VisDrone line of work), top-down pose
  estimation (RTMPose and ViTPose), skeleton-based action recognition (the
  ST-GCN family), and search-and-rescue datasets (SARD, Okutama).
- Classical restoration: Wiener filtering and Richardson-Lucy deconvolution;
  the GoPro benchmark for learned deblurring (we study the classical regime
  with a known point spread function).
- Local features: SIFT, ORB, the HPatches benchmark, and learned patch
  descriptors (the L2-Net and HardNet family; TinyDescNet is a scaled-down
  model in the HardNet style).

## 3. Data (about 0.75 pages)
- A table of the datasets used, with their sizes. Include the verified
  finding that Okutama has no Waving class, so the taxonomy is motionless,
  stationary and mobile (a limitation and future work).
- The RQ3 degradation protocol: synthetic blur and noise with known ground
  truth on VisDrone frames (the professor's specification explicitly allows
  this); list the six conditions from `src/eval_restore.py::conditions()`.
- RQ4: the HPatches sequences (the viewpoint and illumination subsets) and
  the synthetic-warp local benchmark (controlled viewpoint, scale and
  illumination changes; no image overlap with the descriptor-training
  frames).
- The table that maps actions to states, with the precedence rule (from
  `src/config.py`).
- The state distribution table with real numbers from the parser (for
  example, the sample video holds 8,433 stationary, 5,559 mobile and 1,337
  motionless boxes).

## 4. Method (about 1.5 pages)
- The pipeline figure: detection, tracking, pose, features, classification
  and triage.
- Detection and tracking: YOLO11 with ByteTrack (cite both), and the
  fine-tuning setup.
- Pose: RTMPose applied top-down on the tracked boxes, and why the top-down
  approach suits aerial scales.
- Features: the normalized keypoints, the body-axis angle, and the temporal
  wrist and center motion (list them from `src/features.py`); the window
  size is W = 15.
- Classifiers: the PoseMLP against the AppearanceCNN; emphasize that both
  use identical splits and the identical training loop.
- Restoration (RQ3): the degradation model g = h * f + n; the inverse
  filter against the Wiener filter (derive the K term as regularization)
  against Richardson-Lucy; the spatial baselines; why circular convolution
  isolates the method behavior from the boundary handling.
- Matching and registration (RQ4): the descriptor comparison with a fixed
  detector (SIFT keypoints for both the SIFT descriptors and TinyDescNet);
  the ratio test; the RANSAC homography; the chained homographies that form
  the mosaic; the foot-point projection that produces the victim map (the
  planar-scene assumption).
- Metrics: AP50 (our own implementation), PCK@0.1 normalized by the box
  size and stratified by pixel height (justify the box-size normalization
  at aerial scale), macro-F1, PSNR and SSIM (our own implementations), MMA
  at several thresholds, recall at 3 pixels, and the homography corner
  error.

## 5. Experiments and Results
### 5.1 Detection foundation
- The before-and-after table from results/tables/detection_baseline.csv
  (local, small-scale) together with the full numbers from notebook 01.
  The resolution ablation compares image sizes 640 and 1280.
### 5.2 RQ1 – the aerial domain gap for pose
- The ground-level reference PCK (the COCO sanity check,
  results/tables/pck_coco_sanity.json).
- The aerial zero-shot PCK across the height bins (UAV-Human), which is the
  main domain-gap figure (rq1_pck_vs_scale.png), and the fine-tuned
  recovery (the pseudo-label student, the UAV-Human fine-tune, or both).
### 5.3 RQ2 – pose against appearance
- results/tables/rq2_local.csv (small-scale) and rq2_full.csv (Colab, with
  the video-level split).
- The confusion matrices (results/figures/confusion_*.png), with a
  per-class discussion: on which states pose outperforms the appearance
  baseline, and why (the body-axis angle feature captures lying).
- The ablations: the window size, the crop resolution, and the features
  with and without the temporal part.
### 5.4 RQ3 – restoration
- The intrinsic table (results/tables/restoration_intrinsic.csv) with PSNR,
  SSIM and runtime per condition and method; the inverse-filter failure
  illustrates noise amplification.
- The Wiener K sweep and the PSF-mismatch figure (restoration_wiener_k.png)
  as the ablation.
- The extrinsic tables (restoration_extrinsic_detection.csv and
  restoration_extrinsic_pose.csv): does restoration recover AP50 and PCK?
  Discuss when it does not (the detector was trained on clean data; the
  median filter on salt-and-pepper noise is the case where the
  frequency-domain method fails).
### 5.5 RQ4 – matching and the victim map
- The matching table (matching_local.csv and the notebook-06 HPatches
  numbers): MMA at several thresholds, recall, the homography corner error,
  and the runtime, per condition.
- The precision-recall figure across the ratio threshold, and the MMA
  curves figure.
- The SIFT, ORB and TinyDescNet discussion: where the learned descriptor
  performs better or worse (the Easy, Hard and Tough splits), and the
  trade-off between speed and accuracy.
- The applied result: the registration mosaic and the victim map figures
  (registration_mosaic_*.png and victim_map_*.png), with the registration
  statistics (the mean number of RANSAC inliers per link).
### 5.6 Qualitative and failure analysis
- Six to eight frames from the demo, and 10 to 20 misclassified crops with
  categorized causes (small person, occlusion, pose collapse, ambiguous
  ground truth).
- The RQ3 and RQ4 failure examples: an inverse-filter noise-amplification
  crop, and a mosaic drift example (the chained homography error grows with
  the sweep length).

## 6. Demo
- One paragraph and a link or a frame of the final demo in results/videos/
  (held-out footage, fine-tuned weights, and the triage overlay).

## 7. Limitations and Future Work
- There is no Waving class (UAV-Gesture is the remedy); there is no aerial
  3D ground truth (the lifting is qualitative only); geolocation needs
  camera metadata (the demo assumes flat ground); the single-video
  small-scale results stand against the full-set Colab results.
- RQ3: the setting is non-blind (the point spread function is known); the
  degradation model uses circular convolution (no realistic boundary
  effects); blind or cepstral PSF estimation and learned deblurring are
  future work (GoPro Part D shows the gap).
- RQ4: the planar-scene homography assumption; chained registration drifts
  over long sweeps (loop closure or bundle adjustment as future work); the
  victim map is in mosaic coordinates rather than geographic coordinates
  (this would need the camera GPS and intrinsics).

## 8. Contributions statement
- State who did what; the course requires this section.

## Appendix
- Reproducibility: the exact commands (in the README), the seeds, the
  hardware, and the runtimes.
