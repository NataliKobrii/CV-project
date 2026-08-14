# Video Script and Recording Instructions

**Project:** Pose-Based Human State Recognition for Search-and-Rescue Drones
**Team:** Nataliia Kobrii, Daniel Vinitski, Nityam Goyal
**Target length:** fifteen minutes (the course limit is thirty minutes).

---

## Recording instructions

1. **All three members must appear on camera.** At the start (Slide 1), each
   member shows their face and clearly states their full name and role. A webcam
   inset in the corner of the shared slides is sufficient for the remaining
   slides.
2. **Screen and voice.** Share the slide deck full-screen and narrate. When a
   figure or the demonstration video is referenced, display it on screen.
3. **Speaker order.** Follow the assignments below. Each speaker states their own
   contributions when their section begins and again on the contributions slide.
4. **Demonstration.** On Slide 13, play `results/videos/demo_final.mp4` for about
   forty-five to sixty seconds; narrate what the overlay shows.
5. **Status.** Slide 14 explicitly states what is fully functional, what is
   partially functional, the known limitations, and the assumptions, as required.
6. **Accessibility.** Upload the recording to YouTube as **Unlisted** (not
   Private), and confirm the repository and any cloud links are accessible to the
   instructors and teaching assistants before the deadline.
7. **Pace.** The narration below is written for roughly one hundred and forty
   words per minute. Rehearse once to confirm the total stays near fifteen
   minutes.

---

## Timed script

### Slide 1 — Title (all members) — 0:00–0:40
*(Each member, in turn, on camera.)*
"Good afternoon. This is our EECS 4422 project, Pose-Based Human State
Recognition for Search-and-Rescue Drones. I am Nataliia Kobrii." — "I am Daniel
Vinitski." — "I am Nityam Goyal." "We will present the motivation, the design,
the implementation, our four experiments, the demonstration, and our
conclusions."

### Slide 2 — Motivation (Nataliia) — 0:40–1:40
"In a search-and-rescue operation, a drone can sweep a disaster area far faster
than a ground team. However, the raw aerial video still requires a human operator
to locate people and to judge their condition, so the operator becomes the
bottleneck. Detection alone produces only a map of dots. The information an
operator needs first is a triage priority list: who is lying motionless and who
is merely standing. That judgement — distinguishing a motionless person from an
active one — is fundamentally a human-pose problem, and that observation
motivates the entire pipeline."

### Slide 3 — Research questions (Nataliia) — 1:40–2:40
"We framed the project as four controlled questions. The first asks how much a
pose estimator trained on ground-level images degrades on aerial imagery, and how
much of that loss fine-tuning can recover. The second asks whether an explicit
pose representation outperforms an appearance-only baseline for classifying a
person's state. The third asks whether classical, frequency-domain restoration of
degraded frames recovers not only pixel quality but also downstream task accuracy.
The fourth compares local feature descriptors and builds a single victim map by
registration."

### Slide 4 — Pipeline architecture (Nataliia) — 2:40–3:40
"The pipeline has six stages: detection, tracking, pose estimation, feature
extraction, classification, and the triage overlay. We use a YOLO11s detector,
the ByteTrack tracker, and RTMPose applied top-down to each tracked box. From the
keypoints we build a forty-seven-dimensional feature vector per track window,
combining normalised keypoints, geometric features such as the body-axis angle,
and temporal motion features. We then compare two classifiers, a pose-based
multilayer perceptron and an appearance convolutional network, under identical
splits so that the comparison isolates the representation. I will now hand over to
Daniel for the data and the first two questions."

### Slide 5 — Data and taxonomy (Daniel) — 3:40–4:30
"We use five datasets: VisDrone for detection, COCO and UAV-Human for pose,
Okutama-Action for state classification and the demonstration, and HPatches for
matching. We verified that Okutama contains no Waving class, so our taxonomy has
three search-and-rescue-relevant states: motionless, stationary, and mobile. When
a box carries several actions, a precedence rule selects the most urgent, placing
motionless first. For restoration we apply synthetic degradations with known
ground truth, which lets us measure both image quality and task performance."

### Slide 6 — Detection foundation (Daniel) — 4:30–5:30
"Detection is the foundation, so we first quantified the aerial domain gap using
our own average-precision evaluator on five hundred and forty-eight validation
images. The zero-shot COCO model reaches an AP at fifty of only 0.437, because it
was trained on ground-level people. After fine-tuning for thirty epochs, the AP
rises to 0.709 and recall to 0.859. The training curves on this slide show stable
convergence. This confirms that the aerial gap is substantial and that a modest
fine-tune closes much of it."

### Slide 7 — RQ1: the aerial pose gap (Daniel) — 5:30–6:50
"For the first research question we evaluated pose accuracy with the percentage of
correct keypoints. On ground-level images the score is 0.95, which validates the
estimator and our evaluator. We then ran the same measurement over more than
twenty-two thousand aerial persons. The overall aerial score is within one point
of ground level, which is deceptive, because when we stratify by person height
the accuracy collapses: it remains high above one hundred pixels, is halved in the
fifty-to-one-hundred-pixel band, and falls to 0.09 below fifty pixels. The figure
makes this clear. The central finding is that the aerial pose gap is a
small-person problem, not a viewpoint problem."

### Slide 8 — RQ1: recovery (Daniel) — 6:50–7:30
"We also tested a label-free recovery. We used a strong teacher model to
pseudo-label Okutama frames and fine-tuned a compact student pose model on those
labels. The student reaches a pose mean average precision of 0.508. A full
before-and-after quantification of the recovery is one of our future-work items.
I will pass to Nityam for the state-classification results."

### Slide 9 — RQ2: pose against appearance (Nityam) — 7:30–8:50
"The second research question compares the pose-based classifier with the
appearance baseline. We held out whole videos, which is a strict test of
generalisation across scenes. The pose model reaches a macro-F1 of 0.363 and the
appearance model 0.399, so the appearance model leads only narrowly. The important
observation is in the confusion matrices: both models fail entirely on the rare
motionless class, which appears as a zero row. This is exactly the small-person
regime that the first research question identified, because a person lying down at
low resolution produces unreliable keypoints. The result is honest and
informative: at this scale, explicit pose does not yet beat appearance."

### Slide 10 — RQ3: restoration (Nataliia) — 8:50–10:10
"The third research question studies restoration. We degrade frames with six known
conditions and compare frequency-domain methods with spatial filters. On image
quality, the Wiener filter matches Richardson–Lucy deconvolution on blur at a
fraction of the cost, and spatial filters are best on noise. The central result,
however, is that pixel quality does not equal task utility. On additive noise,
non-local means achieves a high peak signal-to-noise ratio, yet it reduces
detection accuracy below the degraded baseline by smoothing away small people. The
simple median filter is the only method that improves detection. The lesson is
that a restoration method must be validated on the downstream task, not on a pixel
metric alone."

### Slide 11 — RQ4: descriptor comparison (Nityam) — 10:10–11:10
"The fourth research question compares local feature descriptors on all one
hundred and sixteen HPatches sequences. SIFT is the most accurate and
geometrically precise. ORB is three to five times faster but weaker at strict
thresholds, especially under viewpoint change. Our small self-supervised
descriptor, TinyDescNet, approaches SIFT under illumination change but is weaker
under viewpoint change, which is consistent with its training on appearance
variation rather than geometric warps."

### Slide 12 — RQ4: the victim map (Nityam) — 11:10–11:55
"The applied result is a victim map. Using SIFT and RANSAC, we register every
twelfth frame of a two-hundred-and-eighty-eight-frame sweep, with an average of
about nine hundred inliers per link. We chain the homographies to a common frame,
build a mosaic, and project the foot point of every tracked person, coloured by
state. The per-frame triage overlay becomes a single operator-facing map. I hand
back to Nataliia for the demonstration."

### Slide 13 — Demonstration (Nataliia) — 11:55–13:05
"Here is the full pipeline running on eighteen hundred frames of held-out Okutama
footage." *(Play the demo.)* "Each detected person is tracked, their pose is
estimated, and their state is classified. The overlay colours each person by
urgency, so an operator sees at a glance who is motionless and who is moving. This
is the end-to-end system that all four experiments analyse."

### Slide 14 — Status (Nityam) — 13:05–14:05
"To be explicit about status. Fully functional: the end-to-end pipeline,
the detection fine-tuning, the full-scale evaluations for all four research
questions, and this demonstration."Partially functional: the pose
recovery is trained but not fully quantified, and the classification ablations are
not run at full scale; registration is demonstrated on one video."Our
assumptions are a three-state taxonomy, a box-normalised pose metric, a known
blur function for restoration, and a planar scene for registration."

### Slide 15 — Lessons and conclusions (Nataliia) — 14:05–14:40
"Three lessons stand out. The aerial gap is driven by person size, and small
people break both pose and detection. Restoration must be judged on the task, not
on pixel quality. Classical descriptors remain strong baselines, and a small
learned descriptor is competitive only within its training regime. Together, the
project gives a realistic account of where an aerial triage pipeline succeeds and
where it fails."

### Slide 16 — Contributions and closing (all) — 14:40–15:00
Daniel: "I led the data preparation, the detector fine-tuning, and the aerial
pose-gap study." Nataliia: "I led the state-classification comparison and the
matching, registration, and victim-map study." Nityam: "I led the pipeline
integration, the restoration study, the demonstration, and the report." All:
"Thank you. The repository, the report, and the demonstration are linked in our
submission."
