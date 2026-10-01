# Problem Statement

## Objective: Online multi-object tracking (MOT) of stationary objects from monocular RGB video, using a moving camera.

Given a monocular RGB video $\{I_1, I_2, \dots, I_T\}$ and, at each frame $t$, a set of detections $D_t = \{d_t^1, \dots, d_t^{n_t}\}$ (bounding boxes, with appearance features), produce a set of tracks

$$\mathcal{T} = \{\tau_1, \dots, \tau_K\}, \qquad \tau_k = \{(t, d_t^{i}) : \text{detection } d_t^i \text{ assigned identity } k\}$$

such that each track corresponds to one physical, stationary object and identity is preserved consistently across frames, through occlusion, visual ambiguity, and camera motion — including when the camera revisits a previously seen location.

Tracks are produced using appearance cues, motion cues, and a landmark cue computed from each detection's estimated position in the map frame, $p_t^k \in \mathbb{R}^3$, where $p_t^k$ must itself be estimated online, per frame, directly from RGB.

### Constraints

- **Input modality:** monocular RGB only (no depth, no odometry).
- **Online:** both the landmark estimate $p_t^k$ and the track assignment at frame $t$ may use frames $1, \dots, t$ only — no offline per-scene optimization, no revisiting of past frames once processed.
- **Scope of objects:** stationary objects only, viewed from a moving camera.

### Evaluation

#### Axis 1 — Tracking accuracy
**Datasets:** LettuceMOT, LFSD, a mango orchard dataset.
**Metrics:** MOTA, HOTA, IDF1, ID switches.

#### Axis 2 — Re-identification robustness
**Focus case:** sequences in which the camera revisits a previously tracked, stationary object after a gap, where identity persistence is most sensitive to drift in the online landmark estimate.
**Metrics:** ID switches and IDF1 restricted to revisit events, alongside a direct analysis of landmark drift at the moment of re-encounter.

---

# Sub-task: Online Mapping

This is the core sub-problem this project addresses. It replaces LocalizeSORT's offline RLoc module with an online, feed-forward, per-frame mapping module, holding the rest of the pipeline (Detector, AppearanceEncoder, TrackAssociation, EKF) fixed. All downstream tracking accuracy in the parent [Evaluation](#evaluation) is mediated entirely through the quality of this sub-task's output, $p_t^i$.

## Objective: recover per-detection 3D landmarks, causally, from monocular RGB alone

Given the frames observed so far, $\{I_1, \dots, I_t\}$, and the current frame's detections $D_t = \{d_t^1, \dots, d_t^{n_t}\}$, emit for each detection a **landmark observation** in a single, incrementally-built map frame $M$: a Gaussian $\mathcal{N}(p_t^i, \Sigma_t^i)$ over the object's position, $p_t^i \in \mathbb{R}^3$, $\Sigma_t^i \in \mathbb{R}^{3 \times 3}$, plus the object's spatial **extent** $S_t^i \in \mathbb{R}^{3 \times 3}$.

$\Sigma_t^i$ and $S_t^i$ are distinct quantities and must not be conflated. $S_t^i$ is the object's physical spread in the map frame — a property of the lettuce, fixed regardless of how well it is observed. $\Sigma_t^i$ is the uncertainty about where its *centre* lies, which shrinks as more evidence accumulates (for a backend pooling $n$ points over the bbox, roughly $S_t^i / n$). Only $\Sigma_t^i$ may be fused across observations; fusing $S_t^i$ would claim precision finer than the object itself.

A detection may yield **no landmark** — the bbox encloses no usable geometry. This is a first-class outcome, not a sentinel value: such detections carry no position at all and fall back to appearance-only association downstream.

The module may internally recover a per-frame camera pose and per-pixel scene geometry as the mechanism for producing $p_t^i$ (e.g. pooling a predicted per-pixel pointmap over $d_t^i$'s bounding box) — but the camera pose itself is not a deliverable of this sub-task and is not evaluated; only $p_t^i$, $\Sigma_t^i$ and $S_t^i$ are.

This directly replaces RLoc: where RLoc solves for $\{p^i\}$ once, offline, via SfM over the complete sequence $\{I_1,\dots,I_T\}$, this sub-task must produce the landmark observation online, at frame $t$, before $I_{t+1},\dots,I_T$ are observed.

**Offline reference backend.** The same Gaussian is extracted from a COLMAP reconstruction as pseudo-GT (`scripts/landmarks/sfm_mapping.py`): the triangulated keypoints inside a detection's bbox, with depth outliers rejected, give $p_t^i$ as their median and $S_t^i$ as their scatter. LocalizeSORT's original RLoc instead took the single keypoint nearest the bbox centre and emitted its 3D point, with no uncertainty — retained as a baseline mode for reproducing its numbers.

### Constraints

- **Input modality:** monocular RGB only — no depth, no odometry, no known intrinsics/extrinsics beyond what the model infers from the images.
- **Causality:** the estimate at frame $t$ ($p_t^i$, $\Sigma_t^i$, $S_t^i$, and any internal camera-pose/geometry state the module maintains) depends on $I_1,\dots,I_t$ only. No access to future frames, and no revisiting or refining of past frames' estimates once emitted — no global bundle adjustment, no retroactive loop-closure correction, no windowed test-time optimization.
- **Feed-forward:** camera pose and scene geometry come from a single forward pass of a learned model per frame, not an iterative offline solver.
- **Map-frame consistency:** every $p_t^i$ across the sequence must sit in the same map frame $M$ at a consistent scale — established once, at the start of the sequence, and never reset or re-anchored mid-sequence (e.g. by a persistent-state reset in a recurrent backbone) — since TrackAssociation and the EKF compare positions across frames against a persistent Kalman state, with no global re-optimization available to correct scale drift or a frame discontinuity after the fact.
- **Runtime:** must run online, per frame — Mapping must not reintroduce the whole-sequence latency it is replacing RLoc to remove.

### Evaluation

The sub-task's output is not directly observable as ground truth (no depth/odometry sensor), so reference values are taken from the existing offline SfM reconstructions (RLoc) on the sequences where they are available — the same LettuceMOT, LFSD, and mango orchard sequences as the parent [Evaluation](#evaluation). Offline SfM is treated as pseudo-ground-truth: a whole-sequence-optimized, drift-corrected estimate, not a true sensor measurement.

**Alignment.** The predicted map frame $M$ and the GT/offline map frame $M^*$ are related by an unknown similarity transform $S \in \mathrm{Sim}(3)$ (rotation, translation, and — since monocular RGB gives no metric scale reference — a single global scale). $S$ is computed once per sequence, entirely on the evaluation side, and is never available to (or used by) the online module itself. Both the online module and RLoc emit a full camera trajectory as a byproduct of reconstruction ($T_t^{pred}$ and $T_t^{GT}$ respectively) even though camera pose itself isn't scored — so $S$ is obtained in closed form via Umeyama alignment over the corresponding trajectory pair $\{(T_t^{pred}, T_t^{GT})\}_{t=1}^{T}$, the same procedure standard SLAM benchmarks use to align an estimated trajectory to ground truth (e.g. `evo`). Fitting on the full trajectory, rather than on a single frame's pose or on the sparse per-object correspondences, avoids two failure modes: pinning the whole sequence's alignment to one frame's estimation noise, and overfitting a 7-DOF transform to the same handful of points it is later scored on. This alignment is a post-hoc evaluation device only — it does not violate the causality constraint above, since it is computed after the sequence is processed and plays no role in the online system itself.

#### Axis 1 — Object-position accuracy
**What:** how closely each online per-detection position, after alignment, $S(p_t^i)$, matches its offline-SfM counterpart, at first observation and pooled over the sequence.
**Metrics:** mean/median Euclidean error between $S(p_t^i)$ and the GT reference position, broken down by time since sequence start, to characterize drift growth.

#### Axis 2 — Revisit drift
**What:** for a stationary object observed online at two well-separated times $t_1 \ll t_2$, the discrepancy between $p_{t_1}^i$ and $p_{t_2}^i$ is the quantity most directly responsible for ID switches in the parent problem's re-identification axis, since TrackAssociation and the EKF assume a stationary object's map-frame position is stable over time. This comparison is intrinsic to $M$ — a similarity transform scales distances uniformly, so no alignment to $M^*$ is needed to compare $p_{t_1}^i$ and $p_{t_2}^i$ to each other; $S$'s fitted scale is used only to report the drift in physical units.
**Metrics:** $\lVert p_{t_1}^i - p_{t_2}^i \rVert$ (scaled to physical units via $S$) at matched revisit pairs (identified via offline SfM correspondence), and calibration of $\Sigma_t^i$ against this empirical error (e.g. Mahalanobis distance / NEES) — a miscalibrated $\Sigma_t^i$ degrades the downstream association cost regardless of the point estimate's accuracy. Calibration matters more here than in a hand-tuned distance gate: association thresholds the Mahalanobis distance, which is scale-free only if $\Sigma_t^i$ is honest, so an over-confident $\Sigma_t^i$ silently tightens the gate.

Axes 1–2 evaluate this sub-task in isolation; the parent problem's Axes 1–2 evaluate it in situ, as the delta in MOTA/HOTA/IDF1/ID-switches when substituting this module's $p_t^i$ for RLoc's, with every other pipeline component held fixed.