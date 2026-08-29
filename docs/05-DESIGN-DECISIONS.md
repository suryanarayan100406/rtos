# DRISHTI — Design Decisions

**Purpose:** The *why* behind DRISHTI. Every consequential architectural choice recorded as an Architecture Decision Record (ADR): the context that forced it, the options considered, what we chose, the single-pass rationale, the trade-offs we accepted, and the condition that would make us revisit. This is the document that shows our reasoning is deliberate, not accidental.

**Audience:** National Technical Research Organisation (NTRO) technical evaluators and the DRISHTI build team. Assumes familiarity with [How It Works](02-HOW-IT-WORKS.md) and the [Technology Stack](03-TECHNOLOGY-STACK.md); the pipeline is defined once in [`_internal/CANONICAL-ARCHITECTURE-SPEC.md`](_internal/CANONICAL-ARCHITECTURE-SPEC.md).

**TL;DR**

- Fifteen decisions define DRISHTI. The through-line: **the single-pass, no-second-look constraint changes the right answer** at nearly every step versus a conventional multi-pass photogrammetry pipeline.
- The biggest bet (ADR-01): **prior-assisted feed-forward geometry** over classical Structure-from-Motion (SfM) as the primary path — because short single-pass baselines starve triangulation — with classical SfM kept as the verifiable fallback, not discarded.
- The metric bet (ADR-03/04): **sensor-injected scale from a tightly-coupled GNSS+IMU+visual factor graph**, not learned-only scale and not Ground Control Points (GCPs) — the only way to be metric *and* GCP-free in one pass.
- The reliability bet (ADR-09): **confidence is a first-class output at every stage**, and every stage has a fallback — so the system degrades gracefully instead of hard-failing.
- The honesty bet, present in every ADR's trade-off section: we name what we *lose* (absolute accuracy on planned GCP missions, facade completeness, mesh polish) versus offline incumbents, and why the single-pass niche is still worth owning.
- Trade-offs and external numbers follow the project honesty policy: performance figures attributed to a method are **as reported by its authors**; our targets are labelled *(design target — to be measured)*.

---

## 1. How decisions are recorded

Each decision is an **ADR** with a fixed shape, so an evaluator can audit our reasoning uniformly:

| Field | What it captures |
|-------|------------------|
| **Context** | The forcing constraint — usually a consequence of single-pass capture. |
| **Options** | The real alternatives we weighed (not strawmen). |
| **Decision** | What we chose. |
| **Rationale (single-pass angle)** | Why this choice specifically serves the single-pass, no-second-look regime. |
| **Trade-offs & risks** | What we give up, honestly, and what could bite us. |
| **Revisit if** | The concrete signal that would reopen the decision. |

ADRs are immutable once accepted; a superseding decision gets a new number and references the old one. This mirrors how the [Technology Stack](03-TECHNOLOGY-STACK.md) treats every model as replaceable — decisions, like models, are versioned rather than silently overwritten.

---

## 2. Decision index

| ADR | Decision | Primary driver |
|-----|----------|----------------|
| 01 | Prior-assisted feed-forward geometry as primary; classical SfM/MVS as fallback | Short single-pass baselines starve triangulation |
| 02 | Feed-forward backbone with a mandatory classical fallback | Bleeding-edge models are OOD on aerial data |
| 03 | Sensor-injected metric scale, not learned-only, not GCPs | Metric *and* GCP-free in one pass |
| 04 | Tightly-coupled factor graph over loose EKF fusion | Robust scale/georef from noisy single-pass sensors |
| 05 | Two output paths (Live S0–S5 · Refine S6–S10), not one monolith | "Real-time" and "accurate" are different jobs |
| 06 | Few-shot 3D Gaussian Splatting over NeRF / classical MVS for dense fill | Sparse single-pass views, fast textured output |
| 07 | Surface-aware meshing (2DGS/SuGaR) with Poisson fallback | Measurable surface from a Gaussian field |
| 08 | Belt-and-suspenders dynamic masking | An unmasked mover is a permanent ghost |
| 09 | Confidence as a first-class, propagated output | No second look to catch errors |
| 10 | ROS 2 + Isaac ROS + DeepStream over a custom runtime | Determinism, reuse, tri-tier data plane |
| 11 | TensorRT INT8/FP16 on the edge over FP32 | UAV power/thermal budget |
| 12 | Store-and-forward + deterministic ground reprocess | Link loss must lose nothing; auditability |
| 13 | nvblox (edge) + Open3D (ground) for fusion | Right fusion engine per tier |
| 14 | Dual-track hardware: DJI COTS + PX4 open/sovereign | De-risk procurement & sovereignty |
| 15 | Explicit CRS/geoid handling (UTM + EGM2008) | Orthometric height, not ellipsoidal |

---

## 3. The decision records

### ADR-01 — Prior-assisted feed-forward geometry as the primary reconstruction path

- **Context.** A single pass produces short baselines and narrow view-angle diversity. Classical SfM/Multi-View Stereo (MVS) needs wide, overlapping, multi-angle views to triangulate well; starve it and it produces holes, drift, and broken geometry — exactly where single-pass capture is weakest.
- **Options.** (a) Classical SfM/MVS (COLMAP + dense MVS) as primary; (b) feed-forward geometry transformers (predict pose + dense 3D in one forward pass) as primary; (c) NeRF/3DGS optimized directly from images.
- **Decision.** Feed-forward geometry (option b) is the **primary** geometry engine; classical SfM (GLOMAP/COLMAP) is retained as a **verifiable fallback and accuracy oracle**, not thrown away.
- **Rationale (single-pass angle).** Feed-forward models regress geometry from limited-angle input in ~seconds, tolerating the short baselines that break triangulation. They give us a *dense first estimate everywhere*, which we then constrain with sensor priors and refine — rather than failing to converge.
- **Trade-offs & risks.** Feed-forward models are newer, less battle-tested, and largely trained off-distribution from aerial imagery (see ADR-02). Classical SfM, when it *does* have the views, is more accurate and fully interpretable — so we keep it as the oracle and the fallback, and use it to validate feed-forward output on the event dataset.
- **Revisit if.** The provided dataset turns out to have rich multi-pass overlap (then classical SfM becomes primary), or feed-forward metric accuracy on aerial data fails to reach the [accuracy budget](_internal/CANONICAL-ARCHITECTURE-SPEC.md) after fine-tuning.

### ADR-02 — Feed-forward backbone with a *mandatory* classical fallback

- **Context.** The 2024–2026 geometry SOTA (VGGT, Depth Anything 3, MapAnything, Pi3) is months old and validated mostly on object-centric, indoor, or driving data — not high-altitude aerial. A backbone that silently degrades OOD is dangerous in a no-second-look setting.
- **Options.** (a) Commit to one backbone; (b) backbone + classical fallback triggered by confidence; (c) permanent ensemble of two backbones.
- **Decision.** One primary backbone (permissive default: **Depth Anything 3 / MapAnything**, per [Technology Stack §5](03-TECHNOLOGY-STACK.md)) **plus a confidence-gated classical fallback** (GLOMAP/COLMAP). Not a permanent ensemble — that doubles cost for marginal gain most of the time.
- **Rationale (single-pass angle).** When the backbone's confidence collapses (OOD terrain, degenerate motion), we fall to classical geometry rather than emitting a confident-but-wrong model. This is reliability ladder rung **L5** (neural failure → classical path) from the spec.
- **Trade-offs & risks.** Maintaining two geometry engines is more integration work and more to calibrate. The fallback is slower; a mission that leans on it heavily loses the near-real-time preview quality. We accept that: a slow correct answer beats a fast wrong one.
- **Revisit if.** Independent aerial benchmarks show the backbone is robust enough to drop the fallback, or a single model subsumes both roles with confidence.

### ADR-03 — Sensor-injected metric scale, not learned-only scale, not GCPs

- **Context.** The deliverable is a **metric, georeferenced** model **without** placing GCPs (which would defeat rapid single-pass survey). Monocular vision is scale-ambiguous; learned metric depth is impressive but not survey-grade on its own; GCPs are accurate but operationally expensive and slow.
- **Options.** (a) GCPs (classical photogrammetry); (b) learned metric depth alone for scale; (c) fuse onboard GNSS(+RTK/PPK)+IMU to inject metric scale and georeference, using learned depth only as a *relative* prior.
- **Decision.** Option (c): **the sensor suite sets absolute scale and georeference**; learned depth contributes relative structure and gets scale-aligned to the sensor-anchored trajectory (7-DoF Umeyama/Sim3 alignment + GNSS-prior bundle adjustment). This is the **metric spine (anchor A2)**.
- **Rationale (single-pass angle).** RTK/PPK GNSS gives centimetre-class positions along the flight path with zero ground setup — the one metric anchor available in a single autonomous pass. Learned depth alone would leave us with plausible-but-unverifiable scale.
- **Trade-offs & risks.** Accuracy becomes **sensor-configuration-dependent**: centimetre-class with RTK/PPK (Regime A), sub-metre to metre-class GPS-only (Regime B). Vertical accuracy is the weak axis. We state this explicitly rather than claiming a single accuracy number — per the honesty policy and the [Evaluation Criteria](_internal/PROBLEM_STATEMENT.md) two-regime table.
- **Revisit if.** A future learned model demonstrates survey-grade metric accuracy from vision alone (then sensors become a cross-check, not the anchor), or a mission profile permits sparse GCPs for a hybrid boost.

### ADR-04 — Tightly-coupled factor graph over loose EKF fusion

- **Context.** We must fuse GNSS, IMU, visual odometry, RTK corrections, and barometer into one trajectory. A loosely-coupled Extended Kalman Filter (EKF) fuses *post-hoc estimates*; a tightly-coupled factor graph fuses *raw measurements* jointly.
- **Options.** (a) Loosely-coupled EKF (simpler, common in flight controllers); (b) tightly-coupled factor-graph smoothing (GTSAM/iSAM2) fusing raw factors with global bundle adjustment on the ground.
- **Decision.** **Tightly-coupled factor graph** (GTSAM iSAM2), incrementally on the edge and re-optimized globally on the ground.
- **Rationale (single-pass angle).** With only one pass and no loop closures from revisiting, every measurement counts. Jointly optimizing raw GNSS/IMU/visual factors extracts maximum consistency and yields per-estimate covariance — which feeds the confidence spine (ADR-09). A loose EKF discards cross-correlations we cannot afford to lose.
- **Trade-offs & risks.** More complex, heavier to implement and tune than an EKF; real-time incremental smoothing on the edge must be carefully bounded. Mitigation: incremental iSAM2 on-edge for the Live path, full batch re-optimization on the Ground path.
- **Revisit if.** Edge compute cannot sustain incremental smoothing within budget (fall back to a tighter sliding-window filter on-edge while keeping the factor graph on the ground).

### ADR-05 — Two output paths, not one monolithic pipeline

- **Context.** "Do everything in real time" and "produce a survey-accurate textured model" are in direct tension: the full-fidelity model cannot be computed in hard real time on a drone.
- **Options.** (a) One pipeline tuned for speed (sacrifice fidelity); (b) one tuned for accuracy (sacrifice liveness); (c) two coordinated paths sharing state.
- **Decision.** **Two paths** (anchor A3): a **Live path (S0–S5)** on the edge for a near-real-time coarse preview + coverage/quality feedback in flight, and a **Refine path (S6–S10)** on the ground for the accurate, textured, georeferenced deliverables — sharing the same recorded stream and metric spine.
- **Rationale (single-pass angle).** The Live path's real job is to *guarantee the single pass was good enough* — coverage gaps, blur, exposure, GNSS dropouts surfaced *while still flyable* (even if the plan permits only opportunistic re-observation). The Refine path then spends minutes, not milliseconds, to hit accuracy. This is the honest reading of "real-time": near-real-time preview + minutes-scale refinement, never a full 4K textured mesh in hard real time on the drone.
- **Trade-offs & risks.** Two paths mean two codepaths to keep consistent; the preview is deliberately coarser than the final model, which must be communicated so no one mistakes the live view for the deliverable.
- **Revisit if.** Edge compute grows enough to move Refine-path stages forward, or a mission needs a single deferred batch product only (then the Live path becomes optional QA).

### ADR-06 — Few-shot 3D Gaussian Splatting for dense reconstruction

- **Context.** After sparse/metric geometry we need dense, textured surfaces. Single-pass gives sparse, low-overlap views — hard for classical dense MVS and slow for vanilla NeRF.
- **Options.** (a) Classical dense MVS (PatchMatch); (b) NeRF; (c) 3D Gaussian Splatting (3DGS) with few-shot regularization and feed-forward initialization.
- **Decision.** **Few-shot 3DGS** (permissive **gsplat**, InstantSplat-style init from feed-forward geometry, with depth/normal/confidence regularization), with classical dense MVS available where views suffice.
- **Rationale (single-pass angle).** 3DGS trains and renders far faster than NeRF and, initialized from feed-forward geometry plus depth priors, fills plausible texture from sparse views. Regularization fights the under-constraint that sparse single-pass coverage causes.
- **Trade-offs & risks.** Gaussian fields are not natively a measurable surface (addressed in ADR-07) and can hallucinate in unobserved regions — so splatted fill in unseen areas is flagged as **inferred, not measured** (honesty policy). NeRF may still win on certain view-dependent effects; we accept the speed/robustness trade.
- **Revisit if.** A feed-forward splat model (NoPoSplat/AnySplat class) matures enough to replace the optimization step, or dense MVS proves sufficient on the event data.

### ADR-07 — Surface-aware meshing (2DGS/SuGaR) with Poisson fallback

- **Context.** Deliverables include a **measurable textured mesh** and point cloud. A raw Gaussian field is not directly a watertight, metric surface.
- **Options.** (a) Screened Poisson on the fused point cloud only; (b) surface-aligned Gaussians (2DGS/SuGaR) to extract a view-consistent mesh; (c) marching cubes on the TSDF only.
- **Decision.** **Surface-aware extraction (2DGS/SuGaR)** as primary, **screened Poisson on the metric point cloud as the robust fallback** when the Gaussian field is under-constrained.
- **Rationale (single-pass angle).** Surface-aligned Gaussians tie the fast 3DGS representation back to a measurable surface, preserving texture fidelity; Poisson guarantees a watertight mesh even when splatting is unreliable — reliability over polish.
- **Trade-offs & risks.** Surface extraction from Gaussians is research-stage and must be reimplemented on a permissive base (see [Technology Stack §5](03-TECHNOLOGY-STACK.md)); Poisson can over-smooth and close real gaps, so filled regions are marked inferred.
- **Revisit if.** A robust, permissive Gaussian-to-mesh method matures, or evaluation shows the TSDF mesh alone meets the accuracy budget.

### ADR-08 — Belt-and-suspenders dynamic masking

- **Context.** Moving objects (vehicles, people, animals, swaying vegetation) violate the static-scene assumption. In multi-pass capture a mover seen once is outvoted; in **single-pass, an unmasked mover becomes a permanent ghost or smear** with no second look to correct it.
- **Options.** (a) Semantic segmentation only; (b) geometric/motion consistency only; (c) both, fused (segment likely-dynamic classes *and* detect motion via flow-vs-epipolar residual, tracked across frames).
- **Decision.** **Both cues fused** — SAM 2 + detector + ByteTrack for appearance/semantics, RAFT flow vs epipolar geometry for motion — because either alone has failure modes.
- **Rationale (single-pass angle).** High recall matters more than precision here: a missed mover is unrecoverable, while over-masking merely costs some coverage that the confidence map will flag. Two independent cues raise recall.
- **Trade-offs & risks.** Over-masking removes real static structure and adds compute on the edge; a static-but-movable object (parked car) is ambiguous. Mitigation: track persistence and confidence weighting rather than hard binary removal.
- **Revisit if.** A single model reliably handles both semantic and motion segmentation within the edge budget.

### ADR-09 — Confidence as a first-class, propagated output

- **Context.** The single-pass, no-second-look constraint means we cannot re-fly to catch an error; the system must instead *tell us where it is unsure*.
- **Options.** (a) Post-hoc quality heuristics on the final model; (b) confidence emitted and propagated by every stage end-to-end.
- **Decision.** **Every stage emits a confidence/uncertainty signal that propagates** into the final per-point/per-region completeness-and-uncertainty report (anchor A4, the reliability spine). No learned output enters the pipeline without a confidence wrapper (multi-view agreement, geometric residual, or model uncertainty).
- **Rationale (single-pass angle).** A georeferenced model is only actionable if the user knows which parts are trustworthy. Confidence also drives the graceful-degradation ladder (which fallback fires, when) and lets us mark inferred/occlusion-completed surfaces honestly.
- **Trade-offs & risks.** Extra computation and engineering to produce, calibrate, and carry confidence everywhere; calibration itself must be validated so the numbers mean something. We accept this as core to the value proposition, not optional.
- **Revisit if.** Never fully — this is foundational. Calibration *methods* will evolve.

### ADR-10 — ROS 2 + Isaac ROS + DeepStream over a custom runtime

- **Context.** We need a deterministic data plane spanning edge and ground, with GPU-accelerated perception and video handling.
- **Options.** (a) Bespoke async framework; (b) ROS 2 + NVIDIA Isaac ROS + DeepStream/GStreamer.
- **Decision.** **ROS 2 (Humble/Jazzy) + Isaac ROS + DeepStream.**
- **Rationale (single-pass angle).** ROS 2's Quality-of-Service (QoS) pub/sub gives deterministic, replayable dataflow (essential for the deterministic reprocess in ADR-12); Isaac ROS provides GPU-accelerated nodes (nvblox, image pipeline) already tuned for Jetson; DeepStream gives zero-copy NVDEC→inference. Reuse over reinvention, matching the adopt-don't-invent philosophy.
- **Trade-offs & risks.** ROS 2 has a learning curve and runtime overhead; some Isaac components carry NVIDIA licensing (see stack). We accept the overhead for determinism and ecosystem.
- **Revisit if.** Real-time constraints demand a leaner runtime on the edge for specific nodes (we can drop to raw CUDA nodes selectively).

### ADR-11 — TensorRT INT8/FP16 on the edge over FP32

- **Context.** The edge (Jetson Orin) has a hard power/thermal/latency budget on the UAV; full-precision models won't fit the Live-path latency target.
- **Options.** (a) FP32/ONNX portable; (b) TensorRT with INT8/FP16 quantization and calibration.
- **Decision.** **TensorRT INT8/FP16** for edge inference; keep FP32 on the ground where accuracy is paramount.
- **Rationale (single-pass angle).** Quantization is what makes the near-real-time Live preview physically possible within the UAV budget; the ground path re-runs at full precision so the *deliverable* accuracy is never quantization-limited.
- **Trade-offs & risks.** Quantization can cost accuracy; INT8 needs a representative calibration set (aerial), and accuracy-vs-latency must be measured on the event hardware. Mitigation: the authoritative product comes from the FP32 ground path, so edge quantization only affects the preview.
- **Revisit if.** Measured INT8 accuracy loss is unacceptable even for preview (fall to FP16), or edge hardware gains headroom.

### ADR-12 — Store-and-forward + deterministic ground reprocess

- **Context.** The radio link can drop; a single-pass mission cannot be repeated cheaply, so **no captured data may be lost to a link outage**, and accuracy claims must be auditable.
- **Options.** (a) Rely on the live stream only; (b) record near-raw H.265 + telemetry to onboard NVMe and forward opportunistically, then deterministically reprocess on the ground.
- **Decision.** **Store-and-forward** to onboard storage with **deterministic ground reprocess** (monotonic frame IDs, pinned versions, fixed seeds) from the recorded near-raw stream + telemetry.
- **Rationale (single-pass angle).** The recording is the ground truth of the one pass; the Live path is best-effort over the radio, but the authoritative model is rebuilt from the complete on-board record. Determinism makes the result reproducible and the accuracy report checkable. Also underpins sovereignty/air-gap (no cloud dependency).
- **Trade-offs & risks.** Onboard storage, I/O bandwidth, and post-flight transfer time; the authoritative product is deferred to after landing. Accepted: correctness and auditability outrank immediacy for the deliverable.
- **Revisit if.** Links become reliably high-bandwidth enough to stream near-raw in real time (unlikely in contested environments).

### ADR-13 — nvblox on the edge, Open3D on the ground for fusion

- **Context.** Both tiers fuse depth into a volume, but under very different compute budgets and goals (live coarse map vs scalable accurate fusion).
- **Options.** (a) One fusion engine everywhere; (b) tier-appropriate engines: nvblox on edge, Open3D on ground.
- **Decision.** **nvblox (edge)** for GPU-accelerated live TSDF within the Jetson budget; **Open3D (ground)** for scalable, accurate TSDF/point fusion.
- **Rationale (single-pass angle).** The edge needs a fast, dynamic-masked coarse map *now* for the Live preview; the ground needs scale and accuracy. Matching the engine to the tier beats compromising on one. Both are permissively licensed (Apache/MIT).
- **Trade-offs & risks.** Two fusion codepaths; representations must reconcile between tiers. Mitigation: shared metric frame and voxel conventions from the spec's coordinate-frames section.
- **Revisit if.** One engine covers both budgets, or a superior permissive fusion library appears.

### ADR-14 — Dual-track hardware: DJI COTS + PX4 open/sovereign

- **Context.** We need a platform for development and demos now, and a sovereign, customizable platform for deployment — with different procurement, licensing, and data-control profiles.
- **Options.** (a) DJI-only (fast, integrated RTK, but closed and foreign); (b) open PX4/ArduPilot-only (sovereign but slower to stand up); (c) both tracks.
- **Decision.** **Both** — DJI Matrice 350 RTK + Zenmuse as the fast COTS track for development/benchmarking; a **PX4/ArduPilot + Jetson + open GNSS (NavIC-capable)** track for the sovereign deployment.
- **Rationale (single-pass angle).** DJI lets us validate the single-pass pipeline immediately on integrated RTK hardware; the open track ensures the *deployed* system is customizable, air-gappable, and free of foreign platform dependence — the sovereignty requirement. The software is hardware-agnostic behind a sensor-abstraction layer, so both feed the same pipeline.
- **Trade-offs & risks.** Supporting two hardware profiles doubles integration/calibration surface; DJI's closed ecosystem limits low-level access. Mitigation: sensor-abstraction interface so the pipeline is platform-neutral.
- **Revisit if.** The sovereign platform matures enough to be the sole track, or mission constraints forbid DJI entirely (then open-track only).

### ADR-15 — Explicit CRS and geoid handling (UTM + EGM2008)

- **Context.** "Metric and georeferenced" requires an explicit Coordinate Reference System (CRS) and a vertical datum. GNSS gives ellipsoidal height; users and maps expect **orthometric** height (above mean sea level).
- **Options.** (a) Report raw WGS84 lat/long + ellipsoidal height; (b) project to local UTM/EPSG and apply a geoid model (EGM2008 + local geoid) for orthometric height.
- **Decision.** **Project to the appropriate UTM/EPSG zone (PROJ/GDAL) and apply EGM2008 (+ local geoid where available) for orthometric height**; carry CRS/datum metadata in every exported product.
- **Rationale (single-pass angle).** A single autonomous pass still has to drop into the user's mapping frame correctly the first time; getting CRS/geoid wrong silently corrupts every downstream measurement. Explicit handling makes the georeferenced deliverables interoperable and auditable.
- **Trade-offs & risks.** Geoid grids must be bundled for offline/air-gapped use; wrong zone selection is a footgun. Mitigation: derive zone from GNSS, validate against telemetry, and record datum in metadata.
- **Revisit if.** A deployment standardizes on a different national CRS/geoid (swap the grids; the mechanism is unchanged).

---

## 4. Cross-cutting trade-off map

How each decision serves the [Evaluation Criteria](_internal/PROBLEM_STATEMENT.md) and answers a specific single-pass challenge (challenge→mechanism matrix in the [spec §9](_internal/CANONICAL-ARCHITECTURE-SPEC.md)):

| Challenge (single-pass) | Decision(s) | Evaluation criterion served |
|-------------------------|-------------|-----------------------------|
| Short baselines starve triangulation | ADR-01, 06 | Geometric completeness; reconstruction of structures |
| Metric scale without GCPs | ADR-03, 04, 15 | Absolute/relative accuracy (both regimes) |
| Bleeding-edge models OOD on aerial | ADR-02 | Robustness; reliability |
| "Real-time" vs "accurate" tension | ADR-05, 11, 13 | Timeliness; near-real-time preview |
| Moving objects → permanent ghosts | ADR-08 | Model cleanliness; dynamic-object handling |
| No second look to catch errors | ADR-09, 12 | Uncertainty reporting; auditability |
| Link loss / contested comms | ADR-12, 14 | Reliability; data sovereignty |
| Sovereign, offline deployment | ADR-14, 12, 15 | Deployability in NTRO context |
| Cross-tier determinism & reuse | ADR-10, 13 | Engineering robustness; reproducibility |

**What we knowingly lose** (the honest column): against offline photogrammetry incumbents (Pix4D, Metashape, DJI Terra, RealityCapture, ContextCapture, OpenDroneMap) on a *planned multi-pass mission with GCPs*, we expect to lose on **absolute accuracy, facade completeness, and final mesh polish**. Against real-time SLAM/VIO systems we produce a far richer product than their sparse maps. Against NeRF/GS cloud services we are metric, georeferenced, and offline where they are not. DRISHTI's defensible niche — confirmed by the [competitive landscape](_internal/research/8-competitive-landscape.md) — is **single-pass → georeferenced metric textured 3D, near-real-time, with graceful degradation**, which no incumbent currently occupies. We compete there, not on incumbents' home turf.

---

## Open questions / risks

- **Backbone reconciliation.** ADR-02's shipped backbone is the permissive default (Depth Anything 3 / MapAnything), which differs from the spec's registry naming (VGGT/MASt3R). This must be folded into [`CANONICAL-ARCHITECTURE-SPEC.md`](_internal/CANONICAL-ARCHITECTURE-SPEC.md) §7 — tracked jointly with [Technology Stack §5](03-TECHNOLOGY-STACK.md).
- **Aerial fine-tuning is a plan, not a result.** ADR-01/02/06 assume domain fine-tuning brings feed-forward metric accuracy into budget; this is unproven until measured on aerial data (owned with [AI/Deep Learning](roles/3-ai-deep-learning-research.md)).
- **Confidence calibration** (ADR-09) must itself be validated — an uncalibrated confidence is worse than none.
- **Edge smoothing bound** (ADR-04) and **INT8 accuracy** (ADR-11) both need measurement on the event hardware.
- **Two-of-everything cost.** ADR-02/05/13/14 each maintain a primary + fallback or per-tier variant; total integration surface is a real schedule risk for the hackathon MVP (mitigated by the [MVP subset](03-TECHNOLOGY-STACK.md)).

## Further reading

- [How It Works](02-HOW-IT-WORKS.md) — the pipeline these decisions shape.
- [Technology Stack](03-TECHNOLOGY-STACK.md) — the concrete components chosen, with licensing.
- [Integration](04-INTEGRATION.md) — how the decisions compose across tiers.
- [Theory](01-THEORY.md) — the first principles behind the single-pass reasoning.
- Internal: [`CANONICAL-ARCHITECTURE-SPEC.md`](_internal/CANONICAL-ARCHITECTURE-SPEC.md) §9 (challenge→mechanism), research dossier [`8-competitive-landscape.md`](_internal/research/8-competitive-landscape.md).
