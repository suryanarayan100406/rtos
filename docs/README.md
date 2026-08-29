# DRISHTI

**D**rone-based **R**eal-time **I**maging for **S**ingle-pass **H**igh-fidelity **T**errain **I**ntelligence

> *One pass. The whole picture — a georeferenced, measurable 3D model of everything the drone flew
> over, built while it flies.*

**Problem:** NTRO / Smart India Hackathon — *"Single-Pass Drone Video to Accurate 3D Model Generation
System."* Generate a georeferenced, metrically accurate, textured 3D model from a **single** drone
pass — no multi-pass grids, no extensive Ground Control Points, near-real-time.

---

## The idea in one paragraph

Classical drone mapping needs many overlapping passes and hours of Structure-from-Motion + Multi-View
Stereo — it cannot handle the "one chance to fly" reality of disaster response, reconnaissance and
rapid mapping. DRISHTI reframes the problem: a single pass gives **weak multi-view geometry** but a
**strong inertial + GNSS + learned-prior signal**. So DRISHTI fuses **feed-forward learned geometry**
(VGGT / MASt3R pointmaps + metric monocular depth) with a **tightly-coupled GNSS+IMU+RTK factor graph**
that supplies metric scale and georeference **without GCPs**, and runs it as a **streaming, two-tier**
system: a live coarse map on the drone's edge computer during flight, and a full metric textured model
on the ground tier in minutes. Every stage emits a confidence signal and has a fallback, so the system
**degrades gracefully and never hard-fails**.

---

## How to read this documentation

- **Judges / NTRO evaluators (15 min):** start with **[Master Overview](00-MASTER-OVERVIEW.md)** →
  **[Pitch & Presentation](06-PITCH-AND-PRESENTATION.md)**.
- **Technical reviewers:** **[Theory](01-THEORY.md)** → **[How It Works](02-HOW-IT-WORKS.md)** →
  **[Technology Stack](03-TECHNOLOGY-STACK.md)** → **[Integration](04-INTEGRATION.md)** →
  **[Design Decisions](05-DESIGN-DECISIONS.md)**.
- **Build team:** your **[role document](#role-documents)** + How It Works + Integration.

---

## Document map

### Core documents

| Doc | What it covers |
|-----|----------------|
| [00 · Master Overview](00-MASTER-OVERVIEW.md) | The whole system at a glance; thesis, architecture, outputs, evaluation, applications, roadmap, risks. Start here. |
| [01 · Theory & Scientific Foundations](01-THEORY.md) | The math/science: camera geometry, why single-pass is hard, SfM/BA, monocular & feed-forward depth, VIO fusion, GNSS/RTK, georeferencing, 3DGS, TSDF, accuracy theory. |
| [02 · How It Works](02-HOW-IT-WORKS.md) | End-to-end pipeline, stage by stage (S0–S10), the two output paths, a worked example, data-flow, challenge→mechanism traceability. |
| [03 · Technology Stack](03-TECHNOLOGY-STACK.md) | Concrete tools/models/hardware with versions, licenses, rationale, alternatives; BOM; sovereignty/air-gap notes. |
| [04 · Integration](04-INTEGRATION.md) | How components work together: interfaces, message contracts, coordinate frames, time sync, edge↔ground protocol, hardware/software plumbing, failure handling. |
| [05 · Design Decisions](05-DESIGN-DECISIONS.md) | Every major decision with options considered, rationale, trade-offs, and risks. |
| [06 · Pitch & Presentation](06-PITCH-AND-PRESENTATION.md) | Elevator + 3-min scripts, slide-by-slide deck, demo storyboard, judge Q&A. |

### Role documents

| Role | Owns (pipeline stages) |
|------|------------------------|
| [1 · 3D Reconstruction Research Lead](roles/1-3d-reconstruction-lead.md) | Reconstruction backbone: S4 geometry, S5 fusion, S6 global opt, S7 dense/3DGS, S8 mesh/texture |
| [2 · Computer Vision & Video Intelligence](roles/2-computer-vision-video-intelligence.md) | Front-end: S1 ingest/QA/keyframes, matching, flow, S3 dynamic + semantic masking |
| [3 · AI / Deep Learning Research](roles/3-ai-deep-learning-research.md) | Learned models: metric depth, pointmaps, segmentation, completion, confidence, on-device optimization |
| [4 · Drone & Sensor / Hardware Integration](roles/4-drone-sensor-hardware-integration.md) | Capture system: S0 capture/sync, platform, sensors, calibration, comms, edge compute integration |
| [5 · Geospatial & Accuracy Research](roles/5-geospatial-accuracy-research.md) | Metric spine, GNSS/RTK factors (S6), S9 georeferencing/DSM/ortho, accuracy validation, formats |
| [6 · Systems & Edge / Compute Optimization](roles/6-systems-edge-compute-optimization.md) | Runtime: tiering, ROS 2, TensorRT, streaming, reliability spine, deployment |

### Reference (internal)

| File | Purpose |
|------|---------|
| [Problem statement + Output/Evaluation tables](_internal/PROBLEM_STATEMENT.md) | Canonical problem + the two required tables |
| [Canonical architecture spec](_internal/CANONICAL-ARCHITECTURE-SPEC.md) | Single source of truth for tiers, stages, models, spines |
| [Style guide](_internal/STYLE_GUIDE.md) | Authoring conventions & honesty policy |
| `_internal/research/*.md` | Six domain research dossiers underpinning the design |

---

## Quick facts

- **Architecture:** three tiers (Edge on Jetson Orin · Ground GPU · optional Cloud), two output paths
  (Live coarse map S0–S5 · Refined metric model S6–S10).
- **Accuracy (design targets):** RTK/PPK → 3–8 cm horizontal; GPS-only → sub-metre; GSD ≈ 2 cm/px.
- **Reliability:** graceful-degradation ladder L0–L6; always emits a best-effort model + uncertainty
  report; store-and-forward so link loss never loses data.
- **Outputs:** LAS/LAZ point cloud, textured glTF/OBJ/OGC 3D Tiles mesh, 3DGS scene, DSM/DTM +
  orthomosaic (GeoTIFF), semantic layers, measurements, and an accuracy report.

---

*See the [Master Overview](00-MASTER-OVERVIEW.md) for the full story.*
