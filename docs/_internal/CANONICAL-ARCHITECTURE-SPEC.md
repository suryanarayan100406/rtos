# DRISHTI — Canonical Architecture Spec (single source of truth)

> **Status:** v1 (author's synthesis). Will be reconciled with the research dossiers in
> `research/*.md`. **This file is authoritative** for tiers, stage names (S0–S10), model choices,
> interfaces, and the reliability/metric spines. All public documents must conform to it or raise a
> conflict in their "Open questions" section. Do not invent alternative names.

---

## 0. Thesis

Traditional drone mapping is **multi-pass photogrammetry**: fly a dense lawnmower grid with 70–80%
overlap, land, then run hours of Structure-from-Motion (SfM) + Multi-View Stereo (MVS). It fails the
NTRO scenario because there is **one pass, one chance**.

DRISHTI reframes the problem. A single pass gives **weak multi-view geometry** (short baselines,
limited angles) but a **strong temporal + inertial + GNSS signal** and **strong learned priors**.
So DRISHTI is a **prior-assisted, sensor-fused, streaming reconstruction system**, not a classical
photogrammetry clone. Three ideas carry the whole design:

1. **Learned geometry beats missing views.** Feed-forward multi-view models (VGGT / MASt3R) and
   metric monocular depth (Metric3D v2 / UniDepth / Depth Anything V2) supply dense geometry where
   triangulation baselines are too short — exactly the single-pass weakness.
2. **Sensors supply the metric truth GCPs normally would.** A tightly-coupled
   GNSS(+RTK/PPK)+IMU+visual **factor graph** fixes scale and georeference without control points.
3. **Everything streams and everything degrades gracefully.** A live coarse map is built on the edge
   during flight; a metric textured model is refined on the ground in minutes. Every stage emits
   confidence and has a fallback, so the system never hard-fails.

---

## 1. Design principles (the four anchors — repeated across all docs)

- **A1 — Prior-assisted geometry.** Fuse classical multi-view constraints with learned depth/pointmap
  priors; never rely on triangulation alone.
- **A2 — Metric spine.** One factor graph fuses VIO + GNSS + RTK/PPK + IMU pre-integration + baro →
  metric, georeferenced camera trajectory without GCPs.
- **A3 — Two output paths.** *Live path* (S0–S5, edge, near-real-time coarse map) and *Refine path*
  (S6–S10, ground, minutes-scale metric textured model). The live path is always available even if
  the refine path is interrupted.
- **A4 — Reliability spine.** Every stage emits a confidence/uncertainty signal and has a defined
  fallback; the system always emits a best-effort model **plus** an explicit uncertainty/completeness
  report. No single point of failure.

---

## 2. Three-tier topology

```mermaid
flowchart LR
  subgraph EDGE["Edge Tier — on UAV (NVIDIA Jetson AGX Orin)"]
    direction TB
    E1[S0 Capture & Sync]
    E2[S1 Ingest & Frame QA]
    E3[S2 Odometry & Localization VIO+GNSS+IMU]
    E4[S3 Perception & Masking]
    E5[S4 Depth & Geometry live]
    E6[S5 Live Fusion nvblox TSDF]
  end
  subgraph GROUND["Ground Tier — station / server GPU"]
    direction TB
    G1[S6 Global Optimization BA+GNSS]
    G2[S7 Dense Recon & 3DGS]
    G3[S8 Meshing & Texturing]
    G4[S9 Georeferencing & Semantics]
    G5[S10 Export & Serve]
  end
  subgraph CLOUD["Cloud Tier — optional"]
    C1[Scale-out reprocess / tiling]
    C2[3D Tiles serving / digital twin]
  end
  EDGE -->|keyframes+poses+depth+masks+confidence, store-and-forward| GROUND
  GROUND -->|large-area / archive| CLOUD
  EDGE -.->|live coarse map stream| OP[Operator viewer]
  GROUND -->|final metric model + reports| OP
```

- **Edge Tier (on-UAV, Jetson AGX Orin 64GB / Orin NX 16GB):** runs the **live path** S0–S5. Produces
  the live coarse map for situational awareness and the compact keyframe package for the ground tier.
- **Ground Tier (laptop/workstation with RTX GPU, or rugged field server):** runs the **refine path**
  S6–S10. Produces the metric, textured, georeferenced deliverables in minutes.
- **Cloud Tier (optional):** large-area tiling, batch re-processing, 3D Tiles / digital-twin serving.
  Not required for a mission to succeed (air-gap friendly for defense use).

> Deployment flexes: for a **fully offline / air-gapped** mission the ground tier is a rugged laptop
> in the field; the cloud tier is omitted. For the **hackathon** everything can run on one
> workstation; the edge/ground split is emulated.

---

## 3. Canonical pipeline — stages S0–S10

Legend: **[E]** edge, **[G]** ground, **[E→G]** starts on edge, refined on ground.
Each stage lists: inputs → method/tools → outputs, plus **conf** (confidence signal) and **fallback**.

### S0 — Capture & Sync **[E]**
- **In:** camera frames (1080p/4K H.264/H.265), GNSS, IMU, baro, gimbal encoders, RTK stream (opt).
- **Method:** hardware/PTP/PPS + GPS-time stamping; align every frame to IMU/GNSS/baro on a common
  clock; log intrinsics (EXIF) or trigger self-calibration.
- **Out:** time-synchronized sensor streams with per-sample timestamps + quality flags.
- **conf:** per-sensor validity flags. **fallback:** software timestamp interpolation if PPS absent;
  flag increased temporal uncertainty.

### S1 — Ingest & Frame QA **[E]**
- **In:** synced video + IMU.
- **Method:** NVDEC/GStreamer decode; **blur gating** (variance-of-Laplacian + IMU angular-rate);
  exposure/over-under gating; **keyframe selection** by parallax + sharpness + coverage; optional
  **IMU-aided / learned deblur** (NAFNet-class) on borderline frames.
- **Out:** clean keyframe set + per-frame quality scores.
- **conf:** sharpness/exposure score per keyframe. **fallback:** widen keyframe spacing when frames
  are poor; mark temporal gaps rather than forcing bad frames.

### S2 — Odometry & Localization (metric spine) **[E→G]**
- **In:** keyframes + IMU + GNSS(+RTK/PPK) + baro.
- **Method:** real-time **VIO** (OpenVINS / VINS-Fusion) on edge; **tightly-coupled factor graph**
  (GTSAM/iSAM2) fusing IMU pre-integration + GNSS factors + baro + (opt) RTK; robust kernels
  (Huber/DCS) reject GNSS outliers. Global refinement continues in S6.
- **Out:** metric, gravity-aligned, georeferenced 6-DoF camera trajectory + covariance.
- **conf:** pose covariance. **fallback:** GNSS dropout → VIO/IMU dead-reckoning (drift-bounded);
  visual loss → inertial propagation with rising uncertainty; re-anchor on GNSS re-acquire.

### S3 — Perception & Masking **[E]**
- **In:** keyframes + optical flow.
- **Method:** **dynamic-object** instance seg + tracking (YOLO11-seg / SAM2 + ByteTrack) for
  vehicles/humans/animals; **motion detection** via RAFT flow vs epipolar/rigid-flow residual;
  **semantic** labels (Mask2Former/OneFormer) → building/roof, road/infra, vegetation, terrain,
  obstacle. Dynamic masks are removed from geometry; optionally kept as a separate moving-object layer.
- **Out:** per-keyframe dynamic masks + semantic label maps.
- **conf:** mask/segment confidence. **fallback:** if seg model fails, use motion-residual masking
  alone; conservative dilation to avoid contaminating the static map.

### S4 — Depth & Geometry **[E→G]**
- **In:** keyframes + poses + masks.
- **Method:** **metric monocular depth** (Metric3D v2 / UniDepthV2 / Depth Anything V2-metric, Depth
  Pro for edges) **scale-aligned** to the metric trajectory; **feed-forward multi-view geometry**
  (VGGT / MASt3R pointmaps) over local keyframe windows for cross-view consistency; optional learned
  **MVS** (CasMVSNet/PatchmatchNet) where baselines allow. All produce **per-pixel confidence**.
- **Out:** confidence-weighted metric depth/pointmaps per keyframe (edge: fast model; ground: full).
- **conf:** per-pixel depth confidence + multi-view agreement. **fallback:** disagreeing regions kept
  at low confidence; monocular-only where multi-view fails.

### S5 — Live Fusion **[E]**
- **In:** depth/pointmaps + poses + masks + confidence.
- **Method:** **incremental confidence-weighted TSDF** (NVIDIA **nvblox** on Jetson); dynamic-masked;
  progressive **LOD**; live colored coarse mesh/point cloud streamed to the operator.
- **Out:** live coarse georeferenced map (situational awareness) + compact keyframe package for ground.
- **conf:** per-voxel weight. **fallback:** if edge compute saturates, drop to point-splat preview and
  defer meshing to ground.

### S6 — Global Optimization **[G]**
- **In:** full keyframe package (frames, poses+cov, GNSS/RTK/PPK, depth, masks).
- **Method:** global **bundle adjustment** / pose-graph with **GNSS + IMU + (opt) RTK/PPK factors**
  and depth/pointmap constraints (GTSAM / COLMAP-style / VGGT-consistent); camera **self-calibration**;
  loop/overlap closure where the single path self-intersects.
- **Out:** globally consistent metric camera set + refined sparse structure + accuracy covariance.
- **conf:** posterior covariance, reprojection RMSE. **fallback:** if BA diverges, keep VIO+GNSS prior
  poses and flag reduced global accuracy.

### S7 — Dense Reconstruction & 3DGS **[G]**
- **In:** optimized poses + keyframes + depth priors + masks.
- **Method:** **few-shot 3D Gaussian Splatting** initialized from feed-forward geometry
  (InstantSplat-style init from VGGT/MASt3R) with **depth + normal + confidence regularization** to
  fight single-pass under-constraint; **per-image appearance embeddings** for illumination/shadow
  variation; **occlusion completion** via learned + geometric priors (planarity/symmetry for man-made
  structure), with completed regions flagged low-confidence.
- **Out:** dense 3DGS scene + dense fused point cloud.
- **conf:** per-Gaussian opacity/consistency + completion flag. **fallback:** if GS under-constrained,
  fall back to TSDF/MVS fused cloud at reduced fidelity.

### S8 — Meshing & Texturing **[G]**
- **In:** 3DGS / fused cloud.
- **Method:** surface extraction via **2DGS / SuGaR** or **screened Poisson**; watertighting +
  decimation → multi-**LOD** mesh; **texturing** by photometric best-view selection / MVS-Texturing;
  optional PBR-ish material split.
- **Out:** textured multi-LOD mesh.
- **conf:** per-face texture/geo confidence. **fallback:** vertex-colored mesh if texturing fails.

### S9 — Georeferencing & Semantics **[G]**
- **In:** mesh + cloud + poses + semantic labels.
- **Method:** transform to target **CRS** (WGS84 → UTM/EPSG, geoid model for orthometric height);
  propagate semantic labels to mesh/cloud; rasterize **DSM/DTM** and render **true orthomosaic**;
  compute **measurements** (areas/volumes/heights).
- **Out:** georeferenced classified mesh/cloud + DSM/DTM + orthomosaic + measurements.
- **conf:** georef residuals. **fallback:** local ENU frame + relative-only products if GNSS was poor.

### S10 — Export & Serve **[G/Cloud]**
- **In:** all products.
- **Method:** export **LAS/LAZ, PLY, OBJ/glTF/GLB, OGC 3D Tiles, GeoTIFF (COG) DSM/DTM/ortho,
  CityJSON**; generate **accuracy & confidence report** (RMSE/GSD/coverage/occlusion); publish to web
  viewer (Cesium/Potree) + measurement API; write **reproducible project bundle**.
- **conf:** whole-model accuracy report. **fallback:** always emit at least point cloud + report.

---

## 4. The metric / georeferencing spine (A2, detail)

```mermaid
flowchart LR
  IMU[IMU preintegration] --> FG((Factor graph GTSAM/iSAM2))
  VIS[Visual keyframe factors] --> FG
  GNSS[GNSS factors] --> FG
  RTK[RTK/PPK factors opt] --> FG
  BARO[Barometer factor] --> FG
  FG --> POSE[Metric georeferenced trajectory + covariance]
  POSE --> BA[Global BA S6]
  BA --> POSE
```

- **Scale** comes from IMU (accelerometer + gravity) and GNSS baselines — this is what removes the
  monocular scale ambiguity **without GCPs**.
- **Georeference** comes from GNSS(+RTK/PPK); we align the local metric solution to ECEF/WGS84 then
  project to UTM.
- **Robustness:** Huber/DCS robust kernels + GNSS outlier rejection; covariance is carried through to
  the accuracy report. Opportunistic GCPs (if any exist in-scene) can be added as extra factors but
  are **not required**.

## 5. Reliability spine (A4) — graceful-degradation ladder

| Level | Trigger | Behavior | Product impact |
|------|---------|----------|----------------|
| L0 | All sensors nominal (+RTK) | Full path | Best accuracy (cm) |
| L1 | No RTK/PPK | GNSS+IMU+VIO scale | Sub-meter absolute, strong relative |
| L2 | GNSS dropout (canyon/denied) | VIO+IMU dead-reckon; re-anchor on re-acquire | Local metric map; georef on re-acquire |
| L3 | Brief visual loss | IMU inertial propagation | Short gap, flagged high-uncertainty |
| L4 | Bad frames (blur/dark) | Gate out; widen keyframes; mark gaps | Coverage holes flagged, not garbage |
| L5 | Neural model OOM/fail | Fall back to classical MVS/monocular; lower LOD | Reduced fidelity, still valid |
| L6 | Edge saturated | Point-splat preview; defer to ground | Live preview simpler; refine unaffected |
| — | Always | Store-and-forward raw stream; deterministic ground reprocess | Full quality recoverable post-mission |

**Invariant:** the system always emits (a) a best-effort model and (b) an explicit
uncertainty/coverage report. It never crashes silently or emits unflagged garbage.

## 6. Coordinate frames & conventions

- **Camera:** OpenCV convention (x-right, y-down, z-forward), intrinsics `K`, distortion.
- **Body/IMU:** ROS REP-103 FLU (x-forward, y-left, z-up); camera↔IMU extrinsics from **Kalibr**.
- **Local map:** ENU, gravity-aligned (REP-105 `map` frame).
- **Global:** ECEF ↔ geographic WGS84 ↔ projected **UTM/EPSG**; orthometric height via geoid (e.g.
  EGM2008 / local geoid).
- Time: single monotonic clock; GPS-time as global reference; PPS/PTP where available.

## 7. Model & tool registry (adopted choices)

| Role | Adopted | Notable alternatives |
|------|---------|----------------------|
| Real-time VIO | OpenVINS / VINS-Fusion | ORB-SLAM3, Kimera |
| Sensor fusion / BA | GTSAM (iSAM2) + COLMAP-style BA | Ceres, g2o |
| Feed-forward geometry | VGGT, MASt3R / MASt3R-SfM | DUSt3R, Fast3R, Spann3R, CUT3R |
| Metric mono depth | Metric3D v2, UniDepthV2, Depth Anything V2 | ZoeDepth, Depth Pro, Marigold |
| Learned MVS (opt) | CasMVSNet / PatchmatchNet | Vis-MVSNet, MVSFormer |
| Dense matching (hard pairs) | RoMa / DKM; SuperPoint+LightGlue | LoFTR, ALIKED, DISK |
| Dynamic object seg+track | YOLO11-seg / SAM2 + ByteTrack | Mask2Former video, Cutie |
| Semantic segmentation | Mask2Former / OneFormer | SegFormer, InternImage |
| Optical flow | RAFT | GMFlow, SEA-RAFT |
| Live TSDF fusion | NVIDIA nvblox (edge), Open3D (ground) | Voxblox, VDBFusion |
| Few-shot 3DGS | InstantSplat-style (VGGT/MASt3R init) + depth/normal reg | FSGS, DNGaussian, MVSplat |
| Mesh from GS | 2DGS / SuGaR | Poisson (screened), GOF, Neuralangelo |
| Texturing | MVS-Texturing / best-view | nvdiffrast bake |
| Geospatial I/O | PROJ/pyproj, GDAL, PDAL, LAStools/Entwine | — |
| 3D Tiles / viewer | py3dtiles + CesiumJS; Potree | — |
| Middleware | ROS 2 (Humble/Jazzy) + Isaac ROS; DeepStream/GStreamer | custom async + DDS |
| Edge runtime | TensorRT (INT8/FP16), CUDA, JetPack | ONNX Runtime |
| Compute | Jetson AGX Orin 64GB (edge) · RTX-class GPU (ground) | Orin NX 16GB (lighter) |

## 8. Hardware reference

- **Primary UAV:** DJI **Matrice 350 RTK** + **Zenmuse** payload (mechanical shutter option, RTK
  built-in, PSDK/onboard access) — COTS, defensible, RTK-native.
- **Open alternative:** custom **PX4/ArduPilot** airframe + global-shutter machine-vision camera +
  survey GNSS (RTK) + tactical/industrial IMU + **Jetson** companion; **MAVLink/MAVSDK** control.
- **Positioning:** GNSS multi-constellation; **RTK via NTRIP** base/caster or **PPK** post-process;
  barometer; magnetometer. **Cam–IMU calibration:** Kalibr. **Intrinsics:** EXIF or self-calibration.
- **Comms:** RTSP/SRT video + MAVLink telemetry downlink; **store-and-forward** on the edge so link
  loss never loses data.

## 9. Challenge → mechanism traceability matrix

| # | Key challenge | DRISHTI mechanism (stages) |
|---|---------------|-----------------------------|
| i | Limited viewing angles | Feed-forward multi-view priors + metric mono depth (S4); oblique-gimbal capture SOP; occlusion completion + honest flagging (S7) |
| ii | Motion blur & compression | Frame QA gating + IMU-aided/learned deblur (S1); artifact-tolerant dense matching RoMa (S4) |
| iii | Illumination & shadows | Exposure normalization (S1); per-image appearance embeddings + lighting-robust depth (S4/S7) |
| iv | Dynamic objects | Seg+track masking + motion-residual detection; excluded from fusion (S3/S5) |
| v | GPS inaccuracy & noise | Tightly-coupled factor graph, robust kernels, GNSS outlier rejection, RTK/PPK (S2/S6) |
| vi | Real-time need | Edge live path + ground refine path; TensorRT; incremental nvblox TSDF; LOD (S1–S5) |
| vii | Occluded surfaces | Learned + geometric completion priors, multi-view where possible, low-confidence flags (S7) |
| viii | Metric accuracy w/o GCP | GNSS/RTK/PPK+IMU+baro scale + BA with GNSS factors + self-calibration + uncertainty report (S2/S6/S9) |

## 10. Deployment configurations

- **Rapid recon / disaster (GPS-only, air-gapped):** L1 accuracy; live coarse map first, full model on
  a field laptop in minutes; no cloud.
- **Survey-grade (RTK/PPK):** L0 accuracy (cm); full deliverable set with accuracy certificate.
- **GPS-denied (urban/EW):** L2; VIO-metric local map, georeferenced on GNSS re-acquire or via
  known landmarks.

## 11. Hackathon MVP vs full system

- **MVP (buildable in the event, on the provided dataset):** video+GPS+metadata → keyframe QA →
  poses (VGGT/MASt3R or COLMAP) → metric mono depth scale-aligned to GPS → fused cloud/TSDF (Open3D)
  → few-shot 3DGS (InstantSplat) → mesh (2DGS/Poisson) → georeference to UTM → export
  LAS/glTF + DSM + ortho + **accuracy report vs COLMAP/Metashape reference (C2C/C2M)** + web viewer;
  **dynamic-object masking** (YOLO/SAM2) shown; **graceful-degradation** demoed by injecting GPS
  noise / blur.
- **Live stretch:** stream a clip through an edge-emulated path with a live nvblox coarse preview.
- **Full system:** on-UAV Jetson deployment, RTK/NTRIP, ROS 2 pipeline, cloud tiling & 3D Tiles
  serving, hardened reliability spine.

**Our defensible IP** is the *orchestration*: prior+geometry fusion, GNSS/IMU scale-alignment of
learned depth, confidence propagation, single-pass tuning, and the graceful-degradation + georeferenced
reporting layer — not any single third-party model.

## 12. Accuracy budget (design targets, to be measured)

| Config | Absolute H | Absolute V | Relative / local | GSD (≈80 m AGL) |
|--------|-----------|-----------|------------------|------------------|
| RTK/PPK (L0) | 3–8 cm | 5–12 cm | <1% of distance | ~2 cm/px |
| GNSS-only (L1) | 0.5–2 m | 0.8–2.5 m | dm-level local | ~2 cm/px |
| GPS-denied (L2) | georef on re-acquire | — | dm-level local | ~2 cm/px |

## 13. Glossary (shared)

VIO — Visual-Inertial Odometry · GNSS — Global Navigation Satellite System · RTK/PPK — Real-Time
Kinematic / Post-Processed Kinematic · GCP — Ground Control Point · GSD — Ground Sampling Distance ·
TSDF — Truncated Signed Distance Function · 3DGS — 3D Gaussian Splatting · MVS — Multi-View Stereo ·
SfM — Structure-from-Motion · BA — Bundle Adjustment · CRS — Coordinate Reference System · DSM/DTM —
Digital Surface/Terrain Model · LOD — Level of Detail · AGL — Above Ground Level · pointmap —
per-pixel 3D point prediction from a feed-forward network.
