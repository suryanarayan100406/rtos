# Problem Statement (Reference) + Filled Output & Evaluation Tables

> This is the canonical restatement of the NTRO problem plus the two tables the brief asked us to
> author ("Add 'Desired Output' and 'Evaluation Criteria' table here"). Every document derives its
> scope from this file. Do not silently change scope; if scope must change, update this file first.

---

## 1. Problem (verbatim, condensed)

**Title:** Single-Pass Drone Video to Accurate 3D Model Generation System
**Organization / Department:** National Technical Research Organisation (NTRO)
**Category:** Software · **Theme:** Robotics and Drones
**Dataset:** Provided in real time (at the event).

**Background.** Accurate 3D models of buildings, infrastructure, terrain and objects normally require
multiple drone passes, heavy image overlap, specialized flight planning and long post-processing. In
disaster response, surveillance, infrastructure inspection, military reconnaissance and rapid mapping
there is often **only one opportunity** to capture data. A system that produces an accurate, textured
3D model from a **single drone pass** would cut mission time, operator effort, data-acquisition
requirements and processing complexity, and enable **near-real-time situational awareness**.

**Task.** Build an AI-enabled system that generates a **georeferenced, metrically accurate 3D model**
from a **single-pass drone video** captured by a moving UAV, reconstructing: (i) 3D terrain &
structures, (ii) building facades & rooftops, (iii) roads & infrastructure, (iv) vegetation &
obstacles, (v) textured 3D meshes or point clouds — suitable for **visualization, measurement and
analysis**.

**Key challenges.** (i) Limited viewing angles from a single flight path; (ii) motion blur & video
compression artifacts; (iii) variable illumination & shadows; (iv) dynamic objects (vehicles, humans,
animals); (v) GPS inaccuracy & sensor noise; (vi) real-time / near-real-time processing; (vii)
reconstruction of occluded surfaces; (viii) metric accuracy **without extensive Ground Control Points
(GCPs)**.

**Inputs.**
- **Mandatory:** Drone video (1080p/4K); GPS coordinates; flight metadata.
- **Optional:** IMU data; barometric altitude; camera intrinsic parameters; RTK/PPK corrections.

**Applications.** Border & strategic-area mapping; disaster damage assessment; urban planning & smart
cities; infrastructure inspection; construction progress monitoring; archaeological documentation;
digital-twin generation; military reconnaissance & mission planning.

---

## 2. Desired Output (our authored deliverable table)

Everything below is produced from **one flight** and exported in open, interoperable, GIS/CAD/
digital-twin-ready formats. Each artifact carries a per-region **confidence** value.

| # | Deliverable | Primary format(s) | Purpose / use |
|---|-------------|-------------------|----------------|
| 1 | **Georeferenced dense point cloud** | LAS/LAZ, PLY | Measurement, GIS ingest, ground truth for downstream products |
| 2 | **Textured 3D mesh (multi-LOD)** | OBJ+MTL, glTF/GLB, OGC **3D Tiles**, OSGB | Visualization, inspection, digital twin |
| 3 | **3D Gaussian-Splat scene** | `.ply` / `.splat` / `.ksplat` | Photorealistic free-viewpoint situational awareness |
| 4 | **Digital Surface Model (DSM) + Digital Terrain Model (DTM)** | GeoTIFF (float32, COG) | Elevation, volumetrics, terrain & slope analysis |
| 5 | **True orthomosaic** | GeoTIFF (COG) | Top-down basemap, planning, change detection |
| 6 | **Semantic layers** | GeoJSON / Shapefile + labeled cloud & mesh | building/roof, road/infra, vegetation, terrain, obstacle; dynamic objects removed |
| 7 | **Measurements & analytics** | JSON + in-viewer tools | Distances, areas, volumes, heights, slopes, clearances |
| 8 | **Accuracy & confidence report** | PDF + JSON | RMSE (H/V), GSD, coverage %, per-region confidence, occlusion map |
| 9 | **Live coarse map + telemetry** | Streamed 3D + 2D (WebRTC/RTSP → web viewer) | Near-real-time situational awareness **during** the flight |
| 10 | **Reproducible project bundle** | Archive (video ref, poses, calibration, logs, metadata) | Audit, re-processing, chain-of-custody |

---

## 3. Evaluation Criteria (our authored assessment table)

Targets assume a nadir/oblique single pass at typical AGL. Two accuracy regimes are reported because
RTK/PPK is *optional* input: **(A) RTK/PPK available** and **(B) GPS-only**.

| # | Criterion | Metric / how measured | Target (A: RTK/PPK · B: GPS-only) |
|---|-----------|------------------------|------------------------------------|
| 1 | **Absolute geometric accuracy** | Horizontal & vertical RMSE vs independent checkpoints/survey | A: ≤ 2–5 cm + 1×GSD · B: ~0.5–2 m (report with CI) |
| 2 | **Relative accuracy** | Known-distance / scale-bar error; local RMSE | A: < 1% of distance · B: report measured value |
| 3 | **Ground Sampling Distance (GSD)** | cm/pixel from AGL & sensor geometry | ~1.5–3 cm/px at typical mapping altitude |
| 4 | **Completeness / coverage** | % target surface above confidence threshold; occlusion-flagged area | Maximize; **explicitly report** unseen / low-confidence regions |
| 5 | **Reconstruction fidelity** | Point density (pts/m²), mesh detail, texture sharpness, hole ratio | Quantitative + qualitative panel |
| 6 | **Quality vs reference** | Cloud-to-cloud (C2C), cloud-to-mesh (C2M), Chamfer vs COLMAP/Metashape or LiDAR | Minimize distance; report percentiles |
| 7 | **Processing latency** | (a) live-preview latency per keyframe; (b) time-to-final-model per minute of video / km² | a: near-real-time (≈1–2 s/keyframe, edge) · b: minutes (ground GPU) |
| 8 | **Robustness** | Degradation curve under injected blur, illumination change, GPS noise, dynamic-object density | Graceful, monotonic; **no hard failure** |
| 9 | **Dynamic-object rejection** | Precision/recall of masked movers; residual "ghost" density in map | High recall; near-zero ghosting |
| 10 | **Georeferencing correctness** | Absolute position error of known features; CRS/datum/geoid correctness | Within accuracy budget; correct EPSG + geoid model |
| 11 | **Real-world deployability** | Runs on Jetson Orin + ground GPU; integrates with DJI/PX4; power/thermal within budget | Demonstrated on real hardware |
| 12 | **Reliability / availability** | Recovery from tracking loss, link loss, sensor dropout; % missions yielding a usable model | High; documented fallback ladder |
| 13 | **Usability & interoperability** | Visualization + measurement UX; export into GIS/CAD/twin toolchains | Standards-compliant, tool-agnostic outputs |

---

## 4. Honesty guardrails (read before writing any claim)

- "Real-time" = **near-real-time live coarse preview on the edge** + **full refined model in minutes**
  on the ground tier. We never claim a full 4K textured mesh in hard real-time on the drone.
- "Never fails" = **graceful degradation with no single point of failure**; the system always emits a
  best-effort model **plus an explicit uncertainty/completeness report**, rather than crashing or
  silently emitting garbage.
- "Metric without GCPs" = accurate **because** of tightly-fused GNSS(+RTK/PPK)+IMU+visual scale, with
  honest, sensor-configuration-dependent accuracy numbers — not a claim that GCP-grade survey accuracy
  comes for free from GPS-only input.
