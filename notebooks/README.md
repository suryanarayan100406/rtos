# DRISHTI notebooks

Two ways to run DRISHTI on a **free cloud T4** (Google Colab or Kaggle). Both install the *same*
`drishti` package and run the *same* CLI as every other tier — the only difference is the CUDA runtime
the host provides. Nothing is stubbed: the doctor and the run report real hardware and real results, and
**fail loudly** if a dependency, the GPU, or an input is missing. Pick by where you want the *light*
stages to run.

> **New here? Follow [`HOW_TO_USE.md`](HOW_TO_USE.md)** — a click-by-click walkthrough from opening the
> notebook to getting the model back on your machine. This file is the quick reference.

| Notebook | Runs on the T4 | You do locally | Use when |
|----------|----------------|----------------|----------|
| [`drishti_colab_full.ipynb`](drishti_colab_full.ipynb) | **all 11 stages** (`s0_ingest → report`) | only *view / serve* the finished bundle | you don't have the heavy deps locally and just want the whole model built for you |
| [`drishti_cloud_t4.ipynb`](drishti_cloud_t4.ipynb) | the GPU-heavy stages (`--upto s7_dense`) | `drishti resume` the remaining light stages | you have the light extras locally and want to keep the CPU stages on your machine |

---

## `drishti_colab_full.ipynb` — run EVERYTHING on the GPU

Installs the full stack on the T4 and runs the **entire pipeline end-to-end there**, then hands you the
complete bundle (model + deliverables + accuracy report + **logs**) to download. Locally you only
*view/serve* it — no heavy dependencies required.

It covers the full **import → run → export** loop: generate a realistic dataset on the T4 (or bring your
own via Drive/upload, or resume a partial local bundle), run all stages with verbose logs tee'd into the
bundle, then `inspect`/`verify` and download the whole thing (full or slim archive).

Beyond the `pyproject` extras it installs two packages the extras don't list but a full run genuinely
needs — **`transformers`** (S3/S4 load their models from it) and **`laspy`** (S10 writes the LAS
deliverable) — and, optionally, portable **Blender 4.2** so the **FBX** output is real rather than
skipped. It runs the **`balanced`** profile: `--profile max` selects `dense.method=gaussian` +
`mesh.method=2dgs`, which this build does not implement and which **raise loudly** at `s7_dense`.

**Quick start**

1. Open in Colab → *Runtime → Change runtime type → GPU (T4)* (or enable the Kaggle GPU accelerator).
2. Edit the `REPO_URL` / `REPO_REF` cell (§1) to point at your DRISHTI repo.
3. Run §0–§4 (GPU → code → install → doctor), then pick **one** import cell in §5.
4. Run §6 (full pipeline) → §7–§9 (inspect, logs, download).
5. Back home, just view it:
   ```bash
   unzip <run_id>_full.zip -d runs/
   drishti inspect runs/<run_id>          # the same manifest you saw on the T4
   drishti verify  runs/<run_id>          # confirm the download is intact
   python -m server.app                   # then browse the viewer
   ```

## `drishti_cloud_t4.ipynb` — tier split (heavy on the T4, light at home)

Runs the **GPU-heavy** stages (`s2_poses`, `s3_masking`, `s4_depth`, `s7_dense`) on the T4 via
`--upto s7_dense`, then produces a downloadable bundle you finish and serve on the CPU ground station.

**Quick start**

1. Open in Colab → *Runtime → Change runtime type → GPU (T4)* (or enable the Kaggle GPU accelerator).
2. Edit the `REPO_URL` / `REPO_REF` cell to point at your DRISHTI repo.
3. Run top to bottom; provide your `mission.yaml` (+ video/telemetry) when prompted.
4. Download the zipped bundle and **resume** the remaining stages locally:
   ```bash
   unzip <run_id>.zip -d runs/
   drishti resume runs/<run_id>           # continues at s8_mesh; the GPU stages are already 'done'
   ```

`resume` reuses the config recorded in the bundle (no `--profile`/`--dataset` flags); it skips whatever
the T4 already completed because the bundle is content-hashed and resumable.

---

Both bundles are content-hashed and byte-compatible across tiers. See
`docs/implementation/02-SYSTEM-DESIGN.md` §6 for the local-vs-cloud placement policy and
[`docs/GUIDE.md`](../docs/GUIDE.md) for the full usage guide.
