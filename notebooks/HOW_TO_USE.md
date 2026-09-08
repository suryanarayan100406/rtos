# How to use DRISHTI on a free cloud GPU (Colab / Kaggle)

You don't need a local GPU. This is a **click-by-click walkthrough** for running the whole DRISHTI
pipeline on a free **T4** and getting the finished 3D model + deliverables + report + logs back onto your
machine.

- **Reference** (what each notebook *is*): [`README.md`](README.md) in this folder.
- **Full usage + how-it-works** (local runs, the dataset descriptor, every stage in detail):
  [`../docs/GUIDE.md`](../docs/GUIDE.md).
- **This file**: the shortest path from "nothing" to "a model on my disk."

---

## Which notebook?

| If you want to… | Use | Local deps needed |
|-----------------|-----|-------------------|
| **Build the whole model on the GPU**, then just look at it | [`drishti_colab_full.ipynb`](drishti_colab_full.ipynb) | none (only to *view*) |
| Keep the CPU stages on your own machine (tier split) | [`drishti_cloud_t4.ipynb`](drishti_cloud_t4.ipynb) | the light extras |

If you're unsure, use **`drishti_colab_full.ipynb`** — everything runs on the T4 and you don't install
anything heavy locally. The rest of this guide follows that notebook; the tier-split variant is covered
at the end.

---

## Step 0 — Open the notebook on a GPU

1. Upload `notebooks/drishti_colab_full.ipynb` to <https://colab.research.google.com> (**File → Upload
   notebook**), or open it from your GitHub repo (**File → Open notebook → GitHub**).
2. **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.**
   *(Kaggle: New Notebook → upload the `.ipynb` → Session options → Accelerator → GPU T4.)*
3. Run the first cell (**§0 · Confirm the GPU**). You should see a table from `nvidia-smi` naming a
   **Tesla T4**. No GPU listed → go back to step 2; the neural and dense stages need CUDA.

> Colab free sessions are time-limited and can disconnect. A big dataset (`aukerman`, `lewis`) plus the
> full run can take a while — start with `brighton_beach` or `synthetic` for your first pass.

## Step 1 — Point it at the code

In **§1 · Get the code**, set one line and run the cell:

```python
REPO_URL = "https://github.com/<your-org>/drishti.git"   # <-- your DRISHTI repo
REPO_REF = "main"                                         # branch, tag, or commit
```

The cell clones the repo and `cd`s into it. (On Kaggle you can instead attach the repo as a dataset and
set `REPO_DIR` to its path, then skip the clone.)

## Step 2 — Install everything

Run **§2 · Install**. You don't have to edit anything — just run the cell. It is self-healing:

- **It installs DRISHTI core first**, then each heavy group in isolation, so one un-buildable wheel can
  never abort the whole install (that failure is what leaves you with `drishti: command not found`).
- **It works around Colab's Python 3.13.** The dense/mesh stages need **Open3D**, which ships wheels only
  up to Python 3.12, and today's Colab/Kaggle run 3.13. So on a > 3.12 kernel the cell builds an isolated
  **Python 3.12** environment (via [`uv`](https://github.com/astral-sh/uv)), installs everything there,
  and puts it first on `PATH` — every later `!drishti` / `!python` transparently uses it. This is what
  makes a genuinely **full** run (Open3D and all) possible on current runtimes. Expect it to take a few
  minutes the first time (it downloads a Python plus CUDA PyTorch).
- **It adds two packages the `pyproject` extras don't list** but a full run needs: **`transformers`**
  (S3/S4 load their models through it) and **`laspy`** (S10 writes the LAS deliverable).

The cell finishes with a capability probe printing `OK` / `MISSING` per module — that's your first honest
look at what's installed; §4 `doctor` then gates each stage on the same imports.

## Step 3 — (optional) Blender, for the FBX file

Run **§3** only if you want the **FBX** deliverable. It downloads portable **Blender 4.2** (DRISHTI
exports FBX through headless Blender ≥ 4.0). Skip it and FBX is simply recorded as *skipped* and
`s10_export` marked **degraded** — every other format (OBJ, PLY, LAS, GeoTIFF, glTF) still exports.

## Step 4 — Doctor: the honest gate

Run **§4 · Doctor**. Spend GPU minutes only if this looks right:

- `cuda: yes` (and a Tesla T4 name),
- every stage **ready** — with §2 done (and §3 if you want FBX), nothing should say `blocked`.

Anything still `blocked` here will **fail loudly** when the run reaches it. Fix it before continuing.

## Step 5 — Give it inputs (run ONE cell in §5)

A DRISHTI input is a small `mission.yaml` **descriptor** plus the files it points at (a drone video + a
GPS/telemetry track; IMU/baro/intrinsics/RTK are optional). Pick **one**:

- **Option D — generate here (easiest first run).** No upload. Set `DATASET` and run:
  - `aukerman` — real buildings (~543 MB), the meatiest demo;
  - `brighton_beach` — small & fast (~62 MB), good for a first end-to-end pass;
  - `synthetic` — a ground-truth city (known geometry → accuracy is *checkable*).

  It builds the real `MP4 + SRT` from open, GPS-tagged imagery (only playback timing is synthesized) and
  sets `MISSION` for you.

- **Options A/B — your own mission.** Uncomment one: mount **Google Drive** (best for a large video) and
  set `MISSION` to your descriptor, or use the **upload** widget for small files. Make sure the files the
  descriptor references are present too.

- **Option C — resume a partial bundle** you started locally (e.g. S0/S1 ran, then S2 failed for lack of
  GPU deps). Set `RUN_OPTION_C = True`. **Important:** the runner re-checks S0's inputs, so the original
  video + telemetry must exist at the paths the descriptor names — regenerate them (if they came from
  `make_sample_dataset.py`, same paths) or upload/mount your originals **before** resuming.

## Step 6 — Run the whole pipeline

Run **§6**. It executes all 11 stages on the GPU with the **`balanced`** profile, streams the log live,
and **tees the entire console into `logs/run_console.log`** inside the bundle so it downloads with
everything else. The first run also fetches the model checkpoints (RT-DETR, Depth-Anything-V2; both
Apache-2.0).

- The cell prints `[exit 0]` and the bundle path when it finishes.
- **If a stage fails** (that's the honest, fail-loud behaviour), read the error, fix the cause, then set
  `RESUME_BUNDLE = BUNDLE` and re-run the cell — completed stages are skipped, so it picks up where it
  stopped.

> **Why `balanced` and not `max`?** `max` selects reconstruction methods this build doesn't implement
> (`gaussian` / `2dgs`), which **raise loudly** at `s7_dense` / `s8_mesh`. For a higher-quality run on
> the implemented path, use `--profile max --set dense.method=tsdf --set mesh.method=poisson`.

## Step 7 — Confirm what happened

Run **§7 · Inspect + verify** and **§8 · Logs & report**. This is the ground truth (not a progress bar):

- `drishti inspect` — per-stage status, where it ran, wall-time, **measured** confidence, any degradation.
- `drishti verify` — recomputes every output hash against the manifest.
- §8 lists `logs/`, tails `run_console.log`, and prints the `report.json` summary.

## Step 8 — Get your results out (EXPORT)

Run **§9 · Export**. It builds two archives and downloads one:

- **full** — the complete bundle: `manifest.json`, `inputs/`, `logs/`, `report/`, and every stage dir
  (frames, depth, dense cloud, mesh, and all `s10_export` deliverables). Everything, reproducible.
- **slim** — just deliverables + report + logs + manifest, for a quick share.

For a very large `full` zip, use the commented **copy-to-Drive** lines instead of a browser download.

**What's in the bundle** — the required deliverable set: **OBJ · PLY · LAS · GeoTIFF · glTF · FBX**
(FBX only if you ran §3), a DSM/DTM + orthomosaic, an accuracy report (`report/report.html`), and the
full manifest.

## Step 9 — Back on your own machine (view / serve)

You ran everything on the T4, so locally you only need to look at it — **no heavy deps**:

```bash
unzip <run_id>_full.zip -d runs/
drishti inspect runs/<run_id>            # the same manifest you saw on the T4
drishti verify  runs/<run_id>            # confirm the download is intact

pip install -e ".[server]"
python -m server.app                     # http://localhost:8000/api/health
cd viewer && npm install && npm run dev  # browse the georeferenced 3D model
```

`report/report.html` is the shareable accuracy report; `s10_export/` holds the model files.

---

## Alternative: tier-split (`drishti_cloud_t4.ipynb`)

Use this if you *do* have the light extras locally and want to keep the CPU stages on your machine. It
runs the GPU-heavy stages on the T4 and stops after dense:

```bash
# on the T4 (inside the notebook):
drishti run --dataset "$MISSION" --profile max --set dense.method=tsdf --set mesh.method=poisson --upto s7_dense
```

Then download the bundle and finish locally — `resume` reuses the config recorded in the bundle and skips
the GPU stages already marked done:

```bash
unzip <run_id>.zip -d runs/
drishti resume runs/<run_id>             # continues at s8_mesh; the GPU stages are already 'done'
```

---

## Honest gotchas (so nothing surprises you)

- **Colab runs Python 3.13; Open3D has no 3.13 wheel.** Open3D (used by `s7_dense` TSDF and `s8_mesh`
  Poisson) publishes wheels only through cp312, so a naïve `pip install .[recon]` fails on Colab and,
  because it's one atomic command, takes the whole install down (`drishti: command not found`). §2 fixes
  this by building an isolated **Python 3.12** env with `uv` and routing the notebook through it. If `uv`
  can't fetch a Python in your session, the alternative is `condacolab` (ask and I'll wire it in).
- **`--profile max` crashes today.** It picks `gaussian` / `2dgs`, which aren't implemented; both cloud
  notebooks already override those to TSDF + Poisson. See [`../docs/GUIDE.md`](../docs/GUIDE.md) §5.
- **`transformers` / `laspy` aren't in any `pyproject` extra.** The notebooks install them explicitly;
  if you build your own environment, add them or S3/S4 and the LAS export will be blocked/degraded.
- **FBX needs Blender ≥ 4.0.** Without §3 it's skipped and `s10_export` is degraded (not failed).
- **Resuming (Option C) re-checks S0's inputs.** The original video + telemetry must be present at the
  recorded paths, or S0 fails loudly on the missing video.
- **Nothing is faked.** Missing dependencies, no GPU, or missing inputs **fail loudly** with a message
  saying exactly what's wrong — that's the design, not a bug.
