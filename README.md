# widefield_pipeline

Widefield calcium-imaging **preprocessing + LocaNMF spout-position decode/encode analysis** for the
Sabatini-lab VLS-stroke study (mice PS92/PS93/PS94/PS95). Carved out of `Widefield_DAQ_recorder`
(which keeps the live DAQ-recorder GUI + camera acquisition) on 2026-08-08.

## Two-machine architecture

The pipeline is split across two computers by lifecycle + hardware:

| Stage | Machine | What runs | Runbook |
|---|---|---|---|
| Acquisition | **imaging computer** | DAQ recorder GUI + cameras (`Widefield_DAQ_recorder` repo) | — |
| **Preprocessing** | **imaging computer** | motion correction → SVD/wfield → Allen alignment → cue/lick maps → upload to MICROSCOPE | [`runbooks/imaging_computer_nightly.md`](runbooks/imaging_computer_nightly.md) |
| **Analysis** | **this / behavior GPU box** | LocaNMF → decode/encode → cross-mouse/RSA → deck → push | [`runbooks/analysis_computer_nightly.md`](runbooks/analysis_computer_nightly.md) |

The two stages talk only through **files on MICROSCOPE** (the imaging computer writes
`motion_corrected/`, `wfield_local_results/`, `SVTcorr.npy`, `U_atlas.npy`; the analysis box consumes
them). There are **no cross-imports** between this repo and the recorder repo.

## Layout

```
wfield_local/          the pipeline package (motion/SVD/Allen preprocessing + LocaNMF + decode/encode/RSA/decks)
  nightly_figs.py      analysis orchestrator: python -m wfield_local.nightly_figs <MMDD>
configs/               single source of truth (mirrors stroke_orofacial_pipeline)
  animals.yaml         per-animal metadata + characteristics (e.g. PS93 right orofacial deficit) + date policy
  sessions.yaml        per (animal, date) imaging session: mc / h5 / regime / behavior_trials overrides
  paths.yaml           MICROSCOPE + labcams + DAQ + behavior mounts (logical roots -> platform paths)
  defaults.yaml        analysis params (LocaNMF r2/loc/maxrank, decode windows/CV, sync, lick detection)
runbooks/              the per-machine nightly runbooks (source of truth: imaging_computer_nightly / analysis_computer_nightly)
docs (*.md at root)    DECISIONS (analysis decisions + findings), STROBE_BIT1_RECOVERY, CLAUDE; docs/archive/ = retired one-offs
_*.py / _*.json        legacy per-session drivers + state (to be folded into the config-driven flow — see "Roadmap")
```

## Setup on a new machine

```bash
git clone https://github.com/priya-srikanth/widefield_pipeline.git
cd widefield_pipeline
conda env create -f environment.yml      # env "locanmf" (python 3.10); runs `pip install -e .`
conda activate locanmf
```

Then install the **GPU / custom pieces** (not pip-resolvable — see [`docs/archive/GPU_LOCANMF_KICKOFF.md`](docs/archive/GPU_LOCANMF_KICKOFF.md)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build (RTX 4060: 2.6.0+cu124)
pip install wfield==0.6.0
# LocaNMF (localnmf) + cuhals: custom Windows build; patches in wfield_local/*.patch
```

Mount MICROSCOPE as `M:` (`net use M: \\research.files.med.harvard.edu\Neurobio`). `pip install -e .`
means **no `PYTHONPATH=` hack** is needed — `python -m wfield_local.<module>` works from anywhere.

**`meegkit`** is required for the hemodynamic-correction variants (`hemo_variants`, built per session during
preprocessing). It is a declared dependency (`pyproject.toml`), so `pip install -e .` installs it; on a
prebuilt env you don't re-resolve, add it explicitly: `pip install meegkit` (pulls statsmodels/patsy/
pyriemann/array-api-compat — **does not touch numpy/numba**, safe on the imaging box's `numpy<2.1` pin).

### Per-machine environments (they differ — this is expected)

This package deliberately declares **lower-bound-only** dependencies (see `pyproject.toml`) and does **not**
pin the heavy/GPU/custom pieces, because it **layers onto a machine's prebuilt env** rather than defining
one. Different machines / use-cases legitimately run **different package combinations**, and
`pip install -e .` must not force-upgrade a working stack. The two production envs today:

| | **Imaging computer** (`wfield` env) | **Analysis / behavior GPU box** (`locanmf` env) |
|---|---|---|
| Runs | acquisition + preprocessing (motion/SVD/Allen, cue/lick maps) | LocaNMF + decode/encode/RSA + decks |
| numpy | **< 2.1** (e.g. 1.26.4) — pinned indirectly by numba | **2.2.6** |
| numba | **required** (motion correction); numba 0.60 caps numpy `< 2.1` | **not installed / not needed** (nothing on the analysis path uses it) |
| GPU/custom | wfield | torch (CUDA, cu124) + wfield + LocaNMF/localnmf + cuhals |

**Two more envs on the analysis box, for the orofacial keypoint work.** Neither is on the nightly
path; both are entered by hand for a DLC round. They are SEPARATE FROM `locanmf` AND FROM EACH OTHER,
and the reason is measured rather than cautious — see below.

| | **`dlc`** | **`lp`** (added 2026-09-26) |
|---|---|---|
| Runs | DeepLabCut 3.0.1 — `wfield_local.dlc_*` (project, frames, prelabel, train, prior, guides) | Lightning Pose 2.4.2 — temporal / pose-PCA / multi-view losses |
| python | 3.10.20 | 3.10.20 |
| numpy | 2.2.6 | 2.2.6 |
| torch | **2.11.0+cu128** | **2.11.0+cu128** |
| also | aniposelib (calibration) | lightning 2.5.6, transformers, jax, kornia, aniposelib |

**`torch` must be cu128 on this box and that is not optional.** The GPU is an RTX 5060 (Blackwell,
`sm_120`, 8 GiB). A cu124 build does not have kernels for it. Install torch FIRST, from the cu128
index, and let the framework install on top — both DeepLabCut and Lightning Pose declare `torch`
loosely enough (`torch <3.0.0`) that they keep whatever is already there. Verified 2026-09-26:
`lightning-pose` installed over a pre-existing cu128 torch without touching it.

```powershell
conda create -n lp python=3.10 -y
conda activate lp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # FIRST
pip install lightning-pose
pip install -e . --no-deps          # makes this repo importable; --no-deps per the rule above
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

**Why `lp` is not just installed into `dlc`.** Measured, not assumed: the `dlc` env has no
`lightning` at all, so `pip install lightning-pose` there would add lightning 2.5.6,
pytorch-lightning, transformers, jax/jaxlib, kornia and ~60 other packages on top of a working
DeepLabCut 3.0.1 stack. That is precisely the "must not force-upgrade a working stack" case this
section exists for. The two frameworks also disagree about who owns the training loop, so sharing an
env buys nothing.

**They share data, not code.** Lightning Pose reads a CONVERTED copy of the DLC labelling project;
the DLC project on the share stays the single source of truth for labels, exactly as
`dlc_train.stage()` treats its own training copy. Nothing in `wfield_local` imports
`lightning_pose` at module scope, so the `dlc` and `locanmf` envs keep working without it.

Key rule: **`numba` self-pins the numpy upper bound**, so do not raise the numpy floor here. If a machine
needs numba on a newer numpy, install a numba build that supports it (≥ 0.61) rather than downgrading numpy.
If an `import` breaks after an install, check for **user-site shadows** (`AppData\Roaming\Python\...\site-packages`)
overriding the conda versions and remove them. Keep each machine's env separate; there is no single lockfile
that fits both (the `requirements.txt` pin set is the *analysis-box* reference, not a cross-machine contract).

## Caching (cross-day analyses)

Per-session results (decode recall/EV, RDMs, crossnobis, hemisphere) are memoized to disk by
`wfield_local/session_cache.py`, so the cross-day figures only compute new/changed sessions (a cache hit is
~340x faster) and don't recompute the same session across cross-mouse/within-animal/RSA within one run.
- Cache dir: `WIDEFIELD_SESSION_CACHE` (default `C:\Users\sabatini\source\.widefield_session_cache`), outside the repo.
- Auto-invalidates when a session's LocaNMF `C.npy` / h5 / `behavior_trials` changes (e.g. a LocaNMF re-run).
- Force a full recompute with `WIDEFIELD_NO_CACHE=1`; clearing = delete the cache dir. **Bump `CACHE_VERSION`
  in `session_cache.py` whenever you change a cached function's logic** (mtimes don't see code changes).

## Running the nightly

- **Imaging computer:** follow `runbooks/imaging_computer_nightly.md` (acquire → preprocess → upload).
- **Analysis box:** follow [`runbooks/analysis_computer_nightly.md`](runbooks/analysis_computer_nightly.md)
  (camera + behavior, then LocaNMF + `nightly_figs`, then push the deck) — the source of truth for this box.

## Key docs

- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — what the code is, in layers, with a diagram:
  162 modules, 109 of them runnable, grouped by pipeline stage. Read this before the module list
  below. It also names the three seams that keep causing bugs here.
- **[`runbooks/first_full_analysis.md`](runbooks/first_full_analysis.md)** — a first run from raw
  sessions to the deck, for someone who has not done it before. The other runbooks assume you have.
- **[`docs/status/`](docs/status)** — sixteen dated handoffs, with an index of what each settled.

- [`runbooks/analysis_computer_nightly.md`](runbooks/analysis_computer_nightly.md) — analysis-box nightly runbook (source of truth)
- [`runbooks/imaging_computer_nightly.md`](runbooks/imaging_computer_nightly.md) — imaging-computer preprocessing runbook (source of truth)
- [`DECISIONS.md`](DECISIONS.md) — analysis decisions + findings F1–F17 (decode/encode/RSA) + stroke plan + server layout/regimes
- [`STROBE_BIT1_RECOVERY.md`](STROBE_BIT1_RECOVERY.md) — dead-strobe-bit position recovery (behavior-log + cam1)
- [`docs/GUI_TRIALS_LOGGING.md`](docs/GUI_TRIALS_LOGGING.md) — the GUI `trials.csv` `pos_idx` bug and why
  behavior trials are sourced from the DAQ recorder `.h5` (DAQ primary, behavior log fallback)

## Roadmap

1. **Config consumption** — refactor `wfield_local` to read `configs/*.yaml` (replace the hardcoded
   `SESSIONS` in `locanmf_cue_lick_analysis.py` with a loader), add `tests/` for the config loader +
   animals, mirroring `stroke_orofacial_pipeline`.
2. ~~Incremental cross-day analysis (efficiency)~~ — **DONE** (`wfield_local/session_cache.py`): the
   per-session compute in cross-mouse / within-animal / RSA / crossnobis / hemisphere is memoized to disk,
   keyed by the LocaNMF `C.npy` + h5 + `behavior_trials` mtimes + params + `CACHE_VERSION`. Only new/changed
   sessions recompute (~340x on a cache hit); also dedupes within a single run. See "Caching" above.
3. **Post-stroke analysis** — intention readout (frozen decoder) + representational similarity
   (crossnobis / encoder residuals); prerequisites in [`DECISIONS.md`](DECISIONS.md).
4. Fold the legacy `_*_run.py` drivers into the config-driven orchestrator; then `scripts/` cleanup.
