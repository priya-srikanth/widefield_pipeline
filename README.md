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
runbooks/              the per-machine nightly prompts (codified from LOCANMF_NIGHTLY_PIPELINE.md / NIGHTLY_PIPELINE.md)
docs (*.md at root)    LOCANMF_NIGHTLY_PIPELINE, LOCANMF_LICK_CUE_ANALYSIS, STROBE_BIT1_RECOVERY, GPU_LOCANMF_*, DECISIONS, ...
_*.py / _*.json        legacy per-session drivers + state (to be folded into the config-driven flow — see "Roadmap")
```

## Setup on a new machine

```bash
git clone https://github.com/priya-srikanth/widefield_pipeline.git
cd widefield_pipeline
conda env create -f environment.yml      # env "locanmf" (python 3.10); runs `pip install -e .`
conda activate locanmf
```

Then install the **GPU / custom pieces** (not pip-resolvable — see [`GPU_LOCANMF_KICKOFF.md`](GPU_LOCANMF_KICKOFF.md)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build (RTX 4060: 2.6.0+cu124)
pip install wfield==0.6.0
# LocaNMF (localnmf) + cuhals: custom Windows build; patches in wfield_local/*.patch
```

Mount MICROSCOPE as `M:` (`net use M: \\research.files.med.harvard.edu\Neurobio`). `pip install -e .`
means **no `PYTHONPATH=` hack** is needed — `python -m wfield_local.<module>` works from anywhere.

## Running the nightly

- **Imaging computer:** follow `runbooks/imaging_computer_nightly.md` (acquire → preprocess → upload).
- **Analysis box:** follow `runbooks/analysis_computer_nightly.md`, i.e. `python -m wfield_local.nightly_figs <MMDD>`
  then rebuild + push the deck. The detailed source of truth is [`LOCANMF_NIGHTLY_PIPELINE.md`](LOCANMF_NIGHTLY_PIPELINE.md).

## Key docs

- [`LOCANMF_NIGHTLY_PIPELINE.md`](LOCANMF_NIGHTLY_PIPELINE.md) — analysis-box nightly runbook (source of truth)
- [`NIGHTLY_PIPELINE.md`](NIGHTLY_PIPELINE.md) — imaging-computer preprocessing runbook
- [`LOCANMF_LICK_CUE_ANALYSIS.md`](LOCANMF_LICK_CUE_ANALYSIS.md) — decisions + findings (decode/encode/RSA)
- [`STROBE_BIT1_RECOVERY.md`](STROBE_BIT1_RECOVERY.md) — dead-strobe-bit position recovery (behavior-log + cam1)
- [`DECISIONS.md`](DECISIONS.md) — server layout, regimes, load-bearing decisions

## Roadmap

1. **Config consumption** — refactor `wfield_local` to read `configs/*.yaml` (replace the hardcoded
   `SESSIONS` in `locanmf_cue_lick_analysis.py` with a loader), add `tests/` for the config loader +
   animals, mirroring `stroke_orofacial_pipeline`.
2. **Incremental cross-day analysis (efficiency).** The cross-mouse / within-animal / RSA (+ crossnobis)
   steps currently recompute every session from scratch each night (the slow pole). Cache each session's
   per-session outputs (recall/EV, RDM, crossnobis) keyed by session + a hash of (LocaNMF-output mtime,
   params); each night compute only the NEW session(s) and load the rest. Invalidate on LocaNMF/param change.
3. **Post-stroke analysis** — intention readout (frozen decoder) + representational similarity
   (crossnobis / encoder residuals); prerequisites in [`LOCANMF_LICK_CUE_ANALYSIS.md`].
4. Fold the legacy `_*_run.py` drivers into the config-driven orchestrator; then `scripts/` cleanup.
