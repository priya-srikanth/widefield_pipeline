# CLAUDE.md

Project-level instructions for Claude Code. Read in full at the start of every session in this repo.

`widefield_pipeline` = widefield calcium-imaging **preprocessing + LocaNMF spout-position decode/encode
analysis** for the Sabatini-lab VLS-stroke study (mice PS92/PS93/PS94/PS95; PS93 has a right orofacial
deficit). Carved out of `Widefield_DAQ_recorder` on 2026-08-08 (that repo keeps only the DAQ recorder GUI +
camera acquisition). See `README.md` for setup, `docs/MIGRATION.md` for the split record.

---

## Ground rules (non-negotiable)

1. **Never modify/delete source data.** Raw imaging, DAQ `.h5`, the imaging computer's preprocessing outputs
   on MICROSCOPE (`motion_corrected/`, `wfield_local_results/`, `SVTcorr.npy`, `U_atlas.npy`,
   `*cleanpairs_frame_map.npz`), behavior logs, and the cam1 recovered-position CSVs are READ-ONLY inputs.
   Only ever write inside `MICROSCOPE/Priya/…`; never another person's folder; never delete on MICROSCOPE.
   Never delete local staging (`D:`/`E:`) until byte-verified and the user confirms.
2. **Never modify the `Widefield_DAQ_recorder` repo** from here — it is the separate recorder GUI. Zero
   cross-imports; the two talk only through files on MICROSCOPE.
3. **`configs/*.yaml` is the single source of truth.** Register sessions in `configs/sessions.yaml`, animals/
   colors/date-policy in `configs/animals.yaml`, params in `configs/defaults.yaml`, mounts in
   `configs/paths.yaml`. The hardcoded `SESSIONS`/`ANIMAL_COLOR`/`L`/`D` are RETIRED — do not re-add them.
   MMDD dates are QUOTED strings (`0606` unquoted parses as octal 390).
4. **Rig commit procedure:** `export CONDA_PREFIX=C:/Users/sabatini/.conda/envs/locanmf`; `git add -A` →
   commit → `git fetch origin` → `git rebase origin/main` → `git push`. NEVER force-push; re-fetch/rebase if
   rejected. Both machines push `main`, so always fetch/rebase first.
5. **Cross-day caching:** per-session results are memoized (`wfield_local/session_cache.py`). **Bump
   `CACHE_VERSION` whenever you change a cached function's logic** (mtimes don't see code changes).
6. **Per-machine envs differ by design** (README "Per-machine environments"): imaging box = `wfield` env,
   numba + numpy<2.1; this box = `locanmf` env, numpy 2.2.6, no numba. Deps are lower-bounds-only so
   `pip install -e .` never force-upgrades a working stack.

## Architecture (two machines)

- **Imaging (PCO) computer** — acquisition (recorder GUI, separate repo) + preprocessing (motion/SVD/Allen,
  cue/lick maps) from THIS repo → uploads to MICROSCOPE. Mounts: `N:`=MICROSCOPE, `E:`=local, `M:`=standby
  (`M:\collaborations\Priya\Widefield\labcams`). Runbook: `runbooks/imaging_computer_nightly.md`.
- **Analysis / behavior GPU box (this)** — LocaNMF + decode/encode/RSA + decks. `M:`=MICROSCOPE. Orchestrator
  `python -m wfield_local.nightly_figs <MMDD>`. Runbook: `runbooks/analysis_computer_nightly.md`.

## Key decisions (see LOCANMF_LICK_CUE_ANALYSIS.md / DECISIONS.md for the full set)

- Decode = multinomial logistic regression on individual LocaNMF components, first-lick 2 s, NO per-trial
  baseline, block-CV (GroupKFold by ~6-trial position blocks), chance 0.167. SSp dominates, MO secondary.
- Pre-cue no-lick decode above chance = motor-independent maintained code (the key pre-stroke readout).
- Positions from DAQ strobe bits; dead bit1 (Aug-2026) auto-repairs from the behavior log
  (`classify_cues_with_backup`), or a `behavior_trials` recovered CSV when the log is empty (PS93 8/5, cam1).
- The DAQ cue/strobe stream tracks the REWARDED subset (reward held after ~6 misses) → an engagement filter.
  Keep it; unrewarded trials are for the FUTURE post-stroke failed-attempt analysis (movement-gated).
- Cross-session comparisons use the CURATED set: 6/6-6/8 + 8/6 onward (exclude noisy early June 6/1-6/5 and
  the wonky 8/5), from `configs/animals.yaml date_policy`. Crossnobis is the noise-unbiased RDM metric.

## Restructure roadmap — mirror `../stroke_orofacial_pipeline`

Target = that repo's mature config-driven layout. **DONE:** `configs/{animals,sessions,paths,defaults}.yaml`;
config loader (`wfield_local/config.py` ≈ their `config_loader.py`+`animals.py`); `tests/`; installable
packaging (`pyproject`/`environment.yml`/`requirements`, layers onto a prebuilt env); `README`; `runbooks/`;
incremental per-session caching; `docs/MIGRATION.md`; this `CLAUDE.md`.

**NEXT (priority order):**
1. **Consume `defaults.yaml` in code.** LocaNMF params (r2/loc/maxrank), decode windows/CV/max_rt, sync
   params are still hardcoded in module `_args()`/constants — wire them to `config.defaults()` so params live
   in ONE place. Add `session_overrides.yaml` (per-session param overrides layered on defaults), mirroring
   their defaults+overrides pattern.
2. **PathResolver** (mirror their `paths.py`): logical roots → platform mounts; store `sessions.yaml` as
   root+relative paths instead of full `M:` paths, so the imaging box (`N:`/`E:`) and this box (`M:`) resolve
   the SAME config. Currently `sessions.yaml` holds full `M:` paths (this-box-only).
3. **`.githooks/`** (pre-commit/pre-push running `ruff` + `pytest`), mirroring theirs; wire via
   `git config core.hooksPath`.
4. **Fold the ~84 legacy `_*_run.py` drivers + `_*.json` state into `scripts/`** and a config-driven
   orchestrator; retire the per-date one-offs (the imaging computer's `_mc_svd_*`/`_maps_*`/`_nightly_*`).
   NB some `_*.json` are read by relative path by the imaging computer's active pipeline — coordinate.
5. **Optional `src/` layout** — move `wfield_local/` under `src/` and/or split into submodules like their
   `src/pkg/{alignment,figures,stats,…}`. KEEP the `wfield_local` import name (both machines + docs depend
   on `python -m wfield_local.*`).
6. **`_writeguard.py`-style guard** against overwriting MICROSCOPE source data; **exclusions with dotted-tag
   scoping** (their `Exclusion.applies`) for per-analysis-context date/animal exclusions.
7. **More tests + CI** (session enumeration, animals, param loading).

**Then the science:** post-stroke prerequisites — per-trial behavioral-state table (spout-contact + DAQ lick
→ hit/miss/failed, latency, executed position), and packaging the frozen pre-stroke model + baseline
noise-floor — followed by the post-stroke intention-readout (frozen decoder) and representational-similarity
(crossnobis / encoder-residual) analyses. Design in `LOCANMF_LICK_CUE_ANALYSIS.md`.

## Reference
- `configs/` — source of truth. `wfield_local/config.py` — loader. `wfield_local/nightly_figs.py` — orchestrator.
- `runbooks/` — the per-machine nightly prompts (Priya's canonical prompts + notes).
- Env: `C:/Users/sabatini/.conda/envs/locanmf/python.exe`; repo is `pip install -e .` (no PYTHONPATH).
- Deck: built into `C:/Users/sabatini/source/cue_lick`, copied to
  `M:\MICROSCOPE\Priya\Widefield\labcams\locanmf_lick_pooled\cue_analysis\spout_position_decoder_summary.pptx`.
