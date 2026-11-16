# State-space arm: code map

How the state-space scripts (`scripts/state_space_*.py`) fit together: where each quantity is defined, how to run
them, and the conventions to keep. The science, results and open items are elsewhere:

- plan: `docs/STATE_SPACE_RECOVERY_PLAN.md`
- handoff: `docs/status/STATUS_2026-11-07_STATE_SPACE_HANDOFF.md` (newest dated block first)
- results page: the State-Space Recovery artifact

Consolidated 2026-11-13 (branch `refactor/ss-cleanup-1113`); the section at the end lists what moved.

## 1. One definition per quantity

| Quantity | Defined in | Used by |
|---|---|---|
| Every input and output path; the run tags XTAG / WTAG; ROOT (outputs) | `state_space_sim` (top block) | every script |
| `check_root()` | `state_space_sim` | `load_sessions`, `prefill`, `page_figs.main` |
| `POS` / `POS_LABEL` / `RINGS` (canonical spout order = `orofacial_cohort.POS_ORDER`) | `state_space_sim` | every script |
| `EPOCHS` / `POST` | `state_space_sim` | every script |
| Area sets `OROFACIAL` / `MOTOR` / `SHIFT_REGIONS` | `state_space_sim` | balance and shift scripts (see section 4) |
| Basis switch (`BASIS` allen / lnmf / svd) | `state_space_sim` | every script |
| Alignment (`ALIGN`, `SEGMENTS`, `BAL_BINS`) | `state_space_sim` | every script |
| Per-trial transforms (`XFORM`) | `state_space_sim` | every script |
| THE session loader `load_sessions(hook)` | `state_space_sim` | every script except `sim.run` (section 4) |
| Readouts `readouts`, `fit_gain`, `topk`, `spout_means`, `hemi`, `evoked` | `state_space_sim` | |
| Session-vs-session readouts `pair_readouts`, `matched`, `template` | `state_space_recovery_trials` | |
| Hemispheric balance `balance_sides` / `balance` (FINDINGS 0b) | `state_space_recovery_trials` | `balance_cells`, `recovery_trials` (bal_*) |
| Cohort SVD basis, split areas, region weights, pixel-RMS unit, ROI outlines | `state_space_svdmaps` | 8 map scripts |
| RESTnl level / ENL trials per session: `rest_level`, `enl_trials` | `state_space_chronic_maps` | `shift_*`, `chronic_maps` |
| Change vs own pre, per epoch (animal-weighted nested bootstrap) | `analysis_kit.epoch_change` (on `boot_delta`) | every table and figure with a change |
| Within-animal shuffle p: `permute_within`, `within_shuffle_p` | `analysis_kit` | step_vs_gradual, per_position, detour_vs_balance, within_session_behavior, shift_precue, shift_rawref |
| LMM backup (dagger) | `analysis_kit.epoch_change_lmm` | `page_figs.lmm_p` |
| Stars in the top band, no top spine | `sig_marks.star_top`, `reserve_headroom` | figures |
| PNG + svg twin | `figure_layout.save_png_svg` | figures |

## 2. Environment switches

All are read at import time, so set them per process; never inside one.

| Variable | Values (default first) | Effect | Output suffix |
|---|---|---|---|
| `WF_SS_BASIS` | `allen`, `lnmf`, `svd` | which per-trial files are read; `svd` is needed by the map scripts | `_lnmf` / `_svd` |
| `WF_SS_ALIGN` | `cue`, `cue25`, `lick`, `splice` | the time bins (`SEGMENTS`) | `_<align>` (cue = none) |
| `WF_SS_K` | `8`, … | dimension of the pre-stroke top-k space | `_k<K>` |
| `WF_SS_XFORM` | `""`, `z`, `gsr`, `spout` | per-trial transform at load | `_x<name>` |
| `WF_SS_ROOT` | unset | output root for VERIFICATION runs only (refactor pins); never in a production queue | |
| `WF_SS_LNMF_MASK` | `brain` | which LocaNMF bundle mask | |
| `WF_SS_MAPBASE` / `WF_SS_MAPWIN` / `WF_SS_FR` / `WF_SS_VERSIONS` / `WF_SS_NOSTATS` / `WF_SS_VMAX` / `WF_SS_PREMIN` / `WF_SS_PREMAX` | see `state_space_chronic_maps` docstring | map baseline / window / movement regression / versions / quick look | in the map names |
| `WF_SS_SHIFTBASE` | `""`, `rest`, `enl` | `shift_carried`'s reference | |
| `WF_SS_LOOPS` | `""`, `aligned` | `trajectories` loop alignment | |

The arm runs WITHOUT the rebuild render profile. The profile's write fence refuses `state_space_recovery` (queue token
`noprofile`, v20u). The inputs are still the rebuild run's (`state_space_sim.PROFILE`).

## 3. Scripts by role

**Core (imported, not run):**
- `state_space_sim`
- `state_space_recovery_trials` (also a production script)
- `state_space_svdmaps`
- `state_space_chronic_maps`: loaders, also the map script
- `state_space_page_figs`: figure helpers, also the page figures

**Production (Allen basis), the v20u queue.** Run order:
1. `fill`: `S.prefill`, the binned cache
2. `recovery_trials`, with `--condition on_target | landing | contact | incomplete`, and the k / xform variants
3. `path_tests` (`--trial-alpha` after it)
4. `repertoire_power`
5. `trajectories`
6. `pertrial`, then `per_position` and `within_session_behavior`
7. `step_vs_gradual`, after trials and trajectories
8. `contact`
9. `noise`
10. `balance_cells` (`--motor`)
11. `balance_angle --collect`, then `balance_angle` (`--motor`)
12. `sim`, the method check
13. `change_parts` (`--sessions`)
14. `page_figs --only …`

Cue and lick alignments each run steps 2–14; splice runs trials and trajectories only. Inputs needed first:
`state_space_landing` (`landing/first_lick_landing.csv`).

**SVD map family (`WF_SS_BASIS=svd`):**
- baselines first: `rest_baseline`, `enl_baseline`
- maps: `chronic_maps`, `part_maps`, `june_maps`, `why_cleaner`
- shift analyses: `shift_behavior`, which writes the trial table the others read; then `shift_carried`, `shift_precue`, `shift_rawref`, `shift_spout_timing`, `shift_movement`, `shift_recovery`
- others: `rest_decomp`, `fr_coupling`, `encoder_subtract`, `encoder_window_kbs`

All are run by hand from the handoff's commands. None is in a queue yet.

**Diagnostics, run once by hand:**
- `detour_vs_balance`
- `orth_check`
- `june_drift`

**Archived 2026-11-13 (Priya OK):** `state_space_recovery` (the first-pass encoder-PETH version, 58 areas) and `pc2` (a
one-off probe) are in `scripts/archive/` with a line each in its README.

Every script's module docstring names its question, inputs, outputs and readouts. Run any of them as
`python scripts/state_space_<name>.py [flags]` from the repo root, with the environment from section 2.

## 4. Conventions that change numbers if broken

- **Three different "orofacial" area sets.** They stay distinct; merging them moves published numbers.
  - `OROFACIAL` = SSp-m, SSp-n, SSs: the balance.
  - `MOTOR` = ALM, tjM1: the motor balance.
  - `SHIFT_REGIONS["orofacial"]` = tjM1, SSp-m, SSp-n: the shift maps.
- **Area-set names (Priya 11-16).** "Orofacial" alone means SENSORIMOTOR (`SHIFT_REGIONS["orofacial"]` = tjM1, SSp-m,
  SSp-n). The balance sets say what they are: `OROFACIAL` = orofacial SENSORY (SSp-m, SSp-n, SSs), `MOTOR` = orofacial
  MOTOR (ALM, tjM1). Figure / page text takes the name from `S.AREA_SET_NAME`; identifiers and columns (`OROFACIAL`,
  `bal_oro*`, `*_motor`) predate the rule and stay.
- **Hemispheric balance only on real responses** (`recovery_trials.balance_defined`, Priya 11-16). The balance is
  log2 (contra / ipsi) of response SIZES, so it exists only when both sizes are positive: raw and `WF_SS_XFORM=z` (the
  per-area scale cancels in change vs own pre; lick bal_oro +.11 / +.19 / +.21 vs raw +.13 / +.19 / +.20). Under `gsr`
  (cortex-wide mode removed) and `spout` (cross-spout mean removed) a size becomes a difference from a shared pattern,
  can be <= 0, and the log ratio is undefined or explodes (11-16: spout 30-83 % of sessions valid, far-ring values
  near -5.5; gsr pre spread +-.9 vs +-.3), so those tables carry no `bal_*` columns. Removing the cortex-wide mode is not
  needed for the balance in any case: a log RATIO is invariant to a bilateral gain by construction, and the balance
  shift does not follow the cortex-wide amplitude: within animal, post-stroke sessions, r(amplitude ratio, bal_oro) =
  +.11 / +.74 / -.54 / -.31 (PS92-95, lick; far ring -.19 / +.74 / -.22 / -.42), within-animal slope -.002 (lick) /
  +.18 (cue, carried by PS93): no consistent sign.
- **Shuffle p** = share of within-animal shuffles with |null| ≥ |observed|, two-sided, no +1. It is not
  `lick_templates.perm_p`, which is one-sided with the +1.
- **Draw order.** `epoch_change` takes one `boot_delta` per epoch, in order, on the caller's rng. A loop that
  interleaves epochs with other draws passes `epochs=[ep]`. Do not reorder calls on a shared rng.
- **`sim.run` keeps its own pre-session loop** (file order; `load_sessions` sorts by day). Switching it would move the
  method-check detection limits.
- **Epochs come from the extract**, baked in at extraction, plus the June override. They are not re-derived from
  `epoch_boundaries.json`.
- **Units.** `svdmaps.pre_unit_px` (pixel RMS of the pre template's spout deviations), `chronic_maps.full_unit` (RMS of
  the pre mean response) and `change_parts`' unit are different quantities.
- **Caches.**
  - The binned cache is keyed by the analysis-mask signature (`S._bcache`).
  - `chronic_maps_resp/<SVD bundle>_v<RESP_VERSION>`: bump `RESP_VERSION` after changing how responses are built
    (CLAUDE.md rule 5).

## 5. Verifying a refactor

Run the affected steps with `WF_SS_ROOT=<scratch>`; copy `cache/` and the `landing*/` tables there first. Then compare
every table with the production outputs:
`python docs/status/ss_rerun_1112/pin_compare.py <scratch> <production root>` (analysis/attempt-coding branch).
Pin new shared helpers with exact-equality tests against frozen copies of the code they replace
(`tests/test_epoch_change.py`).

## 6. What the 11-13 consolidation moved

- **Change vs pre:** 9 copies became `analysis_kit.epoch_change`.
- **Within-animal shuffles:** 6 copies became `analysis_kit.within_shuffle_p`.
- **Map setup:** 8 inline copies of the SVD basis / split areas / region weights / pixel-RMS unit / ROI outlines became
  `state_space_svdmaps`.
- **RESTnl / ENL loads:** 5 inline copies became `chronic_maps.rest_level` / `enl_trials`.
- **Constants:** epoch lists, spout labels, area sets and bundle paths are defined in `state_space_sim` only.
- **Balance:** `balance_cells` uses `recovery_trials.balance_sides`; `shift_spout_timing` uses `S.spout_means`.
- **Parallel reads:** `landing.build` and `shift_movement.face_running` read their files in parallel, in file order.
- **Cache key:** `chronic_maps`' response cache is keyed by bundle and version.
- **Archived:** `state_space_recovery.py`, `state_space_pc2.py`.

**Verified 2026-11-13.** Every Allen-basis step of the v20u queue was re-run on the new code into a scratch root
(v20v): 137 of 138 tables were identical to production. The SVD map family was run on the old and the new code: all 38
tables were identical. The one difference is a correction, not a change of result: `n_animals` of `bal_oro_far` (lick
alignment, spout transform) had counted animals with post-stroke sessions but no pre value; it now counts the animals the
bootstrap uses (acute 2 -> 1, subacute 3 -> 2, chronic 3 -> 1). Deltas and intervals are unchanged.
