# RESTnl maps for the imaging page (11-01 night -> 11-02 01:00)

Priya 11-01: RESTnl primary for the imaging page's rest-referenced maps (figures 7.1-7.11, 7.16-7.20, 8.1, 8.2), old REST
kept as a supplement, FLAT RESTnl, rule wr, AUGUST-ONLY pre. Plan: `RESTNL_ROLLOUT_PLAN_20261101.md` (map section,
Option A). Shortcut check + all-session addendum: `RESTNL_MAP_SHORTCUT_CHECK_20261101.md`.

Code: worktree `C:/Users/SabatiniLab/Github/wf_restnl`, branch `analysis/restnl-maps` (off widefield_pipeline main),
**uncommitted** (Priya commits). Nothing on M: was deleted or renamed; every output is in a NEW folder.

## 1. What was built

| piece | file | what |
|---|---|---|
| RESTnl baseline | `wfield_local/restnl_mask.py` (new) | Port of restw_strict_check mask `strict` + restnl.py rule `wr`: restdock05 rest minus every DAQ lick and every cam4 `licks_all.csv` lick, [0.5 s before, 2.0 s after], min_rest_s floor re-applied after each intersection; frames from `rest_frames_by_position(docked=False, engaged_only=True)` by in-process loader substitution (module file untouched); per-position SVT median, equal-weight mean over positions with >= 50 frames, >= 4 of 6 (strict also stored). `b_nl` (K,) and `base_map = U_atlas @ b_nl` (540 x 640, same grid / NaN zeroing as the maps). Own cache dir `C:/Users/SabatiniLab/.widefield_session_cache_restnl/` (one npz per session, key = module source + params + session_signature + rest mask + U_atlas + licks_all). `python -m wfield_local.restnl_mask [--measure]`. |
| derived reference | `wfield_local/position_reference_maps.py` | `DERIVED_REFERENCES = ("restnl",)` OUTSIDE `REFERENCES` (cache key untouched); attached in `maps_by_epoch` only when named in `WIDEFIELD_MAP_REFS`; `reference_maps(parts, "restnl") = raw - base_map`; `reference_tag -> RESTNL`; `render_references()` for the families. **Plus `WIDEFIELD_PRM_LEGACY_DATE8`** (see section 2). Default behavior unchanged (kind `prm-cue-working-79dbbe9e7a79` as before). |
| caption | `wfield_local/epoch_maps.py` | `_REF_TEXT["restnl"]` (title "RESTnl (lick-free rest, flat; rest licks excluded: DAQ contact + cam4 LP video)", the definition, the flat-vs-time-local caveat incl. the PS94 late-epoch offset, August-only pre); 15r / 15rpa loop over `render_references()`. Also fixed the stale 15r subtitle sentence "Color scale is PER ROW" (it is shared, row_scaled=False; flagged in prov_s7). |
| redraw bundles | `wfield_local/epoch_figures.py` | `panel_titles` (tuple keys) were dropped from every map bundle ("NOT serialisable"), so `--plot-only` redraws lost the per-panel sessions / n / r lines; now stored as triples and restored by `replot_map`. |
| 15r driver | `scripts/restnl_maps.py` (new) | `--arm contact|lp|enl1s [--plot-only]`: sets the arm env, `WIDEFIELD_PRE_FROM=0801`, `WIDEFIELD_MAP_REFS=restnl`, the legacy pin, `WIDEFIELD_MAP_JOBS<=4`; calls the page's own `_fig_15r_reference_maps` / `_fig_15rpa_reference_by_animal`; writes per-animal amplitude tables (RESTnl beside REST / RESTW / RAW on the same August-only sessions). |
| drift | `scripts/map_drift_prepre.py` | `--refs restnl` (sets `WIDEFIELD_MAP_REFS`), RESTnl title + lick sources + note, `--plot-only`. June sessions use their own RESTnl baselines (drift needs June by construction). |
| attempt maps | `wfield_local/attempt_maps.py` | `--rest-variant restnl`: plain-SVT features (the shared `svt:rank<K>` tf cache) minus `b_nl` in every bin (exact, flat); output `attempt_maps_restnl/`, figure suffix `_restnl`, RESTnl title; `--plot-only` alias of `--figures`. Default (`rest`) unchanged. |

Not touched: `CACHE_VERSION`, `REFERENCES`, `rest_by_position`, `quiet_periods`, `rest_engagement`, `session_raw_maps`,
`map_collect`, `defaults.yaml`. ruff clean; `tests/test_epoch_figures.py` 53 passed.

## 2. A cache trap found on the way (affects the main checkout too)

The 10-25 warm `prm` caches (`prm-*-3b1e0fc8b886`, ~33 GB) are UNREACHABLE from current `main`: commit 8e77bd0a (10-29)
changed `f"2026{mmdd}"` -> `config.date8(mmdd)` in `rest_by_position` and `rest_engagement`, both hashed into
`_raw_maps_code()`, so the digest moved (`3b1e0fc8b886` -> `79dbbe9e7a79`) although the maps are identical for cohort 1.
**The next 15r / drift render on main will rebuild every `prm` cache cold (~1.5 h + ~33 GB) unless something is done.**
Here: `WIDEFIELD_PRM_LEGACY_DATE8=1` hashes the current sources with that one replacement reversed; verified by
reproducing the on-disk kinds bit-for-bit (`3b1e0fc8b886` for precue/cue/lick, `9c0a7ab373bd` for the 1.0 s ENL
window), and it refuses any session whose `date8` is not `"2026"+mmdd` (cohort 2). Decision for Priya: adopt the pin
on main, or accept one cold rebuild.

## 3. Validation

- **Frame counts: 107 / 107 sessions EQUAL** to wf_attempt's `restw_check/sessions/<label>.npz strict__nfr_pos` (all 6
  positions, exact). The other 12 = June pre (not in restw_check).
- **The 8 check sessions reproduce the check doc exactly**: PS92_0814 nfr [269, 147, 64, 178, 59, 483]; RESTnl - RESTW
  base map p99 7.45 / 7.09 / 4.27 / 0.68 (pre) and 3.61 / 12.59 / 5.21 / 1.75 (acute) x1e-3; PS94_0818 far_R 3.23,
  flat - TL 0.48 (14.9 %).
- Coverage: 120 curated sessions; **PS93_0813 excluded (no licks_all.csv)**; rule wr fails: none (119 / 119 ok; 109 use
  6 positions, 9 use 5, 1 uses 4; strict rule would keep 109). All 12 June sessions pass with 6 positions. Under
  August-only pre the map arms hold 108 (contact) / 107 (LP) sessions, 107 with RESTnl.
- Shortcut exactness: `reference_maps` subtracts the same `base_map` from every position's cached raw map (the check
  verified the flat identity on RESTW to <= 1.3e-7); the attempt-map variant subtracts `b_nl` from every bin's features
  (one session compared against the REST version: r 0.985-0.998 per class, as expected for a level change).

## 4. Flat vs time-local on all sessions (detail in the check doc addendum)

Pre and acute confirmed (far-contra change median 2.8 % / 5.2 % of the map, no epoch dependence). **NEW: PS94
subacute + chronic drift upward within session; there a flat baseline is a COMMON level offset of 20-40 % (median) of
the far-contra map** (position contrasts unaffected). The 15r caption now says so. Consequence below (PS94 chronic).

## 5. Outputs (all new; `G` = `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures`)

Each map figure has `data/<name>_bundle.{npz,json}` + `_meta` / `_stats` / summary CSVs (figdata); redraw with
`python scripts/restnl_maps.py --arm <arm> --plot-only` / `map_drift_prepre.py ... --plot-only` /
`attempt_maps --rest-variant restnl --plot-only` (tested on ENL1s: 31 s, panel titles kept). Map figures carry
contours, not stars, so the star-band rule does not apply; no bar / dot figures were made.

| page fig | was | RESTnl version |
|---|---|---|
| 7.1 | map_REST_precue_contact | `G/epoch_restnl/epoch_15r_position_RESTNLref_precue_working.png` |
| 7.2 | drift_REST_precue_contact | `G/map_drift_prepre_restnl/contact/epoch_15rdrift_RESTNLref_precue_working.png` |
| 7.3 | map_REST_precue_LP | `G/LP_licks/epoch_restnl/epoch_15r_position_RESTNLref_precue_working.png` |
| 7.4 | drift_REST_precue_LP | `G/map_drift_prepre_restnl/LP_licks/epoch_15rdrift_RESTNLref_precue_working.png` |
| 7.5 | map_REST_enl1s_contact | `G/DAQ_ENL1s/epoch_restnl/epoch_15r_position_RESTNLref_precue_working.png` |
| 7.6 | drift_REST_enl1s_contact | `G/map_drift_prepre_restnl/DAQ_ENL1s/epoch_15rdrift_RESTNLref_precue_working.png` |
| 7.7 | map_REST_cue_contact | `G/epoch_restnl/epoch_15r_position_RESTNLref_cue_working.png` |
| 7.8 | map_REST_cue_contact_by_animal | `G/epoch_restnl/epoch_15rpa_position_RESTNLref_by_animal_cue_working.png` |
| 7.9 | drift_REST_cue_contact | `G/map_drift_prepre_restnl/contact/epoch_15rdrift_RESTNLref_cue_working.png` |
| 7.10 | map_REST_cue_LP | `G/LP_licks/epoch_restnl/epoch_15r_position_RESTNLref_cue_working.png` |
| 7.11 | drift_REST_cue_LP | `G/map_drift_prepre_restnl/LP_licks/epoch_15rdrift_RESTNLref_cue_working.png` |
| 7.16 | map_REST_lick_contact | `G/epoch_restnl/epoch_15r_position_RESTNLref_lick_lick.png` |
| 7.17 | drift_REST_lick_contact | `G/map_drift_prepre_restnl/contact/epoch_15rdrift_RESTNLref_lick_lick.png` |
| 7.18 | map_REST_lick_LP | `G/LP_licks/epoch_restnl/epoch_15r_position_RESTNLref_lick_lick.png` |
| 7.19 | map_REST_lick_LP_by_animal | `G/LP_licks/epoch_restnl/epoch_15rpa_position_RESTNLref_by_animal_lick_lick.png` |
| 7.20 | drift_REST_lick_LP | `G/map_drift_prepre_restnl/LP_licks/epoch_15rdrift_RESTNLref_lick_lick.png` |
| 8.1 | a_maps | `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/attempt_maps_restnl/figures/attempt_maps_lick_far_contra_restnl.png` |
| 8.2 | a_maps_minus | same folder, `attempt_maps_lick_far_contra_minus_pre_restnl.png` |

Extras (not on the page): 15rpa for every other arm-unit (`epoch_15rpa_position_RESTNLref_by_animal_{precue,cue,lick}_*`
in all three `epoch_restnl` folders); attempt maps for cue / precue windows and far_mid / far_ipsi (18 figures +
`maps_<align>_<animal>.npz`, `counts_*`, `hemispheres_*.csv`); per-animal amplitude tables
`<epoch_restnl>/data/restnl_amplitude_per_animal_<align>_<variant>.csv`; drift `drift_vs_stroke_panels.csv` per arm.
Every figure was looked at (15r / 15rpa / drift individually or on contact sheets; attempt maps on a sheet): titles
and color bars do not overlap data. The 15r subtitle is long (it carries the definition and caveat), but sits above
the grid.

## 6. RESTnl vs the page's REST: headline readouts, per animal first

Ratio = RMS(epoch mean map) / RMS(pre mean map), brain mask, per animal (`within_animal_pooled`, one animal at a time).
Page REST = time-local, all pre (June + Aug); RESTnl = flat lick-free, August pre. Pooled = the 15r figure's own
acute-amplitude line. Tables: `<scratchpad>/restnl/compare_far_acute.csv` and the per-animal CSVs above.

**Far-contra, ACUTE / pre:**

| arm / window | PS92 | PS93 | PS94 | PS95 | < 1 | pooled |
|---|---|---|---|---|---|---|
| contact cue (7.7) REST page | .57 | .45 | .52 | .41 | 4/4 | .48 |
| contact cue RESTnl | .59 | .52 | .60 | .43 | **4/4** | **.53** |
| LP cue (7.10) REST / RESTnl | .55 / .56 | .47 / .53 | .53 / .60 | .41 / .43 | 4/4 / 4/4 | .48 / .52 |
| LP lick (7.18) REST page | .47 | .52 | .72 | 1.01 | 3/4 | .67 |
| LP lick RESTnl | .50 | .55 | .82 | 1.07 | **3/4** (PS95 not) | **.72** |
| contact lick (7.16) | empty (< 20 contact trials acute far-contra), as on the page | | | | | |
| contact precue (7.1) REST / RESTnl | 1.74 / 2.00 | 1.16 / 1.17 | **.92 / 1.17** | 1.40 / 1.63 | >1: 3/4 / **4/4** | 1.44 / **1.80** |
| ENL1s precue (7.5) REST / RESTnl | 2.13 / 2.26 | 1.21 / 1.24 | **.82 / 1.13** | 1.40 / 1.50 | >1: 3/4 / **4/4** | 1.67 / **2.04** |
| LP precue (7.3) REST / RESTnl | 1.66 / 1.55 | .89 / .83 | .69 / .94 | 1.33 / 1.38 | >1: 2/4 / 2/4 | 1.45 / 1.59 |

**Far-contra, CHRONIC / pre** (cue): REST page .95 / 1.01 / .89 / 1.13; RESTnl 1.01 / 1.20 / **.71** / 1.21. LP lick:
REST 1.02 / 1.07 / .93 / 1.10, RESTnl 1.03 / 1.16 / **.72** / 1.15. PS94 chronic is the cell flagged in section 4
(flat-baseline level offset); RAW gives .98 / .88 for PS94, i.e. close to REST, so PS94's chronic drop under RESTnl is
most likely the baseline, not cortex.

**Drift (June -> August, no stroke) vs stroke, far-contra, RESTnl** (ratio, significant bins of 2,022): contact cue
drift .93 (20, rim-suppressed) vs acute .53 (947); LP lick drift 1.03 (5, suppressed) vs acute .72 (321); LP cue .93
(21) vs .52 (942). Page REST: .95 (7) vs .49 (1,178); 1.04 (15) vs .67 (500). Precue: RESTnl drift .71 (216 bins,
contact) / .79 (109, LP) / .68 (159, ENL1s) vs REST .67 (67) / .65 (37) / .63 (48) -- the June -> August drift of the
PRE-CUE map now reaches significance in more bins, while the acute rise (1.80 / 1.59 / 2.04) stays well beyond it.

**Attempt maps (8.1 / 8.2), lick window, far contra, whole-brain mean:** acute failed attempts below that animal's
pre-stroke hits in 3/4 (both REST and RESTnl; PS95 not), below pre-stroke failed attempts in 2/4 under RESTnl (REST
version here 3/4 on all-pre; the page states 2/4). Unchanged reading.

### Conclusions that change

1. **None of the cue / lick headlines change.** Acute far-contra cue-window loss stays 4/4 animals (pooled .48 -> .53),
   LP lick-window loss stays 3/4 (PS95 not; pooled .67 -> .72); far middle cue .66 -> .76 (contact), still not
   separable from drift. Both amplitudes move slightly toward 1 under RESTnl.
2. **Pre-cue (ENL) far-contra RISE becomes 4/4 under RESTnl** (contact 2 s and DAQ 1 s windows): PS94 flips from .92 /
   .82 (below pre) to 1.17 / 1.13; pooled 1.44 -> 1.80 and 1.67 -> 2.04. RAW agrees with RESTnl for PS94 (1.13 / 1.15).
   LP precue stays 2/4.
3. **PS94 chronic amplitudes drop under RESTnl (~.7 vs ~.9)**, carried by the flat-baseline offset of PS94's
   upward-drifting late sessions (section 4); read PS94 chronic (and pooled chronic) RESTnl amplitudes with that caveat.
4. The June -> August drift of the pre-cue maps is larger / more significant under RESTnl; acute changes still exceed it.

## 7. Timing (shared box, other sessions' jobs running; <= 4 workers; at most 2 arms at once)

baselines + flat-vs-TL measure 120 sessions 22.6 min; 15r + 15rpa contact 25 min (re-run once, 25 min, to pick up the
bundle / caption fixes), LP 25 min, ENL1s 8 min (collection on warm caches ~6-8 min per align with 4 workers, the rest
is the bootstrap); drift contact 44 min, ENL1s 9 min, LP 45 min; attempt maps (3 windows x 4 animals) 35 min; page-REST
all-pre comparison tables ~30 min (serial). Wall ~2 h 30 min (22:32 -> 00:55). Free RAM stayed >= 33 GB.

## 8. Open / for Priya

- Section 2: pin the legacy digest on main or accept a cold `prm` rebuild.
- Page edits not done (7.x / 8.x captions to RESTnl, REST to a supplement block; contact-gated 7.x figures become
  "REST baseline: LP cam4 + DAQ"); PS93_0813 has no RESTnl column (decision: drop, as done here).
- PS94 late-epoch flat-baseline offset: accept the caveat, or show RAW beside RESTnl for chronic.
