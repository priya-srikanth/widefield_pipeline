# RESTnl rollout plan: every rest-referenced figure on the two results pages (11-01)

Priya 11-01: "can we add restnl baseline for all appropriate figures (including maps)". This is the INVENTORY + PLAN only.
Nothing has been run, rendered, extracted or committed for it.

RESTnl = `scripts/restnl.py` (wf_attempt branch `analysis/attempt-coding` only; NOT on `main`): the restdock05 rest mask minus
every cam4 LP lick (`licks_all.csv`, contact or not) and every DAQ lick onset, buffer [0.5 s before, 2.0 s after], min_rest_s
floor re-applied, engagement-gated (`rest_frames_by_position(docked=False, engaged_only=True)`), per-position MEDIAN, rule
`wr` = equal-weight mean over positions with >= 50 frames, >= 4 of 6, renormalised (107 / 107 sessions). FINDINGS.md 10-29
(adoption) / 10-30 (rescue rule) / 11-01 (captions).

Classes used below:
- **(a)** rest-referenced (RESTW / REST / restdock05 rest level subtracted) and not yet RESTnl -> needs RESTnl.
- **(b)** already RESTnl (RESTW kept only as a bracketed / sensitivity number).
- **(c)** not rest-referenced -> RESTnl not appropriate (own pre-cue / PRECUE / MEAN reference, own pre-lick baseline, no
  baseline, decoders / encoders / R2 / distances). Gated TRIALS do not make a figure rest-referenced.
- **(d)** REST / QUIET FRAMES USED AS DATA, not as a baseline (rest decoders, rest-state coupling, quiet-period events, the
  rest class of the state decoder). RESTnl as a baseline does not apply; the question is whether those frames should also
  exclude LP (non-contact) licks. Listed separately with a recommendation.

## 1. Counts

| page | units scanned | (a) needs RESTnl | (b) already RESTnl | (d) rest frames as data | (c) not appropriate |
|---|---|---|---|---|---|
| Encoding goal page (`_goal_page/index.html`) | 282 figures + 63 supplement blocks = 345 | 2 (+1 caption-only) | 7 | 3 | 333 |
| Imaging page (`imaging_results.html`) | 83 figures | 18 | 0 | 11 | 54 |

Method: every `<figure>` and every `s<sec>-<fig>-<n>` supplement block was scanned for RESTnl / RESTW / rest / restdock /
quiet / reference / baseline wording; hits were read in context (the many "MOs-rest / MOp-rest / MO-rest" area names,
"one-vs-rest" decoders and "the rest of cortex" phrases are not baselines), and the scripts named were grepped for
`restw`, `rest_by_position`, `restdock`, `precue_restw`, `quiet_frame_path`, `session_rest_svt*`.

## 2. Inventory table

Cost = wall time on this box with <= 4 workers (shared box rule). "here" = must run on the analysis desktop (inputs on M:,
map caches on `C:/Users/SabatiniLab/.widefield_session_cache`).

### 2a. Encoding goal page

| page | fig | fid | current baseline | action | script | code change | est. cost |
|---|---|---|---|---|---|---|---|
| goal | 14.21 | lick_precue_restw | RESTnl (rescue wr) | (b) caption only: the caption still says "Numbers below in this caption are the earlier restw values" | scripts/lick_precue_restw.py --rest-variant restnl | none; rewrite the caption numbers from the existing `*_restnl` CSVs (precue_restw_changes*_restnl.csv) | text only, ~15 min |
| goal | 14.21 S1 | s14-21-1 | RESTnl | (b) none | lick_precue_restw.py --rest-variant restnl | none | 0 |
| goal | 14.21 S2 | s14-21-2 | RESTnl | (b) none | lick_precue_restw_traces.py --rest-variant restnl | none | 0 |
| goal | 14.21 S11 | s14-21-11 | RESTnl | (b) none | stopped_trial_traces.py --rest-variant restnl | none | 0 |
| goal | 14.21 S12 | s14-21-12 | RESTnl (A, D); own pre-cue ref (threshold, A own-pre-cue) | (b) none | lick_cue_to_lick_extract.py / lick_cue_to_lick.py / lick_cue_to_lick_precue.py --rest-variant restnl | none | 0 |
| goal | 14.21 S13 | s14-21-13 | RESTnl | (b) none | arrival_to_cue_traces.py --rest-variant restnl | none | 0 |
| goal | 14.18 S28 | s14-18-28 | RESTnl (S28c / S28f), RESTW quoted as sensitivity | (b) none | disengagement_tests.py (uses restnl.level) | none | 0 |
| goal | 14.21 S7 | s14-21-7 (revup_neural_traces_quartile_far_contra_right.png, revup_neural_late_vs_normal_vs_spont.png) | **RESTW** ("whole-session traces above restw (as Fig 14.21)"; per-trial pre-cue level above RESTW in (5b)) | **(a)** add `--rest-variant restnl` | scripts/revup_neural.py | the per-session cache `<OUT_W>/revup/sessions/<label>.npz` stores traces MINUS RESTW (line ~118, `src["restw"]` from `precue_restw/<label>.npz`); RESTnl is a flat per-session x area shift, so swap at LOAD: `+ restw - restnl.level(label, COLS)` (exact; same pattern as stopped_trial_traces), drop sessions with ok False, suffix `_restnl` (restnl.SFX), `restnl.relabel` on text. No re-extraction. Within-session latency analyses are unaffected by a flat shift; only levels / between-session numbers move | ~30 lines; run ~10-20 min (stats + plots from cache), here or anywhere with M: |
| goal | 14.21 S4 | s14-21-4 (B4 sentence: "not a session-level baseline shift (restw-level time course does not match ...)") | **RESTW** used as a CONTROL covariate (features minus session restw; restw-level x beta term; pre-cue contra-ipsi level above restw) | **(a, low priority)** add a RESTnl variant of the B4 control | scripts/lick_enl_offset_explore.py (--offset, B4 block ~lines 408-476) | read the RESTnl level per session (restnl.level, or the `_restnl` precue sessions table written by lick_precue_restw --rest-variant restnl) beside `precue_restw_sessions.csv`; add variant `_restnl` (features minus session RESTnl / area SD) with its own frozen pre decoder; `restnl_term`; pc_ci above RESTnl | ~40 lines; run ~10-30 min (frozen ENL decoders refit per variant) |
| goal | 14.21 S14 | s14-21-14 | RESTW vs strict vs rescue comparison | (c) keep as is: it IS the RESTW-vs-RESTnl audit | restw_strict_check.py, restnl_rescue.py | none | 0 |
| goal | 14.18 S10 | s14-18-10 (lick_interhemi_simple.png, ihf_*.png) | rest = docked restdock05 rest pseudo-events (DATA: a "rest" state set next to the lick sets) | **(d)** recommend a lick-free (RESTnl-mask) rest set | scripts/lick_interhemispheric.py (rest_extract -> `<OUT_W>/interhemi_rest_w1/`), then interhemi_crossarea.py, coupling_vigour_sm.py | rest_extract: intersect the restdock05 docked mask with the RESTnl exclusions (cam4 licks_all + DAQ, [0.5, 2.0] s) before taking >= 0.95 s periods; new cache dir `interhemi_restnl_w1`; set tag `rest_nl` | rebuild rest windows ~107 sessions with imaging load: ~20-30 min at 4 workers; metrics + plots ~30-60 min. Check event counts per animal x epoch first (strict keeps only 17-35 % of pre-stroke rest in PS92-94 and docked periods are already ~1.0-1.5 s) |
| goal | 14.18 S17 | s14-18-17 (xa_*.png, cv_coupling_vs_movement.png) | docked quiet rest set as DATA | (d) same as S10 | scripts/interhemi_crossarea.py (reads interhemi_rest_w1) | read the new rest_nl cache | ~20-40 min after S10's cache |
| goal | 14.18 S18 | s14-18-18 (sm_direction_change.png, sm_xcorr_curves.png) | docked rest set as DATA | (d) same as S10 | scripts/coupling_vigour_sm.py | read the new rest_nl cache | ~15-30 min after S10's cache |
| goal | all other figures and supplements (e.g. 0.x frozen encoder, 11.x-13.x decoders, 14.16 / 14.20 / 14.28 area maps with own pre-lick baseline [-1.0, -0.5] s, 15.x one-vs-rest level maps, 14.21 S8 jaw energy / pre-cue jaw median, 12.14 "2 s quiet" lick set) | - | own pre-lick / pre-cue reference, change-from-pre of raw dF/F AUC, decoders, encoders | (c) not appropriate | - | - | - |

### 2b. Imaging page

All section-7 REST maps use **REST = RESTref, the TIME-LOCAL frame-weighted rest baseline** (12 bins per session, median of
engaged restdock05 rest frames per bin, interpolated; `position_reference_maps.session_rest_svt_timelocal`), NOT RESTW
(provenance prov_s7.md: "All page REST = RESTref (not RESTW)"). Its rest frames exclude DAQ contact licks only, buffer
[0.5, 1.0] s. Pre = all curated pre (June + August).

| page | fig | fid | current baseline | action | script / renderer | code change | est. cost |
|---|---|---|---|---|---|---|---|
| imaging | 7.1 | map_REST_precue_contact | REST (time-local) | (a) RESTnl map | epoch_grant_figures 15r, contact arm (grant_figures/epoch), align precue | RESTNL derived reference (section 3) | shares arm cost below |
| imaging | 7.2 | drift_REST_precue_contact | REST | (a) | scripts/map_drift_prepre.py --arm contact --refs restnl --aligns precue | add restnl to `--refs` (reads maps_by_epoch) | shares drift cost |
| imaging | 7.3 | map_REST_precue_LP | REST (DAQ rest frames, LP gate) | (a) | 15r, LP arm (WIDEFIELD_LICK_SOURCE=lp; grant_figures/LP_licks/epoch) | same | arm |
| imaging | 7.4 | drift_REST_precue_LP | REST | (a) | map_drift_prepre.py --arm lp | same | drift |
| imaging | 7.5 | map_REST_enl1s_contact | REST | (a) | 15r, DAQ_ENL1s arm (precue 1 s) | same | arm |
| imaging | 7.6 | drift_REST_enl1s_contact | REST | (a) | map_drift_prepre.py --arm enl1s | same | drift |
| imaging | 7.7 | map_REST_cue_contact | REST | (a) | 15r contact, cue | same | arm |
| imaging | 7.8 | map_REST_cue_contact_by_animal | REST | (a) | 15rpa contact, cue (epoch_maps._fig_15rpa_reference_by_animal) | loop over the derived reference too | arm |
| imaging | 7.9 | drift_REST_cue_contact | REST | (a) | map_drift_prepre.py --arm contact --aligns cue | same | drift |
| imaging | 7.10 | map_REST_cue_LP | REST | (a) | 15r LP, cue | same | arm |
| imaging | 7.11 | drift_REST_cue_LP | REST | (a) | map_drift_prepre.py --arm lp | same | drift |
| imaging | 7.16 | map_REST_lick_contact | REST | (a) | 15r contact, lick | same | arm |
| imaging | 7.17 | drift_REST_lick_contact | REST | (a) | map_drift_prepre.py --arm contact --aligns lick | same | drift |
| imaging | 7.18 | map_REST_lick_LP | REST | (a) | 15r LP, lick | same | arm |
| imaging | 7.19 | map_REST_lick_LP_by_animal | REST | (a) | 15rpa LP, lick | same | arm |
| imaging | 7.20 | drift_REST_lick_LP | REST | (a) | map_drift_prepre.py --arm lp --aligns lick | same | drift |
| imaging | 8.1 | a_maps | REST (time-local; DAQ rest frames), LP classes | (a) | wfield_local/attempt_maps.py (WIDEFIELD_LICK_SOURCE=lp; LP_analysis/.../imaging_encoding/attempt_maps/) | `--rest-variant restnl`: replace `session_rest_svt_timelocal(s, v)` (line ~89) by the flat RESTnl SVT base of section 3 (broadcast over frames), new `signal_key` (`...:restnl-wr`), output suffix `_restnl` | ~20 lines; run lick align, 4 animals, ~15-25 min (cold trial features for the new key, ~1.6 MB / session) |
| imaging | 8.2 | a_maps_minus | REST, minus each animal's pre hits | (a) | attempt_maps.py (same run) | same (the pre-hit subtrahend is itself rest-referenced, the baselines do not cancel across sessions) | in 8.1 run |
| imaging | 7.12-7.15, 7.21-7.24 (8) | map/drift_PRECUE_{cue,lick}_{contact,LP} | PRECUE (own 1 s pre-cue mean) | (c) | - | - | - |
| imaging | 14.1-14.9 (9) | haufe*, haufez*, haufepa* | MEAN (one-vs-rest decoder patterns) | (c) | - | - | - |
| imaging | 1.1-1.3, 2.1-2.4, 4.1-4.3, 5.1-5.2, 11.1-11.2, 13.1-13.2 (17) | decoders | no per-trial baseline (decoders) | (c) | - | - | - |
| imaging | 6.1-6.3, 13.3-13.6 (7) | e_*, m_tmpl_time, m_trend_* | encoder template / R2 (session z-scored on engaged trials; an additive level is absorbed) | (c) | - | - | - |
| imaging | 9.1-9.11, 10.1-10.2 (13) | c_*, cp_*, r_crossnobis, d_drift | coding directions (z-scored in the pre frame; "mean of the rest" = other positions), crossnobis distances | (c) | - | - | - |
| imaging | 0.1 | b_timecourse | behavior | (c) | - | - | - |
| imaging | 12.1 | d_rest15f | rest FRAMES as decoder input (docked, lick buffer [0.5, 1.0] DAQ, lick-gap matched) | **(d)** recommend an LP-lick-free sensitivity | scripts/rest_migration/rest_frozen_decoder.py | rest periods from the RESTnl mask (cam4 + DAQ, [0.5, 2.0]); a NEW frozen pre-stroke rest decoder spec (frozen_models rule 10: never overwrite the restdock05 one) | ~1 h incl. frozen fit; here |
| imaging | 3.1-3.3 | d_state, d_state_delta, s_per_animal | rest = one of the frozen STATE decoder's three classes (DATA) | (d) recommend NO change now | wfield_local/locomotor_decoder.py, epoch_state.py | (would need a new frozen state model, kind `state`) | - |
| imaging | 15.1-15.7 | q_fr, q_examples, q_fr_detail, q_threshold, q_spectrum, q_sdmaps, q_eta_maps | QUIET frames as DATA (no DAQ contact lick or running within 1.5 s; spout state ignored); event baseline = 2-s running median / pre robust SD (not rest) | (d) recommend an LP-lick-excluded sensitivity for 15.1, 15.6, 15.7 | scripts/cohort2_qc/quiet_spikes_v2(_fr).py `busy_frames`, quiet_fluct.py, quiet_eta_fr.py | add cam4 `licks_all` onsets (+-1.5 s, as the DAQ rule) to `busy_frames` behind a flag; outputs suffixed | ~30-60 min per script family; here |

## 3. The epoch map renderer: where the rest mask is built and how RESTnl fits

**Where REST / RESTW come from today (widefield_pipeline `main`):**
1. Mask: `quiet_periods.rest_mask` (restdock05: docked, not running, DAQ contact licks excluded [0.5, 1.0] s, min_rest_s),
   written per session as `*quiet_sample.npy` / `*quiet_frame.npy` under the session's `mc` (READ-ONLY inputs).
2. REST (the page's): `position_reference_maps.session_rest_svt_timelocal` reads `quiet_frame_path(mc)`, ANDs the
   engagement gate (`rest_engagement.engaged_frame_mask`), 12-bin median, interpolated -> subtracted from the SVT BEFORE
   `_working_xy` (per-trial, time-local); load 3 of `session_raw_maps` (`raw_rest`).
3. RESTW: `session_restw_svt` -> `rest_by_position.rest_frames_by_position(session, T, docked=False)` (sample mask from
   `_session_daq`) -> `restw_from_frames` (flat per-position median, equal weight, 200 frames, all 6 positions) -> one
   (K,) vector subtracted from the SVT; load 4 (`raw_restw`).
4. `reference_maps(parts, ref)` picks the column; `maps_by_epoch` pools; `epoch_maps._fig_15r_reference_maps`,
   `_fig_15rpa_reference_by_animal` and `scripts/map_drift_prepre.py` loop over `REFERENCES = (mean, rest, restw, precue, raw)`
   (each needs an `epoch_maps._REF_TEXT` entry and a `reference_tag`).

**Is the per-session RESTnl level from restnl.py enough? No.** It is one number per AREA (ROI v8 area means of the encoding
signal after `combined_encoding._slow_basis` drift removal, x1e3, split rests), read from
`iti_licks/restw_check/sessions/<label>.npz`. The maps need the baseline in the MAP signal's own space (the SVT from
`joint_basis._load_session`, projected through U_atlas), so a per-pixel RESTnl baseline must be built: the RESTnl MASK
applied to the SVT frames. The mask is not on disk (restw_strict_check rebuilds it in memory).

**But it does NOT need per-trial re-extraction.** RESTnl (like RESTW) is FLAT: one (K,) vector per session, the same for all
trials and positions. The window features are linear means, so for every position
`map_RESTnl[q] = raw[q] - (U @ b_nl).reshape(MAP_SHAPE)` exactly, where `raw` is already in every cached `prm-*` pickle
(same trial set: raw / restw / rest loads all call `_working_xy(..., baseline="none")`). So:

- **Option A (recommended, = the goal page's RESTnl):** per session, build `b_nl` (K,) on the SVT + its pixel map
  `U @ b_nl`, cache it (tiny), and form the RESTnl column after collection. The ~33 GB of warm `prm` caches stay valid.
- **Option B (time-local lick-free REST, closest to the page's current REST):** a time-varying subtrahend cannot be taken
  out of the averaged map; it needs a 5th load inside `session_raw_maps` -> changes `_raw_maps_code()` -> every `prm` cache
  goes cold for every arm. Only if Priya wants the drift-tracking property kept (note: RESTW went flat on 09-14 because
  `flat_vs_timelocal` moved no conclusion).

**Code changes (Option A), on a branch off `main` (another session commits to main; do not touch its working tree):**
1. NEW module `wfield_local/restnl_mask.py` (do NOT edit `rest_by_position.py` / `quiet_periods.py` / `rest_engagement.py`
   / `position_reference_maps.session_raw_maps`: they are inside `_raw_maps_code()`, and editing them invalidates every map
   cache). Port from `scripts/restw_strict_check.py` (`widen`, `floor_runs`, `_daq_lick_onsets`, the strict mask:
   `floor_runs(rest0 & ~widen(daq_on,[0.5,2.0]) & ~widen(cam4 licks_all daq_on_s,[0.5,2.0]), min_rest_s)`), the frame
   selection by substituting the mask into `rest_frames_by_position(docked=False, engaged_only=True)` (as
   restw_strict_check does, in-process) and the `wr` estimator (>= 50 frames, >= 4 of 6, renormalised; `strict` as an
   option). `licks_all.csv` path from `lp_licks` (exists on main). Session-cache kind `restnl-svt-<digest>` holding
   `b_nl`, the pixel map, positions used, frame counts per position, ok flag. Never write on the `mc` directories.
2. `position_reference_maps`: `DERIVED_REFERENCES = ("restnl",)`; KEEP the `refs` entry of the `prm` cache spec equal to
   the original 5-tuple (adding to `REFERENCES` itself changes the digest -> cold caches); `reference_maps(parts,"restnl")`
   and `maps_by_epoch` attach `raw - base_map` per label from a second `map_collect.collect` over the restnl cache.
   `reference_tag("restnl") -> "RESTNLref"`.
3. `epoch_maps._REF_TEXT["restnl"]` (the documented trap: a reference without an entry raises per arm mid-render);
   `_fig_15r_reference_maps` / `_fig_15rpa_reference_by_animal` loop over `REFERENCES + DERIVED_REFERENCES`; an
   env / flag to render ONLY restnl so the other references are not redrawn.
4. `scripts/map_drift_prepre.py`: accept `--refs restnl`.
5. `attempt_maps.py`: `--rest-variant restnl` (section 2b).
6. Checks before any render: (i) on the warm `prm` pickles (no imaging load), `raw[q] - raw_restw[q]` must be identical
   across positions within a session (proves the flat shortcut on the existing RESTW column); (ii) the new module's area-
   space level on 3 sessions must reproduce `restw_check/sessions/<label>.npz` `strict__posmed` exactly; (iii) print the
   per-animal x epoch table of kept sessions / frames per position (rule 8), incl. the June pre sessions.

**Sessions:** the map arms hold 120 (contact) / 119 (LP) sessions incl. June pre; restw_check / RESTnl exist for 107
(August pre + post, built 10-29 in ~7 min with 4 workers). June sessions need new masks (`licks_all.csv` exists for them,
12 checked); PS93_0813 has no LP licks (no behavior video) -> loses its RESTnl column (or DAQ-only fallback, decision).

**Cache sizes (measured 11-01, `C:/Users/SabatiniLab/.widefield_session_cache`, 45 GB total, 452 GB free on C:):** `prm-*`
34-42 MB / session / arm (4.1-5.1 GB per arm x 7 arms ~ 33 GB); `bm-*` ~16 MB / session; `tf-*` 0.8-3.2 MB / session.
Option A adds < 1 MB / session (~0.1 GB total). Option B adds a full new `prm` set (~33 GB) and orphans the old one (deletable
as superseded derived output, Priya's rule).

## 4. Cost estimates

| step | what | cost | where |
|---|---|---|---|
| M0 | code (section 3) + checks (i)-(iii) | ~half a day of code; checks ~15 min | anywhere / here |
| M1 | RESTnl SVT bases, ~120 sessions (DAQ h5 + licks_all + SVT + U_atlas load) | ~10-20 min at 4 workers (restw_check precedent: 107 sessions in ~7 min) | here |
| M2 | 15r + 15rpa, restnl only, 7 arm-units (contact precue / cue / lick; LP precue / cue / lick; ENL1s precue) on warm `prm` caches | collection ~0.13 s / session warm; stats (nested bootstrap 2000 on 2,022 bins) + draw ~10-15 min per unit -> ~1-2 h serial, less with `--parallel-arms 2` (~6 GB RAM per arm) | here |
| M3 | drift: map_drift_prepre --refs restnl, 3 arms (7 align-units) | ~30-60 min | here |
| M4 | attempt_maps --rest-variant restnl (lick, LP) | ~15-25 min | here |
| M2-B | Option B instead: cold `prm` recollection | +~1.5 h (120 x ~6.2 s effective at 4 workers x 7 arms; 10-21 measured 3 h 41 m for 15r+15e+14m cold, serial arms) and +33 GB | here |
| G1 | revup_neural --rest-variant restnl | ~10-20 min | anywhere with M: |
| G2 | lick_enl_offset_explore B4 restnl variant | ~10-30 min | anywhere with M: |
| D1 | interhemispheric rest set rebuilt on the RESTnl mask + S10 / S17 / S18 | ~1-2 h total | here |
| D2 | rest_frozen_decoder on lick-free rest (new frozen spec) | ~1 h | here |
| D3 | quiet events with LP licks excluded (15.1 / 15.6 / 15.7) | ~1-2 h | here |

Shared-box rule for all: <= 4 workers (`WIDEFIELD_MAP_JOBS=4`, `--jobs 4`), stop only PIDs started by the run, nothing
on the GPU.

## 5. Proposed order

1. **Text only (no run):** goal Fig 14.21 caption numbers from the existing `*_restnl` CSVs.
2. **Cheap swaps on existing caches:** G1 revup_neural (14.21 S7), then G2 (14.21 S4 B4 control). Page edits after.
3. **Read-only verification of the map shortcut:** check (i) on the warm `prm` pickles.
4. **Code (branch off main, section 3):** `restnl_mask.py`, derived reference, `_REF_TEXT`, drift `--refs restnl`,
   attempt_maps `--rest-variant`; checks (ii), (iii).
5. **Map runs:** M1 bases -> M2 15r / 15rpa (contact first: 7.7 / 7.16 carry the main claims) -> M3 drift -> M4 attempt maps.
   Then the imaging page section 7 / 8 edits and lick tags (7.x contact figures become "REST baseline: LP cam4 + DAQ").
6. **Rest-frames-as-data sensitivities (after Priya decides, section 6 item 5):** D2 (12.1) first, then D1, then D3.

## 6. Open decisions for Priya

1. **Goal page: keep RESTW next to RESTnl as sensitivity, or drop it?** Current state (10-29): RESTnl primary, RESTW kept in
   brackets / as `*_restw` files. For S7 / S4: replace the RESTW version with RESTnl (bracketing the RESTW numbers), or add?
2. **Imaging page maps: RESTnl as a NEW reference next to REST, or REPLACE REST?** Note RESTnl differs from the page's REST
   in TWO ways: flat position-weighted estimator (= RESTW, figures already on disk 10-21 / 10-22 / 10-24) AND the lick-free
   mask. Suggestion: RESTnl primary in 7.x / 8.x (consistent with the goal page), REST kept in a collapsed sensitivity
   block, RESTW available to separate estimator from mask.
3. **Flat (Option A, = goal page RESTnl) or time-local lick-free (Option B)** for the maps? A is cheap and exact; B keeps the
   drift tracking that motivated the time-local REST (09-13) at ~1.5 h + 33 GB extra.
4. **Pre baseline for the maps:** the imaging page pools all curated pre (June + August), the goal page August only. RESTnl
   for June needs new masks; June sessions may fail the >= 4 of 6 rule. Keep all-pre, or move the RESTnl maps to August-only?
   PS93_0813 (no LP licks): drop its RESTnl column, or DAQ-only fallback?
5. **Rest / quiet frames used as DATA: should they also exclude LP (non-contact) licks?** Recommendation:
   - imaging 12.1 rest decoder: YES as a sensitivity, high priority. Non-contact licks are aimed at the spout, so licks inside
     "rest" could carry position directly; restdock05 rest held 13-57 licks / min (S14), most pre-stroke, so the contamination
     changes with epoch exactly as the retained fraction does.
   - goal 14.18 S10 / S17 / S18 rest state: YES as a sensitivity (the "global coupling rise also at rest" could partly be
     licking inside rest; check event counts first).
   - imaging 15.1 / 15.6 / 15.7 quiet events: YES as a sensitivity (the acute ~5x orofacial event rate could include
     non-contact licks; face regression only partly controls for it).
   - imaging 3.1-3.3 state decoder: NO for now (frozen state model; the claim is the position / state RATIO, and lick
     contamination of the rest class would lower state accuracy, not position).
6. **RESTnl rule for the maps:** `wr` (default, >= 4 of 6, 50 frames) as on the goal page; confirm.
