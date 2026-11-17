# RESTnl with cam1 licks (RESTnlc1): inventory, cost, re-run, old vs new (2026-11-17, overnight)

Priya 11-17: lick-free rest (RESTnl) should exclude DAQ + cam4 + cam1 licks (cam1-only events, water droplets removed),
the same lick train as the within-trial lick-free windows. Branch `analysis/restnl-cam1-1117` (worktree `wf_restnlc1`,
from origin/main abb1623b). Not pushed. Pages, FINDINGS and decks are NOT edited. Running list:
`docs/status/RUNNING_LIST_2026-11-17_RESTNL_CAM1.md`.

## 0. The new mask version

`wfield_local/restnl_licks.py` (new): switch `WIDEFIELD_RESTNL_LICKS=cam4|cam4cam1` (default cam4 = every output up to
11-17). cam4cam1 ("RESTnlc1", v2) = DAQ lick onsets + cam4 `licks_all.csv` UNION the cam1 events of
`cam1_licks/<a>_<date>_events_all.csv` (whole trial windows: pre-cue, post-cue, ITI, ~98-99 % of the video) that are
NOT cam4-matched and NOT water droplets (`droplet` column = `cam1_licks.droplet_flags`), `daq_on_s`; widened [0.5 s
before, 2.0 s after], min_rest_s floor re-applied after the DAQ step and after the video step (as before). Union => the
c1 mask removes a superset of the cam4 mask's frames (floor is monotone). A session without an events_all file gets
cam4 only and is logged by name; PS93_0813 (no licks_all.csv) has no RESTnl in either version.

**Correction (14:3x, coordinator / LICK_SOURCE_AUDIT_20261117):** v1 of this module (commit 1c72de37) took the cam1
events from `lp_licks.lp_lick_times`, whose cam1 file `<a>_<date>_events.csv` holds POST-CUE events only (cue .. trial
stop). Measured on PS92_0814: 415 cam1-only onsets added, rest samples 235,883 -> 235,874 (9 samples), frames per
position unchanged -- post-cue events barely touch rest. v2 (aac676a8) reads events_all. The v1 timing-run files stay
where they were written (`session_cache_restnlc1/PS92_0814__...npz`, `iti_licks/<T>/restw_check_c1/sessions/
PS92_0814.npz`) and are never read (v2 uses `session_cache_restnlc1v2/`, `restw_check_c1v2/`). Delete list for Priya:
section 7.

**Fig 12.1 consequence:** the 15f helper's "cam4 + cam1" run (`rest15f_restnl_20261117/`, `--lp cam4cam1`) used
`lp_lick_times` too, so it added post-cue cam1 events only and is NOT the cam4 + cam1 rest version. Re-run here with
`rest15f_restnl_1117.py --lp c1all` (new LP set in that script, merged from f30eb700): same spec otherwise (working
brackets, duration + gap matching with the gap to the nearest DAQ / cam4 / cam1 lick, June + August pre primary,
August-only sensitivity, PS92 kept with a caveat, 200 block permutations), new frozen align `rest-restnlc1v2wk-...`,
output `<G>/<T>/epoch/rest15f_restnlc1all_20261117/{allpre,augpre}/`.

Where the switch enters (new names everywhere; no cam4 file is overwritten or read as c1):

| family | code | cam4 (unchanged) | cam4cam1 |
|---|---|---|---|
| map baseline | `restnl_mask` (hook lines marked `# [gate-source hook]`, so the default digest d7591e88ae51 and cache names are unchanged) | `<session_cache>_restnl[_glp..]/<label>__restnl-<tag>__<key>.npz` | `<session_cache>_restnlc1v2[_glp..]/<label>__restnlc1v2-<tag>__<key>.npz` (key + cam1 events + LP-train code) |
| map reference | `position_reference_maps` derived ref, `epoch_maps._REF_TEXT`, `map_drift_prepre` | `restnl` / `_RESTNLref_` | `restnlc1` / `_RESTNLC1ref_`; WIDEFIELD_MAP_REFS=restnlc1 -> run-tag part `refs-restnlc1` (new folder) |
| attempt maps | `attempt_maps --rest-variant` | `attempt_maps_restnl/` | `attempt_maps_restnlc1/` |
| rest decoder 12.1b | `rest_frozen_decoder_nl --mask` | `restnl` (cache `restnl/`, frozen align `rest-restnl-b4-...`) | `restnlc1` (cache `restnlc1/`, frozen align `rest-restnlc1-b4-...`, out `restnlc1_<pre>pre/`) |
| area level (goal page) | `restw_strict_check`, `restnl.py` (+ `lick_precue_restw` reader, ENL cache, `disengagement_tests`) | `restw_check/`, suffix `_restnl`, caches `<name>_wr` | `restw_check_c1v2/`, suffix `_restnlc1`, caches `<name>_wr_c1`, `precue_enl_<src><W>s_c1`, `disengagement_restnlc1/` |
| interhemi rest set | `lick_interhemispheric` / `interhemi_crossarea` / `coupling_vigour_sm --rest-variant` | `restnl` (`interhemi_rest_w1_restnl/`, `_restnl`) | `restnlc1` (`interhemi_rest_w1_restnlc1/`, `_restnlc1`) |
| state space | `state_space_rest_baseline` + readers | `rest_baseline_svd/` | see section 1 E (status there) |

ITI windows (coordinator note, DECISIONS 2026-11-17): no RESTnl consumer computes an ITI response AUC (grep for AUC /
RESP_WIN in the consumers: only `iti_area_maps` / `lick_balance_angle_lmm`, which do not read RESTnl, and they already
use `iti_licks.ITI_RESP_WIN`). Nothing here hard-codes 0..1 s for ITI.

## 1. Inventory: every RESTnl consumer and the page / deck figure built on it

Page versions read: imaging v58 (scratch `pages/img/img_v58.html`), encoding / goal 11-16 copy (`pages/enc/page_new.html`,
live ~v315), state space 11-17 copy (`pages/ss/ss_page_1117.html`). Kinematics, FaceRhythm, behavior: no RESTnl.
"old run" = the cam4 output on the rebuild (queue `render_daqlk` step, same env) used as the OLD side of section 5.

### A. Map family (`restnl_mask` baseline maps)

| page fig | figure | producer (old rebuild step, wall) | cam4cam1 run |
|---|---|---|---|
| IMG 7.3b (pre-cue, + by animal, period, jaw-only panels) | `LPfinal/LP_licks_gateLP_<T>_refs-mean+rest+restw+precue+raw+restnl/epoch/epoch_15r*_RESTNLref_precue_*` | `lp_final_render.py maps` (g5_lpfinal_maps_g, 81 min, all refs) | same script, WIDEFIELD_MAP_REFS=restnlc1 -> `..._refs-restnlc1/` |
| IMG 7.10b | same folder, `_cue_working` | same | same |
| IMG 7.18b, 7.19b | same folder, `_lick_lick`, `15rpa` | same | same |
| IMG 7.3c / 7.10c / 7.18c / 7.19c (session-mean twins) | RAW reference | same step | NOT affected (no rest subtraction) |
| IMG 7.4b, 7.11b, 7.20b (drift, LP) | page cites the PRE-rebuild 11-02 run (`labcams/grant_figures/map_drift_prepre_restnl/LP_licks/`); rebuild run exists (g5_map_drift_lp_g, 27 min) | `map_drift_prepre.py --arm lp --refs ... restnl` | `--refs restnlc1 --out <G>/map_drift_prepre_restnlc1_20261117` |
| IMG 8.1b, 8.2b (attempt maps) | page cites the PRE-rebuild run (`LP_analysis/.../attempt_maps_restnl/`); rebuild run exists (d_attempt_maps_restnl_r, 42 min) | `attempt_maps --rest-variant restnl` (LP licks + LP gate) | `--rest-variant restnlc1` |
| IMG 12.1 (rest decoder, working brackets = the 15f re-run) | old: `<G>/<T>/epoch/rest15f_restnl_cam4_20261117/` (cam4, Priya's interim primary) | 15f helper `rest15f_restnl_1117.py --lp cam4` | `--lp c1all --pre all` (+ `--pre aug`) -> `rest15f_restnlc1all_20261117/` (the helper's `--lp cam4cam1` run is post-cue cam1 only, section 0) |
| 12.1b (rest decoder on RESTnl, engaged brackets; HEADLINE_TABLE 11-09; not on the v58 page) | `<G>/rest_frozen_decoder_restnl/<T'>/restnl_augpre/` | `rest_frozen_decoder_nl.py --mask restnl --pre aug` (g5_rest_frozen_nl_r, 25 min) | `--mask restnlc1 --pre aug`; `--pre all` if the fairness gate refuses an animal |
| goal 14.18 S10 / S17 / S18 (interhemispheric rest set; xa / cv-sm) | `iti_licks/<T>/interhemi_*_restnl*`, `cross_area_restnl/`, `coupling_vigour_restnl/` | l_interhemi_rest_restnl_g 17 min, l_interhemi_restnl_g 51 min, l_xa_restnl_g 44 min, l_cv_sm_restnl_g 35 min (+ plots) | `--rest-variant restnlc1` (new folders / suffix) |
| goal 14.20 (rest LEVEL vs aim, h2h) | scratch `prelick_test/restnl_levels.py` -> `head_to_head/` (`restnl_levels_sessions.csv`, `h2h_*`) | scratch scripts (restnl_mask.lickfree_mask) | copy run under the switch into scratch `rc1/restnl_levels_c1/` |
| SS 9.1-9.4, 9.7, 9.8, 9.10, 10.17, 10.19, 10.21-10.26 | `<state_space_recovery>/rest_baseline_svd/` -> chronic_maps / shift_* / why_cleaner | `state_space_rest_baseline.py` then the SS figure scripts | section 1 E |
| (not on a page) `restnl_maps.py` arms contact / lp / enl1s, `restnl_position_coverage.py`, `swapmask_compare_restnl.py` | RESTnl map extras / audits | - | not re-run (no page figure) |

### B. Area family (`restw_check` levels via `scripts/restnl.py`; goal page)

| page fig | producer (old rebuild step, wall) | cam4cam1 run (env `WIDEFIELD_RESTNL_LICKS=cam4cam1`, outputs `_restnlc1`) |
|---|---|---|
| (input) `iti_licks/<T>/restw_check/sessions/` | `restw_strict_check.py --jobs 3` (l_restw_check_g, 49 min) | -> `restw_check_c1v2/` |
| goal 14.22 + S1 (pre-cue level maps vs RESTnl, fixed window) | `lick_precue_restw.py --precue-window fixed --rest-variant restnl` (l_precue_restnl_g, 7 min) | same |
| goal 14.22 S2 (pre-cue traces) | `lick_precue_restw_traces.py --precue-window fixed --rest-variant restnl` (5 min) | same |
| goal 14.22 ENL-window twins (LP licks) | `lick_precue_restw.py --rest-variant restnl` + traces, WIDEFIELD_LICK_SOURCE=lp (11 + 4 min) | same |
| goal 14.22 S7 (rev-up neural) | `revup_neural.py --rest-variant restnl` (21 min) | same |
| goal 14.22 S11 (stopped trials) | `stopped_trial_traces.py --rest-variant restnl` (4 min) | same |
| goal 14.22 S12 (cue -> lick A / D) | `lick_cue_to_lick_extract.py` (3 min), `lick_cue_to_lick.py` (33 min), `lick_cue_to_lick_precue.py` (17 min), `--rest-variant restnl` | same |
| goal 14.22 S13 (arrival -> cue) | `arrival_to_cue_traces.py --rest-variant restnl` (no rebuild log found: l_arrival_restnl failed early 02:07) | same, if its inputs exist |
| goal 14.18 S28c / S28f (quit-period RESTnl level) | `disengagement_tests.py all` (l_disengage_g, 58 min) | same (-> `disengagement_restnlc1/`) |
| goal 14.21 S14 (RESTW vs strict vs rescue audit) | `restnl_rescue.py` (112 min) | NOT re-run (an audit of the rule; it defines no page number that moves) |

### C. Deck (PI overview, `presentations/Overview_2026-11-06.pptx`, read-only)

Slides whose figure reads the RESTnl mask: **24** (IMG 7.10b cue maps), **25** (IMG 7.18b lick maps), **26** (IMG 7.3b
pre-cue maps), **186** (backup pre-cue maps LP, 7.3b), **187** (backup cue-window six spouts, 7.10b), **188** (backup
lick-window six spouts, 7.18b). Hidden 204-206 are the session-mean twins (not affected). No deck slide uses 8.x, 12.1,
SS section 9 / 10, goal 14.18 S10-S28 or 14.20-14.22 (searched notes + text for RESTnl / rest / lick-free; slide 112
gain_bio uses quiet frames, not RESTnl). New paths: section 6.

### D. ITI outputs (a8 window)

No ITI figure reads RESTnl (the ITI docked / rest sets use the restdock05 docked mask, `iti_licks.docked`; grep of the
ITI scripts for restnl / restw_check / lickfree_mask: none). Nothing of a8's is touched.

### E. State-space family

`state_space_rest_baseline.py` builds `rest_baseline_svd/<label>.npz` from `restnl_mask`; the map / shift scripts read it
through `state_space_chronic_maps._rest`. Status of the c1 version: see section 4 (run only if the SS output naming can
be kept apart without re-computing the trial caches).

## 2. Old outputs (cam4) used for comparison
Rebuild (tag T = `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`), queue `render_daqlk` logs (scratch `C--Users-SabatiniLab/.../render_daqlk/logs/<step>.log`).
