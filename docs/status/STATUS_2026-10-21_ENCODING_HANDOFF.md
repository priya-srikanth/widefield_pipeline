# Handoff — 2026-10-21: attempt-based deficit, target vs execution vs neither, within-spout per spout + contact

**START HERE for the encoding model / target-vs-executed work (the ENCODING window).** Supersedes
`STATUS_2026-10-20_EVENING_ENCODING_HANDOFF.md` (still the reference for the 3-D contest set-up, §6 commands there).
The pose / 3-D / cam1 / jaw work is the OTHER window: `STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`.
**Results: `docs/ENCODING_FINDINGS.md`** (status-tagged, updated 10-21). Reasoning + numbers: `DECISIONS.md` 2026-10-21
entries. Scratchpad (launchers, logs, 3-D tables): `C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`
(`S`). Output root `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\` (`E`).

## 0. Rules (Priya; unchanged + new today)

- Curated sessions only (`analysis_kit.curated_labels()`); August-only pre baseline primary. Lead with drop dR2;
  per-animal tables first (rule 8); do not dismiss bilateral changes; canonical spout order (`orofacial_cohort.POS_ORDER`).
- Mixed models primary, epochs descriptive. No circular behavioural index (lateral error descriptive only).
- Brain maps ipsi = image LEFT; cam1 face video ipsi = image RIGHT.
- Priya runs O2 / transfer node / process-stopping commands: give `Get-CimInstance ... | Select ProcessId, CommandLine`
  + `taskkill /PID <pid> /T /F`. **Kill the TOP shell of a queue** (walk ParentProcessId up to the bash under
  claude.exe) -- killing a sub-shell lets the queue start its next step (happened 10-21 08:07).
- Write only under MICROSCOPE/Priya; never delete there; copy superseded outputs aside. Stage files by name; commit and
  push; DECISIONS entries. Results figures in `E/figures/{mixed_models,behaviour,within_spout/<run>}`, tables in `E/qc`.
- Lick-zone analyses use 3-D calibrated zones (cam4 zones failed). 3-D jaw on hold.
- **Toward-midline:** the per-spout `mid:<pos>` test is the headline within-spout test, and the POOLED map is not
  robust to contact (memory note `toward-midline-headline`).

## 1. Done today (details ENCODING_FINDINGS + DECISIONS 2026-10-21)

- **Behavioural index:** `scripts/trial_attempts.py` (hit / incomplete / jaw_only / no_try / unknown; jaw from
  `qc/jaw_onsets.csv` cam4 OR cam1, re-onset rule for jaw deflected at the cue); deficit = **time to first far-contra
  contact** on attempted trials (`mixed_models.DEFICITS["ttc"]`, `--deficit` in encoding_mixed / imaging_lmm /
  within_spout_mixed). Deficit is execution, not initiation (acute attempt .81-.93, completion .01-.11).
- **jaw_onsets rewrite 07:57** (cam1 all PS93, PS94 to 08-17) -> watcher reran attempts + all LMMs 08:00-08:37.
  Unknown <= 6 %; target tracks the deficit more strongly (3-D z -8.3, cam4 -8.2 / firstlick -6.3); strict executed null.
- **Target vs execution vs neither:** 3-D contest (2.4b), lick-template rerun (QUOTE the 20261021 tables), destination
  vs 3-D landing zones (2.4c): post-stroke activity resembles the pre-stroke TARGET; execution-like 4 % acute only.
- **Within-spout per spout + contact (cam4, 107 sessions):** pooled toward-midline chronic reversal NOT robust
  (cluster 28 % base -> nothing with contact / hits-only); per-spout far_ipsi chronic loss and close_contra acute flip
  survive both controls; commanded-axis between = miss / contact confound pending the adjusted fit.
- **Speed:** runs were network-bound (gigabit link to the share saturated by the other window's video jobs; per-read
  latency the rest). `within_spout_deviation.py --variants contact,contact_hits,base` (one load, several fits) and
  `combined_encoding` local input cache `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache` (Priya OK'd; disposable
  INPUTS only; 5 s vs 80-150 s per session, verified identical; 41 GB at 09:52). **Any edit to a cached loader module
  (`combined_encoding`, `orofacial_cohort`, `orofacial_outputs`, `trial_windows`, `beta_maps`, `daq_trials`,
  `pose_kinematics_demo`) invalidates every entry** -- the next run re-reads the share once. 5 workers beat 2 (latency-bound).
- **Contact-adjusted across-spout fit** (a00045b): under `--contact`, spout means moved to the session's mean contact
  count with the within-spout contact coefficient. NOT yet in any output.

## 2. RUNNING at handoff (10-21 ~09:55)

| what | task / log | then |
|---|---|---|
| 3-D within-spout variants, PS92 / PS93 (47), contact / contact_hits / base + mixed models | `S/run_ws_polar_variants.sh`, logs `S/ws_polar_variants.log`, `S/ws_mixed_polar_*.log`; old pooled folder copied to `qc/superseded_within_spout_polar_pooled_20261021/` | report per-spout `mid:<pos>` (3-D), compare with cam4 |
| lick-template pass 2 with the free-kernel ceiling | `S/lick_templates_ceiling.log`, started 07:11 | report target_like / exec_like / neither_share (cortex, animal x epoch) |
| direct contact check (does contact count rise with toward-midline deviation, per spout, pre vs chronic?) | `S/contact_dev_check.py` -> `S/contact_dev_check.{csv,log}` | confirms or rules out the contact confound in the pooled reversal |
| jaw_onsets watcher | task b2d4hbh9m (4 h from ~06:40) | reruns attempts + LMMs on any further rewrite |

## 3. To-do (priority)

1. Read the three running results above; ENCODING_FINDINGS + DECISIONS.
2. Rerun the CONTACT variants with the contact-adjusted between (cam4 then 3-D; cached, minutes):
   `--variants contact,contact_hits --redo` (+ 3-D env) then `within_spout_mixed.py --folder ...` for each.
   Read `between_cmd` vs `between` with contact control and hits-only.
3. Figures: attempt classes / ttc; destination summary; target / execution / neither summary; per-spout toward-midline
   maps (contact-controlled); shorten the LMM x-axis label (overlaps panels).
4. Destination test for PS94 / PS95 when their 3-D exists.
5. Whole-session-lick refits need an O2 re-bundle (Priya submits).
6. Old copies Priya may delete: stray empty `qc/figures/imaging_lmm_epoch`, old figure copies under `qc/figures/` and
   `qc/within_spout_deviation*/figures/`.
7. Carried over (10-20 evening §4): first-lick latency cam4 vs cam4 + cam1 vs DAQ; LocaNMF cross-check of within-spout
   slopes; pre-cue -> lick-window carry-over decode; 3-D jaw when revived.

## 4. Pitfalls (new today)

- `jaw_pass_qc` in v36 = "jaw moved", not a tracking-QC flag. cam1 jaw needs `jaw_onsets.csv` (old cam1 windows were on
  the frame-index clock). True/False strings vs 1/0 read as 0 % agreement.
- Colons in figure filenames create NTFS alternate data streams (`imaging_lmm.py` now sanitises).
- `combined_encoding_figures.py` on `drop_win` folders exits cleanly (no `alone:` models).
- The pooled within-spout slope weights spouts by their lateral spread (PS94_0806: pooled +0.00027 vs equal-weight
  -0.00024) -- read `equal` / per-spout rows.
- The cluster test needs EVERY animal to agree and is all-or-nothing at |t| 3.18 (df 3): "nothing survives" after a
  control can mean a modestly weaker map; read the mixed-model z alongside.
- Two pre-push test runs at once (both windows) take ~1 h on a busy network; a rejected push ("cannot lock ref") can
  still land if the other window's push carried the commits -- check `git branch -r --contains <sha>`.

## 5. Commands

```bash
cd /c/Users/SabatiniLab/Github/widefield_pipeline
export PYTHONPATH=. WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
# cam4 within-spout, all variants in one pass, then mixed models
python scripts/within_spout_deviation.py --align lick --jobs 5 --redo --variants contact,contact_hits,base
for d in within_spout_deviation_lick_contact within_spout_deviation_lick_contact_hits within_spout_deviation_lick; do
  python scripts/within_spout_mixed.py --folder $d; done
# 3-D: same with the env of S/run_ws_polar_variants.sh (20261021 tables) and $(cat S/labels_3d.txt)
# behaviour index / LMMs
python scripts/trial_attempts.py && python scripts/encoding_mixed.py && python scripts/imaging_lmm.py
WIDEFIELD_PRE_FROM=0801 python scripts/best_match_destination.py
python -m pytest -q tests/test_within_spout_deviation.py tests/test_input_cache.py tests/test_trial_attempts.py
```
