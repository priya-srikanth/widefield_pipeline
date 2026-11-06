# Which DAQ lick detector is better? (2026-11-04)

Priya's question: "which lick detector is better?" Also two side tasks: a check for sessions with no DAQ contacts in the
v36 video tables (task 2), and deleting the unregistered epoch PNGs (task 3).

Scripts, tables and figures: `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/lick_detector_1104/`.
No library code was changed.

## 1. The two detectors

Both detectors run `lick_detection.detect_licks` on the same `lick_analog` channel. The signal sits high at rest and
drops when the tongue touches the spout. A lick starts when the signal falls below 2.5 V in both detectors.

| | IMAGING profile | BEHAVIOUR profile |
|---|---|---|
| used by | `analysis_kit.lick_samples`, `_load_daq_events`, ~39 imaging files | `behavior_events`, `daq_trials`, `dlc_frames` (DAQ contacts in the video tables) |
| onset threshold | 2.5 V | 2.5 V |
| offset threshold (sets the 1-20 ms lockout) | 1.0 V | 0.5 V |
| minimum gap between kept licks | **100 ms refractory** (the 40 ms floor is subsumed) | **40 ms floor** |

Note: `daq_trials.decode` falls back to 40 ms when the config has no `min_ili_ms`, `behavior_events` falls back to 0.
The config sets 40, so both run at 40 today.

## 2. Design

- 16 curated sessions: every animal at pre (0811), acute, subacute and chronic.
  PS92 0811/0819/0824/0915, PS93 0811/0819/0824/0915, PS94 0811/0819/0827/0917, PS95 0811/0817/0820/0917.
- Both profiles were run on the raw signal, plus two in-between versions. This splits every disagreement into its cause:
  the offset threshold, the 100 ms refractory, or the 40 ms floor.
- Independent check: the cam4 Lightning Pose licks of the v36 run (`licks_all.csv`, every lick in the session, tongue-out
  span on the DAQ clock). Only DAQ onsets inside camera coverage were scored (in practice all of them).
  - Precision = share of DAQ onsets that fall inside a camera lick (span +- 30 ms).
  - Recall = share of camera licks that contain a DAQ onset. Reported for camera licks that reached the spout distance
    (`reach_frac` >= 0.9), since short licks never touch the spout.
- First lick per trial: first onset from cue + 70 ms to cue + 3.5 s (capped at the next cue). Also run without the
  70 ms floor, which is what `_trial_features` does.
- Raw-signal examples at disagreements: Fig 2.

## 3. Per-animal table (4 sessions each)

| | PS92 | PS93 | PS94 | PS95 | all |
|---|---|---|---|---|---|
| licks, imaging | 30,683 | 27,395 | 24,080 | 19,658 | 101,816 |
| licks, behaviour | 30,785 | 27,542 | 24,139 | 19,940 | 102,406 |
| extra in behaviour (all from the 100 ms refractory) | 102 (0.33 %) | 147 (0.53 %) | 60 (0.25 %) | 282 (1.41 %) | 591 (0.58 %) |
| extra from the offset threshold (1.0 vs 0.5 V) | 0 | 0 | 0 | 0 | 0 |
| extras inside a camera lick | 97 | 141 | 55 | 263 | 556 (94 %) |
| 40 ms floor removes (behaviour) | 9 | 4 | 9 | 4 | 26 |
| behaviour intervals 40-100 ms: two separate camera licks / same camera lick / no camera lick | 55 / 36 / 21 | 101 / 28 / 25 | 39 / 6 / 17 | 255 / 7 / 42 | 450 / 77 / 105 |
| behaviour intervals 90-100 ms: two separate / same camera lick | 32 / 3 | 77 / 0 | 30 / 2 | 202 / 1 | 341 / 6 |
| precision, imaging / behaviour | .903 / .904 | .955 / .955 | .955 / .955 | .944 / .944 | .939 / .939 |
| recall (reaching camera licks), imaging / behaviour | .883 / .885 | .862 / .866 | .890 / .892 | .925 / .938 | .890 / .895 |
| DAQ onset after camera tongue-out (median ms) | 8.4 | 17.5 | 9.7 | 11.1 | 11.7 (same in both) |
| trials / hits | 1596 / 1296 | 2086 / 1661 | 2236 / 1626 | 2590 / 1937 | 8508 / 6520 |
| hits that differ | 0 | 0 | 0 | 0 | **0** |
| first-lick times that differ (70 ms floor) | 2 | 6 | 0 | 2 | **10** |
| ... of those, behaviour closer to the camera's first lick | 2 | 5 | 0 | 2 | 9 |
| first-lick times that differ (no floor, as imaging uses it) | 0 | 0 | 0 | 0 | **0** |

Per session: `task1_per_session.csv`. Per animal: `task1_per_animal.csv`. Per epoch: `task1_per_epoch.csv`. Every
session follows the same pattern. The refractory drops 0.1-2.5 % of licks per session. It drops the most in PS95,
the fastest licker (1.4-1.7 % in 3 of its 4 sessions).

## 4. Results

1. **The offset threshold makes no difference.** Not one onset differs between 1.0 V and 0.5 V in 16 sessions. The
   signal is close to binary. Contacts swing from about 3.5 V to about 0 V (median low point 0.001 V), so both
   thresholds see the same offsets.
2. **The only real difference is imaging's 100 ms refractory.** It removes 591 of 102,406 licks (0.58 %).
   - 94 % of the removed licks sit inside a camera lick.
   - Intervals of 90-100 ms are real licks. 341 of 347 camera-judged pairs are two separate camera licks, i.e.
     10-11 Hz licking.
   - Below 90 ms some pairs are one lick counted twice (77 of 632 at 40-100 ms). Most of these are in PS92/PS93 pre,
     where a tongue touches, lifts and touches again within one protrusion.
   - So the refractory mostly drops real fast licks. It is not mainly removing glitches. Fig 1 shows imaging's
     distribution cut at 100 ms. The camera and behaviour distributions continue below it, most clearly in PS95.
3. **Glitches are not the issue.** No disagreement onset was a sub-millisecond spike. The disputed contacts are 20-75 ms
   long and full-swing (Fig 2).
4. **Agreement with the camera is equal, with behaviour slightly ahead.** Precision is identical (.939). Recall is
   0.5 points higher for behaviour (1.3 points in PS95). Timing against the camera is identical (DAQ contact 12 ms
   after tongue-out). Behaviour double-counts slightly more often: 0.10 % of camera licks get two onsets, against
   0.03 % for imaging.
5. **First lick, hit and RT are the same in practice.** 0 of 8,508 trials change hit/miss. With the 70 ms floor,
   10 of 6,520 RTs change, each by 91-156 ms (imaging later). In 9 of those 10 the behaviour RT is closer to the
   camera. Without the floor, which is what `_trial_features` uses, 0 RTs change. The ENL keeps the cue period
   lick-free, so a refractory carried over from a pre-cue lick almost never fires.

The figures:
- `fig1_ili_histograms.png`: inter-lick intervals per animal, camera vs both profiles. The camera curve has 4 ms
  spikes; that is the 250 fps frame grid, not biology.
- `fig2_disagreement_traces.png`: raw signal around 7 refractory-removed licks (one session each, spread over animals)
  and the one imaging-only onset.

## 5. Recommendation

- **(a) First lick, hit, RT: either profile.** They agree on every hit and on every RT as imaging computes RT.
  Switching imaging to the behaviour profile would change **0 hits and 0 RTs** (10 of 6,520 RTs, 0.15 %, if a
  cue + 70 ms floor is applied).
- **(b) Lick counts, rates, intervals, lick-aligned events: the BEHAVIOUR profile.** The 100 ms refractory deletes real
  10-11 Hz licks, and it does so unevenly: 1.4 % in PS95 against 0.25-0.5 % in the others. Any lick-rate or interval
  measure, and any "is there a lick here" gate, should use the behaviour profile. That is already true for the
  behaviour pipeline.
- **One profile can serve both. It should be the behaviour profile** (0.5 V, 40 ms floor). The 1.0 vs 0.5 V choice is
  moot, and the 40 ms floor removes only 0-9 doubles per animal.
- **Cost of switching imaging.** Lick-aligned imaging uses the FIRST lick after the cue, so the trial set and the
  alignment times would not move. Other users of `lick_samples` would see 0.6 % more licks, mostly inside existing bouts
  (lick-free windows, pre-cue lick-free sliding, lick-locked averages). A switch means editing one place
  (`analysis_kit.lick_samples` and the `_load_daq_events` calls), bumping the cache version and re-rendering. It
  should not be done alone. The measured effect is too small to justify a re-render. Fold it into the cohort-1 rebuild
  if that window agrees.

## 6. Limits

- 16 of 120 sessions. The pattern was the same in every one of them, so more sessions are unlikely to change the
  answer.
- The camera is a reference, not ground truth. LP misses some licks (recall against all camera licks is only ~.76,
  because many camera licks are short and never touch). 6 % of DAQ onsets sit outside any camera lick in both
  profiles. These are probably camera misses or very brief touches.
- "Two separate camera licks" relies on the LP lick segmentation, at 250 fps (4 ms frames).
- The camera RT check uses the first camera lick, which can be an incomplete lick that never touched. So it only
  judges which DAQ RT is closer. It does not show that either one is wrong.

## 7. Task 2: sessions with no DAQ contacts in the v36 video tables

Checked all 120 curated sessions. For each: trials in v36 `trials.csv`, the sum and missing count of `n_daq_contacts`,
and the imaging-profile DAQ lick count from `analysis_kit.lick_samples`.

- **No session has all-zero or missing contacts while the DAQ has licks.** Every session with a v36 table has contacts
  on 174-783 trials and no missing values.
- **One flag: PS93_0813.** It has no v36 outputs at all (no `trials.csv`) but 7,445 DAQ licks. This is the known
  missing behaviour video for that session (`docs/EXPERIMENT_ERRORS.md`). It is not the empty-`.h5` failure of
  `dlc_frames`.
- The lowest contact counts are acute sessions (PS94_0819: contacts on 174 of 600 trials, 3,386 DAQ licks). They match
  the low acute hit rates, so they look genuine.

Table: `task2_zero_contacts.csv`. Script: `task2_zero_contacts.py`.

## 8. Task 3: unregistered epoch PNGs deleted

Every top-level PNG in `labcams/grant_figures/epoch/` (468) was matched against every filename/glob in
`wfield_local/deck_registry.py` (161 patterns, f-strings already expanded at import, `{}` templates read as `*`).
Each review-group candidate was then searched by exact filename and stem in:
- the repo (code, docs, DECISIONS)
- the five results-page source folders
- `presentations/`, including the PI deck `figure_provenance.csv`

The deletion review lists themselves (STATUS_2026-09-21_ENGINEERING.md, CODE_REVIEW_2026-11-04_raw.json) were not
counted as references. Each file was deleted through `writeguard.assert_deletable(path, derived=True)`.

The review groups held 28 PNGs. The 09-21 review counted 26; there are 17, not 18, eroded/mf075 files today.

**Deleted: 22.**
- 17 `epoch_15h_*` / `epoch_15k_*` files ending `_erodedgate` / `_mf075`
- `epoch_15r_position_RAWref_cue_working.png` and `epoch_15r_position_RAWref_lick_lick.png`
- `epoch_1ddelta_engagement_by_position.png` and `epoch_1e_engagement_timecourse.png`
- `epoch_24_first_last_quartile_gated_h90.png` (named only in the 09-21 review list; superseded by the quintile version)

**Kept because the registry matches them: 3.** The `epoch_15rpa_position_RAWref_by_animal_*` files match the
registered glob `epoch_15rpa_position_*ref_by_animal_*_*.png`. They are live deck figures, so they are not
unregistered. That is why the review's "15r RAWref (3)" means only the three non-`pa` files.

**Kept, ask Priya: 3.**
- `epoch_15r_position_RAWref_precue_working.png`. `docs/PRELIM_DATA_VLS_STROKE.md` (~line 847) cites numbers from this
  render (2.16 / 1.69 / 2.58).
- `epoch_1d_engagement_by_position.png`. It is a row key in `docs/status/A3_LMM_IMAGING_STARS_2026-11-03.csv`, so
  stars were computed for this family on 11-03.
- `epoch_23_quit_prodrome.png` (ungated). `scripts/rest_migration/quit_prodrome.py` still writes it
  (`epoch_23_quit_prodrome{tag}.png` with an empty tag), so a re-run will bring it back. It is also named in the
  `encoding_goal_page_src` backup notes.

Files: `decoder_mix_checks/DELETED_EPOCH_PNGS_2026-11-04.csv` (path, size, mtime, why unmatched; written before deleting),
`decoder_mix_checks/KEPT_EPOCH_PNGS_2026-11-04.csv`, full audit `lick_detector_1104/task3_epoch_png_audit.csv`.

## 9. Files

`M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/lick_detector_1104/`:
- `task1_lick_detector_compare.py`
- `task1_summarise.py`
- `task1_per_session.csv`
- `task1_per_animal.csv`
- `task1_per_epoch.csv`
- `task1_results.pkl`
- `fig1_ili_histograms.png`
- `fig2_disagreement_traces.png`
- `task2_zero_contacts.py`
- `task2_zero_contacts.csv`
- `task3_epoch_png_audit.py`
- `task3_epoch_png_audit.csv`
- logs `task*.log`
