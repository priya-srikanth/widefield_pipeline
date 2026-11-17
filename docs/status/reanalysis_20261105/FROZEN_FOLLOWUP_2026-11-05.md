# Frozen-encoder contact-only R2 drop: two follow-up checks (2026-11-05, a8 helper)

Requested by window 34; Priya approved. This was read-only. No repo file was edited, nothing was committed, and nothing was applied to a page or deck.

- Scripts (this folder): `check1_trend.py`, `check2_licks.py`, `check2_stats.py`. They were run with the locanmf python from `widefield_pipeline`.
- Outputs, in `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/frozen_followup_1105/`:
  - check 1: `check1_log.txt`, `check1_per_animal.csv`, `check1_sessions.csv`, `check1_pooled.csv`
  - check 2: `check2_sessions.csv`, `check2_spearman.csv`, `check2_log.txt`
- Quantity: contact trials (cls `hit`), frozen pre-stroke encoder, trial-to-trial R2, cortex, cue -3..+3 s.
  - Frames: `v7g3_t3d_spont` ("spont") and `v7g3_t3d` ("nospont").

**Path note.** The per-trial files are under the REBUILD root, not the path in the brief:
`M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-..._daqlk/frozen_encoder_gate-attempted_v7g3_t3d{_spont}/joint/sessions/<label>_trials.npz`. They hold 107 sessions per frame. Check 1 reads 34's tables. Check 2 recomputes the per-session R2 from these daqlk npz. Over all 107 sessions, in both frames, the recomputed values match 34's tables (max |diff| 8e-17). So both checks rest on the daqlk run, and nothing came from an old-tag folder.

## Check 1: is the post-stroke drop more than each animal's pre-stroke trend?

**Question.** PS92 declines through its pre sessions. Does the post-stroke drop survive after removing a per-animal pre-stroke linear trend?

**Method.**
- Per animal: OLS fit of `trial_R2 ~ day` (days since stroke, `epochs.days_since_stroke`) on PRE sessions only.
- Residual = observed minus the line extrapolated to each post session; then the mean residual per epoch.
- Pooled tests:
  - `boot_ci`: nested animal -> session bootstrap, SESSION-weighted flat pool (its documented weighting), line fixed.
  - Animal-weighted: mean of the 4 animal means, nested animal -> session bootstrap, line fixed.
  - Refit bootstrap: animal-weighted, also resamples pre sessions and refits each line, so it carries the extrapolation uncertainty.
  - One-sample t on the 4 animal means (df 3).
- Sensitivity baselines that do not extrapolate per animal:
  - "common": one within-animal pre slope shared by all animals, with animal intercepts.
  - "late3": the mean of each animal's last 3 pre sessions.

**Per-animal table, spont frame.** Slope is ×1e-3 per day with a 95 % t CI. Residuals are observed minus trend. Values in parentheses are vs the flat pre mean.

| animal | n pre | pre days | slope | acute | subacute | chronic |
|---|---|---|---|---|---|---|
| PS92 | 8 | -11..-3 | -26.3 [-37.0, -15.6] | +0.187 (-0.069) | +0.260 (-0.128) | +0.761 (-0.218) |
| PS93 | 7 | -11..-3 | -1.8 [-20.7, +17.2] | -0.117 (-0.134) | -0.046 (-0.071) | -0.065 (-0.131) |
| PS94 | 8 | -10..-2 | +24.7 [+11.2, +38.2] | -0.287 (-0.055) | -0.542 (-0.030) | -1.086 (-0.018) |
| PS95 | 8 | -10..-2 | +7.1 [-4.9, +19.0] | -0.155 (-0.107) | -0.119 (-0.037) | -0.568 (-0.301) |
| animal mean | | | | -0.093 (-0.091) | -0.112 (-0.066) | -0.240 (-0.167) |

The post epochs cover days 1-7 (acute), 2-22 (subacute) and 11-50 (chronic). Each pre span is only 8 days, so the chronic extrapolation reaches ~5x the fitted span forward. Two animals have steep pre slopes of opposite sign: PS92 falls and PS94 rises. At chronic their extrapolated "expected" R2 is about -0.3 for PS92 and about 1.6 for PS94, both impossible. The SE of the line at the chronic mean day is 0.16-0.28.

The nospont frame gives the same picture (`check1_log.txt`):
- Per-animal trend residuals, animal mean: -0.090 / -0.120 / -0.264.
- Flat pre mean: -0.079 / -0.059 / -0.146.

**Pooled results (animal mean, 95 % interval).**

| frame | epoch | per-animal trend, line fixed | refit bootstrap | t (4 animals) | common slope | late3 |
|---|---|---|---|---|---|---|
| spont | acute | -0.093 [-0.244, +0.098] | [-0.258, +0.099] | p .42 | -0.100 [-0.147, -0.053] | -0.093 [-0.147, -0.031] |
| spont | subacute | -0.112 [-0.420, +0.168] | [-0.431, +0.165] | p .55 | -0.082 [-0.125, -0.040] | -0.068 [-0.097, -0.043] |
| spont | chronic | -0.240 [-0.881, +0.436] | [-0.924, +0.497] | p .59 | -0.207 [-0.316, -0.100] | -0.169 [-0.279, -0.092] |
| nospont | acute | -0.090 [-0.241, +0.092] | [-0.252, +0.093] | p .42 | -0.099 [-0.151, -0.056] | -0.083 [-0.139, -0.030] |
| nospont | subacute | -0.120 [-0.423, +0.154] | [-0.433, +0.154] | p .52 | -0.094 [-0.143, -0.043] | -0.063 [-0.093, -0.031] |
| nospont | chronic | -0.264 [-0.898, +0.410] | [-0.941, +0.468] | p .55 | -0.234 [-0.308, -0.149] | -0.150 [-0.218, -0.094] |

With the line fixed, the session-weighted `boot_ci` gives:

| frame | acute | subacute | chronic |
|---|---|---|---|
| spont | -0.088 [-0.28, +0.14] | -0.186 [-0.44, +0.08] | -0.153 [-0.80, +0.48] |
| nospont | -0.091 [-0.28, +0.14] | -0.193 [-0.44, +0.07] | -0.179 [-0.82, +0.45] |

The common pre slope is +1.0e-3 per day (spont) and +2.3e-3 per day (nospont): the cohort has no shared pre decline.

**Result.**
- Per-animal trend: the animal-mean point estimates stay negative and about the size of the flat-baseline drop. The intervals are very wide and include 0, because PS92's and PS94's opposite slopes extrapolate to impossible values.
- No-extrapolation baselines (common slope, late3): the drop is clear in every epoch and every frame. For PS92, measured against its last 3 pre sessions, the drop is ~0 at acute, -0.065 at subacute and -0.155 at chronic.

**Reading.** PS92's pre decline accounts for its acute drop, but not its chronic drop. The cohort-level drop does not come from pre trends. It survives every baseline that does not extrapolate a per-animal line 50 days forward, and the per-animal extrapolation is too fragile to test anything.

**Caveats.**
- 7-8 pre sessions over 8 days per animal.
- The linear model is unbounded, and R2 is bounded.
- n = 4 animals.
- late3 uses 3 sessions and is sensitive to day-to-day noise.

## Check 2: do licks explain the spont vs nospont difference (PS95 focus)?

**Question.** Does the spont-frame R2 fall below the nospont R2 (diff = spont - nospont) in lick-heavy sessions? That would mean the pre-stroke spont / lick kernels over-predict.

**Method.**
- Per session, from the daqlk per-trial npz: contact R2 in each frame, recomputed exactly as `encoder_trials.class_summary` does for a frozen file.
- The npz has no per-trial lick counts. Lick counts come from LP lick onsets (`lp_licks._read_times`: cam4 + unmatched cam1) and from the cached DAQ lick train (`dlc_frames._cached_licks`). The DAQ count for PS95_0917 is 7,114, which matches the new detector.
- Lick measures:
  - `post_contact`: licks per contact trial, cue..+3 s.
  - `pre_contact`: licks per contact trial, -3..0 s.
  - `iti_per_min`: licks per minute between a trial's stop and the next cue.
- Statistics:
  - Spearman correlation per animal over all sessions, and over post sessions only.
  - Within-animal pooled correlation: ranks taken within each animal.

**Per-animal epoch means (diff; LP ITI licks per minute).**

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 diff | +0.005 | +0.010 | +0.007 | +0.004 |
| PS93 diff | +0.007 | +0.005 | +0.007 | +0.008 |
| PS94 diff | +0.005 | +0.004 | +0.005 | +0.007 |
| PS95 diff | +0.039 | -0.010 | +0.009 | -0.047 |
| PS95 LP ITI /min | 8.3 | 24.1 | 13.0 | 23.7 |

**Spearman diff ~ licks per animal, all sessions.** Each cell is rho (p).

| animal | LP post_contact | LP ITI /min | DAQ post_contact | DAQ ITI /min |
|---|---|---|---|---|
| PS92 | -0.31 (.11) | -0.39 (.045) | -0.31 (.11) | -0.39 (.045) |
| PS93 | +0.16 (.42) | +0.30 (.13) | +0.21 (.30) | +0.23 (.25) |
| PS94 | -0.24 (.23) | -0.08 (.71) | -0.31 (.11) | -0.26 (.19) |
| PS95 | -0.71 (<.001) | -0.81 (<.001) | -0.70 (<.001) | -0.47 (.014) |

Within-animal pooled correlations for diff:

| lick measure | r | p | animals with negative rho |
|---|---|---|---|
| LP post_contact | -0.29 | .003 | 3/4 |
| LP ITI | -0.26 | .008 | 3/4 |

In both frames, R2 itself falls with more contact-trial licks: within-animal r -0.41 (spont) and -0.39 (nospont), negative in all 4 animals.

**PS95 per-session table.** All 27 sessions are in `check2_log.txt`. In the pre sessions, diff is +0.02 to +0.06 and LP ITI is 4.5-15 per minute. Among the chronic sessions:
- 0907: diff -0.350 (spont R2 -0.275 vs nospont 0.076), with 71.7 LP ITI licks per minute and 14.9 licks per contact trial.
- 0928, 1001, 1005: diff -0.08 / -0.05 / -0.04, with LP ITI 21-30 per minute.
- 0903, 0910, 0924: lighter-licking sessions with diff +0.02.

Without 0907, PS95's chronic means are 0.122 (spont) and 0.139 (nospont), diff -0.017, and rho(diff, LP ITI) is still -0.78.

**Result.** Yes for PS95: the spont-minus-nospont difference goes more negative as licks increase, both ITI and contact-trial licks, all sessions and post-only (LP ITI rho -0.78). The other three animals show a weak or no relationship, and their diff never leaves +0.004..+0.010.

**Reading.** In PS95 the spont kernels do over-predict lick-heavy post sessions. But that explains only part of PS95's chronic drop:
- The no-spont R2 also falls: 0.348 pre to 0.133 chronic.
- The spont-specific part is ~0.09 of the ~0.30 spont-frame drop, and most of it comes from one session, 0907.
- Removing spont does not change the other animals.

**Caveats.**
- Lick rate rises after the stroke in PS95, so lick count and epoch are confounded.
- LP and DAQ ITI rates disagree a lot in PS95 (0907: LP 72 vs DAQ 16 per minute). LP ITI licks may include cam1 events that are not tongue contacts.
- The ITI window is stop -> next cue. It overlaps the encoder's -3..0 s window, which the `pre_contact` measure covers separately.
- Spearman uses 26-27 sessions per animal, and epochs are pinned to the table's epoch column, which matches `epochs.epoch_of`.
- No figure was made.
