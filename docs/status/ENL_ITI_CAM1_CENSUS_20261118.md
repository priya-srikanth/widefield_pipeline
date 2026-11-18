# ENL / ITI lick-free windows + kept cam1-only licks: census (2026-11-18)

This census answers one question: what would change if the pre-cue lick-free ENL window and the ITI lick-free windows also
removed the KEPT cam1-only licks? Kept means the `cam1_licks.keep_mask` rule of 11-17: not a droplet, `med_out_px` >= -120 px,
not cam4-matched, read from cam1 `events_all`. This is the same train RESTnlc1 uses. The census is read-only. Nothing was
re-run and no page was edited.

Script: `scripts/enl_iti_cam1_census.py`, run under the rebuild profile. It covers 107 of the 108 curated cohort-1
sessions; PS93_0813 is skipped because it has no video. Outputs are in
`M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/enl_iti_cam1_census_20261118/`:
- `enl_animal_epoch.csv`: per animal x epoch x population x window x rule x subset.
- `iti_animal_epoch.csv`
- `sessions.csv`
- `trials.csv.gz`: per trial, current and new window starts plus a status of unchanged, shift, drop or invalid_now.
- `c1_kept_static_share.csv`

## What the current trains are (from the code)
- **ENL, under LP.** The window uses `lp_licks.gate_licks_s`, which is the edited LP train: cam4 `licks_all` plus cam1
  `events.csv`. That cam1 file is POST-CUE only, so the pre-cue search sees cam4 only. **DAQ licks are not in this train**,
  because the LP source replaces DAQ. The search is `enl_window_starts_s` / `precue_window_start`: the fixed window
  [cue-W, cue) if it is clean, otherwise the latest gap in [strobe, cue), otherwise the trial is dropped. The trace families
  use `sliding_lags`, which adds a 0.3 s settle and a -2 s bound.
- **ITI A, the `iti_licks` bout starts.** These are cam4 ITI licks whose preceding gap is at least 2 s (main) or at least
  1 s. The gap is measured against cam4 plus behavior/DAQ onsets, and there must be no cue or strobe in [-2, +0.75] s.
  "Isolated" means no lick 0.05-0.75 s after the onset. The family's "0..0.5 s" is the response-AUC window, not a lick-free
  window.
- **ITI B, the `quiet_task_events` T12 mask.** This covers the trial stop to the next strobe. It is clean when there is no
  DAQ contact lick and no running bout within 0.5 s (`QS_LICK_SOURCE=daq`).
- **COVERED** means a kept cam1-only onset lies within 0.1 s of a DAQ onset or a current cam4 train onset.

Population: engaged trials, meaning the next LP lick falls within `max_rt`. The camera working-trial gate was not re-applied.
The all-LP-trial population gives the same picture.

## Result (ENL 1.0 s, decoder rule, engaged)
- Pooled: 48,554 valid windows. Of these, 2,237 (4.6 %) shift and 721 (1.5 %) drop. The median shift is 0.53 s.
- Changes are at most about 5 % in PS92 (all epochs) and in PS93 and PS95 pre / acute / subacute.
- PS94 changes more: subacute 9.3 % (1.1 % drop) and chronic 12.8 % (1.9 % drop).
- **PS95 chronic changes 18 % (8.9 % drop). At far contra, 49 % of its windows are hit and 30 % are dropped; at far ipsi,
  41 % are hit and 22 % are dropped.**
- With W = 1.5 s, or with the families' 0.3 s settle, drops rise. PS94 chronic reaches 7-11 % drop and PS95 chronic reaches
  12-14 %.

## ITI
- **ITI A, bout starts with a gap of at least 2 s.** The share lost is 1-14 % in most animal x epoch cells. PS93 loses
  11-17 % in every epoch, pre included. PS95 chronic loses 26 %.
- **ITI B, the clean ITI time removed.** This is 1-12 % in most cells and 17 % in PS95 chronic. ITIs that lose their 1 s
  clean stretch are at most 8 %, apart from PS95 chronic at 14 %.

## The cam1 events behind it
- Few of the kept pre-cue cam1-only licks are covered: 29 % pooled, and only 6-7 % in PS94 and PS95 chronic.
- The sessions that change most are driven by short events that look static. These are 6-7 frames long, with
  `range_px` about 3-6 and `spread_px` about 1.5, and they sit -50 to -90 px behind the mouth line, which is just inside
  the -120 px guard. They cluster at the far spouts. For example, PS95_0917 has about 1,170 such events at each of far_L
  and far_R.
- Events with `med_out_px` < 0 and `range_px` < 10 make up 79 % of the kept pre-cue cam1-only events in PS95 chronic,
  65 % in PS94 chronic, and 20-60 % elsewhere.
- So most of the large effect is probably the static-label artifact class that sits just above the guard, rather than
  tongue movements. This has not been checked by eye.

## Reading
- **Negligible:** PS92 everywhere, and the pre / acute / subacute epochs of PS93 and PS95.
- **Not negligible:** PS94 subacute / chronic and especially PS95 chronic. There the far-contra ENL population would lose
  about a third of its trials, and the ITI bout set would lose about a quarter.
- **Before any re-run:** decide whether the static cam1 events at -50 to -90 px are licks. The far-from-mouth guard
  (or a range / spread floor) would need extending first. Adopting cam1 in ENL / ITI with the current rule would inject a
  chronic-specific, far-spout-specific trial loss.

## Blind review of the static class (Priya 11-18: "yes let's do review of the motionless calls")
- **Already in hand, from the 11-17 review:** 25 of its test events fall in this class (kept cam1-only, `range_px` < 10,
  `med_out_px` -120..0). Priya scored 24 of them as tongue (23 lips, 1 out) and 1 as none. Almost all came from PS92 / PS93.
- **New sheet (11-18), aimed at the census hot spots:** `<G>/cam1_static_review_20261118/review/review_sheet.html`
  (G = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures`). It holds 42 shuffled candidates, pre-cue / ITI
  context only:
  - static PS95 chronic far 10, PS95 chronic close 4, PS94 chronic far 8, PS94 subacute 4, PS95 pre 3;
  - reference: 5 kept cam1-only events that move (`range_px` >= 10, chronic);
  - controls: 4 cam4-matched contact licks, 4 flagged droplets.
  It uses the 11-17 format (cam1, cam1 with the detector point, cam4, cam1 -400..+400 ms). `review/sample.csv` is the key.
  The code is in `code/` (`c1s_sample.py`; `c1x_review.py` / `c1x_sheet.py` are copied from 11-17, with only the sheet's
  title and storage key changed).
- The static class is this share of kept pre-cue / ITI cam1-only events in PS94 / PS95:

  | | pre | acute | subacute | chronic |
  |---|---|---|---|---|
  | PS94 | 0.49 | 0.45 | 0.46 | 0.52 |
  | PS95 | 0.33 | 0.28 | 0.56 | **0.78** |
- **Scored by Priya (11-18 ~14:18):** `review/scored_priya_20261118.csv`, copied from Downloads. All 42 were answered.

  | group | lips | out | droplet |
  |---|---|---|---|
  | static test events (29) | **29** | 0 | 0 |
  | moving cam1-only reference (5) | 4 | 1 | 0 |
  | control: cam4-matched contact licks (4) | 0 | 4 | 0 |
  | control: flagged droplets (4) | 0 | 1 | 3 |

  - Static test events, by stratum: PS95 chronic far 10/10, PS95 chronic close 4/4, PS94 chronic far 8/8, PS94 subacute 4/4,
    PS95 pre 3/3.
  - With the 11-17 review, this class now stands at **53 / 54 tongue**.
- **Reading:** the short, nearly motionless cam1-only events at -50 to -90 px are the tongue at the lips, not static
  labels.
  - The cam1 rule needs no movement floor; keep_mask stays as it is.
  - The RESTnlc1 / Fig 12.1 primary (cam4 + kept cam1) is not undermined by these events.
  - The census effect is REAL licks. The current cam4-only ENL / ITI lick-free windows in PS95 chronic (and PS94
    subacute / chronic) contain tongue-at-lips licks that cam4 misses, concentrated at the far spouts. Whether to re-run
    ENL / ITI with cam1 is Priya's call: she leaned away 11-18, and cam2/3 will be added later.
