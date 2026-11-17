# Are the cam1-only events that RESTnlc1 now removes from rest real licks or water droplets? (2026-11-17)

Priya's question (11-17): the cam1-only lick events newly removed from lick-free rest (RESTnlc1) might be real licks that cam4
missed, or they might be water droplets / artifacts. The droplet filter (`cam1_licks.DROPLET_*`, blind review 10-19/10-20) was validated on POST-CUE events
only, while RESTnlc1 reads `events_all` (whole trial windows). This is a READ-ONLY check: no mask or pipeline code was changed.
Branch `analysis/restnl-cam1-1117`; record of the re-run: `RESTNL_CAM1_RERUN_20261117.md`.

Output folder `<G>/restnl_cam1_compare_20261117/cam1_extra_check/` (G = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures`):
`table_{outside,removed,postcue,droplet}.csv` (per animal x epoch + ALL), `shape_quantiles.csv`, `sessions_summary.csv`,
`sessions/<label>.csv.gz` (every cam1-only row + up to 400 cam4-matched post-cue rows per session, annotated), `figures/fig1-3`,
`review/` (blind sheet), `code/` (the four scripts: `c1x_events.py` per session, `c1x_summary.py` tables / figures / sample,
`c1x_review.py` strips, `c1x_sheet.py` page).

## Method

- 119 curated sessions (PS93_0813 skipped). Event set = the RESTnlc1 rule: `events_all` rows NOT cam4-matched, NOT `droplet`, finite
  `daq_on_s` (`restnl_licks.cam1_only_onsets`; 104,837 events).
  - **outside** = peak not in cue..stop (context pre_cue / iti); 64,270 events, 61 %.
  - **removed** = the event's [-0.5, +2.0] s window contains a frame that is in the cam4 RESTnl frame set (engaged, position-labelled)
    but not in the cam4+cam1 set; 4,560 events.
  - Both masks were rebuilt from one DAQ read with the module's own functions (`restnl_mask.widen / floor_runs / frames_by_position`).
    Lick-free samples and frames match the a1 caches in all 119 sessions, and the frames removed per animal x epoch reproduce section 4 of the
    re-run record exactly.
- Definitions:
  - Tail = after the trial stop of the last camera-working cue (`trial_classes.camera_engaged_period`, rebuild profile).
  - Quintiles = session time (first strobe - 0.5 s .. last stop + 3 s).
  - Nearest lick = the nearest DAQ onset (the mask's detector) or cam4 `licks_all` lick.
  - NN = nearest neighbour in the DAQ + cam4 + cam1-only train.
  - Previous trial = the trial that has just finished (behaviour `_trials.csv`). The 0806 sessions take DAQ cues from `trial_bounds`; PS95_0806
    has no trial context because the trial counts differ.
  - Docked = `docked_mask_any`.
  - Near-droplet = spread < 4.5 px, range < 18 px and med_out > -45 px (the droplet thresholds are 3 / 12 / -35).
- Reference sets:
  - **postcue** = cam1-only non-droplet events inside cue..stop. This is the population the 10-19/10-20 review called real (round 3: 24/24 real).
  - **droplet** = flagged droplets.
  - **matched** = cam4-matched post-cue events.

## Per animal x epoch

**A. cam1-only non-droplet events OUTSIDE post-cue (cue..stop)**

| animal | epoch | n | /min | % of c1x | tail % | Q1-Q5 % | d lick med (s) | <=0.25 s % | <=0.5 s % | NN 100-250 ms % | prev hit % | prev miss, rew % | prev miss, no rew % | close_center % | docked % | spread / range / med_out (px, median) | near-droplet % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | all | 64270 | 4.84 | 61 | 1.0 | 22 24 22 17 14 | 0.22 | 53 | 67 | 41 | 89 | 5 | 4 | 12 | 15 | 2.4 / 7.9 / -59 | 9 |
| PS92 | pre | 5545 | 4.06 | 59 | 0.1 | 23 20 22 19 16 | 0.01 | 91 | 96 | 26 | 88 | 1 | 0 | 34 | 3 | 7.1 / 22.7 / 15 | 9 |
| PS92 | acute | 1449 | 2.48 | 61 | 0.4 | 26 23 25 19 8 | 0.1 | 74 | 89 | 33 | 74 | 20 | 6 | 12 | 7 | 4.3 / 13.1 / -46 | 5 |
| PS92 | subacute | 798 | 3.31 | 64 | 0.0 | 26 19 25 19 11 | 0.03 | 82 | 91 | 27 | 90 | 5 | 4 | 16 | 8 | 5.4 / 12.2 / -47 | 5 |
| PS92 | chronic | 6558 | 4.54 | 57 | 1.3 | 21 21 20 18 19 | 0.16 | 64 | 77 | 32 | 98 | 2 | 0 | 8 | 9 | 2.3 / 8.3 / -52 | 6 |
| PS93 | pre | 5449 | 4.12 | 60 | 0.5 | 24 22 19 18 16 | 0.21 | 56 | 72 | 43 | 90 | 9 | 0 | 9 | 16 | 2.4 / 8.8 / -41 | 41 |
| PS93 | acute | 1621 | 3.35 | 60 | 2.5 | 27 23 21 13 17 | 0.19 | 60 | 71 | 35 | 70 | 20 | 9 | 11 | 17 | 2.8 / 9.4 / -48 | 21 |
| PS93 | subacute | 1251 | 3.44 | 65 | 1.4 | 29 20 22 15 13 | 0.19 | 60 | 69 | 31 | 84 | 12 | 4 | 12 | 18 | 2.9 / 9.6 / -55 | 15 |
| PS93 | chronic | 6899 | 4.76 | 59 | 1.4 | 28 21 20 16 15 | 0.22 | 53 | 66 | 39 | 96 | 2 | 2 | 10 | 20 | 2.5 / 8.0 / -58 | 12 |
| PS94 | pre | 4588 | 4.19 | 65 | 0.2 | 22 23 26 18 11 | 0.16 | 64 | 76 | 35 | 98 | 1 | 0 | 11 | 11 | 2.8 / 9.9 / -55 | 4 |
| PS94 | acute | 2224 | 3.61 | 61 | 1.9 | 24 24 17 14 22 | 0.29 | 46 | 60 | 35 | 44 | 30 | 26 | 15 | 10 | 3.1 / 9.8 / -61 | 7 |
| PS94 | subacute | 3063 | 6.04 | 66 | 1.0 | 19 22 21 17 20 | 0.24 | 51 | 65 | 40 | 81 | 6 | 13 | 20 | 11 | 3.0 / 9.1 / -56 | 7 |
| PS94 | chronic | 4309 | 5.36 | 63 | 0.1 | 25 23 22 18 13 | 0.25 | 50 | 68 | 39 | 95 | 3 | 2 | 16 | 8 | 2.8 / 9.2 / -68 | 2 |
| PS95 | pre | 1917 | 1.7 | 34 | 9.4 | 16 13 15 23 33 | 0.34 | 47 | 54 | 23 | 63 | 6 | 19 | 15 | 15 | 4.4 / 13.3 / -50 | 8 |
| PS95 | acute | 197 | 1.63 | 47 | 0.0 | 24 17 20 32 7 | 0.06 | 76 | 81 | 44 | 58 | 26 | 16 | 16 | 12 | 5.5 / 15.7 / -36 | 6 |
| PS95 | subacute | 2261 | 3.24 | 53 | 1.1 | 21 24 15 17 23 | 0.72 | 37 | 46 | 31 | 61 | 20 | 19 | 13 | 18 | 2.1 / 6.5 / -76 | 5 |
| PS95 | chronic | 16141 | 15.12 | 72 | 0.3 | 18 33 27 15 6 | 0.52 | 30 | 49 | 58 | 97 | 2 | 2 | 3 | 23 | 1.7 / 5.0 / -80 | 1 |

**B. cam1-only non-droplet events that removed RESTnl rest frames**

| animal | epoch | n | /min | % of c1x | tail % | Q1-Q5 % | d lick med (s) | <=0.25 s % | <=0.5 s % | NN 100-250 ms % | prev hit % | prev miss, rew % | prev miss, no rew % | close_center % | docked % | spread / range / med_out (px, median) | near-droplet % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | all | 4560 | 0.34 | 4 | 0 | 13 19 20 20 27 | 1.88 | 8 | 16 | 25 | 72 | 10 | 17 | 13 | 35 | 1.8 / 5.4 / -67 | 7 |
| PS92 | pre | 58 | 0.04 | 1 | 0 | 17 22 29 21 10 | 0.76 | 33 | 43 | 40 | 86 | 12 | 0 | 14 | 31 | 1.6 / 3.8 / -57 | 3 |
| PS92 | acute | 29 | 0.05 | 1 | 0 | 28 31 17 10 14 | 0.46 | 31 | 52 | 45 | 48 | 52 | 0 | 0 | 14 | 0.8 / 2.8 / -66 | 7 |
| PS92 | subacute | 14 | 0.06 | 1 | 0 | 7 7 36 7 43 | 0.96 | 21 | 43 | 43 | 71 | 14 | 14 | 14 | 14 | 1.1 / 2.5 / -77 | 7 |
| PS92 | chronic | 418 | 0.29 | 4 | 0 | 17 20 25 17 21 | 0.84 | 13 | 31 | 36 | 97 | 3 | 0 | 14 | 35 | 1.1 / 3.1 / -64 | 2 |
| PS93 | pre | 412 | 0.31 | 4 | 0 | 21 26 21 17 15 | 1.46 | 13 | 21 | 23 | 95 | 5 | 0 | 9 | 32 | 1.8 / 5.7 / -46 | 37 |
| PS93 | acute | 138 | 0.29 | 5 | 0 | 13 18 18 13 38 | 1.72 | 11 | 18 | 26 | 52 | 20 | 28 | 22 | 24 | 2.3 / 5.9 / -56 | 14 |
| PS93 | subacute | 173 | 0.48 | 9 | 0 | 9 25 24 25 17 | 1.25 | 13 | 23 | 21 | 83 | 12 | 5 | 24 | 32 | 2.7 / 8.4 / -60 | 4 |
| PS93 | chronic | 660 | 0.46 | 6 | 0 | 8 21 23 24 25 | 1.2 | 10 | 22 | 20 | 98 | 1 | 1 | 18 | 26 | 2.3 / 7.1 / -62 | 2 |
| PS94 | pre | 191 | 0.17 | 3 | 0 | 29 16 23 21 10 | 1.24 | 15 | 23 | 12 | 99 | 1 | 0 | 12 | 48 | 1.5 / 4.9 / -64 | 2 |
| PS94 | acute | 301 | 0.49 | 8 | 0 | 10 11 14 20 45 | 9.17 | 1 | 5 | 15 | 13 | 29 | 58 | 12 | 17 | 2.3 / 7.1 / -65 | 13 |
| PS94 | subacute | 199 | 0.39 | 4 | 0 | 5 8 10 13 65 | 40.81 | 4 | 7 | 14 | 30 | 11 | 59 | 22 | 23 | 2.8 / 7.0 / -76 | 7 |
| PS94 | chronic | 110 | 0.14 | 2 | 0 | 5 21 20 19 35 | 1.46 | 14 | 28 | 21 | 75 | 16 | 8 | 6 | 28 | 1.9 / 5.1 / -77 | 2 |
| PS95 | pre | 365 | 0.32 | 6 | 0 | 10 14 13 18 45 | 3.73 | 6 | 8 | 15 | 47 | 10 | 33 | 9 | 28 | 2.3 / 7.0 / -62 | 9 |
| PS95 | acute | 16 | 0.13 | 4 | 0 | 19 0 25 50 6 | 10.27 | 0 | 6 | 38 | 25 | 12 | 62 | 0 | 69 | 0.6 / 1.8 / -137 | 0 |
| PS95 | subacute | 395 | 0.57 | 9 | 0 | 7 12 9 21 51 | 24.76 | 4 | 7 | 24 | 22 | 27 | 51 | 21 | 38 | 1.4 / 4.4 / -105 | 5 |
| PS95 | chronic | 1081 | 1.01 | 5 | 0 | 16 23 26 22 13 | 2.24 | 5 | 11 | 36 | 86 | 5 | 8 | 7 | 50 | 1.6 / 4.7 / -80 | 2 |

Column notes:
- "% of c1x" = share of all cam1-only non-droplet events.
- "prev miss, rew" = the previous trial was a miss with `reward_delivered` (uncollected water).
- med_out < 0 = dorsal of the cam1 mouth origin. Real tongue tips at the lips sit at <= -44; droplets sit at > -35.

**Reference sets (ALL):**
- Post-cue (n = 40,567): 3.05/min, tail 0.4 %. Median distance to the nearest lick 0.16 s; 71 % within 0.25 s; NN 100-250 ms 64 %.
  Spread / range / med_out medians 3.0 / 10.7 / -54; near-droplet 10 %.
- Flagged droplets (n = 10,959): tail **9.8 %**, last quintile **44 %**. Median distance to the nearest lick 1.26 s. Previous trial a hit only
  47 % (miss with reward 18 %, miss without reward 33 %). close_center 23 %. med_out median **+11** (outside the mouth).

**Shape and duration:**
- med_out quantiles (10 / 50 / 90 %): outside -87 / -59 / +35; removed **-158 / -67 / -42**; postcue -80 / -54 / +55;
  droplets -31 / +11 / +74.
- Static (spread < 3 and range < 12): removed 67 %, outside 53 %, postcue 45 %, isolated postcue 65 %, droplets 100 %.
- Median duration: removed 24 ms (7 frames), postcue 32 ms, cam4-matched 72 ms.
- DAQ contact in the event: removed 0.1 %.
- med_out < -120 px (far from the mouth, on the dorsal side): removed 15.6 %, outside 4.6 %, postcue 0.4 %. Concentrated in a few sessions:
  PS92_0929 (104), PS95_0825 (74), PS94_0903 (56), PS95_0910 (49), PS94_0820 (47).

**Figures:**
- `figures/fig1_shape_vs_thresholds.png`: spread, range and med_out with the thresholds.
- `figures/fig2_distance_to_licks.png`: distance to the nearest cam4 / DAQ lick; nearest neighbour.
- `figures/fig3_session_quintiles.png`: per animal x epoch.

## Reading (mine; NOT ground truth -- the blind review below is the test)

1. **Most cam1-only events outside post-cue are not droplet-like.**
   - They sit at or behind the mouth opening (med_out median -59, the same as the validated post-cue events). Flagged droplets sit out toward
     the spouts (+11).
   - They are not in the quit tail: 1.0 % vs 9.8 % for droplets. (The RESTnl mask never uses tail rest anyway: rest needs both bracketing
     trials engaged.)
   - They follow hits (89 %), and two-thirds are within 0.5 s of a cam4 / DAQ lick, i.e. inside lick bouts.
   - The 10-19 failure mode was a droplet out in front of the mouth after uncollected water. These events are not that.
   - One partial exception: PS93 pre. 41 % of its outside events are near-droplet, sitting just past the -35 px med_out cut (median -41).
2. **The events that actually remove rest frames are a different, selected subset.**
   - By construction they sit away from licks: median 1.9 s from the nearest cam4 / DAQ lick, peaking 2-3 s after a bout, just outside the 2 s buffer.
   - They are short (24 ms) and static (67 %), with no DAQ contact.
   - They sit at or behind the mouth opening: 89 % have med_out <= -44 (the 'lips' zone), and 96 % are size class lips.
   - They are not droplet-shaped (7 % near-droplet), and none are in the tail (0 %).
   - My read of about a dozen strips: in most, the jaw opens briefly and the detector's point sits in the dark mouth gap. Sometimes the tongue surface
     shows at the lips; sometimes nothing tongue-like is visible. They look like post-bout mouth / jaw movements (swallowing, chewing, tongue at the
     lips) rather than water, or licks that cam4 missed.
   - The minority with med_out < -120 px (16 %, a few sessions; e.g. PS92_0929 at -340 px, where the point falls outside the crop) is a static
     label far from the mouth, i.e. a tracking artifact, not a tongue.
3. **Quit-tail test (Priya): no end-of-session pile-up for the bulk of the set.**
   - Last quintile: 14 % for outside and 27 % for removed, vs 44 % for droplets.
   - PS92 / PS93 chronic (the largest rest losses, 16 / 23 %) are flat across quintiles.
   - Late excess appears only where the removal is small or moderate:
     - PS94 acute / subacute: 45 / 65 % in Q5, median 9-41 s from any lick, previous trial mostly a miss without reward.
     - PS95 pre / subacute: 45 / 51 % in Q5.
   - Those strata behave most like quiet-period artifacts or disengagement.
4. **Implication for RESTnlc1.**
   - Most of the extra frames removed (PS92 / PS93 chronic) come from isolated mouth-opening events at the lips. Whether those count as 'licks'
     (a tongue at the lips vs a jaw movement) is a definitional call, not a droplet question.
   - The droplet rule cannot catch them: they are static, but at the mouth opening, which is the zone the rule deliberately exempts.
   - If the review says they are not tongue, a candidate rule is "cam1-only, static, > 1 s from any cam4 / DAQ lick = not a lick", plus a
     far-from-mouth guard (med_out < -120 px). Not applied here: it would change the mask.

## Blind review sheet (for Priya)

- **File:** `<G>/restnl_cam1_compare_20261117/cam1_extra_check/review/review_sheet.html`. Self-contained (13 MB); open it in a browser.
  Answers are kept in browser storage, and **Export** writes scored.csv (id, answer, note).
- **Answers:** out / lips / no tongue: droplet / no tongue: other / unsure.
- **52 candidates, shuffled:**
  - **40 test events.** All are ISOLATED (> 0.5 s from any cam4 / DAQ lick) and outside post-cue:
    - 20 from PS92 / PS93 chronic (half of them events that removed rest frames);
    - 1 rest-removing event per other animal x epoch;
    - 6 fill events from PS92 / PS93 chronic.
  - **12 blind controls:** 6 flagged droplets outside post-cue, 6 cam4-matched contact licks.
- **Format:** as in the 10-19 rounds 2/3. Per candidate: the cam1 crop; the same crop with the detector point circled; the cam4 crop
  (7 frames, -24 to +24 ms around the peak). One new row: cam1 at -400 to +400 ms (a droplet stays put).
- **Key:** `review/key.csv` (stratum, session, timing and shape per id). `candidates.json` holds the blind images. Not scored here.

## Housekeeping

- The run was kept light: 2 workers at first. When free RAM fell to 10 GB, I stopped my own PIDs (44036, 2508, 26984) and continued with 1 worker
  behind a >= 14 GB free-RAM guard.
- No other process was touched. Nothing was deleted. No pipeline file was edited.

## Priya's blind review, scored (11-17 ~18:00)

Answers: `review/scored_priya_20261117.csv` (copied from Downloads; key = `review/key.csv`).

| group | out | lips | no tongue: other | droplet |
|---|---|---|---|---|
| controls: cam4-matched contact licks (6) | 6 | 0 | 0 | 0 |
| controls: flagged droplets (6) | 2 | 2 | 0 | 2 |
| test, PS92/93 chronic, not rest-removing (10) | 1 | 9 | 0 | 0 |
| test, PS92/93 chronic, rest-removing (10) | 0 | 6 | 4 | 0 |
| test, other animal x epoch, rest-removing (14) | 1 | 8 | 3 | 2 |
| test, fill (6) | 0 | 5 | 1 | 0 |

- Test events: 30 / 40 tongue (out or lips); rest-removing: 15 / 24 tongue.
- "No tongue: other" = 8; 7 of them have med_out < -120 px (static label far from the mouth: PS92_0929 x3, PS93_1002,
  PS94_0903, PS95_0827, PS95_0817). No tongue-scored event has med_out < -120, so a far-from-mouth guard
  (drop cam1-only events with med_out < -120 px) removes 7 / 8 with no false positives in this sample.
- Droplets that escaped the filter: 2 / 40 (PS95_0812, PS94_0823), both out in front (+49, +110 px), 12-49 s from any
  lick, after a miss.
- The "static and > 1 s from any lick" option (reading 4 above) is NOT supported: 17 tongue-scored events are > 1 s from a
  cam4 / DAQ lick.
- Flagged-droplet controls: only 2 / 6 scored droplet (4 tongue), so the droplet filter over-removes in this direction
  (removes some real licks), which is conservative for a lick-free mask.
