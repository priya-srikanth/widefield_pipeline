# Why the video (LP) lick detector misses the contacting lick (11-04, read-only)

Cohort 1 (PS92-95). The 70 contact trials come from the census `CONTACT_DAQ_MISMATCH_20261104.md` Q9:
- 64 trials whose first LP lick is more than 50 ms after the first DAQ contact: 54 at 50-100 ms and 10 at > 100 ms;
- 6 trials whose first LP lick is after 3.5 s.

Sources:
- LP run v36 (`lp_r4_heatmap_20261013__clean_v36`) per-session `licks_all.csv`, `windows.npz` and `clean.npz`;
- the cam1 2-D lick events;
- the DAQ lick train;
- the video, with frames picked by the alignment templates' per-frame camera times.

The code is wf_rebuild with render profile `cohort1_rebuild_20261103`. Nothing was edited or committed, and nothing was
written to M:.

## Summary

**The lag is the LP lick ONSET, not the peak.**
- `first_lick_latency.py:34-35` takes the first `licks.csv` `on_ms` with `on_ms >= 0` (cam4). cam1 events also use
  `on_ms`, and `combined` is the earlier of the two. `lp_licks.py:394` reads `combined.fillna(cam4)`.
- `on_ms` is the first frame of the lick's VISIBLE run in cam4 (`tongue_kinematics.lick_extent`), not the detector's
  rise start.
- No re-computation on the onset was needed.

**Normal onset-to-contact lag is negative, and there is no natural gap at 50 or 100 ms (Fig. 1).**
- Over all 46,825 contact trials, the first LP onset comes before the first DAQ contact:
  - close spouts: median -16 ms, 99th percentile +4.6 ms, 99.5th +10 ms;
  - far spouts: median -48 ms, 99.9th percentile -5.5 ms.
- The distribution ends by +10 to +20 ms. Counts per 10 ms bin: 642 trials in (0, 10], 35 in (10, 20].
- Beyond that is a flat, sparse tail of about 10 trials per 10 ms bin out to 100 ms, with a small bump at 80-90 ms
  (one lick interval). Only 10 trials lie beyond 100 ms.
- So the "> 50 ms" and "> 100 ms" groups are one population of misses, and the 50-100 ms trials are not normal
  contact-to-peak lag.
  - In 52 of the 54 trials at 50-100 ms, the first cam4 lick peaks more than 60 ms after the contact. It is the NEXT
    lick, about one inter-lick interval later.
  - In the other 2, the lick touching the spout was detected but its onset was set late.

**Causes of the 64 late trials, from traces, detector re-runs and video (Table 1):**

| cause | n | where |
|---|---|---|
| (1) tongue occluded in cam4 at a close spout: a small lick, the tongue tip just reaching the spout | **34** | close_mid 27, close_ipsi 7; 30 pre-stroke; PS95 22, PS92 11 |
| -- 1a: cam1 sees it, but the cam1 droplet flag drops the event | 27 | |
| -- 1b: cam1 shows the tip at the spout, too brief to become a cam1 event | 7 | |
| (4) visible in cam4 for <= 3 frames, or held on the spout as a flat plateau: the detector makes no peak | **11** | close spouts only |
| (3) lick detected, then dropped or timed late | **11** | close_ipsi 8, close_contra 3 |
| -- 3a: peak found, dropped by gate M (next peak 56-76 ms later, under the 80 ms interval) | 8 | |
| -- 3b: lick kept, but its onset (first visible frame) falls after the contact | 3 | |
| (6) DAQ contact by a PAW on the spout; no tongue in either camera | **8** | far spouts 7, close_contra 1; mostly acute / chronic |
| (2) likelihood just under the 0.4 cutoff | 0 | |
| (5) frame drop or timing misalignment | 0 | |

Notes on the table:
- Cause (2): in the occluded trials, 2 of 680 frames within ±40 ms of the contact have likelihood 0.2-0.4. The tongue
  is absent from cam4 (likelihood about 0), not marginal.
- Cause (5): the largest frame interval within ±40 ms of any contact is 4.004 ms.

**The 6 trials with the first LP lick after 3.5 s are not misses.** The contact itself is after 3.5 s (3533-3686 ms).
LP times the contacting lick correctly (4 trials), or an earlier non-contact lick (PS92_0826 t263), or the cam1 event
(PS92_0821 t218). These are the census Q6 cause (a) late contacts.

**Split by lag band:**
- **50-100 ms (54):** 1a 24, 1b 7, 3a 8, 3b 3, 4 7, paw 5.
- **> 100 ms (10):** 1a 3, 4 4, paw 3.
- The > 100 ms group has proportionally more paw artefacts (3 of 10, against 5 of 54). Neither band is
  artefact-dominated.

**Grooming vs real-lick-missed (Priya's split of the sheets, applied per trial; section below).**
- **Grooming: 8 of the 64.** This is cause (6), a paw on the spout. 4 are clear and 4 probable.
  - Spouts: far 7 (far_mid 4, far_ipsi 2, far_contra 1) and close_contra 1.
  - Epochs: chronic 4, acute 2, pre 1, subacute 1.
- **Real lick missed: 56 of the 64**, all at close spouts (close_mid 34, close_ipsi 19, close_contra 3).
  - Epochs: pre 41, acute 8, subacute 5, chronic 2.
- **Neither camera records the real misses with a usable time.**
  - In all 56, cam4's own first lick is also > 50 ms late.
  - In 34 the tongue never appears in cam4 (likelihood about 0).
  - In 22 cam4 shows the tongue, but the detector drops it or times it late.
  - cam1 shows the tip at the spout in all 34 occluded trials. In 27 of them it is dropped by the droplet flag, and in 7
    it never becomes an event.
- **So DAQ timing (`timing_source = daq`) is the right handling until the side cameras (cam2/3) arrive.**
- **The grooming trials stay `contact`, correctly, because every one also has a later real tongue contact.** What is
  wrong is the DAQ first-lick time, which is the paw time.

**Recommendation: use a 50 ms DAQ-fallback tolerance, not 100 ms, and do not apply it at far spouts.** The cause mix does
not support 100 ms.
- **At close spouts**, when the LP onset is > contact + 50 ms, use the DAQ contact (`timing_source = daq`).
  - That is 57 trials: 56 real misses and 1 grooming (PS93_0810 t403, close_contra).
- **At far spouts**, the same condition flags grooming: 7 of 7 trials, with 0 real licks among them. Far-spout tongue
  contacts are always seen (99.9th percentile lag -5.5 ms). Keep the LP time there and set `groom_contact = True`.
- **What 100 ms would do instead:** it would leave 49 real misses timed about one lick (median 83 ms) late, and it
  would still apply the fallback to 3 grooming trials.
- **At 50 ms**, the fallback replaces 64 LP times:
  - 56 of them are real tongue contacts that LP missed or timed late, and DAQ timing is correct there;
  - 8 are paw contacts, where the LP time is the real first lick and the fallback makes the time worse.
- **At 100 ms**, it replaces only 10 (7 real, 3 paw). The other 49 real misses keep an LP time about one lick
  (median 83 ms) late.
- No normal trial lies above 50 ms. Even 30 ms would touch none, since the 99.5th percentile at close spouts is +10 ms.
- **The 20-50 ms band (35 trials, cause heuristic only; no video or paw screen):** 20 cam4 visible-but-no-peak,
  4 droplet-flagged cam1, 4 onset late, 3 gate M, 3 neither camera, 1 cam1 event late.
  - 20 of the 35 first cam4 licks peak within 60 ms of the contact, so they are the contacting lick with a hidden
    start. These are smaller timing errors.
  - 50 ms is the cleaner cut. A tighter tolerance is defensible but changes timing in a mixed band.

**Fixes that would recover misses at the source (estimates; nothing changed):**
- **(a) cam1 droplet flag.** Keep droplet-flagged cam1 events that overlap a DAQ contact.
  - Recovers 27 of the 64, plus about 4 in the 20-50 ms band.
  - Cohort-wide, 741 of the 4,847 droplet-flagged post-cue events overlap a contact. 736 of these are at close spouts
    (614 at close_mid). Far-spout droplets almost never touch a contact (5 of 1,985).
  - The 3 droplet-flagged events I checked on video were a tongue tip wrapping the spout tip. The blind review that set
    the rule did not use contacts.
  - False-lick risk is low, because something must touch the spout at the same moment.
- **(b) Gate M tie-break.** When a dropped peak spans a DAQ contact, keep it and drop the other.
  - Recovers 8, plus about 3 in the 20-50 ms band.
  - It adds no lick without a contact. It does add some ILIs of 56-76 ms.
- **(c) Onset of the contacting lick.** Use min(`on_ms`, `rise_start_ms`) when the lick spans a contact.
  - Recovers 3, plus about 4 in the 20-50 ms band.
- **Not fixable in the video detector:**
  - 1b (7) and cause 4 (11): 1-3 frame glimpses; relaxing the detector's minimums would admit noise;
  - grooming (8): no paw keypoint; handled by the far-spout flag below.
- (a) + (b) + (c) would clear 38 of the 56 real misses. A 50 ms close-spout DAQ fallback covers the other 18.

**Grooming contacts move the DAQ first-lick time, not the class.**
- In the 8 grooming trials the FIRST DAQ contact is a paw. Every one also has later real tongue contacts, so
  `contact` / hit stay right.
- What is wrong is the DAQ-timed first lick: 64-352 ms when the first tongue contact is at 196-3452 ms.
- The far-spout flag (next section) catches 7 of the 8 with no false alarms on these trials.

## Grooming vs real lick missed (Priya's split)

**Labels.** Grooming = a paw on the spout at the first DAQ contact; real lick = everything else.

| label | trial | spout | epoch | lag | call | how sure |
|---|---|---|---|---|---|---|
| PS92_0918 | 206 | far_mid | chronic | 255 | grooming | clear: paw on the spout in cam1 at the contact (sheet) |
| PS94_0819 | 475 | far_ipsi | acute | 196 | grooming | clear: forepaw reaches the spout, no tongue (sheet) |
| PS95_1005 | 482 | far_contra | chronic | 86 | grooming | clear: both paws on the spout; tongue only at +40 ms (sheet) |
| PS93_0824 | 166 | far_mid | subacute | 76 | grooming | clear in the strip (paws at the spout in both cameras) |
| PS93_0810 | 403 | close_contra | pre | 373 | grooming | probable: paws flank the spout; the cam4 "tongue" plateau is likely the paw (sheet) |
| PS93_0819 | 522 | far_ipsi | acute | 63 | grooming | probable: paws up at the spout (strip only) |
| PS95_0917 | 552 | far_mid | chronic | 62 | grooming | probable: paws under the spout (strip only) |
| PS95_0924 | 382 | far_mid | chronic | 66 | grooming | probable: paws under the spout (strip only) |
| the other 56 (per-trial list) | | close spouts | | 50-1006 | real lick missed | tongue seen in video in all checked |

How the 56 real-lick calls were checked:
- **Clear on sheets:** PS95_0809 t101, PS92_0812 t223, PS94_0810 t519, PS95_0810 t410, PS93_0822 t42, PS92_0809 t3,
  PS94_0818 t278, PS95_0807 t627, PS92_0806 t47.
- **Strips:** the rest of the 37 non-1a trials.
- **1a (27):** each has a cam1 event at the contact. 3 were checked on sheets; the others are by the same pattern.
- **PS94_0818 t278** has paws near the spout, but a detected lick spans its contact. It is counted as a real lick
  (both may be touching).
- **Can't tell from the sheets:** the 4 "probable" grooming trials, where a paw is near but contact is not certain, and
  the 1b trials, where the cam1 tip is low-contrast.

**Counts.**

| call | close_ipsi | close_mid | close_contra | far_ipsi | far_mid | far_contra | all |
|---|---|---|---|---|---|---|---|
| real lick missed | 19 | 34 | 3 | 0 | 0 | 0 | 56 |
| grooming | 0 | 0 | 1 | 2 | 4 | 1 | 8 |

| call | pre | acute | subacute | chronic |
|---|---|---|---|---|
| real lick missed | 41 | 8 | 5 | 2 |
| grooming | 1 | 2 | 1 | 4 |

The 6 trials with the first LP lick after 3.5 s are neither: they are contacts after 3.5 s that LP times correctly.

**What the grooming trials get now.**
- **Camera class:** raw `contact` and final `contact` (a contact trial cannot be stopped or after_end) in all 8.
- **DAQ lick arm** (`nolick_decoder.categorize`, DAQ gate):
  - 7 are `engaged`, with the first DAQ lick at 64-352 ms, which is the paw time.
  - PS93_0819 t522 is `late_rewarded`, with its first DAQ lick at 2850 ms.
- **Hit rate:** they score as hits.
- **Is "contact" still right?** Yes. All 8 also contain later REAL tongue contacts: 2-24 LP licks with a DAQ contact,
  the first at 196-3452 ms. The class and hit status are therefore correct.
  - **What is wrong is the first-lick time.** DAQ-timed lick-aligned windows (the primary profile) align these 8 trials
    to the paw.
  - A blanket DAQ fallback would import that error into the LP train.
  - The LP first lick (156-592 ms) is a real tongue lick in all 8. In 3 of them it is a lick without contact.

**Cheap automatic grooming flag, scored on these 64 (8 grooming, 56 real).**

| rule | hits | false alarms | notes |
|---|---|---|---|
| far spout AND first LP onset > first DAQ contact + 50 ms | **7 / 8** | **0 / 56** | misses the one close_contra case. A far spout is only reached with a full protrusion, which cam4 always sees (99.9th percentile lag -5.5 ms over 21,902 far trials), so an unseen far contact is non-lingual. Population false-alarm bound: only 7 far-spout contact trials cohort-wide exceed 50 ms, and all 7 are these grooming trials. |
| no tongue in cam4 AND no cam1 event within ±40 ms of the contact | 4 / 8 | 7 / 56 | the close-spout occluded licks (1b) look the same |
| no jaw opening at the contact (jaw keypoint, ±40 ms vs -300..-100 ms) | -- | -- | does not separate: grooming median 35 px, real-lick groups 40-55 px, ranges overlap; the jaw is often hidden (likelihood < 0.4) |
| paw / nose keypoint near the spout | not testable | | the v36 cam4 model has no paw keypoint (parts: nose, jaw, tongue, spout); the tongue keypoint sometimes lands on the paw |
| long continuous contact | not tested | | the DAQ lick train I used holds contact onsets only, no durations |

The first rule is cheap and needs no new model: `groom_contact = far spout & LP onset > contact + 50 ms`. It cannot
catch close-spout grooming, which was 1 here. The side cameras or a paw keypoint would be needed for that.

**For the real misses, does cam4 also miss?** Yes, in all 56.
- cam4's own first lick (`cam4` column) is also more than 50 ms after the contact, because `combined` is the earlier of
  cam4 and cam1.
- In 34 the tongue is not in cam4 at all at the contact: likelihood about 0, protrusion 0. The close spout sits in front
  of the mouth.
- In 22 cam4 shows the tongue, but the detector drops the peak (gate M, 8), times the onset late (3), or makes no peak
  from a 1-3 frame glimpse or plateau (11).
- cam1 shows the tip at the spout in the 34 occluded trials, but as an event that the droplet flag drops (27) or as no
  event (7).
- The DAQ contact is the only record with a correct time. Using it (`timing_source = daq`) is the right handling until
  cam2/3 can see the tongue at close_center.

## Fig. 1: onset-minus-contact lag, all contact trials (`lag_histogram_close_far.png`)

![lag](lag_histogram_close_far.png)

Panels show close and far spouts, with a log y-axis and dashed lines at 50 ms (orange) and 100 ms (red). Bottom row:
zoom on -100..300 ms in 4 ms (one-frame) bins.

| lag bin (ms, all spouts) | trials |
|---|---|
| (-20, -10] | 11,598 |
| (-10, 0] | 4,448 |
| (0, 10] | 642 |
| (10, 20] | 35 |
| (20, 30] | 11 |
| (30, 40] | 14 |
| (40, 50] | 10 |
| (50, 60] | 9 |
| (60, 70] | 8 |
| (70, 80] | 10 |
| (80, 90] | 18 |
| (90, 100] | 9 |
| > 100 | 10 |

cam4 alone (no cam1) has 1,265 close and 265 far trials with lag > 50 ms. Merging cam1 events already rescues all but
64.

## Table 1: cause x spout (canonical order), all 70

| cause | close_ipsi | close_mid | close_contra | far_ipsi | far_mid | far_contra | all |
|---|---|---|---|---|---|---|---|
| 1a occluded in cam4; cam1 event dropped by droplet flag | 5 | 22 | 0 | 0 | 0 | 0 | 27 |
| 1b occluded in cam4; cam1 tip at spout, no cam1 event | 2 | 5 | 0 | 0 | 0 | 0 | 7 |
| 3a peak found, dropped by gate M | 5 | 0 | 3 | 0 | 0 | 0 | 8 |
| 3b lick kept, onset after contact | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| 4 visible <= 3 frames / plateau: no peak | 4 | 7 | 0 | 0 | 0 | 0 | 11 |
| 6 grooming (paw on spout) | 0 | 0 | 1 | 2 | 4 | 1 | 8 |
| N not a miss (contact after 3.5 s) | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| all | 20 | 35 | 5 | 3 | 5 | 2 | 70 |

## Table 2: cause x epoch, and cause x band

| cause | pre | acute | subacute | chronic | 50-100 ms | > 100 ms | > 3.5 s |
|---|---|---|---|---|---|---|---|
| 1a | 24 | 0 | 1 | 2 | 24 | 3 | 0 |
| 1b | 6 | 1 | 0 | 0 | 7 | 0 | 0 |
| 3a | 3 | 2 | 3 | 0 | 8 | 0 | 0 |
| 3b | 1 | 2 | 0 | 0 | 3 | 0 | 0 |
| 4 | 7 | 1 | 2 | 1 | 7 | 4 | 0 |
| 6 grooming (paw) | 1 | 2 | 1 | 4 | 5 | 3 | 0 |
| N | 1 | 3 | 2 | 0 | 0 | 0 | 6 |
| all | 43 | 11 | 9 | 7 | 54 | 10 | 6 |

Per animal: PS92 22, PS93 9, PS94 5, PS95 34.
- PS95 is mostly 1a (19), all from close_mid pre-stroke sessions 0806-0812. That animal makes a quick, small first
  touch about 90-125 ms after the cue, then a full lick about 85 ms later.
- Cause 1 is therefore concentrated in one animal and one epoch. Read it as a camera-geometry issue at close spouts,
  not a stroke effect.

## What each cause looks like (contact sheets: cam4 top, cam1 bottom, red = frame nearest the DAQ contact)

**1 occluded.**
- The close spout sits directly in front of the mouth in cam4. A tongue tip that only reaches the spout tip is hidden
  behind it: cam4 likelihood about 0 and cleaned protrusion 0.
- cam1 sees the tip wrapping the spout tip for 3-13 frames. The cam1 droplet rule (cam1-only event, spread < 3 px,
  range < 12 px, median out > -35 px; `cam1_licks.py:26-32`) matches this static touch and drops it.
- Sheets:
  - `sheet_1-occluded_cam1-droplet-flag_PS95_0809_t101.png`
  - `..._PS92_0812_t223.png`
  - `..._PS94_0810_t519.png`
  - `sheet_1or6-neither-camera_PS95_0810_t410.png`: 1b, the cam1 tip is visible but no cam1 event was made.

**3a gate M.**
- The v7 detector found the contacting peak. Gate M (`tongue_detect.run_m_gate`, `m_min_interval_ms` 80; 60 when both
  peaks are inside the first 250 ms) then dropped it as the lower-confidence member of a pair 56-76 ms apart.
- Sheets: `sheet_3a-M-gate-drop_PS93_0822_t42.png` and `..._PS92_0809_t3.png`.

**3b onset late.**
- The lick spans the contact, but `on_ms` (`lick_extent`, the visible run back from the peak) starts after the contact.
- Sheet: `sheet_3b-onset-late_PS94_0818_t278.png`.

**4 no peak.**
- A 2-3 frame glimpse (35-53 px) fails the v7 rise / fall / finite-frame minimums (16 ms, 16 ms, 5 frames).
- Or the tongue is held on the spout as a flat 60 px plateau with no rise. PS92_0806 t47 then has no lick for 1 s,
  which gives the 1006 ms lag.
- Amplitude is not the problem: the protrusion is above the 20 px `min_peak_y_abs` and the 60 px `MIN_REACH_PX` is not
  used by this detector.
- Sheets: `sheet_4-short-blip-no-peak_PS95_0807_t627.png` and `sheet_4-plateau-no-peak_PS92_0806_t47.png`.

**6 grooming (paw).**
- One or both forepaws are on the spout at the contact, and the tongue comes out later.
- Sheets:
  - `sheet_1or6-neither-camera_PS94_0819_t475.png`, `..._PS95_1005_t482.png` and `..._PS92_0918_t206.png`;
  - `sheet_4-plateau-no-peak_PS93_0810_t403.png`: classed as paw after the screen; the cam4 "plateau" is the paw.
- The paw screen over the 37 non-1a trials (cam1 at -12 / 0 / +12 ms, cam4 at 0 ms) is in
  `paw_screen_strip_a.png` / `_b.png`.
- PS94_0818 t278 also has paws near the spout, but a detected lick spans its contact, so it stays 3b.

**All 70 traces.** `traces_all70.png` shows, for each trial:
- cam4 protrusion (black) and tongue likelihood (orange; dotted line = 0.4);
- every DAQ contact (blue; 0 = first contact);
- LP licks rise..fall (green; ^ = `on_ms`);
- cam1 events (red = droplet-flagged, magenta = kept);
- window -200..+300 ms.

## Uncertain / to check

- **Paw calls are visual,** from the strips and sheets, one look per trial. Paws can sit near the spout without touching
  it. The 4 "neither camera" paw trials are clear; the 4 reassigned from cause 4 (PS93_0819 t522, PS93_0810 t403,
  PS95_0917 t552, PS95_0924 t382) are likely but less certain.
- **The 1b calls** rest on a visible cam1 tip in the strip, not on a cam1 event.
- **Detector re-run.** I re-ran `run_v7_pipeline_per_trial` on the saved cleaned trace.
  - The kept licks match production in all 70 trials: same licks and intervals.
  - In 24 trials the times are shifted by a constant 4-40 ms because my cue frame differs from production's padded
    index.
  - The gate-M decisions are read from the re-run.
- **The 20-50 ms band** classes are the trace heuristic only, with no video and no paw screen.
- **Fix estimates** count only these trials. Population effects of fixes (a)-(c) on lick counts and ILIs were not run,
  except the droplet-contact overlap count.
- **Fallback timing bias.** The DAQ contact comes after a normal onset (median -16 ms close, -48 ms far). A fallback
  that inserts the contact time is therefore late relative to an LP onset by that much. Record `timing_source = daq`.
- **One `combined` value disagrees with today's cam1 file.** For PS95_0806 t380, `combined` (264 ms) ignores a kept
  cam1 event at 136 ms that is in the current events file (mtime 10-23). The attempts table may have been built from an
  older cam1 file. Not chased.

## Per-trial list (64 + 6)

| label | trial | spout | epoch | band | contact ms | LP onset ms | lag | cause |
|---|---|---|---|---|---|---|---|---|
| PS92_0807 | 28 | close_mid | pre | 50-100 | 846 | 904 | 58 | 1a |
| PS92_0809 | 201 | close_ipsi | pre | 50-100 | 252 | 332 | 80 | 1a |
| PS92_0811 | 215 | close_ipsi | pre | 50-100 | 784 | 848 | 64 | 1a |
| PS92_0812 | 25 | close_mid | pre | 50-100 | 199 | 252 | 53 | 1a |
| PS92_0812 | 223 | close_mid | pre | 50-100 | 131 | 208 | 77 | 1a |
| PS92_0814 | 265 | close_mid | pre | 50-100 | 608 | 696 | 88 | 1a |
| PS94_0810 | 519 | close_mid | pre | 50-100 | 159 | 212 | 53 | 1a |
| PS95_0806 | 308 | close_mid | pre | 50-100 | 92 | 172 | 80 | 1a |
| PS95_0807 | 157 | close_mid | pre | 50-100 | 110 | 196 | 86 | 1a |
| PS95_0809 | 101 | close_mid | pre | 50-100 | 111 | 192 | 81 | 1a |
| PS95_0809 | 265 | close_mid | pre | 50-100 | 112 | 204 | 92 | 1a |
| PS95_0809 | 391 | close_mid | pre | 50-100 | 109 | 200 | 91 | 1a |
| PS95_0809 | 460 | close_mid | pre | 50-100 | 116 | 204 | 88 | 1a |
| PS95_0809 | 574 | close_mid | pre | 50-100 | 102 | 176 | 74 | 1a |
| PS95_0809 | 628 | close_mid | pre | 50-100 | 105 | 188 | 83 | 1a |
| PS95_0809 | 651 | close_mid | pre | 50-100 | 102 | 188 | 86 | 1a |
| PS95_0810 | 165 | close_mid | pre | 50-100 | 93 | 180 | 87 | 1a |
| PS95_0810 | 487 | close_mid | pre | 50-100 | 107 | 192 | 85 | 1a |
| PS95_0811 | 36 | close_mid | pre | 50-100 | 127 | 212 | 85 | 1a |
| PS95_0811 | 224 | close_mid | pre | 50-100 | 97 | 196 | 99 | 1a |
| PS95_0812 | 79 | close_mid | pre | 50-100 | 125 | 208 | 83 | 1a |
| PS95_0812 | 81 | close_mid | pre | 50-100 | 92 | 180 | 88 | 1a |
| PS95_0914 | 128 | close_mid | chronic | 50-100 | 121 | 220 | 99 | 1a |
| PS95_0917 | 250 | close_mid | chronic | 50-100 | 123 | 192 | 69 | 1a |
| PS92_0809 | 2 | close_ipsi | pre | 50-100 | 221 | 300 | 79 | 1b |
| PS92_0813 | 241 | close_mid | pre | 50-100 | 314 | 368 | 54 | 1b |
| PS92_0814 | 70 | close_mid | pre | 50-100 | 95 | 148 | 53 | 1b |
| PS92_0819 | 40 | close_ipsi | acute | 50-100 | 208 | 268 | 60 | 1b |
| PS95_0807 | 394 | close_mid | pre | 50-100 | 87 | 172 | 85 | 1b |
| PS95_0810 | 410 | close_mid | pre | 50-100 | 112 | 192 | 80 | 1b |
| PS95_0811 | 318 | close_mid | pre | 50-100 | 104 | 200 | 96 | 1b |
| PS92_0807 | 155 | close_ipsi | pre | 50-100 | 242 | 292 | 50 | 3a |
| PS92_0809 | 3 | close_ipsi | pre | 50-100 | 222 | 312 | 90 | 3a |
| PS92_0819 | 210 | close_ipsi | acute | 50-100 | 246 | 324 | 78 | 3a |
| PS92_0821 | 283 | close_ipsi | acute | 50-100 | 924 | 988 | 64 | 3a |
| PS93_0822 | 25 | close_contra | subacute | 50-100 | 825 | 900 | 75 | 3a |
| PS93_0822 | 42 | close_contra | subacute | 50-100 | 340 | 420 | 80 | 3a |
| PS93_0822 | 264 | close_contra | subacute | 50-100 | 1385 | 1460 | 75 | 3a |
| PS94_0806 | 68 | close_ipsi | pre | 50-100 | 102 | 196 | 94 | 3a |
| PS92_0811 | 31 | close_ipsi | pre | 50-100 | 906 | 956 | 50 | 3b |
| PS92_0818 | 26 | close_ipsi | acute | 50-100 | 138 | 188 | 50 | 3b |
| PS94_0818 | 278 | close_ipsi | acute | 50-100 | 71 | 132 | 61 | 3b |
| PS92_0822 | 109 | close_ipsi | acute | 50-100 | 82 | 148 | 66 | 4 |
| PS95_0806 | 339 | close_mid | pre | 50-100 | 93 | 172 | 79 | 4 |
| PS95_0807 | 627 | close_mid | pre | 50-100 | 158 | 252 | 94 | 4 |
| PS95_0809 | 187 | close_ipsi | pre | 50-100 | 123 | 204 | 81 | 4 |
| PS95_0809 | 463 | close_mid | pre | 50-100 | 126 | 208 | 82 | 4 |
| PS95_0812 | 540 | close_mid | pre | 50-100 | 189 | 280 | 91 | 4 |
| PS95_0827 | 310 | close_ipsi | subacute | 50-100 | 226 | 320 | 94 | 4 |
| PS93_0819 | 522 | far_ipsi | acute | 50-100 | 2857 | 2920 | 63 | 6 grooming (paw) |
| PS93_0824 | 166 | far_mid | subacute | 50-100 | 176 | 252 | 76 | 6 grooming (paw) |
| PS95_0917 | 552 | far_mid | chronic | 50-100 | 114 | 176 | 62 | 6 grooming (paw) |
| PS95_0924 | 382 | far_mid | chronic | 50-100 | 130 | 196 | 66 | 6 grooming (paw) |
| PS95_1005 | 482 | far_contra | chronic | 50-100 | 70 | 156 | 86 | 6 grooming (paw) |
| PS92_0813 | 249 | close_ipsi | pre | > 100 | 293 | 396 | 103 | 1a |
| PS95_0806 | 380 | close_ipsi | pre | > 100 | 99 | 264 | 165 | 1a |
| PS95_0820 | 465 | close_ipsi | subacute | > 100 | 357 | 480 | 123 | 1a |
| PS92_0806 | 47 | close_ipsi | pre | > 100 | 74 | 1080 | 1006 | 4 |
| PS94_0809 | 491 | close_mid | pre | > 100 | 241 | 348 | 107 | 4 |
| PS95_0818 | 346 | close_mid | subacute | > 100 | 90 | 220 | 130 | 4 |
| PS95_0907 | 27 | close_mid | chronic | > 100 | 117 | 232 | 115 | 4 |
| PS92_0918 | 206 | far_mid | chronic | > 100 | 337 | 592 | 255 | 6 grooming (paw) |
| PS93_0810 | 403 | close_contra | pre | > 100 | 199 | 572 | 373 | 6 grooming (paw) |
| PS94_0819 | 475 | far_ipsi | acute | > 100 | 120 | 316 | 196 | 6 grooming (paw) |
| PS92_0821 | 218 | far_mid | acute | > 3.5 s | 3546 | 3520 | -26 | N |
| PS92_0826 | 263 | far_contra | subacute | > 3.5 s | 3686 | 3520 | -166 | N |
| PS93_0810 | 458 | close_mid | pre | > 3.5 s | 3533 | 3504 | -29 | N |
| PS93_0818 | 259 | close_ipsi | acute | > 3.5 s | 3661 | 3648 | -13 | N |
| PS93_0822 | 353 | close_contra | subacute | > 3.5 s | 3539 | 3520 | -19 | N |
| PS95_0817 | 610 | far_ipsi | acute | > 3.5 s | 3670 | 3624 | -46 | N |

## Method (wf_rebuild; file:line)

- **Trial list.** The census `M.pkl` joined to the run's `trial_attempts.csv`, with LP = `combined.fillna(cam4)` and
  lag = LP - first DAQ contact (`trial_contact_times`, cue + 70 ms .. stop). This reproduces 54 + 10 + 6.
- **Detector.** The v7 path (`tongue_kinematics.run_v7_pipeline_per_trial`) with config `orofacial_kinematics.tongue`.
  - Window: from cue + 70 ms to the trial stop.
  - Detection runs on the protrusion trace: slope 250 px/s, rise / fall >= 16 ms, >= 5 finite frames, peak >= 20 px,
    50 ms dedupe.
  - Gate M: interval 80 ms (60 ms inside the first 250 ms post-cue).
  - Visibility cutoff 0.4 (`orofacial_clean`).
- **Per-trial evidence.**
  - In ±40 ms around the contact (the contact moved onto the table's cue-frame clock by that trial's median
    `t_daq_ms - t_ms` offset): raw LP tongue likelihood, cleaned protrusion (fill 0 = measured), and frame intervals
    from `ix_daq_s`.
  - LP licks whose rise..fall ±8 ms covers the contact; cam1 events overlapping ±40 ms; every DAQ contact from
    `dlc_frames.lick_onsets`.
- **Rules, in order:**
  1. contact after 3.5 s -> N;
  2. a lick spans the contact -> 3b;
  3. a gate-M drop within -40..+100 ms and >= 2 measured frames > 20 px -> 3a;
  4. >= 2 measured frames > 20 px -> 4;
  5. a droplet-flagged cam1 event -> 1a;
  6. otherwise -> 1b or 6, decided on video. The 37 non-1a trials were screened on video, which moved 4 trials from
     cause 4 to paw.
- **Video.** Paths from `o2_inference.videos_for`. Frames from `camera_sync.templates_for` -> `frame_daq_seconds`,
  nearest frame in time to the contact + offset, as in `o2_inference.trial_frame_spans_ts`. Never frame index x fps.
- **Scripts** (session scratchpad `lpmiss/`, not committed): `list.py`, `diag.py`, `traces.py`, `feat.py`,
  `redetect.py`, `classify.py`, `final.py`, `hist.py`, `band.py`, `firstlick.py`, `dropl.py`, `sheets.py`, `strip.py`.
