# DAQ no-lick vs camera contact, unjudged close-mid, LP timing fallback (census 11-04, read-only)

Cohort 1 only (PS92-95). Rebuild code: wf_rebuild at 571b3955. The run used render profile `cohort1_rebuild_20261103`
(run tag `...pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopRun6j`): camera gate, cue_stop window,
end at the last camera lick, stopRun6j, and August-only pre.

**Sessions.** 108 curated sessions; 107 have camera classes.
- PS93_0813 has no video and is excluded.
- Two sessions have partial video coverage: PS92_0908 (68 of 389 cues unmatched, 82.5 %) and PS92_0904 (1 cue
  unmatched). Unmatched cues are left out of every count here.
- 57,048 matched trials.

## Summary

**Q6. DAQ no-lick trials that the camera calls contact: 23, and all of them are cause (a).**
- That is 0.23 % of the 10,160 DAQ no-lick trials and 0.05 % of the 46,825 camera-contact trials.
- Per animal x epoch, the count is 0-6 (PS93 pre is the highest, with 6).
- 19 of the 23 are at far spouts: far ipsi 10, far mid 5, far contra 4. The close spouts have 4.
- Every one is cause (a): the first spout contact lands at 3501-3686 ms after the cue. That is after max_rt
  (3.5 s), 0.5-162 ms before the trial stop (median 53 ms).
- The DAQ lick detector does see that same lick: its first lick is at 3522-3682 ms, quantised to imaging frames.
  So these are genuinely late contacts, not detector misses.
- **(b) table mismatch: 0. (c) other: 0.**
- In 13 of the 23 the video saw a lick well before 3 s (first camera lick 0-2464 ms), and the contact came only at
  the very end. In 8 the first video lick is itself after 3 s; 2 have no video lick.
- The same 23 trials appear in the existing `qc_camera_contact_nolick_trials.csv` (24 rows there; the extra one is
  PS95_0607, a June session).
- The reverse mismatch is also tiny: 2 DAQ-engaged trials have no camera contact.
- 182 camera-contact trials have no DAQ category, because the cue has no position code. They are outside both
  arms.

**Q4. Unjudged close-mid no_response trials: 1,121 in all; 264 of them inside the working period.**
- Every one is at close_center (= close_mid). That is 3.1 % of the 8,626 close-mid working-period trials.
- 852 of the 1,121 sit inside stopped runs; 5 are after_end.
- Inside the working period, the rate by epoch is: pre 2.0 %, acute 7.4 %, subacute 3.9 %, chronic 2.0 %.
- Per animal x epoch it runs from 0 to 9.1 %; PS94 acute is the highest (43 of 472).
- Side-camera jaw judging can only move no_response to jaw_only. Under production (`jaw_counts_in_stop`), jaw_only
  counts toward the stop run like no_response does, so the stopped trials keep their membership. **At most ~264
  isolated-lapse trials could change class.**

**Q9. Contact trials with no LP lick time (the timing fallback): 0 of 46,825.**
- No trial needs the DAQ fallback. Every camera-contact trial has an LP first-lick time (`combined`, else `cam4`) in
  the run's attempts table.
- 7 have no cam4 lick and get their time from cam1 only.
- The practical fallback case is a different one: the video **missed the contacting lick** and timed a later lick.
  - The first LP lick is more than 100 ms after the first DAQ contact in **10 trials (0.02 %)**; 64 are more than
    50 ms after. Lags run 103-1006 ms.
  - 8 of the 10 are at close spouts, which fits "close spouts worst", but the numbers are tiny.
- 6 contact trials have their first LP lick after 3.5 s. These are the late (a) contacts above; under the LP lick
  source they fall in the no-lick arm.
- Where these trials land now: raw class `contact`, final class `contact` (a contact trial cannot be stopped or
  after_end). So they are **in the working set**.

## Uncertain / to check
- **DAQ no-lick definition.**
  - I used `nolick_decoder.categorize` == "undetected" with the DAQ gate forced during that call only, as
    `scripts/camera_coverage_tables.py` does. That means no detected DAQ lick within min(max_rt 3.5 s, response
    window 3.5 s; every session's window is 3.5 s).
  - This equals `not is_engaged(...)` at decode.max_rt_s = 3.5 s. The categorize module's own `late_rewarded` arm
    (lick between 2.0 s and 3.5 s) is NOT counted as no-lick.
- **Frame quantisation.** The DAQ reaction time is quantised to imaging frames (~32 ms). A contact at 3480-3500 ms
  could be pushed past 3.5 s. None were found: the earliest (a) contact is 3501 ms.
- **Q9 counting.** "LP never saw the tongue" is counted as no `combined` / `cam4` time in the run's attempts table.
  - Note: `cam4` there is the first cam4 lick with on_ms >= 0, not capped at the stop (`first_lick_latency.py:35`).
  - The "> 100 ms after contact" count is my proxy for "the contacting lick was missed". The 100 ms threshold is a
    choice; > 50 ms gives 64.
- **Q4 working period.** I took it as final class not in {stopped, after_end, early_run, unknown}: the camera working
  period, isolated lapses included.
- **Example frame ranges.** They come from per-frame camera times through the alignment template, the
  `trial_frame_spans_ts` logic, over cue - 0.5 s .. cue + stop_ms. There is no extra 3 s post-stop pad.
- **One example dropped.** I picked 12 Q6 examples; PS92_0826 trial 263 was lost in a float-key merge, so 11 are
  shown. Its row is in the full trial CSV.

## Files (next to this one)
- `contact_daq_mismatch_q6_animal_epoch_spout.csv`
- `contact_daq_mismatch_q6_animal_epoch.csv`
- `contact_daq_mismatch_q6_trials.csv`: all 23 trials.
- `contact_daq_mismatch_q6_examples.csv`
- `contact_daq_mismatch_q4_unjudged_close_mid.csv`
- `contact_daq_mismatch_q9_lp_fallback_animal_epoch_spout.csv`
- `contact_daq_mismatch_q9_examples.csv`

## Q6 tables

Spouts are in canonical order. Cell = n (DAQ no-lick AND camera contact) / n DAQ no-lick / n camera contact.

| animal | epoch | close_ipsi | close_mid | close_contra | far_ipsi | far_mid | far_contra |
|---|---|---|---|---|---|---|---|
| PS92 | pre | 0 / 9 / 491 | 0 / 4 / 466 | 0 / 12 / 489 | 1 / 25 / 442 | 0 / 8 / 471 | 0 / 22 / 482 |
| PS92 | acute | 0 / 126 / 311 | 0 / 125 / 312 | 0 / 126 / 325 | 0 / 196 / 268 | 1 / 235 / 214 | 0 / 429 / 15 |
| PS92 | subacute | 0 / 3 / 97 | 0 / 0 / 95 | 0 / 0 / 101 | 0 / 4 / 87 | 0 / 20 / 95 | 1 / 33 / 57 |
| PS92 | chronic | 0 / 0 / 644 | 0 / 0 / 646 | 0 / 0 / 640 | 0 / 26 / 638 | 1 / 3 / 647 | 0 / 8 / 643 |
| PS93 | pre | 0 / 15 / 614 | 1 / 20 / 627 | 0 / 12 / 620 | 4 / 186 / 424 | 0 / 103 / 523 | 1 / 28 / 596 |
| PS93 | acute | 2 / 138 / 249 | 0 / 133 / 254 | 0 / 107 / 269 | 0 / 239 / 137 | 1 / 177 / 210 | 0 / 349 / 35 |
| PS93 | subacute | 0 / 56 / 219 | 0 / 46 / 225 | 1 / 38 / 237 | 1 / 89 / 177 | 0 / 53 / 212 | 1 / 101 / 166 |
| PS93 | chronic | 0 / 25 / 1000 | 0 / 19 / 993 | 0 / 14 / 1002 | 2 / 79 / 923 | 0 / 18 / 995 | 0 / 43 / 959 |
| PS94 | pre | 0 / 66 / 705 | 0 / 54 / 698 | 0 / 57 / 698 | 0 / 70 / 688 | 1 / 85 / 675 | 0 / 86 / 676 |
| PS94 | acute | 0 / 186 / 420 | 0 / 195 / 416 | 0 / 215 / 392 | 1 / 307 / 304 | 1 / 496 / 109 | 0 / 592 / 13 |
| PS94 | subacute | 0 / 91 / 409 | 0 / 85 / 415 | 0 / 80 / 420 | 0 / 94 / 406 | 0 / 105 / 395 | 0 / 146 / 354 |
| PS94 | chronic | 0 / 58 / 742 | 0 / 55 / 745 | 0 / 50 / 750 | 0 / 69 / 731 | 0 / 53 / 747 | 1 / 93 / 708 |
| PS95 | pre | 0 / 86 / 825 | 0 / 110 / 808 | 0 / 89 / 842 | 0 / 137 / 787 | 0 / 122 / 808 | 0 / 123 / 804 |
| PS95 | acute | 0 / 1 / 119 | 0 / 7 / 113 | 0 / 7 / 113 | 1 / 23 / 98 | 0 / 108 / 12 | 0 / 119 / 1 |
| PS95 | subacute | 0 / 219 / 499 | 0 / 222 / 493 | 0 / 262 / 451 | 0 / 241 / 471 | 0 / 289 / 425 | 0 / 287 / 416 |
| PS95 | chronic | 0 / 135 / 965 | 0 / 106 / 994 | 0 / 135 / 965 | 0 / 145 / 955 | 0 / 154 / 946 | 0 / 143 / 957 |


Per animal x epoch, with the cause split:


| animal | epoch | n_daq_nolick | n_cam_contact | n_nolick_camcontact | pct_of_daq_nolick | pct_of_cam_contact | n_a_after_max_rt | n_b_inside_window | n_c_other |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | 80 | 2841 | 1 | 1.25 | 0.035 | 1 | 0 | 0 |
| PS92 | acute | 1237 | 1445 | 1 | 0.08 | 0.069 | 1 | 0 | 0 |
| PS92 | subacute | 60 | 532 | 1 | 1.67 | 0.188 | 1 | 0 | 0 |
| PS92 | chronic | 37 | 3858 | 1 | 2.7 | 0.026 | 1 | 0 | 0 |
| PS93 | pre | 364 | 3404 | 6 | 1.65 | 0.176 | 6 | 0 | 0 |
| PS93 | acute | 1143 | 1154 | 3 | 0.26 | 0.26 | 3 | 0 | 0 |
| PS93 | subacute | 383 | 1236 | 3 | 0.78 | 0.243 | 3 | 0 | 0 |
| PS93 | chronic | 198 | 5872 | 2 | 1.01 | 0.034 | 2 | 0 | 0 |
| PS94 | pre | 418 | 4140 | 1 | 0.24 | 0.024 | 1 | 0 | 0 |
| PS94 | acute | 1991 | 1654 | 2 | 0.1 | 0.121 | 2 | 0 | 0 |
| PS94 | subacute | 601 | 2399 | 0 | 0.0 | 0.0 | 0 | 0 | 0 |
| PS94 | chronic | 378 | 4423 | 1 | 0.26 | 0.023 | 1 | 0 | 0 |
| PS95 | pre | 667 | 4874 | 0 | 0.0 | 0.0 | 0 | 0 | 0 |
| PS95 | acute | 265 | 456 | 1 | 0.38 | 0.219 | 1 | 0 | 0 |
| PS95 | subacute | 1520 | 2755 | 0 | 0.0 | 0.0 | 0 | 0 | 0 |
| PS95 | chronic | 818 | 5782 | 0 | 0.0 | 0.0 | 0 | 0 | 0 |


## Q6 examples (all cause (a); no (b) exists)

Times are ms from the cue. contact = first DAQ contact (the class's contact table); daq_first_lick = categorize's first detected lick, quantised to imaging frames; cam_lick = first video lick in cue..stop; stop = trial stop. Frames run from cue - 0.5 s to the stop. cam4 video paths are relative to M:/MICROSCOPE/Priya/Behavior_Cameras/Widefield/; the cam1 video is the same path with cam1_. The full paths are in the CSV.


| label | epoch | trial_id | pos | cause | contact_ms | daq_first_lick_ms | cam_lick_ms | stop | cam4_frames | cam4_cue_frame | cam1_frames | cam4_video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PS92_0809 | pre | 414 | far_ipsi | (a) | 3501 | 3522 | 3468 | 3537 | 2085650-2086659 | 2085775 | 2085643-2086651 | 20260809/PS92/cam4_2026-08-09T13_21_12.avi |
| PS93_0810 | pre | 458 | close_mid | (a) | 3533 | 3522 | 3504 | 3628 | 1985310-1986341 | 1985434 | 1985306-1986337 | 20260810/PS93/cam4_2026-08-10T14_23_19.avi |
| PS94_0810 | pre | 111 | far_mid | (a) | 3555 | 3554 | 164 | 3669 | 312852-313893 | 312976 | 312844-313886 | 20260810/PS94/cam4_2026-08-10T17_23_38.avi |
| PS93_0818 | acute | 259 | close_ipsi | (a) | 3661 | 3650 | 3648 | 3688 | 1033063-1034109 | 1033187 | 1033059-1034105 | 20260818/PS93/cam4_2026-08-18T14_52_10.avi |
| PS94_0817 | acute | 226 | far_ipsi | (a) | 3582 | 3586 | 3340 | 3744 | 544514-545574 | 544638 | 544502-545562 | 20260817/PS94/cam4_2026-08-17T09_52_26.avi |
| PS95_0817 | acute | 610 | far_ipsi | (a) | 3670 | 3682 | 3624 | 3723 | 1539150-1540205 | 1539275 | 1539147-1540202 | 20260817/PS95/cam4_2026-08-17T12_19_17.avi |
| PS92_0821 | acute | 218 | far_mid | (a) | 3546 | 3554 | none | 3548 | 912797-913809 | 912922 | 912793-913805 | 20260821/PS92/cam4_2026-08-21T20_03_35.avi |
| PS93_0822 | subacute | 353 | close_contra | (a) | 3539 | 3554 | 3520 | 3642 | 1244557-1245591 | 1244681 | 1244552-1245587 | 20260822/PS93/cam4_2026-08-22T13_36_09.avi |
| PS94_0921 | chronic | 327 | far_contra | (a) | 3531 | 3522 | 1620 | 3569 | 827251-828267 | 827375 | 827239-828256 | 20260921/PS94/cam4_2026-09-21T10_02_39.avi |
| PS93_0918 | chronic | 325 | far_ipsi | (a) | 3622 | 3650 | 0 | 3686 | 1223160-1224205 | 1223284 | 1223156-1224201 | 20260918/PS93/cam4_2026-09-18T11_54_06.avi |
| PS92_0918 | chronic | 374 | far_mid | (a) | 3528 | 3522 | 1328 | 3529 | 1790376-1791382 | 1790500 | 1790368-1791375 | 20260918/PS92/cam4_2026-09-18T14_26_42.avi |


## Q4 unjudged close-mid no_response, per animal x epoch


| animal | epoch | n_close_mid_trials | n_close_mid_working_period | n_unjudged_nr_all | n_unjudged_nr_working_period | pct_of_working_period | n_unjudged_nr_in_stopped_run | n_unjudged_nr_after_end | n_unjudged_nr_early_run |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | 470 | 469 | 4 | 3 | 0.64 | 0 | 1 | 0 |
| PS92 | acute | 437 | 342 | 117 | 22 | 6.43 | 95 | 0 | 0 |
| PS92 | subacute | 95 | 95 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| PS92 | chronic | 646 | 646 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| PS93 | pre | 646 | 640 | 18 | 12 | 1.88 | 6 | 0 | 0 |
| PS93 | acute | 387 | 275 | 129 | 17 | 6.18 | 112 | 0 | 0 |
| PS93 | subacute | 271 | 237 | 42 | 8 | 3.38 | 34 | 0 | 0 |
| PS93 | chronic | 1012 | 994 | 19 | 1 | 0.1 | 17 | 1 | 0 |
| PS94 | pre | 752 | 712 | 53 | 13 | 1.83 | 40 | 0 | 0 |
| PS94 | acute | 611 | 472 | 182 | 43 | 9.11 | 139 | 0 | 0 |
| PS94 | subacute | 500 | 448 | 69 | 17 | 3.79 | 52 | 0 | 0 |
| PS94 | chronic | 800 | 764 | 51 | 15 | 1.96 | 36 | 0 | 0 |
| PS95 | pre | 921 | 838 | 109 | 26 | 3.1 | 82 | 1 | 0 |
| PS95 | acute | 120 | 120 | 7 | 7 | 5.83 | 0 | 0 | 0 |
| PS95 | subacute | 715 | 525 | 216 | 26 | 4.95 | 190 | 0 | 0 |
| PS95 | chronic | 1100 | 1049 | 105 | 54 | 5.15 | 49 | 2 | 0 |


By epoch (working period): pre 54 / 2659 (2.0 %), acute 89 / 1209 (7.4 %), subacute 51 / 1305 (3.9 %), chronic 70 / 3453 (2.0 %).

## Q9 timing fallback

Cell = contact trials with no LP time / contact trials whose first LP lick is > 100 ms after the first DAQ contact / contact trials. Spouts are in canonical order. The full columns (no cam4 time, > 50 ms, LP first lick after 3.5 s, %) are in the CSV.

Totals:
- No LP time: 0 of 46,825.
- No cam4 time (cam1 only): 7.
- LP first lick > contact + 50 ms: 64; > 100 ms: 10 (pre 5, acute 1, subacute 2, chronic 2; 8 of 10 at close spouts).
- LP first lick after 3.5 s: 6.

Where these trials go now: raw class contact, final class contact, so they are in the working set.
- Primary profile: timing = DAQ.
- WIDEFIELD_LICK_SOURCE=lp: a trial with no LP time (none exist) or an LP first lick > 3.5 s (6) is NOT attempted in the LP train, so it lands in the no-lick arm. A trial with a late LP lick (10) is aligned to that late lick.

| animal | epoch | close_ipsi | close_mid | close_contra | far_ipsi | far_mid | far_contra |
|---|---|---|---|---|---|---|---|
| PS92 | pre | 0 / 2 / 491 | 0 / 0 / 466 | 0 / 0 / 489 | 0 / 0 / 442 | 0 / 0 / 471 | 0 / 0 / 482 |
| PS92 | acute | 0 / 0 / 311 | 0 / 0 / 312 | 0 / 0 / 325 | 0 / 0 / 268 | 0 / 0 / 214 | 0 / 0 / 15 |
| PS92 | subacute | 0 / 0 / 97 | 0 / 0 / 95 | 0 / 0 / 101 | 0 / 0 / 87 | 0 / 0 / 95 | 0 / 0 / 57 |
| PS92 | chronic | 0 / 0 / 644 | 0 / 0 / 646 | 0 / 0 / 640 | 0 / 0 / 638 | 0 / 1 / 647 | 0 / 0 / 643 |
| PS93 | pre | 0 / 0 / 614 | 0 / 0 / 627 | 0 / 1 / 620 | 0 / 0 / 424 | 0 / 0 / 523 | 0 / 0 / 596 |
| PS93 | acute | 0 / 0 / 249 | 0 / 0 / 254 | 0 / 0 / 269 | 0 / 0 / 137 | 0 / 0 / 210 | 0 / 0 / 35 |
| PS93 | subacute | 0 / 0 / 219 | 0 / 0 / 225 | 0 / 0 / 237 | 0 / 0 / 177 | 0 / 0 / 212 | 0 / 0 / 166 |
| PS93 | chronic | 0 / 0 / 1000 | 0 / 0 / 993 | 0 / 0 / 1002 | 0 / 0 / 923 | 0 / 0 / 995 | 0 / 0 / 959 |
| PS94 | pre | 0 / 0 / 705 | 0 / 1 / 698 | 0 / 0 / 698 | 0 / 0 / 688 | 0 / 0 / 675 | 0 / 0 / 676 |
| PS94 | acute | 0 / 0 / 420 | 0 / 0 / 416 | 0 / 0 / 392 | 0 / 1 / 304 | 0 / 0 / 109 | 0 / 0 / 13 |
| PS94 | subacute | 0 / 0 / 409 | 0 / 0 / 415 | 0 / 0 / 420 | 0 / 0 / 406 | 0 / 0 / 395 | 0 / 0 / 354 |
| PS94 | chronic | 0 / 0 / 742 | 0 / 0 / 745 | 0 / 0 / 750 | 0 / 0 / 731 | 0 / 0 / 747 | 0 / 0 / 708 |
| PS95 | pre | 0 / 1 / 825 | 0 / 0 / 808 | 0 / 0 / 842 | 0 / 0 / 787 | 0 / 0 / 808 | 0 / 0 / 804 |
| PS95 | acute | 0 / 0 / 119 | 0 / 0 / 113 | 0 / 0 / 113 | 0 / 0 / 98 | 0 / 0 / 12 | 0 / 0 / 1 |
| PS95 | subacute | 0 / 1 / 499 | 0 / 1 / 493 | 0 / 0 / 451 | 0 / 0 / 471 | 0 / 0 / 425 | 0 / 0 / 416 |
| PS95 | chronic | 0 / 0 / 965 | 0 / 1 / 994 | 0 / 0 / 965 | 0 / 0 / 955 | 0 / 0 / 946 | 0 / 0 / 957 |


### Q9 examples (LP first lick > 100 ms after the DAQ contact)


| label | epoch | trial_id | pos | contact_ms | daq_first_lick_ms | lp_first_ms | lag_ms | stop | cam4_frames | cam4_cue_frame | cam1_frames | cam4_video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PS92_0806 | pre | 47 | close_ipsi | 74 | 64 | 1080 | 1006 | 3680 | 180818-181862 | 180943 | 180806-181851 | 20260806/PS92/cam4_2026-08-06T12_47_40.avi |
| PS93_0810 | pre | 403 | close_contra | 199 | 192 | 572 | 373 | 3793 | 1789285-1790357 | 1789409 | 1789281-1790354 | 20260810/PS93/cam4_2026-08-10T14_23_19.avi |
| PS92_0918 | chronic | 206 | far_mid | 337 | 352 | 592 | 255 | 3866 | 1086611-1087702 | 1086736 | 1086603-1087694 | 20260918/PS92/cam4_2026-09-18T14_26_42.avi |
| PS94_0819 | acute | 475 | far_ipsi | 120 | 128 | 316 | 196 | 3646 | 1256138-1257174 | 1256262 | 1256134-1257170 | 20260819/PS94/cam4_2026-08-19T10_08_20.avi |

## Method (wf_rebuild 571b3955; file:line)

**Setup**
- The profile was applied in-process with `render_profile.apply("cohort1_rebuild_20261103")`
  (`wfield_local/render_profile.py:94`), with PYTHONPATH set to wf_rebuild.
- Sessions: `analysis_kit.curated_sessions` (`analysis_kit.py:399`), with WIDEFIELD_PRE_FROM=0801.
- Epoch: `epochs.epoch_of` (`epochs.py:434`).
- Spout: `orofacial_cohort.pos_key` (`:124`; close_center = close_mid) and `POS_ORDER` (`:69`). Lesion side is L for
  every animal.

**Camera class**
- `trial_classes.session_trials` (`:178`). The class is chosen at `:262`; contact = DAQ contact table count > 0,
  window cue + 70 ms .. stop (`CONTACT_FROM_CUE_S` `:86`).
- Matched to DAQ cues by time with `cue_classes` (`:879`, 20 ms).
- Session rules (stopped / early_run / after_end): `cue_classes_for` (`:902`) -> `classify_session` (`:632`).
- Contact time = the first DAQ contact in cue + 70 ms .. stop, from `trial_contact_times` (`:151`).
- `jaw_judged`: `:261`. Unjudged spouts: `JAW_UNJUDGED_POSITIONS` (`:85`, applied at `:251`). Only `close_center`
  occurs as a raw position name.

**DAQ no-lick**
- `nolick_decoder.categorize` (`:172`) == "undetected", with WIDEFIELD_TRIAL_GATE=daq set only around that call.
  Under camera, `:279` would move undetected + contact to late_rewarded.
- Lick train: `_load_daq_events(..., "lick_analog", ...)` (`:192`).
- First lick after the cue frame: `:221`.
- Boundary rule: `category_for_rt` (`:154`), which applies `locanmf_position_decoder.is_engaged` (`:124`) at
  min(max_rt, response window). Response window: `response_window_for` (`:112`); 3.5 s in every session.
- Cause split as in `scripts/camera_coverage_tables.py:46` (`disagreement_causes`):
  - (a) contact_ms > 3500;
  - (b) contact inside the window, sub-coded cue_frame / refractory_merge / detector-vs-contact-table;
  - (c) no contact time.

**Q4**
- Unjudged = raw class no_response AND jaw_judged False AND spout close_mid.
- Working period = final class not in {stopped, after_end, early_run, unknown}.

**Q9**
- Source: the run's attempts table, `lp_licks.attempts_path()` (`lp_licks.py:98`) =
  `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/
  sel-camera-cue_stop-endlick-cls-contact-incomplete-stopRun6j/trial_attempts.csv`.
- Its `cls_raw` agrees with the census class on all 56,979 joined trials.
- LP time = `combined.fillna(cam4)`, built by `scripts/first_lick_latency.py:35` (cam4 = first licks.csv on_ms >= 0)
  and `:49` (combined = min of cam4 and cam1).
- **How lick-aligned code gets the LP first lick (where the fallback would be wired):**
  1. `lp_licks._session_lp` reads `lat = A.combined.fillna(A.cam4)` (`lp_licks.py:394`).
  2. A cue is `attempted` only if its class is hit / incomplete AND `lat` is finite (`:401`).
  3. `edit_train` (`:330`; called at `:404`) makes the first lick after an attempted cue = cue + lat (`:341`).
     A contact trial with no lat is not attempted, so its LP licks in [cue, cue + max_rt] are removed (`:347`).
  4. Consumers swap the DAQ train for this one at `locanmf_position_decoder.py:458-459` (also
     `nolick_decoder.categorize` LP branch, `attempt_coding.py:209`, `gate_source.py:156`). The first lick is taken
     at `locanmf_position_decoder.py:493-494` and gated by `is_engaged`.
  - A fallback would go at `lp_licks.py:394-403`: when the class is hit and `lat` is NaN (or `lat` > contact +
    tolerance), use the DAQ contact time and record `timing_source = daq`.
  - Under the primary profile (no WIDEFIELD_LICK_SOURCE) lick-aligned windows are DAQ-timed anyway; the LP train is
    used only in the LP render blocks.

**Video frames**
- Paths: `o2_inference.videos_for` (`:95`).
- Per-frame DAQ times: `camera_sync.frame_daq_seconds` (`:300`) through `camera_sync.templates_for` (`:330`).
- Frame range = searchsorted over cue - 0.5 s and cue + stop, as in `o2_inference.trial_frame_spans_ts` (`:464`).
  The cue frame is the nearest frame in time.

**Scripts** (in the session scratchpad, not committed):
- `census.py` / `census2.py`: per-trial table.
- `analyze.py`: tables.
- `frames.py`: example frame ranges.
