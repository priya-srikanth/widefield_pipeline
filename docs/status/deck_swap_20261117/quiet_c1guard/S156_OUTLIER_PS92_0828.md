# Slide 156 outlier: PS92_0828 (chronic, day 11), 11-18

Priya 11-18: "can the y axis be a bit lower, esp for limb ... and can we look into the outlier chronic session that has
> 80 orofacial events?" Run folder: `v2_quiet_lp_c1guard_k375_20261117/fr_regressed_own` (face regressed, k = 4).

## What it is

| | orofacial ipsi | orofacial contra | limb ipsi | limb contra |
|---|---|---|---|---|
| table rate (8.4 quiet min) | 80.1 | 91.7 | 60.2 | 66.8 |
| task period only (to last cue + 10 s; 2.4 min) | 55.8 | 68.8 | 69.6 | 78.0 |
| post-task tail (6.0 min) | 89.7 | 100.8 | 56.4 | 62.3 |
| other PS92 chronic sessions (11) | 3-16 | 2-18 | 13-24 | 13-29 |

1. **Post-task tail counted as engaged.** The last cue is at 120.2 min; imaging runs to 126.8 min. Every trial was a
   contact, so the quit is censored and `quiet_area_spikes.session_traces` keeps `engaged` = the whole recording. The
   animal stops running and licking after the task ends (quiet 2 % of frames before 118 min, 64 % after), so **6.0 of
   the session's 8.4 quiet minutes and ~80 % of its orofacial events come after the task ended**. In that tail
   the cortex shows a regular global event every ~2 s (same in orofacial, limb and whole cortex, each event after a small
   face movement): a disengaged-rest pattern, like the pre-stroke quit period (T4).
2. **The task-period rate is still high but thin.** 2.4 min of quiet, all in brief gaps between running bouts
   (PS92 runs almost continuously in this session); quiet SD 0.86 vs 0.38-0.51 in the other PS92 chronic sessions;
   face-model cross-validated R2 -0.02 (others 0.3-0.6), so the face regression removes nothing here (raw 84 vs
   residual 80). Not a denominator artifact alone, not a single burst; unreliable rather than clearly artifactual.
3. **The same tail problem in other sessions (quit censored, imaging after the last cue):** PS93_0806 (pre)
   11.7-min tail: orofacial ipsi 22.2 -> 1.8 per min and contra 31.5 -> 3.1 per min with the tail removed. So the tail
   inflates a PRE value as well. PS92_0908 has a 14-min tail, but its face video ends at ~107 min, so the tail was
   already excluded (face_frac 0.84). All other tails are < 2.2 min.

## Decision needed (Priya)

Should `engaged` end at the last cue (plus the response window) when the quit is censored? That would change
`session_traces` for every quiet-event output, and would need a re-run of the quiet chain on the cam1-guard folder.
Until then, slide 156 keeps every session and draws PS92_0828 at the top edge with its values; the notes say why.

## Figure

`quiet_spikes_v2_fr.py --plot-only` -> `fr_regressed_own/quiet_events_face_regressed_2x2.png`. This is a native 2x2
(the old image was cut from the 1x4). Each row has its own y top, = 1.1 x the row's 99th-percentile session (76 for
orofacial, 43 for limb). Swapped onto the deck with `s156_ylim_swap.py`; backup
`Overview_2026-11-06.bak_pre_a2_s156ylim_20261118.pptx`. Diagnosis scratch (trace plots): the a2 session scratchpad,
`trace_0828.py`.
