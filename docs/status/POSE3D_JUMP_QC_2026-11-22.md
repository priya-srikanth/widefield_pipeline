# 3-D keypoint jump QC, four-camera production run (2026-11-22)

Priya 11-22: "in the mouse 9 pre-stroke clip example, the nose point jumps with '3d' interpolation from the cam1 view -
do these nose jumps happen a lot?" Script `scripts/pose3d_jump_qc.py` (read-only on
`session_poses/pose3d_4cam_3net_20261121_sync60`); outputs `.../qc_jumps_20261122/` (`sessions.csv`,
`steps_by_views.csv`, `strip_*.png`).

## The jump in the mouse 9 clip is a DISPLAY switch, not a 3-D jump
Mouse 9 (PS92 0813) pre-stroke trial 138: cam1 lost the nose on 66 of 429 frames (11 drop-outs of 7-28 frames); the v2
renderer then drew the fused 3-D nose reprojected into cam1 (white ring). The 3-D nose itself is smooth there (step
median 0.02 mm, max 0.17 mm per 4 ms; 4 views when cam1 tracks, 3 when not). The cam1 2-D nose and the reprojection
differ by 6.5 px median (p90 9.2, max 12.5), so each switch moves the drawn dot 12-17 px. (Trial 29 had no drop-out.)
My 11-22 report that no reprojected nose appeared in the clips was wrong (it was read off the stills only).
Proposal for the videos: draw the cam1 nose only where cam1 tracked it (no reprojected fill); smoothing the 3-D nose
would not remove the 2-D vs reprojection offset.

## Cohort numbers (126 sessions, consecutive 4 ms frames within a trial)
| part | jumps > 1 mm per 10k steps | spikes > 1 mm (back within 3 frames) per 10k | trials with a spike > 1 mm | 3-D licks with one |
|---|---|---|---|---|
| nose | 0.30 | 0.055 | 0.48 % | 0.001 % |
| jaw | 0.95 | 0.32 | 3.4 % | 0.04 % |
| tongue | 369 (real fast movement) | 1.66 | 4.2 % | 0.29 % |

Nose jumps > 1 mm sit where the support is thin AND the view set changes: 2 views + set changed = 59 % of nose jumps
(548 per 10k such steps), one-camera fill (views -1) 20 %, 3 views + change 10.5 %; 0 jumps on 113 M four-view steps
with an unchanged set. Typical case (strip `strip_nose_PS92_20260813_t2_+424ms.png`): the paws over the face
(grooming), the nose seen by only two cameras alternating between pairs. By animal / epoch, nose spikes > 1 mm are
0-0.17 per 10k; trials affected 0-1.6 % (highest PS92 chronic 1.6 %, PS92 pre incl. June 1.0 %).

## What uses the 3-D nose
- `scripts/jaw_onsets.py` appends `nose3d_*` columns (displacement, rest, back = head pulled back, moved / onset) to
  `qc/jaw_onsets.csv`; they are DESCRIPTIVE: `trial_classes` reads only the `jaw3d_*` columns.
- `scripts/stopped_trial_traces.py` plots `nose3d_rest_mm` (pre-cue displacement, diagnostic figure).
- The manuscript (main.tex, manuscript_code) uses no 3-D nose measure; the face-prelaunch analysis uses the cam4 2-D nose.
So the nose jumps (0.5 % of trials, mostly grooming / 2-view frames) cannot move a class, a figure or a manuscript
number; the `nose3d_back_mm` / `_disp_mm` 95th percentiles could be inflated on those trials if they are ever used.
Jaw: the 3-D jaw call feeds the close_center jaw_only class (four-camera source); its onset rule needs a sustained
deflection (5 of 8 frames), so <= 3-frame spikes do not trigger it, but the 95th-percentile `jaw3d_disp_mm` can see
them (3.4 % of trials have a jaw spike > 1 mm).
