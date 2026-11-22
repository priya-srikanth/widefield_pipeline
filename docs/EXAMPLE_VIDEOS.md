# Example videos (candidates for presentations)

Rendered videos worth reusing on slides: where each is, what it shows, and the command that remakes it. Add a row when
a video is made for review and Priya marks it as a keeper; never delete a listed one without asking.
Paths are under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/` (analysis desktop; `/n/files/Neurobio/MICROSCOPE/Priya/...`
on O2).

## SUPPLEMENTARY DELIVERABLE v3: far-contralesional videos, Mice 9 / 11 / 12 (2026-11-22) -- CURRENT

v2 (below) plus, at Priya's requests 11-22: (a) the bottom-view (cam1) nose drawn on EVERY frame as the fused 3-D nose
reprojected (white-ringed, "nose (3-D)"), never cam1's own point and never switching between the two -- the v2 "jump"
was that switch (cam1 2-D vs reprojection 6.5 px apart; the 3-D nose itself is smooth, see
`docs/status/POSE3D_JUMP_QC_2026-11-22.md`); (b) first-lick latency (peak time after the cue) within the animal's IQR for
the epoch, for the pre- and post-stroke picks; (c) cue and water as ONE marker at t = 0 ("cue + water (delivered
together)"), no valve-offset marker (gui_config auto_reward_delay = 0; the DAQ valve opens 6-12 ms after the cue on ~88 %
of trials and 110-130 ms on ~11 %, a controller latency, in every PS95 session checked; v2's mouse 12 day-5 picks were
~120 ms trials).

Deliverables: `session_poses/pose3d_4cam_3net_20261121_sync60/supp_videos_v2_20261122/SuppVideo_mouse{9,11,12}_far_contralesional_v3.mp4`
(+ `_win.mp4`), `SuppVideo_mapping_v3.csv`, blocks + stills in `blocks_v3/`; copied byte-identical (without `_v3`) to
`vls_stroke_paper/supplementary_videos/` with `SuppVideo_far_contralesional_mapping.csv` and the README section.

| mouse | size (win) | duration | picks (session: trial error deg, latency ms) |
|---|---|---|---|
| 9 (PS92) | 12.2 (15.2) MB | 104.5 s | pre 0813: t29 -15.9 232, t277 -17.5 472; acute 0819: t167 -49.8 836, t106 -49.0 808; subacute 0824: t279 -38.6 720, t22 -39.4 688; chronic 0908: t173 -22.5 400, t84 -23.0 184 |
| 11 (PS94) | 11.7 (14.6) MB | 83.2 s | pre 0814: t64 -23.8 176, t214 -23.4 192; acute 0821: t47 -63.7 524, t194 -63.2 280; subacute 0825: t86 -45.9 168, t361 -45.1 168; chronic 0924: t474 -45.8 164, t223 -45.3 156 |
| 12 (PS95) | 12.6 (15.8) MB | 81.2 s | pre 0814: t194 -14.4 152, t146 -14.6 144; acute 0817: t478 -53.4 268, t338 -51.1 332; subacute 0821: t145 -36.3 168, t211 -36.5 144; chronic 0928: t72 -26.1 132, t381 -26.1 128 |

Pre-stroke latency median / IQR: mouse 9 296 / 156-660 ms, mouse 11 192 / 172-236, mouse 12 156 / 144-172. Epoch medians
unchanged from v2 (table below). `_win` copies: Constrained Baseline 3.1, 1200 x 752, 30 fps, no B-frames.

    python -m scripts.pose3d_supp_clips --animal PS92 --style v3 --licks fused --n 2 --jobs 2 --out <P3D>/supp_videos_v2_20261122
    (same for PS94, PS95)

DELETED 2026-11-22 (Priya: "you can delete superseded videos"): all v2 mp4s (6 deliverables in `supp_videos_v2_20261122/`
+ 24 blocks in `blocks_v2/`, list in lp_stage/clips_20261122/deleted_v2_mp4.txt), the 6 v2 copies in
`vls_stroke_paper/supplementary_videos/` (replaced by v3 under the same names), and (by the coordinator) the v1 / cam4-list
mp4s in `clips_supp_PS94_farcontra_20261122/` and `clips_supp_PS95_farcontra_20261122/`. CSVs and stills of v1 / v2 kept.

## SUPPLEMENTARY DELIVERABLE v2 (2026-11-22) -- videos DELETED 11-22 (superseded by v3; csv / stills kept)

Priya 11-22 (v2): manuscript animal numbers (`vls_stroke_paper/ANIMAL_KEY.csv`: PS92 = 9, PS94 = 11, PS95 = 12), no lab
IDs / dates / trial numbers in frames, title cards or deliverable names; "Pre-stroke" / "Post-stroke day N (epoch)"; spout
"far contralesional"; Fig 3C v26c framing (`fig3c_four_view_strip.CROPS_V26B`, grown with a 30 px margin to hold every
keypoint of the epoch's clips at the v26c aspect) and keypoint style (small filled dots, word labels, white-ringed =
reprojected from 3-D); "spout contact" row; dotted first-lick direction vs dashed target direction; cue + water marker at
0 with the DAQ water-valve time marked separately (6-12 ms after the cue in most trials, 116-122 ms in PS95 0821),
spout arrival (strobe) and departure (trial stop), "CUE + WATER" badge for 200 ms of real time, persistent t-from-cue
readout; aim error signed + = contralesional (manuscript convention; columns `*_deg_pos_contra`).

Deliverables: `session_poses/pose3d_4cam_3net_20261121_sync60/supp_videos_v2_20261122/SuppVideo_mouse{9,11,12}_far_contralesional_v2.mp4`
(+ `_win.mp4`), `SuppVideo_mapping_v2.csv` (lab ID, session, trial, aim errors, rule, crops, cue -> valve ms, strobe / stop),
building blocks + stills in `blocks_v2/`. Copied (byte-identical, without `_v2`) to
`stroke_orofacial_pipeline_outputs/vls_stroke_paper/supplementary_videos/` with `SuppVideo_far_contralesional_mapping.csv`
and a README section with suggested legends.

| mouse | size (win) | duration | pre (contact) | acute | subacute | chronic |
|---|---|---|---|---|---|---|
| 9 (PS92) | 15.1 (18.8) MB | 126.6 s | -20 (73 %) | -50 (-30), day 2 | -38 (-17), day 7 | -23 (-3), day 22 |
| 11 (PS94) | 12.0 (15.1) MB | 89.1 s | -30 (74 %) | -64 (-34), day 5 | -46 (-16), day 9 | -46 (-16), day 39 |
| 12 (PS95) | 12.3 (15.3) MB | 80.4 s | -22 (40 %) | -54 (-32), day 1, n 39 | -36 (-15), day 5 | -25 (-3), day 43 |

(epoch-median first-lick aim error vs target, deg, + = contralesional; change from the pre-stroke median in brackets)

Rules: pre-stroke = first licks that CONTACTED the spout (DAQ), all four views (cams = all four, <= 10 % single-view
frames); session = the pre session (>= 20 contacted first licks) whose 25th percentile of ipsilesional error is closest
to the epoch's; the two licks closest to that 25th percentile (better-aimed quarter). Post-stroke = session whose median
is closest to the epoch median; the two licks closest to it (<= 10 % single-view). Why the pre rule changed: the measure
is not 0 at a hit -- pre far-contra first licks that contact the spout still end a median 18-28 deg medial of the spout
direction (PS94: 74 % contact, only 0.9 % within 15 deg of the target; PS95: 40 % contact, 24 % within 15 deg), so the
pre "median" lick looked abnormal although it hit. Selected trials (session, trial, error): PS92 pre 0813 t138 -16.0,
t29 -15.9; acute 0819 t167 -49.8, t198 -50.0; subacute 0824 t279 -38.6, t22 -39.4; chronic 0908 t173 -22.5, t84 -23.0.
PS94 pre 0814 t160 -23.7, t64 -23.8; acute 0821 t135, t258; subacute 0825 t86, t361; chronic 0924 t474, t223 (as below).
PS95 pre 0814 t194 -14.4, t146 -14.6; acute 0817 t476, t478; subacute 0821 t145, t211; chronic 0928 t417, t259.

    python -m scripts.pose3d_supp_clips --animal PS92 --style v2 --licks fused --n 2 --jobs 2 --out <P3D>/supp_videos_v2_20261122
    (same for PS94, PS95)

## SUPPLEMENTARY DELIVERABLE (v1): one far-contra video per animal, four-camera fused lick list (2026-11-22) -- videos DELETED 11-22

Priya 11-22: all figures use the four-camera fused lick list, kinematics only on licks with 3-D support; PS95 examples
too, same rule; the supplement is ONE combined video per animal (epochs pre -> acute -> subacute -> chronic, a 3 s title
card before each epoch: animal, epoch, post-stroke day, spout, epoch-median aim error and its change from pre; an epoch
banner in the epoch colour in every frame). Selection = the rule of the section below, run on
`kinematics_fused/licks.csv.gz` (context post_cue, n_3d > 0, cam1-only licks out, first lick = the earliest fused lick
after the cue by daq_peak_s, >= 15 3-D frames for the medians; `python -m scripts.pose3d_supp_clips --licks fused`).

| deliverable | size | duration | `_win.mp4` copy |
|---|---|---|---|
| `clips_supp_PS94_farcontra_20261122/supp4cam_PS94_farcontra_allepochs_fused.mp4` | 12.8 MB | 89.1 s | 16.1 MB, Constrained Baseline 3.1, 1200 x 768, 30 fps |
| `clips_supp_PS95_farcontra_20261122/supp4cam_PS95_farcontra_allepochs_fused.mp4` | 12.6 MB | 80.4 s | 15.9 MB, Constrained Baseline 3.1, 1200 x 768, 30 fps |

Building blocks beside them: `supp4cam_<animal>_<date>_<epoch>_farcontra_fused.mp4` (+ `_win`), `clips_fused.csv`,
`checks_vs_production_fused.csv` (fusion re-run vs production 3-D: PS94 32/32 part checks agree; PS95 32/32 within
0.002 mm, one part's used-keypoint agreement 0.9988), `stills/*_fused_*.png` (frame at each first-lick peak).

First-lick aim error (`midline_err_ipsi`, + = toward ipsi), epoch medians, fused list (cam4-window list in brackets):

| animal | pre | acute (change) | subacute (change) | chronic (change) |
|---|---|---|---|---|
| PS94 | +29.8 (29.8), n 682 | +63.9 (63.9), +34.1, n 264 | +46.3 (46.5), +16.4, n 408 | +45.6 (45.6), +15.7, n 739 |
| PS95 | +21.8 (21.9), n 804 | +53.8 (53.8), +32.1, n 39 | +36.4 (36.6), +14.6, n 450 | +25.1 (25.6), +3.3, n 956 |

Selected trials (session, trial, first-lick aim error): PS94 pre 0814 t433 +29.7, t154 +29.6; acute 0821 t135 +63.8,
t258 +63.7 (first lick at +1140 ms); subacute 0825 t86 +45.9, t361 +45.1; chronic 0924 t474 +45.8, t223 +45.3. PS95 pre 0813
t33 +21.8, t695 +21.8; acute 0817 t476 +53.8, t478 +53.4 (acute has few far-contra first licks: n 39); subacute 0821
t145 +36.3, t211 +36.5; chronic 0928 t417 +25.0, t259 +26.0.
Fused vs cam4-window list: PS94 changes only the second pre trial (t318 -> t154, both +29.6; ranks tied to 0.1 deg); PS95
chronic moves to session 0928 (cam4 list: 0903), and its other epochs pick different trials at the same aim error. PS94
stays 16 deg above pre at chronic; PS95 returns to within 3 deg.

    python -m scripts.pose3d_supp_clips --animal PS94 --licks fused --n 2 --jobs 2 --out <P3D>/clips_supp_PS94_farcontra_20261122
    python -m scripts.pose3d_supp_clips --animal PS95 --licks fused --n 2 --jobs 2 --out <P3D>/clips_supp_PS95_farcontra_20261122

## Supplementary manuscript clips: PS94 far contra licks by epoch, cam4-window list (2026-11-22) -- videos DELETED 11-22

Kept until Priya compares with the fused deliverable above (no epoch banner / title cards; combined file
`supp4cam_PS94_farcontra_4epochs.mp4`).

Folder: `session_poses/pose3d_4cam_3net_20261121_sync60/clips_supp_PS94_farcontra_20261122/` (`clips.csv` = every rendered
trial with its aim error, rule rank, crop sizes; `checks_vs_production.csv` = the fusion re-run against the production 3-D,
all 32 part checks agree; `stills/` = the frame at each first-lick peak). Priya 11-22: "4-view clips as supplemental (ensure
the nose is not cut off in the cam1 view and the tongue isn't cut off in the cam4 view), with the 3d tracking, of a few PS94
example far contra licks from pre, acute, subacute, and chronic (showing the aim deviation)".

Each frame: four views (cam4 front, cam1 below, cam2 left side, cam3 right side; crops padded, never clipped; cam1 crop
holds the nose with a margin and the stretch ahead of it, cam4 crop holds every tongue point and the target spout), with
the cleaned 2-D keypoints and the production 3-D point reprojected; a top-down panel (mouth zero at +, six commanded spouts,
target filled, dashed = target direction, solid = the first lick's direction from its peak extension on, with its aim
error); a side panel; a slim timeline (trial period, DAQ contacts, 3-D licks, first lick highlighted). 10x slow motion.
Aim error = `midline_err_ipsi` of the first lick (degrees out of the midline plane at peak extension, + = toward ipsi;
`kinematics/licks.csv.gz`, cam4 lick windows on the four-camera 3-D). NOTE: this metric is not 0 before the stroke
(PS94 pre median +30 deg), so the deviation to read is relative to pre; PS94's chronic median (+46) equals its subacute one
(+46), i.e. PS94 has not returned to its pre value at far contra.

Selection (rule, not by eye; `scripts/pose3d_supp_clips.py` docstring): pool = PS94 far_R first licks, August-onward,
>= 15 3-D frames; per epoch the session whose median is closest to the epoch median (>= 20 licks), then that session's
licks with <= 10 % single-view frames ranked by |aim error - epoch median|, all-licks->=3-cameras trials first; two trials
per epoch.

| video | session (epoch, day) | trials (first-lick aim error; epoch median) |
|---|---|---|
| `supp4cam_PS94_20260814_pre_farcontra.mp4` | PS94 0814 (pre, day -2) | 433 (+29.7), 318 (+29.6); median +29.8, n 673 |
| `supp4cam_PS94_20260821_acute_farcontra.mp4` | PS94 0821 (acute, day +5) | 135 (+63.8), 258 (+63.7, first lick at +1140 ms); median +63.9, n 260 |
| `supp4cam_PS94_20260825_subacute_farcontra.mp4` | PS94 0825 (subacute, day +9) | 86 (+45.9), 361 (+45.1); median +46.5, n 401 |
| `supp4cam_PS94_20260924_chronic_farcontra.mp4` | PS94 0924 (chronic, day +39) | 474 (+45.8), 223 (+45.3); median +45.6, n 734 |
| `supp4cam_PS94_farcontra_4epochs.mp4` | the four above in sequence | |

Encoding: the `.mp4` files are H.264 Main / level 4.0 / 25 fps; every one also has a `_win.mp4` copy for the default
Windows players (Constrained Baseline, level 3.1, no B-frames, yuv420p, 1200 x 736, constant 30 fps, avc1, faststart;
Priya 11-22: earlier clips opened only in VLC).

Remake (dlc env, worktree root, `MKL_THREADING_LAYER=SEQUENTIAL`, `PYTHONPATH=<worktree>`):

    python -m scripts.pose3d_supp_clips --out M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/clips_supp_PS94_farcontra_20261122 --n 2 --jobs 2

## Four-camera lick clips, production outputs (2026-11-22) -- CURRENT four-camera examples

Folder: `session_poses/pose3d_4cam_3net_20261121_sync60/clips_20261122/` (README.md there: inputs, what each frame shows,
selection rule, commands; `clips.csv` = every rendered trial; `checks_vs_production.csv`; `stills/` = PNG per trial).
Priya 11-22: "can you make new example 4-camera clips with licks, showing each camera view and the 3d projection? These
can replace the older clips you made before we ran the full cam2/3 models."

All from PRODUCTION outputs: the whole-session (sync-span) four-camera 3-D with the 60 ms video-lick floor
(`pose3d_4cam_3net_20261121_sync60`, side model `lp_side_r2b_prod_20261118` O2 bundle), the adopted fused lick train
(`camera_licks/lp_side_r2b_prod_20261118_3net_20261121_sync60_gapfix60`), DAQ contacts / rewards. Each frame: the four raw
views (cam1 below, cam4 front, cam2 / cam3 sides; fitted tiles) with cleaned 2-D keypoints (dot = used, ring = rejected)
and the production 3-D point reprojected (x); 3-D profile + front-oblique panels (commanded spouts as rings, target
filled, 120 ms trails); a lick timeline (trial period + reward, DAQ contacts, cam4 / cam1 / cam2 / cam3 events, fused
train colored by source with dropped licks + reason, 3-D licks with "M" = merged by the 60 ms floor) with a cursor and
the current lick's source; header = animal, date, epoch, trial, spout (rig name + ipsi / contra), time from cue, 10x slow
motion. Colors: nose yellow, jaw green, tongue red, spout blue. 25 fps H.264 (PowerPoint). Trials seeded, not chosen by
eye (README "Selection rule").

| video | session (epoch) | trials (spout: trial id) | what to look for |
|---|---|---|---|
| `clip4cam_PS93_20260814_trials.mp4` | PS93 0814 (pre) | close_L 83, close_center 224, close_R 102, far_L 332, far_center 64, far_R 368; cue -100 .. +700 ms | clean four-camera licks at every spout; far_L: first lick missed by cam4, kept from cam1 + both sides |
| `clip4cam_PS93_20260821_trials.mp4` | PS93 0821 (acute) | close_L 123, close_center 447, close_R 521, far_L 205, far_center 269, far_R 31; same window | acute: far_L first lick without a DAQ contact; 2 licks at the far spouts |
| `clip4cam_PS93_20260908_trials.mp4` | PS93 0908 (chronic) | close_L 293, close_center 42, close_R 166, far_L 125, far_center 128, far_R 432 | chronic: 4-5 licks per trial at every spout |
| `clip4cam_PS92_20260820_trials.mp4` | PS92 0820 (acute) | close_L 30, close_center 121, close_R 318, far_L 51, far_center 142, far_R 168 | all licks four-camera; side events split into pieces the fusion joins |
| `clip4cam_PS94_20260903_trials.mp4` | PS94 0903 (subacute) | close_L 463, close_center 435, close_R 22, far_L 65, far_center 105, far_R 388 | every lick four-camera |
| `clip4cam_PS95_20260821_trials.mp4` | PS95 0821 (subacute) | close_L 152, close_center 231, close_R 200, far_L 259, far_center 116, far_R 138 | far_L: cam4 calls one long lick where cam1, the sides, the 3-D track and the DAQ see two; the fused train follows cam4 |
| `clip4cam_PS93_20260814_bout_t193.mp4` | PS93 0814 (pre) | far_L 193, +698 .. +2198 ms | 1.5 s rhythmic bout (9 licks); DAQ contacts stop after ~+1.3 s while licking continues |
| `clip4cam_PS93_20260814_iti_t237.mp4` | PS93 0814 (pre) | far_R 237, +3942 .. +5442 ms (ITI, after the trial stop) | ITI bout (9 licks), later ones cam4 + cam1 only; whole-session windows add only ~0.2 s at the head of this session, so no clip shows licks unique to the sync span |
| `clip4cam_PS93_20260814_merge_t331.mp4` | PS93 0814 (pre) | far_R 331, +2282 .. +2982 ms | a 3-D lick merged from two tongue runs by the 60 ms floor ("M") |

Remake (dlc env, worktree root, `MKL_THREADING_LAYER=SEQUENTIAL`, `PYTHONPATH=<worktree>`):

    O=M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/clips_20261122
    python -m scripts.pose3d_lick_clips PS93:20260814 PS93:20260821 PS93:20260908 PS92:20260820 PS94:20260903         PS95:20260821 --preset 3net_20261121_sync60_gapfix60 --bout PS93:20260814 --iti PS93:20260814 --stills --jobs 3 --out $O
    python -m scripts.pose3d_lick_clips PS93:20260814 --preset 3net_20261121_sync60_gapfix60 --no-trials PS93:20260814         --merge PS93:20260814 --stills --jobs 1 --out $O

## Four-camera 3-D pose (side model r2b, 2026-11-18) -- SUPERSEDED by the 2026-11-22 clips above

Made before the whole-session cam2 / cam3 run, from PILOT side predictions on short clip folders
(`scripts/pose3d_side_video.py --fit`). Replaced by the production lick clips above (same six sessions). DELETED 2026-11-22 (Priya: "you can delete the superceded clip videos"): the six r2b `_fit.mp4` files, the
round-1 crop2 renders and `video_4cam_PS93_20260814_t540_r2b.mp4` (9 files). The `fused_*.csv.gz` tables were kept.
Were:
`session_poses/pose3d_side/r2b_20261118/video_4cam_{PS93_20260814,PS93_20260821,PS93_20260908,PS92_20260820,PS94_20260903,PS95_20260821}_fit.mp4`
(+ the `fused_<a>_<d>.csv.gz` tables beside them). The round-1 crop2 renders one folder up
(`session_poses/pose3d_side/video_4cam_PS93_*.mp4`) are the older model comparison, also superseded.
