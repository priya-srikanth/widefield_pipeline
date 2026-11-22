# Example videos (candidates for presentations)

Rendered videos worth reusing on slides: where each is, what it shows, and the command that remakes it. Add a row when
a video is made for review and Priya marks it as a keeper; never delete a listed one without asking.
Paths are under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/` (analysis desktop; `/n/files/Neurobio/MICROSCOPE/Priya/...`
on O2).

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
