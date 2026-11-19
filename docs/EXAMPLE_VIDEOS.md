# Example videos (candidates for presentations)

Rendered videos worth reusing on slides: where each is, what it shows, and the command that remakes it. Add a row when
a video is made for review and Priya marks it as a keeper; never delete a listed one without asking.
Paths are under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/` (analysis desktop; `/n/files/Neurobio/MICROSCOPE/Priya/...`
on O2).

## Four-camera 3-D pose (side model r2b, 2026-11-18)

`session_poses/pose3d_side/r2b_20261118/video_4cam_<animal>_<date>_fit.mp4` -- Priya 11-18: "it looks good! ... we may
want them as examples for presentations".

| video | trials | notes |
|---|---|---|
| `video_4cam_PS93_20260814_fit.mp4` | 1 per spout position, cue -100 .. +700 ms | pre-stroke; heavy water droplets (492 droplet-rule frames) |
| `video_4cam_PS93_20260821_fit.mp4` | same | acute; licks with a droplet on the rod (the side tongue rejected by the fusion) |
| `video_4cam_PS93_20260908_fit.mp4` | same | chronic; clean far-left lick at ~+12 ms (all four views on the tongue tip) |
| `video_4cam_PS92_20260820_fit.mp4` | same (Part A clip set, cue -1000 .. +296 ms available) | session contributed side training frames: illustration, not an independent test |
| `video_4cam_PS94_20260903_fit.mp4` | same | side cameras fill 17 cam1 spout frames (spout 73 % -> 100 %) |
| `video_4cam_PS95_20260821_fit.mp4` | same | nose cam1 + cam4 36 % -> 100 % |

What each frame shows: the four raw views (cam1 below, cam4 front, cam2 / cam3 sides; each tile fitted to the nose, the
six commanded spouts and every used keypoint -- `--fit`), the CLEANED 2-D keypoints (dot = used in the 3-D point, ring
= rejected by the robust fusion), the fused 3-D point reprojected (x, with that camera's offset: cam1 spout edge, side
nose / tongue), and the 3-D pose from cam2's side and a front-oblique view (commanded spouts as rings, the trial's target
filled; 120 ms tongue / jaw trails). Colors: nose yellow, jaw green, tongue red, spout blue. 25 fps (10x slow motion).

Numbers behind them (POSE_MODELS Part E, "r2b results" / "O2 side bundle"): with cam2 / cam3 the jaw is 3-D on 100 % of
frames at every position (cam1 + cam4 alone: close_center 0.15-0.83, far_center 0.26-0.67), the tongue gains ~10-25 %
more frames at the close spouts (PS93, cue .. +700 ms), and the cameras agree to a median 0.03-0.32 mm.

Remake (dlc env, repo root; side predictions from `litpose predict` of the side model on the clip folder):

    python -m scripts.pose3d_side_video PS93:20260814 PS93:20260821 PS93:20260908 --fit \
        --preds C:/Users/SabatiniLab/lp_stage/lp_side_r2b_20261118/r2_full/models/side_r2b_r2_full/video_preds \
        --out M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_side/r2b_20261118
    python -m scripts.pose3d_side_video PS92:20260820 PS94:20260903 PS95:20260821 --fit \
        --clips inference_check_20261104_partA_clips --preds <same> --out <same>

Older four-camera renders with the round-1 side model (crop2, fixed tiles) stay one folder up
(`session_poses/pose3d_side/video_4cam_PS93_*.mp4`) as the model comparison.
