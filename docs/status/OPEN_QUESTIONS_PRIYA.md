# Waiting on Priya — running list

Kept by Claude; newest at the top of each section. Answer any item whenever convenient (a line in chat is enough);
resolved items move to the bottom with the answer and date.

## Decisions / reviews

1. **Small incomplete licks** (added 2026-10-16, QC done). Lowering the detector floor from 20 to 2 px adds 0-2 licks
   per session (none at 2-10 px): `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v35\qc\lick_floor\`
   (`peak_protrusion_histogram.png`, `new_licks_10_20px_*.png`, `control_licks_20_30px_*.png`). The real limit is LP:
   it almost never reports a confident tongue within ~20 px of the mouth (1st percentile of confident distance 17-29 px;
   median likelihood within 25 px = 0.0), so a tongue at the lips is "not visible" and filled to the mouth. **Question:**
   do tiny (< ~20 px) incomplete licks matter for the deficit? If yes, the fix is a labelling round of tongue-at-the-lips
   frames (+ retrain); if no, nothing to change (the floor itself is moot). Jaw movement without a tongue lick
   (`mismatch.png`) already catches attempts the tongue tracking misses.
   *Labels check (10-16):* the closest labelled tongue point in the LP r4 training set is 26.8 px from the mouth (93
   placeable labels; none < 25 px, median 107 px; 329 of 540 frames have the tongue blank) -- the model's ~20 px
   visibility edge matches the labels. Option: a sheet of the blank-tongue labelled frames to check for tongue tips just
   past the lips that were left unlabelled.
2. **Stroke_orofacial Family 8.2 equivalent** (added 2026-10-16): cue-aligned continuous LATERAL tongue position,
   minus each animal's pre-stroke baseline, by position and epoch (their likely paper figure). Build it? (Family 8.1 —
   lick 1 / lick 2 angle minus pre — is done: `lick1_lick2_minus_pre.png`.)
3. **Which lick-phase angle convention to lead with** (added 2026-10-16): `phase_norm_angle_deg_*` (direction while the
   tongue is out) vs `phase_norm_angle_mouth0_deg_*` (stroke_orofacial's: in-mouth frames count as 0°, which makes the
   parabola). Both are rendered; pick one for the deck.
4. **This desktop's machine profile** (carried over from 10-15): `configs/paths.yaml` detects it as `analysis`, so local
   roots point at `C:/Users/sabatini` and every run needs the two `WIDEFIELD_*` overrides. Fix the profile, or keep
   the overrides?

## Clean-up of `M:\MICROSCOPE\Priya\DeepLabCut\Widefield` (my deletes are blocked by the session's safety check)

5. **Approved deletions — please run** (PowerShell):
   ```powershell
   $W = "M:\MICROSCOPE\Priya\DeepLabCut\Widefield"
   Remove-Item -Recurse -Force "$W\o2\r3_PS93_20261002", "$W\o2\lp_r4_pilot_20261015", "$W\_frame_staging_unassigned", "$W\round4_scan", "$W\cam1_round2_scan", "$W\_guide_archive"
   Remove-Item -Force "$W\LABELLING_GUIDE.bak-2026*.html"
   ```
   (Pilot outputs are byte-identical copies already in `o2\lp_r4_prod_20261015\out`.)
6. **Old LP project folders** `lightning-pose\cam4-2026-09-26` and `cam4-2026-09-28`: every labelled frame in them is
   also in `cam4-2026-10-13` (360 / 407 of 540). Proposal: keep `cam4-2026-09-28\eval_ep170_20260930` and
   `eval_occl_20260930` (round-3 evaluations cited in `POSE_MODELS.md`), delete the rest. OK?
7. **`session_poses\`** (23 GB). Keep the key model comparisons (0814 / 0821 / 0907 for all four animals) and
   `jaw_lateral\`. Proposal to delete the older runs: `*_r4` folders, empty `PS94_20260908`, `PS93_20260908`,
   `PS93_20260814_full`, `PS93_20260821_full`, `_pre_full_20261006` (~3.6 GB). Optional: the 15 `windows.mp4` clips
   (20.9 GB, regenerable from the raw video). OK / which?
8. **`_frame_staging\`** is exactly the size and file count of the DLC project folder (1.6 GB, ~4.7k files) — probably
   a duplicate. Want me to verify byte-for-byte?

## O2 (when you are on the cluster)

9. **Copy back finished videos** (transfer node, COMMANDS step 5) whenever more finish; I process them here. Jobs:
   56464442 (original, throttle 24), 56481998 (V100 re-runs), 56504738 (6 long videos, 10 h limit).
10. **Final coverage check** when `squeue -u ps150` is empty: the done.json / squeue check in
    `runbooks/o2_pose_inference.md` should print `not done: 0`; resubmit anything listed.

## FYI (no action unless you want one)

- PS94 0601 cannot be camera-aligned (cam4 dropped 274,857 frames); it is outside the curated set.
- PS92 0908: cam4 stopped ~14 min before the DAQ — its last 68 trials have no pose data (EXPERIMENT_ERRORS).

## Resolved

- 2026-10-16 — outputs location: moved to `Priya/Widefield/LP_analysis` (Priya).
- 2026-10-16 — deleting old DLC r3 O2 run, LP r4 pilot, superseded LP models, labelling scratch: approved (Priya);
  pending execution (item 5).
