# Running list 2026-11-21: four-camera 3-D re-run (overnight, window de)

Branch `pilot/side4cam-wholesession` (worktree `C:/Users/SabatiniLab/Github/wf_side4cam_pilot`), NOT pushed: Priya reviews
in the morning. Plan: `STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md` (sections 0c, 2-4), `MANUSCRIPT_C2_REFRESH.md`, and the
Mac prompt `vls_stroke_paper/PROMPT_c2_4cam_3d_refresh.md` (Phase A gate -> B tables -> C READY file).

## Decisions needed (Priya) -- newest first

- **3-D lick floor: 40 ms applied as authorized, but 40-80 ms looks like fragments too (small).** Pilot four-camera 3-D
  licks under the adopted split, peak-to-peak intervals (PS93 0814 / PS94 0817): <40 ms 37 / 18 (now merged), 40-60 ms
  27 / 5, 60-80 ms 22 / 6; cam4's own lick detector has 0 / 0 below 60 ms and 10 / 1 at 60-80 ms; the 3-D median is
  160 / 168 ms. So ~0.9 % of PS93 3-D licks (40-80 ms) are probably still split. Option: raise the 3-D floor to 60 or
  80 ms (cam4's empirical minimum) -- a parameter (`pose3d.lick_peaks(min_ili_ms=)`), re-applied from the frames
  without re-running the 3-D. Running with 40 ms until you decide. Script: `lp_stage/pilot4cam/split/ili_hist.py`.

## Authorizations (Priya, 11-21 ~05:30)

1. Run from this branch; no push until she reviews.
2. 40 ms min inter-lick floor on every 3-D lick track (check the 40-60 ms intervals first; stop if they look like fragments).
3. Regenerated outputs go to new folders / run tags; scripts that rewrite pages / tables in place: keep a copy of the old
   version beside it. Superseded outputs stay; list them for review + deletion when done.
4. Delete local `lp_stage/lp_side_r2b_prod_local/out/tmp*` once the merge into the bundle's out/ is verified; clear local
   disk as needed (ask if unsure).
5. Kinematics before neural analyses (pending manuscript).

## Log

- 05:00 all 274 side tasks done and verified (frame counts == timestamp CSVs; O2 1-206, 227-230, 44 on the share;
  desktop 207-226 + 263-274; other computer 231-262). Task 208 (PS95 0603 cam3): 1 more video frame than timestamp
  lines -- check the template handling at collect.
- 05:30 local outputs copied into the bundle's out/ (96 files); other computer's out_helper copying.
- 05:40 merged origin/main (PR #2: `docs/MANUSCRIPT_C2_REFRESH.md`, c2 correlation scripts) into the branch (d93fa42a;
  DECISIONS.md conflict = two additions, both kept).
- 05:45 three independent blast-radius reviewers started (forward from inputs / backward from manuscript + pages /
  recent commits + plan audit); reports to `C:/Users/SabatiniLab/lp_stage/blast_radius/`.
- June side-view spout-fit check running (Priya: same lenses June to August).
- 06:10 all 274 verified in the bundle's out/ (273 OK + task 208 known off-by-one: the video has one frame past the last
  timestamp line; frame ids are contiguous, so it is the trailing frame). Local temp folders deleted (48 GB).
- 06:15 collect for every session started (`o2_inference collect --missing-only --jobs 6`; log
  `lp_stage/rerun_20261121/logs/collect_side.log`).
- 06:30 01fa06ad: 40 ms floor in `pose3d.lick_peaks` (merges sub-floor runs over the union of frames; tests) +
  `WIDEFIELD_POSE3D_TAG` (run tag `_3net_20261121` -> `session_poses/pose3d_4cam_3net_20261121`; never on 2cam).
- 06:35 Phase A prep agent: parked fixes in the 3-D path + one-session smoke test of the tagged production run.
