# Running list 2026-11-21: four-camera 3-D re-run (overnight, window de)

Branch `pilot/side4cam-wholesession` (worktree `C:/Users/SabatiniLab/Github/wf_side4cam_pilot`), NOT pushed: Priya reviews
in the morning. Plan: `STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md` (sections 0c, 2-4), `MANUSCRIPT_C2_REFRESH.md`, and the
Mac prompt `vls_stroke_paper/PROMPT_c2_4cam_3d_refresh.md` (Phase A gate -> B tables -> C READY file).

## Decisions needed (Priya) -- newest first

- (none yet)

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
