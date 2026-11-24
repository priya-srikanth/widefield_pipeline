# Blind check of the 4-cam 3-D jaw call at close_center (2026-11-22)

Priya approved this check (11-22) to run before any jaw-only counts are quoted. Full record, sheets, key and scores:
`M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/qc_jaw_blind_20261122/README.md`.
Code: `scripts/jaw3d_blind_review.py`.

- **A** = 40 close_center trials moved no_response -> jaw_only by the 3-D call (pool 763). **B** = 40 close_center controls,
  3-D "not moved". **C** = 20 per-camera jaw_only trials at other spouts (positive reference). Seeded and stratified by
  animal x epoch.
- Pass 2 is primary: native-resolution chin crops plus a difference row. Scores (yes / unclear / no):
  - A: 23 / 12 / 5
  - B: 7 / 16 / 17
  - C: 11 / 8 / 1

  Pass 1 (0.5x crops) was too insensitive: C was "yes" in only 3 / 20.
- A matches the per-camera positives (58 % against 55 % yes). Clear false positives are 5 / 40, all with 95th-percentile 3-D
  displacement <= 0.22 mm (4 of the 5 are PS94). A by displacement: at <= 0.2 mm, 4 of 11 are "no"; above 0.2 mm, 1 of 29.
  251 / 763 pool trials are at <= 0.2 mm.
- The 3-D call misses 7 / 40 controls with clear movement, mostly where pre-cue jaw noise is at or above the post-cue
  displacement.
- Proposal for Priya (NOT applied): keep the 4-cam call at close_center, add a 0.2 mm displacement floor (or quote the
  count with and without it), and say that close_center no_response still contains some jaw-only trials.
