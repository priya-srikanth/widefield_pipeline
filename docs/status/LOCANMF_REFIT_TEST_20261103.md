# LocaNMF refit vs projection test (cohort 1, swapmask variant) — 2026-11-03

Question: for the cohort 1 rebuild (decision D1 in `rebuild_20261103/RECONCILIATION.md`), must LocaNMF be REFIT on the
corrected data (`hemo_meegkit_hpfit_lag_refitT_swapmask`), or can the corrected data be PROJECTED onto the existing
footprints (fixed A, least-squares C)?

Code: worktree `C:/Users/SabatiniLab/Github/wf_swapmask` (branch analysis/cohort1-swapmask), `scripts/locanmf_refit_test/`.
Outputs: `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/locanmf_refit_test_20261103/` + local scratch
`C:/Users/SabatiniLab/locanmf_refit_test_20261103/`. Session cache: `C:/Users/SabatiniLab/.widefield_session_cache_locanmftest`.

Caveat: the final rebuild also replaces the drift step (slow band only, measured separately); this test uses the
swapmask variant as built on 11-02.

## Log

- 2026-11-03 08:30: started. GPU idle (12 MiB used). Production per-session fit = `wfield_local.batch_locanmf` ->
  `wfield_local.run_locanmf` (`compute_locaNMF`, minrank 1, maxrank 20, loc_thresh 80, r2_thresh 0.95, min_pixels 100,
  signed temporal, device auto; U_atlas + atlas + brain mask from `allen_aligned_affine8v1`). It sets NO seed.
  Joint basis = `joint_locanmf.build` (seeded 0, rank 100, same LocaNMF params).
- 2026-11-03 ~09:15: scripts written (`inventory.py`, `refit_vs_projection.py`, `seeded_run_locanmf.py`,
  `joint_refit_test.py`). Every session has block_size_max 8, so "long-block" was read as LONGEST recording:
  PS92_0810 (292k frames), PS93_0813 (313k). Sessions: pre PS92_0810, PS93_0813, PS94_0810 (470 swapped), PS95_0810;
  acute PS92_0819, PS93_0819, PS94_0819, PS95_0817; plus PS92_0828 (879 swapped, chronic) and PS93_0826 (subacute).
- 2026-11-03 ~09:40: PS93_0826 refits on corrected data: unseeded r1 100 comps (118 s), r2 100 (153 s); seeded 0
  s0a 92 (81 s), s0b 95 (133 s) -> seeding numpy + torch does NOT make the fit deterministic (stored: 93).
  Projection sanity: projecting the ADOPTED data on the stored A reproduces stored C at median r .9994 (min .974).
  Corrected projection vs stored C: median r .838 (p10 .62) -- that is the hemo change itself (this session had
  ~10% of variance in swap frames). Decode (precue / cue): stored .334 / .776, projection .386 / .749, refit r1
  .361 / .756. Remaining 9 refits running in a chain (one GPU job at a time).
- 2026-11-03 ~10:05: SAME-MACHINE STABILITY (PS93_0826, 4 refits of identical input: 2 unseeded, 2 seed 0). No two
  are identical. Pairwise Hungarian footprint r median .951-.967, p10 .61-.81, 2-9 pairs < .5, 0-8 unmatched;
  matched C r median .85-.95. Decode across the 4 refits: precue .361-.386, cue .731-.766 (ranges .025 / .035).
  Refit vs stored footprints: median .944-.954 -- the same size as refit vs refit. Projection decode (.386 / .749)
  sits inside the refit spread.
- 2026-11-03 ~10:40: all 10 per-session refits + analyses done (table below; `summary_table.md` on the share).
  Found: stored summaries record the SVTcorr path with the pre-09-30 drive letter (N:), so the adopted-data check
  first silently skipped; re-resolved through `config.svtcorr_path(mc, "meegkit_hpfit")` and re-run
  (`adopted_check` stage, also decodes the projection of the ADOPTED data = method control).
  Found: scripts run by path imported the editable install (`Github/widefield_pipeline`), not this worktree; every
  module on the per-session decode path is byte-identical between the two (checked), `joint_locanmf.build` too;
  scripts now put the worktree first on sys.path.
- 2026-11-03 ~11:00: JOINT PS94 (PS94 / PS95 both have 11 sessions; PS94 picked) on the corrected data, production
  `joint_locanmf.build` (seed 0, rank 100), separate dir `C:/Users/SabatiniLab/locanmf_refit_test_20261103/joint/`:
  161 s total (LocaNMF 67 s), 87 comps vs current a5e70d5ecd36 90. Footprint median r .946 (p10 .65, 6 < .5,
  3 unmatched); C r median vs current basis's projection of the same corrected data .857 (.971 where A >= .95).
  A second seeded build of IDENTICAL inputs: 103 comps, 377 s (LocaNMF 288 s), first-vs-second footprint median r
  .942 (p10 .67, 16 unmatched) -> the joint fit is not reproducible either; new-vs-current = rebuild noise.
  `joint_locanmf.load("PS94")` still returns a5e70d5ecd36 after both builds.

### Per-session table (refit r1 = one unseeded production fit on the corrected data)

| session | frames | n stored / refit | A r med / p10 (refit vs stored) | # A r<.5 | unmatched | proj vs stored C r med | adopted-proj vs stored C r med | precue stored / adopted-proj / proj / refit | cue stored / adopted-proj / proj / refit | refit s |
|---|---|---|---|---|---|---|---|---|---|---|
| PS92_0810 pre | 292126 | 154 / 150 | .917 / .59 | 13 | 4 | .946 | .995 | .410 / .420 / .428 / .351 | .893 / .900 / .898 / .856 | 546 |
| PS93_0813 pre | 312562 | 81 / 128 | .911 / .40 | 9 | 47 | .980 | 1.000 | .467 / .477 / .473 / .564 | .759 / .774 / .786 / .793 | 392 |
| PS94_0810 pre | 260141 | 89 / 107 | .926 / .23 | 12 | 18 | .957 | 1.000 | .616 / .613 / .611 / .581 | .866 / .867 / .887 / .855 | 248 |
| PS95_0810 pre | 231406 | 127 / 135 | .940 / .63 | 6 | 8 | .970 | .999 | .407 / .406 / .394 / .409 | .896 / .912 / .912 / .920 | 471 |
| PS92_0819 acute | 231486 | 154 / 163 | .929 / .62 | 10 | 9 | .825 | .998 | .630 / .616 / .620 / .639 | .872 / .866 / .879 / .875 | 654 |
| PS93_0819 acute | 232227 | 89 / 82 | .952 / .20 | 12 | 7 | .959 | 1.000 | .448 / .444 / .426 / .444 | .650 / .650 / .671 / .686 | 62 |
| PS94_0819 acute | 195218 | 90 / 96 | .941 / .46 | 10 | 6 | .964 | 1.000 | .318 / .312 / .399 / .382 | .672 / .661 / .690 / .713 | 51 |
| PS95_0817 acute | 226203 | 146 / 137 | .923 / .38 | 15 | 9 | .933 | .996 | .586 / .606 / .604 / .604 | .930 / .916 / .923 / .923 | 467 |
| PS92_0828 chronic | 237779 | 106 / 123 | .947 / .08 | 18 | 17 | .909 | 1.000 | .397 / .426 / .439 / .472 | .852 / .859 / .862 / .895 | 380 |
| PS93_0826 subacute | 226232 | 93 / 100 | .944 / .63 | 7 | 7 | .838 | .999 | .334 / .328 / .386 / .361 | .776 / .770 / .749 / .756 | 118 |

Decode differences over the 10 sessions (median |d| / max |d|), precue; cue: method (adopted-proj - stored) .008 / .030;
.007 / .016. Hemo with A fixed (proj - adopted-proj) .010 / .087; .013 / .029. Refit - projection .022 / .091;
.011 / .042 (mean signed +.003; +.001). Refit noise floor (PS93_0826, 4 refits): SD .011 / .015, range .025 / .035.

## Verdict (11:10)

Projection is equivalent to a refit for every per-session readout measured here, and a refit is not reproducible.
Footprints of a refit differ from the stored ones (median r .91-.95) by about as much as two refits of the same data
differ from each other (.95-.97 on PS93_0826; joint .942); decoding refit vs projection is within the refit noise
floor on average (mean +.003 / +.001), with single-session gaps up to .09 precue (PS92_0810, PS93_0813) that are
larger than the one-session floor. Recommend PROJECTION for D1. Caveats: noise floor from ONE session; the drift step
change is not in this data; LS projection reproduces the stored C less well for 150-component sessions (p10 r ~.8,
PS92 / PS95) although decoding is unaffected (method median |d| .008).
