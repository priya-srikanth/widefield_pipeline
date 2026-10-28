# Overnight log 10-27 -> 10-28 (encoding / ITI-lick window, sabatinilab-07)

Priya (10-27 late, going to sleep): "proceed and keep a running list of results, findings, decisions or questions you
need me for". Newest entries at the bottom of each section. Goal page: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d

## Queue (in order)
1. Masked-LocaNMF swap (Priya "swap masked locanmf"): basis + where decoders rerunning (scratchpad run_maskloc.sh) ->
   update page 11.5-11.7 figures + captions (and 12.1 LocaNMF sensitivity line if it changes) -> publish v141 together
   with the SVD wording edit (Priya "edit the wording").
2. ALM / tjM1 / MOs-rest / MOp-rest rows in encoder / FaceRhythm area figures (task #2; helper in a worktree: cost
   table -> implement cheap ones, prove no M: writes) -> merge -> run -> captions -> publish.
3. Anything the helper costs as expensive (re-extraction) -> WAIT for Priya (listed under Questions).
4. Housekeeping that needs no decision: handoff + FINDINGS kept current; commit each step.
5. EARLIER APPROVED BACKLOG (handoff 00-LATE + SVD_GAUSS queue part C / D; on the split basis + Gaussian reference
   decided tonight), in this order:
   5a. Four COVARIATE references for the like-for-like offset: landing only (= current) / + kinematics (reach,
       extension, peak speed, elevation) / + FaceRhythm factors around the lick / + FR + kinematics (Gaussian kernel on
       landing + linear covariate terms fit on pre licks; same session tests + mixed models).
   5b. SAME-TARGET same-place check ("more misses" vs "each lick read differently").
   5c. New per-session Haufe MAP targets (SVD basis, iti_session_patterns / lick_session_level_maps): x, y, z, angle,
       elevation, reach; pooled + per spout; epoch and epoch - pre; per-animal panels.
   5d. Per-animal change-vs-pre scatter; first-lick row on the lick-order / per-spout figure (old 3.29).
   5e. PIXEL mixed models on maps with TFCE (time model step + recovery maps, behaviour model aim map; session-label
       permutation) -- heaviest, last.
   5f. LOW: non-linear decoder; behaviour-only decoder.
NOT done without Priya: deleting v1 `sessions/`, `roi_v2_backup_20261027/`, `superseded_split_20261027/`.

## Results / findings (overnight)
- (10-27 22:40) Page v140 published: ROI v8 + split basis everywhere; changed results listed in FINDINGS 4d0-V8.
- (10-27 late) Cued lateral r chronic "4/4 -> n.s. 3/4": not a real change -- PS93 (-10 %) and PS94 (-21 %) carry it in
  v2 and v8; PS92 / PS95 sit at pre (-0.2 / -0.05 % in v2, +0.8 / -0.3 % in v8). Honest reading in both: recovers in
  2 animals, stays down in 2.

## Decisions taken (Priya, tonight)
- Basis: Allen + ALM / tjM1 split EVERYWHERE; 'beyond the pull' -> like-for-like; Gaussian reference primary,
  local-linear supplement; masked LocaNMF = the LocaNMF robustness row; SVD sentence in 12.1 reworded, not moved;
  page SOURCE files committed (PNGs excluded).

## Questions for Priya
- (carried over) FIRST-LICK weighted aim as the behaviour axis + LMM deficit? (split-half 0.98, r 0.94 with all-lick,
  chronic deficit ~2x larger) -- awaiting your OK.
- (carried over) Pixel map stats: switch to the decoder page's FWER max-stat for consistency? z-scored pattern-map check?
- (carried over) The SVD_GAUSS_SUPERSEDE_QUEUE "SVD basis = primary" plan is superseded by tonight's split-basis decision
  (marked in that doc); its controls (5a-5f) carry over on the split basis.
