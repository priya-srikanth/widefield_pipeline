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
4b. NEW (Priya decided 10-27 late, DECISIONS): (i) behaviour axis = FIRST-LICK weighted aim (all-lick = sensitivity):
    switch recovery measure / recovery link / mixed-model behaviour model / lmm figures; old vs new side by side;
    (ii) pixel map stats -> FWER max-statistic (pattern_stats / spout_pattern_stats / pattern_area_stats).
5. EARLIER APPROVED BACKLOG (handoff 00-LATE + SVD_GAUSS queue part C / D; on the split basis + Gaussian reference
   decided tonight), in this order:
   5a. Four COVARIATE references for the like-for-like offset: landing only (= current) / + kinematics (reach,
       extension, peak speed, elevation) / + FaceRhythm factors around the lick / + FR + kinematics (Gaussian kernel on
       landing + linear covariate terms fit on pre licks; same session tests + mixed models).
   5b. SAME-TARGET same-place check ("more misses" vs "each lick read differently").
   5c. New per-session Haufe MAP targets (SVD basis, iti_session_patterns / lick_session_level_maps): x, y, z, angle,
       elevation, reach; pooled + per spout; epoch and epoch - pre; per-animal panels.
   5c-bis. (Priya "yes do both", 10-27 late) ANGLE (+ reach) OFFSET-CONTRIBUTION maps, Gaussian reference
       (lick_offset_contrib_map.py logic on the angle decoder): where the extra contralesional angle reading comes from;
       uniform cortex-wide part vs patterned part. Supplementary.
   5c-ter. (same) Z-SCORED pattern-map check: per-session level maps (lateral + per spout) with each SVD component
       divided by its session frame SD (iti_basis_frame_sd), side by side with the unscaled maps -> are the ITI
       cortex-wide components real or a scale effect? Supplementary.
   5d. Per-animal change-vs-pre scatter; first-lick row on the lick-order / per-spout figure (old 3.29).
   5e. PIXEL mixed models on maps with TFCE (time model step + recovery maps, behaviour model aim map; session-label
       permutation) -- heaviest, last.
   5f. LOW: non-linear decoder; behaviour-only decoder.
NOT done without Priya: deleting v1 `sessions/`, `roi_v2_backup_20261027/`, `superseded_split_20261027/`.

## Results / findings (overnight)
- (10-28 ~01:50) PAGE v142 PUBLISHED (b9d94cd5): ALM / tjM1 rows (0.13, 2.2, 2.6, 18.5 + split supplements under 18.6 /
  18.9, 22.3, 22.4); first-lick aim (16.3-16.6, 16.9, 16.11-16.12); FWER map stats (15.1, 15.17-15.20); new 11.11-11.13
  angle / reach offset-contribution maps; 15.2 caveat + scale-compare supplements. DIRECTION / SIGNIFICANCE CHANGES vs
  v141: cued lateral pattern change significant CHRONICALLY (78 % FWER vs 32 % BH); cued sagittal pattern changes
  chronically (84 % vs 16 %); ITI lateral level subacute 25 % (BH 80 %) / chronic 20 % (BH 0 %); per-spout cued new
  chronic close-ipsi / close-contra changes (21 / 16 %); recovery link: cued lateral offset 3/4 -> 4/4, cued sagittal
  4/4 -> 3/4; with the ALL-lick aim the key-ROI response left at pre aim is +0.19 (p 0.03) while first-lick gives +0.05
  n.s. (16.12 says so). CORRECTION: angle-vs-lateral contribution maps r = 0.90-0.95 (mean maps), 0.72-0.99 per animal
  (PS93 ITI 0.72-0.78) -- not 0.92-0.98 (that was PS94 only).
- Known cosmetic issue: 16.5 session-scatter title cut off on the left (iti_recovery_link.py plot) -- fix queued.
- (10-28 ~01:30) 5c-ter DONE (4 animals): z-scored level maps confirm -- the cortex-wide SVD component's loading in the
  session patterns is 5-13x a typical component UNSCALED but 0.1-1.75x in its own units (median per animal x epoch;
  highest PS95 ITI pre 1.75, PS94 ITI subacute 1.56 / first-lick 1.63) -> the large cortex-wide parts of the unscaled
  ITI maps are mainly a VARIANCE effect. Page 15.2 gets the caveat. Page pass for v142 (ALM rows, 4b, 5c) running
  (helper editing the local page only; I review + publish).
- (10-28 ~01:10) 4b RUN DONE (16 steps, 0 errors). Held-out decoder-pattern map stats, FWER max-stat vs old bootstrap
  BH (% of cortex significant): NOT uniformly stricter on 4 animals. ITI LATERAL CHANGE maps ~0 in every epoch (0.1 /
  6.9 / 0.0 %; old FDR 0 / 0 / 0) -> the ITI lateral pattern does not significantly change shape; ITI lateral LEVEL maps
  19 / 1 / 25 / 20 % (pre / acute / sub / chronic). CUED LATERAL CHANGE 78 / 26 / 78 % (old 87 / 50 / 32 %) -> cued
  lateral pattern changes acutely AND chronically (chronic stronger than before). Cued sagittal change 0 / 0 / 84 % (old
  0 / 0 / 16 %). ITI sagittal change 38 / 0.2 / 89 %. Captions 15.17-15.20 need these (decoder_pattern_stats.csv,
  spout_pattern_stats*.csv).
- (10-28 ~00:45) 5c-bis DONE (4 animals; offset_contrib/*_angle_* / *_reach_*, figures lick_offset_contrib_{angle,
  reach,angle_vs_lateral}.png). ANGLE same-place offset = almost entirely PATTERNED: ITI +5.7 / +3.0 / +3.5 deg (mean of
  animals; uniform cortex-wide part -0.10 / -0.03 / -0.29 deg, i.e. ~0); 4/4 positive every ITI epoch (PS92 2.5 / 3.5 /
  1.9, PS93 1.8 / 0.5 / 1.0, PS94 7.1 / 4.5 / 5.8, PS95 11.3 / 3.6 / 5.3); cued +2.6 acute (4/4), +1.2 subacute, ~0
  chronic. REACH: read SHORTER, mostly patterned, cued subacute -0.28 mm (4/4), chronic -0.15 (4/4); ITI small. The
  uniform part being ~0 matches the lateral result: no global-gain artefact in the angle offset either.
- (10-28 ~00:45) 5c-ter first run FAILED at its last step: the held-out comparison read a pattern-stats column that 4b
  renamed (frac_sig_max -> frac_sig = FWER). Reader fixed to accept both (5b217bce); levels_z rerunning (run_5c_levels.sh).
- (10-28 ~00:30) 4b merged (b7af100d) + running (run_4b.sh, ~30-45 min). (i) FIRST-LICK weighted aim (WFIELD_AIM, default
  first; all-lick tagged _aimall). Per animal chronic - pre, all-lick -> first-lick: PS92 -0.5 -> -2.2, PS93 -3.0 ->
  -6.7, PS94 -5.7 -> -11.8, PS95 -2.4 -> -4.3 deg; deficit left after acute 4/39/27/15 % -> 14/54/48/24 %; ranking
  unchanged (PS92 > PS95 > PS93 > PS94). Mixed model: day-1 -19.7 deg, +4.0 per e-fold; epochs -17.8 / -9.5 / -6.2.
  BEHAVIOUR MODEL CHANGES (deficit was far-contra time-to-contact in mixed_models, now first-lick aim): key-ROI response
  at deficit 0 0.36 (p 1e-4) -> 0.05 n.s.; ITI lateral offset 0.25 -> 0.19 (still sig); cued lateral offset -0.01 n.s.
  -> -0.11 (p < 1e-4); cued lateral r -0.08 -> 0.005 n.s. Recovery link (offset vs first-lick aim, within animal, time
  removed): r -0.53, p 0.0002, 4/4 negative (far-contra angle gave -0.54). (ii) map stats = permutation FWER max-stat
  (solid = FWER p < 0.05, dashed = + >= 3/4 agree; old FDR kept as columns). Much stricter, e.g. PS92 test: ITI lateral
  subacute change 82 % (FDR) -> 1 % (FWER) of cortex. Pages 15.17-15.20 / 16.3-16.6 / 16.9 captions change.
- (10-28 ~00:15) 5c-bis / 5c-ter merged (24767177, a4d7307a) and running on M: (run_5c.sh, ~35 min). PS94 test results
  (full 4-animal numbers follow): ANGLE offset is mostly PATTERNED -- ITI chronic 5.78 deg = 0.82 global + 4.96 pattern
  (uniform part ~14 %); angle and lateral contribution maps correlate r 0.92-0.98 -> the angle offset comes from the same
  cortex as the lateral one. Single-area terms cancel (collinear), so read maps, not single areas.
  Z-SCORED MAP CHECK (all 4 animals): the large cortex-wide components of the unscaled ITI level maps do NOT survive
  scaling (global share 0.1-0.9 -> <= 0.07); the cortex-wide SVD component's coupling to the lick target is typical in
  its own units (0.1-1.1x RMS; exceptions PS95 ITI pre 1.75, PS94 ITI subacute 1.56) -> its dominance of the unscaled
  ITI maps is a VARIANCE (scale) effect, not a specific cortex-wide lick signal. Caveat: z-scoring up-weights
  low-variance high-frequency components, so a ~0 global share is partly built in; the feature-space loading test is the
  real evidence. -> page 15.2 claim "ITI maps carry large cortex-wide components" needs the caveat.
- (10-27 ~23:50) ALM / tjM1 rows (task #2): cheap figures implemented + merged (e87ba04f; plan / cost table
  docs/status/ALM_TJM1_ROWS_PLAN_20261027.md): 22.3 / 22.4 (FR direction), 2.2 / 2.6 residual panels, 18.5 (+ _split
  companions of 18.6 / 18.9), 0.13. Regenerating on M: now (run_alm_rows.sh). Checks: block-sum area R2 vs exact <= 0.0008;
  ALM + MOs_rest reproduce MOs within 0.05 (x1e3 dF/F). Example FR-A lick chronic residual z: tjM1 contra 2.72 [1.60,
  3.83] ~ MOp contra 2.76; ALM contra 1.29 ~ MOs contra 1.31.
- (10-27 ~23:10) QUEUE 1 DONE: page v141 = masked joint LocaNMF in 11.5-11.7 (ITI lateral -22 / -17 / -20 %, 4/4
  each; cued -20 / -11 / -8 %; same picture as unmasked) + SVD wording in 12.1.
- (10-27 22:40) Page v140 published: ROI v8 + split basis everywhere; changed results listed in FINDINGS 4d0-V8.
- (10-27 late) Cued lateral r chronic "4/4 -> n.s. 3/4": not a real change -- PS93 (-10 %) and PS94 (-21 %) carry it in
  v2 and v8; PS92 / PS95 sit at pre (-0.2 / -0.05 % in v2, +0.8 / -0.3 % in v8). Honest reading in both: recovers in
  2 animals, stays down in 2.

## Decisions taken (Priya, tonight)
- Basis: Allen + ALM / tjM1 split EVERYWHERE; 'beyond the pull' -> like-for-like; Gaussian reference primary,
  local-linear supplement; masked LocaNMF = the LocaNMF robustness row; SVD sentence in 12.1 reworded, not moved;
  page SOURCE files committed (PNGs excluded).

## Questions for Priya
- (INCIDENT, 10-27 ~23:55) The harness flagged the 4b helper (first-lick aim / FWER stats) for "interfering with
  workloads". The three idle `pytest -q` python processes seen earlier (PIDs 3240, 28260, 13112; possibly another
  session's) are gone; my own runs (ALM rows, 5c helper) are intact. Helper's account: it stopped its OWN runaway
  harness (PIDs 27496, 408) with a "stop every process whose ParentProcessId = <pid>" filter; Windows parent PIDs go
  stale and get reused, so the PID-408 step very likely killed the two idle pytest processes (3240, 13112; 28260 was
  already gone/reused). No other process actions. Lesson saved to memory (helpers stop only exact PIDs they started).
  Nothing to undo -- they were idle test runs; flagging in case they were another session's.
- (ALM rows, 10-27 ~23:50) Encoder / FR area rows use the v8 ALM / tjM1 ROIs INTERSECTED with the encoder's 16-px stat
  mask (the mask every existing encoder area value uses), so ALM + MOs_rest = MOs exactly; ~11 % of v8 ALM / tjM1 pixels
  (thin lateral rim + ALM front line) fall outside it. Keep (consistent with existing rows), or include the rim where
  the source has those pixels (FaceRhythm block maps; one-line change)? On the stat mask ALM = 0.38 of MOs, tjM1 = 0.47 of MOp.
- (ALM rows) EXPENSIVE, not done (need refits): 0.3 / 0.4 / 0.9-0.11 frozen Musall R2 (basis reruns ~1 h, or exact area
  reruns ~2-2.5 h per frame); 21.4 / 21.11 FaceRhythm trial-vs-PETH (O2 rerun, ~5 h collection); 2.2 / 2.6 Layer-2 panels
  (~19 min frA; ~15 h for the random-split ref). Which, if any?
- ANSWERED 10-27 late: first-lick aim -> yes (primary); pixel stats -> FWER max-stat (yes). 
- ANSWERED: 5c-bis + z-scored map check -> "yes do both" (queued as 5c-bis / 5c-ter).
- (carried over) The SVD_GAUSS_SUPERSEDE_QUEUE "SVD basis = primary" plan is superseded by tonight's split-basis decision
  (marked in that doc); its controls (5a-5f) carry over on the split basis.
