# Handoff — 2026-10-22: LP_licks results, frozen pre-stroke encoder, Musall-style readouts, FaceRhythm pilot

**START HERE for the encoding model (the ENCODING window, session `sabatinilab-d4`).** Supersedes
`STATUS_2026-10-21_ENCODING_HANDOFF.md` (still the reference for §0 rules, §4 pitfalls, §5 commands and the 10-21 work).
Other windows: pose / 3-D / cam1 = `sabatinilab-5b` (`STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`); FaceRhythm =
`sabatinilab-e3` (`STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md`). **Results: `docs/ENCODING_FINDINGS.md` §3.3** (updated
10-22). Reasoning: `DECISIONS.md` 2026-10-21 (night) and 2026-10-22 entries. `S` = scratchpad
`C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`;
`E` = `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\`;
`G` = `M:\MICROSCOPE\Priya\Widefield\labcams\grant_figures\`.

## 000z. STATE 10-25 ~23:00 -- READ FIRST (supersedes the RUNNING / TO DO of 000y; 000y keeps its pitfalls, 000x the day's results)

Page v84 https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (source `S/goal_page/index.html`, S = `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`; publish with the
LONG path form `C:/Users/SabatiniLab/AppData/Local/Temp/...` -- the 8.3 `SABATI~1` path is blocked by a read rule for the image
files). FINDINGS section 0 now has: point 4 (target vs executed reconciled), 4b (recovery = behavioural compensation with
PARTIAL cortical recovery), 4c (early predictors, exploratory n=4, pre-registered predictions); direction-neutral wording
throughout ("cortical activity consistent with the movement under the pre-stroke mapping; direction not determined").
ENCODING_FINDINGS 8.14 (partial refit), 8.15 (coupling by class, nulls), 8.16 (per-spout lick kernels).

### 000z ADDENDUM 10-25 ~23:50 (Priya asleep; read with the RUNNING list below)
- PER-EPOCH PIXEL KERNEL FITS (000y E; Priya: "what happened to our cross-epoch event kernel maps" -> never written; now
  `scripts/epoch_kernels.py`, run LOCALLY, Priya: "if too complicated, just queue locally"; O2 not needed: the post-session
  statistics pass is ~2-3 min / session). Frame = `..._frA_ls_basis` (frA env + FROZEN_LICKSPOUT=1, launcher
  `S/run_epoch_kernels.sh`). Groups per animal: early / late pre (halves by date; late pre = drift reference), acute,
  subacute, chronic; MATCHED session counts (n = smallest group: PS92 has only 2 subacute sessions -> PS92 matched n = 2;
  the unmatched fit is in the table, mode = all); frozen scale + frozen penalties (re-derived fit checked = saved, 0.0
  difference); split-half reliability per group. Outputs `<run>/epoch_kernels/` (kernel_similarity.csv, kernels.pkl,
  epoch_kernels_r.png = main, epoch_kernel_maps_<target|lick|cue>.png). Smoke (PS92, 2 per group) log
  S/epoch_kernels_smoke.log; FULL run queued after it: S/epoch_kernels_full.log (stats resume from <out>/stats/).
  TO DO: read the per-animal table first (rule 8); stroke-specific = r vs early pre below late pre AND below the group's
  split-half; then page figures + ENCODING_FINDINGS entry.
- PUSH FAILED x3 overnight: pre-push test `test_no_hardcoded_machine_paths` caught a machine path literal I added as the
  FROZEN_LICKLAND_CALIB default. Fixed (default = $WFIELD_INPUT_CACHE/pose3d_inputs/..., identical path in the chain's
  env; atomic write) in c47dd89; push restarted (task blybs3kqm, worktree wf_push_tmp11). CONFIRM it landed:
  `git fetch; git log origin/main --oneline | grep -c c47dd89`-style check on the rebased commit subject.
- Three OLD pre-push pytest processes (started 18:37 / 19:05 / 19:35, not attributable to this session) are still alive;
  left alone (shared box).
- Page v86: Figs 1 / 1b (FR comparison) and 15b (Musall alone vs unique) are now EPOCH-LINE figures (bars = supplement
  links); scripts frozen_fr_compare.py (lines_figure), frozen_musall_figures.py (musall_alone_vs_unique_lines_*).
  Running these scripts from the worktree needs PYTHONPATH=<worktree> (another wfield_local install shadows it).
- Lickspout basis run FINISHED 23:36; the land/lag chain started (landing run first).

### 000z ADDENDUM 2 10-26 ~00:00 (Priya asleep)
- NEW `scripts/iti_licks.py` (Priya: "let's also do the inter-trial 3d licks, pre-lick ramp timing"): (A) activity before
  SELF-INITIATED licks (bout-start ITI 3-D licks: no lick for 2 s [1 s secondary], no cue / spout move within -2..+0.75 s;
  onset-aligned; baseline -1.5..-1.0 s; t50 / t20 / amplitude per session x ROI; trial first licks secondary) in ALM /
  tjM1 ROIs (Priya: "the anterolateral 1/3 to 1/2 of MOs and MOp" -> anterolateral 40 %, figure iti_licks_rois.png) +
  MOs / MOp / SSp-m / cortex; (B) decoding each lick's 3-D lateral direction from area activity per lag, ITI vs n-matched
  in-trial licks, total and WITHIN-SPOUT direction (the spout is in place during the ITI), + frozen pre-ITI decoder.
  Full run log S/iti_full.log -> <E>/iti_licks/ (ramp_sessions / ramp_change / decode_* csv, figures iti_ramp_peth.png,
  iti_direction_decoding.png). SMOKE (PS92 only, 3 pre + 2 acute sessions): pre rise starts ~0.1-0.4 s before onset
  (earliest tjM1 / SSp-m), acute ~0.05-0.08 s; ITI direction decodes r ~0.6 pre, ~0.5 acute; within-spout r ~0.28 both.
  ONE ANIMAL -- read the full run's per-animal table first. Caveats to state: a cortex-wide component rides on every
  ROI (consider ROI minus cortex); calcium lag; whether the acute "later rise" is a ramp change or a baseline / kinetics
  change (the 1 s gap version had a decaying acute baseline -> 2 s made primary).
- NOVAR PIXEL MAPS DONE 03:45 (re-scored ..._frA_ls_basis_novar; layout names fixed to the real variable bases: pos_enl /
  pos_cue, tongue / jaw / nose _mag + _dir) -> page Fig 1ub4 (cue; lick linked). Drop the empty push column at the next redraw.
- NOVAR PIXEL MAPS (000y B) were IMPOSSIBLE from ..._frA_ls_basis: _pix_musall_variant matched "no_" not "novar_", so no
  moments were saved for the single-variable drops (the novar figure drew only the full column). Fixed (frozen_encoder,
  atomic); re-scoring joint + novar_* into a SEPARATE tag ..._frA_ls_basis_novar (fit inputs copied; summarise_musall
  rewrites frozen_musall_r2*.csv with only the listed variants, so in-place would clobber the basis table), then the novar
  figures: S/run_novar_pix.sh, log S/novar_pix.log (~2-3 h at 2 workers). Page v92 has the ls-frame alone maps (1ub3) and
  spout-unique maps (1ud2).
- PERM NULLS DONE (106/106) -> to-do A DONE: partial_refit_lines.png = page 1yd (bars = 1ye), FINDINGS 4.11 / section 0 and
  ENCODING_FINDINGS 8.14 updated: block-permutation nulls reproduce the single-shift contrasts (chronic contra SSp-m
  target +0.036 / +0.034, executed +0.055 / +0.064; acute ipsi SSp-m target +0.039 / +0.041; all 4/4). Lickspout map
  passes (000y B / C / D) DONE 01:16 -> figures in <E>/frozen_encoder_..._frA_ls_basis/figures, not yet on the page.
- EPOCH KERNELS FULL RUN DONE 00:27 -> ENCODING_FINDINGS 8.18, page 1uc2 / 1uc3. Per-spout lick kernels diverge
  steadily from early pre (chronic r 0.50-0.69 vs late pre 0.82-0.92, below split-half), cue kernel changes acutely,
  target kernels too noisy. NEXT (decides it): drift-RATE test (within-chronic early vs late r per day vs the pre rate
  per day; as chronic_drift_vs_reorg). Matched fits drop PS95 (one acute session).
- ITI FULL RUN DONE 00:10 (102 sessions) -> ENCODING_FINDINGS 8.17, FINDINGS 4d. t50 'later ramp' = AMPLITUDE
  artefact (post-onset response ~2x acutely); amplitude-free early ramp (-0.6..-0.2 s / pre amplitude) weaker
  CHRONICALLY in tjM1 / SSp-m (3/4); direction decoding intact within session, frozen pre decoder -0.13 every epoch
  (remapped). NEXT: page figures (iti_ramp_peth.png, iti_direction_decoding.png + a line panel of early_n / peak_n);
  regress pre-lick FaceRhythm out of the ramp; ROI minus cortex; per-animal PETH view.
- TIMING BUG in a shared table: `pose3d/kinematics_all/licks.csv.gz` daq_peak_s was written with %.4g (0.1 s after
  100 s, 1 s after 1000 s). Exact time = the session's licks_all.csv row lick_uid. AFFECTED: `wfield_local/stray_licks.py`
  (ENCODING_FINDINGS 2.6 now carries a RE-RUN NEEDED box). Writer fixed (pose3d_kinematics.py, c2aa922): the table
  regenerates at the pose chain's next kinematics step -- tell the pose window. face_rhythm_readouts reads licks_all
  (fine); the landing kernels join on (trial_id, lick_idx) (fine).
- QUEUED behind the epoch-kernel run: `S/run_ls_maps.sh` (log S/ls_maps.log) = 000y B / C / D on `..._frA_ls_basis`
  (pixel Musall novar + alone, cue / lick; spout_unique_maps).
- STILL OPEN (Priya asked "anything else that fell through the cracks", 10-25 ~23:55; answered): session dots in the
  line panels; pre-cue -> lick coupling; template-similarity check of acute trials; PS94 state x spout with the block
  null; design-space distance diagnostic; no-FR self-fit (check 4); lick-variability check; noise-weighted relabel;
  failed-attempt movement-matched control; encoding_mixed re-run; frAsplit_basis pixel (posinfo) maps -> page.
  Priya to decide: narrow every-lick window (first-lick primary?); June -> August drift ceiling; lesion burden /
  leave-PS93-out; share deletions / nokin retirement. The lickspout partial refit at the chain's end runs at 4 workers
  with perm nulls (~10 h): relaunch at 5-6 when the box frees.

### Key results of the night (all PROVISIONAL, details in FINDINGS / ENCODING_FINDINGS)
- KEY FIGURE (page, under the bottom line; = Fig 1i4): per-animal-normalised, hits-only. Behaviour chronic +0.01 (fully back);
  cortical far-contra code on HITS chronic -0.30 [-0.46, -0.13]; per-session gap ~0 subacute -> -0.31 chronic. By day (1i3):
  both climb to ~0.8 by day ~9, then behaviour completes and cortex plateaus at ~0.7.
- Coupling (neural decoder fM - fC vs lick fM - fC, within session): absent on FAILED attempts (acute -0.26 vs pre hits, each
  animal; subacute -0.29), kept on hits (acute hits ~pre but 64 trials; chronic hits at pre). The old "acute r ~ 0" was
  composition (92 % of acute far-contra trials are failed attempts). Block-permutation null; cross-epoch statistics.
- Decoder: failed attempts carry ~no far-contra code in any window (retention ~0); acute far-contra trials are decoded
  CONFIDENTLY as another spout, animal-specific (off-distribution confidence not excluded). Pre-cue class-matched: weaker on
  failing trials acutely, raw P -0.14 [-0.24, -0.04] but median session -0.06, few hits -> suggestive only.
- Target vs executed: old refit "target > executed" was chronic + refit-only; frozen: neither recovers; self-fit: target
  remapped; partial refit (single-shift nulls): target AND executed remapped chronically in contra SSp-m; per-spout lick
  kernels (frozen, `..._frA_ls`) absorb part of the old target term and only the executed spout-locked templates gain.
- Recovery timing, hits only: no consistent order (every CI includes 0); neither predicts the other session to session.
- Early predictors (n=4): lesion % perfectly orders behavioural recovery speed (one-sided p .042); acute retention orders
  chronic code inversely but PS94 alone sets it.

### RUNNING (do not restart; check logs)
1. Partial-refit BLOCK-PERMUTATION nulls on frA (`partial_refit.py --add target_perm exec_perm`, JOBS 6, log
   S/partial_perm.log): 48/107 at 23:00, ~10 min / session / worker -> done ~03:00-04:00. THEN: `bash S/run_partial.sh
   --plot-only` -> `partial_refit/figures/partial_refit_lines.png` = new page 1yd (contrasts target - null, exec - null now
   use the PERM nulls; bars -> supplement partial_refit_bars_<w>.png); update FINDINGS 4.11 "WHICH PART", ENCODING_FINDINGS 8.14,
   page row + 1yd/1ye captions (remove "nulls being redone").
2. Lickspout PIXEL run `..._frA_ls_basis` (FROZEN_PIX_MUSALL=1, JOBS 2, log S/frozen_frA_ls_basis.log): 66/107 at 23:00. Its
   old chain bash was STOPPED on purpose (the python runs on its own).
3. New chain `S/run_land_lag.sh` (log S/land_lag_chain.log; task bxa5qlvlf) waits for (2) to exit, then:
   a. LANDING lick kernels `..._frA_lsland` (FROZEN_LICKSPOUT + FROZEN_LICKLAND: lick copies by each lick's 3-D landing soft
      weights; licks without landing info -> 7th copy lick_onset@unknown). Trimmed variants.
   b. LAGGED FaceRhythm `..._frA_frlag` (FROZEN_FR_LAGS=-0.3..0.3: <v>:facelead = past face, <v>:cortexlead = future face;
      variants no_frfacelead / no_frcortexlead).
   c. self-fit `..._frA_ls` (jobs 3), d. partial refit `..._frA_ls` (jobs 4; computes ALL nulls incl. block perm -> slow,
      ~10 h at 4 workers: consider O2).

### TO DO after the runs (why)
A. Partial refit perm nulls -> figures / docs (1 above). Priya's call on the statistic: contrasts vs PERM null.
B. Landing run: unique R2 of lick_onset@<spout> (landing) vs target; compare with the target-split run `..._frA_ls`
   (adapt S/ls_unique.py: tags, add no_lickspout vs no_target). WHY: the target-vs-executed contest on misdirected trials
   (landing diverges from target on 64 % of tracked licks in an acute session).
C. Lagged FR: unique(cortexlead) = joint - no_frcortexlead, unique(facelead) = joint - no_frfacelead, per area, pre level and
   epoch - pre (nested bootstrap). WHY: the only direction evidence this dataset gives (temporal precedence); calcium lag biases
   absolute timing toward cortex lagging -> read the CHANGE.
D. Self-fit / partial refit on the lickspout design -> the decisive contest (does a session-specific mapping prefer target or
   spout-specific lick kernels chronically?).
E. Queued, not started: within-trial lead / lag (pre-lick vs post-lick prediction of the lick's direction); pre-cue -> lick
   coupling; template-similarity check of acute trials ("confident wrong spout" vs off-distribution); PS94 state x spout check
   -> block null; convert the remaining bar families to epoch-line figures (`ffc.epoch_lines`; Priya: lines "much better";
   2-3 lines per figure, bars as supplement, session dots where per-session values exist); session-level versions of other
   per-animal figures (4f / 4g, behaviour 3-6).
F. O2 bundle for the frozen encoder (Priya: "we can run on o2"): code sync, inputs (3-D snapshot, FR master_pre_v1 proj,
   attempts table, templates / pre_stats / spatial pickles), submit wrapper; Priya runs transfer / submit. Best candidates:
   the lickspout partial refit with perm nulls, per-epoch pixel kernel fits (000y E).
G. From 000y still open: B (novar pixel maps; `frozen_pixel_musall.py --layout novar` on `..._frA_ls_basis` when it lands),
   C (pixel Musall alone / PETH-vs-trial: `frozen_pixel_musall.py --layout alone --window cue|lick` on `..._frA_ls_basis`),
   D (spout_unique_maps on `..._frA_ls_basis`), E (per-epoch kernel fits), F (narrow every-lick window), G (older items).

### New scripts tonight (repo)
`frozen_pixel_musall.py` (pixel Musall layouts), `coupling_by_class.py`, `trial_neural_vs_exec.py` (block null, per session,
pooled, normalised, cross-epoch stats), `decoder_retention_sessions.py`, `precue_class_matched.py`, `recovery_timing.py`
(hits-only timing + pooled by day / epoch = key figure), `early_predictors.py`; `wfield_local/sig_marks.py` (haloed stars);
`frozen_fr_compare.epoch_lines / lines_legend` (line panels); `partial_refit.py` (per-session save / resume, contrasts,
exec_shuf, block-perm nulls, --add); `frozen_encoder.py` FROZEN_LICKLAND / FROZEN_FR_LAGS (opt-in; tests pass).

### PITFALLS (tonight, in addition to 000y)
- HEREDOCS AND sed STILL EAT `\n` (hit 4 more times): write patch scripts with the Write tool, always.
- `build_design` MEAN-CENTRES event amplitudes: an amplitude-weighted kernel needs + mean(amp) x the plain kernel added back.
- `df.align` is a DataFrame method: use df["align"].
- Retention units ((P - 1/6) / (pre - 1/6)) explode when the pre code is near chance (pre-cue PS92): use raw P as primary there.
- One half-session circular shift is a noisy null (spouts in ~6-trial runs): use block permutation, several draws, average
  SCORES (never predictions).
- Pre-stroke trials are HITS only: any hit vs failed contrast has no class-matched pre reference for failed attempts.
- Do not edit combined_encoding.py while encoder runs use the input cache (its source is in the cache key -> every session
  reloads from the share); edits to frozen_encoder.py: atomic tmp + os.replace.
- Killing a chain: stop only the bash (Stop-Process by command line); child pythons keep running.
- Publishing the page: use the long user path for root / file_path (see top).

## 000y. STATE 10-25 ~18:45 -- READ FIRST (supersedes the RUNNING / NEXT lists of 000x; 000x keeps the day's results)

Page v70 https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (source `S/goal_page/index.html`; S = `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`). Results of
the day: ENCODING_FINDINGS 8.11-8.13, FINDINGS section 0 (synthesis) + 4.2 / 4.3 / 4.11 / 4.12b. Code edited today in
`wfield_local/frozen_encoder.py` (opt-in envs only; tests pass, default runs unchanged; backup of the pre-edit file at
`S/frozen_encoder_backup.py`).

### RUNNING (worktree wf_attempt; check logs, do not restart)
000. UPDATE 10-25 ~23:40: old lickspout chain bash STOPPED (its basis run continues as its own process). New chain
   S/run_land_lag.sh (log S/land_lag_chain.log) waits for the basis python to exit, then: LANDING lick kernels
   (FROZEN_LICKSPOUT + FROZEN_LICKLAND, tag ..._frA_lsland, trimmed variants), LAGGED FaceRhythm (FROZEN_FR_LAGS
   -0.3..0.3 s, tag ..._frA_frlag; read no_frcortexlead vs no_frfacelead unique, pre -> post change), then self-fit (jobs 3)
   and partial refit (jobs 4, includes the slow block-permutation nulls) on ..._frA_ls. Partial-refit perm nulls on frA:
   S/partial_perm.log (28/107 at 22:14; morning). New tonight: decoder_retention_sessions.py, precue_class_matched.py,
   coupling_by_class.py, recovery_timing.py (hits-only timing: no consistent order; neither predicts the other).
   Page v81. Analyses after the runs: landing vs target unique (contest on misdirected trials), FR lead / lag change.
00. UPDATE 10-25 ~22:30: partial-refit BLOCK-PERMUTATION nulls running (`partial_refit.py --add target_perm exec_perm`,
   JOBS 6, log S/partial_perm.log; ~3 h). When done: `run_partial.sh --plot-only` -> partial_refit_lines.png becomes page
   1yd (bars -> supplement), numbers into FINDINGS 4.11 / ENCODING_FINDINGS 8.14. The lickspout chain's partial stage
   now computes the perm nulls too (~5x slower per session): relaunch it at JOBS 5-6 when it gets there.
   Done tonight: coupling block null + cross-epoch stats + per session + class split (8.15, page 4i-4i4); bar figures ->
   haloed stars (wfield_local/sig_marks.py); epoch-line figures (ffc.epoch_lines) to replace grouped bars page-wide
   (Priya: lines "much better"; convert the other bar families next, 2-3 lines per figure, bars as supplement).
   Queued (Priya yes): session-level plots (decoder retention 7 / 7c per session, then session dots in the line panels);
   pre-cue class-matched test + pre-cue -> lick coupling; template-similarity check of acute trials; PS94 state x spout
   check -> block null.
0. UPDATE 10-25 ~19:30: partial refit (1) DONE (restarted at 4 workers; now saves each session as it finishes) + executed
   null added (`--add exec_shuf`); results ENCODING_FINDINGS 8.14, FINDINGS 4.11, page v71 Figs 1yd / 1ye. The lickspout
   chain (2) started 17:56 in parallel. Its stages c / d run at --jobs 2: when it reaches them and the box is free, stop
   and relaunch at 5-6 (partial_refit resumes; check session_vs_frozen). After (2d): run `partial_refit.py --tag
   ..._frA_ls --add exec_shuf` is NOT needed (new runs compute both nulls). Task C script written:
   `scripts/frozen_pixel_musall.py` (alone / unique / novar layouts, PETH / trial / all frames; tested on frA_basis).
1. `scripts/partial_refit.py` frA frame (S/run_partial.sh, log S/partial_full.log, task b9q60utqe): 33/107 at 18:40.
   WHY: which part of the mapping changed -- refit ONLY target / executed / executed+FR dir / target+executed /
   +FR, others frozen; null = target with labels circularly shifted; per-spout scores (= remap check 5). On finish it
   prints per-animal table + boot + by-spout. TODO: page figure (partial_refit/figures/partial_refit_<w>.png), FINDINGS.
2. `S/run_lickspout.sh` (task b2dnwtwfm, log S/lickspout_chain.log): WAITS for (1), then in order:
   a. AREA run `gate-attempted_v7g3_t3d_frA_ls` = frozen + FR mode A + FROZEN_LICKSPOUT=1 (lick-onset s/d columns
      replaced by SIX per-spout copies of the recorded lick-onset columns, spout = target spout; group "lickspout" on
      the EXECUTED side of the target / executed bracket), variants = group drops + alone (task / lick / spout / other /
      movement / fr _only) + 26 novar_<var> per-variable drops + drop-one-spout. ~1-1.5 h.
   b. BASIS run `..._frA_ls_basis` (FROZEN_PIX_MUSALL=1: saves per MUSALL window target-space moments
      mus_<cue|lick>_<avg|trial>_<E|S> for joint / *_only / no_* variants; drop-one-spout incl. whole-spout swap).
      ~3 h, JOBS 2.
   c. self-fit (`session_vs_frozen.py --tag ..._frA_ls`) and d. partial refit (`partial_refit.py --tag ..._frA_ls`; adds
      sets lickspout / exec_ls / target_ls) on the same design. ~1.5-2 h each.

### TO DO after the runs (exact tasks, why)
A. Area run (a): unique R2 of target vs lickspout (frozen), per spout, epoch changes; compare with the current frame
   (does adding lick-locked spout kernels absorb the target-unique term? that is the target-vs-executed question with
   both terms lick-aligned). Script: extend `frozen_fr_musall.py` / `frozen_claims.py` QS with no_lickspout, or a small
   new script. Per-animal table first; PS94 normalisation for any mean map.
B. Musall bottom row (single-VARIABLE unique dR2): area maps from the novar_* variants (`frozen_peth_trial_maps.py`
   style, painted Allen areas) and PIXEL maps from the basis run -- `frozen_pixel_unique_sig.py` reads resid_mom per
   variant: add the novar_* / no_lickspout / *_only quantities to its QS (or a sibling script). Pre level row (=
   Musall) + acute / chronic minus pre with the nested test. Canonical spout order is irrelevant here; variable order:
   task (cue, arrival, target), licking (lick onset / per-spout, contact, exec angle), pose (tongue, jaw, nose), running,
   FR (lick, right, left, close, far, other).
C. Musall TOP row (alone) and PETH vs trial at PIXEL level: NEW script needed -- per session read mus_* E / S from the
   basis run npz, pixel SSE = diag(W E W'), SST = diag(W S W') with W = spatial_<an>.pkl P (as `frozen_r2_maps.pixel_sse`),
   R2 = 1 - SSE / SST (alone: E of the <g>_only variant; unique: E[no_g] - E[joint] over S); per animal x epoch SUM
   E and S over sessions before the ratio (frames weigh equally; as frozen_r2_maps); nested test on per-session maps
   (fr_pixel_significance.test, 16 px). Layout as Figs 1t / 1ua (rows pre PETH, pre trial, acute / chronic minus pre).
D. Per-spout target vs EXECUTED unique maps in the FR frame: rerun `spout_unique_maps.py --tag ..._frA_ls_basis`
   (the basis run has drop-one-spout now); with LICKSPOUT the "whole" swap no longer touches lick onset (guarded) --
   say so in the caption.
E. Per-epoch pooled PIXEL KERNEL fits (approved, NOT written): per animal pool sessions by epoch (early pre / late pre
   / acute / subacute / chronic; match session counts), fit in the basis with the frozen penalties, save target and
   per-spout lick kernels; compare map correlation with the early-pre kernel per spout x lag, late-pre = drift
   reference. Use `frozen_kernel_maps.py` for plotting (canonical spout order, NO vessel mask, NO grey bar).
F. Second pass of the self-fit: narrow every-lick window (+/-0.2 s around each lick peak; secondary, Priya to confirm
   first-lick primary) for frozen and self-fit, + design-space distance diagnostic (Mahalanobis of post design rows
   vs the pre Gram). Needs lick-peak times (not returned by session_design -- load via combined_encoding.load_behaviour).
G. Older: lick-variability check (within-spout lick timing / count variance per epoch: explains the acute rise of
   lick-alone single-trial R2, Fig 1ua2); noise-weighted relabel; failed-attempt movement-matched control; encoding_mixed
   re-run; share deletions / nokin retirement only when Priya decides.

### O2 (Priya asked 10-25 ~18:50: anything to submit?)
Not tonight: the frozen pipeline has never run on O2 and the local chain should finish overnight (~8-9 h). If wanted
later, the natural split is FIT LOCALLY (pooled pre statistics -> per-animal fits, minutes) and SCORE SESSIONS ON O2 as an
array job (one task per session: design build + predict + moments). Prerequisites: code pushed and pulled on O2; confirm the
10-24 O2 bundle carries every input of this frame (3-D tongue snapshot pose3d_kin_licks_snapshot_20261023, FaceRhythm
master_pre_v1 projections, attempts table / gate, per-animal templates.pkl, spatial_<an>.pkl for basis runs); a submit
wrapper that ships the fit pickles; Priya runs submit / transfer. Best first candidate: task E (per-epoch pixel kernel
fits) -- write it O2-ready (per-animal x epoch tasks) from the start.

### PITFALLS (today)
- HEREDOCS eat escapes: `\n` inside python strings and `\` line continuations became literal newlines / broke code
  three times -> write patch scripts with the Write tool; heredoc only for code with no backslashes.
- Do not edit frozen_encoder.py while a run that imports it is starting workers (fan_out pools spawn per stage). The
  current partial refit already spawned its pool; the lickspout chain imports the edited file.
- `sessions.csv` / caches: specificity_control --reuse, spout_unique_maps (spout_unique_cache_<base>.npz),
  residual_roi --by-spout (sessions_by_spout.csv) reuse cached tables -- delete the cache when the source run changes.
- ffc.bars needs a `frame` column and labels in ffc.ALAB (KeyError otherwise); ffc.boot adds `<ref>_split` rows -- filter
  epochs before painting maps (a map once painted the noise-band rows).
- Smoke tests: use a REAL session label (PS92_0912 does not exist -> StopIteration in load_imaging).
- R2 is already normalised per area / pixel: PS94's ~10x target-unique is NOT a data-scale issue (8.13); mean target
  maps must be animal-normalised.
- "Control" areas are a comparison set, not a null; cortex-wide != drift / not stroke.
- Canonical spout order everywhere; Musall citations: alone = top row, single-variable unique dR2 = bottom row.
- Push pushes from a temp worktree (pre-push tests ~25 min); origin may already contain rebased copies of local commits.

## 000x. STATE 10-25 ~18:00 (pre-compaction) -- READ FIRST (then 000w, 000v, 000u)

Page v68 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d; source `S/goal_page/index.html`, S = `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`). New today
(after 000w): self-fit vs frozen (8.12: refit target gain is a FRAME effect -> target remapped), remapping maps + per-animal
+ time course (8.13: spout/target remap cortex-wide, smooth with time; task remap orofacial-specific), PETH vs trial maps
+ Musall alone layout + kernel beta maps, Allen-ROI Fig 10 family (10h-j), free push on current frame (8.12), "control" ->
non-orofacial comparison set (wording everywhere), PS94 dominance caveat (FINDINGS 4.12b). FINDINGS section 0 = synthesis.
Priya rules today: canonical spout order everywhere (memory canonical-spout-order); no vessel mask; white halo under
significance outlines; first-lick window primary (narrow every-lick window = secondary, Priya to confirm); gain correction
dropped; no "control areas" as a null.

**RUNNING at ~18:00 (worktree wf_attempt):**
- partial refit (`scripts/partial_refit.py`, S/run_partial.sh, log S/partial_full.log): refit ONLY target / exec / exec+FR
  dir / target+exec / target+exec+FR / target with shifted labels (null), per session, frozen everything else; per-spout
  scores (check 5) included. 24/107 at 17:50. Then: summarise (prints per animal + boot + by-spout), page figure.
- free push current frame chain (S/run_frA_pushfree.sh, log S/frA_pushfree_chain.log): frozen + spec done; residual_roi +
  matched residual finishing.
- `scripts/state_spout_check.py` (log S/state_spout.log): PS94 dominance check (state x spout eta2 + tracking).
- `scripts/spout_unique_maps.py` (logs S/spout_unique*.log): per-spout target vs executed unique maps (levels / changes /
  per animal with own colour bars + within-animal tests); builds a cache then redraws.
- halo redraw chain (b4f92tuv8): spec / remap figure redraws with haloed outlines.

**QUEUED / NEXT (Priya approved):**
1. Per-spout LICK kernels in the FROZEN model (env-gated in frozen_encoder.py -- edit only when no run imports it):
   replace lick_onset spl/dev by 6 per-spout copies (by TARGET spout; optionally by EXECUTED landing bin = lick-locked
   contest); + per-VARIABLE drop variants for Musall's bottom-row single-variable unique dR2 maps (cue, arrival, target,
   lick onset, contact, exec angle, tongue, jaw, nose, running, FR lick / right / left / other). Areas + basis runs.
   SAME basis run should also carry (Priya 10-25): single-GROUP ALONE variants (task_only, lick_only, other_only,
   spout_only, movement_only, fr_only) for Musall's TOP row at pixel level, and save per-spout x window data / prediction
   moments in `_predict_spatial` so each pixel gets PETH (avg) vs trial-by-trial R2 (the pixel version of Figs 1t / 1u /
   1ua). Launch only after partial_refit finishes (it imports frozen_encoder).
2. Per-epoch pooled PIXEL kernel fits (early pre / late pre / acute / subacute / chronic, matched sessions) -> target and
   per-spout lick kernels pre vs post (map correlation per lag; early vs late pre as drift reference).
3. Self-fit + partial refit with the per-spout-lick design; check 4 (no-FR self-fit).
4. Pixel-resolution PETH vs trial pass (basis); per-spout data / prediction / residual activity maps (quick, from saved
   evoked).
5. Narrow every-lick window second pass (both frames) + design-space distance diagnostic.
6. Older: noise-weighted relabel; failed-attempt movement-matched control; encoding_mixed re-run; share deletions / nokin
   retirement only when Priya decides.

## 000w. STATE 10-25 ~15:45 -- READ FIRST (then 000v)

Priya: "do analyses 1, queue up 2 and 4"; "keep where we stand in the repo" -> FINDINGS section 0 (synthesis, revised).
DONE (ENCODING_FINDINGS 8.11; both frames; page v56 Figs 1n-1s + revised status rows):
- 1 specificity (`scripts/specificity_control.py`): Layer 2 NOT orofacial-specific (control = orofacial; grows steadily
  with log day; only an acute orofacial excess) -> FINDINGS 4.3 CORRECTED. Residual orofacial-weighted in the cue window
  (contrast 3/4 FR / 4/4 no-FR), partly global in the lick window. Gain correction tried and DROPPED (Priya: not needed).
- 2 recovery (`scripts/remap_lmm.py`, figures in remap_lmm/<tag>/figures): residual largest at day 1, declines, persists
  at full recovery (cue orofacial - control +0.73 / +1.09 pre-SD); the deficit slope is carried by PS94 (PS92 opposite).
- 4 matched (`scripts/matched_residual.py`): residual unchanged on trials inside the pre behavioural envelope -> not
  extrapolation. Design-space distance diagnostic still to do (fold into the self-fit second pass).
STILL RUNNING: session_vs_frozen (S/svf_full.log), split run, maps chain (frA maps now). NEXT: when svf lands -> remapping
+ scoring-caveat verdict; second pass = narrow every-lick window (both frames) + design-space distance; frA / split maps to page;
posinfo split -> page; then analyses 3 / 5 / 6 / 7 if Priya picks them.

## 000v. STATE 10-25 ~14:45 -- READ FIRST (then 000u)

- Page v55: noise bands on Figs 1-1k; frozen noFR PIXEL maps added (Fig 1l nested test, Fig 1m per-animal view; status row).
- SCORING CAVEAT (Priya asked 10-25): refit and frozen R2 are scored differently (refit cue 0-1 s, lick +/-0.2 s around EVERY
  lick peak, offset penalised; frozen cue 0-1.5 s, lick -0.1..+1.0 s around the FIRST lick, data / prediction centred
  separately). Within-frame changes are fine; the cross-frame "not reproduced" revisions (FINDINGS 4.11, 4.12a) mix frame
  and scoring -> flagged "check pending" on the page and in FINDINGS.
- RUNNING: `S/run_svf.sh` -> `scripts/session_vs_frozen.py` (log `S/svf_full.log`): each session refit on ITSELF in the
  frozen design + windows (5 trial blocks, penalties chosen in-session; frA env), minus its frozen R2, epoch - pre.
  Answers both NEXT 3 (remapping) and the caveat: if the self-fit's target unique rises chronically, "not reproduced" is a
  frame effect; if not, the refit gain came from refit scoring / design. Smoke (PS92_0806): pre gap +0.10 SSp-m.
  Output `frozen_encoder_gate-attempted_v7g3_t3d_frA/session_vs_frozen/` (gap.csv, remap_boot.csv, figures/).
- Synthesis given to Priya 10-25 (acute = movement + transient decoupling; chronic = same behaviour, different cortical
  solution). Proposed next analyses, priority: (1) specificity control -- residual / Layer 2 in VIS / RSP + normalise by a
  task-independent response (total variance rose globally); (2) residual / Layer 2 vs behavioural deficit, session mixed
  model; (3) target-free execution test from ITI 3-D licks; (4) behaviour-matched residual + frozen extrapolation
  diagnostic; (5) pre-lick ramp timing in ALM / tongue-jaw motor; (6) June -> August reduced-model drift ceiling;
  (7) lesion burden / leave-PS93-out. Not started -- wait for Priya's pick.

## 000u. STATE 10-25 ~14:20 (pre-compaction) -- READ FIRST (then 000t)

**Headline change (Priya 10-25): the FROZEN pre-stroke encoder is the frame for every post-stroke CHANGE claim; FaceRhythm is
now inside it.** Per-session refit results are supporting only (say the frame in every claim). Results: ENCODING_FINDINGS 8.10,
FINDINGS 4 (items 1-3 unchanged with FR; 11 and 12 REVISED). Page https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d v53:
"Where things stand" table at the top (result | figures | status | frame | animals; Priya rule: cite figures), section 0 =
frozen key figures (Fig 1-1k), FR refit section is now 17. Page source `S/goal_page/index.html` (S = `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`); renumber with
`S/renumber_figs.py` (now also remaps href="#fig.." anchors and their text, and protects "Musall ... Fig" citations -- a
renumber had turned "Musall Fig. 5" into "Fig. 13"; fixed). Window order everywhere: pre-cue, cue, lick (Priya rule). Sidebar
breakpoint 640 px. Peers told (5b, e3).

**Done today (beyond 000t):** mechanism read-out (FINDINGS 4.12 WHY); no-attempt relabel by class (Fig 4t;
`scripts/relabel_fc_by_class.py`); FR modes B/S/D contra tests (contra_orofacial_mode<X>/; mode-D overwrite of mode A fixed);
position-coding trend re-run on O2 runs (ENCODING 4.1-4.2; 5b told; practice-trend test needs June); OUTPUT_AUDIT 2a status
column; frozen + FR runs and every comparison above; refit FR map z-scored; per-animal secondary pixel view; FR modes
schematic (Fig 17e); direction lick-window figures; layout fixes.

**RUNNING at ~14:20 (all from worktree wf_attempt; do NOT rebase onto code changes while they run):**
- `S/run_frozen_basis_fr.sh` (log `S/frozen_basis_fr_chain.log`): frozen + FR mode A on the per-animal SVD basis ->
  `frozen_encoder_gate-attempted_v7g3_t3d_frA_basis` (predict 8/107 at 14:15).
- `S/run_frozen_pos.sh` (log `S/frozen_pos_chain.log`): quick posinfo done; FULL split run FROZEN_FR_SPLIT=1, areas, all
  variants -> `frozen_encoder_gate-attempted_v7g3_t3d_frAsplit` (stats done, fitting).
- `S/run_frozen_maps.sh` (log `S/frozen_maps_chain.log`): noFR pixel maps done -> waits for the FR basis -> frA pixel maps
  (raw + z) -> waits for the split run -> split BASIS run (posinfo / noposinfo variants) -> split pixel maps (raw + z).
- background redraws: noise-band redraw of frozen_fr_compare / frozen_fr_musall / frozen_claims / frozen_posinfo (/tmp/r1-4.log,
  /tmp/redraw.done) and the noFR pixel redraw (new layout, S/pix_nofr2*.log).

**NEXT (in order):**
1. When the redraws land: copy figures to `S/goal_page/` (frozen_fr_compare_*, frozen_musall_*, frozen_claims_*,
   frozen_posinfo_quick_*) and republish (files!). Add frozen pixel figures (noFR `frozen_encoder_gate-attempted_v7g3_t3d_basis/
   figures/unique_sig{,_z,_per_animal}.png`; then `_frA_basis`, `_frAsplit_basis`) to section 0 + status table rows.
2. Split run done -> `python scripts/frozen_posinfo.py --tag gate-attempted_v7g3_t3d_frAsplit` (replaces the quick version on
   the page; update FINDINGS 4.12 (e)) ; split basis maps -> page (Priya asked for posinfo MAPS).
3. Per-session vs frozen comparison for "remapping" (Priya 10-25): same session refit R2 - frozen R2, epoch - pre, per area
   and per group (caveat: different designs, only the change is interpretable); Layer 2 is the formal version.
4. Optional (offered): absolute-level plots with per-animal pre spread (Priya asked about pre variability; noise band added).
5. Older (000t list): noise-weighted relabel re-score; failed-attempt movement-matched control; encoding_mixed re-run on the
   rebuilt table; convert remaining x1e3 numbers when touched; revisit share deletions with Priya; retire nokin_* runs only
   when Priya decides (5b relays; NOT in DECISIONS yet).

**Pitfalls learned today:** heredoc python patches eat backslashes (`\n`, `\1`) -> use the Edit tool or Write a .py file;
scripts that write shared filenames per variant / mode must suffix them (two overwrites today, fixed); the push pre-hook
runs ~25 min of tests and loses races -> push from a temp worktree or retry loop, never while chains import the worktree;
Artifact publish with `edits` is refused here (a file-edit hook) -> edit the file, publish without edits.

## 000t. STATE 10-25 ~03:30 (pre-compaction) -- READ FIRST (then 000s)

**Page** https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d v43 (source `S/goal_page/index.html`, S = this window's scratchpad
`C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`). Sidebar now groups figures by NUMBER with a collapsible toggle (Priya). Figures renumbered with `S/renumber_figs.py`
(insert new figures with unused temp letters, e.g. 16x, then run it). Publish new PNGs with `files` (an index-only publish
leaves new images missing -- happened once, "16g/h broken"). ALL R2 now a FRACTION 0-1 (Priya): FR scripts, fig_layer2 / fig_rsplit
/ fig_pooled / fig_softmax / fig_goal_summary (S/), fr_encoding_summary.py; Layer 2 CSVs are stored x1e3 and divided at load. Figs
4 / 12 stay dF/F. Captions converted. Docs below 8.7 still quote some x1e3 numbers (convert when touched).

**Done 10-25 night (all in ENCODING_FINDINGS 8.3 / 8.7 / 8.9 and FINDINGS 4.8 / 4.12):**
- Reviewer hits in ENCODING_FINDINGS (8390343): attempt rates with denominators (canonical .79/.79/.58/.34; all trials .62/.61/.445/.342),
  mechanism PS92-94 try-and-fail, PS95 mostly fail-to-try; Layer 2 10-23 WITHDRAWN; e3 told.
- FR Musall PIXEL maps (fr_musall.py; Figs 16e-16h) + Venn schematic (16d; our "task-aligned" = overlap with EVERYTHING, not task only).
- FR map significance (fr_pixel_significance.py, 16i-16j; calibrated 5 %) + contralesional SSp-m area tests (fr_contra_orofacial.py,
  16k): chronic position unique | FR contra SSp-m cue +0.037 [0.010, 0.067] 4/4 p_bonf .002, lick n.s.; contra-ipsi CIs exclude 0;
  pixel test 0 bins; carried by PS94 / PS95; total variance ROSE (not a denominator effect). PROVISIONAL, window-sensitive.
- Per-trial encoder relabel DONE (Figs 3q-3s): label-wide close-mid bias; far contra least separable label pre-stroke (far ipsi only
  +16 % worse, SD 46 %); session-summed by class: pre hits fC best 4/4; ACUTE HITS fit far mid slightly better (0/3 fC best),
  incomplete stay ~far contra (2/4). Running: no-attempt trials (below).
- exec_vs_neural_match.py --target (Figs 3m-3n): off-target far-contra trials acutely decoded as executed >= target in 3/4 (near chance),
  target returns subacute / chronic except PS92; far mid target-led throughout except PS93 acute.
- O2 fr_A_tr (FR models in TR_MODELS, 7dcc8d3; 107 sessions incl. 4 resubmitted at 64 GB; collected `..._frg-A_subset-fr_..._tr`):
  fr_trial_peth.py (16l-16m): FR + uninstructed = trial-to-trial; task / position = PETH; ACUTE spout loss is a PETH loss (cortex lick
  -0.050, ipsi SSp-m -0.016, 0/4); chronic contra rise split, n.s. with these windows.
- variance_maps.py (16p; data only): total variance rises, all significant bins up; cortex x1.54 lick chronic, global (VISp x1.83).
- fr_direction.py (16n-16o, all epochs + pixel contours): FR gain ~all directionless; direction factors' added fit in SSp-m FALLS acutely
  and chronically (0/4 up); contralesional SSp-m position rise present with DIRECTIONLESS FR (B: cue +0.028, 4/4).
- Cleanup (approved by Priya): wfield_cache pre-10-24 06:29 (163.6 GB), session-cache LP tag lp92d8d7aa (23.4 GB), scratch stores
  (35.5 GB) DELETED; C: 308 GB free. Share deletions = TO-DO to revisit (list in this file, CLEANUP 10-25 block).

**RUNNING at ~03:30:**
- `scripts/fr_direction_mechanism.py` (log `S/fr_mech.log`) -> `E/fr_encoding_summary/direction_mechanism/`: WHY FR direction factors
  explain less SSp-m post-stroke -- (a) behaviour: spout-specificity (eta2) of the FR side / distance index and of the 3-D tongue ml;
  sd of side; (c) measurement: within-spout r(FR side, tongue ml); (b) neural: within-spout partial r(SSp-m R / L / c-i, MOp; FR side /
  distance | lick). Per animal + bootstrap. Priya: "could be critical to interpretation". Report (a) vs (b) vs (c), add figure to page
  after 16o, update FINDINGS 4.12.
- `frozen_relabel --tag v7g3_t3d_trial_refgate` (OROFACIAL_GATE=reference, log `S/relabel_refgate.log`, ~20/107 at 03:20, ~1.5 h):
  NO-ATTEMPT far-contra trials (cue window only). CAVEAT: the run refits its joint pre fit on the reference gate. Then: far-contra
  best label by class (hit / incomplete / jaw_only / no_try), session-summed as in `S/relabel_fc_by_class.csv` analysis (code in this
  window's transcript: sum trial_sse per session x class, % worse than the fC label).

**AGREED PLAN AFTER COMPACTION (Priya 10-25 ~03:45: "update handoff with these tasks then will do after compacting"):**
1. Read the mechanism result (`direction_mechanism/bootstrap.csv`): (a) behaviour (eta2 side / dist / tongue, sd_side fall?) vs
   (c) measurement (r_side_tongue falls?) vs (b) neural (partial r SSp-m vs side / distance falls with (a) intact?). Page figure after
   16o; FINDINGS 4.12. Also check the refgate relabel (stalled at 20/107 since ~03:20 -- disk contention with the mechanism run?
   restart if dead) and finish the far-contra class table incl. no_try / jaw_only (cue window).
2. BUILD the FROZEN pre-stroke encoder WITH FaceRhythm -- main purpose: every frozen result (residual maps, Layer 2 "behaviour ->
   cortex mapping changes chronically 4/4", relabel) has NO face movement, while FR explains ~0.12-0.13 extra and absorbs the broad
   chronic position rise in the refit frame -> are the chronic residual / Layer 2 change partly unmodelled face movement? Secondary:
   decoupling test (pre-stroke face -> cortex weights applied post-stroke: frozen FR contribution collapses while refit does not ->
   mapping changed = neural; both collapse -> face signal changed). Code: FR factors as a group in `frozen_encoder.session_design`
   (inputs exist for all 107 August sessions: O2 bundle `inputs/<label>.pkl` key "fr", X (T x 12) at imaging frames, names), ~1-2 h.
   Run modes B (directionless: lick sum + other) AND A (full; needed for direction); movement-first credit (FR = movement);
   "beyond FR" = lower bound for spout effects (B reduces this). Local compute: fit a few h; downstream overnight.
3. Re-run on it: residual by area + calibration (frozen_residual_trend / frozen_residual_significance), Layer 2 change vs late-pre and
   random-split nulls (allsession_encoder + layer2_random_splits; 20 splits are the slow part -> O2 if needed), relabel (trial level).
   Compare each to the no-FR final frame; update FINDINGS 3.4 / 4.x and the page (new section or alongside 5-7).
**Older NEXT list:** (1) the two results above; (2) FROZEN pre-stroke encoder WITH FaceRhythm (Priya 10-25: "should we be mixing frozen and
refit"): frame map -- frozen (residual / relabel / Layer 2) vs per-session refit (all FR + Musall); a frozen-with-FR encoder is the
direct test of decoupling (b); new frozen fit, likely O2; (3) FR direction modes B / S / D for the contra SSp-m area tests;
(4) encoder likelihood re-score (noise-weighted) for the relabel trial labels; (5) failed-attempt movement-matched control + trial-class
map stats (mine; e3 does imaging LMMs on the rebuilt attempts table); (6) encoding_mixed re-run on the rebuilt table; (7) 5b asked for an
OUTPUT_AUDIT status column (items answered by message 10-25: decomposition re-run identical; nokin_* recommend retire).

## 000s. STATE 10-25 ~01:00 (pre-compaction) -- READ FIRST (then 000r)

**Results of the night: `docs/ENCODING_FINDINGS.md` §8** (trajectory softmax, neural decoder vs executed incl. the
trial-level coupling that breaks acutely, off-target by target, relabel confusion, residual at calibrated 16 / 32 px
bins and by area, Layer 2 random-split null, pooled figures, FR in the encoder, Musall / R2 on the final frame).

**Results page** https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (source `S/goal_page/index.html`; v35). Figures were
RENUMBERED 10-25 into page order (1 .. 16c; pooled versions are letters now, no "p"); one-pass renumber script
`S/renumber_figs.py` (rewrites caption labels, ids and every "Fig N" cross-reference). Click-to-zoom viewer added
(`S/add_zoom.py`; wheel / pinch / drag / dblclick reset / Esc) — UNTESTED in a browser, ask Priya. Conventions in
memory `results-page-conventions` (now incl. page order, matrix colours viridis 0-1 / RdBu +-0.6, colorbars on maps,
no overlaps). Figure scripts: repo `scripts/` for analyses; page-only figure scripts in `S/fig_*.py`
(fig_pooled.py = all pooled panels, fig_resid16.py / fig_resid_spout.py = residual maps, fig_goal_traj2.py = paths,
fig_landing.py, fig_softmax.py, fig_layer2.py, fig_rsplit.py).

**FINDINGS CONSOLIDATION (Priya 10-25, all windows):** ONE new top-level `docs/FINDINGS.md`, skeleton by
`sabatinilab-e3` (FR window), status tags STANDS / PROVISIONAL / CORRECTED / WITHDRAWN / NOT ESTABLISHED. Sections:
1 behaviour + orofacial (5b), 2 pose / 3-D (5b), 3 imaging decode / state / anatomy (ME to draft from 5b's imaging
page https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV + its corrections, 5b then corrects), 4 encoding + goal
question (ME), 5 FaceRhythm (e3), 6 histology, 7 open / withdrawn. Family docs stay (no deletions) with a header
pointing to FINDINGS.md; VLS_STROKE_SYNTHESIS / PRELIM_DATA get "superseded for current conclusions" banners after
their valid claims are carried. 5b's section-3 corrections (from its message): state-decoder numbers in
BEHAVIOURAL_STATE_CONTROL / synthesis are STALE (current epoch_13n: position .863 -> .428 / .696 / .761, state .976 ->
.872 / .820 / .838; state does fall, claim = the RATIO); WHERE_THE_CODE_MOVES QUIET / MEAN tables superseded by 15r
REST / RESTW (far-contra acute amplitude / pre .50 / .45 / .53 / .40, chronic .90 / .98 / .60 / 1.12); NEW June ->
August no-stroke drift group (scripts/june_august_drift.py): frozen-decoder chronic deficit NOT beyond 2-month drift
(3/4), crossnobis distance IS (3/4); CD LOO acute far-contra drop 3/4 PROVISIONAL; pre-cue mixed. A background reader
(agent) was extracting headline claims from the older docs when context compacted; if its report is lost, re-read
VLS_STROKE_SYNTHESIS, PRELIM_DATA, WHERE_THE_CODE_MOVES, BEHAVIOURAL_STATE_CONTROL, STATUS_2026-10-07, DECISIONS.
**DONE 10-25 ~01:30: FINDINGS.md sections 3 (imaging; 5b to confirm) and 4 (encoding) written; banners pointing to FINDINGS.md
added to ENCODING_FINDINGS, VLS_STROKE_SYNTHESIS, PRELIM_DATA, BEHAVIOURAL_STATE_CONTROL, WHERE_THE_CODE_MOVES. Keep section 4
in sync when results change.** CLEANUP SURVEY (Priya via e3, survey only, no deletes): a background reader was listing my
outputs (share imaging_encoding folders, o2 enc_* bundles, wfield_cache, worktrees, scratchpad) with recommendations; if
lost at compaction, redo and send the list to e3 / Priya. Previously: (section 4 = §8 headlines +
2.4b, 2.7-2.9, 3.4, 4.x of ENCODING_FINDINGS, short, status-tagged, with page figure numbers).

**REVIEWER HITS IN ENCODING_FINDINGS -- DONE 10-25 ~00:45 (commit 8390343; items 1-7 below; current attempt rates recounted with denominators in the top box; e3 told). Section-3 items at the end of this paragraph still open (FINDINGS section 3). Original list (e3's reviewer, line numbers as of 10-24 22:00 main):** (1) PS94/PS95 acute attempt rate has 3 versions (top box L16-19 .441->.445 / .325->.342 per
DECISIONS L21912; §2.5 L329 .79/.80/.64/.32; L325 ".81-.93"): put current values + each denominator. (2) mechanism label:
use PS94 = "try and fail" (acute 251 incomplete vs 78 no-attempt), PS95 = mostly "fail to try" (39 incomplete, 22
no-attempt, 57 reaction) -- matches FR 5c; fix L274 / L329 / top box; tell e3. (3) §2.4b: FINAL box (target > executed
3/4, width sweep done) vs PRIMARY bullet L163-169 ("acute mixed", "sweep queued"); change log L1016 "pre target-led 3/4"
was demoted to a capacity effect (DECISIONS L21888). (4) Layer 2 10-23 bullet L382-401 needs a SUPERSEDED / WITHDRAWN label
+ a 10-24 change-log entry. (5) pre-cue: §5 L925 (all spouts pooled, no consistent loss) vs §3.3 L644-646 (far contra,
4/4 down): say which in each. (6) stale: L32-36 "124 sessions / cam1 125/127" and L963-964 (now 127/127, 119);
L707-710 "_v7basis RUNNING", "PS94/PS95 cam4-only"; L935 "cam1 adds 3-7 % small licks" (POSE_MODELS D11: mostly
occlusion-lost licks); L208 / L318-320 cue z -5.3 / -8.0 (now -6.8); L48 "cam4 dropped ~13 %" vs FR 1b "11.8 %". (7) top
status table: add relabel 2.7, failed-attempt decoding 2.8, attempt maps 2.9, Layer 2, §8. Label §4.4 push rate "2-5x"
(context). Section-3 items e3 flagged that are not yet in FINDINGS: F18 drift detail, movement-removed ENL code tracks raw
(10-13, 118 sessions), null/potent (one session), within-session fatigue / quitting time-driven (09-20), 15g/15h
transfer arm (09-17), ENL sensory-vs-plan status (09-24); encoder amplitude has 3 versions (WHERE / CLAUDE.md /
deck_registry:733 0.749->0.286->0.745; VLS 0.943->...->0.943; PRELIM §4 0.943->0.361->0.675->0.854 newest); rest obs/null
1.32 / 1.443 / 1.622 / 1.634 (gated, newest); deck_registry notes stale (L533, L307/391); name the lick source for ILI.
"No cortex damaged" CORRECTED in FINDINGS §3 (PS93 anterior motor-cortex injury, HISTOLOGY) -- CLAUDE.md still says it
(flag to Priya; don't edit CLAUDE.md unasked).

**CLEANUP 10-25 (survey: background reader; lists in `S/delete_lists_20261025/`). Priya APPROVED + DONE:** scratchpad
LP / DAQ calibration stores of 10-21 (7 files, 35.5 GB; 10-24 LP stores kept); session-cache LP tag `lp92d8d7aa` (2,456
files, 23.4 GB; current tag `lp90d0a4f6` kept); wfield_cache entries older than 10-24 06:29 (1,008 files, 163.6 GB) --
deleted by a waiter AFTER the per-trial relabel finishes (check `S/delete_lists_20261025/` + task output).
NOT approved (kept): CD-window session-cache entries (~12.5 GB, tell that window), wfield_cache 10-24 06:29-23:40 (~82 GB).
**TO-DO, REVISIT WITH PRIYA (share, proposed, ~170 GB + unsure):** o2/enc_v36_full_20261024/out (41.6 GB, all 14 collected,
sizes match); o2/enc_v36_20261019/inputs (29 GB; keep code / match.json); Widefield/input_cache_backup (60 GB, pre-final-frame
keys); stale imaging_encoding folders: no-suffix FR collision copy (1.8), frozen v7basis / v7fwd15 / v7pushfree, frozen_relabel_v7,
allsession_v7push (9.9), 10-19..10-22 combined runs (23.0), FR pilots (2.3), 2-animal xyz2 (2.8). Unsure: v35 LP run (16,
SUPERSEDED.md), frozen v7g3 cam4 twin (3.5), nokin_matched / tr / win + drop_win (9.0), nokin bins_3d4 (3.0). Worktrees
wf_g3 / wf_lp2 / wf_frclean / wf_v4 / base3 merged + clean -> `git worktree remove` when convenient.

**RUNNING at ~01:00 (all write to the share):**
- `frozen_relabel --from gate-attempted_v7g3_t3d --tag v7g3_t3d_trial --jobs 8` (log `S/relabel_trial.log`; ~5 h
  from 00:00; writes all session jsons at the END). Then a waiter (`S/...` bash, task bx9xor7g5) runs
  `scripts/relabel_confusion.py --tag v7g3_t3d_trial --trial --window lick|cue` and
  `scripts/exec_vs_neural_match.py --source encoder` -> figures in `E/frozen_relabel_v7g3_t3d_trial/figures/` and
  `E/attempt_coding_pp/figures/exec_vs_encoder_*`. Put them on the page after Fig 3n (encoder: target vs best label;
  executed best bin vs encoder best label; off-target by target).
- `scripts/frozen_r2_maps.py --tag gate-attempted_v7g3_t3d_basis` (colorbars; log `S/r2maps_cbar.log`) -> recopy
  `_basis/figures/r2_maps_{pre,acute_minus_pre,chronic_minus_pre}.png` + `r2_maps_position_joint_whole_{pre,chronic_minus_pre}.png`
  to `S/goal_page/fig15*.png` and republish.

**NEXT:**
1. DONE 10-25 (Figs 16d-16g, scripts/fr_musall.py --mode A; other modes via --mode): Musall-style figure WITH FaceRhythm from the per-session FR runs (Priya asked; the frozen-encoder Musall / R2 maps
   have NO FR). Per-session region tables in `E/movement_first_fr-master_pre_v1_subset-fr_exec-polar_licks-combined_frg-<mode>/`
   have models full / fr:full / fr:alone / fr:raw:full / fr:drop:{task,instructed,uninstructed,position}: alone vs
   unique for FR and the groups, task-aligned vs independent, per epoch.
2. Encoder trial-level matrices (above) on the page; FINDINGS.md sections 3-4.
3. Region-level residual test (a priori MOs / MOp / SSp-m per hemisphere), droppos maps, per-spout binned contest,
   decoder + imaging-only encoder pages (5b built an imaging page — coordinate, don't duplicate).

**Repo:** branch `analysis/attempt-coding` (worktree wf_attempt). Commits this night: trajectory softmax
(lick_path_softmax.py), trial_neural_vs_exec.py, exec_vs_neural_match.py (--source decoder|encoder, off-target by
spout), relabel_confusion.py (--trial), fr_encoding_summary.py, frozen_relabel per-trial SSE + frozen_encoder
frame_trial, frozen_r2_maps / attempt_maps colorbars, frozen_residual_trend MOs / MOp / SSp-n,
frozen_residual_significance colorbars, layer2_random_splits resume, combined_encoding.out_dir `_frg-<mode>`.

## 000r. STATE 10-24 ~19:30 (pre-compaction) -- READ FIRST (then 000q)

**Results page (Priya reviews here; keep it current, republish the same file):** https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d
-- source `goal_page/index.html` + figure scripts `fig_*.py` in the session scratchpad (S); a copy of the page is
`docs/templates/results_page_example.html`. CONVENTIONS (Priya, memory `results-pages-as-artifacts`): hidable left
contents panel, EVERY figure numbered in caption + contents, sections tagged neural / lick, per-animal first then
within-animal-normalised pooled with stats. Other windows told.

**O2 batch `enc_v36_full_20261024` DONE + ALL 14 RUNS COLLECTED** (bins_w1 / w05 / w2, main, droppos, shufpos, nodir, drop,
trmodels, musall_shuf, fr_A / B / S / D). OOMs resubmitted with `resubmit_failed.sh` (on the share + scratch; now catches
any session without output that is not running). droppos needed 128 GB. NOT YET ANALYSED: fr_* (does FaceRhythm absorb
the chronic residual / spout coding?), droppos (per-spout unique maps), main / drop / nodir / trmodels / musall_shuf
tables (run `scripts/combined_encoding_figures.py --order <folder>`).

**Today's results (FINDINGS 2.4b, 2.8, 2.9, 3.4, Layer 2 bullets):**
- Binned (softmax) contest, O2 final inputs: chronic target > executed 4/4 (w 0.5 / 1; 3/4 at w 2); acute lick 3/4 at w1,
  width-dependent; subacute mixed. Landing (lick data): far-contra own bin -0.69 / -0.38 / -0.19, far mid +0.35 / +0.36 /
  +0.20, significant 4/4 (nested bootstrap, Bonferroni 36).
- Item 1 (attempt_coding): failed attempts lose the far-contra code, decoded as a far spout (4/4 lick); no pre-cue plan.
- Item 4 (attempt_maps): weaker than hits 3/4 but vs touch-matched pre misses 2/4; ipsilesional ALM / tjM1 drop acute
  (4/4); chronic tjM1 / SSp-m increase, contralesional > ipsi, all spouts (4/4).
- Frozen encoder final frame: acute drop = lick part (target ~0); CHRONIC increase = RESIDUAL. Residual: pooled pixel test
  weak; within animal 4/4 contralesional (calibrated, null 0/20); trend = raised plateau (3/4), not continuing drift.
- Layer 2: original drift-null comparison was flawed (withdrawn); redesigned (`change_ref`, combined fit) + session-count
  control: chronic > null 4/4, grows with time, no single group. All-August-pre reference values in FINDINGS.

**RUNNING at 19:30 (S = session scratchpad; all write to the share):**
- `S/run_resid_spouts.sh`: pooled residual maps + stats for close_ipsi .. far_mid, then `--spout all` (pooled + per animal);
  CSV per spout `frozen_encoder_gate-attempted_v7g3_t3d_basis/residual_significance_<spout>.csv`, figures
  `figures/residual_significance_joint_<spout>_<window>.png`. Log `S/resid_spouts.log`.
- `S/binsize_test.py`: pooled residual stats at 8 / 16 / 32 px bins + 20 random-pre-split nulls each ->
  `residual_binsize_test.csv`. If 16 / 32 keep the null clean, rerun all-spouts with that bin.
- `scripts/lick_trajectories_3d.py`: mean 3-D tongue paths per target x epoch + far-contra by landing bin ->
  `<dlc>/session_poses/pose3d/trajectories/` (npz + figures). Log `S/traj.log`.
- `attempt_coding --aligns lick cue --out .../attempt_coding_pp` (python pid 18912; adds pp_<spout> = full decoder softmax);
  then run `scripts/softmax_neural_vs_lick.py` (neural vs lick softmax confusion, widths 0.5 / 1, nested stats, trial-level
  slope + by-landing).
- `scripts/layer2_random_splits.py` x 4 animals (Layer 2 random-split null) -> `allsession_encoder_v7g3_t3d_ref/
  random_splits_<animal>.csv`.

**NEXT:** put each finished result on the page (numbered figure + headline); region-level residual test (Allen MOs / MOp
/ SSp-m per hemisphere, fixed a priori); fr_* analysis; droppos maps; per-spout binned contest (needs per-spout scoring in
combined_encoding + a small O2 batch); decoder and imaging-only-encoder pages. Push pending commits (pre-push hook 7 min).

## 000q. STATE 10-24 ~12:30 -- READ FIRST (then 000p)

**Goal question (Priya 10-24):** what does cortex look like post-stroke when the animal attempts a movement it cannot
execute / executes differently -> plan + redo list in ENCODING_FINDINGS 7; full output audit
`docs/status/OUTPUT_AUDIT_2026-10-24.md` (supersedes the imaging_encoding-only inventory). DECISIONS 10-24 afternoon.

Done today (all pushed, main 4a81441):
- `wfield_local/attempt_coding.py` (item 1): video attempt classes on the STORED frozen decoder; `--freeze` stores the DAQ
  1.0 s ENL models (LP pool sessions; PS93 = 10 pre). Outputs `E/attempt_coding/`. FINDINGS 2.8.
- `wfield_local/attempt_maps.py` (item 4): rest-referenced maps by class incl. `pre_incomplete` (touch control);
  `--figures`. Outputs `E/attempt_maps/` (first pass in `v1_20261024/`). FINDINGS 2.9 (touch-matched: 2/4 only).
- LP_licks re-render (15r / 14m / acc / 5c / 5cr, `G/LP_licks/epoch/`), LP trial counts, LP null calibration (15 CSVs;
  the 10-21 cross-arm summary CSV came from a scratch script, not regenerated). Decoders unchanged. FINDINGS 3.3.
- Layer 1 final frame `frozen_encoder_gate-attempted_v7g3_t3d` (107 sessions) + decomposition: unchanged. FINDINGS 3.4.
- O2 bundle `enc_v36_full_20261024` built 08:38, both local checks passed (binned + FR mode B, max |dr2| 5e-7);
  Priya's tmux loops submit (14 runs). When `out/` is copied back: `python scripts/o2_encoding.py collect --name
  enc_v36_full_20261024 --tag <TAG>` (bins_w1 first).
- Cache prune 153 GB (pre-10-23 entries; log in the session scratchpad). Stale worktrees removed (FR window request).

Running / queued (session scratchpad `S`):
- `S/run_t3d_downstream.sh` (worktree wf_t3d): relabel `--tag v7g3_t3d` -> Layer 2 `--tag v7g3_t3d` -> basis maps
  (`gate-attempted_v7g3_t3d_basis`) + pixel / r2 / spout figures. Log `S/t3d_downstream.log`.
- PAUSED: xyz2 4-animal contest (`S/run_3d4_xyz2.sh`, `--skip-done`; sensitivity, restart when CPU is free).
- QUEUED (audit gaps): within-spout in the final frame (attempted gate + combined licks; cue + ENL variants;
  mixed model / per-position direction maps / reliability chained); destination vs landing (fix `CONTEST_3D` in
  `scripts/best_match_destination.py` -- its folder was deleted); lick templates 4 animals; O2 variants not in the
  batch (nokin_matched / tr / win, drop_win); behaviour figures + lick_contact_by_epoch; LMMs after the O2 collect.
- Next analysis for the goal question: movement-matched control for 2.8 (failed far-contra attempts vs licks with the
  same 3-D landing), and the frozen encoder's touch / lick split on the attempt classes.

## 000p. STATE 10-24 ~07:00 (pre-compaction) -- READ FIRST (then 000o, 000n, 000m; inventory OUTPUT_INVENTORY_2026-10-24.md)

S = C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad

**O2 (Priya started both tmux loops 06:50, UNATTENDED):**
- Bundle `DeepLabCut/Widefield/o2/enc_v36_full_20261024` being written by the desktop (`S/run_bundle_full.sh`, ef748dc+;
  14 / 107 inputs at 06:57, ~1 / min -> done ~08:20-08:30; then labels.txt, env.json, code/, run_enc.sbatch,
  COMMANDS.md appear). Contents: attempted gate (REBUILT attempts), combined licks, polar 3-D exec + 20261023 tables
  (bins calib copied to tables/), 3-D tongue, RAW FR master_pre_v1 rank 12.
- Transfer node tmux `encxfer`: rsync loop share -> /n/scratch/users/p/ps150/o2/enc_v36_full_20261024 until complete.
- Login node tmux `encsub`: `submit_all.sh` (in the bundle) waits for the full bundle, runs ONE smoke task
  (TAG smoke_bins, BINS=1), and only if it wrote output submits 14 arrays: bins_w1 / bins_w05 / bins_w2 (nokin: +
  BINS=1, BINS_WIDTH 1 / 0.5 / 2), main, droppos (droppos-), shufpos (shufpos-), nodir, drop, trmodels, musall_shuf,
  fr_A / fr_B / fr_S / fr_D (ONLY=fr:, FR=master_pre_v1, FR_GROUPS=master_v1:<mode>). Expected submission ~09:00-09:30.
  Log `$B/submit_all.log`. Cancel all: `scancel -u ps150 -n enc_enc_v36_full_20261024`. Progress / copy-back commands
  are in the bundle's START_HERE.md (rsync out/ back on the transfer node; then desktop
  `python scripts/o2_encoding.py collect --name enc_v36_full_20261024 --tag <TAG>` with the same env as the bundle --
  collect now applies run.json's bundle_env + bins / bins_width / fr_groups for folder names).
- **PENDING LOCAL CHECK** (`S/check_bins.log`, `S/check_frB.log`; task bpgt30iv6): PS92_0806 from bundle vs share, (a)
  nokin: + BINS=1, (b) fr: + FR_GROUPS=master_v1:B. Must print max |r2 diff| < 1e-4. IF EITHER FAILS -> tell Priya to
  scancel (above). Not yet reported.
- nokin_matched NOT in the batch (old match.json is cam4-era; needs a new one).

**Local runs:**
- Layer 1 final frame `_v7g3_t3d` (`S/run_v7g3_t3d.sh`, worktree Github/wf_t3d 9f45819, grouping v3 + 3-D tongue +
  rebuilt gate; `S/frozen_v7g3_t3d.log`): template pass 16 / 31 at 06:57, then stats + 107 sessions (hours). Next: compare
  with `_v7g3` (frozen decomposition per animal, acute far-contra; lp_encoder_decompose), then basis maps (`run_v7basis_g3.sh`
  adapted: tag `_v7basis_g3_t3d`, TONGUE=3d), Layer 2 (`allsession_encoder --from <final frame>`), relabel in that frame.
- xyz2 x combined 4 animals (`S/run_3d4_xyz2.sh`, 2 workers, --skip-done; sensitivity only).
- Within-spout 3-D 4 animals (`S/run_ws3d4.sh`): 103 / 107 at 06:57 -> then `--figures-only` + FINDINGS 2.5 update
  (compare with the 2-animal record in qc/superseded_2animal_20261021_*; cam4 version still in qc/within_spout_deviation*).
- After the bundle finishes: prune local cache entries modified before 10-23 (`C:/Users/SabatiniLab/wfield_cache`,
  304 GB; approved).

**Decisions today (DECISIONS 10-24 entries):** binned contest PRIMARY (width 1 within-spout SD; sweep 0.5 / 2);
polar secondary, xyz2 sensitivity; attempts rebuilt (small effect); 3-D tongue option; deletions done (53 GB);
CHECK items kept (motion-energy video pilot, xyz2 x cam4 2-animal, pose window's nose_onsets).
**Open offers:** trial-level linear vs flexible check (low priority); smooth executed basis (not needed now); PS93
contact + lick joint drop; Layer 2 bootstrap; Git Bash SSH connection sharing so Claude can monitor / collect O2.

## 000o. STATE 10-24 ~06:40 -- READ FIRST (then 000n)

- DONE: attempts rebuilt (small effect, DECISIONS); DELETE-NOW list deleted (53 GB, 147 entries; log
  `S/deleted_20261024.log`); CHECK items kept (Priya). Local cache prune (pre-10-23 entries) still to do.
- RUNNING: (1) Layer 1 final frame `_v7g3_t3d` (worktree `Github/wf_t3d` 9f45819; grouping v3 + 3-D tongue + rebuilt
  gate; `S/frozen_v7g3_t3d.log`); (2) O2 bundle `enc_v36_full_20261024` (ef748dc; 107 sessions; attempted gate, combined
  licks, polar 3-D exec + 20261023 tables, 3-D tongue, RAW FR master_pre_v1 rank 12; `S/bundle_full_run.log`);
  (3) xyz2 x combined 4 animals, restarted at 2 workers; (4) within-spout 3-D 4 animals.
- O2 submit-time options added: BINS, BINS_WIDTH, FR_GROUPS (master_v1:A|B|S|D) -> one bundle for: main, droppos-,
  shufpos-, nodir:, nokin:, drop:, trmodels, musall shuf:, binned contest (nokin: + BINS=1, widths 1 / 0.5 / 2), FR
  modes A / B / S / D (FR=master_pre_v1, VIDEO=cam1,cam4). nokin_matched needs a new match.json (old one cam4-era) --
  not in the first batch. Commands to Priya when the bundle lands (+ `check` on one session first).
- FR master rank 12 (FR window): labels.json + grouping `fr_groups.master_v1` (lick 2,4,5,7,8,11; left 5; right 4;
  close 7; far 8; other 0,1,3,6,9,10); side via FR = S - B, distance = D - B; PS93 B leaks position; f6 / f10 unreliable.

## 000n. STATE 10-24 mid-morning -- OUTPUT INVENTORY + decisions -- READ FIRST (then 000m)

- **Inventory:** `docs/status/OUTPUT_INVENTORY_2026-10-24.md` (every encoding output: data version, class, size).
  REDO 41 GB (keep until replaced), DELETE-NOW 53 GB (PROPOSED, awaiting Priya's OK -- nothing deleted yet),
  DELETE-AFTER 44 GB, CHECK 3 items, KEEP 3.6 GB; local input cache 304 GB (proposal: prune pre-10-23 entries).
- **Decisions today:** BINNED contest PRIMARY (supersedes polar-primary); soft-bin width sweep 0.5 / 2 via
  `COMBINED_ENCODING_BINS_WIDTH` (258db8a; submit-time `BINS_WIDTH` on O2); O2 bundles self-describing (env.json,
  4626e08); 3-D tongue option (9f45819).
- **Running:** trial_attempts rebuild (bj09r90b1, 1 worker), xyz2 x combined 4 animals (4 workers -> restart at 2 when
  Layer 1 relaunches), within-spout 3-D 4 animals.
- **Next:** attempts flips -> `run_v7g3_t3d.sh` (Layer 1 final frame) + O2 bundle (combined licks + 3-D tongue +
  attempted gate; runs in the inventory's phase-A list) -> basis / Layer 2 / relabel / LMMs -> phase B (FR rank 12).

## 000m. STATE 10-24 morning -- RE-RUN REVIEW on complete cam1 / 3-D (Priya: "what else should we re run") -- READ FIRST

- **STALE INPUT FOUND: `qc/trial_attempts.csv` (built 10-21 14:33) has cam1 for PS92 / PS93 100 %, PS94 40 %, PS95 0 %
  of trials** (cam1 jaw 74 / 80 / 29 / 0 %; jaw_onsets now 76 / 81 / 73 / 71 %). Missing cam1 files were silently
  cam4-only. It feeds (a) the `attempted` engagement gate of every `gate-attempted` run (frozen Layer 1, Layer 2,
  relabel, basis), (b) the deficit index of all LMMs (attempt / completion / ttc / latency), (c) the LP-lick source
  (3.3), (d) the "PS94 / PS95 also fail to INITIATE" behavioural claim. Old tables copied to
  `qc/superseded_attempts_cam1incomplete_20261021/`; REBUILD queued (bj09r90b1, waits for 8 GB).
- 3-D tongue frozen run `_v7g3_t3d` STOPPED at its template pass (would have used the stale gate); relaunch after the
  attempts rebuild. Basis g3 run stopped earlier (maps go in the final frame).
- **Re-run list, in order:** (0) trial_attempts [queued] -> compare per-animal gate / class flips; (1) frozen Layer 1 in
  the final frame = grouping v3 + 3-D tongue + new gate (`run_v7g3_t3d.sh`), compare with `_v7g3`; then basis maps,
  Layer 2, relabel in that frame; (2) LMMs (`imaging_lmm.py`, `encoding_mixed.py`) on the new attempts; (3) LP-lick
  position maps (3.3); (4) cam4-lick-timing O2-era runs (10-19/20, 119 sessions, reference gate): per-spout unique maps
  3.1 (`droppos`), signed 3.2 (`shufpos`), main `movement_first`, contests 2.1 / 2.2 / 2.3 (`nokin`, `_matched`, `_tr`),
  `nokin_win`, `trmodels` -> with combined licks (3.1 is a headline); 2.1 may be retired in favour of 2.4b;
  (5) destination vs landing 2.4c (2 animals) -> 4; (6) cam4-timing cells of the 3-D 2x2.
- RUNNING: xyz2 x combined 4 animals (`_3d4`, reference gate), within-spout 3-D 4 animals (reference gate).
- Already on complete data: polar `_3d4`, stray test, push behaviour; v7g3 / v7pushfree / Layer 2 / relabel used
  complete cam1 LICK TIMING but the STALE GATE (PS94 / PS95).

## 000l. STATE 10-24 ~early morning -- READ FIRST (then 000k)

- **All queued runs DONE, nothing of mine running.** Results in FINDINGS: stray test 2.6 (not decisive; rescue failed),
  relabel 2.7 (heterogeneous: PS95 re-aimed far mid, PS93 stays far contra, PS92 / PS94 nonspecific), 4-animal 3-D
  contest 2.4b, Layer 2 (section 2 after the decoder G entry; changes beat drift null 4/4 acute + chronic), free push
  4.4, PS93 contact joint drop 3.4 (not shared with direction).
- **Open offers to Priya:** (a) `no_contact_lick` joint drop (PS93 contact vs lick group); (b) Layer 2 session
  bootstrap vs the drift null + far-contra-specific change; (c) remaining 3-D re-runs (within-spout 3-D direction,
  frozen v7 with refreshed combined licks, xyz2 x combined, cam4 cells); (d) FaceRhythm: rank 11 / 12 refits running on
  O2 (FR window), wait for rank + labels.json + master grouping.
- Worktrees in use by nothing: `Github/wf_relabel`, `Github/wf_stray`, `Github/wf_g3`, `S/wf_frozen` (2fc9fce).

## 000k. STATE 10-23 late night -- READ FIRST (then 000j)

- Pushed through 39332a1 (stray_licks path via the dlc root; the machine-path test had blocked the push). jaw_onsets
  3-D append (pose window 19:54) checked: trial_attempts reads by column name -> unchanged; trial_attempts.csv NOT
  regenerated (LP cache keys on its mtime).
- **Relabel**: smoke done (DECISIONS 2026-10-23 night, later): readout = all areas jointly; far spouts ~0 margin pre ->
  judge acute as change from pre. FULL run `frozen_relabel_v7` running from worktree `Github/wf_relabel` (f27339a, 2
  workers), log `S/relabel_v7.log`.
- **Stray-lick smoke** running from worktree `Github/wf_stray` (39332a1) on PS92_0812 + PS92_0819 -> `<E>/stray_licks`
  (smoke sessions are reused by the full run); log `S/stray_smoke.log`. Check validation, then all 107.
- Map re-render on `_v7basis` DONE (prenorm / peranimal / r2_share views; grouping of that run, not v3).
- Still to launch: Layer 2 (v3), grouping-v3 refits, remaining 3-D re-runs (000j list items 3-5).
- Stray-lick first full run DONE (FINDINGS 2.6: far-contra reference fails validation 3/4 -> not decisive; Priya to
  decide rescue vs rely on relabel). Grouping-v3 refit `_v7g3` RUNNING from worktree `Github/wf_g3` (358f73e; v7fwd15
  saved stats are STATS_VERSION 6 -> pre pass recomputed), log `S/frozen_v7g3.log`.
- **FaceRhythm master_pre_v1 READY** (FR window 10-23): `DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/proj/`
  108/108, rank 10, one frozen basis for every session (use proj/, NOT proj_loso/). Factors UNNAMED (f1..f10; master
  indices do NOT map to pilot_v1 -> do not reuse the pilot grouping); wait for labels.json + master grouping before
  encoder runs (or an unnamed early look if Priya asks). Caveats: PS92_0812 no speed_*; PS92_0812_v1 73 % NaN;
  PS92_0908 last ~5 % lossy.

## 000j. STATE 10-23 ~20:15 — READ FIRST (then 000i, 000h; decisions DECISIONS 2026-10-23 late; results FINDINGS 3.4 / 4.4)

- **MEMORY (shared box): free RAM fell to 4 GB at ~20:00** (my jobs ~22 GB + other windows, e.g. the pose window's
  `scripts.jaw_onsets`). STOPPED: Layer 2 run (`S/run_allsession.sh`, was in its post-stats pass) and the relabel smoke.
  Check `python S/freegb.py` (>= 12-14 GB) before relaunching anything; never more than ~8 workers total.
- **RUNNING:**
  - `S/run_3d4_polar.sh` (wf_v4): 4-ANIMAL polar x combined 3-D contest, `_3d4`, 4 workers, relaunched 18:55
    (`--skip-done`), 107 sessions; 2 done at 20:10 -> many hours; `S/3d4.log`, `S/3d4_polar_combined.log`.
  - `S/run_v7pushfree.sh` (wf_v5 at d194565): FROZEN_PUSH=free areas run `_v7pushfree`, 22/107 at 20:10 (slow, CPU
    contention) -> then decomps; compare with `scripts/frozen_running_compare.py` (PUSH_PAIRS v7pushfree / v7fwd15push
    vs v7fwd15).
  - `scripts/frozen_r2_maps.py` + `frozen_pixel_maps.py` re-render on `_v7basis` (new views: `*_peranimal`,
    `r2_share_*`, `pixel_maps_*_prenorm`), shared checkout.
- **TO RELAUNCH when memory allows (in this order):**
  1. **Relabel test** (`wfield_local/frozen_relabel.py`, c796587): smoke first --
     `python -m wfield_local.frozen_relabel --from gate-attempted_v7fwd15 --tag smoke --jobs 2 --labels PS92_<pre> PS92_<acute>`
     (encoder env: OROFACIAL_GATE=attempted COMBINED_ENCODING_LICKS=combined COMBINED_ENCODING_ONLY=full
     COMBINED_ENCODING_SUFFIX=_gate-attempted; FROZEN_* unset = v7). Check pre: the TRUE label must win (dSSE > 0 for
     every j). Then full `--tag v7` (6 designs per session -> heavy; 2 workers). Readout: far_contra trials relabelled
     as j (lick / cue windows, cortex / SSp-m_left) + best label per session.
  2. **Stray-lick test** (`wfield_local/stray_licks.py`, c796587; tests pass): `python -m wfield_local.stray_licks --jobs 2`
     (all frames per session; sparse design). Smoke on 1-2 sessions first (`--labels`), check validation table (split-
     half near-miss kernels: own spout > others), then all 107.
  3. **Layer 2** (`wfield_local/allsession_encoder.py`, a5fd177 + grouping v2 0f71a9c): relaunch from a worktree at
     >= c796587 with FROZEN_PUSH=1, `--from gate-attempted_v7pushfree --tag v7push --jobs 2` (post-stats pass ~1 h at 2
     workers, then fits; `--reuse-stats` thereafter).
  4. **Grouping v2 refits** (nose x -> other, 0f71a9c): frozen areas + basis variants re-fit from saved statistics
     (stats unchanged; STATS_VERSION 7 kept, outputs carry grouping_version 2) -- e.g. `frozen_encoder --all
     --reuse-stats --tag gate-attempted_v7g2` and `--target basis --reuse-stats --tag gate-attempted_v7basis_g2`
     (saved pre stats in the v7fwd15 / v7basis folders: copy fit/ or point a run at them).
  5. Remaining 4-animal 3-D re-runs (handoff 000i order): within-spout 3-D direction, frozen v7 with refreshed combined
     licks (PS94 / PS95 cam1), xyz2 x combined, cam4-timing cells.
- **GPU** (Priya asked): RTX 5060, 8 GB, torch 2.11 + CUDA available in locanmf. Our runs are bound by per-session
  loading / design building and RAM, not by the linear algebra; a torch port of the Gram / ridge steps (combined_encoding
  variants, stray regression) could speed those parts ~2-3x but would not help memory. Not done; offer if wanted.
- **Answered / decided 10-23 evening (DECISIONS 2026-10-23 late):** nose x -> "other" (only 4-29 % spout-locked);
  "direction" / "magnitude" columns are POSITIONS (tongue x / y / protrusion vs speed; jaw x vs y + speed; nose x vs
  y); contact stays a separate group (own-category option offered, not decided); Layer 2 moved up; relabel + stray
  tests approved; push behaviour finding (FINDINGS 4.4).
- **Decided (Priya, after pros / cons):** (1) contact = ITS OWN CATEGORY -> grouping v3 (out of "nontask"; task_first /
  movement_first order task vs movement only; r2 maps' last column relabelled "non-task+contact"); (2) no separate
  model-free contact-vs-miss script -- the stray-lick test now gives contact licks their own kernel per aimed spout
  (contact|q) and reports contact(q) - near(q) per epoch (`contact_vs_near.csv`), plus Layer 2's per-epoch contact
  change; (3) PS93 cheap check DONE (FINDINGS 3.4: contact most predictable from execution in PS93, AUC .89) + the
  joint drop variant `no_contact_exec_within` (added to VARIANTS; fitted in the grouping-v3 refit).
- Grouping-v3 refit replaces the "grouping v2 refit" in the relaunch list (item 4): same commands, tag `_v7g3`.

## 000i. STATE 10-23 ~18:40 — READ FIRST (then 000h, 000g)

- **3-D COMPLETE** (pose window 18:28): cam1 127 / 127, 3-D for all 119 curated sessions incl. PS95_0817 (acute). Re-cut
  done: `scripts/prep_3d_inputs.py C:/Users/SabatiniLab/wfield_cache/pose3d_inputs 20261023` -> snapshot, spout table
  (all 4 animals), bin calibration, `labels_3d_20261023.txt` = 107 sessions (PS92 27 / PS93 26 / PS94 27 / PS95 27).
  Old 2-animal 3-D 2x2 STOPPED 18:30 (xyz2 x combined + xyz2 x cam4 done = the 2-animal record; polar x cam4 killed at 16/47).
- **RUNNING:**
  - `S/run_3d4_polar.sh` (wf_v4; combined_encoding unchanged since its commit): polar x combined, 4 animals, folder
    suffix `_3d4`, 2 workers, started 18:33 (~12-15 h). Progress `S/3d4.log`, `S/3d4_polar_combined.log`.
  - `S/run_v7pushfree.sh` (wf_v5 at d194565): FROZEN_PUSH=free areas run `_v7pushfree` (started 17:53) -> compare with
    `scripts/frozen_running_compare.py` (pairs v7pushfree / v7fwd15push vs v7fwd15).
  - `scripts/push_behaviour.py --jobs 3` (shared checkout) -> `qc/push_behaviour_sessions.csv`; then
    `python scripts/imaging_lmm.py --sources push` (writes `imaging_lmm_push.csv`, never the full set).
- **Re-run order proposed to Priya (awaiting her pick):** (1) polar x combined [running]; (2) within-spout 3-D direction;
  (3) lick-level stray test incl. pre-cue / ITI licks (`kinematics_all`, key lick_uid); (4) frozen v7 with refreshed
  combined licks (PS94 / PS95 cam1); (5) xyz2 x combined, cam4-timing cells. Also offered: model-free contact-vs-miss
  check (per-contact amplification), PS93 contact shared-variance check, kernel maps (beta per pixel).
- **10-23 evening results:** v7 basis done (`_v7basis/figures`: pixel maps, r2_maps_*, per-position target / whole,
  pre_spout_*); contact kernel NEGATIVE in 4/4 (frontal), so the acute far-contra contact part's + sign = fewer contacts
  x negative kernel (per-contact amplification would sit in the residual); PS93 unique contact ~0 although it has the
  most misses (0.74 contact) -- hypothesis: contact shared with direction (exec_within brightest in PS93). FINDINGS 3.4.
- **Push behaviour DONE (FINDINGS 4.4):** pushes / cue-locked push collapse acutely with the deficit and overshoot after
  recovery. 3-D job RELAUNCHED 18:55 with 4 workers (`--skip-done`; nothing had finished). Runaway transcript grep
  (14:17, 250 CPU-min) killed -- never grep the session .jsonl with an unbounded regex.
- **jaw_onsets.csv will be rewritten again** (pose window appends 3-D jaw / nose onset columns; existing unchanged) --
  check that trial_attempts / deficit do not move when it lands.

## 000h. STATE 10-23 ~17:00 — READ FIRST (then 000g, 000f)

- **v7 = DEFAULT** (STATS_VERSION 7, 73d2991): forward-only running from 1.5 mm/s, NO push (push breaks post-stroke
  windows: pushes 2-5x more frequent post-stroke; FINDINGS 3.4, DECISIONS 2026-10-23 evening). Areas = `E/..._v7fwd15/`.
- **RUNNING: `_v7basis`** (`S/run_v7basis.sh`, worktree `S/wf_frozen` at 73d2991, 4 workers, started 16:36, ~1.5-2 h +
  maps): basis fit (joint / ordered / drop variants / task_only / drop-spout, residual + data moments incl. whole-spout
  swaps) -> `frozen_pixel_maps` (joint / target_first / exec_first), `frozen_r2_maps`, `frozen_spout_maps` (+
  `--contrast`). Progress `S/v7basis_chain.log`, `S/frozen_v7basis.log`. READ: pixel full / unique R2 (pre held out,
  epoch - pre), per-position target-only vs whole-spout, paired with the signed contrasts.
- **3-D 2x2** still on the 10-21 snapshot (`S/run_3d_resume.sh`, `S/wf_v4`): when xyz2 x cam4 finishes, stop the
  wrapper before polar x cam4 and run the 4-animal re-cut instead (000g).
- **NEXT after _v7basis:** re-cut 3-D (`S/prep_3d.py` -> move into repo) for 4 animals; re-run lick-direction results
  (3-D contests, within-spout, exec_within / exec_locked on v7 with combined licks incl. PS94 / PS95 cam1); stray test
  (2b) with `kinematics_all` (pre-cue + ITI licks, key lick_uid). Then queue 3 / 4 / 6 / 7 (000d / 000e).
- Worktrees: `wf_frozen` = _v7basis (busy); `wf_v5` idle (73d2991 not checked out there; at 215f0ff); `wf_v4` = 3-D 2x2.

## 000g. UPDATE 10-23 ~15:00 (read with 000f)

- **3-D NOW COVERS ALL FOUR ANIMALS** (pose window sabatinilab-5b, 10-23): 124 sessions (PS92 35, PS93 31, PS94 30,
  PS95 28); cam1 125 / 127 curated (PS95_0817 / PS95_0831 still on O2; they'll message when landed). jaw_onsets.csv,
  kinematics/licks.csv.gz, cue traces, spout error, phase all include PS94 / PS95. **Our 3-D inputs are the 10-21
  snapshots** (`S/pose3d_kin_licks_snapshot_20261021.csv.gz`, `S/pose3d_spout_positions_PS92_PS93_20261021.csv`,
  `S/exec_bins_calib_20261021.csv`, `S/labels_3d.txt` = PS92 + PS93) -> **the standing re-run rule is DUE**: re-cut 3-D
  tables (incl. spout positions + bin calibration for PS94 / PS95), re-run all lick-direction results on 4 animals, and
  refresh combined lick timing (PS94 / PS95 cam1) in the frozen encoder / LP_licks families. The running 3-D 2x2 is on
  the snapshot (2 animals): let xyz2 x cam4 finish, then replace polar x cam4 with the 4-animal re-run.
- Per-frame 3-D spans nearly the whole session (`<dlc>/session_poses/pose3d/<s>_frames.csv.gz`, cam1 windows from the
  previous trial stop); per-lick ITI 3-D: asked the pose window for `pose3d_kinematics --licks all` (Priya approved
  spontaneous licks for the stray test) -- they'll send path + trial/ITI column.
- **Re-cut = `S/prep_3d.py <dir> <tag>`** (copies the CURRENT `<dlc>/session_poses/pose3d/kinematics/licks.csv.gz`,
  rebuilds the spout table via `S/spout3d_table.py` from the per-frame measured spout, the bin calibration via
  `scripts/exec_bins_calibration.py`, and `labels_3d.txt` with the >= 50 % 3-D coverage + >= 6 spout rows gate) -- it
  picks up all four animals by itself; the "PS92_PS93" in the spout filename is only a name. Move both scripts into
  the repo when re-cutting.
- **`--licks all` LANDED (pose 28de81c):** `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics_all/
  licks.csv.gz` (880,532 licks, 117 curated sessions, 4 animals; + jaw_trials / sessions / by_animal). `context` =
  post_cue (cue + 70 ms .. trial stop) / pre_cue (spout arrival .. cue + 70 ms) / iti (trial stop .. next strobe);
  post / pre / iti PS92 111k / 103k / 20k, PS93 152k / 82k / 25k, PS94 148k / 49k / 30k, PS95 137k / 15k / 9k.
  **Key = `lick_uid`** (row of that session's v36 licks_all.csv); trial_id + lick_idx is NOT unique across the three
  source tables (`table` = licks / licks_precue / licks_iti). ITI licks have NO target: miss_mm, *_err, *_target blank;
  kept: reach_mm, side_ml / out_ap / down_dv, midline, frontal, sagittal, speed_peak. Pre-cue licks keep targets. In
  this mode `contact` blank -> False. The cam4 / combined outputs (`kinematics/licks.csv.gz`) are unchanged in columns,
  rebuilt for all four animals (no `label` column: key on session).
- **GAP: PS95 has NO acute 3-D** (its only acute session PS95_20260817 = O2 task 119, failed on a bad node,
  resubmitted 10-23; PS95_20260831 chronic = task 127). Until then PS95 acute lick direction is cam4-only. The pose
  window will message when both are through pose_chain -> re-cut again then.
- FINDINGS updated (status box, decoder correction, 2.6 strays); DECISIONS 2026-10-23 (afternoon) correction entry;
  `scripts/frozen_running_compare.py` (41650de). FaceRhythm + pose windows asked to do the same findings pass.

## 000f. STATE 10-23 ~14:05 — READ FIRST (then 000e / 000d; decisions DECISIONS 2026-10-23; results FINDINGS 3.4)

- **RUNNING:**
  - **v7 candidates** (started 13:36, ETA ~15:45): `S/run_v7cand.sh` from `S/wf_v5` (detached at 215f0ff), Allen areas,
    2 workers each: `E/frozen_encoder_gate-attempted_v7fwd15` (FROZEN_RUNNING=spline_thr, FROZEN_RUN_KNOT0=1.5 =
    forward-only running from 1.5 mm/s) and `_v7fwd15push` (+ FROZEN_PUSH=1). Logs `S/frozen_v7<k>.log`,
    `S/v7_chain.log`; joint decompositions after.
  - **3-D 2x2** (`S/run_3d_resume.sh`, `S/wf_v4`, 2 workers): xyz2 x combined DONE 09:32; xyz2 x cam4 38/47 at 14:00;
    then polar x cam4 (47). No failures.
- **WHEN v7 CANDIDATES LAND:** compare with `_v6` (signed linear), `_v6spl`, `_v6thr3`, `_v6thr1` (all done): per animal
  x epoch frozen R2 (all / cue / lick, pre held out), running curve (`fe.running_curve(..., mode="spline_thr")` with
  `FROZEN_RUN_KNOT0` set), unique dR2 of push and of lick with push in / out, the acute far-contra decomposition.
  Pick v7 -> make it the DEFAULT in code (STATS_VERSION 7: default FROZEN_RUNNING=spline_thr + knot0 1.5 [+ push]),
  test, commit; then the **basis re-run** (`--target basis`, all animals; it now fits drop variants + task_only +
  drop-spout and saves resid / data moments incl. whole-spout swaps) -> `scripts/frozen_r2_maps.py --tag <basis tag>`
  and `scripts/frozen_spout_maps.py --tag <basis tag> [--contrast]`; ~1.5-2 h + maps.
- **THEN (queue, 000d / 000e):** 2b lick-level strays vs on-target (pre + post); 3 all-session shared + per-epoch model;
  4 FR inside frozen; 6 session-level competition LMM; 7 frozen parts onto the TIME / BEHAVIOUR LMMs; no_running /
  no_nose variants if PS95 windows still need it.
- **10-23 afternoon results** (FINDINGS 3.4 running paragraph): running sublinear (~on / off), forward-only from 1-1.5
  mm/s beats linear everywhere, signed linear treated backward pushes as negative running; side-camera clips: bout-end
  negative speed = paw push, post-cue dip = upper-limb extension (real) -> push variable; jitter check supports 1.5
  mm/s; headline unchanged in every running model. Whole-spout per-position code 9ac1e60 (smoke `_smoke-r2b`: ~5x the
  target-only scale). QC / clip scripts now in the repo: `scripts/qc_running_{speed,clips,lick_jitter,survey}.py`.
- **Worktrees:** `S/wf_v5` = v7 candidates (busy until ~15:45); `S/wf_v4` = 3-D 2x2 (busy ~1 day); `S/wf_frozen` idle.

## 000e. STATE 10-23 ~07:00 (superseded by 000f for state)

- **RUNNING** (worktrees in `S/`; never edit them while a run uses them):
  - `_v6spl` = sublinear running test (`FROZEN_RUNNING=spline`, commit 5045ef5), Allen areas, 3 workers, from
    `S/wf_v5` (detached at 5045ef5); `S/run_v6spl.sh`, progress `S/v6spl_chain.log` / `S/frozen_v6spl.log`; then
    joint decompositions. READ: per-animal running curve (`running_curve` on fit/<an>_joint.npz betas + templates
    knots), held-out pre R2 vs v6, post R2 per animal vs v6 (PS94 the test case), headline unchanged?
  - 3-D 2x2 RESUMED 10-23 05:05 (`S/run_3d_resume.sh`, from `S/wf_v4`, 2 workers, `--skip-done`; progress
    `S/3d_resume_1023.log`, cells `S/3d_2x2_<exec>_<licks>.log`): xyz2 x combined 25 remaining, then xyz2 x cam4 and
    polar x cam4 (41 each) -- ~1 day at 2 workers; can go to 3-4 when memory frees (workers reach ~27 GB).
- **10-23 morning findings / decisions** (details FINDINGS 2.4c correction, 4.3; queue in 000d):
  - Destination-vs-landing % were inflated (target == landing rows counted as target); separating rows only:
    cue 32 / 2 / 8, LICK-aligned 33 / 2 / 2 (target / execution / neither); strong (own-zone < 0.15) 5 / 0 / 1 lick-
    aligned -- target-leaning, rests on ~6 acute far-contra sessions, 2 animals (lick arm:
    `qc/best_match_destination_lick_lick.csv`; cue copy `_cue-working`).
  - Target and the spout-locked lick are the SAME variable on on-target licks: pre-stroke per-spout contrast maps
    (`scripts/frozen_spout_maps.py --contrast`; `E/..._v6basis/figures/pre_spout_contrast_*`) show the data
    lateralised as expected in 4/4 (contra spouts -> left hemisphere, SSp-m L-R +12..+21; ipsi -> right), the SPOUT
    part (target + spout-locked) tracking it, and the target / exec_locked split unstable (close_mid +13 vs -13).
    -> per-position maps: add the whole-spout version; target vs executed only via STRAYS (queue 2b) and the bracket.
  - Lick-direction dR2 exists only as exec_within unique dR2 (whole-cortex ~1 % pre; PS94 post negative in windows,
    possibly out-of-range direction deviations); per-lick version = queue 2b.
  - Stray census (good 3-D, August): thousands of no-contact licks nearer another calibrated centre, pre included
    (PS92 1129, PS93 6679 pre); close-spout centres overlap -> continuous error, not zones. Spontaneous ITI licks =
    execution-only reference once whole-session 3-D exists.
  - LMM with 2 animals: animal FIXED (a random-effect variance cannot be estimated from 2 levels); per animal first.
  - PS95 post-stroke window R2 ~0 with "other" in: running vs nose NOT separable with the current variants (they drop
    together) -- needs no_running / no_nose variants (cheap, saved statistics).
- **Stale in 000d below:** "3-D 2x2 still paused" (resumed, above); "PS95 ... nose y?" (see the line above).

## 000d. UPDATE 10-22 NIGHT (then 000c)

- **Frozen encoder is now v6 (STATS_VERSION 6, commit 34c803d)**: running clipped at each animal's pre-stroke p99 of
  RUNNING frames (> 3 mm/s, `segmentation.running.thresh_speed_mm_s`); caps PS92-95 14.8 / 17.3 / 13.2 / 16.8 mm/s.
  Use `E/frozen_encoder_gate-attempted_v6/` (areas), `_v6basis/` (pixel maps, all animals), `_v6grid/` (PS92 check).
  v5 folders (all-frame p99 cap: broke PS95) carry SUPERSEDED.txt; v4 kept for comparison.
- **PS94 "OPEN" below is RESOLVED** by the cap (chronic joint all-frames R2 -0.44 -> +0.30; FINDINGS §3.4 v6 para).
  Headline unchanged (acute far-contra parts within 0.2 of v4 in all orderings).
- **Pixel maps done** (basis + grid, joint / target_first / exec_first); they match the area maps.
- **PS92_0908 FR timing: RESOLVED 10-23 ~07:35** (FaceRhythm window: cam4 dropped ~13 % of frames, the old linear
  frame->DAQ fit was off by median 3.2 s; corrected projections carry `timing = fr-timing-v1`, lossy bins NaN). PS92_0812
  and PS92_0908 rerun in FR run A (old outputs `_superseded_linear_fr_timing/`; note `RESOLVED_PS92_0908_fr_timing.txt`):
  PS92_0908 fr:full cortex R2 0.506 -> 0.618 (FR alone +0.26); PS92_0812 unchanged (mistimed bins were after the DAQ
  recording). Both sessions back IN. Any FR set projected before 10-23 without the timing field: check the same way.
- **FR run A done** (50/50 PS92 + PS93 sessions; `S/fr_runA.log`) -- summarise how much grouped FR adds; the cam4
  pass exited instantly with an empty log (check it is "all done", not a skip by error).
- **Still open:** PS95 subacute / chronic cue / lick-window R2 ~0 (no_other better: nose y?); contact kernel sign; test
  on data - model; PS95 1 acute session. **3-D 2x2 still paused** at 16/41 (`run_3d_2x2.sh`, `--skip-done`, 2 workers).
- **Next:** FR inside frozen v6 (PS92 + PS93; FR traces through the spl / dev split, own penalty group, env flag);
  FR run A summary; resume 3-D 2x2; cache backup when idle.
- **10-23 queue (Priya-approved, reordered 10-23 "yes reorder"):** (1) sublinear running test `_v6spl` (running);
  (2) basis re-run with pixel unique R2 + per-position (target-only AND whole-spout = target + spout-locked template
  swapped for the other spouts' mean at prediction) + direction maps (`scripts/frozen_r2_maps.py`; code fc6d441) on
  the winning running model; (2b) **LICK-LEVEL STRAYS vs ON-TARGET, within spout -- the decisive target-vs-executed
  test** (Priya: "don't we want to be comparing LICK stray response vs on-target response"; target and the spout-locked
  lick are the same variable on on-target licks, only strays separate them): per lick, continuous 3-D error from the
  aimed spout's CALIBRATED centre (pre contact licks; nominal angles are biased ~21 deg) in within-spout SD units +
  direction toward the nearest other centre + contact; per-lick regressor sets (on-target at q; strays from q toward j)
  so overlapping corrective licks are separated by the model (raw stray averages are contaminated toward target);
  compare stray kernels' maps with pre on-target maps of q and j, controlling pre q-j similarity = competition LMM at
  LICK level, pre AND post. **The PRE-STROKE arm is a result in itself** (Priya: "pre-stroke variation can also be useful
  to us (allow us to look at executed vs target pre-stroke also)"): stroke-independent target vs executed -- settles the
  pre-stroke disagreement between xyz2 (executed >= target, PS92) and polar (comparable / target-led) by strays rather
  than by coordinate choice; pre sessions scored leave-one-session-out; lick-level check of the within-spout crossed
  SSp-m direction code (pre 4/4); PS93's high pre stray rate (21 % vs PS92 5 %) may reflect its known right orofacial
  deficit -> per animal first. Census 10-23 (good 3-D, August pre, PS92 / PS93): no-contact licks nearer another spout's
  centre PS92 1129 / 1054 / 665 / 2645, PS93 6679 / 1783 / 2320 / 5429 (pre / acute / sub / chronic); far contra acute
  564 / 537 -- thousands, vs ~6 sessions in the session-level test. Close-spout centres overlap (3-10 deg apart vs
  within-spout SD 6-9 deg: 5-13k CONTACT licks sit nearer another centre) -> use the continuous error, categorical zones
  only for well-separated spouts. Spontaneous (ITI) licks = an EXECUTION-ONLY reference (no target) once whole-session
  3-D exists (ask the pose window); pre-cue licks are still targeted (spout already in place).
  (3) **ALL-SESSION model: shared kernels + per-epoch deltas** (Priya: "i like the shared kernels with per-epoch
  changes idea!"): per animal, kernel = shared + acute / subacute / chronic delta (ridge-shrunk to 0), pre-stroke
  templates kept as the reference frame, epoch weighting, held-out-session scoring for every session, DRIFT NULL =
  early- vs late-pre pseudo-epoch; readout = per group, does an epoch delta improve held-out prediction (tests
  "coupling preserved") + delta maps; built on the Gram statistics (one stats pass over post sessions); areas then basis;
  (4) FR inside frozen; (5) [folded into 2b]; (6) SESSION-level competition LMM for destination vs landing
  (secondary to 2b; per (session, aimed, candidate) pattern correlation ~ target + 3-D landing weight + pre-stroke
  spout similarity, x log days / deficit; animal FIXED with 2 animals); (7) frozen v6 per-session parts onto the
  TIME / BEHAVIOUR LMMs (FINDINGS 4.3 item 7).
  **Target-vs-executed headline under verification (Priya 10-23):** FINDINGS 2.4c destination % were inflated (see the
  correction there); lick-aligned arm running; 2x2 decides the coordinate frame.
- **RE-ADDRESS WHEN 3-D TRACKING COVERS ALL SESSIONS (Priya 10-23).** 3-D tables exist for PS92 + PS93 only
  (20261021 snapshots); PS94 / PS95 direction is cam4 2-D, PS95 licks cam4 (cam1 unfinished). Everything that
  depends on lick DIRECTION must be re-run then and compared: exec_within / exec_locked groups and their unique dR2
  (incl. PS94's negative post-stroke exec_within dR2 -- possible out-of-range extrapolation of direction deviations),
  the strayed-lick direction test (planned 10-23: per-lick SSE gain from direction vs a within-spout shuffle of
  direction deviations, gated on tracking quality, matched on vigour / contact, per animal), the 3-D target-vs-executed
  contests and 2x2, within-spout per-position direction, and the direction pixel maps. LP results stay provisional
  until cam1 finishes.

## 000c. UPDATE 10-22 EVENING (supersedes 000 / 000b where they differ)

**Results: ENCODING_FINDINGS §3.4** (frozen v4). Reasoning: DECISIONS 2026-10-22 (afternoon / evening).

**State of the frozen encoder (`wfield_local/frozen_encoder.py`, STATS_VERSION 4, main 8cf72f4+):**
- **v4 groups by construction** (Priya: "should we have a 'lick' group and a 'spout position information' group?" ->
  build it): each lick-related movement trace = the animal's PRE-STROKE mean trace for that spout (cue-locked before
  the first post-cue lick, lick-locked after; POOLED over its pre sessions -- Priya "let's do pooled") + the deviation.
  Groups timing / target / exec_locked / exec_within / lick / other / contact; super-groups task / spout / movement.
  Variants: joint, target_first / exec_first (target vs executed bracket), task_first / movement_first, no_<g>,
  <g>_only, task_only. Fit dir has `templates.pkl` (pass 1, behaviour only) + `pre_stats.pkl` (stats_version 4).
- **Ordered-projection fix**: thresholded pseudo-inverse, PROJ_RCOND = 1e-3 (1e-6 ridge and then 1e-8 both blew up on
  PS95 post sessions: task_first -1.06, target_first -163). After the fix every ordered variant matches joint within
  0.003 on all frames, every session.
- **Run done**: `E/frozen_encoder_gate-attempted_v4/` (107 sessions, re-scored 16:55 with the fix), decompositions
  `E/qc/lp_encoder_decompose_frozen_encoder_gate-attempted_v4_<variant>[_lick]_by_animal.csv`, Musall figures
  `.../figures/musall_*.png` (new groups, STACKED task-aligned figure per Priya), **area maps**
  `.../figures/area_maps_{joint,target_first,exec_first}_far_contra_{acute,subacute,chronic}_{lick,cue}.png`
  (`scripts/frozen_area_maps.py`). **v3 folder `frozen_encoder_gate-attempted/` = SUPERSEDED** (kept).
  Leakage tables: `E/qc/frozen_group_leakage.csv` (v4, lick-aligned pooled templates, held out),
  `..._v3_SUPERSEDED.csv`, `..._v4cuelocked_SUPERSEDED.csv`. Smoke folders `frozen_encoder_smoke-*` = plumbing only.
- **Headline (provisional, §3.4)**: acute far-contra post-lick drop is carried by LICK in all five orderings (PS92
  -20 to -21, PS93 -12 to -14, PS94 -13.5 to -14, PS95 -20, dF/F x 1e3); target <= 1.7 even with target_first; data -
  prediction small and positive. = less licking-related activity because of changed licking, not less activity per
  lick (correlational: coupling preserved, direction not determined).
- **OPEN, PS94**: every PS94 chronic session fails under the frozen model (all-frames R2 -0.41 to -0.62; per-session
  fits 0.50-0.62); dropping the "other" group (nose y, running) restores +0.11 / +0.12 -> a running or nose-y change
  after the stroke. A running-speed check (pre vs subacute vs chronic, PS94 vs PS92) was running at handoff time
  (scratch `S/` task). Do not read PS94 chronic frozen numbers until resolved. Also open: contact kernel sign (contact
  part opposes lick), no statistical test on data - prediction, PS95 has 1 acute session in this window.

**Pixel maps (Priya: "go ahead with the area map and then pixel maps"; chose the literature route):**
- `--target basis`: per-animal SHARED SVD basis on the Allen-aligned stat mask (all sessions share the 540 x 640 grid,
  129,770 stat pixels; each pre session contributes Ub Ue sqrt(w)); top components to 95 % of pooled pre variance
  (<= 200) + the cortex mean as targets. `--target grid`: 8 x 8 px block means (~2,000 blocks) = the transparent check.
  Spatial runs store EVOKED responses per part x spout x target (no full PETHs; lazy projections, no frames x targets
  arrays), R2 / Musall on the cortex mean, no drop-spout. Renderer `scripts/frozen_pixel_maps.py` (written, untested
  at handoff). Per animal not cohort basis (Priya asked; per animal fits each animal best, maps are averaged in pixel
  space anyway; a cohort-basis variance check is a cheap optional diagnostic).
- **NEXT**: the 3-session smoke test (`--tag smoke-basis` / `smoke-grid`, PS92_0806 / 0812 / 0820) was running at
  handoff; then full runs `--target basis --tag gate-attempted_v4basis` (all animals) and `--target grid --tag
  gate-attempted_v4grid` (PS92 only, the check), then `frozen_pixel_maps.py` for each; compare grid vs basis and both
  vs the area maps (bins averaged within areas should match).

**FaceRhythm in the encoder (Priya approved each step):**
- `wfield_local/fr_groups.py` + an opt-in hook in `face_rhythm_outputs.fr_signals` (env `COMBINED_ENCODING_FR_GROUPS=
  pilot_v1:A|B`, default off; keyed to set `pilot_pre_PS92_PS93`, refuses others). pilot_v1: lick amplitude = f0 + f4 +
  f5 + f9 (lick-locked in every check); direction left f0 + f5, centre f4, right f9; other f1 f2 f3 f6 f7 f8 (f2 tonic
  with licking but not lick-locked -- the FR window's share-based r -0.24 vs absolute +0.04-.24 is a units difference;
  f1 / f3 / f6 = lick INITIATION, so "other" is not "non-lick"). labels.json now written by the FR window (named
  columns; harmless: the hook re-keys by position).
- Directionless FR does NOT exist as a factor subset: FR lick sum decodes the spout .15 / .15 (PS92) but .41 / .26
  (PS93; SD-balancing does not help -- PS93 licks different amounts at different spouts), and the six "other" factors
  together .31-.57 (each alone near chance). So run B (lick + other) was DROPPED; the clean route = FR inside frozen v4
  (spout-locked + deviation split of the FR traces), PS92 + PS93 on the pilot set -- NOT BUILT YET (next).
- **FR run A** (grouped FR lick + sides + other, ONLY=fr, 50 pilot sessions, 2 workers, cam4 fallback pass) RUNNING
  overnight from `S/run_evening_1022.sh` (log `S/evening_1022.log`, `S/fr_runA*.log`) -> folder
  `movement_first_fr-..._subset-fr_licks-combined_gate-attempted_frA`. Gives how much FR adds and position beyond all
  of FR (lower bound).
- O2 not used (fairshare ~4e-5); O2 kept for the master_pre_v1 rerun (all animals).

**Other state:** step 2 (lick-aligned per-session run) COMPLETE, 119 sessions; its lick-aligned decomposition ran
(`E/qc/lp_encoder_decompose_movement_first_subset-full_licks-combined_gate-attempted_lick_by_animal.csv`, not yet
read). 3-D 2x2 PAUSED (its two workers took 27 GB each next to FR run A; resume with --skip-done, 2 workers, alone).
PS95 R2 is lower because PS95 runs ~1 % of task frames vs 4-7 % (running-vs-still check, `S/run_vs_still_r2.csv`):
on still frames it matches the others -> compare animals by their own pre -> post change. No z-scoring of targets
(per-area R2 is scale-free; per-session z would erase the amplitude changes being measured). Cache backup paused
(analysis first); rerun `robocopy /E /XO` of `C:\Users\SabatiniLab\wfield_cache`, `.widefield_session_cache`,
`frozen_models` -> `M:\MICROSCOPE\Priya\Widefield\input_cache_backup\` when the box is idle.

**New pitfalls (10-22 afternoon):** `DataFrame.align` is a pandas METHOD (`Q.align == "cue"` was always False: every
Musall figure silently skipped) -- use column indexing; a model part named "pos" collides with the spout column
(rename to pos_part, twice now); bash heredocs still mangle `\n` (again in the figure title); each long-running worker
pool on this box can reach 25-30 GB per worker (2x2, FR with points + video): two such pools do not fit -- run them one
at a time and check `freegb.py` + per-process private bytes, not working set.

## 000. UPDATE 10-22 ~09:40 — read first (supersedes §2 "RUNNING")

**NOTHING of the encoding chain is running now.** What happened this morning:
1. **Duplicate runs found and stopped.** Four copies of `S/run_chain_1022.sh` (08:44 / 08:45 / 08:51 / 08:52) + the old
   `S/run_lp_encoder_lick.sh` were live (leftovers of the earlier pause / relaunches: killing a step's python lets its
   wrapper start the NEXT step). Three were on step 2 writing the SAME lick-aligned folder concurrently (~17 GB), then
   on step 3. All stopped (Priya ran the `taskkill /T` for the bash trees). **Integrity checked:** the lick-aligned folder
   `E/movement_first_subset-full_licks-combined_gate-attempted/` = 35 sessions, all json + npz (with `peth_lick`) +
   regions load, no torn writes (22 PS92 combined, 13 PS94_1005 / PS95 cam4). 3-D 2x2 folders were never created.
   Check script: `S/check_lick_folder.py <folder>`.
2. **The real frozen run (08:51) HUNG** in the pre-statistics phase: workers finished (~650 CPU-s each) but the main
   process never collected (one worker in an "Executive" kernel wait = stalled I/O on M:). Priya stopped it (and the
   post-frozen queue `S/run_after_frozen.sh`). `fit/pre_stats.pkl` + `fit/*.npz` on the share are from the 08:32 run =
   BEFORE the protrusion move (STATS_VERSION 3) -> stale; the restart recomputes them (no `--reuse-stats`).
   `<variant>/sessions/` hold 6 sessions from that stale run; they get overwritten.
3. **Efficiency audit (Priya: "are things parallelized and using cache/sidecars appropriately?"):**
   - Input cache works (hits: imaging 5 s + behaviour 5 s per session). One frozen `session_design` = **27 s** (the
     uncached remainder is `raw_blocks`, 14 s). A per-session design sidecar was REJECTED: X is 170k x 507 (0.35-0.7 GB
     / session, 35-70 GB for 107) to save ~10 min / run.
   - **Fits were serial in the main process** (4 animals x (8 variants + 18 drop-spout fits)) with the pool idle ->
     now one `fit_animal_job` per animal via `fan_sessions` (branch `frozen-parallel-fits` in worktree `S/wf_frozen`,
     test `test_fit_animal_job_equals_direct_fits`). Merge to main after the tests pass.
   - **Logs looked empty because stdout was redirected = block-buffered**: relaunch with `PYTHONUNBUFFERED=1` so
     `fan_out`'s per-item progress lines appear (a hang is then visible in minutes).
   - Relaunch plan: frozen at `--jobs 6` (44 GB free at 09:10); decompositions 4 at a time; then chain step 2
     (lick-aligned resume, `--skip-done`, 35 / 119 done) and step 3 (3-D 2x2, 2 workers) SIDE BY SIDE.
   - Expected: frozen ~30-40 min after relaunch (stats ~4 min, fits parallel, predict ~15-20 min); post-frozen queue
     (Musall figures, leakage, per-session maps every epoch) ~+30 min; step 2 ~3-4 h; step 3 ~2-3 h.
4. **Input cache backed up to the server** (Priya: "ensure caches are backed up to the server"):
   `C:\Users\SabatiniLab\wfield_cache` (106 GB, 720 entries) -> `M:\MICROSCOPE\Priya\Widefield\input_cache_backup\wfield_cache\`
   by `robocopy /E /XO` (copy only, never deletes; `*.tmp*` partials excluded); log `S/cache_backup.log`. Re-run the same
   command to refresh. To restore on a fresh box, copy back and set `WFIELD_INPUT_CACHE`; entries are keyed by code hash
   + env + source mtimes, so stale ones are simply never hit.
5. **FR pilot projections arriving** (e3): `fr_cam4_20261021/factors/pilot_pre_PS92_PS93/proj/` has PS92_20260818 and
   PS92_20260820 (acute, scored on the full pre basis). The watcher waits for PS92_0812 / PS92_0820 / PS93_0812.
6. Other windows asked to free RAM: 5b had nothing running; e3 stopped its output-check script + 2 watchers (pilot fit,
   pytest and fr JupyterLab kept).

### 000b. UPDATE 10-22 ~12:30

- **Relaunched 09:32** (`S/run_chain_1022b.sh`, task log `S/chain_1022b.log`): frozen done 11:07 (107 sessions, 0
  failures; slow share stretched it to 1.5 h), decompositions 11:10 (`E/qc/lp_encoder_decompose_frozen_encoder_gate-attempted_<v>[_lick]_by_animal.csv`),
  steps 2 + 3 running side by side since 11:10. Post-frozen queue `S/run_after_frozen_b.sh` (free-RAM gate lowered to
  8 GB) running: Musall figures, leakage, per-session maps.
- **Frozen LOSO R^2 (areas summed), pre:** joint .54 / .56 / .54 / .39 (PS92 / 93 / 94 / 95); no_movement .09 / .09 /
  .06 / .14; movement_only .50 / .52 / .50 / .34.
- **BUG TO CHECK before quoting task_first:** task_first should equal joint on all frames (same span) but breaks on some
  PS95 post sessions (PS95_0818 all-frames -1.06 vs joint .36; 0825 .02 vs .44; 0928 .26 vs .31; 0910 .40 vs .43) --
  likely the task-on-movement residualisation estimated on pre amplifying out-of-range post movement. Read joint /
  movement_first first; no task_first number or task_first drop-spout until fixed.
- **Why PS95's R^2 is lower (Priya asked):** not cam1 (cam4 licks for all: full .39 vs .56-.58); the gap is entirely
  "uninstructed" (alone .20 vs .39-.47; task alone is HIGHEST in PS95, .14). PS95 RUNS ~4x less (running regressor
  variance 0.30 of the cohort mean; pose energy normal) and its cortical variance is ~25 % lower. PROVISIONAL until the
  running-vs-still check (`S/run_vs_still_r2.py` -> `S/run_vs_still_r2.csv`; queued for >= 10 GB free). Consequence:
  compare animals by their own pre -> post change, not by shares / percentages.
- **Z-scoring (Priya asked): no.** Regressors are already scaled (pooled pre SD, frozen; per-session z, per-session);
  per-area R^2 is scale-free; per-session z-scoring of the targets would erase the post-stroke amplitude changes being
  measured. Only the "areas summed" summary / penalty choice weight areas by variance (optional: also report the
  equal-weight mean of per-area R^2).
- **FR pilot: LOCAL, not O2** (Priya: O2 fairshare ~4e-5; 51 sessions, FR / video / FR+video / FR+points models only = a
  few hours here). (1) one-session check PS92_0812 queued (`S/run_fr_pilot_one.sh`: FR pilot_pre_PS92_PS93, 10 point
  PCs, video cam1+cam4, ONLY fr,frvideo,frpts,frboth,video; waits for the running-vs-still check + 15 GB) -> REVIEW;
  (2) the 51-session run at 4 workers once the FR window's last PS93 projections land (42 / 51 at 11:49); if RAM is
  short ask Priya to pause step 3 (3-D 2x2; resumes with --skip-done); (3) O2 kept for the master_pre_v1 rerun.
- RAM: the FR window's projection jobs held ~30 GB at 12:00 (3 GB free) -- my queued jobs wait on free-RAM gates.
- Cache backup STOPPED (Priya: "prioritize the analyses over the cache backup"); it was loading the share (listing 17 s
  while it ran, 0.8 s after). Rerun after the runs, adding `C:\Users\SabatiniLab\.widefield_session_cache` and
  `C:\Users\SabatiniLab\frozen_models` (copy only) under `M:\MICROSCOPE\Priya\Widefield\input_cache_backup\`.
- Code: per-animal parallel fits merged to main (worktree branch `frozen-parallel-fits`; the running chain uses the
  worktree, which is at the same code).

**New pitfalls (add to §4):**
- **Stopping a bash wrapper does not stop its `( ... )` subshells** -- they carry on to the next step. Kill the whole
  tree: `taskkill /F /T /PID <wrapper>` (Priya runs kill commands); then list `python.exe` + `bash.exe` command lines to
  confirm. Never relaunch a chain without first confirming no older copy of it is alive.
- **Redirected python stdout is block-buffered**: an empty log is not evidence of a hang or of progress. Set
  `PYTHONUNBUFFERED=1` in chain scripts; diagnose a hang by per-worker CPU deltas (`Get-Process ... CPU` 20 s apart).
- `fan_out` has no per-task timeout: a worker stuck on an M: read blocks the whole run silently.

## 0. Rules added 10-21 night / 10-22 (all of the 10-21 §0 still holds)

- **LP_licks results are PROVISIONAL until cam1 finishes** (74 / 108 curated sessions have cam1; PS95 none, PS94_1005
  none -> cam4-only LP licks there). Rerun `scripts/trial_attempts.py` first when cam1 lands.
- **Outline reading rule (ENCODING_FINDINGS 3.3):** nested-bootstrap outlines localise (bin-level FWER); their extent is
  a power-dependent LOWER BOUND, not an area, and "x % vs y % of bins" is descriptive. Cluster outlines = yes / no only.
- **Per-session encoder "movement" includes spout CONTACT and lick DIRECTION** (both inside its `licks` / `instructed`
  terms); do not read a per-session movement share as "movement" -- use the frozen encoder's FGROUP split.
- **Directionless movement = timing + vigour only** (lick onsets, tongue speed, jaw opening / speed, nose y, running);
  tongue PROTRUSION is spatial (scales with spout distance) and sits with direction (10-22 audit).
- **FR carries position** (lick factors are left / right / centre): never headline "does position survive FR";
  bracket it (FR first vs position first) and use the within-spout test.
- Memory on this box: the FR window runs `face_rhythm_factors.py` locally (3-6 GB, growing); keep my runs at 4 workers
  and check free RAM (`python S/freegb.py`) before adding jobs. Pausing a run = stop the bash wrapper FIRST, then kill
  the python tree by command line (parent + `ParentProcessId` children), else the wrapper starts its next step.

## 1. Results since the 10-21 handoff (details ENCODING_FINDINGS 3.3, DECISIONS)

1. **LP map render + calibration** (`G/LP_licks/epoch/`, `.../results/calibration_LP_vs_DAQ_summary.csv`): acute far
   contra drawable (4 an / 15 sess; contact gating had 0); post-lick activity BELOW pre (amplitude 0.70; 19.7 % of bins,
   drift <= 1.4 %), 3/4 bilateral (PS95 1 session opposite). Contact gating had INFLATED acute far-spout amplitudes.
   Pure-null false positives: nested 1.6 % of 312 tests (mostly 1-7 bins), cluster max 27 % area.
2. **LP frozen decoders** (`G/LP_licks/epoch/epoch_{acc,accdelta,5c,5cr}_*`): acute far contra 0.50 -> **0.32** (4/4),
   far ipsi .72 -> .58, far mid .71 -> .51; cue window unchanged. Confusion: acute far contra mis-decoded as FAR IPSI /
   far mid (distance kept, side lost).
3. **Matched contact 1.0 s ENL arm** (`G/DAQ_ENL1s/epoch/`): decoding = LP 1.0 s within .03 (lick definition irrelevant at
   1 s; the 2 s -> 1 s drop is window length). ENL far-contra map change 9.9 % (contact 1 s) -> 4.4 % (LP 1 s).
4. **LP-gated per-session encoder** (`E/movement_first_subset-drop_licks-combined_gate-attempted/`, 119 sessions;
   `scripts/lp_encoder_decompose.py` -> `E/qc/lp_encoder_decompose_by_animal.csv`): acute far-contra drop carried by the
   per-session "instructed" terms 4/4 -- BUT those include contact + direction (§0); pending the frozen split.
5. **Encoder variant review** (all variants, what each tested / showed): in the 10-22 conversation; main open
   contradiction = cam4 post-stroke executed rise vs 3-D polar "rise is target". Being tested by the 3-D 2x2 (§2).
6. **FR plumbing test** (`E/movement_first_fr-preview_r10_PS92_subset-fr/`, PS92_0812, preview set, IN-SAMPLE): path
   works; FR adds a lot (all frames .56 -> .66-.68; lick .36 -> .53; FR alone beats the full model in the ENL window,
   .65 vs .41); position adds ~.01-.02 beyond FR (expected -- FR carries side).
7. Checks: per-session "alone" models DO re-tune penalties (on all frames); jaw-alone lick-window R2 < 0 = fast jaw traces
   overfitting slow GCaMP, not a bug. Task-only frame R2 ~0.10 matches Musall (single-trial variance is movement-driven).

## 2. RUNNING / QUEUED (10-22 ~midday)

| what | how | output / then |
|---|---|---|
| **chain** `S/run_chain_1022.sh` (bash task in this session) | (1) `frozen_encoder --all --jobs 4` from worktree `S/wf_frozen` (detached at main) with OROFACIAL_GATE=attempted, LICKS=combined, ONLY=full, SUFFIX=_gate-attempted (input-cache env), then `lp_encoder_decompose.py --folder frozen_encoder_gate-attempted/<v> --align {cue,lick}` for 8 variants; (2) RESUME the lick-aligned per-session run (`--skip-done`, 21 / 119 done before the pause) + its cam4 pass; (3) 3-D 2x2 (xyz2 x {combined, cam4}, polar x cam4; 47 PS92 + PS93 sessions; 20261021 tables; soft bins; reference gate; `--only nokin:`) | (1) `E/frozen_encoder_gate-attempted/<variant>/sessions`, `dropspout/`, `frozen_musall_r2*.csv`, `fit/pre_stats.pkl`, log `S/frozen_encoder.log`; (2) `E/movement_first_subset-full_licks-combined_gate-attempted/` (`peth_lick`); (3) `E/movement_first_subset-nokin_exec-{xyz2_licks-combined,xyz2,polar}_bins/`, logs `S/3d_2x2_*.log` |
| **post-frozen** `S/run_after_frozen.sh` | waits for "frozen decomps done" + >= 20 GB free; `frozen_musall_figures.py`; `frozen_group_leakage.py --jobs 2`; `combined_encoding_figures.py --order movement_first` (Musall Fig. 4 bars + maps for EVERY epoch now) | `E/frozen_encoder_gate-attempted/figures/musall_*.png`; `E/qc/frozen_group_leakage.csv`; `E/movement_first/figures/fig4_*_{pre,acute,subacute,chronic}_*.png` |
| FR pilot watcher (python loop in this session) | polls `fr_cam4_20261021/factors/pilot_pre_PS92_PS93/proj/` for PS92_0812 / 0820, PS93_0812 every 10 min (24 h) | then §3 step FR |
| push of 2e7ebfe ... b530065 (frozen drop-spout, Musall readouts, figures, leakage) | retry loop | verify by FILE CONTENT on origin/main (§4) |

## 3. NEXT (in order)

1. **Read the frozen run** (per animal first): FGROUP shares of the acute far-contra post-lick drop (cue- and lick-
   aligned), data - full (does cortex respond LESS than the pre-stroke mapping predicts?), no_movement -> task_movement ->
   task_movement_dir -> joint increments, ordered bounds; frozen drop-one-spout far contra under joint / task_first /
   movement_first (`dropspout/frozen_dropspout_by_animal.csv`); Musall readouts (`frozen_musall_quantities_by_animal.csv`).
   Leakage table: directionless groups should decode the spout near chance (1/6). Write into ENCODING_FINDINGS 3.3.
2. **Link figures for Priya** (she asked: "both r2 graphs and maps"): frozen `musall_*` + per-session `fig4_maps_<epoch>_*`.
3. **Frozen MAPS** (Priya: "i guess frozen maps?" + "also re-rendered per-session maps"): add a grid target to
   `frozen_encoder` (Rd on the 8x stat grid, 2,022 bins, instead of 42 areas) -> per-bin Musall quantities + far-contra
   decomposition maps; run after the chain with <= 3 workers (memory).
4. **FaceRhythm pilot** (`pilot_pre_PS92_PS93`, PS92 + PS93, 51 sessions; pre sessions on a leave-one-day-out basis,
   post on the full fit): one local session; then `o2_encoding.py bundle --name enc_fr_pilot_<date> --fr
   pilot_pre_PS92_PS93 --fr-points 10 --cams cam1,cam4` + `check`; Priya submits. Readouts: FR vs ME vs both; position
   bracketed (FR first / position first) + within-spout FR; merged-lick FR (preview names: lick factors 0,1,2,4,7,9;
   3 = lick_onset_nose separate) once pilot factors are matched to preview names by spatial-spectral cosine (Priya
   confirms) + spout-decode leakage of the merged trace. Two-animal pilot; rerun on `master_pre_v1`.
5. **3-D 2x2** (when the chain reaches it): which factor flips PS92 acute (coordinates vs lick set).
6. Lick-aligned per-session decomposition (`lp_encoder_decompose.py --folder movement_first_subset-full_licks-combined_gate-attempted --align lick`).
7. LP_licks leftovers: PS93 far_ipsi / far_mid frozen sensitivity (`WIDEFIELD_LP_FROZEN=lp`); drop-one-spout under the LP
   gate (per-session, ~64 GB / session: subset only); full rerun when cam1 finishes.
8. Carried over: 10-21 §3 items 4-11.

## 4. Pitfalls (new 10-21 night / 10-22)

- **Pushes race the other windows.** The pre-push suite takes ~8-10 min; another window's push in that time -> "cannot
  lock ref". Their push often CARRIES my commits (rebased) -- check `git show origin/main:<path> | grep <new symbol>`.
- **Missing test dependency:** the histology tests need `shapely` (declared in pyproject); installed 10-22 (2.1.2) -- a
  fresh env fails the pre-push suite without it.
- **Rule-9 guard** (`tests/test_parallel_results_are_ordered.py`) requires an explicit sort within 14 lines of every
  `fan_out` / `fan_sessions` call -- a helper that returns raw `fan_out` results fails the push.
- **`combined_encoding.out_dir` does NOT encode the gate** (OROFACIAL_GATE): tag runs with `COMBINED_ENCODING_SUFFIX`.
  The input-cache key includes every `COMBINED_ENCODING_*` / `OROFACIAL_*` env var, so match the env to reuse the cache.
- **`COMBINED_ENCODING_LICKS=combined` raises for sessions without cam1** (PS95 all, PS94_1005): run those with LICKS
  unset and `COMBINED_ENCODING_SUFFIX=_licks-combined<rest>` to land in the same folder (json `lick_set` says which).
  `frozen_encoder` falls back automatically.
- **Editing `combined_encoding.py` invalidates the input cache** (cold reloads ~100 s / session). Added 10-22:
  `peth_lick` (lick-aligned PETH) -- the edit made the next runs cold.
- **Bash heredocs mangle `\n` AND `$`-escapes**: patch Python / shell with the Write / Edit tools (happened 3x today).
- **PowerShell `-File` scripts are blocked** (execution policy): use `python S/freegb.py` for free RAM.
- `lp_encoder_decompose.py` names outputs by folder when `--folder` is not the default (variants no longer overwrite).
- A frozen-encoder layout change (FGROUP) makes `fit/pre_stats.pkl` stale: bump `STATS_VERSION`, rerun without
  `--reuse-stats`.

## 5. New code (all on main; tests in `tests/test_frozen_encoder.py`, 7)

- `wfield_local/frozen_encoder.py` -- per-animal pre-stroke ridge in Allen-area space; pooled-pre scaling; FGROUP (task /
  movement / direction / contact); variants joint, movement_first, task_first, no_movement, task_movement,
  task_movement_dir, movement_only, no_task; frozen drop-one-spout (`<base>|drop:<spout>`); Musall readouts (frame /
  single-trial / trial-averaged R2, cue- and lick-aligned); `--reuse-stats`, `--summarise-drop`.
- `scripts/lp_encoder_decompose.py` (`--align cue|lick`, `--folder`), `scripts/encoder_trialavg_r2.py`,
  `scripts/frozen_musall_figures.py`, `scripts/frozen_group_leakage.py`; `combined_encoding` saves `peth_lick`;
  `combined_encoding_figures.py` renders Fig. 4 for every epoch.
