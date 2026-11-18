# HANDOFF 2026-11-16 (window a2): encoding-model design review, TIM, behavioural states, block permutation

START HERE for this arm. Design / decisions: `docs/status/ENCODING_DESIGN_TIM_STATES_2026-11-16.md`.
Block-permutation audit: `docs/status/BLOCK_PERMUTATION_AUDIT_2026-11-16.md`. The encoding model's own state
(production runs, pages, deck) stays in `STATUS_2026-10-26_ENCODING_HANDOFF.md` (block 00-COMPACT-41).

## 0. Where the work is (both branches PUSHED to origin 11-17, NOT merged into main; off origin/main 20839101 / b70f6a44; listed in docs/START_HERE.md section 1 J on main)

| branch | worktree | holds |
|---|---|---|
| `analysis/tim-encdesign-1116` | `C:/Users/SabatiniLab/Github/wf_encdesign` | TIM (`wfield_local/task_independent_movement.py`), GLM-HMM (`glm_hmm.py`), AR-HMM (`arhmm_features.py`, `arhmm_model.py`, `arhmm_states.py`, `scripts/o2_arhmm.py`), docs (this file, design doc, audit + measurement CSV), tests `test_glm_hmm.py` |
| `fix/block-perm-null-1116` | `C:/Users/SabatiniLab/Github/wf_blockperm` | `analysis_kit.block_permute` / `pooled_block_ids`, `nolick_analysis.permutation_null(blocks=)`, callers wired, `tests/test_block_permute.py` (213 related tests pass) |

Run code from a worktree with `PYTHONPATH=$(pwd -W)` (Windows form). AR-HMM model steps run in the NEW conda env
`arhmm` (`C:/Users/SabatiniLab/miniconda3/envs/arhmm`: CPU JAX 0.10.2 + jax-moseq 0.3.9, OpenBLAS).

## 1. Decisions (Priya, 11-16)

- Instructed / uninstructed = a LABEL by CONTEXT, not body part (pre-cue + ITI tongue / jaw / contacts / FR lick
  factors -> uninstructed); task-aligned / task-independent is MEASURED. Not built yet (context-gated TE design).
- Block permutation everywhere it makes sense; change the code (measure before / after in the same run).
- TIM: include the cued response period (scored separately); within-session (refit) AND frozen-pre frames.
- GLM-HMM: states from ALL trials; not only the response period (trial CYCLES, several observations: running, cued vs
  spontaneous licking); contact + incomplete merged as ATTEMPT (4-class = sensitivity); choose K by held-out
  explained variance. After the pooled-pre fit turned out to encode animal identity: fit PER ANIMAL over ALL epochs.
- AR-HMM before the richer GLM-HMM: port Hillary Tong's model (github.com/htong1/Priya_ARHMM2026) and CHANGE its
  decisions for this rig: 25 Hz; the ENCODER'S behaviour variables (3-D tongue, cam4 jaw / nose, session_speed; "don't
  reinvent the wheel") + the existing FR projections (5 Hz between trials is enough: r >= 0.99 vs 25 Hz); lateral
  columns contralesional-positive; running floor 0.3 mm/s both directions; fixed pooled scaling (never per session);
  POOLED over animals, ALL epochs (frozen-pre fit kept for the log-likelihood recovery score); run on O2.
- States: QC + merge as Hillary did (cosine on independent representations + video gate).

## 2. Results so far (all PROVISIONAL)

- **Block null (measured, `docs/status/BLOCK_PERMUTATION_MEASURE_2026-11-16.csv`)**: per-session decoders (108 sessions
  x precue / cue, `nolick_decoder.analyse_session`): engaged arm unchanged at cue (108/108), precue loses 2/107 (weak
  acute PS92 0822 / 0824); no-lick / undetected / late arms: 15 cells lose and 10 gain p < 0.05, all near 0.05; null
  upper bound widens 0.044 (cue) / 0.017 (precue). 0 mixed blocks. Pooled / frozen arms NOT measured yet.
- **TIM** (all 119 sessions, both frames; tables on the share `LP_analysis/<run>/task_independent_movement/v2/sessions`):
  within-session TIM does NOT rise post-stroke (tongue falls acutely 3/4: less spontaneous licking; PS95 rises), unlike
  Yin et al.; the frozen-frame nose rise (4/4 acute) is a contralesional nose POSTURE shift (-2.1 / -0.8 / -5.1 / -4.5
  px), not variability. Fits the quiet-events picture (stiller acute state).
- **GLM-HMM** (pooled August pre, K = 1..5; share `LP_analysis/<run>/glm_hmm/v1/select_k.csv`): pseudo-R2 .313 / .386 /
  .419 / .431 / .437; BUT states sorted by ANIMAL (PS93 67 % S0, PS94 56 % S1, PS95 51 % S3) at the same mean session
  position -> pre animals barely disengage; superseded by the per-animal all-epoch plan.
- **AR-HMM**: features for 120 sessions being extracted; pipeline smoke-tested end to end (fit -> decode -> similarity
  -> profile); NO meaningful state result yet (smoke fits = 2-3 iterations).

- **GLM-HMM per animal, all epochs (11-17, `scratchpad/glmhmm_v1`)**: held-out pseudo-R2 by K (PS92-95) K1 .25/.23/.20/.19,
  K3 .39/.41/.38/.37, K5 .44/.44/.41/.42; outcome + lick terms saturate at K = 3, extra states model running. BUT the
  per-animal K = 3 states are NOT comparable across animals: the quit tail (trials after the last camera lick) sits in
  state 1 for PS92 / PS93 (.83 / .75) and in state 0 for PS94 / PS95 (.71 / .58); "state 2 = disengaged" holds only
  for PS94 / PS95 (state 2 occupancy acute .36 / .39). NB PS95 is the LEAST severe animal, so that acute rise is not a
  severity ordering. Quit tails are short (acute median 19 / 21.5 / 27 / 2 trials PS92-95). -> pooled all-epoch fit
  (`glm_hmm --pooled-all --select-k`: shared emissions, per-animal bias inputs) running 11-17.
- **GLM-HMM POOLED, all animals, all August-onward epochs (11-17, `scratchpad/glmhmm_v1/*pooled_all*`; report script
  `scratchpad/glmhmm_pooled_report.py`)**: held-out pseudo-R2 total K1-5 .281 / .381 / .427 / .443 / .459; outcome
  .484 / .540 / .599 / .605 / .619 (saturates at K = 3; K >= 4 mostly running). K = 3: 82 % of trials assigned. ONE
  state is shared by profile across all four animals: state 2 "disengaged" (attempt .16-.42, no response .31-.47,
  almost no ITI licking, running .20-.44, mean session position .76). Its occupancy pre / acute / subacute / chronic:
  PS92 .02 / .23 / .05 / .00, PS93 .03 / .12 / .08 / .02, PS94 .09 / .32 / .18 / .09, PS95 .08 / .05 / .24 / .06.
  States 0 / 1 (both engaged) still differ by animal (e.g. ITI licking in state 0: PS95 .05 vs PS93 .87). Quit tail:
  only 30-60 % in state 2, the rest in state 0 (still, not licking, not running); most sessions have NO tail (median
  tail 0 / 0 / 9 / 3 trials PS92-95). K = 4 states are NOT consistent across animals -> K = 3. Two design causes,
  to test next: (1) the animal bias columns are STATE-SPECIFIC weights, so states can absorb animal identity -- make
  the animal offsets shared across states; (2) previous-outcome inputs let a no-response streak be explained inside
  an engaged state -- fit without them so disengagement must be carried by the state.
- **GLM-HMM DESIGN v2 (11-17, Priya "fix the design issues"; `glm_hmm --design v2`, default): animal offsets SHARED
  across states, no previous-outcome input.** Pooled, all animals, August-onward, K = 3 (86 % assigned): all three
  states now mean the same thing in every animal -- S0 engaged + frequent spontaneous licking (attempt 1.00, ITI lick
  .60-.94), S1 engaged with little spontaneous licking (attempt .93-.98, ITI lick .06-.57), S2 DISENGAGED (attempt
  .05-.15, no response .51-.59, session position .81). QUIT TAIL: 99-100 % of tail trials in S2 in all four animals
  (v1: 30-60 %). S2 occupancy pre / acute / subacute / chronic: PS92 .01 / .31 / .04 / .00, PS93 .02 / .32 / .15 /
  .02, PS94 .09 / .42 / .20 / .07, PS95 .11 / .11 / .35 / .12 (PS95 peaks subacute, matching its raw no-response
  count: 894 subacute vs 57 acute). K = 4 adds a fourth state that is ALSO consistent across animals: engaged while
  running (running .28-.32 in every animal). Held-out K selection for v2 running (`select_k_pooled_all_v2.csv`). K = 5 (74 % assigned;
  PS95 acute 50 % unassigned) is ALSO consistent across animals: adds a PARTIALLY ENGAGED state (attempt .39-.56, no
  response .18-.23, session position .72) that rises acutely in PS94 (.17) and PS95 (.18) vs <= .04 pre; quit tail
  89-98 % in the disengaged state. Decision rule: K = 5 only if its held-out gain over K = 4 exceeds the fold spread,
  else K = 4 with the partial-engagement state as a sensitivity result. HELD-OUT K SELECTION v2 DONE (`select_k_pooled_all_v2.csv`,
  4 interleaved session folds): total pseudo-R2 K1-5 .108 / .340 / .396 / .444 / .470; outcome .029 / .604 / .617 /
  .617 / .630 (engagement captured by K = 2-3). K3 -> 4 gains in EVERY fold (+.046-.051, almost all running); K4 -> 5
  only in two of four folds (+.042 / .041 vs +.013 / .008). RECOMMENDED K = 4 (proposed to Priya 11-17); K = 5 partial
  engagement = sensitivity. Disengaged occupancy is the same at K = 3 and 4, so the engagement readout does not
  depend on the choice. **DECIDED (Priya 11-17): K = 5** ("we can always combine states post-hoc"); tables
  K3 / K4 / K5 (design v2) + `select_k_pooled_all_v2.csv` copied to the share `LP_analysis/<run>/glm_hmm/v1/`. Priya
  also asked where the gains plateau: K = 6-8 on the same folds running (`--ks 6,7,8` ->
  `select_k_pooled_all_v2_K6-7-8.csv`); read per output (outcome flat from K = 3) + cross-animal consistency of any
  new state, not total pseudo-R2 alone.

- **AR-HMM kappa / noise floor (11-17)**: O2 scan (chunked, no dither) gave median 0.08-0.32 s for kappa 1e5-1e11 --
  states predicted the smooth 25 Hz PCs to ~0.007 (min log|Q| -109), so every change out-bid the stickiness (FR is 5 Hz
  interpolated: ~half its frames are exact ramps; Priya's June repo ran 12.5 Hz with per-bin tongue summaries). Fix:
  PC-space dither (`DITHER` 0.1). Scored on 6 sessions against DAQ labels (NMI, Priya's repo metric;
  `scratchpad/arhmm_score.py`): side-within-contact .08 (no dither) -> .36 (dither 0.1, kappa 1e7, median 0.44 s);
  contact / running unchanged. Fits submitted at kappa 5e7 (5x the scan for 31 vs 6 sessions), bundle code ca84b6e1.

- **AR-HMM vs sticky Gaussian HMM (11-17, Priya "is AR-HMM better than standard HMM for us?")**: same 6 sessions /
  features / dither 0.1, Gaussian HMM by MAP-EM (`scratchpad/ghmm_compare.py`), matched at median 0.44 s (GHMM kappa
  1e6 vs AR kappa 1e7). NMI vs DAQ labels, GHMM / AR: contact .113 / .154, contact side .127 / .199, running .072 /
  .091, side within contacts .170 / .362. AR-HMM better on every label (one seed; engines differ: EM vs Gibbs; GHMM
  on instantaneous features, not per-bin summaries). Keep the AR-HMM.

## 3b. 11-17 night: STATES RESULTS PAGE + one module for every state readout -- START HERE for the states arm

- **Page:** https://claude.ai/artifact/ChbBM9K12WWRohx6nU6rCX (source `docs/results_pages/states/`, README = rebuild
  steps). Sections: 1 GLM-HMM engagement, 2 treadmill push / walking, 3 AR-HMM syllables, 4 TIM (figures pending),
  5 frozen pre-stroke likelihood (scoring running), 6 methods. FINDINGS section 9 (main).
- **Code (Priya: modular, no redefinitions):** `wfield_local/behavior_states.py` holds every choice once (GLM-HMM K /
  design, roles FROM PROFILES via `glmhmm_roles`, groups from the reviewed `state_labels.csv` via `state_groups`,
  epochs = `epochs.epoch_of` + `config.early_pre`, walking = segmentation.running, push = NEW `segmentation.push` in
  configs/defaults.yaml, lick gate = `trial_classes.camera_engaged_period`, stats = `orofacial_cohort.epoch_tables`).
  `python -m wfield_local.behavior_states collect` -> `arhmm/v1/poststroke_pooled_all_s1/` (csv.gz + traces.npz);
  `python scripts/states_page_figs.py [--only ...]` -> `page_figs/`. Test `tests/test_behavior_states.py`. The scratch
  scripts (scripts/states_poststroke, share `arhmm/v1/poststroke_s1/`) are SUPERSEDED (folder kept; derived, can go).
- **Data on the share now:** features `arhmm/features_v3` (= `arhmm_features.out_dir()`), fits + decoded states
  `arhmm/v1/fits` (`arhmm_states.fits_dir()`, PROVENANCE.txt: pooled_all_s1 = local refit = the REVIEWED reference;
  s2 / s3 / pre_s1-3 = O2; pre_s1_f1..f8 = O2 held-out folds).
- **Frozen likelihood:** `arhmm_model score --tag pre_s1_f<k>` (marginal LL per frame, jax-moseq
  `marginal_log_likelihood`) running for all 8 folds x 107 sessions -> `fits/ll_pre_s1_f<k>.csv`; combined by
  `behavior_states.frozen_distance` (within-model anchor, pre days only where held out); figure family `frozen`.
- state_labels.csv groups updated: 21 / 22 -> jaw_chew, 4 -> walk_slow (notes in the rows).
- **Frozen likelihood RESULT (page 5.1, v2):** change vs own held-out pre, nats / frame: acute +.52 [.26, .83] (4/4),
  subacute -.21 (n.s.), chronic -.66 [-1.07, -.23] (4/4; PS92 only -.02). Acute "more pre-like" = stiller (quiet
  frames are predictable) -> NEXT: the score on movement frames only / adjusted for the quiet share (per-frame
  forward-filter normalizers), before calling it a distance from the pre-stroke repertoire.

## 3c. 11-18: stillness control, frozen-model groups, TIM figures, K 6-8 (page v4)

- **Frozen likelihood, stillness control** (`arhmm_model score --per-frame` -> `fits/llframe_<model>/`, verified: frames
  sum to the marginal; `behavior_states.frozen_distance_frames`): movement frames only: acute -.31 (3/4, n.s.),
  subacute -.42 (n.s.), chronic -.61 [-.80, -.39] (4/4). By group, chronic: ipsi licks -.78, contra -.43, quiet -.11,
  slow walk -.15 (all 4/4); push +.93 (4/4, MORE stereotyped than pre); acute chewing -.75. The +.51 acute all-frame
  rise = stillness. Page 5.1 / 5.2.
- **Frozen pre_s1 model combined into the reviewed groups** (`behavior_states.group_stats`; Priya 11-18: merges apply
  as groups): quiet +.10 acute; contra licks -.03 acute / -.01 subacute, bouts -0.10 s subacute and chronic; ipsi
  +.025 / central +.010 / push +.014 chronic; no new states. Page 5.3-5.5 (per-state version superseded).
- **TIM** (`behavior_states.tim_trials`, page 4.1 / 4.2): no rise; acute response tongue -3.8 px, ITI jaw -2.1 px;
  pre-cue nose TIM higher when not engaged (Yin effect, 4/4 pre, +2.2..4.3 px), unchanged after the stroke.
- **GLM-HMM K 6-8:** total pseudo-R2 .491 / .494 / .499 (K5 .470); 5 -> 6 in 2/4 folds, then ~0; outcome / lick flat
  from K 3; extra states = running. K = 5 stands.
- Open: LP direction cross-check (other window); contra lick bout length vs the kinematics page; merge of the
  branches into main.

## 3a. 11-17 ~18:00: held-out pre baseline + pre seed stability

- **Pre-model seed stability** (`arhmm_seed_match_noQ.py`, ref pre_s1): pre_s2 frame agreement .823 / ARI .847; pre_s3
  .642 / .555. The quiet face is ONE pre state (45 % of time; cosine .97-1.00, frames .98 / .70), so no 5-19-8-style
  split. Weak: s2 18, 22; s3 8, 16, 9. Use **pre_s1** as the frozen reference, s2 / s3 as robustness.
- **Held-out pre folds (Priya OK 11-17):** scoring pre days with the in-sample pre model would make the baseline fit
  look better than any post day. `arhmm_model fit --holdout-fold k` drops each animal's k-th August-pre day (8 folds
  cover all 31; PS93 has 7 so sits out fold 8); tag `pre_s1_f<k>`, model stores `held_out`. Post days are scored by the
  full pre_s1. Smoke-tested the exact command locally (fit 2 iters -> save -> load in locanmf -> decode a held-out day).
  Bundle **`arhmm_20261117`** (code f1822fc4; features + epochs byte-identical to 20261116b), `foldfit.sbatch`
  (array 1-8, 16 cores, 96 GB, ~2 h each). Then: decode each fold's held-out days locally, per-frame log-likelihood
  score vs own held-out pre.
- **Merges deferred** (Priya 11-17: "we can merge later"); 5-19-8 still the only proposed merge.
- GLM-HMM K 6-8 select-k still running (PID 30040, serial, writes only at the end; expected ~19:30-20:30).

- **Priya's reel review 11-17 (state_labels.csv in reels_pooled_all_s1):** groups quiet 5-8-19 (merge OK); nose_slow 1;
  lick ipsi 0-7-9-23 (0 = bout start); lick central 10-12-18; lick contra 2-15-20; jaw without tongue 21-22 (check cam1);
  push 13 (+ rare 11); walk 6; 4 open (walk or push). Data check (`scratchpad/arhmm_state_check.py`, every 2nd session):
  4 = forward wheel 4.7 mm/s, no backward, onsets not cue-locked (5.6 % in 0-1 s) -> slow walking; 13 = backward wheel
  but NOT cue-locked (9.9 %, median 7.2 s after cue); the cue-locked push is **11** (65 % of onsets 0-1 s after cue,
  0.48 % usage, below the reel cut); 0, 7, 9, 20 onsets cue-locked (42-58 %); 1 not (2.9 %). Offered: reels with a
  wheel trace (4, 6, 11, 13) + cam1 clips for 21 / 22.

- **First post-stroke state readouts (11-17, `scratchpad/poststroke_states.py`, delta vs own August pre, boot_delta):**
  GLM-HMM K5: disengaged (state 4, no-response .60) +.17 acute (3/4) / +.10 subacute (4/4, Holm); partial (state 3)
  +.11 acute (4/4, Holm); chronic back to pre. AR-HMM whole-session group share: quiet +.13 acute (4/4); contra lick
  -.033 acute / -.013 subacute (4/4); central -.011 acute; "jaw no tongue" (21 + 22) -.07 acute; chronic ipsi lick
  +.019 and push +.016 (4/4). Caveats: whole session (disengagement inflates quiet), lick groups mix amount and
  direction -> next: engaged trials only + lick group as share of lick frames.
- **Jaw-only trials vs states 21 / 22 (Priya surprised 21 + 22 FALL; `jaw_only_states.py`, `jaw_onset_states.py`):**
  21 / 22 are peri-lick chewing (69 % of 21 frames within 0.5 s of a camera lick; 22 40-67 %), so they fall with
  licking. Jaw-only trials (531 pre -> 1387 acute) get NO syllable of their own: at the jaw onset (+-0.15 s) the states
  are 1 (slow nose, .25-.32), 5 (quiet), 4 / 6 (walking, ~.40 together), 21 only .04-.05, the same pre and post. So
  state 1 = Priya's "failed attempt / jaw only" read; the trial-class jaw call stays the jaw-only measure. Flag:
  ~40 % of jaw-only onsets fall in walking states (run-onset jaw movement, FR factor fr_run_onset_jaw?), pre and post.

- **Post-stroke round 2 (Priya OK items 1, 2, 3, 6; dropped 4 = LP lick direction per spout already does it, 5 =
  per-lick kinematics already do it; `poststroke_states2.py`; outputs + scripts copied to share
  `LP_analysis/<run>/arhmm/v1/poststroke_s1/`).** Engaged = GLM-HMM K5 argmax in {0,1,2}; trial cycle = prev stop ..
  stop. (1) engaged cycles: quiet +.11 acute (4/4, Holm) -> not only disengagement; contra lick -.021 acute; ipsi
  +.027 acute, +.020 chronic; push +.017 chronic (4/4, Holm). (2) contra share of LICK frames .44 pre: -.12 acute,
  -.07 subacute, -.07 chronic (4/4 each, Holm) -- chronic deficit persists; CROSS-CHECK against LP per-lick direction
  before using. (3) Fig 1 cue-aligned groups (engaged): acute contra / central lick peaks lower, ipsi peak higher,
  quiet stays up after the cue, slow-nose stays up (failed attempts); PRE-CUE jaw chewing 0.25 -> 0.05 acute (above
  pre chronically); cue-locked push peak 0.19 pre -> 0.42 chronic. (6) transitions (exploratory, ~140 tests, Holm
  floor .036): entropy -0.20 bits acute (4/4); chronic push -> contra lick -.11, quiet -> contra -.06, walk -> central
  -.17, walk -> ipsi +.16; fewer lick -> chew.

- **LP direction cross-check (Priya OK 11-17; another window is working on the LP direction analyses, so NOT run
  here):** is the AR-HMM chronic contra lick-share drop (-.07, 4/4) (a) aim -- tongue lateral per lick WITHIN spout
  shifted -- or (b) distribution -- fewer licks per trial at contra spouts? Compare with that window's per-spout
  results when they land. Fig 1 restyled: repo EPOCH_LINE colors, +-SEM band across animals (alpha .07, edges .25).

- **Peri-cue push / walking from the treadmill alone (Priya 11-17; `cue_push.py`, figs 2a / 2b on the share):** push =
  backward wheel > 0.5 mm/s in [-0.25, +0.75) s of the cue (control [-2, -1) s); walking = forward > 3 mm/s. Push is a
  TASK behavior: ~0 outside the lick working period in every epoch. Working period: push rate .15 pre, -.11 acute
  (4/4, Holm), +.25 chronic (4/4, Holm); control-window rate tiny (.01). Peak backward speed on push trials rises
  +.22 subacute / +.38 chronic, latency earlier (-.07 / -.10 s). Cue-window walking unchanged (only GLM-engaged
  chronic -.014). Gate comparison: GLM-HMM engaged (states 0-2) is a strict SUBSET of the lick working period
  (GLM-only = 0); in-period but GLM-not-engaged = 6 % pre, 33 % acute, 21 % subacute, 5 % chronic. Push results are the
  same under either gate. Backward speed plotted negative (Priya).
- **GLM-HMM per-state response curves by spout (fig 3, `glmhmm_state_curves.py`; weights not saved -> observed
  rates weighted by state posterior, per animal):** states 0-2 contact ~1 close, .8-.95 far; 0 vs 1 differ by pre-cue
  / ITI licking (0 highest); 2 = running, pre-cue licking drops at far spouts; 3 partial = contact .6-.7 close but
  ~.2 far with jaw-only ~.4 at far spouts (a far-spout failure state); 4 = no response ~.5 + jaw-only ~.4.
  Caveat (Priya 11-17): jaw-only is 0 at nM because neither camera can judge the jaw there (cam4 chin hidden by the
  spout; cam1 failed review; `trial_classes` JAW_UNJUDGED_POSITIONS) -> no-lick nM trials default to no response
  (jaw_judged False); state 4's no-response peak at nM is the same artifact. Fig 3 caption says so; spout codes
  nI .. fC only (Priya: one convention).
  Not "psychometric" (spout is not a graded stimulus strength).

- **GLM-HMM state curves by epoch (fig 3b, `glmhmm_state_curves_epoch.py`):** ENGAGEMENT and EXECUTION separate.
  In the engaged states 0-2, acute far-spout trials are still attempted but MISSED: contact at fC .05-.13 acute vs
  .94-.98 pre, incomplete .76-.82 acute (fM: contact .48-.59, incomplete .32-.39); jaw-only / no response stay ~0;
  subacute partly recovered (fC contact .71-.83), chronic = pre. Disengaged state 4 occupancy rises acutely at EVERY
  spout (~.2 vs .04 pre; subacute ~.14) -> spout-independent; partial state 3 rises acutely mainly at far spouts
  (fM .23, fC .28 vs ~.03 pre) -> spout-specific. The emissions are shared across epochs, so the engaged states'
  acute far-spout outcome is a misfit the model absorbs (state = engagement, outcome = execution) -- fine for an
  engagement gate, but per-state outcome rates are not stationary.

## 3. Running / state at 11-17 ~14:00 (background in session 37fdbabf; scratchpad
`C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/37fdbabf-285a-455b-bbb7-d5927a5affe6/scratchpad`)

- **O2 AR-HMM fits RESUBMITTED** (bundle `arhmm_20261116b`, code e4253857): pooled_all x seeds 1-3 (PER_CELL 2) + pre x
  seeds 1-3, kappa 5e7, dither 0.1, 100 iterations, `-p short -t 3h -c 8 --mem=12G` (measured MaxRSS 5.1-5.6 GB,
  0.9-1.9 h). The FIRST submission (11-17 ~11:20) ran all 100 iterations and then crashed at the pickle save
  (`Design.__module__` relabel under `python -m`; 0-byte models) -- fixed: design saved as plain data
  (`Design.to_dict` / `from_dict`), atomic write; verified by a local end-to-end smoke test with the exact command
  form (fit -> save -> decode -> load in the analysis env; `scratchpad/arhmm_e2e`). When done: Priya pastes the `done`
  lines + `ls out/`; then decode (`decode.sbatch`, TAG = pooled_all_s1 and pre_s1; COMMANDS.md step 4), then copy back.
- **AR-HMM merge-QC reels (11-17)**: `wfield_local/arhmm_reels.py` -> share
  `LP_analysis/lp_r4_heatmap_20261013__clean_v36/arhmm/v1/reels_pooled_all_s1/` (REVIEW.md, state_XX.avi, contact
  sheet; 8 syllables per state, all four animals; split-day PS92 0812 via `camera_sync.templates_for`). WAITING on
  Priya's video decisions (candidates 5-19-8, 1, 0-13, 10-23; state 22 = lick rhythm without tongue?). Priya 11-17:
  state 0 = LEFT-LICK BOUT INITIATION (kept apart from 13; `state_labels.csv` in the reels folder). SEED STABILITY
  (`scratchpad/arhmm_seed_match.py`, every 3rd session, states matched by mean feature profile): frame agreement with
  seed 1 = .66 (seed 2) / .49 (seed 3); with the quiet / face cluster 5-19-8 merged .82 / .80 (ARI .85 / .80) -- the
  quiet cluster (~half the frames) splits differently in every seed (seed 2 merges 5 + 8), so MERGE 5-19-8 is
  proposed (cosine 5 / 5 + seed-unstable). Stable in both seeds (frames agree >= .7): 13 push, 7 ipsi lick, 12 central
  lick, 20 contra lick, 6 running, 22 lick rhythm without tongue, 18 ipsi lick, (1 face .77 / .68). Provisional:
  21 pre-lick jaw (.25 / .42), lick states 2 / 9 / 15 / 23 and slow running 4 (one seed only), 0 (seed 2 .88, seed 3
  no partner). O2 models
  copied back so far: pooled_all s1-s3, pre s2 (pre s1 / s3 pending); decode them LOCALLY (~6 min for 107 sessions),
  not on O2.
- **Local hedge fit**: pooled_all seed 1 (same sessions as O2 task 1) -> `scratchpad/arhmm_local_fit/` (~31 s / iter,
  ~52 min); start state QC on whichever copy finishes first.
- **GLM-HMM design v2**: held-out K selection K = 1-5 (`glm_hmm --pooled-all --select-k --design v2 --k 3`) ->
  `scratchpad/glmhmm_v1/select_k_pooled_all_v2.csv` (~3 h from 13:06); K = 5 profile fit -> `states_pooled_all_K5_v2`.
  Report script: `scratchpad/glmhmm_pooled_report_v2.py`. Sequences cached (`glmhmm_v1/sequences_v1_*.pkl`).

## 4. To do (in order)

1. GLM-HMM K: choose K = 3 / 4 / 5 by held-out gain beyond the fold spread + the extra state consistent across animals
   + the quit tail staying in one state (K = 4's engaged-while-running state passes the last two). Then: states vs
   trial classes / working gate; TIM and quiet-period events by state; copy the chosen tables to the share
   (`LP_analysis/<run>/glm_hmm/v1`).
2. AR-HMM: check the six fits' median syllables (~0.4-0.5 s expected; else adjust kappa), decode, copy back.
3. State QC on the AR-HMM models: `arhmm_states similarity` / `profile`, the DAQ-label NMI scoring
   (`scratchpad/arhmm_score.py`), video reels per candidate state (reuse `behavior_clips._clip`, cam4 via the camera
   templates), Priya's video gate, `merge`, dwell / transitions; frozen-pre per-frame log-likelihood (the
   "manifold distortion" score of Priya's June repo).
4. GLM-HMM + AR-HMM: add syllable usage per trial cycle as an observation (later).
5. TIM follow-ups: split by trial class / state; response-period frozen TIM vs aim offset (sanity); nose shift = real
   posture vs mouth-origin estimate; the four Yin-linked tests + quiet-events tests (design doc section 4).
6. Block permutation, rest: no-lick-arm block ids (`_trial_features` cached -> bump CACHE_VERSION), poststroke /
   frozen no-lick arms, scripts on the audit list (FR arm: window 89 waits; state space: window 3c fixing its own),
   encoder `shuf:position` -> block permutation, movement shuffle -> block-segment permutation; then measure the
   pooled arms and record changed numbers in DECISIONS before merging `fix/block-perm-null-1116`.
7. Context-gated instructed / uninstructed TE design (new design tag) -- after the state work.
8. CONDITIONAL: input-driven AR-HMM ("GLM-AR-HMM": cue / spout / contact driving syllable transitions). Not now (Priya
   11-16 OK): jax-moseq has no inputs; dynamax EM over-segments (Hillary's DECISIONS 06-08). First measure task
   dependence post hoc (syllable probability around cue / first lick / contact, by spout and epoch); build it only if
   that dependence is strong and changes post-stroke, with the post-hoc result as the benchmark.

## 5. Pitfalls found today

- pose3d `t_daq_ms` is CUE-RELATIVE (add the trial's `cue_daq`).
- `face_rhythm_outputs.combined_series` takes `session_keys` (projections), not `source_files` (includes master.npz).
- conda-forge numpy linked to MKL crashed (`0xc06d007f`) in the new env -> OpenBLAS (`libblas=*=*openblas`).
- Scripts run as `python -m` pickle classes under `__main__` -> set `__module__`, load with a remapping unpickler.
- Heredoc-written Python: `\n` inside a quoted heredoc became a real newline once (argparse description) -- use chr(10).
- GLM-HMM forward-backward with per-step logsumexp was the slow path's suspect but the real cost was pytest/conftest +
  CPU contention: measure before optimizing (the scaled recursion is kept; exact to the log-space one).
- Pooled pre-stroke state models encode ANIMAL identity when behaviour barely varies within animals.
- (11-17) `make_data` padded every segment to the LONGEST (a whole session, up to 182k bins): 21-27x the real frames,
  found only after the first O2 scan sat at ~1 h with no output. The 14-min 2-session check fit was the warning sign
  and was extrapolated instead of profiled. Fix: `CHUNK_BINS` 1500 (60 s, overlap NLAGS) -> 1.16x; per-iteration 5.4 ->
  1.25 s on one session. Before ANY O2 submission: print the padded data shape vs real frames and time a few iterations.
- (11-17) Six O2 fits lost at the SAVE: pickling a class from a `python -m` module (relabelled `__module__`) fails with
  "not the same object". Only loading had been tested, and every run that day was a scan (which never saves). Rule:
  before any O2 submission run the exact command end to end on a tiny input (fit -> save -> load -> decode) and check
  the files are non-empty; write outputs atomically.
