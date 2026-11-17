# HANDOFF 2026-11-16 (window a2): encoding-model design review, TIM, behavioural states, block permutation

START HERE for this arm. Design / decisions: `docs/status/ENCODING_DESIGN_TIM_STATES_2026-11-16.md`.
Block-permutation audit: `docs/status/BLOCK_PERMUTATION_AUDIT_2026-11-16.md`. The encoding model's own state
(production runs, pages, deck) stays in `STATUS_2026-10-26_ENCODING_HANDOFF.md` (block 00-COMPACT-41).

## 0. Where the work is (nothing pushed; both branches off origin/main 20839101 / b70f6a44)

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
  running (running .28-.32 in every animal). Held-out K selection for v2 running (`select_k_pooled_all_v2.csv`).

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

## 3. Running at handoff (background in session 37fdbabf; scratchpad
`C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/37fdbabf-285a-455b-bbb7-d5927a5affe6/scratchpad`)

- `arhmm_features --all --jobs 4` -> `scratchpad/arhmm_feat/sessions/` (~64 / 120 at 18:xx; ~50 s per session).
- AR-HMM CHECK FIT: PS92_20260811 + PS92_20260822, kappa 1e8, 40 iters, tag `check_k1e8` -> `scratchpad/arhmm_check/`
  (log run.log). NOTE: started before the pickle fix; its model loads through `arhmm_model.load_model`.

**UPDATE 11-16 evening:** check fit done (states sensible: quiet / running / lick contra + ipsi with matching FR side
factors / pre-lick jaw; problems: median syllable 0.28 s at kappa 1e8, state 1 a running + face catch-all, 9 lick states;
figure `scratchpad/arhmm_check/example_syllables.png`). Found + fixed: cam4 jaw gaps (tongue hides the jaw) cut licks out
of the model -> features v3 fill them from the 3-D jaw (R2 0.90-0.995), min segment 2 s: median 99.8 % of each session
modelled (PS92_0812 70 % = its crash gap; PS94_0817 92 % untriangulated tongue; PS95_0903 / 0907 92 / 90 % jaw).
First bundle `arhmm_20261116` (scan) CANCELLED; current bundle `arhmm_20261116b` (code 9c15dcd1, features
`scratchpad/arhmm_feat_v3`); Priya created the O2 env `arhmm`. Pilot GLM-HMM per animal over all epochs
(`glm_hmm --select-k --per-animal`) running locally -> `scratchpad/glmhmm_v1/select_k_per_animal.csv`.

## 4. To do (in order)

1. Check fit sanity: `python -m wfield_local.arhmm_states profile --out <scratch>/arhmm_check --feat <scratch>/arhmm_feat
   --tag check_k1e8` -> median syllable (target ~0.5 s; hers 0.7 s), states used, lick enrichment, side of lick states,
   running / quiet states. Report to Priya BEFORE the O2 bundle.
2. When extraction finishes: `python scripts/o2_arhmm.py bundle --name arhmm_20261116 --user ps150 --feat
   <scratch>/arhmm_feat` (needs a clean worktree), give Priya the transfer / login commands from its COMMANDS.md
   (env creation once; kappa scan 7 tasks; fits pooled_all + pre x 3 seeds; decode every session).
3. State QC on the O2 models: `arhmm_states similarity` / `profile`, video reels per candidate state (reuse
   `behavior_clips._clip`, cam4 via the camera templates), Priya's video gate, `merge`, dwell / transitions.
4. GLM-HMM per animal over ALL epochs, observations = cued outcome + spontaneous licks + whole-cycle running + (later)
   AR-HMM syllable usage per cycle; K per animal by held-out pseudo-R2.
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
