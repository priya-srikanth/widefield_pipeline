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
