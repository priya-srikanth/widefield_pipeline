# Encoding design review, TIM and behavioural states (2026-11-16, window a2)

Branches: `analysis/tim-encdesign-1116` (worktree `wf_encdesign`; TIM, GLM-HMM, this doc, the block-permutation audit)
and `fix/block-perm-null-1116` (worktree `wf_blockperm`; the shared block null). Nothing pushed yet.
Sources compared: Musall et al. 2019 (and the lab's `ridgeModel` code), Musall et al. 2023, Yin, Melin, ... Churchland
(Neuron 2025; "Spontaneous movements and their relationship to neural activity fluctuate with latent engagement
states").

## 1. Instructed vs uninstructed; task-aligned vs task-independent

- The two axes are different kinds of thing. Instructed / uninstructed is a LABEL we assign to each regressor.
  Task-aligned / task-independent is MEASURED per variable from the fits: task-independent = cvR2(task + variable) -
  cvR2(task alone); task-aligned = cvR2(variable alone) - task-independent.
- Churchland code: every lick after the spouts move in is instructed; earlier licks are deleted; the ITI is never
  modelled (trials cut to -1.8 .. +4.5 s around the grab). Their task never has a possible-but-unrequired lick.
- Ours (production `LICKS_ALL=split`): pre-cue and ITI lick EVENTS are uninstructed, but their tongue / jaw traces and
  all contacts stay instructed over the whole window (documented "accepted caveat"); the frozen encoder's Musall bars
  use a different "instructed" (the lick deviation group only); FaceRhythm lick factors sit in neither group.
- PROPOSED (Priya agreed in principle 11-16; not built): label by CONTEXT, not body part -- instructed = licking in the
  post-cue response period (events, contacts, tongue / jaw traces, FR lick factors); uninstructed = the same in the
  pre-cue (ENL; licking there is forbidden) and ITI periods. Implement as context-gated copies of each lick-related
  column on the TE design (`FROZEN_DESIGN=te`, which already fits ITI frames; 10-26), new design tag.

## 2. Task-independent movement (TIM) -- `wfield_local/task_independent_movement.py`

Yin et al.: body-part positions regressed frame-wise on task variables; TIM = Euclidean residual per trial. Ours:
tongue / jaw (mouth frame) and nose (mouth-relative) from the v36 LP windows, 40 ms bins, three periods (pre-cue /
response / ITI, scored separately; Priya: include the response period, where TIM also carries aim error), ridge per
(period, time bin) on spout, previous spout, previous hit (+ current hit in the response period).

| frame | fit | answers |
|---|---|---|
| within (refit) | per session, block-grouped CV | how stereotyped movement is given the task (the paper's sense); a systematic aim bias is absorbed |
| frozen | animal's August-pre sessions pooled (pre scored leave-one-session-out) | how far movement departs from the pre-stroke task-predicted movement; includes aim offset |

TIM is NOT the aim offset: aim = signed first-lick lateral error; TIM = unsigned, all parts, all periods, timing and
vigour included. In the response period frozen-TIM should correlate with aim offset (sanity check).
Status: PS92 run (both frames) done; cohort run started 11-16. First look (PS92 only, not a result): acute tongue TIM
down (less licking), nose TIM up in both frames. Check pending: frozen nose TIM is higher than within even for pre
sessions (session-to-session nose posture not removed by the mouth-relative frame).

## 3. Behavioural states -- `wfield_local/glm_hmm.py`

Priya 11-16: states from ALL trials (not engaged / working); not only the cued-response period ("running might come
out as its own state"; "cued vs spontaneous may be their own classes"); choose K by explained variance.
- Unit = one TRIAL CYCLE (the ITI before the trial + its pre-cue + its response); cycles tile the session incl. the
  quit tail.
- Five outputs, each a state-specific multinomial logistic emission of the inputs: cued outcome (3 classes: attempt =
  contact + incomplete / jaw-only / no response; Priya: "unclear if contact and incomplete should be different" ->
  merged so engagement states do not encode the aim deficit; 4-class = sensitivity, `GLMHMM_OUTCOME=4`), pre-cue lick,
  ITI lick, running pre-cue, running ITI (still / some / running).
- Inputs: per-animal bias, spout, previous outcome, spout moved. No session-time input (it would stand in for states).
- Frames: FROZEN pooled August-pre fit primary (Priya 11-16), pre sessions leave-one-session-out; per-animal refit as
  comparison. Trials assigned to a state at posterior > 0.8 (paper).
- K: held-out sessions, bits per cycle vs K = 1 and McFadden pseudo-R2 per output (`--select-k`). Running 11-16.

Relation to Hillary's AR-HMM (`github.com/htong1/Priya_ARHMM2026`, PS46-50; DECISIONS 2026-10-23: not ported, low
priority): that model segments MOVEMENT into syllables frame by frame; the GLM-HMM finds ENGAGEMENT states from trial
cycles. Uses here: syllables could say WHICH movement makes up TIM (grooming / whisking / aborted licks), and a frozen
pre-stroke AR-HMM log-likelihood is the dynamics analogue of frozen TIM (the "global recovery score" the 10-23
decision said to revisit).

## 4. Planned tests (after the TIM + state runs; every readout as delta vs the animal's own pre)

1. Pre-cue TIM vs pre-cue (ENL) position decoding: low- vs high-TIM trials, pre vs post (does unaligned movement
   degrade the spout code -- Yin et al.'s "unaligned movements may do the opposite").
2. Task-aligned vs task-independent R2 in the frozen encoder by TIM tertile / by state (trial counts matched).
3. Does pre-cue TIM predict the next failed attempt / quitting, more so post-stroke (their TIM vs pupil result)?
4. Lick- and cue-evoked response amplitude vs trial TIM (both amplitude references).
5. Quiet-period events (STATUS_2026-11-01): they are face-regressed and occur in a STILLER state that resembles the
   pre-stroke disengaged quit period -- the opposite sign of Yin et al. (disengaged = more TIM). Tests: TIM by epoch in
   pre-cue / ITI (does it fall, as engaged-quiet face speed halves acutely?); quiet-event rate vs the surrounding
   trials' TIM; quiet events, TIM and GLM-HMM disengaged state on the same trials.

## 5. Other items from the review

- Block permutation everywhere it makes sense (Priya 11-16): audit `BLOCK_PERMUTATION_AUDIT_2026-11-16.md`; shared
  null on `fix/block-perm-null-1116`; before / after measurement running.
- Within-trial shuffles (Musall 2023) only as a sensitivity check; the drop dR2 (refit without the variable) stays
  primary; shuffle nulls for movement -> block-segment permutation.
- Lick kernel length -0.5 .. 2 s vs +5 s in the lab's later code: check under GCaMP6s.
