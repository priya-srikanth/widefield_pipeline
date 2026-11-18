# Post-stroke behavioral-state readouts (11-17)

The analysis scripts behind the 11-17 state results: the GLM-HMM engagement states, the AR-HMM syllables and the
treadmill push. They run in the `locanmf` env and take one argument, the scratch root that holds the inputs:
`arhmm_local_fit/`, `arhmm_feat_v3/`, `glmhmm_v1/` and `poststroke_states/`. Outputs and figures are copied to
`Widefield/LP_analysis/<run>/arhmm/v1/poststroke_s1/`.
The results and their caveats are recorded in `docs/status/STATUS_2026-11-16_ENCDESIGN_TIM_STATES_HANDOFF.md`, in the
section 3a block.

Every readout is the change from each animal's OWN August pre-stroke sessions (`analysis_kit.boot_delta`, paired
animals -> sessions), with the per-animal table printed first. Holm correction is applied within each family.

| script | what | figure / table |
|---|---|---|
| `poststroke_states.py` | AR-HMM syllable-group share per session (whole session) + GLM-HMM K5 occupancy | `poststroke_delta_vs_own_pre.csv` |
| `poststroke_states2.py` | engaged-cycle usage, lick direction share, cue-aligned group timeline, group transitions | Fig 1 `fig1_cue_ethogram_engaged.png` |
| `cue_push.py` | treadmill only: peri-cue push (backward > 0.5 mm/s) and walking (forward > 3 mm/s), under the lick working-period gate and the GLM-HMM gate; gate agreement | Figs 2a / 2b |
| `glmhmm_state_curves.py` | GLM-HMM per-state response curves by spout (posterior-weighted observed rates) | Fig 3 |
| `glmhmm_state_curves_epoch.py` | the same curves split by epoch, plus P(state given spout) | Fig 3b |
| `jaw_only_states.py`, `jaw_onset_states.py` | which syllables occur on jaw-only trials and at the jaw onset | tables (handoff) |
| `arhmm_state_check.py` | per-state wheel speed and cue timing (used to name states 4, 11 and 13) | `arhmm_state_check.csv` |

Conventions:
- Epoch colors come from `orofacial_cohort.EPOCH_LINE`.
- The SEM band is alpha 0.08 with no edge lines, the same as the `lick_matched_readout` smoothed curves.
- Spouts are labeled nI, nM, nC, fI, fM, fC.
- Backward wheel velocity is plotted as negative.

Known caveats:
- **nM jaw-only.** Jaw-only cannot be called at nM, so no-lick trials there default to no response.
- **Non-stationary GLM-HMM emissions.** The emissions are shared across epochs. Acutely, the engaged states take on
  far-spout misses.
- **AR-HMM groups.** The syllable groups follow Priya's 11-17 reel review (`state_labels.csv`). States 21 and 22 are
  peri-lick chewing, not the jaw-only attempt.
