# Camera trial classes: inventory of outcome-gated analyses (11-04, read-only; wf_rebuild at 571b3955)

**Decision (Priya 11-04).** Split by camera class wherever trials are grouped, gated or scored by outcome.
- Classes: contact, incomplete, jaw_only, none (code: `no_response`), after_end.
- At close-mid the jaw is not judged: the trial goes to none, with `jaw_judged = False`.
- Working = contact + incomplete, cut at the last camera lick.
- Stopped = >= 6 consecutive no_response / jaw_only, with the run starting >= 300 s after the first cue.
- Report per class where meaningful.

**Headline.** Most of this is wired already: the camera gate exists in code, and most working / stopped splits read it.
What it does not cover:
- The DAQ **lick / engaged arm** (`is_engaged`).
- The **state discriminators** trained on DAQ lick vs no-lick.
- **Behaviour hit rate.**
- The **quiet-event quit mask.**
- The default encoder gate.

Two things to know about the current wiring:
- `configs/defaults.yaml:73-79` still says `trial_selection.gate: daq`. Camera applies only when a render flips it.
- Under camera, **jaw_only and isolated no_response no-lick trials are dropped**, not reported as classes.

## 1. Definitions found
**Camera classes (`wfield_local/trial_classes.py`)**
- `session_trials` :178-269 gives one class per trial: contact / incomplete / jaw_only / no_response / unknown.
  - Window: cue to stop.
  - contact = DAQ contact table, counted from cue + 70 ms.
  - `jaw_judged` :261.
  - `JAW_UNJUDGED_POSITIONS` :85 = close_center **and** close_mid. The docstring at :10-15 names close_center only.
- `classify_session` :632-722 applies the session rules:
  - **stopped**: run of >= 6 no_response | jaw_only (`jaw_counts_in_stop` :841);
  - **early_run**: a run that starts < 300 s after the first cue (neither working nor stopped);
  - **after_end**: after the last camera lick (`before_session_end` :358, anchor "lick");
  - an isolated lapse keeps its class.
- **Per-cue hook:** `cue_classes_for` :902. It matches DAQ cues by time within 20 ms.
- **Imaging-row helpers:** `row_classes` :936, `pool_row_classes` :971, `camera_rows` :947, `camera_select` :991.
- **Rest gates:** `camera_engaged_period` :562 (first to last camera lick).
- **Sets:**
  - `GATES` :74-82;
  - `in_sel` / `sel_classes` :333-348: production working = contact | incomplete; miss_working = **incomplete only**;
  - `working_trials.WORKING_CLASSES` :51.
- **The same classes under other names, in `scripts/trial_attempts.py` :55-100.**
  - contact -> hit and no_response -> no_try; output is `qc/trial_attempts.csv`.
  - The run rules (stopped / early_run / after_end) are applied **only if it ran with `WIDEFIELD_TRIAL_GATE=camera`**
    (:83-99).

**DAQ outcome labels**
- **`locanmf_position_decoder.is_engaged` :124.** Rule: first DAQ lick within max_rt (3.5 s).
  - It splits every imaging pool into the engaged arm (XE) and the no-lick arm (XU), at :559.
  - Under `WIDEFIELD_LICK_SOURCE=lp` the train is the LP attempt train (`lp_licks.py:54`, hit / incomplete). The LP
    engaged arm is therefore about equal to contact + incomplete within max_rt.
- **`nolick_decoder.categorize` :172** (with `category_for_rt` :154). Labels: engaged / late_rewarded / undetected, by
  DAQ reaction time.
- **`precue_engagement_states.engagement_gate` :124.** The DAQ quit tail. It is fail-closed under camera.
- **Behaviour.**
  - `daq_trials` :193-242: `hit` / `responded`.
  - `spout_behavior.reference_engagement` :682 and `flag_engagement` :771.
  - `session_metrics` :822-895: hit_rate, engaged-gated.
- **Quit.** `scripts/rest_migration/quit_point.session_quit` :103 (S28 quit); `analysis_kit.session_behavior` :513.
- **`gate_source.gate_responded` :169.** An LP-inclusive "responded" option for the DAQ gate.
- **Encoder gate.** `orofacial_cohort.engagement_keep` :1638.
  - The env `OROFACIAL_GATE` defaults to "reference", the DAQ behaviour column (:1618).
  - Read through `combined_encoding.load_behaviour` (`T.engaged` :544).

## 2. Already on camera classes (no change, given gate=camera at render)
- **Working / stopped arms of the grant and map families.**
  - Via `grant_kit._class_select` :995 -> `camera_select`.
  - Families: beta_maps, position_reference_maps :391, hemo_map_control :126, attempt_maps, position_axes,
    grant_confusion post_working, epoch 12 / 12b.
- **Section D / G frozen position decoders, training set for the cue / precue windows.**
  - `locanmf_frozen_decoder.pool_work` :361-376 trains on engaged and no-lick rows of class contact | incomplete.
- **No-lick split under camera.**
  - `nolick_decoder` :266-283: undetected -> miss_working (incomplete) or stopped.
  - A contact-class "undetected" trial moves to late_rewarded.
  - Consumers: cd_trajectories :798-812, enl_decode :219-252, enl_states.states_for :126-138, precue states 5 / 6
    :265-267, position_coding_directions :808-813, poststroke `undetected_state_split` / `nolick_state_vector`.
- **Rest-baseline working period.** rest_engagement :63-67 and rest_by_position :164-166.
- **`attempt_coding.classify` :78-112.** Classes: pre_hit / hit / incomplete / noattempt (isolated lapse) /
  reaction (jaw_only) / stopped. It is the model for "per class" (Figs 5.1 / 5.2: fig_goal_summary, fig_goal_epochs).
- **Readers of `trial_attempts.csv` classes.**
  - Encoder trial set: `OROFACIAL_GATE=attempted` in ENCODER_RENDER_COMMANDS.md:22, so frozen_encoder :542 and
    combined_encoding :1161 / :1213.
  - jaw_only_incomplete_figs, fig_farcontra_attempts_sessions, learned_compensation :149, kinematic_normality :148-149,
    precue_class_matched, decoder_retention_sessions, coupling_by_class, early_predictors, cohort2_qc/attempt_effort.
  - quiet_task_events :417-428, quiet_lick_lfo :592, lick_cue_to_lick, lick_contact_by_epoch.
  - CAVEAT (uncertain which file is on disk): these follow stopped / after_end / early_run only if trial_attempts.csv
    was rebuilt under gate=camera. Otherwise no_try mixes lapses with the quit tail, and after_end licks count as hits.
- **Refuse under camera.** These cannot mislabel, but they need wiring before use:
  - enl_state_counts, stopped_trial_traces, channel_position_maps :916-917, engagement_axis_balance, nocontact_census;
  - register: tests/test_camera_gate_fail_closed.py.

## 3. Should switch
**3a. Frozen state discriminators: refit, new model ids (add a class tag to the spec)**
- **Section G lick vs no-lick.**
  - Code: `poststroke_compare.balanced_lick_sample` :127, used by:
    - `looks_like_which` :478 (kind g_lick_disc);
    - `fits_engaged_distribution` :586 (g_lick_disc_folds);
    - `undetected_state_split` :1025.
  - Now: trained on pre DAQ engaged vs pre DAQ no-lick. The no-lick side holds **incomplete, jaw_only, no_response and
    stopped together**. It scores all post DAQ no-lick rows as one arm.
  - Proposed: train contact vs no_response (or vs stopped; Q8), then score post per class.
  - Affects: section G panels, plot_poststroke :176-209 ("post UNDETECTED", engaged-like fraction), section_g_figures.
- **Precue engagement-state discriminator.**
  - Code: `precue_engagement_states` :311-345 (stored `<align>-disc`).
  - Now: trained on pre DAQ engaged vs pre DAQ no-lick, both in the late window, minus the SPATIAL_MISS positions.
    States 1 / 2 / 4 are DAQ arms.
  - Proposed: same training choice as above; states by class.
  - Affects: Fig 3.3 s_per_animal (state-decoder column) and the precue-state figure.
- **`spout_scaling` state bal_em** (opt-in, default OFF). It uses the same two DAQ "state classes" and follows whichever
  choice is made above.

**3b. Frozen position decoders: lick arm and scored sets**
- **Lick-window training.** `pool_work` returns None for align "lick" (`locanmf_frozen_decoder` :370), so this arm
  trains on every DAQ-engaged row.
  - Proposed filter: class contact, and not after_end / stopped / unknown. The spec moves, so new ids.
- **"lick" class.**
  - `camera_select` :1006 keeps the engaged rows unchanged (DAQ). Used by:
    - epoch_grant_figures ARMS lick / cuelick / ENLlick :128-150;
    - grant_figures `_session_trials(..., "lick")` :515 / :607;
    - the pattern families' pre reference, which is "LICK trials" (`design_note` :291-297).
  - Proposed: engaged AND contact AND not after_end. Class is a filter; timing stays DAQ.
- **Scored sets of section D (pooled frozen) and G.** Now "engaged / all trials" (`design_note`), and
  `crossed_confusion` :755 post_nolick = all DAQ no-lick. Proposed: score per class.
- **DAQ arms elsewhere.**
  - Success arms: enl_decode :252, enl_states :137, cd_trajectories :809.
  - position_coding_directions `prestroke_lick` / `poststroke_lick` / `prestroke_nolick` (`CLASSES_FULL` :114).
    prestroke_nolick is all pre DAQ no-lick.
  - Proposed: success = contact; split prestroke_nolick by class.

**3c. No-lick arm composition (imaging)**
- **Dropped classes.** Under camera, jaw_only and isolated no_response no-lick trials leave every no-lick analysis
  (nolick_decoder :281-283). Proposed: add them as arms, "reaction" and "lapse", as attempt_coding does.
- **Arm lists that are still DAQ.**
  - nolick_analysis ARMS :380-381 (engaged / late_rewarded / undetected_working / undetected_disengaged /
    nolick_pooled);
  - plot_nolick_reference ARMS :27;
  - plot_poststroke G1 :43-66 (engaged / undetected counts per position, "the primary finding").
  - Proposed: counts and readouts per class.
- **Contact-class rows inside the DAQ no-lick arm.** These are in neither arm (`contact_nolick_counts` :786;
  qc_camera_contact_nolick.csv). Keep that, and report n.

**3d. Behaviour tables / figures**
- **Hit rate.**
  - Now: `spout_behavior.session_metrics` :822-895. Hit = DAQ lick in the response window; engaged = DAQ reference
    quit tail.
  - It feeds cohort_session_metrics.csv, which feeds:
    - **Fig 0.2 / PI slide 4** (fig_behaviour_hits :11-17);
    - grant_behaviour :149 / :222;
    - epoch_behaviour :42-186;
    - engagement_by_epoch :47-76 (engaged fraction);
    - session_pace_table :7-10 / :44 (hit_rate and quit_min are DAQ; n_engaged is already "attempted").
  - Proposed, per spout over the camera working period: shares of contact / incomplete / jaw_only / no_response, plus
    stopped / after_end counts. Hit rate = contact / denominator (Q2, Q3).
- **Clip labels.** behavior_clips.categorise :58-69 (DAQ `responded` + flag_engagement), behavior_deck,
  behavior_clip_deck. Proposed: relabel by class.
- **quiet_lick_lfo :127-132.** The far-contra hit rate on trials before the DAQ quit.

**3e. Encoder / ITI / lick chain**
- **Encoder default gate.** Still "reference" (DAQ), `orofacial_cohort` :1618; only the render env sets attempted.
  Proposed: attempted as the default, or refuse "reference" under camera.
- **ITI first-lick extracts** (iti_licks :334 / :443).
  - Now: `T.engaged` (env gate), then the first behaviour (DAQ) lick onset in cue..stop. Incomplete trials therefore
    yield no first-lick row.
  - Proposed: keep the timing; write the class onto each first-lick row so the families can split contact vs
    incomplete.
  - The same `T.engaged` is read by lick_precue_restw :121 and restw_strict_check :251.
- **rt_drift :83.** Uses `is_engaged` (DAQ). Proposed: contact only, or note it.

**3f. Quiet events**
- `cohort2_qc/quiet_area_spikes.session_traces` :94-111 gives an engaged mask = before the DAQ quit
  (session_behavior / quit_point).
- Every quiet_* script reads that mask, e.g. quiet_quit_state :74 (engaged vs quit), quiet_task_events :139, quiet_eta_split,
  quiet_spikes_v2, quiet_lick_lfo.
- Proposed: engaged = camera working period; quit = camera stopped run(s) plus after_end (Q1).
- quiet_task_events_followup :237-240 refuses bal_em centre = working (its centre is every engaged row). It needs the
  camera classes to run under camera.

**3g. Imaging averages split by outcome (refuse now; wire when they are next needed)**
- stopped_trial_traces (quit-tail traces), channel_position_maps (quit-period exclusion).
- engagement_decomposition (2 class refs; not checked in detail).

## 4. Cannot switch, or should stay DAQ
- **PS93_0813 (no video).** Every cue is "unknown", so it is in neither working nor stopped.
  - It stays only in the DAQ "lick" class (`camera_select` :1006) and in the bal_em centre fallback
    (defaults.yaml:83).
  - Every per-class report drops it (`working_excluded_sessions` :300).
- **Partial video coverage** (PS92_0908 17.5 % unknown). Covered trials only (`coverage_note` :766).
- **June sessions.** No v36 classes were found for them (uncertain). They are outside the Aug-only pre set; June users
  (iti_june_ref, june_august_drift, the drift steps) stay DAQ.
- **Lick-aligned imaging.**
  - Windows start at the DAQ first lick (`locanmf_position_decoder` :559-560), or the LP train under lick_source=lp.
  - The class **can** be used as a filter (contact, not after_end).
  - Incomplete trials have no DAQ lick time; only the LP arm can align them.
  - The no-lick arm stays out of lick alignment.
- **Per-lick outcome** (a lick hit / missed the spout): within_lick_correction OUTC :488, lick_correction :539. This is
  a property of the lick, not a trial class; stays.
- **Measures OF engagement** stay DAQ, each opened with `daq_quit_rule`.
  - quit_point.session_quit, spout_behavior.flag_engagement, quit_prodrome, lick_bout_structure, disengagement_tests;
  - gate_lp_check, lp_gate_option_counts.
  - A camera "time of first stopped run" could be added alongside.
- **Epoch boundaries.** epochs.py :206-227 (hit + licks series, engaged trials) are pinned (`load_boundaries` :361).
  Recomputing them would move epochs, so keep them pinned (Q11).
- **s1 tables' quit-tail rows** (`rule="daq"`). A labelled DAQ robustness row.
- **Cohort 2** (PS118 / 120-122). `trial_classes` reads the cohort-1 v36 LP outputs, and no cohort-2 classes were found
  (uncertain). Cohort-2 quiet / QC runs stay DAQ and are never pooled.

## 5. Needs a decision (to Priya)
1. **after_end.** Should it count in per-class tables, as its own class, or be excluded? Should quiet events treat it
   as "quit"?
2. **Hit-rate numerator and denominator.**
   - Numerator: contact only?
   - Denominator: all working-period trials (contact + incomplete + jaw_only + no_response), or attempts only?
   - Is jaw_only a "response" anywhere?
3. **Hit-rate period.** Should it be computed inside the camera working period (first cue -> last camera lick, minus
   stopped runs) instead of the DAQ quit tail? That changes Fig 0.2 and every cohort_session_metrics reader.
4. **Unjudged close spouts.** Pool close_center / close_mid unjudged no_response with no_response, or report it apart
   (`jaw_judged`)? The code treats **both** close_center and close_mid as unjudged (:85).
5. **No-video sessions** (PS93_0813, partial-coverage cues). In per-class reports: drop them, or show a DAQ-only row?
6. **DAQ no-lick trials with camera class contact** (contact after max_rt / table mismatch). Keep them excluded from
   both arms?
7. **early_run** (a stop run that starts < 300 s). Report it as its own class or fold it into no_response?
8. **State discriminators.** Train contact vs no_response, vs stopped, or vs no_response + stopped? Is incomplete
   scored only (never trained)?
9. **"lick" class.** Contact only (timing DAQ), or contact + incomplete with LP timing for the lick-aligned arms?
10. **Encoder default gate.** Make attempted the default under camera (and refuse "reference")?
11. **Epoch boundaries.** Keep them pinned on the DAQ hit-rate series?
12. **Isolated lapses (no_response) and jaw_only in the no-lick decoders.** Add them as arms (as attempt_coding does),
    or leave them out?
