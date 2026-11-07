# Repo consolidation plan, 2026-11-03 (DRAFT, for Priya's approval)

Goal (Priya 11-03): anything used by more than one analysis is defined ONCE, in one function, and imported
everywhere. Write it for systems neuroscientists, not coders.

Evidence:
- `REPO_AUDIT_2026-11-03.md`: the audit.
- `REPO_AUDIT_VERIFY_2026-11-03.md`: an independent check; every claim marked CONFIRMED / PARTLY / REFUTED, with its
  cohort-1 impact.

Nothing below has been changed yet. Start only after Priya approves and the cohort-1 swap-mask rebuild
(sabatinilab-34) has landed.

**Rule for every step:** the new shared function must reproduce the current output bit for bit where the definitions
already agree. A regression check on a few cohort-1 sessions proves it. Where definitions DISAGREE, Priya picks one
first (Tier A), and the change is logged in DECISIONS with before / after numbers. A second agent reviews each step.

## Priya's answers (11-03)

- **A1:** 2.0 s was NOT deliberate. Read the window per session (`daq_trials.response_window_s`) and RE-RUN the quit
  point and everything downstream of `analysis_kit.RESP_S`.
- **A2:** fix: classify / GATES everywhere `jaw_pass_qc` is used as a filter.
- **A3:** ok, with the condition "don't waste multiple-session power". The animal-weighted nested bootstrap
  (animals → sessions within animal) keeps every session: more sessions make each animal's mean tighter and the
  interval narrower; it only stops an animal with many sessions from out-voting the others. Report the session-level
  mixed model (`mixed_models.lmm_table`, animal random effect) beside it as the check that uses every session's
  variance. For each re-rendered family, list every star that appears or disappears.
- **A4 (decided):** ONE trial classification for every analysis: the 4 kinematics classes (contact / incomplete
  / jaw-only / no response), built on `classify`, with:
  - **jaw = cam4 OR cam1** where that camera can judge it (moved if either camera says moved; unknown only when
    neither can judge, e.g. cam1 at close_center);
  - **time window as a parameter:** default cue → trial stop; a whole-trial option for analyses that need it (e.g.
    ENL / pre-cue work).
  `scripts/trial_attempts.py`'s attempt classes (encoder page) then come from this one function; "attempted" =
  contact + incomplete, as already decided on 10-21. Details to carry over from trial_attempts:
  - a jaw already deflected at the cue counts through its re-onset (`jaw_onsets.jaw_reonset_ms`), else unknown;
  - the close_mid mask;
  - the contact source. Both use DAQ contact underneath; classify matches it to video licks. Settle which is primary
    and log it.
  Owners:
  - 5b: classify / GATES.
  - 34's lineage: trial_attempts.py (last 58f8c14f, no open edits). Its consumers are imaging sections 5 and 8 (failed-
    attempt decoder / maps), the goal-page attempt sections, mixed_models.DEFICITS and the attempt-coding / prelaunch
    scripts; all are in 34's rebuild.
  34 OK'd (11-03) e3 rebuilding trial_attempts on classify BEFORE its renders, with:
  - a regression test against the current classes where the definitions agree;
  - a DECISIONS entry with before / after class counts per animal × epoch;
  - Priya's 10-21 meaning kept (attempted = hit + incomplete; jaw_only = reaction, not attempt; jaw deflected at the
    cue without a re-onset = unknown).
  Send 34 the commit; it merges with A1–A5 and renders once.
- **A5:** ok.

## Coordination rules (Priya 11-03: "coordinate carefully between windows and branches")

- **One owner per file at a time.** Before editing a shared file, ask the windows (34 cohort-1 rebuild, 5b trial
  classes / imaging gates, a8 deck / figure rules, encoder window on wf_attempt) and wait for an OK.
  Record the claim in the table below; release it on commit.
- **Never mid-run.** Don't change a module that a running job imports. Ask the job's owner first.
- **main first, then the branch.** Each shared fix lands on main with a regression check. The attempt branch picks it up
  by merging main into analysis/attempt-coding in the encoder window, by agreement, never by force. B1 (re-unifying
  the package) is scheduled with that window, not done unilaterally.
- **Shared clone hygiene:** stage by name; `git rebase --autostash`; never commit another window's uncommitted files.

**Timing changed (34, 11-03):** the cohort-1 rebuild is ≥ 1–2 days from rendering (pending a drift-cutoff check, an
inventory reconciliation, Priya's LocaNMF-refit call, code on a branch, then ~1.5–2 days of renders). 34 proposes that
A1, A2, A3 and A5 land on **main BEFORE its renders**. It then merges main into its rebuild branch and renders each
affected family ONCE (quit / engagement, 8g / 10 / 10b / 11, rest / nolick gates) instead of twice. 34's jobs run in
their own worktrees (wf_frozendef, wf_swapmask), so main edits don't mix versions mid-run; the risk is only at merge.

| file | owner / claim | status |
|---|---|---|
| wfield_local/epoch_figures.py, epoch_behaviour.py (A3) | a8 has uncommitted edits (figure rules) | waiting for a8 |
| analysis_kit.py RESP_S + quit_point / engagement_decomposition / channel_position_maps (A1) | not 34's | waiting for a8 / 5b |
| orofacial_cohort.py, cam1_incomplete_licks.py (A2) | not 34's | waiting for 5b |
| scripts/jaw_only_incomplete_figs.py `classify` (A4) | **e3** (handed over by 5b 11-03; 5b won't edit it meanwhile). Keep the interface fixed: GATES, `session_classes(label)` → trial_id, position, cue_daq, cls, has_cam1 over every trial; `cue_classes(table, cue_s)`, `in_gate(cls, gate)`. Window as a keyword. | claimed; send the commit to 5b (re-runs imaging section 1) and 34 |
| scripts/trial_attempts.py (A4) | the encoding window = 34 (session f082ee0c, STATUS_2026-10-26_ENCODING_HANDOFF); 5b's and 34's attributions agree | switch under Priya's 11-03 "one definition for all", logged as hers |

**A4 window default (34 + 5b, 11-03):**
- classify's default becomes `window="cue_stop"`; whole-trial stays an option.
- The "working" gate uses cue_stop for BOTH ENL and cue decoding: "working" = responded to THIS trial's cue, and ENL
  is lick-free by its own rule.
- The window name joins `source` in the trial-selection signature, so cache keys and frozen specs separate.
| rest_engagement / rest_by_position / nolick_decoder (A5) | e3 + 34 jointly: must build on 5b's GATES ("working") so there is ONE definition; 34 edits grant_kit `_class_select` | design agreed with 34 (below); signature to be sent to 34 when fixed |

**A5 design (agreed with 34, 11-03):**
- `responded(..., source="camera" | "daq")`, living next to GATES. The default is "camera" (Priya 11-03); "daq" (contact
  within the per-session response window from A1) is the robustness row.
- The source name enters EVERY cache key and frozen-model spec built on it (tf-* features, the epoch_decoder training
  set), via a source tag on the trial-selection signature, so daq and camera runs never collide or load each other's
  models.
- "unknown" trials (no video, unmatched cue) are excluded under camera, as in GATES, and logged per session
  (PS93_0813, PS92_0908 visible).
- The "lick" gate (contact | incomplete) stays separate from "working".
- 34 wires grant_kit `_class_select` to it in the rebuild branch after A5 lands on main.
| hemo_variants, configs/defaults.yaml, combined_encoding, joint_locanmf, session_cache, map_collect, precue_engagement_states, frozen_models, grant_kit `_class_select`, **config.py pre_from / date8** (B2 / B3) | **34** (rebuild) | don't touch; schedule B2 / B3 with 34 |
| branches analysis/frozen-default, analysis/cohort1-swapmask | 34 | merge main in by agreement |

## Tier A — definitions that disagree and can change a cohort-1 number: Priya decides

| # | what | where it differs | proposal |
|---|---|---|---|
| A1 | **"Responded" window = 2.0 s** in the S28 quit point | `analysis_kit.RESP_S = 2.0` → `session_behavior`, `quit_point`, `engagement_decomposition`, `channel_position_maps`. The task's window is 3.5 s in every session from June on (`gui_config.json` `timing.response_window`; May training 3.0 s). No decision records 2.0 s as intended. | Read the window per session with `daq_trials.response_window_s`. Re-run the quit point and show before / after. **Was 2.0 s deliberate?** |
| A2 | **`jaw_pass_qc` (= "jaw moved") used as a trial FILTER** | `orofacial_cohort.py:501/526/546`, `cam1_incomplete_licks.py:56`: denominators are jaw-moved trials only (same issue as section 6, now fixed there). NaN → True in 18 / 4,130 trials. | Use `jaw_only_incomplete_figs.classify` / its GATES for every trial class and jaw gate. Fig 1.3 row 3 and the cam1 incomplete tables then move onto all engaged trials. |
| A3 | **Change-vs-pre statistics weight SESSIONS, not animals** | `epoch_figures.scalar_contrast_draws`, `contrast_draws`, `matrix_bootstrap`, `epoch_behaviour`; the point estimate includes animals the draws lack. `analysis_kit.boot_delta` is animal-weighted, the repo's stated rule. | Route them through `boot_delta`; re-render the affected families (8g / 10 / 10b / 11) and list any star that changes. |
| A4 | **Two trial / attempt classifiers** | `scripts/trial_attempts.py` (encoder attempt classes: DAQ contact, 0..stop window, cam4 OR cam1 jaw) vs `classify` (video contact, cam4 jaw). They don't feed the same figure yet. | Decide whether "attempt" (encoder) and "trial class" (kinematics, imaging gates) should be ONE classification. If yes, one function with documented options; if no, document why they differ. |
| A5 | **Three "responded" definitions + the new imaging "working" gate** | `rest_engagement`, `rest_by_position._engaged_trials`, `nolick_decoder.categorize` (lick detector, interval edges, frame binning differ). The effect on cohort 1 is unmeasured. | Measure the trial-level overlap on cohort 1 (cheap), then fold all of them into one `responded(trials, source=…)`. |

## Tier B — latent today (cohort 2, env vars, or a future run): make safe now

| # | what | proposal |
|---|---|---|
| B1 | **Branches diverged:** main vs analysis/attempt-coding, 60 shared files differ. 150 / 391 attempt scripts import MAIN's package unless sys.path is pinned. Attempt's `config.py` lacks `date8` / cohort2 logic. | Bring the shared package back to one copy: merge or port the attempt-branch changes into main's `wfield_local`, then pin every attempt script. The biggest single job; needs the encoder window's agreement. |
| B2 | Dates as `"2026"+mmdd` (100 sites) and animal id as `label[:4]` (96 attempt, 36 main): breaks PS118-style ids | `config.date8` / one `parse_label()` everywhere. |
| B3 | "August pre" cut-off hard-coded as ≥0801 and ≥0806 at ~60 sites | `config.pre_from()` everywhere. No current number changes. |
| B4 | DAQ lick-detection parameters hard-coded in 28 copies (1.0 / 100 ms vs config 0.5 / none; DECISIONS:19184) | One `daq_licks(session)` reading config. |
| B5 | `rest_engagement._response_window_s`: wrong key + ms-vs-s; only right by accident (falls back to 3.5 s) | Delete it; call `daq_trials.response_window_s`. |
| B6 | 5 hand-written BH implementations; ad-hoc Bonferroni m | One `stats.bh()` / `bonferroni(m=…)` in analysis_kit. |
| B7 | ~17 ad-hoc mixed models (log-day on post-only) | Route through `mixed_models.lmm_table`. |
| B8 | Spout order / sign flips / epoch + animal colors re-typed; e.g. `locanmf_decoder_weights:175` draws PS94 in PS93's red | `orofacial_cohort.POS_ORDER`, one `to_contra(x, animal)`, one palette module. |
| B9 | Hard-coded drive paths (e.g. `publish_cache.py:33` writes to `N:`) | `PathResolver` only. |

## Tier C — clarity for non-coders

- **C1.** ARCHITECTURE.md gets a table "**Where each quantity is defined**": epochs, engagement, responded, response
  window, licks (DAQ / cam4 / cam1), trial classes and gates, spout order and sign, baseline and bootstrap,
  corrections, time model, paths, colors, significance stars. One row each: function, module, one-line meaning.
- **C2.** Rename `docs/WHERE_THE_CODE_MOVES.md`. It is about cortex, not code.
- **C3.** A usage line at the top of every script ("python -m scripts.x — what it makes, what it reads"); a
  plain-language docstring on each shared function (what it measures, units, what it excludes).
- **C4.** `iti_recovery_measure` docstring: "all pre" means August pre (the audit's claim 3 was refuted in practice).

## Order of work (after approval)

1. **A1–A5 decisions with Priya.** A1 and A2 first: they touch published behavior numbers.
2. **B3, B5, B6, B4, B8, B9:** small, bit-identical where the definitions agree.
3. **B2, then B1:** coordinate with the encoder window.
4. **C1–C4:** the map grows as each concept gets its single home.

Each step: own files only, staged by name, a regression check, a second-agent review, a DECISIONS entry where a number
moves.
