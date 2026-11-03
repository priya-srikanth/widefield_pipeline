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
| B8 | Spout order / sign flips / epoch + animal colours re-typed; e.g. `locanmf_decoder_weights:175` draws PS94 in PS93's red | `orofacial_cohort.POS_ORDER`, one `to_contra(x, animal)`, one palette module. |
| B9 | Hard-coded drive paths (e.g. `publish_cache.py:33` writes to `N:`) | `PathResolver` only. |

## Tier C — clarity for non-coders

- **C1.** ARCHITECTURE.md gets a table "**Where each quantity is defined**": epochs, engagement, responded, response
  window, licks (DAQ / cam4 / cam1), trial classes and gates, spout order and sign, baseline and bootstrap,
  corrections, time model, paths, colours, significance stars. One row each: function, module, one-line meaning.
- **C2.** Rename `docs/WHERE_THE_CODE_MOVES.md`. It is about cortex, not code.
- **C3.** A usage line at the top of every script ("python -m scripts.x — what it makes, what it reads"); a
  plain-language docstring on each shared function (what it measures, units, what it excludes).
- **C4.** `iti_recovery_measure` docstring: "all pre" means August pre (the audit's claim 3 was refuted in practice).

## Order of work (after approval)

1. **A1–A5 decisions with Priya.** A1 and A2 first: they touch published behaviour numbers.
2. **B3, B5, B6, B4, B8, B9:** small, bit-identical where the definitions agree.
3. **B2, then B1:** coordinate with the encoder window.
4. **C1–C4:** the map grows as each concept gets its single home.

Each step: own files only, staged by name, a regression check, a second-agent review, a DECISIONS entry where a number
moves.
