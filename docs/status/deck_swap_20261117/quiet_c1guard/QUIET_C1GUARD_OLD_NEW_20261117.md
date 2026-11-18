# Quiet-period events: cam1 far-from-mouth guard, old vs new (a8, 11-17)

**Why:** Priya OK'd the re-run via 3c. Under the new rule, cam1-only events count as licks only if they are non-droplet
AND med_out_px >= -120. The audit is `CAM1_FARMOUTH_BLAST_RADIUS_20261117.md` on branch analysis/restnl-cam1-1117.

**Runs:**
- **Code:** guard worktree wf_c1guard1117, branch fix/cam1-farmouth-guard-1117. HEAD at run time was 0100aabf; its
  code is identical to 3c's 95ba249b (only one test file differs).
- **Env:** replayed from the production run (v2_quiet_lp_20261117/logs/quiet_v2.log). A diff confirmed that only the
  code path and the output directory differ.
- **Outputs**, under `rebuild_20261103/labcams/<tag>/cohort1_quiet_spikes/`:
  - `v2_quiet_lp_c1guard_20261117`: the full re-run (108/108 sessions, 14 min). Its leave-one-animal-out threshold
    choice moved orofacial k from 3.75 to 4.5.
  - `v2_quiet_lp_c1guard_k375_20261117`: the same per-session data, summarized at the PRODUCTION orofacial k (3.75).
    `--k` sets one common k, so its limb numbers are at 3.75 and are NOT comparable. Its BH q-values are also not
    comparable, because the limb terms in the same family changed; use p.
- **Scripts:** `run_quiet_v2.sh`, `run_quiet_v2_k375.sh`, `compare_old_new.py`. Tables are in `COMPARE_OLD_NEW.md`
  (re-chosen k) and `COMPARE_OLD_NEW_k375.md` (fixed k).

## 1. The guard itself (same k = 3.75): essentially no change
- **Quiet time:** +0.0 to +0.1 min per animal x epoch. By session: PS92_0929 +52 % (1.99 -> 3.03 min), PS93_0929
  +11 %, PS93_1002 +9 %, PS93_0812 (pre) +8 %, PS94_0903 +6 %, the rest <= 3 %.
- **Orofacial rate, change vs own pre, chronic, old -> new (events/min):** PS92 +8.50 -> +8.49, PS93 -0.21 -> -0.20,
  PS94 -1.87 -> -1.80, PS95 +1.12 -> +1.14. Acute and subacute move by < 0.3.
- **Orofacial LMM (log rate), old -> new:**
  - coefficients: acute .585 -> .586, subacute .359 -> .362, chronic .126 -> .130
  - p: chronic p .124 -> .111; contralesional (left) chronic p .018 -> .017; both polarities chronic p .004 -> .004
- **Limb (k 8.75 in both full runs):** per-animal means move by <= 0.15 events/min (PS92 chronic 6.87 -> 6.72);
  LMM chronic .608 -> .607.

## 2. The full re-run with the threshold re-chosen (k 4.5)
- **Why k moved:** the k choice is the median of the 4 leave-one-animal-out choices. Two folds moved from 3.5 to 4.5
  (held-out PS92, PS95), and the pooled AUC is flat across 3.5-4.5 (.977 / .979 / .979 -> .980). So the move comes
  from a flat AUC curve, not from the data changing.
- **Effect of k 4.5:** every orofacial rate is lower, pre included (e.g. PS92 pre 4.76 -> 2.51). Changes vs own pre
  shrink in absolute size. Log-rate LMM: acute .69, subacute .41, chronic .14 (p .22); contralesional chronic .19
  (p .023); both polarities chronic .21 (p .020). Same direction everywhere.
- **Quiet minutes LMM:** q acute .02 -> .34 with the same coefficients (5.7 -> 4.8, -7.2 -> -7.2). The random-effect
  structure switched from a random intercept (~1) to a per-animal slope (~post), which roughly doubles the SE.
  lmm_table picks the structure by fit, so a tiny data change flipped it. This is a model-selection artifact, not a
  change in quiet time.

## Conclusion (for Priya, before any page / deck change)
- The guard does not change the quiet-event result. At the production threshold, every per-animal number and model
  term moves by under 2 % of its size.
- The only material differences come from (a) the leave-one-animal-out threshold choice flipping 3.75 -> 4.5 on a
  flat AUC curve, and (b) the quiet-minutes LMM switching its random-effect structure.
- Options:
  - keep k 3.75 and update captions only (the lick rule and code), or
  - adopt the re-chosen k 4.5, which re-renders the figures with lower absolute rates and the same conclusions.
- Page section 23 and deck slides 157 / 158 are NOT changed. 3c also listed slide 112.

## Decision and what was changed (Priya 11-17: "k = 3.75 is good")
- Figures and numbers kept as rendered at k 3.75. Captions and notes now say that cam1 far-from-mouth labels were
  still counted in this render, and that removing them changes every rate and model term by < 2 %.
  - Deck slides 112, 154-158 and 203: the notes "Licks:" line is extended (quiet_licks_note.py; backup
    .bak_pre_a8_quietnote_20261117).
  - Goal page v320, section 23: 23 lick sentences are extended the same way.
- **Flag for Priya:** goal page section 23 still shows the PRE-REBUILD quiet-event outputs (cohort1_quiet_spikes/
  v2_quiet_lp/, 'k = 4'). The deck uses the rebuilt run (v2_quiet_lp_20261117). Swapping the page to the rebuild is a
  separate pass; nothing was swapped here.
- Once 3c's guard branch is merged (Priya's OK on the diff), future quiet re-runs use the guard automatically.

## Correction later on 11-17: the guard run is production; follow-on figures at k 4 (deck applied 11-18)
- Priya clarified: "k 3.75 with cam1 fix" = the guard run is production. Then: "k 4 iw fine". The follow-on scripts keep
  their own fixed k = 4, as in the 11-17 deck pass. Then: "re run both" (slides 154 and 157 were still on the
  pre-rebuild `labcams/cohort1_quiet_spikes/v2_quiet_lp`). The "< 2 % (re-run, same k 3.75)" note above was wrong
  about k for these slides. It is removed from every slide, and from slide 112 (gain_bio pipeline, never measured).
- **Run:** `v2_quiet_lp_c1guard_k375_20261117`, with `sessions/` copied from the guard re-run. Follow-on steps go
  through `run_downstream_k375.sh <step>`: examples, fr, eta, task, tasksumm, hemo, still, stillplot, movie. Logs are
  in `<run>/logs/`. The fr and hemo logs show exit 2 because the runner was edited while bash was reading it. Python
  had finished and written every output in both, and the logs are annotated. Never edit a script while it runs.
- **Task summary (slide 158):** the guard re-keys the camera-class csig (3d7bbe921e -> 0f1b20ae2d), and attempt_coding
  has no guard-keyed `trials_precue` tables. So `task_events_summarise_prodptrue.py` reads the PRODUCTION-key p_true
  tables. p_true comes from the frozen decoder on the DAQ 2-s window, which no cam1 label enters. Event flags come from
  the guard run. The summary log says 108 sessions, but t13 pre n_sessions = 31 as before (not yet explained; same as
  the old table).
- **Old (deck) -> new, per slide:**
  - **155, example traces:** PS92 acute orofacial ipsi 46 -> 45, contra 56; pre 1 event; thresholds 1.26 / 1.22 %.
  - **156, face-regressed rate:** x6.6 / x3.3 / x1.7 unchanged (coefficients .8205 -> .8228, .5128 -> .5167,
    .2192 -> .2198); max |delta coef| .017 over 108 terms, and no q crosses .05.
    - Pre -> acute: PS93 49.6 -> 49.7, PS94 30.9 -> 31.0, PS95 22.9 -> 23.1.
    - Ipsi x6.5 -> x6.4, contra x6.0 -> x6.1; raw x4.9 / 2.6 / 1.6 and limb x2.3 unchanged.
  - **158, pre-cue event (strict):** pre -.020 -> -.019; acute -.017, subacute -.020, chronic -.010 unchanged.
    - Per-animal pre: PS93 -.035 -> -.032, PS94 -.001 -> .000.
    - Far-contra no-attempt +.045 unchanged.
  - **203, event maps:** outlined bins at -0.3 / 0 / +0.2 / +1.0 s go 58 / 55 / 542 / 70 -> 60 / 56 / 542 / 85; rim
    1.9x -> 1.8x; decay -0.0005 s (p .96) -> -0.0001 s (p .99).
  - **154, movie** (pre-rebuild -> guard; mixes the rebuild with the guard): same clips, pre start 8505 -> 8503 s.
    - Events 13 / 30 -> 13 / 40, because PS92's threshold drops 1.86 / 1.84 -> 1.26 / 1.22 % after the rebuild.
  - **157 left, stillness** (pre-rebuild -> guard): acute rise per bin +.93 ... +.36 (all q < .01) -> +.74 / +.67 /
    +.52 / +.41 / +.25.
    - Most-active bin q .066; the other bins q <= .001.
    - Time at < 0.5x pre face speed 19 % -> 47 %, unchanged.
  - **157 right, 415 nm:**
    - 415 stays flat (q .34-.75).
    - Corrected 470 SD +.21 -> +.28; 0.5-1 Hz subacute q .02 -> .057.
    - 415 response at 470 events +.37 (q .001) -> +.20 (q .22).
- **Deck (Priya 11-18 "ok"):** `deck_swap_c1guard.py` (exact-match edits) and `c1guard_crops.py`. Re-cutting the old
  renders reproduces every current deck image pixel for pixel, so the new images use the same geometry.
  - Backup `.bak_pre_a8_c1guard_20261118`.
  - Pictures: 155-158 and 203. Movie and poster on 154.
  - Notes: numbers and FIGURE SOURCE. The Licks line now says cam1-only labels far from the mouth are not counted.
  - Slide 158 bullet: pre -.019.
  - Slide 112: the note is removed.
  - QC by exporting the slides from a scratch copy.
- **Not changed:** goal page section 23 (3c's). It still shows the pre-rebuild quiet outputs plus the v320 "< 2 %"
  sentences, which are now wrong about k. 3c was told and is relaying to Priya.
- **Superseded (deletable on Priya's OK, not deleted):** the follow-on outputs in `v2_quiet_lp_20261117/`
  (fr_regressed_own, eta_fr, task_events, examples) and `v2_quiet_lp_c1guard_20261117/` (k 4.5 summary).
