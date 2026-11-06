# Running list — night of 2026-11-05 (PI-deck / code-cleanup window, sabatinilab-a8)

Priya asleep from ~late 11-05: "proceed overnight with everything you can do; keep a running list of findings or
decisions / questions for me". Newest entries at the top of each section. Nothing here changes a published number
until Priya answers the matching question.

## 1. Decisions / questions for Priya (answer these first)

- **Q4 DONE (Priya "can change", relayed by 34).** Slide 161 bullet 2 now: "The few swallows left carry about a third
  of the barium (smaller and less dense bolus) and clear the esophagus slowly" (was "about half the bolus" = area
  only; amount x0.34 day 1). reword_fluoro_bullet_1105.py, backup backup_before_fluoro_bullet_1105.
- **Q5 DONE (Priya "ok", 11-06).** CLAUDE.md rule 4 now says fetch -> merge -> push. The pre-push hook's header comment
  still says "rebase"; that is left for the next real hook change, since any `.githooks/` edit triggers the full
  suite (~2 h 40 min). Was: **Q5. CLAUDE.md rule 4 -> merge?** (follows Q3: Priya asked "which is safer". Answer given: merge, because several
  windows push one main and a rebase rewrites commits another window may have seen; neither is risky without a
  force-push.) Proposed edit: "fetch -> merge origin/main -> push". Waiting for OK before editing CLAUDE.md.
- **Q1 follow-up.** Priya answered "yes"; read as yes to the 34 / a8 plan (warning now, default switch after the
  page-replacement pass). The warning lives in modules 34 renders from, so it goes on a branch and merges after 34's
  render. If "yes" meant switch the default now, say so (that renames folders and needs 34 to change four blocks).
- ANSWERED 11-05 night: Q1 "yes" (above); Q2 slide 41 "reword for accuracy" (DONE, below); Q3 "which is safer" ->
  merge (Q5).

- **Slide 11 pre-cue wording (Priya 11-06: "can wait on the re-run").** The bullet "Pre-cue = position information
  before any lick" is inaccurate. The window is a lick-free (camera + spout) window between spout arrival and the cue
  (`precue_window_start`). The figure / numbers are the 2.0 s table (LP_licks_ENL2s); the primary 1.0 s table (Priya
  11-04) is still to be made in the post-rebuild pass. Change the text, figure and numbers together to 1 s after that
  re-run. Post-cue window = cue to +2.0 s (`decode.cue_post_s`).
- **34's fix/log-deficit-primary-20261105 -> main: Priya OK (11-06, in a8's window).** 34 asks for the OK in its
  own window before pushing; a8's slide-18 pre-dots branch builds on it.

## 2. Findings

- **Fail-loud DAQ contacts (item 5): no current output took the silent-zero path.** All 133 v36 sessions: DAQ .h5
  present, n_daq_contacts never all-zero / NaN / missing, 2,006-10,805 DAQ licks each; rebuild trial_attempts (107
  sessions, 57,048 rows) has no session without contact trials. Lowest contact shares vs the animal-epoch median
  (all with DAQ, so behavioural): PS94_0819 acute .29 (median .47), PS95_0820 subacute .47 (.61), PS93_0822
  subacute .54 (.85), PS95_0910 chronic .60 (.88), PS94_0806 pre .67 (.91). No cache bump needed.
- **Baseline sweep, all of it: no session moves.** ~180 sites (126 files). Every "0801" / month / LATE_PRE_FROM
  site on an uncurated input was checked on disk: no 0801-0805 entries in any input (0805 exists only in the pose3d
  frames glob, which already used 0806, and in orofacial session dirs without licks_all.csv). The month tests equal
  the cut because pre sessions are June and August only. 191 + 153 + 24 tests pass, ruff clean.
- **Deck notes carried the OLD engaged-trial numbers (fixed 11-05 night).** After the working-period swap, the
  bullets of slides 37-43 had the new numbers but the speaker notes of 37 / 39 / 40 still quoted the 11-03
  engaged-trial tables (far contra contact .95 / .05 / .74 / .98, incomplete +.57 [.40, .72], per animal .74 / .64 /
  .60 / .32, "1 unknown of 50,929 trials", dots "colour = animal"), and the method lines on 37 / 39-43 said "all
  ENGAGED trials". Regenerated from e0's working-period tables (notes_working_period_1105.py,
  notes_numbers_working_1105.py, notes_s38_working_1105.py; backups beside them). New: contact .96 / .05 / .74 /
  .96; acute incomplete +.60 [.42, .75] **; per animal incomplete .80 / .71 / .63 / .36, jaw-only .09 / .10 / .21 /
  .49 (PS95 = one acute session, 109 trials); 0 unknown of 51,355 working-period trials. Slide 38: flagged LMM fits
  23 -> 25 of 72. Slide 36 (hidden old slide) keeps its old notes, as it shows the old figure.
- Side note: `scripts/cohort2_qc/quiet_modes.py:116` splits pre / post with `session[-4:] < "0816"` (raw MMDD
  compare of the stroke date, not the baseline cut); could move to `epochs.epoch_of` later.

## 3. Done / state

- **Deck now 192 slides / 85 hidden.**
  - 11-05 night: fluoro all_graphs swapped to the 22:50 ten-panel A-J figure (G density, H amount; E / I / J now
    tested) on slide 161, notes rewritten (density x0.60 / x0.57, amount x0.34 / x0.28, q < .001; transit x2.13 /
    x6.33, sitting x3.88 / x7.63, gap -9.3 / -18.4 pct points; new footnote); NEW hidden slide 162 "How bolus density
    is measured" (density_method.png). update_fluoro_density_1105.py, backup backup_before_fluoro_density_1105.
    Requested by Priya via 34; a8 did the edit (sole writer).
  - 11-05 night: slide 41 reworded (Priya "reword for accuracy"): "acute far-contra contact licks open less at first
    (peak ~1.2 vs ~2.3 mm pre) but stay more open from ~2 s (3 animals, 61 trials)". reword_slide41_1105.py.
  - 11-05: new slide 35 (trial shares sum to 1), old 35 hidden (now 36); 37-43 swapped to working-period figures;
    slide 40 re-swapped (open markers); slide 156 VFSS reword; fluoro slides added (now 160-161).
- **Item 3 response-window fallback: ON MAIN** (9ad4fbcc; pre-push full suite passed 11-06).
- **Hemo 470 slot** + this running list: ON MAIN (merge f1fd9f9f, 11-06). The pre-push hook now runs only the tests
  touched by the push (53 passed in 32 s), so pushes no longer take ~2 h.
- **Item 2 August-only cut: ON MAIN** (37bacf12, merge 5fa8253c, 11-06). Pre-push ran the full suite because the push
  touched ~126 files: 5053 passed, 179 skipped, 2 h 42 min. 34's render is in its own pinned worktree, unaffected.
- **Item 5 fail-loud DAQ contacts**: committed 5834f8de on `cleanup/daq-contacts-failloud-20261105` (wf_cleanup4),
  not pushed. Touches trial_classes / tongue_kinematics / dlc_frames / kinematics_chain / orofacial_outputs; a
  session without a DAQ now raises (or is class 'unknown' and not covered) instead of silent zero contacts. HOLD the
  merge until 34's render finishes (review said "wait: yes" for trial classes). 34 (11-06): render is under a
  low-RAM STOP, so it won't finish before the weekend.
- 34's overnight page now records 5fa8253c as the cohort key (Priya's "yes switch to 0806"); bare-run selection is
  unchanged, and its A4 (= Q1 here) is open and not started.
- O2 FR null: sbatch throttle is `%60` (16 CPU / 96 GB each); raise live with
  `scontrol update JobId=<id> ArrayTaskThrottle=N`; check pending reasons with squeue %r.

## 4. Still open

- Aim-circularity note in the 13b plan (e0's area).
- Slide-155 cohort-2 docs commit f3945b44 is local only in the shared checkout; needs its owner to push.
