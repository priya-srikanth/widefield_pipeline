# Running list — night of 2026-11-05 (PI-deck / code-cleanup window, sabatinilab-a8)

Priya asleep from ~late 11-05: "proceed overnight with everything you can do; keep a running list of findings or
decisions / questions for me". Newest entries at the top of each section. Nothing here changes a published number
until Priya answers the matching question.

## 1. Decisions / questions for Priya (answer these first)

- **Q1. Bare hand runs: default to August-only?** Today `config.pre_from()` is None unless `WIDEFIELD_PRE_FROM` is set,
  so `epoch_grant_figures --only <key>` run by hand pools June + August pre (different frozen spec_id and numbers from
  the nightly). Making the cohort cut the default would (a) rename run-tag folders (`pre0801` would appear on runs that
  had none) and (b) break the blocks that get June by UNSETTING the variable (june_august_drift, restnl_mask,
  map_drift_prepre, reembedding). Held; to agree with the encoding window (34) first. Proposed: unset -> cohort cut,
  explicit `""`/`all` -> June + August, and change those four blocks to pass `all`.
  **Encoding window (34) view:** right intent, but not before the page-replacement pass. Middle ground for now: a
  loud WARNING (no behaviour / tag change) when a cohort-1 module runs with WIDEFIELD_PRE_FROM unset and no profile
  ("pooling June + August pre"); full switch after the pages are replaced, with a test pinning each June block. 34
  has the same question on its list (wf_attempt docs/status/OVERNIGHT_2026-11-05_FOR_PRIYA.md). I agree with 34.
- **Q2. Slide 41 wording (hidden, "Jaw opening by trial class").** Bullet says "acute far-contra contact licks open
  less". On the working-period figure that holds for the first ~1.5 s (acute peak ~1.2 mm vs ~2.3 mm pre) but from
  ~2 s the acute trace stays MORE open (~1.1 vs ~0.6 mm), and acute far-contra contact is 3 animals / 10 sessions /
  61 trials. Proposed: "acute far-contra contact licks open less at first (peak ~1.2 vs ~2.3 mm) and stay partly open
  late (3 animals, 61 trials)". Not changed.
- **Q3. Push rule: rebase vs merge.** CLAUDE.md rule 4 says fetch -> rebase -> push; 34 says your standing rule for
  the rebuild is never rebase / never force, merge instead. I rebased one local, never-pushed branch today (no force,
  nothing on origin rewritten) and will MERGE from now on. Should CLAUDE.md rule 4 be updated to say merge?
- (more added below as the night goes)

## 2. Findings

- **Deck notes carried the OLD engaged-trial numbers (fixed 11-05 night).** After the working-period swap, the
  bullets of slides 37-43 had the new numbers but the speaker notes of 37 / 39 / 40 still quoted the 11-03
  engaged-trial tables (far contra contact .95 / .05 / .74 / .98, incomplete +.57 [.40, .72], per animal .74 / .64 /
  .60 / .32, "1 unknown of 50,929 trials", dots "colour = animal"), and the method lines on 37 / 39-43 said "all
  ENGAGED trials". Regenerated from e0's working-period tables (scripts in the deck source folder:
  notes_working_period_1105.py, notes_numbers_working_1105.py; backups backup_before_notes_working_1105 /
  backup_before_notes_numbers_1105; hidden states unchanged, 191 slides / 84 hidden). New: contact .96 / .05 / .74 /
  .96; acute incomplete +.60 [.42, .75] **; per animal incomplete .80 / .71 / .63 / .36, jaw-only .09 / .10 / .21 /
  .49 (PS95 = one acute session, 109 trials); 0 unknown of 51,355 working-period trials (shares sum to 1). Slide 36
  (the hidden old slide) keeps its old notes, as it shows the old figure.
- **Baseline sweep, cohort2_qc / quiet scripts: 46 sites in 20 files, NO session moves** (all were ">= 0806"; on disk
  0 June and 0 0801-0805 sessions in every input folder). Lint + 153 tests pass. Side note: `quiet_modes.py:116`
  splits pre / post with `session[-4:] < "0816"` (a raw MMDD compare, not the baseline cut); could move to
  `epochs.epoch_of` later.

## 3. Done / state

- **Deck (11-05, all with backups, hidden states diffed before / after):** new slide 35 "Far contra: acute trials end
  as incomplete licks" (single-classification trial shares: contact / incomplete / jaw only / no response, shares sum
  to 1), old slide 35 hidden (now 36); slides 37-43 swapped to e0's working-period figures (numbers in bullets / notes
  rewritten); slide 40 re-swapped to e0's fixed figure (open markers, unclipped title); slide 156 VFSS bullet reworded
  (swallow counts in notes); fluoro slides 158-159 added earlier.
- **Item 3 response-window fallback** (one key `behavior.response_window_fallback_s` 3.5; raster band = session's own
  window): first push passed 5045 tests but was rejected (main moved during the 2 h 17 min run); rebased, pushing
  again.
- **Hemo 470 slot** (`hemo_variants` defaults to the session's slot): committed, pushes after item 3.
- **Item 2 August-only cut** (branch `cleanup/august-pre-key-20261105`, worktree wf_cleanup3): cohorts.yaml
  `baseline_pre_from: "0806"` (Priya 11-05 "switch to 0806"); `config.baseline_pre_from()` / `config.early_pre(x)`
  (year-aware); orofacial `LATE_PRE_FROM` reads it; `WIDEFIELD_PRE_FROM` stays "0801" (names folders; test pins that
  it selects the same pooled pre set). Script sweep running (two helpers); every site where 0805 data would now be
  dropped will be listed in section 2 before merging.
- **Item 5 fail-loud DAQ contacts** (worktree wf_cleanup4): helper running; includes a read-only check of current
  trial tables for all-zero-contact sessions. Any cache-version bump held for Priya.

## 4. Still open (not started tonight)

- Slide 41 wording ("acute far-contra contact licks open less") vs the new jaw traces.
- Aim-circularity note in the 13b plan (e0's area).
- Slide-155 cohort-2 docs commit f3945b44 is local only in the shared checkout; needs its owner to push.
