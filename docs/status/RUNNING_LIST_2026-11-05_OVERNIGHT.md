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
- (more added below as the night goes)

## 2. Findings

- (none yet)

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
