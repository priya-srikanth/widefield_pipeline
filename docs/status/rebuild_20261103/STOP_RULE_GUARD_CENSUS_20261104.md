# Stop-rule guard census, 2026-11-04 (read-only)

Question (Priya, via a8): does the camera working period drop genuine failures at the deficit spouts? The two risks are (a) the end cut drops a final far-spout block of failures, and (b) a mid-session "stop" run is really one or two far-spout blocks of failures. Either would inflate the hit rate where the stroke acts.

Code: `wf_rebuild` on `rebuild/cohort1-20261103`, with profile `cohort1_rebuild_20261103` applied in-process. Selection = `sel-camera-cue_stop-endlick-cls-contact-incomplete-stopRun6j`: run >= 6, jaw_only counts toward the run, 300 s start rule, end at the last camera lick. Nothing in the repo was edited.

## Summary

**The guard matters, mostly through mid-session runs; the end cut barely matters.**

1. **Risk (a), the end cut, is small.** Under the current rule only **43 trials** in the whole cohort are `after_end`. The quit tail is almost always caught first as a `stopped` run, which is 5,882 trials. The guarded end moves later in 33 / 107 sessions (trial-level), by a median of 0 trials and a maximum of 25. In 13 sessions no reference-spout trial comes after the last lick, so under the guard the session has no end cut (listed below).
2. **Risk (b), mid-session runs, is real.** Of the 277 current stop runs, **85 contain no close-ipsi or close-mid trial at all**. Of these, 48 are acute, 15 subacute, 13 chronic and 9 pre. In acute, the reference trial just before such a run was a **contact in 39 / 48** cases. The trial that ends the run was a lick in 45 / 48 (28 contacts, 17 incomplete). So in acute these look like an animal still working at the close spouts while failing at the far ones. Example: PS95_0817, the only acute PS95 session, has 62 trials stopped under the current rule. They are runs of a single far-contra block (trials 133-138, 179-184, 637-642) and far-contra/far-mid pairs. The guard puts all 62 back.
3. **Totals.** Stopped trials drop from 5,882 to **4,152** under the trial-level guard and to 3,265 under the block-level guard. The guard changes classes in **57 / 107 sessions** (trial-level) and 64 / 107 (block-level), including **all 16 acute sessions**.
4. **Hit rate, cohort mean of the four animals, acute, current to trial guard:**

   | spout | current | trial guard |
   |---|---|---|
   | far mid | .470 | **.406** |
   | far ipsi | .739 | .662 |
   | close contra | .916 | .845 |
   | far contra | .054 | .048 (already at floor) |
   | pooled | .691 | .631 |

   The acute deficit vs pre changes as follows:

   | spout | current | trial guard |
   |---|---|---|
   | far mid | -.467 | -.507 |
   | far ipsi | -.159 | -.212 |
   | close contra | -.065 | -.122 |
   | far contra | -.904 | -.884 |
   | pooled | -.265 | -.310 |

   Subacute pooled goes from -.059 to -.083. Chronic is essentially unchanged: pooled +.018 to +.022.
5. **The guard is not deficit-specific.** It puts back trials at every non-reference spout: close contra, far ipsi, far mid and far contra in roughly equal numbers. It does this in **pre** too: pre pooled hit .956 to .941; PS95 pre .951 to .920; PS94 pre .976 to .959. The 9 pre no-reference runs have a contact at the reference spout before them only 3 / 9 times, against 39 / 48 in acute. So in pre, much of what the guard puts back is disengagement. Under the guard, part of the acute drop is that same disengagement. Read the acute deficit against the guarded pre, not the current pre; the deficit tables above already do this.
6. **Decoder working set and rest period: unchanged in all 107 sessions.** This holds by construction, and I checked it on the DAQ cue train with `cue_classes_for`'s matching. The contact / incomplete trials in the working period are identical, and the rest span (first to last camera lick, `_lick_span`) is identical. Lick trials are never stopped, because a lick ends a run. With the end at the last camera lick, lick trials are never `after_end` either, and the guard only moves the end later or removes stops. **Contact-given-attempt is identical** under every rule.

   What does change is the **`stopped` set**: 5,882 to 4,152 trials on the table train, the same counts on the cue train. That set is read by:
   - the no-lick decoder's stopped arm (`nolick_decoder.py:277-282`);
   - the ENL working-vs-stopped contrast (`enl_states.py:129-137`).

   Those two families would change, and so would the attempt-rate denominator (`trial_attempts.py:94-100`).
7. **jaw_only trials in stop runs:** current 2,329 stopped and 24 after_end; trial guard 1,453 and 13; block guard 1,084 and 2. no_response stopped: 3,553, 2,699 and 2,181. About 55-64 % of trials in the no-reference runs are jaw_only, so much of what the guard puts back is jaw_only. jaw_only is in the hit-rate denominator but is not a working trial for the decoders.

**Recommendation (for Priya):**
- Use the **trial-level** guard, not the block-level one. The block-level version puts back close-ipsi and close-mid failures that sit inside runs with only a partial reference block: 36-91 trials per epoch at the reference spouts. That contradicts the guard's own logic. The trial-level version leaves the reference spouts almost untouched: 5-13 trials per epoch, all from the moved end.
- Report the hit rate both ways. The current rule is the lower bound on the deficit; the guarded rule is closer to an upper bound, because it also brings back some disengagement (point 5).
- Two cheap refinements would make the guard more specific (not run here):
  - Require a **reference-spout contact on both sides** of a no-reference run before keeping it.
  - **Do not trim the head** of runs that do contain a reference trial. The `notrim` column shows that trimming adds about 900 of the ~1,730 trials the guard puts back. The trimmed heads have median 2 and max 28 trials.

## Trial structure

- Spouts come in **blocks**: runs of the same position, median 6 trials (IQR 5-7, range 1-16). There are about 88 blocks per session (39-148) and 225-871 trials per session.
- About 29.5 blocks per session (14-50) are at a reference spout. All six positions recur throughout the session, so a stretch with no reference spout is at most a few blocks long. The longest no-reference stop run was 25 trials over 4 blocks.
- "Block" is therefore unambiguous. I implemented both a trial-level and a block-level guard. The trial-level one is cleaner (see the recommendation).

## Sessions

- 107 sessions are in the census: cohort 1 (PS92-95), curated, regime B, with v36 video outputs and an epoch under the profile. June pre is excluded by `WIDEFIELD_PRE_FROM=0801`.
  - PS92: pre 8, acute 5, subacute 2, chronic 12.
  - PS93: pre 7, acute 4, subacute 3, chronic 12.
  - PS94: pre 8, acute 6, subacute 5, chronic 8.
  - PS95: pre 8, acute 1, subacute 7, chronic 11.
- PS93_0813 (no video) is not in the census; `sessions_available` already leaves it out.
- **Partial coverage:** PS92_0908 (chronic). 82.5 % of its cues are matched to video and the tail is uncovered, so neither rule applies an end cut.
  - Only on the DAQ cue train (the decoder check): the uncovered tail is "unknown" there. The census tables use the per-video-trial table, as `trial_attempts.py` does.
  - All other sessions: coverage >= 99 %. One cue in the whole cohort is `unknown`.
- Spouts are named canonically. All four lesions are LEFT, so close_L = close ipsi, close_center = close mid, close_R = close contra, far_L = far ipsi, far_center = far mid, far_R = far contra (`orofacial_cohort.pos_key`, :124).

## Tables

Hit rate = contact / working-period trials, where working-period trials = contact + incomplete + jaw_only + no_response + early_run, i.e. not stopped / after_end / unknown. Pooled over trials within animal x epoch. Cohort numbers are the mean of the four animals. Contact-given-attempt = contact / (contact + incomplete) in the working period.

### T1. Trials removed (stopped + after_end), cohort totals by epoch x spout

| epoch | rule | close ipsi | close mid | close contra | far ipsi | far mid | far contra | all |
|---|---|---|---|---|---|---|---|---|
| pre | current | 120 | 130 | 111 | 164 | 155 | 139 | 819 |
| pre | guard trial | 115 | 127 | 72 | 85 | 70 | 54 | 523 |
| pre | guard block | 89 | 111 | 52 | 72 | 62 | 31 | 417 |
| acute | current | 357 | 346 | 327 | 466 | 476 | 478 | 2450 |
| acute | guard trial | 351 | 336 | 196 | 302 | 270 | 270 | 1725 |
| acute | guard block | 317 | 291 | 153 | 225 | 209 | 236 | 1431 |
| subacute | current | 291 | 276 | 282 | 271 | 351 | 296 | 1767 |
| subacute | guard trial | 278 | 265 | 184 | 185 | 230 | 177 | 1319 |
| subacute | guard block | 208 | 233 | 142 | 127 | 185 | 138 | 1033 |
| chronic | current | 137 | 105 | 141 | 170 | 161 | 175 | 889 |
| chronic | guard trial | 130 | 101 | 86 | 95 | 87 | 116 | 615 |
| chronic | guard block | 96 | 78 | 40 | 58 | 54 | 75 | 401 |

- Per animal x epoch x spout counts, including the number of blocks removed in full (`blocks_removed`), are in `stop_rule_guard_by_animal_epoch_spout.csv`.
- Current rule split: 5,882 stopped and 43 after_end. Trial guard: 4,152 and 30. Block guard: 3,265 and 17.

### T2. Hit rate per animal x epoch, all spouts pooled (current / trial guard / block guard)

| animal | epoch | current | guard trial | guard block | removed (current / trial / block) |
|---|---|---|---|---|---|
| PS92 | pre | .980 | .974 | .973 | 22 / 3 / 1 |
| PS92 | acute | .734 | .671 | .638 | 713 / 527 / 416 |
| PS92 | subacute | .909 | .897 | .897 | 8 / 0 / 0 |
| PS92 | chronic | .991 | .991 | .991 | 0 / 0 / 0 |
| PS93 | pre | .918 | .911 | .911 | 54 / 25 / 25 |
| PS93 | acute | .707 | .671 | .638 | 661 / 572 / 484 |
| PS93 | subacute | .876 | .840 | .839 | 205 / 145 / 142 |
| PS93 | chronic | .985 | .981 | .981 | 107 / 82 / 81 |
| PS94 | pre | .976 | .959 | .950 | 314 / 239 / 197 |
| PS94 | acute | .629 | .548 | .531 | 1014 / 626 / 531 |
| PS94 | subacute | .916 | .876 | .857 | 380 / 262 / 201 |
| PS94 | chronic | .963 | .943 | .939 | 207 / 111 / 89 |
| PS95 | pre | .951 | .920 | .909 | 429 / 256 / 194 |
| PS95 | acute | .693 | .633 | .633 | 62 / 0 / 0 |
| PS95 | subacute | .888 | .819 | .768 | 1174 / 912 / 690 |
| PS95 | chronic | .960 | .936 | .908 | 575 / 422 / 231 |

### T3. Hit rate per spout, cohort mean of animals (current / trial guard)

| epoch | close ipsi | close mid | close contra | far ipsi | far mid | far contra | pooled |
|---|---|---|---|---|---|---|---|
| pre | .980 / .978 | .979 / .979 | .981 / .968 | .898 / .875 | .937 / .912 | .958 / .932 | .956 / .941 |
| acute | .933 / .930 | .915 / .910 | .916 / .845 | .739 / .662 | .470 / .406 | .054 / .048 | .691 / .631 |
| subacute | .944 / .937 | .954 / .946 | .948 / .905 | .880 / .840 | .903 / .827 | .748 / .698 | .897 / .858 |
| chronic | .978 / .976 | .980 / .980 | .984 / .969 | .957 / .938 | .981 / .962 | .967 / .952 | .975 / .963 |

- Block guard and no-trim columns: `stop_rule_guard_cohort_hit.csv`.
- Per animal x spout: `stop_rule_guard_by_animal_epoch_spout.csv` (`hit`, `cga`, `denom`, `contact`, ...).
- Per session: `stop_rule_guard_by_session_spout.csv`.

### T4. Deficit vs the animal's own August pre, cohort mean (current / trial guard / block guard)

| epoch | close ipsi | close mid | close contra | far ipsi | far mid | far contra | pooled |
|---|---|---|---|---|---|---|---|
| acute | -.047 / -.049 / -.059 | -.065 / -.069 / -.094 | -.065 / -.122 / -.140 | -.159 / -.212 / -.240 | -.467 / -.507 / -.527 | -.904 / -.884 / -.879 | -.265 / -.310 / -.325 |
| subacute | -.036 / -.042 / -.064 | -.026 / -.032 / -.041 | -.032 / -.063 / -.071 | -.018 / -.035 / -.052 | -.034 / -.085 / -.099 | -.210 / -.234 / -.241 | -.059 / -.083 / -.095 |
| chronic | -.003 / -.002 / -.002 | .001 / .001 / .000 | .003 / .001 / -.003 | .059 / .063 / .059 | .044 / .050 / .045 | .009 / .020 / .017 | .018 / .022 / .019 |

### T5. Contact-given-attempt

Identical under all rules in every animal x epoch x spout cell, because the working contact / incomplete set does not change. Example values (current = guard):

| animal | spout | acute | pre |
|---|---|---|---|
| PS94 | far mid | .350 | .993 |
| PS94 | far contra | .048 | .975 |
| PS92 | far contra | .054 | .986 |

### T6. Sessions where the rules differ

- Trial guard: **57 / 107** (pre 14 / 31, acute 16 / 16, subacute 15 / 17, chronic 12 / 43).
- Block guard: 64 / 107.

Five largest differences, trial guard (all are trials moved from stopped to no_response / jaw_only):

| session | epoch | trials changed | trial range (index of n) | spouts of changed trials |
|---|---|---|---|---|
| PS94_0817 | acute | 84 | 214-642 of 643 | close contra 5, far ipsi 33, far mid 7, far contra 39 (+2 after_end) |
| PS95_0823 | subacute | 78 | 315-588 of 600 | close ipsi 1, close mid 3, close contra 18, far ipsi 15, far mid 15, far contra 26 |
| PS92_0818 | acute | 77 | 203-542 of 543 | close ipsi 2, close contra 13, far ipsi 20, far mid 31, far contra 11 |
| PS95_0806 | **pre** | 75 | 447-572 of 573 | close contra 15, far ipsi 23, far mid 16, far contra 21 (55 of 75 jaw_only) |
| PS94_0820 | acute | 71 | 281-551 of 600 | close contra 17, far ipsi 15, far mid 18, far contra 21 |

- PS94_0817 has eleven no-reference runs of 6-11 trials, mostly far contra + far ipsi.
- PS95_0806 is pre: three no-reference runs of 12-24 trials. That is disengagement, not a deficit.
- All differences per session (both guards): `stop_rule_guard_session_diffs.csv`.
- Every current stop run, with whether it has a reference trial, the head before the first reference trial, and its spouts: `stop_rule_guard_current_runs.csv`.
- The 85 no-reference runs with the class of the reference trial before / after: `stop_rule_guard_noref_runs.csv`.

**Sessions where the trial guard has no end cut** (no reference trial after the last lick): PS92_0814, PS92_0818, PS92_0819, PS93_0811, PS93_0818, PS94_0810, PS94_0812, PS94_0817, PS94_0921, PS95_0806, PS95_0817, PS95_0827, PS95_0831. In those sessions, a failure tail at non-reference spouts stays in as no_response / jaw_only. Example: PS92_0818, 19 trials at close contra / far spouts at the very end.

### T7. Effect on decoder inputs (DAQ cue train, `cue_classes_for`'s matching)

| set | changes? |
|---|---|
| working contact + incomplete (what the decoders train / score on) | **no**, 107 / 107 identical (trial and block guard) |
| rest period (first to last camera lick) | **no**, 107 / 107 identical |
| `stopped` set (no-lick decoder stopped arm; ENL working vs stopped) | **yes**: 5,882 to 4,152 (trial) / 3,265 (block) |
| `after_end` | 43 to 30 / 17 |
| jaw_only stopped | 2,329 to 1,453 / 1,084 |

My re-implementation of the current rule matches `TC.classify_session` and `TC.cue_classes_for` exactly in all 107 sessions (asserted per session).

## Method

Rule sources:
- `wfield_local/trial_classes.py` (wf_rebuild): `REFERENCE_SPOUTS = ("close_L", "close_center")` :88, the existing "close" end-anchor reference (barely affected by the stroke), used here unchanged as close ipsi + close mid; `before_session_end` :358; `_lick_span` :583; `classify_session` :632; `stop_run_min` :808 (6); `stop_min_start_s` :817 (300 s); `jaw_counts_in_stop` :841 (True under in_working); `cue_classes_for` :902.
- `scripts/trial_attempts.py` :70-100: the attempt-rate denominator = not in `RUN_EXCLUDED` (:30).
- Profile `configs/render_profiles/cohort1_rebuild_20261103.yaml` applied with `render_profile.apply` before the imports (asserted: tag ends `stopRun6j`).

Per session:
1. `TC.session_trials(label, "cue_stop")`, sorted by cue time. Raw classes contact / incomplete / jaw_only / no_response / unknown.
2. **Current** = `TC.classify_session(raw, None, pos, rule="run", min_run=6, end="lick", cue_s, min_start_s=300, jaw_counts=True)`, the same call as `trial_attempts.py:94`.
3. **Guard, trial-level**, reimplemented in scratch with the same loop as `classify_session`:
   - End = the first reference-spout trial after the last camera lick. All trials from there on are after_end, unless they are stopped. With no reference trial after the last lick there is no end cut.
   - A run of consecutive no_response / jaw_only trials (unknown skipped, a lick ends it) is stopped **from its first reference-spout trial onward**. The part from there must still be >= 6 trials and start >= 300 s after the first cue; otherwise it is early_run. Trials of the run before its first reference trial stay no_response / jaw_only. A run with no reference trial is never stopped.
4. **Guard, block-level:**
   - End = the first trial of the first reference-spout block that starts after the last lick.
   - A run qualifies only if it contains a complete reference block. It is stopped from that block's first trial, and that part must still be >= 6 trials.
5. **Guard, trial-level, no trim** (sensitivity column `g_trial_notrim`): a run with any reference trial is stopped whole.
6. The decoder check repeats 2-4 on the DAQ cue train with video coverage (`daq_cue_seconds`, `cue_classes`, matched flags, `_effective_coverage`). It compares the camera_lick-in-period set, the `_lick_span` rest set and the stopped set.

Scripts and raw outputs (scratch, not committed):
- `C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/srg/census.py`
- `C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/srg/analyze.py`
- `C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/srg/decomp.py`
- `C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/srg/more.py`
- Per-trial classes under all four rules: `C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/srg/trials.csv`

## Uncertain / open

- **Where a stop starts.** For runs that contain a reference trial, the stop could start at the first reference failure (trimmed, used here) or at the run's start (no trim). This choice accounts for about half of what the guard puts back. It needs Priya's call.
- **Disengagement in the guarded set.** The guard keeps some disengagement as "failures", as pre shows. The bracket test (a reference-spout contact on both sides of the run) is the obvious tightening and is not implemented.
- **Sessions with no end cut.** In the 13 sessions without a reference trial after the last lick, the guard applies no end cut. A real quit that happened to fall during non-reference blocks would count as failures.
- **close mid.** Its no_response is a default (the jaw cannot be judged there; `jaw_judged` = False). Using close mid as a reference makes the guard treat an unjudged no_response there as evidence of stopping.
