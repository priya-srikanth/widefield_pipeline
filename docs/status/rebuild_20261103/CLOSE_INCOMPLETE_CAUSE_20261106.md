# Why close-spout INCOMPLETE counts rise 3-5x in the re-run of kinematics section 6 (Figs 6.1 / 6.2)

Read-only investigation, 2026-11-06 evening. Code: `C:/Users/SabatiniLab/Github/wf_render_v9k` = ec0ff642.
Scripts and per-trial tables: `scratchpad/kin3d/cc3.py` (5 sessions), `cc5.py` (all 107 sessions), `cc4.py` / `cc6.py`
(summaries), `cc3_trials.csv` / `cc5_trials.csv`, `old_tc_23c8.py` (= `git show 23c8f775:wfield_local/trial_classes.py`).

## Verdict

**Different trials are included; the class definition did not cause it.** The extra close-spout incomplete trials are
trials AFTER the behaviour pipeline's DAQ quit. The old gate (`orofacial_cohort.engagement_keep`, the behaviour
`engaged` column, a run of DAQ misses = quit) cut them. The camera working period runs to the session's LAST CAMERA
LICK. A camera lick that misses the spout (incomplete) therefore keeps the post-DAQ-quit tail in. In that tail the
animal still licks but no longer touches the spout. This is exactly the approved rule, so it is a **correct
consequence of the gate, not a bug**:
* Priya 11-03 / 11-03 late: working = contact + incomplete camera licks, cut at the last camera lick, and the DAQ
  quit-tail rule is demoted to a robustness arm (DECISIONS 2026-11-03 "End-of-session cutoff" and "Working excludes
  jaw-only").
* A20, Priya 11-06: kinematics uses the camera working period.

## Where in the code

* `scripts/jaw_only_incomplete_figs.py` (ec0ff642) lines 138-146, `session_job`. With `kin_gate() == "working"` (the
  default since commit **41f47e15**, "Priya 11-06 A20-A23"), the trials are
  `TC.working_period_trials(s)`. Before that commit they were `TC.session_trials(...)` filtered by
  `oc.engagement_keep(...)`. That old path is now reachable only with `WFIELD_KIN_GATE=legacy`.
* `wfield_local/trial_classes.py` lines 1360-1380, `working_period_trials`. It calls `cue_classes_for` →
  `classify_session` (end = last camera lick, A2 stopped runs and excision), then `in_working_period`.
  * `in_working_period` (line 1300) keeps every trial whose class is not in `OUT_OF_WORKING_PERIOD` (line 1297).
  * `OUT_OF_WORKING_PERIOD` = stopped, stopped_licking, early_run, after_end, unknown.
* Old page tables: `session_poses/cam1_licks/trial_classes/sessions.csv` (11-03 09:30). They were produced by
  23c8f775-era code: `session_trials` + engaged gate.

## Evidence 1: per-trial, 5 sessions (2 animals, pre + acute)

The old and new per-trial classes were recomputed from identical inputs:
* `jaw_onsets.csv` has the same md5 in the default tree and in the rebuild seed;
* v36 `licks.csv` / `trials.csv` date from 10-20;
* cam1 `_events.csv` dates from 10-21.

Both reproduce the published tables exactly: every `n_incomplete` cell matches OLD `sessions.csv` and NEW
`sessions.csv`.

Close spouts (close_L + close_center + close_R):

| session | epoch | OLD incomplete / trials (engaged) | NEW incomplete / trials (working) | last engaged trial id → last working trial id |
|---|---|---|---|---|
| PS92_0810 | pre | 0 / 199 | 0 / 199 | 404 → 404 (no quit) |
| PS92_0819 | acute | 0 / 199 | **7** / 217 | 410 → 540 |
| PS94_0811 | pre | 0 / 200 | 1 / 204 | 399 → 411 |
| PS94_0817 | acute | 11 / 324 | 11 / 294 | 643 → 641 (no DAQ quit) |
| PS94_0823 | acute | 0 / 184 | **18** / 227 | 390 → 587 |

* **Class definition: 0 changes at the close spouts.** On every close-spout trial of these sessions, old
  `session_trials` (23c8f775) and new `session_trials` (ec0ff642) give the same class.
  * This holds despite the 11-04 response-window rules (9f0d6efe): licks judged by peak at cue + 70 ms .. min(cue +
    3.5 s, stop), and contacts after cue + 3.5 s dropped.
  * Contact, incomplete, jaw_only and no_response crosstab: diagonal only.
* **All 26 added incomplete trials** were `engaged == False` and lie after the last engaged trial: 22 / 22 and 72 / 72
  of the not-engaged working trials in PS92_0819 and PS94_0823 come after the DAQ quit. None lies mid-session.
  * Each added trial was already class incomplete under the old definition: n_contacts 0, 1-11 video licks.
  * Their trial stop is about 3.52-3.72 s, i.e. a full-length trial with no hit.
  * The old gate simply never counted them.
* PS94_0823 tail after the DAQ quit (all spouts, C = contact, I = incomplete, J = jaw-only, n = no response):
  `nnJnJJJnnJJnnnnJnIJJnJInJnnnnCnnnJInIJnnnJnnnJnJnnnnnnnnnInnnnCnCnnIIIIIJIInnInnIIIJIIICInnnJJ...`.
  * Repeated incomplete runs with scattered contacts continue for about 200 trials after the DAQ quit.
  * Stopped runs inside the tail are excised. The rest, up to the last camera lick, is working.
* **The denominator moves both ways.**
  * Added: post-DAQ-quit trials up to the last camera lick. PS94_0823 gains 72 trials: 32 incomplete, 23 no_response,
    9 contact, 8 jaw_only.
  * Removed: engaged no_response / jaw_only trials in sustained stops (A2) or after the last camera lick. PS94_0817
    loses 49 trials: 34 no_response, 15 jaw_only.
  * This is why no_response / jaw_only shares fall while incomplete rises.

## Evidence 2: whole cohort (all 107 sessions)

COHORT_PLACEHOLDER

## Reading for the page

* "More incomplete trials exist" is false: no trial changed class.
* "Different trials are included" is the whole story: the post-DAQ-quit tail, in which licks keep coming but miss the
  spout.
* The tail is long mainly in acute / subacute sessions, which is why the pooled close-spout incomplete change now
  reaches * / ** acute and subacute.
* It is a real behavioural fact under the approved gate: incomplete licks after the animal stops hitting. It is not
  necessarily a close-spout aiming deficit. The section 6 text should describe it that way; for example, the close-spout
  incomplete rise is carried by late-session no-contact licking after the DAQ quit.
* A sensitivity row is already available as `WFIELD_KIN_GATE=legacy` (writes `jaw_only_incomplete_figs_legacygate`) or
  the "contact" end anchor. No code fix is needed.
