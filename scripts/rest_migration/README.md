# The measurements behind `segmentation.rest`

Every parameter of the REST definition was set by one of these, on 2026-09-12. They are kept so the
numbers in `configs/defaults.yaml` and `docs/REST_BASELINE_MIGRATION.md` can be re-derived rather
than trusted, and so the same checks can be re-run if the task timing ever changes.

Run any of them from the repo root with the `locanmf` env. All are read-only.

| script | answers | what it found |
|---|---|---|
| `trial_anchors.py` | Where are the trial's real boundaries? | `trial_start` precedes the spout strobe on **16,607 of 16,607** trials, median 0.925 s — that gap is the spout's travel time. So rest must END at `trial_start`: the strobe fires AFTER the movement. The interval from the response window's close to the next `trial_start` is median **2.47 s** and **never negative**. |
| `settle.py` | How long a settle after the response window? | Reward lands **6 ms** after the cue (max 0.153 s, none after the window closes), so the trial window already contains every reward and the settle is pure margin. At 0.5 s, **100%** of trials still admit a 1 s segment and rest per trial is 2.02 s, against 1.52 s at 1.0 s. |
| `bout_durations.py` | How long are rest bouts, per candidate definition? | The retired definition gives a **1.15 s** median — reproducing the 1.10 s that `BEHAVIOURAL_STATE_CONTROL.md` derived independently, and its 82% 2 s-discard against that doc's 83%. Confirms the state decoder's 1 s window. |
| `rest_smoke.py` | Does the wired definition behave on real sessions? | **Caught a units bug**: `trial_exclusion` converted seconds to samples as `a / sr * fs` with `sr = fs`, i.e. not at all, so almost nothing was excluded and rest measured 62% of an acute session — against 36% for the variant that applies NO trial exclusion. An impossible number is why this script exists. |

Two more measurements live in the session scratchpad rather than here because they sweep all 92
sessions and take ~80 minutes: the quiet-variant comparison (which established acute/pre = 3.89 for
the retired definition against 1.10 for the trial-anchored one, and that **licking**, not reward,
is the term that moves with epoch) and the between-animal consistency null. Their results are
recorded in `DECISIONS.md` and `docs/REST_BASELINE_MIGRATION.md`.

**These are diagnostics, not pipeline steps.** Nothing in `nightly` calls them.
