# GUI `trials.csv` mislabels `pos_idx` — why behavior trials now come from the DAQ

**Status:** diagnosed and worked around here (2026-08-09). **The GUI bug itself is NOT fixed** —
it is present in the current `BehaviorGUI_MobileSpouts_Arduino_vs_Teensy_v46.py`, so every session
recorded to date is affected and new ones will be until the GUI is patched.

## Symptom

Per-trial position in the task-controller log disagrees with the DAQ strobe code on **~15% of
trials in nearly every session**, across all four animals and both June and August blocks.

## Root cause — the GUI, not the DAQ

The GUI builds each trial row incrementally and lets **every later event row overwrite** the
position field while the row is still open
(`_update_trial_from_event_row`, [v46:507-508](../../Behavior_setup/BehaviorGUI_MobileSpouts_Arduino_vs_Teensy_v46.py)):

```python
if row.get("pos_idx", "") != "":
    t["pos_idx"] = row.get("pos_idx", t.get("pos_idx", ""))
```

and an event carrying no explicit `pos`/`idx` falls back to the **live device status** ([v46:548](../../Behavior_setup/BehaviorGUI_MobileSpouts_Arduino_vs_Teensy_v46.py)):

```python
pos_idx = kv.get("pos", kv.get("idx", latest_status.get("current_pos", "")))
```

A trial row is only finalized when the **next** `trial_start` arrives, so any event landing after
the firmware has already moved to the next position stamps that next position onto the previous,
still-open row. The row ends up **one trial ahead**.

Measured on sessions with complete logs: the disagreement hits **exactly** the trials where the
position changed (rate 1.000, vs 0.008 on unchanged trials) and `gui[k-1] == daq[k]` on **100%** of
mismatches. It is invisible when consecutive trials share a position, which is why it went unnoticed.

`pos_dist_mm_after_trial` is corrupted by the same overwrite ([v46:512-515](../../Behavior_setup/BehaviorGUI_MobileSpouts_Arduino_vs_Teensy_v46.py)) — an apparent
"spout moved mid-trial" in the log is a logging artifact, not a real move.

**The DAQ is correct by construction.** Firmware `startTrial` moves the spout, *then* calls
`emitPositionCode(currentTrialPos)`, which sets `spout_bit0/1/2` and only then pulses a 10 ms
`spout_strobe` (`Behavior_MobileSpouts_Zaber_Arduino_v36.ino:1541-1550`, `:1072-1077`). One strobe
per trial, bits already settled, ~3 s before that trial's cue. Verified stable: the decoded code is
identical sampled at the strobe edge, strobe+1 s, and at the cue.

## Fix here: `wfield_local/daq_trials.py`, DAQ-primary with the log as fallback

`spout_behavior.load_trials` now sources trials from the DAQ recorder `.h5`:

| field | source |
|---|---|
| `pos_idx` | `spout_strobe` + `spout_bit0/1/2`, paired to each cue by **time** (last strobe at or before it) |
| `hit` / `miss` / `latency` | DAQ licks (canonical `lick_detection` params) in `[cue, min(cue+window, next_cue)]` |
| response window | **per session** from its `gui_config.json` `timing.response_window` (ms, string) |
| anticipatory licks | licks in the pre-cue ENL window `[trial_start, cue)` |
| reward | `reward_ttl` rising edges |
| free-reward designation | merged from `trials.csv` (the DAQ cannot know it) |

Pairing is by time, not index, because `moveToNamedPosition` also emits a code on a manual move —
a session can legitimately carry extra strobes (PS94 6/8 has 628 strobes for 627 cues). This is the
same rule the imaging side already uses (`plot_spout_trial_averages._classify_cues`), so behavior
and imaging now resolve identical positions by construction.

### Why the response window had to change

The task ran **3500 ms** (`gui_config.json timing.response_window`); `configs/defaults.yaml`
carried **2.0 s**, which was never the task's window. It is read per session because it is a task
setting that can be retuned; `behavior.licking.response_window_s` is now only the fallback.

### Validation

DAQ scoring reproduces the GUI's own hit/miss **exactly** where the log is intact:

| session | DAQ | GUI (`summary_end.json`) |
|---|---|---|
| PS92 6/6 | 283/304 = 0.9309 | 283/304 = 0.9309 |
| PS94 6/8 | 604/627 = 0.9633 | 604/627 = 0.9633 |
| PS93 6/7 | 509/536 = 0.9496 | 507/536 = 0.9459 |
| PS93 6/6 | 454/493 = 0.9209 | 451/488 = 0.9242 (log broken, totals only) |

## The fallback matters: dead `spout_bit1` (Aug 2026)

The DAQ is *not* unconditionally right. On **8/5 and 8/6** `spout_bit1` was dead, so the strobe
codes collapse onto `{0,1,4,5}` — 4 distinct positions instead of 6. `daq_trials.quality()` rejects
those sessions (also: unpaired cues, out-of-range codes) and `load_trials` falls back to
`trials.csv`. Bit1 was fixed for 8/7 onward, and the 8/7 sessions gate clean at 6 positions. See
[`../STROBE_BIT1_RECOVERY.md`](../STROBE_BIT1_RECOVERY.md); that recovery path stays as backup.

So the policy is **DAQ primary, behavior log fallback — never DAQ-only.**

## Scope: which sessions change

Measured `pos_idx` disagreement per session (`==prev` = fraction of mismatches where the GUI row
holds the previous trial's position, i.e. the signature of this bug):

- **June 6/2–6/8, and 8/7 onward:** ~14–17% of trials mismatched, `==prev` = **1.00** → the GUI was
  wrong, the DAQ is now used. One exception: **PS94 6/8 = 0.0%** (the GUI happened to log cleanly).
- **8/5–8/6:** 42–48% mismatched, `==prev` ≈ 0.25 → *the DAQ* was wrong (dead bit1); these fall back
  to the log and are unchanged.

## Consequences

- **Every per-session figure and the cohort CSV must be REGENERATED, not appended to** — positions
  move on ~15% of trials and hit/miss is re-scored under the real 3.5 s window.
- **The imaging analysis is unaffected** — it already read positions from the DAQ strobe. This
  change removes a disagreement between the two halves of the pipeline, it does not introduce one.
- Sessions whose `trials.csv`/`events.csv` were never written now analyse from the DAQ alone:
  **PS93 6/6** (488 trials logged in `summary_end.json`, 0 rows written), PS94 6/3, PS93 8/5.

## Recommended GUI fix (upstream, not done here)

Stamp `pos_idx`/`pos_name`/`pos_dist_mm_before_trial` **once** from the `trial_start` event and do
not let later event rows overwrite them; only `pos_dist_mm_after_trial` should track later updates.
