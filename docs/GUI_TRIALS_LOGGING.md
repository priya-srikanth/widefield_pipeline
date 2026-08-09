# GUI `trials.csv` mislabels `pos_idx` — why behavior trials now come from the DAQ

**Status:** diagnosed 2026-08-09, worked around here (DAQ-primary trials) **and fixed upstream** in
`mobile_spout_behavior` as **GUI `v47`** (`bb16533` + `bce483d`; `v46` is kept exactly as it shipped).
The rigs move to `v47` from 2026-08-09 onward — **record the first session date here once it is
running**, because sessions recorded before it keep the corrupted column, which is every session to
date. The DAQ-primary path stays necessary regardless: it covers the old sessions, and it is what
makes behavior and imaging agree on trial identity.

Once `v47` is running, `trials.csv pos_idx` should agree with the DAQ strobe codes trial-for-trial.
That is worth confirming on the first session (the same comparison used to find this: decode the
strobe codes, pair each cue with the last strobe at or before it, and check agreement is 1.000).

The same firmware trial-id lag exists on the 2pRAM and GB219 Teensy rigs, so **their `trials.csv`
carries the identical corruption** and `v47` fixes them too. Anything analysing those datasets from
`pos_idx` needs the same treatment — and unlike the widefield rig they may have no DAQ strobe record
to fall back on, in which case invert the shift (below) instead.

## Symptom

Per-trial position in the task-controller log disagrees with the DAQ strobe code on **~15% of
trials in nearly every session**, across all four animals and both June and August blocks.

## Root cause — the GUI, not the DAQ

A **trial-id collision**, not a stale position readout. The firmware emits `trial_start` BEFORE
`totalTrials++`, so it reports the PREVIOUS trial's id, while that same trial's cue/hit/reward
events (emitted after the increment) report id+1. Straight from a real `events.csv`:

```
trial_start    trial_id=0     <- trial 1 begins
cue/reward/hit trial_id=1     <- trial 1's body
trial_start    trial_id=1     <- trial 2 begins, SAME id as the open row
```

`_update_trial_from_event_row` finalized the open row only when the ids **differed**, so trial 2's
`trial_start` merged into trial 1's still-open row — and the position lines below it then overwrote
`pos_idx` / `pos_name` / `pos_dist_mm_after_trial` with trial 2's position (the device sets
`currentTrialPos` immediately before emitting `trial_start`).

So the corruption is a **uniform one-trial shift**: `gui[N] == true[N+1]` for every N. It only looks
sporadic because it is invisible when consecutive trials share a position — which is exactly why the
measured mismatch rate (~15%) equals the position-change rate, and why `gui[k-1] == daq[k]` on 100%
of mismatches. Two consequences worth keeping in mind:

* **The existing logs are invertible**: `true[N] = gui[N-1]`. Only the first trial is unrecoverable,
  and the last row is uncorrupted (no trial follows it) as a free consistency check.
* **An integer trial-offset aligner absorbs it.** That is why the imaging side was never affected —
  see "What this did NOT affect" below.

Fixed upstream in `mobile_spout_behavior` (commit `bb16533`): a `trial_start` now always finalizes
the open row and starts a new one, whatever id the device reports. Note that stamping/locking the
position fields — the fix this document previously recommended — would **not** have worked: the
offending `trial_start` carries an explicit `pos=` too. Closing the row is what makes it correct.

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

- **Every per-session figure and the cohort CSV had to be REGENERATED, not appended to** — positions
  move on ~15% of trials and hit/miss is re-scored under the real 3.5 s window. DONE 2026-08-09 for
  the curated set (0606-0608, 0806, 0807): 20 sessions x 2 figures + cohort.
- Sessions whose `trials.csv`/`events.csv` were never written now analyse from the DAQ alone:
  **PS93 6/6** (488 trials in `summary_end.json`, 0 rows written) — recovered, 493 trials, all 6
  positions; also PS94 6/3. PS93 8/5 has manually-labelled positions (`behavior_trials` in
  `sessions.yaml`), which take priority over both the DAQ and the log.

## What this did NOT affect: the imaging / LocaNMF analysis

**No re-run of preprocessing, LocaNMF decode/encode/RSA, or either deck was needed.** Two reasons:

1. **Healthy-DAQ sessions never read the log.** Both consumers bail out early when the strobe gives
   all 6 positions — `framemap_event_maps._behavior_cue_codes` ("DAQ strobe healthy (>=6 positions);
   using DAQ cue codes (trials.csv ignored)") and `behavior_position.classify_cues_with_backup`
   (`if len(good) >= 6: return daq`). That covers all of June and 8/7 onward.
2. **On the dead-bit sessions the offset aligner absorbs the shift.** 8/5-8/6 DO consult the log,
   but both aligners search an integer trial offset, and this corruption is a uniform one-trial
   shift — so the search lands on the true assignment instead of being defeated by it.
   `classify_cues_with_backup` validated at **agreement 1.000 at offset +1** on all seven dead-bit
   sessions that have a log (that `+1` IS the shift being absorbed), and the maps path produced all
   6 positions with per-position counts summing to the cue count (e.g. PS93 8/6:
   93+92+87+82+87+89 = 530).

The behavior report was the only consumer that read the corrupted column **without** an
offset-search safety net — which is why it, and only it, needed the fix and the regeneration.

## The upstream GUI fix (`mobile_spout_behavior` bb16533)

A `trial_start` now always finalizes the open trial row and starts a new one, whatever trial id the
device reports. Ships with `tests/test_trial_logging.py`, which replays the real device event
ordering through `SessionLogger`; its two bug-targeting tests fail on unpatched v46.

Not fixed there (pre-existing, cosmetic): `trial_id` in `trials.csv` still lags the device's own
trial number by one, and each trial still yields an unscored `trial_start` row alongside its scored
row. Both follow from `totalTrials++` happening after the trial_start emit, so fixing them means a
firmware change. Neither affects analysis (we key on the DAQ, and scored rows are filtered by
hit XOR miss).
