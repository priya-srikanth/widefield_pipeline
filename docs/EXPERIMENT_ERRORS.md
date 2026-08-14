# Experiment errors and acquisition incidents — running log

Acquisition and rig problems that constrain what can be analysed, and what we decided to do about each.
**This is about the EXPERIMENT, not the code.** Analysis/method errors live in
[`DECISIONS.md`](../DECISIONS.md) and [`PREPROCESSING_DECISION.md`](PREPROCESSING_DECISION.md).

Newest first. Each entry: what happened → what it costs → what we do → what is still open.

---

## 2026-08-13 — PS95: blue LED only (no 415 alternation) for the first ~32 min

**What happened.** Only the 470 nm LED was armed at session start; the 415 nm isosbestic channel did not
alternate until ~32 min in.

**What it costs.** For that period there is **no hemodynamic correction and no way to construct one**.
The 415 channel is the *measurement* of the hemodynamic artifact; without it there is nothing to
regress out. `rcoeffs` from the later alternating portion cannot rescue it either — the coefficient
converts a 415 *timecourse* into 470 units, and with no 415 samples there is no timecourse to convert.
So: **exclude the blue-only period from all GCaMP imaging analysis.** Behavior (DAQ + task log) and the
behavior cameras cover the full session and stay fully usable.

**Is there a workaround?** Not an honest one for anything at these timescales. Hemodynamic power sits at
0.1–13 Hz (vasomotion, breathing, heartbeat) and is 0.58–0.84 correlated with the raw 470 before
correction (measured, `hemo_residual_check`), so uncorrected blue is dominated by it. A model
predicting the artifact from the corrected portion's spatial structure would be unvalidated
extrapolation into exactly the period we cannot check. Not worth it for 32 min.

**⚠ OPERATIONAL RISK — flag before preprocessing.** The frame-pairing step (`trim_illuminated_labcams`
/ `cleanpairs`) infers frame identity from **LED parity**, i.e. it assumes strict 470/415 alternation.
A `.dat` whose first ~32 min is single-channel and whose remainder alternates can therefore corrupt the
frame map for the WHOLE session, not just the prefix — a half-frame offset propagating through
everything. The blue-only prefix should be **split off before preprocessing**, not trimmed afterwards.
Also note the blue-only period is effectively double frame rate for that channel, which is a second
reason its samples cannot simply be concatenated with the alternating remainder.

---

## 2026-08-13 — PS93: behavior camera videos not recorded

**What happened.** The behavior cameras did not record. The behavior log and the DAQ stream are intact.

**What it costs.** No camera-based analysis for this session — no video-derived movement regressors, no
DLC/pose, and nothing for the movement-potent/movement-null subspace work. Behavior events (licks,
trials, positions, latencies) come from the DAQ and are unaffected, so the session remains fully usable
for behavior and for imaging analyses that do not need video.

**Still open — needs Priya's clarification.** The note says "we should still analyse camera data with
behavior events". If the cameras produced nothing for PS93, there is no camera data to analyse for it;
this may mean (a) analyse the *other* animals' camera data as usual, or (b) some partial camera data
does exist. Not assumed either way — ask before acting.

---

## 2026-08-12 — PS92: imaging computer froze mid-session; day recorded as two acquisitions

**What happened.** The acquisition crashed after ~9.6 min (`PS92_20260812_151741`,
`recording_complete=false`), ~41 min went unrecorded during the restart, and a second acquisition ran
87 min (`PS92_20260812_161728`). The imaging box joined them with `concat_split_session` into
`PS92_20260812_concat`, keeping 19,742 clean frame-pairs from part 1 (of 29,112 recorded) plus all
163,594 of part 2 = 183,336, and zero-padding the 40.7 min gap so the sample timeline stays
wall-clock accurate.

**What it costs.** 225 cues versus 304–472 for other PS92 sessions — **the short day is real, not a
processing loss.** Verified at both levels: frames (183,276 clean pairs vs 183,336 declared, normal
attrition) and events (24 + 201 = 225 exactly). Consequences: per-session decode/encode/RDM estimates
for this session are noisier than its siblings' and should be weighted accordingly rather than read as
a change in the animal; the 40.7 min dark gap also makes its photobleaching trend non-comparable; and
its detrend mask-eligible fraction is the lowest of the curated set (11.4% vs a 25.3% median), which is
expected given the zero-padded stretch.

**Also note.** The concat folder deliberately carries NO camlog — part 1's is truncated by hundreds of
frames — so the `.dat` plus the DAQ `pco` pulses are the authoritative frame record. Always use the
`_concat` h5, never either segment's own. Full provenance is in `configs/sessions.yaml` under PS92
`"0812"`.

**REQUESTED (Priya, 2026-08-13) — a behavior-only figure spanning the FULL session.** The behavior log
and cameras cover the 41 min the imaging missed, so there is more behavior than imaging for this day. A
separate behavior figure over the complete concatenated behavior record would recover that.

*Blocking question before building it:* the imaging-side DAQ almost certainly does not cover the gap
(same machine that crashed), so a full-session behavior figure has to come from the **task-controller
log** rather than from `daq_trials`. But the GUI log **mislabels `pos_idx` on ~15% of trials** — every
position-change trial — which is precisely why the pipeline switched to DAQ-derived positions
(`docs/GUI_TRIALS_LOGGING.md`). It was fixed upstream in `mobile_spout_behavior` bb16533, but only for
sessions recorded after that deployed. **Was bb16533 deployed before 8/12?** If yes, the log's
positions are trustworthy and the figure is straightforward. If no, the gap's trials can be plotted for
*timing, licking and hit rate* but their *position labels* are unreliable and any per-position panel
over the gap would be wrong.

*Video in the gap:* deferred. Priya's read is that behavior video is most valuable where there is
concomitant neural recording, which by definition excludes the gap.

---

## Conventions for this log

* Add the entry when the incident is found, not when it is resolved — a known-but-unfixed problem that
  is written down is safe; one that is only remembered is not.
* Say what it COSTS in analysis terms, not just what happened, so a reader can tell whether a given
  figure is affected.
* Record rejected workarounds and why, so they are not re-proposed.
* Anything needing a human decision goes under "still open" with the specific question.
