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

**MEASURED SPLIT POINT (`wfield_local.led_alternation_qc`, run 2026-08-13).** The DAQ carries
`led415_ttl` / `led470_ttl` (analog) and `pco_exposure` (digital), so each camera exposure can be
labelled with the LED that fired — no reliance on image content, no assumption of alternation, and
frame *i* of the `.dat` is exposure pulse *i*:

```
532220 exposures over 142.2 min   415=206724  470=325496  neither=0  both=0
first 415 exposure: frame 119104 at 32.00 min
-> SPLIT AT FRAME 119104 (starts on 415: True), keeping 413116 frames
after the split: 415=206724  470=206392   332 phase slips (0.08%)
```

**Split at frame 119104.** It lands on a 415 frame, so pairing stays (415, 470) as
`functional_channel: 1` expects — starting on the wrong parity would swap the channels for the whole
session and look like a plausible dataset rather than an error. The 332 phase slips in 413,116 frames
are ordinary dropped-frame attrition that `cleanpairs` already handles. (An earlier version of the QC
reported "205,222 violations" by comparing against an ideal 0,1,0,1 phase — a single dropped frame
flips the phase for everything after it, so that metric is useless for telling a healthy recording from
a broken one. It now counts adjacent repeats.)

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

**Clarified (Priya):** the four behavior-camera `.avi` files are missing; the **neural imaging `.dat`
is present and fine.**

**What it costs.** Nothing for imaging — this session is fully usable for preprocessing, LocaNMF,
decoding and encoding. What is lost is video-derived movement regressors, DLC/pose, and this session's
contribution to the movement-potent / movement-null subspace analysis. Behavior events (licks, trials,
positions, latencies) come from the DAQ and are unaffected.

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

*RESOLVED — the log's positions are trustworthy (Priya: fixed as of v47; verified 2026-08-13).* The
concern was that the GUI log historically mislabelled `pos_idx` on ~15% of trials, shifting each label
by ONE trial (`docs/GUI_TRIALS_LOGGING.md`), which is why the pipeline moved to DAQ-derived positions.
Tested directly against `daq_trials.positions_for_cues` on three recent sessions with full DAQ coverage:

| session | aligned agreement | agreement if GUI shifted +1 trial |
|---|---|---|
| PS93 8/12 | **0.996** | 0.827 |
| PS95 8/11 | **0.990** | 0.827 |
| PS94 8/11 | **0.984** | 0.818 |

If the bug were still present the SHIFTED column would be the high one. It is not, decisively. So the
full-session figure can use the log's positions including per-position panels. The concatenated log
already exists (`Behavior_logs/Widefield/PS92_20260812_concat/trials.csv`, **563 trials vs 225 DAQ
cues**), so the gap's behavior is available to plot.

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
