# Experiment errors and acquisition incidents — running log

Acquisition and rig problems that constrain what can be analysed, and what we decided to do about each.
**This is about the EXPERIMENT, not the code.** Analysis/method errors live in
[`DECISIONS.md`](../DECISIONS.md) and [`PREPROCESSING_DECISION.md`](PREPROCESSING_DECISION.md).

Newest first. Each entry: what happened → what it costs → what we do → what is still open.

---

## 2026-08-28 — PS92: 415/470 excitation channels swapped (imaging only; behavior intact)

**What happened.** For `PS92_20260828` the 415 nm (isosbestic) and 470 nm (functional) excitation
channels were swapped — visible on the preprocessing-deck mean-image slide. Motion correction is
correct (the `.dat` de-interleaves into 415/470 pairs fine); only the SVD + hemodynamic-correction step
took the wrong half of each pair as functional (`svd.functional_channel: 1` in `defaults.yaml`), so
`SVTcorr.npy` — and everything derived from it — is computed on the isosbestic with the hemodynamic
correction running backwards.

**What it costs.** Every IMAGING-derived result for PS92 8/28 is invalid: the SVD/activity maps, LocaNMF
components, within-session decode (still 0.78 — a fresh decoder finds *some* structure in the corrupted
signal), and most visibly the frozen pre-stroke decoder, which reads **0.016 with a systematic close↔far
permutation** (grant_5c: cL→fL, cC→fR, cR→fC, fL→cC, fC→cR, fR→cC) rather than a near-chance smear. A
decoder trained on correct-channel pre-stroke data cannot read a swapped-channel session, so the
permutation is the tell that this is technical, not biological — a 2-day jump from day 9 (0826) = 0.91 to
day 11 (0828) = 0.016 confirms it. **BEHAVIOUR IS UNAFFECTED**: positions and licks come from the DAQ,
independent of the optical channels — PS92 8/28 hit rate is 1.0/1.0/1.0/0.91/0.98/0.98 over 311 engaged
trials. **PS93 8/28 is fine** (frozen 0.80), so the swap was PS92's session only, not the rig.

**What we do.** Fix at PREPROCESSING (imaging box), because the corruption is baked into `SVTcorr` and
cannot be relabelled downstream. Add a per-session override in `configs/session_overrides.yaml` flipping
`svd.functional_channel` 1→0 for PS92 8/28, then re-run ONLY the SVD + hemodynamic step with `--redo`
(it otherwise skips on the existing `SVTcorr` — `preprocess.py:567`); motion correction is reused. Re-upload
the corrected `SVTcorr`, re-fit LocaNMF, then the analysis box redoes ONLY PS92 8/28's affected
decode/encode/RSA/pattern figures and rebuilds the deck. Behaviour stays in throughout.

**What is still open.** As of 2026-08-29 the corrected preprocessing has not yet been produced; the
published deck still shows the corrupted PS92 8/28 (frozen 0.016, flipped maps). Corrected once the
imaging box re-runs the SVD step and the analysis is redone.

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
labelled with the LED that fired — no reliance on image content and no assumption of alternation.
(That frame *i* of the `.dat` is exposure pulse *i* was assumed here and is now **verified** — see
"Alignment verified" below; it does not follow from the counts agreeing.):

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

**⚠ SECOND, WORSE PROBLEM — the file is SINGLE-CHANNEL (found 2026-08-14).** Because one LED was
armed at start, labcams was configured for ONE channel and stayed that way for the whole recording,
even after 415 came on. The file is `..._1_460_480_uint16.dat` — 532,219 flat unpaired frames, the only
`_1_` file in the dataset against nine `_2_` ones.

This is more dangerous than the missing 415 alone: `run_wfield_local` decides whether to
hemodynamically correct by testing `dat.shape[1] == 2`, so a 1-channel file passes straight through and
yields an **UNCORRECTED `SVTcorr` with no error raised** — a silently wrong session rather than a
failed one.

**Recoverable.** 532,219 file frames match the DAQ's 532,220 exposures to within one, and the DAQ says
which LED fired on every exposure, so the alternating portion can be re-paired EXPLICITLY rather than
by assuming parity. `wfield_local/repair_single_channel.py` does this: drop the prefix, then emit a
pair only where a 415 is immediately followed by a 470, skipping the 332 phase slips instead of letting
one shift every pair after it. ~206,391 pairs survive, and the output order is (415, 470) to match
`functional_channel: 1`. It writes `frame_index_map.npy` (repaired-pair index → original exposure index)
because the DAQ `.h5` still describes the ORIGINAL sequence, and a `repair_manifest.json` so a repaired
session's provenance is never in doubt.

**ALIGNMENT VERIFIED, not assumed (2026-08-14).** The repair indexes the `.dat` by DAQ exposure number,
which is only valid if exposure *i* is file frame *i*. The counts agreeing (delta +1) does **not**
establish that: a dropped write mid-session plus a trailing unflushed exposure produces the same delta
while shifting every index after the drop — swapping 415 and 470 for the remainder and still yielding a
plausible dataset. Two independent checks, neither of which is a count:

*Camlog (the camera's own file).* labcams writes one `frame_id,timestamp` line per saved frame and the
LED controller interleaves `#LED:<state>,<counter>,<ms>` lines:

| quantity | camlog | DAQ |
|---|---|---|
| frames written | 532,219, **zero `frame_id` gaps** | — |
| exposures / LED commands | 532,220 LED lines | 532,220 `pco_exposure` pulses |
| first 415 | LED state 5 first at line **119,104** | `first_415_frame` = **119,104** |
| phase slips | 332-frame state-5/state-6 imbalance | **332** adjacent repeats |

*Pixels (decisive for the offset).* The camlog alone cannot settle the offset — its interleave is
`LL F L F L…`, so frame *k* follows LED line *k+1*, and that leading double-LED is equally explicable as
a logging race or a real extra exposure; the write-timestamp lag is ambiguous too (5.3 ms vs 21.3 ms
straddles the 16 ms frame period). So `verify_offset` reads the actual frames: 415 and 470 differ
grossly in mean intensity, and because the sequence alternates, a misaligned comparison is
ANTI-correlated rather than merely noisy.

```
offset -1: 0.001    offset 0: 1.000    offset +1: 0.001      (n=1600 frames)
pixel alternation onset 119,104 = DAQ split frame exactly (delta +0)
mean intensity  415 = 12,417   470 = 15,417   (violet is the dimmer channel, as expected)
```

Parity has period 2 and so cannot separate offset 0 from ±2; the onset check pins that, and **both must
pass**. This now runs automatically inside `repair()` and refuses to write on failure — the off-by-one
it guards against is silent by construction, so it must not depend on someone remembering to check.

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
already exists (`Behavior_logs/Widefield/PS92_20260812_concat/trials.csv`).

**CORRECTED 2026-08-14 — it is not "563 trials".** The concat log has 563 ROWS, but `trial_id` runs
0-282 with no resets: roughly two rows per trial (an open row and the scored row, the same
open-row behaviour that causes the `pos_idx` bug). The real accounting is **283 trials, 280 scored,
vs 225 DAQ cues** — so the DAQ recorder's crash cost ~55 trials, not the ~338 that "563 vs 225"
implied.

**Full-session figure: BUILT** (`PS92_20260812_concat_logsrc_behavior.png`). The existing
`PS92_20260812_concat_behavior.png` was misleading: DAQ-primary means it shows the 225 DAQ-covered
trials under a filename that says "concat". `spout_behavior` now takes an explicit trial source
(`load_trials(..., source="auto"|"daq"|"log")`) and a forced source gets its own `_logsrc` filename,
because the two figures show DIFFERENT trial sets and silently overwriting one with the other would
be undetectable.

**Limitation of the log-sourced figure: NO first-lick latencies** (all NaN). Latency is measured from
DAQ licks, and the log's timestamps are GUI milliseconds, not DAQ samples, so trials outside the DAQ
record cannot be given one. Accuracy and per-position hit rate cover the full session; latency and
the lick-microstructure metrics do not. Per-position accuracy from the log is trustworthy because v47
positions were verified against DAQ codes (0.984-0.996 aligned vs 0.818-0.827 shifted).

*Video in the gap:* deferred. Priya's read is that behavior video is most valuable where there is
concomitant neural recording, which by definition excludes the gap.

---

## 2026-08-13 — extreme quiet/running imbalance in three of four sessions

**What happened.** Not a fault, but a session property that changes what some panels mean. Quiet-frame
counts for 8/13: **PS92 255**, **PS94 2,488**, **PS95 14,326** (running frames 2,189 / 13,245 / 571).
PS92 was quiet on 0.1% of samples and PS95 on 7.4%, so the two sit at opposite extremes.

**What it costs.** The quiet-vs-running activity maps and any quiet-period BASELINE are estimated from
those frames. PS92 8/13's quiet map rests on 255 frames (~8 s of data) and should not be read
quantitatively; PS95 8/13's *running* map has the mirror-image problem at 571 frames. Cross-animal
comparison of quiet baselines on this date is not like-for-like.

**What we do.** Use them qualitatively, and prefer trial-referenced measures on this date. No pipeline
change: the counts are printed by `plot_running_activity_maps` on every run, which is how this was
noticed.

---

## 2026-08-05 / 2026-08-06 — DAQ `spout_bit1` dead: only 4 of 6 positions in the strobe

**What happened.** The DAQ strobe lost bit 1 for these two days, across **all four animals**. The
3-bit position code therefore collapses to 4 distinguishable positions instead of 6.

**What it costs.** Position identity cannot be read from the DAQ alone on these sessions, which is the
pipeline's primary source for BOTH behavior and imaging.

**What we do.** `classify_cues_with_backup` repairs the code from the behavior log's `pos_idx`, and
only when the repair validates at >=0.9 against the DAQ's still-good positions. `daq_trials.quality`
detects the collapse and falls back to the log for behavior scoring — never DAQ-only. **8/6 is kept**
in the curated set on that basis; **8/5 is excluded** for separate reasons (see below), so the two
should not be lumped together as "the bad LED days".

---

## 2026-08-05 — PS93: behavior log empty; positions recovered from the camera CSV

**What happened.** The task-controller log for PS93 8/5 came out empty, so the usual fallback for the
dead-`spout_bit1` repair was not available for that session.

**What it costs.** Without a second source there is no way to disambiguate the collapsed strobe code.

**What we do.** Positions come from a `behavior_trials` recovered CSV derived from cam1, treated as
READ-ONLY input (CLAUDE.md rule 1). This is the ONLY session using that path.

**Still open — 8/5 is excluded from the curated set anyway** ("the wonky 8/5"), so this recovery
currently affects no headline result. It matters if 8/5 is ever readmitted.

---

## 2026-08-13 — photobleach QC covers only PS93 (analysis-side sequencing error, not a rig fault)

**What happened.** The date's photobleach summary contains **1 of 4 sessions**. Photobleach memmaps the
RAW `.dat`, and on 14 Aug the 8/13 sessions were preprocessed one at a time with `--skip-photobleach`
(to avoid redoing per-date work four times) and each animal's local raw was cleaned immediately after
its `.bin` reached standby. By the time the date-level pass ran, only PS93's raw was still local.

**What it costs.** Nothing for the science — photobleach is a QC trend, and the raw is safe on standby.
The 8/13 photobleach slide simply shows PS93 alone (drift 415 −19.0%, 470 −14.2% over 166.9 min).

**RESOLVED 2026-08-14 — all four sessions are in the summary.** Re-staging was never necessary: the
raw was read STRAIGHT FROM STANDBY. Photobleach samples `NSAMP = 3000` frames (~1.3 GB), not the whole
164–257 GiB file, so the cost is 3,000 latency-bound scatter reads over SMB — about a minute per
session. `photobleach.run(..., merge=True)` then unioned the three with PS93's existing record, so the
date's summary rebuilt COMPLETE rather than replacing it. Backfilled values: PS92 415 −17.8% / 470
−14.5%; PS95 415 −16.8% / 470 −26.6% over 142 min. PS95 deliberately read its ORIGINAL single-channel
`_1_` file — channel labels come from the DAQ indexed by frame, and that indexing is verified offset-0,
so the figure honestly shows 415 BEGINNING 32 min in; the repaired `_2_` file would have been wrong
there, since its frame *k* is no longer exposure *k*.

The earlier judgement in this entry ("not worth recovering") was wrong because it assumed photobleach
reads the whole file. It reads 3,000 frames.

**What we do.** The summary is a pure function of the per-session records on disk, so the fix is
ordering: run photobleach BEFORE cleaning a session's raw. `archive_day clean` now WARNS by name when
it is about to delete a raw whose photobleach record does not exist. Not a refusal — the raw is going
to standby, not disappearing — but it can no longer happen silently.

---

## 2026-08-13 — PS95 repaired session: 23% of trials had NO imaging and were decoded anyway

**Found 2026-08-16** after Priya asked why one PS95 session decoded poorly. **This entry replaces an
earlier one that blamed PS95 8/12: that was my misreading — 8/12 decodes normally (0.88 cue, 510
engaged trials). The outlier is 8/13 (0.61 cue) — the REPAIRED single-channel session — and it has the
HIGHEST engaged count of the eleven (786), so it was never a trial-count effect.**

**What happened.** PS95 8/13 was recorded single-channel for its first 32 min; the repair correctly
dropped those 119,104 exposures and kept 206,391 pairs from the alternating remainder. But the DAQ
covers the whole 142 min, so **197 of 871 cues (23%) occur before any surviving imaging frame**.
`_nearest_corrected_frame` CLIPS rather than rejecting, so every one of those trials was assigned
frame ~0 and decoded as if it were real.

**What it cost.** Cue-aligned 0.61 vs ~0.90 for that animal; pre-cue 0.29 vs ~0.37. Roughly a quarter
of the session's trials were noise labelled with real positions.

**Why it was invisible.** For every normal session imaging spans the whole recording, so the clip never
bites and the code had been correct for 60 sessions. Only a session with a HOLE exposes it — and the
repair, which is what creates the hole, was written two days earlier. Its docstring warned that
"downstream frame-mapping must be told about the offset"; the frame MAP was fixed, the coverage check
was not.

**Fix.** `coverage_mask` marks events outside the imaging span, and `_frames` now returns -1 for them,
which every caller already skips. The exclusion PRINTS the count, so a session losing trials this way
announces itself. Tested with an explicit gap.

**Maps path.** `framemap_event_maps` used the same clipping helper directly. Its figures are trial
AVERAGES, so 23% of trials landing on frame 0 diluted rather than corrupted them — it now takes the
same guard, and 8/13's maps were regenerated (197/871 cues and 1,144/4,582 licks excluded).

**Verified 2026-08-17.** Cue-aligned went **0.61 → 0.78**, lick-aligned **0.65 → 0.85**, engaged
n 786 → 589. That is a large recovery but short of the ~0.90 predicted, so the obvious worry was a
SECOND fault in the repair. It is not:

* Positions stay balanced (82–116 per class) and no class is dead — recall 0.63–0.91 — so trial
  labelling is intact. A mislabelled position collapses one class, and none has.
* Lick-aligned recovers to within 0.04 of the animal's floor, so the retained movie is fine. A bad
  decomposition or a residual frame offset would hurt both alignments alike.
* Accuracy by trial-order third is **0.822 / 0.796 / 0.602** — the trials immediately after the
  repair seam are the BEST, not the worst. There is no boundary contamination, which is what a
  settling artifact or an edge effect in motion correction would have produced.
* PS95 declines across EVERY session (T3−T1 tilt −0.09 to −0.22 on 8/10–8/14). The LED fault
  destroyed 8/13's first 23% — precisely the portion that decodes best — so its session mean was
  being compared against controls that still had their good third. Matched on the same last-77%
  slice (ROI features): 8/10 0.843, 8/12 0.799, 8/11 0.778, 8/14 0.749, **8/13 0.740**. Within the
  normal spread, not an outlier.

So the residual is a real and correctly-handled consequence of the fault, not a further bug. **The
cost of the LED fault is that 8/13 contributes only its declining tail** — that is the caveat to
carry, rather than treating 0.78 as a defect to chase. Nothing about the repair needs revisiting.

---

## Conventions for this log

* Add the entry when the incident is found, not when it is resolved — a known-but-unfixed problem that
  is written down is safe; one that is only remembered is not.
* Say what it COSTS in analysis terms, not just what happened, so a reader can tell whether a given
  figure is affected.
* Record rejected workarounds and why, so they are not re-proposed.
* Anything needing a human decision goes under "still open" with the specific question.
