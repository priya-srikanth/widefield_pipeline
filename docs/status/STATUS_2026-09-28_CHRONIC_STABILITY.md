# Is the chronic neural data stable? — per-session, per-animal, per-readout (2026-09-28)

**START HERE if the question is whether to stop recording.** Priya, 2026-09-28: *"behavior has
stabilized so I'm wondering if we should deem the experiment 'complete' and stop recording (and sac
the animals for histology). But I want to ensure the neural data is also stable."*

This document does not make that call. It says what each readout does across each animal's chronic
sessions, by the same rule that already decides when an animal is chronic, and it names the two
things found on the way that change how the chronic numbers should be read.

**Reproduce:** `python -m scripts.chronic_stability` → `<labcams>/chronic_stability/chronic_stability.{csv,png}`.
Every number below is in that CSV. Render set 2026-09-26; cohort pre 44, chronic PS92 9 / PS93 9 /
PS94 5 / PS95 8 (the trajectory family counts PS93 day 11 as chronic, the map family as subacute —
one session, boundary-adjacent).

---

## The test, and why it is this one

The behavioural chronic boundary is set by `epochs._plateau_index`: the far-contra hit rate, as a
fraction of the animal's own pre-stroke mean, must be **flat** (total drift across the window ≤ 1.0 ×
pre-stroke SD, one-sided — only a still-*rising* series fails) and **settled** (residual around the
fitted line ≤ 0.4 × pre-stroke SD) from some session onward, and stay so. That rule was argued over
for two weeks and is the project's definition of "stopped changing". The honest test of "is the
neural data stable too" is to ask the *same* rule of each neural readout, not to invent a friendlier
one. Four animals, so no cohort p-value — per-animal verdicts and their consistency are the evidence.

Two consequences to hold onto. **The settled bar is strict**: 0.4 × pre SD is tight enough that
PS94's *behaviour* fails it (0.63×), which is exactly why its chronic is a manual pin. So read the
residual as a ratio, not the flag. And **the flat test is one-sided**: a readout still *falling*
passes, a readout still *rising* fails — so "RISING" below means the animal is still moving away
from its pre-stroke state on that readout.

Readouts: frozen and refit decoder accuracy and their difference *G* (reorganisation index);
encoder ceiling, training-matched frozen encoder, and gain; best-match template accuracy; crossnobis
distance from each position's own pre-stroke template; cue-evoked map amplitude at far-contra and
near-ipsi; and behaviour, as the reference. Per-session values come from the deck's own `_sessions.csv`
sidecars (row order verified against a known outlier) and the recovery-trajectory CSV.

---

## Two findings that must be applied before reading any trajectory

### PS92_0922 is a frame↔DAQ misalignment — a pipeline bug, recoverable — not neural instability

Day 36 has both decoders and the encoder at chance (frozen 0.16, refit 0.30, ceiling 0.43,
best-match 0.17) with **perfect behaviour** (100% at all six positions, 333 trials). The cause: the
DAQ recorder was started **2.466 s after the camera** that day (camlog 12:55:04.295 vs DAQ acquisition
12:55:06.761; the DAQ file opens with the exposure train already running), leaving 154 camera frames
with no DAQ pulse at the *head* of the `.dat`; the relabel step (`1a68c7c`, written that same night)
assumed the surplus was a *tail* and mapped every frame to the pulse 154 positions later. Every trial
window read 2.466 s of pre-cue baseline, and 4.6% of frames landed in the wrong LED channel, which is
what the "anti-phase vascular oscillation" was. Established by the camlog's own per-frame LED record
(100.0000% agreement at offset 154, 95.4% at 0) and three independent timing measurements. **The brain
data are intact** — the cue transient is at full amplitude once realigned. Full record and the fix:
`docs/EXPERIMENT_ERRORS.md`, 2026-09-22. Until the session is re-preprocessed, every PS92 row below
is given both ways; afterwards this table should be re-run.

### PS95's late rise is real signal, not an imaging change; PS95_0924 is a drift day that survived

PS95's cue-evoked map amplitude steps up at **day 29** (0914) at both positions and stays up. That
session has PS95's *highest* evoked energy (16.6 vs 8.9–13.5), *lowest* motion (0.11 px) and highest
calcium-band fraction — and **no change in mean intensity** (470 mean 13777, neighbours 13549/13345),
so it is not an LED or window change. The elevated response persists at days 32–39. Separately,
day 39 (0924) has a large slow drift in both channels (sd ~54, motion 1.58 px) that the haemodynamic
correction handled — decoding is normal that day. Neither is a data-quality exclusion; the first is a
finding.

---

## The verdicts

Chronic level is the fraction of that animal's pre-stroke mean (G and map amplitude in raw units).
Drift and residual are in pre-stroke-SD units; **RISING** = fails the one-sided flat test.
PS92 excludes 0922 (with it, every PS92 residual roughly doubles and nothing else changes).

| readout | PS92 (8) | PS93 (9) | PS94 (5, pinned) | PS95 (8) |
|---|---|---|---|---|
| **behaviour** far-contra hit | 1.04 · flat · **0.05×** settled | 1.02 · flat · **0.09×** settled | 1.04 · flat · 0.63× | 1.07 · flat · **0.15×** settled |
| decoder **frozen** | 1.05 · flat · 0.56× | 0.87 · falling −2.2 · 1.21× | 0.83 · +0.8 · 0.64× | 0.90 · flat · 0.64× |
| decoder **refit** | 1.11 · flat · **0.29×** settled | 1.08 · flat · **0.31×** settled | 1.00 · flat · 0.91× | 0.99 · flat · 1.27× |
| **reorganisation G** | +0.05 · **RISING +1.4** · 0.85× | +0.14 · **RISING +1.0** · 0.84× | +0.13 · falling −2.2 · 0.56× | +0.07 · **RISING +1.4** · 0.61× |
| encoder **ceiling** | 1.28 · flat · 0.46× | 1.08 · flat · **0.24×** | 0.97 · flat · 1.00× | 1.08 · **RISING +1.1** · 0.30× |
| encoder **frozen** (matched) | 0.93 · flat · 0.47× | 0.61 · flat · 0.84× | 0.75 · falling −1.1 · 0.60× | 0.47 · falling −2.6 · 0.67× |
| encoder **gain** | 1.28 · flat · 0.54× | 0.85 · flat · 0.75× | 0.79 · falling −1.4 · 0.73× | 0.75 · falling −2.0 · 0.61× |
| **best-match** template | 0.98 · flat · 0.82× | 0.74 · falling −2.2 · 0.62× | 0.90 · flat · 0.56× | 0.79 · falling −2.8 · 0.59× |
| **crossnobis** far-contra | 2.14 · flat · 0.67× | 1.95 · flat · 0.76× | 1.76 · falling −1.0 · 0.85× | 2.39 · **RISING +1.0** · 0.70× |
| **crossnobis** mean (6 pos) | 2.57 · flat · 0.90× | 2.32 · flat · 0.72× | 1.60 · falling −1.3 · 0.97× | 2.40 · **RISING +1.6** · 0.71× |
| **map** far-contra | 0.023 · flat · **0.30×** settled | 0.017 · flat · 0.89× | 0.006 · flat · 0.48× | 0.016 · **RISING +3.8** · 1.00× |
| **map** near-ipsi | 0.013 · falling −3.3 · 0.72× | 0.020 · flat · 0.76× | 0.006 · flat · 0.89× | 0.016 · **RISING +6.3** · 2.29× |

Encoder, template and crossnobis rows use the pooled pre value as the level and the animal's own
late scatter as the tolerance (their sidecars carry no per-session pre), so for those rows the
*drift* verdict and the *level* are meaningful and the residual ratio is descriptive only.

### Per animal

**PS92 — stable, and recovered.** With 0922 set aside, every readout is flat; frozen and refit
decoders, best-match and map amplitude sit at or above pre-stroke. The one exception is *G*, which
is small (+0.05) but still rising. This is the animal whose frozen readout came back.

**PS93 — stable in level, not converging.** Everything is flat or falling; nothing rises except *G*
(+0.14, the largest of the four, still climbing). But the levels are far from pre and not returning:
frozen decoder 0.87 and *falling* (−2.2 SD over chronic: 0.61 → 0.49), best-match 0.74 and falling,
frozen encoder 0.61, crossnobis 1.9–2.3× pre distance. The refit decoder is settled at 1.08. PS93 —
the deficit animal — has a stable *replaced* code that the pre-stroke readout reads a little worse
each week. Whether that is "stable" depends on which claim the histology is meant to anchor.

**PS94 — too few sessions to say, and the boundary is a pin.** Five chronic sessions, boundary
ratified by hand because behaviour wobbles. Neural readouts are flat or falling, none rising; the
frozen decoder is still creeping up (+0.8 SD) toward 0.83. Nothing here argues *against* stability,
and five points cannot argue *for* it either.

**PS95 — NOT stable.** Six readouts are still moving in the *same* direction: refit ceiling, *G*,
crossnobis (both), and map amplitude (both) **rising**; frozen encoder, gain and best-match
**falling**. That is one coherent motion — the within-session code strengthening while the
pre-stroke readout loses grip — continuing through days 15–39, with a step at day 29 that is not
instrumental. PS95 is still reorganising. If any animal would give different chronic numbers with
four more weeks of sessions, it is this one.

---

## What this does and does not license

* **Behaviour has plateaued in all four** (PS94 wobbling within a recovered band). That is
  established by the rule and is not in question.
* **The readouts that track behaviour — frozen and refit decoding — have plateaued in all four.**
  No animal's decoder is still rising.
* **The readouts that index ongoing reorganisation — G, crossnobis distance, map amplitude — have
  plateaued in PS92, PS93 and PS94, and have not in PS95.** G is still rising slowly in three of four.
* **"Stable" and "recovered" are different claims** (PRELIM_DATA §10). PS92 is both; PS93 and PS95
  are stable-in-behaviour with a representation still drifting away from pre-stroke; PS94 is
  undetermined. Histology anchors a *state*; for PS95 that state is not yet final on these readouts.
* **The strict settled bar is not met by most neural readouts in any animal** (residuals 0.5–1.3×
  pre SD against the 0.4× rule). Session-to-session scatter in a decoder is inherently larger than in
  a hit rate; this is a reason to read the ratios rather than the flag, not evidence of instability.

## What would change the picture

* **Re-preprocess PS92_0922 with the fixed relabel** (imaging box, from the standby raw; recipe in
  `docs/EXPERIMENT_ERRORS.md`), let the analysis box re-derive PS92, and re-run this table. Expect
  PS92 to gain a ninth clean chronic session and to remain the strongest stability case in the
  cohort. Until then it is excluded from the imaging arms; its behaviour stays.
* **Two to four more PS95 sessions** would show whether the day-29 step is a new plateau or a slope.
  The same is true, more weakly, of PS94's frozen decoder.
* **Re-render `15k`/`15d` on the grown cohort** — unrelated to stability, but the chronic region
  tables PRELIM_DATA quotes are still on 18 sessions.
* This analysis reads existing sidecars; it adds no new estimator. A readout it cannot see is the
  rest arm's per-session frozen decoder (`epoch_15f_*_sessions.csv`, which carries labels and could be
  added directly) — worth a row if the rest code's stability matters to the claim.
