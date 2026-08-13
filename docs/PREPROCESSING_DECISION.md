# Drift removal and hemodynamic correction — the decision record

**Status: PENDING the 36-session comparison (running 2026-08-13).** This document holds the evidence
and the reasoning; the final row of the decision table is filled in when that comparison lands. Nothing
downstream has been re-run yet, and `SVTcorr.npy` on MICROSCOPE is still the original `zerophase`
product.

---

## The problem

`wfield.hemodynamic_correction` removes slow drift with a **zero-phase** (`scipy.filtfilt`) 0.1 Hz
Butterworth applied to both channels, and the filtered blue channel is what becomes `SVTcorr`. A
zero-phase filter's impulse response is symmetric in time — measured on this exact filter, an impulse
deposits **−0.496 before itself and −0.496 after**, with −0.209 landing in the single preceding second.

A high-pass is `identity − lowpass`; at 0.1 Hz the lowpass kernel is ~10 s wide, so half of it falls
*before* each event and is subtracted. A position-specific POST-cue response therefore casts a scaled,
**sign-flipped shadow backwards** into the pre-cue window. A linear decoder is indifferent to sign, so
that shadow reads as position information in a window that biologically cannot contain it.

**This is the upstream method, not a local bug.** `churchlandlab/WidefieldImager/Analysis/SvdHemoCorrect.m`
overwrites both channels in place with `filtfilt` and keeps no unfiltered copy; `jcouto/wfield` is a
faithful port; Musall et al. 2019 state it in their methods. The artifact class is published — van
Driel, Olivers & Fahrenfort 2021, *J Neurosci Methods* — including the negative sign, with trial-masked
robust detrending as the recommended fix.

## What it costs (36 curated sessions, ROI features, `T` reused so only the filter varies)

| variant | PRE-CUE | post-cue (control) | corr(pre,post) | sign test | vs zerophase |
|---|---|---|---|---|---|
| `zerophase` (current) | 0.486 | 0.684 | −0.483 | neg in **30/36** | — |
| `fitonly` | 0.273 | 0.686 | +0.672 | neg in 1/36 | −0.213, 35/36, p=1.5e-10 |
| `taskdetrend` | 0.247 | 0.737 | +0.429 | neg in 4/36 | −0.239, 36/36, p=2.9e-11 |
| `strobedetrend` | 0.306 | 0.731 | +0.525 | neg in **0/36** | −0.181, 35/36, p=4.1e-10 |

Post-cue is unchanged or better under every correction, which is what makes the pre-cue drop
interpretable rather than a general loss of signal.

## Three constraints the fix has to satisfy

**1. The drift estimator must be slower than the position blocks.** Positions are presented in ~6-trial
blocks lasting **57–121 s**, so any estimator flexible enough to track that timescale removes position
signal along with drift — they are temporally inseparable by construction of the task. Measured
directly (4 sessions, masked-median detrend, window swept):

| window | 30 s | **60 s** | 120 s | 300 s | 600 s | 900 s | meegkit (ord 5) |
|---|---|---|---|---|---|---|---|
| pre-cue | 0.317 | **0.302** | 0.337 | 0.378 | 0.381 | 0.365 | 0.377 |

The minimum sits at 60 s — the block duration — and recovers to a plateau by 300–600 s, where the
windowed median and the robust polynomial agree to three decimals.

**2. The real drift is slower than that anyway.** Power of the global brain-mean trace, by band:

| chan | >5 min | 5–2 min | 2–1 min (BLOCK) | 60–10 s | 10 s–0.1 Hz | >0.1 Hz |
|---|---|---|---|---|---|---|
| 415 (isosbestic ⇒ drift) | **27–64%** | 5–14% | **2.0–3.5%** | 6–10% | 11–22% | 8–23% |
| 470 (functional) | 5–14% | 7–11% | 5–10% | 18–33% | 25–42% | 11–17% |

**3. The coefficient fit still needs the high-pass.** `rcoeffs` is ONE scalar per pixel — the OLS slope
converting 415 units into 470 units — so it can only be optimal for whichever band dominates the data
it is fitted on. Removing the high-pass lets slow variance dominate and the coefficient gets tuned away
from the hemodynamic band. Residual |r| with the 415 channel:

| variant | <0.1 Hz | 0.1–1 Hz vasomotion | 1–5 Hz breathing | 5–13 Hz heartbeat |
|---|---|---|---|---|
| UNCORRECTED | 0.170 | 0.576 | 0.837 | 0.634 |
| `zerophase` | 0.276 | 0.152 | 0.216 | 0.327 |
| `strobedetrend` (detrend both) | 0.301 | 0.300 | 0.459 | 0.085 |
| **`detrend_hpfit`** (hybrid) | 0.443 | **0.132** | **0.216** | 0.327 |
| **`meegkit_hpfit`** (hybrid) | 0.444 | **0.132** | **0.216** | 0.327 |

## The candidate: `meegkit_hpfit`

Separate the two jobs the high-pass was silently doing:

* **band-select for the FIT** — keep the 0.1 Hz high-pass, which is what it is actually for;
* **remove drift from the OUTPUT** — replace it with de Cheveigné robust polynomial detrending
  (`meegkit.detrend`, order 5, iteratively reweighted) on a mask that excludes the whole trial
  (strobe−0.25 s → cue+4 s), so the drift fit never sees the measured window and nothing is smeared in
  time.

Nothing overwrites the originals: variants are written to `hemo_<variant>[_refitT]/` subdirectories
with their own `SVTcorr`/`T`/`rcoeffs` and a `manifest.json` (see `CLAUDE.md`).

## What the correction does NOT change

Post-cue, lick-aligned and RSA analyses are unaffected — the real response sits inside those windows,
and the control above shows them unmoved or improved. The lick-free and vision confound controls remain
valid: they asked whether pre-cue information could be explained by licking or by vision (no, and no),
which was never a question about *when* the information arrived.

## What survives, corrected

Pre-cue position information is **real and significant** — 36 sessions × 200 block-label permutations,
significant in **34/36**, bootstrap CI above chance in 32/36, against an **empirical null of
0.136–0.147** (reliably below the nominal 1/6, so testing against 0.167 is conservative):

| animal | corrected pre-cue | perm p<0.05 |
|---|---|---|
| PS92 | 0.234 | 7/9 |
| PS93 | 0.258 | 9/9 |
| PS94 | **0.418** | 9/9 |
| PS95 | 0.298 | 9/9 |

About **half** the published size, and the cohort is genuinely non-uniform rather than the artifactual
0.47–0.52 flatness. Terminology follows: **"pre-cue position information"**, not "maintained motor
plan" — the spout arrives ~3 s before the cue, so a sustained sensory response and a held intention are
temporally coextensive and this design cannot separate them. It does not need to: a pre-cue position
signal that changes post-stroke is the readout either way.

**Why this matters MORE under that framing, not less:** the shadow is a scaled copy of the post-cue
response, so its size tracks post-cue amplitude — which the stroke will change. A post-stroke drop in
pre-cue information could be caused entirely by a smaller post-cue response leaking backwards. That
confound points the same way as the hypothesis and does not cancel in a pre/post comparison.

## Decoder settings that follow (measured, 16 sessions, corrected data)

* **pre-cue: 4 × 0.5 s sub-bins over the 2 s window** — beats the 2 s mean by +0.032 in 14/16 sessions.
  8 × 0.25 s is worse (+0.023, 10/16): 250 ms resolution adds features, not information.
* **do NOT shorten the window.** On corrected data 2.0 s is best; `mean1.5` −0.020 (1/16) and
  `mean1.0` −0.028 (1/16). The apparent advantage of a shorter window on the old data was the shadow
  being strongest nearest the cue.
* the `last1.0 − first1.0` asymmetry, which read as "the code builds toward the cue", was **+0.246 on
  filtered data and −0.040 corrected** — it was the shadow's decay profile, not the plan's.

## Errors made and corrected along the way

Recorded because each was found by a challenge rather than by the check that should have caught it.

1. Applied a high-pass-derived `T` to detrended traces (correct for isolating one variable, wrong for a
   product) — fixed by `refit_T`, validated against the saved coefficients to 1.5e-6.
2. Called PS92 "at chance" from `taskdetrend`, whose mask left 1.5 s of the pre-cue window inside the
   drift fit. Under `strobedetrend` PS92 is 0.231–0.234 and significant in 7/9 sessions.
3. Asserted the drift kinetics instead of measuring them; the measurement supported the conclusion but
   also showed the 60 s window sat on the worst possible value.
4. `remove_drift` bound `order=MEEGKIT_ORDER` as a default argument, so a sweep setting the module
   attribute silently reused the definition-time value — every "order" in the meegkit arm was order 5.
   Caught only because the numbers were impossibly identical.
5. Predicted post-cue would stay flat across the timescale sweep; it rose 0.769 → 0.869, so aggressive
   detrending was removing signal broadly, not only in the pre-cue window.

## Reproduce

```
python -m wfield_local.filter_acausality_test --modes zerophase,strobedetrend,meegkit_hpfit --refit-t
python -m wfield_local.hemo_residual_check --variants zerophase,meegkit_hpfit
python -m wfield_local.precue_significance --variant meegkit_hpfit --n-perm 200
python -m wfield_local.precue_window_sweep --modes zerophase,meegkit_hpfit --refit-t
python -m wfield_local.hemo_variants --list
```
