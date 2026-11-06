# 490 nm vs 470 nm excitation — single-session comparison (PS94)

**Date:** 2026-10-08 · **Animal:** PS94 · **Status:** exploratory, N=1 for 490 (not conclusive)

## What this is

The last bonus session of data collection (2026-10-08) imaged PS94 with a **490 nm**
excitation LED in place of the usual **470 nm**, keeping the **415 nm isosbestic** unchanged.
The DAQ `led470_ttl` line physically carried the 490 LED, so the pipeline processed it
normally (slot 0 = 415 isosbestic, slot 1 = 490 functional, `functional_channel=1`). Session:
`20261008/PS94_490exc_415iso_20261008_103519` (DAQ `PS94_490exc_415iso_20261008_103602.h5`).

Question: is dF/F range / signal / SNR better or worse at 490 vs 470?

## Method

Compared the 490 session against the two most recent **470** PS94 sessions (same animal/window):
`20261005/PS94_20261005_094301` and `20261001/PS94_20261001_094021`.

- **Why dF/F, not raw F:** LED power is manually titrated day to day, so raw fluorescence is
  not comparable across sessions. ΔF/F divides out F0 and is the fair metric. All three use the
  identical 415-isosbestic hemodynamic correction (`meegkit_hpfit`), so the only variable is the
  functional excitation wavelength.
- **Identical ROI (important):** metrics are computed over ONE common ROI =
  Allen brain mask ∩ window-coverage of *all three* sessions, in the shared **540×640 CCF grid**
  (affine-aligned `U_atlas` / `frames_average_atlas` / delta maps). **200,670 px, identical for
  every session.** The Allen masks were verified byte-identical across sessions. (An earlier pass
  used per-session *native*-grid masks of different sizes, 57k vs 67k px — a confound; moving to
  the common CCF ROI did not change any conclusion.)
- **Whole-session ΔF/F range:** per-pixel temporal std of the corrected functional signal,
  `sqrt(diag(U_atlas · cov(SVTcorr) · U_atlasᵀ))`, summarised over the ROI (median, 95th pct).
- **Evoked signal / noise / SNR:** cue-evoked trial-averaged delta maps (`spout_positions_1s_pre_post`),
  **100 trials/position for all three sessions** (so the averaged-map noise floor ∝ 1/√N is matched).
  Signal = 99.5th-pct ΔF/F over ROI; noise = 1.4826·MAD over ROI (responses are spatially sparse);
  SNR = signal/noise; both averaged over the 6 spout positions.

## Results (common 200,670-px CCF ROI)

| session      | ΔF/F corr med / 95% | peak evoked ΔF/F | bg noise | evoked SNR |
|--------------|---------------------|------------------|----------|------------|
| **490 (1008)** | **2.62% / 6.17%**   | **2.86%**        | 0.569%   | 5.33       |
| 470 (1005)   | 2.32% / 5.68%       | 2.36%            | 0.495%   | 4.87       |
| 470 (1001)   | 2.29% / 5.61%       | 2.61%            | 0.387%   | 7.00       |

Supporting (native-grid, LED-confounded) raw levels: raw functional F was ~2–3% *lower* at 490
(25.5k vs ~26.0–26.2k counts); the 415-correction removed 6.3% of variance at 490, within the
470 range (4.3–13%); rcoeff 1.13, within the 470 range (1.04–1.31) — no hemodynamic penalty.

## Conclusions

- **More signal at 490 — robust.** Larger whole-session ΔF/F range *and* larger cue-evoked peak
  ΔF/F than *both* 470 days (~10–20%). Physically expected: 490 nm is nearer GCaMP's excitation
  peak (~485–490 nm), so a larger fraction of emitted photons is Ca-dependent.
- **Higher noise at 490 too.** Background noise was the highest of the three, likely because raw
  F was slightly lower that day (fewer photons → more shot noise) and/or a marginally hazier
  window — not a hemodynamic effect (correction magnitude was normal).
- **Net evoked SNR: not demonstrably better.** 490's SNR (5.33) sits *inside* the 470 day-to-day
  range (4.87–7.00); the best SNR of the three was a 470 day (1001). The larger signal is offset
  by the larger noise.

## Caveats

- **N=1 for 490** vs 2 for 470, **single animal, different days.** ΔF/F amplitude also depends on
  how active/engaged the animal was; part of the 490 signal advantage may be activity-driven.
- The **470 day-to-day SNR spread (4.87 vs 7.00) is larger than any 490-vs-470 difference**, so a
  single 490 session cannot resolve a wavelength effect on SNR.
- This is ΔF/F **dynamic range / evoked amplitude**, not a controlled SNR measurement.
- To decide properly: **matched, ideally same-day interleaved 490/470 blocks** (same window
  clarity, same behavioral state), across ≥2–3 animals.

*Reproduce:* metrics computed from the archived N: outputs (`U_atlas.npy`, `SVTcorr.npy`,
`frames_average_atlas.npy`, `allen_brain_mask_native_grid.npy`, and the cue
`*_spout_positions_1s_pre_post_delta_maps.npz`) for the three sessions above. Raw `.dat`/`.bin`
are on M: standby.

---

## Cohort2 (2026-11): 490 nm across GCaMP6s (PS118) and RiboL1-jGCaMP8s (PS121/PS122)

Cohort2 switched the functional LED to **490 nm, driven ABOVE its rated current** — PS121/PS122 from
2026-11-04, PS118 (first GCaMP6s 490) 2026-11-05 (folder tag `_490_`). Each animal is compared to its own
prior **470** sessions; N=1 490 session per animal so far. Scripts: `scripts/cohort2_qc/led_490_vs_470.py`
(brightness F0, per-pixel-SD/shot-floor SNR, within-session fade) and `led_490_vs_470_dprime.py`
(**single-trial response SNR** = mean / across-trial SD of the baseline-subtracted task response — a
standardized effect size, NOT a signal-detection d′). Both cover all three animals; figures are on
N: `cohort2/490_vs_470/` and in the cohort2 RiboL1 deck.

| animal (indicator) | metric | 470 range | 490 (one session) | read |
|---|---|---|---|---|
| **PS118 (GCaMP6s)** | F0 / SNR / ΔF/F | 22–25.5k / 2.7–3.3 / 2.8–3.1% | 22.8k / 3.1 / 3.12% | **no gain** (mid-range) |
| | func fade | 0 to −3.6% | −8.0% | a bit more, but < its own 415 fade (−17%) |
| | cue / lick resp-SNR | 1.13–2.01 / 1.88–2.86 | 1.39 / 2.37 | mid-range (below best 470 days) |
| **PS121 (RiboL1-8s)** | SNR / ΔF/F | 0.6–0.9 / 0.64–0.80% | 1.0 / 0.92% | **top of range** |
| | cue / lick resp-SNR | 0.44–0.85 / 1.10–1.80 | 0.97 / 1.58 | cue = best day |
| **PS122 (RiboL1-8s)** | SNR / ΔF/F | 0.9–1.1 / 0.84–0.95% | 1.3 / 1.18% | **top of range** |
| | cue / lick resp-SNR | 1.01–1.43 / 1.54–2.35 | 1.45 / 2.35 | cue = best day; lick ties best |

**Conclusion.**
* **RiboL1-jGCaMP8s: 490 helps.** Brightness, per-pixel ΔF/F and SNR land at the **top** of each animal's
  470 range, and single-trial **response SNR** is the **best day** on cue (both) and tied-best on lick (PS122).
* **GCaMP6s (PS118): 490 ~neutral.** F0, SNR and ΔF/F are mid-range vs its 470 days — no improvement (470
  already excites GCaMP6s efficiently), and single-trial response SNR is mid-range too (cue 1.39 / lick 2.37,
  below its best 470 days 2.01 / 2.86).
* **No excess overdrive LED droop in any animal.** The 490 functional channel's within-session fade is
  **smaller than the same session's 415 fade** (PS118 −8% vs −17%; RiboL1 similar), and the 415 LED is
  separate and unchanged — so it cannot be drooping from the 490 overdrive. PS118's 490 fade (−8%) is
  modestly above its 470 days (≤−4%) but is mostly general session bleaching, not the LED dimming.
* **Caveat:** N=1 490 session per animal; every 490 value falls **inside** the animal's 470 session-to-
  session spread, so treat "helps 8s / neutral for 6s" as a strong hint, not settled — a few more 490
  sessions (ideally same-day interleaved 490/470 blocks, as this doc's pilot section urged) would confirm.
