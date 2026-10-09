# Pilot / one-off test sessions

Exploratory widefield test sessions recorded *after* the main PS92–95 cohort finished data
collection (~2026-10-06). **These are NOT part of the cohort** and are deliberately kept out of the
nightly pipeline, cross-day products, and decks. They test new excitation settings / indicators.

---

## 2026-10-08 · PS94 · 490 nm vs 470 nm excitation
Standard-GCaMP PS94, 490 nm excitation LED in place of the usual 470 nm (415 isosbestic unchanged).
Full analysis and conclusions: **`docs/490_VS_470_EXCITATION.md`**. Summary: 490 gives modestly more
ΔF/F with no hemodynamic penalty, but net evoked SNR is within the 470 day-to-day range (N=1).

---

## 2026-10-09 · PS121 · RiboL1-GCaMP8s pilot (NEW animal, NEW indicator)

**Animal:** PS121 — genotype **VGLUT1-IRES-Cre × LSL-RiboL1-GCaMP8s**: soma-localized
(ribosome-tethered, RiboL1) GCaMP8s expressed in VGLUT1+ cortical excitatory neurons.

**Why:** first test of whether **soma-localized RiboL1-GCaMP8s gives enough widefield signal on this
PCO setup**, and whether soma-localization yields visibly **more regionalized (spatially discrete)
activity** than standard (cytoplasmic) GCaMP.

**Setup (all three sessions):** PCO camera only — **no DAQ / no sync .h5**; a few **unlogged** random
water rewards (so not time-alignable). The camlog shows the usual **415 + functional LED alternation**
(LED ids 5/6), i.e. these are 2-channel recordings even without the DAQ — the "490-only" description
refers to the *functional* LED; the 415 isosbestic ran alongside it in every session. **No hemodynamic
correction / no SVD processing applied** (no DAQ frame-map) — ΔF/F below is simple (F−F0)/F0 on the
functional channel, so it still contains hemodynamics.

**The three sessions** (`E:/labcams_data/20261009/`):

| folder | functional LED | ΔF/F std (med) | raw F (cortex) | %16-bit | saturated | dur |
|---|---|---|---|---|---|---|
| `PS121_test_...184722`        | 490 nm @ **1 A**    | 2.53% | 39,930 | 61% | 0% | 142 s |
| `PS121_test_700mA_...185356`  | 490 nm @ **700 mA** | 2.06% | 34,007 | 52% | 0% | 54 s |
| `PS121_test_470nmLED_...185739` | **470 nm**        | 1.60% | 34,643 | 53% | 0% | 138 s |

### Signal — verdict: plenty
- Healthy ΔF/F everywhere (1.6–2.5% median per-pixel), comparable to standard GCaMP on this rig
  (PS94 470/490 were ~2.3–2.6% *hemo-corrected*; these are uncorrected). **RiboL1-GCaMP8s gives ample
  widefield signal.**
- **No saturation** at any setting (even 1 A → 61% of range). The 415 channel is steady (~32k) across
  all three (good internal control).
- **490 > 470** for ΔF/F (consistent with GCaMP8s's excitation peak near 490).
- The 1 A-vs-700 mA ΔF/F gap is most likely a **duration artifact** (the 700 mA run was only 54 s), not
  a power effect — ΔF/F is power-normalized. **700 mA is a fine operating point** and avoids the 1 A
  overdrive (note: 490 LED is the M490L3, rated **350 mA**; see LED-current discussion — a sub-350 mA
  run is still worth doing).

### Regionalization — suggestive YES (real, not noise)
Spatial correlation length of the ΔF/F fluctuations (uncorrected 490, matched central crop), shorter =
more spatially discrete:

| | corr length @0.5 | @1/e |
|---|---|---|
| **PS121 RiboL1-GCaMP8s (490)** | **4 px** | 6–8 px |
| PS94 standard GCaMP (490)      | 22 px    | 33 px |

RiboL1 activity is **~5× more fine-grained** than standard GCaMP — the direction soma-localization
predicts (less diffuse neuropil signal → more discrete). **Confirmed real, not pixel noise:** the 4 px
length is **stable across 1×/4×/16× temporal binning** (noise would be suppressed by averaging and the
length would grow; it doesn't → the fine structure is temporally coherent). It also survives *despite*
no hemodynamic correction (hemo adds smooth large-scale structure that would bias toward *longer*
lengths), so the true neural regionalization may be even finer.

**Caveats:** N=1 animal each; the PS121-vs-PS94 magnitude is **not perfectly controlled** (different
animal / window / focus / FOV; PS121 raw & un-motion-corrected vs PS94 motion-corrected) — so focus/
window differences could contribute to the 5×, even though the within-PS121 noise check is clean.
Pixel→mm scale uncalibrated. To nail the indicator attribution: matched same-rig comparison (RiboL1 vs
standard GCaMP, matched focus), with event-triggered maps / split-half spatial reliability.

*Reproduce:* raw `.dat` + camlog on E: (and whatever is archived); metrics computed directly from the
functional interleaved stream (brighter of the two camlog LED channels). No DAQ, so the standard
pipeline does not apply to these.
