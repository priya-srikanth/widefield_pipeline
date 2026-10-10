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

### Regionalization — NOT ESTABLISHED (confounded); signal is real & fine, but can't be attributed

This went through several iterations; the honest end state is below. **IMPORTANT STATE NOTE:** PS121 was
imaged **post-anesthesia and NOT doing the behavioral task**; PS94 was **awake and behaving**. That
state difference confounds every spatial-scale comparison between them.

**Channel integrity first (no DAQ/TTL):** verified all three sessions have no dropped-frame parity flip
(camlog alternates strictly — 0 consecutive-equal LED ids; frame timestamps perfectly regular, 0 gaps;
even/odd brightness sign constant throughout). So the 415/490 assignment is reliable and the hemo
regression can trust the channel labels. func = brighter (odd) stream, isosbestic 415 = even.

**The analysis arc (and two mistakes corrected along the way):**
1. *Raw* ΔF/F (no SVD, no hemo): PS121 looked ~5× finer than PS94 (corr len ~4 px vs ~22 px). **This was
   largely BLOOD FLOW** — caught when the activity maps were noted to "look like blood flow."
2. After **motion correction + 415-isosbestic hemodynamic regression**: the 415 removed **35–63%** of
   PS121 variance; PS121 corr length collapsed to **~1 px** while PS94 stayed **~26 px** (only 7%
   removed, same procedure). I initially called the 1 px "noise" — **also wrong.**
3. **Temporal autocorrelation** of the corrected signal: PS121 lag-1 = **0.26–0.66**, decaying over
   ~10 frames, vs a white-noise floor of ~0.03. So the fine corrected signal is **temporally real, not
   noise** — consistent with genuinely fine (~1 px) regionalized activity, as hypothesised for a
   soma-localized indicator.
4. **SVD** of the corrected ΔF/F: PS94 is **extremely low-rank** (90% of variance in **3** components —
   awake cortex dominated by a few global modes); PS121 is **very high-rank** (90% needs **~700–970**
   components — variance spread across hundreds of fine, spatially-distributed, temporally-structured
   modes). High-rank + temporally-real is consistent with fine/sparse regionalized activity.

**Why it's still NOT established:** the anesthetized/no-task state *by itself* produces high rank and
loss of the global coherent modes that dominate awake cortex — i.e. the **state confound and the
indicator hypothesis push in the same direction**, so this data cannot separate them. Also: SVD
denoising is biased toward low-rank/coherent structure (it would *suppress* genuinely fine signal), so
"denoised corr length" is not a fair regionalization metric here; N=1 each; PS94 is the wrong reference
(awake). 

**Bottom line:** RiboL1-GCaMP8s gives real, fine-scale, temporally-structured signal after blood-flow
removal — **consistent with** genuine fine regionalization but **not proof of it**. To actually answer
it: **state-matched** RiboL1 vs standard GCaMP (awake/task, or matched anesthesia), same rig/focus, with
proper SVD + pipeline hemo correction and event-based analysis.

*Figures (all on N: `20261009/`):* `PS121_regionalization_vs_PS94_std_GCaMP.png` (raw — SUPERSEDED/
misleading, mostly blood flow); `PS121_regionalization_CORRECTED_allsessions.png` (motion+hemo
corrected, uncorrected-vs-corrected); `PS121_SVD_check.png` (SVD spectrum + leading components).
*Reproduce:* metrics computed directly from the functional interleaved stream
(brighter of the two camlog LED channels) + 415 even stream; no DAQ, so the standard pipeline does not
apply — motion correction (rigid, skimage) and hemo (per-pixel 415 regression) were done ad hoc.
