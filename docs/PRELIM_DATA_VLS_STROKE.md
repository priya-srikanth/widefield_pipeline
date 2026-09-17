# Preliminary data: what VLS stroke does to cortical position coding

**Rewritten clean on 2026-09-17.** Earlier versions carried four stacked "SUPERSEDED" strata
(2026-09-09, 09-10, 09-12, 09-16), numbers read off renders that no longer exist, and one whole
section whose sign convention was inverted. Priya, 2026-09-17: *"make the prelim data doc accurate
(can remove incorrect information from prior iterations)."* **Everything below is current; nothing
is retained for history** — `DECISIONS.md` and the STATUS docs hold that.

**PROVENANCE.** Figures and their `.csv` companions in
`N:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/`, 2026-09-16 render set, `restdock05`
rest definition. **Every number here is read from the `.csv` beside its figure or from
`kw.stat_rows` in its `*_bundle.json`, never off a heatmap** — so each claim can be checked against
the panel that drew it. Session set throughout: **N=4 animals, n=96 sessions — pre 44 (11 each),
acute 16 (92:5 93:4 94:6 95:1), subacute 18, chronic 18 (PS94 has none)**.

`*` = interval excludes zero · `**` = survives Bonferroni correction.

**READ `docs/REST_ENGAGEMENT_AUDIT.md` BEFORE QUOTING ANY REST NUMBER.** Two silent defects were
found on 2026-09-16 across seven rest analyses; everything in §6 is post-fix.

---

## 1. THE BEHAVIOURAL DEFICIT — what the animal actually does

`epoch_1b_behaviour_by_position` · response rate (lick within the session's real 3.5 s window).

| epoch | Near Ipsi | Near Mid | Near Contra | Far Ipsi | Far Mid | **Far Contra** |
|---|---|---|---|---|---|---|
| pre | 0.969 | 0.970 | 0.970 | 0.887 | 0.930 | **0.935** |
| **acute** | 0.897 | 0.888 * | 0.869 ** | 0.647 ** | 0.452 ** | **0.052** ** |
| subacute | 0.967 | 0.967 | 0.959 | 0.913 | 0.901 | 0.797 ** |
| chronic | 0.988 | 0.987 | 0.982 | 0.947 | — | — |

**Acutely the animal essentially stops attempting far-contralateral — a response rate of 0.052,
down from 0.935.** The deficit is graded near→far and ipsi→contra, and it recovers to 0.797 by
subacute. This is the behavioural anchor for everything below, and the reason the lick-aligned
imaging arm is blind to acute far-contra (§5).

---

## 2. THE POSITION READOUT — a frozen pre-stroke decoder

`epoch_acc_by_position_cue_working` / `epoch_accdelta_by_position_cue_working`, post-cue, lick +
miss-while-working. Chance 1/6.

| epoch | nI | nM | nC | fI | fM | **fC** |
|---|---|---|---|---|---|---|
| pre | 0.934 | 0.846 | 0.923 | 0.867 | 0.847 | **0.896** |
| **acute** | 0.700 * | 0.575 ** | 0.610 ** | 0.494 ** | 0.440 ** | **0.321** ** |
| subacute | 0.885 | 0.685 ** | 0.882 | 0.724 ** | 0.699 * | 0.615 ** |
| chronic | 0.963 | 0.691 * | 0.941 | 0.770 | 0.790 | 0.795 ** |

Acute change from pre, Bonferroni-corrected: nI −0.234 (crosses zero), nM −0.271, nC −0.313,
fI −0.373, fM −0.407, **fC −0.575 [−0.738, −0.394]**. **A monotone near→far gradient with
far-contra worst**, five of six surviving correction.

**By chronic only far-contra remains significantly below pre** (−0.101, corrected
[−0.204, −0.0007]). Everything else has recovered.

---

## 3. IS THE CODE LOST, OR PRESENT AND MISREAD? — frozen vs refit

`epoch_5rgap_*` (raw gap) · `epoch_5rgapdelta_*` (the claim) · `epoch_5rm*` (training-set matched).

**SIGN CONVENTION: the plotted quantity is `refit − frozen`** — `_gap_at` returns
`mean(refit correct) − mean(frozen correct)`, and the figure's own `_meta.csv` `ylabel` reads
`refit - frozen accuracy`. **POSITIVE = the session's own decoder reads a position the frozen
pre-stroke model cannot: information PRESENT but DISPLACED.**

**THE PRE ROW IS NOT ZERO BY CONSTRUCTION.** The frozen arm trains on ten pre-stroke sessions and
the refit on four fifths of one, so pre carries a training-set-SIZE handicap with no lesion in it.
The claim is therefore the **delta**, not the raw gap.

**THE MATCHED FAMILY (`epoch_5rmgapdelta_*`) IS THE AUTHORITATIVE TEST; the unmatched one
(`epoch_5rgapdelta_*`) is reported only as a bracket.** Unmatched, the frozen arm trains on ten
pre-stroke sessions against the refit's four fifths of one, so its pre gap (−0.116) is a
training-set-SIZE handicap with no lesion in it. Matching the frozen model to the refit's training
size — drawn as whole pre-stroke BLOCKS, seeded per scored session — flips the pre gap to **+0.036**:
at equal data a within-session fit wins, because it shares that session's own nuisance structure
while a matched frozen model must generalise across days. **So the no-lesion baseline is BRACKETED
(−0.116 unmatched / +0.036 matched), not known**, which is why both are drawn and both read as
epoch-minus-pre.

**MATCHED acute delta (epoch − pre), Bonferroni-corrected — the number to quote:**

| position | point | 95% CI | corrected | |
|---|---|---|---|---|
| **far-contra** | **+0.158** | [0.046, 0.284] | **[0.0004, 0.337]** | the only acute cell surviving |
| near-contra | +0.128 | [0.004, 0.235] | [−0.060, 0.290] | |
| **far-middle** | **−0.132** | **[−0.242, −0.021]** | [−0.291, 0.035] | |
| near-ipsi | +0.093 | [−0.069, 0.228] | — | |

**Matching SHARPENS the dissociation rather than shrinking it.** Far-contra keeps a significant
positive recoverable component once the handicap is removed, while **far-middle turns NEGATIVE** —
refitting buys *less* there than before the lesion, which is degradation as a positive finding
rather than an absent one. **Both are invisible unmatched**, where far-middle is a flat +0.044 and
far-contra's +0.196 does not survive correction.

For reference, the unmatched family's two Bonferroni survivors are acute near-contra **+0.168**
[0.039, 0.269] and subacute far-contra **+0.189** [0.005, 0.349].

**THE DATA-POVERTY OBJECTION RUNS THE OTHER WAY.** Too few same-day far-contra trials would give the
refit arm less to learn from and push the gap NEGATIVE. The observed gap is POSITIVE, so scarcity
makes this harder to obtain, not easier. Selection was checked rather than asserted: the
`_sessions.csv` carries **all 16 acute sessions at every one of the six positions**.

---

## 4. THE ENCODER — gain acutely, shape chronically

`epoch_11amp_encoder_amplitude_cue_working` · `epoch_11_encoder_gain_shape_cue_working` ·
`epoch_11c_encoder_ceiling_cue_working`.

**Fitted gain:** 0.943 pre → **0.361** ** acute → 0.687 ** subacute → **0.939** chronic (unmarked —
back to baseline).

| epoch | frozen EV | EV after rescale | ceiling (refit, cross-validated) |
|---|---|---|---|
| pre | 0.557 | 0.580 | 0.706 |
| acute | **−0.388** ** | 0.134 ** | 0.568 |
| subacute | 0.186 ** | 0.320 ** | 0.642 |
| chronic | 0.383 ** | 0.410 ** | **0.783** |

**ACUTELY THE FROZEN ENCODER IS WORSE THAN PREDICTING THE MEAN (−0.388), AND A PURE RESCALE
RECOVERS MOST OF WHAT IT LOST (0.134).** The acute deficit is largely a **GAIN** change — the
encoder's version of "present but misread". **Chronically rescaling buys almost nothing** (0.383 →
0.410), so what remains is a **SHAPE** change.

**THE CEILING TELLS THE SAME STORY AS THE REST DECODER.** A refit, cross-validated encoder reaches
**0.783 chronically — above the 0.706 it reached pre-stroke** — while the frozen one reaches only
0.410. Chronic cortex is *more* predictable than pre-stroke cortex, just not by the pre-stroke
model. **That is replacement, measured on the task side.**

---

## 5. WHERE THE CODE IS, AND WHERE IT GOES

### Position maps against the position-weighted rest baseline
`epoch_15r_position_RESTWref_cue_working`, from `kw.stat_rows`. Amplitude relative to each
position's own pre-stroke value.

| | Near Ipsi | Near Contra | Near Mid | Far Ipsi | Far Mid | **Far Contra** |
|---|---|---|---|---|---|---|
| acute / pre | 1.46 | 1.26 | 1.20 | 0.99 | 0.68 | **0.47** |

**Monotone near→far, and far-contra collapses to less than half** — the largest significant area in
the figure, **1,186 of 2,022 in-mask bins** (next largest 394). Subacute far-contra 0.89 with
**ZERO** significant bins; chronic 1.08. The frame-weighted reference (`_RESTref_`) gives
1.42 / 1.24 / 1.17 / 0.98 / 0.66 / **0.48** — identical ordering, 1–3% apart. Split-half reliability
r = 0.94–0.99.

**ALWAYS CHECK `suppressed` BEFORE QUOTING A CELL.** Rim-concentrated panels (>2× edge enrichment)
are suppressed as imaging-window artefacts. Currently suppressed: cue Far Ipsi acute−pre (2.365);
pre-cue Near Middle subacute−pre (3.695) and Far Contra chronic−pre (2.467).

### The pre-cue / post-cue dissociation — the finding worth chasing
**Far-contra pre-cue INCREASES acutely (1.57, 310 significant bins) while far-contra post-cue
collapses (0.47, 1,186 bins) — in the same sessions and the same trials.** The pre-cue position
signal is not simply lost with the motor output.
→ read `epoch_15r_position_RESTWref_precue_working.png` **beside** the cue figure; **no single panel
carries this claim.**

The same dissociation appears in best-match (`epoch_10_best_match_acc_*`): post-cue recovers to
**0.889** chronically while pre-cue stays at **0.583** subacute and chronic. **But that is a POOLED
figure — see §5b**, where the per-position breakdown shows the positions failing to recover at ENL
are near-middle, far-ipsi and far-middle, NOT far-contralateral (0.889).

### The lick-aligned arm is blind to the acute deficit BY CONSTRUCTION
Its Far Contra acute cell is **absent entirely** — there are no lick-aligned far-contra trials
acutely, because the animal stops licking there, which IS the deficit (§1: response rate 0.052).
Raw counts confirm it: far_R acute is **64 trials across 13 sessions** against ~1,100 at each near
position. **Do not read that absence as a null.**

### Geometry — acute is focal, chronic is diffuse
`epoch_8diagdelta_matrices_crossnobis_cue_working`, distance from each position's own pre-stroke
template (Bonferroni-corrected):

| epoch | nI | nM | nC | fI | fM | **fC** |
|---|---|---|---|---|---|---|
| acute Δ | +0.211 ** | +0.008 | +0.245 ** | +0.304 ** | +0.319 | **+1.017** ** |
| chronic Δ | **+0.927** ** | +0.526 ** | +0.660 ** | +0.281 ** | +0.693 ** | +0.447 ** |

**Acute displacement is POSITION-SPECIFIC** — far-contra +1.017, more than three times the next
largest, with near-middle unmoved (+0.008). **Chronic displacement is DIFFUSE** — every position has
moved and the largest is near-IPSI, not far-contra. A diffuse chronic displacement is what
replacement looks like in geometry, agreeing with the encoder ceiling (§4) and the rest decoder (§6).

---

## 5b. THE FOUR WINDOWS — REST, ENL, CUE, LICK

Every trial-aligned family renders on three arms, and rest is the fourth window with no
trial alignment. The windows are **not interchangeable**, and reading a conclusion from one of them
as if it applied to the code generally is the commonest error this section exists to prevent.

| window | what it is | family |
|---|---|---|
| **REST** | inter-trial, spout DOCKED and out of reach, no target present | `epoch_15f/15s/15x/15d` |
| **ENL (pre-cue)** | after the spout arrives, before the cue — the anticipatory window | `*_precue_working` |
| **CUE (post-cue)** | 0 to +2 s from the cue — the evoked window | `*_cue_working` |
| **LICK (post-lick)** | aligned to the first lick — the execution window | `*_lick_lick` |

### Pre-stroke, information increases toward movement
Frozen-decoder accuracy, far-contralateral (`epoch_acc_by_position_*.csv`, chance 0.167):
**ENL 0.507 · CUE 0.896 · LICK 0.952.** The ENL window carries real but much weaker position
information — near-middle and far-middle sit at 0.453 and 0.413, barely above chance. **Any ENL
claim starts from a low baseline and should be stated as such.**

### Acute: far-contralateral, fraction of above-chance performance retained

| window | pre | acute | retained |
|---|---|---|---|
| ENL | 0.507 | 0.250 ** | **0.24** |
| CUE | 0.896 | 0.321 ** | **0.21** |
| LICK | 0.952 | 0.500 ** | **0.42** |

**THE LICK WINDOW'S APPARENT RESILIENCE IS SELECTION, NOT PRESERVATION.** It conditions on a
detected lick, and acutely the animal barely licks far-contralateral — **64 trials across 13
sessions, against ~1,100 at each near position**. What survives there is the subset of trials the
animal did attempt, so 0.42 is "when the animal managed it, the code was better preserved", which is
a different statement from "the code was better preserved". The lick arm cannot be used for an acute
far-contra claim; in the map families its far-contra acute cell is **refused entirely** by the
20-trial floor.

### Chronic: the windows recover differently
Best-match to each position's own pre-stroke template (`epoch_10b_best_match_by_position_*.csv`):

| window | nI | nM | nC | fI | fM | fC |
|---|---|---|---|---|---|---|
| ENL chronic | 0.667 * | **0.111** ** | 0.944 | **0.444** ** | **0.444** ** | 0.889 |
| CUE chronic | 1.000 | 0.556 | 1.000 | 0.889 | 0.889 | **1.000** |
| LICK chronic | 1.000 | 0.556 | 1.000 | 1.000 | 0.889 | 1.000 |

**CUE and LICK recover their template match almost completely. ENL does not** — and the failure is
**position-specific**: near-middle collapses to 0.111 and the far-ipsi/far-middle pair to 0.444,
while **far-contralateral ENL recovers to 0.889**.

> **A CORRECTION.** An earlier version of this document said "pre-cue does not recover at all
> (0.583)". That number is the POOLED average across positions
> (`epoch_10_best_match_acc_precue_working.csv`) and it is correct as a pooled figure, but read as a
> statement about the pre-cue code it is too coarse: the positions that fail to recover are
> **near-middle, far-ipsi and far-middle — not far-contralateral**, which is the position the lesion
> targets. The pooled number and the per-position breakdown answer different questions.

**So the position whose ENL representation never returns is not the one with the behavioural
deficit.** That is the same pattern the chronic crossnobis shows (§5: largest displacement at
near-**ipsi**), and it is the strongest evidence here that chronic reorganisation is not confined to
the impaired representation.

### The matched frozen-vs-refit gap, by window
`epoch_5rmgapdelta_frozen_vs_refit_*.csv`, acute, far-contralateral: **ENL +0.006 · CUE +0.158 ·
LICK —** (refused). **The displaced-but-present signature is a POST-CUE phenomenon.** At ENL the
matched gap is essentially zero at every position acutely (+0.088 to −0.012), meaning a same-day
refit recovers nothing the frozen model missed — consistent with there being little ENL position
information to recover in the first place.

### Rest, for comparison
Rest is the only window measurable when the animal does not attempt the task, and it carries
position robustly (1.634 observed/null, 43/44 sessions). But it is **a different code** — cosine to
the task map 0.09 post-cue and 0.16 pre-cue (§6) — and its far-contralateral representation
**holds up acutely** (0.50 → 0.38) where the task's collapses. **Rest is a complementary readout,
not a proxy for any of the three trial windows.**

---

## 6. THE REST ARM — post-audit values

All values from runs with the engagement gate applied AND the repaired classifier. Figures:
`epoch_15f_rest_frozen_restdock05*`, `epoch_15s_shared_position_restdock05`,
`epoch_15x_REST_by_position_by_animal`, `epoch_15d_delta_rest_flatpool_minus_restw`.

### Rest carries position — the solid one
Circular-shift permutation, docked, 200 permutations: **observed/null 1.634, 43/44 pre-stroke
sessions, 4/4 animals** (PS92 1.467, PS93 1.999, PS94 1.724, PS95 1.458). Per-session decode:
**94 of 96 sessions above their own null.** Map-level: between/within-position RMS ratio **1.97**,
largest significant area 1,468 of 2,022 bins, 6/6 positions in 4/4 animals.

> **Do NOT read a trajectory off the per-epoch obs−null column.** It is a cohort mean, and the
> acute-dip claim derived from exactly that quantity was WITHDRAWN on 2026-09-15 because PS95 rises.

### 15d — the composition bias GROWS post-stroke, so `restw` is load-bearing
`epoch_15d_delta_rest_flatpool_minus_restw.{png,csv}`. The delta is `flatpool − restw`, both arms
FLAT so only COMPOSITION differs (production `rest` is time-local, so `rest − restw` would move on
two axes at once — the confound that withdrew `rest_vs_restw`). **One map per epoch, not six: the
position's own data cancels**, `flatpool_q − restw_q = restw − quiet_flat`.

| | pre | acute | subacute | chronic | chronic/pre |
|---|---|---|---|---|---|
| cohort | 0.00056 | 0.00059 | 0.00071 | 0.00145 | **2.56** |
| PS92 | 0.00081 | 0.00154 | 0.00369 | 0.00205 | 2.51 |
| PS93 | 0.00061 | 0.00052 | 0.00055 | 0.00106 | 1.74 |
| PS95 | 0.00062 | 0.00227 | 0.00179 | 0.00132 | 2.14 |

**PREDICTION CONFIRMED: the bias is ~2.6× larger chronically than pre-stroke, in 3/3 animals with
chronic data** (PS94 has none). A frame-weighted rest baseline drifts with the deficit, exactly as
`restw` was designed to prevent. **`restw` is correct AND load-bearing, not correct-but-inert** —
so the 200-frame floor, the 4-of-6 rule and the column-drop apparatus are warranted.

### 15f — the pre-stroke rest code is partly REPLACED, not recovered
Frozen pre-stroke rest decoder, all four controls (duration-, lick-gap- and training-set-matched,
`blockperm` null), LOSO pre baseline.

| epoch | frozen retained | refit retained |
|---|---|---|
| pre | 1.000 | 1.000 |
| subacute | 0.306 | 0.575 |
| **chronic** | **0.611** | **1.181** |

**Chronically the within-session refit reaches 1.181 of its own pre-stroke level while the frozen
model reaches 0.611** — position is in chronic rest, read by something other than the pre-stroke
code. Both arms use the SAME periods, so the only difference is the estimator.

> **ACUTE IS UNDERPOWERED — do not quote it.** Gated per animal: 0.609 / 0.277 / 0.086 / **−0.093**.
> Removing ~4% of periods moved it that far because **acute is 16 sessions and PS95 contributes
> ONE**. It is not a 4/4 replication.

**CONFUSIONS — acute errors COLLAPSE, they do not scatter.** Acute predictions pile onto near-contra
in every row and far-row errors shift toward near columns (0.56 → 0.67). **Far-contra REST is the
best-decoded position pre-stroke (0.50) and holds up (0.38 acute, 0.39 chronic) — the OPPOSITE of
far-contra TASK, which collapses 1.46 → 0.47.** Rest and task dissociate at the lesioned position.

### 15s — the rest code is NOT the task code, and the pre-cue trace points BACKWARDS
`shared_p = <rest_p − restw, trial_p − restw> / ||trial_p − restw||²`, rest from ODD position-blocks
and trials from EVEN so shared drift cannot manufacture it, circular-shift null. Null-corrected,
animal as unit:

| window | pre | acute | chronic |
|---|---|---|---|
| pre-cue | **+0.060** (4/4) | **+0.188** (4/4) | +0.077 (3/3) |
| post-cue | −0.002 | +0.005 | +0.000 |

**Post-cue is flat against its null at every epoch; pre-cue is consistently positive.**

**BUT THE EFFECT IS SMALL.** The cosine between a position's task map and its own rest map is
**0.09 post-cue, 0.16 pre-cue** — close to unrelated, and not a magnitude artefact (the amplitude
ratio is 0.605 pre-cue, so a strong shape match would have shown). **Rest carries position in a
largely DIFFERENT spatial code from the task**, consistent with the 15f confusions above.

**WHICH WAY THE SHARED COMPONENT POINTS.** Within a block the position just licked and the one
coming next are identical; only at a block BOUNDARY do they separate. Gated:
**r_prev − r_next = +0.0772, 4/4 animals**, 2,983 boundary periods. **Rest resembles the position
JUST LICKED AT.** So the pre-cue window's shared component is a trace of the LAST target, not
preparation for the next — **a real constraint on reading the pre-cue signal as anticipatory**, and
consistent with this project's standing refusal to call it "a maintained motor plan".

> **EXCLUDE 15s SUBACUTE** — it is PS92 alone (−0.639 against +0.212, +0.044, +0.027), and PS92 is
> the cohort's SNR floor (the six lowest rest fractions are all PS92).

---

## 7. THE SPECIFICITY CONTROL — does everything degrade, or only the target?

The lesion is ventrolateral STRIATAL, so no cortex is damaged. Same window, same basis, same
estimator, same frozen-model discipline — **only the LABEL changes**
(`epoch_13n_state_vs_position_cue`; full record in `docs/BEHAVIOURAL_STATE_CONTROL.md`).

As a fraction of above-chance performance retained, acute vs pre:

| | pre | acute | retained |
|---|---|---|---|
| POSITION (6-way) | 0.863 | **0.428** | **0.496 — loses 50%** |
| STATE (3-way) | 0.976 | **0.873** | **0.894 — loses 11%** |

**Both fall. Position falls roughly five times further.** The claim is the RATIO, never that the
state readout is flat — and its worst epoch is SUBACUTE (0.771), not acute, so the two do not even
share a temporal pattern.

---
---

## 9. WHAT WE CAN AND CANNOT CONCLUDE — by hypothesis

Each hypothesis names **the best available test**, the number, and the verdict. Where a matched and
an unmatched version of a test exist, **the matched one is authoritative** and the unmatched one is
reported only as a bracket. File paths are relative to
`N:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/`; every figure has a same-named `.csv`.

---

### H1 — "The lesion degrades cortical position coding."
**SUPPORTED, with a graded spatial signature.**

*Best test:* frozen pre-stroke decoder, per position, on engaged + miss-while-working trials so
non-responses are not silently dropped — `epoch_acc_by_position_cue_working.{png,csv}` and
`epoch_accdelta_by_position_cue_working.csv`.

Acute change from pre, Bonferroni-corrected: **far-contra −0.575 [−0.738, −0.394]**, far-middle
−0.407, far-ipsi −0.373, near-contra −0.313, near-middle −0.271; near-ipsi −0.234 does not survive
correction. **Five of six positions significant, monotone near→far and ipsi→contra**, matching the
behavioural gradient (§1) position for position.

*Cannot conclude:* that cortex is the site of the deficit. The lesion is striatal; these are
downstream cortical consequences.

---

### H2 — "The deficit is specific to position, not a general cortical failure."
**SUPPORTED, and this is the control that makes H1 interpretable.**

*Best test:* the same window, basis, estimator and frozen-model discipline with **only the label
changed** — `epoch_13n_state_vs_position_cue.{png,csv}`, full record in
`docs/BEHAVIOURAL_STATE_CONTROL.md`.

As fraction of above-chance performance retained, acute vs pre: **position 0.496** (0.863 → 0.428),
**behavioural state 0.894** (0.976 → 0.873). **Position falls roughly five times further.**

*Cannot conclude:* that the state readout is unaffected. It falls too, and its worst epoch is
**subacute (0.771), not acute** — so the two do not share a temporal pattern, which is further
evidence against a common cause. **The claim is the RATIO.**

---

### H3 — "Acutely the code is LOST." vs "It is PRESENT but the pre-stroke readout can no longer see it."
**PARTIALLY SUPPORTED for displacement, and the honest answer is position-dependent.**

*Best test:* **the training-set-MATCHED frozen-vs-refit family**,
`epoch_5rmgapdelta_frozen_vs_refit_cue_working.csv` — **not** the unmatched `5rgapdelta`. The
unmatched frozen arm trains on ten pre-stroke sessions against the refit's four fifths of one, so
its pre row carries a size handicap with no lesion in it. Matching removes that and **flips the pre
gap from −0.116 to +0.036**, which also shows the no-lesion baseline is *bracketed rather than
known*.

Matched acute delta (epoch − pre), Bonferroni-corrected:

| position | point | corrected interval | |
|---|---|---|---|
| **far-contra** | **+0.158** | **[0.0004, 0.337]** | the only acute cell surviving |
| near-contra | +0.128 | [−0.060, 0.290] | |
| **far-middle** | **−0.132** | [−0.291, 0.035] | uncorrected [−0.242, −0.021] |

**At far-contralateral the information is PRESENT and DISPLACED** — a same-day refit reads a
position the frozen model cannot. **At far-middle the gap goes NEGATIVE**: refitting buys *less*
there than before the lesion, which is degradation as a positive finding rather than an absent one.
Both are invisible in the unmatched family, where far-middle is a flat +0.044.

*Cannot conclude:* a single verdict for "the code". Two adjacent positions behave oppositely.
*Cannot conclude:* that scarcity explains it — too few far-contra trials would push the gap
**negative**, so it makes a positive gap harder to obtain. All 16 acute sessions contribute at all
six positions (`epoch_5rgap_..._sessions.csv`).

---

### H4 — "The acute change is a change of GAIN, not of spatial pattern."
**SUPPORTED acutely; REVERSED chronically.**

*Best test:* the encoder's gain/shape decomposition, which asks whether a pure rescale of the
frozen model recovers what it lost — `epoch_11_encoder_gain_shape_cue_working.{png,csv}`.

| epoch | frozen EV | EV after rescale | interpretation |
|---|---|---|---|
| acute | **−0.388** ** | **0.134** ** | worse than predicting the mean; a rescale recovers most → **GAIN** |
| chronic | 0.383 ** | 0.410 ** | rescaling buys almost nothing → **SHAPE** |

Corroborated independently by geometry — `epoch_8diagdelta_matrices_crossnobis_cue_working.csv`:
**acute displacement is FOCAL** (far-contra +1.017, >3× the next largest; near-middle +0.008, i.e.
unmoved) while **chronic displacement is DIFFUSE** (all six positions +0.28 to +0.93, largest at
near-**ipsi**).

*Cannot conclude:* that the acute gain change is purely neural. Acutely the animal barely attempts
far-contralateral (response rate 0.052), so attempt-related drive is genuinely absent — a real
biological contributor to reduced gain, not an artefact, but it means "gain" here is not
necessarily synaptic.

---

### H5 — "Recovery restores the pre-stroke code."
**NOT SUPPORTED. This is the central negative result, and three independent estimators agree.**

*Best test:* any frozen-vs-refit contrast at chronic, because a restored code would be readable by
the pre-stroke model while a replaced one is readable only within session.

| estimator | frozen (pre-stroke model) | refit (same-day) | file |
|---|---|---|---|
| **task encoder** (matched) | 0.307, i.e. **0.715 of pre** | ceiling **0.783 = 1.109 of pre** | `epoch_11c_encoder_ceiling_cue_working.csv` |
| **rest decoder** | **0.611** retained | **1.181** retained | `epoch_15f_rest_frozen_restdock05*.csv` |
| **best-match, pre-cue** | **0.583**, flat from subacute | — | `epoch_10_best_match_acc_precue_working.csv` |

**A refit encoder reaches 0.783 chronically — ABOVE its pre-stroke 0.706 — while the frozen one
reaches 0.410.** Chronic cortex is *more* predictable than pre-stroke cortex, just not by the
pre-stroke model. Meanwhile behaviour (0.988/0.987/0.982 near, 0.947 far-ipsi) and per-position
decoding (only far-contra still below pre) have largely recovered.

**Behaviour recovers; the readout does not.** Recovery is reorganisation, not restitution.

*Cannot conclude:* **what** the new code is. "Replacement" is inferred from the divergence between a
frozen and a refit model, not from identifying the substitute representation.
*Cannot conclude:* that the new code *causes* the recovered behaviour. This is correlational; no
causal manipulation was performed.
*Cannot conclude:* chronic claims for PS94, which has **no chronic sessions**.

---

### H6 — "The preparatory (pre-cue) signal follows the same trajectory as the execution (post-cue) signal."
**NOT SUPPORTED — they dissociate, in two independent measures.**

*Best test:* the same analysis run on both windows.

* `epoch_15r_position_RESTWref_precue_working.png` vs `..._cue_working.png` — **far-contra pre-cue
  INCREASES acutely (1.57, 310 significant bins) while far-contra post-cue COLLAPSES (0.47, 1,186
  bins), in the same sessions and the same trials.** **No single panel carries this claim**; it
  requires both.
* `epoch_10_best_match_acc_precue_working.csv` vs `..._cue_working.csv` — post-cue best-match
  recovers to **0.889** chronically; **pre-cue does not recover at all, 0.583 → 0.583.**

**AND THE WINDOWS RECOVER DIFFERENTLY (§5b).** Best-match chronically: CUE and LICK return to
~1.000 at almost every position; **ENL does not, and the failure is position-specific** —
near-middle 0.111, far-ipsi 0.444, far-middle 0.444, while **far-contralateral ENL recovers to
0.889**. So the position whose anticipatory representation never returns is **not** the one with the
behavioural deficit.

*Cannot conclude:* that the LICK window is more resilient. Its acute far-contra retention (0.42
against CUE's 0.21) is **SELECTION** — it conditions on a detected lick, and acutely there are only
**64 far-contra lick trials across 13 sessions** against ~1,100 at each near position. In the map
families that cell is refused outright by the 20-trial floor.

*Cannot conclude:* that the pre-cue signal is a motor plan or an intention. The spout arrives ~3 s
before the cue, so a sustained sensory response and a held intention are temporally coextensive and
**this design cannot separate them**. §6's boundary result tightens this further: the component
pre-cue shares with rest points **backwards**, toward the last target.

---

### H7 — "Resting activity carries the position code, and tracks the same trajectory."
**FIRST HALF SUPPORTED; SECOND HALF NOT — rest carries position in a largely DIFFERENT code.**

*Best test:* circular-shift permutation that keeps block-time structure inside the null, plus a
frozen/refit split on the same rest periods.

* **Rest carries position:** observed/null **1.634**, 43/44 pre-stroke sessions, 4/4 animals;
  94/96 sessions decode above their own null
  (`epoch_15x_REST_by_position_by_animal.csv`, `docs/REST_ENGAGEMENT_AUDIT.md`).
* **But it is not the task code:** cosine between a position's task map and its own rest map is
  **0.09 post-cue, 0.16 pre-cue** — close to unrelated, and not a magnitude artefact
  (`epoch_15s_shared_position_restdock05.csv`).
* **And it dissociates from task at the lesioned position:** far-contra REST is the best-decoded
  position pre-stroke (0.50) and **holds up** (0.38 acute, 0.39 chronic), while far-contra TASK
  collapses 1.46 → 0.47 (`epoch_15f_..._confusion.csv`).

*Cannot conclude:* anything from 15f's **acute** cell — 16 sessions with PS95 contributing one, and
a 4% data change moves it by up to 0.21.
*Cannot conclude:* that rest is free of undetected orofacial movement. The lick sensor is
contact-thresholded and the spout docks out of reach, so licking at nothing is invisible; the
stratified control **bounds** this but cannot measure it. **DLC tongue tracking is the only thing
that would settle it, and it is not built.**

---

## 10. WHAT THIS CHANGES ABOUT HOW WE THINK ABOUT STROKE RECOVERY

### The measurement that was not previously available
Recovery is almost always scored on **behaviour**: does the animal, or the patient, do the task
again? That measurement is blind by construction to *how* the nervous system is doing it. What this
dataset adds is a **frozen-model readout** — a decoder and an encoder fixed on the pre-injury code
and never refitted — run beside a same-day refit on identical trials. The pair separates two things
a behavioural score conflates:

* **restitution** — the pre-injury representation returns, so a frozen model recovers with behaviour;
* **substitution** — a different representation supports the same behaviour, so behaviour recovers
  while the frozen model does not.

**Here it is substitution.** Behaviour returns to 0.95–0.99 and per-position decoding to within
noise of pre-stroke at five of six positions, yet the pre-stroke encoder recovers only to 0.715 of
its matched baseline while a same-day refit exceeds pre-stroke performance outright (1.109). The
information is there in greater quantity than before; the old readout cannot see it.

### Recovery is at least two processes, not one
The acute and chronic deficits are different in kind, and each is characterised by its own best test:

* **Acute = a focal GAIN loss.** A rescale of the frozen encoder recovers most of what it lost
  (−0.388 → 0.134), and displacement is concentrated at far-contralateral (+1.017) with near-middle
  untouched (+0.008).
* **Chronic = a diffuse SHAPE change.** Rescaling recovers almost nothing (0.383 → 0.410), and every
  position has moved, the largest being near-**ipsi** — a position with no acute deficit at all.

**The chronic reorganisation is not confined to the impaired representation.** Positions that were
never behaviourally affected have moved the most by chronic. That is hard to reconcile with a
purely local repair account and easier to read as a network-wide re-solution of the task.

### Practical consequences
1. **A fixed decoder calibrated pre-injury will fail chronically even in a behaviourally recovered
   subject** — 0.715 of baseline for the encoder, 0.611 for the rest decoder. Neuroprosthetic and
   BCI approaches that assume representational stability across a lesion need recalibration on a
   timescale this dataset can now quantify.
2. **Interventions should be timed to the mechanism.** Anything aimed at restoring gain addresses
   the acute phase; anything aimed at the chronic phase is addressing a changed spatial pattern, and
   the two are not the same target.
3. **Resting activity gives a task-free readout.** A severely impaired subject cannot perform the
   task, so task-based measures are confounded by inability to attempt — precisely the trap that
   makes the lick-aligned arm blind to acute far-contralateral (64 trials against ~1,100). Rest
   carries position robustly (1.634, 43/44) and requires no attempt, so it remains measurable
   exactly when the task-based readout fails. **But it is a different code** (cosine 0.09–0.16), so
   it is a complementary readout, not a substitute for the task one.
4. **"Recovered" needs to be stated at a level.** Behaviourally recovered, representationally
   reorganised, and readout-incompatible are three different states, and this cohort is all three at
   once.

### What would move this from suggestive to established
* **Identify the substitute code**, rather than inferring it from frozen/refit divergence.
* **A causal test** — does perturbing the new representation disrupt the recovered behaviour?
* **More animals, and chronic data for PS94**; n=4 with one animal missing the chronic epoch and
  another contributing a single acute session is the main limit on every chronic claim here.
* **DLC tongue tracking**, to close the undetected-movement gap in the rest arm.
* **A second lesion model or task**, since everything here is one striatal lesion and one
  six-position licking task.
