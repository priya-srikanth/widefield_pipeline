# Preliminary data: what VLS stroke does to cortical position coding

**Status: DRAFT for a career-development grant. Numbers are POINT ESTIMATES READ OFF THE POOLED
EPOCH FIGURES, not pulled from the underlying JSON. Pull exact values and intervals before
submission.** Source run: `nightly_figs 20260907`, completed 2026-09-08 15:13, 0 failed steps,
96/96 grant units. Figures under `labcams/grant_figures/epoch/`.

Written 2026-09-09 at Priya's request, so the reasoning behind the paragraph is recoverable rather
than living in a chat log.

---

## STATE OF THE EVIDENCE, **2026-09-16** -- SUPERSEDES EVERY CAVEAT BELOW

**THE SOURCE RUN FOR THIS DOCUMENT (`nightly_figs 20260907`) IS TWO GENERATIONS STALE.** The current
render is **2026-09-16 06:49**, on the `restdock05` rest definition, 877 slides / 1,333 figures /
0 missing. Everything in this document predates the entire rest-baseline migration.

**WHAT CHANGED UNDER THE NUMBERS**

1. **The rest baseline is a different quantity.** `restdock05` = docked window (spout parked, no
   target present) + `lick_buffer_s [0.5, 1.0]` + `treadmill_buffer_s [1.0, 2.0]`. Every
   rest-referenced amplitude in this document was computed on a retired definition.
2. **THE EPOCH TABLE BELOW IS ALSO WRONG NOW.** It says PS95 chronic = day 15. The boundaries are
   DERIVED from behaviour and the 2026-09-16 render's own audit gives **PS92 11, PS93 11, PS94 none,
   PS95 11**, matching `animals.yaml` on all four animals. PS95 moved 15 -> 11.
3. **`restw` is NOT retired** (see DECISIONS.md 2026-09-15). Rest carries position: observed/null
   **1.622**, **44/44** pre-stroke sessions, 4/4 animals, 0 skipped.
   FIGURES: `epoch_15x_REST_by_position_by_animal.png` (the rest baseline's own per-position
   structure — if rest were position-independent every cell would be noise; it is not), and
   `E:/cue_lick/rest_migration/rest_position_decode_restdock05.png` (per-session decode, 93/94
   sessions above their own circular-shift null). **The 1.622 itself has NO figure** — it is
   printed by `scripts/rest_migration/rest_position_permutation.py`, and 15x is the picture of the
   same phenomenon measured a different way, not a plot of that number.

**VERIFIED CURRENT NUMBERS — safe to quote, read off the 2026-09-16 render**

Acute amplitude relative to each position's own pre-stroke value, position-weighted flat rest
reference (`epoch_15r_position_RESTWref_cue_working`):

| position | acute / pre |
|---|---|
| Near Ipsi | 1.46 |
| Near Contra | 1.26 |
| Near Middle | 1.20 |
| Far Ipsi | 0.99 |
| Far Middle | 0.68 |
| **Far Contra** | **0.47** |

The frame-weighted reference (`_RESTref_`) gives 1.42 / 1.24 / 1.17 / 0.98 / 0.66 / **0.48** --
**identical ordering, 1-3% apart**, and far-contra is the ONLY position where the two disagree in
direction, which is exactly where composition bias is predicted. Split-half reliability r = 0.94-0.99
throughout.

### POSITION MAPS vs THE POSITION-WEIGHTED REST BASELINE — EXACT VALUES

Pulled from `kw.stat_rows` in `epoch_15r_position_RESTWref_*_bundle.json` (2026-09-16 render), NOT
read off the figures. `amplitude_vs_pre` = that cell's amplitude relative to its own pre-stroke
value; `sig` = bins significant under the nested animals->sessions bootstrap, max-statistic
corrected, out of 2,022 in-mask bins.

**ALL FIGURES BELOW ARE IN `N:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/`**, and every
number is from the matching `*_bundle.json` `kw.stat_rows` or `.csv` in the **2026-09-16 06:49**
render. Figure and statistic are the SAME quantity by construction, so a claim can be checked
against its panel.

**PROVENANCE — VERIFIED FILE TIMES, because these paths get overwritten.** Every value below was
read from files stamped **09-16 03:50–06:58** (the dev-box render 03:38→06:50, plus the 15x
re-render at 06:58), confirmed at 08:46 as not yet replaced:
`epoch_5rgap_frozen_vs_refit_cue_working.csv` 05:13 · `epoch_12b_stopped_pooled_similarity_cue.csv`
05:09 · `..._precue.csv` 03:50 · `epoch_15r_position_RESTWref_cue_working_bundle.json` 04:46 ·
`epoch_15x_...png` 06:58.

**THE NIGHTLY BOX WRITES TO THESE SAME PATHS.** Its run was landing epoch figures from ~08:45 on
2026-09-16, and its date list may differ from this render's (which ends at **0914** and excludes the
9/15 sessions). **Check a file's mtime before assuming a number here still matches what is on
disk** — a table that silently mixes two runs is the failure this stamp exists to prevent.

**POST-CUE, working trials — the headline, and a clean monotone gradient**
FIGURE: `epoch_15r_position_RESTWref_cue_working.png`
· per animal: `epoch_15rpa_position_RESTWref_by_animal_cue_working.png`
· frame-weighted counterpart (agrees to 1-3%): `epoch_15r_position_RESTref_cue_working.png`

| position | acute/pre | sig | subacute/pre | sig | chronic/pre | sig |
|---|---|---|---|---|---|---|
| Near Ipsi | 1.46 | 394 | 1.26 | 126 | 1.50 | 390 |
| Near Contra | 1.26 | 237 | 1.19 | 123 | 1.25 | 329 |
| Near Middle | 1.20 | 67 | 1.15 | 145 | 1.16 | 285 |
| Far Ipsi | *0.99* | *20* | 1.02 | 0 | 1.24 | 71 |
| Far Middle | 0.68 | 148 | 0.97 | 1 | 1.13 | 167 |
| **Far Contra** | **0.47** | **1186** | 0.89 | **0** | 1.08 | 235 |

1. **Acute is monotone near -> far: 1.46, 1.26, 1.20 | 0.99, 0.68, 0.47.** Near positions
   INCREASE, far positions decrease, far-contra collapses to less than half.
2. **Far-contra acute is the single largest significant area in the figure — 1,186 of 2,022 bins
   (59%).** Nothing else approaches it; the next largest is 394.
3. **Subacute far-contra is 0.89 with ZERO significant bins** — not distinguishable from pre-stroke.
4. **Chronic far-contra is 1.08 (235 sig)** — at or slightly above baseline.
5. *Far Ipsi acute (italic) is **SUPPRESSED**: edge_enrichment 2.365, i.e. >2x concentrated in the
   mask rim. **Do not quote it.*** It is an imaging-window artefact, not a result.

**PRE-CUE — DISSOCIATES FROM POST-CUE, and this is the finding worth chasing**
FIGURE: `epoch_15r_position_RESTWref_precue_working.png`
· per animal: `epoch_15rpa_position_RESTWref_by_animal_precue_working.png`
· **READ IT BESIDE THE POST-CUE FIGURE ABOVE** — the dissociation is a comparison ACROSS two
  figures, so neither panel shows it on its own. That is the one claim here no single figure carries.

| position | acute/pre | sig | chronic/pre | sig |
|---|---|---|---|---|
| Near Ipsi | 1.90 | 0 | 2.67 | 14 |
| Near Middle | 0.47 | 46 | 0.69 | 468 |
| Near Contra | 0.78 | 0 | 1.00 | 45 |
| Far Ipsi | 1.28 | 40 | 0.79 | 0 |
| Far Middle | 0.94 | 0 | 0.55 | 740 |
| **Far Contra** | **1.57** | **310** | *0.72* | *15* |

**Far-contra pre-cue INCREASES acutely (1.57, 310 sig) while far-contra post-cue COLLAPSES (0.47,
1186 sig), in the same sessions and the same trials.** The pre-cue position signal is not simply
lost with the motor output. Chronically the pattern inverts again: the largest sustained decreases
are Far Middle 0.55 (740 sig) and Near Middle 0.69 (468 sig).
*Two cells SUPPRESSED (do not quote): Near Middle subacute-pre (edge 3.695), Far Contra chronic-pre
(edge 2.467).*

**LICK-ALIGNED — and the reason it cannot carry the headline**
FIGURE: `epoch_15r_position_RESTWref_lick_lick.png` (the Far Contra / acute-pre cell is BLANK)

Acute cells are uniformly elevated (Near Ipsi 1.54, Near Contra 1.38, Far Ipsi 1.43, Far Middle
1.39, Near Middle 1.32) and **the Far Contra acute cell is ABSENT ENTIRELY.** There are no
lick-aligned far-contra trials to average acutely, because the animal stops licking there — which IS
the deficit. **The lick-aligned arm is blind to the acute far-contra effect by construction**, and a
reader comparing alignments would otherwise read its absence as a null. Far Middle acute carries
edge_enrichment 1.633 (elevated, below the 2x suppression threshold).

### FROZEN vs REFIT — "reorganisation"
FIGURES: `epoch_5rgap_frozen_vs_refit_cue_working.png` (raw gap) ·
`epoch_5rgapdelta_frozen_vs_refit_cue_working.png` (epoch − pre, the actual claim) ·
`epoch_5rmgap*` / `epoch_5rmgapdelta*` (the TRAINING-SET-MATCHED family) · DATA: same-named `.csv`.

**SIGN CONVENTION: the plotted quantity is `refit − frozen`** — `_gap_at` in
`epoch_grant_figures.py` returns `mean(refit correct) − mean(frozen correct)` on the paired record
whose column 0 is the frozen model and column 1 the within-session refit, and the figure's own
`ylabel` reads `refit - frozen accuracy`. **POSITIVE = the session's own decoder reads a position
the frozen pre-stroke model cannot: information PRESENT but DISPLACED. Zero with both arms low = the
code is genuinely degraded.** An earlier version of this section stated the convention as "frozen
minus refit" and read every sign backwards; the numbers below are unchanged, the reading is inverted.

**RAW GAP, post-cue working (`5r`)** — `*` = interval excludes zero:

| epoch | nI | nM | nC | fI | fM | fC |
|---|---|---|---|---|---|---|
| pre | −0.050 | −0.099 | −0.058 | −0.065 | −0.077 | −0.087 |
| acute | −0.024 | **+0.033\*** | **+0.109\*\*** | −0.032 | −0.033 | **+0.109\*** |
| subacute | −0.026 | **+0.034\*** | −0.080 | +0.019 | −0.008 | **+0.102\*\*** |
| chronic | −0.058 | **+0.144\*** | −0.032 | −0.009 | +0.007 | −0.003 |

**THE PRE ROW IS THE CONTROL AND IS NOT ZERO BY CONSTRUCTION.** All six pre cells are negative
because the frozen arm trains on ten pre-stroke sessions (LOSO) and the refit arm on four fifths of
one — a training-set-SIZE handicap with no lesion in it. The claim is therefore the **delta**, not
the raw gap.

**DELTA (epoch − pre), post-cue working (`5rgapdelta`)** — point [95% CI] / [Bonferroni-corrected]:

| epoch | nC | fC | nM |
|---|---|---|---|
| acute | **+0.168 [0.081, 0.240] / [0.039, 0.269]** | +0.196 [0.034, 0.345] / [−0.039, 0.412] | +0.132 [0.026, 0.225] / [−0.030, 0.268] |
| subacute | −0.021 | **+0.189 [0.058, 0.300] / [0.005, 0.349]** | +0.133 [0.021, 0.237] / [−0.050, 0.300] |
| chronic | +0.026 | +0.084 [−0.018, 0.199] | +0.243 [0.024, 0.463] / [−0.088, 0.531] |

Two cells survive Bonferroni: **acute near-contra +0.168** and **subacute far-contra +0.189**. Acute
far-contra is the largest point estimate (+0.196) but its corrected interval crosses zero.

**THE TRAINING-SET-MATCHED FAMILY (`5rm`) EXISTS FOR EXACTLY THIS QUESTION** and is the reason the
data-poverty objection can be answered rather than deferred. `paired_matched` replaces the FROZEN
model only: it is refitted on a size-matched random subset of pre-stroke BLOCKS, drawn under the
same leave-one-session-out discipline, so both arms get the same amount of training data. What that
exposes is that the no-lesion baseline does not go to zero — it goes the **other way**: matched pre
gaps are all POSITIVE (+0.068 to +0.127), because a within-session fit shares that session's own
nuisance structure while a matched frozen model must generalise across days.

**So the no-lesion baseline is BRACKETED, not known**: −0.073 unmatched (frozen has ~10× the data)
and +0.090 matched (frozen must cross sessions). Both bounds are real; neither is "the" answer;
which is why both families are drawn and both are read as epoch-minus-pre.

**MATCHED DELTA (epoch − pre), post-cue working (`5rmgapdelta`), beside the unmatched:**

| | nI | nM | nC | fI | fM | fC |
|---|---|---|---|---|---|---|
| `5r` acute | +0.027 | +0.132 | **+0.168\*\*** | +0.034 | +0.044 | +0.196\* |
| `5rm` acute | +0.093 | +0.039 | +0.128\* | −0.036 | **−0.132\*** | **+0.158\*\*** |

**MATCHING SHARPENS THE DISSOCIATION RATHER THAN SHRINKING IT.** Far-contra keeps a significant
positive recoverable component once the handicap is removed (**+0.158, corrected interval
[0.0004, 0.337]**, the only acute cell that survives Bonferroni in the matched family), while
**FAR-MIDDLE turns negative (−0.132, uncorrected [−0.242, −0.021]; corrected interval crosses
zero)** — refitting buys *less* there than it bought before the lesion, which is "the code is
degraded" as a positive finding rather than as an absent one. In the unmatched family far-middle is
a flat +0.044 and this is invisible.

**WHY THE DATA-POVERTY OBJECTION RUNS THE OTHER WAY.** Too few same-day far-contra trials would give
the refit arm *less* to learn from and push the gap NEGATIVE. The observed gap is POSITIVE, so trial
scarcity makes this result harder to obtain, not easier — it is a conservative confound here, and
the matched family removes the one direction in which training-set size could manufacture the effect.

**WHAT IS ACTUALLY LEFT TO CHECK, and it is small.** Trials whose class a session could not train on
(`grant_figures.MIN_REFIT_CLASS = 10`, `MIN_REFIT_SHARE = 1/18`) are dropped from BOTH arms
together, which preserves the pairing but could in principle select sessions.
**It does not, in this arm:** `epoch_5rgap_frozen_vs_refit_cue_working_sessions.csv` carries
**all 16 acute sessions at every one of the six positions** (pre 44, acute 16, subacute 18,
chronic 18 — no position loses a session). The gating DOES bite in the lick-aligned arm, where acute
far-contra is absent entirely, and that absence is the deficit rather than a null (see the 15r lick
figure above). Remaining caveat worth stating on a slide: **acute n is 16 sessions and PS95
contributes only 1** (92:5 93:4 94:6 95:1, from the figure's own subtitle).

### PATTERN SIMILARITY TO PRE-STROKE (stopped/quit trials)
FIGURES: `epoch_12b_stopped_pooled_similarity_{cue,precue}.png` · DATA: same-named `.csv`

| window | comparison | pre | acute | subacute | chronic |
|---|---|---|---|---|---|
| cue | vs pre-stroke **STOPPED** | 0.841 | **0.639** | 0.786 | 0.868 |
| cue | vs pre-stroke ENGAGED | −0.032 | 0.016 | −0.047 | −0.037 |
| pre-cue | vs pre-stroke **STOPPED** | 0.704 | **0.450** | 0.593 | 0.566 |
| pre-cue | vs pre-stroke ENGAGED | 0.018 | 0.004 | 0.006 | **0.049** |

**The stopped pattern stays a stopped pattern.** Similarity to pre-stroke STOPPED is high (0.70–0.87)
while similarity to pre-stroke ENGAGED sits at ~zero in every epoch — the two states are distinct and
stay distinct. **Similarity dips acutely and recovers**: cue 0.841 → 0.639 → 0.786 → 0.868; pre-cue
0.704 → 0.450 → 0.593 → 0.566 (pre-cue does NOT return to baseline). The only ENGAGED cell whose
interval excludes zero is pre-cue chronic (+0.049) — small, and worth a second look before it is
called anything.

> **⚠ PRE-AUDIT NUMBERS — 2026-09-16.** The 15f and 15s sections below were written from the
> UNGATED runs, before the engagement-gate audit, and 15s additionally from a doubly-docked
> baseline. **Do not quote them.** `docs/REST_ENGAGEMENT_AUDIT.md` carries the re-run values:
> finding 11 **1.634 (43/44)**, persistence **+0.0772 (4/4)**, per-session decode **94/96**, 15f
> acute retained **0.609 / 0.277 / 0.086 / −0.093** per animal (underpowered — PS95 contributes ONE
> acute session; quote 15f's CHRONIC 0.611 frozen vs 1.181 refit instead), 15s null-corrected
> **+0.060 pre / +0.188 acute** pre-cue with post-cue flat. Rewriting this section is the first
> pending item in `docs/STATUS_2026-09-16.md` §4.

### REST, FROZEN DECODER — does the pre-stroke rest code survive, or is it REPLACED?
BUILT 2026-09-16. SCRIPT: `scripts/rest_migration/rest_frozen_decoder.py` ·
DATA: `E:/cue_lick/rest_migration/rest_frozen_decoder_restdock05_final.{csv,png}` +
`_sessions.csv`. 96 sessions, **0 skipped**, duration-matched, `blockperm` null, 200 permutations.

**THE QUESTION NO EARLIER FIGURE COULD REACH.** `rest_position_decode` fits a decoder WITHIN each
session, so a chronic session scoring well says only that THAT DAY'S rest carries position — by
whatever code that day happens to use. Recovery and replacement produce the identical number. This
arm trains on pre-stroke rest, freezes, and applies across epochs; the refit arm is computed on the
**same periods, same labels, same block vector**, so the only difference between the two is the
estimator.

| epoch | frozen acc | frozen retained | refit retained | gap (refit − frozen) | p<0.05 |
|---|---|---|---|---|---|
| pre (LOSO) | 0.385 | 1.000 | 1.000 | −0.094 | 44/44 |
| acute | 0.216 | **0.227** → **0.332** proximity-matched | 0.495 → 0.665 | +0.012 | **8/16** |
| subacute | 0.237 | 0.322 | 0.565 | −0.000 | 13/18 |
| chronic | 0.303 | **0.627** → **0.611** proximity-matched | **1.271** → **1.181** | +0.021 | 18/18 |

**THE PRE-STROKE REST CODE IS NOT SIMPLY RESTORED.** Chronically the within-session readout reaches
**1.27× its own pre-stroke level** while the frozen model reaches **0.627**. Position is present in
chronic rest and is read by something other than the pre-stroke code — the replacement signature.
All four animals dip acutely (PS92 0.401, PS93 0.187, PS94 0.134, PS95 0.120); **PS94 has no
chronic sessions** and **PS92's subacute 0.759 is far out of line** with the other three
(0.341, 0.229, 0.285), which is the same subacute non-replication flagged elsewhere.

**RETAINED 0.227 ACUTE, against the TASK arm's 0.496 and the state control's 0.902.** Same animals,
same joint LocaNMF footprints, same estimator, same frozen discipline — only the WINDOW changes.

**ACUTE IS MARGINAL AT THE SESSION LEVEL AND MUST BE QUOTED THAT WAY.** Only **8 of 16** acute
sessions reach p<0.05 under the block-permutation null; mean acute accuracy 0.216 sits essentially
at the null's 95th percentile (0.206). An earlier note of "16/16 above null" used `acc > null mean`,
which is not a test. The epoch-level effect and the 4/4 animal replication stand; the per-session
claim does not.

**THE BASIS IS THE JOINT LocaNMF ONE, AND IT HAD TO BE.** `rest_position_decode` uses each session's
OWN SVD components — correct within a session, meaningless frozen, because component *i* is a
different cortical patch on each day. A model carried across days in that basis returns a low number
that reads exactly like a lesion effect.

**THE NULL IS `blockperm`, NOT A CIRCULAR SHIFT — and this corrected a repo-wide claim.**
`rest_position_decode`'s docstring says a circular shift "keeps blocks as blocks". MEASURED: with
the unequal block lengths real data has, a roll misaligns the boundaries and leaves only **~31%** of
blocks internally constant, against **100%** for `blockperm` and **~1%** for a trial shuffle. The
shift therefore gives a null that is too weak. All three are computed: they agree on the null MEAN
(0.166) and disagree on significance exactly as predicted — acute 8 (blockperm) / 9 (shift) /
**11 (trial)** of 16. Second trap recorded in the tests: under a trial shuffle the BALANCED-accuracy
null is ~1/ncls by construction, so a balanced null sitting at 1/6 is NOT evidence the permutation
is working.

**DURATION CONTROL: INERT.** Median scored rest-period length moves **0.01 s** across epochs, because
these are docked periods between two same-position trials — the ITI sets their length, not how much
the animal rests. Retained 0.227 matched vs 0.224 unmatched.

**THE UNDETECTED-LICKING CONTROL (Priya, 2026-09-16).** *"PS92 is VERY licky pre-stroke and during
recovery so much of the 'rest' probably includes licks without spout contact (due to docked spout
position)."* The lick channel is threshold-on-CONTACT and the spout docks out of reach, so licking
at nothing produces no deflection, and `lick_buffer_s` — keyed on detected licks — cannot exclude
what was never detected. Stratifying the SAME frozen predictions by gap to the nearest detected
lick: **far − near is negative in 12 of 15 cells, INCLUDING pre-stroke in all four animals**
(−0.024 to −0.086). So it is a general property of the rest signal, not a post-stroke artefact.

**THE MEDIAN SPLIT WAS TOO WEAK, AND ITS REASSURING ANSWER DID NOT SURVIVE THE PROPER TEST.** An
earlier version of this section concluded the confound was bounded "at most ~10–25% of the effect"
from the median split alone. **That figure is WITHDRAWN.** A median split is animal-RELATIVE — PS92's
"far" half begins at 1.0 s, which licking bouts outlast — so it compares different things in
different animals, which is precisely wrong when one animal is the licky one. The absolute
**≥3 s** stratum (`--lick-far-s`, same threshold in every animal) says something much less
comfortable:

| animal | pre above-chance, ALL periods | ≥3 s from any lick | % of periods kept |
|---|---|---|---|
| PS92 | 0.224 | **0.057** | 7% |
| PS93 | 0.284 | **0.131** | 19% |
| PS94 | 0.325 | **0.165** | 19% |
| PS95 | 0.213 | 0.191 | 76% |

**In three of four animals roughly half to three-quarters of the PRE-STROKE rest position signal
sits within 3 s of a detected lick.** That is a far larger proximity dependence than the median
split implied.

**WHAT SAVES IT FROM BEING FATAL, AND WHAT DOES NOT.** The test is UNDERPOWERED in exactly the
animals where it matters: for PS92/PS93/PS94 the ≥3 s stratum is a rare 7–19% tail, and the
retained fractions recomputed inside it are unusable (PS92 subacute −1.697, PS93 acute −0.061 —
above-chance estimates going negative on thin samples). **PS95 is the one animal where the test IS
well powered** — 76% of its periods are ≥3 s from a lick — and there the signal is essentially
intact, 0.213 → 0.191, a 10% loss. So proximity dependence is not universal. But PS95 is also the
animal with the LOWEST retained fraction (0.120 acute), so it is not the one carrying the
survival story.

**A SECOND, SHARPER THREAT THIS SURFACED: LICK PROXIMITY ITSELF MOVES WITH EPOCH.** Acute animals
lick less, so their rest sits further from licks — PS93's median gap goes 1.21 s → 4.58 s, PS94's
1.00 s → 4.06 s. If decoding depends on proximity AND proximity changes across epochs, part of the
acute drop is COMPOSITION rather than code loss. Three of four animals are consistent with that
account: PS92's gap barely moves (1.00 → 1.25) and it has the SMALLEST acute drop (0.401); PS93 and
PS94 have the large gap shifts and the large drops (0.187, 0.134). **PS95 breaks it** — gap
essentially static (3.41 → 3.19) with the largest drop of all (0.120).

**RESOLVED BY PROXIMITY MATCHING (`--match-lickgap`), and the confound was REAL BUT PARTIAL.**
Restricting train and test to the central range of the PRE-STROKE lick-gap distribution — matching,
not stratifying, so the full sample survives — gives (also duration-matched, `blockperm` null,
96 sessions, 0 skipped):

| epoch | frozen acc | frozen retained | refit retained | p<0.05 |
|---|---|---|---|---|
| pre (LOSO) | 0.402 | 1.000 | 1.000 | 44/44 |
| acute | 0.259 | **0.332** (was 0.227) | 0.665 | 8/16 |
| subacute | 0.239 | 0.306 (was 0.322) | 0.575 | 11/18 |
| chronic | 0.311 | **0.611** (was 0.627) | **1.181** | 17/18 |

**THE ACUTE DROP WAS PARTLY COMPOSITION AND MOSTLY NOT.** Matching lick proximity moves acute
retained from 0.227 to **0.332** — so a real slice of the acute deficit was "acute animals lick
less, so their rest sits further from licks", exactly as the gap shift predicted. But two thirds of
the acute signal is still gone after the control, so the confound inflated the effect rather than
producing it.

**THE CHRONIC RESULT IS ROBUST TO THE CONTROL**: 0.627 → 0.611 frozen, and the refit arm still
exceeds its own pre-stroke level (1.271 → **1.181**). The replacement conclusion does not depend on
lick proximity, which is expected — it is a within-epoch contrast between two estimators on one set
of periods, and proximity composition cancels.

**WHAT MOVED PER ANIMAL, AND WHY ONE NUMBER SHOULD NOT BE QUOTED.** Acute retained, gap-matched:
PS92 0.614, PS93 0.270, PS94 0.111, PS95 **0.748**. PS95 swung from 0.120 to 0.748 — **it has
exactly ONE acute session**, so its acute cell is a single measurement and moves freely under any
reweighting. PS92 0.401 → 0.614 is the licky animal gaining most from the control, which is the
predicted direction. **PS94 is stable (0.134 → 0.111)** and is the cleanest acute cell.

**DO NOT COMPARE 0.332 DIRECTLY TO THE TASK ARM'S 0.496.** Only the rest arm has been
proximity-controlled. The task arm's window is post-cue trials, where licking IS the behaviour
rather than a contaminant, so the same control does not transfer unmodified — the comparison needs
either a matched task-side analysis or an explicit statement that one side is controlled and the
other is not.

**TRAINING-SET MATCHING, the rest-side counterpart of the task arm's `5rm` (Priya asked for it
2026-09-16).** The frozen rest model trains on every pre-stroke session of the animal (~10,000
periods) against the refit's four fifths of one (~300) — a size handicap far larger than the task
arm's, and the raw gap cannot be read against it. `--match-train` refits the frozen model on a
size-matched random subset of pre-stroke **BLOCKS** (whole blocks, `grant_figures._matched_frozen`'s
rule: sampling loose periods would remove the size difference while introducing a
within-block-correlation one), seeded per scored session.

**ALL CONTROLS ON** — duration-matched + lick-gap-matched + training-set-matched, `blockperm` null,
96 sessions, 0 skipped:

| epoch | frozen | matched frozen | refit | gap | **gapM** | frozen ret | refit ret |
|---|---|---|---|---|---|---|---|
| pre | 0.402 | 0.251 | 0.286 | −0.116 | **+0.036** | 1.000 | 1.000 |
| acute | 0.259 | 0.195 | 0.258 | −0.001 | **+0.063** | 0.332 | 0.665 |
| subacute | 0.239 | 0.189 | 0.236 | −0.003 | **+0.047** | 0.306 | 0.575 |
| chronic | 0.311 | 0.225 | 0.308 | −0.003 | **+0.083** | 0.611 | 1.181 |

**MATCHING FLIPS THE PRE GAP, exactly as it does on the task side**: −0.116 → **+0.036**. Unmatched,
the frozen model wins on data volume; at equal volume the refit wins, because a within-session fit
shares that session's own nuisance structure while the matched frozen model must generalise across
days. **So the rest arm's no-lesion baseline is BRACKETED, not known** — the same conclusion `5rm`
reached for the task arm (−0.073 unmatched / +0.090 matched), and the reason both families are
drawn there and both numbers are reported here.

**READ `gapM` MINUS ITS PRE VALUE, which is the recoverable component with the handicap removed:**
acute **+0.027**, subacute **+0.011**, chronic **+0.047**. It is LARGEST AT CHRONIC — the refit arm
finds position the frozen model cannot, and does so most at the epoch where the frozen arm has
partly recovered. That is the replacement signature measured without the size confound, and it
agrees with the refit retained fraction exceeding 1.0 at chronic (1.181).

**THE MATCHED FROZEN ARM IS A NOISIER ESTIMATOR and its own retained fraction should not be quoted
as the headline**: matched accuracy is 0.251 pre against the full model's 0.402, so matchRet
(1.000 → 0.167 → 0.265 → 0.695) carries far more sampling variance than frozen retained. Matching
exists to make the GAP readable, not to replace the frozen arm.

**AND IT STILL DOES NOT DISCRIMINATE THE TWO ACCOUNTS.** A period far from a detected lick is also
far in TIME from the trial's motor events, so a decaying persistent trace predicts the same
profile as undetected continuation licking. **Only tongue tracking from the behaviour cameras (DLC,
PARKED) separates them** — say so wherever this control is quoted.
### STATE DECODER
**NOT re-pulled here.** Its numbers live in `docs/BEHAVIOURAL_STATE_CONTROL.md` (frozen pre-stroke
decoders as fraction of above-chance retained: position 0.86 → 0.43, behavioural state 0.97 → 0.88,
running 0.99 → 0.96; re-measured on the REST definition 2026-09-13). Figures `epoch_12b*`,
`epoch_13*`, deck section I. **That document is the authority; do not re-derive from the deck.**

**STILL NOT RE-PULLED:** encoder R^2/FEVE, RSA/crossnobis (105 `crossnob*` CSVs exist in
`grant_figures/epoch/`), and the `epoch_10_best_match_acc_*` family. All have `.csv` + `_sessions.csv`
+ `_meta.csv` companions, so this is a bounded pull, not a re-run.

**Pending work that will move numbers again:** the docked FROZEN decoder arm (the current rest
decode is PER-SESSION and cannot distinguish chronic recovery from chronic replacement), and `15s`.
See `docs/STATUS_2026-09-16.md`.

---

## STATE OF THE EVIDENCE, 2026-09-10 -- SUPERSEDED BY THE BLOCK ABOVE

Two structural things changed today and both invalidate numbers in this document.

**1. THE EPOCH BOUNDARIES MOVED.** The chronic FLAT test is now total drift across the window
rather than a per-session slope, and the "recovered to X% of baseline" bars are retired. Chronic is
now **three animals, not one**:

| animal | chronic from | note |
|---|---|---|
| PS92 | day 11 | unchanged |
| PS93 | **day 11** | new; was blocked by licks alone, by 0.001 of tolerance |
| PS94 | none | hit rate still improving -- day 22 -> 25 jumps 0.732 -> 1.000 |
| PS95 | **day 15** | MOVED LATER from 11, which was promoted before PS95's day-25 session existed |

Sessions moved between the subacute and chronic panels in both directions, so **every subacute and
chronic number in this document is stale**. Acute is untouched. Re-render before quoting either.

The `n = 1` caveat that used to head the caveat list is retired: it is now n = 3.

**2. TWO MEASUREMENT ERRORS WERE FOUND AND FIXED**, both of which inflated a headline:

* The best-match baseline was one-hotted from the AVERAGED leave-one-session-out matrix while every
  post-stroke epoch averages PER-SESSION one-hots. `argmax(mean) != mean(argmax)`, and the averaged
  form is a perfect identity for every animal, so the pre-stroke baseline was 1.00 by construction.
  Far-contra's fall is **0.92, not 0.94**.
* The encoder's acute "half amplitude, half shape" split is NOT SUPPORTED -- see the encoder
  section below. Explained variance after one best rescale is 0.134 acutely, and
  `_enc_terms`' own rule says that when that number is low the split is meaningless, because a code
  that is simply gone also "recovers" a lot under rescaling.

## The paragraph

> Widefield calcium imaging of dorsal cortex during a six-position mobile-spout licking task,
> decomposed into localised components by LocaNMF, shows that unilateral left ventrolateral striatal
> (VLS) stroke degrades the cortical representation of reach target by collapsing its spatial
> specificity rather than abolishing it. An L2-regularised multinomial logistic-regression decoder
> (C = 0.5) trained on z-scored component time courses and frozen on each animal's pre-stroke
> sessions falls from 0.89 to 0.52 accuracy acutely (chance 0.167; N = 4 animals, 7,354 acute vs
> 21,017 pre-stroke trials), with a spatial gradient that tracks the behavioural deficit: -0.24 at
> the near-ipsilateral spout versus -0.58 at the far-contralateral spout. Critically, the resulting
> errors are structured rather than random -- far-contralateral trials are preferentially
> misclassified as OTHER FAR TARGETS (far-ipsilateral, far-middle) rather than as near targets, so
> the acute deficit is a loss of discriminability within the far-target subspace rather than a
> uniform degradation. Three measures indicate that the affected code is PARTLY relocated rather than
> simply lost (a within-session refit recovers a third of the far-contralateral deficit and under
> an eighth of the far-ipsilateral one -- see the frozen-vs-refit section below before quoting
> "relocated" unqualified).
> Within-session split-half pattern reliability is preserved at the most impaired position (+0.12 at
> far-contralateral, interval excluding zero), so the post-stroke representation remains as
> internally repeatable as the pre-stroke one even as the frozen decoder fails on it; a similarity
> drop that reliability cannot explain is a moved code, not a noisier one. Cross-validated crossnobis
> distance between each position's post-stroke pattern and its own pre-stroke pattern -- normalised
> so 1.0 equals the separation between two DIFFERENT pre-stroke positions -- rises acutely to +1.02
> at far-contralateral, and the fraction of sessions in which that position's pattern still
> best-matches its own pre-stroke pattern falls by 0.92. A ridge encoder mapping position to
> component activity loses 0.94 of its pre-stroke explained variance acutely, roughly half
> attributable to response gain and half to pattern shape, indicating that the change is not simple
> amplitude scaling. The deficit is also non-uniform across the trial: pre-cue (ENL) decoding falls
> far less than post-cue decoding and does so evenly across positions (-0.11 to -0.26 versus -0.24 to
> -0.58), and on post-stroke trials with no detected lick the pre-cue code survives above chance
> while the post-cue code does not. Decoding recovers substantially by the subacute period (0.77) and
> approaches baseline chronically (0.87), yet the geometric displacement persists. Together these
> data support a model in which VLS stroke spares target selection while disrupting the
> transformation from selected target to executed movement, and in which behavioural recovery
> proceeds on a reorganised rather than a restored cortical code.

## HOW the code changed -- the mechanistic claim and its evidence

Measured post-cue, lick + miss-while-working, 2026-09-09. **Exact values, not heatmap readings.**

### Crossnobis own-position diagonal, RAW vs ROW-CENTRED

Row-centring subtracts each row's own mean. It matters because
`d(post P, pre Q) = |mu_postP|^2 - 2 mu_postP . mu_preQ + |mu_preQ|^2` has a first term depending
ONLY on P: a change in the magnitude of P's post-stroke response shifts its distance to every
pre-stroke position equally. Raw therefore mixes "moved" with "got bigger/smaller"; row-centred
isolates the within-row contrast, which is where a substitution lives.

| position | ROW-CENTRED acute-pre | ROW-CENTRED chronic-pre | RAW acute-pre | RAW chronic-pre |
|---|---|---|---|---|
| near ipsi   | +0.463 | **-0.032** | +0.211 | **+0.862** |
| near middle | +0.188 | +0.107 | +0.008 | +0.400 |
| near contra | +0.455 | **-0.063** | +0.245 | **+0.776** |
| far ipsi    | +0.421 | +0.061 | +0.304 | +0.220 |
| far middle  | +0.272 | **+0.010** | +0.319 | **+0.845** |
| **far contra** | **+0.665** | **+0.177** | **+1.017** | +0.264 |

Three readings, in order of confidence:

1. **ACUTELY THE PATTERN GENUINELY MOVES, most at the impaired position.** Row-centred far-contra
   +0.665 against +0.188 to +0.463 elsewhere; raw far-contra +1.017, i.e. as far from its own
   baseline as two DIFFERENT pre-stroke positions are from each other.
2. **CHRONICALLY THE PATTERN RESOLVES EVERYWHERE EXCEPT THE IMPAIRED POSITION.** Row-centred
   returns to ~0 at near-ipsi (-0.032), near-contra (-0.063) and far-middle (+0.010), but far-contra
   remains +0.177 (95% interval excludes zero on `epoch_8rcdiagdelta_...`).
3. **THE LARGE CHRONIC RAW VALUES ARE AMPLITUDE, NOT REORGANISATION.** Raw is most elevated
   chronically at SPARED positions -- near-ipsi +0.862, far-middle +0.845, near-contra +0.776 --
   exactly where row-centred is ~0. A global gain change moves every row; only row-centring
   separates it. This is also why the deck's old claim that crossnobis is "immune to uniform
   amplitude change" was wrong and has been corrected.

### GAIN vs MOVE: the two are separated in TIME, not mixed

Measured 2026-09-09 from `_matrices_crossnobis` / `_matrices_crossnobis_rowcentred`, post-cue,
lick + miss-while-working, pooled as a mean over sessions.

Row-centring is an EXACT decomposition, not an approximation. Because
`rowcentred[i,j] = raw[i,j] - mean_j raw[i,:]`, the own-position distance splits into two terms
that sum back to it:

    raw_diag[i]  =  rowmean[i]         +  rc_diag[i]
                    ^GAIN               ^MOVE
                    position-NONspecific  position-SPECIFIC
                    (shifts the whole row) (which column it moved toward)

Verified numerically: max |raw_diag - rowmean - rc_diag| = 5.6e-16 in every epoch.

Change from pre-stroke in each term:

| position | acute ΔGAIN | acute ΔMOVE | subacute ΔGAIN | subacute ΔMOVE | chronic ΔGAIN | chronic ΔMOVE |
|---|---|---|---|---|---|---|
| near ipsi   | **-0.252** | +0.463 | +0.540 | +0.056 | **+0.894** | **-0.032** |
| near middle | -0.180 | +0.188 | +0.239 | +0.127 | +0.293 | +0.107 |
| near contra | **-0.211** | +0.455 | +0.334 | +0.098 | **+0.839** | **-0.063** |
| far ipsi    | -0.117 | +0.421 | +0.112 | +0.163 | +0.159 | +0.061 |
| far middle  | +0.047 | +0.272 | +0.309 | +0.099 | **+0.835** | **+0.010** |
| **far contra** | +0.352 | **+0.665** | +0.117 | **+0.296** | +0.087 | **+0.177** |

**ACUTE IS A MOVE, NOT A GAIN CHANGE.** ΔMOVE is positive at all six positions (+0.19 to +0.67)
while ΔGAIN is NEGATIVE at four of six. Whatever the acute lesion does, it is not turning the
response volume down uniformly -- if anything the non-specific term shrinks.

**CHRONIC IS A GAIN CHANGE, NOT A MOVE.** The sign flips: ΔGAIN reaches +0.89 / +0.84 / +0.84 at
near-ipsi, near-contra and far-middle while ΔMOVE at those same positions is -0.03 / -0.06 / +0.01.
This is the whole reason the RAW chronic diagonal looks alarming at SPARED positions -- it is
reading a global amplitude change as though it were reorganisation.

**SUBACUTE IS THE CROSSOVER**, with both terms present and neither dominant.

**FAR-CONTRA IS THE EXCEPTION IN BOTH DIRECTIONS.** It is the only position whose ΔGAIN stays small
throughout (+0.35 / +0.12 / +0.09) and the only one whose ΔMOVE never returns to zero
(+0.665 -> +0.296 -> +0.177). The impaired target's deficit is position-specific at every epoch;
the spared positions' chronic change is not position-specific at all.

Equivalently, in terms of the OWN-POSITION ADVANTAGE (`rc_diag`, negative = closer to its own
pre-stroke pattern than to the average pre-stroke pattern), far-contra runs
pre **-0.831** -> acute **-0.166** -> subacute **-0.535** -> chronic **-0.654**: acutely the
advantage is all but abolished, and it never fully returns.

### Direction of the acute move

`epoch_8rc_matrices_crossnobis_rowcentred_cue_working`, acute-minus-pre panel. Exact far-contra row,
acute minus pre (negative = moved TOWARD that pre-stroke position):

| toward | nI | nM | nC | fI | fM | fC (own) |
|---|---|---|---|---|---|---|
| acute    | **-0.410** | -0.062 | +0.350 | **-0.466** | -0.076 | **+0.665** |
| subacute | -0.052 | -0.039 | +0.088 | -0.215 | -0.078 | +0.296 |
| chronic  | -0.157 | -0.040 | -0.038 | +0.052 | +0.006 | +0.177 |

The two columns it moves toward acutely are both IPSILESIONAL-side targets (far-ipsi -0.466,
near-ipsi -0.410); it moves AWAY from near-contra (+0.350) and from its own pre-stroke pattern
(+0.665). By chronic only the near-ipsi pull survives (-0.157). Consistent with the decoder
confusions, which send far-contra's lost recall to far-ipsi and far-middle
(`epoch_5c_frozen_confusion_cue_working`, acute-pre panel), and with best-match fraction falling
0.94 (`epoch_10bdelta_best_match_by_position_cue_working`).

### What this does NOT establish

The off-diagonal values above are exact point estimates, but they carry NO per-cell interval. The
claim "toward ipsilateral" is supported by three measures pointing the same way (row-centred
off-diagonal, decoder confusions, best-match fraction) and by no significance test on any
off-diagonal cell. A per-cell interval on the row-centred off-diagonal would settle it.

Epoch coverage for the tables above is NOT balanced across animals: acute = PS94 6 / PS92 5 /
PS93 4 / PS95 1; subacute = PS95 10 / PS93 7 / PS94 5 / PS92 2; **chronic = PS92 4, one animal**.
Pooling is a mean over sessions, so subacute leans on PS95 and chronic is a single-animal claim.
**SUPERSEDED for the 5r family as of 2026-09-11: chronic is now 12 sessions across PS92/PS93/PS95 —
see the 2026-09-12 section below. This line still describes the crossnobis tables above it.**

## DOES RECOVERY RESTORE THE PRE-STROKE CODE? The chronic frozen-vs-refit result, 2026-09-12

Priya's hypothesis, stated for a grant: *"recovery proceeds by re-establishing pre-stroke activity
patterns rather than by building new ones. These accounts are separable: a decoder trained on
pre-stroke activity recovers only if the original patterns return, whereas a decoder re-trained after
stroke recovers either way."*

**The framing is right and the data support it for five of six positions.** The discriminating
evidence is the SECOND clause, not the first, and it is stronger than the first.

> **SOURCE AND PRECISION.** The numbers here are read off the rendered figures of 2026-09-11
> (`grant_figures/epoch/`), which are the current session set; the session and animal counts are
> exact (they are printed in each subtitle). **Bar values are figure-read and good to about ±0.01.**
> For more decimals run `scripts/prelim_numbers_frozen_vs_refit.py`. Everything in the 2026-09-09
> section below it is the OLD session set for subacute and chronic — acute is unaffected.
>
> **WHY A FIGURE HAS TO BE RE-READ AT ALL — an open gap.** The epoch family writes PNG and SVG and
> NOTHING ELSE: no CSV, no JSON, no run log carrying the plotted values. The only machine-readable
> path to these numbers is re-running `scripts/prelim_numbers_frozen_vs_refit.py`, whose own
> docstring says the per-position ratio "is computed here and nowhere in the package". That is the
> mechanism by which this document drifted from its own figures: the render moved on 2026-09-11, the
> text did not, and nothing could have flagged the disagreement. **A per-figure sidecar of the
> plotted values (position x epoch, point estimate + both intervals) would close it** and make every
> table here checkable against the render instead of against someone's reading of a bar.

Epochs: n=90 sessions, pre 44, acute 16, subacute 18, **chronic 12 (PS92 4, PS93 4, PS95 4)**.
PS94 never qualifies as chronic. The "chronic is one animal" caveat is retired.

### 1. The frozen decoder returns to baseline (`epoch_acc_by_position_cue_working`)

| position | pre | acute | subacute | chronic |
|---|---|---|---|---|
| near ipsi | 0.95 | 0.70 | 0.88 | 0.96 |
| near middle | 0.85 | 0.58 | 0.70 | 0.71 |
| near contra | 0.92 | 0.61 | 0.87 | 0.93 |
| far ipsi | 0.87 | 0.49 | 0.71 | 0.80 |
| far middle | 0.85 | 0.44 | 0.70 | 0.81 |
| **far contra** | 0.89 | **0.32** | 0.63 | **0.78** |

Pre-subtracted (`epoch_accdelta_by_position_cue_working`), the chronic−pre interval **crosses zero at
five of six positions**. Far-contra is −0.12, excluding zero at uncorrected 95% but not under the
Bonferroni-over-18 line the figure also draws.

This is the NECESSARY condition. On its own it is not sufficient: a frozen decoder could in principle
recover because the readout axes happen to survive a reorganisation.

### 2. The refit advantage DISAPPEARS by chronic — the test that separates the accounts

Acutely, refitting buys real accuracy: far-contra **+0.19** unmatched / **+0.16** matched, near-contra
+0.17 / +0.13. Information is present and misread — that is the "displaced" result.

Chronically that advantage is gone, in BOTH families (`epoch_5rgapdelta_*`, `epoch_5rmgapdelta_*`),
as change from the pre gap:

| | near I | near M | near C | far I | far M | far C |
|---|---|---|---|---|---|---|
| unmatched gap Δ | −0.01 | **+0.22** | +0.03 | +0.02 | +0.05 | +0.08 |
| matched gap Δ | 0.00 | **+0.27** | −0.03 | −0.03 | +0.04 | +0.07 |

Only near-middle excludes zero, and only at uncorrected 95%.

**Under the "new patterns" account a within-session refit should have KEPT or GROWN its advantage** —
a reorganised code is legible to a refitted model and invisible to a frozen one. It does not.

**THAT THE TWO FAMILIES AGREE IS THE LOAD-BEARING FACT.** Their baselines are handicapped in opposite
directions (pre gap −0.073 unmatched, where frozen has ten sessions to refit's one; +0.090 matched,
where frozen must generalise across days) and they still give the same chronic answer. The result is
therefore not an artefact of either training-set convention — which is exactly why both were built.

### 3. The exception: near-middle, a SPARED position

Near-middle is the one place both signs point to reorganisation: the frozen decoder is still −0.14
below pre at chronic, AND the refit gap is the largest in the figure (+0.22 / +0.27). There is
decodable position information at chronic that the pre-stroke readout cannot reach.

**State this rather than averaging it away.** It is the single positive instance of the alternative
account in the data set, and it is not at the impaired position.

### What this does NOT license

1. **"Recovery PROCEEDS by…" is a claim about the route; this is an endpoint comparison.** Three
   post-stroke epochs. Chronic resembling pre-stroke does not establish how it got there.
2. **The acute state is not purely displaced.** Refitting recovers a third of far-contra's acute
   deficit and 9–11% at far-ipsi/far-middle. Most of the acute loss is information the population no
   longer carries linearly. That does not contradict a claim about recovery, but "the patterns
   returned" must not be read backwards as "the patterns were only hidden".
3. **Geometry has not fully returned.** Row-centred crossnobis displacement at far-contra is still
   +0.177 chronically while decoding comes back. "The frozen readout works again" is strictly weaker
   than "the representation is identical".
4. **Three animals, and recovery is animal-specific** elsewhere in this document.

### The defensible sentence

> Recovery restores a position representation that the **pre-stroke readout can still read**: a
> decoder frozen on pre-stroke activity returns to baseline at five of six positions, and the
> advantage a within-session refit held acutely is gone by chronic under both matched and unmatched
> training sets. One spared position (near-middle) retains a refit advantage, indicating locally
> reorganised coding.

That is the operational form of the hypothesis and it is what a frozen decoder can establish. The
original wording claims more than the design carries on two counts — it is an endpoint result, and
"rather than building new ones" is contradicted at near-middle.

## LOST or MISREAD? The frozen-vs-refit arm (figures 5r), 2026-09-09

> **SESSION-SET CAVEAT, added 2026-09-09 late.** Every table below, and the crossnobis tables above,
> were computed on the 0606-0907 session set. `PS92_0908` (day 22, CHRONIC) and `PS93_0908` (day 22,
> SUBACUTE) were registered by the poller and analysed afterwards, and they are now in the pooled
> bundle. The ACUTE numbers are unaffected -- neither session is acute -- but subacute and chronic
> have shifted in the third decimal (post-cue far-contra recovery subacute +0.130 -> +0.136, pooled
> subacute refit 0.764 -> 0.767). Regenerate with `scripts/prelim_numbers_frozen_vs_refit.py` and
> `scripts/prelim_numbers_crossnobis.py` before quoting a subacute or chronic value, and re-render
> the epoch figures so the deck agrees with the text.


The "relocated rather than lost" claim above rested on three INDIRECT measures -- split-half
reliability, crossnobis displacement, best-match fraction. `epoch_5rgap_frozen_vs_refit_*` tests it
directly: refit a decoder WITHIN each session on the SAME trials, same estimator, same block
grouping, and ask whether the position becomes decodable again. Post-cue, lick + miss-while-working.

    gap ~ 0, both arms low  ->  the code is degraded; no model recovers it
    gap > 0                 ->  the code is present and DISPLACED; only the frozen readout fails

**THE PRE PANEL IS NOT ZERO AND IS NOT AN EFFECT.** The frozen arm trains on ten pre-stroke sessions
and the refit arm on one, so refitting COSTS 0.073 accuracy at baseline (frozen 0.886 vs refit
0.813). Every number below is the gap at that epoch MINUS the pre gap.

| epoch | frozen | refit | gap | gap - pre gap |
|---|---|---|---|---|
| pre | 0.886 | 0.813 | -0.073 | -- |
| acute | 0.525 | 0.552 | **+0.027** | **+0.100** |
| subacute | 0.772 | 0.764 | -0.008 | +0.065 |
| chronic | 0.846 | 0.841 | -0.005 | +0.068 |

The sign of the gap FLIPS acutely: post-stroke, refitting stops costing accuracy and starts buying
it. Per position, acute (gap minus pre gap), against each position's own frozen deficit:

| position | frozen deficit (acute - pre) | recovered by refitting | fraction recovered |
|---|---|---|---|
| near ipsi   | -0.234 | +0.027 | 12% |
| near middle | -0.273 | +0.132 | 48% |
| near contra | -0.313 | +0.168 | 54% |
| far ipsi    | -0.373 | +0.034 | **9%** |
| far middle  | -0.408 | +0.044 | **11%** |
| **far contra** | **-0.573** | **+0.196** | 34% |

**THIS QUALIFIES THE "MOVED NOT LOST" CLAIM RATHER THAN CONFIRMING IT.** Three readings:

1. **The displaced component is REAL and largest in absolute terms at the impaired position**
   (+0.196 at far-contra, the largest of the six), which is where crossnobis displacement is also
   largest. Two independent methods agree on where the code moved.
2. **It is a MINORITY of the deficit.** A third of far-contra's acute drop is recovered by
   refitting; two thirds is not. The paragraph's "relocated rather than lost" overstates it --
   the defensible claim is "partly relocated, mostly lost, and the relocated part is
   position-specific."
3. **The far ipsilateral and far middle positions lose information OUTRIGHT** -- 9% and 11%
   recovered, the two smallest fractions. Their deficits are not a readout problem at all. So the
   acute lesion does two different things at once, and which one dominates depends on the position.

The gap remains positive subacutely (+0.065) and chronically (+0.068) even as frozen accuracy
returns to 0.846, so a residual readout mismatch outlives the behavioural recovery -- consistent
with the chronic row-centred far-contra displacement of +0.177.

### The cell that is NOT drawn, and why

Acute far-contralateral is GATED OUT of the lick-aligned arm. That arm conditions on a detected
lick, and acutely far-contra is the spout the animal does not lick: every acute session holds it at
0.0-4.3% of trials against a pre-stroke 16.5%. A within-session refit will not predict a class at
4% prior in a six-way problem, so the cell read -0.42 -- the mouse not licking, presented as the
code being gone, in the direction that would have flattered the "lost" reading. Gated by
`grant_figures.MIN_REFIT_SHARE` (a third of uniform); the gate fires on 78 of 37,562 trials, 0.21%,
all of them that one cell, and never in the cue or pre-cue arms. A count-only floor did NOT catch it
and made it worse -- see the constant's own note.

## THE POSITIONS STOP BEING DIFFERENT FROM EACH OTHER (split-half off-diagonal), 2026-09-10

Priya, reading the deck: "after stroke, split-half similarity suggests significantly more similarity
between R/L/center spout position brain activity compared to pre-stroke, when these were more
different than each other." That reading is correct, and it is the most direct statement of the
effect in the whole figure set.

`_split_half_matrix` splits each position's trials into halves WITHIN one session. The diagonal is
that position's own reliability -- corr(half A at P, half B at P) -- and the OFF-diagonal is how
similar two DIFFERENT positions look, measured on independent halves so no cell is a mean correlated
with itself. Post-cue, lick + miss-while-working:

| epoch | own-position (diagonal) | between-position (off-diagonal) | separation |
|---|---|---|---|
| pre | 0.791 | **-0.169** | 0.959 |
| **acute** | 0.724 | **+0.091** | **0.632** |
| subacute | 0.785 | -0.133 | 0.918 |
| chronic | 0.858 | -0.180 | 1.038 |

**EACH POSITION STAYS ABOUT AS REPEATABLE AS BEFORE WHILE THE POSITIONS STOP BEING DISTINGUISHABLE
FROM EACH OTHER.** The diagonal moves 0.791 -> 0.724, a 0.067 drop. The off-diagonal moves -0.169 ->
+0.091, a swing of +0.260 -- four times larger, and in the direction that says different targets now
evoke the same pattern. A code that had gone NOISY would show the opposite: diagonal collapsing,
off-diagonal unchanged.

The far-contralateral row is where it happens. Its correlation with the other far positions, acute
minus pre: far-ipsi -0.24 -> **+0.58**, far-middle +0.15 -> **+0.63**. The impaired target's pattern
does not become noise; it becomes the OTHER far positions' pattern.

**FOUR MEASURES, ONE EVENT -- and "independent" needs qualifying.** The split-half off-diagonal
(patterns merge), the frozen decoder's confusions (far-contra misread as far-ipsi/far-middle), the
row-centred crossnobis (far-contra moves toward ipsilesional targets) and the best-match fraction
(its nearest pre-stroke neighbour stops being itself) are four DIFFERENT quantities and they agree.
They are analytically independent -- different metric, different reference frame -- but NOT
statistically independent: all four are computed from the same per-trial joint-LocaNMF features, the
same engagement gate and the same trial sets, so a fault in those would move all four together. The
convergence rules out four different analysis choices, not one bad feature matrix.

It reverses by subacute: off-diagonal back to -0.133, and to -0.180 chronically, slightly BELOW its
own pre-stroke value.

### Split-half off-diagonal vs row-centred crossnobis: related, not redundant

Priya, 2026-09-10: "are these essentially the same?" No -- they agree on the acute event and
disagree in ways that identify what each one measures.

| epoch | Spearman rho, 30 off-diagonal cells | mean delta, split-half | mean delta, row-centred crossnobis |
|---|---|---|---|
| acute | **+0.711** | **+0.260** | **+0.082** |
| subacute | +0.472 | +0.036 | +0.028 |
| chronic | -0.115 | -0.012 | +0.009 |

Three differences, and each one is the reason to keep both:

1. **REFERENCE FRAME.** Split-half lives entirely inside ONE session: both halves come from the same
   session, so it asks "are these positions distinguishable right now". The crossnobis matrices are
   POST x PRE, so every cell carries the pre-stroke geometry and asks "which position did this one
   move TOWARD". A code that scrambled into a brand-new configuration would show up in the first and
   not the second.
2. **WHAT ROW-CENTRING REMOVES.** Row-centring subtracts each row's mean, which removes anything
   that shifted the whole row -- including the amplitude term. Split-half has no such term to
   remove.

   **CORRECTION, 2026-09-10.** An earlier version of this section said the +0.260 vs +0.082
   difference meant "roughly two thirds of the merging is a COMMON shift and one third is the
   differential substitution". That was wrong and should not be quoted. The two numbers are a
   CORRELATION change and a NORMALISED DISTANCE change: different units, no reason for their
   magnitudes to be commensurate, and no ratio to take between them. Priya caught it by asking the
   right question -- split-half is a Pearson correlation, so it is already fully blind to gain,
   global or per position, and cannot contain an amplitude component for row-centring to be
   removing. The defensible comparison between the two measures is their RANK agreement, not their
   magnitudes.

   **AND ROW-CENTRED CROSSNOBIS IS NOT FULLY GAIN-BLIND EITHER.** Writing the row out,

       d(P,Q) - mean_Q d(P,.) = -2 mu_postP . (mu_preQ - mean mu_pre) + (|mu_preQ|^2 - mean |mu_pre|^2)

   the first term scales LINEARLY with |mu_postP| and the second does not depend on P at all. So
   row-centring removes gain from the row's OFFSET but leaves it multiplying the row's SHAPE: double
   P's response and the whole row-centred profile doubles. Row-centred values are therefore
   comparable in SIGN and in RANK across epochs but not in MAGNITUDE when amplitude has changed --
   and the encoder says it changed a lot (fitted gain a: 0.94 pre to 0.36 acute, post-cue). Dividing
   each row by its own SD across columns would make it scale-free as well; that is not currently
   done.
3. **METRIC.** Pearson correlation is gain-blind by construction; crossnobis is a noise-normalised
   distance and is gain-SENSITIVE, which is the whole reason the row-centred family had to be built.

They dissociate exactly where that matters. Far-contra against NEAR-contra, acute minus pre:
split-half **+0.152** (slightly more alike) but row-centred crossnobis **-0.350** (moved AWAY).
Same trials, opposite sign, because one is absolute similarity within the session and the other is
similarity relative to that row's own mean.

Where they DO agree they agree completely: the far-contra row's five off-diagonal columns come out in
IDENTICAL rank order under both measures (rho = +1.000) -- far-ipsi > near-ipsi > far-middle >
near-middle > near-contra. That is the substitution claim, made twice from different arithmetic.



### Why a gain change cannot explain the similarity drop

`_corr_matrix` uses `np.corrcoef`, i.e. Pearson, which centres and scales each pattern vector. A
uniform gain change on all components leaves *r* exactly unchanged -- so a drop in mean-pattern
similarity is a change in the SHAPE of the pattern across components, never its size. The inverse is
the one to watch: a LINEAR DECODER is sensitive to gain (fixed hyperplane, fixed intercepts), so
amplitude can break decoding while correlation holds. Anyone reasoning "the components still decode,
so the pattern must be intact" has it backwards.

Nor is it attenuation. Split-half reliability falls only 0.791 -> 0.724 acutely, so the most
attenuation can account for is a factor sqrt(0.724 / 0.791) = 0.957: it would take the mean-pattern
similarity from 0.741 to 0.709, not to the observed **0.350**. That is also what the disattenuated
third panel of `grant_7b_reliability_*` shows cell by cell.

## THE ENCODER, AND WHY THE "HALF AMPLITUDE, HALF SHAPE" SPLIT IS WITHDRAWN

Measured 2026-09-10. The ridge encoder maps position -> component activity, frozen on pre-stroke.
For a post-stroke session with measured mean pattern `m` and frozen prediction `p`:

    frozen EV        = 1 - sum|m - p|^2 / sum|m|^2      what it actually achieves
    amplitude factor = sum m.p / sum p.p                 the single best rescaling, a
    EV after rescale = 1 - sum|m - a p|^2 / sum|m|^2     the best it could do if amplitude were free

Post-cue:

| epoch | frozen EV | EV after rescale | rescaling buys | amplitude factor a |
|---|---|---|---|---|
| pre | 0.557 | 0.580 | 0.023 | 0.943 |
| **acute** | **-0.388** | **0.134** | **0.521** | **0.361** |
| subacute | 0.215 | 0.333 | 0.118 | 0.707 |
| chronic | 0.424 | 0.447 | 0.023 | 1.015 |

**READ `EV after rescale` FIRST.** The acute gap of 0.521 looks like an amplitude story and is not:
a code that is simply GONE also recovers a lot under rescaling, because the best scale collapses
toward zero and predicting nothing beats predicting an unrelated pattern. `_enc_terms`' docstring
states the rule -- "an amplitude change only when `gain` itself is high" -- and acutely it is 0.134.
So the acute encoder failure is dominated by TUNING, and the paragraph's "roughly half attributable
to response gain and half to pattern shape" overstates the amplitude half. Withdrawn.

**AMPLITUDE FELL, IT DID NOT RISE.** `a` goes 0.943 -> 0.361 acutely: the position-DIFFERENTIAL
response (patterns are centred across positions first, so this is tuning depth, not overall
brightness) collapsed to about a third. This also explains the uniformly red ROWS in the raw
crossnobis epoch-minus-pre matrices, which are largest in the pre-cue and lick arms at far-middle
and far-contra: for a response that shrinks while keeping its shape, `d = (1-a)^2 |mu|^2`, so
SHRINKING pushes a row away from the template exactly as growing would. A red row is "far from the
pre-stroke template in a position-nonspecific way", not "bigger".

**THE WORD "GAIN" MEANT THREE THINGS** in this family until 2026-09-10 -- the EV after rescaling,
the amplitude factor, and "the thing removed" -- and the figures have been relabelled accordingly.
Quote "EV after rescale" and "amplitude factor a"; do not write "gain".

## DISPLACED vs DEGRADED, per position -- the encoder ceiling (figures 11c / 11cpos / 11cfrac)

Added 2026-09-10. This is the finding that reorganises the rest of the document, and it comes from
giving the frozen encoder something to be read against.

**THE PROBLEM IT FIXES.** `frozen EV` is an R^2 and acutely it is -0.388 post-cue: "worse than
predicting the mean", and mute about what was ACHIEVABLE in that session. A within-session split-half
refit supplies the ceiling. A refit encoder must be cross-validated or it is 1.0 by construction --
ridge on a one-hot design reduces to the per-position mean, so a session predicting its own means
from themselves is an identity.

### What is actually being measured, and on which trials

**THE "ENCODER" IS THE PER-POSITION MEAN PATTERN.** Six vectors, one per spout position, each the
average over that position's trials of the z-scored LocaNMF component time courses in sub-bins (4
for cue/pre-cue, 8 for lick). Nothing is fitted. Earlier notes described it as "ridge on a one-hot
position design", which is what it is EQUIVALENT to and not what the code does: with a one-hot design
the least-squares solution for each column IS that position's mean, and ridge only shrinks each
toward zero by `n_q / (n_q + alpha)`. An actual `Ridge(alpha=1.0)` on a one-hot design is fitted
elsewhere -- `locanmf_cross_mouse._per_session_compute`, a different figure -- but not here.

**THE VARIANCE IS THE VARIANCE OF THE MEANS, NOT OF THE TRIALS.** `M` and `P` are both 6 x features
matrices of per-position means, centred across positions, and the denominator is `sum M^2`: the
BETWEEN-POSITION variance of the measured means. Trial-to-trial variance never enters it. So the
score answers "how much of the measured position-to-position pattern does the template's
position-to-position pattern reproduce", NOT "how much of the trial variance is explained" -- a
trial-level R^2 would be far lower, because single-trial noise is large and is excluded here by
construction.

**CENTRING ACROSS POSITIONS** is the line `M = M - M.mean(0)`. `M` is 6 positions x features;
`M.mean(0)` averages down the POSITION axis, giving one grand-mean pattern -- what the cortex does on
an average trial regardless of where the spout was. Subtracting it leaves each position as its
DEVIATION from that average, so anything common to all six is gone and only what DIFFERS between
positions survives. A session that is 20% dimmer overall, or has a different F0 or SNR or arousal
level, moves all six rows together and vanishes under this operation. The encoder is not being asked
to predict the average trial, so it is neither charged nor credited for it. The denominator
`sum M^2` is therefore the size of the position-DIFFERENTIAL signal, which is also why a session with
strong overall activity but no position tuning collapses it (guarded: `tot <= 1e-12` returns NaN).

Not to be confused with the ROW-centring in the crossnobis matrices, which subtracts each row's mean
across the reference-position COLUMNS. Same spirit -- remove what is common, keep what discriminates
-- but a different axis of a different matrix.

It also means the CEILING's shortfall from 1.0 is entirely sampling noise in the half-session means,
which is why halving the trials lowers it and why the size-matched arm was needed at all.

**ENGAGED TRIALS ONLY, AND NOT SYMMETRICALLY.** The terminal quit period is excluded everywhere
(`~not_eng`, the reference-restricted backdated gate). But the two sides are not the same trial
class:

* the PRE-STROKE reference is built from LICK trials only;
* the post-stroke set is `working` = lick PLUS miss-while-working.

That is deliberate -- a pre-stroke animal is not missing, so `working` would add almost nothing and
would make the reference a different KIND of trial from itself -- but it does mean **the template is
made of successful trials and is scored against a set that includes failures.** At far-contralateral
acutely that set is mostly failures, which is exactly the population the deficit lives in. It is the
right comparison for "does the old template describe what the animal is doing now", and it is not a
like-for-like comparison of two trial classes.

### Pooled: amplitude recovers, shape does not

| epoch | ceiling | frozen (matched) | gap | Δ vs pre | template captures | amplitude a |
|---|---|---|---|---|---|---|
| pre | 0.706 | 0.429 | 0.276 | -- | 61% | 0.749 |
| **acute** | 0.568 | 0.100 | 0.468 | **+0.192** | **18%** | **0.286** |
| subacute | 0.631 | 0.226 | 0.405 | +0.128 | 36% | 0.527 |
| chronic | 0.778 | 0.315 | 0.463 | **+0.186** | 41% | 0.745 |

The fitted amplitude returns to baseline (0.286 -> 0.745 against 0.749 pre-stroke). The shape
mismatch does not (+0.192 -> +0.128 -> +0.186, flat, and no better at chronic than acute). **The
pre-stroke gap of 0.276 is not an effect** -- it is the cost of a template coming from other sessions
at equal training-set size, the same asymmetry the matched frozen DECODER arm exposed.

`amplitude a` is the factor the MATCHED arm fits. The unmatched, as-actually-used frozen arm fits a
larger one (0.943 / 0.361 / 0.677 / 0.947); that is the number behind the `R² = 1 - (1-a)²/a² =
-2.13` statement about raw EV, and the two have been confused once.

None of the `Δ vs pre` values carries an interval -- they are differences of pooled point estimates
-- so "flat" is a reading of three numbers without error bars, not a test.

### Per position: THE DISSOCIATION

| position | pre ceiling / captured | acute ceiling / captured | **Δ ceiling** |
|---|---|---|---|
| near ipsi | 0.740 / 0.67 | 0.584 / 0.28 | -0.156 |
| near middle | 0.462 / 0.31 | 0.455 / 0.22 | -0.007 |
| near contra | 0.731 / 0.63 | 0.616 / 0.27 | -0.115 |
| **far ipsi** | 0.629 / 0.56 | 0.306 / 0.22 | **-0.323** |
| **far middle** | 0.721 / 0.63 | 0.451 / 0.42 | **-0.270** |
| **far contra** | 0.628 / 0.55 | **0.626** / **0.01** | **-0.002** |

**FAR-CONTRALATERAL LOSES NO STRUCTURE AND LOSES ITS TEMPLATE ENTIRELY.** Its ceiling acutely is
0.626 against 0.628 pre-stroke, a change of -0.002 -- its own trials predict each other exactly as
well as before the lesion -- while the pre-stroke template's capture falls 0.55 -> 0.01 and recovers
only to 0.28-0.34. The positions that lose CEILING are the flanking ones, far-ipsi (-0.323) and
far-middle (-0.270).

The captured fraction is clipped to [0, 1]. At far-contra acutely that clip hides a sign: the
underlying matched EV is **-0.060**, i.e. the pre-stroke template is further from that position's
measured pattern than predicting zero would be. Per-position scores use the session-GLOBAL scale
factor, which at a position whose amplitude has collapsed is the wrong one. Quote the fraction.

**THIS CONVERGES WITH THE DECODER ARM** from the same day: refitting recovers 34% of far-contra's
acute deficit and only **9% at far-ipsi and 11% at far-middle**. Two analyses, opposite directions of
fit, entirely different statistics, one dissociation:

> **Far-contra's code is DISPLACED. Its neighbours' codes are DEGRADED.**

Both bar families carry the standard animals-then-sessions bootstrap, per-session dots and
multiple-comparison marks; the captured fraction is computed PER SESSION and then pooled, so it has a
distribution behind it rather than being one pooled number divided by another.

**WHAT THIS CHANGES ABOVE.** The earlier pooled statement that "roughly 28% of the available
structure is lost" averages two opposite things and must not be quoted without the split. And the
"partly relocated, mostly lost" summary needs the same qualification: at the IMPAIRED position it is
not mostly lost at all -- nothing measurable is lost there. The loss is at the neighbours.

## Where each number comes from

| claim | figure | value |
|---|---|---|
| post-cue decoder Δ, acute | `epoch_accdelta_by_position_cue_working` | nI −0.24, nM −0.28, nC −0.31, fI −0.38, fM −0.41, **fC −0.58** |
| post-cue decoder Δ, subacute | same | nI −0.03, nM −0.15, nC −0.03, fI −0.13, fM −0.12, fC −0.22 |
| post-cue decoder Δ, chronic | same | nI +0.07, nM −0.01, nC +0.03, fI −0.06, fM −0.02, fC −0.13 |
| pre-cue decoder Δ, acute | `epoch_accdelta_by_position_precue_working` | nI −0.23, nM −0.11, nC −0.23, fI −0.16, fM −0.13, fC −0.26 |
| encoder Δ (variance, gain) | `epoch_11delta_encoder_gain_shape_cue_working` | acute −0.94 / −0.45; subacute −0.34 / −0.26; chronic −0.05 / −0.03 |
| within-session reliability Δ | `epoch_7diagdelta_matrices_splithalf_cue_working` | acute nI −0.12, nM −0.33, nC −0.07, fI +0.01, fM −0.03, **fC +0.12** |
| crossnobis displacement | `epoch_8diagdelta_matrices_crossnobis_cue_working` | acute fC **+1.02**; chronic nI +0.98, nC +0.86, fM +1.01 |
| best-match DESTINATION, acute fC | `epoch_10c_matrices_best_match_destination_cue_working` | fM 0.50, fI 0.25, nI 0.12, nC 0.06, itself 0.06 |
| best-match fraction fC, pre -> acute | `epoch_10cdiagdelta_...` | 0.98 -> 0.06 (a drop of **0.92**, not 0.94 -- see below) |
| recoverable, MATCHED training sets | `epoch_5rmgapdelta_frozen_vs_refit_cue_working` | pre gap **+0.090**; acute fC **+0.158**, nC +0.128, nI +0.093 |
| pre-stroke frozen decoder accuracy | run log, `frozen decoder [.]` | cue .87/.78/.93/.92; precue .47/.46/.66/.45; lick .92/.88/.95/.93 |
| no-lick dissociation | run log, `=>` verdicts | PS95 pre-cue 0.36 vs post-cue 0.09 |
| refit-minus-frozen gap | `epoch_5rgapdelta_frozen_vs_refit_cue_working` | pre -0.073; acute +0.027, subacute -0.008, chronic -0.005 |
| refit recovery by position, acute | same | nI +0.03, nM +0.13, nC +0.17, fI +0.03, fM +0.04, **fC +0.20** |

Epoch n: pre 44 sessions, acute 16, subacute 23, **chronic 3** -- as of the 2026-09-07 run, i.e.
BEFORE the 2026-09-10 boundary change and before the 0908/0910 sessions. Recount after re-rendering.

## Sign conventions that are easy to get wrong

* **`_matrices_crossnobis` returns RAW distances** normalised to pre-stroke units, so on the epoch
  figures **larger = further from baseline = more changed**.
* **`_mats_crossnobis` (figure 8d) NEGATES them** (`sign=-1`) so that "larger diagonal = more
  preserved", matching the correlation figures. The two are one letter apart in the name and carry
  opposite signs. Check which one a figure used before describing its direction.
* The matrices are **post x pre cross-matrices**, so the diagonal is position *i* post against
  position *i* pre -- not a within-session RDM.

## Caveats that must survive into any submitted version

1. **THE CHRONIC EPOCH IS NO LONGER ONE ANIMAL, as of 2026-09-10** -- and every chronic number in
   this document predates that. It was PS92 alone; under `flat_mode: drift` it is PS92 from day 11,
   PS93 from day 11 and PS95 from day 15, with PS94 still not qualifying. PS95 also MOVED LATER,
   from a day-11 boundary the behaviour box had promoted under the old per-session rate test before
   PS95's day-25 session existed.

   **CONSEQUENCE: EVERY CHRONIC AND SUBACUTE NUMBER BELOW IS STALE**, because sessions moved between
   the two panels in both directions. The acute numbers are untouched. Re-render and re-read before
   quoting anything from those two epochs. The upside is that the caveat this entry used to carry --
   "the most quotable claim is the least supported, n = 1" -- is retired: three animals now reach
   chronic.
2. **Pre-cue is not "spared".** Its baseline is only ~0.45-0.66, so −0.26 at far-contra leaves it
   near chance (0.167). The defensible claim is about the *shape* of the loss (flat across positions
   vs graded), not its absence.
3. **Reliability preservation is not uniform.** It holds at far-contra (+0.12) but near-middle
   drops −0.33 acutely. The "moved not lost" argument is strongest exactly at the impaired position,
   which is convenient -- and therefore worth stating precisely rather than generalising.
4. **The no-lick dissociation is basis-dependent.** It holds under both the Allen-ROI and joint
   LocaNMF bases for PS92 and PS95; PS93 and PS94 flip to "no clear dissociation" depending on basis
   and RT cut. The run flags `BASES DISAGREE`. Quote it as a per-animal result, not a cohort one.
5. **Several intervals cross zero** (subacute near-ipsi and near-contra decoder). The paragraph reads
   as more uniform than the intervals support.
6. **"Relocated rather than lost" is now QUALIFIED by the refit arm.** A within-session refit
   recovers about a third of the acute far-contralateral deficit and under an eighth of the
   far-ipsilateral and far-middle ones. The displacement is real, position-specific and agrees with
   the crossnobis geometry on WHERE, but most of the acute deficit is information the population no
   longer carries linearly. Do not quote "relocated" without it.
7. **Seven placed figures were stale at deck build** (`section_g_smalllesion_*`,
   `poststroke_G7d_smalllesion_*`, `coding_rtdrift`; 17-21 days old, predating both engagement-gate
   changes). None are cited above, but do not cite them until re-run.

## Engagement gate

All post-stroke numbers above use the reference-restricted, backdated gate
(`precue_engagement_states.engagement_gate`): judged only at close_L / close_center, requiring a
non-recovering collapse, backdated to the start of the run of misses that trips it. This matters
because the previous position-blind gate discarded far-position motor failures as "disengagement" --
i.e. deleted the effect as the confound. See `docs/STATUS_2026-09-07.md` and
`tests/test_one_engagement_gate.py`.

---

# A manipulation hypothesis these data motivate

Priya, 2026-09-09: "what might be a reasonable hypothesis to test how we can manipulate
post-stroke recovery? eg DREADD- or optogenetic manipulation of activity in contralateral striatum
vs ipsi/contra orofacial motor cortex in subacute post-stroke?" Recorded as a FRAMING to argue
with, not a recommendation -- the design choices are the lab's.

## The question the data poses

The impaired target's pattern moves TOWARD ipsilateral target representations
(`epoch_8rc_matrices_crossnobis_rowcentred_cue_working`, acute-minus-pre). Two readings make
OPPOSITE predictions, which is what makes it worth an experiment:

* **MALADAPTIVE CAPTURE.** The intact contralesional hemisphere captures the impaired target's
  code and the drift IMPEDES recovery. Rodent analogue of the interhemispheric-rivalry rationale
  behind contralesional low-frequency rTMS in human stroke.
* **COMPENSATORY.** The drift IS the recovery -- the intact circuit takes the target over.

## What the data already says, and how weakly

PS92 -- the only animal to reach chronic -- recovered to 378/378 hits while retaining a residual
far-contra row-centred displacement of +0.177. A displaced-but-functional code is what the
COMPENSATORY account predicts. This is n = 1 and cannot separate "the displacement is the
mechanism" from "the displacement is a harmless scar".

## Predictions by target, subacute window

| target | if MALADAPTIVE | if COMPENSATORY |
|---|---|---|
| contralesional (R) orofacial M1, inhibit | far-contra displacement falls, hit rate rises | displacement rises, recovery stalls |
| perilesional (L) orofacial M1, excite | contra-specific drive restored; displacement falls | little effect, or gain-only change |
| contralesional (R) VLS, inhibit | as R-M1 but slower onset | recovery blocked if the striatal route carries it |

## Two design points the preliminary data adds

1. **USE THE IMAGING READOUT AS THE DEPENDENT VARIABLE, NOT HIT RATE.** Behaviour saturates: PS92
   hit 378/378 while its representation was still displaced. The row-centred far-contra
   displacement is unsaturated and position-specific, so it can detect an effect hit rate cannot.
2. **PREDICT A DISSOCIATION, NOT A GLOBAL EFFECT.** The deficit is post-cue and execution-side;
   pre-cue selection is comparatively preserved (LOSO 0.510 vs 0.873 post-cue) and the plan
   survives on failed trials. A manipulation of motor-output circuits should move the POST-cue code
   and leave the PRE-cue code alone. One that moved both would argue for arousal or engagement
   rather than the transformation -- and the engagement gate and no-lick arm already exist to catch
   exactly that confound.

## Timing

Row-centred far-contra displacement runs +0.665 acute -> +0.30 subacute -> +0.177 chronic, so the
trajectory is set during subacute. That supports subacute as the intervention window.
