# REST ENGAGEMENT AUDIT — which rest analyses gate the quit period, and which do not

**2026-09-16.** Triggered by Priya: *"all only on working trials?"*, then *"the rest maps used for
the map deltas — are those engagement gated?"*, then *"please do a careful review of the analysis
pipeline and identify any other figures and analyses that need to be redone."*

---

## 1. THE DEFECT, AND ITS EXACT SHAPE

`flag_engagement`'s **terminal quit period** — the sated tail where the animal has stopped working —
is a different behavioural state. Rest is most ABUNDANT there (the animal has stopped working, so
inter-trial intervals lengthen and multiply), it sits at the END of the session where drift is
largest, and **its share grows after the lesion**. Measured over all 111 sessions:

| epoch | % of TRIALS in the quit period | % of REST FRAMES in the quit period |
|---|---|---|
| pre | 4.1% | **3.1%** |
| acute | **21.8%** | **18.7%** |
| subacute | 18.6% | 17.7% |
| chronic | 6.9% | 4.7% |

Worst individual sessions, by share of rest frames: PS94_0819 acute **60.6%**, PS95_0820 subacute
51.3%, PS93_0820 acute 47.0%, PS95_0910 chronic 37.6%.

**So the contaminant is not uniform — it tracks the independent variable**, and it is ~6x larger
acutely than pre-stroke. Any rest quantity compared ACROSS EPOCHS, or any rest baseline subtracted
from an across-epoch contrast, carries that.

**THIS IS THE ARGUMENT `restw` WAS BUILT ON, ONE AXIS OVER.** `session_restw_svt`'s docstring:
*"`rest` averages over rest FRAMES, so a position contributing more rest frames pulls the baseline
toward its own resting state … Post-stroke the animal stops attempting the far positions … so the
baseline changes WITH the deficit."* Substitute "behavioural state" for "position" and it is the
same sentence. `restw` fixed the POSITION axis on 2026-09-13; the ENGAGEMENT axis was fixed for
`restw` on 2026-09-14 and **nowhere else**.

---

## 2. THE AUDIT — every consumer of rest frames or the rest baseline

Method: enumerate every module matching `quiet_frame_path | quiet_sample.npy |
rest_frames_by_position | session_rest_svt_timelocal | session_restw_svt | _quiet_baseline_local |
rest_starts | quiet_starts`, then classify by whether the gate appears in CODE (comments stripped)
and whether the consumer makes a **position** or **across-epoch** contrast. A consumer that only
describes rest itself is NOT exposed by this defect.

### A. GATED — correct, no action

| consumer | how |
|---|---|
| `rest_by_position.rest_frames_by_position` | `engaged_only=True` by default; requires BOTH bracketing trials engaged. Added 2026-09-14. |
| `position_reference_maps.session_restw_svt` | via the above → the **`_RESTWref_`** map family is CLEAN |
| `locanmf_position_encoder._restw_baseline` | via the above → the settled **encoder baseline is CLEAN** |
| `scripts/rest_migration/restw_reliability.py` | via the above |
| `scripts/rest_migration/rest_frozen_decoder.py` (**15f**) | fixed 2026-09-16 |
| `scripts/rest_migration/shared_position_projection.py` (**15s**) | fixed 2026-09-16 |

**An earlier version of this audit claimed the map deltas were ungated. That was wrong for
`restw`** — `rest_frames_by_position` has carried `engaged_only=True` since 2026-09-14, and the
headline `_RESTWref_` family was never affected. The file-level check that produced that claim was
too coarse: `position_reference_maps` contains a gated builder AND an ungated one.

### B. UNGATED AND EXPOSED — these need re-running

| consumer | what it produces | why exposed |
|---|---|---|
| `rest_position_permutation.py` | **finding 11**, observed/null **1.622, 44/44** | position contrast |
| `rest_position_decode.py` | **93/94 sessions above null**, the per-session trajectory | position + across-epoch. **Also CLAIMS the gate** in its docstring and prints "(working trials…)" in its own header |
| `rest_position_vs_drift.py` | **`epoch_15x_REST_by_position_by_animal`** (published, on the share, in the nightly) | position contrast |
| `rest_carries_position.py` | the earlier position-in-rest measurement | position contrast |
| `rest_block_boundary.py` | persistence vs anticipation (`r_prev − r_next`) | position contrast |
| `position_reference_maps.session_rest_svt_timelocal` | the **`_RESTref_`** (time-local) baseline | across-epoch baseline |
| `flatpool_vs_restw.py` | the composition test the `restw` decision rests on | position + across-epoch |

### C. UNGATED AND CORRECTLY SO — do NOT gate these

| consumer | why gating would be WRONG |
|---|---|
| `quiet_periods.rest_mask`, `behavior_events` | mask BUILDERS. "The animal is at rest" is true regardless of engagement. Gating here would redefine the mask and force the whole `recompute_masks` + `SCHEMA_VERSION` + `CACHE_VERSION` cascade — for an *inclusion criterion*, which belongs at consumption. The trial side already does it that way (`_quit_mask` inside `_working_xy`). |
| `plot_running_activity_maps.py` | quiet-vs-running is a LOCOMOTOR question with no position or epoch contrast; quit-period quiet is legitimately quiet. **No imaging-box re-run needed.** |
| `locomotor_state.py` | rest is a **CLASS being decoded**, not a baseline. Gating would redefine the class and invalidate `BEHAVIOURAL_STATE_CONTROL.md`'s design. |
| `locanmf_position_encoder._quiet_baseline_local` | SUPERSEDED 2026-09-14; retained only for a drift DIAGNOSTIC figure whose subject is how a time-local estimate moves. |
| mask management + smoke + diagnostics | `recompute_masks`, `fix_mask_names`, `docked_rest_check`, `why_no_rest`, `restw_smoke`, `restw_bin_coverage`, `restw_column_drops`, `timelocal_needed`, `worktrunc_result_impact`, `regen_lick_maps`, `filter_acausality_test`, `preprocess` |

### D. FLAGGED, NOT YET DECIDED

1. **`_quiet_zscore` — RESOLVED 2026-09-16, no action.** It z-scores every LocaNMF trace by
   quiet-frame mean/SD, and the quit period's share of those frames moves with epoch. Measured,
   gated vs ungated, median over components then over sessions:

   | epoch | \|Δmean\|/σ | SD ratio | frames dropped (median session) |
   |---|---|---|---|
   | pre | 0.0000 | 1.0000 | 0.0% |
   | acute | **0.0333** | 1.0000 | 15.7% |
   | subacute | 0.0340 | 1.0000 | 16.5% |
   | chronic | 0.0000 | 1.0000 | 0.0% |

   **The SD does not move at all** (1.0000 everywhere), so the scaling half is untouched. The MEAN
   shifts 0.033 σ, and only in acute/subacute — so it does track the epoch. It is absorbed
   regardless: a per-component affine shift is removed by the `StandardScaler` inside every
   per-session decoder, and any baseline-subtracting analysis cancels it.

   **The frozen cross-session decoder never reads this path** — `grant_figures._pooled_bundle` goes
   through `joint_locanmf`, which carries no quiet normalisation. That was the exposure worth
   worrying about and it does not exist. `_quiet_zscore` reaches only
   `locanmf_cue_lick_analysis`, `locanmf_cue_auc` and `locanmf_lick_aligned`.

   Incidental: the median PRE and CHRONIC session has NO terminal quit period at all — those rows
   are exactly zero because the animals worked to the end.
2. **A `docked=` inconsistency**, found during this audit. `session_restw_svt` defaults
   `docked=False` and `position_reference_maps` line 486 calls it that way — correct, because the
   `restdock05` MASK is already docked and the term must not be applied twice. `15s` was passing
   `docked=True`, so it was built on a different rest window from the `15r` family it is compared
   with. **Fixed 2026-09-16**; no other caller passes `docked=True`.

---

## 3. WHAT THIS CHANGES IN RESULTS ALREADY REPORTED

**15f, measured both ways** (all four controls on, `blockperm` null). Acute retained, per animal:

| | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| ungated | 0.401 | 0.187 | 0.134 | 0.120 |
| **gated** | **0.609** | **0.277** | **0.086** | **−0.093** |

**The correction is large and NOT uniform in sign** — two animals up, two down, and PS95 falls below
its own null. The prediction that gating would uniformly raise acute (quit-period rest being
noisier) was WRONG. The acute cohort figure is no longer a clean 4/4 replication, which **weakens
the acute claim** rather than rescuing it.

### IS THE GATE ACTUALLY BITING? Verified against an independent measurement

Priya, 2026-09-16: *"is 15f correct?"* In the fully-matched 15f run the gate removed only **4.2%**
of acute periods, against a quit-period share of **18.7%** of acute rest FRAMES. Two readings with
opposite consequences: the other filters had already removed them, or the gate is mis-indexed.

Settled by calling `_collect` with `gate=True/False` and NOTHING else changed — no duration, lick-gap
or training-set matching, no fitting:

| epoch | ungated periods | gated | dropped | expected (frame share) |
|---|---|---|---|---|
| pre | 15,843 | 15,294 | **3.5%** | 3.1% |
| acute | 6,646 | 5,411 | **18.6%** | **18.7%** |
| subacute | 5,809 | 4,807 | **17.2%** | 17.7% |
| chronic | 5,884 | 5,598 | **4.9%** | 4.7% |

**Every epoch lands on its predicted share: the gate is correct.** The small effect in the matched
run is `--match-lickgap` having already excluded quit-period rest — it sits far from any detected
lick, so the pre-stroke gap window drops it before the gate sees it. **Two controls doing
overlapping work**, which is worth knowing when reading either of them as independent.

**BUT 15f'S ACUTE CELL IS FRAGILE, AND THAT IS A SEPARATE FACT FROM CORRECTNESS.** Removing that 4%
moved acute retained per animal from 0.401/0.187/0.134/0.120 to 0.609/0.277/0.086/**−0.093**. That
much movement from that little data says the estimate is underpowered, not that it is wrong: acute
is 16 sessions and **PS95 contributes ONE**, so its cell moves freely under any reweighting. Quote
15f's CHRONIC result (0.611 frozen vs 1.181 refit, stable across every control); treat acute as
underpowered.

### Finding 11 — RE-RUN, and it SURVIVES both fixes

`rest_position_permutation --docked --perm 200`, with the engagement gate AND the repaired
classifier:

| | published | gated + repaired |
|---|---|---|
| observed/null | 1.622 | **1.634** |
| sessions above null | 44/44 | **43/44** |
| mean over animals | 1.650 | **1.662** |
| PS92 / PS93 / PS94 / PS95 | 1.444 / 1.996 / 1.662 / 1.500 | **1.467 / 1.999 / 1.724 / 1.458** |

The two corrections very nearly cancel. One PS95 session drops below its own null; nothing else
moves materially. **REST CARRIES POSITION stands.**

**A 4-session smoke test of the same run gave 1.469 and was read here as "the direction is down".
It was not — the cohort went slightly UP.** Four of forty-four sessions carried no information
about the total, which is the extrapolation trap `CLAUDE.md` documents for the render timings,
recurring in a statistic. Do not read a partial cohort as a trend.

### The other two re-runs — both SURVIVE

**Block boundary (persistence vs anticipation)**, gated + repaired classifier, `--docked`:
`r_prev - r_next = +0.0772`, **4/4 animals positive**, 2,983 boundary periods, 0 sessions skipped
(PS92 +0.105, PS93 +0.089, PS94 +0.044, PS95 +0.071). The published value is **+0.0754** — the two
fixes moved it by 0.002.

**THIS IS THE ANALYSIS `15s` COULD NOT DO.** Within a block the position just licked and the one
coming next are the same, so nothing separates persistence from anticipation. At a BOUNDARY they
differ, and rest resembles **the position just licked at**. So `15s`'s finding that the PRE-CUE
window shares more with rest than post-cue does is a shared component pointing BACKWARDS — a trace
of the last target, not preparation for the next. That is a real constraint on reading the pre-cue
signal as anticipatory, and it is consistent with this project's standing refusal to call it "a
maintained motor plan".

**Per-session decode**, gated + repaired: **94 of 96 sessions above their own null** (pre 44/44,
acute 15/16, subacute 17/18, chronic 18/18), against the published 93/94.

| epoch | obs | null | obs − null |
|---|---|---|---|
| pre | 0.372 | 0.167 | 0.205 |
| acute | 0.300 | 0.173 | 0.126 |
| subacute | 0.310 | 0.171 | 0.139 |
| chronic | 0.430 | 0.168 | 0.262 |

**Do NOT read a trajectory off that column.** It is a cohort mean, and the acute-dip claim derived
from exactly this quantity was WITHDRAWN on 2026-09-15 because PS95 rises. Plot the animals first.

### `rest_carries_position` — re-run gated, and it came out STRONGER

| | pre-audit | gated + repaired |
|---|---|---|
| between/within-position RMS ratio | 1.45 | **1.97** |
| largest significant area | 1,042 / 2,022 bins | **1,468 / 2,022** |

6/6 positions in 4/4 animals; 15,458 of 19,131 rest bouts carried unambiguous labels. Verdict
unchanged: REST CARRIES POSITION INFORMATION.

**The gate RAISED this statistic, and the direction is informative.** The ratio's denominator is
WITHIN-position split-half noise; the quit period is a different behavioural state sitting where
drift is largest, so removing it cuts that denominator. A contaminant that had been INFLATING the
effect would have moved it the other way. Same direction as finding 11 holding at 1.634.

### A THIRD templating failure, caught by the linter

The patch that added the gate to `rest_carries_position` wrote
`getattr(a, "no_engagement_gate", False)` — but `a` there is the rest-bout LOOP VARIABLE, not an
argument namespace; that file has no argparse. Undefined on the first session, a stale sample index
afterwards. `ruff F821` caught it before it ran. Copying a snippet between scripts whose argument
names differ is the same mechanism that spread the missing gate and the raw classifier.

**The guard is now an ENUMERATED list.** `tests/test_rest_engagement.py::EXPOSED` names all seven
rest analyses that make a position or across-epoch contrast, and parametrised tests assert each one
builds the gate AND uses the repaired classifier, with no surviving raw `_classify_cues(` call.
Written out rather than discovered, because the failure is a script being added — or patched for one
defect and not the other — and nobody noticing. On 2026-09-16 `rest_block_boundary` and
`rest_carries_position` were described here as "patched for both" when they had only the classifier
fix; a count of call sites said otherwise.

**Every rest number produced before 2026-09-16 is provisional** until its analysis is re-run:
finding 11's 1.622, the 93/94 per-session decode, `epoch_15x`, the `_RESTref_` maps, and the
`flatpool_vs_restw` composition test. **`_RESTWref_` and the encoder baseline are unaffected.**

---

## 4. THE GUARD

`wfield_local/rest_engagement.py` is the single definition, resolving to
`precue_engagement_states.engagement_gate` — the same gate `beta_maps._quit_mask` and
`grant_figures._gate_all` already use, so the rest arm's "working" means what "working" means
everywhere else. A fourth re-derivation would agree today and diverge later;
`quiet_periods.rest_mask` records that exact failure having already happened once, when two modules
computed the rest mask independently and "a comment in the second claimed they agreed".

`tests/test_rest_engagement.py` asserts the gate appears in the **code** of each collector — applied
to BOTH bracketing trials — not merely in its docstring. That assertion is what was missing: four
analyses documented a gate none of them ran, one of them printing "(working trials…)" as it did so.

`--no-engagement-gate` preserves the old behaviour so the SIZE of each correction can be measured
rather than asserted. **A fix whose magnitude is unknown is not yet a finding.**
