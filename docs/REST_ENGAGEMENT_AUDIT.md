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

1. **`locanmf_cue_lick_analysis._quiet_zscore`** z-scores every LocaNMF trace by quiet-frame
   mean/SD, and that composition moves with epoch. A per-feature affine transform is absorbed by the
   `StandardScaler` inside each within-session decoder, so within-session results should be
   indifferent — but the **FROZEN cross-session decoder** may not be, because the normalisation is
   what makes sessions commensurable in the first place. **Measure before assuming either way.**
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
