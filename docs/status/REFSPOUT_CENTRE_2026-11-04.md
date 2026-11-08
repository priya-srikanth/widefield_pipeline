# Can the reference spouts anchor the frozen decoder's day offset? And what does an offset-free readout say? (2026-11-04)

Cohort 1 (PS92-95). A read-only diagnostic. No production default, model, deck or page was changed. Every number
uses the stored 11-03-late production models (frozen-hit). The only fits are in memory: per-session class offsets
(Iref) and a within-session refit for the upper bound (S1roi). None was stored or published.
Scripts, CSVs and logs are in `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/refspout_centre_1104/`. The
pools, frame engine and section-D reproduction are reused from `CENTRE_MIXSHIFT_2026-11-04.md` (`centre_mixshift_1104/`),
and the rest centers from `REST_CENTRE_2026-11-04.md` (`rest_centre_1104/`).

## 1. Question

The frozen spout decoder's per-session center is Wt (DECISIONS 11-04): the equal-weight mean of the per-spout means
of all working rows. Priya's point is that any center built from post-stroke data absorbs some of the stroke, and that
post-stroke activity changes at every spout, close spouts included. Two arms test this:

- **A. Reference-spout anchor (Rref).** Estimate each session's day offset from the reference spouts only
  (`trial_classes.REFERENCE_SPOUTS` = close_L and close_center, i.e. close ipsi and close mid). This works only if
  those spouts do not change, so their change is MEASURED (checks 1-3).
- **B. Intercept-only refit (Iref).** Keep the frozen weights and let each session have free class offsets, fitted
  with cross-validation across blocks. A common feature offset c changes every class score by the constant −w_k·c, so
  free intercepts absorb any per-session offset exactly. That makes Iref the readout least dependent on a center
  choice.

## 2. Design

Everything is shared with the earlier tests:

- the pool: section D's ROI pool;
- the models: full model for post sessions, leave-that-session-out model for held-out pre sessions;
- the scored rows: each session's camera working rows, cue and precue.

T(i) is the scoring model's set of training sessions (August-only pre).

| tag | frame (stored model) |
|---|---|
| W / Wt / N | as before: production bal_em; true-label balanced working center (the default); pooled pre, no session center |
| **Rref_w_a** | center = mean_q μ_iq − mean_q mean_{s∈T(i)} μ_sq + c̄6. Here μ_iq is the session's working-row mean at reference spout q (equal weight over the 2 spouts; a spout needs ≥ 5 rows) and c̄6 is the pooled pre six-spout balanced center, all in raw units. Scale = Wt's per-session SD |
| Rref_w_b | the same center; scale = SD of the session's reference-spout rows × the pooled pre ratio (working SD / reference SD, per feature) |
| Rref_c_a / _c_b | the same with contact rows only at the reference spouts |
| **Iref_Wt / Iref_N** | frame Wt or N. S = the frozen model's decision scores. GroupKFold over position blocks (n = min(5, #blocks); DAQ-lick rows use pool BE, no-lick rows use runs of position, production's `_scored_rows` rule). On the training folds, fit 6 offsets b by maximum likelihood of softmax(S + b) (ridge 1e-2, only to keep b finite); score the held-out fold by argmax(S + b) |
| Iref_*_bal | the same, with class-balanced likelihood weights (each spout's rows weigh 1/n_spout), so b carries no session class-frequency prior |
| S1roi | upper bound: production's estimator (`_pipe`) refitted within the session on the same rows and folds |

### Assumption checks

**Check 1: does the reference-spout response change?** Δ_q = (μ_iq − P_i) − mean_{s∈T(i)} (μ_sq − P_s), where P is a
day-offset proxy. A pre session is compared leave-one-out. The proxies are:

- none (raw dF/F);
- Wt (the session's six-spout balanced center);
- R5 (the session's rest center);
- **contrast**: close_L − close_center. This needs no proxy at all.

Two size measures:

- **rel** = ‖Δ‖ / RMS of the pre sessions' leave-one-out ‖Δ‖. It is about 1 at pre by construction.
- **rmsz** = the RMS over features of Δ divided by the pre between-session SD.

Δ is projected on the scoring model as b_k = −w_k·Δ/sdN (w = coef / scaler scale), centered over spouts. That gives a
log-odds bias per spout, the same projection as rest_centre_1104 A2.

**Check 2: how do the reference spouts themselves decode?** Recall and confusion at close_L and close_center, under each
frame.

**Check 3: how much bias does Rref pick up from the reference spouts' own change?** "refmean" is the projected bias of
the Δ averaged over the reference spouts, which is exactly the Rref center's own error relative to that proxy. It is
set beside the realized recall change Rref − Wt at each spout.

### Reproduction (passed before any new frame was read)

- **Production section D, all 8 units.** Model ids are as expected, per-session accuracy and confusion are exact, and
  the W predictions are identical row for row.
- **The earlier session tables.** W / Wt / N equal `centre_mixshift_1104` `D_sessions_frames.csv` to 1e-16 (642
  session × frame cells).
- **The headlines.**

  | align | frame | far contra, acute | overall, acute |
  |---|---|---|---|
  | cue | W | .184 | .570 |
  | cue | Wt | .230 | .575 |
  | precue | W | .135 | .356 |
  | precue | Wt | .198 | .355 |

- **Reference spouts:** `REFERENCE_SPOUTS` maps to display codes 1 / 0. Both had ≥ 5 working and contact rows in every
  scored session (49-119 per spout per session).

## 3. Results

### 3a. Per animal (cue unless stated; session means; pre = held-out LOSO)

**Cue far contra.**

| animal | epoch | W | Wt | N | Rref_w_a | Rref_w_b | Iref_Wt | Iref_Wt_bal | Iref_N | Iref_N_bal | S1roi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | .865 | .867 | .858 | .856 | .843 | .854 | .853 | .849 | .849 | .774 |
| PS92 | acute | .095 | .156 | .175 | .094 | .085 | .287 | .349 | .325 | .366 | .551 |
| PS92 | chronic | .654 | .717 | .314 | .668 | .614 | .829 | .841 | .841 | .842 | .751 |
| PS93 | pre | .753 | .746 | .726 | .731 | .723 | .725 | .739 | .707 | .714 | .695 |
| PS93 | acute | .166 | .160 | .238 | .151 | .184 | .137 | .160 | .151 | .182 | .311 |
| PS93 | chronic | .573 | .640 | .608 | .588 | .632 | .662 | .687 | .651 | .668 | .700 |
| PS94 | pre | .915 | .919 | .919 | .909 | .906 | .913 | .915 | .914 | .919 | .862 |
| PS94 | acute | .282 | .359 | .329 | .292 | .306 | .277 | .429 | .400 | .547 | .465 |
| PS94 | chronic | .581 | .675 | .335 | .483 | .425 | .817 | .824 | .781 | .786 | .864 |
| PS95 | pre | .929 | .926 | .922 | .910 | .907 | .917 | .917 | .916 | .917 | .861 |
| PS95 | acute | .195 | .244 | .293 | .146 | .195 | .244 | .341 | .341 | .463 | .415 |
| PS95 | chronic | .815 | .847 | .690 | .796 | .772 | .885 | .889 | .886 | .888 | .860 |

**Cue overall, acute.**

| animal | W | Wt | N | Rref_w_a | Rref_w_b | Iref_Wt | Iref_N | Iref_N_bal | S1roi |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | .572 | .590 | .585 | .588 | .575 | .579 | .595 | .613 | .618 |
| PS93 | .379 | .380 | .392 | .385 | .386 | .363 | .381 | .404 | .478 |
| PS94 | .561 | .580 | .666 | .604 | .666 | .604 | .697 | .712 | .685 |
| PS95 | .769 | .749 | .833 | .794 | .813 | .767 | .829 | .813 | .787 |

**Precue far contra, acute.**

| animal | W | Wt | N | Rref_w_a | Iref_Wt | Iref_Wt_bal | Iref_N | Iref_N_bal | S1roi |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | .101 | .156 | .148 | .115 | .147 | .198 | .130 | .182 | .363 |
| PS93 | .064 | .093 | .166 | .053 | .064 | .104 | .080 | .108 | .260 |
| PS94 | .228 | .324 | .423 | .172 | .232 | .346 | .225 | .339 | .252 |
| PS95 | .146 | .220 | .268 | .122 | .049 | .244 | .098 | .244 | .195 |

The contact variants (Rref_c_*) are within .01-.05 of the working variants: at the close spouts nearly every working
row is a contact row. All spouts, classes and epochs are in `tables/A1_<align>_<readout>.csv`.

### 3b. Headlines, animal-weighted (level; frame − Wt with `boot_delta` 95%)

| quantity | W | Wt | N | Rref_w_a | Rref_w_b | Iref_Wt | Iref_Wt_bal | Iref_N | Iref_N_bal | S1roi |
|---|---|---|---|---|---|---|---|---|---|---|
| **cue far contra, acute** | .184 | **.230** | .259 | .171 (−.059 [−.090, −.024]) | .192 (−.037 [−.068, +.009]) | .236 (+.007 [−.068, +.096]) | .320 (+.090 [+.023, +.163]) | .304 (+.075 [+.003, +.148]) | .390 (+.160 [+.067, +.228]) | .435 (+.206) |
| **cue overall, acute** | .570 | .575 | .619 | .593 (+.018 [−.001, +.038]) | .610 (+.035) | .578 (+.003) | .578 (+.003) | .626 (+.051 [−.000, +.107]) | .636 (+.061 [+.020, +.116]) | .642 (+.068) |
| **cue far contra, chronic** | .656 | .720 | .487 | .634 (−.086 [−.157, −.037]) | .611 (−.109) | .798 (+.078 [+.026, +.129]) | .810 (+.090) | .790 (+.070) | .796 (+.076 [+.030, +.125]) | .794 (+.074) |
| cue overall, chronic | .767 | .802 | .670 | .782 (−.020) | .768 (−.033) | .820 (+.018) | .830 (+.028) | .810 (+.008) | .820 (+.018) | .839 |
| cue far contra, subacute | .565 | .602 | .505 | .570 (−.032) | .576 | .639 | .649 | .661 | .677 | .667 |
| **precue far contra, acute** | .135 | **.198** | .251 | .115 (−.083 [−.131, −.039]) | .118 (−.080) | .123 (−.075 [−.141, −.014]) | .223 (+.025 [+.012, +.041]) | .133 (−.065) | .218 (+.020 [+.004, +.036]) | .267 |
| precue overall, acute | .356 | .355 | .345 | .371 (+.016 [+.007, +.025]) | .372 | .351 | .359 | .356 | .362 | .380 |
| cue acute, contact / incomplete | .662 / .205 | .658 / .249 | .705 / .285 | .691 / .205 | .699 / .269 | .666 / .237 | .650 / .302 | .708 / .309 | .708 / .358 | .714 / .362 |
| precue acute, contact / incomplete | .403 / .180 | .395 / .204 | .372 / .243 | .430 / .149 | .429 / .160 | .403 / .147 | .395 / .221 | .408 / .150 | .403 / .209 | .414 / .246 |

All cells and intervals are in `tables/A2_headlines.csv`.

### 3c. Held-out pre accuracy (the no-pre-cost sanity check; frame − Wt)

| | Wt | Rref_w_a | Rref_w_b | Iref_Wt | Iref_Wt_bal | Iref_N | Iref_N_bal | S1roi |
|---|---|---|---|---|---|---|---|---|
| cue overall | .851 | −.009 [−.016, −.003] | −.012 | −.007 [−.011, −.003] | −.001 [−.005, +.003] | −.005 | **+.000 [−.007, +.006]** | −.078 |
| cue far contra | .864 | −.013 [−.026, −.001] | −.020 | −.012 | −.009 [−.017, −.001] | −.018 | −.015 [−.031, −.000] | −.066 |
| precue overall | .522 | −.004 | −.009 | −.024 [−.040, −.012] | **+.002** | −.023 | −.003 | −.112 |
| precue far contra | .519 | −.033 [−.059, −.009] | −.041 | −.026 | +.005 | −.025 | −.001 | −.118 |

- **Rref has a small pre cost:** −.009 overall on cue and −.033 at precue far contra.
- **Plain Iref costs precue:** −.024. Its intercepts take up the session's class mix.
- **The balanced Iref variants cost essentially nothing.**
- **S1roi is NOT an upper bound at pre** (.773 against .851). Roughly 300 rows per session are too few to refit 264
  features against a model pooled over 7-8 days. It is the upper bound only post-stroke, where it is the only arm that
  can learn a new pattern.
- **The s1 table agrees with S1roi on overall accuracy** (joint LocaNMF, `camlick` / `camlickpl_lick`; the two are
  identical here):

  | epoch | s1 | S1roi |
  |---|---|---|
  | pre | .783 | .773 |
  | acute | .646 | .642 |
  | chronic | .849 | .839 |

  The s1 table has no per-spout values (`tables/A3_upper_*.csv`).

### 3d. The far-contra deficit (epoch − held-out pre, per animal, then `boot_delta`)

| frame | PS92 | PS93 | PS94 | PS95 | pooled, cue acute | pooled, cue chronic | pooled, precue acute |
|---|---|---|---|---|---|---|---|
| W | −.770 | −.586 | −.633 | −.734 | −.681 [−.766, −.595] | −.210 | −.366 |
| Wt | −.712 | −.586 | −.560 | −.682 | −.635 [−.710, −.558] | −.145 [−.221, −.080] | −.321 |
| N | −.683 | −.488 | −.590 | −.629 | −.597 | −.369 | −.271 |
| Rref_w_a | −.761 | −.580 | −.617 | −.764 | −.681 [−.779, −.583] | −.218 | −.370 |
| Iref_Wt | −.567 | −.588 | −.636 | −.673 | −.616 [−.671, −.559] | −.054 [−.095, −.009] | −.370 |
| Iref_Wt_bal | −.504 | −.579 | −.486 | −.575 | −.536 [−.586, −.473] | −.046 | −.301 |
| Iref_N | −.524 | −.556 | −.514 | −.575 | −.542 [−.587, −.490] | −.057 | −.361 |
| **Iref_N_bal** | −.482 | −.532 | −.371 | −.454 | **−.460 [−.527, −.392]** | **−.053 [−.114, +.001]** | −.299 [−.369, −.222] |
| S1roi | −.223 | −.385 | −.397 | −.447 | −.363 [−.443, −.262] | −.005 | −.133 |

The per-animal columns are the cue acute values. Chronic and precue per animal: `tables/A4_deficit.csv`,
`deficit_refspout_log.txt`.

### 3e. Is the Iref gain bought with false alarms? (cue acute; non-far-contra rows called far contra)

| animal | Wt false alarms | Iref_Wt_bal | Iref_N_bal | pre (Wt) | far-contra precision, Wt / Iref_N_bal |
|---|---|---|---|---|---|
| PS92 | .047 | .105 | .109 | .027 | .370 / .373 |
| PS93 | .168 | .176 | .168 | .058 | .156 / .173 |
| PS94 | .100 | .118 | .088 | .014 | .350 / .455 |
| PS95 (1 session) | .022 | .111 | .075 | .014 | .476 / .333 |

- PS93 and PS94 gain far-contra recall at no precision cost.
- PS92 and PS95 pay with about 2-5x more far-contra calls on other spouts.
- So part of the balanced-Iref gain is a re-thresholding of far contra, not a recovered pattern.

## 4. Assumption checks

### Check 1: the reference spouts change after the stroke

**Size of the change** (cue, working rows; rel, where pre ≈ 1 by construction).

| animal | epoch | contrast (close_L − close_center, no proxy) | close_L vs Wt proxy | close_center vs Wt proxy | close_L raw | close_center raw |
|---|---|---|---|---|---|---|
| PS92 | acute | 0.76 | 1.60 | 0.96 | 0.87 | 0.63 |
| PS92 | chronic | 0.80 | 2.16 | 1.74 | 2.63 | 1.78 |
| PS93 | acute | 1.83 | 3.70 | 0.71 | 2.31 | 1.01 |
| PS93 | chronic | 2.35 | 2.89 | 1.90 | 2.98 | 2.17 |
| PS94 | acute | 2.24 | 2.45 | 2.33 | 2.37 | 3.08 |
| PS94 | chronic | 2.37 | 2.02 | 1.35 | 1.59 | 2.93 |
| PS95 | acute | 1.58 | 2.39 | 1.41 | 2.54 | 1.84 |
| PS95 | chronic | 1.49 | 2.18 | 1.48 | 2.63 | 2.29 |

- **On the proxy-free contrast the change exceeds pre day-to-day variation in 3/4 animals acutely and chronically.**
  - pooled post − pre: acute +.65 [+.06, +1.18], chronic +.80 [+.13, +1.45];
  - PS92 is within pre range.
- **Against the Wt proxy, close_L changes more than close_center** in all four animals acutely (1.6-3.7 against
  0.7-2.3).
- **In units of the pre between-session SD (rmsz)**, post values are 0.8-4.9, against 1.2-1.3 at pre. Full tables:
  `tables/C1_size_*`.
- **Precue** changes less: contrast acute +.29 [−.21, +.84]; chronic +.43 [+.03, +.81].

**Projected on the decoder** (cue, far-contra log-odds bias of the reference-spout mean shift, "refmean"; pre ≈ −.07).

| animal | acute, Wt proxy | acute, R5 (rest) proxy | chronic, Wt proxy | chronic, R5 proxy |
|---|---|---|---|---|
| PS92 | −.50 | +.22 | +.25 | +2.05 |
| PS93 | −.46 | −.89 | +.74 | +1.93 |
| PS94 | −.35 | +2.01 | −.95 | +2.93 |
| PS95 | −1.33 | +1.46 | −.06 | +2.16 |
| pooled post − pre | −.59 [−1.08, −.21] | +.79 [−.35, +1.92] | +.07 [−.60, +.68] | +2.35 [+1.83, +2.92] |

**The size and even the sign of the reference spouts' "own" change depend on which proxy stands in for the day
offset.** The two proxies disagree because the rest center itself moves after the stroke (REST_CENTRE §5d). The shift
is also not confined to far contra. Under the Wt proxy, acute biases reach +1.84 at close_ipsi (PS95) and −1.14 at
close_contra (PS93). Per spout: `tables/C1_bias_*`.

### Check 2: decoding at the reference spouts changes too (cue; recall; main error destination)

| animal | close_L pre (Wt) | close_L acute: W / Wt / N / Rref / Iref_N | close_L acute errors (Wt) | close_center pre (Wt) | close_center acute: W / Wt / N / Rref / Iref_N |
|---|---|---|---|---|---|
| PS92 | .885 | .781 / .767 / .863 / .798 / .783 | close_mid .150 | .838 | .703 / .707 / .509 / .720 / .616 |
| PS93 | .902 | .590 / .628 / .730 / .641 / .763 | close_mid .130, far_mid .107 | .650 | .558 / .488 / .374 / .554 / .359 |
| PS94 | .968 | .539 / .598 / .932 / .693 / .860 | **close_mid .359** | .890 | .903 / .832 / .666 / .879 / .717 |
| PS95 | .917 | .899 / .731 / .966 / .958 / .924 | close_mid .210 | .851 | .841 / .796 / .885 / .903 / .894 |

- **Under the session-centered frames, close_L loses .12-.37 acutely in every animal.** Most of the errors go to
  close_center.
- **Under N it is spared, but close_center falls instead.** PS92 .509, with its errors going to close_ipsi .19 and
  close_contra .17; PS94 .666.
- So the session offset shifts the close_L / close_center boundary, and the two reference spouts trade accuracy
  depending on the center.
- **Chronic, PS93:** close_center stays low under Wt (.457, errors to close_ipsi .25 and close_contra .24). Rref raises
  it to .767.
- Full confusions: `tables/C2_ref_confusion.csv`.

### Check 3: how much bias the reference spouts' change adds under Rref

| animal | Rref − Wt center distance, acute (pre) | projected far-contra bias, acute (Wt proxy) | realized far-contra recall Rref − Wt, acute (pre) | realized close_L recall Rref − Wt, acute |
|---|---|---|---|---|
| PS92 | .21 (.10) | −.50 | −.061 (−.011) | +.030 |
| PS93 | .28 (.13) | −.46 | −.009 (−.015) | +.012 |
| PS94 | .31 (.12) | −.35 | −.066 (−.009) | +.095 |
| PS95 | .26 (.13) | −1.33 | −.098 (−.016) | +.227 |

The center distance is in pooled-pre working SD units.

- **Every post-stroke center distance is 2-3x the pre value.** The Rref center departs from Wt by exactly the reference
  spouts' change relative to the other four spouts.
- **Rref moves the spout scores as one pattern.** It raises recall at the anchor spouts (close_L +.01 to +.23) and
  lowers far contra (−.01 to −.10) in all four animals.
- **What remains unknown is which center is unbiased.** Against the rest proxy the same Rref error would push far contra
  UP (+.79 pooled).
- **Precue behaves the same way:** far contra −.04 to −.15 and close_L / close_center −.01 to +.14 (`tables/C3_*`).

### The fitted offsets (Iref, full session, class-centered log-odds; cue acute)

| animal | Iref_Wt far contra | Iref_Wt close_ipsi | Iref_N far contra | Iref_N_bal far contra | Iref_N close_ipsi |
|---|---|---|---|---|---|
| PS92 | +.82 | −.39 | +1.34 | +1.44 | −1.23 |
| PS93 | −.12 | +.61 | −.65 | −.51 | +.68 |
| PS94 | −.32 | +.51 | +.99 | +1.56 | −2.05 |
| PS95 | +.06 | +.68 | +1.67 | +2.95 | −.99 |

Pre values are within ±.2 everywhere.

- **In Wt's frame the data-optimal far-contra offset has no consistent sign.** Wt leaves no systematic far-contra
  offset for the data to correct, which is why Iref_Wt (.236) ≈ Wt (.230).
- **In N's frame three of four animals want a large positive far-contra offset.**
- **Chronically the offsets grow** in both frames, most at close_ipsi (Iref_N −3.0 to −3.9). The fixed frames drift
  further from the data over time, which is why Iref recovers chronic far contra to .79-.81.
- Full table: `tables/C4_iref_offsets_*.csv`.

## 5. Reading

**Does Rref agree with Wt? No; it reads lower at far contra.**

| quantity | Rref_w_a | Wt | Rref − Wt | lower in |
|---|---|---|---|---|
| cue far contra, acute | .171 | .230 | −.059 [−.090, −.024] | 4/4 animals (PS93 only −.009) |
| cue far contra, chronic | .634 | .720 | −.086 [−.157, −.037] | |
| precue far contra, acute | .115 | .198 | −.083 [−.131, −.039] | 4/4 animals |

- **Overall it is slightly higher** (acute +.018). The gain is at the close spouts, the anchors.
- **The rescaled variant Rref_w_b is closer to Wt** (.192 acute), but its pre cost is larger (−.012 overall, −.020 far contra).
- **Rref has a small pre cost:** −.009 cue overall, −.033 precue far contra.

**Is its assumption violated? Yes, and the size of the bias cannot be fixed without an independent day-offset
reference.**

- **The reference spouts change after the stroke.** The proxy-free close_L − close_center contrast moves 1.5-2.4x
  beyond pre day-to-day variation in 3/4 animals, acutely and chronically. Against a six-spout center, close_L changes
  most (1.6-3.7x).
- **Their decoding changes too:** close_L loses .12-.37 acutely under Wt, mostly into close_center.
- **Under Rref this change is transferred to every spout.** Relative to Wt it projects to −.35 to −1.33 log-odds at far
  contra acutely, realized as −.01 to −.10 recall.
- **The sign flips under the rest proxy.** So "how much Rref is biased" depends on which post-stroke quantity is taken
  to be unchanged. That is the same circularity Priya raised: Rref swaps the six-spout assumption for a two-spout one,
  and the two-spout one is measurably violated.
- Rref is therefore a different center, not a stroke-free one.

**What does Iref (frozen weights, offsets free) say the far-contra deficit is?**

- **Large, in every animal, whichever frame and likelihood is used.** Cue acute far contra is .24-.39 against .85-.86
  pre held-out:

  | frame | acute − pre |
  |---|---|
  | Iref_Wt | −.62 [−.67, −.56] |
  | Iref_N | −.54 [−.59, −.49] |
  | Iref_N_bal | −.46 [−.53, −.39] |

  Per animal (Iref_N_bal) the deficit is −.37 to −.53. For comparison, Wt's deficit is −.64 [−.71, −.56].
- **Up to .17 of Wt's acute far-contra loss is offset-dependent** (.02 for plain Iref in Wt's frame, .17 for balanced Iref in N). Free offsets recover it. The rest (−.46 to −.62)
  is the pre-stroke pattern geometry failing at far contra.
  - Two cautions: in PS92 / PS95 part of the balanced gain comes with 2-5x more far-contra false alarms;
  - and the plain-ML Iref_Wt equals Wt (.236).
- **Precue tells the same story.** The acute far-contra deficit under the balanced Iref is −.30 [−.37, −.22], against
  Wt's −.32.
- **Chronically, with offsets free, cue far contra is back near pre:** Iref −.05 [−.11, +.00], against Wt −.145. With
  free offsets most of Wt's chronic residual deficit is a day offset, not a pattern change. On cue the full within-session
  refit (S1roi) agrees (−.005). At precue a deficit remains (Iref −.10, Wt −.14).
- **The within-session refit (S1roi) is the post-stroke ceiling.** Acute far contra .435 (deficit −.36) and overall
  .642 (the s1 table gives .646). So a session-specific decoder can read far contra better than the frozen geometry
  can, even with free offsets. At least part of the post-stroke far-contra pattern is new rather than shifted.

Nothing here is a production recommendation; Priya decides.

## 6. Limits

- **n = 4 animals.** PS95 has ONE acute session, and its acute cells are single-session values.
- **Iref uses the scored session's labels to fit its offsets.** This is cross-validated over blocks, but it is
  label-informed recalibration with 5 free parameters per session. It is an upper bound on "pattern present, offset
  unknown", not a deployable frame.
  - Plain ML also imports the session's class mix as a prior, and acute far-contra working rows are scarce. The
    balanced variant removes that, but it can re-threshold far contra (§3e).
- **Intercepts absorb offsets, not scale.** Iref_Wt and Iref_N differ only in the per-session scale (session SD vs
  pooled pre SD), and they give .236 vs .304. Scale stays a choice.
- **There is no independent day-offset reference.** Every proxy in check 1 is itself post-stroke data (the six-spout
  center, rest), except the close_L − close_center contrast. The contrast shows that the reference spouts change
  RELATIVE to each other, but it cannot say how either changes against a true baseline.
- **The projection in check 1 / 3 is linear and in pooled-pre SD units.** It sets the direction and rough size of a
  score bias. The realized recall change is the measured effect.
- **S1roi is the ROI-pool refit** on the same rows and folds. The s1 table is the joint-LocaNMF refit and has only
  overall accuracy, so the per-spout upper bound comes from S1roi alone.
- **Reference spouts have ample rows** (≥ 49 per spout per session). The contact and working variants coincide because
  close-spout working rows are almost all contacts.

## 7. Files (`decoder_mix_checks/refspout_centre_1104/`)

**Scripts:**

- `refspout.py`: reproduction, the Rref frames, Iref, S1roi and checks 1 / 3, per (animal, align); 3 workers.
- `summarise_refspout.py`: sections R and 1-8.
- `deficit_refspout.py`: epoch − pre deficits.

**Logs:**

- `refspout_log.txt`, plus `refspout_test_PS95_precue_log.txt` (the first unit, run alone);
- `summarise_refspout_log.txt`;
- `deficit_refspout_log.txt`.

**Per unit (`decoder/`):** `<an>_<al>_rows.csv.gz`, `_sessions.csv`, `_refshift.csv`, `_iref_b.csv`, `_repro.json`.

**Tables (`tables/`):**

- `sessions_frames.csv`;
- `A1_<align>_<readout>.csv` (animal × epoch × frame);
- `A2_headlines.csv`;
- `A3_upper_<align>.csv`;
- `A4_deficit.csv`;
- `C1_refshift_sessions.csv`, `C1_size_*`, `C1_bias_*`, `C1_pooled_post_minus_pre.csv`;
- `C2_ref_confusion.csv`;
- `C3_realised_Rref_minus_Wt_*`, `C3_projected_refmean_*`;
- `C4_iref_offsets_*`;
- `C5_sessions_info.csv`;
- `C6_far_contra_false_alarms_*`.
