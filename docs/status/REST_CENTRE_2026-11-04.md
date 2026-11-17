# Should the frozen spout decoder's per-session frame come from REST? (2026-11-04)

Cohort 1 (PS92-95). A read-only diagnostic. No production default, model, deck or page was changed. The rest-frame
decoders were refitted IN MEMORY only; nothing was stored to or published from the frozen store.
Scripts, CSVs and logs are in `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/rest_centre_1104/`. The design
follows `CENTRE_MIXSHIFT_2026-11-04.md`, reusing its raw pools and its frame engine.

## 1. Question

Priya asked whether the per-session frame could be set from REST instead of from trials. Rest is free of the trial mix
by construction: no spout mix, no incomplete share, no labels.

The concern was that rest itself changes after the stroke:

- quiet-period orofacial events run ~5x acutely;
- rest is a larger share of post-stroke sessions;
- the quiet baseline drifts.

A rest center would absorb all of that.

## 2. Design

Every arm uses:

- the same pool, features and scored rows: section D's ROI pool (66 Allen areas x 4 sub-bins = 264 columns), scored on
  each session's camera working rows, cue and precue;
- the same training rows and estimator: production's pre-stroke working rows (August-only pre, `WIDEFIELD_PRE_FROM=0801`)
  and `locanmf_frozen_decoder._pipe`;
- the full model for post sessions and the leave-that-session-out model for each held-out pre session.

| tag | frame per session (training pre AND scored) | model |
|---|---|---|
| W | production bal_em of the working rows | stored (frozen-hit) |
| Wt | true-label balanced center of the working rows (the provisional default) | stored |
| N | pooled pre, no session center | stored |
| Wt_fit | Wt, with production's training frame refitted in memory | refit check |
| **R5** | center = frame mean, scale = SD of the 0.5 s box mean, over the session's restdock05 rest frames, engagement-gated | refit |
| **RW** | restw: per-position MEDIAN over same-position rest periods, equal weight; scale = sqrt(mean per-position variance) | refit |
| **R5q** | R5 minus frames within −0.5 to +1.0 s of a quiet-period orofacial event | refit |
| R5c | R5 rest center + working-row SD (decomposition) | refit |
| R5s | Wt center + R5 rest SD (decomposition) | refit |

### Rest sets

- **R5:** `quiet_periods.quiet_frame_path` (the restdock05 variant, no fallback) AND
  `rest_engagement.engaged_frame_mask` (frames before the terminal quit). This is the audit's rule: the mask is the
  builder and the gate is applied where the mask is used.
- **RW:** `rest_by_position.rest_frames_by_position(docked=False, engaged_only=True)` and then `restw_from_frames`,
  with production's floors (>= 200 frames per position, all 6 positions).
- **R5q:** the quiet-events arm's detector, `scripts/cohort2_qc/quiet_spikes_v2.detect` (stored per session in
  `cohort1_quiet_spikes/v2_quiet/sessions/`).
  - Columns: orofacial_left / orofacial_right, either polarity.
  - Threshold: height > 4 x the animal's August-pre robust SD (the `quiet_spikes_v2.load_all` rule).
  - These are RAW events, not face-regressed, so movement-coupled events are removed too.

### Units

The rest statistics are computed on the same trace the features are windowed from
(`locanmf_position_decoder._build_signal(s, "roi")`). They are tiled over the 4 bins and mapped onto the pool's columns
with `_align_many`'s keying. The mapping is checked bitwise: one session per animal, recomputed features against the
pool's raw rows.

## 3. Checks

All checks passed before any rest arm was read.

- **Production section D reproduced exactly in all 8 units.** The model ids are as expected, and the per-session
  accuracy and confusion are exact. Cue far contra acute is .184 and overall .570; precue far contra is .135 and
  overall .356.
- **The in-memory refit reproduces the store.** Wt_fit predictions equal the stored model's on every scored row, in all
  8 units. So the rest arms differ from Wt only by their frame.
- **Column map:** bitwise in 4 / 4 animals.

## 4. Rest frame counts (check 4)

107 scored sessions, 31 of them held-out pre.

| | pre | acute | subacute | chronic |
|---|---|---|---|---|
| R5 frames (median) | 16,002 | 14,297 | 11,118 | 13,133 |
| R5q frames (median) | 14,843 | 9,386 | 8,192 | 10,575 |
| share of R5 removed by events | .10 | **.35** | .21 | .19 |
| RW frames (median) | 13,225 | 11,912 | 9,225 | 10,932 |

- The minimum is 3,947 R5 frames (PS92_0812), 3,580 R5q and 3,208 RW.
- Every session has restw at all 6 positions.
- **No session fell below the 500-frame floor**, so none was excluded from any arm.
- Most events: PS93_0820 (acute) loses 53% of its R5 frames to events.

## 5. Results

### 5a. Per animal, cue (session means; pre = held-out LOSO)

**Far contra.**

| animal | epoch | W | Wt | N | R5 | RW | R5q |
|---|---|---|---|---|---|---|---|
| PS92 | pre | .865 | .867 | .858 | .828 | .834 | .818 |
| PS92 | acute | .095 | .156 | .175 | .179 | .152 | .181 |
| PS92 | chronic | .654 | .717 | .314 | .311 | .242 | .293 |
| PS93 | pre | .753 | .746 | .726 | .717 | .723 | .728 |
| PS93 | acute | .166 | .160 | .238 | .164 | .178 | .194 |
| PS93 | chronic | .573 | .640 | .608 | .320 | .315 | .315 |
| PS94 | pre | .915 | .919 | .919 | .916 | .920 | .911 |
| PS94 | acute | .282 | .359 | .329 | .139 | .113 | .158 |
| PS94 | chronic | .581 | .675 | .335 | **.035** | .046 | .024 |
| PS95 | pre | .929 | .926 | .922 | .915 | .913 | .919 |
| PS95 | acute | .195 | .244 | .293 | .122 | .098 | .171 |
| PS95 | chronic | .815 | .847 | .690 | .520 | .497 | .559 |

**Overall.**

| animal | epoch | W | Wt | N | R5 | RW | R5q |
|---|---|---|---|---|---|---|---|
| PS92 | acute | .572 | .590 | .585 | .573 | .562 | .577 |
| PS92 | subacute | .778 | .782 | .632 | .433 | .426 | .388 |
| PS92 | chronic | .801 | .835 | .645 | .681 | .640 | .659 |
| PS93 | acute | .379 | .380 | .392 | .400 | .396 | .392 |
| PS93 | chronic | .637 | .680 | .570 | .551 | .552 | .547 |
| PS94 | acute | .561 | .580 | .666 | .575 | .563 | .594 |
| PS94 | chronic | .777 | .821 | .700 | .564 | .574 | .557 |
| PS95 | acute | .769 | .749 | .833 | .646 | .677 | .670 |
| PS95 | chronic | .852 | .872 | .764 | .669 | .670 | .681 |

All spouts, classes and epochs are in `tables/A1_<align>_<readout>.csv`.

### 5b. Headlines, animal-weighted (frame − Wt, `boot_delta` 95%)

| quantity | W | Wt | N | R5 | RW | R5q |
|---|---|---|---|---|---|---|
| **cue far contra, acute** | .184 | **.230** | .259 | .151 (−.079 [−.185, +.019]) | .135 (−.095 [−.207, +.010]) | .176 (−.054 [−.159, +.032]) |
| **cue overall, acute** | .570 | .575 | .619 | .548 (−.026 [−.078, +.019]) | .549 (−.025) | .558 (−.017) |
| **cue far contra, chronic** | .656 | .720 | .487 | .296 (−.423 [−.571, −.307]) | .275 (−.444) | .298 (−.422) |
| cue overall, chronic | .767 | .802 | .670 | .616 (−.186 [−.241, −.134]) | .609 (−.193) | .611 (−.191) |
| cue far contra, subacute | .565 | .602 | .505 | .337 (−.265 [−.408, −.128]) | .331 (−.271) | .321 (−.281) |
| cue acute, contact / incomplete | .662 / .205 | .658 / .249 | .705 / .285 | .633 / .213 | .635 / .208 | .642 / .224 |
| **precue far contra, acute** | .135 | **.198** | .251 | .198 (+.000 [−.043, +.053]) | .182 (−.015) | .204 (+.006) |
| precue overall, acute | .356 | .355 | .345 | .329 (−.026 [−.054, −.004]) | .333 (−.022) | .338 (−.017) |

- **Cue far contra acute, per animal (R5 vs Wt):**
  - PS92 .179 vs .156 and PS93 .164 vs .160 agree;
  - PS94 .139 vs .359 and PS95 .122 vs .244 are lower.
  - Two of the four animals carry the pooled gap.

### 5c. Pre-stroke sanity (check 3; held-out pre, frame − Wt)

| | Wt | R5 | RW | R5q | R5c | R5s |
|---|---|---|---|---|---|---|
| cue overall | .851 | .821 (−.030 [−.052, −.010]) | .823 (−.028 [−.050, −.010]) | .818 (−.032) | .830 (−.021) | .849 (−.002) |
| cue far contra | .864 | .844 (−.020 [−.048, −.001]) | .847 (−.017) | .844 (−.021) | .852 (−.012) | .861 (−.004) |
| precue overall | .522 | .512 (−.009) | .512 (−.009) | .510 (−.011) | .514 (−.008) | .521 (−.000) |

- **The rest frames cost pre accuracy.** The loss is mostly in PS92 (−.049 overall) and PS95 (−.042); PS93 is −.023
  and PS94 −.004.
- So a rest frame fails the "must not cost pre accuracy" criterion on cue.
- The cost comes from the rest CENTER (R5c −.021), not the rest scale (R5s −.002).

### 5d. Does the rest state move after the stroke? (check 2)

**What the bias is.** For each session, the shift of its rest center from the pre mean is projected onto the arm's full
model (leave-one-out mean for a pre session). The shift is in training-frame units (pre-mean rest SD), and the weights
are `coef / scaler.scale`. The result is the class-centered score bias b_k = −w_k·Δc, in log-odds; it is 0 at pre by
construction.

**Two references for the shift:**

- **raw** = the rest center in dF/F. dF/F is relative to the session's mean frame.
- **vsWt** = the rest center minus the session's own true-label working center. This is the bias the rest frame adds
  RELATIVE to Wt.

**Far-contra bias, cue** (session means; median W margin, top 1 − top 2, for scale):

| animal | epoch | R5 raw | R5q raw | event part (R5 − R5q) | R5 vsWt | RW vsWt | W margin |
|---|---|---|---|---|---|---|---|
| PS92 | acute | −.22 | −.14 | −.08 | −.51 | −.64 | 1.61 |
| PS92 | chronic | +.10 | +.19 | −.09 | −1.60 | −1.73 | 2.47 |
| PS93 | acute | −.11 | −.00 | −.11 | +.41 | +.62 | 1.16 |
| PS93 | chronic | −.48 | −.49 | +.01 | −1.08 | −1.15 | 1.63 |
| PS94 | acute | −.99 | −.74 | −.25 | −2.74 | −2.88 | 1.70 |
| PS94 | chronic | −1.40 | −1.53 | +.13 | −4.16 | −3.93 | 2.18 |
| PS95 | acute | −1.23 | −.77 | −.46 | −3.26 | −3.36 | 1.94 |
| PS95 | chronic | −.49 | −.48 | −.01 | −1.71 | −1.79 | 2.89 |

**Pooled R5 far-contra bias:**

| reference | acute | subacute | chronic |
|---|---|---|---|
| raw | −.64 [−1.12, −.16] | −.39 [−.86, −.05] | −.57 [−1.16, −.07] |
| vsWt | −1.53 [−3.04, −.05] | −1.93 [−3.12, −.73] | −2.14 [−3.51, −1.22] |

Per-spout tables: `tables/A2_bias_<align>_<ref>.csv`.

**Rest SD ratio, post / pre** (R5, 0.5 s box SD, median over regions), set beside the SD of the TRIAL working rows:

| animal | acute rest (trial) | subacute rest (trial) | chronic rest (trial) |
|---|---|---|---|
| PS92 | 1.24 (1.11) | 3.15 (1.93) | 1.48 (1.14) |
| PS93 | 1.50 (1.11) | 1.44 (1.23) | 1.44 (1.21) |
| PS94 | 1.32 (1.41) | 1.36 (1.37) | 1.47 (1.25) |
| PS95 | 1.28 (1.14) | 1.05 (1.13) | 1.11 (1.08) |

Without the event frames (R5q) the ratios are almost the same (PS92-95 acute 1.23 / 1.49 / 1.30 / 1.24).

**Reading the rest-state check:**

- **The rest center moves after the stroke, and in the same direction in every animal.**
  - Its shift projects to a NEGATIVE far-contra score bias in 4 / 4 animals acutely (raw).
  - The shift is a large fraction of the score margin: PS94 −.99 against a margin of 1.70, PS95 −1.23 against 1.94.
  - It does not return in chronic.
- **The quiet events carry part of the acute bias, not most of it:**

  | animal | event part of the acute far-contra bias | share |
  |---|---|---|
  | PS92 | −.08 | 37% |
  | PS93 | −.11 | ~100% |
  | PS94 | −.25 | 25% |
  | PS95 | −.46 | 38% |

  Subacute and chronic, the event part is ~0. The events do not carry the SD rise either (R5q ≈ R5).
- **Rest SD rises 1.2-1.5x after the stroke in every animal, and stays up in three.** That is consistent with the
  quiet-events SD maps (15.6: quiet SD up over most of dorsal cortex at every epoch). Acutely it rises more than the
  trial-row SD in PS92 / PS93 / PS95.

### 5e. Which half of the rest frame does the damage (decomposition, cue far contra, frame − Wt)

| epoch | Wt | R5 | R5c (rest center, trial SD) | R5s (trial center, rest SD) |
|---|---|---|---|---|
| pre | .864 | .844 (−.020) | .852 (−.012) | .861 (−.004) |
| acute | .230 | .151 (−.079 [−.185, +.019]) | .191 (−.038 [−.096, .000]) | .312 (+.083 [+.038, +.126]) |
| subacute | .602 | .337 (−.265) | .399 (−.203 [−.365, −.078]) | .606 (+.004) |
| chronic | .720 | .296 (−.423) | .423 (−.297 [−.442, −.150]) | .688 (−.032 [−.145, +.046]) |

Overall, chronic: R5c −.162, R5s .000. Precue: every decomposition arm stays within ±.04 of Wt.

- **The rest CENTER is what collapses the post-stroke scores.** The rest-to-task offset is not stable across days.
- **The rest SCALE inflates acute far contra** (+.083 [+.038, +.126]). Rest SD rises more than trial SD acutely, which
  shrinks the session's features toward the center.

### 5f. Why the center matters for a FROZEN decoder

- **Frozen decoder.** A per-session offset c changes class score k by −w_k·c/sd. Predictions change unless w_k·c is
  equal across classes. A scale change multiplies every w_k·x, so it moves the margins against the intercepts (the
  training prior).
- **Per-session refit.** A decoder refitted within each session absorbs a common offset in its intercepts and its
  scaler. It cannot be biased this way.

Measured per scored row, cue (class-centered decision scores; same stored model for W vs Wt; R5 = refit model in its own
frame):

| animal | epoch | median W margin | W vs Wt: median max shift | flips | W vs Wt: mean far-contra shift | W vs R5: median max shift | flips |
|---|---|---|---|---|---|---|---|
| PS92 | pre | 2.74 | .15 | 2% | +.01 | 1.83 | 14% |
| PS92 | acute | 1.61 | .75 | 10% | +.51 | 1.87 | 27% |
| PS93 | pre | 1.54 | .24 | 3% | −.07 | 1.06 | 15% |
| PS93 | acute | 1.16 | .33 | 7% | +.08 | 1.32 | 23% |
| PS94 | pre | 3.89 | .18 | 1% | +.02 | 1.41 | 5% |
| PS94 | acute | 1.70 | .81 | 13% | +.56 | 3.08 | 42% |
| PS95 | pre | 3.51 | .14 | 1% | −.01 | 1.70 | 10% |
| PS95 | acute | 1.94 | 1.09 | 12% | +1.05 | 5.11 | 33% |

- **W vs Wt.** At pre the shift is 4-15% of the margin. Acutely it is 28-56%, and 7-13% of predictions flip. The shift
  is directed at far contra (+.5 to +1.0 log-odds in PS92 / PS94 / PS95), which is W's EM stripping far-contra mass.
- **W vs R5.** Already at pre the class scores move by 36-67% of the margin. Acutely they move by 1.1-2.6 margins, and
  23-42% of predictions flip.
- **Precue.** Margins are ~0.6-1.2 there, so the same shifts matter more: W vs Wt flips 8-12% acutely.

Full table: `tables/A5_score_shifts.csv`.

## 6. Reading

**Cue: the rest arms do NOT agree with Wt.**

- **Far contra acute (pooled).** R5 .151, RW .135 and R5q .176 against Wt .230. The intervals span 0, because two
  animals agree and PS94 / PS95 fall.
- **Subacute and chronic.** The rest arms collapse: chronic far contra ~.29 against .72; overall ~.61 against .80. They
  fall below N, the no-center frame. They also cost held-out pre accuracy (−.03 overall).
- So the trial-based acute far-contra loss (~.18-.23) is not an artifact that a mix-free rest frame removes. The rest
  frames make far contra lower, not higher.

**The rest arm is the one absorbing a stroke effect (check 2).**

- The rest center shifts post-stroke so that far contra loses log-odds in 4 / 4 animals.
- Quiet events carry 25-100% of that acutely and ~0 later. Most of it is a broader change in the rest state.
- The rest SD rises 1.2-1.5x in every animal.
- Decomposed:
  - the rest CENTER is not stable relative to task activity across days, and that loses far-contra and overall accuracy
    through chronic;
  - the rest SCALE inflates acute far contra (+.08).
- Neither half is mix-free in the sense that matters. Both track the post-stroke change in the rest state, which is the
  concern Priya raised.

**Precue: the rest arms agree with Wt** (far contra acute .198 / .182 / .204 against .198; chronic within +.04). The
pre-cue window sits closer to the rest state, so referencing it to rest is roughly neutral there. Overall precue acute
is slightly lower (−.02).

**Wt is still the frame these results favor.** It agrees with the mix-stable contact center Cto where the mix is normal
(CENTRE_MIXSHIFT §5), it costs no pre accuracy, and its acute far contra (.230) lies between W (.184) and N (.259).

No production recommendation is made here; Priya decides.

## 7. Limits

- **n = 4 animals.** The acute far-contra gap between R5 and Wt is carried by PS94 / PS95. PS95 has ONE acute session.
- **The rest feature is an approximation.** A rest "feature" is a frame mean tiled over the 4 bins. Its scale is the SD
  of a 15-frame box mean; this matches the bin's averaging but not the windows' placement relative to the cue.
  Frame-level SD ratios agree with the box ones to within .08, except PS92 subacute (2.81 vs 3.15).
- **The raw-reference bias is relative to the session's mean frame.** dF/F is computed against that frame, which moves
  when rest is a larger share of the session. The vsWt reference is free of that, but it includes any change in the
  task center. Both point the same way.
- **Events are the raw v2 detector at k = 4**, chosen to separate pre from acute. Face-regressed events were not used.
  The drop window (−0.5 to +1.0 s) is a choice; it removes 35% of acute rest frames.
- **RW uses per-position MEDIANS for its center and means for its scale.** RW ≈ R5 throughout.
- **The decomposition arms (R5c / R5s) are diagnostic only.** R5s uses the scored session's labels in its center, as Wt
  does.
- **The rest SD rise is not separated from day-to-day amplitude drift.** The trial-row SD also rises (1.08-1.41
  chronic). The rest-over-trial excess is the stroke-linked part only if the two share the same drift.

## 8. Files (`decoder_mix_checks/rest_centre_1104/`)

**Scripts:**

- `rest_stats.py`: per-session rest statistics for R5 / RW / R5q, written to `rest/<label>.npz` (derived).
- `rest_decoder.py`: section-D reproduction, the in-memory refits, scoring, the bias projection and the score shifts.
- `rest_decomp.py`: R5c / R5s and the trial-row SD.
- `summarise_rest.py`, `summarise_decomp.py`.
- `probe.py`: a setup probe.

**Logs:** `rest_stats_log.txt`, `rest_decoder_log.txt`, `rest_decomp_log.txt`, `summarise_rest_log.txt` (R = the
reproduction, then sections 1-5), `summarise_decomp_log.txt` (section 6).

**Tables (`tables/`):**

- `sessions_frames.csv`, `A1_*` (animal x epoch, per spout / class), `A1b_headlines.csv`;
- `A2_bias_sessions.csv`, `A2_bias_<al>_<ref>.csv`, `A2_eventpart_<al>.csv`, `A2_sdratio_{box,frame}_<al>.csv`;
- `A4_rest_counts.csv`;
- `A5_score_shifts.csv`;
- `A6_decomp_*`, `A6_sd_ratio_work_vs_rest.csv`.
