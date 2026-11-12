# Caption TBD fill: C6 (hemisphere balance) and D1 (state space), 2026-11-12

Fills the `{{TBD after re-run}}` placeholders of `CAPTION_PASS_DRAFT_2026-11-10.md` (branch docs/caption-pass-draft-1110,
checkout `wf_captiondraft`) sections C6 and D1 from the glue-fixed v20 re-run. READ-ONLY: nothing on M:, on a page or in a
deck was changed, nothing was re-run. Helper report for the caption pass; numbers go to Priya before any text changes.

**Three columns for every number:**

| Column | What it is | Source |
|---|---|---|
| live | the live GOAL page caption (10-28 / 10-29 pipeline, before the 11-03 rebuild) | `wf_decisions1111/_goal_page/index.html` (live text; the caption draft's "Current" column quotes the same text) |
| glue-in | the 11-03 rebuild WITHOUT the painted-glue mask (queue `_r2` / v8 runs, 11-07 .. 11-09) | step logs `scratchpad/render_daqlk/logs/<step>_r2.log` (or `<step>.log`); the tables themselves were overwritten |
| glue-out | the v20 glue re-run (code 8aed3a7b, 11-11 01:00-07:30) | the CSVs below (all written 11-11) + `<step>_g.log` |

- Glue-in -> glue-out isolates the MASK. Live -> glue-out also contains the rebuild changes (camera working-trial gate,
  LP licks, new pre definition, ...). Where a number moved, the report says which of the two moved it.
- Table folder (all paths below are relative to it): `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/
  lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/<TAG>/`, TAG = `hv-meegkit_hpfit_lag_refitT_swapmask_
  lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`. Every balance file used
  here has mtime 11-11 (post glue fix).
- The glue report `GLUE_BEFORE_AFTER_2026-11-10_2130.md` has NO row for any balance step (`l_lateral_ratio`, `l_bal_*`,
  `l_recov_primary`, `l_balimg*`): those steps finished after it was written. The glue-in/out comparison below is new
  (log-vs-log); it does not repeat anything in that report.
- Every readout is the change from the animal's OWN pre. "*" = `analysis_kit.boot_delta` 95 % CI (animals -> sessions)
  excludes 0. Mixed-model p values are quoted only where the caption is about a model (marked LMM).
- Cohort 1 only (PS92-95). PS95 has ONE acute session (n = 1) in every table below; its acute value is one session.
- Two references (RESTnl / session mean): no C6 table has them. Every balance number is on a per-lick baseline
  (-1.0..-0.5 s, `iti_licks.BASE_RESP`), which a ratio needs (decision D3). The only interhemispheric tables with two
  references are `interhemi_recovery_restnl.csv` / `interhemi_raw_recovery.csv` (`scripts/lick_interhemispheric.py`), not
  on any C6 figure. Q2 of the caption draft (does the raw 0-0.2 s decoder-window supplement need a RESTnl twin) stays open.

---

## 0. For Priya: conclusion changes (before any text changes)

### 0.1 Caused by the glue mask (glue-in -> glue-out, same rebuild code path)

| # | Fig | What changes | glue-in | glue-out | Proposed handling |
|---|---|---|---|---|---|
| G1 | 14.19 (1st image) | F0-partialled primary balance, cued, acute LOSES its star | -0.249 * | -0.258 [-0.550, +0.006] n.s.; PS92 +0.09, PS93 -0.13, PS94 -0.68, PS95 -0.31 | Live says "partialling it out (conservative) leaves cued acute -0.23" (no star stated, "leaves" reads as "survives"). Reword: "leaves a cued acute shift of -0.26 in 3 / 4 animals (CI just includes 0)". CI-edge case like caption Q2 (11-12: borderline, not claimed). **Needs Priya's OK.** |
| G2 | 12.19 (1st image) | primary balance -> offset slope shrinks | -0.233 mm per log2 unit, p 0.003 | -0.175 [-0.317, -0.033], p 0.016 (LMM) | Still significant; live says -0.32, p < 0.001. Number change, conclusion holds. |
| G3 | 12.17 | cross-animal transfer of the total ITI offset weakens | LOAO 0.379 | LOAO 0.255 (days 0.167; shuffle p 0.001) | Live 0.34. Still beats days; wording "only the total ITI offset transfers" holds. |
| G4 | 12.20 | ITI head-to-head orofacial term: glue-in rebuild had LOST significance, glue-out restores it | -0.018, p 0.054 | -0.028, p 0.0016 (LMM) | Live "both p < 0.01" holds again. |
| G5 | 12.21 (not in C6, flagged because it is the same family) | ITI original-decoder offset, PS93 subacute flips sign | +0.036 | -0.023 | Live "4 / 4 animals every epoch in every variant" -> "3 / 4 subacute (PS93 ~0)". |

Everything else in C6 moves by <= 0.04 (lateral ratio, angle LMM, raw imaging control: unchanged to 0.01; primary
balance: acute cued -0.32 -> -0.35, i.e. slightly LARGER with SSs on its glue-free pixels).

### 0.2 Caused by the rebuild, not the glue (glue-in and glue-out agree; the live page is older)

| # | Fig | Live wording | Rebuild (glue-out) | Wording change |
|---|---|---|---|---|
| R1 | 14.14 | "subacute -0.11 at order 0 only" (significant) | subacute order 0 -0.10 n.s.; significant at orders 4 (-0.21) and 5 (-1.08, mostly PS94) | yes: "subacute: only later orders (4-5) below pre" |
| R2 | 14.15 | cued, same angle + order: "acute below pre ... tjM1 licks 0-2"; "subacute mildly below early" | tjM1: no acute order below pre; subacute: no order below pre (ALM ABOVE pre at 0-3) | yes (drop both clauses). Acute SSp-m / SSp-n / pool 0-8 and chronic overshoot (tjM1, pool 1-6; ALM 0-6; now also SSp-n 2-6) hold |
| R3 | 14.10 | SSp-m contra side "+0.06 / +0.10 / +0.12 (chronic only)" | +0.07 / +0.14* / +0.13* | yes: "subacute and chronic" |
| R4 | 14.11 | ITI "distance x epoch significant for tjM1 and ALM only" | no pair significant (LR p 0.17-0.98) | yes: "ITI: no distance x epoch effect" |
| R5 | 12.20 | "non-orofacial adds nothing (cued +0.017, the opposite sign)" | cued non-orofacial term +0.033, p 0.0003 (opposite sign; glue-in +0.031, p 0.003) | yes: it enters with the OPPOSITE sign and is significant (a suppressor, not "nothing"); ITI n.s. |
| R6 | 12.19 (2nd image) | cued "every pair <= -0.53" | tjM1 PS92 r -0.25 (glue-in -0.27) | yes: "every pair <= -0.64 except tjM1 in PS92 (-0.25)" |
| R7 | 14.18 S13 | gain ratio "window free 0.69, same 0.2-0.35 s chosen in 4/4 folds" | nested gain ratio, cued first lick 0.49 (glue-in 0.50); raw difference 0.69; folds choose different windows | yes, the window-selection supplement needs rewriting (not a C6-listed figure, but the primary marker's justification cites it) |
| R8 | 14.18 / 14.17 | numbers for the uncapped first-lick set (image "every first post-cue lick", cued 0.65; r 0.77 / 0.87 / 0.94 / 0.74), the lateral-position reference, the Gaussian reference (ITI 0.10), the kinematics-matched reference (0.63) | NOT produced in the rebuild (no `WFIELD_BAL_FULL` / `WFIELD_BAL_REF` / `WFIELD_BAL_KIN` step; `recprim_*_full.csv` absent) | TBD (section 3) |

### 0.3 Not changed (key claims hold in glue-out)

- Primary balance: acute cued shift, 4 / 4 animals negative at every order 0-4 (14.14); back to ~0 chronically.
- Recovery marker beats days since stroke in an unseen animal: cued 0.665 vs 0.226; ITI first after quiet -> ITI angle
  0.437 vs 0.167 (PS95 r 0.04: still "3 / 4, PS95 no"); within a known animal the calendar wins (blocked 0.54 vs 0.75).
- Far-spout contralesional shift of the contra / ipsi ratio, 4 / 4 every epoch (14.9 / 14.11); far > close (cued p <= 5e-5).
- F0 drops acutely (wrong sign for an artifact) and undoing it makes the shift larger; 415 nm balance ~0.
- Target-aware reference removes the cued acute deficit; ITI keeps a state shift (14.18 S3 / S4).

### 0.4 D1 (state space): NOT re-run

`M:/MICROSCOPE/Priya/Widefield/state_space_recovery/` has no output newer than 11-09 19:43; its Allen binned cache was
moved to `_superseded_glue_20261109/cache` at 11-09 17:10 and `cache/` holds only `allen_area_names.json`. The v20 queue
has no state-space step. Every D1 response-size / balance / loop number stays `{{TBD after re-run}}`; only the area count
could be filled (45, section 4.1). Producing them needs a state-space re-run on the glue-free extracts (cache refill from
107 ITI extracts + 6-8 scripts; well over 2 min; NOT launched).

---

## 1. Per figure: C6 hemisphere-balance family

Unless stated: cued = first post-cue lick (LP video licks, cam4) on camera working trials (contact + incomplete); ITI =
LP video licks; per-lick baseline -1.0..-0.5 s; change vs the animal's own pre; "*" = boot_delta CI excludes 0.

### 1.1 Fig 14.9 (`lick_lateral_ratio_spout`), contra / ipsi ratio per spout

Source: `lateral_ratio_tests.csv`, columns `delta`, `lo`, `hi`, `sig`, `d_PS92..d_PS95` (set = cued, window =
`logr_0-1` or `logr_0-0.5`); script `scripts/lick_lateral_ratio.py`. Glue effect: none at caption precision (0-0.5 s table,
210 cells: 0 star changes, max |change| 0.04, all in SSp-n; glue-in log `l_lateral_ratio.log` 11-07 vs `l_lateral_ratio_g.log`).

Per animal, orofacial pool, far contra, 0-1.0 s (log2 contra / ipsi, change vs own pre):

| epoch | PS92 | PS93 | PS94 | PS95 | pooled |
|---|---|---|---|---|---|
| acute | +0.60 | +0.14 | +0.85 | +0.78 | +0.59 * |
| subacute | +0.42 | +0.25 | +0.62 | +0.22 | +0.37 * |
| chronic | +0.24 | +0.33 | +0.57 | +0.21 | +0.33 * |

Other caption numbers (old live -> new glue-out):

| Caption item | live | glue-out |
|---|---|---|
| pool far contra (0-1 s) a / s / c, k/4 | +0.58 / +0.33 / +0.32, 4/4 every epoch | +0.59* / +0.37* / +0.33*, 4/4 every epoch |
| median CI width per spout (0-1 s): pool / tjM1 / ALM / SSp-n | 0.24 / 0.24 / 0.17 / 0.50 | 0.27 / 0.29 / 0.18 / 0.50 |
| median abs z: pool / ALM / tjM1 | 2.3 / 2.8 / 1.9 | 2.2 / 3.8 / 2.0 (z approximated as delta / (CI width / 3.92); live definition not in the script output) |
| pre tjM1 close ipsi / close contra / far contra (range over animals) | +0.36..+0.52 / -0.37..-0.73 / -0.36..-0.69 | +0.46..+0.58 / -0.24..-0.94 / -0.29..-0.84 |
| pre SSp-n, largest ratio | up to ~2.4x | up to 2.6x (PS94 close contra -1.38 log2) |
| 0-0.5 s far contra tjM1 / SSp-n | +0.54 / +0.32 / +0.32; +0.88 / +0.55 / +0.55 | +0.60* / +0.41* / +0.34* (tjM1 acute +0.66 / +0.21 / +0.86 / +0.65); +0.87* / +0.59* / +0.55* |
| 0-0.5 s far mid tjM1 | +0.25 / +0.12 / +0.17 | +0.30* / +0.19* / +0.18* |
| 0-0.5 s far ipsi tjM1 / SSp-n / ALM | +0.07 / +0.19 / +0.20; +0.10 / +0.32 / +0.42; +0.16 / +0.20 / +0.22 | +0.12 / +0.25* / +0.20 (n.s.); +0.08 / +0.31* / +0.40*; +0.21* / +0.26* / +0.25* |
| 0-0.5 s close ipsi SSp-n acute | -0.20 | -0.22 * |

Conclusion wording: unchanged ("far spouts shift toward the contralesional hemisphere, far ipsi too subacutely /
chronically, close ipsi does not"). Note tjM1 far ipsi chronic is n.s. (pool far ipsi 0-0.5 s +0.10* / +0.27* / +0.27*).

Proposed sentence (replaces the numbers in the current caption; definition per C6 of the draft): "... far contra
+0.59 / +0.37 / +0.33 (orofacial pool, 0-1.0 s; acute / subacute / chronic, positive in 4 / 4 animals every epoch) ...
Post, cued, 0-0.5 s: far contra tjM1 +0.60 / +0.41 / +0.34 (4/4 every epoch), SSp-n +0.87 / +0.59 / +0.55; far mid
+0.30 / +0.19 / +0.18; far ipsi +0.12 / +0.25 / +0.20 (SSp-n +0.08 / +0.31 / +0.40, ALM +0.21 / +0.26 / +0.25) while
close ipsi stays near 0 (SSp-n -0.22 acute) ..."

### 1.2 Fig 14.10 (`lick_lateral_ratio_side_auc`), by spout side, 0-1.0 s

Source as 1.1 (spout = side_ipsi / side_mid / side_contra); pre levels = `level` (epoch = pre).

Per animal, tjM1, contra side (change vs own pre):

| epoch | PS92 | PS93 | PS94 | PS95 | pooled |
|---|---|---|---|---|---|
| acute | +0.31 | -0.04 | +0.42 | +0.28 | +0.24 * |
| subacute | +0.22 | +0.11 | +0.48 | +0.17 | +0.25 * |
| chronic | +0.10 | +0.12 | +0.43 | +0.24 | +0.22 * |

| Caption item | live | glue-out |
|---|---|---|
| pre cued tjM1 ipsi / mid / contra side | +0.48 / -0.04 / -0.52 | +0.52 / -0.04 / -0.53 |
| pre cued SSp-n ipsi / mid / contra side | +0.68 / -0.14 / -1.03 | +0.72 / -0.11 / -1.02 |
| contra side tjM1 / SSp-n / ALM | +0.20 x3; +0.43 / +0.45 / +0.47; +0.19 / +0.15 / +0.16 (all significant) | +0.24* / +0.25* / +0.22*; +0.43* / +0.47* / +0.48*; +0.22* / +0.21* / +0.18* |
| contra side SSp-m | +0.06 / +0.10 / +0.12 (chronic only) | +0.07 / +0.14* / +0.13* (**R3**: subacute and chronic) |
| per animal tjM1 acute | +0.23 / -0.02 / +0.33 / +0.28 (PS93 ~0) | +0.31 / -0.04 / +0.42 / +0.28 (PS93 ~0) |
| mid side tjM1 | +0.08 / +0.09 / +0.08 | +0.12* / +0.14* / +0.09* |
| ipsi side tjM1; SSp-m / SSp-n acute | -0.10 / +0.05 / -0.02; -0.16 / -0.22 | -0.09 / +0.05 / -0.03; -0.17* / -0.25* |
| ITI contra side tjM1 acute; SSp-n | +0.29; +0.76 / +0.68 / +0.58 | +0.42*; +0.70* / +0.58* / +0.50 (chronic n.s.) |

Conclusion: unchanged except R3. Glue: none (<= 0.01).

### 1.3 Fig 14.11 (`lick_lateral_ratio_rings`), close vs far ring, 0-1.0 s

Source as 1.1 (spout = ring_far / ring_close / ring_diff); `lateral_ratio_models.csv` (test = "distance x epoch").

Per animal, tjM1, far ring:

| epoch | PS92 | PS93 | PS94 | PS95 | pooled |
|---|---|---|---|---|---|
| acute | +0.35 | +0.11 | +0.31 | +0.32 | +0.27 * |
| subacute | +0.29 | +0.25 | +0.24 | +0.10 | +0.22 * |
| chronic | +0.04 | +0.16 | +0.24 | +0.28 | +0.18 * |

| Caption item | live | glue-out |
|---|---|---|
| far ring tjM1 / SSp-n / SSp-m / ALM | +0.24 / +0.17 / +0.17; +0.37 / +0.32 / +0.40; +0.16 / +0.13 / +0.13; +0.23 / +0.14 / +0.19 | +0.27* / +0.22* / +0.18*; +0.34* / +0.30* / +0.40*; +0.17* / +0.15* / +0.13*; +0.27* / +0.19* / +0.21* |
| close ring tjM1 / SSp-m / SSp-n / ALM | 0.00 / +0.07 / +0.01; -0.08 / 0.00 / 0.00; 0.00 / +0.19 / +0.23; +0.06 / +0.08 / +0.06 | +0.03 / +0.10* / +0.03; -0.07* / +0.02 / 0.00; -0.02 / +0.16 / +0.22*; +0.09* / +0.12* / +0.09* |
| far minus close, cued: tjM1 / SSp-m / SSp-n / ALM / pool | +0.24* / +0.10* / +0.16*; +0.24* / +0.12* / +0.13*; +0.37* / +0.13 / +0.17*; +0.17* / +0.06* / +0.12*; +0.26* / +0.11* / +0.16* | +0.25* / +0.12* / +0.16*; +0.24* / +0.13* / +0.13*; +0.36* / +0.14* / +0.18*; +0.18* / +0.07* / +0.12*; +0.26* / +0.12* / +0.16* |
| cue-aligned far minus close | "subacute / chronic significant in every pair" | every pair except SSp-n subacute (+0.08 n.s.) |
| distance x epoch, cued (LR) | p < 1e-4 every pair | p <= 4.7e-5 every pair |
| ITI acute far / close tjM1 | +0.48 / +0.30 | +0.44* / +0.32 (n.s.) |
| ITI distance x epoch | significant for tjM1 and ALM only | none (p 0.17-0.98) (**R4**) |

Conclusion: unchanged for cued; R4 for ITI. Glue: none.

### 1.4 Fig 14.12, by side over time and the 0-0.5 s version

Source as 1.1 (window `logr_0-0.5`). Contra side tjM1 +0.24 / +0.22 / +0.22 -> +0.30* / +0.28* / +0.24* (acute per animal
+0.41 / +0.07 / +0.46 / +0.25, 4/4); SSp-n +0.44 / +0.40 / +0.45 -> +0.44* / +0.43* / +0.45*; ipsi side "~0" -> tjM1
+0.02 / +0.13* / +0.04, SSp-n -0.09 / +0.13* / +0.20* (wording: "ipsi side small, SSp-n rises chronically"). Glue: none.

### 1.5 Fig 14.13 (`lick_balance_angle`), balance vs lick angle (SUPPLEMENT per D3)

Source: `balance_angle_lmm.csv` (columns `slope_pre`, `slope_change_post`, `shift_post_same_angle`, model = acute /
subacute / chronic; LMM, sessions x spouts as observations, anti-conservative) and `balance_angle_animals.csv` (`slope`);
script `scripts/lick_balance_angle_lmm.py`. Glue effect <= 0.008 on every caption number (`l_bal_angle_lmm_r2.log` vs `_g`).

Per animal, cued response ratio, tjM1 slope per deg (pre -> acute / subacute / chronic):

| animal | pre | acute | subacute | chronic | live |
|---|---|---|---|---|---|
| PS92 | -0.024 | -0.023 | -0.025 | -0.021 | -0.023 -> -0.022 / -0.024 / -0.020 |
| PS93 | -0.029 | -0.031 | -0.026 | -0.016 | -0.029 -> -0.033 / -0.028 / -0.016 |
| PS94 | -0.066 | -0.024 | -0.038 | -0.041 | -0.064 -> -0.023 / -0.036 / -0.040 |
| PS95 | -0.040 | -0.021 | -0.023 | -0.030 | -0.037 -> -0.020 / -0.023 / -0.028 |

| Caption item (cued, response ratio) | live | glue-out |
|---|---|---|
| pre slope tjM1 / SSp-n (per deg) | -0.033 / -0.056 | -0.034 / -0.057 |
| slope change a / s / c, tjM1 / SSp-n / SSp-m | +0.011 / +0.007 / +0.009; +0.020 / +0.014 / +0.014; +0.010 / +0.006 / +0.009 | +0.010 / +0.008 / +0.010; +0.021 / +0.015 / +0.014; +0.011 / +0.007 / +0.009 (all p <= 0.002) |
| shift at same angle tjM1 / SSp-m / SSp-n | -0.22 / -0.14 / -0.06; -0.37 / -0.24 / -0.12; -0.39 / -0.15 / +0.04 | -0.20 / -0.12 / -0.05; -0.38 / -0.24 / -0.12; -0.41 / -0.18 / +0.04 |
| pool slope change; shift | +0.011 / +0.007 / +0.009; -0.30 / -0.17 / -0.04 | +0.011 / +0.008 / +0.009; -0.30 / -0.16 / -0.03 |
| later licks (decoder window) tjM1 acute; chronic tjM1 / SSp-n / ALM | -4.9; +1.33 / +2.16 / +1.55 | -4.83; +1.27 / +2.07 / +1.48 |
| later licks per animal tjM1 pre -> chronic | -0.99 -> +0.94, 7.6 -> 10.8, -7.0 -> -9.3, -2.2 -> +0.2 (3/4 up) | -0.84 -> +1.03, 7.63 -> 10.87, -7.30 -> -9.45, -1.58 -> +0.57 (3/4 up) |
| ITI pre slope tjM1; ITI decoder-window shift tjM1; ITI response-ratio shift ALM / tjM1 | -0.010; -1.6; +0.14 / +0.15 | -0.010; -1.55 (post); +0.07 / +0.10 (post model; n.s.) |

Conclusion: unchanged.

### 1.6 Fig 14.14 (`lick_balance_order_primary`), PRIMARY balance over lick order

Source: `balorder_primary.csv` (columns `delta`, `lo`, `hi`, `sig`, `d_PS92..d_PS95`; `p` is empty, so `sig` = CI excludes 0,
no BH); script `scripts/lick_balance_order_primary.py`. Hemispheric sign (+ = contralesional larger).

Per animal, cued, acute (change vs own pre):

| order | PS92 | PS93 | PS94 | PS95 | pooled | glue-in pooled |
|---|---|---|---|---|---|---|
| 0 | -0.10 | -0.29 | -0.69 | -0.33 | -0.35 * | -0.32 * |
| 1 | -0.15 | -0.27 | -0.76 | -0.24 | -0.36 * | -0.32 * |
| 2 | -0.23 | -0.20 | -1.03 | -0.21 | -0.42 * | -0.38 * |
| 3 | -0.37 | -0.23 | -0.92 | -0.12 | -0.41 * | -0.38 * |
| 4 | -0.44 | -0.35 | -0.99 | -0.08 | -0.46 * | -0.44 * |

Subacute: -0.10 / -0.07 / -0.07 / -0.08 / -0.21* at orders 0-4 (order 5 -1.08*, PS94 -3.55). Chronic: -0.03 / +0.03 / +0.03 /
+0.06 / +0.02 (none significant). ITI: acute bout start -0.26* (-0.24 / -0.18 / -0.65 / +0.01); later orders and epochs n.s.

| Caption item | live | glue-out |
|---|---|---|
| cued acute orders 0-4 | -0.34 / -0.33 / -0.38 / -0.37 / -0.43 (all *) | -0.35* / -0.36* / -0.42* / -0.41* / -0.46* |
| PS94 range | -0.66 to -0.98 | -0.69 to -1.03 |
| subacute | -0.11 at order 0 only (significant) | order 0 -0.10 n.s.; significant at order 4 (-0.21) and 5 (**R1**) |
| chronic | ~0 every order | ~0 every order |
| ITI acute bout start | -0.28 * | -0.26 * |

Glue: makes the acute cued shift ~10 % larger (SSs on its glue-free pixels). Proposed sentence: "Cued: acute -0.35 /
-0.36 / -0.42 / -0.41 / -0.46 at orders 0-4 (all CIs exclude 0; PS94 largest, -0.69 to -1.03); subacute below pre only at
later orders (order 4 -0.21); chronic ~0 at every order. ITI: acute bout start -0.26 (CI excludes 0); other orders and
epochs ~0." Add the A10 baseline clause per the draft.

### 1.7 Fig 14.15 (`lick_balance_order`), RAW 0-0.2 s decoder-window difference (SUPPLEMENT)

Source: `balance_order_sig.csv` (`sig_bh` per set x ref x pair x epoch x lag; ref `angord` = same angle AND order);
script `scripts/lick_balance_order.py`. Glue effect: only ITI acute SSp-n / pool order lists shrink (e.g. pool ITI acute
[0, 2, 3, 4, 5, 8] -> [0, 3, 8], angle ref); cued same-angle-and-order lists identical except chronic SSp-m/pool by one order.

Significant orders (BH), cued, same angle and order (glue-out): acute below pre at 0-8 in SSp-m, SSp-n, pool; tjM1 and
ALM none. Subacute: ALM ABOVE pre 0-3, nothing below. Chronic ABOVE pre: tjM1 1-6, pool 1-6, ALM 0-6, SSp-n 2-6. ITI, same
angle and order: acute below pre SSp-m 0, 2-8; SSp-n 0, 2, 3; pool 0, 2, 3, 8.

Wording change R2: drop "tjM1 licks 0-2" and "subacute mildly below early"; add SSp-n to the chronic overshoot. (The
convergence "by lick 9+" holds: no order 9 significant acutely.)

### 1.8 Fig 14.17 (`lick_balance_explainer`), worked example PS94

Source: `l_bal_explainer_g.log` (gains; the figure's numbers), `recprim_sessions_inside.csv`, `recprim_cv_inside.csv`,
`recprim_sessions.csv` (`n_licks`); balance = log2(gain_c) - log2(gain_i) computed from the logged gains.

| Item | live | glue-in | glue-out |
|---|---|---|---|
| acute PS94_0817 gain contra / ipsi -> balance | 2.08 / 4.26 -> -1.04 | 2.28 / 4.61 -> -1.01 | 2.19 / 4.72 -> -1.11 |
| chronic PS94_1005 gain contra / ipsi -> balance | 0.98 / 1.13 -> -0.20 | 1.01 / 1.17 -> -0.21 | 1.01 / 1.18 -> -0.23 |
| extrapolated share of first licks (outside the pre range) acute / chronic | 33 % / 12-16 % | - | cued 33 % / 12 % (PS94 58 % acute); ITI 34 % / 16 % (not mask-dependent) |
| inside-range only: cued unseen-animal R2 vs days | 0.53 vs 0.23 | 0.44 | 0.48 vs 0.23 |
| inside-range acute balance vs all licks | -0.26 vs -0.38 | - | -0.24* vs -0.35* (per animal inside: -0.09 / -0.28 / -0.37 / -0.24) |
| inside-range ITI cross-animal | 0.07 (fails); within 3/4 | 0.03 | 0.12 (still fails vs days 0.17); within r 0.60 / 0.83 / 0.84 / 0.00 (3/4) |
| Gaussian same-place reference cued / ITI | 0.37 / 0.10 | - | cued 0.36 (`balcontact_cv.csv`, gauss_all__all); ITI TBD |
| lateral position reference | equivalent | - | TBD (not run in the rebuild) |

Conclusion: unchanged.

### 1.9 Fig 14.18 (`lick_recovery_primary`), PRIMARY recovery marker

Source: `recprim_cv.csv` / `recprim_models.csv` / `recprim_sessions.csv` (+ `_inside`), `l_recov_primary_g.log`; script
`scripts/lick_recovery_primary.py`. NOTE: the rebuild ran only the DEFAULT (capped-like, median 132.5 first licks per
session, min 89) set = the live "S1" version. The live headline image (every first post-cue lick, `WFIELD_BAL_FULL=1`) has no
rebuild run (R8).

Per animal (post sessions), r of marker with the recovery axis (r of log(day) in brackets):

| panel | PS92 | PS93 | PS94 | PS95 | live (S1 capped) |
|---|---|---|---|---|---|
| cued first lick -> far-contra aim | 0.73 (0.92) | 0.89 (0.89) | 0.94 (0.87) | 0.78 (0.93) | 0.71 / 0.82 / 0.94 / 0.75 |
| ITI first after quiet -> ITI angle | 0.69 (0.91) | 0.83 (0.76) | 0.91 (0.64) | 0.04 (0.68) | 0.58 / 0.80 / 0.91 / 0.10 |
| ITI first after quiet -> cued aim | 0.67 | 0.87 | 0.86 | 0.33 | 0.55 / 0.78 / 0.85 / 0.37 |

Per animal, marker change vs own pre (cued first lick): acute -0.10 / -0.29 / -0.69 / -0.33 -> -0.35* [-0.58, -0.16];
subacute +0.08 / -0.04 / -0.20 / -0.25 -> -0.10 n.s.; chronic +0.05 / +0.02 / -0.19 / -0.01 -> -0.03 n.s. ITI first
after quiet: acute -0.27 / -0.19 / -0.79 / -0.04 -> -0.32* ; subacute / chronic n.s.

| Caption item | live (capped S1 unless noted) | glue-in | glue-out |
|---|---|---|---|
| cued unseen animal (LOAO) R2; days | 0.70 (headline image 0.65); 0.23 | 0.656 | 0.665 (shuffle p < 0.001); 0.226 |
| cued LOSO unknown / known / known + days | 0.70 / 0.70 / 0.85 | 0.669 / 0.686 / 0.853 | 0.680 / 0.701 / 0.854 (days 0.479 / 0.831) |
| cued blocked (within animal) | 0.55 (days 0.75, both 0.76) | 0.526 / 0.763 | 0.543 (days 0.754, both 0.763) |
| cued within-animal slope, log(day); per-animal spline (LMM) | +12.6 deg per log2 unit (p 0.001); +13.7 (p 0.002) | +13.6 (p 0.0003); +17.5 | +13.2 (p 0.0003); +18.2 (p < 1e-4) |
| ITI -> ITI angle LOAO; days; blocked | 0.42; 0.17; 0.14 (days 0.43) | 0.387 | 0.437; 0.167; 0.167 (days 0.430) |
| ITI -> ITI angle slope (LMM) | +5.4 (p 0.002) | +5.73 (p 0.003) | +5.63 (p 0.003); spline +5.34 (p 0.007) |
| ITI -> cued aim LOAO; slope | 0.51; +11.0 (p 5e-7) | 0.495 | 0.567; +12.5 (p < 1e-4) |
| inside pre range: cued / ITI / ITI -> aim | 0.53 / 0.07 / 0.06 | 0.44 / 0.03 / 0.03 | 0.48 / 0.12 / 0.17 |
| inside: cued per-animal r | 0.70 / 0.82 / 0.88 / 0.72 | 0.67 / 0.89 / 0.83 / 0.72 | 0.71 / 0.89 / 0.86 / 0.74 (LMM p 0.019) |
| inside: ITI within-animal r | 0.50 / 0.81 / 0.84 / 0.06 (p 0.03) | - | 0.60 / 0.83 / 0.84 / 0.00 (LMM p 0.08) |
| lick-count checks (>= 20 licks / count-weighted) cued; ITI | 0.70 / 0.68; 0.43 / 0.41 | - | 0.665 / 0.656; 0.457 / 0.411 |
| next-session prediction | none (p 0.24-0.80) | - | none (p 0.23-0.96) |
| S2 contact vs miss: acute balance contact / miss | -0.12 / -0.13, -0.29 / -0.26, -0.67 / -1.08, -0.23 / -0.22 | - | -0.08 / -0.15, -0.32 / -0.28, -0.70 / -1.16, -0.27 / -0.30 (`l_bal_contact_g.log`, line_all) |
| S2 recovery r contact; miss; LOAO | 0.73 / 0.83 / 0.91 / 0.73 (0.66); 0.44 / 0.91 / 0.97 / 0.68 (0.66) | - | 0.75 / 0.91 / 0.90 / 0.74 (0.62); 0.63 / 0.91 / 0.97 / 0.70 (0.65); strict-miss PS92 / PS95 r -0.11 / 0.08 |
| S3 target-aware cued acute (per animal) | -0.34 -> +0.01 (+0.09 / -0.08 / +0.09 / -0.04) | - | -0.36* -> +0.01 n.s. (+0.13 / -0.12 / +0.11 / -0.08) |
| S3 target-aware cued LOAO; landing error alone | -0.40; 0.79 | - | -0.43; 0.79 |
| S3 ITI target-aware acute; LOAO | -0.32 (4/4); 0.32-0.37 | - | -0.29* (4/4); 0.36-0.38 |
| S4 cued within-spout acute; sub / chr | +0.02 [-0.07, 0.09] (1/4 neg); +0.13 / +0.16 (4/4) | +0.05 / +0.15 / +0.15 | +0.03 [-0.08, 0.12] (1/4 neg); +0.15* / +0.16* (4/4) |
| S4 target-blind on same licks; ITI within-spout acute; ITI decoder offset | -0.15 (4/4); -0.18 (4/4); +0.29 / +0.19 / +0.22 mm | -0.12; -0.17; +0.26 / +0.19 / +0.20 | -0.15* (4/4); -0.18* (4/4); +0.27* / +0.20* / +0.22* mm (4/4) (`balmatch_tests.csv`) |

Conclusion: unchanged. Proposed headline: "... estimates how far an unseen animal has recovered - cued R2 0.67 vs 0.23 for
days since stroke (0.48 with only licks inside the pre range); for ITI licks R2 0.44 vs 0.17 (within animals 3 of 4;
PS95 r 0.04) ..." Uncapped-set numbers (live headline 0.65) stay TBD (R8).

### 1.10 Fig 14.19 (`lick_balance_imgctrl`), imaging control

**First image (primary balance)** - source `balimg_primary_change.csv` (`delta`, `lo`, `hi`, `sig`, `d_PS9x`; measure
`f0_470`, `marker`, `marker_dF` (= F0 scaling undone), `marker_part` (F0 partialled), `h415_rdiff_n`, `neu_rdiff_n`),
`balimg_primary_link.csv` (`loao`, `r_PS9x`, `p_w`); script `scripts/lick_balance_imaging_control_primary.py`.

Per animal, cued first lick, acute (change vs own pre):

| measure | PS92 | PS93 | PS94 | PS95 | pooled glue-out | pooled glue-in |
|---|---|---|---|---|---|---|
| F0 asymmetry (470) | -0.21 | -0.18 | -0.01 | -0.02 | -0.105 * | -0.098 * |
| balance (marker) | -0.10 | -0.29 | -0.69 | -0.33 | -0.354 * | -0.323 * |
| F0 undone (marker_dF) | -0.31 | -0.47 | -0.70 | -0.36 | -0.459 * | -0.421 * |
| F0 partialled (marker_part) | +0.09 | -0.13 | -0.68 | -0.31 | -0.258 [-0.550, +0.006] n.s. | -0.249 * |

| Caption item | live | glue-out |
|---|---|---|
| F0 acute; PS92 / PS93 | -0.105; -0.21 / -0.18 | -0.105*; -0.21 / -0.18 |
| undoing F0: cued acute | -0.34 -> -0.44 | -0.35 -> -0.46 |
| 415 nm balance vs neural (raw difference) cued acute | -0.006 vs -0.21 | -0.007 vs -0.216* |
| within-animal r(marker, F0) cued | +0.35-0.67 | post sessions 0.64-0.74 (all sessions 0.33-0.62) |
| F0 partialled: cued acute; PS94 / PS95 | -0.23; -0.64 / -0.26 | -0.26 n.s. (**G1**); -0.68 / -0.31 |
| F0 partialled recovery link LOAO: cued (p); ITI (p) | 0.42 (p 0.015); 0.27 (p 0.004) | 0.39 (within-animal LMM p 0.005); 0.28 (p 0.010) |
| F0 alone predicts recovery | no | no (LOAO -0.18 cued, -0.29 ITI) |

**Second image (raw ratio, SUPPLEMENT)** - source `balimg_change.csv` (set cued, spout side_contra, pair tjM1),
`balimg_models.csv`; script `scripts/lick_balance_imaging_control.py`. Glue: identical to 0.001 (glue-in `l_balimg.log`).

Per animal, cued contra side, tjM1, neural balance:

| epoch | PS92 | PS93 | PS94 | PS95 | pooled |
|---|---|---|---|---|---|
| acute | +0.31 | -0.04 | +0.42 | +0.28 | +0.24 * |
| subacute | +0.22 | +0.11 | +0.48 | +0.17 | +0.25 * |
| chronic | +0.10 | +0.12 | +0.43 | +0.24 | +0.22 * |

| Caption item | live | glue-out |
|---|---|---|
| neural acute / chronic | +0.20 / +0.20 | +0.24* / +0.22* |
| per animal acute; chronic | +0.23 / -0.02 / +0.33 / +0.28; +0.08 / +0.14 / +0.36 / +0.23 | +0.31 / -0.04 / +0.42 / +0.28; +0.10 / +0.12 / +0.43 / +0.24 |
| F0 chronic; acute PS92 / PS93 | -0.02 (all four); -0.39 / -0.15 | -0.02* (all four negative); -0.39 / -0.14 |
| F0 undone acute; chronic | +0.08 n.s.; +0.18* | +0.11 n.s.; +0.20* |
| 415 nm balance | -0.005 to -0.018 n.s. | -0.005 to -0.018 n.s. |
| clean pixels | +0.22 / +0.21 / +0.21 | +0.25* / +0.25* / +0.23* |
| uncorrected 470 | +0.09 n.s. / +0.15* / +0.12* | +0.09 n.s. / +0.14* / +0.12* |
| adjusted for F0 470 + F0 415 + 415 balance (LMM) | +0.15 / +0.20 / +0.20 | +0.18 / +0.23 / +0.21 |

Conclusion: unchanged except G1 (first image, partialled).

### 1.11 Fig 12.18 (`lick_balance_reference`), balance as offset covariate (SUPPLEMENT measure)

Source: `l_bal_reference_g.log` (share table; `balref_explained.csv` holds the same); script `scripts/lick_balance_reference.py`.
Areas = ALM, tjM1, SSp-m, SSp-n only (no SSs), so the glue effect is small.

| ITI share explained a / s / c | live | glue-in | glue-out |
|---|---|---|---|
| four balances | 0.43 / 0.51 / 0.34 | 0.41 / 0.55 / 0.36 | 0.42 / 0.52 / 0.35 |
| one mean balance | 0.39 / 0.43 / 0.34 | 0.35 / 0.41 / 0.40 | 0.36 / 0.37 / 0.39 |
| SSp-n alone | 0.44 / 0.54 / 0.36 | 0.43 / 0.58 / 0.41 | 0.43 / 0.53 / 0.40 |
| SSp-m / tjM1 / ALM | 0.33 / 0.35 / 0.31; 0.29 / 0.26 / 0.28; 0.05 / 0.13 / 0.09 | - | 0.32 / 0.30 / 0.36; 0.23 / 0.16 / 0.29; 0.01 / 0.10 / 0.14 |
| per animal (mean balance) acute; chronic | 0.88 / 0.70 / 0.57 / 0.01; 0.89 / -0.05 / 0.46 / 0.03 | 0.45 / 0.88 / 0.61 / 0.01; 1.37 / -0.06 / 0.58 / 0.01 | 0.46 / 0.88 / 0.61 / 0.01; 1.00 / -0.10 / 0.53 / 0.02 |
| cued acute: four / mean | 0.60 / 0.86 | 0.61 / 0.81 | 0.59 / 0.77 |

Offset after the four-balance reference (live +0.21 / +0.11 / +0.18 mm): glue-out change_ref +0.20 / +0.11 / +0.17
(log `change_ref`, pooled). Conclusion: unchanged ("about half the ITI offset behaves like a uniform shift; SSp-n carries
most, ALM almost none").

### 1.12 Fig 12.19 (`lick_balance_offset_primary` + `lick_balance_offset_lmm`)

Source: `l_bal_offset_primary_g.log`; `baloff_models.csv` / `l_bal_offset_lmm_g.log`.

First image (primary balance; LMM slope mm per log2 unit with log(day)):

| panel | live | glue-in | glue-out |
|---|---|---|---|
| cued r per animal | -0.70 / -0.80 / -0.92 / -0.58 | -0.68 / -0.83 / -0.90 / -0.72 | -0.70 / -0.87 / -0.90 / -0.74 |
| cued slope (p) | -0.32 (p < 0.001) | -0.233 (p 0.003) | -0.175 [-0.317, -0.033] (p 0.016) (**G2**) |
| cued inside pre range | -0.38 (p < 0.001) | -0.319 (p 0.002) | -0.281 (p 0.003) |
| ITI r per animal; slope (p) | -0.14 / -0.54 / -0.70 / +0.15; (p 0.004) | -0.36 / -0.57 / -0.72 / +0.15; -0.208 (p 0.001) | -0.29 / -0.55 / -0.66 / +0.14; -0.163 (p 0.004) |

Second image (raw 0-0.2 s, SUPPLEMENT): orofacial pool within-animal r cued -0.90 / -0.92 / -0.95 / -0.76 ->
-0.84 / -0.92 / -0.95 / -0.76; ITI -0.52 / -0.52 / -0.70 / +0.09 -> -0.20 / -0.66 / -0.73 / +0.01; slope cued -0.032 ->
-0.024, ITI -0.021 -> -0.018 (p 0.006). "Every pair <= -0.53" fails for tjM1 PS92 (-0.25; R6).

### 1.13 Fig 12.20 (`lick_balance_regions`)

Source: `l_bal_regions_g.log`, `balreg_models.csv`.

| Item | live | glue-in | glue-out |
|---|---|---|---|
| head-to-head orofacial cued; ITI | -0.037; -0.025 (both p < 0.01) | -0.035 (p < 1e-4); -0.018 (p 0.054) | -0.034 (p < 1e-4); -0.028 (p 0.0016) (G4) |
| head-to-head non-orofacial cued; ITI | +0.017 (opposite sign, "adds nothing"); n.s. | +0.031 (p 0.003); n.s. | +0.033 (p 0.0003); +0.021 (p 0.10) (R5) |
| within-animal corr(oro, non-oro) cued / ITI | 0.82 / 0.67 | 0.82 / 0.65 | 0.83 / 0.71 |
| log-likelihood orofacial alone vs non-orofacial alone (cued) | 107 vs 78 | 92 vs 74 | 93 vs 76 |
| non-orofacial pool r cued per animal | -0.51 to -0.91 | - | -0.48 / -0.79 / -0.89 / -0.60 |
| SSp-un alone; added to pool (cued LL) | slope -0.013 (p 0.002), LL 80; 101 vs 107 | -0.0074 (p 0.08) | -0.0057 (p 0.17), LL 79; 89 vs 93 |

Note: the non-orofacial pool lost VISp_left, VISl_left, VISal_left and other glue-only targets (section 4.1); the pool
definition in the caption should say "glue-free areas only". SSp-un is no longer significant alone (rebuild-level).

---

## 2. SSp-n specifically (Priya: how much SSp-n contributes, before vs after the glue fix)

### 2.1 SSp-n difference term of the frozen decoder offset (Fig 12.17)

Source: `offterm_sessions.csv` (columns `total`, `diff_SSp-n`, `diff_all`, `unpaired`, `other`), glue-in
`l_offset_terms_r5.log` (11-09, code 90a5e25f), glue-out `l_offset_terms_g.log`; script
`scripts/lick_offset_terms_recovery.py`. Term = 1/2 (beta_r - beta_l)(dh_r - dh_l) for SSp-n, mm of decoded position.

Per animal, ITI, change vs own pre (glue-out, boot_delta):

| epoch | total offset (92 / 93 / 94 / 95) | SSp-n difference term (92 / 93 / 94 / 95) | pooled total | pooled SSp-n | SSp-n / total per animal |
|---|---|---|---|---|---|
| acute | +0.22 / +0.14 / +0.55 / +0.59 | +0.49 / +0.07 / +0.46 / +0.22 | +0.376 * | +0.310 * | 2.27 / 0.50 / 0.84 / 0.36 |
| subacute | +0.30 / -0.01 / +0.42 / +0.29 | +0.40 / +0.01 / +0.38 / +0.19 | +0.249 * | +0.245 * | 1.38 / -1.24 / 0.91 / 0.65 |
| chronic | +0.14 / +0.07 / +0.53 / +0.35 | +0.25 / +0.00 / +0.43 / +0.08 | +0.272 * | +0.190 * | 1.85 / 0.02 / 0.82 / 0.19 |

Cued (trial), acute: total +0.31 / +0.10 / +0.22 / +0.15 -> +0.194*; SSp-n +0.40 / +0.14 / +0.11 / +0.14 -> +0.196*;
subacute SSp-n +0.086*; chronic -0.027 n.s.

Glue-in vs glue-out, pooled epoch means (the logs print pre-referenced means, not boot_delta; same quantity both runs):

| set, epoch | total in -> out | SSp-n term in -> out | SSp-n share of total in -> out | "other" areas in -> out |
|---|---|---|---|---|
| ITI acute | +0.359 -> +0.341 | +0.419 -> +0.354 | 117 % -> 104 % | -0.144 -> -0.008 |
| ITI subacute | +0.248 -> +0.264 | +0.280 -> +0.230 | 113 % -> 87 % | -0.090 -> +0.061 |
| ITI chronic | +0.238 -> +0.238 | +0.195 -> +0.165 | 82 % -> 69 % | -0.046 -> +0.042 |
| cued acute | +0.203 -> +0.213 | +0.226 -> +0.208 | 111 % -> 98 % | -0.139 -> -0.085 |

Reading: with glue in, SSp-n "carried more than all" of the ITI offset because large opposite terms from glue-covered
areas (VIS / RSP / AUD) cancelled part of it; with those areas cleaned or dropped, the SSp-n term falls ~15 % and its share
falls to ~70-100 % of the offset. SSp-n is still the largest single contributor. Glue-out also has a nonzero "unpaired"
term (-0.035 / -0.087 / -0.055 ITI) because glue-only areas dropped on ONE side leave their partner unpaired.

Within-animal tracking of recovery by the SSp-n term (r per animal, glue-in -> glue-out; the live values in brackets):
ITI -> ITI angle -0.76 / -0.67 / -0.48 / -0.54 -> same (live -0.66 / -0.72 / -0.51 / -0.45); cued -> aim -0.95 / -0.86 /
-0.87 / -0.71 -> -0.95 / -0.86 / -0.86 / -0.71 (live -0.94 / -0.84 / -0.84 / -0.69). "4 / 4 animals in every set" holds.
Cross-animal: SSp-n term LOAO ITI -> ITI angle 0.21 -> 0.06 (live did not claim cross-animal transfer for components).

### 2.2 SSp-n as a ratio / balance area (glue-out vs glue-in)

- Contra / ipsi ratio (14.9-14.12): SSp-n cells move by <= 0.04 (e.g. 0-0.5 s cued far contra +0.89* / +0.56* / +0.53* ->
  +0.89* / +0.55* / +0.52*; pre ratios move by <= 0.05). The stat-mask rim takes 9 % of SSp-n_left pixels (GLUE_MASK_FIX).
- 12.18 share of the ITI offset explained by SSp-n balance alone: 0.43 / 0.58 / 0.41 -> 0.43 / 0.53 / 0.40.
- 12.19 raw SSp-n balance vs offset, within-animal r: cued -0.90 / -0.77 / -0.96 / -0.81 -> -0.90 / -0.89 / -0.97 / -0.79;
  ITI -0.42 / -0.46 / -0.77 / +0.10 -> -0.39 / -0.41 / -0.78 / 0.00.
- Glue report rows touching SSp-n are all non-significant on both sides (e.g. `l_c2l_restw_g` SSp-n_right chronic -0.03 ->
  +0.02, CI +-1.9).

### 2.3 Not available: SSp-n shares in SVD / LocaNMF

Priya's 11-11 follow-up (per-lick SSp-n R / L offset shares in the SVD and LocaNMF bases via pixel projection) was NOT queued
(RUNNING_LIST 11-11 ~08:12: "Follow-up idea, not queued"). The v20m fix (4b5fa2e1) SKIPS the per-lick SSp-n term in
non-area bases, so `terms_svdF.csv` / `terms_locanmfFM.csv` have no SSp-n term. Needs new code + a run.

---

## 3. C6 TBDs that could NOT be filled

| Item | Why |
|---|---|
| 14.18 headline image (uncapped first licks; live 0.65, r 0.77 / 0.87 / 0.94 / 0.74, LOSO 0.71, pre noise SD 0.078 vs 0.092) | no rebuild run with `WFIELD_BAL_FULL=1` (the `firstlick_full` extract exists; `lick_balance_subsets` + `lick_recovery_primary` would need a run, ~10 min) |
| 14.18 / 14.17 lateral-position reference (live 0.71 / 0.40 / 0.51), Gaussian ITI (0.10), kinematics-matched (0.63) | no `WFIELD_BAL_REF=lateral`, `WFIELD_BAL_REFMETHOD=gauss` or `WFIELD_BAL_KIN=1` step in the rebuild |
| 14.18 (vi) share of the decoder offset explained by the balance (cued 65 %, ITI 5-15 %) | not re-read here (`l_bal_offset_*` gives slopes only); would need the share table of the offset script |
| 14.18 S12 trial-by-trial r, S13 subset search, S15 / S16 per-area LOAO | logs exist (`l_bal_trial_g`, `l_bal_subsets_full_g`, `l_recov_markers_g`) but not tabulated here; S13 moved at the rebuild level (R7) and needs its own rewrite |
| 14.13 "median abs z" | z here is approximated from the CI; the live definition is not in the script output |

---

## 4. Per figure: D1 state-space family (SS Figs 3.3-3.8, 4.5, 4.6, 5.1, 5.2, 7.5-7.10)

### 4.1 Filled: the Allen area count

**45 glue-free Allen areas** (was 62). Source: `cols` of every cohort-1 ITI extract
`<rebuild>/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801/sessions_roi8fr/PS9[2-5]_*.npz`
(written 11-09 19:10, `mask_sig` present = post-fix): 45 non-"cortex" columns in all 107 sessions, one identical set. The
glue-in copies in `_superseded_glue_20261109/...` have 62. The 17 dropped (identical in all four animals): AUDd, AUDp, AUDpo,
TEa (both hemispheres), VISli, VISpl, VISpor (both), VISp_left, VISl_left, VISal_left. SSs, SSp-n, SSp-m, ALM, tjM1 are kept
(SSs on its glue-free pixels only). These are exactly the 17 "glue-only targets" of GLUE_MASK_FIX.

Proposed: "the norm over the 45 Allen areas outside the painted fiber-glue region (glue-only areas not analyzed: AUD, TEa,
VISli / VISpl / VISpor both hemispheres, ipsilesional VISp / VISl / VISal) ..." and in 3.5-3.8 / 7.5 "(45 areas)".

### 4.2 Not filled: every response size, balance and loop number

The state-space scripts have NOT been re-run on the glue-free extracts (section 0.4). Old values (live SS page v14,
`wf_decisions1111/docs/results_pages/state_space/index.html`) that become `{{TBD after re-run}}`, and the script that
regenerates each:

| Fig | Old numbers in the caption | Regenerated by |
|---|---|---|
| 3.3 | chronic +.55 / +.60 / +.49 (close, 4/4), +.69 / +.75 / +.33 (far; 3/4, 3/4 n.s., 4/4); acute far contra -1.57 (0/4) | `state_space_trajectories.py` (`trajectories/resp_size.csv`) |
| 3.4 | chronic +.56 / +.47 / +.41 (close, 4/4), +.47 / +.39 / +.27 (far; 3/4, 3/4 n.s., 3/4) | same |
| 3.5 / 3.6 | chronic percentile shifts +.35 to +.6; acute far contra 10th -.29 / 90th +.15 | `state_space_pertrial.py` |
| 3.7 / 3.8 | size2 +.95 / +.84, baseline noise +.25 / +.26, response noise +.16 / +.09, excess +.17 / +.29, acute excess +.52, subacute excess +.40 | `state_space_noise.py` |
| 4.5 / 4.6 | incomplete - contact response size pre -.31, acute -.64 [-.82, -.39], subacute -.22, chronic -.31; pattern r -.04* / -.09* / -.03* / -.02 | `state_space_contact.py` |
| 5.1 | balance (SSp-m, SSp-n, SSs) by target / on target / matched; in-range acute far contra 98 / 100 / 37 / 78 %; section text far contra +.27 / +.35, +.60 -> +.33 (2/4) | `state_space_balance_cells.py`, `state_space_balance_angle.py` |
| 5.2 | spread pre 1.34 / 1.24 / 2.46 / 1.51 -> chronic 1.26 / 1.12 / 1.76 / 1.33; change -.47 / -.32 / -.27 | `state_space_balance_cells.py` |
| 7.5-7.10 | top-2 PCs 93 % / 96 % of pre variance; PC2 loadings; transient counts 5 vs 44 / 33 (far 3 -> 16 / 13) | `state_space_trajectories.py` |

Expected direction (not a substitute for the re-run): 5.1 / 5.2 pool SSs, which loses 90-98 % of its pixels to the glue;
in the C6 primary balance (same three areas plus ALM / tjM1) the glue fix made the acute shift ~10 % LARGER and changed no
star. The response-size norm loses 17 mostly posterior / lateral areas, so its absolute size and noise ratios will change;
the per-spout change vs pre may or may not. Two references (RESTnl / session mean) for the response-size twins:
`{{figure pending, WF_SS_RESPBASE}}` as in the draft; not produced.

What the re-run needs (not launched): refill the mask-keyed binned cache (`state_space_sim._bcache()`, reads the 107
glue-free extracts), then the scripts above for the cue and lick alignments; page figures via
`state_space_page_figs.py`. Outputs land under `M:/MICROSCOPE/Priya/Widefield/state_space_recovery/`.

---

## 5. Sources read (all read-only)

- `wf_captiondraft/docs/status/CAPTION_PASS_DRAFT_2026-11-10.md` (C6, D1, E6, G, H, Priya 11-11 answers)
- `wf_attempt/docs/status/GLUE_BEFORE_AFTER_2026-11-10_2130.md` (no balance step in it), `RUNNING_LIST_2026-11-06.md`
- `wf_decisions1111/_goal_page/index.html` (live GOAL captions 12.17-12.21, 14.9-14.19 + supplements),
  `docs/results_pages/state_space/index.html` (SS v14), `docs/status/GLUE_MASK_FIX_2026-11-09.md`, scripts named above
- step logs `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/render_daqlk/logs/`
  (`l_lateral_ratio{,_g}`, `l_bal_{order_primary,order,angle_lmm,explainer,contact,target_model,trial,nested,reference,
  regions,offset_lmm,offset_primary,blind}_{r2|r3,g}`, `l_recov_primary{,_inside}_{r2,g}`, `l_balimg{,_g}`,
  `l_balimg_primary_{r2,g}`, `l_balmatch_{r2,g}`, `l_offset_terms_{r5,g}`)
- per-animal x epoch boot_delta for `recprim_sessions*.csv` and `offterm_sessions.csv` recomputed read-only with
  `analysis_kit.boot_delta` (rng seed 0); epochs checked against `wfield_local.epochs.epoch_of` under profile
  cohort1_rebuild_20261103 (0 mismatches).
