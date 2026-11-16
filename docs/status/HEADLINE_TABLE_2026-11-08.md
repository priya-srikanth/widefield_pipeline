# Headline claims: old (FINDINGS / pages) vs cohort-1 REBUILD render (11-08)

Cohort 1 only (PS92-PS95). Read-only comparison; no new bootstraps were computed. Old numbers come from
`wf_render_v9zs/docs/FINDINGS.md` (claim numbers in brackets: 3.x = imaging section, 4.x = encoding section,
S = synthesis section 0). New numbers are quoted from the rebuild logs and sidecars.

Path shorthands:
- `$ROOT` = M:/MICROSCOPE/Priya/Widefield/rebuild_20261103
- `$TAG` = hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk
- `$EP` = $ROOT/labcams/grant_figures/$TAG/epoch/data
- `$RF` = $ROOT/rebuild_figs/$TAG
- `$LOG` = scratchpad/render_daqlk/logs (step logs; all quoted steps are `.done` unless stated)

How to read the table:
- **Same** = same direction and the same significance call.
- **Weaker** = same call, but smaller or carried by fewer animals.
- **CHANGED** = the significance call flips, or part of the claim no longer holds.
- n = 4 rule: a claim that at least 3 animals do not show individually is fragile. "computed mean" = a simple mean over
  per-animal values or session rows that I computed. It is never a bootstrap.
- Frame note: the encoder rows use the rebuild's `v7g3_t3d_spont` designs, which add spontaneous-lick terms. The old
  numbers came from `v7g3_t3d` without them. Imaging uses DAQ licks (`_daqlk`).

## 1. CHANGED (the significance call or the content differs)

| # | Claim (old ref) | OLD | NEW | Verdict | New source |
|---|---|---|---|---|---|
| C1 | Frozen decoder, LICK window, far-contra acute loss (3.1, bal_em) | -.483 (3 animals), significant | -.367 [-.666, -.066]; **Bonferroni CI [-.858, +.041] crosses 0**. Per-animal session means: PS92 -.37 (2 sessions), PS93 -.54 (3), PS94 -.33 (1), PS95 none. 3/3 negative, but very thin | CHANGED: loses corrected significance; acute lick far contra rests on 6 sessions | `$EP/epoch_accdelta_by_position_lick_lick.csv` + `_sessions.csv` |
| C2 | Chronic residual is "bilateral ALM / tongue-jaw motor + contralesional SSp" (4.2) | ALM part of the chronic residual | ALM chronic n.s. (cue -1.97 / -0.17; lick +0.46 / +2.58, p_bonf .20-1). **tjM1 chronic significant on both sides** (cue +4.7 / +6.8, lick +7.5 / +9.3 x1e3, 4/4). SSp-m chronic significant (below) | CHANGED anatomy: drop ALM, name tjM1 + SSp-m | `$LOG/b_residual_roi_r.log` |
| C3 | Persistent orofacial-SPECIFIC residual in the cue window (4.2; S3) | no-FR: orofacial - control chronic +1.33 [0.17, 3.09], 4/4. FR: +0.88 (3/4, borderline) | no-FR: **+0.97 [-0.27, 2.56], 3/4**. FR-A: **+0.71 [-0.41, 2.02], 3/4**. Acute orofacial - control still significant (cue +2.72 [1.12, 4.45] 4/4; lick +2.13, 4/4) | CHANGED: the chronic orofacial-specific part is n.s. in both frames. The at-recovery mixed model (z 3.2 / 4.0) was not found in the rebuild logs | `$LOG/b_specificity.log`, `b_specificity_plain.log` |
| C4 | Refit-over-frozen gap G keeps rising (3.3) | log-day slope +.030, z 4.5 | matched +.027, **z 1.75, q .096**; unmatched +.015, q .25 | CHANGED: no longer significant. G at full recovery is still > 0 (+.158 matched, z 7.2) | `$LOG/b_imaging_lmm_r.log` (time_post, behaviour_post) |
| C5 | Per-spout SPREAD tracks the behavioral deficit (3.13) | z -4.5 (ttc) / -5.3 (contact) | spread post:deficit z **-1.8, q .071** | CHANGED: n.s. Pre-cue spread weaker but still significant (C5b below) | `$LOG/b_imaging_lmm_r.log` (=== behaviour) |
| C6 | Frozen target-unique R2 stays DOWN chronically (4.11; S4b) | cortex lick -0.016 [-0.028, -0.004], 0/4 up; with FR -0.008, 0/4 | no-FR lick **-0.0096 [-0.020, +0.004], 1/4 up** (n.s.); FR-A lick -0.0090 [-0.016, -0.001], 1/4 up (significant); no-FR cue -0.0097 [-0.016, -0.004], 0/4 up (significant) | CHANGED only in the no-FR lick cell. The headline holds in the cue window and with FR | `$LOG/z_frozen_claims.log` (section A) |
| C7 | Per-animal half-recovery day, cortex minus behavior (4b(e)) | PS92 -1.5, PS93 +1.9, PS94 -5.7, PS95 -1.8 days (all CIs include 0) | Cortex half-day **not estimable in 3/4** (flag few_hits: PS92 / PS94 / PS95); PS93 not reached. The aim series is unavailable (aim table missing) | CHANGED: per-animal timing comparison is gone. The pooled gap (S7 below) is the same | `$LOG/d_recovery_timing_r2.log` |

## 2. WEAKER (same call, smaller or fewer animals)

| # | Claim (old ref) | OLD | NEW | Verdict | New source |
|---|---|---|---|---|---|
| W1 | Frozen decoder far MIDDLE acute loss, cue (3.1, bal_em) | -.484 | -.318 [-.444, -.208], corrected [-.526, -.167]. Per-animal means -.19 / -.27 / -.36 / -.47 (4/4) | Weaker (about 2/3 of old), still significant, 4/4 | `$EP/epoch_accdelta_by_position_cue_working*.csv` |
| W2 | Near positions acute, cue (3.1) | -.13 / -.20 / -.20 (near ipsi / mid / contra) | -.202 / **-.095** / -.191. Near middle corrected CI [-.236, +.009] | Mixed: near ipsi larger; near middle loses corrected significance | same |
| W3 | Near middle still below pre chronically (3.1) | -.163 | -.145 [-.275, -.014]; **corrected [-.310, +.014]** | Weaker: n.s. after correction | same |
| W4 | Chronic crossnobis distance from August template exceeds June -> August drift (3.8) | 3/4 (.63/1.01, .95/1.16, .43/.43, .59/.81) | .63/.93, .96/1.15, .42/.43, .61/.63 | Weaker: clear in 2/4 (PS92, PS93). PS94 and PS95 are at drift. PROVISIONAL claim, now fragile | `$LOG/g5_june_aug.log` |
| C5b | Pre-cue (ENL) spread tracks the deficit (3.13) | z -4.2 / -4.6 | z -2.6, q .014 | Weaker, still significant | `$LOG/b_imaging_lmm_r.log` |
| W5 | Acute orofacial excess in the Layer-2 mapping change (4.3) | +0.025, z 2.6 (mixed model) | FR-A bootstrap +0.023 [-0.0005, 0.049], 4/4. No-FR +0.024 [-0.011, 0.064], 2/4 | Weaker: borderline / n.s. in the bootstrap (different test from the old mixed model) | `$LOG/b_specificity.log`, `b_specificity_plain.log` |
| W6 | Acute far-contra MAP loss, cue, rest reference (3.5 / 3.14) | plain REST pooled .48 (RESTW .47); per animal .50 / .45 / .53 / .40 | REST, contact gate, vs LATE pre: .58 (968 / 2609 bins, maxstat). Per-animal cue RESTW not rendered (see P8) | Weaker magnitude, same call (far beyond drift). Reference epoch differs | `$ROOT/labcams/grant_figures/map_drift_prepre/<tag>_refs-rest+precue+restnl/drift_vs_stroke_panels.csv` |
| W7 | Acute contralesional SSp-m measured drop (4.1) | -.30 of pre (contra SSp-m) | computed mean of per-animal (d_data / pre_data): -.29, but PS92 -.59, PS93 -.56, PS94 -.12, PS95 **+.13** (1 acute session) | Weaker per animal: clear in 2/4. The ipsilesional row is the robust one (S9) | `$LOG/a_decomp_noFR_lick.log` |

## 3. SAME (same direction and significance)

| # | Claim (old ref) | OLD | NEW | Animals | New source |
|---|---|---|---|---|---|
| S1 | Frozen position decoder retained above chance pre / acute / sub / chronic (Fig 3.4; 3.2) | .86 / .43 / .70 / .76 | .840 / .465 / .727 / .758 (established this session). Delta acute -.347; subacute -.139 (Holm .0015); chronic -.077 (Holm .057). Sidecar 13n prints .842 / .483 / .733 / .752 | as page | `$RF/state_position_sessions.csv`, `$EP/epoch_13n_state_vs_position_cue.csv` |
| S2 | State decoder (bal_em) loses little (3.2) | retained .972 -> .927 / .950 / .922; balanced-accuracy delta -.022 [-.047, -.003] / -.018 [-.041, .001] / -.037 [-.073, -.009] | .972 -> .942 / .949 / .923; delta -.016 [-.029, -.004] / -.019 [-.041, .002] / -.037 [-.077, -.009] | chronic balanced accuracy .94 / .97 / .89 / .98 (old .93 / .98 / .90 / .98) | `$EP/epoch_13delta_state_decoder_cue.csv`, `$RF/state_decoder_balacc_per_animal.csv` |
| S3 | Position falls far more than state (ratio; 3.2, S2) | ~11x / 8x / 2.3x | computed from S1 / S2 fractions lost: position 45 / 13 / 10 %, state 3.1 / 2.4 / 5.0 % -> ~14x / 5.6x / 2x | ratio holds; still mainly acute | derived from S1 / S2 |
| S4 | Frozen decoder far-contra acute loss, cue (3.1, bal_em) | -.644 (4/4) | -.618 [-.695, -.531], corrected [-.722, -.488]. Graded far contra > far mid > far ipsi | per-animal means -.67 / -.63 / -.62 / -.61 (4/4; PS95 1 acute session) | `$EP/epoch_accdelta_by_position_cue_working*.csv` |
| S5 | Far contra still below pre chronically, cue (3.1) | -.179 | -.154 [-.269, -.065], corrected [-.317, -.041] | -.10 / -.14 / -.32 / -.06 (4/4 negative; PS95 small) | same |
| S6 | Pre-cue far-contra acute loss (3.1) | -.271 | -.289 [-.428, -.145], corrected [-.480, -.061] | -.47 / -.37 / -.24 / -.11 (4/4) | `$EP/epoch_accdelta_by_position_precue_working*.csv` |
| S7 | Far-contra code on chronic HITS only ~70-77 % of pre (S4b(b)) | lick -.30 [-.46, -.13], cue -.24 [-.40, -.11], pre-cue -.27 [-.50, -.04] | lick -.288 [-.433, -.133], cue -.269 [-.404, -.144], pre-cue -.273 [-.446, -.093] | lick chronic-hit retention .76 / .60 / .56 / .94 (4/4 < 1; PS95 near pre) | `$LOG/d_decoder_retention_r2.log` |
| S8 | Failed attempts carry ~no far-contra code; hits recover to ~.71 (4.9) | ~.1 of pre; hits .71 | acute failed retention, lick: -.10 / -.13 / .17 / .19 (cue .03 / -.10 / .22 / .23); chronic hits lick, computed mean .72 | 4/4 failed near 0 | same |
| S9 | Behavior back, cortex gap opens chronically (S4b(e)) | behavior +.01 [-.03, .06]; gap -.31 [-.47, -.13]; lagged beh~cortex +.03, cortex~beh +.18 | behavior +.005 [-.026, .029]; gap -.294 [-.431, -.144]; lagged +.038 / +.121 (both n.s.) | pooled | `$LOG/d_recovery_timing_r2.log` |
| S10 | Acute far-contra drop is the LICK, target ~0 (4.1) | measured -.53 / -.30 (ipsi / contra SSp-m), predicted -.62; target <= 1.7 x1e3 | ipsilesional SSp-m measured (computed mean) -.52, predicted -.69 (no-FR) / -.67 (FR-A); target -0.8..+1.9 x1e3 | ipsilesional 4/4 drop (-.71 / -.64 / -.45 / -.27); see W7 for contra | `$LOG/a_decomp_noFR_lick.log`, `a_decomp_frA_lick.log` |
| S11 | Chronic residual in SSp-m (4.2) | cue +9.4 / +9.4, lick +12.4 / +14.0 x1e3 | cue SSp-m contra / ipsi +9.2 / +8.1, lick +11.3 / +11.0 (Bonferroni) | 4/4 | `$LOG/b_residual_roi_r.log` |
| S12 | Residual partly GLOBAL in the lick window (4.2) | control areas lick chronic +1.46 pre-SD, 3/4; cue +0.25 n.s. | lick +1.74 [0.72, 2.74], 4/4; cue +0.42 n.s. | 4/4 (stronger) | `$LOG/b_specificity.log` |
| S13 | Residual unchanged on behavior-matched trials (4.2) | chronic SSp-m lick +5.5 / +5.6, 4/4 | inside envelope +5.33 / +5.45 (all trials +5.45 / +5.87) | 4/4 (ipsi), 3/4 (contra) | `$LOG/b_matched_residual.log` |
| S14 | Layer-2 chronic change NOT orofacial-specific (4.3) | orofacial +.053 vs control +.052; contrast +.002 [-.017, .018] | +.059 vs +.057; contrast +.003 [-.020, .026] (no-FR -.001) | control 4/4, orofacial 4/4 | `$LOG/b_specificity.log` |
| S15 | Spout package shifts contralesionally (4.12a) | contra - ipsi SSp-m +.039 [.011, .081] cue / +.048 [.015, .094] lick (FR), 4/4 | +.0388 [.0144, .0722] / +.0481 [.0196, .0829] (FR-A); no-FR +.029 / +.032 | 4/4 | `$LOG/z_frozen_claims.log` |
| S16 | Executed spout-locked part rises acutely (4.11) | cue cortex +.041, 4/4 | +.049 [.019, .081], 4/4 (no-FR) | 4/4 | `$LOG/z_frozen_claims.log` |
| S17 | Self-fit: target remapped, not lost (S4c, 4.11) | chronic target-unique SSp-m +.017 / +.019, cortex +.010 / +.013 (3-4/4); self-fit gap +.06-.09 | lick SSp-m +.017 / +.013, cortex +.020 (4/4); cue SSp-m +.009 / +.009 (3/4), cortex +.016 (4/4); gap chronic .04-.19, 4/4 | 3-4/4 | `$LOG/b_svf_frA.log` |
| S18 | Template amplitude falls 4/4, returns 2/4; capture stays below 4/4 (3.12) | amplitude pre -> chronic .96->1.19, .96->.84, .93->.68, .91->.73; captured chronic .45/.38/.47/.24 | .96->1.20, .95->.85, .93->.67, .91->.74; captured .47/.38/.46/.26 | 4/4 | `$RF/template_amplitude_per_animal.csv`, `template_captured_per_animal.csv` |
| S19 | Frozen decoder tracks deficit; not back at recovery (3.3, 3.13) | -.040 / unit, z -3.2; F(0) +.083; G vs behavior q .16 | frozen -.097 z -3.35 (matched -.075 z -3.38); F at full recovery +.065 (q .027) / +.107 (z 3.2); G vs deficit q .11-.40 | pooled LMM | `$LOG/b_imaging_lmm_r.log` |
| S20 | Coding ceiling: no acute step, rises with log-day, chronically above pre in 3/4 (3.13) | acute -.014 (z -1.0); log-day +.014 (z 2.4); chronic +.038 (z 5.0); vs contact deficit z -3.9 | acute -.007 (q .74); log-day +.017 (z 4.5); chronic +.040 (q < .001); vs deficit z -4.9 | per animal pre -> chronic .050->.126, .056->.114, **.060->.045 (PS94)**, .050->.081 (3/4, as before) | `$LOG/b_imaging_lmm_r.log`, `m_trend_recovery.log` |
| S21 | Frozen-decoder chronic deficit is within June -> August drift (3.3) | not beyond drift, 3/4 | F chronic vs F drift .04 vs .13, .09 vs .30, **.23 vs .21**, .13 vs .19 | 3/4 (PS94 at drift) | `$LOG/g5_june_aug.log` |
| S22 | Map drift: far contra beyond drift; far middle (cue) not separable; chronic far-middle PRE-CUE loss beyond drift (3.14) | far contra drift .95 vs acute .49; far mid .77 vs .71; pre-cue far mid chronic .53 vs drift .86-.97 | drift = LATE - EARLY pre (new design): far contra .96 vs acute .58 (968 bins); far mid .80 vs .90 (44 bins); pre-cue far mid chronic .52 (623 bins) vs drift .98 (rim-suppressed) | pooled maps | `drift_vs_stroke_panels.csv` (W6 path) |
| S23 | ITI read-out offset fades as aim recovers (0b / 4d) | z-scored -0.48 (p 2e-4), 4/4 (-.35 / -.44 / -.62 / -.51); ITI lateral error -0.45, 4/4 | lfl_lat detrended -0.474 (p .0002), 4/4 (-.40 / -.46 / -.33 / -.70); mae_lat -0.449, 4/4 | 4/4 | `$LOG/e_rlink_angle_z_r.log` |
| S24 | ITI contralesional offset persists chronically (4d0) | +.33 / +.22 / +.25 mm (4/4 each; PS94 +.55 chronic) | like-for-like lateral, computed mean of animals +.35 / +.21 / +.25; PS94 +.44 chronic | 4/4 every epoch | `$LOG/e_rlink_angle_z_r.log` (iti per-animal table) |
| S25 | ITI decoder r falls and does not recover (4d0f) | step -.21, recovery +.004 n.s. | r_norm change -.234 / -.212 / -.235 (all CIs exclude 0) | pooled | `$LOG/e_iti_key.log` |
| S26 | Quiet orofacial events up acutely, LP quiet frames (3.Q) | LMM +0.82 log10 acute (x6.7) | rate_both orofacial +0.74 / +0.45 / +0.21 (q < .03 each); rate_neg +0.85 / +0.50 / +0.30 | LOAO k choice AUC 1.0 in 4/4 | `$LOG/gb_quiet_lp.log` |
| N1 | NEW control: mix-matched within-session refit (Fig 3.4 control) | none (new) | acute far ipsi / mid / contra -.221 / -.148 / -.259 (Holm .0045 each); subacute n.s.; chronic near ipsi +.152 | 4 animals | `$RF/refit_matched_pre_stats.csv` |

## 4. NOT YET RENDERED, FAILED, or NOT FOUND in this run

| # | Headline (old ref) | OLD | State | Producing step |
|---|---|---|---|---|
| P1 | Partial refit: chronic target +0.034 / executed +0.054 / +0.064 specific in contra SSp-m, 4/4 (S4d) | as stated | **RUNNING** (10/54 sessions at 11:24, 11-08) | `b_partial_frA` (`.claim`, no `.done`). Lick-kernel partial refit `a_partial_ls` is done but was not compared here |
| P2 | Frozen encoder explains less trial-to-trial variance on contact trials, -.09 / -.07 / -.17 R2 (S2, 11-05) | trend, LMM p .06-.09 | Not found. `t_te_readout` is `done_nopartial` and prints per-group unique R2, not this whole-model readout | te readout / FROZEN_B12 reanalysis (not in the chain) |
| P3 | Map drift under the LP-lick gate (3.14 LP panels) | far contra drift .94 / acute .49 | **FAILED** | `g5_map_drift_lp_r` |
| P4 | Aim recovery link + aim half-recovery timing (4b(e), 0b) | see C7 | **FAILED**; aim table missing | `e_rlink_aim_r`; `d_recovery_timing_r2` aim input |
| P5 | Quiet-event gain / statistics (3.Q gain arm) | x5.1 / x2.5 / x1.4 face-regressed | **FAILED** (assertion: oro - post does not reproduce oro_post) | `gb_stats` |
| P6 | Mix-matched refit, lick-trial version | n/a (table only) | **FAILED** | `m_mixmatch_cue_l` |
| P7 | Goal summary figure / section G figure copy | page figures | `d_fig_goal_summary` **FAILED**; `g7_section_g_figs_r4` has no `.done` | those steps |
| P8 | Per-animal cue-window RESTW far-contra map amplitude .50 / .45 / .53 / .40 (3.5) | as stated | Epoch 15r rendered only `lick_lick` (acute far contra empty under the lick gate) | epoch family 15r, cue arm |
| P9 | Cued orofacial BALANCE predicts unseen-animal recovery, cvR2 .50 vs days .23 (0b) | as stated | No step in this render produces it (`l_balimg` = imaging control only) | lick-balance biomarker script |
| P10 | Trial-by-trial coupling lost on failed attempts, -0.26 (4.6) | 4/4 | No coupling step found in the chain | `coupling_by_class.py` |
| P11 | Rest carries spout position, observed / null 1.634 (3.9) | 4/4 | Not found (15f renders the frozen decoder ON rest: retained .32 / .52 / .60, a different readout) | rest position decode |
| P12 | Within-lick correction summary (4.21) | null result | `e_wlc_summarise` has no `.done` | `e_wlc_summarise` |
| P13 | Basis frame SD | n/a | **FAILED** | `l_basis_frame_sd` |

Caveats on the comparison:
- Old encoder numbers used designs without spontaneous-lick terms; the rebuild uses `_spont` designs.
- Old imaging used a 0.10 s lick refractory; the rebuild uses `_daqlk` licks.
- Old map drift compared June with August; the rebuild compares LATE with EARLY pre.
- Rows marked "computed mean" are simple means of the logged per-animal values; they carry no CI.
