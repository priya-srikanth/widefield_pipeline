# Kinematics page, 3-D side: OLD (pre-gate) vs NEW (camera working-period gate, A20)

Read-only comparison, 2026-11-06. The page claims come from the live source `M:/MICROSCOPE/Priya/Widefield/kin_page_src/kin_page/index.html`.
NEW = `rebuild_20261103/session_poses_derived/<tag>/...`. OLD = `DeepLabCut/Widefield/session_poses/pose3d/...`, `session_poses/cam1_licks/trial_classes/` and `kin_page_src/kin_page/tc_share_all*.csv`.
Scripts and raw outputs: `scratchpad/kin3d/cmp*.py`, `out*.txt`.
Contra = far_R / close_R (all lesions are left). Per-animal values are always given in the order PS92 / PS93 / PS94 / PS95.

## 0. Headline

* **The gate barely touches lick-based 3-D measures.**
  * The spout error `per_session.csv` changes in only 3 of 3,495 rows, all of them PS93_20260821 far_R (acute): |d miss| ≤ 0.03 mm.
  * Phase curves change only at far_R acute for PS93, and the significance bars are identical.
  * Decoding is identical.
  * Lick-level 3-D results therefore stand exactly as written.
* **Every conclusion change is in section 6 (trial-class shares, Figs 6.1 and 6.2).** The denominator there is now the working period, not all engaged trials.
* **Pipeline gap (affects Fig 7.5, Fig 9.6, the 3-D rows of the section 3 / 5 / 7 tables).** At code 8d6da32d, `scripts/kinematics_pooled_figs.py` still reads the UNGATED `pose3d/kinematics/sessions.csv` (dated 10-23; there is no k_kinematics step).
  * So `pose3d_change_pooled*.csv` and `pose3d_days_pooled.csv` in NEW are byte-identical to OLD. They were not re-gated.
  * Only its cue-trace and phase inputs are routed and gated.
  * The impact is expected to be negligible. The gated spout-error rows are unchanged, and the gated jaw cross-check moves 3-D FAR_DIR by ≤ 0.003 mm. This is UNCERTAIN until the table is re-gated.

## 1. Conclusions that change direction or significance, or quoted numbers that move materially

All items are in section 6. The NEW figures are `jaw_only_incomplete_figs/figures/trial_class_share_raw*_tests_added.csv`. Values are change vs own pre. ** = Bonferroni, * = CI excludes 0.

| Page item | OLD per animal (92/93/94/95) | OLD pooled | NEW per animal | NEW pooled |
|---|---|---|---|---|
| Fig 6.2 far contra jaw-only, chronic | −.014/−.002/−.013/−.023 | **−0.013 \*** | −.009/+.003/−.005/−.005 | **−0.004 n.s.** |
| Fig 6.2 far contra no response, chronic | −.012/−.004/−.010/−.008 | **−0.009 \*** | −.004/+.001/−.001/+.002 | **−0.001 n.s.** |
| Fig 6.1 text "close contra **" (no response acute, close_R) | +.042/+.047/+.032/+.030 | +0.038 ** | +.004/+.014/+.035/+.025 | **+0.020 \* (loses Bonferroni)** |
| Fig 6.2 jaw-only close contra acute (not quoted, star on figure) | +.024/+.040/+.078/+.009 | +0.038 ** | +.018/+.018/+.078/−.001 | +0.028 * |
| Fig 6.2 "far mid incomplete +0.22* acute only" | +.150/−.022/+.430/+.328 | +0.222 * | +.167/−.004/+.451/+.326 | **+0.235 \*\*** |
| Fig 6.2 far contra jaw-only subacute "+0.03*" | +.042/+.018/+.019/+.027 | +0.027 * | +.059/+.027/+.017/+.029 | +0.033 ** |
| Fig 6.2 "close mid no response +0.07**" | +.070/+.116/+.071/+.032 | +0.072 ** | +.054/+.034/+.069/+.031 | **+0.047 \*\*** |
| Section 6 text, acute far contra no response per animal ".08/.07/.15/.18", pooled ".12" | raw .079/.074/.145/.183 | .120 | raw .064/.068/.132/.147 | **.102** |
| Section 6 / Fig 6.1 "chronically ... no response ≤ .008 everywhere" | max .008 (close mid) | | close mid **.019** (PS92 .000 / PS93 .001 / PS94 .021 / PS95 .054); far ipsi .008; far contra .007 | change vs pre at close mid chronic −0.001 n.s. |
| Fig 6.1 far ipsi no response acute ".10" | .104/.115/.121/.050 | .098 | .049/.073/.130/.043 | **.074** |
| Fig 6.1 close mid no response acute ".09" | | .089 | | .067 |
| Close no response / jaw-only acute "~.04 each" | | no resp .044/.045, jaw-only .037/.046 (ipsi/contra) | | no resp .027/.026, jaw-only .030/.037 |

**New stars that are not on the page.** They are small, but they will show on the re-rendered Figs 6.1 and 6.2.

* Incomplete at the close spouts now reaches significance. Close ipsi: acute +0.014 *, subacute +0.014 **. Close mid: acute +0.017 *, subacute +0.013 *. Close contra: subacute +0.013 *.
  * Example per animal, close ipsi subacute: +.020/+.015/+.018/+.005.
  * Raw incomplete counts at the close spouts roughly triple to quintuple (for example close_L acute 3 → 19 trials). UNCERTAIN why: probably the shared per-trial class table now counts cam1-only / low tongue events as incomplete.
* Contact at close ipsi, subacute: −0.024 n.s. → **−0.040 \*\*** (−.042/−.038/−.023/−.058).
* Jaw-only: close ipsi subacute, close contra subacute and far ipsi subacute become * (+0.017 / +0.018 / +0.029).
* Chronic below-pre stars are lost: no response at close ipsi (−0.004 * → n.s.) and at far ipsi (−0.008 * → +0.003 n.s.).

**Read.** The section 6 story holds: acute far-spout failure by three routes, with every class back at pre chronically. The chronic return is now cleaner, because the small below-pre stars are gone.

Text that needs rewriting:
* the "(close contra **)" note;
* the "no response ≤ .008" bound;
* no response .12 → .10 and its per-animal values;
* far ipsi .10 → .07 and close mid .09 → .07 (raw);
* the Fig 6.2 reading line (jaw-only −0.01*, no response −0.01*, close mid +0.07**, far mid +0.22*).

## 2. Numbers that move slightly (no conclusion change)

**Fig 6.1 / 6.2 far contra, acute / subacute / chronic.** Per animal is shown before pooled.

* Contact change.
  * Per animal acute: OLD −.916/−.832/−.916/−.937, NEW −.927/−.844/−.907/−.939.
  * Pooled: OLD −0.90** / −0.21** / +0.02, NEW −0.904** / −0.217** / +0.005.
* Incomplete change.
  * Per animal acute: OLD +.739/+.643/+.597/+.319, NEW +.783/+.676/+.601/+.340.
  * Pooled: OLD +0.57** / +0.17** / 0.00, NEW +0.600** / +0.169** / −0.001.
* Jaw-only acute.
  * Per animal: OLD +.111/+.120/+.189/+.446, NEW +.085/+.100/+.189/+.462.
  * Pooled: OLD +0.22**, NEW +0.209**.
* No response acute: OLD +0.11**, NEW +0.095**.
* Raw far contra shares, mean of animals:
  * contact pre / acute / chronic: OLD .952 / .052 / .975, NEW .958 / .054 / .963;
  * incomplete: OLD .02 → .596 (subacute .186), NEW .022 → .623 (subacute .191);
  * jaw-only: OLD .016 → .232, NEW .012 → .221.
* Far mid raw acute: contact .43 → .45; incomplete .26 → .28; jaw-only .20 → .18.
* Far ipsi: contact .69 → .71; contact change −0.21** → −0.189**.
* Close contact acute change: OLD −0.064 / −0.075** / −0.081**, NEW −0.053 n.s. / −0.064** / −0.069** (ipsi / mid / contra). The page says "−0.06 to −0.08", which becomes "−0.05 to −0.07".
* "Chronic far-ipsi incomplete −0.06 is PS93": −0.058 → −0.056 (PS93 −0.228). Unchanged.
* Trials per epoch (sum over session × spout): pre 15,929 → 15,970; acute 7,306 → 7,112; subacute 7,518 → 7,777; chronic 20,176 → 20,496. Session counts are unchanged (pre 31 / acute 16 / subacute 17 / chronic 43).

**Fig 6.3 time model** (`time_lmm.csv`). Jaw-only steps shrink slightly:
* far ipsi post 0.119 → 0.103;
* far mid 0.156 → 0.141;
* close contra 0.049 → 0.041.

Incomplete at far contra: post 0.620 → 0.644, slope −0.179 → −0.186. All terms stay q < .01. UNCERTAIN: the OLD file lacks a `cls` column, so rows were matched by order.

**Fig 6.4 jaw traces.** These are crude checks (session max 200–1500 ms, then animal mean, then mean of animals). They are not the figure statistic, so all are UNCERTAIN.
* Jaw-only openings stay 0.27–0.49 mm at the far spouts in every epoch.
* Contact trials stay ~1.9–2.8 mm; far contra acute contact is 2.04 → 1.89.
* Incomplete stays ~1–2 mm at the far spouts.
* "~0.3 mm jaw-only / ~2 mm contact / ~1 mm incomplete" still holds.

**Fig 3.6 / section 3 pooled table (cam4 executed angle, `angle_change_pooled_contra.csv`).**
* Far contra acute per animal: OLD −28.1/−10.7/−43.1/−34.4, NEW −28.4/−10.7/−43.1/−34.4.
* Far contra acute pooled: OLD −29.1 [−40.2, −16.3], NEW −29.2 [−40.3, −16.3]. Subacute and chronic are unchanged (−7.8, −2.3 [−6.0, 0.6]).
* Other cells move ≤ 0.5° per animal and ≤ 0.12° pooled. Far mid acute −9.4 → −9.5. Far mid chronic −2.9 → −3.0 (the page says "−2.9 at the middle spouts"). Stars are unchanged: all CIs keep the same side of 0.

**Section 10 / Fig 10.1 jaw opening agreement r** (median over sessions):
* OLD .64/.57/.67/.83, NEW .64/.575/.74/.86.
* `fig_pose_validation_sessions` already flags the PS94 and PS95 table mismatches.
* The direction is unchanged; agreement improves.

**Section 10 other values.**
* Decoding (`vs_cam4/decoding.csv`) is identical: cam4 .53/.42/.78/.59, 3-D .805/.67/.86/.74.
* `effect_sizes.csv`: 3 z_cam4 and 6 z_3d rows of 2,736 change (max 0.20). Not quoted on the page.

**3-D cross-check (`jaw_agreement.csv`) FAR_DIR change vs pre.** This independently confirms the section 5 3-D rows.
* Acute: OLD −.255/−.207/−.410/−.325, NEW −.257/−.209/−.408/−.324.
* Chronic: OLD −.136/−.153/−.024/−.023, NEW −.136/−.153/−.025/−.026.
* The "2 of 4 chronic" read is unchanged.

**Fig 9.7 phase** (`pose3d_phase_pooled_contra_figdata.csv`).
* Only far_R acute changes (driven by PS93): max |d| 0.06° / mm.
* Range far contra acute −38.5 to −26.7°, chronic −13.9 to −4.1°, distance acute min −0.84 mm. All are the same as OLD.
* Cluster bars: 0 of 1,188 rows differ.

**Fig 4.11 pooled 3-D paths.** Max |d| = 0.0007 mm.

**Figs 3.2–3.5 / 3.7 / 6.6 cue traces** (`by_epoch.csv`, set engaged → working). Far contra d_mid per animal:
* acute: OLD 35.26/14.58/45.92/37.36, NEW 35.21/14.58/45.94/37.37;
* chronic: OLD 4.39/9.62/14.92/2.92, NEW 4.39/9.62/14.98/3.01.

These are stored with + = ipsi. The page's 3-D midline numbers come from other tables (see 3a). Further moves:
* d_jaw_open: max |d| 0.115 mm (PS95 chronic far_R +0.18 → +0.11).
* first-lick latency: max +15 ms (PS93 acute 996 → 1011 ms).

No page number depends on these values.

**Section 3 table, closest approach.** Gated per_session.csv, far_R `all`, minus August pre:
* acute +2.062/+0.364/+2.503/+2.081;
* subacute +0.621/+0.002/+1.427/+0.296;
* chronic +0.163/+0.066/+1.023/+0.162.

These are identical OLD and NEW. The page's +2.06/+0.34/+2.45/+2.09 come from a slightly different (unrouted, ungated) table. The only shift is in first-lick PS93 acute (0.741 → 0.733).

## 3. Not compared, or compared only partly

a. **Fig 7.5, Fig 9.6, and the 3-D rows of the section 3 / 5 / 7 tables are not re-gated.**
   * The tables involved: midline error −25.0/−9.5/−36.1/−32.7; FAR_DIR −0.26/−0.18/−0.40/−0.31; reach −0.06/−0.63/−0.44/−1.50.
   * NEW `pose3d_change_pooled_contra.csv` and `pose3d_days_pooled.csv` equal OLD exactly. `kinematics_pooled_figs` reads the ungated `pose3d/kinematics/sessions.csv` (10-23), and `pose3d/kinematics/` was not re-run.
   * The page's Fig 7.5 numbers therefore stand only because the input is unchanged: miss +1.75 [0.82, 2.38], reach −0.71, speed −56, jaw −2.4, chronic miss +0.35, midline −26.9.
   * Expected impact ≈ 0 (see section 0). UNCERTAIN until the table is re-gated.
   * `pose3d_summary_fig` (Figs 9.1–9.4) does re-gate internally (`gated_sessions`) but writes PNGs only, so there is no table to compare.
b. **Figs 9.1–9.4 / 3.2–3.5 / 4.7–4.10 / 7.1–7.4 / 9.5:** per-animal PNGs, compared only through their source tables above.
c. **Section 6 "cam1-corrected jaw-only share" table** (.277/.282/.387/.577), **section 7 "no-contact share" table**, and **Fig 7.6:** these come from `cam1_licks/incomplete/` and `presentations/_source_PI_overview_2026-11-02/`. None was re-run in the K steps.
d. **Figs 1.4–1.6 (3-D first-lick aim):** these come from `imaging_encoding/iti_licks/`. They are not part of this kinematics re-run and were not compared.
e. **LMM supplements** (`lmm_hidden_20261106` from k_lmm_curves) were not compared.
f. **Old lick-aligned cue traces** (`pose3d/lick_traces/`) vs NEW `pose3d_cue_traces_first_lick`: not compared, since the page quotes no numbers from them.
g. **Fig 8.1 W2, and cam4 cohort figures:** out of scope (A27 already done).
