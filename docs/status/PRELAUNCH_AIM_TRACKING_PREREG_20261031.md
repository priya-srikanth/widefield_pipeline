# Does cortex carry the RE-AIMED launch command before the tongue moves? Across-spout frozen decoder, tracking at the same target: design written BEFORE any number (2026-10-31)

Question (Priya 10-31). Before the stroke, the launch aim varies only a few degrees within a spout, which is about the
size of the tracking noise (5-8 deg per lick). So the within-spout pre-launch decoders failed: S25 areas (gate 0/4,
`PRELICK_HEMISPHERE_TUNING_PREREG_20261030.md`) and S25b pixels (nominal 3/4, positive control failed,
`PRELICK_SVD_TUNING_PREREG_20261030.md`). Here **the STROKE supplies the variation**. After the stroke, launches at the
same spout deviate by up to ~13 deg (far contra), which is large. The design:
1. Train an ACROSS-SPOUT launch-angle decoder on each animal's pooled August pre licks.
2. Freeze it.
3. Ask, in post sessions, whether **at the same target** the decoded angle follows the deviated launch (cortex carries
   the aim) or stays at the target (the aim error is set downstream of what cortex shows).

The same decomposition already exists for the PRE-CUE window. There the decoded angle barely moved (-0.19 deg, ns) while
execution moved -12.9 deg (4/4): ENL reads the target (FINDINGS "ENL -> first-lick angle", `lick_enl_offset_explore.py`
B7). After the lick, the cued balance follows the landing error. **The PRE-LAUNCH window (after the cue, before the
tongue) has not been tested this way.** A second arm uses spontaneous ITI licks. They have no cue, so the pre-lick
window is not contaminated by a cue response, and their post-stroke bias persists (docked ITI landing -7.5 deg chronic vs
pre after the same previous spout; FINDINGS 4.22).

Code (stage 2): `scripts/prelaunch_aim_tracking.py` (`extract` / `fit` / `summarise`, `--plot-only`; figdata pkl).
Outputs only under `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/
prelaunch_aim_tracking/` (`writeguard.assert_writable`).

**What was looked at before writing this:**
* the code of `lick_enl_angle_decode.py`, `lick_enl_offset_explore.py`, `likeforlike.py`,
  `prelick_hemisphere_tuning.py`, `prelick_svd_tuning.py` (S25b prereg), `iti_licks.py`, `iti_dock_audit.py`,
  `learned_compensation.py`, `joint_locanmf.py` / `joint_xsession.py` / `joint_basis.py`, `frozen_models.make_spec`,
  `analysis_kit.boot_delta*`;
* the joint-basis manifests;
* the frozen-store listing;
* BEHAVIOUR-only counts and launch-angle SDs (section 0).

**No imaging value, decoded value or tracking number of this analysis has been computed.**

## 0. Feasibility (behaviour only; imaging sessions of S25 = 107 curated sessions, 31 August pre)
* **CUED (S25's first-lick set, cached in `prelick_hemisphere_tuning/sessions/<label>.npz`):** pre licks per animal
  PS92 1750 / PS93 2751 / PS94 3178 / PS95 3611 (7-8 sessions each).
  * theta_L SD total 12.4-18.2 deg; within session x spout 10.0-15.3 deg.
  * The between-spout share of theta_L variance is only 0.26-0.39. Spout means span about -18 .. +18 deg (e.g. PS92
    far_ipsi -17.7, far_contra +13.9).
  * So an across-spout decoder has a usable spread to learn. Most of the lick-level variance is still within-spout noise,
    which pre-stroke cortex does not carry (S25 / S25b).
* **ITI (docked phase, `learned_compensation/iti_licks.csv.gz` = `iti_dock_audit` phases, gap >= 1 s, onset_ok, finite
  theta_L):**
  * Pre licks PS92 1022 / PS93 1660 / PS94 1381 / **PS95 163** (8 / 7 / 8 / 8 sessions; per-session median 143, range 8-308).
  * Acute 628 / 575 / 357 / **28 (PS95, 1 session)**; subacute 209 / 427 / 697 / 131; chronic 1294 / 1608 / 1155 / 774.
  * With gap >= 2 s: pre 793 / 887 / 886 / 131.
  * theta_L SD total 15.3-21.6 deg; within session x previous spout 9.0-14.4 deg; between share 0.37-0.75.
  * Onset after trial stop: 5th / 50th / 95th pct 0.91 / 1.59 / 2.24 s.
* **Launch angle is available for ITI licks** (`licks_scored.csv.gz` covers context `iti`; onset_ok in ~97 % of the
  docked gap >= 1 s licks). So theta_L is used in both arms. No substitute angle is needed.
* **Jaw onsets** (`imaging_encoding/qc/jaw_onsets.csv`, cam4 / cam1 onset + re-onset per cued trial; 10-21 pose handoff):
  * They exist for CUED trials only. They give a jaw-referenced first lick (S25 `firstjaw` rule) for 64-94 % of S25's
    cued first licks per session (median 80 %; every session >= 40).
  * **There are NO jaw onsets for ITI licks.** The table is per trial and cue-referenced. No whole-session ITI jaw
    detector exists.
* **LocaNMF:** a SHARED per-animal joint basis exists (`joint_locanmf`; newest per animal, built 08-15 over 11 PRE
  sessions June 06-08 + Aug 06-14; PS92 `76d884873920` 95 components, PS93 `0141bb830372` 87, PS94 `a5e70d5ecd36` 90,
  PS95 `a9170190179c` 95; local `C:/wf_local/joint_bases` = the share copy).
  * Footprints are fitted on pre data only, label-free.
  * The repo's frozen pre-cue joint decoder (`decoder_precue_joint76d884_*`) already uses it.
  * -> feature set (c) is FEASIBLE.
* **Frozen store:** `frozen_models.local_dir()` = `C:\Users\SabatiniLab\frozen_models` (WIDEFIELD_FIGURES_WORKING =
  C:/Users/SabatiniLab/cue_lick). It exists and holds decoder / encoder / enl_angle / ... kinds (42 models for PS92).
  This is re-checked and printed before any fit.

## 1. Lick sets, angle, target reference (fixed)
Sessions:
* curated, PS92-PS95 only (cohort 2 PS118 / 120-122 never);
* imaging sessions of S25 (`prelick_hemisphere_tuning/extract_summary.csv`);
* epochs `wfield_local.epochs.epoch_of`; **pre = AUGUST pre only**; June not used.

**CUED arm = S25's PRIMARY set, read from its cache** (no re-selection): cued FIRST post-cue licks of engaged trials, with
* onset_ok;
* finite theta_L, reach, peak speed and RT;
* **no other lick onset (licks_all, any context, + behaviour licks) in the 1.0 s before tongue onset**.

Per lick the cache holds `first_t` (tongue onset, DAQ), `first_pos`, `first_theta_L`, `first_trial_id`, `first_rt`,
reach and speed.

**ITI arm:** licks of `learned_compensation/iti_licks.csv.gz` with
* `phase == docked` (spout parked; `iti_dock_audit`);
* onset_ok and finite theta_L;
* previous trial's spout one of the six canonical spouts;
* **no lick onset in the 1.0 s before** (S25's rule: licks_all ∪ behaviour lick onsets, 50 ms same-lick tolerance; the
  table's own `gap >= 1` must also hold), i.e. bout-first;
* no cue / strobe in [t - 2.0, t + 0.75] s (S25 / iti_licks `CUE_EXCL`);
* imaging frames covering the window.

**Angle = launch angle theta_L** (`within_lick_correction`, first 12 ms of protrusion, deg, **+ = CONTRALESIONAL =
mouse right**), both arms.

**Target reference** (the "same target" cell):
* CUED = the spout.
* ITI = the PREVIOUS trial's spout (`prev_pos`). ITI licks are aimed relative to it (gain ~0.23 pre, FINDINGS 4.22).
  The spout itself is docked, i.e. at one parked position for every ITI lick, so the docked position cannot index a
  target. It is the same for all licks and enters only as the constant context of the arm.

## 2. Signals, windows, features (fixed)
Source: `combined_encoding.load_imaging` (adopted hemo variant SVTcorr V, U_atlas on the brain mask;
`WFIELD_INPUT_CACHE`), with the slow drift projected out (`ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S)`), x 1e3. Sampled on
the 10-ms grid and averaged into **0.05-s bins from -0.80 to +0.30 s around tongue onset**. This is S25's
`_sample` / `_binned`; every step is linear, so it is done in V-space and projected (S25b's trick).

**Windows (s from tongue onset):**
* **PRE-LAUNCH (primary): the 9 bins -0.50 .. -0.05**, kept as SEPARATE features (dynamic).
* PRE-LAUNCH, end -0.10 (pre-registered sensitivity, Priya 10-31): the 8 bins -0.50 .. -0.10.
* **POST-CONTACT +0.05 .. +0.30** (5 bins) = positive control.

**Baseline (both arms): each bin minus the BASE -0.80 .. -0.50 s mean** (S25).
* CUED: identical to S25. For most licks BASE is before the cue, but for RT > 0.5 s BASE contains the cue onset; this
  is stated.
* ITI: the same window. It is LICK-FREE by construction, because the 1.0 s gap rule means no lick onset in
  [-1.0, 0) s. It keeps the arms identical, and the bins are referred to the lick, not to a cue.
  * The residue of the previous lick's GCaMP6s response is not lick-free: it decays across BASE and the window.
  * Sensitivity: **gap >= 2.0 s** (BASE then starts >= 1.2 s after the last onset).

Sensitivity for both arms: **NO baseline subtraction** (raw bin levels, the ENL convention). Subtracting BASE removes
the sustained pre-cue position code, so the primary decodes the CHANGE since BASE, and this sensitivity decodes the level.

**Feature sets** (each a separate decoder; features = time bins x components, flattened):
* **(a) ALLEN (PRIMARY)**: the 45 split-basis area features of `iti_licks.area_features` (every Allen area x hemisphere,
  ALM / tjM1 + MOs / MOp rest, ROI v8), each divided by the session's frame SD (`iti_licks.area_sd`, the ENL decoder's
  z-scoring). 9 x 45 = 405 features.
  * Why primary:
    1. It is the ENL decoder's exact feature definition, so the pre-cue (-0.19 / -12.9) and pre-launch answers differ
       only in the window.
    2. It has the fewest features for 1.2-3.6k pre licks.
    3. It is interpretable by area.
    4. Pixels did not demonstrate better power at these trial counts (S25b: POST positive control failed at K = 10).
* **(b) WHOLE-BRAIN SVD**: SVTcorr reconstruction in Allen space over the brain mask (the `load_imaging` mask; ONE pixel
  set per animal = the intersection over its sessions; asserted near-identical), **8 x 8 block means** (~155 um; S25b
  used 2 x 2 on ALM + tjM1 only; 8 x 8 is set by memory for the whole brain, decided now, not on a result).
  * Stored in V-space per lick (bins x K_svd) + the session's binned U, so no pixel movie is formed. Pixel features =
    V_bin @ U_blk^T (dF/F x 1e3, no per-pixel scaling, as S25b).
  * **PCA on the flattened (bins x blocks) features, fitted on the TRAINING sessions only**, inside every outer fold.
    Label-free, so it is reused across permutations.
  * **K = 20 primary, 10 and 40 sensitivity**; PC scores standardised on the training sessions.
* **(c) LocaNMF (shared joint basis)**: the newest per-animal joint basis (ids in section 0).
  * Components = `Basis._operator` (pinv(A) U_joint) applied to U_joint[mask]^T U_atlas V.
  * It is computed in V-space from `load_imaging`'s V / U, so frames match `ft` by construction. This is `Basis.project`
    except that pixels outside `load_imaging`'s mask (MOB, glue) are dropped, which is stated.
  * It is applied to EVERY session including the in-fit pre sessions, for a consistent definition.
  * Footprint-scaled, then drift-removed and x 1e3 as above; each component divided by its session frame SD.
    ~87-95 components x 9 bins.
  * `variance_captured` reported per session.

## 3. Decoder (fixed) -- per animal x arm x feature set x window
* Target y = theta_L (deg; TOTAL angle, across spouts). No within-spout centring of y or of the features: the target IS
  part of what is to be decoded. Features are not session-centred either; ENL did not centre, so the frozen decoder
  transfers unchanged.
* Pipeline: [PCA, set (b)] -> StandardScaler (training) -> ridge. **Alpha from 10^{-1..5} (13 values, ENL `ALPHAS`)
  chosen by inner leave-one-SESSION-out CV** on the training sessions (maximise pooled held-out r), then refit on all
  training sessions. A trial-level GCV would leak within-session structure. Edge-of-grid alphas are counted and reported.
* **Pre scores = leave-one-session-out (LOSO)** over the animal's August pre sessions. Pre predictions are held out.
* **FROZEN** (rule 10): the all-pre decoder AND its LOSO siblings go in one payload through
  `frozen_models.load_or_fit`. The payload holds the scaler, PCA, alpha, weights and a SHA-1 of (X, y), with a loud
  warning on mismatch (as `lick_enl_angle_decode.frozen_fits`).
  * Kind **`prelaunch_aim`**; `make_spec(animal, "prelaunch_aim", align=f"{arm}_{window}[_jaw]",
    source=f"{allen_split_r8 | svd_blk8_K{K} | joint{basis_id[:6]}}[_{variant}]", train_labels=<August pre labels>,
    n_features=...)`.
  * One model per animal x arm x feature set x window x variant that scores post sessions.
  * **LOCAL store only**; `local_dir()` printed and checked to exist and already hold other kinds before fitting; if
    absent or empty, stop. The smoke test writes to a scratchpad store (`WIDEFIELD_FROZEN_MODEL_DIR`), as S25b.
* Post sessions are scored by the frozen all-pre decoder.
* Minimum counts: an animal x arm is fitted with >= 3 pre sessions and >= 150 pre licks; a session enters the epoch
  tables with >= 10 licks (ENL `MIN_SESS`); a session x target cell with >= 3 licks.

## 4. Pre-stroke gate and calibration (computed and printed BEFORE any post-stroke statistic)
* **Gate per arm x feature set (PRE window):**
  * Statistic: pooled LOSO r (all pre licks of the animal, held-out predictions vs theta_L).
  * Null: **200 permutations of y within SESSION** with the whole label pipeline refit (inner alpha CV, LOSO; PCA
    unchanged by construction).
  * p = (1 + #{null >= obs}) / 201.
  * **An arm x feature set passes if p < 0.05 in >= 3 / 4 animals.**
  * Also: per-session r (mean over pre sessions) and null 95th pct.
* **Within-spout partial r** (pre, expected ~0 from S25 / S25b): corr of held-out prediction and theta_L, both centred
  within session x target cell; 200-permutation null within cell. This is the term that would make tracking possible
  lick by lick; reported, not gated.
* **Calibration (compression) slopes**, pre LOSO predictions:
  * **b_sp (PRIMARY normaliser)** = OLS slope of the session x target-cell MEAN decoded on the cell MEAN theta_L
    ("how far the decoded angle moves per degree of angle across targets pre-stroke");
  * b_all = lick-level slope (secondary normaliser);
  * plus the decoded / actual SD ratio.
  * b_sp is chosen as primary because a target-reading decoder is attenuated at the lick level by the within-spout
    noise it cannot see, so normalising by b_all would inflate any tracking by roughly 1 / (between-share) ~ 3x.
  * b_sp <= 0 or its pre CI including 0 -> that animal x arm has no calibration, and its index is "not estimable".
* **POSITIVE CONTROL (POST window)**: same gate. If POST fails the gate, the method cannot read angle even after
  contact, and a PRE result in that arm says nothing (reported in those words).
* Arm fails the gate -> its post-stroke numbers are DESCRIPTIVE, labelled "gate failed".

## 5. Primary post-stroke readout: the TRACKING INDEX at the same target
This reuses the ENL B7 decomposition (`lick_enl_offset_explore.task_b`), with gain normalisation added. Per lick, with
the reference = the animal's pre means at the lick's TARGET cell (spout; ITI: previous spout):
* pre LOSO predictions;
* the pre mean over the animal's pre sessions with equal session weight;
* for a PRE lick, its own session left out (LOSO reference).

Definitions:
* `d_dec` = decoded - pre mean decoded at that target; **normalised `d_dec* = d_dec / b_sp`** (in angle units);
* `d_exe` = theta_L - pre mean theta_L at that target (the behavioural aim change; ENL's `ang_shift`).

Session value = mean over its licks (as B7). Then, per animal x epoch (rule 8: the PER-ANIMAL TABLE FIRST, with n
sessions / n licks, d_dec*, d_exe, per-animal index):
**TI = Δ d_dec* / Δ d_exe**, Δ = epoch - August pre. Pooled: `analysis_kit.boot_delta_columns` (paired nested animals
-> sessions, seed 0, the SAME draws as `boot_delta`) on the columns [d_dec*, d_exe]. The pooled TI is the ratio of the
two animal-weighted Δs; its 95 % CI is the percentile of the per-draw ratio.

Reading the index:
* **TI = 0: the decoded angle stays at the target.**
* **TI = 1: it follows the executed launch as far as it follows angle across targets pre-stroke.**

Tests:
* **H0 (TI = 0) is tested on the numerator**: `boot_p` of the Δ d_dec* draws (TI = 0 iff Δ d_dec* = 0; no ratio
  instability).
* H1 (full tracking): `boot_p` of Δ(d_dec* - d_exe) (TI = 1 iff it is 0). This is the gain-normalised like-for-like
  offset.

**Estimability**: TI is reported as a number only where Δ d_exe's CI excludes 0 and |Δ d_exe| >= 3 deg. Otherwise it is
"not estimable (no execution change)" and only the numerator is reported. The share of draws with |Δ d_exe| < 1 deg is
printed.

**PRIMARY cells: arm (CUED, ITI) x post epoch (acute, subacute, chronic), Allen, PRE window = 6 tests; Holm over the 6**
(numerator p). A tracking result = numerator CI excludes 0 AND >= 3 / 4 animals with the same sign AND Holm p < 0.05.
Also printed:
* the raw (un-normalised) Δ decoded and Δ executed, i.e. ENL's two B7 rows;
* `likeforlike.offset` (method nw, sigma by held-out pre error) with its exact split offset = target-matched change +
  execution term.
The pre-cue ENL rows (-0.19 / -12.9, from `enl_offx_variants.csv`, not recomputed) are drawn beside the CUED arm as the
reference. NOTE: ENL's angle is the 3-D first-lick angle (arcsin side_ml / reach), not theta_L.

**Predictions:**
* Aim in cortex: TI well above 0 (CI excludes 0) acutely in the PRE-LAUNCH window, in both arms (ITI has no cue to
  carry it).
* Aim error set downstream: TI ~ 0 (as pre-cue), with a CI upper bound well below 1.
* Positive control: POST-window TI > 0.
* Partial tracking (0 < TI < 1, CI excluding both) is possible and reported as such.

## 6. Dynamics, jaw, and secondary analyses (descriptive; no correction unless stated)
* **Per-bin tracking**:
  1. EXACT per-bin decomposition of the primary decoder's Δ d_dec* (linear: Σ_bins w_b · Δ x_b, scaled through the
     scaler / PCA; ENL B3's form), per arm x epoch.
  2. **Single-bin decoders**: one Allen decoder per 0.05-s bin from -0.80 to +0.30 s (BASE-subtracted bins are 0 inside
     BASE, so the BASE bins use raw levels), each frozen, with gate r and TI per bin. This gives a time course showing
     WHEN tracking appears.
* **Window end -0.10 s** (Priya 10-31; both arms, all feature sets): gate + TI.
* **Jaw-referenced variant (CUED only)** (Priya 10-31; jaw leads the tongue by 52-120 ms, FINDINGS 4.18):
  * Licks: cued first licks with a jaw onset after the cue, before tongue onset, not at the cue (S25's `firstjaw` rule,
    `jaw_onsets.csv` `jaw_onset_ms` + cue_s).
  * Bins: **-0.50 .. -0.05 s from JAW onset, BASE -0.80 .. -0.50 s from jaw onset**; its own frozen decoder (Allen
    primary; SVD / LocaNMF secondary); gate + TI + per-bin decomposition and single-bin indices.
  * The TONGUE-referenced decoder is also scored on exactly these licks (like-for-like subset).
  * Reading: tracking only in bins AFTER jaw onset (tongue-referenced last bins, absent jaw-referenced) = movement /
    feedback-locked, not a pre-movement aim.
  * **ITI: no jaw onsets exist** (section 0). The -0.10 s end is the only jaw-related check there, and it does not
    clear a 120-ms jaw lead. Stated as a caveat. An ITI jaw detector is not built for this analysis.
* **Far-contra only** (CUED spout far_contra; ITI previous spout far_contra), where the execution change is largest:
  Δ decoded vs Δ executed at the same target, per animal x epoch + TI.
* **Dose / offset control (key secondary)**: per animal x epoch, the session x target-cell Δ d_dec* regressed on Δ d_exe
  across the six targets WITH an intercept. A uniform decoded shift unrelated to aim (baseline / F0 / state change)
  loads on the intercept; tracking loads on the slope. Per-animal slopes + pooled. This is the cell-level analogue of
  ENL figure B7-8.
  * Also the through-origin cell-level slope as a TI sensitivity.
  * Also TI normalised by b_all.
* **Per-hemisphere decoders** (Allen, left = ipsilesional = image left; right = contralesional): gate + TI.
* **CUED-trained decoder applied to ITI licks** (frozen cued all-pre decoder, ITI reference from its ITI pre
  predictions): TI.
* **Other feature sets**: SVD (K 20; 10 / 40) and LocaNMF, gate + TI, with the same rules (not in the Holm family).
* **No-baseline** and **ITI gap >= 2 s** sensitivities; **ITI dock01** (onset >= dock + 0.1 s) and **since_stop >= 1.5 s**
  (window clear of the spout retraction).
* **Within-session refit** across-spout r (5-fold CV stratified by target, ridge, alpha by 3-fold inner CV). **FLAGGED
  BIASED near zero signal** (S25b: the `_cv_r` form, fold-wise intercepts, sits below 0). Read only as a change.
* **History**:
  * CUED reference cells = target spout x previous trial's spout (`lick_enl_offset_explore` previous-trial rule).
  * Both arms: lick-level regression within post sessions of d_dec* on the current d_exe and on the PREVIOUS lick's
    theta_L deviation (the GCaMP6s residue of a deviated previous lick can make the window "track" without any aim
    signal; the means cannot separate them because both are shifted).
* **Face movement**: FaceRhythm master factors (`iti_licks.fr_unblanked`; 25 Hz in trials, ~5 Hz in the ITI, a caveat),
  averaged over the PRE window, on sessions with a projection:
  1. a face-only decoder (same pipeline) and its TI = the behavioural reference;
  2. the neural decoder with features residualised on the face factors (regression fitted on pre training sessions,
     frozen with the decoder) -> TI.
* **RT (CUED)**: post-stroke RTs shift the cue response within the tongue-referenced bins. TI within pre-defined RT
  terciles (pre August cut points).
* **POST window with landing theta_E** as the executed angle (the contact-driven signal should follow landing).

## 7. Interpretation rules (fixed)
* Gate failed (arm, Allen PRE) -> that arm's TI is descriptive; no aim conclusion from it.
* POST positive control failed (arm) -> "the method cannot read executed angle here even after contact"; a PRE TI ~ 0
  in that arm is uninformative.
* **"Cortex carries the re-aimed launch" (PROVISIONAL at most)** requires all of the following:
  1. the gate and POST control pass;
  2. the primary acute (or subacute) numerator meets the tracking criterion;
  3. the dose control shows a positive slope, not an intercept;
  4. the effect is present in the window ending -0.10 s and, for CUED, in the jaw-referenced window.
* **"Aim error set downstream / not visible in cortex"** requires: gate + POST pass, PRE TI CI includes 0 with upper
  bound < 0.5, in both arms.
* Anything else = INCONCLUSIVE, stated with the per-animal table.
* Tracking only in bins after jaw onset -> movement / feedback-locked, not an aim.
* n = 4 animals throughout: NOT ESTABLISHED / PROVISIONAL language only.

## 8. Statistics, figures, outputs
* Session loop via `analysis_kit.fan_sessions`, **<= 4 workers**, free RAM checked first. Seeds fixed (`SEED = 0`;
  per-cell seeds from animal / arm / set / window).
* Bootstraps via `analysis_kit` (`boot_delta`, `boot_delta_columns`, `boot_p`, `holm`, `boot_ci` for levels).
* **Figures** (Priya's defaults):
  * sessions = OPEN markers, shape = animal (PS92 o, PS93 s, PS94 ^, PS95 D), colour = epoch
    (`orofacial_cohort.EPOCH_LINE`);
  * pooled = FILLED epoch-colour marker with 95 % CI;
  * no figure numbers in titles / images (panel letters only); no lines between epochs;
  * stars in the top band (`within_lick_correction.star_top` / `reserve_headroom`);
  * nothing overlapping data or neighbouring panels.
* **Primary figure**:
  * **A** pre gate per animal x arm x feature set (r vs null 95th pct);
  * **B** calibration (pre decoded vs actual, cell means, b_sp);
  * **C** TI by epoch, CUED pre-launch (+ ENL pre-cue reference);
  * **D** same ITI;
  * **E** POST-window positive control;
  * **F** far-contra Δ decoded vs Δ executed at the same target.
* Secondary figure: per-bin time course (tongue- and jaw-referenced), dose control, sensitivities.
* Outputs:
  * `features/<label>.npz`, `extract_summary.csv`, `lick_table.csv.gz`;
  * `gate.csv` / `gate.json`, `calibration.csv`, `frozen_status.csv`, `licks_pred.csv.gz`, `sessions.csv`;
  * `per_animal.csv`, `tracking.csv`, `primary_holm.csv`, `secondary_*.csv`;
  * `figures/`, `figdata/prelaunch_aim_tracking.pkl`, logs.

## 9. Caveats known now
* n = 4. PS95's ITI arm is thin (163 pre licks; 1 acute session with 28 licks).
* GCaMP6s lags spiking by ~0.1-0.2 s, so the PRE window reflects activity up to ~0.3-0.5 s before onset. It also carries
  the decaying response of the previous lick (>= 1 s earlier).
* CUED: the window contains the cue response at RT-dependent lags, and BASE contains the cue for RT > 0.5 s.
* The jaw leads the tongue by 52-120 ms; ITI has no jaw reference.
* Docked ITI windows can overlap the end of the spout retraction (onset 0.91 s after stop at the 5th pct; dock reached
  0.65-0.98 s).
* The calibration slope's own uncertainty is not propagated into the TI interval (reported per animal instead).
* Pre-stroke, cortex shows no within-spout launch signal (S25 / S25b). So any tracking must come from a decoder that
  reads across-target structure, i.e. TI > 0 needs the post-stroke deviated launches to look, in cortex, like launches
  to the target they resemble.
* FaceRhythm in the ITI is ~5 Hz.
* U_atlas registration jitter blurs the SVD features across sessions.

## Outcome
(empty -- filled after the run, gate result first)

## Deviations
(none yet)
