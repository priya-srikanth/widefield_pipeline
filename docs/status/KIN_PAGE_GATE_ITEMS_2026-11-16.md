# Kinematics page: working-period gate, Priya's 11-16 answers (running list 11-06 section A items 1-3)

Branch `fix/kin-gated-pooled-1116` (from origin/main 98d2a627; no code change needed). Page = "Licking Kinematics After
Stroke" (https://claude.ai/artifact/DKcKRJKbhBoratxxRc6qZv, version 1794081327-c0d5, read 11-16, NOT republished).
`<tag>` = `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`;
`<D>` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/session_poses_derived/<tag>`.
Scripts / raw outputs of this pass: session scratchpad `kingate/` (cmp_pooled.py, cmp_cue*.py, cmp_sess.py, s6_numbers.py,
cmp_gates.py, run_legacy.py, run_redraw.py, run_pooled.py and their .txt / .log).

## Item 1 (ACCEPTED): section 6 trial-class shares on the camera working period

Page figure -> gated file (all under `<D>`):

| page | live image | gated source (numbers, 11-06 17:06) | page-ready redraw (11-16, current main: American spelling) |
|---|---|---|---|
| Fig 6.1 | tc_share_all.png | jaw_only_incomplete_figs/figures/trial_class_share_raw.png (+ .csv, _tests_added.csv) | jaw_only_incomplete_figs_redraw_20261116/figures/trial_class_share_raw.png |
| Fig 6.1 LMM | lmm/tc_share_all_lmm.png | DeepLabCut/.../cam1_licks/trial_classes_working_1105/figures/tc_share_all_lmm.png (inputs md5-identical to the 11-06 tables) | same |
| Fig 6.2 | tc_share_all_change.png | jaw_only_incomplete_figs/figures/trial_class_share_raw_change.png | ..._redraw_20261116/figures/trial_class_share_raw_change.png |
| Fig 6.3 | tc_share_days.png | jaw_only_incomplete_figs/figures/trial_class_share_days.png | KEEP the 11-06 render (see note) |
| Fig 6.4 | tc_jaw_traces.png | jaw_only_incomplete_figs/figures/trial_class_jaw_traces.png | ..._redraw_20261116/figures/trial_class_jaw_traces.png |
| Fig 6.5 | tc_tongue_traces.png | jaw_only_incomplete_figs/figures/trial_class_tongue_traces.png | ..._redraw_20261116/figures/trial_class_tongue_traces.png |
| Fig 6.6 | t_jaw_PS94.png | pose3d_cue_traces_cue/figures/cue_jaw_open_PS94.png | - |

Redraw = `jaw_only_incomplete_figs --reuse --out <new>` on copies of the 11-06 sessions.csv / traces.csv.gz +
`trial_class_raw_share_fig --all-engaged <new> [--change]`. Share / change tables byte-identical; tests.csv differs only by
the 4-significant-digit round trip (|d| <= 1e-4, 0 star flips). **Fig 6.3 note:** the time model is numerically fragile
-- after the round trip the far-mid incomplete random-effects ladder picks ~post + post:lday instead of ~post and its
step / slope lose q < .05 (q 1e-4 -> .15, 1e-18 -> .07). Keep the full-precision 11-06 render (and the existing caveat
"use Figs 6.1-6.2 for inference").

Not re-gated (outside section 6 class shares; listed only): section 6 per-animal "jaw movement and no lick" table and
section 7 "no-contact share" table and Fig 7.6 (cam1_licks/incomplete, PI-deck source); Figs 1.3 (cam4 cohort, A27).

## Item 3: pooled 3-D table (Fig 7.5, Fig 9.6, 3-D rows of the section 3 / 5 / 7 tables)

Already fixed and re-run on 11-07 (Priya A3 yes): `ec67a0d5` / main `96f77156`, queue v9x `k_pooled_r` ->
`<D>/kinematics_pooled_figs/` (09:58). Old = `DeepLabCut/Widefield/session_poses/pose3d/figures/pooled/` (10-30 / 11-03,
ungated 10-23 sessions.csv). 11-16: re-rendered with current main (11-07 epoch-color convention) into
`<D>/kinematics_pooled_figs_20261116/`.

Old (page) vs gated, `pose3d_change_pooled_contra.csv` (96 cells) and `pose3d_days_pooled.csv` (228 cells):

* **Lick measures unchanged**: midline error <= 0.004 deg, miss <= 0.0003 mm, reach <= 0.006 mm, peak speed <= 0.05 mm/s,
  jaw FAR_DIR <= 0.0014 mm, CLOSE_DIR <= 0.003 mm, per animal and pooled; no star changes. Every page number of Fig 7.5
  (miss, reach, speed, midline, chronic miss), Fig 9.6 (FAR_DIR -0.08 at 36-70 d) and the section 3 / 5 / 7 3-D table rows
  stands. Fig 3.7 / 4.11 / 9.7 inputs: window means <= 0.3 deg, jaw <= 0.06 mm, paths <= 0.0007 mm, phase <= 0.013.
* **Widest jaw opening (`open_max_mm`, Fig 7.5 row 5, Fig 9.6 jaw-opening row) CHANGES.** Per-session values rise
  (0 to +4.0 mm) only where the gate drops stopped / post-quit trials with little or no jaw opening (largest in acute
  sessions: PS95_0820 +4.0, PS94_0819 +2.6, PS92_0820 +1.4, PS93_0820 +1.3; pre +0.05 to +0.3). Far contra acute (PS92 /
  PS93 / PS94 / PS95; pooled [95 % CI]): -1.62 / -1.69 / -2.79 / -3.59; -2.42 [-3.29, -1.59] -> -1.01 / -1.13 / -1.87 /
  -3.38; -1.85 [-2.86, -1.05] (still 4 / 4, CI excludes 0). Stars LOST: far contra subacute -0.57 [-1.06, -0.15] -> -0.27
  [-0.65, +0.01]; far ipsi acute -0.99 -> -0.42 [-1.13, +0.14]; far ipsi subacute -0.53 -> -0.27; close ipsi subacute
  -0.25 -> -0.11. Fig 9.6 far contra jaw opening by day bin 1-3 / 4-7 / 8-14: -2.10 / -1.41 / -0.32 -> -1.44 / -0.61 /
  -0.20 (PS95 days 4-7 -1.80 -> +0.07).
* `cam4 angle` (Fig 3.6, `angle_change_pooled_contra.csv`): pooled <= 0.12 deg, per animal <= 0.47 deg, no star change
  (far contra acute -29.1 -> -29.2).

## Item 2: close-spout incomplete rise + legacy-gate sensitivity row

Cause (CLOSE_INCOMPLETE_CAUSE_20261106, commit 52b5dc18): no trial changed class; the working period keeps the
post-DAQ-quit tail up to the last camera lick, where the animal keeps licking without contact. Close-spout incomplete
trials, gated vs legacy, summed over sessions: pre 22 vs 15, acute 87 vs 28, subacute 70 vs 24, chronic 36 vs 7 (+141;
PS94 acute 57 vs 18).

Sensitivity row = `WFIELD_KIN_GATE=legacy` (engaged trials, `orofacial_cohort.engagement_keep`) on current origin/main,
production profile: `<D>/jaw_only_incomplete_figs_legacygate/` (tables + figures; 107 sessions, 16 acute). It
reproduces the live page's (11-03) tests to 0.004 with one star difference (jaw-only far ipsi chronic -0.009 * vs n.s.).

| close-spout incomplete, change vs own pre | close ipsi | close mid | close contra |
|---|---|---|---|
| working period, acute | +0.014 * | +0.017 * | +0.021 |
| engaged trials (legacy), acute | +0.002 | +0.003 | +0.005 |
| working period, subacute | +0.014 ** | +0.013 * | +0.013 * |
| engaged trials (legacy), subacute | +0.006 | +0.005 | +0.005 |
| working period, chronic | +0.002 | -0.000 | +0.002 |
| engaged trials (legacy), chronic | -0.001 | -0.001 | -0.001 |

Per animal, acute (PS92 / PS93 / PS94 / PS95): close ipsi working +.012 / +.016 / +.029 / .000 vs legacy +.005 / +.006 /
-.002 / .000; close mid +.023 / +.017 / +.032 / -.004 vs +.005 / +.003 / +.008 / -.003; close contra +.020 / +.006 /
+.069 / -.010 vs +.005 / +.001 / +.024 / -.008.
Other cells where the two gates disagree on a star: close ipsi contact subacute (-0.040 ** vs -0.024), far mid incomplete
acute (** vs *), jaw-only close contra acute (* vs **), close contra no response acute (+0.020 * vs +0.039 **), and the
chronic below-pre cells (no response close ipsi / far ipsi / far contra, jaw-only far ipsi / far contra: * under legacy,
n.s. on the working period).
