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

Re-render 11-16 (`<D>/kinematics_pooled_figs_20261116/`): a full recompute (`kinematics_pooled_figs --out`) was stopped
after 2.5 h -- profiled: re-gating 119 sessions is network-bound (nt.stat ~17 ms each inside `config.load_sessions` /
`PathResolver`, ~25-65 s per session today vs ~40 min total on 11-07). Instead the 11-07 gated per-session table
(`<D>/kinematics_pooled_figs/pose3d_sessions.csv`, written by the same `sessions3d()`; trial-class code digest unchanged
since) was preloaded into the script's memo and only the figures were drawn (6.2 min). Every pooled CSV equals the 11-07
gated run to <= 6e-14. Page files: Fig 7.5 = pose3d_change_pooled_contra.png, Fig 9.6 = pose3d_days_pooled_contra.png,
Fig 3.6 = angle_change_pooled_contra.png, Fig 3.7 = pose3d_cue_pooled_contra.png, Fig 4.11 =
pose3d_paths_pooled_contra_cam.png, Fig 9.7 = pose3d_phase_pooled_contra.png (open markers: shape = animal, color =
epoch).

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

## Item (d), 11-16 (branch fix/todo-1116): section 6 jaw-only table, section 7 no-contact table, Fig 7.6 re-gated

Gate = the section 6 class shares' working period: `canonical` trials of the production camera run's per-trial table
(`rebuild_20261103/LP_analysis/<v36>/imaging_encoding/qc/sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70/trial_attempts.csv`,
11-04). Checked first: its canonical trials reproduce the gated `<D>/jaw_only_incomplete_figs/sessions.csv` exactly (642 / 642
session x spout cells, identical n_trials, class shares within the 4-digit rounding). Code: `cam1_incomplete_licks.py
--working <that csv> --out <dir>` (commit c6f7a924; trials, their cam4 / cam1-only licks and the run's jaw-only class;
v36 trial ids / positions asserted). Run 46 s (0.2 s per session; no per-session re-gating). Fig 7.6:
`fig_farcontra_attempts_sessions.py --incomplete <new sessions.csv> --out <new>` under profile cohort1_rebuild_20261103.

Outputs (new only): `<D>/cam1_incomplete_working_20261116/` (sessions.csv, by_epoch.csv, figures/) and
`<D>/farcontra_attempts_sessions_working_20261116/` (farcontra_attempts_sessions.png / .csv, data/).

Far contra (far_R), mean of session values per animal (PS92 / PS93 / PS94 / PS95; mean of animals):

| measure | epoch | page (live) | 11-03 all-trial table | working period (new) |
|---|---|---|---|---|
| jaw moves, no lick (section 6 table) | pre | .016 / .009 / .050 / .075 | .014 / .008 / .049 / .073 | .009 / .000 / .016 / .024; .012 |
| | acute | .277 / .282 / .387 / .577 | .201 / .209 / .278 / .475 | .094 / .100 / .205 / .486 (1 session); .221 |
| | subacute | - | .056 / .092 / .094 / .140 | .068 / .027 / .033 / .053; .045 |
| | chronic | .000 / .022 / .037 / .065 | .000 / .021 / .036 / .062 | .000 / .003 / .011 / .020; .008 |
| licks without contact (section 7 table) | pre / acute / chronic | unchanged | | per session <= 0.0009 (no 3-digit change) |

* The page's section 6 jaw-only table and Fig 7.6 jaw panel are the pre-11-03 jaw-QC'd-trial version (mean .04 / .38 / .11
  / .03); the working period gives .012 / .221 / .045 / .008 = the jaw-only row of Fig 6.1 (same trials). Acute still
  above pre in 4 / 4 and back near pre chronically; the acute size roughly halves in PS92 / PS93 (most jaw-only trials
  there were in the quit tail / stopped runs); PS95 (one session) .486.
* Licks without contact: dropping the outside-period trials removes almost no licks (8,494 of 9,481 far_R trials kept;
  largest change PS93_0821 .820 -> .819), so the section 7 table and Fig 7.6 panel 3 stand (.25 / .89 / .48 / .39).
* Fig 7.6 attempt rate / completion: unchanged by this gate (already the run's camera classes), but the page image
  predates the rebuild: rebuild values attempt .980 / .676 / .932 / .985 (page .97 / .65 / .93 / .99; acute per animal
  .842 / .832 / .663 / .367, page .79 / .80 / .65 / .34), completion .977 / .068 / .796 / .978 (acute .053 / .152 / .042
  / .025, page .05 / .15 / .04 / .02). Caption's "nothing re-run" / "engaged trials" / "all-trial" set sentences need
  the working-period wording.
