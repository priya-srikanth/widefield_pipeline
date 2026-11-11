# Page text draft from the glue-fixed rebuild re-run, for Priya's review (2026-11-10, late)

DRAFT ONLY. No page, deck or M: file was changed, and nothing here is published. The live pages still show
PRE-rebuild results; this text replaces them only after Priya signs off at the rebuild page pass.

**Source of every number.** Queue v20 glue re-run (`_g` steps), code main 8aed3a7b, profile cohort1_rebuild_20261103.
Only steps with a `.done` marker are used. `$LOG` =
`C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/render_daqlk/logs`.
`$ENC` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`.
`$ITI` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`.
`$LPFM` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/LPfinal/LP_licks_gateLP_hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk_refs-mean+rest+restw+precue+raw+restnl/epoch`.

**Conventions applied.** Every number is a change from each animal's OWN pre (August pre, 8 / 7 / 8 / 8 sessions for
PS92 / 93 / 94 / 95; PS93_0813 has no video). Per-animal values come before the pooled value. `*` = the 95% CI of the
paired nested bootstrap (`analysis_kit.boot_delta`, animals -> sessions) excludes 0; no LMM daggers are used here.
Post sessions: acute 5 / 4 / 6 / 1, subacute 2 / 3 / 5 / 7, chronic 12 / 12 / 8 / 11 (PS95 has one acute session).
Canonical spout order. On brain maps, image left = ipsilesional. Paste-in sentences A1 / A4 / A6 / A8 / A9 are the ones in
`docs/status/CAPTION_PASS_DRAFT_2026-11-10.md` (branch docs/caption-pass-draft-1110) section A.

**Live pages.** GOAL = https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (read 11-10). IMG = imaging page
https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV (read 11-10). Figure numbers are the live ones.

**Each family has a verdict line.** "Same conclusion" = the live claim holds on the glue-fixed rebuild numbers.
"Wording change for sign-off" = the glue did not change it, but the REBUILD changed a stated claim (sign-off needed
before paste, as for the 11-09 headline changes).

---

## D1. Frozen encoder: target vs executed, and trial-to-trial vs PETH, in the glue-free areas (GOAL section 0)

**Replaces:** GOAL Figs 0.3 (`frozen_claims_contest_cue`), 0.9 (`..._contest_lick`), 0.4 (`frozen_claims_peth_cue`),
0.10 (`..._peth_lick`): the split-row (ALM / tjM1 / MOs / MOp) and cortex sentences of the captions, and the
"Less spout-specific tuning" bullet under "Where things stand".
**Step:** `z_frozen_claims_g` (.done 11-10 16:38; `$LOG/z_frozen_claims_g.log`). Numbers from
`$ENC/frozen_fr_compare/claims_contest.csv` and `claims_peth.csv` (written 16:16 / 16:35 by this step).
**Figures:** `$ENC/frozen_fr_compare/figures/frozen_claims_contest_cue.png`, `..._contest_lick.png`,
`..._peth_cue.png`, `..._peth_lick.png`.
**Verdict:** wording changes for sign-off (a-c below). The glue left this step unchanged: 0 star changes
(GLUE_BEFORE_AFTER_2026-11-10_2130.md, coverage row).

**Draft caption (Fig 0.3; 0.9 is the same with "lick window"):**
"Fig 0.3 · Target vs executed unique R², frozen pre-stroke encoder, cue window. Model: one encoder per animal, fit on its
August pre sessions (31 sessions; pre scored leave-one-session-out) and applied unchanged to every post session; unique R²
of a group = R²(joint) - R²(joint without that group). Gray = no FaceRhythm; pink = with FaceRhythm (mode A). Window: cue,
0-1.5 s after the cue (lick window: -0.1..1.0 s from the first lick). Trials: attempted trials (camera contact + incomplete
(LP licks), within the working period (first-last camera lick), stopped runs (A2 + lick excise) and quit tail removed;
not: jaw-only, no response, unknown, early-run) [encoder gate OROFACIAL_GATE=attempted]. Licks: lick-onset kernels = LP
video licks (cam4 + cam1-only events); contact kernel = DAQ spout contacts. Baseline: change from each animal's own pre.
Stats: bars = mean of animals' epoch - pre, whisker = 95% CI of the paired nested bootstrap (animals -> sessions), * = CI
excludes 0, dots = animals; no multiple-comparison correction. Mask: the painted fiber-glue region is excluded; Allen
areas lying entirely under glue are not analyzed. n = 4 animals; sessions pre 31 / acute 16 / subacute 17 / chronic 43."

**Result text (numbers = epoch - pre, no FaceRhythm, contralesional = right hemisphere):**
- Target unique falls ACUTELY in every frontal area, 0 / 4 animals up: ALM -.019 [-.028, -.009]* (cue; per animal
  -.008 / -.022 / -.026 / -.018), MOs -.012*, MOp -.012*, tjM1 -.019 [-.042, -.001]* (1 / 4 up); lick window ALM -.022*,
  MOs -.019*, MOp -.020*, tjM1 -.022*. CHRONICALLY it stays down only in MOs (cue -.012 [-.020, -.005]*, per animal -.019 /
  -.012 / -.011 / -.007, 0 / 4 up) and MOp (cue -.009*, 0 / 4); contralesional ALM is back (-.003 [-.013, +.010], 1 / 4 up)
  and so is tjM1 (-.002, n.s.). Whole cortex: cue chronic -.010 [-.017, -.005]* (0 / 4 up), lick chronic -.010 [-.021,
  +.004] (1 / 4 up).
- The executed movement's spout-locked part RISES acutely in 4 / 4 animals in every frontal area and SSp-m (cue: ALM
  +.040*, MOs +.034*, MOp +.044*, tjM1 +.034*, SSp-m +.033*) and stays up CHRONICALLY in 4 / 4 in contralesional ALM
  (+.025 [+.015, +.035]*; per animal +.031 / +.020 / +.036 / +.014), tjM1 (+.017*), MOs (+.015*), MOs-rest (+.011*) and
  MOp (+.012*; MOp-rest +.010*, 3 / 4). With FaceRhythm the chronic gain is smaller and starred only in MOs (+.009*,
  4 / 4) and MOs-rest / MOp-rest (cue).
- PETH (Fig 0.4 / 0.10): the target PETH unique falls chronically in contralesional MOs (cue -.055 [-.107, -.000]*,
  0 / 4 up; lick -.066*) and in whole cortex in the lick window (-.068 [-.107, -.025]*, 0 / 4 up; with FaceRhythm -.055*);
  contralesional ALM cue chronic -.031 [-.072, +.003] (0 / 4 up, n.s.). Movement PETH unique RISES chronically in 4 / 4 in
  contralesional ALM (+.126*), tjM1 (+.096*), MOs (+.069*), SSp-m (+.070*) and ipsilesional tjM1 (+.062*) (cue).

**What changed vs the live page:** trials = attempted gate on the rebuilt camera classes (live: "LP lick or DAQ contact,
trial_attempts.csv"); August-only pre with PS93 at 7 sessions; glue mask. Three stated claims change:
(a) live "target unique falls in ALM as in MOs (contralesional chronic -0.018 [-0.029, -0.010], 0/4)" -> chronic ALM loss
gone (-.003, 1 / 4); the chronic loss holds in MOs / MOp only. (b) live "executed spout-locked gain is chronic and 4/4
in contralesional ALM (+0.031) and tjM1 (+0.024), NOT in MOs-rest / MOp-rest (n.s.)" -> MOs-rest and MOp-rest are now
also starred (+.011 4 / 4, +.010 3 / 4): the chronic gain is frontal-wide, not ALM / tjM1-specific. (c) live "movement
PETH unique rises 4/4 only in tjM1" -> also ALM, MOs and SSp-m chronically. Cortex lick target chronic (live -0.016*,
0/4) is now n.s. (-.010, 1 / 4); the 11-08 headline table already had this (C6, -0.0096 n.s.). PETH cortex lick chronic
stays starred (live -0.096 -> -.068).

---

## D2. Per-spout target vs executed unique maps, per-spout lick-kernel frame (GOAL Fig 1.7)

**Replaces:** GOAL Fig 1.7 (`ls_spout_unique_maps`) caption (the live caption has no result numbers).
**Step:** `z_spout_unique_maps_g` (.done 11-10 17:51; `$LOG/z_spout_unique_maps_g.log`, table of significant bins).
**Figures:** `$ENC/frozen_encoder_gate-attempted_v7g3_t3d_spont_frA_ls_basis/figures/spout_unique_maps.png` (+
`spout_unique_levels.png`, `spout_unique_per_animal_target.png`, `..._executed.png` as supplements).
**Verdict:** same conclusion as Fig 1.1's reading (executed changes acutely and spout-specifically, target little). The glue
re-run moved no star (4 count changes > 25%, 0 star changes; GLUE_BEFORE_AFTER coverage row).

**Draft caption:**
"Fig 1.7 · Per-spout target vs executed unique R² maps, frozen pre-stroke encoder with per-spout lick kernels and
FaceRhythm (mode A). Per spout, on its own trials: target unique = loss when that spout's target response is folded into
the others'; executed unique = the further loss when its spout-locked movement templates and within-spout direction are
also swapped (the per-spout lick-onset kernels are not swapped). Window: all encoder frames of the trial. Trials:
attempted trials (A9). Licks: lick-onset kernels = LP video licks (cam4 + cam1-only events); contact kernel = DAQ spout
contacts. Baseline: epoch minus each animal's own pre (August pre, held out). Plot: mean of 4 animals after scaling each
by its own pre 99th percentile (scales: target .007 / .002 / .015 / .009, executed .033 / .022 / .155 / .065 for PS92-95).
Stats: contours = significant 16 px bins on the raw maps, nested bootstrap animals -> sessions, max-statistic over the
514 bins of each panel (family-wise per panel); corner = bins up / down. Mask: the painted fiber-glue region is excluded.
Canonical spout order; image left = ipsilesional. n = 4 animals."

**Result text:** Executed unique changes acutely and spout-specifically: far contra RISES (29 of 514 bins up) while far
middle FALLS (27 down); the other spouts move little (close ipsi 2 up, close middle 4 down, close contra 1 up, far ipsi
2 down). Chronically the far spouts are back (far contra 0, far middle 5 down) and close ipsi is mixed (27 up, 14 down).
Target unique changes little: acutely 2 bins down at far contra and 1 at far middle; chronically 16 down at far middle and
4 up at close ipsi.

**What changed vs the live page:** live Fig 1.7 had no numbers. Rebuild inputs: attempted gate on camera classes, August
pre (PS93 7 sessions), glue mask. Note for the page: Fig 1.1 (no-FaceRhythm frame, live "far contra 52 up") and Figs
1.8-1.10 (no-FaceRhythm levels) were NOT re-run in v20 under their own tag; this step writes levels / per-animal figures
in the FaceRhythm lick-kernel frame only (see HELD).

---

## D3. Pixel R² maps and R² share maps, frozen encoder without FaceRhythm (GOAL Figs 20.3-20.5)

**Replaces:** GOAL Figs 20.3 (`fig15c_r2_chronic`), 20.4 (`fig15_r2_pre`), 20.5 (`fig15b_r2_acute`) images and captions;
adds the share view.
**Step:** `b_r2_maps_g` (.done 11-10 15:35; the log prints no readouts). Numbers below were computed for this draft from the
step's own `$ENC/frozen_encoder_gate-attempted_v7g3_t3d_spont_basis/figures/r2_maps.npz` (per animal: mean over the
32,450 map pixels with a value of epoch - pre; descriptive, no test). Please confirm you want these in the caption.
**Figures:** same folder: `r2_maps_pre.png`, `r2_maps_{acute,subacute,chronic}_minus_pre.png` (+ `_peranimal`),
`r2_share_pre.png`, `r2_share_{acute,subacute,chronic}_minus_pre.png` (+ `_peranimal`).
**Verdict:** same conclusion (full R² lower over most of cortex chronically; per-animal patterns differ). No glue-in copy
exists to compare (outputs overwritten in place).

**Draft caption (Fig 20.3; 20.5 = acute):**
"Fig 20.3 · Chronic minus pre pixel R² maps, frozen pre-stroke encoder without FaceRhythm (basis targets: data rebuilt in
each animal's 200-component pre basis). Full R² and each group's unique R² (= SSE without the group minus SSE of the joint
model, / SST); SSE and SST summed over the epoch's sessions per animal, then the mean of animals. Pre sessions scored with
leave-one-session-out betas, post sessions with the frozen all-pre betas. Frames: all imaging frames of the encoder
trials. Trials: attempted trials (A9). Licks: lick-onset kernels = LP video licks (cam4 + cam1-only events); contact
kernel = DAQ spout contacts. Baseline: epoch minus each animal's own pre. Stats: none (descriptive maps). Share view
(`r2_share_*`): unique R² / that animal's full R² (pixels with full R² < 0.02 blank). Mask: the painted fiber-glue region is
excluded. Image left = ipsilesional. n = 4 animals."

**Result text:** Pre-stroke full R² averages .55 / .56 / .54 / .43 (PS92-95, map mean). Acutely full R² falls in every
animal (-.11 / -.19 / -.09 / -.12; the mean map is below pre in every pixel), mostly through movement (-.13 / -.12 / -.02 /
-.13) and "other" unique R². Chronically it stays lower in 3 / 4 (-.18 / -.11 / +.01 / -.17; PS94 back at pre), lower
than pre in 93% of pixels of the mean map; PS95 also loses lick (-.046) and spout (-.015) unique R².

**What changed vs the live page:** rebuild gate / pre / glue mask as D1; the live caption had no per-animal values; new
PS94 statement (no chronic full-R² loss) and the share view are new.

---

## D4. ITI lick direction decoding and ITI ramp timing (GOAL Fig 10.3; timing sentences of Fig 14.1)

**Replaces:** GOAL Fig 10.3 (`iti_direction_decoding`) caption numbers; the build-up / timing reading in Fig 14.1
(`iti_ramp_sessions`). Fig 14.22's AUC numbers are NOT drafted (see HELD).
**Step:** `e_iti_main_g` (.done 11-09 20:03; `$LOG/e_iti_main_g.log`, tables "decoding change vs pre", "frozen decoder
change vs pre", "ramp change vs pre", per-animal blocks).
**Figure:** `$ITI/figures/lick_direction_decoding.png` (written 11-09 20:02 by this step).
**Verdict:** same conclusion. Glue: 0 star changes; two n.s. sign flips (per-session ITI direction acute / subacute,
-.009 -> +.002 and -.011 -> +.003) and two n.s. ramp sign flips (borderline list B).

**Draft caption (Fig 10.3):**
"Fig 10.3 · Lateral tongue-tip direction of ITI and in-trial licks. Decoders: left / middle = PER-SESSION (refit within
each session, cross-validated); right = FROZEN pre-stroke (fit once on each animal's August pre ITI licks, never refit;
pre scored leave-one-session-out). Measured: r between decoded and actual lateral position (3-D, + = contralesional),
lags 0-0.2 s from the lick peak (curves: every lag). Events: bout-start ITI licks (LP video licks, cam4 detector, 3-D
position from cam1 + cam4 pose; no trial, no cue, spout retracted); per animal pre / acute / subacute / chronic
1809 / 1180 / 364 / 2255 (PS92), 3177 / 1105 / 778 / 3102 (PS93), 2494 / 880 / 1382 / 2107 (PS94), 427 / 65 / 337 / 1679
(PS95); in-trial set = post-cue licks matched in number. Relative = direction relative to the previous trial's spout (ITI)
or the spout in place (in-trial). Features: z-scored Allen areas with the ALM / tjM1 split. Baseline: change from each
animal's own pre. Stats: paired nested bootstrap (animals -> sessions), 95% CI; per-bin bars = BH-FDR q = .05 within each
curve. Mask: the painted fiber-glue region is excluded; glue-only areas are not features. n = 4 animals."

**Result text:** Within session, cortex still decodes ITI lick direction as before: pre r .63 / .51 / .65 / .41 (PS92-95);
pooled change acute +.002 [-.072, +.120], subacute +.003, chronic +.012 [-.052, +.113] (n.s.). ITI direction relative to
the previous spout is decoded BETTER chronically (+.088 [+.043, +.152]*). The FROZEN pre-stroke decoder loses ITI
direction in every epoch in 4 / 4 animals: per animal pre -> chronic .66 -> .61, .53 -> .45, .66 -> .43, .65 -> .48;
pooled acute -.130 [-.177, -.083]*, subacute -.122*, chronic -.130 [-.208, -.058]*. Frozen relative direction:
acute -.110*, subacute -.064*, chronic -.041 [-.104, +.051] (n.s.).

**Ramp timing (Fig 14.1 text; `e_iti_main_g` "ramp change vs pre"; bout-start ITI licks, dF/F per area):** the rise
before the lick starts LATER and the early ramp (-0.6..-0.2 s, / the animal's pre amplitude) is weaker chronically in
orofacial cortex. t50 (s from lick onset; + = later): contralesional tjM1 pre -.29 / -.23 / -.68 / -.22 -> chronic
-.01 / -.04 / -.09 / -.13 (4 / 4 later), pooled +.291 [+.124, +.500]*; also later acutely (+.217*) and subacutely (+.242*).
Early ramp, chronic: tjM1 contra -.400 [-.704, -.083]* and ipsi -.340*, SSp-m contra -.486* and ipsi -.392* (3 / 4 animals
each); whole cortex -.139 n.s. The post-lick peak (normalized) is up acutely in 4 / 4 in every orofacial area (tjM1
contra +1.43*, SSp-m contra +1.61*).

**What changed vs the live page:** frozen ITI loss -0.15 / -0.13 / -0.14 (live) -> -.130 / -.122 / -.130; per-session
relative chronic +0.064 -> +.088; pre r .54 -> .55. Rebuild inputs: August pre (PS93 7 sessions), frozen models
re-frozen on the rebuild input set (log: "NEW model is being frozen"), glue mask.

---

## D5. Does pre-cue (ENL) activity predict the angle of the upcoming first lick? (GOAL Fig 14.21 S3)

**Replaces:** GOAL Fig 14.21 S3 (`lick_enl_angle_decode`) caption.
**Step:** `l_enl_angle_g` (.done 11-10 12:32; `$LOG/l_enl_angle_g.log`, "PRE W=1.0 lickfree=slide", "change vs pre
(paired nested bootstrap)" rows window_s 1.0, lickfree slide = PRIMARY).
**Figure:** `$ITI/figures/lick_enl_angle_decode.png`.
**Verdict:** same conclusion for every claim the live caption makes. One primary row not on the page changed star (Q5).

**Draft caption:**
"Fig 14.21 S3 · Does pre-cue (ENL) activity predict the angle of the upcoming first lick? Features: mean of every Allen
area x hemisphere (ALM / tjM1 split), each / its frame-wise SD in the session, over the ENL window. Window (primary,
sliding lick-free): lags -1.0..-0.10 s from the cue if no LP lick onset (licks_all) falls in the last 1.0 s before the
cue, else the latest lick-free 1.0 s gap after spout arrival (starting >= 0.3 s after the strobe and >= 2 s before the
cue); trials without one are dropped. Target: 3-D angle of the first post-cue lick (LP video, + = contralesional).
Trials: first-lick trials with matched 3-D kinematics (pre 2267 / 3014 / 3696 / 4445 for PS92-95). Model: ridge per
animal trained on August pre sessions (pre scored leave-one-pre-session-out), frozen and applied to every post session.
Measures: raw r; beyond-spout r (features and angle residualized within session x spout); like-for-like offset (decoded
minus what pre trials with the same actual angle decoded, Gaussian reference). Baseline: change from each animal's own
pre. Stats: paired nested bootstrap (animals -> sessions), 95% CI; nulls 200x permutation within session x spout.
Robustness: fixed 1.0 s window on trials with no lick in the last 2 s, and 0.5 s windows (same conclusions). Mask: the
painted fiber-glue region is excluded. n = 4 animals."

**Result text:** Pre-stroke, ENL activity predicts the angle weakly (r .26 / .35 / .51 / .45), far below knowing the spout
(.88 / .79 / .81 / .76); the part beyond the spout is small but above its null in every animal (r .09 / .09 / .10 / .11).
After the stroke the frozen decoder's raw r drops acutely in 4 / 4 (-.12 / -.21 / -.25 / -.10; pooled -.169 [-.248,
-.095]*) and is back chronically (-.007 [-.044, +.030]); the beyond-spout part stays lower chronically in 4 / 4 (-.063 /
-.033 / -.109 / -.088; -.073 [-.111, -.035]*). Like-for-like offset: acutely +1.45 deg contralesional [+0.74, +2.38]*,
4 / 4 (+0.98 / +2.26 / +1.83 / +0.74), gone by subacute (+0.31, n.s.) and chronic (+0.54, n.s.).

**What changed vs the live page:** PRIMARY window is now the sliding lick-free 1.0 s ENL window (Priya 11-04; live: fixed
-1.0..-0.1 s on trials with no lick in 2 s, now the robustness row); trials under the rebuild camera gate (live:
"engaged gate built from DAQ spout contacts"); August pre (PS93 7 sessions); glue mask. Numbers: raw acute -0.17 ->
-.169, chronic -0.02 -> -.007, beyond chronic -0.07 -> -.073, offset acute +1.55 [+0.67, +2.9] -> +1.45 [+0.74, +2.38].

---

## D6. Far-contra map amplitude, cue and lick windows, LP arm, RESTnl and session mean (IMG section 7)

**Replaces:** IMG Figs 7.10b (cue, RESTnl, LP) and 7.18b / 7.19b (lick, RESTnl, LP, pooled / per animal); the two status
rows "Far-contra map amplitude roughly halves acutely, back by chronic" and "RESTnl version of the rest-referenced maps"
(cue and lick parts only; the pre-cue part is HELD). New session-mean twins (D2 rule).
**Steps:** `g5_lpfinal_maps_g` (.done; log ends 11-10 02:41; per-animal tables "far-contra amplitude / pre per animal
(cue/working)" and "(lick/lick)") and `g5_restnl_maps_lp_g` (.done; same acute values). Pooled amplitude and
significant bins from `$LPFM/data/epoch_15r_position_{RESTNLref,RAWref}_{cue_working,lick_lick}_stats.csv`.
**Figures:** `$LPFM/epoch_15r_position_RESTNLref_cue_working.png`, `..._RAWref_cue_working.png` (session mean),
`..._RESTNLref_cue_working_period.png` (supplement), `..._RESTNLref_cue_jaw_only.png` (jaw-only panel);
`..._RESTNLref_lick_lick.png`, `..._RAWref_lick_lick.png`; per animal `epoch_15rpa_position_{RESTNL,RAW}ref_by_animal_*`.
**Verdict:** RESTnl: same as the agreed 11-09 wording #7 (acute far-contra drop in 3 / 4, PS95 none). Session mean: PS95
also below pre (cue .80, lick .93) -> wording question Q6 before paste.

**Draft caption (cue; the lick figure swaps Window and Trials):**
"Fig 7.10b · Cue-window far-contra map amplitude, LP arm, RESTnl reference (twin: session mean). Window: 0-2 s after the
cue (lick figure: 0-2 s from the first LP lick). Trials: camera contact + incomplete (LP licks), within the working period
(first-last camera lick), stopped runs (A2 + lick excise) and quit tail removed; excluded: jaw-only, no response, unknown
(lick figure: lick trials, LP licks: a video lick, contact or incomplete (cam4 + cam1) within 3.5 s of the cue; excluded:
no-lick trials). Supplement row: all working-period trials; jaw-only panel pooled per animal x epoch. Reference (primary):
RESTnl = lick-free docked rest, one flat level per session (A6); twin: session mean (maps: zero = the slow inter-trial
trend removed by the order-10 robust detrend). Measured: per-animal map amplitude ratio, epoch / own pre (each animal /
its pre RMS, within-animal pooled). Pre: August, 31 sessions. Stats: nested bootstrap animals -> sessions (2,000 draws) on
2,022 bins (8x downsampled, eroded statistics mask), family-wise by max statistic (contours). Mask: the painted fiber-glue
region is excluded; Allen areas lying entirely under glue are not analyzed. Image left = ipsilesional. n = 4 animals;
PS95 one acute session."

**Result text (cue, working trials):** acute far-contra amplitude / pre, RESTnl .43 / .50 / .75 / .98 (PS92-95): below
pre in 3 / 4, PS95's single acute session shows no drop; session mean .14 / .13 / .40 / .80. Pooled RESTnl .65 (362 of
2,022 bins significant); session mean .23 (827 bins). Chronic RESTnl 1.00 / 1.21 / .70 / 1.17 (pooled 1.00; PS94 still
below), session mean 1.30 / 1.14 / .99 / 1.01. **Lick window (lick trials):** acute RESTnl .52 / .58 / .88 / 1.06 (3 / 4
below; pooled .75, 284 bins); session mean .22 / .30 / .46 / .93 (pooled .39, 591 bins); chronic RESTnl 1.03 / 1.17 / .71 /
1.14.

**What changed vs the live page:** live 7.10b / 7.18b used "working trials = licking trials plus misses while still
working" with the DAQ-contact / LP gate and REST twins; now camera working trials (cue) and LP lick trials (lick), RESTnl
+ session mean, glue mask (2,022 bins vs 2,609). Per-animal acute cue RESTnl was not on the live page as such (live
"cue .53, 4/4" was the contact gate); lick RESTnl per animal live -50 / -45 / -18 / ... -> -48 / -42 / -12 / +6%
(= HEADLINE M-lick .52 / .58 / .88 / 1.06, unchanged by the glue). Pooled lick RESTnl live .72 -> .75.

---

## HELD (no text drafted)

- CD overlap / geometry, IMG Figs 9.1-9.11: HELD pending Priya (PS95 overlap no longer below its null; `cd_overlap_null_g`, `_lp_g`).
- Frozen-encoder VIS / SSs area stars (f_frA_g per-area tables; GOAL Figs 2.x area maps, 0.2 / 0.8 area rows): HELD pending Priya.
- Re-embedding (state-space page sections 9-11; `l_reembed_g`): HELD pending Priya.
- ITI in-trial relative frozen decoder (`e_iti_key_g`) and with it GOAL Fig 14.22's AUC numbers (same step): HELD pending Priya.
- LP lick map 14m / 14z clusters (`g5_epoch_lp_lick_g`; IMG Haufe 14.x lick rows): HELD pending Priya.
- Pre-cue far-contra map amplitude (IMG 7.3b / 7.4b, headline #4): HELD pending Priya (references disagree, Q7).
- O2 combined-model section (GOAL contest / refit / Musall figs from `enc_rebuild_20261110g`, SECTIONS=O2 lane running): HELD pending Priya.
- Partial refits (GOAL Figs 1.3, 1.4, 1.12; `b_partial_frA_g`, `a_partial_ls_g`, `t_partial_te_g` - te still on O2): HELD pending Priya.
- Frozen relabel (GOAL section 7; `a_relabel_*_g` only `.done_stale`, re-run): HELD pending Priya.
- Steps not `.done`: `b_fig_layer2_g` (GOAL 19.x), `b_fr_compare_g` (0.1 / 0.7), `b_imaging_lmm_g` (IMG 13.x), `b_residual_roi_frA_g`, `b_rsplit_PS92_g`, `z_epoch_kernels_g` (3.11 / 3.12), `a_svf_ls_g`, `t_svf_te_g`, `e_basis_g`, `e_spout_patterns_g` / `e_spout_pat_stats_g` / `e_pat_stats_maps_g` (15.x pattern stats), `e_split_where_g`, `e_plain_where_g`, `e_mae_g`, `e_perspout_g`, `e_rmse_unsc_g`, `e_filter_cmp_g`, `l_basis_extract_g`, `l_interhemi_rest_restnl_g`, `x_enlv_regress_g`, `d_fig_goal_summary*`, `g7_section_g_figs*`: HELD pending Priya.
- GOAL Fig 1.1 and Figs 1.8-1.10 (no-FaceRhythm spout-unique frame): not re-run under their tag in v20: HELD pending Priya.

## Summary

**Drafted (6):** D1 frozen encoder target / executed / PETH in ALM, tjM1, MOs / MOp, SSp-m (+ cortex rows); D2 per-spout
unique maps (Fig 1.7); D3 pixel R² and R² share maps (20.3-20.5); D4 ITI direction decoding (10.3) + ITI ramp timing
(14.1 text); D5 ENL angle, primary 1.0 s sliding window (14.21 S3); D6 LP far-contra map amplitude, cue + lick, RESTnl and
session mean (7.10b, 7.18b, 7.19b).

**Same conclusion as the live page:** D2, D3, D4, D5; D6 under RESTnl. **Wording changes from the rebuild that need
sign-off:** D1 (a)-(c); D6 session-mean twin (Q6).

## Open questions for Priya

- **Q1 (D1).** Approve the three rebuild-level wording changes: chronic target loss in MOs / MOp but no longer in ALM;
  chronic executed gain frontal-wide (MOs-rest / MOp-rest now starred); movement PETH up chronically in ALM / MOs / SSp-m
  as well as tjM1?
- **Q2 (D1, glue).** f_frA_g has two CI-edge star changes in expected-stable areas with identical point estimates
  (uniq_target lick MOp_right acute -.010 -> -.010*; uniq_task cue SSp-m_right acute -.016* -> -.016). Not used in the
  drafted text. OK to treat as borderline?
- **Q3 (D3).** The R² map numbers are pixel means computed from `r2_maps.npz` for this draft (the step prints none).
  Keep them in the caption, or show the maps without numbers?
- **Q4 (D2).** Fig 1.1 (no FaceRhythm) vs Fig 1.7 (FaceRhythm, per-spout lick kernels): the re-run produced only the 1.7
  frame. Re-run 1.1 / 1.8-1.10 under their tag, or make 1.7 the primary and drop 1.1?
- **Q5 (D5).** On the primary 1.0 s sliding window, beyond-spout PARTIAL r chronic gained a star with the glue fix (-.039
  [-.083, +.009] -> -.050 [-.090, -.005]*); not a page claim. Mention it, or leave the caption as drafted? (The glue report
  collapses the 0.5 / 1.0 s rows, so its "slide r_partial subacute" flag is the 0.5 s variant.)
- **Q6 (D6).** Under the session-mean twin PS95 is also below pre (cue .80, lick .93; RESTnl .98 / 1.06). Keep the
  agreed #7 wording ("falls in 3/4; PS95 shows no drop") with the twin numbers beside it, or reword?
- **Q7 (held, #4).** Pre-cue far-contra acute amplitude: RESTnl .82 / .61 / .73 / .85 (4 / 4 below pre) but session mean
  .67 / .70 / 1.68 / 1.15 (2 / 4 below; `g5_lpfinal_maps_g` precue/working). Which reference carries the #4 sentence?
- **Q8 (D6).** Cue RESTnl chronic far contra is back in amplitude (pooled 1.00) but 261 of 2,022 bins still differ from
  pre (lick: .99, 263 bins). Add "shape not fully back" to the caption?
