# Within-lick corrections (analysis 2.1): design written BEFORE any post-stroke number (2026-10-30)

Question (RECOVERY_MECHANISMS_LIT_20261030.md item 2.1; Bollu et al. 2021 Nature): **is recovery carried by WITHIN-LICK
corrections** -- re-aiming inside one protrusion, seen as a launch direction that differs from where the tongue ends up,
and as corrective submovements (CSMs: extra speed peaks inside a lick)? Bollu: CSMs are ALM-dependent and more frequent
for cue-evoked licks and far targets. BETWEEN-lick corrections (lick k -> k+1) are already done
(`scripts/lick_correction.py`, FINDINGS 0b "CORRECTION LICKS") and are not redone here.
Code: `scripts/within_lick_correction.py` (`run` / `summarise` / `plot`, `--plot-only`). Outputs only under
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/within_lick_correction/`.
Only data looked at before this was written: four AUGUST-PRE sessions for feasibility (below). No post-stroke number.

## 0. Feasibility (pre-stroke only; PS92_20260807, PS94_20260807, `scratchpad feas.py / feas2.py`)
* Per-frame 3-D tongue: `<dlc>/session_poses/pose3d/<animal>_<date>_frames.csv.gz` (`tongue_ml/ap/dv` mm, origin = the
  spout mouth zero; `tongue_src` 1 = both cameras, 2 / 3 = one camera + neighbor depth). Frame interval exactly 4 ms
  (250 Hz; the true effective rate, frames pair exactly on the 4 ms grid). Lick windows = v36 `licks_all.csv`
  (on_ms .. off_ms; row index = `lick_uid` of `kinematics_all/licks.csv.gz`).
* Frames per lick window (post-cue, PS92 0807, 4777 licks): median 21 (p10 10, p90 26); 3-D frames median 20; longest
  consecutive 3-D run >= 8 frames in 86 %; fully tracked (every window frame 3-D) 44 %. Single-camera share of 3-D frames
  median 0.31 (the aim coordinate ml is MEASURED either way; depth is interpolated on those frames).
* Fully tracked licks: 21-24 frames (84-96 ms) whole lick, but onset -> extension peak (the PROTRUSION proper) only
  **median 7-8 frames (28-32 ms; p10 5-6)**; first tracked frame median 1.5 mm from the mouth zero, extension 4.5-4.8 mm;
  peak speed median ~225 mm/s.
* Noise: residual from a 7-frame quadratic Savitzky-Golay fit (an UPPER bound: it contains real fast motion) ->
  per-axis SD (ml, ap, dv) 0.10 / 0.17 / 0.09 mm (PS92) and 0.10 / 0.16 / 0.10 (PS94); the held-out fill error is
  0.13-0.14 mm at 8 ms gaps (pose3d.session_3d docstring). Speed from a 5-frame derivative: noise ~10-15 mm/s.
* Speed peaks per lick (SG 5 frames, quadratic): a normal lick has 2 (out, back). Extra peaks: share of licks with > 2
  peaks at prominence 40 / 60 / 80 mm/s = 0.20-0.27 / 0.08-0.15 / 0.03-0.08; a noise surrogate (smooth track + Gaussian
  noise at the upper-bound SD) creates an extra peak in 0.09-0.11 / 0.07 / 0.05 of licks.
* **VERDICT: launch-vs-endpoint aim is resolvable** (launch over the first 3 frames, endpoint at the extension peak 7-8
  frames later; direction noise ~5-8 deg per lick, averaged over hundreds of licks per session). **CSMs are MARGINAL**:
  the protrusion has fewer than 8 frames, and extra speed peaks occur at rates only ~1-2x a conservative noise
  surrogate. CSMs are therefore SECONDARY, scored with a noise-calibrated threshold (section 2) and always reported next
  to the surrogate's false-CSM rate per session; a CSM change counts only if it exceeds the change in that false rate.
  If no threshold meets the calibration rule, CSMs are declared not resolvable and only the aim measures are reported.

## 1. Data (fixed)
* Sessions: `analysis_kit.curated_labels()`, animals PS92-PS95 only (cohort2 PS118 / 120-122 never), present in
  `kinematics_all/licks.csv.gz` (i.e. passing `pose3d.failed_sessions` gates). Epochs `wfield_local.epochs.epoch_of`;
  **pre = AUGUST pre sessions only** (June excluded everywhere, incl. targets and thresholds).
* Licks: v36 `licks_all.csv` windows. PRIMARY = cued POST-CUE licks (`context == post_cue`); SECONDARY = ITI licks
  (no target: only CSM / curvature / duration). Contact (hit) = the v36 `contact` flag (DAQ lick-sensor onset inside
  the lick; same as `kinematics_all`). Spout = `orofacial_cohort.pos_key(position, "L")` (all lesions LEFT), canonical
  order close_ipsi, close_mid, close_contra, far_ipsi, far_mid, far_contra. Order: FIRST post-cue lick of the trial
  (lick_idx 0) vs LATER licks (lick_idx >= 1).
* Track per lick: the 3-D frames inside on_ms .. off_ms; single missing frames (one 4 ms gap) inside a run linearly
  interpolated; the longest such run is the lick's track. SG smoothing = Savitzky-Golay, 5 frames (20 ms), order 2
  (position and derivative). Extension peak k = frame of max distance from the mouth zero (raw track).
* Scorable for AIM: the run starts <= 8 ms after on_ms (onset seen) and k >= 4 frames after the run start.
  Scorable for CSM / curvature: additionally the run ends >= off_ms - 8 ms (whole lick seen), >= 10 frames.
  Coverage (share scorable), single-camera share and per-session noise SD are reported per epoch (tracking control).

## 2. Per-lick metrics (fixed)
SIGNS: every angle is the MIDLINE angle asin(ml / |v|) in degrees, **+ = CONTRALESIONAL (mouse right)**; the sign is
forced so the August-pre far-contra endpoint mean is > 0 (as lick_correction).
* **LAUNCH angle** theta_L = midline angle of the SG-smoothed displacement from the first tracked frame to frame 3
  (N = 3 frames = 12 ms after the tip leaves the mouth).
* **ENDPOINT angle** theta_E = midline angle of the raw tip position at the extension peak (from the mouth zero; = the
  established `kinematics_all.midline` / lick_correction `angle`). Secondary endpoint: angle at closest approach to the
  target spout (theta_C), and the late heading (SG displacement over the 3 frames before the peak).
* Targets (per animal x spout, AUGUST pre, CONTACT post-cue licks): tgt_L = mean theta_L, tgt_E = mean theta_E.
  Errors e_L = theta_L - tgt_L, e_E = theta_E - tgt_E (deg; - = ipsilesional of the pre-stroke hit pattern).
* **Within-lick correction** corr = e_E - e_L (deg, + = the endpoint lies more contralesional than the launch, relative
  to the pre-stroke pattern; at far contra + = TOWARD the target). General toward-target form: toward = -sign(e_L) * corr
  for licks with |e_L| >= 5 deg (+ = moved toward the target during the lick). Within-lick GAIN g_w = OLS slope of corr
  on -e_L (spout fixed effects; 1 = endpoint fully back on the pre pattern, 0 = endpoint keeps the launch error; noise in
  theta_L inflates g_w by regression to the mean, constant across epochs only if noise is -> tracking control).
* **CSM count** = sum over the two phases (frames before / after k) of max(0, n speed peaks in that phase - 1); speed =
  |SG derivative| (mm/s); peaks = `scipy.signal.find_peaks` with prominence >= theta_s on the zero-padded speed trace.
  Lateral variant CSM_lat: the same on |v_ml| with theta_ml. Per-lick flag CSM >= 1.
  **Threshold rule (computed once, AUGUST pre post-cue licks of all four animals, seed 0, before any post data are
  read):** per-axis noise SD per SESSION = SD(raw - SG(7, 2)) / sqrt(1 - 1/3) over interior frames of its fully tracked
  post-cue licks; surrogate per lick = SG(9, 3) smoothed track + one Gaussian noise draw at the session's SD (seed =
  crc32 of the session id); theta = the smallest value on 30, 40, ..., 200 mm/s for which, pooled over the AUGUST-pre
  post-cue licks of all four animals, the share of licks whose surrogate CSM count exceeds the smooth track's is
  <= 5 %. Saved to `csm_threshold.json`. The same surrogate runs on every session (`csm_false` = that share per
  session at the frozen theta), so a post-stroke change in tracking noise is visible next to the CSM rate.
  (Implementation: peak prominences are stored per lick, so the count at any theta is exact without re-reading.)
* Path CURVATURE = SG path length / chord, first tracked frame -> extension peak (1 = straight).
* Durations: protrusion_ms = t(k) - t(first frame); lick_ms = off_ms - on_ms.
* HYPOMETRIA hyp_mm = max over the track of the projection of the tip on the unit vector to the commanded target spout
  minus the spout's distance (mm; - = fell short); also `miss_mm` (closest 3-D approach).
* Peak speed (SG), reach (mm at k).

## 3. Cells and summaries (fixed)
Per session x spout group (all / far_contra / other5 / each of the 6 spouts) x order (first / later / any) x outcome
(hit / miss / any): means of e_L, e_E, corr, toward, CSM share, CSM count, CSM_lat share, curvature, protrusion_ms,
lick_ms, hyp_mm, peak speed; g_w; n; coverage; csm_false. A cell needs >= 10 scorable licks.
**PRIMARY (confirmatory; 3 tests x 3 epochs)**, cued post-cue licks, FAR CONTRA, any order, any outcome:
  P1 e_L (launch error; restitution predicts a return to 0), P2 corr (within-lick correction), P3 CSM share.
Everything else (other spouts, first vs later, hit vs miss, toward / g_w, curvature, hypometria, durations, ITI, theta
sensitivity x 2/3 and x 3/2) is SECONDARY / descriptive.

## 4. Hypotheses (fixed)
* **Compensation by within-lick correction:** e_L stays ipsilesional (< 0) into the chronic epoch while corr (> 0) and
  CSM share rise post-stroke, most at far contra (and in later > first licks if corrections are learned within a trial).
* **Restitution:** e_L returns toward 0 chronically; corr and CSM share return to pre.
* **Correction elsewhere:** e_L biased chronically with corr ~ 0 -> recovery is carried by the between-lick
  corrections / later licks (lick_correction), not within the protrusion.
* Bollu prediction checked pre-stroke (descriptive): CSM share far > close spouts, cued > ITI.

## 5. Neural part (secondary, pre-registered)
Imaging sessions with the cached `lick_correction` extraction (`iti_licks/lick_correction/sessions/<label>.npz`: ROI v8
area traces, drift removed, dF/F x1e3, lags -0.5 .. 0.8 s around each post-cue lick's exact DAQ peak; joined on
`lick_uid`). Areas ALM, tjM1, SSp-m, SSp-n, SSs, each hemisphere (ipsilesional = LEFT = `_left`, contralesional =
`_right`), plus POOL = their sum per hemisphere; control RSPd. Response y_w = mean over window w after the lick peak
(primary 0.0-0.2 s; secondary 0.2-0.35 s) minus the level -0.2 .. -0.1 s.
* N1 (per session, OLS): y_w = b . [1, theta_L, theta_L^2, theta_E, reach, peak speed, contact, first, spout FE,
  next lick's angle, contact and peak-to-peak interval (GCaMP6s overlap; 0 / mean-filled with an indicator when there is
  no next lick within 0.35 s)] + gamma_csm * CSM flag. gamma_csm + = CSM licks evoke more at matched kinematics. Same
  with |corr| (deg) in place of the CSM flag (gamma_corr per 10 deg). >= 10 CSM and >= 10 CSM-free licks per session.
  Sensitivity: FIRST licks only (no preceding-lick overlap); deconvolved (revup_neural.deconv, tau 0.6 s).
* N2 "carried by correction licks?": per session mean y_w on ALL scorable licks and on CSM-free licks with |corr| < 10
  deg ("clean"); epoch change vs pre for each. Carried share = 1 - delta(clean) / delta(all) (pooled boot_delta points),
  reported only where delta(all) is a change by the criterion below. Pre-registered reading: the larger post-stroke lick
  response is "carried by correction licks" only if delta(clean) is NOT a change AND carried share >= 0.5.
* Question 2 (does gamma_csm itself grow post-stroke?): gamma per epoch vs pre.

## 6. Statistics (fixed)
Per-animal table (animal x epoch session means, n sessions) printed and saved FIRST (CLAUDE.md rule 8). Then
`analysis_kit.boot_delta` (paired nested animals -> sessions, seed 0) per post epoch vs AUGUST pre, pooled levels by
`boot_ci`. **A change = 95 % interval excludes 0 AND >= 3 / 4 animals with the same sign.** Primary: also Holm over the
9 primary tests (reported). Recovery link (post sessions; lick_correction `track` form): MixedLM aim ~ within-animal value
+ between-animal value + log(day) + (1 | animal), aim = far-contra first-lick angle - pre (lick_correction
`corr_session_aim.csv`, deg, - = ipsilesional), plus value ~ log(day) and per-animal r; for P1-P3 at far contra.
Figures: per-animal lines + pooled; stars only in a reserved top band; no many-thin-bar figures; figdata saved.

## Outcome (2026-10-30, run after this design was fixed; `within_lick_correction.py run` then `summarise`)

Outputs: `.../imaging_encoding/within_lick_correction/` (`csm_threshold.json`, `licks_scored.csv.gz`, `sessions.csv`,
`per_animal.csv`, `boot.csv`, `primary_holm.csv`, `track.csv`, `neural_*.csv`, `figures/wlc_fig1..6_*.png`,
`figdata/within_lick_correction.pkl`, `run_log.txt`, `summarise_log.txt`). 119 curated sessions extracted (31 August
pre first), 107 in the summary (pre = August only), 581,467 scored licks; 107 imaging sessions in the neural part.
Sign factor +1 (ml + = mouse right = contralesional, as expected).

**Threshold (August pre only, 103,673 fully tracked post-cue licks, 31 sessions):** theta_s = 50 mm/s (surrogate false
CSM 5.0 %, observed CSM share 15.6 %; at 40: 6.8 % vs 21.8 %; at 80: 1.6 % vs 5.2 %); theta_ml = 50 mm/s (false 3.3 %,
observed 13.6 %). Observed ~3x the conservative noise surrogate -> CSMs RESOLVABLE by the rule.

**Pre-stroke (Bollu check, descriptive):** CSM share far contra .28 [.14, .42], far ipsi .33 [.15, .43], far mid .09,
close spouts .04-.08; cued (all spouts) .16 vs ITI .10 -> more CSMs for far side targets and cued licks, as in Bollu.
Within-lick correction pre: hits ~0 (-0.1 deg), MISSES +9.4 deg (later misses +10.9): missed licks re-aim inside the lick.

### Primary (cued post-cue, FAR CONTRA, any order / outcome): per-animal session means, pre / acute / subacute / chronic
(deg; + = contralesional; corr + = toward the far-contra target)
| animal | e_L launch | e_E endpoint | corr | CSM share | surrogate false | n sessions |
|---|---|---|---|---|---|---|
| PS92 | -0.7 / -21.6 / -3.5 / +2.3 | -0.3 / -23.5 / -8.4 / -0.4 | +0.4 / -1.9 / -4.9 / -2.7 | .18 / .15 / .08 / .05 | .05 / .20 / .07 / .03 | 8 / 5 / 2 / 12 |
| PS93 | -1.8 / -15.5 / -6.3 / -3.1 | +2.4 / -11.2 / -3.4 / -1.7 | +4.3 / +4.3 / +2.9 / +1.4 | .11 / .06 / .10 / .12 | .06 / .04 / .08 / .07 | 7 / 4 / 3 / 12 |
| PS94 | -2.2 / -37.5 / -17.6 / -13.4 | -1.8 / -38.4 / -15.6 / -10.0 | +0.4 / -0.8 / +2.0 / +3.4 | .39 / .08 / .03 / .02 | .08 / .06 / .01 / .01 | 8 / 6 / 5 / 8 |
| PS95 | -1.6 / -34.4 / -6.9 / -4.8 | -0.8 / -31.1 / -6.1 / -2.4 | +0.9 / +3.2 / +0.8 / +2.4 | .45 / .32 / .31 / .37 | .10 / .09 / .08 / .09 | 8 / 1 / 7 / 11 |

boot_delta vs August pre (95 %; Holm over the 9 primary tests):
* P1 e_L: acute -25.7 [-34.3, -17.0] (4/4, p_holm .002); subacute -7.0 [-12.7, -2.6] (4/4, .002); chronic -3.2
  [-8.8, +1.5] (3/4; PS92 +3.0) -> NOT a change chronically.
* P2 corr: acute -0.3 [-2.4, +1.8]; subacute -1.3 [-4.0, +1.1]; chronic -0.4 [-3.1, +2.4] (p_holm 1.0) -> NO change.
* P3 CSM share: acute -0.13 [-0.25, -0.03] (4/4, p_holm .004); subacute -0.15 [-0.29, -0.03] (3/4, .009); chronic
  -0.14 [-0.30, -0.02] (3/4; PS93 +0.02; p_holm .068) -> CSMs DECREASE; they do not rise.

Secondary, far contra: e_E acute -26.0, subacute -8.3, chronic -3.5 [-6.6, -0.8] (4/4) -- the endpoint error shrinks in
step with the launch error. toward: acute -5.1 [-6.4, -4.0] (4/4), subacute -1.8, chronic -1.5 (n.s.). g_w pre .78 ->
.64 / .63 (subacute change, 4/4) / .74. CSM excess over the surrogate pre .21 -> acute .01 / subacute .11 / chronic .09
(changes, 3-4/4); the surrogate false rate is unchanged (chronic -.024 [-.059, +.002]); theta x2/3 and x3/2 give the same
direction (chronic -.17 / -.09, both changes). Lateral-velocity CSM unchanged (~.04). Curvature acute +0.14 (4/4), then
back; protrusion acute +14 ms (4/4), then back; hypometria acute -1.6 mm (4/4), chronic -0.37 [-0.82, -0.03] (3/4); miss
distance chronic +0.35 mm (4/4); peak speed acute -54 mm/s (4/4), back by subacute. Tracking: fully tracked share .89-.91
every epoch; single-camera share slightly LOWER chronically (-.014).
Order / outcome (far contra): FIRST-lick corr chronic -3.1 [-6.1, -0.4] (3/4, i.e. LESS correction); first MISSES keep
the launch bias chronically (e_L -8.5, 4/4). LATER MISSES lose their pre-stroke in-lick correction (pre +10.9 -> acute
-8.0 [-11.7, -4.3] 4/4, subacute -5.0 4/4, chronic -2.7 n.s.). LATER HITS: CSM share down every epoch (chronic -.16,
3/4). Per spout: corr changes nowhere (all CIs include 0); CSM share falls at far contra (all epochs), far ipsi (acute
only), far mid (subacute / chronic -.03, 4/4); close_mid acute +.03 (4/4); ITI CSM share unchanged.
Recovery link (post sessions, within animal, log day in the model): corr does not track aim (+0.10 [-0.33, +0.53];
per-animal r -.19 / -.68 / +.72 / +.17); csm_share slope -22.5 [-36.9, -8.2] but per-animal r -.87 / +.68 / -.76 / +.28
(inconsistent); csm_excess none. (e_L / e_E vs aim are near-circular: aim is the far-contra first-lick angle.)
Mean paths (Fig 5): acute far-contra paths are straight and displaced ipsilesionally from launch on; no late bend
toward the target.

### Neural (107 imaging sessions; orofacial POOL = ALM + tjM1 + SSp-m + SSp-n + SSs per hemisphere; raw, 0-0.2 s)
* gamma_csm (CSM minus CSM-free at matched launch / endpoint / reach / speed / contact / order / spout / next lick): pre
  POOL c +4.0 [-1.1, +10.9], i +5.1 [+0.5, +10.8]; acute about unchanged (-1.1 / -1.6); subacute -11.1 / -10.5 and
  chronic -10.7 [-17.0, -5.6] / -10.2 [-15.2, -6.1], 4/4 each; every orofacial area x hemisphere does the same
  (subacute / chronic, 4/4), plus RSPd_i subacute. First licks only: chronic -3.9 / -4.9 (4/4). Hemisphere difference
  c - i: no change (except first-lick subacute +4.5, 4/4). Deconvolved (tau 0.6): no change.
  Per-animal POOL c pre / subacute / chronic: PS92 +2.3 / -19.9 / -16.9, PS93 +14.3 / +8.0 / +6.3, PS94 -2.1 / -13.3 /
  -12.9, PS95 +2.9 / -1.9 / -1.8. -> post-stroke, CSM licks evoke LESS (not more) than matched CSM-free licks.
* N2 carried?: the ACUTE rise in the lick response is a change on all licks (POOL c +18.1, i +17.7) AND on clean licks
  (+21.2 / +22.5) -> carried share -0.17 / -0.27; subacute / chronic: no change on either set. "Carried by correction
  licks" = 0 of 168 cells (all areas, windows, tau). The larger post-stroke lick response is NOT carried by correction
  licks.

### Reading
Restitution-like for the launch (P1 not a change chronically, except PS94 -11 deg); NO within-lick compensation (P2 flat;
toward-target correction lower acutely; the in-lick re-aiming of missed later licks is lost acutely and only partly
back); FEWER CSMs (P3 down, most in PS94; 3x surrogate noise, robust to threshold, not a tracking change). The endpoint
follows the launch: a lick that starts off target ends off target. With lick_correction, recovery at far contra is
carried by between-lick / later-lick behavior and the launch itself returning, not by within-protrusion corrections.
Neural: the post-stroke lick-response increase is on clean licks; CSM licks lose their extra response.

### Deviations
1. Noise SD per SESSION instead of per animal (prereg edited before any run; the calibration used only August pre).
2. Peak speed falls acutely (-54 mm/s); with an absolute prominence threshold slower licks have smaller peaks, so the
   ACUTE CSM decrease is partly vigor. Chronic speed is back (+11 n.s.) while CSMs stay down, so the chronic decrease
   is not.
3. Per-cell minimum 10 licks: "first hit" far contra acute has PS93 only (cells with < 3 animals are not changes).
4. The recovery link for e_L / e_E is near-circular (aim = far-contra first-lick angle); reported, not interpreted.
5. The run log prints whole item tuples (cosmetic).
Caveats: n = 4; 250 Hz gives a 7-8 frame protrusion, so CSMs are submovements of >= ~20 ms only; launch direction noise
~5-8 deg per lick; GCaMP6s overlap handled only by next-lick covariates and the first-lick sensitivity; imaging is
lick-locked levels, not onsets. PROVISIONAL.

## Deviation 2026-11-05 -- licks of the CAMERA working period only (Priya 11-05)
Every post-cue / ITI lick of a session was scored, including licks after the animal stopped. Priya 11-05: behavior-only
analyses are cut to the camera working period (first cue -> the session's last camera lick, minus the stopped runs;
`trial_classes.working_trial_ids`, by each lick's licks_all trial_id). `within_lick_correction.load_licks` now drops the
other licks -- in the August-pre calibration (`csm_threshold.json`) as well as in the post-stroke cells -- and leaves out
sessions without camera classes (listed). The per-lick extraction caches are unchanged. Nothing re-run yet; the Outcome
above is the whole-session version.
