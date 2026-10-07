# Handoff — 2026-10-06: first whole-session (O2) analyses, movement-model upgrades, contact classes, video regressors

**START HERE for the imaging x movement analyses and the orofacial cleaning changes of 10-06.** Continues
`STATUS_2026-10-05_CAM1_LP_EKS_O2_HANDOFF.md` (tracking / labelling / O2; still the entry point for those) and
`STATUS_2026-10-05_IMAGING_MOVEMENT_HANDOFF.md` (the pre-registered movement analyses). Reasoning and numbers:
`DECISIONS.md` 2026-10-06 entry + afternoon addendum. Analysis desktop; M: = MICROSCOPE. Everything below is ONE
pre-stroke session (PS93 0814) tracked with the ROUND-3 cam4 model -- pipeline tests with real numbers, not results.

## 0. RESUME CHECKLIST

**Running / just finished:**
1. Video motion energy, PS93 0814, all 4 cameras: DONE (unmasked; `session_poses/PS93_20260814_full/
   video_motion_<cam>.{npz,png}`). Masks decided against (below) -> these ARE the regressors.
2. Model comparison: A/B/C done (logs `~/lp_stage/full_0814_logs/compare/`): RUNNING SPEED is the big new term
   for motor cortex (unique MOp_L 0.33, MOp_R 0.24, MOs 0.13-0.16; SSp 0.01-0.05; full R^2 MOp_L 0.10 -> 0.43);
   target unique shrinks a little (MOp_L 0.0074 -> 0.0031). D (+ video, 4 cams) running (relaunched after a
   one-line fix). Then: `movement_position_angle PS93:20260814:full --perms 200 --running --video cam1 cam2 cam3
   cam4` (options added 10-06; outputs tagged `_video1234_run`).
3. Multi-view LP pilot: built (`wfield_local/lp_multiview.py`, tests, `configs/lightning_pose_multiview_pilot.yaml`,
   launcher `C:\Users\SabatiniLab\train_lp_multiview_fg.sh`, copy in `scripts/wsl/`). Export: 526 moments (177
   paired, 119 cam1-only, 230 cam4-only), split-rule visible codes. Both smoke tests passed (arm a val RMSE 88 ->
   40 px in 2 epochs; GPU 4.9 GB). **Arm (a) FULL training started 10-06 ~17:40** (~2.9 min/epoch, ~15 h; log
   `~/lp_stage/lp_train_mv_a_full.log`); arm (b) after it: `Start-Process wsl.exe -WindowStyle Hidden
   -ArgumentList '-e','bash','/mnt/c/Users/SabatiniLab/train_lp_multiview_fg.sh','b','full'`. Priya OK'd: random
   frame split (pilot), 640 px. Arm (b) = calibration + 3-D augmentation + projection losses vs (a) none. NB LP's
   3-D augmentation warps each image with a 2-D similarity fitted to >= 3 TRIANGULATED keypoints (labelled in both
   views); only 105 / 526 moments qualify today, the rest get no geometric augmentation in (b) -> (b) understates
   calibration until more paired labels exist. Patch masking on (LP multi-view default; trains cross-view inference).

**MUST REVISIT (Priya, 2026-10-07: "this would be very important to clarify"):** the executed-angle readout
test -- does post-stroke cortex follow the EXECUTED lick angle within a cued position, or the TARGET? Current
answer (10-05, `scripts/lick_template_match.py`, calibrated within-position readout): no evidence, pooled r -0.05
acute, -0.10 chronic. That rests on 60-trial windows, the round-3 cam4 model, the pre-10-06 tongue cleaning and a
DLC-only movement model. Rerun it, pre-registered settings unchanged (baseline pre_cue), when ALL of these exist:
(1) the next cam4 model (round 4 labels; DLC and/or LP, multi-view if the pilot wins) and cam1 for angle checks;
(2) WHOLE sessions from O2 for several pre sessions AND the post sessions (acute / subacute / chronic per animal),
`:full` specs; (3) the retuned cleaning (Rule 4 swap, contact windows, grooming excluded); (4) the movement-removed
version with the full movement model (DLC + running + video), alongside the raw one; (5) angle = at peak
protrusion, straddled-spout licks flagged and reported with / without. Report per animal and pooled, all
variants run.

**Next, in order:**
1. Read the comparison: unique variance of video beyond DLC (+ running); then rerun the residual position test
   (`movement_position_angle`, 200 perms) with the richest movement model (needs a `--video/--running` option like
   `movement_encoding_session`'s, not built yet).
2. **Contact-class survey across sessions** (`scripts/contact_survey.py`, not built yet): DAQ contact durations for
   EVERY session (pre + post; long touches at close spouts = water bridges?) and no-tongue counts where DLC exists,
   before fixing `contact_classes` thresholds. NB `segmentation.grooming` (behavior_events) is an older,
   never-refined single-spout proxy (> 0.4 s contact, OFF); `contact_classes` supersedes it once validated.
3. Review / commit the multi-view LP pilot; launch full training (arm a, then b) per the agent's report.
4. Open questions for Priya: lick-direction-shift paths to image orientation too? angle reference (keep the
   far_center axis, ~2 deg from image vertical on 0814, relabelled -- recommended -- or image vertical)? lick-1
   definition for a re-protrusion that starts while the tongue is still out from a pre-cue lick (trial 28)?
5. Later: paw-at-face detection for grooming without a contact; 2pRAM-style top-k lick-subspace removal; production
   O2 run after the next cam4 model; whole post-stroke sessions -> the pre-registered template-match readout test.

## 1. Findings (PS93 0814, whole session, round-3 cam4 DLC)
* O2 output == desktop (median 0.2-0.4 px on 239,564 shared frames). All 288 cam1+cam4 videos on O2 scratch.
* 60-trial subset was representative of the whole session (except close_L peak angle, likely bimodal).
* Movement dominates (SSp unique 0.20-0.25); target position unique small but positive (0.002-0.010); residual
  position-specific +0.003..+0.011, p < 0.005 all areas (200 perms); angle beyond position tiny (MOs / SSp-m).
* Far spouts lack a cue-locked residual transient because first licks are late and variable (450-930 ms, IQR
  0.6-1.0 s vs close ~190 ms, ~50 ms); lick-aligned they show an SSp dip + slow ramp. The residual is relative to a
  POSITION-BLIND movement model, so part of "position-specific" is likely position-specific touch feedback.
* Smooth kernels (raised cosine, 0.15 s) fit as well as per-frame FIR with ~1/4 the weights; contact licks give a
  big SSp response (~0.005 dF/F/lick, peak ~0.45 s), no-contact licks about half, slower.
* Tongue straddles the spout on licks: DLC's tip switches lobes (single-frame hops, and persistent switches into
  retraction). Rule 4 retuned (below) removes the hops; persistent switches remain (accepted, Priya).
* Contacts: 3,919 / 3,978 inside a lick window; 47 tongue out but no kept lick; 12 no tongue; 1 grooming bout
  (paws at face, 323 ms contact = the longest; no contact > 400 ms in the session).

## 2. Decisions (10-06)
* Whole-session trial windows CUT at the next trial's start (`o2_inference.trial_frame_spans`); post-stop tail
  kept; guards `session_poses.read_index` / `assert_no_double_events`.
* Event kernels: smooth raised-cosine basis available (`--kernels smooth`); per-frame FIR stays the pre-registered
  default. Lick events: `--licks split` = DISJOINT contact / no-contact lick kernels + non-lick contacts (robust to
  post-stroke misses); default stays tongue onset + contact.
* Rule 4 (x outliers) retuned for this rig: direction-agnostic bursts <= 3 frames, threshold 20 px (rig p99.9),
  x-only re-interpolation (`x_jump_mode: swap`); stroke_orofacial's "snapback" kept for parity (module default).
* Contact-to-lick matching = lick window peak -60 .. +120 ms or the detector's rise / fall if wider
  (`contact.match: span`); grooming = touch >= 250 ms with the tongue out < 50 % of it, widened +-5 s
  (`contact_classes`; thresholds provisional until the survey).
* Figures: cohort palette / order (`spout_behavior.position_style`); trajectories in IMAGE orientation (mouth top
  centre, image-right right); distance / protrusion axes down = further from the mouth.
* Angles are tongue angle AT PEAK PROTRUSION everywhere (not stroke_orofacial's max-signed angle).
* Motion energy: parallel per camera (chunked decoding). SPOUT: motorised -> must not enter; handled in TIME by
  blanking the video regressors outside position strobe .. trial end (repositioning happens between trial end and
  the strobe; a stationary spout makes no motion energy, so close-spout pixels stay in on far trials). TREADMILL:
  NOT masked -- only the paws move it, so it is locomotion (a drum mask also covered the paws); DAQ running speed
  (`behavior_events.session_speed`, the one treadmill definition) added as a separate regressor.

## 3. Pitfalls (new)
1. **Scripts silently dropped the `:full` tag** (`pose_inputs`, `movement_position_angle`) -- fixed; any new
   session-level script must pass the spec through `session_pieces(..., spec=)`.
2. **Overwriting cross-session summaries**: backups in `session_poses/_pre_full_20261006/` (overlap-window and
   20-perm versions); variant runs write tagged files (`_smooth`, `_splitlicks`).
3. **Background jobs > 2 h die with the bash tool**: launch long runs detached (`Start-Process cmd.exe /c ...`),
   find / stop them by COMMAND LINE (shared box), never by name.
4. **Standardised kernel weights look tiny** (per SD of a sparse column): use `kernels(model, per_event=True)`.
5. **The detector's rise start / fall end are narrower than the physical lick** -- contacts land ~28 ms before
   the peak; do not match contacts on that span alone.
6. **Motion energy sees the motorised spout** (target information) -- use video regressors only inside position
   strobe .. trial end (`movement_encoding_session.video_signals` does this).
7. **HMS scratch**: deletion is by modification time and touching dates is a policy violation; copy with
   `rsync -ah`.
8. Heredocs with quotes / backslashes break inline Python edits -- write edit scripts to files (Write tool).

## 4. Where things are
| what | where |
|---|---|
| whole-session poses + outputs | `M:\...\DeepLabCut\Widefield\session_poses\PS93_20260814_full\` |
| backups of overwritten summaries | `session_poses\_pre_full_20261006\` |
| analysis logs | `C:\Users\SabatiniLab\lp_stage\full_0814_logs\` (`cut\`, `video\`) |
| new code | `wfield_local/{video_motion,contact_classes}.py`, `scripts/{residual_lick_aligned,video_motion_session}.py`, Rule 4 in `tongue_detect.py`, basis in `movement_encoding.py` |
