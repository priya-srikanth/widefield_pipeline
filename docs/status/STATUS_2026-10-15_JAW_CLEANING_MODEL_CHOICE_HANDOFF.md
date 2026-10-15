# Handoff — 2026-10-15 (morning): jaw cleaning v35, jaw lateral readout across the cohort, LP r4 ctx-semi scored, jaw-source choice

**START HERE.** Supersedes `STATUS_2026-10-14_POSE_MODELS_HANDOFF.md` as the entry point (still the reference for the
human lick review, the EKS fix, the side-camera plan and WSL GPU safety -- its pitfalls list still applies). Model
register: **`docs/POSE_MODELS.md`** (read before touching tracking). Reasoning: `DECISIONS.md` addenda 2026-10-14
(afternoon) and 2026-10-15. Analysis desktop; M: = MICROSCOPE. Material for Priya goes on the server.

## 0. RUNNING (2026-10-15 ~08:00)
1. **LP r3 ctx-semi** (round-3 labels, same recipe as r4 ctx-semi: `heatmap_mhcrnn` + unlabelled-video temporal loss,
   WSL, 512 px, batch 1, `lp_capped.py` cap 0.75 + 6.8 GB guard), launched 07:57 by hand after the GPU chain's launch
   failed (Git Bash rewrote `/mnt/c/...`; chain script fixed with `MSYS_NO_PATHCONV=1`). ~10 h, then predicts the two
   PS93 clips + six review clips. Log `lp_stage\lp_train_semi_ctx_r3_wsl.log`; model
   `/root/lp/cam4-2026-09-28/models/r3_mhcrnn_semi_512_20261014`; predictions -> `lp_stage\r3_vs_r4_lp\r3mhcrnn_semi_512\`
   (check the script's copy step). Low priority (a like-for-like arm); score with `model_lick_compare --review`.
2. Done overnight: DLC round 3 on nine 60-trial jaw clips (PS92 / PS94 / PS95 x 0814 pre, 0821 acute, chronic 0908
   for PS92 and **0907 for PS94 / PS95 -- not recorded 0908**) -> `session_poses/<a>_<d>/windows_DLC.csv`.

## 1. Findings since the 10-14 handoff
1. **Jaw x ceiling (to-do 9) -- DONE, cleaning v35 adopted.** The v3.4 ceiling (40 px, old rig) wiped 9 % of PS93
   0814's wide-open frames (18 % PS95), mostly real smooth excursions; the errors are one-frame side-flips. v35 =
   `x_ceiling_px` 75 + median-filter x jump test (25 px) + bracketed flip rule (x steps > 30 px in and out within 200
   ms, ONLY when the flanks agree <= 15 px and sit within 25 px of the resting jaw, refilling X ONLY) + **hidden jaw =
   NaN** (no baseline fill). All Priya-approved after QC sheets (`session_poses/jaw_lateral/jaw_ceiling_qc.png`,
   `jaw_flip_qc.png`). The y ceiling (125 px) stays: what it removes is grooming.
2. **PS93 jaw anatomy (Priya).** Two candidate "lowest points": the pronounced true one deviated to the MOUSE'S LEFT
   (image-right; what the labels mark) and a smaller local one on the mouse's right (image-left). When the true one is
   unclear, models (DLC most) pick the other = PS93's jaw side-flips. Attributed to PS93's pre-existing right deficit.
3. **cam4 image-right = the mouse's LEFT** (3-D stage fit: *_L = mouse's left; *_L spouts image-right).
4. **Jaw lateral readout (to-do 10), all four animals.** Close / centre spouts cover the chin (jaw point lands on the
   visible edge, away from the spout), so read only far_L / far_R trials. `FAR_DIR` = half the far_L - far_R
   difference = how far the jaw moves TOWARD the target (px, opening > 50 px, spout frame); `FAR_BIAS` = their mean.
   FAR_DIR (DLC r3, cleaning v35, 4-10 trials per far position):

   | | pre 0814 | acute 0821 | chronic 0907/0908 |
   |---|---|---|---|
   | PS92 | +10.0 | +1.4 | +5.2 |
   | PS93 | +9.4 | -5.8 | +0.3 |
   | PS94 | +14.6 | -1.6 | +14.6 |
   | PS95 | +18.1 | +11.6 | +13.8 |

   Jaw steering toward the target is present pre in all four, drops acutely in all four (least PS95), recovers in PS94
   / PS95, partly PS92, not PS93. Angle form agrees. FAR_BIAS is animal-specific (PS93 a steady +10-15 = anatomy, item
   2). PRELIMINARY: few trials, single-camera DLC r3; acute licks may be shallower (jaw following a shorter tongue
   reach -- control with protrusion). Worth a whole-session check with the production model.
5. **LP r4 ctx-semi scored** (`docs/POSE_MODELS.md`): review precision 90 % / share 95 % (LP r4 85 / 98), contact
   recall 97.6 / 97.1 % (below the 99 % gate; LP r4 98.5 / 99.6), 0 raw jaw flips -- BUT partly by ABSTAINING: jaw
   confident on only 55-62 % of frames at close_L / close_center (LP r4 76-81 %). EKS r4+semi keeps LP r4 recall, not
   the clean jaw.
6. **DLC vs LP for jaw / the jaw-source choice.** Raw: DLC reports the jaw more (incl. wide-open beside the rod, where
   LP says "not visible" because blanks = occluded) but flips ~5x more. After v35 cleaning on PS93 0814: LP r4 jaw 0
   jumps > 30 px at 96 % coverage; DLC r3 26 jumps at 99 %; ctx-semi 0 at 90 % (close_L 61 %). Priya judged LP r4's
   jaw points correct on trial 47 close_center where ctx-semi abstains (and DLC r3 twice took the fake minimum)
   -> **LP r4 is the best jaw source so far.**
7. **Cleaned traces** (`pose_cue_traces compare --cleaned [--fill] [--trials N]`): all models agree closely on the
   tongue; DLC r3 holds a confident tongue point at the lips at close_center between licks. Figures in
   `session_poses/PS93_<d>/r4_eval/cue_traces_compare_all_v35_cleaned[_filled_6trials].png`.

## 2. Decisions (10-14 afternoon .. 10-15)
* Jaw cleaning v35 in `configs/defaults.yaml` (see 1.1); hidden jaw left NaN; tongue keeps its mouth fill PENDING
  Priya's look at the labelled clip (`cleaned_labels_LP_r4.mp4`, below).
* Jaw lateral deviation is read from the far pair only; per-animal tables before pooling.
* Model choice still open; current leaning **LP r4 alone** (best jaw after cleaning, best tongue recall/coverage),
  ctx-semi as the precision check for no-contact licks.

## 3. Code (all pushed)
* `wfield_local/orofacial_clean.py` `jaw_v34`: `frame_jump_thr_x_px`, `x_flip_step_px` / `x_flip_max_ms` /
  `x_flip_flank_agree_px` / `x_flip_flank_rest_px` (x-only refill), `hidden_fill`; tests in
  `tests/test_orofacial_clean.py`.
* `scripts/jaw_lateral.py`: `ceiling` (rule comparison + QC sheet), `flips` (flip-event sheet, `--overlay` other
  models), `deviation` (per-trial lr / angle, far-pair FAR_BIAS / FAR_DIR, figure). Outputs `session_poses/jaw_lateral/`.
* `scripts/model_lick_compare.py`: `--review` (precision / share vs Priya's review), jump counts, `--tag`.
* `scripts/pose_cue_traces.py compare --trials N`.
* `scripts/cleaned_label_clips.py`: labelled video of the CLEANED tongue / jaw (measured / interpolated / mouth fill /
  hidden) -> `session_poses/<s>/r4_eval/cleaned_labels_<model>.mp4`.

## 4. Pending / to-do
1. **Priya: watch `session_poses/PS93_20260814/r4_eval/cleaned_labels_LP_r4.mp4`** -- does the tongue's mouth fill
   make sense, or leave it blank too? (Tongue consumers read `x_masked` already; the kinematics use the fill only via
   `x_final`.) Make the same clip for other models / 0821 if asked.
2. **Decide the cam4 production model with Priya** (leaning LP r4). Before: check which contacts ctx-semi misses
   (findable?), and add jaw coverage per position to the selection rule (POSE_MODELS §4).
3. Run the chosen model on the twelve jaw clips + other sessions; redo the jaw readout with it (and a
   tongue-protrusion control for the acute FAR_DIR drop); then the production run -> executed-angle readout (MUST
   REVISIT).
4. Score LP r3 ctx-semi when it finishes (low priority).
5. Labelling (Priya's call): label the jaw's inferable position when it is mostly visible beside the rod, blank only
   when truly hidden -- would let LP keep the jaw on wide openings.
6. Earlier to-dos still open (10-14 handoff §4): side cameras (student, SIDECAM Part A), pixel-vs-LocaNMF gap,
   FaceRhythm, cam1 Parts 2 / 3.

## 5. Pitfalls (new; 10-14 list still applies)
1. **Git Bash rewrites `/mnt/c/...` arguments** passed to `wsl` (-> `C:/Program Files/Git/mnt/c/...`): prefix
   `MSYS_NO_PATHCONV=1`. It silently cost the overnight r3 launch.
2. **A model's "zero flips" can be abstention**: report coverage (confident / cleaned frames per position) next to any
   flip or jump count.
3. **The jaw "rest" fill is not rest**: with v34 it drew a hidden jaw as closed; now NaN. Anything plotting the jaw
   with `--fill` shows gaps where the jaw is hidden.
4. **The `_r4` session_poses folders hold only 2 trials per position** (round-4 scan cache) -- too few for per-position
   readouts; use the 60-trial clips.
5. **Shell edit scripts and `\n` inside f-strings** keep biting (a real newline lands in the source): write the edit
   script to a file, or use the Edit tool.
