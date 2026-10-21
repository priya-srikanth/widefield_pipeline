# Handoff — 2026-10-20 (midday): template timing, cam1 2-D licks, jaw anchor, full v36 + 3-D re-run in flight

**START HERE for pose tracking / 3-D / cam1 licks.** Supersedes `STATUS_2026-10-19_POSE3D_HANDOFF.md` (still the reference
for the 3-D pipeline's construction: PnP, spout mouth zero, fill, gates, its §5 O2 commands and §8 pitfalls 1-14, copied
below). Method detail and numbers: `docs/POSE_MODELS.md` (D10-D12 + the dated notes at its end). Analysis desktop; M: =
MICROSCOPE. Write only inside `MICROSCOPE/Priya/...`; never delete original data; never kill python.exe by name.

## 0. Resume checklist (in order)

**Status 10-20 ~13:00: items 1-3 below are DONE (both branches finished, results in §4a, encoding window told). Nothing
local is running. Open: item 4 (O2 cam1 copy-backs for PS94 / PS95) and §7.**

**Status 10-20 20:45 (overnight, autonomous): nothing of mine running.** Jaw anchor OFF and the jaw-only re-run DONE
(20:30): 3-D production (curated; PS93 0814 / 0821 / 0908 / 0911 now from PRODUCTION cam1, not clips -- tell the
encoding window), kinematics both lick sets, summary fig, jaw crosscheck (FAR_DIR r 0.95, 47 sessions), phase, cue /
first-lick traces. No src 4 / 5 jaw frames remain; centre-spout jaw coverage (cue + 70 ms on) median close_center 0.86,
far_center 0.81 (some PS92 sessions 0.53-0.63). PS92 jaw readouts unchanged to 3 decimals; PS93 (more production
sessions): FAR_DIR 0.19 -> 0.02 -> 0.05, CLOSE_DIR -0.05 -> -0.12 -> -0.22 (pre / acute / chronic).
**Morning to-do:** (1) O2 cam1 array 56830841: 65 curated pending (Priority); copy back + `o2_inference collect`, then the
§6 chain for PS94 / PS95 (first time 3-D / cam1 for them -> per-animal incomplete, CLOSE_DIR, cue / lick traces, aim
angles). (2) Local cam1 LP run deferred (box shared) -- reverse order from task 137 if Priya OKs. (3) Optional: a
mouth-open check of the tongue-at-lips frames (Priya judged sheet B real; quantify per epoch). (4) Practice vs stroke for
the chronic licks-per-trial rise needs a non-stroke control (none in this cohort) -- phrase as "candidate".

1. **Two local background branches were running at hand-off (§3).** Read their logs; if the session is gone, check the
   outputs' timestamps and re-run what is missing with the §6 commands (both are safe to re-run).
2. When Branch B is done: look at `pose3d/figures/summary_PS92.png` / `_PS93.png`, `pose3d/phase/*`, the jaw cross-check
   (FAR_DIR r), `pose3d_vs_cam4`, and `cam1_licks/incomplete/` -- all sessions now on template time + the jaw anchor.
3. When Branch A is done: tell the encoding window (it waits for the re-run before reloading v36 cohort tables; Priya
   relays) and spot-check `cohort/tables/prepost/epoch_by_animal.csv` vs yesterday's for the 125 unaffected sessions
   (expected: tiny changes only).
4. O2 cam1 production still running (§5): 58 / 137 cam1 videos back at 11:20 (PS92 all 35 days; PS93 06-05 .. 08-10+).
   After each copy-back: §6 chain (3-D + cam1 licks pick up new sessions automatically).
5. Open decisions for Priya: §7 items 1-3.

## 1. What was done (10-19 night -> 10-20)

* **Video timing now comes from the ALIGNMENT TEMPLATES' per-frame camera times** (Priya: "all timings should use the
  alignment templates"). `o2_inference.build_windows` -> `trial_frame_spans_ts`: cue frame = frame nearest the cue in
  template time, window edges by time, `t_ms` = template time from the cue frame snapped to 4 ms (drops leave GAPS);
  `t_daq_ms` = unsnapped true time; frames repeating a trial time dropped (PS92 0605). Old path (`trial_frame_spans`, the
  frame-index affine) only as fallback. `scripts/cam4_timing_check.py` -> `session_poses/cam4_timing_check.csv`.
* **3-D pairs the cameras on `t_daq_ms`** (nearest within 2 ms, one-to-one); output `t_ms` = cam4's. Session gates
  `pose3d.failed_sessions` (spout fit > 0.2 mm, median tongue reprojection > 6 px). `pose3d production <subset>` now MERGES
  into `summary_production.csv` (no longer overwrites).
* **Jaw rest-depth anchor** (`pose3d.anchor_single_view`, `ANCHOR_PARTS = ("jaw",)`, src 4 cam1 / 5 cam4): long one-camera
  jaw stretches (centre spouts: rod end-on in cam1's view; close_center also hides the chin in cam4) placed on the seen
  camera's ray at the trial's resting-jaw camera depth. Check: `scripts/pose3d_jaw_anchor_check.py`.
* **cam1 2-D lick detector** (`wfield_local/cam1_licks.py`, `scripts/cam1_licks.py run|review [--round 1|2|3]`): events =
  runs of cleaned cam1 tongue frames; `size_class` lips / out; flags cam4_matched, in_3d, contact, jaw_only_trial;
  `droplet` (water droplets: static, away from the mouth opening; cam1-only events only); `miss_prev3`; spread / range /
  spout distance. Three blind reviews by Priya (artifact pages, db collections `cam1_lick_review`, `cam1_lick_review2`,
  `cam1_lick_review3b`; scored CSVs in `session_poses/cam1_licks/review{,2,3}/scored.csv`).
* **Readout** `scripts/cam1_incomplete_licks.py` -> `session_poses/cam1_licks/incomplete/` (per session x position: cam4
  no-contact, cam1-only, at-the-lips, incomplete share, jaw-without-lick cam4 vs cam1-corrected; figure per animal).
* **3-D kinematics**: jaw `jaw_close_dir_mm` / `jaw_close_bias_mm` (close side spouts) beside FAR_DIR; summary figure
  bottom strip shows all four.
* **Phase figures** (`scripts/pose3d_phase.py`): method line in every title (lick set, phase range, in-mouth rule, 3-D
  source, averaging), wrapped; phase shown 0.1-0.9 (as cam4); FIRST post-cue lick set (`*_first_*`); in-mouth versions
  (`phase_dist` = in-mouth 0 / `phase_dist_out`; `phase_angle` out-only / `phase_angle_mouth0`); NEW `phase_elev` (angle
  below the horizontal plane; + = down) + `_mouth0`; trajectory rows named by viewpoint (front = facing the mouse as
  cam4; side = from the mouse's left; bottom = from ABOVE, mouse facing down the page), each row one shared mm range.
* **v36 re-run in place** (`--force`): first the 9 timing sessions (10-20 morning; cohort + jaw-lateral rebuilt 09:07 /
  09:14), then ALL 133 sessions (10:28-11:19, ok 133). Cohort code: `orofacial_cohort._row_of` looks rows up by time
  (template windows have gaps; the old `row0 + (t - t0) / 4` crashed on PS92 0908).

## 2. Decisions and why

| decision | why |
|---|---|
| all video timing from the templates' per-frame camera times | the frame-index affine drifts 4 ms per dropped frame; 9 / 134 cam4 sessions off by 26 ms - 1.3 s at the cue (PS92 0605 / 0606 / 0908, PS93 0605 / 0606, PS94 0603 / 0605 / 0805, PS95 0605); and the cameras run at 249.81 fps (4.003 ms), so frame-count time is 0.076 % slow everywhere (2.3 ms at 3 s, ~76 ms at 100 s) |
| v36 re-run for ALL sessions (not just the 9) | one consistent time base for v36, 3-D and cam1 (Priya: "re run v36 please"); within the response window the change is < 1 frame |
| 3-D cameras paired on true time, output `t_ms` = cam4's | dropped frames had paired wrong partners (PS92 0908 tongue reprojection 8.5 px); the "June outliers" PS92 0606 / PS93 0605-0606 were drops too |
| session gates: spout fit, tongue reprojection > 6 px | catch timing / pose failures automatically (normal 2.7-4.0 px); now only PS92 0605 fails |
| no depth prior for one-camera TONGUE points | held-out 0.6-1.7 mm on licks < 1 mm (neighbour fill 0.13-0.23) |
| jaw rest-depth anchor for long one-camera JAW stretches | held-out ml error 0.03-0.05 mm, per-trial lr_wide r 0.95-0.99, FAR_DIR within 0.02 mm; ap from cam4 alone 0.15-0.39 mm off -> don't use anchored frames for the opening's backward part |
| incomplete licks from cam1 2-D, not 3-D | counting needs no depth |
| droplet flag on cam1-only events: spread < 3 px, range < 12 px, median out > -35 px | Priya's review: every false cam1-only event was a water droplet; 16 / 17 caught, 0 / 70 real lost; PS93 round 3: 24 / 24 cam1-only real |
| recent misses (`miss_prev3`) kept as a covariate, not a gate | droplets follow uncollected water (pre 3 % of trials after a hit -> 15 % after 3 misses); the shape rule already works |
| `lips` class means "at the lips" only for cam1-ONLY events | cam4-matched "lips" events were all full sideways licks |
| lick phase: onset -> offset per lick, shown 0.1-0.9; peak-to-peak secondary | as cam4's per-lick normalised time; keeps isolated (acute far contra) licks |
| trajectory "≥ half the licks have a point" kept, lowest share printed in the title | within 0.1-0.9 coverage is 0.88-1.0 almost everywhere (lowest ~0.64 at close_L / close_center ends) |
| cam4 chronic jaw lateral confirmed | far_L / far_R per-trial r with 3-D 0.80 chronic vs 0.90 pre; low all-position r came from close / centre spouts cam4's readout already excludes |

## 3. Running / pending at hand-off (10-20 ~11:25)

| what | task (this session) | output | if lost |
|---|---|---|---|
| **Branch A**: v36 cohort products in parallel -- cohort x4 (default / `--all-trials` / `--baseline all` / both), jaw_lateral x4, W2 x2, heatmaps, misdirected, robustness, deck | `bw8v4dtg7` | `LP_analysis/lp_r4_heatmap_20261013__clean_v36/cohort/` | §6 "v36 cohort" block |
| **Branch B**: 3-D all sessions (8 workers) -> kinematics x2 -> spout error, crosscheck, summary fig, phase, jaw crosscheck, vs_cam4 -> cam1 licks (6 workers) -> incomplete readout | `bw6he0d24` | `DeepLabCut/Widefield/session_poses/{pose3d,cam1_licks}/` | §6 "3-D + cam1" block |
| O2 cam1 production array **56830841** | Priya | scratch `o2/lp_cam1_ctx_prod_20261019/out/` | §5 |

v36 itself is DONE (133 of 136 cam4 sessions; 137 cam4 videos because PS92 0812 is a split day). Not processed, none in the
curated cohort: PS93 0805 (the behaviour log is a failed start -- GUI v46 logging race, 0 trials; see
`Behavior_logs/Widefield/PS93_20260805_202013/..._reconstruction.md`), PS94 0601 and PS95 0601 (behaviour fine, NO cam4
alignment template for those recordings).

**Curated cohort only (Priya 10-20: "we can stop processing sessions that will not be part of analysis").** `pose3d
production`, `pose3d_kinematics`, `cam1_licks run`, `cam1_incomplete_licks` and `orofacial_outputs process` now default to
`analysis_kit.curated_ids()` (120 sessions: per animal 3 June + 16-17 Aug + 8-9 Sep + 2 Oct); `--all` overrides. Not
curated: PS92 0602-0605 / 0805, PS93 0605 / 0805, PS94 and PS95 0601-0605 / 0805. Branch B's 3-D step had already started
on all 57 available sessions (7 non-curated) -- its later steps use the curated default; non-curated frame files on disk are
simply ignored downstream.

## 4a. 10-20 afternoon: both branches DONE (A 12:02, B 12:17), template time + jaw anchor everywhere

* **Branch A check** (no pre-re-run v36 copy survives -- compared with v35 `prepost_latepre`, which differs only in June):
  107 sessions, every animal x epoch cell identical except PS92 chronic (0908, the late-drop session: latency 223 -> 189
  ms, contact 0.858 -> 0.894) and ITI lick rate +-0.04 Hz (249.81 fps time base). Encoding window told via Priya.
* **Per-animal acute far-contra protrusion** (to-do 3, v36 `position_metrics`, August pre): -12 / -27 / -30 / -65 px
  (PS92-95), but far ipsi also -9 to -19 in all four; contra-minus-ipsi -4 / -8 / -16 / -46. PS95 = ONE acute session (96
  licks); PS94's first two acute days have 24 / 63 licks. Don't quote "-30 to -48 px"; contact loss (-0.37 to -0.83) holds.
* **Jaw anchor coverage** (6 sessions, cue + 70 ms on): centre spouts 0.84 -> 1.00 (anchored 13-16 % of frames); side
  spouts already ~1.00. FAR_DIR cam4 vs 3-D r = 0.96 over 45 sessions.
* **Jaw per epoch** (median of sessions, mm, + = ipsi / toward target): FAR_DIR PS92 0.27 -> 0.01 -> 0.12 (pre / acute /
  chronic), PS93 0.22 -> -0.01 -> 0.04. **CLOSE_DIR PS92 -0.13 -> -0.22 -> -0.25, PS93 -0.03 -> -0.14 -> -0.20 (same
  direction in both)**; CLOSE_BIAS PS92 -0.28 -> +0.02 -> -0.37 but PS93 +0.56 -> +0.27 -> +0.29 (opposite signs -- not a
  shared effect). Video sheet `pose3d/close_dir_sheet/` (`scripts/pose3d_close_dir_sheet.py`): the jaw opens beside the
  rod, away from it; cam4 often loses the chin at close_R peak opening. (A 3-session reading of the sheet suggested
  "bias, not dir" -- the cohort medians contradict it.)
* **PS93 cam1 production now covers acute / chronic** (cam1_licks: 23 PS93 sessions). Incomplete share far_R PS93 0.39
  pre -> 0.73 acute -> 0.33 chronic (PS92 0.17 -> 0.86 -> 0.35: replicates). PS93 far_L incomplete ~0.6 in EVERY epoch and
  at-the-lips events sit on the LEFT spouts in PS93 (right in PS92). CHECKED 10-20, NOT a side flip: PS93's August-pre
  licks aim to the mouse's RIGHT and barely reach far_L in every independent measure -- DAQ contact far_L 0.39 / far_center
  0.52 / far_R 0.69 (others 0.77-0.94); cam4 peak angle centre -13 / -12 deg (others +4 to +10), far_L +1 (others
  +17-18); 3-D sideways at far_L 0.20 mm (PS92 1.47), far_center 0.74 mm to the right; cam1 tongue offset at the centre
  spouts -15 / -21 px (PS92 -1 / -3) with L > centre > R ordering intact. PS93's left-side incomplete licks are its
  pre-stroke behaviour (the "right orofacial deficit"); use within-animal changes only.
  Jaw-without-lick acute 0.14-0.32 in both, ~0 chronic; cam1 correction small.
* **Cue-aligned 3-D traces** (to-do 1b, `scripts/pose3d_cue_traces.py` -> `pose3d/cue_traces/`; engaged trials incl.
  no-lick trials, in-mouth = 0, mean over cue + 70 .. 1500 ms). far_R acute, pre -> acute: trial-averaged distance PS92
  1.72 -> 0.20 mm, PS93 1.46 -> 0.29; tongue-out share 0.44 -> 0.06 / 0.42 -> 0.10; tongue-out-only distance 3.84 ->
  3.08 / 3.41 -> 3.01 (-20 / -12 %); jaw opening 1.64 -> 0.50 / 1.25 -> 0.43. Trials WITH a lick are just as low (0.26 /
  0.35 mm). So the acute "lower amplitude" is mostly LESS TIME WITH THE TONGUE OUT (fewer, later, isolated licks) plus
  less time near full extension -- not shorter peak reach (per-lick 3-D peak 5.10 vs 5.14 mm). Also smaller at far_L /
  far_center acute (PS92 1.58 -> 0.94, 1.05 -> 0.69). Chronic back to pre. Note: acute far_R sits ~0.25 mm above 0 even
  BEFORE the cue (pre ~0.05) -- tongue at the lips between trials? check before quoting.
* **First-lick-aligned 3-D traces** (`pose3d_cue_traces --align first_lick` -> `pose3d/lick_traces/`; Priya liked these).
  acute far_R: the first lick is LATE (PS92 1184 vs 265 ms, PS93 979 vs 375) but FULL SIZE (peak 0-200 ms 4.30 vs 4.18
  mm PS92; 3.80 vs 4.29 PS93), then the train stops (tongue-out share ~5-10 % after it). So acute far_R "lower amplitude"
  = late single lick + no train, not shorter licks. (A first look misread a 1.2 mm second attempt as the first lick.)
* **Chronic "increase" in the in-mouth = 0 distance** (both animals, close spouts; PS93 also far_L / far_center): ~all
  TIME OUT (x1.2-1.8), reach x1.00-1.10; driven by earlier, cue-locked first licks (PS93 close_center 154 vs 350 ms,
  far_L 273 vs 762) -- first-lick aligned the difference mostly vanishes. Chronic sessions are later in training:
  possibly practice, not stroke -- TO-DO: early vs late August pre trend.
* Pre-cue 3-D tongue = real sparse licks: 3-D tongue-out share in -500 .. 0 ms (close 0.6-1.7 %, far 3-8 %) matches
  cam4's licks in the LAST 500 ms (0.12-0.21 / 0.34-0.75 Hz; cam4 `precue_rate_hz` counts from spout arrival, ~1 Hz).
  Figures: one trial set + a tongue-out-share row, pre-cue shaded, tongue-out-only distance hidden where < 5 % out.
* **3-D aim angles in the cue / lick traces** (`mid` = deg out of the midline, + = ipsi; `elev` = deg below horizontal;
  tongue-out frames >= 0.5 mm from the mouth zero, hidden where < 5 % out). Displacement (in-mouth = 0) mixes aim with
  time out; the angle is aim alone. Acute shift toward ipsi at EVERY spout: far_R PS92 -23 -> +11 deg (past the midline),
  PS93 -28 -> -11; other spouts +2 to +12; chronic mostly back. Lines more transparent (alpha 0.65, SEM 0.1; Priya).
* **Spout-error figures** (`pose3d_spout_error.py`): lateral axis now + = IPSI (mouse left) at every spout, drawn to
  the right as cam4's front view -- the old "+ = outward" flipped between L and R spouts, so acute far_R (3 mm inward =
  toward ipsi) looked flipped to Priya. New epoch versions `scatter_epoch_<a>.png` / `traces_epoch_<a>.png` (session
  medians faded, epoch mean bold).
* **far_center side-view diagonal (PS92 chronic)**: closest-approach points on a line parallel to the rod's slope, in
  EVERY chronic session (28-53 % of far_center licks), 96 % DAQ contact, single-view share 0.59 vs 0.50 -- real contacts,
  probably the tongue sliding along the rod rather than tapping its tip. Unconfirmed: video sheet of those licks to-do.
* **Check sheet** (`scripts/pose3d_check_sheet.py`, dlc env -> `pose3d/check_sheet/{A,B,C}_*.png`; 3-D point projected
  with the session's PnP cameras + each camera's 2-D point):
  A. far_center diagonal CONFIRMED as contact geometry: the tongue wraps the rod's SIDE; the tracked tip lands at a
     different height along the shaft each lick, so closest approach traces the rod's slope. 3-D and 2-D agree.
  (Priya's review 10-20.) A: one point (trial 25, cam4 2-D none = cam1 + neighbour depth) projects onto the spout end,
     not the tongue tip -- either the tip is hidden behind the rod in cam4 or the fill depth is off. TO-DO: share of
     far_center closest-approach frames that are one-camera before reading the diagonal.
     RESOLVED (10-20 night): closest-approach frame source, far_center, PS92: on-diagonal licks are ~45-55 % cam1 +
     neighbour depth (cam4 lost the tip) vs 2 % cam4 + neighbours; off-diagonal the reverse. cam1's line of sight is
     ~45 deg in the side view, like the diagonal -- BUT with both-camera frames ONLY the diagonal persists (on-line share
     chronic 0.46, pre 0.40 vs 0.45 / 0.32 with all frames). So it is real rod-contact geometry; cam4 loses the tip
     because the rod blocks it during side contact (hence the cam1 fills, and trial 25). Present pre too -- not chronic.
  B. acute far_R PRE-CUE tongue: Priya reads them as REAL (tongue at the lips; my first reading "false points on closed
     lips" was wrong). So acute far_R has the tongue at the lips before the cue more often (6.6 % vs 3.1 % of frames),
     consistent with cam1's rise in at-the-lips events. 3817 frames in 56 / 90 trials (PS92 0818).
  C. anchored jaw mostly right, BUT cam1 + anchor frames (cam4 missed) sit TOO HIGH when the jaw is OPEN (3 / 4 shown):
     the anchor keeps the resting camera depth, and an opening jaw moves down + back, partly along cam1's ~45 deg line of
     sight -> placed too close to cam1 -> too high in cam4 = opening UNDERESTIMATED (held-out cam1-only dv 0.07-0.15 mm
     vs cam4 0.02-0.04). Proposed fix: per-session depth model from the 2-D jaw position (fit on both-camera frames),
     held-out tested. Affects centre-spout jaw opening only (FAR_DIR / CLOSE_DIR = side spouts).
     **TESTED** (`scripts/pose3d_jaw_depth_model_check.py`, dlc env; 4 sessions, ~1.1 M held-out side-spout frames):
     depth model = OLS per session x seen camera, depth change from rest ~ the seen camera's 2-D jaw displacement from
     the trial's resting 2-D jaw (dx, dy, squares, product) + tongue out (either camera) + the seen camera's 2-D tongue
     distance; fitted on both-camera frames of other trials (5-fold by trial). OPEN-jaw 3-D error anchor -> model:
     cam1-only 0.54 -> 0.25 mm, cam4-only 1.39 -> 0.31; opening bias cam4-only -0.55 -> -0.10 (cam1 -0.04 -> 0.00);
     closed frames 0.11-0.16 -> 0.11-0.13; per-session consistent; lr_wide r and FAR_DIR unchanged (0.95-0.99, within
     0.03 mm). The anchor's earlier "ap 0.15-0.39" averaged open with closed frames and hid this. Tongue features not
     yet ablated. RECOMMENDED: replace the rest-depth anchor in `pose3d.anchor_single_view` with this model, then re-run
     3-D production (curated) + kinematics + figures (awaiting Priya; the box was busy).
     **SUPERSEDED -- ANCHOR TURNED OFF (10-20 evening, Priya: "should we just give up on filling in occluded jaw like
     this?" -- yes).** Sheet C with both placements: the depth model is still too high on OPEN close_center jaws seen by
     cam1 only (3 / 4 such tiles) -- both cameras almost never see an open jaw there, so the model extrapolates.
     `pose3d.ANCHOR_PARTS = ()`; one-camera jaw beyond the 200 ms fill stays NaN (centre spouts ~84 % coverage, the
     missing frames mostly OPEN close_center jaws -> close_center jaw OPENING biased low: leave it out of opening
     comparisons, as cam4 does). FAR_DIR / CLOSE_DIR unaffected (side spouts, ~0 % anchored). Re-run (task bjj6xs7sf,
     3 workers, serial kinematics -- the other window was using ~25 GB): 3-D production (curated) -> kinematics cam4 ->
     combined -> summary_fig, jaw_crosscheck, phase -> cue / first-lick traces. NOT re-run (no 3-D jaw input): v36,
     cam1 licks, incomplete readout, spout_error. Superseded by this: the 10-20 "anchor coverage 0.84 -> 1.00" result.
* **Practice vs stroke (chronic close-spout licking)**, cam4 v36 all four animals, early vs late August pre vs chronic:
  first-lick latency was still FALLING through pre in PS92 / PS93 (-8 to -35 ms / day) and PS94 / PS95 were already
  fast and do not change chronic (close 149 -> 158, 144 -> 145 ms) -> the chronic speed-up looks like continued
  learning. Licks per responded trial at close spouts is higher chronic in ALL FOUR (12 -> 14, 11 -> 16, 10 -> 14,
  7 -> 11) with no upward pre trend -> candidate real chronic change (or long-term practice; no unlesioned control).
* **Reach epoch figure** `spout_error/reach_epoch_<a>.png` (|tip - mouth zero| through the lick, beside the
  distance-to-spout one). Why the spout-distance plot does not show protrusion loss: per-lick 3-D reach DOES match cam4
  per animal (far_R PS92 -3 % vs cam4 -7 %; PS93 -15 % vs -16 %) -- PS92 has little loss; distance to the spout is not
  monotonic in reach (licks overshoot) and is tongue-out-only. `OROFACIAL_FINDINGS.md` far-contra protrusion row now
  carries the per-animal caveat.
* **O2 cam1 array 56830841 (10-20 evening)**: 61 / 137 done (PS92 all, PS93 25). Pending tasks already ask
  `gres/gpu:1` (any type); they wait on Priority (fairshare). Cancelled the 10 non-curated tasks
  (`56830841_[68-71,75,103-106,110]` = PS94 / PS95 0601-0605 + 0805) -> 65 curated pending. `gpu_requeue` accepts 10 h
  jobs (`--test-only` OK; the multi-partition `scontrol update` on the queued array is what failed), but its estimated
  start was ~27 h out, so not resubmitted. O2 wall per video: median 3.5 h, p90 6.4, max 7.5. LOCAL cam1 LP run
  (reverse order: PS95 1005 = task 137, 1001, 0928, ...; WSL, sequence_length 16, ~10-12 h / video) DEFERRED by Priya
  -- the other window is using the box. If started: `scancel 56830841_<idx>` for each video done locally.
* `apparatus_mouth.py` now per-session PnP (reprojection 0.3-0.7 px); cam1 cleaning origin = PnP mouth + constant 30-32 px
  (rod edge). Phase speed figures renamed `v_speed_norm_*` / `v_speed_peak_*` (old `speed_*` PNGs remain on M:).

## 4. Results (PS92 = June + August pre, acute .. chronic d+50; PS93 = June / August pre production + 3 clip sessions)

* **2-D vs 3-D**: target spout from the lick endpoint (LOSO, August pre), 6 positions: PS92 0.53 cam4 vs 0.80 3-D, PS93
  0.42 vs 0.68; centre close vs far PS92 0.59 vs 0.87. cam4 lateral is faithful (r 0.95-0.97); cam4 "protrusion" = the
  DOWN axis. Extra 3-D coverage changes no lick-level effect (cam4 vs combined lick sets: z within ~1) -- it matters for
  COUNTS (no-contact +12 %, cam1-only events, jaw-only correction).
* **far_R acute (PS92)**: licks as long as pre in 3-D (5.10 vs 5.14 mm) but aimed elsewhere: down -5 % (cam4 protrusion
  165 vs 176 px, same), forward doubled (1.64 vs 0.81 mm), sideways across the midline (-0.87 vs +1.42 mm), closest
  approach 3.0 vs 0.9 mm, 0-33 % contact, 4x fewer licks. "Lower amplitude" appears only when trials without licks count.
* **Jaw**: FAR_DIR cam4 vs 3-D r 0.95 (34 sessions); collapse acute in both. **CLOSE_DIR (new, unvalidated)**: PS92 -0.14
  pre -> -0.27 acute / chronic mm (jaw AWAY from the target at close spouts); PS93 -0.05 -> -0.13 / -0.18 (clips).
* **cam1 licks** (droplets removed, template time): PS92 cam1-only 7.5 % pre / 3.9 % acute (pre figure pending the re-run);
  PS93 pre 5.4 %. Blind reviews: PS92 pre 23 / 24 real, acute off jaw-only trials 10 / 10, on acute jaw-only trials 0 / 5;
  PS93 24 / 24, size class 24 / 24.
* **Incomplete licks** (`cam1_incomplete_licks`, PS92, pre = August): far_R incomplete share 0.17 pre -> 0.86 acute ->
  0.35 chronic; cam1 at-the-lips events / trial rise 3-4x chronic on the right (far_R 0.51 -> 2.16, close_R 0.42 -> 1.42);
  jaw-without-lick 1-3 % pre -> 12-30 % acute at every position -> ~0 chronic; cam1 correction small (pre close_center
  3.5 -> 0.9 %).
* **Vertical aim (`phase_elev`)**: acute licks flatter (less steep below horizontal) at most positions.
* **PS93 contact** is genuinely lower than PS92 (August 62-69 %; cam4 v36 70 %).

## 5. O2: check jobs and copy outputs (Priya runs these)

Login node (`ssh ps150@o2.hms.harvard.edu`):
```
B=/n/scratch/users/p/ps150/o2/lp_cam1_ctx_prod_20261019
ls $B/out/*/*_done.json 2>/dev/null | wc -l                      # finished videos (of 137)
squeue -u ps150 -j 56830841 -h -t R | wc -l                       # running
squeue -u ps150 -j 56830841 -h -t PD -r | wc -l                   # pending
squeue -u ps150 -j 56830841 -t PD -o "%.18i %r" | head -3         # why pending
sacct -j 56830841 --state=FAILED,TIMEOUT,OUT_OF_MEMORY -X -o JobID,NodeList,State,Elapsed   # send to Claude if not empty
```
Copy back (transfer node `ssh ps150@transfer.rc.hms.harvard.edu`; partial copies fine):
```
rsync -ah --partial --exclude 'tmp*' /n/scratch/users/p/ps150/o2/lp_cam1_ctx_prod_20261019/out/  /n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/lp_cam1_ctx_prod_20261019/out/
```
Empty session folders on M: = videos still running on O2 (outputs sit in excluded `tmp*` until done). A video is complete
when its `_done.json` exists.

## 6. Local commands (repo root; locanmf env unless marked dlc)

```
# after a copy-back
python -m wfield_local.o2_inference collect --name lp_cam1_ctx_prod_20261019
# --- 3-D + cam1 (Branch B) ---
python -m scripts.pose3d production --jobs 8                          # dlc; all sessions in both bundles (subset = merge)
python -m scripts.pose3d clips PS93:20260814 PS93:20260821 PS93:20260908 --jobs 3   # dlc; until PS93 production lands
python -m scripts.pose3d_kinematics  &  python -m scripts.pose3d_kinematics --licks combined     # parallel ok
python -m scripts.pose3d_spout_error / pose3d_crosscheck / pose3d_summary_fig / pose3d_phase / pose3d_jaw_crosscheck / pose3d_vs_cam4   # parallel ok
python -m scripts.cam1_licks run --jobs 6            # after pose3d (in_3d flag)
python -m scripts.cam1_incomplete_licks
python -m scripts.pose3d_cue_traces --jobs 6           # cue-aligned 3-D traces incl. no-lick trials
python -m scripts.pose3d_close_dir_sheet [--animal PS92]   # CLOSE_DIR video sheet
# --- v36 (only when windows / cleaning code changes) ---
python -m wfield_local.orofacial_outputs process --bundle lp_r4_prod_20261015 --force --jobs 4 [--sessions A:D ...]
# --- v36 cohort (Branch A; the 4 variants of each can run in parallel, --jobs 3 each) ---
python -m wfield_local.orofacial_cohort --jobs 3 [--all-trials] [--baseline all]
python -m wfield_local.orofacial_jaw_lateral --jobs 3 [--all-trials] [--baseline all]
python -m wfield_local.orofacial_w2 --jobs 3 [--baseline all]; python -m wfield_local.orofacial_heatmaps --jobs 3
python -m scripts.misdirected_lick_counts; python -m scripts.orofacial_robustness_table; python -m wfield_local.orofacial_deck
# --- checks ---
python -m scripts.cam4_timing_check --jobs 4         # cue / window timing per session
PYTHONPATH=. python -m scripts.pose3d_jaw_anchor_check     # dlc
python -m scripts.cam1_licks review --round 3 --per 12      # blind review sample (page generators in the session scratchpad)
```
The box has 24 cores / 63 GB; ~30 % used by others at 11:20. 8 pose3d workers + 4 cohort jobs x 3 peak ~20 GB.

## 7. To-do

1. **Priya to decide**: (a) CLOSE_DIR -- sheet DONE 10-20 (§4a; jaw opens beside the rod, CLOSE_DIR same direction in
   PS92 / PS93) -- Priya to look; (b) cue-aligned 3-D distance plot DONE 10-20 (§4a); (c) `v_speed_*` rename DONE;
   (d) PS94 / PS95 cam1 when O2 reaches them -> re-run §6 chain, then CLOSE_DIR / incomplete per animal (rule 8).
   (e) acute far_R pre-cue tongue ~0.25 mm (cue_traces) -- check video before quoting. (f) chronic close-spout
   "increase" = earlier cue-locked licking: test practice vs stroke with an early vs late August pre comparison.
2. After Branch B: re-check the jaw anchor's coverage gain at far_center / close_center on all sessions; confirm FAR_DIR
   r and CLOSE_DIR values with the anchor in.
3. DONE 10-20 (§4a): per-animal far-contra protrusion -- mostly global at far spouts; PS95 = one session. Update
   `docs/OROFACIAL_FINDINGS.md` far-contra protrusion row (-30 / -26 px) with this caveat (encoding window owns that file?
   ask Priya).
4. Cohort tables for 3-D / cam1 when PS93-95 production lands (per-animal first, `analysis_kit.boot_delta`, August pre).
5. Labelling (if needed): paw-in-face frames as hard negatives.
6. Open: `scripts/apparatus_mouth.py` still uses the 09-11 extrinsics.

## 8. Pitfalls

New 10-20:
15. **Frame-index time is NOT true time**: drops shift every later frame by 4 ms each, and 249.81 fps makes it 0.076 %
    slow everywhere. Use template times (`t_daq_ms` / the new `t_ms`). Anything computing rows as `row0 + (t - t0) / 4 ms`
    breaks on gaps -- look rows up by time (`orofacial_cohort._row_of`).
16. **Review strips must pick frames by template time per camera** -- on drop sessions cam1 and cam4 rows showed different
    moments (round 3, first build).
17. **cam1-only flags depend on BOTH cameras' timing**: on drop sessions real licks were mislabelled cam1-only (PS93 pre
    9.3 % -> 5.4 % after the fix).
18. **Shell quoting**: `sed` / heredocs turn `\n` in replacement strings into real newlines in Python sources -- use the
    Edit tool or a scratchpad .py patch; lint after every patch.
19. **matplotlib tight_layout already reserves suptitle space**: adding a rect top margin doubles the gap.
20. **Static detections are not all droplets**: a static point AT the mouth opening is a real tongue tip at the lips; a
    static point that cam4 also calls a lick is the tongue held on the spout -- the droplet flag is cam1-only events only.
21. **TaskStop the chain shell** before relaunching; check no `orofacial_` / `pose3d` / `cam1_licks` python processes
    remain (by command line, never by name).

From 10-19 (still true): (1) heredoc quoting; (2) scratch files shadowing packages (`coverage.py`); (3) subset pose3d runs
now merge, but still re-run kinematics after; (4) never read `*_frames.csv.gz` while a chain rewrites them; (5) never
silence step output; (6) PnP object points and the fit check change together; (7) aniposelib only in the dlc env; (8)
`config.stroke_date` is MMDD (`config._to8`); (9) paw-in-face frames; (10) post-stroke set point ~0.5 mm closer; (11) PS93
acute / chronic still clips; (12) push races with the other window -- check `git status -sb`; (13) apparatus_mouth calib
extrinsics; (14) never name scripts `*_test.py` outside `tests/`.
