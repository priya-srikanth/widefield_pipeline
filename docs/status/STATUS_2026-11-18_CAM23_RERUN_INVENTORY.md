# cam2 / cam3 re-run: inventory of what must be re-run (started 2026-11-18, window a2)

**Why this exists.** Priya 11-18: "we are going to run the LP cam2/3 soon so in the coming days we'll have updated 3d
coordinates (and probably licks, from cam 1-4 instead of just cam 1+4) - please start a handoff with an inventory of
all the analyses that will need to be re-run with these new coordinates."

This is a living handoff. Whoever runs the re-run: tick items in section 4 as they land, record changed numbers in
`docs/DECISIONS.md`, and keep section 0 current.

## 0. State (11-18)

- Nothing re-run yet. cam2 / cam3 whole-session LP inference not started (Priya runs O2).
- The four-camera 3-D pipeline is BUILT and validated (`wfield_local/pose3d_side.py`, `scripts/pose3d.py production
  --four-cam <side bundle>`, writes `session_poses/pose3d_4cam/`; side cleaning approved by Priya 11-04;
  `docs/POSE_MODELS.md` Part E). It waits on the round-2 side model and the whole-session side inference.
- A **side-camera lick detector does not exist yet**. Today's camera licks are cam4 LP (`licks_all.csv`) plus cam1
  (`session_poses/cam1_licks/`); "licks from cam 1-4" needs a cam2 / cam3 detector and a fusion rule (D2 below).
- Three fixes were deliberately parked for THIS re-run and must go in with it:
  1. the spout-position rule (ENL no-lick medians everywhere): fix list in
     [`SPOUT_POSITION_AUDIT_2026-11-04.md`](SPOUT_POSITION_AUDIT_2026-11-04.md) (Priya 11-04: "ok to fix with our
     next 3d re-run (when we have cam 2/3)");
  2. the per-frame spout fix in production `pose3d` (bddf02e7) and the commanded 3-D spout table
     (`prep_3d_inputs.spout_table` -> `pose3d_spout_positions_commanded_<tag>.csv`, done 11-04);
  3. judging the jaw at the close spouts that neither cam4 nor cam1 can judge (close_center / close_mid) from cam2 /
     cam3 (Priya 11-05). Today those no-lick trials default to no response with `jaw_judged = False`.
- Also on the list from earlier handoffs: re-check the side tongue model bias and re-run
  `scripts/side_label_offsets.py` with the Part B labels; June side sessions stay skipped until their intrinsics are
  checked; the lick-template test (`scripts/lick_template_match.py`, Priya 10-07 "very important") was to be revisited
  once the next pose model exists.

## 0b. Lick fusion DRAFT (11-18, a2; Priya: "start drafting code for licks across cameras")

`wfield_local/camera_licks.py` + `scripts/camera_licks.py clips` (dlc env), tests `tests/test_camera_licks.py`. cam4 =
the production `licks_all.csv`; cam1 = the production `_events_all.csv` through `cam1_licks.keep_mask`; cam2 / cam3 =
`cam1_licks.detect` on the side tongue after `pose3d_side.fuse_session` (side cleaning + side droplet rule), in a
per-view mouth frame, with cam1's droplet rule rescaled by px / mm (`cam1_licks.droplet_flags(scale=)`, the only change
to an existing function). `fuse`: overlapping events of one trial (12 ms) = one lick, with the cameras that saw it;
`lick_train(min_views=)` = decision D2. Output: `DeepLabCut/Widefield/session_poses/camera_licks/clips_side_r2b_r2_full/`.

**First result, r2b side model on the review clips (7 sessions, 102 trials, cue -0.2 .. +1.5 s, 263 licks):**
220 (84 %) seen by all four cameras; 16 cam1-only (6 %; median 28 ms, 4 of them one PS93 0814 trial 540 with 556-748 ms
spans: review that clip); 4 cam4-only; 3 cam2 + cam3 + cam4 without cam1; 4 cam2-only, all static labels (dropped by
the rescaled droplet rule); **0 kept licks that only a side camera saw**. On these clips the side views confirm licks
rather than add them; whole sessions (more pre-cue / ITI time, more misses) may differ.

**Trial 540 reviewed (11-18, four-camera clip `session_poses/pose3d_side/video_4cam_PS93_20260814_t540_r2b.mp4`,
`scripts/pose3d_side_video.py --trials 540 --preds <r2b> --tag _t540_r2b`):** NOT licks. cam1's tongue label sits on a
large water drop hanging from the spout tip for the whole trial (the trial after 3 misses: uncollected water); cam4 and
both side views show no tongue. cam1's droplet rule misses it because the drop wobbles (range 10-24 px vs the rule's
12, spread 3.5-6 vs 3). **Production impact (census of every cam1 post-cue `_events.csv`, cam1-only events that pass
`keep_mask`):** 114 of 46,697 last >= 300 ms (97 trials, 27 sessions; pre 67 / acute 32 / subacute 3 / chronic 12
events); none in 3-D, 3.5 % with a DAQ contact, 1.23 misses in the 3 prior trials vs 0.21 for all cam1-only events,
closer to the parked spout (median 36 vs 54 px). Those events enter today's LP lick train and can make a trial
"incomplete". DECISION for Priya: (a) leave it (small), (b) drop cam1-only events >= 300 ms now, or (c) let the
four-camera fusion decide (a cam1-only event the side views do not see).

**Guards (11-18):** side events carry ``static`` (cam1's droplet rule) and ``far`` (cam1's far-from-mouth rule
`cam1_licks.FAR_FROM_MOUTH_PX`), both rescaled by px / mm; a guarded event counts only when another camera confirms it.
On the clips: ``static`` fired on 9 unconfirmed side events (all dropped: 4 cam2-only licks) and on 13 confirmed ones
(kept); ``far`` fired on none (confirmed side events' median position >= +5.6 px, threshold far behind the mouth).

**Whole-session path:** `python -m scripts.camera_licks session A:YYYYMMDD --bundle <side bundle>` (same functions,
`fuse_session(side_bundle=...)`, production events unrestricted). Untested until a whole-session side bundle exists.

**Follow-up (Priya 11-18): "these sound like errors if not in 3d and always after misses (water drop), right?" -- partly.**
Of the 114 long (>= 300 ms) cam1-only events, 65 had no miss in the 3 prior trials (59 in CONTACT trials, late,
median 1.8 s after the cue: the tongue / a drop at the spout after drinking; class unaffected, contact = DAQ) and 49
followed misses (31 trials now "incomplete": 11 with no other lick evidence and 12 with only short cam1-only events ->
no response, none had a jaw movement; 8 also have a cam4 lick and stay incomplete). ~23 trials would move, 11 acute.

**Rule ADOPTED in code (Priya 11-18: a "stationary and jumping from the mouth" rule, applied "before cleaning /
interpolating"):** `cam1_licks.mask_detached` -- a run of raw confident cam1 tongue frames (gaps <= 24 ms joined) that
never comes within +30 px of the mouth line and has a spread < 6 px is set to "not visible" BEFORE `orofacial_clean`.
Wired at both cam1 cleaning sites: `scripts/cam1_licks.py run` (lick events) and `pose3d.cleaned_xy` (3-D, so also
`pose3d_side.fuse_session`). `cam1_frame` moved verbatim from the script into `cam1_licks` (re-exported). Evidence
(76 sessions; code in `cam23_rerun_20261118/code/`): 0 / 118 blind-review tongue runs masked; 9 / 10 no-tongue runs in
front of the mouth caught; trial 540 1,099 / 1,124 frames; 0.30 % of confident frames masked over 25 sessions, 7.6 % of
those inside a cam4 lick. NOT caught: no-tongue labels at the mouth line (7 / 24; not separable by position).
**cam4 keep (Priya 11-18: "are these real? should we keep those with a cam4 correlate?" -- yes):** of 12 sampled
masked runs inside a cam4 lick, 8 were the real tongue held at the spout, 2 grooming, 2 the label on a drop as the tongue
retracted; 6 / 6 sampled masked runs without a cam4 lick were a drop or the bare rod (sheets
`DeepLabCut/Widefield/session_poses/cam1_licks/drop_mask_check_20261118/masked_sheet_{0,1,2}.png`, key beside them;
code `masked_sheet.py`). So a run overlapping a production cam4 lick (`cam1_licks.cam4_lick_windows`, `licks_all.csv`)
is never masked: `mask_detached(keep_windows=)`, threaded as ``cam1_keep`` through `pose3d.cleaned_xy` /
`session_3d` / `pose3d_side.four_camera_points`; passed by `scripts/pose3d.py` production, `pose3d_side.fuse_session`
and `scripts/cam1_licks.py`. Other (QC) callers without windows mask unprotected and print so. Trial 540 is unchanged
(1,099 frames masked; cam4 saw no lick); PS93 0814 overall 1,684 -> 1,665 masked frames.
**Takes effect when cam1 events / 3-D are regenerated (the re-run): do NOT regenerate single sessions before it, or
the cohort mixes two cam1 definitions.** Bump `lp_licks.LP_VERSION` with that regeneration.

Open: the clip-edge licks (side frames end at +1.5 s), and the first whole-session run.

## 0c. 11-21 whole-session pilot (two sessions; branch `pilot/side4cam-wholesession`, not merged)

**Decisions (Priya 11-21 ~03:00-03:15; DECISIONS.md 2026-11-21):** D1 = (a) new cam4 2-D RUN v37 with the spout fix
(recorded only, not started); D3 = licks from every camera over the FULL session. **D2 REVISED ~04:00** (after the
numbers below): "cam 1 DOES capture the tiny tongue at the lips licks so we shouldn't get rid of those, and maybe we'll
just use the cam1 rules to get rid of water?" -> a lick from ANY camera counts (cam1 after `keep_mask`, side views only
when confirmed, cam4 alone counts), and water drops = cam1's rules plus the narrow form of (c): a cam1-only lick
>= 300 ms that no other camera sees is dropped (`camera_licks.lick_train` default; two views stays an option).

**What ran** (O2 side run `lp_side_r2b_prod_20261118`; PS93 0814 + PS94 0817, both cam2 / cam3 videos verified):
1. `python -m wfield_local.o2_inference collect --name lp_side_r2b_prod_20261118 --sessions PS93:20260814 PS94:20260817`
   (locanmf env): frame counts match the videos; new folders `session_poses/<a>_<d>_cam2_full`, `_cam3_full` (575 and
   643 trials). 8 min for both sessions (~2 min per video).
2. `python -m scripts.pose3d production <s> --four-cam lp_side_r2b_prod_20261118 --out
   session_poses/pose3d_4cam_pilot_20261121` (dlc env), 12 and 7.5 min per session. Written to a PILOT folder, not
   `pose3d_4cam/`: the production run must carry the parked fixes (section 0), and the cam1 events in use predate
   `mask_detached` (section 0b). Ran without errors.
3. `python -m scripts.camera_licks session <s> --bundle lp_side_r2b_prod_20261118` (dlc env), ~8.5 min per session:
   `session_poses/camera_licks/lp_side_r2b_prod_20261118/<a>_<d>_{fused,events}.csv`, `views.csv`. Summary script:
   `C:/Users/SabatiniLab/lp_stage/pilot4cam/fusion_summary.py` (logs beside it).

**3-D (four cameras vs today's two):** PS93 0814: 2,203,161 frames (= the two-camera pairing); tongue in 3-D 11.3 % of
frames (cam1 + cam4 alone 8.3 %, one-view fill 1.6 %); jaw 99.98 %; world fit residual <= 0.04 mm; cam1 tongue
reprojection 3.4 px; 14,447 droplet frames. PS94 0817: tongue 3.4 % (cam1 + cam4 2.4 %), fit <= 0.10 mm, 71 droplet
frames. Per-lick medians barely move (PS93 reach 4.33 -> 4.41 mm, miss 1.36 -> 1.29, lateral error -0.79 -> -0.74;
PS94 reach 3.75 -> 3.70, miss 1.01 -> 0.81, lateral 0.36 -> 0.29).
**But fewer 3-D licks: PS93 5,851 -> 5,599 (-4 %), PS94 1,718 -> 1,618 (-6 %),** while 5,786 / 5,851 and 1,704 / 1,718
two-camera licks have a four-camera peak within 40 ms. Two causes (worked out ~04:00, see "Splitting" below): mostly the
TWO-camera count is inflated -- `pose3d.lick_peaks` breaks a run at any missing frame, and 273 two-camera PS93 "licks"
are 32 ms fragments of a neighboring cam4 lick split off by a one-frame dropout (8 ms gap in 55 %, tongue 3.8 mm out:
not a return to the mouth; 270 two-camera inter-lick intervals < 40 ms vs 37 four-camera); the four-camera track
fills those dropouts. A smaller part is real merging: 62 four-camera runs (40 two-camera) hold two cam4 licks.

**Lick fusion, whole session** (licks; share of all fused licks):

| | PS93 0814 | PS94 0817 |
|---|---|---|
| fused licks (union) | 10,691 | 3,072 |
| seen by all four cameras | 8,194 (77 %) | 2,273 (74 %) |
| cam1-only (median 28 ms; >= 300 ms) | 681 (6.4 %; 9) | 409 (13.3 %; 27) |
| cam4-only (median 24-28 ms) | 222 (2.1 %) | 71 (2.3 %) |
| side-only (no cam1 / cam4 event) | 282, 247 guarded (dropped), 35 kept by the union | 6, 1 guarded, 5 kept |
| train, union (min_views 1) | 10,444 | 3,071 |
| train, D2 = two views | 9,509 (-935, -9 %) | 2,590 (-481, -16 %) |
| production cam4 licks seen by >= 1 other camera | 9,429 / 9,651 (98 %) | 2,529 / 2,600 (97 %) |

Coverage (D3): the side windows are the same trial windows as cam1 / cam4 (strobe - 0.5 s .. stop + 3 s, cut at the
next trial) and cover 99.7 % of PS93's video (2,203,161 / 2,209,339 frames); only the minutes before the first and
after the last trial are outside. Licks by where they fall (two-view licks / all): PS93 pre-cue 3,564 / 4,004, cue to
stop 5,297 / 5,904, after stop 649 / 783; PS94 943 / 1,129, 1,573 / 1,803, 74 / 140. So the full-session train
(D3) needs only the edges added (frames before trial 1 / after the last trial), not a new detector.

**Findings / open questions for Priya:**
- **D2 = two views costs 9-16 % of licks**, almost all single-view events of 16-44 ms (four-view licks: 76-104 ms).
  PS94 far_center: 188 of its 241 licks are cam1-only. They are either short tongue flicks only the view below sees or
  cam1 false positives at that spout. Review a clip sheet of cam1-only far_center events before adopting D2 = 2.
- **Fusion merges consecutive licks**: 310 fused PS93 licks contain 2-4 production cam4 licks (cam4 9,994 events ->
  9,651 fused licks). 218 of them because cam4's OWN v36 spans overlap (225 overlapping consecutive pairs; `fuse`
  chained them with no other camera), the rest bridged by a side / cam1 event. Fixed ~04:00, see "Splitting" below.
- Side detector fragments licks (about 1,450 PS93 licks hold 2+ events of one side camera): harmless after fusion.
- 35 side-only licks kept by the union in PS93 (0.3 %); with D2 = 2 they need both side views (4 in PS93, 4 in PS94).
- One cam3 video (local run, task 208, PS95 0603) has 1 frame more than its timestamp CSV; `collect` checks
  against the video's own frame count, so it will pass. Check that the template timing handles the extra frame.

**Splitting back-to-back licks + D2 revised (11-21 ~04:00, commit d207a12a).** Priya: "yes can try splitting at other
camera lick boundaries - but shouldn't we also be able to tell based on the position the tongue returns to (the
'mouth')?" Rules compared on both sessions (`C:/Users/SabatiniLab/lp_stage/pilot4cam/split/compare.py`, log
`compare.log`; re-fused from the saved `_events.csv` and the pilot 3-D frames, no re-run of `fuse_session`):
(a) cut at the cam4 (then cam1) lick boundaries (`camera_licks.event_cuts`: middle of the gap or overlap, between the
peaks); (b) cut at prominent minima of the 3-D tip-to-mouth distance (`camera_licks.mouth_cuts`, prominence 0.5-1.5
mm, 40 ms peak floor); (a+b) = (a)'s count, placed where the 3-D tongue is closest to the mouth between the two peaks,
and no cut when the tongue does not return inside one unbroken run (`event_cuts(..., S=frames)`).

| PS93 0814 (PS94 0817) | 3-D licks (2-cam 5,851 / 1,718) | 3-D licks with 2 cam4 peaks | 3-D ILI < 40 ms | fused licks | fused with 2+ cam4 | train, D2 revised |
|---|---|---|---|---|---|---|
| no split | 5,599 (1,618) | 62 (0) | 37 (17) | 10,684 (3,071) | 310 (14) | 10,428 (3,043) |
| (a) boundaries, span midpoint | 5,741 (1,633) | 0 (0) | 84 (30) | 11,050 (3,098) | 0 (0) | 10,794 (3,070) |
| **(a+b) boundaries at the tongue's return** | **5,661 (1,622)** | **14 (0)** | **37 (18)** | **11,025 (3,098)** | **19 (0)** | **10,768 (3,069)** |
| (b) alone, 0.75 mm | 5,725 (1,625) | 19 (0) | 39 (18) | 10,813 (3,075) | 253 (14) | 10,557 (3,047) |

- (a) at the span midpoint makes 47 PS93 3-D "licks" of 3-4 frames: the cut lands on the first lick's retracting
  tongue and the piece's farthest frame sits right after it (ILI 37 -> 84). Placing the cut at the tongue's return
  removes all of them (37, as uncut).
- The 14 PS93 3-D runs still holding two cam4 peaks (a+b) are a held tongue: ~3 mm out for 70-120 ms with short cam4
  events at both ends and no return between them, so no split (Priya's rule).
- (b) alone agreed poorly with the camera boundaries: of (a)'s 179 cuts inside PS93 3-D runs, 38 had a (b) cut within
  20 ms (0.75 mm); of (b)'s 126 cuts, 37 had an (a) cut. It splits only 72 of the 310 merged fused licks and adds cuts
  inside single licks (the tip distance dips and re-extends within one lick). Tongue return between two cam4 licks
  that shared a run: median 1.37 mm (10 % below 0.77 mm). Kept as an option, not the default.
- **Recommended and now the default: (a+b)** in `camera_licks.fuse` (scripts/camera_licks) and
  `pose3d.lick_peaks` (scripts/pose3d `--four-cam`). Four-camera 3-D licks are then -3 % / -6 % vs two-camera,
  explained by the two-camera fragments above (not by merging).
- **D2 revised counts** (train = any camera, guarded side-only dropped): PS93 10,768; keeping cam1-only >= 300 ms too
  10,777 (the rule drops 9); two views 9,788. PS94 3,069; 3,096 (drops 27); 2,595.
- Still open: whether `lick_peaks` should bridge one-frame dropouts (as `pose3d.lick_events` bridges 12 ms) -- it
  would remove the two-camera fragments; the four-camera run mostly does not need it.

## 1. Decisions needed before running (Priya)

| # | decision | why it matters | options / default proposed |
|---|---|---|---|
| D1 | **DECIDED 11-21: (a)** Does the cam4 **2-D** chain (LP run `orofacial_cohort.RUN = lp_r4_heatmap_20261013__clean_v36`) get regenerated in this re-run, under a new run tag? | The spout-rule fix (item 1 above) changes the v36 params digest, so it cannot be patched under v36. A new RUN moves EVERY output folder under `LP_analysis/<RUN>/` (licks, kinematics tables, `arhmm/`, `glm_hmm/`, `imaging_encoding/qc/` s1 snapshot, encoder outputs) | (a) new RUN `..._v37` with the spout fix, everything downstream re-pointed (proposed: one clean regeneration); (b) keep v36 2-D, re-run only 3-D and licks (cheaper; 2-D readouts keep the small, unbiased spout deviation) |
| D2 | **DECIDED 11-21, REVISED ~04:00: any camera (cam1 after its rules, side views when confirmed), cam1-only >= 300 ms dropped (see 0c)** **Lick fusion rule** for four cameras | Defines LP_POST, camera classes, RESTnl, quiet-event exclusion, encoder lick kernels | union of cameras with duplicate merge (as cam1 today: cam1-only events added when not cam4-matched, droplets removed) vs requiring 2 views; droplet / far-from-mouth guards per side camera |
| D3 | **DECIDED 11-21: the full session, every camera** **Lick coverage**: post-cue only (as cam1 `_events.csv` today) or the whole trial window (`events_all`) | Today the ENL lick-free window is cam4-only because cam1 is scored only cue -> stop (`LICK_SOURCE_AUDIT_20261117.md`). Whole-window side licks would make the pre-cue / ITI / rest exclusions four-camera too | proposed: whole trial windows for every camera, so one train serves every window |
| D4 | **Kinematics in 3-D or 2-D** -- DECIDED (Priya 11-18): "kinematics - we'll want mostly 3d" | The kinematics page mixes cam4 2-D (v36) and cam1 + cam4 3-D readouts | the kinematics readouts move to the four-camera 3-D (`pose3d_4cam`); 2-D only where a readout has no 3-D form |
| D5 | **GLM-HMM / AR-HMM: refit or re-infer** | A refit renumbers states and changes every state figure; re-inferring with the saved parameters keeps the states | proposed: re-infer with the saved parameters (`glm_hmm/v1/params_pooled_all_K5_v2.json`; AR-HMM fits on the share) and report how many trials change state; refit only if the inputs moved a lot |
| D6 | **Frozen pre-stroke decoders / encoder** | If the working gate's pre-stroke trial set changes, `spec_id` changes and new frozen models get fitted; the old ones stay as the reference (CLAUDE.md rule 10, `FROZEN_MODELS.md` section 7) | re-fit under the new gate as a new spec, keep the old store, compare |
| D7 | **What counts as "changed enough" to re-render** | 870 files depend on these inputs (section 5); re-rendering everything is days of compute | proposed: measure first (section 3 checks), re-render everything whose inputs changed, and for families where < x % of trials change class, record the change and re-render on the next scheduled render instead |
| D8 | Cohort 2 | Do PS118 / PS120-122 get cam2 / cam3 too? | this inventory is cohort 1 (PS92-95) unless Priya says otherwise |

## 2. Run order (upstream first)

Each layer reads only the layers above it. Do not start a layer until the one above is checked (section 3).

| layer | what | code / command | output |
|---|---|---|---|
| L0 | cam2 / cam3 whole-session LP (round-2 side model) | O2, Priya runs; runbook `runbooks/o2_pose_inference.md`; side bundle from main | side predictions (model space) |
| L1a | side cleaning + four-camera 3-D | `python -m scripts.pose3d production --four-cam <side bundle>` (`pose3d_side.fuse_session`); with bddf02e7 + commanded spout table; `scripts/side_label_offsets.py` (Part B labels) | `session_poses/pose3d_4cam/` (frames, licks, kinematics, phase, cue traces) |
| L1b | (if D1 = a) cam4 2-D chain under the new RUN with the spout fix | `orofacial_outputs.py` (line ~184 -> `spout_frame.ruler_medians`), `tongue_kinematics.trial_spout_reference` -> ENL medians, the whole-window scripts in the spout audit | `LP_analysis/<new RUN>/sessions/...` (`licks.csv`, `licks_all.csv`, trials, jaw, precue, ITI) |
| L1c | side-camera lick detector + four-camera lick events (NEW CODE) | model on `cam1_licks.py` (`detect`, `keep_mask`) + `scripts/cam1_licks.py`; fusion per D2 / D3 | `session_poses/<camN>_licks/` or one fused table |
| L1d | close-spout jaw from cam2 / cam3 | `jaw_onsets` / `trial_classes` jaw rule; review sheet like `scripts/cam1_jaw_review.py` | jaw calls at close_center / close_mid |
| L2 | shared definitions | `lp_licks` (bump `LP_VERSION` 3 -> 4; cache tag follows the camera folders), `trial_classes` + `scripts/trial_attempts.py`, s1 snapshot `scripts/s1_trial_class_snapshot.py` (-> `<RUN>/imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz`, read by `working_trials`), `gate_source` (LP gate), `restnl_mask` (+ the cam1 RESTnl branch `analysis/restnl-cam1-1117`), `iti_licks` extracts, `lick_provenance` wording | the trial classes, working gate, LP lick train, RESTnl mask every branch reads |
| L3 | branch analyses | section 4 | pages, decks |

`session_cache` / `map_collect` caches: bump `CACHE_VERSION` if a cached function's logic changes (CLAUDE.md rule 5);
`lp_licks` caches key on the cam1 folder signature, so add the new camera folders to that signature.

## 3. Checks before anything downstream (old vs new, per session; write them to `cam23_rerun_20261118/`)

1. **3-D**: |tongue tip ml - spout ml| at contact vs the rod radius (0.61 mm in cam1 + cam4); four-camera vs two-camera
   tip position per lick (median / p90 shift, by spout); coverage gained (frames with a 3-D tongue / jaw); spout
   residual vs commanded (today median 0.05 / max 0.12 mm).
2. **Licks**: per session and camera, lick counts; licks added by cam2 / cam3 (by window: pre-cue, post-cue, ITI);
   droplet / far-from-mouth rejections; agreement with DAQ contact.
3. **Trial classes**: old -> new class transition matrix per epoch and spout (contact / incomplete / jaw_only /
   no_response); how many close_center / close_mid trials gain a jaw call; working-gate membership changes per
   session (this is what moves every gated imaging family).
4. **RESTnl**: rest frames lost per session from the added licks.
5. Print the per-animal table first (rule 8). Show Priya 1-4 before L3.

## 4. Inventory by branch (what to re-run, and why)

Inputs: **3D** = pose3d tables; **LICK** = camera lick trains (LP train, `licks_all`, cam1 events, RESTnl);
**CLASS** = camera trial classes / working gate; **SPOUT** = the cam4 2-D spout frame and kinematics chain (only if
D1 = a). Page = the results page that shows it.

### B. Pose / 3-D (owner: pose window)
- [ ] `pose3d_4cam` production (L1a) and its checks: `scripts/pose3d_side_check.py`, `pose3d_side_loo.py`,
      `pose3d_side_video.py`, `side_clean_qc.py`, `pose3d_crosscheck.py`, `pose3d_spout_error.py`,
      `pose3d_vs_cam4.py`, `pose3d_jaw_crosscheck.py`, `pose3d_check_sheet.py`, `pose3d_close_dir_sheet.py`
- [ ] `apparatus_mouth.py` (3-D mouth), `pose3d_june_reference.py` (June, once intrinsics are checked)
- [ ] `docs/POSE_MODELS.md` Part E: the production record

### D. Trial classes and gates (shared; owner: whoever runs L2)
- [ ] `trial_classes` (incomplete from four cameras; jaw at the close spouts; `jaw_judged`), `scripts/trial_attempts.py`
- [ ] s1 snapshot (`scripts/s1_trial_class_snapshot.py`) -> `working_trials` (`in_working`, `in_working_jaw`)
- [ ] `lp_licks` (LP_POST, LP_PRECUE, first-lick timing), `gate_source` (LP gate), `lick_provenance` wording (the
      caption phrases change: "cam4 + cam1" -> four cameras)
- [ ] `restnl_mask` / `restnl_licks`; `scripts/restnl*.py`
- [ ] `iti_licks` extracts (first licks, ITI licks; read by state space, FR variance, cued ramp)
- [ ] tables: `camera_coverage_tables.py`, `lp_trial_counts.py`, `trial_class_raw_share_fig.py`, `misdirected_lick_counts.py`

### A. Behavior (behavior page)
- DAQ readouts (hit rate, latency, DAQ licks, engagement, quit) are UNAFFECTED.
- [ ] camera-class panels: `scripts/within_session_decline.py`, `disengagement_tests.py`, `trial_class_raw_share_fig.py`,
      `jaw_only_incomplete_figs.py`, `stopped_trial_traces.py`

### C. Kinematics (kinematics page; OROFACIAL_FINDINGS.md)
- [ ] 3-D: `scripts/pose3d_kinematics.py`, `pose3d_phase.py`, `pose3d_cue_traces.py`, `pose3d_summary_fig.py`,
      `pose3d_figures.py`, `lick_trajectories_3d.py`, `movement_position_angle.py`, `lick_path_softmax.py`,
      `contact_direction_predictability.py`, `lick_vigor_perspout.py`, `kinematic_normality.py`, `lmm_curve_summaries.py`
- [ ] lick-level / aim: `cam1_lick_aim.py`, `face_prelaunch_aim.py`, `prelaunch_aim_tracking.py`, `lick_correction.py`,
      `within_lick_correction.py`, `tongue_angle_phase.py`, `first_lick_latency.py`, `iti_first_lick.py`, `iti_lick_rate.py`
- [ ] jaw / nose: `jaw_lateral.py` (+ `orofacial_jaw_lateral`), `jaw_onsets.py`, `nose_onsets.py`
- [ ] 2-D cohort tables (if D1 = a): `orofacial_cohort` (cohort tables, prepost stats), `orofacial_outputs`,
      `orofacial_heatmaps`, `orofacial_w2`, `kinematics_pooled_figs.py`, `session_lick_summary.py`, `kinematics_checks.py`
- [ ] **lick-template test** `scripts/lick_template_match.py` (Priya 10-07 MUST REVISIT)
- [ ] attempts: `jaw_only_incomplete_figs.py`, `cam1_incomplete_licks.py`, `rebuild_figs/fig_farcontra_attempts_sessions.py`,
      `rebuild_figs/fig_pose_validation_sessions.py`

### E. FaceRhythm (FaceRhythm page)
- Factors and projections come from raw video: UNAFFECTED (`face_rhythm/`, `master_pre_v1`).
- [ ] lick-window readouts (cam4 `licks_all`): `face_rhythm_readouts.py` (lick windows, asymmetry, lick rate),
      `face_rhythm_lick_epochs.py`, `face_rhythm_validate.py`, `fr_enl_decode.py` (LP train + DAQ),
      `enl_fr_removed_decode.py`
- [ ] gated FR panels (working gate), incl. FR Fig 5.6 cue-aligned factors (from the states collector, below)

### F. Spout (target) coding page (imaging; rebuild root `rebuild_20261103`)
- DAQ lick class, DAQ alignment and the restdock05 reference are UNAFFECTED; the trial SELECTION under the camera gate
  (`sel-camera-...` run tags) is affected through the s1 snapshot.
- [ ] every family rendered under the camera gate: `epoch_grant_figures` / `grant_figures` families via the rebuild
      queue (`docs/status/REBUILD_CODE_20261103.md`, `rebuild_20261103/`), incl. 15r maps, decoders, coding
      directions, RSA, retained fraction
- [ ] LP_licks family (`WIDEFIELD_LICK_SOURCE=lp`): `scripts/lp_final_render.py` (maps, decoders),
      `nolick_decoder`, `attempt_coding`, `attempt_maps`, `enl_window`, `precue_lickfree`, ENL censuses
      (`enl_lp_census.py`, `enl_iti_cam1_census.py`, `enl_lick_audit.py`)
- [ ] frozen decoders keyed on the gated pre-stroke set (D6): `locanmf_frozen_decoder`, `frozen_models`,
      `rest_migration/rest_frozen_decoder_nl.py`
- [ ] RESTnl arm: `rest_migration/rest15f_restnl_1117.py`, `restnl_maps.py`, `restnl_position_coverage.py`
- [ ] misdirected / attempt families: `cued_misdirected_licks.py`, `misdirected_lick_decode.py`, `relabel_fc_by_class.py`
- [ ] analysis deck key figures (`deck_keyfigs`, `locanmf_analysis_deck`) and the page's lick-source provenance table
      (`docs/results_pages/imaging/provenance/lick_source.tsv`)

### G. Movement (execution) coding page: encoder
- [ ] `combined_encoding` (TARGET3D spout table, EXEC3D lick table = pose3d `kinematics/licks.csv.gz`, lick kernels,
      `OROFACIAL_GATE=attempted`, tongue analogs from 2-D) -> production re-run (O2 `scripts/o2_encoding.py`; diff the
      new bundle's env.json against the last production bundle first, and smoke-test the exact command locally)
- [ ] `frozen_encoder` (pose3d inputs; lick_onset / licks_pre / licks_iti trains) and its figures
- [ ] decoding of tongue position / direction from cortex (movement page), `softmax_neural_vs_lick.py`,
      `trial_neural_vs_exec.py`, `exec_bins_calibration.py`, `exec_vs_neural_match.py`, `within_spout_deviation.py`
- [ ] lick-balance / lick-recovery family (`scripts/lick_balance_*.py`, `lick_recovery_*.py`, `lick_enlv_*`,
      `lick_cued_ramp.py`, `lick_iti_*`)

### H. Quiet events (movement page section 23; PI deck; owner a8)
- [ ] `quiet_spikes_v2` lick exclusion (`QS_LICK_SOURCE=daq+lp`: DAQ + cam4 `licks_all` + cam1 `events_all`) -> add
      cam2 / cam3; `cohort2_qc/lp_licks.py`; `cohort2_qc/quiet_task_events.py` (pre-cue gate),
      `quiet_lp_sensitivity.py`, `quiet_lick_lfo.py`, `quiet_area_spikes.py`
- [ ] the restyle / section-23 runners on the sync-end run (a8's current chain)
- [ ] **HOW to re-run (a8, 11-18):** ONLY via `scripts/cohort2_qc/run_quiet_chain.sh <OUT_DIRNAME>` (on main, b58420f8+)
      from a clean checkout. It runs the whole chain in order: base chain, section-23 follow-ons, T13 follow-up,
      restyled deck / page figures. Every 11-18 decision is in the code (last-sync end incl. quit period, k 4, cam1
      guard, bootstrap stars, stable seed, quit cache). No hand-run pieces, no cross-key patches.
- [ ] **Prerequisite** (after the new LP licks + FR series): the attempt_coding pre-cue tables under the SAME env / key
      (`WIDEFIELD_LICK_SOURCE=daq`; may freeze decoders, CLAUDE.md rule 10). The runner's preflight refuses to start
      without them, and `quiet_task_events` fails on a missing table. Never read T13 p_true across keys
      (`task_events_summarise_prodptrue.py` is marked do-not-reuse).
- [ ] Fig 23.23 (T13 follow-up) stays on PRE-rebuild data by Priya's choice (11-18); it moves to the new build only
      through this runner.

### I. State space page
- [ ] `state_space_enl_baseline.py` (LP train), `state_space_rest_baseline.py`, `state_space_rest_decomp.py` (RESTnl),
      `state_space_shift_movement.py`, `state_space_shift_behavior.py`, `state_space_landing.py`,
      `state_space_sim.py`, `state_space_balance_angle.py`, the aligned decoder (`aligned_decoder.py`, gated),
      `state_space_fr_coupling.py`; the iti_licks first-lick extracts

### J. Behavioral states page (this window's arm; `behavior_states.py` is the single source)
- [ ] GLM-HMM (inputs: camera classes + cam4 pre-cue / ITI licks + DAQ running): re-infer with the saved K5 v2
      parameters or refit (D5): `python -m wfield_local.glm_hmm --pooled-all --k 5 ...`
- [ ] AR-HMM features (`arhmm_features`, VERSION 3: pose3d jaw + v36 2-D tongue / nose) -> bump VERSION, rebuild
      features, re-infer (or refit on O2, `scripts/o2_arhmm.py`); syllable reels; `arhmm_states`
- [ ] TIM (`task_independent_movement`, `TIM_RUN`), frozen pre-stroke likelihood
- [ ] collector + figures: `python -m wfield_local.behavior_states collect`, `scripts/states_page_figs.py` (all
      families), GLM-HMM reels (`behavior_states reels`); FR Fig 5.6 comes from the same collector

### Decks and grants
- [ ] analysis deck (auto, nightly); PI deck (a8)
- K08 grant figures (`grants/k08_2026/`) were submitted: do NOT regenerate; record any changed number instead.

## 5. Appendix: full dependency scan

`docs/status/cam23_rerun_20261118/dependency_scan.csv` (from `scripts/rerun_dependency_scan.py`, 11-18): every module /
script outside `archive/` and `tests/` that reads one of the four inputs, with its direct and transitive tags. 333 files
read an input directly (3D 105, LICK 234, CLASS 87, SPOUT 129); 537 more only through a `wfield_local` import. The scan is a
regex over imports and file names, so it over-includes (QC, review and demo scripts); section 4 is the curated list of
what feeds a page or a deck. Re-run the scan after the re-run's code changes and diff the CSV.
