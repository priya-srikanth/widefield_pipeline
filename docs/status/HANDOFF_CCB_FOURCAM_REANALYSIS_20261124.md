# CC-B handoff, 2026-11-24 evening: the four-camera 3-D re-analysis

**Read this first.** It records today's state for the next CC-B window, the lanes and Priya. Companion files:
- `vls_stroke_paper/CHANGES_SINCE_20261124.md`: manuscript-relevant changes for the next draft.
- `run_records/2026-11-23/INCONSISTENCY_FIXES.md`: the 11-23 to 11-24 decision record.
- `run_records/2026-11-23/ITI_BLAST_RADIUS_20261124.md`: every consumer of the buggy ITI tables.
- `run_records/2026-11-23/O2_G4_READY.md` and `O2_FROZEN_WAVE1_READY.md`: O2 bundles and commands.
- Lane E's `lp_stage/lanes_20261124/laneE/` (logs, `DELETE_CANDIDATES_laneE_20261124.md`, `changelog_notes_laneE.md`).

## 0. Objective
Every analysis we have (results pages, PI deck, manuscript panels) is to be repeated on the **four-camera 3-D keypoint
data**. That means the inputs below, each verified from the inputs a figure actually read (not from the settings a run was meant to
use), and recorded in that figure's caption provenance:
- **Licks:** the fused four-camera lick list `lp_side_r2b_prod_20261118_3net_20261121_sync60_gapfix60`. It covers cam1-4;
  licks seen only by cam1 (below) are excluded. Joined everywhere by `lick_key`, never by time.
- **3-D:** `session_poses/pose3d_4cam_3net_20261121_sync60`, the adopted run (cleaned tongue and jaw; nose raw at
  confidence >= 0.9). PS95_0818 / 0820 use the cam3-on swap (11-23).
- **Trials:** jaw rule `3d_mouth`, camera working period, the attempted gate. Encoders use `OROFACIAL_GATE=attempted`.

## 1. Who is doing what

| Who | Scope |
|---|---|
| **Priya** | Decisions; all O2 submits and copy-backs; publishing; account switches (HMS = where the results pages live) |
| **CC-B** (this window) | Coordination, O2 commands, decisions relay, change list, this handoff |
| **Helper "inconsistency"** (CC-B agent) | Imaging queue J1-J6 (`lp_stage/imaging_fourcam_20261123`), rev-up reruns, figure restore |
| **Lane E** (CC-B agent) | Movement page: ITI recount and refit, docked classes, readers2, S13F rebuild, page rebuild and CHANGELOG |
| **Lane F** (CC-B agent) | G4 O2 bundle; frozen wave 1 (A/B) local bundling plus O2 port |
| **Lane B** (CC-B agent, session ended) | FaceRhythm band-power O2 job (collected); J7 maps, then Figs 8.2-8.4 |
| **Naming helper** (CC-B agent) | Phase 2: short distinct titles, provenance tables, the Fig 2.1 re-run, on all pages |
| **O2-port helper** (CC-B agent) | Frozen-frame input build on O2, for wave 2 |
| **CC-A** (other window) | Manuscript: S13F redraw, main.tex text checks (never edited by CC-B) |

## 2. Errors found today (what, impact, fix, status)

### 2a. Wrong or incomplete data used
1. **ITI timing audit joined by exact time to cam4 `licks_all.csv`.**
   - Impact: 17.8% of ITI licks had no gap value, and the previous-lick lists were cam4-only, so some gaps were too long.
   - Fixed (lane E b50dfe5e): fused table, `lick_key` join. Rebuilt for 107 sessions.
2. **ITI lick-order cache (11-22) had the same bug.** Impact: it kept only ~82% of fused ITI licks (69,747 of 84,827).
   - Recounted to 84,804 rows.
   - The frozen ITI lick-order decoders were REFIT as new store entries (PS92 a4fe3f25b0bb, PS93 9925f479160c,
     PS94 bc8284cc06a3, PS95 275ad70d633e). Priya approved option (a).
3. **Dock audit keyed on the truncated cache.** 143 docked flags flipped. Rebuilt on the recount
   (`iti_licks_dockaudit2_laneE_20261124`). The 69,831-row interim `iti_licks_dockaudit_laneE_20261124` is SUPERSEDED.
4. **Other cam4-onset readers.**
   - Fixed: iti_dock_audit, lick_order bout index, recovery markers, code_subspace, cue_to_lick, enlv extract,
     enl_angle_decode, revup scripts, contact / rise readers, iti_lick_rate, lick_iti_prelick,
     prelaunch_aim_tracking, frozen_encoder.py:478.
   - Now refuse in four-camera mode: lick_enlv_extract (needs the rebuilt enl audit), reembedding_test.
   - New static test: `tests/test_no_cam4_lick_tables.py`.
5. **PS95 08-18 / 08-20 read the 11-22 3-D (cam3 dropped).** Re-extracted and patched; a read guard refuses the old files.
6. **G4 bundle #1 (`enc_fourcam_20261124`) used 2-D jaw / nose plus signed linear running.** Cancelled (all jobs) and rebuilt
   as `enc_fourcam3d_20261124`.
7. **Execution-bin calibration was from the 10-23 cam4-era table.** Rebuilt on four-camera data; 7.2% of licks change hard bin.
8. **Push cap at the pre-only p99 clipped 5-10% of post-stroke pushes** (3 animals). G4 now uses per-animal all-session caps;
   `main_precap` is the sensitivity tag.
9. **Rev-up neural / followup took ITI licks from cam4-only `licks_all`.** Rerun (helper, d2fe3b0e / c906c0a6).
   - revup_neural: unchanged (554 stars).
   - followup part A: +3 stars on the whole-window excess.
10. **Learned compensation T2/T3 (S13F)** matched only 0.53 of ITI licks to the dock audit. Rebuilt (0.86). Primary
    definition under Priya's decision: see section 6.
11. **ACAd midline control plus 5 G10 readers** (offset_contrib_map, offset_where, gain_prediction, nonlinear_decoder,
    behaviour_decoder) used the truncated cache. They run in readers2. The 01:03-01:10 ACAd store entries are delete
    candidates.
12. **Behavior page.**
    - Fig 1.1's caption window was wrong. The code counts licks from cue + 70 ms to cue + 3.5 s (or the trial stop).
    - Fig 2.1 used a June + August pre baseline, which breaks the 11-07 own-August-pre rule.
    - Both are being fixed by the naming helper.
13. **Not on the sync60 lineage, but traps.** sm_coherence_cca and lick_nonlinear_preds would have read the buggy base
    tables. Fixed, with a guard that refuses a truncated dock audit.

### 2b. Silent fallbacks and process bugs
- **Docked where readers fell back to all-ITI caches when the docked ones were missing** (lane E's interim run only; the live
  v326 page was verified clean). The loader now refuses.
- **Docked-ITI filters read the production dock audit even when redirected** (b655d620). Fixed.
- **`iti_dock_where_preds` read unpatched caches.** Fixed.
- **Lane E's s12 env-stripping wrapper dropped its switches.** Fixed; the switches are passed as arguments.
- **Missing output folders** made one lo_build step proceed on the OLD cache. Stopped before any write and relaunched.
- **G4 smoke log was silent for 18 min** (no progress lines). Runner progress lines added.
- **The smoke left a `.lock` behind.** Lock removal is not done on that path: **bug to fix** (lane F) before the next bundle.
  Locks from running array tasks are released normally (seen as "file has vanished" in rsync).
- **Captions stated the INTENDED inputs, not the actual ones.**
  - New rule (memory figure-provenance-in-captions): read provenance from the loaded inputs.
  - Every extract records its lick set and 3-D source; footers print what was read; readers refuse a mismatch
    (lane E, in progress).
- **FaceRhythm O2 commands named an outdated conda module** (`24.11.3-0`). Fixed to `26.9.4-1`.

### 2c. Accidental overwrites and unapproved actions
- **Helper, 00:27:** overwrote 6 PRODUCTION rev-up behavior figures in `imaging_encoding/iti_licks/figures/`.
  - **Restored** 17:21 from the 11-23 cam3on tables. Provenance is in `README_revup_figures.txt`.
  - The helper's versions are in `revup_fourcam_jaw3dmouth_wp_ccB1123/figures_behaviour_jaw3dmouth_wp/`.
- **Lane E, 14:04:** a reader step (`lick_enlv_balance_offset --table`) silently froze 12 sync60 lick_reader entries
  BEFORE approval.
  - **Priya accepted them** after the fact (trained on the corrected fused licks).
  - The step is now load-only unless `--allow-freeze`.
  - The 11-03 v36 entries in labcams/frozen_models are KEEP.
- **Duplicate G4 smoke (60617630)** submitted by mistake; cancelled.
- **The widefield push used `--no-verify`** (Priya approved, AM). Origin/main is 82bc5e80.

### 2d. Reporting errors (CC-B and helpers), corrected in-thread
- **"122 GB free"** was disk, not RAM.
- **Lane F's "54/107"** was really 43/107.
- **The "PS95 newly drops from acute" S13F claim** was wrong; live never had PS95 in acute.
- **The "dock event 0.4 s late" claim** was wrong: trial stop = spout departure; the logged dock event is correct.
- **The "S1/S2 legend pending" item** was wrong; main.tex was already updated 01:54.
- **Lane E's first dependency search** missed 5 G10 readers; the independent blast-radius trace caught the rest.

## 3. Done and checked
- Integration merge to origin/main 82bc5e80 (targeted tests passed; full suite not run, by Priya's choice).
- Supplementary videos: S1 = mouse 11 v5 (median rule, 5-s cards); S2 = VFSS v2 (label dropped), in
  `supplementary_videos_final_v28/`. Legends already in main.tex.
- FaceRhythm band power on O2: 108/108 collected; every self-check <= 3.7e-7.
- G4 smoke on O2: PASS.
  - Structure identical to 11-10.
  - Settings verified.
  - Full cortex R2 on PS93_0814 rose 0.550 -> 0.681 (one session).
- G4 inputs: value-identical to the production build on the final code; the env diff shows only intended keys.
- Nose confidence cut: kept at 0.9. Lower cuts add < 0.4% of frames and 21-32% more jumps; the reproduction check is exact.
- ITI recount and decoder refit; dock-class count table; retraction readout (n.s.; does not support the latency hypothesis).
- Restore of the 6 rev-up production figures, with a README.
- ITI blast-radius trace across all arms. Clean: states, behavior, FaceRhythm, quiet / 23.23, rest masks, state space, spout
  page, kinematics ITI rate, S10, S13A-E, PI deck. The manuscript is touched only through S13F.
- The frozen frame frA does not use FROZEN_LICKSPOUT (no "|lickspout" in its run tag).

## 4. Done, needs verification or discussion (Priya)
- **S13F.** The primary is the docked class from the logged dock event with NO quiet rule (Priya, latest). The rerun result
  is pending from lane E.
  - Earlier, option (b) with the 1 s quiet rule: D_off -14.7* / -8.5* / -5.7*, n = 3 mice paired in every epoch (= live).
  - PS95 lacks a pre D_off: it needs >= 4 previous-spout cells with >= 10 docked licks, and has at most 2 under (b).
  - Open alternatives Priya raised: an unsplit or reweighted pre reference (option 1 / 2), to try to get PS95 back.
  - CC-A re-checks Results l.162 and Discussion l.216.
- **Docked-ITI definition.**
  - Logged dock event: median 0.93 s after trial stop (stop = spout departure); per spout 0.64-0.97 s. The fallback
    (stop + spout p95) was used only for PS92_0812.
  - Docked = bouts whose first lick starts at or after dock arrival; retraction class = bouts starting between stop and
    dock.
- **Quiet rule by measure type (Priya):**
  - behavioral landing: none;
  - at-lick neural readouts: >= 1 s;
  - neural windows before the lick or cue: >= the window length (2 s for -2 s windows).

  Lane E is checking each reader.
- **Rev-up 14.21 caption.**
  - The early excess stays ABOVE chance (+0.08 acute).
  - The whole-window excess is BELOW chance after the stroke (-0.25* / -0.22* / -0.20*).
  - Not a correction of the live direction. Swaps are wired.
- **Cumulative first-lick figure** (`revup_cumfirstlick.png`, new rule plus a cam3on twin; PS IDs): for the page, if
  wanted.
- **Frozen A vs B** (B = running and push refit per session): the comparison is "A as published vs B", not
  single-factor.
- **G4 running / push.**
  - All-session caps: #11's pre-only cap would clip 28-62% of post-stroke running.
  - Comparison tags: main_thr_nopush, main_thr_pushhinge, main_spline, main_precap, bins_w1_xyz2.

## 5. Running / pending (owner)

| Item | Owner | State |
|---|---|---|
| G4 `enc_fourcam3d_20261124` on O2 (20 tags x 107) | Priya / O2 | Smoke 60641941 passed. Complete: bins_w1/w05/w2, droppos. Others running (fr_A etc.). Priya re-syncs with `rsync -ahu`. |
| G4 collect + `g4_running_compare.py` | CC-B / lane F | After all tags reach 107 |
| Frozen wave-1 partial-refit check (O2 job 60661908, PS92_0806) | Priya -> lane F compare | Pending on O2; wave 1 stays HOLD until it passes |
| Frozen wave 1 local bundling (2/11 done: spont_frA A, spont_frA_freerp B) | Lane F, chain PID 44284 | Waits for 20 GB free; ~9-10 h |
| Readers2 on P2 (72 order-dependent + 23 independent), prelick2, rp_a | Lane E | Running, 8 GB gate |
| S13F primary (docked, no quiet) | Lane E | Rerunning; reply file for CC-A |
| ENLv comparison, caption diff (with the n-mice-per-figure table NEW vs LIVE), page rebuild, CHANGELOG | Lane E | After readers2 |
| Naming / caption phase 2 (all pages; Fig 2.1 re-run; untraced inputs from docs) | Naming helper | Running |
| FaceRhythm J7 maps / gradient / center, then Figs 8.2-8.4 | CC-B to run directly once RAM frees; lane B or a new helper for the figures | Power collected; the maps have NOT run yet (J7 waits for 12 GB) |
| Imaging queue: quiet chain (Fig 23.23, ~27 GB), lp_final maps | Helper "inconsistency" | Running; then the 23.23 old-vs-new compare |
| Wave-2 O2 input-build port | O2-port helper | Verification predicts in progress |
| S13F redraw + main.tex text checks | CC-A | Waits for lane E's reply |

## 6. Still to do
1. **After G4 lands:** collect each tag; the running/push comparisons; then the G4-fed figures (1.6, 1.13-1.15, 17, 21, 22)
   and the goal-summary step that failed on missing G4 output.
2. **Frozen waves:** wave 1 O2 submit (after the check), then collect and the A-vs-B readout. Wave 2 via the O2 port.
3. **Fix the lock-release bug** in the G4 runner before the next bundle (lane F).
4. **Movement page:** finish readers2, swaps, rebuild. Other pages: naming phase 2. **Nothing gets published until Priya
   reviews.** Pages republish from the HMS account; live movement page = claude.ai artifact Fnpn2KHknMwh3r13xADH7d.
5. **Figures still on the old lineage:**
   - recovery_link x4 / recovery_marker / check_figures need a camera-gated trial_attempts_sessions;
   - precue_restw, roi_*, slide95_sessz (pre-existing failures);
   - the RESTnl twins (no RESTnl levels on this lineage);
   - revup part B jaw energy (cam4 2-D jaw);
   - Fig 15.16 and 15.17 S1;
   - Fig 11.5's LocaNMF column, if not done.
6. **Delete list (Priya's OK required; never the frozen store):**
   - the cancelled G4 bundle (share and scratch);
   - one-session check bundles and the o2f test bundles;
   - lane E's superseded interim folders;
   - the ACAd 01:0x store entries;
   - `te_basis_only_laneE_timing_20261124`.

   See `DELETE_CANDIDATES_laneE_20261124.md` and `O2_G4_READY.md` A.9.
7. **Reboot the desktop when the queue is idle.** Windows holds ~14 GB in the paged pool; drive letters may move afterwards
   (check that M: = MICROSCOPE).
8. **Resume the 2pRAM window (sabatinilab-c3)** after the page and deck work, as promised.
9. **Push branches** (`ccb/laneE-1124`, `ccb/o2frozen-1124-*`, `ccb/revup-wp-1123`, `ccb/frB-1124`, the naming / port helpers)
   after review. Ask Priya first; merge, never rebase.

## 7. Standing rules confirmed today (also in memory)
- **Lab IDs PS92-95 everywhere** except inside the current manuscript files (#9-#12).
- **Caption provenance is read from the actual inputs;** unverifiable values are labelled "unverified".
- **Frozen pre-stroke models are new store entries only.** Never overwrite; never delete from either store.
- **Shared desktop:** 8 GB free-RAM floor, CREATE_NO_WINDOW, stop only your own exact PIDs.
- **Every manuscript-relevant change** goes in `vls_stroke_paper/CHANGES_SINCE_20261124.md`.

## Addendum, 11-24 ~18:45
- **Another accidental overwrite (lane E's own derived folder):** at ~15:13, rp_a's g9_where_scatter and g9_where_3axis
  wrote a reordered copy over the patch folder's `where_preds_splitr8.csv` / `where_3axis_preds_splitr8.csv.gz`. They
  wrote through a READ path (`pred_file`), and P2 inherited the copy.
  - Detected because the docked where-family steps failed the alignment check.
  - Fixed in code (`pred_file_out`, which writes only under OUT_X, with a test).
  - The bad copies were moved aside as `*_old_overwritten_1513` (delete candidates). The tables were restored from a fresh
    splice, `iti_licks_resplice_laneE_20261124`; `REPAIR_NOTE.md` is in P and P2.
  - Lane E is listing any reader that read the bad copies between 15:13 and the restore, to re-run it.
- **Quiet chain:** quiet_v2_fr finishes ~18:40; the rest of the chain (section 23 follow-ons, Fig 23.23, compare) takes
  ~4-6 h more. Cache-and-refilter design proposed (helper): ~3-4 h of code, ~1.5-3x faster; do it before the next
  rule change.
- **Efficiency audit (read-only) running:** `run_records/2026-11-24/EFFICIENCY_AUDIT_20261124.md`.
- **Naming phase 2 DONE on all 5 pages** (new `*_20261124_names.html`; the movement mapping went to lane E).
  - Fig 2.1 re-run against each mouse's own August pre: acute -.16* (misses Holm), subacute -.10** (4 of 4 mice).
- **S13F.** The primary is the docked class with no quiet rule (-14.1* / -7.9* / -4.7*, n = 3).
  - CC-A's new request: a pooled pre baseline for PS95, written up in `vls_stroke_paper/PROMPT_CC_B_S13F_pooled_baseline_2026-11-24.md`.
  - Lane E has it at top priority.
- **G4 on the share:** 11 tags at 107; fr_A 60, fr_B 0. fr_S, fr_D, xyz2 and the 4 running/push comparison tags were not yet synced.
