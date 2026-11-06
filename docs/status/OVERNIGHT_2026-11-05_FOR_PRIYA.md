# Overnight 11-05 -> 11-06: for Priya (decisions / questions first, then findings, then what ran)

Detailed log: `docs/status/RUNNING_LIST_2026-11-01.md` (11-05 entries). This page is the short version, updated through
the night.

## A. Decisions / questions waiting for you

1. **Slide 102 (gain_bio) port: five choices** (port done, branch `port/gain-bio-20261105`, not pushed):
   (1) copy the existing 1.6 GB FaceRhythm cache into the rebuild root (else ~0.5-1 TB re-read)? rec: yes;
   (2) quiet events: DAQ or the unified 1 s LP version? rec: LP 1 s;
   (3) task window: all cues before quit, or camera working rows only? rec: working rows;
   (4) the "415 is about half the gain" comparison mixes the new corrected signal with raw 470 / 415 -- keep and check?;
   (5) rest frames use the stored rest masks built with the OLD lick detector (affects all rest analyses) -- note as a
   limitation, or rebuild the masks?
2. **FR pixel maps: pre-stroke drift** (FR_PIXEL_CALIBRATION_20261105.md). The false-positive check is fine (4.6-4.9 %
   pooled), but late-pre differs from early-pre in 10 / 14 maps (4/4 animals same sign). unique_position cue chronic:
   all 232 significant bins share the pre-drift sign, 90 are drift-significant alone -> part of that "post-stroke change"
   continues a pre trend. Options: add a drift-control panel (late-vs-early pre, or epoch vs late pre); rewrite captions
   "max-statistic per panel" (they say Bonferroni, which is wrong everywhere it appears); 250 null splits + per-map
   seeds. Which?
3. **Spont co-variation (optional)**: add `no_spont` to the pixmusall twins / frAsplit_basis frames (cheap)? A "spont alone"
   Musall row needs a small code change + two frame re-renders -- only if you want a claim about spontaneous licks.

4. **(from a8, code cleanup) Bare hand runs and the pre-stroke cut.** Should a module run by hand with no profile and no
   WIDEFIELD_PRE_FROM default to the August-only cut (instead of pooling June + August)? It changes run tags and needs
   every June block (june_august_drift, restnl_mask, map_drift_prepre, reembedding) to pass an explicit "all". My rec:
   not before the replacement pass; for now only a loud WARNING when a cohort-1 run pools June; then the full switch with
   tests. (nightly_figs already defaults to August without a profile.)

5. **PS93 gets no pre-cue engagement discriminator even at the 15-trial floor** (only 12 late no-response trials before the
   stroke; PS92 has 16, right at the floor). Accept PS93 as missing, or relax the late window for PS93 only?
6. **chronic_stability's behaviour row** reads the 1b hit-rate table, which is now the camera working period. Keep it on
   camera (consistent), or keep that row on DAQ?
7. **Local disk cleanup:** the encoder input cache on C: grows 6-10 GB/h (85 GB free at 23:30); ~17 GB of 11-04 imaging
   entries are superseded. OK to delete superseded LOCAL cache entries (derived, reproducible)? A disk watcher now
   halts the render queue if C: < 20 GB free.

8. ~~**PI deck fluoroscopy wording**~~ ANSWERED 23:45 "can change" -> sent to a8. (a8): slide 161 bullet 2 says "about half the bolus"; with the new barium-amount
   measure it is closer to "about a third of the barium" (severe day 1 amount ~0.35x pre). Reword?

## B. Findings (early; nothing on pages changed)

- Mixed models, binned contest, LOG deficit (now primary): cue window -- spout identity tracks recovery (-8.1, q .015)
  and landing weakly (-3.15, q .004); lick window -- landing tracks recovery (-6.8, q < .001), spout identity does not
  (-4.5, q .55) though it is above pre post-stroke (+10.5, q .042). Log fits better than linear wherever there is a
  relation (dAIC -8 to -20). Figures: `.../imaging_encoding/<_daqlk tag>/figures/mixed_models/` (open markers, shape =
  animal, colour = epoch).

## C. What ran / is running (summary)

- Render `_daqlk`: lanes G, 2 (A B D E F Z), T, D, E, X. 11 / ~227 steps done at 22:34; per-step times 2-4x the
  estimates (network-bound at ~92 MB/s). Revised forecast: ~weekend unless the local input cache helps (measuring).

## D. Overnight updates (newest last)
- ~23:00: PI deck fluoroscopy update (your request): new all_graphs (10 panels, + G bolus density, H barium amount)
  to replace slide 159 + notes with the density / amount numbers; density_method as a HIDDEN supporting slide after 159.
  The deck is a8's (PI-deck window, has the fluoro scripts / backups) -> coordinating with a8 so only one of us edits;
  will not edit while PowerPoint has it open.
- ~23:30: **Decisions follow-up done** (branch fix/decisions-followup-20261105, wf_decisions2, unpushed): precue
  discriminator floor 15 (PS92 / PS94 / PS95 get one; PS93 not -- A5); 1b hit rate on the camera working period (DAQ =
  robustness figure; 1c stays DAQ so epoch boundaries stay pinned); iti_lick_rate + within_lick_correction cut to the working
  period; **trial table rebuilt under the new lick detector: IDENTICAL** to the 11-04 table (0 class / hit changes over
  57,048 trials) -> the O2 encoder runs need nothing.
- ~23:30: **Local input cache: not worth it mid-render** (RENDER_IO_LOCAL_CACHE_20261105.md): most heavy steps already read
  locally; the remaining network reads are ~1-2.5 h over 15+ h of queue (< 10 % gain); steps are CPU / RAM bound. Option
  for later: point the per-session result cache at a local copy (no code change). Disk risk found -> watcher started (A7).
- ~23:30: decisions push rejected twice by the same 14 pre-push tests (pass when run alone) -> likely test-order pollution
  from the new tests importing scripts; diagnosing, will fix the tests (not bypass).
- ~23:40: **PI deck fluoroscopy update DONE by a8** (192 slides / 85 hidden): slide 161 = the new 10-panel all_graphs,
  notes with density / amount + sitting / gap / transit stats; NEW hidden slide 162 "How bolus density is measured"
  (density_method). Backup backup_before_fluoro_density_1105.pptx; hidden states asserted unchanged. Wording question = A8.
- ~23:55: **slide 161 reworded by a8** (A8): "The few swallows left carry about a third of the barium (smaller and less dense
  bolus) and clear the esophagus slowly" (amount x0.34 day 1 / x0.28 day 2). Backup backup_before_fluoro_bullet_1105.pptx;
  hidden states unchanged (192 / 85).
- ~00:05: decisions push failures traced: `tests/test_behaviour_camera_period_1105.py` imports disengagement_tests, which
  sets WIDEFIELD_PRE_FROM=0801 at import; modules first loaded then keep August-only values for later tests. Fix: the test
  drops modules it imported (autouse fixture). Re-testing, then push.
- ~00:05: **C: disk**: 85 GB free (unchanged since 23:30). The render uses only `wfield_cache/input_cache_rebuild_20261104`
  (204 GB). The older default caches `wfield_cache/imaging` (64 GB) and `wfield_cache/behaviour` (39 GB) hold superseded
  hash versions from 10-28 .. 11-04; the 10-28 .. 11-02 ones (~58 GB) are safe to delete (reproducible). Waiting for OK (A7).
- ~00:10: free RAM fell to 2 GB (three te / spont steps at once, ~34 GB). Stopped the t_partial_te step by exact PIDs
  (parent + its 3 workers); it re-runs on its own (with te_readout, which ran without it) after svf_te ends. Test fixes
  for the push committed (7a7a60be; also cherry-picked to the follow-up branch); the decisions push is in its pre-push
  run (slow: CPU shared with the render). Render: 14 steps done at 01:10.
- ~02:00: **decisions branch PUSHED** (`fix/decisions-f1-f2-enl-20261105`, incl. the test fix 7a7a60be). Log-deficit and
  follow-up branch pushes held until RAM recovers (their pre-push suites are heavy).
- ~01:52: the queue's low-RAM guard wrote STOP (available RAM 0-5 GB with ~6 render steps + a Jupyter `fr` kernel that is
  not ours + attempt_coding). Lanes halt at their next step boundary; nothing killed. My attempt to stop t_partial_te
  again was blocked by the auto-mode check (it selected worker PIDs by parent, against our own rule) -- not retried.
  Will clear STOP and relaunch lanes ONE OR TWO AT A TIME once the running steps finish (the 6-lane overlap is what keeps
  driving RAM to the floor). **Question A9: OK to run at most 3 lanes at once from here (slower but no RAM stops)?**
- ~03:45 (from a8): on main now: response-window fallback (9ad4fbcc), hemo 470 slot (f1fd9f9f), and the **August-only
  cohort key (5fa8253c, cohort1 = 0806, your "yes switch to 0806" in a8's window; existing cut sites now read it)**.
  This is NOT A4: what a bare run selects is unchanged and no warning was added -- A4 stays open, not started.
  WIDEFIELD_PRE_FROM stays 0801 (same pre sessions as 0806, pinned by a test), so render folder names don't change.
  a8 holds item 5 (fail-loud DAQ contacts) until the _daqlk render is done. Pre-push now runs only the touched tests.
- ~03:45: render still under the low-RAM STOP: lanes G, E, SVF halted cleanly at step boundaries (g5_epoch, e_iti_key,
  b_svf_frA done); 4 steps still finishing. Will relaunch a few lanes at a time once they end.
- ~06:30: **render shortcut (Priya OK)**: d_attempt_pp_lickcue / pp_precue ran the same command + env as d_attempt_lp (7.7 h),
  only a different --out (attempt_coding_pp, read by the page scripts). attempt_coding reuses trials files already in its
  folder, so the 12 trials_*.gz were copied (byte-checked) into attempt_coding_pp/<tag>; the pp steps now only redo
  summary / contrasts (minutes). No other duplicate steps in the queue. STOP cleared 05:43; lanes 2, X, TE re-run, D running
  (max ~3 heavy at once). Log-deficit branch PUSHED; follow-up branch push in its pre-push run.
- ~06:30: **speed-ups for FUTURE runs (Priya OK)** on branch perf/attempt-perm-iti-lags-20261106 (wf_speedup), by a helper:
  iti_licks --key frozen lag sweep (6.2 h, serial 656 readouts) -> fan over cores; attempt_coding per-session loading
  (~35 min per animal x alignment, serial) -> fan_sessions. Exact-equality tests. Correction: the permutation tests are
  NOT the cost (< 1 min); I said so earlier and was wrong.
- ~06:40: **queue v7 = heavy-step lock (Priya OK)**: one pixel-encoder / refit / epoch-render step at a time (overnight 3-4
  overlapped at ~10 GB per worker, paging peaked at 77 GB -> the 2-4x slow-down). Remaining heavy steps reserved for v7
  lanes; RAM check now reads AVAILABLE memory. v7 lanes G, T, E, A/B-heavy launched; v6 lanes 2, X, D + the te re-run
  continue (they skip reserved steps). d_attempt_pp_* finished in ~1 min each (shortcut). Found: t_frame_te_basis had
  never re-run with the calibration (stale claim from the no-calibration run) -> re-queued under the lock.
- ~08:30: **log-deficit branch ON MAIN** (e5f53dbd, Priya OK here; incl. ecac4e23: imaging_lmm decoder / CD post-only
  panels now get the log deficit's raw axis + curve, found by a8). Decisions branches next (after the follow-up push).
  Priya: ALM / tjM1 on slide 94 = YES, coloured in like the other split maps; unique R2 for regressors (no_posinfo
  per-trial re-run) = LOW priority, after the render + deck / page update.
- ~08:30: **speed-up branch done** (perf/attempt-perm-iti-lags-20261106, unpushed): b9ebb0c5 iti_licks lag sweep fanned
  per (animal, lag); 4495e60e attempt_coding per-session prefetch + new parallel.parent_blas_kept(). Exact equality on
  real data (PS92) + 10 new tests; 588 related tests pass. FOUND: `fan_out` pins the CALLING process to 2 BLAS threads
  and never restores it, so serial code after a fan-out runs at 2 threads; in attempt_coding that moved a projection by
  ~1e-4, changed the frozen-model input hash (SPEC-CHANGED -> a new frozen model minted). Measured cost centres (not
  changed): paths.cohorts() re-parses the YAML ~9,000x per animal (~2/3 of attempt_coding's warm time);
  frozen_models._store_versions re-scans every manifest on each store write (10 s per write; likely most of e_iti_key's
  6.2 h). **Questions A10-A12**: memoise paths.cohorts()? index _store_versions (rule 10 code)? make fan_out restore
  the parent's BLAS threads everywhere (may move numbers at the 1e-4 level in other callers -> would need a check)?
- ~09:30: **ALM / tjM1 map audit (Priya: "check that there are no other maps without ALM/tjM1")**. Area-level maps on a
  page / deck, NOT or only partly split: (1) frozen_peth_trial_maps (Fig 3.1/3.2/3.7/3.8/3.10, slide 94) and
  (2) session_vs_frozen --maps (Fig 4.3/4.4, slide 139) -- both in progress (Priya OK); (3) remap_checks per-animal maps
  (Fig 4.8) -- same data as (2), added to the branch; (4) residual_roi --by-spout (Fig 18.6/18.9): split versions already
  rendered as supplements "S1" -> page swap in the replacement pass; (5) specificity_control Layer 2 panel (Fig 2.6):
  needs allsession_encoder --roi-targets (~20 min), added to the render. Off-page (share only, not split):
  frozen_area_maps, fr_musall area maps, iti_area_maps non-split main, iti_hemo_check f0 areas (deliberately excluded
  ALM/tjM1), reembedding_test reembed_areas (BUG: merges ALM back into MOs while its title says they are drawn inside --
  fixing). All other area maps on the page already show ALM / tjM1. Pixel maps need no split.
- ~10:00: **ALM / tjM1 code done** (branch feat/alm-tjm1-slides-94-139-20261106, 78ca7859 + a902aedd, unpushed): slide 94
  (frozen_peth_trial_maps --split -> *_split companions), slide 139 (session_vs_frozen --roi-targets -> NEW folder
  session_vs_frozen_roi, --maps paints the split), Fig 4.8 remap_checks per-animal (split), reembedding map (now paints
  ALM / tjM1 as its title says). Same painter as the other split maps (encoder_split_rows.split_image). Allen outputs
  bit-identical; basis-route MOs / MOp within 0.003 R2 of the area run. Render steps: one pixmusall "union" re-run
  (~2.5-3 h) + svf ROI self-fit (~2.5-3 h), both heavy. Specificity Layer 2 ROI run (a_allsession_frA_roi) already
  in the queue. **Questions A13-A15**: (13) slide 94: show the *_split figures INSTEAD of the Allen ones (rec) or beside?
  (14) no_lickspout has no column on slide 139 in the frA frame (only in the _frA_ls frame) -- as before; OK?
  (15) split-ROI remap gap mixes basis-route frozen R2 with an exact-ROI self-fit (out-of-basis variance, mostly
  cancels in epoch - pre); an exact frozen side = an ROI-target area run (~2-2.5 h) -- needed?
  NOTE: the helper's duplicate dry run briefly took available RAM to ~4 GB around 08:00 -> likely contributed to the
  08:01 STOP.
- ~10:00: follow-up branch push FAILED: its pre-push suite hung at ~58 % for ~3 h (no failure; pytest idle). Stopped my
  own pytest (exact PID), re-running the suite verbosely to find the hanging test.
- ~09:30: **queue v7b = pause, don't halt.** STOPs kept recurring (08:01, 08:38) because the two remaining v6 steps
  (t_partial_te ~21 GB, x_enl_move) still run v6 watchdogs that read FREE memory and write STOP, which halted every lane.
  v7b ignores STOP: each step waits before starting until available RAM >= 10 GB; low RAM is logged, never halts. Manual
  brake = HALT file. frozen_relabel now counts as heavy. All 158 remaining steps were reserved for v7/v7b lanes, so the
  v6 lanes drain after their current steps. All lanes (G T A/B D F/Z X E) running v7b since 09:24.
- ~09:30: decisions -> main push: the pre-push hook ran the FULL suite because a NEW remote branch name was in the
  push (hook rule), and on this loaded box it hung / took hours. Pushing HEAD:main only runs the touched tests; my
  attempt at 08:58 was itself using 7 GB at 1-3 GB available, so I stopped it (its task; an orphan pytest finished on its
  own). Retrying when RAM allows. Not pushed yet.
- ~09:45: **coverage gap in the render (my miss to raise)**: rebuild_order_B "extract -> consumer" steps 3-14 (~30
  scripts feeding goal-page lick families: lick_cued_ramp + lateral_ratio / vigor / balance_imaging_control, the
  precue_restw / restnl / revup / cue_to_lick chain, enl_angle / enlv, the balance family, decoder refits, lick_code,
  session-level maps, interhemi, lick_correction -> prelick_* -> face_*, dock subsets, simple_key_figures) were never in
  the queue -- noted in RENDER_QUEUE_DAQLK item 6 but not brought to Priya, and NOT in the weekend estimate. Symptoms
  so far: e_iti_area_maps (needs cued_wide), x_covref (dep fix), d_fig_goal_summary (needs an LP_licks combined-encoding
  table no step makes). A helper is drafting all missing steps (commands, env, deps, page figures) for queue v8.
  Stale claims from halted v7 lanes released (d_attempt_maps_plot re-running now; x_fig_enl_move after x_enl_move).
- ~10:30 (from a8, PROVISIONAL): **posinfo check 1 (regressor variance) done** on the frAsplit frame. The acute rise in
  "movement WITH lick-position info" unique R2 (contra MOs / MOp / SSp-m / SSp-n, cue window +0.022 [+0.005, +0.044],
  +90 %, 4/4) is NOT explained by more regressor variance: WITH-regressor variance is flat or down (cue +3 %, lick -10 %;
  only PS94 up); unique R2 per unit variance +107 %, 4/4. Lick window weaker (3/4); data variance in those areas +77 %.
  Open: decorrelation (needs the no_posinfo per-trial re-run, LOW priority per Priya) vs stronger coupling.
  **Conclusion change -> Priya before any text**: page / deck text calling the acute WITH gain "more variable licks"
  would be wrong. Outputs: <E>/../posinfo_checks_v7g3_t3d_spont_frAsplit/.
- ~10:30: v8 code (render/daqlk-v8-20261106 = decisions + ALM/tjM1 + speed-ups) passes 38/38 targeted tests; queue v8
  being assembled (104 missing steps + decisions + ALM + 9 fixes). Decisions push to main waits for 20 GB RAM.
- ~10:10: **queue v8 LAUNCHED** (4 lanes, code wf_render_v8 5db29b07 = decisions + ALM/tjM1 + speed-ups, preflight OK).
  151 steps: the rebuild_order_B goal-page lick families (L0-L10), the approved decisions steps, ALM/tjM1 (section M),
  fixed versions of 9 broken v7b steps, 13 re-runs moved from v7b so they run once on the new code; mixed models use
  --deficits ttc_log ttc contact. 8 dependencies v7b never declared were added. Held: fr_enl_decode + revup_behaviour
  (write outside the rebuild root / untagged -> routing fix first), lick_acad_midline_control (cut), F8 e_plain_* (Priya).
  x_enlv_extract released from the old v7b lane via a brief skip marker (no process killed) and moved to v8.
  v7b lanes G T A/B F/Z E continue the rest; one heavy step at a time across both queues (shared lock).
- ~10:30: **dependency audit of the remaining v7b steps + v8** (V7B_DEP_AUDIT.md in the render scratchpad): 5 v7b steps
  would have failed on inputs made later or by no step (e_recov_meas, e_recov_marker, e_dock_sspn0, b_fr_compare,
  z_fig_pooled), e_checkfigs would be skipped, and 3 v8 steps had missing deps that would have halted v8 lanes
  (l_recov_markers needs iti_area_maps --split / --split --cued, which no step ran; l_iti_dock_audit needs x_covref;
  l_enl_angle needs cued_wide). None had started. All 18 were reserved for **queue v8b** (22 steps: the 18 with correct
  deps + 4 missing producers: iti_area_maps --split, --split --cued, frozen_residual_trend frA joint and no_fr_dir),
  launched 10:30 in 2 lanes. e_iti_hemo (failed after 49 min on an undeclared dep) re-runs via a waiter after
  e_iti_area_maps.
- ~11:40 **Priya's answers**: (1) merge the ALM/tjM1 + speed-up branches to main: YES. (2) F8 e_plain_*: run as TRUE plain
  Allen (WFIELD_LICK_DECODER=areas). (3) slide 94: the *_split maps REPLACE the Allen ones (replacement pass).
  (4) slide 139 without a no_lickspout column: OK. (5) slide 139 split-ROI remap gap: do the EXACT frozen side (an area
  run with ROI targets, penalties held, ~2-2.5 h) instead of the basis-route approximation.
- ~11:50 **Priya's answers 6-7**: (6) slide 92 / Fig 2.11 wording ACCEPTED as recommended: "acutely, these cortical areas
  track lick-position movement more strongly (not explained by more variable movement)", marked provisional until the
  per-trial check -> sent to a8 for the PI deck; page Fig 2.11 caption in the replacement pass. (7) C: cache: DELETED 375
  superseded files (10-28..11-02, 58.3 GB; list in rebuild_20261103/DELETED_LOCAL_CACHE_20261106.txt); C: 113 GB free.
  F8 + exact slide-139 frozen side: steps held for queue v8c (helpers building); ALM + speed-up -> main push waits for RAM.
- ~12:00: **F8 + exact slide-139 side running.** v8c (11:38, lanes P / S): e_plain_* as TRUE plain Allen
  (WFIELD_AREA_BASIS=plain; outputs *_plain / _r8 names) + 2 split producers the mislabelled "plain" steps used to make
  (e_split_where, e_split_loclin_all) + 9 readers re-depped + the notdock dock arm x_enlv_balance needs. v8d (12:00, code
  wf_render_v8d d2d26e84 = v8 + 435160e2): b_roiexact_frA = exact ROI-target frozen run (area-run penalties + statistics
  held; Allen bit-identical; vs basis route max |d| 0.004 R2, unique R2 <= 0.002), then slide 139 / Fig 4.8 maps with
  --roi-frozen exact. Slide 94 stays on the basis-route union run; the deck should point at the *_split.png files.
  Main: 5db29b07 (decisions + ALM + speed-ups) push waits for RAM; 435160e2 follows.
