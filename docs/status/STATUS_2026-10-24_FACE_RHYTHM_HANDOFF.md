# FaceRhythm handoff (2026-10-24; afternoon update at the top of section 2) -- START HERE

Supersedes `STATUS_2026-10-22_FACE_RHYTHM_HANDOFF.md` for current state (that file keeps the 10-22 / 10-23 history:
pilot, timing fix, master planning, rank-10 fit). Findings: `docs/FACE_RHYTHM_FINDINGS.md` (status table at the top).
Decisions: `DECISIONS.md` 2026-10-24 (rank 12; names + grouping master_v1; close / far parts). How it runs:
`docs/FACE_RHYTHM_PIPELINE.md`.

Rules still in force: don't run heavy things locally without checking the box (shared; d4's encoder runs); commit only
your own files, staged by name; never delete data; write only under MICROSCOPE/Priya. Document as you go.

## 1. State

**Master = `master_pre_v1`, RANK 12, frozen 10-24 05:49, spec `f4f84d67773df9bc`**
(`M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/`). Fit on O2 (job 57395128_12):
4 animals x 120 trials (20 / position, even over pre days), pre-cue rows, run / rest between rows, true bin times;
EVR .803, stability .912, leave-one-animal-out factor similarity .82-.90, split half .911.

| on the share | state |
|---|---|
| `master.npz`, `rank12.*`, `labels.json` | done; names in FINDINGS 4a |
| `proj/` 108 | done (full frozen basis for every session; `basis` full_in_sample 32 / full 76) -> **ENCODER** |
| `loso/rank12_*` 31, `proj_loso/` 32 | done (pre sessions on their left-out-day basis; FR scoring only) |
| `splithalf/rank12.json` | done (.911; lick pair .96 / .99; f6 run_onset_jaw .36, f10 prep_jaw .76) |
| `own/` (3 starts) + `own8/` (8 starts, O2 rerun 10-24) | done, 108 each; readouts / matching read `own8/` automatically (`face_rhythm_readouts.own_sub`) |
| `report/` | rank-12 gallery (abs + share, with names), rank sweep, stability, loso tables |
| rank-10 outputs | record kept: `master_r10.npz`, `rank10.*`, `own_r10/`, `report_r10/`, validation `master_pre_v1_r10` |

Local clips (one pre session per animal, 12 factors) `face_rhythm/cam4/factor_clips/master_pre_v1/`; validation
`face_rhythm/cam4/validation/master_pre_v1/` (+ `lick_epochs/`).

**Encoder (d4)**: has proj/ + labels + grouping `fr_groups.master_v1` (lick 2, 4, 5, 7, 8, 11; left 5; right 4; close 7;
far 8; other 0, 1, 3, 6, 9, 10; modes A / B / S / D; side = S - B, distance = D - B). Bundling into
`enc_v36_full_20261024` (107 sessions; raw 12 factors, grouping applied at submit).

## 1b. Results and figure index (10-24 afternoon; FINDINGS 5b-master, 5d, 5e, 6-master, 6b-6d)

All under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/` unless noted.

| question (Priya) | result | figure(s) / script |
|---|---|---|
| contralesional deficit per lick? | lick_right per right lick .57-.79 of pre chronically, 4 / 4 | `validation/master_pre_v1/lick_epochs/` (`face_rhythm_lick_epochs.py`) |
| peak factor amplitude over recovery (manuscript FR panel) | contra-spout lick_right chronic -.30 [-.40, -.21] 4/4; contra-spout lick_left +1.08 [.35, 1.87] 4/4 (ratio scale); per position: far_R loses most contralesional, close_R gains most ipsilesional | `peak/peak_by_day*.png`, `peak_cohort_by_{day,epoch}*.png`, `peak_side_headline.png`, `peak_position_headline.png` -- LOG2 scale since 10-24 pm (`face_rhythm_peak.py --split none/side/position`) |
| KEY: contralesional licks made with an ipsilesional face movement | factors (above) + raw point movement, lick-aligned, far_R asym pre +.22-.27 -> chronic below pre 4/4 | `asym_aligned/asym_aligned_traces.png` (cue / lick / bout-onset columns), `asym_aligned_by_animal.png` (`face_rhythm_asym_aligned.py`) |
| does the pre vocabulary fit post? (EVR, normalized) | frozen / own ratio vs own pre: acute -.035 [-.045, -.025] 4/4, chronic -.008 (3/4, borderline) | `evr_own_vs_frozen.png` (bottom row pre = 1, +-1 SEM), `evr_epochs.png` (`face_rhythm_evr_epochs.py`) |
| why does own EVR drop too? | less movement vs fixed tracking noise + less stereotyped movement (partial: r with between-trial speed .48-.66 in PS94 / PS95 only); the ratio divides it out | -- |
| factor angles / morphing | NOT consistent across animals (Priya: "no clear consistent changes") -> secondary; lead with amplitudes | `matching/match_by_day.png`, `match_example.png` (`face_rhythm_match.py`) |
| asymmetry two ways (points vs own-fit loadings) | agree acutely; loadings: lick_right moves off the contralesional face | `matching/asymmetry_two_ways.png` |
| per-position face speed maps | whole face moves less acutely at every position; contralesional half more, strongest far_R; chronic overshoot, ipsi half up on R spouts | `position_maps/position_maps_response_{mean,PS9x}.png` (`face_rhythm_position_maps.py`; cue 0..+4 s) |
| lick- / bout-aligned maps + patch statistics | DONE: whole face down per TRIAL acutely, not per LICK; per lick the contralesional half loses more at close_R / far_R, 4/4, every epoch (FINDINGS 5f) | `aligned_maps/aligned_maps_{cue,lick,bout}.png`, `halves_*.png` (`face_rhythm_aligned_maps.py`, needs the cache from `face_rhythm_asym_aligned.py`) |
| close_R asymmetry before bout onset? | carry-over from earlier licks + post-cue preparation + small sustained offset; mirror at close_L (target-directed), FINDINGS 5f | `asym_aligned/bout_split/` (`face_rhythm_bout_split.py`, `--plot-only`) |
| non-task drop = less active? | no: drops within rest (-.99 log2, 4/4) and "other" (-.37, 4/4); running time not consistently down (FINDINGS 6d) | `nontask_state/` (`face_rhythm_nontask_state.py`, `--plot-only`) |
| WHEN is each factor expressed vs the cue? | lick_right on contra-spout trials LATER subacute / chronic (+.40 / +.41 s, 4/4) while the first lick is EARLIER (-.06 s, 4/4): re-timed within the response, not reaction time (FINDINGS 5g) | `peak/timing_by_epoch_{com,t_half}[_side,_position].png`, data/timing_epoch_stats*.csv (`face_rhythm_peak.py`) |
| factor shape, cleaner test | SUBMITTED to O2 (Priya): `fitepoch`, one pooled own fit per animal x epoch + two pre-day halves (noise floor), 11,000 rows each | -> `own_epoch/` on the share; analysis script still to write (angle post vs preA / preB against preA vs preB) |
| pre + post master for pushes? | no: frozen error LOWER at pushes, 27 / 32 cells | `push_check/` (`face_rhythm_push_check.py`); DECISIONS 10-24 |
| task vs non-task, severity | non-task NOT preserved acutely; no ordering by lesion size (n = 4) | `task_vs_nontask.png`, `severity.png` |

## 2. Next, in order

00-A8-1118. **11-18 (a8) -- START HERE; 00-A8-1117c below holds the record up to 11-17 ~19:45.**
   **NEWEST (11-18 late): quiet events are on the SYNC-END run, deck swapped.** Run
   cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118 (QS_ENGAGED_END=lastsync = Priya's "the end of the analyzed
   time is the LAST SYNC PULSE"; common k 4; code 871be8f9; runner quiet_c1guard/run_chain_syncend.sh; restyled figures
   in its restyled_20261118/). One old-vs-new list: docs/status/quiet_restyle_20261118/QUIET_SYNCEND_OLD_NEW_20261118.md.
   - Priya 11-18: "yes you can swap the figures and numbers". DONE: deck slides 155-159 + 204 (pictures, movie, notes;
     158 title "small at 415 nm", 159 title "weaker code pre, acute and chronic"); script
     quiet_c1guard/deck_swap_syncend.py, images quiet_c1guard/syncend_crops.py -> swap_inputs/*_syncend.png; backup
     .bak_pre_a8_quiet_syncend_20261118; export QC from a scratch copy.
   - Page section 23 (Movement page): 3c's, messaged with the run + list; a8 does not touch it.
   - 11-18 night: Movement page v324 = section 23 on the sync-end run (3c, Priya OK). Slide 158 retitled "415 nm power,
     rate flat" (Priya). Quit-time cache on main (quiet_area_spikes.quit_time_s; reproduction check exact except the
     injection seed, fixed to crc32 in quiet_spikes_v2); chain 71 vs 75 min. Deleted (Priya OK, list in
     quiet_restyle_20261118/DELETED_SUPERSEDED_20261118.md): superseded quiet runs, 12 deck backups, early rebuild
     render; C: input-cache behaviour entries written 11-04..11-08 (250 GB) + temp > 7 d. Section-23 follow-ons
     re-rendered on the sync-end run (run_all_followons_syncend.sh; paths sent to 3c); 23.23 follow-up re-running
     with WIDEFIELD_LICK_SOURCE=daq. NEXT: 3c builds the old-vs-new list from its captions -> Priya -> page swap ->
     delete pre-rebuild labcams/cohort1_quiet_spikes/v2_quiet + v2_quiet_lp (Priya ok, once uncited). Open: local
     session_cache_rebuild_20261103 (78 GB, 8396/8439 files duplicated on M:) awaits Priya.
   - 11-18 late night: Movement page v325 (3c) = 11 section-23 figures on the sync-end run with BOOTSTRAP stars
     (quiet_boot_stars.py, fae48012; * = boot change vs own pre, dagger = LMM). Quit period now ends at the last sync
     pulse (Priya; analyzed_window, 9701d914): pre quit min PS92 2.7->0, PS93 16.7->4.2, PS94 38.4->35.3, PS95
     69.5->56.7; T1/T3 unchanged; T4 3 animals; still-UP rates pre quit/eng 5.6-7.4x (table
     quit_state/still_up_rates_eng_vs_quit.csv), sent to 3c for 23.7/23.8. 23.23 follow-up: stored rebuild decoder
     loaded by id (QS_FOLLOWUP_STORED_DECODER=1, Priya ok) but scoring refuses -- the script centres on all engaged
     rows, the stored model on camera WORKING rows. ASKED Priya: (a) test with centre_rows=engaged + rebuilt-vs-stored
     p_true check, (b) implement working centring, (c) keep 23.23 pre-rebuild. Local session cache delete was blocked by
     the permission classifier (Priya can delete C:/Users/SabatiniLab/wfield_cache/session_cache_rebuild_20261103).
     Pre-rebuild v2_quiet + v2_quiet_lp DELETED (Priya ok) except v2_quiet_lp/task_events + task_events_followup (23.23).
     Movement page v326: 23.7/23.8 on the sync-end run (3c). 23.23 OPEN: follow-up now loads the stored rebuild decoders
     by id (PS93 = stored 7-session model without PS93_0813) with the production camera-working centre (c8c002c2), but
     its p_true check FAILS vs the stored T13 p_true (median |diff| 0.14; levels ~.45 vs ~.32) although features match
     (feat_err 0). Suspect the scoring transform (stored T13 run said to be session_z). Log
     <syncend>/logs/quiet_task_events_followup_working_centre.log. Nothing from it on the page.
     Then (Priya 11-18 "1 ok"): production scored frame (session z + kind_centre_fn wt centre, dead7b25) -> check
     STILL fails (median |diff| 0.133; corr rebuilt vs stored p_true 0.65, rebuilt ~0.10 higher; rows aligned). Not the
     centre. Open suspects: the stored T13 tables were written under the PRE-guard camera classes (csig 3d7bbe921e) and
     an older env; next step = reproduce one session's stored p_true with attempt_coding.run_animal under that key.
     Within-follow-up (same rebuilt decoder) the event effect survives exclusion (log _prodframe). 23.23 stays pre-rebuild.
     DECIDED (Priya 11-18): 23.23 STAYS on the pre-rebuild data; keep labcams/cohort1_quiet_spikes/v2_quiet_lp/
     task_events + task_events_followup permanently. The rebuilt follow-up outputs in <syncend>/task_events_followup are
     an unpublished diagnostic (code switches QS_FOLLOWUP_STORED_DECODER, default off).
   - FR page v56: Fig 5.6 = a2's cue-aligned factor figure + cue-window stats (Priya "yes" / "add stats").
   - OPEN: (1) caching refactor of the quiet scripts (Priya "yes", after the run: cache session traces + quiet /
     engaged masks, apply the engaged gate at summary time, verify exact reproduction of the syncend tables + timing);
     (2) delete OK pending for superseded quiet folders (k375 follow-ons, k4_20261118 incl. restyled, engend_20261118,
     v2_quiet_lp_20261117 follow-ons, the c1guard_20261117 k4.5 summary); (3) slide 158 caveat for Priya: the 415
     response at events rises acutely +0.22 % dF/F (corrected 470 +0.31), mostly PS94.
   **Quiet events are now on the cam1-guard production run.** Priya, late 11-17: "k 3.75 with cam1 fix", then "k 4 iw
   fine" (the follow-on figures keep their fixed k = 4), then "re run both" (154 / 157). Run folder:
   cohort1_quiet_spikes/v2_quiet_lp_c1guard_k375_20261117 (guard code wf_c1guard1117 f3484347; runner
   quiet_c1guard/run_downstream_k375.sh).
   - **Deck (Priya 11-18 "ok"):** slides 154-158 and 203 swapped (pictures, movie, notes); the wrong "< 2 % ... same
     k 3.75" note was removed everywhere, including slide 112. Backup .bak_pre_a8_c1guard_20261118.
   - **Record:** quiet_c1guard/QUIET_C1GUARD_OLD_NEW_20261117.md, last section: old vs new per slide, the
     production-key p_true wrapper for slide 158, superseded folders.
   - **OPEN:**
     - Goal page section 23: Priya OK'd the swap to the guard-run quiet data (11-18, via 3c); 3c is doing the page
       side from QUIET_C1GUARD_OLD_NEW_20261117.md. a8 does not touch it. Priya accepted the rebuild + guard figures
       there; a8 rendered the k 4 threshold figure for 23.18 (v2_quiet_lp_c1guard_k4_20261118, 95b66db6, path sent to 3c).
       11-18 later: Priya "use k = 4 for all" + restyle OK (via 3c). DONE, waiting on Priya before any page / deck swap:
       restyled figures in v2_quiet_lp_c1guard_k4_20261118/restyled_20261118/ (scripts/cohort2_qc/quiet_restyle_figs.py,
       branch figures/quiet-restyle-1118, worktree wf_qstyle1118, on main 63b74576); star list
       docs/status/quiet_restyle_20261118/QUIET_RESTYLE_K4_STAR_CHANGES_20261118.md (k alone: 1 LMM star; LMM -> boot: 53 of
       259 cells). Next if adopted: re-cut deck crops 155-159 / 204 from the restyled renders (new layout).
     - Slide 27 panel C: Priya "yes add sessions to c" (11-18, typed here) -> 3c re-rendering all three 15f images; swap
       the slide-27 image when 3c sends the path.
     - Page rename live (Priya OK): imaging page = "Spout (Target) Coding After Stroke", goal page = "Movement (Execution)
       Coding After Stroke". Deck notes citing the old names: list from 3c in wf_pagerename
       docs/status/page_rename_20261118/deck_mentions_Overview_2026-11-06.txt (Priya OK'd updating them). NOT done yet.
     - The attempt_coding csig re-key under the guard: no guard-keyed tables yet.
     - Delete the superseded v2_quiet_lp_20261117 follow-ons and the v2_quiet_lp_c1guard_20261117 k 4.5 summary only on
       Priya's OK.
     - **DONE 11-18: slide 27 added (shown)** = rest frozen decoder, floor 30 + central 90 % (s27_rest15f.py; backup
       .bak_pre_a8_s27rest15f_20261118; deck now 208 slides, every later slide +1: quiet slides = 155-159 and 204).
       Title shortened to fit: "Lick-free rest decoder: drops acutely, stays below pre". History below.
     - (was) **NEW SLIDE QUEUED as slide 27, after the position maps 24-26 (Priya OK 11-18, via 3c):** rest frozen decoder
       (15f), title "Frozen pre-stroke decoder on lick-free rest: drops acutely, borderline below pre chronically".
       - **11-18 change (Priya, via 3c): the PRIMARY is now the FLOOR-30 version** (post-stroke scored sessions need
         >= 30 matched lick-free rest periods; a 40 floor changes no outcome). Floor 30: acute -0.122 [-0.194, -0.040]*,
         subacute -0.030 n.s., chronic -0.083 [-0.154, -0.007]*; PS93 acute now 4 sessions.
       - Picture: the title-fixed floor-30 render in a NEW folder rest15f_restnlc1all_floor30_20261118/... (NOT the
         20261117 floor-40 file). Hold placement until 3c sends the exact path.
       - Notes follow imaging page Fig 12.1 (floor-30 caption). The old single-session wording (PS93 acute / PS92
         subacute on ONE session) was for floor 40; re-check per-cell session counts from 3c's caption, don't copy it.
       - **11-18 later (Priya, via 3c): PRIMARY = floor 30 + CENTRAL-90% pre matching** (supersedes the line above).
         All-pre calls: acute -0.139 [-0.200, -0.070]*, subacute -0.069 [-0.128, -0.008]* (newly significant),
         chronic -0.098 [-0.170, -0.024]*. Title becomes e.g. "Frozen pre-stroke decoder on lick-free rest: drops
         acutely and stays below pre" (3c's wording; confirm with Priya when placing). Notes add: central-80% window ->
         subacute n.s., acute / chronic hold; a 40-period floor changes nothing. Renders in a new folder; hold for 3c's path.

00-A8-1117c. **11-17 LATE (a8) -- START HERE (state at ~19:30: all published; only the quiet-event re-run is queued, waiting on 3c's commit); 00-A8-1117b below is the evening state (still valid for everything else).**
   Like-for-like curves rebuilt as ONE figure (Priya 11-17): columns cued all / cued first / ITI all / ITI bout-start first
   lick / ITI docked; rows all post licks vs STRICT (post licks only where the animal's own pre curve of the same lick set
   exists). Gaussian-smoothed curves, local-linear reference (DECISIONS 2026-11-17 'Like-for-like curves'); FINDINGS 4d0k.
   **Published:** goal page v317 (3c owns the page; a8 edited section 12 + the brief card only, 3c told): Fig 12.2 =
   lick_matched_panels.png (#f-lick_matched_panels; old id kept), non-linear curves -> Fig 12.2 S2, 3 old curve PNGs
   removed (page at 509 / 511 files). **Deck:** slide 125 replaced + retitled 'Decoded vs landed: ITI shifted every
   epoch, cued acutely' (backup .bak_pre_a8_panels_20261117; specs/s125_panels.json). **Outputs:** iti_licks/
   panels_loclin_first_20261117 (all-ITI, figure + figdata) and panels_loclin_dock_20261117; semband_{allITI,dock,gap2dock}
   _20261117 (binned loclin curves with SEM band, superseded for display). **Code (default-off):** lick_matched_readout
   curve_panel / --sem-band / --inside-curves; scripts/lick_matched_panels.py. Record: DOCKED_DECODED_VS_EXECUTED_20261117.md
   (combined panel section). **Open (offered, not asked):** within-spout like-for-like version (each spout's own pre licks;
   ~20 min per set); per-session lick counts kept under the strict rule; Fig 12.1 read-out still on the Gaussian-mean
   reference (loclin agrees within 0.03 mm) -- switch only if Priya wants it. Nothing of a8's is running.
   **Later 11-17:** figure title 'Decoded vs landed lateral position' (not LIKE-FOR-LIKE; page v318). NON-LINEAR version:
   page Fig 12.2 S2 (v319, replaces the old non-linear curves; 2 old PNGs removed) + NEW HIDDEN deck slide 126 'Same with a
   non-linear decoder' (s125b_nonlinear.py; backup .bak_pre_a8_s126nl_20261117). DECK NOW 207 SLIDES: every slide after
   125 moved +1 (quiet-event 156/157 -> 157/158; rest maps 187/188/204-206 -> 188/189/205-207). **Queued (3c, Priya OK):**
   quiet-event re-run with the cam1 far-from-mouth guard (cam1-only events count as licks only if non-droplet AND
   med_out_px >= -120; audit docs/status/CAM1_FARMOUTH_BLAST_RADIUS_20261117.md on branch analysis/restnl-cam1-1117).
   Wait for 3c's commit on fix/cam1-farmouth-guard-1117; run from that worktree; production env = cohort1_quiet_spikes/
   v2_quiet_lp_20261117/logs/quiet_v2.log (QS_LICK_SOURCE=daq+lp, rebuild env); new QS_V2_DIRNAME; then per-animal old vs
   new chronic + chronic-vs-pre numbers to Priya BEFORE page (section 23) / deck (157 / 158; 3c named 112 too) changes.
   **DONE ~19:45:** re-run + fixed-k summary; docs/status/deck_swap_20261117/quiet_c1guard/QUIET_C1GUARD_OLD_NEW_20261117.md.
   Guard changes nothing material at the production k 3.75 (< 2 %); the full re-run's LOAO k flipped 3.75 -> 4.5 on a
   flat AUC curve. PRIYA: keep k 3.75. Notes extended on deck slides 112 / 154-158 / 203 (backup
   .bak_pre_a8_quietnote_20261117) + goal page v320 section 23 lick sentences. OPEN (flagged): page section 23 still
   shows the PRE-REBUILD quiet outputs (v2_quiet_lp, k 4); the deck has the rebuilt run -- a swap pass if Priya wants.

00-A8-1117b. **11-17 EVENING (a8) -- START HERE; item 00-A8-1117 below is the day's earlier state (its "Running" list is DONE).**
   Day log: docs/status/RUNNING_LIST_2026-11-08_a8_june.md (11-17 entries). Nothing of a8's is running.
   **Deck (Overview_2026-11-06.pptx) done today, on top of 00-A8-1117:** groups C (LMM incl. docked + per-session slide 12 +
   111 / 116; other incl. 22 record, 157 rewrite, 139 full set), docked-ITI slides, low-hit (<20) cells shown (120, 121, 141-146),
   ITI response window 0..0.5 s (DECISIONS 2026-11-17; ITI_WINDOW_AUDIT_20261117.md), cued per-spout window 0..1.0 s (113 / 114),
   slide 139 retitled "Cued ipsi-contra balance places an unseen animal on recovery" (4-column figure, cued left / ITI right),
   slide 21 both scales + decomposition result in notes, slide 20 / 147 layout fixes (overlap_check.py), 33 lick-source wording
   fixes (3c audit). Backups kept: .bak_pre_a8_swap_20261117 (before today), .bak_pre_a8_licksrc_20261117; 17 others deleted
   (Priya OK; DELETED_DECK_BACKUPS_20261117.txt). Applier + specs + logs: docs/status/deck_swap_20261117/ (README = spec schema).
   **Results documented today:** SLIDE21_DECOMPOSITION.md (trial gate drives the slide 21 change: old gate on the rebuild -.397 /
   -.191 / -.112; ~-.02 from old fitted-vs-projected joint time courses; pre set / scaling / lick detector none);
   ITI_NEG_AUC_DIAGNOSTIC.md; ITI_MATCHED_ANGLE_MARKERS.md (sections 1-5b: cued -> cued angle .68; all-ITI balance holds vs
   matched / docked angles, DOCKED balance loses the recovery link; far-contra within-spout raw balance; bounded indices dec_idx /
   resp_idx: dec's negative ITI shifts vanish, cued far-contra shift survives; gap >= 2 s check); DOCKED_DECODED_VS_EXECUTED_20261117.md
   (docked ITI offset +0.30 / +0.21 / +0.27 mm holds; raw offset-aim link gone on docked, survives detrended). FINDINGS recovery-
   marker block has the 11-17 entries; FINDINGS / ENCODING_FINDINGS ITI lines updated to 0..0.5 s.
   **FR page:** v55 published (12 lick-source captions: FR lick windows = LP video cam4 only, licks_all.csv).
   **Code added (all flags default-off, outputs byte-identical without them):** iti_licks CUED_RESP_WIN / ITI_RESP_WIN;
   lick_balance_angle_lmm --idx / --gap2 + bounded_index (tests/test_bounded_index.py); lick_iti_matched_markers.py;
   lick_matched_readout --first-curves; lick_matched_angle_reach --curves; likeforlike smooth_curves / curve_coverage;
   trial_set_text.footer_bottom; frozen_claims --trend-only / --plot-only; decoder_retention_sessions / recovery_timing /
   key_figure --show-low-hits; learned_compensation refactor (t3_cells); lmm-hidden-20261103 branch (wf_lmm_main) pushed.
   **Grant figure check RESOLVED:** K08 code is now in the repo (grants/k08_2026/, render_fig4_rows.py row_DE); panel D stars
   = boot_star of lmm_hidden_20261103/cued_ramp_summary_row2_lmm_w0-1_verify.csv; it does NOT star acute ipsi tjM1 (+.36 n.s.),
   same as the rebuild (+.37 n.s.); contra tjM1 up every epoch (+.51* / +.40* / +.41*). No grant change needed.
   **Open:** (2) 3c: goal-page swaps of the ITI 0..0.5 s figures, Fig
   14.7, the idx supplement (PAGE_SUPPLEMENT_IDX_20261117.md) -- 3c owns the page. (3) After the RESTnl-cam1 rerun (3c helper):
   rest-map slides 24-26, 187, 188, 204-206 wording + figures, after Priya sees old vs new numbers. (4) Optional: per-session /
   docked-only figures not rendered: docked _far1 / _gap2far1 / _nonlinear / _plain curve variants; F_lnmf decomposition arm.
   Rules learned: export slides only from a scratch COPY; helpers must set PYTHONPATH=<worktree> (main checkout was stale);
   brief helpers to reuse repo constants / loaders (memory efficient-no-redo).

00-A8-1117. **11-17 (a8) -- START HERE (supersedes 00-A8-1116b as the entry point).** Day log:
   docs/status/RUNNING_LIST_2026-11-08_a8_june.md (11-17 entries). The day was the PI-deck rebuild swap pass, not FR.
   **Done (deck Overview_2026-11-06.pptx; every step backed up as .bak_pre_a8_<step>_20261117.pptx; specs + applier +
   apply logs in docs/status/deck_swap_20261117/, README there = spec schema):** A+B stale slides (63), group C renders
   (LMM companions incl. per-session slide 12, docked-ITI slides, 111 / 116, quiet events, 139 full set), low-hit (<20 hits)
   cells shown not tested (120, 121, 141-146), Priya decisions (22 pre-rebuild record, 157 rewrite, 21 both scales),
   ITI response window 0..0.5 s (DECISIONS 2026-11-17; audit ITI_WINDOW_AUDIT_20261117.md; slides 105, 108, 110, 111, 135,
   136-138, 203 applied; FINDINGS / ENCODING_FINDINGS lines updated). ~120 slides updated in total.
   **Results documented:** ITI_NEG_AUC_DIAGNOSTIC.md (ITI biphasic dip, pre-heavy -> 0..0.5 s), ITI_MATCHED_ANGLE_MARKERS.md
   (cued -> cued angle .68; all-ITI balance holds vs spout-matched / docked angles; DOCKED balance loses the recovery link;
   far-contra within-spout raw balance: cued same-angle shift +.24 log2), FINDINGS recovery-marker block (two 11-17 entries).
   **Running (background helpers; check their outputs before relaunching):**
   (1) slide 21 one-factor decomposition -> decomp_slide21_20261117/ + slide21_results.jsonl; done A_pre / D_scale / B_gate /
       C_lick, E_hemo / E2 / F running. INTERIM: the TRIAL GATE drives the change (B_gate = old gate: acute -.397 vs old -.405;
       post accuracy back to old levels); pre set (A_pre -.344), scaling (D_scale -.334), lick detector (C_lick -.348) ~no
       effect. Likely mechanism: the camera gate keeps contact + incomplete trials and drops no-attempt misses that the old
       gate kept. Report -> SLIDE21_DECOMPOSITION.md, then tell Priya (her 11-17 expectation was the opposite).
   (2) bounded indices dec_idx (0..0.2 s) + resp_idx (resp window) in lick_balance_angle_lmm + far-contra models ->
       balangle_{all,dock}_decidx_20261117/, section 5 of ITI_MATCHED_ANGLE_MARKERS.md (reuse-repo-code brief given).
   (3) docked rebuild of decoded-vs-executed figures (matched curves incl. gap2dock + cued first lick, angle versions,
       recovery link 16.3-16.5, where scatter / delta, per-lick error) + curves footer-overlap fix + dashed-coverage note ->
       *_dockfig_20261117/, DOCKED_DECODED_VS_EXECUTED_20261117.md.
   **Waiting on Priya:** slide 139 title (layout decided: caveat kept + cued-angle bullet, figure full width; spec
   specs/s139_cuedang.json NOT applied); slides 113 / 114 cued per-spout window 0..0.5 s vs cued 0..1 s (flagged).
   **From other windows:** 3c swaps goal-page ITI figures (list sent) and sends RESTnl-cam1 rerun + lick-source caption fix
   lists for the deck. Deck rule learned: export slides only from a scratch COPY (COM attaches to an open PowerPoint).

00-A8-1116b. **11-16 LATE (a8) -- START HERE; the to-do list below (item 00-A8-1116) is DONE except where noted.**
   Page = **v54**. Priya: deck slide yes; delete superseded n12 = yes (done, 118 files, list in the running list); block-
   permutation hold = ok.
   * DONE: (1) epoch_shape_angles redrawn without shading from the v1 tables (`--plot-only --match-rule v1`, both
     own_epoch and own_epoch_cohort; numbers unchanged) -> page 9.11 / 10.18. (2) Per-trial timing = page Fig 5.5
     (`per_trial_timing.png`); section 5 take line + status row rewritten: acute later lick_right follows the later first
     right lick; chronic close contra peaks .50 s later relative to it (*, PROVISIONAL). (3) Amplitude three ways = page
     Fig 2.6 (`per_trial_amplitude.png`, collapsed under 2.5) with (5) the isolated-lick numbers in its take line.
     (4) Fig 7.13 caption: one sentence on the own-fit vs frozen far-contra lick_left gap. (6d) Spout-in-mesh control =
     page Fig 6.2 (collapsed under 6.1); section 6 note's "not yet checked" sentence replaced.
   * DECK: new **slide 81** "Per trial: contra lick movements are smaller, not just rarer" (deck crop
     per_trial/per_trial_deck.png; notes = methods + FIGURE SOURCE), after per-lick slide 79 + hidden LMM 80. Backup
     `Overview_2026-11-06.bak_pre_a8_pertrial_slide_20261116.pptx`. **EVERY SLIDE AFTER 80 MOVED +1** (deck now 206):
     FR slides 71-96; old 87 / 89 / 90 / 93 / 94 -> 88 / 90 / 91 / 94 / 95; old 163 -> 164; quiet events 151-156 ->
     152-157. Other windows' docs citing slide numbers > 80 are now one low.
   * RESOLVED 11-16 (Priya): June vs August-frozen references = drift sub-analyses only (DECISIONS 2026-11-16).
     "il.OUT diagnostic" DROPPED from the FR list: it was the 11-08 render-queue audit (docs/status/QUEUE_AUDIT_2026-11-08.md,
     ~40 of 191 queued steps reading wrong `il.OUT` / imaging_encoding paths) -- belongs to the render-queue owners, not FR.
   * DECK "Data:" PASS STARTED 11-16 (Priya "yes go ahead"): 7-line block (Window / Licks / Trial period / Trial types /
     Gate / Baseline / Correction) inserted at the top of the notes, old text kept below, on 21 slides: 8 from CONFIRMED
     provenance rows (4, 5, 36, 99, 102, 147, 200, 201; matched by picture name, the table's slide numbers are 11-02's)
     and 13 FR slides (72-82, 87, 89, 94) from the FR page captions; deck now 32 / 206 with the block. Record:
     docs/status/DECK_DATA_BLOCK_PASS_2026-11-16.csv; backup Overview_2026-11-06.bak_pre_a8_datablock_20261116.pptx;
     script scripts/archive/deck_data_block_20261116.py. Slide 77 notes "peak expression per
     trial" -> "peak of the trial-averaged expression". CONFLICT left for its owner: slide 4 confirmed row = camera
     working-period trials (DAQ engaged gate = robustness) vs its notes "engaged trials only (spout_behavior
     flag_engagement)". REMAINING ~150 picture slides: INFERRED / UNKNOWN rows only -> wait for confirmation.
   * (history) deck-wide notes "Data:" pass (Priya's 11-01 request: a fixed 7-line provenance block at the top of every
     deck note, Window / Licks / Trial period / Trial types / Gate / Baseline / Correction, filled ONLY from CONFIRMED rows
     of docs/status/rebuild_20261103/FIGURE_PROVENANCE_20261106.csv -- still 45 CONFIRMED / 635 inferred / 152 unknown of
     832 as of 11-16, last updated 11-06 -> blocked on that table; Priya unsure it is still wanted). Block-permutation
     hold (a2).

00-A8-1116. **11-16 NIGHT (a8).** Page https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1 = **v52**
   (source: a8 session scratchpad fr_page/; from a new session READ the URL first, then publish with `url`). Log of the
   day: `docs/status/RUNNING_LIST_2026-11-08_a8_june.md` (11-16 entries).
   * **RUNNING: nothing.** All a8 jobs finished; no O2 jobs of ours pending.
   * **DONE 11-16:**
     - lick_left match check: runner-up vs matched (`face_rhythm_lick_left_match_check.py`, flagged_lick_left_<animal>.png,
       data/flagged_lick_left.csv); lick_left contralesional asymmetry shift STANDS acute (4/4) + subacute (3/4), chronic
       small / PROVISIONAL (Priya approved wording; FINDINGS + page + deck slide 90 / 163 notes).
     - Trial-aligned own-fit expression by spout (`face_rhythm_own_temporal.py`, ALL engaged trials; cache
       ~/wfield_cache/fr_own_temporal/*_nall.npz, 108 sessions; reads trial files with plain reads, NOT memmap -- the
       memmap held 8-10 GB per worker on the shared box). Page Figs 7.12-7.27.
     - Per-trial / per-lick readouts on the FROZEN master (`face_rhythm_per_trial.py`; outputs readouts/master_pre_v1/
       per_trial/; `--plot-only` reuses the pkl tables, `--fig-only` redraws from the CSVs). Page section 2, Fig 2.5.
       Result: lick_right at far contra smaller on every right-lick trial (-.83 / -.58 / -.61 log2, 4/4) = lasting
       deficit per movement; acute later expression follows later first licks; lick_left rises at contra spouts through
       MORE leftward-lick trials (not larger) and on contra trials with no leftward lick (contra targets licked with the
       lick_left pattern). Frozen lick_left at far contra rises acutely (+.73, 4/4) where own-fit Fig 7.13 is flat.
     - Page renumbered in order (7a->7, 7b->8, 7->9, 8->10, 9->11; v51); doc pointers + deck notes (slides 87 / 89 / 90 /
       93 / 94) updated; deck backups `.bak_pre_a8_*_20261116.pptx` in presentations/.
     - Fig 7.3 (now 9.3): epoch shading removed, marks centred (Priya); same shading removed IN CODE from
       `epoch_shape_angles` (page 9.11 / 10.18) -- NOT yet re-rendered.
   * **PENDING / ASK PRIYA:**
     - Deck slide for the per-trial result (page only so far)? Priya said "update deck" = the notes renumber; a new slide
       was not requested yet.
     - Superseded derived outputs on M: (Priya allows deletion of superseded derived; list, then ask): the 12-trial
       preview files `lick_left_match_check/own_temporal_lick*_n12.png` + `data/own_temporal_lick*_n12.csv`, and the
       `_n12` cache files in ~/wfield_cache/fr_own_temporal/ (local, ours).
     - HOLD new p values on block-permutation class-A scripts (fr_enl_decode, enl_fr_removed_decode,
       enl_movement_multisession, enl_video_spout_check, movement_position_angle) until window a2 lands
       `analysis_kit.block_permute` + its trial-wise vs block-wise comparison.
   * **TO DO, in order:**
     1. Re-render `epoch_shape_angles` (page 9.11 cohort, 10.18 per-animal) with the shading fix; swap on the page.
     2. Per-trial TIMING on the page: Figs 5.3 / 5.4 carry "caveat: trial average"; the conditioned per-trial timing
        (peak time and first lick of the same direction on the same trials) is in per_trial_stats.csv -- a figure + a
        caption rewrite of the 5.3 / 5.4 claim ("re-timed later, not delayed at its start"): acute = follows the later
        first lick; chronic small (+.24 s* peak, first lick on time). Ask Priya before rewording.
     3. Per-trial version of Figs 1.1 / 2.1 (median per-trial peak + conditioned peak, per side / spout): numbers in
        per_trial_stats.csv (`median_trial_peak`, `lick_trial_peak`); not drawn.
     4. Section 7 own-fit vs frozen discrepancy (far-contra lick_left acute): one sentence is in the Fig 2.5 take line;
        decide with Priya whether section 7 needs its own note / sensitivity (e.g. own-fit matched + runner-up summed).
     5. Isolated vs in-bout licks (isolated leftward licks ~30 % less lick_left acutely; 15 % of far-contra leftward
        licks isolated acutely vs 2 % pre): CSV only; mention if Priya wants it.
     6. Older open items (unchanged): which non-FR frozen analyses extend to June; the il.OUT diagnostic; deck-wide notes
        "Data:" pass; spout-in-mesh control figure on the page (item 00 below).

00. **11-05 (window e0, at a8's request "run now whatever can be run").**
   * DONE: FR SPOUT-IN-MESH CONTROL (`scripts/fr_spout_mesh_control.py`, ~3 h at 3 jobs; outputs readouts/master_pre_v1/
     spout_mesh_control/; reduced projections cached in C:/Users/SabatiniLab/wfield_cache/fr_spout_mesh_control/). The
     ENL face decode does NOT come from the spout (FACE_RHYTHM_FINDINGS 7, FINDINGS 13). Not yet on the FR page (one
     figure + one sentence in section 6; the page is published from the HMS account).
   * READY, PRIYA SUBMITS: >= 500 permutations -> `run_fitcohortnull500.sbatch` (share copy of the set dir; new conda
     module, --n-perm 500, array skips the 600 done). Matters for the specificity claim (several of the other ten sit at
     the p .0099 floor), not the lick pair. Then rsync own_epoch_cohort_null/ back + `face_rhythm_cohort_null.py`.
     11-07 (a8) SUBMITTED + RUNNING; INTERIM SCORE on the first copy-back (acute 482 pairs; subacute / chronic still the
     original 100): readouts/master_pre_v1/epoch_shape_cohort/interim_500perm_20261107/ (page outputs one level up,
     n = 100, untouched; scored by importing compute(), no code change). Lick pair unchanged (lick_right angle_spatial /
     angle_spectral / asym / side_spec p .002-.004, Holm < .01; lick_left n.s.). SPECIFICITY WEAKENS ACUTELY: off the
     floor, of the other ten, Holm < .05 for angle_spatial sniff_prelick, nose_prelick_slow, lick_ongoing_7Hz,
     lick_fast_close, whisk_run; angle_spectral lick_fast_close; asym lick_ongoing_7Hz, lick_fast_close, lick_slow_late.
     NOTE sniff_prelick == nose_prelick_slow and run_onset_jaw == prep_jaw give identical spatial readouts (two master
     factors matched to one fit column) -- also in the n = 100 run; check before quoting them as separate factors.
     CHECKED 11-07 (Priya asked): master spatial twins -- sniff_prelick / nose_prelick_slow and run_onset_jaw / prep_jaw
     spatial cos .99 (differ only in spectrum), and a jaw cluster (lick_ongoing_7Hz, run_onset_jaw, lick_fast_close,
     prep_jaw, lick_slow_late) at spatial cos .91-.99. Spatial-only best matching (MATCH_ON spatial, ASSIGN best) cannot
     separate them: sniff / nose share a column in 98 % of null fits, the jaw pair 47 %, lick_right / lick_left 0 %.
     Re-score with MATCH_ON product (interim folder, cohort_null_productmatch.csv; figure master_factor_twins.png):
     lick pair identical; sniff_prelick drops out (p .48); lick_ongoing_7Hz + lick_fast_close spatial angle hold;
     asym for lick_ongoing_7Hz / lick_slow_late and whisk_run spatial angle lose Holm. Method choice = Priya's
     (10-25 decision was spatial best-match); asked her 11-07.
     11-07 Priya: compare best vs Hungarian (spatial score) -> cohort_null_hungarian.csv: lick_right identical; Hungarian
     re-creates the 10-25 lick_left problem (acute lick_left .86 -> .79, results flip) and forces poor twin matches
     (nose_prelick_slow .51-.61) -> Hungarian rejected again. Priya then proposed: spatial score, frequency only to break
     near-ties. Tested (tie margin .02 / .05 / .10; .05 scored -> cohort_null_tiebreak05.csv): 10 matches change across
     the 4 cohort fits; figure match_changes_tiebreak05.png (maps + spectra per change). 8 / 10 clearly better (spectrum
     ~ master, spatial within .01-.05); acute lick_fast_close + acute / subacute prep_jaw: neither pick matches the
     master spectrum (factor likely not recovered there). lick_right unchanged. Awaiting Priya's read of the figure.
     RECOVERY (Priya 11-07: mark poor fits, the factors that DISAPPEAR are interesting; combine similar footprints?):
     match_quality_tiebreak05.csv. Null fits (should hold every factor): spatial 1st pct .84, freq 5th pct .46 overall,
     but per factor run_onset_jaw .26, prep_jaw .33, sniff_prelick .46 (unstable even pre; the real COHORT_pre already
     loses run_onset_jaw, freq .25). Rule tried: recovered = spatial >= .8 and freq >= .5. ACUTE loses lick_ongoing_7Hz
     (freq .50, borderline) and lick_fast_close (.46): lost in 0-1 % of null fits -> a real acute loss of the fast-lick
     jaw rhythm, back by subacute. prep_jaw subacute loss = in 21-34 % of null fits -> not a stroke effect. Proposal to
     Priya: merge only the unstable twins (sniff_prelick + nose_prelick_slow = prelick nose; run_onset_jaw + prep_jaw =
     slow jaw); keep the stable fast jaw factors separate (merging the whole jaw cluster would hide the acute loss).
     11-07 Priya "yes" -> CODE (ed350a43): `--match-rule v2` on face_rhythm_epoch_shape / drift_floor / cohort_null
     (v1 default, reproduces old numbers; v2 -> <folder>_v2/) + NEW scripts/face_rhythm_recovery.py (factor loss vs the
     permutation null). RUN on interim null (acute 497, sub / chronic 100): epoch_shape_v2/, epoch_shape_cohort_v2/.
     RESULT: acute LOSES lick_fast_close (p .002, Holm .004; 1/4 animals) and lick_ongoing_7Hz (p .014, Holm .014;
     2/4); back by subacute. v1 -> v2 diff (scratchpad v1v2_diff.txt, rerun scripts there): lick_right IDENTICAL in
     cohort / null / drift; per-animal acute lick pair 4/4 -> 3/3 (PS95 acute not recovered at the stricter threshold),
     so per-animal mean asym change acute lick_right -.125 -> -.053, lick_left +.144 -> +.302; cohort subacute lick_left
     loses one LOAO fit. Fast jaw factors acute move from "shape changed" to "lost". prelick_nose spatial acute p .004*
     (one unit now). chronic lick_far_nose: v1 had picked nose_steps' component (6 deg) -> v2 43 deg. Priya asked 11-07
     whether to switch page + deck to v2: answer = yes once the full null lands; page / deck text to her first.
     11-07 later: ACUTE NULL COMPLETE (500 pairs; subacute 111, chronic 100). v2 rescored (epoch_shape_cohort_v2/):
     acute Holm < .05 = lick_right (all 4 measures), prelick_nose + whisk_run + lick_slow_late spatial angle, whisk_run
     spectral; LOST lick_fast_close (p .002, Holm .004), lick_ongoing_7Hz (p .014, Holm .014). Waiting on sub / chronic.
     11-07 eve (Priya): v2 recovery spatial threshold .8 -> .7 (PS95 acute lick_right = same column as pre, spectral .94,
     spatial .78: changed, not lost; .5 / .7 identical on real fits; no null cell < .7). MAP STATS (Priya OK): cluster
     mass = the map test, max-statistic = point level, within-animal session-label shuffles (face_rhythm.stats
     map_permutation). Speed maps `face_rhythm_position_maps.py --perm 2000` (August-only pre, refreshed today): acute
     LESS movement cluster at every spout (far_R -.62, 926 pts), sub / chronic MORE at close + center. Pooled-fit null
     maps with clusters: lick_right acute 2 clusters (604 pts; max-stat 2 pts), lick_left acute none. NEW per-session
     weighting test `face_rhythm_session_loading_perm.py` (own8 fits, v2) -> session_loading_v2/: lick_right acute loses
     upper / central weight (645 pts), gains ipsi lower cheek + jaw; lick_left roughly mirrored; max-stat 263 / 413 pts.
     CAVEAT: session shuffle = fixed effects + pre drift -> OPEN: early-vs-late pre drift control for the speed and
     session-loading tests; re-run aligned_maps (cue / lick / bout, still 10-24 baseline) with --perm. PS93: own acute fit
     runs OPPOSITE to PS92 / 94 / 95 (pre-existing right facial deficit, 10-24 flag; Priya 11-07 confirms) -> keep PS93
     in main analyses, show "without PS93" (= LOAO-PS93, retitle) as sensitivity; open: PS93 per-session check. Maps now
     drawn like the speed maps (all points colored, red / blue rings). Rhythm-band movement maps per lick factor:
     proposed, not started. Lick-rate (ILI) figure: requested, not started.
     11-07 late (Priya "go ahead with all"): CORRECTION -- FR "pre" was August-only all along (aligned_maps cache = the
     same 31 pre sessions); position-map vs aligned-map differences = window (0..4 s RMS vs 0..2 s mean speed from point
     PCs), not baseline. DONE: shared `face_rhythm.stats.session_shuffle_test` / `drift_split` / `chain_adjacency`;
     `--perm` + `--drift-control` on position_maps, aligned_maps (cue / lick / bout -> aligned_maps_<al>[_driftpre]_perm.png)
     and session_loading_perm; spectral null clusters along frequency; "LOAO-x" shown as "without x"; consistency maps
     captioned descriptive. DRIFT: position maps none; aligned cue 3 UP clusters (far_center 374, far_R 315 + 280 pts),
     lick 1 UP (close_center 254) -> acute DOWN effects safe, chronic UP clusters need the drift caveat; session loading:
     lick_right none, lick_left 145 pts MORE ipsi-central (opposite sign / place to its acute change), also
     lick_ongoing_7Hz, nose_steps, lick_far_nose. SPECTRAL: clusters find fewer than max-stat (lick_right acute max-stat 5
     bins, clusters 0): spectral changes are narrow peak shifts -> keep max-stat for spectra (recommend). PS93 per
     session: acute 08-18 / 08-20 go WITH PS92 / 94 / 95 (r +.52 / +.34), 08-19 / 08-21 AGAINST (r -.43 / -.57) ->
     PS93 inconsistent acutely, not reversed (session_loading_v2/ps93_acute_sessions_lick_right.png).
     11-07 NIGHT -- STATE FOR THE NEXT WINDOW (decisions: DECISIONS 2026-11-07). Per-session tests = main shape tests
     (session_shape_v2/, session_loading_v2/; scripts face_rhythm_session_shape_tests.py / _session_loading_perm.py),
     all delta vs own pre, drift-controlled. RESULTS (acute unless noted): spatial angle vs own pre up for most units
     (lick_right +14.6 deg, lick_left +12.1, Holm < .05; pre drift ~0.6 deg, none Holm); asymmetry lick_right -.15*,
     lick_left +.23* (also sub / chronic); side specificity lick_right -.38*, lick_left -.31*; LOSS per session:
     lick_ongoing_7Hz -.44 (p .006, Holm .065), lick_fast_close -.21 (p .044), chronic lick_left -.28 (Holm .005, mostly
     PS93 2/12) -> per-session loss weaker than the pooled-fit loss; MAPS lick_right robust (838 / 880 scaled / 786
     unambiguous pts acute), lick_left acute untestable without ambiguous sessions (2 left); right-lick share does NOT
     explain the lick_right map change acutely (rho -.14, p .61); fragility: lick_left 7.2x, lick_far_nose 5.8x (pooled
     inconclusive); lick direction: acute right share -.16, left +.24 (p < .001). NEXT: (1) make v2 + per-session the
     default and re-render every FR figure in one pass (wait for Priya on the sub / chronic O2 null: keep or scancel);
     (2) draft FR page (fr_page_src on M:, published from Priya's HMS account) + PI deck FR slides + FINDINGS wording ->
     Priya approves before any text change; (3) rhythm-band movement maps per lick factor (proposed, not started);
     (4) lick-rate (ILI) figure for page + deck (requested, not started); (5) encoder FR regressors read proj/ (pre
     sessions slightly in-sample; proj_loso exists) -- flag only.
     11-07 LATE -- FINAL STATE. 10,000-shuffle per-session tests done, Holm within family ACROSS all epochs x
     measures (Priya); FINDINGS 5.10 (PER-SESSION paragraph), FINDINGS 0 "Where things stand, face movement",
     FACE_RHYTHM_FINDINGS 6e written. Figures: session_shape_v2/session_scalars.png (normalized, leave-one-out pre
     column) + session_scalars_raw.png (Priya: both, separate) + session_loss.png; headline map = scaled
     (session_maps_scaled_lick.png); `face_rhythm_session_shape_tests.py --plot-only` redraws. FR PAGE: a helper agent
     was updating it (new per-session section before section 7, speed-map permutation figures, "Where things stand");
     check the live page version before more edits. PI DECK NOT UPDATED: Overview_2026-11-06.pptx was open (lock) all
     evening -- when closed, update slides 83 (matching method -> v2), 85 (loading maps -> session_maps_scaled_lick),
     86 (spectral), 87 (hidden drift floor -> session_loading_driftpre), add per-session shape (session_scalars) + loss
     slides; notes Data: block per slide; Priya approves the slide text. Lick direction is LP kinematics -> not an FR
     figure. O2 sub / chronic null kept running (Priya) -> rescore pooled view when files land (descriptive only).
     11-07 EVENING: PAGE v40 + DECK (198 slides) UPDATED -- details and slide map in the a8 handoff top block
     (STATUS_2026-11-01_QUIET_EVENTS_HANDOFF). Page: cue- and lick-aligned perm maps are the main cue / lick speed maps
     (same speed measure), 0-4 s response maps = supplement (Priya). Open: rhythm-band movement
     maps per lick factor, re-render remaining FR
     figures (pooled-fit figures with session dots / pre column), rescore the pooled null when the O2 sub / chronic
     files land.
     11-07 NIGHT: v2 is now the DEFAULT `--match-rule` in epoch_shape / drift_floor / cohort_null (v1 still selectable;
     output folders keyed on the rule name, so v2 keeps writing to *_v2 and v1 outputs stay put). LICK-RATE FIGURE
     (Priya asked): `scripts/orofacial_lick_rate_figure.py` redraws the ILI + fast-lick-share rows of the v36 cohort
     engagement figure from its saved tables -> LP_analysis/.../cohort/figures/prepost/cohort_lick_rate_by_position.png.
     ILI +15-25 ms at EVERY spout acutely (*), back by subacute except close spouts; fast-lick share falls acutely only
     at far contra (-.44) / far mid (-.13) -- partly fewer licks (far contra 3.1 licks / trial vs 8.7 pre; an isolated
     lick has no neighbor), so ILI is the rhythm measure. Not yet on page / deck (caption to Priya first).
     11-07 NIGHT (2): ALL MAP TESTS NOW 10,000 SHUFFLES (Priya), like the scalars: position, aligned, session_loading
     (default now 10000) and the new band maps re-run with drift controls; 2000-shuffle cluster tables kept in the session
     scratchpad for the comparison. Changes vs 2000: position maps none (17 cells, pts +-6); aligned: lick close_L
     subacute up (258 pts) and close_center chronic down (131) NEW; CUE DRIFT CONTROL 3 -> 2 clusters (far_center 374 pts
     gone; both left at far_R, where the post-stroke change is a decrease) -> page 3.10a / 3.10b text + slide 82 bullet
     need rewording (Priya approves); session loading: lick pair unchanged; lick_slow_late acute more (172) gone, whisk_run
     acute more (112) new (exploratory). RHYTHM-BAND MAPS (`scripts/face_rhythm_band_maps.py`, readouts/<set>/band_maps/):
     acute decrease at every spout in every lick band, graded by rhythm -- face mean log2 7Hz -.30, fast_close -.20, lick
     rhythm -.16, far_nose -.12, slow_late -.09, slow_jaw / prelick_nose ~0 -- fits the ILI slowing; later epochs +.06 to
     +.13 at close spouts; drift control has 13 small clusters (mean drift +.03; far_L up, close_R down) so later-epoch
     band increases need that caveat. The band gradient is not yet a test (proposed: per-session 7Hz-minus-rhythm contrast,
     within-animal shuffles). Not on page / deck yet.
     11-07 NIGHT (3): GRADIENT TEST DONE (`face_rhythm_band_maps.py --gradient`): 7 Hz band minus lick rhythm -.13 log2
     acute (Holm .002), -.04 subacute (.03); slower lick bands fall less; drift none -> FINDINGS 0 / 5.10, FACE_RHYTHM 6f.
     PAGE v41 + DECK (201 slides; 83 lick rate, 84 band maps, 85 hidden band drift) updated with Priya's OK, incl. the
     10k rewording of 3.10a / 3.10b / slide 82. Open: rescore the pooled null when the O2 sub / chronic files land
     (descriptive); the deck-wide notes "Data:" pass (waits on 34).
     11-07 NIGHT (4): REWORDED as a FREQUENCY SHIFT (Priya: what does "the 7 Hz band falls more" mean -- the
     lick_ongoing_7Hz profile is the upper lick spectrum, 6.5-15 Hz). New readout `face_rhythm_band_maps.py --centre`:
     face movement spectrum center 6.77 -> 6.56 Hz acute (4 / 4, Holm .0003), back subacute; licks 6.45 -> 5.71 Hz.
     Page v42 (7b heading / text / status row / 7b.3 title), slide 84 title + bullets + notes
     (`reword_band_slide_1107e.py`), FINDINGS 0 / 5.10, FACE_RHYTHM 6f, DECISIONS updated.
     11-07 NIGHT (5): POOLED NULL INTERIM RESCORE (v2). Copied-back O2 null pairs: acute 500, SUBACUTE 379 (was 111),
     chronic still 100. Scored into readouts/master_pre_v1/epoch_shape_cohort_v2/interim_sub379_chr100_20261107/
     (page outputs in data/ untouched; scratchpad interim_null_score.py imports compute(), no code change).
     Acute and chronic p identical to the page.
     Subacute: the six cells that change Holm status are all crossing from "not" to "Holm < .05":
     - lick_ongoing_7Hz / lick_fast_close / lick_slow_late / whisk_run spatial angle: .07-.09 -> .02-.03;
     - whisk_run spectral angle: .07 -> .02;
     - lick_right asym: .052 -> .021.
     This is the p floor (1/112 = .009 -> 1/380 = .003), not new data: observed values unchanged.
     lick_left side_spec p .009 -> .032 (still Holm < .05).
     Chronic is still floor-limited at 100 perms (several cells at p .010 without a star).
     Page / deck NOT changed (pooled view is descriptive; Priya to decide whether to swap in after chronic lands).
     Priya 11-07: "can wait" -> rescore again when chronic perms land; no page change before.
     11-08 (overnight) JUNE FR (Priya: June as a reference): June ROIs warped from the master through the existing June
     fields (`face_rhythm_warp_rois.py to-sessions --june`; landmark errors as August's), QC `register/qc_june_rois.png`;
     O2 bundle `o2/fr_cam4_june_20261108` (12 tasks, production settings) SUBMITTED by Priya. Off-frame mesh points:
     PS95 06/07 4, 06/08 29 of 1282 (exclude in readouts). June face is 1.17-1.24x larger: displacement amplitudes need
     scale normalization before comparing with August. Running list: RUNNING_LIST_2026-11-08_a8_june.md.
