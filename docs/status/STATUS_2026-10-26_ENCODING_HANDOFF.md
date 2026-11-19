# Encoding handoff — 2026-10-26 (updated ~22:45) — READ FIRST

## 00. START HERE (22:45, second session of 10-26; supersedes section 0 for state / to-dos, section 0 kept as history)

Session = `f082ee0c` (new scratchpad `S2 = C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad`;
push scripts `S2/push<NN>.sh`, last `push39.sh`). Python / PYTHONPATH as below; `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache`
for anything that loads imaging. E = `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding`.

### 00a. Page state
- LIVE: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d = **v118** (this session published v104-v118; the Artifact
  tool works here). Source `_goal_page/index.html` + PNGs (backup `M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/`,
  refreshed at every publish).
- Section 3 was renumbered TWICE tonight: new 3.21 (`iti_where_delta.png`) and 3.22 (`iti_where_per_spout.png`) -> old
  3.21-3.31 are now 3.23-3.32 (kinematics control = 3.27 now). Map: `_goal_page/fig_map.csv` (last renumber only) +
  `docs/status/GOAL_PAGE_FIG_MAP_2026-10-26.csv`. Insert new figures with a temp label like `Fig 93x` (renumber_sections.py
  cannot parse `3.20b`) and run `S/renumber_sections.py <index.html>`.
- Publishing: (1) `Artifact read` first in a new session (or a refusal hands you the live version); (2) replacing old
  PNGs needs `Artifact list scope=files` first; (3) the full PNG set (~65 MB) is over the 64 MB cap -> batches < 35 MB;
  (4) `edits` on publish is refused in this session (file-edit hook) -> Edit the file, then publish without edits.

### 00b. RUNNING at 22:45 (background in session f082ee0c; may die with it)
1. **Sagittal chain** `S2/run_sagittal.sh`, log `S2/sagittal.log` (started 22:43): iti_where_decoder (3 bases) ->
   iti_where_scatter -> iti_where_beyond_pull -> iti_zscore_check -> iti_where_per_spout (+ --z) -> iti_where_mae ->
   iti_decoder_contrib. Ends with `sagittal chain done`. If it died: rerun the missing steps (same order; each reads the
   previous step's CSV). Old lateral + OUTWARD outputs backed up in `<E>/iti_licks/outward_version_20261026/`.
2. **Push** `S2/push39.sh` (4972bdf sig bars + handoff). Then push **f253fa7** (bar style / y-labels) and this handoff
   commit by cherry-pick (template = push39.sh with the new SHA). origin/main had bf0b70e (FR window) at 22:44.

### 00c. AFTER the sagittal chain (in order)
1. Check every redrawn figure (3.15, 3.15 MAE, 3.16-3.18 bases / LocaNMF / SVD, 3.19-3.21 scatter + delta, 3.22 +
   z version, contrib). Compare the SAGITTAL numbers with the outward ones (KEY row says cued licks "read closer" on the
   OUTWARD axis: beyond-pull -0.30 / -0.30 / -0.31, z-scored -0.22 / -0.21 / -0.17). **Tell Priya before changing the KEY
   row / captions / findings if sagittal differs.**
2. Copy into `_goal_page`: iti_where_decoder.png, iti_where_decoder_bases.png, iti_where_decoder_locanmf.png,
   iti_where_decoder_svdbasis.png, iti_where_mae.png, iti_where_scatter_{trial,iti}.png, iti_where_delta.png,
   iti_where_per_spout{,_z}.png; update captions (axes are now lateral + SAGITTAL extension = sqrt(forward^2 + down^2);
   angle = atan2(lateral, sagittal) = the encoder's exec_angle; outward still scored in where_decoder.csv); publish with
   those files; back up; commit; push.
3. Then tell **sabatinilab-e3** (PI deck) the decoder figures are final: per figure the share PNG path, page Fig N.k,
   script + args, headline numbers, methods text (they asked 10-26 ~22:30; reply sent: not final yet).

### 00-COMPACT-45 (11-18 ~23:30) -- START HERE; additions to 44

- **Movement (Execution) page** https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d = **v326**. Section 23 (quiet-state) is on
  a8's sync-end quiet run `cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118` for every figure except 23.23.
  - v324: 23.1/2/4/5/6/11/15/17/18/21 (restyled, boot_delta stars). v325: 23.3/9/10/12/13/14/16/19/20/22/24 (scripts' own
    layouts re-plotted with bootstrap stars, `*_boot_stars.csv` beside each png; code fae48012). v326: 23.7 + 23.8.
  - 23.8 quit period = quit .. last sync pulse (Priya 11-18, code 9701d914): pre still-UP quit / engaged rate x5.6 / x7.4 /
    x5.6 (PS93 / 94 / 95; PS93 = 1 session, 4.2 min); PS92 has no pre quit period; acute ENGAGED 7.6-15.3 /min >= pre quit
    5.4-17.1 /min, so "acute engaged quiet looks like pre-stroke disengagement" stands at a smaller ratio (was 8-60x).
  - Caption audit + bootstrap corrections: `docs/status/quiet_restyle_20261118/S23_PAGE_CAPTION_AUDIT_20261118.md`; page
    edit scripts `quiet_restyle_20261118/page_s23_scripts/` (harness `s23_check.py <part>` on the saved page source).
    Page source for the next edit: session f082ee0c scratchpad `s23_page3.html` (else `Artifact read` the live page).
- **23.23 BLOCKED** (a8): stored rebuild decoder loads, features match, but rebuilt full-window p_true vs stored T13 p_true
  differs (median |diff| 0.14; lead: stored run used session_z-scaled models). Options with Priya: fix the transform /
  working centring, or keep 23.23 pre-rebuild. Until then KEEP `labcams/cohort1_quiet_spikes/v2_quiet_lp/task_events` +
  `task_events_followup`; the rest of the pre-rebuild v2_quiet / v2_quiet_lp was deleted by a8 with Priya's OK.
- Spout (Target) page PvraHi cites no v2_quiet paths (its section 15 = frozen copy, images inside the artifact).

### 00-COMPACT-44 (11-17 ~19:30) -- START HERE; additions to 43 / 42
Detail: `docs/status/PAGE_PUBLISH_QUEUE_20261117.md` (branch docs/publish-queue-1117, worktree wf_imgswap1117, UNPUSHED).

- **Pages (live):**
  - IMG v58 (Fig 2.3 footer fix, section 13 on the log deficit, 42 lick-source edits).
  - SS v44 (per-basis captions 6.3 / 6.4, 142 lick-source edits, SVD 11.x + 10.13-10.16 cam4 wording, Methods box).
  - KIN v32 (27 lick-source edits).
  - GOAL v316 (mine: ITI 0..0.5 s swap, Fig 14.7 cued 0..1.0 s, 110 lick-source edits, "(was Fig 14.20)" fix, idx supplement collapsed after 14.13).
  - Then a8 published GOAL v317-v319 (section 12 only: Fig 12.2 = combined like-for-like, 12.2 S2 non-linear).
  - **Build any goal edit on v319+.**
  - Still on pre-rebuild data on the goal page: Fig 14.12, the 14.18 S14a / S14c captions, the status-row "earlier definition", and the 14.25 baseline-sensitivity image. Re-render offered to Priya; no answer yet.
- **PI deck (a8):** hidden slide 126 inserted, so slides after 125 are +1. RESTnl slides to re-point are 24-26 and 187-189. Match slides by title.
- **cam1 extra licks:**
  - Priya's blind review is scored (30/40 tongue; 7/8 no-tongue events are far-from-mouth labels).
  - RULE (Priya 11-17): cam1-only events count only if non-droplet AND `med_out_px >= -120`.
  - Records: `CAM1_EXTRA_LICKS_CHECK_20261117.md`, `CAM1_FARMOUTH_BLAST_RADIUS_20261117.md` (branch analysis/restnl-cam1-1117, worktree wf_restnlc1, UNPUSHED).
- **RUNNING 1: RESTnlc1 v3 overnight queue** (`restnl_licks` v3, 3bf2dd41; helper re-run with v2 outputs moved aside to `_superseded_restnlc1v2_20261117/`, record `RESTNL_V3_RERUN_20261117.md`).
  - Queue: `scratchpad/rc1/queue_rc1.sh`; `touch rc1/STOP` halts it.
  - Lane D finished 18:58; B and A/C were running at 19:20 (ETA ~21:00-21:30).
  - Then: old-vs-new tables per animal to Priya. Only after that: place IMG Fig 12.1 (c1all, wording approved: drops acutely, somewhat below pre chronically, subacute n.s., ~half the old size, provisional), update the FINDINGS rest line, re-point deck 15f, and send a8 the RESTnl slide paths.
  - Queue fixes 11-17: the SS rest baseline reads `enc_rebuild_20261110g/inputs` via `WF_SS_ENC_INPUTS` (04b moved aside in the glue fix; imaging byte-identical, checked on 3 sessions), and `restnl.variant()` accepts restnlc1 (d2fc1d83).
- **RUNNING 2: guard in code everywhere** (Priya OK "with the droplet exclusions").
  - Helper on branch fix/cam1-farmouth-guard-1117 (worktree wf_c1guard1117): one shared keep-mask used by trial_classes, first_lick_latency, lp_licks, combined_encoding, revup_behaviour, cam1_incomplete_licks, cohort2_qc/lp_licks.lick_times; cache bumps; tests.
  - Show Priya the diff before any push. Send a8 the commit hash.
- **a8 owns the quiet-event re-run** (goal section 15, deck 112 / 157 / 158; quiet time loses 0.9 %, chronic cells up to 2.9 %). It runs from the guard branch and brings Priya per-animal chronic numbers first.
- **Approved, not started:** switch the pre-cue (ENL) / ITI lick-free windows to cam4 + cam1 with the v3 rule. Start after the RESTnlc1 tables. Show old vs new before pages.
- **Priya said NO** to a cohort-wide check of sessions where cam4 video ends early. PS92_0908: camera classes cover trials 1-321 only, which is correct.
- **Unpushed branches (each needs Priya's OK):** docs/publish-queue-1117, analysis/rest15f-rebuild-1117, analysis/restnl-cam1-1117, fix/cam1-farmouth-guard-1117.
- **After the re-runs:** stale-iteration cleanup discussion (queue doc section 4).

### 00-COMPACT-43 (11-17 late morning) -- additions to 42 (42 still the state record)

- PUSHED 11-17: fix/page-figs-1116, fix/deck-section-g-1116 (+ test_enl_window_source pin for the source-aware deck window call),
  analysis/ss-gluefree-1117 (compare script only).
- SS SVD / LocaNMF Figs 6.3 / 6.4: glue-free re-run (rebuild_20261103/state_space_recovery/svd_lnmf_gluefree_20261117/)
  reproduced the 11-08 tables and figures byte for byte; HOLD lifted (Priya; a8 told: no PI slide uses them). SS page v41 adds
  one per-basis sentence to each caption (LocaNMF axis within-epoch p .054; SVD position .049; step wins 1/24, 3/24).
- Image page v56: the 1.0 s LP lick-free pre-cue window is PRIMARY (Fig 2.5); 1.5 s = Fig 2.6, window-length check;
  Pending item rewritten (only the stored-frozen-default question stays open).
- MAIN CHECKOUT fast-forwarded 11-17 (was 663 behind; the editable install points there). Rule in CLAUDE.md Reference:
  PYTHONPATH=<worktree> for every `python scripts/...` run from a worktree.
- RUNNING: analysis-deck helper (branch fix/deck-figs-1117, worktree wf_adeck1117): deletes the 4 wrong frozen decoders
  (list -> docs/status/DELETED_DERIVED_20261117.md; Priya OK), matches Key fig 2 / 12.2 / 14.18 rebuild renders, places
  IMG 2.4 / 8.1-8.2, 15r skip-no-camera fix + render, CD leave-one-out re-run on the rebuild (old vs new to Priya before the
  image page), M_POSTSTROKE + section K captions rewritten, and 18 page key figures (Priya OK'd the list; IMG 2.5 not 2.6).

### 00-COMPACT-42 (11-16 evening) -- READ FIRST; supersedes 41 for STATE (41's naming / two-measure / sign-frame notes and 38 section 1 RULES still apply)
Session f082ee0c; scratchpad `C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad`. Detail: RUNNING_LIST_2026-11-06.md entry "11-16 (evening)".

**ON MAIN (pushed 11-16 evening, Priya "ok to push"):** feat/ss-balance-level-row-1116 (619c95a3), feat/balance-angle-sharedy-1116
(c6416ed3), analysis/attempt-coding MERGED into main (704f71f1; one conflict s1_refit_basis_table.py -> main's 1.0 s LP ENL + the
branch's --suffix / camlickpl arms), fix/ss-nulls-1116 (affe7d3e: circular shift + whole-spout-block splits), fix/kin-gated-pooled-1116
(cc8ce9d0: doc KIN_PAGE_GATE_ITEMS_2026-11-16.md). Earlier: FINDINGS decoder-offset wording (fa9a4907). attempt-coding is no longer
a separate line of history: branch new work from origin/main.

**RESULTS (11-16 evening, scratch `prelick_test/` + `prelick_test/head_to_head/`):**
- RESTnl session levels (107 sessions, checkpointed `rest_sessions/`): orofacial REST_R - REST_L change vs own pre acute -0.73 / -1.98 /
  -2.82 / -2.55 (every pair 4/4), chronic +0.80 / -2.13 / -0.46 / -0.82 (session-mean reference: partly the mirror of the task-period
  contralesional rise). dec_rest - pre_rest == decbc to 1.6e-14.
- Head-to-head vs far-contra aim (post n = 75, circular shift 10k, ln(day)-detrended): 14.18 marker +.78 / +.54 cued, +.70 / +.59 ITI
  (p <= .0001); RESTnl level +.49 / +.34 (4/4); raw evoked ratio resp02 n.s. / +.29 (2/4); gain, dec, pre-lick variants lose it after
  detrending; r(rest, marker) ~ +.3 n.s. Raw ratio rises acutely while the angle-matched marker falls (acute licks land ipsilesionally;
  the pre line at that angle predicts an even larger contralesional share).
- Null fix: no star changes (SS 6.3 / 6.4 / 10.25); section 10 p values move; Priya: KEEP Fig 10.23 framing (time and recovery are linked).
- Kinematics gate: Priya "new figures look good"; jaw opening far contra acute -2.42 -> -1.85 mm, 4 small jaw stars lost.

**RUNNING (delegated helpers, this session):**
1. Page update (3 pages): KINEMATICS (gate figures + captions; PI-slide-35 far-contra outcome figure becomes Fig 6.1, rest renumbered),
   SS (circular-shift wording + p values; figures copied to a page_figs_nullfix_20261116 folder), ENCODING / goal page (new rest-level
   figure after the section 14 balance figures; 14.18 trend-safe sentence; old ITI "-0.15 r, -23 %" -> Fig 10.3's -0.130 / -0.122 / -0.130).
2. Analysis deck (branch fix/deck-section-g-1116, worktree wf_adeck1116): G2b / G3 / G6 / counts -> LP video licks; outdated text out;
   small-lesion slides dropped; NEW key-figures section from the four pages; of the missing families outside the queue render only
   those cited on a page or in FINDINGS (report before any > 30 min render). Push the branch when it finishes.

**DECISIONS (Priya 11-16):** 14.18 pool = current five areas; rest-level result on the ENCODING page with a title / caption that says it
has no lick in it; 14.18 caption gets the trend-safe sentence; whole-number figure numbering, no letters; deck as in RUNNING item 2.

**TO DO / OPEN:**
- Relay page links + deck report to Priya; push fix/deck-section-g-1116 after the rebuild.
- Production path bugs (from the null-fix helper): within_session_behavior.WSD and detour_vs_balance.RECPRIM resolve under
  rebuild_20261103 where the files do not exist (join silently skipped; detour would fail); block_size_max_for finds no behavior dir
  on this box (every session uses 8).
- Fig 14.13 shared-y version onto the goal page (Priya not yet asked to decide); block-permutation p for the per-lick fits (open).
- Fig 9.10 re-run (bundle moved: o2/enc_rebuild_20261104b/env.json); SS section 7 detour + Fig 3.9 / 6.12 sources; CUT_FIGURES rows for
  the IMG cuts; re-render deck sections A-C after the code cleanup.
- Kinematics: section 6 per-animal jaw-only table, section 7 no-contact table and Fig 7.6 not re-gated (outside the class shares).

### 00-COMPACT-41 (11-16) -- READ FIRST; supersedes 40 for STATE (38 section 1 RULES still apply)
State after the glue-free re-run (v20, done 11-12) and the 11-16 page / deck pass. Detail: RUNNING_LIST_2026-11-06.md,
entries "11-13" and "11-16"; per-artifact reports in the session scratchpad pages/<goal|img|ss|adeck|pideck>/REPORT.md.
- PUBLISHED 11-16: GOAL page v313 (D1-D5), IMG page v53/54 (D6 + session-mean twins, Q9 cuts), SS page v38 (45 areas,
  61 number edits, 5-row balance figures, hemi-gain wording, naming). PI deck Overview_2026-11-06.pptx rewritten 13:03
  (backup .bak_pre_rebuild_20261116); FR window then edited slides 90 and 163 (its own backups).
- CODE ON MAIN (11-13, cfb83124): state-space consolidation (docs/STATE_SPACE_CODE.md); state_space_recovery output root
  moved to rebuild_20261103/state_space_recovery. UNPUSHED BRANCHES: feat/ss-balance-level-row-1116 (5-row balance
  figures, AREA_SET_NAME naming, balance_defined: no balance under gsr / spout), feat/balance-angle-sharedy-1116 (Fig
  14.13 one y scale per column), fix/deck-section-g-1116 (deck text; A-C + small-lesion removal in progress).
- NAMING (Priya 11-16): "orofacial" alone = SENSORIMOTOR; say "orofacial sensory" (SSp-m, SSp-n, SSs) / "orofacial motor"
  (ALM, tjM1) for the subsets. The pooled sets differ by figure (14.13 = ALM + tjM1 + SSp-m + SSp-n; SS section 10 =
  tjM1 + SSp-m + SSp-n; 14.18 primary marker = those four + SSs): state the areas in each caption.
- TWO BALANCE MEASURES (keep apart; they disagree on ITI licks):
  * RESPONSE RATIO (resp; 14.9 / 14.13 image 1): log2(contra / ipsi AUC 0..1.0 s after the lick, each hemisphere minus
    its own -1.0..-0.5 s baseline). Needs lick-free windows on both sides, so it exists only for the FIRST cued lick and
    the ITI bout start after >= 2 s without licks (6-7 Hz licking puts ~6 other lick responses inside a later lick's
    1 s window). A ratio: invariant to a bilateral gain.
  * DECODER-WINDOW BALANCE (dec; 12.17 / 14.13 image 2): X_right - X_left averaged over 0..0.2 s after the lick peak,
    x1e3 dF/F, NO baseline -- X = the per-lick Allen-area traces (iti_X / tr_X) that are the INPUT FEATURES of the
    per-lick lateral WHERE-decoder (iti_lick_order / iti_where_decoder: ridge on Allen areas with the ALM / tjM1 split,
    lags 0..0.2 s, features z-scored within session, trained on August pre licks per set, pre sessions leave-one-out;
    predicts the lick's lateral landing in mm). Defined for EVERY lick, but carries the pre-lick level (tonic offsets,
    earlier licks' responses) and, as a difference, scales with response size. The where-decoder's lateral weights are
    largely ipsi-minus-contra (11.13), so this balance is roughly a one-number summary of what that decoder reads: it
    behaves like the decoder (ITI similar to cued, smaller) where the response ratio does not (partly circular; see
    lick_balance_reference.py docstring). NOT the frozen spout-position decoder of section G / IMG 2.x.
  * 11-16 scratch (scratchpad bal_angle4/): decoder-window balance for first cued / all cued / ITI bout start (>= 1 s
    gap) / all ITI licks: all-lick rows are tighter and steeper; ITI licks track angle in every area incl. ALM / tjM1
    (about half the cued slope); post-stroke ITI shows an ipsilesional shift at the same angle (orofacial -3.3 / -1.7 /
    -1.6) that the response ratio does not. Exact ">= 2 s quiet" ITI-first version pending (needs the per-lick table).
- TWO MEASURE FAMILIES -- say which one every number belongs to (Priya 11-16: "hard to keep straight"):
  * DECODER-FEATURE family: raw lick-time activity 0..0.2 s, NO per-lick baseline, zero = each area's SESSION MEAN
    (drift removal incl. the constant), divided by the frame-level SD -- no per-lick re-centering. Hemispheres combine as
    a DIFFERENCE X_right - X_left (or the where-decoder's weighted sum). Picks up evoked lateralization + TONIC level
    relative to the session mean + overall GAIN. Members: frozen where-decoder output and its like-for-like offset
    (12.x, 14.22, FINDINGS 10-29 hypothesis), "hemispheric difference" figures (14.13 image 2, 12.17 decomposition,
    lick_balance_{order,offset_lmm,regions,recovery,maps,reference}), pre-lick test rows 1-3. A whole-session constant
    shift does NOT reach it; a change of lick-time activity RELATIVE to the session mean does.
  * RESPONSE (AUC) family: evoked response = window minus that hemisphere's own pre-lick baseline; hemispheres combine as
    a RATIO log2(contra / ipsi) = THE BALANCE. Evoked lateralization only (gain and pre-lick level removed). Members:
    response ratio (14.9, 14.13 image 1), primary recovery marker (14.16 / 14.18, angle-matched gain ratio), state-space
    balance + hemisphere gain (SS section 5).
  * Labels: "balance" = ratio only; X_right - X_left = "hemispheric difference" (branch feat/balance-angle-sharedy-1116,
    2d2107c5). SIGN FRAMES: hemisphere frame (which side larger; - = ipsilesional larger) vs read-out frame (decoder: + =
    lick read more contralesional) are OPPOSITE: an ipsilesional-hemisphere shift reads as a contralesional lick.
  * Why "contra gains relative to ipsi" (ratio) and "difference shifts ipsilesional" coexist: pre-stroke ipsi > contra at a
    typical lick (e.g. 4 vs 5); both grow, contra relatively more (6.4 vs 7.5): ratio .80 -> .85 (contralesional), difference
    -1.0 -> -1.1 (ipsilesional).
- PER-LICK pre-lick test (11-16; scratch prelick_test/per_trial.py, OLS with session-clustered SEs -- MixedLM stalled at 0 on
  one fit): ITI first >= 2 s, orofacial, a/s/c: raw difference -4.1 / -1.8 / -2.1, pre-lick level -1.7 / -1.4 / -1.4, evoked
  difference -2.5 / -0.5 / -0.6; angle slope raw -.31 = evoked -.30, pre-lick -.01 (session-level fits agree). The
  difference is session-mean referenced (NOT RESTnl); a RESTnl version is being built (helper; two-references rule).
- DIRECT DECODER CHECK running (Priya OK 11-16; scratch decoder_check/): re-score the frozen where-decoder with (A) the
  pre-lick level removed, (B) session response size divided out, (A+B); how much of the contralesional ITI offset
  (+0.39 / +0.29) survives. If little: the FINDINGS 10-29 hypothesis wording ("code shifts contralesional") changes ->
  Priya BEFORE any text.
- RUNNING (helpers, 11-16): analysis-deck rebuild without A-C + small-lesion slides; kinematics gate items (section 6
  accepted; incomplete-lick text + legacy-gate sensitivity row; gated pooled re-run on fix/kin-gated-pooled-1116).
- OPEN WITH PRIYA: 14.18 marker with / without SSs (table check offered); ">= 2 s" ITI version of the 4-set figure; goal
  page old ITI numbers vs new Fig 10.3; SS section 7 detour + Fig 3.9 / 6.12 sources not re-run glue-free; Fig 9.10 re-run
  blocked (state_space_why_cleaner reads o2/enc_rebuild_20261104b/env.json, bundle moved 11-11); A-C re-render after the
  code cleanup (agreed: yes, later); C6 balance numbers + HELD list (PAGE_DRAFT_REBUILD_2026-11-10.md) unchanged.

### 00-COMPACT-40 (11-09 ~07:50) -- READ FIRST; supersedes 39 for STATE (38 section 1 RULES still apply)

**Where things are:** running list `docs/status/RUNNING_LIST_2026-11-06.md` (11-09 entries at the end);
`docs/status/HEADLINE_TABLE_2026-11-09_CORRECTED.md` (8 conclusion changes; Priya approved 1-6 + 8 for pages / decks, #7
resolved below); `docs/status/AMPLITUDE_TWO_REFS_AUDIT_2026-11-09.md` (two-reference gaps). Render folder
`scratchpad/render_daqlk` (session f082ee0c), queues 9zw..14, `logs/CHAIN.log`; chain scripts `scratchpad/chain_v1*.sh`.

#### 1. Decisions since block 39 (Priya 11-09)
- G4 = camera working vs no_response (as G1-G3); re-run ONLY G4 (`poststroke_section_g --only g4`, backup first).
- June camera trial tables = SIDE table + `WIDEFIELD_ATTEMPTS_SUPPLEMENT` for the LP drift step only (registered table
  untouched; its mtime/size keys every LP cache).
- Two LP lick epoch decoder sets (float-noise inputs) ACCEPTED for now (set 1 epoch arm lick figs; set 2 grant lick units).
- contactOnly section G arm DEFERRED; when run it supersedes the mistaken r4 run, whose outputs may then be deleted.
- Interhemi / cross-area / coupling: PRIMARY lick - RESTnl rest; SUPPLEMENT "raw" = lick-set metrics, no rest reference.
  restdock05 is NOT raw (its _r5 chain marked NOT RUN). Session-wide event set dropped.
- AMPLITUDE outputs need TWO references: RESTnl + session mean (memory amplitude-two-references). D0: map "raw" accepted
  (caption: zero = slow inter-trial trend). Per-trial-baseline families: add absolute-level twins + state-space response-
  size twins at the page update. Hemisphere-difference balances: no RESTnl if lick-peak fluorescence not normalized
  within trial (verify). CD projections: HOLD.
- Working period start = FIRST camera lick (settled). RESTnl boundary rule kept (rest periods need both bracketing
  trials in the period).
- #7 (acute cue far-contra map, RESTW per animal): wording "falls in 3/4; PS95's single acute session has only 40
  far-contra working trials (1 hit + 39 incomplete; of 120 cues 53 jaw-only, 16 no response, 11 stopped) and shows no
  drop on the trials it attempted" (old "halving" averaged non-attempts). EVERY figure must state EXACTLY which trials
  were included / excluded (memory figure-provenance-in-captions). Map trial sets = option (c): primary working trials
  (contact + incomplete), SUPPLEMENT = all trials in the working period (+ jaw-only + no response; stopped runs, quit
  tail, unknown excluded), PLUS a jaw-only panel POOLED per animal x epoch (floor 20; pre-stroke jaw-only = PS95 only).
- Grant: re-render ONCE after the caption wiring + grant memos (the 06:51 re-render was stopped).
- test_cohorts: test-side fix of 2 stale expectations (cohort2 per-animal reference dates) OK.

#### 2. On main since block 39
d8f4e900 memo keeps csig; b83f376d G1-G3; 2944eca4 te-basis hardening; 3f1c5e90 June supplement; 90a5e25f G4 +
--only g4; 10cdb08a share writers classified; f992cb8f redraw-pass fixes (lick-source label, grant stamp band + tick
thinning, matched-pre --lick-dir); dff97e56 interhemi RESTnl + raw. PUSHING: feat/trial-set-captions-jaw-panel-1109
(0899267d, via chain_v14), fix/test-cohorts-stale-refs-1109 then perf/grant-loader-memos-1109 (helper).
NOT yet: feat/trial-set-wiring-1109 (helper wiring the trial-set sentence into grant, section G, other epoch families,
cd chain, ITI / lick, encoding, interhemi + stale deck wording).

**UPDATED 11-10 ~21:00 -- READ FIRST (state; 16:00 entry below still holds the layout).** Combined model enc_rebuild_20261110g: ALL 15 TAGs collected; O2_COLLECT_g.done touched; SECTIONS=O2 lane (v20-O2) running since 20:52. Partial refits: frA + ls collected from O2 (PS92_0806 == desktop ~3e-15), .done written; te desktop check passed, submitted on O2 ~21:00 -> when 107 back: compare PS92_0806 vs scratchpad o2partial_check_te/desktop, `o2_partial.py collect --step t_partial_te_g --name partial_rebuild_20261109` (wf_o2partial, PYTHONPATH Windows form), touch logs/t_partial_te_g.done. o2_partial fix 0b6999fa (same_design tolerance: column sums vs sum|x|) is on O2 and M:. Glue checks: 14m far_L / 14z far_center clusters and the ITI in-trial relative decoder change = the glue mask (traced; running list 17:30). Re-run 131+ done, no failures today.

**UPDATED 11-10 ~16:00 -- READ FIRST (supersedes 06:20 for state; pre-compaction).**
RE-RUN v20 (main 8aed3a7b, detached wf_render_v20; logs scratchpad C--Users-SabatiniLab/f082ee0c-.../render_daqlk): 114 _g done, no real failures. Lanes: v20 (CD + MAPS sections finished; ENC x3, ITI, ITI+DOWN x3, DOWN x2, FIG) + side queues, all same step names: 20b (ITI extract first), 20c (firstlick after split_order, done), 20d (8 steps with corrected deps; c_fr_pix* wait O2_COLLECT_g), 20e (66 steps that were READY but never claimed -- lanes are strictly file-ordered: run the ready-but-unclaimed listing regularly), 20f (3 relabel steps RE-RUN: their 13:23 runs reused glue-in joint_fits.pkl + sessions/; moved aside, manifest move_aside_relabel_20261110_134738.csv), 20g (b_residual_roi_frA_g after b_specificity_g), 20h (b_fr_compare_g + a_relabel_noFR_g; e_mae / e_perspout / e_rmse_unsc + e_scatter_g). Stale audits: #1 resume audit (22:55, 10 items moved), #2 mechanical (all 102 finished clean; 2 more fixed, manifest move_aside_audit2_*), deps audit (2 gaps -> 20h). Monitor regex must be \[v20[a-z]?-. RAM: brief 0 GB dips (10:11, 15:12) with many lanes; nothing failed.
O2 COMBINED = enc_rebuild_20261110g (inputs 28 GB; env identical to 11-04 production; check passed; every copied session gate attempted / exec polar / licks combined). 11 TAGs complete (main, main_nolicksall, shufpos, nodir, drop, trmodels, musall_shuf, fr_A/B/S/D) -- COLLECT RUNNING (scratchpad 179e1046.../o2_collect_1110.log; ~20 min / TAG; env o2_env_combined.sh). Resubmitted by Priya (running/pending): bins_w1 8-10, bins_w05 77, bins_w2 69, droppos 4,5,7,8,27,30-34 @64G (droppos OOM'd at 48G). When those land: collect bins_w1 / bins_w05 / bins_w2 / droppos, then touch render_daqlk/logs/O2_COLLECT_g.done and launch SECTIONS=O2 (queue_daqlk20.sh; the v20d c_fr_pix* lanes also wait on it). enc_rebuild_20261109g (WRONG settings) out/ must be moved aside, never collected.
O2 PARTIAL REFITS (new code, branch feat/o2-partial-refit-1109 in wf_o2partial; bundle M:/.../rebuild_20261103/o2/partial_rebuild_20261109: imaging 15 GB + per step inputs 19 / 19 / 17 GB): b_partial_frA_g + a_partial_ls_g bundled + checked bit-exact (ls: 9875 values, 0.0) -- NOT YET SUBMITTED as of 15:50 (Priya transferring now; commands given). t_partial_te_g bundling (88/107 at 15:48, ETA ~16:45), then `o2_partial.py check --step t_partial_te_g --name partial_rebuild_20261109 --out <scratch> PS92_0806` with PYTHONPATH in WINDOWS form (C:/.../wf_o2partial;C:/.../scripts), then give Priya the rsync + 2 sbatch lines. The 3 steps are HELD by claim dirs (render_daqlk/logs/PARTIALS_HELD_FOR_O2.txt); after O2 + rsync back: `python scripts/o2_partial.py collect --step <step> --name partial_rebuild_20261109` (wf_o2partial), then write render_daqlk/logs/<step>.done.
OPEN with Priya: push a docs-only branch (this window's records + DECISIONS.md D1-D9; they live only on unpushed analysis/attempt-coding); caption draft questions (docs/caption-pass-draft-1110); PS95 CD overlap no longer significant (GLUE_BEFORE_AFTER_2026-11-10.md); push of feat/o2-partial-refit-1109. After the re-run: re-run scripts/glue_before_after.py, per-animal before/after to Priya BEFORE page / deck text; caption pass; LP lick decoder thread check; contactOnly arm.**

**UPDATED 11-10 ~06:20 -- READ FIRST (supersedes 00:05 for state).** Glue re-run v20: 63 _g steps done, no failures; CD (DAQ + LP) and MAPS sections COMPLETE (3 c_fr_pix* wait for O2_COLLECT_g by design). O2 COMBINED: the 11-10 00:00 submission (enc_rebuild_20261109g) had WRONG settings (bundle built without OROFACIAL_GATE=attempted / COMBINED_ENCODING_EXEC=polar / LICKS=combined / TONGUE=3d / EXEC3D / TARGET3D / BINS_CALIB) -> cancelled; its copied out/ (main 1, bins_w1 45) must be moved aside, never collected. CORRECT bundle enc_rebuild_20261110g: env.json identical to the 11-04 production bundle (21/21 keys, 4/4 tables sha1), check passed (PS94_0809), all copied sessions verified gate attempted / exec polar / licks combined (bins_w1 97, bins_w05 48 at 06:14; trials + alphas == production; ALM / tjM1 / MOp / SSp-m R2 unchanged, SSs / VIS moved, glue-only areas gone). Env for check / collect: session scratchpad o2_env_combined.sh. Collect each TAG when 107: `o2_encoding.py --profile cohort1_rebuild_20261103 collect --name enc_rebuild_20261110g --tag <TAG>` (from wf_render_v20 with that env), then touch render_daqlk/logs/O2_COLLECT_g.done + launch SECTIONS=O2. O2 PARTIAL REFITS: bundle partial_rebuild_20261109 (b_partial_frA_g; env 25/25 = queue step env; check bit-exact) -- commands given to Priya, no outputs yet; collect from wf_o2partial (`o2_partial.py collect --step b_partial_frA_g --name partial_rebuild_20261109`), then write render_daqlk/logs/b_partial_frA_g.done; add a_partial_ls_g / t_partial_te_g to the bundle when f_frA_ls_g / t_frame_te_g are done (te: check first). Side work done: GLUE_BEFORE_AFTER_2026-11-10.md (conclusion changes: CD overlap PS95 no longer significant; LP lick map clusters; VIS encoder stars -- sent to Priya), D2 copy-back (494 entries verified), caption draft (branch docs/caption-pass-draft-1110, 14 questions to Priya). OPEN with Priya: push a docs-only branch with this window's decision records + DECISIONS.md entries (they exist only on unpushed analysis/attempt-coding); caption questions; PS95 CD overlap. LESSON added: diff a new O2 bundle's env.json against the last production bundle before giving commands.**

**UPDATED 11-10 ~00:05 -- READ FIRST (supersedes 11-09 ~20:00 for state).** Glue re-run (queue v20, main 8aed3a7b, detached wf_render_v20) running clean: 26 _g steps done (CD DAQ + LP chains, epoch DAQ + LP-ENL, ITI split / z / main / hemo / audits, encoder frames f_noFR / f_frA / f_frAsplit / f_frA_pos / *_basis, b_wsd_lick). Side queues (same step names, corrected deps): queue_daqlk20b (ITI extract first), 20c (e_firstlick_extract_g after e_split_order_all_g), 20d (8 steps that read moved-aside files: c_fr_pix* wait for O2_COLLECT_g; spout patterns / where decoders wait for l_basis_extract_g; e_filter_cmp_g waits for the split-order + gauss steps). RESUME AUDIT 22:55: finished steps clean; 10 stale resume-by-existence outputs moved aside (manifests _superseded_glue_20261109/move_aside_partial_refit_* and move_aside_resume_audit*_*.csv). O2: (1) COMBINED model bundle enc_rebuild_20261109g (existing o2_encoding.py; _daqlk tag; mask_sig 5f9eb36f36b2) -- Priya SUBMITTED all 16 jobs 11-10 00:00; local check PS94_0809 running (needs env COMBINED_ENCODING_ENL_SOURCE=lp LICKS_ALL=split RUN=lp_r4_heatmap_20261013__clean_v36); when O2 finishes: rsync back, collect each TAG, touch render_daqlk/logs/O2_COLLECT_g.done, launch SECTIONS=O2. (2) PARTIAL REFITS on O2 (NEW code, Priya OK): branch feat/o2-partial-refit-1109 (wf_o2partial, scripts/o2_partial.py), validated bit-exact vs desktop; bundle rebuild_20261103/o2/partial_rebuild_20261109 packing 107 sessions; b_partial_frA_g / a_partial_ls_g / t_partial_te_g HELD by claim dirs (render_daqlk/logs/PARTIALS_HELD_FOR_O2.txt) -- after O2 collect, the coordinator creates their .done markers (or rmdir the claims to run on the desktop). C: cleanup: 93 GB stale imaging cache deleted (Priya OK; list in session scratchpad deleted_stale_imaging_cache_20261109.csv). LESSONS: producer check must include folder-listing readers; after clearing a failed step also clear its dependency-skip cascade; move-aside must cover resume-by-existence outputs, not only mask-dependent ones; monitors must match v20b/c/d lanes.**

**UPDATED 11-09 ~20:00 -- READ FIRST: GLUE RE-RUN RUNNING (queue v20).** Main = 8aed3a7b (integration of glue fix, mask-keyed caches, frozen spout D, trial-set wiring, grant memos, style leak / softmax / gate, precue ENL + stamp, test-isolation fixes; 6297 tests). Glue-in outputs MOVED (not deleted) to <root>/_superseded_glue_20261109/ (manifest scratchpad/glue_rerun/move_aside_manifest_20261109_171046.csv). Lanes run from DETACHED wf_render_v20 @ 8aed3a7b: scratchpad/render_daqlk/queue_daqlk20.sh (376 steps; plan glue_rerun/PLAN.md; ETA Wed 11-11 evening). Settings check passed (union 40,663 / stat 129,770 / PS92 exclusion 50,885; csig 3d7bbe921e). Lanes: CD (done), MAPS (g5_epoch_g), ENC x3, ITI, ITI+DOWN x3 (L7, L11, L12), DOWN x2, FIG. Incidents handled: (1) e_iti_split_sd_g listed sessions from the moved-aside folder -> one-step queue_daqlk20b.sh extracted 107 first, then split/z-score ran; (2) the failure cascaded 31 dependency-skips (claims held) -> cleared by literal-path glue_rerun/clear_cascade.sh, two fresh ITI+DOWN lanes. RAM guard scratchpad/ramguard_v20.ps1 (stops only the frAsplit encoder step if < 3 GB twice). Monitor must include 'SKIPPED (dependency'. MANUAL LATER: O2 re-bundle (new name) after the ITI inputs, then touch logs/O2_COLLECT_g.done + SECTIONS=O2 lane; state-space window re-runs state_space_sim after e_iti_main_g. After the re-run: per-animal before/after tables (balance etc.) to Priya BEFORE page / deck text; then the caption / page pass.**

**UPDATED 11-09 ~13:40 -- GLUE RE-RUN PREPARED, waiting on pushes.** Push queue chain_v20 (sequential): glue fix (+0d3c6f4d star-band fix for the state-space window's B40-B42 scripts, which failed main's star test), mask keys (fix/mask-keyed-caches-1109), frozen spout D, wiring (+2355e68c test fixes), grant memos, style leak / softmax / gate, precue ENL (+ mask stamp follow-up). Then: python scratchpad/glue_rerun/move_aside.py --do (HALT must be set; 96 items, 196 GB -> <root>/_superseded_glue_20261109/, never delete), detached worktree at the new main, put its SHA in queue_daqlk20.sh EXPECT_SHA, remove HALT, launch lanes (CD + MAPS + ITI + 3 ENC lanes together; PLAN.md lane layout). 2.5-3 days. Manual later: O2 re-bundle (new name) after the ITI inputs; FOR THE STATE-SPACE WINDOW: its caches are now mask-keyed (old ones moved aside); re-run state_space_sim after e_iti_main_g. Frozen spout: D = the single LP decoder (A/B deleted, kept list docs/status/DELETED_FROZEN_SPOUT_AB_2026-11-09.md); LP joint_xsession done (loads D). Balance primary stays the log2 gain ratio (per-lick baseline in the caption).**

**UPDATED 11-09 ~11:40 -- READ: REBUILD GLUE MASK MISSING (A9) -- Priya: fix shared code + audit, HOLD re-renders. Under the rebuild profile the painted glue mask is silently empty (stat mask 166,974 vs 129,770 px): SSs ~50 % glue, VIS / AUD / TEa / RSPagl_L affected; ALM / tjM1 / MO / SSp-m/n/un clean; LocaNMF decoders clean. Fix + cache-key + audit helper on fix/rebuild-glue-mask-1109 (wf_gluefix). HELD until merged: grant + epoch caption re-renders (approved), xa restnl re-run (fix e4b04637 on fix/xa-restnl-areas-1109), any map / area-extract / encoder re-render; v14a maps = preview only. Other helpers: precue lick-free ENL (feat/precue-enl-window-1109), style leak + 3 bugs + encoder-fig gate (fix/style-leak-misc-1109). Pushes queued in chain_v15 (stamp fix -> wiring -> memos). Open with Priya: balance primary A (raw c - i, no per-lick baseline) vs B (keep ratio); frozen spout decoder recs (D = the LP decoder, LP joint_xsession ~35 min, delete A/B copies). Details: RUNNING_LIST 10:25-11:40.**

**UPDATED 11-09 ~10:25: still no failures. The v14a map steps went silent after 108/108 ref-maps (09:30): MEASURED as CPU starvation (main thread 'Ready'; 3 random-split jobs x 47 threads on 24 cores) during the serial RESTnl baseline attach (restnl_mask.base_maps, 3-9 s per session even on a cache hit). Not restarted; split PIDs 2184/37244/26396 set BelowNormal (map CPU 0.03 -> 0.26 core). Caption push retrying (10:19 attempt in tests). Stamp fix ready: fix/test-stamp-order-1109 6bb2a103 -> push after captions, then memos (wf_grantmemo). Trial-set wiring helper still running (wf_tswire).**

**UPDATED 11-09 ~09:30: no new failures since 06:25 (all earlier ones fixed: offset_primary _r3, ENL-v table _r3 / balance + curves _r4, v12 relaunched). Map re-renders RUNNING from a DETACHED worktree wf_render_v14a at the reviewed caption commit 0899267d (queue_daqlk14a.sh: g5_restnl_maps_lp_ts + g5_lpfinal_maps_ts; reference maps 107/108 after 20 min -- warm caches; new period / jaw-only sets next). Caption-branch push (chain_v14) passed its tests but lost a push race, retrying; its lanes will skip the v14a steps (claims). Grant memos blocked only by the ORDER-DEPENDENT test test_grant_stamp_fit (helper fixing on fix/test-stamp-order-1109). Caption wiring for the remaining families: helper on feat/trial-set-wiring-1109. Random splits 3 lanes in parallel; interhemi RESTnl rest extract done 08:11. test_cohorts fix ON MAIN 98e6a353.**

#### 3. Render state (~07:50)
DONE overnight (all on fixed code): section G LP (3 h 14) + G4-only (34 min) + G4 figure; LP epoch arms; LP cd chain (Fig
9.1); matched-pre lick half + LP table; ITI chain; gain-bio; rest family; balance / ENL-slide / ENLv chain (4 missing
producers added: ITI spout patterns, balance _inside, ENL-v regress, notdock dock steps); June LP drift; grant LP 96/96
(old layout); encoding bar re-summaries (stars / order). 12 parked old lanes stopped (exact PIDs) + claims cleaned.
RUNNING: v11 (44 previously stuck steps; B5 random splits now in 3 parallel lanes, ETA ~09:30; L5R lick family),
v12 IH (interhemi RESTnl + raw), v14 MP + LF (LP restnl maps + LPfinal maps with option (c) + jaw panel, ~4-5 h each).
NEXT: grant re-render (~2-3 h with memos) after wiring; caption replots for figdata families; then the PAGE / DECK
UPDATE (headline table approved: 1-6, 8, #7 wording) -- ETA renders ~13:00-14:00, pages / decks this evening.

**Lessons (11-09):** (1) before launching a chain, check every input FILE has a producer (4 missing-producer failures);
(2) rename re-runs (_r2.._r5) when old lanes wait on the original names, and stop parked old lanes once audited;
(3) a code edit in a digested module moves cache keys -- verify csig / map digests after every such edit; (4) parallelize
independent per-animal steps with extra lanes on the same section (claims prevent duplicates).

### 00-COMPACT-39 (11-09 ~01:30) -- READ FIRST; supersedes 38 for STATE (38 section 1 RULES still apply verbatim)

**Where things are:** running list `docs/status/RUNNING_LIST_2026-11-06.md` (sections 11-08 ~13:20 onward, newest at the
end); scratchpad (session f082ee0c): `rerun_settings_check.md` (per-step settings audit of the re-run), `epoch_trim.md`
(page figure -> family / arm / lick dependence; page repointing list), `june_camera_tables.md`, render folder
`render_daqlk` (queues `queue_daqlk9zw|9zx|9zy|9zz|9zz2|9zz3|10.sh`, `logs/CHAIN.log`). Rules: block 38 section 1
(read the decision docs first; 9-item SETTINGS CHECK on every step's live log; no redundancy; never let a known-stale
step take the heavy lock) -- all followed in this pass.

#### 1. Decisions taken 11-08 / 11-09 (Priya)
- Lick class = every camera lick, LP source, LP onset, ENL 1.0 s, LP-trained frozen decoders; successful lick = DAQ
  contact, supplemental only. LP lanes use LICK_SOURCE=lp + GATE_SOURCE=lp + store frozen_models_lpfinal (as LPfinal).
- Trimmed re-run plan approved (memo push; full section G with memo; cue arm included; page-feeding lick decoders first).
- G1-G3 AND G4 (11-09): train camera WORKING (contact + incomplete) vs raw NO_RESPONSE; unknown never trained; scored
  no-lick set = working-period no_response rows; stopped / quit tail / jaw_only / unknown scored as their own groups.
  DAQ-grouping supplement NOT run. undetected_state_split now mostly UNDECIDABLE (by design; by_class answers it).
- June-vs-August drift (Figs 7.4/7.4b, 7.11/7.11b, 7.15, 7.20/7.20b, 7.24): build June camera trial tables as a SIDE
  table + opt-in supplement (registered trial_attempts table untouched; its mtime/size key every LP cache).
- G4 change: "only rerun what needs to be rerun" -> `poststroke_section_g --only g4` (replaces 3 G4 keys, backup first).

#### 2. On main (this pass)
9bc33b9b (D7 floor 10 only, no display floors, D6 matched-pre), 21a005ca (stars top band, POS_ORDER, G1-G3 camera),
597cbed9 + d8f4e900 (section G DAQ-cue memo, 6x, identical; memo kept OUT of the class digest: csig stays 3d7bbe921e),
b83f376d (G1-G3 no_response), 2944eca4 (te-basis path hardening), 3f1c5e90 (June supplement: trial_attempts --labels,
lp_licks WIDEFIELD_ATTEMPTS_SUPPLEMENT), G4 branch fix/g4-noresp-1109 (70d976f0 + 2b87c996 --only g4) being pushed by
`scratchpad/chain_v10.sh` (log chain_v10.log), which then launches queue v10.

**UPDATED 11-09 ~05:10: RE-RUN PASS COMPLETE -- every group done on fixed main (G4-only 34 min; June LP drift 33 min; grant LP 96/96 in 5 h 24 min; balance / ENL chain via v9zz2-v9zz5 incl. 4 added missing producers). LP lick class VERIFIED correct (trace: scratchpad/lp_epoch_decoder_trace.md; the "DAQ lick-arm trials" log text is a stale hard-coded label). Two near-identical LP lick epoch decoder sets (float-noise inputs) -> Priya decision A; grant memos measured 2.2x / later alignments 130 s, identical -> Priya decision B. Next: stop parked old lanes, redraw pass, then the replacement pass (headline table -> Priya). See running list section "11-09 ~05:10".**

#### 3. Render state (~01:30)
DONE on fixed code (live-log settings checked): section G on LP (3 h 14 min, was 13.1 h), LP epoch arms ENL / lick /
cue (5r 14m / acc 14m / scal mat), LP cd chain (Fig 9.1 -> LP epoch folder), matched-pre lick half (LP, 799 pairs),
ITI chain incl. 2 never-run producers (e_spout_patterns, e_spout_pat_stats), gain-bio, restnl masks (rebuild IDENTICAL
to disk -> 2 re-runs marked redundant), LP restnl maps, rest frozen (+nl), attempt maps restnl, recovery_cue, l_* restw
chain through restw_check, balance chain head + the _inside producers (v9zz3).
RUNNING: g6_grant_lp (LP; slow: 6/96 at 00:46 -- cold LP collection; measure, profile only if it stays slow),
v9zx-RS (chronic_stability_r, l_* restnl variants), v9zz2-BAL (59-step balance / ENL-slide chain, all names _r2, own
marker ENL_SLIDE_FIX_r2 so old lanes stay parked), v10 G4 (--only g4 -> G4 figure redraw) + JD (June LP drift).
OLD lanes still parked on pre-fix code (v8b / v9zd / v9zr / v9zc / v8-L4.. waiting on original names): harmless; stop by
exact PID once the _r2 chain is through.

#### 4. Next
1. When grant + BAL + RS finish: figdata redraws (stars / order), the matched-pre figure with the LP lick half, G4 deck
   slide, gb_fig_r redraw; later lick decoders on LP (pcd / nolick / precue states / joint xsession).
2. Replacement pass: headline table re-read on corrected outputs -> conclusion changes to Priya BEFORE any page / deck
   text -> repoint page figures (epoch_trim.md + rerun_settings_check.md lists: LPfinal / LP folders, successful-lick
   supplement labels, drops 2.2 / 7.1 / 7.5).
3. Priya's earlier asks: C3 / C4 careful look on corrected outputs + per-analysis spontaneous-lick grouping table;
   frozen spout decoder 3 versions (only if it changes a number); D2 local cache copy-back (delete needs Priya OK).

**Lessons (11-08 / 11-09):** (1) a speed-only edit to class code moved the cache digest -> keep memos out of
digested modules (hook mark / separate module) and verify csig after any trial_classes edit; (2) "missing producer"
failures (ITI spout patterns, balance _inside) -> when a step fails on a missing input, grep the queues for its
producer before re-running; (3) rename re-runs (_r2) when old lanes wait on the original names; (4) check that a
"stale" input really is stale (restnl masks were already post-fix) before re-running its readers.

### 00-COMPACT-38 (11-08 ~13:20) -- READ FIRST; supersedes 37 for state

**Where things are written:** running list `docs/status/RUNNING_LIST_2026-11-06.md`, section "11-08 ~13:20 --
CONSOLIDATED" (questions for Priya / findings / progress); audits in `S2/lick_source_audit.md`,
`S2/settings_audit_decisions.md`, `S2/settings_audit_actual.md`; headline table `docs/status/HEADLINE_TABLE_2026-11-08.md`
(PRE-correction: its lick-window rows are SUCCESSFUL-LICK numbers). Render: `S2/render_daqlk` (queues, `logs/CHAIN.log`).

#### 1. Rules Priya restated today (apply before ANY step, question or re-run)
- **Read the docs before asking Priya.** Several questions on 11-08 were already answered: ENL 1.0 s (10-21
  enl_lp_census), LP-trained frozen decoders (10-24, ENCODING_FINDINGS 3.3), LP onset not peak + 50 ms close-spout DAQ
  fallback (11-04 LP_MISSED_CONTACT), LP arm spec (docs/status/LP_FINAL_ARM_20261102.md), working / stopped / camera
  classes (11-03 TRIAL_GATES handoff), A9 / B section G decisions (running list 11-07). Grep DECISIONS.md, docs/status,
  wf_attempt docs/status and ENCODING_FINDINGS first.
- **SETTINGS CHECK for every render step (do it, never assume):** read the step's `# env` / `# cmd` / code-sha header and
  confirm (1) lick source: lick class = LP (`WIDEFIELD_LICK_SOURCE=lp`), successful-lick supplement = DAQ; (2) working =
  camera in_working (contact + incomplete), end at last camera lick, stop rule A2; (3) ENL 1.0 s lick-free on LP licks;
  (4) pre0801 (6 agreed exceptions); (5) hemo / lnmfproj / Wt / max_rt 3500 + 70 ms; (6) the code sha contains every
  number-changing fix (ba5b63bf rest engagement, D7 floors, D9 paths, a49795b3 / ee21ac5d / f069b949 / 825a4e0f camera
  fixes); (7) snapshot inputs were regenerated after any decision that changes them (cf. A20); (8) frozen store = the
  rebuild one (LPfinal: frozen_models_lpfinal); (9) cohort 1 only. A decision reaching the CODE is not enough: check it
  reached the DATA the render reads.
- **EFFICIENCY / NO REDUNDANCY (Priya 11-06 + 11-08; memory efficient-no-redo):** before re-running, list what already
  exists and is valid and re-run only what changed (LPfinal = pre-cue acc + 15r maps for precue / cue / lick; A20 identical; encoder / ITI / kinematics /
  LP-goal unaffected); re-run only the affected cells of a big step (section G: lick window, lick-only arm, G1-G3);
  profile the hot path for repeated loading (partial_refit rebuilt inputs 13x per session = 91 %, fixed 2.0x
  exact-equal; section G re-pools the animal per session unit -- profile running); reuse cached session features
  (LPfinal LP cache); never let a known-stale step take the heavy lock (stop its lane by exact PID, re-queue on fixed code).
- Standing: write only under M:/MICROSCOPE/Priya + repos; push only with Priya's DIRECT OK; merge origin/main, never
  rebase / force; lanes only from DETACHED worktrees; stop only exact PIDs I started (identify by start time or command
  line, never by parent); rmdir the empty claim dirs of stopped lanes (LITERAL paths: the rm safety check refuses
  variable paths); park dependents of parked steps; never edit a running queue file; measure before restart; captions
  state window / lick source / trial period / trial types / gate / baseline / correction; stars in the top band; ipsi on
  image left; canonical spout order; American spelling in new text.

#### 2. The lick-class error and the two classes
Main renders ran with the DAQ lick source (the profile never set WIDEFIELD_LICK_SOURCE;
`trial_classes.camera_select("lick")` = DAQ-engaged rows). Documented rule: **lick = every camera-seen lick (contact +
incomplete), LP source, aligned to the LP lick ONSET** (first visible cam4 frame; close-spout DAQ fallback if the LP onset
is > contact + 50 ms), ENL 1.0 s lick-free on LP licks, LP-trained frozen decoders; **successful lick = DAQ contact,
supplemental only** (Priya 11-08; 11-07 "never gate on successful licks"). So the DAQ-source "lick" outputs are the
successful-lick class (relabel, keep, never headline); `LPfinal` (scripts/lp_final_render.py) is the lick class for
acc / accdelta for the PRE-CUE window only (lp_final_render.py:260) + 15r / 15rpa maps for precue / cue / lick (corrected 11-08 ~13:50). cam1 is complete for all 4 animals (125 event files).

#### 3. State at ~13:20
- ON MAIN: 99f0d464 (D9 paths), 65cf7ead (rim overlay), e0e6d0ad (partial_refit input reuse).
- PUSHING (Priya OK): `fix/refit-floor-10-no-display-floor` (wf_mixmatch; b27a7acd + b29d834f on top of
  feature/refit-mixmatch-control = D6 matched-pre control, renamed refit_matched_pre): D7 refit floor 10 only, display
  floors removed; 915 tests passed; the first push lost a race with another window, re-push running (`S2/push7.sh`,
  WTS=wf_mixmatch). Check its output and note the sha.
- Helper running: `fix/stars-spoutorder-g13-1108` (wf_fixes1108 @ 68e53e33): stars to the top band, canonical spout
  order, section G G1-G3 per (f)1 + camera classes, DECISIONS.md correction + 11-08 entry. Review its commits and tests;
  Priya said "ok to push" for today's fixes -- confirm the scope covers this branch before pushing.
- Lanes: v9zv-PF holds the heavy lock (b_partial_frA, 96/107 at 13:10, ~13:25), then runs AB's 19 remaining steps
  (a76352f3 + reuse fix: CHECK each against the audits before it runs; the b_* encoder residual steps should be
  unaffected). Waiting for the lock, allowed (rest fix present): g5_epoch_15r_r (v9s), gb_sessions_r (v9r),
  l_stopped_restw_r + the rest of v9zs-BH (99f0d464), l_block_compare (v8, no rest dependence). t_frame_te_basis = v9zu-TB
  outside the lock, gate >= 20 GB free. Monitor re-armed every 30 min.
- Stopped today (all mine, exact PIDs): v9y 36768, [T] 34776, AB 8516 (+ b_partial_frA 4740 + workers), g6_grant_r4 34848
  (+ workers), v8-XL2L3 34044, v8s2-L2L3 3560. Their unfinished steps must be re-queued: the restnl chain
  (l_precue_restnl, l_precue_restnl_traces, l_c2l_extract_restnl, l_c2l_restnl, l_c2l_precue_restnl) and the l_bal_*
  balance chain -> l_recov_markers -> e_recov_meas / l_balmatch / e_dock_* / l_enl_offset -> ENL_SLIDE_FIX -> x_enlv_*.

#### 4. Re-run plan (measured step times from this render's logs; NOT started; Priya OK: two heavy slots)
| group | why | steps (measured) |
|---|---|---|
| section G -> figs -> grant | lick cells on LP; G1-G3 (f)1 | r4 13.1 h (-j 4); figs 15 min; grant ~1.8 h (Sept) -- **scope pending the profile** |
| epoch render, DAQ source | rest fix, D7, display floors; lick arm relabeled successful-lick | g5_epoch 8.9 h + 1b 1.7 h -- re-run only the families each fix touches |
| epoch families on LP not in LPfinal | lick class | TRIMMED 11-08 ~13:50 (running list): 3 single-arm LP renders on current main -- ENL `--only 5r 14m`, lick `--only acc 14m`, cue `--only scal mat` (pre reference); ~1.75-2.5 h wall at 3 at once; cuelick / ENLlick + other families not needed; g5_epoch_15r_r redundant |
| cd chain | lick-aligned + rest fix | 0.5 + 2.5 + 0.6 h |
| lick-aligned decoders / states | lick class | pcd 3.3, nolick 2.4, precue states 0.5, joint xsession 0.6, matched-pre lick 0.7 h -- mark page-feeding vs supporting (FIGURE_PROVENANCE: only cd_traj feeds a page fig, Fig 9.1) |
| rest family | ba5b63bf missing | restw_check 0.9, precue_restw 0.4, revup 0.4, c2l 0.4, rest_frozen(_nl) 1.9, RESTnl masks 0.4, attempt_maps_restnl 1.3, chronic_stability 1.0 h + the stopped v8 restnl / l_bal chains |
| gain-bio | stale | gb_sessions_r (queued, has the fix) + summarise + stats |
| redraws | stars, session-level figs | minutes, from figdata |

**UPDATED 11-08 ~17:45: Priya approved the trimmed plan; re-runs LAUNCHED on fixed main from detached worktrees (queues v9zw EP+IT, v9zx CD+RS+MP, v9zy IT2 done, v9zz SG = section G + grant on LP). State, lanes, what is still to queue and the one open question: running list section "11-08 ~17:45 -- state". Settings check per step: scratchpad/rerun_settings_check.md; epoch trim: scratchpad/epoch_trim.md.**
**UPDATED 11-08 ~15:00: measurements DONE -- the trimmed plan + 6 decisions for Priya are in the running list section "11-08 ~15:00 -- TRIMMED RE-RUN PLAN" (section G memo 6x, identical; epoch = 3 single-arm LP renders; v9s-P stopped; both fix branches ON MAIN 9bc33b9b + 21a005ca).** Original order: finish the two measurements (section G one-unit profile `S2/prof_g/prof_PS94_5.txt`; one LP epoch family) ->
trim the list (reuse / partial / full) -> show Priya -> queue from ONE detached worktree at the new main (after both fix
branches land) -> settings check on the first step of each group.

#### 5. Other open items
- Priya asked to look carefully at C4 (refit-minus-frozen gap, z 4.5 -> 1.75) and C3 (orofacial-specific chronic
  residual, +1.33 -> +0.97) AFTER the lick fix, including how the spontaneous-lick terms are grouped in each analysis
  (grouped with licks vs left out): build a per-analysis table for her.
- Frozen spout decoder stored 3x (11-05, 11-07 x2; different pre input sets): trace the cue / lick differences.
- 392 section G r4 entries exist only in the LOCAL session cache (C:/Users/SabatiniLab/wfield_cache/
  session_cache_rebuild_20261103): D2 copy-back to M:, verify, then delete the local copy only with a kept list +
  Priya's OK.
- Robustness rows (DAQ quit-tail, drift1500, stop-rule arms, contact-only section G) not yet rendered.
- D6 result (working half, valid): acute far -0.22 / -0.15 / -0.26 (Holm .0045); figure refit_matched_pre.png.
- Then the replacement pass: headline table re-read on corrected outputs -> conclusion changes to Priya -> pages / decks.

**Lessons (11-08):** (1) read the decision docs before asking; (2) verify each decision reached the data a render
reads; (3) queue audits must check lick source, gate source, code-sha fixes and snapshot dates, not only dependencies;
(4) never commit before reading the test output; (5) when the heavy lock is about to free, check which steps will take it.

### 00-COMPACT-37 (11-08 ~08:20, Priya awake) -- READ FIRST; supersedes 36 for state
Running list = `docs/status/RUNNING_LIST_2026-11-06.md` OVERNIGHT section (D1-D9 + log, newest last). Queue audit report =
`docs/status/QUEUE_AUDIT_2026-11-08.md`. Render dir `S2/render_daqlk`, log `logs/CHAIN.log`.
- **ON MAIN since 36**: cd6da055 (D4 lp_final_render frozen store in the rebuild root), 480accea (D5 Fig 3.4 post-stroke frozen
  recall from the frozen-mode sidecar + optional `session` column in _position_bars sidecars), 236dae4e (D8 lpfinal pre-session
  count), c3e01ed8 (ROI feature cache, 80090a57 + c3a8b049). **Pushing**: fix/d9-input-paths (76da97e5 stopped_trial_traces
  jaw_onsets / restnl_rescue precue_restnl_wr / simple_key_figures lick_correction; 143cee02 section_g_figures --suffix on
  every name; ae867505 run_mode.deficit_qc_dir(name) routes contact to the tagged encoding qc + imaging_lmm trend from there).
- **Section G r4** (v9zh, ROI cache): 65/88 at 07:51, ~7.8 min/unit -> ~11:00-11:30 (extrapolated); then g7_section_g_figs_r4,
  g6_grant_r4 (runs -j 4 WITHOUT the heavy lock: watch RAM). Then section C contact-only by hand (`after:g5_section_g_r4`), its
  figures with --suffix (needs the d9 branch on main). Then D2 (b): copy new local session_cache entries back to M:, verify,
  delete the local copy.
- **Fixed + re-run tonight**: g7_fig_state_position (v9zn), g7_fig_s_per_animal (v9zk: epoch_sessions_long was never queued),
  g5_lpfinal_dec (v9zl), g7_cd_geometry (v9zj: cd_trajectories --orth on + lick; NO ceiling -- cd_overlap_null never queued),
  l_reembed_r + m_reembed_plot_r (v9zi, D1), e_iti_hemo + e_iti_hemo_sum (after e_iti_area_maps), e_checkfigs_r,
  b_specificity_plain + b_residual_roi_r (v9zr).
- **Decisions (Priya 11-08)**: refit share floor 1/18 was an earlier window's choice, never discussed -> D7: ALL refit analyses
  move to a 10-trial floor only, with the D6 mix-matched control beside them (5r/5rm/5ro/5rmo + s1_refit_locanmf_table +
  june_august_drift re-run, single-arm timing first). 5rm (size-matched) is the refit-vs-frozen comparison. No class-balanced
  refit. D6 control: per post session, every August pre session of the same animal; BOTH subsampled to the post session's mix at
  the largest common size (symmetric; the one-sided first version was biased, 453/800 pairs skipped); refit within session; no
  floor. Lane v9zo (wf_render_v9zo @ 76e7d604, --inputs cache) running cue/working then cue/lick -> per-animal table first.
- **Lanes I made (all from DETACHED worktrees)**: v9zo (D6), v9zq (ITI svdbasis chain on 2f5799a9, replaces v9zg; waits for
  t_frame_te_basis, which another window re-runs with calibration), v9zr (batch A on main: e_mm_r / e_lmm_*_r / e_fpa_*_r wait
  for e_recov_meas / e_pat_extract). Batch B retries to queue after the d9 push. Batch C (producers to trace first):
  l_offset_where_r (svd / locanmf contrib terms), e_roi_foci_r2 / e_roi_cand_r2 (iti_spout_pattern_stats), e_rim_overlay_r
  (sessions_roi2), the ENLv chain (lick_enlv_extract --regress, enlv code family), l_key_figure_r2 (lick_same_target_place),
  l_bal_offset_primary (WFIELD_BAL_INSIDE), l_arrival_restnl (arrival_to_cue without --rest-variant), b_encoding_key_figures.
- **Parked RESERVED_HOLD** (logged in CHAIN): the audit's 12 + l_session_levels + dependents l_face_offset_partial,
  l_disengage_add (a lane waits FOREVER on a parked dep -- park dependents too).
- **Bottleneck**: the heavy lock is serial; b_partial_frA holds it since 06:26 (a_partial_ls took 8 h); behind it
  t_frame_te_basis, l_precue_restnl, g5_epoch_15r_r, gb_sessions_r, l_block_compare, l_stopped_restw (old code, will fail).
- **Lessons**: never edit a worktree a lane runs from (spawned workers re-import code); stopping a lane leaves EMPTY claim dirs
  that other lanes treat as taken -> rmdir them; check the full producer chain before queueing a consumer.

### 00-COMPACT-36 (11-08 ~00:05, Priya asleep) -- READ FIRST; supersedes 35 for state
Running list = `docs/status/RUNNING_LIST_2026-11-06.md`, section **OVERNIGHT 11-07 -> 11-08** (decisions first). Render dir
`S2/render_daqlk`, event log `logs/CHAIN.log`.
- **Section G = restart #4**, queue `v9zh`, worktree `wf_render_v9zh` @ **c3a8b049** (perf/roi-feature-cache), env token
  `WIDEFIELD_SESSION_CACHE=C:/Users/SabatiniLab/wfield_cache/session_cache_rebuild_20261103` (local robocopy of M: session_cache,
  7945 files 77.4 GB). Phase 1 done 23:52; phase 2 (88 units) from 23:52. Then g7_section_g_figs_r4, g6_grant_r4. NO further
  restarts (Priya). Restarts #2 (12/88) and #3 (4/88) were stopped: #3's local-cache premise was wrong -- the real cost was
  **Allen-ROI features never cached** (`trial_features_cached` refused non-LocaNMF sources). Fixed 80090a57: ROI kind keyed on
  (path, mtime, size) of U_atlas / atlas / mask / SVTcorr (`roi_signal_key`); measured PS92_0812 read 0.26-0.91 s vs 6-14 s,
  byte-identical. Old tests that asserted "roi uncached" now pin `svt` (c3a8b049). **Push pending**: waiter re-pushes
  `wf_roicache` after a_partial_ls ends (first try stopped by me for LOW RAM 4 GB, second failed on those two tests).
- **D2 (Priya: option b)**: after section G r4 + figs + grant finish -> robocopy new local entries back to M: session_cache
  (no overwrite), verify one ROI read from M: hits, THEN delete the local copy (guarded rm, keep list).
- **ON MAIN tonight**: 2f5799a9 (ITI TE-basis path never _spont + extraction exits 1 on failed items + iti_attempt_parallel
  fixture fix), fa0e4d87 (D1: June ref epoch None -> '' ; loader tolerates it), cc6d11df (D3: --fit-from copies spatial_*.pkl).
- **Retry lanes waiting**: `v9zg` (wf_render_v9zg @ 2f5799a9) waits for **t_frame_te_basis** (heavy slot) -> l_basis_extract_r
  (svdbasis half only), l_basis_frame_sd_r, e_basis_r, l_offset_where_r, l_session_levels_z_r, l_levels_compare_r,
  e_pat_stats_maps_r, e_roi_check_r2, e_roi_foci_r2 (parked RESERVED_HOLD: l_session_levels_z, l_levels_compare; the v9u copy
  of e_pat_stats_maps will fail harmlessly). `v9zi` (wf_render_v9zi @ 64d54bc3 = D1 commit) -> l_reembed_r, m_reembed_plot_r
  behind the heavy lock.
- Pixel-map fixes 11-07: spatial basis hand-copied (cmp) into frAsplit_basis_spont, basis_pixmusall, frA_basis_pixmusall_union;
  all V/PX steps done.
- A12b (pre-cue by camera class) reported: jaw-only above own-pre working at far contra acutely 4/4 (pooled 3.30), but also far
  ipsi and pre-stroke (PS95) -> trial-state effect; supplementary panel proposed, awaiting Priya. Files `S2/a12b_classes/`.
- Lesson recorded in memory (`measure-before-restart`): profile + time a fix before proposing to stop a long step.

### 00-COMPACT-35 (11-07 ~15:15) -- READ FIRST; supersedes 34 for state (34 / 33 hold the A7 machinery + remaining-work list)

**Live lists:** docs/status/RUNNING_LIST_2026-11-06.md (A9-A13 = today's decisions; C = progress + "code changed after run" table).
American spelling in all new text (Priya, relayed by a8 11-07).

**On main (all Priya-approved pushes today):** d12dfc56 + eeb36d48 contact-only selection + section G preserved positions counted on
camera WORKING rows (never on successful/DAQ licks) -- 4353fbd2 retained above-chance figure (section_g_retained) -- 32968a57 LOSO
figure skips no-video session -- 25a049d7 recovery fit flags (half-recovery RELATIVE TO PRE + completeness; flag ok/bound/unstable/
not_reached; MIN_HITS = 20 for hit-gated values; AIM primary, hit rate secondary; epoch-colour figures) -- 63229f3a Fig 10.2 text
(both axes lost on cued licks acutely; later cued sagittal loss = short licks missing the spout) + DECISIONS: UNSCALED where-decoder
primary, z-scored supplementary only -- becc486b csig fix (the contact-only branch in trial_classes.sel_classes hook-marked; class-code
digest back to 0fa0c1a2b432 = the render's, so camera-gated caches hit again) -- figures/epoch-convention (A13: shared cat_points
helpers day_dots / epoch_mean_sem / MODEL_COLORS; ~20 producers to open session markers, shape = animal, colour = epoch, no
cross-epoch lines, mean +- SEM epoch-coloured dot; model-comparison figures KEEP model-coloured lines in a palette distinct from
epochs; Priya OK'd 4 layout changes) -- push in flight at 15:13 (scratchpad/push15.out).

**Render state 15:13:** main HEAVY = v_frAsplit_spont_new (since 14:19), HEAVY3 = l_basis_extract (since 10:38, ~2.3 GB, slow multi-
pass). SECTION G restart #2 = g5_section_g_r2 in lane v9zd3 (queue_daqlk9zd3.sh = copy with poststroke_section_g removed from
HEAVY_RE for that lane only; runs as a 4th job; worktree wf_render_v9zd @ becc486b); phase 1 at 9/12 pools; phase 2 cache reuse
not yet measured. Then g7_section_g_figs_r2 + g6_grant_r2 (same lane); section C (contact-only supplement) launch LAST by hand:
SECTIONS="C" bash queue_daqlk9zd.sh. g6_grant / g7_section_g_figs (v7) stay RESERVED_HOLD (replaced by _r2). Lane R (v9zd): recovery
re-run d_decoder_retention_r2 / d_recovery_timing_r2 / a_recov_timing_aim_r2 DONE; l_key_figure_r2 waits on e_recov_meas
(l_key_figure parked RESERVED_HOLD). v9za: g5_joint_xsession_r DONE (all 6 jsons). v9zb: unscaled recovery-link angle / lateral done,
aim in progress (Fig 16.3-16.5). Stopped today (exact PIDs, logged in CHAIN): g5_section_g 11:26 (DAQ-lick positions), g5_section_g_r
14:19 (cold, csig). Failures: g5_map_drift_lp_r (June not in LP classes, deferred with A4).

**Open with Priya:** A10 done (text on main). A11: 4c wording after checks (aim-based; PS95 hit fast / aim ~4 d; PS94 aim plateau
-16 deg, completeness 0.55); A12: main fig = working trials; NEW supplementary pre-cue split by class (contact / incomplete /
JAW-ONLY / NO RESPONSE / stopped separately; Priya: "interesting if pre-cue increased on jaw-only") -- RUNNING (subagent,
scratchpad/a12b_classes); A13 branch pushing. Queued after the render: cued-sagittal attribution test (old vs new inputs one change
at a time: hemo / swapmask / lnmfproj / pre0801).

**Next:** watch section G phase 2 (cache hits, rate) -> estimate; push15 result -> plot-only re-render of the converted page figures
from rebuild outputs; A12b result to Priya; replacement pass (provenance check of every page/deck output's code SHA vs the
code-changed table first).

### 00-COMPACT-34 (11-07 ~10:15) -- READ FIRST; supersedes 33 for state (33 holds the remaining-work estimate + Priya's A1-A7)

**Live lists:** docs/status/RUNNING_LIST_2026-11-06.md. All decisions A1-A8 ANSWERED and acted on; nothing open with Priya.

**On main now (newest last):** ec0ff642 (seven 11-06 branches) · 8d6da32d kin cache routed · ba5b63bf rest boundary · 0e2154f0 jaw
xcheck routed · 4ecf72ed 15r empty-session · 72729126 LP no-video skip (A5) · 96f77156 pooled 3-D gate (A3) · 7388c713
stopped-trial traces on camera classes (A6/A8: success / incomplete / jaw_only / no_response inside the working period + stopped
as a 5th trace; DAQ path unchanged).

**Render machinery since the A7 restart (09:23-09:30):** lanes run from scratchpad/render_daqlk/queue_daqlk{7b,8,8b,8c,8d,8f,9r,9s,
9u,9w}_a7.sh with A7MIN=35 (a7_try: main HEAVY.lock, or HEAVY3.lock while the main holder is in LOWRAM_HOLDERS="g5_section_g" and
>= 35 GB available; XL.lock so partial refits never overlap; a_partial_ls / b_partial_frA now --jobs 4); slot-2 lanes from the
ORIGINAL queue_daqlk8_s2.sh with S2MIN=35. Original queue files untouched; pre-restart lane list lanes_before_A7.txt. Extra queues:
v9w (LP restnl / lpfinal / map drift, main 72729126), v9x (k_pooled_r DONE 10:01, main 96f77156), v9y (l_stopped_restw ->
l_stopped_restnl, A7 rules, main 7388c713, waiting for a slot), v9u (decoder-pattern stats + 4 ROI steps, waits t_frame_te_basis).
To relaunch a lane: A7MIN=35 SECTIONS="..." nohup bash queue_daqlk<q>_a7.sh > logs/queue_<q>_<S>_A7.out 2>&1 &.

**State 10:15:** main slot = g5_section_g (session 51 / 88, ~5 / h -> ~17:00). HEAVY3 / HEAVY2 free: available RAM 29-31 GB
< 35 while the 17 GB g5_restnl_maps_lp_r (v9w) runs beside section G; when it ends, HEAVY3 admits the next BIG step (partial
refit at 4 workers, t_frame_te_basis, g5_epoch_15r_r, gb_sessions_r, l_enl_angle ...).

**C: disk (Priya OK 11-07):** deleted wfield_cache/imaging + behaviour (49 GB, cohort-1 non-rebuild input caches, unused by the
render) and 6 old scratch folders (11 GB); list docs/status/rebuild_20261103/DELETED_LOCAL_20261107.txt (ae33e838). a8 cleared ~9 GB
of its own + 7 old merged worktrees (kept wf_trendfix, wf_attempt). C: 98 GB free. input_cache_rebuild_20261104 (299 GB) is LIVE
until the render + catch-up + replacement pass (~Mon). Survey of the rest of C: running (bg task b0j2irdi5) -> bring Priya a
candidate list with sizes and owners BEFORE deleting anything. After the render: remove my merged worktrees (wf_p1106_*, wf_deffix,
wf_svg, wf_kincache, wf_reststart, wf_15rfix, wf_lpnovideo, wf_pooledkin, wf_stoppedcam, wf_render_v9*).

**Next actions:** watch HEAVY3 admission once v9w's map step ends (first time two big steps run together -- check RAM log, the
step logs' "min available RAM" lines); re-arm the monitor every 30 min; update the running list as steps land; when section G
ends, the main slot takes the next big step and HEAVY3 stops admitting (main holder no longer low-RAM) unless >= 35 GB with
the XL rule; then the final catch-up + replacement pass (00-COMPACT-33 list).

### 00-COMPACT-33 (11-07 ~09:00) -- READ FIRST; supersedes 32 for state (32 / 31 / 30 hold for mechanics + tables)

**Live lists:** docs/status/RUNNING_LIST_2026-11-06.md (A decisions / B findings / C running; newest at the end of each section).
Update blocks below this one (11-06 15:35 .. 11-07 02:10) hold the detail of each fix.

**Priya's answers 11-07 ~09:00:** A1 section-6 changes -- she asked WHAT they are (answered: per-trial classes contact /
incomplete / jaw-only / no response, shares per spout x epoch; no trial changed class, only WHICH trials count -- working period
vs the old DAQ engaged cut). A2 -- the close-spout incomplete rise = post-cue licks in trials after the old DAQ quit that miss the
spout; under the working definition they ARE working trials and are now correctly included (wording: late-session misses, not
an aiming deficit). A3 YES (pooled 3-D table gate gap). A4 OK (e_drift_june deferred to after the render). A5 YES (push LP
no-video fix). A6 (a) wire stopped_trial_traces to trial_classes. A7 YES (second slot may take BIG steps beside a low-RAM main
holder at >= 30 GB available; coordinated restart).

**Code on main (all 11-06 / 11-07 fixes):** ec0ff642 seven 11-06 branches; 8d6da32d kinematics cache routed; ba5b63bf camera-gate
REST boundary (8 sessions); 0e2154f0 jaw cross-check routed; 4ecf72ed 15r empty-session tuple; LP no-video skip 72729126 (push
running 09:00, task bh9lq0u7y). In flight (helpers, branches, NOT pushed): fix/pooled-kin-gate-20261107 (A3, worktree
wf_pooledkin), fix/stopped-traces-camera-20261107 (A6, worktree wf_stoppedcam).

**Render state 09:00:** main slot = g5_section_g since 11-06 20:49 (session 46 / 88 of its main pass, ~5 / h -> ~17:00); slot 2
idle (everything ready is claimed by slot-1-only lanes). Done since 11-06 15:00: all A20 / A27 / A30 kinematics (v9k, v9k2, v9r-K),
rest attempt maps re-run (v9r-R), contact + ENL-1s restnl maps + map drift (v9s), decoder retention + recovery timing + rlink
(v9t), aim + recovery --aim (v9v), split-order chain, e_angle_reach, e_where_z, e_first_lick, l_cued_ramp_extract, b_wsd_lick,
b_within_spout_mixed, l_c2l_restw, l_restw_check, l_revup_restnl.
Waiting on the MAIN slot: l_enl_angle (lick-angle decoder), t_frame_te_basis (-> v9u e_pat_stats_maps + 4 ROI steps),
g5_epoch_15r_r (v9s), gb_sessions_r (v9r-B, slide 102), a_partial_ls, b_partial_frA, t_frame_frlag, b_pix_union_frA,
b_svf_frA_roi, g5_epoch_1b_v8, v_frAsplit_spont_new. Parked (HOLD): 10 mixed-model / goal-summary steps (catch-up),
g5_lpfinal_maps (-> A5 fix), l_stopped_restw/_restnl (-> A6), e_rlink_hit (superseded by _r).

**Remaining work (estimate 09:00):** ~9 BIG steps serial ~40 h + section G ~8 h; with A7 (two big at once) ~25-30 h. ~30 medium
(slot 2, ~10-15 h, parallel) and ~150 light figure / table steps (minutes each, gated by the heavy ones). Then the final catch-up
(10 HOLD steps, A25 LP epoch arm ~6-9 h heavy, LP map re-runs after A5, A3 / A6 re-runs) -> render + catch-up done ~Sun 11-08
evening with A7 (~Mon without). Then the replacement pass: provenance merge (CONFIRMED rows, data_block7), 69 caption conflicts,
OROFACIAL_FINDINGS A27 numbers, kinematics section 6 text, recovery-timing text, REST-column numbers (8 sessions back), slide 102,
decks (a8) -- about a day.

**Update 09:35:** A5 on main 72729126 (v9w), A3 on main 96f77156 (v9x k_pooled_r), A7 RESTART DONE 09:23-09:30: lanes now run
from scratchpad/render_daqlk/queue_daqlk{7b,8,8b,8c,8d,8f,9r,9s,9u,9w}_a7.sh (A7 block: a7_try -> HEAVY.lock, or HEAVY3.lock when
the main holder is in LOWRAM_HOLDERS="g5_section_g" and >= A7MIN GB; XL.lock for XL_STEPS partial refits; heavy_release frees
both) with A7MIN=35; slot-2 lanes from queue_daqlk8_s2.sh with S2MIN=35. Pre-restart lane list: lanes_before_A7.txt. The
original queue files are untouched. A6 fix ready (e79afbf9) -- push + meaning-change question to Priya (running list A8).
**Next actions (in order):** (1) A5 push lands -> queue: g5_restnl_maps_lp, g5_lpfinal_maps, g5_map_drift_lp re-run on the new
main; (2) A7 coordinated restart (HALT -> waiting lanes return claims -> relaunch from copies with BIG allowed in slot 2 when the
main holder is low-RAM and >= 30 GB available); (3) A3 / A6 helpers report -> review, ask Priya to push, queue their re-runs;
(4) keep the running list + this block current; re-arm the monitor every 30 min.

### 00-COMPACT-32 (11-06 ~15:15) -- READ FIRST; supersedes 31 for state (31 / 30 hold for mechanics + tables)

**Organisation check (Priya again 15:10: "not layering changes onto in-flight analyses").** State:
1. **Running lanes are pinned** (table in 31): v7b a76352f3, v8/v8b/v8c/v8e 5db29b07, v8d d2d26e84, v8e2 6033cbbb,
   v8f/v8g 110c4efd. Nothing merges into those worktrees while lanes run.
2. **Parked (RESERVED_HOLD, 10 steps):** b_imaging_lmm, b_remap_lmm, b_remap_checks, e_mm, e_lmm_response, e_lmm_all
   (deficit-path fix), d_fig_goal_summary + _f1/_f2/_f3 (computed titles). The three exec_vs steps were RELEASED (A23 not a bug).
3. **Six branches going to main** (Priya "yes"; scratchpad push6.sh, sequential, merge origin/main first; at 15:02 the first,
   priya1106-small, was in pre-push tests): small, frdrift, gainbio, gates-captions, deficit-qc-dir, svg. They do NOT touch
   the running lanes (pinned). Gate-dependent outputs (kinematics / cam4) route to gate-named folders, so new and old cannot
   mix on the share.
4. **In flight, NOT yet anywhere:** fix/priya1106-cam4-style (helper; A27 whole cam4 cohort -> camera working period,
   A28 read-only calibration comparison, A29 aim producer step, A30 style). Needs Priya's OK to push.
5. **ONE final catch-up queue** (build after 3 + 4 land; code = that main): the 10 parked steps, kinematics section K,
   cam4 cohort re-run + its figures, aim producer, recovery_timing / decoder_retention / fig_goal_summary (gates D lines),
   A25 LP epoch arm (`--block epoch_maps`, same g5 family list; heavy 6-9 h). SVG code included.
6. **Redraw-67 for SVGs (Priya via a8): DO IT SAFELY** -- main's drawing code differs from the render code for some
   figures (computed titles, max-statistic captions, kinematics gate, gainbio). Redraw into SCRATCH first, compare each PNG's
   sha256 with the share's; copy back ONLY byte-identical PNGs (+ their SVG). Any figure whose PNG differs is NOT copied
   (its drawing changed) -> it is redrawn in the catch-up queue with matching data; list it for a8 with its deck slides.
7. **Rule still in force:** new code -> branch; park affected not-started steps BEFORE they run; one catch-up queue on one
   final code version; never edit a running queue file or render worktree.

**Update ~15:35:** cam4-style helper DONE: branch fix/priya1106-cam4-style 2196c27f + main merged e34349d7 (wf_p1106_cam4),
tests pass, not pushed. A28 calibration identical (no decision). Step-line draft (K cohort + A29 aim):
scratchpad/a27_a29_cam4_steps_draft.sh -- in the catch-up queue give b_imaging_lmm + l_disengage* a dep on k_cohort; kinematics
page Figs 1.1-1.3 / 2.x source paths move to cohort/<tag>/prepost_working after the re-run. Asked Priya for push OK.
push6 at 15:23: branch 1 (small) in full-suite pre-push (configs/ changed). Render: all lanes wait on heavy lock held by v6
t_partial_te (105/107) + x_enl_move (last chronic sessions).

**Update ~15:45:** push6 FAILED at branch 1 (full suite 28 min: 4 failed / 5163 passed; nothing pushed). Three were already
failing on main (frozen_encoder --roi-targets unsorted fan_out res; pin_blas test reading names moved to parallel._BLAS_VARS;
hooks test rejecting this box's absolute core.hooksPath), one from A5 (test_map_cache_key: precue_engagement_states section
unclassified -> OUTSIDE_MAPS, map builders only import engagement_gate). Fixed on the small branch 9d740bf2 (32 tests pass).
Priya said YES to cam4-style -> scratchpad push7.sh = the six + wf_p1106_cam4, sequential, merge-only (bg task bn32updgj;
per-branch push logs scratchpad/push7.<wt>.log). Each push re-runs the full suite (~28 min) when configs/ is in the diff.

**Update 11-07 ~02:10 (overnight; live lists in RUNNING_LIST_2026-11-06.md).** Extra queues run on main 4ecf72ed (worktree
wf_render_v9s): v9t (decoder retention + recovery_timing lick + e_rlink_* re-runs) DONE; v9v (A29 aim + recovery_timing --aim)
DONE; v9u (iti_decoder_pattern_stats -- was in NO queue -- then e_roi_check/foci/cand/rim_overlay re-runs) waits on
t_frame_te_basis. LP no-video fix ready on fix/lp-novideo-skip-20261106 (72729126, NOT pushed, needs Priya): blocks
g5_restnl_maps_lp (failed), g5_lpfinal_maps (HOLD), g5_map_drift_lp (skipped) and the A25 LP epoch arm. e_drift_june fails (June
object arrays; superseded sensitivity; asked Priya to defer). LESSON: an ONLY= lane bypasses FILE ORDER, which hides undeclared
producer deps (l_restnl_rescue ran before l_precue_restnl -> failed fast; markers moved *_early_0206 so its own lane re-runs it in
order). Only put steps in ONLY= lanes whose producers are already DONE, not just their declared deps. restnl_maps.py uses ~17 GB
(single process) -- treat as heavy.

**Update 20:45 -- Priya: "parallelize as much as safe, proceed through all pending tasks, keep a running list" ->
docs/status/RUNNING_LIST_2026-11-06.md (A decisions / B findings / C running).** 15r fix on main 4ecf72ed; queue v9s
(wf_render_v9s @ 4ecf72ed): lane P = g5_epoch_15r_r (heavy, waits main lock) ; lane P (ONLY) = the 7 ex-HOLD light map steps.
Unblocking lanes added: v8 SECTIONS=E (took e_basis, waits l_basis_extract -- harmless), v8 SECTIONS=E ONLY=e_split_order_* (->
v7b lane E -> e_angle_reach -> lick-angle chain), v8_s2 ONLY=l_cued_ramp_extract (slot 2, running 20:39 -> l_enl_angle).
v9k2 + v9r K lanes DONE: all A20 / A27 / A30 kinematics outputs present. Helper comparing new 3-D kinematics vs the page ->
scratchpad/kin3d_compare.md.

**Update ~20:45 -- 15r TypeError diagnosed (separate from the rest bug).** position_reference_maps.session_raw_maps:482 returns
a legacy 6-tuple for a session with no trials in the arm; the session cache stores it; maps_by_epoch (line ~801) reads it as a
dict -> TypeError kills the ENL + cue arms of 15r (g5_epoch caught it, exit 0; only lick/lick written). Only session hit:
PS93_0813 (no behaviour video -> zero camera working trials) in prm-cue-working + prm-precue-working. Main 0e2154f0 still has it.
Fix: branch fix/15r-typeerror-20261106 (worktree wf_15rfix, 4ecf72ed, from 0e2154f0): _as_parts() normalises the empty tuple
(raises on non-empty), called in maps_by_epoch + reference_maps; session_raw_maps untouched so the prm cache key is unchanged
(2950853f103b); tests/test_prm_empty_session.py (7) + map_cache_key + reference_registry: 41 pass. NOT pushed (Priya OK).
Without it the parked restnl_maps / lp_final_render maps / map_drift_prepre steps would exit non-zero. After push: re-run
g5_epoch --only 15r (warm prm cache) + the 7 parked steps, on code with ba5b63bf + 4ecf72ed.

**Update 20:20 -- Priya "yes": both fixes on main 0e2154f0 (rest boundary ba5b63bf + jaw-xcheck routing). Queue v9r RUNNING**
(scratchpad/render_daqlk/queue_daqlk9r.sh, worktree wf_render_v9r @ 0e2154f0, lanes R / B / K): R = move the 12 rest-variant
attempt_maps npz aside (done, MOVED.txt) -> d_attempt_maps_rest_r -> d_attempt_maps_plot_r; B = move the 8 gain_bio session npz
aside (done) -> gb_sessions_r -> gb_summarise_r -> gb_stats_r -> gb_fig_r (slide 102); K = k_jaw_xcheck (v9k2 failure moved to
*_v9k2_fence) + k_pose_validation. Helper move script scratchpad/render_daqlk/move_aside_restbug.py. STILL PARKED until the 15r
TypeError is understood (helper diagnosing; fix branch would be fix/15r-typeerror-20261106): the g5_epoch REST column re-run
(15r / 15rpa families), g5_restnl_maps_*, g5_lpfinal_maps, g5_map_drift_*; then d_fig_goal_summary_f3 in the final catch-up.

**Update ~20:00 -- rest-bug impact (audit: rebuild_20261103/ENGAGED_MASK_IMPACT_20261106.md).** The 8 sessions were DROPPED from
REST-referenced outputs ("no engaged rest frames"), not given a short baseline; gain_bio kept them with empty R5 / R5far / TASK.
17 steps: DONE affected = g5_epoch (REST column only; 15r also hit an UNRELATED TypeError in the ENL + cue arms -> only
lick/lick RESTref written), d_attempt_maps_rest, d_attempt_maps_plot, gb_sessions, gb_summarise; failed gb_stats; skipped gb_fig.
PARKED on RESERVED_HOLD (were RESERVED_v7, live G lane): g5_restnl_maps_{contact,lp,enl1s} (only the REST rows of the amplitude
CSV), g5_lpfinal_maps, g5_map_drift_{contact,lp,enl1s}. Held downstream already: d_fig_goal_summary_f3 (+ aggregate).
NOT affected: restw / RESTW / MEAN / PRECUE / RESTnl columns, cd_traj (contrast ref), restnl masks, chronic_stability (PRECUE).
RE-RUN NEEDS cache moves: attempt_maps skips existing rest-variant maps_<align>_<animal>.npz (move all 12 aside); gain_bio
returns cached sessions/<label>.npz (VERSION 2) -> move the 8 aside; map_collect recomputes itself. Waiting on Priya: push OK for
fix/camera-rest-start + fix/jaw-xcheck-routed, and re-run plan (catch-up queue on the fixed main).

**Update ~19:40 -- CAMERA-GATE REST BUG (conclusion-level; Priya told, nothing on pages changed).** v8g gb_stats EXIT 1
(assert oro - post == oro_post) because 8 / 107 sessions have NaN R5 gain: rest_engagement.engaged_frame_mask put the quit
boundary at the FIRST non-engaged cue; under the camera gate the working period is a span (first .. last camera lick), so a
session whose first cue is not a camera lick got quit = cue 0 -> only its pre-first-cue frames (~190-800) were "engaged rest".
Sessions: PS92_0813 (pre), PS92_0822, PS93_0818, PS94_0817, PS94_0820, PS95_0818 (acute), PS93_0822, PS93_0824 (subacute).
engaged_by_cue / rest_by_position / restw NOT affected. Also reaches: position_reference_maps time-local rest (_RESTref_),
cd_trajectories rest baseline, restnl_mask measure, gain_bio R5. Fix: branch fix/camera-rest-start-20261106 (worktree
wf_reststart, ba5b63bf, from main 8d6da32d): frame_gate_bounds -> quit after LAST engaged cue, camera start at the span's
first cue; unaffected sessions identical (PS92_0812 verified; PS92_0813 now 287,423 / 288,213 engaged). 99 tests pass. NOT
pushed. Impact audit helper running -> scratchpad/engaged_mask_impact.md (affected steps by status). gb_fig skipped.

**Update ~19:10:** v9k2 k_jaw_xcheck EXIT 1 = RebuildFenceError (pose3d_jaw_crosscheck wrote session_poses/pose3d/crosscheck
directly; its cam4 side is the gate-named A27 jaw table, so the output is gate-dependent). Fix on branch
fix/jaw-xcheck-routed-20261106 (wf_kincache, 18875631, on top of 8d6da32d): writer + fig_pose_validation reader via
kin_out_dir(..., "crosscheck", script="pose3d_jaw_crosscheck"); 310 tests pass. NOT pushed (needs Priya OK). k_pose_validation
-> RESERVED_HOLD (would read the old jaw_agreement). Checked: the 11-04 pose3d edits (bddf02e7 pnp / session_3d, ae2bcd86 source
switch) do NOT touch lick_events / kinematics, and every *_frames.csv.gz is dated 10-20..10-23, so the recomputed kinematics
cache equals the 10-23 snapshot tables (no mixed provenance). Every other v9k2 script already writes via kin_out_dir / --out.
After the push: re-run k_jaw_xcheck + k_pose_validation (small queue on the new main).

**Update 18:55 -- Priya "ok": kinematics cache fix on main 8d6da32d; queue v9k2 RUNNING** (scratchpad/render_daqlk/queue_daqlk9k2.sh,
worktree wf_render_v9k2 @ 8d6da32d, lanes K + C): k_spout_error + k_phase (v9k markers/logs moved to *_v9k_fence) and the 7
ex-HOLD steps (RESERVED_v9k2). Priya's "ok" also read as YES to updating OROFACIAL_FINDINGS with the A27 working-period numbers
in the replacement pass (responded-trials row relabel + far-spout range, far-contra first contact +978 ms, pre-cue -0.41 Hz).

**Update ~18:25 -- v9k 3-D steps blocked by a stale, unrouted kinematics cache.** k_spout_error EXIT 1 (and k_phase failing the
same way): pose3d_kinematics.session_result cached in session_poses/pose3d/_cache/kinematics_<mode> (built 10-26); pose3d.py
changed 11-04 (bddf02e7, ae2bcd86) -> code_version miss -> recompute -> write to the unrouted folder refused by the rebuild
fence. Fix on branch fix/kin-cache-routed-20261106 (worktree wf_kincache, 7df3fd21, from main ec0ff642): cache via
trial_classes.kin_cache_dir (+ tests/test_kin_cache_routed.py; 22 pass). NOT pushed (a8 freeze + needs Priya OK). Parked on
RESERVED_HOLD: k_phase_contra k_summary_fig k_vs_cam4 k_pose_validation k_pooled k_lmm_curves k_jaw_xcheck. After the push:
v9k2 queue on the new main = k_spout_error, k_phase (+ those 7). The first run recomputes all 119 sessions' kinematics (cost
unmeasured). Also unrouted: scripts/jaw_onsets.py cache (imaging_encoding/qc/_cache/jaw_onsets) -- hit today, would fail if stale.
A27 cam4 result (k_cohort, 18:15): no headline change; 13 borderline cells; findings-number updates await Priya (overnight page).

**PUSH FREEZE LIFTED ~18:40 (a8: decks branch landed, main = 3e543de5; decks now under presentations/pipeline/<kind>, PathResolver.deck_dir). Was:** no pushes to widefield main until a8 reports feat/decks-to-presentations-20261106 landed (~2 h full suite). Message a8 first if a push is urgent.

**Update 16:57 (real clock; the ~16:10 / ~16:40 labels above were ~30-40 min early) -- ALL SEVEN 11-06 BRANCHES ON MAIN
(ec0ff642): small 25832aad, frdrift 53cd46e5, gainbio ee140449, gates cd37f143, deffix 1bac0341, svg 6d44e774, cam4 ec0ff642.**
Fixes needed on the way: 4 main-suite failures (9d740bf2: map-cache-key classification, frozen_encoder sorted fan_out, pin_blas
test, hooksPath test) + K08 grant renderer exempted from the sidecar test (25832aad); merge conflicts resolved: DECISIONS.md
(both entries), svg x gates (fig_goal_summary computed reading + save_png_svg; exec_vs_neural_match import), fr_pixel_significance
drift-control -> save_png_svg, pose3d_phase _save -> no_top_spine + save_png_svg (test updated ec0ff642).
**Second heavy slot LAUNCHED 15:48 (Priya yes):** lanes v8s2-{L0L1L8L10, L4L5L6L7L9, L2L3}; first step l_precue_restw 20 min,
min available 27 GB beside b_roiexact_frA. **Queue v9k RUNNING since 16:55** on worktree wf_render_v9k @ ec0ff642 (lanes K, C).
a8 told (SVG on main; redraw-67 plan stands). Final catch-up queue code = main ec0ff642 (or later if fixes land).

**Update ~16:40 -- second heavy slot INVESTIGATED (Priya "look into it"), NOT launched, awaiting her go.** Box 63 GB / 24 cores,
~44 GB available with only x_enl_move heavy. ~55 heavy steps remain one at a time: ~12 BIG (partial_refit a_partial_ls /
b_partial_frA [t_partial_te took 855 min], frozen_encoder t_frame_*/b_pix_union/b_roiexact/v_*, svf b_svf_frA_roi, epoch 1b,
section_g, gain_bio/quiet_spikes) + ~40 MEDIUM (v8 L family + b_wsd_lick; no duration / RAM measured yet). Ready, not run:
scratchpad/render_daqlk/queue_daqlk8_s2.sh = copy of queue_daqlk8.sh (running file untouched; same worktree / claims) with lock
HEAVY2.lock admitted only at >= S2MIN (25) GB available, never claims BIG_RE steps, logs min available RAM per step. Launch =
3 lanes: SECTIONS="L0 L1 L8 L10" / "L4 L5 L6 L7 L9" / "L2 L3" nohup bash queue_daqlk8_s2.sh > logs/queue_v8s2_<S>.out.
Undo = TaskStop those lanes (their running step finishes; claims of waiting steps: release by hand).

**Update ~16:10 -- EARLY video/pose catch-up = queue v9k (Priya "yes if it won't cause problems"; "plenty of CPU and memory").**
scratchpad/render_daqlk/queue_daqlk9k.sh (v8g machinery: shared lock, RAM pause, HALT brake; claims fresh / RESERVED_v9k).
22 steps: section K = A20 3-D kinematics (gates helper K lines: jaw_only figs, trial-class share, pose3d_cue_traces cue/lick,
spout_error, phase (+contra), summary_fig, vs_cam4, pose_validation); section C = A27 cam4 cohort + jaw-lateral (+_allpre),
robustness, cover_sens, deck, pooled_angle, jaw_xcheck, k_pooled, k_lmm_curves. pose3d_* NOT in HEAVY_RE (2 workers).
EXCLUDED (read render outputs -> FINAL catch-up queue): k_imaging_lmm, aim producer / decoder_retention / recovery_timing,
the 10 HOLD steps, A25 LP epoch arm. g7_fig_pose_validation (v7b, old code) -> RESERVED_HOLD (k_pose_validation replaces it).
launch_v9k.sh (bg) waits for push7 ALL DONE, checks cam4 is on main, makes detached worktree wf_render_v9k at that SHA, sets
EXPECT_SHA, starts lanes K and C. Old v7 lane E (queue_daqlk7, obeys legacy STOP) exited at 15:38 -- harmless: e_iti_area_maps is
RESERVED_v8, v7b lane E is alive. t_partial_te DONE 15:35; heavy lock now waits only on x_enl_move.

**Priya OK'd cam4-style push (15:3x).** (was: waiting on Priya: OK to push fix/priya1106-cam4-style when the helper reports (A27 changes cam4 numbers -> report
changed conclusions BEFORE any page text).

### 00-COMPACT-31 (11-06 ~14:30; times written as 16:00-16:20 below were estimates, real ~14:10-14:30) -- READ FIRST; supersedes 30 for state (30 holds for queue mechanics)

**Priya 11-06 16:00: "make sure we're not layering changes onto in-flight analyses."** Rule from now on:
1. **No more mid-render code changes into running queues.** Any new code goes on a branch; the not-started steps it
   affects are parked with `RESERVED_HOLD` (no queue takes it) BEFORE they can run on old code; parked steps run ONCE, in a
   single final catch-up queue, on one final code version.
2. Each queue's code is pinned by its worktree; never merge / checkout in a render worktree while its lanes run.
3. A step already DONE on code that has since been fixed is listed below for re-run in the catch-up queue (not silently
   left).

**Code versions in the render (same run tag _daqlk; outputs share folders):**

| code (worktree @ sha) | queues | contents |
|---|---|---|
| wf_render_daqlk @ a76352f3 | v7b (+ v6 leftovers t_partial_te, x_enl_move) | 11-05 merged main, pinned |
| wf_render_v8 @ 5db29b07 | v8, v8b, v8c, v8e | + decisions, ALM/tjM1 maps, speed-ups |
| wf_render_v8d @ d2d26e84 | v8d | + exact ROI frozen run (435160e2) |
| wf_trendfix @ 6033cbbb | v8e2 | 5db29b07 + trend-recovery *_trials.npz fix |
| wf_render_v8f @ 110c4efd | v8f, v8g | main 604c09d3 + priya1106-small / -frdrift / -gainbio |

Each change was additive for the steps that use it (tests pin Allen / old outputs unchanged), but a FIGURE FAMILY can mix
code versions; the provenance table records each step's code sha (log header).

**Branches NOT in the render (in flight):**
- `fix/priya1106-gates-captions` (wf_p1106_gates, helper running): A20 kinematics + 3-D post-cue-lick scripts -> camera
  working period; A21 recovery_timing denominator; A22 fig_goal_summary computed titles; A23 exec_vs_neural_match tag.
  Render overlap: kinematics / 3-D / recovery_timing / epoch_behaviour scripts are in NO render queue (page owners render
  them) -> no overlap. Overlap PARKED (RESERVED_HOLD, 16:00): b_exec_vs_decoder, b_exec_vs_far_contra, b_exec_vs_far_mid,
  d_fig_goal_summary(_f1/_f2/_f3). RE-RUN after the fix: d_exec_vs_encoder (ran on the stale relabel tag).
- `feat/svg-page-figures-20261106` (wf_svg, helper running): adds SVG beside every page-figure PNG; PNG byte-identical.
  Not in the render; SVGs come from the redraw pass (REDRAW_INVENTORY_2026-11-06.md, helper running) after it is on main.
- `feat/priya1106-small / -frdrift / -gainbio`: in the render (v8f / v8g) but NOT on main (push hold for a8).

**Push hold on main (a8, since 13:20)** still on. When it lifts: merge origin/main into each branch, push one at a time
(priya1106-small, -frdrift, -gainbio, gates-captions, svg) with Priya's OK per branch.

- `fix/deficit-qc-dir-20261106` (wf_deffix, fa391f25, 14:20): imaging_lmm / remap_lmm / remap_checks / iti_lmm_figures read the
  deficit table from <E>/qc (missing in a tagged run) -> run_mode.deficit_qc_dir(); static test. b_imaging_lmm FAILED on it
  (13:28). PARKED (RESERVED_HOLD): b_imaging_lmm, b_remap_lmm, b_remap_checks, e_mm, e_lmm_response, e_lmm_all.
  (b_within_spout_mixed + encoding_mixed already use deficit_qc_dir.)
- a8 asks: provenance rows for current deck slides 31, 111, 153, 154 (cohort2 490 vs 470: labcams/cohort2/490_vs_470/
  cohort2_490_vs_470{_dprime,}.png, led_490_vs_470{_dprime,}.py), 164 -> add in the post-render merge.

- a8 also asks (post-render merge): make producing_script explicit for ~50 deck rows without one (current slides 5, 7, 12,
  14, 16, 19, 33, 38, 45-57, 97-136, 141, 142, 181-192; mostly LMM / stars slides) so it can tell which read the deficit table
  (held) and which do not. a8 already flagged slides 8, 17, 20, 121, 122, 139 as waiting for the catch-up queue.

**14:55 Priya A25-A31 + catch-up queue contents (build after push6 + the cam4/style helper):** code = origin/main (after the six
branches land) + fix/priya1106-cam4-style (helper running: A27 whole cam4 orofacial_cohort -> camera working period,
A28 read-only calibration comparison, A29 lick_cued_aim_by_spout step, A30 style). Steps: the RESERVED_HOLD set (6 LMM +
d_fig_goal_summary f1/f2/f3) + kinematics section K (gates helper's lines; HEAVY_RE += pose3d_cue_traces|pose3d_phase|
pose3d_spout_error) + cam4 cohort re-run + aim producer + recovery_timing / decoder_retention (gates helper's D lines)
+ **A25 LP epoch arm**: `step g5_epoch_lp cont g2b_restnl_mask_lp "--block epoch_maps" "base" "$PY" -m
wfield_local.epoch_grant_figures --parallel-arms 2 --only <g5_epoch family list>` (block_env = WIDEFIELD_LICK_SOURCE=lp; outputs
route to LP_licks_<tag>; heavy ~6-9 h). SVG code included (A31: final queue only). A26: SVG scope = page + deck only.

**Redraw-67 (Priya via a8, ~15:05):** after feat/svg-page-figures is on main (push6 order: small, frdrift, gainbio, gates,
deficit-qc, svg), redraw ONLY the ~67 redraw-only figures with stored inputs today (REDRAW_INVENTORY_2026-11-06.csv) from a
fresh worktree at that main, <= 2 workers, >= 6 GB free; record PNG sha256 before / after; send a8 the list redrawn /
failed / PNGs that differ + their deck slides (a8 re-crops). No partials, no epoch_grant_figures.

**Final catch-up queue (to build once gates-captions is merged):** the 7 parked steps + d_exec_vs_encoder re-run (+ the
A25 LP epoch arm if Priya says yes; + any figure redraws for SVG per the inventory).

**Open for Priya:** A25 (LP arm of the imaging epoch figures, LP licks + ENL 1 s, ~9 h heavy; only the DAQ arm ran).

**Provenance table (complete merge):** `wf_attempt docs/status/rebuild_20261103/FIGURE_PROVENANCE_20261106.{csv,md}`
(297cedc9): 832 rows, 45 CONFIRMED / 635 INFERRED / 152 UNKNOWN, 69 caption conflicts, Data: template, canonical
definitions, findings (incl. corrections stated without multiple-comparison correction; far-contra decoder LP-trained).
Scratch tools: scratchpad `prov/` (parse_steps.py, merge.py, build_md.py) -- re-run after the render for CONFIRMED.

**Deck-note format (Priya via a8, 16:20):** fixed 7-line Data: block at the TOP of each PI-deck note (Window / Licks / Trial period
/ Trial types / Gate / Baseline / Correction). a8 writes notes only from CONFIRMED rows. TODO in the post-render merge (prov/merge.py):
add a `correction` column and a `data_block7` column (the 7 lines), then message a8. A23 deck = old slides 107 + 180 (page Fig 6.9).

**Helpers running at 16:00:** gates-captions (A20-A23), SVG writers, redraw inventory (for a8). Their reports arrive as
agent messages; nothing pushes.

### 00-COMPACT-30 (11-06 ~13:00) -- READ FIRST; supersedes 29 for state (29 / 28 hold for rules and history)

**Running list for Priya:** `docs/status/OVERNIGHT_2026-11-05_FOR_PRIYA.md` (sections A-D + timestamped updates through
11-06 ~12:50). Keep appending there; commit each time.

**Render `_daqlk` = SEVEN queue files, one shared logs/ dir, all in the scratchpad `render_daqlk/`:**
- Scratchpad = `C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\f082ee0c-b33c-49a8-a537-dcee3bb7c84c\scratchpad\render_daqlk`.
  `logs/CHAIN.log` = the event log; `<step>.done / .failed / .skipped / .claim/`.
- **Claims / reservations**: `logs/<step>.claim/RESERVED_<q>` says which queue may run a step; a lane takes it with
  `mkdir <claim>/<q>`. Swap a not-started step to another queue SAFELY: `mkdir <claim>/<old>` (mutex) -> touch
  RESERVED_<new> -> rm RESERVED_<old> -> rmdir <old> (use `rm -- "${c:?}/${r:?}"`; the safety hook blocks bare `rm $c/$r`).
  Never edit a queue file a lane is reading (bash reads scripts incrementally) -> make a new file / queue.
- Shared heavy lock `HEAVY.lock` (one pixel-encoder / refit / epoch-render step at a time across ALL queues) +
  `HEAVY_EXTERNAL.txt` (t_partial_te 23728, x_enl_move 36140: still-running v6 steps, ~until Fri late evening).
- v7b+ lanes PAUSE (wait for available RAM >= 10 GB before each step), never halt on low RAM; the legacy STOP file is
  ignored; manual brake = `HALT`. A low-RAM line in CHAIN is only a log.

| queue | code (worktree @ sha) | what | lanes |
|---|---|---|---|
| v7b | wf_render_daqlk @ a76352f3 (pinned) | the original 196-step plan minus what moved | G T AB FZ E (X, D ended) |
| v8 | wf_render_v8 @ 5db29b07 | rebuild_order_B steps 3-14 (L0-L10), decisions steps, ALM sect. M, F1-F9 fixes, 13 moved re-runs | GBMD, L0L1EL8L10, XL2L3, L4L5L6L7L9 |
| v8b | wf_render_v8 @ 5db29b07 | 18 dep-fixed steps (V7B_DEP_AUDIT.md) + 4 producers (iti_area_maps --split/--split --cued, frozen_residual_trend frA x2) | N, R |
| v8c | wf_render_v8 @ 5db29b07 | F8: e_plain_* TRUE plain Allen (WFIELD_AREA_BASIS=plain) + e_split_where / e_split_loclin_all + notdock dock arm | P, S |
| v8d | wf_render_v8d @ d2d26e84 | exact ROI frozen run b_roiexact_frA (heavy) -> b_svf_frA_roi_maps, b_remap_checks_roi (--roi-frozen exact) | M |
| v8e | wf_render_v8 @ 5db29b07 | mixed-model steps + z_frozen_claims waiting on m_trend_recovery | T8 |
| v8e2 | wf_trendfix @ 6033cbbb | m_trend_recovery (position_coding_trend_recovery; *_trials.npz glob fix) + z_frozen_claims + b_imaging_lmm | T9 |

- Notes for the queues: `V8_NOTES.md`, `V7B_DEP_AUDIT.md`, `v8_missing_steps_draft.sh` (all in the scratchpad).
  `RENDER_QUEUE_DAQLK_20261105.md` (wf_attempt docs) is the v1-v6 plan.
- **State at 12:55:** 65 steps done. Failures that remain: a_decomp_noFR_cue and a_relabel_noFR (old, superseded) and
  e_iti_hemo (undeclared dep; a waiter task re-runs e_iti_hemo + e_iti_hemo_sum after e_iti_area_maps.done).
  Reserved and not started: 110 v7, 18 v8, 15 v8b, 3 v8c, 2 v8d, 5 v8e, 2 v8e2. Available RAM 22 GB; C: 110 GB free.
- **Critical path:** t_partial_te (heavy; ~39/107 at 07:30, many h left) -> then one at a time: g5_section_g,
  t_frame_te_basis, a_partial_ls, b_partial_frA, t_frame_frlag, b_pix_union_frA, b_svf_frA_roi, b_roiexact_frA, the
  v8 heavy L steps. Forecast: render done ~Sun night / Mon; then the replacement pass (pages, analysis deck, PI deck).
- **Watch for:** steps failing on inputs nobody produces / undeclared deps (pattern today: fix by holding the step,
  adding the producer in a new small queue). Re-arm the CHAIN monitor every 30 min (filter: EXIT [1-9], queue end,
  heavy lock taken, SKIPPED, ABORT, HALT; available RAM < 3 GB for 3 checks).

**Main (all pushed with Priya's direct OK here):** 5db29b07 (decisions f1/f2/ENL + follow-up, ALM/tjM1 maps, speed-ups),
05dcab8d (exact ROI run 435160e2), 604c09d3 (trend-recovery fix). Log-deficit branch on main since 08:30 (e5f53dbd).
Branches done: fix/decisions-*, fix/log-deficit-primary-20261105, feat/alm-tjm1-slides-94-139-20261106,
perf/attempt-perm-iti-lags-20261106, fix/trend-recovery-trials-npz. Render worktrees above: do NOT check out / merge
in them while lanes run (I merged main into wf_trendfix once by mistake and reset it to 6033cbbb within a minute).

**Priya's decisions 11-06 (recorded on the overnight page):**
1. merge ALM + speed-up -> done. 2. F8 e_plain_* true plain Allen -> v8c. 3. slide 94 = *_split maps REPLACE Allen
   (deck must point at `peth_trial_maps_<w>[_alone]_split.png`). 4. slide 139 without no_lickspout: OK. 5. slide 139 exact
   frozen ROI side -> v8d. 6. slide 92 wording -> a8 applied ("tracks position-carrying movement more strongly (not
   explained by more variable movement; provisional)"); page Fig 2.11 caption in the replacement pass.
   7. C: cache deleted (58 GB; list rebuild_20261103/DELETED_LOCAL_CACHE_20261106.txt).
- Earlier today: lock + pause queues OK; render shortcut (attempt_coding_pp reuses trials files) OK; future-run
  speed-ups OK; ALM/tjM1 on slides 94 + 139 + every area map (audit: Fig 4.8 and reembed added; 18.6 / 18.9 split panels
  to swap in on the page; specificity Layer 2 ROI via a_allsession_frA_roi); posinfo unique-R2 per-trial re-run = LOW
  priority, after the render + deck / page update.

**Still open for Priya** (explained in plain words in the chat 11-06 ~11:30): 8 speed-up follow-ups A10 (memoise
paths.cohorts, rec yes now), A11 (index frozen-model store, after render), A12 (fan_out restores parent BLAS threads,
after render); 9 slide 102 five choices (A1); 10 FR pixel drift panel + "Bonferroni" captions (A2); 11 optional no_spont
(A3); 12 bare-run default (A4, warning only for now); 13 PS93 no discriminator (A5); 14 chronic-stability row camera vs
DAQ (A6).

**Held for routing fixes (not queued):** fr_enl_decode (writes <dlc>/face_rhythm, outside the rebuild root),
revup_behaviour (untagged iti_licks folder). Cut: lick_acad_midline_control.

**a8 (PI deck window):** deck = `presentations/Overview_2026-11-06.pptx`. Waiting on me: slide 94 re-tile when the
`*_split` maps land (b_peth_trial_maps_split / _alone_split after b_pix_union_frA); its item 5 (fail-loud DAQ contacts)
holds until the render is done. a8 ran the posinfo variance check (posinfo_checks_v7g3_t3d_spont_frAsplit/).

**Helpers this session (finished):** speed-ups, ALM/tjM1 (+ exact ROI), v8 / v8b / v8c assembly, area-map audit.

**13:10 Priya answered 8-14** (overnight page): A10 memo now (A11/A12 after render); slide 102 = copy FR cache, 1 s
quiet events, camera working period, 415 comparison like-for-like (corrected), rest masks rebuild (SCOPE first); FR
captions fix + supplemental drift-control panel; no_spont on pixmusall twins / frAsplit_basis; bare-run WARNING; PS93
relaxed late window (PS93 only); chronic-stability row = camera (no change). **Three helpers RUNNING** (no pushes; each
reports code + tests + draft queue steps): worktrees wf_p1106_small (branch feat/priya1106-small: A10, A4 warning, A5 PS93),
wf_p1106_gainbio (feat/priya1106-gainbio: merges port/gain-bio-20261105 + choices 2-4 + rest-mask scope + slide-102 queue
steps), wf_p1106_frdrift (feat/priya1106-frdrift: Bonferroni->max-statistic captions, drift-control supplement, no_spont
steps). Next: review, push to main with Priya's OK, queue their steps as a new small queue (v8f) with reservations.

**13:20 PUSH HOLD on main for a8** (~2 h, until a8 messages): its 21-commit cleanup push (merged to 604c09d3, 1933 passed)
lost the race and is re-running the full suite; a8 helper also testing feat/decks-to-presentations-20261106 (decks ->
presentations/pipeline/). Do NOT push to main until a8 says go; then merge origin/main into each branch before pushing.

**Background tasks of this session (re-arm / check after compaction):**
- render CHAIN monitor (30-min Monitor; re-arm on expiry; filter in the bullet above);
- waiter re-running e_iti_hemo + e_iti_hemo_sum after e_iti_area_maps.done (v7b, from wf_render_daqlk);
- disk watcher (diskwatch.sh, writes STOP if C: < 20 GB -- note v7b+ lanes IGNORE STOP; only v6 leftovers obey it);
- three helpers (wf_p1106_small / _gainbio / _frdrift) -- their reports arrive as agent messages.
- Lanes: v7b G T AB FZ E; v8 GBMD, L0L1EL8L10, XL2L3, L4L5L6L7L9; v8b N R; v8c P S; v8d M; v8e T8; v8e2 T9; v6 leftovers
  t_partial_te (TE2) and x_enl_move (X) until they finish.

### 00-COMPACT-29 (11-05 ~22:45) -- READ FIRST; supersedes 28 for state (28 / 27 hold for rules and history)

Priya is ASLEEP: work overnight on everything possible. Her page = `docs/status/OVERNIGHT_2026-11-05_FOR_PRIYA.md`
(decisions / questions first, findings, status, then "D. Overnight updates" -- append there + commit each time).
Detailed log = RUNNING_LIST_2026-11-01.md (11-05 entries, newest at the end).

**Code on main.** origin/main has the merge (136807cf, 16:18) + my fixes 5bd7faa2 (frozen_relabel routed),
2dc96a9e / 76066e01 (encoding_mixed 3-D BINNED panels u_tgtbin, --plot-only + saved long table), 5a6b7986 (ttc_log),
5993d70e (--bin block size), e53f1cd2 (fig10 title), a6edab62 (misdirected bundle_dir), 411c8385; a8's cleanup
(d39ee11c library moves, item 3 response window pushing).
**Unpushed branches (mine):**
- `fix/decisions-f1-f2-enl-20261105` (worktree wf_decisions, tip 9fd25043 incl. origin/main merge): (f)1 precue
  engagement discriminator camera classes, (f)2 Fig 0.2 camera period, ENL slide (wfield_local/enl_window.py), behaviour
  camera period (disengagement camera quit, revup attempts path). Priya OK to push. First push REJECTED by pre-push
  (14 failed / 574; test_rebuild_root require_default... / r3_rebuild_figs..., test_spout_behavior curated_mmdds...; both
  files pass alone, 87). RETRY running (task bzqykfs5k; full log scratchpad/push_decisions_2.log). If it fails again:
  read the log, fix, don't bypass.
- `fix/log-deficit-primary-20261105` (worktree wf_merge_main, off main d882558d): b185ad70 log deficit PRIMARY (DEFICIT =
  ttc_log, Deficit carries transform, plot_fits draws original axis + curve), ef663dc7 no top spine, 905f3ca0 plot_fits open
  markers shape = animal / colour = epoch (Priya 11-05). Priya approved both changes -> push after the decisions push lands
  (merge origin/main first).
- `port/gain-bio-20261105` (worktree wf_gainbio, 1b9eb446): slide 102 chain ported (scripts/gain_bio/). NOT to render
  until Priya answers the 5 choices (overnight page A1).
- `fix/decisions-followup-20261105` (worktree wf_decisions2; HELPER running): 1b hit rate -> camera period; iti_lick_rate +
  within_lick_correction cut to the working period; trial_attempts rebuilt under daqlk into scratchpad/attempts_daqlk + diff
  vs the 11-04 table; precue discriminator floor 20 -> 15 trials (late window kept). Review its report; push only with
  approval-covered items (all four were approved by Priya 11-05).

**Render `_daqlk`** (scratchpad `render_daqlk/`; worktree wf_render_daqlk @ a76352f3 = 136807cf + 2 fixes; EXPECT_SHA
accepts 136807cf 9aaa4cc6 a76352f3). Queue versions -- NEVER edit a queue file a running lane reads (bash reads as it
goes); make a new version: v2 = lane 2 (A B D E F Z, no claims); v3 = claims (mkdir logs/<step>.claim); v4 = E ITI order;
v5 = section X; v6 = + no_spont lsland/frlag, svf no_spont/no_other, Fig 11.2 = LP 1 s (WIDEFIELD_LICK_SOURCE=lp). Lanes
now: G (queue_daqlk.sh, g5_epoch since 17:03), 2 (v2, a_relabel_noFR 90/107 at 22:34), T (v6, relaunched 22:30, waits on
the te re-run lane = v5 ONLY te steps, t_frame_te 68/107), D (v3), E (v6, e_iti_key), X (v6, x_enl_move LP), SVF (v6 ONLY
a_svf_ls / b_svf_frA, 47/107). Only 11 / ~227 steps done at 22:34; per-step times 2-4x estimates (network-bound ~92 MB/s).
- Low-RAM STOP: the queue writes `render_daqlk/STOP` when free < 6 GB twice (nothing killed); lanes halt at the next step
  boundary. Clear it (move to logs/) only when RAM is back, then relaunch halted lanes (log it in CHAIN.log).
- Monitor: re-arm the CHAIN.log poll monitor every 30 min (EXIT [1-9] / HALTED / SKIPPED / queue end / STOP / LOW + free
  RAM < 6 GB).
- Held: lane-X ENLv chain waits for `logs/ENL_SLIDE_FIX.done` -- create it only when the decisions branch is merged AND in
  the code lane X imports (worktree HEAD at step start).
- After the decisions + log-deficit branches land: bring them into the render code (new worktree / commit + EXPECT_SHA in a
  NEW queue version) and add steps: precue_engagement_states (--align precue, cue), fig_behaviour_hits, lick_enl_angle_decode
  --extract -> lick_enl_offset_explore --history --offset -> lick_enl_target_test -> re-run e_pat_summarise, revup_neural,
  fr_enl_decode, disengagement_tests, revup_behaviour; re-run the behaviour-model LMM steps (imaging_lmm, remap_lmm,
  remap_checks, within_spout_mixed, iti_lmm_figures) for log deficit + new markers; then ENL_SLIDE_FIX.done.
- Fixed tonight (logged): te frame needed exec_bins_calib in the input cache (copied, sha 732b0d12 = bundle); lsland run
  without no_spont stopped + moved aside; iti order (split_sd -> zscore_pre -> main; zscore_pre marked done after its SD
  tables, full e_zscore later).

**Helpers running:** decisions follow-up (above); render IO / local-cache measurement (report ->
rebuild_20261103/RENDER_IO_LOCAL_CACHE_20261105.md): decide start-now vs after only if safe + clearly worthwhile
(Priya approved the local cache for the next render; mid-render needs a new worktree / queue version), else put it on her
list.

**Results tonight (EARLY, no page text changed):** encoding_mixed binned + LOG deficit (primary): cue -- spout identity
tracks recovery (-8.1 q .015), landing -3.15 q .004; lick -- landing -6.8 q < .001, spout identity n.s. (-4.5 q .55) but
above pre (+10.5 q .042). Log beats linear (dAIC -8..-20; qc/encoding_mixed_lin_vs_log.csv). FR pixel calibration fine
(4.6-4.9 %); pre-stroke DRIFT is real (FR_PIXEL_CALIBRATION_20261105.md) -> Priya's call. Decision audit
(DECISION_AUDIT_20261105.md) + spont co-variation audit (SPONT_COVARIATION_AUDIT_20261105.md): conventions to fix at the
replacement pass (sig_marks.sig_star stars on data -> star_top; spout order in 3 scripts; "Bonferroni" -> max-statistic
captions; stale uninstructed / lick labels; fig10 done).

**Pitfalls learned today:** PYTHONPATH in Git Bash must use `$(pwd -W)` (Windows paths) -- `/c/...` is ignored and Python
imports the shared widefield_pipeline checkout; pre-push hook tests can use 13 GB (watch RAM; the render STOP fires);
queue "EXIT 0" can hide per-session failures (te: "!! stats" on every session) -> check outputs, not just exit codes;
`--help` dry runs miss runtime path bugs (relabel routing, decompose absolute --folder, ITI table order).

**Added ~23:05:** PI deck fluoroscopy update (Priya's request: new all_graphs A-J with G bolus density / H barium amount,
density_method hidden after 159) is being done by a8 (deck owner, sole writer) -- don't touch the deck; log a8's done
message on the overnight page. a8's bare-run pre-cut question is on Priya's overnight page (A4) with my rec. Render monitor
= task re-armed every 30 min (last bmn0d58bm). Decisions push retry = task bzqykfs5k.

**Added ~23:30:** decisions push REJECTED twice by the same 14 tests (test_camera_gate_fail_closed x2, encoder_licks_all,
lick_detector_daq cache_versions, lp_gate_option x2, merge_review_b1, priya_decisions_1103 coverage_note, rebuild_root x5,
spout_behavior curated_mmdds); they pass alone -> test-order pollution, suspect the new tests' imports of scripts modules
(sys.modules caching under a no-profile env). Bisect running (task bkc510ddf: each new test file + rebuild_root +
lp_gate_option). Fix the tests, re-push; never --no-verify. Follow-up branch fix/decisions-followup-20261105 (wf_decisions2:
ee21ac5d floor 15, f069b949 1b camera / iti_lick_rate / within_lick_correction / trial_attempts --out) -> push after.
Trial table under daqlk = identical. IO report: local cache < 10 % gain, don't switch mid-render; DISK watcher
render_daqlk/diskwatch.sh (task b5mvbzfir) writes STOP if C: < 20 GB (cache grows 6-10 GB/h; 85 GB free 23:30).

**Peers:** a8 (cleanup; told: no rebase, number-changing items held for Priya; bare-run pre default -> my rec in the
overnight page A4); e0 (kinematics; waiting for my lick_aim_deficit / iti_lick_rate renders). Priya's pending answers are
in the overnight page section A.

### 00-COMPACT-28 (11-05 ~13:00) -- READ FIRST; supersedes 27 for state (27 / 26 hold for rules and history)

Day log with every step: docs/status/RUNNING_LIST_2026-11-01.md, sections "11-05 overnight" onward.

**Code.**
- rebuild/cohort1-20261103 = **31ec5c1a**. Not pushed by itself; it reaches origin through the merge below.
  - 90b7ddd5: DAQ behaviour lick detector, run-tag part `daqlk` (log 8.53).
  - e05ece18: O2 bundles recorded without `daqlk` are accepted (`o2_bundle.NON_INPUT_TAG_PARTS`, log 8.54).
  - 31ec5c1a: `collect --move` (log 8.55).
- **rebuild -> main MERGE.**
  - Branch merge/rebuild-into-main-20261105, worktree C:/Users/SabatiniLab/Github/wf_merge_main.
  - 2730033a: the merge of 31ec5c1a. Then test fixes, and 3284cc25 (non-profile nightly_figs defaults cohort 1 to
    August-only pre; Priya OK).
  - 1ba23601 and the next merge: origin/main through a99a612a merged in.
  - Report: docs/status/MERGE_REBUILD_INTO_MAIN_20261105.md (on that branch).
  - **Push to main started ~12:55 with Priya's direct OK.** Fast-forward. The pre-push hook runs the full suite
    (~80 min shard). Check: `git -C wf_merge_main fetch origin; git log -1 origin/main`.
  - When it lands, send a8 the SHA. a8 holds ALL main pushes until then; it then does decision 9 (ITI stats -> ak) and
    its post-merge items.
  - After the push, new rebuild work goes on main (or a branch off it), not on rebuild/cohort1-20261103.
- The frozen lanes run from worktree wf_rebuild_render_spont @ 90b7ddd5. Don't touch it until they finish.

**Data (all under the `_daqlk` run tag** `.../imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/`**).**
- **O2 bundle b (enc_rebuild_20261104b): ALL 15 TAGs collected.**
  - Every TAG has 107 sessions.
  - main and main_nolicksall were copied, sha256-verified, and the bundle copies deleted.
  - The other 13 were moved with `collect --move`.
  - droppos needed a 96G resubmit of 14 OOM tasks (48G limit).
  - Re-copies from Priya's whole-out/ rsyncs were sha256-matched and deleted, with markers in each TAG folder.
  - Nothing is left to copy from O2. Bundle b's run.json and markers are kept.
  - Folder names: see RUNNING_LIST 11-05 ~10:20 / ~11:40.
- **Frozen v7g3_t3d_spont (primary) frames done:** noFR, frA, *_basis, *_pixmusall, frAsplit_basis, and v7g3_t3d (the
  no-spont robustness frame).
- **Running at 12:05 (watcher bz3az9kf1):**
  - frA_ls: 43/107 at 12:05. lane 3 then runs frA_ls_basis.
  - frA_nospont: 33/107 at 12:05.
  - ETA ~17:00 for both. Logs: scratchpad/frozen_v7g3_spont/logs/.

**Results so far (EARLY, no page text changed; conclusions go to Priya first).**
- Frozen contact-only R2 (trials.npz, cortex, cue): animal mean pre / acute / subacute / chronic = .507 / .415 / .440 /
  .340 (spont) and .492 / .413 / .433 / .347 (no-spont).
  - boot_delta excludes 0 in every epoch; epoch LMM (df 3) p .059 / .089 / .068, so no star (a8,
    FROZEN_B12_CIS_2026-11-05.md in widefield_pipeline docs/status/reanalysis_20261105/).
  - Caveats: PS92 is a steady trend from its first pre session; PS95 late chronic is partly kernel over-prediction;
    the comparison controls the class mix only.
- B12 unique pos_enl: the spont kernels change it by < .005 in every epoch (all CIs include 0).
  - The chronic rise is +19 [+5, +41] x1e-3, LMM p .086, and is carried by PS92.
- Recommendation given to Priya: spont kernels primary, no-spont as a robustness row (PS95 late chronic shown).

**Helpers.**
- Section C render (O2-dependent encoder figures; ENCODER_RENDER_COMMANDS.md section C) is RUNNING.
  - Report goes to wf_attempt docs/status/rebuild_20261103/SECTION_C_RENDER_20261105.md.
  - It was told droppos is now available.
- a8 (another session): its 11-05 decision relays are recorded in widefield_pipeline DECISIONS_PENDING_MERGE (f) 1-9.
  Mine for the replacement pass:
  - A14: per-class encoder figure on the MAIN encoding page;
  - slide 95 numbers (tell a8 if they change);
  - A17: the lick-detector methods line.

**Next, in order.**
1. Push lands -> SHA to a8 -> RUNNING_LIST entry.
2. Section C report: review it.
3. After the frozen lanes: build the render queue for `_daqlk` on the MERGED code (queue2.sh is stale), in a new
   detached worktree at the pushed main SHA. Run G0-G8, then the encoder family A/B/D/E/F
   (rebuild_20261103/ENCODER_RENDER_COMMANDS.md). Overnight; keep >= 6 GB free and <= 8 workers.
4. Verifiers, then the replacement pass (pages + decks), including A14 / A17 / slide 95.
5. Small items:
   - delete the superseded old-tag frozen partial folders (27: `frozen_encoder_gate-attempted_v7g3_t3d{,_frA,_frAsplit}`
     under `...stopA2jexrt3500c70/`), with a kept list;
   - give the cleanup window the trial_attempts per-row YAML caching (byte-identical check sha256 bef06b0a).

### 00-COMPACT-27 (11-05 ~00:40) -- READ FIRST; updates 26 (26 holds unless changed here)

**DECISION (Priya 11-05 ~00:15): imaging switches to the DAQ BEHAVIOUR lick detector now, before the render.**
- Measurements: widefield_pipeline docs/status/LICK_DETECTOR_COMPARE_2026-11-04.md (main a69fb9c3; relayed by a8).
  - Change: from `_load_daq_events(h5,"lick_analog",2.5,1.0,(0.001,0.020),0.10)` to the yaml `lick_detection` profile
    (2.5 / 0.5 V, 40 ms min ILI, no 0.10 s refractory).
  - Effect: 0.58 % of licks restored (PS95 1.4 %); 0 hit changes; 10 / 6,520 RTs change.
- Consequence: everything on tag `...stopA2jexrt3500c70` is SUPERSEDED once the new tag exists. The new tag gets a
  detector token.

**State at 00:40.**
- **Code tip.** rebuild/cohort1-20261103 = **34425934**. It is c149decf plus the merge of rebuild/frozen-licks (910c408b +
  efc4d517, log 8.51): frozen design **v7g3_t3d_spont** (`FROZEN_LICKS_ALL=split`, the profile default; robustness arm
  `frozenNoSpont` = v7g3_t3d). With the switch off it is byte-identical to v7g3_t3d.
  - The full suite on 34425934 is NOT done. I stopped it at 40 % because one pytest process reached 14 GB. Re-run in 4
    shards, one at a time.
- **Helper 1 (frozen-trials):** branch rebuild/frozen-trials in worktree wf_rebuild_frozenlicks, off 34425934.
  - Adds frozen `<label>_trials.npz`: per-trial SSE / SST, cue- and lick-binned, in the encoder layout (one reader), with
    cls (hit / incomplete), held-out flag, areas + frA frames.
  - Must also score existing fits post hoc (so v7g3_t3d can get it without a refit). Log 8.52.
  - Purpose: test whether the frozen post-stroke R2 drop survives within contact trials (class-composition confound).
- **Helper 2 (daq-lick-detector):** branch rebuild/daq-lick-detector in worktree wf_rebuild_lickdet, off 34425934.
  - One yaml accessor for all imaging lick call sites (~40); min_ili fallback fixed to 40 ms; cache / LP version bumps;
    new run-tag token; onset counts pinned before / after (incl. PS95); trial_attempts diff preview. Log 8.53.
- **Frozen v7g3_t3d render: STOPPED** at 00:15. STOP file in scratchpad/frozen_v7g3/; exact PIDs stopped.
  - Partial folders (superseded, delete with a kept list later):
    `frozen_encoder_gate-attempted_v7g3_t3d{,_frA,_frAsplit}` under the rebuild root `.../hv-...stopA2jexrt3500c70/`.
  - Lane scripts for reuse: scratchpad/frozen_v7g3/ (lane1_noFR, lane2_frA, lane3_variants; lane_lib waits for 14 GB
    free). Spont copies: scratchpad/frozen_v7g3_spont/ (te removed, no_spont added).
  - Measured speed: ~0.4 sessions/min per lane with 2 lanes running, not 0.8. A 107-session frame takes ~4.5 h incl.
    templates.
- **O2 bundle b (enc_rebuild_20261104b):**
  - Priya cancelled 13 arrays. Kept: 58602842 `main` and 58602843 `main_nolicksall`, as an EARLY look at B12 only.
  - Copied so far: bins_w1 / w05 / w2 (183 sessions, all checked: run.json env right, fitted = attempted exactly, no NaN,
    cortex R2 median ~.52). These are superseded.
  - Expected by design, not bugs:
    - 'unknown'-class cues (no imaging / camera, e.g. PS92_0908 68 cues past imaging end) have no trials.npz row;
    - shuf:* models are NaN for scored-only trials (flagged not scorable).
  - When main + main_nolicksall are copied back: collect with LICKS_ALL=split + ENL_SOURCE=lp exported. Report B12
    early: drop:pos_enl split vs nolicksall, per animal x epoch, LP enl + fixed2s.
  - Bundle b is then deleted (M: + scratch; kept list in RENDER_LOG) when the new bundle is built. Priya does not need
    to re-stage until then.
- **Stray process:** PID 30404 (11-03 encdrift chain) was stopped with Priya's OK. Its chain then started a drift900
  re-run (PID 3384), which I stopped within ~1 min. No outputs were touched (checked).
  - Lesson: check a process's parent chain loop before stopping it.

**Next, in order.**
1. Helpers land, then merge frozen-trials + daq-lick-detector into rebuild/cohort1-20261103 and run the full suite in
   4 shards.
2. Rebuild trial_attempts under the new tag and diff it against the stopA2jexrt3500c70 table (class / RT /
   timing_source changes).
3. Build O2 bundle c. First set the conda module to 26.9.4-1 in the code (scripts/o2_encoding.py:336 and the other
   sites listed in 26); bundle b was only patched by hand. Run the local checks, then give Priya the stage + submit +
   delete-b commands.
4. Frozen frames, both designs with trials output, from one merged commit, in a new detached render worktree (not
   wf_rebuild_render, which is at c149decf). Primary v7g3_t3d_spont first, then v7g3_t3d (robustness), then lane-3
   variants. Keep >= 6 GB free.
5. G0-G8 render queue for the new tag (queue2.sh is stale), the encoder family, and the open items from 26
   (render / verifiers / replacement pass).
6. a8 has been told the change goes in now; add one line to the per-slide old -> new list when it lands. DECISIONS
   entry held in widefield_pipeline docs/status/DECISIONS_PENDING_MERGE_2026-11-04.md.

### 00-COMPACT-26 (11-05 ~00:10) -- READ FIRST; updates 25 (25 holds for code / data / queue)

**O2 (per-session encoder, bundle enc_rebuild_20261104b).**
- The first test job, 58601937, FAILED in 3 s: O2 retired conda/miniforge3/24.11.3-0; the current module is 26.9.4-1.
  - Fixed: run_enc.sbatch patched on M: and on scratch (Priya ran the sed). Priya cancelled the 15 early arrays.
- Test job 58602462 is RUNNING (good start line, empty .err). Priya checks for "full cvR2" + _trials.npz, then submits
  the 15 arrays.
- Commands given to Priya (with A=ATTEMPTS=tables/trial_attempts.csv):
  - progress: squeue / per-TAG json counts / sacct failures;
  - OOM resubmit: --mem=64G, same --export;
  - transfer-node rsync of out/.
- When main + main_nolicksall are copied back, collect (export COMBINED_ENCODING_LICKS_ALL=split + ENL_SOURCE=lp), then
  report B12 FIRST:
  - drop:pos_enl split vs nolicksall per animal x epoch, LP enl + fixed2s, n, CIs;
  - with it, the frozen target columns before / after.
- TODO (code): replace the hard-coded conda module everywhere and tell e0 / a8. Sites:
  - configs/defaults.yaml :947 / :1335;
  - scripts/o2_encoding.py :336;
  - o2_face_rhythm.py :763 / :780 / :1116;
  - o2_lp_train_bundle.py :108;
  - runbooks/o2_pose_inference.md.
- Swap-mask check: the interpolated swap pairs are a median 0.07 % / max 0.37 % of frames, so they are kept in the fit
  (not dropped). Harmless.

**Frozen encoder on this box (Priya: start every job when able, parallelized).**
- v7g3_t3d (robustness, no spontaneous lick terms): runner told GO at ~23:55.
  - Code c149decf, new tag; output folders must carry the design name; RAM-sized parallelism, <= 6-8 workers.
  - Progress is in RENDER_LOG.md.
- v7g3_t3d_spont (PRIMARY: licks / licks_pre / licks_iti, pre + ITI uninstructed): branch rebuild/frozen-licks, worktree
  wf_rebuild_frozenlicks, first commit 910c408b. Smokes + suite still to finish.
  - Then: merge into rebuild/cohort1-20261103, run the suite, update the render worktree, launch its frames.
  - Expected ~03:00-05:00.

**Next after that.** Rebuild the render queue for the new tag (the old queue2.sh is stale): G0-G8, the encoder family,
rebuild_figs, drift1500, plus the new figures (B11 panel, G4, slide-95 raw + z). Then the queued code items in 25, the
two verifiers, and the replacement pass.

### 00-COMPACT-25 (11-04 ~22:50) -- READ FIRST; updates 24 (24/23 still hold for rules)
Every decision and finding of 11-04 is in the running list (RUNNING_LIST_2026-11-01.md, block "11-03 EVENING", newest at
the end). Read it in full before acting.

**Code.** rebuild/cohort1-20261103, tip c149decf, NOT pushed. Full suite: 4,923 passed / 0 failed. It merges four
branches (log sections 8.43-8.50):
- encoder-classes:
  - per-trial outputs <label>_trials.npz + wfield_local/encoder_trials.py;
  - LICKS_ALL=split in production (licks instructed; licks_pre / licks_iti uninstructed), robustness TAG main_nolicksall;
  - enl window = lick-free sliding, LP 1.0 s (old window = enl_fixed2s);
  - ATTEMPTS re-derivation on O2; refresh-code; the --profile keeps-caller-vars fix; 48G.
- cue70:
  - cue + 70 ms by peak; response window to max_rt 3.5 s;
  - stop rule A2 "both sides" (arms A0 / A3 / strict / tail-event);
  - stopped_licking excise; stop_kind_for (run / tail);
  - LP timing fallback: close spouts DAQ > 50 ms, far spouts grooming flag;
  - tag ...stopA2jexrt3500c70.
- enl-window: one ENL window (lp_licks.enl_post_s, enl_window.precue_window_start), sliding lick-free for every ENL
  predictor; quiet events MIN_CLEAN_ENL = 0.5 x window.
- bcdg-port:
  - a8's B / C / D / G (D scored on working rows, G by camera class, G4 camera-class discriminator primary, G9 camera
    classes with guards);
  - centre Wt FINAL (decoder + projections; eax z on contact rows); projection frame ON;
  - sc-* tag parts and arms centreN / centreCto; the contact-vs-incomplete panel.
- The render worktree wf_rebuild_render is at c149decf.

**Data.** trial_attempts under ...stopA2jexrt3500c70 (19:16): working 49,087, stopped 5,347, stopped_licking 276.

**O2.** Bundle enc_rebuild_20261104b is READY: 107 sessions / 30 GB; checks passed (5e-8); commanded TARGET3D; split;
ENL lp.
- Priya was given the rsync + 1-task test + 15 sbatch lines (48G) at ~22:45. She is to confirm _trials.npz in the
  test output.
- The old bundle enc_rebuild_20261104 is DELETED (M: + O2; list in RENDER_LOG).
- Collect needs COMBINED_ENCODING_LICKS_ALL=split + ENL_SOURCE=lp exported.
- **When O2 lands, report FIRST:** B12 = drop:pos_enl split vs nolicksall per animal x epoch (LP enl + fixed2s, n,
  CIs), plus the frozen target columns before / after. Claim list: rebuild_20261103/B12_PRECUE_POSITION_CLAIMS_20261104.md.

**Running.** The frozen encoder is getting spontaneous lick terms on branch rebuild/frozen-licks (worktree
wf_rebuild_frozenlicks): new design / ids, v7g3_t3d kept as robustness. Merge it before the frozen frames render.

**Queued.**
1. Lick / FR ENL families (lick_enl_*, prelaunch / revup pre-cue, fr_enl_decode) switch to the sliding window
   (Priya); guard: start >= strobe + 0.3 s.
2. Lick-direction frame check re-run on the rebuild lick sets (a8 / Priya; per animal first).
3. G9 figure switch, ~2.5-3 days, after the render.
4. Port main's prep_3d_inputs (commanded table) at the next main merge.
5. Align far-spout no-LP timing between encoder and lp_licks (moot: 0 trials).
6. The render itself: queue2.sh is OUTDATED (old tag). Rebuild the render queue on c149decf: G0-G8, the encoder family
   (ENCODER_RENDER_COMMANDS: update env blocks for the frozen-licks design), scripts/rebuild_figs, drift1500, plus the
   new figures (B11 panel, G4, slide-95 raw + z pair).
7. Two verifiers, then the replacement pass. Page edits: widefield_pipeline PAGE_EDITS_PENDING_2026-11-04.md, plus my
   additions A13-A16 / B12-B16 (sent to a8).

**Open for Priya:** none pending from me.

**Watch list:**
- B12 (pre-cue position credit).
- Slide 95: raw vs session-z, wording is Priya's. Hidden slides 100 / 101 are in the deck; the gain is brain-wide and
  biological (GAIN_BIO).

### 00-COMPACT-24 (11-04 ~09:50) -- READ FIRST; updates 23 (23 still holds for the code rules / decisions)
**Code tip:** rebuild/cohort1-20261103 at 571b3955 (log 8.41 / 8.42), NOT pushed.
- c3cb3c0c: jaw_only counts toward the 6 in a stop run (tag ...stopRun6j). Full suite 4,735 / 0 failed.
- fb3b6304: imaging input-cache key.
- a6a29dd7: split area basis, wfield_local/area_basis.py (v8 roi_masks; iti_licks delegates). Encoder region table + Ra on
  split + plain; "cortex" = v8, "cortex_stat" = old mask.
- a7aec872: o2_encoding rebuild-aware. Bundle in <root>/o2/<name>; full profile env in env.json, re-applied on O2; identity
  refusal; check < 1e-4; collect to the tagged out_dir; 14 TAGs, fr_A_tr dropped.

**O2 bundle `enc_rebuild_20261104`:** building since 09:27, helper a0dc6e4f1c77fb99f, --jobs 6, ~1.8 sessions / min.
- ETA: inputs ~10:25; then `check` on PS93_0814 (pre) + PS93_0818 (acute), done ~10:45.
- Then send Priya the share path + COMMANDS.md: transfer-node rsync, login-node test task, 14 sbatch arrays on `short`
  (4 c / 32G / 6 h, %60), rsync back, collect.
- Smoke on PS92_0806: v8 cortex R2 .6152 vs cortex_stat .6160.

**Render: HELD** (runner a0db502ebea3b948d handed back; resume it by SendMessage or a new runner).
- Done under stopRun6j: init (6 seeds), trial_attempts, coverage tables.
- Queue: scratchpad `queue2.sh`. To launch: remove HOLD_j_g2a / HOLD_j_g2b, then `nohup bash queue2.sh &`, after the bundle is
  written.
- Then: the encoder family (ENCODER_RENDER_COMMANDS.md A, B, D-F), scripts/rebuild_figs, the drift1500 arm, the O2 collect +
  section C readers.
- RENDER_LOG.md in the root lists all SUPERSEDED old-rule caches / models (none deleted).

**Decisions since 23 (all in the running list):**
- jaw_only counts toward the 6 (done).
- Split area basis everywhere: v8 masks; orofacial = ALM, tjM1, SSp-m, SSp-n; MOs-vs-MOp contrasts become ALM vs tjM1;
  residual_roi pixel test primary + area tests corrected over the split family; cortex rows kept (v8); LocaNMF-label analyses
  left. Done in the encoder path; the FULL SWITCH comes after the bundle (helper; inventory
  docs/status/rebuild_20261103/SPLIT_BASIS_INVENTORY_20261104.md).
- Priya via a8: B / D / G. D scores working trials (engaged = robustness); G groups by camera class. Port a8's main commits;
  C is still open.
- Rest start = first camera lick (settled). The PS93_0813 DAQ fallback is NOT used (failed check).
- O2 encoder runs: rebuild them (Priya runs O2).
- Slide 94 caveat added (a8).
- Priya waits for the rebuild: no interim swaps.

**Finding (stop rule):** stop runs 177 -> 283, stopped trials 3,267 -> 5,944; working trials unchanged.

**Open for Priya:** the pattern-family pre reference (lick vs working); the FR validation window (keep -2..+4 s recommended).

### 00-COMPACT-23 (11-04 ~08:05) -- updated by 24
Session f082ee0c. The running list (`docs/status/RUNNING_LIST_2026-11-01.md`, block "11-03 EVENING", newest at the end) holds
every decision and finding in order. Commit docs on wf_attempt BY EXPLICIT PATH ONLY.

**Code: rebuild/cohort1-20261103 (worktree wf_rebuild, NOT pushed), tip c3cb3c0c.** Last full suite 4,723 / 0 failed
(7b5d3334); after c3cb3c0c only the related tests have run (144 passed).
- Log: `wf_rebuild/docs/status/REBUILD_CODE_20261103.md` sections 8.10-8.40.
- Profile: `cohort1_rebuild_20261103`. Run tag: `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopRun6j`.
- Rebuild root: `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103`, fenced.
- The profile also sets `code_root`. Render commands need PYTHONPATH = the render worktree: the editable install is MAIN's code.
- Rules now in force:
  - camera gate;
  - working = contact + incomplete, cut at the last camera lick; jaw_only is its own class;
  - stopped = >= 6 consecutive no_response OR jaw_only, starting >= 5 min after the first cue (Priya 11-04); earlier runs = early_run;
  - rest = docked lick-free stretches from the first to the last camera lick;
  - August-only pre for everything except june_august_drift / 12.1b / map_drift_prepre / restnl_mask (their blocks unset it);
  - lick counts = cue + 70 ms -> trial stop on one detector;
  - bal_em centre on working rows (>= 20), with PS93_0813 excluded from working selections (no video);
  - every pre-stroke model is stored (FREEZE_VERSION 2), keyed by label;
  - all 47 former require_default scripts are routed to <default dir>/<tag>;
  - the scratch-only page figure scripts are in scripts/rebuild_figs;
  - the attempts table comes from the run's camera classes (qc/<selection tag>/);
  - section 5b is on the working path.

**Render:**
- Runner agent a0db502ebea3b948d. Worktree `C:/Users/SabatiniLab/Github/wf_rebuild_render` at c3cb3c0c.
- Log: `<root>/RENDER_LOG.md`.
- Running now: trial_attempts under stopRun6j (started ~08:05, about 63 min serial), the coverage tables, the RESTnl DAQ gate,
  then the RESTnl LP gate.
- HELD until the O2 bundle is written: the epoch decoders, then grant_figures, section G, the encoder frames (see
  `docs/status/rebuild_20261103/ENCODER_RENDER_COMMANDS.md`), the ITI / lick chain and the page figures.
- Earlier attempts were superseded: the old-rule tag (stopRun6) got no figures, and its stored PS92-94 models are listed as
  superseded, not deleted.

**O2 encoder bundle (Priya wants it ASAP):**
- Code helper a0dc6e4f1c77fb99f is making scripts/o2_encoding.py rebuild-aware:
  - bundle inside the root;
  - the full profile env in env.json, applied on O2;
  - tag refusal;
  - a bundle-vs-local check.
- Then it builds `enc_rebuild_20261104` (production env = S1 run_bundle_full.sh; `--fr master_pre_v1`) and writes COMMANDS.md
  (transfer-node rsync, login-node test task, 15 TAG arrays on `short`: 4 c / 32G / 6 h, %60, then rsync back and collect).
- The bundle needs the stopRun6j attempts table.
- OPEN: does combined_encoding use Allen areas? If so, the new split-basis decision has to land in the encoder path first.

**New decisions pending implementation:**
- Split area basis for ALL Allen-ROI analyses (Priya via a8): one shared function with iti_licks, new spec ids. A read-only
  inventory agent (aea1c6b2f2540339c) is running; send its result to a8.
- B / D / G (Priya via a8): D scores working trials (engaged = robustness); G groups by camera class; B state discriminators in
  the spout frame. a8's helper builds these on MAIN; the rebuild PORTS those commits. C (projections) is still being re-checked.

**Open questions for Priya** (running list):
- the pattern-family pre reference (lick vs working);
- the FR validation window (-2..+4 s vs -2 s -> trial stop; I recommend keeping it).

The PS93_0813 DAQ fallback failed its check and is not used. Deleted with Priya's OK: the 4 mistaken qc tables.

**Owed:**
- a8: a per-slide old-vs-new list at the replacement pass (PI slides 9-13, 19, 89, everything rebuilt; imaging page sections
  1-4, 6, 7);
- e3: the *_sessions.csv paths.

Scratchpad launchers are backed up at `M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/scratchpad_scripts_backup_20261104/`.

**Next:** O2 bundle -> Priya runs O2 -> epoch decoders and the rest of the render -> collect O2 -> encoder figures -> two
verifiers (per animal first, then what moved) -> replacement pass (conclusion changes go to Priya first).

### 00-COMPACT-22 (11-03 late night, Priya asleep) -- superseded by 23 for state
Session f082ee0c. Running list = docs/status/RUNNING_LIST_2026-11-01.md, block "11-03 EVENING" (newest entries at the end of
its Questions list). Docs commits on wf_attempt BY EXPLICIT PATH ONLY.
- **Rebuild code** (branch rebuild/cohort1-20261103, worktree wf_rebuild, NOT pushed): tip 3a772a26+, full suite 4,576 / 0 failed.
  - main merged (698c4e57).
  - 5-min stop rule in; early_run class; warm-up gone.
  - bal_em centre = camera-working rows; >= 20 rows; PS93_0813 = explicit contact-row fallback, recorded.
  - LOSO keyed by label; section G refuses instead of refitting.
  - Lick counts on one detector (imaging alignment unchanged).
  - Per-block LP lick-source env.
- **Reviews:** fifth review + re-verify at 3a772a26 = NO BLOCKERS (REBUILD_CODE_20261103.md 8.10-8.25).
- **Priya 11-03 late:**
  - (1) August-only pre PROFILE-WIDE (except june_august_drift, 12.1b).
  - (2) PS93_0813 in DAQ-only behaviour rates via a DAQ-defined working period IF the agreement check passes (<= 2 points
    hit / no-response, <= 20 ms latency on every Aug pre session with video); fail -> back to Priya.
  - (3) JAW-ONLY OUT OF WORKING (via a8; main 39bb3d5e): working = contact + incomplete; jaw_only its own class. Default
    (flagged to Priya): jaw_only bridges stop runs, not counted.
- **11-04 state:**
  - Rebuild tip a8791963, suite green.
  - Aug-only pre is profile-wide.
  - Jaw-only-out matches main 48f2cc6d.
  - The DAQ fallback for PS93_0813 FAILED its agreement check and is not used.
  - Dry count OK.
- **11-04 later: RENDER RUNNING.**
  - Render runner agent a0db502ebea3b948d, from the pinned worktree C:/Users/SabatiniLab/Github/wf_rebuild_render at 47a36184.
  - Env: PYTHONPATH = that worktree (the editable install is MAIN's code!).
  - Log: M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/RENDER_LOG.md.
  - The code helper is routing the 47 require_default scripts (encoder / fr / layer2 / ...) family by family. Move the render
    worktree to each routed commit and render that family.
  - Next: two verifiers (per animal first, then what moved) -> replacement pass; owe a8 a per-slide old-vs-new list.
- **(resolved) BLOCKER (helper a0dc6e4f1c77fb99f):** working imaging rows must be camera contact + INCOMPLETE (mirror main's row
  construction); the rebuild had DAQ-contact rows only.
  - Validation, arm (b): rebuild code on the checks' old inputs must hit .856 / .727.
  - Arm (a) minus arm (b) = the data effect.
  - Then a reviewer re-verify -> rebuild_root init -> render.
- **Reviewer:** aa0138ac1e9c354a7.
- **Next:**
  - helper reports -> re-verify the new code (reviewer)
  - -> `python -m wfield_local.rebuild_root init --profile cohort1_rebuild_20261103`
  - -> render G0-G8 (plan COHORT1_SWAPMASK_RERUN_20261102.md)
  - -> two verifiers (per animal first, then what moved)
  - -> replacement pass. Conclusion changes go to Priya before any text is rewritten.

### 00-COMPACT-21 (11-03 evening) -- superseded by 22 for state (kept for the decision history)
Session f082ee0c. Docs commit on wf_attempt (analysis/attempt-coding) BY EXPLICIT PATH ONLY. This window = the
COHORT 1 FULL REBUILD. Running list: docs/status/RUNNING_LIST_2026-11-01.md (every decision + finding of 11-03 is
logged there, in order). Plan + decisions: COHORT1_SWAPMASK_RERUN_20261102.md. Pages: imaging LIVE v50 (e0 published
section 1); goal LIVE v312. No page edits until the rebuild's page pass; e0 can publish for us.

**Decided (all in the rebuild; full list in the running list, 11-03):**
- **Data:**
  - Variant `hemo_meegkit_hpfit_lag_refitT_swapmask` (built, 120 sessions).
  - NO new shared drift step (DRIFT_CUTOFF_CHECK_20261103.md).
  - LocaNMF = PROJECTION onto the stored per-session and joint footprints; refits are not reproducible (LOCANMF_REFIT_TEST_20261103.md).
- **Decoders:**
  - August-only pre per decoder block.
  - LP arm LP-trained (stored); DAQ-trained is the cross-check.
  - Frozen-model IDs carry a training-data fingerprint + selection tag + hemo variant (FREEZE_VERSION 2, fresh store).
- **Trials:** one classifier, wfield_local.trial_classes.classify_session (cue_stop window). The rules:
  - working = contact | incomplete | jaw_only, ending at the last camera lick;
  - stopped = >= 6 consecutive no_response; a jaw-only trial after the last lick bridges a run but is not counted or stopped;
  - noattempt = isolated no_response inside the working period;
  - warm-up = no_response before the first lick, in neither class (PROVISIONAL);
  - MIN_UNCOVERED_RUN = 5;
  - partial video coverage: rules are applied inside the covered span (PS92_0908 kept);
  - no-lick analysis counts to the trial stop with the same contact table;
  - rest period ends at the last camera lick;
  - PS93_0813 (no video) kept for contact-only analyses.
  - Recorded in DECISIONS "2026-11-03 (night) -- Camera trial gate" on the rebuild branch.
- **Decks:** preprocessing deck map slides regenerated; analysis decks regenerated; imaging section 15 -> pointer to goal section 23.
- **Encoder / ITI:** keep their own 600 s drift step (check: 600 s recommended + a 20-25 min sensitivity row, Priya to confirm).
- **LMM hidden slides:** DONE. 16 + 9 split images in grant_figures/lmm_hidden_20261103/; in the PI deck (182 slides, a8).
  About half of the bootstrap stars drop under the LMM.

**Code: branch rebuild/cohort1-20261103, worktree C:/Users/SabatiniLab/Github/wf_rebuild (NOT pushed).**
- Code tip 3ef70b5b; full suite 4,476 passed / 0 failed.
- Log: docs/status/REBUILD_CODE_20261103.md. Reviews: REBUILD_CODE_REVIEW_20261103.md, passes 1-4 + addendum.
- Render profile `cohort1_rebuild_20261103` -> run tag hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_sel-camera-cue_stop-endlick-stopRun6.
- **NOT SAFE TO RENDER YET** (fourth pass: R1 Fig 12.1b stale + default overwrite, R2 deck reads untagged figures,
  R3 rebuild_figs read old inputs, R4 attempt_maps silently partial, R5 guards fooled by LP / ENL / stray env).
- **IN PROGRESS (code helper):** the structural REBUILD ROOT. Under the profile, every derived output goes to
  M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/; writeguard refuses writes outside it; readers resolve inside it; the
  profile clears stray WIDEFIELD_* vars. Plus R1-R5, the stale T3 text / DECISIONS additions, and the test gaps. Then a
  FIFTH review pass before any render.
- Inventories + reconciliation + check tables: docs/status/rebuild_20261103/ (wf_attempt).

**Decided late 11-03 (Priya); these replace the open items listed earlier:**
1. **Encoder / ITI drift:** keep 600 s primary, plus a 1500 s sensitivity arm for the spout contra - ipsi cells and the ITI
   pre-cue level readouts. The override comes from analysis/encoder-drift-cutoff 25e294d2; ENCODER_DRIFT_CUTOFF_20261103.md.
   "82 % of the cycle band" is the slow band of PS92 subacute (drift-dominated), NOT 82 % of the position signal.
2. **Task rule:** after the first lick the animal has 3.5 s to keep licking; with no lick, the trial ends at cue + 3.5 s
   (trial length = first-lick latency + 3.5 s).
   - Licks per trial = TOTAL licks from the cue to the trial stop while the spout is present.
   - Lick RATE comes from the inter-lick intervals, NOT count / time.
   - The helper verifies the rule on real trials.
3. **"Stopped"** = a sustained run (>= 6 no_response) that starts >= 5 min after the session's first cue. This replaces the
   warm-up rule; earlier runs are in neither class. The helper first reports whether any current stopped run starts in the
   first 5 min (Priya expects none).
4. **LMM captions:** the main slides keep their bootstrap captions. Each LMM hidden slide carries the LMM results AND its own
   on-slide LMM caption: rec_wlc csm_share p .13 / .13 / .17; rec_kn chronic norm p .14; spec_groups acute residuals
   p .065-.16; recovery_timing acute gap p .21. Asked of a8; confirm it is done.
**Open for Priya:** none right now. Next decisions come with the fifth review and with the render results (per animal first).

**Coordination:**
- e3 owns trial_classes and the consolidation: A1 / A2 / A4 landed on main; A3 / A5 still to land.
  Merge main into the rebuild branch before rendering.
- a8 owns the PI deck. e0 publishes pages.
- Tell e3 when the rebuild's *_sessions.csv exist (PI slides 10 / 11 / 12 / 19).

**Next:**
1. Rebuild-root fix.
2. Fifth review.
3. Merge main (A3 / A5).
4. Render G0-G8 into the rebuild root, logged.
5. Two verifier agents: per-animal first, then what moved.
6. Replacement pass (share, decks, pages, PI deck); conclusion changes go to Priya first.

**Helper rule (11-03 incident):** stop only via TaskStop on your own task, or by your own reported PIDs. Never select
processes by parent or name.

### 00-COMPACT-20 (11-03 ~midday; UPDATED 11-03 afternoon, see the block below the "Next" line) -- superseded by 21, then docs/status/COHORT1_SWAPMASK_RERUN_20261102.md + RUNNING_LIST_2026-11-01.md
Same session f082ee0c; docs commit on wf_attempt branch analysis/attempt-coding **by explicit path only** (another session,
sabatinilab-5b, keeps uncommitted work in this worktree; `git commit -am` swept it once 11-03, undone). This window = the
**COHORT 1 FULL REBUILD**. Pages: imaging LIVE v46 (other sessions edit section 1 [5b] - read live before any edit); goal
LIVE v312 (section 23 = quiet window; at its 512-file limit). NO page edits until the rebuild's page pass.

**Decided by Priya (11-02 / 11-03), all go into ONE rebuild:**
1. Cohort 1 adopts `hemo_meegkit_hpfit_lag_refitT_swapmask` (swapped pairs interpolated, 415 advanced 16 ms, T refit; built
   for all 120 curated sessions 11-02; post-SVD, no .dat / SVD redo).
2. Drift: REPLACE the order-10 polynomial step (keep the 0.1 Hz high-pass used only for the T fit) with a BLOCK-PROTECTED
   slow cosine basis (cosine + one indicator per position block, subtract only the cosine part), in the SHARED signal.
   Cutoff >= 10 min, set from measurement; must NOT remove block-level information (PS92 / PS93 blocks can be minutes).
3. Decoders: August-only pre; LP arm LP-trained (stored), DAQ-trained opt-in cross-check.
4. New default "working" = camera-defined (jaw moved OR any camera-seen lick); DAQ quit-tail = robustness row. Single
   definition: widefield_pipeline scripts/jaw_only_incomplete_figs.classify() + gate helper (sabatinilab-5b finishing it).
5. One cache rebuild (drop WIDEFIELD_PRM_LEGACY_DATE8, convert precue_engagement_states, narrow the map-cache code digest),
   fresh frozen store; old caches deleted afterwards from a kept list. 5rm stored under one seed-keyed spec.
6. Merge restnl-maps, lp-final, rest-decoder-nl, frozen-default, cohort1-swapmask into main BEFORE rendering; render from main.
7. Replace (after verification) in the share, imaging decks, artifact pages, PI deck (owned by sabatinilab-e3: coordinate).
   Conclusions that change direction / significance go to Priya before their text is rewritten. Captions are rewritten
   clean (no CORRECTION notes). Page-text items B9-B14 (section 7 takeaway, PS93 / DAQ pre-cue caveats, 12.1b, 12.1 caption,
   Fig 2.4 caption) are done in the page pass. Remove files of cut figures from the goal page to free slots.
8. Also on corrected data: RESTnl level over epochs (across + within session, per animal), PS92 subacute per-session check.
9. Quiet-event swap guard: shrink from 0.5 s to +/-1 pair on swap-masked data (quiet window's code; coordinate).

**Running / done (helpers in this session; they never commit):**
- Frozen-store pass 2 (wf_frozendef, analysis/frozen-default; doc FROZEN_ALL_20261102.md): converting the last read-outs; told
  to switch the LP default to LP-trained and to list kinds whose spec misses the variant / pre set. Commits so far up to 1553db6a.
- Swap-mask re-run (11-02/03) DONE: no headline flips; results in COHORT1_SWAPMASK_RERUN_20261102.md Log; code on
  analysis/cohort1-swapmask (wf_swapmask: b334316a, 4813c3cf, 3cd00814) + wf_swapmask_enc e80dbef7.
- Drift cutoff check RUNNING (wf_swapmask, doc DRIFT_CUTOFF_CHECK_20261103.md, outputs grant_figures/drift_cutoff_check_20261103):
  block durations, spout-order check, residual band power, block-protected removal at 10/15/20/30 min, replacement arm.
- Inventories DONE: docs/status/rebuild_20261103/ (A outputs->code 5,536 rows; B code->outputs 548 rows + traps).
  Reconciler RUNNING -> MASTER_MANIFEST.csv + RECONCILIATION.md (commit them when it reports).

**Open for Priya:** per-session LocaNMF families (recommend: project corrected data onto existing per-session footprints);
preprocessing decks (935 images), stale 10-24 analysis decks, imaging section 15 (copy of goal 23), closed 10-27 review
page (reconciler will bring recommendations); the drift cutoff (after the check).

**Next:** reconciler report -> commit + review decisions with Priya -> drift check -> code changes on a branch off main
(single hemo switch driving hemo_variants.ADOPTED + defaults.yaml + load_imaging; project_all; output-root overrides;
session-cache prune guard; camera working; Aug-only pre; LP-trained) with tests -> renders into NEW folders, logged ->
2 verifier agents -> replacement pass. Traps: rebuild_20261103/rebuild_order_B.md + RECONCILIATION.md.

**11-03 AFTERNOON UPDATE (supersedes the Running / Open / Next lines above):**
- **More decisions by Priya** (items 10-16 extend the list above):
  - 10. ONE trial classification for everything: wfield_local.trial_classes (e3 owns; window "cue_stop" default, whole-trial
    option). "working" = contact | incomplete | jaw_only, "camera_lick" = contact | incomplete, "attempted" (GATES). Applied
    EVERYWHERE a trial selection is made (behavior / kinematics too), but NOT as a gate where engagement is the measured
    quantity (quit, engagement decline, attempt rate). Cache keys / specs carry trial_classes.selection_tag(source, window).
  - 11. PS93_0813 (no video) kept wherever an analysis needs only spout contact; drops only where camera classes are needed.
    PS92_0908: 17.5% of cues "unknown" (in neither gate).
  - 12. Frozen-model IDs carry a TRAINING-DATA FINGERPRINT (hash of X / y + selection_tag + hemo variant name). Reason: specs
    were blind to trial/behaviour changes; PS92 section G models of 11-02 21:23 no longer reproduce with an identical spec.
  - 13. Preprocessing decks: REGENERATE the map slides (maps rebuilt anyway for 15e; new folders, never in place: the
    affine8v1 tag has no hemo tag). Analysis decks regenerated. Imaging section 15 -> pointer to goal 23. 10-27 review page
    and June alignment decks untouched. Imaging 5.1/5.2 + goal 6.x/8.x LP-trained. Lost scripts recreated (3 recovered,
    scripts/recovered_20261103/).
  - 14. Once the up-front drift step is in the shared signal, the encoder / ITI / stray_licks OWN drift steps are removed
    (kept as a sensitivity row).
  - 15. e3's consolidation A1 (per-session response window), A2 (jaw_pass_qc -> classes), A3 (animal-weighted change-vs-pre),
    A4 (trial_attempts on trial_classes: LANDED main 28f2a632; 62,746 / 62,771 trials identical) and A5 (one responded(),
    source camera default / daq) land on main BEFORE the rebuild renders. e3 does not touch the rebuild's reserved files
    (list in RUNNING_LIST, 11-03).
  - 16. LocaNMF: test before deciding refit vs projection (see below).
- **Done:**
  - Frozen-store pass 2 (wf_frozendef analysis/frozen-default up to 876c87b0; FROZEN_ALL_20261102.md P2.1-P2.12): every family
    identical to in-memory; the LP arm defaults to LP-trained. Section G conversion is UNCOMMITTED in wf_frozendef
    (poststroke_compare.py + tests; tests were re-running) because of the PS92 mismatch: commit it with that note, or fold it
    into the rebuild. frozen_encoder not converted (plan P2.6). Bug to fix in the rebuild: nolick_decoder KeyError on arms with
    < 30 trials.
  - Reconciliation committed: docs/status/rebuild_20261103/MASTER_MANIFEST.csv (6,099 rows: 3,415 REDO / 930 NO / 1,754
    DECISION, now mostly resolved by Priya) + RECONCILIATION.md (render order G0-G8, 32 traps, decisions D0-D12).
- **Not caught by either inventory, now in the plan:**
  - the FINAL variant does not exist yet (new drift step), so the 11-02 swapmask outputs are a comparison arm only;
  - no imaging family has a camera-gate path yet;
  - deck captions in deck_registry.py / deck_text.py state the old rules.
- **RUNNING (helpers; they never commit in wf_attempt):**
  - Drift cutoff check (wf_swapmask, DRIFT_CUTOFF_CHECK_20261103.md): block lengths per animal x epoch, spout order,
    residual band power, block-protected cosine at 10/15/20/30 min, plus the replacement arm (new drift step in place of
    the polynomial).
  - LocaNMF refit test (wf_swapmask, scripts/locanmf_refit_test/, LOCANMF_REFIT_TEST_20261103.md): about 8 sessions
    refit vs projection, same-machine stability, one joint basis timed in a separate dir.
  Both report to me; I bring Priya per-animal tables for 2 decisions: the drift cutoff, and refit vs project.
- **Next:**
  1. Drift decision -> build the final variant (about 1 h).
  2. LocaNMF decision.
  3. Wait for A1-A5 on main.
  4. Merge the branches into main.
  5. Code changes on a branch off main, with tests:
     - single hemo switch; project_all / basis selection; output-root tags;
     - session-cache prune guard; selection_tag + fingerprint in make_spec;
     - camera working in _class_select; Aug-only pre per decoder block; precue_engagement_states;
     - map-digest narrowing; drop the legacy date shim; nolick bug; encoder / ITI drift removal.
  6. Renders G0-G8 into NEW folders, logged.
  7. 2 verifier agents.
  8. Replacement pass (share, decks, pages, PI deck with e3; page text B9-B14).
  Estimated about 1.5-2 days of compute, plus O2 for the encoder bundle.

### 00-COMPACT-19 (11-01 ~23:00, Priya asleep, autonomous) -- READ FIRST, then docs/status/RUNNING_LIST_2026-11-01.md
Same session f082ee0c / worktree wf_attempt / branch analysis/attempt-coding. Goal page LIVE **v305** (section 23 = quiet window, merge their updates before publishing); imaging page (repo
widefield_pipeline main, source docs/results_pages/imaging/, publish from repo HTML; must READ the live version first) LIVE
**v39**. Today: C: disk cleanup (lists docs/status/DELETED_LOCAL_*_2026-11-01.txt); lick-source tags on every figure title
(both pages; tables docs/status/GOAL_PAGE_LICK_SOURCE_20261101.tsv, imaging provenance/lick_source.tsv); 14.21 S1/S2/S11-S13
captions on RESTnl (the acute far-contra pre-cue raise is NOT orofacial-specific under RESTnl); S12 stars in the panel
(wfield_local/sig_marks.star_top + reserve_headroom now also in this branch). Plans: RESTNL_ROLLOUT_PLAN_20261101.md (Priya
"ok" to my recommendations: RESTnl primary, old baselines collapsed sensitivity, flat RESTnl for maps, August-only pre,
wr rule, rest decoder 12.1 lick-free first). Running: engagement-gate vs LP-attempt check -> GATE_LP_CHECK_20261101.md.
Rules unchanged (helpers: exact PIDs only, never commit; <= 4 workers; another session commits on widefield_pipeline main ->
branch off main for renderer code, never rebase / force-push).
**11-02 03:05 update:** goal page live v306 (quiet window). DONE: gate check, LP gate option (dea1de5f), 14.21 + S7 on
RESTnl, RESTnl MAPS (branch analysis/restnl-maps 00f1e2c2 in worktree wf_restnl; results RESTNL_MAPS_20261101.md; pre-cue
far-contra rise now 4/4; PS94 late flat-baseline caveat). Priya 11-01: "ensure we discuss before you take anything off the
page or replace headline results" -> ADD only. RUNNING (helpers): imaging-page RESTnl additions (b-suffix figs); FINAL LP
ARM render (new worktree, branch analysis/lp-final, doc LP_FINAL_ARM_20261102.md there). On their reports: commit their
work (helpers never commit), log in the running list, then add LP-final figures to the pages (add only). Two decisions
for Priya at the top of the running list (prm cache digest on main; flat RESTnl for PS94 late).
**11-02 06:30 update:** imaging page LIVE v44 (v43 RESTnl twins 7.xb/8.xb; v44 Final LP arm block #lpfinal Figs 2.5-2.8);
page source committed on analysis/restnl-maps (wf_restnl). Goal page LIVE v308 (quiet window), merged locally. Final LP arm
committed on analysis/lp-final (wf_lpfinal, 5822cbad): no headline change. Fig 2.4 caption error (decoder refit in memory,
not the stored DAQ model) -> correction note WAITS for Priya. RUNNING: rest decoder 12.1 lick-free (helper, branch
analysis/rest-decoder-nl). Decisions for Priya: top of RUNNING_LIST_2026-11-01.md.

### 00-COMPACT-18 (10-30 ~late night) -- (previous; then the running list; -17 for the 10-29 state)
Same session f082ee0c / worktree wf_attempt / branch analysis/attempt-coding. Page LIVE **v292**
https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (local source `_goal_page/index.html`, images gitignored -> publish via the
`files` map; supplement anchors `id="s14-18-N"`). Last commits ccfae557 (page/figure review fixes + FINDINGS 4.23 + S25),
8aa75cab (running list). **NOTHING RUNNING.** Running list = `docs/status/RUNNING_LIST_2026-10-29.md` (10-30 entries at the end).
DONE 10-30 (each pre-registered, Outcome section filled; FINDINGS §4 items; page 14.18 supplements):
- ITPC / coherence (4.17): ITPC rise restates the larger lick response; GCaMP6s kinetics caveat (LTI deconvolution leaves
  ITPC / coherence unchanged by construction) -> S19 DEMOTED (Priya OK). CCA gate failed; no DLAG.
- Direction-information timing S vs M (4.18, S20; DIRECTION_INFO_LAG_PREREG): pre motor leads ~23 ms; chronic ~10 ms, 4/4.
- Literature deep dive: `docs/RECOVERY_MECHANISMS_LIT_20261030.md` (predictions vs results table; pathway routes A-D,
  route D 'ALM -> bilateral premotor' UNVERIFIED (Priya asked it as a question); ranked analyses; cohort-2 perturbation
  logic: direction x epoch interaction, contralesional activation, ~3-week viral lead time).
- Kinematic normality (4.19, S22a-e): hits still less normal chronically (-0.10, 4/4; PS94 most); median deviation raised,
  95th pct back; recovery-score figure all 4 animals (`scripts/kn_hits_deviation.py`).
- Re-embedding (4.20, S23): alignment failed gate G1 -> NOT ESTABLISHED; half the chronic frozen-refit gap is drift-sized.
- Within-lick correction (4.21, S21a-d): 'the brain fixes the LAUNCH command itself rather than adding feedback
  corrections' (Priya likes this wording); pre-stroke CSM validation reproduces Bollu 2021 (S21a).
- Learned compensation (4.22, S24): no within-session learning (gains overnight), no ITI carry-over; hit rate FALLS within
  post sessions; ITI bias = cued landing error (NO gap -- Priya: important; FINDINGS §0 4b note).
- Pre-launch hemispheric tuning (4.23, S25): gate failed 0/4 at AREA level; contraversive pattern only post-contact.
FIGURE DEFAULTS (Priya 10-30; memory results-page-conventions + docs/templates/README.md): sessions = OPEN markers, shape =
animal (PS92 o, PS93 s, PS94 ^, PS95 D), color = epoch (EPOCH_LINE); pooled = FILLED epoch-color marker; June gray; NO
figure numbers inside images (panel letters OK); NO connecting lines between epochs; distributions = faint session curves
+ thick pooled, each epoch's own 50 % / 95 % marked.
UNREVIEWED BY PRIYA (she said 10-30 late: keep these in the handoff until she reads them):
- My answer to her 10-30 figure review (fixes applied in page v292): Bollu signatures = S21a; no-gap result bolded (S24b,
  FINDINGS §0 4b); P(pre) explained in S22a; hit rate worsens within sessions, aim does not (S24a); the two pink acute dots
  in S24b panel A = close vs far mid (now split open/filled); S2 / S11 / S16 were always text-only; S3 overlaps the offset
  analyses (recommend collapsing); S25 gate failed.
- My answer 'why area pools, not SVD within ALM / tjM1' (item 1 below) -> Priya then said RUN IT (10-30 late; in progress).
OPEN (Priya to decide; asked, no answer yet):
1. **NEW 10-30: pixel / SVD version of S25** -- Priya asked why area pools rather than SVD within ALM / tjM1 (Allen masks).
   Answer given: my choice (trial counts, hemisphere comparability), not principled; area averaging may cancel a spatial
   pattern. Proposed add-on prereg: mirrored ALM + tjM1 masks per hemisphere, SVTcorr projected on masked U_atlas, top
   components; one FROZEN decoder per animal x hemisphere trained on pooled August pre sessions (load_or_fit), scored on
   post; same pre gate (>= 3/4 animals, within-spout CV vs shuffle); post-contact window as positive control. Limits either
   way: launch noise 5-8 deg / lick, GCaMP6s over a 250 ms window. GO given 10-30 late -> prereg
   docs/status/PRELICK_SVD_TUNING_PREREG_20261030.md + scripts/prelick_svd_tuning.py (helper).
   DONE 10-30 late: POST positive control FAILED (2/4), PRE nominal 3/4 ipsilesional-only r ~0.05 -> NOT ESTABLISHED;
   no post change; FINDINGS 4.23 addendum, page S25b (v293). UNREVIEWED by Priya. Open check: `_cv_r` negative bias.
1b. **DONE 10-31 (FINDINGS 4.24, page S26 v294; UNREVIEWED): no pre-launch tracking, inconclusive by rule; post-contact
   tracks; FaceRhythm pre-launch tracks the deviated launch (0.49, 4/4).** Was: pre-launch AIM-TRACKING test (Priya GO): frozen across-spout angle decoder (pooled August pre) in
   the pre-launch window (0.05 s bins, -0.50..-0.05 s) -- at the same target, does the decoded angle follow the deviated
   launch (aim in cortex) or stay at the target (as pre-cue did)? CUED + ITI (docked, pre-lick, dynamic) arms; Allen /
   whole-brain SVD / LocaNMF. Two-stage helper (prereg first, I commit, then run). Context given to Priya (UNREVIEWED):
   within-spout angle IS decodable weakly AFTER contact (4d0g slope +0.17-0.23 4/4; S25 slopes D -0.55 4/4; pixel POST
   2/4), never before launch; across-spout angle decodes but mostly as spout identity.
1c. **DONE 10-31 (FINDINGS 4.25, page S27; UNREVIEWED): face tracks aim only after movement starts; ITI offset after FR partialling retained 0.78-0.82 (PARTIAL; PS92 face-carried, PS94/95 kept). Open from prereg deviations: dose control, lick-level history regression, intersection lick set not computed.** Was: FaceRhythm pre-launch aim follow-up (Priya GO; prereg docs/status/FACE_PRELAUNCH_AIM_PREREG_20261031.md
   expected, script scripts/face_prelaunch_aim.py): which face factor / jaw measure carries the pre-launch angle + deviation,
   timing vs jaw onset, recovery time course, within-spout pre, and Q5 = FR partialled out of cortex -> does the frozen-DECODER
   OFFSET change? (Answer given to Priya, UNREVIEWED: FR partialling was done for the SSp-n L/R term (unchanged) and inside
   the frozen encoder, but NOT on the decoder offset itself.)
1d. **DONE 10-31 (FINDINGS 0b disengagement line, page S28; UNREVIEWED): not satiety (stop after -290 ul less water, 4/4), no failure-specific frustration and no tongue slowing before quits (narrow tests; broader frustration / fatigue NOT ruled out, 11-01 wording fix); quit state = running, still cortex BELOW engaged (post hoc). Primary epoch early post chosen by me (unreviewed). Addendum DONE (S28d2-g): water shortfall partly session length (80-min -105, 3/4); running-regressed level -7.8 4/4 confirms; self-activation not supported. FOR PRIYA: check session-end classes (a0_end_class.csv); P5t trials cap 300 uninformative -> report uncapped with bounds or drop?** Was: disengagement tests 1-4 (Priya GO): 1 satiety (water before quit), 2 frustration / local reward
   rate threshold, 3 motor fatigue predicting the quit, 4 state of the quit period (face / running / cortical pre-cue level +
   slow fluctuations; pupil availability check). Two-stage helper; prereg docs/status/DISENGAGEMENT_PREREG_20261031.md,
   script scripts/disengagement_tests.py, outputs E/disengagement/. Plus a licks-per-responded-trial figure (claim had none).
   UNREVIEWED answers given 10-31: differential compensation (lesion-group pattern, PS94 least recovered, PS92 offset
   face-carried; per-animal profile table proposed, no answer yet); neural correlate of recovery (orofacial post-lick
   balance: cued marker largely landing error via target; ITI shift genuine within-animal, partial; post-movement state,
   no lead); figure sources for "compensate not restore" (S21, FR 5d / items 2,4,6, OROFACIAL item 7).
2. Figures for text-only S2 / S11 / S16? 3. Collapse S3 into the offset section? 4. Publish reembed frozen refs (local
   store, kind `reembed`) to MICROSCOPE? 5. New re-embedding prereg with spout-resolving label-free alignment?
6. Cohort-2 design (longer pre baseline; perturbation arm) -- viral lead time means decide soon.

### 00-COMPACT-17 (10-29 ~03:00) -- (previous; then -16 / -15 for the balance definition and decomposition, -12 env)
Same session / worktree / branch; page LIVE v223 https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d; last commit adf258ee+.
DONE this stretch (all on the page + FINDINGS 0b):
- 'Biology in brief' section at the top (7 cards, hypothesis line). 14.18 primary image = UNCAPPED first licks
  (lick_recovery_primary_full.png; capped version collapsed under it).
- SSp-n checks (lick_sspn_checks.py; 12.17 supplement): decoder WITHOUT SSp-n still tracks recovery but weaker (largest
  single contributor, not necessary); WEIGHT-FREE left/right difference: SSp-n #1-2 of 21 areas but SSp-ul, SSp-un,
  MO-rest, SSp-m, tjM1, ACAd close -> 'broad sensorimotor left/right imbalance, strongest in SSp-n' (hypothesis wording
  updated in WTS, Biology in brief, FINDINGS); FaceRhythm partialled: unchanged. ACAd = midline strip -> needs vessel /
  registration check before being called.
- Contact vs miss (lick_balance_contact.py; 14.18 supplement): balance as large or larger on MISS licks, tracks recovery
  as well (0.66 both) -> not mainly touch; strict misses lose the link in PS92/95.
- Imaging control on the primary balance (lick_balance_imaging_control_primary.py; first image of the imaging-control
  figure): F0 asymmetry wrong sign for an artifact; 415 nm flat; partialling F0 leaves cued 0.42 / ITI 0.27 cvR2.
- SEVERITY CORRECTED (Priya): phenotype PS94 most severe, PS95 least; histology tissue loss PS93 55 ~ PS92 53 > PS94 32 >
  PS95 18 % (PS92/93 lesioned twice). Response-size wording now by LESION GROUP; WTS row 'Lesion size vs phenotype':
  acute bilateral increase + acute balance follow behavioral severity (PS94), chronic growth follows tissue loss.
RUNNING at this note: helper agent -> scripts/lick_enl_angle_decode.py (Priya: "decode lick angle from ENL?": pre-cue
lick-free window -> upcoming first-lick angle, raw and BEYOND the spout; frozen post-stroke read-out + offset); outputs
<OUT_W>/enl_angle_decode*.csv, figures/lick_enl_angle_decode.png -> report, add to page (section 14 near 14.21 pre-cue).
NEXT: cued-hit balance for the strict-miss subset question.
ADDED 10-29 (later): ACAd midline control DONE (scripts/lick_acad_midline_control.py; 12.17 supplement; FINDINGS 0b):
undetermined, probably not ACAd-specific (link lives on the sagittal-sinus strip, shares variance with MOs-rest /
SSp-n; registration ruled out) -> not a grant candidate region. Grant discussion with Priya (chemogenetics): balance =
secondary readout next to behavior, measured off-drug and without the manipulated area; priority SSp-m/n both
hemispheres, contra tjM1, ALM lower; bilateral arm as specificity control; VIS/RSP as control region.
ALL 10-29 NIGHT HELPERS DONE (page v234; full state in docs/status/RUNNING_LIST_2026-10-29.md -- decisions needed
from Priya at its top). New: pre-cue target test (lick_enl_target_test.py: target code DEGRADED but NOT re-aimed;
landing signature only after the cue; precue attempt_coding_pp outputs generated with approval; its run.json was
rewritten to list lick+cue+precue, original backed up in the session scratchpad); rev-up behavior (revup_behaviour.py:
wait, not aborted attempts) and neural (revup_neural.py: delayed start, not slower ramp; acute far-contra overshoot;
late-lick engagement-definition conflict open). Page artifact hit the 512-file cap: 75 unreferenced images removed.
DONE 10-29 night: target model (14.18 supp), matched error (14.18 supp; cued within-spout balance overshoots pre;
grant note), ITI pre-lick (12.17 supp; lick-onset-locked; no ITI spout contact), ENL follow-ups (14.21 supp; CAVEAT:
'pre-cue reads the target' NOT established -- decoder compression, far-contra moves in 3/4), implied-direction display
(WFIELD_BAL_DISPLAY) + sign box, ACAd control (not ACAd-specific). Page v231.
GRANT NOTE (Priya 10-29): cued within-spout balance rises ABOVE pre over weeks (contralesional hemisphere gains
relative weight, independent of landing; FINDINGS 0b matched-error entry) -> chronic contralesional silencing tests
whether that extra share supports the recovered behavior.
TO-DO (Priya 10-29) 'REVVING UP' to far-contra: anecdotally post-stroke animals take much longer to start the first lick
to far-contra, sometimes starting after the spout leaves. Plan: (A) behavior -- cue -> first-lick-onset and -> first
contact latency distributions / hazard per spout x epoch, share of trials whose first lick starts after the spout
leaves (DAQ spout-move strobe); (B) kinematics between cue and first lick -- jaw onsets without tongue (incomplete
licks, cam1 jaw re-onset), sub-threshold tongue protrusions, FaceRhythm lick-vigor ramp; slow single ramp vs repeated
aborted attempts; (C) speed-accuracy: within session x spout, does a longer rev-up give better aim?; (D) neural --
cue- and first-lick-aligned ALM / tjM1 / SSp-m / SSp-n traces by latency quartile (ramp slope vs delayed start, to a
common level?), late (after-spout-left) licks vs normal / ITI licks; (E) does the raised pre-cue contralesional level
(14.21) predict latency?; (F) rev-up duration as a recovery marker vs the balance. Reuse: attempt-based deficit / time
to contact (10-21 encoding handoff), jaw onsets + re-onset (10-21 pose handoff), iti_licks first-lick times.
TIMING CAVEAT (Priya): GCaMP6s is slow (rise ~0.1-0.2 s, decay ~0.5-1 s; widefield smoother still) -- fluorescence
lags spikes, so a rise at -0.2 s means activity earlier; a step at 0-0.1 s after lick onset cannot separate a late
motor command from early self-generated feedback; post-lick decay leaks into the next window. Read every timing claim
(ITI pre-lick, rev-up ramps) with this; consider a deconvolution sensitivity check.
ENL angle decode DONE (scripts/lick_enl_angle_decode.py; 14.21 supplement; FINDINGS 0b): weak, mostly spout; tiny
within-spout part partly trial history; acute contralesional offset +1.55 deg 4/4, not sustained. OPEN: route its
pre decoders through frozen_models.load_or_fit (rule 10).

### 00-COMPACT-16 (10-29 ~01:30) -- (previous)
Same session / worktree / branch; page LIVE v218+ https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d. NOTHING RUNNING.
HYPOTHESIS (updated with Priya): FINDINGS 0b "HYPOTHESIS (updated 10-29)" + the WTS row of the same name.
NEW this stretch:
- 12.17 KEY = exact decomposition of the decoder's like-for-like offset into per-area left/right DIFFERENCE and SHARED
  parts (lick_offset_contrib_map.py now saves terms_sessions_<tag>.csv + terms_licks_<tag>.csv.gz; analysis
  lick_offset_terms_recovery.py -> offterm_*.csv). ITI offset ~70-75% hemispheric difference, SSp-n difference +0.34 /
  +0.20 (acute / chronic); SSp-n term tracks recovery within every animal (4/4, ITI + cued), not across animals
  (per-animal decoder scale). Old 12.18 (balance-explains-offset) and 12.20 (regions) collapsed in place.
- 14.14 = PRIMARY balance by lick order (lick_balance_order_primary.py, from balsub_order.csv written by
  lick_balance_subsets.py): cued acute ~-0.35 at orders 0-4, chronic ~0 (raw overshoot gone); raw order figure collapsed.
- Uncapped cued first licks: lick_firstlick_extract.py -> <OUT_W>/firstlick_full/<label>.npz (every first post-cue
  lick, ~476 / session vs ~138 in the capped decoder set; identical values on shared licks; shared session files NOT
  changed); WFIELD_BAL_FULL=1 in lick_balance_subsets / lick_recovery_primary (outputs *_full): within-animal r
  0.77/0.87/0.94/0.74, unseen animal 0.65. Not yet the page default (14.18 still capped; full in its robustness (vii)).
- Session pace (session_pace_table.py): acute excess of first licks = 1500-lick cap (N_TRIAL_MAX) x shorter bouts.
- Trial-by-trial (balance and SSp-n term): dominated by the normal direction code / decoder shrinkage (present pre) ->
  uninformative about cause; pre-lick balance predicts nothing.
- Page: section 12 and 14 renumbered by the inserts (refs fixed by script: collapsed-fig summaries regenerated from
  inner fnum; retired-group numbers kept as historical). Collapsed figures keep their <figure> tags -> renumber() counts
  them; after any insert, re-sync the collapsed summaries (see scratchpad page_fixrefs.py logic) .
OPEN / NEXT: (1) cued HIT vs MISS (contact) balance -- sensory-reafference vs motor test; (2) make the uncapped first-lick
set the 14.18 default?; (3) redo the imaging control (14.19, F0 / 415) on the primary balance; (4) 'Biology in brief'
section (list awaiting Priya); (5) collapse WTS-unreferenced supporting figures.

### 00-COMPACT-15 (10-29 ~00:30) -- (previous)
Same session / worktree / branch; page LIVE v213+ https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d. Decisions:
DECISIONS "2026-10-28 (late night)"; findings FINDINGS 0b (PRIMARY marker block + trial-by-trial).
PRIMARY BALANCE (fixed): angle-matched GAIN RATIO, log2(contra obs/pred) - log2(ipsi obs/pred), summed over the session's
licks and ALM + tjM1 + SSp-m + SSp-n + SSs, response 0.1..0.5 s after the FIRST lick minus -1..-0.5 s; cued = first
post-cue lick, ITI = first lick after >= 2 s quiet; pred = animal's PRE straight line on lick angle per area x hemisphere
x lick order (pre sessions leave-one-session-out). Code: scripts/lick_balance_subsets.py (per-lick features + session
sums; env WFIELD_BAL_REF=lateral, WFIELD_BAL_REFMETHOD=gauss, WFIELD_BAL_INSIDE=1, WFIELD_BAL_KIN=1, WFIELD_BAL_MO=1 ->
suffixed outputs; --sessions-only), scripts/lick_recovery_primary.py (models + figure; same env suffixes).
PAGE: 14.16 = HOW THE BALANCE IS CALCULATED (method box + area map + PS94 worked example, lick_balance_explainer.py);
14.17 = PRIMARY marker (KEY) with supplements: window/subset selection (lick_balance_nested.py, lick_balance_subsets.py),
trial-by-trial (lick_balance_trial.py), and the RETIRED earlier definition (old 14.18-14.20 folded into one collapsed
supplement, anchors kept). 12.18 first image = primary balance vs decoder offset (lick_balance_offset_primary.py).
14.14 / 14.15 relabelled as the RAW per-lick hemispheric difference. 14.20 = pre-cue level vs restw + traces
(lick_precue_restw.py, lick_precue_restw_traces.py; sustained acute raise at far contra, not a ramp).
RESULTS: cued unseen-animal R2 0.70 (days 0.23); 0.53 with only in-pre-range licks (no extrapolation); ITI 0.42 (days 0.17)
but 0.07 without extrapolation (within-animal link remains 3/4); Gaussian ref worse; lateral ref equivalent; lick count
cancels; equal area weights equivalent; tracks decoder offset (cued r -0.58..-0.92 4/4). Trial by trial: post-lick
balance follows the lick's direction r 0.3-0.5 incl. PRE (normal code); pre-lick balance r ~ 0 -> no pre-movement cause.
Biology hypothesis discussed with Priya: impairment = loss of normal between-hemisphere balance of orofacial cortex for a
movement (ipsilesional cortex over-responds acutely), recovery = re-balancing (not restoring size); evidence now favors
the balance as a post-movement STATE / readout rather than a pre-movement cause.
DONE after this note (page v216, 80f43e3e; NOTHING RUNNING): kinematics-matched balance -- cued survives (0.63), ITI
cross-animal fails; MO-rest adds nothing; acute excess of cued first licks = 1500-lick random cap in iti_licks.extract
(N_TRIAL_MAX), not behavior (session_pace_table.py) -- 'first-lick count tracks recovery' WITHDRAWN; balance explains ~65%
of the cued acute decoder offset but 5-15% of the ITI offset -> for ITI the decoder offset (0.39 unseen animal) is the
better marker. Page decluttered (goal_page_declutter.py: 110 figures collapsed IN PLACE, numbers unchanged, links open
them; 171 open). NEXT candidates: (i) re-extract cued first licks WITHOUT the 1500 cap (more precise cued marker);
(ii) per-area decomposition of the ITI decoder offset beyond the balance (within-hemisphere pattern); (iii) 'Biology in
brief' section (list proposed to Priya, awaiting OK); (iv) collapse supporting figures not referenced from WTS.
(was) RUNNING at this note: (1) task bxa9t3iuo: WFIELD_BAL_KIN=1 sessions + primary (test 2: reference on angle + reach + speed
+ elevation; outputs *_kin), then WFIELD_BAL_MO=1 full subsets (MOs_rest / MOp_rest added; balsub_*_mo) -- logs
S2/balsub_kin.log, recprim_kin.log, balsub_mo.log; (2) helper agent: scripts/session_pace_table.py -> session_pace.csv /
figures/session_pace.png (why acute sessions have MORE first post-cue licks: PS92 128/196/120/122 ... pre/acute/sub/chronic).
-> report both, add kin result to 14.17 robustness + WTS row, MO result to the subset supplement.
PENDING with Priya: (a) PAGE DECLUTTER plan (281 figures; proposal: per section, collapse all 'supplementary' tier and
method-only 'supporting' figures into one 'More figures' details; keep KEY + biology-relevant visible) -- awaiting her OK;
(b) WTS figure-number references elsewhere may be stale after the section-14 renumbering (links are fine; check
plain-text numbers); (c) spout-contact covariate for test 2 not available in covref (would need pose contact table).

### 00-COMPACT-14 (10-28 ~20:30) -- (previous)
Same session / worktree / branch; page LIVE v204 https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d. Same page-edit rules
(goal_page_insert + fix "CHECK plain mention" by hand; Artifact `edits` refused -> edit file, publish with files={...}).
Decisions: docs/DECISIONS.md "2026-10-28 (night)". Findings: FINDINGS 0b (recovery markers, pre-cue, attribution).
NEW / CHANGED on the page this stretch:
- 11.13 legend fixed; NEW 11.14 nested blocks + ladder + Shapley (scripts/lick_block_compare.py; main image =
  WFIELD_BLOCK_VARIANT=oroSSs, SSs in the orofacial group; 6-group split = supplement). oro+SSs ~40% of code and ITI offset.
- 14.16 gray pre cloud; NEW 14.17 KEY recovery markers (scripts/lick_recovery_markers.py): balance tracks recovery
  (cued LOAO 0.50, epoch-out 0.41 vs days 0.23), SIZE / AUC do not (sign flips with lesion severity PS94/95 vs PS92/93);
  per-area supplement incl. z-scored balance; ITI first-after-quiet balance 0.29. NEW 14.20 pre-cue level vs restw
  (scripts/lick_precue_restw.py, helper-built, validated vs cued_wide traces).
- 18.5-18.9 residual-ROI maps: per-area outlines, untested areas blanked (the 'fragments'), no frames.
- Non-linear decoder supplements on 12.1 / 12.15 / 12.21 / 13.1 / 13.6. Musall split-row sentences on 0.3 / 0.4 / 0.9 /
  0.10 / 0.11 (claims_contest / claims_peth / claims_trend_levels; the old "tjM1 r 0.09" note could not be traced and was
  not used; per-animal r tjM1 contra vs MOp contra movement PETH -0.16).
- FINDINGS 0b + "Where things stand" recovery row have the bounds (within-animal days 0.70-0.81; LOSO animal-known days wins).
DONE after this note (page v205): fair weights fail across animals (cued LOAO -0.08; own-animal 0.04; plain average 0.50); ITI recovery axis: ITI licks shift ipsilesionally and recover (mean angle -12.6 / -6.0 / -3.8 deg, r with cued aim 0.77-0.95), but only SSp-n first-after-quiet balance predicts it in an unseen animal (0.32 vs days 0.17) -> 14.17 caption + ITI-axis supplement. Nothing running.
(was) RUNNING at handoff: scripts/lick_recovery_markers.py full run (task b7rvq4dsk; log S2/recmark.log) adding (a) FAIR pre-fit
weighted balance oro_balwx (weights from the OTHER animals' pre licks) and (b) ITI's own recovery axis (iti_ang = session
mean ITI lick angle vs pre, iti_ang90 = contralesional 90th pct; new figure lick_recovery_markers_iti_axis.png). When it
ends: read recmark_cv.csv (y column: aim / iti_ang / iti_ang90), copy lick_recovery_markers_{cv,size,areas,iti_axis}.png +
lick_balance_recovery.png to _goal_page, add the ITI-axis image as a supplement of 14.17 and the oro_balwx numbers to its
caption, publish, commit. Priya: the ITI axis "asks a different question" -- say so in the caption.
OPEN: PS93 acute MOp / MOp-rest movement-PETH values +0.43..+0.92 (0.4 caption) -- check for an R2 artifact; the
--models-only path of lick_recovery_markers does not know the ITI axes (run the full script).

### 00-COMPACT-13 (10-28 ~15:45) -- (previous)
Session f082ee0c, worktree C:/Users/SabatiniLab/Github/wf_attempt, branch analysis/attempt-coding (commits through 01a944f1+).
Page LIVE v198 https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d. Env as 00-COMPACT-12. Page edits: scripts/goal_page_insert.py
(insert + renumber; fix the "CHECK plain mention" lines by hand); the Artifact `edits` option is REFUSED here (file-edit hook)
-> edit _goal_page/index.html with a .py script, then publish with files={...}. Bash heredocs with "\n" inside Python strings
corrupt files (happened ~6x today) -> use the Write / Edit tools for code. Backgrounded `( ... ) &` subshells LOSE
non-exported shell vars (logs went to "/x.log") -> literal absolute log paths.
DECISIONS today (all in docs/DECISIONS.md 2026-10-28 afternoon / evening / late):
- Response-AUC window = 0..1.0 s for ITI AND cued (Priya: one window, easier to defend); 0..0.5 s secondary. Response
  baseline -1.0..-0.5 s; PETH curves + build-up keep -1.5..-1.0 s. ITI 0..1 s AUCs come from the wide cued_wide traces
  (stored ITI PETHs end at +0.75 s).
- Contra/ipsi ratio figures: PER-SPOUT = KEY (14.9; far ipsi shifts, close ipsi not); by side = supporting (14.10); close vs
  far ring (14.11; far-close test per epoch); side traces collapsed (14.12). All-spouts pooling demoted (averages
  opposite-side ratios). Traces: mean over animals defined at each lag, dotted where < 4.
- Orofacial pool = ALM + tjM1 + SSp-m + SSp-n (SSp-un in NEITHER pool: intermediate, adding it worsens the fit).
- Area maps: each significant area outlined on its own (black on white halo), non-sig faded, the MOs/MOp edge strip
  lateral to ALM/tjM1 blanked -- it was a DISPLAY artifact (those pixels are in no ROI mean; checked) ->
  iti_area_maps.trim_rest, also used by encoder_split_rows.split_image (lick_offset_where / contrib / residual_roi redrawn).
RESULTS today (PROVISIONAL; FINDINGS 0b + "Where things stand" KEY group):
- Lick-evoked activity larger post-stroke in both hemispheres; cued far-contra acute = early decay (0.5-1.0 s below pre,
  survives vigor + landing regression, 14.8). Regression audit: all covariate adjustments correct; nulls real (12.16, 14.8,
  14.19 captions).
- HEMISPHERIC BALANCE (X_right - X_left, decoder 0..0.2 s window): pre-stroke tracks lick angle (each hemisphere favors
  contraversive licks); post ~30% flatter (2-3/4) + shifted ipsilesional at the same angle; per lick order: acute
  ipsilesional at every early lick, chronic overshoot contralesional licks 1-6 (14.13 / 14.14, angle+order reference).
  Session balance tracks the read-out offset (12.18, cued r -0.76..-0.95 4/4); ONLY orofacial balance carries it
  (12.19 head-to-head; non-oro adds nothing). Balance-blind decoders (12.20): offset unchanged with global / orofacial-pair
  / non-oro differences removed -> offset carried by the orofacial PATTERN, balance tracks but is not its sole carrier;
  VIS+RSP-only decoder r 0.15. Imaging control (14.17): F0 / 415 nm do not explain the subacute/chronic shift.
  Musall-style task vs movement maps of the balance (14.15, area + SVD pixel): lateral orofacial only; cued = spout,
  ITI = movement; acutely movement share halves.
- IMPORTANT (Priya): cued orofacial balance predicts an UNSEEN animal's recovery (far-contra first-lick aim) cvR2 0.50 vs
  days 0.23 (non-oro -0.19; shuffle null <= -0.12); 3/4 held-out animals; leave-one-session-out 0.63 vs 0.48 (animal
  unknown); with animal known days wins 0.83. Same-session link survives nonlinear time (per-animal splines p 0.006); no
  next-session prediction (14.16).
DONE after this note (page v199, 5ddf82a1): NEW 11.13 = decoder ALONE / UNIQUE maps (scripts/lick_area_unique_maps.py;
Priya's actual "Musall-style" request): orofacial block alone ~full (ITI r 0.59 vs 0.62), unique +0.062 ITI (4/4) / +0.036
cued; non-orofacial block alone ALSO decodes (0.53 / 0.71 -- adjacent S1 areas) and shows the offset -> code + offset
redundant across sensorimotor cortex; section 11 renumbered (weight-1sd 11.14 ... behavior-only 11.20). Nothing running.
Keep 14.15 (balance task/movement maps) unless Priya says drop it.
TODO / open: two tiny edge fragments may remain on the per-spout residual-ROI maps (check vs masks); residual_roi maps
still use the merged outline style; 14.8 pre-cue row (restw) still awaiting Priya; Musall split-row caption numbers;
orofacial area AUC vs recovery LMM (old item 9) partly superseded by the balance-recovery analysis.
New scripts today: lick_vigor_perspout, lick_lateral_ratio, lick_balance_{reference,angle_lmm,order,offset_lmm,regions,
maps,maps_svd,recovery,blind_decoder,imaging_control}, lick_area_unique_maps.

### 00-COMPACT-12 (10-28 ~07:20) -- READ FIRST. Session f082ee0c, branch analysis/attempt-coding in worktree
C:/Users/SabatiniLab/Github/wf_attempt (commits through 3483a064+). Page LIVE v165+ https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d
(source _goal_page/index.html, committed). Running log + decisions: docs/status/OVERNIGHT_20261027.md (10-28 morning
section) + docs/DECISIONS.md "2026-10-28". Run env: source S2/rr2/frozenenv.sh; PYTHONPATH="wf_attempt;wf_attempt/scripts";
python C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe; WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache.
Use the Write/Edit tools (or a .py file) for Python patches -- bash heredocs with nested quotes / "\n" broke 4x today.
Page edits that insert / move figures: scripts/goal_page_insert.py (insert + renumber section, links, fig_ids; prints
plain-text mentions to fix by hand -- earlier insertions shifted "Statistics as 14.x" etc. by hand).

DONE 10-28 (Priya reviewing the page live, many small requests):
- Key figure 1 = scripts/key_figure.py (A behavior = far-contra FIRST-lick aim; B/C frozen vs epoch-refit decoders + old
  far-contra code on hits; D ITI like-for-like offset any-spout + same-target). Key figure 2 = Fig 13.2 (lick order).
  "Where things stand" bottom line rewritten 10-28. Cued "read ahead of the failing movement" RETIRED = target
  composition (12.15), page + FINDINGS re-worded.
- 10.3 legend; 10.5 x = far-contra first-lick aim + pooled; 10.7/10.8 pooled pre-normalized column; 12.1 y label; 12.2
  Gaussian-smoothed curves (solid 4 animals, dashed 3); 12.15 panel titles; 12.17 pooled column; 13.7 NEW decoded vs
  actual position per spout (absolute; epochs-together + per-epoch images); 13.8 (decomposition) dots = mean of animals
  (were flat session mean), left column decoded (filled) vs landing (open) change; 13.9 legend; section 13 renumbered.
- ITI FaceRhythm controls on UNBLANKED FR (helper fr; 14.12/14.13 captions updated; old vs new CSVs). Shared write guard
  S2/f5/guard/sitecustomize.py is BROKEN (SyntaxError, never loaded) -- use S2/fr/guard or S2/mu/guard (fixed).
- Section 14 (lick-locked activity): cued versions (scripts/lick_cued_ramp.py: --extract re-samples -2..+3 s traces per
  session into iti_licks/cued_wide/<label>.npz, validated vs stored; first cued lick lick-/cue-aligned, ITI bout starts;
  row 3 = difference from the animal's own pre curve; window sensitivity 0-0.3/0.5/1.0/1.5 + peak; masked ratio; per
  spout). KEY block now 14.1 ITI (-2..+3), 14.2, 14.3 isolated, 14.4 cued lick, 14.5 cue, 14.6 per spout. 14.14 ITI area
  maps with ALM/tjM1 split + far-contra previous-spout version; 14.15 NEW cued area maps (all + far-contra target).
  iti_licks.extract now saves bout_prev_pos (bout_pos = window label, never used; no analysis affected).
- Deleted on M: iti_licks/sessions (v1) + superseded_split_20261027 (Priya OK); roi_v2_backup kept ~1 week.
- Rim check (ALM/tjM1 rim): no difference -> stat mask kept (scripts/checks/rim_check_alm_tjm1.py).

RUNNING / PENDING at compaction:
1. Musall + Layer 2 frA (helper mu code merged b6809c27; run S2/mu/run.sh with W=wf_attempt, task b4jttvnw8, ~1.25-1.5 h
   from ~06:55; logs S2/mu/run_*.log, ends MU_DONE). Then update captions 0.3, 0.4, 0.9, 0.10 (claims_*.csv) and 2.2 / 2.6
   (specificity_control/gate-attempted_v7g3_t3d_frA, layer2_split_boot.csv); copy PNGs, publish.
2. ITI per-spout traces by PREVIOUS spout (lick_cued_ramp.py compute, PID 29604 started 06:56, log S2/mk/wide_compute3.log)
   -> lick_iti_ramp_perspout.png -> add as 3rd image of 14.6 (+ caption).
3. 14.7 replacement: scripts/lick_iti_ramp_animals.py (animal-averaged ITI PETH -2..+3 s, RAMP_ROIS) running (task
   b5mfm6g95) -> put right after 14.1 as SUPPLEMENT, drop the old 14.7 (iti_ramp_peth) figure; renumber with goal_page_insert.
4. 14.15 (cued area maps) -> Priya: "include a pre-cue also" (add row: pre-cue AUC -0.5..0 s before the cue on first_cue,
   cue-aligned baseline -1.0..-0.5; scripts/iti_area_maps.py session_table_split cued branch) and "14.15 is KEY" (move into
   the section-14 key block after the per-spout figure, tier key, renumber).
5. Per-spout cued lick-response table (task b27ty861g; per spout tjM1_right / cortex change vs pre) -> to answer Priya's
   reading below.
6. 10.6 (scripts/iti_lick_rate.py, pooled + over days + vs first-lick aim) PID 18132 running since 05:43 (5000-perm x 2
   behavior axes); when done copy lick_iti_rate.png, update 10.6 caption (numbers in iti_lick_rate_recovery.csv,
   behavior column fc_aim vs hit), publish.
7. Priya's synthesis to CHECK then write into "Where things stand" + FINDINGS: (a) lick-aligned activity rises
   dramatically across cortex acutely; ITI: decreases with time (14.14: 41 areas acute -> 5 chronic: tjM1, MOp-rest contra,
   SSp-m), cued: persists chronically (14.15: 35 areas chronic); (b) cued: mostly from close spouts, least far spouts
   (verify with item 5; far-contra target acute lick response NOT up, n.s.); (c) "less orofacial activity acutely, above
   pre subacute-chronic" = true for far-contra target trials (acute cue response LOWER, ALM contra -3.4, 27 areas down;
   lick response n.s.), NOT for all cued trials (acute lick response up +5.0 tjM1 contra). Cue response: unchanged
   acutely, up subacute/chronic.
9. Priya 10-28 ~07:25: "plot orofacial regional AUC vs recovery index (and LMMs)". The ITI key-ROI response is already in
   the recovery link / mixed models (section 16, e.g. 16.12 key-ROI response vs first-lick aim; scripts/iti_recovery_link.py,
   iti_mixed_models.py). TO DO: per session, orofacial area AUCs (tjM1, ALM, MOp-rest, MOs-rest, SSp-m; contra / ipsi) for
   ITI bout starts AND the first cued lick (lick response, cue response) vs the far-contra first-lick aim, within animal
   (scatter + r, log(day) removed) + LMM (response ~ aim deficit + log(day) + (1|animal), as iti_mixed_models behavior
   model); sources area_auc_sessions_split*.csv / cued_ramp_auc_sessions.csv, aim = recovery_measure_sessions.csv.
10. Baseline sensitivity (Priya OK 10-28: keep current baselines, add -1.0..-0.5 s as a check): lick_cued_ramp.py
    --baseline-sens (task bm8akbebt) -> ramp_baseline_sens.csv + figures/lick_ramp_baseline_sens.png -> add as an image /
    caption note of the window-sensitivity figure (14.10). Recovery analysis (item 9): AUC 0-0.5 s primary + 0-1.5 s
    (also on ISOLATED ITI licks, to tell a slower decay from more licks); pre-cue level needs the RESTW baseline (Priya:
    an in-trial baseline already carries the pre-cue state) -> per-session pass with the restw mask, then the 14.15
    pre-cue row (Priya agreed in principle; confirm restw before running).
    10b. Priya: "add the 0-1.5 check" -> --baseline-sens now also has ISOLATED ITI licks (iti_iso); RERUN it after the
    first run (task bm8akbebt, without iso) finishes: python scripts/lick_cued_ramp.py --baseline-sens.
11. Page v166: NEW 11.11 = LATERAL offset contribution maps (lick_{iti,cued}_offset_contrib_map.png, all bases);
    section 11 renumbered (angle 11.12, reach 11.13, angle-vs-lateral 11.14, NON-LINEAR 11.15, behavior-only 11.16).
    Priya: DROP the unmasked LocaNMF row (lick_offset_contrib_map.py TAGS: remove 'locanmfF', keep locanmfFM) and
    redraw --plot-only; mask the drawn maps to the atlas/brain (LocaNMF |contribution| outside the atlas is only 2-3 %, the
    'olfactory bulb' look is footprints of kept components extending past the mask). BASES DISAGREE on WHERE: mean-of-
    animal contribution maps, area-sum r split-areas vs SVD 0.08 / 0.09 (ITI acute / chronic), vs masked LocaNMF 0.36 /
    0.36, SVD vs LocaNMF 0.20 / 0.03 (cued -0.23..0.42) although the OFFSET totals agree (ITI +0.39 / +0.34 / +0.40 acute).
    -> say in 11.11-11.14 captions + FINDINGS: the offset is basis-robust, its spatial attribution is NOT (ridge weights
    are not unique with correlated features; each basis spreads them differently). Pixel-level 'where' = the per-session
    pattern maps (section 15), not these.
12. Non-linear decoder follow-ups (Priya "sure do the 11.14 run and caption correction" = now 11.15): table
    lick_order_preds_nonlinear.csv.gz built by scripts/lick_nonlinear_preds.py (linear column reproduces KEY dec, max
    0.03 mm); WFIELD_LICK_DECODER=nonlinear runs of lick_matched_readout --ref gauss, iti_lick_order, lick_same_target_place
    (task b1p5excn3, S2/run_nl.sh, logs S2/mk/nl_*.log) -> add NON-LINEAR versions of 12.2 (lick_matched_gauss_nonlinear_
    curves.png) and 13.2 (lick_order_iti_cued_slope_nonlinear.png) next to their linear figures; same-target result
    (sametarget_*_nonlinear) into the 11.15 caption. Shrinkage check DONE: cued extra non-linear offset is ~2x larger for
    licks landing where few pre licks landed (lowest tercile +0.08 / +0.12 / +0.06 vs highest +0.04 / +0.06 / +0.03 mm,
    acute / sub / chronic) -> part kernel shrinkage, not all. 11.15 caption: cued offset LARGER non-linear (+0.06 / +0.08
    / +0.05 mm, 4/4), subacute +0.11 [0.001, 0.22] 3/4 where linear is +0.03; angle +2.2 deg subacute 4/4.
13. (10-28 ~07:45) DONE: 11.11 main 4th column = decoder weight PER AREA (sum of pixel weights), per-pixel weights as
    supplement images; unmasked LocaNMF row removed; non-linear 12.2 / 13.2 as 2nd images of 12.2 / 13.2; 11.15 caption
    corrected (cued offset larger non-linear, partly shrinkage, same-target +0.055 acute -> still composition).
    RUNNING: (a) SVD first-50-component maps (task bpkvm2wd3: WFIELD_LICK_DECODER=svd WFIELD_CONTRIB_KMAX=50
    lick_offset_contrib_map.py, then --plot-only) -> new rows 'SVD first 50 components' + 'SVD summed per Allen area
    (ALM/tjM1)' (Priya asked for both) -> copy lick_{iti,cued}_offset_contrib_map.png, caption: share of the offset in
    components 1-50 (summary_svdF_k50.csv share_of_offset; exact for an orthonormal basis), publish. (b) Drop-one-area
    importance scripts/lick_area_importance.py: first run (task btyd4o8fp) uses the OLD groups -> RERUN after it
    finishes (groups now incl. ALM+tjM1, SSp-m+n, ALM+tjM1+SSp-m+n, MOs_rest, MOp_rest, VIS, RSP each ipsi / contra /
    bilateral; Priya 10-28) -> new figure in section 11 after 11.11 (lick_area_importance.png) via goal_page_insert.
    Priya: the SVD per-area weights look as expected for ALM / tjM1 (strong frontal weights, opposite signs across
    hemispheres); LocaNMF maps look vessel-contaminated (queued idea: vessel-share per component + refit without them,
    not yet approved).
14. (10-28 ~08:20) DONE: Musall + Layer 2 run (MU_DONE): figures 0.3/0.4/0.9/0.10/0.11 (frozen_claims_*.png) and 2.2/2.6
    (spec_groups.png / spec_area_maps.png <- specificity_control/.../figures/specificity_*.png) published v172 with notes;
    Layer 2 chronic tjM1 +0.075/+0.073 (largest motor area), ALM +0.05, orofacial = control (+0.053 / +0.052).
    PENDING: a detailed read of the Musall split rows (claims_contest / claims_peth: ALM tracks MOs r~0.9, tjM1 contra differs
    from MOp r 0.09 PETH) -> proper caption numbers. 10.6 published (ITI rate vs aim r +0.56, detrended +0.26 p 0.03).
    Importance figure = NEW 11.12 (section 11 renumbered: angle 11.13, reach 11.14, angle-vs-lateral 11.15, NON-LINEAR
    11.16, behavior-only 11.17): redundant coding; only SSp-n alone and the ALM+tjM1+SSp-m+n block matter (-0.062 r ITI).
    STILL RUNNING / TODO: items 2 (ITI per-spout traces), 3 (animal-averaged 14.7), 4 (14.15 pre-cue restw + key), 5
    (per-spout table), 7 (Priya synthesis -> Where things stand), 9 (orofacial AUC vs recovery + LMM), 10b (baseline
    sensitivity rerun with iso).
15. (10-28 ~08:30) Importance detail (11.12): ITI like-for-like offset with any block dropped changes by <= +-0.04 mm
    (full +0.39 / +0.24 / +0.29; extremes MOp_rest -0.039 acute, ALM+tjM1+SSp-m+n ipsi +0.032 acute, full block
    bilateral -0.033 / -0.037) -> the ITI offset is not carried by any single area or block. Cued offset moves more (up to
    +0.10 with the ipsi block acute; full +0.17), consistent with target composition. Priya asked why the SVD per-area
    weights look more 'biological' than the area basis: SVD = smooth high-variance components (implicit smoothness prior,
    orthogonal, well-determined weights); area basis = ~45 collinear z-scored features (ridge shares weight arbitrarily;
    small edge areas can get large coefs); AND per-area SUM of a smooth pixel map grows with area size (MOs / ALM big).
    PROPOSED (not yet approved): basis-comparable 'effective weight' per area = decoder change for a 1-SD fluctuation of the
    area mean (saved weights x per-session area SD), redraw 11.11 column 4.
16. (10-28 ~09:30, page v176, commits 6ab8fded+) DONE:
    - EFFECTIVE WEIGHT + WHERE (Priya "decoder per 1 SD map"; "where carries the offset, ITI + cued, Allen / LocaNMF /
      SVD"): scripts/lick_offset_where.py -> NEW 11.13 (lick_decoder_weight_1sd.png) + 11.14 (lick_{iti,cued}_offset_where_
      maps.png, lick_offset_where_bars.png); section 11 renumbered (angle 11.15 ... behavior-only 11.19). FINDING: the 11.x
      weight / contribution maps were per feature-SD (features z-scored per session) -> exact for Allen, MIXED for SVD /
      LocaNMF (11.11 caption now says so; DECISIONS 10-28 late morning). On pixel dF/F (unit check 0.999-1.002) the bases
      AGREE on weights: ipsi-minus-contra SSp-n / tjM1 / ALM, SSp-m opposite; ALM/tjM1 ~1/3-1/2 of SSp-n per SD. WHERE:
      ITI offset spread, no consistent locus; cued acute in ipsi SSp-n (+0.11-0.17) + ipsi tjM1 (+0.06-0.13).
      Needs terms_<tag>.csv (re-ran lick_offset_contrib_map for areas_split + locanmf mask; maps/summary unchanged).
    - bout_prev_pos CACHE: <OUT_W>/bout_prev_pos.csv (lick_cued_ramp.py --prev-cache, 8 workers, 18 min; read with
      float_precision="round_trip" -- the default parser missed 15 %). Section-14 jobs now ~2.5 min instead of ~2 h.
    - Section 14: 14.2 NEW animal-averaged ITI PETH (replaces old 14.7 iti_ramp_peth); 14.7 per-spout + ITI by previous
      spout image + per-spout cued table in caption (far contra: cued lick response NOT up acutely, up chronically; ITI
      licks after far-contra trials: largest acute rise); 14.8 = cued area maps, now KEY (was 14.15); 14.11 window figure +
      baseline-sensitivity image (conclusions baseline-independent; iti_iso 0-1.5 s under -1..-0.5 unstable).
    STILL TODO: 14.8 pre-cue row (needs restw confirmation from Priya); item 7 synthesis -> Where things stand / FINDINGS
    (per-spout table now supports (b): far-contra target the exception acutely); item 9 orofacial AUC vs recovery + LMM;
    Musall split-row caption numbers (item 14).
8. Open questions for Priya: restw baseline for the 14.8 pre-cue row (yes / no).

### 00-COMPACT-11 (10-28 ~03:25) -- READ FIRST: docs/status/OVERNIGHT_20261027.md is the running log (queue, results,
decisions, questions for Priya). Page v145 live (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d). Done overnight:
masked LocaNMF (v141), ALM / tjM1 rows in encoder / FR figures (cheap ones; expensive refits = question), first-lick aim
behavior axis + FWER max-stat map stats (4b), angle / reach offset-contribution maps (5c-bis), z-scored map check
(5c-ter: ITI cortex-wide components = variance effect), covariate refs / same-target / per-animal (5a/5b/5d: CUED offset
= target composition, ITI offset survives), pixel mixed models (5e). RUNNING: 5f helper (non-linear + behavior-only
decoders; worktree; writes run_5f.sh) -> merge, run, page. Helper rules: worktree + write guard + no M: writes in
dev; NEVER kill processes not started by them (memory helpers-never-kill-by-parent). Open questions: see the log.

### 00-COMPACT-10 (10-27 ~22:40) -- READ FIRST
- **Goal page v140 PUBLISHED** (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d): ROI v8 + split basis everywhere,
  90 PNGs refreshed, sections 10-16 captions re-derived from v8 CSVs (page helper + 8 section helpers; pre-edit copy
  scratchpad pg8/index_before_v8.html; changed-PNG list pg8/changed_pngs.txt); 12.1/12.2/13.6 = Gaussian like-for-like
  (place + local-linear in collapsible supplements); NEW 11.10 lick_matched_angle_reach; NEW 10.2b lick_refit_decoders;
  section 15 re-tier (15.1 / 15.2 = per-session LEVEL maps Key; change maps 15.8-15.10 Supporting; epoch-decoder +
  held-out maps Supplementary; renumbered); 16.12 recomputed; "What changed 10-27" note at the top of section 10.
  `_goal_page/` is git-EXCLUDED (.git/info/exclude) -- not committed, by design.
- **Results that CHANGED direction / significance vs v139 (tell Priya; update FINDINGS):** ITI sagittal r now lost
  subacute / chronic (-13 / -11 %, 4/4; n.s. in plain); cued lateral r chronic n.s. (-8 %, 3/4); "read further OUT
  acutely" GONE (cued down +0.20 -> +0.03 n.s.; cued read CLOSER subacute -0.25 / chronic -0.13, 4/4); cued reach
  like-for-like acute n.s., -0.27 subacute / -0.14 chronic (4/4); cued LATER licks now contralesional acutely (+0.14,
  4/4); 16.12 response at pre aim n.s. everywhere; 11.5 lag bars hold only for split. Full list in the page helper report
  (this conversation) -- not yet in FINDINGS.md.
- **Open to Priya:** 11.5 LocaNMF column is the UNMASKED joint LocaNMF (DECISIONS names masked as robustness) -- add masked?
  SVD-basis like-for-like is weaker (ITI only acute significant). FINDINGS 4d0 / 4d0b ("read ahead of the failing
  movement") need the like-for-like numbers (cued acute +0.16 4/4; far contra +0.22 4/4; pull-based 60 % / ratio
  details obsolete).
- **Next:** FINDINGS.md update; tell e3 / 5b (v140); ALM / tjM1 rows in encoder / FR area figures; delete v1 sessions/ +
  v2 backup + superseded_split (ask Priya).

### 00-COMPACT-9 (10-27 ~21:30) -- READ FIRST, then 8
- **Merged** the split-basis + like-for-like branch (147575c6; helper commits 4785cff3, 0918aab1). Switches:
  WFIELD_AREA_BASIS = split (default) | plain (iti_licks.AREA_BASIS; helpers il.area_features / area_sd / split_table /
  pred_file / split_only; loader iti_where_decoder.area_basis()); WFIELD_LFL_REF = gauss (default, Priya "ok gaussian")
  | loclin (likeforlike.offset). iti_lick_order default SOURCE = areas_split. CANONICAL UNTAGGED outputs = split basis
  (page stems unchanged); plain = "_plain"; per-lick caches carry the basis (*_splitr8; lick_order_preds_areas_splitr8,
  plain lick_order_preds_r8). Scripts without basis tags refuse a plain run (il.split_only). 'beyond' columns -> 'lfl'
  (lick_order, per_spout, first_lick, overcorrection, recovery_link, 3axis/check_figures, where_beyond_pull ->
  where_lfl_*, mixed-model off_*); iti_where_angle_reach lost its bey_* panels (-> lick_matched_angle_reach, now Gaussian
  primary + local-linear edge check beside it, 4cc5eb04).
- **v1 folder fix done (3ab92858):** `iti_licks.labels()` (current ROI folder; raises if empty or if any frozen-run
  session is missing) replaces 16 globs of the v1 `sessions/`; iti_nmatch_check retired to scripts/archive. Remaining
  SESS_NAME globs take an explicit sess_dir (June). v1 `sessions/` may now go on the delete list after the split run.
- **INCIDENT:** the helper's PS93-only tests overwrote 31 `iti_licks/figdata/*.pkl` on M: (pathlib writes slipped its
  redirect; list scratchpad split_test_figdata_overwritten.txt). Derived; all regenerated by the split run. Until
  V8SPLIT_DONE do not --plot-only those stems.
- **SPLIT RUN DONE (67 steps, 0 errors, 21:29-21:55); all 31 overwritten figdata regenerated.** v2 vs v8:
  docs/status/V2_VS_V8_SPLIT_20261027.md (tjM1 chronic build-up: nothing lost v1 -> v2 -> v8).
- **REFIT DECODERS (Priya "re-do the per session / per-epoch decoders ... to show they can be fit"):** new
  `scripts/lick_refit_decoders.py` (50c9500d; split, z-scored; frozen vs epoch refit LOSO-within-epoch vs session refit
  5 contiguous blocks; epoch refit tested against pre trained on the SAME number of licks) -> refit_decoders_{sessions,
  tests}.csv, figures/lick_refit_decoders.png. Lateral ITI frozen -0.16 / -0.14 / -0.15 (4/4) vs epoch refit -0.07
  [-0.10,-0.04] 3/3 / -0.04 n.s. / -0.02 n.s.; cued acute frozen -0.16 vs refit -0.08 3/3; angle same pattern (frozen ITI
  chronic -0.13, refit -0.00). Reading: post-acute the code is still decodable, the frozen read-out moved; acute = a
  smaller real loss plus the move. PS95 has 1 acute session (no epoch refit). NOT yet on the page (add next to the
  where-decoder figure after the page helper finishes).
- (was) **RUNNING:** scratchpad run_v8split.sh (started 21:29; logs rr8/ws_*.log; moves superseded files -> <E>/iti_licks/
  superseded_split_20261027/ (list rr8/ws_moved.txt); ends V8SPLIT_DONE + "ws tracebacks in:"). Est. 25-35 min.
- **NEXT:** check tracebacks; v2 vs v8 comparison (roi_v2_backup) incl. ramp_session_tests tjM1 build-up for Priya;
  page update (re-copy PNGs, captions 10-16, swaps 13.1/13.2 lfl, 11.9 -> lick_matched_angle_reach, 3.14 basis 4 cols,
  re-tier level maps Key, 15.2/15.4 captions, Fig 16.12 stale caption), publish, tell e3 / 5b. Then ALM/tjM1 rows in
  encoder / FR area figures (0.3, 0.4, 0.9-0.11, 0.13, 2.2, 2.6, 18.x, 21.4, 21.11, 22.3, 22.4).

### 00-COMPACT-8 (10-27 ~22:30) -- READ FIRST, then 7b / 7 / 6
- **Priya answered:** primary basis = Allen + ALM/tjM1 split ("I agree"; DECISIONS 10-27 late; robustness = plain Allen,
  masked LocaNMF, SVD). The "lost tjM1 chronic build-up significance" figure = `lick_iti_ramp_summary`
  (ramp_session_tests.csv). v1 vs v2 there: NOTHING lost -- raw build_auc chronic CI crosses 0 in both (right v1 -1.03
  [-2.27, 0.23], v2 -1.05 [-2.22, 0.08]); normalized right significant in both (v1 -0.19 [-0.36, -0.011], v2 -0.17 [-0.32,
  -0.019]), normalized left n.s. in both (upper CI +0.010 / +0.003). Check v8 and answer Priya with all three.
- **Code (0f56a173):** iti_licks default ROI_VERSION = "8"; plain-Allen per-lick preds CACHE tagged `_r8`
  (lick_order_preds_r8.csv.gz; outputs stay untagged = canonical); offset map TAGS + filter-compare BASIS -> areas_splitr8.
- **v2 tables backed up** (295 csv / csv.gz, 270 MB) -> <E>/iti_licks/roi_v2_backup_20261027/ (same relative paths) for
  the v2 vs v8 comparison (v1 = roi_v1_backup_20261027/). Add to SUPERSEDED list once v8 is on the page.
- **RUNNING:** scratchpad run_v8redo.sh (frozen env; logs + runtimes in scratchpad rr8/; ends with V8REDO_DONE and a
  "tracebacks in:" line): track A = iti_licks main/key/controls/binned, check_figures, area_maps, hemo (+summarize), f0;
  track B = split decoder (all + gap2 / far1 / gap2far1, gauss read-out), plain Allen (lick_order, gauss, decomposition,
  where_decoder); then offset maps (areas, split), mixed models, filter compare, lmm figures.
- **10-27 late decisions (DECISIONS.md):** split basis EVERYWHERE ("use split bases, for consistency"); 'beyond the
  pull' -> like-for-like offset everywhere ("yes replace beyond pull"); like-for-like reference = LOCAL-LINEAR ("ok local
  linear it is"; lateral + angle tie with Gaussian, Gaussian = supplement). A helper is coding split + beyond-pull in an
  isolated WORKTREE (no M: writes) and writes scratchpad run_v8split.sh -> merge after waves 2/3, then run it.
- **TODO after the split merge (Priya: "fix the code base so we don't need the v1 folder"):** ~18 scripts list labels
  from `il.OUT / "sessions"` (v1): add `iti_licks.labels()` (stems of OUT/SESS_NAME, fail loudly if empty / < 107) and
  replace every call site (iti_area_maps x2, iti_area_split_sd, iti_contact_check, iti_decoder_contrib, iti_hemo_check,
  iti_lick_order, iti_lick_rate, iti_lick_timing_audit, figdata.imaging_grid, iti_where_decoder.load, iti_zscore_check.sds,
  iti_basis_decoder, lick_session_level_maps, iti_pattern_area_stats, lick_roi_*, lick_offset_contrib_map -- re-grep);
  retire / convert iti_nmatch_check (hard-codes sessions_v3). Then v1 `sessions/` can join the delete list.
- **Duplicate run:** the stopped run_v8wave2.sh waiter survived TaskStop (orphan PID 7444) and started a DUPLICATE wave 2
  at 20:52 when V8REDO_DONE appeared; Priya asked to `taskkill /PID 7444 /T /F`. Harmless for results (split pass redoes
  all) but costs CPU. Pitfall: TaskStop does not kill a Git-Bash script's subshell tree -- stop waiters by PID tree.
- **PAGE-FIGURE AUDIT (helper, all 235 page images; map in scratchpad audit_map.txt):** pattern maps 15.x unaffected
  (SVD basis; lick times identical in v8 files); sections 0-9 / 17-22 unaffected (encoder stat mask). STALE + not in
  waves 1/2 -> WAVE 3 (scratchpad run_v8wave3.sh, logs rr8/w3_*, ends V8WAVE3_DONE): plain --ref place / spout / pooled
  (12.1, 12.2, 13.6, matched_sessions.csv) + gain_prediction; split decomposition + split --ref place + split loclin on
  gap2 / far1 / gap2far1; after V8WAVE2_DONE: filter_compare (now reads matched_loclin_*), mixed models, lmm ALL groups
  (iti_mixed_models.matched default ref place -> loclin). Wave 1's check_figures output is stale (use wave 2's rerun).
  Lateral loclin = gauss to 2 decimals on v8 split (share beyond pre range <= 8 %; ref-refit spread ~ +/-0.03 mm).
  OPEN for Priya (code needed): (i) split basis for the iti_licks frozen/key/binned + the z-scored where-decoder family
  (10.x, 11.x, 12.3-12.13, 13.3-13.9, 16.3-16.5) -- no split option there yet; (ii) replace 'beyond the pull' columns by
  loclin in lick_order / per_spout / first_lick / overcorrection / recovery_link / 3axis. New-rows-requested figures
  (ALM / tjM1 / rest rows): 0.3, 0.4, 0.9-0.11, 0.13, 2.2, 2.6, 18.5/6/9, 21.4, 21.11, 22.3, 22.4.
  Page swaps planned: 13.1 / 13.2 -> _areas_splitr8; 12.1 / 12.2 / 13.6 -> loclin; 11.9 -> lick_matched_angle_reach.
  Do NOT delete v1 `sessions/`: ~20 scripts list labels from it.
- **WAVE 2 STARTED EARLY (~23:15)** as scratchpad run_v8wave2b.sh (run_v8wave2.sh's waiter stopped before it ran anything):
  wave 1 was down to the hemo check (3 workers, CPU 20 %), and wave 2 needs only the wave-1 decoders. More workers
  (june_ref 6, zscore 8, rim_overlay 8) + added `lick_matched_angle_reach.py` (after angle_reach) and plain-Allen
  `lick_matched_readout --ref loclin`; its final check_figures / mixed models / lmm wait for the exact line V8REDO_DONE.
  Local-linear test on the split basis ran separately (rr8/loclin_split_all.log). CPU ~94 %, 36 GB free after start.
- **WAVE 2 queued (superseded by the line above):** scratchpad run_v8wave2.sh (waits for V8REDO_DONE; logs rr8/w2_*.log; ends V8WAVE2_DONE). Helper
  audit (10-27): iti_zscore_check was NOT in wave 1 -> every `_z` page figure was still on v1 numbers (where_decoder_z /
  where_preds_z dated 10-26); wave 2 runs zscore_check -> where_decoder --z -> scatter / per_spout / rmse / spread /
  overcorrection / recovery_measure / recovery_link (angle_z, lateral_z, hit), june_ref (june_ref/ held v1 only) -> drift
  --june, basis --summarise, decoder_contrib, farcontra, 3axis, drift, angle_reach (+extension_dist), first_lick (+misdirected,
  aim_deficit), contact_check (+recovery_marker), gain_prediction, ROI-mask figures; then check_figures / mixed models / lmm
  AGAIN (wave 1 ran them on stale inputs). Deleted 2 stale caches first (first_lick_preds, where_angle_reach_preds; copies in
  roi_v2_backup). NOT affected (helper): decoder / spout / session pattern maps + level maps (lick times / SVD / stat-mask
  only). Known: iti_nmatch_check hard-codes sessions_v3 (not on page; left); ~20 scripts list labels from v1 `sessions/`
  (same 107 files; harmless today).
- **Second wave (audit by a helper):** every other script whose outputs depend on the v8 session files / SDs / area
  features (Haufe pattern maps, spout patterns, session level maps, ...) -- run after the first wave, then compare v2 vs v8,
  re-tier + publish page, tell e3 / 5b.

### 00-COMPACT-7b (10-27 ~22:00): the env-correct v8 extraction FINISHED and PASSED -- 107 files in sessions_roi8fr/, FR env
applied (master_pre_v1 / master_v1:A / attempted), 0 sessions with all-NaN bout_fr, 0 failures. NEXT = 00-COMPACT-6 step 2
(default ROI_VERSION -> "8", full redo under the frozen env).

### 00-COMPACT-7 (10-27 ~21:15) -- READ FIRST, then 00-COMPACT-6 (still valid except as noted here)
- **v8 extraction RUNNING (env-correct):** S2/run_roi8fr.sh (sources S2/rr2/frozenenv.sh, WFIELD_ROI_VERSION=8) ->
  <E>/iti_licks/sessions_roi8fr/ (iti_licks._ROI["8"] now points there). Log S2/roi8fr.log ends with ROI8FR_DONE + a check
  line "107 files; sessions with all-NaN bout_fr: 0 []" -- if that count is not 0, the env did not apply: stop and fix.
  SD tables for v8 already exist (zscore_check/session_sd_roi8_rimlat_almline.csv, session_sd_split_roi8_rimlat_almline.csv;
  imaging-only, env-independent). The env-less first v8 run (sessions_roi8/) was deleted.
- **Then (00-COMPACT-6 step 2 onward):** default ROI_VERSION -> "8"; full redo UNDER THE FROZEN ENV (source rr2/frozenenv.sh
  before every iti_licks-based script); split decoder tag becomes areas_splitr8 (check lick_iti_filter_compare BASIS and
  lick_offset_contrib_map TAGS, currently areas_splitr2); plain Allen decoder chain too; compare v2 (sessions_roi2 +
  roi_v1_backup for v1) vs v8; then page re-tier + publish + tell e3 / 5b.
- **Split decoder on v2 ROIs finished** (areas_splitr2: all ITI + gap2 / far1 / gap2far1 read-outs, offset map, filter
  compare figure redrawn with BASIS areas_splitr2) -- numbers not yet reviewed; superseded by v8 anyway.
- **Deleted on M: with Priya's permission (10-27 ~21:00):** 5.7 GB + small files, listed (rows marked DELETED) in
  docs/status/SUPERSEDED_OUTPUTS_20261027.md. Remaining rows wait for their stated condition; Priya: "ask me if you are
  unsure" -- standing permission to delete superseded / incorrect derived outputs on M: (still never original data).
- **Open questions to Priya:** primary decoder basis (recommendation: Allen + ALM/tjM1 split; masked LocaNMF + SVD
  robustness); which figure showed "lost tjM1 chronic build-up significance" (v2 numbers kept it: -0.37 [-0.63, -0.09],
  17/20 bins); within-spout change test option (relax done; first-lick rows hidden).

### 00-COMPACT-6 (10-27 ~20:30) -- READ FIRST (supersedes 00-COMPACT-3..5 for state; they stay as history)
**Pages:** goal page LIVE v139 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d; v136 = 23 named sections, v137 = Haufe pre
map PuOr_r vs change RdBu_r + collapsible *_levels supplements, v138 = sign key in 11 Haufe captions, v139 = Fig 12.14 ITI
lick-set comparison). `_goal_page/index.html` ALSO holds UNPUBLISHED v2-ROI caption edits (20, by the ROI-redo helper; list in
its report = scratchpad rr2/, pre-edit copy rr2/index_before_roi2.html) + 16 refreshed PNGs (lick_iti_ramp_*, area_compare /
groups, hemo variants / state_gap, check_f0*, kinematics / face control, binned_*, lmm_*_response) -- they will be redone
for v8, so do not publish them as-is unless v8 is abandoned. Review page (loose ends + ROI decisions):
https://claude.ai/artifact/P4Md3KfTSt3Nqq9hxU2knq (v11; source `_review_1027/`, sections A-G).
**Decisions today (DECISIONS.md):**
- ITI licks: ALL ITI licks primary; 2 s quiet / >= 1 s from window end and next spout arrival / both = SECONDARY
  (WFIELD_ITI_FILTER gap2 / far1 / gap2far1; offset holds in all; page 12.14). The ITI is only ~3.1 s; ~70 % of ITI licks
  sit within 1 s of a trial edge (iti_lick_timing_audit.py).
- ALM / tjM1 / mask FINAL = ROI v8 (`scripts/iti_licks.py` `_ROI`, WFIELD_ROI_VERSION): ALM 40 % of MOs, tjM1 50 % of MOp,
  16-px rim thinned to 4 px along the LATERAL edge (anterior of AP -0.5), ALM continued to one horizontal front line
  (8-px front rim), >= 0.19 mm from FRP, one connected region each. Literature: ALM Guo 2014 / Esmaeili 2021 (AP 2.5 ML 1.5),
  tjM1 Mayrhofer 2019 (AP 1.8 ML 2.6) inside; CFA / wM2 outside; Esmaeili's tjM1 targeting point (2.0, 2.0) falls in Allen MOs
  here. Per-pixel map statistics keep the 16-px stat mask. v3-v7 rejected (DECISIONS).
- Per-session Haufe LEVEL maps (amplitude kept; `scripts/lick_session_level_maps.py`) -> Key map figure; per-session
  amplitude-kept change maps Supporting; held-out-epoch figures (15.12-15.15) + epoch-decoder maps (15.6-15.8) ->
  Supplementary. Within-spout first-lick rows hidden (decoders fit ~nothing). APPROVED; re-tier together with the v8 redo.
- Map tests: NO session / animal gate (Priya never asked for >= 3 / >= 3): every animal with >= 1 session per group;
  per-animal effect size with a pooled within-animal session SD, summed / sqrt k (shared `iti_session_cluster_tests`
  change_z / level_z / test_one / test_level); ">= 3 animals agree" only as the dashed outline. Big moves: shape-only ITI
  lateral subacute change 83 -> 7.5 %; cued lateral level acute / subacute 1 / 7 -> 60 / 46 %; ITI far-contra acute level 91 %.
**Basis question (OPEN, Priya leaning "if LocaNMF is better use it"):** held-out pre lateral r ITI / cued: Allen split
0.625 / 0.768, Allen 0.622 / 0.767, LocaNMF masked (glue / MOB comps dropped) 0.622 / 0.761, LocaNMF 0.608 / 0.754, SVD 0.597 /
0.736. Recommendation given: Allen + ALM/tjM1 split primary, masked LocaNMF + SVD robustness. Offset map
(`lick_offset_contrib_map.py`): uniform cortex-wide change contributes ~0; SVD's smaller read-out = held-out pre-reference
term. The joint LocaNMF basis is PRE-STROKE only (corrected earlier claim) and ~1/3 of its comps sit on glue / MOB -> spout
classifiers on it need the same mask check.
**Findings today (PROVISIONAL):** ITI offset robust to stricter ITI sets; cued "layout drops then returns" holds (per-session
maps); ITI "coding increases and spreads" NOT supported as a sustained effect (subacute, PS94-driven, PS95 opposite; the
cortex-wide component is clearest in ITI first licks chronic); the strongest pre lick response sits at AP -0.1 ML 2.4-2.6 =
Allen SSp-ul (limb S1 or a registration shift of orofacial S1 -- unresolved; describe by position); ALM is not noisier than
MOs (SNR 1.35 vs 1.24); tjM1 40 -> 50 % changed tjM1 numbers by ~5-10 %, all directions unchanged (helper report).
**RUNNING / NEXT (in order):**
1. Priya asked to kill the v8 extraction (PID 18800; started WITHOUT the frozen env -> empty FaceRhythm fields). Relaunch v8
   under the frozen env: source scratchpad rr2/frozenenv.sh (COMBINED_ENCODING_FR=master_pre_v1, ..._FR_GROUPS=master_v1:A,
   OROFACIAL_GATE=attempted, COMBINED_ENCODING_LICKS=combined, ... see the file) then S2/roi8_extract.py with
   WFIELD_ROI_VERSION=8; writes sessions_roi8fr/ (sessions_roi8/ = complete env-less run, superseded) -- check 107 files and bout_fr not NaN (the launcher S2/run_roi8fr.sh does).
   The split-SD pass (session_sd_split_roi8_...) does not depend on that env. ALSO compute session_sd_roi8 (iti_zscore_check
   sds inside roi8_extract.py). EVERY extraction must use the frozen env (pitfall).
2. Switch iti_licks ROI_VERSION default to "8"; rerun under the frozen env: the ROI-redo set (helper's list: iti_licks
   --summarise / --binned / --key / --controls, iti_area_maps, iti_hemo_check (cache per ROI version: extend to v8),
   iti_f0_check, iti_check_figures, iti_mixed_models, iti_lmm_figures --groups response) + split decoder
   (WFIELD_LICK_DECODER=areas_split; tag areas_splitr8 -- check SRC_TAG rule 'r<ROI_VERSION>') + the 3 ITI filters +
   lick_offset_contrib_map + lick_iti_filter_compare (BASIS -> areas_splitr8) + the PLAIN Allen decoder chain (where_decoder,
   lick_order, lick_matched_readout gauss, decomposition, lmm figures): the plain areas features change with the lateral rim.
   Compare v2 vs v8.
3. Then: re-tier the page (level maps Key, etc.), fix Fig 16.12 caption (stale deficit numbers -- helper found it; new
   values in its report), update 15.2 / 15.4 captions for the new test, publish, tell e3 / 5b.
4. Encoder / FR area figures: per-pixel SSE / SST are saved as 2x2-block sums -> ALM / tjM1 / MOs-rest / MOp-rest R2 rows can
   be re-aggregated cheaply; trial-PETH-by-area figures may need re-extraction (cost it first). Priya: yes, after the ITI redo.
5. Basis decision with Priya; then the SVD-GAUSS queue items (SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md).
**Superseded outputs to delete later (Priya decides):** docs/status/SUPERSEDED_OUTPUTS_20261027.md.
**Pitfalls (today):** (a) any iti_licks extraction MUST run under the frozen env (rr2/frozenenv.sh) -- without it bout_fr /
first_fr are NaN and the first-lick set shrinks, silently (log=None hides it); (b) Windows spawn: driver scripts need
`if __name__ == "__main__":`; (c) never delete on M: -- use new file names (an empty session_sd_roi2.csv blocked a cache);
(d) `R.map` is the DataFrame method -- use R['map']; (e) lick_order_preds are cached by SRC_TAG -- new inputs need a new tag.

### 00-COMPACT-5 (10-27 ~19:00)
Page v138: Haufe figures pre map PuOr_r vs change RdBu_r, collapsible *_levels.png supplements (15.12-15.15), sign key in
11 Haufe captions. Priya: lick_decoder_pattern_stats_lat_levels "may be a key figure"; also candidates for Key: 15.5 within-
spout, 15.4 position, 15.6 decoder patterns, 15.8 cued spout patterns -> re-tier ONCE the per-session level maps land.
RUNNING: helper building scripts/lick_session_level_maps.py = per-session self-fit Haufe maps WITH amplitude (/ animal's pre
RMS), level TFCE (sign-flip) + change TFCE, all licks AND first licks, global-share per animal (tests Priya's reading
"ITI lateral coding increases and spreads; cued drops then returns" against a cortex-wide component).
ALM / tjM1 check (scripts/lick_roi_check.py, figures/lick_roi_check.png, roi_check.csv): within MOs the anterolateral 40 %
carries 1.4-2.5x the |pre map| of the rest (ALM placement supported); tjM1 mixed (0.7-2.7x; the cued SAGITTAL hot spot sits
posterior to tjM1, ratio < 1); the strongest lateral coding is LATERAL to both ROIs (lateral orofacial S1 edge).

### 00-COMPACT-4 (10-27 ~18:00) -- BASIS COMPARISON (primary basis = Priya's call, pending)
Held-out pre lateral r (ITI / cued): Allen split (ALM + MOs_rest, tjM1 + MOp_rest; source areas_split) 0.625 / 0.768;
Allen 0.622 / 0.767; LocaNMF masked (comps > 50 % outside glue / MOB mask dropped; tag _locanmfFM) 0.622 / 0.761; LocaNMF
0.608 / 0.754; SVD frame-scaled 0.597 / 0.736.
Same-place gauss, change vs pre (acute / subacute / chronic): ITI all licks -- split Allen +0.37 / +0.22 / +0.26 (all 4/4,
chronic CI [0.09, 0.46]); masked LocaNMF +0.36 / +0.25 / +0.23 (chronic CI incl. 0, 3/4); SVD +0.19 / +0.07 / +0.14.
Cued first lick -- split Allen +0.32 / +0.22 / +0.15 (4/4 each).
Offset map (scripts/lick_offset_contrib_map.py; offset_contrib/): uniform cortex-wide change contributes ~0 in Allen and
SVD (global part <= 0.02). With all-pre scoring the chronic ITI activity-change offset is similar in every basis (Allen
0.25, split 0.27, LocaNMF 0.29, masked 0.25, SVD 0.22); SVD's smaller READ-OUT (0.14) is the held-out pre-reference term
(-0.08 chronic, -0.15 acute; PS95 -0.18, PS94 -0.09): with 200 features, held-out pre sessions are read differently from
in-sample ones (least stable decoder; matches its lowest held-out accuracy). Unmasked LocaNMF decoder leans on one
posterior-right component + edge pixels (see maps) -> do not use unmasked LocaNMF; the joint-LocaNMF SPOUT classifiers
(sections with trial spout decoding) need the same mask check.

### 00-COMPACT-3 (10-27 ~17:00)
**CORRECTION:** the 'joint' LocaNMF basis is joint ACROSS PRE-STROKE SESSIONS ONLY (manifest labels: 11 sessions per animal,
June + August, last day -2/-3); NOT fit on post-stroke data. So all three bases are fixed before the stroke and the decoder
is frozen pre-stroke in all three. The SVD-vs-Allen/LocaNMF persistence gap is NOT 'pre-only vs joint basis'. Leading
difference: LocaNMF components are LOCALIZED (area-constrained) like Allen areas; SVD components are global modes.
Proposed diagnostic: per-pixel offset-contribution map per basis (decoder pixel weights x post - pre activity at matched
landing) to show where in cortex each basis's chronic offset comes from. LocaNMF frame-scaled run (tag _locanmfF)
rerunning after a NaN-padding fix in iti_lick_order.py (component counts differ by animal).
**Haufe pre-map colormap + collapsible absolute maps:** queued (queue doc), after the figdata / restructure helpers.

### 00-COMPACT-2 (10-27 ~16:00) -- READ FIRST (adds to 00-COMPACT below)
**Priya's answers on the review page (P4Md3KfTSt3Nqq9hxU2knq):** (1) aim-deficit comparison figure -> SUPPLEMENTARY next to
the recovery measure; (2) TFCE contours replace the per-pixel contours on the per-session change-map figures (per-pixel kept
as supplementary; caption: TFCE = family-wise, session labels permuted within animal; a large significant area = broad
layout change of a shape-only map); (5) shape-distance dots figure (lick_session_shape_d.png) REPLACES the whole-map
heatmap (heatmap linked as table view); (6, 7) map similarity over days (cued, ITI) -> SUPPLEMENTARY; (8) drift sensitivity
-> repo only (no page).
**Page restructure (Priya):** split ballooning sections into named topical sections of ~6-15 figures, Key / Supporting /
Supplementary within each, every sidebar entry = number + short descriptive name, stable #f-<id> links. A helper agent is
doing the restructure + the figure changes above on `_goal_page/index.html` (NOT published; backup
encoding_goal_page_src/index_pre_sections_20261027.html; map docs/status/GOAL_PAGE_FIG_MAP_2026-10-27_SECTIONS.csv). When
it reports: check, copy the listed new images into the publish `files`, publish as v136, tell e3 + 5b (numbers changed;
stable ids unchanged).
**Open observation (not a result):** PS95 cued FAR-MID per-session map similarity to pre dips to 0.31-0.41 on days 25-39
(pre floor 0.45-0.87, mean ~0.66), recovers by days 46-50; close mid also dips (days 25-36); far contra unaffected (0.79-0.90);
behavior unchanged (far-mid landing -0.20..-0.27 mm, aim improving, lick counts normal). Offered checks (Priya: not yet
answered): session state in 9/10-9/24 (running, engagement, lick-response size); imaging / acquisition notes around 9/10
(docs/EXPERIMENT_ERRORS.md, preprocessing QC).
**Priority notes:** compensation check (first-lick contact rate etc.) LOW, after the non-linear decoder.
**FRAME-SCALED SVD (svdF) RESULT (16:00):** same as lick-scaled and unscaled SVD -> scaling is NOT the reason SVD differs.
Same-place gauss, change vs pre: ITI all licks +0.19 (4/4, CI [+0.01,+0.40]) / +0.07 / +0.14 (n.s.); ITI bout starts +0.27 (4/4) /
+0.12 / +0.17 (4/4, CI excl 0); cued first lick +0.23 (4/4) / +0.10 / +0.06; cued far contra -0.32 chronic (4/4). PS93
negative in SVD in later epochs (ITI -0.14 / -0.09), PS95 ~0; PS92 / PS94 positive. So: the acute ITI and first-lick
contralesional read-out is basis-robust; its PERSISTENCE (subacute / chronic) is Allen / LocaNMF-dependent and in the SVD
basis rests on PS92 + PS94. Decision needed with Priya before switching the page to SVD (which basis is primary for the
persistence claim; report basis-dependence explicitly). Run LocaNMF frame-scaled for completeness.
**Push:** 15 commits since 78e6593d pushing via S2/push51.sh (log S2/push51.log); later commits need a new push script.

### 00-COMPACT (10-27 ~15:40) -- READ FIRST; then the queue doc docs/status/SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md
**Pages:** goal page LIVE v135 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d; source `_goal_page/index.html`, untracked,
backup M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/). Review page for loose-end figures (not yet on the goal page):
https://claude.ai/artifact/P4Md3KfTSt3Nqq9hxU2knq (source `_review_1027/`, untracked) -- Priya has not answered its 5
decisions yet (aim figure on page? TFCE contours replace per-pixel? shape-d replaces heatmap? shape-over-days as
Supporting? drift table in 3.13 caption?). Figure links: stable `#f-<image stem>` (registry `_goal_page/fig_ids.csv`);
renumber only via `scripts/goal_page_reorder.py --version vNNN --date ...` (new figures need a UNIQUE placeholder number
+ data-tier first; see scratchpad page_1027f.py). Other windows: e3 (PI deck) waits for "decoder figures final"; 5b
(imaging page) uses stable ids too.

**Decisions today (DECISIONS.md / queue doc / FINDINGS 4d0-LATE..4d0j):**
- Offsets: like-for-like SAME-PLACE read-out (each lick vs the animal's pre licks that landed at the same lateral place),
  Gaussian kernel (`lick_matched_readout.py --ref gauss`, width by held-out pre-session error: 0.1 mm); 'beyond the pull'
  RETIRED (sign depended on the assumed pull slope). Secondary: per-lick error coding vs pre (same-spout, misdirected).
- PRIMARY decoder = frozen pre-stroke decoder on the animal's pre-stroke SVD basis (`WFIELD_LICK_DECODER=svd`); Allen +
  LocaNMF = basis-robustness checks; decoder r and per-session refit decoders to SVD too; section-2 / 3.19-3.22 trial
  spout classifiers (joint LocaNMF now) -> SVD (needs an SVD frozen spout decoder via frozen_models.load_or_fit).
- Mixed models: TIME model primary (shared wfield_local/mixed_models.py), epoch model as paired supplementary panel,
  behavior-model deficit = FIRST-lick weighted signed aim (`lick_aim_deficit.py`, reliability 0.98); significant terms
  bold; joint 'any change' test + AIC shape comparison approved (not done); 'time_drift' variant = sensitivity only.
- Behavior axis everywhere = first-lick weighted aim; far-contra first-lick aim beside far-contra-specific measures.
- Figure names lick_ / lick_iti_ / lick_cued_; every figure script to save figdata + --plot-only (helper agent).

**Findings (PROVISIONAL):**
- ITI licks read contralesional of where they land: robust ACUTELY in all bases (Allen +0.32, LocaNMF +0.46, SVD +0.20,
  4/4); later epochs basis-dependent (Allen +0.21 / +0.24 4/4; LocaNMF +0.29 / +0.25; SVD lick-scaled +0.07 / +0.14 n.s.;
  SVD UNSCALED +0.06 / +0.14 n.s. -> lick-window scaling is NOT the explanation; PS93 negative in SVD). Frame-scaled SVD run
  in progress. Not explained by the larger lick response (attribution has the opposite sign); not previous-target
  carry-over (strongest after far-IPSI trials).
- Cued: the acute contralesional read-out comes largely from MORE target-carrying misses (misdirected licks read toward
  their target, like the rare pre-stroke misses); it tracks aim (~0 / slightly ipsi at pre aim). Cortex changes
  substantially (lateral r loss, compression, far-contra map replaced acutely, bigger lick response), mostly recovering.
  First cued lick +0.22..+0.35 acute in all bases; later epochs basis-dependent.
- Decomposition: ITI = uniform contralesional shift + compression; cued = compression + small uniform ipsi shift; chronic
  far-contra ipsilesional read-out = compression at the edge.
- First-lick aim shows ~2x the chronic deficit of the all-lick aim (corrective licks hide it).
- Mixed models (first-lick aim deficit): at pre aim ITI read-out +0.15, ITI lateral r -0.21 (unrelated to aim); cued r loss
  fully explained by aim.

**RUNNING at handoff (background bash in session f082ee0c; logs in S2 = scratchpad):**
- frame-scaled SVD read-out: `WFIELD_LICK_DECODER=svd` iti_lick_order + `lick_matched_readout --ref gauss`, outputs tagged
  `_svdF` (S2/svdF.log, ends with DONE). Then run LocaNMF frame-scaled the same way (basis_locanmf/frame_sd.csv exists).
- helper agent: figdata + --plot-only for every figure script -> docs/status/FIGDATA_PLOT_ONLY_2026-10-27.md when done;
  it edits the map / check / scatter scripts -- do not touch those until it reports.

**TO DO (order; figure ids in the queue doc):** (1) read svdF -> decide SVD scaling; switch page figures to SVD + gauss
(supersede tables A / B); (2) local-linear vs Gaussian reference (held-out error); (3) four covariate references (landing /
+kin / +FR / +FR+kin; no June FR); (4) same-target same-place; (5) mixed-model pixel maps + x/y/z / angle / elevation /
reach Haufe maps (after the helper); (6) epoch panels for every LMM figure, joint test, AIC shapes, copy-link icon on the
page; (7) June SVD projection -> June-anchored drift supplement; (8) recovery link on the new measure; per-animal
change-vs-pre scatter + CI markers; first-lick row on 3.29; (9) spout classifiers -> SVD; (10) non-linear decoder;
behavior-only (FR) decoder; then the compensation check (first-lick contact rate etc.); LOW: 2-D match, June whole-map
reference. Then FINDINGS + page synthesis with the revised cued wording; tell e3 when decoder figures are final.
Unpushed local commits since 78e6593d (pushed as 4c2da52f): push with S2/push<NN>.sh.

**PITFALLS (today):**
- Bash heredocs turn "\n" inside Python string literals into real newlines and some multi-heredoc commands fail to parse
  -> write Python with the Write / Edit tools; bit us ~15 times.
- Artifact publish: files must be under the workspace (repo); M: and some scratchpad paths are refused -> copy into
  `_goal_page/` or `_review_1027/` first.
- Push: cherry-pick named commits onto origin/main in a temp worktree; pre-push pytest ~8-11 min; the test
  `test_parallel_results_are_ordered` rejects a fan_out result used without `sorted(` within 14 lines.
- Never kill a running job (Priya runs kills); never rename on M: (write copies).
- `lick_order_preds{SRC_TAG}.csv.gz` is reused if present -> change SRC_TAG (e.g. the F suffix) when inputs change.
- `#figS-N` old-number anchors follow the LAST renumber; quote stable ids.

### 00-LATE (10-27 evening, Priya back) -- read FIRST (supersedes 00-NIGHT for state / to-dos)
**Page:** LIVE v130 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d). v129 = REORDER: every section Key ->
Supporting -> Supplementary, STABLE ids `#f-<image stem>` (registry `_goal_page/fig_ids.csv`), old numbers kept as
`#figS-N` anchors; renumber with `python scripts/goal_page_reorder.py --version vNNN --date YYYY-MM-DD` (it rewrites all
cross-references; insert new figures with a unique placeholder number + data-tier first, see scratchpad page_1027f.py).
Maps: docs/status/GOAL_PAGE_FIG_MAP_2026-10-27_REORDER.csv, _goal_page/fig_map_20261027_reorder.csv. Other windows
(e3 deck, 5b imaging page) told and remapped; e3 waits for "decoder figures final".
**Decisions (DECISIONS 10-27 late):** z-scored where-decoder figures primary (unscaled supplementary); mixed models WITH
TIME primary (shared `wfield_local/mixed_models.py`; `scripts/iti_lmm_figures.py`, figs lmm_time_* / lmm_behaviour_*),
epoch figures supplementary; SAME-PLACE like-for-like read-out = primary offset measure (`scripts/lick_matched_readout.py
--ref place|spout|pooled`), per-lick error coding vs pre = secondary; per-session self-fit maps primary for maps; scatter
ellipses 1 SD + session-level CI markers (TO DO); rename all iti_/cued_ figures to lick_ / lick_iti_ / lick_cued_ (helper
agent running at handoff: copies on M:, scripts' savefig, page img src; fids unchanged; table
docs/status/FIGURE_RENAME_LICK_20261027.csv) + captions rewritten to same-place numbers (log
docs/status/CAPTION_REWRITE_20261027.txt).
**New scripts:** lick_matched_readout.py, lick_gain_prediction.py, iti_decoded_decomposition.py, cued_misdirected_licks.py,
iti_lick_order.py, iti_lmm_figures.py, iti_session_cluster_tests.py (TFCE, session-level), goal_page_reorder.py.
**Results:** FINDINGS 4d0-LATE (synthesis), 4d0e-4d0j; review list REVIEW_FOR_PRIYA_2026-10-27.md.
**TO DO (in order):** (1) after the rename agent: check its report, publish, push; (2) REDO the figures that still PLOT
'beyond the pull' with the same-place measure -- lick order first/later + slope (old 3.54 / 3.71 / 3.72), recovery link
(3.37 / 3.39 / 3.40), per-spout offsets (3.22, overcorrection 3.55), cued per-lick histogram pull triangles (3.43), LMM
read-out / lickclass / perspout panels -- old versions to Supplementary; (3) scatter ellipses: session-level CI markers;
(4) animal shapes (orofacial_cohort.ANIMAL_MARKER) on every pooled session graph (done for lick_matched_*); (5) 2-D
(lateral + extension) matched sensitivity if Priya OKs; (6) 3.70 follow-ups (per-animal self-fit redraw, cross-animal
null, global-activity confound); (7) first-lick map shape (re-extract all first licks); (8) tell e3 when decoder figures
are final. Push: `S2/push<NN>.sh` cherry-picks named commits onto origin/main in a temp worktree; pre-push pytest takes
~8-11 min (`test_parallel_results_are_ordered` guards fan_out ordering -- sort results).
**00-LATE addendum (10-27 ~15:20) -- READ WITH the queue doc docs/status/SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md:**
- Page LIVE v135. Decided: PRIMARY = SVD-basis decoder + Gaussian same-place reference (`--ref gauss`, width by held-out
  pre sessions: 0.1 mm Allen / SVD, 0.15 LocaNMF); Allen + LocaNMF = basis-robustness checks; mixed models (time primary,
  epoch supplementary; behavior deficit = weighted signed aim) on the same measures; significant terms bold.
- BASIS CHECK (same-place gauss, change vs pre): all bases agree ACUTELY (ITI +0.20..+0.46, cued first +0.22..+0.35,
  4/4); later epochs basis-dependent -- ITI chronic Allen +0.24 (4/4), LocaNMF +0.25 (3/4), SVD +0.14 n.s.; cued first
  chronic Allen +0.15, LocaNMF 0.00, SVD +0.06. OPEN: SVD features are z-scored by their SD over the session's LICKS
  (Allen by the SD over all FRAMES) -- the larger post-stroke lick response inflates the lick SD and may divide part of the
  effect out. Running: `WFIELD_LICK_DECODER=svdraw` (unscaled SVD), log S2/svdraw.log. Next: frame-level SVD SD (needs
  the session's SVD time courses), then decide the SVD scaling before switching the page.
- FIRST-LICK weighted aim (recovery_measure_first_vs_all_sessions.csv): split-half reliability 0.98 (all-lick 1.00),
  r = 0.94 with the all-lick measure, chronic deficit about 2x larger (PS94 -12.1 vs -5.8 deg) -- awaiting Priya's OK to
  switch the behavior axis + LMM deficit to it.
- Drift: `mixed_models` model "time_drift" (time model + linear day term): day-1 step z unchanged, recovery terms trade
  off with drift -> sensitivity only. Pre-only drift extrapolation: ITI read-out trend is the opposite sign; r trend same
  sign but wide CI. June-anchored drift supplement queued (June has 3-D landing; NO June FaceRhythm).
- Approved and queued: x/y/z + angle + elevation + reach Haufe maps (pooled + per spout, epoch and epoch - pre), PIXEL
  mixed models with TFCE, four covariate references (landing / +kin / +FR / +FR+kin), local-linear vs Gaussian
  reference, per-animal change-vs-pre scatter, first-lick row on 3.29 (3.60 done in the script).
- Helper agent running: figdata + --plot-only for every figure script (docs/status/FIGDATA_PLOT_ONLY_2026-10-27.md when
  done); do not edit the map / check / scatter scripts until it reports.
- Parallel session loading in `iti_where_decoder.load` (fan_sessions); basis frame ranges cached (basis_frame_range/).

**QUEUE after the SVD switch (Priya 10-27 late, in this order):** (a) SVD-basis decoder = primary (chain S2/run_svd.sh,
env WFIELD_LICK_DECODER=svd; compare with the Allen-area version, then switch page figures); (b) Gaussian-kernel
same-place reference (width chosen by leave-one-pre-session-out prediction error, floor = landing noise; sensitivity
0.1-0.4 mm); (c) landing + FaceRhythm reference (kernel on landing + linear FR terms; also + kinematics); (d) same-target
same-place comparison ('more misses' vs 'each lick read differently'; lower priority but BEFORE the non-linear decoder);
(e) redo recovery-link (3.4 / 3.41 / 3.42) on the new measure (the behavior-model figures already are the read-out vs
aim regression); (f) angle-matched secondary + coordinate family (lateral / sagittal / angle / elevation / reach / xyz);
(g) non-linear decoder (low priority). Mixed models now: like-for-like read-out, behavior deficit = angle aim
(`iti_lmm_figures.py --deficit aim`), epoch model drawn by `mixed_models.plot_epoch`, significant terms bold.
**Pitfall:** python heredocs in bash turn "
" in string literals into real newlines -- use Write/Edit for Python.

### 00-NIGHT (10-27) -- read FIRST
- Page v124 (figs through 3.70; source backed up). Running list of results / decisions / open questions for Priya:
  `docs/status/REVIEW_FOR_PRIYA_2026-10-27.md` section "AFTERNOON / EVENING RUNNING LIST" -- present it all when she is back.
- Decisions today: DECISIONS.md 2026-10-27 (decoder r + offsets LATERAL primary; behavior ANGLE as change vs pre; no
  extension filter; max-stat + FDR contours). FINDINGS 4d0 (synthesis), 4d0b (far-contra read-out vs landing), 4d0c
  (confound ledger, checks done), 4d0d (first lick read toward the target).
- New scripts today: iti_session_patterns / iti_session_pattern_tests (per-session maps, whole-map permutation, drift
  floor), iti_first_lick, iti_overcorrection [--angle], iti_recovery_measure, iti_where_angle_reach (polar), iti_check_figures
  (coord_table: lateral / angle / x-y-z), iti_hemo_check (agent), iti_mixed_models, iti_spout_animal_panels,
  iti_extension_dist, iti_june_ref + iti_where_drift --june, iti_where_decoder / iti_where_scatter --z (agent).
- Pitfall: pandas `T.map` is the DataFrame METHOD -- use T["map"] (silently broke subtitles once).

### 00-AFTERNOON (10-27) -- read FIRST
- Page v122 (figs 3.39-3.51 appended after 3.38; 3.37 now draws pre sessions; "Where things stand" first row =
  Priya's synthesis, FINDINGS 4d0). Source backed up to M:/.../encoding_goal_page_src.
- New scripts: iti_check_figures.py (per-lick error histograms ITI + cued with landing row, pull-only control, F0,
  3-axis, per-session courses; cued figures named cued_check_*), iti_decoder_pattern_stats.py / iti_spout_pattern_stats.py
  [--iti] (held-out per-session Haufe maps, pixel BH, >= 3/4), iti_pattern_area_stats.py (Allen areas), iti_june_ref.py +
  iti_where_drift.py --june, iti_recovery_link.py --plot-only (pooled + session scatter).
- PENDING to page: 3.35 stats (iti_spout_pattern_stats.png) + ITI version of 3.35 (iti_spout_pattern_stats_iti.png) +
  area-stat figures (iti_pattern_area_stats_{position,spout,spout_iti}.png) -- were re-drawing with the 3/4 rule.
- Priya to decide: switch pixel stats to the decoder page's FWER max-stat for consistency? z-scored pattern-map check?
  behavior-only decoder (approved, low priority, not started).
- PITFALL again: Python heredoc replacement strings with 
 become real newlines -> fix with Edit (hit 5x today).

### 00-OVERNIGHT (10-27 ~04:00, autonomous while Priya slept) -- read FIRST
- REVIEW LIST for Priya: `docs/status/REVIEW_FOR_PRIYA_2026-10-27.md` (key results, questions, issues). Page v121.
- DONE overnight (all in FINDINGS 4d, scripts committed, pushed through d6ebdb1 + this batch):
  spread-matched r (`iti_where_spread_matched.py`), recovery link on the z-scored decoder + per-animal figure
  (`iti_recovery_link.py --beh angle --z`, page 3.37), far-contra pattern test (`iti_farcontra_pattern.py`, 3.38),
  drift vs day gap (`iti_where_drift.py`, 3.36), Haufe position maps (`iti_decoder_patterns.py`, 3.33 / 3.34), per-spout
  one-vs-rest maps (`iti_spout_patterns.py`, 3.35), F0 check (`iti_f0_check.py`), three-axis decoder
  (`iti_where_3axis.py`), significance bars on 3.14 / 3.23 / 3.24 / 3.26 / 3.31. New "Where things stand" row on the
  lateral r loss. Page figures 3.33-3.38 APPENDED (no renumbering; fig maps updated).
- NOT DONE / blocked: jaw per far-contra lick (licks.csv.gz has no per-lick jaw; jaw_trials is per trial -> needs the pose
  window); 415 nm lick response (hemodynamic arm closed per CLAUDE.md -> asked Priya); RMSE bias/SD split, ALM/tjM1
  non-overlapping ROIs (optional); older 00f item 7 items untouched.
- iti_where_decoder.coords now returns 5 columns (+ down_dv at index 4); load() also stores `<set>_pos` per lick.
- Pitfall hit 4x tonight: python heredoc replacement strings containing "\n" write a REAL newline into the file ->
  use the Edit tool for any string with 
.

### 00-NOW (10-27 ~00:30) -- read before 00a-00g
- Sagittal chain DONE (22:56); page v119 PUBLISHED with the sagittal figures (3.15-3.22 redrawn + recaptioned; scatter
  y = 0 line removed; decoder suptitle gap fixed) and the NEW KEY row (Priya liked: cued licks read AHEAD of the failing
  movement acutely; "as aim recovers, the ITI contralesional offset fades"). Z-SCORED features = PRIMARY readout in the
  KEY row for BOTH axes (Priya 10-26: OK for sagittal; lateral agrees either way); figures 3.15-3.22 still show
  unscaled (3.22 links the z version). Page src backed up; push of b5de1e7..HEAD queued (push41).
- Recovery link: `scripts/iti_recovery_link.py --beh {hit,angle,lateral}` (outputs recovery_link_*_{tag}.csv); hit rate
  saturates -> angle is the recovery measure (chronic PS92 1.01 / PS93 0.77 / PS94 0.13 / PS95 0.89). Priya says PS95
  recovered best -- ask which measure ranks PS95 first.
- TO-DO (low priority, Priya 10-27): 3-AXIS version (lateral, outward = forward, down) in one run, z-scored; sagittal
  ~ |down| (r 0.96), outward independent (r -0.05). Then: spread-matched r for lateral AND sagittal, projection on the
  pre far-contra pattern axis, jaw per far-contra lick. sabatinilab-e3 not yet told the figures are final.

### 00d. Results this session (all PROVISIONAL; FINDINGS 4d (xiii), ENCODING_FINDINGS 8.17 KEY block)
- **KEY (page "Where things stand"): cortex read with the frozen pre-stroke decoder places self-initiated (ITI) licks
  further CONTRALESIONAL, and cued licks CLOSER, than where they land.** Measure = decoded minus where the animal's
  unchanged pre-stroke pull toward the middle would put the lick (`iti_where_beyond_pull.py`; pull slope lateral
  0.31-0.65, outward 0.13-0.36). ITI lateral +0.34 / +0.20 / +0.23 mm (acute 4/4); cued outward -0.30 every epoch
  (0/4 further). Confirmed on licks at their usual (pre) spot (no pull model) and with within-session z-scored features
  (ITI +0.33 / +0.22 / +0.25, 4/4 every epoch; cued -0.22 / -0.21 / -0.17).
- **Per spout (3.22; ITI by PREVIOUS trial's spout):** ITI contralesional after every spout acutely, chronically after
  far-ipsi / far-mid trials; cued closer at every spout; chronic cued FAR-CONTRA licks read IPSIlesional (-0.24, 0/4
  contra) -- not the pull: usual-spot -0.22 / -0.46 / -0.17 (PS92 / 93 / 95; PS94 none), no mirror shift at far ipsi.
  23:10 far-contra decomposition -- CORRECTED 23:40 (Priya). Kept: ACUTE cued far-contra decoded is contralesional of
  the (ipsi-shifted) landing BEYOND the pull in 3/4 (+0.46 / +0.21 / +0.68, PS95 -0.05); CHRONIC read ipsi of the
  landing (sagittal run: -0.24 [-0.35, -0.12], 0/4 contra); chronic lateral landings near pre (1.26 / 1.18 / 1.27 vs
  1.36 / 1.43 / 1.43; PS94 0.13 vs 1.02); within-spout lateral r ~0. WITHDRAWN: (a) "decoded on the spout's side" --
  the decoder never sees the spout, and pre landings cluster MEDIAL of the spout (side_ml is the tongue tip, not the
  spout); say "contralesional of the landing". (b) the "smaller decoded size" / "acute size jumps" -- size was
  measured from the MOUTH in lateral x forward (out_ap) only; forward is not protrusion (licks run mostly down): on
  SAGITTAL extension acute far-contra protrusion is LOWER (4.69 / 4.23 / 4.57 / 4.03 vs pre 4.86 / 4.32 / 5.50 / 5.01).
  (c) the pull shrinks toward the mean of ALL the animal's pre licks (one decoder, spout unknown; center lateral -0.11
  / +0.38 / -0.30 / +0.07, sagittal 4.15 / 3.97 / 4.56 / 4.54 mm), not toward the mouth or each spout's mean -> only
  beyond-pull per axis is a valid offset. Interpretation open (downstream gain / changed strategy / degraded far-contra
  pattern); tests: projection on the pre far-contra pattern axis, jaw per lick on far-contra.
  RECOVERY: PS95 recovered best, PS94 least (Priya); far-contra hit rate saturates chronically (1.03 / 1.00 / 1.01 /
  1.01) -> use tongue angle / lateral protrusion as the recovery measure (`scripts/iti_recovery_link.py --beh`).
- **r is unaffected by offsets / pull** (correlation ignores shift + scale); the r loss is a separate finding. Part of
  the lateral r loss is a narrower spread of lick positions (fully: ITI PS92 / PS95, in-trial PS93; not ITI PS93 / PS94).
  Lateral falls more than (horizontal) angle chronically (paired, 4/4 areas). In-trial angle r above pre 4/4.
- **Not a scale artifact:** `iti_zscore_check.py` (each area / its frame-wise SD over ALL frames of the session; NOT over
  licks, NOT a pre-lick baseline). r losses unchanged or larger.
- **Per-area split** of the decoded shift (`iti_decoder_contrib.py`): many small mixed-sign contributions (ridge spreads
  credit), no single area; cued lick-window activity UP in PS92 / PS93, DOWN in PS94 / PS95 (so "more tjM1/ALM" was not
  the premise). ALM / tjM1 ROIs are excluded from decoder features because they are subsets of MOs / MOp (MOs / MOp are in).
- **MAE in mm** (3.15 panel 2; includes pull + offsets, no direction): ITI lateral +0.39 acute; outward error stays up
  chronically for both lick types (ITI +0.09, cued +0.18, 4/4).
- **Significance bars** on 3.7 / 3.8 (`iti_licks.sig_bars`; per bin paired nested bootstrap, BH-FDR q 0.05 across bins;
  lw 2.6 alpha 0.5 = the kinematics trajectory style). 3.8: ITI lateral below pre in every bin chronically; cued below
  pre only AFTER the lick chronically. 3.7: per-session ITI relative-to-previous-spout decoding ABOVE pre chronically.
- Partial-refit figures now name their model in the image title (`partial_refit.model_label`); 1.37 draws landing -
  target lick kernels (old to-do 2 done).
- **Retracted tonight:** "all-licks ITI offset = moving licks + pull" (used the overall ipsi shift; nearest-spout grouping
  makes the within-spout shift much smaller).

### 00e. Decisions (Priya, 10-26 evening)
- Unscaled features = PRIMARY; within-session z-scored = SUPPLEMENT (label z-scored figures in title, axes, caption).
- Never subtract a pre-lick baseline from decoder features (the ITI carries position information).
- Decoder axes: lateral + SAGITTAL extension (replaces outward); angle = the encoder's exec_angle definition.
- Per-spout ITI grouping = previous trial's spout (not nearest-to-landing). Session-dot layout (3.9 style) for per-spout.
- "Licks at their usual spot" (not "landing held fixed"); always compare the change vs pre (delta error).
- MAE added as a companion to r (says nothing about direction; RMSE^2 = bias^2 + SD^2 split offered, not done).
- The encoder uses CARTESIAN tongue traces (lateral, forward, 3-D extension) + a per-lick within-spout angle kernel;
  a polar encoder rerun was offered, not requested.

### 00f. TO DO (after 00c)
1. Significance bars on the remaining time courses: 3.14 basis decoder (`iti_basis_decoder.py --summarise`), ITI PETHs
   (`iti_licks.py --ramp` / `--key`), orofacial-vs-control PETHs (`iti_area_maps.py --compare`) -- reuse `sig_bars`.
2. F0 / dF check on the larger post-lick response (raw F0 per session vs days; response in dF without /F0); 415 nm
   lick-aligned response pre vs post.
3. Frozen-decoder drift vs day gap within pre (is the ITI r loss beyond elapsed-time drift?).
4. Spread-matched r for lateral and the encoder-style angle (licks restricted / reweighted to the pre spread).
5. Per-epoch decoder PATTERN maps (Haufe: cov x weights) in each animal's pre SVD basis, back-projected; split-half
   reliability; per-spout one-vs-rest patterns across epochs (Priya's "map the decoder weights per position").
6. Optional: RMSE = bias + SD split panel; non-overlapping ALM / MOs-rest, tjM1 / MOp-rest ROIs.
7. Older: jaw per ITI lick (3.27 row); stray_licks rerun after sabatinilab-5b's kinematics_all swap; PRE-level TE maps;
   epoch-kernel drift-rate test; fold scratchpad figure scripts (fig_goal_summary.py etc. in S) into scripts/.

### 00g. Pitfalls (new tonight)
- Bash heredocs / python -c: a `\n` inside a Python string literal written through the tool layer becomes a real newline
  and breaks the file (hit twice). Use the Write / Edit tools for any code containing `\n`.
- ruff blocks commits (e.g. B007 unused loop variables); fix and re-commit, then regenerate the push script (a push
  script built after a failed commit cherry-picks nothing / the wrong SHA).
- Shared-y axes: raising ylim per axis compounds; raise once (`_headroom([axs[0, 0]])`).
- Nearest-spout grouping changes group membership when licks move: never scale an overall shift into a per-spout
  prediction.
- Per-spout r is low and spread-sensitive (small within-spout range); read offsets first.
- Z-score over all frames, never over licks (that forces each session's mean lick feature to 0 and erases offsets).
- `iti_where_*` scripts now write lateral + SAGITTAL columns (act_sag / dec_sag); `_groups` still uses outward for the
  nearest-spout assignment; `iti_decoder_contrib.AXES = {"lat": 0, "sag": 3}` (coords column order: lat, out, reach, sag).

Supersedes the RUNNING / TO-DO lists of `STATUS_2026-10-22_ENCODING_HANDOFF.md` §000z. Window = session `sabatinilab-d4`
(pose window `sabatinilab-5b`, FaceRhythm `sabatinilab-e3`). Worktree `C:/Users/SabatiniLab/Github/wf_attempt` (branch
analysis/attempt-coding). Old scratchpad (this session only; a NEW session can read it by absolute path but cannot
publish from it): `S = C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`.
Python: `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe`, run from the worktree with
`PYTHONPATH=$(cygpath -w $PWD)` (another `wfield_local` install shadows it). ITI outputs: `<E>/iti_licks/` =
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/` (figures in `figures/`).

## 0. START HERE (19:35)

### 0a. PUBLISH THE PAGE — exact steps for a new session

**PUBLISHED 10-26 evening by a new session: live = v105** (local `_goal_page/index.html` + all 183 referenced PNGs). Two
publishes (v104 = index + 121 PNGs, v105 = the other 62): the referenced PNGs total ~65 MB, over the 64 MB per-publish
cap, so split `files` into batches under ~35 MB. Old images not read in a session block the publish ("published content
is not what you last saw"): `action: "list", scope: "files"` on the URL first counts as seeing them. Spot-checked 1.37,
3.7, 3.15-3.20, 3.25 (published byte sizes = local). The steps below are kept as the procedure for the next republish.

**10-26 late (new session):** partial-refit figures now name their model in the image title (`partial_refit.model_label`;
1.37 also draws landing - target lick kernels = to-do 2 done); 3.19 / 3.20 redrawn (legend clear, symbols spelled out);
NEW 3.21 = change in decoded - actual position vs pre (`iti_where_delta.png`) -> old 3.21-3.30 are now 3.22-3.31
(`_goal_page/fig_map.csv`; kinematics control = 3.26). Signed-error / outward results: ENCODING_FINDINGS 8.17.
Pending Priya: KEY row wording (ITI-only contralesional reading), spread-matched lateral / angle check (angle here =
horizontal atan2(lat, out), not the encoder's exec_angle), relabel ITI / in-trial rows in 3.15-3.18.

**10-26 later (live v112):** KEY row = beyond-pull offsets (ITI read contralesional, cued read closer), robust to
within-session z-scoring (`iti_zscore_check.py`); per-spout version = NEW 3.22 (`iti_where_per_spout.py`) -> old
3.22-3.31 are now 3.23-3.32 (`_goal_page/fig_map.csv`). Per-area split (`iti_decoder_contrib.py`): no single area.
DONE 10-26 late: per-bin significance bars on 3.7 / 3.8 (`iti_licks.sig_bars`, `--plot-direction`,
`--plot-frozen-tc`; tables sig_bins_3_7/3_8.csv); MAE panel in 3.15; z-scored figures labeled. STILL TO GET BARS:
3.14 (basis decoder), the ITI PETH figures, orofacial-vs-control PETHs. NEXT: lateral + SAGITTAL extension
(sqrt(forward^2 + down^2)) replacing outward in 3.15-3.22 (Priya 10-26), z-scored kept supplemental. Then: F0 / dF check on the larger response; frozen-decoder drift vs day gap within pre; spread-matched
r (encoder-style angle); per-epoch decoder PATTERN maps (Haufe) in the SVD basis; 3.7 / 3.8 significance bars.

The Artifact tool was switched off in this session from ~14:15, so everything since v103 was LOCAL ONLY until v105.
- Live: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d = **v103** (12:00). Everything after it exists only locally.
- **Local source (newest, complete): `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html`** (+ its 190 PNGs;
  untracked, listed in `.git/info/exclude`; 63 MB). Backup copy: `M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/`.
  The old `S/goal_page/` was identical at 19:30.
- What the local page adds over v103:
  - a TITLE above every figure (`<div class="ftitle">`, also used by the sidebar; source `_goal_page/fig_titles.csv`);
  - section-3 captions open with "Decoder: frozen / per-session" and "Measured: r vs dF/F";
  - 3.9 caption carries the paired ITI vs in-trial test;
  - new figures: 1.37 TE partial refit, 3.11 lick rate, 3.14 basis decoder, 3.15 where decoder, 3.16 lateral x bases,
    3.28 orofacial vs control PETHs; supplements 3.17 / 3.18 = the where decoder (every target) in the joint LocaNMF
    and pre-stroke SVD bases (`iti_where_decoder_locanmf.png`, `iti_where_decoder_svdbasis.png`);
  - corrected whole-cortex claim; 3.25 kinematics control with a face-adjusted row;
  - all section-3 figures redrawn: frame / measure labels, stars above data, SEM bands on 3.7 / 3.21 / 3.24,
    3.15 split in two.
- **Step 1 — DONE 20:40 (chain finished 19:33, all images copied into `_goal_page` + backup). Only needed again if figures are redrawn later; check `S/relabel.log`
  ends with `relabel3 done`):
  ```bash
  F="M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/figures"
  G=/c/Users/SabatiniLab/Github/wf_attempt/_goal_page
  for n in iti_basis_decoder iti_area_maps iti_area_peth iti_area_groups iti_area_compare iti_where_decoder \
           iti_where_decoder_bases iti_where_decoder_locanmf iti_where_decoder_svdbasis iti_frozen_timecourse iti_frozen_summary iti_ramp_sessions iti_ramp_summary \
           iti_ramp_peth iti_ramp_peth_isolated iti_kinematics_control; do cp "$F/$n.png" $G/; done
  ```
- **Step 2 — read the live page first.** Call Artifact with `action: "read"` and the URL. A publish to an artifact the
  session has not read is refused. Confirm the live page is our v103: its 3.x sections end at the whole-cortex
  supplement, and it has no `ftitle` divs. The local file is a strict superset of v103: no one edited the live page
  after 12:00.
- **Step 3 — publish** with ALL changed images in `files` (publishing index.html alone does NOT upload PNGs; v98-v102
  showed 9 broken figures that way):
  - `action: "publish"`, `url`: the page URL above;
  - `file_path`: `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html`;
  - `root`: `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page`;
  - `files`: every PNG in `_goal_page` modified after 2026-10-26 12:00, each mapped to itself. Generate the map with:
    ```bash
    cd /c/Users/SabatiniLab/Github/wf_attempt/_goal_page && find . -maxdepth 1 -name "*.png" -newermt "2026-10-26 12:00" \
      | sed 's|^\./||' | awk 'BEGIN{printf "{"} {printf "%s\"%s\": \"%s\"", (NR>1?", ":""), $0, $0} END{print "}"}'
    ```
    About 25 files. Expect at least: te_partial_refit_lines, iti_area_compare, iti_basis_decoder, iti_where_decoder,
    iti_where_decoder_bases, iti_lick_rate, iti_direction_decoding, iti_frozen_timecourse, iti_frozen_summary,
    iti_ramp_sessions, iti_ramp_summary, iti_binned_sweep, iti_binned_summary, iti_recovery_marker,
    iti_kinematics_control (+ the step-1 files).
  - `label`: e.g. "Titles, ITI labels, where + refit".
  - If the refusal hands back the live version, check it is v103 and publish the local file again. It is the merge,
    since local = v103 + our edits.
  - Then open the published page and spot-check 3.7, 3.15, 3.16, 3.25 and 1.37 for images.
- Numbering: **Priya may quote LIVE (v103) numbers.** Local numbers moved after several inserts: her "3.17" on
  10-26 evening was the kinematics control, which is 3.25 locally. Map with `_goal_page/fig_map.csv` (last renumber)
  and `docs/status/GOAL_PAGE_FIG_MAP_2026-10-26.csv` (pre-10-26 → current).

### 0b. RUNNING (background in THIS session; may die with it)
- **Redraw chain** `S/run_relabel.sh`, log `S/relabel.log`, started 18:45. Steps:
  1. plot-direction / key / binned / recovery marker: DONE;
  2. basis decoder `--summarise` was running at 19:30;
  3. then `iti_area_maps.py`, `iti_area_maps.py --compare`, `iti_where_decoder.py` (full, ~25 min);
  4. then `iti_licks.py --key` again (3.8 / 3.9 / 3.22 with the star fix);
  5. then `iti_licks.py --ramp` (SEM bands on 3.21 / 3.24).

  If the log does not end with `relabel3 done`, rerun the missing steps from the worktree with the PYTHONPATH above.
  Each step redraws from saved session data; nothing is lost. Then do 0a step 1.

- **DONE 20:10: decoded vs actual landing** (`scripts/iti_where_scatter.py`, per-lick `where_preds.csv`; LOCAL page
  3.19 in-trial / 3.20 ITI). Rig names in `tr_pos` (close_L ...) are mapped to lesion-relative names (all lesions LEFT).
  Ipsilesional-bias test (Priya): `where_bias_sessions.csv`, ENCODING_FINDINGS 8.17 top. Executed licks shift ipsi
  4/4; the decoded ITI position does NOT follow (bias at fixed actual position +0.18 mm contra chronic, 0/4 ipsi;
  gain lower) -> not an ipsilesional decoder deviation.
- **DONE 20:10: full movement regression** for the post-lick response (kinematics control, LOCAL 3.25 row 5: 3-D tongue
  position, reach, speed, duration, retraction + FaceRhythm post): response survives (tjM1 contra chronic +0.86).
  **JAW still missing:** jaw kinematics exist per trial (`kinematics_all/jaw_trials.csv.gz`), not per ITI lick ->
  needs a per-lick jaw extraction from the whole-session 3-D jaw trace (ask the pose window / see pose handoff 10-21),
  then add to `_more_kin` in `iti_licks.py` and rerun `--controls`.
- **Helper agents launched ~19:45 (results may land in this session):**
  - (1) TITLE VERIFICATION: DONE 20:45 and APPLIED to `_goal_page/index.html` (details
    `_goal_page/title_verification.csv`; 46 figures, 45 high confidence).
    - Corrected: 1.25 / 1.26 / 1.27 = frozen encoder WITHOUT FaceRhythm (run `..._v7g3_t3d_basis`, md5-matched), not the
      lick-kernel run.
    - Clarified: 9.13 / 9.14 = PER-SESSION FaceRhythm encoder (`combined_encoding` fr_A_tr, 10-fold within session).
    - Confirmed: 3.1-3.6 frozen pre-stroke decoder (`attempt_coding.py`); 1.15 / 1.16 / 1.24 frozen lick-kernel run with
      FR; 1.18 frozen without FR; 1.21-1.22 per-epoch pooled refit (`epoch_kernels.py`); 1.28 self-fit minus frozen;
      1.44-1.47 per-session encoder; 7.1-7.6 all-session shared encoder (`allsession_encoder.py`; the page may not
      define "Layer 2"); spot checks in sections 5 / 6 / 8 correct.
    - Writer not found: 6.1 (`fig7_residual_far_contra_16px.png`; frozen frame supported only by ENCODING_FINDINGS
      8.4).
    - Many page PNGs were drawn by scratchpad scripts outside the repo (`fig_goal_summary.py`, `fig_pooled.py`,
      `fig_layer2.py`, ...; in S). Folding them into scripts/ is a to-do.
  - (2) PUBLISH by a subagent: FAILED 19:50 -- subagents in this session do not get the Artifact tool either. Publish
    from a NEW session via section 0a (the page is still v103).

### 0c. WAITING ON OTHERS
- **sabatinilab-5b:** "go" for the kinematics_all swap sent 15:50; Priya runs the rename lines 5b gives her. When the
  table is live, re-run `python -m wfield_local.stray_licks`. Move the old `<E>/stray_licks/sessions` aside first; it
  skips sessions already done. These are derived outputs, so no approval is needed for that rename. Then update
  ENCODING_FINDINGS 2.6.

### 0d. TO DO (in order)
1. **Per-time-bin significance bars on the time-course figures** (Priya 19:30: "for the post-lick response curves and
   decoder curves -- significance testing and mark with lines (colored per epoch) where in the time bins curves
   differ from pre"). Plan, agreed in chat:
   - one helper in `scripts/iti_licks.py`, e.g. `sig_bars(ax, X_by_animal_epoch, lags, colors)`;
   - for each post epoch, `ak.boot_delta_columns(post, pre, rng)` with `{animal: array (n_sessions, n_lags)}`. This is
     the same paired animals → sessions draw as every dot-plot star, vectorized over time bins;
   - then `ak.boot_p_columns(draws)` → `ak.bh(p, q=0.05)` across the bins of that curve;
   - draw contiguous significant runs as thick horizontal segments in the epoch color (`EPOCH_LINE`) in a strip below
     the curves, one row per post epoch (x in data, y in axes: blended transform);
   - say "paired nested bootstrap per time bin, BH-FDR q = 0.05 across bins" in the suptitle and caption.

   Figures and their data:
   - 3.7 left / middle: per-session decoder per lag, `decode_sessions.csv` (label, animal, epoch, set, target, lag, r)
     → redraw via `--plot-direction`.
   - 3.8: frozen decoder per lag, `decode_frozen_lags_sessions.csv` → the `key_figures` block (`--key`).
   - 3.14: basis decoder time courses (`iti_basis_decoder.py --summarise`).
   - 3.21 / 3.24: `ramp_figure`, animal-mean PETH normalized by pre amplitude. Test on per-session PETHs (same
     normalization) → `--ramp`.
   - 3.22: `iti_ramp_sessions.png`, raw + normalized rows, per-session PETHs → `--key`.
   - 3.28 orofacial vs control PETHs (`iti_area_maps.py --compare`); optionally 3.27.

   Then update captions and republish.
2. TE partial-refit line figure (page 1.37): the title still names only the target null. Add the landing − target
   lick-kernel contrast, which is significant but not drawn (`boot.csv`).
3. 3.9 caption: add the lick-count control line (matched 40 licks / session: same numbers; `iti_nmatch_check.py`).
4. stray_licks re-run (0c).
5. Older open items: PRE-level TE group-kernel maps (replace 1.20-1.21); epoch-kernel drift-RATE test (8.18);
   every-lick window (Priya to confirm first-lick primary); June→August drift ceiling; lesion burden /
   leave-PS93-out; share deletions / nokin retirement (Priya decides).
6. **Push**: origin has everything through 66a287e. Pushed through b644af5 (20:05). Later local commits need a cherry-pick push:
   ```bash
   git fetch origin; git worktree add --detach ../wf_push_tmpNN origin/main; cd ../wf_push_tmpNN
   git cherry-pick <commits>; git push origin HEAD:main   # retry: fetch + rebase origin/main
   ```
   The pre-push tests take ~7–14 min. Never rebase the local branch, never force-push. `S/push28.sh` is the template.

### 0e. Results since 13:50 (PROVISIONAL; ENCODING_FINDINGS 8.17 / 8.20, FINDINGS 4d vii-x)
- **ITI lick rate (BEHAVIOR):**
  - PS92-94 drop acutely and stay 14-28 % below pre chronically.
  - Only PS95 ends above pre: about 4×, from near-zero pre ITI licking.
  - Rate tracks far-contra hit rate within animal (r +0.67; +0.51 with the log-day trend removed, PS94 ~0).
  - So rate is not the cause of the decoder's ITI loss.
- **Where decoder:**
  - The ITI loss is LATERAL, in every basis (−19 to −20 %, 4/4); outward position is weakly decoded and unchanged.
  - In-trial: outward drops acutely and recovers; angle ends above pre (+7 to +11 %).
- **ITI vs in-trial, paired test (3.9):** chronic −0.062 [−0.110, −0.014], 3/4 animals (PS93 0.00). Say "in three of
  four animals", not "confidently".
- **TE partial refit (1.37):**
  - Landing-labeled minus target-labeled lick kernels: lower chronically in every area (contra SSp-m −0.019, cortex
    −0.024; 0/4 up).
  - Acute target advantage falls (−0.126), back by chronic.
  - Same without PS95_0907.
- **Kinematics control + FACE (3.25 row 4):** the larger post-lick response survives regressing out reach + speed +
  post-onset FaceRhythm (tjM1 contra chronic +0.74 [+0.22, +1.26], SSp-m +0.63 [+0.14, +1.11]); unchanged from reach
  + speed alone.
- **Ipsilesional bias? (3.19 / 3.20):** no. Executed licks shift ipsi in every epoch (4/4); decoded ITI positions stay
  near pre, so the bias at fixed executed position is CONTRAlesional (+0.32 acute, +0.18 chronic mm) and the ITI gain
  is lower. In-trial: no consistent bias.
- **CORRECTION: in-trial r is lower too** (-0.146 acute -> -0.063 chronic, CI excludes 0), smaller than ITI; it follows
  lick accuracy (target-only reader -0.246 -> -0.036; session corr 0.82). Never say the change is ITI-only; say ITI
  loss = remapping, in-trial loss = mostly deviated licks read as on-target. Page captions 3.9 / 3.10 may still say
  'in-trial recovers' -> soften to 'smaller and shrinking' (to do).
- **Ipsi-deviated in-trial licks look like on-target licks** (more so after the stroke): decoder follows 26 % of a
  within-spout deviation pre, 8 % acute, 14 % chronic (4/4 lower); spout layout unchanged (ENCODING_FINDINGS 8.17 top).
  Candidate figure for the page (not drawn yet): decoded diff vs actual diff per epoch.
- **Full tongue + face regression (3.25 row 5):** the larger post-lick response survives (tjM1 contra chronic +0.86
  [+0.24, +1.55]). Jaw not yet included.
- **Decoder AUC (Priya):** answered "keep the time-binned decoder (3.12-3.13) as the one number per session"; the area under
  the per-lag r curve double-counts overlapping lags; ROC-AUC needs a binary target (an ipsi-vs-contra AUC was offered
  as an optional supplement).
- **Decoder r vs AUC:**
  - r = information about lick position (decoded vs actual lateral tongue-tip position).
  - AUC = amount of activity (dF/F integrated over a window); not ROC.
  - They can move independently: post-stroke AUC is up, frozen ITI r is down.

### 0f. NEW PITFALLS (evening)
- **The Artifact tool can be switched off mid-session.** Keep the page source somewhere a new session can publish from
  (now `_goal_page/` in the worktree). Pass every changed PNG in `files`.
- **Regex over the page:** `<figure id="...">.*?X` with re.S spans ACROSS figures. This hit Fig 3.1, whose title and
  caption were overwritten, then restored. Match inside one figure: `<figure id="(fig[^"]+)">(?:(?!</figure>).)*?X`.
- **Escapes in Bash:** a `\\n` inside a Bash heredoc reaches Python as a real newline (the tool layer unescapes it).
  Build backslashes with `chr(92) + "n"`, write the script with the Write tool, or use Edit.
- **`renumber_sections.py`** defaults to `goal_page/index.html` next to the script. In `_goal_page/`, pass the
  index.html path explicitly.
- **Background jobs:** a script launched with `&` inside a run_in_background shell gives no completion notice (wait on
  a log line instead). Define every variable (`S=...`) inside chain scripts; one chain ran 8 steps with `$S` empty.
- **Editing a script while a chain runs:** a step already started keeps the old code. 3.8 / 3.9 were redrawn before the
  star fix, hence the queued `--key` rerun.
- **Old-design vs new-design figure names:** `partial_refit_lines.png` = OLD design (1.38);
  `te_partial_refit_lines.png` = TE design (1.37).

# EARLIER 10-26 (history; superseded by section 0)

## RUNNING as of 16:20 (history)

### NOW (16:20) -- read this block first
- **DONE 15:46: TE partial refit** (exit 0; ENCODING_FINDINGS 8.20 top, FINDINGS 4d x; LOCAL page 1.37 as
  `te_partial_refit_lines.png` -- NOT `partial_refit_lines.png`, which is the OLD design's figure, now 1.38). Result:
  landing-labeled minus target-labeled lick kernels lower chronically in every area (0/4 up); acute target advantage
  falls. Same without PS95_0907. To do: retitle the line figure for the TE nulls + draw the landing - target contrast.
- **"go" SENT to sabatinilab-5b 15:50; waiting for them to report the table live.** Was: AFTER the refit exits: message sabatinilab-5b "go" for the kinematics_all swap (their regenerated table in
  `session_poses/pose3d/kinematics_all__tmp_20261026/`: only daq_peak_s differs, diff checked; Priya APPROVED the rename
  10-26 and runs the lines 5b gives her). Then re-run `python -m wfield_local.stray_licks` (patched 1d6e227; move the old
  `<E>/stray_licks/sessions` aside first -- it skips sessions already done; derived outputs, Priya's OK not needed to
  rename) and update ENCODING_FINDINGS 2.6.
- **DONE 14:35: frozen decoder in other bases** (`iti_basis_decoder.py`): result holds in joint LocaNMF and the pre-stroke
  SVD basis (ENCODING_FINDINGS 8.17 top, FINDINGS 4d vii); figure added to the LOCAL page after 3.13 (unpublished).
- **WHERE decoder** (Priya 10-26: angle / distance together, "or lateral location instead of angle"):
  `scripts/iti_where_decoder.py` DONE 15:00 (ENCODING_FINDINGS 8.17 top, FINDINGS 4d viii); on the LOCAL page as 3.15
  (unpublished) (it picks up the basis
  folders if present and raises if a basis file does not line up). Targets lateral (side_ml), outward (out_ap), 2-D
  vector r, angle / reach polar check; prints reach spread per epoch. -> page next to 3.9.
- **ITI lick RATE vs recovery** (Priya 10-26: "does inter-trial licking correlate with recovery (and exceed pre-stroke
  levels)"): `scripts/iti_lick_rate.py` DONE 15:40 (ENCODING_FINDINGS 8.17 top, FINDINGS 4d ix; LOCAL page 3.11; `--plot-only`
  redraws from the CSVs). Answer: dips with the deficit and returns, exceeds pre only in PS95. Rate = ITI licks / ITI minutes (next strobe - stop, clipped 0..60 s), no-contact rate, per trial, contact share;
  change vs pre + within-animal correlation with far-contra behavior (raw, log-day detrended). Earlier count check:
  fewer ITI licks post-stroke in PS92-94 (so exceeding pre is unlikely in counts; rate is the fair test).
- **Page publish pending**: files = iti_area_compare.png, iti_basis_decoder.png, iti_where_decoder.png, iti_lick_rate.png, iti_direction_decoding.png (3.7 now with SEM bands), iti_where_decoder_bases.png (new 3.16) + every section-3 PNG redrawn 10-26 evening (frame + measure labels, stars above data, SEM on 3.21 / 3.24); every figure now has a title (.ftitle, fig_titles.csv in the scratchpad),
  te_partial_refit_lines.png
  (all copied into S/goal_page; renumbered 16:00, map in GOAL_PAGE_FIG_MAP_2026-10-26.csv).
- Pushed to origin through e9656c8; LOCAL commits since (1d6e227 stray fix, b60d363, ff61daa, d47d32c, + this) need a
  cherry-pick push (memory note: never rebase this branch; cherry-pick onto origin/main in a temp worktree).


1. **DONE 11:05 (areas 08:49, basis 10:10, self-fit 11:05; first readouts ENCODING_FINDINGS 8.20, page 1.34 / 1.35,
   `scripts/te_readout.py`). TE partial refit now running (`S/partial_te.log`).** 1. **Encoder redesign, FROZEN_DESIGN=te** (`S/run_te.sh`, log `S/te_chain.log`; started 06:34):
   frozen AREA run `frozen_encoder_gate-attempted_v7g3_t3d_frA_te` (log `S/frozen_te.log`, `--no-drop-spout`, default
   variants) → PIXEL basis run `..._frA_te_basis` (FROZEN_PIX_MUSALL=1; `S/frozen_te_basis.log`) → self-fit
   (`session_vs_frozen.py --tag ..._frA_te --variants joint no_target no_execution no_contact no_task no_movement`, `S/svf_te.log`).
   Design (DECISIONS 2026-10-26; code `wfield_local/frozen_encoder.py` TE_BLOCK + `te_lick_parts`): TARGET = pos_enl,
   pos_cue + lick-locked kernels labeled by the trial's target spout (post-cue, pre-cue/ENL, cam1-only licks; never ITI);
   EXECUTION = unsplit 3-D tongue/jaw, exec angle, all FaceRhythm, lick-locked kernels by 3-D LANDING (+ unknown), ITI
   licks included; CONTACT = every DAQ-detected lick of the session (Priya: contacts just outside trials count); nuisance
   timing / other. Frames = whole engaged session incl. ITI (14 % of frames pre, 29 % acute). Smoke PS92_0806 / 0819 OK.
2. **DONE 07:55 -- ITI chain v4** (wide decoding lags -1..+1 s): `S/iti_full4.log` (re-extract + summary) → `S/iti_key4.log` (key
   figures) → `S/iti_ctrl4.log` (controls) → `S/iti_binned.log` (binned frozen-decoder sweep). Outputs `<E>/iti_licks/`
   (session npz backups of earlier versions: `sessions_v1..v3`, figures `figures_v3`; derived, deletable).
   At 06:59: TE frozen area run in its pre-statistics pass (17/31 pre sessions); ITI v4 extraction DONE (summary
   running, then key / controls / binned); whole-cortex maps (job 4) STARTED; old partial refit 64/101.
3. **DONE 08:31 (results ENCODING_FINDINGS 8.19 top; figures in its partial_refit/figures) -- Lick-kernel partial refit, OLD design** (`S/run_partial_ls.sh`, 7 workers, `S/partial_ls7.log`; 62/101 at 06:56).
   Superseded for the contest by the TE design, but let it finish: then `run_partial.sh`-style `--plot-only` with
   `--tag gate-attempted_v7g3_t3d_frA_ls`.
4. **DONE (re-run 08:30 on RAW dF/F AUC; per-area normalization dropped) -- Whole-cortex supplement** (Priya 10-26: "non-orofacial per-area activity maps ... supplemental figure with VIS
   areas"): `scripts/iti_area_maps.py`, queued to start when the ITI re-extraction finishes (`S/run_area.sh`, log
   `S/iti_area.log`) -> `iti_area_maps.png` (per-area response / build-up AUC change from pre on the Allen atlas,
   significant areas outlined), `iti_area_peth.png` (non-orofacial PETHs: VISp, VISl, VISam, RSPd, SSp-bfd / ul / ll / tr,
   ACAd, PL), `iti_area_groups.png` (orofacial vs visual + RSP, session dots) -> page supplement (section 3).
5. **DONE (all 10-26 work through 13:00 on origin)** -- Push of the 10-26 commits: the first attempt hit a rebase CONFLICT in DECISIONS.md (another session appended a
   10-25 FaceRhythm entry at the same place); resolved by keeping both entries (theirs first), rebase completed, push
   retrying (task in this session). CONFIRM on origin: `git log origin/main --oneline | grep "Handoff 2026-10-26"`.

## TO DO (in order, with why)

A0. **DONE v98.** **Whole-cortex supplement → page** (running job 4): say in the caption whether the larger response / weaker
   build-up are orofacial-specific or cortex-wide (orofacial vs visual + RSP test).
A. **DONE v98 (results ENCODING_FINDINGS 8.17 top, FINDINGS 4d iii-iv).** **ITI v4 outputs → page**: frozen time courses now -1..+1 s (3.8), refreshed 3.9 / 3.12 / 3.13; binned decoder
   (`iti_binned_sweep.png` = configuration sweep, boxed = chosen on PRE only; `iti_binned_summary.png` = one number per
   session at the chosen configuration, raw + normalized, all areas / orofacial / visual+RSP). Priya asked for this
   ("binned time dynamics ... single-number decoder comparison across sessions").
B. **(i) + (ii) DONE 09:00** (ENCODING_FINDINGS 8.17 top, page 3.9 caption; `scripts/iti_contact_check.py`): no-contact
   in-trial licks recover too; paired ITI - in-trial gap widens chronically -0.062 [-0.110, -0.014]. (iii) DONE 09:30 (`iti_recovery_marker.py`,
   page 3.10): in-trial tracks the recovery time course (+0.62; spread-controlled +0.53), ITI does not;
   beyond the time trend +0.16 n.s.
   B. **3.9 decisive checks** (Priya: "3.9 looks really good — is it as important as I think?"): (i) in-trial licks
   WITHOUT spout contact through the frozen decoder (does in-trial recovery ride on contact?); (ii) session-paired
   nested test of the ITI − in-trial frozen difference (per animal chronic gap: PS92 -0.11, PS94 -0.16, PS95 -0.24,
   PS93 -0.01). (iii) **Recovery marker** (Priya: "perhaps a marker of recovery?"): within animal, does the in-trial
   frozen decoder track that day's behavior (far-contra hit rate / pre, `recovery_timing.series`) across post sessions,
   and the ITI one not? Chronic behavior has no between-animal spread, so it is a within-animal time-course test.
C. **Frozen unique + self-fit DONE (8.20).** Pixel CHANGE maps of the group parts DONE (page 1.36, `frozen_pixel_maps.py` now shows `execution`). Still open:
   PRE-level group-kernel maps (each group's evoked part per spout in pre; replace 1.20-1.21);
   pixel unique maps; TE partial refit readout when it lands (`partial_refit.py --plot-only --tag ..._frA_te`
   with FROZEN_DESIGN=te). C. **TE readouts** when run 1 lands: unique(target) vs unique(execution) (frozen, epoch − pre, nested; per area +
   pixel); self-fit remapping; GROUP-KERNEL maps = each group's summed contribution averaged around the event per spout
   (from the basis run's per-group parts, PART_VARIANTS) — replaces page 1.20–1.21 single-beta maps and the contests
   1.33–1.36. **TE partial refit: code DONE (61743c3) and QUEUED** (`S/run_partial_te.sh`, log `S/partial_te.log`; waits
   for `done` in te_chain.log and `exit` in partial_ls7.log, then 6 workers; ~11 design builds x ~2 min per session ->
   ~6 h local, no O2). Sets target / target_spout / target_lick / execution / exec_land / contact / target_land; nulls:
   ONE block permutation of trial spouts fed to raw_blocks AND te_lick_parts (`tpos_perm`), landing weights circularly
   shifted along the landed licks (`land_roll`, 10-90 %). Smoke PS92_0819: target null changes only pos_enl / pos_cue /
   lick_onset@tgt:*, landing null only lick_onset@land:*, defaults rebuild identically (label agreement 0.11).
   Decoder / encoder signals = per-session adopted SVD x U_atlas averaged per Allen area (NOT a shared LocaNMF basis;
   Priya asked 10-26) -- registration drift cannot explain ITI-down / in-trial-recovered (same sessions, same features).
D. **Pose window TOLD 11:35 (sabatinilab-5b): it regenerates the table into kinematics_all__tmp_20261026/, diffs vs live (only daq_peak_s may differ), sends me the diff; swap = rename on M: -> needs Priya, and only after the TE partial refit ends (~16:30). stray_licks PATCHED (1d6e227, lick_uid join); re-run after the refit (memory).** Tell the pose window: `pose3d/kinematics_all/licks.csv.gz` daq_peak_s was rounded to 4 significant digits (writer fixed
   c2aa922; regenerates on its next kinematics step); `stray_licks.py` (ENCODING_FINDINGS 2.6) must be re-run with exact
   times (lick_uid → licks_all.csv).
E0. **Frozen decoder in other bases RUNNING** (Priya 10-26: LocaNMF + the pre-stroke SVD basis; `scripts/iti_basis_decoder.py`,
   log `S/iti_basis.log`): joint LocaNMF bases are per animal, fit 08-15 on 11 PRE sessions (06-06..08-14), every session
   PROJECTED from current SVTcorr; SVD basis = te_basis fit/spatial_<animal>.pkl. Decoder trained on August pre only.
   Per-animal basis is fine (decoders are per animal; no cross-animal basis needed). -> page next to 3.9.
E. Open from before: epoch-kernel drift-RATE test (8.18; within-chronic r per day vs pre rate); the narrow every-lick
   window (Priya to confirm first-lick primary); June→August drift ceiling; lesion burden / leave-PS93-out; share
   deletions / nokin retirement (Priya decides).

## Results of 10-26 (all PROVISIONAL; details ENCODING_FINDINGS 8.17–8.19, FINDINGS 4d / 4e)

- **Self-initiated ITI licks (KEY, page 3.7–3.30):** within-session direction decoding intact; FROZEN pre-stroke decoder
  down every epoch for ITI licks (-0.135 r chronic, -22 %), largely recovers for cued in-trial licks (-20 % acute → -9 %
  chronic). Each lick evokes a larger response (~2× acutely, +0.6–0.8 chronic, also isolated licks; post-stroke bouts are
  SHORTER); slow build-up before the lick weaker chronically in tjM1 / SSp-m (3/4). Not lick kinematics (post-stroke ITI
  licks slightly slower; kinematics-adjusted / range-matched responses unchanged) and not face movement (face-adjusted
  build-up still weaker). "Relative" direction for ITI = relative to the PREVIOUS trial's spout (spout retracted in ITI).
- **Lick-count control (Priya: "is this a confound")**: ITI licks are fewer post-stroke (PS92-94), but the frozen
  decoder on test sets matched to 40 licks per session gives the same numbers (ITI -0.136 / -0.115 / -0.123,
  in-trial -0.144 / -0.080 / -0.064); more licks -> slightly lower r within epoch. `scripts/iti_nmatch_check.py`
  (uses the complete v3 extraction). Not on the page yet: add one line to the 3.9 caption.
- **FaceRhythm lick factor (1.16 / 1.17):** chronic trial-by-trial unique gain in bilateral orofacial motor cortex with
  LOWER trial-to-trial variance (-40 % / -44 %) → tighter coupling, not more variability.
- **Block-permutation nulls** reproduce the single-shift partial-refit contrasts (1.33 lines).
- **Landing vs target lick kernels, lagged FR, self-fit with lick kernels** (8.19; 1.35 / 1.36 / 9.21): landing kernels do
  not absorb the chronic target term; no shift toward cortex leading the face; remapped spout information is shared
  between target and lick-locked spout terms (old design — superseded by TE).
- **Epoch kernel fits** (8.18; 1.20 / 1.21): lick kernels diverge steadily from early pre but not yet beyond drift; the
  single-lag lick-kernel maps are leftover terms (collinear with templates / kinematics) → replaced by TE group kernels.

## PITFALLS (10-26)

- Heredocs / `python -` with `\n` inside string literals STILL break (hit twice more): write patch scripts with the Write
  tool; for small fixes use the Edit tool (Edit needs a prior Read of that file in the session).
- The whole-session 3-D table rounds daq_peak_s (above); `t_peak_3d` (<= 6.9 s) is fine.
- ITI lick `position` = the trial WINDOW's label (tail = trial just finished, lead = UPCOMING trial); use the previous
  trial's spout (`prev_pos`).
- `_pix_musall_variant` must accept `novar_*` (fixed); `frozen_pixel_musall.py --layout novar` uses the real variable bases.
- `summarise_musall` rewrites `frozen_musall_r2*.csv` with only the listed variants: re-score extra variants into a
  SEPARATE tag (copy fit/templates.pkl, pre_stats.pkl, spatial_*.pkl), as `..._frA_ls_basis_novar`.
- Editing `iti_licks.py` while its chain runs breaks later steps if array shapes change (LAGS_D): wait for the chain.
- TE runs need FROZEN_LICKSPOUT / LICKLAND / FR_SPLIT / FR_LAGS unset (TE raises otherwise); frozen_encoder backups:
  `S/frozen_encoder_backup3.py` (before TE).
- Push: pre-push tests ~8–14 min; races with other sessions retry automatically (temp worktree loop).
- PAGE IMAGES: publishing index.html alone does NOT upload new / changed PNGs (v98-v102 had 9 broken figures). Always
  pass every new or regenerated image in the Artifact `files` map (root = goal_page) on the same publish.
