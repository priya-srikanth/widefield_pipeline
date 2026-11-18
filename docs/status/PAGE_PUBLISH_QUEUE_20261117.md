# Page publish queue, 2026-11-17 (encoding window, sabatinilab-3c)

Why this exists: from ~12:30 on 11-17 the Artifact tool in this session was signed in to a different claude.ai account,
so the results pages (owned by the original account) could not be read or published. Priya can switch back ~1 h later.
Everything below is approved and waiting only for the account. Before publishing each page: Artifact `read` the LIVE
version, Read every line of the saved HTML, apply the edits to THAT version (not to an older local copy), republish to the
same URL, then re-start the artifact watch. Tick each item here and commit.

## 1. Image page — https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV (live v57)

- [x] **Fig 2.3 layout fix** (published v58, version id 1794948981-2aab, 11-17) (Priya OK 11-17). Prepared from v57:
  `scratchpad/pages/img/img_v58.html` + new published file `scratchpad/pages/img/d_acc_cue_fb1117.png`
  (scratchpad = `C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad`).
  Source render: `rebuild_20261103/labcams/grant_figures/<T>/epoch_footerfix_20261117/` (plot-only redraw, numbers
  byte-identical to 11-09). Three edits, all in Fig 2.3: image, provenance clause, source path. If live != v57, re-apply
  those three edits to live.
- [x] **Section 13 (CD leave-one-out mixed models) on the rebuild** (published v58, 1794948981-2aab: table rows / input / takeaway lead with the log deficit, linear + contact as sensitivity; status row, closed-list line and section 9 sentence updated; numbers checked against the CSV) — Priya OK 11-17 (rewrite as proposed).
  Table: `rebuild_20261103/.../imaging_encoding/qc/imaging_lmm_cdloo_rebuild_20261117.csv`
  (script `scripts/cd_loo_lmm.py`, record `docs/status/DECK_IMG_FOLLOWUPS_20261117.md`). Proposed: lead with the log
  time-to-contact deficit (primary since 11-05): pre-cue tracks it (-.20, q .019); cue vs contact deficit loses its star
  (q .48); linear deficit as a one-line sensitivity (pre-cue q .054). Update takeaway, status row, closed list, and the
  section 9 "due a re-run" sentence.
- [ ] **Fig 12.1 rest frozen decoder** (cam4-only run done 11-17: `.../epoch/rest15f_restnl_cam4_20261117/`; cam4+cam1: `.../epoch/rest15f_restnl_20261117/`; Priya OK 11-17: place cam4+cam1 after the overnight RESTnl-cam1 check, cam4 as sensitivity, wording: drops acutely, somewhat below pre chronically, subacute n.s., ~half the old size, status provisional; update the FINDINGS rest line the same way; re-point the deck 15f slot) — decisions made 11-17 (PS92 kept with caveat). Run:
  `.../epoch/rest15f_restnl_20261117/{allpre,augpre}/`, record `docs/status/REST15F_RESTNL_REBUILD_20261117.md`
  (branch analysis/rest15f-rebuild-1117, 308468b5, unpushed). Status would go stands -> provisional.

## 2. Goal / encoding page — https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (last known v315)

- [x] **ITI 0..0.5 s swap** — DONE in goal v316 (1794955854-a69e, built on live v315 1794870505-aeaf): images + captions 14.8-14.11, 14.13, 14.23, 14.25 (cued windows image swapped from the same rebuild folder; baseline-sensitivity image kept, labelled pre-rebuild), 14.29 / 14.31 / 14.32, recovery markers after 14.18 (now Fig 14.18 S14b); section-14 header window sentence; status rows (lick-evoked, self-initiated (c)). Numbers from the folders' CSVs. 14.12 image / caption NOT changed (old all-ITI side_auc_0-0.5). (Priya OK 11-17 for slides and pages; DECISIONS 2026-11-17 top entry; audit
  `docs/status/deck_swap_20261117/ITI_WINDOW_AUDIT_20261117.md`). Root
  `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/<folder>/<tag>/figures/`:
  - 14.8 `vigor_iti05_20261117/lick_vigor_perspout.png`
  - 14.9 `lateral_ratio_dock_iti05_20261117/lick_lateral_ratio_auc.png` (twins `_0-1` / `_0-0.5`)
  - 14.10 same folder `lick_lateral_ratio_side_auc.png`
  - 14.11 `lateral_ratio_iti05_20261117/lick_lateral_ratio_rings.png`
  - 14.13 `lateral_ratio_iti05_20261117/lick_balance_angle_resp.png` (+ `_dec`) — replaces the 11-16 shared-y ITI panels
  - 14.23 `cued_ramp_iti05_20261117/lick_iti_ramp_summary_wide.png` (docked twin `itidock_iti05_20261117`)
  - 14.25 `cued_ramp_iti05_20261117/lick_ramp_windows_iti.png`
  - 14.29 / 14.31 / 14.32 `area_maps_dock_iti05_20261117/lick_iti_area_maps_split{,_farcontra}{,_levels}.png`,
    `lick_iti_area_compare.png`, `lick_iti_area_groups.png`
  - inline after 14.18 `area_maps_iti05_20261117/lick_recovery_markers_size.png`
  Caption conclusion changes: **14.13** ITI balance at matched angle: ALM +.06 p .005 significant; tjM1 +.02 p .58;
  orofacial +.04 p .16 (was orofacial +.15 p .010). **14.29** chronic = orofacial + contralesional SSp-ll / -tr / -ul
  (not orofacial only; chronic starred areas all-ITI 4 -> 10, docked 11). Every caption: ITI response window 0-0.5 s,
  cued stays 0-1.0 s. All ITI increases shrink ~1/3-1/2, none reverses. 14.18 / 14.20 / head-to-head unaffected.
- [x] **Fig 14.7 cued per-spout window -> 0..1.0 s** — DONE in goal v316 (1794955854-a69e, built on live v315 1794870505-aeaf) (4 images from cued_ramp_cued01; docked ITI per-spout image unchanged) (Priya 11-17 via a8, "update 113 and 114 to match"): `iti_licks/cued_ramp_cued01_20261117/<tag>/figures/lick_cued_ramp_perspout.png` (+ cue-aligned twin and `*_perspout_windows`, 0..1.0 s row first). Caption: acute far contra now below pre (n.s.); acute far mid loses its star; chronic far contra +.48*. Table: a8's ITI_WINDOW_AUDIT_20261117.md section 7.
- [x] **idx supplement next to Fig 14.13** — DONE in goal v316 (1794955854-a69e, built on live v315 1794870505-aeaf) (coordinator relayed a8 go 11-17): collapsed <details> "Fig 14.13 S1-S4", captions from PAGE_SUPPLEMENT_IDX (S-a..S-d -> S1..S4), gap >= 2 s caveat on each; 4 new files, 4 unreferenced duplicates removed (iti_area_compare/groups/peth, iti_ramp_sessions .png) to stay at 511 (Priya 11-17 via a8: collapsible, collapsed by default, clear captions). 4 figures
  (far contra all-ITI + cued, far contra docked, all-spout dec_idx / resp_idx); captions + paths in
  `docs/status/deck_swap_20261117/PAGE_SUPPLEMENT_IDX_20261117.md`. WAIT for a8's ping: the gap >= 2 s ITI dec_idx variant
  updates the caveat line first.
- [ ] Optional, only if Priya asks: a8's lick-angle controls for the marker (after 14.18), figures under
  `iti_licks/{bal_full_cuedang,iti_matched_itimatch,bal_dock_itimatch,balangle_all_itimatch,balangle_dock_itimatch}_20261117/`.

- [ ] Lick-source caption edits (**goal DONE** in v316 (1794955854-a69e, built on live v315 1794870505-aeaf): regenerated against live after the ITI edits, 110 / 110 pairs matched once (identical to the pre-live list), incl. Fig 8.6 WRONG fix; "(was Fig 14.18/19/20)" labels replaced by Fig 14.18 S14a / S14b / S14c; applied list scratchpad/pages/goal/v316/lick_source_edits_applied_v316.json) (audit 11-17, branch fix/lick-source-wording-1117): `scratchpad/pages/{img,goal,ss,kin,fr}/lick_source_edits_1117.json`. **Image page DONE** (v58, 1794948981-2aab): all 42 pairs matched once on live v57 + Fig 2.3; also set the remaining pre-cue lick-free-window mentions to cam4 only (section 1 input + table header, section 2 LP-arm input, status row, Figs 2.5 / 2.6 window + tag, Figs 7.3b / 7.3c window). Goal / ss / kin / fr still open (42 / 110 / 142 / 27 / 12); regenerate against LIVE html with `scratchpad/licksrc/make_edits.py` (PYTHONPATH=wf_licksrc1117) before applying; re-run after the RESTnl-cam1 outputs are placed. Also fix the goal caption "(was Fig 14.20)" (annotation). FaceRhythm page is not this window's — send its list to the FR window.

- Goal page later published by a8 as **v317** (1794959333-ffc1, built on v316): section 12 + Biology-in-brief only (Fig 12.2 = combined like-for-like lick_matched_panels.png; old gauss curves -> 12.2 S2; files net -2). Build any further goal edit on v317+. Then v318 (Fig 12.2 title) and **v319** (1794960591-2ba2, 12.2 S2 non-linear). a8's PI deck: hidden slide 126 inserted 11-17 -> every slide after 125 is +1 (207 slides; quiet 156/157 -> 157/158; rest maps 187/188/204-206 -> 188/189/205-207; RESTnl slides to re-point = 24-26 and 187-189 by the new numbering; match by title). Goal **v320** (1794965716-2ffc, a8): section 23 lick sentences note far-from-mouth cam1 labels counted (< 2 % effect; quiet cam1-guard re-run, Priya kept k 3.75 + current figures; QUIET_C1GUARD_OLD_NEW_20261117.md on main). Section 23 still shows PRE-REBUILD quiet outputs (a8 told Priya).

## 3. State-space (9sGELLFEwADEt9k64fXTxe, v41) and kinematics (DKcKRJKbhBoratxxRc6qZv, v31)

- Nothing pending. Re-start their watches after the switch.

## 3b. Decision pending a check (Priya 11-17)

- Switch the LP pre-cue (ENL) and ITI lick-free windows (and pre-cue / ITI lick kernels) from cam4 only to cam4 + cam1
  (cam1 `events_all`, cam1-only, droplets removed) — Priya: YES, pending the check that the extra cam1 events outside
  post-cue are real licks, not droplets (helper running; record `docs/status/CAM1_EXTRA_LICKS_CHECK_20261117.md` on
  branch analysis/restnl-cam1-1117; blind review sheet for Priya). If it passes: change `lp_licks` so the window licks
  read events_all, re-run every pre-cue / ITI figure (Fig 2.5 / 2.6 frozen decoders, encoder enl kernels, ENL-window
  maps, ITI families), old vs new to Priya before pages.

## 3c. cam1 check decided (Priya 11-17 ~18:00)

- Blind review scored (30 / 40 tongue; far-from-mouth labels = the no-tongue set). Priya: YES to cam4 + cam1 with a
  far-from-mouth guard (cam1-only events with med_out_px < -120 dropped) = RESTnlc1 v3 (commit 3bf2dd41 on
  analysis/restnl-cam1-1117). Every v2 RESTnlc1 output of 11-17 is superseded; the whole queue re-runs under v3
  (record docs/status/RESTNL_V3_RERUN_20261117.md on that branch). Fig 12.1 waits for the v3 tables.
- The 3b switch (pre-cue / ITI windows to cam4 + cam1) is approved and uses the same v3 rule; not started yet.
- Priya: no cohort-wide check of sessions where cam4 video ends before the last trial.

## 4. To do after the re-runs (Priya 11-17): clean up stale iterations

Discuss with Priya before deleting anything; deleted-derived list as in DELETED_DERIVED_20261117.md; never original data.
- Per result family on M: (rebuild_20261103/...): list the CURRENT version pages / decks point to, the older iterations
  (e.g. _superseded_glue_20261109, 11-08 vs 11-16 vs 11-17 folders, session_cache_restnl_glp* variants, deck backups
  __2026111*_*.pptx, replay copies such as state_space_recovery/svd_lnmf_gluefree_20261117/ ~7 GB of cache copies),
  sizes, and what would be lost.
- Local C: had 79 GB free on 11-17. Biggest: C:/Users/SabatiniLab/wfield_cache 444 GB =
  input_cache_rebuild_20261104 366 GB (ACTIVE: the rebuild render env's WFIELD_INPUT_CACHE; other windows use it) +
  session_cache_rebuild_20261103 78 GB (check whether still read). 161 git worktrees under Github/ (remove those whose
  branches are merged and that no running job uses; `git worktree remove`, never delete a dirty one).
