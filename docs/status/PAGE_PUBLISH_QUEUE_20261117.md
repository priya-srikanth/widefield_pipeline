# Page publish queue, 2026-11-17 (encoding window, sabatinilab-3c)

Why this exists: from ~12:30 on 11-17 the Artifact tool in this session was signed in to a different claude.ai account,
so the results pages (owned by the original account) could not be read or published. Priya can switch back ~1 h later.
Everything below is approved and waiting only for the account. Before publishing each page: Artifact `read` the LIVE
version, Read every line of the saved HTML, apply the edits to THAT version (not to an older local copy), republish to the
same URL, then re-start the artifact watch. Tick each item here and commit.

## 1. Image page — https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV (live v57)

- [ ] **Fig 2.3 layout fix** (Priya OK 11-17). Prepared from v57:
  `scratchpad/pages/img/img_v58.html` + new published file `scratchpad/pages/img/d_acc_cue_fb1117.png`
  (scratchpad = `C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad`).
  Source render: `rebuild_20261103/labcams/grant_figures/<T>/epoch_footerfix_20261117/` (plot-only redraw, numbers
  byte-identical to 11-09). Three edits, all in Fig 2.3: image, provenance clause, source path. If live != v57, re-apply
  those three edits to live.
- [ ] **Section 13 (CD leave-one-out mixed models) on the rebuild** — WAITING for Priya's OK on the conclusion change.
  Table: `rebuild_20261103/.../imaging_encoding/qc/imaging_lmm_cdloo_rebuild_20261117.csv`
  (script `scripts/cd_loo_lmm.py`, record `docs/status/DECK_IMG_FOLLOWUPS_20261117.md`). Proposed: lead with the log
  time-to-contact deficit (primary since 11-05): pre-cue tracks it (-.20, q .019); cue vs contact deficit loses its star
  (q .48); linear deficit as a one-line sensitivity (pre-cue q .054). Update takeaway, status row, closed list, and the
  section 9 "due a re-run" sentence.
- [ ] **Fig 12.1 rest frozen decoder** — WAITING for Priya's choices (lick definition; PS92). Run:
  `.../epoch/rest15f_restnl_20261117/{allpre,augpre}/`, record `docs/status/REST15F_RESTNL_REBUILD_20261117.md`
  (branch analysis/rest15f-rebuild-1117, 308468b5, unpushed). Status would go stands -> provisional.

## 2. Goal / encoding page — https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (last known v315)

- [ ] **ITI 0..0.5 s swap** (Priya OK 11-17 for slides and pages; DECISIONS 2026-11-17 top entry; audit
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
- [ ] Optional, only if Priya asks: a8's lick-angle controls for the marker (after 14.18), figures under
  `iti_licks/{bal_full_cuedang,iti_matched_itimatch,bal_dock_itimatch,balangle_all_itimatch,balangle_dock_itimatch}_20261117/`.

## 3. State-space (9sGELLFEwADEt9k64fXTxe, v41) and kinematics (DKcKRJKbhBoratxxRc6qZv, v31)

- Nothing pending. Re-start their watches after the switch.
