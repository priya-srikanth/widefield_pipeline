# K08 2026 figure provenance, by final panel

One row per panel of the **submitted** Research Strategy figures, lettered as in the final legends (Figures 3, 4
and 5). For each panel: which script drew or placed it, the file it read (as staged in the grant figures folder,
`~/Documents/Sabatini Lab/Grants & applications/K08/Proposal/2026/figures/`), and the server source behind that
file. Written 2026-11-17 from the compositors (`make_fig1/2/3.py`), the renderers and their source comments, and
[`FIGURE_SOURCES.md`](FIGURE_SOURCES.md). **`FIGURE_SOURCES.md` is the narrative record** (why each source was
chosen, the corrections, the stats columns). It stops at 9 Nov 11:28. Where a panel changed after that, this file
says so.

Server paths are relative to `MICROSCOPE/Priya/Widefield/` unless they say otherwise. `lmm_hidden` =
`labcams/grant_figures/lmm_hidden_20261103/`. The deck's own master key, for any panel that started life on an
Overview deck slide, is `presentations/_source_PI_overview_2026-11-02/figure_provenance.csv` (slide, shape, original
path, transform).

The compositors use draft letters (a, b, c, ...). The final letters were set by hand in Illustrator, so the
**Draft** column records which draft panel each final panel came from.

## Figure 3: behavior and deficit  (`make_fig1.py` → `Figure3_behavior_DRAFT4.svg` → `Figure3_behavior_relettered_v2.ai`)

| Final | Content | Draft | Drawn / placed by | Staged input | Source |
|---|---|---|---|---|---|
| A | VLS infarct, day 7 (DAPI / NeuN / GFAP) | b | `make_fig1.py`, 600 dpi raster of the `.ai` | `airaster/panel_hist.png` | `BWF CAMS/Research proposal/2026/Behavior_CAMS_fig.ai`, panel B (laptop) |
| B | Ipsilesional thalamus, ± GFAP, Paxinos outlines | c | `make_fig1.py` (`thal_panel`); outline options from `thalamus_overlay_options.py` | `airaster/panel_thal.png` | VLS stroke methods paper, `histology_panels/PS46-55_sections/` (laptop) |
| C | Videofluoroscopy frames | d | `make_fig1.py`, raster of the `.ai` | `airaster/panel_fluoro.png` | `AAN CDA/Fluoroscopy_AAN_CDA.ai`, panel A (laptop) |
| E–J | Swallow rate, interval, licks per swallow, lick rate, bolus amount, esophageal transit, by severity | e–j | `make_fig1.py` (`flu()` crops) | `fluoro/all_graphs.svg` | `MICROSCOPE/Priya/Fluoroscopy/grant figures/all_graphs.svg` |
| K, L | Six-position task, trial structure | a | `make_fig1.py`, raster of the `.ai` | `airaster/panel_task.png` | `BWF CAMS/Research proposal/2026/6-spout_behavior_CAMS_fig.ai` (laptop) |
| M | Hit rate by spout, change vs own pre | k | `render_grant_panels.py` `hit_rate_single()` | `hitstats/epoch_1bdelta_behaviour_by_position{,_sessions}.csv`, `hitstats/epoch_1b_behaviour_by_position_sessions.csv` | `rebuild_20261103/labcams/grant_figures/<tag>/epoch/data/` (full tag in `FIGURE_SOURCES.md`, "Hit-rate panel: corrected source"). Stars: Bonferroni interval excludes 0 |
| N | cam1 + cam4 frames, tracked tongue / jaw / spout / nose | m | `make_camframes.py` | `camframes/PS92_20260807_far_R_t18_cam{1,4}_raw.png`, `camframes/kp_t18.json` | `MICROSCOPE/Priya/Claude outputs/k08_camframes/` (candidates t14 / t18 / t41 / t46; t18 used). Pose models cam1 `lp_cam1_ctx_prod_20261019`, cam4 `lp_r4_prod_20261015` |
| O | First-lick aim by spout | l | `render_grant_panels.py` `first_lick_aim_single()` | `aimstats/aim_by_spout_{sessions,tests}.csv` | `LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/`. Stars: `sig_bonf` |
| P | 3-D tongue and jaw paths, front view | n | `paths_panel.py` → `Fig3P_paths_front.svg` (placed in Illustrator) | `pose3d_paths_pooled.csv`, `paths3d_front_sig.csv` (kept in `panels_src/`) | Pooling of `kinematics_pooled_figs.fig_paths3d`; markers = cluster permutation on session changes, labels permuted within animal, p < .05/3 |

Traced 17 Nov (supersedes the earlier "not traced" note here):

- `Fig3O_paths_front.svg` is the output of `session_scripts/paths_narrow.py` (9 Nov 23:06), written as
  `Fig3O_paths_narrow.svg` and renamed on delivery; `Fig3O_paths_short.svg` is `session_scripts/paths_short.py`
  (23:29). All three paths SVGs lost about 900 bytes of font declarations to `fontnorm.py` on the way in, which is
  the only difference between the session copies and the ones in the grant folder.
- Lettering: her hand-lettered `Figure3_behavior.svg` ran A, B, C, then E-P, with no D.
  `session_scripts/edit_svg.py` closed the gap (E->D, F->E, ... P->O), so the submitted
  `Figure3_behavior_relettered_v2.svg` carries a continuous A-O, and the v18 legend matches it
  ("(C) Videofluoroscopy frames ... (D-I) Swallowing ... (J) Six-position task ... (O) 3-D paths").
  The "C ... E-J" legend was the pre-shift draft.
- `paths3d_front_sig.csv` was a one-off export of `kinematics_pooled_figs._paths3d_sig`, which the pipeline computes
  for the figure but never writes to CSV. The original export was not saved, but
  `session_scripts/export_paths3d_front_sig.py` reproduces it exactly (verified 17 Nov against the staged file:
  1476 / 1476 rows, 960 significant bins), from `curves.csv.gz` alone. `aug` is derived from the curves table
  itself through `config.early_pre` (31 August-pre sessions), so no pose3d kinematics snapshot is needed.

## Figure 4: FaceRhythm  (`make_fig2.py` → `Figure4_facerhythm_DRAFT4.svg` → `Figure4_facerhythm_Dsplit_light.ai`)

| Final | Content | Drawn / placed by | Staged input | Source |
|---|---|---|---|---|
| A | Tracked mesh on the face | `make_fig2.py` | `fr/fr_mesh_face.png` (`panels_src/fr/`) | Overview deck slide 72 (`f3_FR_master_face.png`); master face PS95_20260812 |
| B | Four of twelve frozen factors | `make_fig2.py` | `fr/fr_4factors.png` (`panels_src/fr/`) | `DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/report/factor_gallery_abs.png` (deck slide 74) |
| C | Face speed per point, far contra, epoch minus pre | `make_fig2.py` (crop) + `make_cbars.py` (`cbar_speed`) + `fr_mesh_underlay.py` | `fr/aligned_maps_lick_perm_mesh.svg` | FaceRhythm aligned maps (`scripts/face_rhythm_aligned_maps.py`), lick-aligned, permutation clusters |
| D | Lick-factor loading change per point | `make_fig2.py` (crop) + `make_cbars.py` (`cbar_loading`) + `fr_mesh_underlay.py`; the split and light versions by `session_scripts/split_d.py` then `session_scripts/declip.py`, re-saved from Illustrator as the `.ai` | `fr/session_maps_scaled_lick_mesh.svg` | FR page (`Widefield/fr_page_src/fr_page/`) |
| E | Side specificity of lick-factor loading | **`make_fig2.py`, crop of `session_scalars.svg`** — confirmed 17 Nov: the submitted SVG carries that crop's own strings ("loading asymmetry", "PS92 sessions") and none of the re-plot's labels, so `render_grant_panels.py` `fr_side_specificity()` was never used in the figure | `fr/session_scalars.svg` (`panels_src/fr/`, `_frstage/`); `frstats/peak_{epoch_stats,sessions}_side.csv` | FR page `fig/` and `lmm/` (`Widefield/fr_page_src/fr_page/`). Stars: Holm-corrected within-animal permutation |

## Figure 5: decoder and encoder  (`make_fig3.py` → `Figure5_decoder_encoder_DRAFT4.svg` → `Figure5_decoder_encoder_cuedD.ai`)

Rows A–B, D–E and F–H are each rendered as one canvas by `render_fig4_rows.py` (`fig4_row_AB.svg`, `fig4_row_DE.svg`,
`fig4_row_FGH.svg`; axes positions in `rows_fig4_layout.json`) and placed at scale 1.

| Final | Content | Drawn / placed by | Staged input | Source |
|---|---|---|---|---|
| A | Frozen / refit / gap, pooled | `render_fig4_rows.py` `row_AB()` | `frbasis/sessions.csv`, `frbasis/stats.csv` (basis `lnmf`) | **`scripts/frozen_refit_basis_compare.py`** -> `state_space_recovery/frozen_refit_basis/{sessions,stats}.csv` (9–10 Nov, August-only pre; staged 9 Nov 19:58). That script is on `origin/main` but postdates the 9 Nov checkout, which is why it was not found earlier. This replaced the `epoch_5rmo_frozen_refit_overall_cue_working` source in `FIGURE_SOURCES.md` (that one is still drawn by `render_grant_panels.py` `frozen_refit_gap_pooled()` but is not the submitted panel). Stars: Bonferroni-corrected paired bootstrap vs pre |
| B | Readout offset of the first cued lick | `render_fig4_rows.py` `row_AB()` (earlier: `render_grant_panels.py` `offset_cued_first()`) | `data/lick_cued_first_lick_{sessions,lmm}.csv` | `lmm_hidden/lick_cued_first_lick_*`. Stars: `star` |
| C | Response around lick onset by region and epoch | `make_fig3.py`, deck raster with redrawn axes | `deck2/f3_traces_4roi.png` | Overview deck ROI-trace row (slide 96 family, `f3_cued_ROI_row.png` in `FIGURE_SOURCES.md`), cut to tjM1 + SSp-m both sides |
| D | Lick-response AUC, fold of own pre | `render_fig4_rows.py` `row_DE()` | `data/cued_ramp_w01_{sessions,verify}.csv` | `lmm_hidden/cued_ramp_summary_row2_lmm_w0-1_{sessions,verify}.csv`. Stars: `boot_star` (see `FIGURE_SOURCES.md`, "Correction, 9 Nov"). Final "cued D" version: `Fig5D_cued.svg` (10 Nov 01:15), placed in Illustrator |
| E | Contra : ipsi balance, far contra | `render_fig4_rows.py` `row_DE()` | `data/lick_lateral_ratio_auc_docked_{sessions,lmm}.csv` | `lmm_hidden/lick_lateral_ratio_auc_docked_*`. Stars: `star` |
| F | Balance (SSp-m) vs far-contra aim, per session | `render_fig4_rows.py` `row_FGH()` | `recov/balance_aim_sessions.csv`, `recov/balance_aim_models.csv`, `data/lick_lateral_ratio_auc_docked_sessions.csv` | Within-animal slope, mixed model adjusted for log time from stroke (`recov/` staged 9 Nov 14:34) |
| G | Position on pre → chronic axis vs days | `render_fig4_rows.py` `row_FGH()` | `ssdata/days_sessions.csv` | **`scripts/state_space_trajectories.py`** -> `state_space_recovery/trajectories_lick_svd/days_sessions.csv`, copied unchanged. Byte-identical, checked 17 Nov |
| H | Sessions in the axis × acute-departure plane | `render_fig4_rows.py` `row_FGH()` | `data/ss_plane_sessions.csv` | **`scripts/state_space_per_position.py`** -> `state_space_recovery/per_position_lick_svd/per_position_sessions.csv`, renamed on copy. Byte-identical, checked 17 Nov |
| I | Reference areas (Allen key) | `make_fig3.py`, raster from PDF | `airaster/panel_areakey_nolab.png` | `Widefield_CAMS_fig.ai`, panel H (v7 Figure 5H), crop x 232–322, y 316–396 pt |
| J | Regional change along the recovery axis, `frKBs` control, full and cortex-wide mode removed | `make_ss_panel.py` → `ss_full_nocw_frKBs.svg` | `ss/ss_cmap_{full,nocw}_rest_frKBs_lick_svd_w.png` | `state_space_recovery/page_figs/ss_cmap_{full,nocw}_rest_frKBs_lick_svd.png`; columns "mean over spouts" and "far contra" |

## Gaps in this record

- Staged inputs live only in the grant figures folder on the laptop (`data/`, `aimstats/`, `hitstats/`, `frstats/`,
  `frbasis/`, `ssdata/`, `recov/`, `fr/`, `ss/`, `deck2/`, `airaster/`, `camframes/`, `fluoro/`). Several of those
  folders are no longer there. The server source above is the one to re-pull from. All thirteen were still held in
  the 9-10 Nov session's scratch directory on 17 Nov, including every CSV listed in this file.
- The relettering, the Fig 4D split and the Fig 5D cued swap were done BY SCRIPT, not by hand: see
  `session_scripts/` (recovered 17 Nov). Illustrator was used only to open those SVGs and save the `.ai`
  (and, for Fig 4, to re-save a more compact SVG). Fig 5C is `make_fig3.py`, a deck raster with redrawn axes.
- **Staged inputs were often renamed on copy, so a filename that no script writes is not necessarily a file with
  no code.** Checked byte for byte on 17 Nov: `data/ss_plane_sessions.csv` is
  `per_position_lick_svd/per_position_sessions.csv` (`scripts/state_space_per_position.py`),
  `ssdata/days_sessions.csv` is `trajectories_lick_svd/days_sessions.csv`
  (`scripts/state_space_trajectories.py`), and `frbasis/{sessions,stats}.csv` are
  `frozen_refit_basis/{sessions,stats}.csv` (`scripts/frozen_refit_basis_compare.py`), all identical to their server
  originals. `data/cued_ramp_w01_*` is the documented rename of `lmm_hidden/cued_ramp_summary_row2_lmm_w0-1_*`.
- **The one file still without code:** `recov/balance_aim_{sessions,models}.csv`. Derived, not copied: no file on the
  share matches either one, and the extraction that made them from
  `iti_licks/<daqlk tag>/figdata/lick_recovery_primary.pkl` (`T` and `M`, `set == 'cued first lick'`) plus
  `balimg_primary_link.csv` ran inline in the figure session and was never written to a file. To rebuild it, see
  `session_scripts/recov_README.txt` for the columns and the model numbers it must reproduce, and
  `scripts/lick_recovery_primary.py` for the pipeline step behind the pkl.
- Where this file and `FIGURE_SOURCES.md` disagree, this file is newer for the submitted figures (Fig 5A is the known
  case).
