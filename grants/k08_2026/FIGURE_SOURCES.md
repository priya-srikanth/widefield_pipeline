# K08 figure panel provenance

Every panel pulled so far, where it came from, and what was done to it. Written because the
underlying figures are being re-rendered: when a source is regenerated, re-pull from the path in
the **Source** column and re-run the build script. Nothing here is hand-traced or redrawn, so a
re-pull is always a straight swap.

Layout follows `K08_figs_rough_draft.pptx` (9 Nov): slide 1 = behaviour, slide 2 = FaceRhythm,
slide 3 = decoder/encoder. Panel letters below are the build script's, not the slide's labels.

Paths beginning `~/Documents` are on the laptop. Paths beginning `/Volumes/Neurobio/MICROSCOPE`
are on the lab server, **which was not mounted when this was written** — anything marked
*server offline* could not be re-checked.

---

## Figure 1 — model and behavioural deficit  (`Figure1_behavior_DRAFT2.svg`)

| Panel | Content | Source | Treatment |
|---|---|---|---|
| a | Six-spout task schematic + trial structure | `~/Documents/Sabatini Lab/Grants & applications/BWF CAMS/Research proposal/2026/6-spout_behavior_CAMS_fig.ai` | .ai is PDF-compatible → `pdftocairo -svg`; cropped to 34–212 x 28–150 pt. **Live vector** |
| b | VLS stroke histology, DAPI/NeuN/GFAP, d7 | `~/Documents/.../BWF CAMS/Research proposal/2026/Behavior_CAMS_fig.ai`, panel B | same route; cropped to 0.95–2.15 x 0.02–1.30 in. **Live vector** |
| c | cam1 + cam4 labelled frame, tongue out | `Overview_2026-11-06.pptx` slide 29 → mp4 → frame | frame chosen by scanning 20 extracted frames for tongue-marker pixels; far_center trial, PS92 20260807 trial 409; 3-D plots cropped off. Raster |
| d | Fluoroscopy: labelled swallow frames | `~/Documents/.../AAN CDA/Fluoroscopy_AAN_CDA.ai`, panel A | same route; cropped to 0.05–5.55 x 0.10–1.20 in. **Live vector** |
| e | Inter-swallow interval | `/Volumes/Neurobio/MICROSCOPE/Priya/Fluoroscopy/mouse_vfss_backup/2026-11-05/outputs/metrics_all/per_swallow_distributions_panel_inter_swallow_interval.svg` | placed whole. Live vector |
| f | Licks per interval | same folder, `..._panel_licks_per_interval.svg` | placed whole. Live vector |
| g | Sitting / pharyngeal transit | `/Volumes/.../Priya/Fluoroscopy/grant figures/sitting_transit.svg` | placed whole. **Too small to read at this width — needs a compact re-render** |
| h | Hit rate by spout, by epoch | `Overview_2026-11-06.pptx` slide 4 | extracted at 2000x919. Raster |
| i | 3-D tongue and jaw paths across epochs | `Overview_2026-11-06.pptx` slide 61 | extracted at 1650x559. Raster. **Slide shows 6-across; this is 3-across** |
| j | First-lick aim by spout | `Overview_2026-11-06.pptx` slide 6 | extracted 1183x632, cropped to 910x632 to drop the WEIGHTED panel. **You asked for 6-across like the hit-rate panel — needs regenerating** |

**Not yet sourced for Figure 1**
- Far-contra trials by outcome, stacked bar (contact / incomplete / jaw-only / no response).
- Three of the six fluoroscopy metrics on your slide: swallows per minute, licks per minute,
  barium amount. Likely in `/Volumes/.../Priya/Fluoroscopy/grant figures/all_graphs.svg`, which
  was not staged before the share went offline; `bolus_density.svg` and `swallow_bolus.svg` are
  already staged locally and may cover "barium amount".

---

## Figure 2 — FaceRhythm  (slide 2; not yet built)

| Content | Source | Status |
|---|---|---|
| FR master face / mesh | `Overview_2026-11-06.pptx` slide 72 → `f3_FR_master_face.png` | extracted, raster |
| FR factors, spatial + HOW + spectrum | slide 74 → `f3_FR_12_factors.png` (4460x2356) | extracted. Slide shows **4 factors**, not 12 — needs subsetting |
| Far-contra licks: less contralesional face movement, 3 epoch maps | slide 85 → `f3_FR_loading_change_maps.png` | extracted; check this is the right triplet |
| lick_right / lick_left loading post-stroke, 2x3 maps | slide 85 family | needs the 2x3 version identified |
| Side specificity delta, lick_right vs lick_left | slide 77 → `f3_FR_lickright_lickleft.png` | extracted, raster |

Vector originals for the FR panels are expected under
`/Volumes/.../Priya/Widefield/.../grant_figures/` — *server offline*, not yet confirmed.

---

## Figure 3 — decoder and encoder  (slide 3; partially built as `Figure2_representation_DRAFT.svg`)

| Content | Source | Status |
|---|---|---|
| Frozen / within-session refit / refit − frozen gap, by epoch, pooled | `.../grant_figures/svg/grant_3b_frozen_vs_within.svg` + deck slide 19/20 | **vector in hand** for part; the pooled gap scatter is currently the deck PNG |
| Decoder offset, cued and ITI, FIRST lick | deck slide 95/96 family | the **cued FIRST lick** panel specifically is not yet identified |
| Cue-aligned activity maps, near ipsi + far contra rows | deck slide 25 → `f2_lick_maps_nearipsi_farcontra.png`; vector at `.../grant_figures/epoch/svg/epoch_14_beta_maps_MEANref_lick_lick.svg` | vector version is the full 5 epochs x 6 positions grid, needs subsetting |
| Cued lick hemispheric balance | deck | **you want far contra only** — needs subsetting |
| Lick response traces, tjM1 R/L and SSp-m R/L | `f3_cued_ROI_row.png`, slide 96 | raster; **subset to tjM1 and SSp-m R/L** |
| Lick response AUC | slide 96 → `slide96_image92.png` | raster; **you want this compressed to one graph** — needs regenerating |
| Encoder residual / variance maps | slide 95 → `slide95_image90.png` | raster; needs the **pre-stroke level** panel adding |
| Decoder accuracy by position, cue and pre-cue | `.../grant_figures/epoch/svg/epoch_acc_by_position_cue_working.svg`, `..._precue_working.svg` | **vector in hand** |

---

## Panels that cannot be re-pulled and must be re-rendered from the pipeline

These are the ones where the slide asks for a layout that does not exist in any current output:

1. First-lick aim, **6-across** to match the hit-rate panel (currently 2x3).
2. Lick response AUC **compressed into a single graph** (currently four).
3. Cued lick hemispheric balance, **far contra only**.
4. 3-D tongue/jaw paths, **6-across** (currently 3).
5. A compact fluoroscopy transit panel that stays legible at ~2.4 in wide.
6. Encoder residual maps including the **pre-stroke level**, not only the differences.

`widefield_pipeline/scripts/grant_figures_k08_2026.py` exists for exactly this and writes to
`K08_2026_grant_panels/`. Add a `PANELS` entry per item above rather than cropping a deck slide.

---

## Mechanics, so a re-pull is cheap

- `.ai` files from Illustrator are PDF-compatible: `pdftocairo -svg file.ai out.svg` keeps text and
  paths live. Crop at placement time with `place_crop(path, prefix, x, y, width, (x0,y0,x1,y1))`
  rather than editing the source.
- `build_fig.py` namespaces every id on import (`a_`, `b_`, ...) so clip paths and glyphs from
  different sources cannot collide.
- `make_fig1.py` holds the Figure 1 layout as a list of rows; change a path there and re-run.
- `svg2pdf.py file.svg` renders a PDF plus a full-page PNG preview.

---

# Re-plotted grant panels (9 Nov, server back online)

The deck figures carry supertitles, method blurbs, full-width legends and every position and ROI.
Rather than crop them, these are **re-plotted from the same session CSVs the deck figures were drawn
from**, so no value is re-derived — only the styling and the subset change.

Source data folder (all of it):
`/Volumes/Neurobio/MICROSCOPE/Priya/Widefield/labcams/grant_figures/lmm_hidden_20261103/`
Each figure there ships `<name>_sessions.csv` (one row per session: position, panel, epoch, animal,
label, value) and `<name>_lmm.csv` (beta, SE, df, CI, p, n, star). See that folder's README.

| Output | Built from | Subset applied |
|---|---|---|
| `decoder_offset_cued_first_lick.svg` | `lick_cued_first_lick_sessions.csv` | positions `all spouts\|lfl:first` and `far contra\|lfl:first` — the **cued, FIRST-lick** like-for-like offset you asked for |
| `hemispheric_balance_cued_farcontra.svg` | `lick_lateral_ratio_auc_docked_sessions.csv` | panels ending `, cued`; positions ending `:far_contra`; ROIs **tjM1 and SSp-m only** |
| `lick_response_auc_compressed.svg` | `cued_ramp_summary_row2_lmm_sessions.csv` (**`value` column, fold of own pre**) | four panels **compressed to one axes**, ROI on x, epoch as colour: tjM1 R/L and SSp-m R/L, **far-contra spout only** |

Grant styling applied to all three, in `grant_panels.py`:
Arial throughout; 8 pt axis labels and titles, 7.5 pt ticks, 6.8 pt legend; top and right spines
dropped; epoch colours fixed to the deck's (`pre #808080`, `acute #e0218a`, `subacute #d4a017`,
`chronic #1b7f79`); open markers per animal, filled dot with SEM for the epoch mean; axis labels
shortened to pre / acu / sub / chr so adjacent panels do not collide; `svg.fonttype: none` so every
label stays editable in Illustrator. Legends are frameless, compact, and auto-placed into the
emptiest quadrant of each panel by `emptiest_corner()`.

**Known imperfection:** the auto-placer counts scatter points but not error-bar containers, so on
one or two panels the key sits a little close to the acute cloud. Nudging the legend group in
Illustrator is a two-second fix, or pass `loc=` explicitly in `render_grant_panels.py`.

## Vector sources located on the server this session

| Panel wanted | File found |
|---|---|
| Frozen / refit / gap, pooled, post-cue | `epoch/svg/epoch_5ro_frozen_refit_overall_cue_working.svg` (bar form) and `lmm_hidden_20261103/key_figure_BC.svg` (change-from-pre form). **Neither matches the 3-panel scatter on your slide** — that shape needs re-plotting from `frozen_refit_pooled_lmm_lmm.csv` |
| Cue-aligned activity maps | `epoch/svg/epoch_14_beta_maps_MEANref_cue_working.svg` (full 5 epochs x 6 positions; subset to near-ipsi and far-contra rows) |
| Decoder accuracy by position, cue and pre-cue | `epoch/svg/epoch_acc_by_position_cue_working.svg`, `..._precue_working.svg` |
| Six fluoroscopy metrics | `Fluoroscopy/grant figures/all_graphs.svg` — now staged, covers the full set on your slide |
| ITI ramp row | `lmm_hidden_20261103/iti_ramp_row1_lmm.svg` + its sessions CSV |

## Still to do

1. Re-plot the 3-panel frozen / refit / gap scatter from `frozen_refit_pooled_lmm_lmm.csv`.
2. Subset the cue-aligned beta maps to near-ipsi and far-contra rows.
3. Pull the six fluoroscopy metric panels out of `all_graphs.svg` and place them compactly.
4. First-lick aim 6-across, and 3-D paths 6-across: not in `lmm_hidden`, need the behaviour pipeline.
5. FaceRhythm vector sources: not yet located; `Priya/FaceRhythm/` has only two run folders.

---

# Statistics and legends (9 Nov, second pass)

**Asterisks now come from each figure's own statistics file, never from me.**

| Panel | Stats file | Column |
|---|---|---|
| `frozen_refit_gap_pooled` | `grant_figures/epoch/data/epoch_5rmo_frozen_refit_overall_cue_working.csv` | `mark` |
| `decoder_offset_cued_first_lick` | `lmm_hidden_20261103/lick_cued_first_lick_lmm.csv` | `star` |
| `hemispheric_balance_cued_farcontra` | `lmm_hidden_20261103/lick_lateral_ratio_auc_docked_lmm.csv` | `star` |
| `lick_response_auc_compressed` | `lmm_hidden_20261103/cued_ramp_summary_row2_lmm_lmm.csv` | `star` |

Marks are drawn in the epoch's own colour above that epoch's cloud. `*` and `**` carry whatever
meaning the source file assigns (for the pooled panel, change vs pre with `**` Bonferroni over 9;
for the LMM panels, `*` p < .05 uncorrected and `**` Holm across the figure's cells x epochs — see
the `lmm_hidden_20261103/README.md`). If the deck's convention changes on re-render, re-pull the
stats file and the marks follow automatically.

**Legends cannot cover data, overhang the axes, or sit on the stars.** Three mechanisms, applied
in this order by every renderer:

1. `headroom(ax, frac)` expands the y-limit upward, opening a clean band at the top of the panel.
2. `epoch_legend(ax, style=...)` draws the key, measures it, and if it runs past the axes box it
   falls back through narrower shapes (`row` four-across to `pair` to `block` 2 x 2 to `stack` one
   column) and then through smaller type until it fits. The style argument is a preference, not a
   promise, so a narrow panel silently gets the shape that works.
3. `star_band(ax, lg)` returns the y position just under the drawn legend, and the stars are placed
   there. Stars are therefore always below the key and above the data, whatever shape the key took.

Asterisks are 10-11 pt, deliberately larger than the axis type, so they read at panel size.

# Found via the deck's own provenance file

`presentations/_source_PI_overview_2026-11-02/figure_provenance.csv` (189 rows: slide, shape name,
original path, how it was transformed) is the master key for every deck figure.

| Wanted | Found |
|---|---|
| 3-panel frozen / refit / gap scatter | built by `fig_frozen_refit_pooled.py` from `grant_figures/epoch/data/epoch_5rmo_frozen_refit_overall_cue_working_sessions.csv`; slide script `_source_PI_overview_2026-11-02/add_pooled_gap.py`. **Now re-plotted** |
| FaceRhythm rank sweep | `DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/report/rank_sweep.png` |
| FaceRhythm factor gallery | same folder, `factor_gallery_abs.png` (deck re-laid it out as 2 columns) |
| FR per-lick stats | `presentations/_source_PI_overview_2026-11-02/build_inputs/fr_page/fig/per_lick_stats.png` (deck used 4 of 14 panels) |
| FR matched lick pair | `lmm_hidden_20261103/fr_PS94_examples/fr_matched_lick_pair_PS94.png` |
| FR cohort drift floor | `build_inputs/fr_page/fig/cohort_drift_floor.png` (deck used rows 1-2) |
| Quiet-period FR | `labcams/cohort1_quiet_spikes/v2_quiet_lp/fr_regressed_own/quiet_events_face_regressed.png` |

The FaceRhythm page itself is built under `Widefield/fr_page_src/fr_page/` (`rebuild.py`, `fig/`,
`lmm/`), so FR panels have the same sessions/LMM CSV structure available for re-plotting.

## Correction, 9 Nov: lick-response AUC

The first version of this panel plotted the raw `auc` column pooled over all six spout positions
while drawing stars from the far-contra LMM cell. That put significance marks on a quantity the
test never saw, and the marks looked wrong because they were.

The far-contra-only version that followed was also wrong for this panel: far contra is the weakest
of the six spout cells, so restricting to it lost almost every mark.

**Final version** uses the `_w0-1` variant of the same extractor, which is the quantity the deck's
main slide actually plots: **AUC 0-1.0 s as a fold of each animal's own pre, spouts POOLED**, one
row per ROI.

- values: `lmm_hidden_20261103/cued_ramp_summary_row2_lmm_w0-1_sessions.csv` (`value` column; `raw`
  holds the un-normalised AUC)
- marks: `lmm_hidden_20261103/cued_ramp_summary_row2_lmm_w0-1_verify.csv`, column **`boot_star`** —
  the published paired nested-bootstrap test (CI excludes 0), which is what the deck slide shows.
  The same file's `star` column is the stricter LMM result, and `published` reproduces the deck's
  per-cell change exactly, so the two tests can be compared cell for cell.

Result: 11 of 12 ROI x epoch cells starred, the exception being tjM1 left acute. That matches the
deck, and it is why pooling matters — the per-spout LMM is only significant in 2-5 of 6 spouts.

**Lesson worth keeping:** these extractors ship several variants per figure (`_peak`, `_w0-1`, the
per-spout default). Check which variant the deck slide used before pulling stats, because they test
different cells.

---

# Assembled figures (to the 9 Nov slide layouts)

| File | Slide | Panels |
|---|---|---|
| `Figure1_behavior_DRAFT2.svg` | 1 | a task schematic + trial structure (.ai, vector) · b histology (.ai, vector) · c labelled cam frames · d fluoroscopy swallow frames (.ai, vector) · e-g three fluoroscopy metrics · h hit rate by spout · i 3-D paths · j first-lick aim |
| `Figure3_decoder_encoder_DRAFT2.svg` | 3 | a frozen / refit / gap pooled · b cued first-lick decoder offset · c cue-aligned activity maps, near-ipsi + far-contra · d hemispheric balance, far contra · e lick-response time courses, tjM1 + SSp-m both sides · f lick-response AUC pooled · g encoder residual maps |

In Figure 3, panels **a, b, d and f are re-plotted grant panels** carrying their own statistics from
the source stats files. Panels c, e and g are still deck rasters: c is already the near-ipsi /
far-contra subset, e is cropped from the 7-ROI row to tjM1 and SSp-m both sides, g still needs the
pre-stroke residual level alongside the difference maps.

Figure 3 currently stands 723 pt (about 10 in) tall at 7.4 in wide. It has to come down to roughly
4-5 in, so three or four panels will not survive. My read: a, c and f carry the argument; b and d
are the detail; e is method.

Figure 2 (slide 2, FaceRhythm) is not yet assembled. Sources are located (see above) but the panels
need pulling from `DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/report/` and
`build_inputs/fr_page/fig/`, and the FR page under `Widefield/fr_page_src/fr_page/` has the same
sessions/LMM CSV structure if those panels should be re-plotted rather than cropped.

---

# Panel c replacement: state-space change maps (10 Nov)

Replacing the cue-aligned activity maps in Figure 3 with the state-space "full change" maps.

- Source: `/Volumes/Neurobio/MICROSCOPE/Priya/Widefield/state_space_recovery/page_figs/ss_cmap_full_rest_lick_svd.png`
  (page figure 9.1, "full change, lick-aligned, RESTnl"). Seven columns: mean over spouts, then the
  six spouts; three rows: acute, subacute, chronic minus pre.
- Treatment: alpha flattened onto white (the PNG is transparent), then cropped to the **mean over
  spouts** and **far contra** columns plus the shared colour bar, giving a 2 x 3 grid.
  Output: `panels_src/ss_meanspouts_farcontra.png`.
- Why this instead of `epoch_15r_position_RESTWref_lick_lick`: significance contours are already on
  every panel (nested bootstrap animals -> sessions, max-stat family-wise per panel), unanalysed
  tissue is masked grey with the analysed field of view outlined, and values are normalised to each
  animal's own pre response RMS.
- **Role in the figure:** this is the *setup* panel, not evidence of reorganisation. It shows that
  response magnitude changes widely, including contralesionally, and that the change is graded by
  spout. The caption should say that magnitude alone cannot distinguish a representation that
  returned from one that was replaced, which is what the frozen/refit panel then answers.

**Open gap:** there is no pre-stroke LEVEL map in `state_space_recovery/page_figs` - every `ss_cmap`
variant is change-vs-pre. A pre-level reference map in a sequential colour map (as in the encoder
figure's row 1) would have to come either from `epoch_15r_position_RESTref_lick_lick` (different
normalisation: time-local REST baseline rather than the session RESTnl rest level, so not strictly
comparable) or from one extra panel emitted by the state-space code, which already computes the pre
reference in order to form the difference.

---

# Allen area key (prior K08 v7, Figure 5H)

Source: `Widefield_CAMS_fig.ai` (the BWF/CAMS widefield figure; v7 Figure 5H is panel H of it),
converted with `pdftocairo -svg` to `panels_src/widefield.svg` so it stays live vector.

Crop box for `place_crop`, in the file's own points: **x 187.2 to 336.0, y 307.2 to 403.2**.
Raster reference of the same region: `panels_src/areakey_5H.png`.

It labels MOs-m, MOp-m, SSp-m, SSp-n, SSp-ul, SSp-ll and SSp-tr on a dorsal cortex outline, which
is exactly the key a reviewer needs to read the encoder and state-space maps. Place it once, beside
whichever map panel comes first, and the caption can simply say the areas are labelled there.

Note the naming caveat already recorded above: these are **Allen CCF areas**. ALM and tjM1, used in
the ROI panels, are functional sectors of MOs and MOp and do not appear in this key.

# State-space maps, second pull (10 Nov)

`ss_cmap_full_rest_lick_svd.png` regenerated with the pre-stroke response row (viridis, own scale)
and the fibre-glue mask switched from grey to white to match the encoder figures. Cropped to
**mean over spouts** and **far contra**, four rows (pre response, then acute / subacute / chronic
minus pre), both colour bars kept: `panels_src/ss_pre_meanspouts_farcontra.png`.

Column bounds in that PNG, measured from the viridis row, for future re-pulls:
col 1 x 35-212, col 7 x 1315-1495, colour bars x 1500 onward, content starts y 95.
**A further re-pull is expected: the maps are being redone to show the full CCF.**

# Aim panel statistics: now stored, not transcribed (9 Nov)

`first_lick_aim_by_spout_single` previously carried marks I had transcribed by eye from the
published PNG behind a `USE_BOOT` flag. It now reads the stored paired nested bootstrap directly:

| file | what it gives |
|---|---|
| `.../imaging_encoding/iti_licks/aim_by_spout_tests.csv` | `mean`, `lo`, `hi` (95% CI), `lo_bonf`, `hi_bonf`, `sig`, `sig_bonf`, `n_animals`, `n_sessions`, per-animal columns |
| `.../imaging_encoding/iti_licks/aim_by_spout_sessions.csv` | per-session `chg` (change vs that animal's own pre), 749 rows |

Full path on the server: `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\iti_licks`.
Local copies in `aimstats/`.

**Stars are `sig_bonf`** (Bonferroni over the 24 spout x epoch comparisons): 15 of 24 cells.
The panel now also shows all four epochs with faded session dots, so the pre group shows the
null band the deficits are measured against.

**Two corrections to the transcribed marks:**
- `close_contra` chronic is **not** significant (-2.20, CI -5.14 to +0.46). It was starred before.
- `far_contra` chronic **is** significant (-6.99, CI -13.13 to -2.97), confirming the recollection.
- `close_mid` chronic and `far_ipsi` chronic are significant raw but not Bonferroni, so they are unstarred.

**Still open: the hit-rate panel.** `hit_rate_by_spout_single` still reads
`data/behaviour_hits_by_spout_lmm_lmm.csv` (LMM, no session rows) and still carries transcribed
marks via `BOOT_HITS`. A bootstrap alternative exists at
`cohort/tables/prepost/stats/change_vs_pre_stats.csv` with per-session rows in
`cohort/tables/prepost/position_metrics.csv`, but it is a different quantity: the closest metrics
are `contact_frac` (far contra acute -0.680) and `resp_rate` (far contra acute -0.342), neither of
which matches the LMM hits values currently plotted (far contra acute -0.90). Needs a decision on
which definition of "hit rate" the figure should show before it can be unified with the aim panel.

# Hit-rate panel: corrected source (9 Nov, supersedes the note above)

The `contact_frac` route in the note above was the WRONG source. The canonical hit-rate figure is
the purpose-built grant figure:

`rebuild_20261103/labcams/grant_figures/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/epoch/`

| file | what it gives |
|---|---|
| `data/epoch_1bdelta_behaviour_by_position.csv` | `point`, `lo95`, `hi95`, `lo_corrected`, `hi_corrected` (Bonferroni over 18 comparisons) |
| `data/epoch_1bdelta_behaviour_by_position_sessions.csv` | 456 per-session values |
| `data/epoch_1bdelta_behaviour_by_position_meta.csv` | the settings |
| `*_daq_quittail*` | the same figures on the DAQ engagement gate: the robustness version |

Bootstrap: animals -> sessions -> position blocks within session, 2000 draws. A hit is a spout
contact. Working period = first cue to last camera lick, stopped runs removed. N = 4 animals,
n = 102 sessions (pre 26, acute 16, subacute 17, chronic 43; PS93_0813 excluded as a working-period
session). Local copies in `hitstats/`.

**Stars = the Bonferroni-corrected interval excludes zero**, matching `sig_bonf` on the aim panel.
5 of 18 cells: acute close-mid, far-ipsi, far-mid, far-contra; subacute far-contra. Every chronic
cell is at zero, which is the point: hit rate recovers completely while the aim offset does not.

**Two things this is NOT:**
- `behaviour_hits_by_spout.png` (deck / page, in `rebuild_20261103/rebuild_figs/<same tag>/`) is
  per-session dots with mean +- SEM over animals and carries no bootstrap of its own. It was the
  source of the old LMM panel and of my transcribed marks.
- `cohort/tables/prepost/stats/change_vs_pre_stats.csv` metric `contact_frac` is a different
  quantity on a different engagement gate. Not used.

Pre has no row in the delta table, so the panel draws no pre group; the dashed line at zero is pre.
(The aim panel does have pre rows and does draw the group. The two panels differ in that one
respect, by source.)

# Figure 4 panel a: frA only (9 Nov)

Dropped the as-recorded columns. Panel a is now the FaceRhythm-regressed (frA, all FR covariates)
maps alone, mean over spouts and far contra, built by `ss/ss_frA_only.png` which crops
`ss/frA_w.png` (row labels, column 1, column 7, both colour bars; title band kept, caption band
dropped). Width 0.56 -> 0.34. The as-recorded / regressed comparison now lives in the text.

Numbers for that text, chronic, as-recorded vs frA, from `chronic_maps_lick_svd_rest_{fr0,frA}`:
mean over spouts 95 per cent of the magnitude retained, map r 0.996, significance-mask Dice 0.89;
far contra 123 per cent, r 0.946, Dice 0.75. The spout-specific component is the one that does not
survive: far contra 59 per cent, r 0.719, Dice 0.00. frB (directionless FR) behaves the same:
100 per cent / r 0.999 for the global change, 48 per cent for the spout-specific component.
