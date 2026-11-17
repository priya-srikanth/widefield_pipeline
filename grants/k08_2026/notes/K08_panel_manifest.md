# K08 panel manifest — your selections, 6 Nov 2026

Everything you have picked, with where it comes from. Deck = `Overview_2026-11-06.pptx`.
Extracted panels are on the server in `Priya/Claude outputs/k08_deck_panels/`.

---

## Figure 1 — Model, deficit, behaviour

| # | Panel | Source | Status |
|---|---|---|---|
| 1 | VLS stroke histology, representative | existing grant figure | reuse, settled |
| 2 | Fluoroscopy: labeled swallow frames | `Fluoroscopy/grant figures/swallow_frames.svg` | located |
| 3 | Fluoroscopy metrics: swallow rate, lick rate, ISI, licks/swallow | `per_swallow_distributions_panel_*.svg` | located |
| 4 | Fluoroscopy: transit time + **bolus amount (size x density)** | `sitting_transit.svg`, `bolus_density.svg`, `swallow_bolus.svg` | located |
| 5 | Six-spout behaviour schematic | existing grant figure | reuse, settled |
| 6 | Hit rate by spout across epochs | **deck slide 4** → `f1_hitrate_by_spout.png` (2000x919) | extracted |
| 7 | cam1 + cam4 labeled frame, tongue out, **no 3-D plots** | **deck slide 29**, mp4 → frame extracted, cropped to the camera block | extracted |
| 8 | First-lick aim by spout, **without the WEIGHTED aim panel** | **deck slide 6** → `f1_firstlick_aim_by_spout.png` (1183x632), crop out bottom-right | extracted, needs crop |
| 9 | 3-D paths across epochs | **deck slide 61** → `f1_3d_paths.png` (1650x559) | extracted |
| 10 | Far-contra trial outcomes (contact / incomplete / jaw-only / no response) | **deck slide 37** → `f1_farcontra_trial_outcomes.png` (1700x1440) | extracted |

Order per your note: labeled frames (7) come **before** the aim panel (8), which comes **before**
the trajectories (9).

## Figure 2 — Cortical representation

| # | Panel | Source | Status |
|---|---|---|---|
| 1 | Frozen / within-session refit / refit−frozen gap, by epoch, pooled | **deck slide 20** → `f2_frozen_refit_gap_by_epoch.png` (1954x1051) | extracted |
| 2 | Frozen decoder by position, cue window | **deck slide 11** → `f2_frozen_cue_by_position.png` | extracted |
| 3 | Frozen decoder by position, pre-cue (ENL) | **deck slide 13** → `f2_frozen_precue_by_position.png` | extracted |
| 4 | Lick-aligned maps, Near Ipsi + Far Contra: pre / acute / subacute / chronic + the three minus-pre maps | **deck slide 25** → `f2_lick_maps_nearipsi_farcontra.png` (3224x935) | extracted |

Panel 4 replaces the 30-brain grid entirely — it is already the two-row subset with difference
maps and significance contours.

## Figure 3 — FaceRhythm and the encoder

| # | Panel | Source | Status |
|---|---|---|---|
| 1 | FaceRhythm master face, ROI grid | **deck slide 72** → `f3_FR_master_face.png` | extracted |
| 2 | The 12 frozen FR factors (spatial + spectral + temporal) | **deck slide 74** → `f3_FR_12_factors.png` (4460x2356) | extracted |
| 3 | lick_right drops / lick_left substitutes, by spout side | **deck slide 77** → `f3_FR_lickright_lickleft.png` | extracted |
| 4 | FR loading-change maps (lick_right / lick_left x epoch) | **deck slide 85** → `f3_FR_loading_change_maps.png` | extracted |
| 5 | **Encoder residual maps: pre-stroke level + chronic−pre, LICK window** | **deck slide 95** → `slide95_image90.png`; pre-stroke level to come from `encoding_goal_page_src/fig7*_residual*` | partly located |
| 6 | **Encoder ROI quantification, CUED version, tjM1 and SSp-m R and L only** | cued twin of the ITI ROI row; `lr_cued_summary_row2.png` + slide 96 media | located, needs subsetting to 4 ROIs |
| 7 | **Like-for-like readout offset, cued FIRST lick** | the 2x3 ITI/cued x all/first/later grid — take the **cued, FIRST lick** panel only | needs source file identified |
| 8 | Optional: decoded vs landing place, cued | the ITI/cued curve pair, cued panel | optional |
| 9 | Readout 2x2: lateral decoder r (ITI, cued) + like-for-like offset (ITI, cued), by epoch | `crops/lmm_epoch_readout_2x2.png` | located |
| 10 | Lick-order curves, **cued**: landing (behaviour), decoded position (neural), like-for-like offset | lick-order family; `o_lick_order_cued_row.png` / `o_lick_order_curves.png` in crops | likely located |
| 11 | Per-ROI first cued lick spout coding by epoch (tjM1, SSp-n, orofacial), **far contra spout only** | source to identify; needs subsetting to the far-contra column | needs source + crop |

Your note on the encoder: the residual already uses FaceRhythm, so panels 5 plus 6 may be enough
without a separate Musall-style variance panel. Stats contours are not required; a clean picture
is fine.

---

## The problem I have to flag

This is now **25 panels across 3 figures**. The Research Strategy has one page plus about four inches
of genuinely new space, and the existing figures already occupy 23 inches of stacked height that
these would replace.

Figure 3 alone now carries 11 panels, several of which are dense multi-brain or multi-factor
grids. The 12-factor panel at 4460x2356 cannot be shrunk to a grant panel and stay legible.

Before I build, it would help to know, for each figure, which **two or three** panels carry the
argument and which are supporting. My read:

- **Figure 1**: hit rate, aim, trial outcomes carry it. Fluoroscopy is the clinical hook. The
  labeled frame and 3-D paths are method, and could be small insets.
- **Figure 2**: frozen/refit/gap by epoch, plus the lick-aligned maps. The two by-position panels
  are the detail; pre-cue earns its place only as the movement-confound control.
- **Figure 3**: lick_right/lick_left (the STANDS 4/4 result) plus the encoder residual. The
  12-factor gallery is method and probably belongs as a small inset or not at all.
