# K08 Research Strategy figure plan, v3 (6 Nov 2026)

Supersedes v1 and v2. Adds the evidence-status review from the repo `FINDINGS` docs and the
Overview deck, and records the decisions made so far.

---

## Decisions settled

| Question | Decision |
|---|---|
| Histology | Keep the generic VLS stroke histology already in the grant, one representative image. No regeneration. |
| 6-spout behaviour schematic | One already exists in the grant. Use it in place of the two-spout schematic. |
| Vector vs raster | `epoch/svg/` has vector for the core decoder panels (352 SVGs). PNG is an accepted fallback elsewhere: highest resolution, placed at final size, never upscaled. |
| Fluoroscopy bolus | Include bolus amount (size x density). |
| Two-spout paradigm | Retired from text and figures. |
| FR encoder map stats | Significance contours not required. A clean picture is fine for the grant. |
| Fig 1 panel order | Labeled camera frames come **before** the ipsilesional first-lick aim, which comes **before** the 3-D trajectory. |
| Fig 2 gap panel | Use the refit − frozen scatter **pooled across positions**, not the per-position version. |

---

## The filter that should drive panel choice

`docs/FINDINGS.md`, `docs/OROFACIAL_FINDINGS.md` and `docs/FACE_RHYTHM_FINDINGS.md` label every
result. Only **STANDS** results belong in a figure that carries an argument. PROVISIONAL results
can appear as context if the text says so. **WITHDRAWN and NOT ESTABLISHED results must not appear
at all**, and one of them is already in the proposal's orbit.

### STANDS, four of four animals — safe to build on

- Acute global slowing plus a far-contra-graded deficit: contact −0.67, first lick +488 ms,
  first contact +898 ms (cam4, 4/4; PS95 acute is a single session)
- Acute licks aim ipsilesionally, +30 deg far contra, +7 to +13 deg elsewhere, **confirmed in 3-D**
- Jaw stops steering toward the target acutely, FAR_DIR −11.4 px, confirmed in 3-D
- Jaw moves without a lick on more acute trials; blind review confirms these are reactions, not attempts
- Most acute response loss was quitting
- Chronic persistence: more licks per trial, +1.5 to +4.2
- Frozen decoder loses far contra, .90 → .32 acutely
- FaceRhythm tracking quality, bin timing, pre-stroke noise floor
- **FaceRhythm 5d: contralesional trials recruit the ipsilesional lick pattern, 4/4** — the single
  strongest FaceRhythm result and the one a FaceRhythm panel should rest on

### PROVISIONAL — context only, caption must say so

- Far-contra protrusion loss −26 px (largely a far-spout effect; far ipsi also drops in all four)
- Tongue cloud shift, nose posture shift
- Quiet-period 2-3 Hz event result
- ITI and cued-lick decoder offset
- FaceRhythm section 7 / 7b, spout position from face movement in the ENL window

### Do not use

- **Chronic jaw-steering deficit: CORRECTED to NOT ESTABLISHED per animal.** The pooled cell is
  significant but only PS92 and PS93 keep a deficit; PS95 and PS94 do not. Two of four.
- **The lesion-side bias in acute orofacial 1-4 Hz power: WITHDRAWN (T22).** There is no laterality
  support for the reduced-thalamocortical-drive hypothesis. If any version of that claim is still
  in the Research Strategy text it needs to come out.
- FaceRhythm 6c factor "morphing": not consistent across animals, secondary.
- Anything from the PILOT two-animal FaceRhythm sections, superseded by the four-animal master.

**Caveat to carry into the drift control:** June to August drift is as large as the chronic
effect (deck slide 15). Any claim that chronic change exceeds drift needs the matched comparison,
not a raw pre-to-chronic difference.

---

## Figure 1 — Model and behavioural deficit (~3.6 in)

Reordered per your note: fluoroscopy now opens the behavioural case, as evidence of swallowing
impairment, **before** the six-spout task is introduced. That reads as deficit first, then the
assay built to measure it.

| Panel | Source | Status |
|---|---|---|
| a. VLS stroke histology, representative | existing grant figure | settled, reuse |
| b. Fluoroscopy: labeled example frames, oral / pharyngeal / esophageal | `Fluoroscopy/grant figures/swallow_frames.svg`, or adapt `Fluoroscopy_MGN_progress_report.ai` | settled |
| c. Fluoroscopy metrics: swallowing rate, licking rate, inter-swallow interval, licks per swallow | `per_swallow_distributions_panel_{inter_swallow_interval, licks_per_interval, swallow_duration}.svg` | |
| d. Fluoroscopy: transit time and bolus amount (size x density) | `sitting_transit.svg` or `..._pharyngeal_transit_time.svg`; `bolus_density.svg`, `swallow_bolus.svg`, `..._pharyngeal_bolus_area.svg` | |
| e. Six-spout behaviour schematic | existing grant figure | settled, reuse |
| f. Hit rate by spout position across epochs | `epoch/svg/epoch_1c_behaviour_timecourse.svg` or `grant_1_behaviour_by_position.svg` | STANDS |
| g. Single frame, tongue out, both camera views, no 3-D plots underneath | still from the overview-deck video (deck 21/22) | settled |
| h. Ipsilesional first-lick aim by position across epochs | `Behavior_overlaid_20260929/angle_deg_delta_L_ipsilesional_overlaid_sigbar.svg` | STANDS, 3-D confirmed |
| i. 3-D trajectory across epochs | `session_poses/pose3d/figures/pooled/pose3d_paths_pooled_shared.png`; deck 38, 41 | STANDS |

Fluoroscopy method insets available if wanted: `density_method.svg`, `rois_and_masks.svg`.

This is a large figure. If it will not hold, the fluoroscopy metrics (c and d) compress to a
single multi-metric panel, and `all_graphs.svg` already exists as that composite.

## Figure 2 — Representation: frozen vs refit, and where it moves (~3.4 in)

| Panel | Source | Status |
|---|---|---|
| a. Frozen vs refit over recovery | `grant_3b_frozen_vs_within.svg`; deck 11 | STANDS |
| b. Refit minus frozen gap, **pooled across positions** | pooled scatter, panel still to be identified (not `epoch_5rdelta_*`, which is per position); deck 12 | STANDS |
| c. Frozen decoder by position, far contra .90 → .32 | `epoch/svg/epoch_acc_by_position_cue_working.svg`; deck 10 | STANDS |
| d. Lick-aligned activity maps, far contra: pre, acute−pre, subacute−pre, chronic−pre | `epoch/svg/epoch_14_beta_maps_MEANref_lick_lick.svg`, per-animal twin `epoch_14pa_*` | STANDS |
| e. Pre-cue (ENL) decoding by position across epochs — *recommended, see below* | `epoch/svg/epoch_acc_by_position_precue_working.svg`; deck 19, prov fig1 | STANDS |
| f. Optional: decoder offset, ITI and cued first lick | `encoding_goal_page_src/lick_cued_where_scatter_z.png`, `lick_iti_where_scatter_z.png` | PROVISIONAL |

**Correction to v1: both of these exist.** They are rendered PNGs in
`Widefield/encoding_goal_page_src/` (348 files, no SVG there, so PNG fallback applies):

- decoder-offset / "where" scatters: `lick_cued_where_scatter{,_z}.png`,
  `lick_iti_where_scatter{,_z}.png`, `iti_where_scatter_{iti,trial}.png`
- supporting: `iti_where_per_spout{,_z}.png`, `iti_where_delta.png`, `iti_where_mae.png`,
  `iti_where_drift{,_june}.png`, `lick_where_angle_reach.png`, `lick_where_decoder_bases.png`

Generators are `scripts/iti_where_scatter.py` and `scripts/lick_offset_where.py`; source outputs
sit under `LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks`.

**Caution:** `docs/status/reanalysis_20261105/REPLACEMENT_PASS_MAP_2026-11-05.csv` marks several
of these "not yet rendered" under the 11-05 rebuild root, so the copies in
`encoding_goal_page_src/` may predate the reanalysis. Check currency before using one.

The offset result is labelled **PROVISIONAL** in `FINDINGS.md`. Include it if you want it, but
the caption should not carry weight the evidence does not.

### Should the pre-cue and lick-aligned per-position panels also go in?

All three windows exist as vector: `epoch_acc_by_position_{precue_working, cue_working,
lick_lick}.svg`.

**Include pre-cue. Skip lick-aligned.**

The cue-window panel is the headline result. The pre-cue panel earns its space for a different
reason: it is the answer to the first objection a reviewer will raise, that the decoder is reading
movement rather than a representation. Position is decodable *before* any movement, and the frozen
pre-cue decoder loses far contra in 4/4 animals (deck 19). That single panel pre-empts the
confound, and `FINDINGS.md` also carries a movement-removed version of the analysis if the text
needs to go further.

The lick-aligned panel adds the least and is the most movement-contaminated of the three, so it is
the one to leave out.

Caveat to put in the caption: pre-cue accuracy is low in absolute terms (far contra ENL 0.51
pre-stroke) and the slow-band sensitivity check moves it by 0.03-0.09 (prov fig1b). State the
effect as a drop from an already-modest baseline rather than implying a strong pre-cue code.

## Figure 3 — FaceRhythm and the encoder (~2.4 in)

| Panel | Source | Status |
|---|---|---|
| a. The 12 frozen master factors, spatial and frequency loadings, incl. lick-left, lick-right, whisking | deck 53; `FaceRhythm/` master_pre_v1 rank 12 | descriptive |
| b. lick_right drops, lick_left substitutes on contralesional trials | deck 56; findings 5d | **STANDS 4/4** |
| c. Face asymmetry falls at contralesional spouts | deck 58; findings 6d | check status before use |
| d. **Musall-style encoder maps**: cortical variance explained by the FaceRhythm lick kernel, and change from pre | see below | |

### Pared-down, post-stroke-difference versions — these already exist

**`fr_musall_cortex_lick_modeA.png` is already epoch-resolved**: pre / acute / subacute / chronic
across four columns. Cropping to its **top row alone** gives a publication-ready panel showing the
Musall-style variance split by epoch. What it shows: FaceRhythm's total contribution *rises* after
stroke (~0.31 pre to ~0.38 chronic) while uninstructed movement falls (~0.26 to ~0.18) and task
grows (~0.18 to ~0.31). That is a clean statement that more of the cortical signal is accounted for
by face movement after the stroke.

**`variance_maps_lick.png` is the spatial difference map**: pre-stroke level, then acute−pre,
subacute−pre and chronic−pre, each with a log2-ratio map carrying Bonferroni-corrected significance
contours. Cropping to the difference and ratio columns gives exactly the "what changed" map.
**Caveat:** this is *total* widefield signal variance, data only, not the FR kernel, so it cannot be
captioned as a FaceRhythm result.

**`epoch_kernel_maps_lick.png`** is the lick-kernel map across 5 epochs x 6 spout positions, 30
brains. Too large as-is; subset to the far-contra column across epochs if you want the kernel itself.

### All Musall-style encoder panels in `Widefield/encoding_goal_page_src/` (PNG)

- `fr_musall_cortex_lick_modeA.png` and `fr_musall_cortex_cue_modeA.png` — cortical maps of
  FaceRhythm contribution, lick-aligned and cue-aligned. **This is the closest match to what you
  described.**
- `frozen_unique_sig_frA.png` vs `frozen_unique_sig_noFR.png` — unique-variance maps with and
  without the FaceRhythm regressors. The pair is the cleanest way to show what FR adds, and
  `frozen_unique_sig_noFR_per_animal.png` gives the per-animal version.
- `fr_lick_variance.png` — variance explained by the lick factor
- `kernel_maps.png`, `kernel_maps_target6.png`, `epoch_kernel_maps_lick.png` — kernel loading maps
- `lick_area_unique_maps_cued.png`, `lick_area_unique_maps_iti.png` — unique maps by lick area

Vector alternatives for the non-map encoder quantities are in `epoch/svg/`:
`epoch_11_encoder_gain_shape_*`, `epoch_11amp_encoder_amplitude_*` and its delta
`epoch_11ampdelta_*`, `epoch_11c_encoder_ceiling_*`, `epoch_11cfrac_encoder_captured_by_position_*`.

Deck 64-76 is the encoder narrative; the load-bearing claim is that the acute drop in encoder fit
is largely movement (deck 65, 66) while the residual shows cortex responding more per lick
(deck 70, 71).

Do **not** build this figure on the loading-shape change (deck 59, 61, 62), which is the 6c
morphing result and is not consistent across animals.

The argument this figure must make is that the techniques fold together: facial kinematics carry
the same compensation signature as cortex, contralesional trials executed with the ipsilesional
pattern, and the encoder maps show where on cortex that movement signal lands. That is what 5d
plus the Musall maps say together. If space runs short, fold panel b into Figure 1 as a kinematic
readout and keep only the Musall pair here.

## Everything is now located

### The master provenance file
`docs/status/reanalysis_20261105/REPLACEMENT_PASS_MAP_2026-11-05.csv` maps **every deck slide** to
its source page, the original file path on the server, and the already-trimmed deck crop. This is
the file to work from, not filename guesses.

### Key locations

| What | Where |
|---|---|
| Overview deck | `Widefield/presentations/Overview_2026-11-06.pptx` |
| **Deck-ready crops, 103 files, already trimmed** | `Widefield/presentations/_source_PI_overview_2026-11-02/build_inputs/crops/` |
| FR factor gallery (rank-12 master) | `DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/report/` — `factor_gallery.png`, `factor_gallery_abs.png`, `factor_epochs.png`, `factor_epochs_abs.png`, plus `rank_sweep` and `stability_per_factor.csv` |
| FR shape loading by epoch | `cohort_epoch_shape_loading_lick_{left,right}.png`, `epoch_shape_loading_*` |
| FR matched factor gallery | `cohort_matched_factor_gallery_lick_{left,right,ongoing_7Hz,fast_close,slow_late,far_nose}.png` |
| 3-D trajectories | `DeepLabCut/Widefield/session_poses/pose3d/figures/pooled/` — `pose3d_paths_pooled{,_shared}.png`, `pose3d_phase_pooled.png`, `pose3d_change_pooled{,_contra}.png`, `angle_change_pooled{,_contra}.png`, each with a CSV twin |
| Kinematics page source | `Widefield/kin_page_src/kin_page/` |
| Encoder / decoder-offset panels | `Widefield/encoding_goal_page_src/` (348 files) |

**Prefer the deck crops** where one exists: they are already selected by you, trimmed, and have
suptitles removed, which is most of the work of making a grant panel.

### Findings pages as artifacts
The provenance map cites two live pages: a **Kinematics page** and an **Encoding (goal) page**.
Those are the "where things stand" documents, with per-figure ids such as `#f-p_paths` and
`#f-lick_recovery_primary` that the map keys against.

---

## Previously "still to locate" — resolved

All found; see the table above. The FR factor panels are under `DeepLabCut/Widefield/o2/`, not
under `FaceRhythm/`, which is why the earlier search missed them.

## Open question for the text, not the figures

Check whether the Research Strategy still asserts a lesion-side bias in acute low-frequency power
or anything resting on reduced thalamocortical drive. That claim was withdrawn at T22 and should
not survive into the resubmission.
