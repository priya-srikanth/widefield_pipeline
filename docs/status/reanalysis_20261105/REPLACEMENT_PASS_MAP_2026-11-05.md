# Replacement-pass map, cohort-1 rebuild (2026-11-05)

Read-only map for the one-pass figure replacement after the cohort-1 rebuild (requested by sabatinilab-34, built by a
helper of sabatinilab-a8). Nothing was edited, published, rendered or committed. The table is
`REPLACEMENT_PASS_MAP_2026-11-05.csv` (beside this file). It has one row per figure, inline image, LMM toggle or deck
slide, plus rows for the new / pending items in PAGE_EDITS_PENDING_2026-11-04.

**Columns:** output, number, hidden (deck only), fig_id (deck = picture shape names), image_file (deck = slide title),
script (deck = the notes' FIGURE SOURCE line), tag_or_folder, old_png_source, rerender (yes / no / check / text),
reason, owner.

## Sources used

- **Encoding page:** the saved live HTML `tool-results/artifact-77c5daa3-1793703992-c30f.html` (304 `<figure>`s +
  129 images outside figures). Scripts come from the captions, then from MASTER_MANIFEST.csv
  (wf_attempt rebuild_20261103). Where neither names one, the script is inferred from ENCODER_RENDER_COMMANDS and
  marked "(inferred)".
- **Imaging page:** the LIVE artifact (Artifact read, version 1793740803-1eef, 111 figures). The local repo copy
  (`docs/results_pages/imaging/imaging_page/imaging_results.html`, 11-01) is stale: it has 83 figures and lacks
  2.4b-2.8, 1.4, 3.4, 3.5, 4.4 and others.
- **Kinematics page:** `M:/.../kin_page_src/kin_page/index.html` (11-03 22:15, post-split with the TOC toggles; 68
  figures + 12 LMM toggles).
- **Behaviour page:** `M:/.../behaviour_page_src/behaviour_page/index.html` (3 figures + 2 LMM toggles).
- **Deck:** PI_overview_2026-11-02.pptx, read with python-pptx and not saved (187 slides, 82 hidden).
  spout_balem_swaps.json was cross-read. Its slide numbers are from an older numbering, so match on the shape names
  (fig:d_acc_cue = slide 13 now, etc.).
- **Rules:** REBUILD_CODE 8.51-8.56 (wf_rebuild, newest), ENCODER_RENDER_COMMANDS.md, RECONCILIATION.md (D1, D4, D8),
  RENDER_LOG.md (step 6 spont lanes) and PAGE_EDITS_PENDING A/B.
- **Not covered:** the FaceRhythm page (the e0 owner list includes it). The deck's FaceRhythm slides are mapped "no".

## Path note

The `_daqlk` folders exist only under the **rebuild root**, not under the path in the request:
`M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-...stopA2jexrt3500c70_daqlk/`.
That folder holds:
- 11 `frozen_encoder_gate-attempted_v7g3_t3d_spont*` frames, plus the no-spont `_v7g3_t3d` and `_frA`. The no-spont
  frA lane was still running at 09:42.
- 15 collected O2 run folders (`movement_first_*`, `musall_subset-shuf`, fr A/B/D/S), plus `fr_encoding_summary`,
  `variance_maps` and `qc`. The request says 16 TAGs; 15 run folders are visible.

No attempt_coding, iti_licks or quiet-event outputs exist under the rebuild root yet. The frozen figure scripts
(ENCODER_RENDER_COMMANDS section B) have not been run.

## Counts (rows)

| output | owner | yes | check | no | text |
|---|---|---|---|---|---|
| encoding page: figures (304) | 34 | 298 | 4 | 2 (schematics 21.13, 21.14) | |
| encoding page: inline images (129) | 34 | 99 | 30 | | |
| encoding page: new items (A14, A17, B8, B11-B13) | 34 | 5 | | | 1 |
| imaging page: figures (111) | e0 | 103 | 8 | | |
| imaging page: new items (A11, B1-B7, B11) | e0 | 6 | | | |
| kinematics page: figures (68) + 12 LMM toggles | e0 | 9 | 8 | 63 | |
| kinematics page: new items (B9, B14) | e0 | 2 | | | |
| behaviour page: 3 figures + 2 toggles + 2 new | e0 | 7 | | | |
| PI deck: 187 slides | a8 | 95 | 17 | 75 | |

Encoding figures by input family:

| family | rows | what to re-render from |
|---|---|---|
| ITI / lick chain | 130 + 91 inline | rendered from chain E (iti_licks with ENV-frozen labels from the spont frA frame) |
| frozen encoder | 82 + 6 | read from the v7g3_t3d_spont frames |
| quiet events (sec 23) | 24 | |
| decoder / attempt_coding | 24 + 2 | |
| O2 FR (sec 21-22) | 18 | |
| self-fit (sec 4) | 8 | |
| gate-only behaviour (sec 5 landing) | 8 | |
| binned contest (1.6, 1.13-1.15) | 4 | read from the O2 `_daqlk` runs |

## "check" rows (what is unknown)

- **Encoding 10.6, 16.6, 16.9, 16.10 and kinematics 1.7, 1.8 (behaviour-only; manifest NO):** does their engaged gate
  move to camera classes (A6) or the August-only pre? Deck slide 8 (16.9) follows.
- **Encoding: 30 inline behaviour panels of Fig 14.18 / 14.21 (rec_wlc_*, rec_kn_*, rec_lc/pht/pst/pat/fpa,
  dis_*, revup_*):** chain E re-runs within_lick_correction / learned_compensation / disengagement. Do the
  behaviour-only PNGs actually move? The dis_* panels also depend on A6 (camera quit;
  ENGAGEMENT_CAMERA_2026-11-05). Deck slides 57-60 and 62-64 follow. Slide 61 (rec_wlc_neural) = yes.
- **Imaging 15.1-15.7 (quiet-event copy):** RECONCILIATION D4 recommends replacing them with a pointer to encoding
  section 23 instead of re-rendering. No decision was found.
- **Imaging 11.2 d_enl_move:** a per-session LocaNMF family (manifest DECISION, D1 projection). Confirm the rebuild
  renders it.
- **Kinematics 6.1-6.5 + toggle (trial classes over engaged trials):** do A6 camera quit / A16 stopped excision
  re-gate them? Deck slides 36-42 follow.
- **Deck 102 (B7 gain_sessions):** unknown whether 34 re-renders gain_bio_1104 under `_daqlk`.
- **Deck 144 (quiet example movie):** the clip windows and 4 x SD event thresholds could move with the A17 detector.

## Deck slides to swap (rerender = yes; 95)

- **Visible (53):** 4, 11, 13, 15, 17, 22, 23, 24, 25, 35, 66, 67, 68, 89, 90, 91, 94, 95, 103, 105, 107, 108, 109,
  110, 112, 114, 115, 117, 118, 120, 122, 125, 129, 131, 133, 134, 136, 137, 138, 139, 143, 145, 146, 147, 170, 171,
  172, 177, 182, 183, 184, 186, 187.
- **Hidden (42):** 5, 12, 14, 16, 18, 19, 20, 21, 26, 61, 65, 69, 92, 93, 96-101, 104, 106, 111, 113, 116, 119, 121,
  123, 124, 126, 127, 128, 130, 132, 135, 140, 141, 178, 179, 180, 181, 185. LMM companions are re-fit when their
  main slide is.
- **Slide 95 (B8):** the title is already reworded ("Lick response larger from subacute on, beyond global gain").
  Swap the figures, and the encoding page wording must match.
- **No swap (75):**
  - title / divider slides;
  - histology (3);
  - kinematics camera slides (6, 7, 9, 32-34, 43-56, 173-176);
  - pose videos / QC (29-31);
  - FaceRhythm (71-86);
  - cohort 2 (149-151, 154);
  - VFSS (155-157);
  - 2pRAM (158-168).

## Main decisions encoded

- Kinematics camera figures stay "no" in this pass. They use the behaviour-pipeline gate, and the jaw-only rule
  leaves kinematics outputs unchanged. A7 / A15 change a methods line only. 3-D / spout changes land with the next
  3-D re-run (PAGE_EDITS "NOT in this pass").
- Behaviour hit-rate / engagement figures are "yes" under A6. This supersedes D8. They are a behaviour-pipeline
  re-render, not a `_daqlk` run.
- Encoding section 5 landing figures (and their kinematics 4.12-4.19 copies) are "yes". Their captions say trials =
  the encoder run's attempted gate, which is now the camera-selection trial_attempts table.
