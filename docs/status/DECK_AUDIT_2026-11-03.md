# PI deck audit, 2026-11-03

Deck: `M:/MICROSCOPE/Priya/Widefield/presentations/PI_overview_2026-11-02.pptx`. It had 134 slides before the audit and has 138 after.

The audit compared every slide (including hidden and backup slides) with these pages, read 11-03:
- imaging page v49 (PvraHiF82BFXur1hnNUCeV)
- kinematics page v18 (DKcKRJKbhBoratxxRc6qZv)
- encoding / goal page v312 (Fnpn2KHknMwh3r13xADH7d)
- FaceRhythm page, 10-26 evening (X8NDdQNHi3zMP77PBrZAC1)

It also used FINDINGS, DECISIONS (10-25 onward), the 11-01 and 11-02 running lists, the quiet-events handoff, and Priya's 11-03 answers to decisions 1-9 as relayed by the coordinator. Nothing was published to any page.

## Scripts

These run-once scripts are in `presentations/_source_PI_overview_2026-11-02/`. Each one backs up the deck first.

| Script | Backup | What it does |
|---|---|---|
| `audit_1103.py` | `backup_before_audit_1103.pptx` | Main audit step. Takes `build_inputs/page_figmap_1103.json`, the page figure-tag map from the 11-03 reads. |
| `fix_audit_1103.py` | `backup_before_fix_audit_1103.pptx` | Moves the new trial-class slides after the hidden supports of slide 27. Applies the slide-13 "pre-fix record" marker. |
| `fix2_audit_1103.py` | `backup_before_fix2_audit_1103.pptx` | Fixes found in the export check: title wrap, LB333 caption overlap, and review-sheet bullet wording. |

Helpers used:
- `build_inputs/kin_contra_crops_engaged.py`
- `build_inputs/retile_restw_maps.py`
- `build_inputs/twop/retile_red_detection_gallery.py`

`figure_provenance.csv` is updated, including the renumbering. Every changed slide was checked with a PowerPoint COM PNG export of a read-only copy.

4. `short_titles.py` (backup `backup_before_short_titles.pptx`; run LAST, Priya 11-03 "shorter titles"): 126 titles
   shortened by an explicit old -> new map, saved as [`DECK_TITLES_2026-11-03.csv`](DECK_TITLES_2026-11-03.csv)
   (slide, old, new); `python short_titles.py --reverse` undoes it from the CSV. Titles only (bullets / notes
   untouched; a few notes still quote an old title, e.g. the trial-class DENOMINATOR line). Long titles that had been
   set at 24 pt went back to the deck's 28 pt; all 138 title bands checked in a PowerPoint export (one line each).
   `build_inputs/kin_camera_view_swaps.json` did not exist when this ran, so no spec was remapped: a later spec must
   use the NEW titles (map in the CSV).

## Change table

Slide titles in this table are the titles BEFORE the short-title step (map: DECK_TITLES_2026-11-03.csv).

Slide numbers are the final numbers (138-slide deck).

| Slide | Title (short) | What was stale | What changed | Source |
|---|---|---|---|---|
| 10 | Frozen decoder fails acutely | Bullet ".32" in the lick window came from the decoder the page now flags as LP-trained (Fig 2.4 correction, 11-02). The notes gave no lick source. | Bullet now reads ".32 (.26 with the DAQ-trained decoder)". The notes gain the correction, the DAQ-trained numbers per animal, and the lick source (cue = DAQ contact; lick window = LP video). | imaging Figs 2.4 / 2.4b |
| 13 (hidden) | Position fails far more than state | The notes still said PS94 loses LICKING (read as quiet). A pre-fix paragraph (~5x, state chronic .45, etc.) read as current. | The notes now say PS94 loses its RUNNING class (11-03 correction). The pre-fix paragraph is marked as superseded, and the current dot values were added (.97 / .92 / .86 / .86). The bullets were left as they are (Priya approved them). | imaging Fig 3.5; FINDINGS 3.2 |
| 27-30 | Trial classes (jaw-moved table) | The denominator was described as "jaw passed QC". jaw_pass_qc is actually the jaw-moved call. | A DENOMINATOR line in the notes says these slides cover jaw-moved engaged trials only, and points to slide 31. | coordinator / trial_class_all_engaged.py |
| 31 NEW | Every engaged trial classified | New | MAIN slide. Kinematics Fig 6.1 `tc_share_all.png` (suptitle cropped). All numbers were checked against `tc_share_all.csv` / `tc_share_all_tests_added.csv`. Placed after the hidden supports of slide 27, so in the show it directly follows slide 27. | kinematics page section 6 (Fig 6.1) |
| 32 NEW (hidden) | Change vs pre, every class | New | Fig 6.2 `tc_share_all_change.png`, suptitle cropped. | kinematics Fig 6.2 |
| 34 | Acute licks aim ipsilesional | The notes gave 3-D midline error as "+25/+10/+36/+33 toward ipsi (3-D keeps + = ipsi)". | Now −25 / −10 / −36 / −33 (+ = contra, page v18). Text only. | kinematics v18 |
| 35 | The shift spans the whole lick | The picture was the prepost_alltrials render. The notes said "(+ = ipsi)" rows, and the 3-D twin was "+30-40°". | Picture swapped to the ENGAGED-trial (prepost) + = contra crop, same recipe (Priya 11-03). The notes now say engaged trials, + = contra, and a 3-D twin of −30 to −40°. | kinematics Fig 2.5; Priya decision 9a |
| 37 | 3-D: far-contra licks miss | The notes said "this figure keeps + = ipsilesional". | That note is marked superseded (+ = contra since v18). | kinematics v18 |
| 99 → 101 | Pre-cue event / weaker code | Bullet 1 did not say what the claim rests on. The notes cited only the DAQ-defined T13 follow-up. | Bullet 1: "After the stroke a pre-cue orofacial event goes with a weaker position code (main and exclusion tests, 4 / 4; two pre-stroke sub-tests also lean negative)". The notes add the LP-defined re-run (unchanged, −0.016, p .001; per animal), that 4-8 Hz was not adopted, and that T20 laterality stays withdrawn. | encoding Figs 23.5 / 23.23; Priya decision 6 |
| 97 | ~7x more common | Fold change checked: the deck says ~7x / ×6.7, which matches the current page (LP quiet frames; DAQ-only ×5.1). | The notes record the check. No visible change. | encoding Fig 23.1 |
| 109 | Pre-stroke VFSS | The notes said "stroke date unknown" and "20261029 folder likely mis-dated". | Stroke 11-01. Clips are labelled 10-30 (pre). The notes say the 20261029/ folder is the 10-29 session saved under 20261028_* names (files not renamed), and that no clip here comes from it. | Priya decision 2 / coordinator |
| 110 | Post-stroke VFSS | The slide said "First post-stroke session (11-02)" and the notes asked to "confirm stroke date". | Bullet now reads "Post-stroke day 1 (11-02; stroke 11-01)". Notes updated. | same |
| 111 (hidden) | Other post-stroke clips | Same | Title now reads "the other day-1 (11-02) clips". Notes updated. | same |
| 113 NEW (hidden) | NES-mCherry (PS110): positive vs negative | New | Review sheet (see the mCherry section below). | 2p-RAM red-detection gallery |
| 114 NEW (hidden) | H2B-mCherry (PS104): positive vs negative | New | Review sheet. | same |
| 118, 119 (hidden) | Lick-direction cells, map | Notes said ipsi / contra were "renamed relative to the planned lesion". The map choice was awaiting a decision. | The notes now say ipsi / contra are relative to the planned lesion side AND the imaged LEFT hemisphere. The map choice is confirmed (it matches grant panel A, 131 / 231 / 3,390). | Priya decision 5a / 5c |
| 122 (hidden) | Single GCaMP axons | No construct, injection site or target was recorded. | Bullet: "LB333 axon-GCaMP8s from ipsilesional (left) VM thalamus". Caption added below the image (the image was shrunk to make room). Construct added to the notes. | Priya decision 5b |
| 124 | Backup: pre-cue maps | It showed the DAQ-lick-free ENL arm (1.84 / .38, 1,102 bins), which disagreed with the main pre-cue slide 18 (LP version, 1.56). | Swapped to the LP-lick RESTW pre-cue figure, re-tiled with the same recipe; the recipe was validated by reproducing the old DAQ crop. Bullets now give 1.56 and .48 (1,155 bins), beyond drift (LP drift Fig 7.4: far middle .97 vs chronic .30). New title: "Backup: pre-cue position maps (1 s video-lick-free)". The notes were rewritten with LP numbers and the old DAQ numbers. | LP_licks/epoch/epoch_15r_position_RESTWref_precue_working(_stats.csv); imaging Figs 7.3 / 7.4 |
| 127, 128 | Backup: whole trial, in-mouth excl / incl | The pictures were the prepost_alltrials render. | Swapped to the ENGAGED-trial + = contra crops. The notes now say engaged trials. | kinematics Figs 2.1 / 2.2; Priya 9a |
| 64 slides with a page figure | (all) | Most notes did not name the lick source. | Each slide's notes gain a line "LICK SOURCE (page tag, read 11-03): Fig N [licks: ...]", copied from the page figure titles. This was skipped for slides 24, 25 and 36, which another helper is re-plotting. | imaging v49 / encoding v312 tags |

These were checked and left unchanged because they are current:
- Slides 5-7, 33, 36, 38-39: + = contra already.
- Slides 9, 11-12, 15-19: page primary is DAQ for the cue-window decoders, and LP for the maps and the pre-cue decoder.
- Slides 43-47, 63-93, 125-138: every embedded picture is byte- or pixel-identical to its current page source, and the page figure ids are unchanged.
- FaceRhythm slides 49-60: numbers match the page; it already uses LP licks.
- Quiet slides 97-100, 137: LP quiet frames already.
- Cohort2 slides 103-105.
- Big picture (106) and slide 6: approved by Priya as they stand.

## Main-point changes flagged for Priya

1. **Slide 124 (backup pre-cue maps).** The point is the same: far contra is not lost before the cue, and far middle is lost chronically beyond drift. The numbers change (1.84 → 1.56 acute far contra; .38 → .48 chronic far middle), because the backup now uses the LP version to match main slide 18.
2. **Slide 101 (quiet events).** The first bullet now states what the claim rests on (main + exclusion tests; pre-stroke sub-tests also lean negative). This is weaker than "only after the stroke", as Priya asked.
3. **Slide 10.** The headline .32 is kept, because it is the page's take. The DAQ-trained decoder gives .26 (deeper loss), and the slide now shows both.
4. **Trial classes (slides 27-30 vs new 31-32).**
   - Under the all-engaged table the main point of slide 27 still holds, with smaller numbers. Far contra incomplete is +.57 (was +.63), 4 / 4.
   - Jaw-only is +.21 / +.19 / +.09 at far contra / mid / ipsi (was +.27 / +.23 / +.11).
   - A **new class appears: no-response trials, +.09 to +.11 acutely at every far spout (**)**. These are invisible in the jaw-moved table, where 'none' is ~0 by construction.
   - "Close spouts barely change" still holds (close mid / contra contact −.07, **).
   - **Option for Priya, not applied:** make slide 31 (composition over all engaged trials) the main trial-class slide and demote slide 27 (trial_class_share_epochs, jaw-moved table) to hidden.

## mCherry review sheets (task 2)

Found on M:, not built new. Each animal's red-detection class gallery comes from `2pRAM_pipeline_multiacq` `scripts/red_detection_gallery.py` (Priya 09-28). It shows 8 random red objects per class (seed 0), with GCaMP | corrected mCherry (0-15 σ) | overlay, the ROI outline, and 40 µm panels.

The deck re-tiles only the classes "own ROI, called +" and "own ROI, called −", 2 × 4 cells each. The pixels are unchanged.

- **Slide 113, NES-mCherry, PS110 2026-09-25**
  - Source: `M:/MICROSCOPE/Priya/2p-RAM/PS110/20260925/analysis/mcherry_call/PS110_20260925_red_detection_gallery_revb02835f.png`
  - Refined call rev74c59ab: 891 / 4,276 called (union); 528 strict.
  - Red objects: 883 under called ROIs, 750 under uncalled ROIs.
- **Slide 114, H2B-mCherry, PS104 2026-10-01**
  - Source: `M:/MICROSCOPE/Priya/2p-RAM/PS104/20261001/analysis_thr08/mcherry_call/PS104_20261001_red_detection_gallery_thr08_rev0f1de17.png`
  - Refined call rev0f1de17 (the call of record; 30 µm² floor): 826 / 4,285 called (19.3 %).
  - Red objects: 848 under called ROIs, 1,288 under uncalled ROIs.
- **Both are HIDDEN.** The identification slide (112) already has example cells.
- **NES is placed above H2B**, as in grant panel C.
- **The "negatives" are near misses.** They are ROIs that carry a red object the call rejected. They are not typical unlabelled cells, and the notes say so.

Other candidates considered:
- `mcherry_qc/*_mcherry_gate_gallery_*`: PS104 10-01 rev a79c20b and PS110 09-25 mosaic rev2e3c800-dirty. These show an older gate (15 µm² / contained 0.55), not the call of record.
- `analysis/mcherry_labels/examples_*_calib_rev2374a48.png`: positives only, with animals mixed.

## Unresolved items for Priya

1. **Slide 25 (hidden, k_acc_pos per-spout figure).** Its "executed peak angle (+ = ipsi)" row has no + = contra version, and the figure was not re-plotted. Priya said not to touch this figure (another helper is re-plotting it). The slide-24 figure and the 2-D / 3-D path slides (36, and 39 / 47 on the encoding page) were left for that helper's swap spec.
2. **Slide 25 also draws lines across epochs** (orofacial_cohort figure style). It needs a re-plot to follow the deck style.
3. **Big picture (slide 106), quiet bullet.** It still says "it goes with a weaker pre-cue position code" without the main + exclusion qualifier. It was left as approved; should the qualifier be added?
4. **Imaging section 7 RESTnl maps.** The RESTnl versions sit beside REST on the page ("which reference leads is for discussion"). The deck uses RESTW (slides 16-18, 124-126). Pick a reference to lead.
5. **Imaging section 15** (frozen copy) still says ~5× / DAQ in its figure titles. The current numbers are in encoding section 23. This is a page issue, not a deck issue.
6. **Kinematics page section 6 numbering** of the new trial-class figures is assumed to be 6.1 / 6.2 (cited as "section 6"). Confirm once published.
7. **PS110 mCherry counts differ by source.** The deck's slide 115 quotes 559 / 3,804 (strict soma gate, per-tile dedupe, grant summary). The review sheet uses the refined union, 891 / 4,276. Both are explained in the notes; pick one for the main text.
8. **State decoder decision 7** is still open: class-balanced centring vs the plain z-score.
