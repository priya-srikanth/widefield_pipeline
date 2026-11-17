# Inventory A: from the outputs back to the code (cohort 1 rebuild, 2026-11-03)

CSV: `rebuild_inventory_A.csv` (5,536 rows, UTF-8 with BOM), in the same folder as these notes.
Built by `work/assemble.py` from the parsed pages and decks in `work/`, so it can be re-run.
Everything was read-only. No render ran and no process was touched.

## Method
1. **Pages: the LIVE versions, not the local copies.** I parsed every `<figure>` (fid, number, image, title, caption)
   in each page.
   - **Imaging page:** live artifact, 111 figures.
   - **Goal page:** live v312, 306 figures.
   - **Inline images on the goal page:** 127 `<img>` sit outside any `<figure>`, in `<details>` supplements and the
     "Biology in brief" cards. They have no figure number. Their rows are labeled "inline supplement (details)".
   - **Section text:** each page's prose and tables with numbers get one row per section.
   - **Pages that hold no figures:** the 10-27 review page.
2. **Kinematics and FaceRhythm pages.** Sources are on the share (`kin_page_src/kin_page/index.html`,
   `fr_page_src/fr_page/index.html`), not in Github. I scanned every caption for imaging terms (dF/F, cortex, SVT,
   LocaNMF, encoder, hemo, 415, Allen). The only hits were folder names (`imaging_encoding/...` tables) and figures
   copied from goal-page section 5. Every figure on both pages is behavior / pose / FaceRhythm only, so all are **NO**.
3. **Decks.** I used python-pptx in the locanmf env, read-only. For every picture I listed the slide, hidden flag,
   title, shape name and alt text (descr), plus a SHA of the image.
   - PI decks: the `fig:<fid>` shape names were joined to `_source_PI_overview_2026-11-02/figure_provenance.csv` and
     then to the page rows.
   - Analysis and preprocessing decks: the alt text holds the source PNG name. Rows were classified by filename family.
   - Tables and videos were listed separately.
4. **Hemo selection.** I read `wfield_local/config.py` (`svtcorr_path` / `svtcorr_in` follow `WIDEFIELD_HEMO_VARIANT`),
   `combined_encoding.load_imaging`, and grepped for hard-coded `hemo_{ADOPTED}`.

## Counts (REDO / NO / UNSURE)

| destination | REDO | NO | UNSURE |
|---|---|---|---|
| imaging results page (111 figs + 17 text rows) | 124 | 2 | 2 |
| goal/encoding page (306 figs + 127 inline + 4 text rows) | 399 | 38 | 0 |
| kinematics page (70) | 0 | 70 | 0 |
| FaceRhythm page (73) | 0 | 73 | 0 |
| review page 10-27 (19 images) | 0 | 1 | 18 |
| PI deck 11-02 (138 slides; pictures, tables, videos, titles / notes) | 62 | 70 | 0 |
| PI deck 10-26 (59 slides) | 26 | 30 | 0 |
| analysis deck: summary (1,048 slides, 1,632 pictures) | 1,620 | 12 | 0 |
| analysis deck: narrative (628 slides) | 957 | 12 | 0 |
| analysis deck: per-session appendix (431 slides) | 663 | 0 | 0 |
| preprocessing decks PS92-95 | 935 | 410 | 0 |
| other share pptx (alignment decks, history, backups) | 0 | 3 | 8 |

**Imaging page NO:** only Figs 0.1 and 0.2 (behavior).

**Goal page NO (38):**
- Figures: 3.4, 5.1-5.8, 10.6, 16.6, 16.9, 16.10, 21.13, 21.14 (schematics).
- 23 behavior supplements: 14.18 S21a/b/c, S22a-e, S24a/b, S27a, S28a-e; the revup behavior panels; enl_lick_audit.

## How the UNSURE rows were handled
- **Resolved to NO:**
  - `epoch_23_quit_prodrome` (analysis decks): the slide text shows it is lick rate before the quit.
  - The PI-deck pose / FaceRhythm crops: names did not match a page image, but the originals are under
    `DeepLabCut/...` or `kin_page` / `fr_page`.
- **Resolved to REDO:**
  - Goal 14.35 is tagged BEHAVIOR, but it reports the corrected lick response after gap matching.
  - Goal 23.11 (415 control) carries a corrected-470 row.
  - PI 10-26 slides 4 and 7 are native tables with refit / state decoder numbers.
- **Still UNSURE (28 rows):**
  - **Imaging Fig 11.2 `d_enl_move`.** It reads PER-SESSION LocaNMF components, not the joint basis. The rerun doc says
    a per-session LocaNMF refit is not reproducible (95 vs 93 components) and was not run. It needs a decision:
    project onto the joint basis, or keep as is.
  - **Imaging page "Closed / corrected / pending" text.** Re-read after the rebuild.
  - **Review page 10-27, 18 imaging images** (ROI placement on pre maps, TFCE change maps, offset contribution maps).
    This is a closed decision page and archival. Refresh only if it is still read.
  - **`labcams/20260601/alignment_comparison_PS94_PS95*.pptx` (8 decks).** June methods decks, some with lick / cue
    activity maps. Not opened. Recommend leaving them.

## Things that look surprising or easy to miss
1. **The live imaging page is ahead of every local copy.** It has 5 figures that no repo copy has: 0.2
   `behaviour_hits_by_spout`, 1.4 `decoding_refit_sessions`, 3.4 `state_position_sessions`, 3.5
   `state_ps94_diagnostic`, 4.4 `frozen_refit_pooled`. The live captions of 3.1-3.3 also carry the 11-02 z-score
   re-render. Their plot scripts live only in `M:/.../presentations/_source_PI_overview_2026-11-02/`
   (`fig_*.py`), not in any repo.
   - `wf_restnl` = 106 figures; `wf_frozendef` / `wf_restdec` / `wf_swapmask` = 105 (no 2.4b);
     `wf_lpfinal` / `widefield_pipeline` = 83.
   - **The rebuild must start from the live page.**
2. **The goal page has 127 images outside `<figure>`.** This includes the whole 14.18 S1-S28 supplement series,
   14.21 S1-S14, 12.x S1, 15.x level / scale checks and 18.x split ROIs. They have no fig_ids rows. 104 are imaging
   (REDO). Several are in the PI deck: slides 39-44 are 14.18 S21 / S22.
3. **Goal 23.23 / 23.24 exist only on live v312, as inline base64 images.** Their PNGs are on the share under
   `v2_quiet_lp/task_events_followup` and `v2_quiet_lp/sync_4to8`. There is no `_goal_page` file or fid row for them.
4. **Hard-coded `hemo_{ADOPTED}`: these scripts will not switch through the env var.** Code changes are needed in:
   - `scripts/slowband_decode_check.py` (imaging 1.3, 1.4, PI slide 9);
   - `scripts/iti_hemo_check.py` and `lick_balance_imaging_control*.py` (goal 14.3, 14.19, 14.32-14.35, 16.2);
     these compare variants explicitly, so the new variant must be ADDED as the reference row;
   - `scripts/s1_refit_basis_table.py`, `encoding_drift_check.py`, `iti_mixed_models.py`, `o2_video_me.py`,
     `wfield_local/postcue_window_test.py`.
   - Separately, `combined_encoding.load_imaging` is env-aware only on `wf_swapmask` (b334316a). It is hard-coded on
     `wf_attempt`, where every goal-page lick / quiet script runs.
5. **Imaging-derived figures that live only in decks:**
   - PI 11-02 slides 16-18 and 124-126, and PI 10-26 slides 8-10: RESTW position maps from
     `grant_figures/LP_licks/epoch/epoch_15r_position_RESTWref_*`.
   - PI 11-02 slide 98: the quiet pre-vs-acute movie.
   Those deck-only figures are REDO.
6. **O2-dependent figures.** The per-session encoder (goal 1.6, 1.13-1.15, sections 21-22; imaging 13.4-13.6 trend
   runs) came from O2 batch `enc_v36_full_20261024`. Re-rendering needs an O2 run on the variant, or the variant SVTcorr
   staged to O2.
7. **Pre-baseline exposure.** These imaging-page figures use decoders with a June+August pre:
   - 1.1, 1.2, 2.1-2.4b, 3.1-3.5, 4.1, 4.2, 4.4;
   - 5.1-5.2 (stored DAQ model; pre set not stated, UNSURE);
   - 9.1-9.2 (pre set not stated);
   - 11.1, 12.1, 14.1-14.9.

   On the goal page the same applies to sections 6 and 8, 0.5 / 0.6 / 0.12 (attempt-coding stored DAQ model) and the
   23.5 / 23.23 precue decoder.

   Fig 4.3 (j_drift) and the drift maps keep June **by design**: June is the no-stroke reference.
8. **LP-arm training.**
   - Fig 2.4 is LP-trained but was fit in memory (corrected caption).
   - 2.4b, 2.5 and 2.7 are DAQ-trained stored; they become the cross-check.
   - 2.6 and 2.8 are LP-trained stored (August-only already).
   - Open: 5.1 / 5.2 and goal 6.x / 8.x use the stored DAQ model on LP-gated trials, so the LP-trained rule plausibly
     applies there too.
9. **Analysis decks (`labcams/spout_position_analysis_summary*.pptx`) were last built 10-24.** So they already differ
   from the pages (pre-z-fix state decoder, no RESTnl, no LPfinal). A rebuild by `locanmf_analysis_deck` / nightly_figs
   regenerates them wholesale.
   - The analysis decks' per-session `locanmf_position_session_*` decoders use per-session LocaNMF, the same issue as
     Fig 11.2.
   - The registry's grant / epoch sections are the deck's sections H and I. Section K is the LP_licks arm.
10. **Preprocessing decks PS92-95 (10-06, 35 sessions each).** The per-session activity maps (cue / lick / quiet maps,
    7 slide types) are REDO. Motion QC, photobleaching, 415/470 mean + Allen overlay, and the cross-day vasculature /
    raw ROI slides are NO, because they come before hemo correction. These decks also predate the October chronic
    sessions.
11. **Imaging section 15** is a frozen copy ("moved to encoding page 23, no longer updated"). Decide whether to
    re-render it or label it superseded.
12. **Figures with no traceable source in a repo:**
    - Imaging 3.3 `per_animal_fig.py`: copy in `imaging_page_src/s_per_animal_regen_20261102/`.
    - Imaging 5.1 / 5.2 and goal 8.1 / 8.2: `fig_goal_summary.py` and `fig_goal_epochs.py`, "session scratchpad, outside
      repo" per `title_verification.csv`.
    - Imaging 11.2: plotted by a session script that is not in the repo.
    - Imaging 0.2, 1.4, 3.4, 3.5, 4.4: share-only `fig_*.py` scripts.
    - PI slide 51 `master_mesh12`: "none on the share".
    - Several imaging-page figures have `?` in `provenance/fig_sources.txt`: the section 7 maps / drift, the `cp_*`
      figures and `m_trend_beh_contact`. Their scripts are named in the captions, but the exact source PNG path is not.

## Not covered / caveats
- I did not open the 8 June alignment decks or the 75 deck_history snapshots.
- I did not diff each analysis-deck image against `deck_registry.py`, so registry orphans are not listed.
  `scripts/deck_figure_coverage.py` does that.
- Pre baseline and frozen store for the analysis-deck rows are set at family level, not per image.
- The goal page's section prose is one row per block, not one row per number.
