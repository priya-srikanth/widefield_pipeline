# Deck / image-page follow-ups, 11-17 (Fig 2.3 footer, section 13 CD leave-one-out models, G0, outdated slots)

Branch `fix/deck-img-followups-1117` (worktree `wf_deckfix1117`, from 01af2e62; it was not yet on origin/main). Priya
OK'd 11-17. T = `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`.

## 1. Fig 2.3 footer (acc family)

The cause: `epoch_figures._save_png_svg` wrapped the run stamp only when it had no newline. The camera-gate coverage note
makes the stamp multi-line, so its long line ran off the left edge, and the stamp was drawn after the layout, on top
of the Near / Far labels. The fix is `epoch_figures._fit_stamp`, which wraps every line, narrows the wrap until the
stamp fits the canvas, and grows the canvas at the bottom by the overlap so the axes keep their size in inches. A
figure the stamp does not touch is left unchanged. `replot_figdata` now restores the bal_em fallback sessions from the
meta sidecar; without that, a replot dropped the "contact-row fallback for PS93_0813" stamp line from the lick figures.
Test: `tests/test_epoch_stamp_fit.py`.

Redrawn plot-only from the stored `_figdata.pkl` (no re-scoring, ~1.5 min) into a NEW folder,
`rebuild_20261103/labcams/grant_figures/T/epoch_footerfix_20261117/`, covering all 10 acc / accdelta figures (cue_working,
precue_working, cue_lick, precue_lick, lick_lick). Every `data/*.csv` (values, sessions, meta) is byte-identical to `epoch/data/`.
Other figures that get the fix at their next render or replot: every epoch figure saved through `_save_png_svg` in a
tagged run whose stamp overlaps (for example the LPfinal / LP_licks acc figures, including key figure 2.5). Those were not
redrawn here.

Deck: key figure 2.3 and section I acc / accdelta now place the footer-fixed folder
(`deck_registry.EPOCH_FIGURES_REBUILD_DIR`, tagged build only).

## 2a. Section 13: CD leave-one-out mixed models on the rebuild

Script: `scripts/cd_loo_lmm.py` (the `imaging_lmm.epoch_tables` transform, source `epoch:cd_loo` only, FDR
within the source, as in imaging_lmm). Run on the 10-21 file with the default env, it reproduces the page's 10-25 numbers
exactly. Outputs: `rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/imaging_lmm_cdloo_{rebuild,pre-rebuild_repro}_20261117.csv`.

Far contra, n = 105 session rows, 4 animals. Deficits: ttc = linear time to contact (the page's 10-25 column), ttc_log =
the primary deficit since 11-05, contact = the contact share.

| row / term | old (10-25) | new (rebuild) |
|---|---|---|
| pre-cue: step at stroke | -.52 (z -3.0, q .019)* | -.52 (z -3.0, q .010)* |
| pre-cue: log-day slope | +.099 n.s. (q .14) | +.091 n.s. (q .15) |
| pre-cue: vs deficit, ttc linear | -.080 n.s. (q .25) | -.063 n.s. (z -2.4, q .054) |
| pre-cue: vs deficit, ttc_log (primary since 11-05) | not run | -.20 (z -2.8, q .019)* |
| pre-cue: vs deficit, contact | -.45 (q .050) borderline | -.58 (q .051) borderline |
| cue: step at stroke | -.23 (q .062) n.s. | -.37 (z -1.7, q .18) n.s. |
| cue: log-day slope | n.s. (q .53) | n.s. (q .62) |
| cue: vs deficit, ttc linear | n.s. (q .13) | n.s. (q .38) |
| cue: vs deficit, ttc_log | not run | n.s. (q .43) |
| cue: vs deficit, contact | -.22 (z -2.3, q .043)* | -.24 (z -0.9, q .48) n.s. |
| epoch model, cue acute | -.30 (q .002) | -.40 (q < .001) |
| epoch model, cue chronic | -.12 (q .14) | -.15 (q .022)* |
| epoch model, pre-cue acute | -.50 (q .008) | -.57 (q .001) |

CONCLUSION CHANGES, so the page was NOT edited (Priya to decide):
- cue vs contact deficit loses significance (q .043* -> .48);
- pre-cue vs deficit is significant under the current primary deficit (ttc_log, q .019*), where the page says "no longer
  tracks the time-to-contact deficit (q .25)". Under the linear definition it is still n.s. (q .054).
- The section 13 table, the takeaway, the status row, the closed-list line "CD leave-one-out vs deficit no longer
  significant" and the section 9 sentence "mixed-model tables ... due a re-run" were all left as they are.

## 2b. G0 slide (tagged build)

`deck_text.G0_REBUILD` replaces the reference / trials / engagement / pool bullets: the August-only pre (31 sessions),
the camera trial gate, the working period plus quit tail as the only engagement cut, and the epoch session counts. The
arm, lick-source and replication bullets are kept. An untagged build is unchanged.

## 3. Outdated slots left out of tagged builds

`deck_registry.REBUILD_DROPPED_EPOCH` (31 section I patterns: 1f, 1g, 15, 15pa, 15x, 15d, 15s, 15j, 15g, 15h x5,
15k x6, 16, 20, 21, channel_position_epoch x2, 23, 24, 25, 26 x3), `REBUILD_DROPPED_LP` (LP cue-window accuracy, LP 5c
confusion, DAQ_ENL1s pre-cue) and `REBUILD_DROPS_SECTIONS_EF` (cross-mouse, within-animal consistency, RSA x3 plus the
hemisphere RSA; the rolling and encoder-EV slides were already out with `--no-within-day`). Test:
`tests/test_deck_rebuild_dropped.py`. Checked against the 12:0x build's missing list (41, captured without writing):
every missing item was on Priya's list except the two 15f files, which were kept on purpose. Nothing unlisted was missing.

## Deck rebuild

The previous deck was copied aside as `spout_position_analysis_summary__20261117_1239.pptx`. The rebuild used the
profile env and `--no-within-day --run-start 1794286800 --allow-missing 2`: 594 slides (was 601), 811 figures placed,
2 missing (`epoch_15f_rest_frozen_restdock05_durmatched_gapmatched.png` and its `_confusion.png`). G0 (slide 33) carries
the rebuild bullets, and key figure 2.3 (slide 13) is the footer-fixed render.

## Image page

The edited HTML (from v57) is `scratchpad/pages/img/img_v58.html`, with the new image `d_acc_cue_fb1117.png`; all other
files stay as they are. The edits cover Fig 2.3 only: the image, the caption provenance and the source path. The page
was not published because the Artifact tool is signed in to another account.
