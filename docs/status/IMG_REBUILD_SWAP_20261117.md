# Image page rebuild swaps, FINDINGS, deck section G divider (11-17)

Branch `fix/img-rebuild-swap-1117` (worktree `wf_imgswap1117`, from origin/main a36615b8 + merge of
`fix/deck-figs-1117` bc504b2a, which was NOT yet on origin/main or local main). Priya OK'd 11-17. T = run tag
`hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`.

## A. Imaging page (https://claude.ai/code/artifact/PvraHiF82BFXur1hnNUCeV) v56 -> v57

Edit script: scratchpad `pages/img/edit_v57_rb1117.py` (20 exact replacements on the saved v56 HTML). New images are
published files `*_rb1117.png`; every other file kept.

| figure | placed file | old -> new numbers |
|---|---|---|
| 2.3 frozen decoder, cue | `labcams/grant_figures/T/epoch/epoch_acc_by_position_cue_working.png` (11-09 post-glue) | far contra .896 -> .321, -.575 [Bonf -.738, -.394] -> .879 [.802, .930] -> .240 [.137, .375], -.618 [-.695, -.531] (Bonf [-.722, -.488]); per animal .86/.78/.92/.93 -> .19/.15/.30/.32 (4 / 4); all 6 positions fall acutely (5 Bonferroni) |
| 3.4 position vs state | `rebuild_figs/T/state_position_postglue_20261117_footer/state_position_sessions.png` | position "loses .41" (-.41/-.19/-.09) -> .835 -> .488, -.347 [-.412, -.287] Holm .0015 / -.139 Holm .0015 / -.077 Holm .057 n.s.; state -.033/-.026/-.055 -> -.023 [-.044, -.006] Holm .012 / -.028 n.s. / -.056 Holm .006 |
| 9.3-9.11 CD leave-one-out | `labcams/grant_figures/T/epoch/cd_loo/` (11-17 run, DAQ lick trials, August pre 32 incl. PS93_0813) | see the table in DECK_KEYFIGS_20261117.md section 5; precue far contra acute -.805 n.s. 3/4 -> -.687 [-1.26, -.10]* 4/4; cue far middle acute -.474* -> -.404 [-.72, -.05]* 3/4; 10 of 54 starred (was 5) |

Also rewritten: status rows (frozen decoder, CD leave-one-out), section 2 input line + pooled takeaway, section 3
11-03 note "Change vs pre (Fig 3.4)", section 9 input line, the per-animal LOO table (rebuild session means) and the
held-out pre line (0.63-0.86; acute pre-cue far contra 4 / 4).

Fig 3.4 footer: the trial-set footer was stamped at save time, after `tight_layout`, and sat on the tick labels.
`fig_state_position_sessions._footer_bottom` now stamps it before the layout and reserves its height. Re-rendered
`--plot-only` into the new `_footer` folder (the 11-17 render beside it is untouched).

Not changed, noted: Fig 2.3's own footer (epoch_accuracy render) overlaps the Near / Far group labels and its long
lines are clipped at the left edge; a re-render of that epoch family would be needed. Section 13 LMM rows on the CD
leave-one-out values are pre-rebuild (not re-run).

## B. docs/FINDINGS.md

Replaced: claim 1 (frozen decoder far contra), claim 2 (Fig 3.4 per-session sentence; the ratio is now an acute /
subacute claim, chronic position change n.s.), claim 10 (CD leave-one-out), claim 14 (far_center chronic -.669).

## C. Analysis deck

`deck_text.G_DIVIDER_REBUILD` is the section G divider subtitle in a tagged build (per-animal lesion dates, epochs,
sessions, behavior first); the untagged "Lesion <cutoff>; post-stroke pool = ..." line is kept for untagged builds.
Key figure 3.4 points at the footer-fixed render. Previous deck copied aside as
`spout_position_analysis_summary__20261117_1201.pptx`. Build: render profile `cohort1_rebuild_20261103` env, then
`python -m wfield_local.locanmf_analysis_deck --no-within-day --run-start 1794286800` (run_start 2026-11-10, as the
previous manifest).
