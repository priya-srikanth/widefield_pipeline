# Rebuild analysis deck, 11-17: key figures, section K, CD leave-one-out, 15r skip

Branch `fix/deck-figs-1117` (worktree `wf_adeck1117`, from origin/main 36cabf95). Deck:
`rebuild_20261103/presentations/analysis/<run tag>/spout_position_analysis_summary.pptx`. Run tag below = T
(`hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`).

## 1. Key figures: page figure -> file placed (matched by script + content)

Code: `wfield_local/deck_keyfigs.py` (generated from the matcher's table; page captions reused verbatim only where the
placed file IS the page image), placed by `locanmf_analysis_deck` after the goal page's own key figures.

| page / figure | placed file (under rebuild_20261103 unless noted) | what it is |
|---|---|---|
| Goal Key figure 2 | `.../iti_licks/lick_order_dock_20261117/T/figures/lick_order_iti_cued_slope.png` | rebuild render 11-17 of `scripts/iti_lick_order.py` with `WFIELD_ITI_FILTER=dock` (the page's `_docked` banner copy is the same figure, docked ITI subset) |
| Goal Fig 12.2 | `.../iti_licks/matched_gauss_dock_20261117/T/figures/lick_matched_gauss_curves.png` | rebuild render 11-17, `lick_matched_readout.py --ref gauss`, docked ITI |
| Goal Fig 14.18 | `.../iti_licks/bal_full_20261117/T/figures/lick_recovery_primary_full.png` | rebuild render 11-17, `lick_recovery_primary.py`, `WFIELD_BAL_FULL=1` (uncapped, as on the page) |
| Image 0.2 | `rebuild_figs/T/behaviour_hits_by_spout.png` | rebuild render (same script, recreated) |
| Image 2.3 | `labcams/grant_figures/T/epoch/epoch_acc_by_position_cue_working.png` | rebuild render 11-09 (post-glue) |
| Image 2.5 (1.0 s ENL, primary) | `labcams/grant_figures/LPfinal/trained_lp/LP_licks_gateLP_T/epoch/epoch_acc_by_position_precue_working.png` | rebuild render 11-08, BEFORE the 11-09 glue re-run (not re-rendered since) |
| Image 3.4 | `rebuild_figs/T/state_position_postglue_20261117/state_position_sessions.png` | re-rendered 11-17 by this helper from the post-glue sidecars (the 11-08 render predates them) |
| Image 1.4 | share `presentations/_source_PI_overview_2026-11-02/build_inputs/decoding_refit_sessions_campllick.png` | page image, NOT a rebuild render (none exists; input not regenerated) |
| Image 7.18b / 7.18c | `labcams/grant_figures/LPfinal/LP_licks_gateLP_T_refs-.../epoch/epoch_15r_position_{RESTNL,RAW}ref_lick_lick.png` | the page image (page shows a white-background re-save) |
| State-space 3.2 / 3.9 / 7.2 / 9.1d | `state_space_recovery/page_figs/ss_{change_vs_pre_lick,xform_lick,traj_days_lick,cmap_full_rest_frKBs_lick_svd}.png` | page image = rebuild render (byte-identical) |
| State-space 6.3 (Allen) | `state_space_recovery/page_figs_nullfix_20261116/ss_step_lick.png` | page image = rebuild render |
| State-space 11.3 | `labcams/grant_figures/LP_licks_gateLP_T_inmem-epoch/epoch/data/aligned_decoder/aligned_split.png` | page image = rebuild render |
| Kinematics 1.4 | `.../iti_licks/T/figures/lick_cued_aim_by_spout.png` | rebuild render (page copy renamed k_aim_spout.png) |
| Kinematics 1.1 | `LP_analysis/.../cohort/T/figures/prepost_working/cohort_metrics_minus_pre_contra.png` | rebuild render |
| Kinematics 6.1 | `session_poses_derived/T/jaw_only_incomplete_figs_redraw_20261116/figures/trial_class_share_raw.png` | rebuild render of the same classes for all six spouts (the page's 5-panel far-contra figure has no on-disk copy); far-contra values identical to the page |
| Kinematics 4.12 / 7.5 | `rebuild_figs/T/fig9c_softmax_landing_stats.png`, `session_poses_derived/T/kinematics_pooled_figs_20261116/pose3d_change_pooled_contra.png` | page image = rebuild render |
| Kinematics 10.1 | `rebuild_figs/T/pose_validation_sessions.png` | rebuild render |

Numbers asked for: image 2.3 far contra, rebuild: pre .879 -> acute .240 [.137, .375] (page .896 -> .321); acute change
-0.618 [-0.695, -0.531]. Image 3.4 (post-glue re-render): position retained .835 -> .488 acute (change -.347
[-.412, -.287]), state .972 -> .949 (-.023 [-.044, -.006]); page: position loses .41, state .03.

## 2. Section K (LP-gated) in the rebuild deck

`deck_registry.LP_GATED_REBUILD`: per slot, the rebuild file and a caption written from that render's tables (the
registry legends' WHAT IT SHOWS numbers are pre-rebuild and are not used in a tagged build). Placed: accuracy by
position lick window LP gate + its delta (image page Fig 2.4; `LP_licks_gateLP_T`), attempt maps lick / minus-pre /
cue (image page Figs 8.1-8.2; `attempt_maps/gateLP_T`, csig 3d7bbe921e), 15r / 15rpa REST-reference LP lick maps
(rendered 11-17, see 3). Not placed (no rebuild render): LP cue-window accuracy, LP 5c confusion, DAQ_ENL1s 1.0 s
pre-cue accuracy (the LP-trained 1.0 s figure is key figure 2.5).

## 3. 15r REST-reference LP lick maps

`map_collect` already skipped a never-filmed session (11-06, 72729126); the 11-16 failure ran code without it. This
branch names and counts the skipped sessions (`position_reference_maps.SKIPPED_NO_VIDEO`, log line and the 15r / 15rpa
captions: "Sessions skipped for having no camera / LP lick table ...: 1 (PS93_0813)"); test
`tests/test_lp_novideo_skip.py::test_skipped_sessions_are_named_and_counted_in_the_caption`. Rendered from this
worktree into `labcams/grant_figures/LP_licks_T/epoch/`.

## 4. Post-stroke methods note

`deck_text.M_POSTSTROKE_REBUILD` replaces `M_POSTSTROKE` in a tagged build (lesion dates and phases, epochs, the
31 / 16 / 17 / 43 sessions, camera trial gate and which rows each family scores, frozen references, behavior first with
the rebuild hit rates and trial classes). The old note (August, one post-stroke session, 44-session reference) stays
for an untagged build.

## 5. CD leave-one-out on the rebuild (image page Figs 9.3-9.5 family)

Run 11-17 from this worktree under the rebuild profile: `scripts/cd_loo_sessions.py --jobs 4` (fan over animals with
`analysis_kit.fan_sessions`), then `cd_loo_figures.py` and `cd_loo_pooled.py` (+ `--plot-only` for the `_nolines`
window figures). Timing: one hold-out fit (PS92, cue) 122-136 s cold; the full run took 2,012 s (33.5 min; 4 animals x
3 alignments x (1 post fit + 8 hold-outs)). Fixes needed first (this branch): `cd_loo_sessions._rows`,
`cd_loo_figures.fig_traj` and `cd_loo_pooled.fig_traj` still read `CD_WINDOWS` with a numeric pre-cue start; it is now
None (= the dump's -win_s). Outputs: `rebuild_20261103/labcams/grant_figures/T/epoch/{data/cd_window_projection_loo_
sessions_long.csv, data/cd_loo_traces.npz, cd_loo/}`. New frozen `cd_directions` hold-out models in the rebuild store.

Old (pre-rebuild, `labcams/grant_figures/epoch/cd_loo/cd_loo_pooled_stats.csv`, 10-25) vs new (rebuild), pooled change
vs own pre, pole units, 95% CI, * = CI excludes 0, k = animals below pre:

| align / position / epoch | old | new |
|---|---|---|
| cue far_center (far mid) acute | -0.474 [-0.76, -0.09] * (3/4) | -0.404 [-0.72, -0.05] * (3/4) |
| cue far_R (far contra) acute | -0.205 [-0.49, 0.07] (3/4) | -0.282 [-0.57, 0.03] (3/4) |
| cue far_R subacute | -0.145 [-0.26, -0.03] * (4/4) | -0.189 [-0.31, -0.07] * (4/4) |
| precue far_R acute | -0.805 [-1.69, 0.02] (3/4) | -0.687 [-1.26, -0.10] * (4/4) |
| precue far_R subacute | -0.275 [-0.45, -0.09] * (4/4) | -0.192 [-0.44, 0.17] (3/4) |
| precue far_R chronic | -0.180 [-0.37, 0.01] (4/4) | -0.214 [-0.40, -0.01] * (4/4) |
| precue far_center subacute | -0.116 [-0.75, 0.80] (3/4) | -0.536 [-1.00, -0.17] * (4/4) |
| precue far_center chronic | -0.514 [-0.73, -0.30] * (4/4) | -0.669 [-1.03, -0.40] * (4/4) |
| lick far_R acute | +0.386 [-0.31, 1.40] (2/4) | +0.392 [-0.22, 1.31] (2/4) |
| lick far_R subacute | -0.077 [-0.22, 0.05] (2/4) | -0.128 [-0.24, -0.03] * (4/4) |

Starred cells: 5 of 54 old, 10 of 54 new. Mark changes: precue far_L acute, precue far_center subacute, precue far_R
acute / chronic, cue close_R acute (+0.172), lick far_R subacute gain a star; precue far_R subacute loses it.
Held-out pre (mean over positions, PS92-95): pre-cue old .87 / .70 / .66 / .69, new .86 / .63 / .67 / .74.
The image page was NOT edited.
