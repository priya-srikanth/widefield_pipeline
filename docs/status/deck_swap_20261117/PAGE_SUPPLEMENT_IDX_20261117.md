# Goal page: collapsible supplement for the bounded balance indices (Priya 11-17)

Priya 11-17: "can you add idx figures as collapsible supplement with clear captions to the appropriate page (when able)".
Page = goal / encoding page (owner window 3c), next to Fig 14.13 (balance vs lick angle). Collapsed by default.
Source: docs/status/deck_swap_20261117/ITI_MATCHED_ANGLE_MARKERS.md section 5 (code e83b055e, test 867ce290).
Root: `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/`,
tag folder `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`.

## Supplement title
"Balance vs lick angle with four balance measures (baseline-corrected bounded indices)"

## Figure S-a: far contra only (target / previous spout fixed), four measures
File: `balangle_all_decidx_20261117/<tag>/figures/farcontra_balance_angle.png`

Caption: Far-contra spout only, so the target (cued) or the previous spout (ITI) is fixed. One open marker per session
(shape = animal, color = epoch); x = the session's mean lick angle at far contra (deg, + = contralesional); y = the
session's contra / ipsi balance, four measures in columns: resp = log2(contra / ipsi response AUC, baseline -1.0..-0.5 s;
cued 0..1.0 s, ITI 0..0.5 s); dec = right minus left mean dF/F 0..0.2 s after the lick peak, no baseline (the lateral
where-decoder's inputs); resp_idx and dec_idx = the bounded index (contra - ipsi) / (|contra| + |ipsi|) of the same
baseline-subtracted quantities (resp window, and 0..0.2 s). Lines = mixed-model fits y ~ angle x post + (1 | animal)
(black pre, dashed post); * = post-stroke shift at the same angle, p < .05. Cued: the shift at the same angle survives
in every baseline-corrected measure (resp +.24 log2, resp_idx +.079, dec_idx +.095; inside each animal's pre angle
range +.23 / +.076 / +.063; 4/4 animals); dec's +1.7 does not hold inside the pre range. ITI (all ITI licks after a
far-contra trial): dec's negative shift (-1.08) disappears in dec_idx (+.039). Licks: DAQ lick detector (daqlk) for lick times, lick angle from the 3-D pose (LP); trial
period: camera working period; trial types: cued = first post-cue lick with target far contra, ITI = bout start (>= 2 s
quiet) after a far-contra trial; gate: camera trial gate; baseline: August pre; statistics: MixedLM per measure, no
multiple-comparison correction across the four measures (descriptive comparison). PROVISIONAL (n = 4; 7-8 pre far-contra
sessions per animal).

## Figure S-b: far contra, docked ITI licks
File: `balangle_dock_decidx_20261117/<tag>/figures/farcontra_balance_angle.png`

Caption: As S-a, ITI licks restricted to the docked spout (onset between the spout reaching the dock and the next trial
start). dec shows a negative post-stroke shift at the same angle (-1.82, p .001; -1.43 inside the pre range) that does
not survive baseline correction: dec_idx +.031 (n.s.), resp_idx +.036 (n.s.), resp +.11 (n.s.) -- the dec shift reflects
pre-lick level / overall response size, not the evoked contra / ipsi balance. Per animal dec_idx post - pre +.13 / +.07 /
+.22 / -.05.

## Figure S-c / S-d: all spouts (session x spout), dec_idx and resp_idx
Files: `balangle_all_decidx_20261117/<tag>/figures/lick_balance_angle_dec_idx.png`, `..._resp_idx.png`
(docked twins in `balangle_dock_decidx_20261117/<tag>/figures/`)

Caption: Same layout as Fig 14.13 (one point per session x spout; cued = target spout, ITI = previous trial's spout),
balance = the bounded index. Pre-stroke slope negative in every measure and set (cued, all ITI, docked ITI; p < 1e-4,
4/4 animals): each hemisphere responds more to licks toward the opposite side. Cued: resp_idx shifts at the same angle
(-.035) like resp (-.101); dec_idx does not (.000). ITI: no same-angle shift in dec_idx / resp_idx on all ITI licks;
docked resp_idx +.048 (p .002), like resp +.158. Caveat: the previous spout explains 57 % (docked 49 %) of the ITI angle
variance across cells, so the far-contra versions (S-a, S-b) are the cleaner tests.

## Note for the page owner
resp and resp_idx agree in sign and significance everywhere (log ratio vs bounded index does not matter); the window
(0..0.2 s vs the response window) and the lick set do. For ITI bout starts with a 1-2 s gap, dec_idx's baseline window
can contain the previous lick's response.

Caption caveat line (add to S-a / S-b / S-c-d; ITI_MATCHED_ANGLE_MARKERS.md section 5b): "dec / dec_idx ITI bout starts
are split at >= 1 s gaps, so for 1-2 s gaps the -1.0..-0.5 s baseline can include the previous lick. Restricted to gaps
>= 2 s (resp's rule; 65 % of bout starts kept), the conclusions are unchanged. Docked far-contra dec -2.02 (p < .001)
vs dec_idx +.030 (n.s.); far-contra all-ITI dec_idx +.036 (p .02). Docked all-spout dec_idx becomes +.036 (p .002), in
line with resp_idx."
Gap >= 2 s figures: `balangle_{all,dock}_decidx_gap2_20261117/<tag>/figures/` (same file names).
