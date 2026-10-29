## 2026-10-28 (late night) -- balance definition for recovery markers (Priya)
- Recovery-marker balance is measured on the FIRST POST-CUE LICK (cued) and the FIRST ITI LICK AFTER >= 2 s QUIET (ITI),
  not on all licks (Priya: "we'll use first iti lick after quiet and first post cue lick based windows").
- Cued window CONFIRMED on held-out animals (scripts/lick_balance_nested.py; window chosen on 3 animals, tested on the
  4th): first-post-cue-lick gain ratio 0.2..0.35 s, chosen in 4/4 folds, nested LOAO 0.69 (null p 0.005; days 0.23).
  ITI: no window survives the held-out selection on its own (timing families <= 0.11); 'any definition' 0.22 vs days
  0.17 (p 0.01). Area subsets: scripts/lick_balance_subsets.py (running at this note).
- Leave-one-EPOCH-out dropped as a criterion (Priya: tests extrapolation to an unseen aim range, 3 unequal folds);
  replaced by within-animal BLOCKED CV (4 contiguous blocks per animal, 1-session gap, animal intercepts;
  lick_recovery_markers.cv_blocked).

## 2026-10-28 (night) -- recovery markers, Shapley grouping, pre-cue reference (Priya)
- Pre-cue LEVEL is referenced to RESTW (Priya "3. yes"): position-weighted flat rest median, restdock05, engagement-gated,
  docked=False as the other restw consumers (scripts/lick_precue_restw.py; 0/107 fallbacks). Standalone figure 14.20,
  not a row of the AUC maps (a level, different units).
- Shapley grouping: SSs FOLDED INTO the orofacial group for the main 11.14 image (Priya; SSs unique share 0.000 4/4);
  6-group split kept as a supplement (WFIELD_BLOCK_VARIANT unset). The Allen atlas has no MO trunk area -> body = SSp-ul,
  -ll, -tr.
- Recovery-marker cross-validation is now a committed script (scripts/lick_recovery_markers.py; reproduces the earlier
  ad-hoc 14.16 numbers exactly): LOAO (with within-animal shuffle null), leave-one-EPOCH-out, LOSO animal unknown / known.
  Pre sessions scored leave-one-pre-session-out for the grey reference cloud (14.16, 14.17). Fair pre-fit weights =
  fit on the OTHER animals' pre licks (z-scored balances).
- Residual-ROI maps: per-area outlines + fading; atlas areas with no value (FRP, AUDd / AUDpo right, VISp left) blanked
  in the display (they were the 'edge fragments').
- Non-linear decoder versions added as collapsible supplements (12.1, 12.15, 12.21, 13.1, 13.6).


## 2026-10-28 -- Key figure and cued wording (Priya)
- KEY FIGURE replaced (Priya: "yes to your A and B; and use the newer behavior recovery metric"): Key figure 1 =
  scripts/key_figure.py (A behaviour = far-contra FIRST-lick aim; B/C frozen vs epoch-refit lateral decoders, + the old
  far-contra code on hits; D ITI like-for-like offset, any-spout and same-target). Key figure 2 = Fig 13.2 (lick order;
  Priya "another key figure"). The old key figure stays as Fig 0.6.
- CUED "read ahead of the failing movement" RETIRED (Priya "ok to reword"): the cued like-for-like offset is target
  composition (same-target same-place +0.01 n.s.; first cued lick +0.05 n.s.); page status rows, 12.x, 13.4 title and
  FINDINGS 4d0-V8 / 4d0-NIGHT re-worded. The ITI offset is the changed read-out.
- 'Where things stand' bottom line rewritten 10-28 (10-25 version: FINDINGS section 0).
- ALM / tjM1 encoder rows keep the STAT MASK (rim check: no difference, scripts/checks/rim_check_alm_tjm1.py).
- Like-for-like curves (12.2) Gaussian-smoothed with the offset's sigma, no display bins; SEM over ANIMALS.
