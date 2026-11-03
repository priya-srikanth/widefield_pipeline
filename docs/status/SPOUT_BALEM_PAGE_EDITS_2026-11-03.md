# Spout-position decoder on bal_em: results-page edits (2026-11-03)

Priya 11-03: "I would adopt B everywhere". The frozen spout-position decoder is now scored with `session_bal_em` at
precue, cue and lick (DECISIONS 2026-11-03, "Spout-position frozen decoder: bal_em adopted as PRODUCTION"). This doc
lists the page figures and numbers that move. **Nothing here is published yet.** Rewrite captions in place; do not add
CORRECTION notes (the change record is DECISIONS / this doc).

Status: code, frozen models and verification DONE; the family RE-RENDERS are held until A3 (the LMM epoch-star rule,
another window) lands on origin/main. Numbers marked *(render)* are filled from the re-rendered sidecars; the numbers
given here are the verification run (cached production features, reproduced end to end through `_collect_5c`:
`labcams/decoder_mix_checks/spout_balem_production/summarise_log.txt`).

## 1. Method line for every affected caption / methods box (replace the normalisation sentence)

> Frozen pre-stroke decoder; each pre-stroke training session centred on the equal-weight mean of its six true spout
> means; each scored session centred, without labels, on the equal-weight mean of EM-estimated (Saerens et al. 2002)
> soft spout means of its engaged trials, engaged-trial SD (bal_em).

Limit to state once on the page (decoding section intro): when a spout is nearly absent from the engaged set (acute far
contra) ~30-40 % of the class-mix bias remains, in the conservative direction; at precue only ~37 % is removed.

## 2. Families to re-render (they read this decoder)

| family | module | arms |
|---|---|---|
| `acc` (acc_by_position) | `epoch_figures._per_position_accuracy` <- `_collect_5c` | DAQ epoch, LP_licks, LPfinal (all trained / gate variants) |
| `5c` | `_confusion_rows` <- `_collect_5c` | same |
| `5r`, `5rm`, `5ro`, `5rmo` (frozen vs refit; refit side unchanged, frozen side and gap move) | `epoch_accuracy` <- `_collect_5c(paired / paired_matched)` | same |
| `13s` (state vs position; POSITION panel moves, state unchanged) | `epoch_state` <- `_collect_5c` | DAQ epoch (+ LP if rendered) |
| grant `5c` (per-session confusions), `5b` (recompute path only) | `grant_confusion` | grant_figures |
| recovery trajectory | `recovery_trajectory` <- `_collect_5c(paired)` | |
| section D frozen JSON / figures | `nightly_figs` frozen step (`pooled_frozen_loso`, ROI) + `joint_xsession` | `--skip-frozen` off |
| section G | `poststroke_section_g` | |
| attempt_coding (p_true, softmax) and its readers | `attempt_coding` (LP run: freeze first with `WIDEFIELD_ENL_WINDOW_S=1.0 ... --freeze`); `scripts/decoder_retention_sessions.py`, `coupling_by_class.py`, `early_predictors.py`, `exec_vs_neural_match.py`, `precue_class_matched.py` | LP_licks |
| scripts | `scripts/prelim_numbers_frozen_vs_refit.py`, `scripts/s1_refit_locanmf_table.py` (refit column unchanged) | |

NOT affected (production frame, not this decoder): `5cr` (refit confusions), 6-12b pattern / crossnobis / split-half /
best-match families, 14 / 15 maps, the state decoder (13s state panel), lick-landing decoders.

## 3. Imaging page: figures and numbers that move

| page fig | what | old (session_z) -> new (bal_em), acute vs pre unless noted |
|---|---|---|
| 2.3 `d_acc_cue` (acc by position, cue/working) | per-position change | far contra -.536 -> -.644; far mid -.385 -> -.484; close ipsi -.191 -> -.130; close mid -.250 -> -.196; close contra -.255 -> -.202; far ipsi -.275 -> -.265; overall -.314 -> -.318. Chronic far contra -.130 -> -.179; chronic overall -.055 -> -.071 |
| 2.x acc by position, precue/working | | far contra -.254 -> -.271; far mid -.121 -> -.162; close +.03-.04 smaller losses; overall -.177 -> -.171 |
| 2.x acc by position, lick/lick | | far contra -.465 -> -.483 (3 animals); close contra -.191 -> -.131; close ipsi -.168 -> -.121; overall -.192 -> -.165; chronic far contra -.207 -> -.238 |
| 2.8 `d_lpf_enl1_lp` (LPfinal trained_lp, precue) | | *(render)* -- expect the DAQ precue pattern above (far contra slightly larger loss, close smaller) |
| 3.1 / 3.4 position retained (state vs position) | position panel | *(render)* -- per-session mean of six recalls moves with the table above; state panel unchanged |
| 4.2 `d_refitgap`, 13.1 `m_dec_time` | frozen side of frozen - refit | *(render)*; refit unchanged |
| 7d / 7e (decoder retention on hits / far-contra code) | attempt_coding p_true | *(render)* |

Text that moves:
- "Frozen decoder loses far contra (.90 -> .32)" style statements: the acute far-contra loss is ~.11 LARGER (cue/working,
  4/4) and the close-position losses ~.05-.06 SMALLER; overall accuracy within .03. Fill exact levels from the render.
- Any sentence that says the near positions also fall substantially acutely: the fall is smaller (close ipsi -.13, CI
  now includes 0; close mid -.20; close contra -.20). The ordering (far contra falls most) is unchanged and more specific.
- Pre-stroke bars / bands: unchanged to within .003 (cue), .005 (lick), .01 (precue) pooled.

## 4. Encoding page / FINDINGS cross-references

FINDINGS 2 (acute decoder loss) and 3.x position-vs-state ratios are updated in `docs/FINDINGS.md` with the verification
numbers; the page captions follow once the renders exist.
