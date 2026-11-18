# Slide 21 — why the frozen pre-stroke POSITION readout changed (old render vs cohort-1 rebuild), one factor at a time

2026-11-17, approved by Priya ("yes go ahead"). VLS-stroke widefield, cohort 1 (PS92-95). Readout =
`scripts/rebuild_figs/fig_state_position_sessions.py` (position panel: frozen 6-way decoder, cue window 0-2 s, working
trials, fraction of above-chance performance retained; change vs the animal's OWN pre, `boot_delta` animals -> sessions,
Holm over the 6 tests, seed 20261102). Every row below is that script's own `load()` + `stats()` on one epoch/data folder
(`slide21_readout_table.py`; raw rows in `slide21_results.jsonl`).

## 1. Answer

**The trial gate explains most of it. A smaller share comes from how the joint-basis time courses were made for the
pre-stroke sessions (fitted vs projected), NOT from the hemo data itself. The pre set, spout scaling and lick detector do
not explain it.**

- Reverting ONLY the trial gate (camera gate -> the old DAQ quit-tail, working = lick + miss-while-working, frozen model
  trained on engaged rows) brings the post-stroke LEVELS back to the old render's: acute .454/.276/.476/.600 vs old
  .451/.288/.466/.607 (rebuild .499/.309/.515/.629). Change: -.397 / -.191 / -.112 (chronic Holm .006). Old:
  -.405 / -.192 / -.093. Rebuild: -.347 / -.139 / -.078.
- Reverting ONLY the hemo variant (meegkit_hpfit) lowers the acute and subacute levels: change -.367 / -.162 / -.077
  (chronic Holm .012), about 40 % of the acute gap. But under the old hemo the joint basis ALSO serves its in-fit
  (pre-stroke) sessions their FITTED C. E2 (old hemo, every session projected as in the rebuild) lands back on R0:
  -.338 / -.145 / -.066. So the E_hemo shift is the C METHOD, not the hemo data: the old frozen model was trained on
  fitted C (pre sessions in the basis fit) and scored post-stroke sessions on PROJECTED C, most likely a train / score
  mismatch that lowered post-stroke accuracy (inferred from E vs E2; not tested further). The rebuild projects every session (`project_all`, "NEVER MIX"), which removes it.
- The gate and joint-C-method shifts together (-.070 / -.075 / -.033 vs R0) are about the size of the whole old-minus-rebuild gap
  (-.058 / -.053 / -.016). The pre set (+.003 / +.026 / +.007) and session_z scaling (+.013 / +.006 / -.001) push slightly
  the other way. One-factor effects need not add exactly.
- So the camera gate RAISED post-stroke frozen accuracy, the opposite of the expectation recorded on 11-17 (non-contact
  lick trials were expected to lower it). One plausible reason, not tested here: the camera gate drops stopped / jaw-only /
  no-response trials that the DAQ quit-tail kept as "miss while working", and the frozen model is now trained on camera
  working rows (contact + incomplete) rather than engaged (lick) rows.
- The chronic star is fragile in every arm: Holm .0015 (old) / .006 (gate) / .012 (hemo, fitted C) / .050 (hemo, projected) / .029 (pre) / .035 (scaling) /
  .055-.058 (R0, lick). Its point estimate barely moves (-.071 to -.112); only the interval crosses the Holm line.

## 2. Factor list: old render vs rebuild (with evidence)

OLD = `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/data` (5r / 5rgap / acc sidecars 2026-10-07 10:38-13:24,
state 11-03). REBUILD = the 11-09 glue re-run `g5_epoch_g` (17:11-20:45, code 8aed3a7b, env in
`render_daqlk/logs/g5_epoch_g.log`). NOT the 11-05 g5_epoch: the slide-21 sidecars are dated 11-09 17:47-19:00.

| factor | old (10-07) | rebuild (11-09) | evidence |
|---|---|---|---|
| pre set | June 06-06..08 + August, 11 / animal (LOSO pre, frozen model on all 11) | August only (`WIDEFIELD_PRE_FROM=0801`), 8 / animal (PS93 7: 0813 has no video) | old 5r subtitle "pre 44 (11 each)"; profile; DECISIONS 11-03 |
| trial gate / classes | DAQ quit-tail; working = lick + miss-while-working; working-class frozen model trained on ENGAGED rows | camera gate: contact + incomplete, window cue+70 ms..min(cue+3.5 s, stop), end at last camera lick, stopA2j, excise; model trained on camera working rows | old titles; `sel-camera-...` tag; 48f2cc6d (11-04) |
| spout scaling | session_z (engaged-trial z-score) | session_bal_em, scored centre Wt | 99be6122 (11-03), DECISIONS 11-03 "bal_em adopted"; 724862f9 (centre options) |
| lick detector | old imaging literal 2.5 / 1.0 V, lockout 1-20 ms, 0.10 s refractory | behaviour DAQ detector 2.5 / 0.5 V, no refractory, 40 ms floor (`daqlk`) | cd35e594 (11-05, REBUILD_CODE 8.53) |
| hemo variant | meegkit_hpfit (untagged = `config.UNTAGGED_HEMO_VARIANT`) | meegkit_hpfit_lag_refitT_swapmask | profile; config.py |
| joint-basis C (what these decoders read) | in-fit (pre) sessions = their FITTED joint C, others projected | every session PROJECTED onto the joint footprints (`joint_locanmf.project_all`, forced by the hemo change: NEVER MIX) | 4813c3cf (11-02) |
| per-session LocaNMF | stored C | projection (`lnmfproj`) | profile. Not read by acc / 5r / 13s (they use JOINT-basis features, `grant_kit._pooled_bundle`); not run (F_lnmf skipped) |
| frozen store / code | legacy store, main ~660bcbe7 | fresh store `rebuild_20261103/frozen_models`, 8aed3a7b (glue mask: maps only) | |
| session set (post) | 16 / 17 / 43 | identical | sidecar counts |

## 3. Method and isolation

- Every variant = the 11-09 rebuild env verbatim with ONE factor reverted, code = `git archive 8aed3a7b` (the rebuild's
  code; `wfield_local.__file__` checked under the exact command form), `python -m wfield_local.epoch_grant_figures --arm
  cue --only acc 5r 13s --output <variant>/grant_figures/epoch`. Readout always from the wf_cleanup2 checkout (printed).
- Outputs, per-variant session cache and per-variant frozen store all under
  `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/decomp_slide21_20261117/` (`<v>/grant_figures/<tag>/epoch`,
  `<v>/<tag>/session_cache`, `frozen_models_<v>`). R0 / A_pre / D_scale caches were seeded with COPIES of 597 tf-cue
  entries from the rebuild cache (per-session features do not depend on the pre set or scaling); all other variants cold.
- A write jail (`sitecustomize.py` on PYTHONPATH, every process incl. pool workers) refused any share write outside the
  decomposition folder and capped the default worker count at 3. Its log has no refusals from the runs (only two
  deliberate test writes, and null-device opens that a first version wrongly refused; fixed before the variants). Runs
  one at a time, each after free RAM >= 8 GB. Nothing deleted. `decomp_slide21_20261117/jailtest.txt` is a leftover
  test file of the jail check (harmless; left, not deleted).
- Scripts: `docs/status/deck_swap_20261117/slide21_decomp/` (runner, chain, jail, cache seed / scan); logs per run in
  the same folder (`logs/`).

## 4. Per-variant table (position readout; levels = fraction retained, mean over sessions per animal PS92/93/94/95)

| run | reverted factor | pre n | pre | acute | subacute | chronic | change acute | change subacute | change chronic (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| OLD sidecars | (all) | 11/11/11/11 | .850/.744/.923/.915 | .451/.288/.466/.607 | .674/.527/.720/.742 | .826/.680/.726/.827 | -.405 ** | -.192 ** | -.093 (.0015) |
| OLD, June dropped from readout only | (sidecar check) | 8/8/8/8 | .851/.725/.934/.917 | same | same | same | -.404 ** | -.191 ** | -.092 (.012) |
| REBUILD sidecars | — | 8/7/8/8 | .829/.681/.922/.908 | .499/.309/.515/.629 | .680/.568/.730/.807 | .822/.615/.738/.857 | -.347 ** | -.139 ** | -.077 (.057) |
| **R0 control** | none | 8/7/8/8 | .829/.681/.922/.909 | .499/.309/.515/.629 | .678/.568/.731/.807 | .821/.615/.738/.857 | -.347 ** | -.139 ** | -.078 (.055) |
| A_pre | June + August pre (frozen model retrained on 11) | 11/10/11/11 | .851/.715/.921/.910 | .515/.320/.535/.652 | .739/.605/.771/.829 | .834/.661/.751/.868 | -.344 ** | -.113 ** | -.071 (.029) |
| D_scale | session_z scaling | 8/7/8/8 | .833/.676/.921/.908 | .501/.313/.515/.670 | .685/.576/.740/.806 | .817/.610/.738/.854 | -.334 ** | -.133 ** | -.079 (.035) |
| **B_gate** | DAQ quit-tail gate | 8/8/8/8 | .837/.712/.922/.922 | .454/.276/.476/.600 | .619/.547/.716/.748 | .814/.588/.707/.839 | **-.397 **** | **-.191 **** | **-.112 (.006)** |
| C_lick | old imaging lick detector | 8/7/8/8 | .828/.681/.923/.909 | .499/.308/.513/.629 | .680/.569/.730/.806 | .821/.615/.738/.857 | -.348 ** | -.139 ** | -.078 (.058) |
| **E_hemo** | meegkit_hpfit (+ fitted joint C for in-fit sessions) | 8/7/8/8 | .829/.674/.912/.908 | .495/.298/.484/.580 | .656/.509/.724/.787 | .803/.637/.753/.824 | **-.367 **** | **-.162 **** | -.077 (.012) |
| E2_hemo_projall | meegkit_hpfit, ALL sessions projected (method as rebuild) | 8/7/8/8 | .828/.672/.916/.898 | .479/.298/.515/.669 | .687/.508/.732/.805 | .827/.630/.748/.847 | -.338 ** | -.145 ** | -.066 (.050) |

** = Holm < .01 (all acute / subacute rows are Holm .0015).

**R0 noise floor.** R0 reran the rebuild exactly (same frozen spec ids, e.g. PS92 133c81fb55c9): 29 of 480 per-session
x position recall cells differ from the rebuild sidecar by ~one trial (max |d| .020), and the readout moves by <= .002
(chronic Holm .057 -> .055). Shifts below ~.005 in a change, or Holm wobble around .05, are within this floor.

## 5. Cost table (measured on this box, one variant at a time, <= 3 workers)

| variant | caches rebuilt | wall time |
|---|---|---|
| R0 | frozen models + 5r refits (features seeded) | 44 min |
| A_pre | + June frozen training / LOSO (features seeded) | 51 min |
| D_scale | frozen models, refits (features seeded) | 31 min |
| B_gate | all features cold (new selection kind), frozen, refits | 52 min |
| C_lick | all cold | 42 min |
| E_hemo | all cold, old-hemo projection onto the joint basis | 54 min |
| E2_hemo_projall | all cold | 69 min (box busier) |
| F_lnmf (skipped) | cold; per-session C is not read by these families, expected = R0 | ~45 min |
| per-session LocaNMF REFIT on the new hemo data | ~120 GPU refits at 51-654 s (LOCANMF_REFIT_TEST_20261103) ~= 11-12 h, not reproducible run to run | not run |

## 6. Plain-language attribution

The old slide had a bigger post-stroke drop mainly because of WHICH TRIALS were scored and trained on. The old DAQ
quit-tail rule counted every trial up to the quit point as "working", including trials the camera gate now calls stopped,
jaw-only or no-response, and the frozen model was trained on licking trials. With the camera gate the scored post-stroke
trials are ones where the animal actually attempted (contact or incomplete lick), and those are decoded better, so acute /
subacute / chronic levels rise and the drop shrinks (-.405 -> -.347 acute). A smaller part (about -.02 acute and subacute) came from the old
render training the frozen decoder on joint-basis FITTED time courses for the pre-stroke sessions while scoring the
post-stroke sessions on PROJECTED ones; the new hemodynamic correction itself changes nothing (E2 = R0), and the
rebuild projects every session, so that mismatch is gone. The August-only baseline, the bal_em scaling and the new lick detector
change the answer by < .03 and mostly in the other direction. The chronic star was never robust: across all arms its
estimate stays -.07 to -.11 while Holm moves between .0015 and .058.
