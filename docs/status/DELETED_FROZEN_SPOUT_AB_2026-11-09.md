# Deleted frozen spout decoder copies A and B (2026-11-09)

Priya approved 11-09 ~11:50 (RUNNING_LIST_2026-11-06 entry 11:50): copy D (`d_attempt_lp`, LP-trained) is THE single
LP frozen spout (position) decoder; copies A and B are deleted; copy C (DAQ lane) is kept. This file was written
BEFORE the deletion and lists every deleted folder.

Store: `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/frozen_models/<animal>/` (derived outputs only; no original
data). Each folder held `manifest.json` + `model.joblib` and nothing else.

## Checks done before deleting

- Each folder exists, holds exactly `manifest.json` + `model.joblib`, and its manifest matches the A / B spec below
  (kind decoder, source locanmf joint basis, lick source DAQ, align, window, pool size, freeze time).
- No reference to any of the 18 ids outside A's and B's own logs: grep of every session scratchpad under
  `%TEMP%/claude/C--Users-SabatiniLab` (*.log / *.txt / *.json / *.md / *.py; hits only in
  `render_daqlk/logs/d_attempt_freeze.log` (A, 12 ids), `render_daqlk/logs/g5_joint_xsession.log` (B, 6 ids) and the
  read-only trace script `frozen_spout_trace/cmp_models.py`); `git grep` of origin/main (2009bf45) = none;
  `figures_working` (json / md / txt / csv) = none; `RENDER_LOG.md`, `REBUILD_ROOT.json` = none;
  `LP_analysis` + `rebuild_figs` (json / md / txt / csv) = none.

## A = step `d_attempt_freeze` (11-05 18:58, code a76352f3, DAQ, `WIDEFIELD_ENL_WINDOW_S=1.0`), 12 models

Reason: DAQ-trained 1.0 s copy frozen for an LP reuse path that was superseded by the LP-trained copy D; read by
nothing.

| path | size | manifest |
|---|---|---|
| PS92/decoder_cue_joint76d884_8a8bb700498a | 264 KB | cue, post 2.0 s, DAQ, 8 pre (0806-0814), 27 pooled, camera working trials, frozen 11-06 00:29 |
| PS92/decoder_precue_joint76d884_af4bcd22a8c6 | 264 KB | precue, post 1.0 s, DAQ, 8 pre, 27 pooled, camera working, 11-06 00:22 |
| PS92/decoder_lick_joint76d884_e4028046485b | 504 KB | lick, post 2.0 s, DAQ, 8 pre, 27 pooled, engaged, 11-06 00:39 |
| PS93/decoder_cue_joint0141bb_c1ab284104fb | 216 KB | cue, post 2.0 s, DAQ, 7 pre, 26 pooled, camera working, 11-06 01:01 |
| PS93/decoder_precue_joint0141bb_ee333260f928 | 216 KB | precue, post 1.0 s, DAQ, 7 pre, 26 pooled, camera working, 11-06 00:56 |
| PS93/decoder_lick_joint0141bb_84182aacd6c2 | 408 KB | lick, post 2.0 s, DAQ, 7 pre, 26 pooled, engaged, 11-06 01:08 |
| PS94/decoder_cue_jointa5e70d_16919a45432e | 248 KB | cue, post 2.0 s, DAQ, 8 pre, 27 pooled, camera working, 11-06 01:33 |
| PS94/decoder_precue_jointa5e70d_8ff8605213b0 | 248 KB | precue, post 1.0 s, DAQ, 8 pre, 27 pooled, camera working, 11-06 01:20 |
| PS94/decoder_lick_jointa5e70d_69ed108cf50b | 472 KB | lick, post 2.0 s, DAQ, 8 pre, 27 pooled, engaged, 11-06 01:46 |
| PS95/decoder_cue_jointa91701_ef945446ce45 | 264 KB | cue, post 2.0 s, DAQ, 8 pre, 27 pooled, camera working, 11-06 02:09 |
| PS95/decoder_precue_jointa91701_45ebfa05f43c | 264 KB | precue, post 1.0 s, DAQ, 8 pre, 27 pooled, camera working, 11-06 02:03 |
| PS95/decoder_lick_jointa91701_ed8357a4e2bd | 504 KB | lick, post 2.0 s, DAQ, 8 pre, 27 pooled, engaged, 11-06 02:10 |

## B = step `g5_joint_xsession` (11-07 12:51, code a76352f3, DAQ, failed with KeyError 'PS93_0813'), 6 models

Reason: orphans of a failed run, re-done as `g5_joint_xsession_r` (C); read by nothing.

| path | size | manifest |
|---|---|---|
| PS93/decoder_cue_joint0141bb_80be25f6cbf2 | 216 KB | cue, post 2.0 s, DAQ, 7 pre, 8 pooled, camera working, 11-07 17:54 UTC |
| PS93/decoder_lick_joint0141bb_957936c98b82 | 464 KB | lick, post 2.0 s, DAQ, 8 pre, 8 pooled, engaged, 11-07 18:05 |
| PS92/decoder_precue_joint76d884_cee59cbf94fb | 264 KB | precue, post 2.0 s, DAQ, 8 pre, 8 pooled, camera working, 11-07 18:00 |
| PS93/decoder_precue_joint0141bb_aa9d66d73898 | 216 KB | precue, post 2.0 s, DAQ, 7 pre, 8 pooled, camera working, 11-07 18:00 |
| PS94/decoder_precue_jointa5e70d_2fe4405156fc | 248 KB | precue, post 2.0 s, DAQ, 8 pre, 8 pooled, camera working, 11-07 18:00 |
| PS95/decoder_precue_jointa91701_86595af5a294 | 264 KB | precue, post 2.0 s, DAQ, 8 pre, 8 pooled, camera working, 11-07 18:00 |

Total 18 folders, about 5.6 MB.

## Kept

- **C** = `g5_joint_xsession_r` (11-07, DAQ, pre-cue 2.0 s): the DAQ lane (LOSO slides, DAQ grant figs 2 / 2b / 4).
  PS92 84ad535cf3e7 / 1de01fbc75f4 / aae560a05805; PS93 714f188d3d89 / ad2b5c3cb999 / 5fa99f7f566f;
  PS94 535675b93dd7 / aff208370123 / ee3c3aa29915; PS95 74789c2f3a29 / 27ea0bab2232 / 039df402a201 (cue / precue / lick).
- **D** = `d_attempt_lp` (11-05/06, LP-trained, pre-cue 1.0 s, PS93_0813 not in the pool): THE LP frozen spout decoder
  (attempt coding; the LP joint cross-session step).
  PS92 a9addfabb37f / 4503e20e2ae1 / 2876156ef91a; PS93 22c3ae8371e8 / 7803a1ea6330 / 0e0719187108;
  PS94 573944df0b07 / 52202e05b8d8 / fb61dbd54bac; PS95 fab7df8581d6 / 90564a376277 / 14b3251439d9 (cue / precue / lick).
- Every other folder in the store (ROI decoders, encoders, attempt axes, CD directions, ...) is untouched, as is
  `frozen_models_lpfinal`.

## Done 11-09

- **Deleted 11-09 ~11:42**: the 18 folders above, each re-verified (contents, spec id, kind, align, window, DAQ, freeze
  date) immediately before `shutil.rmtree` on its literal path; none remains. Nothing else in the store was touched.
- **Gate source does not change D's training rows.** PS92 cue, attempt-coding pool (27 sessions), current code, render
  profile + `WIDEFIELD_LICK_SOURCE=lp`, with and without `WIDEFIELD_GATE_SOURCE=lp`: identical training arrays (2,863
  rows; X / y / session all `array_equal`), identical selection tag, spec id a9addfabb37f = D's both ways. The camera
  gate's working classes do not read the gate source, and frozen specs do not carry it.
- **LP joint cross-session run loads D (no fit).** Under LP the `joint_xsession` decoder spec equals D's by exact
  match (precue window 1.0 s comes from `make_spec` under LP; PS93_0813 drops out of the LP pool as in attempt coding,
  "no LP lick tables"). One change was needed: the LP lick unit now builds its pool with the no-lick reference at the
  would-be lick, as attempt coding does; with the cue reference the engaged rows differ by <= 1e-4 and the fingerprint
  missed D (PS93 lick 362a5d6c2e1c vs 0e0719187108). The run used `--decoder-only` (no LP joint encoder is stored; the
  grant figures read only the decoder JSON) and `WIDEFIELD_FROZEN_LOAD_ONLY=1` (a miss raises instead of fitting).
  Run 11:44-11:58 (14.5 min, -j 4) from detached worktree wf_render_jxlp @ 19c2325b; log
  `render_daqlk/logs/jx_lp.log`. All 12 units `frozen-hit`, `frozen_model_id` = the D ids above; outputs
  `figures_working/LP_licks_gateLP_<tag>/joint_xsession_decoder_{cue,precue,lick}__LP_licks_gateLP_<tag>.json` (the
  names `g6_grant_lp` reported MISSING). No new model in the store.

Pre-stroke LOSO accuracy (pre-stroke band; chance 1/6), LP-trained D, LP gate:

| align | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| pre-cue (1.0 s, working rows) | 0.341 | 0.393 | 0.588 | 0.438 |
| cue (0-2 s, working rows) | 0.855 | 0.734 | 0.935 | 0.923 |
| lick (0-2 s, engaged rows) | 0.930 | 0.797 | 0.943 | 0.928 |
