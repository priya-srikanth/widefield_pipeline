# Encoder-family render commands for the cohort-1 rebuild (11-04)

These are the production commands of the LIVE goal / encoding pages, from the handoffs and the session scratchpads
(S1 = session cdd1e016, S2 = session f082ee0c). They are adapted to the rebuild profile. The scratchpad scripts are backed up at
`M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/scratchpad_scripts_backup_20261104/` (S1 = the 10-21..10-26 session).

## Every command

- cwd and `PYTHONPATH` = `C:/Users/SabatiniLab/Github/wf_rebuild_render` (add `;<that>/scripts` for scripts that import siblings).
- Apply the profile (`render_profile cohort1_rebuild_20261103 --shell bash [--block ...]`, eval, `--check` exit 0), THEN set the
  extra env below. The profile clears every `WIDEFIELD_*` / `WFIELD_*` / `COMBINED_ENCODING_*` variable, so the extra env has to be
  set after it.
- `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache`. Re-set it after the profile. Under the rebuild code it is keyed on the
  source path, so it is safe to share.
- Never `--reuse-stats` from an adopted dir. `--reuse-stats` is allowed only to re-score a frame this render fitted, inside the
  root.
- Keep the page's tag names (`gate-attempted_v7g3_t3d...`) so the figure scripts' defaults resolve. Under the profile the frames
  land in `imaging_encoding/<run tag>/frozen_encoder_<tag>`.

### ENV-noFR (the noFR frame and everything that reads it)
```
export OROFACIAL_GATE=attempted COMBINED_ENCODING_LICKS=combined COMBINED_ENCODING_ONLY=full COMBINED_ENCODING_SUFFIX=_gate-attempted
export COMBINED_ENCODING_TONGUE=3d COMBINED_ENCODING_EXEC3D_TABLE=C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/pose3d_kin_licks_snapshot_20261023.csv.gz
unset FROZEN_RUNNING FROZEN_RUN_KNOT0 FROZEN_PUSH FROZEN_FR FROZEN_FR_SPLIT FROZEN_LICKSPOUT FROZEN_LICKLAND FROZEN_FR_LAGS FROZEN_PIX_MUSALL
```
### ENV-frA = ENV-noFR plus
```
export FROZEN_FR=1 COMBINED_ENCODING_FR=master_pre_v1 COMBINED_ENCODING_FR_GROUPS=master_v1:A
```
### ENV-frozen (frozenenv.sh) = ENV-frA plus `FROZEN_LICKSPOUT=1`

Use ENV-frozen for the ITI extraction and its consumers, the lick chain, within_lick_correction, learned_compensation,
prelaunch_aim_tracking, face_prelaunch_aim and epoch_kernels. Without it, `bout_fr` / `first_fr` come out NaN with no warning.

## A. Frozen encoder frames (local; in this order)

1. noFR frame (ENV-noFR):
   `python -m wfield_local.frozen_encoder --all --jobs 2 --tag gate-attempted_v7g3_t3d`
2. noFR downstream (ENV-noFR):
   - `for al in cue lick; do python scripts/lp_encoder_decompose.py --folder <root enc>/frozen_encoder_gate-attempted_v7g3_t3d/joint --align $al; done`
   - `python -m wfield_local.frozen_relabel --from gate-attempted_v7g3_t3d --tag v7g3_t3d --jobs 2`
   - `python -m wfield_local.allsession_encoder --from gate-attempted_v7g3_t3d --tag v7g3_t3d_ref --jobs 2`
     (Layer 2 for Figs 19.x; the page used the `_ref` tag. This command is inferred: no launcher was found.)
   - `python -m wfield_local.frozen_encoder --all --jobs 2 --target basis --tag gate-attempted_v7g3_t3d_basis`
   - `python -m wfield_local.frozen_relabel --from gate-attempted_v7g3_t3d --tag v7g3_t3d_trial --jobs 4`, then
     `python scripts/relabel_confusion.py --tag v7g3_t3d_trial --trial --window lick` and `--window cue`
3. frA frame (ENV-frA):
   - `python -m wfield_local.frozen_encoder --all --jobs 3 --tag gate-attempted_v7g3_t3d_frA`
   - its decompositions (cue / lick), as in step 2
   - `python -m wfield_local.allsession_encoder --from gate-attempted_v7g3_t3d_frA --tag v7g3_t3d_frA --jobs 3`
   - `python -m wfield_local.allsession_encoder --from gate-attempted_v7g3_t3d_frA --tag v7g3_t3d_frA_roi --roi-targets --post-from v7g3_t3d_frA --jobs 3`
   - `python -m wfield_local.frozen_relabel --from gate-attempted_v7g3_t3d_frA --tag v7g3_t3d_frA --jobs 3`
4. frA variant frames (ENV-frA unless noted):
   - `--all --jobs 3 --target basis --no-drop-spout --tag gate-attempted_v7g3_t3d_frA_basis --variants joint no_target no_spout no_task no_movement no_fr no_fr_dir no_fr_all no_lick no_exec_within no_other`
   - `--all --jobs 3 --tag gate-attempted_v7g3_t3d_frA_pos --reuse-stats --no-drop-spout --variants joint no_posinfo no_noposinfo posinfo_only noposinfo_only`
     (`--reuse-stats` reads the frA frame fitted in step 3, inside the root)
   - with `FROZEN_FR_SPLIT=1`: `--all --jobs 3 --tag gate-attempted_v7g3_t3d_frAsplit --no-drop-spout`, and
     `--all --jobs 3 --target basis --no-drop-spout --tag gate-attempted_v7g3_t3d_frAsplit_basis --variants joint no_posinfo no_noposinfo no_target no_spout no_fr_all no_fr_dir`
   - with `FROZEN_LICKSPOUT=1` (ENV-frozen): `--tag gate-attempted_v7g3_t3d_frA_ls --variants $GROUPV $NOVAR` and
     `FROZEN_PIX_MUSALL=1 ... --target basis --tag gate-attempted_v7g3_t3d_frA_ls_basis --variants $GROUPV $NOVAR`
     (GROUPV / NOVAR are verbatim in backup `S1_cdd1e016/run_lickspout.sh`); then
     `python scripts/session_vs_frozen.py --tag gate-attempted_v7g3_t3d_frA_ls --jobs 2` and
     `python scripts/partial_refit.py --tag gate-attempted_v7g3_t3d_frA_ls --jobs 2`
   - with `FROZEN_DESIGN=te` (LICKSPOUT unset): `--all --jobs 3 --no-drop-spout --tag gate-attempted_v7g3_t3d_frA_te`, and
     `FROZEN_PIX_MUSALL=1 ... --jobs 2 --no-drop-spout --target basis --tag gate-attempted_v7g3_t3d_frA_te_basis --variants joint target_first exec_first no_timing no_target no_execution no_contact no_other no_task no_movement task_only target_only execution_only other_only movement_only`;
     then `session_vs_frozen.py --tag gate-attempted_v7g3_t3d_frA_te --jobs 3 --variants joint no_target no_execution no_contact no_task no_movement`
5. pixmusall twins (ENV-frozen with `unset FROZEN_LICKSPOUT`; `V="joint no_target no_exec_locked no_exec_within no_spout no_movement"`):
   - noFR (also unset FROZEN_FR / COMBINED_ENCODING_FR / _FR_GROUPS): `FROZEN_PIX_MUSALL=1 python -m wfield_local.frozen_encoder --all --target basis --tag gate-attempted_v7g3_t3d_basis_pixmusall --fit-from <root enc>/frozen_encoder_gate-attempted_v7g3_t3d_basis --alphas-from <root enc>/frozen_encoder_gate-attempted_v7g3_t3d --variants $V --no-drop-spout --jobs 4`
   - frA: the same with `--tag ..._frA_basis_pixmusall --fit-from .../frozen_encoder_gate-attempted_v7g3_t3d_frA_basis --alphas-from .../frozen_encoder_gate-attempted_v7g3_t3d_frA --variants $V no_fr_all`
   - The `--fit-from` / `--alphas-from` paths point INSIDE the root (this render's frames).

SUPERSEDED, do not render: v7pushfree, v7fwd15, v7basis, v7g3 (cam4 tongue), _frA_pushfree, allsession_encoder_v7push.

## B. Encoder figure scripts (after A)

These are the backup's `rerender_stars.sh` set plus the `mu/run.sh` items:
- `frozen_fr_compare.py`, `frozen_fr_musall.py`, `frozen_fr_musall_v2.py`
- `frozen_posinfo.py --tag gate-attempted_v7g3_t3d_frAsplit`
- `specificity_control.py --tag gate-attempted_v7g3_t3d_frA --layer2 allsession_encoder_v7g3_t3d_frA_roi`
- `session_vs_frozen.py --tag gate-attempted_v7g3_t3d_frA` (full run, then `--plot-only`)
- `matched_residual.py --tag gate-attempted_v7g3_t3d_frA`
- `frozen_pixel_unique_sig.py`, three runs: `--tag gate-attempted_v7g3_t3d_basis [--zscore]`, `--tag ..._frA_basis`, `--tag ..._frAsplit_basis`
- `frozen_pixel_musall.py --tag gate-attempted_v7g3_t3d_frA_ls_basis --layout {novar,alone} --window {cue,lick}`
- `spout_unique_maps.py --tag gate-attempted_v7g3_t3d_frA_ls_basis`
- `frozen_residual_significance.py --spout <each spout>` and `--spout all [--per-animal]`
- `residual_roi.py --tag gate-attempted_v7g3_t3d [--by-spout]` and `--tag gate-attempted_v7g3_t3d_frA`
- `partial_refit.py --tag gate-attempted_v7g3_t3d_frA` and `--tag gate-attempted_v7g3_t3d_frA_te --jobs 6`
- `epoch_kernels.py` (ENV-frozen)
- `te_readout.py`, `contest_lines.py`, `remap_lmm.py`, `remap_checks.py`, `movement_null_plan_maps.py [--frames FR-A noFR]`, `frozen_peth_trial_maps.py`, `frozen_kernel_maps.py`
- `frozen_pixel_maps.py --tag gate-attempted_v7g3_t3d_basis --variant {joint,target_first,exec_first}`, `frozen_r2_maps.py --tag gate-attempted_v7g3_t3d_basis`, `frozen_spout_maps.py --tag gate-attempted_v7g3_t3d_basis [--contrast]`
- `frozen_claims.py` (last)
- `exec_vs_neural_match.py` with each of: `--source decoder`, `--target far_contra`, `--target far_mid`, `--source encoder` (the last after the v7g3_t3d_trial relabel; needs attempt_coding_pp, step D)
- `encoding_key_figures.py`
- Layer 2:
  - `layer2_matched_n.py`
  - `layer2_random_splits.py <animal> --max-splits 20`, once each for PS92 / PS93 / PS94 / PS95 (20 is inferred from the existing CSVs)
  - The Layer-2 figures came from scratch `fig_layer2.py`, `fig_rsplit.py`, `fig_pooled.py` (backup S1). They are NOT in the repo; LOG this as a gap and don't run the scratch copies.

## C. O2-dependent (DEFER; decision for Priya)

The per-session `combined_encoding` runs used by Figs 1.13-1.15, 21.x, 21.12, 22.1 and the fr_* scripts come from the O2 bundle
`enc_v36_full_20261024` (adopted imaging inputs). A rebuild needs a NEW bundle built with the rebuild code and data, plus an O2
submission:
- 14 TAG arrays; the per-TAG settings are in backup `S1_cdd1e016/run_bundle_full.sh` and the bundle's submit_all.sh;
- then collect.

Not done unattended. That leaves fr_encoding_summary, fr_musall, fr_direction, fr_trial_peth, fr_pixel_significance,
variance_maps, fr_direction_mechanism, encoding_mixed (old bundles, share-only) and the binned contest (1.13-1.15) waiting on it.

## D. attempt_coding / attempt_maps (after trial_attempts)

1. `WIDEFIELD_ENL_WINDOW_S=1.0 python -m wfield_local.attempt_coding --freeze` (DAQ path, LICK_SOURCE unset)
2. `WIDEFIELD_LICK_SOURCE=lp python -m wfield_local.attempt_coding`
3. `WIDEFIELD_LICK_SOURCE=lp python -m wfield_local.attempt_coding --aligns lick cue --out <imaging_encoding>/attempt_coding_pp`,
   then `--aligns precue --out` the same folder
4. `WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_GATE_SOURCE=lp python -m wfield_local.attempt_maps --rest-variant restnl --jobs 3` and
   `--rest-variant rest`, then `--plot-only`. PRE_FROM comes from the profile.

## E. ITI family and goal-page lick chain (ENV-frozen; after the frA frame of A3)

- Extract: `python scripts/iti_licks.py --jobs 3`. Its labels come from the frA frame's sessions, so it depends on A3.
- Consumers: waves 1-3 and the docked arm, as listed in backup `S2_f082ee0c/run_v8redo.sh`, `run_v8wave2b.sh`, `run_v8wave3.sh`,
  `run_dock.sh` and `fr/run_main.sh` (ITI FR controls), in the "Extract -> consumer order" of rebuild_order_B.md steps 1-15.
- Then within_lick_correction `run` / `summarise` / `plot` -> learned_compensation `run` / `summarise` / `plot` ->
  prelaunch_aim_tracking `extract` / `fit` / `summarise` -> face_prelaunch_aim `extract` / `fit` / `summarise` / `offset`.

## F. misdirected licks (after trial_attempts)

- `python scripts/misdirected_lick_counts.py`
- `python scripts/misdirected_lick_decode.py`

Both use their defaults (`--run lp_r4_heatmap_20261013__clean_v36 --dev 15`; decode adds `--name enc_v36_20261019 --min-probe 15 --method templates`).
The decode's `--name` is an O2 bundle of adopted inputs, so if it needs that bundle's per-session encoder: DEFER (C).
