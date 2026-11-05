# Render queue for `_daqlk` (cohort-1 rebuild, merged main), 11-05

**UPDATE 11-05 evening (Priya's decisions).**
- She accepted the order G > A, B > D > E > F, Z, and asked to RENDER the te and lsland frames.
- 9aaa4cc6 is on main as a6edab62 (+ e53f1cd2, the fig10 title).
- Lane G is RUNNING from `queue_daqlk.sh` (SECTIONS=G, started 16:38). That file is left untouched.
- **Lane 2 launches from the NEW file `queue_daqlk2.sh`** in the same folder. It is identical except for a new
  section **T** (below) and the default SECTIONS. Command:

  `cd <scratchpad>/render_daqlk && SECTIONS="A B T D E F Z" nohup bash queue_daqlk2.sh > logs/laneENC.out 2>&1 &`

  Both lanes share `logs/`, so lane 2's cross-lane dependencies on lane G's `.done` files work.

**Code.** Worktree `C:/Users/SabatiniLab/Github/wf_render_daqlk`:
- origin/main **136807cf** (the rebuild -> main merge). origin/main has since moved to 40fbe48b, a docs-only commit.
- One local commit on top of it, on branch `fix/misdirected-decode-bundle-dir-20261105`: **9aaa4cc6**. It is not
  pushed (see "Code fix").
- The queue accepts either SHA. `render_profile cohort1_rebuild_20261103 --check` exits 0, imports wfield_local from
  this worktree, and gives the expected run tag.

**Script.** Queue: `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/render_daqlk/queue_daqlk.sh`.
In the same folder:
- `envrun.sh`: runs one command under a step's exact env;
- `dryrun_check.py`: the dry-run checker;
- `logs/`: chain log, per-step logs, PIDS, FAILED / SKIPPED, `dryrun_steps.tsv`, `dryrun_check.json`.

**Tag folder.** `ENC` = `<rebuild root>/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`.
`F` = `gate-attempted_v7g3_t3d_spont`, the primary frame stem. FROZEN_LICKS_ALL=split comes from the profile, so every
frozen reader resolves to the spont frames. The queue passes the spont tags explicitly anyway.

## How it runs
- **Preflight.** Checks the worktree SHA, then `render_profile --check`. It aborts if the check fails, the run tag
  differs, or wfield_local is not imported from the worktree.
- **Per step:**
  - a fresh subshell with the profile (`--block` where needed), then `PYTHONPATH=<wt>;<wt>/scripts` and
    `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104` (the frozen frames' cache; keyed
    on the source path);
  - then the env block: base / ENV-noFR / ENV-frA / ENV-frozen, as in ENCODER_RENDER_COMMANDS, plus FROZEN_LICKS_ALL=split;
  - its own `--check`, a log header with the env, and START / EXIT plus free RAM in `logs/CHAIN.log`;
  - its msys and Windows PIDs go to `logs/PIDS.txt`.
- **RAM.** Before each step the queue waits until free RAM is >= 10 GB. A watchdog checks every 60 s. If free RAM is
  below 6 GB twice in a row, it writes the STOP file and kills nothing.
- **STOP.** A `STOP` file in the queue folder halts the queue between steps.
- **Resume.** Steps with `logs/<name>.done` are skipped, so relaunching continues where the queue stopped.
- **Failure:**
  - mode `stop`: the queue halts. Used for producers that later steps read.
  - mode `cont`: the failure is recorded and the queue continues. Used for leaf figures and independent families.
  - In both cases every step that lists the failed step in DEPS is skipped and logged.
- **Workers.** Steps run one at a time per lane, with `--jobs` <= 4 for G and <= 3 for the encoder families. Worker
  counts the scripts hard-coded (the 8 / 6 in the old wave scripts) were lowered to 3-4 with their own flags.
  `epoch_grant_figures --parallel-arms 2`, `section_g -j 4` and `grant_figures -j 4` keep their documented values; their
  internal pools were not measured.

**Launch (decision for Priya / coordinator).** Running everything serially takes about 36-40 h (estimate below), so
two lanes are better:
- Lane 1 now: `SECTIONS="G" nohup bash queue_daqlk.sh > logs/laneG.out 2>&1 &`
- Lane 2 after frozen lane 3 (frA_ls_basis, about midnight): `SECTIONS="A B D E F Z" nohup bash queue_daqlk.sh > logs/laneENC.out 2>&1 &`
- Peak workers: 4 + 3 = 7, plus about 2 for lane 3 if lane 2 starts before lane 3 ends.
- Lane 2's `d_attempt_maps_restnl` waits for lane 1's `g2b_restnl_mask_lp`. So lane 2 run alone, without lane 1, would
  wait at that step.

## Steps (227 + 1 gate)
Runtime column:
- "meas." = measured earlier (RENDER_LOG / Section C / rr8 runtimes / this dry run).
- "est." = my estimate with no measurement under the rebuild.

Everything writes under the rebuild root: the profile's fence refuses writes outside it, and every explicit
`--out` / `--output` is under the root.

### G (31 steps; lane 1; about 10-12 h, mostly est.)
| step | script / args | env | mode, deps | output | runtime |
|---|---|---|---|---|---|
| g2a_restnl_mask | `-m wfield_local.restnl_mask --jobs 4` | `--block restnl_mask` (PRE_FROM unset) | stop | `<root>/session_cache_restnl/` (new `_daqlk` stems; old stopRun6 files are NOT pruned, since the stem carries the tag) | 11 min meas. |
| g2b_restnl_mask_lp | same, `WIDEFIELD_GATE_SOURCE=lp` | block restnl_mask | stop | `session_cache_restnl_glp*` | 11 min meas. |
| g5_epoch | `-m wfield_local.epoch_grant_figures --parallel-arms 2 --only 1b 1c acc 5c 5cr 5r 5rm 5ro 5rmo 10e 10cs 12s 12b 13s 14m 15r mat scal` | base | stop, g2a | `labcams/grant_figures/<tag>/epoch` (+ `data/` sidecars) | 2-3 h est. (never completed under the rebuild) |
| g5_section_g | `-m wfield_local.poststroke_section_g --output <root>/figures_working -j 4` | base | stop | figures_working | ~1 h est. |
| g5_cd_traj | `-m wfield_local.cd_trajectories --align precue cue` | base | stop | epoch dir `results/` | ~30 min est. |
| g5_recovery_cue / _precue | `-m wfield_local.recovery_trajectory [--align precue]` | base | stop | grant_figures tag dir | ~20 min each est. |
| g5_joint_xsession | `--output FW --align cue precue lick -j 4` | base | cont | figures_working | ~45 min est. |
| g5_nolick, g5_pcd (`-j 4`) | nolick_decoder, position_coding_directions | base | cont | grant_figures tag dir | ~30 min each est. |
| g5_chronic_stability, g5_chronic_drift | `-m scripts.chronic_stability`, `scripts/chronic_drift_vs_reorg.py` | base | cont | routed | ~15 min each est. |
| g5_june_aug | `-m scripts.june_august_drift` | `--block june_august_drift` | cont | routed | ~20 min est. |
| g5_rest_frozen, g5_rest_frozen_nl | rest_frozen_decoder; `rest_frozen_decoder_nl.py --mask restnl --pre aug` | `--block rest_frozen_decoder_nl` for nl | cont (nl: g2a) | routed | ~30 min each est. |
| g5_restnl_maps_{contact,lp,enl1s} | `scripts/restnl_maps.py --arm X` (lp: `--block restnl_maps`) | base | cont, g2a / g2b | routed | ~30 min each est. |
| g5_lpfinal_maps / _dec | `lp_final_render.py maps`; `decoders --window 1.0 --trained lp` | `--block lp_final_render` | cont | routed | ~45 min each est. |
| g5_map_drift_{contact,lp,enl1s} | `map_drift_prepre.py --arm X --refs rest precue restnl` | `--block map_drift_prepre` | cont, the matching restnl_maps | routed | ~20 min each est. |
| g6_grant | `-m wfield_local.grant_figures -j 4` | base | stop, after epoch / section_g / recovery x2 / cd_traj | grant_figures tag dir | 30-60 min est. |
| g7_section_g_figs | `section_g_figures --src FW --output FW` | base | cont, section_g | FW | min |
| g7_cd_geometry | `scripts/cd_geometry_figure.py --dir <grant_figures/<tag>/epoch>` | base | cont, cd_traj | epoch dir | min |
| g7_fig_frozen_refit_pooled | `rebuild_figs/... --src <epoch data>/epoch_5rmo_frozen_refit_overall_cue_working_sessions.csv` | base | cont, epoch | `<root>/rebuild_figs/<tag>/` | min |
| g7_fig_state_position, g7_fig_s_per_animal | `--data-dir <epoch data>` | base | cont, epoch | rebuild_figs | min |
| g7_fig_farcontra_attempts | `rebuild_figs/fig_farcontra_attempts_sessions.py` | base | cont | rebuild_figs | **1 min meas.: RUN FOR REAL in the dry run, exit 0, `.done` set** |
| g7_fig_pose_validation | `rebuild_figs/fig_pose_validation_sessions.py` | base | cont | rebuild_figs | few min est. |

### A: frozen-encoder downstream (14 steps; about 6-8 h est.)
The frames are already rendered by lanes 1-4.

| step | args | env | mode, deps | output |
|---|---|---|---|---|
| a_decomp_noFR_{cue,lick} | `lp_encoder_decompose.py --folder ENC/frozen_encoder_F/joint --align X` | noFR | stop | `ENC/qc/lp_encoder_decompose_*` (about 5 min each) |
| a_relabel_noFR | `-m wfield_local.frozen_relabel --from F --tag v7g3_t3d_spont --jobs 2` | noFR | stop | relabel folder |
| a_allsession_ref | `-m wfield_local.allsession_encoder --from F --tag v7g3_t3d_spont_ref --jobs 2` | noFR | stop | `ENC/allsession_encoder_v7g3_t3d_spont_ref` (Layer 2; command inferred, as in the doc) |
| a_relabel_trial (+ relabel_confusion `--trial --window lick/cue`) | `--tag v7g3_t3d_spont_trial --jobs 3` | noFR | stop / cont | |
| a_decomp_frA_{cue,lick} | as above on `F_frA/joint` | frA | stop | |
| a_allsession_frA, a_allsession_frA_roi | `--from F_frA --tag v7g3_t3d_spont_frA[_roi] [--roi-targets --post-from v7g3_t3d_spont_frA] --jobs 3` | frA | stop | |
| a_relabel_frA | `--from F_frA --tag v7g3_t3d_spont_frA --jobs 3` | frA | stop | |
| a_svf_ls, a_partial_ls | `session_vs_frozen.py` / `partial_refit.py --tag F_frA_ls --jobs 2` (needs frA_ls only, which is done) | frozen | cont | inside the frame folder |

### B: encoder figure scripts (48 steps; about 6 h est.)
- **Steps (all env frA unless noted):**
  - frozen_fr_compare, frozen_fr_musall, frozen_fr_musall_v2;
  - frozen_posinfo `--tag F_frAsplit` (env frA + FROZEN_FR_SPLIT=1);
  - specificity_control `--tag F_frA --layer2 allsession_encoder_v7g3_t3d_spont_frA_roi`;
  - session_vs_frozen `--tag F_frA --jobs 3`, then `--plot-only`;
  - matched_residual;
  - partial_refit `--tag F_frA --jobs 3`;
  - frozen_pixel_unique_sig on `F_basis` (noFR, with and without `--zscore`), on `F_frA_basis`, and on
    `F_frAsplit_basis` (split env);
  - frozen_residual_significance `--tag F_basis` (noFR, mode stop):
    - one call per spout, in the canonical order close ipsi / mid / contra, then far ipsi / mid / contra;
    - then `--spout all` and `--spout all --per-animal`;
    - after each call the queue copies the fixed-name `residual_significance.csv` to `residual_significance_<spout>.csv`,
      as `run_resid_spouts.sh` did. The script overwrites one name per call.
  - frozen_residual_trend `--tag F` (needed by fig_pooled);
  - residual_roi `--tag F [--by-spout]` and `--tag F_frA`;
  - remap_lmm, remap_checks;
  - movement_null_plan_maps `--frames FR-A noFR --jobs 3`;
  - frozen_peth_trial_maps, frozen_kernel_maps;
  - frozen_pixel_maps `--tag F_basis --variant {joint,target_first,exec_first}`, frozen_r2_maps, frozen_spout_maps
    `[--contrast]` (noFR);
  - layer2_matched_n, layer2_random_splits PS92-95 `--max-splits 20`, then rebuild_figs fig_layer2 / fig_rsplit
    `--allsession-tag v7g3_t3d_spont_ref`;
  - fig_resid_per_animal `--encoder-tag F_basis`;
  - exec_vs_neural_match `--source decoder`, `--target far_contra`, `--target far_mid`;
  - encoding_key_figures.
- **Runtimes (est.):** session_vs_frozen and partial_refit are about 1-1.5 h each. The rest are 2-20 min each.

### T: te + lsland + frlag frames and their readers (8 steps + gate; `queue_daqlk2.sh` only; about 16 h est.)
- **Gate.** T waits until frozen lane 3 has finished. It checks for `frA_ls_basis` at 107 sessions plus its musall csv, or
  for lane 3's chain log reporting EXIT / FAILED / lane done. It then proceeds either way: the T frames do not read
  ls_basis, the gate is only for RAM and workers.
- **te: NOT the spont design.** The merged `frozen_encoder` refuses `FROZEN_DESIGN=te` together with
  `FROZEN_LICKS_ALL=split` ("te has its own pre-cue / ITI lick families: set FROZEN_LICKS_ALL=none"). So the te frames
  and every reader of them run with `FROZEN_LICKS_ALL=none`. Their tag is `gate-attempted_v7g3_t3d_frA_te[_basis]`,
  with no `spont` segment; `check_tag` enforces this and passes in the dry run. Pre-cue / ITI licks are inside the te
  design (te_lick_parts), so this is not a mix with the no-spont robustness row.
- **lsland variant list.** From backup `S1_cdd1e016/run_land_lag.sh` (10-25), the "trimmed variants":
  `joint target_first exec_first no_target no_spout no_task no_movement no_lickspout no_exec_locked no_exec_within
  no_lick no_fr`. It contains the two variants contest_lines reads (no_target, no_lickspout). `no_spont` is not in the
  10-25 list and was not added.
- **frlag frame (not in the request, ADDED).** `contest_lines.py` also reads `..._frA_frlag` (lagged FaceRhythm, its
  second figure). It comes from the same backup script with the same variants. Without it, contest_lines would fail
  after its first figure.

| step | args | env (after the profile) | deps | output | runtime est. |
|---|---|---|---|---|---|
| t_frame_te | `-m wfield_local.frozen_encoder --all --jobs 3 --no-drop-spout --tag gate-attempted_v7g3_t3d_frA_te` | frA + FROZEN_DESIGN=te + FROZEN_LICKS_ALL=none (LICKSPOUT unset) | gate | `ENC/frozen_encoder_gate-attempted_v7g3_t3d_frA_te` | 3-4 h |
| t_frame_te_basis | `... --jobs 2 --no-drop-spout --target basis --tag ..._frA_te_basis --variants joint target_first exec_first no_timing no_target no_execution no_contact no_other no_task no_movement task_only target_only execution_only other_only movement_only` | same + FROZEN_PIX_MUSALL=1 | t_frame_te | `..._frA_te_basis` | ~3 h |
| t_svf_te | `session_vs_frozen.py --tag ..._frA_te --jobs 3 --variants joint no_target no_execution no_contact no_task no_movement` | te env | t_frame_te | in the frame | ~1.5 h |
| t_partial_te | `partial_refit.py --tag ..._frA_te --jobs 3` (the doc's `--jobs 6`, capped) | te env | t_frame_te | in the frame | ~1.5 h |
| t_te_readout | `te_readout.py --tag ..._frA_te` | te env | t_frame_te | `<frame>/figures/te_unique_lines.png` | min |
| t_frame_lsland | `--all --jobs 3 --tag gate-attempted_v7g3_t3d_spont_frA_lsland --variants <trimmed list>` | frozen (frA + LICKSPOUT) + FROZEN_LICKLAND=1 | gate | `..._spont_frA_lsland` | ~3.5 h |
| t_frame_frlag | `--all --jobs 3 --tag gate-attempted_v7g3_t3d_spont_frA_frlag --variants joint no_fr no_fr_dir no_frfacelead no_frcortexlead no_movement` | frA + FROZEN_FR_LAGS=-0.3,-0.2,-0.1,0.1,0.2,0.3 | gate | `..._spont_frA_frlag` | ~3 h |
| t_contest_lines | `contest_lines.py` | frA | lsland, frlag, a_svf_ls, b_svf_frA | frozen_fr_compare figures | min |

- **Dry run.** Ran under each step's exact env:
  - `--help` exits 0 for all of them, and every flag exists.
  - Every requested variant is in `frozen_encoder.VARIANTS`.
  - `check_tag` accepts each tag, with design `v7g3_t3d` for te and `v7g3_t3d_spont` for lsland / frlag.

  No frame was run.
- **Order / workers.** In lane 2, T sits after B, as the coordinator preferred. Each frame step is serial with
  `--jobs` <= 3, alongside lane G's 4: 7 workers.
- **Cost of that order.** Lane 2 grows from about 25 h to about 41 h, and D / E start about 16 h later.
- **Alternative (my suggestion).** Once lane G has ended (its CHAIN line "queue end (sections [G])", expected early
  morning):
  - lane 2 runs `SECTIONS="A B D E F Z"`;
  - T runs as its own lane, `SECTIONS="T" nohup bash queue_daqlk2.sh > logs/laneT.out 2>&1 &`.

  That gives 3 + 3 = 6 workers. MINFREE 10 GB still gates every step, and `t_contest_lines` waits for lane 2's two
  self-fits. Don't run T as a third lane while lane G is still running: that would be 10 workers.

### D: attempt_coding / attempt_maps (10 steps; about 3-4 h est.)
- **Steps:**
  - `attempt_coding --freeze` (WIDEFIELD_ENL_WINDOW_S=1.0, DAQ);
  - LP run;
  - `--aligns lick cue` and `--aligns precue`, both with `--out <imaging_encoding>/attempt_coding_pp`. The output is
    routed under the tag (lp_family).
  - attempt_maps `--rest-variant restnl` / `rest --jobs 3` (LP lick + gate source), then `--plot-only`. The plot-only
    replays this render's figdata only, after both runs.
- **Readers of attempt_coding, moved here from B / G7:**
  - exec_vs_neural_match `--source encoder` (after relabel_trial);
  - fig_goal_epochs `--coding-dir <imaging_encoding>/attempt_coding/<tag>`;
  - fig_goal_summary `--enc-dir <imaging_encoding>`.

### E: ITI family + goal-page lick chain (114 steps; ENV-frozen; about 5-7 h est.)
- **Sources.** Waves 1-3, the docked arm and the lick chain from backup `S2_f082ee0c/run_v8redo.sh`, `run_v8wave2b.sh`,
  `run_v8wave3.sh`, `run_dock.sh` and `fr/run_main.sh`.
- **Order.** Serialised in their dependency order. Filter / decoder variants are passed as env tokens
  (WFIELD_LICK_DECODER=areas_split / areas, WFIELD_ITI_FILTER=gap2 / far1 / gap2far1 / dock / far1s / dock01 / gap2dock).
- **Output.** Under the root: `iti_licks.OUT` = `<imaging_encoding>/iti_licks/<tag>` (checked in the dry run).
- **Runtimes.**
  - Measured 10-27 (rr8 runtimes; inputs were cached then): wave 1 about 45 min plus iti_hemo_check 40 min; wave 2
    about 1 h if serialised; wave 3 about 30 min.
  - The fresh `iti_licks.py --jobs 3` extraction will be longer (about 1-2 h est.).
- **Changes from the old scripts:**
  - The FR-controls run (`fr/run_main.sh`) is the same command as wave 1's `--controls`, so it runs once. The
    unblanked-FR code is in the merged code (`fr_unblanked` in iti_licks.py).
  - `lick_firstlick_extract` was added before the dock arm, because `lick_balance_subsets` needs it (rebuild_order_B
    consumer step 6).
  - Then within_lick_correction run / summarise / plot -> learned_compensation run / summarise / plot ->
    prelaunch_aim_tracking extract / fit / summarise -> face_prelaunch_aim extract / fit / summarise / offset.

### F: misdirected licks (2 steps; about 30 min est.)
- `misdirected_lick_counts.py`.
- `misdirected_lick_decode.py --name enc_rebuild_20261104b`. This needs the fix below.

### Z: gated on frA_ls_basis, then fig_pooled (9 steps + gate)
- **The gate** waits, polling every 15 min, until `ENC/frozen_encoder_F_frA_ls_basis/joint/sessions` has 107 json files
  AND the frame's `frozen_musall_r2.csv` exists.
  - At 16:33 it was at 35/107; lane 3's ETA is about midnight.
  - If lane 3's chain log reports `frA_ls_basis EXIT != 0` or `FAILED`, the gated steps are skipped.
- **Gated steps:**
  - frozen_pixel_musall `--tag F_frA_ls_basis --layout {novar,alone} --window {cue,lick}`;
  - spout_unique_maps;
  - epoch_kernels `--tag F_frA_ls_basis --jobs 3` (about 1 h est.).
- **Then:**
  - frozen_claims, last of the encoder figures;
  - rebuild_figs/fig_pooled, after a_decomp_noFR_lick, a_allsession_ref, d_attempt_lp and b_resid_trend. The
    SECTION_C report left it open until these exist.

**Total.** About 36-40 h of serial work. With two lanes: G about 11 h, A-Z about 25 h. One night covers G plus A and part
of B. Priya should choose the priority order (see Decisions).

## Dry run (16:2x-16:36)
- `DRYRUN=1` lists all 228 lines (227 steps + 1 gate).
- **Script and flag checks** (`dryrun_check.py`):
  - Every script exists in the worktree.
  - 69 argparse scripts: `--help` was run under each step's exact profile and env. All exited 0, and every flag used
    is in the help.
  - The hand-parsed scripts (sys.argv; mostly ITI / lick): every flag and subcommand was found as a quoted literal in
    the source, and every file compiles.
- **Paths not present yet:** only outputs of earlier queue steps (figures_working, the epoch dir and sidecars,
  attempt_coding[_pp]) and the frA_ls_basis frame (32-35/107). Every other input frame has 107 sessions.
- **Output paths checked:**
  - ITI OUT, frozen_encoder.out_root, attempt_coding.out_dir, `_common.default_out` and the epoch data dir all
    resolve under `<root>/.../<tag>`.
  - `run_mode.cache_tag()` carries `_daqlk`, so session_cache / restnl prunes never touch the stopRun6 /
    stopA2jexrt3500c70 entries.
- **Run for real:** `g7_fig_farcontra_attempts`, through the queue itself (1 min, exit 0). Its output
  `<root>/rebuild_figs/<tag>/farcontra_attempts_sessions.png` follows the conventions:
  - no top spine;
  - no stars;
  - open markers, shape = animal, colour = epoch.

  It is marked done.
- **Preflight-only test** (`ONLY=__none__`): passes on 9aaa4cc6. The first attempt also blocked on the Z gate. I
  stopped my own two processes (bash 578616, sleep 578648; exact PIDs) and fixed the gate so it is skipped when `ONLY`
  names no `z_` step. Nothing else ran.

## Code fix (local branch, NOT pushed)
- **What:** `9aaa4cc6` on `fix/misdirected-decode-bundle-dir-20261105` (worktree wf_render_daqlk, parent 136807cf).
  `scripts/misdirected_lick_decode.py` now reads the bundle through `o2_encoding.bundle_dir`, not
  `o2_inference.bundle_dir`.
- **Why:** under a rebuild root the old call pointed at `DeepLabCut/Widefield/o2/<name>`, so every session would have
  been skipped ("no bundle inputs"). The new call gives `<root>/o2/<name>`, and its default run is unchanged.
- **Inputs:** bundle b's `inputs/` holds 107 pkl files.
- **Tests:** `tests/test_routed_redo_scripts.py` + `tests/test_o2_encoding_env.py`: 81 passed. ruff pre-commit clean.
- **Next:** merge it to main with the next main push. Until then the queue runs on 9aaa4cc6.

## Not placed / looks wrong (needs a decision or code)
1. **te frames.** RESOLVED 11-05 evening: section T renders them with FROZEN_LICKS_ALL=none (the code refuses te + spont).
2. **contest_lines.py.** RESOLVED 11-05 evening: section T renders lsland, and also frlag, which contest_lines needs.
3. **fig_resid16 (`--bin 16` / `32`) and fig_resid_spout have no producer.** They read
   `residual_sig_<F>px.csv` / `resid<F>_*.npz`, but no script in the repo writes them, and frozen_residual_significance
   has no `--bin` flag. The binned-residual producer was presumably a scratchpad script and must be recreated before
   these Figs 7 / 7c / 7d can be drawn.
4. **G7 share-only figures not placed:**
   - fig_decoding_sessions: needs `slowband_decode_check.py --all` under the variant. It is a G0 "parameterise" item and
     was not verified.
   - fig_state_ps94_diagnostic: the producer of its cache is unknown.
   - fig_enl_move: needs enl_movement_multisession (D1).
   - fig_behaviour_hits: reads behaviour only, so it stays on its gate per D8.
5. **Not in this queue:**
   - drift1500 arm: needs its own frozen frames and an ITI re-run;
   - B11 panel, G4 figure, slide-95 raw + z: no command found in the handoffs;
   - miss_vs_stopped JSON figure;
   - quiet events (G4);
   - 15e (preprocess maps);
   - G8 pages / decks / publish_cache;
   - downstream of the no-spont robustness frames (v7g3_t3d / _frA: decompose / relabel / Layer 2), not asked for.
6. **rebuild_order_B "Extract -> consumer" steps 3-15 are mostly not in the waves.** These are lick_cued_ramp, the
   precue_restw / restnl / revup / cue_to_lick chain, enlv / enl_angle, the balance family on the KEY preds, the decoder
   refits, lick_code, session-level maps, interhemi, lick_correction -> prelick_* and the dock subsets.
   - Only the waves, dock and the lick chain are placed.
   - prelaunch_aim_tracking / face_prelaunch_aim may depend on lick_correction -> prelick_svd_tuning (B step 13). They
     are `cont`, so a failure will show.
   - `iti_where_signed_error.py` was left out: it only prints, and it needs the ITI OUT as an argument.
7. **Hard-coded text and conventions (seen, not changed):** SECTION_C already flags these:
   - fig10's hard-coded suptitle sentence;
   - the top spine on fig10;
   - the non-canonical spout order in the droppos / shufpos maps.

   No new violations were seen in the one figure rendered here.
8. **Input cache.** The queue uses `wfield_cache/input_cache_rebuild_20261104`, as the frames did. Section C used the
   plain `wfield_cache`. Both are keyed on the source path, so the results are the same; only cache hits differ.
9. **The full suite was not run on 9aaa4cc6**, only the two relevant test files. 136807cf passed the pre-push suite.

## Decisions for Priya
- **A. Priority order.** About 36-40 h of work cannot finish in one night. Suggested order:
  1. G2 + G5 decoders + G6 grant (lane 1);
  2. A + B headline figures, then D (lane 2);
  3. E (ITI, about 6 h);
  4. F + Z.
- **B. te and lsland frames.** DECIDED (render): section T. Two points to confirm:
  - te runs as design v7g3_t3d (no spont kernels; te has its own pre-cue / ITI lick families);
  - frlag was added because contest_lines reads it.
- **C. Binned residual figures.** Recreate the `residual_sig_<F>px` producer for Figs 7 / 7c / 7d, or drop those figures.
- **D. Merge the decode fix.** 9aaa4cc6 should go to main.
