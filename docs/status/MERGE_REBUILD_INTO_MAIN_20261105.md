# Merge: rebuild/cohort1-20261103 into main (2026-11-05)

Priya approved starting the merge on 11-05. **Not pushed.** Review the branch first.

- Branch `merge/rebuild-into-main-20261105`, worktree `C:/Users/SabatiniLab/Github/wf_merge_main`.
- Parents: origin/main `6c00efbb` and rebuild `31ec5c1a9c764fd6d12490dd0787080cdd110455`. The rebuild branch had not
  moved from 31ec5c1a.
- Merge base `e8491dc8`. The rebuild has 789 commits since the base; main has 177.
- Merge commit `2730033a`. Follow-up commits fix the tests (listed below).

## Conflicts (21 files) and how each was resolved

All 21 conflicted files were also edited on the rebuild, which hand-ported every main commit that touched them:

| main commit | rebuild port |
|---|---|
| 26fd5b44 | 5b6f35df, 724862f9 |
| 48f2cc6d | db33dac2, 2c59f85b |
| 4a37cbc8 | 724862f9, merged in 9fb2d964 |
| b67e3dbd (Wt) | 3bedd477 |

I checked the main commit list file by file. No main commit touches these files apart from the ports above, plus:

- cec38954: a docstring in `trial_classes`;
- 02a9cc64: the s1 snapshot;
- ae2bcd86: the softmax script;
- fc420c2e: the bundle-memo test.

A hunk-level merge put main's pre-port code next to the rebuild's port of it, for example a second
`grant_kit._frozen_frame` with main's old signature. So for the module files I took the **whole rebuild file**. I then
checked main's added lines against it (scratch diff) to find any main-only fix.

| file | resolution |
|---|---|
| wfield_local/attempt_coding.py, grant_confusion.py, grant_kit.py, joint_xsession.py, locanmf_frozen_decoder.py, position_axes.py, position_coding_directions.py, poststroke_compare.py, poststroke_section_g.py, precue_engagement_states.py, spout_scaling.py, trial_classes.py, working_trials.py (add/add) | Rebuild file (the port of main's B/C/D/G, working = contact + incomplete, Wt centre). |
| wfield_local/grant_similarity.py | Rebuild file, plus main's explicit closure binding in the `7b_disatt` lambda (`pre_x=pre_x, ...`) instead of the rebuild's `# noqa: B023`. |
| wfield_local/nightly_figs.py | Rebuild file. Main's cohort-1 `os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801")` and `IMAGING_PRE_FROM` are **not** kept. See decision 1. |
| scripts/s1_trial_class_snapshot.py | Rebuild's `working_pl_<anchor>` (from `classify_session`), plus main's jaw-free `working_pl_<anchor>_licks = working_pl_<anchor> & camera_lick` columns. `p_img_s1_table_20261103.py` reads them. |
| scripts/softmax_neural_vs_lick.py | Rebuild's routed `encoding_root` / attempts imports, plus main's `pose3d_dir()` path (ae2bcd86). |
| tests/test_pooled_bundle_memo.py | Rebuild key `(an, align, _tck.gate_mode(), _tck.run_selection_tag())` (a8 note b). |
| tests/test_stopped_trial_class.py | Rebuild side (comment only). |
| tests/test_bcdg_1104.py (add/add) | Rebuild's adapted port of main's test. |
| DECISIONS.md | Union: main's 11-03 / 11-04 entries first, then the rebuild's 11-03 (night) / 11-04 entries. |

### Checked in files that auto-merged

- **a8 note (a).** `scripts/kinematics_pooled_figs.py` and `scripts/lmm_curve_summaries.py` keep main's
  `pose3d_dir()` and use `FL.sidecar_for`.
- **prep_3d_inputs.** It keeps main's commanded-tip `TARGET3D` (cf7c4cad) and `pose3d_dir` (ae2bcd86). The rebuild adds
  only its `require_default` guard on top.
- **Lick detector.** The old imaging literal `(2.5, 1.0, (0.001, 0.020), 0.10)` remains only in `scripts/archive/`
  and `scripts/rest_migration/archive/`, as on the rebuild. Every live caller uses `config.lick_detector()`
  (REBUILD_CODE 8.53).
- **Version constants.** `session_cache.CACHE_VERSION` = 14 (main 13, rebuild 14, so the max).
  `behavior_events.SCHEMA_VERSION` = 5 on both. The two script-local `CACHE_VERSION = 1` constants are unchanged.
- **Hooks.** `core.hooksPath` = `.githooks` (a8 note c). The pre-commit ruff hook passed on all 560 changed `.py` files.

## Follow-up commits (merge-introduced test failures)

- **`fed41d8d` `tests/test_working_trials.py`.** This test file exists only on main. Two of its tests pinned main's
  `grant_kit._class_select(work=)` and the spec field `working_trials.tag()`. The rebuild selects working rows from the
  camera classes (`_class_select(cls=)` / `trial_classes.camera_select`) and records `trial_classes.run_selection()`,
  which carries main's `-cls-contact-incomplete` part. I rewrote both tests on that API with the same selections:
  in_working, in_working_jaw, the DAQ rule, and no fallback without classes. The frame check uses the label-free
  working centre (`WIDEFIELD_SPOUT_SCORE_CENTRE=working`), as `test_bcdg_1104` does, because the synthetic pool is too
  thin at some spouts for the default Wt centre.
- **`fed41d8d` `tests/test_share_writers_mode_safe.py`.**
  - Added to BEHAVIOUR: main's pose-only side-camera writers (`pose3d_side_check`, `pose3d_side_loo`,
    `pose3d_side_video`, `side_clean_qc`, `side_label_offsets`).
  - Removed: `iti_first_lick`, `pose3d_close_dir_sheet`, `pose3d_figures` and `pose3d_summary_fig`. After
    `pose3d_dir()` / `il.OUT` they are no longer unrouted share writers.
- **`51dae7cd` `tests/test_rebuild_root.py`.**
  - Registered main's `wfield_local/dlc_sidecam.py` (side-camera labelling frames) as RESOLVED.
  - Dropped `wfield_local/locanmf_event_master.py`, which main deleted in ed516423 (Priya approved 11-04).

## Decisions for Priya

1. **August-only pre outside the render profile.** Main's `nightly_figs` set `WIDEFIELD_PRE_FROM=0801` for cohort 1
   by default. The rebuild puts it in the render profile instead (`configs/render_profiles/cohort1_rebuild_20261103.yaml`).
   The profile *unsets* it per block (june_august_drift, 12.1b, map_drift_prepre, restnl_mask), and a `setdefault` in
   `nightly_figs` would put it back in those blocks. So the merged code follows the rebuild. A plain
   `nightly_figs` without the profile now uses all pre sessions (June + August) unless the variable is set. Should
   non-profile runs get the August default back? If so, it needs a profile-aware form.
2. **Main APIs that changed.** `working_trials.row_groups` now reads the live classes
   (`trial_classes.cue_groups_for`, with `YE=` / `YU=`) instead of main's s1 snapshot. `grant_kit._class_select(work=)`
   is now `cls=`. Spec `trials` = `trial_classes.run_selection()`, not `working_trials.tag()`. Main's tests of these
   were adapted, not kept verbatim. A grep of `scripts/` and `wfield_local/` finds only callers that already use the
   new signatures. Hand scripts outside the repo would need updating. This is information, not a blocker.

## Tests

Ruff passed on every merge-changed `.py` file. Full suite: 4 shards, run on the resolved tree before the merge commit.

| | passed | skipped | failed |
|---|---|---|---|
| shard 0 | 1793 | 1 | 3 |
| shard 1 | 1033 | 26 | 1 |
| shard 2 | 1250 | 143 | 0 |
| shard 3 | 935 | 5 | 2 |
| **total** | **5011** | **175** | **6** |

The 6 failures:

- **`test_o2_encoding_env` (2): not a code failure.** These tests ship `git HEAD`, and before the commit HEAD was still
  main's 6c00efbb, which has no `o2_bundle`. After the merge commit, `tests/test_o2_encoding_env.py` gives
  13 passed, 0 failed.
- **The other 4 were caused by the merge** (the main-only test file, and writer registers missing main's new
  scripts). The follow-up commits fix them, and the re-run files pass:
  - `tests/test_working_trials.py`: 7 passed
  - `tests/test_share_writers_mode_safe.py`: 5 passed
  - `tests/test_rebuild_root.py`: 20 passed

No failure was pre-existing on either parent. Shard 1 is slow (about 80 min) because `test_frozen_default`'s section G
stored-vs-in-memory tests read real data. The tests read from the shares but do not write to M: or N:.

## Decision 1 resolved (Priya 11-05: "ok")
`nightly_figs` restores main's August-only pre default for cohort 1 (`IMAGING_PRE_FROM = "0801"`, `setdefault`) ONLY when
no render profile is active (`rebuild_root.active()` false). Under the profile, WIDEFIELD_PRE_FROM stays the profile's
(including its per-block unsets). An explicit WIDEFIELD_PRE_FROM still wins. Test: tests/test_nightly_pre_from_default.py.

## Second merge of origin/main (138048a3, 4 commits: cohort2 decks, FR spout-in-mesh control)
- One conflict, wfield_local/preprocess_deck.py: kept both. main's `__SPOUTMAPS_NATIVE__` cohort-level branch runs first,
  then the rebuild's `_tagged_map_dir(subdir)`. Bucket loop: main's `for animals, _n in buckets:` (the index was unused).
- Guard tests: fr_spout_mesh_control.py classified behaviour-only (FR readouts dir); cohort2_qc/spout_maps.py dropped
  from LITERAL (it now takes its share root from seizure_events_video.MICRO; the runtime fence still applies).
- Re-run after the fix: test_preprocess_deck, test_crossday_intensity, test_share_writers_mode_safe, test_rebuild_root,
  test_nightly_pre_from_default, test_entrypoint_after_defs, test_cli_help_is_console_safe: all pass.
