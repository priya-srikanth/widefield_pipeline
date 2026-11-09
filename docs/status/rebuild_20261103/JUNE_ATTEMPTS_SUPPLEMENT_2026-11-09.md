# June camera attempts supplement (Priya 11-09, route 2b)

**Problem.** The LP arm of `map_drift_prepre` (block `map_drift_prepre`, PRE_FROM unset) failed on June: the run's
registered attempts table (`qc/sel-camera-...-stopA2jexrt3500c70/trial_attempts.csv`, 107 sessions) was built under
`WIDEFIELD_PRE_FROM=0801`, so it has no June rows, and `lp_licks._session_lp` raised `LPUnavailable` for them.

**Decision (Priya 11-09).** Do not rewrite the registered table. Rewriting it would change `lp_licks.spec_token` and orphan
every `-lp<sha>` / `glp<sha>` cache entry. Instead, build a June-only side table and serve it through an opt-in
supplement that can only ADD sessions.

## Side table

`M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/qc/trial_attempts_june/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/`

- `trial_attempts.csv`: 5,792 rows, 12 sessions (PS92-95 x 0606/0607/0608), 304-627 trials each. Same 28 columns and
  dtypes as the registered table. `gate = camera:sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70`.
- `trial_attempts_sessions.csv`: per-session wide table. Its `lateral_bias` / `lateral_scatter` columns are NaN,
  because the build holds no August-pre reference. The LP source never reads this file. There is no gate table,
  because the build holds no August rows.
- Built by `scripts/trial_attempts.py --jobs 4 --out <above> --labels <12 June labels>` under
  `render_profile cohort1_rebuild_20261103 --block map_drift_prepre` (PRE_FROM unset), branch
  `feat/june-attempts-supplement-1109`, 11-09 00:56-00:57.
- **August invariance check.** The same code and env were used to build PS92_0807 (pre), PS93_0818 (acute) and
  PS94_0825 (subacute) into `qc/trial_attempts_june_augcheck/<same tag>/`. All 1,584 rows are byte-identical to the
  registered table's rows for those sessions (line-for-line CSV text, and `assert_frame_equal`).
- The registered table and its sessions CSV were not touched (sha1 15bd912b83e8 / 48688e68c170 and mtimes from 11-04
  19:16, checked before and after).

## Switch

`WIDEFIELD_ATTEMPTS_SUPPLEMENT=<side table>/trial_attempts.csv` (`lp_licks.SUPPLEMENT_ENV`).

- `_session_lp` looks a session up in the registered table first, and in the supplement only if the session is absent
  there. If a label appears in both tables, the supplement is refused (`AttemptsSupplementOverlap`). A missing
  supplement file also raises.
- **Keys.** `spec_token` / `cache_tag` do not change (`lpa120a06d` with or without the switch). A session served from
  the supplement gets `supplement_token` = `sup:<size>:<mtime_ns>`, which `session_cache.cached` folds into that
  session's signature, and only under an LP switch. A rewrite of the supplement therefore recomputes only June entries.
  The kinds and folder names (`session_cache_restnl_glp<sha>`) stay the same.
- With the switch unset, the code paths and keys are identical to before (tests/test_attempts_supplement.py).
- Side effect: `lp_licks.py`'s source enters the version of its local `lp_licks/<label>__times` sidecars, so those
  sidecars re-read their CSVs once on first use. A cached prm hit never reaches them.
