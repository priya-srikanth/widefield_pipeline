# cam1 far-from-mouth guard in code: one rule for every cam1 lick reader (11-17)

Branch `fix/cam1-farmouth-guard-1117` (from `analysis/restnl-cam1-1117` at 5a43b845). Rule approved by Priya 11-17
(blind review: `CAM1_EXTRA_LICKS_CHECK_20261117.md`; blast radius: `CAM1_FARMOUTH_BLAST_RADIUS_20261117.md`).
Code only. Nothing was re-rendered and no M: output was written.

## The rule

`wfield_local/cam1_licks.keep_mask(E, cam4_matched=None, *, label=None)` returns `bool[len(E)]`. True means the row
may count as a cam1 lick. A row is dropped when either holds:
- **Water droplet.** The `droplet` column; if the column is absent, `droplet_flags` on the cam1-only rows.
- **Far-from-mouth label.** The row is cam1-only (not cam4-matched) and `med_out_px < FAR_FROM_MOUTH_PX` (-120 px).

Other details:
- cam4-matched rows are never dropped. Callers that want cam1-ONLY licks still AND the result with `~cam4_matched`.
- A table without `med_out_px` gets no far guard. This is printed once per session.
- `FAR_FROM_MOUTH_PX` now lives in `cam1_licks`; `restnl_licks.FAR_FROM_MOUTH_PX` re-exports it.
- `cam1_licks.flag_col` is the shared True/False column parser (missing / NaN = False).

## Consumers switched to `keep_mask`

| consumer | before | now |
|---|---|---|
| `restnl_licks.cam1_only_onsets` (RESTnlc1) | own copy of droplet + far | `~cam4_matched & keep_mask` (output identical, pinned) |
| `trial_classes.session_trials` | `~droplet` in the incomplete filter and the jaw-only "seen" set | `keep_mask` in both |
| `scripts/first_lick_latency.session_latency` | `~droplet` | `keep_mask` |
| `lp_licks._read_times` | `~cam4_matched & ~droplet` | `~cam4_matched & keep_mask` |
| `combined_encoding._cam1_lick_events` | `~cam4_matched & ~droplet` | `~cam4_matched & keep_mask` |
| `scripts/revup_behaviour.py` | `~droplet` | `keep_mask` |
| `scripts/cam1_incomplete_licks.py` | `~droplet` | `keep_mask`, per file before the concat |
| `scripts/cohort2_qc/lp_licks.lick_times` (cohort1 quiet chain + cohort2 quiet runs) | `~cam4_matched & ~droplet` (both branches) | `~cam4_matched & keep_mask` |

**Missing droplet exclusion.** Every consumer named in the audit already excluded droplets.

Two other readers do not exclude droplets. Neither was changed; both need a decision:
- **`scripts/restw_strict_check.py`, the `strict_c1` arm and the `cam1_n` / `cam1_in_rest` / `cam1_only_in_rest`
  notes.** These read every `events_all` row: no droplet filter, no cam1-only filter (only the last note drops
  cam4-matched rows), and no far guard.
  - It is a superseded diagnostic arm. Its "strict" arm (RESTnlc1) goes through `restnl_licks` and is guarded.
  - The per-session npz cache is keyed on ONE `VERSION`, shared with the live `strict` arm. Changing `strict_c1` would
    therefore need a `VERSION` bump, which re-runs the whole RESTnlc1 `restw_check_c1v3` set.
  - Left as is. Decide before its next run.
- **`scripts/cam1_licks.summarise` (the producer's own summary and review sampling) and
  `scripts/cam1_jaw_review._cands`.** These exclude droplets but not far labels.
  - Both are review / sampling tools for blind reviews already scored. Changing them would change which events
    those seeded rounds draw.
  - Left as is, on purpose.

## Caches

**Invalidated by this change:**
1. **`trial_classes.class_source_signature` (csig).** The code part moved from d51d27981a to b95ce95484 (measured on
   empty input dirs). It now also hashes `map_collect.closure(cam1_licks, "keep_mask")`, so a later rule edit
   invalidates too.
   - Every camera-gated session-cache kind changes through it (`run_mode` part `csig`, cache only), and so do the map
     caches (`map_collect`, through `session_cache.effective_kind`).
   - So does every `kin_gate_signature` result cache (`pose3d_phase`, `pose3d_spout_error`, `pose3d_cue_traces`,
     `trial_classes.kin_cache_dir`).
   - NB: `attempt_maps` figure names carry csig, and `deck_registry` pins `*csig-3d7bbe921e.png` patterns. A
     re-render writes new names, so the registry pattern must follow at that point (not edited here).
2. **`lp_licks.LP_VERSION` 2 -> 3.** This changes `spec_token` / `cache_tag`, i.e. every session-cache kind under
   `WIDEFIELD_LICK_SOURCE=lp`, plus the `lp_lick_times` sidecar (`lp3-<module code>`).
3. **`combined_encoding` behaviour cache (`CACHE_ENV`, kind `behaviour`).** `_CACHE_CODE` hashes the edited
   `combined_encoding.py` and now also `wfield_local/cam1_licks.py`. Every behaviour entry recomputes, in every lick
   mode. Imaging entries are untouched (keyed on `_load_imaging` + `hemo_dir` only).
4. **`revup_behaviour` per-session pkl.** It is existence-keyed, so the folder moved from `behaviour_sessions/` to
   `behaviour_sessions_c1far/`. The old folder is not read.

**Deliberately NOT invalidated:**
- **RESTnlc1 (`restnl_licks.code_digest`).** It still returns the v3 digest `e34a85c2b4b0`, through
  `_DIGEST_ALIAS`. The refactor changes no onset:
  - pinned against a frozen copy of the v3 body (`tests/test_cam1_keep_mask.py`);
  - identical on PS94_0823 (985 onsets), PS92_0929 (984) and PS95_0924 (1860).

  So the v3 caches of the re-run in progress stay warm. The alias matches only this exact source plus
  `keep_mask`'s closure; any later edit misses it and invalidates as before. Without the alias, every RESTnlc1 mask /
  15f c1all / rest_frozen_decoder_nl entry would have recomputed for a no-op.
- **`session_cache.CACHE_VERSION`** (14, unchanged). Bumping it would cold-start every DAQ-source and cam4 kind, which
  this change does not touch. The lick-dependent kinds are invalidated by the targeted tokens above (the pattern
  `class_source_signature` and `lp_licks.cache_tag` exist for).

**Not caches, but stale until re-run** (outputs; they do not regenerate themselves):
- `trial_attempts.csv`. It feeds the LP gate; its rewrite changes the LP `spec_token` again, automatically.
- `first_lick_latency.csv`.
- `cam1_licks/incomplete/*`.
- The quiet family `v2_quiet_lp` (the audit's re-run priority 1).
- The revup tables.

**Also not invalidated:**
- **Frozen-encoder saved statistics** under `COMBINED_ENCODING_LICKS=combined` (`STATS_VERSION` 7, unchanged). Force a
  refit when that family is re-rendered.
- **Frozen decoders / stores keyed on `selection_tag`.** These are references, not caches (rule 10).

## Tests

- **`tests/test_cam1_keep_mask.py` (new, 8 tests).** Covers:
  - the droplet row, the far row, a cam4-matched far row kept, and the threshold edge;
  - the `cam4_matched` argument;
  - the `droplet_flags` fallback, and the missing-`med_out_px` fallback printed once;
  - `flag_col`;
  - `restnl_licks.cam1_only_onsets` against the frozen v3 body (EXACT, 400 random rows through CSV, with and without
    the droplet / med_out columns);
  - the digest alias, and that it misses on any other source;
  - `session_trials`: a far cam1-only lick no longer makes a trial incomplete, and a far "seen" row no longer does.
- **Related suites.** 539 tests ran (cam1, restnl, trial_classes, lp_licks, attempts, camera gate, cue70, working
  selection, kin gate, combined / frozen encoder, gain_bio, quiet-chain routing, map cache key, ...): 538 passed.
  - The one failure is `test_share_writers_mode_safe::test_every_share_writer_is_routed_refusing_or_classified`. It
    lists `scripts/restnl_cam1_compare.py` as an unclassified share writer.
  - That file is not touched here, so the failure comes from the base branch.

## Real-data check (read-only)

Script and output: scratchpad `c1guard_verify/verify.py`, `verify_3sessions.csv`. It compares
`trial_classes.session_trials` / `cue_classes_for` / `stop_kind_for` and the first LP lick
(`first_lick_latency.session_latency`). Code at 5a43b845 is compared with this branch, and the result against the
audit's `blast_c1far/sessions.csv`.

| session | raw class changes | transitions | session-level changes | working lost | stop-kind | first LP lick changed (-> none) | vs audit |
|---|---:|---|---:|---:|---:|---:|---|
| PS94_0823 (acute) | 6 | incomplete>jaw_only x3, >no_response x3 | 11 | 6 | 8 | 8 (6) | all 11 columns match |
| PS92_0929 (chronic) | 0 | | 0 | 0 | 0 | 0 | match |
| PS95_0924 (chronic) | 4 | incomplete>jaw_only x3, >no_response x1 | 5 | 4 | 2 | 5 (4) | match |

The quiet-chain lick train (`cohort2_qc/lp_licks.lick_times`) drops 106 / 324 / 22 cam1-only onsets in these sessions.
That equals the audit's far events_all counts, and the result is now the same cam1 set as RESTnlc1.

## 11-17 late: digest alias dropped

The `_DIGEST_ALIAS` in `restnl_licks.code_digest` failed in the full pre-push suite (passed alone and in a fresh
interpreter here) and was dropped as fragile. Consequence: RESTnlc1 caches written by the v3 code before the keep_mask
refactor (the 11-17 re-run, which imports wf_restnlc1) recompute once when a later run uses this code. The test now
checks only that the digest follows keep_mask's closure.
