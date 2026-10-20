# Handoff — 2026-10-20: encoding model (3-D executed sets, drop-p per-spout maps, recovery timing, slow band)

**START HERE for the encoding model / imaging x orofacial work.** Supersedes
`STATUS_2026-10-19_EVENING_ENCODING_HANDOFF.md` (still the reference for the O2 path, the 10-19 runs and their numbers).
**Results live in `docs/ENCODING_FINDINGS.md`** (living summary with status tags — update it when a result changes).
Reasoning and numbers: `DECISIONS.md` 2026-10-19 (night) ... 2026-10-20. Scratchpad (launchers, logs, snapshots):
`C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`.

## 0. Rules (Priya)

- **Curated sessions only** (10-20): `analysis_kit.curated_labels()` — no early June (0601-0605), no 0805, no
  PS92 / PS93 0817 (recorded after the initial lesion that was "too mild", Priya 10-20; not curated). June 0606-0608 are curated but NOT baseline by default (sensitivity only, say so). August-only pre baseline is
  primary. Check any new session list against `curated_labels()` before running.
- Lesions all LEFT; contra = mouse's RIGHT; brain maps image left = left hemisphere (ipsilesional); cam1 face video
  ipsi = image right. `*_L` spouts = mouse's left (verified 10-20, both rings).
- No video in the lick window for target / executed. Lead with drop dR2. Per-animal tables before pooled (rule 8).
- Priya runs O2, transfer-node and process-stopping commands; give copyable commands; stop processes by command line
  (`taskkill /PID <pid> /T /F` via `Get-CimInstance Win32_Process` match). Write only under `MICROSCOPE/Priya`. Stage
  files by name (other windows commit here). Commit + push; DECISIONS entries.

## 1. Where things stand (details + numbers: ENCODING_FINDINGS.md)

| finding | status |
|---|---|
| post-stroke rise of unique executed (cam4, 4/4; survives lick-matching) | STANDS |
| chronic position-coding gain real in PS92 / PS93 / PS95 (matched); beyond the pre-stroke practice trend | STANDS |
| chronic gain LAGS behavioural recovery ~1 week (PS92 9 vs 15, PS93 7 vs 15; PS94 recovers partly without it) | PROVISIONAL |
| pre-stroke "cortex tracks target, not tongue" | **SUSPENDED**: PS92 coordinate-matched 3-D (xyz2) cortex target 8.0 vs executed 11.7 (cam4 31.7 vs 10.8); MOs executed-leaning; trial-to-trial MOs executed 12.6 vs -1.7 |
| trial-to-trial (cam4): no general post-stroke executed shift; PS94 chronic exception | STANDS (cam4-limited) |
| drop-p per-spout: acute far contra -> ~0 in 4/4; chronic gain = PS92 / PS93 far mid | STANDS (PS92 chronic pending 0908 rerun) |
| signed maps: close spouts crossed beyond movement; far-spout crossed activity shared with lick direction (drop-p's uncrossed far residual is NOT biology) | STANDS |
| misdirected licks: no plan signature | STANDS |
| pre-cue slow band: rescales pre-cue decode ~.03-.09 in every epoch; PS93 acute pre-cue drop = slow band | DONE (size caveat) |
| binned contest (`COMBINED_ENCODING_BINS`) + within-spout deviation check | coded + tested; within-spout cam4 cohort RUNNING (`scratchpad\within_spout.log`) |

## 2. Decisions this window (all in DECISIONS)

- **3-D executed sets** (`COMBINED_ENCODING_EXEC`): `cam4m` (cam4 set on 3-D-matched licks), `3d` (angle out of the
  midline plane, angle^2, 3-D reach), `3dxyz` RETIRED (linear executed vs angular target = coordinate mismatch),
  `xyz2` (tongue-peak 3-D ml / ap / dv vs spout 3-D ml / ap / dv), **`polar`** (both sides: lateral angle, its square,
  forward tilt, 3-D distance from the mouth zero; 4 per side — NOT tilt^2: 5 target terms would saturate the 5-dof
  six-spout space and turn the contest into the strict one). Inputs: snapshot of pose3d `kinematics/licks.csv.gz`
  (`EXEC3D_TABLE_ENV`) and per-session spout 3-D medians (`TARGET3D_TABLE_ENV`), both in the scratchpad.
- **Strict executed** model `nokin:category+exec` (executed beyond all six spout kernels = lick-to-lick); readouts
  `u:exec|category` (region table) and `u_exec_strict` (trial-to-trial table).
- **Combined lick events** (`COMBINED_ENCODING_LICKS=combined|lips`): cam1-only events (not cam4-matched, not droplets)
  join the lick TIMING kernel; `lips` gives lips-class events their own kernel. Timing only (no angle / 3-D depth).
- **DROP-p** per-spout model (`ONLY=droppos-`): spout p constrained to the trial-weighted mean of the others (deleting
  p's block loses nothing — the six deviation blocks sum to zero). Replaces one-vs-rest relabelling (`shufpos-`, which
  mislabels donor trials) as the per-spout measure.
- **Non-linear model DEFERRED** (only as a ceiling check later; attribution needs comparable model classes).
- **v36 timing fix** (other window): only PS92_0908 (+ June PS92_0606 / PS93_0606) affected in the encoding sets.

## 3. RUNNING / PENDING at handoff (10-20 ~12:30)

| job | where | then |
|---|---|---|
| slow-band re-split, all curated sessions (`slowband_decode_check.py --all`, started under the OLD list -> includes PS92_0817 / PS93_0817, the "stroke too mild" sessions) | box, task log `scratchpad\slowband_all.log` | when done: re-run `--all` once (reuses rows, writes the curated-only CSV), read adopted vs fast300 pre-cue by animal x epoch, append result to DECISIONS 2026-10-20 slow-band entry and ENCODING_FINDINGS section 5 |
| other window: v36 per-session rerun DONE (10-20 pm); cohort products (Branch A) + 3-D chain / kinematics / cam1 licks (Branch B) still running | box | per-session licks.csv OK to read; WAIT for Branch B before refreshing the 3-D snapshot tables (polar pilot now also sets `COMBINED_ENCODING_BINS=1`) and for Branch A before cohort tables |
| cam1 LP production on O2 (array 56830841; tasks 62-137 pending at last check) | O2 | rsync -> `o2_inference collect` (other window) -> 3-D chain -> cohort 3-D contest |
| O2 fair-share 0.013 (10-20) | — | submit nothing new to O2; box runs |

## 4. To-do (priority)

1. **After the v36 rerun:** (a) rerun PS92_0908 in every `imaging_encoding` folder (movement_first, subset-nokin,
   -nokin_matched [MATCH=bundle match.json], -nokin_tr [SUFFIX _tr], -nodir, -trmodels, musall_subset-shuf,
   -shufpos-, -droppos- [1 worker, ~40-60 GB peak]) and June PS92_0606 / PS93_0606 in movement_first + subset-nokin;
   re-render; refresh per-spout table. (b) `run_polar_pilot.ps1` (PS92 11 sessions, 2 workers, ~2 h) -> readout with
   `scratchpad\exec3d_readout.py` (add 'polar' to its mode lists) and `scripts/exec3d_pilot_figure.py` (swap 3dxyz for
   polar in MODES). (c) smoke-test `COMBINED_ENCODING_LICKS=combined|lips` on 1-2 sessions.
2. **Cohort 3-D contest** when cam1 production + 3-D chain land: xyz2 + polar, strict executed, combined licks; needs a
   spout-position table for all sessions (`scratchpad\spout3d_table.py` does PS92; generalise) and a fresh kinematics
   snapshot. The deciding analysis for target vs executed.
3. Read the within-spout cam4 cohort (`qc/within_spout_deviation/`); binned contest on PS92 polar
   (`COMBINED_ENCODING_EXEC=polar COMBINED_ENCODING_BINS=1 ... --only nokin:`), then in the cohort 3-D contest.
4. Pre-cue -> lick-window carry-over decode: cross-window decode (train pre-cue, test lick window vs within-window),
   built on the CD-trajectory code (`STATUS_2026-09-24_CD_TRAJECTORIES.md`; mind its one-vs-rest artefact).
5. Move the main model's tongue traces to the 3-D track (adds forward; includes cam1 frames).
6. Deferred: non-linear ceiling check. Older: orofacial findings rewrite on v36; touch-free contest.

## 5. Pitfalls (new this window)

- **Heredocs mangle Python string escapes** (`\n` becomes a real newline) — happened twice more today; patch Python
  with the Edit tool, never `sed` / heredoc. Pre-commit ruff catches it; nothing bad was committed.
- **Machine path literals fail the test suite** (`test_no_hardcoded_machine_paths`): resolve via `PathResolver().root()`.
- **Never read licks.csv / pose3d tables while another window rewrites them** (truncated / mixed inputs).
- **O2 memory per tag**: nokin / matched / trmodels 24 GB, shufpos 32 GB, **droppos 64 GB**; `-p short` only (< 12 h
  jobs; `medium` is rejected). Resubmit failures: `IDS=$(sacct -j <job> -X -n -o JobID%20,State | awk '$2 !~
  /COMPLETED|RUNNING|PENDING/ {split($1,a,"_"); print a[2]}' | paste -sd, -)`. Scheduler-cancelled tasks (3 s) are not
  failures: resubmit.
- **Box RAM**: an encoding worker peaks ~20 GB (drop-p more); 4 + 4 + 2 workers paged the box to a halt on 10-19.
- **O2 `collect` folder names keep the trailing dash** of `ONLY=shufpos-` / `droppos-` (`movement_first_subset-droppos-`);
  figures route them.
- **Ranks in tests**: a kernel's effective rank < its columns (basis redundancy); test ranks against the shared
  kernel's rank, not column counts.
- **Coordinate systems must match in a contest** (3dxyz lesson); and > 4 target terms saturate the six-spout space.
- **"After movement"** = after licks, tongue, jaw, nose, running (cam4 kinematics; no video). Shared variance goes to
  movement in movement-first; order-free drop credits it to nobody.

## 6. Commands

```bash
# readouts
python scripts/combined_encoding_figures.py --order movement_first_subset-droppos-        # per-spout + share maps
python scripts/combined_encoding_figures.py --order movement_first --signed-only           # signed per-spout maps
python scripts/combined_encoding_figures.py --order movement_first_subset-nokin_exec-xyz2  # contest + trial-to-trial
PYTHONPATH=. python scripts/exec3d_pilot_figure.py                                         # cam4 vs 3-D side by side
PYTHONPATH=. python scripts/position_coding_trend_recovery.py                              # trend + lead/lag
PYTHONPATH=. python scripts/slowband_caveat_figure.py ; python scripts/slowband_decode_check.py --all
python -m pytest -q tests/test_combined_encoding_options.py                                # 15 checks, ~1 s
# 3-D / lick options (box)
COMBINED_ENCODING_EXEC=polar COMBINED_ENCODING_EXEC3D_TABLE=<snapshot> COMBINED_ENCODING_TARGET3D_TABLE=<spouts> \
  python -m wfield_local.combined_encoding --jobs 2 --only nokin: --skip-done <labels>
COMBINED_ENCODING_LICKS=combined python -m wfield_local.combined_encoding --jobs 2 --only nokin: --skip-done <labels>
```
