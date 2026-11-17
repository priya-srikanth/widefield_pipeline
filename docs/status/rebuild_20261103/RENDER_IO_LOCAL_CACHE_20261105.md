# Render I/O and a local input cache: measurement, 2026-11-05 22:35-22:55

Read-only measurement of the `_daqlk` multi-lane render (worktree `wf_render_daqlk` @ a76352f3, queue
`render_daqlk/queue_daqlk6.sh`). No process was touched, no data copied, no code edited. The question was whether copying
the hot inputs from M: to local disk would speed up the remaining queue, and how big that copy would be.

## 0. Answer in brief

* **Most of the render already reads from local disk.** Every queue step exports `WFIELD_INPUT_CACHE` =
  `C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104`. `combined_encoding.load_imaging` and `load_behaviour`
  (used by the encoder, imaging-map and ITI families, 77 of the remaining steps) already hit it. I recomputed the imaging
  key from a stat of the sources for PS92_0806 and PS95_1005, and both are hits.
* **The link was saturated at 22:36 (107 MB/s of a 1 GbE line), but partly by a job outside the queue.**
  `scripts/trial_attempts.py` was running from worktree `wf_decisions2` (started 22:22, 3 workers, about 50 MB/s of
  process reads). Once it finished (by 22:54), the queue alone drew **about 55 MB/s, half the link**, with CPU at 64 %.
* **The busy steps are compute-bound per process.** Over 20 s, the workers of the encoder, svf, relabel and epoch steps
  each used 90-160 % of a core. Wall time is set by the number of workers the RAM allows times single-worker compute,
  not by the network. The low total CPU (20-68 % of 24 cores) comes from the RAM limit (MINFREE 10 GB / FLOOR 6 GB;
  there was a LOW-RAM STOP at 22:23), not from I/O waits.
* **What is not cached:** the per-session result cache `WIDEFIELD_SESSION_CACHE` (on M:, set by the profile), the DAQ h5
  read directly by about 37 `h5py.File` call sites, the FaceRhythm projection npz, the pose3d frame tables, the LP
  tables, and the videos.
* **Expected speed-up of a new local cache: small.** Perhaps 10-30 % on the 28 LocaNMF / session-cache / DAQ steps
  (section L below), none on the encoder and imaging steps (already local), and **under about 10 % on overall wall
  clock**. The larger levers are: don't run other I/O-heavy jobs beside the render; keep RAM headroom for more workers;
  and x_enl_move, which is decode-bound and has about 18 h left.
* **The input cache is growing about 6-10 GB/h on C: (87 GB free), and the queue has no disk watchdog** (MINFREE/FLOOR
  are RAM). This is a risk whether or not anything new is added (section 4).

## 1. Read throughput per running step (Win32_Process ReadTransferCount, 60.6 s window from 22:36:25)

ReadTransferCount counts all reads: network, local disk and RAM page-cache hits. Over the same window the NIC received
**106.9 MB/s**, C: physical reads were **5.2 MB/s**, and the summed process reads were **182 MB/s**. So about 70 MB/s
was served from RAM: the 9.5 GB standby list, SMB-cached files, and pickles that were just written.

| step (Windows PID tree) | MB/s, parent + workers | read ops/s | cumulative GB at 22:35 | notes |
|---|---|---|---|---|
| g5_epoch (6944 -> arm 38048 `--arm lick` + 2 workers; arm 15428 `cuelick`) | 53 (3.5 + 25.9 + 23.9) | 4 per worker = ~6 MB per op | 44 + 2.7 + 2.6 | large pickle loads: session_cache prm/bm entries on M: (40 / 16 MB). Workers also wrote 3.4 MB/s each |
| *trial_attempts.py (NOT a queue step; wf_decisions2, 14652 + 3 workers)* | *50.5* | 2600-5000 | *65* | DAQ h5 / LP tables from M:; finished by 22:54 |
| t_frame_te (21448 + 3 workers) | 33.8 | 3000 per worker | 100 | 72/107 sessions at 22:38, about 1.4 GB of process reads per session, mostly local imaging/behaviour pickles |
| d_attempt_lp (10188) | 15.9 | 1900 | 37.7 | 80 sessions in 84 min, about 0.47 GB per session |
| a_svf_ls (36848 + 2 workers) | 14.8 | 2600 | 157 | 53/107; about 3 GB of process reads per session (local pickles re-read) |
| a_relabel_noFR (33900 + 2 workers) | 9.4 | 58 | 345 | 94/107; running since 17:52 |
| x_enl_move (36140 + pool of 3, re-spawned per session) | 2.3 (30.8 in an earlier 24 s window) | 73 | 40 (parent) | **12 of 112 sessions in 2 h 12 min (about 11 min each), so about 18 h left**; video decode, CPU-bound |
| e_iti_key (20464) | 0.1 | 103 | 5 | CPU |
| pytest runs (3), count15.py (other sessions) | 1.3 | | | not the queue |

Total network receive: 106.9 MB/s at 22:36 (link is 1 Gbps, about 117 MB/s usable), 71.8 MB/s at 22:40, and
**54.7 MB/s at 22:54 after trial_attempts ended**. CPU total: 68 % (22:40) and 64 % (22:54).

Per-process CPU over 20 s (22:42): t_frame_te workers 108-162 %, a_svf_ls workers 111-117 %, a_relabel workers 89-111 %,
g5 arm 91 %, d_attempt_lp 70 %, e_iti_key 74 % (100 % = one core). These are mostly compute-bound, with a small I/O share
for d_attempt_lp and e_iti_key.

## 2. What each loader reads, and which loaders use the local cache

| loader | reads from M: | local cache? |
|---|---|---|
| `combined_encoding.load_imaging` (`_load_imaging`: hemo-variant SVTcorr, U_atlas, atlas masks, DAQ h5 pco edges, cleanpairs frame map) | about 0.6 GB/session on a miss | **YES** via `WFIELD_INPUT_CACHE` (`_cached`, key = loader source + stat of sources). Current key verified as a hit |
| `combined_encoding.load_behaviour` (orofacial LP session dir, DAQ h5, trials csv, attempts table, pose3d frames + 3-D licks table under TONGUE=3d) | about 0.3-0.4 GB/session on a miss | **YES**, keyed on every `COMBINED_ENCODING_*` / `OROFACIAL_*` env value + 7 module sources + source mtimes. Each new env combination is a new 130-190 MB entry per session (section 4) |
| `frozen_encoder.session_design` (svf, partial_refit, relabel, allsession, layer2, residual_roi, specificity) | the two above, plus `_fr_columns` (FaceRhythm proj npz, about 26 MB/session, frA only), plus `licks.csv` / `licks_all.csv` from the LP dir, plus whole-cohort 3-D tables (`EXEC3D` = local snapshot) | imaging/behaviour yes; FR and LP csvs **no** |
| `locanmf_session.load_C` (WIDEFIELD_LOCANMF_PROJECT=1) | on a miss: U_atlas + stored A + SVTcorr (about 380 MB/session), then writes projected C (about 115 MB) | **no**: goes to `session_cache` on **M:** (`WIDEFIELD_SESSION_CACHE`). Only 12 entries exist (LP-tagged, written by x_enl_move); the kind carries the LP tag, so the DAQ and LP arms each compute their own |
| `session_cache.cached` (tf-* trial features, prm-*, bm-*, jointvc) used by locanmf_position_decoder / cd_trajectories / epoch_grant / attempt_* / nolick / joint_xsession | the pickles themselves (on M:), plus C + h5 on a miss | **no**: the store is on M: (`M:/.../rebuild_20261103/session_cache`) |
| DAQ h5 direct (`plot_spout_trial_averages._load_daq_events` reads the whole packed digital array + the lick analog channel; `behavior_events.session_speed`; lp_licks; about 37 `h5py.File` call sites) | large part of 235 MB/session per call | **no** |
| `frames_average.npy` (iti_hemo_check, iti_f0_check, iti_check_figures) | 3.5 MB/session | no (negligible) |
| pose3d `<a>_<d>_frames.csv.gz` (within_lick_correction, face_prelaunch_aim, iti_first_lick, lick_covariate_reference) | 67 MB/session | no |
| videos (x_enl_move: cam1-4 `.avi`, ENL windows only, via seeks) | up to 34 GB/session on disk; only windows decoded | no (and not cacheable: 3.6 TB) |

Other existing users of `WFIELD_INPUT_CACHE`: `frozen_encoder` exec-bins calibration table
(`<cache>/pose3d_inputs/exec_bins_calib_20261023.csv`), `scripts/gate_lp_check.py` (`<cache>/gate_lp_check`), the
cohort-2 qc scripts, and o2_bundle/o2_encoding (treated as local-only, not recorded).

### Remaining steps grouped by input (234 not done, of which 8 are running or claimed)

| family | steps | per-session network today |
|---|---|---|
| E: encoder session designs (frozen_encoder frames, svf, partial, relabel, allsession, layer2, residual_roi, specificity) | 26 (4 running) | small: imaging and behaviour are local; FR npz plus LP csvs about 30-60 MB/session; plus a behaviour rebuild (about 0.35 GB) for each new env combination |
| I: imaging maps / ITI extracts via `load_imaging` (b_pix, b_rsig x8, pixel/r2/spout maps, z_pixmusall x4, e_iti_*, lick_roi_*, e_pat_*) | 51 | about 0 (local imaging pickle) |
| L: LocaNMF / session_cache / DAQ (g5_* x21, g6_grant, d_attempt_* x6) | 28 (g5_epoch, d_attempt_lp running) | **the one family still on M:**: session_cache pickles 2-60 MB/session/arm, plus h5, plus projected C on misses; measured about 0.47 GB/session (d_attempt_lp) |
| P: pose3d / FR tables only | 10 | 67-90 MB/session |
| V: video (x_enl_move) | 1 (running) | window seeks into 4 videos; decode-bound |
| A: aggregate outputs only (figures, ITI readers, dock, lmm, ...) | 118 | negligible (summary CSV/JSON) |

The grouping is by grep of each script's own loader calls. A few "A" scripts (e.g. b_matched_residual, b_posinfo) may
read per-session tables through helpers; that does not change the conclusion.

## 3. Sizes (os.stat only, 107 rebuild sessions = the frA_ls_basis frame's session list)

| input | total | per session | used by |
|---|---|---|---|
| SVTcorr `hemo_meegkit_hpfit_lag_refitT_swapmask` | 9.15 GB | 85.5 MB | imaging (cached), projection |
| U_atlas (`allen_aligned_affine8v1`) | 14.79 GB | 138 MB | imaging (cached), projection |
| U.npy (native) | 9.45 GB | 88 MB | only stat-ed (session_cache signature) |
| atlas masks (2 files) | 0.18 GB | | imaging (cached) |
| LocaNMF A (stored, `locanmf_affine8v1_hemo_meegkit_hpfit`) | 17.09 GB | 160 MB | projection |
| LocaNMF C (stored) | 10.60 GB | 99 MB | only stat-ed under PROJECT=1 |
| frame maps (cleanpairs) | 0.77 GB | | imaging (cached) |
| frames_average | 0.38 GB | 3.5 MB | ITI hemo/f0 |
| **DAQ h5** | **25.15 GB** | 235 MB | behaviour builds, L family, session_speed, lp_licks |
| orofacial LP session dirs (12 files/session) | 11.62 GB | 109 MB | behaviour builds, encoder licks csv |
| pose3d frames csv.gz | 7.19 GB | 67 MB | P family, behaviour builds |
| FaceRhythm proj npz (cam4 master_pre_v1) | 2.83 GB | 26 MB | frA encoder steps |
| behavior_out session files (trials csv, events) | 0.05 GB | | everything (tiny) |
| **M: session_cache, these 107 labels**: prm 12.79 GB (321), bm 5.27 GB (321), tf 3.01 GB (1881), lnmfproj 1.40 GB (12), jointvc ~0 | **22.5 GB** | | L family |
| videos .avi, 4 cams | 3633 GB | 8.4 GB/file | x_enl_move only |

Existing local input cache (C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104): imaging 34 GB (215 entries:
the current key set from 11-04 20-21 h plus a superseded 11-04 08 h set of about 17 GB), behaviour 154 GB (990 entries,
7-15 keys per session).

## 4. Local disk

* C: 953 GB, **87 GB free** (22:52). F: is optical. M: and N: are network shares. There is no second local data drive.
* Input-cache growth: behaviour entries written since 19:00 = 232 (38.7 GB); since 21:00 = 74 (12.2 GB). That is
  **about 6-10 GB/h**, from steps whose `COMBINED_ENCODING_*` / `OROFACIAL_*` env combination is new (each costs about
  0.35 GB of network on the miss).
* At that rate C: fills in about 9-14 h of rendering, and the queue only watches RAM. Reclaimable without loss (derived,
  disposable by design; **ask Priya first**): the superseded 11-04 08 h imaging set (about 17 GB, verified not to be the
  current key for 2 sessions) and behaviour keys from earlier env combinations (most of the 154 GB).

## 5. Estimate

**How often each uncached input is read in the remaining queue:**
* session_cache pickles: about 15 heavy L steps x 1-3 arms x 107 sessions, each loading its prm/bm/tf entry (2-60 MB):
  about 100-250 GB of M: reads.
* DAQ h5: the L steps that miss or recompute trial features, plus lp_licks/gate, plus session_speed, plus behaviour
  misses. Roughly 10-20 reads x 107 x about 0.1-0.2 GB effective (the digital array + lick channel): **about 150-400 GB**.
* Projected C: computed (380 MB/session) only on a miss under a new lick-source tag. About 2 more passes (DAQ and LP
  attempt arms): about 80 GB.
* FR npz: about 10 frA encoder steps x 2.8 GB = 28 GB.
* pose3d frames: about 6 x 7.2 GB = 43 GB.
* LP tables: about 10-20 partial reads x 11.6 GB, which is under 100 GB.
* **Total uncached remaining: about 0.4-0.9 TB, i.e. 1-2.5 h of a fully saturated 1 GbE link, spread over the more than
  15 h the queue still needs** (x_enl_move alone has about 18 h left).

**A one-time copy of the hot uncached set:** session_cache current-tag entries 22.5 GB + DAQ h5 25.2 GB = **48 GB**,
about 8 min at the full link or 15-25 min alongside the render. Adding FR + pose3d + LP tables brings it to 69 GB, which
is too much for 87 GB free plus 6-10 GB/h of growth.

**Expected speed-up:** the L-family steps would lose most of their M: reads. They are partly I/O-bound (d_attempt_lp at
70 % of a core, g5 workers doing 6 MB pickle loads), so they would run perhaps 10-30 % faster. E and I steps would not
change, because they already read local pickles and are compute-bound (100-160 % of a core per worker). x_enl_move is
decode-bound and video cannot be cached. **Net wall-clock gain on the remaining queue: probably under 10 %.** The same or
more is gained, with no code change, by not running other I/O-heavy jobs beside the render: trial_attempts took about
half the link for 30 min.

## 6. The smallest change that captures most of the gain

1. **No code change: point `WIDEFIELD_SESSION_CACHE` at a local copy.** The variable already exists
   (session_cache `_default_cache_dir`). The profile exports it as M:, so the override must come after the profile eval.
   Add an env token `WIDEFIELD_SESSION_CACHE=C:/Users/SabatiniLab/wfield_cache/session_cache_rebuild_20261103` to the L
   steps (or to `profile_env`, next to `WFIELD_INPUT_CACHE`) in a **new queue file**. Pre-seed it with
   `robocopy <M: session_cache> <local> /E /COPY:DAT /XO` (22.5 GB for these labels; the whole dir is 2803 entries) and
   copy back with `/XO` when the render ends. The fence allows it: a local path that is not a share, not
   figures_working, and not the default `.widefield_session_cache` (`rebuild_root.write_refusal`). `writeguard` already
   exempts whatever `session_cache.CACHE_DIR` is. This removes the session_cache pickle reads and projected-C reads, and
   with them most h5 reads in L steps that hit their tf/prm/bm entries.
2. **Small code change, only if the h5 reads still matter after (1): a DAQ-h5 read mirror.** About 37 `h5py.File` call
   sites make per-site edits large. The one-place version is a wrapper installed at import (like the fence audit hook in
   `wfield_local/__init__.py`). When `WFIELD_DAQ_MIRROR` is set, it rewrites a read-only open of
   `.../DAQ_recorder_output/<date>/<file>.h5` to the mirror copy if that copy has the same size and mtime (robocopy
   `/COPY:DAT` keeps mtime). Callers keep passing and stat-ing the M: path, so **no cache key changes**: the imaging key
   includes the h5 *path string*; session_cache signatures use mtime and size. The wrapper must NOT go into
   `combined_encoding.py`, `orofacial_*`, `trial_windows.py`, `beta_maps.py`, `daq_trials.py` or
   `pose_kinematics_demo.py`. Those files' source is in the input-cache key (`_CACHE_CODE` / the imaging loader hash), so
   editing them would invalidate all 154 GB of behaviour entries and force about 107 x 0.35 GB of re-reads per env
   combination. About 25 lines plus a test.
   Not recommended mid-render: it needs a new commit, a new worktree and an `EXPECT_SHA` change.

## 7. Recommendation and risks

**Do not change anything under the running steps.** Bash reads a queue script as it runs, so editing
`queue_daqlk6.sh` in place is unsafe. The pinned worktree must stay at a76352f3 while any step from it runs.

If Priya wants the gain:
* **Option (1), no code, at a step boundary after the current L steps (g5_epoch, d_attempt_lp) finish:** write
  `queue_daqlk7.sh` with the session-cache token, seed the local copy (48 GB with the h5 if (2) follows, 22.5 GB without),
  touch STOP, and relaunch the lanes from v7 (resume works from `.done` files; claims prevent double runs).
* **Do the disk housekeeping first** (ask before deleting): free the superseded imaging set (about 17 GB) and old
  behaviour keys, and add a disk-free check to the queue's watchdog. Without that, even today's queue can fill C:
  overnight.
* **Cheapest win now:** keep other I/O-heavy jobs (trial_attempts-style rebuilds, big pytest runs) off this box or off
  M: while lanes run, and keep RAM headroom so lanes can run more workers. Those are the actual limits.

**Risks of the session-cache redirect:**
* **Correctness:** entries are content-keyed (stat of SVTcorr / U / C / h5 + params + CACHE_VERSION + run-mode/LP/gate
  tags in the kind) and the copy is byte-for-byte, so the same bytes are served. Stats still go to the M: sources, so a
  changed input still misses.
* **Split store:** entries computed after the switch exist only locally until copied back. Other lanes, O2 and other
  worktrees that still point at M: will recompute them (correct, but it costs time). The `_store` prune of superseded
  signatures then acts per directory: harmless.
* **Copy race:** seeding while g5 / d_attempt write and prune entries can miss the newest ones. The cost is a recompute,
  never a wrong value. `/XO` copy-back must not overwrite newer M: entries.
* **Disk space:** 22.5-48 GB on top of 87 GB free, with 6-10 GB/h of input-cache growth.
* **Write fence:** the local path is outside the rebuild root but outside every fenced tree, so it is allowed. Do not put
  it under `C:/Users/sabatini/...figures_working` siblings (the default `.widefield_session_cache` is fenced).
* **Mid-render code:** option (1) needs none. Option (2) needs a new SHA, a new worktree and a new queue version, so
  mixing it with steps already rendered at a76352f3 must be recorded in the render-queue doc.

Measurement scripts (stat-only): scratchpad `io_measure/sizes.py`, `io_measure/key.py`, `io_measure/steps.tsv`.
