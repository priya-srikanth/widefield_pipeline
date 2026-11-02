# RESTnl map shortcut: read-only check (2026-11-01)

This checks the map section of `RESTNL_ROLLOUT_PLAN_20261101.md`, §3 Option A and check (i). Everything here was read-only:
no code was changed in either repo, nothing was rendered, nothing was written on M:, and nothing was committed. The scripts
live only in the session scratchpad. They load each session's SVT and U_atlas, and its DAQ h5, licks_all.csv and quiet
mask, all read-only. They call the `widefield_pipeline` `main` functions in memory. The strict-mask helpers (`widen`,
`floor_runs`, `_daq_lick_onsets`) are imported from `wf_attempt/scripts/restw_strict_check.py`; its `session_worker`,
which writes npz files, was not called.

Sessions checked: the cue/working contact arm (`prm-cue-working-3b1e0fc8b886`) for PS92, PS93, PS94 and PS95, one pre
(0814) and one acute session each (PS92/93 0819, PS94/95 0818), so 8 sessions. The cross-arm check used PS92_0814 and
PS94_0818 in all 4 arms listed below.

## VERDICT

1. **The shortcut is valid.** On every map arm, a flat RESTnl map is exactly `cached raw[q] - U @ b_nl`. No
   re-extraction is needed, and the warm `prm` caches stay usable.
2. **A flat baseline is appropriate, with a caveat.** Time-local tracking changes the referenced maps by 1-5 % of the
   map RMS in most sessions, and by up to ~15 % where the acute far-contra map is near zero. The bias does not grow
   acute vs pre, and its sign does not follow session time. The caveat: an acute far-contra amplitude that is close to
   zero carries a baseline uncertainty of about ±0.5e-3. A time-local lick-free version is also not feasible: its 12 bins
   would hold 0-85 frames in the thin sessions.

## 1. What a `prm-*` cache entry holds (position_reference_maps.session_raw_maps)

Each entry is a dict with these fields:

| field | contents |
|---|---|
| `raw` | Per position: `U @ mean over trials and bins of the SVT window features`, with no reference subtracted (`baseline="none"`). Absolute pixel maps on the Allen grid. |
| `raw_rest` | Already referenced to page REST. The time-local baseline is subtracted from the SVT before `trial_features`. |
| `raw_restw` | Already referenced to RESTW. The flat position-weighted baseline is subtracted from the SVT. |
| `raw_precue` | Already referenced to each trial's own 1 s pre-cue mean (`baseline="precue"`). Empty for the precue alignment. |
| `trial_mean` | The MEAN reference. `reference_maps` subtracts it after loading. |
| `quiet_flat` | The retired flat session-mean map. |
| `used` / `used_rest` / `used_restw` / `used_precue` | Trial counts per position. |
| `restw_positions` | Positions that contributed to RESTW. |

How the two rest references are built:

- **REST, the page's reference.** It uses `session_rest_svt_timelocal`: the quiet-frame mask AND
  `engaged_frame_mask`, split into 12 equal session-time bins, the per-component median in each bin, interpolated to
  every frame. The result is a (K, T) baseline subtracted from the SVT, so every trial is referenced at its own time.
  Averaged over each position's trials, it becomes a different map per position. **It cannot be applied after the
  fact.**
- **RESTW.** It uses `session_restw_svt` -> `rest_by_position.rest_frames_by_position(docked=False,
  engaged_only=True)` -> `restw_from_frames`. That is the per-position median of each SVT component over that
  position's rest frames, with a 200-frame floor and all 6 positions required, then the equal-weight mean. The result
  is one flat (K,) vector. Window features are linear means, so `raw[q] - raw_restw[q] = U @ b_w` for every `q`.

## 2. Numerical verification on the warm caches

### RESTW is one flat map

`max |(raw - raw_restw)[q] - (raw - raw_restw)[q0]|` over all six positions, where `|b|max` is the largest absolute
value of the baseline map:

| session | epoch | max deviation across q | \|b\|max | relative |
|---|---|---|---|---|
| PS92_0814 | pre | 4.5e-08 | 0.104 | 4e-07 |
| PS92_0819 | acute | 6.0e-08 | 0.019 | 3e-06 |
| PS94_0814 | pre | 2.5e-08 | 0.026 | 1e-06 |
| PS94_0818 | acute | 1.1e-07 | 0.105 | 1e-06 |
| PS93 / PS95 0814, 0818/0819 | pre / acute | 1e-08 to 7e-08 | - | float32 |

- **The trial sets are identical.** `used == used_restw` in every session checked.
- **The maps rebuild from source.** I recomputed `U @ session_restw_svt(s, SVT)` from the session's SVT and U_atlas.
  It matches the cached `raw - raw_restw` to 1e-08 to 7e-08 (float32) in all 8 sessions. So the projected flat baseline
  is exactly the column the cache carries.
- **The baseline map is the same in every arm and alignment.** I checked PS92_0814 and PS94_0818 in 4 arms:
  `cue-working` contact, `cue-working-lp90d0a4f6`, `lick-lick` and `precue-working-9c0a7ab373bd`. The RESTW baseline
  map is the same to ≤ 7.5e-08, and it is flat within each arm to ≤ 1.3e-07. So **one RESTnl baseline map per session
  serves all 7 `prm` kinds** (cue / lick / precue x contact / LP).

### The time-local REST is not one map

`raw - raw_rest` is a different map for each position. Its position-dependent part is 0.06-0.85e-3 RMS over the brain
(max |.| 0.4-4e-3). That is the time-local baseline averaged at each position's trial times. You would need to
re-extract to change it.

## 3. What a RESTnl baseline map needs (Option A)

Per session, built in memory and validated here on 8 sessions:

1. **The lick-free rest sample mask.** This is the `restw_strict_check` strict mask: `floor_runs(rest0 &
   ~widen(DAQ lick onsets, [0.5, 2.0] s) & ~widen(cam4 licks_all daq_on_s, [0.5, 2.0] s), min_rest_s)`. `rest0` is
   the restdock05 `*quiet_sample.npy` from `rbp._session_daq`. The DAQ licks come from the mask's own `detect_licks`
   config criteria. The cam4 licks come from `orofacial_outputs.session_dir(PathResolver(), orofacial_cohort.RUN, an,
   date)/licks_all.csv`.
2. **The frames.** Call `rest_frames_by_position(s, T, docked=False, engaged_only=True)` with `_session_daq` returning
   the strict mask and `_engaged_trials` returning the DAQ gate. This is the same in-process substitution
   `restw_strict_check` uses. Here it was done with try/finally restore.
3. **The wr level on the SVT.** For each position with ≥ 50 frames, take the per-component median `np.median(V[:,
   frames], 1)`. Take the equal-weight mean over those positions, renormalised. The session is OK only if ≥ 4 of 6
   positions qualify. All 8 sessions passed. Note that PS92_0814 had only 4 positions with ≥ 50 frames
   ([269, 147, 64, 178, 59, 483]), so it is marginal.
4. **The projection.** `base_map = (U_atlas_flat @ b_nl).reshape(MAP_SHAPE)`, with U from `joint_basis._load_session`.
   That is the same U, grid and NaN zeroing as the maps, so it uses the same pixel mask automatically. RESTnl[q] =
   `raw[q] - base_map`.

How far RESTnl moves from RESTW (`base_nl - base_w`, RMS over brain pixels, x1e-3):

| session | epoch | RMS | p99 |
|---|---|---|---|
| PS92_0814 | pre | 2.67 | 7.45 |
| PS92_0819 | acute | 1.05 | 3.61 |
| PS93_0814 | pre | 2.00 | 7.09 |
| PS93_0819 | acute | 3.54 | 12.59 |
| PS94_0814 | pre | 1.75 | 4.27 |
| PS94_0818 | acute | 1.95 | 5.21 |
| PS95_0814 | pre | 0.21 | 0.68 |
| PS95_0818 | acute | 0.75 | 1.75 |

For scale, the REST-referenced far_R map is 3-19e-3 RMS. So RESTnl is a materially different subtrahend from RESTW,
and that is why it is worth rendering.

**Two cautions for the plan:**

- **Check (ii) cannot match exactly.** The plan's check (ii) says the module's area-space level must reproduce
  `restw_check/sessions/<label>.npz strict__posmed` exactly. It cannot, for two reasons:
  - The median does not commute with the ROI-mean projection. The maps take the median per SVT component and then
    project; restw_check takes the median of the area trace.
  - The restw_check area series also has `combined_encoding._slow_basis` drift removed and is x1e3.

  Use one of these instead:
  - Reproduce the frame COUNTS (`strict__nfr_pos`) exactly, and the area levels only approximately.
  - Or compare against the area projection with the same median-then-project order.

  The mask and frame selection are identical, so the frame counts are the exact check.
- **Session coverage needs a separate check.** The map arms hold 120 sessions; restw_check / RESTnl exist for 107. The
  June pre sessions and any session without `licks_all.csv` need either the cam4 file or a stated fallback (DAQ-only
  `daq2`), because the strict mask reads it. This check did not test that coverage.

**Per-session cost**, measured with 2 processes in parallel and BLAS at 4 threads:

| step | time per session |
|---|---|
| SVT + U_atlas load from M: | 2.1-4.7 s |
| RESTnl mask + frames + wr level: DAQ h5 read, two lick detections, licks_all, frame selection, 6 medians | **6.5-11.0 s** (median ~7.5 s) |
| projection | < 0.5 s |

- **Total: ~10-15 s per session**, so ~25 min serial for 120 sessions, or ~7-8 min with 4 workers.
- That builds one baseline per session, shared by all 7 `prm` kinds.
- The existing RESTW rebuild took 10-39 s on the same sessions; the first session in each process includes imports.
- Nothing in this needs the `prm` caches to be rebuilt.

## 4. Is a FLAT baseline appropriate, or does within-session drift need the time-local version?

### Existing record

- **`meegkit_hpfit` drift removal is already in the signal.** The drift fit excludes strobe-0.25 s to cue+4 s
  (`position_reference_maps.reference_maps`, the `raw` comment). So the maps' signal is already high-passed, and slow
  drift is removed before any rest reference is applied.
- **`flat_vs_timelocal`** (DECISIONS.md 2026-09-14, full cohort, position-weighted arm): flat and time-local give the
  same ordering, the same near->far gradient and the same far-contra collapse.
  - Acute/pre amplitude ratios differ by 0.04-0.07, e.g. far_R 0.483 flat vs 0.445 time-local.
  - Between-animal far-contra agreement differs by ≤ 0.035 (acute 0.812 vs 0.838), against an effect of 0.81 over a
    null of 0.07.
  - "Time-local MOVES NO CONCLUSION, so the simpler estimator wins." This is why RESTW is flat.
  - **Limitation (DECISIONS ~l.13728):** that verdict was tested on the weighted arm only. The page's pooled REST
    stayed time-local by default, not because of a test.
- **`uniform_shift_check`** (DECISIONS ~l.13831; STATUS_2026-09-17 §0b): the restdock05 baseline moves by a median
  **3.1 %** of the evoked signal (p90 0.091, max 0.155), and `cos(d, evoked)` is negative in 12/19 sessions.
  - **Scope:** this measures how much the baseline shifts when the lick buffer is relaxed ([1,2] -> [0.5,1]). It is
    NOT a within-session drift measure.
- **`rest_baseline_epoch_drift`** (`epoch_15j`, STATUS_2026-09-17 §A2): the restw baseline drifts ACROSS epochs by
  7-43 % of the evoked norm (median ~0.15). It is significant in 6/11 animal-epoch cells and in 3/4 animals.
  - This is the session LEVEL changing between days. A flat or time-local construction within a session does not
    address it; only `raw` avoids it. It applies equally to RESTnl, flat or time-local, and to the page's REST.

### New measurement, 8 sessions

Construction:

- The page's REST mask (quiet & engaged) is split into 12 bins.
- Each bin's per-component median is projected to pixels. Bins with fewer than 20 frames are set to NaN.
- "Flat - TL" is the change to each position's REST-referenced map if the time-local baseline is replaced by one median
  over the same mask, holding composition fixed. It is computed as `(raw - raw_rest)[q] - U @ median(SVT[:, mask])`.
- All values are x1e-3 dF/F, pixels over the brain mask.

Table columns:

- **SD across bins**: bin-to-bin SD of the time-local baseline, median over pixels / p99 over pixels.
- **bin range**: max - min across bins, median over pixels.
- **slope**: brain-mean baseline change from the first to the last bin.
- **far_R map RMS**: RMS of the REST-referenced far_R map.
- **contrast RMS**: RMS of the between-position contrast (map minus the mean over positions).
- **Flat - TL far_R RMS (% of map)**: change to the far_R map, as RMS and as % of the far_R map RMS.
- **common / max pos-dependent**: the parts of Flat - TL shared by all positions / varying by position.

| session | epoch | SD across bins (med / p99) | bin range med | slope | far_R map RMS | contrast RMS | Flat - TL far_R RMS (% of map) | common / max pos-dependent |
|---|---|---|---|---|---|---|---|---|
| PS92_0814 | pre | 3.49 / 15.4 | 12.1 | +0.8 | 18.4 | 3.39 | 0.74 (4.0 %) | 0.54 / 0.85 |
| PS93_0814 | pre | 2.27 / 9.0 | 8.2 | -4.3 | 19.0 | 3.58 | 0.20 (1.1 %) | 0.32 / 0.41 |
| PS94_0814 | pre | 1.92 / 12.3 | 6.8 | +3.4 | 12.9 | 4.95 | 0.48 (3.7 %) | 0.56 / 0.54 |
| PS95_0814 | pre | 1.43 / 4.2 | 4.5 | -1.1 | 17.4 | 4.46 | 0.22 (1.3 %) | 0.13 / 0.23 |
| PS92_0819 | acute | 2.40 / 11.4 | 8.3 | -4.2 | 10.6 | 7.26 | 0.57 (5.4 %) | 1.04 / 0.77 |
| PS93_0819 | acute | 1.91 / 6.9 | 6.0 | -0.8 | 5.8 | 8.42 | 0.24 (4.1 %) | 0.11 / 0.36 |
| PS94_0818 | acute | 2.11 / 5.0 | 6.7 | -3.2 | 3.2 | 10.68 | 0.48 (14.9 %) | 0.28 / 0.42 |
| PS95_0818 | acute | 1.56 / 4.4 | 5.3 | +1.5 | 17.7 | 3.55 | 0.15 (0.9 %) | 0.08 / 0.19 |

Reading it:

1. **The bins do vary.** The time-local baseline's bin-to-bin SD (1.4-3.5e-3 at the median pixel) is comparable to
   the between-position contrasts (3.4-10.7e-3).
   - Much of that is bin-median noise, not slow drift. The brain-mean sequences are jagged rather than monotone (e.g.
     PS92_0814: -3.0, +2.7, -9.5, +3.0, ...). The thinnest-bin session, PS92_0814 with 119-929 frames per bin, has the
     largest SD.
   - On the lick-free mask the bins get thinner (0-85 frames in some PS92_0814 bins) and the SD rises: 5.8 vs 3.5e-3 in
     PS92_0814, 6.0 vs 1.9e-3 in PS93_0819. **A time-local RESTnl would mostly add noise.** A per-position time-local
     version is not formable at all.
2. **What a trial-averaged map sees is small.** Flat minus time-local changes the far_R map by 0.15-0.74e-3 RMS. That
   is 0.9-5.4 % of the far_R map in 7/8 sessions and 14.9 % in PS94_0818, where the acute far_R map is nearly gone
   (3.2e-3 RMS).
   - The position-dependent part, the part that could change between-position contrasts, is ≤ 0.85e-3. That is
     ≤ ~12 % of the contrast RMS in the worst session (PS92_0814: 0.39 of 3.39 for far_R) and 2-9 % elsewhere.
3. **No epoch dependence.** Far_R Flat - TL RMS averages 0.41e-3 pre vs 0.36e-3 acute; the common part is 0.39 pre vs
   0.38 acute.
   - The slope sign is mixed in both epochs: pre +, -, +, -; acute -, -, -, +. Acute is 3/4 negative but pre is 2/4,
     so this is not a stroke signature at n = 4.
   - For comparison, the across-epoch level drift (epoch_15j, 7-43 %) is roughly an order of magnitude larger.
4. **The late-session quit period is already outside both baselines.** The engagement gate leaves the last 2-3 of 12
   bins EMPTY in the acute PS92, PS93 and PS94 sessions, and the last 0-1 bins pre. So composition does not track the
   quit period.
   - A late low bin within the engaged period appears in 2 sessions: PS93_0814 last two bins -11.3 / -10.2 vs ~-4.5;
     PS92_0819 last bins -8.6 / -12.7 vs ~-4.
   - Even there, a flat baseline moves those sessions' maps by only 0.2e-3 / 0.6e-3 RMS (far_R), because the trials
     are spread across the session.

**Verdict on flatness: flat is appropriate, with a caveat.**

- A flat lick-free baseline is consistent with the RESTW decision and with `flat_vs_timelocal`.
- Its effect on the referenced maps is a few % of the signal and does not differ between pre and acute.
- A time-local lick-free baseline would be noisier, because its bins hold far fewer frames.
- **The caveat:** cells whose map is near zero (acute far-contra; PS94_0818 at 15 %) carry about ±0.5e-3 of baseline
  uncertainty. Read their amplitudes as "near zero", not as a precise ratio.
- Across-epoch level drift (epoch_15j) is a separate, larger issue that affects every rest reference equally. The
  page's `raw` column remains the control for it.
- Limits: n = 8 sessions, one pre and one acute per animal, cue/working contact arm only.
- If more certainty is wanted before render M2: rerun this scratch measurement on all 120 sessions. It takes ~15 s per
  session, so ~8 min with 4 workers.
