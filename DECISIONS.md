# Analysis decisions & findings (widefield LocaNMF spout-position study)

The durable choices and results behind the widefield analysis pipeline, so future runs are reproducible
and the per-decision rationale is explicit. Merged 2026-08-09 from the former `DECISIONS.md`
(pipeline/preprocessing) and `LOCANMF_LICK_CUE_ANALYSIS.md` (behavioral analysis + findings F1–F17).

**Companions:** open/actionable items → `TASKS.md`; how to run the nightly → `runbooks/`; step-level
commands → `wfield_local/README.md`; dated incident/status records pruned from here →
`docs/archive/ANALYSIS_HISTORY.md`; dead-strobe-bit recovery → `STROBE_BIT1_RECOVERY.md`.

## The endpoint this is building toward
Compare **cortex-wide activity when the animal tries to lick to each spout position, pre vs post stroke**.
Stroke = **ventrolateral striatum (subcortical)** → the imaged cortex is structurally intact, so this is a
*functional* reorganization question, not lesioned-cortex. Mice PS92/PS93/PS94/PS95; **PS93 has a RIGHT
orofacial deficit** (tongue deviates right, minimal right whisking) — the lateralization angle.

---

# Part I — Preprocessing & pipeline decisions

## Global decisions
- **Dual-wavelength**: 470 nm = functional (GCaMP), 415 nm = isosbestic reference. Hemodynamic correction =
  470 − β·415, fit in SVD space by `wfield` (`hemodynamic_correction`), which highpass-filters both channels
  at 0.1 Hz (this already removes slow LED drift — see photobleaching note).
- **Channel identity comes from the DAQ LED TTLs**, not frame parity. The relabel step
  (`trim_illuminated_labcams`) assigns 415/470 from `led415_ttl`/`led470_ttl`, so channel identity is correct
  regardless of the per-session parity ambiguity.
- **Allen alignment grid**: all spatial maps are warped to the **540×640 Allen atlas grid**
  (`apply_allen_transform --dims 540 640`), not native ROI size, with the atlas built in **reference space**
  (`do_transform=False`). Keeps ROI-cropped recordings aligned with the atlas.
- **Alignment transform**: **8-point affine** with the lateral anchors (OB_center/L/R, RSP_base, MOp_L/R,
  SS_L/R), from the hand-placed `dorsal_cortex_landmarks_v1.json` per session. The lateral MOp/SS points break
  medial collinearity so an affine (independent AP/ML scale + shear) is well-constrained, vs the earlier
  4-point similarity. Output dirs/labels use the tag **`affine8v1`**.
- **Cue-aligned maps**: per spout position, mean over the pre-cue and post-cue windows (**both 2 s**, from
  `configs/defaults.yaml preprocess.maps.cue_pre_s` / `cue_post_s`), plus the post−pre **delta**. Spout
  position from `spout_strobe` + `spout_bit0/1/2` (code = bit0 + 2·bit1 + 4·bit2 at the most recent strobe
  before the cue). **NB the `1s` in the figure/npz filename
  (`*_spout_positions_1s_pre_post_delta_*`) is a stale label** from when the pre-window was 1 s: it is kept
  deliberately because `preprocess_deck.PER_DATE_TYPES` globs on it, so renaming would orphan every
  historical session's figure in the deck. The true window is in each figure's `*_summary.json`
  (`pre_s`/`post_s`). Corrected 2026-08-11.
- **Lick-aligned maps**: per spout position, mean over 150 ms post-lick. Licks from `lick_analog`
  (upper/lower thresholds 2.5/1.0 V, 1–20 ms lockout, refractory) + a **40 ms physiological min-ILI floor**
  (`configs/defaults.yaml lick_detection.min_ili_ms`, applied pipeline-wide). **Delta-position lick maps** =
  pairwise position contrasts.
- **Mean image + Allen overlay**: 415/470 mean motion-corrected frames warped to the atlas grid, with Allen
  region outlines (from `frames_average_atlas.npy`), drawn by the shared `atlas_overlay.region_edges`.
- **Frame rate**: 31.23 Hz per channel.
- **Sign-fixed motion correction** (`run_wfield_motion` → `motion_correct_fixed.py`, `--mode 2d`) is the
  standard path; the wfield 0.4.2 sign bug is remediated (history: `docs/archive/MOTION_CORRECTION_SIGN_BUG.md`).
- **Figure-dir/label versioning** tracks **code iteration**, not the landmark-JSON version; `affine8v1`
  denotes the 8-pt-affine landmarks-v1 alignment.

## Map units — what "dF/F" means here (verified 2026-08-11)
The event-aligned maps **are** dF/F, because the normalization happens *inside* the SVD, not after it:
`wfield.decomposition.approximate_svd` is called with its default `divide_by_average=True`, so every block is
reduced to `(F − F0)/F0` **before** the decomposition (`run_wfield_local.py` never overrides it). Hence
`U @ SVT` is already fractional. Two consequences that decide how the maps may be read:
- **F0 is the per-channel SESSION-MEAN image** (`frames_average.npy`, shape `(2, H, W)`, raw counts) — *not* a
  pre-trial or pre-cue baseline. So values are relative to the whole-session average.
- **The DC level is gone.** `hemodynamic_correction` high-passes both channels at 0.1 Hz and subtracts each
  component's temporal mean, so the maps are zero-centred deviations. **Read them as CONTRASTS** (post−pre, the
  pairwise position contrasts, or the quiet-normalized lick map — the one map with a genuine behavioural
  baseline, via `quiet_periods.quiet_baseline_svt`) — **never as absolute dF/F magnitudes.**

Empirical check (PS95 8/10): reconstructed pixel values have std 0.021 and 1st/99th percentiles −4.8 %/+7.9 %
— fractional, versus ~14 000 raw counts in `frames_average`. NB `approximate_svd`'s own docstring says it "does
not compute df/f"; that describes its `divide_by_average=False` branch and is misleading for this pipeline.

## Event→frame mapping: two regimes (session-specific)
The corrected movie (SVTcorr) is indexed by paired 415/470 timepoints. Mapping a DAQ event to a
corrected-frame index depends on whether the movie was relabeled:
- **Regime A — raw recording (no relabel)**: corrected frame = (nearest `pco_exposure` pulse to the event)
  // 2. Used when the movie is the full contiguous recording (`frame_align=pco`).
- **Regime B — relabeled "cleanpairs" movie**: the movie is a non-contiguous subset of kept pairs, so raw//2
  is wrong. Each corrected frame `t` maps to DAQ sample
  `pco_samples[frame_map["original_frame_index_ch0"][t] + chosen_exposure_offset]`; events map to the nearest
  such sample. `chosen_exposure_offset` is read from the per-session `*_cleanpairs_summary.json` (**differs
  per session**). A contiguity guard rejects windows crossing a trial/kept-frame boundary. **Regime is
  validated by SENSIBLE DECODING (SSp ≫ chance), not by RT.** (6/2–8/7 have all been B.)

## Photobleaching / LED drift
The **isosbestic 415 declines ~9–16%** over a session while the **functional 470 is stable (±2–3%)**. True
GCaMP photobleaching would hit 470 hardest, so the 415-specific decline is **violet-LED drift**, not
bleaching. The 0.1 Hz highpass in hemo-correction already removes it, so ΔF/F is uncontaminated.
`run_wfield_local` also exposes `--detrend-order` + `--freq-highpass` for a gentler highpass when wanted.

**EXCEPTION observed 2026-08-11 — the 470 stability claim no longer holds for every animal.** On 8/11
two of four animals showed a large *functional-channel* decline, well outside the ±2–3% norm:

| session | duration | 415 % | **470 %** |
|---|---|---|---|
| PS92_0811 | 153.7 min | −18.6 | **−13.4** |
| PS93_0811 | 151.3 min | −8.2 | −1.8 |
| PS94_0811 | 74.3 min | −7.4 | −2.3 |
| PS95_0811 | 107.3 min | −14.6 | **−7.3** |

**It is not session length** — PS93 ran 151 min (essentially the same as PS92) with a normal −1.8% on 470.
So this is animal-specific, and a 470-dominant decline is the signature of REAL GCaMP photobleaching
rather than violet-LED drift. The 0.1 Hz highpass should still absorb it (the decline is far below
0.1 Hz), so ΔF/F is probably fine, but two things follow: (1) do not treat "470 is stable" as an
invariant when QC-ing a session; (2) if PS92/PS95 keep declining across nights it is an
illumination-power / expression issue worth acting on at the rig, and it would bias any analysis that
compares raw amplitudes across animals without the highpass.

## Cross-day & cross-animal alignment policy
- **Within animal, across days**: register the motion-corrected **mean 470 nm vasculature** to a chosen
  reference session (`cross_day_align.py`): landmark-init → intensity-based ECC affine refine (SIFT+RANSAC
  fallback), composed into the reference/CCF frame. Vasculature is a denser, more repeatable fiducial than
  the ~8 landmarks. Keep the **same ROI/zoom** across days (full-FOV↔ROI pairs register poorly). QC = masked
  NCC + red/green vessel overlay.
- **Across animals**: vasculature is not shared, so the **only** common frame is the Allen atlas. Compare at
  the **ROI / Allen-area level** (or via LocaNMF components), not pixelwise; group pixel maps are for
  visualization only.

## Task-controller v47 stub rows (2026-08-10 onward) — the DAQ-cue-count invariant needs a caveat
**v47 fixed the `pos_idx` overwriting** described in [`docs/GUI_TRIALS_LOGGING.md`](docs/GUI_TRIALS_LOGGING.md):
the colliding row is now CLOSED and a fresh one opened, instead of being overwritten with the next
trial's position. But v47 **keeps that stub row in `trials.csv`** — duplicate `trial_id`, sub-second
duration, neither `hit` nor `miss` set.

This **amends the 2026-08-09 correction above** ("DAQ cue count equals the log's scored-trial count
exactly in every session"). That held for every pre-v47 session and still does — but from 8/10 the raw
log is LONGER than the DAQ cue stream, and the invariant only holds after the stubs are filtered:

| session | DAQ cues | log rows (raw) | log rows (stubs dropped) | best-offset agreement |
|---|---|---|---|---|
| PS94_0806 (pre-v47) | 462 | 462 | 462 | — (dead-strobe session) |
| PS94_0810 | 720 | 1009 | **720** | 21.2% → **99.4%** |
| PS94_0811 | 436 | 554 | **436** | 23.0% → **98.4%** |
| PS95_0811 | 670 | 679 | **670** | 72.1% → **99.0%** |
| PS92_0811 / PS93_0811 | 386 / 492 | — | **386 / 492** | **100.0%** |

**Latent bug this hid (fixed 2026-08-12).** `classify_cues_with_backup` substitutes log positions only
when agreement is >= 0.9. With the stubs left in, the two sequences differ in LENGTH, no integer
trial-offset can align them, and agreement sat at ~21% — so **the dead-strobe fallback was silently
disabled on every v47 session**. It fails safe (DAQ codes are kept, never corrupted) and changed no
published result — positions come from the DAQ and every affected session had all 6 strobe positions —
but a dead strobe bit on a v47 session would have left degraded positions with NO recovery path, which
is precisely the 8/5–8/6 scenario that path exists for. `behavior_position._scored_rows` now requires
a row to be real (`start != end`) AND scored (`hit` or `miss` set); it degrades to the old filter on
pre-v47 schemas with no `hit`/`miss` columns. Tests: `tests/test_behavior_position_stubs.py`.

**Bonus:** the post-fix 98–100% agreement is now independent CONFIRMATION of the DAQ position codes —
two unrelated sources agree where they previously appeared to disagree.

## Session curation: 8/6 is KEPT despite a degraded DAQ (decided 2026-08-11)
All four animals' **8/6** sessions hit the dead `spout_bit1` and recorded only **4 of 6 positions** in the
DAQ, so their per-trial positions come ENTIRELY from the behavior-log repair
(`behavior_position.classify_cues_with_backup`), not from the strobe. 8/6 is nevertheless in the CURATED
cross-session set, so every cross-session result leans on that repair.

**Kept**, because the repair validates at **1.00 agreement on the DAQ's good positions, trial-offset +1,
for all four animals** — the log and the DAQ agree perfectly everywhere the DAQ was intact, so there is no
evidence the substituted labels are wrong (and `classify_cues_with_backup` only substitutes when it
validates >=0.9 AND the DAQ is actually short a position, so healthy sessions are never touched).

Two caveats to keep in view: 8/5 has the **identical** hardware defect and is excluded for *separate*
quality reasons (the exclusion is not about the strobe, so it is not precedent for dropping 8/6); and if
the repair is ever called into question, **8/6 is the first date to drop** — it is the one curated date
whose labels have no independent DAQ confirmation. See `STROBE_BIT1_RECOVERY.md`.

## Relabel step for future recordings
Recommended even with the trial-gated acquire-enable firmware: it drops stray illuminated/dark frames and
guarantees deterministic 415/470 pairing + the `frame_map` that regime-B alignment needs. Use
`--relabel-mode acquire-enable` for trial-gated recordings; `rescue` for older continuously-saved sessions.
The cleanpairs **movie** is a deletable/regenerable intermediate, but the relabel **step** + its small
`frame_map` stay in the pipeline.

## Quiet-period baseline (behavior-controlled F0)
Trial-triggered acquisition records no true inter-trial rest, so a behavior-controlled baseline detects
"quiet" frames (not running, not near a lick, not peri-reward), intersected with the pre-cue ENL window or
pooled as F0 (`quiet_periods.py` → `*_quiet_frame.npy`; ported from stroke_orofacial `find_quiet_bouts`).
Two rig-specific decisions: (1) **grooming OFF by default** — the stroke detector needs two spouts;
single-spout long-touch is unreliable (a true long close-spout lick also looks long). (2) **thresholds are
provisional** — running/quiet speed, min durations, lick/reward/treadmill buffers are stroke defaults; tune
per rig, ideally validated against DLC/FaceRhythm movement (see `TASKS.md`).
- **Quiet-normalized lick activity**: pass `--quiet-frame` to the lick plotter to emit both the raw
  post-lick map and a `*_quietnorm*` map (post-lick − mean quiet baseline = lick-evoked relative to the
  not-running/not-licking state).

## Server layout, archiving & safety
Two institutional file servers (copy-only; **never delete from a server without explicit per-action
permission**, and **only ever write inside the `Priya\` folder**):
- **M: (standby)** = `\\standby.files.med.harvard.edu\hms\neurobio\sabatini\collaborations\Priya`. The
  **huge files**: raw `.dat` in `raw_widefield_data\` AND the motion-corrected `.bin` in `motion_corrected\`
  (the bin lives here, NOT on MICROSCOPE, to save N: space).
- **N: (MICROSCOPE)** = `\\research.files.med.harvard.edu\Neurobio`, folder `N:\MICROSCOPE\Priya\`.
  **Analyzed data except the corrected video**: SVD (U/SVT/SVTcorr), Allen alignment, maps/QC, DAQ, decks →
  `…\Widefield\labcams\`. LocaNMF only needs SVTcorr + the atlas, so the GPU is unaffected. Copy excludes the
  regenerable raw + cleanpairs `*_uint16.dat`.
- **DAQ `.h5` on N: is organized by date, one level only**:
  `…\Widefield\DAQ_recorder_output\<YYYYMMDD>\<animal>_<date>_<time>.h5`. The E: layout is *not*
  canonical (files sit loose under `E:\DAQ_recorder_output`), so `archive_day` derives the destination
  from `--date`, never from the E: parent dir. Before 2026-08-09 it mirrored the parent dir name, which
  put loose E: files into a nested `DAQ_recorder_output\DAQ_recorder_output\` on N: (fixed; the stray
  nested copies were duplicates of the per-date ones).
- `wfield_local/archive_day.py` implements this (raw + bin → M:, rest → N:); `writeguard.assert_writable`
  refuses writes/deletes outside the Priya subtree.

### Running preprocessing on a THIRD (helper) box — no local raw (added 2026-08-11)
Operational runbook (incl. the Windows-Update suspension on the Priya lab desktop and how to undo it):
[`runbooks/helper_box_setup.md`](runbooks/helper_box_setup.md).
A spare workstation can absorb one animal's preprocessing when the imaging box is busy (done for PS95 8/10
on the Priya lab desktop). It mounts MICROSCOPE at `N:` like the imaging box but has **no `E:`**, so
`paths.detect_machine()` finds no signature mount and defaults to `analysis` → every root resolves to a
non-existent `M:`. Set up without editing `configs/`:
```powershell
subst E: C:\wf_local                 # E:\labcams_data\<DATE>\<session>\raw_widefield_data\, E:\DAQ_recorder_output\
$env:WIDEFIELD_MACHINE = "imaging"   # forces the N: mounts
```
Then copy that session's raw `.dat` + DAQ `.h5` under `E:` and run `preprocess <DATE> --only <ANIMAL>`
normally; the ~190 GB `.bin` stays on local scratch and only the results are pushed to `N:`. **Neither
setting survives a reboot.** Hard-won cautions:
- **Verify the raw `.sha256` sidecar**, and expect the `N:` mount to drop mid-transfer (`ERROR 53`). Plain
  `robocopy` discards a partial 190 GB file on a drop; use a resumable copier that hashes in the same pass.
- **Do NOT run `preprocess_deck` (`build_decks`)** — it globs `cross-session_preprocessing*.pptx` and
  **deletes every sibling deck it did not write this run**, destroying the other animals' decks. Call
  `build_deck` (singular) with `sessions` filtered to the one animal.
- **Do NOT run the photobleach step via `preprocess`** (omit it with `--skip-photobleach`): `photobleach.run`
  calls `summary()`, which rewrites the date's **shared** `photobleach_SUMMARY.png` + `photobleach_results.json`
  with only the animals in *this* run. Call `photobleach.analyze()` alone — the deck only reads the
  per-session `photobleach_<ANIMAL>_<MMDD>.png`.
- `refresh_xall` and `crossday_intensity` are safe: the first is per-animal, the second is a whole-tree
  rollup that is *improved* by re-running after the push (it then includes the new session).

### Recovering a crashed / force-split session — `concat_split_session` (added 2026-08-12)
When labcams + the DAQ recorder are force-closed mid-session (updater kill) and restarted, you get N
camera `.dat` + N DAQ `.h5` segments covering one recording. `wfield_local/concat_split_session.py` rejoins
them: byte-concatenate the `.dat` (camera was off during each gap → no gap frames), concatenate the DAQ
per-sample streams with each **inter-segment gap zero-padded** (keeps the sample timeline wall-clock
accurate so the uninterrupted behavior program + behavior camera still align via the shared sync pulse),
and write a manifest of per-segment sample/frame boundaries. It verifies the invariant **`pco_exposure`
rising edges == 2 × `.dat` frame-pairs** per segment (camera and DAQ agree on frame count).

**Staggered crash (`--trim-to-sync`).** A real *crash* (vs a clean force-close) stops labcams, the DAQ, and
the behavior box at **different** times, so a segment's `.dat` frame count and its DAQ `pco` pairs no longer
match and the strict invariant fails. `--trim-to-sync` uses the **`pco` pulses as ground truth** (start
alignment: `.dat` frame 0 == the first `pco` pulse) and trims each segment to the imaging↔sync **overlap**
(`min(dat_pairs, pco_rises//2)`): `.dat` longer than sync → keep the first that-many frames (drop the
sync-less tail); DAQ longer than `.dat` → cut the DAQ just after the last synced edge; a crash-truncated
partial last frame is floored. The default stays strict (no silent trimming). Tests in
`tests/test_concat_split.py`.

**PS92 20260812 (imaging-computer crash).** Session 1 (DAQ `152628`, imaging `151741`) recorded **29,112**
`.dat` frame-pairs but only **19,742** `pco` pairs — labcams kept writing ~9,370 frames after the DAQ had
stopped, so those tail frames have no sync and were dropped. Session 2 (`161746`/`161728`) was whole
(163,594). Concatenated with the 40.7-min inter-session gap zero-padded →
`labcams/20260812/PS92_20260812_concat/` (`.dat` = 183,336 pairs) + `DAQ_recorder_output/…/PS92_20260812_concat.h5`
on MICROSCOPE; preprocessing + LocaNMF + behavior run on that concat'd session, not either raw one. The
folder-name timestamps are misleading here (labcams `151741` was *open* ~9 min before it started
acquiring) — **use the sync pulse, not the folder stamp**, to reason about alignment.

---

# Part II — LocaNMF decomposition

Runs on the GPU box (`wfield_local/run_locanmf.py` → `wfield.local_nmf.compute_locaNMF` on an
`allen_aligned_*` folder; needs PyTorch + `locanmf` + a CUDA GPU). **`cuhals` (the compiled CUDA/C++ HALS
kernel) is OPTIONAL** — `locanmf/demix.py` wraps `import cuhals` in `try/except ImportError` and falls back
to `native_update`, a pure-PyTorch HALS that still runs on the GPU. So a box with **no CUDA toolkit and no
MSVC** (and no admin rights to install them) can run LocaNMF: install torch + `pip install .` on the
locaNMF clone **without** `--with-extension`, applying only `locanmf_torch_compat.patch` (the
`locanmf_cuhals_win_build.patch` is needed only when building the extension). Validated 2026-08-11 by
re-running PS95 8/9 on the fallback: **164 components vs the cuhals build's 160, same 64 regions, 54/64
regions with identical component counts** (the rest ±1 — the rank line-search is greedy, so exact equality
is not expected). Building the extension is worth it on a box that runs LocaNMF regularly: measured on the
same session, **9.3 min with `cuhals` vs ~55 min on the fallback (~6x)**, 167 vs 164 components. Full
install/build walkthrough (VS Build Tools -> CUDA Toolkit -> MKL -> patch -> build, and the three traps:
`DISTUTILS_USE_SDK=1`, PATH-before-vcvars, and `pip install .` silently dropping `--with-extension`):
[`docs/GPU_SETUP.md`](docs/GPU_SETUP.md). NB the torch-compat patch may not apply cleanly with `git apply`
(EOF/whitespace drift in `factor.py`); the three edits are small enough to apply by hand.
Also: **Blackwell GPUs (RTX 50-series, `sm_120`) need cu128** — the `cu124` build in the README/kickoff doc
does not support them. Kickoff + env recipe:
`docs/archive/GPU_LOCANMF_KICKOFF.md` / `GPU_LOCANMF_RUNLOG.md`. LocaNMF consumes exactly what preprocessing
already produces (low-rank `U`/`SVTcorr` + the native-grid atlas), so it is a clean bolt-on; vs raw SVD
components (delocalized, not reproducible) its components are interpretable and **reproducible across sessions
and animals** — the natural unit for cross-animal / functional-subnetwork analyses.

## Parameters (decided 2026-06-04): `r2_thresh=0.95, loc_thresh=80, maxrank=20`
Chosen from a 2×2 sweep (`r2_thresh` ∈ {0.95, 0.99} × `loc_thresh` ∈ {70, 80}) on PS94 6/3 and PS95 6/3
(`sweep_locanmf.py`), driven by a per-component **localization metric** = fraction of a component's spatial
energy inside its seed Allen region (artifacts are delocalized → score low).
- **`loc_thresh=80` is the artifact knob and a near-free cleanup.** Raising 70→80 removed the
  low-localization artifact tail while barely changing component count (PS94 6/3 → median loc 83 %, **0**
  components <50 %; PS95 6/3 → median 89 %, **0** <30 %). Only bleed/vessel components are affected.
- **`r2_thresh=0.95` over 0.99.** 0.99 over-splits (near-duplicate pairs + noise-fitting); 0.95 captures the
  same structure with far less junk, and over-splitting is separable post-hoc by the localization metric.
- **Bilateral patterns are preserved**, not lost. The atlas seeds each hemisphere separately, so bilateral
  activity = a homotopic L/R pair of unilateral components with correlated traces (recover via trace
  correlation or map summing), never a single bilateral map — true at any `loc_thresh`. Cleaner
  per-hemisphere traces make bilateral synchrony measured *better* (PS94 6/3 SSp-m L/R trace r 0.35→0.85) and
  reveal lateralization (SSp bilateral ~0.8; MOp/MOs lateralized ~0.4) a forced-bilateral component masks.

QC per run: two montages (`*_components.png` region-ordered + `*_components_byenergy.png` energy-ranked) +
the localization metric. Outputs go to a new `locanmf_*` subfolder; nothing prior overwritten.

## Component units: why `C` is footprint-scaled before any analysis
Every downstream analysis (decoder, encoder, cross-mouse, RSA) uses
`sig = _footprint_scale(A)[:, None] * C`, **not** raw `C` (`locanmf_position_decoder._build_signal`,
`_footprint_scale` in `locanmf_crossanimal_dff.py`). This is required, not cosmetic:
- **NMF has a per-component scale indeterminacy.** The factorization only constrains the *product*
  `Aᵢ Cᵢᵀ`; rescaling `Aᵢ → Aᵢ/k`, `Cᵢ → k·Cᵢ` reconstructs the identical movie. So a raw `C` amplitude is
  meaningless on its own, and is **not comparable across components, sessions, or animals** — pooling raw
  `C` would weight components by an arbitrary constant.
- **The fix pins the scale to a physical quantity.** With `Y ≈ A Cᵀ` (pixels × time), component *i*
  contributes `Aᵢ(x)·Cᵢ(t)` at pixel `x`. Its `Aᵢ`-weighted spatial mean is
  `Σₓ Aᵢ(x)·[Aᵢ(x)Cᵢ(t)] / Σₓ Aᵢ(x) = Cᵢ(t)·ΣAᵢ²/ΣAᵢ` — exactly `_footprint_scale(A)[i] = ΣAᵢ²/ΣAᵢ`
  (`nansum` over the map; 0 when `ΣAᵢ ≈ 0`). The scale factor is **invariant to the indeterminacy**: under
  `Aᵢ → Aᵢ/k`, `ΣAᵢ²/ΣAᵢ → (1/k)·ΣAᵢ²/ΣAᵢ`, which cancels the `k·Cᵢ`.
- **So `sig[i]` is the footprint-weighted mean dF/F inside component i's own footprint**, in the same
  units as the pixel maps (subject to the DC caveat in Part I "Map units"). That is what makes summing
  components within an Allen region, and pooling across animals with different LocaNMF bases, legitimate.

## Decoder regularization: `C=0.5` MEASURED, not assumed (2026-08-11)
`LogisticRegression(C=0.5)` had been a bare literal with no recorded sweep. Measured over **10 sessions
(all four animals, cue-aligned 2 s, block CV)** with `wfield_local/decoder_c_sweep.py`:

| C | 0.01 | 0.05 | 0.1 | 0.25 | **0.5** | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| mean acc | .690 | .721 | .731 | .732 | **.734** | .730 | .725 | .722 | .718 |

- **C=0.5 is the argmax** — the existing default is optimal on this data.
- **The optimum is FLAT**: C ∈ [0.1, 1.0] all within 0.004 of the peak. The exact value barely matters
  over an order of magnitude, so regularization is not a lever worth tuning further.
- **Per-session tuning buys nothing.** Honest nested CV (C chosen inside each outer TRAIN fold, applied
  to the untouched TEST fold) = 0.7381 vs 0.7342 fixed → **+0.004, well inside the ±0.039 SEM**. The
  inner CV's picks scatter across the whole grid (0.05 ×11, 0.25 ×12, 0.5 ×7, 10.0 ×4) — the signature
  of an argmax that is noise on a flat surface.

**Decision: keep the fixed `C=0.5`; do not tune per session.** Re-run the sweep if the feature space
changes materially (e.g. a switch to pooled ROI features or a different window).

## When is LocaNMF actually helpful (the synthesis)
- **Not** for region-level evoked responses / ROI summaries → that equals an Allen-ROI average (F5) with
  extra scale-ambiguity; use ROIs/pixel maps there.
- **Yes** for: (a) demixing overlapping/contaminated sources, (b) sub-region structure (multiple
  components/region), and especially (c) a **compact, denoised, interpretable, atlas-anchored basis for
  models** — decoding/encoding/connectivity/single-trial — where it beats SVD (interpretability) and ROIs
  (single-trial SNR, multi-area joint model). The position decoder (F10) is the canonical good use.

---

# Part III — Behavioral analysis decisions

- **Lick events split by the 6 spout positions** (close/far × L/center/R) via `_classify_events` /
  `_classify_cues` (spout strobe + bits). Cue = the *intended* position. **The cue is a 1 kHz, 75 ms tone.**
- **Components kept INDIVIDUAL, labeled by Allen region — NOT pooled.** Pooling within a region ≈ an Allen-ROI
  average (F5), throwing away LocaNMF's only value-add. Region label is for cross-animal *identity*, not
  averaging.
- **Normalization → use ΔF/F for cross-animal magnitude** (the journey, F3/F4): quiet-period z is unusable
  cross-animal (PS92's quiet mask is contaminated → deflated z); a data-driven "quietest-frames SD"
  over-corrects; pre-cue-1 s z is a comparable baseline; but **physical ΔF/F** via the scale-invariant
  footprint weight `s_i = Σ Aᵢ²/Σ Aᵢ` (`dff_i = s_i·C_i`, Churchland convention) is best — no SD, removes
  LocaNMF's scale ambiguity. **Within-component contrasts** (e.g. contralateral index) are
  normalization-robust regardless.
- **Stats: animal is the unit of replication** (mean ± SEM across mice), not across components/trials.
- **Engagement (critical for the stroke comparison):** exclude *disengaged blocks* (extended no-lick **and**
  no-movement), but define engagement by **movement/arousal, NOT per-trial lick success** — post-stroke a
  no-lick trial is a *failed attempt* (the deficit to keep). Lick-density gating is a first pass; upgrade to
  movement-gated once DLC is up. Engagement gating is applied as a **reporting** choice on the full trial
  table, never as a parsing filter, so the unrewarded/failed-attempt trials stay available for the future
  post-stroke analysis.
  - **CORRECTED 2026-08-09 — the DAQ cue/strobe stream is NOT a "rewarded subset".** This was previously
    recorded as "the DAQ tracks the rewarded subset (reward held after ~6 misses), which doubles as an
    engagement filter". Measured across every June/August session: the DAQ cue count **equals the behavior
    log's scored-trial count exactly** (304/304, 536/536, 627/627, 430/430 …), and DAQ cues include
    unrewarded trials (PS92 6/6: 302 of 304 rewarded). There is no reward-based subsetting to rely on, so
    the disengaged tail must be — and is — removed by `flag_engagement`, not by the DAQ stream.
- **DAQ detects real licks inside the ENL that the GUI misses (2026-08-10).** The task enforces a 2–3 s
  enforced-no-lick (ENL) before each cue (`gui_config` timing `precue_min/max` 2000/3000). Yet the DAQ-primary
  lick train shows licks in the 2 s pre-cue window on **~16 % of trials** (PS92 8/7: 70 / 430 cues), while the
  **GUI lick stream shows zero** there — the GUI's ENL is perfectly self-consistent. Inspecting the raw
  `lick_analog` at those events (QC `behavior_summary/qc/*_precue_daq_gui_mismatch.png`): they are
  **full-amplitude licks** — the active-low sensor drops from its ~5.5 V idle to **0 V** for ~50–80 ms,
  identical in shape/depth to GUI-confirmed licks. So this is **not a threshold effect** (the GUI threshold
  ≈2000 counts ≈1.6–2.4 V; these licks bottom out at 0 V, below any threshold, and below the DAQ's own 2.5 V);
  the GUI misses them because its Teensy main loop **polls the lick line intermittently / with debounce**,
  whereas the DAQ samples continuously at 5 kHz. Consequences: (1) our analysis is unaffected — we use
  **DAQ-primary** licks everywhere, so these are captured; the ENL "violations" are GUI blind spots only.
  (2) It quantifies the **F13 caveat**: genuine pre-cue tongue movement exists on ~1 in 6 trials, so a
  post-stroke pre-cue-decode change could partly be movement, not representation. (3) If the *task* should
  truly enforce the ENL on these, the fix is GUI-firmware-side (faster/interrupt-driven lick polling), not a
  threshold change — that lives in the separate behavior-rig GUI/firmware, not this repo.

---

# Part IV — Findings (F1–F17)

- **F1. Orofacial responses are predominantly lick(motor)-driven**, not cue(sensory): on cue-no-lick trials
  the orofacial cortex is ~flat while cue+lick is large.
- **F2. Cue→first-lick RT ≈ 0.16 s** (very fast) → cue and lick are tightly collinear; hard to separate
  sensory from motor.
- **F3. The "PS95 > PS94 > PS92" cross-animal amplitude order was mostly a normalization artifact.** PS92's
  quiet mask has quiet-SD/total-SD ≈ 0.97 so its quiet-z denominator is inflated (z deflated ~45× vs ~10×).
  In **ΔF/F + animal-level stats the three mice are tightly consistent.**
- **F4. NOT a 470/415 frame-parity artifact** (lag-1 autocorr +0.99, ~0 % Nyquist power; a parity artifact
  would be ≈ −1 / ~80 %). Frames are indexed correctly.
- **F5. Region-pooled LocaNMF ≡ Allen-ROI ΔF/F** (`mean_pixels(U_region)·SVTcorr`): trace correlation median
  **r ≈ 0.99** (SS) / 0.985 (MO). So **for region-level work use ROIs**; reserve LocaNMF for per-component
  analyses.
- **F6. Contralateral somatosensory tuning is real and reproduced by LocaNMF.** Within-component
  post-lick-150 ms position×hemisphere contrast: contralateral-spout licks larger in **100 % of SSp
  components**. Motor contralateral *modulation* (scale-free index) is **comparable to SSp**, just more
  variable — the earlier "motor less lateralized" claim was an absolute-amplitude artifact and is retracted.
- **F7. far_center outlier = PS95-specific** (both its sessions ~1.8–2.2× others), not sampling.
- **F8. Cue+lick FIR encoding model**: R² ~3–6 % only, cue/lick collinear → the sensory/motor *split is
  unstable*. Consistent with Musall (cortex movement-dominated); a real separation needs **video-derived
  movement regressors**.
- **F9. Auditory positive-control inconclusive due to coverage.** Cue is auditory (tone), but AUD ROI signal
  is **~3–4× weaker than SSp** every session (auditory cortex under-sampled at the lateral window edge). Not
  evidence against an auditory response.
- **F10. Spout position decodes strongly from cortex.** 6-way logistic regression on individual LocaNMF
  components, engaged trials. **Canonical method (F12): no per-trial baseline, block-aware CV, first-lick
  aligned, 2 s window** → **0.67 / 0.83 / 0.85** (PS92/PS94/PS95 6/3; chance 0.17). **SSp carries it
  (SSp-only 0.62–0.77) >> MO (MO-only 0.33–0.52).** Top features are **orofacial SSp subfields (SSp-m/n/un/bfd),
  CONTRALATERAL to the spout**; MOp/MOs ~9–16 % of weight, most for *far* positions. Confusions are between
  adjacent spouts. First-lick > cue alignment. **Window ~2 s is optimal** (integrates the lick bout; ITI ~7 s
  so no cross-trial bleed).
- **F11. No-lick trials decode at ≈ chance** (train engaged, apply to no-lick, cue-aligned: 6/3
  0.29/0.11/0.20; chance 0.17). (a) A clean **negative control** (no confound exploited); (b) confirms the
  position code is **lick/engagement-driven** (F1); (c) baseline no-lick = **disengagement**, but post-stroke
  no-lick = **failed attempts** — the real test (if they decode *above* this chance baseline, intent is
  preserved despite motor failure). **Must separate failed-attempt from disengaged via movement/video.**
- **F12. Decoder methodology (load-bearing for the stroke pre/post).** (i) Positions are presented in
  ~6-trial **BLOCKS** (P(stay)≈0.84); with random k-fold, same-block trials land in train *and* test → the
  decoder reads each block's slow-drift fingerprint (inflation; symptom: pre-cue "decoded" 0.47–0.65). **Use
  block-aware CV (leave-whole-blocks-out).** (ii) **No per-trial baseline** — a session-constant baseline is
  removed by feature standardization (identical decoding); a per-trial pre-cue baseline **over-subtracts**
  real anticipatory signal. (iii) The pre-cue window decodes position **above chance even under block-CV**
  (6/3 0.40–0.56) = genuine anticipatory coding. `locanmf_position_decoder.py` defaults `--baseline none
  --cv block`; toggles `--baseline precue`, `--cv random`.
- **F13. Pre-cue (anticipatory) decoding — the motor-independent post-stroke readout.** Train on engaged
  trials' pre-cue window, block-CV (6/3 0.40/0.55/0.56). Applied to **no-lick** trials it decodes **above
  chance** (6/3 0.26/0.34/0.22) — unlike the post-cue no-lick decode (F11). So the **maintained position code
  is readable without a lick** — the ideal readout for post-stroke failed attempts. **Validated by a
  time-only control** (position not decodable from cue-time alone, 0.02–0.11). *Caveat:* pre-cue may carry
  ongoing inter-trial movement → needs video.
- **F14. Baseline (pre-stroke) variability is LARGE — 3 days/animal.** Post-lick 2 s decoding: PS92 range
  0.21; PS94 range **0.55**; PS95 range 0.41. 6/4 is the low day for PS94 & PS95, tracking engagement.
  **Design implication:** a single pre-vs-post contrast is uninterpretable — need multiple baseline days,
  engagement/movement matching, and per-position/per-region contrasts (more stable) + the F13 pre-cue readout.
- **F15/F15a. PS93 hemisphere asymmetry (holds at n=3, 6/5–6/7; `locanmf_cross_mouse.py`).** Motivated by
  PS93's RIGHT orofacial deficit. Per-mouse SSp-hemisphere-only decoding (first-lick 2 s, block-CV): **PS93
  SSp-LEFT 0.43 << SSp-RIGHT 0.55 (L−R = −0.12)** vs near-symmetric others (PS92 +0.02, PS94 +0.06, PS95
  +0.01) — PS93 is the only mouse with a large SSp L-vs-R asymmetry, its LEFT (contralateral to the deficit)
  weakest. Behavioral R-spout recall intact (0.82) → **cortical-hemisphere, not spout-side**. Encoding EV
  reinforces it (PS93 close_R negative, close_L well-encoded). Bar panels show mean ± SEM with session points;
  the encoder reports raw EV and **normalized-to-1.0 FEVE**.
- **F16. RSA — representational geometry (within vs across animals; `locanmf_rsa.py`).** Per session a 6×6
  RDM = 1 − corr between the 6 position patterns; 2nd-order RSA = Spearman between RDMs (basis-free). **Within-
  animal RDM similarity > across-animal for ALL four** (within 0.45–0.62 vs across 0.25–0.32) → a stable
  individual geometry. Calibrated by a **split-half noise ceiling**: PS92/PS94 essentially at ceiling
  (geometry as stable across days as within a session); **PS95 shows genuine cross-day drift**; PS93
  intermediate. **PS93 is NOT the geometric outlier** — its RDM is most like PS92's (0.77); PS94 is most
  distinct. So PS93's deficit is the *lateralized* SSp-L≪R asymmetry, **dissociated** from a (preserved)
  global geometry.
- **F17. Hemisphere-resolved RDM exposes PS93's lateralization the pooled RDM misses
  (`locanmf_rsa.fig_rsa_hemisphere`).** Building the RDM separately from L- vs R-hemisphere components: the
  disattenuated within-session **L-vs-R RDM agreement is PS93 0.44 — the lowest** (PS92 0.69, PS94 0.80, PS95
  0.91). PS93's LEFT hemisphere is **not** unreliable (split-half 0.78) — so the deficit **reshapes** the
  contralateral (left) position geometry rather than abolishing it. Agrees with F15. Caveats: n=3 PS93;
  disattenuation unstable when a hemisphere's reliability is low; next probe = per-cell RDM_L−RDM_R.

---

# Part V — Stroke-study analysis plan (decided)
- **Model = the per-position decoder** (multinomial logistic regression, L2, standardized, block-CV). Linear
  + interpretable, correct for the n/p regime (~300–450 trials, 66–152 features).
- **Anchor on the cue** (intended position) so it applies to post-stroke **no-lick failed attempts**; train
  on baseline **engaged (cue+lick)** trials.
- **Two complementary comparisons:** (1) **Per-session decoders (primary)** — train per session, compare
  per-position recall + confusion + SSp/MO breakdown pre vs post; no common feature space needed (sidesteps
  cross-day component correspondence); also apply each session's decoder to its own no-lick trials.
  (2) **Frozen pre-stroke decoder (confirmatory)** — needs a common basis: fixed pre-stroke `A`, refit `C`
  (`C_new = pinv(A_ref)·U_new·SVTcorr_new`; valid because the stroke is subcortical) or Allen-ROI features.
- **Prerequisites** (open — see `TASKS.md`): cross-day vasculature registration (`cross_day_align.py`);
  DLC/facerhythm movement regressors time-synced to widefield+DAQ (to separate "cortex codes position
  differently" from "the movement just changed"); the packaged frozen pre-stroke model + baseline noise floor.

## The frozen ROI decoder transfers across days at NO cost (measured 2026-08-11)
The main risk to the confirmatory arm was that a frozen decoder would decay across days on its own, so a
post-stroke drop could not be attributed to the lesion. **It does not.** `locanmf_frozen_decoder --loso`
pools each animal's curated sessions in Allen-ROI space (z-scored per session, groups = SESSION, so each
held-out fold is an entire unseen day) and compares against the honest same-day ceiling (per-session
block CV):

| animal | pooled trials | LOSO (unseen day) | within-session | transfer cost |
|---|---|---|---|---|
| PS92 | 2593 | 0.701 | 0.600 | **+0.102** |
| PS93 | — | 0.597 | 0.585 | **+0.012** |
| PS94 | 3535 | 0.753 | 0.709 | **+0.044** |
| PS95 | 3660 | 0.863 | 0.816 | **+0.047** |

**Every animal is POSITIVE** — the frozen model *beats* the same-day model, and only 1 of 28 sessions
showed any drop. LOSO trains on ~3000 trials vs ~500 within-session, and ROI features are stable enough
across days that the extra data more than pays for the day gap. So post-stroke degradation can be read as
lesion effect. (LocaNMF components cannot do this — session-specific count AND identity; ROI features are
atlas-anchored, so column j is the same area every day. This is the "Allen-ROI features" branch above.)

## Cross-session RDM basis: what we tested and what we rejected (2026-08-12)
The per-session LocaNMF RDM is **not comparable across sessions** (see "RSA basis" below). Four
shared-basis candidates were measured on all four animals, 8 curated sessions each, lick-aligned,
scoring mean 2nd-order RSA of each session to its siblings in BOTH metrics:

**These numbers were RE-MEASURED on 2026-08-12 after the whitening fix below; the table now shows the
corrected crossnobis.** The 1−Pearson column is unaffected by whitening and reproduced to 6 decimal
places, which is the integrity check that the re-run changed only what it should.

| basis | 1−Pearson | **crossnobis** | worst animal (crossnobis) |
|---|---|---|---|
| per-session LocaNMF (status quo) | +0.503 | **+0.767** (worst) | +0.729 |
| Allen-ROI | +0.571 | **+0.817** (best) | +0.791 |
| frozen fixed-A, ref 6/6 | +0.572 | +0.799 | +0.778 |
| frozen fixed-A, ref 8/6 | +0.677 | +0.799 | +0.759 |
| frozen fixed-A, ref 8/9 | **+0.729** (best) | +0.784 | +0.757 |
| joint concatenated, rank 100 | +0.658 | +0.806 | +0.785 |
| joint concatenated, rank 200 | +0.569 | +0.778 | +0.746 |

**Rank 100 beats rank 200** in all four animals (−0.014 to −0.040). The earlier finding that 200 was
better was a diagonal-whitening artifact: under the old metric r200 led by +0.065 but disagreed in
sign across animals, which is what a noise-dominated estimator looks like. The supporting argument —
that variance retained was still climbing at r200 (99.53% → 99.81%) — was a category error: that
measures reconstruction of the MOVIE, not representation of the six conditions. Note also that after
LocaNMF pruning, r200 returns essentially the same basis for PS93 (96→98) and PS94 (98→98), so half
the "rank sweep" compared a thing to itself.

**REJECTED — frozen fixed-A / refit-C, despite the best mean scores.** Its result depends on which
session you nominate as the reference, and **no reference wins for every animal**: best is 8/9 for
PS92/PS93, 8/6 for PS94, but 6/6 for PS95 (+0.845 there vs +0.436 for PS92). The swing within one
animal reaches **0.36** — larger than the gap between most methods — and the two metrics disagree
about which reference is best overall (8/9 on Pearson, 6/6 on crossnobis). Choosing per animal would
be tuning a free parameter on the outcome. Kept in code (`project_C_fixed_A`) for the post-stroke
fixed-basis path, but **not used for RSA and not in the deck**.

**ADOPTED — joint concatenated basis** (`wfield_local/joint_basis.py`): reference-free, so it has no
such knob, and on the corrected metric it is second only to Allen-ROI (+0.806 vs +0.817) with the
lowest across-animal spread of any method (sd 0.017). It beats every single-session frozen reference
(+0.784–0.799), so it is a strictly better version of the frozen idea.

ROI (+0.817) and joint (+0.806) are **not distinguishable**. LocaNMF is stochastic: two runs of
identical code over identical sessions returned different component counts (PS92 123 vs 128, PS93
93 vs 96, PS94 95 vs 98, PS95 137 vs 135) and moved the 1−Pearson RSA by up to 0.054 — five times the
gap being adjudicated. Detected by the integrity check, which found drift of exactly 0.000000 for the
five deterministic methods and non-zero drift only for the two that refit LocaNMF.

**The joint basis is therefore BUILT ONCE, SEEDED, AND PERSISTED** (`wfield_local/joint_locanmf.py`),
never refit per run. Randomness enters via `randomized_svd` (numpy global RNG) and `torch.randperm` in
LocaNMF's initialisation; seeding both makes a rebuild reproducible. A `basis_id` hashes the session
set, input file signatures, rank, LocaNMF params and seed, and is stamped into every result, so a
refit lands in a new directory and no figure can silently mix two bases. Nothing is overwritten and
superseded bases stay loadable, so today's figures remain reproducible after a refit. **A refit over
the final curated pre-stroke set is planned and expected** (Priya, 2026-08-12); it is a versioned
event, not a nightly one — a basis refit over a growing session set would silently make last week's
numbers incomparable with this week's, and the post-stroke reference frame must be fixed BEFORE the
manipulation.

### ⚠ THE ZERO-PHASE HIGH-PASS INFLATES THE PRE-CUE CODE BY ~2x (2026-08-13)
**The single most consequential finding so far, and it invalidates the published pre-cue numbers.**

`wfield.hemodynamic_correction` high-passes BOTH channels at 0.1 Hz using scipy `filtfilt` — forward
AND backward — and the high-passed 470 channel is what becomes `SVTcorr`. `filtfilt` is zero-phase,
which is another way of saying ACAUSAL: its impulse response is symmetric in time. Measured on the
pipeline's own filter (2nd-order Butterworth, 0.1 Hz, fs 31.23): an impulse deposits **−0.496 BEFORE
itself and −0.496 after**, with −0.209 landing in the single second preceding it. `lfilter` (causal)
deposits exactly 0.000 before.

A high-pass is `identity − lowpass`; at 0.1 Hz the lowpass kernel is ~10 s wide, and zero-phase centres
it on each event, so half of it falls before the event and is SUBTRACTED. A position-specific post-cue
response therefore casts a scaled, SIGN-FLIPPED shadow across the preceding seconds. A linear decoder
is indifferent to sign, so that shadow is decodable position information in a window that, biologically,
cannot contain it. Simulation with position-tuned post-cue responses and NO pre-cue signal at all
reproduced it: the pre-cue window came out 42–67% of post-cue amplitude and correlated **−1.00** with
the position tuning.

**Measured on real data**, rebuilding `SVTcorr` from the retained `SVT.npy` with the saved transform
`T` held fixed so ONLY the filter varies (12 sessions: 4 animals x 6/7, 8/10, 8/11):

| variant | PRE-CUE | post-cue (control) |
|---|---|---|
| `zerophase` (current pipeline) | **0.498** | 0.718 |
| `causal` (lfilter) | 0.265 | 0.616 |
| `fitonly` (HP for the coefficient fit only) | **0.288** | **0.717** |

Pre-cue falls in **12 of 12 sessions**. The control is decisive: under `fitonly` the post-cue decode is
0.717 vs 0.718 — unchanged to a thousandth — so removing the backward smear costs the post-cue readout
NOTHING and costs pre-cue 42% of its accuracy. (`causal` also lowers post-cue, to 0.616, because a
causal filter distorts phase inside the post-cue window too; that is why `fitonly` is the preferred fix
rather than merely switching to `lfilter`.) The `zerophase` arm reproduces the live pipeline's numbers,
which is the check that the rebuild is faithful.

Per animal (`fitonly`, chance 0.167): PS92 **0.202** (−0.277), PS93 0.268 (−0.216), PS94 **0.434**
(−0.058), PS95 0.247 (−0.288). PS94's pre-cue code is essentially intact and unambiguously real; PS93
and PS95 retain a real but much smaller one; **PS92 is not convincingly above chance** (naive z=3.2,
and block-CV correlation makes that optimistic). So the cohort is no longer uniform — which is itself a
result, not just a downgrade.

**SCOPE.** Only PRE-CUE analyses are materially affected: the pre-cue decode slides, the frozen and
joint-basis pre-cue arms, `precue_lickfree`, `precue_attribution`, and the "motor-independent
maintained code" claim below. Post-cue and lick-aligned analyses are NOT — the real response sits
inside those windows (`locanmf_rsa` uses `align="lick"`), and the control above shows them unmoved.
The lick and vision controls remain VALID: they asked whether pre-cue information could be explained by
licking or vision, and the answers (no, and no) stand. Neither was ever a question about where the
information came from in TIME, which is why none of them caught this — the filter is upstream of all
three.

**Lesson.** Three independent confound controls were stacked on top of a readout whose PREPROCESSING
had never been checked for acausality. Controls downstream of a broken transform all inherit the break.
Check the signal chain before stress-testing the inference.

**The `last1.0` >> `first1.0` asymmetry is the shadow's shape, not the plan's.** A window sweep
(16 sessions) had found the last 1.0–1.5 s of the pre-cue window carrying ~2x the position information
of the first 1.0 s, which reads naturally as "the plan builds toward the cue". The shadow is largest
immediately adjacent to the event and decays away — the same profile. The sweep must be repeated on
`fitonly` data before any window default is changed; setting it to 1.2 s on the current data would have
tuned the analysis toward the artifact. Sub-binning the window (2/4/8 bins) never beat the plain mean
(−0.014 to −0.034, better in only 5–6/16), so a time-series decoder is not the answer either.

**PROVENANCE — this comes from the upstream method, not from our code (checked 2026-08-13).**
Priya asked whether the Churchland lab (whose pipeline this descends from) does the same. They do.
`churchlandlab/WidefieldImager/Analysis/SvdHemoCorrect.m` builds `[b,a] = butter(2, 0.2/sRate,'high')`
— MATLAB normalises Wn to Nyquist, so `0.2/sRate` is **exactly 0.1 Hz at any frame rate** — then
overwrites BOTH channels IN PLACE with the zero-phase result:

    blueV(~isnan(blueV(:,1)),:) = single(filtfilt(b,a,double(blueV(~isnan(blueV(:,1)),:))));
    hemoV(~isnan(hemoV(:,1)),:) = single(filtfilt(b,a,double(hemoV(~isnan(hemoV(:,1)),:))));
    ...
    Vout = blueV - hemoV*T';

No unfiltered copy of `blueV` is kept, so the FILTERED blue channel is what becomes the corrected
output. `jcouto/wfield` (`wfield/utils.py`, still current on master) is a faithful Python port —
`butter(2, w/(fs/2.), btype='highpass')` + `filtfilt(..., padlen=50)`, default `freq_highpass=0.1`.
Musall et al. 2019's methods state it outright: "SVT was high-pass filtered above 0.1Hz using a
zero-phase, second-order Butterworth filter." So this is the field-standard widefield pipeline, and we
inherited it unchanged. (NB an automated read of the MATLAB initially reported that the output used the
UNFILTERED blueV; verbatim inspection showed the in-place overwrite. Do not trust a summary on a point
this load-bearing.)

**The artifact class is published, just not flagged in widefield pipelines.** van Driel, Olivers &
Fahrenfort (2021, *J Neurosci Methods* 371:109080; bioRxiv 530220), "High-pass filtering artifacts in
multivariate classification of neural time series data": high-pass filtering temporally DISPLACES
information and produces spurious decoding in windows that should be empty, including the pre-stimulus
window, at cutoffs as low as **0.05 Hz** — and they report the spurious decoding as NEGATIVE, matching
the sign-flipped shadow. Their recommended fix is trial-masked robust detrending (mask the evoked
events out of each trial before estimating the drift), which is a better long-term answer than either
`causal` or `fitonly`.

**Why the Churchland analyses are less exposed than ours.** The artifact bites hardest when you
(a) DECODE (b) a condition-specific label (c) from a window PRECEDING the condition-specific response —
which is precisely our pre-cue analysis. Musall et al.'s headline analyses are encoding models (ridge
from task + video-derived movement regressors to activity) and variance-explained comparisons; their
model spans the pre-stimulus baseline but the central claim (uninstructed movements dominate) does not
rest on above-chance pre-stimulus condition decoding. Caveat: that characterisation is from the methods
section and abstract, not a full read of the paper.

**This does NOT contradict the prior that motor plans are decodable before action** (Priya's point, and
it is well founded). That literature is largely spiking data, which carries no such filter. Our own
corrected numbers agree with the prior: PS94 still decodes position at 0.434 pre-cue (chance 0.167),
PS93 0.268, PS95 0.247. The expectation was right; what was wrong was the effect SIZE, inflated ~2x by
preprocessing.

**QUIET-ONLY DETRENDING IS NOT ESTIMABLE IN THIS TASK (measured 2026-08-13).** Priya asked whether the
drift could be fitted on behaviourally quiet periods only, so that running/licking activity (which
Musall et al. show dominates cortex) cannot contaminate the trend. Correct in principle, but there is
almost no such data: quiet bouts from `behavior_events` (not running AND not licking AND not
peri-reward) occupy **0.3% of PS92's 8/10 session — 30 seconds out of 156 minutes** — 2.4% for PS93 and
PS94, and 10.5% for PS95, against 5,000–10,800 licks per session. Intersected with "outside every
trial" the mask collapses to 0.0 / 0.4 / 0.7 / 5.3%, and the detrender silently returned the data
untouched (PS92 and PS94 reproduced `fitonly` to the digit). The only reliably quiet epoch in this task
IS the enforced-no-lick ENL, which is the window under measurement, so it cannot serve as the drift
reference. NB this also bears on the quiet-vs-running activity maps in the preprocessing deck: PS92's
"quiet" map rests on ~30 s of data, and the 0.3%-vs-10.5% spread makes cross-animal quiet/running
contrasts badly unbalanced.

**TASK-MASKED DETRENDING IS THE BEST VARIANT MEASURED (`taskdetrend`).** Drop the quiet requirement,
mask only the evoked window (cue−0.5 s → cue+4 s), and estimate the trend as a 60 s windowed MEDIAN over
the remaining 56–73% of frames, interpolated across gaps. A 60 s median spans ~5 trials, so it cannot
track a 2 s trial-locked response — which is what van Driel's trial-masking is guarding against — and
being a median over a mask rather than a convolution, it has no impulse response and cannot displace
signal in time at all.

| variant | PRE-CUE | post-cue (control) |
|---|---|---|
| `zerophase` (current) | 0.488 | 0.684 |
| `fitonly` | 0.306 | 0.702 |
| **`taskdetrend`** | **0.256** | **0.740** |

**`taskdetrend` gives the BEST post-cue decoding of the three** while pushing pre-cue the lowest: the
variant that most improves the readout we trust is the one that most reduces the readout we suspect.
It also shows `fitonly`'s 0.306 was itself partly inflated by the residual drift `fitonly` leaves in.
Per animal, `taskdetrend` pre-cue (chance 0.167): PS92 **0.141** (at/below chance), PS93 0.244,
PS94 0.347, PS95 0.291.

**RECOMMENDATION: `taskdetrend`, not `fitonly`.** Still to do before adopting: repeat over more
sessions, run the pre-vs-post sign test under it (the zerophase/fitonly sign test is done; the
taskdetrend arm is not), and sanity-check the 60 s window and the cue−0.5/+4 s mask bounds.

**EARLIER PROPOSAL, now second choice: `fitonly`.** Keep the high-pass for estimating `rcoeffs` — that is
what it is for, keeping slow drift from biasing the 470-vs-415 regression — and apply the correction to
UNFILTERED data, so no filter fingerprint reaches the analysed signal. `wfield`'s function does not
support this, so it needs a local reimplementation in `run_wfield_local`. Re-running the correction is
cheap (`SVT.npy` and `U.npy` are retained; seconds per session), but everything downstream of
`SVTcorr` must be redone: LocaNMF (GPU hours), the joint bases, every decoder/encoder, the decks.

### The joint basis is now used for DECODING/ENCODING across days, not only RSA (2026-08-13)
`wfield_local/joint_xsession.py`. The cross-day decoder/encoder (`pooled_frozen_loso`) had to use
Allen-ROI features because a session's own LocaNMF components are session-specific in count AND
identity — it raises `ValueError` if you try. That left the entire cross-day arm, including the
pre-stroke dress rehearsal for train-pre/apply-post, resting on ONE parcellation, and on the coarser
of the two: 66 anatomical ROIs, where LocaNMF's ~95–137 functional components decode better within a
session in 4/4 animals (0.824 vs 0.763).

The persisted joint basis removes the blocker — its footprints A are shared across the animal's
sessions by construction, and a day not in the fit is PROJECTED onto them (`Basis.project`) rather
than refitted — so component *j* is the same footprint every day and pooling is well posed. First
measurement (PS95, pre-cue, 9 curated sessions): **joint LOSO 0.636 vs ROI 0.586**, within-day 0.583
vs 0.528. Both bases now appear side by side in deck Section D; a cross-day effect present in only one
of them is a fact about the parcellation, not about cortex.

**⚠ CORRECTION (2026-08-13, same day): `variance_captured` is NOT a sufficient health check.** The
claim below — that it "keeps the post-stroke readout falsifiable" — is too strong and was refuted by the
first measurement that could test it. 8/12 was the only PROJECTED day for each animal, and the open
question was whether its joint-basis drop was the projection or the day. ROI adjudicates it, because ROI
has no in-fit/projected distinction. Post-cue, 9 curated sessions, delta = 8/12 minus the mean of the
other eight:

| animal | ROI | joint | projection cost (joint−ROI) | variance_captured |
|---|---|---|---|---|
| PS92 | +0.059 | −0.113 | **−0.172** | 99.2% |
| PS93 | +0.051 | +0.014 | −0.037 | 98.9% |
| PS94 | −0.038 | −0.005 | +0.033 | 99.0% |
| PS95 | +0.005 | −0.136 | **−0.141** | 99.4% |

**It is the PROJECTION, not the day**: in ROI, 8/12 is an ordinary session and in fact BETTER than its
siblings in 3 of 4 animals (mean +0.019), while the joint basis costs it a mean −0.079. And
`variance_captured` sits at 98.9–99.4% throughout — it shows green while PS92 loses 0.172. It measures
whether a session's total ENERGY lies in the frozen subspace, not whether the position-DISCRIMINATIVE
directions survive, and those are different questions: the discriminative signal is a tiny fraction of
the variance, so it can be mangled while 99% of the energy is reproduced.

**Consequence for the post-stroke design, which is the reason this matters.** Every post-stroke session
will be a projected day, so a projection cost of ~0.08 (up to 0.17 per animal) is baked in BEFORE any
lesion effect, and the shipped diagnostic will not reveal it. Options, in order of preference:
  1. **Measure the projection cost on held-out PRE-stroke days** — refit the basis without day *k*,
     project day *k*, record the drop, for every *k*. That gives a per-animal projection-cost baseline
     against which post-stroke changes are read (like the frozen decoder's transfer cost, which is
     already reported this way). Costs N basis refits per animal in GPU time, and is the honest answer.
  2. Keep **Allen-ROI as the PRIMARY** cross-day/post-stroke readout — it involves no projection at all
     — and use the joint basis as the sensitivity analysis rather than the headline.
  3. Do NOT simply refit the basis to include every pre-stroke day and call it solved: that removes this
     instance of the problem while leaving post-stroke days projected and uncalibrated.
A per-animal projection cost is also worth reporting on the basis-health slide ALONGSIDE
`variance_captured`, since the two disagree.

**Two honesty requirements, both shipped.** (1) `variance_captured` is plotted per session
(`joint_basis_health_*.png`): in-fit days are 1.0 by construction, projected days are not, so a
projected day that decodes poorly *and* spans poorly is under-described by the basis rather than
representationally changed. Every post-stroke session will be a projected day, so this panel is the
one that keeps the post-stroke readout falsifiable. (2) The basis was fitted using the in-fit days'
data — unsupervised, no labels, but still transductive — so those days carry a small advantage that
projected days do not. **Open check:** 8/12 is the only projected day per animal and, in PS95, the only
session with a negative transfer cost (−0.027, LOSO 0.506 vs +0.025…+0.095 for the eight in-fit days).
That is either the transductive gap or a property of 8/12. It is decidable: the ROI basis has no
in-fit/projected distinction, so if 8/12 also drops there it is the day, not the basis.

**RESOLVED — the ROI caution above was correct.** The original table had Allen-ROI worst on crossnobis
(+0.258) while fine on 1−Pearson (+0.571), and this file recorded the suspicion that it was an
estimator artifact of the DIAGONAL whitening. It was. Switching `_crossnobis_rdm` to the Ledoit-Wolf
shrunk inverse covariance moved ROI from **+0.258 to +0.817 — worst to best.** Measured cause: the
noise covariance of 66 Allen ROIs has ONE eigenvalue holding 80.7% of the variance and a participation
ratio of **1.5 effective dimensions out of 66** (PS95_0810). The ROIs tile 99% of the masked brain, so
they are a partition of a spatially smooth signal and a brain-wide fluctuation (arousal, breathing,
residual hemodynamics) moves all of them together. Diagonal whitening cannot remove a shared mode — it
rescales axes but cannot rotate them. The fitted shrinkage is small (λ ≈ 0.013–0.025), so it is the
full covariance doing the work, not the shrinkage.

For LocaNMF the shrinkage is not an improvement but a PREREQUISITE: its noise covariance is exactly
singular (smallest eigenvalue 0, condition number 1e28), because ~151 components are fitted to
rank-~100 SVD data and must be linearly dependent. There is no unshrunk inverse to use.

**Σ is estimated from HELD-OUT folds** (2026-08-12): the folds supplying neither pattern in the
cross-fold product, so the whitening matrix is independent of the data it whitens. Effect measured and
negligible — ROI +0.817 → +0.821, LocaNMF +0.767 → +0.754. Kept because it is the defensible
estimator, not because it changed an answer. The circularity was WITHIN a session while the metric is
agreement BETWEEN sessions, which is why it could bias distance magnitudes without manufacturing
cross-session agreement.

## Projecting NEW sessions into a frozen joint basis (2026-08-12)
`joint_locanmf.Basis.project()` refits C with the footprints A held FIXED, so a session outside the fit
— a new pre-stroke day, and every post-stroke day — is expressed in the SAME components. Without it the
basis would only describe the sessions it was built from, which is not a reference frame.

`pinv(A)` is computed as `pinv(A'A) A'` (exact at any rank) rather than on the tall (345600, ncomp)
matrix, whose SVD **does not converge**; A is ~40% NaN (LocaNMF writes NaN outside each footprint's
region) and must be zero-filled first. The least-squares is restricted to in-brain pixels — a pixel
finite for ANY component — because LocaNMF never fit the rest.

**Coefficient agreement is NOT the acceptance test, and would have failed.** Against the in-fit
components, per-component correlation is excellent at the median (0.996–0.999) but has a tail
(10th percentile 0.36–0.61, min ≈ 0). **Ridge does not fix this — measured, it makes the tail
monotonically worse** (λ/mean-eig 1e-4 → 10th pct 0.24, 1e-2 → 0.03). λ=0 retained.

**Diagnosed, not inferred (2026-08-12).** The tail is a LABELLING AMBIGUITY between overlapping
components, not lost signal. LocaNMF puts several components in one Allen region and `loc_thresh=80`
confines each to that region, so their footprints are near-parallel: median pairwise cosine overlap
**0.977**, with 71–79% of components overlapping another by >0.5. Where `A₁ ≈ A₂`, the reconstruction
depends on `c₁ + c₂` and the data cannot fix the split, so LocaNMF's regularised solution and a
fixed-A least-squares refit land at different points on the same ridge. Three predictions were tested:

| prediction | result |
|---|---|
| agreement falls as a component's max overlap rises | **confirmed**, Spearman ρ = −0.819 / −0.834; the 29 and 38 low-overlap components recover at median r = **1.0000** |
| the GROUP SUM over overlapping components recovers where members do not | **confirmed**: 10th-pct r rises **+0.340 → +0.910** (PS92) and **+0.496 → +0.988** (PS95) |
| the failures are the WEAK components | **refuted**, ρ = −0.02 / −0.32; the weaker half agrees slightly *better* |

This is why the answers are unaffected. Splitting amplitude between collinear components is an
invertible linear reparameterisation of the feature vector, and both downstream readouts are (near)
invariant to one: Mahalanobis/crossnobis distance is exactly invariant under an invertible linear map,
and regularised logistic regression nearly so. Hence RDM agreement +0.950–0.993 and decode within
±0.02. The caveat that follows: **do not interpret an INDIVIDUAL projected component's amplitude** as
that patch's activity — interpret the overlapping group, or use a low-overlap component.

What settles it is that the components are only ever a FEATURE SET, so the test is whether the answers
move. On in-fit sessions, features from `project()` vs LocaNMF's own C give **decode accuracy within
−0.022 to +0.014 (mean −0.007) and crossnobis RDM agreement +0.950 to +0.993**. The tail is cosmetic.

`variance_captured` is returned alongside and must be reported with any projected result — it is the
health check for applying a pre-stroke basis to post-stroke data (in-fit sessions: 99.4–99.7%). It is
**not** a guard against using the wrong animal's basis: a PS92 basis spans a PS95 session at 97%,
because both are cortex on the same Allen grid. Only the label can catch that, so `project()` refuses a
cross-animal session unless `allow_cross_animal=True`.

## "Pre-cue" is AFTER the spout arrives — the maintained-code claim needs rewording (2026-08-13)
**Measured: the spout reaches its position a median 3.0 s before the cue** (p10 2.3 s; 100% of trials
lead by more than the 0.5 s analysis window). The pipeline's pre-cue window is the 2 s ENDING at the
cue, so it lies ENTIRELY AFTER spout arrival. Above-chance decoding there therefore cannot, on its own,
demonstrate a motor-independent MAINTAINED PLAN: it is equally consistent with an ongoing sensory or
postural consequence of the spout already being in place. The claim as recorded ("pre-cue no-lick
decode above chance = motor-independent maintained code") overstates what the window can support.

Two candidate confounds were tested, with opposite outcomes:

**VISION — REJECTED.** Removing every visual ROI costs nothing: 0.397→0.397, 0.371→0.400, 0.436→0.425,
0.257→0.266 (PS92/PS95, spout-arrival aligned). VIS alone stays above chance, but that is redundancy,
not sourcing — the same sufficiency/necessity trap `precue_attribution` exists to avoid. Note VIS
carries 71 of 151 LocaNMF components and decodes at 0.590 alone, which looks damning and is not.

**SOMATOSENSATION — OPEN, and the leading account.** `precue_attribution` (4 animals x 8 sessions, both
bases) finds **SSp is the ONLY family whose removal costs anything** — ROI necessity +0.061..+0.106,
LocaNMF +0.011..+0.056 — while MOp/MOs removal costs ~0 everywhere. That is what whisker/proprioceptive
contact with the positioned spout would look like. BUT **MOs carries comparable encoding EV**
(LocaNMF: PS93 +0.049 = SSp's, PS92 +0.032, PS95 +0.026) with zero necessity, i.e. premotor cortex
REPRESENTS position without being required to decode it — which somatosensory contact does not
explain. So the maintained-plan account is not refuted, merely unsupported by pre-cue accuracy alone.

**A pre-spout-arrival test is NOT available from this design.** Restricting to first-in-block trials
(position just changed) was tried; its negative control FAILED — decoding ran 0.29-0.49 BEFORE the
spout moved, against 0.167 chance. Cause diagnosed: **the task avoids recent repeats.** When the last
5 blocks were all distinct, the next block was the missing position 45-53% of the time (vs ~17%
uniform), so lingering representation of recent positions carries real information about the upcoming
one. That is task statistics plus history, not prediction. The analysis was discarded rather than
interpreted. Any future "does the code precede the stimulus" test needs either randomisation with
replacement or an explicit history regressor.

## Reliability ≠ information: the RSA ranking does not say ROI carries more (2026-08-12)
Mean sibling RSA is a RELIABILITY measure and must not be read as "which basis carries more positional
information". Measured on the same sessions, ROI vs per-session LocaNMF:

| | ROI | LocaNMF | winner |
|---|---|---|---|
| within-session split-half RDM, **crossnobis** | **+0.812** | +0.694 | ROI, 4/4 animals |
| within-session split-half RDM, **1−Pearson** | +0.696 | **+0.744** | LocaNMF, 3/4 |
| cross-session sibling RSA, crossnobis | **+0.821** | +0.754 | ROI, 4/4 |
| **block-CV decode accuracy** (chance 0.167) | 0.763 | **0.824** | LocaNMF, **4/4** |

**LocaNMF decodes better in every animal** (+0.061 mean; PS92 +0.089, PS93 +0.064, PS95 +0.066, PS94
+0.024) — including PS93, whose ROI decoder was flagged as weak. So LocaNMF carries MORE position
information and still yields a NOISIER RDM. The dissociation tracks exactly one thing: whether the
quantity requires estimating a covariance in feature space. Decoding does not (per-feature scaling +
regularized logistic regression) and LocaNMF wins; the 1−Pearson RDM does not and LocaNMF wins; the
crossnobis RDM does, and ROI's 66 well-conditioned features beat LocaNMF's 151 rank-deficient ones.
The RSA penalty is an estimability cost, not an information deficit.

Consequence for the deck: **basis choice follows the question.** Decoders and encoders are reported in
BOTH bases; the RSA figures use a shared basis and carry this caveat, because an ROI-space RDM is
blind to whatever information the decoder shows LocaNMF is capturing.

## Pre-cue and post-cue geometry are largely the SAME (2026-08-12)
Per session, the 6×6 RDM built from the 2 s window ENDING at the cue vs the 2 s after it, then
correlated. **Re-measured with the corrected whitening (2026-08-12); the crossnobis values below
supersede the earlier +0.572 / +0.218, which were diagonal-whitening artifacts:**

| | precue↔cue | precue↔lick | cue↔lick |
|---|---|---|---|
| LocaNMF, 1−Pearson | +0.694 | +0.755 | +0.869 |
| LocaNMF, crossnobis | **+0.827** | **+0.817** | +0.931 |
| Allen-ROI, 1−Pearson | +0.500 | +0.562 | +0.863 |
| Allen-ROI, crossnobis | **+0.843** | **+0.850** | +0.939 |

The conclusion is STRONGER than the old numbers supported: pre-cue geometry agrees with both post-cue
windows at +0.82–0.85, not +0.57–0.63. The positional geometry is largely established BEFORE movement
and is not reorganized by execution — consistent with the pre-cue decode being above chance. Caveat:
cue↔lick (+0.93) is inflated by window overlap (the lick usually falls within 2 s of the cue), so the
pre-cue pairs are the clean comparisons — and they are the ones that moved most.
NB the "% of reliability ceiling" column in that analysis is INVALID as computed (split-half
reliability uses half the data and was not Spearman-Brown corrected, and several sessions have
near-zero or negative pre-cue reliability, giving impossible >100% values). Use the raw agreement.

## ROI vs LocaNMF features: LocaNMF wins per-session, POOLING erases the gap
Head-to-head on the same 10 sessions, same alignment (cue 2 s), same block CV, same `C=0.5`:

| | mean accuracy | vs LocaNMF within-day | paired t |
|---|---|---|---|
| **LocaNMF, within-day** | **0.734** | — | — |
| Allen-ROI, within-day | 0.697 | **−0.037** (LocaNMF wins 8/10) | t=3.03, **p=0.014** |
| Allen-ROI, **FROZEN** (held-out day) | 0.731 | −0.003 (LocaNMF wins 7/10) | t=0.18, **p=0.86** |

- **LocaNMF is genuinely the better feature space per session** (+0.037, significant) — it is not
  merely a prettier basis. That justifies keeping it as the per-session primary.
- **But a frozen ROI decoder that has NEVER SEEN the test day matches a LocaNMF decoder trained on
  that very day** (p=0.86). Pooling ~3000 trials across days exactly offsets the feature-quality
  loss.

So for the cross-day / post-stroke arm, Allen-ROI is **not a compromise** — it buys transferability
for free. Use LocaNMF per-session (primary) and frozen ROI across days (confirmatory), as planned.

## The frozen ENCODER does NOT transfer like the decoder (measured 2026-08-11/12)
`pooled_frozen_encoder` mirrors the frozen decoder (ridge one-hot position -> Allen-ROI activity,
fit on an animal's OTHER curated days, evaluated on the held-out day). Over 8 curated sessions/animal:

| animal | decoder transfer cost | **encoder** transfer cost | encoder FEVE |
|---|---|---|---|
| PS92 | +0.103 | +0.001 | 0.51 |
| PS93 | +0.009 | **−0.063** | **0.09** |
| PS94 | +0.042 | +0.006 | 0.70 |
| PS95 | +0.053 | **−0.032** | 0.60 |

**The two disagree in sign, and that is the point.** The DECISION BOUNDARY transfers across days —
every animal's decoder *gains* from pooling ~3000 trials instead of ~500. The ACTIVITY MAGNITUDES do
not: the encoder estimates only 6 position means per feature, so it has little to gain from extra
trials and is actively hurt by day-to-day differences in the mapping that per-session z-scoring does
not remove.

Two consequences for the stroke arm:
1. Read the intention with the **frozen decoder**, not the encoder.
2. Judge post-stroke encoder residuals against this **non-zero pre-stroke cross-day cost**, not
   against zero — otherwise normal day-to-day drift will read as a lesion effect.

PS93 is the outlier on both (FEVE 0.09, several NEGATIVE frozen EVs), consistent with its being the
animal with the lowest noise ceiling: least position signal available, not a model failure. Always
report PS93's ceiling beside its accuracy.

## Confidence is NOT evidence: the OOD control is mandatory (`ood_control`)
A softmax decoder never abstains — on input with no position information it still emits a normalized
distribution, and it does so **confidently**. Measured on quiet/running windows (no position is even
defined there): normalized entropy 0.24–0.54 and mean max-probability up to **0.997**, with predictions
collapsing onto a single attractor position (PS92 8/10 quiet: 100% of windows → `far_center`). That is
*lower* entropy — i.e. more confident — than the shuffled-label floor (0.97–0.99).

So a frozen decoder's post-stroke confidence must never be read as preserved coding on its own. Always
report it against the two references `ood_control` emits: the **shuffled-label entropy floor** (empirical
no-information) and the **no-lick trials**.

**No-lick trials carry no position code** (the nearest pre-stroke analogue of a failed attempt). Pooled
across curated sessions, well powered at last (per-session n is only 6–153): PS95 n=396 acc 0.179 CI
[0.142, 0.217]; PS94 n=351 acc 0.205 CI [0.163, 0.247]; PS92 n=189 acc 0.132 CI [0.084, 0.181] — **all
CIs include chance 0.167**, against 0.71–0.86 for engaged trials in the same sessions. The encoder agrees:
EV on no-lick trials is ≈0 once the engaged-vs-unengaged baseline offset is removed (raw −0.28 to −1.04
is mostly a mean shift, not an inverted mapping; re-centred: −0.06, −0.03, −0.02).
Combined with pre-cue decoding at 0.62 on engaged trials, the readout is a maintained position code that is
**gated by engagement** — present before movement, absent on trials the animal will not act on.

---

# Module & output reference
Key analysis modules (`wfield_local/`): `run_locanmf.py` (LocaNMF/sNMF on the GPU box) ·
`locanmf_position_decoder.py` (**the decoder**; `--source locanmf|roi`, `--align cue|lick|precue`, per-area
accuracy + per-position recall + confusion) · `locanmf_position_encoder.py` (per-position EV / FEVE /
predicted maps) · `locanmf_cross_mouse.py` (cross-mouse + within-animal consistency) · `locanmf_rsa.py` (RSA
+ noise ceiling + hemisphere-resolved + crossnobis) · `locanmf_decoder_weights.py` (rolling/temporal figs) ·
`roi_activity.py` (CPU Allen-area ROI traces) · `quiet_periods.py` (quiet-frame mask) · `atlas_overlay.py`
(shared region outlines) · `framemap_event_maps.py` (regime-B cue/lick maps) · `qc_motion_correction.py` ·
`cross_day_align.py`. The nightly orchestrators are `preprocess.py` (imaging) and `nightly_figs.py`
(analysis). Figures/tables on MICROSCOPE under `labcams/locanmf_lick_pooled/…/cue_analysis/`; per-session
LocaNMF outputs in each session's `motion_corrected/locanmf_affine8v1_final/`. The full historical module list
(early lick/cue exploratory modules) is in `docs/archive/ANALYSIS_HISTORY.md`.
