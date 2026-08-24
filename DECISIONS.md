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
  - **False-start / multi-recording DAQ caveat (found 2026-08-18, PS93_0818).** The relabel's automatic
    exposure-offset search only tries **0/1**, so it assumes the `.dat`'s first frame is the DAQ's first
    `pco_exposure` pulse. That breaks when the DAQ ran through an aborted first acquisition: PS93 on 8/18 had
    a false-start recording (camera started, unsure it was recording, stopped + deleted, PCO timestamp
    renamed, restarted), so the DAQ's `pco_exposure` had **1,174 orphan pulses (t=2.2–21 s), a 20 s gap, then
    recording #2 (t=41 s→, = the analyzed `.dat`)**. The relabel mis-locked to the false-start pulses
    (`chosen_exposure_offset=0`), which — because rescue-mode pairing interacts with the leading chunk —
    corrupted the **415/470 pairing for ~58 % of the session** (verified by regenerating the frame map and
    diffing `original_frame_index`), not merely the cue times. Symptom downstream: `hemo_variants` divides by
    all-zero weights (`weights are all zero for channel 0` → `UnboundLocalError 'V'`) and maps report cues
    outside coverage. **Recovery:** trim the DAQ to drop everything before recording #2's first exposure, then
    re-run the session's **full** preprocessing (the bin is NOT reusable — the pairing changed). The original
    untrimmed DAQ is kept for provenance at
    `labcams/20260818/PS93_20260818_145203/daq_falsestart_recording1/PS93_20260818_recording1_falsestart_UNTRIMMED_DAQ.h5`
    (+ a `README.txt` there; local backup `E:\_daq_provenance\`). It is deliberately **NOT** under
    `DAQ_recorder_output/` — `archive_day.discover_daq` walks that tree and re-uploads any `.h5` whose name
    contains the date, so a provenance copy there gets swept back onto the server as a session DAQ. A cleaner
    long-term fix would be to widen the offset search to detect a leading orphan-recording block automatically.
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
- **`preprocess_deck` (`build_decks`) is SAFE to run** — see "Deck writes are guarded, not banned"
  below. The old blanket prohibition described a prune that no longer exists unguarded, and it had a
  cost: it routed people onto a hand-rolled `build_deck` call, which is exactly where 257 MB of PS93's
  deck was destroyed on 2026-08-19.
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

**NUMBERS REPLACED 2026-08-17 (bug 17).** The 8/11 table below the line is superseded: from
2026-08-14 to 2026-08-17 `_align_many` resolved tiled sub-bin labels with `list.index()`, so every
ROI frozen number ran on four copies of bin 0. The CONCLUSION survives intact -- transfer cost is
positive for every animal in both alignments -- but the magnitudes were wrong. Recomputed on all 11
curated sessions per animal:

| animal | align | LOSO (unseen day) | within-session | transfer cost |
|---|---|---|---|---|
| PS92 | post-cue | 0.890 | 0.750 | **+0.140** |
| PS93 | post-cue | 0.803 | 0.734 | **+0.068** |
| PS94 | post-cue | 0.935 | 0.864 | **+0.071** |
| PS95 | post-cue | 0.922 | 0.850 | **+0.072** |
| PS92 | PRE-cue | 0.471 | 0.312 | **+0.159** |
| PS93 | PRE-cue | 0.464 | 0.386 | **+0.078** |
| PS94 | PRE-cue | 0.658 | 0.540 | **+0.117** |
| PS95 | PRE-cue | 0.448 | 0.324 | **+0.124** |

Post-cue moved by +0.23 to +0.41 and pre-cue barely moved, which is the mechanism confirming itself:
collapsing to bin 0 destroys a window with real temporal structure and costs almost nothing on a
near-stationary one. Corrected ROI post-cue now agrees with the joint-LocaNMF basis (0.80-0.94 vs
0.935), as two bases should.

<details><summary>superseded 8/11 numbers (single-mean features, pre-sub-binning)</summary>

| animal | pooled trials | LOSO (unseen day) | within-session | transfer cost |
|---|---|---|---|---|
| PS92 | 2593 | 0.701 | 0.600 | **+0.102** |
| PS93 | — | 0.597 | 0.585 | **+0.012** |
| PS94 | 3535 | 0.753 | 0.709 | **+0.044** |
| PS95 | 3660 | 0.863 | 0.816 | **+0.047** |

</details>

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

### ⚠ WINDOW SWEEP RE-RUN ON CORRECTED DATA — BOTH ANSWERS REVERSE (2026-08-13)
The original sweep ran on `zerophase` data and was withdrawn as "measuring the shadow's shape". Repeated
on rebuilt `SVTcorr` with the drift fit held OUT of the measured window (`strobedetrend`), Allen-ROI
features in BOTH arms so the comparison is matched, 16 sessions (4 animals x 6/7, 8/6, 8/10, 8/11), C by
nested CV:

| arm | zerophase | vs mean2.0 | strobedetrend | vs mean2.0 |
|---|---|---|---|---|
| **rolling** (4 x 0.5 s) | 0.495 | −0.020 (better 4/16) | **0.354** | **+0.045 (better 12/16)** |
| mean 2.0 s | 0.515 | — | 0.309 | — |
| mean 1.5 s | 0.549 | +0.034 (12/16) | 0.295 | −0.013 (4/16) |
| mean 1.0 s | 0.546 | +0.031 (11/16) | 0.286 | −0.023 (3/16) |
| first 1.0 s | 0.300 | −0.214 | 0.330 | +0.021 (10/16) |
| **asymmetry** last1.0−first1.0 | **+0.245** | | **−0.044** | |

1. **The asymmetry WAS the artifact.** +0.245 -> −0.044. "The code builds toward the cue" is gone; if
   anything the FIRST second is marginally better. Position information is spread evenly across the
   pre-cue window — what a maintained code looks like, not what a decaying backward shadow looks like.
2. **Shortening the window no longer helps.** mean1.5 went +0.034 (12/16) -> −0.013 (4/16). **2.0 s is
   now the best mean**, so `precue_post_s: 2.0` STAYS. The 1.2/1.5 s idea was chasing the shadow, and
   adopting it would have tuned the analysis onto the artifact — the reason it was not adopted.
3. **Rolling now WINS**, having lost before: −0.020 (4/16) -> **+0.045 (12/16)**. A 4 x 0.5 s sub-binned
   time course beats the 2 s mean once the artifact is gone. Priya's original instinct was right; it was
   unmeasurable through the shadow, which is large, smooth and temporally structured, so sub-binning
   split it and added noise. Remove it and real temporal structure becomes the signal.

CAVEATS: ROI features (LocaNMF would need a GPU refit per variant); 16 sessions not 36; and
`strobedetrend` has NOT yet been tested with its own refitted `T` (see below). Confirm all three before
changing any default.

### Sub-binning POST-cue and POST-lick — both my predictions were wrong (2026-08-14)
Same 16 sessions, adopted `meegkit_hpfit`, `--align cue` and `--align lick`. I predicted (a) sub-binning
would help MORE post-event than pre-cue, because the response has real dynamics to resolve, and (b)
`roll4x0.5` would again beat `roll8x0.25`. **Neither held.**

| arm | POST-CUE | vs mean2.0 | POST-LICK | vs mean2.0 |
|---|---|---|---|---|
| roll 4 x 0.5 s | 0.791 | +0.020 (10/16) | 0.831 | +0.014 (8/16) |
| roll 8 x 0.25 s | **0.793** | +0.022 (8/16) | **0.840** | **+0.023 (12/16)** |
| roll 2 x 1.0 s | 0.790 | +0.019 (8/16) | 0.827 | +0.010 (7/16) |
| mean 2.0 s | 0.771 | — | 0.817 | — |
| first 1.0 s | 0.688 | −0.084 (1/16) | 0.800 | −0.017 (5/16) |
| last 1.0 s | 0.777 | +0.006 (7/16) | 0.770 | −0.047 (2/16) |

1. ~~**Sub-binning helps LESS post-event than pre-cue**, not more: +0.020/+0.023 here vs +0.032
   pre-cue.~~ **WITHDRAWN 2026-08-17.** The +0.032 came from a 16-session pilot and does not
   replicate: re-measured on all **44 curated sessions**, pre-cue sub-binning gains **+0.009, better
   in 23/44** — a coin flip. `roll2x1.0` is nominally best (0.395, +0.016, 28/44) and `roll8x0.25`
   is actually *worse* than the mean (−0.002, 18/44), but with six arms scored on the same sessions
   the winner is partly selection, and the whole spread is ~0.02. The honest statement is that
   **pre-cue sub-binning is unestablished**, so the comparison this point rested on cannot be made.
   `decode.bins.precue` stays at 4 because moving it would shift every pre-cue number again for no
   demonstrated gain — not because it is better. Post-cue and post-lick were NOT re-run and remain
   16-session values.
2. **Bin width matters post-event but the direction FLIPS vs pre-cue.** Pre-cue, 0.25 s over-sliced and
   lost to 0.5 s. Post-lick, 0.25 s WINS (+0.009, 12/16); post-cue the two tie (+0.002, 8/16). Read
   together: fine bins pay off only where there are fast dynamics to resolve, and post-lick is the
   fastest-moving window we have. This is consistent with the 150 ms lick MAP resolution.
3. **The temporal asymmetry REVERSES between alignments**, which is the most interpretable result here.
   Post-cue `last1.0 − first1.0` = **+0.089** (12/16) — information accumulates after the cue. Post-lick
   it is **−0.030** (6/16) — information is already maximal at lick onset and decays. That is what
   movement-locked information should look like, and it is the opposite of an accumulating cue response.
4. **PRE-CUE, the asymmetry points AWAY from the cue (44 sessions, 2026-08-17).**
   `last1.0 − first1.0` = **−0.054**: the FIRST second of the pre-cue window (−2.0 to −1.0 s) carries
   MORE position information than the second adjacent to the cue (−1.0 to 0 s), which is also why
   `first1.0` (0.387) beats `last1.0` (0.333) outright and ties the adopted sub-binned arm.

   This matters more than the bin-count question it came from. The ORIGINAL, pre-correction sweep
   found the opposite — the last 1.0–1.5 s carrying ~2× the information — and that was withdrawn as
   the zero-phase filter's backward shadow (largest adjacent to the event, decaying away). With the
   shadow gone the gradient does not merely shrink, it **flips**: information is highest FURTHEST
   from the cue, i.e. closest to spout arrival, and decays as the animal sits quiet through the ENL.

   That is evidence on the question `DECISIONS.md` already flags as unresolvable by this design
   ("Pre-cue is AFTER the spout arrives"): a held intention should not decay while the animal waits,
   whereas a somatosensory/contact response to a spout that arrived seconds ago should. It does not
   settle it — the animal's last licks also sit at the early end of the window, and lick-free trials
   are the control for that (`precue_lickfree`) — but it is the first result that discriminates at
   all, and it leans somatosensory. Worth re-running restricted to lick-free trials before it is
   quoted.

**NOW THE PRODUCTION DEFAULT (2026-08-14).** These arms previously existed ONLY in
`precue_window_sweep.ARMS`, a research harness, so every deck and cross-session number was
still built from a single window mean -- the adopted sub-binning changed nothing until
`locanmf_position_decoder` learned to build it. It now does (`_window_feature`), with the bin
count per alignment in `configs/defaults.yaml decode.bins` (precue 4, cue 4, lick 8) and
`--bins 1` restoring the historical single mean. Feature count goes 66 -> 264 (4 bins) or 528
(8 bins) for ROI; component->region labels are tiled with the features so the encoder still
groups correctly.

### Activity MAPS carried the filter shadow on disk for a day (2026-08-14)
Flipping `hemo.variant` changed what analyses COMPUTE. It did nothing to map PNGs already rendered, so
every curated date from 6/6 to 8/12 kept showing the zero-phase shadow while the decoders had already
moved to `meegkit_hpfit`. Found by Priya noticing that pre-cue maps still looked anti-correlated with
the cue maps. They were.

Mean correlation between each position's PRE-cue and POST-cue map:

| session | variant | mean r(pre, post) |
|---|---|---|
| PS93_0810 | zerophase | **−0.927** (per-position −0.79 … −0.99) |
| PS92_0811 | zerophase | **−0.929** (per-position −0.88 … −0.99) |
| PS94_0811 | zerophase | −0.277 |
| PS94_0812 | zerophase | −0.146 |
| PS94_0813 | meegkit_hpfit | +0.394 |
| PS93_0813 | meegkit_hpfit | +0.020 |
| PS92_0813 | meegkit_hpfit | −0.070 |
| PS95_0813 | meegkit_hpfit | −0.153 |

On the worst sessions **the pre-cue map is literally the negative of the post-cue map** — the cleanest
picture of the artifact we have, and worth a before/after panel in the deck: far more convincing than
the decode numbers alone. Corrected, the signature is gone (mean ≈ +0.05, scattered both ways).

All ten curated dates were re-rendered. The DURABLE fix is provenance: the map steps record `svtcorr`
in their summary json, `preprocess_deck.map_variant_of()` reads it, and `build_deck` warns by name for
any session whose maps were not rendered from the configured variant — including a summary with NO
provenance, which means it predates 14 Aug and is zerophase in practice. The stale list is returned in
the build summary so a caller can gate on it.

GENERAL LESSON: a config flip propagates to computation instantly and to RENDERED ARTIFACTS not at all.
Anything already on disk needs re-rendering, and needs to carry provenance so nobody has to remember.

### ENCODER sub-binning: NO — keep the window mean (2026-08-14)
The decoder builds features through `_trial_features`, and so does the encoder, so adopting sub-binned
features would have silently turned the ENCODER into a model of each component's 8-bin time course
instead of its mean activity — changing what R², EV and the FEVE ceiling mean. Pinned to `bins=1`
explicitly and then TESTED (`wfield_local/encoder_bins_test.py`, 16 sessions, ROI features on the
adopted variant, ridge position→activity under block CV).

The comparison must use **FEVE, not EV**: more bins = more targets, each intrinsically noisier, so EV
falls with bin count almost by construction. The noise ceiling (between-position SS / total SS)
absorbs exactly that.

| | bins=4 vs 1 | bins=8 vs 1 |
|---|---|---|
| all 16 sessions | +0.024 mean, **+0.004 median**, 9/16 | +0.022 mean, −0.001 median, 8/16 |
| **where the encoder works** (FEVE₁ > 0.5, n=10) | **−0.007**, 4/10 | **−0.012**, 3/10 |
| where it fails (n=6) | +0.075 | +0.079 |

**The headline mean is an artifact and must not be quoted alone.** It is positive only because of the
six sessions where the forward model FAILS — PS92_0607 has FEVE −0.586, i.e. worse than predicting the
grand mean, and sub-binning "improves" it to −0.461, which is one failure replaced by a smaller
failure. Where the model actually works, sub-binning is slightly WORSE, and the median across all
sessions is +0.004 with a 9/16 sign test: a coin flip. `summarise()` now prints the median and the
works/fails split so the mean can never be read alone.

**DECISION: encoder stays on the window mean.** This is a real asymmetry worth stating in the deck —
temporal detail helps you READ position OUT (decoder +0.020 to +0.032) but does not help you PREDICT
activity FROM position.

ADOPTED: `roll4x0.5` for post-cue (the three rolling arms are within 0.003, so take the convention
already used pre-cue), `roll8x0.25` for post-lick (it wins on both effect size and sign test). The
robust claim is the weaker one: ANY sub-binning beats the 2 s mean in both alignments; among rolling
arms the differences are small.

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

**SYSTEMATIC RESULT — ALL 36 CURATED SESSIONS** (4 animals x 9 dates, one pass per session so `SVT.npy`
is loaded once; every variant sees the identical trials and folds, so the comparison is PAIRED and the
per-session difference is the estimate). `python -m wfield_local.filter_acausality_test --modes
zerophase,fitonly,taskdetrend`:

| variant | PRE-CUE | post-cue (control) | corr(pre,post) | vs zerophase |
|---|---|---|---|---|
| `zerophase` (current) | 0.486 | 0.684 | **−0.483** | — |
| `fitonly` | 0.273 | 0.686 | +0.672 | −0.213, lower in **35/36**, p=1.5e-10 |
| **`taskdetrend`** | 0.247 | **0.737** | +0.429 | −0.239, lower in **36/36**, p=2.9e-11 |

**The sign test replicates cohort-wide**: `corr(pre,post)` is NEGATIVE in **30 of 36** sessions under the
current pipeline, and in only 1/36 (`fitonly`) or 4/36 (`taskdetrend`) after correction. The pre-cue
pattern really is an inverted copy of the post-cue pattern.

Per animal, pre-cue (chance 0.167):

| animal | zerophase | fitonly | taskdetrend |
|---|---|---|---|
| PS92 | 0.470 | 0.176 | **0.151 — AT CHANCE** |
| PS93 | 0.477 | 0.252 | 0.227 |
| PS94 | 0.475 | **0.416** | 0.358 |
| PS95 | 0.523 | 0.249 | 0.252 |

**`taskdetrend` is the only variant that IMPROVES post-cue decoding** (+0.053 over the current
pipeline), i.e. it is better preprocessing rather than merely different, and it is simultaneously the
one that most reduces pre-cue. That is the strongest form this comparison could take.

**FOUR VARIANTS, ALL 36 CURATED SESSIONS — `strobedetrend` WINS.** (`--modes
zerophase,fitonly,taskdetrend,strobedetrend`.) `strobedetrend` masks the WHOLE trial
(strobe−0.25 s → cue+4 s) so the drift fit never sees the measured window:

| variant | PRE-CUE | post-cue | corr(pre,post) | sign test | vs zerophase |
|---|---|---|---|---|---|
| `zerophase` (current) | 0.486 | 0.684 | −0.483 | neg in **30/36** | — |
| `fitonly` | 0.273 | 0.686 | +0.672 | neg in 1/36 | −0.213, 35/36, p=1.5e-10 |
| `taskdetrend` | 0.247 | 0.737 | +0.429 | neg in 4/36 | −0.239, 36/36, p=2.9e-11 |
| **`strobedetrend`** | **0.306** | **0.731** | **+0.525** | **neg in 0/36** | −0.181, 35/36, p=4.1e-10 |

Per animal (chance 0.167): PS92 0.231, PS93 0.269, PS94 **0.424**, PS95 0.299 — **all four above
chance**. The earlier "PS92 is at chance" was a `taskdetrend` MASK artifact (0.151), not a property of
the animal; `taskdetrend` left 1.5 s of the pre-cue window inside its drift fit and shaved real signal.
Corrected, the cohort spread is 0.23–0.42 rather than the artifactual 0.47–0.52 uniformity.

### What drift actually needs removing — MEASURED, not assumed (2026-08-13)
Priya challenged the premise that the drift is slow ("is slow drift over hours really the kinetics we
want?"). Two things settle it.

**The hard constraint: position is BLOCKED at 57–121 s** (median 6 trials × 9.5–18.7 s inter-cue). Any
temporal filter with a cutoff at or below the block timescale removes position SIGNAL along with drift
— they are temporally inseparable by construction of the task. This, not "bleaching is smooth", is the
real reason temporal drift removal must stay well slower than a block, and it is exactly what the
hand-rolled 60 s windowed median violated.

**The measurement.** Fraction of variance of the global brain-mean trace by timescale:

| chan | >5 min | 5–2 min | 2–1 min (BLOCK) | 60–10 s | 10 s–0.1 Hz | >0.1 Hz |
|---|---|---|---|---|---|---|
| 415 (isosbestic = drift) | **27–64%** | 5–14% | **2.0–3.5%** | 6–10% | 11–22% | 8–23% |
| 470 (functional) | 5–14% | 7–11% | 5–10% | 18–33% | 25–42% | 11–17% |

415 carries no calcium, so it indexes drift: it is **dominated by >5 min** and has almost nothing at the
block timescale. 470 is the opposite shape — most power at 60–10 s and faster, i.e. signal. So a low-order
polynomial over the session is well matched to the real drift, and a filter reaching the block timescale
would remove mostly signal. LIMITATIONS: the global mean only sees SPATIALLY UNIFORM drift (uneven
bleaching or a focus shift with structure would not appear), and 415 is not pure drift — it carries
blood volume too.

**A cleaner axis, untested: `globalregress`.** Drift is largely GLOBAL; position information is
spatially PATTERNED. Regressing out the global mean (or the first few global components) removes drift
at EVERY timescale — including the block timescale no temporal filter can touch — while leaving
position-differential structure, since a decoder reads across-position differences and the common mode
cancels. Same logic as global signal regression in fMRI, and it makes no claim about drift KINETICS at
all. Worth adding to the variant set and testing against `meegkit`.

**HONEST CAVEAT AGAINST THE PREFERRED OPTION.** `taskdetrend` masks [cue−0.5 s, cue+4 s], so 1.5 s of
the 2 s pre-cue window is still inside the drift fit. A 60 s median cannot track a trial-locked
response, but it may shave a little genuine pre-cue signal, making 0.247 a slight UNDER-estimate.
`fitonly` never touches that window but leaves drift in, which adds nuisance variance. The two bracket
the truth from opposite directions: **the real pre-cue effect is most likely between 0.247 and 0.273.**

**RECOMMENDATION: `taskdetrend`.** Remaining before adoption: a sensitivity check over the detrend
window length and the mask bounds (extending the mask to cue−2 s would keep the fit out of the measured
window entirely) — chosen by reasoning so far, not by measurement, and they must not be tuned on the
outcome they are judged by. Also still to do: re-run the comparison using `precue_lickfree`'s SEARCHED
window as the pre-cue definition, since Section C's numbers use that and are not covered by this test.

**EARLIER PROPOSAL, now second choice: `fitonly`.** Keep the high-pass for estimating `rcoeffs` — that is
what it is for, keeping slow drift from biasing the 470-vs-415 regression — and apply the correction to
UNFILTERED data, so no filter fingerprint reaches the analysed signal. `wfield`'s function does not
support this, so it needs a local reimplementation in `run_wfield_local`. Re-running the correction is
cheap (`SVT.npy` and `U.npy` are retained; seconds per session), but everything downstream of
`SVTcorr` must be redone: LocaNMF (GPU hours), the joint bases, every decoder/encoder, the decks.

### Window and alignment for the POST-CUE / response decoder (2026-08-13)
Priya asked whether to extend the post-cue window to 3 s, because PS93's far-L responses are often
late. Measured on the adopted preprocessing (`postcue_window_test`, 36 curated sessions, per animal AND
per position, because a late-far-L story predicts a per-POSITION effect while "more averaging helps"
predicts a uniform one):

**CUE-aligned, overall accuracy:**

| animal | 2.0 s | 2.5 s | 3.0 s | 3.5 s |
|---|---|---|---|---|
| PS92 | 0.697 | **0.724** | 0.718 | 0.700 |
| PS93 | 0.673 | **0.685** | 0.681 | 0.673 |
| PS94 | **0.839** | 0.826 | 0.814 | 0.796 |
| PS95 | **0.827** | 0.812 | 0.787 | 0.761 |

**DO NOT extend the cue-aligned window.** PS92/PS93 peak at 2.5 s, PS94/PS95 decline monotonically
(−0.043, −0.066 by 3.5 s); averaged across animals 2.5 s beats 2.0 s by +0.003, a wash that changes
sign per animal. And the motivating position does NOT improve: PS93 far_L goes 0.436 → 0.454 → 0.431 →
0.429. `cue_post_s` stays 2.0.

**RT by position (engaged trials, curated sessions) — the lateness is real:**

| animal | close_L | close_ctr | close_R | far_L | far_ctr | far_R |
|---|---|---|---|---|---|---|
| PS92 | 0.128 | 0.160 | 0.160 | 0.224 | 0.256 | 0.288 |
| PS93 | 0.160 | 0.192 | 0.160 | **0.384** | 0.352 | 0.224 |
| PS94 | 0.128 | 0.128 | 0.128 | 0.160 | 0.192 | 0.192 |
| PS95 | 0.128 | 0.128 | 0.128 | 0.160 | 0.160 | 0.224 |

PS93 far_L has the longest RT of any animal×position — but 0.384 s sits comfortably inside a 2 s
window, which is why lengthening it changed nothing. **The window was never the binding constraint.**

**LICK-ALIGNED IS BETTER FOR EVERY ANIMAL**, and it is the correct instrument for the problem:

| animal | cue-aligned 2.0 s | lick-aligned (best) | gain |
|---|---|---|---|
| PS92 | 0.697 | **0.772** (2.0 s) | +0.075 |
| PS93 | 0.673 | **0.744** (1.5 s) | +0.071 |
| PS94 | 0.839 | 0.855 (1.5 s) | +0.016 |
| PS95 | 0.827 | **0.859** (1.0 s) | +0.032 |

The two animals gaining most are the two with the longest and most variable RTs. **PS93 far_L: 0.436 →
0.556 (+0.120).** The limiting factor is RT VARIABILITY, not RT magnitude: cue-aligned, a jittered
response smears across the averaging window and a longer window adds noise without fixing the smear;
lick-aligned, the response is locked and stays sharp. Optimal windows are correspondingly SHORTER when
lick-aligned (1.0–1.5 s vs 2.0 s), since the window no longer needs slack for RT spread.

**Consequence for PS93's asymmetry.** far_L remains its worst position lick-aligned (0.556 vs 0.891 for
close_L), so a real asymmetry survives — but roughly half the apparent deficit at 0.436 was
cue-alignment jitter rather than biology. Relevant to the stroke arm, where PS93's pre-stroke asymmetry
is a reference point.

**PRE-LICK instead of pre-cue: NO. In addition: yes, with an RT control.** With RT at 0.128–0.384 s a
2 s pre-lick window is ~90% the same data as the pre-cue window, shifted by RT — but the shift DIFFERS
BY POSITION (0.256 s spread), so a pre-lick window admits a position-dependent amount of the large,
position-specific post-cue response. A decoder can exploit "how much post-cue response leaked in",
which is a proxy for RT, which correlates with position: **pre-lick position decoding would partly be
decoding reaction time.** The cue is experimenter-controlled and fixed; the lick is behaviour-determined
and position-dependent, which is the property you do not want in a reference event. As an ADDITIONAL
analysis it is the standard framing for movement preparation (Kaufman/Churchland output-null) and worth
having — but it needs RT matched or regressed out, and PS93 is exactly where an uncontrolled version
would look most striking, having both the longest RT and the largest across-position spread.

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

### ~~No-lick trials carry no position code~~ — SUPERSEDED 2026-08-17

The original claim: pooled across curated sessions, PS95 n=396 acc 0.179 CI [0.142, 0.217]; PS94
n=351 acc 0.205 CI [0.163, 0.247]; PS92 n=189 acc 0.132 CI [0.084, 0.181] — all CIs including chance
0.167, against 0.71–0.86 engaged. Conclusion drawn: a maintained position code **gated by
engagement**, absent on trials the animal will not act on.

**That conclusion does not survive, and the reason is instructive.**

1. **The null was wrong.** Accuracy was compared against a uniform 1/6. These trials are heavily
   skewed across positions (PS93's are 49% `far_center`, 25% `far_L` — animals decline the far
   spouts) and the decoder's predictions on them are skewed too (PS94 puts 33% on one position). Two
   overlapping biases score above 1/6 with no information at all: PS93's independence null is
   **0.211**, and a constant "always guess far_center" scores **0.490**, beating the decoder's own
   0.293 outright. The same flawed comparison later flipped to reporting "above chance" for all four
   animals — the flag was uninformative in both directions.
2. **The sample has roughly doubled** (PS95 396 → 816 no-detected-lick trials), so the wide CIs that
   contained chance no longer do.
3. **The category was a mixture.** "No lick" pooled *licked late* (2–5 s) with *never detected*.
   Split, on PS93 8/12 the pre-cue survival is carried entirely by LATE trials (balanced 0.532,
   p=0.003) while genuinely undetected trials show nothing (0.153, p=0.76).

**Corrected finding** (`nolick_analysis` / `nolick_decoder`, headline = balanced accuracy, whose null
expectation is exactly 1/6 however skewed either side is; raw accuracy against a permutation null
computed on these trials with predictions held fixed; position-matched subsample as a check; run in
BOTH poolable bases). PS93 pooled over 11 sessions, ROI / joint agreeing: engaged 0.508 / 0.525
balanced, post-cue survival ratio **0.357 / 0.422**; pre-cue survives far better than post-cue.

So the readout is **not** "engagement gates the code". It is that the **POST-cue** code is largely
movement-driven and collapses without a lick, while the **PRE-cue** code substantially survives —
which is exactly the discrimination the post-stroke arm needs, and biologically expected (Priya,
2026-08-17): an animal can know where the spout is and still not lick.

**"No detected lick" is not "no attempt."** The sensor requires contact, so an executed but short
lick registers as nothing. PS93 has a pre-existing rightward tongue bias and reaches `far_L` poorly
(Priya, 2026-08-17) — making PS93 `far_L` a pre-stroke, within-subject instance of the post-stroke
phenotype, with ground truth owing nothing to the stroke. Recorded in
`nolick_decoder.ATTEMPT_CONFOUNDED` as the DLC/facial-tracking target list.

The ENCODER half of the original entry stands and is untouched: EV on these trials is ≈0 once the
baseline offset is removed (raw −0.28 to −1.04 is mostly a mean shift, not an inverted mapping;
re-centred −0.06, −0.03, −0.02).

**Method lesson.** Both the original claim and its later inversion came from one unexamined
assumption — that chance is 1/6 because there are six positions. It is 1/6 only when the trials are
balanced *and* the predictions unbiased. Neither holds on an arm defined by the animal declining to
respond, and an arm defined that way is exactly where the post-stroke question lives.

---

## Trials with NO DETECTED LICK — the post-stroke readout (decided 2026-08-17)

Post-stroke a failed trial can mean the plan was never formed or that it was formed and the movement
failed. Identical in the behaviour log; opposite predictions in imaging — plan-intact keeps the
PRE-cue code while the POST-cue code collapses, because post-cue decoding is largely lick-driven.
`nolick_analysis` / `nolick_decoder`; reference frozen to `nolick_reference_prestroke.json`.

**Chance is NOT 1/6 on this arm.** These trials are skewed across positions (PS93: 49% far_center)
AND the decoder's predictions on them are skewed, so an information-free decoder scores 0.211 on
PS93 and a constant "always guess far_center" scores 0.490 — above the 0.293 measured. Headline is
BALANCED accuracy (null expectation exactly 1/6 however skewed either side is); raw accuracy is
judged against a permutation null computed on these trials with predictions held fixed; a
position-matched subsample is the independent check. Test is ONE-SIDED (directional hypothesis); the
two-sided interval is reported but labelled descriptive, because at the margin they disagree.

**NOTHING IS ANALYSED AFTER THE RESPONSE WINDOW (Priya, 2026-08-17).** The spout begins MOVING when
the window closes, so any window extending past it samples the next trial's setup. The response
window — read PER SESSION from `gui_config.json` (3500 ms; NOT the decoder's 2.0 s `max_rt_s`) — is a
hard ceiling on every category boundary and every feature window. Categories: `engaged` (lick within
the cut), `late_rewarded` (after the cut, within the window — a HIT the decoder's 2 s convention
discards), `undetected` (no detected lick within the window, INCLUDING later licks, which is what the
task scores them as).

**Two splits, both of which changed the conclusion.** `late_rewarded` vs `undetected`: on PS93 8/12
the pre-cue survival is carried ENTIRELY by late trials (0.532, p=0.003) while undetected show
nothing (0.153, p=0.76). `undetected_working` vs `undetected_disengaged` (reusing `flag_engagement`): a
run of misses at the END of a session is satiation, not motor failure (Priya) — PS95 8/14 has 39 of
57 undetected trials in a terminal run, PS93 8/12 has 36 of 39 inside working stretches.

**Result.** Direction is robust: pre-cue survival exceeds post-cue in 16/16 animal × basis × cut
comparisons (1.1–3.5×). Significance is not: at the 2.0 s cut PS93 and PS95 in both bases, PS94 in
joint only (p=0.037); at the response-window cut PS95 alone. Mechanism is legible — the wider cut
reclassifies the late-but-successful trials as engaged, removing the trials carrying the signal.

**"No detected lick" is not "no attempt."** The sensor needs contact. PS93's rightward tongue bias
makes far_L a PRE-stroke, within-subject instance of the post-stroke phenotype, with ground truth
owing nothing to the stroke. `ATTEMPT_CONFOUNDED` is the DLC/facial-tracking target list.

**PRE- vs POST-STROKE TRIAL CRITERIA.** Pre-stroke uses lick-restricted trials; post-stroke will need
ALL trials, because a missing detection may be a protrusion that fell short (Priya). Both directions
are traps and `assert_comparable` RAISES rather than warns: engaged-only-pre vs all-trials-post
scores lower with NO stroke required, and post-stroke "engaged" is SURVIVORSHIP-selected toward
preserved function. Compute every criterion; compare like with like.
`reference_position_engagement` measures motivation only at positions the deficit is expected to
spare (Priya's close_L/close_center), parameterised because which are spared is a phenotype question.

## Per-session LocaNMF component COUNT is not a stable quantity (noted 2026-08-18)

Fitting the first post-stroke night, PS92 8/17 came out at 143 components with regions 19/25/26
absorbing 19/20/16 of them while every other region had 1-4 -- which looks alarming until it is
compared with that animal's own history. Every PS92 session looks like this: 121-196 components,
the same three regions taking 16-20 each, including 8/14 at 143 with exactly regions 19/25/26. Its
post-stroke session is indistinguishable from its baseline.

The variable animal is PS93, and it varies WITHIN the pre-stroke period: clean on 6/5, 6/6, 8/6,
8/10, 8/13, 8/14 (80-91 components, max 2-3 per region) and heavily split on 6/7, 6/8, 8/5, 8/7,
8/9, 8/11, 8/12 (98-166, max 8-20). Its 8/17 falls in the clean group.

**Do not read a change in component count or per-region concentration as a lesion effect.** It swings
by a factor of two within an animal across pre-stroke days, so the post-stroke value carries no
information on its own. It also explains fit runtimes differing 7x between sessions (PS92 425 s vs
PS93 62 s) -- that is the splitting, not a problem.

The post-stroke comparisons are immune by construction: the frozen decoder uses Allen-ROI features
and the joint analyses use SHARED footprints with new days PROJECTED, so neither depends on how many
components a session's own fit produced. That immunity is the reason the cross-day work was put in
those two bases in the first place, and this is the first observation that tests it.

Worth a proper QC metric later (per-region concentration over sessions) if anyone wants to know what
drives the swing; it is not needed for the stroke comparison.

## PLANNED vs EXECUTED direction — pending DLC/FR (Priya, 2026-08-17)

Priya: once DLC/facial tracking lands, pre-cue trials could be binned by the direction the TONGUE
actually went rather than by spout position — on the first trial after a position change the animal
sometimes licks toward the OLD position and corrects on the next lick. Her question was whether it
is fair to do this PRE-stroke, given the post-stroke arm is specifically looking for a
planned/executed disconnect.

**Yes, and it is closer to necessary than optional.** Without a pre-stroke rate of plan/execution
mismatch, a post-stroke dissociation cannot be attributed to the lesion — it could be what the
analysis does to any data. The pre-stroke error trials are that control.

**BUT: two labels, never a relabelling.** Binning pre-cue by executed direction makes "does the plan
match execution" tautological — the label IS the execution. Keep both per trial:

    target label     spout position (what the animal should do)
    executed label   DLC tongue direction (what it did)

They agree on most trials; the MIS-DIRECTED trials are the whole measurement. Train on target labels
over all trials, then score the held-out error trials BOTH ways. No refit, and the contrast is the
result: predicts target -> the pre-cue code is about the stimulus; predicts execution -> it is a plan.

**This also breaks a tie the design was thought unable to break.** See '"Pre-cue" is AFTER the spout
arrives' above: the spout has been in position ~3 s by cue time, so a held intention and a sustained
somatosensory response are temporally coextensive and this design cannot separate them. On a
mis-directed trial it can — the spout (somatosensory input) says A while the tongue goes to B, so
pre-cue activity predicting B is a plan by construction. The 44-session asymmetry result (2026-08-17:
information highest FURTHEST from the cue, decaying through the ENL) only leans somatosensory; this
would settle it.

**Feasibility.** First-trial-after-position-change is 16.1-16.3% of trials — 651 (PS92), 967 (PS93),
997 (PS94), 1120 (PS95) across the curated set. What fraction are actually mis-directed is unknown
until DLC exists, but even 10% leaves 65-110 per animal: adequate POOLED, thin per session.

**Build in from the start:**
* Error trials are SELECTED, not random — they follow position changes and will cluster on far
  positions and on PS93's biased direction. Compare within condition (erred vs not-erred among
  first-after-change trials), never against the general trial pool.
* State the rule for correct-on-second-lick trials. Recommend labelling by the FIRST lick: that is
  the executed plan, and the correction is a separate event.
* Record the labelling scheme per result and extend `nolick_analysis.assert_comparable` to cover it.
  A target-labelled result compared with an execution-labelled one is the same trap as the
  pre/post trial-criterion mismatch, and would manufacture exactly the effect being looked for.

## Lick BOUT ONSETS as the motor event set (decided 2026-08-17)

The lick-aligned decoder uses ONE lick per trial and discards 80–93% of lick events. `lick_bout_events`
/ `lick_bout_decoder` use bout ONSETS instead — 1.16–4.24 per trial. Bouts, not individual licks
(Priya): licks within a bout are 5–7 Hz and correlated, so counting each would multiply n 5–15× while
adding almost nothing and making CV anticonservative.

* **Labelled by the SPOUT STROBE, not the preceding cue.** The spout moves and strobes BEFORE the
  cue, so the cue rule gives arrival-window licks the PREVIOUS trial's position. ~6-trial blocks hide
  it; measured, the genuinely mislabelled fraction is PS93 7.3%, PS94 5.0%, PS95 0.8%, concentrated
  at block transitions.
* **CV groups are TRIALS.** Bouts share a trial; grouping by bout leaks and inflates accuracy in
  proportion to the apparent gain — where it is least visible and most flattering.
* **Phases scored separately** (`approach` pre-cue vs `response`): PS92 is 74% approach, PS95 6%, so
  a pooled number is a different quantity per animal.

**Read the approach result correctly.** PS92 8/12 approach bouts decode at 0.761 balanced (n=704,
p=0.003) — but the animal is LICKING THE SPOUT, so position is available somatosensorily. That is a
motor/sensory readout, NOT a pre-cue plan. Its value is a 4× larger movement-locked reference.

---

# Module & output reference
Key analysis modules (`wfield_local/`): `run_locanmf.py` (LocaNMF/sNMF on the GPU box) ·
`locanmf_position_decoder.py` (**the decoder**; `--source locanmf|roi`, `--align cue|lick|precue`, per-area
accuracy + per-position recall + confusion) · `locanmf_position_encoder.py` (per-position EV / FEVE /
predicted maps) · `locanmf_cross_mouse.py` (cross-mouse + within-animal consistency) · `locanmf_rsa.py` (RSA
+ noise ceiling + hemisphere-resolved + crossnobis) · `locanmf_decoder_weights.py` (rolling/temporal figs) ·
`nolick_analysis.py` + `nolick_decoder.py` + `plot_nolick_reference.py` (the no-detected-lick arm and its frozen pre-stroke reference) · `lick_bout_events.py` + `lick_bout_decoder.py` (bout-onset motor event set) · `roi_activity.py` (CPU Allen-area ROI traces) · `quiet_periods.py` (quiet-frame mask) · `atlas_overlay.py`
(shared region outlines) · `framemap_event_maps.py` (regime-B cue/lick maps) · `qc_motion_correction.py` ·
`cross_day_align.py`. The nightly orchestrators are `preprocess.py` (imaging) and `nightly_figs.py`
(analysis). Figures/tables on MICROSCOPE under `labcams/locanmf_lick_pooled/…/cue_analysis/`; per-session
LocaNMF outputs in each session's `motion_corrected/locanmf_affine8v1_final/`. The full historical module list
(early lick/cue exploratory modules) is in `docs/archive/ANALYSIS_HISTORY.md`.

## POST-STROKE ENGAGEMENT FILTERING IS RETIRED (Priya, 2026-08-18)

**There is no valid post-stroke construction of "disengaged", so no post-stroke analysis may split on
one.** `poststroke_compare.POSTSTROKE_ENGAGEMENT_FILTERING = False`; guarded by
`tests/test_deck_section_g.py`.

### Two gates, both wrong, for different reasons

**The pre-stroke gate (`flag_engagement`) was wrong for the obvious reason.** It calls an animal
disengaged when its trailing response rate collapses — and after a lesion that rate collapses BECAUSE
the animal cannot reach the far positions. It labels the effect being measured as the confound. On
PS94 8/17 it called **59%** of trials disengaged, against **6.8%** for a spared-position gate. Any
"no plan was formed" conclusion drawn through it is circular.

**The spared-position gate that replaced it is better and still not valid.** It judges engagement only
at positions the animal can still reach (close_L, close_center — far_L and close_R deliberately
excluded in case they are affected). That removes the circularity but not the ambiguity: it marked 29
PS94 trials "disengaged" because they were no-lick trials falling where the trailing response rate at
the reference positions dipped below 0.5 over 15 reference trials. **A short run of MOTOR failures
produces that dip exactly as readily as a motivational lapse.** Nothing in the spout data
distinguishes them. Priya raised this directly: those 29 may simply be one-off misses.

**And it has no general form.** In a severe stroke *every* spout position may be impaired, leaving no
spared reference to anchor engagement on — so spout contact may not be usable as an engagement readout
at all. A gate that only works for mild lesions is not a gate.

### What this invalidates

`undetected_state_split`'s comparison class was never established, so its output is
**UNINTERPRETABLE, not negative**:

| animal | n working | n disengaged | working − disengaged | what I first said | what it means |
|---|---|---|---|---|---|
| PS94 | 116 | 29 | −0.060 | "no separation → global post-stroke shift" | nothing; the second class is unvalidated |
| PS95 | 44 | 0 | — | "UNDECIDABLE" | right, for the wrong reason |

I reported the PS94 number as evidence for a global shift. It is not evidence for anything. It must
not be quoted, and it is not shown in the deck.

The rolling response rate at spared positions survives as a **DESCRIPTIVE statistic** — PS94 0.89,
PS95 0.97 — which is a sound thing to report and an unsound thing to split trials on.
`poststroke_engagement` is kept for that purpose with the restriction in its docstring.

### What replaces it: `impaired_nolick_readout`

Ask the same question WITHIN the post-stroke session, splitting no-lick trials on the **true spout
position** — impaired vs preserved — which is *measured* rather than inferred and needs no engagement
label. Apply the frozen pre-stroke decoder and ask whether it still reads out position:

- above that arm's own permutation null at IMPAIRED positions → the position was represented and the
  movement did not happen: **execution failure, plan intact**;
- at null everywhere → no readable plan, and execution failure is not supported.

**Measured on PS94 8/17 (ROI, 6-way, per-arm permutation nulls):**

| alignment | arm | n | balanced | null | p |
|---|---|---|---|---|---|
| pre-cue | preserved | 145 | 0.214 | 0.178 | 0.167 |
| pre-cue | impaired | 209 | 0.167 | 0.151 | 0.206 |
| post-cue | preserved | 145 | 0.044 | 0.090 | 0.958 |
| post-cue | impaired | 209 | 0.372 | 0.333 | 0.068 |

**Nothing reaches significance.** The execution-failure hypothesis is NOT supported by this test for
PS94. Note how high the nulls run (0.333 for the post-cue impaired arm): these trials are heavily
skewed across positions, so a uniform 1/6 chance line would have manufactured a result here — which is
why every arm carries its own permutation null.

A null result here is weaker evidence than a positive one would have been, and it stays ambiguous
between "no plan" and "plan formed, plus a tongue protrusion the spout never registered".

### The limit that no amount of statistics fixes

**"No lick detected" is not "no tongue protrusion."** The spout requires contact, so a short or
misdirected lick registers as nothing. PS93 already shows this PRE-stroke at far_L (a pre-existing
rightward bias), which makes it a within-subject instance of the phenotype being looked for. DLC /
facial tracking replaces this inference with a measurement, and until it lands every no-lick
conclusion in section G carries the ambiguity. See "PLANNED vs EXECUTED direction — pending DLC/FR".

### Consequence for trial selection

Post-stroke analyses use **ALL trials** (`nolick_analysis.SANCTIONED_MISMATCHES` declares the
pre-engaged vs post-all pair by name). Post-lick-bout analyses are the sole exception, since they need
a lick to align to.

## Hemispheric raw fluorescence: the 470 and 415 questions are ONE measurement (2026-08-18)

Priya's observation: *"post-stroke there is higher GCaMP 470 nm signal in the L hemisphere, especially
parietally... as well as changes in 415 nm hemodynamic signal (is there evidence of L hemisphere
hypoperfusion after striatal stroke?)"*

**These cannot be asked separately**, because 415 nm is the isosbestic channel and therefore the
control for 470: only the ratio of ratios is GCaMP-specific.

> ### ⚠ RETRACTED 2026-08-18, same day: the absorption argument was wrong
>
> This section originally argued that haemoglobin absorbs, so more blood means less light, so
> hypoperfusion RAISES the 415 counts. Priya challenged it — *"I think typically increased blood means
> increased 415 and 470 signal, a-la neurovascular coupling"* — and the data agree with her.
> Cue-triggered averages of the RAW violet trace (`scratchpad/hemo_sign_check.py`), whole brain:
>
> | session | 470 evoked (positive control) | 415 evoked |
> |---|---|---|
> | PS94_0814 | +3.69% | **+1.96%** |
> | PS95_0814 | +3.54% | **+1.00%** |
> | PS93_0812 | +2.66% | **+0.93%** |
> | PS92_0813 | +2.31% | **+0.54%** |
>
> The 415 signal **rises** with activation in every animal, tracking the blue at about a third of its
> amplitude. A simple absorption account predicts a dip. Why it rises is open: 415 nm is not exactly
> GCaMP's isosbestic (~410 nm) so calcium can leak in; near the Soret band HbO and HbR absorb very
> differently, so an HbO rise with an HbR fall need not raise total absorption; and flavoprotein
> autofluorescence sits in this range.
>
> **The sign cannot simply be flipped either.** That test characterises the DYNAMIC, task-locked
> regime; the L/R ratio is a STATIC baseline difference over months, and the two need not agree. So the
> perfusion DIRECTION of a 415 change is **unresolved**. The null result below stands — what is
> withdrawn is the claim that a change would have meant hypoperfusion. Settling it needs an independent
> perfusion measure (laser speckle, or a manipulation of known direction).
>
> Two getting-it-wrong notes worth keeping: the first version of the check reported +692% and −2881%
> evoked responses, because `U @ SVT` reconstructs the DEVIATION from each channel's mean (the
> reconstructed means are 0) and I divided by them. And it assumed the channel order from a docstring
> rather than deriving it — `SVTcorr` is the corrected BLUE channel, so whichever half of `SVT` it
> correlates with IS blue, which is now how it is determined.

Only the ratio of ratios separates them (`wfield_local/hemispheric_intensity.py`):

| quantity | meaning |
|---|---|
| `R_415 = median(415, L) / median(415, R)` | optical asymmetry, ~calcium-free. Perfusion direction **unresolved** — see the retraction above |
| `R_470 = median(470, L) / median(470, R)` | optical **+** neural. Not interpretable alone |
| `G = R_470 / R_415` | GCaMP-specific, absorption divided out |

**Why ratios, not absolute counts.** `crossday_intensity` already tracks the absolute brain-ROI median
and warns on its own figure that LED power is titrated by hand day to day. That confound is fatal for a
cross-day claim about one hemisphere, but it is common to both hemispheres *within* a session and
cancels in an L/R ratio — as do exposure, gain and bleaching. What does not cancel is anything
spatially asymmetric (window clarity, focus tilt, uneven illumination, headplate shift), so the
question is never "is L/R ≠ 1" — it never is — but "did L/R **move** from this animal's own pre-stroke
range".

### Result on 8/17: no detected change

Every measure sits inside the animal's own pre-stroke range, whole-hemisphere and in SSp:

| animal | region | R_415 | R_470 | G (GCaMP-specific) |
|---|---|---|---|---|
| PS94 | all | z = +0.7 | z = +0.3 | z = −0.9 |
| PS95 | all | z = +0.3 | z = −0.3 | z = −0.9 |
| PS94 | SSp | z = +0.2 | z = +0.3 | z = −0.3 |
| PS95 | SSp | z = +0.4 | z = +0.8 | z = −0.3 |

No evidence for a left-sided 470 increase, and no detectable change in the optical asymmetry, one day after the lesion. (Whether an asymmetry change would have indicated hypo- or hyper-perfusion is unresolved.)

### Read that null with its power — two real limits

**The 415 L/R ratio DRIFTS monotonically across the whole pre-stroke period in all four animals**
(PS94 SSp 0.66 → 0.82; PS92 0.52 → 0.75; PS95 0.53 → 0.73). The min-max pre-stroke band therefore spans
a *trend*, not noise, and a step change would have to be large to escape it. A sharper test compares
against the extrapolated trend, or against the last few sessions only. The drift is present in
PS92/PS93 too, who had no effective lesion until after 8/17, so it is **not lesion-related** — it is a
property of the preparation or the rig, and it is unexplained. Worth explaining on its own account.

**And this measures the wrong thing for "activity".** It is the session MEAN image: static baseline
fluorescence. An impression of more activity is about DYNAMICS, which a mean image cannot show. The
matching test is a per-hemisphere temporal SD or task-evoked amplitude, which is not run here.

n = 1 post-stroke session, one day post-lesion; perfusion changes evolve.

## Crossed confusion must include the trials with NO detected lick (Priya, 2026-08-18)

*"even though far R lick was never successful, pre-cue looked like far R, or far center looks like
far R (because tongue is deviated leftward, animal has to try harder to get tongue to the right)"*

The engaged-only matrix cannot answer this. PS94 has **zero** engaged trials at far_center and far_R —
the two positions the lesion abolished, and the two rows worth reading — so both were blank. But the
animal was still *cued* to them (104 and 105 no-lick trials), and those trials are the only evidence
that exists there. `crossed_confusion(post_all_trials=True)` fills the rows; the PRE arm stays
engaged-only, because it is the reference for what the code looks like when the movement succeeds.

**The column baseline is mandatory.** The frozen decoder predicts far_R on ~35% of ALL PS94
post-stroke trials, so far_R "recall" is inflated by prediction bias before any position information
is involved. Under a label permutation the expected recall for a position is exactly its prediction
rate — which is what the figure prints under each column, and what a diagonal must clear. Reading these
diagonals against 1/6 would manufacture a result, the same error the `impaired_nolick_readout` nulls
were built to prevent.


## ⚠ PS92/PS93 ARE NOT A NEGATIVE CONTROL — they were lesioned too (Priya, 2026-08-18)

> *"Note that PS92/93 are NOT a 'control' - they did have strokes, they were just too small to cause a
> big behavioral deficit."*

I called their 8/17 session a NEGATIVE CONTROL, in the deck (G7/G7b/G7c), in `M_POSTSTROKE`,
`M_HEMIDYN`, `hemispheric_dynamics`, and in two commit messages — and I leaned on it, arguing that
because they show no decoding change the PS94/PS95 effects must be the lesion rather than the day. The
premise is false. The 8/16 laser **did** produce strokes in them; they were simply small enough to
leave no overt behavioural deficit.

**What survives.** They still control for the **day and the procedure**: same recording day, rig,
anaesthesia, handling, preprocessing and frozen decoder. An artefact of 8/17 would have hit all four
animals, and it did not. That half of the argument is intact, and it is the half that matters for
"is the far_R over-prediction a property of the decoder or of the session".

**What does not.** They cannot show that a lesion is **necessary** for an effect. A null in an animal
with a small stroke is equally consistent with *small stroke, small effect* — so lesion-vs-no-lesion is
not separable here, and no claim of that form may rest on them.

**What they are instead**, and this is arguably more useful than a true control: the **small-lesion arm
of a severity contrast**. Two animals with large strokes and overt deficits, two with small strokes and
none, recorded the same day on the same rig. That is a dose axis, and the right way to read Section G's
comparisons.

Renamed SMALL-LESION COMPARISON throughout rather than merely re-worded, so a reader skimming a slide
title cannot pick up the wrong reading.


## Adjacent same-position blocks MERGE in the pipeline's block IDs (audited 2026-08-18)

Priya: *"I want to be sure we are appropriately labeling two blocks when the GUI randomly put two
blocks of the same position next to each other."*

She is right that it happens and right that the pipeline mislabels it. `_trial_features` starts a new
block whenever the POSITION changes, so two consecutive blocks at the same position become one.

### The firmware records the ground truth, and I first said it didn't

`device_snapshot_end.json` carries **`block_number`** — the firmware's own count of blocks scheduled
(`block_pos`, `block_trial`, `current_block_size` are there too). I initially reported no block ID
anywhere in the logs, having checked only `events.csv` event names and `config_changes.csv` keys, and
missed `session_manifest.json` and the device snapshots.

It is a **total, not per-trial**: the GUI polls device status every 1 s
(`auto_status_poll_interval_s: 1.0`) but `logging.timeseries_enabled: False`, so the polls were never
written. Turning that flag on would give per-trial block IDs for all future sessions and is the
cheapest permanent fix.

`firmware_blocks − observed_runs` is therefore the exact merge count per session, with no inference.

### Audited over all 48 curated + 8/17 sessions

**118 merges / 4216 firmware blocks = 2.8%**, range 0–8.2% per session. A run longer than
`block_size_max` (8) cannot be one block, and that detector catches ~92% (108/118). The residual ~10
are 4+4 merges at run-length exactly 8 and are not detectable from run length.

**Positions must come from the DAQ, not `trials.csv`.** The GUI's `pos_idx` is mislabelled on
position-change trials (docs/GUI_TRIALS_LOGGING.md) — exactly the trials that define a boundary. Using
it gave PS94 8/17 an impossible count: 107 runs with 9 runs longer than 8, i.e. ≥116 blocks, against a
firmware count of 110.

**Two sessions have MORE runs than firmware blocks**, which cannot happen: PS93_0806 (−1) and
PS92_0812 (−4). Both are already-flagged: dead `spout_bit1` behaviour-log fallback, and the
crash-and-concat session. The check is worth keeping as a position-labelling guard regardless of the
merge question.

### Impact: the error runs in the SAFE direction

Merging makes GroupKFold groups larger, so more correlated data is held out together — the current CV
is *more* conservative, not inflated. Measured on the eight worst-affected sessions, both alignments:

| | delta (split − merged) |
|---|---|
| mean | **+0.0105** |
| max abs | 0.0528 (PS93_0817 pre-cue) |
| sign | 10/16 positive |

Splitting RAISES accuracy on average, confirming the direction. **Caveat I cannot yet remove:** the
±0.05 scatter is probably dominated by fold-reassignment noise rather than the merge itself — changing
group membership reshuffles folds and moves accuracy regardless. Separating the two needs a shuffled-
group control at matched group count, which is not run. So "+0.011 mean" should be read as *no
evidence of inflation*, not as a measured merge effect.

### Decision

Not urgent, because nothing is inflated. When done: split over-long runs at `block_size_max`, bump
`CACHE_VERSION`, and assert per session that the block count matches `device_snapshot_end.json` — that
last part turns the firmware count into a permanent guard. Leaves unfixed, and to be documented rather
than hidden: the ~10 hidden 4+4 merges, and the placement of a split inside an over-long run (11 could
be 4+7, 5+6, 6+5 or 7+4).

## ⚠ THE POST-STROKE TRIAL SET AND POSITION SET — four corrections, 2026-08-19

Four errors, all the same shape: **a quantity treated as a fixed property of the animal was actually a
property of the trial set**, and comparisons were made across changes in it. Each silently changed what
a number meant while the number kept looking reasonable. All were caught by Priya reading the results,
not by a check.

### 1. Four of six post-stroke analyses filtered to ENGAGED trials

`decode_matched`, `recoding_test`, `pattern_similarity`, `spatial_reorganisation` and
`evoked_amplitude` used the engaged arm only — while `M_POSTSTROKE` had been asserting the all-trials
rule since it was written. Only `crossed_confusion` and `fixed_scale_maps` complied. The deck mixed
both conventions without saying so, which is why the fixed-scale maps showed PS94's far positions
nearly silent while `evoked_amplitude` reported them elevated: different trial sets, never reconciled.

Not a technicality: **PS94 8/18 is 40% engaged**, so those analyses read a minority subset selected by
the behaviour the lesion disrupted.

**Now:** every comparison exposes `post_all_trials`, defaulting to True, and BOTH arms are reported
side by side — the difference separates "the code degraded" from "the code is fine when the animal
manages to lick". The no-lick readouts (`looks_like_which`, `fits_engaged_distribution`,
`impaired_nolick_readout`) are exempt because reading that arm is their purpose.

### 2. `preserved_positions` pooled across sessions — i.e. took the UNION

"Positions the animal still attempts" is a **per-session** behavioural state. PS95 attempted
far_center/far_R on 8/18 (99 and 84 trials) but not on 8/17 (10 and 1), so the pooled set was six
positions and **PS95's 8/17 numbers ran over a position with a single engaged trial**. Registering 8/18
also moved that result's chance level from 0.25 to 0.167 with nothing about 8/17 having changed.

**Now:** per-session, with pooled comparisons defaulting to the INTERSECTION — positions attempted on
every post day, because a pooled statistic must be defensible for each session inside it.

### 3. The ALL-trials arm was restricted to lick-defined positions

The arm built to examine failed movements inherited a position set defined by *where licking survived*.
PS94's far_center and far_R have **zero engaged and ~105 no-lick trials each** — a hundred trials per
position, absent from every decoding number reported for that animal.

Consequently **every "intact" result was scoped to positions the animal could still reach.** "PS95 was
never degraded" meant *not degraded where it still licked*; its affected positions were unexamined,
not normal.

**Now:** the ALL arm scores all six positions (chance 1/6, fixed across sessions and animals, so it is
the only arm comparable across sessions); the LICK-ONLY arm keeps the restriction, where it is forced.

### 4. Block IDs merged adjacent same-position blocks

See the separate entry. 118 of 4216 blocks (2.8%); CACHE_VERSION 9; error ran in the conservative
direction.

### What these corrections did to the results

| claim | status |
|---|---|
| "PS94's information is INTACT, only the code changed" | **WITHDRAWN** — an engaged-only artefact |
| "PS95 was degraded on day 1 and recovered" | **WITHDRAWN** — a union-basis artefact (0.877 inside on its own 4-position basis vs 0.719 outside on the pooled 6-position one) |
| "amplitude rose 2–3× at every position" | **WITHDRAWN** — the summed measure conflates amplitude with spatial EXTENT; peak rises only at close_L/close_center and FALLS at the far positions |
| PS94 day-1 plan/execution dissociation | **STRENGTHENED** — see below |

### The result that survived and got stronger

**PS94, 8/17, all trials, all six positions:** pre-cue within-session **0.521** against a pre-stroke
band of 0.534 [0.443–0.618] (**z = −0.2, inside**); post-cue **0.633** against 0.866 [0.806–0.926]
(**z = −7.1, outside**).

On the **same trials**, including the ~210 trials at positions where the animal never licked, the
pre-cue window carries normal position information while the post-cue window is severely degraded.

This is a **within-session, within-trial contrast** — two windows on one trial set — so every
session-level confound (LED power, baseline F, amplitude, arousal, engagement, trial count) affects
both equally and cannot produce a difference between them. That rules out the entire class of
artefacts the other corrections were about.

By **8/18 the dissociation is gone**: pre-cue 0.337 (z = −3.4), post-cue 0.472 (z = −12.1). The plan
survives the lesion on day 1 and is gone by day 2, while execution-phase coding is impaired from day 1.

Caveats: one animal; one session per day; and "plan" means pre-cue position information, not a
demonstrated motor intention — the spout arrives ~3 s before the cue and this design cannot separate a
held intention from a sustained sensory response (see the terminology entry above).


## THE PLAN/EXECUTION DISSOCIATION — REPLICATED IN ALL FOUR ANIMALS (2026-08-19)

On the first session after an **effective** lesion, within-session decoding places the **pre-cue**
window inside that animal's pre-stroke band and the **post-cue** window outside it. All-trials arm,
six positions, chance 1/6:

| animal | day 1 | laser | pre-cue z | post-cue z |
|---|---|---|---|---|
| PS94 | 8/17 | 3 mW | **−0.2** | **−7.1** |
| PS95 | 8/17 | 3 mW | **+1.6** | **−3.4** |
| PS92 | 8/18 | 3.75 mW | **+0.2** | **−2.4** |
| PS93 | 8/18 | 5.5 mW | **−0.5** | **−3.6** |

Four animals, two lesion days, three laser powers, no exception.

### Why this survives when session-level comparisons did not

Pre-cue and post-cue are **two windows on the same trials**. LED power, baseline F, evoked amplitude,
arousal, engagement and trial count act on both equally and cannot produce a difference between them.
It is a **within-trial** contrast, which is precisely the property that the four trial-set errors of
2026-08-19 destroyed in every session-level measure.

### PS92 and PS93 give a within-animal before/after control

Their **8/17** sessions follow the 8/16 laser that did not take, and show nothing outside the band at
any alignment (PS92 −0.1, +0.2, −0.3; PS93 −1.4, −0.0, −0.5). One day later, after the effective 8/17
lesion, the dissociation is present. Same animal, same rig, one day apart.

That control exists **only because the excluded sessions were kept analysable** rather than dropped —
the `post_labels` override added on 2026-08-18 for what was then framed as a negative control.

### Day 2 separates the animals

PS94 loses the plan as well (pre-cue z=−3.4, post-cue z=−12.1). PS95 returns fully inside the band
(+2.1, +1.3) alongside behavioural recovery of the far positions (far_center 10→99 trials, far_R 1→84).
Deterioration versus recovery, tracking behaviour in both cases.

### Laser power does not predict magnitude

PS94 at the **lowest** dose (3 mW) has the largest post-cue deficit (z=−7.1); PS93 at the **highest**
(5.5 mW) has −3.6. Behavioural severity tracks the effect; dose does not. PS93 was nominated in advance
as the dose test and the prediction failed.

### Caveats that stay attached

One session per animal per day. And **"pre-cue" means pre-cue position information, not a demonstrated
motor intention** — the spout arrives ~3 s before the cue, so a sustained sensory response and a held
plan are temporally coextensive and this design cannot separate them. The dissociation between two
WINDOWS is solid; calling the earlier one a plan is an interpretation.

## ⚠ A HARDCODED DATE LIST DELETED HALF THE POST-STROKE DATA (found 2026-08-19)

`evoked_amplitude.collect` selected sessions with

```python
keep_dates = set(config.curated_dates()) | {"0817"}
```

Correct the day it was written — 8/17 was the only post-stroke date. The moment the 8/18 sessions were
registered it became a silent data-deletion bug:

- **PS92 and PS93 produced no post-stroke row at all.** Their effective lesion is 8/18, and 8/17 is in
  their `exclude` list, so after filtering they had zero non-pre sessions. The summary printed
  `--- PS92 ---` with nothing under it and moved on.
- **PS94 and PS95 were truncated to day 1.** Their 8/18 sessions were dropped, so every amplitude
  number reported for them described 8/17 only while appearing to describe "post-stroke".

Nothing raised. A second instance of the same shape sat ten lines below: `share_z` read `post[0]`, the
first post-stroke session, so per-area redistribution was reported for day 1 alone even when day 2 was
present.

### The error class, third instance today

This is the same failure as `labs = excluded_labels(an) or None` in the grid runner and as the union
`preserved_positions`: **a set that is a property of the current data was written down as a literal and
then outlived the data.** Every instance failed silently in the direction of reporting less than it
claimed, and every one was caught by a person reading the output, not by a check.

### Fix

Curation now applies to the **pre-stroke reference only** — its actual purpose. It exists to keep
noisy early sessions (PS95_0605, mean |amplitude| 16.3 against ~0.53 elsewhere) out of the reference
BAND; applied to a post-stroke session it deletes the measurement instead. The selection is split out
as `evoked_amplitude.sessions_to_measure` so the invariant is testable without touching data:

- a session whose phase is not `pre` is never removed by curation;
- a non-curated PRE-stroke date still is.

Both directions are pinned in `tests/test_stroke_phase.py`, next to the tests guarding the opposite
leak (post-stroke data entering a pre-stroke pool). `share_z` is now keyed by post-stroke date.

### The guard that vouched for the wrong thing

Fixing this surfaced a weak test. `test_ps92_ps93_are_never_called_a_negative_control` checked
`low.index(phrase)` — the **first** occurrence only. A properly hedged code comment at line 1089
therefore vouched for every later mention in the module, and one had slipped through: the G9 slide told
the reader "G7 uses them as the negative control" in user-visible text, contradicting the 2026-08-18
correction. The guard now checks every occurrence, and the G9 bullet has been rewritten to say what
those sessions actually are: the within-animal before/after control.

## LATERALISATION COLLAPSE IS PS94's ALONE — and DIRECTION is what says so (2026-08-19)

The `evoked_amplitude` headline had been **"amplitude rises in all four animals, graded by
severity."** It was written from two animals. PS92 and PS93 contributed no post-stroke rows at all
(see the hardcoded-date entry above), so "all four" described PS94 and PS95. Measured properly, PS92
shows no rise anywhere.

### What replaces it

Restrict to positions that were **lateralised before the lesion** (|pre-stroke R−L| > 0.15) — a
position with no lateralisation to begin with cannot lose any, and including them manufactures
"changes" that are excursions around zero.

| animal | lateralised positions | toward zero | sign reversal | AWAY from zero |
|---|---|---|---|---|
| **PS94 8/17** | 6 | **4** | 1 (far_center) | 0 |
| **PS94 8/18** | 6 | **4** | 1 (far_center) | 0 |
| PS95 8/17 | 4 | 0 | 0 | 1 (far_R) |
| PS95 8/18 | 4 | 2 | 0 | 0 |
| PS93 8/18 | 3 | 0–1 | 1 (close_L) | 0–1 (far_R) |
| PS92 8/18 | 3–4 | 0 | 0 | 0 |

PS94's four toward-zero positions are `close_L, close_center, close_R, far_R` — **the same four on
both days and at both alignments**, with `far_center` reversing sign each time (−0.30 → +0.31).
Four independent readings, no disagreement. `close_R` goes −0.81 → −0.26 (8/17) → −0.20 (8/18).

### The methodological point, which is the recurring one

**Counting "positions outside the pre-stroke band" would score PS93 4/6 and PS94 5/6 and make them
look alike.** They are moving in opposite directions: PS93's far_R goes −0.19 → −0.47 lick-aligned,
*more* lateralised. Only the direction separates a collapse from an intensification.

This is the same direction-blindness that twice painted an above-band value as a deficit (PS95's
recoding verdict, then the grid figure's colouring). Third instance in three days, so the note and
the slide subtitle now lead with direction rather than with "outside".

### Two caveats kept attached

**The summed measure conflates amplitude with spatial extent.** `abs_total` sums |response| over 66
areas. G8d's common-scale maps separate them: PS94 peak amplitude rises only at close_L
(0.039 → 0.073) and close_center (0.029 → 0.055) and *falls* at the far positions, while the summed
measure rose at far_L (0.278 → 1.070) with its peak flat. The response became spatially **broader**,
not stronger.

**The far-position amplitude drop covaries with the animal's attempts.** far_R falls in all four
animals on day 1, and PS95's far_R is 0.086 with one lick trial on 8/17 and 0.643 with 84 on 8/18 —
tracking behaviour exactly. On the all-trials arm that is partly trial composition, not necessarily a
lesion effect on the response to a given movement.

## THE MIDLINE TEST IS A NULL; PS94's PRE-CUE GEOMETRY IS NOT (2026-08-19)

Two tests on the spatial maps (`spatial_reorganisation`), both following from one prediction: if
PS94's lateralisation collapses and its position information survives but is unreadable by the frozen
decoder, those should be one fact seen from two sides.

### Crossnobis convergence, both arms, position-matched bands

z against that animal's pre-stroke range, rebuilt over **each session's own positions**:

| session | post-cue ALL | post-cue LICK | pre-cue ALL | pre-cue LICK |
|---|---|---|---|---|
| PS92 8/18 | **−3.3** | −1.3 | −0.1 | −0.4 |
| PS93 8/18 | **−2.3** | −1.4 | −1.2 | −1.4 |
| PS94 8/17 | **−3.0** | −1.4 | **−3.7** | **−3.2** |
| PS94 8/18 | **−4.2** | **−3.4** | **−5.5** | **−5.4** |
| PS95 8/17 | −1.2 | +2.4 | +0.2 | −1.7 |
| PS95 8/18 | +1.6 | +2.7 | +0.5 | +0.4 |

Post-cue geometry converges in three of four on day 1. On the lick-only arm those weaken sharply, so
much of the post-cue convergence is carried by the no-lick trials — expected if the missing movement
is what changed, and a reason to read both arms rather than pick one.

### The result that qualifies the headline

**PS94's PRE-CUE geometry is degraded too**, and it survives the lick-only arm almost unchanged
(−3.7 → −3.2, −5.5 → −5.4). So it is *not* an artefact of folding heterogeneous no-lick trials into a
within-position covariance estimate — the obvious explanation, tested for exactly that reason. The
other three animals' pre-cue geometry stays inside the band.

This reconciles with PS94's pre-cue **decoding** sitting inside its band (z=−0.2) because crossnobis
measures distance in noise units while decoding asks whether a boundary can still be drawn. PS94's
pre-stroke pre-cue crossnobis is unusually large — **5.71** against 1.30 (PS92), 1.98 (PS93), 1.65
(PS95) — so falling to 2.45 leaves it roughly where the other three animals normally sit, comfortably
decodable.

**So for PS94 the dissociation is a matter of DEGREE**: both windows lose separability and only the
post-cue loss crosses the threshold where six-way decoding fails. It is not "pre-cue untouched". For
PS92, PS93 and PS95 the dissociation is clean on crossnobis as well.

### Two corrections to the criteria, both found by putting real numbers through them

**The bands were not position-matched.** `mean_distance` averages over the position PAIRS a session
has — 15 for six positions, 6 for four. On the lick-only arm PS94 keeps four positions and PS95 kept
five on 8/17, so scoring them against a six-position band compared a mean over one pair set with a
mean over another. A different quantity, not a smaller one. Same error class as the decoding arms'
chance level moving with behaviour, arriving through the pair set instead.

**TRANSFER did not require the pattern to resemble anything.** PS94 far_center, cue-aligned:
normal_r −0.632 / mirror_r −0.480 on 8/17, and −0.100 / +0.043 on 8/18. The post-stroke pattern is
*anti*-correlated with its own pre-stroke pattern and barely correlated with the mirrored one — it
resembles neither — yet mirror beat normal by more than the margin, so both days were flagged
TRANSFER. That would have put "the representation relocated across the midline" in the deck for the
position where the representation had **disappeared**. An earlier version of the same rule flagged a
0.005 correlation difference. Three verdicts now: TRANSFER (needs `mirror_r ≥ 0.20`, ahead of normal,
and beating the pre-stroke baseline by 0.15), REDUCED ASYMMETRY, and **PATTERN LOST**.

### Results after the corrections

**Midline transfer is a clean null** — no transfer at any position, in any animal, at either
alignment, on either arm. The "left map moved right" reading of the map observation is not supported.

What is there instead is **pattern loss**: cue-aligned on day 1, far_R has lost its pattern in all
four animals, and PS94 and PS95 lose far_center too. Those are the positions the animals stop
attempting, so on that arm the finding is confounded with the absence of the movement and must not be
read as a lesioned sensory representation. The **pre-cue** arm, which precedes the movement, shows no
such far-position concentration — its losses are scattered and mostly PS95's close positions. That
asymmetry is the caveat, not a footnote to it.


## Deck writes are guarded, not banned (2026-08-19)

The single place to look before touching deck builders. It replaces a blanket ban in this file and
another in `runbooks/helper_box_setup.md`, both of which described code that stopped existing on
2026-08-14 and disagreed with `docs/STATUS_2026-08-14.md`.

Three failure modes, three guards, each at the point where the damage would happen:

**A partial run must not prune another run's decks.** `build_decks` deletes siblings it did not write,
which is right after a re-split and catastrophic when a date is split across machines (imaging box:
PS92/PS93; helper box: PS94/PS95). The prune is gated on `writeguard.covers_all(covered,
all_animals)` and a partial run declines loudly instead. This is why the plural form is now safe to
run, and why it is *preferable* to hand-rolling a filtered `build_deck`.

**Nothing may be deleted outside the Priya subtree.** Every prune delete goes through
`writeguard.assert_writable` (rule 1).

**A rebuild that found nothing must not replace one that did.** A deck is rebuilt in place and an
empty deck is perfectly valid, so this failed silently: PS93's deck was rebuilt with `sessions`
filtered on `s.get("animal")` — a key those dicts do not have; they carry `label`, and `_animal_of`
exists for exactly this — and 257 MB became 264 kB. `_check_replacement` now REFUSES when no figure
of any type was found, and `_warn_if_shrunk` REPORTS a sharp size drop without blocking, because
legitimate rebuilds do shrink (excluding the regime-A sessions dropped five cohort-wide). `force=True`
skips both.

**The general rule this came from:** when a guard is needed, put it where the damage happens and let
the operation stay usable. A prohibition in prose protects only the reader who finds it, and pushes
everyone else onto a less-tested path.

## THE ALLEN-ALIGNED SVD BASIS IS NOT ORTHONORMAL — coefficient-space RDMs use the wrong metric (2026-08-19)

Priya asked for RSA on the SVD maps themselves. The shortcut is to take distances between per-position
SVT **coefficient** vectors and call it pixel space, since the map is `U @ svt` and `U` is a fixed
linear map. That is only valid if `U`'s columns are orthonormal. After Allen registration they are
not — the affine warp resamples the spatial components, and resampling does not preserve
orthogonality.

Measured over **all 52 sessions** (`pixel_rsa`, `G = UᵀU` over the Allen brain mask, k=100):

| | range across sessions |
|---|---|
| `‖G − I‖_F / √k` | **0.217 – 0.290** |
| diagonal of `G` | as low as **0.408** (PS94_0814); typical 0.70–1.31 |
| max off-diagonal | up to **0.43** (PS95_0814) |

On a random coefficient pair the naive distance is **off by 14.6%**. This is not a quirk of one
session: every session is 20–30% away from the metric that coefficient-space distance assumes.

### The fix, and why it costs nothing

The exact pixel-space squared distance is `d²(a,b) = (a−b)ᵀ G (a−b)`. Taking the Cholesky factor
`G = LᵀL` and transforming `z = L a` makes ordinary Euclidean distance in `z` **exactly** pixel
distance — verified to **2.8e-16** — so every existing crossnobis/RDM routine can be reused unchanged.
`G` is only k×k, so no pixel maps are ever held in memory.

The mask is load-bearing: `G` computed over the whole frame is dominated by background pixels, which
carry registration edge artefacts and no signal.

### What this does and does not invalidate

It does **not** touch any result computed on **Allen ROIs or LocaNMF components** — those are
averages over anatomically defined sets, not SVD coefficients, and their geometry is whatever those
features are. Every RDM currently in the deck is of that kind.

It **does** mean that a coefficient-space RDM — the obvious way to "do RSA on the maps" — would have
been measuring the basis as much as the brain, and that the pixel-space version is the one worth
running. Whether the convergence and midline results survive in pixel space is the open question this
was built to answer.

## WHICH BASIS FOR WHICH QUESTION — within-day vs cross-day (Priya, 2026-08-19/20)

Priya, on being shown the pixel-space mirror result: *"maybe pixel basis is not the right thing to
use, as pixels shift day to day."* That is correct, it is a better diagnosis than the one I gave, and
it splits the geometry analyses cleanly in two.

### The measurement

Pixel-space correlations, PS94, `close_center`:

| comparison | r |
|---|---|
| same position, 8/12 vs 8/13 | +0.666 |
| same position, 8/12 vs 8/14 | **+0.189** |
| same position, 8/13 vs 8/14 | +0.443 |
| **different** positions, within 8/12 | median **+0.704** |
| **different** positions, within 8/13 | median **+0.684** |

**In pixel space, two different spout positions on the same day resemble each other more than the
same position does across days.** Day-to-day variation — registration residual, hemodynamics, LED
setting, brain state — dominates position identity.

### The consequence

- **WITHIN-DAY questions are safe in pixel space.** Crossnobis convergence is computed inside a
  single session (between-position distances, cross-validated over that session's blocks). A
  day-to-day shift moves all six positions together and largely cancels. This is why pixel and ROI
  agreed on **12 of 12** convergence verdicts with a worst z gap of 0.8.
- **CROSS-DAY questions are not.** The mirror test correlates a post-stroke pattern against the MEAN
  OF OTHER SESSIONS' patterns. In pixel space that comparison is dominated by the day term, and no
  threshold repairs it. **The pixel mirror verdicts are withdrawn.** The ROI mirror null stands.

ROI averaging is therefore not merely "smoothing that inflates correlation" — for a cross-day
comparison it buys robustness to exactly this spatial wobble. That is a reason to use it, not a
defect.

### The threshold problem was real but secondary

`MIN_RESEMBLANCE = 0.20` was calibrated on ROI correlations (median post-vs-pre +0.830) and carried
unchanged into pixel space (median +0.219), where it flagged 46% of positions "pattern lost" against
25% in ROI. `_flag_mirror` now takes the floor as an argument and records which floor produced each
verdict. But the deeper point is that a correctly-calibrated floor would still not make a cross-day
pixel correlation measure what the mirror test needs.

### What each basis is for

| | retains of the pixel map | cross-session basis | use for |
|---|---|---|---|
| Allen ROI | 64.5% (92% lost in MOp_right) | fixed anatomy | cross-day pattern correlation |
| per-session LocaNMF | 98.6% | **no** — refit each day | within-session only |
| joint LocaNMF | ~98% | yes, fitted once | the right vehicle for cross-day, untested |
| pixel (Gram-corrected) | 100% by construction | no | within-day geometry |

The Allen-transformation correction (`pixel_rsa`, `G = UᵀU`) is a separate matter and still holds:
warping U onto the Allen grid leaves its columns neither unit-length nor orthogonal (‖G−I‖_F/√k =
0.217–0.290 across 52 sessions), so coefficient distance is not map distance. That is true regardless
of which basis is chosen for which question.

## THE MIDLINE NULL REPLICATES IN THE JOINT BASIS — and the floors measured something (2026-08-20)

The mirror test re-run in the **joint-LocaNMF basis**: fitted once with every session projected onto
fixed footprints, so cross-day comparison is legitimate, and retaining ~98% of the pixel map against
Allen ROI's 64.5%. Same k×k algebra as `pixel_rsa`, different footprints.

**Transfer at exactly one position** — PS94 8/18 pre-cue `far_center` — across four animals, three
post-stroke days and two alignments. The "left hemisphere's map relocated to the right" reading
remains unsupported in a basis where the averaging could not be hiding it.

### The data-derived floor turned out to be a measurement

The floor is no longer a picked constant: for each position it is **the worst that that animal's own
PRE-stroke sessions manage against each other in the same basis**. So it reports day-to-day pattern
reproducibility in an intact animal:

| animal | post-cue (mean / worst) | pre-cue (mean / worst) |
|---|---|---|
| PS92 | 0.96 / 0.89 | **0.69 / 0.19** |
| PS93 | 0.94 / 0.86 | **0.82 / 0.23** |
| PS94 | 0.88 / 0.72 | **0.84 / 0.41** |
| PS95 | 0.93 / 0.82 | **0.67 / 0.07** |

**Post-cue spatial patterns are highly reproducible across days (0.88–0.96). Pre-cue patterns are
not** — PS95's worst pre-stroke pairing is r = 0.07.

Consequence, and it is a live caution rather than a footnote: any **cross-day pre-cue PATTERN
comparison** rests on a signal that is not stable across days even before a lesion. This does NOT
touch the pre-cue decoding results or the crossnobis convergence — both are computed within a
session — but it applies directly to pre-cue pattern similarity, and it is a candidate explanation
for why the pre-cue results have been the hardest to pin down.

PS94 remains the outlier on loss: its post-cue pattern falls below the pre-stroke floor at 5–6 of 6
positions on all three post-stroke days.


---

## THE ENGAGED CUT WAS 2.0 s WHILE THE TASK'S WINDOW IS 3.5 s (fixed 2026-08-21)

`decode.max_rt_s` defined "engaged" for every imaging analysis at 2.0 s, while the task has run a
3500 ms response window throughout and the behaviour pipeline scores hit/miss on it. **A lick at
2.5 s was a rewarded HIT that every decoder filed under "no lick".**

The deck already carried the measurement and a standing warning (`M_NOLICK`): the contamination is
9.7% of PS94's no-lick arm and 4.7% of PS95's, **but 39.3% for PS92 and 33.9% for PS93** — and
"when PS92/PS93 re-enter as post-stroke animals these slides must be rebuilt on the three-arm split,
or they will report a late-lick effect as a no-lick effect." They re-entered on 8/18. The condition
was met and the rebuild was overdue.

Measured on the trial population before changing it: late-but-rewarded trials are **1.8% of
pre-stroke responded trials and 3.1% post-stroke**, graded by DISTANCE not laterality — far_L worst
in both phases (4.1% / 7.3%), close positions lowest (0.5–1.8%). Small against all trials, large
against the no-lick ARM, which is the denominator that matters because that arm is what the no-lick
analyses are about. **It is also present pre-stroke**, so the contamination sat in the REFERENCE that
defines every coding direction, not only in the post-stroke classes.

Two denominators, and the wrong one was quoted first: "% of responded trials" (1.8/3.1) reads as
negligible; "% of the no-lick arm" (up to 39%) is the one that governs.

### What changed and what did not
- `decode.max_rt_s` 2.0 → 3.5. Analysis WINDOWS stay at 2.0 s: the cut is a claim about whether the
  animal responded, the window a claim about how much activity to average.
- The lick window was already safe — it starts at the FIRST LICK wherever that falls, so a lick at
  2.5 s takes 2.5–4.5 s. Worst case under the new cut ends at 5.5 s, inside the **minimum measured
  cue-to-cue interval of 8.00 s**, so no window can reach the next trial.
- `nolick_decoder` KEEPS its own hardcoded 2.0 s boundary. With both cuts at 3.5 s its
  `late_rewarded` arm would be empty by construction, and that split is a real result: on PS93 8/12
  the entire pre-cue survival sat in the LATE arm (balanced 0.532, p=0.003) while undetected trials
  showed nothing (0.153, p=0.76).
- `CACHE_VERSION` 9 → 10. **Every number computed before this used the 2.0 s cut and will move.**

---

## PER-POSITION CODING DIRECTIONS — construction, and four things that were wrong first (2026-08-20/21)

For each spout position P, a direction fitted on PRE-STROKE trials with a SUCCESSFUL LICK, P against
the other five, in the shared joint-LocaNMF basis. The feature space is one axis per (component,
time sub-bin) — 348–380 dims for ENL/cue, ~700 for lick — so a direction is a weight per component
PER MOMENT. It is a CONTRAST: without the comparison there is no axis.

### 1. Reported as a LINEAR projection, not a probability
`predict_proba` was carried over from an earlier analysis and was never a considered choice. A
sigmoid saturates; these directions reach AUC 0.98, so pre-stroke lick already sat in the flat
region. Degradation measured from a saturated reference is understated, and **unevenly** between
positions of different separability — which corrupts exactly the orderings the analysis is for. The
squashing also depends on the regularisation and feature scale, so probabilities are not
commensurable across panels shown side by side. Now `x·w` on a unit vector, pole-normalised so
0 = pre-stroke not-this-position and 1 = pre-stroke lick here.

### 2. Difference-of-means directions are heavily contaminated by ENGAGEMENT
`cos(w, engagement axis)` reaches **0.82 / 0.91 / 0.71 / 0.52** in PS92/93/94/95 — and lands on a
DIFFERENT position in each animal (far_center, far_center, far_L, far_R), so it cannot be inspected
around. PS93's far_center direction is 91% engagement axis wearing a position label.

Symptom that found it: pre-stroke no-lick scattering from **−2.03 to +1.38** across axes that should
all read alike. After Gram-Schmidt removal they collapse to **0.16–0.17** and the pre-stroke lick
diagonal IMPROVES rather than degrading. Logistic directions were already clean (|cos| ≤ 0.07)
because they account for covariance; kept as the independent check.

**Consequence: raw `dom` must not be used for the no-lick classes.** ENL and cue show `dom_orth`.

### 3. One-vs-rest axes for MIDDLE positions are largely close-vs-far
"Not P" mixes the five other positions, and for a middle position that mixture is majority-far. On
PS94's close_center axis the ordering is **close_L 1.23 > close_R 0.83 > close_center 0.71** — the
position the axis is named for is third on its own axis, which is how a cell exceeded 1.0. Its
direction also has the lowest AUC (0.78) of the six for the same reason. **Prefer the pairwise
(A-vs-B) panels for remapping questions**; one-vs-rest is context.

### 4. Off-diagonal magnitude is meaningless without the pre-stroke baseline
Neighbouring positions are intrinsically similar before any lesion — pre-stroke far_center already
scores 0.76 on the far_R direction. Post-stroke cross-matrices are therefore reported as a
DIFFERENCE from the pre-stroke lick matrix; remapping is a departure from that baseline, not a large
number.

### Thresholds
`MIN_TRIALS` was 12, chosen for no reason and inconsistent with this project's existing rule
(`plot_poststroke.MIN_N` = 10, which G2b red-hatches below). Now 10. Separately, the within-session
figure suppresses a cell on **PRECISION rather than count**: drawn only if its own SEM < 0.25, a
quarter of the pole separation. Measured within-class SD of the projection is ~1.08 (median over 168
well-populated cells), so n=12 bought a SEM of 0.31 — a third of the entire scale, enough to invent
a shape from noise. SEM 0.25 ≈ n ≥ 20 at the median SD, and is still generous: its 95% interval
spans nearly the whole pole separation.

---

## WHAT THE CODING DIRECTIONS SHOW — one replication, two nulls (2026-08-20/21)

### Replicated: post-stroke LICK degradation tracks the behavioural deficit
In the lick window the pre-stroke position code is near-identical across animals (**0.807–0.874**)
and post-stroke it degrades in proportion to the deficit: **PS92 0.776, PS95 0.752, PS93 0.545,
PS94 0.380.** The same ordering appears in the cue window's post-stroke-lick column. PS94 is the
animal whose far_center/far_R response rates reached 0.00 and whose far_R first-lick latency went
0.255 s → 2.439 s.

**Not under-description by the basis.** Variance captured is 0.98–0.995 for every post-stroke
session, and it runs OPPOSITE to the ordering — PS94 is the BEST-spanned animal (0.992–0.995) while
showing the largest drop; PS92 the worst-spanned (0.983) while retaining the most. Under-description
would make those move together.

This contrast is like-for-like (both sides contain a lick), which the no-lick classes in the cue and
lick windows are not.

### Null: the two post-stroke failure modes are indistinguishable
MISS-while-working exceeds STOPPED at **14/20 positions in ENL and 10/20 in cue**, magnitudes
swinging both ways. PS94 alone was 6/6 in ENL and the other three did not follow. ENL also has
little dynamic range here — every class sits between 0.21 and 0.56, consistent with pre-cue decoding
being ~0.3 against ~0.7 post-cue.

### Null: no within-session gradient in the miss class (PS94, ENL)
Engagement is graded, not binary, so a miss just before the quit might not be the state a miss at
trial 50 is. Binned by within-session quartile, the two positions with enough trials show no
monotone drift (far_R 0.55 → 0.60 → 0.40 → 0.95; far_center 0.4 → 0.95 → 0.55 → 0.35), while
pre-stroke lick stays flat at ~1.0 across all four quartiles — so a drift WOULD have been visible.
One animal, one window; not settled.

STOPPED is terminal by construction, so its early bins are structurally near-empty (0 trials in the
first quartile at every position). Four-trial points landing at ±2.2 dominated the eye and invented
a shape until the precision rule above suppressed them.

---

## THE POST-STROKE ENGAGEMENT GATE — revisited, and what the data actually supports (2026-08-20/21)

`POSTSTROKE_ENGAGEMENT_FILTERING` stays **False**; nothing filters on any of this. What follows is
reported only.

### The post-quit window IS a genuine, position-general stop
Inside it the response rate is ~0 at EVERY position, close ones included (PS92_0818 0.00 across all
six; PS94_0819 0.02/0.02/0.02/0.02/0.00/0.03). An earlier reading of PS95's class composition as
"still licking at far, failing at close" was wrong — that window is 82 trials and the task simply
presented more close positions in it. So "stopped" is what it claims.

### The gate needs NON-RECOVERY, not a rolling rate
The 2026-08-18 retirement note objected that "a short run of MOTOR failures produces that dip just
as readily as a motivational lapse". True of a rolling rate. Requiring the reference-position
collapse to be terminal — to not recover — separates them: PS94_0817's reference rate drops around
trial 420 and is back near 0.95 by 480, and that session should not be called disengaged at all.
With non-recovery required it flags **0%** of that session (was 7%), while the genuine quits keep
their flags.

### The discriminator test failed its own control, and then the control failed too
Pushing states through a pre-stroke lick/no-lick axis gave a drift control (early vs late ENGAGED
trials, a null contrast by construction) of **0.09–0.12 with the sign FLIPPING between animals** —
as large as the state effect. But D1 was not a valid null: engagement is graded, so late engaged
trials genuinely sit closer to disengaged. A stronger control on pre-stroke LICK trials
(thousands, LOSO) gave **+0.008 / −0.022 / +0.033 / +0.105** — near zero for three of four animals,
so the axis is NOT a clock and my first explanation was wrong.

Then the position audit showed what actually drove it: the classes differ enormously in position
composition (total variation **0.31–0.65**; MISS is 34–44% far_R, STOPPED near-uniform) and pre-cue
activity CARRIES position. A position-blind axis was comparing the spout, not the state. That is why
every subsequent analysis is fitted per position and compared only WITHIN a position.

---

## THE LICK WINDOW CAN CARRY THE NO-LICK CLASSES, AT AN INFERRED TIME (2026-08-21)

At `align="lick"` an engaged trial's window starts at its FIRST LICK while a no-lick trial's started
at the CUE — offset by the whole reaction time. Pre-stroke that is 0.137–0.255 s and merely untidy;
at post-stroke far_R the median is **2.439 s against a 2 s window, so the arms do not overlap at
all**. `_trial_features` returns those trials populated and plausible-looking either way, which is
the same trap as the post-lick confusion bug of 2026-08-20.

`nolick_ref="would_be_lick"` starts the window at the cue plus **this session's own median RT at
that position**, taken from its engaged trials — latency differs by animal, by position and tenfold
between phases, so a cohort constant would be wrong on all three axes. Under 5 engaged trials at a
position falls back to the session median; no engaged trials drops those trials rather than guessing.

**It is an inference, not a measurement.** The time comes from other trials; this one has no lick,
which is the point. It is weakest exactly where the deficit is worst — and PS94's most extreme value
sits on it (post-stroke miss at far_R, −0.55 orthogonalised, n=295, window placed ~2.4 s after the
cue). Any slide from this must say so.

**In this window the engagement axis IS a licking axis** (movement present vs absent), so the
orthogonalised variants ask what position structure survives once movement PRESENCE is removed. That
is right for the no-lick classes and deliberately conservative for the lick classes: licks to
different spouts differ in kinematics, so position and position-specific MOVEMENT are not separable
there by any projection.

---

## DECK SELF-CHECKS — what the deck can now refuse (2026-08-20)

Three failures shipped a healthy-looking deck before these existed.

1. **Missing figures block the publish.** A rebuild that cannot find every figure will not overwrite
   an existing deck, and it NAMES the gaps — the caller truncates exceptions to 80 chars, which is
   exactly the detail needed. `allow_missing=N` for the deliberate case.
2. **A run with FAILED STEPS blocks the publish.** The missing-figure gate cannot see this: a step
   that dies PART WAY leaves its earlier outputs rewritten and its later ones at yesterday's values —
   every file present, nothing missing, a deck silently mixing two days. That is what
   `spatial_reorganisation` did on 2026-08-20 (all-trials arm rewritten, lick-only arm dead on a
   KeyError in a PRINT statement). The run already knew; the deck never asked.
3. **A manifest of every placed figure and its mtime**, with the ones this run did not refresh named
   in the log. REPORTED, not enforced: of 1311 PNGs in the tree only 402 were touched by that night's
   run, the rest being one-off analyses back to June. A blanket freshness rule would fire on two
   thirds of the tree nightly. Scoped to what the deck PLACES it is informative — 119 of 465 — and
   sorting oldest-first surfaced an entire cluster of ORPHANS (`poststroke_grid`, `G7_control_*`,
   `G4b`, `G7c`): slides reading filenames no step writes any more, frozen for two days, invisible to
   every other check because the files were present.

**Deck slides must not be added before the figure is in the nightly.** A slide reading a figure no
step regenerates is an orphan the day it is made.


---

## THE WOULD-BE-LICK FALLBACK WAS WRONG EXACTLY WHERE IT MATTERED (found and fixed 2026-08-21)

The lick-window no-lick reference added earlier the same day places a no-lick trial's window at the
cue plus **that position's median RT among the session's engaged trials**. Positions with fewer than
five engaged trials fell back to the SESSION median. That fallback was the failure: **it fires
precisely where the animal has stopped licking, while the session median is set by the CLOSE
positions that still work.** Measured across the post-stroke sessions:

| animal / position | engaged trials per session | offset used | own median when any lick exists |
|---|---|---|---|
| PS94 far_R | 0 / 1 / 2 / 0 (of ~100 no-lick each) | 0.17–0.23 s | **1.80 s, 2.25 s** |
| PS94 far_center | 0 on 0817 | 0.20 s | 0.75 s the next day |
| PS92 far_R | 0 on 0818 | 0.23 s | **2.20 s** on 0819 |
| PS95 far_R | 1 on 0817 | 0.20 s | 0.37 s (harmless) |

So those windows sat up to **2.1 s early inside a 2 s window — they did not overlap the inferred lick
at all**, and the error was largest at the most impaired positions. That correlates the artefact with
SEVERITY, which is the very axis these figures report. PS94 far_R miss was the most extreme value in
the whole table (−0.55) and it is one of the affected cells.

This is the same shape as the bug the reference was introduced to fix, one level down: a plausible
number returned for a window placed nowhere near the event it claims to describe.

### The fix
A position with **no engaged trial gets no offset and its no-lick trials are DROPPED** — this session
says nothing about that position's latency, and borrowing one from the positions that still work is
worse than having none. A position with one to four engaged trials **keeps its own median**, flagged:
it is order-of-magnitude evidence rather than a real median, and being off by a factor of two beats
being off by 2 s. `min_trials` now gates only the flag. The drop count and the weak positions are
printed per session. `CACHE_VERSION` 10 → 11.

The test that pinned the old fallback (`test_a_position_with_too_few_engaged_trials_falls_back`) now
asserts the opposite, with the measurement above in its docstring — a test can pin a decision that
turns out to be wrong, and rewriting it needs the reason attached or it will be "corrected" back.

### Two controls that came out clean, and are worth keeping
**Amplitude — tested twice, and the second test is the one that settles it.** The linear projection
is unbounded, so a trial sitting further from its session's engaged centroid scores higher whether or
not its ANGLE to the direction changed. PS92 reached 2.12 and PS95 1.32, FURTHER along the pre-stroke
axis than pre-stroke licks themselves, which needed ruling out.

*First pass, per-position norm ratios (lick window 0.79–1.51; the ENL window is tighter at 0.84–1.09,
and quoting the ENL range for a lick-window result was the first version of this note).* The
BETWEEN-ANIMAL ordering is not amplitude: mean norm ratio is 1.02–1.10 for every animal while mean
projection runs 0.46–1.12, and the ordering inverts — **PS93 has the highest mean ratio (1.10) and the
lowest projection (0.46)**. Over all 22 measurable cells r = +0.19, against a mean absolute gap of
0.39 from the projection-equals-ratio line that pure gain would produce. But WITHIN PS92 the five
cells correlated with amplitude at **r = +0.97**, and this note first concluded that PS92's
cell-to-cell pattern must not be read as position structure.

*That conclusion was wrong, and the reasoning behind it was too.* The projection and the norm are NOT
independent measurements — a trial moving further out ALONG the direction raises both — so a high
correlation between them is partly guaranteed by construction and cannot distinguish an artefact from
genuine increased separability. What separates them is re-projecting UNIT-NORMALISED trials,
`cos(x, w)`: blind to magnitude, sensitive only to direction.

    PS92 far_center   2.12 -> 1.75      every other PS92 cell moves <= 0.07
    PS93/94/95        all cells move <= 0.15
    r(raw, unit-norm) +0.99 / +0.86 / +1.00 / +0.97   (PS92/93/94/95)

**The pattern is directional.** Magnitude contributes about a third of far_center's excess over 1.0
and direction carries the rest; it stays well above 1. PS92's r=+0.97 reflects far_center being both
the most distinctive and the highest-norm cell — one phenomenon measured twice. Post-stroke values
above 1.0 are real, and the note that said otherwise stood for about six hours.

**Within-session RT drift — holds where it was used, and NOT in general (corrected 2026-08-22).** A
session-CONSTANT offset misplaces late trials progressively if the animal slowed through the session,
which is the exact shape of the gradient below. The first pass measured the session-wide median for
PS94 (0.200/0.200/0.200/0.200 on 0817) and concluded ≤0.05 s. Building it as a figure, per position
and per animal, showed the answer splits by RING:

    CLOSE positions   flat in every animal, every position: <= 0.03 s drift over the four quartiles
    FAR positions     PS94 <= 0.05, PS95 <= 0.03 -- but PS92 +0.13 / +0.15 / +0.23, and
                      PS93 far_center +0.27, far_L +0.50 (0.53 s -> 1.03 s, nearly doubling)

**The control survives for the claim it was made about**: the within-session neural decline sits at
CLOSE positions in PS94/PS95, and latency there is flat in every animal, so a session-constant offset
cannot have manufactured it. **It does not survive as a general statement.** For PS93's far_L the
session median near 0.6 s against a last-quartile 1.03 s misplaces those windows by ~0.4 s — a fifth
of a 2 s window, a smaller repeat of the fallback bug fixed the day before, and the reason to move to
a per-quartile offset if the far-position no-lick cells are ever read closely.

**And it separates two processes that had been treated as one.** DISENGAGEMENT is uniform across
positions and shows as SKIPPING; FATIGUE is position-specific and shows as SLOWING at the animal's
hard positions. PS93's far_L and far_center — where its right orofacial deficit lives — slow steadily
through a session while its close positions do not move at all, and the two animals that disengage
MOST (PS94, PS95) barely slow anywhere. "Trials skipped, not slowed" is therefore right for PS94/PS95
and wrong for PS92/PS93 at far positions, where both happen.

---

## THE WITHIN-SESSION GRADIENT DID NOT SURVIVE THE OFFSET FIX — and the control it rested on is not flat (2026-08-21)

Recorded here because it was written up as a finding hours before it was withdrawn, and the reason it
looked real is the useful part.

**The claim.** The 2026-08-20 null said PS94's miss class showed no monotone within-session drift,
measured in the ENL window. In the LICK window the same animal and class appeared to decline
monotonically across within-session quartiles — far_R −0.26 → −0.44 → −0.69 → **−1.49**, far_center
0.21 → −0.12 → −0.27 → −0.47 — so the null looked window-specific rather than general. Two mechanical
explanations were checked and both failed: the would-be-lick offset is a session CONSTANT, so it
cannot manufacture a gradient, and median engaged RT does not drift within a session.

**What it actually was.** Both checks were sound and both were beside the point. The offset does not
drift within a session, but it was WRONG at far_R in the sessions that dominate the late quartiles:
PS94_0817 and PS94_0820 contributed nearly all the 4th-quartile far_R trials, and those are exactly
the two sessions where far_R had zero engaged trials and the window was placed ~2 s early. Dropping
them removes the 4th quartile entirely and flattens the rest:

    far_R       old  -0.26 -> -0.44 -> -0.69 -> -1.49        (swing 1.23)
                new  -0.74+-0.10 -> -0.96+-0.21 -> -1.02+-0.32 -> (n=0)   (swing 0.28, <1 SEM of the difference)
    far_center  old  +0.19 -> -0.14 -> -0.26 -> -0.45        (monotone)
                new  +0.34 -> -0.15 -> -0.01 -> (n=0)        (not monotone)

So the gradient was largely WHICH SESSIONS landed in which quartile, and the ones that landed late
were the mis-placed ones. **The ENL null stands, and the lick window does not contradict it.** Note
that the pooled far_R miss value moved the other way (−0.55 → −0.86, n=295 → 113): the mis-placed
windows sat near zero because they described an arbitrary moment, so they were DILUTING the pooled
effect while inventing the within-session one.

**And the control was not a control.** "Pre-stroke LICK stays flat at ~1.0 across all four quartiles,
so a drift would have been visible" is true at the far positions (far_R 0.81 → 1.08, far_center 0.94
→ 1.01) and false at the close ones, where it declines steadily and with tight errors:

    close_center  1.39+-0.05 -> 1.02+-0.06 -> 0.80+-0.06 -> 0.67+-0.09     (n = 275/255/244/183)
    close_L       1.33+-0.04 -> 0.97+-0.03 -> 0.92+-0.04 -> 0.71+-0.06

That is a ~0.7 decline over the session on SUCCESSFUL pre-stroke licks, many SEM wide, at the
positions with the most trials. Whatever it is — satiety, arousal, engagement, slow signal drift — it
is present in trials where the animal did the task correctly every time, so **a within-session
comparison cannot assume a flat baseline**. Any future version of this analysis has to subtract the
pre-stroke lick profile at the SAME position rather than treat 1.0 as a fixed reference.

---

## THE PRE-STROKE WITHIN-SESSION DECLINE IS DISENGAGEMENT, READ THROUGH AN ASYMMETRIC AXIS (2026-08-21)

Priya's reading of the close-position decline — "this is mostly because of lack of engagement at the
end, right?" — is correct at the ANIMAL level and does not survive at the POSITION level, and the gap
between those two is the useful part.

### The animals that disengage are exactly the animals with the decline

Pre-stroke response rate, first to last within-session quartile, pooled over 11 sessions each:

| animal | q1 → q4 | spread over the 6 positions | within-session neural slope |
|---|---|---|---|
| PS92 | 0.99 → 0.90 (**−0.09**) | −0.05 to −0.14 | all ≤ 0.45, no close/far pattern |
| PS93 | 0.91 → 0.85 (**−0.06**) | −0.05 to −0.11 | all ≤ 0.33, no close/far pattern |
| **PS94** | 0.99 → 0.74 (**−0.25**) | −0.22 to −0.29 | close −0.35/−0.72/−0.62, far +0.27/+0.08/+0.12 |
| **PS95** | 0.99 → 0.64 (**−0.35**) | −0.31 to −0.39 | close −0.50/−1.19/−0.56, far −0.01/+0.30/+0.02 |

Median RT is FLAT across quartiles throughout (PS94/PS95 sit at 0.13 s in every quartile at the close
positions). These are trials **skipped, not slowed** — the sated tail that `flag_engagement` was
built for, not fatigue. PS93's far_L is the one behavioural exception, 0.53 s → 1.03 s with a
response rate of 0.71 → 0.60, which is its known orofacial deficit and not a within-session effect.

### But the disengagement is uniform across positions and the readout is not

Within PS94 and PS95 the behavioural drop is the SAME at every position (PS95: −0.31 to −0.39 at all
six), while the neural decline lands on the close axes and leaves the far ones flat or rising. So the
decline cannot be read position by position, and a position-specific interpretation of it would be
wrong.

**The reason is structural, and it is the one-vs-rest flaw again.** "Not close_center" is majority-far,
so a close-position axis is largely a close-vs-far contrast, and a uniform state shift along that
dimension loads on it asymmetrically. Measured directly — cos between each position's direction and
the pre-stroke close-vs-far axis, against that position's within-session slope, over all 24
animal-position pairs:

    r = -0.567
    axes pointing toward CLOSE (cos > 0, n=12):  mean slope  -0.347
    axes pointing toward FAR   (cos <= 0, n=12): mean slope  +0.030

The one-vs-rest axes really are close-vs-far contrasts: |cos| runs 0.26–0.91 and its SIGN matches the
position's ring in 22 of 24 cases. The two exceptions are both PS93 (far_L +0.43, close_center −0.01),
the animal whose orofacial deficit already distorts its far-left geometry.

The relation is carried by PS94 and PS95; PS92 and PS93 have slopes near zero at every position, so
there is little to correlate. That is what the account predicts — the mechanism needs a state shift to
project, and those two animals barely have one.

**This is NOT the engagement axis.** The directions are already Gram-Schmidt orthogonalised against
lick-vs-no-lick, and cos(close-vs-far axis, engagement axis) is only −0.43 to +0.34. The within-session
drift runs along a state dimension distinct from the one that separates a lick trial from a no-lick
trial — which is why orthogonalising did not remove it.

### Consequences
- A within-session panel must be read against the **pre-stroke profile at the same position**, never
  against the poles. 1.0 is not a flat baseline.
- The pairwise (A-vs-B) panels should be far less exposed, since those axes are not majority-far.
  Untested — the obvious next check.
- Anything that compares a post-stroke class to pre-stroke lick WITHIN a session position is exposed
  to this, because post-stroke misses concentrate late in a session while pre-stroke licks do not.

---

## THE PAIRWISE AXES: WITHIN-RING IS SAFE, CROSS-RING IS NOT (2026-08-22)

The open question left by the disengagement result — whether the pairwise A-vs-B panels escape the
drift that the one-vs-rest axes carry. Tested on pre-stroke LICK trials, all 15 position pairs per
animal, with a built-in control rather than a single correlation: within-ring pairs (close-close,
far-far) should carry little of the close-vs-far dimension, cross-ring pairs should carry a lot.

**Half the prediction landed immediately.** Mean |cos| with the pre-stroke close-vs-far axis is
**0.33 for within-ring pairs against 0.70 for cross-ring** — the pairwise construction does strip out
most of that dimension when both positions share a ring.

**And the aggregate looked like a refutation.** Over all 60 pairs r(cos, slope) = −0.143 against
−0.567 for one-vs-rest, and mean |slope| was the same for both groups (0.176 vs 0.194). Read alone,
that says the pairwise axes drift as much as anything else and the mechanism is wrong.

**It was hiding the effect by pooling animals with and without a state drift to project.** Split by
whether the animal actually disengages:

| | cross-ring (A is always the FAR position) | within-ring |
|---|---|---|
| **PS94, PS95** (lose 0.25/0.35 of response rate) | mean **+0.187**, **positive 18/18** | mean −0.010, positive 6/12 |
| **PS92, PS93** (lose 0.09/0.06) | mean +0.009, positive 9/18 | mean −0.049, positive 5/12 |

Every cell except one is a coin flip. The exception is **18 out of 18 in the same direction**
(binomial p ≈ 4e-6), in exactly the two animals with the behavioural drift and exactly the pair type
that carries the close-vs-far dimension.

**The sign is the part that could have failed and did not.** In these pairs A is always the far
position, so a positive slope means far trials become MORE far-like as the session runs. In the
one-vs-rest axes, close trials became LESS close-like. Both are the same drift along close→far, seen
from opposite ends. That is a prediction the data could have contradicted and did not, which is what
turns the account of 2026-08-21 from plausible into measured.

### What to use
- **Within-ring pairwise (close-vs-close, far-vs-far) is the safe contrast for remapping questions.**
  Half the close-vs-far loading and no coherent drift even in the disengaging animals.
- **Cross-ring pairwise is not safe**, though at roughly half the magnitude of one-vs-rest (+0.19
  against −0.35). The deck note previously said "prefer the pairwise panels" without qualification;
  that was right for within-ring and wrong for cross-ring.
- One-vs-rest remains the most exposed of the three.

### Caveat
Two animals carry the whole result. PS92 and PS93 are not a negative control for the MECHANISM — they
are animals with no state drift to project, so their 9/18 is what "nothing to detect" looks like, not
evidence against it. A third disengaging animal would be the real test.

---

## THE POST-STROKE FAR-POSITION POST-LICK MAPS ARE MAPS OF ITI LICKING (2026-08-22)

Priya, looking at the preprocessing decks: "in the SVD post-lick 150 ms maps there are overt
differences across positions post-stroke" (PS94 slides 72 vs 73), and then "scale is also wildly
different". Both observations are correct, and chasing the second one found something worse than a
scaling problem.

### What the map's position label actually means
`framemap_event_maps.run_lick` labels **every detected lick** with a position and averages the 150 ms
after it. There is no response-window gate, no first-lick-per-trial restriction and no engagement
filter — by design, since the figure is "post-lick averages by spout position". With
`--behavior-trials` (which `preprocess.py` always passes) the label is **the most recent CUE's
position**, so every lick between one cue and the next carries that cue's label, however long after
it falls. The cue-to-cue interval is at least 8 s.

So the label means "which spout was most recently cued", NOT "the animal licked at this spout".
Pre-stroke those two coincide. Post-stroke, at a position the animal has stopped attempting, they
come apart completely.

### Measured on PS94 (counts reproduce the map's own summary json exactly)

| session | position | n licks | ≤3.5 s (a response) | >3.5 s (ITI) | % ITI | median t after cue |
|---|---|---|---|---|---|---|
| **0817** | far_center | 93 | **0** | 93 | **100%** | **7.20 s** |
| **0817** | far_R | 83 | **0** | 83 | **100%** | **7.64 s** |
| 0817 | far_L | 245 | 122 | 123 | 50% | 3.58 s |
| 0817 | close_R | 292 | 156 | 136 | 47% | 0.74 s |
| 0817 | close_L | 849 | 610 | 239 | 28% | 0.81 s |
| 0814 (pre) | far_R | 914 | 750 | 164 | 18% | 0.81 s |
| 0814 (pre) | far_center | 713 | 583 | 130 | 18% | 0.63 s |

**Slide 73's two most saturated panels contain no task responses at all.** PS94 made zero engaged
trials at far_center and far_R on 8/17 (independently confirmed: `would_be_lick_offsets` finds 0
engaged trials at both), and the licks in those maps sit a median of 7.2–7.6 s after the cue — late
ITI, essentially just before the next trial.

**And the contamination is graded by severity**, which is the dangerous part: 18% ITI pre-stroke at
every position, against 28% / 47% / 50% / 100% / 100% post-stroke, ordered exactly like the deficit.
An artefact that tracks severity reads as a result about severity.

### What this does and does not invalidate
- The **close** positions' post-stroke maps are still majority-response (72%, 53%) and can be read,
  with that caveat attached.
- The **far** positions' post-stroke maps cannot be read as position-specific responses. They are
  maps of whatever the animal was doing while that spout was pending.
- It does NOT touch any decoder, encoder, RSA or coding-direction result: every one of those uses
  the FIRST lick within the response window on ENGAGED trials, so an ITI lick cannot enter. It is
  also why `fixed_scale_maps` prints "not attempted" for PS94 far_center and far_R — under 8 engaged
  TRIALS — while the per-session map happily shows n=93 and n=83 licks.

### The scale question, answered separately
`plot_lick_aligned_averages` / `framemap_event_maps` set the colour limit from a percentile of THAT
session's own maps, so cross-day comparison of two slides compares two different scales: PS94 is
±0.02425 on 8/14 and ±0.08854 on 8/17, a factor of **3.65**. On 8/17's range the whole of 8/14's
negative range would render near white, so "the post-stroke map lost its blue" is the scale, not the
biology.

On a COMMON scale (`fixed_scale_maps --post-s 0.15`, added 2026-08-22 — the module took a `post_s`
argument that no caller had ever passed, so only the 2 s version existed):
- at **2 s** the amplitude rise is real and large: pre-stroke peaks 0.019–0.052, post-stroke
  0.059–0.083.
- at **150 ms** it is much smaller: close positions 0.031–0.049 pre against 0.038–0.061 post
  (1.1–1.45x), with only far_L larger (0.025 → 0.058) on n=25 trials.

So the amplitude story is a 2 s story, not a 150 ms one, and the dramatic 150 ms per-session
appearance is scale plus low n plus ITI licking.

### Open
The map is doing what it says; the label is what misleads. The contained fix is for `run_lick` to
record the response/ITI split per position in its summary json and print it on the figure, so a
panel built from non-responses says so. That is a change to the IMAGING box's preprocessing pipeline
and has not been made.

---

## THE POST-LICK WINDOW SWEEP: 150 ms IS REAL BUT WEAK, AND THE BINNING GAIN IS UNDERSTATED (2026-08-22)

Priya, looking at the preprocessing decks' 150 ms post-lick maps: "could we try an individual
post-lick-based decoder?" Answered by sweeping the window rather than testing one length. One feature
extraction per session at 13 sub-bins over the 2 s lick window (154 ms each, matching the maps'
150 ms almost exactly), then decoding from cumulative subsets — six window lengths for the price of
one extraction, 44 pre-stroke and 18 post-stroke sessions.

### Two mistakes on the way, both caught by asking the obvious question

**Plain accuracy said post-stroke BEAT pre-stroke at 154 ms** — 0.614 against 0.576, which would have
supported the maps' appearance. It is class imbalance: the post-stroke sessions are precisely the
ones that LOST positions (PS94_0817 has zero engaged trials at far_R and far_center, so it is a
4-way problem scoring 0.824 against a chance of 0.25). On BALANCED accuracy the ordering reverses to
0.530 against 0.573. The gap between plain and balanced is 0.084 post-stroke and 0.003 pre-stroke,
which is the imbalance measured directly.

**The window curve confounded DURATION with DIMENSIONALITY** (Priya: "is this a mean or binned?").
Keeping the first k sub-bins as separate features grows the feature count with the window, 90 → 1170,
so a rising curve says nothing about time on its own. Re-run with the MEAN over [0, X) — features
held at ~90 for every row, only duration varying — and the two runs agree exactly at 154 ms, where
one bin IS the mean.

### Chance-corrected (BA − 1/K)/(1 − 1/K), so sessions with different K are comparable

| window | duration only (pre / post) | with sub-bins (pre / post) | binning gains (pre / post) |
|---|---|---|---|
| 154 ms | 0.488 / 0.436 | 0.488 / 0.436 | +0.000 / +0.000 |
| 308 ms | 0.595 / 0.529 | 0.702 / 0.587 | **+0.107** / +0.057 |
| 462 ms | 0.680 / 0.596 | 0.797 / 0.646 | **+0.117** / +0.049 |
| 769 ms | 0.767 / 0.650 | 0.855 / 0.699 | +0.087 / +0.049 |
| 1231 ms | 0.811 / 0.671 | 0.873 / 0.705 | +0.061 / +0.034 |
| 2000 ms | **0.816 / 0.671** | 0.865 / 0.694 | +0.049 / +0.023 |

### Three conclusions

**`decode.lick_post_s = 2.0` stays.** An earlier version of this note recommended 1231 ms on the
strength of the binned curve peaking there. That peak is dimensionality — going from 8 to 13 bins
costs a little to overfitting — and on duration alone the curve rises monotonically to 2 s. The
recommendation was withdrawn before it reached the config.

**The +0.023 recorded for sub-binning is a 2 s number and understates it badly elsewhere.** It is
right where it was measured and reaches **+0.117** at 462 ms. The temporal PROFILE within the window
carries most of what a longer window adds, which is worth knowing before anyone shortens a window to
save compute.

**A dedicated 150 ms post-lick decoder is not worth building.** It works — 0.488 pre and 0.436 post
against a chance of 0, so the window genuinely carries position — but that is ~60% of the full
window's information, and the pre→post drop is SMALLEST there (−0.052, against −0.145 at 2 s). The
deficit does not live in the first 150 ms. The observation that prompted it is separately explained:
those map differences were ITI licking (100% of the licks at PS94's post-stroke far positions) plus a
3.65x colour-scale difference, and the decoder never saw either, since it uses the first lick inside
the response window on engaged trials.

### Caveat kept attached
Three post-stroke sessions have fewer than six positions — PS92_0818 (K=5), PS94_0817 (K=4),
PS94_0820 (K=5) — and they are the MOST impaired days. The chance-corrected metric makes them
comparable, but a position the animal never attempts contributes no error, so the post-stroke column
is optimistic about exactly the sessions that matter most. Per-session results are in
`E:/cue_lick/lick_window_sweep.json` and `..._mean.json`.

## THE UNIFORM 1/6 NULL HAS FOLLOWED THE SKEW INTO THE ENGAGED ARM (2026-08-23)

On 2026-08-17 the entry above ("The null was wrong") recorded that comparing accuracy to a uniform
1/6 is invalid when the trials are skewed across positions, and the corrected machinery -- balanced
accuracy as headline, plus a permutation null computed on those trials with the model's predictions
held FIXED -- went into `nolick_analysis` and is used by `poststroke_compare` and
`impaired_nolick_readout`.

It was applied to the arm where the problem was noticed, not to the condition that causes it. The
condition has since arrived in the ENGAGED arm by a different route. There the skew came from
animals DECLINING far spouts; here it comes from post-stroke animals ABANDONING positions outright,
and the engaged trial counts are no longer close to uniform:

| session | n | uniform | best constant guess | engaged counts |
|---|---:|---:|---:|---|
| PS94_0812 (pre) | 456 | 0.167 | 0.178 | 78, 70, 78, 81, 73, 76 |
| PS94_0820 | 291 | 0.167 | **0.247** | 71, 72, 68, 63, 17, **0** |
| PS93_0818 | 380 | 0.167 | **0.232** | 88, 84, 77, 42, 79, 10 |
| PS93_0820 | 235 | 0.167 | **0.255** | 50, 48, 60, 28, 40, 9 |
| PS92_0822 | 294 | 0.167 | **0.276** | 58, 65, 81, 44, 41, 5 |

Pre-stroke the two agree to within 0.011 and the distinction does not matter. Post-stroke they
diverge by up to 65%: a constant "always guess close_R" scores 0.276 on PS92_0822 knowing nothing.

WHAT WAS AND WAS NOT AFFECTED. The headline post-stroke claims go through `evaluate_arm` and are
sound -- they were already balanced and permutation-tested. What was misleading is the PER-SESSION
decoder panels in sections A-C, which drew a flat `axhline(1/6)` and titled themselves "chance .17"
for post-stroke sessions whose floor is nowhere near that.

FIX (`locanmf_position_decoder`). Every session now computes, from predictions already in hand and
so at no fitting cost: the majority-class floor, balanced accuracy, and a permutation null with
predictions held fixed. The floor is drawn on the recall panel ONLY when it exceeds 1/6 by more than
0.005 -- pre-stroke the lines coincide and a second one would be clutter -- and the title carries
balanced accuracy with its permutation p. All of it lands in the per-session summary JSON.

THE GENERAL LESSON, which is the reason this is written down rather than just fixed: a reference
level is a property of the DATA, not of the task design. 1/6 was correct while the animals attempted
every position, and became wrong the moment they stopped -- without anything in the code changing.
The same shape has now appeared six times this week (an empty class scoring 1.0, a completeness
check against a shrunken date set, another against an inflated animal x date set, a guard running
after the thing it guarded, a refusal reported as a write). Each time the number was true and the
thing it was measured against was not.

---

## THE PER-SESSION VIEW OVERTURNS THE POOLED ONE, AND THE BASIS IS NOT TO BLAME (2026-08-23)

Priya asked whether the coding-direction plots were pooled over post-stroke sessions. They were —
`trials(c, P)` was called without a session argument everywhere except the one-vs-rest time course,
so the cross matrix and every pairwise cell averaged all post-stroke days into one number. The pooled
figure was chosen deliberately (the per-session view "showed swings as large as any trend", PS94's
miss-while-working going +1.05 to −0.68 between adjacent sessions), and its honest cost is that a
recovery and a collapse average to "no change".

`cross_by_session` and `pairwise_by_session` now store the same quantities per post-stroke session,
using the same directions and poles — only the projected trials change, so nothing is refitted.

### What the pooled numbers were hiding

Post-stroke LICK, pairwise mean over all pairs, ENL:

| animal | 0817 | 0818 | 0819 | 0820 | 0821 | 0822 |
|---|---|---|---|---|---|---|
| PS95 | **1.27** | 0.98 | 0.92 | 0.62 | 0.63 | — |
| PS94 | 0.82 | 0.60 | 0.60 | 0.71 | **0.48** | — |
| PS92 | — | 0.73 | 1.10 | 1.04 | 0.78 | **0.63** |
| PS93 | — | 0.56 | 0.67 | 0.65 | 0.68 | 0.65 |

**PS95 declines monotonically 1.27 → 0.63** — the animal the pooled value (+1.05) called intact.
**PS93 is flat at ~0.65 from day one.** All four converge near 0.6 by 8/21–8/22. So "severity-graded"
is the wrong frame for the pooled numbers; what the data shows is a **progressive decline that
severity sets the ONSET of, not the endpoint**, with PS93 starting where the others finish.

Separately, PS94's cross matrix GEOMETRY is settled from 0817 and unchanged through 0821 — close
trials shifted toward the far columns and away from close_center, identically every session — while
its magnitude declines. Geometry and magnitude are separable claims and should be kept apart.

### THE BASIS ARTEFACT CHECK — negative, and this is the one that licensed the above

The joint basis has FIXED footprints; a session outside the fitting set is PROJECTED onto them. If it
described later days progressively worse, every projection would shrink and manufacture exactly this
decline. `variance_captured` is now stored per session (with `in_basis`, because sessions that built
the basis are recorded as 1.0 by CONSTRUCTION and would poison the correlation).

    r(variance_captured, coding value) over 20 projected sessions = +0.174
    variance_captured spans 0.9631-0.9952 (0.032);  coding value spans 0.623
    at the fitted slope the FULL vc span explains 0.126, i.e. 20% of the observed spread
    dropping the single worst-fit session (PS94_0821, vc 0.9631): r = -0.228

The basis describes every post-stroke session above 96% and within 3.2% of the others. **The decline
is neural.** Caveat: this tests basis QUALITY, not basis APPROPRIATENESS — footprints can describe a
reorganised cortex's variance well while no longer being the right parts. Only refitting could see
that, and refitting defeats the point of a frozen basis.

---

## THE MISS CLASS IS WHERE THE CONTRAVERSIVE DATA LIVES (2026-08-23)

Priya: "what about the miss but working trials?" The lick class cannot speak to contraversive-far
encoding at all — the animal stopped licking there, so `far_R|far_L` has **n=5 for PS94 and n=15 for
PS92**. The MISS class has **182–399**: the animal was cued to far_R and did not respond, so the
trial exists and carries the intended position with no movement. That is the pre-cue plan, which is
what this study is about.

Pre-cue, miss-while-working, pure-lateral far axis (1 = pre-stroke A, 0 = pre-stroke partner):

| animal | far_R trials | far_L trials |
|---|---|---|
| **PS92** | **−0.00** (n=355) | **+1.35** (n=132) |
| PS93 | +0.30 | +0.38 |
| PS94 | +0.59 | +0.48 |
| PS95 | +0.61 | +0.61 |

**PS92's far_R pre-cue pattern sits exactly on pre-stroke far_L** while its far_L is hyper-preserved.
A lateralisation-specific collapse, and the cleanest in the cohort.

**THIS CORRECTS A CLAIM MADE EARLIER THE SAME DAY.** PS92 was put in the "code intact" group on its
pooled LICK value of +1.12. That number is computed over the positions it still licks, and PS92 on
8/21 was 4 hits / 79 trials at far_R with a 2.02 s latency. The lick class was reporting that the
animal is fine because it only licks where it can.

---

## WHAT CHANGES ABOUT LATERAL ENCODING — the analysis, and three things that had to be fixed first

Priya, 2026-08-23: "what is changing about how lateralized tongue movements are encoded in cortex?"
The six-label position code cannot answer this directly, so it is split into its two task dimensions
and the LATERAL one is fitted as its own axis: `w_side = mean(LEFT spouts) − mean(RIGHT spouts)`,
difference-of-means, unit-normalised, orthogonalised against engagement, in the (component × 0.5 s
sub-bin) ENL feature space. Every animal is lesioned LEFT, so RIGHT is contraversive.

### First it said the wrong thing, for three separate reasons

**1. The distance component did not always cancel.** Pooling `close_L+far_L` against `close_R+far_R`
removes close-vs-far only if the two sets match in close/far composition. Measured: fine pre-stroke
(|imbalance| ≤ 0.05) and fine post-stroke for PS94 (+0.01) and PS95 (−0.04), but **PS93 −0.17** and
PS92 −0.11. So the axis is fitted WITHIN A RING — `far_L` vs `far_R`, `close_L` vs `close_R` — where
no distance component can enter. Note that both sets go from ~0.48 far pre-stroke to ~0.80 post: the
post-stroke side axis is fitted mostly on far trials in every animal.

**2. There was no null.** `cos(pre-axis, post-axis)` means nothing without knowing how reproducible
the axis is at that trial count. Pre-stroke NO-LICK was the obvious null and is a bad one — Priya:
those trials are the sated tail, "a fundamentally different animal state", and orthogonalising
against the engagement axis removes only its linear component. **SPLIT-HALF of pre-stroke LICK,
subsampled to the post-stroke n**, is the right floor: same animal, same state, no lesion, 40 draws.

**3. The window is PRE-CUE, which defuses the movement objection.** The ENL window is lick-free by
construction, so NEITHER class contains a movement and the lick/no-lick difference is not a movement
difference. What remains is trial count, which is exactly what the split-half measures.

### The result: different beyond noise, in 7 of 7 measurable cells

| animal | ring | cos(pre-LICK, post-MISS) | split-half floor at matched n |
|---|---|---|---|
| PS92 | far | +0.151 | +0.724 [0.585, 0.814] |
| PS92 | close | +0.318 | +0.635 [0.474, 0.792] |
| PS93 | far | **−0.334** | +0.778 [0.672, 0.854] |
| PS93 | close | +0.187 | +0.446 [0.243, 0.595] |
| PS94 | far | **+0.053** | +0.832 [0.754, 0.897] |
| PS94 | close | +0.294 | +0.755 [0.674, 0.846] |
| PS95 | far | +0.150 | +0.456 [0.157, 0.662] — **marginal** |

The floor also VALIDATES the method: two halves of pre-stroke lick reproduce each other at 0.45–0.83
at these n, so the post-stroke values are not a small-sample artefact. Distance below the floor
orders PS93 > PS94 > PS92 > PS95, roughly the behavioural severity, and PS95 — least affected — is
the one marginal cell. **Left-hemisphere share stays 0.41–0.53 throughout**: the lateral code does
NOT change hemispheres, consistent with the midline test never finding relocation.

### AND A WORD THAT WAS NOT EARNED

This was written up as the lateral axis "re-forming". Priya: "How does the table you showed indicate
the lateral axis reforms?" It does not. A low cosine with the pre-stroke axis shows the post-stroke
axis is DIFFERENT; it cannot show that a coherent new axis exists. **PS94's far value of +0.053 is
exactly what NO lateral information looks like.** The distinguishing measurement is the post-stroke
axis's OWN split-half reproducibility — near 0 means the axis is gone, high means a stable new axis
near-orthogonal to the old one — and it had not been built. The pre-stroke floor existed and the
post-stroke one did not, which is the asymmetry that let the overstatement through.

Until that lands the defensible statement is: **the lateral axis is not what it was, and whether
anything coherent replaced it is unmeasured.**

### Still open
- **The same-class control**: post-stroke LICK vs pre-stroke LICK at the CLOSE ring, where post-stroke
  lick trials are plentiful. Same class both sides, only the lesion differs — kills the will-lick /
  won't-lick confound that the noise floor does not address.
- **A scale-free population-vector measure** (Priya's suggestion): cosine similarity between mean
  (component × 4 × 0.5 s bin) ENL vectors, position by position, pre vs post. Cosine is
  SCALE-INVARIANT, which separates "the pattern changed" from "the amplitude changed" — the confound
  that made PS92's coding-direction magnitudes uninterpretable, and a real one given the measured
  2–3× post-stroke amplitude rise at 2 s. Must be computed on residuals after subtracting the
  across-position mean, or the large shared task-evoked component dominates every comparison.
- **Tongue kinematics from the cameras.** The deepest limit on all of this: spout position is a proxy
  for tongue direction, and the deficit removes exactly the trials the question needs. Tracking the
  tongue would decouple "which spout was cued" from "how the tongue actually moved", give a
  continuous variable instead of six labels, and work on the trials the animal DOES perform. The
  video and the DAQ↔camera alignment templates already exist; the tracking does not.

---

## DEFERRED: POOLING POST-STROKE SESSIONS BY EPOCH (2026-08-23)

Priya: "we may later pool sessions by poststroke epoch (eg acute, subacute, chronic) based on
behavioral phenotype, but too soon to do this while still collecting data."

Recorded now because it constrains how the per-session analyses are STORED, and that decision is
being made today whether or not anyone notices.

**Why per-session is the right granularity while the epochs are unknown.** `position_axes.json` and
the `cross_by_session` / `pairwise_by_session` blocks key every cell by session label, so pooling by
epoch later is a regrouping of stored numbers rather than another pass -- about 40 minutes per animal
per window, since the cost is the feature load and not the fitting. Had these been written as pooled
summaries, the epoch question would have required recomputing everything, and the epoch boundaries
would then have been chosen while looking at the answers.

**The trap to avoid when the time comes.** Epochs defined by BEHAVIOURAL phenotype and then used to
test for NEURAL differences between epochs are not independent evidence -- the grouping already
encodes the behaviour, so "the neural measure differs between epochs" can be true by construction.
Two ways out, both cheaper to adopt now than to retrofit:

  * fix the epoch criterion IN ADVANCE from behaviour alone (a stated response-rate or latency rule,
    written down before the neural numbers are looked at), and treat the epoch labels as data rather
    than as something to tune; or
  * skip the categories entirely and report the neural measure as a CONTINUOUS function of the
    behavioural one -- day-by-day response rate at the contraversive positions against day-by-day
    disattenuated axis change. That needs no boundaries, uses every session, and is what the
    per-session storage already supports.

The second is the stronger analysis and the reason not to rush the first. Categories would be worth
it only if the trajectory turns out to be genuinely stepwise rather than graded, which is itself an
empirical question the per-session view can answer.

**Not yet, deliberately.** The cohort is still being collected, the animals are 5-6 post-stroke
sessions in, and the one trajectory measured so far (post-stroke lick coding value: PS95 1.27 -> 0.63,
PS93 flat at ~0.65, all four converging near 0.6) has no obvious breakpoint. Defining acute /
subacute / chronic on 5 days of data would be fitting boundaries to noise.

---

## "STOPPED" AND "MISS" ARE DEFINED BY SPOUT CONTACT, NOT BY ATTEMPTING (2026-08-23)

Priya, watching the videos: "its not clear the animal isnt trying in the 'stopped' trials - i still
sometimes see reactive jaw movement."

This qualifies every result that leans on the miss / stopped distinction, so it is recorded before
the distinction gets used further.

### What the labels actually measure

A "lick" in this pipeline is a detected event on `lick_analog` -- **tongue contact with the spout**,
double-threshold with a lockout and the 40 ms ILI floor. `engagement_gate` sees only `responded`,
which is "a detected lick inside the response window". Therefore:

    MISS-while-working   no spout CONTACT on this trial, while still contacting on others
    STOPPED              no spout CONTACT for a sustained, non-recovering run

Neither is a statement about attempting. A jaw movement, a tongue protrusion that falls short, a
mistimed reach -- all produce zero lick events and are indistinguishable from lying still.

### Why that matters most exactly where the deficit is

If the lesion impairs REACHING the contraversive spout, "failed to contact" is the deficit's
signature rather than evidence about intention. The behavioural readout and the thing being explained
are then the same variable measured once.

Specifically qualified:

- **STOPPED IS NOT A "NOT TRYING" CONTROL.** It was added on 2026-08-23 to separate two readings of
  the miss result -- changed in miss but intact in stopped would mean the change is specific to
  attempted-and-failed trials. That inference does not hold if stopped trials contain attempts, and
  a null there is ambiguous rather than informative.
- **The 22% of miss-class position axes with NO coherent representation** may be trials with a real
  attempt and an intact plan, not trials where no plan formed.
- **The engagement gate's own justification** -- reward is auto-held after a miss run, so a sated
  animal's late misses are disengagement rather than spatial inaccuracy -- rests on the same
  assumption that non-response means non-attempt.

### The fix, and why it is now foundational rather than an enhancement

Jaw and tongue movement are visible on the Blackfly video, and the DAQ-camera alignment templates
already exist (`wfield_local/camera_sync.py`, one per cam per date). With movement onset taken from
VIDEO instead of contact from the spout sensor, the miss class splits into two that are currently
fused:

    attempted, no contact   -> a motor / reaching deficit
    no attempt              -> a plan or motivation deficit

Those have opposite predictions for the pre-cue axis, and no DAQ-only analysis can separate them.
This was previously proposed as a way to measure tongue KINEMATICS better (decoupling "which spout
was cued" from "how the tongue actually moved"). It is more than that: the current trial taxonomy may
be mislabelled at the class level, and every neural result conditioned on those classes inherits it.

### What is NOT affected
The `poststroke_lick` class is unaffected -- those trials have a contact by definition. Results
comparing pre-stroke lick with post-stroke lick (the same-class control) do not depend on this at
all, which is a further reason to prefer that comparison over the cross-class one wherever both are
available.

---

## HOW THE POSITION CODE IS REDISTRIBUTED — about half of each changed axis is new structure (2026-08-23)

Priya: "HOW is the code redistributed?" Everything up to this point established only that a
post-stroke axis was RELIABLE and at a DIFFERENT ANGLE from its pre-stroke counterpart. That is
*whether*, not *how* — it says nothing about where the code went.

`position_axes.decompose()` answers it by expressing each changed post-stroke axis in a coordinate
system of pre-stroke references: every other position axis, the close-vs-far axis, and the engagement
axis. **The residual outside the span of all of them is the load-bearing number** — the references
are not orthogonal to each other (position axes share structure), so the individual cosines overlap
and must never be summed, whereas the fraction lying outside their span is well defined by least
squares.

### The result, pooled over post-stroke sessions, pre-cue, 2-session blocks

| class | LOST | REDIST | modest | preserved | median reliability | median residual |
|---|---|---|---|---|---|---|
| **poststroke_lick** | **0** | 19 | 16 | 6 | **+0.69** | **0.56** |
| poststroke_miss_working | 14 | 11 | 2 | 0 | +0.40 | 0.67 |
| poststroke_stopped | 27 | 7 | 1 | 0 | +0.30 | 0.75 |

**On trials with a spout contact, nothing is lost** — zero cells below the no-axis bar, and 35 of 41
interpretable axes changed. This is the one row that depends on none of the day's caveats, because
`poststroke_lick` trials have a contact by definition (see the entry on MISS/STOPPED being defined by
contact rather than by attempting).

**About half of each changed axis is structure with no pre-stroke counterpart.** Median residual 0.56
for the lick class. Not a rotation onto some other existing position axis — new directions.

**The interpretable half is not random.** Post-stroke `close_R|close_L` aligns with pre-stroke
`close_R|close_L` (+0.76 in PS94, +0.68 in PS95): the axis KEEPS ITS OWN IDENTITY while acquiring new
structure. The more interesting cases are cross-ring — PS95's post-stroke `close_center|close_L`
aligning with the pre-stroke `far_R|far_center` axis (+0.76), a close-ring contrast coming to
resemble a far-ring one.

So "redistributed" resolves into three parts: **partly preserved in identity, partly moved onto other
positions' pre-stroke axes, and about half into structure that did not previously exist.**

### The caveat that limits the residual, stated because it is the obvious objection

Residuals rise monotonically across the three classes — 0.56, 0.67, 0.75 — exactly tracking FALLING
reliability (+0.69, +0.40, +0.30). Noise inflates a residual, since an axis fitted through noise is
orthogonal to everything. So 0.56 is an UPPER BOUND on "genuinely new structure" even in the best
class, and the miss and stopped residuals are uninterpretable on their own. The right next step, if
this matters, is a residual computed on split halves of PRE-STROKE lick at matched n -- the same
noise-floor logic used for the cosines -- which would say how much residual pure sampling noise
produces at each reliability. That has NOT been done.

### Two other results from the same run
- **`pool 2` bought far less than trial count predicts.** Spearman-Brown gives +0.47 -> +0.64 at
  double n; observed +0.52, about 30% of the expected gain. The two days are NOT exchangeable — the
  axis moves between sessions. Pooling therefore costs reliability as well as time resolution, and
  any epoch-based analysis will be weaker than a naive trial-count argument suggests.
- **LOST behaves oppositely in the two classes as trials accumulate.** Lick: 12% -> 9% -> 0% (it was
  power). Miss: 7% -> 12% -> 22% (adding data confirms the absence rather than revealing an axis).
  Under the contact/attempt ambiguity this cannot be read as "no plan formed" — only as "no coherent
  representation on trials with no contact".

## THE PLAN IS THERE WHEN THE ANIMAL IS TRYING: MISS-WHILE-WORKING vs STOPPED (2026-08-23)

Priya asked whether the no-lick analysis could be done per position using the trials where the
animal is still WORKING rather than the ones where it has stopped. It can, the machinery already
existed (`position_coding_directions` stores every value per position, per session, per class), and
the contrast had never been drawn side by side. `miss_vs_stopped` now does, as G6b.

WHY THIS IS NOT A REFINEMENT OF G6 BUT A DIFFERENT QUESTION. G6 applies a frozen decoder to no-lick
trials at impaired positions and POOLS the two failure modes. They are not one phenomenon:
MISS-WHILE-WORKING is position-specific and 34-44% far_R; STOPPED is the animal having quit and is
position-GENERAL (response ~0 everywhere, close included). They differ in position composition by a
total variation of 0.31-0.65 and ENL activity carries position, so pooling them compares the spout
rather than the state -- which is what produced a spurious PS95 effect on the first pass.

### The result: miss retains the code, stopped does not -- and far_L says it is not an artefact

ENL window, value = that position's own pre-stroke pole (1.0), 0 = no code.

| animal | position | MISS-WHILE-WORKING, per session | STOPPED |
|---|---|---|---|
| PS92 | far_center | +2.01, +1.78, +1.01, +1.78, +1.63 | +0.78, -0.06, +0.81, +1.22 |
| PS93 | far_center | +0.79, +0.31, +0.63, +0.54, +1.06 | -0.06, +0.13, +0.01 |
| PS94 | far_R | +0.88, +0.30, +0.11, +0.62, +0.62 (4/5 at >=2 SEM, n=33-105) | +0.15, +0.07, -0.02 |

**far_L is the control and shows nothing**: PS94 miss -0.65/-0.77 against stopped ~0; PS95 miss
+2.08 against stopped +2.40. The effect appears at the impaired-but-ATTEMPTED positions and not at
the least affected one, which is what makes it evidence rather than "miss beats stopped everywhere".

PS92 shows NOTHING at far_R (~0) and its strongest effect at far_center. That fits the documented
severity ordering far_R > far_center > far_L: far_R is far enough gone in PS92 that there is no code
left to find, while far_center is impaired-but-attempted -- exactly where a preserved plan should be
visible.

This is the plan-intact / execution-failed signature, in three animals, with an internal control.

### OPEN QUESTION: does stopped-trial coding predict recovery? (Priya's hypothesis, unresolved)

PS95 is the animal this analysis cannot speak to -- it recovered, and a working animal generates few
misses (n 119 -> 24 -> 20 -> 4). Its STOPPED values also EXCEED its miss values (+2.72 and +1.08 at
far_R, +1.92 at far_center), inverting the pattern.

Priya's reading: that inversion may be WHY it recovered. STOPPED would then mean two different
things -- a motivational quit in a structurally intact animal (code preserved) versus a
representational collapse (code gone) -- and only the first predicts recovery.

Lining stopped-code up against far_R response rate:

| animal | far_R response by session | stopped far_R coding | recovered |
|---|---|---|---|
| PS95 | 1% -> **79%** -> 74% -> 38% -> 55% | +2.72, +1.08, +0.25 | yes, fast |
| PS94 | 0, 1, 2, 0, 2% | +0.15, +0.07, -0.02 | no |
| PS92 | 0, 6, 1, 5, 6% | -0.66, +0.44, +0.13, +0.19 | no |
| PS93 | 11, 12, 9, 4% -> **36%** | -0.06, -0.14, +0.08 | yes, late |

PS95 vs PS94 fits. PS92 fits. **PS93 is a clean counterexample**: it recovered far_R from 4% to 36%
on 8/22 with stopped-code flat at zero, and its MISS coding was declining to -0.16 over the same
period -- the code looked worst immediately before the behaviour improved.

The causal arrow is also unsupported as things stand: **PS95's stopped values are from 8/19-8/21,
after its 8/18 recovery**, so they cannot have predicted it. They are equally consistent with
recovery restoring the code.

WHAT WOULD SETTLE IT. (1) Day-1 stopped-code as the predictor -- PS95's 8/17 has 119 miss trials but
no stopped cell, so the decisive datapoint is missing and it is worth checking whether that session
genuinely had no post-quit window or whether the detector did not fire. (2) Whether PS93's 8/22
recovery persists: if it does, a hypothesis resting on stopped-code fails. (3) n=4 animals cannot
establish this either way -- it is a hypothesis to carry, not a result.

---

## THE DRIFT NULL, AND THE THREE WRONG NULLS BEFORE IT (2026-08-23)

Every "the post-stroke axis CHANGED" verdict compares a post-stroke axis against a pre-stroke
reference. What that comparison must be judged against went through three wrong answers before the
right one, and each wrong answer changed which animals had an effect. Recorded in order, because the
reasoning is the reusable part.

**Null 1 — same-day split-half.** Measures sampling noise only. Says nothing about time, so it
cannot distinguish a lesion effect from ordinary day-to-day change.

**Null 2 — session-to-session drift.** Measured directly, and drift is real:

| animal | 1-3 days | 4-10 days | June-August (~60 d) |
|---|---|---|---|
| PS92 | +0.97 | +0.74 | +0.62 |
| PS93 | +1.02 | +0.86 | +0.86 |
| PS94 | +0.89 | +0.77 | +0.68 |
| PS95 | +0.83 | +0.43 | +0.57 |

Axes are stable within 1-3 days (0.83-1.02) and drift over weeks to months. Applying the
June-August rate as the null retired PS92 and PS95 from the result. **That was wrong on two counts**
(Priya: "the drift doesnt nullify the post stroke result - because these arent over the same long
time period"): the pre-to-post interval is 3-9 DAYS (8/14 is the last pre-stroke session, post-stroke
is 8/17-8/23), not 60; and session-to-session comparison OVERSTATES the null anyway, because the
actual analysis pools eleven pre-stroke sessions and pooling averages drift out.

**Null 3 — pooled-vs-held-out, one session.** Structurally right, but unusable in PS92: its median
per-session axis reliability is +0.47, right at the 0.5 gate, so 1-2 cells survived per held-out
session and NONE at the matched 8/14 gap.

**Null 4 — pooled-vs-held-out, TWO sessions.** The one to use. Same operation as the post-stroke
comparison with no lesion in it, at a granularity every animal can support:

    PS93 +0.93    PS94 +0.89    PS95 +0.76 (+0.84 excluding 8/13)    PS92 +0.73 (+0.79)

### 8/13 is a known-degraded session still inside the curated set
PS95 8/13 was recorded single-channel for its first 32 min; the repair left 197/871 cues (23%)
outside the surviving imaging span, and after the coverage fix it reached 0.78 cue-aligned against
~0.90 for that animal (docs/EXPERIMENT_ERRORS.md). In the leave-2-out null its block is **+0.46
[-0.15, +0.60]** against 0.74-0.95 for that animal's other blocks -- half the value, IQR crossing
zero -- and excluding it moves PS95's null from 0.76 to 0.84, which is enough to change that
animal's verdict. PS92's block containing it is also its joint-lowest. PS93 and PS94 are untouched.
`cross_session_exclude` is a COHORT-WIDE date list and cannot express "PS95 8/13 only", which is
exactly the per-animal exclusion left open in the restructure roadmap.

---

## THE POST-STROKE SIDE WAS ALSO POOLED, WHICH HID A RECOVERY (2026-08-23)

Having built per-session storage precisely because pooling averages a recovery and a collapse into
"no change", the null comparison was then made against a POOLED post-stroke median (Priya: "but that
is still pooled post stroke? ps95 was most affected for only 1 session"). Read at the same 2-session
granularity as the null, the four animals have four different trajectories:

| animal | null | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| PS93 | 0.93 | **0.33** (8/8 cells below) | **0.17** (2/2) | **0.11** (1/1) |
| PS94 | 0.89 | **0.72** (5/5) | **0.57** (5/6) | **0.76** (3/4) |
| PS95 | 0.84 | **0.66** (10/10) | -- | 0.93 (0/2) |
| PS92 | 0.79 | 0.87 (1/4) | 0.83 (2/6) | 0.55 (1/1) |

**PS95 has a real effect confined to its first block** -- 10 of 10 cells below the null at
0817+0818, back to 0.93 by 0821. The pooled median of 0.81 averaged that with the recovery and
looked like nothing. **PS93 worsens progressively** (0.33 -> 0.17 -> 0.11). **PS94 dips and partly
recovers.** **PS92 shows nothing** until a single last-block cell.

The same lesion therefore produces recovery in one animal, progression in another and a
dip-and-rebound in a third. Only PS93's and PS94's are sampled well enough to call trajectories; the
0822 blocks are n=1 and PS95's middle block has no interpretable cells at all.

### A "CELL" IS ONE POSITION PAIR, AND CELLS ARE NOT INDEPENDENT
15 pairs come from 6 positions, so `far_R|far_L` and `far_R|close_R` share all their far_R trials.
"10/10 cells below the null" is a CONSISTENCY statement -- the effect is not carried by one odd pair
-- and NOT ten independent tests. Anything needing a real p-value has to permute pre/post labels at
the TRIAL level and recompute, which has not been done.

---

## THE OUTCOME SPLIT CUTS THE TRIALS THE WRONG WAY FOR THE POSITIONS THAT MATTER (2026-08-23)

Priya: "we need to be able to use data from the most affected spout positions!" The reason far_R kept
failing is not reliability and not total trials -- it is that outcome sorts the two sides of a
contrast into DIFFERENT classes:

    LICK class   far_R has 5 (PS94) to 15 (PS92) trials; its partners have hundreds
    MISS class   far_R has 182-399; its partners have few, because the animal still LICKS there
                 -> PS94's far_R|far_L miss cell is n=[116, 24], limited by the GOOD position

Measured: 0/15 far_R cells usable in the lick class for PS92, PS93 and PS94; 0-2 in the miss class.

**The fix is an outcome-blind axis**: pool every trial at a position regardless of what the animal
subsequently did. far_R then has ~370 and far_L ~400. In the PRE-CUE window this is also the more
defensible measure -- nothing has happened yet, so splitting by outcome conditions on the future --
and it sidesteps the contact/attempt ambiguity entirely, since it never reads the outcome label.

Two variants are computed so the choice is visible: `poststroke_all_working` (lick +
miss-while-working, STOPPED excluded, and excluded from the pre-stroke reference too since its
equivalent is the sated tail) and `poststroke_all` (everything) as its comparison. Prefer the former.

**The confound that survives this and cannot be fixed by trial selection**: post-stroke outcome
composition differs BY position -- far_R is mostly misses, close mostly licks -- so an outcome-blind
axis can still pick up outcome-correlated state that happens to correlate with position. Separating
those needs attempts defined independently of spout contact, i.e. the video.

---

## ON- vs OFF-MANIFOLD: WITHIN-MANIFOLD, AND ITS OWN INTERNAL CONTROL SAYS SO (2026-08-23)

The Sadtler/Golub/Batista distinction, since it carries a prognosis: a within-manifold rearrangement
is learnable in hours, an outside-manifold excursion over days or not at all (Oby 2019).

Manifold = PCA on pre-stroke (trial, 0.5 s bin) points in LocaNMF component space, top k for 90% of
variance (k = 14-16 of 87-95).

**Activity stays on it.** Post-stroke VAF 0.832-0.883 against held-out pre-stroke ceilings of
0.894-0.908, every animal within or at the edge of the held-out range. No evidence of an
outside-manifold excursion.

**The coding axes drop slightly -- but NOT in the animals with the lesion effect:**

| animal | axis-in-manifold pre -> post | drop | lesion effect? |
|---|---|---|---|
| PS92 | 0.866 -> 0.788 | 0.078 | none |
| PS95 | 0.883 -> 0.817 | 0.066 | first block only |
| PS94 | 0.880 -> 0.820 | 0.060 | yes |
| PS93 | 0.923 -> 0.867 | 0.056 | yes |

The two animals with the WEAKEST lesion effect show the LARGEST drops. That is the opposite of what
a lesion-driven change predicts, so the ~0.06-0.08 decline is a generic pre-versus-post difference
rather than the lesion -- most plausibly the same drift. An earlier draft of this reported "coding
axes move partly out of the manifold post-stroke"; the internal control says do not.

**Two caveats.** VAF is a VARIANCE ratio, so high-variance directions dominate: a coding axis could
leave the manifold entirely and barely move it, because the position code carries little variance
against the global task response. That is why the axis measure exists separately, and why reporting
VAF alone would have flattered the on-manifold conclusion. And the manifold is defined WITHIN the
90-component LocaNMF basis, itself a fixed anatomically constrained reduction -- so "on-manifold" is
a lower bound on any real excursion, not an estimate of it. Sensitivity to the 90% variance target
has NOT been tested.

---

## THE COMPOSITION CONTROL: OUTCOME MIXTURE DOES NOT ROTATE A POSITION AXIS (2026-08-23)

The outcome-blind arm is what finally gives far_R usable cells, and in it every far_R pair reads
CHANGED. That is either the result or it is arithmetic: post-stroke far_R is nearly all misses while
its partners are nearly all licks, and the pre-stroke reference is mostly licks, so the two sides of
the contrast are drawn from different outcome mixtures. `scripts/axis_composition_null.py` builds
that same asymmetry BY HAND inside pre-stroke, where there is no lesion -- position `a` from no-lick
trials, position `b` from lick trials -- and measures how far it moves the outcome-blind axis.

**It does not move it.**

| animal | arm | n pairs | median disatt |
|---|---|---|---|
| PS93 | miss-while-working | 6 | **+1.26** [+1.09, +1.34] |
| PS94 | miss-while-working | 3 | **+1.09** [+1.02, +1.11] |
| PS95 | miss-while-working | 1 | **+0.90** |
| PS95 | disengaged / sated tail | 8 | **+0.98** [+0.86, +1.10] |
| PS92 | either | 0 | not testable |

Values at or above 1.0 are AT CEILING -- indistinguishable from unchanged, and unstable as estimates
(the same >1 instability documented for per-session cells). The reading is that an axis fitted with
one side drawn entirely from no-lick trials is the SAME axis, not a rotated one. Post-stroke far_R
cells sit at +0.23 to −0.06 in the cue window. Composition cannot manufacture that.

**The sated-tail arm is Priya's, and it is deliberately ONE-SIDED** (2026-08-23). The terminal sated
tail is a far larger state difference than a miss, so a null result there is conservative: if
satiety does not rotate the axis, an outcome difference cannot. A positive result would have been
inconclusive rather than a refutation, because satiety is not the post-stroke miss state. It came
back +0.98, so the conservative reading applies.

**The one exception is the headline pair.** PS95's `far_R|far_L` in the sated-tail arm is +0.52 --
the only cell below 0.8 in the entire control, and the far lateral axis is exactly the contrast the
post-stroke story rests on. So the far lateral axis IS the most state-sensitive of the fifteen. It
still does not reach the post-stroke values (PS94 −0.06, PS93 +0.23), but any claim resting on
far_R|far_L alone should carry this number with it.

**Why PS92 could not be tested is not what it looked like.** An earlier reading of the first run --
"pre-stroke no-lick is overwhelmingly the sated tail" -- was WRONG, and the per-position census
added to the failure branch is what showed it:

| animal | pre-stroke no-lick | miss-while-working | sated tail |
|---|---|---|---|
| PS92 | 132 | **126** | 6 |
| PS93 | 530 | **492** | 38 |
| PS94 | 475 | 227 | 248 |
| PS95 | 778 | 391 | 387 |

PS92 and PS93 are almost ALL miss-while-working; PS94 and PS95 split about evenly. PS92 failed for
total n -- 132 no-lick trials across eleven sessions, because the animal almost never failed -- and
the cells fell to the reliability gate, not the count gate. That also explains the reciprocal
pattern: the two animals with a large sated tail (PS94, PS95) are the only ones where the tail arm
could be run at all.

### CONSEQUENCE
The outcome-blind arm is a valid measurement, not an artefact of trial mixture, and the far_R
verdicts stand. What the control does NOT license: the residual worry is not composition but
MOVEMENT. In the cue window a pre-stroke far_R trial contains a lick and a post-stroke one usually
does not, and the control above is built in the pre-cue window, which is lick-free on both sides by
construction. So the pre-cue far_R result is controlled; the sharper cue-window far_R result is not,
and its extra sharpness is exactly what either explanation predicts.

---

## WHAT THE OUTCOME-BLIND ARM SHOWS, AND WHERE IT IS AND IS NOT POSITION-SPECIFIC (2026-08-23)

Pre-cue (disattenuated median, cells with reliability >= 0.5, against each animal's matched null):

| animal | null | pooled | far_R pairs | non-far_R | per block |
|---|---|---|---|---|---|
| PS92 | 0.79 | +0.77 (11/15 below) | **+0.35** | +0.78 | 0.76 / 0.68 / 0.80 |
| PS93 | 0.93 | +0.39 (11/11) | +0.25 | +0.41 | 0.26 / 0.35 / 0.21 |
| PS94 | 0.89 | +0.60 (12/14) | +0.54 | +0.60 | 0.44 / 0.53 / 0.53 |
| PS95 | 0.84 | +0.75 (12/13) | +0.74 | +0.75 | **0.59 -> 0.96** |

Cue:

| animal | pooled | far_R pairs | non-far_R | per block |
|---|---|---|---|---|
| PS92 | +0.71 | **+0.27** | +0.89 | 0.87 / 0.71 / 0.51 |
| PS93 | +0.45 | **+0.23** | +0.52 | 0.39 / 0.35 / 0.87 |
| PS94 | +0.35 | **−0.06** | +0.50 | **−0.12 -> 0.38 -> 0.60** |
| PS95 | +0.74 | +0.64 | +0.75 | 0.69 / 0.70 / 0.64 |

**Two recoveries, found independently.** PS95 pre-cue block 1 is 0.59 with 11/12 cells below the
null and 0.96 with 0/3 by 0821 -- the same recovery the LICK class showed, now reproduced in an arm
built from different trials. PS94's cue-aligned first block is −0.12 (not rotated: unrelated) and
climbs to +0.60 by 0821+0823. Cross-arm reproduction is the strongest confirmation available at n=4.

**POSITION-SPECIFICITY IS PRESENT IN CUE AND MOSTLY ABSENT IN PRE-CUE.** In the cue window three of
four animals separate far_R (+0.27, +0.23, −0.06) from everything else (+0.89, +0.52, +0.50). In the
pre-cue window only PS92 does (+0.35 vs +0.78); in PS93, PS94 and PS95 every pair changed by about
the same amount. **A uniform shift across all fifteen pairs is what a global change looks like, not
a lateralised lesion effect** -- and the pre-cue window is the CONTROLLED one. The honest statement
is therefore narrower than the cue table suggests:

    pre-cue   a broad change in the pre-cue state, position-specific only in PS92
    cue       a far_R-specific collapse -- in the window where the movement confound is largest

Do not merge the two into "the far_R code is lost". The windows disagree, and they disagree in the
direction that the movement confound predicts.
