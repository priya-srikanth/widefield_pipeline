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

---

## PER SPOUT POSITION, PER WINDOW, ACROSS POST-STROKE TIME (2026-08-24)

Priya asked for the whole thing laid out at once: if and how coding changed, per position, per
window, per animal, over post-stroke days. The axis measure is PAIRWISE, so a per-position number is
the **median disattenuated value over the five pairs involving that position** -- a position reads
low when its contrasts against the others have moved, whatever the partner. Outcome-blind arm
(`poststroke_all_working`), 2-session blocks, cells gated as everywhere else (both reliabilities
>= 0.5, ratios > 1 dropped as at-ceiling). A BLANK IS "NOT MEASURABLE", NOT "UNCHANGED" -- the pre-cue
window loses whole blocks this way and the lick window loses none, which is a power difference, not
a result. Regenerate with `scripts/position_by_position.py`.

### Pooled over post-stroke sessions

| window | animal | null | far_R | far_center | far_L | close_R | close_center | close_L |
|---|---|---|---|---|---|---|---|---|
| pre-cue | PS92 | 0.79 | **+0.35** | +0.77 | +0.58 | +0.88 | +0.78 | +0.77 |
| | PS93 | 0.93 | +0.25 | +0.45 | +0.49 | +0.43 | **−0.01** | +0.39 |
| | PS94 | 0.89 | +0.54 | +0.53 | +0.64 | +0.58 | +0.58 | +0.78 |
| | PS95 | 0.84 | +0.74 | +0.77 | +0.79 | +0.75 | +0.66 | +0.83 |
| cue | PS92 | | **+0.27** | +0.90 | +0.58 | +0.87 | +0.87 | +0.92 |
| | PS93 | | +0.23 | +0.73 | +0.47 | +0.54 | +0.31 | +0.23 |
| | PS94 | | **−0.06** | +0.32 | +0.43 | +0.57 | +0.43 | +0.44 |
| | PS95 | | +0.64 | +0.53 | +0.76 | +0.77 | +0.76 | +0.75 |
| lick | PS92 | | **+0.23** | +0.91 | +0.74 | +0.90 | +0.84 | +0.92 |
| | PS93 | | **+0.32** | +0.76 | +0.45 | +0.64 | +0.37 | +0.39 |
| | PS94 | | **+0.01** | +0.40 | +0.49 | +0.63 | +0.49 | +0.53 |
| | PS95 | | +0.63 | +0.56 | +0.77 | +0.80 | +0.77 | +0.72 |

### Across post-stroke time -- four different animals, four different stories

**PS92 -- FOCAL, ALL THREE WINDOWS, PERSISTENT.** far_R at +0.35 / +0.27 / +0.23 (pre-cue / cue /
lick) against 0.74-0.92 at every other position, stable across all three blocks in the lick window
(+0.31, +0.20, +0.38) with full cell coverage. far_L is intermediate (+0.58 to +0.74); the rest are
untouched. **This is the cleanest result in the dataset, and it is in the animal that showed NOTHING
in the lick CLASS** -- there its far_R had 15 trials. It exists only because the outcome-blind arm
exists. Its 0822 block shows the OTHER positions declining (close_center +0.45, far_L +0.22 in cue),
which reads as a late session-level problem rather than lesion progression.

**PS93 -- BROAD, NOT LATERALISED, far_R IMPROVING.** Pre-cue every position is 0.25-0.49 against a
0.93 null and the LOWEST is close_center (−0.01), not far_R. In cue and lick far_R is lowest but
close_center and close_L are nearly as low. far_R does improve monotonically in the lick window
(+0.09 -> +0.41 -> +0.66). The 0822 cue values jumping to +0.87 rest on n=1 cell each and are not
recovery. This is the animal whose LICK CLASS was progressive (0.33 -> 0.17 -> 0.11) -- the two
measures disagree about direction, and the class result is the better-controlled one.

**PS94 -- FOCAL POST-CUE; THE NEIGHBOURS RECOVER AND far_R DOES NOT.** far_R is −0.06 (cue) and +0.01
(lick) pooled -- not rotated, UNRELATED -- and per block −0.43 -> 0.00 -> +0.14 (cue), −0.38 -> +0.23
-> +0.18 (lick). Over the same blocks far_center climbs −0.27 -> +0.38 -> +0.77 (cue) and −0.24 ->
+0.58 -> +0.81 (lick), close_R +0.35 -> +0.79. **The dissociation is the finding**: reorganisation
and return at the neighbouring positions, no return at the contraversive one. Pre-cue is flat and
mild (0.40-0.78, no ordering).

**PS95 -- A TRANSIENT GLOBAL PRE-CUE HIT.** Pre-cue block 1 is uniformly depressed (0.41-0.67 at all
six positions) and by 0821 every measurable position is 0.91-0.97. Cue and lick show none of it --
flat 0.53-0.85 with no trend, and far_center rather than far_R is the lowest. Its recovery is real
and reproduces across arms, but it is a PRE-CUE phenomenon and it is not position-specific.

### WHAT THE PATTERN AMOUNTS TO, WITH THE CONFOUND ORDERING ATTACHED

1. **Position-specific change is real in PS92 and PS94**, follows the behavioural severity ordering
   (far_R worst, far_L intermediate, close untouched), and in PS92 it holds in ALL THREE windows
   including the lick-free pre-cue one. That is the one place a position-specific effect cannot be a
   movement artefact.
2. **PS93 and PS95 change broadly, not focally.** Same lesion, different geometry of change.
3. **Recovery, where it happens, is never at far_R.** PS95 recovers globally in pre-cue; PS94
   recovers at far_center and close_R while far_R stays at zero. Nothing shows a contraversive
   position's code returning.
4. **Cue and lick agree closely in every animal** -- expected, since the windows overlap. The lick
   window adds POWER (full coverage where pre-cue loses whole blocks), not an independent test.

CONFOUND ORDERING, which decides how much of each row to believe:

    pre-cue   lick-free on both sides by construction; the composition control was run here
    cue       contains the movement, and post-stroke far_R trials mostly have none
    lick      lick trials start at the FIRST LICK; no-lick trials start at the CUE, because
              `position_axes` passes nolick_ref="cue" -- see the correction below

So the window where position-specificity is strongest is also the most exposed. PS92 surviving in
pre-cue is what makes it the animal to build on; PS94's far_R result is stronger in magnitude but
lives entirely in the two confounded windows.

### CORRECTION: AT THE IMPAIRED POSITIONS THE LICK WINDOW *IS* THE CUE WINDOW (2026-08-24)

Priya asked what the lick analysis actually compares, and checking it retired a claim made an hour
earlier -- that the lick window aligns no-lick trials to an inferred would-be-lick time. It does not.
`position_coding_directions` does that; `position_axes` calls
`features_with_indices(basis, nolick_ref="cue")` unconditionally, so in the lick alignment a LICK
trial's window starts at its first lick and a NO-LICK trial's starts at the CUE.

Post-stroke NO-LICK fraction per position -- the fraction of the sample placed at the cue:

| animal | far_R | far_center | far_L | close_R | close_center | close_L |
|---|---|---|---|---|---|---|
| PS92 | **96%** | 42% | 33% | 14% | 15% | 12% |
| PS93 | **81%** | 28% | 50% | 9% | 17% | 18% |
| PS94 | **97%** | 77% | 37% | 19% | 13% | 9% |
| PS95 | 41% | 35% | 14% | 8% | 6% | 4% |

**At far_R in PS92 and PS94 the lick window is 96-97% cue-aligned -- it is the cue window**, which is
why those columns nearly coincide (+0.23 vs +0.27; +0.01 vs −0.06). The three-window table above must
not be read as three measurements at the impaired positions: cue and lick are ONE measurement there,
and only pre-cue is independent. The claim "PS92's far_R effect replicates in all three windows"
is therefore wrong; it replicates in two, one of which re-runs the other. **PS92's pre-cue result
(+0.35 against 0.77-0.88) is untouched and remains the load-bearing observation.**

The residual misalignment is modest rather than catastrophic: pre-stroke reaction times are
0.137-0.255 s, so the pre-stroke reference sits near cue+0.2 s against post-stroke far_R at cue+0.
But it is graded by severity -- the no-lick fraction IS the impairment -- and the ~4% of far_R trials
that do lick enter ~2.4 s later, making the post-stroke far_R sample a two-mode mixture.

WHY THIS IS NOT SIMPLY FIXED BY SWITCHING TO would_be_lick. That reference DROPS any position with
no engaged trial in a session (the 2026-08-21 fix, because the old session-median fallback misplaced
windows by up to 2.1 s at exactly the impaired positions). PS94 far_R has 13 lick trials across all
post-stroke sessions, so the drop would take back the far_R data the outcome-blind arm exists to
provide. The options are to report the lick window only where the lick fraction is high, or to run
both references and show them side by side. For now: **the cue window is the sound one for the
outcome-blind arm** -- there BOTH arms are cue-referenced, so no misalignment exists at all -- and
the lick column should be read as a power-boosted repeat of it, not as corroboration.

Regenerate the fractions with `scripts/nolick_fraction.py`.

---

## "MISS-WHILE-WORKING" IS A POST-STROKE CATEGORY AND WAS APPLIED TO PRE-STROKE TRIALS (2026-08-24)

Priya: "how are you calling 'miss-while-working' pre-stroke? This was something we defined
post-stroke. It is most relevant for PS93 far_L pre-stroke bc it often had leftward licks without
spout contact."

Correct on both counts. `position_axes` applies the SAME two masks to both phases -- `u_pre &
~not_eng` and `u_pre & not_eng` -- and the post-stroke names came with them. Pre-stroke there is no
motor deficit, so `miss_working` is not "tried and failed" there. What the mask actually selects is
**a trial with NO SPOUT CONTACT that is not part of the terminal disengagement run**, and because
licks are detected BY CONTACT, an OFF-TARGET LICK is indistinguishable from an inattentive trial.
This is the contact-vs-attempt ambiguity already recorded for the post-stroke classes; the entry did
not say it transfers to the pre-stroke side. It does.

### The census says PS93 far_L is real and extreme

`scripts/nocontact_census.py`, pre-stroke no-contact rate per position:

| animal | far_R | far_center | far_L | close_R | close_center | close_L | spread |
|---|---|---|---|---|---|---|---|
| PS92 | 3.7% | 2.9% | **6.2%** | 2.6% | 1.4% | 2.3% | 4.4x |
| **PS93** | 4.3% | 14.1% | **26.8%** (268 trials) | 0.4% | 2.4% | 1.9% | **64x** |
| PS94 | **6.2%** | 4.3% | 3.0% | 3.3% | 3.2% | 2.1% | 2.9x |
| PS95 | 7.3% | 6.4% | **7.7%** | 4.4% | 5.1% | 3.9% | 1.9x |

PS93 has a clean far-ring gradient on the LEFT (far_L 26.8% > far_center 14.1% > far_R 4.3%) against
0.4-2.4% at the close positions. 268 trials concentrated on one position, in the animal with a known
orofacial deficit, is not inattention -- and the behavioural entry above already recorded PS93 far_L
as its exception (RT 0.53 -> 1.03 s, response 0.71 -> 0.60). PS92 leans the same way weakly, PS95 is
near-flat, PS94 runs the OTHER way (far_R highest).

### THE CONSEQUENCE IS NOT THE LABEL, IT IS THE ENGAGEMENT AXIS

`engagement_axis` is `mean(pre-stroke lick) - mean(pre-stroke no-lick)`, and EVERY position axis in
EVERY arm is orthogonalised against it. That is a state correction only if the no-lick trials are
POSITION-NEUTRAL, which was never checked. In PS93, **54% of the no-lick side is far_L and 82% is far
positions**, while the lick side is balanced across all six (724-959 trials each). The axis therefore
carries a far_L-versus-rest component, and orthogonalising against it removes far_L POSITION
structure from all fifteen pairs.

That WOULD be a mechanical candidate for PS93's profile. **It was measured, and it is not the
explanation** -- see the next section. The imbalance is real; the inference from it to a distorted
axis does not go through.

### MEASURED: THE IMBALANCE DOES NOT STEER THE AXIS, AND THE REAL PROBLEM IS BIGGER

`scripts/engagement_axis_balance.py`:

| animal | cos(current, balanced) | cos(current, far_L-vs-rest) |
|---|---|---|
| PS92 | +0.997 | −0.753 |
| PS93 | **+0.958** | +0.306 |
| PS94 | +1.000 | +0.707 |
| PS95 | +0.999 | +0.304 |

Balancing per position leaves the axis essentially where it was, and PS93 -- the animal with the 64x
imbalance -- is LESS far_L-aligned than PS92 or PS94. The reasoning that produced the hypothesis
(counts are lopsided -> the axis must be lopsided) skipped the step where that has to be shown, and
the step fails: each position's own lick-minus-no-lick difference points nearly the same way, so
averaging over positions changes almost nothing.

**What the check found instead was ALREADY IN THIS FILE.** Priya, 2026-08-24: "i think we did
this initially when deciding on this analysis so please check our logs / decisions doc." Correct --
the 2026-08-20/21 construction entry above records `cos(w, engagement axis)` at **0.82 / 0.91 / 0.71
/ 0.52** for PS92/93/94/95 on the ONE-VS-REST directions. The numbers below are the same measurement
on the PAIRWISE axes: same magnitudes, same animal ordering, PS93 highest in both. It was reported
here as a discovery, which it was not. **Check the decisions file before writing a finding into it.**

The original entry also carries two things that the "the projection is destroying position
structure" framing ignored, both measured at the time:

- **Orthogonalisation was validated, not assumed.** The symptom was pre-stroke no-lick projections
  scattering from −2.03 to +1.38 across axes that should read alike; after Gram-Schmidt they
  collapse to 0.16-0.17 **and the pre-stroke lick diagonal IMPROVES rather than degrading**. The
  projection removes engagement contamination without costing position readout.
- **An independent check already exists.** LOGISTIC (`lr`) directions were clean from the start
  (|cos| <= 0.07) because they account for covariance, and were kept for exactly this purpose.
  `dom_orth` agreeing with `lr` is what would show the projection is not distorting anything; they
  are not stored by default (`--methods` defaults to `dom dom_orth`, the lr fits being far too slow
  for a nightly), so the audit needs its own run.

With that context, the measurement below stands as a re-measurement:

| animal | most-aligned pair | abs cos with engagement axis |
|---|---|---|
| PS93 | far_center\|close_center | **0.890** |
| PS92 | far_L\|close_L | 0.809 |
| PS94 | far_R\|far_L | 0.702 |
| PS95 | far_R\|far_center | 0.609 |

At 0.89 the orthogonalisation discards nearly all of that position axis and renormalises a residual
of ~46% of the original magnitude. `orthogonalise` warns that "any position information lying along e
goes with it"; how much was never measured, and for the worst pairs it is MOST OF IT.

The mechanism is the far-heaviness of the no-contact population -- 61% PS94, 62% PS95, 67% PS92, 82%
PS93 -- which makes `lick - no-lick` partly a CLOSE-VS-FAR direction. In PS93 the three
most-aligned pairs are all cross-ring. But because balancing does not change the axis, this is not a
counting artefact to be corrected: the genuine no-contact state lives along the same population
directions as the close-vs-far contrast. There is nothing to remove.

**CONSEQUENCE -- QUALIFIED THE SAME DAY, see the raw-vs-orthogonalised entry below.** The
projection removes a large part of the raw position axis, so any analysis reading a COSINE or a
projection directly (`position_coding_directions`, which is not disattenuated) understates position
structure, and a null there is weak evidence -- the existing note that "a class that stops separating
is ambiguous, not negative" carries more weight than its phrasing suggests. It does NOT follow for
the DISATTENUATED position-axis verdicts: measured, raw and orthogonalised ratios differ by 0.02-0.15
because the reliability in the denominator is computed under the same treatment and absorbs the loss.
The cross-ring part of the prediction did not replicate either. This is the same geometry as the within-ring-safe / cross-ring-unsafe
entry and the one-vs-rest flaw, now quantified against the engagement axis specifically.

The position-balanced option stays in `engagement_axis` (off by default) because it costs nothing and
the check should be repeatable, not because it fixes anything.

`engagement_axis` now accepts per-trial position labels and returns the POSITION-BALANCED axis (the
mean over positions of each position's own lick-minus-no-lick difference), so a position contributing
ten times the no-lick trials no longer contributes ten times the axis. **OFF BY DEFAULT** -- every
result on disk predates it and the two must be compared, not silently swapped.
`scripts/engagement_axis_balance.py` measures the gap.

### WHAT THIS DOES AND DOES NOT TOUCH

- **Untouched:** the `poststroke_lick` arm (contact on both sides), and the matched holdout null,
  which is built entirely from lick trials. PS95's recovery lives in the lick arm.
- **Labelling only:** the outcome-blind reference. Pre-stroke no-contact is 132/~2200 trials in PS92
  and 530/~4500 in PS93, so the reference is ~95% contact trials either way and the numbers do not
  move -- only the description of what they compare.
- **Substantive:** the engagement axis in PS92 and PS93, where 95%/93% of its no-lick side is this
  population, and worst in PS93 where that population is far_L-dominated.
- **A dead assumption:** the `_working` variants exclude the sated tail from BOTH phases on the
  grounds that "the pre-stroke equivalent is the sated tail". PS92 and PS93 have essentially no
  pre-stroke tail (6 and 38 trials), so in those two animals the exclusion does nothing at all.

### AND IT PUTS THE PRE-STROKE SIDE BEHIND THE SAME MISSING MEASUREMENT

Nothing in the DAQ can separate an off-target lick from no attempt: the lick sensor reports CONTACT,
so a tongue that misses produces no signal whatever. The behaviour cameras are the only instrument
that sees it. The video-based movement-onset item was already blocking the post-stroke classes; it
now blocks the pre-stroke reference and the engagement axis as well, which raises it from "the next
analysis" to the thing several current results are waiting on.

---

## RAW vs ORTHOGONALISED: THE PROJECTION BARELY MOVES THE DISATTENUATED RATIO (2026-08-24)

Having found that position axes sit at |cos| 0.61-0.89 to the engagement axis, the obvious conclusion
was that the orthogonalised arms must systematically understate position effects. Priya asked for
both forms to be reported. `position_axes` now computes every arm TWICE in one pass -- `pooled` /
`sessions` (projected) and `pooled_raw` / `sessions_raw` (not) -- **and both nulls**
(`prestroke_null`, `prestroke_null_raw`), because a raw cosine judged against an orthogonalised null
compares two different measurements. `scripts/orth_vs_raw.py` reads them side by side.

### The obvious conclusion was wrong

Pre-cue, outcome-blind arm, per position, each against its own null (which also shifts: PS92
0.80 -> 0.70, PS93 0.88 -> 0.92, PS94 0.91 -> 0.88, PS95 0.81 -> 0.79):

| | far_R | far_center | far_L | close_R | close_center | close_L |
|---|---|---|---|---|---|---|
| PS92 orth | +0.35 | +0.77 | +0.58 | +0.87 | +0.77 | +0.77 |
| PS92 raw | **+0.46** | +0.92 | +0.83 | +0.90 | +0.88 | +0.88 |
| PS93 orth | +0.25 | +0.12 | +0.36 | +0.43 | −0.04 | +0.38 |
| PS93 raw | +0.35 | **−0.07** | +0.56 | +0.82 | **−0.10** | +0.38 |
| PS94 orth | +0.54 | +0.52 | +0.64 | +0.59 | +0.59 | +0.79 |
| PS94 raw | +0.48 | +0.48 | +0.64 | +0.61 | +0.63 | +0.66 |
| PS95 orth | +0.80 | +0.83 | +0.79 | +0.74 | +0.71 | +0.83 |
| PS95 raw | +0.87 | +0.82 | +0.80 | +0.80 | +0.68 | +0.84 |

Median |delta| is 0.02-0.15 nearly everywhere. PS92 rises at all six positions, PS93 partly, PS94
and PS95 essentially not at all. By PAIR TYPE the predicted cross-ring damage is weak at best --
PS92 distance +0.25 / diagonal +0.10 against lateral-centre +0.11; PS93 distance +0.16 / diagonal
+0.13; PS94 and PS95 no ordering.

### WHY, and it should have been predictable

**The disattenuation absorbs it.** The ratio divides by the axis's own split-half reliability
COMPUTED UNDER THE SAME TREATMENT. Projecting a direction out removes signal from the cosine AND from
the reliability, so the two largely cancel. Raw and orthogonalised COSINES differ a great deal; their
disattenuated RATIOS do not. The |cos| 0.61-0.89 finding is still true and still means the projection
is a large intervention on the AXES -- it simply does not propagate to this statistic.

**So the earlier claim needs qualifying**: "the orthogonalised arms systematically understate
position effects, worst for cross-ring pairs, in every animal" holds for raw cosines and for any
analysis reading them directly (`position_coding_directions` projections, which are NOT
disattenuated), and does NOT hold for the disattenuated position-axis verdicts, which are what the
post-stroke story rests on. Those are robust to the choice.

### PS93's "BROAD, UNLATERALISED" SURVIVES -- with a different shape than "broad"

Raw, against a 0.92 null: close_center −0.10 ~ far_center −0.07 < far_R +0.35 ~ close_L +0.38 <
far_L +0.56 < close_R +0.82. **far_R is mid-pack in BOTH treatments**; the projection was not hiding
a lateralised effect, which was the hypothesis this run was built to test.

What is there instead is a CENTRE-POSITION concentration: both centre spouts sit near −0.1 while all
four lateral positions run +0.35 to +0.82, in raw and orth alike. That is a centre-versus-lateral
geometry, not a left-right one, and it is a different claim from "broad". It has NOT been checked
against PS93's behaviour and should be before it is called anything.

**PS92 strengthens.** far_R is the lowest position in raw too (+0.46 against a 0.70 null; next
lowest +0.83), so its specificity is not a product of the projection. Holding in the lick-free
pre-cue window AND in both engagement treatments, PS92 is the cleanest case in the cohort.

### THE PATTERN IN THE ERRORS, RECORDED BECAUSE IT REPEATED THREE TIMES IN ONE DAY

Three mechanisms were proposed from real observations and none survived measurement:

1. PS93's far_L trial imbalance distorts its engagement axis -> balancing moves the axis by +0.958.
2. The projection damages cross-ring pairs most -> no consistent ordering by pair type.
3. The projection systematically understates position effects -> true for cosines, absorbed by the
   disattenuation for the verdicts that matter.

Each observation was correct. What failed each time was the UNTESTED STEP between the observation and
its supposed consequence -- lopsided counts do not imply a lopsided axis; a large angle to the
engagement axis does not imply a large change in a ratio that normalises by reliability. The cost of
checking was one script and one run each; the cost of not checking would have been three wrong
statements in the deck. **Propose the mechanism, then measure it before reporting it as a candidate.**

---

## THE CODING-DIRECTION AUDIT, RUN AT LAST: THE PROJECTION IS A CORRECTION (2026-08-24)

Designed 2026-08-20/21 and never run, because `lr` needs a logistic fit per held-out session per pair
(~330 extra fits per animal-window) and `--methods` defaults to `dom dom_orth`. Priya asked for it.
`scripts/coding_direction_audit.py`, ENL window, pooled per-position per-class means -- what the deck
figures show.

`lr` is the reference: it reaches a near-uncontaminated direction WITHOUT any projection, by
accounting for covariance. So the question is whether projecting moves `dom` toward it.

| animal | dom vs lr | dom_orth vs lr | lr_orth vs lr |
|---|---|---|---|
| PS92 | 0.841 (r +0.67) | **0.128** (r +0.87) | 0.055 |
| PS93 | 0.377 (r +0.44) | **0.127** (r +0.59) | 0.072 |
| PS94 | 0.155 (r +0.44) | **0.122** (r +0.59) | 0.034 |
| PS95 | 0.429 (r +0.28) | **0.188** (r +0.41) | 0.014 |

**Orthogonalising moves `dom` toward `lr` in 4/4 animals**, dramatically in PS92 (0.841 -> 0.128). It
is a correction toward the covariance-aware answer.

**`lr_orth` vs `lr` is 0.014-0.072** -- projecting a direction that was never contaminated barely
changes it. That is the direct refutation of the "the projection removes position structure" worry
raised earlier today: if the engagement axis carried position information, removing it from a clean
direction would move that direction. It does not.

**The shift by class lands where the rule says it should.** `dom -> dom_orth` median shift:

    prestroke_lick            0.00  0.00  0.00  0.00   (the class the direction is fitted on)
    prestroke_nolick          1.69  0.62  0.35  0.82   (the class it was introduced for)
    poststroke_miss_working   0.95  0.54  0.17  0.61
    poststroke_stopped        0.78  0.67  0.15  0.46
    poststroke_lick           0.29  0.15  0.04  0.05

The correction acts on the no-lick classes and leaves the lick classes almost untouched -- exactly
the disposition recorded in 2026-08-20/21 ("raw `dom` must not be used for the no-lick classes").

### THE LIMIT THE AUDIT ALSO SHOWS
Residual `dom_orth` vs `lr` disagreement is **0.12-0.19** -- 12-19% of the pole separation -- and the
correlations are only +0.41 to +0.87, worst in PS95 (+0.41 AFTER orthogonalising). So "validated"
means the projection does what it claims, NOT that the two methods give the same numbers. Any claim
resting on a fine ordering between adjacent cells should be checked in both. Only the ENL window has
been audited; cue and lick have not.

---

## PS93 RE-READ IN RAW: FAR_R-SPECIFIC IN THE POWERED WINDOWS, AND THE PRE-CUE CELL COUNTS WERE THIN

far_R, orthogonalised -> raw, per window:

| animal | pre-cue | cue | lick |
|---|---|---|---|
| PS92 | +0.35 -> +0.46 | +0.27 -> +0.32 | +0.23 -> **+0.23** |
| PS93 | +0.25 -> +0.35 | +0.22 -> **−0.00** | +0.33 -> **−0.04** |
| PS94 | +0.54 -> +0.48 | −0.06 -> +0.02 | +0.01 -> +0.04 |
| PS95 | +0.80 -> +0.87 | +0.64 -> +0.48 | +0.63 -> +0.46 |

PS92's lick-window far_R is IDENTICAL between treatments (+0.23 both) -- there the projection does
nothing whatever.

**PS93 is not "broad, unlateralised" in the two well-powered windows.** In raw cue and raw lick,
far_R is lowest by a clear margin (−0.00 and −0.04; next lowest +0.13 and +0.29; everything else
+0.63 to +0.78) and it IMPROVES across blocks in both (cue −0.18 -> +0.11 -> +0.21; lick −0.13 ->
−0.01 -> +0.51). That is a far_R-specific deficit with recovery.

**The centre-position claim made earlier today is withdrawn.** It came from the pre-cue window, whose
PS93 cells number **3 per centre position** against 5 everywhere in cue and lick. At n=3 that is not
a finding, and the counts were printed beside every value specifically so this would be checked.

### WHERE THAT LEAVES PS93 -- an impasse, not a result
The two windows that show the far_R effect are the movement-confounded ones, and PS93's post-stroke
far_R is 81% no-lick, which is exactly the condition that makes them unreliable (the lick window is
then largely cue-aligned; see the correction entry). The controlled window, pre-cue, is too thin to
adjudicate. So PS93 cannot currently be called either way, and it is the same missing measurement as
everything else: attempts defined independently of spout contact.

**PS92 remains the one animal whose far_R effect holds in all three windows, in both engagement
treatments, with full cell coverage.**

---

## THE AUDIT EXTENDED TO CUE AND LICK: ORTHOGONALISING IS RIGHT IN ENL AND WRONG FOR PS92 AFTER THE CUE (2026-08-24)

The ENL result (projection moves `dom` toward `lr` in 4/4 animals) does not generalise. Median
|dom − lr| -> |dom_orth − lr|, all three windows:

| animal | ENL | cue | lick |
|---|---|---|---|
| PS92 | 0.841 -> **0.128** | 0.160 -> **0.192** | 0.202 -> **0.279** |
| PS93 | 0.377 -> 0.127 | 0.279 -> 0.206 | 0.219 -> 0.135 |
| PS94 | 0.155 -> 0.122 | 0.254 -> 0.140 | 0.181 -> 0.112 |
| PS95 | 0.429 -> 0.188 | 0.569 -> 0.244 | 0.499 -> 0.191 |
| | **4/4 toward lr** | 3/4 | 3/4 |

**PS92 moves AWAY from the reference in both post-cue windows**, correlation +0.77 -> +0.69 (cue) and
+0.78 -> +0.62 (lick).

### THE MECHANISM IS IN THE lr_orth COLUMN
Projecting an ALREADY-CLEAN `lr` direction costs 0.014-0.072 in ENL but 0.024-0.116 in cue and lick.
In a window with no movement in it the engagement axis carries almost no position structure, so
removing it is nearly free. After the cue it is a LICKING axis, and removing it takes position-linked
movement structure with it. PS92's plain directions were already the cleanest of the four in those
windows (0.160/0.202 against 0.841 in ENL) -- little contamination to remove, real structure to lose.

The deck's own note called the lick-window projection "deliberately conservative for the lick
classes". Conservative and wrong-signed are different things, and for PS92 it is the second.

### CONSEQUENCE, AND WHAT WAS DELIBERATELY NOT DONE
`_G9_METHOD` shows `dom_orth` in all three windows for all four animals, so **the G9 cue and lick
panels show the WORSE of the two available estimates for PS92**. The note now says so with the
numbers, rather than the method being silently wrong.

It was NOT switched per animal: six G9 panels chosen by different rules are incommensurable with each
other, which is worse than one documented exception. The real decision -- whether cue and lick should
show BOTH variants for every animal -- is left open rather than made by default.

### AND THE LIMIT FOUND IN THE ENL AUDIT STILL APPLIES EVERYWHERE
Residual |dom_orth − lr| is 0.11-0.28 across the three windows with correlations from +0.24 (PS93
cue) to +0.69. The methods are not interchangeable anywhere. Any claim resting on a fine ordering
between adjacent cells has to be checked in both -- which is the ORDERING flag on 13 notes in
`docs/DECK_CLAIM_AUDIT.md`.

---

## G9 SHOWS BOTH VARIANTS WHERE THE CHOICE IS CONTESTED — and twelve slides were missing (2026-08-24)

Priya, after the three-window audit: "let's show both in the deck."

`_G9_METHODS = {"ENL": ("dom_orth",), "cue": ("dom_orth", "dom"), "lick": ("dom_orth", "dom")}`.
Both figure sets already exist -- the nightly's default is `--methods dom dom_orth` -- so this is a
deck-side change with no re-render. Each slide's title now names the variant, and the cue/lick
slides carry the audit numbers in the blurb so a reader can see the size of the disagreement instead
of taking it on trust.

**ENL stays single** because there the audit is 4/4 toward the logistic reference and the plain
direction is badly contaminated (|dom − lr| = 0.841 in PS92). Showing it beside the orthogonalised
one would invite exactly the misreading the projection exists to prevent. Cue and lick are shown
both ways because there the choice is genuinely contested -- PS92 goes the wrong way in both.

**Not switched per animal.** Six panels built by different rules cannot be compared with each other,
which is worse than one documented exception. Showing both everywhere the question is open gets the
same information across without that cost.

**The cohort diagnostics (G9b) stay on the orthogonalised variant alone.** They are arguments about
the geometry of the axes -- how much an axis IS the close-vs-far dimension, and whether that predicts
drift -- and both were measured on the orthogonalised directions. Drawing the plain ones beside them
would put two different measurements under one claim.

### TWELVE BEHAVIOUR SLIDES HAD NEVER APPEARED
The behaviour panel is not method-dependent and its file carries no method in the name
(`coding_engagement_<window>_<animal>.png`), but the loop built every name as
`coding_<kind>_<window>_<method>_<animal>.png`. That never matched, `_f.exists()` skipped it, and
the slide was silently absent -- for all three windows and all four animals -- while the
within-session note called it "THE FIGURE THE WITHIN-SESSION PANEL MUST BE READ AGAINST".

Found only because wiring the second variant meant re-reading the filename construction. **An
`if not path.exists(): continue` is invisible by design**: it cannot distinguish "this figure was
not produced" from "this filename is wrong", and it reports neither. The deck's own
figures-placed/missing counter does not catch it either, because a skipped slide is never counted as
missing.

Slide accounting, measured rather than assumed: the old rule matched 72 figures, the new one matches
132 (+48 plain-direction variants, +12 behaviour), and **nothing that was shown before is dropped**.

---

## TWO FIGURE-LABELLING BUGS FOUND BY READING THE SLIDES (2026-08-24)

Both were found by Priya looking at the deck, not by any check in the pipeline, and neither would
ever have failed a test: the DATA was right in both cases and only the labels lied.

### 1. Every LICK-ONLY crossed-confusion figure was captioned "ALL trials"

`plot_poststroke.fig_confusion_alltrials` hardcoded the post-panel titles and the suptitle as
"ALL trials", while `section_g_figures` calls it ONCE PER ARM and passed the arm into the FILENAME
only. So the G3 slide title said "LICK-ONLY arm" (correct -- it comes from the loop) and the figure
inside it said "POST (frozen, ALL trials)" (wrong). Priya, 2026-08-24: "the matrices are labeled the
same. Which is correct?" The slide title is.

The matrices themselves were always the right ones -- `conf` is indexed `arms[arm]["confusion"]` --
which is what makes this the harder kind of error to catch: nothing about the numbers looks wrong,
and a reader comparing the two G3 slides would conclude the two arms give identical results.

Fixed by giving the function an `arm_name` parameter and passing `arm_name` (already in scope at the
call site, from `ARMS`). The LICK-ONLY suptitle now also says what its missing rows mean: a position
the animal abandoned has NO row in that arm, which is the gap the ALL-trials arm exists to fill, and
an absent row must not be read as a failure to decode.

### 2. `miss_vs_stopped` placed the two classes at the wrong x, in every panel

`fig_miss_vs_stopped` built `x = np.arange(len(s))` SEPARATELY FOR EACH CLASS and called
`set_xticklabels` INSIDE the class loop. Two consequences:

- **The classes were not aligned by session.** The two classes have different session sets in ALL
  TWELVE panels -- miss always has more, because a stopped cell needs a terminal quit period.
  PS94 far_R is miss on 0817-0823 (6 sessions) against stopped on 0818/0819/0820/0823 (4). Drawn at
  0..5 and 0..3, every PS94 miss point sat one session to the LEFT of the stopped point beside it,
  under a figure whose own subtitle promises "same position, SAME SESSION, two failure modes side by
  side".
- **The last class drawn owned the tick labels.** STOPPED is drawn second, so its 4 ticks labelled
  the axis and the 2 extra miss points fell beyond them with no date at all -- which is what Priya
  spotted ("why are there data points that are not labeled with any date?").

Fixed: a shared ordered date list per panel, `x = dates.index(d)` per point, ticks set once outside
the class loop, xlim pinned to the shared range.

**The quoted numbers survive.** Every value in the deck note and in the "THE PLAN IS THERE WHEN THE
ANIMAL IS TRYING" entry was read from `coding_direction.json`, not off the figure, and re-checking
them against the JSON they match exactly (PS92 far_center miss +2.01/+1.78/+1.01/+1.78/+1.63,
stopped +0.78/−0.06/+0.81/+1.22). What was wrong was the picture, and any conclusion drawn by EYE
from the paired trajectories -- particularly PS94, where the offset is a full session.

**One stale count did surface**: the deck says "PS94 far_R miss ~+0.6 on four of five sessions ...
against stopped +0.15/+0.07/−0.02". There are now SIX miss sessions (0.88, 0.30, 0.11, 0.62, 0.62,
0.42) and FOUR stopped (+0.15, +0.07, −0.02, +0.04) -- 0823 was added after the sentence was
written. A `DECK_CLAIM_AUDIT.md` case in the wild.

### THE PATTERN, WHICH IS THE REUSABLE PART
Both bugs are the same shape: a value that varies (the arm, the session) reaching the FILENAME or the
DATA but not the LABEL. Neither can fail a test, neither trips the deck's figures-placed/missing
counter, and both produce a figure that looks entirely reasonable. The only detector that worked was
a person reading the slide and asking why two things that should differ looked the same. Worth
assuming there are more: anything that draws per-arm, per-class or per-window figures from a shared
plotting function is a candidate.

---

## THE FIGURE-LABEL AUDIT: TWO MORE INSTANCES, AND A CHECKER THAT KEPT REPEATING THE BUG (2026-08-24)

After the two label bugs found by reading slides, the obvious question was how many more there are.
`scripts/figure_label_audit.py` answers it statically: for every `fig*` function taking a parameter
that DISCRIMINATES one figure from another (align, arm, meth, cls, window, phase), does that
parameter reach a title?

**Two genuine findings, both now fixed:**

- `joint_xsession.fig_basis_health` is called once per alignment and writes
  `joint_basis_health_{align}.png`, but its title never named the window -- and the span it plots is
  computed ON the aligned window, so the cue and pre-cue figures are different measurements. The deck
  shows the PRE-CUE one alone and its slide title did not say so either. Both now do.
- `position_coding_directions.figure_engagement` put `disp` in the FILENAME and not the caption, so
  its ENL/cue/lick files were captioned identically. They are not the same figure: each alignment
  keeps a different trial set (pre-cue drops trials with no lick-free window, lick drops positions
  with no engaged trial), so the response rates can differ.

**Everything else labels itself correctly** -- 18/18 after the fixes, including `fig_grid`,
`fig_matched` and `fig_similarity`, which were the ones I guessed at in conversation and would have
"verified" by assertion.

### THE CHECKER HAD THE BUG IT WAS WRITTEN TO FIND, THREE TIMES
Each version missed one more level of indirection than the last, and each time the miss LOOKED like a
result:

1. Scanned only the title call site. `ttl = f"...{arm_name}..."; ax.set_title(f"{an} - {ttl}")`
   reported the function it was written to catch as still broken. **False positive.**
2. Resolved locals with `if var in txt` -- a SUBSTRING test. A local named `p`, assigned
   `f"joint_basis_health_{align}.png"`, matched the letter "p" in any prose title and folded the
   FILENAME's `align` into it, turning the one true positive into a pass. **False negative**, and
   the more dangerous direction.
3. Handled only `Name` assignment targets, so `disp, R = dict(ALIGNS)[align], res[...]` -- a TUPLE
   target -- never marked `disp` as derived from `align`, and eleven correctly-labelled functions
   were flagged. **False positives**, which would have wasted an hour of "fixing" working code.

The pattern in all three is the pattern in the bugs themselves: a value reaching its destination by
one more hop than whatever the check accounted for. Worth stating plainly -- **a checker for this bug
class is itself unusually prone to this bug class**, and its output has to be spot-checked by hand
against at least one known-good and one known-bad case before any of it is believed.

Final rule that removed the last false positive: **a parameter the body never reads cannot
discriminate anything.** `figure_engagement(res, out, align, meth)` takes `meth` only to match the
uniform signature its caller dispatches on; the behaviour panel is method-independent, which is why
its filename carries no method either.

---

## G9 PAIRWISE IS ALREADY PER-POSITION; THE BLURB DESCRIBED THE CONTRAST AND NOT THE LAYOUT

Priya, 2026-08-24: "why isn't each position on its own graph (eg for trials truly at far R, show
far R trials)". It is -- `figure_pairwise` draws a 2x3 grid with `set_title(f"trials truly at {A}")`,
one panel per position, and has since it was written.

The misreading is the slide's fault. The `direction` blurb opens "One panel per spout position, MOST
IMPAIRED first"; the `pairwise` blurb opened "Each contrast is A vs B ALONE" and described the
CONTRAST while never saying what a PANEL is -- and since the x-tick labels inside each panel are
position names, "positions on the x-axis, not split per position" is the natural reading. The tag now
says ONE PANEL PER POSITION and the blurb opens by stating that the panel is the trials' TRUE
position and the x-axis is the PARTNER position.

**A figure being right does not make the slide right.** This one cost a round trip on a figure that
never needed changing, and the fix was eleven words of layout description that the neighbouring
blurb already had.

### FOLLOW-UP: "why isn't there a far_R column in the far_R panel"
Because an axis is a contrast between TWO positions and no far_R-vs-far_R axis exists
(`others = [B for B in BY_SEVERITY if B != A]`). The panel's own position is the SCALE, not a
column: each pair's axis is anchored 1 = pre-stroke lick at the panel's position, 0 = pre-stroke lick
at the partner, so the flat line at 1.0 IS far_R and every class reads as its distance from it. The
FIGURE says exactly this in its suptitle; the SLIDE did not -- the same gap as the layout one above,
the explanation existing in the place the reader was not looking.

A self column would have to be a ONE-VS-REST value, and putting one on a line whose other five points
are pairwise contrasts mixes two axis constructions in a single series. The one-vs-rest view already
answers that question: the G9 time-course panels and the diagonal of the cross-position matrix. Now
stated on the slide so the absence reads as a decision rather than an omission.

---

## WHAT DOES POST-STROKE ACTIVITY AT AN IMPAIRED POSITION LOOK LIKE? (2026-08-24)

Priya asked the question the cross-position matrices were built for and had never been read as a
table: for far_R trials in an animal that cannot lick far_R, does ENL activity resemble pre-stroke
far_R, or pre-stroke far_L? And she added the observation that decides how to read it -- on far_R
miss trials she often sees **incomplete LEFTWARD or CENTRAL licks after the cue**.

**THE TWO WINDOWS SEPARATE THE TWO ANSWERS.** ENL is lick-free by construction, so a match there is
the PLAN and cannot be a movement artefact. The cue window contains the attempted movement, so a
far_L match there is cortex following the EXECUTED direction. `scripts/best_match.py` reads
`cross_matrix` / `cross_by_session` from `coding_direction.json` (G9 slides 277-280 ENL, 301-304 cue)
and reports, per class, the argmax column and the shift from the pre-stroke row.

### far_R MISS-WHILE-WORKING, pooled

| animal | resp | ENL own | ENL best | cue own | cue best |
|---|---|---|---|---|---|
| PS92 | 0.12 | +0.05 | **far_L +1.27** | −0.20 | close_L +0.53 |
| PS93 | 0.23 | +0.20 | **far_L +0.59** | +0.03 | **far_L +0.54** |
| PS94 | 0.02 | **+0.57 (own)** | far_R | **−0.46** | close_L +0.56 |
| PS95 | 0.48 | +0.54 | far_center +0.61 | +0.46 | **far_R (own)** |

**PS94 is the clean intention/execution dissociation Priya predicted**: the plan is still far_R in the
lick-free window (+0.57, own-best) while the cue window collapses to −0.46 and matches close_L /
close_center / far_L -- exactly "the tongue went left or centre". Per session its ENL even returns to
own-best on 0820 and 0821 (+0.62 each) while the cue window never does.

**PS92 is NOT that pattern.** Its far_R miss trials look like far_L ALREADY IN ENL (+1.27 against
+0.05 own), on all five sessions (+0.87 to +2.03). Before the cue, with no movement in the window,
the pattern is leftward. That is a substituted plan, not a deviated execution.

**PS93 is leftward in both windows** (ENL +0.59, cue +0.54) and then loses even that: the per-session
best match runs far_L, far_L, far_L (0818-0820) then close_center (0821, 0822). Drift toward the
central/close pattern, i.e. loss of position specificity rather than recovery.

**PS95, the recovering animal, keeps far_R as far_R** in both windows, on essentially every session
(lick class: ENL +0.83/+0.70/+1.04/+0.69, cue +1.01/+0.73/+0.86/+0.57).

### STOPPED IS A STATE, AND THE TABLE SHOWS IT
In PS93, PS94 and PS95 the stopped class moves toward **close_center** at every impaired position
(shift +0.77 to +1.80), in both windows. That is the position-GENERAL signature already documented
for stopped, arriving independently here. PS92's stopped still reads far_L, on n=74. So the miss/
stopped split is not a refinement of the same measurement: miss carries position-specific structure,
stopped carries a state that looks like the central spout in everyone.

### Q2 -- AT POSITIONS THE ANIMAL STILL LICKS, IS MOTOR ENCODING INTACT? (lick window, own column)

| animal | preserved positions | reading |
|---|---|---|
| PS92 | far_center +1.69, close_R +1.01, close_L +0.91, close_center +0.68, far_L +0.67 | own-best everywhere, at or above pole, stable across sessions -- **intact** |
| PS93 | far_center +0.36 (best close_center), close_center +0.29 (best close_R), close_L +0.35 (best close_center) | **THREE OF FIVE licking positions no longer match themselves best** -- genuine motor-encoding change where behaviour is preserved |
| PS94 | close_R +0.55, close_L +0.69, close_center +0.31 -- rising across sessions (close_R 0.37 -> 0.88, close_center −0.13 -> 0.70, far_L 0.25 -> 0.61) | own-best but reduced, **recovering** |
| PS95 | close_R +1.25, far_L +1.10, close_center +1.04, close_L +0.87 | own-best, at pole, **intact**; a uniform dip on 0823 only |

**PS93 is the important row.** It is the one animal whose lick-related activity has meaningfully
shifted AT POSITIONS IT STILL PERFORMS -- and that dissociates neural change from behavioural
deficit, since those positions have response rates of 0.60-0.70.

### WHAT THESE NUMBERS CANNOT SETTLE
- **The cue-window match is confounded by movement presence.** A post-stroke miss trial has no
  contact where its pre-stroke reference had one. The confound is not total -- pre-stroke far_L is
  itself a LICK trial, so "matches far_L" is matching a lick pattern rather than matching absence --
  but the ENL rows are the ones that carry weight, and they are the lick-free ones.
- **MISS is defined by spout CONTACT.** An incomplete leftward lick produces no signal and is scored
  a miss, which is exactly the population Priya is describing; the DAQ cannot confirm the executed
  direction on any single trial. **Video-based tongue tracking would turn this from a population
  argument into a per-trial one** -- and it is the same missing measurement everything else waits on.
- Raw argmax is reported beside the SHIFT from the pre-stroke row because positions are intrinsically
  similar before any lesion; a raw winner that is flat in the shift was always that similar.
- PS92's cue rows are the least trustworthy in the table: the 2026-08-24 audit found the
  orthogonalised directions move AWAY from the covariance-aware reference for PS92 in cue and lick.

### AND 144 FIGURES THAT ANSWER THIS WERE NEVER IN THE DECK
`coding_{crosssess,pairsess}_{window}_{method}_{class}_{animal}.png` -- the per-session, per-class
versions of both matrices, 3 windows x 2 methods x 3 classes x 4 animals -- are rendered every night
by `position_coding_directions` and were referenced by no slide. They are the recovery view this
question needs. Now placed as **G9c**, one method, three classes.

---

## VERDICT: POST-LICK IS ABSENT FROM THE POST-STROKE ANALYSIS, AND WAS NOT "FILLED IN" (2026-08-24)

Priya asked directly whether I had filled in the post-lick analysis. **I did not.** What I did was
establish WHY it is missing and finish a sentence in the deck that had been promising the reason
without giving it. Stated plainly so nobody later reads the grant figures as though the gap were
closed:

| where | post-lick present? |
|---|---|
| PRE-STROKE cross-session decoding (grant figs 2, 2b, 4) | **YES** -- computed 2026-08-24 via `joint_xsession --align lick`; only cue and precue existed before |
| POST-STROKE frozen analysis, ALL-trials arm (`section_g.json`) | **NO, and it cannot be** |
| POST-STROKE frozen analysis, LICK-ONLY arm | **ALREADY COMPUTED -- see the correction below** |
| Coding directions (G9, grant fig 3a) | YES -- with no-lick classes at an INFERRED would-be-lick time |

**Why the all-trials arm cannot have it.** That arm exists to include trials with NO detected lick,
and a lick-aligned window cannot be defined for a trial that has no lick. At the impaired positions
that is most of the trials -- which is exactly where the question is. Checked, not assumed:
`post-lick` appears in 0 of 24 session records, while `pre-cue` and `post-cue` appear in all 24.

### CORRECTION, SAME DAY: THE LICK-ONLY ARM HAD IT ALL ALONG
Priya then asked me to build the post-lick frozen decoder. There was nothing to build. `post-lick`
and `post-lick within-session` are present in **all 24 session records** of `arms["lickonly"]`, with
permutation nulls, per-position recall and the pre-stroke band (PS94 8/20: accuracy 0.883 against a
0.954-0.993 band, 4 positions, chance 0.25). The guard in `poststroke_section_g` is
`if align == "lick" and arm_all: continue` -- it skips post-lick for the ALL-TRIALS arm only, and
has always computed it for lick-only.

**It is also already in the deck**, on slide 154 (G2, LICK-ONLY arm): `section_g_figures` builds the
matched dict from `("post-cue", "post-lick", "pre-cue")` and every session supplies all three.

**HOW I GOT IT WRONG, and it is the same mistake twice in one day.** I checked `arms["all"]`, found
post-lick in 0 of 24 records, and generalised from one arm to the whole analysis -- exactly the shape
of the earlier error where checking one deck builder reported 700 files as unreferenced. An absence
established in one place is an absence in that place. The verdict above stands for the all-trials
arm, where the construction argument is real; it was never true of the analysis.

WHAT WAS ACTUALLY MISSING was post-lick in the GRANT figure, which had two rows because I had
believed my own verdict. It now has three, drawn from the lick-only arm with a PER-SESSION chance
step -- that arm scores each session on its own preserved positions, so chance is 0.25 on a 4-way
day and 0.167 on a 6-way one, and one flat 1/6 line would make a 4-way 0.5 read as twice chance when
it is exactly twice a different chance. Those panels are not comparable across sessions, which the
figure says.

Worth noting from the new row: in the post-lick window the FROZEN decoder often beats the
within-session one (PS92 day 1: 0.83 vs 0.75; PS94 days 5-7). That is not a paradox -- the frozen
decoder is trained on eleven pooled sessions and the within-session one on a single session with
block CV, so the frozen model has far more data and no CV penalty. It is a reminder that the two
lines are not on a common footing and only their SHAPE over days is comparable.

**A deck sentence was shipping unfinished.** The G2 blurb read "...All six positions, chance 1/6 on
every panel. POST-LICK IS ABSENT HERE BY " and stopped -- an incomplete clause on slide 153,
promising a reason it never gave. Found only because the grant figures hit the same absence in the
data and had to work out the answer independently. Now completed with the construction argument.

---

## GRANT FIGURES: WHAT WAS BUILT, AND THE FIVE THINGS THE FIRST DRAFTS GOT WRONG (2026-08-24)

`wfield_local/grant_figures.py` -> `<labcams>/grant_figures/`. Deliberately not deck figures: the
deck carries every caveat on the slide, which is right there and wrong in a grant, so each figure
makes one point and the caveats live in the module docstring and here.

  1   behaviour, six positions per animal vs DAYS FROM LESION, Wilson CIs, June collapsed
  1b  the same with the WHOLE pre-stroke baseline as one mean +/- SEM point per position (Priya)
  2   pre-stroke cross-session decoding per animal, ENL/cue/lick
  2b  the cohort version -- **ENL 0.55, post-cue 0.89, post-lick 0.93**, chance 0.17 (Priya)
  3a  coding retained, impaired vs preserved positions, three windows, over days
  3b  frozen vs within-session decoding over days, with the pre-stroke band
  4   mean pre-stroke LOSO confusion, 2x2 animals, one file per window (Priya) --
      diagonals 0.89-0.95 in the lick window

### The errors, all found by looking at the rendered figure rather than by reasoning
1. **`config.stroke_date` returns MMDD, not YYYYMMDD.** Slicing it as the longer form gave an empty
   string and an int() crash.
2. **Fig 1 spent 85% of its width on empty space** -- the June block is at day -70 and the entire
   result lives inside +/-8.
3. **Fig 1 drew a line from day -2 to day +1**, straight through the lesion, implying a continuous
   decline that was never measured. Pre and post are separate segments now.
4. **Fig 3b's green line had ONE point per animal.** Within-session accuracy lives in `section_g`
   under `"<cond> within-session"`, but each session's block carries only ITS OWN post row -- taking
   the list from the first session and stopping is the obvious read and loses the trajectory.
   `poststroke_grid.json`, the other obvious source, holds only days 1-2.
5. **Fig 3a as six lines per panel was unreadable AND structurally wrong**: it read the lick class,
   so the impaired positions -- the ones the figure exists to describe -- had no cell to plot.

### Two definitional choices worth keeping
- **"Impaired" is defined by the WORST post-stroke session, not the pooled rate.** Pooling reported
  PS95 as having no impaired position at all, in the animal whose day-1 far_R collapse (0.00 -> 0.87
  by day 2) is the cleanest in the cohort; the recovery averages the deficit away.
- **In fig 2b the ANIMAL is the unit** -- mean of four per-animal LOSO accuracies, SEM across
  animals. Pooling all ~44 held-out sessions gives a much tighter interval that describes how much a
  SESSION varies, not an ANIMAL, and a cohort claim is about animals.

### And the guard earned its keep
`FIG = Path("E:/cue_lick")` failed `tests/test_no_hardcoded_machine_paths.py` immediately -- that
literal is the analysis box's path and would be wrong on the imaging box. Now `PathResolver().root
("figures_working")`.

---

## MEAN-PATTERN SIMILARITY CORROBORATES THE CODING DIRECTIONS BY A DIFFERENT ROUTE (2026-08-25)

Priya asked whether the ENCODING models would help or whether coding directions are better. The
useful part of the encoder framing turns out to be obtainable without fitting one: correlate the
post-stroke MEAN ACTIVITY PATTERN at each position against the pre-stroke mean pattern at EVERY
position. `poststroke_compare.pattern_similarity` already computed the DIAGONAL of that ("is it the
same code"); the OFF-DIAGONAL -- what it looks like INSTEAD -- did not exist.
`grant_figures.fig_pattern_similarity` (figure 6) is it.

**WHY THIS IS NOT REDUNDANT WITH THE CODING DIRECTIONS.** The two fail in different ways:

| | needs a contrast? | sensitive to global gain? |
|---|---|---|
| coding direction | YES -- and so it FAILS at exactly the impaired positions, where one side has no trials | no (unit vectors, plus the normunit control) |
| mean-pattern correlation | no -- far_R's mean pattern is well defined from 400 miss trials with no partner | YES -- a uniform amplitude change moves every cell together |

So agreement between them is worth much more than either alone.

**THE BASELINE PANEL IS THE MEASUREMENT, not decoration.** Positions are intrinsically similar
pre-lesion, so a raw r of 0.8 is meaningless alone. Both panels are scored against the SAME
reference -- one half of the pre-stroke trials -- so the left panel is the no-lesion expectation and
its diagonal is the split-half CEILING (0.93-0.99 in every animal).

### The result, post-cue window, lick + miss-while-working

Own-position correlation (right-panel diagonal), against that ~0.95 ceiling:

| animal | close_L | close_center | close_R | far_L | far_center | far_R |
|---|---|---|---|---|---|---|
| PS92 | 0.88 | 0.67 | 0.84 | 0.47 | 0.83 | **−0.01** |
| PS93 | 0.24 | 0.04 | 0.64 | 0.66 | 0.56 | **−0.26** |
| PS94 | 0.76 | 0.24 | 0.60 | 0.10 | 0.31 | **−0.01** |
| PS95 | 0.75 | 0.76 | 0.76 | 0.64 | 0.17 | 0.33 |

**far_R's own pattern collapses to zero or below in PS92, PS93 and PS94.**

**AND THE CROSS-POSITION ROW REPRODUCES THE far_L SUBSTITUTION.** PS93's post-stroke far_R
correlates **+0.63 with PRE-stroke far_L** while correlating **−0.26 with its own** pre-stroke
pattern. The coding directions reached the same conclusion for PS93 (far_R -> far_L in ENL and cue,
2026-08-24) by a construction that shares none of this one's assumptions. PS92 leans the same way
weakly (far_center +0.31, far_L +0.23, own −0.01); PS94's row is flat near zero, i.e. no substitute
rather than a substitution; PS95's own diagonal is still its best (+0.33), the recovering animal
again.

### THE STRUCTURAL FEATURE TO KEEP IN VIEW
The baseline panels carry strong NEGATIVE close-vs-far correlations (PS92 close_L vs far_R −0.78),
so this pattern space is dominated by a close-vs-far axis -- the same geometry that produced the
one-vs-rest flaw and the cross-ring warning. Post-stroke those negatives soften toward zero in every
animal, which is a GLOBAL change and not a far_R one. Read the far_R collapse against that, not in
isolation.

### CLASS VARIANTS, AND ONE THAT DELIBERATELY DOES NOT EXIST
`lick` (trials with a lick) for all three windows; `working` (lick + miss-while-working, terminal
quit period removed) for ENL and cue ONLY. There is no `working` variant in the LICK window because
a no-lick trial is placed at the CUE there, so pooling the two classes would average patterns from
two different times and call the result a position effect -- the same trap as the lick-window
alignment issue recorded on 2026-08-24.

---

## THE COHORT IS UNEQUAL RIGHT NOW, AND THE FIGURES SAY SO (2026-08-25)

Priya: "there is new PS92 and PS93 data from 8/24 - are you using this?" Partly, and the partial
answer was the problem.

| source | PS92 8/24 | PS93 8/24 |
|---|---|---|
| behaviour CSVs (figs 1, 1b) | yes | n/a |
| recomputed from `phase_labels("post")` (5b, 5c, 6, 6b) | **yes** | **no** |
| `section_g.json` (3b, 5) | **no** | no |
| `coding_direction.json` (3a) | **no** | no |

**Both animals' 8/24 sessions are preprocessed and LocaNMF'd on MICROSCOPE.** Only PS92_0824 was
registered (`await_locanmf`, commit ec5dd4e); PS93_0824 exists on disk and is absent from
`sessions.yaml`, so it is invisible to everything. And `section_g.json` / `coding_direction.json`
predate even the PS92 registration.

**The dangerous form of this is not the missing day, it is the ASYMMETRY.** The figures that
recompute give PS92 six post-stroke sessions and PS93 five, in a cohort comparison, with nothing on
the figure to say so. A uniformly missing day is a known limitation; an unequal one is a silent bias
toward whichever animal happens to be registered.

**Decision (Priya, 2026-08-25): leave the data alone and regenerate after the next nightly.** That
only works if the gap is legible meanwhile, so `grant_figures.coverage_note()` stamps every
post-stroke figure with the per-animal session count and warns when they differ. It is COMPUTED from
`config.phase_labels("post")`, never typed, so it corrects itself when PS93_0824 is registered and
stops warning once the counts match.

**PENDING, for whoever runs the next nightly:**
1. Register PS93_0824 (preprocessed and LocaNMF'd; `await_locanmf` should pick it up).
2. Re-run `poststroke_section_g` and `position_coding_directions` so the JSON-derived figures include
   8/24 for both animals.
3. Regenerate the grant figures (`python -m wfield_local.grant_figures`) and confirm the footer no
   longer warns.

---

## INTERVALS AND A NULL FOR THE PATTERN MATRICES — AND WHY SESSIONS ARE NOT RESAMPLED (2026-08-25)

Priya: "how can we add significance / error bars? permutation? bootstrapping?" -- then, decisively:
"session-level is problematic with things dynamically changing over sessions, no?"

**She is right, and it settled the design.** A session bootstrap assumes days are exchangeable draws
from one distribution. They are not, and the per-session figures now prove it rather than assert it:

    mean own-position similarity, post-cue, working trials
    PS92  PRE 0.95 | 0.41 0.65 0.54 0.48 0.37 0.59
    PS93  PRE 0.96 | 0.36 0.22 0.15 0.19 0.32
    PS94  PRE 0.97 | 0.14 0.06 0.28 0.34 0.25 0.55     <- 0.06 to 0.55 across one week
    PS95  PRE 0.97 | 0.42 0.58 0.59 0.49 0.47 0.36

Resampling those days would fold a ninefold trajectory into "sampling noise" and produce an interval
for a post-stroke state that does not exist.

### WHAT WAS BUILT INSTEAD
- **Stratified bootstrap, sessions held FIXED**: trials resampled WITHIN each session, session
  composition untouched. The interval then means "how well determined is this estimate GIVEN THESE
  DAYS". It does NOT license generalisation to other days -- with a moving target and six sessions
  that question needs the trajectory, which the per-session figure shows rather than summarises.
- **Permutation null, position labels shuffled WITHIN session**: post-stroke trials keep their
  session, their counts and the global post-stroke pattern; only WHICH POSITION they belong to is
  shuffled. A ring on the figure therefore means POSITION-SPECIFIC STRUCTURE, not "r != 0" -- and
  since positions are intrinsically similar, a zero-null would ring nearly every cell. Verified on
  synthetic data: the null centres at 0.580 against a theoretical global-mean correlation of 0.579.
- **The claim gets the interval, not the cells**: a third panel plots post MINUS baseline for each
  position's own-position r, differenced DRAW BY DRAW. The two panels share the same resampled
  reference, so their errors are correlated and differencing the published CIs would have overstated
  the interval.

**A performance bug worth recording**: the first permutation rebuilt Python lists of individual trial
rows on every resample -- O(sessions x trials) per iteration, minutes per animal. Stacking each
session's trials once with a label vector makes a permutation one `rng.permutation` and six boolean
means. Same statistic, ~100x faster.

### THE RESULT THE INTERVALS CHANGE
Post minus baseline, own position, post-cue, working trials (95% stratified bootstrap):

| | close_L | close_center | close_R | far_L | far_center | far_R |
|---|---|---|---|---|---|---|
| PS92 | −0.08 | −0.30 | −0.15 | −0.45 | −0.10 | **−0.92** |
| PS93 | −0.58 | −0.78 | −0.24 | −0.22 | −0.25 | **−1.15** |
| PS94 | −0.24 | −0.66 | −0.38 | −0.85 | −0.66 | **−0.95** |
| PS95 | −0.22 | −0.20 | −0.21 | −0.34 | −0.77 | −0.65 |

far_R is the largest drop in all four animals, **and every position drops with an interval excluding
zero**. So the effect is GRADED, not far_R-exclusive. That is a materially more careful statement
than "the far_R code is lost", and it is the same global-change signature that has surfaced in the
manifold analysis, the pre-cue outcome-blind arm and the close-vs-far geometry of the baseline
panels. Quote the gradient, not the exclusivity.

### ORDERING, REVERSED FROM WHAT I SAID EARLIER
I called the pooled figure the one to quote and the per-session one a diagnostic. With
non-exchangeable sessions that is backwards: **the per-session figure is primary** -- the trajectory
IS the result -- and the pooled figure is a summary whose interval is conditional on those six days.

---

## PS93_0824 REGISTERED: THE COHORT IS BALANCED, AND THE NUMBERS MOVED A LITTLE (2026-08-25)

`await_locanmf` registered PS93_0824 in commit 4f33d61 while this work was in flight, so the cohort
is now 6/6/6/6 and `coverage_note()` has switched its own warning off, as designed.

**The five-session PS93 numbers reported earlier are SUPERSEDED**, not wrong-then-right: post-stroke
far_R own-position r moved from −0.26 to **−0.17**, and the far_L substitution from +0.63 to
**+0.70** (post-cue, working). The conclusion is unchanged and the cell is ringed against the
permutation null, which is the useful fact -- a finding that survives a 20% change in the data behind
it. PS93's per-session diagonals show no plateau (0.36 / 0.22 / 0.15 / 0.19 / 0.32), so a sixth day
was not a small perturbation of a stable average and the numbers moving was expected.

**STILL STALE, and the balanced footer does NOT cover it:** `section_g.json` and
`coding_direction.json` predate BOTH 0824 sessions, so grant figures 3a, 3b and 5 -- and every G9 and
section-G deck slide -- stop at 8/22 for PS92/PS93. The cohort-coverage footer reads
`config.phase_labels("post")` and therefore reports 6/6/6/6 on those figures too, which is true of
the CONFIG and not of the JSON they were built from. Re-running `poststroke_section_g` and
`position_coding_directions` is left to the nightly (Priya, 2026-08-25) and remains item 0 in
`docs/STATUS_2026-08-23.md`.

---

## THE GRANT FIGURES ARE NOW A DECK SECTION (H), NOT A SIDE DELIVERABLE (2026-08-25)

Priya: "add these figures to the analysis deck code too."

All 25 PNGs under `<labcams>/grant_figures` are placed by `locanmf_analysis_deck.py` as **section H**
(H1/H1b behaviour, H2/H2b pre-stroke cross-day decoding, H3/H3b coding retained + frozen-vs-within,
H4 pre-stroke confusion, H5/H5b/H5c frozen decoder pre-vs-post, H6/H6b pattern similarity). A
placement dry-run confirms **zero unplaced PNGs** — the failure mode this repo keeps hitting is a
figure that exists on disk and appears in no deck, and the check is cheap.

**Section H is deliberately the opposite of the rest of the deck.** The figures are caveat-LIGHT for a
reader who has not been in the weeds; every caveat that would change a reading therefore moves into
the SPEAKER NOTES, which travel with the slide: the row/column convention flips between the confusion
figures (rows = true) and the pattern figures (rows = post-stroke); H5b's quit-period gate is not
validated; H6b is PRIMARY over H6 when sessions move. Prose that lives only in DECISIONS.md is prose
that goes stale unread.

### TWO THINGS THE CHANGE FORCED

1. **`build_analysis_deck(..., grant_dir=None)` is an explicit parameter.** The grant set is the ONE
   deck input that does not live under `src`: it is a deliverable under `labcams`, not an analysis
   intermediate under `figures_working`. Resolving it inline made the two hermetic deck tests reach
   onto the MICROSCOPE share and place 25 real figures into a build whose whole premise was an empty
   figure directory (`assert 25 == 0`). The tests now pass an empty tmp `grant_dir`. A test that
   silently depends on a network share is not testing the builder.
2. **The per-file suffix is derived from the GLOB, not from the stem.** `stem.split("_", 2)[-1]` gave
   "confusion prestroke cue" — repeating the family name already in the slide title. Taking the part
   of the stem the `*` matched gives "cue", "precue lick", "per session precue working".

### AND IT SURFACED TWO FIGURES THAT HAD NO SLIDE
`grant_3a_coding_retained.png` and `grant_3b_frozen_vs_within.png` were missing from the first pass —
figure 3 is the one Priya asked for by name ("changes in ENL, cue and lick neural activity coding
after stroke, shown over time and for each animal"). They are H3/H3b, and the H labels below them
were shifted rather than leaving a gap.

Re-rendered 3a/3b/5 so they carry the corrected source-aware footer. It now reads
`PS92 5, PS93 5, PS94 6, PS95 6 — UNEQUAL — STALE: registered but ABSENT here: PS92_0824, PS93_0824`,
which is the point of the change: those three figures are built from `coding_direction.json` /
`section_g.json`, and the config-reading footer had been reporting the balanced 6/6/6/6 cohort on
figures that do not contain it.

---

## THE PRE-STROKE BASELINE WAS POOLED WHERE THE POST COLUMNS WERE PER-SESSION (2026-08-25)

Priya asked two questions that turned into a correction: *"building each post-stroke session's
split-half correlation matrix (to see if part of the 'lost code' is just that responses are more
variable)"*, and then *"can you redo the other pre vs post figures if needed to account for higher
pre-stroke numbers?"*

**She was right that there was something to account for, and it was worse than it looked.**

### THE ERROR
Figures 6 and 6b drew a random half of the POOLED pre-stroke TRIALS as the reference and the other
half as the no-lesion baseline. Both halves therefore came from the **same days**. The baseline
contained within-session trial noise and *no day-to-day drift whatever*, while every post-stroke
panel compares DIFFERENT days against those days. It was a ceiling no across-day comparison can
reach — and the headline result, "post minus baseline is negative at every position", was measured
against it.

The first draft of figure 7 had the same error in a starker form and was caught before it was
believed: it compared the split-half reliability of the POOLED pre-stroke set (six sessions,
0.97–0.99) against one post-stroke session at a time (0.14–0.67) and appeared to show that
post-stroke responses were half as repeatable as baseline. Split-half reliability rises with trial
count; almost all of that gap was six times the trials. The same animal's INDIVIDUAL pre-stroke
sessions read 0.77–0.97.

### THE FIX
**Leave-one-session-out everywhere the comparison is a correlation.** Figure 6 splits pre-stroke
SESSIONS rather than trials; 6b, 7, 7b, 8 and 8b score each pre-stroke session against the pool of
the OTHERS and average. One session against other days — exactly the shape of every post-stroke
column, and disjoint, so nothing is scored against itself.

### WHAT WAS NOT WRONG, AND WHY
The decoding matrices were already safe: `crossed_confusion` uses
`cross_val_predict(LeaveOneGroupOut, groups=GE)`, the no-lick control trains on the other pre-stroke
sessions, and 5b has an explicit LOSO loop. There is also a structural reason the decoders were never
much exposed: **a decoder classifies each trial individually**, so test-set size moves the PRECISION
of an accuracy estimate, not its expectation. A correlation between two MEANS is attenuated by how
many trials went into each mean. That is why the damage was confined to the correlation figures, and
it is the rule to apply when adding the next one.

### THE CEILING IS NOT 1.0 — AND THAT IS A RESULT, NOT A DEFECT
PS94 post-cue disattenuates to **0.75–0.86** in the leave-one-out column. Disattenuating by a
WITHIN-session reliability removes trial noise and nothing else; two pre-stroke days also differ by
ordinary drift, which no within-session split half can see. So the PRE column is the ceiling, the
same way the axis analysis uses a matched null instead of comparing cosines with unity. The two
constructions agree independently — PS94 scores 0.80 here and 0.89 on the axis null.

**Read every post column against PRE, never against 1.** A post value of 0.7 read against 1.0 rather
than against 0.8 turns "unchanged" into "30% lost".

---

## IS THE LOST CODE A MOVED CODE OR A NOISIER ONE? — IT MOVED (2026-08-25)

The control Priya asked for, built as figures 7 (within-session split-half matrices) and 7b (figure
6's diagonal, raw and disattenuated by `raw / sqrt(rel_post * rel_pre)`).

**Disattenuation barely moves anything**, because reliability is mostly 0.8–0.95 and there was little
attenuation to remove. Post-cue, working trials:

| | PS92 far_R | PS93 far_R | PS94 far_R | PS95 far_R |
|---|---|---|---|---|
| reliability, post-stroke days | 0.90–0.95 | 0.79–0.96 | 0.82–0.98 | 0.81–0.99 |
| raw similarity to pre-stroke | −0.09 … 0.02 | −0.33 … 0.16 | −0.26 … 0.37 | 0.06 … 0.57 |
| PRE ceiling | 0.77 | 0.73 | 0.71 | 0.79 |

PS92's far_R pattern is measured *more* repeatably after the lesion than before (0.90–0.95 against a
pre-stroke 0.80) and has **zero** relationship to the pre-stroke far_R pattern. That is not a noisy
code. It moved.

**The control is not vacuous — it caught a different position.** `close_center` reliability collapses
in three animals (PS92 d4 = 0.21, PS93 d5 = 0.01, PS94 d4 = 0.32) and those cells are suppressed
rather than printed, because dividing by sqrt(0.01) yields a number with no information in it.
**close_center's apparent change must not be quoted as a lesion effect.** Why it is poorly estimated
is open; trial count and behavioural variability are both plausible and checkable.

**Spearman-Brown is load-bearing.** `_split_half` correlates two means of n/2 trials, but figure 6
correlates the mean of all n. Skipping the projection makes every reliability too low, and since 7b
DIVIDES by sqrt(rel_post × rel_pre), too small a denominator inflates every disattenuated value — it
would have manufactured the "code moved" verdict the panel exists to test.

---

## THE CROSSNOBIS VERSION, AND WHAT IT CAN AND CANNOT SAY PER POSITION (2026-08-25)

Priya: *"would this still give us any per-position information though? or just overall
representational geometry similarity?"* **Both, but they are different questions**, so they are two
figures:

| | keeps per-position patterns | invariant to a global gain change |
|---|---|---|
| 6 (correlation, cross-set) | yes | **no** |
| **8** — crossnobis distance d(post P, pre Q) | yes | **no** (but noise-UNBIASED) |
| **8b** — second-order RDM correlation | row-wise only | **yes** |

Crossnobis's native object is a within-set RDM, so there is no single translation of figure 6. Figure
8 keeps every position and gives up gain-invariance; **8b is RSA proper** and cannot be fooled by a
uniform amplitude change, with per-position information surviving as each position's ROW — "is far_R
still arranged relative to everything else the way it was" is answerable, "did far_R's pattern move"
is not.

**Why the pair matters:** figure 6's headline — every position drops, far_R most — is precisely the
signature a global amplitude change would leave. 8b is the measure that distinguishes them. Verified
on synthetic data rather than asserted: with post = pre scaled by 0.4 and nothing moved, the
second-order correlation stays >0.9 while the cross-set distance rises.

**Is 6b "a version of RSA"?** Partly. It is first-order representational geometry — in Kriegeskorte's
terms the pre×post off-diagonal block of a joint RDM — but it has no second-order step and it
REQUIRES feature correspondence (the shared joint-LocaNMF basis), which is the very thing RSA exists
to avoid needing. Call it "pattern similarity", or "RSA-style"; reserve "RSA" for section F
(crossnobis) and 8b, or a reader will assume a gain-invariance this measure does not have.

---

## DELTA VIEWS, AND A FIGURE-7 ASYMMETRY THAT WAS PURE NOISE (2026-08-25)

Priya: *"make additional versions of fig 5, 6, 7 as differences from prestroke."* Built as **5d, 6d,
7d**: column 1 is the pre-stroke reference in its own units, every later column is that day MINUS it.

**Why it earns a separate figure rather than a reader's subtraction.** The pre-stroke reference is
not uniform — close positions are intrinsically confusable and far ones are not — so an absolute
cell of 0.3 may sit against a baseline of 0.35 (nothing happened) or 0.9 (most of the code gone). It
made the substitution result legible at a glance: in every animal the far_R ROW shows a negative
diagonal with POSITIVE off-diagonal at the close/central columns, i.e. post-stroke far_R activity
came to resemble pre-stroke close_L/close_center. **That matches Priya's behavioural observation**
that far_R miss trials carry incomplete leftward or central licks (2026-08-24). Present in the
absolute panels, but only visible to a reader who remembered what those cells looked like before.

**r is differenced, NOT r-squared** (Priya's phrasing was "deltas of r2 / accuracy"): the sign is the
result — a far_R pattern moving ONTO far_L shows as a positive off-diagonal — and squaring erases
exactly that.

### THE FIGURE-7 ASYMMETRY
Priya, on the first render: *"why aren't the fig 7 matrices symmetrical about the diagonal?"*
`M[P,Q]` was `corr(half A at P, half B at Q)`, so (far_R, close_L) and (close_L, far_R) were **two
estimates of one quantity**, differing only in which random half of each position's trials landed on
which side. The asymmetry was pure estimation noise drawn as structure — in a figure whose entire job
is to quantify estimation noise. Off-diagonal cells now average both pairings: symmetric, same
expectation, still cross-validated (no half is ever correlated with itself), half the variance. The
discarded difference is kept as `_split_half_asymmetry` — a real noise read, reported rather than
smuggled into the picture.

This does NOT apply to 6/6b/8, where rows are post-stroke and columns pre-stroke: those are genuinely
different sets and the asymmetry IS the substitution signal. `_crossnobis_within` was already
symmetric.

---

## SECTION G's `pattern_similarity` HAD NO CEILING AT ALL (2026-08-25)

It reported a bare pre-vs-post `r`, which invites comparison with 1.0 — the error the whole
leave-one-out correction above is about. It now also emits **`r_pre_loo`** (each pre-stroke session
against the pool of the others, averaged) and `fig_similarity` draws it as a grey line per position,
with the suptitle saying to read the bars against IT and never against 1.

**Additive on purpose.** A `section_g.json` written before today has no such field, so the band
simply does not appear rather than crashing a render that also holds fifteen other panels. Tests pin
both paths.

---

## GRANT-FIGURE STALENESS AUDIT (2026-08-25, 17:00)

Asked directly ("are the other grant_figures accurate / not stale?"), so recorded rather than
answered once:

| figure | built from | state |
|---|---|---|
| 1, 1b behaviour | live, config + behaviour tables | CURRENT (both 0824 sessions registered 08-24 17:34 / 20:08, before the 06:05 render) |
| 2, 2b pre-stroke decoding | live, PRE-STROKE ONLY | CURRENT — 0824 is post-stroke and cannot affect them |
| **3a coding retained** | `coding_direction.json`, mtime 08-24 05:44 | **STALE** — missing PS92_0824, PS93_0824 |
| **3b frozen vs within** | `section_g.json`, mtime 08-24 03:57 | **STALE** — same |
| 4 pre-stroke confusion | live, LOSO, pre only | CURRENT |
| **5 frozen pre/post** | `section_g.json` | **STALE** — same |
| 5b, 5c, 5d | live recompute | CURRENT |
| 6, 6b, 6d, 7, 7b, 7d, 8, 8b | live recompute, corrected baseline | CURRENT (rendered 13:19–16:46 today) |

**The three stale figures declare it themselves** — this is what the source-aware `coverage_note`
was built for on 2026-08-25 morning. Their footer reads *"PS92 5, PS93 5, PS94 6, PS95 6 — UNEQUAL —
STALE: registered but ABSENT here: PS92_0824, PS93_0824"*. A config-reading footer would have printed
a confident 6/6/6/6 on all three, which would have been true of the config and false of the figure.

`poststroke_section_g` was re-running as this was written (started 16:52), which will refresh
`section_g.json` and with it figures 3b and 5. **It will NOT add `r_pre_loo`**: the nightly runs from
the MAIN checkout, which is at `a7583e6`, and that field landed in `5ed4d75`. The main checkout was
deliberately left alone rather than pulled mid-run — swapping code under a running multi-step
pipeline would make different steps run different versions, which is a worse failure than a
one-cycle delay.

---

## BOOTSTRAPPING THE POST-STROKE FIGURES: BLOCKS YES, SESSIONS NO, PERMUTATION NO (2026-08-25/26)

Priya: *"can we bootstrap / permute the post-stroke session blocks or trials?"* All three parts of
that question turned out to have different answers, and the reasons are worth keeping.

### BLOCKS, NOT TRIALS
Trials adjacent in time share arousal, satiety and slow drift, so an i.i.d. trial bootstrap treats
correlated samples as independent and returns intervals that are **too narrow**. The scheduler's own
~6-trial position blocks are the natural unit and the pipeline already uses them for GroupKFold —
`pool_sessions` had been returning them all along as `BE`, discarded at every call site as `_B`.

A block belongs to ONE position by construction (a new block starts when the position changes, or at
`block_size_max`), so resampling a session's blocks also resamples each position's trial count —
which is correct, since how many trials a position got is itself uncertain.

**Tested, not asserted.** With a per-block offset shared by six trials, the block bootstrap's spread
is **>1.5x** the i.i.d. one. That is the entire justification for the added machinery, so it is a
test rather than a docstring claim.

### SESSIONS HELD FIXED
Priya's own earlier objection, and it still governs: days are not exchangeable while an animal is
recovering. PS94's figure-8 diagonal runs 0.99 → 0.46 across one week; there is no single
post-stroke value for an interval to be *about*. An interval therefore means "how well determined
GIVEN THESE DAYS" and licenses nothing about days not recorded — the trajectory is what speaks to
that, which is why figure 9 is per-day rather than pooled.

### BOOTSTRAP, NOT PERMUTATION, FOR ASYMMETRY
Permuting position labels equalises the condition means, so the true distances collapse toward zero.
But the sampling variance of a crossnobis distance **scales with the true difference vector**, so a
real and perfectly SYMMETRIC separation still yields a larger |D − Dᵀ| than permuted data does. The
permuted null therefore sits too low and would call ordinary noise "asymmetry". A bootstrap interval
on `D[P,Q] − D[Q,P]` has no such problem, and matches trial counts by construction.

### THE BUG THE TEST CAUGHT BEFORE ANY FIGURE DID
The first `_delta_diag_ci` computed its baseline as `mats_fn(ref, ref)` — the resampled reference
correlated **against itself**, whose diagonal is exactly 1.0 by construction — so every delta came
out at about −1 regardless of the data. On real data that would have rendered as a catastrophic and
suspiciously *uniform* loss at every position in every animal, which is plausible enough to quote.

**That is the third instance in two days of one error shape: a reference built from a pool that
contains the thing being scored.** The others were figure 6's half-split baseline (both halves from
the same days) and 8b's ceiling (each session's RDM against the pooled RDM containing it). Worth
stating as a standing check: *whenever a baseline is computed, ask what is inside it.*

### RESULT, POST-CUE / WORKING (change in mean own-position r, 95% block bootstrap)

| | d1 | d2 | d3 | d4 | d5 | d7 | d9 |
|---|---|---|---|---|---|---|---|
| PS92 | −0.36 | −0.13 | −0.23 | −0.27 | −0.40 | −0.18 | — |
| PS93 | −0.37 | −0.52 | −0.59 | −0.55 | −0.41 | −0.22 | — |
| PS94 | −0.65 | −0.72 | −0.48 | −0.43 | −0.52 | −0.19 | −0.32 |
| PS95 | −0.29 | −0.13 | −0.12 | −0.23 | −0.26 | −0.34 | −0.38 |

**Every one of the 27 intervals excludes zero.** And the intervals separate trajectories the point
estimates only hinted at: PS93 and PS94 RECOVER (day-7 endpoints not overlapping their own worst
days), PS95 does the OPPOSITE — −0.13 at day 2 drifting to −0.38 at day 9, day 9 not overlapping
days 2 or 3 — and PS92 is flat throughout. A pooled figure averages all four into "a decline".

### FIGURE 9 EXISTS BECAUSE THE INTERVALS WERE UNREADABLE
Asked whether any figure showed the bootstrap results, the answer was no: they were text above 6x6
matrices in 6d/7d/8d. Whether a position is recovering, holding or worsening is a trajectory, and a
trajectory read out of twenty-eight small matrices by eye is not read at all. Figure 9 plots it, per
animal and per position.

**Zero is a real null there.** It means "this day differs from the pre-stroke reference no more than
one pre-stroke day differs from the others", because the baseline is leave-one-session-out. Read
against 1.0 instead, every point in every panel would look like a large loss.

### TWO LAYOUT FAULTS, NOW MEASURED RATHER THAN EYEBALLED
Both shipped and both were reported by Priya rather than caught here, which is the argument for
measuring them: a figure that is merely ugly looks much like a figure that is fine.

- The reference colour bar was drawn over the day-1 matrices. The first fix moved only the BAR;
  matplotlib draws ticks and axis label on the right by default, so the rotated label still reached
  over the first panel. The test now takes the bounding box of the bar, its tick labels AND its
  axis label, and requires all three to end before the day-1 panel begins.
- Describing the bootstrap in 6d's header produced a 420-character line, and `bbox_inches="tight"`
  sizes the canvas around everything it contains — the figure went from aspect 2.1 to 3.3 with the
  panels squashed into a third of it. `_delta_grid` now wraps lines over 150 characters while
  preserving the caller's own newlines.

A note for the next such test: the reference bar is added with `fig.add_axes` and therefore does NOT
carry matplotlib's `"<colorbar>"` label. Selecting on that label finds only the delta bar on the far
right and reports a false overlap — which is what the first version of the check did.
## THE FROZEN MODELS WERE TRAINING ON POST-STROKE SESSIONS (found and fixed 2026-08-26)

Priya, on being shown the pool composition: *"yes the frozen models were supposed to be pre stroke
only!!"*

### What was happening

`pooled_frozen_loso` predicted with `cross_val_predict(_pipe(), XE, YE, cv=LeaveOneGroupOut(),
groups=GE)` over **every pooled session**, and `pooled_frozen_encoder` used `tr = ~te`. Both mean
"train on all the other sessions" — which was exactly right when written on 2026-08-11, because every
curated session was pre-stroke. The strokes had not happened yet.

As post-stroke nights registered, they joined the training pool silently. Measured 2026-08-26:

| animal | pre | post IN THE TRAINING POOL |
|---|---|---|
| PS92 | 16 | 6 — 0818–0822, 0824 |
| PS93 | 13 | 6 — 0818–0822, 0824 |
| PS94 | 16 | 7 — 0817–0821, 0823, 0825 |
| PS95 | 16 | 7 — 0817–0821, 0823, 0825 |

About **30% of the training data** behind a number whose entire purpose is to be a lesion-free
baseline, growing every night. The drift is visible in the numbers: PS92 post-cue transfer cost was
**+0.140** on 11 pre-stroke sessions in the 8/11 entry above, and **+0.123** on the 17-session mixed
pool on 8/23.

### Why it matters, and why the encoder is worse

The argument this project rests on is: transfer cost is positive pre-stroke → a frozen decoder does
not decay across days on its own → post-stroke degradation can be attributed to the lesion. That
inference **requires the reference to be lesion-free**. It was not.

For the ENCODER it is worse than a weakened inference. Its residual on post-stroke trials IS the
representational-change readout, so training on post-stroke trials fits the model to the very data
whose departure from it is the result — shrinking the effect toward zero by construction.

### Why it was invisible

Two reasons, both worth keeping.

**The word "curated" does not name a phase.** `config.curated_dates()` already defaulted to
`phase="pre"` and was stroke-aware; it was not the direct cause. But `nightly_figs` built its own
label list from `from_list` (all phases) and called `pooled_frozen_loso` directly, bypassing it — and
at every review "the curated sessions" read as correct, because for the first ten weeks of this
project curated *did* mean pre-stroke. A name that states the phase cannot quietly change meaning
when the cohort does. Added `config.prestroke_dates()` / `poststroke_dates()`.

**Nothing in the output said what the pool was.** The result JSON recorded accuracies but never the
training set, so no consumer could notice. It now carries `training_phase`, `pre_labels`,
`post_labels`, `n_pre_sessions`, `n_post_sessions`.

Same failure class as the hardcoded-date-list entry of 2026-08-19: a date list treated as fixed while
its meaning moved underneath.

### The fix

Pooling for FEATURE ALIGNMENT still uses every session — that is what puts post-stroke sessions in a
comparable feature space at all, and restricting it would delete the post-stroke rows the deck is
built from. Only TRAINING is restricted:

- a held-out **pre-stroke** session is scored leave-one-out among pre-stroke sessions only;
- a **post-stroke** session is scored by one model fitted on ALL pre-stroke sessions and applied
  unchanged — the train-pre / apply-post design the deck already claimed;
- `loso_accuracy` and `mean_within`, and therefore `transfer_cost`, are computed over **pre-stroke
  sessions only**. Both terms have to be pre-stroke or the contamination re-enters through the other
  side of the subtraction.
- fewer than 2 pre-stroke sessions REFUSES rather than approximating: there is no cross-day reference
  to be had, and a number computed some other way would be worse than none.

Guarded by `tests/test_frozen_models_are_prestroke_only.py`. **Every frozen number published before
2026-08-26 was computed on a contaminated pool and needs regenerating** — the models have to be
rebuilt, not just recomputed forward.

### Not affected

Anything that never pooled across the lesion: within-session decoders, the per-session encoders, RSA,
coding directions, and the `spout_behavior` outputs. The joint BASES are also unaffected — they were
fitted 2026-08-16 over 11 pre-stroke sessions each, before any lesion, and `joint_locanmf` refuses to
refit silently precisely so that a reference frame cannot move.

---

## WHICH GRANT FIGURES THE FROZEN-MODEL CONTAMINATION REACHED (2026-08-26)

Companion to the fix in `3de237c` ("frozen models were training on post-stroke sessions"). That
entry records the bug; this one records the blast radius on the figures, so nobody has to re-derive
it, and — more usefully — WHY most of the set escaped.

Audited all **78 model-fit sites** in `wfield_local/` (every `.fit(`, `cross_val_predict`,
`LeaveOneGroupOut`), classified by whether the training rows are visibly phase-restricted: 22
pre-restricted, 39 within-session (no restriction needed), 17 with none visible — of which most are
Ledoit-Wolf whiteners on residuals rather than decoders.

### THE VERDICT

| figure | source | verdict |
|---|---|---|
| 2, 2b, 4 (precue + cue) | `joint_xsession_decoder_*.json` | **CONTAMINATED** — 24 post-stroke labels in the pool, no `training_phase` key |
| 4 (lick) | `joint_xsession_decoder_lick.json` | clean — 0 post-stroke labels |
| 3a | `coding_direction.json` | clean — no pooled frozen fit |
| 3b, 5 | `section_g.json` | clean, see below |
| 5b, 5c, 5d | recomputed live | clean — fit on `XE[e_pre]` |
| 6, 6b, 6d, 7, 7b, 7d, 8, 8b, 8d, 8e, 9 | recomputed live | clean — no decoder at all |

**Figures 2, 2b and 4 are the ones the deck calls "the strongest and least contestable result in
the set -- no lesion, no trial-class definitions, no engagement gate, no alignment inference".** They
were the least contestable in every respect except which sessions trained the decoder.

### WHY 5 AND 3b ARE CLEAN, WHICH IS NOT OBVIOUS
They DO use a frozen pre-stroke decoder — the first pass of this audit grepped for
`pooled_frozen_loso|pooled_frozen_encoder`, found none, and wrongly concluded "no frozen decoder"
(Priya: *"doesn't figure 5 use the frozen decoder?"*). It does; it builds its own. Every fit site in
`poststroke_compare` masks training to `pre_i`:

    decode_matched      tr = isin(GE, pre_i) & kp;  LOSO groups=GE[tr]
    crossed_confusion   tr = isin(GE, pre_i);       LOSO groups=GE[tr]
    pre-no-lick control trn = tr & (GE != gsess)    -- leave-one-out WITHIN pre
    engaged-vs-undetected  pre_e / pre_u both isin(GU/GE, pre_i)

**The right test is what the training mask is, not which helper is called.** A grep for a function
name answers a different question than the one being asked.

### WHY THE NEW FIGURES ESCAPED — NOT FORESIGHT
Figures 5b onward fit their own classifier from `pool_sessions`, which returns raw pooled data and
FORCES the caller to state what it trains on. `pooled_frozen_loso` decides internally, and its
internal decision (`LeaveOneGroupOut` over everything) was correct for the ten weeks when every
curated session was pre-stroke. The lesson is about where the choice lives: **a helper that picks
the training set silently will eventually pick the wrong one, and nothing at the call site will
say so.**

### THE SAME SHAPE, ONE MORE TIME
This is a fourth instance of the pattern catalogued a day earlier — *a reference built from a pool
that contains the thing being scored* — now at cohort scale: the pre-stroke baseline's TRAINING pool
contained post-stroke sessions. The earlier three were figure 6's half-split baseline, 8b's ceiling,
and `_delta_diag_ci`'s `mats_fn(ref, ref)`.

### A LIVE TRAP LEFT BEHIND, mechanism not artifact
`nolick_reference_prestroke.json` is written by `if not frozen.exists()` copying whatever
`nolick_reference.json` holds at that instant. **The existing artifact is clean** — 11 pre-stroke
dates, 44 pre-stroke labels, checked rather than assumed (an earlier claim here that it was
contaminated was inferred from its 8/19 mtime being after the lesion, and was wrong). But the live
file now holds 19 dates including 0817–0824, so it is pre-only **by an accident of timing, not by
construction**: delete it, move the root, or freeze on a fresh clone and the result is a
contaminated "pre-stroke reference" with nothing objecting. The behavior box has its own copy and
its own `E:` root.

The durable fix is to build it from `config.prestroke_dates()` explicitly and to validate on LOAD —
a frozen artifact whose `dates` contain a post-stroke date should be refused, which is exactly the
STALE contract `prestroke_reference.load_or_freeze` already implements.

---

## 2026-08-26 — Figure 8b gets an interval, and it retracts the figure's headline

Priya: *"let's add intervals everywhere we can (as you deem appropriate scientifically)."* 8b was the
last figure with none, and it is the one that most needed one: it is the **only** measure in the
grant set that a global amplitude change cannot move, which is why it is the arbiter for every claim
about geometry.

### THE HEADLINE DOES NOT SURVIVE

`_rdm_ci` block-bootstraps the scheduler's position blocks within session, sessions held fixed, with
the leave-one-session-out ceiling resampled in the SAME draw so the delta is taken draw by draw.
On the `cue` window, `lick` class:

| | median 95% width | sessions whose change from the PRE ceiling excludes 0 |
|---|---|---|
| whole-RDM correlation | **0.278** | **2 of 27** |

The figure was being read as a positive result — post-stroke values of 0.7–0.9 against a ceiling of
0.88 look like *the geometry is preserved*. They are not. **They are undetermined.** PS93 day 1
reads 0.64 against a ceiling of 0.89 — a drop of 0.25 — with an interval of [−0.69, +0.08]. At this
trial count the whole-RDM correlation cannot separate "unchanged" from "substantially rearranged",
and 8b must not be quoted as evidence for either. Only PS95 days 1 (−0.37 [−0.71, −0.04]) and 3
(−0.20 [−0.37, −0.05]) are callable.

The **per-position rows** are the sharper instrument — five distances instead of one number pooled
over fifteen — and `close_center` is the position whose row most often excludes zero (PS92 d7,
PS94 d3/d4/d5, PS95 d1). That is the same position the lick-free control has been singling out.

This is another grant figure saying something it was not entitled to say, and again the cause was a
missing denominator rather than a wrong number. **The interval was the whole finding.**

### THE WHITENER IS NOT A FREE PARAMETER — measured, not assumed

The obvious speed-up (a 200-draw bootstrap over ~125 ms pseudo-inverses runs for four hours) is to
fix the whitener once per animal and reuse it across draws, as `_mats_crossnobis` already does for
the distance figures. **That is not available here.** Measured over all four animals, on the mean
post-stroke-minus-PRE contrast:

| whitener rule | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| per set, as `_crossnobis_within` does (SHIPPED) | −0.067 | −0.053 | −0.064 | −0.090 |
| per set, reference subsampled to matched n | −0.180 | −0.092 | −0.077 | −0.092 |
| one common whitener, pre + post pooled | −0.258 | +0.014 | −0.111 | +0.003 |
| one common whitener, PRE ONLY | −0.509 | −0.079 | −0.330 | −0.242 |

Individual sessions move by up to **0.60**. An interval computed under one rule and printed beside
an estimate computed under another describes nothing — precisely how figure 8d's
`+0.28 [+6.63, +69.41]` announced itself. So `_fast_rdm` had to be a faster ROUTE to the same
estimator, not a cheaper one: one Cholesky solve replaces a pinv (~25 ms against ~125 ms) by using
the fact that all fifteen quadratic forms share a right-hand side, and `tests/test_fast_rdm.py`
asserts agreement with `_crossnobis_within` on the same rng state to 1e-8.

**Two separate things are in that table, and only one is an artefact.**

1. *Regularisation asymmetry — an artefact.* One session's Ledoit-Wolf whitener comes from ~355
   residual rows in 380 dimensions (n < p, shrunk nearly to a scaled identity); the pooled pre
   reference has ~3900 and is barely shrunk. Row 2 equalises them and roughly doubles PS92's
   measured loss. The asymmetry is MATCHED between the PRE column and the post columns (a held-out
   single session against a pooled rest, exactly like a post day against the pool), so the contrast
   the figure endorses is protected — but the absolute values are not comparable to anything else.
2. *The noise covariance itself changed.* Residual-covariance correlation is 0.82–0.90 between
   pre-stroke sessions and 0.69–0.85 between pre and post, with PS92 the largest drop and PS92 also
   the most metric-sensitive animal. **A per-session whitener measures the post-stroke geometry with
   a ruler that deformed along with the thing being measured.** Row 4 is not the fix — whitening
   post data by a pre-only metric is biased against post by construction. Row 3 is the fair version,
   and it still moves PS92 to −0.26 and PS93/PS95 to zero.

**Kept the per-set rule** (standard RSA practice; the noise genuinely does differ per session, and
the contrast is matched) **and did not silently change the point estimates.** But 8b's absolute
numbers are a property of the metric as much as of the brain, and the honest reading is the
contrast, with the interval, and only for the two sessions where it excludes zero.

### IMPLEMENTATION NOTES

- `_rdm_scores` extracted: the per-position row correlation had been written out THREE times (8b
  inline, `_rdm_rows`, and the new bootstrap). A row that means one thing in the heatmap and another
  in the trajectory is a divergence this repo has already been bitten by.
- `N_LOO_DRAW = 4` of 11 held-out sessions per draw, halving the run. **Checked rather than
  assumed**: at n_loo 4 / 8 / 11 the delta widths are 0.80 / 0.73 / 0.71 (PS93 d1) and
  0.82 / 0.95 / 0.97 (d3) — indistinguishable. The width is dominated by the post-stroke session's
  own trial noise, not by the ceiling.
- Sessions are held FIXED, as in every other interval here, so this is **trial noise only**. It says
  nothing about how much a NEW post-stroke day would differ; the spread of the PRE ceiling across
  sessions is the figure's own estimate of that, and it is the larger of the two.
- Drawn as a shaded band on the whole-RDM trace and as a **boxed cell** on the per-position heatmap
  (an outline, not a printed number, so it survives the `--compact` variant that drops in-cell text).

---

## 2026-08-26 — The frozen ENCODER, and the amplitude-versus-tuning question finally answered

Priya: *"then consider what encoder analyses make the most sense. I like the pre-stroke then
post-stroke by session analysis structure. We don't have DLC yet."* Figure 11 is that, and it is the
first encoder figure in the grant set.

### WHY AN ENCODER, AND WHY THIS ONE

The decoder asks whether position can be READ OUT of cortex. It answers with one number per session
and, because it pools across components by construction, it cannot say what changed. The encoder
asks whether the position → activity MAPPING still holds, and its residual is a per-position object.

More to the point: it is the only framing in which the question figures 6, 7, 8 and 8b have circled
for a fortnight — **did the code MOVE, or did it merely get SMALLER?** — becomes two separately
estimated numbers rather than two readings of one. Correlations (6, 6b, 8b) are blind to amplitude
by construction; distances (8, 8d) are dominated by it. Neither can decompose.

A frozen one-hot position encoder trained on pre-stroke sessions only predicts, for a trial at
position q, the pre-stroke mean pattern at q — ridge on a one-hot design IS the per-position mean,
shrunk. So the whole analysis runs off `_collect_7`, which means figure 11 uses **the same trials,
the same engagement gate, the same classes and the same leave-one-session-out reference** as figures
6, 7, 8 and 8b. That consistency is not incidental; a sixth data path would be a sixth chance for
two figures claiming to describe the same trials to stop doing so.

Fitting ONE gain per session splits the failure (`_enc_terms`):

    raw   = 1 - S|m - p|^2 / S|m|^2      what the frozen encoder actually achieves
    a     = S m.p / S p.p                the single best gain for this session
    gain  = 1 - S|m - a p|^2 / S|m|^2    what it would achieve if allowed to rescale

Patterns are centred on the session's own across-position mean first, so a session-wide shift in F0
or SNR — which carries no position information — is charged to neither term.

**The gain is ONE number per session, deliberately.** A per-position gain would absorb exactly the
position-specific amplitude loss that IS the deficit, and the decomposition would report nothing.

### THE TRAP IN THE DECOMPOSITION, found by a test before it reached a caption

The first docstring said `gain - raw` is *the part of the failure that is pure amplitude*. **It is
not.** It is the part that RESCALING RECOVERS, and a code that is simply GONE also recovers a lot,
because the best gain collapses towards zero and predicting nothing beats predicting an unrelated
pattern. Unrelated patterns give raw = −1.37, gain = 0.01 — a difference of 1.38 that is not an
amplitude story at all. Only `gain` itself separates the two, and the reading order is fixed:

    gain high, a far from 1  -> same code, smaller (or larger). Pure amplitude.
    gain high, a near 1      -> nothing changed.
    gain low                 -> the tuning changed, whatever `a` says.

`test_gain_minus_raw_is_not_by_itself_an_amplitude_claim` pins both cases side by side so the
caption cannot drift back.

### WHAT IT SAYS (cue window, lick class, PRE = leave-one-session-out ceiling)

| animal | PRE `raw` | post-stroke `raw` | post `gain` | post amplitude `a` | verdict |
|---|---|---|---|---|---|
| PS92 | +0.57 | +0.47 … +0.61 | 0.48–0.65 | 0.85–1.42 | **no detectable change on any day** |
| PS93 | +0.63 | **−1.63 … −0.47** (d1–5), +0.42 (d7) | 0.03–0.19 | 0.12–0.35 | collapse, then full recovery by d7 |
| PS94 | +0.56 | −0.11 … +0.39 | 0.11–0.40 | 0.41–0.95 | sustained loss, partial recovery d7 |
| PS95 | +0.46 | +0.07 … +0.44 | 0.20–0.44 | 0.55–1.12 | intermediate; d4, d7, d9 down |

**The decisive column is `gain`.** For PS93, PS94 and PS95 it stays at 0.03–0.44 against ceilings of
0.46–0.67, so **rescaling does not recover the loss — the tuning itself changed.** A pure amplitude
story predicts the opposite: `gain` back at the ceiling with only `a` moved. It is not observed in
any impaired session. Note the honest qualifier that follows from the trap above: where `gain` is
this low, `a` is no longer a clean amplitude estimate either (regression dilution drives it towards
zero), so the correct statement is **"the loss is not explained by amplitude alone"**, not "the
amplitude is unchanged".

PS92 is the control case the figure needs: an animal where the encoder transfers as well
post-stroke as it does between pre-stroke days, showing the measure is not simply detecting the
passage of time.

### A BOOTSTRAP BIAS THAT SHOWED UP THE FIRST TIME IT MET REAL DATA

Several point estimates sat at or ABOVE the upper limit of their own percentile interval — PS92's
encoder PRE read +0.57 against [+0.32, +0.56]. **An interval that does not contain its own
estimate**, which is exactly how figure 8d announced itself.

The cause is general and applies to every correlation or R² bootstrapped here: resampling blocks
with replacement leaves only ~63% of a session's distinct trials in a draw, so every resampled mean
is noisier than the observed one, and two noisier means agree less. The bootstrap distribution
genuinely sits below the estimate.

`_anchor` shifts a percentile interval by (estimate − bootstrap median): **the bootstrap supplies
the WIDTH, the plotted estimate supplies the LOCATION.** Width and skew survive, and the band is
guaranteed to contain the point drawn on top of it. A pivotal interval (2θ̂ − hi, 2θ̂ − lo) corrects
the same bias but reflects the asymmetry, and where the bias exceeds half the width it returns a
band lying entirely to one side of the estimate — true to the arithmetic, unreadable on a figure.

Intervals are clipped to the parameter space, because a correlation of 0.98 shifted upward otherwise
advertises an upper limit of 1.08. Bounds apply to correlations and R² and NOT to the gain `a` or to
any difference of two correlations.

**Applied to `_rdm_ci` as well**, so figures 8b, 8g and 11 follow one rule. For the DELTAS the bias
largely cancels — day and ceiling are resampled in the same draw and both are pulled down together —
and the shift there does come out small, which is a check on the correction rather than a use of it.

### LIMITS, STATED ON THE FIGURE

**No movement regressors — no DLC yet.** A position → activity encoder attributes to POSITION
anything that co-varies with it, including how differently the animal moves to reach each spout, so
a post-stroke change in movement appears here as a change in tuning. The `lick` class, the pre-cue
window (which contains no lick at all) and the `working` class bound that differently: a difference
holding across all three is not a movement artefact, and one appearing only in the lick window
probably is. **This is the single largest caveat on figure 11 and it is on the figure, not only
here.**

### A CACHE HAZARD WORTH REMEMBERING

`_rdm_ci` anchors on `_rdm_rows`, and both memoise on `(align, variant, min_trials)`. Clearing only
one in a test left a stale point estimate from the PREVIOUS test driving this one's interval, and
the scrambled-geometry case came out as "no change" — correct code, a ceiling computed from someone
else's data. Any test that monkeypatches `_collect_7` must clear EVERY cache keyed on that tuple.

## 2026-08-26 — The joint basis loader can see the server, and why "newest" beats "local"

`publish_basis` (earlier today) put the eight joint LocaNMF bases on MICROSCOPE, byte-verified. That
was only half a fix: `joint_locanmf.load` still looked in the LOCAL directory alone, so the share had
the bases and no box could load them from it. This is the other half.

### THE FAILURE IT CLOSES

The behavior box ran the 8/24 and 8/25 analysis and could not build a single joint-basis figure —
its deck hit the completeness gate and refused — because `BASIS_DIR` resolves to a machine-local path
(`E:/joint_bases` on the helper box, elsewhere on the others) and nothing else was consulted. The
bases were not missing. They were unreachable, which looked identical from the log.

`load` now gathers candidates from BOTH the local directory and `labcams/joint_bases`, and `listing`
reports which root answered — because "the basis does not exist" and "the basis is on the share but
not on this box" are different problems with different fixes, and a listing that only sees local disk
cannot tell them apart.

### NEWEST WINS ACROSS ROOTS — NOT LOCAL-FIRST

The tempting rule is "prefer local, fall back to the server". It is wrong. `load` has always meant
*the newest saved basis*; preferring local wholesale would silently serve a superseded reference frame
on whichever box happened to be behind. That is the frozen-model contamination shape again — an
artifact whose name asserts a currency that nothing checks — and it would be harder to catch here,
because a basis mismatch does not change a number's plausibility, only its meaning.

So candidates are pooled and sorted by `built_utc`, exactly as before, with the search widened. Both
directions are pinned by test: a newer server basis beats an older local one, AND a newer local basis
beats an older server one, so the test cannot pass by always choosing one root.

### THE ONE PLACE LOCAL IS PREFERRED, AND WHY IT IS SAFE

When the same `basis_id` appears in both roots — the normal state after publishing — the local copy is
used. A `basis_id` is a hash of its own inputs, so two directories carrying the same id necessarily
hold the same basis; this is not a judgement call about which is better, it is a choice between two
copies known to be identical. VERIFIED, not assumed: `A.npy` for PS92/76d884873920 is SHA-256 equal
across the two roots (`26a8fbc7180fb11d…`), so the preference costs nothing but a 180 MB SMB read.

### WHAT STAYS LOCAL

`build` still writes locally, and a missing basis still RAISES rather than refitting. The module
exists to stop a silent refit — "a refit over a grown session set is a DIFFERENT reference frame" —
and a server fallback must not become a back door to that. The error message now names both places it
looked and points at `python -m wfield_local.publish_basis`, so the fix is discoverable from the
failure instead of from this file.

### MEASURED, on the real share

With `WIDEFIELD_JOINT_BASIS_DIR` pointed at an empty directory — the behavior box's exact situation —
`load('PS92')` returns `76d884873920` (11 sessions, ncomp 95) from MICROSCOPE, `A` reads as
(540, 640, 95) float32 and `signal('PS92_0606')` returns (95, 143788). The non-finite entries in `A`
are the off-brain mask, present identically in the local copy; they are data, not a bad network read.

A half-copied basis directory — publishing is a file-by-file copy, so one can exist mid-flight with an
unreadable manifest — is skipped rather than raising, and does not shadow a good basis. Tested.

## 2026-08-26 — A coverage gap was being reported as a licking failure (PS95_0813)

`_frames` excludes cues that fall outside the imaging coverage by marking their frame `-1`. That part
was right. What was wrong is what happened next: `_trial_features` let those trials fall through to
`precue_window_start`, which computes `fixed = -1 - win_n < 0`, returns `None`, and lands the trial on
`n_dropped_dirty` — the LICK-FREE counter.

### MEASURED, on the session that exposed it

    PS95_0813   total cues                                        871
                outside the imaging coverage                      197
                genuine lick-free drops                             1
                kept (587 engaged + 86 no-lick)                   673

The log therefore announced *"dropped 198 trial(s) with no lick-free 2 s window"* — 23% of the
session — when the true lick-free exclusion rate is 1/674 = 0.15%. Its neighbours PS95_0812 and
PS95_0814 report no lick-free drops at all, so the session looked like a behavioural outlier and was
not one. The cause is an imaging gap (a repaired single-channel prefix), which `_frames` had already
said in the line above.

### THE KEPT SET NEVER CHANGED — which is why it survived two weeks

Both paths drop the same trial: coverage-excluded cues were reaching `ref0 < 0` or `None` and being
skipped either way. Every number computed from the features was correct before this change and is
byte-identical after it (587 + 86 = 673, verified on PS95_0813, and PS95_0812/0814 unchanged at
720/633). Nothing about the reference band moves.

What was wrong was the REASON GIVEN, on a pre-stroke session that feeds the reference band. This is
the same shape as the frozen models trained on post-stroke data and the "curated" dates that meant
pre-stroke only by historical accident: a label asserting a cause that nothing checks. Here it was
pointing an exclusion at the animal's behaviour when the cause was a gap in the imaging — the kind of
error that does not corrupt a number but does corrupt what someone concludes from it.

### THE FIX, and what the test pins

Coverage-excluded cues are now dropped explicitly, BEFORE the lick-free accounting, with their own
counter and their own log line that says the count is excluded from the one below it. The ordering is
what matters and is what the test pins: behind the `precue_window_start` call the trial is still
dropped but still misattributed, and a test that only checked the counter existed would pass on a
broken build.

---

## 2026-08-26 — Figure 10's ceiling was 100% by construction, and it hid a pre-existing deficit

Priya, reading the first render: *"pre-stroke number is '1' but there are 11 sessions right?"*

Yes. `_match_tables` took the argmax of `_matrices_pattern`'s `"PRE"` entry, which is the **mean of
the eleven leave-one-out matrices**, so each row could contribute exactly one count. Every PRE panel
read **100% match self, in all four animals** — printed directly beneath a caption instructing the
reader *"read the middle panel against it, never against 100%."* The figure contradicted its own
instruction, and nothing in the render log could have said so.

**Averaging is not the same as counting.** Eleven averaged matrices have had their per-session noise
removed; the post-stroke columns each still carry theirs. The comparison was never like for like, so
the post-stroke scores (56–90%) were being measured against perfection. Same family as figure 6's
half-split, 8b's old ceiling and `_delta_diag_ci`'s `mats_fn(ref, ref)`: **a baseline that is not
subject to the noise it is the baseline for.**

`_pre_loo_matrices` now exposes the eleven matrices and `_match_tables` counts one per held-out
session, exactly as the post panel counts one per day. Each panel prints its own `n`, because the
cell totals are no longer on one scale (11 versus 6–7) and only the percentage is comparable.

### THE CEILING WAS LOAD-BEARING, AND IT CHANGES A READING

| animal | PRE ceiling | post | the row that moved |
|---|---|---|---|
| PS92 | 100% | 90% | — |
| PS93 | **92%** | 56% | **close_center matched itself only 7 of 11 PRE-STROKE sessions** (2 → close_L, 2 → far_C) |
| PS94 | 98% | 59% | far_R 10/11 |
| PS95 | 97% | 78% | close_center 10/11, close_R 10/11 |

PS93's `close_center` substitutes on **every** post-stroke day in figure 10b (cR, cR, fR, cR, cR,
cL — never itself). Against a 100% ceiling that reads as a total lesion-induced collapse. Against
its real 7/11 ceiling, a large part of it is **pre-existing instability in that animal at that
position**. This is the third time `close_center` has turned out to be the position where a control
bites, after the reliability collapse recorded on 25 Aug. **close_center must not be quoted from
figure 10 or 10b without its PRE column beside it.**

## Figure 10b — the per-session view

Priya: *"make a version that shows the matching matrix for each session over the post-stroke course
(like our other first-column pre-stroke, subsequent columns post-stroke sessions)"*.

Rows = position, columns = PRE then each post-stroke day, in the grammar of 5c/6b/8b. **Text** = the
position matched best. **Colour** = the RANK of the true position among six, because a cell reading
`fL` means one thing when the correct answer ranked second and another when it ranked sixth, and the
text alone cannot say which. **Boxed** = still matched itself, so the intact diagonal survives
`--compact`, which drops the text. PRE collapses the eleven held-out sessions: colour = mean rank,
text = modal match with the fraction that agreed.

What it shows that the pooled figure cannot: **persistence**. PS95's `far_C` matches `far_L` on
every day but one; PS94's `close_center` alternates. Pooled counts cannot distinguish a substitution
that held all week from one present on a single day.

### PER BLOCK — asked for, and not directly possible

Priya: *"can we do this on a per-block rather than per session basis?"*

**A 6x6 match matrix needs all six positions.** A scheduler block is ~6 trials at ONE position, so a
block cannot produce a matrix at all. The nearest honest unit is a time-contiguous CHUNK spanning
blocks at all six positions — split each session in two or three. That multiplies the units and is
the only construction here that would expose WITHIN-session variability, which nothing currently
shows. It needs a per-chunk trial-count gate: a third of a session at an impaired position will
often fall under `min_trials` and blank the row, which is the same estimator threshold that empties
whole days in 8g. Not built.

## 2026-08-27 — `_trial_features` is cached to disk, and why the key is the whole change

`_trial_features` is the per-session workhorse every downstream analysis is built from, and it had no
memoisation at any level. Measured: ~5 rebuilds per distinct session in a grant render, with the
uncached calls accounting for roughly 6 of the nightly's 9.62 h on top of most of the 8-10 h render.
The `lru_cache`s that existed sat one level too high, at the collectors, so each collector
independently rebuilt the same per-session features.

### DISK, NOT `lru_cache` — the detail that decides whether it works at all

The nightly is **17 separate processes**; `cli()` shells out to `python -m <module>`. An `lru_cache`
is discarded at every step boundary, so the obvious choice — and it is the obvious one, since every
existing collector cache uses it — would have fixed the grant render and left the nightly at 9.6 h.
`session_cache.cached` is the only tier that crosses a process, and it also carries results between
NIGHTS: per-session feature extraction does not depend on which other sessions exist, so yesterday's
sessions never need rebuilding. That is where most of the win is.

### EVERYTHING RESULT-CHANGING GOES IN `kind`, NOT `params`

`session_cache.cached` prunes with `glob(f"{lab}__{kind}__*.pkl")` after every write. Two callers
sharing a kind but differing in params would therefore **evict each other on every single call** — a
cache strictly slower than none, which still looks like it is working. Folding into the kind is what
the two existing users already do, and this is why.

### THE DISCRIMINATOR THAT IS NOT IN `session_signature`

The same session and the same args return **completely different features** depending on whether the
signal is that session's own LocaNMF fit or a projection onto a shared joint basis — 256 feature
columns against 380, measured on PS92_0602. `session_signature` stats `locanmf_C.npy`, the h5 and the
behaviour table; it stats neither the basis nor anything that moves with it. A key without the basis
id would serve joint features to the per-session path and back again — silently, across processes,
persisting for days. `feature_cache_kind` folds in `signal_key`, plus the RESOLVED `bins` and
`lickfree`, which come from `defaults.yaml` and move no mtime the signature looks at. (`decode.max_rt_s`
going 2.0 -> 3.5 s on 2026-08-21 invalidated every number computed before it. That is this failure
mode, already realised once.)

### TWO REFUSALS, BOTH FAIL-SAFE

`_trial_features` feeds the decoder, encoder, RSA and cross-mouse, so a key bug is wrong numbers
everywhere rather than a failed render. Anything the key cannot describe is computed instead:

* **an injected signal with no `signal_key`** — the array cannot go in a key (it is the expensive
  thing being avoided; hashing ~100 MB per call would cost more than the rebuild), so a caller that
  injects without saying where it came from gets a correct uncached answer rather than a fast one.
* **`source` other than `locanmf` with no injected signal** — `_build_signal` then reads `U_atlas.npy`
  and the SVTcorr, which the signature does not stat. Those sources are diagnostics, so leaving them
  uncached is honest where widening a signature every other cached kind depends on would not be.

### THE PROJECTION IS DEFERRED BEHIND THE CACHE

Both joint feature builders used to project first and build features second. Once the features are on
disk that ordering wastes the entire saving. `joint_locanmf.BasisSource` makes the signal a callable
invoked only on a miss, so a warm cache never touches the basis. `variance_captured` is cached
separately as one float, written on the cold pass, so the diagnostic survives a hit without a
projection performed purely to report it.

### VERIFIED — and note what the OBVIOUS A/B cannot catch

The natural check is `WIDEFIELD_NO_CACHE=1` against a cached run of one figure. It **cannot detect the
failure that matters**: run entirely on the joint path it uses one provenance in both arms, so it
passes on a build that cross-serves. The check run instead does two things:

    PS92_0602, all three alignments      cached == uncached, element-wise
      cue     OWN  X(243, 256)   JOINT X(243, 380)   both exact
      precue  OWN  X(142, 256)   JOINT X(142, 380)   both exact
      lick    OWN  X(243, 512)   JOINT X(243, 760)   both exact
    own != joint on every alignment      the key DISCRIMINATES, so neither was served for the other
    with_indices 6-tuple vs 8-tuple      the shape-changing kwargs do not collide
    nolick_ref moves the no-lick arm     placement is in the key

CACHE_VERSION 10 -> 11. 800 tests pass.

### ONE THING TO EXPECT IN THE LOGS, so a quiet log is not misread

`[precue lick-free]`, `[coverage]` and `[behavior-backup]` print from inside `_trial_features`, so a
cache hit prints nothing. Their counts will drop sharply — `[behavior-backup]` from 56 to 4,
`[coverage]` for PS95_0813 from 53 to 1. That is the cache working, not the dead-strobe-bit repair
having stopped, and a quiet log is exactly what a broken repair would also look like.

## 2026-08-27 — The frozen decoder and encoder become OBJECTS, not recipes

Until now nothing in this repo persisted a fitted frozen model. `pooled_frozen_loso` fitted `_pipe()`
inline on every call, `pooled_frozen_encoder` fitted its Ridge inline, and every JSON on the server
held RESULT NUMBERS — accuracies, confusions, EV — with no weights behind them. `save_session_decoder`
is the one exception and nothing ever loads its `.joblib` back. "The frozen pre-stroke decoder" was a
recipe re-executed nightly.

### THAT IS WHY THE CONTAMINATION WAS POSSIBLE

On 2026-08-26 the frozen models were found to have been training on post-stroke sessions — roughly
30–39% of the training data by then — because the code was written when "curated" meant pre-stroke and
post-stroke nights joined the pool silently. **A model refitted every run has no identity to
interrogate.** There was nothing on disk to ask "what were you trained on?", so the drift was
invisible until someone recomputed the number by hand. A stored model carries its own answer, and
adding a session cannot retroactively change what an already-frozen model saw. This is a bug-class
removal first and a time saving second.

### WHAT DETERMINES A MODEL'S IDENTITY — and what does not

    animal, kind (decoder|encoder), align (precue|cue|lick), source (roi|locanmf),
    basis_id, post_s, zscore, alpha, n_features,
    train_labels  + a per-session INPUT SIGNATURE for each

**Trial inclusion is deliberately NOT in the spec**, and this is the part most likely to be
mis-assumed. Training always uses the pre-stroke ENGAGED trials (`XE`); "all", "lick + miss-while-
working" and "lick only" select which POST-STROKE trials are pushed through the finished model. They
are scoring-time populations, not training-time variants. So the inventory is **4 animals × 3
alignments × 2 sources = 24 decoders and 24 encoders**, not 72 of each — and because every population
is scored by the SAME model, per-class results stay summable, which is the property
`position_coding_directions` already relies on for its raw-count confusions.

The input signature matters as much as the label list: on 2026-08-14 the switch to the meegkit_hpfit
SVTcorr changed the underlying data while every label stayed identical. A model keyed on labels alone
would have been served for data it was never fitted to.

ROI and joint models remain separate artifacts by construction — 264 features against 380, with no
mapping between them — and `source` + `basis_id` in the spec make cross-serving impossible rather
than merely unlikely.

### ONE ARTIFACT HOLDS BOTH ARMS

`{"full": …, "loso": {held_out_label: …}}`. The all-pre model scores post-stroke days; the
leave-one-out models are the pre-stroke reference band. They are determined by the same spec, and
splitting them would let one be re-frozen without the other.

### A MISMATCH IS NEVER RESOLVED SILENTLY

A changed pre-stroke set mints a NEW model under a new id; the old one is left untouched and the
change is reported by name ("added PS92_0830"). Refitting in place would move a reference that
post-stroke results are already quoted against; reusing blindly would score today's data against a
model built from data that is gone. `refreeze='<reason>'` is the deliberate supersede path and
RENAMES rather than deletes, matching the convention already on disk for the nolick reference.

### VERIFIED: freezing changed no number

Freezing changes WHEN the fit happens, never WHAT is fit, so a run that fits-and-stores and a run
that loads must agree to the last digit. Both were run through the real `pooled_frozen_loso` /
`pooled_frozen_encoder` with the store first emptied, and compared on `loso_accuracy`, `mean_within`,
`transfer_cost`, every per-session accuracy, and the encoder's `mean_ev` / `mean_feve` /
`transfer_cost_ev` / per-session EV.

### STILL A RECIPE, and named so it is not mistaken for done

`grant_figures`, `nolick_decoder`, `ood_control` and `poststroke_compare` each still fit their own
pre-stroke model. They agree with these today because they share `_pipe()` and the same trial
conventions, but nothing enforces it. That is the next step, not this one.

## 2026-08-28 — The no-lick "pre-stroke reference" was neither, in two independent ways

`nolick_reference_prestroke.json` calls itself, in its own `kind` field, the frozen PRE-STROKE
no-detected-lick reference. Two separate things were not pre-stroke.

### THE MODEL — the third instance of the same contamination

`analyse_animal` fitted `clf = _pipe().fit(XE, YE)` over every pooled session, and ran the engaged
arm's leave-one-session-out over all of them too. This module exists, in its own docstring, as *"the
pre-stroke reference for reading post-stroke failed trials"* — and the model reading them was being
trained on them. The position-matching target (`eng_frac`, the engaged position profile every other
arm is matched to) also spanned all phases.

That is the same class as `pooled_frozen_loso` (fixed 2026-08-26) and `ood_control` (fixed
2026-08-28). Three instances now, all with the same cause: code written when the cohort was
pre-stroke only, and post-stroke sessions joining a list silently.

Training is now restricted to pre-stroke; the engaged arm is leave-one-out among pre; the matching
target is the pre-stroke profile. **Pooling is unchanged** and still spans every session — it is what
reconciles the feature columns and makes post-stroke rows comparable at all. Post-stroke sessions are
still SCORED, by a model that never saw one. That is the measurement. The result now carries
`training_phase`, `pre_labels` and `post_labels`, and refuses outright below two pre-stroke sessions.

### THE ARTIFACT — a guard that could only ever say no

The freeze was `if not frozen.exists()` plus, since 2026-08-26, a refusal when the live reference
covered post-stroke dates. The refusal was right. But it was built from `from_list`, which now ALWAYS
contains post-stroke dates — so the guard could only ever refuse. Every night logged a refusal, and
the only pre-stroke reference in existence stayed the one written on 2026-08-19, which is clean by
accident of timing. **A guard that can never pass is not a mechanism.** It also meant the behavior box
could never mint one at all.

`build_reference(phase="pre")` now restricts the date list itself, so the artifact matches its name by
construction. The post-stroke check survives as a second line rather than the only one, and it now
interrogates the PRODUCED artifact's own `dates` rather than the caller's request — asking for
pre-stroke and trusting compliance is the same category of mistake as `exists()` standing in for "is
pre-stroke". A bad artifact is RENAMED to `.REFUSED_contaminated.json`, not deleted.

### WHY IT COULD NOT SIMPLY LOAD THE FROZEN DECODER

It looks like `pooled_frozen_loso` — same `_pipe()`, same z-scoring rationale, ROI or joint basis —
but it is three ways different, and only one of those is a bug. Its engaged cut is **2.0 s** (or the
session's own response window), deliberately, because that is what keeps the `late_rewarded` arm
addressable at all; with both cuts at 3.5 s that arm is empty by construction and the
late-versus-undetected distinction disappears — a real result (PS93 8/12: the entire pre-cue survival
sat in the LATE arm, balanced 0.532, p=0.003, while undetected showed nothing, 0.153, p=0.76). It also
builds features with its own `session_features`, whose pre-cue window is not lick-free. Different
training rows and a different feature space mean a different model, so it correctly keeps its own —
it just has to be pre-stroke, which it now is.

### FOR THE DECK'S DERIVED CAVEAT

The other window is having the deck mark D2/G6 by reading the artifact rather than hardcoding a
warning, so the caveat disappears on its own once a clean artifact lands. Keying on
`dates ∩ poststroke_dates()` would now over-fire: the LIVE reference legitimately spans both phases —
that is the comparison — while its model is pre-stroke-only. `training_phase` is the honest
discriminator, and is written for exactly that.

## 2026-08-28 — The deck's slide budget, measured, and four consolidations

Priya: *"I'd like to consolidate the powerpoint so it isn't several hundred slides."* Before proposing
anything, the deck was measured — a static scan cannot answer where the slides are, because most
figure names are built with f-strings and `f"{stem}.png"` matches every file. So a real build was
instrumented at `add_picture` and the slides counted by section.

### WHERE THE SLIDES WERE (published deck, 8/27: 712 slides)

    section D   155   one slide per 4 held-out days, per animal, per alignment, per basis
    G3          112   one slide per post-stroke session, per alignment, per arm
    G9 + G9c    206   coding directions, per animal x alignment x method
    H            98   grant figures
    everything else

### FOUR CONSOLIDATIONS, 712 → 521 SLIDES

**Section D, 155 → 40.** `write_animal_confusion_grid` puts every held-out day for one animal on one
figure — a 6-wide grid of small confusion matrices, post-stroke dates in red. The per-day PNGs are
still written and the deck still falls back to paging them if the grid is absent. Dropped from the
grid deliberately: the per-position recall bars (they have their own summary two slides later) and
the in-cell numbers, which at ~1.5 in are too small to read and compete with the colour that is not.

**G3, 112 → 32.** Four sessions per slide in a 2×2, grouped by ANIMAL first so a slide never
straddles two — the earlier bug here was a title naming only the date, which made PS92_0818 and
PS93_0818 two slides with identical headings.

**G8d, +8 and a bug fixed.** `fixed_scale_maps` paginates at four post-stroke days and part 1 keeps
the historical filename, so the deck's glob matched page 1 alone: the LATER post-stroke days — the
ones the recovery story is about — were written nightly and shown nowhere. One slide per animal is
not available here and it is worth recording why rather than re-attempting it: nine rows by six
positions of square maps has a natural aspect of 1.5, and a slide with a title leaves 0.48.

**D2, +4.** `nolick_per_session_*.png` had been written every night and embedded nowhere.

### THE AUDIT METHOD, because it is reusable

Instrumenting `add_picture` on a real build and diffing against the figures directory found **1229 of
2336 PNGs never placed**, in 333 families — of which 79 were current rather than superseded. Two
weaker methods were tried first and both failed: grepping for distinctive stems reported everything
present (any 4-character chunk matches something), and converting f-strings to regexes reported
everything covered (`[^/\]*\.png` is a wildcard). Only the instrumented build is sound.

### RELATED, from the other window and recorded here so it is found

`_save` prints `bad[:6]`, so **every layout-fault count taken from a log is a floor, not a count** —
their 5c "6" was really 40, and the 204-fault report of 2026-08-26 was a floor too. Measure with
`_overlaps(fig)` directly.

### THE COMPACT GRANT VARIANT IS GONE

It cost a second full render pass — roughly half the total — for panels 3% narrower (13.2 in against
13.6 in). It was built on the assumption that in-cell numbers forced the panels wide; the six rotated
TICK LABELS set the floor and are present in both variants. `_txt` survives as a seam even though it
now gates nothing, because eight call sites routed through one function is how the next global change
to in-cell text stays a one-line change.

### 2026-08-28, later — figure geometry sized for the slide it lands on

`coding_cross_` was **22.7 in** wide and `coding_direction_` **19.6 in**, both placed at 12.7 in. A
reader sees `fontsize × (placed width / figure width)`, so their 7 pt and 6.5 pt labels arrived at
**3.9 pt and 4.7 pt**. Reshaped to 10.8 in and 12.2 in with the type enlarged: fewer inches and larger
points, so the figures are physically smaller and MORE legible. `crosssess` got the same treatment
(6 sessions per row instead of 4, one shared colour bar instead of one per panel — every panel already
shared vmin/vmax, so N of them asserted a scale that does not exist while taking width from the maps).

**Two animals per slide only where the width allows it**, measured rather than assumed: `pooled`
(11.5 in) and `normunit` (12.0 in) survive 2-up; the dense kinds in the same loop are 16–23 in and
would land at 3–4 pt, so they are explicitly excluded via `_G9_PACK_2UP`.

**The caveats travel with the packed slide.** The per-animal path appends the lick-window inference
note and the plain-vs-orth pair note to each blurb; the packed path initially did not, which would
have made consolidating the deck a way of quietly stripping the reasons its numbers are conditional.

### Deck state: 712 → 501 slides, 20 section headers, 98% carrying notes

### PLAIN vs ORTHOGONALISED — kept, and why (Priya asked 2026-08-28)

Orthogonalising projects the lick/no-lick engagement axis out of the position direction. It is needed:
`cos(w, engagement)` reaches 0.82 / 0.91 / 0.71 / 0.52 across PS92–95 and lands on a different
position in each animal, so PS93's far_center "position" direction is 91% engagement axis. But the
2026-08-24 audit against logistic directions shows orth moves TOWARD that reference in PS93/94/95 and
AWAY in PS92 (cue 0.160→0.192, lick 0.202→0.279). **For PS92, plain is the better estimate**, and in
the lick window orthogonalising also removes position-linked MOVEMENT, not only confound. Both stay.


---

## 2026-08-28 — EARLY vs LATE rewarded trials, and a prediction the data refuted

`decode.max_rt_s` moved to 3.5 s on 2026-08-21 — the task's real response window — because the
no-lick arm was holding rewarded hits (39.3% of it for PS92, 33.9% for PS93). That was right, and it
put a 0.2 s lick and a 3.0 s lick in one ENGAGED class, so the class every decode number is computed
on could change composition after the lesion with no figure able to show it.

`position_coding_directions._class_confusions` now splits `poststroke_lick` at a **fixed 2.0 s** —
`nolick_decoder`'s cut, and what `max_rt_s` was before 8/21. **Not the session median**: that is
session-relative, so "late" would mean a different thing on every day and be comparable neither
across sessions nor against the `late_rewarded` category the no-lick reference already defines.

**The split is a PARTITION, not two new siblings.** `poststroke_lick_early`/`_late` live in a new
`CONFUSION_SUBCLASSES` mapping and are deliberately kept OUT of `CONFUSION_CLASSES`. That tuple's
whole invariant is that summing a subset gives a population; a fourth and fifth sibling would have
made "all trials" count every lick trial twice, silently, and the result would still have looked like
a confusion matrix. Same mask, same frozen model, raw counts, so `early + late == poststroke_lick`
cell for cell — asserted in the code and verified element-wise on all 12 animal-windows.

### The measurement contradicted the premise, including the note written for it

| | PS92 | PS93 | PS94 | PS95 | all |
|---|---|---|---|---|---|
| late (RT ≥ 2.0 s) | 5.6% | 7.5% | **1.0% (26 trials)** | **0.8% (27 trials)** | **3.4%** (1029/30645) |

The design — and the deck note, the slide subtitle and `RT_SPLIT_S`'s own comment, all of which I
wrote — asserted that post-stroke the mass moves late. It does not. **That is a result, not a failed
figure**, and it agrees with the rest of section G: the two most impaired animals do not lick
SLOWLY, they lick fast or not at all. Slowed execution would have filled the late arm; "the missing
licks are the phenotype" predicts exactly this, and the no-detected-lick arm is where those trials
went. All three claims were corrected and a test fails if the prediction returns without the
measurement beside it.

**PS94's and PS95's late panels cannot be read and say so in red.** 26 trials over six positions is
~4 per row, where a per-position recall is 0.0 or 0.33 by arithmetic rather than by measurement. A
panel below `len(positions) * MIN_TRIALS` is titled TOO FEW TO READ; recall points below
`MIN_TRIALS` are hollow and carry their n; below `FLOOR_TRIALS` none is drawn, because plotting 0.0
on two trials looks measured. Both floors are the ones `_stats` and `_cells` already use.

Deck: **G9e**, one animal per slide (6×6 ticks at 7pt reach the reader at ~7.4pt placed at 11.0in,
and 4.2pt if packed 2-up). D2's response-window arm is **unplaced** — its question stands, but G9e
answers it directly, since the trials that move between the two cuts ARE its late class. Figures
still written, so it is reversible; the note records that respwin uses each session's own window
against G9e's fixed 2.0 s, so the populations are close but not identical.

`CACHE_VERSION` was **not** bumped. `with_rt=False` returns a tuple byte-identical to the old one, so
the key omits the flag when false and every warm entry stays valid; bumping would have discarded
~1 MB per session-alignment at ~168 s each *and* invalidated every unrelated cached kind, to record a
value none of them can see. `CACHE_VERSION` is for when the COMPUTE CODE moves.

---

## 2026-08-28 — Six recipes for one frozen decoder, three for one discriminator, four for one cut

Four de-duplications, all of the same shape: several places computing "the same" object and agreeing
only by coincidence. **That is not a tidiness problem.** It is how the training contamination fixed
on 2026-08-26 survived eight days — whichever copy you read looked defensible on its own.

**The frozen decoder (`73b651d`).** `pooled_frozen_loso` froze it; `poststroke_compare` rebuilt the
same two models five more times with a bare `_pipe().fit(XE[tr], YE[tr])`. `poststroke_section_g`
mutates only `d["post_i"]` and calls back in, so a fit that does not depend on `post_i` at all was
redone once per post-stroke session per arm per alignment. `frozen_decoder_models()` now owns the
spec and the fit and both modules call it. **The acceptance test is equivalence**, not that the call
happens: the frozen path predicts identically to the old inline fit, and `models["loso"]` reproduces
`cross_val_predict(LeaveOneGroupOut)` fold for fold. Keyed by LABEL, not pooled index — the index
depends on the order the caller assembled its pool. `_pooled` now calls `config.pooled_labels` rather
than rebuilding the list: for ROI features `_align_many` intersects region×bin columns across the
pool, so a different pool is a different `n_features`, a different spec id, and a permanent cache
MISS — a failure that looks like slowness, not an error.

**NOT substituted, and must not be:** `decode_matched`'s lick-only arm (class-filtered to preserved
positions — 4-way for PS94/PS95, a different chance level) and `_within_accuracy` (a within-session
ceiling). Both pinned.

**The lick discriminator (`c63ff54`).** Three copies, and **the coincidence was already broken**:
`looks_like_which` and `undetected_state_split` each start a fresh `RandomState(seed)` and draw the
same sample, but `fits_engaged_distribution.balanced_fit` shares one generator across its
leave-one-out loop, so its full-pool fit is a DIFFERENT sample from a function that reads as though
it were the same. `balanced_lick_sample` takes the rng as an ARGUMENT so every call site keeps its
exact draw — the refactor moved no number, asserted draw for draw. Making `balanced_fit`
deterministic per fold is defensible and would move published numbers: a decision to take on its own,
not a side effect of removing duplication. It is **not** the position decoder — same
hyperparameters, two-class label space against six positions — so anything keyed on a stored model
needs `kind="lick_discriminator"`, never `kind="decoder"`.

**The engaged cut (`b93fbc0`). There were FOUR modules, not three.** `postcue_window_test.py` was
missed by the by-hand survey and found on the first run of an AST walk. It is where the literal did
the most damage: that module sweeps the post-cue WINDOW over [2.0, 2.5, 3.0, 3.5] s, so the longer
windows were scored on trials selected by the shortest one — window and trials disagreeing by up to
1.5 s. All four now read `decode.max_rt_s` **and announce the cut at run time**, because fixing only
the code moves the disagreement into the docs: every number recorded for these modules was measured
at 2.0 s. `nolick_decoder`/`nolick_analysis` are exempt and a test asserts the exemption still
carries its reason — an exemption that stops explaining itself is indistinguishable from the bug.

---

## 2026-08-28 — Figures rendered nightly and shown to nobody: measure, do not grep

**14 placed; 496 → 525 slides.** Measured by instrumenting `add_picture` on a real build. 739 of 2430
PNGs on disk are placed and most of the rest SHOULD stay unplaced — superseded dates, non-curated
variants. Two cheaper methods were tried and both reported full coverage wrongly: grepping
distinctive stems, and converting the deck's f-strings to regexes (`f"{stem}.png"` becomes
`[^/\\]*\.png`, a wildcard). **Only the build knows.**

- **G2c lick-only arm.** The all-trials with-control grid carried the headline since 2026-08-20
  while its sibling was written nightly and never shown. It is the control for the standing
  objection — that the dissociation is about missing MOVEMENTS rather than coding — since every
  trial it scores has a lick. The cost is positions: an abandoned position drops out and chance
  moves 6-way → 4-way. Neither arm is complete, so showing one silently was choosing a result.
- **G2d, the small-lesion family (9).** `section_g_figures` renders the whole readout family twice,
  and the second copy had never reached a slide — only the two grey squares inside G2c stood for it.
  PS92/PS93 on 8/17 after a laser that did not take is the strongest control this design has, and
  **what it should show is NOTHING** — which cannot be said of a figure nobody has looked at.
- **Joint-basis health, all three alignments.** The span is computed on the ALIGNED window, so the
  three are different numbers. The deck showed the pre-cue one and said in its own subtitle that the
  others "are not shown here" — an accurate note about an incomplete slide.
- **Hemisphere-resolved RSA (2).** The one RSA arm explicitly about the lesion's SIDE, placed on
  none of the three slides it belongs beside. Put in **F, not G**: it is a PRE-stroke geometry
  measurement, the reference a post-stroke asymmetry must be read against; in G it would read as a
  result about the stroke.

**Deliberately still unplaced:** `coding_cosslope_*_dom` and `coding_pairsplit_*_dom`. G9b shows the
ORTHOGONALISED variant alone because both cohort diagnostics were measured on it; drawing the plain
ones beside them would put two different measurements under one claim.

### A missing figure is an absence, not an error — which is why it needs measuring

`figure_cross` indexed `axes[0][k]` after its panels were reshaped to two rows of three, and failed
16 times in one run. It hid twice over: **the lick window never fails** (2 classes → `_nr` is 1, so
row 0 IS the grid), and **`_draw` caught it** — that wrapper exists so a plotting bug cannot discard
40 minutes of pooling, and it did its job. The cost is that the failure presented as missing figures.
Recovered by redrawing from `coding_direction.json` with no re-pooling, since `run_animal`'s result
is exactly what is persisted. Third layout fault this month found by DRIVING the function rather than
reading the diff.

---

## 2026-08-28 — A logical-root override must reach every consumer (third-machine readiness)

`WIDEFIELD_FIGURES_WORKING` has been documented in `configs/paths.yaml` since 2026-08-11 as the way
to correct a machine whose profile resolves `figures_working` wrongly. **It was read in exactly one
place** — `nightly_figs._default_out` — while **19 modules** called
`PathResolver().root("figures_working")` directly and never saw it: every standalone figure CLI, and
`joint_locanmf.BASIS_DIR`, which derives the joint-basis directory from that root. Setting the
documented variable fixed the nightly and silently fixed nothing run by hand, nor where the basis was
looked for — and for the basis a wrong answer is not a missing figure but a WRONG FEATURE SPACE (the
same session gives 256 columns vs 380).

The override now lives in `PathResolver.root`, derived generically as `WIDEFIELD_<ROOT_NAME>` so a new
logical root gets one without a branch. `nightly_figs` keeps its loud failure and gives up the
override.

**This matters for any THIRD machine.** `detect_machine` knows three profiles — `analysis`,
`imaging`, `mac` — and falls back to `analysis`, so a new box adopts another machine's local paths by
default. The failure that fallback causes is on record: a box with the imaging profile's mounts doing
analysis work sent every figure to the other machine's `C:/Users/sabatini/...` path and built a deck
with 80 slides and 287 missing figures, **exit code 0**.

### What a third box can and cannot do (audited 2026-08-28)

**Present and sufficient:** all 74 pooled sessions have `SVTcorr.npy`, `allen_aligned_*/U_atlas.npy`
and LocaNMF outputs on MICROSCOPE — so a new box does **not** need to run LocaNMF for existing
sessions, only for new nights. Joint bases for all four animals are on the server
(`labcams/joint_bases`), reachable by the 2026-08-27 server fallback. The repo is config-driven and
carries no machine-specific paths (`test_no_hardcoded_machine_paths`).

**Gaps a new box hits:**
- **No machine profile.** Needs `WIDEFIELD_MACHINE` and `WIDEFIELD_FIGURES_WORKING` set explicitly,
  or it silently adopts another box's local paths.
- **Deck source.** `--src` defaults to the LOCAL `figures_working`, empty on a fresh box. Build from
  `cue_analysis_out` on the server, or run the analysis first.
- **GPU stack is not pip-resolvable.** torch (CUDA), `wfield==0.6.0`, LocaNMF/localnmf and cuhals are
  a custom Windows build with patches in `wfield_local/*.patch`. This, not data availability, is the
  real blocker for *running* LocaNMF on a new machine.
- **Cold `session_cache`** → the full ~9.6 h analysis stage rather than the cached path.
- **`frozen_models` on the server holds PS92 only** (decoder + encoder, cue). Everything else refits
  as `frozen-new` — correct, deterministic, and slow.
