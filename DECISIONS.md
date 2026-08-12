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
