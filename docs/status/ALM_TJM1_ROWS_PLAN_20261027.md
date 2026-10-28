# ALM / tjM1 / MOs_rest / MOp_rest rows in the encoder + FaceRhythm area figures (Priya 10-27)

Request (Priya 10-27): add ALM / tjM1 / MOs-rest / MOp-rest area rows to the encoder and FaceRhythm area figures on the
goal page (page ids below). ROIs = `scripts/iti_licks.py` `roi_masks` at ROI v8 (ALM = anterolateral 40 % of MOs, tjM1 =
anterolateral 50 % of MOp); rests = parent minus sub-ROI per hemisphere. Shared code: `scripts/encoder_split_rows.py`.

**ROI choice for the encoder rows (decision to confirm with Priya):** the v8 ROIs are intersected with the encoder's 16-px
stat mask, the mask every existing encoder area value is computed on. Then ALM + MOs_rest = the encoder's MOs and tjM1 +
MOp_rest = its MOp exactly (SSE / SST add; evoked values add with pixel-share weights). ~11 % of the v8 ALM / tjM1 pixels
(the thin lateral rim and ALM's front line) are outside the stat mask and are not in these rows. The frozen basis runs
only cover stat pixels, so the rim could not be included there anyway; the per-session block maps do cover it (an
unrestricted variant is a one-line change in `encoder_split_rows.masks` for the FaceRhythm figures only).

## Step 1 -- cost table

| page fig | script | where the area numbers come from | class | cost of new rows | done here |
|---|---|---|---|---|---|
| 22.3, 22.4 | `fr_direction.py` | per-session FR encoder (`combined_encoding`, modes A/B/S/D): per-pixel held-out SSE / SST stored as **2 x 2-block sums** per model x window (`maps`) | **a** | re-aggregate blocks over the ROI masks: ~4 min for 107 sessions x 4 modes (6 workers) | YES |
| 2.2 (residual panels), 2.6 (residual maps) | `specificity_control.py` | frozen area-target run: saved area-mean PETHs (data, full) | **a** via the **basis twin** (`frozen_encoder_<tag>_basis`: evoked per spout of data / every part in the pre-stroke basis) | seconds per session | YES |
| 2.2 / 2.6 Layer 2 panel | `specificity_control.py` (reads `allsession_encoder_*`) | area-mean traces of the all-session encoder | **c** | re-run `allsession_encoder` with the ROI targets: frA run took ~19 min wall, `_ref` (with random splits) ~15 h | no |
| 18.5, 18.6, 18.9 | `residual_roi.py` (pooled + `--by-spout`) | specificity `sessions.csv` / area-run per-spout PETHs | **a** (basis twin) | seconds per session | YES |
| 0.13 | `early_predictors.py` | specificity (FR-A) `sessions.csv` residual_z | **a** (via the above) | seconds | YES (2 predictor columns) |
| 0.3, 0.9 (contest), 0.4, 0.10 (PETH vs trial), 0.11 (trend) | `frozen_claims.py` (via `frozen_fr_compare.session_r2`) | frozen area-target run `r2_musall` (Musall frame / trial / avg R2 of AREA-MEAN segments, per variant) | **c** | see below | no |
| 21.4, 21.11 | `fr_trial_peth.py` | O2 run `fr_A_tr`: time-resolved SSE / SST of AREA-MEAN traces (`sse`, `sse_trial`, ...) | **c** | O2 re-run of `combined_encoding` with ROI targets in `Ra` (`fr_A_tr` collected over ~5 h); the block maps hold window totals only, no trial / PETH split | no |

Block resolution check (class a, BLOCK): ROI SSE = sum over 2 x 2 blocks of (ROI share of the block's brain pixels) x
block SSE. Against the EXACT per-pixel region R2 (`<label>_regions.csv`), 42 areas x 3 models x 2 windows x 2 sessions
(PS92_0806, PS93_1002): max |error| 0.0008 R2, median 0.00002; MOs / MOp / SSp-m <= 0.0001. 3-4 % of an ROI's pixels
sit in partial blocks.

Basis check (class a, BASIS): ROI residual from the basis twin vs the area run's `sessions.csv`, 7 areas x 2 windows x 4
sessions (one per animal): r = 1.000; residual max |difference| 0.15 x1e3 dF/F (residuals span about +-1..18), median
0.02 (noFR; identical alphas) / 0.05 (FR-A; its basis run chose lick alpha 1e4 vs the area run's 1e3).

### Class c options (not run; Priya to choose)
1. **Frozen claims (0.3, 0.4, 0.9-0.11)** -- cheapest exact-enough route: re-run the two BASIS runs (noFR, FR-A) with
   `FROZEN_PIX_MUSALL=1` for the variants the claims need (joint, no_target, no_exec_locked, no_exec_within, no_spout,
   no_movement, no_fr_all) after adding a FRAME moment to `frozen_encoder.musall_moments` (it saves avg and trial only);
   then ROI R2 = 1 - w'Ew / w'Sw for any mask. Measured basis-run cost: fit ~22 min + predict ~33 min per frame
   (~1 h each, both frames in parallel ~1 h). Caveat: FR-A basis alphas differ (lick) from the FR-A area run.
   Exact route: re-run the AREA runs with the ROI targets added and the alphas held at the existing fits (code needed in
   `frozen_encoder.fit_animal`: otherwise the alpha search, which sums over areas, moves every existing number).
   Measured area-run cost: stats ~26 min + predict ~90-120 min per frame (~2-2.5 h; both ~2.5 h in parallel).
   Note 0.11 plots cortex only; its new rows would be CSV-only unless the figure changes.
2. **fr_trial_peth (21.4, 21.11)** -- O2 re-run of the `fr_A_tr` encoder with the ROI masks added to `Ra` (one change in
   `combined_encoding`; it changes the input-cache code hash). ~5 h on O2 plus collection.
3. **Layer 2 (2.2 / 2.6 panel)** -- re-run `allsession_encoder` (frA ~20 min wall) with the ROI targets.

## Step 2 -- implemented (branch commits; run script `scratchpad/run_alm_rows.sh`)
- `encoder_split_rows.py`: masks (one definition: `iti_licks.roi_masks` on the stat mask), block weights / block R2,
  basis weights / evoked, `residual_rows`, `split_image` (maps with MOs / MOp drawn as their split ROIs).
- `fr_direction.py`: 8 split rows beside MOp / MOs (layout 4 x 2 panels, split rows on a grey band). Old rows of
  `areas.csv` / `bootstrap.csv` identical to the share's (max |diff| 0).
- `specificity_control.py`: split rows in `sessions.csv` / `area_boot.csv`, new `split_boot.csv`, 4th panel in
  `specificity_groups.png` (MOp, tjM1, MOp_rest, MOs, ALM, MOs_rest, cue vs lick), split panels in
  `specificity_area_maps.png`. Groups unchanged (split ROIs belong to no group).
- `residual_roi.py`: split rows tested with the SAME Bonferroni count (Allen areas only); pooled figure gets one split
  row per window; per-spout figures get `*_split.png` companions; the Allen figures keep their scale / counts.
- `early_predictors.py`: `resid_day1_{cue,lick}_almtjm1` predictor columns.

## Tests (10-27, every output redirected to a worktree scratch folder; inputs read from M:)
- Old rows unchanged vs the share: fr_direction areas.csv / bootstrap.csv max |diff| 0; specificity (both frames)
  sessions / boot / area_boot / layer2_boot <= 4e-6 (CSV float round trip); residual_roi pooled + by-spout <= 7e-7;
  early_predictors table / rho <= 2e-7. The noFR specificity run reproduces the share's layer2_boot exactly with
  `--layer2 allsession_encoder_v7g3_t3d_ref` (so that is the flag in the run script).
- Additivity, PS94, residual (noFR), all sessions x windows: pixel-share weighted (ALM 0.382 / tjM1 0.470 of the parent
  on the stat mask) parts vs the existing MOs / MOp row: max |diff| 0.07 x1e3 dF/F, r = 1.000 (basis vs area source).
  FR block rows: SST-weighted parts reproduce the parent R2 exactly; SSE sums add to 0.
- M: untouched: 7,084 files under the output folders listed before / after; 1 changed
  (`iti_licks/figdata/lick_basis_decoder.pkl`, 22:31:41) = the main session's run_maskloc.sh (rr8/mk_basis.log 22:31:42),
  not touched by these scripts.

## Page figures whose captions need new numbers after the run
22.3, 22.4 (layout changed to 4 x 2 + split rows), 2.2 (4th panel), 2.6 (split panels), 18.5 (split rows), 0.13 (2
predictor columns). 18.6 / 18.9 keep their figures (Allen); `residual_roi_by_spout_{cue,lick}_raw_split.png` are new
companions (swap in or add). Not changed: 0.3, 0.4, 0.9, 0.10, 0.11, 21.4, 21.11 (class c).
