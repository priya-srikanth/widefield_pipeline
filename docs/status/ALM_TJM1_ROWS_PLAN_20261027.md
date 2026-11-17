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
- `fr_direction.py`: 8 split rows beside MOp / MOs (layout 4 x 2 panels, split rows on a gray band). Old rows of
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

## Step 3 (10-28) -- class c, options 1 and 3 (Priya 10-28: "yes let's redo musall", "can definitely do the frA panel")
Stat mask kept (Priya 10-28). Code in this branch; run script `scratchpad/mu/run.sh` (not yet run on the share).
- **Frozen claims (0.3, 0.4, 0.9-0.11), cheapest route WITH the area-run alphas held.** `frozen_encoder.musall_moments`
  saves a FRAME moment too (mus_<w>_frame_{E,S}); new flags `--fit-from` (reuse another run's pre statistics / templates /
  basis, read only; refuses if the saved run tag differs from the environment), `--alphas-from` (hold another run's alphas
  per animal x variant), `--predict` (subset), `--out-root`; `_predict_spatial` now checks the column layout. Re-runs:
  `frozen_encoder_<frame>_basis_pixmusall` (NEW folders). ROI R2 = 1 - w'Ew / w'Sw: `encoder_split_rows.musall_rows`.
  `frozen_claims.py`: MOs (area run) + the 8 split rows after their parents on a gray band, per-animal tables, the trend
  CSVs get the motor rows (figure 0.11 stays cortex).
- **Why the alphas are held:** the basis runs' alpha search (summed over basis targets) picked different alphas from the
  area runs in MORE cells than noted above, noFR too: FR-A joint PS92 / PS94, no_exec_within PS92 / PS94, no_target PS92,
  no_movement PS93; noFR no_exec_locked PS92, no_movement PS93, no_spout PS92 / PS95 (FR-A basis has no no_exec_locked).
  Effect on the CORTEX numbers already saved by both runs (all 107 sessions): per-session unique R2 differs up to 0.051
  (FR-A exec_within, avg), epoch - pre per animal up to 0.026 (avg) / 0.0025 (frame); noFR <= 0.0023 / 0.0015. Holding the
  alphas removes this: re-run cortex = area run to 1.5e-14.
- **Validation (guarded dry run, PS92 + PS93 x pre / acute / chronic, both frames, all 7 variants):** basis-route R2 vs the
  area run's MOs / MOp / SSp-m (each hemisphere): max |d| 0.006 (avg), 0.002 (frame / trial). Plotted unique R2 per session:
  contest (frame) <= 0.0016 (typical |value| 0.01, p90 0.05); trend (frame, spout) <= 0.0014 (0.04 / 0.11); PETH trial
  <= 0.0020 (0.05 / 0.31), avg <= 0.0039 (0.06 / 0.21); epoch - pre deltas <= 0.0027 (median |delta| ~0.015). The
  residual difference = the out-of-basis variance + whole-trial-only moments. Re-run vs the existing basis runs where
  alphas agree: evoked / moments bit-identical.
- **Layer 2 with ROI targets:** `allsession_encoder --roi-targets [--post-from <tag>] [--out-root]`: the 8 ROI means appended
  as extra targets (`session_design(extra_targets=)`, `encoder_split_rows.roi_targets`); Allen statistics kept from the
  saved runs (X'X checked: 0 difference), kappa chosen on the Allen targets only. Dry run PS93: every Allen row of
  allsession_changes_by_session.csv identical (max |d| 0), kappa unchanged. `specificity_control.py` (`--out`, `--layer2`
  may be a path): "LAYER 2 by ROI" panel, `layer2_split_boot.csv`, split Layer 2 map panel; all existing CSVs identical.
- Runtime estimate for run.sh: ~1-1.5 h wall (three parallel tracks of 4 workers, ~18 GB RAM peak).

## Step 4 (11-06) -- ALM / tjM1 on deck slides 94 / 139 and every encoder area map (Priya 11-06)
Branch `feat/alm-tjm1-slides-94-139-20261106`. Coloring = the existing split maps: `encoder_split_rows.split_image`
(MOs / MOp drawn as ALM + MOs_rest / tjM1 + MOp_rest, rim trimmed), Allen color scale, white-halo / black outline for
significance, Bonferroni count = the Allen areas only (`encoder_split_rows.bonf_allen`, as residual_roi).
- **Frozen side: `frozen_encoder_<frA tag>_basis_pixmusall_union`** (NEW folder; `encoder_split_rows.PIX_UNION`): the
  FROZEN_PIX_MUSALL basis re-run with `--fit-from <tag>_basis --alphas-from <tag>` and the union of the variants the two
  slides read: joint no_task no_spout no_movement no_fr no_fr_dir no_target no_spont no_other task_only lick_only
  other_only spout_only movement_only. `no_lickspout` is NOT predictable here: it is a variant only under
  FROZEN_LICKSPOUT=1 (the `_frA_ls` frame), so the frA area run has no `no_lickspout` either and session_vs_frozen
  already drops it for `--tag ..._frA` (no lickspout column on slide 139 in either the Allen or the split maps).
- **Slide 94** `frozen_peth_trial_maps.py --split [--alone]`: Allen outputs unchanged; adds `peth_trial_maps_<w>[_alone]
  _split.png` + `peth_trial_area[_alone]_split.csv` (split ROIs: own bootstrap, Allen Bonferroni count).
- **Slide 139** `session_vs_frozen.py --roi-targets`: the self-fit adds the 8 ROI-mean targets
  (`encoder_split_rows.roi_targets` via `session_design(extra_targets=)`), penalties chosen on the Allen targets and HELD
  for the ROI targets; output `session_vs_frozen_roi/` (the existing folder untouched). `--roi-targets --plot-only
  --maps` paints the split maps (frozen ROI R2 from the union run). `session_design` now computes the Allen target
  columns as the same product as a run without extra targets (bit-identical Allen targets).
- `remap_checks.py --roi-targets`: check 2 (`remap_per_animal_<w>.png`) in `session_vs_frozen_roi/` with the split
  painting (check 1 is unchanged by ROI rows and stays in `session_vs_frozen/`).
- `reembedding_test.py` `reembed_areas.png`: the 45 split-basis areas are painted on their own territory
  (`split_map` / `area_values_map`); the old map summed ALM + MOs_rest back into MOs although the title said otherwise.
  Display only.
- **Validation (11-06, scratch outputs only):** union dry run PS93_0807 (pre) + PS93_0925 (chronic), all 14 variants:
  alphas = the area run's in every cell; basis-route vs area-run R2, MOs / MOp (4 rows) max |d| 0.0029 avg / 0.0010
  frame / 0.0010 trial (median 0.0004 / 0.0002 / 0.0001), SSp-m <= 0.0032, cortex 0 exactly; the moments of the
  variants shared with the 10-28 twin (joint, no_target, no_spout, no_movement) bit-identical. Self-fit PS93_0925, 10
  variants: Allen R2 with --roi-targets == without (exact) and == the share's session_vs_frozen JSON (max |diff| 0,
  alphas equal). Runtime per session (1 worker, loaded box): self-fit 6.5 min plain / 8.7 min with ROIs; union predict
  ~4.5 min per session per worker for 14 variants.
- Caveat (as Step 3): the frozen ROI R2 is the basis route (ROI mean of the basis-projected data), the self-fit ROI R2 is
  the exact ROI mean; the self - frozen gap of a split ROI therefore carries the out-of-basis variance (constant per
  session, mostly removed by epoch - pre).

### Step 4b (11-06, Priya): EXACT split-ROI frozen side for slide 139 / remap_checks
`frozen_encoder --roi-targets --fit-from <area run> --alphas-from <area run>` -> NEW folder
`frozen_encoder_<tag>_roiexact` (`encoder_split_rows.ROI_EXACT`): the area run re-scored with the 8 ROI means as extra
targets. Per animal x variant an Allen fit (the area run's saved pre statistics, penalties held = the area run's betas)
and an ROI fit (same saved X'X / sxx, ROI X'Y / Y'Y from a fresh pass over the pre sessions, `fit/pre_stats_roi.pkl`,
X'X checked), penalties held; predictions use the area run's operations on the Allen columns. Output = per variant x
session JSON with r2_musall + frozen_r2_cortex for areas = Allen + ROI (no PETH / per-trial / drop-spout files).
`session_vs_frozen --roi-targets --maps` and `remap_checks --roi-targets` read it (`--roi-frozen auto|exact|pixmusall`,
auto = exact when present, else the pixmusall union run; `encoder_split_rows.roi_frozen_source`). Slide 94 stays on the
pixmusall union route.
- Dry run PS93_0807 + PS93_0925, 9 variants (scratch): ROI statistics X'X max rel. diff 0; Allen r2_musall and
  frozen_r2_cortex of all 18 session x variant JSONs bit-identical to the area run. Exact vs basis route (864 ROI values):
  max |d| 0.0042 avg / 0.0034 frame / 0.0026 trial (median 0.0006 / 0.0003 / 0.0003; mean ~0); unique R2 (joint -
  no_g, frame) max |d| 0.0020 (no_movement), 0.0018 (no_other), <= 0.0006 the others.
- Cost (loaded box, 2 workers): ROI statistics ~5 min per pre session per worker (31 sessions), predict ~3 min per
  session per worker (107) -> ~4 h at 2 workers, ~2.7 h at 3.
