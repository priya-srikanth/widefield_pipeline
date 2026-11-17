# Docked re-render of the decoded-vs-executed figures (Priya 11-17: "we should re-render docked")

Rebuild tag `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`
(`<tag>` below). Cohort 1 (PS92-95). Root `I = M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks`.

## Plain-language reading

- **Docking does not change the answer.** When the spout is parked and the mouse licks on its own, cortex still reads
  its licks as landing further toward the contralesional side than they actually land. The like-for-like ITI offset is
  +0.30 / +0.21 / +0.27 mm (acute / subacute / chronic) for docked licks, against +0.36 / +0.24 / +0.27 mm for all ITI
  licks. The pre-rebuild docked values were +0.31 / +0.21 / +0.28.
- **ITI licks after quiet (bout start ≥ 2 s after the previous lick, spout docked) show the same offset, slightly
  smaller:** +0.22 / +0.18 / +0.21 mm. That is close to the production gap2 values (+0.21 / +0.20 / +0.17, where the
  decoder was refit on gap2 licks). The offset does not come from licks that ride on a running bout.
- **Cued licks are unchanged by the filter**, because it only drops ITI rows. The cued first lick carries the cleanest
  cued offset: +0.35 / +0.22 / +0.16 mm. All cued licks give +0.19 / +0.08 / −0.03.
- **The angle versions agree.** The like-for-like angle offset for docked ITI licks is +4.3 / +2.6 / +3.4° (all ITI
  +5.3 / +3.0 / +3.3°; gap2dock +3.6 / +2.6 / +2.6°), and the curves move up in the same way.
- **One thing does change: the raw within-animal link between the ITI offset and aim recovery.** With all ITI licks it is
  r = −0.33 (p 0.004). With docked licks it is r = +0.03 (p 0.81); the pre-rebuild docked value was +0.06. With log(day)
  regressed out, docked licks still give r = −0.30 (p 0.013). So the session-to-session tracking of recovery holds once
  the shared time trend is removed. Without that, it depends on the undocked licks.
- **Why the docked ITI acute line is dashed:** PS95 has one acute session with only 53 docked ITI licks (26 first
  licks). At sigma 0.1 mm the kernel weight never reaches 10, so PS95 has no acute curve, and the acute line is the mean
  of 3 animals throughout. The figure now says so in the panel.

## How it was run

- Docked = `WFIELD_ITI_FILTER=dock` and gap2dock = `WFIELD_ITI_FILTER=gap2dock`. These are the existing **subset**
  filters (`iti_licks.ITI_SUBSET`, `iti_subset_keep`, flags from `iti_dock_audit.csv`). The frozen all-ITI decoders are
  used and **nothing is refit**; only ITI rows are dropped downstream, and cued rows are untouched.
- The env is replayed from the rebuild step logs (`render_daqlk/logs/e_split_gauss_all_g.log`, which is identical for
  `e_matched_axes_g`, `e_angle_reach_g`, `e_rlink_*_g`, `e_scatter_z_g` and `e_checkfigs_g`). On top of it:
  `WFIELD_ITI_FILTER`, `WFIELD_ITI_OUT_W=<new folder>` and `PYTHONPATH=<worktree>;<worktree>/scripts`. Runner and
  batches: `docs/status/deck_swap_20261117/dockfig_scripts/` (`run_step.sh`, `batch.sh`, `batch2.sh`, `redraw.sh`,
  `redraw2.sh`, `compare.py` → `compare.txt`). Each step has a log in `<folder>/logs/<step>.log` with the code sha,
  the command and the full env.
- The docked where caches were reused: byte copies of `itidock_main_20261117/<tag>/` (where_preds, angle_reach,
  zscore where_preds_z + where_decoder_z, where_dockflag). The gap2dock where caches were built by
  `scripts/iti_dock_where_preds.py` from the production all-ITI caches (subset by alignment, nothing refit; check vs
  the published r: max |dr| 5e-16).
- **Reproduction check:** the docked rerun of `lick_matched_readout --ref gauss` gives `matched_gauss_{sessions,tests,bin_tests,sigma_cv}.csv`
  **byte-identical** to `matched_gauss_dock_20261117`. The all-ITI runs of the default code path give
  `matched_gauss_*` and `matched_axes_*` tables byte-identical to production.
- **The 11-17 ITI response window (0..0.5 s) does not apply.** None of these scripts uses a response-AUC window: the
  decoders read lags 0..0.2 s around the lick peak, and the like-for-like reference, the scatter, the delta and the error
  figures have no AUC. Checked by grep for `RESP_WIN` / `resp_auc` / `BASE_RESP`. The only `resp_auc` use is in the
  F0 figures of `iti_check_figures`, which are not drawn in a subset run.
- **Figure fixes (Priya 11-17, same pass), redrawn plot-only:**
  - The purple trial-set footer no longer sits on the x axis. `trial_set_text.footer_bottom` (moved from
    `rebuild_figs/fig_state_position_sessions._footer_bottom`) is now the bottom of each tight_layout rect. It applies to
    the curves, readout, per-spout, animal-change, angle, recovery-link (legend stacked above the footer), scatter,
    delta, check-error and where_angle_reach figures.
  - Where a curve is mostly dashed (3 animals), a note in the panel names the missing animal and its landing coverage,
    and the coverage table is printed in the log.

## Figures (production all-ITI → docked)

Paths: `P = I/<tag>/figures/`; `D = I/dock_dockfig_20261117/<tag>/figures/`; `G = I/gap2dock_dockfig_20261117/<tag>/figures/`;
`A = I/allITI_dockfig_20261117/<tag>/figures/` (all-ITI twins of the new figures); `R = I/allITI_curves_redraw_dockfig_20261117/<tag>/figures/`.

| figure | all ITI (production) | docked, all ITI licks | docked, ITI after quiet (gap2dock) |
|---|---|---|---|
| decoded vs landing curves, ITI + cued all (12.2) | `P/lick_matched_gauss_curves.png` (redrawn with the fixes: `R/…`) | `D/lick_matched_gauss_curves.png` (same numbers as `matched_gauss_dock_20261117/<tag>/figures/lick_matched_gauss_curves.png`) | `G/lick_matched_gauss_curves.png` |
| curves, ITI FIRST lick (bout start) + cued FIRST lick (new, `--first-curves`) | `A/lick_matched_gauss_curves_first.png` | `D/lick_matched_gauss_curves_first.png` | `G/lick_matched_gauss_curves_first.png` (ITI rows = all gap2dock licks; every one is a bout start) |
| ITI after ≥ 2 s quiet | `P/lick_matched_gauss_gap2_curves.png` (decoder REFIT on gap2) | — | `G/lick_matched_gauss_curves.png` (frozen all-ITI decoder, subset) |
| like-for-like read-out / per spout / per animal (12.1, 13.6) | `P/lick_matched_gauss_{readout,perspout,animal_change}.png` | `D/…` | `G/…` |
| angle like-for-like (read-out) | `P/lick_matched_angle_reach.png` | `D/lick_matched_angle_reach.png` | `G/lick_matched_angle_reach.png` |
| angle curves, decoded vs executed ANGLE, all + first (new, `--curves`) | `A/lick_matched_angle_reach_curves.png` | `D/lick_matched_angle_reach_curves.png` | `G/lick_matched_angle_reach_curves.png` |
| angle / reach / elevation r / pre | `P/lick_where_angle_reach.png` | `D/lick_where_angle_reach.png` | `G/lick_where_angle_reach.png` |
| decoder offset vs aim recovery (16.3-16.5) | `P/lick_recovery_link_{pooled_,,sessions_}aim.png` (unscaled); z: `A/lick_recovery_link_{pooled_,,sessions_}aim_z.png` | `D/lick_recovery_link_{pooled_,,sessions_}aim_z.png` | `G/…aim_z.png` |
| same, angle behavior axis | `P/lick_recovery_link_{pooled_,,sessions_}angle_z.png` | `D/…angle_z.png` | `G/…angle_z.png` |
| decoded vs actual scatter (12.6) | `P/lick_iti_where_scatter_z.png` | `D/lick_iti_where_scatter_z.png` | `G/lick_iti_where_scatter_z.png` |
| where delta (12.7) | `P/lick_where_delta_z.png` | `D/lick_where_delta_z.png` | `G/lick_where_delta_z.png` |
| per-lick error (12.3) + sessions (12.8) | `P/lick_iti_check_error_per_lick{,_angle}.png`, `P/lick_iti_check_error_sessions.png` | `D/…` (same three) | `G/…` |

Not rendered docked: the `_far1`, `_gap2far1`, `_nonlinear` and `_plain` curve variants. Docked and gap2dock **loclin**
curves already exist in the rebuild at `I/<tag>/itifilter_{dock,gap2dock}/figures/lick_matched_loclin_curves.png`
(rebuild steps `e_dock_loclin_*_g`). The docked recovery link is z-scored only, because `iti_dock_where_preds.py`
writes no unscaled r table.

## Key numbers (change vs pre; acute | subacute | chronic; * = 95% paired nested bootstrap CI excludes 0; k/n animals)

Full output: `dockfig_scripts/compare.txt`.

**1. Like-for-like lateral offset (Gaussian same-place reference), all spouts, mm**

| set | lick set | all-ITI (production) | docked | gap2dock | pre-rebuild docked |
|---|---|---|---|---|---|
| ITI | all | +0.36* / +0.24* / +0.27* | +0.30* / +0.21* / +0.27* | +0.22* / +0.18* / +0.21* | +0.31* / +0.21* / +0.28* |
| ITI | first (bout start) | +0.38* / +0.26* / +0.26* | +0.37* / +0.22* / +0.26* | (= all) | +0.38* / +0.23* / +0.26* |
| cued | all | +0.19* / +0.08 / −0.03 | same | same | +0.16* / +0.03 / −0.03 |
| cued | first | +0.35* / +0.22* / +0.16* | same | same | +0.31* / +0.22* / +0.15* |

Production gap2 (refit decoder), ITI: +0.21* / +0.20* / +0.17*. Per animal, docked ITI all: acute PS92 +0.26,
PS93 +0.07, PS94 +0.40, PS95 +0.46; chronic +0.19 / +0.05 / +0.54 / +0.31. PS93 is the small one in every set (all-ITI:
acute +0.12, chronic +0.06).

**2. Curve shift** (mean of the epoch curve minus the pre curve over the shared 4-animal landing range, mm)

| curves | all-ITI | docked | gap2dock |
|---|---|---|---|
| ITI all | +0.34 / +0.21 / +0.23 | +0.16 (3 animals, −1.30..−0.15 mm) / +0.12 / +0.24 | — / +0.08 / +0.23 |
| ITI first | +0.30 / +0.19 / +0.24 | +0.16 (3 animals) / +0.11 / +0.22 | (= all) |
| cued all | +0.18 / +0.08 / +0.00 | same | same |
| cued first | +0.22 / +0.12 / +0.09 | same | same |
| ANGLE ITI all (deg) | +4.9 / +2.5 / +3.3 | +3.7 / +2.0 / +3.5 | — / +0.9 / +2.5 |
| ANGLE cued first (deg) | +3.4 / +1.5 / +0.5 | same | same |

The docked and gap2dock curve shifts are averaged over narrower landing ranges than the all-ITI ones: with fewer licks,
fewer landing points reach the kernel weight in all 4 animals. Read the offset table (1) as the like-for-like number.

**3. Like-for-like angle offset (Gaussian reference), deg**: ITI all-ITI +5.26* / +3.03* / +3.29*; docked
+4.28* / +2.63* / +3.35*; gap2dock +3.64* / +2.60* / +2.59*. Cued +2.73* / +1.39* / −0.18 (unchanged).

**4. Where delta (z-scored decoder, lateral error-vector change vs pre, all licks), mm**

| set | all-ITI | docked | gap2dock | pre-rebuild docked |
|---|---|---|---|---|
| ITI | +0.48* / +0.28* / +0.22* | +0.61* / +0.30* / +0.31* | +0.74* / +0.35* / +0.34* | +0.62* / +0.30* / +0.32* |
| cued | +0.58* / +0.23* / +0.03 | same | same | +0.56* / +0.19 / +0.04 |

**5. Recovery link (aim_z): ITI like-for-like lateral offset per animal, mm (acute / subacute / chronic), and the
within-animal r(offset, aim) over post sessions**

| animal (aim change, deg) | all-ITI (z) | docked | gap2dock | pre-rebuild docked |
|---|---|---|---|---|
| PS92 (−15.8 / −8.8 / −2.2) | +0.22 / +0.30 / +0.15 | +0.25 / +0.34 / +0.19 | +0.08 / +0.22 / +0.11 | +0.22 / +0.30 / +0.17 |
| PS93 (−12.4 / −7.4 / −6.7) | +0.12 / −0.02 / +0.06 | +0.06 / −0.07 / +0.05 | +0.02 / −0.03 / +0.08 | +0.08 / −0.03 / +0.07 |
| PS94 (−24.6 / −12.1 / −11.8) | +0.48 / +0.39 / +0.49 | +0.37 / +0.39 / +0.52 | +0.38 / +0.38 / +0.47 | +0.41 / +0.42 / +0.58 |
| PS95 (−18.2 / −10.1 / −4.3) | +0.58 / +0.27 / +0.34 | +0.47 / +0.16 / +0.31 | +0.41 / +0.09 / +0.18 | +0.48 / +0.12 / +0.30 |
| ITI within-animal r, raw | −0.33 (p 0.004) | +0.03 (p 0.81) | +0.17 (p 0.16) | +0.06 (p 0.63) |
| ITI within-animal r, log(day) detrended | −0.46 (p < 0.001) | −0.30 (p 0.013) | −0.23 (p 0.057) | −0.38 (p 0.002) |
| cued within-animal r, raw / detrended | −0.80 / −0.49 | same | same | −0.76 / −0.57 |

Sessions per animal (acute / subacute / chronic): PS92 5/2/12, PS93 4/3/12, PS94 6/5/8, PS95 1/7/11. Between animals,
chronic PS94 has the largest ITI offset and the least recovered aim in every set (descriptive, n = 4).

**6. Per-lick ITI error shift (session mean of decoded − actual, change vs pre)**: lateral, all-ITI
+0.92* / +0.51* / +0.45* mm, docked +0.98* / +0.48* / +0.50*, gap2dock +1.25* / +0.71* / +0.59*. Angle: all-ITI
+13.0* / +6.6* / +6.2°, docked +13.6* / +6.3* / +6.8°, gap2dock +17.3* / +8.8* / +7.7°.

**7. Lick counts (ITI rows of the where cache; pre / acute / subacute / chronic)**: all-ITI 26872 / 8977 / 10506 / 32601;
docked 9829 / 2983 / 3532 / 11624; gap2dock 2806 / 1086 / 986 / 3504.

## Dashed-line coverage (why a curve is dashed)

Dashed = landing grid points where only 3 animals have a curve (kernel weight ≥ 10 at sigma 0.1 mm). Full tables are in
`<folder>/logs/redraw_matched.log`.

- Docked ITI, all licks: **acute is dashed throughout. PS95 has no curve: 1 session, 53 docked licks.** The other three
  animals cover acute PS92 −2.60..+1.00, PS93 −1.55..+1.45 and PS94 −3.00..−0.15 mm. Subacute and chronic are mostly solid.
- Docked ITI, first lick: acute PS95 has no curve (26 licks). Subacute and chronic are mostly dashed because PS94 covers
  only −3.00..−0.10 / −3.00..+0.05 mm; its landings sit on the ipsilesional side.
- gap2dock ITI: acute PS95 has no curve (8 licks). Subacute and chronic are mostly dashed because PS94 covers only
  −3.00..−0.25 mm.
- All-ITI first-lick curves: acute is dashed because PS95 covers only −1.75..−0.70 mm (1 session, 106 licks).
- Angle curves: docked ITI first-lick acute has no PS95 curve, and PS94 covers only −36..0° (subacute) / −37..0°
  (chronic). For all-ITI first licks, acute PS95 covers −32..−13° only.

## Code changes (this pass; defaults unchanged)

- `scripts/lick_matched_readout.py`:
  - `--first-curves` draws `lick_matched<TAG>_curves_first.png` and `matched<TAG>_first_bin_tests.csv`.
  - The curves figure is now a helper, `_curves_figure`.
  - The smoothing calls `likeforlike.smooth_curves`. It is exactly equal to the old inline loop (checked).
  - The coverage table and dash notes come from `likeforlike`.
  - The footer margin comes from `trial_set_text.footer_bottom`.
  - `plot()` creates `figures/`, so `--plot-only` works in a seeded folder that holds figdata only.
- `scripts/lick_matched_angle_reach.py`:
  - Writes go to `il.OUT_W`, which equals OUT by default.
  - Under a subset filter it reads only the filtered where caches; the 3-axis table has none, so it is skipped.
  - `--curves` draws the angle curves, all + first lick. Lick order comes from the lick_order cache by alignment.
  - Footer margin.
- `scripts/iti_where_angle_reach.py`:
  - Writes go to `il.OUT_W`.
  - Under a subset filter it refuses unless the filtered prediction copy exists, so it never refits.
  - Under a subset filter only, it applies the session floor `MIN_SESSION_LICKS`.
  - Footer margin.
- `scripts/iti_recovery_link.py`:
  - Under a subset filter the default `--src` is `il.OUT_W`, and it refuses unless the filtered z caches are there.
  - Footer margin; the pooled legend sits above the footer.
- `scripts/iti_check_figures.py`:
  - A subset run also draws `lick_iti_check_error_per_lick_angle`.
  - `--plot-only` in a subset run draws only the subset figures.
  - Footer margin.
- `scripts/iti_where_scatter.py`: footer margin.
- `scripts/lick_refit_decoders.where_sets`: new `filt=` argument, default "", so the caption names the subset filter.
- `scripts/likeforlike.py`: new `smooth_curves`, `curve_coverage` and `dash_notes`.
- `wfield_local/trial_set_text.py`: new `footer_bottom`, moved from `scripts/rebuild_figs/fig_state_position_sessions._footer_bottom`,
  which now calls it.

## Addendum: slide 147 footer fix (same pass)

- `scripts/frozen_claims.py`: the trend figure (Fig 0.11, slide 147) reserves the footer margin
  (`trial_set_text.footer_bottom`). New flags, defaults unchanged:
  - `--trend-only` runs section C and the trend figure only, moved into `trend_data`.
  - `--plot-only` redraws from `<out>/figdata/frozen_claims_trend.pkl`, which every run now saves.
- Re-render: `I/../<tag>/frozen_fr_compare_footerfix_20261117/figures/frozen_claims_trend.png` (env of `z_frozen_claims_g`;
  `dockfig_scripts/s147_footerfix.sh`). `claims_trend_{halfday,levels}.csv` are byte-identical to production.
- Deck crop: `presentations/_source_PI_overview_2026-11-17/swap_inputs/s147_frozen_claims_trend_footerfix.png`
  (`dockfig_scripts/crop_s147.py`; 3 panels + legend, white-padded to the old 1680 x 401). Spec:
  `specs/s147_footerfix.json` (picture swap only, box unchanged).

## Combined like-for-like panel figure (Priya 11-17 evening)

- **Figure:** `I/panels_loclin_first_20261117/<tag>/figures/lick_matched_panels.png`. Columns: cued all licks, cued first
  lick, ITI all licks, ITI first lick (bout start), ITI docked licks, so the cued panel is no longer repeated across
  figures.
  - Row 1: all post-stroke licks, with the local-linear edge line carried beyond the pre range.
  - Row 2: only post-stroke licks inside each animal's pre landing range (strict like-for-like).
- **Curves:** Gaussian-smoothed per animal with the loclin sigma (0.1 mm), shown as mean +/- SEM of the 4 animals and
  dashed where only 3 animals contribute. Lines are thinner and translucent (width 1.2, opacity 0.75) so overlapping
  epochs stay visible.
- **Bars:** BH-surviving 0.25-mm bins, tested on each row's own lick set.
- **Why loclin only:** Priya's decision was loclin with a Gaussian kernel. A draft with a Gaussian-reference row and a
  loclin row came out almost identical, because the reference changes only the bars and the read-outs, never the
  curves.
- **Inputs:** `lick_matched_readout.py --ref loclin --inside-curves` (all-ITI with `--first-curves`; dock run in
  `panels_loclin_dock_20261117`). The recompute matches the production loclin tests exactly (432/432).
- **Binned loclin with SEM band** (`--sem-band`, plot only): `I/semband_{allITI,dock,gap2dock}_20261117/<tag>/figures/
  lick_matched_loclin_curves.png`. The script is `dockfig_scripts/semband.sh`.
- **Code:** both options are default-off.
  - `lick_matched_readout`: new `curve_panel` helper (the panel drawing, now shared), `--sem-band`, `--inside-curves`
    (`inside_curves`).
  - New `scripts/lick_matched_panels.py`.
  - A plot-only redraw with default options was byte-identical before and after the refactor (gauss curves, loclin
    sem-band).
