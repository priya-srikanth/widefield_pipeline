# ITI star tables: local bootstrap-p / BH copies vs `wfield_local.analysis_kit` (2026-11-05)

Code review item 7 (`docs/status/CODE_REVIEW_2026-11-04.md`; raw json lists `iti_decoder_pattern_stats.py:139`,
`iti_pattern_area_stats.py:68`, `iti_session_pattern_tests.py:158`, `iti_spout_pattern_stats.py:97`). No repo file was
edited. Script: `compare.py` in this folder. It runs each script's own test code (imported from main) on the saved caches
(`decoder_pattern_sessions.pkl`, `spout_pattern_sessions[_iti].pkl`, `session_patterns/PS9x.pkl`), with `_boot` / `_bh`
wrapped so that each BH call also gets `ak.boot_p_columns` on the same draws and `ak.bh`. Same rng seeds and order, and no
imaging or decoder recomputation. Outputs are in
`M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/iti_star_shared_stats_1105/`
(`diff_*.csv`, `compare_pattern_area_stats_*.csv`, `rerun_local_*.csv`). Logs: `run1.log`, `run2.log`.

## Which tables (four scripts, eight outputs)

| script | table (iti_licks/) | what carries the stars |
|---|---|---|
| iti_decoder_pattern_stats | decoder_pattern_stats.csv | per-pixel BH, dashed outline, `frac_sig*` |
| iti_spout_pattern_stats (+ --iti) | spout_pattern_stats.csv, spout_pattern_stats_iti.csv | same |
| iti_pattern_area_stats | pattern_area_stats_{position,spout,spout_iti}.csv | per-area `*` = BH (per group) AND >= 3/4 sign |
| iti_session_pattern_tests | session_pattern_maplevel.csv (+ change-map figures, no CSV) | maplevel `sig_bh`; change-map dashed outline / "sig %" |

**Provenance:** the published decoder/spout/area CSVs (10-27 23:16-23:26) come from commit `98ca70ff`, which is on the
rebuild/* branches and not on main. In that version the PRIMARY significance is the FWER max-statistic permutation. The
bootstrap + BH result is kept only as the comparison columns `frac_sig_fdr`, `frac_sig_fdr_agree`, `p_boot`, `sig_fdr`
and `sig_fdr_3of4`. Main's version still draws BH as the stars. The bootstrap code is the same in both, so the comparison
below is against those `*_fdr` / `p_boot` columns.

## Reproduction (local copy against the published table)

- decoder / spout / spout_iti: rerun `frac_sig`, `frac_sig_agree` = published `frac_sig_fdr`, `frac_sig_fdr_agree`
  **exactly** (max diff 0.0, 28 / 42 / 42 cells).
- area (3 files): all 1120 / 1680 / 1680 rows match. max |p - p_boot| = 6e-17, and `sig` = `sig_fdr` and
  `sig_3of4` = `sig_fdr_3of4` in every row.
- maplevel: local `_bh(p_cohort)` = published `sig_bh` (60 / 60 rows).
- Session change maps: no saved table exists. The local path is the script's own code with the same seed.

## Differences

**Cause, in every case: the +1/N term.** The local p is `min(1, 2*min(share<=0, share>=0) + 1/N)`, while `ak.boot_p` is
`min(1, max(2*min, 1/N))`. They agree only when 2*min = 0 (both give 1/N). Otherwise the local p is exactly 1/N larger
(0.001, with N = 1000 draws in every script; area uses 1/len(B), which is also 1000). Other parts of the two copies give
the same results:
- Two-sided convention: same formula in both.
- NaN handling: zero NaN p in every table, and `ak.bh` on the local p equals the local `_bh` in every cell
  (`bh_copy_differs` = 0).
- Family definition: the same family in both (per cell over pixels; per group for area; all 60 rows for maplevel).

So ak's p is never larger. Stars only ever get **added** (0 lost anywhere). The area rows whose p is unchanged at 0.001
are pulled in by BH's step-up once more p values fall under the threshold.

### Area tables (per-area `*`): 22 of 4480 rows gain a star; 0 lose one

| file | group | area | epoch | kind | p local | p ak | n_dir/n |
|---|---|---|---|---|---|---|---|
| position | iti/sag | MOs_left | acute | change | 0.025 | 0.024 | 3/3 |
| position | iti/sag | RSPv_right | acute | change | 0.025 | 0.024 | 3/3 |
| position | iti/sag | ACAd_right | chronic | change | 0.025 | 0.024 | 4/4 |
| position | trial/lat | VISa_left | chronic | change | 0.027 | 0.026 | 3/4 |
| spout | close_mid | SSp-tr_right | acute | level | 0.005 | 0.004 | 3/3 |
| spout | close_mid | RSPagl_left | chronic | level | 0.005 | 0.004 | 4/4 |
| spout | close_mid | VISa_right | chronic | level | 0.005 | 0.004 | 4/4 |
| spout | close_mid | VISam_left | chronic | level | 0.005 | 0.004 | 4/4 |
| spout | close_contra | VISa_left | pre | level | 0.009 | 0.008 | 4/4 |
| spout | close_contra | VISam_left | pre | level | 0.009 | 0.008 | 4/4 |
| spout_iti | close_mid | MOp_r, MOs_r, SSp-bfd_r, SSp-m_l, SSp-ul_r, SSp-un_r, VISal_r, VISrl_r | acute | level | 0.003 | 0.002 | 3/3 |
| spout_iti | close_mid | SSp-m_r, SSp-n_l, SSp-n_r, SSs_l | acute | level | 0.001 | 0.001 | 3/3 |

Unchanged: position 1116 / 1120, spout 1674 / 1680, spout_iti 1668 / 1680. The spout_iti close_mid acute block goes from 0
to 12 starred areas, so a BH threshold is crossed.

### Pixel tables: outline fraction (`frac_sig_fdr_agree`), changed cells only

The dashed outline grows by 0.1-3 percentage points. No cell goes from 0 to non-zero, and no pixel is lost.

decoder_pattern_stats (10 of 28 changed): iti lat pre level 4.7 -> 5.7 %; iti lat subacute level 80.3 -> 80.6;
iti sag acute change 71.2 -> 72.5; iti sag chronic level 84.5 -> 84.7; trial lat acute level 68.8 -> 69.4;
trial lat subacute level 20.4 -> 21.1; trial lat subacute change 50.1 -> 52.2; trial lat chronic change 31.6 -> 33.2;
trial sag pre level 87.9 -> 88.1; trial sag chronic change 15.5 -> 17.0.

spout_pattern_stats (cued, 13 of 42 changed): close_ipsi pre 89.0 -> 89.1, chronic 21.2 -> 22.0; close_mid chronic
81.6 -> 82.0; close_contra pre 80.4 -> 80.8, subacute 16.8 -> 17.2, chronic 20.2 -> 20.5; far_ipsi chronic 8.8 -> 9.3;
far_mid pre 89.6 -> 89.7, acute 56.4 -> 57.1, subacute 81.1 -> 81.5, chronic 12.8 -> 13.8; far_contra pre
51.3 -> 52.1 (all level); **far_contra acute change 6.7 -> 9.6** (the largest change).

spout_pattern_stats_iti (6 of 42 changed): close_ipsi chronic level 7.5 -> 8.2; close_mid acute level 89.5 -> 89.8;
close_contra chronic level 12.0 -> 12.7, chronic change 56.1 -> 57.0; far_contra acute level 29.3 -> 29.9, acute change
15.8 -> 16.4.

Session change maps (26 of 80 changed, `diff_session_pattern_change_maps.csv`). Largest: iti/sp_far_ipsi acute change
2.8 -> 5.2 %, subacute change 2.5 -> 4.8; trial/lat chronic change 4.1 -> 5.9; trial/wlat pre 2.9 -> 4.7;
trial/sp_far_contra chronic change 7.1 -> 8.9; iti/wsag chronic change 4.0 -> 5.5; trial/sp_close_ipsi subacute change
23.7 -> 25.3. All others gain 1 pp or less.

maplevel (`sig_bh`, 60 rows): **no change.** Its p is a permutation p, (k+1)/(N+1), not boot_p, and `ak.bh` = `_bh`
with no NaN. All 56 published stars stand.

## Would any page / deck star change?

- **3.66 maplevel stars: no.**
- **3.50-3.53** (decoder lat / sag, cued spout, ITI spout pixel stats). In the published 10-27-late figures the outlines
  are FWER (98ca70ff). BH is only in the CSV `*_fdr` columns, so the figures do not change. If the page still shows the
  main-code (pre-FWER) versions, the dashed outline grows by the amounts above and no new region appears. The visible
  case is 3.52 far_contra acute change, 6.7 -> 9.6 %.
- **3.67-3.69** (session change maps, `lick_*_session_pattern_change_*.png`, 10-27 16:07). The dashed BH outline and
  "sig %" titles **would change** in the 26 cells above (e.g. 3.68 trial lat chronic 4 -> 6 %; ITI-spout far_ipsi acute
  3 -> 5 %). The solid FWER line does not change. On the rebuild branch the page caption calls 3.67 "TFCE", so check
  which PNG the page uses.
- **Area figures** (`lick_*pattern_area_stats_*.png`): 22 `*` would be added under main's BH version. The published
  23:20 PNGs star FWER instead, and they are not in the page figure list.
- No deck reference to these tables was found.
- Swapping in `ak.*` changes no star from yes to no. Every change adds a star, and all of them come from the 1/N term.

## Addendum (a8, 11-05): which LIVE page figures this touches (encoding page version 1793703992-c30f)
The "3.66-3.69" numbers above come from an older page source (wf_rebuild _goal_page, 11-03). On the live page:
- Fig 15.3 (`lick_session_shape_d.png`, map-level permutation test): unchanged.
- Figs 15.4 / 15.6 (`lick_cued_session_cluster_spout.png`, `lick_session_cluster_position.png`): SOLID = TFCE. The
  captions say TFCE and the PNGs are the session_cluster (TFCE) renders, so the "TFCE caption" query is resolved: they
  match, and boot_p does not enter.
- Fig 15.7 (`lick_session_pattern_change_within_spout.png`): SOLID = max-statistic FWER, DASHED = BH-FDR and >= 3/4.
  The ONLY live figure whose drawn contours use the bootstrap p, so decision 9 can grow its dashed outline slightly.
- Figs 15.8 / 15.9 and 15.17-15.20 (session change amplitude, decoder / spout pattern stats): SOLID = FWER. Unchanged;
  the bootstrap BH-FDR survives only as the CSV column frac_sig_fdr.
