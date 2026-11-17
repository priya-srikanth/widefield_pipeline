# K08 2026: session-only figure scripts, recovered 17 Nov

These ran in a Claude session on 9-10 Nov and wrote outputs straight into the grant figures folder.
None was in `grants/k08_2026/`; all were living only in the session's temporary directory. Recovered
and kept here so the submitted figures are reproducible. Times are local (EST), matching Finder.

| Script | Ran | Made |
|---|---|---|
| `paths_narrow.py` | 9 Nov 23:06 | `Fig3O_paths_narrow.*`, delivered as **`Fig3O_paths_front.svg/.pdf`**. Same as `paths_panel.py` with `W_CELL` 1.02 -> 0.92, tighter gutters, `LEG` 0.95 -> 0.82 |
| `paths_short.py` | 9 Nov 23:29 | `Fig3O_paths_short.*`. As narrow, but `H_CELL` fixed at 1.05 in (equal aspect dropped) and shorter bottom margin |
| `edit_svg.py` | 9 Nov 23:07 | `Figure3_behavior_relettered.svg`: shifts panel letters E->D ... P->O, swaps in the ipsi-labelled aim raster, inlines `Fig3O_paths_narrow.svg` as `Panel_O_paths3d` |
| `caption_fix.py` | 9 Nov 23:09 | re-wraps the Figure 3 caption block in `Figure3_behavior_relettered.svg` |
| `edit_v2.py` | 9 Nov 23:29 | `Figure3_behavior_relettered_v2.svg`: replaces the paths group with `Fig3O_paths_short.svg` and restores each raster's scale factor |
| `split_d.py` | 9 Nov 23:32 | `Figure4_facerhythm_Dsplit.svg` from `Figure4_facerhythm.svg` (panel D split) |
| `declip.py` | 9 Nov 23:55 | `Figure4_facerhythm_Dsplit_light.svg` (unclips and lightens D) |
| `render_5d.py` | 10 Nov 01:14 | `Fig5D_cued.svg/.pdf/.png` from `iti_licks/figdata/lick_cued_ramp.pkl`, drawn in Figure 5 page points |
| `cap5.py` | 10 Nov 01:15 | `Figure5_cuedD.svg`: swaps panel D and rewraps the Figure 5 caption |
| `capio.py`, `update.py` | 10 Nov 01:20-01:21 | read the legends out of `..._v18.docx` and write the `*_legend.svg` variants of Figures 3, 4 and 5 |
| `rast.py` | 9 Nov 12:22 | headless-Chromium rasteriser for SVG panels |
| `overlay.py` | 9 Nov 19:37 | thalamus overlay sheets (`*_thalamus_zoom{,_noGFAP}.png`), the review step behind `thalamus_overlay_options.py` |
| `export_paths3d_front_sig.py` | written 17 Nov | re-export of `paths3d_front_sig.csv`; the original export was never saved |

`recov_README.txt` is the provenance note that was staged alongside `recov/balance_aim_*.csv`.

## Still without code anywhere

- `paths3d_front_sig.csv` -- recomputable with `export_paths3d_front_sig.py`; verify it against the staged copy.
- `data/ss_plane_sessions.csv` -- no script in `scripts/` on `origin/main` writes that name.
- `recov/balance_aim_{sessions,models}.csv` -- derived from `lick_recovery_primary` figdata; see `recov_README.txt`.
