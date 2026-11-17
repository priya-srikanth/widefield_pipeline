# K08 resubmission, November 2026: figure code

The code that built the preliminary-data figures in the K08 Research Strategy (final version 10 Nov 2026):
**Figure 3** (behavior and deficit), **Figure 4** (FaceRhythm) and **Figure 5** (decoder and encoder).
Copied into the repo on 2026-11-17 from the grant folder on Priya's laptop:

    ~/Documents/Sabatini Lab/Grants & applications/K08/Proposal/2026/figures/

That folder still holds everything that is not code: the staged inputs, the rendered panels
(`K08_2026_grant_panels/`, `grant_panels/`), the composed `.svg` / `.pdf` drafts, and the Illustrator `.ai`
files the final figures were finished in. Those stay there, not here (the repo is public, and the `.ai` files are
the deliverable, edited by hand after the code ran).

**Grant only.** Nothing here recomputes a result. Every panel is re-plotted from CSVs the pipeline already wrote
(session tables, LMM tables, bootstrap `_verify` / `_tests` files) or cropped from a figure the pipeline already
rendered. Significance marks are read from each figure's own statistics file, never typed in. See
[`../README.md`](../README.md) for why this lives outside `scripts/`.

## Read first

| Doc | What it is |
|---|---|
| [`PROVENANCE.md`](PROVENANCE.md) | **Provenance by final panel** (Fig 3A-P, 4A-E, 5A-J): the script that drew or placed it, the staged input and the server source. Includes changes made after `FIGURE_SOURCES.md` was last edited (for example, Fig 5A's source changed to `frozen_refit_basis`). |
| [`FIGURE_SOURCES.md`](FIGURE_SOURCES.md) | **The narrative provenance record, 6-9 Nov**, copied unchanged from the grant folder: why each source was chosen, which stats file and column each panel's stars come from, and the corrections made along the way (lick-response AUC variant, hit-rate source, aim-panel stars). Read this before re-pulling any panel. |
| [`notes/DRAFT_LAYOUT_NOTES_20261106.md`](notes/DRAFT_LAYOUT_NOTES_20261106.md) | The figure folder's original README (6 Nov): house style (Arial, 11 pt bold panel letters, 7.4 in wide), how panels are inlined and id-namespaced, the evidence filter. Its panel maps describe the 6 Nov three-figure draft and are superseded by `FIGURE_SOURCES.md`. |
| [`notes/K08_panel_manifest.md`](notes/K08_panel_manifest.md) | Priya's panel selections, 6 Nov, with deck slide sources. |
| [`notes/K08_figure_plan_v3.md`](notes/K08_figure_plan_v3.md) | The figure plan and evidence-status review (only STANDS results carry an argument; WITHDRAWN / NOT ESTABLISHED results stay out). |
| [`session_scripts/README.md`](session_scripts/README.md) | **The last-mile scripts, recovered 17 Nov.** Thirteen scripts that ran on 9-10 Nov and wrote straight into the grant folder without ever being copied out: the two alternative paths panels, the Figure 3 relettering, the Figure 4D split and lighten, the Figure 5D cued panel and its caption swap, the legend readers, plus a re-export of the one CSV no saved code produced. Without these the submitted figures are not reproducible from this folder alone. |

## The code, by stage

All scripts run **from the grant figures folder as the working directory**: they import each other as siblings
(`sys.path.insert(0, '.')`) and read and write cwd-relative paths. They are kept here flat, unchanged, so they still
run that way.

**1. Panel renderers** (matplotlib; SVG with live text + PNG, into `K08_2026_grant_panels/`)

| Script | Panels |
|---|---|
| `grant_panels.py` | Shared style and helpers, not run directly: epoch colors (pre `#808080`, acute `#e0218a`, subacute `#d4a017`, chronic `#1b7f79`), animal markers, `epoch_strip`, `headroom`, `epoch_legend` / `fig_legend` (falls back to narrower shapes until it fits), `star_band` / `stars` (marks under the key, above the data), `save` |
| `render_grant_panels.py` | Fig 3 hit rate by spout and first-lick aim by spout; Fig 4 FR side specificity; Fig 5 frozen / refit / gap pooled, cued first-lick decoder offset, hemispheric balance (far contra), lick-response AUC (pooled), state-space plane and path over days. Reads `data/`, `aimstats/`, `hitstats/`, `frstats/` |
| `render_fig4_rows.py` | Figure 5 rows rendered as one canvas per row at final size, so side-by-side axes share a baseline; writes `rows_fig4_layout.json` (axes positions for the panel letters) |
| `make_ss_panel.py` | Fig 5 state-space change maps, full and cortex-wide-mode removed, kinematics + contact + face control (`frKBs`) |
| `paths_panel.py` | Fig 3P 3-D tongue and jaw paths, front view, 2 x 3, with cluster-permutation markers (`pose3d_paths_pooled.csv`, `paths3d_front_sig.csv`) |
| `make_camframes.py` | Fig 3N cam1 + cam4 frames with tongue / jaw / spout / nose (PS92 2026-08-07, far contra, trial 18; `camframes/`) |
| `make_cbars.py` | Stand-alone color bars for the FR map panels |
| `make_fr_panel.py` | FR-encoder position-unique variance panel (standalone draft panel, `FR_encoding_panel_DRAFT.svg`) |
| `grant_figures_k08_2026.py` | The first K08 renderer (6 Nov, previously `scripts/grant_figures_k08_2026.py`): `--list` prints the early manifest; draws the epoch-CSV panels. Superseded by the scripts above for the submitted figures; kept for its manifest |

**2. Figure compositors** (inline SVG panels into one page-width SVG, ids namespaced per panel)

| Script | Builds |
|---|---|
| `build_fig.py` | Library: `place` / `place_crop` / `place_png` / `caption` |
| `make_fig1.py` | `Figure3_behavior_DRAFT4.svg` (Research Strategy Figure 3) |
| `make_fig2.py` | `Figure4_facerhythm_DRAFT4.svg` (Figure 4) |
| `make_fig3.py` | `Figure5_decoder_encoder_DRAFT4.svg` (Figure 5; reads `rows_fig4_layout.json`) |
| `make_figs.py` | The 6 Nov "everything in" drafts (superseded) |

The script numbers are the drafts' numbers; the Research Strategy renumbered them 3 / 4 / 5. Set `NOCAP=1` to write
the `_panels` version without the caption.

The DRAFT4 SVGs were then taken the last mile by the scripts in [`session_scripts/`](session_scripts/README.md), not
by hand: the Figure 3 relettering and paths swap (`edit_svg.py`, `caption_fix.py`, `edit_v2.py`), the Figure 4D split
and lighten (`split_d.py`, `declip.py`), the Figure 5D cued panel and caption (`render_5d.py`, `cap5.py`).
Illustrator was used only to open the result and save the `.ai` (and for Figure 4, to re-save a more compact SVG).
The latest `.ai` files in the grant folder are `Figure3_behavior_relettered_v2.ai`,
`Figure4_facerhythm_Dsplit_light.ai` and `Figure5_decoder_encoder_cuedD.ai`.

**3. Post-processing**

| Script | Does |
|---|---|
| `fontnorm.py <svg...>` | Sets every text element to Arial so Illustrator substitutes nothing |
| `fr_mesh_underlay.py` | Adds a faint grey stroke to the FaceRhythm mesh points so the face outline shows where nothing changed |
| `svg2pdf.py <svg>` | PDF + full-page PNG preview via headless Chromium (playwright) |
| `fontfix.py <pdf...>` | Renames embedded Liberation Sans to Arial's PostScript names in the PDF |
| `thalamus_overlay_options.py` | Candidate Paxinos-outline overlays on the thalamus histology (Fig 3B). Input: `~/Documents/Sabatini Lab/Manuscripts/VLS stroke methods paper/histology_panels/PS46-55_sections/` on the laptop, outlines at `plates/plate_zoom_fig<N>.svg`, images at `<stem>_thalamus_zoom{,_noGFAP}.png`. `session_scripts/overlay.py` is the review-sheet version |

## Re-running

1. Copy the staged inputs into the grant figures folder (paths in `FIGURE_SOURCES.md`; server data under
   `MICROSCOPE/Priya/Widefield/...`, mainly `labcams/grant_figures/lmm_hidden_20261103/` and
   `labcams/grant_figures/epoch/data/`). Folder names the scripts expect: `data/`, `aimstats/`, `hitstats/`,
   `frstats/`, `frbasis/`, `ssdata/`, `recov/`, `fr/`, `ss/`, `fluoro/`, `airaster/`, `deck/`, `deck2/`, `camframes/`.
   Two of those are newer than `FIGURE_SOURCES.md` and are listed only in `PROVENANCE.md`: `frbasis/` from
   `scripts/frozen_refit_basis_compare.py` (`state_space_recovery/frozen_refit_basis/`) and `ssdata/` from
   `scripts/state_space_trajectories.py` (`state_space_recovery/trajectories_lick_svd/days_sessions.csv`).
2. `cd` into the grant figures folder and run with this folder's scripts on the path, e.g.
   `PYTHONPATH=<repo>/grants/k08_2026 python <repo>/grants/k08_2026/render_grant_panels.py`; or copy the scripts in.
3. Compose (`make_fig1/2/3.py`), then the matching `session_scripts/` step for that figure's last mile, then
   `fontnorm.py`, then `svg2pdf.py`, then `fontfix.py` on the PDF. `fontnorm.py` is what makes the grant-folder SVGs
   about 900 bytes smaller than the session copies; that is the only difference between them.

Dependencies beyond the `locanmf` env: `playwright` (+ `playwright install chromium`) for `svg2pdf.py` and
`thalamus_overlay_options.py`, `pikepdf` for `fontfix.py`, Pillow.

**Known not to run as-is:** `thalamus_overlay_options.py` and `session_scripts/overlay.py` have hard-coded input
paths from the sessions that wrote them (the staging mount `/mnt/user-data/uploads/...`; the real folder is in the
post-processing table above); `session_scripts/edit_svg.py` and `edit_v2.py` hard-code a session scratch directory
that no longer exists, so set `D` to the grant figures folder; `render_5d.py` and `session_scripts/update.py` likewise
point at staging paths for `lick_cued_ramp.pkl` and the `.docx`. `make_fig1.py` / `make_fig3.py` read deck crops
(`deck2/`) that were staged by hand. When a source is regenerated,
re-pull it from the path in `FIGURE_SOURCES.md` and re-run; nothing in the figures was hand-traced, so a re-pull is a
straight swap.
