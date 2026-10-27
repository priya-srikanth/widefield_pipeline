# Imaging results page — source and rebuild

Published page: https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV (v34, 2026-10-27: v27 = section.position figure
numbers, figure titles, full-method captions; v28-29 add section 14, the Haufe decoder maps; v30 maps its encoding-page cross-reference to that page's 10-27 renumbering (3.33-3.35 -> 3.64-3.66, linked by stable id; v31 -> 3.69-3.71 after its v130 insertions; v34 -> 15.6-15.8 after its v136 section split); v32 = stable figure ids; v33 = section 1 holds only the refit decoder (frozen pre-cue figure moved to 2.1, refit by-position Figs 1.1-1.2 added); format rule in `docs/templates/README.md`).

**The figure PNGs are not in git** (83 files, ~188 MB). The full source, PNGs included, is on the share at
`M:/MICROSCOPE/Priya/Widefield/imaging_page_src/` (`imaging_page/` = exactly what is published). This folder holds
the HTML, the scripts and the provenance notes.

## Rebuild

Run from this folder (each script edits `imaging_page/imaging_results.html` in place, and asserts before writing):

```
cp imaging_results_v12_backup.html imaging_page/imaging_results.html
python p_img_s7v3.py       # section 7: the window x reference x arm map grid
python p_img_winorder.py   # section 7 window order
python p_img_s1face.py     # section 1 face figure
python p_img_renum.py      # Fig N.k numbering, ids fig<N>-<k>, cross-refs, per-section sidebar; writes imaging_fig_map.csv
python p_img_titles.py     # figure titles + method-first captions + script line; caption corrections
python p_img_haufe.py      # section 14: Haufe decoder maps (families 14 / 14z / 14pa, 9 figures)
python p_img_stableids.py  # stable ids #f-<image name>; old #figN-k kept as anchors; writes imaging_fig_ids.csv
python p_img_s1refit.py    # section 1 = refit decoder only (5r figures in, frozen pre-cue fig to 2.1); renumbers 1-2; rewrites the registry
```

Checked 2026-10-26 (through p_img_titles): this chain reproduces the published `imaging_page/imaging_results.html` byte for byte.
`fig2_attempt_decoder.png` and `fig2b_attempt_decoder_epochs.png` are on the published artifact but not in the local
PNG folder; publish with `files` for the changed PNGs only, so they are kept. To publish, copy the result next to the PNGs in `imaging_page_src/imaging_page/` and publish that HTML to the URL above.
The share copy of `imaging_page/imaging_results.html` is still v27 (not overwritten); the repo copy here is current (v34).

## Linking to a figure (2026-10-27)

**Link by STABLE id: `#f-<image name>`** (e.g. `#f-drift_REST_cue_contact`); it never changes when figures are
renumbered or reordered. `imaging_fig_ids.csv` (fid, number, title) is the registry: fid -> current number.
`#figN-k` still works, but means the figure that had number N.k when stable ids were introduced (v32), not
whatever is N.k later. Same scheme as the encoding goal page (`_goal_page/fig_ids.csv`). A printed number next
to a link is still a number and must be refreshed from the registry after a renumbering.
**`imaging_fig_ids.csv` has a reader outside this folder**: the PI deck's figure provenance (e3's `edit11.py`) takes
imaging figure numbers from it. Any page change that moves numbers must re-run the chain through `p_img_stableids.py`
and commit the regenerated CSV with the page.

- `imaging_fig_map.csv` — old figure label -> new N.k number (for reading older notes that cite e.g. "Fig 7.2c").
- `page_figs.py` — copies and downscales a source figure into `imaging_page/`.
- `history/` — the earlier patch scripts already baked into `imaging_results_v12_backup.html` (kept as the record).
- `provenance/` — per-figure source PNG and the code-traced methods behind each caption (`prov_*.md`,
  `prov_table.tsv`). Captions were written from these, not from the old captions.

## Known, not changed (2026-10-26)

- Figs 6.1-6.3 and 10.1 PNGs date from 10-07 and use pre = 44 sessions (June + August); later tables use August pre.
- Stale text inside some figure images (noted in their captions, not re-rendered): section-7 map subtitles say
  "PER ROW" scale (one scale is used); Fig 0.1 note says only PS92 has stabilised; Fig 12.1 legend says
  "circular-shift" (the null is a block permutation).
- Caption numbers without a sidecar table to check against: Fig 1.1, 4.2, 9.4.
