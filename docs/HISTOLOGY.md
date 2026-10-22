# Histology: post-stroke lesion burden (PS92-95)

Added 2026-10-21. Decisions are logged in `DECISIONS.md` under the same date.

## What it measures

Per coronal section, **100 x ipsilesional / contralesional caudoputamen (striatal) area**. Lower means
more striatal tissue lost. This is the same quantity as Figure 1F of the VLS stroke methods paper
("Caudoputamen ipsi/contra % area", PS46-55), drawn the same way: faded points are sections, and black
is the animal mean +/- SEM across sections.

## Where the numbers come from

| step | what | where |
|---|---|---|
| tissue | DAPI / GFAP / NeuN, VS200 10x scans | `MICROSCOPE/Priya/VS200/202610{19,20,21}_PS9x` |
| outlines | one outline per striatum, two per section, drawn by Priya in QuPath 0.6 | `/Users/Priya2/Documents/data/Immuno/PS92-95_widefield_stroke_cohort` |
| import | QuPath project -> one row per section | `scripts/import_histology_areas.py` -> `configs/histology_striatum_area.csv` |
| figure | plot, per-animal table, CSV, meta | `wfield_local/histology_area.py` |

QuPath does not need to be running, and Java isn't needed: `wfield_local/qupath_io.py` reads the
`data.qpdata` files directly. QuPath writes annotation ROIs in two encodings, a WKB `byte[]` (brush /
wand / merged) and a pair of `float[]` x / y arrays (polygon tool), and both are handled. Two checks
make a silent drop impossible:

* the number of outlines decoded per image must equal QuPath's own count in `summary.json`, or the
  import raises an error;
* areas were checked against the area QuPath stores beside each ROI (e.g. 3,815,749.5 px^2 stored vs
  3,815,750 px^2 recomputed).

## Rules (and why)

* **Pairing.** Outlines are unnamed and unclassified, so a section's two outlines are paired as
  mutual nearest neighbours by centroid. Each pair must sit on one slide row (|dy| < 1500 px) at
  hemisphere spacing (3000-9000 px at 0.65 um/px), or the import fails. A mis-pairing is caught
  rather than turned into a plausible ratio.
* **Ipsi = the smaller outline of each section** (Priya: "the one with smaller area will always be
  ipsi - some sections are mounted in flipped orientation"). Image side cannot be used: both PS93
  slides and the first section on the PS92 slide are mounted flipped. Checked against the GFAP scar on
  every slide thumbnail on 2026-10-21, and it agreed on all 24 sections.
  * Consequence: the ratio is <= 100% by construction, so it measures lesion-driven loss and cannot
    show a contralesional > ipsilesional asymmetry. That is harmless for these animals (every section
    is <= 69%), but this rule should not be applied to shams or unlesioned controls.
* **Section number = reading order on the slide** (row, then x). It is not a verified AP order.
* **Colours** come from `configs/animals.yaml` (the paper's notebook used matplotlib's cycle). The
  jitter is seeded, so a re-render does not move the points.

## Results (2026-10-21, after exclusions)

| animal | sections | mean % | SEM | range |
|---|---|---|---|---|
| PS92 | 7 | 46.7 | 1.4 | 41.9-51.1 |
| PS93 | 5 | 45.0 | 3.7 | 37.7-56.6 |
| PS94 | 5 | 67.7 | 1.1 | 64.9-70.3 |
| PS95 | 6 | 81.9 | 0.4 | 79.9-82.9 |

**Excluded:** PS93 section 2 (`PS93_DAPI_GFAP_NeuN_01.vsi` #2, 69.1%). Priya: it is the most posterior section and
lies outside the central stroke (only the lesion edge), so it inflates the % area. Confirmed from her QuPath
screenshot. Exclusions live in `configs/histology_section_exclusions.csv` (animal, image, slide_section, reason).
They are dropped from the plot and statistics but kept in the output CSV with `included=False`. With it, PS93 was
49.0 +/- 5.0 (n=6).

## Running it

```bash
# after editing outlines in QuPath (save the image so data.qpdata is rewritten)
python -m scripts.import_histology_areas [--project /path/to/qupath_project]
python -m wfield_local.histology_area             # -> <microscope>/Widefield/histology
```

Outputs: `histology_striatum_ipsi_contra_pct_area.{png,svg,csv}` + `data/..._meta.csv`. By Priya's
request, the PNG / SVG / CSV are also kept in the QuPath project folder
(`Documents/data/Immuno/PS92-95_widefield_stroke_cohort/`). An earlier copy also sits in
`labcams/grant_figures/histology/`.

## Provenance of the paper version

The methods-paper panel's plotting code was not in either repo. It is the notebook
`/Users/Priya2/Documents/data/Immuno/Stroke area quantification plots.ipynb`, with per-section ratios
typed in by hand (`stroke_orofacial_pipeline_outputs/vls_stroke_paper/recovered_panels/` holds a
re-render recovered from the SVG). That notebook was the reference for this one.

---

# Stroke-extent schematic (PS92-95)

Added 2026-10-21. Package `wfield_local/histology_map/`; cohort data `configs/histology_map/PS92-95/`; figure
`<MICROSCOPE>/Priya/Widefield/histology/PS92-95_stroke_schematic_v2.{png,svg,pdf}` (+ `_sections.json`) and
`PS92-95_scale_check.png`. Each animal's stroke is drawn on outline-only Paxinos & Franklin plates, lesion left,
with the animals overlaid with transparency in their `animals.yaml` colours.

## Pipeline

| step | module | runs where | output |
|---|---|---|---|
| crop sections at level 3 (5.2 um/px), all channels | `histology_map.ets` (+ `jobs`) | Mac / rig (needs the VS200 scans) | `<crops_dir>/<animal>_s<n>_L3.npz` |
| standardise + Claude's suggestion + painter page | `prepare` | anywhere with crops + QuPath project | `sections.json` transforms, painter HTML |
| Priya paints masks | Stroke Painter artifact (`painter_template.html`) | claude.ai | `masks` collection -> `configs/histology_map/<cohort>/masks/` |
| section -> plate fit at Priya's AP levels | `fit` | anywhere | `sections.json` (figure, kx, ky, ty, err) |
| schematic | `schematic` | anywhere | png / svg / pdf + per-section areas |

`python -m wfield_local.histology_map --help` lists the commands. The scan reader needs only numpy + Pillow
(JPEG2000 tiles), so it runs on the Mac's Cowork VM where Bio-Formats / slideio are unavailable.

## Decisions (Priya, 2026-10-21; each one corrected an earlier version)

* **Atlas**: Paxinos & Franklin 2nd ed. vector plates (Fig 13-48 = PDF pages 63-98). Outlines only: labels, label
  leader lines, the unlabelled reference circle in the right CPu, and stray line art outside the brain removed;
  ventricles light grey. Cite the atlas in the legend and check the journal's permission policy for adapted plates.
* **Lesion always on the LEFT**, whatever the mounting (flipped sections mirrored).
* **Midline = dorsal interhemispheric fissure**, not the outline's mirror-symmetry axis: atrophy pulls the symmetry
  axis 0.1-0.3 mm toward the intact side (largest in PS93), which then mis-scales the intact hemisphere.
* **Scaling on the contralesional hemisphere only**, anchored on the midline, independent x / y, by boundary
  distance (outline + striatum), range 0.82-1.15. Area-overlap scoring over-sized every section. When striatum and
  cortex edge disagree the **striatum wins** (PS93's cortex edge then sits inside the atlas outline; x ~0.81-0.86).
  **The lesioned hemisphere is never rescaled**: it rides the intact side's transform, so lost volume shows as
  absent territory instead of being stretched away.
* **What counts as stroke** (for the automatic suggestion, `footprint.py`): NeuN-blank tissue, the GFAP scar ring
  with low NeuN, and the autofluorescent necrotic core, each judged against the mirror point; normal GPe (low NeuN
  density / higher GFAP on both sides) is not stroke, a necrotic core extending into GPe is. Footprint = the visible
  lesion, always fully included, spilled laterally then dorsally into territory absent ipsilaterally but present
  contralaterally (the mirrored contra striatum anchored on the surviving striatum's intact medial / dorsomedial
  border). Medial and usually dorsal striatum spared; DLS involvement in some animals (a fibre-hit vessel). Because
  of encephalomalacia the absent area exceeds the visible lesion; that is expected.
* **Masks in the figure = Priya's painted masks** (24 sections incl. PS93 s2). PROVISIONAL: she painted them
  quickly ("don't treat my masks as 100% ground truth"); they are a reference, not a calibration target.
  Her masks vs the automatic suggestion: IoU 0.3-0.7.
* **AP levels = Priya's figure numbers** (`priya_figures` in sections.json; "?" = `priya_unsure`: PS93 s5, s6,
  PS95 s1). Ranges resolved by the better fit (PS92 s2 -> Fig 29, s5 -> Fig 30). The automatic AP estimate (shape
  fit, and the atlas Nissl / AChE photomicrographs at PDF page = figure + 182, odd = Nissl) narrowed the range to
  +0.50..-0.10 but could not separate neighbouring 0.12-mm levels; it is not used.
* **No skipped levels**: levels between an animal's measured ones are filled by signed-distance interpolation,
  drawn dashed / lighter and labelled "interp.".

## Reproducibility

`sections.json` pins every transform the painted masks depend on (flip, rotation angle + centre, midline, painter
crop + scale). Re-running `schematic` from the repo reproduces the published areas exactly. Re-running `fit`
reproduces the stored transforms to ~0.01 in scale and ~0.06 mm in ty (the plate masks were cleaned after the
original fit); `schematic` uses the stored values.

### Hand-edited atlas shapes (2026-10-22, supersedes the painted-mask route)

After several rounds of automatic placement Priya edited every stroke directly on its Paxinos plate in a polygon
editor (`wfield_local/histology_map/atlas_editor_template.html`: move, vertex drag, rotate, scale/stretch,
paint/erase brush; a progress bar compares the shape with that section's lost striatal area, contra minus surviving
ipsi from the QuPath outlines, scaled into atlas mm² by the section's contra-hemisphere kx·ky). Plate levels were
re-checked in a click-through review against the chosen plate ±2 (`level_review_template.html`, Nissl/AChE photos
under the line art). The saved shapes are `configs/histology_map/<cohort>/atlas_polygons/<section>.json`
(`poly` in plate mm: x from the midline, negative = left; y from the page top) and `schematic` draws from them
whenever that folder exists. Placement rules she applied: lateral stroke edge against the striatum's lateral border
(corpus callosum / external capsule); blob extending dorsolaterally from the core, only a touch ventral.

### Figure versions (2026-10-22)
- All levels, coloured outlines, interpolated levels dashed: `python -m wfield_local.histology_map schematic --cohort PS92-95 --style outline` -> `PS92-95_stroke_schematic_v3_outlines.*` (filled version without `--style`).
- Selected levels, real sections only (neighbour-level fill-ins listed in `_sources.json`): `scripts/histology_stroke_extent_figure.py <atlas_polygons dir> <out_stem> PS92-95` -> `Widefield_stroke_extent_PS92-95.*`.

## Stroke outlines v4 (2026-10-22)
- Painted in the PS92-95 Stroke Outline Editor (claude.ai artifact, db `outlines`): every kept section, separate stroke
  regions allowed (perforating-vessel strokes can sit dorsal to the main lesion), section flips recorded.
  `configs/histology_map/PS92-95/atlas_polygons_v4/<section>.json` holds `polys` (list of regions, plate mm) and
  `exclude` (no stroke on that section). `cohort.yaml polygons_dir` selects v4; v3 stays in `atlas_polygons/`.
  PS95 e36s4 was not re-saved in the editor and carries its v3 outline (PS95_s3).
- Orientation: all four strokes are left hemisphere; the striatal lesion, thalamic and GPe / cerebral peduncle gliosis
  and PS93's anterior motor-cortex injury are ipsilesional (Priya), so sections are shown lesion left.
- Outputs (MICROSCOPE `Priya/Widefield/histology/`): `PS92-95_stroke_schematic_v4{,_outlines}.*`,
  `Widefield_stroke_extent_PS92-95.*`, and `histology_panels/` (overview grid of all 186 kept sections, plates with v4).
- `plates.brain_lines`: Fig 15 draws the brain outline as two halves; their union is used.
