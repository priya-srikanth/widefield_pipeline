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
