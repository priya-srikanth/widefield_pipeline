"""Stroke-extent maps on Paxinos & Franklin atlas plates (the "stroke schematic"), from VS200 histology.

Built with Priya 2026-10-21 for the PS92-95 widefield cohort; designed to be re-run for other cohorts
(PS46-55 next). Full method + every decision: ``docs/HISTOLOGY.md`` ("Stroke-extent schematic").

Pipeline (each step is a module; ``python -m wfield_local.histology_map --help`` runs them):

  ets        read a region of an Olympus .vsi/.ets pyramid (JPEG2000 tiles) without Bio-Formats.
             Run where the scans are (the Mac / a rig box): crops every section at level 3 (8x, 5.2 um/px).
  plates     Paxinos & Franklin vector plates -> outline-only line art (labels, leader lines and the unlabelled
             reference circle removed) + mm-calibrated brain / CPu masks.
  sections   per-section standardisation: lesion LEFT (flipped mounts mirrored), rotation from outline
             symmetry, and the ANATOMICAL midline (dorsal interhemispheric fissure).
  footprint  automatic stroke suggestion (NeuN-blank / GFAP scar / autofluorescent core, each judged against the
             mirror point; spill into territory absent ipsi but present contra) -- a starting point for painting.
  painter    the QC page where Priya paints the masks; build its data and decode what it saved.
  fit        section -> plate: contralesional hemisphere ONLY, scaled from the midline, independent x / y,
             boundary distance to the plate brain + CPu outlines, striatum favoured.
  schematic  masks -> plates (same transform), animals overlaid, gaps between measured levels interpolated.

Data that defines a published figure lives in ``configs/histology_map/<cohort>/``: ``sections.json``
(per-section transforms + AP level + fit), ``masks/<animal>_s<n>.json`` (the painted masks).
"""
UM_PER_PX_L3 = 0.65 * 8          # VS200 10x pixel (0.65 um) x 2**3 pyramid level
