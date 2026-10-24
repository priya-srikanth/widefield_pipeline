"""FaceRhythm ANALYSIS layer: the shared pieces every FaceRhythm readout script uses.

FaceRhythm (FR) turns cam4 face video into a dense mesh of tracked points and their spectrograms (O2,
`scripts/o2_face_rhythm.py`); a non-negative CP decomposition of pre-stroke sessions gives a FROZEN factor set (the
"master", `scripts/face_rhythm_factors.py`, run on O2) that every session is projected onto (``<set>/proj/``). The
readout scripts (`scripts/face_rhythm_*.py`) turn those projections into figures; `python -m wfield_local.face_rhythm
list` prints every analysis with the question it answers, and the guide is docs/FACE_RHYTHM_GUIDE.md.

Modules (import them, do not copy them into a script):

  common   cohort constants (epochs, spout positions in ONE order, contra / ipsi sides), the default factor set, the
           projected-session table (`sessions`), session labels and days since stroke, the repo palettes
  face     face movement from the frozen point PCs: per-frame contra / ipsi mean speed, event-aligned averages,
           event-window weights (per-point maps)
  licks    LP lick table with bouts (re-exported), bout classes, first-lick latency
  stats    log2 change vs each animal's own pre, the per-animal epoch table + `analysis_kit.boot_delta` (rule 8 first),
           Holm marks (* / **)
  __main__ the analysis catalogue: `python -m wfield_local.face_rhythm list | show <name> | run <name> [args]`

Modules that other code imports by name stay flat in `wfield_local/` and are NOT moved here: `face_rhythm_outputs`
and `fr_groups` (the encoding model), `face_rhythm_timing` and `face_rhythm_multiview` (copied next to the O2 factor
script as sidecars).
"""
