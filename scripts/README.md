# scripts/ — kept one-off / recovery utilities

Not part of the nightly pipeline (that is `python -m wfield_local.preprocess` /
`preprocess_deck` on the imaging box and `nightly_figs` on the analysis box — see
`wfield_local/README.md`). These are occasional-use tools kept because they may be
needed again; they are not imported by the package.

- `_qc_from_standby.py` — regenerate motion-correction QC figures (with bottom example
  images) for dates whose raw movies were already cleaned off `E:`. Reads the corrected
  `.bin` **and** raw `.dat` from **M: standby**, and writes QC on **N:**. Auto-discovers
  sessions. Usage: `python scripts/_qc_from_standby.py 20260605 20260606 ...`.
