"""Re-export `paths3d_front_sig.csv`, the marker table read by `paths_panel.py`, `paths_narrow.py` and
`paths_short.py` (the Figure 3 3-D paths panel). No saved script wrote the original: it was a one-off export of
`kinematics_pooled_figs._paths3d_sig`, which the pipeline computes for the figure but never writes to CSV.

VERIFIED 2026-11-17 against the staged `paths3d_front_sig.csv` from the submitted figure: 1476 / 1476 rows agree,
960 significant bins in each, on an identical (position, epoch, measure, x) grid.

Inputs: `curves.csv.gz` only. Everything else is derived from it, so this does not need the pose3d kinematics
snapshot or `sessions3d()`. By default it reads the routed, gate-named path, the same as `fig_paths3d`; pass a file
to read a copy instead (the archived one is in
`MICROSCOPE/Priya/Widefield/grant_figures_archive/k08_2026/staged_inputs/paths/curves.csv.gz`).

    PYTHONPATH=<repo> python export_paths3d_front_sig.py [curves.csv.gz] [-o paths3d_front_sig.csv]

The test, unchanged from the pipeline: each session's phase curve minus its animal's own August-pre mean curve,
`cluster_permutation` of the epoch's sessions against the August-pre sessions, labels permuted within animal,
at ALPHA / 3 because three coordinates are tested per path. The FRONT view uses two of them, `side` and `down`,
so the table is filtered to those after the test; the /3 correction stays as the pipeline applies it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
import kinematics_pooled_figs as K  # noqa: E402
from wfield_local import config  # noqa: E402

FRONT = ("side", "down")
MEASURES = ["side", "out", "down", "j_side", "j_out", "j_down"]


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("curves", nargs="?", help="curves.csv.gz; default = the routed, gate-named path")
    a.add_argument("-o", "--out", default="paths3d_front_sig.csv")
    a.add_argument("--all-measures", action="store_true", help="keep out / jaw rows too, as _paths3d_sig returns them")
    o = a.parse_args(argv)

    src = Path(o.curves) if o.curves else K._tc().kin_out_dir(K.pose3d_dir(), "phase") / "curves.csv.gz"
    C = pd.read_csv(src, dtype={"date": str, "epoch": str})

    # the gate fig_paths3d applies, verbatim
    C = C[(C.kind == "norm") & (C.lickset == "all") & C.measure.isin(MEASURES) & (C["cov"] >= 0.5)]
    aug = C[(C.epoch == "pre") & (~C.date.astype(str).map(config.early_pre))]
    print("gated rows %d, August-pre sessions %d" % (len(C), aug.session.nunique()))

    sig = K._paths3d_sig(C, aug)
    out = pd.concat([pd.DataFrame({"position": p, "epoch": e, "measure": m,
                                   "x": s.index.astype(float), "sig": s.to_numpy()})
                     for (p, e, m), s in sig.items()], ignore_index=True)
    if not o.all_measures:
        out = out[out.measure.isin(FRONT)]
    out = out.sort_values(["position", "epoch", "measure", "x"]).reset_index(drop=True)
    out.to_csv(o.out, index=False)
    print("wrote %s: %d rows, %d significant bins" % (o.out, len(out), int(out.sig.sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
