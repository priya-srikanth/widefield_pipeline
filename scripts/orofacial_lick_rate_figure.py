"""Lick rate per spout across epochs, as its own figure (Priya 11-07: "do we have a lick rate figure? ... if not let's
add"). Two rows of the cohort engagement figure, redrawn from the tables that render already wrote -- no recompute:

  * within-bout inter-lick interval (ILI, ms; median of intervals < 300 ms inside a trial) -- the lick rhythm;
  * fraction of licks with a neighbour < 250 ms -- the fast-lick share (the video counterpart of the FaceRhythm
    fast-lick factors).

Same rules as `orofacial_cohort.fig_cohort_epochs_by_position`: each session's change from its animal's OWN pre mean
(late pre, >= 0801), engaged trials, all post-cue LP licks, session MWU (BH-FDR within metric) + animal-weighted
bootstrap marks from `stats/change_vs_pre_stats.csv`.

    python scripts/orofacial_lick_rate_figure.py [--run lp_r4_heatmap_20261013__clean_v36]
"""
import argparse
from pathlib import Path

import pandas as pd

from wfield_local import orofacial_cohort as oc

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lp_r4_heatmap_20261013__clean_v36")
    a = ap.parse_args(argv)
    base = ROOT / a.run / "cohort"
    tables, figs = base / "tables" / "prepost", base / "figures" / "prepost"
    Pm = pd.read_csv(tables / "position_metrics.csv", dtype={"date": str})
    ST = pd.read_csv(tables / "stats" / "change_vs_pre_stats.csv")
    n = Pm.drop_duplicates(["animal", "date"]).groupby("animal").size()
    title = (f"Lick rate per spout, pre vs post stroke -- {a.run} -- ENGAGED trials only -- baseline: LATE pre only "
             f"(>= 0801) -- sessions: " + ", ".join(f"{k} {v}" for k, v in n.items()) + "\n"
             "ILI = median within-trial interval < 300 ms; fast licks = licks with a neighbour < 250 ms")
    out = figs / "cohort_lick_rate_by_position.png"
    oc.fig_cohort_epochs_by_position(Pm, ("ili_ms", "neighbor_frac"), out, title, ST)
    print(out)


if __name__ == "__main__":
    main()
