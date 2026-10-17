"""Re-draw every session's qc.png from its saved outputs so each also gets svg/qc.svg (Priya 10-17: PNG + SVG for all
orofacial figures). New sessions get both from `orofacial_outputs process` directly; this is for folders made before.

    python scripts/orofacial_svg_backfill.py [--run lp_r4_heatmap_20261013__clean_v35]
"""
import argparse

from wfield_local import orofacial_outputs as oo
from wfield_local.paths import PathResolver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lp_r4_heatmap_20261013__clean_v35")
    a = ap.parse_args()
    rv = PathResolver()
    base = oo.root(rv) / a.run / "sessions"
    for sd in sorted(p for p in base.glob("*/*") if (p / "manifest.json").exists()):
        animal, date = sd.parent.name, sd.name
        try:
            S = oo.load_session(rv, a.run, animal, date)
            c = S["clean"]
            oo.qc_figure(S["index"], S["pose"], c.tongue_y.to_numpy(), c.tongue_x.to_numpy(), c.jaw_y.to_numpy(),
                         sd / "qc.png", f"{animal} {date} ({a.run})")
            print("ok", animal, date, flush=True)
        except Exception as e:                       # report and continue; one bad folder must not stop the rest
            print("FAIL", animal, date, repr(e), flush=True)


if __name__ == "__main__":
    main()
