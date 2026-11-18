"""Quiet-period events: production (v2_quiet_lp_20261117) vs cam1-guard re-run (v2_quiet_lp_c1guard_20261117). Per animal
FIRST (rule 8): epoch means of quiet minutes and event rates, and each animal's change vs its OWN pre (chronic first, as
Priya asked), then the LMM epoch terms side by side. usage: python compare_old_new.py [out_md] [new_dirname]"""
import sys
from pathlib import Path

import pandas as pd

Q = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_"
         "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/cohort1_quiet_spikes")
OLD = Q / "v2_quiet_lp_20261117"
NEW = Q / (sys.argv[2] if len(sys.argv) > 2 else "v2_quiet_lp_c1guard_20261117")   # e.g. the fixed-k summary
EPS = ["pre", "acute", "subacute", "chronic"]
COLS = ["quiet_min", "orofacial|rate", "orofacial|rate_both", "limb|rate"]


def epoch_means(d):
    return d.groupby(["animal", "epoch"])[COLS].mean().unstack("epoch")


def main():
    o, n = (pd.read_csv(p / "quiet_events_sessions.csv") for p in (OLD, NEW))
    out = [f"sessions: old {len(o)}, new {len(n)}"]
    mo, mn = epoch_means(o), epoch_means(n)
    for c in COLS:
        t = pd.DataFrame({(ep, k): (mo if k == "old" else mn)[(c, ep)] for ep in EPS for k in ("old", "new")})
        for ep in EPS[1:]:
            for k, m in (("old", mo), ("new", mn)):
                t[(f"{ep}-pre", k)] = m[(c, ep)] - m[(c, "pre")]
        out += [f"\n== {c}: per-animal epoch means and change vs own pre (old -> new)",
                t.round(3).to_string()]
    lo, ln = (pd.read_csv(p / "lmm_v2.csv") for p in (OLD, NEW))
    k = ["model", "measure", "column", "term"]
    L = lo.merge(ln, on=k, suffixes=("_old", "_new"))
    L = L[(L.model == "epoch") & L.term.str.contains("epoch") & L.column.isin(["session", "orofacial", "limb",
                                                                                 "orofacial_left", "orofacial_right"])]
    out += ["\n== LMM epoch terms vs August pre (rate = log10(rate + 0.2)), old -> new",
            L[k[1:] + ["coef_old", "coef_new", "q_old", "q_new"]].round(4).to_string(index=False)]
    for p in (OLD, NEW):
        log = (p / "logs" / "quiet_v2.log").read_text(encoding="utf-8", errors="replace").splitlines()
        out += [f"{p.name}: " + next((x for x in log if x.startswith("chosen k")), "chosen k: (not found)")]
    txt = "\n".join(out)
    print(txt)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text("```\n" + txt + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
