"""DOES THE CHRONIC REMAPPING TRACK RECOVERY? (Priya 10-25, analysis 2 of the synthesis.) Session mixed models
(`wfield_local.mixed_models`, the shared definition: time y ~ post + post:lday, behaviour y ~ post + post:deficit, epoch;
random intercept / step / slope per animal when estimable) with the two strongest chronic results as the OUTCOME:

  residual   frozen-encoder residual (data - prediction, evoked, mean of six spouts) per session, from the specificity
             control's sessions table (`scripts/specificity_control.py`): raw pre-SD units and GAIN-CORRECTED (global scale
             estimated on control areas removed), for orofacial contra / ipsi, control, and orofacial - control
  layer2     Layer 2 change_ref per session (combined fit; pre rows = the LATE-pre sessions, i.e. drift from early pre,
             so `post` is the change beyond pre-stroke drift), same area groups

Behaviour model deficit: `ttc` (far-contra time to contact, primary) and `contact` (both deficit definitions, as
imaging_lmm). Reading: post:deficit < 0 with post > 0 = the remapping is largest when behaviour is worst and shrinks with
recovery (a disruption); post:deficit ~ 0 with post > 0 = present at full recovery, independent of the deficit (a new
steady state); post:deficit > 0 would mean more remapping in worse sessions only if deficit is coded worse = larger
(it is: 0 = pre level, larger = worse). Compensation predicts remapping that GROWS as the deficit shrinks (time slope
> 0, deficit slope < 0).

Usage: python scripts/remap_lmm.py [--tag gate-attempted_v7g3_t3d_frA] [--layer2 allsession_encoder_v7g3_t3d_frA]
Output: <E>/remap_lmm/<tag>/remap_lmm_<deficit>.csv (+ summary)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402
from specificity_control import GROUPS, group_means  # noqa: E402


def residual_long(tag: str) -> pd.DataFrame:
    T = pd.read_csv(E / "specificity_control" / tag / "sessions.csv")
    G = group_means(T, ("residual_z", "resid_gc_z"))
    G = G[G.area.isin(["orofacial contra", "orofacial ipsi", "control", "orofacial - control"])]
    L = G.melt(id_vars=["label", "window", "area"], value_vars=["residual_z", "resid_gc_z"], var_name="q", value_name="y")
    return L.assign(source="residual", measure=L.q + " | " + L.window + " | " + L.area)[["label", "source", "measure", "y"]]


def layer2_long(folder: str) -> pd.DataFrame:
    L = pd.read_csv(E / folder / "allsession_changes_by_session.csv")
    L = L[(L.fit == "combined") & (L.epoch != "pre")].copy()        # pre rows = late pre (drift from early pre)
    L["y"] = L.change_ref / 1e3
    rows = []
    for _an, g in L.groupby("animal"):
        nlab = g.label.nunique()
        have = set(g.groupby("area").label.nunique().loc[lambda s, n=nlab: s == n].index)
        for gname, areas in GROUPS.items():
            use = [x for x in areas if x in have]
            if use:
                rows.append(g[g.area.isin(use)].groupby("label").y.mean().reset_index().assign(area=gname))
    G = pd.concat(rows, ignore_index=True)
    o, c = G[G.area == "orofacial"].set_index("label").y, G[G.area == "control"].set_index("label").y
    G = pd.concat([G, (o - c).rename("y").reset_index().assign(area="orofacial - control")], ignore_index=True)
    G = G[G.area.isin(["orofacial contra", "orofacial ipsi", "control", "orofacial - control"])]
    return G.assign(source="layer2", measure="change_ref | " + G.area)[["label", "source", "measure", "y"]]


def main(argv=None) -> int:
    from wfield_local import mixed_models as mm
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    ap.add_argument("--layer2", default="allsession_encoder_v7g3_t3d_frA")
    ap.add_argument("--deficits", nargs="+", default=["ttc", "contact"])
    a = ap.parse_args(argv)
    out = E / "remap_lmm" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    L = mm.add_time_covariates(pd.concat([residual_long(a.tag), layer2_long(a.layer2)], ignore_index=True))
    pd.set_option("display.width", 230)
    for dname in a.deficits:
        Ld = mm.add_deficit(L, mm.deficit(E / "qc", dname))
        T = mm.lmm_table(Ld, ["source", "measure"], ("time", "behaviour", "epoch"), fdr_by=("source",),
                         print_animals=(dname == a.deficits[0]))
        T.to_csv(out / f"remap_lmm_{dname}.csv", index=False, float_format="%.5g")
        mm.summary_table(T, ["source", "measure"]).to_csv(out / f"remap_lmm_{dname}_summary.csv")
        for model in ("time", "behaviour"):
            S = T[(T.model == model) & (T.term != "Intercept")]
            print(f"\n=== {model} (deficit {dname})\n" + S.pivot_table(index=["source", "measure"], columns="term",
                                                                     values=["coef", "z", "q"]).round(3).to_string())
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
