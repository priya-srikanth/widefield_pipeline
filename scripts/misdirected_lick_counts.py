"""Feasibility counts for the misdirected-lick question (Priya 2026-10-19: "in the acute far_contra there are times when
the animal clearly has a lick but it doesn't contact the spout (left deviated) -- does that activity look like a planned
+ executed far right lick or the lick that it executes?").

    python scripts/misdirected_lick_counts.py [--run lp_r4_heatmap_20261013__clean_v36] [--dev 15]

Per animal x epoch x spout position (ENGAGED trials, measured lick peaks with reach >= the encoding model's
MIN_REACH_PX): all licks, contacted, no-contact, and no-contact licks whose executed peak angle deviates from the spout
angle by > DEV deg toward IPSI (`ipsi_dev`, + = ipsi = image right) / toward contra / within +-DEV (on target but short).
Also, pre-stroke, CONTACTED licks binned by executed angle: the template pool for "what a successful lick to where the
misdirected tongue actually went looks like". August-only pre (baseline rule).

Output -> <LP_analysis>/<run>/cohort/tables/misdirected_licks/{counts.csv, pre_contacted_by_angle.csv}
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--dev", type=float, default=15.0, help="deviation threshold (deg) for 'misdirected'")
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = oo.root(rv) / a.run / "cohort" / "tables" / "misdirected_licks"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in ak.curated_sessions():
        lab = s["label"]
        animal, date = lab[:4], "2026" + lab[5:]
        ep = epochs.epoch_of(lab)
        if ep not in ("pre", "acute", "subacute", "chronic") or (ep == "pre" and lab[5:] < oc.LATE_PRE_FROM):
            continue
        sd = oo.session_dir(rv, a.run, animal, date)
        if not (sd / "licks.csv").exists():
            continue
        keep, _gate = oc.engagement_keep(animal, date, sd)
        L = pd.read_csv(sd / "licks.csv")
        if keep is not None:
            L = L[L.trial_id.isin(keep)]
        ok = (~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX) & np.isfinite(L.peak_angle_deg)
        L = L[ok].assign(pos=[oc.pos_key(p, "L") for p in L[ok].position])
        dev = L.peak_angle_deg - L.spout_angle_deg                  # + = toward ipsi (image right)
        nc = ~L.contact.astype(bool)
        L = L.assign(nc=nc, ipsi_dev=nc & (dev > a.dev), contra_dev=nc & (dev < -a.dev),
                     on_target_nc=nc & (dev.abs() <= a.dev))
        g = L.groupby("pos").agg(licks=("nc", "size"), no_contact=("nc", "sum"), ipsi_dev=("ipsi_dev", "sum"),
                                 contra_dev=("contra_dev", "sum"), on_target_nc=("on_target_nc", "sum"))
        g = g.reset_index().assign(label=lab, animal=animal, epoch=ep)
        rows.append(g)
        if ep == "pre":
            C = L[~L.nc].assign(angle_bin=(np.round(L[~L.nc].peak_angle_deg / 10) * 10))
            rows.append(C.groupby(["pos", "angle_bin"]).size().rename("contacted").reset_index()
                        .assign(label=lab, animal=animal, epoch="pre_template"))
    D = pd.concat(rows, ignore_index=True)
    cnt = D[D.epoch != "pre_template"]
    tpl = D[D.epoch == "pre_template"]
    cnt.to_csv(out / "counts.csv", index=False)
    tpl.to_csv(out / "pre_contacted_by_angle.csv", index=False)
    pd.set_option("display.width", 220)
    T = cnt.groupby(["animal", "epoch", "pos"])[["licks", "no_contact", "ipsi_dev", "contra_dev", "on_target_nc"]].sum()
    for pos in ("far_contra", "far_mid", "close_contra"):
        print(f"\n== {pos}: licks / no-contact / no-contact deviated IPSI > {a.dev:.0f} deg / CONTRA / on-target")
        X = T.xs(pos, level="pos").unstack("epoch")
        print(X.reindex(columns=["pre", "acute", "subacute", "chronic"], level=1).fillna(0).astype(int).to_string())
    print("\n== pre-stroke CONTACTED licks by executed angle (10-deg bins), summed over positions -- template pool")
    P = tpl.groupby(["animal", "angle_bin"]).contacted.sum().unstack("angle_bin").fillna(0).astype(int)
    print(P.to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
