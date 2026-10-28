"""LIKE-FOR-LIKE (same-place) offset of the where-decoder's per-lick predictions, lateral and sagittal. CONVERTED 10-27 late
(Priya: "yes replace beyond pull"): this script used to report the offset BEYOND the pre-stroke pull (decoded - (a + b x
actual), a, b fit on the animal's pre licks), whose sign depended on the assumed pull slope (DECISIONS 2026-10-27 late).
The file name is kept so run scripts and links still find it; the outputs are renamed (where_lfl_*).

Per animal x set (ITI / in-trial) x axis (lateral, sagittal): lfl = decoded - the animal's PRE licks of the set that
landed at the same place on that axis (`likeforlike.offset`: Gaussian-weighted pre reference by default, local-linear
with WFIELD_LFL_REF=loclin; pre licks leave-one-pre-session-out; width by held-out pre error). Pre sessions sit ~0 out of
sample. + lateral = read more contralesional, - sagittal = read closer to the mouth than same-place pre licks.
Input: the where-decoder's per-lick predictions (`where_preds<PRED_TAG>.csv`, iti_where_scatter.py; `--z` the z-scored
ones of iti_zscore_check.py). Output <E>/iti_licks/[zscore_check/]where_lfl[_z]_{by_animal,tests}.csv.
Usage: python scripts/iti_where_beyond_pull.py [--z]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import likeforlike as lfl  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP = ["acute", "subacute", "chronic"]


def main(z: bool | None = None) -> int:
    il.split_only("iti_where_beyond_pull")
    z = ("--z" in sys.argv) if z is None else z
    src = il.pred_file("zscore_check/where_preds_z.csv" if z else "where_preds.csv")
    od, sfx = (il.OUT / "zscore_check", "_z") if z else (il.OUT, "")
    p = pd.read_csv(src).dropna(subset=["act_lat", "act_sag", "dec_lat", "dec_sag"]).reset_index(drop=True)
    for ax_ in ("lat", "sag"):
        p[f"lfl_{ax_}"] = lfl.offset(p, f"act_{ax_}", f"dec_{ax_}", lfl.SIGMA_MM, f"{ax_} (mm)")[0]
    pd.set_option("display.width", 200)
    ses = p.groupby(["set", "animal", "epoch", "label"])[["lfl_lat", "lfl_sag"]].mean().reset_index()
    pa = ses.groupby(["set", "animal", "epoch"])[["lfl_lat", "lfl_sag"]].mean().unstack("epoch")
    print("\n== per animal (rule 8): like-for-like offset (decoded vs same-place pre licks), mm")
    print(pa.reindex(columns=["pre", *EP], level=1).round(2).to_string())
    pa.to_csv(od / f"where_lfl{sfx}_by_animal.csv")
    tests = []
    for st in ("trial", "iti"):
        x = ses[ses.set == st]
        for ax_ in ("lat", "sag"):
            prd = {a: g[f"lfl_{ax_}"].dropna().to_numpy() for a, g in x[x.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: g[f"lfl_{ax_}"].dropna().to_numpy() for a, g in x[x.epoch == ep].groupby("animal") if a in prd}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = x[x.epoch == ep].groupby("animal")[f"lfl_{ax_}"].mean()
                tests.append(dict(set=st, axis=ax_, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                  n_pos=int((am > 0).sum()), n=len(am), ref=lfl.REF))
    T = pd.DataFrame(tests)
    print("\n== change vs pre (paired nested bootstrap); n_pos = animals with + mean")
    print(T.round(3).to_string(index=False))
    T.to_csv(od / f"where_lfl{sfx}_tests.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
