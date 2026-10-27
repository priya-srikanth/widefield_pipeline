"""RMSE of the frozen where-decoder split into BIAS and SCATTER (handoff 00f optional): per session, error e = decoded -
actual; RMSE^2 = mean(e)^2 + SD(e)^2. Does the post-stroke error grow because predictions are shifted (bias, the
offsets of 3.21 / 3.22) or because they scatter more around the licks (SD, what r loses)? Per lick set x axis
(lateral, sagittal), decoder on z-scored features (`zscore_check/where_preds_z.csv`; --unscaled for where_preds.csv).
Change vs pre per epoch: paired nested bootstrap. Output <E>/iti_licks/where_rmse_split[_unscaled][_sessions].csv.
Usage: python scripts/iti_where_rmse_split.py [--unscaled]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP = ["acute", "subacute", "chronic"]


def main(argv=None) -> int:
    unscaled = "--unscaled" in (argv if argv is not None else sys.argv[1:])
    p = pd.read_csv(il.OUT / "where_preds.csv" if unscaled else il.OUT / "zscore_check" / "where_preds_z.csv")
    rows = []
    for (st, a, ep, lab), g in p.groupby(["set", "animal", "epoch", "label"]):
        for ax_ in ("lat", "sag"):
            e = (g[f"dec_{ax_}"] - g[f"act_{ax_}"]).dropna().to_numpy(float)
            if len(e) >= 20:
                rows.append(dict(set=st, animal=a, epoch=ep, label=lab, axis=ax_, rmse=float(np.sqrt(np.mean(e ** 2))),
                                 bias=float(abs(e.mean())), sd=float(e.std())))
    S = pd.DataFrame(rows)
    S.to_csv(il.OUT / f"where_rmse_split{'_unscaled' if unscaled else ''}_sessions.csv", index=False)
    pd.set_option("display.width", 200)
    print("== per animal (rule 8), chronic minus pre: RMSE, |bias|, SD (mm)")
    m = S.groupby(["set", "axis", "animal", "epoch"])[["rmse", "bias", "sd"]].mean()
    print((m.xs("chronic", level="epoch") - m.xs("pre", level="epoch")).round(3).to_string())
    tests = []
    for (st, ax_), g in S.groupby(["set", "axis"]):
        for val in ("rmse", "bias", "sd"):
            prd = {a: h[val].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: h[val].to_numpy() for a, h in g[g.epoch == ep].groupby("animal")}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                tests.append(dict(set=st, axis=ax_, measure=val, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi))
    T = pd.DataFrame(tests)
    T.to_csv(il.OUT / f"where_rmse_split{'_unscaled' if unscaled else ''}.csv", index=False)
    print("\n== change vs pre (paired nested bootstrap)\n" + T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
