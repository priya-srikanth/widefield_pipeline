"""Decoded lick position BEYOND the decoder's pre-stroke pull toward the middle (Priya 10-26: "in 3.19 it's clearly not
just pulling toward the center").

The frozen ridge decoder shrinks predictions toward its training mean, separately on each axis. Per animal, set (ITI /
in-trial) and axis (lateral, sagittal), the pull is measured on that animal's PRE licks (held-out predictions, so the
same frame as every post score): decoded = a + b * actual (least squares). For every lick, expected = a + b * actual;
beyond = decoded - expected. Pre sessions average ~0 by construction; a post epoch's mean is where cortex is read
relative to where the unchanged pull alone would put that lick (uses ALL licks; assumes the pull extrapolates linearly
to post-stroke landing positions). + lateral = contralesional, - sagittal = decoded closer to the mouth.
Output <E>/iti_licks/where_beyond_pull_{by_animal,tests}.csv; usage: python scripts/iti_where_beyond_pull.py
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


def main() -> int:
    p = pd.read_csv(il.OUT / "where_preds.csv").dropna(subset=["act_lat", "act_sag", "dec_lat", "dec_sag"])
    rows, pull = [], []
    for (st, a), g in p.groupby(["set", "animal"]):
        g = g.copy()
        for ax_ in ("lat", "sag"):
            pre = g[g.epoch == "pre"]
            b, c = np.polyfit(pre[f"act_{ax_}"], pre[f"dec_{ax_}"], 1)
            g[f"bey_{ax_}"] = g[f"dec_{ax_}"] - (c + b * g[f"act_{ax_}"])
            pull.append(dict(set=st, animal=a, axis=ax_, slope=b))
        rows.append(g)
    p = pd.concat(rows, ignore_index=True)
    pd.set_option("display.width", 200)
    print("== pre-stroke pull: slope of decoded on actual (1 = no pull)")
    print(pd.DataFrame(pull).pivot_table(index=["set", "animal"], columns="axis", values="slope").round(2).to_string())
    ses = p.groupby(["set", "animal", "epoch", "label"])[["bey_lat", "bey_sag"]].mean().reset_index()
    pa = ses.groupby(["set", "animal", "epoch"])[["bey_lat", "bey_sag"]].mean().unstack("epoch")
    print("\n== per animal (rule 8): mean decoded position beyond the pre pull, mm")
    print(pa.reindex(columns=["pre", *EP], level=1).round(2).to_string())
    pa.to_csv(il.OUT / "where_beyond_pull_by_animal.csv")
    tests = []
    for st in ("trial", "iti"):
        x = ses[ses.set == st]
        for ax_ in ("lat", "sag"):
            prd = {a: g[f"bey_{ax_}"].to_numpy() for a, g in x[x.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: g[f"bey_{ax_}"].to_numpy() for a, g in x[x.epoch == ep].groupby("animal")}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = x[x.epoch == ep].groupby("animal")[f"bey_{ax_}"].mean()
                tests.append(dict(set=st, axis=ax_, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                  n_pos=int((am > 0).sum()), n=len(am)))
    T = pd.DataFrame(tests)
    print("\n== change vs pre (paired nested bootstrap); n_pos = animals with + mean")
    print(T.round(3).to_string(index=False))
    T.to_csv(il.OUT / "where_beyond_pull_tests.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
