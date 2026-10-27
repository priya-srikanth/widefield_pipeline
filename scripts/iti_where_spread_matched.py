"""Is the frozen decoder's r loss a narrower SPREAD of lick positions? (handoff 00f to-do; r is a correlation, so a
session whose licks span a narrower range scores lower with unchanged cortex.)

Per session, licks are resampled so the distribution of ACTUAL position on the scored axis matches the animal's PRE
distribution for that lick set (importance weights = pre density / session density in 0.2 mm bins; bins the session
does not reach cannot be filled, so the match is partial when the post range is narrower than pre -- `sd_ratio`
reports how close it got). Draws without replacement of half the session's licks (N_DRAW times); r = mean over draws.
Pre sessions get the same treatment (their own distribution is close to the pooled pre, so little changes).
Predictions = the frozen decoder's per-lick predictions (`where_preds.csv`, pre held out by session; --z = within-
session z-scored features). Change vs pre per epoch: paired nested bootstrap (animals > sessions).
Output <E>/iti_licks/where_spread_matched[_z]_{sessions,tests}.csv. Usage: python scripts/iti_where_spread_matched.py [--z]
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
BIN = 0.2
N_DRAW = 30


def matched_r(act: np.ndarray, dec: np.ndarray, pre_act: np.ndarray, rng) -> tuple[float, float, float]:
    """(raw r, spread-matched r, matched SD / pre SD)."""
    lo, hi = np.nanmin(np.r_[act, pre_act]), np.nanmax(np.r_[act, pre_act])
    edges = np.arange(lo, hi + BIN, BIN)
    hp = np.histogram(pre_act, edges)[0] / len(pre_act)
    hs = np.histogram(act, edges)[0] / len(act)
    b = np.clip(np.digitize(act, edges) - 1, 0, len(hp) - 1)
    w = np.where(hs[b] > 0, hp[b] / np.where(hs[b] > 0, hs[b], 1), 0)
    raw = float(np.corrcoef(act, dec)[0, 1])
    if w.sum() <= 0:
        return raw, np.nan, np.nan
    p = w / w.sum()
    n = max(20, min(len(act) // 2, int((p > 0).sum() * 0.9)))
    rs, sds = [], []
    for _ in range(N_DRAW):
        idx = rng.choice(len(act), size=n, replace=False, p=p) if (p > 0).sum() >= n else \
            rng.choice(len(act), size=n, replace=True, p=p)
        rs.append(np.corrcoef(act[idx], dec[idx])[0, 1])
        sds.append(act[idx].std())
    return raw, float(np.mean(rs)), float(np.mean(sds) / pre_act.std())


def main(argv=None) -> int:
    z = "--z" in (argv if argv is not None else sys.argv[1:])
    fp = il.OUT / "zscore_check" / "where_preds_z.csv" if z else il.OUT / "where_preds.csv"
    p = pd.read_csv(fp)
    rng = np.random.default_rng(0)
    rows = []
    for (st, a), g in p.groupby(["set", "animal"]):
        for ax_ in ("lat", "sag"):
            gg = g.dropna(subset=[f"act_{ax_}", f"dec_{ax_}"])
            pre_act = gg[gg.epoch == "pre"][f"act_{ax_}"].to_numpy(float)
            for (lab, ep), h in gg.groupby(["label", "epoch"]):
                if len(h) < 40:
                    continue
                raw, mr, sdr = matched_r(h[f"act_{ax_}"].to_numpy(float), h[f"dec_{ax_}"].to_numpy(float), pre_act, rng)
                rows.append(dict(set=st, animal=a, axis=ax_, label=lab, epoch=ep, n=len(h), r_raw=raw, r_matched=mr,
                                 sd_ratio_raw=float(h[f"act_{ax_}"].std() / pre_act.std()), sd_ratio_matched=sdr))
    S = pd.DataFrame(rows)
    sfx = "_z" if z else ""
    S.to_csv(il.OUT / f"where_spread_matched{sfx}_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): session SD / pre SD (raw -> matched) and r (raw -> matched), chronic")
    c = S[S.epoch == "chronic"].groupby(["set", "axis", "animal"])[["sd_ratio_raw", "sd_ratio_matched", "r_raw",
                                                                    "r_matched"]].mean()
    print(c.round(2).to_string())
    tests = []
    for (st, ax_), g in S.groupby(["set", "axis"]):
        for col in ("r_raw", "r_matched"):
            prd = {a: h[col].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: h[col].to_numpy() for a, h in g[g.epoch == ep].groupby("animal")}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = g[g.epoch == ep].groupby("animal")[col].mean() - g[g.epoch == "pre"].groupby("animal")[col].mean()
                tests.append(dict(set=st, axis=ax_, measure=col, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                  n_down=int((am < 0).sum()), n=int(am.notna().sum())))
    T = pd.DataFrame(tests)
    T.to_csv(il.OUT / f"where_spread_matched{sfx}_tests.csv", index=False)
    print("\n== change in r vs pre (paired nested bootstrap); n_down = animals below pre")
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
