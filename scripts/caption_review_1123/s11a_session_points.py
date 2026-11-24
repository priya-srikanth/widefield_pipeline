"""S11A caption review (Priya 11-23, item 7): all-lick 3-D reach by position with SESSION points instead of animal
means of session changes; same markers, CIs and stars (the single six-position scalar family, metric reach_all).

Copy of the manuscript renderer `c2_matched.reach3d` (manuscript_code/figures; called by c2_scalar_aligned_render.py
for C2_reach_all) with ONE change: the faded points come from the per-session changes that the family tested
(`c2_scalar_family_aligned.reach_all`: <pooled>/pose3d_sessions.csv, reach_mm|<spout>, August pre only, change vs the
animal's own pre mean at that spout) instead of the pooled table's per-animal epoch means. The animal-mean mode
(``--points animal``) reproduces the live panel for the check.

    python s11a_session_points.py <manuscript_code/figures> <pooled_dir> <family.csv> <out_stem> [--points session|animal]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LR_POS = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
          "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}


def session_changes(pooled: Path) -> pd.DataFrame:
    from wfield_local import config
    S = pd.read_csv(pooled / "pose3d_sessions.csv", dtype={"date": str})
    rows = [S[["animal", "date", "epoch", f"reach_mm|{lr}"]].rename(columns={f"reach_mm|{lr}": "v"}).assign(pos=p)
            for lr, p in LR_POS.items() if f"reach_mm|{lr}" in S]
    L = pd.concat(rows, ignore_index=True).dropna(subset=["v"])
    aug = (L.epoch == "pre") & (~L.date.astype(str).map(config.early_pre))
    L = L[aug | L.epoch.isin(["acute", "subacute", "chronic"])].copy()
    b = L[L.epoch == "pre"].groupby(["animal", "pos"]).v.mean()
    L["chg"] = L.v - [b.get((x, p), np.nan) for x, p in zip(L.animal, L.pos)]
    return L.dropna(subset=["chg"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("figdir", type=Path)
    ap.add_argument("pooled", type=Path)
    ap.add_argument("family", type=Path)
    ap.add_argument("out_stem")
    ap.add_argument("--points", choices=("session", "animal"), default="session")
    a = ap.parse_args()
    sys.path.insert(0, str(a.figdir))
    import matplotlib.pyplot as plt
    import vls_style as S
    from c2_panels import grouped, star, EPS, C, aligned_tg
    from c2_matched import POS, POS3, XT
    S.apply()
    pc = pd.read_csv(a.pooled / "pose3d_change_pooled_contra.csv")
    r = pc[(pc.measure == "reach_mm") & (pc.group != "all")].copy()
    r["pos"] = r.group.map(POS3)
    fam = pd.read_csv(a.family)
    al = fam[fam.metric == "reach_all"]
    L = session_changes(a.pooled)
    an = ["PS92", "PS93", "PS94", "PS95"]
    # check: animal means of the session changes = the pooled table's per-animal columns (as the family asserts)
    am = L[L.epoch != "pre"].groupby(["pos", "epoch", "animal"]).chg.mean().unstack()
    chk = r.set_index(["pos", "epoch"])[an]
    dev = np.nanmax(np.abs(chk.to_numpy(float) - am.reindex(chk.index)[an].to_numpy(float)))
    print(f"animal means of session changes vs pooled table: max dev {dev:.2e}")
    assert dev < 1e-6
    n = L[L.epoch != "pre"].groupby(["pos", "epoch"]).size()
    print("post sessions per cell:\n", n.unstack().to_string())

    fig, ax = plt.subplots(figsize=(4.0, 1.75))
    S.zero_line(ax)

    def tg0(p, e):
        x = r[(r.pos == p) & (r.epoch == e)]
        if x.empty:
            return None
        x = x.iloc[0]
        return x.est, x.lo, x.hi, int(x.lo > 0 or x.hi < 0)
    tg = aligned_tg(al, tg0)
    if a.points == "animal":
        get = lambda p, e: r[(r.pos == p) & (r.epoch == e)][an].values.ravel() if e != "pre" else np.array([])  # noqa: E731
    else:
        get = lambda p, e: L[(L.pos == p) & (L.epoch == e)].chg.values if e != "pre" else np.array([])  # noqa: E731
    w = grouped(ax, POS, get, tg, seed=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(XT)
    ax.set_xlim(-.5, 5.5)
    ax.set_ylabel("3-D reach\n(Δ mm)")
    if a.points == "session":                       # session points spread wider than animal means: keep 1-99 %
        v = L[L.epoch != "pre"].chg.values
        lo_, hi_ = np.nanpercentile(v, 0.5), np.nanpercentile(v, 99.5)
        pad = 0.08 * (hi_ - lo_)
        ax.set_ylim(lo_ - pad, hi_ + pad)
        print(f"y clipped to {lo_ - pad:.2f}..{hi_ + pad:.2f}; points outside: "
              f"{int(((v < lo_ - pad) | (v > hi_ + pad)).sum())} of {len(v)}")
    S.headroom(ax, 0.55)
    n_star = 0
    for i, p in enumerate(POS):
        for j, e in enumerate(EPS):
            x = tg(p, e)
            if x and x[3]:
                star(ax, i + (j - 1.5) * w, j, C[e], level=x[3])
                n_star += 1
    print("stars drawn:", n_star)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.25, top=0.92)
    S.save(fig, a.out_stem)


if __name__ == "__main__":
    main()
