"""SSp-n (and every Allen area's) share of the frozen ITI / cued lateral-decoder offset in the AREA, SVD and LocaNMF bases
(Priya 11-11 follow-up, go-ahead 11-12). Read-only: sums the saved pixel contribution maps c(p) = w(p) dx(p)
(lick_offset_contrib_map maps_<tag>.npz, glue-out run 11-11) over each Allen area's pixels (the stat-mask area masks of
lick_offset_contrib_map._grid, the same masks the area decoder uses). share = area sum / total sum over ALL pixels (= the
offset beta . dh). Region contribution = beta_R dh_R + beta_L dh_L in the area basis (checked against terms_areas.csv).
Run from the wf_decisions1111 checkout with the rebuild profile applied:  python <this> <out_dir>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import lick_offset_contrib_map as L  # noqa: E402

OUT = Path(sys.argv[1])
BASES = {"areas": "maps_areas.npz", "svd": "maps_svdF.npz", "locanmf": "maps_locanmfFM.npz"}
rows = []
for a in L.AN:
    flat, pstat, M = L._grid(a)
    full = {}
    for n, m in M.items():
        img = np.zeros(L.H * L.W, bool)
        img[flat[m] if m.size == flat.size else flat[pstat][m]] = True
        full[n] = img
    stat_img = np.zeros(L.H * L.W, bool)
    stat_img[flat[pstat]] = True
    for basis, f in BASES.items():
        z = np.load(L.OUT / f)
        for st in L.SETS:
            for ep in L.EP:
                k = f"{st}|{a}|{ep}"
                if k not in z.files:
                    continue
                c = np.nan_to_num(z[k].astype(float).ravel())
                tot = c.sum()
                base = dict(basis=basis, set=st, animal=a, epoch=ep, total=tot,
                            outside_stat=c[~stat_img].sum())
                for n, img in full.items():
                    rows.append(dict(base, area=n, contrib=c[img].sum(), share=c[img].sum() / tot if tot else np.nan))
t = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
t.to_csv(OUT / "region_shares_long.csv", index=False)
# SSp-n: right, left, both, per basis x set x epoch x animal
s = t[t.area.str.startswith("SSp-n")].copy()
s["side"] = s.area.str.split("_").str[-1]
p = s.pivot_table(index=["set", "epoch", "animal", "basis"], columns="side", values="contrib", aggfunc="sum")
p["both"] = p.sum(1)
tot = t.drop_duplicates(["basis", "set", "animal", "epoch"]).set_index(["set", "epoch", "animal", "basis"])["total"]
p["total"] = tot
p["share_both"] = p.both / p.total
p.reset_index().to_csv(OUT / "sspn_by_basis.csv", index=False)
pd.set_option("display.width", 200)
print(p.round(3).to_string())
# top-5 areas by |share| per basis x set x epoch, pooled = mean over animals of contrib / mean of total
g = t.groupby(["basis", "set", "epoch", "area"]).agg(contrib=("contrib", "mean"), total=("total", "mean")).reset_index()
g["pooled_share"] = g.contrib / g.total
top = g.assign(a=g.pooled_share.abs()).sort_values("a", ascending=False).groupby(["basis", "set", "epoch"]).head(5)
top.drop(columns="a").to_csv(OUT / "top_areas_by_basis.csv", index=False)
print(top.drop(columns="a").sort_values(["set", "epoch", "basis", "pooled_share"]).round(3).to_string(index=False))
