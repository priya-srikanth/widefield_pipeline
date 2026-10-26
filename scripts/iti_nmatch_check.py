"""Are ITI lick numbers a confound for the frozen-decoder result (page 3.8 / 3.9)? Counts per session by epoch; r vs n
within epoch; frozen decoder (trained on all pre licks, pre LOSO) scored on test sets SUBSAMPLED to the same n in every
session (200 draws, r averaged). Uses the complete v3 extraction (lags -0.5..0.5)."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/SabatiniLab/Github/wf_attempt/scripts")
sys.path.insert(0, r"C:/Users/SabatiniLab/Github/wf_attempt")
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
V3 = Path(r"M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/sessions_v3")
LAGS = np.round(np.arange(-0.5, 0.5001, 0.05), 3)
W = (LAGS >= 0.0) & (LAGS <= 0.2)
D = {}
for f in sorted(glob.glob(str(V3 / "*.npz"))):
    z = np.load(f)
    D[Path(f).stem] = {k: z[k] for k in ("cols", "epoch", "animal", "iti_X", "iti_side", "tr_X", "tr_side")}
rows = [{"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "n_iti": len(d["iti_side"]),
         "n_trial": len(d["tr_side"])} for x, d in D.items()]
N = pd.DataFrame(rows)
print("== ITI licks per session (median [min-max]) by animal x epoch")
print(N.groupby(["animal", "epoch"]).n_iti.agg(lambda v: f"{int(np.median(v))} [{v.min()}-{v.max()}]").unstack()
      .reindex(columns=EP).to_string())
rng = np.random.default_rng(0)
out = []
for st, kx, ky in (("iti", "iti_X", "iti_side"), ("trial", "tr_X", "tr_side")):
    for a in AN:
        S = {x: d for x, d in D.items() if str(d["animal"]) == a and len(d[ky]) >= 20}
        pre = [x for x, d in S.items() if str(d["epoch"]) == "pre"]
        if len(pre) < 3:
            continue
        feat = [i for i, c in enumerate(next(iter(S.values()))["cols"]) if c != "cortex" and not str(c).startswith(("ALM", "tjM1"))]
        X = {x: S[x][kx].astype(float)[:, W][:, :, feat].mean(1) for x in S}
        fits = {}
        for held in [*pre, None]:
            trn = [q for q in pre if q != held]
            Xt = np.vstack([X[q] for q in trn])
            sc = StandardScaler().fit(Xt)
            fits[held] = (sc, RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(Xt), np.concatenate([S[q][ky] for q in trn])))
        for x in S:
            sc, m = fits[x if x in pre else None]
            pr, y = m.predict(sc.transform(X[x])), S[x][ky]
            r_full = float(np.corrcoef(pr, y)[0, 1])
            n0 = 40
            rs = [np.corrcoef(pr[i], y[i])[0, 1] for i in (rng.choice(len(y), n0, replace=False) for _ in range(200))]
            out.append({"set": st, "label": x, "animal": a, "epoch": str(S[x]["epoch"]), "n": len(y), "r_full": r_full,
                        "r_n40": float(np.nanmean(rs))})
R = pd.DataFrame(out)
R.to_csv(V3.parent / "nmatch_check.csv", index=False)
for st in ("iti", "trial"):
    t = R[R.set == st]
    print(f"\n== {st}: within-epoch Spearman(n, r_full), pooled sessions")
    for ep in EP:
        u = t[t.epoch == ep]
        print(f"  {ep}: rho {u[['n', 'r_full']].corr('spearman').iloc[0, 1]:+.2f} (n sessions {len(u)})")
    for val in ("r_full", "r_n40"):
        ref = {a: t[(t.animal == a) & (t.epoch == "pre")][val].to_numpy() for a in AN}
        cells = []
        for ep in EP[1:]:
            post = {a: t[(t.animal == a) & (t.epoch == ep)][val].to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            cells.append(f"{ep} {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}]")
        print(f"  {val}: " + "; ".join(cells))
