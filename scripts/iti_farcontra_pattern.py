"""Is the chronic cued FAR-CONTRA cortical pattern a WEAKER COPY of its pre-stroke pattern, or has it DRIFTED toward
another spout's pattern? (Priya 10-26: "how can we make sense of the chronic far-contra ipsi reading for cued trials,
biologically?"; handoff 00d proposed test.)

Features = Allen-area activity at lags 0..0.2 s from the lick peak (the where-decoder's features), every area divided
by its within-session frame SD (z-scored, `iti_zscore_check.py` session_sd.csv). Cued licks by TARGET spout. Per animal
x epoch: spout pattern P[s] = mean over the epoch's sessions of (session mean of the spout's licks) minus the mean over
the six spouts (removes any epoch-wide change, so only the SPATIAL arrangement across spouts remains).
For far contra (and far ipsi as the mirror control):
  gain    = <P_ep, P_pre> / |P_pre|^2     (1 = same strength along the pre pattern; < 1 = weaker copy)
  cos     = cosine(P_ep, P_pre)          (1 = same pattern up to scale)
  cos_to  = cosine(P_ep, P_pre[other spout]) for every spout (drift toward a neighbour shows as a rise here)
Noise floor: pre sessions split in two halves (alternating by date), the same numbers between halves.
"usual" rows repeat it on licks landing within 0.25 mm (lateral x sagittal) of that spout's pre landing mean.
Output <E>/iti_licks/farcontra_pattern.csv; usage: python scripts/iti_farcontra_pattern.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
USUAL_MM = 0.25


def licks() -> dict:
    """{animal: list of (label, epoch, X (n, areas) z-scored, lat, sag, pos)} for cued licks."""
    from iti_decoder_contrib import feature_names

    from wfield_local import orofacial_cohort as oc
    S = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load("areas")
    out: dict = {}
    for x, d in sorted(D.items()):
        X, C = d["trial"]
        P = il._load(x)["tr_pos"].astype(str)
        P = np.array([p if p in POS or "_" not in p else oc.pos_key(p, "L") for p in P])
        ok = np.isfinite(C[:, [0, 3]]).all(1)
        sd = S.loc[x, names].to_numpy(float)
        out.setdefault(d["animal"], []).append((x, d["epoch"], X[ok] / sd[None, :], C[ok, 0], C[ok, 3], P[ok]))
    return out


def patterns(rows: list, usual: dict | None = None) -> dict | None:
    """{spout: centred pattern} from a list of session tuples (session means first, then mean of sessions)."""
    per = {p: [] for p in POS}
    for _x, _ep, X, lat, sag, P in rows:
        for p in POS:
            m = P == p
            if usual is not None and p in usual:
                m &= np.hypot(lat - usual[p][0], sag - usual[p][1]) <= USUAL_MM
            if m.sum() >= 5:
                per[p].append(X[m].mean(0))
    if any(len(v) == 0 for v in per.values()):
        return None
    M = {p: np.mean(v, 0) for p, v in per.items()}
    g = np.mean(list(M.values()), 0)
    return {p: v - g for p, v in M.items()}


def _cos(u, v):
    return float(u @ v / np.sqrt((u @ u) * (v @ v)))


def main() -> int:
    L = licks()
    rows = []
    for a in AN:
        sess = L.get(a, [])
        by = {ep: [s for s in sess if s[1] == ep] for ep in EP}
        pre_all = by["pre"]
        cen = {}
        for p in POS:
            lat = np.concatenate([s[3][s[5] == p] for s in pre_all])
            sag = np.concatenate([s[4][s[5] == p] for s in pre_all])
            cen[p] = (lat.mean(), sag.mean())
        for subset, us in (("all", None), ("usual", cen)):
            Pp = patterns(pre_all, us)
            if Pp is None:
                continue
            halves = (patterns(pre_all[0::2], us), patterns(pre_all[1::2], us))
            for tgt in ("far_contra", "far_ipsi"):
                ref = Pp[tgt]
                if all(h is not None for h in halves):
                    h1, h2 = halves[0][tgt], halves[1][tgt]
                    rows.append(dict(animal=a, subset=subset, spout=tgt, epoch="pre (half vs half)",
                                     gain=float(h1 @ h2 / (h2 @ h2)), cos=_cos(h1, h2),
                                     **{f"cos_to_{q}": _cos(h1, halves[1][q]) for q in POS}))
                for ep in EP[1:]:
                    Pe = patterns(by[ep], us) if by[ep] else None
                    if Pe is None:
                        continue
                    v = Pe[tgt]
                    rows.append(dict(animal=a, subset=subset, spout=tgt, epoch=ep, gain=float(v @ ref / (ref @ ref)),
                                     cos=_cos(v, ref), **{f"cos_to_{q}": _cos(v, Pp[q]) for q in POS}))
    T = pd.DataFrame(rows)
    T.to_csv(il.OUT / "farcontra_pattern.csv", index=False)
    pd.set_option("display.width", 250)
    for sub in ("all", "usual"):
        print(f"\n== {sub} licks: per animal (rule 8). gain along the pre pattern, cos to it, cos to each pre spout pattern")
        print(T[T.subset == sub].set_index(["spout", "animal", "epoch"]).drop(columns="subset").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
