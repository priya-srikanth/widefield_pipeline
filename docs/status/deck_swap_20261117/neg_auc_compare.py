"""Compare the alternative ratio metrics with the production 0..1 s log ratio (neg_auc_diag step 4).
usage: python neg_auc_compare.py <out_dir with alt_tests.csv, alt_models.csv>"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 400)
d = Path(sys.argv[1])
T = pd.read_csv(d / "alt_tests.csv")
M = pd.read_csv(d / "alt_models.csv")
T["setm"] = np.where(T["mode"] == "dock", "iti(dock)", T.set)
M["setm"] = np.where(M["mode"] == "dock", "iti(dock)", M.set)
REF = "logr_0-1"
k = ["setm", "pair", "spout", "epoch"]
ref = T[T.metric == REF].set_index(k)
rows = []
for met, g in T.groupby("metric"):
    g = g.set_index(k).reindex(ref.index)
    for st in ["cued", "cue", "iti", "iti(dock)"]:
        m = ref.index.get_level_values(0) == st
        r, a = ref[m], g[m]
        rs, as_ = r.sig.fillna(False).astype(bool), a.sig.fillna(False).astype(bool)
        sgn = np.sign(r.delta) == np.sign(a.delta)
        rows.append({"metric": met, "set": st, "cells": int(m.sum()), "stars_ref": int(rs.sum()), "stars": int(as_.sum()),
                     "both": int((rs & as_).sum()), "lost": int((rs & ~as_).sum()), "gained": int((~rs & as_).sum()),
                     "sign_agree": round(float(sgn[a.delta.notna()].mean()), 3),
                     "star_flip_sign": int((rs & as_ & ~sgn).sum())})
S = pd.DataFrame(rows)
print("### boot_delta change-vs-pre cells (per spout + all + pooled, x 3 epochs, x 5 pairs) vs PRODUCTION logr_0-1")
print(S.pivot_table(index="metric", columns="set", values=["stars", "lost", "gained"], aggfunc="first").to_string())
print("\nsign agreement / starred-with-flipped-sign:")
print(S.pivot_table(index="metric", columns="set", values=["sign_agree", "star_flip_sign"], aggfunc="first").to_string())
S.to_csv(d / "compare_boot.csv", index=False)

print("\n### mixed-model LR tests: p (x = not converged / LR not finite)")
M["cell"] = [("x" if not c else "") + (f"{p:.3g}" if np.isfinite(p) else "nan") for p, c in zip(M.p, M.converged)]
for test in ["pre spout", "spout x epoch", "epoch"]:
    t = M[M.test == test].pivot_table(index=["setm", "pair"], columns="metric", values="cell", aggfunc="first")
    print(f"\n-- {test}")
    print(t.reindex(columns=[c for c in ["logr_0-1", "logr_0-0.5", "bi_0-1", "bi_0-0.5", "diff_0-1", "logr_be_0-1",
                                        "logr_bl_0-1", "logr_0-1_n10", "logr_0-1_n15"] if c in t.columns]).to_string())
# summary of LR agreement at p < 0.05
ag = []
for test in ["pre spout", "spout x epoch", "epoch"]:
    base = M[(M.test == test) & (M.metric == REF)].set_index(["setm", "pair"]).p
    for met in M.metric.unique():
        x = M[(M.test == test) & (M.metric == met)].set_index(["setm", "pair"]).p.reindex(base.index)
        ag.append({"test": test, "metric": met, "sig_ref": int((base < 0.05).sum()), "sig": int((x < 0.05).sum()),
                   "agree": int(((base < 0.05) == (x < 0.05)).sum()), "of": len(base)})
print("\n### LR p<0.05 agreement with production")
print(pd.DataFrame(ag).pivot_table(index="metric", columns="test", values=["sig", "agree"], aggfunc="first").to_string())
# focused: pooled / all rows table, the headline cells
f = T[T.spout.isin(["all", "pooled"]) & T.metric.isin(["logr_0-1", "logr_0-0.5", "bi_0-1", "bi_0-0.5", "diff_0-1"])].copy()
f["c"] = [f"{dl:+.2f}{'*' if s else ''}" for dl, s in zip(f.delta, f.sig)]
print("\n### spout-mean ('all') and pooled change vs pre, a / s / c")
print(f.pivot_table(index=["setm", "pair", "spout"], columns=["metric", "epoch"], values="c", aggfunc="first")
      .reindex(columns=pd.MultiIndex.from_product([["logr_0-1", "logr_0-0.5", "bi_0-1", "bi_0-0.5", "diff_0-1"], ["acute", "subacute", "chronic"]]))
      .to_string())
