"""NON-LINEAR per-lick prediction table in the iti_lick_order format (Priya 10-28: "plot the nonlinear version of the
decoded vs actual offset for ITI and cued licks (12.2 and 13.2)" + the 11.15 checks).

Takes the KEY lick-order table (lick_order_preds_areas_splitr8.csv.gz: set, animal, label, epoch, pos, order, act, dec) and
replaces `dec` with the RBF kernel-ridge prediction of the same lick (`lick_nonlinear_decoder`,
nonlinear_decoder_preds.csv.gz), matched per (set, label) by an ORDERED subsequence match on the exact landing
(`iti_licks.subseq_match`; the two tables index licks differently). Check printed: the non-linear table's own LINEAR
column must reproduce the KEY `dec` (it is the same decoder refit on the same folds).
Output <OUT>/lick_order_preds_nonlinear.csv.gz = iti_lick_order.preds_path() under WFIELD_LICK_DECODER=nonlinear, so
lick_matched_readout / iti_lick_order / lick_same_target_place run unchanged on it (outputs tagged "_nonlinear").
Usage: python scripts/lick_nonlinear_preds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402


def main() -> int:
    N = pd.read_csv(il.OUT / "nonlinear_decoder_preds.csv.gz")
    L = pd.read_csv(il.OUT / "lick_order_preds_areas_splitr8.csv.gz")
    out = []
    for (st, lab), g in L.groupby(["set", "label"], sort=False):
        n = N[(N.set == st) & (N.label == lab)]
        m = il.subseq_match(n.act_lateral.round(6).to_numpy(), g.act.round(6).to_numpy())
        ok = m >= 0
        h = g.copy()
        h["lin_check"] = np.where(ok, n.linear_lateral.to_numpy()[np.maximum(m, 0)], np.nan)
        h["dec"] = np.where(ok, n.nonlinear_lateral.to_numpy()[np.maximum(m, 0)], np.nan)
        out.append(h)
    M = pd.concat(out, ignore_index=True)
    d = np.abs(M.lin_check - L.dec.to_numpy())
    print(f"matched {M.dec.notna().mean():.4f}; non-linear table's linear column vs KEY dec: max |d| {np.nanmax(d):.4f} mm, "
          f"r {np.corrcoef(M.lin_check[M.lin_check.notna()], L.dec[M.lin_check.notna()])[0, 1]:.5f}")
    if np.nanmax(d) > 0.1:
        raise SystemExit("linear columns disagree -- the two tables are not the same licks / decoder")
    M.drop(columns="lin_check").to_csv(il.OUT / "lick_order_preds_nonlinear.csv.gz", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
