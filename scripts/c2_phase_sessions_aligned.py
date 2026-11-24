"""Cohort-2 phase-curve 'sig' rows with the ALIGNED cluster rule (Priya 2026-11-19; DECISIONS.md "Cohort-2 statistics
aligned with cohort 1"): every cluster's permutation p (`orofacial_stats.cluster_permutation_p`), BH-FDR within each
MEASURE (one family per lick set x measure x alignment, across its six positions x three post epochs; cells with no
observed cluster carry no p and are not in the family), bars where q < 0.05.

Input = the manuscript's `analysis/c2_phase_curves_sessions.py` table `c2_phase_<lickset>_<measure>[_peak]_sessions.csv`
and the per-session curves it was built from. The test substrate is rebuilt here exactly as that script builds it
(curves file, kind, lick set, coverage >= 0.5, window, early-pre exclusion, contra sign, per-session change from the
animal's own August-pre curve, bins finite in every session, >= 3 bins, >= 2 epoch sessions, CLUSTER_PERM permutations,
seed 0). PARITY GATE: the uncorrected mask (p_raw < .05, which reproduces `cluster_permutation` exactly; pinned in
tests/test_c2_stats_alignment.py) must equal the existing file's 'sig' rows bin for bin, or nothing is written.

Output = `<same name>_aligned.csv` next to the input (never overwrites): every row of the input unchanged except 'sig'
rows, whose `v` is the aligned mask; added columns on 'sig' rows: `v_uncorrected` (the old mask), `cluster_id`,
`p_cluster`, `q_cluster` (NaN outside clusters). The renderer (kin_v37 c2_rows / c2_grid / c2_path) reads it unchanged.
Also writes `<same name>_aligned_clusters.csv` (one row per observed cluster).

Usage: python scripts/c2_phase_sessions_aligned.py <sessions.csv> <curves.csv(.gz)> [--kind peak|norm]
(11-22: ``--kind`` existed only in this line until the runbook used it and the parser rejected it; it now overrides the
kind read from the file name. The file name may carry a suffix after ``_sessions`` -- e.g. ``_sessions_NEW.csv`` from
``pose3d_phase_curves --out-suffix _NEW`` -- which the old fixed-length parse read as measure ``<m>_peak_sess``.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.orofacial_cohort import CLUSTER_PERM
from wfield_local.orofacial_stats import ALPHA, bh_fdr, cluster_permutation_p

EPOCHS = ("acute", "subacute", "chronic")
PHASE_WIN = (0.1, 0.9)                     # scripts/kinematics_pooled_figs.PHASE_WIN (norm kind)
SIGNED = ("mid", "mid0", "side", "side0")  # + = contralesional (c2_phase_curves_sessions.py)


def _parse_name(p: Path):
    """(lick set, measure, kind) from ``c2_phase_<lickset>_<measure>[_peak]_sessions[<suffix>].csv``."""
    import re
    m_ = re.fullmatch(r"c2_phase_(.+)_sessions(_[A-Za-z0-9]+)?\.csv", p.name)
    if not m_:
        raise ValueError(f"{p.name}: expected c2_phase_<lickset>_<measure>[_peak]_sessions[_suffix].csv")
    stem = m_.group(1)
    lset, rest = stem.split("_", 1)
    kind = "peak" if rest.endswith("_peak") else "norm"
    meas = rest[:-len("_peak")] if kind == "peak" else rest
    return lset, meas, kind


def run(sessions_csv: Path, curves: Path, n_perm: int = CLUSTER_PERM, kind: str | None = None, out_suffix: str = ""):
    lset, m, kind_ = _parse_name(sessions_csv)
    kind = kind or kind_
    out = sessions_csv.with_name(sessions_csv.stem + "_aligned" + out_suffix + ".csv")
    out_cl = sessions_csv.with_name(sessions_csv.stem + "_aligned_clusters" + out_suffix + ".csv")
    for o in (out, out_cl):
        if o.exists():
            sys.exit(f"refusing to overwrite {o}")
    win = PHASE_WIN if kind == "norm" else (-80, 80)
    C = pd.read_csv(curves)
    C = C[(C.kind == kind) & (C.lickset == lset) & (C.measure == m) & (C["cov"] >= 0.5) & C.x.between(*win)]
    C = C[(C.epoch != "pre") | (~C.date.astype(str).map(config.early_pre))]
    if C.empty:
        sys.exit(f"{sessions_csv.name}: no curves for {lset}/{m}/{kind} in {curves}")
    sg = -1.0 if m in SIGNED else 1.0
    cells, clusters, parity = [], [], {}
    for p, g in C.groupby("position"):
        pv = g.pivot_table(index=["animal", "session", "epoch"], columns="x", values="v") * sg
        an = pv.index.get_level_values(0).to_numpy()
        ep = pv.index.get_level_values(2).to_numpy()
        base = pv[ep == "pre"].groupby(level=0).mean()
        ch = pv.to_numpy(float) - base.reindex(an).to_numpy(float)
        xs = pv.columns.to_numpy(float)
        for e in EPOCHS:
            sel = ep == e
            if not sel.any() or sel.sum() < 2:
                continue
            keep = (ep == "pre") | sel
            sub = ch[keep]
            ok = np.isfinite(sub).all(0)
            cl = (cluster_permutation_p(sub[:, ok], sel[keep], an[keep], n_perm=n_perm)
                  if ok.sum() >= 3 else [])
            xok = xs[ok]
            # parity gate runs at the INPUT file's count (CLUSTER_PERM): at another n_perm the uncorrected mask can
            # legitimately differ (11-23: 5,000 for the paper)
            clp = cl if n_perm == CLUSTER_PERM or ok.sum() < 3 else                 cluster_permutation_p(sub[:, ok], sel[keep], an[keep], n_perm=CLUSTER_PERM)
            par = np.zeros(len(xs), bool)
            for c in clp:
                if c["p_raw"] < ALPHA:
                    par |= np.isin(xs, xok[c["start"]:c["stop"]])
            parity[(p, e)] = par
            for k, c in enumerate(cl):
                clusters.append({"lickset": lset, "measure": m, "kind": kind, "position": p, "epoch": e, "cluster": k,
                                 "x_start": xok[c["start"]], "x_end": xok[c["end_bin"]], "n_bins": c["stop"] - c["start"],
                                 "sign": c["sign"], "mass": c["mass"], "p_raw": c["p_raw"], "p": c["p"],
                                 "t_threshold": c["t_threshold"], "_xs": xok[c["start"]:c["stop"]]})
            cells.append((p, e, xs))
    q = bh_fdr([c["p"] for c in clusters]) if clusters else np.array([])
    for c, qq in zip(clusters, q):
        c["q"] = float(qq)
        c["sig_uncorrected"] = c["p_raw"] < ALPHA
        c["sig_aligned"] = bool(np.isfinite(qq) and qq < ALPHA)

    old = pd.read_csv(sessions_csv, float_precision="round_trip")   # unchanged rows written back bit for bit
    new = old.copy()
    for col in ("v_uncorrected", "cluster_id", "p_cluster", "q_cluster"):
        new[col] = np.nan
    sig_rows = old.row == "sig"
    n_cells_old = old[sig_rows].groupby(["position", "epoch"]).ngroups
    if n_cells_old != len(cells):
        sys.exit(f"{sessions_csv.name}: {len(cells)} tested cells here vs {n_cells_old} 'sig' cells in the file")
    for p, e, xs in cells:
        idx = old.index[sig_rows & (old.position == p) & (old.epoch == e)]
        ox = old.loc[idx, "x"].to_numpy(float)
        if len(idx) != len(xs) or not np.allclose(ox, xs):
            sys.exit(f"{sessions_csv.name}: bins differ for {p}/{e}")
        unc = np.zeros(len(xs), bool)
        ali = np.zeros(len(xs), bool)
        pc, qc, cid = np.full(len(xs), np.nan), np.full(len(xs), np.nan), np.full(len(xs), np.nan)
        for c in (c for c in clusters if c["position"] == p and c["epoch"] == e):
            at = np.isin(xs, c["_xs"])
            unc |= at & c["sig_uncorrected"]
            ali |= at & c["sig_aligned"]
            pc[at], qc[at], cid[at] = c["p"], c["q"], c["cluster"]
        was = old.loc[idx, "v"].astype(str).isin(["True", "1", "1.0"]).to_numpy()
        if not np.array_equal(was, parity[(p, e)]):
            sys.exit(f"PARITY FAILED {sessions_csv.name} {p}/{e}: recomputed uncorrected mask differs from the file "
                     f"({int(was.sum())} vs {int(parity[(p, e)].sum())} bins); nothing written")
        new.loc[idx, "v"] = ali.astype(float)            # 0.0 / 1.0, as the input file writes them
        new.loc[idx, "v_uncorrected"] = unc.astype(float)
        new.loc[idx, "p_cluster"], new.loc[idx, "q_cluster"], new.loc[idx, "cluster_id"] = pc, qc, cid
    new.to_csv(out, index=False)
    cl_df = pd.DataFrame([{k: v for k, v in c.items() if k != "_xs"} for c in clusters])
    cl_df.to_csv(out_cl, index=False)
    n_unc = int(cl_df.sig_uncorrected.sum()) if len(cl_df) else 0
    n_ali = int(cl_df.sig_aligned.sum()) if len(cl_df) else 0
    print(f"wrote {out.name}: {len(cl_df)} clusters in the family, significant uncorrected {n_unc} -> BH {n_ali}",
          flush=True)
    return cl_df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions_csv", type=Path)
    ap.add_argument("curves", type=Path)
    ap.add_argument("--kind", choices=("peak", "norm"), default=None,
                    help="curve kind (default: from the file name, '_peak' = peak)")
    ap.add_argument("--n-perm", type=int, default=CLUSTER_PERM,
                    help=f"cluster-test permutations (default CLUSTER_PERM = {CLUSTER_PERM}; the paper uses 5000)")
    ap.add_argument("--out-suffix", default="", help="appended to the output names (never overwrite)")
    a = ap.parse_args(argv)
    print(f"cluster permutations: {a.n_perm} (seed 0)", flush=True)
    run(a.sessions_csv, a.curves, n_perm=a.n_perm, kind=a.kind, out_suffix=a.out_suffix)


if __name__ == "__main__":
    main()
