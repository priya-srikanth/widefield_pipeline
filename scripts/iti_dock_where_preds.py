"""DOCKED-SPOUT copies of the frozen WHERE-decoder per-lick caches (Priya 10-29 late: the docked-spout ITI licks become
the primary ITI set; "replace the main ITI figures with this (if they support the same conclusions)").

The dock flags (iti_dock_audit.py -> <OUT>/iti_dock_audit.csv) are keyed on the frozen lick_order prediction cache
(label x k). The where-decoder caches (zscore_check/where_preds_z<PRED_TAG>.csv, where_angle_reach_preds<PRED_TAG>.csv.gz,
where_preds<PRED_TAG>.csv) hold the same ITI licks in the same order plus 18 licks the lick_order cache lacks; per session
the two are ALIGNED on the sequence of lateral landings (difflib, exact to 1e-4 mm; every lick_order row is matched,
checked here) and each where row inherits its lick's flag (unmatched rows = not kept). Nothing is refit (rule 10): the
cached frozen predictions are only subset.
Writes, under iti_licks.OUT_W (the filter's folder or WFIELD_ITI_OUT_W; never OUT itself):
  zscore_check/where_preds_z<PRED_TAG>.csv, where_angle_reach_preds<PRED_TAG>.csv.gz, where_preds<PRED_TAG>.csv
      ITI rows kept only if flagged, cued rows unchanged
  zscore_check/where_decoder_z.csv   the per-session r table of iti_where_decoder (z-scored features), ITI rows
      recomputed from the kept licks (lateral, sagittal, outward, angle, reach, 2-D = vector r with the animal's PRE
      kept-lick SDs), sessions with < MIN_N kept licks dropped; cued rows unchanged
  where_dockflag.csv   per where_preds_z ITI row: label, i (row within the session), k (lick_order row or -1), keep
`iti_licks.pred_file` returns these copies when a subset filter is active and they exist.
Usage: WFIELD_ITI_FILTER=dock [WFIELD_ITI_OUT_W=<dir>] python scripts/iti_dock_where_preds.py
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402

MIN_N = 20          # as iti_where_decoder.decode (>= 20 licks per session)


def _vec_r(P: np.ndarray, Y: np.ndarray) -> float:
    P, Y = P - P.mean(0), Y - Y.mean(0)
    return float((P * Y).sum() / np.sqrt((P ** 2).sum() * (Y ** 2).sum()))


def flags(W: pd.DataFrame) -> pd.DataFrame:
    """Dock flag per ITI row of a where cache (cache order), by sequence alignment to the lick_order cache."""
    L = pd.read_csv(lo.preds_path(), usecols=["set", "label", "act"])
    L = L[L.set == "iti"].copy()
    L["k"] = L.groupby("label").cumcount()
    keepL = il.iti_subset_keep(L.assign(set="iti"))
    L["keep"] = keepL
    out, unmatched = [], 0
    Wi = W[W.set == "iti"]
    for lab, w in Wi.groupby("label", sort=False):
        l_ = L[L.label == lab]
        a, b = list(np.round(w.act_lat.to_numpy(float), 4)), list(np.round(l_.act.to_numpy(float), 4))
        k = np.full(len(a), -1)
        for bl in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            k[bl.a:bl.a + bl.size] = np.arange(bl.b, bl.b + bl.size)
        unmatched += len(b) - int((k >= 0).sum())
        kp = np.where(k >= 0, l_.keep.to_numpy()[np.clip(k, 0, None)], False)
        out.append(pd.DataFrame({"label": lab, "i": np.arange(len(a)), "k": k, "keep": kp}))
    if unmatched:
        raise SystemExit(f"{unmatched} lick_order ITI rows not matched in the where cache -- alignment failed")
    return pd.concat(out, ignore_index=True)


def subset(W: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    iti = (W.set == "iti").to_numpy()
    keep = np.ones(len(W), bool)
    keep[iti] = F.keep.to_numpy()        # F is in the cache's ITI row order
    return W[keep].reset_index(drop=True)


def r_table(Wz: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """iti_where_decoder.decode's per-session r rows, from cached predictions (ITI only)."""
    rows = []
    Wz, A = Wz[Wz.set == "iti"], A[A.set == "iti"]
    for an, g in Wz.groupby("animal"):
        pre = g[g.epoch == "pre"]
        sd = np.array([pre.act_lat.std(ddof=0), pre.act_sag.std(ddof=0)])
        for lab, h in g.groupby("label", sort=False):
            if len(h) < MIN_N:
                continue
            a = A[A.label == lab]
            Y = np.column_stack([h.act_lat, h.act_sag])
            Pr = np.column_stack([h.dec_lat, h.dec_sag])
            r = {"lateral": np.corrcoef(h.dec_lat, h.act_lat)[0, 1], "sagittal": np.corrcoef(h.dec_sag, h.act_sag)[0, 1],
                 "2-D": _vec_r(Pr / sd, Y / sd), "angle": np.corrcoef(a.dec_ang, a.act_ang)[0, 1],
                 "reach": np.corrcoef(a.dec_reach, a.act_reach)[0, 1], "outward": np.corrcoef(h.dec_out, h.act_out)[0, 1]}
            for tg, v in r.items():
                rows.append({"source": "areas", "set": "iti", "label": lab, "animal": an, "epoch": h.epoch.iloc[0],
                             "target": tg, "r": float(v), "n": len(h), "sd_lateral": float(h.act_lat.std(ddof=0)),
                             "sd_sagittal": float(h.act_sag.std(ddof=0)), "sd_angle": float(a.act_ang.std(ddof=0)),
                             "sd_reach": float(a.act_reach.std(ddof=0)), "sd_outward": float(h.act_out.std(ddof=0))})
    return pd.DataFrame(rows)


def main() -> int:
    if not il.ITI_SUBSET or il.OUT_W == il.OUT:
        raise SystemExit("needs WFIELD_ITI_FILTER in " + ", ".join(il.ITI_SUBSET_FILTERS) + " (writes under OUT_W only)")
    rel_z = Path("zscore_check") / il.tagged("where_preds_z.csv", il.PRED_TAG)
    rel_a = Path(il.tagged("where_angle_reach_preds.csv.gz", il.PRED_TAG))
    rel_u = Path(il.tagged("where_preds.csv", il.PRED_TAG))
    Wz = pd.read_csv(il.OUT / rel_z)
    A = pd.read_csv(il.OUT / rel_a)
    U = pd.read_csv(il.OUT / rel_u)
    # the three caches are row-aligned (same extraction): check before reusing one flag vector
    for nm, X, c in (("angle_reach", A, "act_sag"), ("unscaled", U, "act_lat")):
        if len(X) != len(Wz) or not np.allclose(X[c].to_numpy(float), Wz[c].to_numpy(float), equal_nan=True):
            raise SystemExit(f"{nm} cache not row-aligned with {rel_z}")
    F = flags(Wz)
    (il.OUT_W / "zscore_check").mkdir(parents=True, exist_ok=True)
    F.to_csv(il.OUT_W / "where_dockflag.csv", index=False)
    print(f"ITI subset {il.ITI_SUBSET}: {int(F.keep.sum())} of {len(F)} where-cache ITI rows kept "
          f"({int((F.k < 0).sum())} rows without a lick_order match, dropped)")
    Wz2, A2, U2 = subset(Wz, F), subset(A, F), subset(U, F)
    Wz2.to_csv(il.OUT_W / rel_z, index=False)
    A2.to_csv(il.OUT_W / rel_a, index=False)
    U2.to_csv(il.OUT_W / rel_u, index=False)
    R0 = pd.read_csv(il.OUT / "zscore_check" / il.tagged("where_decoder_z.csv"))
    chk = r_table(Wz, A)                                     # reproduce the published ITI r from the full caches
    m = R0[R0.set == "iti"].merge(chk, on=["label", "target"], suffixes=("", "_chk"))
    print(f"check vs published where_decoder_z ITI rows: max |dr| {np.abs(m.r - m.r_chk).max():.2e} over {len(m)} rows"
          f" (2-D uses the pre SD of all pre licks there, ddof 0)")
    R = pd.concat([r_table(Wz2, A2), R0[R0.set != "iti"]], ignore_index=True)[R0.columns]
    R.to_csv(il.OUT_W / "zscore_check" / il.tagged("where_decoder_z.csv"), index=False)
    print(R[R.set == "iti"].groupby(["animal", "epoch"]).n.agg(["size", "min", "median"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
