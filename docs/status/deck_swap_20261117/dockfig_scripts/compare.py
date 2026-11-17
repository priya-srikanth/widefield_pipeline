"""Key numbers for DOCKED_DECODED_VS_EXECUTED_20261117.md: docked (dock, gap2dock) vs all-ITI on the _daqlk rebuild, and
the pre-rebuild docked set (itidock_main). Reads only finished tables / figure data; writes compare.txt beside this file.
Usage: python compare.py   (any env with pandas; paths are absolute)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

I = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
         "iti_licks")
TAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
OLD = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/"
           "itidock_main")
PROD = I / TAG
DOCK0 = I / "matched_gauss_dock_20261117" / TAG          # the 11-17 docked run (reused)
DOCK = I / "dock_dockfig_20261117" / TAG
GAP2D = I / "gap2dock_dockfig_20261117" / TAG
ALL = I / "allITI_dockfig_20261117" / TAG
EP = ["acute", "subacute", "chronic"]
AN = ["PS92", "PS93", "PS94", "PS95"]
out = []


def say(s: str = "") -> None:
    print(s)
    out.append(s)


def fmt(r) -> str:
    sig = "*" if (r["lo"] > 0 or r["hi"] < 0) else ""
    nd = f" {int(r['n_dir'])}/{int(r['n'])}" if "n_dir" in r and pd.notna(r.get("n_dir")) else \
        (f" {int(r['n_pos'])}/{int(r['n'])}" if "n_pos" in r else "")
    return f"{r['point']:+.2f}{sig} [{r['lo']:+.2f},{r['hi']:+.2f}]{nd}"


def matched(path: Path, label: str, sets=("iti", "trial"), clss=("all", "first")) -> None:
    if not path.exists():
        say(f"  {label}: MISSING {path}")
        return
    T = pd.read_csv(path)
    T = T[(T.scope == "all spouts") & (T.measure == "resid")]
    for st in sets:
        for c in clss:
            g = T[(T.set == st) & (T.cls == c)].set_index("epoch")
            say(f"  {label:34s} {st:5s} {c:5s} " + " | ".join(fmt(g.loc[e]) if e in g.index else "-" for e in EP))


def section_matched() -> None:
    say("== 1. LIKE-FOR-LIKE lateral offset (Gaussian same-place reference), all spouts, change vs pre (mm; "
        "acute | subacute | chronic; point* [95% CI] k/n)")
    matched(PROD / "matched_gauss_tests.csv", "all-ITI (production)")
    matched(PROD / "matched_gauss_gap2_tests.csv", "all-ITI gap2 (refit, production)", sets=("iti",))
    matched(DOCK0 / "matched_gauss_tests.csv", "docked (matched_gauss_dock 11-17)")
    matched(DOCK / "matched_gauss_tests.csv", "docked (this rerun)")
    matched(GAP2D / "matched_gauss_tests.csv", "gap2dock (this run)")
    matched(OLD / "matched_gauss_tests.csv", "PRE-REBUILD docked (itidock_main)")
    a, b = DOCK0 / "matched_gauss_sessions.csv", DOCK / "matched_gauss_sessions.csv"
    if a.exists() and b.exists():
        A, B = pd.read_csv(a), pd.read_csv(b)
        same = A.shape == B.shape and np.allclose(A.resid.to_numpy(float), B.resid.to_numpy(float), equal_nan=True)
        say(f"  reproduction check: docked rerun sessions table == matched_gauss_dock_20261117: {same}")
    say("  per animal, ITI all licks (change vs pre, mm):")
    for path, lab in ((PROD / "matched_gauss_tests.csv", "all-ITI"), (DOCK0 / "matched_gauss_tests.csv", "docked"),
                      (GAP2D / "matched_gauss_tests.csv", "gap2dock")):
        if not path.exists():
            continue
        T = pd.read_csv(path)
        g = T[(T.scope == "all spouts") & (T.measure == "resid") & (T.set == "iti") & (T.cls == "all")].set_index("epoch")
        say(f"    {lab:9s} " + " | ".join(f"{e[:3]} " + " ".join(f"{a[2:]}:{g.loc[e, f'animal_{a}']:+.2f}" for a in AN)
                                         for e in EP if e in g.index))


def curve_shift(fd: Path, key: str, label: str) -> None:
    """Mean vertical shift of each epoch's animal-mean curve vs pre over landing grid points where pre and the epoch
    both have all 4 animals (the solid stretches of the figure)."""
    if not fd.exists():
        say(f"  {label}: MISSING {fd}")
        return
    D = pickle.loads(fd.read_bytes())
    A = D.get(key)
    if A is None or len(A) == 0:
        say(f"  {label}: no {key} in {fd.name}")
        return
    for st in ("iti", "trial"):
        g = A[A.set == st]
        mu = g.groupby(["epoch", "mid"]).y.agg(["mean", "count"])
        pre = mu.loc["pre"]
        cells = []
        for e in EP:
            if e not in mu.index.get_level_values(0):
                cells.append("-")
                continue
            m = mu.loc[e].join(pre, rsuffix="_pre")
            m4 = m[(m["count"] >= 4) & (m["count_pre"] >= 4)]
            tag3 = "" if len(m4) else " (3 animals)"
            m = m4 if len(m4) else m[(m["count"] >= 3) & (m["count_pre"] >= 4)]
            cells.append(f"{(m['mean'] - m['mean_pre']).mean():+.2f} over {m.index.min():+.2f}..{m.index.max():+.2f}{tag3}"
                         if len(m) else "-")
        say(f"  {label:30s} {st:5s} " + " | ".join(cells))


def section_curves() -> None:
    say("\n== 2. CURVE SHIFT: mean (epoch - pre) of the animal-mean decoded-vs-landing curve over the shared 4-animal "
        "landing range (mm; acute | subacute | chronic)")
    curve_shift(PROD / "figdata" / "lick_matched_gauss.pkl", "A", "all-ITI all licks")
    curve_shift(ALL / "figdata" / "lick_matched_gauss.pkl", "A1", "all-ITI FIRST licks")
    curve_shift(DOCK0 / "figdata" / "lick_matched_gauss.pkl", "A", "docked all licks")
    curve_shift(DOCK / "figdata" / "lick_matched_gauss.pkl", "A1", "docked FIRST licks")
    curve_shift(GAP2D / "figdata" / "lick_matched_gauss.pkl", "A", "gap2dock (bout start)")
    say("  ANGLE curves (deg):")
    for root, lab in ((ALL, "all-ITI"), (DOCK, "docked"), (GAP2D, "gap2dock")):
        fd = root / "figdata" / "lick_matched_angle_reach_curves.pkl"
        if not fd.exists():
            say(f"  {lab}: MISSING {fd}")
            continue
        C = pickle.loads(fd.read_bytes())
        for cls in ("all", "first"):
            curve_shift_df(C["curves"][C["curves"].cls == cls], f"{lab} {cls} (sigma {C['sigma']:g} deg)")


def curve_shift_df(A: pd.DataFrame, label: str) -> None:
    for st in ("iti", "trial"):
        g = A[A.set == st]
        mu = g.groupby(["epoch", "mid"]).y.agg(["mean", "count"])
        if "pre" not in mu.index.get_level_values(0):
            continue
        pre = mu.loc["pre"]
        cells = []
        for e in EP:
            if e not in mu.index.get_level_values(0):
                cells.append("-")
                continue
            m = mu.loc[e].join(pre, rsuffix="_pre")
            m4 = m[(m["count"] >= 4) & (m["count_pre"] >= 4)]
            tag3 = "" if len(m4) else " (3 animals)"
            m = m4 if len(m4) else m[(m["count"] >= 3) & (m["count_pre"] >= 4)]
            cells.append(f"{(m['mean'] - m['mean_pre']).mean():+.1f} over {m.index.min():+.0f}..{m.index.max():+.0f}{tag3}"
                         if len(m) else "-")
        say(f"  {label:34s} {st:5s} " + " | ".join(cells))


def section_angle() -> None:
    say("\n== 3. LIKE-FOR-LIKE ANGLE offset (Gaussian reference), change vs pre (deg; acute | subacute | chronic)")
    for root, lab in ((PROD, "all-ITI (production)"), (DOCK, "docked"), (GAP2D, "gap2dock")):
        p = root / "matched_axes_tests.csv"
        if not p.exists():
            say(f"  {lab}: MISSING {p}")
            continue
        T = pd.read_csv(p)
        for ax in ("angle", "reach"):
            for st in ("iti", "trial"):
                g = T[(T.axis == ax) & (T.scope == "gauss") & (T.set == st)].set_index("epoch")
                say(f"  {lab:22s} {ax:6s} {st:5s} " + " | ".join(fmt(g.loc[e]) if e in g.index else "-" for e in EP))


def section_delta() -> None:
    say("\n== 4. WHERE DELTA (decoding error vector change vs pre, z-scored decoder), LATERAL axis, all licks "
        "(mm; acute | subacute | chronic; k/n animals > 0)")
    for root, lab in ((PROD, "all-ITI (production)"), (DOCK, "docked"), (GAP2D, "gap2dock"),
                      (OLD, "PRE-REBUILD docked")):
        p = root / "zscore_check" / "where_delta_tests_z.csv"
        if not p.exists():
            say(f"  {lab}: MISSING {p}")
            continue
        T = pd.read_csv(p)
        for st in ("iti", "trial"):
            g = T[(T.set == st) & (T.licks == "all") & (T.axis == "lat")].set_index("epoch")
            say(f"  {lab:22s} {st:5s} " + " | ".join(fmt(g.loc[e]) if e in g.index else "-" for e in EP))


def section_rlink() -> None:
    say("\n== 5. RECOVERY LINK (aim_z): ITI like-for-like lateral offset (lfl_lat, change vs pre, mm) and aim "
        "(change vs pre, deg) per animal and epoch; within-animal r(lfl_lat, aim) over post sessions")
    for root, lab in ((ALL, "all-ITI (z, this run)"), (DOCK, "docked"), (GAP2D, "gap2dock"),
                      (OLD, "PRE-REBUILD docked")):
        s, w = root / "recovery_link_sessions_aim_z.csv", root / "recovery_link_within_aim_z.csv"
        if not s.exists():
            say(f"  {lab}: MISSING {s}")
            continue
        G = pd.read_csv(s)
        for st in ("iti", "trial"):
            g = G[G.set == st].groupby(["animal", "epoch"])[["lfl_lat", "beh"]].mean().unstack("epoch")
            n = G[G.set == st].groupby(["animal", "epoch"]).size().unstack("epoch")
            say(f"  {lab} [{st}] lfl_lat: " + "; ".join(
                f"{a} " + "/".join(f"{g.loc[a, ('lfl_lat', e)]:+.2f}" if (a in g.index and ('lfl_lat', e) in g.columns
                                    and pd.notna(g.loc[a, ('lfl_lat', e)])) else "nan" for e in EP)
                + f" (n {'/'.join(str(int(n.loc[a, e])) if a in n.index and e in n.columns and pd.notna(n.loc[a, e]) else '0' for e in EP)})"
                for a in AN))
        g = G[G.set == "iti"].drop_duplicates("label").groupby(["animal", "epoch"]).beh.mean().unstack("epoch")
        say(f"  {lab} aim (deg): " + "; ".join(f"{a} " + "/".join(f"{g.loc[a, e]:+.1f}" if a in g.index and e in g.columns
                                                                    and pd.notna(g.loc[a, e]) else "nan" for e in EP)
                                                 for a in AN))
        if w.exists():
            W = pd.read_csv(w)
            for st in ("iti", "trial"):
                r = W[(W.set == st) & (W.measure == "lfl_lat")]
                say(f"  {lab} [{st}] within-animal r(lfl_lat, aim): " + "; ".join(
                    f"{k.kind} mean {k.mean_r:+.2f} (p {k.p_perm:.3f}; " + " ".join(f"{a[2:]} {getattr(k, a):+.2f}" for a in AN) + ")"
                    for k in r.itertuples()))


def section_error() -> None:
    """Per-lick error SHIFT (decoded - actual lateral / angle, session mean) change vs pre, ITI: nested bootstrap."""
    sys.path.insert(0, "C:/Users/SabatiniLab/Github/wf_cleanup2")
    from wfield_local import analysis_kit as ak
    say("\n== 6. PER-LICK ERROR SHIFT, ITI (session mean of decoded - actual; change vs pre; acute | subacute | chronic)")
    for root, lab in ((PROD, "all-ITI (production)"), (DOCK, "docked"), (GAP2D, "gap2dock")):
        p = root / "zscore_check" / "where_preds_z_splitr8.csv"
        if not p.exists():
            say(f"  {lab}: MISSING {p}")
            continue
        P = pd.read_csv(p)
        P = P[P.set == "iti"].dropna(subset=["act_lat", "dec_lat"]).copy()
        P["e_lat"] = P.dec_lat - P.act_lat
        ok = P.act_sag.notna() & P.dec_sag.notna()
        P["e_ang"] = np.where(ok, np.degrees(np.arctan2(P.dec_lat, P.dec_sag)) - np.degrees(np.arctan2(P.act_lat, P.act_sag)),
                              np.nan)
        S = P.groupby(["animal", "epoch", "label"])[["e_lat", "e_ang"]].agg(["mean", "size"])
        for c, unit in (("e_lat", "mm"), ("e_ang", "deg")):
            s = S[c]
            s = s[s["size"] >= 20]["mean"].reset_index()
            prd = {a: g["mean"].to_numpy() for a, g in s[s.epoch == "pre"].groupby("animal")}
            cells = []
            for e in EP:
                post = {a: g["mean"].to_numpy() for a, g in s[s.epoch == e].groupby("animal") if a in prd}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                cells.append(f"{ci.point:+.2f}{'*' if ci.lo > 0 or ci.hi < 0 else ''} [{ci.lo:+.2f},{ci.hi:+.2f}]")
            n = s.groupby("epoch").size().reindex(["pre", *EP]).fillna(0).astype(int).tolist()
            say(f"  {lab:22s} {c} ({unit}) " + " | ".join(cells) + f"   sessions >= 20 licks pre/ac/sub/chr {n}")


def section_counts() -> None:
    say("\n== 7. LICK COUNTS (ITI rows of the KEY lateral cache / where cache)")
    for root, lab in ((PROD, "all-ITI"), (DOCK, "docked"), (GAP2D, "gap2dock")):
        p = root / "zscore_check" / "where_preds_z_splitr8.csv"
        if p.exists():
            P = pd.read_csv(p, usecols=["set", "epoch"])
            say(f"  {lab:9s} where ITI licks by epoch: " + str(P[P.set == "iti"].groupby("epoch").size().reindex(["pre", *EP]).to_dict()))


if __name__ == "__main__":
    section_matched()
    section_curves()
    section_angle()
    section_delta()
    section_rlink()
    section_error()
    section_counts()
    (Path(__file__).parent / "compare.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
