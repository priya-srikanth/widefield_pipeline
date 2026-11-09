"""Per-spout decoding error and r (Priya 10-26: "should we look at r and lateral / distance error also per spout
position? (ITI using spout position preceding the ITI)").

Frozen where-decoder per lick (`where_preds<PRED_TAG>.csv`, from `iti_where_scatter.py`; Allen areas with the ALM / tjM1
split since 10-27 late). Grouping: in-trial licks by the trial's
TARGET spout; ITI licks by the PREVIOUS trial's spout (`iti_pos`, fixed before the lick -- unlike the nearest-spout
grouping of 3.20 / 3.21, which depends on where the lick landed). Per spout, change vs pre:
  lfl_lat / lfl_sag        LIKE-FOR-LIKE offset (replaces beyond_lat / beyond_sag, 'offset beyond the pull', Priya
                           10-27 late "yes replace beyond pull"): decoded minus the animal's PRE licks of the set (any
                           spout) that landed at the same place on that axis (`likeforlike.offset`; Gaussian reference by
                           default, WFIELD_LFL_REF=loclin); + lateral = read more contralesional, - sagittal = read closer
                           to the mouth than same-place pre licks;
  r_lat                    per-session r between decoded and actual lateral position within the spout's licks
                           (within-spout range is small, so r is low and spread-sensitive; read the offsets first).
Cells need >= 15 licks per session x spout. Per-animal table first; paired nested bootstrap (animals > sessions).
`--z` uses the within-session z-scored predictions (`iti_zscore_check.py`).
Outputs <E>/iti_licks/per_spout[_z]_{by_animal,tests}.csv, figures/lick_where_per_spout[_z].png; figure data
<E>/iti_licks/figdata/lick_where_per_spout[_z].pkl (`--plot-only` redraws from it alone: no bootstrap).
Usage: python scripts/iti_where_per_spout.py [--z] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import likeforlike as lfl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

POS = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")   # canonical order
EP = ["acute", "subacute", "chronic"]
MEAS = {"lfl_lat": "LATERAL like-for-like\noffset, mm (+ = read contra)",
        "lfl_sag": "SAGITTAL like-for-like\noffset, mm (- = read closer)",
        "r_lat": "frozen r, lateral\n(within spout)"}
EPC = {"acute": "#e7298a", "subacute": "#d4a017", "chronic": "#1b9e77"}
MIN_N = 15


def table(p: pd.DataFrame) -> pd.DataFrame:
    p = p.dropna(subset=["act_lat", "act_sag", "dec_lat", "dec_sag"]).reset_index(drop=True)
    for ax_ in ("lat", "sag"):                     # same-place pre reference over ALL the set's licks, then by spout
        p[f"lfl_{ax_}"] = lfl.offset(p, f"act_{ax_}", f"dec_{ax_}", lfl.SIGMA_MM, f"{ax_} (mm)")[0]
    p = p[p.pos.isin(POS)]

    def cell(g):
        r = np.corrcoef(g.dec_lat, g.act_lat)[0, 1] if g.act_lat.std() > 0 else np.nan
        return pd.Series({"lfl_lat": g.lfl_lat.mean(), "lfl_sag": g.lfl_sag.mean(), "r_lat": r, "n": len(g)})
    s = p.groupby(["set", "animal", "epoch", "label", "pos"]).apply(cell, include_groups=False).reset_index()
    return s[s.n >= MIN_N]


def tests(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per = [], []
    for (st, pos), g in s.groupby(["set", "pos"]):
        for m in MEAS:
            prd = {a: x[m].dropna().to_numpy() for a, x in g[g.epoch == "pre"].groupby("animal")}
            pre_a = g[g.epoch == "pre"].groupby("animal")[m].mean()
            for ep in EP:
                h = g[g.epoch == ep]
                post = {a: x[m].dropna().to_numpy() for a, x in h.groupby("animal")}
                post = {a: v for a, v in post.items() if len(v) and a in prd and len(prd[a])}
                am = h.groupby("animal")[m].mean() - pre_a
                for a, v in am.dropna().items():
                    per.append(dict(set=st, pos=pos, measure=m, epoch=ep, animal=a, change=v))
                ci = ak.boot_delta(post, prd, np.random.default_rng(0)) if len(post) >= 2 else None
                rows.append(dict(set=st, pos=pos, measure=m, epoch=ep,
                                 point=ci.point if ci else np.nan, lo=ci.lo if ci else np.nan,
                                 hi=ci.hi if ci else np.nan, n_pos=int((am.dropna() > 0).sum()),
                                 n=int(am.notna().sum())))
    return pd.DataFrame(rows), pd.DataFrame(per)


def session_changes(s: pd.DataFrame) -> pd.DataFrame:
    """Per session x spout: value minus its animal's PRE mean for that spout and set (pre sessions included, so their
    spread is visible)."""
    out = []
    for m in MEAS:
        pre = s[s.epoch == "pre"].groupby(["set", "animal", "pos"])[m].mean().rename("pre_m")
        x = s.join(pre, on=["set", "animal", "pos"]).dropna(subset=["pre_m", m])
        out.append(x.assign(measure=m, change=x[m] - x.pre_m)[["set", "animal", "epoch", "label", "pos", "measure",
                                                                 "change"]])
    return pd.concat(out, ignore_index=True)


def plot(T: pd.DataFrame, S: pd.DataFrame, tag: str) -> None:
    """3.9 layout: one dot per SESSION (change from its animal's pre mean, all animals pooled), bar = mean +- SEM over
    sessions, * = paired nested bootstrap (animals > sessions) excluding 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep4 = ["pre", *EP]
    col = {"pre": "0.25", **EPC}
    rows = [(st, m) for st in ("trial", "iti") for m in MEAS]
    fig, axs = plt.subplots(len(rows), len(POS), figsize=(2.9 * len(POS), 2.5 * len(rows)), squeeze=False, sharey="row")
    sl = {"trial": "CUED (by target)", "iti": "ITI (by previous spout)"}
    rng = np.random.default_rng(0)
    for i, (st, m) in enumerate(rows):
        for j, pos in enumerate(POS):
            ax = axs[i, j]
            ax.axhline(0, color="0.6", lw=0.7)
            for k, ep in enumerate(ep4):
                v = S[(S.set == st) & (S.pos == pos) & (S.measure == m) & (S.epoch == ep)].change.to_numpy()
                if not len(v):
                    continue
                ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=8, color=col[ep], alpha=0.45, lw=0)
                mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                ax.errorbar(k + 0.3, mu, yerr=se, fmt="o", color="k", ms=4, capsize=2, lw=1.2)
                t = T[(T.set == st) & (T.pos == pos) & (T.measure == m) & (T.epoch == ep)]
                if ep != "pre" and len(t) and np.isfinite(t.point.iloc[0]) and (t.lo.iloc[0] > 0 or t.hi.iloc[0] < 0):
                    ax.text(k, 1.0, "*", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
            ax.set_xticks(range(len(ep4)))
            ax.set_xticklabels(["pre", "acu", "sub", "chr"], fontsize=8)
            if i == 0:
                ax.set_title(pos.replace("_", " "), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sl[st]}\n{MEAS[m]}" + ("\n[decoder on Z-SCORED features]" if tag else ""), fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle(("Z-SCORED FEATURES (each area / its frame-wise SD in the session, pre and post alike) -- " if tag else "")
                 + f"NEURAL: frozen pre-stroke lick-position decoder PER SPOUT ({il.basis_label()}, lags 0..0.2 s)."
                 "\nOne dot per SESSION = change from its "
                 "animal's pre mean (all animals pooled; pre sessions show the pre spread); black = mean +- SEM over "
                 "sessions;\n* = paired nested bootstrap (animals > sessions) excludes 0. Cued licks grouped by target, "
                 "ITI licks by the previous trial's spout (fixed before the lick).\nLike-for-like offset = decoded minus "
                 f"the animal's pre licks that landed at the same place ({lfl.REF_LAB}).", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_png_svg(fig, WOUT / "figures" / f"lick_where_per_spout{tag}.png", dpi=95)
    plt.close(fig)
WOUT = il.OUT_W if il.ITI_SUBSET else il.OUT     # subset runs (dock, ...) write to their own folder (10-29)


def trial_set_sentence(z: bool = False) -> str:
    """ITI + cued licks per spout (Priya 11-09): cells of >= MIN_N licks per session x spout; the subset ITI filter only
    when il.pred_file resolves to its filtered copy."""
    import iti_where_decoder as wd
    rel = "zscore_check/where_preds_z.csv" if z else "where_preds.csv"
    cell = f"cells with >= {MIN_N} licks per session x spout"
    return wd.both_caption(wd.subset_in(il.pred_file(rel)), tracked="lateral + sagittal + outward",
                           spout="spout = the previous trial's spout (one of the six)", iti_extra=[cell],
                           cued_extra=[cell])


def main() -> int:
    from wfield_local import trial_set_text as tst
    z = "--z" in sys.argv
    tag = "_z" if z else ""
    tst.declare(text=trial_set_sentence(z))
    if plot_only():                      # redraw from figdata only: no data, no bootstrap
        F = load_fd(f"lick_where_per_spout{tag}")
        plot(F["T"], F["S"], F["tag"])
        return 0
    il.split_only("iti_where_per_spout")
    p = pd.read_csv(il.pred_file("zscore_check/where_preds_z.csv" if z else "where_preds.csv"))
    s = table(p)
    T, P = tests(s)
    pd.set_option("display.width", 230)
    print("== per animal (rule 8): change vs pre, chronic")
    print(P[P.epoch == "chronic"].pivot_table(index=["set", "measure", "pos"], columns="animal", values="change")
          .reindex(POS, level=2).round(2).to_string())
    print("\n== pooled (nested bootstrap)")
    print(T.pivot_table(index=["set", "measure", "pos"], columns="epoch", values="point").reindex(columns=EP)
          .reindex(POS, level=2).round(2).to_string())
    sig = T[(T.lo > 0) | (T.hi < 0)]
    print("\n== cells whose CI excludes 0\n" + sig.round(3).to_string(index=False))
    P.to_csv(WOUT / f"per_spout{tag}_by_animal.csv", index=False)
    T.to_csv(WOUT / f"per_spout{tag}_tests.csv", index=False)
    S = session_changes(s)
    S.to_csv(WOUT / f"per_spout{tag}_by_session.csv", index=False)
    save_fd(f"lick_where_per_spout{tag}", dict(T=T, S=S, tag=tag))
    plot(T, S, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
