"""Error distance from the spout, in 3-D: where the tongue ends up relative to the TARGET (Priya 10-19: "error distance
from spout is still an interesting (and separate) analysis"). Separate from the direction readout (`pose3d_kinematics`):
this one is referenced to the spout itself, so the post-stroke set-point change (spouts ~0.5 mm closer, slack jaw,
EXPERIMENT_ERRORS) does not enter it.

    python -m scripts.pose3d_spout_error [--pre-from 20260801]      # after scripts.pose3d_kinematics

THREE ERRORS PER LICK (Priya 10-19: closest approach or peak? -- "for close spouts, normal peak lick is well below the
spout, the tongue doesn't just go to the tip of the spout"). They answer different questions, so all three are kept:
  1. CLOSEST APPROACH to the spout tip (``miss_mm`` + its vector ``lat_err_mm`` / ``ap_err_mm`` / ``dv_err_mm``): did the
     tongue get to the spout (water access)? For close spouts this is usually on the way DOWN past the spout.
  2. PEAK ENDPOINT vs the spout (``peak_*``): the 3-D extension peak (`pose3d_kinematics`) minus the spout tip. Normally
     well below the spout (the drop), so its size is not an error by itself.
  3. PEAK ENDPOINT vs the animal's OWN pre-stroke endpoint for that position (``habit_*``; the median August pre peak):
     error in the motor-control sense -- how far the lick ends from where it normally ends -- with no assumption about
     where on / under the spout the target is.
Vectors: lateral + = outward of the spout (centre: + = mouse left), ap + = beyond (away from the mouth), dv + = above. Per session x
position: median miss, the fraction of licks within 0.5 / 1.0 mm, split by DAQ contact (cam4 ``contact``) and by first
lick of the trial vs later licks (``lick_idx`` 0). Distance-to-spout TRACE per lick: |tip - spout| from 20 ms before the
lick's cam4 onset to 200 ms after, 4 ms bins.
Figures (`session_poses/pose3d/spout_error/`): ``scatter_<animal>.png`` -- closest-approach points in a SPOUT-CENTRED frame,
one column per position, rows = FRONT (lateral outward x down) and SIDE (beyond x down), August pre (grey) vs each post
session (colour), medians as large markers, the spout at (0, 0); ``traces_<animal>.png`` -- median distance-to-spout traces.
Tables: per_session.csv; by_animal.csv (post minus the animal's August pre median -- rule 8, printed first).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
EP_COL = {"acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
TRACE_MS = np.arange(-20.0, 200.1, 4.0)


def _summ(g: pd.DataFrame) -> dict:
    return {"n": len(g), "miss_med": g.miss_mm.median(), "within_0p5": float((g.miss_mm <= 0.5).mean()),
            "within_1": float((g.miss_mm <= 1.0).mean()), "lat_med": g.lat_err_mm.median(),
            "ap_med": g.ap_err_mm.median(), "dv_med": g.dv_err_mm.median(),
            "peak_dist_med": g.peak_dist_mm.median(), "peak_lat_med": g.peak_lat_mm.median(),
            "peak_ap_med": g.peak_ap_mm.median(), "peak_dv_med": g.peak_dv_mm.median(),
            "habit_dist_med": g.habit_dist_mm.median(), "habit_lat_med": g.habit_lat_mm.median(),
            "habit_ap_med": g.habit_ap_mm.median(), "habit_dv_med": g.habit_dv_mm.median()}


def add_peak_errors(rv, L: pd.DataFrame, pre_from: str) -> pd.DataFrame:
    """Errors 2 and 3 (module docstring) from the 3-D peak (``side_ml``, ``out_ap``, -``down_dv`` = the peak in the
    set-point frame)."""
    from scripts.apparatus_mouth import _rig
    from wfield_local import spout_world as SW
    L = L.copy()
    P = np.c_[L.side_ml, L.out_ap, -L.down_dv]
    T = np.full_like(P, np.nan)
    for s, idx in L.groupby("session").groups.items():
        animal, date = s.split("_")
        geo = _rig(rv, animal, date)["geometry"]
        rows = L.index.get_indexer(idx)
        T[rows] = np.array([SW.commanded_world(geo, p) for p in L.position.iloc[rows]])
    out = np.where(np.abs(T[:, 0]) > 1e-6, np.sign(T[:, 0]), -1.0)         # outward sign; centre: + = mouse left
    D = P - T
    L["peak_dist_mm"] = np.linalg.norm(D, axis=1)
    L["peak_lat_mm"], L["peak_ap_mm"], L["peak_dv_mm"] = out * D[:, 0], D[:, 1], D[:, 2]
    pre = L[(L.epoch == "pre") & (L.date.astype(str) >= pre_from)]
    hab = pre.assign(px=P[pre.index.to_numpy(), 0], py=P[pre.index.to_numpy(), 1], pz=P[pre.index.to_numpy(), 2]) \
        .groupby(["animal", "position"])[["px", "py", "pz"]].median()
    H = hab.reindex(pd.MultiIndex.from_arrays([L.animal, L.position])).to_numpy(float)
    D2 = P - H
    L["habit_dist_mm"] = np.linalg.norm(D2, axis=1)
    L["habit_lat_mm"], L["habit_ap_mm"], L["habit_dv_mm"] = out * D2[:, 0], D2[:, 1], D2[:, 2]
    return L


def per_session(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, a, d, ep, p), g in L.groupby(["session", "animal", "date", "epoch", "position"], dropna=False):
        base = {"session": s, "animal": a, "date": str(d), "epoch": ep, "position": p}
        rows.append({**base, "subset": "all", **_summ(g)})
        for name, m in (("contact", g.contact.astype(bool)), ("no_contact", ~g.contact.astype(bool)),
                        ("first_lick", g.lick_idx == 0), ("later_licks", g.lick_idx > 0)):
            if m.sum() >= 5:
                rows.append({**base, "subset": name, **_summ(g[m])})
    return pd.DataFrame(rows)


def traces(rv, L: pd.DataFrame) -> pd.DataFrame:
    """Median |tip - target| per session x position at TRACE_MS from the cam4 lick onset."""
    from scripts.apparatus_mouth import _rig
    from wfield_local import spout_world as SW
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    out = []
    for s, g in L.groupby("session"):
        animal, date = s.split("_")
        S = pd.read_csv(d3 / f"{s}_frames.csv.gz", usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"])
        S = S.drop_duplicates(["trial_id", "t_ms"]).set_index(["trial_id", "t_ms"])
        C = pd.read_csv(Path(rv.root("orofacial")) / V36 / "sessions" / animal / date / "licks.csv",
                        usecols=["trial_id", "lick_idx", "on_ms"])
        g = g.merge(C, on=["trial_id", "lick_idx"], how="left")
        geo = _rig(rv, animal, date)["geometry"]
        for p, gp in g.groupby("position"):
            T = SW.commanded_world(geo, p)
            M = np.full((len(gp), len(TRACE_MS)), np.nan)
            for i, r in enumerate(gp.itertuples()):
                if not np.isfinite(r.on_ms):
                    continue
                idx = pd.MultiIndex.from_arrays([np.full(len(TRACE_MS), r.trial_id), np.round(r.on_ms + TRACE_MS, 1)])
                P = S.reindex(idx).to_numpy(float)
                M[i] = np.linalg.norm(P - T, axis=1)
            with np.errstate(all="ignore"):
                med = np.nanmedian(M, axis=0)
            out += [{"session": s, "animal": animal, "date": date, "epoch": gp.epoch.iloc[0], "position": p,
                     "t_ms": t, "dist_med": v, "n": int(np.isfinite(M[:, k]).sum())} for k, (t, v) in enumerate(zip(TRACE_MS, med))]
    return pd.DataFrame(out)


def fig_scatter(L: pd.DataFrame, animal: str, path: Path, pre_from: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = L[L.animal == animal].copy()
    A["lat_ipsi"] = _ipsi(A)
    pre = A[(A.epoch == "pre") & (A.date.astype(str) >= pre_from)]
    post = A[A.epoch.isin(list(EP_COL))]
    fig, axes = plt.subplots(2, len(POS), figsize=(3.0 * len(POS), 6.4), squeeze=False)
    rng = np.random.default_rng(0)
    for j, p in enumerate(POS):
        for i, (xc, xl) in enumerate((("lat_ipsi", LAT_LABEL), ("ap_err_mm", "beyond spout (mm; + = away)"))):
            ax = axes[i, j]
            g = pre[pre.position == p]
            if len(g):
                k = rng.choice(len(g), min(400, len(g)), replace=False)
                ax.scatter(g[xc].iloc[k], g.dv_err_mm.iloc[k], s=3, color="0.6", alpha=0.35, lw=0)
                ax.scatter(g[xc].median(), g.dv_err_mm.median(), s=90, color="0.3", edgecolor="k", zorder=5, label="pre (Aug)")
            for s, gs in post[post.position == p].groupby("session"):
                c = EP_COL.get(gs.epoch.iloc[0], "C0")
                k = rng.choice(len(gs), min(200, len(gs)), replace=False)
                ax.scatter(gs[xc].iloc[k], gs.dv_err_mm.iloc[k], s=4, color=c, alpha=0.35, lw=0)
                ax.scatter(gs[xc].median(), gs.dv_err_mm.median(), s=90, color=c, edgecolor="k", zorder=6,
                           label=f"{s.split('_')[1]} {gs.epoch.iloc[0]}")
            ax.scatter(0, 0, marker="o", s=160, facecolor="none", edgecolor="k", lw=1.5, zorder=7)
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 2)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=7)
            ax.set_ylabel("dv (mm; + = above)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{p} -- {'front' if i == 0 else 'side'}", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="lower left")
    fig.suptitle(f"{animal}: closest approach of the tongue tip to the target spout (spout = open circle at 0, 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _ipsi(g: pd.DataFrame) -> pd.Series:
    """Lateral closest-approach error with ONE sign for every spout: + = toward IPSI (mouse left), drawn to the RIGHT as
    in cam4's front view (facing the mouse). ``lat_err_mm`` is + = OUTWARD (centre: + = mouse left), which flips between
    the L and R spouts (Priya 10-20: acute far_R looked 'flipped' -- it was 3 mm inward = toward ipsi)."""
    return g.lat_err_mm * np.where(g.position.astype(str).str.endswith("_R"), -1.0, 1.0)


LAT_LABEL = "lateral (mm; + = ipsi / mouse left, right as cam4)"


def fig_scatter_epoch(L: pd.DataFrame, animal: str, path: Path, pre_from: str) -> None:
    """Epoch version (Priya 10-20): each SESSION's median closest approach as a faded dot, the epoch mean of the session
    medians as a large marker (pre = August); no per-lick cloud."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = L[L.animal == animal].copy()
    A = A[(A.epoch != "pre") | (A.date.astype(str) >= pre_from)]
    A["lat_ipsi"] = _ipsi(A)
    M = A.groupby(["session", "epoch", "position"])[["lat_ipsi", "ap_err_mm", "dv_err_mm"]].median().reset_index()
    cols = {"pre": "0.3", **EP_COL}
    fig, axes = plt.subplots(2, len(POS), figsize=(3.0 * len(POS), 6.4), squeeze=False)
    for j, p in enumerate(POS):
        for i, (xc, xl) in enumerate((("lat_ipsi", LAT_LABEL), ("ap_err_mm", "beyond spout (mm; + = away)"))):
            ax = axes[i, j]
            for ep, c in cols.items():
                g = M[(M.position == p) & (M.epoch == ep)]
                if g.empty:
                    continue
                ax.scatter(g[xc], g.dv_err_mm, s=22, color=c, alpha=0.35, lw=0)
                ax.scatter(g[xc].mean(), g.dv_err_mm.mean(), s=110, color=c, edgecolor="k", zorder=6,
                           label=f"{ep} (n={len(g)})")
            ax.scatter(0, 0, marker="o", s=160, facecolor="none", edgecolor="k", lw=1.5, zorder=7)
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-2.5, 1.5)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=7)
            ax.set_ylabel("dv (mm; + = above)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{p} -- {'front' if i == 0 else 'side'}", fontsize=8)
    axes[0, -1].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"{animal}: closest approach of the tongue tip to the target spout, by epoch (spout = open circle at 0, "
                 "0)\nfaded = one session's median over its licks; large = epoch mean of the session medians; pre = "
                 "August", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_traces_epoch(Tr: pd.DataFrame, animal: str, path: Path, pre_from: str) -> None:
    """Epoch version: each session's median trace faded, the epoch mean of the session traces bold (pre = August)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = Tr[Tr.animal == animal]
    A = A[(A.epoch != "pre") | (A.date.astype(str) >= pre_from)]
    cols = {"pre": "0.3", **EP_COL}
    fig, axes = plt.subplots(1, len(POS), figsize=(3.0 * len(POS), 3.2), sharey=True)
    for ax, p in zip(axes, POS):
        for ep, c in cols.items():
            g = A[(A.position == p) & (A.epoch == ep)]
            if g.empty:
                continue
            for _s, gs in g.groupby("session"):
                ax.plot(gs.t_ms, gs.dist_med, color=c, lw=0.7, alpha=0.25)
            m = g.groupby("t_ms").dist_med.mean()
            ax.plot(m.index, m.values, color=c, lw=2.0, alpha=0.85, label=f"{ep} (n={g.session.nunique()})")
        ax.set_title(p, fontsize=8)
        ax.set_xlabel("ms from lick onset (cam4)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("tip - target spout (mm, median)", fontsize=7)
    axes[-1].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{animal}: distance from the target spout through the lick, by epoch (faded = one session's median "
                 "over its licks; bold = epoch mean of the sessions; pre = August)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_traces(Tr: pd.DataFrame, animal: str, path: Path, pre_from: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = Tr[Tr.animal == animal]
    fig, axes = plt.subplots(1, len(POS), figsize=(3.0 * len(POS), 3.0), sharey=True)
    for ax, p in zip(axes, POS):
        g = A[A.position == p]
        pre = g[(g.epoch == "pre") & (g.date.astype(str) >= pre_from)]
        if len(pre):
            m = pre.groupby("t_ms").dist_med.median()
            ax.plot(m.index, m.values, color="0.3", lw=2, label="pre (Aug)")
        for s, gs in g[g.epoch.isin(list(EP_COL))].groupby("session"):
            ax.plot(gs.t_ms, gs.dist_med, color=EP_COL[gs.epoch.iloc[0]], lw=1.5, label=f"{s.split('_')[1]} {gs.epoch.iloc[0]}")
        ax.set_title(p, fontsize=8)
        ax.set_xlabel("ms from lick onset (cam4)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("tip - target spout (mm, median)", fontsize=7)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{animal}: distance from the target spout through the lick", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-from", default="20260801", help="baseline sessions from this date (August-only primary, 10-18)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    L = add_peak_errors(rv, pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str}), a.pre_from)
    out = d3 / "spout_error"
    out.mkdir(exist_ok=True)
    P = per_session(L)
    P.to_csv(out / "per_session.csv", index=False)
    Tr = traces(rv, L)
    Tr.to_csv(out / "traces.csv", index=False)
    # rule 8: each animal's post sessions minus its August pre median (per position x subset)
    keys = ["animal", "position", "subset"]
    meas = ["miss_med", "within_0p5", "within_1", "lat_med", "ap_med", "dv_med", "peak_dist_med", "peak_lat_med",
            "peak_ap_med", "peak_dv_med", "habit_dist_med", "habit_lat_med", "habit_ap_med", "habit_dv_med"]
    base = P[(P.epoch == "pre") & (P.date >= a.pre_from)].groupby(keys)[meas].median()
    rows = []
    for r in P[P.epoch.isin(list(EP_COL))].itertuples(index=False):
        k = (r.animal, r.position, r.subset)
        if k in base.index:
            rows.append({"animal": r.animal, "session": r.session, "epoch": r.epoch, "position": r.position,
                         "subset": r.subset, "n": r.n, **{m: getattr(r, m) - base.loc[k, m] for m in meas}})
    B = pd.DataFrame(rows)
    B.to_csv(out / "by_animal.csv", index=False)
    pd.set_option("display.width", 220)
    if len(B):
        cols = ["animal", "session", "epoch", "position", "n", "miss_med", "lat_med", "peak_dist_med", "peak_lat_med",
                "peak_dv_med", "habit_dist_med", "habit_lat_med", "habit_ap_med", "habit_dv_med"]
        print("post minus August pre, all licks (1 closest approach: miss / lat; 2 peak vs spout; 3 peak vs own pre endpoint):")
        print(B[B.subset == "all"][cols].round(2).to_string(index=False))
        print("\nfirst lick of the trial only:")
        print(B[B.subset == "first_lick"][cols].round(2).to_string(index=False))
        pre_hab = P[(P.epoch == "pre") & (P.date >= a.pre_from) & (P.subset == "all")]
        print("\npre (August) medians -- the size of each error before the stroke:")
        print(pre_hab.groupby(["animal", "position"])[["miss_med", "peak_dist_med", "peak_dv_med", "habit_dist_med"]]
              .median().round(2).to_string())
    for animal in sorted(L.animal.unique()):
        fig_scatter(L, animal, out / f"scatter_{animal}.png", a.pre_from)
        fig_traces(Tr, animal, out / f"traces_{animal}.png", a.pre_from)
        fig_scatter_epoch(L, animal, out / f"scatter_epoch_{animal}.png", a.pre_from)
        fig_traces_epoch(Tr, animal, out / f"traces_epoch_{animal}.png", a.pre_from)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
