"""3-D LICK TRAJECTORIES by target spout and epoch (Priya 10-24: "can we compare trajectory instead of landing for the
licks?"). LICK DATA only.

Per lick (pose3d `kinematics/licks` snapshot table, `t_peak_3d` = time of the 3-D peak, ms in trial): the tongue path
from the per-frame 3-D tracks (`session_poses/pose3d/<a>_<d>_frames.csv.gz`, tongue_ml / ap / dv, 250 Hz) from
PRE_MS before the peak to POST_MS after, on a fixed 4 ms grid (frames missing -> NaN). Aggregated per animal x epoch x
target spout (mean path, n), and for far-contra licks also by the lick's DOMINANT landing bin (softmax width 1, as the
encoder) -- does a far-contra lick that lands like far mid follow a far-mid path from the start, or deviate late?
August pre + acute + subacute + up to MAX_CHRONIC chronic sessions per animal (I/O: 7.7 GB of frames in all).

Usage: python scripts/lick_trajectories_3d.py [--jobs 4]
Output: <dlc>/session_poses/pose3d/trajectories/lick_trajectories.npz + figures/lick_trajectories_*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRE_MS, POST_MS, DT = 120, 80, 4
GRID = np.arange(-PRE_MS, POST_MS + DT, DT)
SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
MOUSE = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
         "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
I = Path("C:/Users/SabatiniLab/wfield_cache/pose3d_inputs")
MAX_CHRONIC = 5


def _dlc() -> Path:
    from wfield_local.paths import PathResolver
    return Path(PathResolver().root("dlc"))


def session_job(item):
    """(session, licks rows as records) -> {(target, landing): (sum path (T x 3), count)} for one session."""
    session, L = item
    F = pd.read_csv(_dlc() / "session_poses" / "pose3d" / f"{session}_frames.csv.gz",
                    usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"])
    F = F.sort_values(["trial_id", "t_ms"])
    by_trial = {t: g for t, g in F.groupby("trial_id")}
    acc = {}
    for r in L:
        g = by_trial.get(r["trial_id"])
        if g is None:
            continue
        t = g.t_ms.to_numpy(float) - r["t_peak_3d"]
        m = (t >= -PRE_MS - DT) & (t <= POST_MS + DT)
        if m.sum() < 10:
            continue
        path = np.column_stack([np.interp(GRID, t[m], g[c].to_numpy(float)[m], left=np.nan, right=np.nan)
                                for c in ("tongue_ml", "tongue_ap", "tongue_dv")])
        for key in ((r["target"], "all"), (r["target"], r["landing"])):
            s = acc.setdefault(key, [np.zeros_like(path), np.zeros_like(path)])
            ok = np.isfinite(path)
            s[0][ok] += path[ok]
            s[1][ok] += 1
    return session, acc


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    K = pd.read_csv(I / "pose3d_kin_licks_snapshot_20261023.csv.gz",
                    usecols=["animal", "session", "epoch", "trial_id", "position", "t_peak_3d", "side_ml", "out_ap",
                             "down_dv", "date"])
    K = K[(K.date.astype(str).str[4:6] != "06") & K.t_peak_3d.notna()].dropna(subset=["side_ml", "out_ap", "down_dv"])
    K["target"] = K.position.map(MOUSE)
    C = pd.read_csv(I / "exec_bins_calib_20261023.csv")
    C = C[C.calib_for == "default"]
    land = []
    for an, k in K.groupby("animal"):
        c = C[C.animal == an].set_index("position").reindex(SP)
        G, sc = c[["ml", "ap", "down"]].to_numpy(float), c[["sd_ml", "sd_ap", "sd_down"]].iloc[0].to_numpy(float)
        X = k[["side_ml", "out_ap", "down_dv"]].to_numpy(float)
        d = (((X / sc)[:, None, :] - (G / sc)[None]) ** 2).sum(2)
        land.append(pd.Series(np.array(SP)[d.argmin(1)], index=k.index))
    K["landing"] = pd.concat(land)
    sess = []
    for (_an, ep), g in K.groupby(["animal", "epoch"]):
        ss = sorted(g.session.unique())
        if ep == "chronic":
            ss = ss[:: max(1, len(ss) // MAX_CHRONIC)][:MAX_CHRONIC]
        sess += ss
    items = [(s, K[K.session == s][["trial_id", "t_peak_3d", "target", "landing"]].to_dict("records")) for s in sess]
    print(f"[trajectories] {len(items)} sessions, {sum(len(x[1]) for x in items)} licks", flush=True)
    res, fail = ak.fan_sessions(items, session_job, jobs=a.jobs, key=lambda it: it[0], label="session")
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    meta = K.drop_duplicates("session").set_index("session")[["animal", "epoch"]]
    agg = {}
    for _it, (session, acc) in res:
        an, ep = meta.loc[session, "animal"], meta.loc[session, "epoch"]
        for (tg, ld), (s, n) in acc.items():
            k = (an, ep, tg, ld)
            if k not in agg:
                agg[k] = [np.zeros_like(s), np.zeros_like(n)]
            agg[k][0] += s
            agg[k][1] += n
    out = _dlc() / "session_poses" / "pose3d" / "trajectories"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "lick_trajectories.npz", grid_ms=GRID,
                        **{"|".join(k): np.where(n > 0, s / np.maximum(n, 1), np.nan) for k, (s, n) in agg.items()},
                        **{"n|" + "|".join(k): n[:, 0] for k, (s, n) in agg.items()})
    COL = {"close_ipsi": "#9ecae1", "close_mid": "#969696", "close_contra": "#fdae6b",
           "far_ipsi": "#08519c", "far_mid": "#252525", "far_contra": "#d94801"}
    AN, EP = ["PS92", "PS93", "PS94", "PS95"], ["pre", "acute", "subacute", "chronic"]
    for view, (ix, iy, xl, yl) in {"top": (0, 1, "lateral (ml, mm; + = contralesional)", "forward (ap, mm)"),
                                   "side": (1, 2, "forward (ap, mm)", "down (dv, mm)")}.items():
        fig, axs = plt.subplots(4, 4, figsize=(14, 13), sharex="row", sharey="row")
        for i, an in enumerate(AN):
            for j, ep in enumerate(EP):
                ax = axs[i, j]
                for tg in SP:
                    k = (an, ep, tg, "all")
                    if k not in agg:
                        continue
                    s, n = agg[k]
                    P = np.where(n > 0, s / np.maximum(n, 1), np.nan)
                    lw = 2.4 if tg == "far_contra" else 1.3
                    ax.plot(P[:, ix], P[:, iy], color=COL[tg], lw=lw, label=tg if (i == 0 and j == 0) else None)
                    pk = np.argmin(np.abs(GRID))
                    ax.plot(P[pk, ix], P[pk, iy], "o", color=COL[tg], ms=4)
                nfc = int(agg.get((an, ep, "far_contra", "all"), [None, np.zeros((1, 3))])[1].max())
                ax.set_title(f"{an} {ep} (far-contra licks {nfc})", fontsize=9)
                if j == 0:
                    ax.set_ylabel(yl)
                if i == 3:
                    ax.set_xlabel(xl)
        axs[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"LICK DATA: mean 3-D tongue TRAJECTORY per target spout ({view} view), {PRE_MS} ms before to {POST_MS} ms "
                     f"after the lick's peak (dot = peak). Far contra thick orange.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / "figures" / f"lick_trajectories_{view}.png", dpi=125)
        plt.close(fig)
    # far-contra licks by where they landed
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for j, an in enumerate(AN):
        for r, ep in enumerate(("pre", "acute")):
            ax = axs[r, j]
            for ld in SP:
                k = (an, ep, "far_contra", ld)
                if k in agg and agg[k][1].max() >= 15:
                    s, n = agg[k]
                    P = np.where(n > 0, s / np.maximum(n, 1), np.nan)
                    ax.plot(P[:, 0], P[:, 1], color=COL[ld], lw=2, label=f"landed {ld} (n {int(n.max())})")
            for tg in ("far_mid", "far_ipsi", "far_contra"):
                k = (an, "pre", tg, "all")
                if k in agg:
                    s, n = agg[k]
                    P = np.where(n > 0, s / np.maximum(n, 1), np.nan)
                    ax.plot(P[:, 0], P[:, 1], color=COL[tg], lw=0.8, ls=":")
            ax.set_title(f"{an} {ep}: far-contra licks by landing bin", fontsize=9)
            ax.legend(fontsize=6, frameon=False)
            ax.set_xlabel("lateral (ml, mm)")
            if j == 0:
                ax.set_ylabel("forward (ap, mm)")
    fig.suptitle("LICK DATA: far-contra licks split by the bin they landed in (solid), vs the pre-stroke mean paths to far "
                 "mid / far ipsi / far contra (dotted). Do mis-landing licks follow another spout's path from the start?",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "figures" / "lick_trajectories_farcontra_by_landing.png", dpi=125)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
