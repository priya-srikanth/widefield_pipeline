"""Lick phase and trajectory plots in 3-D, a la the cam4 cohort (`orofacial_cohort._lick_phase`, `_phase_norm`,
`_phase_fig`): Priya 10-19 "can we do phase and trajectory plots a la cam4?".

    python -m scripts.pose3d_phase [--pre-from 20260801]      # after scripts.pose3d_kinematics

Licks = the cam4 lick events with >= 3 3-D frames (`kinematics/licks.csv.gz`), aligned on their 3-D PEAK (`t_peak_3d`).
Per lick, from the 3-D frames (`session_poses/pose3d/<s>_frames.csv.gz`):
  * PEAK-ALIGNED (+-152 ms, 4 ms): ``dist`` = 3-D distance of the tip from the mouth zero (cam4's 'protrusion', without the
    projection); in-mouth frames (no tongue in either camera, within the lick's trial) count as 0, as on cam4; ``vrad`` =
    d(dist)/dt (mm/s); components ``out`` (ap), ``down`` (-dv), ``side`` (ml, + = mouse LEFT = ipsi) and ``mid`` = the
    angle out of the midline plane (deg, + = ipsi; NaN in the mouth).
  * NORMALISED PHASE: each lick from its cam4 onset (u = 0) to offset (u = 1) on 41 points (cam4's U_NORM), same measures.
Session means per position, then epoch mean +- SEM over sessions (August pre = baseline). Figures in
`session_poses/pose3d/phase/`, per animal: ``traces_<a>.png`` (dist vs ms from peak), ``portrait_<a>.png`` (radial speed vs
dist over one cycle, +-76 ms, dot = peak), ``phase_<a>.png`` (midline angle and dist vs normalised phase), and
``trajectories_<a>.png`` (mean peak-aligned path per position in the FRONT (side x down), SIDE (out x down) and TOP (side x
out) planes, spouts marked).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
EP_COL = {"pre": "0.2", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
HALF_MS, PORTRAIT_MS = 152.0, 76.0
T_REL = np.arange(-HALF_MS, HALF_MS + 0.1, 4.0)
U_NORM = np.linspace(0.0, 1.0, 41)
MEAS = ("dist", "vrad", "out", "down", "side", "mid")


def _session(d3: Path, s: str, L: pd.DataFrame) -> list[dict]:
    S = pd.read_csv(d3 / f"{s}_frames.csv.gz", usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"])
    S = S.drop_duplicates(["trial_id", "t_ms"]).set_index(["trial_id", "t_ms"]).sort_index()
    rows = []
    for p, g in L.groupby("position"):
        acc = {("peak", m): [] for m in MEAS} | {("norm", m): [] for m in MEAS}
        for r in g.itertuples():
            for kind, tt in (("peak", r.t_peak_3d + T_REL), ("norm", r.on_ms + U_NORM * (r.off_ms - r.on_ms))):
                tq = np.round(tt / 4.0) * 4.0 if kind == "norm" else tt
                idx = pd.MultiIndex.from_arrays([np.full(len(tq), r.trial_id), np.round(tq, 1)])
                P = S.reindex(idx).to_numpy(float)
                known = np.isin(np.round(tq, 1), S.loc[r.trial_id].index.to_numpy()) if r.trial_id in S.index else \
                    np.zeros(len(tq), bool)
                dist = np.linalg.norm(P, axis=1)
                dist = np.where(np.isfinite(dist), dist, np.where(known, 0.0, np.nan))     # in the mouth = 0, as cam4
                with np.errstate(all="ignore"):
                    mid = np.degrees(np.arcsin(np.clip(-P[:, 0] / np.linalg.norm(P, axis=1), -1, 1)))
                vals = {"dist": dist, "vrad": np.gradient(dist) * 250.0, "out": P[:, 1], "down": -P[:, 2],
                        "side": -P[:, 0], "mid": mid}
                for m in MEAS:
                    acc[(kind, m)].append(vals[m])
        for (kind, m), arrs in acc.items():
            if len(arrs) < 5:
                continue
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    A = np.vstack(arrs)
                    mean, cov = np.nanmean(A, axis=0), np.isfinite(A).mean(axis=0)
            x = T_REL if kind == "peak" else U_NORM
            rows += [{"session": s, "position": p, "kind": kind, "measure": m, "x": float(xi), "v": float(vi),
                      "cov": float(ci), "n": len(arrs)} for xi, vi, ci in zip(x, mean, cov)]
    return rows


def _epoch_curves(P: pd.DataFrame, animal: str, kind: str, m: str, pre_from: str) -> dict:
    g = P[(P.animal == animal) & (P.kind == kind) & (P.measure == m)]
    g = g[(g.epoch != "pre") | (g.date >= pre_from)]
    out = {}
    for (ep, pos), ge in g.groupby(["epoch", "position"]):
        piv = ge.pivot_table(index="session", columns="x", values="v")
        cov = ge.pivot_table(index="session", columns="x", values="cov").mean().to_numpy()
        sem = piv.std(ddof=1) / np.sqrt(len(piv)) if len(piv) > 1 else piv.mean() * np.nan
        out[(ep, pos)] = (piv.columns.to_numpy(), piv.mean().to_numpy(), sem.to_numpy(), len(piv), cov)
    return out


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_lines(P, animal, kind, m, ylab, xlab, path, title, pre_from):
    plt = _plt()
    C = _epoch_curves(P, animal, kind, m, pre_from)
    fig, axes = plt.subplots(1, len(POS), figsize=(2.9 * len(POS), 2.7), sharey=True)
    for ax, p in zip(axes, POS):
        for ep in EP_COL:
            if (ep, p) in C:
                x, y, e, n, _ = C[(ep, p)]
                ax.plot(x, y, color=EP_COL[ep], lw=1.4, label=f"{ep} ({n})")
                ax.fill_between(x, y - e, y + e, color=EP_COL[ep], alpha=0.2, lw=0)
        ax.axvline(0 if kind == "peak" else 0.5, color="0.85", lw=0.8, zorder=0)
        ax.set_title(p, fontsize=8)
        ax.set_xlabel(xlab, fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel(ylab, fontsize=7)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_portrait(P, animal, path, pre_from):
    plt = _plt()
    D = _epoch_curves(P, animal, "peak", "dist", pre_from)
    V = _epoch_curves(P, animal, "peak", "vrad", pre_from)
    fig, axes = plt.subplots(1, len(POS), figsize=(2.9 * len(POS), 2.8), sharey=True, sharex=True)
    for ax, p in zip(axes, POS):
        for ep in EP_COL:
            if (ep, p) in D and (ep, p) in V:
                x, d, _, n, _ = D[(ep, p)]
                _, v, _, _, _ = V[(ep, p)]
                w = np.abs(x) <= PORTRAIT_MS
                ax.plot(d[w], v[w], color=EP_COL[ep], lw=1.4, label=f"{ep} ({n})")
                ax.plot(d[x == 0], v[x == 0], "o", color=EP_COL[ep], ms=3)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(p, fontsize=8)
        ax.set_xlabel("3-D distance from mouth zero (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("radial speed (mm/s)", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{animal}: 3-D lick phase portrait (+-{PORTRAIT_MS:.0f} ms around the 3-D peak, dot = peak; in-mouth = 0 mm)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_trajectories(P, animal, geo, path, pre_from):
    from wfield_local import spout_world as SW
    plt = _plt()
    # NORMALISED PHASE (onset -> offset, 41 points), drawn only where >= half the licks have a 3-D point: a peak-aligned
    # time mean changes which licks it averages along the path (10-19: jagged mean paths)
    comp = {m: _epoch_curves(P, animal, "norm", m, pre_from) for m in ("side", "out", "down")}
    planes = (("front", "side", "down", "side (mm; + = mouse left)", "down (mm)"),
              ("side", "out", "down", "out of the mouth (mm)", "down (mm)"),
              ("top", "side", "out", "side (mm; + = mouse left)", "out of the mouth (mm)"))
    fig, axes = plt.subplots(len(planes), len(POS), figsize=(2.6 * len(POS), 2.6 * len(planes)), squeeze=False)
    tgt = {p: SW.commanded_world(geo, p) for p in POS}
    tv = {p: {"side": -tgt[p][0], "out": tgt[p][1], "down": -tgt[p][2]} for p in POS}
    for i, (name, xa, ya, xl, yl) in enumerate(planes):
        for j, p in enumerate(POS):
            ax = axes[i, j]
            for ep in EP_COL:
                if (ep, p) in comp[xa] and (ep, p) in comp[ya]:
                    x, xv, _, n, cov = comp[xa][(ep, p)]
                    _, yv, _, _, _ = comp[ya][(ep, p)]
                    w = cov >= 0.5
                    ax.plot(xv[w], yv[w], color=EP_COL[ep], lw=1.4, label=f"{ep} ({n})")
                    k = int(np.argmin(np.abs(x - 0.5)))
                    if w[k]:
                        ax.plot(xv[k], yv[k], "o", color=EP_COL[ep], ms=3)
            ax.scatter(tv[p][xa], tv[p][ya], s=90, facecolor="none", edgecolor="k", lw=1.2, zorder=5)
            ax.scatter(0, 0, marker="+", s=60, color="k", zorder=5)
            if ya == "down":
                ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(p, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{name}\n{yl}", fontsize=7)
            ax.set_xlabel(xl, fontsize=6.5)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{animal}: mean 3-D tongue path per position (normalised lick phase, onset -> offset; shown where >= half "
                 "the licks have a 3-D point; dot = mid-lick; open circle = target spout, + = mouth zero)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    from scripts.apparatus_mouth import _rig
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-from", default="20260801")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    L = pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str})
    V36 = Path(rv.root("orofacial")) / "lp_r4_heatmap_20261013__clean_v36" / "sessions"
    rows = []
    for s, g in L.groupby("session"):
        animal, date = s.split("_")
        C = pd.read_csv(V36 / animal / date / "licks.csv", usecols=["trial_id", "lick_idx", "on_ms", "off_ms"])
        g = g.merge(C, on=["trial_id", "lick_idx"], how="left").dropna(subset=["on_ms", "off_ms", "t_peak_3d"])
        for r in _session(d3, s, g):
            rows.append({**r, "animal": animal, "date": date, "epoch": g.epoch.iloc[0]})
        print(f"{s}: {len(g)} licks")
    P = pd.DataFrame(rows)
    out = d3 / "phase"
    out.mkdir(exist_ok=True)
    P.to_csv(out / "curves.csv.gz", index=False, float_format="%.4g")
    for animal in sorted(P.animal.unique()):
        last = sorted(L[L.animal == animal].date.astype(str).unique())[-1]
        geo = _rig(rv, animal, last)["geometry"]
        fig_lines(P, animal, "peak", "dist", "3-D distance from mouth zero (mm)", "ms from 3-D lick peak",
                  out / f"traces_{animal}.png", f"{animal}: peak-aligned 3-D tongue distance (epoch mean +- SEM over "
                  "sessions; in-mouth = 0 mm)", a.pre_from)
        fig_portrait(P, animal, out / f"portrait_{animal}.png", a.pre_from)
        fig_lines(P, animal, "norm", "mid", "angle out of the midline plane (deg; + = ipsi)",
                  "normalised lick phase (0 = onset, 1 = offset)", out / f"phase_angle_{animal}.png",
                  f"{animal}: aim through the lick -- angle out of the midline plane vs normalised phase", a.pre_from)
        fig_lines(P, animal, "norm", "dist", "3-D distance from mouth zero (mm)", "normalised lick phase",
                  out / f"phase_dist_{animal}.png", f"{animal}: 3-D distance vs normalised lick phase", a.pre_from)
        fig_trajectories(P, animal, geo, out / f"trajectories_{animal}.png", a.pre_from)
        print(f"-> {out} ({animal})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
