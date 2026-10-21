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
``trajectories3d_<a>.png`` (oblique 3-D, tongue + jaw), ``trajectories_<a>.png`` (mean path per position in the FRONT (side x down), SIDE (out x down) and TOP (side x
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
#: normalised phase SHOWN only inside this range, as cam4 (`orofacial_cohort.U_SHOW`, Priya 10-18): the onset / offset
#: frames carry edge artefacts (speed differenced across the boundary; in fast trains the offset touches the next lick)
U_SHOW = (0.1, 0.9)
MEAS = ("dist", "dist_out", "vrad", "out", "down", "side", "mid", "mid0", "elev", "elev0", "speed", "v_side", "v_out",
        "v_down", "j_side", "j_out", "j_down")
#: in-mouth CONVENTIONS (as cam4, `orofacial_cohort` phase_norm_angle_deg vs _mouth0, 10-17): ``dist`` / ``mid0`` /
#: ``elev0`` count a frame with no 3-D tongue point as IN THE MOUTH (0 mm / 0 deg); ``dist_out`` / ``mid`` / ``elev`` use
#: tongue-out frames only. ``elev`` (Priya 10-20) = angle BELOW the horizontal plane through the mouth zero, the vertical
#: half of the aim (``mid`` = angle out of the midline plane, the sideways half).
OBLIQUE = (20.0, 45.0)      # matplotlib (elev, azim), as the paired video: in front of the mouse, to its right, 20 deg above


def _frames(d3: Path, s: str) -> pd.DataFrame:
    S = pd.read_csv(d3 / f"{s}_frames.csv.gz", usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv",
                                                         "jaw_ml", "jaw_ap", "jaw_dv"])
    S = S.drop_duplicates(["trial_id", "t_ms"]).set_index(["trial_id", "t_ms"]).sort_index()
    # 3-D tip VELOCITY per frame (central difference over consecutive 4 ms frames of the same trial, mm/s); undefined in the
    # mouth or across a gap (Priya 10-19: "will we add speed trajectory from 3-D?")
    X = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    tr, tt = S.index.get_level_values(0).to_numpy(), S.index.get_level_values(1).to_numpy(float)
    V = np.full_like(X, np.nan)
    c = (tr[2:] == tr[:-2]) & (np.abs(tt[2:] - tt[:-2] - 8.0) < 0.5)
    V[1:-1][c] = (X[2:][c] - X[:-2][c]) / 0.008
    S["v_ml"], S["v_ap"], S["v_dv"] = V[:, 0], V[:, 1], V[:, 2]
    return S


def _session(S: pd.DataFrame, s: str, L: pd.DataFrame) -> list[dict]:
    """Per position: every lick's samples looked up in ONE `get_indexer` call (10-21; it was a pandas reindex per lick
    and kind) -- the same exact-match lookup, so the values are unchanged; a missing (trial, t) row = unknown frame."""
    A = S[["tongue_ml", "tongue_ap", "tongue_dv", "v_ml", "v_ap", "v_dv", "jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    A = np.vstack([A, np.full((1, A.shape[1]), np.nan)])          # row -1 -> all NaN
    rows = []
    for p, g in L.groupby("position"):
        if len(g) < 5:
            continue
        out = {}
        for kind in ("peak", "norm"):
            if kind == "peak":
                TQ = g.t_peak_3d.to_numpy(float)[:, None] + T_REL[None, :]
            else:
                on, off = g.on_ms.to_numpy(float), g.off_ms.to_numpy(float)
                TQ = np.round((on[:, None] + U_NORM[None, :] * (off - on)[:, None]) / 4.0) * 4.0
            TR = np.repeat(g.trial_id.to_numpy(), TQ.shape[1])
            k = S.index.get_indexer(pd.MultiIndex.from_arrays([TR, np.round(TQ, 1).ravel()]))
            known = (k >= 0).reshape(TQ.shape)
            R_ = A[k].reshape(TQ.shape + (A.shape[1],))
            P, Vv, Jj = R_[..., 0:3], R_[..., 3:6], R_[..., 6:9]
            dist = np.linalg.norm(P, axis=2)
            dist = np.where(np.isfinite(dist), dist, np.where(known, 0.0, np.nan))     # in the mouth = 0, as cam4
            with np.errstate(all="ignore"):
                rr = np.linalg.norm(P, axis=2)
                mid = np.degrees(np.arcsin(np.clip(-P[..., 0] / rr, -1, 1)))
                elev = np.degrees(np.arcsin(np.clip(-P[..., 2] / rr, -1, 1)))
            inm = ~np.isfinite(rr) & known                                             # no point but frame exists
            vals = {"dist": dist, "dist_out": rr, "vrad": np.gradient(dist, axis=1) * 250.0, "out": P[..., 1],
                    "down": -P[..., 2], "side": -P[..., 0], "mid": mid, "mid0": np.where(inm, 0.0, mid), "elev": elev,
                    "elev0": np.where(inm, 0.0, elev), "speed": np.linalg.norm(Vv, axis=2), "v_side": -Vv[..., 0],
                    "v_out": Vv[..., 1], "v_down": -Vv[..., 2], "j_side": -Jj[..., 0], "j_out": Jj[..., 1],
                    "j_down": -Jj[..., 2]}
            for m in MEAS:
                out[(kind, m)] = vals[m]
        for kind in ("peak", "norm"):
            for m in MEAS:
                Aa = out[(kind, m)]
                import warnings
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean, cov = np.nanmean(Aa, axis=0), np.isfinite(Aa).mean(axis=0)
                x = T_REL if kind == "peak" else U_NORM
                rows += [{"session": s, "position": p, "kind": kind, "measure": m, "x": float(xi), "v": float(vi),
                          "cov": float(ci), "n": len(Aa)} for xi, vi, ci in zip(x, mean, cov)]
    return rows


def session_curves(job) -> list[dict]:
    """Both lick sets for ONE session (frames read once), CACHED on the session's 3-D frames + v36 licks.csv -- the
    inputs of its kinematics licks (`pose3d_kinematics.session_result`, which supplies them)."""
    import sys

    import scripts.pose3d_kinematics as K
    from wfield_local import result_cache
    from wfield_local.paths import PathResolver
    s = job
    rv = PathResolver()
    animal, date = s.split("_")
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    V36 = Path(rv.root("orofacial")) / "lp_r4_heatmap_20261013__clean_v36" / "sessions"
    fr, lk = d3 / f"{s}_frames.csv.gz", V36 / animal / date / "licks.csv"
    ver = result_cache.code_version(sys.modules[__name__], K)

    def compute():
        g, _J, row = K.session_result((f"{animal}:{date}", "cam4"))
        C = pd.read_csv(lk, usecols=["trial_id", "lick_idx", "on_ms", "off_ms"])
        g = g.merge(C, on=["trial_id", "lick_idx"], how="left").dropna(subset=["on_ms", "off_ms", "t_peak_3d"])
        S = _frames(d3, s)
        out = []
        # lick sets as cam4 (Priya 10-16 / 10-20): ALL post-cue licks and the FIRST post-cue lick of each trial
        for lset, gg in (("all", g), ("first", g[g.lick_idx == 0])):
            out += [{**r, "animal": animal, "date": date, "epoch": row["epoch"], "lickset": lset}
                    for r in _session(S, s, gg)]
        return out
    return result_cache.cached(d3 / "_cache" / "phase", s, [fr, lk], ver, compute)


def _epoch_curves(P: pd.DataFrame, animal: str, kind: str, m: str, pre_from: str) -> dict:
    g = P[(P.animal == animal) & (P.kind == kind) & (P.measure == m)]
    g = g[(g.epoch != "pre") | (g.date >= pre_from)]
    if kind == "norm":
        g = g[(g.x >= U_SHOW[0] - 1e-9) & (g.x <= U_SHOW[1] + 1e-9)]
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


def _how(kind: str, lset: str = "all", mouth: str = "out") -> str:
    """Second title line: how the curve was made (Priya 10-20: as much method detail as the cam4 titles)."""
    lk = "FIRST post-cue lick of each trial" if lset == "first" else "all post-cue licks (cam4 v36 lick events)"
    t = (f"phase {U_SHOW[0]:g}-{U_SHOW[1]:g} of each lick, onset -> offset (cam4 on / off)" if kind == "norm"
         else "time from each lick's 3-D peak (farthest frame from the mouth zero)")
    m = {"zero": "frames with no 3-D tongue point counted as IN THE MOUTH (0)",
         "out": "tongue-out frames only (frames with no 3-D point left out)",
         "na": ""}[mouth]
    return (f"{lk}; {t}\n{m}{'; ' if m else ''}3-D = both cameras or one camera + depth from neighbours (<= 24 ms); "
            "mean per session, then mean +- SEM over sessions; pre = August")


def _wrap(text: str, width: int) -> str:
    """Wrap each line of a title / label to ``width`` characters (Priya 10-20: titles and y labels were cut off)."""
    import textwrap
    return "\n".join(textwrap.fill(line, width) if line else line for line in text.split("\n"))


def fig_lines(P, animal, kind, m, ylab, xlab, path, title, pre_from):
    plt = _plt()
    C = _epoch_curves(P, animal, kind, m, pre_from)
    t = _wrap(title, 150)
    head_in = 0.135 * (t.count("\n") + 1) + 0.05                         # title block height (inches)
    fig, axes = plt.subplots(1, len(POS), figsize=(2.9 * len(POS), 2.6 + head_in), sharey=True)
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
    axes[0].set_ylabel(_wrap(ylab, 26), fontsize=7)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(t, fontsize=8.5)
    fig.tight_layout()                                                     # makes room for the suptitle itself
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


def fig_traj3d(P, animal, geo, path, pre_from, lset="all"):
    """Oblique 3-D view per spout position (Priya 10-19): mean TONGUE path (solid) and mean JAW path (dashed) through the
    lick, per epoch, on normalised phase where >= half the licks have a point; dot = mid-lick; spouts (target
    outlined in black), + = mouth zero. Axes: ml (+ = mouse right), ap (+ = out of the mouth), dv (+ = up)."""
    from wfield_local import spout_world as SW
    plt = _plt()
    comp = {m: _epoch_curves(P, animal, "norm", m, pre_from) for m in ("side", "out", "down", "j_side", "j_out", "j_down")}
    tgt = {p: SW.commanded_world(geo, p) for p in POS}
    COLS = {"far_L": "#1f77b4", "close_L": "#6baed6", "close_center": "#999999", "far_center": "#333333",
            "close_R": "#fc9272", "far_R": "#d62728"}
    fig = plt.figure(figsize=(4.2 * 3, 3.8 * 2))
    for j, p in enumerate(POS):
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        for q, c in tgt.items():
            ax.scatter(*c, s=70 if q == p else 35, color=COLS[q], edgecolor="k" if q == p else "none",
                       linewidth=1.5, alpha=1.0 if q == p else 0.35)
        ax.scatter(0, 0, 0, marker="+", s=70, color="k")
        for ep in EP_COL:
            for part, ls, lw in (("", "-", 2.0), ("j_", "--", 1.4)):
                keys = [f"{part}side", f"{part}out", f"{part}down"]
                if not all((ep, p) in comp[k] for k in keys):
                    continue
                x, side, _, n, cov = comp[keys[0]][(ep, p)]
                out_, down = comp[keys[1]][(ep, p)][1], comp[keys[2]][(ep, p)][1]
                w = cov >= 0.5
                ml, ap, dv = -side[w], out_[w], -down[w]
                ax.plot(ml, ap, dv, ls=ls, lw=lw, color=EP_COL[ep],
                        label=f"{ep} {'jaw' if part else 'tongue'} ({n})" if j == 0 else None)
                k = int(np.argmin(np.abs(x[w] - 0.5))) if w.any() else None
                if k is not None and not part:
                    ax.scatter(ml[k], ap[k], dv[k], s=18, color=EP_COL[ep])
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 4)          # the jaw point sits ~1 mm behind the mouth zero
        ax.set_zlim(-5, 0.5)
        ax.set_xlabel("ml (+R)", fontsize=7)
        ax.set_ylabel("ap (out)", fontsize=7)
        ax.set_zlabel("dv (up)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=OBLIQUE[0], azim=OBLIQUE[1])
        ax.set_title(p, fontsize=9)
    fig.legend(loc="lower center", ncol=4, fontsize=7, frameon=False)
    t = _wrap(f"{animal}: mean 3-D tongue (solid) and jaw (dashed) path through the lick, oblique view (in front of the "
              "mouse, to its right, 20 deg above); drawn where >= half the licks have a point; dot = mid-lick\n"
              + _how("norm", lset) + "; jaw also from one camera + rest-depth anchor", 120)
    fig.suptitle(t, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))                                # bottom strip: the legend
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_trajectories(P, animal, geo, path, pre_from, lset="all"):
    from wfield_local import spout_world as SW
    plt = _plt()
    # NORMALISED PHASE (onset -> offset, 41 points), drawn only where >= half the licks have a 3-D point: a peak-aligned
    # time mean changes which licks it averages along the path (10-19: jagged mean paths)
    comp = {m: _epoch_curves(P, animal, "norm", m, pre_from) for m in ("side", "out", "down")}
    # viewpoints (Priya 10-20): + = mouse LEFT drawn to the right is the view FACING the mouse (row 1, as cam4); row 2 has
    # the snout to the right = seen from the mouse's LEFT; row 3 has "out of the mouth" pointing DOWN the page (y
    # inverted) = seen from ABOVE a mouse facing down the page (its left on the viewer's right), so rows 1 and 3 both
    # read mouth at the top, tongue reaching down the page
    planes = (("front\n(facing the mouse,\nas cam4)", "side", "down", "side (mm; + = mouse left)", "down (mm)"),
              ("side\n(from the\nmouse's left)", "out", "down", "out of the mouth (mm)", "down (mm)"),
              ("above\n(mouse faces\ndown the page)", "side", "out", "side (mm; + = mouse left)", "out (mm)"))
    covmin = []
    fig, axes = plt.subplots(len(planes), len(POS), figsize=(2.6 * len(POS), 2.6 * len(planes)), squeeze=False)
    tgt = {p: SW.commanded_world(geo, p) for p in POS}
    tv = {p: {"side": -tgt[p][0], "out": tgt[p][1], "down": -tgt[p][2]} for p in POS}
    for i, (name, xa, ya, xl, yl) in enumerate(planes):
        ext = [[0.0], [0.0]]                              # this row's x / y extent: paths, spouts, mouth zero
        for j, p in enumerate(POS):
            ax = axes[i, j]
            for ep in EP_COL:
                if (ep, p) in comp[xa] and (ep, p) in comp[ya]:
                    x, xv, _, n, cov = comp[xa][(ep, p)]
                    _, yv, _, _, _ = comp[ya][(ep, p)]
                    w = cov >= 0.5
                    if w.any():
                        covmin.append(float(cov[w].min()))
                        ext[0] += list(xv[w][np.isfinite(xv[w])])
                        ext[1] += list(yv[w][np.isfinite(yv[w])])
                    ax.plot(xv[w], yv[w], color=EP_COL[ep], lw=1.4, label=f"{ep} ({n})")
                    k = int(np.argmin(np.abs(x - 0.5)))
                    if w[k]:
                        ax.plot(xv[k], yv[k], "o", color=EP_COL[ep], ms=3)
            ax.scatter(tv[p][xa], tv[p][ya], s=90, facecolor="none", edgecolor="k", lw=1.2, zorder=5)
            ax.scatter(0, 0, marker="+", s=60, color="k", zorder=5)
            ext[0].append(tv[p][xa])
            ext[1].append(tv[p][ya])
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(p, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{name}\n{yl}", fontsize=7)
            ax.set_xlabel(xl, fontsize=6.5)
        # SAME axes across the row (Priya 10-20), equal mm scaling; down / out of the mouth point DOWN the page
        pad = 0.3
        x0, x1 = min(ext[0]) - pad, max(ext[0]) + pad
        y0, y1 = min(ext[1]) - pad, max(ext[1]) + pad
        for j, ax in enumerate(axes[i]):
            ax.set_xlim(x0, x1)
            ax.set_ylim((y1, y0) if ya in ("down", "out") else (y0, y1))
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelleft=j == 0)              # shared range: y tick labels on the first panel only
    axes[0, 0].legend(fontsize=6, frameon=False)
    cm = min(covmin) if covmin else float("nan")
    t = _wrap(f"{animal}: mean 3-D tongue path per position (dot = mid-lick; open circle = target spout, + = mouth zero; "
              f"drawn where >= half the licks have a 3-D point -- lowest share drawn {cm:.2f})\n" + _how("norm", lset), 150)
    fig.suptitle(t, fontsize=8.5)
    fig.tight_layout(rect=(0.02, 0, 1, 1))                                # left strip: the row labels
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    from scripts.apparatus_mouth import _rig
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-from", default="20260801")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="output folder (checks)")
    a = ap.parse_args(argv)
    from wfield_local import analysis_kit as ak
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    L = pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str})
    res, fails = ak.fan_sessions(sorted(L.session.unique()), session_curves, jobs=a.jobs)
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    PP = pd.DataFrame([r for _, rr in res for r in rr])
    P = PP[PP.lickset == "all"]
    PF = PP[PP.lickset == "first"]
    out = a.out or d3 / "phase"
    out.mkdir(exist_ok=True, parents=True)
    PP.to_csv(out / "curves.csv.gz", index=False, float_format="%.4g")
    for animal in sorted(P.animal.unique()):
        last = sorted(L[L.animal == animal].date.astype(str).unique())[-1]
        geo = _rig(rv, animal, last)["geometry"]
        fig_lines(P, animal, "peak", "dist", "3-D distance from mouth zero (mm)", "ms from 3-D lick peak",
                  out / f"traces_{animal}.png", f"{animal}: peak-aligned 3-D tongue distance\n" + _how("peak", "all", "zero"),
                  a.pre_from)
        fig_portrait(P, animal, out / f"portrait_{animal}.png", a.pre_from)
        xl = "normalised lick phase (0 = onset, 1 = offset)"
        for PX, lset, sfx in ((P, "all", ""), (PF, "first", "_first")):
            who = "FIRST post-cue lick -- " if lset == "first" else ""
            for m, mouth, yl, nm, what in (
                    ("mid", "out", "deg out of midline\n(+ = ipsi)", "phase_angle", "aim, sideways"),
                    ("mid0", "zero", "deg out of midline\n(+ = ipsi)", "phase_angle_mouth0",
                     "aim, sideways (in-mouth = 0)"),
                    ("elev", "out", "deg below horizontal\n(+ = down)", "phase_elev", "aim, vertical"),
                    ("elev0", "zero", "deg below horizontal\n(+ = down)", "phase_elev_mouth0",
                     "aim, vertical (in-mouth = 0)"),
                    ("dist", "zero", "3-D distance from mouth zero (mm)", "phase_dist", "3-D distance (in-mouth = 0)"),
                    ("dist_out", "out", "3-D distance from mouth zero (mm)", "phase_dist_out", "3-D distance, tongue out"),
                    ("speed", "out", "3-D tip speed (mm/s)", "v_speed_norm", "3-D tip speed")):
                fig_lines(PX, animal, "norm", m, yl, xl, out / f"{nm}{sfx}_{animal}.png",
                          f"{animal}: {who}{what} through the lick\n" + _how("norm", lset, mouth), a.pre_from)
            fig_trajectories(PX, animal, geo, out / f"trajectories{sfx}_{animal}.png", a.pre_from, lset)
            fig_traj3d(PX, animal, geo, out / f"trajectories3d{sfx}_{animal}.png", a.pre_from, lset)
        fig_lines(P, animal, "peak", "speed", "3-D tip speed (mm/s)", "ms from 3-D lick peak",
                  out / f"v_speed_peak_{animal}.png", f"{animal}: 3-D tongue-tip speed\n" + _how("peak", "all", "out"),
                  a.pre_from)
        for kind, xlab in (("peak", "ms from 3-D lick peak"), ("norm", xl)):
            for m, lab in (("v_side", "sideways (+ = toward ipsi)"), ("v_out", "forward (+ = out)"),
                           ("v_down", "downward (+ = down)")):
                fig_lines(P, animal, kind, m, f"{lab} velocity (mm/s)", xlab, out / f"{m}_{kind}_{animal}.png",
                          f"{animal}: {lab} velocity of the tongue tip\n" + _how(kind, "all", "out"), a.pre_from)
        print(f"-> {out} ({animal})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
