"""3-D tongue / jaw readouts across sessions, per animal (Priya 10-19: "where are the 3-D kinematics plots").

    python -m scripts.pose3d_summary_fig [--pre-from 20260801]      # after scripts.pose3d_kinematics + scripts.pose3d_spout_error

One figure per animal, `session_poses/pose3d/figures/summary_<animal>.png`: x = days from the stroke (config stroke_date; the
lesion followed that day's session, so day 1 = the first post session), one dot per session (median over its licks / trials),
coloured by epoch; columns = the six positions; grey band = the animal's AUGUST pre range (min-max over sessions), dashed
line = its median. Rows, in the order agreed 10-19 (distance offsets primary, one angle secondary):
  1. closest approach to the target spout (mm)                    -- spout error 1, origin-free
  2. lateral error at closest approach (mm; + = outward, centre + = mouse left)
  3-5. endpoint vs the animal's OWN pre endpoint: lateral / out (+ = beyond) / up (mm)   -- spout error 3
  6-8. DIRECTION from the mouth zero vs the direction to the target spout: the 3-D angle error; the angle OUT OF THE
     MIDLINE PLANE (the sagittal plane through the mouth zero and the close / far centre spouts = ml 0; asin(ml / |r|),
     + = toward ipsi; Priya 10-19) and the forward tilt WITHIN it (atan2(out, down)). Consistent across the post-stroke
     set-point move (Priya 10-19: the spouts are commanded relative to the set point, which marks the mouth that day).
     Non-zero before the stroke (licks run steeper than the line to the spout tips): read against the pre band.
  9. reach from the mouth zero (mm)   10. peak 3-D tongue speed (mm/s)   11. widest jaw opening (mm)
plus a final panel row-strip: jaw FAR_DIR (mm; + = toward the target side) per session.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
EP_COL = {"pre": "0.35", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
ROWS = (("se", "miss_med", "closest approach to spout (mm)"),
        ("se", "lat_med", "lateral error at closest\n(mm; + outward)"),
        ("se", "habit_lat_med", "endpoint vs own pre:\nlateral (mm; + outward)"),
        ("se", "habit_ap_med", "endpoint vs own pre:\nout (mm; + beyond)"),
        ("se", "habit_dv_med", "endpoint vs own pre:\nup (mm)"),
        ("k", "ang3d_err", "3-D angle error vs the\nspout direction (deg)"),
        ("k", "midline_err_ipsi", "angle out of the midline\nplane, error (deg; + ipsi)"),
        ("k", "sagittal_err", "forward tilt in the midline\nplane, error (deg; + forward)"),
        ("k", "reach_mm", "reach from mouth zero (mm)"),
        ("k", "speed_peak", "peak tongue speed (mm/s)"),
        ("k", "open_max_mm", "widest jaw opening (mm)"))


JUNE_X = -22.0      # June pre sessions (days ~-75 .. -70) are drawn in a compressed segment here, labelled "June"


def _x(animal: str, date: str) -> float:
    """Plot position: days from the stroke, June squeezed into a short segment left of August (Priya 10-19: PS92's June days
    sit at day ~-70 and flattened the August / acute part of the axis)."""
    d = _days(animal, date)
    return JUNE_X + 0.4 * (d + 72) if str(date) < "20260801" else float(d)


def _xaxis(ax) -> None:
    lo, hi = ax.get_xlim()
    ticks = [t for t in range(-15, int(hi) + 1, 5)]
    ax.set_xticks([JUNE_X] + ticks)
    ax.set_xticklabels(["June"] + [str(t) for t in ticks])
    ax.axvline(JUNE_X + 3.5, color="0.7", lw=0.6, ls=(0, (2, 2)))


def _days(animal: str, date: str) -> int:
    from wfield_local import config
    sd = pd.Timestamp(config._to8(config.stroke_date(animal)))      # config gives MMDD; _to8 adds the cohort year
    return int((pd.Timestamp(str(date)) - sd).days)


def figure(animal: str, SE: pd.DataFrame, K: pd.DataFrame, out: Path, pre_from: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    se = SE[(SE.animal == animal) & (SE.subset == "all")]
    k = K[K.animal == animal]

    def _has(src, col):
        if src == "se":
            return col in se and se[col].notna().any()
        return any(f"{col}|{p}" in k and k[f"{col}|{p}"].notna().any() for p in POS)

    # rows with no data for THIS animal are dropped (PS94 10-21: no 3-D error rows yet left a blank third of the page and
    # took the spout titles, set on row 0, with them)
    rows = [r for r in ROWS if _has(r[0], r[1])]
    fig, axes = plt.subplots(len(rows) + 1, len(POS), figsize=(2.9 * len(POS), 2.0 * (len(rows) + 1)), squeeze=False)
    for i, (src, col, label) in enumerate(rows):
        for j, p in enumerate(POS):
            ax = axes[i, j]
            if src == "se":
                d = se[se.position == p][["date", "epoch", col]].rename(columns={col: "v"})
            else:
                c = f"{col}|{p}"
                d = k[["date", "epoch"]].assign(v=k[c]) if c in k else pd.DataFrame(columns=["date", "epoch", "v"])
            d = d.dropna(subset=["v"])
            if d.empty:
                ax.set_visible(False)
                continue
            d = d.assign(day=[_x(animal, x) for x in d.date], june=[str(x) < "20260801" for x in d.date])
            pre = d[(d.epoch == "pre") & (d.date.astype(str) >= pre_from)]
            if len(pre):
                ax.axhspan(pre.v.min(), pre.v.max(), color="0.85", zorder=0)
                ax.axhline(pre.v.median(), color="0.4", ls="--", lw=0.8, zorder=1)
            for ep, g in d.groupby("epoch"):
                gj, ga = g[g.june], g[~g.june]
                ax.scatter(ga.day, ga.v, s=18, color=EP_COL.get(ep, "C0"), zorder=3, label=ep)
                if len(gj):
                    ax.scatter(gj.day, gj.v, s=18, facecolor="none", edgecolor=EP_COL.get(ep, "C0"), zorder=3,
                               label="pre (June, reference only)")
            ax.axvline(0.5, color="k", lw=0.5, ls=":")
            _xaxis(ax)
            ax.tick_params(labelsize=6)
            if not any(axes[r, j].get_visible() and axes[r, j].has_data() for r in range(i)):
                ax.set_title(p, fontsize=8)
            ax.set_ylabel(label, fontsize=7) if j == 0 else None
            if i == len(rows) - 1:
                ax.set_xlabel("days from stroke", fontsize=7)
    # jaw steering strip: FAR_DIR (cam4's readout) + CLOSE_DIR (close side spouts, 3-D only, 10-20) and the side biases
    for j in range(len(POS)):
        axes[-1, j].set_visible(False)
    STRIP = (("jaw_far_dir_mm", "jaw FAR_DIR (mm;\n+ toward target)"), ("jaw_close_dir_mm", "jaw CLOSE_DIR (mm;\n+ toward target)"),
             ("jaw_far_bias_mm", "jaw FAR_BIAS (mm;\n+ ipsi)"), ("jaw_close_bias_mm", "jaw CLOSE_BIAS (mm;\n+ ipsi)"))
    for j, (col, lab) in enumerate(STRIP):
        if col not in k:
            continue
        ax = axes[-1, j]
        ax.set_visible(True)
        d = k[["date", "epoch", col]].dropna().assign(day=lambda x: [_x(animal, y) for y in x.date],
                                                      june=lambda x: [str(y) < "20260801" for y in x.date])
        pre = d[(d.epoch == "pre") & (d.date.astype(str) >= pre_from)]
        if len(pre):
            ax.axhspan(pre[col].min(), pre[col].max(), color="0.85", zorder=0)
        for ep, g in d.groupby("epoch"):
            gj, ga = g[g.june], g[~g.june]
            ax.scatter(ga.day, ga[col], s=18, color=EP_COL.get(ep, "C0"), zorder=3, label=ep)
            if len(gj):
                ax.scatter(gj.day, gj[col], s=18, facecolor="none", edgecolor=EP_COL.get(ep, "C0"), zorder=3,
                           label="pre (June, reference only)")
        ax.axvline(0.5, color="k", lw=0.5, ls=":")
        ax.axhline(0, color="0.6", lw=0.5)
        _xaxis(ax)
        ax.set_ylabel(lab, fontsize=7)
        ax.set_xlabel("days from stroke", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="best")
    fig.suptitle(f"{animal}: 3-D tongue / jaw per session (cam1 + cam4; grey = August pre range, dashed = its median)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-from", default="20260801")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    SE = pd.read_csv(d3 / "spout_error" / "per_session.csv", dtype={"date": str})
    K = pd.read_csv(d3 / "kinematics" / "sessions.csv", dtype={"date": str})
    out = d3 / "figures"
    out.mkdir(exist_ok=True)
    for animal in sorted(set(K.animal)):
        f = out / f"summary_{animal}.png"
        figure(animal, SE, K, f, a.pre_from)
        print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
