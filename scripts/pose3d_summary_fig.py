"""3-D tongue / jaw readouts across sessions, per animal (Priya 10-19: "where are the 3-D kinematics plots").

    python -m scripts.pose3d_summary_fig [--pre-from 20260801]      # after scripts.pose3d_kinematics + scripts.pose3d_spout_error

One figure per animal, `session_poses/pose3d/figures/summary_<animal>.png`: x = days from the stroke (config stroke_date; the
lesion followed that day's session, so day 1 = the first post session), one dot per session (median over its licks / trials),
an OPEN marker, shape = animal, color = epoch (black / pink / yellow / green; June pre gray; A30); columns = the six positions; gray band = the animal's AUGUST pre range (min-max over sessions), dashed
line = its median. Rows, in the order agreed 10-19 (distance offsets primary, one angle secondary):
  1. closest approach to the target spout (mm)                    -- spout error 1, origin-free
  2. lateral error at closest approach (mm; + = outward, center + = mouse left)
  3-5. endpoint vs the animal's OWN pre endpoint: lateral / out (+ = beyond) / up (mm)   -- spout error 3
  6-8. DIRECTION from the mouth zero vs the direction to the target spout: the 3-D angle error; the angle OUT OF THE
     MIDLINE PLANE (the sagittal plane through the mouth zero and the close / far center spouts = ml 0; asin(ml / |r|),
     + = toward ipsi; Priya 10-19) and the forward tilt WITHIN it (atan2(out, down)). Consistent across the post-stroke
     set-point move (Priya 10-19: the spouts are commanded relative to the set point, which marks the mouth that day).
     Non-zero before the stroke (licks run steeper than the line to the spout tips): read against the pre band.
  9. reach from the mouth zero (mm)   10. peak 3-D tongue speed (mm/s)   11. widest jaw opening (mm)
plus a final panel row-strip: jaw FAR_DIR (mm; + = toward the target side) per session.

LICK / TRIAL GATE (Priya 11-06, A20): every row is over CAMERA-WORKING-PERIOD trials only (`trial_classes.gate_kin_licks`:
first cue to the session's last camera lick, stopped runs (A2 + excise) out -- the definition of Fig 0.2 / 1b). The
spout-error rows come from `pose3d_spout_error`'s gated per_session.csv; the kinematics rows are re-made from the gated
kinematics licks / jaw trials (`gated_sessions`, the per-session medians of `pose3d_kinematics._session_result`). A
session with NO camera classes (no video) has no working period and contributes NOTHING (as gain_bio). Before 11-06:
every post-cue lick / every 3-D jaw trial, no gate -- ``WFIELD_KIN_GATE=legacy`` (reads kinematics/sessions.csv and
spout_error_legacygate/, writes figures_legacygate/).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
from pathlib import Path

import pandas as pd
from wfield_local.pose3d import pose3d_dir

POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
#: epoch colours = THE session-dot palette (orofacial_cohort.EPOCH_LINE: pre black / acute pink / subacute yellow /
#: chronic green; Priya 11-06 A30, as mixed_models.plot_fits)
from wfield_local.orofacial_cohort import EPOCH_LINE as EP_COL  # noqa: E402
JUNE_EDGE = "0.6"        # June pre sessions (reference only): grey open markers, same shape


def session_dots(ax, day, v, animal: str, epoch: str, june: bool = False):
    """THE session dots (Priya 11-06, A30; results-page convention): OPEN markers, shape = animal
    (`cat_points.animal_marker`, animals.yaml), color = epoch (`EP_COL` = orofacial_cohort.EPOCH_LINE); June pre
    (reference only) in gray."""
    from wfield_local.cat_points import animal_marker
    return ax.scatter(day, v, s=20, marker=animal_marker(animal), facecolors="none",
                      edgecolors=JUNE_EDGE if june else EP_COL.get(epoch, "0.4"), linewidths=0.9, zorder=3,
                      label="pre (June, reference only)" if june else epoch)
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


def session_row(L: pd.DataFrame, J: pd.DataFrame) -> dict:
    """The per-session medians of ONE session's licks ``L`` and jaw trials ``J`` -- the same quantities and names as
    `pose3d_kinematics._session_result` (its lines from ``jm = J.groupby...`` to ``open_max_mm``; mirrored here, not
    edited there, because any edit to pose3d_kinematics invalidates every cached kinematics session -- fold the two into
    one helper at the next kinematics re-run)."""
    from scripts.pose3d_kinematics import LICK_MEASURES
    row = {}
    jm = J.groupby("position").lr_wide_mm.median() if len(J) else pd.Series(dtype=float)
    if {"far_L", "far_R"} <= set(jm.index):
        row["jaw_far_dir_mm"] = (jm["far_L"] - jm["far_R"]) / 2       # + = toward the target side
        row["jaw_far_bias_mm"] = (jm["far_L"] + jm["far_R"]) / 2      # + = ipsi (mouse left)
    if {"close_L", "close_R"} <= set(jm.index):
        row["jaw_close_dir_mm"] = (jm["close_L"] - jm["close_R"]) / 2
        row["jaw_close_bias_mm"] = (jm["close_L"] + jm["close_R"]) / 2
    for p, g in L.groupby("position"):
        for m in LICK_MEASURES:
            if m in g:
                row[f"{m}|{p}"] = float(g[m].median())
    for p, g in J.groupby("position"):
        row[f"open_max_mm|{p}"] = float(g.open_max_mm.median())
    return row


def gated_sessions(L: pd.DataFrame, J: pd.DataFrame) -> pd.DataFrame:
    """kinematics/sessions.csv re-made over the camera working period: ``L`` / ``J`` = the kinematics licks / jaw trials
    (session, animal, date, epoch, position, trial_id, ...) through `trial_classes.gate_kin_licks`, then `session_row`
    per session. A session with no camera classes has no rows left, so it is absent."""
    from wfield_local import trial_classes as tc
    L, J = tc.gate_kin_licks(L), tc.gate_kin_licks(J)
    rows = []
    for s in sorted(set(L.session) | set(J.session)):
        ls, js = L[L.session == s], J[J.session == s]
        first = (ls if len(ls) else js).iloc[0]
        rows.append({"session": s, "animal": first.animal, "date": str(first.date), "epoch": first.epoch,
                     **session_row(ls, js)})
    return pd.DataFrame(rows)


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
    ax.spines["top"].set_visible(False)                  # Priya 11-03: no top frame line


def _days(animal: str, date: str) -> int:
    from wfield_local import config
    sd = pd.Timestamp(config._to8(config.stroke_date(animal)))      # config gives MMDD; _to8 adds the cohort year
    return int((pd.Timestamp(str(date)) - sd).days)


#: ``contra`` (Priya 10-30, goal-page sign): lesion-signed rows negated at plot level; the outward-signed lateral rows
#: are + = mouse LEFT at the CENTRE spouts, so they are negated there only
CONTRA_K = ("midline_err_ipsi", "jaw_far_bias_mm", "jaw_close_bias_mm")
CONTRA_SE_CENTRE = ("lat_med", "habit_lat_med")


def _to_contra(SE: pd.DataFrame, K: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    K = K.copy()
    for c in K.columns:
        if c in CONTRA_K or c.split("|")[0] in CONTRA_K:
            K[c] = -K[c]
    SE = SE.copy()
    cen = SE.position.astype(str).str.contains("center")
    for c in CONTRA_SE_CENTRE:
        if c in SE:
            SE.loc[cen, c] = -SE.loc[cen, c]
    return SE, K


def figure(animal: str, SE: pd.DataFrame, K: pd.DataFrame, out: Path, pre_from: str, contra: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if contra:
        SE, K = _to_contra(SE, K)
    se = SE[(SE.animal == animal) & (SE.subset == "all")]
    k = K[K.animal == animal]
    rel = (lambda s: s.replace("+ ipsi", "+ contra").replace("+ outward)", "+ outward;\ncentre + = contra)")  # noqa: E731
           if contra else s)

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
                session_dots(ax, ga.day, ga.v, animal, ep)
                if len(gj):
                    session_dots(ax, gj.day, gj.v, animal, ep, june=True)
            ax.axvline(0.5, color="k", lw=0.5, ls=":")
            _xaxis(ax)
            ax.tick_params(labelsize=6)
            if not any(axes[r, j].get_visible() and axes[r, j].has_data() for r in range(i)):
                ax.set_title(p, fontsize=8)
            ax.set_ylabel(rel(label), fontsize=7) if j == 0 else None
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
            session_dots(ax, ga.day, ga[col], animal, ep)
            if len(gj):
                session_dots(ax, gj.day, gj[col], animal, ep, june=True)
        ax.axvline(0.5, color="k", lw=0.5, ls=":")
        ax.axhline(0, color="0.6", lw=0.5)
        _xaxis(ax)
        ax.set_ylabel(rel(lab), fontsize=7)
        ax.set_xlabel("days from stroke", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="best")
    from wfield_local import trial_classes as tc
    fig.suptitle(f"{animal}: 3-D tongue / jaw per session (cam1 + cam4; gray = August pre range, dashed = its median)"
                 + ("; camera-working-period trials (first cue -> last camera lick, stopped runs out)"
                    if tc.kin_gate() == "working" else "; every post-cue lick / jaw trial (no gate, legacy)")
                 + ("\nSIGN: lesion-signed rows + = CONTRALESIONAL (mouse right), flipped at plot level; center-spout "
                    "lateral errors negated (outward rows otherwise unchanged)" if contra else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_png_svg(fig, out, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-from", default="20260801")
    ap.add_argument("--contra", action="store_true",
                    help="(default since 11-03, kept for old command lines) also draw summary_<animal>_contra.png")
    ap.add_argument("--no-contra", action="store_true", help="skip summary_<animal>_contra.png (+ = contralesional)")
    ap.add_argument("--contra-only", action="store_true", help="only the _contra figures")
    a = ap.parse_args(argv)
    from wfield_local import trial_classes as tc
    rv = PathResolver()
    d3 = pose3d_dir(rv)
    SE = pd.read_csv(tc.kin_out_dir(d3, "spout_error") / "per_session.csv", dtype={"date": str})
    if tc.kin_gate() == "working":       # A20: the kinematics rows over camera-working-period licks / jaw trials
        K = gated_sessions(pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str}),
                           pd.read_csv(d3 / "kinematics" / "jaw_trials.csv.gz", dtype={"date": str}))
    else:
        K = pd.read_csv(d3 / "kinematics" / "sessions.csv", dtype={"date": str})
    out = tc.kin_out_dir(d3, "figures", "pose3d_summary_fig")
    out.mkdir(exist_ok=True)
    for animal in sorted(set(K.animal)):
        if not a.contra_only:
            f = out / f"summary_{animal}.png"
            figure(animal, SE, K, f, a.pre_from)
            print(f"-> {f}")
        if not a.no_contra or a.contra_only:            # Priya 11-03: + = contralesional is the DEFAULT output
            f = out / f"summary_{animal}_contra.png"
            figure(animal, SE, K, f, a.pre_from, contra=True)
            print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
