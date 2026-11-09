"""ONE summary figure for the hidden deck slide "chronic contralesional changes exceed a 2-month pre-stroke drift floor;
chronic jaw changes do not" (Priya 11-09: yes to the slide). June (06/06-06/08) -> August pre as the floor, in three
panels with their own units:
  FR asymmetry    (contra - ipsi) / (contra + ipsi) mesh-point speed, cue / lick windows    `face_rhythm_june_drift.py`
  FR composition  factor log2 peak minus the session's mean over 12 factors (scale-free)     `face_rhythm_june_drift.py`
  3-D kinematics  mm, A20 working-period licks / jaw trials                                  `pose3d_june_reference.py`
Row: gray bar = June's 95 % interval vs own August pre; acute / chronic dots = change vs own August pre (FR: boot_delta
interval; 3-D: the table holds only deltas vs the August mean, so the interval is the animal -> session bootstrap of
the deltas, without August's own session variance). Right-hand band: "#" = the epoch differs from June by boot_delta
(interval excludes 0, same sign as its change), "##" = also Holm (FR: the drift script's families; 3-D: 6 positions
per measure). Marks sit in a band outside the data (no marks on points).

    python scripts/june_drift_summary_fig.py
-> ``<dlc>/face_rhythm/cam4/readouts/master_pre_v1/june_reference/june_drift_summary.png`` + data/june_drift_3d.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

POS = {"close_L": "close ipsi", "close_center": "close mid", "close_R": "close contra", "far_L": "far ipsi",
       "far_center": "far mid", "far_R": "far contra"}
ASYM_ROWS = [("cue", "close_R"), ("cue", "far_R"), ("lick", "close_R"), ("lick", "far_R"), ("lick", "close_L")]
COMP_ROWS = [("contra", 4), ("centre", 4), ("centre", 7), ("contra", 7), ("contra", 6), ("centre", 6), ("ipsi", 0)]
D3_ROWS = [("out_ap", "close_R", "endpoint out"), ("out_ap", "far_R", "endpoint out"),
           ("lat_med", "close_R", "lateral error"), ("lat_med", "far_R", "lateral error"),
           ("open_max_mm", "close_L", "jaw opening"), ("open_max_mm", "far_R", "jaw opening")]
SIDE = {"ipsi": "ipsi spouts", "centre": "center spouts", "contra": "contra spouts"}


def three_d(n_boot: int = 10000) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    f = Path(rv.root("dlc")) / "session_poses" / "pose3d" / "june_reference" / "sessions_delta.csv"
    S = pd.read_csv(f).sort_values(["measure", "position", "animal", "session"], kind="stable")
    rng = np.random.default_rng(0)
    rows = []
    for (m, p), d in S.groupby(["measure", "position"], sort=True):
        by = {g: {a: list(x.delta) for a, x in d[d.group == g].groupby("animal")} for g in ("june", "acute", "chronic")}
        zero = {a: [0.0] for a in by["june"]}
        for ep in ("june", "acute", "chronic"):
            iv = ak.boot_delta(by[ep], zero, rng, n_boot)
            r = {"measure": m, "position": p, "epoch": ep, "delta": iv.point, "ci_lo": iv.lo, "ci_hi": iv.hi}
            if ep != "june":
                fv, dr = ak.boot_delta(by[ep], by["june"], rng, n_boot, draws=True)
                r.update(floor_delta=fv.point, floor_lo=fv.lo, floor_hi=fv.hi, floor_p=ak.boot_p(dr))
            rows.append(r)
    T = pd.DataFrame(rows)
    T["floor_p_holm"] = np.nan
    m = T.epoch != "june"
    for _, idx in T[m].groupby(["measure", "epoch"]).groups.items():
        T.loc[idx, "floor_p_holm"] = ak.holm(T.loc[idx, "floor_p"].to_numpy(float))
    excl = (T.floor_lo > 0) | (T.floor_hi < 0)
    T["beyond_floor"] = m & excl & (np.sign(T.floor_delta) == np.sign(T.delta))
    T["beyond_floor_holm"] = T.beyond_floor & (T.floor_p_holm < 0.05)
    return T


def panel(ax, rows: list[tuple[str, dict]], xlabel: str, title: str, ec: dict) -> None:
    for i, (_lab, cell) in enumerate(rows):
        j = cell["june"]
        ax.barh(i, j.ci_hi - j.ci_lo, left=j.ci_lo, height=0.7, color="0.85", lw=0, zorder=0)
        ax.plot(j.delta, i, "o", ms=4, mfc="white", mec="0.45", zorder=2)
        marks = []
        for k, ep in enumerate(("acute", "chronic")):
            r = cell[ep]
            y = i + (k - 0.5) * 0.3
            ax.errorbar(r.delta, y, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o", ms=4, lw=1,
                        color=ec[ep], zorder=3)
            marks.append(("##" if r.beyond_floor_holm else "#" if r.beyond_floor else "", ec[ep], y))
        for mk, c, y in marks:
            if mk:
                ax.text(1.02, y, mk, transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=c,
                        clip_on=False)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_figures as frf
    from wfield_local.paths import PathResolver
    out = frc.readouts_dir(fro.SET, PathResolver()) / "june_reference"
    A = pd.read_csv(out / "data" / "june_drift_asymmetry.csv")
    C = pd.read_csv(out / "data" / "june_drift_composition.csv")
    D = three_d()
    D.to_csv(out / "data" / "june_drift_3d.csv", index=False, float_format="%.4g")
    names = C.groupby("factor").name.first().to_dict()

    def cells(T, sel):
        return {ep: T[sel & (T.epoch == ep)].iloc[0] for ep in ("june", "acute", "chronic")}

    ra = [(f"{al}: {POS[p]}", cells(A, (A["align"] == al) & (A.position == p))) for al, p in ASYM_ROWS]
    rc = [(f"{names[k]}: {SIDE[g]}", cells(C, (C.group == g) & (C.factor == k))) for g, k in COMP_ROWS]
    rd = [(f"{lab}: {POS[p]}", cells(D, (D.measure == m) & (D.position == p))) for m, p, lab in D3_ROWS]
    ec = {**frc.epoch_colors()}
    with frf.style():
        fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.75})
        panel(axs[0], ra, "asymmetry change vs own Aug pre", "Face movement asymmetry (FR)", ec)
        panel(axs[1], rc, "log2 composition change vs own Aug pre", "Factor composition (FR)", ec)
        panel(axs[2], rd, "mm vs own Aug pre", "3-D tongue / jaw", ec)
        h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.45", label="June (gray bar = 95 % interval)"),
             plt.Line2D([], [], marker="o", ls="", color=ec["acute"], label="acute"),
             plt.Line2D([], [], marker="o", ls="", color=ec["chronic"], label="chronic")]
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
        fig.subplots_adjust(top=0.84, bottom=0.14, left=0.12, right=0.97)
        frf.save(fig, out / "june_drift_summary.png", """
June (06/06-06/08, 3 sessions per animal) -> each animal's own August pre = a ~2-month pre-stroke floor (slow drift +
training stage + camera set-up). Gray bar = June's 95 % interval, open dot = June; filled = acute / chronic change vs own
August pre. # = the epoch differs from June (paired animals -> sessions bootstrap, 10,000 draws; interval excludes 0,
same sign as its change); ## = also Holm. Left: FaceRhythm contra / ipsi mesh-point speed asymmetry, cue 0..+2 s and LP
lick -0.1..+0.2 s windows, engaged trials (Holm over 6 positions). Middle: FaceRhythm factor composition (log2 peak
minus the session's mean over 12 factors of the frozen August-pre set; scale-free), engaged trials (Holm over 12
factors). Right: 3-D kinematics, A20 working-period camera licks / jaw trials (Holm over 6 positions per measure;
interval = bootstrap of the deltas vs the August mean). Contra = right spouts (lesions left).""")
    print("->", out / "june_drift_summary.png")
    print(D[(D.epoch != "june")].merge(pd.DataFrame(D3_ROWS, columns=["measure", "position", "lab"]))
          [["lab", "position", "epoch", "delta", "floor_delta", "beyond_floor", "beyond_floor_holm"]].round(3)
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
