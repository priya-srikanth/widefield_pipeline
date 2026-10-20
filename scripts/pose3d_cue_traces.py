"""Cue-aligned, trial-averaged 3-D tongue and jaw traces (handoff 10-20 to-do 1b; Priya 10-20 "yes").

    python -m scripts.pose3d_cue_traces [--pre-from 20260801] [--jobs 6] [--all]

WHY: per-lick 3-D measures (`pose3d_kinematics`, `pose3d_phase`) only see licks; PS92's acute far_R licks are as long as
pre in 3-D (5.10 vs 5.14 mm) while cam4's cohort reports "lower amplitude". The difference is the trials WITHOUT licks.
This is the 3-D counterpart of cam4's `orofacial_cohort._cue_traces` (`cohort/tables/.../cue_aligned_traces.csv`): every
ENGAGED trial (`orofacial_cohort.engagement_keep`, the behaviour pipeline's gate), frames pooled per 20 ms bin from the
cue (-500 .. 3000 ms), mean per session x position, then mean +- SEM over sessions per epoch (pre = August, rule 8 /
per animal).

Measures (`t_ms` = template time from the cue, cam4's frame): ``dist`` = 3-D tongue distance from the mouth zero with
frames that have no 3-D tongue point counted as IN THE MOUTH (0 mm; the amplitude a trial-average sees); ``dist_out`` =
tongue-out frames only; ``out_frac`` = share of frames with a 3-D tongue point; ``down`` / ``side`` (+ = mouse LEFT =
ipsi) with in-mouth = 0; ``jaw_open`` = 3-D jaw distance from the trial's resting jaw (pre-cue median, -500 .. 0 ms).
Same, restricted to trials with >= 1 cam4 lick (``licked``), to split "shorter licks" from "fewer licking trials".

Writes `session_poses/pose3d/cue_traces/`: traces.csv.gz (session x position x trial set x measure x bin), window.csv (per
session x position x trial set: mean over cue + 70 .. 1500 ms), by_epoch.csv (per animal x epoch x position, the window
mean; change vs August pre), figures/cue_<measure>_<animal>.png.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
BIN_MS, WIN_MS, MIN_FRAMES = 20.0, (-500.0, 3000.0), 25         # as `orofacial_cohort` CUE_BIN_MS / CUE_WIN_MS / MIN
RESP_MS = (70.0, 1500.0)                                        # window mean (cue + 70 ms: cam4's lick start)
MEAS = {"dist": "3-D tongue distance (mm), in-mouth = 0", "dist_out": "3-D tongue distance (mm), tongue out only",
        "out_frac": "share of frames with the tongue out", "down": "downward (mm), in-mouth = 0",
        "side": "sideways (mm, + = ipsi), in-mouth = 0", "jaw_open": "jaw opening from rest (mm)"}
SETS = ("engaged", "licked")


def session_traces(job) -> list[dict]:
    """fan_sessions worker (module level): rows for one session."""
    s, d3, v36 = job
    from wfield_local import orofacial_cohort as oc
    animal, date = s.split("_")
    sd = Path(v36) / animal / date
    keep, _ = oc.engagement_keep(animal, date, sd)
    S = pd.read_csv(Path(d3) / f"{s}_frames.csv.gz", usecols=["trial_id", "position", "t_ms", "tongue_ml", "tongue_ap",
                                                               "tongue_dv", "jaw_ml", "jaw_ap", "jaw_dv"])
    S = S.drop_duplicates(["trial_id", "t_ms"])
    if keep is not None:
        S = S[S.trial_id.isin(keep)]
    licked = set(pd.read_csv(sd / "licks.csv", usecols=["trial_id"]).trial_id) if (sd / "licks.csv").exists() else set()
    P = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    has = np.isfinite(P).all(1)
    Pz = np.where(has[:, None], P, 0.0)
    J = S[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    t = S.t_ms.to_numpy(float)
    rest = (S.assign(_x=J[:, 0], _y=J[:, 1], _z=J[:, 2])[(t < 0) & (t >= -500)]
            .groupby("trial_id")[["_x", "_y", "_z"]].median())
    R = rest.reindex(S.trial_id).to_numpy(float)
    V = {"dist": np.linalg.norm(Pz, axis=1), "dist_out": np.where(has, np.linalg.norm(Pz, axis=1), np.nan),
         "out_frac": has.astype(float), "down": -Pz[:, 2], "side": -Pz[:, 0], "jaw_open": np.linalg.norm(J - R, axis=1)}
    edges = np.arange(WIN_MS[0], WIN_MS[1] + BIN_MS, BIN_MS)
    b = np.digitize(t, edges) - 1
    inwin = (b >= 0) & (b < len(edges) - 1)
    pos = S.position.to_numpy(str)
    lk = S.trial_id.isin(licked).to_numpy()
    rows = []
    for p in POS:
        for st, sel in (("engaged", np.ones(len(S), bool)), ("licked", lk)):
            m = inwin & (pos == p) & sel
            ntr = int(S.trial_id[m].nunique())
            for k, v in V.items():
                ok = m & np.isfinite(v)
                ssum = np.bincount(b[ok], weights=v[ok], minlength=len(edges) - 1)
                n = np.bincount(b[ok], minlength=len(edges) - 1)
                with np.errstate(all="ignore"):
                    mu = np.where(n >= MIN_FRAMES, ssum / n, np.nan)
                rows += [{"session": s, "animal": animal, "date": date, "position": p, "set": st, "n_trials": ntr,
                          "measure": k, "t_ms": edges[i] + BIN_MS / 2, "v": mu[i]} for i in range(len(edges) - 1)]
    return rows


def figure(T: pd.DataFrame, animal: str, m: str, path: Path, pre_from: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d_phase import _wrap
    from scripts.pose3d_summary_fig import EP_COL
    g = T[(T.animal == animal) & (T.measure == m)]
    g = g[(g.epoch != "pre") | (g.date >= pre_from)]
    fig, axes = plt.subplots(2, len(POS), figsize=(2.9 * len(POS), 5.6), sharey=True, sharex=True, squeeze=False)
    for i, st in enumerate(SETS):
        for j, p in enumerate(POS):
            ax = axes[i, j]
            for ep, col in EP_COL.items():
                ge = g[(g.set == st) & (g.position == p) & (g.epoch == ep)]
                if ge.empty:
                    continue
                piv = ge.pivot_table(index="session", columns="t_ms", values="v")
                mu = piv.mean()
                se = piv.std(ddof=1) / np.sqrt(len(piv)) if len(piv) > 1 else mu * np.nan
                ax.plot(mu.index, mu, color=col, lw=1.2, label=f"{ep} (n={len(piv)})")
                ax.fill_between(mu.index, mu - se, mu + se, color=col, alpha=0.2, lw=0)
            ax.axvline(0, color="0.6", lw=0.6, ls=":")
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(p, fontsize=9)
            if j == 0:
                ax.set_ylabel(_wrap(f"{'ALL engaged trials' if st == 'engaged' else 'trials with >= 1 cam4 lick'}\n"
                                    f"{MEAS[m]}", 34), fontsize=7)
            if i == 1:
                ax.set_xlabel("ms from cue", fontsize=7)
    axes[0, -1].legend(fontsize=6, frameon=False)
    title = (f"{animal}: cue-aligned {MEAS[m]} (3-D)\nengaged trials (behaviour gate) incl. trials WITHOUT licks (top) vs "
             f"licked trials only (bottom); frames pooled per {BIN_MS:g} ms bin (>= {MIN_FRAMES} frames), mean per "
             "session, then mean +- SEM over sessions; template time; 3-D = both cameras or one camera + neighbour "
             "depth; pre = August")
    fig.suptitle(_wrap(title, 150), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import pose3d as P3
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-from", default="20260801")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--all", action="store_true", help="every session with 3-D frames, not just the curated cohort")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    v36 = Path(rv.root("orofacial")) / V36 / "sessions"
    bad = P3.failed_sessions(d3)
    sess = [f.name.replace("_frames.csv.gz", "") for f in sorted(d3.glob("PS*_frames.csv.gz"))]
    sess = [s for s in sess if s not in bad and (a.all or s in ak.curated_ids())
            and (v36 / s.split("_")[0] / s.split("_")[1] / "licks.csv").exists()]
    res, fails = ak.fan_sessions([(s, str(d3), str(v36)) for s in sess], session_traces, jobs=a.jobs,
                                 key=lambda j: j[0])
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.DataFrame([r for _, rr in res for r in rr])
    T["epoch"] = [epochs.epoch_of(f"{x}_{d[4:]}") for x, d in zip(T.animal, T.date)]
    out = d3 / "cue_traces"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "traces.csv.gz", index=False, float_format="%.4g")
    w = T[(T.t_ms >= RESP_MS[0]) & (T.t_ms <= RESP_MS[1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        W = w.groupby(["session", "animal", "date", "epoch", "position", "set", "measure"]).v.mean().unstack("measure")
    W = W.reset_index()
    W.to_csv(out / "window.csv", index=False, float_format="%.4g")
    Wa = W[(W.epoch != "pre") | (W.date >= a.pre_from)]
    B = Wa.groupby(["animal", "set", "position", "epoch"])[list(MEAS)].mean()
    B["sessions"] = Wa.groupby(["animal", "set", "position", "epoch"]).size()
    pre = B.xs("pre", level="epoch")
    for m in MEAS:
        B[f"d_{m}"] = B[m] - pre[m].reindex(B.index.droplevel("epoch")).to_numpy()
    B.to_csv(out / "by_epoch.csv", float_format="%.4g")
    pd.set_option("display.width", 250)
    order = {"pre": 0, "acute": 1, "subacute": 2, "chronic": 3}
    Bp = B.reset_index()
    Bp["o"] = Bp.epoch.map(order)
    for m in ("dist", "dist_out", "out_frac", "jaw_open"):
        print(f"\n{MEAS[m]}, mean over cue + {RESP_MS[0]:g} .. {RESP_MS[1]:g} ms (per animal; pre = August):")
        print(Bp.sort_values("o").pivot_table(index=["animal", "set", "epoch"], columns="position", values=m, sort=False)
              .reindex(columns=list(POS)).round(3).to_string())
    for animal in sorted(T.animal.unique()):
        for m in MEAS:
            figure(T, animal, m, out / "figures" / f"cue_{m}_{animal}.png", a.pre_from)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
