"""Cue-aligned, trial-averaged 3-D tongue and jaw traces (handoff 10-20 to-do 1b; Priya 10-20 "yes").

    python -m scripts.pose3d_cue_traces [--align cue|first_lick] [--pre-from 20260801] [--jobs 6] [--all]

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

``--align first_lick`` (Priya 10-20 "a separate lick-aligned analysis"): time from each trial's FIRST post-cue cam4 lick
onset (v36 ``licks.csv`` min ``on_ms``), -500 .. 2000 ms, trials with a lick only (the jaw rest stays the PRE-CUE
median). Separates LATER licking (the aligned curve recovers its pre shape) from WEAKER licking (it stays low). Window
mean 0 .. 1000 ms; ``first_lick_ms`` (median per session x position) in window.csv. Output `pose3d/lick_traces/`.

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
ALIGN = {"cue": {"win": WIN_MS, "resp": RESP_MS, "dir": "cue_traces", "x": "ms from cue"},
         "first_lick": {"win": (-500.0, 2000.0), "resp": (0.0, 1000.0), "dir": "lick_traces",
                        "x": "ms from the first post-cue lick onset"}}
MEAS = {"dist": "3-D tongue distance (mm), in-mouth = 0", "dist_out": "3-D tongue distance (mm), tongue out only",
        "out_frac": "share of frames with the tongue out", "down": "downward (mm), in-mouth = 0",
        "side": "sideways (mm, + = ipsi), in-mouth = 0", "jaw_open": "jaw opening from rest (mm)",
        "mid": "aim sideways (deg out of the midline, + = ipsi), tongue out", "elev": "aim vertical (deg below the "
        "horizontal plane), tongue out"}
#: measures defined only on tongue-out frames (a mean over the frames WITH a point): masked where < OUT_MIN are out
CONDITIONAL = ("dist_out", "mid", "elev")
#: below this distance from the mouth zero the tongue's direction is undefined (cam4: ANGLE_MIN_PX 20 px ~ 0.6 mm)
ANGLE_MIN_MM = 0.5
SETS = ("engaged", "licked")


def session_traces(job) -> list[dict]:
    """fan_sessions worker (module level): rows for one session, CACHED (`result_cache`, 10-21) on its 3-D frames, v36
    licks.csv and behaviour trials table (the engagement gate), with the alignment and gate setting in the key."""
    import glob
    import sys

    from wfield_local import orofacial_cohort as oc
    from wfield_local import result_cache
    from wfield_local.paths import PathResolver
    s, d3, v36, align = job
    animal, date = s.split("_")
    beh = sorted(glob.glob(str(Path(PathResolver().root("behavior_out")) / "sessions" / animal / date
                               / f"{animal}_{date}_*_trials.csv")))
    inputs = [Path(d3) / f"{s}_frames.csv.gz", Path(v36) / animal / date / "licks.csv"] + [Path(b) for b in beh]
    ver = result_cache.code_version(sys.modules[__name__], oc, extra=(align, oc.ENGAGEMENT_GATE, oc.ENGAGED_ONLY))
    return result_cache.cached(Path(d3) / "_cache" / f"cue_traces_{align}", s, inputs, ver,
                               lambda: _session_traces(job))


def _session_traces(job) -> list[dict]:
    s, d3, v36, align = job
    from wfield_local import orofacial_cohort as oc
    animal, date = s.split("_")
    sd = Path(v36) / animal / date
    keep, _ = oc.engagement_keep(animal, date, sd)
    S = pd.read_csv(Path(d3) / f"{s}_frames.csv.gz", usecols=["trial_id", "position", "t_ms", "tongue_ml", "tongue_ap",
                                                               "tongue_dv", "jaw_ml", "jaw_ap", "jaw_dv"])
    S = S.drop_duplicates(["trial_id", "t_ms"])
    if keep is not None:
        S = S[S.trial_id.isin(keep)]
    Lk = (pd.read_csv(sd / "licks.csv", usecols=["trial_id", "on_ms"]) if (sd / "licks.csv").exists()
          else pd.DataFrame(columns=["trial_id", "on_ms"]))
    licked = set(Lk.trial_id)
    first = Lk.groupby("trial_id").on_ms.min()
    P = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    has = np.isfinite(P).all(1)
    Pz = np.where(has[:, None], P, 0.0)
    J = S[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    t = S.t_ms.to_numpy(float)                       # cue time: the jaw rest is pre-CUE in both alignments
    rest = (S.assign(_x=J[:, 0], _y=J[:, 1], _z=J[:, 2])[(t < 0) & (t >= -500)]
            .groupby("trial_id")[["_x", "_y", "_z"]].median())
    R = rest.reindex(S.trial_id).to_numpy(float)
    V = {"dist": np.linalg.norm(Pz, axis=1), "dist_out": np.where(has, np.linalg.norm(Pz, axis=1), np.nan),
         "out_frac": has.astype(float), "down": -Pz[:, 2], "side": -Pz[:, 0], "jaw_open": np.linalg.norm(J - R, axis=1)}
    rr = np.linalg.norm(P, axis=1)                      # angles as `pose3d_phase` (mid / elev), tongue-out frames only
    with np.errstate(all="ignore"):
        far = has & (rr >= ANGLE_MIN_MM)
        V["mid"] = np.where(far, np.degrees(np.arcsin(np.clip(-P[:, 0] / rr, -1, 1))), np.nan)
        V["elev"] = np.where(far, np.degrees(np.arcsin(np.clip(-P[:, 2] / rr, -1, 1))), np.nan)
    t1 = first.reindex(S.trial_id).to_numpy(float)
    ta = t - t1 if align == "first_lick" else t     # no lick -> NaN -> outside every bin
    win = ALIGN[align]["win"]
    edges = np.arange(win[0], win[1] + BIN_MS, BIN_MS)
    b = np.where(np.isfinite(ta), np.digitize(ta, edges) - 1, -1)
    inwin = (b >= 0) & (b < len(edges) - 1)
    pos = S.position.to_numpy(str)
    lk = S.trial_id.isin(licked).to_numpy()
    rows = []
    for p in POS:
        for st, sel in (("engaged", np.ones(len(S), bool)), ("licked", lk)):
            m = inwin & (pos == p) & sel
            ntr = int(S.trial_id[m].nunique())
            fl = float(np.median(first.reindex(np.unique(S.trial_id[m])).dropna())) if ntr else np.nan
            for k, v in V.items():
                ok = m & np.isfinite(v)
                ssum = np.bincount(b[ok], weights=v[ok], minlength=len(edges) - 1)
                n = np.bincount(b[ok], minlength=len(edges) - 1)
                with np.errstate(all="ignore"):
                    mu = np.where(n >= MIN_FRAMES, ssum / n, np.nan)
                rows += [{"session": s, "animal": animal, "date": date, "position": p, "set": st, "n_trials": ntr,
                          "first_lick_ms": fl, "measure": k, "t_ms": edges[i] + BIN_MS / 2, "v": mu[i]} for i in range(len(edges) - 1)]
    return rows


#: tongue-out-only measures are a mean over the frames WITH a tongue point: shown only where that session's tongue-out share
#: is >= OUT_MIN (Priya 10-20: pre-cue / late bins with a few frames drew full-height flicker that read as "lots of licks")
OUT_MIN = 0.05


def figure(T: pd.DataFrame, animal: str, m: str, path: Path, pre_from: str, align: str = "cue") -> None:
    """Top row = ``m``; bottom row = share of frames with the tongue out (``out_frac``), so time-out vs reach separate
    (10-20: the licked-only row was dropped -- 98-100 % of engaged trials have a lick outside acute; it stays in the
    tables). Engaged trials (cue-aligned) / engaged trials with a lick (first-lick-aligned). t < 0 shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d_phase import _wrap
    from scripts.pose3d_summary_fig import EP_COL
    A = T[(T.animal == animal) & (T.set == "engaged")]
    A = A[(A.epoch != "pre") | (A.date >= pre_from)]
    of = A[A.measure == "out_frac"].set_index(["session", "position", "t_ms"]).v
    rows = [m] if m == "out_frac" else [m, "out_frac"]
    fig, axes = plt.subplots(len(rows), len(POS), figsize=(2.9 * len(POS), 0.9 + 2.4 * len(rows)), sharex=True,
                             sharey="row", squeeze=False)
    for i, mm in enumerate(rows):
        g = A[A.measure == mm].copy()
        if mm in CONDITIONAL:
            share = of.reindex(pd.MultiIndex.from_frame(g[["session", "position", "t_ms"]])).to_numpy()
            g.loc[~(share >= OUT_MIN), "v"] = np.nan
        for j, p in enumerate(POS):
            ax = axes[i, j]
            ax.axvspan(ALIGN[align]["win"][0], 0, color="0.93", zorder=0, lw=0)
            for ep, col in EP_COL.items():
                ge = g[(g.position == p) & (g.epoch == ep)]
                if ge.empty:
                    continue
                piv = ge.pivot_table(index="session", columns="t_ms", values="v", dropna=False)
                mu = piv.mean()
                se = piv.std(ddof=1) / np.sqrt(piv.notna().sum()) if len(piv) > 1 else mu * np.nan
                lab = f"{ep} (n={len(piv)})"
                if align == "first_lick":
                    lab += f", 1st lick {ge.groupby('session').first_lick_ms.first().median():.0f} ms"
                # Priya 10-20: more transparent so overlapping epochs stay visible
                ax.plot(mu.index, mu, color=col, lw=1.1, alpha=0.65, label=lab)
                ax.fill_between(mu.index, mu - se, mu + se, color=col, alpha=0.1, lw=0)
            ax.axvline(0, color="0.5", lw=0.6, ls=":")
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(p, fontsize=9)
            if j == 0:
                ax.set_ylabel(_wrap(MEAS[mm] + (f" (shown where >= {OUT_MIN:g} of frames are out)" if mm in CONDITIONAL
                                                 else ""), 30), fontsize=7)
            if i == len(rows) - 1:
                ax.set_xlabel(ALIGN[align]["x"], fontsize=7)
    axes[0, -1].legend(fontsize=6, frameon=False)
    if align == "cue":
        head = (f"{animal}: cue-aligned {MEAS[m]} (3-D)\nALL engaged trials (behaviour gate), incl. trials without a "
                "lick; grey = before the cue (3-D windows start 500 ms before it)")
    else:
        head = (f"{animal}: FIRST-LICK-aligned {MEAS[m]} (3-D)\nengaged trials with >= 1 post-cue cam4 lick; t = 0 at "
                "the first lick's onset (cam4 v36); grey = before it; jaw rest = pre-CUE median; legend = median "
                "first-lick time from the cue")
    title = (f"{head}; frames pooled per {BIN_MS:g} ms bin (>= {MIN_FRAMES} frames), mean per session, then mean +- "
             "SEM over sessions; template time; 3-D = both cameras or one camera + neighbour depth; pre = August")
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
    ap.add_argument("--align", choices=list(ALIGN), default="cue")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--all", action="store_true", help="every session with 3-D frames, not just the curated cohort")
    ap.add_argument("--out", type=Path, default=None, help="output folder (checks)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    v36 = Path(rv.root("orofacial")) / V36 / "sessions"
    bad = P3.failed_sessions(d3)
    sess = [f.name.replace("_frames.csv.gz", "") for f in sorted(d3.glob("PS*_frames.csv.gz"))]
    sess = [s for s in sess if s not in bad and (a.all or s in ak.curated_ids())
            and (v36 / s.split("_")[0] / s.split("_")[1] / "licks.csv").exists()]
    res, fails = ak.fan_sessions([(s, str(d3), str(v36), a.align) for s in sess], session_traces, jobs=a.jobs,
                                 key=lambda j: j[0])
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.DataFrame([r for _, rr in res for r in rr])
    T["epoch"] = [epochs.epoch_of(f"{x}_{d[4:]}") for x, d in zip(T.animal, T.date)]
    out = a.out or d3 / ALIGN[a.align]["dir"]
    RESP = ALIGN[a.align]["resp"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "traces.csv.gz", index=False, float_format="%.4g")
    w = T[(T.t_ms >= RESP[0]) & (T.t_ms <= RESP[1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        W = w.groupby(["session", "animal", "date", "epoch", "position", "set", "measure"]).v.mean().unstack("measure")
    W = W.reset_index().merge(T.groupby(["session", "position", "set"]).first_lick_ms.first().reset_index(),
                              on=["session", "position", "set"], how="left")
    W.to_csv(out / "window.csv", index=False, float_format="%.4g")
    Wa = W[(W.epoch != "pre") | (W.date >= a.pre_from)]
    B = Wa.groupby(["animal", "set", "position", "epoch"])[list(MEAS) + ["first_lick_ms"]].mean()
    B["sessions"] = Wa.groupby(["animal", "set", "position", "epoch"]).size()
    pre = B.xs("pre", level="epoch")
    for m in MEAS:
        B[f"d_{m}"] = B[m] - pre[m].reindex(B.index.droplevel("epoch")).to_numpy()
    B.to_csv(out / "by_epoch.csv", float_format="%.4g")
    pd.set_option("display.width", 250)
    order = {"pre": 0, "acute": 1, "subacute": 2, "chronic": 3}
    Bp = B.reset_index()
    Bp["o"] = Bp.epoch.map(order)
    for m in ("dist", "dist_out", "out_frac", "jaw_open", "first_lick_ms"):
        print(f"\n{MEAS.get(m, m)}, mean over {RESP[0]:g} .. {RESP[1]:g} {ALIGN[a.align]['x']} (per animal; pre = "
              "August):")
        print(Bp.sort_values("o").pivot_table(index=["animal", "set", "epoch"], columns="position", values=m, sort=False)
              .reindex(columns=list(POS)).round(3).to_string())
    for animal in sorted(T.animal.unique()):
        for m in MEAS:
            figure(T, animal, m, out / "figures" / f"{'cue' if a.align == 'cue' else 'lick'}_{m}_{animal}.png",
                   a.pre_from, a.align)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
