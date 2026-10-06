"""Threshold-retune distributions + the three flagged checks (angle sign, jaw QC by position, velocity window),
both models, on a cue clip or a session-level trial clip. Prints a text report (and writes it next to the data).

    python -m scripts.kinematics_checks                         # PS93 0908 cue clip (lp_vs_dlc_cue_traces)
    python -m scripts.kinematics_checks --session PS93:20260814 # scripts.session_poses output (windows_*)

Priya, 2026-10-01: "proceed with 1-2" (retune px thresholds stage by stage on our data; then angle / jaw /
velocity). Every number is in MOUTH-relative px (tongue) as the pipeline sees it. Sections:
  RETUNE   lick peak y, within-bout troughs, confident tongue y / x, frame-to-frame |dy| |dx|, max dy/dt, and
           what the pre-clean and gates actually removed -- set against the old-rig thresholds' stated meaning
           (stroke_orofacial DECISIONS 9-13: troughs ~30-80, lick peaks ~140-200, frame jitter +-10-30 px).
  ANGLE    sign(angle_max_signed_lick1) vs sign(lick-1 angle at the peak); how far from the mouth the
           max-|angle| frame is (the origin is the mouth, so frames near the lips give short, unstable vectors).
  JAW      jaw_pass_qc by position, threshold branch ("none" = degenerate baseline = cannot judge), jaw
           confidence pre-cue.
  VELOCITY per-lick max dy/dt: the pipeline window (configs velocity_window; theirs "v7" = lmax peak +-16 ms) vs the whole visible
           rise; LP/DLC agreement for licks both models kept.
"""
from __future__ import annotations

import argparse
import collections
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import trial_windows as TW

PCT = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def q(a, ps=PCT) -> str:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return " ".join(f"p{p}={v:.0f}" for p, v in zip(ps, np.percentile(a, ps))) + f"  (n={len(a)})" if len(a) else "n=0"


def full_rise_velocity(r, k, fps: float) -> float:
    """max dy/dt over the whole visible rise: back from the peak while y keeps falling (2 px slack) and stays
    visible (not baseline fill), then forward to the peak."""
    y = np.where(r.is_baseline_fill_clean, np.nan, r.y_clean.astype(float))
    f = int(np.argmin(np.abs(r.t_ms - k["t_ms"])))
    lo = f
    while lo > 0 and np.isfinite(y[lo - 1]) and y[lo - 1] <= y[lo] + 2:
        lo -= 1
    return float(np.nanmax(np.gradient(y[lo:f + 1]) * fps)) if f - lo >= 2 else float("nan")


def report(model: str, T, J, clean, raw: pd.DataFrame, idx: pd.DataFrame, frame) -> pd.DataFrame:
    tg, pm, _ = clean["tongue"]
    fps = tg.fps
    lk_thr = oc.params("tongue")["lk_thr"]
    peaks, troughs, n_vanish, rows = [], [], 0, []
    labels, removed = collections.Counter(), collections.Counter()
    for r in T.trial_results:
        y = np.where(r.is_baseline_fill_clean, np.nan, r.y_clean)
        kl = r.kept_licks
        for a, b in zip(kl, kl[1:]):
            if b["t_ms"] - a["t_ms"] <= 300:
                fa, fb = (int(np.argmin(np.abs(r.t_ms - z["t_ms"]))) for z in (a, b))
                seg = y[fa:fb + 1]
                if np.isfinite(seg).all():
                    troughs.append(np.min(seg))
                else:
                    n_vanish += 1
        for k in kl:
            peaks.append(k["y"])
            rows.append({"model": model, "trial_id": r.trial_id, "position": r.position, "t_ms": k["t_ms"],
                         "y": k["y"], "x": k["x"], "source": k["source"],
                         "win_ms": (k["fall_end_frame"] - k["rise_start_frame"]) * 1000.0 / fps,
                         "v_v7": k["max_velocity_y_px_per_s"], "v_full_rise": full_rise_velocity(r, k, fps)})
        d = r.preclean_diag
        for cl in d.clusters:
            labels[cl.label.split(" ")[0]] += 1
        for key in ("n_artifact_frames", "n_frame_outliers", "n_pchip_extended", "n_interp_wiped",
                    "n_x_hard_outliers", "n_x_joint_outliers"):
            removed[key] += getattr(d, key)
    ok = tg.lk[pm] >= lk_thr
    yr, xr = tg.y_raw[pm] - tg.Y0, tg.x_raw[pm] - tg.X0
    okp = ok[1:] & ok[:-1]
    gates = collections.Counter(x["gate"] for r in T.trial_results for x in r.decisions if not x["keep"])
    L = pd.DataFrame(rows)
    print(f"\n===== {model}: {len(L)} kept licks, {len(T.trial_results)} trials, cleaning cutoff {lk_thr}")
    print("RETUNE (mouth-relative px)")
    print("  lick peak y                 ", q(peaks))
    print("  within-bout trough y        ", q(troughs), f"; tongue vanished between {n_vanish} pairs")
    print("  confident tongue y          ", q(yr[ok]))
    print("  confident tongue x          ", q(xr[ok]))
    print("  |dy| frame-to-frame         ", q(np.abs(np.diff(yr))[okp]))
    print("  |dx| frame-to-frame         ", q(np.abs(np.diff(xr))[okp]))
    print("  pre-clean clusters          ", dict(labels), "| removed:", dict(removed))
    print("  gate rejections             ", dict(gates))
    print("ANGLE")
    pt = T.per_trial
    a1, am = pt.lick1_angle_at_ypeak.to_numpy(), pt.angle_max_signed_lick1.to_numpy()
    both = np.isfinite(a1) & np.isfinite(am)
    print(f"  sign(angle_max_signed_lick1) != sign(lick-1 angle at peak): {int((np.sign(a1[both]) != np.sign(am[both])).sum())}"
          f" / {int(both.sum())} trials; at peak {q(a1, (5, 50, 95))}; max-signed {q(am, (5, 50, 95))}")
    ang = T.angle_per_frame
    if ang is not None:
        xa, ya = tg.x_final + tg.X0 - frame.origin[0], tg.y_final + tg.Y0 - frame.origin[1]
        pad = int(round(float(T.params["detector"]["peak_pad_ms"]) / 1000.0 * fps))
        dm, dp = [], []
        for r in T.trial_results:
            for k in r.kept_licks[:5]:
                c = int(round(r.cue_frame + k["t_ms"] / 1000.0 * fps))
                w = ang[max(0, c - pad):c + pad + 1]
                if np.isfinite(w).any():
                    j = max(0, c - pad) + int(np.nanargmax(np.abs(w)))
                    dm.append(np.hypot(xa[j], ya[j]))
                    dp.append(np.hypot(xa[c], ya[c]))
        print(f"  [theirs: unrestricted +-52 ms] tongue distance from mouth at the max-|angle| frame {q(dm, (10, 50, 90))}; "
              f"at the peak {q(dp, (10, 50, 90))}")
    print("JAW")
    t = idx.t_ms.to_numpy()
    pre = pd.Series(raw.jaw_likelihood.to_numpy()[t < 0] >= lk_thr).groupby(idx.position.to_numpy()[t < 0]).mean()
    g = J.groupby("position").agg(n=("jaw_pass_qc", "size"), passed=("jaw_pass_qc", "sum"),
                                  degenerate=("jaw_thresh_px_used", lambda s: int((s == "none").sum())),
                                  peak_defl=("jaw_peak_abs_deflection", "median"))
    g["jaw_conf_precue_%"] = (pre * 100).round(0)
    print("  " + g.round(1).to_string().replace("\n", "\n  "))
    print("VELOCITY (max dy/dt, px/s)")
    print("  " + L.groupby("source").agg(n=("v_v7", "size"), window_ms=("win_ms", "median"), v7=("v_v7", "median"),
                                         full_rise=("v_full_rise", "median")).round(0).to_string().replace("\n", "\n  "))
    return L


def agreement(L: pd.DataFrame) -> None:
    a, b = L[L.model == "DLC"], L[L.model == "LP"]
    m = []
    for r in a.itertuples():
        g = b[(b.trial_id == r.trial_id) & (np.abs(b.t_ms - r.t_ms) <= 8)]
        if len(g):
            m.append((r.v_v7, g.v_v7.iloc[0], r.v_full_rise, g.v_full_rise.iloc[0], r.source == g.source.iloc[0]))
    m = pd.DataFrame(m, columns=["d7", "l7", "dF", "lF", "same_src"])
    print(f"\n===== VELOCITY AGREEMENT, licks kept by both models (same detector in both: {m.same_src.mean():.0%})")
    for c1, c2, lab in (("d7", "l7", "pipeline window (config)"), ("dF", "lF", "whole visible rise")):
        ok = np.isfinite(m[c1]) & np.isfinite(m[c2]) & (m[c1] > 0) & (m[c2] > 0)
        r = m[c2][ok] / m[c1][ok]
        print(f"  {lab:18s}: n={int(ok.sum())}, LP/DLC p10/50/90 {np.percentile(r, [10, 50, 90]).round(2)}, "
              f">2x apart {np.mean((r > 2) | (r < 0.5)):.0%}, r={np.corrcoef(m[c1][ok], m[c2][ok])[0, 1]:.2f}")


def main(argv=None) -> int:
    import scripts.pose_kinematics_demo as D
    from wfield_local import dlc_project
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="animal:date[:tag] (session_poses folder)")
    ap.add_argument("--lp", type=Path, default=Path.home() / "lp_cue_tmp" / "cue_windows_LP.csv")
    a = ap.parse_args(argv)
    rv = PathResolver()
    if a.session:
        from scripts.session_poses import read_index, session_dir
        animal, date, d = session_dir(rv, a.session)
        idx, poses = read_index(d), {"DLC": d / "windows_DLC.csv", "LP": d / "windows_LP.csv"}
    else:
        animal, date = "PS93", "20260908"
        d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / f"{animal}_{date}"
        idx, poses = pd.read_csv(d / "cue_windows_index.csv"), {"DLC": d / "cue_windows_DLC.csv", "LP": a.lp}
    poses = {m: f for m, f in poses.items() if f.exists()}
    raw0 = oc.read_pose(poses["DLC"]).iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(raw0[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds(animal, date, rv)
    stop = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"{animal} {date}: kinematics checks ({', '.join(poses)}); spout-frame rms {frame.rms_px:.2f} px")
        Ls = []
        for m, f in poses.items():
            T, J, _M, clean = D.run(f, idx, frame, stop)
            Ls.append(report(m, T, J, clean, oc.read_pose(f).iloc[:len(idx)], idx, frame))
        if len(Ls) == 2:
            agreement(pd.concat(Ls, ignore_index=True))
    print(buf.getvalue())
    (d / "kinematics_checks.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(f"-> {d / 'kinematics_checks.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
