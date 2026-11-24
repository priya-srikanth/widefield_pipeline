"""Six-position cohort: peak-speed ROBUSTNESS (smoothed trajectory) and tongue-path NOISE by epoch (Priya 2026-11-23,
approved after the PS95 cam3 jitter test `P3D/qc_cam3_drop_20261122/jitter_test/`). A robustness check only: the
manuscript tables keep the pipeline's raw peak speed.

Per post-cue lick of the four-camera table (`kinematics_fused_postcue/licks.csv.gz`, rows with a finite production
``speed_peak``), from the session's 3-D frames (``<P3D>/<session>_frames.csv.gz``), with the lick window exactly as
`pose3d_kinematics.session_licks` (frames on_ms .. off_ms with a finite tongue point; 3-D peak = the frame farthest from
the origin; speed = 8 ms central difference of the 3-D tip, max over onset .. 3-D peak; the jitter-test recomputation,
which matches the pipeline within the table rounding):
  speed_raw    the recomputed pipeline speed (check column; vs the table's ``speed_peak``)
  speed_sg5    the same peak from a Savitzky-Golay-smoothed trajectory (order 2, 5 frames = 20 ms, along each run of
               consecutive 4 ms frames >= 5 long; shorter runs raw)
  jit_pro      tongue-path noise: RMS of |P[i+1] - 2 P[i] + P[i-1]| (mm) over the protrusion (onset .. 3-D peak; centre
               frames with 4 ms neighbours). It includes true acceleration: compare epochs, not absolute values
  f_le2v_pro   share of protrusion frames with <= 2 tongue views (2-view triangulations + 1-view fills)
  f_le2v       the same over the whole lick window;  f_fill_pro  1-view fills only
  speedpk_le2v the speed-peak frame rests on a <= 2-view frame (the worse of its two central-difference neighbours)

    python scripts/c2_speed_smooth_robustness.py run --out DIR [--jobs 3] [--sessions S ...]
    python scripts/c2_speed_smooth_robustness.py summarise --out DIR

run       -> DIR/per_lick/<session>.csv.gz (cached per session on its frames + the licks table; a re-run skips sessions
             whose inputs are unchanged)
summarise -> DIR/c2_lick3d_session_summary_ungated_sg5.csv (`c2_lick3d_session_summary.summarise` of the licks table
             with speed_peak := speed_sg5; feed it to c2_scalar_family_aligned.py --lick3d),
             DIR/c2_lick3d_session_summary_ungated_rawrecompute.csv (speed_peak := speed_raw; the family on it must
             reproduce the production speed stars),
             DIR/noise_sessions.csv (per session: lick medians of jit_pro, f_le2v_pro, f_le2v, f_fill_pro and the raw - sg5
             peak gap; the share of licks whose speed peak rests on a <= 2-view frame), DIR/noise_by_animal_epoch.csv (rule 8: per animal x epoch means of the
             session values, with n sessions), DIR/noise_by_epoch.csv (animal-weighted epoch means and the change from
             each animal's own pre), DIR/recompute_check.csv.
Early pre sessions (June / 0805; `config.early_pre`) are out, as in the session summary.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FCOLS = ["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv", "tongue_views"]
LCOLS = ["session", "animal", "date", "epoch", "trial_id", "lick_idx", "position", "lick_key", "on_ms", "off_ms",
         "speed_peak", "reach_mm", "side_ml"]
SG_WIN = 5
EPS = ["pre", "acute", "subacute", "chronic"]
NOISE = ["jit_pro", "f_le2v_pro", "f_le2v", "f_fill_pro", "gap_raw_sg5", "speedpk_le2v"]


def central_speed(t: np.ndarray, P: np.ndarray) -> np.ndarray:
    sp = np.full(len(P), np.nan)
    if len(P) >= 3:
        cons = np.abs(t[2:] - t[:-2] - 8.0) < 0.5
        sp[1:-1][cons] = np.linalg.norm(P[2:][cons] - P[:-2][cons], axis=1) / 0.008
    return sp


def second_diff_rms(t: np.ndarray, P: np.ndarray, lo: int, hi: int) -> float:
    if len(P) < 3:
        return np.nan
    i = np.arange(1, len(P) - 1)
    ok = (np.abs(t[i] - t[i - 1] - 4) < 0.5) & (np.abs(t[i + 1] - t[i] - 4) < 0.5) & (i >= lo) & (i <= hi)
    i = i[ok]
    if not len(i):
        return np.nan
    a = np.linalg.norm(P[i + 1] - 2 * P[i] + P[i - 1], axis=1)
    return float(np.sqrt(np.mean(a ** 2)))


def smooth_runs(t: np.ndarray, P: np.ndarray, win: int) -> np.ndarray:
    from scipy.signal import savgol_filter
    Q = P.copy()
    br = np.flatnonzero(np.abs(np.diff(t) - 4) > 0.5) + 1
    for s, e in zip(np.r_[0, br], np.r_[br, len(t)]):
        if e - s >= win:
            Q[s:e] = savgol_filter(P[s:e], win, 2, axis=0)
    return Q


def lick_measures(t_all: np.ndarray, P_all: np.ndarray, v_all: np.ndarray, on: float, off: float) -> dict | None:
    """One lick (pure; tested): the pipeline window / peak / raw speed, the SG5 speed, jitter and view shares."""
    sel = (t_all >= on - 0.5) & (t_all <= off + 0.5)
    t, P, v = t_all[sel], P_all[sel], v_all[sel]
    ok = np.isfinite(P).all(1)
    if ok.sum() < 3:
        return None
    t, P, v = t[ok], P[ok], v[ok]
    k = int(np.argmax(np.linalg.norm(P, axis=1)))
    sp = central_speed(t, P)
    if not np.isfinite(sp[:k + 1]).any():
        return None
    j = int(np.nanargmax(sp[:k + 1]))
    vj = min(v[j - 1], v[j + 1]) if 0 < j < len(v) - 1 else v[j]
    if 0 < j < len(v) - 1 and (v[j - 1] == -1 or v[j + 1] == -1):
        vj = -1
    spw = central_speed(t, smooth_runs(t, P, SG_WIN))
    return {"speed_raw": float(sp[j]),
            "speed_sg5": float(np.nanmax(spw[:k + 1])) if np.isfinite(spw[:k + 1]).any() else np.nan,
            "jit_pro": second_diff_rms(t, P, 0, k),
            "f_le2v_pro": float(np.mean(v[:k + 1] <= 2)), "f_le2v": float(np.mean(v <= 2)),
            "f_fill_pro": float(np.mean(v[:k + 1] == -1)), "speedpk_le2v": float(vj <= 2), "n3d": int(len(P))}


def _licks_path(p3d: Path) -> Path:
    return p3d / "kinematics_fused_postcue" / "licks.csv.gz"


def session_worker(job) -> dict:
    """fan_sessions worker: one session -> OUT/per_lick/<s>.csv.gz (cached on the frames file + licks table)."""
    from wfield_local import result_cache
    s, p3d, out = job
    p3d, out = Path(p3d), Path(out)
    fr, lk = p3d / f"{s}_frames.csv.gz", _licks_path(p3d)
    f = out / "per_lick" / f"{s}.csv.gz"

    def compute():
        L = pd.read_csv(lk, usecols=LCOLS, dtype={"lick_key": str, "date": str})
        L = L[(L.session == s) & np.isfinite(L.speed_peak)]
        F = pd.read_csv(fr, usecols=FCOLS).drop_duplicates(["trial_id", "t_ms"])
        F = F[F.trial_id.isin(set(L.trial_id))]
        G = {k: (g.t_ms.to_numpy(float), g[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float),
                 g.tongue_views.to_numpy()) for k, g in F.sort_values("t_ms").groupby("trial_id")}
        rows = []
        for r in L.itertuples():
            g = G.get(r.trial_id)
            m = lick_measures(*g, r.on_ms, r.off_ms) if g is not None else None
            rows.append({"lick_key": r.lick_key, "trial_id": r.trial_id, **(m or {})})
        return pd.DataFrame(rows)

    from wfield_local.result_cache import code_version
    ver = code_version(sys.modules[__name__])
    D = result_cache.cached(out / "_cache", s, [fr, lk], ver, compute)
    f.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(f, index=False, float_format="%.6g")
    return {"session": s, "n": len(D)}


def cmd_run(p3d: Path, out: Path, jobs: int, sessions: list[str] | None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(out)
    ss = sessions or sorted(pd.read_csv(_licks_path(p3d), usecols=["session"]).session.unique())
    res, fail = ak.fan_sessions([(s, str(p3d), str(out)) for s in ss], session_worker, jobs=jobs)
    print(f"{len(res)} sessions ok, failed {fail[:3]}", flush=True)
    return 1 if fail else 0


def per_lick_table(p3d: Path, out: Path) -> pd.DataFrame:
    L = pd.read_csv(_licks_path(p3d), usecols=LCOLS, dtype={"lick_key": str, "date": str})
    parts = [pd.read_csv(out / "per_lick" / f"{s}.csv.gz", dtype={"lick_key": str}).assign(session=s)
             for s in sorted(L.session.unique()) if (out / "per_lick" / f"{s}.csv.gz").exists()]
    D = pd.concat(parts, ignore_index=True).drop(columns=["trial_id"])
    missing = sorted(set(L.session) - set(D.session))
    if missing:
        raise SystemExit(f"per-lick files missing for {len(missing)} sessions: {missing[:5]} (run first)")
    return L.merge(D, on=["session", "lick_key"], how="left", validate="one_to_one")


def noise_tables(M: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(sessions, per animal x epoch, epoch) noise tables from the merged per-lick table (pure; tested)."""
    M = M.assign(gap_raw_sg5=M.speed_raw - M.speed_sg5)
    g = M.groupby(["session", "animal", "date", "epoch"])
    S = g[[m for m in NOISE if m != "speedpk_le2v"]].median()
    S["speedpk_le2v"] = g.speedpk_le2v.mean()                   # a 0/1 flag per lick: the session SHARE, not a median
    S = S.reset_index()[["session", "animal", "date", "epoch", *NOISE]]
    S["n_licks"] = M.groupby(["session", "animal", "date", "epoch"]).jit_pro.count().to_numpy()
    A = S.groupby(["animal", "epoch"])[NOISE].mean()
    A["n_sessions"] = S.groupby(["animal", "epoch"]).session.nunique()
    A = A.reset_index()
    pre = A[A.epoch == "pre"].set_index("animal")[NOISE]
    rows = []
    for e in EPS:
        g = A[A.epoch == e].set_index("animal")
        r = {"epoch": e, "n_animals": len(g), "n_sessions": int(g.n_sessions.sum())}
        for m in NOISE:
            d = g[m] - pre[m].reindex(g.index)
            r[m] = g[m].mean()
            r[f"{m}_chg"] = d.mean()
            r[f"{m}_chg_min"], r[f"{m}_chg_max"] = d.min(), d.max()
            r[f"{m}_n_up"] = int((d > 0).sum())
        rows.append(r)
    E = pd.DataFrame(rows)
    A["epoch"] = pd.Categorical(A.epoch, EPS, ordered=True)
    return S, A.sort_values(["animal", "epoch"]).reset_index(drop=True), E


def cmd_summarise(p3d: Path, out: Path) -> int:
    from wfield_local import config
    from scripts.c2_lick3d_session_summary import per_animal, summarise
    M = per_lick_table(p3d, out)
    M = M[~((M.epoch == "pre") & M.date.astype(str).map(config.early_pre))]
    chk = M.dropna(subset=["speed_raw"])
    d = (chk.speed_raw - chk.speed_peak).abs()
    C = pd.DataFrame([{"n_licks": len(M), "n_recomputed": len(chk), "max_abs_diff": d.max(),
                       "p99_abs_diff": d.quantile(0.99), "share_gt_0p05": float((d > 0.05).mean())}])
    print("recompute vs table speed_peak:", C.round(4).to_dict("records"), flush=True)
    C.to_csv(out / "recompute_check.csv", index=False)
    for tag, col in (("sg5", "speed_sg5"), ("rawrecompute", "speed_raw")):
        S = summarise(M.assign(speed_peak=M[col]))
        print(f"== {tag}: per animal (rule 8), epoch means of session x position medians")
        print(per_animal(S).round(2).to_string(), flush=True)
        S.to_csv(out / f"c2_lick3d_session_summary_ungated_{tag}.csv", index=False, lineterminator="\n")
    S, A, E = noise_tables(M)
    pd.set_option("display.width", 220)
    print("== noise per animal x epoch (rule 8)")
    print(A.round(4).to_string(index=False))
    print("== noise by epoch (animal-weighted; chg = vs each animal's own pre)")
    print(E.round(4).to_string(index=False), flush=True)
    S.to_csv(out / "noise_sessions.csv", index=False)
    A.to_csv(out / "noise_by_animal_epoch.csv", index=False)
    E.to_csv(out / "noise_by_epoch.csv", index=False)
    print("wrote", out)
    return 0


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    from wfield_local.pose3d import pose3d_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summarise"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pose3d", type=Path, default=None, help="pose3d dir (default: pose3d_dir())")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sessions", nargs="+", default=None)
    a = ap.parse_args(argv)
    p3d = a.pose3d or pose3d_dir(PathResolver())
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        return cmd_run(p3d, a.out, min(a.jobs, 3), a.sessions)
    return cmd_summarise(p3d, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
