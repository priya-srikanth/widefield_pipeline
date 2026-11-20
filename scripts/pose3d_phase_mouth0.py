"""In-mouth-included (in-mouth = 0) 3-D components of the PEAK-ALIGNED FIRST-LICK curves (VLS manuscript Fig 4I and the
cohort-2 path / lateral panels; Priya 11-19 approved the 3-D mm version). Ported 2026-11-19 from the manuscript's
`manuscript_code/analysis/c2_curves_mouth0.py` (v41) so the number is made in the pipeline (PROMPT_pipeline_sync_v41
item 8); the arithmetic is unchanged.

`scripts.pose3d_phase._session` keeps in-mouth-included versions only for ``dist``, ``mid0`` and ``elev0``; ``side`` /
``down`` / ``out`` are NaN in the mouth, so their curves start only once the tongue is visible. This repeats
``_session``'s peak-aligned branch exactly -- same 3-D frames (`<pose3d>/<s>_frames.csv.gz`), same +-152 ms / 4 ms grid
(`pose3d_phase.T_REL`), same in-mouth rule (the frame exists in the lick's trial but has no 3-D tongue point) -- and adds

    side0 = -ml, down0 = -dv, out0 = ap   (mm, 0 in the mouth)       next to   dist, mid0   (as `pose3d_phase`)

Lick set: the FIRST lick of each trial (``lick_idx == 0``) with a 3-D peak in `kinematics/licks.csv.gz`. The manuscript
check (DECISIONS 11-19): that set equals the camera-working-period-gated first-lick set of `pose3d_phase` in all 714
session x position cells (53,105 licks), and the dist / mid0 curves reproduce `phase/curves.csv.gz` to its stored
precision (max |diff| 0.005 mm / deg). Positions with < 5 first licks are skipped (as `_session`).

Output (refuses to overwrite): ``<out>`` (default `<pose3d>/phase/vls_manuscript/curves_mouth0_first_peak.csv.gz`), the
columns of `phase/curves.csv.gz` (session, position, kind='peak', measure, x, v, cov, n, animal, date, epoch,
lickset='first'), values written with ``%.5g`` as the manuscript's copy. Sessions fan out over cores
(`analysis_kit.fan_sessions`, sorted, so the file order is fixed).

    python scripts/pose3d_phase_mouth0.py [--out PATH] [--sessions PS92_20260812 ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.pose3d import pose3d_dir

T_REL = np.arange(-152.0, 152.1, 4.0)                    # == scripts.pose3d_phase.T_REL
MEAS = ("dist", "mid0", "side0", "down0", "out0")
LICK_COLS = ["session", "animal", "date", "epoch", "position", "lick_idx", "t_peak_3d", "trial_id"]


def first_licks(d3: Path) -> pd.DataFrame:
    L = pd.read_csv(d3 / "kinematics" / "licks.csv.gz", usecols=LICK_COLS, dtype={"date": str})
    return L[(L.lick_idx == 0) & L.t_peak_3d.notna()]


def _frames(d3: Path, s: str) -> pd.DataFrame:
    S = pd.read_csv(d3 / f"{s}_frames.csv.gz", usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"])
    return S.drop_duplicates(["trial_id", "t_ms"]).set_index(["trial_id", "t_ms"]).sort_index()


def session_rows(S: pd.DataFrame, s: str, gs: pd.DataFrame) -> list[dict]:
    """One session's first-lick peak-aligned curves (all positions with >= 5 licks)."""
    A = np.vstack([S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float), np.full((1, 3), np.nan)])
    rows = []
    for p, g in gs.groupby("position"):
        if len(g) < 5:
            continue
        TQ = g.t_peak_3d.to_numpy(float)[:, None] + T_REL[None, :]
        TR = np.repeat(g.trial_id.to_numpy(), TQ.shape[1])
        k = S.index.get_indexer(pd.MultiIndex.from_arrays([TR, np.round(TQ, 1).ravel()]))
        known = (k >= 0).reshape(TQ.shape)
        P = A[k].reshape(TQ.shape + (3,))
        rr = np.linalg.norm(P, axis=2)
        inm = ~np.isfinite(rr) & known                       # frame exists in the trial, no 3-D tongue point
        with np.errstate(all="ignore"):
            mid = np.degrees(np.arcsin(np.clip(-P[..., 0] / rr, -1, 1)))

        def z(a, inm=inm):
            return np.where(inm, 0.0, a)

        vals = {"dist": np.where(np.isfinite(rr), rr, np.where(known, 0.0, np.nan)), "mid0": z(mid),
                "side0": z(-P[..., 0]), "down0": z(-P[..., 2]), "out0": z(P[..., 1])}
        for m in MEAS:
            Aa = vals[m]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean, cov = np.nanmean(Aa, 0), np.isfinite(Aa).mean(0)
            rows += [dict(session=s, position=p, kind="peak", measure=m, x=xi, v=vi, cov=ci, n=len(Aa),
                          animal=g.animal.iloc[0], date=g.date.iloc[0], epoch=g.epoch.iloc[0], lickset="first")
                     for xi, vi, ci in zip(T_REL, mean, cov)]
    return rows


@functools.lru_cache(maxsize=2)
def _first_licks_cached(d3: str) -> pd.DataFrame:
    return first_licks(Path(d3))                          # read once per worker process, not once per session


def _worker(job):
    """Module-level for the spawn pool: job = (pose3d dir, session)."""
    d3, s = job
    L = _first_licks_cached(d3)
    return session_rows(_frames(Path(d3), s), s, L[L.session == s])


def build(d3: Path, sessions=None, jobs=None) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    ss = sorted(first_licks(d3).session.unique())
    if sessions:
        ss = [s for s in ss if s in set(sessions)]
    res, fail = ak.fan_sessions([(str(d3), s) for s in ss], _worker, jobs=jobs)
    if fail:
        sys.exit(f"{len(fail)} sessions failed (nothing written): {fail[:3]}")
    return pd.DataFrame([r for _, rows in res for r in rows])      # sessions sorted; MEAS order within a position


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose3d", type=Path, default=None, help="pose3d dir (default: pose3d_dir())")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args(argv)
    d3 = a.pose3d or pose3d_dir()
    out = a.out or d3 / "phase" / "vls_manuscript" / "curves_mouth0_first_peak.csv.gz"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    C = build(d3, a.sessions, a.jobs)
    out.parent.mkdir(parents=True, exist_ok=True)
    C.to_csv(out, index=False, float_format="%.5g")
    print("wrote", out, len(C), "rows,", C.session.nunique(), "sessions", flush=True)


if __name__ == "__main__":
    main()
