"""Post-cue view of the fused four-camera 3-D kinematics: a drop-in for the cam4 ``kinematics/`` folder (Priya 11-22: "we
should use 4 camera licks for all figures (manuscript and otherwise)").

    python scripts/pose3d_kin_postcue.py [--src <pose3d>/kinematics_fused] [--out <pose3d>/kinematics_fused_postcue]

``kinematics_fused/`` (`scripts/pose3d_kinematics --licks fused`) holds EVERY fused lick of the session (pre-cue, post-cue,
ITI, the 0-70 ms exclusion) with ``lick_idx`` counted across all of them, so ``lick_idx == 0`` is usually a pre-cue lick
and its sessions.csv medians pool the contexts. The cam4 ``kinematics/`` folder the trial-window readers were written for
holds post-cue licks only, ``lick_idx`` from 0 within the trial. This script writes that shape from the fused table,
without recomputing any 3-D:
  licks.csv.gz      context == post_cue; lick_idx renumbered 0.. within (session, trial) in peak order (the fused
                    train's own order: ``lick_idx_fused`` keeps the original); + ``on_ms`` / ``off_ms`` = the fused
                    lick's span (the window its 3-D measures were taken over; `session_licks.session_licks`), joined
                    on ``lick_key``. Rows with < 3 3-D frames are KEPT with NaN measures (Q1 11-21: a fused lick the
                    3-D cannot measure is missing data, not a reason to promote the next lick to "first").
  jaw_trials.csv.gz copied (trial-level, lick set independent).
  sessions.csv      the fused sessions.csv's rest / jaw columns, with every ``<measure>|<position>`` lick median
                    recomputed on the post-cue licks; ``licks_postcue`` / ``licks_3d`` / ``not_in_cam4`` counts.
  by_animal.csv     `pose3d_kinematics.by_animal` (rule 8) on that sessions table.
Frame-coverage-gated trials are already absent from the fused table (`kinematics_fused/licks_gated.csv.gz`).

3-D SUPPORT (coordinator 11-22): 6.0 % of fused post-cue licks are seen by ONE camera (cam1 only 4.5 %, cam4 only 1.5 %);
cam1-only licks have median 0 3-D frames, reach 1.85 mm, DAQ contact 0.3 % and come late in the trial, so kinematics must
not be read from them. PRIMARY folder (``kinematics_fused_postcue``): every lick row is KEPT (lick counts unchanged) but
the kinematic measures are blanked (NaN) unless ``kin_ok`` = seen by >= `MIN_VIEWS` cameras (``n_views``). The
``_allviews`` folder keeps single-view kinematics as measured (comparison; how counts / kinematics treat single-view
licks is Priya's call). Both carry ``kin_ok`` and ``n_views``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

LICK_COLS_FIRST = ["trial_id", "lick_idx"]
#: kinematics only from licks seen by at least this many cameras (3-D support); see the module docstring
MIN_VIEWS = 2
ALLVIEWS_SUFFIX = "_allviews"


def _spans(job) -> pd.DataFrame:
    """(session, lick_key, on_ms, off_ms) of one session's fused licks (module level: fanned out)."""
    from wfield_local import session_licks as SL
    s = job
    A = SL.session_licks(s)
    return pd.DataFrame({"session": s, "lick_key": A.lick_key.astype(str), "on_ms": A.on_ms.to_numpy(float),
                         "off_ms": A.off_ms.to_numpy(float)})


def postcue_licks(L: pd.DataFrame, spans: pd.DataFrame | None = None) -> pd.DataFrame:
    """Post-cue rows of a fused kinematics lick table, lick_idx renumbered within (session, trial). Pure (tested)."""
    P = L[L.context == "post_cue"].copy()
    P["lick_idx_fused"] = P.lick_idx
    P = P.sort_values(["session", "trial_id", "t_peak_cam4", "lick_idx_fused"], kind="mergesort")
    P["lick_idx"] = P.groupby(["session", "trial_id"]).cumcount()
    if spans is not None:
        n = len(P)
        P = P.merge(spans, on=["session", "lick_key"], how="left", validate="one_to_one")
        if len(P) != n:
            raise RuntimeError("span join changed the row count")
    return P.reset_index(drop=True)


def mask_single_view(P: pd.DataFrame, min_views: int = MIN_VIEWS) -> pd.DataFrame:
    """``kin_ok`` = n_views >= ``min_views``; kinematic measures NaN where not (rows kept). Pure (tested)."""
    from scripts.pose3d_kinematics import LICK_MEASURES
    P = P.copy()
    P["kin_ok"] = P.n_views.fillna(0).astype(int) >= min_views
    cols = [c for c in LICK_MEASURES + ["t_peak_3d", "midline_target", "frontal_target", "sagittal_target"] if c in P]
    P.loc[~P.kin_ok, cols] = np.nan
    return P


def session_table(S_fused: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """The fused sessions.csv with its lick medians recomputed on the post-cue licks ``P``."""
    from scripts.pose3d_kinematics import LICK_MEASURES
    keep = [c for c in S_fused.columns if not ("|" in c and c.split("|")[0] in LICK_MEASURES)
            and c not in ("cam4_licks", "licks_3d", "not_in_cam4")]
    rows = []
    for r in S_fused[keep].to_dict("records"):
        g = P[P.session == r["session"]]
        r.update({"licks_postcue": len(g), "cam4_licks": int(g.cam4_matched.astype(bool).sum()),
                  "licks_3d": int((g.n_3d >= 3).sum()),
                  "licks_kin": int(g.reach_mm.notna().sum()), "single_view": int((g.n_views < MIN_VIEWS).sum()), "not_in_cam4": int((~g.cam4_matched.astype(bool)).sum())})
        for p, gp in g.groupby("position"):
            for m in LICK_MEASURES:
                r[f"{m}|{p}"] = float(gp[m].median()) if gp[m].notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import session_licks as SL
    from wfield_local.pose3d import pose3d_dir
    from scripts.pose3d_kinematics import by_animal
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=None, help="default <pose3d>/kinematics_fused")
    ap.add_argument("--out", type=Path, default=None, help=f"default <pose3d>/{SL.KIN_POSTCUE}")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    d3 = pose3d_dir()
    src, out = a.src or d3 / "kinematics_fused", a.out or d3 / SL.KIN_POSTCUE
    L = pd.read_csv(src / "licks.csv.gz", dtype={"date": str, "lick_key": str})
    sess = sorted(L.session.unique())
    res, fails = ak.fan_sessions(sess, _spans, jobs=a.jobs)
    for s, e in fails:
        print(f"{s}: span read FAILED {e}")
    spans = pd.concat([d for _, d in res], ignore_index=True)
    P0 = postcue_licks(L, spans)
    miss = P0.on_ms.isna() & P0.lick_key.notna()
    print(f"{len(L)} fused licks -> {len(P0)} post-cue ({P0.session.nunique()} sessions); span missing for {int(miss.sum())}")
    S_fused = pd.read_csv(src / "sessions.csv", dtype={"date": str})
    for o, P, rule in ((out, mask_single_view(P0), f"kinematics only where n_views >= {MIN_VIEWS} (rows kept)"),
                       (out.with_name(out.name + ALLVIEWS_SUFFIX), mask_single_view(P0, 0), "single-view kinematics kept")):
        o.mkdir(parents=True, exist_ok=True)
        if (o / "licks.csv.gz").exists() and not a.force:
            raise SystemExit(f"{o / 'licks.csv.gz'} exists: --force to overwrite")
        sv = int((P.n_views < MIN_VIEWS).sum())
        print(f"{o.name}: {rule}; single-view licks {sv} ({sv / len(P):.1%}); licks with kinematics "
              f"{int(P.reach_mm.notna().sum())}")
        S = session_table(S_fused, P)
        if "daq_peak_s" in P:
            P = P.assign(daq_peak_s=P["daq_peak_s"].map("{:.4f}".format))
        P.to_csv(o / "licks.csv.gz", index=False, float_format="%.4g")
        shutil.copy2(src / "jaw_trials.csv.gz", o / "jaw_trials.csv.gz")
        S.to_csv(o / "sessions.csv", index=False)
        by_animal(S).to_csv(o / "by_animal.csv", index=False)
        (o / "manifest.json").write_text(json.dumps({"source": str(src), "lick_set": SL.describe(),
                                                     "rule": "context == post_cue; lick_idx renumbered within trial; "
                                                     + rule, "n_licks": len(P), "single_view": sv,
                                                     "n_sessions": int(P.session.nunique())}, indent=1))
        print("->", o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
