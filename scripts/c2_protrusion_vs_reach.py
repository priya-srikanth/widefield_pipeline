"""Six-position FRONT-CAMERA 2-D protrusion vs 3-D reach on the SAME licks (Priya 11-22): the like-for-like counterpart
of cohort 1, which had the front camera only. Behind the PENDING Results sentence "protrusion of the same licks
appeared reduced ...".

    python scripts/c2_protrusion_vs_reach.py [--out-dir DIR] [--force]

LICKS: the four-camera post-cue table (`kinematics_fused_postcue`: cam1-only events dropped, lick_idx 0 = the first
post-cue lick) restricted to licks cam4 ALSO saw (``seen_cam4``) WITH 3-D support (``kin_ok``: >= 2 views, finite reach).
Each such lick carries cam4's own peak (its ``lick_key`` = trial:cam4 peak ms), so it joins the cam4 per-lick table of the
adopted run (``<run>/sessions/<a>/<d>/licks.csv``) exactly; there:
  protrusion_px  = ``protrusion_max_measured_px`` with MEASURED peaks only (``peak_interp`` False) -- the cohort-1 /
                   `orofacial_cohort` protrusion_px (xy distance of the tongue tip from the mouth at the lick peak, cam4 px)
  reach_mm       = the 3-D reach of the same lick (mm)
Lick sets: FIRST (lick_idx 0 of the four-camera table; a trial whose first lick is not seen by cam4 / has no 3-D
contributes nothing -- never the second lick) and ALL. Per session x position medians (cells with >= 20 licks, as the
3-D endpoints); sessions = the table's, early pre (June / 0805) out (`config.early_pre`), UNGATED like Fig 4B / 4E.
TESTS: `orofacial_stats.session_change_tests` (session MWU + animal-weighted nested bootstrap, "both" = star), ONE small
family of its own (BH over the 4 metrics x 6 positions x 3 epochs, ``fdr_scope="all"``): this is a method comparison,
not a new endpoint of the scalar family.
Outputs (new names): ``c2_protrusion_vs_reach_sessions.csv``, ``c2_protrusion_vs_reach_stats.csv``,
``c2_protrusion_vs_reach_per_animal.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DEFAULT = Path("M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_4cam_sync60_20261122")
LR_POS = dict(zip(["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"],
                  ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]))   # lesions left
MIN_N = 20
METRICS = ("protrusion_first", "reach_first", "protrusion_all", "reach_all")


def _cam4(job) -> pd.DataFrame:
    """(session, lick_key, protrusion_px, peak_interp) of one session's cam4 post-cue licks. fan_sessions worker."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import session_licks as SL
    from wfield_local.paths import PathResolver
    s = job
    a, d = s.split("_")
    f = oo.session_dir(PathResolver(), oc.RUN, a, d) / "licks.csv"
    if not f.exists():
        return pd.DataFrame()
    C = pd.read_csv(f, usecols=["trial_id", "t_ms", "protrusion_max_measured_px", "peak_interp"])
    return pd.DataFrame({"session": s, "lick_key": SL.lick_key(C.trial_id, C.t_ms),
                         "protrusion_px": C.protrusion_max_measured_px.to_numpy(float),
                         "peak_interp": C.peak_interp.astype(str).str.lower().eq("true").to_numpy()})


def session_table(K: pd.DataFrame) -> pd.DataFrame:
    """Per session x position medians of protrusion and reach for the first and all same-lick sets. Pure (tested)."""
    rows = []
    for (s, a, d, e, p), g in K.groupby(["session", "animal", "date", "epoch", "position"], sort=True):
        f = g[g.lick_idx == 0]
        r = {"session": s, "animal": a, "date": d, "epoch": e, "position": p, "pos": LR_POS.get(p),
             "n_first": len(f), "n_all": len(g)}
        for nm, gg, n in (("first", f, len(f)), ("all", g, len(g))):
            ok = n >= MIN_N
            r[f"protrusion_{nm}"] = float(gg.protrusion_px.median()) if ok else np.nan
            r[f"reach_{nm}"] = float(gg.reach_mm.median()) if ok else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import orofacial_stats as st
    from wfield_local import session_licks as SL
    from wfield_local.pose3d import pose3d_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kin", type=Path, default=None, help=f"default <pose3d>/{SL.KIN_POSTCUE}/licks.csv.gz")
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    names = ("c2_protrusion_vs_reach_sessions.csv", "c2_protrusion_vs_reach_stats.csv",
             "c2_protrusion_vs_reach_per_animal.csv")
    outs = [a.out_dir / n for n in names]
    if any(o.exists() for o in outs) and not a.force:
        raise SystemExit(f"{a.out_dir}: outputs exist (--force)")
    K = pd.read_csv(a.kin or pose3d_dir() / SL.KIN_POSTCUE / "licks.csv.gz", dtype={"date": str, "lick_key": str},
                    usecols=["session", "animal", "date", "epoch", "trial_id", "lick_idx", "position", "lick_key",
                             "seen_cam4", "kin_ok", "reach_mm"])
    K = K[~((K.epoch == "pre") & K.date.map(config.early_pre))]
    n0 = len(K)
    K = K[K.seen_cam4.astype(str).str.lower().eq("true") & K.kin_ok.astype(str).str.lower().eq("true")
          & K.reach_mm.notna()]
    res, fails = ak.fan_sessions(sorted(K.session.unique()), _cam4, jobs=a.jobs)
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    C = pd.concat([d for _, d in res], ignore_index=True)
    J = K.merge(C, on=["session", "lick_key"], how="left", validate="many_to_one")
    unmatched = int(J.protrusion_px.isna().sum())
    J = J[J.protrusion_px.notna() & ~J.peak_interp.astype(bool)]
    print(f"post-cue licks {n0}; seen by cam4 with 3-D {len(K)}; no cam4 row {unmatched}; measured cam4 peak {len(J)}")
    S = session_table(J)
    P = S.rename(columns={})
    R = st.session_change_tests(P, list(METRICS), fdr_scope="all")
    # rule 8: per animal x epoch, mean over positions of the session change from the animal's pre at that position
    pa = {}
    for m in METRICS:
        b = P[P.epoch == "pre"].groupby(["animal", "pos"])[m].mean()
        d = P[P.epoch != "pre"].copy()
        d["chg"] = d[m] - b.reindex(pd.MultiIndex.from_frame(d[["animal", "pos"]])).to_numpy()
        pa[m] = d.groupby(["animal", "epoch"]).chg.mean()
    PA = pd.DataFrame(pa).unstack("epoch").reindex(columns=["acute", "subacute", "chronic"], level=1)
    pd.set_option("display.width", 250)
    print("== per animal (rule 8): change from pre, mean over positions (px / mm)")
    print(PA.round(2).to_string())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    S.to_csv(outs[0], index=False)
    R.to_csv(outs[1], index=False, float_format="%.4g")
    PA.to_csv(outs[2])
    show = R[["metric", "pos", "epoch", "n_sessions", "boot_change", "boot_lo", "boot_hi", "q_fdr", "agree"]]
    print(show.round(3).to_string())
    for o in outs:
        print("wrote", o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
