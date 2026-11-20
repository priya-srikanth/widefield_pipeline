"""Cohort-2 (six-position) 3-D lick-phase curves for the VLS manuscript: per position x epoch mean +- SEM of the change from
each animal's own August-pre curve, the raw curves, and the within-animal cluster test. Ported 2026-11-19 from the
manuscript's `manuscript_code/analysis/c2_phase_curves.py` (``--across animals``) and `c2_phase_curves_sessions.py`
(``--across sessions``), v41 (PROMPT_pipeline_sync_v41 item 8); the arithmetic is unchanged.

Input: per-session curves -- `<pose3d>/phase/curves.csv.gz` (`scripts.pose3d_phase`; kinds 'norm' = cam4 onset -> offset
phase and 'peak' = ms from the 3-D extension peak) or the in-mouth-included first-lick curves of
`scripts/pose3d_phase_mouth0.py` (side0 / down0 / out0 / dist / mid0, kind 'peak').
DEFAULT (Priya 2026-11-20): ``--curves-source mouth0`` = `<pose3d>/phase/vls_manuscript/curves_mouth0_first_peak.csv.gz`
(tongue-in-mouth frames count as 0; first licks, kind 'peak' only, so ``--kind`` defaults to 'peak' there).
``--curves-source pipeline`` = the pipeline's `<pose3d>/phase/curves.csv.gz` (in-mouth frames excluded for side / down /
out; both kinds, both lick sets; ``--kind`` defaults to 'norm'). Both are kept; ``--curves`` overrides either path.
Filters (both summaries): kind, lick set, measure, coverage >= 0.5, window (norm: phase 0.1-0.9 = kinematics_pooled_figs
.PHASE_WIN; peak: -80..+80 ms), early-pre (June) sessions dropped (`config.early_pre`); mid / mid0 / side / side0 signed
+ = contralesional (lesions all left; mid0 is not in the pipeline's SIGNED_3D list, so it is flipped here explicitly).
Per session: curve minus its animal's August-pre mean curve.

  --across animals  (Fig 3 / S9 before v39; `kinematics_pooled_figs.phase_data` generalised): per epoch, mean over
                    animals of each animal's mean, SEM over animals (SD ddof=1 / sqrt(n_animals)).
                    -> <out>/c2_phase_<lickset>_<measure>[_peak].csv
  --across sessions (v39+, Priya 11-19: every session counts once): mean over sessions, SEM over sessions (SD ddof=1 /
                    sqrt(n_sessions)), plus n_sessions; n_animals kept for the renderer's >= 3 mask.
                    -> <out>/c2_phase_<lickset>_<measure>[_peak]_sessions.csv

'sig' rows (both): `orofacial_stats.cluster_permutation` of the epoch's session changes against the August-pre sessions'
changes, labels permuted within animal (`orofacial_cohort.CLUSTER_PERM`), uncorrected cluster p < 0.05, on the bins
finite in every session (>= 3 bins, >= 2 epoch sessions). The manuscript's bars use the ALIGNED rule (per-cluster p,
BH-FDR within each measure): run `scripts/c2_phase_sessions_aligned.py <the _sessions.csv> <curves>` on the output.

Default out = `<pose3d>/phase/vls_manuscript/`; ``--out-suffix S`` inserts S before `.csv`. Refuses to overwrite.

    python scripts/pose3d_phase_curves.py first side0 down0 dist mid0 --across sessions          # mouth0 (default)
    python scripts/pose3d_phase_curves.py first side down --kind norm --curves-source pipeline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.orofacial_cohort import CLUSTER_PERM
from wfield_local.orofacial_stats import cluster_permutation
from wfield_local.pose3d import pose3d_dir

EPOCHS = ("acute", "subacute", "chronic")    # scripts.kinematics_pooled_figs.EPOCHS
PHASE_WIN = (0.1, 0.9)                       # scripts.kinematics_pooled_figs.PHASE_WIN (norm kind)
PEAK_WIN = (-80, 80)                         # ms from the 3-D peak: the in-mouth frames before and after the lick
SIGNED = ("mid", "mid0", "side", "side0")    # stored + = ipsilesional -> + = contralesional
#: per-session curve sources under <pose3d> (Priya 2026-11-20: mouth0 is the default, both kept)
CURVES_SOURCES = {"mouth0": ("phase", "vls_manuscript", "curves_mouth0_first_peak.csv.gz"),
                  "pipeline": ("phase", "curves.csv.gz")}


def curves_path(d3: Path, source: str) -> Path:
    return Path(d3).joinpath(*CURVES_SOURCES[source])


def load_curves(curves, lset: str, meas, kind: str) -> pd.DataFrame:
    # round_trip = the exact double of each stored decimal, whatever the pandas version's default parser does
    C = pd.read_csv(curves, float_precision="round_trip") if not isinstance(curves, pd.DataFrame) else curves
    win = PHASE_WIN if kind == "norm" else PEAK_WIN
    C = C[(C.kind == kind) & (C.lickset == lset) & C.measure.isin(meas) & (C["cov"] >= 0.5) & C.x.between(*win)]
    return C[(C.epoch != "pre") | (~C.date.astype(str).map(config.early_pre))]


def measure_table(C: pd.DataFrame, m: str, across: str) -> pd.DataFrame:
    rows = []
    sg = -1.0 if m in SIGNED else 1.0
    for p, g in C[C.measure == m].groupby("position"):
        pv = g.pivot_table(index=["animal", "session", "epoch"], columns="x", values="v") * sg
        an = pv.index.get_level_values(0).to_numpy()
        ep = pv.index.get_level_values(2).to_numpy()
        base = pv[ep == "pre"].groupby(level=0).mean()
        ch = pd.DataFrame(pv.to_numpy(float) - base.reindex(an).to_numpy(float), index=pv.index, columns=pv.columns)
        xs = pv.columns.to_numpy(float)
        for e in ("pre",) + EPOCHS:
            sel = ep == e
            if not sel.any():
                continue
            for row, src in (("mean", ch), ("raw", pv)):
                if across == "animals":
                    per = src[sel].groupby(level=0).mean()
                    n = per.notna().sum(0).to_numpy()
                    rows.append(pd.DataFrame({"row": row, "position": p, "measure": m, "epoch": e, "x": xs,
                                              "v": per.mean(0).to_numpy(),
                                              "sem": per.std(0).to_numpy() / np.sqrt(np.maximum(n, 1)),
                                              "n_animals": n}))
                else:
                    s = src[sel]
                    ns = s.notna().sum(0).to_numpy()
                    na = s.notna().groupby(level=0).any().sum(0).to_numpy()
                    sd = s.std(0, ddof=1).to_numpy()
                    rows.append(pd.DataFrame({"row": row, "position": p, "measure": m, "epoch": e, "x": xs,
                                              "v": s.mean(0).to_numpy(), "sem": sd / np.sqrt(np.maximum(ns, 1)),
                                              "n_sessions": ns, "n_animals": na}))
            if e == "pre" or sel.sum() < 2:
                continue
            keep = (ep == "pre") | sel
            sub = ch.to_numpy(float)[keep]
            ok = np.isfinite(sub).all(0)
            mask = np.zeros(len(xs), bool)
            if ok.sum() >= 3:
                mask[ok] = cluster_permutation(sub[:, ok], sel[keep], an[keep], n_perm=CLUSTER_PERM)
            rows.append(pd.DataFrame({"row": "sig", "position": p, "measure": m, "epoch": e, "x": xs, "v": mask}))
    return pd.concat(rows)


def out_name(lset: str, m: str, kind: str, across: str, suffix: str = "") -> str:
    return (f"c2_phase_{lset}_{m}" + ("" if kind == "norm" else "_peak")
            + ("_sessions" if across == "sessions" else "") + suffix + ".csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lickset", choices=["first", "all"])
    ap.add_argument("measures", nargs="+")
    ap.add_argument("--kind", choices=["norm", "peak"], default=None,
                    help="default 'peak' with --curves-source mouth0, 'norm' with pipeline")
    ap.add_argument("--across", choices=["animals", "sessions"], default="sessions")
    ap.add_argument("--curves-source", choices=sorted(CURVES_SOURCES), default="mouth0",
                    help="mouth0 (default; in-mouth = 0) or pipeline (<pose3d>/phase/curves.csv.gz, in-mouth excluded)")
    ap.add_argument("--curves", type=Path, default=None, help="explicit curves file (overrides --curves-source)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <pose3d>/phase/vls_manuscript")
    ap.add_argument("--out-suffix", default="", help="inserted before .csv in every output name")
    a = ap.parse_args(argv)
    kind = a.kind or ("peak" if a.curves_source == "mouth0" else "norm")
    if a.curves is None and a.curves_source == "mouth0" and (kind != "peak" or a.lickset != "first"):
        sys.exit("the mouth0 curves hold first licks, kind 'peak' only; use --curves-source pipeline")
    d3 = pose3d_dir()
    out_dir = a.out_dir or d3 / "phase" / "vls_manuscript"
    outs = {m: out_dir / out_name(a.lickset, m, kind, a.across, a.out_suffix) for m in a.measures}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash:
        sys.exit("refusing to overwrite: " + ", ".join(clash))
    curves = a.curves or curves_path(d3, a.curves_source)
    print("curves:", curves, flush=True)
    C = load_curves(curves, a.lickset, a.measures, kind)
    missing = sorted(set(a.measures) - set(C.measure))
    if missing:
        sys.exit(f"no curves for {missing} in {curves}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in a.measures:
        measure_table(C, m, a.across).to_csv(outs[m], index=False)
        print("wrote", outs[m], flush=True)


if __name__ == "__main__":
    main()
