"""Pre-stroke cue-aligned FaceRhythm factor amplitude by spout position (VLS manuscript Fig 5C input), made in the pipeline.

The manuscript's `manuscript_code/data/fr_pre_timecourses_abs.npz` had no producer in this repo (Priya approved adding one,
2026-11-20). Recipe = the factor gallery's "pre-stroke trials by position" (`face_rhythm.factors.report.cmd_report`,
``absolute=True`` -> `*_abs`): for every PRE-stroke proj session of the frozen master
(`face_rhythm_outputs.factor_dir(PathResolver(), BUNDLE, SET)/proj/*.npz`, enumerated by
`face_rhythm.common.sessions(fdir, "proj")`, i.e. sorted by file name), the absolute expression
``trials x trials_D`` (projected temporal weights x the FR normalizer D; engaged trials, cue -2..+4 s, 25 Hz,
151 samples) averaged (nanmean) over that session's trials at each spout position.

Output (refuses to overwrite): ``<readouts>/<SET>/vls_manuscript/fr_pre_timecourses_abs.npz`` (``--out`` to change):
``t_ms`` (151,) and, per position (close_L, close_center, close_R, far_L, far_center, far_R), a float32
(sessions x 151 x factors) array, plus ``sessions_<position>`` (session labels, same order). The manuscript file's
layout is the same without the label keys. ``--compare`` (default: the manuscript file) prints, per position, the
session counts and the max |difference|; ``--compare none`` skips it.

    python scripts/fr_pre_timecourses_abs.py [--out NPZ] [--compare NPZ|none]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wfield_local import face_rhythm_outputs as fro  # noqa: E402
from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.paths import PathResolver  # noqa: E402

POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
MANUSCRIPT = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/manuscript_code/data/"
                  "fr_pre_timecourses_abs.npz")


#: ``--gate working`` (11-23, four-camera rule): only grid trials inside the camera working period (final class of the
#: four-camera trial-class table not out of the period; trials without video dropped)
CLASS_TABLE = ("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/"
               "qc/s1_trial_classes_cue_stop.csv.gz")
NOT_WP = ("stopped", "stopped_licking", "early_run", "after_end", "unknown", "no_video")


def working_mask(label: str, cue: np.ndarray, C) -> np.ndarray:
    """Per grid cue: inside the camera working period (nearest class-table cue within 50 ms). Pure given ``C``."""
    c = C[C.label == label].sort_values("cue_s")
    if c.empty:
        return np.zeros(len(cue), bool)
    cs = c.cue_s.to_numpy(float)
    hi = np.clip(np.searchsorted(cs, cue), 0, len(cs) - 1)
    lo = np.clip(hi - 1, 0, None)
    j = np.where(np.abs(cs[lo] - cue) <= np.abs(cs[hi] - cue), lo, hi)
    ok = np.abs(cs[j] - cue) <= 0.05
    return ok & ~np.isin(c.cls_final.astype(str).to_numpy()[j], NOT_WP)


def build(fdir: Path, gate: str = "none") -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[str]]]:
    C = None
    if gate == "working":
        import pandas as pd
        C = pd.read_csv(CLASS_TABLE, usecols=["label", "cue_s", "cls_final"])
    T = frc.sessions(fdir, "proj")
    T = T[T.epoch == "pre"]
    per = {p: [] for p in POS}
    names = {p: [] for p in POS}
    t_ms = None
    for r in T.itertuples():
        with np.load(r.file) as z:
            t_ms = z["t_ms"]
            tr = z["trials"] * z["trials_D"][..., None]
            pos = z["position"].astype(str)
            if C is not None:
                pos = np.where(working_mask(r.label, z["cue_daq_s"].astype(float), C), pos, "")
        for p in np.unique(pos):
            if p in per:
                per[p].append(np.nanmean(tr[pos == p], 0))
                names[p].append(r.label)
    return t_ms, {p: np.asarray(per[p], np.float32) for p in POS}, names


def compare(arrs: dict[str, np.ndarray], t_ms: np.ndarray, ref: Path) -> None:
    M = np.load(ref)
    print(f"vs {ref}: t_ms max |diff| {float(np.abs(M['t_ms'] - t_ms).max()):.3g}")
    for p in POS:
        a, b = arrs[p], M[p]
        if a.shape != b.shape:
            print(f"  {p}: shape {a.shape} vs {b.shape} -- DIFFERENT")
            continue
        both = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a[both].astype(float) - b[both].astype(float))
        print(f"  {p}: sessions {a.shape[0]} vs {b.shape[0]}; exact {np.array_equal(a, b, equal_nan=True)}; "
              f"max |diff| {d.max() if d.size else 0:.3g} (|ref| max {np.nanmax(np.abs(b)):.3g}); "
              f"NaN pattern equal {np.array_equal(np.isnan(a), np.isnan(b))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="default <readouts>/<SET>/vls_manuscript/fr_pre_timecourses_abs.npz")
    ap.add_argument("--compare", default=str(MANUSCRIPT), help="reference npz, or 'none'")
    ap.add_argument("--gate", default="none", choices=("none", "working"))
    a = ap.parse_args(argv)
    out = a.out or frc.readouts_dir(fro.SET) / "vls_manuscript" / "fr_pre_timecourses_abs.npz"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    t_ms, arrs, names = build(fro.factor_dir(PathResolver(), fro.BUNDLE, fro.SET), a.gate)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t_ms=t_ms.astype(np.float32), **arrs, **{f"sessions_{p}": np.asarray(names[p]) for p in POS})
    print("wrote", out, {p: arrs[p].shape for p in POS}, flush=True)
    if a.compare != "none" and Path(a.compare).exists():
        compare(arrs, t_ms.astype(np.float32), Path(a.compare))
    return 0


if __name__ == "__main__":
    sys.exit(main())
