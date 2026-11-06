"""READ-ONLY check (Priya 11-06, A28): how far would the executed-bin calibration move if it were regenerated on the
CAMERA WORKING PERIOD? The frozen 10-23 snapshot (``exec_bins_calib_20261023.csv``, `frozen_encoder.LICKLAND_CALIB` /
FROZEN_LICKLAND_CALIB, combined_encoding COMBINED_ENCODING_BINS_CALIB) STAYS the encoder input; this script only
measures the difference. Nothing on the share or in the input cache is written: the regenerated tables and the
comparison go to ``--out`` (a scratch folder).

    PYTHONPATH=. python scripts/exec_bins_calib_compare.py --out <scratch dir> \
        [--kin <pose3d_inputs>/pose3d_kin_licks_snapshot_20261023.csv.gz] \
        [--snapshot <pose3d_inputs>/exec_bins_calib_20261023.csv]

Two regenerations from the SAME 3-D lick table the snapshot was cut from (`exec_bins_calibration.pre_licks` / `table`,
the one definition): ``legacy`` (WFIELD_KIN_GATE=legacy = every post-cue lick, the snapshot's rule -- the
reproducibility control: it should match the snapshot up to later changes of the inputs) and ``working`` (the camera
working period, `trial_classes.gate_kin_licks`). For each against the snapshot, per (animal, calib_for, position):
centre shifts (ml / ap / down, mm), the shift in within-spout SD units (|delta centre| / snapshot SD, Euclidean), the
SD (bin scale) changes, and n. The bins have no explicit edges -- a lick goes to the nearest centre in SD units
(`combined_encoding`, `frozen_encoder.lick_landing_weights`) -- so the EDGE effect is reported as the share of pre-stroke
licks whose hard bin changes when the snapshot calibration is swapped for the regenerated one (each pre session with
its own leave-that-session-out row, as the encoder does).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["animal", "calib_for", "position"]
CENTRE, SD = ("ml", "ap", "down"), ("sd_ml", "sd_ap", "sd_down")


def compare(new: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Per (animal, calib_for, position) row present in both: d_<axis> (mm), d_<sd axis>, shift_sd (Euclidean centre
    shift in snapshot-SD units), d_n."""
    m = snap.merge(new, on=KEY, suffixes=("_snap", "_new"), how="outer", indicator=True)
    for c in (*CENTRE, *SD, "n"):
        m[f"d_{c}"] = m[f"{c}_new"] - m[f"{c}_snap"]
    m["shift_mm"] = np.sqrt(sum(m[f"d_{c}"] ** 2 for c in CENTRE))
    m["shift_sd"] = np.sqrt(sum((m[f"d_{c}"] / m[f"sd_{c}_snap"]) ** 2 for c in CENTRE))
    return m


def nearest_bin(X: np.ndarray, C: pd.DataFrame) -> np.ndarray:
    """Hard executed bin of each 3-D lick (rows of X = ml, ap, down) against one calibration set ``C`` (one animal,
    one calib_for): the nearest centre in SD units (the encoder's rule)."""
    C = C.dropna(subset=list(CENTRE))
    G, sc = C[list(CENTRE)].to_numpy(float), C[list(SD)].iloc[0].to_numpy(float)
    d = (((X / sc)[:, None, :] - (G / sc)[None]) ** 2).sum(2)
    return C.position.to_numpy(object)[d.argmin(1)]


def bin_changes(J: pd.DataFrame, new: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Per animal: pre licks (``J``, the regenerated set) and the share whose hard bin differs between ``snap`` and
    ``new``; each session binned with its own leave-out row when the table has one (else "default")."""
    rows = []
    for (an, sess), g in J.groupby(["animal", "session"]):
        X = g[list(CENTRE)].to_numpy(float)
        pick = {}
        for name, T in (("snap", snap), ("new", new)):
            T = T[T.animal == an]
            cf = sess if (T.calib_for == sess).any() else "default"
            pick[name] = nearest_bin(X, T[T.calib_for == cf])
        rows.append({"animal": an, "session": sess, "n_licks": len(g),
                     "n_changed": int((pick["snap"] != pick["new"]).sum())})
    R = pd.DataFrame(rows)
    A = R.groupby("animal")[["n_licks", "n_changed"]].sum().reset_index()
    A = pd.concat([A, pd.DataFrame([{"animal": "ALL", "n_licks": A.n_licks.sum(), "n_changed": A.n_changed.sum()}])])
    A["frac_changed"] = A.n_changed / A.n_licks
    return A


def summary_lines(name: str, M: pd.DataFrame, B: pd.DataFrame) -> list[str]:
    both = M[M._merge == "both"]
    out = [f"== {name} vs snapshot: {len(both)} rows matched, {int((M._merge == 'left_only').sum())} snapshot-only, "
           f"{int((M._merge == 'right_only').sum())} new-only"]
    for lab, sub in (("default (the encoder's rows)", both[both.calib_for == "default"]),
                     ("leave-one-session-out rows", both[both.calib_for != "default"])):
        if not len(sub):
            continue
        out.append(f"  {lab}: n = {len(sub)}")
        for c in (*CENTRE, "shift_mm", "shift_sd", *SD):
            v = sub[f"d_{c}"].abs() if c not in ("shift_mm", "shift_sd") else sub[c]
            unit = "SD" if c == "shift_sd" else "mm"
            out.append(f"    {('|d ' + c + '|') if c not in ('shift_mm', 'shift_sd') else c:<14} max {v.max():.3f} "
                       f"median {v.median():.3f} {unit}")
        out.append(f"    |d n| (licks)  max {sub.d_n.abs().max():.0f} median {sub.d_n.abs().median():.0f}")
        w = sub.loc[sub.shift_sd.idxmax()]
        out.append(f"    largest centre shift: {w.animal} {w.calib_for} {w.position} {w.shift_mm:.3f} mm = "
                   f"{w.shift_sd:.3f} SD (n {w.n_snap:.0f} -> {w.n_new:.0f})")
    out.append("  hard-bin changes of the pre licks (edge effect): "
               + "; ".join(f"{r.animal} {r.n_changed}/{r.n_licks} ({100 * r.frac_changed:.2f} %)"
                           for r in B.itertuples()))
    return out


def main(argv=None) -> int:
    cache = os.environ.get("WFIELD_INPUT_CACHE_POSE3D", "C:/Users/SabatiniLab/wfield_cache/pose3d_inputs")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="scratch folder (created); nothing else is written")
    ap.add_argument("--kin", default=f"{cache}/pose3d_kin_licks_snapshot_20261023.csv.gz")
    ap.add_argument("--snapshot", default=f"{cache}/exec_bins_calib_20261023.csv")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import exec_bins_calibration as ebc
    from wfield_local import trial_classes as tc
    out = Path(a.out)
    for p_ in (a.kin, a.snapshot):
        if Path(p_).resolve().parent == out.resolve():
            raise SystemExit("--out must not be the input folder")
    out.mkdir(parents=True, exist_ok=True)
    snap = pd.read_csv(a.snapshot)
    lines = [f"snapshot: {a.snapshot}", f"3-D licks: {a.kin}"]
    old = os.environ.get(tc.KIN_GATE_ENV)
    try:
        for gate in ("legacy", "working"):
            os.environ[tc.KIN_GATE_ENV] = gate
            if gate == "working":
                lines.append("working selection: " + tc.kin_gate_signature())
            J = ebc.pre_licks(a.kin)
            T = ebc.table(J)
            T.to_csv(out / f"calib_{gate}.csv", index=False, float_format="%.4f")
            M = compare(T, snap)
            M.drop(columns="_merge").to_csv(out / f"compare_{gate}.csv", index=False, float_format="%.4f")
            B = bin_changes(J, T, snap)
            B.to_csv(out / f"bin_changes_{gate}.csv", index=False, float_format="%.4f")
            lines += [f"{gate}: {J.session.nunique()} pre sessions, {len(J)} licks"] + summary_lines(gate, M, B)
    finally:
        if old is None:
            os.environ.pop(tc.KIN_GATE_ENV, None)
        else:
            os.environ[tc.KIN_GATE_ENV] = old
    txt = "\n".join(lines)
    (out / "SUMMARY.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
