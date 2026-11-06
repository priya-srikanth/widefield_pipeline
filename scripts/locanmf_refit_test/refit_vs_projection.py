"""LocaNMF refit vs fixed-footprint projection on the corrected (swapmask) data -- 2026-11-03.

Decides decision D1 of docs/status/rebuild_20261103/RECONCILIATION.md: must the cohort-1 rebuild REFIT per-session
LocaNMF on `hemo_meegkit_hpfit_lag_refitT_swapmask`, or can it PROJECT the corrected data onto the stored footprints?

Two stages, run separately so only one GPU job runs at a time:

  refit    one production per-session fit on the corrected SVTcorr, EXACTLY as `wfield_local.batch_locanmf` runs it
           (`python -m wfield_local.run_locanmf ... --maxrank 20 --loc-thresh 80 --r2-thresh 0.95 --device auto`,
           params from configs/defaults.yaml locanmf). Optional --seed wraps the same call after seeding numpy + torch
           (production sets no seed). Written to <out>/<label>/<tag>/, never to a stored locanmf_* dir.
  analyze  CPU only. Stored A/C (production dir, adopted data) vs projection (corrected data on the stored A, least
           squares over in-brain pixels, the same operator as `joint_locanmf.Basis._operator`) vs each refit:
           component counts, Hungarian-matched footprint r, matched time-course r, variance captured, and the
           within-session block-CV spout-position decoder (`locanmf_position_decoder._trial_features`, uncached,
           signal injected) at the precue and cue alignments. In memory only; one JSON per session.

    python scripts/locanmf_refit_test/refit_vs_projection.py refit   --label PS93_0826 --tag r1
    python scripts/locanmf_refit_test/refit_vs_projection.py analyze --label PS93_0826
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VARIANT = "meegkit_hpfit_lag_refitT_swapmask"
LOCAL_OUT = Path("C:/Users/SabatiniLab/locanmf_refit_test_20261103")
SHARE_OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/locanmf_refit_test_20261103")


def _session(label):
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    return next(s for s in SESSIONS if s["label"] == label)


def _allen_dir(s):
    return glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")[0]


# ------------------------------------------------------------------------------------------------ refit
def do_refit(label, tag, seed=None):
    from wfield_local import config
    s = _session(label)
    lp = config.defaults()["locanmf"]
    out = LOCAL_OUT / "refits" / label / tag
    if (out / f"{label}_locanmf_summary.json").exists():
        print(f"[refit] {label}/{tag} exists, skipping", flush=True)
        return
    svt = config.svtcorr_path(s["mc"], VARIANT)
    stored = Path(config.locanmf_dir(s["mc"]))
    for p in (Path(svt), out):
        # belt and braces: never write into a stored dir
        assert not str(p).startswith(str(stored)), p
    out.mkdir(parents=True, exist_ok=True)
    args = ["--allen-dir", _allen_dir(s), "--label", label, "--output", str(out), "--mode", "locanmf",
            "--maxrank", str(lp["maxrank"]), "--loc-thresh", str(lp["loc_thresh"]),
            "--r2-thresh", str(lp["r2_thresh"]), "--device", "auto", "--svt", svt]
    if seed is None:
        cmd = [sys.executable, "-u", "-m", "wfield_local.run_locanmf"] + args
    else:
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("seeded_run_locanmf.py")), str(seed)] + args
    t0 = time.time()
    with open(out / "run.log", "w", buffering=1) as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    (out / "timing.json").write_text(json.dumps({"label": label, "tag": tag, "seed": seed, "rc": rc,
                                                 "seconds": round(dt, 1), "svt": svt, "cmd": cmd,
                                                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"[refit] {label}/{tag} seed={seed} rc={rc} in {dt:.0f}s", flush=True)
    if rc != 0:
        raise SystemExit(rc)


# ------------------------------------------------------------------------------------------------ metrics
def _flat_A(A):
    A = np.asarray(A, dtype=np.float32)
    return A.reshape(-1, A.shape[2])


def project_C(A, U_flat, V):
    """Least-squares C on fixed footprints, restricted to in-brain pixels (as joint_locanmf.Basis._operator)."""
    Araw = _flat_A(A)
    inside = np.isfinite(Araw).any(axis=1)
    Ai = np.nan_to_num(Araw[inside]).astype(np.float64)
    G = Ai.T @ Ai
    op = np.linalg.pinv(G) @ (Ai.T @ np.nan_to_num(U_flat[inside]).astype(np.float64))   # (ncomp, K)
    return (op @ V.astype(np.float64)), inside


def captured(A, C, U_flat, V, inside):
    """Fraction of the in-brain data energy explained by A @ C (1 - ||UV - AC||^2 / ||UV||^2), in K-space."""
    Ai = np.nan_to_num(_flat_A(A)[inside]).astype(np.float64)
    Ui = np.nan_to_num(U_flat[inside]).astype(np.float64)
    V = V.astype(np.float64); C = np.asarray(C, dtype=np.float64)
    uv = float(np.sum((Ui.T @ Ui) * (V @ V.T)))
    ac = float(np.sum((Ai.T @ Ai) * (C @ C.T)))
    cross = float(np.sum((Ui.T @ Ai) * (V @ C.T)))
    return 1.0 - (uv - 2 * cross + ac) / uv


def _corr_cols(X, Y):
    """Pearson r between every column of X (n, a) and every column of Y (n, b)."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    X = X / np.maximum(np.linalg.norm(X, axis=0), 1e-30)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=0), 1e-30)
    return X.T @ Y


def match(A1, C1, A2, C2, inside):
    """Hungarian match of footprints (spatial r over in-brain pixels), then time-course r of matched pairs."""
    from scipy.optimize import linear_sum_assignment
    a1 = np.nan_to_num(_flat_A(A1)[inside]).astype(np.float64)
    a2 = np.nan_to_num(_flat_A(A2)[inside]).astype(np.float64)
    R = _corr_cols(a1, a2)
    ri, ci = linear_sum_assignment(-R)
    rs = R[ri, ci]
    rc = np.array([np.corrcoef(C1[i], C2[j])[0, 1] for i, j in zip(ri, ci)])
    n1, n2 = a1.shape[1], a2.shape[1]
    return {"n1": int(n1), "n2": int(n2), "n_matched": int(len(ri)), "n_unmatched": int(abs(n1 - n2)),
            "A_r_median": float(np.median(rs)), "A_r_min": float(rs.min()),
            "A_r_p10": float(np.percentile(rs, 10)),
            "n_A_r_lt_0.5": int((rs < 0.5).sum()), "n_A_r_lt_0.8": int((rs < 0.8).sum()),
            "n_A_r_ge_0.95": int((rs >= 0.95).sum()),
            "C_r_median": float(np.median(rc)), "C_r_min": float(rc.min()),
            "C_r_p10": float(np.percentile(rc, 10)),
            "C_r_median_where_A_ge_0.95": float(np.median(rc[rs >= 0.95])) if (rs >= 0.95).any() else None}


def rowwise_r(C1, C2):
    r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(C1, C2)])
    return {"median": float(np.median(r)), "min": float(r.min()), "p10": float(np.percentile(r, 10))}


# ------------------------------------------------------------------------------------------------ decoding
def decode(s, signal, regions):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from wfield_local import config, lp_licks
    from wfield_local.locanmf_position_decoder import _trial_features
    dp = config.defaults()["decode"]
    out = {}
    for align in ("precue", "cue"):
        args = SimpleNamespace(align=align, post_s=lp_licks.post_s_for(align), pre_s=1.0, fs=31.23,
                               max_rt=float(dp["max_rt_s"]), baseline=dp["baseline"], cv=dp["cv"],
                               source="locanmf", bins=None)
        X, y, g, _Xn, _yn, _fr = _trial_features(s, args, signal=signal, feat_region=regions)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        ng = min(5, int(np.unique(g).size))
        pred = cross_val_predict(clf, X, y, cv=GroupKFold(ng), groups=g)
        out[align] = {"acc": float(accuracy_score(y, pred)), "bal_acc": float(balanced_accuracy_score(y, pred)),
                      "n_trials": int(len(y)), "n_feat": int(X.shape[1])}
    return out


def do_analyze(label, tags):
    from wfield_local import config
    from wfield_local.locanmf_crossanimal_dff import _footprint_scale
    s = _session(label)
    t0 = time.time()
    ld = config.locanmf_dir(s["mc"])
    A0 = np.load(f"{ld}/{label}_locanmf_A.npy"); C0 = np.load(f"{ld}/{label}_locanmf_C.npy").astype(np.float64)
    reg0 = np.load(f"{ld}/{label}_locanmf_regions.npy")
    stored_summary = json.loads(Path(f"{ld}/{label}_locanmf_summary.json").read_text())
    U = np.load(f"{_allen_dir(s)}/U_atlas.npy")
    Uf = U.reshape(-1, U.shape[2]); del U
    V_new = np.load(config.svtcorr_path(s["mc"], VARIANT))
    V_old = np.load(stored_summary["svt"]) if Path(stored_summary["svt"]).exists() else None
    print(f"[analyze] {label}: stored {A0.shape[2]} comps, U {Uf.shape}, V {V_new.shape}, "
          f"stored svt {stored_summary['svt']}", flush=True)

    Cp, inside = project_C(A0, Uf, V_new)
    res = {"label": label, "variant": VARIANT, "stored_dir": ld, "stored_svt": stored_summary["svt"],
           "stored_params": stored_summary.get("params"), "n_frames": int(V_new.shape[1]),
           "n_stored": int(A0.shape[2])}
    # sanity: the same projection on the ADOPTED data reproduces the stored C?
    if V_old is not None:
        Cp_old, _ = project_C(A0, Uf, V_old)
        res["proj_adopted_vs_stored_C_r"] = rowwise_r(Cp_old, C0)
        res["captured_stored_on_adopted"] = captured(A0, C0, Uf, V_old, inside)
        del Cp_old
    res["proj_vs_stored_C_r"] = rowwise_r(Cp, C0)
    res["captured_proj_on_corrected"] = captured(A0, Cp, Uf, V_new, inside)

    sig0 = _footprint_scale(A0, A0.shape[2])[:, None] * C0
    sigp = _footprint_scale(A0, A0.shape[2])[:, None] * Cp
    res["decode"] = {"stored": decode(s, sig0, reg0), "projection": decode(s, sigp, reg0)}
    del sig0

    refits = {}
    for tag in tags:
        rd = LOCAL_OUT / "refits" / label / tag
        if not (rd / f"{label}_locanmf_summary.json").exists():
            continue
        A1 = np.load(rd / f"{label}_locanmf_A.npy"); C1 = np.load(rd / f"{label}_locanmf_C.npy").astype(np.float64)
        reg1 = np.load(rd / f"{label}_locanmf_regions.npy")
        tm = json.loads((rd / "timing.json").read_text())
        inside1 = np.isfinite(_flat_A(A1)).any(axis=1) | inside
        refits[tag] = (A1, C1)
        res.setdefault("refits", {})[tag] = {
            "n_comp": int(A1.shape[2]), "seconds": tm["seconds"], "seed": tm.get("seed"),
            "vs_stored": match(A0, C0, A1, C1, inside1),
            "vs_projection": match(A0, Cp, A1, C1, inside1),
            "captured_on_corrected": captured(A1, C1, Uf, V_new, inside1),
            "decode": decode(s, _footprint_scale(A1, A1.shape[2])[:, None] * C1, reg1)}
        print(f"[analyze] {label}/{tag}: {A1.shape[2]} comps  "
              f"A r med {res['refits'][tag]['vs_stored']['A_r_median']:.3f}", flush=True)
    # refit vs refit (same-machine stability) for every pair
    tl = list(refits)
    for i in range(len(tl)):
        for j in range(i + 1, len(tl)):
            (Aa, Ca), (Ab, Cb) = refits[tl[i]], refits[tl[j]]
            ins = np.isfinite(_flat_A(Aa)).any(axis=1) | np.isfinite(_flat_A(Ab)).any(axis=1)
            res.setdefault("refit_pairs", {})[f"{tl[i]}_vs_{tl[j]}"] = {
                **match(Aa, Ca, Ab, Cb, ins),
                "identical_A": bool(Aa.shape == Ab.shape and np.array_equal(np.nan_to_num(Aa), np.nan_to_num(Ab))),
                "max_abs_dA": float(np.nanmax(np.abs(Aa - Ab))) if Aa.shape == Ab.shape else None}
    res["analyze_seconds"] = round(time.time() - t0, 1)
    for root in (LOCAL_OUT / "results", SHARE_OUT / "results"):
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{label}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k in ("decode", "proj_vs_stored_C_r")}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("refit", "analyze"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tags", nargs="*", default=["r1", "r2", "s0a", "s0b"])
    a = ap.parse_args()
    assert os.environ.get("WIDEFIELD_SESSION_CACHE", "").endswith("locanmftest"), "set the test session cache"
    if a.stage == "refit":
        do_refit(a.label, a.tag, a.seed)
    else:
        do_analyze(a.label, a.tags)


if __name__ == "__main__":
    main()
