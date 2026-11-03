"""CURBD PILOT (Priya 10-30): region -> region currents from a data-constrained RNN (Perich et al. 2020; core in
`wfield_local/curbd_core.py`, GPL-3.0-derived) fitted to widefield ENCODING-MODEL RESIDUALS, Allen + ROI-v8 regions,
both hemispheres. PILOT, ONE ANIMAL (PS94: overt deficit, plenty of engaged trials). Stability is the gate.

    python scripts/curbd_pilot.py prep PS94_0807 PS94_0811 PS94_0812 PS94_0910
    python scripts/curbd_pilot.py fit [--sessions ...] [--jobs 4]
    python scripts/curbd_pilot.py summarise
    python scripts/curbd_pilot.py plot            # == --plot-only: figures from the saved tables / npz only

REGIONS (18): SSp-n, SSp-m, SSs, SSp-ul, tjM1, ALM, MOp_rest, MOs_rest, RSPd (posterior control; VISp_left has 0 px on the stat mask in this grid) x left (ipsilesional)
/ right. ROI v8 exactly as `encoder_split_rows.masks` (iti_licks.roi_masks restricted to the encoder's 16-px stat mask:
ALM + MOs_rest = MOs, tjM1 + MOp_rest = MOp); other areas = Allen area on the stat mask.

UNITS. "multi" (primary): each region split into K_UNITS spatial clusters (k-means on pixel row / column, seed 0); a
unit's trace = the mean of the SVD reconstruction over its pixels (V @ mean U). "one": one unit per region (the region
mean = the encoder's area target).

INPUT = ENCODING-MODEL RESIDUALS. Per-frame residuals of the per-session combined encoder were NOT saved (only
residual second moments), so the encoder's FULL design is refitted here on the unit traces: `combined_encoding.
raw_blocks` variables of GROUP_OF (task: cue, arrival, pos_enl, pos_cue; instructed: licks, exec_angle, tongue, jaw;
uninstructed: nose, running), z-scored on the engaged-trial rows (strobe - 0.5 .. stop + 1 s, the encoder's rows),
slow drift (cosines, period >= 600 s) projected out of design and data, ridge with that session's OWN per-group
penalties from the encoder json, 10-fold contiguous-trial-block CV -> HELD-OUT residuals (no orthogonalisation: the full
model's prediction does not depend on it). Then 3-frame block means within each contiguous run (31.25 -> 10.4 Hz).

FIT. Engaged rows only, concatenated; the state is RESET to the data at every run start and every RESET_S seconds
(a multi-step-ahead fit; long residual stretches are stochastic and cannot be generated autonomously). tau 0.2 s,
dt_factor 4, g 1.5, N_TRAIN passes + 1 free pass. Currents from the free pass.

STABILITY (criterion written BEFORE any real fit, docs/status/CURBD_PILOT_PREREG_20261030.md): see that file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/curbd_pilot")
ENC = OUT.parent / "movement_first" / "sessions"
AREAS = ("SSp-n", "SSp-m", "SSs", "SSp-ul", "tjM1", "ALM", "MOp_rest", "MOs_rest", "RSPd")
HEMIS = ("left", "right")                       # left = ipsilesional
REGIONS = [f"{a}_{h}" for h in HEMIS for a in AREAS]
K_UNITS = 4
DS = 3                                          # frames per block mean
RESET_S = 2.0
N_TRAIN = 20
PRE = ("PS94_0807", "PS94_0811", "PS94_0812")
CHRONIC = "PS94_0910"
SEEDS = (0, 1, 2)
SENSORY = ("SSp-n", "SSp-m", "SSs")
MOTOR = ("tjM1", "ALM", "MOp_rest")


def short(r: str) -> str:
    return r.replace("_left", " ipsi").replace("_right", " contra").replace("_rest", "-r")


# ------------------------------------------------------------------------------------------------------------- prep
def region_masks(parea, pstat, names, flat, shape) -> dict:
    import iti_licks as il
    roi = il.roi_masks(parea, pstat, names, flat, shape)
    code = {v: k for k, v in names.items()}
    out = {}
    for h in HEMIS:
        for src, dst in il.SPLIT_PAIRS:
            par = (parea == code[f"{src}_{h}"]) & pstat
            sub = roi[f"{dst}_{h}"] & par
            out[f"{dst}_{h}"], out[f"{src}_rest_{h}"] = sub, par & ~sub
        for a in ("SSp-n", "SSp-m", "SSs", "SSp-ul", "RSPd"):
            out[f"{a}_{h}"] = (parea == code[f"{a}_{h}"]) & pstat
    return out


def prep(label: str) -> Path:
    from sklearn.cluster import KMeans

    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.paths import PathResolver
    t0 = time.time()
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    W_ = int(shape[1])
    rr, cc = flat // W_, flat % W_
    masks = region_masks(parea, pstat, names, flat, shape)
    wm, wo, reg_m, cent = [], [], [], []
    for i, r in enumerate(REGIONS):
        ix = np.flatnonzero(masks[r])
        wo.append(Ub[ix].mean(0))
        lab = KMeans(K_UNITS, n_init=10, random_state=0).fit_predict(np.c_[rr[ix], cc[ix]].astype(float))
        for k in range(K_UNITS):
            jj = ix[lab == k]
            wm.append(Ub[jj].mean(0))
            reg_m.append(i)
            cent.append((rr[jj].mean(), cc[jj].mean(), len(jj)))
    Wm, Wo = np.stack(wm, 1), np.stack(wo, 1)                        # (K, units)
    # unit-pixel label image for the figure
    lab_img = np.full(int(shape[0]) * W_, -1, np.int16)
    u = 0
    for r in REGIONS:
        ix = np.flatnonzero(masks[r])
        lab = KMeans(K_UNITS, n_init=10, random_state=0).fit_predict(np.c_[rr[ix], cc[ix]].astype(float))
        lab_img[flat[ix]] = u + lab
        u += K_UNITS
    # behaviour + design (the per-session encoder's rows and full design)
    animal, date = config.animal_of(label), "2026" + label.split("_")[-1]
    B = ce.load_behaviour(animal, date, PathResolver())
    T = B.trials
    E = T[T.engaged & np.isfinite(T.cue_s)].reset_index(drop=True)
    st = np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0)
    rows = np.zeros(len(ft), bool)
    tr_of = np.full(len(ft), -1)
    for k, (a, b_) in enumerate(zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0)):
        m = (ft >= a) & (ft <= b_)
        rows |= m
        tr_of[m] = k
    ridx = np.flatnonzero(rows)
    blk = np.array_split(np.arange(len(E)), ce.FOLDS)
    f_of_trial = np.zeros(len(E), int)
    for k, bl in enumerate(blk):
        f_of_trial[bl] = k
    fold = f_of_trial[tr_of[ridx]]
    raw = {v: ce._z(X, ridx)[ridx] for v, X in ce.raw_blocks(B, ft).items() if v in ce.GROUP_OF}
    alphas = json.loads((ENC / f"{label}.json").read_text())["alphas"]
    X = np.hstack([raw[v] for v in ce.GROUP_OF]).astype(np.float64)
    pen = np.concatenate([np.full(raw[v].shape[1], alphas[ce.GROUP_OF[v]]) for v in ce.GROUP_OF])
    Q = np.linalg.qr(ce._slow_basis(ft[ridx], ce.DRIFT_MIN_PERIOD_S))[0]
    X -= Q @ (Q.T @ X)
    Ys = {}
    for nm, W in (("multi", Wm), ("one", Wo)):
        Y = (V[ridx].astype(np.float64) @ W.astype(np.float64))
        Y -= Q @ (Q.T @ Y)
        Ys[nm] = Y
    Yall = np.hstack([Ys["multi"], Ys["one"]])
    Xa = np.c_[np.ones(len(X)), X]
    lam = np.diag(np.r_[0.0, pen])
    G = Xa.T @ Xa
    bvec = Xa.T @ Yall
    pred = np.zeros_like(Yall)
    for k in range(ce.FOLDS):
        m = fold == k
        Gk, bk = G - Xa[m].T @ Xa[m], bvec - Xa[m].T @ Yall[m]
        beta = np.linalg.solve(Gk + lam + 1e-8 * np.eye(len(lam)), bk)
        pred[m] = Xa[m] @ beta
    res = Yall - pred
    r2 = 1 - (res ** 2).sum(0) / ((Yall - Yall.mean(0)) ** 2).sum(0)
    # contiguous runs of engaged rows, 3-frame block means
    br = np.flatnonzero(np.diff(ridx) > 1) + 1
    runs = np.split(np.arange(len(ridx)), br)
    seg, starts, tds, n0 = [], [], [], 0
    for rn in runs:
        nb = len(rn) // DS
        if nb < 5:
            continue
        sl = rn[: nb * DS]
        seg.append(res[sl].reshape(nb, DS, -1).mean(1))
        tds.append(ft[ridx[sl]].reshape(nb, DS).mean(1))
        starts.append(n0)
        n0 += nb
    Rds = np.vstack(seg)
    fs = 1.0 / np.median(np.diff(ft))
    nm_ = Ys["multi"].shape[1]
    out = OUT / "data" / f"{label}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, res_multi=Rds[:, :nm_].T.astype(np.float32), res_one=Rds[:, nm_:].T.astype(np.float32),
                        t=np.concatenate(tds), run_starts=np.array(starts), region_multi=np.array(reg_m),
                        region_one=np.arange(len(REGIONS)), regions=np.array(REGIONS), centroids=np.array(cent),
                        lab_img=lab_img, shape=np.array(shape), enc_r2_multi=r2[:nm_], enc_r2_one=r2[nm_:],
                        dt=DS / fs, n_trials=len(E))
    print(f"{label}: {len(E)} engaged trials, {len(ridx)} rows, {len(starts)} runs, {n0} samples at {fs / DS:.2f} Hz; "
          f"encoder held-out R2 (units) median {np.median(r2[:nm_]):.3f}; {time.time() - t0:.0f} s", flush=True)
    return out


# -------------------------------------------------------------------------------------------------------------- fit
def load_data(label: str, variant: str) -> dict:
    d = np.load(OUT / "data" / f"{label}.npz")
    return {"Y": d[f"res_{variant}"].astype(np.float64), "region": d[f"region_{variant}"], "dt": float(d["dt"]),
            "starts": d["run_starts"]}


def resets(starts, T, dt) -> np.ndarray:
    step = max(1, int(round(RESET_S / dt)))
    r = set()
    bounds = list(starts) + [T]
    for a, b in zip(bounds[:-1], bounds[1:]):
        r.update(range(a, b, step))
    return np.array(sorted(r))


def part_slices(starts, T, part):
    """'full' = all samples; 'h1' / 'h2' = first / second half of the session, cut at the run boundary nearest T/2."""
    if part == "full":
        return 0, T
    cut = min(list(starts) + [T], key=lambda s: abs(s - T / 2))
    return (0, cut) if part == "h1" else (cut, T)


def baselines(A, rs) -> dict:
    """pVar of trivial predictors under the same resets: HOLD (rate stays at the reset value) and DECAY (J = 0)."""
    var = A.var()
    hold = np.empty_like(A)
    rs_ = list(rs) + [A.shape[1]]
    for a, b in zip(rs_[:-1], rs_[1:]):
        hold[:, a:b] = A[:, [a]]
    return {"hold": float(1 - np.mean((A - hold) ** 2) / var)}


def fit_job(job) -> dict:
    from wfield_local import curbd_core as cc
    from wfield_local.parallel import pin_blas
    pin_blas(1)
    label, variant, part, seed = job[:4]
    g = job[4] if len(job) > 4 else 1.5                                      # EXPLORATORY g sweep (post-gate)
    fdir = fits_dir(g)
    D = load_data(label, variant)
    A_all, _ = cc.scale_data(D["Y"])
    T = A_all.shape[1]
    a, b = part_slices(D["starts"], T, part)
    A = A_all[:, a:b]
    st = [s - a for s in D["starts"] if a <= s < b]
    rs = resets(st, A.shape[1], D["dt"])
    t0 = time.perf_counter()
    out = cc.train(A, D["dt"], n_train=N_TRAIN, n_free=1, reset_idx=rs, seed=seed, g=g)
    secs = time.perf_counter() - t0
    nreg = len(REGIONS)
    cur = cc.currents(out["J"], out["R"], D["region"], nreg)
    J0run = cc.free_run(out["J0"], A, D["dt"], reset_idx=rs, seed=seed, g=g)       # untrained J, same resets
    res = {"label": label, "variant": variant, "part": part, "seed": seed, "n_units": A.shape[0], "T": A.shape[1],
           "seconds": secs, "pvar_train": out["pvar_train"], "pvar_free": out["pvar_free"][-1],
           "pvar_J0": J0run["pvar_free"][-1], **baselines(A, rs)}
    # per-region pVar (free pass)
    res["pvar_region"] = [float(1 - np.mean((A[D["region"] == i] - out["R"][D["region"] == i]) ** 2)
                                / A[D["region"] == i].var()) for i in range(nreg)]
    if part != "full":                                                        # held-out: free-run on the other half
        a2, b2 = part_slices(D["starts"], T, "h2" if part == "h1" else "h1")
        A2 = A_all[:, a2:b2]
        st2 = [s - a2 for s in D["starts"] if a2 <= s < b2]
        rs2 = resets(st2, A2.shape[1], D["dt"])
        ho = cc.free_run(out["J"], A2, D["dt"], reset_idx=rs2, seed=seed, g=g)
        res["pvar_heldout"] = ho["pvar_free"][-1]
        res["hold_heldout"] = baselines(A2, rs2)["hold"]
    f = fdir / f"{label}_{variant}_{part}_s{seed}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, J=out["J"].astype(np.float32), J0=out["J0"].astype(np.float32), region=D["region"], **{k: v for k, v in cur.items()},
                        meta=json.dumps(res))
    print(f"{f.name}: pVar train {out['pvar_train'][-1]:.3f} free {res['pvar_free']:.3f} hold {res['hold']:.3f} "
          f"J0 {res['pvar_J0']:.3f} heldout {res.get('pvar_heldout', float('nan')):.3f} {secs:.0f} s", flush=True)
    return res


def jobs_for(labels, variants=("multi", "one")) -> list:
    out = []
    for lab in labels:
        for v in variants:
            out += [(lab, v, "full", s) for s in SEEDS]
            out += [(lab, v, p, 0) for p in ("h1", "h2")]
    return out


# -------------------------------------------------------------------------------------------------------- summarise
def fits_dir(g: float = 1.5) -> Path:
    """Primary fits (g = 1.5, pre-stated) in fits/; the post-gate EXPLORATORY g sweep in fits_g<g>/."""
    return OUT / ("fits" if g == 1.5 else f"fits_g{g:g}")


def load_fit(label, variant, part, seed, g: float = 1.5) -> dict:
    d = np.load(fits_dir(g) / f"{label}_{variant}_{part}_s{seed}.npz")
    return {"mean_abs": d["mean_abs"], "mean_signed": d["mean_signed"], "meta": json.loads(str(d["meta"])),
            "J": d["J"], "J0": d["J0"] if "J0" in d else None, "region": d["region"]}


def offdiag(M) -> np.ndarray:
    return M[~np.eye(len(M), dtype=bool)]


def dcenter(M) -> np.ndarray:
    """Off-diagonal entries with source (column) and target (row) main effects removed (NaN diagonal, row / column
    means over off-diagonal entries; 3 sweeps) -> the PAIR-SPECIFIC structure."""
    X = M.astype(float).copy()
    np.fill_diagonal(X, np.nan)
    for _ in range(3):
        X = X - np.nanmean(X, 1, keepdims=True)
        X = X - np.nanmean(X, 0, keepdims=True)
    return X[~np.isnan(X)]


def r_(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def compare(M1, M2) -> dict:
    return {"r": r_(offdiag(M1), offdiag(M2)), "r_pair": r_(dcenter(M1), dcenter(M2)),
            "r_signed": float("nan")}


def summarise(labels) -> dict:
    import itertools

    import pandas as pd
    rows, fitrows = [], []
    for v in ("multi", "one"):
        for lab in labels:
            fs = {s: load_fit(lab, v, "full", s) for s in SEEDS}
            hs = {p: load_fit(lab, v, p, 0) for p in ("h1", "h2")}
            for f in list(fs.values()) + list(hs.values()):
                m = f["meta"]
                fitrows.append({k: m[k] for k in ("label", "variant", "part", "seed", "n_units", "T", "seconds",
                                                   "pvar_free", "pvar_J0", "hold")}
                               | {"pvar_train_last": m["pvar_train"][-1], "pvar_heldout": m.get("pvar_heldout"),
                                  "hold_heldout": m.get("hold_heldout"),
                                  "pvar_region_min": min(m["pvar_region"]), "pvar_region_med":
                                  float(np.median(m["pvar_region"]))})
            for s1, s2 in itertools.combinations(SEEDS, 2):
                for q in ("mean_abs", "mean_signed"):
                    c = compare(fs[s1][q], fs[s2][q])
                    rows.append({"variant": v, "kind": "seeds", "a": f"{lab}_s{s1}", "b": f"{lab}_s{s2}", "q": q,
                                 "r": c["r"], "r_pair": c["r_pair"]})
            for q in ("mean_abs", "mean_signed"):
                c = compare(hs["h1"][q], hs["h2"][q])
                rows.append({"variant": v, "kind": "halves", "a": f"{lab}_h1", "b": f"{lab}_h2", "q": q,
                             "r": c["r"], "r_pair": c["r_pair"]})
        pre = [x for x in labels if x in PRE]
        for l1, l2 in itertools.combinations(pre, 2):
            for q in ("mean_abs", "mean_signed"):
                M1 = np.mean([load_fit(l1, v, "full", s)[q] for s in SEEDS], 0)
                M2 = np.mean([load_fit(l2, v, "full", s)[q] for s in SEEDS], 0)
                c = compare(M1, M2)
                rows.append({"variant": v, "kind": "sessions", "a": l1, "b": l2, "q": q, "r": c["r"],
                             "r_pair": c["r_pair"]})
        # multi vs one
    for lab in labels:
        for q in ("mean_abs", "mean_signed"):
            M1 = np.mean([load_fit(lab, "multi", "full", s)[q] for s in SEEDS], 0)
            M2 = np.mean([load_fit(lab, "one", "full", s)[q] for s in SEEDS], 0)
            c = compare(M1, M2)
            rows.append({"variant": "multi_vs_one", "kind": "variant", "a": lab, "b": lab, "q": q, "r": c["r"],
                         "r_pair": c["r_pair"]})
    S, F = pd.DataFrame(rows), pd.DataFrame(fitrows)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / "tables" / "stability_pairs.csv", index=False)
    F.to_csv(OUT / "tables" / "fits.csv", index=False)
    agg = S.groupby(["variant", "kind", "q"])[["r", "r_pair"]].agg(["median", "min"]).round(3)
    agg.to_csv(OUT / "tables" / "stability_summary.csv")
    print(F.groupby(["label", "variant", "part"])[["pvar_free", "hold", "pvar_J0", "pvar_heldout",
                                                   "hold_heldout", "seconds"]].mean().round(3).to_string())
    print(agg.to_string())
    return {"S": S, "F": F}


def sens_motor(labels, variant="multi") -> "pandas.DataFrame":  # noqa: F821
    """Mean |current| and signed current for sensory -> motor pairs (within / across hemisphere), per session."""
    import pandas as pd
    idx = {r: i for i, r in enumerate(REGIONS)}
    rows = []
    for lab in labels:
        for q in ("mean_abs", "mean_signed"):
            Ms = [load_fit(lab, variant, "full", s)[q] for s in SEEDS]
            M = np.mean(Ms, 0)
            sd = np.std(Ms, 0)
            for s_ in SENSORY:
                for m_ in MOTOR:
                    for hs in HEMIS:
                        for ht in HEMIS:
                            a, b = idx[f"{m_}_{ht}"], idx[f"{s_}_{hs}"]
                            rows.append({"label": lab, "q": q, "source": f"{s_}_{hs}", "target": f"{m_}_{ht}",
                                         "within": hs == ht, "value": M[a, b], "seed_sd": sd[a, b]})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "tables" / f"sensory_motor_{variant}.csv", index=False)
    return D


# ------------------------------------------------------------------------------------------------- explore (post-gate)
G_SWEEP = (0.0, 0.5, 1.0)


def explore_stability(labels=PRE, gs=(1.5,) + G_SWEEP, variant="multi") -> "pandas.DataFrame":  # noqa: F821
    """EXPLORATORY, run AFTER the pre-stated gate failed at g = 1.5: does a weaker initial J (g) make the currents
    stable? Same fits / metrics as `summarise`, per g. Not a gate; a pointer for what to change."""
    import itertools

    import pandas as pd
    rows = []
    for g in gs:
        for lab in labels:
            fs = [load_fit(lab, variant, "full", s, g) for s in SEEDS]
            h = [load_fit(lab, variant, p, 0, g) for p in ("h1", "h2")]
            pv = np.mean([f["meta"]["pvar_free"] for f in fs])
            ho = np.mean([x["meta"]["pvar_heldout"] for x in h])
            for kind, pairs in (("seeds", list(itertools.combinations(fs, 2))), ("halves", [tuple(h)])):
                for a, b in pairs:
                    c = compare(a["mean_abs"], b["mean_abs"])
                    rows.append({"g": g, "label": lab, "kind": kind, "r": c["r"], "r_pair": c["r_pair"],
                                 "pvar_free": pv, "pvar_heldout": ho})
        for l1, l2 in itertools.combinations(labels, 2):
            M1 = np.mean([load_fit(l1, variant, "full", s, g)["mean_abs"] for s in SEEDS], 0)
            M2 = np.mean([load_fit(l2, variant, "full", s, g)["mean_abs"] for s in SEEDS], 0)
            c = compare(M1, M2)
            rows.append({"g": g, "label": f"{l1}|{l2}", "kind": "sessions", "r": c["r"], "r_pair": c["r_pair"]})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "tables" / "explore_g_stability.csv", index=False)
    print(D.groupby(["g", "kind"])[["r", "r_pair", "pvar_free", "pvar_heldout"]].median().round(3).to_string())
    return D


def var_matrix(label, part, variant="multi", alpha=None) -> tuple:
    """EXPLORATORY comparator: ridge VAR(1) on the same scaled residual samples (pairs within a run only),
    x_{t+1} - x_t = B x_t; region current = mean over target units and time of |B[a, b] x_b(t)| (the CURBD summary
    applied to the linear model). alpha = 1e-1 x n_samples unless given. Returns (matrix, one-step R2 of the change)."""
    from wfield_local import curbd_core as cc
    D = load_data(label, variant)
    A, _ = cc.scale_data(D["Y"])
    a, b = part_slices(D["starts"], A.shape[1], part)
    st = set(s_ for s_ in D["starts"])
    t = np.arange(a, b - 1)
    t = t[[(x + 1) not in st for x in t]]
    X, Y = A[:, t].T, (A[:, t + 1] - A[:, t]).T
    lam = (0.1 * len(t)) if alpha is None else alpha
    B = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y).T
    r2 = 1 - ((Y - X @ B.T) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    reg = D["region"]
    n = len(REGIONS)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.abs(B[np.ix_(reg == i, reg == j)] @ X[:, reg == j].T).mean()
    return M, float(r2)


def explore_var(labels=PRE) -> "pandas.DataFrame":  # noqa: F821
    import itertools

    import pandas as pd
    rows = []
    for v in ("multi", "one"):
        full = {lab: var_matrix(lab, "full", v) for lab in labels}
        for lab in labels:
            M1, M2 = var_matrix(lab, "h1", v)[0], var_matrix(lab, "h2", v)[0]
            c = compare(M1, M2)
            rows.append({"variant": v, "kind": "halves", "a": lab, "r": c["r"], "r_pair": c["r_pair"],
                         "r2_step": full[lab][1]})
        for l1, l2 in itertools.combinations(labels, 2):
            c = compare(full[l1][0], full[l2][0])
            rows.append({"variant": v, "kind": "sessions", "a": f"{l1}|{l2}", "r": c["r"], "r_pair": c["r_pair"]})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "tables" / "explore_var1_stability.csv", index=False)
    print(D.to_string())
    return D


# ------------------------------------------------------------------------------------------------------------- plot
def plot():
    import matplotlib.pyplot as plt
    import pandas as pd
    F = pd.read_csv(OUT / "tables" / "fits.csv")
    S = pd.read_csv(OUT / "tables" / "stability_pairs.csv")
    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    tag = "PILOT, PS94 only (n = 1 animal), PROVISIONAL"
    # FIG 1: regions / units + fit quality
    d = np.load(OUT / "data" / f"{PRE[0]}.npz")
    sh = tuple(int(x) for x in d["shape"])
    img = d["lab_img"].reshape(sh).astype(float)
    img[img < 0] = np.nan
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    cm = plt.get_cmap("tab20")
    reg = (img // K_UNITS)
    col = np.full(sh + (4,), 1.0)
    ok = np.isfinite(img)
    shade = 0.55 + 0.45 * ((img[ok] % K_UNITS) / max(K_UNITS - 1, 1))
    col[ok] = cm((reg[ok] % len(AREAS)) / 20.0 * 2)
    col[ok, :3] *= shade[:, None]
    ax[0].imshow(col)
    for i, r in enumerate(REGIONS):
        yy, xx = np.nonzero(reg == i)
        if len(yy):
            ax[0].text(xx.mean(), yy.mean(), short(r).split()[0], fontsize=6, ha="center", va="center")
    ax[0].set_axis_off()
    ax[0].set_title(f"1a  18 regions x {K_UNITS} spatial units (ROI v8)\nimage left = ipsilesional", fontsize=9)
    for j, (v, mk) in enumerate((("multi", "o"), ("one", "s"))):
        G = F[F.variant == v]
        for k, lab in enumerate(sorted(G.label.unique())):
            g = G[G.label == lab]
            x0 = k * 4 + j * 0.35
            ax[1].scatter([x0] * len(g[g.part == "full"]), g[g.part == "full"].pvar_free, marker=mk, c="k", s=14,
                          label=f"{v}: train (free pass)" if k == 0 else None)
            hh = g[g.part != "full"]
            ax[1].scatter([x0 + 1] * len(hh), hh.pvar_heldout, marker=mk, c="tab:red", s=14,
                          label=f"{v}: held-out half" if k == 0 else None)
            ax[1].scatter([x0 + 2] * len(g[g.part == "full"]), g[g.part == "full"].hold, marker=mk, c="0.6", s=14,
                          label=f"{v}: hold-at-reset baseline" if k == 0 else None)
    labs = sorted(F.label.unique())
    for k, lab in enumerate(labs):                     # J = 0 (pure decay to zero) baseline, from the g = 0 fits' J0
        f0 = fits_dir(0.0) / f"{lab}_multi_full_s0.npz"
        if f0.exists():
            ax[1].hlines(load_fit(lab, "multi", "full", 0, 0.0)["meta"]["pvar_J0"], k * 4 - 0.3, k * 4 + 2.6,
                         colors="tab:green", lw=1.2, label="J = 0 decay baseline" if k == 0 else None)
    ax[1].set_xticks([k * 4 + 1 for k in range(len(labs))], labs, fontsize=8)
    ax[1].axhline(0, c="0.8", lw=0.8)
    ax[1].set_ylabel("pVar (scaled residual units)")
    ax[1].legend(fontsize=6, ncol=2, loc="center right")
    ax[1].set_title(f"1b  fit quality, reset every {RESET_S:g} s (behaviour-neutral: residuals)", fontsize=9)
    fig.suptitle(f"CURBD on encoding-model residuals -- {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig1_units_fit_quality.png", dpi=150)
    plt.close(fig)
    # FIG 2: stability (pre-stated gate) + EXPLORATORY comparators
    Eg = pd.read_csv(OUT / "tables" / "explore_g_stability.csv")
    Ev = pd.read_csv(OUT / "tables" / "explore_var1_stability.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    kinds = ["seeds", "halves", "sessions"]
    for j, col_ in enumerate(("r", "r_pair")):
        for k, v in enumerate(("multi", "one")):
            for i, kd in enumerate(kinds):
                g = S[(S.variant == v) & (S.kind == kd) & (S.q == "mean_abs")][col_]
                x = i * 3 + k
                ax[j].scatter(np.full(len(g), x) + np.linspace(-0.15, 0.15, len(g)), g, s=14,
                              c=("tab:blue" if v == "multi" else "tab:orange"), label=v if i == 0 else None)
        ax[j].set_xticks([i * 3 + 0.5 for i in range(3)], kinds)
        ax[j].set_title(("2a  CURBD: r of off-diagonal mean |current|" if col_ == "r" else
                         "2b  CURBD: source & target main effects removed\n(pair-specific structure)"), fontsize=9)
    # 2c: exploratory -- g sweep (multi, halves / seeds) and ridge VAR(1) comparator (halves / sessions)
    xt, xl = [], []
    x = 0
    for gv in sorted(Eg.g.unique()):
        for kd, c_ in (("seeds", "tab:blue"), ("halves", "tab:green")):
            g = Eg[(Eg.g == gv) & (Eg.kind == kd)].r
            ax[2].scatter(np.full(len(g), x) + np.linspace(-0.15, 0.15, len(g)), g, s=12, c=c_,
                          label=f"CURBD {kd}" if gv == Eg.g.min() else None)
            x += 1
        xt.append(x - 1.5)
        xl.append(f"g={gv:g}")
        x += 0.6
    for v, mk in (("multi", "o"), ("one", "s")):
        for kd, c_ in (("halves", "tab:green"), ("sessions", "tab:purple")):
            g = Ev[(Ev.variant == v) & (Ev.kind == kd)].r
            ax[2].scatter(np.full(len(g), x) + np.linspace(-0.15, 0.15, len(g)), g, s=12, c=c_, marker=mk,
                          label=("VAR(1) sessions" if kd == "sessions" and v == "multi" else None))
            x += 1
        xt.append(x - 1.5)
        xl.append(f"VAR(1)\n{v}")
        x += 0.6
    ax[2].set_xticks(xt, xl, fontsize=7)
    ax[2].set_title("2c  EXPLORATORY (after the gate failed): initial-J gain g;\nridge VAR(1) comparator (heavy "
                    "ridge, alpha = 0.1 n)", fontsize=9)
    for a_ in ax:
        a_.axhline(0.7, ls="--", c="0.5", lw=0.8)
        a_.axhline(0, c="0.8", lw=0.8)
    ax[0].set_ylabel("Pearson r between current matrices")
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"Stability of region -> region currents (dashed: pre-stated r = 0.7) -- {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig2_stability.png", dpi=150)
    plt.close(fig)
    # FIG 3: what the instability looks like -- one session, two seeds and two halves (multi, mean |current|)
    lab = PRE[0]
    mats = [(f"{lab} seed 0", load_fit(lab, "multi", "full", 0)["mean_abs"]),
            (f"{lab} seed 1", load_fit(lab, "multi", "full", 1)["mean_abs"]),
            (f"{lab} first half", load_fit(lab, "multi", "h1", 0)["mean_abs"]),
            (f"{lab} second half", load_fit(lab, "multi", "h2", 0)["mean_abs"])]
    chronic = (OUT / "fits" / f"{CHRONIC}_multi_full_s2.npz").exists()
    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2))
    for i_, (a_, (ttl, M)) in enumerate(zip(ax, mats)):
        M = M.copy()
        np.fill_diagonal(M, np.nan)
        im = a_.imshow(M, cmap="viridis")
        a_.set_title(f"3{'abcd'[i_]}  {ttl}", fontsize=9)
        plt.colorbar(im, ax=a_, fraction=0.046)
        a_.set_xticks(range(len(REGIONS)), [short(r) for r in REGIONS], rotation=90, fontsize=6)
        a_.set_yticks(range(len(REGIONS)), [short(r) for r in REGIONS], fontsize=6)
        a_.set_xlabel("source")
        a_.set_ylabel("target")
    fig.suptitle(f"CURBD mean |current| (target <- source, diagonal blanked), multi-unit: the same session refitted "
                 f"with another seed / on the other half {'(chronic fitted)' if chronic else '(chronic NOT fitted: gate failed)'}"
                 f" -- {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig3_current_matrices.png", dpi=150)
    plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "fit", "summarise", "plot", "explore", "explore-summary"])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["multi", "one"])
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only or a.cmd == "plot":
        return plot()
    if a.cmd == "prep":
        for lab in a.labels:
            prep(lab)
    elif a.cmd == "fit":
        from wfield_local.parallel import fan_out
        jobs = jobs_for(a.labels or PRE, a.variants)
        jobs = [j for j in jobs if not (OUT / "fits" / f"{j[0]}_{j[1]}_{j[2]}_s{j[3]}.npz").exists()]
        jobs.sort(key=lambda j: (j[1] != "multi", j[2] != "full"))         # longest first-ish
        res, fail = fan_out(jobs, fit_job, jobs=min(4, a.jobs), label="fit")
        print(f"{len(res)} fits, {len(fail)} failures", fail)
    elif a.cmd == "explore":
        from wfield_local.parallel import fan_out
        jobs = [j + (g,) for g in G_SWEEP for j in jobs_for(a.labels or PRE, ("multi",))]
        jobs = [j for j in jobs if not (fits_dir(j[4]) / f"{j[0]}_{j[1]}_{j[2]}_s{j[3]}.npz").exists()]
        res, fail = fan_out(jobs, fit_job, jobs=min(4, a.jobs), label="explore")
        print(f"{len(res)} fits, {len(fail)} failures", fail)
    elif a.cmd == "explore-summary":
        explore_stability()
        explore_var()
    elif a.cmd == "summarise":
        labs = a.labels or list(PRE)
        summarise(labs)
        sens_motor(labs)


if __name__ == "__main__":
    main()
