"""STRAY-LICK TEST, lick level (Priya 10-23: "don't we want to be comparing LICK stray response vs on-target response";
"go ahead with relabel and stray lick tests"). FINDINGS 2.6.

On an on-target lick the TARGET and the spout-locked executed movement are the same variable; only licks that land
away from the aimed spout separate them. All categories are NON-CONTACT licks, so the touch response cannot confound
the comparison (on-target contact licks carry touch, strays do not):

  stray q->j    aimed at spout q, tongue peak nearest spout j's zone            the test licks
  near_q        aimed at q, nearest q's zone (no contact)                       same TARGET, different execution
  near_j        aimed at j, nearest j's zone (no contact)                       different target, same EXECUTION
  iti_j         between trials (no spout target), nearest j's zone              pure execution, no target

Zones: each spout's CALIBRATED centre = median 3-D tongue-peak position (side_ml, out_ap, down_dv, mm, mouth frame) of
the animal's August pre CONTACT licks there (nominal spout angles are biased ~21 deg); a lick's zone = nearest centre.
Per-lick 3-D: `<dlc>/session_poses/pose3d/kinematics_all/licks.csv.gz` (pre-cue, post-cue and ITI licks; key lick_uid).

Per session: Allen-area traces on ALL imaging frames (drift slower than DRIFT_MIN_PERIOD_S removed) regressed (ridge) on
one lick kernel per category (impulse at the 3-D tongue peak, lags LAGS_S) + nuisance (cue and spout-arrival kernels,
forward running hinge from 1.5 mm/s, backward push); overlapping licks are separated by the fit. Categories with fewer
than MIN_N licks in a session are left out of that session (their licks join a pooled "rest" kernel).

Readout (per animal x epoch, kernels combined over sessions weighted by lick count): spout-specific part S = kernel -
generic (mean of the near_* kernels over spouts, same epoch frame); r_target = corr(S[stray q->j], S_ref[near_q]),
r_exec = corr(S[stray q->j], S_ref[near_j]), r_iti = corr(S[stray q->j], S_ref[iti_j] - generic iti), over lags x
areas. References from the PRE sessions (frozen frame; primary) and from the same epoch (secondary). Validation:
split-half near_q kernels must correlate with their own spout above others (`validate`).

Usage: python -m wfield_local.stray_licks [--jobs 3] [--labels ...] [--summarise-only]
Output: <E>/stray_licks/{sessions/<label>.npz, stray_kernels_by_animal.npz, stray_index.csv, stray_validation.csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KIN_ALL = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics_all/licks.csv.gz")
LAGS_S = (-0.1, 0.5)
CUE_LAGS_S = (-0.5, 2.0)
MIN_N = 15
COORDS = ("side_ml", "out_ap", "down_dv")
PRE_FROM = "20260806"
RUN_ON, PUSH_FLOOR = 1.5, 0.3


def pos_name(p: str) -> str:
    from wfield_local import orofacial_cohort as oc
    return oc.pos_key(p, "L")                           # lesion LEFT: contra = the mouse's right


def centres(K) -> dict:
    """{animal: {spout: (3,) calibrated centre}} from August pre post-cue CONTACT licks."""
    m = (K.epoch == "pre") & (K.date.astype(str) >= PRE_FROM) & (K.context == "post_cue") & K.contact.astype(bool)
    C = K[m].dropna(subset=list(COORDS)).groupby(["animal", "spout"])[list(COORDS)].median()
    return {an: {sp: C.loc[(an, sp)].to_numpy(float) for sp in C.loc[an].index} for an in C.index.get_level_values(0).unique()}


def classify(L, cen: dict) -> np.ndarray:
    """Category per lick (np.ndarray of str): stray|q|j, near|q, iti|j, contact (excluded), '' (no 3-D)."""
    sp = list(cen)
    P = L[list(COORDS)].to_numpy(float)
    D = np.stack([np.linalg.norm(P - cen[s], axis=1) for s in sp], 1)
    ok = np.isfinite(D).all(1)
    nearest = np.array(sp, dtype=object)[np.where(ok, np.nanargmin(np.where(ok[:, None], D, 0), 1), 0)]
    cat = np.full(len(L), "", dtype=object)
    ctx, q, con = L.context.to_numpy(), L.spout.to_numpy(), L.contact.astype(bool).to_numpy()
    for i in np.flatnonzero(ok):
        if ctx[i] == "iti":
            cat[i] = f"iti|{nearest[i]}"
        elif con[i]:
            cat[i] = "contact"
        elif nearest[i] == q[i]:
            cat[i] = f"near|{q[i]}"
        else:
            cat[i] = f"stray|{q[i]}|{nearest[i]}"
    return cat


def _lagged(ev_frames, n: int, lags: np.ndarray):
    """Sparse (n, len(lags)) lagged impulse columns (overlapping events add)."""
    from scipy import sparse
    f = np.asarray(ev_frames, int)
    r = (f[:, None] + lags[None, :]).ravel()
    c = np.tile(np.arange(len(lags)), len(f))
    ok = (r >= 0) & (r < n)
    return sparse.csr_matrix((np.ones(ok.sum()), (r[ok], c[ok])), shape=(n, len(lags)))


def fit_kernels(Xs, D: np.ndarray, Y: np.ndarray, ridge_frac: float = 0.01) -> np.ndarray:
    """Ridge of Y on [Xs (sparse event columns) | D (dense nuisance: drift basis, running)]: event columns penalised by
    ridge_frac x their own Gram diagonal, nuisance ~unpenalised. Returns the event-column betas (Xs.shape[1], R)."""
    XtX = (Xs.T @ Xs).toarray()
    XtD = np.asarray(Xs.T @ D)
    G = np.block([[XtX, XtD], [XtD.T, D.T @ D]])
    b = np.vstack([np.asarray(Xs.T @ Y), D.T @ Y])
    lam = np.concatenate([ridge_frac * np.diag(XtX) + 1e-6, np.full(D.shape[1], 1e-8)])
    beta = np.linalg.solve(G + np.diag(lam), b)
    return beta[:Xs.shape[1]]


def session_job(item) -> dict:
    """One session: per-category lick kernels (lags x areas) on all frames."""
    import pandas as pd

    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    label, cen_json = item
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    cen = {k: np.asarray(v) for k, v in json.loads(cen_json).items()}
    V, ft, Ub, _flat, parea, names, _shape, pstat = ce.load_imaging(label)
    areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
    Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)
    area_names = [names.get(a, str(a)) for a in areas] + ["cortex"]
    Y = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    Y -= Q @ (Q.T @ Y)
    fs = 1.0 / np.median(np.diff(ft))
    lags = np.arange(int(round(LAGS_S[0] * fs)), int(round(LAGS_S[1] * fs)) + 1)
    clags = np.arange(int(round(CUE_LAGS_S[0] * fs)), int(round(CUE_LAGS_S[1] * fs)) + 1)
    B = ce.load_behaviour(an, date, PathResolver())
    T = B.trials
    cue = T.cue_s.to_numpy(float)
    arr = T.strobe_s.to_numpy(float)
    K = pd.read_csv(KIN_ALL, usecols=["session", "position", "context", "contact", "daq_peak_s", *COORDS])
    K = K[K.session == f"{an}_{date}"].copy()
    K["spout"] = [pos_name(p) for p in K.position]
    cat = classify(K, cen)
    fr = np.searchsorted(ft, K.daq_peak_s.to_numpy(float))
    inside = (fr > 0) & (fr < len(ft) - 1)
    cats, counts = np.unique(cat[inside & (cat != "")], return_counts=True)
    keep = [c for c, n in zip(cats, counts) if n >= MIN_N and c != "contact"]
    from scipy import sparse
    blocks, labels = [], []
    for c in keep:
        blocks.append(_lagged(fr[inside & (cat == c)], len(ft), lags))
        labels.append(c)
    rest = inside & (cat != "") & ~np.isin(cat, keep)                    # contact + rare categories: shared kernel
    blocks += [_lagged(fr[rest], len(ft), lags),
               _lagged(np.searchsorted(ft, cue[np.isfinite(cue)]), len(ft), clags),
               _lagged(np.searchsorted(ft, arr[np.isfinite(arr)]), len(ft), clags)]
    rt, rv = (np.asarray(x, float) for x in B.analog["running"])
    sp = np.interp(ft, rt, rv)
    D = np.column_stack([Q, np.maximum(sp - RUN_ON, 0), np.maximum(-sp - PUSH_FLOOR, 0)])
    beta = fit_kernels(sparse.hstack(blocks).tocsr(), D, Y)
    kern = {c: beta[i * len(lags):(i + 1) * len(lags)] for i, c in enumerate(labels)}
    return {"label": label, "animal": an, "epoch": epochs.epoch_of(label), "areas": area_names,
            "lags_s": lags / fs, "kernels": kern,
            "counts": {c: int(n) for c, n in zip(cats, counts)}}


def combine(rows: list[dict], epoch_set: tuple) -> dict:
    """{category: count-weighted mean kernel} over the given epochs' sessions of one animal."""
    acc, w = {}, {}
    for r in rows:
        if r["epoch"] not in epoch_set:
            continue
        for c, k in r["kernels"].items():
            n = r["counts"].get(c, 0)
            acc[c] = acc.get(c, 0.0) + n * k
            w[c] = w.get(c, 0) + n
    return {c: acc[c] / w[c] for c in acc}, w


def _spec(kern: dict, prefix: str) -> np.ndarray | None:
    ks = [k for c, k in kern.items() if c.startswith(prefix)]
    return np.mean(ks, 0) if ks else None


def _corr(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def summarise(out: Path) -> None:
    import pandas as pd
    rows = []
    for f in sorted((out / "sessions").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        rows.append(z["r"].item())
    order = ["pre", "acute", "subacute", "chronic"]
    idx, val = [], []
    for an in sorted({r["animal"] for r in rows}):
        R = [r for r in rows if r["animal"] == an]
        ref, wref = combine(R, ("pre",))
        gen = _spec(ref, "near|")
        geni = _spec(ref, "iti|")
        if gen is None:
            continue
        for ep in order:
            cur, wcur = combine(R, (ep,))
            gen_e = _spec(cur, "near|")
            for c, k in cur.items():
                if not c.startswith("stray|"):
                    continue
                _s, q, j = c.split("|")
                for frame, rk, g, gi in (("pre_ref", ref, gen, geni), ("same_epoch_ref", cur, gen_e, _spec(cur, "iti|"))):
                    if g is None:
                        continue
                    S = k - g
                    row = {"animal": an, "epoch": ep, "stray": f"{q}->{j}", "aimed": q, "landed": j, "n": wcur[c],
                           "ref": frame,
                           "r_target": _corr(S, rk[f"near|{q}"] - g) if f"near|{q}" in rk else np.nan,
                           "r_exec": _corr(S, rk[f"near|{j}"] - g) if f"near|{j}" in rk else np.nan,
                           "r_iti": (_corr(S, rk[f"iti|{j}"] - gi) if (gi is not None and f"iti|{j}" in rk) else np.nan)}
                    row["target_minus_exec"] = row["r_target"] - row["r_exec"]
                    idx.append(row)
        # validation: split-half near_q (sessions alternately) -> own spout vs others
        pre = [r for r in R if r["epoch"] == "pre"]
        h1, _ = combine(pre[0::2], ("pre",))
        h2, _ = combine(pre[1::2], ("pre",))
        g1, g2 = _spec(h1, "near|"), _spec(h2, "near|")
        if g1 is not None and g2 is not None:
            for c in [c for c in h1 if c.startswith("near|") and c in h2]:
                own = _corr(h1[c] - g1, h2[c] - g2)
                oth = [_corr(h1[c] - g1, h2[o] - g2) for o in h2 if o.startswith("near|") and o != c]
                val.append({"animal": an, "spout": c.split("|")[1], "r_own": own, "r_other_mean": np.nanmean(oth)})
    I = pd.DataFrame(idx)
    Vd = pd.DataFrame(val)
    I.to_csv(out / "stray_index.csv", index=False, float_format="%.4g")
    Vd.to_csv(out / "stray_validation.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print("=== validation (pre, split-half near-miss kernels, spout-specific part): own spout vs other spouts\n"
          + Vd.round(3).to_string(index=False))
    if I.empty:
        return
    P = I[I.ref == "pre_ref"]
    print("\n=== stray q->j vs pre-stroke references (spout-specific parts): r_target (near q) vs r_exec (near j), "
          "count-weighted over sessions; target_minus_exec > 0 = TARGET-like")
    S = P.groupby(["animal", "epoch"]).apply(lambda d: pd.Series({
        "n_strays": d.n.sum(), "n_types": len(d),
        "target_minus_exec (weighted)": np.average(d.target_minus_exec.fillna(0), weights=d.n),
        "share target-like": np.mean(d.target_minus_exec > 0)}), include_groups=False)
    print(S.reindex(sorted(S.index, key=lambda k: (k[0], order.index(k[1])))).round(3).to_string())
    F = P[P.aimed == "far_contra"].sort_values(["animal", "epoch", "landed"])
    print("\n=== far_contra strays\n" + F[["animal", "epoch", "landed", "n", "r_target", "r_exec", "r_iti",
                                         "target_minus_exec"]].round(3).to_string(index=False))
    print(f"\n-> {out}")


def main(argv=None) -> int:
    import pandas as pd

    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "stray_licks"
    assert_writable(out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    if a.summarise_only:
        summarise(out)
        return 0
    K = pd.read_csv(KIN_ALL, usecols=["animal", "date", "epoch", "position", "context", "contact", *COORDS])
    K["spout"] = [pos_name(p) for p in K.position]
    cen = centres(K)
    (out / "centres.json").write_text(json.dumps({an: {s: c.tolist() for s, c in d.items()} for an, d in cen.items()},
                                                 indent=1), encoding="utf-8")
    cur = set(ak.curated_labels())
    labels = a.labels or [x for x in sorted(ce.sessions_available()) if x in cur and x[-4:] >= PRE_FROM[4:]]
    done = {f.stem for f in (out / "sessions").glob("*.npz")}
    items = [(x, json.dumps({s: c.tolist() for s, c in cen[config.animal_of(x)].items()})) for x in labels
             if x not in done and config.animal_of(x) in cen]
    got, fail = ak.fan_sessions(items, session_job, jobs=a.jobs, label="stray")
    for _it, r in got:
        np.savez_compressed(out / "sessions" / f"{r['label']}.npz", r=np.array(r, dtype=object))
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    summarise(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
