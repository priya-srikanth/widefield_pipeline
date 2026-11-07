"""State-space view of stroke recovery (Priya 11-07; plan docs/STATE_SPACE_RECOVERY_PLAN.md, analyses 1-4).

BCI perturbation work asks whether population activity after a change is the old activity scaled, shifted, rotated,
reassociated or new. Same question here, per session, against the animal's own August pre-stroke template.

Unit: the trial-averaged condition x time x area matrix X (6 spouts x time bins, 58 Allen areas, both hemispheres) from
the cohort-1 rebuild encoder PETHs (`peth[0]` = data; gate "attempted"). The template T of a pre session is the mean of
the OTHER pre sessions (leave one out), so pre sessions give the noise floor of every number below; the template of a
post session is the mean of all pre sessions.

Per session:
  * vaf_k        share of X's variance (centered per area) inside the pre top-k subspace (manifold test).
  * ladder       held-out (leave one spout out) R2 of nested models of X from T:
                   L0 identity, L1 + per-area offset (uniform shift), L2 + one global gain, L3 per-area gains,
                   L4 L3 then an orthogonal rotation in the pre top-k space (Procrustes), L5 L3 then a free
                   k x k linear map. Rotation beyond gain = L4 - L3 above the pre floor.
  * gain_<area>  per-area gain of L3 fitted on all spouts (left = ipsilesional; all four lesions are left).
  * shared / specific norms: the mean over spouts (condition-independent trajectory) and the deviations from it
                 (position code), each relative to the template's.
  * reassoc_<i>_<j>  correlation of the session's spout-i position code with the template's spout-j code (6 x 6).

Light: ~0.5 MB per session, numpy only. Writes CSVs + figures under Widefield/state_space_recovery/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RUN = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
       "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-"
       "stopA2jexrt3500c70_daqlk/frozen_encoder_gate-attempted_v7g3_t3d_spont_frA/joint/sessions")
OUT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
#: canonical spout order (orofacial_cohort.POS_ORDER), in the encoder files' lesion-relative names
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
POS_LABEL = ["close ipsi", "close mid", "close contra", "far ipsi", "far mid", "far contra"]
WINDOWS = {"cue": ("peth", -0.5, 2.0), "lick": ("peth_lick", -0.5, 1.0)}
BIN_S = 0.1
KS = (4, 8)


def load_session(path: str, align: str):
    key, t0, t1 = WINDOWS[align]
    z = np.load(path, allow_pickle=True)
    meta = json.load(open(path[:-4] + ".json"))
    areas = [str(a) for a in z["areas"]]
    keep = [i for i, a in enumerate(areas) if a != "cortex"]
    pos_order = [str(p) for p in meta.get("pos_order") or POS]
    x = z[key][0][:, keep, :]                                  # (6, areas, t), part 0 = data
    x = x[[pos_order.index(p) for p in POS]]
    t = np.asarray(z["peth_off_s"])
    edges = np.arange(t0, t1 + 1e-9, BIN_S)
    xb = np.stack([np.nanmean(x[:, :, (t >= a) & (t < b)], axis=2) for a, b in zip(edges[:-1], edges[1:])], axis=2)
    xb = np.transpose(xb, (0, 2, 1)).astype(float)             # (6 spouts, bins, areas)
    return xb, [areas[i] for i in keep], meta


def _flat(x):
    return x.reshape(-1, x.shape[-1])


def _topk(Tc, k):
    _, _, vt = np.linalg.svd(Tc, full_matrices=False)
    return vt[:k].T                                            # (areas, k)


def _fit_gain_offset(X, T, per_area=True):
    """X ~ g*T + b per area (least squares over rows)."""
    if per_area:
        Tm, Xm = T.mean(0), X.mean(0)
        num = ((T - Tm) * (X - Xm)).sum(0)
        den = ((T - Tm) ** 2).sum(0)
        g = np.where(den > 0, num / den, 1.0)
        return g, Xm - g * Tm
    Tc, Xc = T - T.mean(0), X - X.mean(0)
    g = float((Tc * Xc).sum() / (Tc ** 2).sum())
    return np.full(T.shape[1], g), X.mean(0) - g * T.mean(0)


def _predict(level, Xtr, Ttr, Tte, P):
    if level == 0:
        return Tte
    if level == 1:
        return Tte + (Xtr - Ttr).mean(0)
    if level == 2:
        g, b = _fit_gain_offset(Xtr, Ttr, per_area=False)
        return Tte * g + b
    g, b = _fit_gain_offset(Xtr, Ttr, per_area=True)
    Ytr, Yte = Ttr * g + b, Tte * g + b
    if level == 3:
        return Yte
    mu = Ytr.mean(0)
    Zy, Zx = (Ytr - mu) @ P, (Xtr - mu) @ P
    if level == 4:                                             # orthogonal Procrustes: Zx ~ Zy R
        u, _, vt = np.linalg.svd(Zy.T @ Zx)
        M = u @ vt
    else:                                                      # free k x k linear map
        M = np.linalg.lstsq(Zy, Zx, rcond=None)[0]
    Zte = (Yte - mu) @ P
    return Yte + (Zte @ M - Zte) @ P.T


def ladder(X, T, P):
    """Held-out R2 per level, leaving one spout out at a time (spouts with NaN skipped)."""
    ok = [s for s in range(6) if np.isfinite(X[s]).all() and np.isfinite(T[s]).all()]
    sse = np.zeros(6)
    sst = 0.0
    for h in ok:
        tr = [s for s in ok if s != h]
        Xtr, Ttr = _flat(X[tr]), _flat(T[tr])
        Xte, Tte = X[h], T[h]
        sst += ((Xte - Xtr.mean(0)) ** 2).sum()
        for lv in range(6):
            sse[lv] += ((Xte - _predict(lv, Xtr, Ttr, Tte, P)) ** 2).sum()
    return 1 - sse / sst, ok


def score(X, T, areas, k_main=KS[-1]):
    ok = [s for s in range(6) if np.isfinite(X[s]).all() and np.isfinite(T[s]).all()]
    Xf, Tf = _flat(X[ok]), _flat(T[ok])
    Xc, Tc = Xf - Xf.mean(0), Tf - Tf.mean(0)
    row = {"n_spouts": len(ok)}
    for k in KS:
        P = _topk(Tc, k)
        row[f"vaf_k{k}"] = float(((Xc @ P) ** 2).sum() / (Xc ** 2).sum())
        # the session's own top-k vs the template's: mean squared cosine of principal angles
        Q = _topk(Xc, k)
        row[f"overlap_k{k}"] = float((np.linalg.svd(P.T @ Q, compute_uv=False) ** 2).mean())
    r2, _ = ladder(X, T, _topk(Tc, k_main))
    for lv, v in enumerate(r2):
        row[f"L{lv}"] = float(v)
    g, b = _fit_gain_offset(Xf, Tf, per_area=True)
    row.update({f"gain_{a}": float(v) for a, v in zip(areas, g)})
    row["gain_global"] = float(_fit_gain_offset(Xf, Tf, per_area=False)[0][0])
    # the regression slope above = r x (sd_X / sd_T): split it, so a decorrelated but LARGER response is not read
    # as a gain loss
    row["amp_ratio"] = float(np.linalg.norm(Xc) / np.linalg.norm(Tc))
    row["r_template"] = float((Xc * Tc).sum() / (np.linalg.norm(Xc) * np.linalg.norm(Tc)))
    row["level_shift"] = float(np.mean(Xf.mean(0) - Tf.mean(0)))
    # condition-independent (shared) vs position-specific parts
    Xs, Ts = np.nanmean(X[ok], 0), np.nanmean(T[ok], 0)
    Xp, Tp = X[ok] - Xs, T[ok] - Ts
    row["shared_norm_ratio"] = float(np.linalg.norm(Xs - Xs.mean(0)) / np.linalg.norm(Ts - Ts.mean(0)))
    row["specific_norm_ratio"] = float(np.linalg.norm(Xp) / np.linalg.norm(Tp))
    row["shared_shift"] = float(np.linalg.norm(Xs - Ts) / np.linalg.norm(Ts - Ts.mean(0)))
    # reassociation: session spout i code vs template spout j code
    for i in range(6):
        for j in range(6):
            a, c = (X[i] - Xs).ravel(), (T[j] - Ts).ravel()
            row[f"reassoc_{i}_{j}"] = (float(np.corrcoef(a, c)[0, 1])
                                       if np.isfinite(a).all() and np.isfinite(c).all() else np.nan)
    return row


def run(align: str):
    files = sorted(f for f in glob.glob(RUN + "/PS9*.npz") if not f.endswith("_trials.npz"))
    data = {}
    for f in files:
        x, areas, meta = load_session(f, align)
        data[meta["label"]] = (x, meta)
    rows = []
    for an in sorted({m["animal"] for _, m in data.values()}):
        labs = [lab for lab, (_, m) in data.items() if m["animal"] == an]
        pre = [lab for lab in labs if data[lab][1]["epoch"] == "pre"]
        for lab in labs:
            x, m = data[lab]
            ref = [p for p in pre if p != lab]
            T = np.nanmean(np.stack([data[p][0] for p in ref]), axis=0)
            row = {"label": lab, "animal": an, "epoch": m["epoch"], "day": m.get("day"),
                   "n_trials": m.get("n_trials"), "n_ref": len(ref), "align": align}
            row.update(score(x, T, areas))
            rows.append(row)
    return pd.DataFrame(rows), areas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--align", nargs="+", default=list(WINDOWS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for al in a.align:
        df, areas = run(al)
        df.to_csv(out / f"state_space_sessions_{al}.csv", index=False)
        cols = ["vaf_k4", "vaf_k8", "overlap_k8", "L0", "L1", "L2", "L3", "L4", "L5", "gain_global",
                "shared_norm_ratio", "specific_norm_ratio", "shared_shift"]
        order = ["pre", "acute", "subacute", "chronic"]
        tab = (df.groupby(["animal", "epoch"])[cols].mean().reindex(order, level="epoch"))
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
            print(f"\n=== {al}-aligned, per animal x epoch (pre = leave-one-out floor) ===")
            print(tab)
        print(f"wrote {out / f'state_space_sessions_{al}.csv'} ({len(df)} sessions)")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
