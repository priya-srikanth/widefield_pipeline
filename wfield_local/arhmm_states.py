"""AR-HMM STATE QC and MERGING (port of Hillary Tong's F11 / F12 workflow, github.com/htong1/Priya_ARHMM2026:
scripts/state_merge_methods.py, merge_by_features.py, apply_merge.py; Priya 11-16: "previously we used cosine
similarity and visual qc to combine states that were very similar").

    python -m wfield_local.arhmm_states similarity --out <arhmm out dir> --feat <features dir> --tag pooled_all_s1
    python -m wfield_local.arhmm_states merge      --out <dir> --feat <dir> --tag pooled_all_s1 --merge 4-18,0-12-14

Her workflow, kept: fit K = 24 (an upper bound), then (1) DISCARD dead states (usage < DEAD_USAGE); (2) score every
pair of live states by COSINE SIMILARITY of per-state vectors in INDEPENDENT representations -- a pair is a merge
CANDIDATE only when the representations agree (per-method RANKS, because AR-dynamics cosines sit systematically lower:
her 6-method vote missed only on the dynamics method at a flat 0.8); NO transition statistics in the similarity (her
mentor's constraint); (3) a HUMAN VIDEO GATE (state reels, `behavior_clips` cutter) before any merge is applied;
(4) apply the merge and re-evaluate dwell times + transitions (merging over-split ping-pong states should lengthen
dwell). Her lesson (F12): two left-lick groups the cosine kept apart were different movements (contact 97 % vs 76 %)
-- the cosine is a shortlist, the video decides.

REPRESENTATIONS (per live state):
  A  mean of the z-scored, block-weighted features (the model's own pre-PCA space)
  B  mean PCA coordinates
  C  AR dynamics: the state's flattened Ab regression matrix (independent of where the features sit)
  D  FaceRhythm SPATIAL map: master_pre_v1 spatial loadings weighted by the state's mean FR activation
  E  FaceRhythm SPECTRUM: master_pre_v1 spectral loadings weighted the same way
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEAD_USAGE = 0.005
#: the REVIEWED reference fit (Priya's 11-17 reel review labels its states): pooled_all seed 1, kappa 5e7, dither 0.1
REFERENCE_TAG = "pooled_all_s1"
RANK_Q = 0.9                      # a pair "agrees" in a method when its cosine is in that method's top 10 % of pairs
MIN_AGREE = 4                     # of 5 methods


def fits_dir() -> Path:
    """Where the fitted models (``model_<tag>.pkl``) and decoded states (``states_<tag>/<label>.npz``) live:
    ``LP_analysis/<run>/arhmm/v1/fits`` (``PROVENANCE.txt`` there says which fit came from where); ARHMM_FITS overrides."""
    import os
    if os.environ.get("ARHMM_FITS"):
        return Path(os.environ["ARHMM_FITS"])
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return Path(oo.root(PathResolver())) / oc.RUN / "arhmm" / "v1" / "fits"


def load(out: Path, tag: str, feat: Path):
    from wfield_local import arhmm_model as am
    M = am.load_model(out / f"model_{tag}.pkl")
    feats = am.load_features(feat, M.get("run_floor", am.RUN_FLOOR))
    Z = {}
    for f in sorted(glob.glob(str(out / f"states_{tag}" / "*.npz"))):
        lab = Path(f).stem
        if lab in feats:
            Z[lab] = np.load(f)["z"].astype(int)
    return M, feats, Z


def usage(Z: dict, K: int) -> np.ndarray:
    z = np.concatenate([v[v >= 0] for v in Z.values()])
    return np.bincount(z, minlength=K) / len(z)


def representations(M, feats, Z, live) -> dict:
    """{method: [n_live, dim]} per-state vectors (module docstring)."""
    d = M["design"]
    cols = d.columns
    A_sum = np.zeros((len(live), len(cols)))
    B_sum = np.zeros((len(live), len(d.var_ratio)))
    n = np.zeros(len(live))
    raw_sum = np.zeros((len(live), len(cols)))
    idx = {k: i for i, k in enumerate(live)}
    for lab, z in Z.items():
        X = feats[lab]["X"]
        ok = (z >= 0) & np.isfinite(X).all(1) & np.isin(z, live)
        if not ok.any():
            continue
        Zs = np.clip((X[ok] - d.mu) / d.sd, -8, 8) * d.w
        P = (Zs - d.zm) @ d.comps.T
        ii = np.array([idx[k] for k in z[ok]])
        np.add.at(A_sum, ii, Zs)
        np.add.at(B_sum, ii, P)
        np.add.at(raw_sum, ii, X[ok])
        np.add.at(n, ii, 1)
    n = np.maximum(n, 1)[:, None]
    rep = {"A_features": A_sum / n, "B_pca": B_sum / n}
    Ab = np.asarray(M["params"]["Ab"])
    rep["C_dynamics"] = Ab[live].reshape(len(live), -1)
    fr = [j for j, c in enumerate(cols) if c.startswith("fr_")]
    if fr:
        from wfield_local import face_rhythm_outputs as fro
        from wfield_local.paths import PathResolver
        m = np.load(fro.factor_dir(PathResolver()) / "master.npz", allow_pickle=True)
        act = (raw_sum / n)[:, fr]                                 # mean absolute FR expression per state
        rep["D_fr_spatial"] = act @ m["spatial"].reshape(-1, m["spatial"].shape[-1]).T
        rep["E_fr_spectrum"] = act @ m["spectral"].T
    return rep


def cosine(V: np.ndarray) -> np.ndarray:
    V = V - V.mean(0) if V.shape[0] > 1 else V                     # centre across states: similarity of DEVIATIONS
    U = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return U @ U.T


def candidates(rep: dict, live) -> pd.DataFrame:
    """Every live pair: cosine per method, whether it is in that method's top RANK_Q, the agreement count."""
    S = {k: cosine(v) for k, v in rep.items()}
    iu = np.triu_indices(len(live), 1)
    rows = pd.DataFrame({"a": np.asarray(live)[iu[0]], "b": np.asarray(live)[iu[1]]})
    for k, s in S.items():
        v = s[iu]
        rows[k] = v
        rows[f"{k}_top"] = v >= np.quantile(v, RANK_Q)
    tops = [c for c in rows if c.endswith("_top")]
    rows["agree"] = rows[tops].sum(1)
    return rows.sort_values(["agree", "A_features"], ascending=False).reset_index(drop=True)


def apply_merge(Z: dict, groups: list[list[int]]) -> dict:
    """State sequences with each group relabelled to its first member."""
    m = {s: g[0] for g in groups for s in g}
    return {lab: np.where(z >= 0, np.vectorize(lambda s: m.get(int(s), int(s)))(z), z) for lab, z in Z.items()}


def dwell_s(Z: dict, fs: float = 25.0) -> pd.DataFrame:
    rows = []
    for _lab, z in Z.items():
        ch = np.flatnonzero(np.diff(z) != 0)
        st = np.r_[0, ch + 1]
        ln = np.diff(np.r_[st, len(z)])
        for s0, n_ in zip(st, ln):
            if z[s0] >= 0:
                rows.append((int(z[s0]), n_ / fs))
    D = pd.DataFrame(rows, columns=["state", "dur_s"])
    return D.groupby("state").dur_s.agg(["median", "mean", "size"])


def lick_onsets(label: str) -> np.ndarray:
    """Camera lick onsets (DAQ s; LP licks_all, every context) -- the lick identity the rest of the pipeline uses."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    a, d = label.split("_")
    L = pd.read_csv(Path(oo.session_dir(PathResolver(), oc.RUN, a, d)) / "licks_all.csv", usecols=["daq_on_s"])
    return np.sort(L.daq_on_s.dropna().to_numpy(float))


def profile(M, feats, Z, feat_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Sanity table per state: usage, median / mean dwell (s), mean raw features, LICK ENRICHMENT (share of camera lick
    onsets that fall in the state / the state's share of time; > 1 = licking state), mean lateral tongue (contra +)."""
    cols = M["design"].columns
    K = int(np.asarray(M["params"]["Ab"]).shape[0])
    u = usage(Z, K)
    dw = dwell_s(Z)
    feat_sum = np.zeros((K, len(cols)))
    n = np.zeros(K)
    lick_n = np.zeros(K)
    for lab, z in Z.items():
        X = feats[lab]["X"]
        ok = (z >= 0) & np.isfinite(X).all(1)
        np.add.at(feat_sum, z[ok], X[ok])
        np.add.at(n, z[ok], 1)
        t = np.load(Path(feat_dir) / "sessions" / f"{lab}.npz")["t_daq_s"]
        lk = lick_onsets(lab)
        i = np.searchsorted(t - 0.02, lk) - 1                     # bin of each lick onset (bins centred, 40 ms)
        i = i[(i >= 0) & (i < len(z))]
        zi = z[i]
        np.add.at(lick_n, zi[zi >= 0], 1)
    P = pd.DataFrame(feat_sum / np.maximum(n, 1)[:, None], columns=cols)
    P.insert(0, "usage", u)
    P.insert(1, "lick_enrich", (lick_n / max(lick_n.sum(), 1)) / np.maximum(u, 1e-12))
    P.insert(2, "dwell_med_s", dw["median"].reindex(range(K)).to_numpy())
    P.insert(3, "dwell_mean_s", dw["mean"].reindex(range(K)).to_numpy())
    D = np.concatenate([np.repeat(dw.loc[k, "median"], 1) for k in dw.index]) if len(dw) else np.zeros(0)
    allz = {lab: z for lab, z in Z.items()}
    durs = []
    for z in allz.values():
        ch = np.flatnonzero(np.diff(z) != 0)
        st = np.r_[0, ch + 1]
        ln = np.diff(np.r_[st, len(z)])
        durs += [x / 25.0 for s0, x in zip(st, ln) if z[s0] >= 0]
    durs = np.asarray(durs)
    summ = {"n_states_used_gt_0.5pct": int((u >= DEAD_USAGE).sum()), "median_syllable_s": float(np.median(durs)),
            "mean_syllable_s": float(durs.mean()), "p10_p90_s": [float(np.percentile(durs, 10)),
                                                                  float(np.percentile(durs, 90))],
            "median_of_state_medians": float(np.median(D)) if len(D) else float("nan")}
    return P, summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("cmd", choices=("similarity", "merge", "profile"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--merge", default="", help="groups by comma, members by hyphen, e.g. 4-18,0-12-14")
    a = ap.parse_args(argv)
    out = Path(a.out)
    M, feats, Z = load(out, a.tag, Path(a.feat))
    K = int(np.asarray(M["params"]["Ab"]).shape[0])
    u = usage(Z, K)
    live = [k for k in range(K) if u[k] >= DEAD_USAGE]
    qc = out / f"states_qc_{a.tag}"
    qc.mkdir(exist_ok=True)
    if a.cmd == "profile":
        P, summ = profile(M, feats, Z, Path(a.feat))
        P.to_csv(qc / "state_profile.csv", index_label="state")
        print(json.dumps(summ, indent=1))
        show = [c for c in ("usage", "lick_enrich", "dwell_med_s", "tongue_ext", "tongue_ml", "jaw_y", "nose_y",
                            "run_fwd", "run_back") if c in P] + [c for c in P if c.startswith("fr_lick")][:3]
        print(P.loc[P.usage >= DEAD_USAGE, show].sort_values("usage", ascending=False).round(3).to_string())
        return 0
    if a.cmd == "similarity":
        rep = representations(M, feats, Z, live)
        C = candidates(rep, live)
        C.to_csv(qc / "pair_similarity.csv", index=False)
        pd.DataFrame({"state": range(K), "usage": u, "live": [k in live for k in range(K)]}).to_csv(
            qc / "usage.csv", index=False)
        print(f"{len(live)} live / {K} states (dead < {DEAD_USAGE:.1%}): dead = {[k for k in range(K) if k not in live]}")
        print(C[C.agree >= MIN_AGREE].round(3).to_string())
        return 0
    groups = [[int(x) for x in g.split("-")] for g in a.merge.split(",") if g]
    before, after = dwell_s(Z), dwell_s(apply_merge(Z, groups))
    (qc / "merge.json").write_text(json.dumps({"groups": groups}))
    print("dwell before (median s):", before["median"].round(2).to_dict())
    print("dwell after  (median s):", after["median"].round(2).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
