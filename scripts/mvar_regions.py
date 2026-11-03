"""REGION-LEVEL RIDGE MVAR on encoding-model residuals (Priya 10-30, after the CURBD pilot failed its gate). PROVISIONAL.

Design + stability gate pre-registered in docs/status/MVAR_PREREG_20261030.md (written before any fit) -- read it.

    python scripts/mvar_regions.py prep  [labels ...]        # curbd_pilot.prep per session (18 region residuals, 10.4 Hz)
    python scripts/mvar_regions.py fit   [labels ...]        # per session: nested-CV R2 (full vs AR-only), W, +-decade, halves
    python scripts/mvar_regions.py gate                      # stability gate on the AUGUST pre sessions, per animal
    python scripts/mvar_regions.py fixed                     # (only if the gate passed) fixed per-animal lambda / p, all sessions
    python scripts/mvar_regions.py summarise                 # pre vs post S->M / M->S tables + boot_delta
    python scripts/mvar_regions.py plot | --plot-only        # figures from the saved figdata only

DATA = `curbd_pilot.prep` variant "one" (cached in curbd_pilot/data/): 18 region means, held-out encoder residuals,
engaged rows, 3-frame means. MODES: raw (primary); gsr (each region minus its regression on the mean of the other 17,
fit on the session); dec (s = x + 0.6 s dx/dt within run, GCaMP6s).
MODEL x_t = c + sum_l B_l x_{t-l}; ridge alpha = lambda * n_train; lambda grid 10^-5..10^1; p in 1..3; AR-only = diagonal
B_l. W = sum_l B_l (standardised). Lagged samples never cross a run (trial) boundary; folds = contiguous blocks of runs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import curbd_pilot as cp  # noqa: E402

OUT = cp.OUT.parent / "mvar"
DATA = cp.OUT / "data"
REGIONS = cp.REGIONS
NR = len(REGIONS)
LAMS = 10.0 ** np.linspace(-5, 1, 19)
PS = (1, 2, 3)
PMAX = 3
K_OUT, K_IN = 5, 4
TAU = 0.6
MODES = ("raw", "gsr", "dec")
SENS = ("SSp-n", "SSp-m", "SSs", "SSp-ul")
MOT = ("tjM1", "ALM", "MOp_rest", "MOs_rest")
EPS = ["pre", "acute", "subacute", "chronic"]
AN = ["PS92", "PS93", "PS94", "PS95"]
COL = {"pre": "black", "acute": "magenta", "subacute": "goldenrod", "chronic": "teal"}
GATE = {"halves": 0.7, "halves_min": 0.5, "sessions": 0.7, "ridge": 0.7, "pair": 0.5, "pred_frac": 0.75, "pred_folds": 4}


# ------------------------------------------------------------------------------------------------------------ sessions
def all_labels(pre_only=False) -> list[str]:
    """Curated sessions with an encoder json and an epoch; pre = AUGUST only (June pre dropped)."""
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    out = []
    for lab in ak.curated_labels():
        ep = epochs.epoch_of(lab)
        if ep is None or not (cp.ENC / f"{lab}.json").exists():
            continue
        if ep == "pre" and lab.split("_")[1][:2] != "08":
            continue
        if pre_only and ep != "pre":
            continue
        out.append(lab)
    return sorted(out)


def epoch(lab):
    from wfield_local import epochs
    return epochs.epoch_of(lab)


def prep_job(lab):
    f = DATA / f"{lab}.npz"
    if f.exists():
        return str(f)
    return str(cp.prep(lab))


# ---------------------------------------------------------------------------------------------------------------- data
def load_runs(lab, mode="raw") -> tuple[list[np.ndarray], float]:
    """List of (T_run, 18) arrays (float64) for one session, transformed per mode."""
    d = np.load(DATA / f"{lab}.npz")
    Y = d["res_one"].astype(np.float64).T                 # (T, 18)
    dt = float(d["dt"])
    st = list(d["run_starts"]) + [len(Y)]
    if mode == "gsr":
        Y = Y - Y.mean(0)
        S = Y.sum(1, keepdims=True)
        for a in range(NR):
            g = (S[:, 0] - Y[:, a]) / (NR - 1)
            Y[:, a] = Y[:, a] - (g @ Y[:, a]) / (g @ g) * g
    runs = [Y[a:b] for a, b in zip(st[:-1], st[1:])]
    if mode == "dec":
        runs = [r + TAU * np.gradient(r, dt, axis=0) for r in runs]
    return runs, dt


def lagged(runs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Common sample set for every p (t >= PMAX within its run): Y (n, 18), L (PMAX, n, 18) with L[l-1] = x_{t-l}, run id."""
    Ys, Ls, rid = [], [], []
    for k, r in enumerate(runs):
        T = len(r)
        if T <= PMAX:
            continue
        Ys.append(r[PMAX:])
        Ls.append(np.stack([r[PMAX - l: T - l] for l in range(1, PMAX + 1)]))
        rid.append(np.full(T - PMAX, k))
    return np.vstack(Ys), np.concatenate(Ls, 1), np.concatenate(rid)


def blocks(rid, K) -> np.ndarray:
    """Fold id per sample: K contiguous blocks of whole runs with ~equal sample counts."""
    u, cnt = np.unique(rid, return_counts=True)
    cum = np.cumsum(cnt) / cnt.sum()
    fr = np.minimum((cum * K - 1e-9).astype(int), K - 1)
    m = dict(zip(u, fr))
    return np.array([m[r] for r in rid])


# --------------------------------------------------------------------------------------------------------------- ridge
class Fitter:
    """Ridge fits on a training index set, standardised with the TRAINING statistics; all lambdas at once."""

    def __init__(self, Y, L, tr):
        self.mu, self.sd = Y[tr].mean(0), Y[tr].std(0) + 1e-12
        self.n = int(tr.sum())
        self.Y, self.L, self.tr = Y, L, tr

    def z(self, idx):
        return (self.Y[idx] - self.mu) / self.sd, (self.L[:, idx] - self.mu) / self.sd

    def full(self, p, lams):
        """-> list over lams of (B (p, 18 tgt, 18 src), c (18,))."""
        Yt, Lt = self.z(self.tr)
        X = np.concatenate([Lt[l] for l in range(p)], 1)
        xm, ym = X.mean(0), Yt.mean(0)
        Xc, Yc = X - xm, Yt - ym
        ev, V = np.linalg.eigh(Xc.T @ Xc)
        XtY = V.T @ (Xc.T @ Yc)
        out = []
        for lam in lams:
            beta = V @ (XtY / (ev + lam * self.n)[:, None])           # (18p, 18)
            B = beta.T.reshape(NR, p, NR).transpose(1, 0, 2)          # (p, tgt, src)
            out.append((B, ym - xm @ beta))
        return out

    def ar(self, p, lams):
        Yt, Lt = self.z(self.tr)
        outs = [(np.zeros((p, NR, NR)), np.zeros(NR)) for _ in lams]
        for a in range(NR):
            X = Lt[:p, :, a].T
            xm, ym = X.mean(0), Yt[:, a].mean()
            Xc, yc = X - xm, Yt[:, a] - ym
            ev, V = np.linalg.eigh(Xc.T @ Xc)
            g = V.T @ (Xc.T @ yc)
            for k, lam in enumerate(lams):
                b = V @ (g / (ev + lam * self.n))
                outs[k][0][:, a, a] = b
                outs[k][1][a] = ym - xm @ b
        return outs

    def sse(self, idx, B, c, zt=None):
        """SSE (standardised units) on samples idx, plus SS around the training mean and SS of the change.
        ``zt`` = precomputed self.z(idx) (reused across the lambda grid)."""
        Yt, Lt = zt if zt is not None else self.z(idx)
        pred = c + sum(Lt[l] @ B[l].T for l in range(len(B)))
        return float(((Yt - pred) ** 2).sum()), float((Yt ** 2).sum()), float(((Yt - Lt[0]) ** 2).sum())


def choose(Y, L, rid, idx, kind, K=K_IN) -> tuple[float, int, np.ndarray]:
    """(lambda, p) minimising blocked-CV SSE over samples idx (contiguous run blocks inside idx). Returns the SSE grid."""
    f = blocks(rid[idx], K)
    grid = np.zeros((len(PS), len(LAMS)))
    ii = np.flatnonzero(idx)
    for k in range(K):
        tr = np.zeros(len(Y), bool)
        te = np.zeros(len(Y), bool)
        tr[ii[f != k]] = True
        te[ii[f == k]] = True
        F = Fitter(Y, L, tr)
        Yt, Lt = F.z(te)
        yy = float((Yt ** 2).sum())
        for i, p in enumerate(PS):
            Xa = np.concatenate([Lt[l] for l in range(p)] + [np.ones((len(Yt), 1))], 1)
            Gte, Hte = Xa.T @ Xa, Xa.T @ Yt                     # exact SSE from test sufficient statistics
            for j, (B, c) in enumerate(getattr(F, kind)(p, LAMS)):
                th = np.vstack([B[l].T for l in range(p)] + [c[None]])   # (18p + 1, 18)
                grid[i, j] += yy - 2 * np.sum(th * Hte) + np.sum(th * (Gte @ th))
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    return float(LAMS[j]), PS[i], grid


def fit_W(Y, L, idx, lam, p, kind="full") -> np.ndarray:
    F = Fitter(Y, L, idx)
    B, _ = getattr(F, kind)(p, [lam])[0]
    return B.sum(0)


def nested_r2(Y, L, rid) -> dict:
    """Outer 5 contiguous blocks; inner 4-fold choice of (lambda, p) per model; pooled held-out R2."""
    f = blocks(rid, K_OUT)
    acc = {m: np.zeros(3) for m in ("full", "ar")}
    folds = []
    for k in range(K_OUT):
        tr, te = f != k, f == k
        row = {"fold": k}
        for m in ("full", "ar"):
            lam, p, _ = choose(Y, L, rid, tr, m)
            F = Fitter(Y, L, tr)
            B, c = getattr(F, m)(p, [lam])[0]
            s = np.array(F.sse(te, B, c))
            acc[m] += s
            row |= {f"r2_{m}": 1 - s[0] / s[1], f"lam_{m}": lam, f"p_{m}": p}
        row["gain"] = row["r2_full"] - row["r2_ar"]
        folds.append(row)
    r2 = {m: 1 - acc[m][0] / acc[m][1] for m in acc}
    return {"r2_full": r2["full"], "r2_ar": r2["ar"], "gain": r2["full"] - r2["ar"],
            "gain_dchange": (acc["ar"][0] - acc["full"][0]) / acc["full"][2],
            "r2_full_dchange": 1 - acc["full"][0] / acc["full"][2], "r2_ar_dchange": 1 - acc["ar"][0] / acc["ar"][2],
            "folds_pos": int(sum(r["gain"] > 0 for r in folds)), "folds": folds}


def halves_idx(rid):
    u, cnt = np.unique(rid, return_counts=True)
    cum = np.cumsum(cnt)
    cut = u[np.argmin(np.abs(cum - cum[-1] / 2))]
    return rid <= cut, rid > cut


def session_fit(job) -> dict:
    """One (label, mode): nested-CV R2, session W at CV (lambda, p) and +-1 decade, halves W, AR-only W diag."""
    from wfield_local.parallel import pin_blas
    pin_blas(1)
    lab, mode = job
    t0 = time.perf_counter()
    runs, dt = load_runs(lab, mode)
    Y, L, rid = lagged(runs)
    allidx = np.ones(len(Y), bool)
    res = {"label": lab, "mode": mode, "n": len(Y), "n_runs": len(np.unique(rid)), "dt": dt}
    res |= nested_r2(Y, L, rid)
    lam, p, grid = choose(Y, L, rid, allidx, "full", K=K_OUT)
    res |= {"lam": lam, "p": p, "lam_edge": bool(lam in (LAMS[0], LAMS[-1]))}
    mats = {"W": fit_W(Y, L, allidx, lam, p), "W_up": fit_W(Y, L, allidx, lam * 10, p),
            "W_dn": fit_W(Y, L, allidx, lam / 10, p), "grid": grid}
    for h, idx in zip(("h1", "h2"), halves_idx(rid)):
        lh, ph, _ = choose(Y, L, rid, idx, "full", K=K_OUT)
        mats[f"W_{h}"] = fit_W(Y, L, idx, lh, ph)
        res |= {f"lam_{h}": lh, f"p_{h}": ph}
    res["seconds"] = time.perf_counter() - t0
    f = OUT / "sessions" / f"{lab}_{mode}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **mats, meta=json.dumps(res))
    print(f"{lab} {mode}: R2 full {res['r2_full']:.4f} AR {res['r2_ar']:.4f} gain {res['gain']:+.4f} "
          f"({res['gain_dchange'] * 100:+.2f} % of change var; folds+ {res['folds_pos']}/5) lam {lam:.1e} p {p} "
          f"{res['seconds']:.0f} s", flush=True)
    return res


def load_sess(lab, mode, sub="sessions"):
    d = np.load(OUT / sub / f"{lab}_{mode}.npz")
    return {k: d[k] for k in d.files if k != "meta"} | {"meta": json.loads(str(d["meta"]))}


# ---------------------------------------------------------------------------------------------------------------- gate
def gate() -> None:
    import pandas as pd
    od, dc, r_ = cp.offdiag, cp.dcenter, cp.r_
    pre = all_labels(pre_only=True)
    rows, sess = [], []
    for mode in MODES:
        S = {lab: load_sess(lab, mode) for lab in pre if (OUT / "sessions" / f"{lab}_{mode}.npz").exists()}
        for lab, s in S.items():
            m = s["meta"]
            sess.append({"animal": lab[:4], "label": lab, "mode": mode, **{k: m[k] for k in (
                "n", "r2_full", "r2_ar", "gain", "gain_dchange", "r2_full_dchange", "r2_ar_dchange", "folds_pos", "lam",
                "p", "lam_edge", "lam_h1", "p_h1", "lam_h2", "p_h2", "seconds")},
                "halves": r_(od(s["W_h1"]), od(s["W_h2"])), "halves_pair": r_(dc(s["W_h1"]), dc(s["W_h2"])),
                "ridge_up": r_(od(s["W"]), od(s["W_up"])), "ridge_dn": r_(od(s["W"]), od(s["W_dn"])),
                "ridge_up_pair": r_(dc(s["W"]), dc(s["W_up"])), "ridge_dn_pair": r_(dc(s["W"]), dc(s["W_dn"]))})
        for an in AN:
            labs = [x for x in S if x.startswith(an)]
            for a, b in itertools.combinations(labs, 2):
                rows.append({"animal": an, "mode": mode, "a": a, "b": b, "r": r_(od(S[a]["W"]), od(S[b]["W"])),
                             "r_pair": r_(dc(S[a]["W"]), dc(S[b]["W"]))})
    D, P = pd.DataFrame(sess), pd.DataFrame(rows)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "tables" / "gate_sessions.csv", index=False)
    P.to_csv(OUT / "tables" / "gate_session_pairs.csv", index=False)
    G = []
    for (mode, an), g in D.groupby(["mode", "animal"], sort=True):
        p = P[(P["mode"] == mode) & (P.animal == an)]
        r = {"mode": mode, "animal": an, "n_sess": len(g),
             "gain_med": g.gain.median(), "gain_dchange_med": g.gain_dchange.median(), "r2_full_med": g.r2_full.median(),
             "r2_ar_med": g.r2_ar.median(), "frac_gain_pos": (g.gain > 0).mean(), "folds_pos_med": g.folds_pos.median(),
             "halves_med": g.halves.median(), "halves_min": g.halves.min(), "sessions_med": p.r.median(),
             "ridge_up_med": g.ridge_up.median(), "ridge_dn_med": g.ridge_dn.median(),
             "halves_pair_med": g.halves_pair.median(), "sessions_pair_med": p.r_pair.median(),
             "lam_geomed": float(10 ** np.median(np.log10(g.lam))), "p_mode": int(g.p.mode().iloc[0]),
             "n_lam_edge": int(g.lam_edge.sum())}
        r["c1_pred"] = bool(r["frac_gain_pos"] >= GATE["pred_frac"] and r["folds_pos_med"] >= GATE["pred_folds"])
        r["c2_halves"] = bool(r["halves_med"] >= GATE["halves"] and r["halves_min"] >= GATE["halves_min"])
        r["c3_sessions"] = bool(r["sessions_med"] >= GATE["sessions"])
        r["c4_ridge"] = bool(min(r["ridge_up_med"], r["ridge_dn_med"]) >= GATE["ridge"])
        r["c5_pair"] = bool(r["halves_pair_med"] >= GATE["pair"])
        r["PASS"] = all(r[k] for k in ("c1_pred", "c2_halves", "c3_sessions", "c4_ridge", "c5_pair"))
        G.append(r)
    G = pd.DataFrame(G)
    G.to_csv(OUT / "tables" / "gate_animals.csv", index=False)
    pd.set_option("display.width", 250)
    print(D.groupby(["mode", "animal"])[["gain", "gain_dchange", "halves", "halves_pair", "ridge_up", "ridge_dn",
                                         "lam", "p", "seconds"]].median().round(4).to_string())
    print(G.round(3).T.to_string())
    for mode in MODES:
        g = G[G["mode"] == mode]
        print(f"{mode}: {int(g.PASS.sum())}/{len(g)} animals pass -> {'PASSED' if g.PASS.sum() >= 3 else 'FAILED'}"
              f"{' (primary)' if mode == 'raw' else ' (sensitivity, not gated)'}")


# ------------------------------------------------------------------------------------------------- fixed-penalty fits
def fixed_job(job) -> dict:
    """W of one (label, mode) at the animal's FIXED (lambda, p) [primary] -- and the per-session CV W is in sessions/."""
    from wfield_local.parallel import pin_blas
    pin_blas(1)
    lab, mode, lam, p = job
    runs, _ = load_runs(lab, mode)
    Y, L, rid = lagged(runs)
    W = fit_W(Y, L, np.ones(len(Y), bool), lam, p)
    f = OUT / "fixed" / f"{lab}_{mode}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, W=W, meta=json.dumps({"label": lab, "mode": mode, "lam": lam, "p": p, "n": len(Y)}))
    return {"label": lab}


def groups() -> dict:
    """Name -> list of (target index, source index)."""
    ix = {r: i for i, r in enumerate(REGIONS)}
    h = {"i": "left", "c": "right"}
    out = {}
    for hs, ht in itertools.product("ic", "ic"):
        out[f"S{hs}->M{ht}"] = [(ix[f"{m}_{h[ht]}"], ix[f"{s}_{h[hs]}"]) for s in SENS for m in MOT]
        out[f"M{hs}->S{ht}"] = [(ix[f"{s}_{h[ht]}"], ix[f"{m}_{h[hs]}"]) for s in SENS for m in MOT]
    return out


def group_values(W) -> dict:
    G = groups()
    v = {k: float(np.mean([W[a, b] for a, b in pr])) for k, pr in G.items()}
    for hs, ht in itertools.product("ic", "ic"):
        v[f"dir S{hs}<->M{ht}"] = v[f"S{hs}->M{ht}"] - v[f"M{ht}->S{hs}"]
    v["S->M intra"] = (v["Si->Mi"] + v["Sc->Mc"]) / 2
    v["S->M inter"] = (v["Si->Mc"] + v["Sc->Mi"]) / 2
    v["M->S intra"] = (v["Mi->Si"] + v["Mc->Sc"]) / 2
    v["M->S inter"] = (v["Mi->Sc"] + v["Mc->Si"]) / 2
    v["dir intra"] = v["S->M intra"] - v["M->S intra"]
    v["dir inter"] = v["S->M inter"] - v["M->S inter"]
    od = W[~np.eye(NR, dtype=bool)]
    v["mean |W| offdiag"] = float(np.abs(od).mean())
    return v


METRICS = ["Si->Mi", "Sc->Mc", "Si->Mc", "Sc->Mi", "Mi->Si", "Mc->Sc", "Mc->Si", "Mi->Sc", "S->M intra", "S->M inter",
           "M->S intra", "M->S inter", "dir Si<->Mi", "dir Sc<->Mc", "dir Si<->Mc", "dir Sc<->Mi", "dir intra",
           "dir inter", "mean |W| offdiag"]


def summarise() -> None:
    import pandas as pd

    from wfield_local import analysis_kit as ak
    rows = []
    for lab in all_labels():
        for mode in MODES:
            for fit, sub in (("fixed", "fixed"), ("cv", "sessions")):
                f = OUT / sub / f"{lab}_{mode}.npz"
                if not f.exists():
                    continue
                W = np.load(f)["W"]
                rows.append({"label": lab, "animal": lab[:4], "epoch": epoch(lab), "mode": mode, "fit": fit,
                             **group_values(W)})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "tables" / "group_weights_sessions.csv", index=False)
    pd.set_option("display.width", 250)
    T, A = [], []
    for (mode, fit), g in D.groupby(["mode", "fit"], sort=True):
        for m in METRICS:
            pre = {a: g[(g.animal == a) & (g.epoch == "pre")][m].tolist() for a in AN}
            for ep in EPS:
                for a in AN:
                    v = g[(g.animal == a) & (g.epoch == ep)][m]
                    A.append({"mode": mode, "fit": fit, "metric": m, "epoch": ep, "animal": a, "n": len(v),
                              "mean": v.mean(), "delta": v.mean() - np.mean(pre[a]) if len(v) else np.nan})
                if ep == "pre":
                    continue
                post = {a: g[(g.animal == a) & (g.epoch == ep)][m].tolist() for a in AN}
                post = {a: v for a, v in post.items() if v}
                iv = ak.boot_delta(post, {a: v for a, v in pre.items() if v}, np.random.default_rng(0))
                dl = [np.mean(post[a]) - np.mean(pre[a]) for a in post if pre.get(a)]
                T.append({"mode": mode, "fit": fit, "metric": m, "epoch": ep, "delta": iv.point if iv else np.nan,
                          "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan, "n_animals": len(dl),
                          "n_down": int(sum(x < 0 for x in dl)), "n_up": int(sum(x > 0 for x in dl)),
                          "pre_mean": float(np.mean([np.mean(v) for v in pre.values() if v]))})
    A, T = pd.DataFrame(A), pd.DataFrame(T)
    A.to_csv(OUT / "tables" / "group_weights_animal.csv", index=False)
    T.to_csv(OUT / "tables" / "group_weights_tests.csv", index=False)
    key = ["S->M intra", "S->M inter", "M->S intra", "M->S inter", "dir intra", "dir inter", "Si->Mi", "Sc->Mc",
           "mean |W| offdiag"]
    for mode in MODES:
        a = A[(A["mode"] == mode) & (A.fit == "fixed") & A.metric.isin(key)]
        print(f"\n=== PER-ANIMAL (mode {mode}, fixed lambda/p): epoch mean; delta vs Aug pre ===")
        print(a.pivot_table(index=["metric", "animal"], columns="epoch", values="delta").reindex(columns=EPS[1:])
              .round(4).to_string())
        print(a[a.epoch == "pre"].pivot_table(index="metric", columns="animal", values="mean").round(4).to_string())
    print("\n=== boot_delta (paired nested) ===")
    t = T[T.metric.isin(key)].copy()
    t["ci_excl0"] = (t.lo > 0) | (t.hi < 0)
    print(t.round(4).to_string())


# ------------------------------------------------------------------------------------------------------------- figures
def short(r):
    return cp.short(r)


def plot() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    fd, td = OUT / "figures", OUT / "tables"
    fd.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(td / "gate_sessions.csv")
    G = pd.read_csv(td / "gate_animals.csv")
    tag = "PROVISIONAL, n = 4 animals, AUGUST pre sessions"
    acol = dict(zip(AN, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    # FIG 1: held-out prediction, full MVAR vs AR-only
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for mode, mk in zip(MODES, ("o", "s", "^")):
        d = D[D["mode"] == mode]
        for an in AN:
            q = d[d.animal == an]
            ax[0].scatter(q.r2_ar, q.r2_full, s=16, marker=mk, c=acol[an], alpha=0.8,
                          label=f"{an}" if mode == "raw" else None)
            x = AN.index(an) + {"raw": -0.22, "gsr": 0, "dec": 0.22}[mode]
            ax[1].scatter(np.full(len(q), x) + np.linspace(-0.06, 0.06, len(q)), 100 * q.gain_dchange, s=14, marker=mk,
                          c=acol[an], alpha=0.8)
        ax[2].scatter(np.log10(d.lam) + {"raw": -0.06, "gsr": 0, "dec": 0.06}[mode], d.p + {"raw": -0.12, "gsr": 0, "dec": 0.12}[mode], s=14,
                      marker=mk, facecolors="none", edgecolors="k", label=mode)
    lo = min(D.r2_ar.min(), D.r2_full.min())
    hi = max(D.r2_ar.max(), D.r2_full.max())
    ax[0].plot([lo, hi], [lo, hi], c="0.6", lw=0.8)
    ax[0].set_xlabel("held-out one-step R2, AR-only (own past)")
    ax[0].set_ylabel("held-out one-step R2, full MVAR")
    ax[0].set_title("1a  per session (o raw, s global-removed, ^ deconvolved)", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].axhline(0, c="0.6", lw=0.8)
    ax[1].set_xticks(range(4), AN)
    ax[1].set_ylabel("cross-region gain, % of one-step CHANGE variance")
    ax[1].set_title("1b  gain = (SSE AR - SSE full) / SS(change), nested blocked CV\n(left raw, middle global-removed, "
                    "right deconvolved)", fontsize=9)
    ax[2].set_xlabel("log10 lambda (alpha = lambda x n), CV-chosen, whole session")
    ax[2].set_ylabel("lag order p (bins of 96 ms)")
    ax[2].set_yticks(PS)
    ax[2].axvline(np.log10(LAMS[0]), c="0.8", ls=":")
    ax[2].axvline(np.log10(LAMS[-1]), c="0.8", ls=":")
    ax[2].legend(fontsize=7, loc="center right")
    ax[2].set_title("1c  CV choices per session\n(dotted = lambda grid edges; p = 3 = top of the range)", fontsize=9)
    fig.suptitle(f"Region MVAR on encoder residuals: held-out prediction beyond each region's own past -- {tag}",
                 fontsize=10)
    fig.savefig(fd / "fig1_prediction.png", dpi=150)
    plt.close(fig)
    # FIG 2: the stability gate per animal
    crit = [("halves_med", "halves\n(median)", GATE["halves"]), ("halves_min", "halves\n(worst session)", GATE["halves_min"]), ("sessions_med", "sessions", GATE["sessions"]),
            ("ridge_up_med", "lambda x10", GATE["ridge"]), ("ridge_dn_med", "lambda /10", GATE["ridge"]),
            ("halves_pair_med", "halves,\npair-specific", GATE["pair"]),
            ("sessions_pair_med", "sessions,\npair-specific\n(not gated)", np.nan)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.5, 1]}, constrained_layout=True)
    for j, (col, _nm, th) in enumerate(crit):
        for mode, mk, dx in zip(MODES, ("o", "s", "^"), (-0.25, 0, 0.25)):
            g = G[G["mode"] == mode]
            for an in AN:
                v = g[g.animal == an][col]
                ax[0].scatter([j + dx + 0.04 * (AN.index(an) - 1.5)], v, marker=mk, c=acol[an], s=22,
                              label=(f"{an}" if (j == 0 and mode == "raw") else None))
        if np.isfinite(th):
            ax[0].hlines(th, j - 0.4, j + 0.4, colors="k", linestyles="--", lw=1)
    ax[0].set_xticks(range(len(crit)), [c[1] for c in crit], fontsize=8)
    ax[0].set_ylabel("median Pearson r of off-diagonal W (lag-summed)")
    ax[0].axhline(0, c="0.85", lw=0.8)
    ax[0].legend(fontsize=7, loc="lower left", title="o raw  s glob-rm  ^ deconv", title_fontsize=7)
    ax[0].set_title("2a  pre-registered stability criteria, medians over pre sessions unless noted (dashes = thresholds; raw is gated)", fontsize=9)
    # 2b: mean pre W (raw), pooled over animals' session means
    Ws = []
    for an in AN:
        labs = sorted(D[(D.animal == an) & (D["mode"] == "raw")].label)
        if labs:
            Ws.append(np.mean([load_sess(lab, "raw")["W"] for lab in labs], 0))
    M = np.mean(Ws, 0)
    np.save(td / "fig2b_mean_pre_W_raw.npy", M)
    Mp = M.copy()
    np.fill_diagonal(Mp, np.nan)
    v = np.nanmax(np.abs(Mp))
    im = ax[1].imshow(Mp, cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax[1], fraction=0.046, label="W (lag-summed, standardised)")
    ax[1].set_xticks(range(NR), [short(r) for r in REGIONS], rotation=90, fontsize=6)
    ax[1].set_yticks(range(NR), [short(r) for r in REGIONS], fontsize=6)
    ax[1].set_xlabel("source")
    ax[1].set_ylabel("target")
    ax[1].set_title("2b  mean pre-stroke W, raw (animal-weighted; diagonal blanked)", fontsize=9)
    fig.suptitle(f"MVAR stability gate ({int(G[G['mode'] == 'raw'].PASS.sum())}/4 animals pass on raw; >= 3 needed) "
                 f"-- {tag}", fontsize=10)
    fig.savefig(fd / "fig2_stability_gate.png", dpi=150)
    plt.close(fig)
    # FIG 3 (only if the post-gate tables exist): change vs pre
    tf = td / "group_weights_tests.csv"
    if tf.exists():
        T = pd.read_csv(tf)
        A = pd.read_csv(td / "group_weights_animal.csv")
        mets = ["S->M intra", "S->M inter", "M->S intra", "M->S inter", "dir intra", "dir inter"]
        fig, axs = plt.subplots(2, len(mets), figsize=(18, 7), sharey="row", constrained_layout=True)
        for i, mode in enumerate(("raw", "gsr")):
            for j, m in enumerate(mets):
                ax = axs[i, j]
                for k, ep in enumerate(EPS[1:]):
                    t = T[(T["mode"] == mode) & (T.fit == "fixed") & (T.metric == m) & (T.epoch == ep)]
                    if len(t):
                        t = t.iloc[0]
                        ax.errorbar(k, t.delta, yerr=[[t.delta - t.lo], [t.hi - t.delta]], fmt="o", c=COL[ep], ms=6,
                                    capsize=3)
                    a = A[(A["mode"] == mode) & (A.fit == "fixed") & (A.metric == m) & (A.epoch == ep)]
                    for an in AN:
                        q = a[a.animal == an].delta
                        if len(q):
                            ax.scatter(k + 0.25, q, s=12, c=COL[ep], marker="os^D"[AN.index(an)], alpha=0.7)
                ax.axhline(0, c="0.6", lw=0.8)
                ax.set_xticks(range(3), EPS[1:], fontsize=8)
                ax.set_title(f"3{'ab'[i]}{j + 1}  {m}", fontsize=9)
            axs[i, 0].set_ylabel(f"change vs Aug pre, W\n({'raw' if mode == 'raw' else 'global mode removed'})")
        fig.suptitle("Sensory <-> motor MVAR weights (mean of 16 signed lag-summed W, fixed per-animal lambda / p): "
                     "pooled paired-bootstrap 95% CI; small markers = animals (o PS92 s PS93 ^ PS94 D PS95) -- "
                     + tag.replace(", AUGUST pre sessions", ""), fontsize=9)
        fig.savefig(fd / "fig3_sensory_motor_change.png", dpi=150)
        plt.close(fig)
    print("figures ->", fd)


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "fit", "gate", "fixed", "summarise", "plot"])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--all", action="store_true", help="all curated sessions (default: August pre only)")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only or a.cmd == "plot":
        return plot()
    from wfield_local import analysis_kit as ak
    jobs = min(4, a.jobs)
    labs = a.labels or all_labels(pre_only=not a.all)
    t0 = time.time()
    if a.cmd == "prep":
        todo = [x for x in labs if not (DATA / f"{x}.npz").exists()]
        res, fail = ak.fan_sessions(todo, prep_job, jobs=jobs, label="prep")
        print(f"prepped {len(res)}, failed {fail}")
    elif a.cmd == "fit":
        todo = [(x, m) for x in labs for m in a.modes if not (OUT / "sessions" / f"{x}_{m}.npz").exists()]
        res, fail = ak.fan_sessions(todo, session_fit, jobs=jobs, label="fit")
        print(f"fitted {len(res)}, failed {fail}")
    elif a.cmd == "gate":
        gate()
    elif a.cmd == "fixed":
        import pandas as pd
        G = pd.read_csv(OUT / "tables" / "gate_animals.csv")
        if G[G["mode"] == "raw"].PASS.sum() < 3:
            sys.exit("gate FAILED on raw (< 3 animals): no pre vs post fits (pre-registered)")
        par = {(r.mode, r.animal): (r.lam_geomed, int(r.p_mode)) for r in G.itertuples()}
        todo = [(x, m) + par[(m, x[:4])] for x in all_labels() for m in a.modes]
        res, fail = ak.fan_sessions(todo, fixed_job, jobs=jobs, label="fixed")
        print(f"fixed fits {len(res)}, failed {fail}")
    elif a.cmd == "summarise":
        summarise()
    print(f"[{a.cmd}] wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
