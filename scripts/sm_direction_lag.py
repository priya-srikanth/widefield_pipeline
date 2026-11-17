"""SENSORY vs MOTOR TIMING OF LICK-DIRECTION INFORMATION (Priya 10-30; Lemke 2024 "direction as information"). PROVISIONAL.

Design, gate and readouts are PRE-REGISTERED in docs/status/DIRECTION_INFO_LAG_PREREG_20261030.md (written before any
fit) -- nothing here may move after seeing data; deviations go in that doc's Outcome section.

Per session, per 0.05-s bin of the -1..+1 s first-lick window, per group (S8 = SSp-n / SSp-m / SSs / SSp-ul, M8 = tjM1 /
ALM / MOp_rest / MOs_rest, both hemispheres; SiMi / ScMc 4 vs 4): ridge decode of the executed lick angle (total, and
within spout) from the group's area values at that bin; 5-fold CV within spout; score = r(prediction, angle) -> curve.
Readouts: L50 (half-max crossing of the baseline-subtracted curve), dL50 = L50_M - L50_S (+ = sensory first), xlag
(amplitude-invariant lag of the normalized curves, + = M lags S), peaks. N = 200 events x 10 draws per session (equal
across epochs); halves = odd / even events (gate); null = 20 angle permutations.

    python scripts/sm_direction_lag.py run [--all] [--labels ...] [--jobs 4]
    python scripts/sm_direction_lag.py gate
    python scripts/sm_direction_lag.py summarize        # refuses unless >= 3 / 4 animals pass
    python scripts/sm_direction_lag.py plot | --plot-only
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interhemi_crossarea as ihc  # noqa: E402
import iti_licks as il  # noqa: E402
import lick_interhemispheric as lih  # noqa: E402
import sm_coherence_cca as smc  # noqa: E402

OUT = lih.OUT_W / "sm_direction_lag"
FIGD, FIGS, TAB = OUT / "figdata", OUT / "figures", OUT / "tables"
PER = TAB / "per_session"
EPS, AN = lih.EPS, lih.AN
DT = 0.05
LAGS = np.round(np.asarray(lih.LAGS, float), 3)
NT = len(LAGS)
SETS = ("cued", "iti_dock")
TARGETS = ("total", "within")
PAIRINGS = {k: smc.PAIRINGS[k] for k in ("S8M8", "SiMi", "ScMc")}
N_EV, N_DRAW, MIN_N, N_PERM, K_FOLD = 200, 10, 60, 20, 5
LAM_PER_FEATURE = 1.0
BASE = (LAGS >= -1.0 - 1e-9) & (LAGS <= -0.7 + 1e-9)
PEAKW = (LAGS >= -0.5 - 1e-9) & (LAGS <= 1.0 + 1e-9)
I_START = int(np.flatnonzero(LAGS >= -0.7 - 1e-9)[0])
MAXLAG = 6
SIG_FRAC, SD_MAX, PASS_ANIMALS = 0.75, 0.05, 3


# ------------------------------------------------------------------------------------------------ decoding
def _folds(pos: np.ndarray, rng) -> np.ndarray:
    f = np.empty(len(pos), int)
    for v in np.unique(pos):
        ix = np.flatnonzero(pos == v)
        f[rng.permutation(ix)] = np.arange(len(ix)) % K_FOLD
    return f


def cv_curve(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """X (n, T, p), y (n,) -> r(t) of the 5-fold CV ridge prediction (features standardized on the training events)."""
    n, T, p = X.shape
    lam = LAM_PER_FEATURE * p
    pred = np.empty((n, T))
    for k in range(K_FOLD):
        tr, te = folds != k, folds == k
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Ztr, Zte = (Xtr - mu) / sd, (X[te] - mu) / sd                    # (n_tr, T, p)
        yc = ytr - ytr.mean()
        A = np.einsum("ntp,ntq->tpq", Ztr, Ztr) + lam * np.eye(p)
        b = np.einsum("ntp,n->tp", Ztr, yc)
        w = np.linalg.solve(A, b[..., None])[..., 0]                      # (T, p)
        pred[te] = np.einsum("ntp,tp->nt", Zte, w) + ytr.mean()
    pc, yc = pred - pred.mean(0), y - y.mean()
    den = np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum())
    return np.where(den > 0, (pc * yc[:, None]).sum(0) / np.where(den > 0, den, 1), 0.0)


def readouts(r: np.ndarray, null95_pk: float = np.nan) -> dict:
    b = float(r[BASE].mean())
    iw = np.flatnonzero(PEAKW)
    ip = int(iw[np.argmax(r[iw])])
    p = float(r[ip])
    out = {"b": b, "p": p, "tpk": float(LAGS[ip]), "sig": bool(p > null95_pk) if np.isfinite(null95_pk) else np.nan,
           "L50": np.nan, "cens": False}
    if np.isfinite(null95_pk) and not p > null95_pk:
        return out
    half = 0.5 * (p - b)
    c = r - b
    for k in range(I_START, ip + 1):
        if c[k] >= half:
            if k == I_START:
                out.update(L50=float(LAGS[k]), cens=True)
            else:
                f = (half - c[k - 1]) / (c[k] - c[k - 1])
                out["L50"] = float(LAGS[k - 1] + f * DT)
            break
    return out


def xlag(rS: np.ndarray, rM: np.ndarray) -> float:
    """Lag (s) maximizing corr(cS(t), cM(t + lag)) over -0.5..+1.0 s (overlap only); + = M lags S."""
    def norm(r):
        c = r - r[BASE].mean()
        pk = c[PEAKW].max()
        return c / pk if pk > 0 else c
    cS, cM = norm(rS), norm(rM)
    iw = np.flatnonzero(PEAKW)
    lo, hi = iw[0], iw[-1]
    ks = np.arange(-MAXLAG, MAXLAG + 1)
    cc = []
    for k in ks:
        t = np.arange(max(lo, lo - k), min(hi, hi - k) + 1)
        a, m = cS[t], cM[t + k]
        cc.append(np.corrcoef(a, m)[0, 1] if a.std() > 0 and m.std() > 0 else np.nan)
    cc = np.asarray(cc)
    if not np.isfinite(cc).any():
        return np.nan
    j = int(np.nanargmax(cc))
    d = 0.0
    if 0 < j < len(ks) - 1 and np.all(np.isfinite(cc[j - 1:j + 2])):
        den = cc[j - 1] - 2 * cc[j] + cc[j + 1]
        d = 0.5 * (cc[j - 1] - cc[j + 1]) / den if den != 0 else 0.0
    return float((ks[j] + d) * DT)


# ------------------------------------------------------------------------------------------------ per session
def _sets(label, meta, dock_t) -> dict:
    """{set: (Z (n, T, 45) area features / area SD, pos, angle, names)}."""
    S = ihc._sets(label, meta, dock_t)
    sd_tab = il.area_sd()
    out = {}
    for st in SETS:
        if st not in S:
            continue
        X, pos, ang, cols = S[st]
        names = il.area_names(cols)
        sd = sd_tab.loc[label, names].to_numpy(float)
        assert np.all(np.isfinite(sd) & (sd > 0)), label
        assert X.shape[1] == NT, (label, st, X.shape)
        ang = np.asarray(ang, float)
        ok = np.isfinite(ang)
        Z = il.area_features(X[ok].astype(np.float64), cols, label) / sd
        out[st] = (Z, np.asarray(pos).astype(str)[ok], ang[ok], names)
    return out


def session_work(item):
    label, meta, dock_t, _f = item
    from wfield_local import config, epochs
    seed = int(label[2:4]) * 10000 + int(label[-4:])
    an, ep = config.animal_of(label), epochs.epoch_of(label)
    rows, curves = [], {}
    for k_set, (st, (Z, pos, ang, names)) in enumerate(_sets(label, meta, dock_t).items()):
        n = len(Z)
        if n < MIN_N:
            continue
        ix = {nm: i for i, nm in enumerate(names)}
        sm = pd.Series(ang).groupby(pos).transform("mean").to_numpy()
        ys = {"total": ang, "within": ang - sm}
        allev = np.arange(n)
        for k_t, (tg, y) in enumerate(ys.items()):
            for k_pr, (pr, (sa, ma)) in enumerate(PAIRINGS.items()):
                G = {"S": Z[..., [ix[a] for a in sa]], "M": Z[..., [ix[a] for a in ma]]}
                rng = np.random.default_rng([seed, k_set, k_t, k_pr])
                base = {"label": label, "animal": an, "epoch": ep, "set": st, "target": tg, "pairing": pr, "n": n,
                        "small": n < N_EV}
                # full: N_EV events x N_DRAW draws, curves averaged
                R = {g: np.zeros(NT) for g in G}
                draws = [rng.choice(n, N_EV, replace=False) if n > N_EV else allev for _ in range(N_DRAW)]
                for ev in draws:
                    f = _folds(pos[ev], rng)
                    for g, X in G.items():
                        R[g] += cv_curve(X[ev], y[ev], f) / N_DRAW
                # null: permuted angle on the first draw
                ev0 = draws[0]
                f0 = _folds(pos[ev0], rng)
                NB, NP = {g: [] for g in G}, {g: [] for g in G}
                for _ in range(N_PERM):
                    yp = rng.permutation(y[ev0])
                    for g, X in G.items():
                        c = cv_curve(X[ev0], yp, f0)
                        NB[g].append(c)
                        NP[g].append(c[PEAKW].max())
                n95 = {g: float(np.percentile(NP[g], 95)) for g in G}
                rdo = {g: readouts(R[g], n95[g]) for g in G}
                rows.append({**base, "part": "full", **_flat(rdo, n95),
                             "dL50": rdo["M"]["L50"] - rdo["S"]["L50"], "xlag": xlag(R["S"], R["M"])})
                curves[(st, tg, pr)] = {"S": R["S"], "M": R["M"],
                                        "nullS95": np.percentile(NB["S"], 95, axis=0),
                                        "nullM95": np.percentile(NB["M"], 95, axis=0)}
                # halves (all events of each half, no subsampling); significance against the full session's null
                for part, ev in (("h1", allev[0::2]), ("h2", allev[1::2])):
                    f = _folds(pos[ev], rng)
                    Rh = {g: cv_curve(X[ev], y[ev], f) for g, X in G.items()}
                    rh = {g: readouts(Rh[g], n95[g]) for g in G}
                    rows.append({**base, "part": part, **_flat(rh, n95),
                                 "dL50": rh["M"]["L50"] - rh["S"]["L50"], "xlag": xlag(Rh["S"], Rh["M"])})
    return pd.DataFrame(rows), curves


def _flat(rdo: dict, n95: dict) -> dict:
    out = {}
    for g, d in rdo.items():
        for k, v in d.items():
            out[f"{k}_{g}"] = v
        out[f"null95_{g}"] = n95[g]
    return out


def run(labs, jobs) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    PER.mkdir(parents=True, exist_ok=True)
    todo = [x for x in labs if not (PER / f"{x}.pkl").exists()]
    print(f"run: {len(labs)} sessions, {len(todo)} to fit", flush=True)
    t0 = time.time()
    res, fail = ak.fan_sessions(smc._meta_items(todo, set()), session_work, jobs=min(jobs, 4), key=lambda it: it[0])
    for it, (df, cv) in res:
        with open(PER / f"{it[0]}.pkl", "wb") as fh:
            pickle.dump({"rows": df, "curves": cv}, fh)
    if fail:
        print("FAILED:", [str(f[0]) for f in fail], flush=True)
    print(f"sessions done {time.time() - t0:.0f} s", flush=True)
    collect()


def _load_all():
    out = {}
    for f in sorted(PER.glob("*.pkl")):
        with open(f, "rb") as fh:
            out[f.stem] = pickle.load(fh)
    return out


def collect() -> pd.DataFrame:
    S = pd.concat([d["rows"] for d in _load_all().values()], ignore_index=True)
    TAB.mkdir(parents=True, exist_ok=True)
    S.to_csv(TAB / "sessions.csv.gz", index=False)
    return S


# ------------------------------------------------------------------------------------------------ gate
def gate_table(S: pd.DataFrame, st="cued", tg="total", pr="S8M8") -> pd.DataFrame:
    P = S[(S.epoch == "pre") & (S.set == st) & (S.target == tg) & (S.pairing == pr)]
    rows = []
    for a in AN:
        g = P[P.animal == a]
        full = g[g.part == "full"]
        if full.empty:
            continue
        sig = (full.sig_S.astype(bool) & full.sig_M.astype(bool)).mean()
        h = g[g.part != "full"].pivot_table(index="label", columns="part", values="dL50")
        d = (h.get("h1") - h.get("h2")).dropna() if {"h1", "h2"} <= set(h.columns) else pd.Series(dtype=float)
        sd = float(np.sqrt(np.mean(d ** 2) / 2)) if len(d) else np.nan
        hx = g[g.part != "full"].pivot_table(index="label", columns="part", values="xlag")
        dx = (hx.get("h1") - hx.get("h2")).dropna() if {"h1", "h2"} <= set(hx.columns) else pd.Series(dtype=float)
        rows.append({"set": st, "target": tg, "pairing": pr, "animal": a, "n_pre": len(full),
                     "frac_sig": sig, "n_half_pairs": len(d), "halfSD_dL50": sd,
                     "halfSD_xlag": float(np.sqrt(np.mean(dx ** 2) / 2)) if len(dx) else np.nan,
                     "betweenSD_dL50": float(full.dL50.std()), "pre_dL50": float(full.dL50.mean()),
                     "pre_xlag": float(full.xlag.mean()), "pre_pS": float(full.p_S.mean()),
                     "pre_pM": float(full.p_M.mean()),
                     "pass": bool(sig >= SIG_FRAC and np.isfinite(sd) and sd <= SD_MAX)})
    return pd.DataFrame(rows)


def gate() -> pd.DataFrame:
    S = collect()
    G = pd.concat([gate_table(S, st, tg, pr) for st in SETS for tg in TARGETS for pr in PAIRINGS],
                  ignore_index=True)
    G.to_csv(TAB / "gate_animals_all.csv", index=False)
    prim = G[(G.set == "cued") & (G.target == "total") & (G.pairing == "S8M8")]
    prim.to_csv(TAB / "gate_animals.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== GATE (primary: cued, total angle, S8M8; August pre) ===")
    print(prim.round(3).to_string(index=False))
    print(f"PASS {int(prim['pass'].sum())} / {len(prim)} (need >= {PASS_ANIMALS})")
    print("\n=== not gated: other targets / pairings / sets ===")
    print(G.groupby(["set", "target", "pairing"])["pass"].sum().to_string())
    return G


def passed() -> bool:
    G = pd.read_csv(TAB / "gate_animals.csv")
    return int(G["pass"].sum()) >= PASS_ANIMALS


# ------------------------------------------------------------------------------------------------ summary
MET = ["dL50", "xlag", "L50_S", "L50_M", "p_S", "p_M"]


def summarise() -> None:
    if not passed():
        sys.exit("primary readout failed the pre-registered gate: summarize refused")
    S = collect()
    F = S[S.part == "full"].copy()
    keys = ["set", "target", "pairing"]
    for kv, g in F[F.pairing == "S8M8"].groupby(["set", "target"]):
        P = g.groupby(["animal", "epoch"])[MET].mean().unstack("epoch")
        print(f"\n=== PER-ANIMAL {kv} epoch means ===\n{P.round(3).to_string()}", flush=True)
    T = ihc._tests(F, keys, MET, np.random.default_rng(0))
    T.to_csv(TAB / "tests_boot_delta.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== boot_delta vs August pre (change = CI excl. 0 AND >= 3/4 same sign) ===")
    print(T[T.pairing == "S8M8"][keys + ["metric", "epoch", "pre_level", "delta", "lo", "hi", "n_same_sign",
                                         "excl0"] + [f"d_{a}" for a in AN]].round(3).to_string(index=False))
    FIGD.mkdir(parents=True, exist_ok=True)
    with open(FIGD / "summary_tables.pkl", "wb") as fh:
        pickle.dump({"T": T, "F": F}, fh)


# ------------------------------------------------------------------------------------------------ figures
def curves_data() -> dict:
    """Per animal x epoch mean curves (cued, S8M8), both targets -> figdata/fig1_curves.pkl."""
    from wfield_local import config, epochs
    acc = {}
    for lab, d in _load_all().items():
        a, ep = config.animal_of(lab), epochs.epoch_of(lab)
        for (st, tg, pr), c in d["curves"].items():
            if st != "cued" or pr != "S8M8":
                continue
            for g in ("S", "M"):
                acc.setdefault((tg, g, a, ep), []).append(c[g])
                acc.setdefault((tg, f"null{g}95", a, ep), []).append(c[f"null{g}95"])
    out = {"lags": LAGS, "curves": {k: np.mean(v, 0) for k, v in acc.items()},
           "n": {k: len(v) for k, v in acc.items()}}
    FIGD.mkdir(parents=True, exist_ok=True)
    with open(FIGD / "fig1_curves.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return out


def plot_curves(plot_only=False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if plot_only:
        with open(FIGD / "fig1_curves.pkl", "rb") as fh:
            D = pickle.load(fh)
    else:
        D = curves_data()
    L, C = D["lags"], D["curves"]
    cols = {"S": "#1b7837", "M": "#762a83"}
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True, constrained_layout=True)
    for i, tg in enumerate(TARGETS):
        for j, ep in enumerate(EPS):
            ax = axs[i, j]
            for g in ("S", "M"):
                cs = [C[(tg, g, a, ep)] for a in AN if (tg, g, a, ep) in C]
                for c in cs:
                    ax.plot(L, c, color=cols[g], lw=0.6, alpha=0.35)
                if cs:
                    ax.plot(L, np.mean(cs, 0), color=cols[g], lw=2.2,
                            label=("sensory (8 areas)" if g == "S" else "motor (8 areas)"))
                nl = [C[(tg, f"null{g}95", a, ep)] for a in AN if (tg, f"null{g}95", a, ep) in C]
                if nl:
                    ax.plot(L, np.mean(nl, 0), color=cols[g], lw=0.8, ls=":")
            ax.axvline(0, color="0.5", lw=0.7)
            ax.axhline(0, color="0.7", lw=0.5)
            if i == 0:
                ax.set_title(ep)
            if j == 0:
                ax.set_ylabel(("total angle" if tg == "total" else "within-spout angle") + "\nCV decoding r")
            if i == 1:
                ax.set_xlabel("time from first lick (s)")
    axs[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Lick-angle information over time, sensory vs motor areas (cued first licks; thin = animals, "
                 "thick = mean of animals, dotted = permutation 95th pct)", fontsize=10)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig1_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", FIGS / "fig1_curves.png")


def lag_data() -> dict:
    S = collect()
    F = S[(S.part == "full") & (S.set == "cued") & (S.pairing == "S8M8")]
    P = F.groupby(["target", "animal", "epoch"])[["dL50", "xlag"]].mean().reset_index()
    G = pd.read_csv(TAB / "gate_animals.csv")
    T = pd.read_csv(TAB / "tests_boot_delta.csv") if (TAB / "tests_boot_delta.csv").exists() else None
    out = {"P": P, "G": G, "T": T}
    with open(FIGD / "fig2_lags.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return out


def plot_lags(plot_only=False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    if plot_only:
        with open(FIGD / "fig2_lags.pkl", "rb") as fh:
            D = pickle.load(fh)
    else:
        D = lag_data()
    P, G, T = D["P"], D["G"], D["T"]
    acol = dict(zip(AN, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    for k, (tg, m) in enumerate([(t, m) for t in TARGETS for m in ("dL50", "xlag")]):
        ax = axs[k]
        q = P[P.target == tg]
        allv = []
        for a in AN:
            v = q[q.animal == a].set_index("epoch").reindex(EPS)[m].to_numpy(float) * 1000
            allv += list(v[np.isfinite(v)])
            ok = G[G.animal == a]["pass"].any() if len(G) else False
            ax.plot(range(len(EPS)), v, "-o", color=acol[a], ms=5, lw=1.2, alpha=1 if ok else 0.45,
                    label=f"{a}{'' if ok else ' (gate fail)'}")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xticks(range(len(EPS)), EPS)
        if allv:
            lo, hi = min(allv + [0]), max(allv + [0])
            pad = 0.12 * (hi - lo + 1)
            ax.set_ylim(lo - pad, hi + 3 * pad)                        # top band reserved for stars
        if T is not None and len(T):
            tt = T[(T.set == "cued") & (T.target == tg) & (T.pairing == "S8M8") & (T.metric == m)]
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            for _, r in tt.iterrows():
                if r.excl0 and r.n_same_sign >= 3:
                    ax.text(EPS.index(r.epoch), 0.95, "*", transform=tr, ha="center", va="top", fontsize=14,
                            fontweight="bold")
        lab = {"dL50": "half-max lag M - S (ms)\n+ = sensory first", "xlag": "curve-shape lag (ms)\n+ = motor lags"}
        ax.set_ylabel(lab[m])
        ax.set_title(f"{'total' if tg == 'total' else 'within-spout'} angle: {m}", fontsize=10)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Sensory vs motor timing of lick-angle information, per animal (cued first licks, S8 vs M8)",
                 fontsize=10)
    fig.savefig(FIGS / "fig2_lags.png", dpi=150)
    plt.close(fig)
    print("wrote", FIGS / "fig2_lags.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "gate", "summarise", "plot"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        labs = a.labels or smc.labels(pre_only=not a.all)
        run(labs, a.jobs)
    elif a.cmd == "gate":
        gate()
    elif a.cmd == "summarise":
        summarise()
    else:
        plot_curves(a.plot_only)
        plot_lags(a.plot_only)


if __name__ == "__main__":
    main()
