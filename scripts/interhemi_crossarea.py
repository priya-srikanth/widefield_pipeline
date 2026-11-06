"""INTERHEMISPHERIC FOLLOW-UPS + CROSS-AREA CORRELATION STRUCTURE OVER RECOVERY (10-30; PROVISIONAL, n = 4).

TASK 1 -- follow-ups of lick_interhemispheric.py (FINDINGS "INTERHEMISPHERIC COORDINATION"): the post-stroke rise in
interhemispheric noise correlation was GLOBAL (equal in VISpm / RSPd and at docked rest); the xcorr shape tilted
contralesional-earlier and Granger direction moved contralesional, also at rest. Two controls, on the SAME sets / windows /
residual model / metric code (functions imported from lick_interhemispheric, not re-written):
  gsr   GLOBAL SIGNAL REMOVED, leave-pair-out: g = unweighted mean of all 45 decoder-basis areas (il.area_features, split
        basis: ALM / tjM1 / MOs_rest / MOp_rest disjoint) EXCLUDING the pair's own areas (both hemispheres; for the oro
        pool all 10 pool areas), residualised on the same design. Noise correlations become PARTIAL correlations given the
        matching g summary (window mean / post-minus-base); time series have beta_h * g removed (one beta per hemisphere
        per window, pooled over events and lags). Leave-pair-out avoids the degenerate case where g is mostly the pair.
  bal   HEMISPHERIC BALANCE PARTIALLED: per event b = (contra - ipsi) of the residual (w1: window mean; w2: post 0.1..0.5
        minus base -1..-0.5 s, the primary balance window), regressed out of every lag of both hemispheres (per-lag beta)
        BEFORE the cross-correlation / Granger. (Noise correlation is degenerate after this -- c and i become identical
        up to the b direction -- so it is not reported for bal.)
  gsr_bal  both, in that order.
  Session level: each lead metric adjusted for the session's (i) log SD ratio contra / ipsi of the residual traces (a gain /
  SNR asymmetry proxy) and (ii) the cued matched-error contralesional share (balmatch_sessions bal_spout, set 'trial'),
  using a slope estimated WITHIN animal x epoch only (so the epoch effect itself cannot set the slope); change vs pre
  recomputed on the adjusted values.
SETS: cued (firstlick_full, uncapped), iti_all (every decoder ITI lick), iti_dock (ITI licks with the spout DOCKED,
  iti_dock_audit 'dock' flag, matched on t -- 35 % of ITI licks), rest (docked engaged quiet rest pseudo-events,
  interhemi_rest_w1). Windows: w1 = -0.5..+0.5 s on every set (lick vs rest comparable), w2 = -1..+1 s lick sets only.

TASK 2 -- CROSS-AREA (Priya 10-30: "cross-area dynamics (synchronization / correlation across areas) pre vs post?"): all
  45 areas (24 contralesional, 21 ipsilesional: VISp / VISl / VISal left are off every session's mask), per session, in
  docked rest, ITI docked licks, ITI all licks and cued first licks, on the w1 window (-0.5..+0.5 s), after the same
  residual model (spout dummies + lick angle; rest intercept only):
    slow  NOISE correlation = correlation ACROSS EVENTS of the window-mean residual (event-to-event co-fluctuation; a
          sub-1-Hz readout by construction)
    fast  WITHIN-WINDOW correlation of the event-de-meaned residual traces (concatenated over events; ~1-10 Hz part,
          what "synchronisation" usually means; GCaMP6s smooths it)
  Variants: raw, and GS-controlled = partial correlation of each pair given the mean of ALL OTHER 43 areas (leave-pair-out,
  closed form from the 45 x 45 covariance). Band-pass filtering is NOT applied: windows are 1 s, so 0.1-1 Hz cannot be
  resolved inside them; slow vs fast IS the band split available, and slow drift (< 1 / DRIFT_MIN_PERIOD) was already
  removed upstream (encoder slow-drift basis).
  Readouts (a) mean Fisher z within ipsi / within contra / between hemispheres (homotopic, heterotopic), orofacial network
  (ALM, tjM1, SSp-m, SSp-n, SSs, both hemispheres) vs the other bilateral areas, on the 21 BILATERAL areas so the two
  hemispheres are compared on the same areas; (b) change-vs-pre matrices (animal-weighted mean of per-animal epoch-mean z
  minus August-pre mean z; ipsi block left / top) + consistency mask (>= 3/4 animals same sign as the pooled change);
  (c) synchrony: PC1 variance fraction of the area correlation matrix and mean R^2 of each area on the global mean, both
  on fixed-size event subsamples (N_SUB events, N_DRAW draws: eigenvalue spread depends on n) + mean pairwise z (all 990
  pairs); (d) within-animal partial r (| log day, post sessions) with far-contra cued aim (recmark_sessions) and far-contra
  hit rate (qc/trial_attempts, as recovery_timing.py) -- many tests, count reported against the 5 % expected.
Statistics: per-animal tables printed BEFORE the bootstrap (rule 8); analysis_kit.boot_delta (paired nested) vs AUGUST pre.
Outputs: <OUT_W>/interhemi_followup/ (ihf_*), <OUT_W>/cross_area/ (xa_*), each with figdata/ + figures/.
Usage: python scripts/interhemi_crossarea.py [--jobs 4] [--labels ...] [--summarise] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import pickle
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_interhemispheric as lih  # noqa: E402
from wfield_local import lp_licks as _lp_licks  # noqa: E402  (coordinator 11-04: the run's attempts table)

OUT_W = lih.OUT_W
O1, O2 = OUT_W / "interhemi_followup", OUT_W / "cross_area"
IN = il.OUT
EPS = lih.EPS
AN = lih.AN
N_BOOT = 2000
MIN_N = 30
N_SUB, N_DRAW = 30, 20
TAUS = (0.0, 0.6)
PAIRS1 = ["oro", "SSp-n", "SSp-m", "tjM1", "ALM", "VISpm", "RSPd"]
POOL = lih.POOL
SETS1 = ["cued", "iti_all", "iti_dock", "rest"]
SETS2 = ["rest", "iti_dock", "iti_all", "cued"]
VARIANTS = ["raw", "bal", "gsr", "gsr_bal"]
LEAD = ["cc0", "asym", "lag_peak", "gc_dir"]
NCM = ["nc"]
ORDER_BASE = ["PL", "ACAd", "ALM", "MOs_rest", "tjM1", "MOp_rest", "SSp-m", "SSp-n", "SSp-un", "SSs", "SSp-bfd",
              "SSp-ul", "SSp-ll", "SSp-tr", "RSPagl", "RSPd", "RSPv", "VISa", "VISam", "VISrl", "VISal", "VISl", "VISpm",
              "VISp"]
W2_POST, W2_BASE = lih.W_POST, lih.W_BASE


def area_order(names: list[str]) -> list[int]:
    """Indices of names: ipsilesional (_left) block first, then contralesional (_right), each in ORDER_BASE."""
    out = []
    for h in ("left", "right"):
        out += [names.index(f"{b}_{h}") for b in ORDER_BASE if f"{b}_{h}" in names]
    assert len(out) == len(names), set(names) - {names[i] for i in out}
    return out


# ------------------------------------------------------------------------------------------------ helpers
def _pcorr_lpo(C: np.ndarray) -> np.ndarray:
    """Leave-pair-out global partial correlation from covariance C (F x F): pair (i, j) controlling h = S - x_i - x_j,
    S = sum of all F areas (scale of h is irrelevant to a partial correlation)."""
    s = C.sum(1)
    V = C.sum()
    d = np.diag(C)
    ci = s[:, None] - d[:, None] - C          # cov(x_i, h_ij)
    cj = s[None, :] - d[None, :] - C          # cov(x_j, h_ij)
    vh = V - 2 * s[:, None] - 2 * s[None, :] + d[:, None] + d[None, :] + 2 * C
    pc = C - ci * cj / vh
    vi = d[:, None] - ci ** 2 / vh
    vj = d[None, :] - cj ** 2 / vh
    P = pc / np.sqrt(vi * vj)
    np.fill_diagonal(P, 1.0)
    return P


def _cov2corr(C):
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return R


def _sync(Z: np.ndarray) -> tuple[float, float]:
    """(PC1 fraction of the correlation matrix, mean R^2 of each column on the column mean) of samples Z (m, F)."""
    Zc = Z - Z.mean(0)
    sd = Zc.std(0)
    Zs = Zc / np.where(sd > 0, sd, 1)
    R = Zs.T @ Zs / len(Zs)
    ev = np.linalg.eigvalsh(R)
    g = Zc.mean(1)
    g = g - g.mean()
    r = (Zc * g[:, None]).sum(0) / np.sqrt((Zc ** 2).sum(0) * (g ** 2).sum())
    return float(ev[-1] / ev.sum()), float(np.mean(r ** 2))


def _partial_nc(c, i, g) -> float:
    """Fisher z of corr(c, i | g) across events."""
    D = np.c_[np.ones(len(g)), g]
    rc = c - D @ np.linalg.lstsq(D, c, rcond=None)[0]
    ri = i - D @ np.linalg.lstsq(D, i, rcond=None)[0]
    return float(np.arctanh(np.clip(lih._corr(rc, ri), -0.999, 0.999)))


def _rm_global(E: np.ndarray, G: np.ndarray) -> np.ndarray:
    """E (n, T, 2) minus beta_h * G (n, T), one beta per hemisphere pooled over events and lags."""
    gg = (G * G).sum()
    b = (E * G[..., None]).sum((0, 1)) / gg
    return E - G[..., None] * b


def _rm_bal(E: np.ndarray, b: np.ndarray) -> np.ndarray:
    D = np.c_[np.ones(len(b)), b]
    return lih._resid(E, D)


def _sets(label: str, meta: pd.DataFrame, dock_t: np.ndarray) -> dict:
    S = lih._set_arrays(label, meta)
    z = il._load(label)
    cols = [str(c) for c in z["cols"]]
    t = np.round(z["iti_t"].astype(float), 4)
    ang = lih._angle(z["iti_side"], meta.set_index("tk").reindex(t).reach_mm.to_numpy(float))
    ok = np.isfinite(ang) & np.isin(t, dock_t)
    S["iti_dock"] = (z["iti_X"][ok], z["iti_pos"].astype(str)[ok], ang[ok], cols)
    return {k: S[k] for k in ("cued", "iti_all", "iti_dock", "rest") if k in S}


class Item(tuple):
    def __repr__(self):
        return str(self[0])


# ------------------------------------------------------------------------------------------------ per session
def session_work(item):
    label, meta, dock_t = item
    from wfield_local import config, epochs
    seed = int(label[2:4]) * 10000 + int(label[-4:])
    an, ep = config.animal_of(label), epochs.epoch_of(label)
    rows1, rows2, mats = [], [], {}
    k_rng = 0
    for st, (X, pos, ang, cols) in _sets(label, meta, dock_t).items():
        n = len(X)
        if n < MIN_N:
            continue
        X = X.astype(np.float64)
        D = lih._design(pos, ang, n)
        A = il.area_features(X, cols, label)                     # (n, T, 45)
        names = il.area_names(cols)
        Ar = lih._resid(A, D)
        lick = X.shape[1] == len(lih.LAGS)
        # ------------------------------------------------ TASK 2: cross-area matrices on w1
        A1 = Ar[:, lih.IDX_W1] if lick else Ar
        W = A1.mean(1)                                           # slow: (n, 45)
        Fd = (A1 - A1.mean(1, keepdims=True))                    # fast: (n, T, 45)
        rng2 = np.random.default_rng([seed, SETS2.index(st) if st in SETS2 else 9])
        for unit, Z, Zev in (("slow", W, None), ("fast", Fd.reshape(-1, Fd.shape[-1]), Fd)):
            C = np.cov(Z, rowvar=False)
            R = _cov2corr(C)
            P = _pcorr_lpo(C)
            mats[(st, unit)] = (R.astype(np.float32), P.astype(np.float32))
            pc1, gr2 = [], []
            for _ in range(N_DRAW):
                ix = rng2.choice(n, N_SUB, replace=False)
                zz = W[ix] if unit == "slow" else Zev[ix].reshape(-1, Zev.shape[-1])
                a, b = _sync(zz)
                pc1.append(a)
                gr2.append(b)
            pf, gf = _sync(Z)
            rows2.append({"label": label, "animal": an, "epoch": ep, "set": st, "unit": unit, "n": n,
                          "pc1_sub": float(np.mean(pc1)), "gsr2_sub": float(np.mean(gr2)), "pc1_full": pf,
                          "gsr2_full": gf})
        # ------------------------------------------------ TASK 1: interhemispheric follow-ups
        if st not in SETS1:
            continue
        PT = lih._pair_traces(X, cols, label)
        for p in PAIRS1:
            Y = PT[p]
            E = lih._resid(Y, D)
            excl = {f"{a}_{h}" for a in (POOL if p == "oro" else [p]) for h in ("left", "right")}
            keep = [j for j, nm in enumerate(names) if nm not in excl]
            G = Ar[..., keep].mean(-1)                           # (n, T) leave-pair-out global, residualised
            wins = {}
            if lick:
                wins["w2"] = (E, G, "post")
                wins["w1"] = (E[:, lih.IDX_W1], G[:, lih.IDX_W1], "win")
            else:
                wins["w1"] = (E, G, "win")
            for win, (Ew, Gw, kind) in wins.items():
                if kind == "post":
                    summ = lambda x: x[:, W2_POST].mean(1) - x[:, W2_BASE].mean(1)  # noqa: E731
                else:
                    summ = lambda x: x.mean(1)  # noqa: E731
                ce, ie, ge = summ(Ew[..., 0]), summ(Ew[..., 1]), summ(Gw)
                nc_raw = float(np.arctanh(np.clip(lih._corr(ce, ie), -0.999, 0.999)))
                nc_gsr = _partial_nc(ce, ie, ge)
                Eg = _rm_global(Ew, Gw)
                b_raw = ce - ie
                cg, ig = summ(Eg[..., 0]), summ(Eg[..., 1])
                var = {"raw": (Ew, nc_raw), "gsr": (Eg, nc_gsr), "bal": (_rm_bal(Ew, b_raw), np.nan),
                       "gsr_bal": (_rm_bal(Eg, cg - ig), np.nan)}
                lsd = float(np.log(Ew[..., 0].std() / Ew[..., 1].std()))
                g_share = float(np.mean([lih._corr(Ew[..., h].ravel(), Gw.ravel()) ** 2 for h in (0, 1)]))
                for vname, (Ev, nc) in var.items():
                    Ed = Ev - Ev.mean(1, keepdims=True)
                    for tau in TAUS:
                        Et = lih._deconv(Ed, tau)
                        Et = Et - Et.mean(1, keepdims=True)
                        cc = lih.xcorr(Et, lih.MAXLW[win])
                        k_rng += 1
                        gc = lih.granger(Et, np.random.default_rng([seed, k_rng]))
                        rows1.append({"label": label, "animal": an, "epoch": ep, "set": st, "win": win, "pair": p,
                                      "variant": vname, "tau": tau, "n": n, "nc": nc if tau == 0 else np.nan,
                                      **lih.cc_summary(cc), "gc_dir": gc["gc_dir"], "lsd": lsd, "g_r2": g_share})
    return pd.DataFrame(rows1), pd.DataFrame(rows2), mats, names


# ------------------------------------------------------------------------------------------------ run
def run(labels, jobs) -> None:
    from wfield_local import analysis_kit as ak
    M = lih._meta()
    Aud = pd.read_csv(IN / "iti_dock_audit.csv", usecols=["label", "t", "dock"])
    Aud = Aud[Aud.dock]
    dk = {x: np.round(g.t.to_numpy(float), 4) for x, g in Aud.groupby("label")}
    empty = pd.DataFrame(columns=["tk", "reach_mm", "gap", "since_stop", "to_strobe"])
    items = [Item((x, M.get(x, empty), dk.get(x, np.zeros(0)))) for x in labels]
    t0 = time.time()
    res, fail = ak.fan_sessions(items, session_work, jobs=min(jobs, 4), key=lambda it: it[0])
    if fail:
        print("FAILED:", [str(f[0]) for f in fail][:10], flush=True)
    print(f"sessions done {time.time() - t0:.0f} s", flush=True)
    S1 = pd.concat([v[0] for _k, v in res], ignore_index=True)
    S2 = pd.concat([v[1] for _k, v in res], ignore_index=True)
    names = res[0][1][3]
    order = area_order(names)
    keys = sorted({k for _k, v in res for k in v[2]})
    lab_list = [str(k[0]) for k, _v in res]
    # matrices: (session, key, variant raw/gs, F, F) in display order, NaN where the set was skipped
    MZ = np.full((len(res), len(keys), 2, len(names), len(names)), np.nan, np.float32)
    for i, (_k, v) in enumerate(res):
        for j, kk in enumerate(keys):
            if kk in v[2]:
                for q in (0, 1):
                    MZ[i, j, q] = v[2][kk][q][np.ix_(order, order)]
    tag = "_test" if len(labels) < 100 else ""
    for d in (O1, O2):
        d.mkdir(parents=True, exist_ok=True)
    S1.to_csv(O1 / f"ihf_sessions{tag}.csv.gz", index=False)
    S2.to_csv(O2 / f"xa_sync_sessions{tag}.csv.gz", index=False)
    np.savez_compressed(O2 / f"xa_matrices{tag}.npz", M=MZ, labels=np.array(lab_list),
                        keys=np.array([f"{a}|{b}" for a, b in keys]), names=np.array([names[i] for i in order]),
                        variants=np.array(["raw", "gs_lpo"]))
    if fail and not tag:
        print("WARNING: partial table written (see FAILED above)")


# ------------------------------------------------------------------------------------------------ summaries
def _aug(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import config
    return S[(S.epoch != "pre") | (~S.label.map(config.early_pre))].copy()


def _tests(SS: pd.DataFrame, keys: list[str], metrics: list[str], rng) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    T = []
    for kv, g in SS.groupby(keys, sort=True):
        kv = kv if isinstance(kv, tuple) else (kv,)
        for m in metrics:
            if m not in g or g[m].notna().sum() == 0:
                continue
            pre = {a: v[m].dropna().to_list() for a, v in g[g.epoch == "pre"].groupby("animal")}
            pre = {a: v for a, v in pre.items() if v}
            lv = ak.boot_ci(pre, rng, n_boot=N_BOOT)
            for ep in EPS[1:]:
                post = {a: v[m].dropna().to_list() for a, v in g[g.epoch == ep].groupby("animal")}
                post = {a: v for a, v in post.items() if v}
                iv = ak.boot_delta(post, pre, rng, n_boot=N_BOOT)
                if iv is None:
                    continue
                an_ = sorted(set(post) & set(pre))
                dl = {a: np.mean(post[a]) - np.mean(pre[a]) for a in an_}
                T.append({**dict(zip(keys, kv)), "metric": m, "epoch": ep, "pre_level": lv.point if lv else np.nan,
                          "delta": iv.point, "lo": iv.lo, "hi": iv.hi, "n_animals": len(an_),
                          "n_same_sign": int(sum(np.sign(x) == np.sign(iv.point) for x in dl.values())),
                          "excl0": bool(iv.lo > 0 or iv.hi < 0), **{f"d_{a}": dl.get(a, np.nan) for a in AN}})
    return pd.DataFrame(T)


def _recov_targets() -> pd.DataFrame:
    R = pd.read_csv(IN / "recmark_sessions.csv", usecols=["label", "day", "aim"]).drop_duplicates("label")
    T = pd.read_csv(_lp_licks.attempts_path(), usecols=["label", "pos", "cls"])
    T = T[T.pos == "far_contra"]
    H = T.assign(hit=T.cls.isin(["hit", "pre_hit"])).groupby("label").hit.mean().rename("fc_hit").reset_index()
    return R.merge(H, on="label", how="outer")


def _recovery(SS: pd.DataFrame, keys: list[str], metrics: list[str], rng) -> pd.DataFrame:
    X = SS[SS.epoch != "pre"].merge(_recov_targets(), on="label", how="left")
    X = X[X.day > 0]
    X["lday"] = np.log(X.day)
    out = []
    for kv, g in X.groupby(keys, sort=True):
        kv = kv if isinstance(kv, tuple) else (kv,)
        for m in metrics:
            for tgt in ("aim", "fc_hit"):
                h = g.dropna(subset=[m, tgt, "lday"])
                rs, res = {}, []
                for a, q in h.groupby("animal"):
                    if len(q) < 5:
                        continue
                    D = np.c_[np.ones(len(q)), q.lday]
                    ry = q[m].to_numpy() - D @ np.linalg.lstsq(D, q[m].to_numpy(), rcond=None)[0]
                    rx = q[tgt].to_numpy() - D @ np.linalg.lstsq(D, q[tgt].to_numpy(), rcond=None)[0]
                    rs[a] = lih._corr(ry, rx)
                    res.append((ry, rx))
                if len(rs) < 3:
                    continue
                ry = np.concatenate([a for a, _ in res])
                rx = np.concatenate([b for _, b in res])
                r0 = lih._corr(ry, rx)
                nul = [lih._corr(np.concatenate([rng.permutation(a) for a, _ in res]), rx) for _ in range(1000)]
                pv = (1 + np.sum(np.abs(nul) >= abs(r0))) / 1001
                out.append({**dict(zip(keys, kv)), "metric": m, "target": tgt, "r_pooled_within": r0, "p_perm": pv,
                            "n_animals": len(rs), "n_same_sign": int(sum(np.sign(v) == np.sign(r0) for v in rs.values())),
                            **{f"r_{a}": rs.get(a, np.nan) for a in AN}, "n_sess": len(h)})
    return pd.DataFrame(out)


def _print_animal(SS, keys, m, title, filt=None):
    q = SS if filt is None else SS.query(filt)
    P = q.groupby(keys + ["animal", "epoch"])[m].mean().unstack("epoch").reindex(columns=EPS)
    print(f"\n=== PER-ANIMAL ({title}) : {m}\n" + P.round(3).to_string(), flush=True)


def summarise1() -> None:
    S = _aug(pd.read_csv(O1 / "ihf_sessions.csv.gz"))
    # session-level covariates: cued matched-error share (balmatch 'trial')
    B = pd.read_csv(IN / "balmatch_sessions.csv")
    B = B[B.set == "trial"][["label", "bal_spout"]].rename(columns={"bal_spout": "share"})
    S = S.merge(B, on="label", how="left")
    G = S.groupby(["set", "win", "pair", "variant", "tau", "animal", "epoch"])
    A = G[NCM + LEAD + ["lsd", "g_r2", "share"]].mean().join(G.size().rename("n_sess")).reset_index()
    A.to_csv(O1 / "ihf_animal.csv", index=False)
    for v in ("raw", "gsr"):
        _print_animal(S, ["set", "win", "pair"], "nc", f"noise corr Fisher z, {v}",
                      f"variant == '{v}' and tau == 0 and pair in ['oro', 'VISpm', 'RSPd']")
    for v in VARIANTS:
        _print_animal(S, ["set", "win"], "asym", f"xcorr asym oro, {v}, tau 0.6", f"variant == '{v}' and tau == 0.6 and pair == 'oro'")
        _print_animal(S, ["set", "win"], "gc_dir", f"Granger dir oro, {v}, tau 0.6", f"variant == '{v}' and tau == 0.6 and pair == 'oro'")
    _print_animal(S, ["set", "win"], "lsd", "log SD ratio contra/ipsi oro", "variant == 'raw' and tau == 0 and pair == 'oro'")
    _print_animal(S, ["set", "win"], "g_r2", "share of oro variance on leave-pair-out global", "variant == 'raw' and tau == 0 and pair == 'oro'")
    rng = np.random.default_rng(0)
    T = _tests(S, ["set", "win", "pair", "variant", "tau"], NCM + LEAD + ["lsd", "g_r2"], rng)
    T.to_csv(O1 / "ihf_tests.csv", index=False)
    # session-level adjustment of lead metrics (slope within animal x epoch only)
    adj = []
    rng = np.random.default_rng(2)
    for (st, win, p, v, tau), g in S[S.pair.isin(["oro", "VISpm", "RSPd"]) & S.variant.isin(["raw", "gsr"])].groupby(
            ["set", "win", "pair", "variant", "tau"], sort=True):
        for cov in ("lsd", "share"):
            for m in ("asym", "gc_dir"):
                h = g.dropna(subset=[m, cov]).copy()
                if h.empty:
                    continue
                dm = lambda c, h=h: h[c] - h.groupby(["animal", "epoch"])[c].transform("mean")  # noqa: E731
                xx, yy = dm(cov), dm(m)
                slope = float((xx * yy).sum() / (xx * xx).sum()) if (xx * xx).sum() > 0 else 0.0
                pre_c = h[h.epoch == "pre"].groupby("animal")[cov].mean()
                h["adj"] = h[m] - slope * (h[cov] - h.animal.map(pre_c))
                tt = _tests(h.assign(metric_raw=h[m]), ["set"], [m, "adj"], rng)
                for ep in EPS[1:]:
                    a0 = tt[(tt.metric == m) & (tt.epoch == ep)]
                    a1 = tt[(tt.metric == "adj") & (tt.epoch == ep)]
                    if a0.empty or a1.empty:
                        continue
                    a0, a1 = a0.iloc[0], a1.iloc[0]
                    adj.append({"set": st, "win": win, "pair": p, "variant": v, "tau": tau, "covariate": cov,
                                "metric": m, "epoch": ep, "slope_within": slope, "delta_raw": a0.delta,
                                "lo_raw": a0.lo, "hi_raw": a0.hi, "same_raw": a0.n_same_sign, "delta_adj": a1.delta,
                                "lo_adj": a1.lo, "hi_adj": a1.hi, "same_adj": a1.n_same_sign,
                                "n_animals": a1.n_animals})
    pd.DataFrame(adj).to_csv(O1 / "ihf_sesslevel_adjust.csv", index=False)
    Rc = _recovery(S[S.pair.isin(["oro"]) & (S.tau == 0.6)], ["set", "win", "variant"], ["asym", "gc_dir"],
                   np.random.default_rng(1))
    Rc.to_csv(O1 / "ihf_recovery.csv", index=False)
    (O1 / "figdata").mkdir(parents=True, exist_ok=True)
    with open(O1 / "figdata" / "ihf_tables.pkl", "wb") as fh:
        pickle.dump({"T": T, "A": A, "adj": pd.DataFrame(adj), "Rc": Rc}, fh)


CATS = ["within_ipsi", "within_contra", "between_homo", "between_hetero", "oro_all", "nonoro_all", "all_pairs"]


def _cat_masks(names: list[str]) -> dict:
    base = [n.rsplit("_", 1)[0] for n in names]
    hemi = [n.rsplit("_", 1)[1] for n in names]
    F = len(names)
    bil = {b for b in base if base.count(b) == 2}
    iu = np.triu(np.ones((F, F), bool), 1)
    B = np.array([b in bil for b in base])
    L = np.array([h == "left" for h in hemi])
    O = np.array([b in POOL for b in base])
    same = np.array(base)[:, None] == np.array(base)[None, :]
    bb = B[:, None] & B[None, :]
    m = {"within_ipsi": iu & bb & L[:, None] & L[None, :],
         "within_contra": iu & bb & ~L[:, None] & ~L[None, :],
         "between_homo": iu & bb & (L[:, None] != L[None, :]) & same,
         "between_hetero": iu & bb & (L[:, None] != L[None, :]) & ~same,
         "oro_all": iu & O[:, None] & O[None, :],
         "nonoro_all": iu & bb & ~O[:, None] & ~O[None, :],
         "all_pairs": iu}
    return m


def summarise2() -> None:
    Z = np.load(O2 / "xa_matrices.npz", allow_pickle=False)
    M, labs, keys, names = Z["M"], [str(x) for x in Z["labels"]], [str(x) for x in Z["keys"]], [str(x) for x in Z["names"]]
    from wfield_local import config, epochs
    meta = pd.DataFrame({"label": labs, "animal": [config.animal_of(x) for x in labs],
                         "epoch": [epochs.epoch_of(x) for x in labs]})
    keep = ((meta.epoch != "pre") | (~meta.label.map(config.early_pre))).to_numpy()
    cm = _cat_masks(names)
    zz = np.arctanh(np.clip(M, -0.999, 0.999))
    rows = []
    for i, lab in enumerate(labs):
        if not keep[i]:
            continue
        for j, k in enumerate(keys):
            st, unit = k.split("|")
            for q, var in enumerate(("raw", "gs")):
                Zm = zz[i, j, q]
                if not np.isfinite(Zm).any():
                    continue
                r = {"label": lab, "animal": meta.animal[i], "epoch": meta.epoch[i], "set": st, "unit": unit,
                     "variant": var}
                for c, mk in cm.items():
                    r[c] = float(np.nanmean(Zm[mk]))
                r["ipsi_minus_contra"] = r["within_ipsi"] - r["within_contra"]
                r["within_minus_between"] = 0.5 * (r["within_ipsi"] + r["within_contra"]) - r["between_hetero"]
                r["oro_minus_nonoro"] = r["oro_all"] - r["nonoro_all"]
                rows.append(r)
    S = pd.DataFrame(rows)
    Y = pd.read_csv(O2 / "xa_sync_sessions.csv.gz")
    Y = Y[Y.label.isin(S.label)].assign(variant="raw")
    S = S.merge(Y[["label", "set", "unit", "variant", "n", "pc1_sub", "gsr2_sub", "pc1_full", "gsr2_full"]],
                on=["label", "set", "unit", "variant"], how="left")
    S.to_csv(O2 / "xa_sessions.csv.gz", index=False)
    mets = CATS + ["ipsi_minus_contra", "within_minus_between", "oro_minus_nonoro", "pc1_sub", "gsr2_sub"]
    G = S.groupby(["set", "unit", "variant", "animal", "epoch"])
    A = G[mets].mean().join(G.size().rename("n_sess")).reset_index()
    A.to_csv(O2 / "xa_animal.csv", index=False)
    for st in SETS2:
        for var in ("raw", "gs"):
            q = A[(A.set == st) & (A.unit == "slow") & (A.variant == var)]
            P = q.set_index(["animal", "epoch"])[["all_pairs", "within_ipsi", "within_contra", "between_hetero",
                                                  "between_homo", "oro_all", "nonoro_all", "pc1_sub", "n_sess"]]
            print(f"\n=== PER-ANIMAL cross-area (slow noise corr, Fisher z), {st}, {var}\n{P.round(3).to_string()}",
                  flush=True)
    T = _tests(S, ["set", "unit", "variant"], mets, np.random.default_rng(0))
    T.to_csv(O2 / "xa_tests.csv", index=False)
    # change matrices: per animal epoch-mean z minus pre-mean z; pooled = mean over animals (animal-weighted)
    CH = {}
    for j, k in enumerate(keys):
        for q, var in enumerate(("raw", "gs")):
            per = {}
            for a in AN:
                ia = (meta.animal == a).to_numpy() & keep
                pre = np.nanmean(zz[ia & (meta.epoch == "pre").to_numpy(), j, q], 0)
                for ep in EPS[1:]:
                    sel = ia & (meta.epoch == ep).to_numpy()
                    if sel.any():
                        per[(a, ep)] = np.nanmean(zz[sel, j, q], 0) - pre
            for ep in EPS[1:]:
                st_ = np.stack([per[(a, ep)] for a in AN if (a, ep) in per])
                pooled = st_.mean(0)
                cons = (np.sign(st_) == np.sign(pooled)[None]).sum(0)
                CH[(k, var, ep)] = (pooled, cons, len(st_))
            pre_all = np.nanmean(np.stack([np.nanmean(zz[((meta.animal == a).to_numpy() & keep
                                                          & (meta.epoch == "pre").to_numpy()), j, q], 0) for a in AN]), 0)
            CH[(k, var, "pre_level")] = (pre_all, None, 4)
    with open(O2 / "figdata" / "xa_change_matrices.pkl", "wb") as fh:
        pickle.dump({"CH": CH, "names": names}, fh)
    Rc = _recovery(S[S.unit == "slow"], ["set", "variant"], ["all_pairs", "within_ipsi", "within_contra",
                                                             "between_hetero", "between_homo", "oro_all",
                                                             "ipsi_minus_contra", "pc1_sub"], np.random.default_rng(1))
    Rc.to_csv(O2 / "xa_recovery.csv", index=False)
    print(f"\nRECOVERY: {len(Rc)} tests, {int((Rc.p_perm < 0.05).sum())} with p < 0.05 (expected ~{0.05 * len(Rc):.1f}); "
          f"{int(((Rc.p_perm < 0.05) & (Rc.n_same_sign >= 4)).sum())} also 4/4 same sign", flush=True)
    print(Rc[Rc.p_perm < 0.05].round(3).to_string(), flush=True)
    with open(O2 / "figdata" / "xa_tables.pkl", "wb") as fh:
        pickle.dump({"T": T, "A": A, "Rc": Rc}, fh)


# ------------------------------------------------------------------------------------------------ figures
def _wrap(s, w=150):
    return "\n".join(textwrap.wrap(s, w))


SET_LAB = {"rest": "DOCKED QUIET REST", "iti_dock": "ITI LICKS, spout docked", "iti_all": "ITI LICKS, all",
           "cued": "CUED first licks"}


def _dots(ax, T, sel, xs, xlabels, per_animal=True):
    """Per-animal open markers + pooled filled marker with 95% CI; three epochs offset."""
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    for k, ep in enumerate(EPS[1:]):
        for x, s in zip(xs, sel):
            r = T.query(s + f" and epoch == '{ep}'")
            if r.empty:
                continue
            r = r.iloc[0]
            xx = x + (k - 1) * 0.27
            if per_animal:
                for a in AN:
                    v = r.get(f"d_{a}", np.nan)
                    if np.isfinite(v):
                        ax.plot(xx - 0.07, v, ANIMAL_MARKER[a], mfc="none", mec=EPOCH_LINE[ep], ms=3.5, alpha=0.7)
            ax.errorbar(xx + 0.05, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="o", color=EPOCH_LINE[ep],
                        ms=5, lw=1.5, mec="k" if r.n_same_sign >= 3 and r.excl0 else EPOCH_LINE[ep])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs, xlabels, fontsize=8)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def _legend(fig):
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    hd = [Line2D([], [], marker="o", ls="-", color=EPOCH_LINE[ep], ms=5) for ep in EPS[1:]]
    hd += [Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", color="0.4", ms=5) for a in AN]
    fig.legend(hd, [f"{ep} - pre (pooled, 95% CI; black edge = CI excl. 0 and >= 3/4)" if i == 0 else f"{ep} - pre"
                    for i, ep in enumerate(EPS[1:])] + AN, loc="outside lower center", ncol=7, fontsize=8, frameon=False)


def plot1() -> None:
    import matplotlib.pyplot as plt
    with open(O1 / "figdata" / "ihf_tables.pkl", "rb") as fh:
        D = pickle.load(fh)
    T = D["T"]
    (O1 / "figures").mkdir(parents=True, exist_ok=True)
    # FIGURE A: noise correlation raw vs GS-removed (oro, VISpm, RSPd) per state (w1, + w2 nc_post for licks)
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharey="row", layout="constrained")
    prs = [("oro", "orofacial"), ("VISpm", "VISpm"), ("RSPd", "RSPd")]
    for j, st in enumerate(SETS1):
        for i, v in enumerate(("raw", "gsr")):
            ax = axs[i, j]
            sel = [f"set == '{st}' and win == 'w1' and pair == '{p}' and variant == '{v}' and tau == 0 and metric == 'nc'"
                   for p, _ in prs]
            _dots(ax, T, sel, list(range(len(prs))), [lb for _p, lb in prs])
            ax.set_title(f"{SET_LAB[st]}: {'RAW' if v == 'raw' else 'GLOBAL REMOVED (leave-pair-out)'}", fontsize=9)
        axs[0, j].axvspan(0.5, 2.5, color="0.94", zorder=0)
        axs[1, j].axvspan(0.5, 2.5, color="0.94", zorder=0)
    for i in range(2):
        axs[i, 0].set_ylabel("change vs pre, interhemispheric\nnoise corr (Fisher z, 1-s window)", fontsize=8.5)
    _legend(fig)
    fig.suptitle(_wrap("Interhemispheric noise correlation (homotopic contra vs ipsi, residual after spout + angle mean, "
                       "window mean -0.5..+0.5 s) BEFORE and AFTER removing the global signal (partial correlation given "
                       "the mean of all other areas). Grey = control pairs. PROVISIONAL n = 4"), fontsize=9.5)
    save_png_svg(fig, O1 / "figures" / "ihf_noisecorr_gsr.png", dpi=120)
    plt.close(fig)
    # FIGURE B: lead metrics under the four variants (oro, tau 0.6; w2 for lick sets, w1 rest)
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharey="row", layout="constrained")
    vl = [("raw", "raw"), ("bal", "balance\npartialled"), ("gsr", "global\nremoved"), ("gsr_bal", "both")]
    for j, st in enumerate(SETS1):
        win = "w1" if st == "rest" else "w2"
        for i, m in enumerate(("asym", "gc_dir")):
            ax = axs[i, j]
            sel = [f"set == '{st}' and win == '{win}' and pair == 'oro' and variant == '{v}' and tau == 0.6 and metric == '{m}'"
                   for v, _ in vl]
            _dots(ax, T, sel, list(range(len(vl))), [lb for _v, lb in vl])
            ax.set_title(f"{SET_LAB[st]} ({win}): {'xcorr asymmetry' if m == 'asym' else 'Granger direction'}",
                         fontsize=9)
    axs[0, 0].set_ylabel("change vs pre in xcorr asymmetry\n(+ = contralesional earlier)", fontsize=8.5)
    axs[1, 0].set_ylabel("change vs pre in Granger direction\n(+ = contra predicts ipsi more)", fontsize=8.5)
    _legend(fig)
    fig.suptitle(_wrap("Orofacial pool, deconvolved (tau 0.6 s): does the contralesional tilt / Granger shift survive "
                       "partialling out the per-event hemispheric balance and / or the global signal? PROVISIONAL n = 4"),
                 fontsize=9.5)
    save_png_svg(fig, O1 / "figures" / "ihf_lead_controls.png", dpi=120)
    plt.close(fig)


def plot2() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    with open(O2 / "figdata" / "xa_tables.pkl", "rb") as fh:
        D = pickle.load(fh)
    with open(O2 / "figdata" / "xa_change_matrices.pkl", "rb") as fh:
        CM = pickle.load(fh)
    T, Rc = D["T"], D["Rc"]
    (O2 / "figures").mkdir(parents=True, exist_ok=True)
    sets = ["rest", "iti_dock", "cued"]
    # FIGURE 1: category summary + synchrony (slow unit), raw and GS
    cats = [("all_pairs", "all\npairs"), ("within_ipsi", "within\nipsi"), ("within_contra", "within\ncontra"),
            ("between_hetero", "between\nhetero"), ("between_homo", "between\nhomotopic"), ("oro_all", "orofacial\nnetwork"),
            ("nonoro_all", "other\nbilateral")]
    fig, axs = plt.subplots(3, 3, figsize=(17, 11.5), layout="constrained",
                            gridspec_kw={"height_ratios": [1, 1, 0.8]})
    for j, st in enumerate(sets):
        for i, var in enumerate(("raw", "gs")):
            ax = axs[i, j]
            sel = [f"set == '{st}' and unit == 'slow' and variant == '{var}' and metric == '{c}'" for c, _ in cats]
            _dots(ax, T, sel, list(range(len(cats))), [lb for _c, lb in cats])
            ax.set_title(f"{SET_LAB[st]}: {'RAW' if var == 'raw' else 'GLOBAL-CONTROLLED (partial | other 43 areas)'}",
                         fontsize=9)
        ax = axs[2, j]
        sel = [f"set == '{st}' and unit == '{u}' and variant == 'raw' and metric == '{m}'"
               for u in ("slow", "fast") for m in ("pc1_sub", "gsr2_sub")]
        _dots(ax, T, sel, [0, 1, 2, 3], ["PC1 frac\nslow", "global R2\nslow", "PC1 frac\nfast", "global R2\nfast"])
        ax.set_title(f"{SET_LAB[st]}: SYNCHRONY index (n = {N_SUB} events per draw)", fontsize=9)
    axs[0, 0].set_ylabel("change vs pre, mean noise corr\n(Fisher z)", fontsize=8.5)
    axs[1, 0].set_ylabel("change vs pre, mean partial corr\n(Fisher z)", fontsize=8.5)
    axs[2, 0].set_ylabel("change vs pre", fontsize=8.5)
    _legend(fig)
    pre = T[(T.set == "rest") & (T.unit == "slow") & (T.variant == "raw") & (T.metric == "all_pairs")].pre_level.iloc[0]
    fig.suptitle(_wrap("CROSS-AREA correlation over recovery: event-to-event (noise) correlation of the residual 1-s window "
                       "mean between all 45 areas (categories on the 21 bilateral areas), raw and global-controlled; bottom "
                       "= low-dimensional synchrony (first-PC variance fraction, mean R2 on the global mean) for the slow "
                       f"(noise) and fast (within-window) units. Rest pre level all-pairs z = {pre:.2f}. PROVISIONAL n = 4"),
                 fontsize=9.5)
    save_png_svg(fig, O2 / "figures" / "xa_summary.png", dpi=120)
    plt.close(fig)
    # FIGURES 2/3: change matrices (slow), raw and GS
    names = CM["names"]
    F = len(names)
    nL = sum(n.endswith("_left") for n in names)
    short = [n.replace("_left", "").replace("_right", "") for n in names]
    for var in ("raw", "gs"):
        fig, axs = plt.subplots(3, 4, figsize=(19, 14.5), layout="constrained",
                                gridspec_kw={"width_ratios": [1, 1, 1, 1]})
        vmax = 0.4
        for i, st in enumerate(sets):
            k = f"{st}|slow"
            lev = CM["CH"][(k, var, "pre_level")][0]
            ax = axs[i, 0]
            im0 = ax.imshow(np.where(np.eye(F, dtype=bool), np.nan, np.tanh(lev)), cmap="RdBu_r",
                            norm=TwoSlopeNorm(0, -1, 1))
            if i == 0:
                ax.set_title("PRE level (r)", fontsize=10)
            ax.text(-0.42, 0.5, SET_LAB[st], transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=10, fontweight="bold")
            for jj, ep in enumerate(EPS[1:]):
                pooled, cons, na = CM["CH"][(k, var, ep)]
                ax = axs[i, jj + 1]
                show = np.where(np.eye(F, dtype=bool), np.nan, pooled)
                im = ax.imshow(show, cmap="PuOr_r", vmin=-vmax, vmax=vmax)
                weak = (cons < 3) & ~np.eye(F, dtype=bool)
                ov = np.zeros((F, F, 4))
                ov[weak] = (1, 1, 1, 0.75)
                ax.imshow(ov)
                if i == 0:
                    ax.set_title(f"{ep} - pre", fontsize=10)
            for ax in axs[i]:
                ax.axhline(nL - 0.5, color="k", lw=0.8)
                ax.axvline(nL - 0.5, color="k", lw=0.8)
                ax.set_yticks(range(F), short if ax is axs[i, 0] else [""] * F, fontsize=5)
                ax.set_xticks(range(F), short if i == len(sets) - 1 else [""] * F, fontsize=5, rotation=90)
        fig.colorbar(im0, ax=axs[:, 0], fraction=0.04, location="left", pad=0.08).set_label("pre r")
        fig.colorbar(im, ax=axs[:, 1:], fraction=0.02).set_label("change in Fisher z (post - pre)")
        fig.suptitle(_wrap(f"Area x area noise correlation, change vs August pre ({'RAW' if var == 'raw' else 'GLOBAL-CONTROLLED: partial correlation given the mean of the other 43 areas'}); "
                           "IPSILESIONAL (left hemisphere) block top / left, contralesional bottom / right, frontal -> "
                           "posterior within each; pooled = mean of per-animal changes; faded = fewer than 3 of 4 animals "
                           "share the pooled sign. PROVISIONAL n = 4", 170), fontsize=9.5)
        save_png_svg(fig, O2 / "figures" / f"xa_change_matrices_{var}.png", dpi=110)
        plt.close(fig)
    # FIGURE 4 (supplement): recovery tests heatmap
    q = Rc.copy()
    q["row"] = q.set + " " + q.variant
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
    for ax, tgt in zip(axs, ("aim", "fc_hit")):
        P = q[q.target == tgt].pivot_table(index="row", columns="metric", values="r_pooled_within")
        Pp = q[q.target == tgt].pivot_table(index="row", columns="metric", values="p_perm").reindex_like(P)
        Pn = q[q.target == tgt].pivot_table(index="row", columns="metric", values="n_same_sign").reindex_like(P)
        im = ax.imshow(P.to_numpy(float), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        for a in range(P.shape[0]):
            for b in range(P.shape[1]):
                v = P.iloc[a, b]
                if np.isfinite(v):
                    ax.text(b, a, f"{v:+.2f}{'*' if Pp.iloc[a, b] < 0.05 else ''}\n{int(Pn.iloc[a, b])}/4", ha="center",
                            va="center", fontsize=7)
        ax.set_xticks(range(P.shape[1]), P.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(P.shape[0]), P.index, fontsize=8)
        ax.set_title(f"within-animal partial r (| log day) with {'far-contra aim' if tgt == 'aim' else 'far-contra hit rate'}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    n_sig, n_all = int((Rc.p_perm < 0.05).sum()), len(Rc)
    fig.suptitle(_wrap(f"Does cross-area coupling track recovery within animals? {n_sig} of {n_all} tests p < 0.05 "
                       f"(uncorrected; ~{0.05 * n_all:.0f} expected by chance). * = p < 0.05; k/4 = animals with the pooled sign."),
                 fontsize=9.5)
    save_png_svg(fig, O2 / "figures" / "xa_recovery.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot1()
        plot2()
        return
    if not a.summarise:
        run(a.labels or il.labels(), a.jobs)
        if a.labels:
            S1 = pd.read_csv(O1 / "ihf_sessions_test.csv.gz")
            print(S1[S1.pair == "oro"].groupby(["set", "win", "variant", "tau"])[NCM + LEAD].mean().round(3).to_string())
            print(pd.read_csv(O2 / "xa_sync_sessions_test.csv.gz").round(3).to_string())
            return
    (O2 / "figdata").mkdir(parents=True, exist_ok=True)
    summarise1()
    summarise2()
    plot1()
    plot2()


if __name__ == "__main__":
    main()
