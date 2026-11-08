"""SENSORY-MOTOR COHERENCE vs CCA around licks (Guo 2021 / Heimbuch 2026 style; Priya 10-30). PROVISIONAL, n = 4.

Design, gate and follow-up are PRE-REGISTERED in docs/status/COHERENCE_CCA_PREREG_20261030.md (written before any fit)
-- read it; nothing here may move after seeing data. Deviations forced by the spec are listed under DEVIATIONS below
and in the doc's Outcome section.

DATA: cued first licks (lick_interhemispheric._set_arrays "cued") and, secondary, ITI docked licks
(interhemi_crossarea._sets "iti_dock"); -1..+1 s at 0.05 s (41 samples); area features il.area_features divided by the
session's frame-level area SD (il.area_sd, the decoders' z-scoring). Groups S = SSp-n, SSp-m, SSs, SSp-ul; M = tjM1, ALM,
MOp_rest, MOs_rest, each x hemisphere (left = ipsilesional). Pairings: S8M8 (primary, both hemispheres) and SiMi, ScMc,
SiMc, ScMi (4 x 4, reported, not gated). Versions: evoked (as is) and ttt (trial-to-trial: minus the session's
per-spout mean trace = lih._resid on spout dummies). Sessions: curated, epoch resolved, PS92-95, pre = AUGUST only.
A. COHERENCE of the S / M group-mean traces: per event linear detrend, Hann taper, rfft zero-padded to 64 (0.3125 Hz);
   magnitude-squared coherence across events and ITPC of each group, on N_SUB = 30 event subsamples x 50 draws (mean);
   coherence null = M taken from a different event (derangement of the subsample); bands low 0.5-1.5, delta 1.5-4 Hz.
   tau 0.6 deconvolution (lih._deconv) as sensitivity.
B. CCA: rows = events x 41 samples, S block / M block columns z-scored on the training events, ridge 1e-3 x trace / dim,
   2-fold by event order (even / odd), held-out canonical correlation averaged over both directions; null = 20
   shuffles of the M events against the S events. CC1, CC2, n pairs above the null 95th percentile, |cos| of the
   two folds' CC1 weights per block.
   STEREOTYPY: CC1 projections (cross-fitted weights) and the top PC of each block after projecting out its CC1
   direction; per event Pearson r with the leave-one-out mean of the other events at the same spout (within fold);
   session median.
GATE (per animal, August pre, cued, S8M8, tau 0; coherence / ITPC on evoked, CCA on ttt): signal >= 75 % of sessions
   above null; split-half SD (odd / even events, the MVAR definition sqrt(mean d^2 / 2)) < between-session SD; CCA also
   median |cos| >= 0.7 in each block. A measure is interpreted only if it passes in >= 3 / 4 animals.

DEVIATIONS (forced; recorded in the doc):
  * stereotypy on ttt data is degenerate (per-spout mean removed -> the leave-one-out mean is -x_i / (k - 1), r = -1),
    so it is computed on the EVOKED traces projected with the version's weights (ttt rows: ttt-fitted weights).
  * ITPC null (unspecified): the same subsamples with each event's phases randomized (uniform), mean of the draws.
  * "exceeds its null" = session value minus the mean of the null draws > 0 (95th-percentile version also tabled).
  * halves with < 30 events: both halves use N = min(30, n_half1, n_half2) events per coherence draw.

    python scripts/sm_coherence_cca.py run [--all] [--labels ...] [--jobs 4]
    python scripts/sm_coherence_cca.py gate
    python scripts/sm_coherence_cca.py summarize      # refuses unless a measure passed the gate
    python scripts/sm_coherence_cca.py plot | --plot-only
    python scripts/sm_coherence_cca.py ctrl-run [--labels ...] [--jobs 4]   # ITPC controls (addendum); frozen env
    python scripts/sm_coherence_cca.py ctrl-summarize                       # then plot -> fig5_itpc_controls.png
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interhemi_crossarea as ihc  # noqa: E402
import iti_licks as il  # noqa: E402
import lick_interhemispheric as lih  # noqa: E402

OUT = lih.OUT_W / "sm_coherence_cca"
FIGD, FIGS, TAB = OUT / "figdata", OUT / "figures", OUT / "tables"
PER = TAB / "per_session"
EPS, AN = lih.EPS, lih.AN
DT = 0.05
NT = len(lih.LAGS)
MIN_N = 30
N_SUB, N_DRAW = 30, 50
NFFT = 64
FREQ = np.fft.rfftfreq(NFFT, DT)
BANDS = {"low": (0.5, 1.5), "delta": (1.5, 4.0)}
BIX = {b: (FREQ >= lo - 1e-9) & (FREQ <= hi + 1e-9) for b, (lo, hi) in BANDS.items()}
TAUS = (0.0, 0.6)
N_SHUF = 20
RIDGE = 1e-3
SETS = ("cued", "iti_dock")
VERSIONS = ("evoked", "ttt")
SENS = ("SSp-n", "SSp-m", "SSs", "SSp-ul")
MOT = ("tjM1", "ALM", "MOp_rest", "MOs_rest")
HEM = {"i": "left", "c": "right"}
PAIRINGS = {"S8M8": ([f"{a}_{h}" for h in ("left", "right") for a in SENS],
                     [f"{a}_{h}" for h in ("left", "right") for a in MOT])}
for _s, _m in (("i", "i"), ("c", "c"), ("i", "c"), ("c", "i")):
    PAIRINGS[f"S{_s}M{_m}"] = ([f"{a}_{HEM[_s]}" for a in SENS], [f"{a}_{HEM[_m]}" for a in MOT])
PARTS = ("full", "h1", "h2")
# gate: (measure, version, value column, null column)
GATE_MEASURES = [("coh_low", "evoked"), ("coh_delta", "evoked"), ("itpcS_low", "evoked"), ("itpcS_delta", "evoked"),
                 ("itpcM_low", "evoked"), ("itpcM_delta", "evoked"), ("cc1", "ttt")]
SIG_FRAC, COS_MIN, PASS_ANIMALS = 0.75, 0.7, 3
EXAMPLE_ANIMAL = "PS92"          # fixed before data: the animal with the most August pre sessions (8)


# ------------------------------------------------------------------------------------------------ sessions
def labels(pre_only: bool) -> list[str]:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    cur = set(ak.curated_labels())
    out = []
    for x in il.labels():
        ep = epochs.epoch_of(x)
        if x not in cur or ep is None or x[:4] not in AN:
            continue
        if ep == "pre" and x.split("_")[1][:2] != "08":
            continue
        if pre_only and ep != "pre":
            continue
        out.append(x)
    return sorted(out)


class Item(tuple):
    def __repr__(self):
        return str(self[0])


# ------------------------------------------------------------------------------------------------ coherence
def _spec(x: np.ndarray) -> np.ndarray:
    """(n, T) -> (n, F) rfft of the detrended, Hann-tapered traces, zero-padded to NFFT."""
    from scipy.signal import detrend
    return np.fft.rfft(detrend(x, axis=1) * np.hanning(x.shape[1]), n=NFFT, axis=1)


def _coh(FS, FM) -> np.ndarray:
    sxy = (FS * FM.conj()).mean(0)
    return np.abs(sxy) ** 2 / ((np.abs(FS) ** 2).mean(0) * (np.abs(FM) ** 2).mean(0))


def _itpc(F) -> np.ndarray:
    return np.abs((F / np.maximum(np.abs(F), 1e-30)).mean(0))


def coherence(s: np.ndarray, m: np.ndarray, ev: np.ndarray, nsub: int, rng) -> dict:
    """Mean over N_DRAW subsamples of nsub events of: coherence, shuffled-pair null, ITPC S / M and their random-phase
    nulls (spectra, (F,))."""
    FS, FM = _spec(s), _spec(m)
    acc = {k: np.zeros(len(FREQ)) for k in ("coh", "coh_null", "itpcS", "itpcS_null", "itpcM", "itpcM_null")}
    for _ in range(N_DRAW):
        ix = rng.choice(ev, nsub, replace=False)
        jx = np.roll(ix, int(rng.integers(1, nsub)))                 # derangement: every M from another event
        a, b = FS[ix], FM[ix]
        acc["coh"] += _coh(a, b)
        acc["coh_null"] += _coh(a, FM[jx])
        acc["itpcS"] += _itpc(a)
        acc["itpcM"] += _itpc(b)
        ph = np.exp(2j * np.pi * rng.random(a.shape))
        acc["itpcS_null"] += _itpc(a * ph)
        acc["itpcM_null"] += _itpc(b * np.exp(2j * np.pi * rng.random(b.shape)))
    return {k: v / N_DRAW for k, v in acc.items()}


def _bands(sp: dict) -> dict:
    out = {}
    for b, ix in BIX.items():
        for k, nm in (("coh", "coh"), ("itpcS", "itpcS"), ("itpcM", "itpcM")):
            out[f"{nm}_{b}"] = float(sp[k][ix].mean())
            out[f"{nm}_{b}_null"] = float(sp[f"{k}_null"][ix].mean())
    return out


# ------------------------------------------------------------------------------------------------ CCA
def _isqrt(C):
    w, V = np.linalg.eigh(C)
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def _ridge(C):
    return C + RIDGE * np.trace(C) / len(C) * np.eye(len(C))


def cca_fit(X, Y):
    """Training rows X (r, p), Y (r, q) -> (mu_x, sd_x, A, mu_y, sd_y, B, canonical corrs)."""
    mx, sx, my, sy = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
    Xz, Yz = (X - mx) / sx, (Y - my) / sy
    r = len(X)
    Cxx, Cyy, Cxy = _ridge(Xz.T @ Xz / r), _ridge(Yz.T @ Yz / r), Xz.T @ Yz / r
    Wx, Wy = _isqrt(Cxx), _isqrt(Cyy)
    U, s, Vt = np.linalg.svd(Wx @ Cxy @ Wy)
    k = min(X.shape[1], Y.shape[1])
    return {"mx": mx, "sx": sx, "A": (Wx @ U)[:, :k], "my": my, "sy": sy, "B": (Wy @ Vt.T)[:, :k], "s": s[:k], "Xz": Xz,
            "Yz": Yz}


def _colcorr(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    return (P * Q).sum(0) / np.sqrt((P * P).sum(0) * (Q * Q).sum(0))


def _cos(a, b):
    return float(abs(a @ b) / np.sqrt((a @ a) * (b @ b)))


def cv_cca(S, M, ev, Mev=None):
    """2-fold (even / odd positions of ev) held-out canonical correlations, mean of both directions. S, M (n, T, p);
    Mev = the M event paired with each ev (shuffle), default ev. Returns (corrs (k,), cos_S, cos_M, fits)."""
    Mev = ev if Mev is None else Mev
    folds = [(ev[0::2], Mev[0::2]), (ev[1::2], Mev[1::2])]
    p, q = S.shape[-1], M.shape[-1]
    fits, cc = [], []
    for f in (0, 1):
        (trs, trm), (tes, tem) = folds[f], folds[1 - f]
        F = cca_fit(S[trs].reshape(-1, p), M[trm].reshape(-1, q))
        Pte = ((S[tes].reshape(-1, p) - F["mx"]) / F["sx"]) @ F["A"]
        Qte = ((M[tem].reshape(-1, q) - F["my"]) / F["sy"]) @ F["B"]
        cc.append(_colcorr(Pte, Qte))
        fits.append(F)
    return np.mean(cc, 0), _cos(fits[0]["A"][:, 0], fits[1]["A"][:, 0]), _cos(fits[0]["B"][:, 0], fits[1]["B"][:, 0]), fits


def cca_block(S, M, ev, rng) -> dict:
    real, cs, cm, fits = cv_cca(S, M, ev)
    nul = []
    for _ in range(N_SHUF):
        p = rng.permutation(ev)
        mev = np.empty_like(ev)
        pos = {e: i for i, e in enumerate(ev)}
        for a, b in zip(p, np.roll(p, 1)):                           # derangement of the M events
            mev[pos[a]] = b
        nul.append(cv_cca(S, M, ev, mev)[0])
    nul = np.array(nul)
    p95 = np.percentile(nul, 95, axis=0)
    return {"cc1": float(real[0]), "cc1_null": float(nul[:, 0].mean()), "cc1_null95": float(p95[0]),
            "cc2": float(real[1]), "cc2_null": float(nul[:, 1].mean()), "n_sig_pairs": int((real > p95).sum()),
            "cos_S": cs, "cos_M": cm}, fits


def _loo_r(P: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Per event r of its trace (row of P (k, T)) with the leave-one-out mean of the other events at its spout."""
    out = np.full(len(P), np.nan)
    for s in np.unique(pos):
        ix = np.where(pos == s)[0]
        if len(ix) < 3:
            continue
        tot = P[ix].sum(0)
        for i in ix:
            loo = (tot - P[i]) / (len(ix) - 1)
            a, b = P[i] - P[i].mean(), loo - loo.mean()
            d = np.sqrt((a * a).sum() * (b * b).sum())
            out[i] = (a * b).sum() / d if d > 0 else np.nan
    return out


def stereotypy(fits, ev, Sev, Mev, pos, keep_proj=False) -> dict:
    """Cross-fitted: events of fold f projected with the weights fitted on the other fold; traces = EVOKED (Sev, Mev)."""
    folds = [ev[0::2], ev[1::2]]
    p, q = Sev.shape[-1], Mev.shape[-1]
    r = {k: [] for k in ("cc_S", "cc_M", "pc_S", "pc_M")}
    proj = {"u": np.full((len(Sev), NT), np.nan), "v": np.full((len(Sev), NT), np.nan)}
    for f in (0, 1):
        F, te = fits[f], folds[1 - f]
        for blk, X, mu, sd, W, Ztr in (("S", Sev, F["mx"], F["sx"], F["A"], F["Xz"]),
                                       ("M", Mev, F["my"], F["sy"], F["B"], F["Yz"])):
            w = W[:, 0]
            Pm = np.eye(len(w)) - np.outer(w, w) / (w @ w)
            ev_, V = np.linalg.eigh(np.cov(Ztr @ Pm, rowvar=False))
            pc = Pm @ V[:, -1]
            Z = (X[te].reshape(-1, p if blk == "S" else q) - mu) / sd
            cc = (Z @ w).reshape(len(te), NT)
            r[f"cc_{blk}"].append(_loo_r(cc, pos[te]))
            r[f"pc_{blk}"].append(_loo_r((Z @ pc).reshape(len(te), NT), pos[te]))
            if keep_proj:
                proj["u" if blk == "S" else "v"][te] = cc * np.sign(fits[0]["A"][:, 0] @ fits[f]["A"][:, 0]
                                                                     if blk == "S" else
                                                                     fits[0]["B"][:, 0] @ fits[f]["B"][:, 0])
    out = {f"stereo_{k}": float(np.nanmedian(np.concatenate(v))) for k, v in r.items()}
    out["stereo_cc"] = 0.5 * (out["stereo_cc_S"] + out["stereo_cc_M"])
    out["stereo_pc"] = 0.5 * (out["stereo_pc_S"] + out["stereo_pc_M"])
    out["stereo_diff"] = out["stereo_cc"] - out["stereo_pc"]
    return (out, proj) if keep_proj else out


# ------------------------------------------------------------------------------------------------ per session
def _arrays(label, meta, dock_t):
    """{set: (Z (n, T, 45) z-scored area features, pos, names)} for the two sets."""
    S = ihc._sets(label, meta, dock_t)
    sd_tab = il.area_sd()
    out = {}
    for st in SETS:
        if st not in S:
            continue
        X, pos, _ang, cols = S[st]
        names = il.area_names(cols)
        sd = sd_tab.loc[label, names].to_numpy(float)
        assert np.all(np.isfinite(sd) & (sd > 0)), label
        assert X.shape[1] == NT, (label, st, X.shape)
        out[st] = (il.area_features(X.astype(np.float64), cols, label) / sd, np.asarray(pos).astype(str), names)
    return out


def session_work(item, extras=False):
    label, meta, dock_t, fams = item
    from wfield_local import config, epochs
    seed = int(label[2:4]) * 10000 + int(label[-4:])
    an, ep = config.animal_of(label), epochs.epoch_of(label)
    rows, ex = [], {}
    for k_set, (st, (Z, pos, names)) in enumerate(_arrays(label, meta, dock_t).items()):
        n = len(Z)
        if n < MIN_N:
            continue
        ix = {nm: i for i, nm in enumerate(names)}
        Zt = lih._resid(Z, lih._design(pos, None, n))                # per-spout mean trace removed
        allev = np.arange(n)
        parts = {"full": allev, "h1": allev[0::2], "h2": allev[1::2]}
        nh = min(N_SUB, len(parts["h1"]), len(parts["h2"]))
        for k_pair, (pr, (sa, ma)) in enumerate(PAIRINGS.items()):
            si, mi = [ix[a] for a in sa], [ix[a] for a in ma]
            for k_v, (ver, ZZ) in enumerate((("evoked", Z), ("ttt", Zt))):
                base = {"label": label, "animal": an, "epoch": ep, "set": st, "pairing": pr, "version": ver, "n": n}
                Sb, Mb = ZZ[..., si], ZZ[..., mi]
                if "coh" in fams:
                    s0, m0 = Sb.mean(-1), Mb.mean(-1)
                    for tau in TAUS:
                        s, m = (lih._deconv(s0, tau), lih._deconv(m0, tau)) if tau else (s0, m0)
                        for k_p, (part, ev) in enumerate(parts.items()):
                            rng = np.random.default_rng([seed, k_set, k_pair, k_v, int(tau * 10), k_p])
                            sp = coherence(s, m, ev, N_SUB if part == "full" else nh, rng)
                            rows.append({**base, "kind": "coh", "tau": tau, "part": part, **_bands(sp)})
                            if extras and part == "full" and pr == "S8M8":
                                ex[(st, ver, tau, "spec")] = sp
                if "cca" in fams:
                    for k_p, (part, ev) in enumerate(parts.items()):
                        rng = np.random.default_rng([seed, k_set, k_pair, k_v, 99, k_p])
                        r, fits = cca_block(Sb, Mb, ev, rng)
                        if part == "full":
                            st_out = stereotypy(fits, ev, Z[..., si], Z[..., mi], pos, keep_proj=extras)
                            if extras:
                                st_out, proj = st_out
                                if pr == "S8M8":
                                    ex[(st, ver, "proj")] = proj
                            r.update(st_out)
                        rows.append({**base, "kind": "cca", "tau": 0.0, "part": part, **r})
        if extras:
            ex[(st, "pos")] = pos
            S8, M8 = PAIRINGS["S8M8"]
            ex[(st, "traces")] = (Z[..., [ix[a] for a in S8]].mean(-1), Z[..., [ix[a] for a in M8]].mean(-1))
    df = pd.DataFrame(rows)
    return (df, ex) if extras else df


def _meta_items(labs, fams):
    M = lih._meta()
    Aud = pd.read_csv(ihc.IN / "iti_dock_audit.csv", usecols=["label", "t", "dock"])
    Aud = Aud[Aud.dock]
    dk = {x: np.round(g.t.to_numpy(float), 4) for x, g in Aud.groupby("label")}
    empty = pd.DataFrame(columns=["tk", "reach_mm", "gap", "since_stop", "to_strobe"])
    return [Item((x, M.get(x, empty), dk.get(x, np.zeros(0)), fams)) for x in labs]


def run(labs, jobs, fams) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    PER.mkdir(parents=True, exist_ok=True)
    todo = [x for x in labs if not (PER / f"{x}.pkl").exists()]
    print(f"run: {len(labs)} sessions, {len(todo)} to fit, families {sorted(fams)}", flush=True)
    t0 = time.time()
    res, fail = ak.fan_sessions(_meta_items(todo, fams), session_work, jobs=min(jobs, 4), key=lambda it: it[0])
    for it, df in res:
        df.to_pickle(PER / f"{it[0]}.pkl")
    if fail:
        print("FAILED:", [str(f[0]) for f in fail], flush=True)
    print(f"sessions done {time.time() - t0:.0f} s", flush=True)
    collect()


def collect() -> pd.DataFrame:
    S = pd.concat([pd.read_pickle(f) for f in sorted(PER.glob("*.pkl"))], ignore_index=True)
    S.to_csv(TAB / "sessions.csv.gz", index=False)
    return S


# ------------------------------------------------------------------------------------------------ gate
def _val(S: pd.DataFrame, meas: str) -> pd.Series:
    if meas == "cc1":
        return S.cc1 - S.cc1_null
    return S[meas] - S[f"{meas}_null"]


def gate_table(S: pd.DataFrame) -> pd.DataFrame:
    """Per (set, pairing, version, tau, measure, animal): signal fraction, split-half SD, between-session SD, |cos|."""
    P = S[S.epoch == "pre"]
    rows = []
    for (st, pr, ver, tau, kind), g in P.groupby(["set", "pairing", "version", "tau", "kind"]):
        meas = ["cc1"] if kind == "cca" else [m for m, _v in GATE_MEASURES if m != "cc1"]
        W = {part: q.set_index("label") for part, q in g.groupby("part")}
        for m in meas:
            full = _val(W["full"], m)
            d = (_val(W["h1"], m) - _val(W["h2"], m)).reindex(full.index)
            for a in AN:
                ia = full.index.str[:4] == a
                if not ia.any():
                    continue
                f, dd = full[ia], d[ia]
                r = {"set": st, "pairing": pr, "version": ver, "tau": tau, "measure": m, "animal": a,
                     "n_sess": int(ia.sum()), "pre_mean": float(f.mean()), "sig_frac": float((f > 0).mean()),
                     "half_sd": float(np.sqrt(np.mean(dd ** 2) / 2)), "between_sd": float(f.std(ddof=1))}
                if m == "cc1":
                    q = W["full"][ia]
                    r["sig95_frac"] = float((q.cc1 > q.cc1_null95).mean())
                    r["cos_S_med"], r["cos_M_med"] = float(q.cos_S.median()), float(q.cos_M.median())
                    r["n_sig_pairs_med"] = float(q.n_sig_pairs.median())
                r["rel_ratio"] = r["half_sd"] / r["between_sd"] if r["between_sd"] > 0 else np.inf
                r["pass_signal"] = r["sig_frac"] >= SIG_FRAC
                r["pass_rel"] = r["half_sd"] < r["between_sd"]
                r["pass_stab"] = (r["cos_S_med"] >= COS_MIN and r["cos_M_med"] >= COS_MIN) if m == "cc1" else True
                r["PASS"] = r["pass_signal"] and r["pass_rel"] and r["pass_stab"]
                rows.append(r)
    return pd.DataFrame(rows)


def _primary(G: pd.DataFrame) -> pd.DataFrame:
    keep = [(m, v) for m, v in GATE_MEASURES]
    q = G[(G.set == "cued") & (G.pairing == "S8M8") & (G.tau == 0)]
    return q[[(m, v) in keep for m, v in zip(q.measure, q.version)]]


def gate() -> pd.DataFrame:
    S = collect()
    G = gate_table(S)
    G.to_csv(TAB / "gate_animals_all.csv", index=False)
    Pg = _primary(G)
    Pg.to_csv(TAB / "gate_animals.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["measure", "version", "animal", "n_sess", "pre_mean", "sig_frac", "half_sd", "between_sd", "rel_ratio",
            "pass_signal", "pass_rel", "PASS"]
    print("\n=== PRE-REGISTERED GATE (cued, S8M8, tau 0; coherence / ITPC evoked, CCA ttt) -- PER ANIMAL ===")
    print(Pg[cols].round(4).to_string(index=False))
    c = Pg[Pg.measure == "cc1"]
    print("\nCCA extras:\n" + c[["animal", "sig95_frac", "cos_S_med", "cos_M_med", "n_sig_pairs_med"]].round(3)
          .to_string(index=False))
    V = Pg.groupby("measure").PASS.sum().rename("n_animals_pass").to_frame()
    V["MEASURE_PASSES"] = V.n_animals_pass >= PASS_ANIMALS
    V.to_csv(TAB / "gate_measures.csv")
    print("\n=== MEASURE VERDICT (>= 3 / 4 animals) ===\n" + V.to_string())
    print(f"\nDLAG criterion (CCA passes in >= 3/4 animals): {bool(V.loc['cc1', 'MEASURE_PASSES'])}")
    # non-gated context: same criteria on the secondary set / other version / deconvolved
    q = G[(G.pairing == "S8M8")].groupby(["set", "version", "tau", "measure"]).agg(
        n_pass=("PASS", "sum"), sig=("sig_frac", "mean"), rel=("rel_ratio", "median")).reset_index()
    print("\n=== context (not gated): S8M8, animals passing per set / version / tau ===\n" + q.round(3).to_string())
    return Pg


def passed_measures() -> list[str]:
    V = pd.read_csv(TAB / "gate_measures.csv", index_col=0)
    return [m for m, ok in V.MEASURE_PASSES.items() if ok]


def families(meas: list[str]) -> set:
    f = set()
    if any(m != "cc1" for m in meas):
        f.add("coh")
    if "cc1" in meas:
        f.add("cca")
    return f


# ------------------------------------------------------------------------------------------------ summarise
def summarise() -> None:
    meas = passed_measures()
    if not meas:
        sys.exit("no measure passed the pre-registered gate: summarize refused")
    S = collect()
    S = S[S.part == "full"].copy()
    keys = ["set", "pairing", "version", "tau"]
    S["cc1_d"] = S.cc1 - S.cc1_null if "cc1" in S else np.nan
    mets = []
    for m in meas:
        if m == "cc1":
            mets += ["cc1_d", "stereo_cc", "stereo_pc", "stereo_diff"]
        else:
            S[f"{m}_d"] = S[m] - S[f"{m}_null"]
            mets.append(f"{m}_d")
    ver_of = dict(GATE_MEASURES)
    SS = []
    for m in mets:
        base = "cc1" if m.startswith(("cc1", "stereo")) else m[:-2]
        q = S[(S.version == ver_of[base]) & (S.tau == 0) & S[m].notna()][["label", "animal", "epoch"] + keys + [m]]
        SS.append(q.rename(columns={m: "value"}).assign(metric=m))
    SS = pd.concat(SS, ignore_index=True)
    SS.to_csv(TAB / "summary_sessions.csv.gz", index=False)
    for (st, pr), g in SS[SS.pairing == "S8M8"].groupby(["set", "pairing"]):
        P = g.groupby(["metric", "animal", "epoch"]).value.mean().unstack("epoch").reindex(columns=EPS)
        print(f"\n=== PER-ANIMAL ({st}, {pr}) epoch means ===\n{P.round(4).to_string()}", flush=True)
    W = SS.pivot_table(index=["label", "animal", "epoch"] + keys, columns="metric", values="value").reset_index()
    W.columns.name = None
    T = ihc._tests(W, ["set", "pairing"], mets, np.random.default_rng(0))
    T.to_csv(TAB / "tests_boot_delta.csv", index=False)
    print("\n=== boot_delta vs August pre (change = CI excl. 0 AND >= 3/4 same sign) ===")
    print(T[T.pairing == "S8M8"][["set", "metric", "epoch", "delta", "lo", "hi", "n_same_sign", "excl0"] +
                                 [f"d_{a}" for a in AN]].round(4).to_string(index=False))
    Rc = ihc._recovery(W[W.pairing == "S8M8"], ["set"], mets, np.random.default_rng(1))
    Rc.to_csv(TAB / "recovery_partial_r.csv", index=False)
    print(f"\nRECOVERY: {len(Rc)} tests, {int((Rc.p_perm < 0.05).sum())} with p < 0.05 "
          f"(expected ~{0.05 * len(Rc):.1f})\n{Rc.round(3).to_string(index=False)}", flush=True)
    FIGD.mkdir(parents=True, exist_ok=True)
    with open(FIGD / "summary_tables.pkl", "wb") as fh:
        pickle.dump({"T": T, "Rc": Rc, "SS": SS, "metrics": mets}, fh)


# ------------------------------------------------------------------------------------------------ figures
def _wrap(s, w=150):
    return "\n".join(textwrap.wrap(s, w))


MEAS_LAB = {"coh_low": "coherence\nlow", "coh_delta": "coherence\ndelta", "itpcS_low": "ITPC S\nlow",
            "itpcS_delta": "ITPC S\ndelta", "itpcM_low": "ITPC M\nlow", "itpcM_delta": "ITPC M\ndelta",
            "cc1": "CCA CC1\n(held-out)"}


def plot_gate() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    G = pd.read_csv(TAB / "gate_animals.csv")
    V = pd.read_csv(TAB / "gate_measures.csv", index_col=0)
    FIGD.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    with open(FIGD / "fig1_gate.pkl", "wb") as fh:
        pickle.dump({"G": G, "V": V}, fh)
    meas = [m for m, _v in GATE_MEASURES]
    xs = np.arange(len(meas))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained", gridspec_kw={"width_ratios": [3, 3, 1.2]})
    off = {a: (k - 1.5) * 0.13 for k, a in enumerate(AN)}
    for x, m in zip(xs, meas):
        g = G[G.measure == m]
        for r in g.itertuples():
            col = "k" if r.PASS else "tab:red"
            axs[0].plot(x + off[r.animal], r.sig_frac, ANIMAL_MARKER[r.animal], mfc="none" if not r.pass_signal else col,
                        mec=col, ms=7)
            axs[1].plot(x + off[r.animal], r.rel_ratio, ANIMAL_MARKER[r.animal], mfc="none" if not r.pass_rel else col,
                        mec=col, ms=7)
            if m == "cc1":
                for j, c in enumerate(("cos_S_med", "cos_M_med")):
                    v = getattr(r, c)
                    axs[2].plot(j + off[r.animal], v, ANIMAL_MARKER[r.animal], mfc=col if v >= COS_MIN else "none",
                                mec=col, ms=7)
    axs[0].axhline(SIG_FRAC, color="0.4", ls="--", lw=1)
    axs[1].axhline(1.0, color="0.4", ls="--", lw=1)
    axs[2].axhline(COS_MIN, color="0.4", ls="--", lw=1)
    axs[1].set_yscale("log")
    for ax in axs[:2]:
        ax.set_xticks(xs, [MEAS_LAB[m] + f"\n{int(V.loc[m, 'n_animals_pass'])}/4 pass" for m in meas], fontsize=8)
        ax.set_xlim(-0.6, len(meas) - 0.4)
    axs[0].set_ylim(-0.05, 1.05)
    axs[2].set_xticks([0, 1], ["S block", "M block"])
    axs[2].set_xlim(-0.6, 1.6)
    axs[2].set_ylim(0, 1.02)
    axs[0].set_ylabel("fraction of pre sessions\nabove the null (>= 0.75)")
    axs[1].set_ylabel("split-half SD / between-session SD\n(< 1 to pass)")
    axs[2].set_ylabel("CCA CC1 weight |cos| between halves,\nmedian over pre sessions (>= 0.7)")
    axs[0].set_title("1  signal", fontsize=10, loc="left")
    axs[1].set_title("2  reliability of the session readout", fontsize=10, loc="left")
    axs[2].set_title("3  CCA weight stability", fontsize=10, loc="left")
    for ax in axs:
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    hd = [Line2D([], [], marker=ANIMAL_MARKER[a], ls="", color="0.3", mfc="none", ms=7) for a in AN]
    hd += [Line2D([], [], marker="o", ls="", color="k", ms=7), Line2D([], [], marker="o", ls="", color="tab:red", ms=7),
           Line2D([], [], marker="o", ls="", color="k", mfc="none", ms=7)]
    fig.legend(hd, AN + ["animal passes the measure", "animal fails", "open = this criterion failed"],
               loc="outside lower center", ncol=7, fontsize=8.5, frameon=False)
    fig.suptitle(_wrap("Pre-registered gate, sensory (SSp-n/m, SSs, SSp-ul) vs motor (tjM1, ALM, MOp/MOs rest) "
                       "around CUED first licks, August pre sessions: coherence / ITPC on evoked traces, CCA on "
                       "trial-to-trial (per-spout mean removed). A measure is interpreted only if >= 3 / 4 animals "
                       "pass all its criteria. n = 4 animals."), fontsize=9.5)
    fig.savefig(FIGS / "fig1_gate.png", dpi=140)
    plt.close(fig)


def plot_change() -> None:
    import matplotlib.pyplot as plt
    f = FIGD / "summary_tables.pkl"
    if not f.exists():
        return
    with open(f, "rb") as fh:
        D = pickle.load(fh)
    T, mets = D["T"], D["metrics"]
    T = T[(T.set == "cued") & (T.pairing == "S8M8")]
    nc = min(4, len(mets))
    nr = int(np.ceil(len(mets) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3.6 * nr), layout="constrained", squeeze=False)
    eps = EPS[1:]
    for ax, m in zip(axs.ravel(), mets):
        q = T[T.metric == m].set_index("epoch").reindex(eps)
        for k, ep in enumerate(eps):
            if not np.isfinite(q.loc[ep, "delta"]):
                continue
            r = q.loc[ep]
            for a in AN:
                v = r.get(f"d_{a}", np.nan)
                if np.isfinite(v):
                    ax.plot(k - 0.12, v, "o", color="0.6", ms=4)
            ok = r.excl0 and r.n_same_sign >= 3
            ax.errorbar(k + 0.08, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="s", color="k",
                        mfc="k" if ok else "w", ms=6, lw=1.4)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(eps)), eps)
        ax.set_xlim(-0.5, len(eps) - 0.5)
        ax.set_ylabel(f"{m}: change vs pre", fontsize=8.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axs.ravel()[len(mets):]:
        ax.axis("off")
    with open(FIGD / "fig2_change_vs_pre.pkl", "wb") as fh:
        pickle.dump({"T": T, "metrics": mets}, fh)
    fig.suptitle(_wrap("Change vs August pre (CUED, S8M8) for the measures that passed the gate: squares = paired "
                       "nested bootstrap (boot_delta) with 95% CI, filled = CI excludes 0 AND >= 3/4 animals share the "
                       "sign; gray dots = animals. n = 4."), fontsize=9.5)
    fig.savefig(FIGS / "fig2_change_vs_pre.png", dpi=140)
    plt.close(fig)


def example_data() -> dict:
    from wfield_local import epochs
    labs = labels(pre_only=False)
    pre = [x for x in labs if x[:4] == EXAMPLE_ANIMAL and epochs.epoch_of(x) == "pre"][-1]
    chr_ = [x for x in labs if x[:4] == EXAMPLE_ANIMAL and epochs.epoch_of(x) == "chronic"][-1]
    out = {}
    for x in (pre, chr_):
        it = _meta_items([x], {"coh", "cca"})[0]
        df, ex = session_work(it, extras=True)
        out[x] = {"rows": df[(df.set == "cued") & (df.pairing == "S8M8") & (df.part == "full")], "ex": ex,
                  "epoch": epochs.epoch_of(x)}
    return out


def plot_example() -> None:
    import matplotlib.pyplot as plt
    f = FIGD / "fig3_example.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            D = pickle.load(fh)
    else:
        D = example_data()
        with open(f, "wb") as fh:
            pickle.dump(D, fh)
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    lags = lih.LAGS
    vmax = 0
    for d in D.values():
        vmax = max(vmax, np.nanpercentile(np.abs(d["ex"][("cued", "ttt", "proj")]["u"]), 98))
    for i, (x, d) in enumerate(D.items()):
        ex = d["ex"]
        s, m = ex[("cued", "traces")]
        ax = axs[i, 0]
        for tr, c, nm in ((s, "tab:blue", "S group"), (m, "tab:orange", "M group")):
            mu, se = tr.mean(0), tr.std(0) / np.sqrt(len(tr))
            ax.plot(lags, mu, color=c, label=nm)
            ax.fill_between(lags, mu - se, mu + se, color=c, alpha=0.25, lw=0)
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set_ylabel(f"{x} ({d['epoch']})\nz (session SD units)")
        ax.set_xlabel("time from lick peak (s)")
        ax = axs[i, 1]
        sp = ex[("cued", "evoked", 0.0, "spec")]
        ax.plot(FREQ, sp["coh"], "k", label="coherence (evoked)")
        ax.plot(FREQ, sp["coh_null"], color="0.6", ls="--", label="shuffled-pair null")
        spt = ex[("cued", "ttt", 0.0, "spec")]
        ax.plot(FREQ, spt["coh"], color="tab:green", label="coherence (trial-to-trial)")
        for b, (lo, hi) in BANDS.items():
            ax.axvspan(lo, hi, color="0.92" if b == "low" else "0.85", zorder=0)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("S-M coherence (30-event draws)")
        pos = ex[("cued", "pos")]
        o = np.argsort(pos, kind="stable")
        P = ex[("cued", "ttt", "proj")]
        for j, (k, nm) in enumerate((("u", "S side"), ("v", "M side"))):
            ax = axs[i, 2 + j]
            im = ax.imshow(P[k][o], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           extent=[lags[0], lags[-1], len(o), 0], interpolation="nearest")
            bnd = np.where(pos[o][1:] != pos[o][:-1])[0] + 1
            for b in bnd:
                ax.axhline(b, color="k", lw=0.5)
            ax.set_yticks([(a + b) / 2 for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(o)])], np.unique(pos[o]),
                          fontsize=7)
            ax.set_xlabel("time from lick peak (s)")
            r = d["rows"]
            rr = r[(r.kind == "cca") & (r.version == "ttt")].iloc[0]
            blk = "S" if k == "u" else "M"
            ax.set_title(f"CC1 projection, {nm} (stereotypy {rr[f'stereo_cc_{blk}']:.2f})", fontsize=9, loc="left")
    axs[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2, fontsize=8, frameon=False)
    axs[0, 1].legend(loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2, fontsize=8, frameon=False)
    fig.colorbar(im, ax=axs[:, 2:], fraction=0.03).set_label("CC1 projection (cross-fitted, evoked traces)")
    txt = []
    for x, d in D.items():
        r = d["rows"]
        c = r[(r.kind == "cca") & (r.version == "ttt")].iloc[0]
        h = r[(r.kind == "coh") & (r.version == "evoked") & (r.tau == 0)].iloc[0]
        txt.append(f"{x}: coh low {h.coh_low:.2f} (null {h.coh_low_null:.2f}), delta {h.coh_delta:.2f} "
                   f"(null {h.coh_delta_null:.2f}); CC1 held-out {c.cc1:.2f} (null {c.cc1_null:.2f}); stereotypy "
                   f"CC1 {c.stereo_cc:.2f} vs within-block PC {c.stereo_pc:.2f}")
    fig.suptitle(_wrap(f"EXAMPLE (illustration only, not a test): {EXAMPLE_ANIMAL} last August pre vs last chronic "
                       "session, CUED first licks. Left: S / M group mean traces (+- SEM); middle: S-M coherence "
                       "spectrum (shaded = low 0.5-1.5 and delta 1.5-4 Hz bands); right: single-event CC1 projections "
                       "(weights fitted on trial-to-trial data, other fold), events grouped by spout.") + "\n"
                 + "\n".join(txt), fontsize=9)
    fig.savefig(FIGS / "fig3_example_pre_vs_chronic.png", dpi=130)
    plt.close(fig)


def itpc_data() -> pd.DataFrame:
    """Per-session ITPC levels (raw and null), cued, S8M8, evoked, tau 0, full session."""
    rows = []
    for f in sorted((OUT / "tables" / "per_session").glob("*.pkl")):
        d = pd.read_pickle(f)
        d = d[(d.set == "cued") & (d.pairing == "S8M8") & (d.version == "evoked") & (d.tau == 0) & (d.part == "full")
              & (d.kind == "coh")]
        cols = [c for c in d.columns if c.startswith("itpc")]
        if len(d):
            rows.append(d[["label", "animal", "epoch"] + cols].iloc[[0]])
    return pd.concat(rows, ignore_index=True)


def plot_itpc() -> None:
    """Fig 4: lick-locked phase consistency (ITPC) LEVELS per animal by epoch, with the random-phase null."""
    import matplotlib.pyplot as plt
    f = FIGD / "fig4_itpc_levels.pkl"
    if f.exists():
        D = pd.read_pickle(f)
    else:
        D = itpc_data()
        D.to_pickle(f)
    acol = {"PS92": "tab:blue", "PS93": "tab:orange", "PS94": "tab:green", "PS95": "tab:purple"}
    panels = [("itpcS_low", "sensory, 0.5-1.5 Hz"), ("itpcS_delta", "sensory, 1.5-4 Hz"),
              ("itpcM_low", "motor, 0.5-1.5 Hz"), ("itpcM_delta", "motor, 1.5-4 Hz")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), layout="constrained", sharey=True)
    x = np.arange(len(EPS))
    for ax, (m, title) in zip(axs, panels):
        for a in AN:
            g = D[D.animal == a].groupby("epoch")
            mu = g[m].mean().reindex(EPS)
            ax.plot(x, mu.values, "-o", color=acol[a], ms=5, lw=1.4, label=a)
            for k, ep in enumerate(EPS):
                v = D[(D.animal == a) & (D.epoch == ep)][m]
                ax.plot(np.full(len(v), k) + (AN.index(a) - 1.5) * 0.06, v, ".", color=acol[a], ms=3, alpha=0.35)
        nl = D.groupby("epoch")[f"{m}_null"].mean().reindex(EPS)
        ax.plot(x, nl.values, "--", color="0.5", lw=1.2, label="random-phase null")
        ax.set_xticks(x, EPS)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel("ITPC (phase consistency\nacross cued first licks)")
    axs[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Lick-locked phase consistency, 8 sensory / 8 motor areas, cued first licks, -1..+1 s: lines = animal "
                 "epoch means, dots = sessions (30-event subsamples, 50 draws)", fontsize=9.5)
    fig.savefig(FIGS / "fig4_itpc_levels.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ ITPC controls
# Addendum of the prereg doc (design written before running): is the post-stroke ITPC rise explained by movement /
# response size? Cued first licks, S8M8, evoked, tau 0, full session. Covariates, coverage rule and the movement design
# are coupling_vigour_sm's (14.18 S17): session_kin + mov_columns + its eligibility rule.
CTRL = TAB / "ctrl_per_session"
CTRL_VERSIONS = ("raw", "kin", "perm")
ITPC_M = ["itpcS_delta", "itpcM_delta", "itpcS_low", "itpcM_low"]
ADJ = ("raw", "snr", "bvar")


def _cv():
    import coupling_vigour_sm as cv
    return cv


def _snr(Y: np.ndarray, ev: np.ndarray) -> dict:
    """Band SNR of traces Y[ev]: power of the event-mean spectrum / mean power of the event deviations (same detrend /
    taper / FFT as the ITPC; all linear, so the spectrum of the mean = the mean of the spectra)."""
    F = _spec(Y[ev])
    mu = F.mean(0)
    pm, pd_ = np.abs(mu) ** 2, (np.abs(F - mu) ** 2).mean(0)
    return {b: float(pm[ix].mean() / pd_[ix].mean()) for b, ix in BIX.items()}


def _within_spout_perm(pos: np.ndarray, rng) -> np.ndarray:
    perm = np.arange(len(pos))
    for s in np.unique(pos):
        ix = np.where(pos == s)[0]
        perm[ix] = rng.permutation(ix)
    return perm


def ctrl_work(item):
    """One session: ITPC of the S / M group means raw, kinematics-regressed and permuted-kinematics control; band SNR;
    the per-event covariates (for behavioral variability in ctrl-summarize)."""
    label, meta, dock_t, _f = item
    cv = _cv()
    from wfield_local import config, epochs
    seed = int(label[2:4]) * 10000 + int(label[-4:])
    an, ep = config.animal_of(label), epochs.epoch_of(label)
    arr = _arrays(label, meta, dock_t)
    if "cued" not in arr or len(arr["cued"][0]) < MIN_N:
        return None
    Z, pos, names = arr["cued"]
    n = len(Z)
    times = cv._event_times(label, meta, dock_t)["cued"]
    if len(times) != n:
        raise RuntimeError(f"{label} cued: {len(times)} event times vs {n} events")
    Kd = cv.session_kin(label, {"cued": times})["cued"]
    Zm, used = cv.mov_columns(Kd)
    Dp = lih._design(pos, None, n)
    # eligibility = coupling_vigour_sm's rule, whose base design is spout + angle (Dp + 1 column)
    ok = bool(Zm.shape[1]) and n >= max(MIN_N, 3 * (Dp.shape[1] + 1 + Zm.shape[1]))
    ix = {nm: i for i, nm in enumerate(names)}
    sa, ma = PAIRINGS["S8M8"]
    s0, m0 = Z[..., [ix[a] for a in sa]].mean(-1), Z[..., [ix[a] for a in ma]].mean(-1)
    tr = {"raw": (s0, m0)}
    r2 = {}
    if ok:
        perm = _within_spout_perm(pos, np.random.default_rng([seed, 7]))
        e0 = {g: lih._resid(Y, Dp) for g, Y in (("S", s0), ("M", m0))}
        for v, Zc in (("kin", Zm), ("perm", Zm[perm])):
            D = np.c_[Dp, Zc]
            out = []
            for g, Y in (("S", s0), ("M", m0)):
                b = np.linalg.lstsq(D, Y, rcond=None)[0]
                Yr = Y - Zc @ b[Dp.shape[1]:]                      # keep spout means + residual
                out.append(Yr)
                e1 = lih._resid(Yr, Dp)
                r2[f"r2_{v}_{g}"] = float(1 - (e1 ** 2).sum() / (e0[g] ** 2).sum())
            tr[v] = tuple(out)
    allev = np.arange(n)
    base = {"label": label, "animal": an, "epoch": ep, "n": n, "mov_ok": ok, "n_mov_cols": int(Zm.shape[1]),
            "used": ",".join(used), "matched": float(Kd.matched.mean()),
            **{f"cov_{c}": float(np.isfinite(Kd[c]).mean()) for c in cv.KIN}, **r2}
    rows = []
    for v, (s, m) in tr.items():
        # the same seed as session_work's (cued, S8M8, evoked, tau 0, full) draw: raw reproduces the published value and
        # the three versions share their subsamples / null phases
        sp = coherence(s, m, allev, N_SUB, np.random.default_rng([seed, 0, 0, 0, 0, 0]))
        rows.append({**base, "version": v, **{k: x for k, x in _bands(sp).items() if k.startswith("itpc")}})
    rs = np.random.default_rng([seed, 0, 0, 0, 0, 0, 1])
    sub = [rs.choice(allev, N_SUB, replace=False) for _ in range(N_DRAW)]
    for g, Y in (("S", s0), ("M", m0)):
        for b, x in _snr(Y, allev).items():
            rows[0][f"snr{g}_{b}"] = x
        for b in BANDS:
            rows[0][f"snr30{g}_{b}"] = float(np.mean([_snr(Y, e)[b] for e in sub]))
    K = Kd.copy()
    K.insert(0, "pos", pos)
    return {"rows": pd.DataFrame(rows), "kin": K}


def ctrl_run(labs, jobs) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    if "COMBINED_ENCODING_FR" not in os.environ:
        sys.exit("run under the frozen env (rr2/frozenenv.sh): COMBINED_ENCODING_FR unset (FaceRhythm covariates)")
    writeguard.assert_writable(OUT)
    CTRL.mkdir(parents=True, exist_ok=True)
    todo = [x for x in labs if not (CTRL / f"{x}.pkl").exists()]
    print(f"ctrl-run: {len(labs)} sessions, {len(todo)} to fit", flush=True)
    t0 = time.time()
    res, fail = ak.fan_sessions(_meta_items(todo, {"ctrl"}), ctrl_work, jobs=min(jobs, 4), key=lambda it: it[0])
    for it, d in res:
        if d is not None:
            with open(CTRL / f"{it[0]}.pkl", "wb") as fh:
                pickle.dump(d, fh)
    if fail:
        print("FAILED:", [str(f[0]) for f in fail], flush=True)
    print(f"ctrl sessions done {time.time() - t0:.0f} s", flush=True)


def _bvar(K: pd.DataFrame) -> pd.DataFrame:
    """Behavioral variability per session: mean over covariates of the within-spout SD (spout dummies removed, the
    coupling_vigour_sm wsd definition) of each covariate z-scored on the animal's pooled pre events. Covariate set per
    animal fixed = those with >= MIN_COV coverage in every one of its sessions (so the mean is over the same set)."""
    cv = _cv()
    out = []
    for a, g in K.groupby("animal"):
        cov = g.groupby("label")[cv.KIN].agg(lambda v: np.isfinite(v).mean())
        keep = [c for c in cv.KIN if (cov[c] >= cv.MIN_COV).all()]
        pre = g[g.epoch == "pre"]
        mu, sd = pre[keep].mean(), pre[keep].std(ddof=1)
        keep = [c for c in keep if np.isfinite(sd[c]) and sd[c] > 0]
        for lab, q in g.groupby("label"):
            Dp = lih._design(q.pos.to_numpy(str), None, len(q))
            w = {}
            for c in keep:
                v = ((q[c] - mu[c]) / sd[c]).to_numpy(float)
                f = np.isfinite(v)
                if f.sum() < 10:
                    continue
                rr = v[f] - Dp[f] @ np.linalg.lstsq(Dp[f], v[f], rcond=None)[0]
                w[c] = float(np.sqrt((rr ** 2).sum() / max(f.sum() - Dp.shape[1], 1)))
            out.append({"label": lab, "animal": a, "epoch": q.epoch.iloc[0], "bvar": float(np.mean(list(w.values()))),
                        "n_cov": len(w), "covs": ",".join(w), **{f"wsdz_{c}": x for c, x in w.items()}})
    return pd.DataFrame(out)


def _adjust(S: pd.DataFrame, m: str, x: str) -> tuple[pd.Series, dict]:
    """Within animal OLS m ~ epoch dummies (pre = reference) + x; returns (adjusted m = m - b (x - mean x), i.e.
    residual + epoch effect, per-animal {b, epoch coefficients})."""
    adj, info = pd.Series(np.nan, index=S.index), {}
    for a, g in S.dropna(subset=[m, x]).groupby("animal"):
        eps = [e for e in EPS[1:] if (g.epoch == e).any()]
        D = np.column_stack([np.ones(len(g))] + [(g.epoch == e).to_numpy(float) for e in eps] + [g[x].to_numpy(float)])
        b = np.linalg.lstsq(D, g[m].to_numpy(float), rcond=None)[0]
        adj[g.index] = g[m] - b[-1] * (g[x] - g[x].mean())
        info[a] = {"slope": float(b[-1]), **{f"eff_{e}": float(b[1 + k]) for k, e in enumerate(eps)}}
    return adj, info


def _share(T: pd.DataFrame, ref: str, alt: str, m: str, ep: str = "chronic") -> dict:
    """Share of the pre -> ep change remaining: alt / ref, per animal (d_<animal>) and pooled (boot_delta point)."""
    a_ = T[(T.metric == m) & (T.epoch == ep)].set_index("variant")
    if ref not in a_.index or alt not in a_.index:
        return {}
    r, q = a_.loc[ref], a_.loc[alt]
    out = {"metric": m, "epoch": ep, "ref": ref, "alt": alt, "ref_delta": r.delta, "alt_delta": q.delta,
           "share_pooled": q.delta / r.delta}
    for a in AN:
        out[f"share_{a}"] = q[f"d_{a}"] / r[f"d_{a}"]
    return out


def ctrl_summarise() -> None:
    cv = _cv()
    files = sorted(CTRL.glob("*.pkl"))
    R, K = [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        R.append(d["rows"])
        k = d["kin"]
        r0 = d["rows"].iloc[0]
        K.append(k.assign(label=r0.label, animal=r0.animal, epoch=r0.epoch))
    R = pd.concat(R, ignore_index=True)
    K = pd.concat(K, ignore_index=True)
    for m in ITPC_M:
        R[f"{m}_d"] = R[m] - R[f"{m}_null"]
    R.to_csv(TAB / "ctrl_sessions.csv.gz", index=False)
    pd.set_option("display.width", 250)
    raw = R[R.version == "raw"].set_index("label")
    # check: raw reproduces the published per-session values
    P = pd.read_csv(TAB / "summary_sessions.csv.gz")
    P = P[(P.set == "cued") & (P.pairing == "S8M8") & (P.metric == "itpcS_delta_d")].set_index("label").value
    dd = (raw.itpcS_delta_d - P.reindex(raw.index)).abs()
    print(f"\nraw vs published itpcS_delta_d: max |diff| {dd.max():.2e} over {dd.notna().sum()} sessions", flush=True)
    # coverage
    ccols = ["matched", "mov_ok"] + [f"cov_{c}" for c in cv.KIN]
    CA = raw.groupby(["animal", "epoch"])[ccols + ["n"]].mean().reindex(EPS, level="epoch").round(3)
    CA.to_csv(TAB / "ctrl_coverage.csv")
    print("\n=== COVERAGE (cued; fraction of events with each covariate; mov_ok = movement design used)\n"
          + CA.to_string(), flush=True)
    print(f"sessions {len(raw)}, mov_ok {int(raw.mov_ok.sum())}; covariate columns used (median) "
          f"{raw.n_mov_cols.median():.0f}", flush=True)
    # behavioural variability
    B = _bvar(K)
    B.to_csv(TAB / "ctrl_behav_var_sessions.csv", index=False)
    print("\n=== PER-ANIMAL behavioral variability (mean within-spout SD, pre-z units)\n"
          + B.groupby(["animal", "epoch"]).bvar.mean().unstack().reindex(columns=EPS).round(3).to_string(), flush=True)
    # wide session table: raw / kin / perm ITPC (minus null) + SNR + bvar
    W = raw[["animal", "epoch", "n"] + [f"{m}_d" for m in ITPC_M]
            + [c for c in raw.columns if c.startswith(("snr", "r2_"))]].reset_index()
    for v in ("kin", "perm"):
        q = R[R.version == v].set_index("label")[[f"{m}_d" for m in ITPC_M]]
        W = W.merge(q.add_suffix(f"_{v}"), left_on="label", right_index=True, how="left")
    W = W.merge(B[["label", "bvar"]], on="label", how="left")
    eff = []
    for m in ITPC_M:
        g_, b_ = m[4], m.split("_")[1]
        W[f"lsnr_{m}"] = np.log(W[f"snr{g_}_{b_}"])
        y = f"{m}_d"
        W[f"{y}_raw"] = W[y]
        for tag, x in (("snr", f"lsnr_{m}"), ("bvar", "bvar")):
            W[f"{y}_{tag}"], info = _adjust(W, y, x)
            for a, i in info.items():
                eff.append({"metric": m, "covariate": tag, "animal": a, **i})
        _, info0 = _adjust(W.assign(_z=0.0), y, "_z")
        for a, i in info0.items():
            eff.append({"metric": m, "covariate": "none", "animal": a, **i})
    W.to_csv(TAB / "ctrl_wide_sessions.csv.gz", index=False)
    E = pd.DataFrame(eff)
    E.to_csv(TAB / "ctrl_epoch_effects_animal.csv", index=False)
    # per-animal tables FIRST (rule 8)
    for m in ITPC_M:
        y = f"{m}_d"
        T_ = pd.concat({v: W.groupby(["animal", "epoch"])[c].mean().unstack().reindex(columns=EPS)
                        for v, c in (("raw", y), ("kin", f"{y}_kin"), ("perm", f"{y}_perm"))}, names=["version"])
        print(f"\n=== PER-ANIMAL {m} (ITPC minus null), raw / kinematics-regressed / permuted control\n"
              + T_.round(3).to_string(), flush=True)
        q = E[E.metric == m].pivot_table(index="animal", columns="covariate", values=["eff_chronic", "slope"])
        print(f"--- {m}: within-animal OLS chronic epoch effect (none / + log SNR / + behav var) and slope\n"
              + q.round(3).to_string(), flush=True)
    print("\n=== PER-ANIMAL log SNR (raw traces, all events)\n"
          + W.groupby(["animal", "epoch"])[[f"lsnr_{m}" for m in ITPC_M]].mean().unstack().round(2).to_string())
    print("\n=== PER-ANIMAL movement R2 of the group traces (event deviations, spout means removed)\n"
          + W.groupby(["animal", "epoch"])[[c for c in W.columns if c.startswith("r2_")]].mean().round(3).to_string())
    # bootstrap: raw (all sessions), raw on the kin-eligible sessions, kin, perm, snr-adjusted, bvar-adjusted
    L = []
    for m in ITPC_M:
        y = f"{m}_d"
        okk = W[f"{y}_kin"].notna()
        for var, col, mask in (("raw", y, None), ("raw_kinset", y, okk), ("kin", f"{y}_kin", okk),
                               ("perm", f"{y}_perm", okk), ("snr", f"{y}_snr", None), ("bvar", f"{y}_bvar", None)):
            q = W if mask is None else W[mask]
            L.append(q[["label", "animal", "epoch"]].assign(metric=m, variant=var, value=q[col].to_numpy()))
    for x in ["bvar"] + [f"lsnr_{m}" for m in ITPC_M]:
        L.append(W[["label", "animal", "epoch"]].assign(metric=x, variant="level", value=W[x].to_numpy()))
    L = pd.concat(L, ignore_index=True)
    WW = L.pivot_table(index=["label", "animal", "epoch", "variant"], columns="metric", values="value").reset_index()
    WW.columns.name = None
    T = ihc._tests(WW, ["variant"], ITPC_M + ["bvar"] + [f"lsnr_{m}" for m in ITPC_M], np.random.default_rng(0))
    T.to_csv(TAB / "ctrl_tests_boot_delta.csv", index=False)
    print("\n=== boot_delta vs August pre (change = CI excl. 0 AND >= 3/4 same sign)\n"
          + T[["variant", "metric", "epoch", "delta", "lo", "hi", "n_same_sign", "excl0"] + [f"d_{a}" for a in AN]]
          .round(4).to_string(index=False), flush=True)
    SH = pd.DataFrame([s for m in ITPC_M for ref, alt in (("raw_kinset", "kin"), ("raw_kinset", "perm"), ("raw", "snr"),
                                                          ("raw", "bvar")) for s in [_share(T, ref, alt, m)] if s])
    SH.to_csv(TAB / "ctrl_share_remaining.csv", index=False)
    print("\n=== SHARE of the pre -> chronic change remaining (alt / ref; per animal and pooled)\n"
          + SH.round(3).to_string(index=False), flush=True)
    FIGD.mkdir(parents=True, exist_ok=True)
    with open(FIGD / "fig5_itpc_controls.pkl", "wb") as fh:
        pickle.dump({"W": W, "T": T, "SH": SH, "B": B, "E": E}, fh)


CTRL_COL = {"raw": "k", "kin": "#d95f02", "perm": "0.6", "snr": "#1b9e77", "bvar": "#7570b3"}
CTRL_LAB = {"raw": "raw", "kin": "kinematics regressed (per event)", "perm": "permuted-kinematics control",
            "snr": "adjusted for log SNR (per session)", "bvar": "adjusted for behavioral variability (per session)"}
ITPC_LAB = {"itpcS_delta": "sensory, 1.5-4 Hz", "itpcM_delta": "motor, 1.5-4 Hz", "itpcS_low": "sensory, 0.5-1.5 Hz",
            "itpcM_low": "motor, 0.5-1.5 Hz"}


def plot_ctrl() -> None:
    """Fig 5: ITPC controls (addendum). Row 1 per-animal ITPC levels by epoch, raw / kinematics-regressed / permuted;
    row 2 change vs pre raw / SNR-adjusted / behavioral-variability-adjusted; row 3 behavioral variability and SNR."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    f = FIGD / "fig5_itpc_controls.pkl"
    if not f.exists():
        return
    with open(f, "rb") as fh:
        D = pickle.load(fh)
    W, T = D["W"], D["T"]
    fig, axs = plt.subplots(3, 4, figsize=(17, 11.5), layout="constrained")
    x = np.arange(len(EPS))
    off = {"raw": -0.18, "kin": 0.0, "perm": 0.18}
    for j, m in enumerate(ITPC_M):
        ax = axs[0, j]
        y = f"{m}_d"
        for v, c in (("raw", y), ("kin", f"{y}_kin"), ("perm", f"{y}_perm")):
            A_ = W.groupby(["animal", "epoch"])[c].mean().unstack().reindex(columns=EPS)
            for a in AN:
                if a in A_.index:
                    ax.plot(x + off[v], A_.loc[a].values, ANIMAL_MARKER[a], color=CTRL_COL[v], mfc="none", ms=4.5,
                            alpha=0.8)
            ax.plot(x + off[v], A_.mean(0).values, "-", color=CTRL_COL[v], lw=2)
        ax.set_xticks(x, EPS)
        ax.set_title(f"A{j + 1}  ITPC, {ITPC_LAB[m]}", fontsize=9.5, loc="left")
        if j == 0:
            ax.set_ylabel("ITPC minus random-phase null\n(animal epoch means; line = mean of animals)", fontsize=8.5)
        ax = axs[1, j]
        eps = EPS[1:]
        offs = {"raw": -0.2, "snr": 0.0, "bvar": 0.2}
        for k, ep in enumerate(eps):
            for v in ADJ:
                r = T[(T.variant == v) & (T.metric == m) & (T.epoch == ep)]
                if r.empty:
                    continue
                r = r.iloc[0]
                xx = k + offs[v]
                dv = [r[f"d_{a}"] for a in AN if np.isfinite(r[f"d_{a}"])]
                ax.plot(np.full(len(dv), xx - 0.06), dv, "o", color="0.65", ms=3)
                okk = r.excl0 and r.n_same_sign >= 3
                ax.errorbar(xx + 0.03, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="s", color=CTRL_COL[v],
                            mfc=CTRL_COL[v] if okk else "w", ms=6, lw=1.4)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(eps)), eps)
        ax.set_xlim(-0.5, len(eps) - 0.5)
        ax.set_title(f"B{j + 1}  change vs pre, {ITPC_LAB[m]}", fontsize=9.5, loc="left")
        if j == 0:
            ax.set_ylabel("change in ITPC minus null vs pre\n(square = boot_delta 95% CI; gray = animals)",
                          fontsize=8.5)
    # row 3: behavioural variability, log SNR (1.5-4 Hz, S and M), movement R2
    B = D["B"]
    ax = axs[2, 0]
    A_ = B.groupby(["animal", "epoch"]).bvar.mean().unstack().reindex(columns=EPS)
    for a in AN:
        ax.plot(x, A_.loc[a].values, "-" + ANIMAL_MARKER[a], color="0.3", mfc="none", ms=5, lw=1)
    ax.plot(x, A_.mean(0).values, "-", color="k", lw=2.2)
    ax.set_xticks(x, EPS)
    ax.set_ylabel("mean within-spout SD of the covariates\n(z units of the animal's pre events)", fontsize=8.5)
    ax.set_title("C1  behavioral variability", fontsize=9.5, loc="left")
    for j, m in enumerate(("itpcS_delta", "itpcM_delta")):
        ax = axs[2, 1 + j]
        A_ = W.groupby(["animal", "epoch"])[f"lsnr_{m}"].mean().unstack().reindex(columns=EPS)
        for a in AN:
            ax.plot(x, A_.loc[a].values, "-" + ANIMAL_MARKER[a], color="0.3", mfc="none", ms=5, lw=1)
        ax.plot(x, A_.mean(0).values, "-", color="k", lw=2.2)
        ax.set_xticks(x, EPS)
        ax.set_ylabel("log SNR (event-mean power /\nmean deviation power)", fontsize=8.5)
        ax.set_title(f"C{2 + j}  response SNR, {ITPC_LAB[m]}", fontsize=9.5, loc="left")
    ax = axs[2, 3]
    for k, c in enumerate(("r2_kin_S", "r2_kin_M")):
        A_ = W.groupby(["animal", "epoch"])[c].mean().unstack().reindex(columns=EPS)
        Ap = W.groupby(["animal", "epoch"])[c.replace("kin", "perm")].mean().unstack().reindex(columns=EPS)
        ls = "-" if k == 0 else "--"
        ax.plot(x, A_.mean(0).values, ls, color=CTRL_COL["kin"], lw=2)
        ax.plot(x, Ap.mean(0).values, ls, color=CTRL_COL["perm"], lw=2)
        for a in AN:
            ax.plot(x + (k - 0.5) * 0.12, A_.loc[a].values, ANIMAL_MARKER[a], color=CTRL_COL["kin"], mfc="none", ms=4)
    ax.set_xticks(x, EPS)
    ax.set_ylabel("variance of the event traces explained\nby the covariates (spout means removed)", fontsize=8.5)
    ax.set_title("C4  movement R2 (solid S, dashed M)", fontsize=9.5, loc="left")
    for ax in axs.ravel():
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    hd = [Line2D([], [], color=CTRL_COL[v], lw=2, marker="s" if v in ("snr", "bvar") else None) for v in
          ("raw", "kin", "perm", "snr", "bvar")]
    hd += [Line2D([], [], marker=ANIMAL_MARKER[a], ls="", color="0.3", mfc="none", ms=6) for a in AN]
    fig.legend(hd, [CTRL_LAB[v] for v in ("raw", "kin", "perm", "snr", "bvar")] + AN, loc="outside lower center",
               ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle(_wrap("ITPC controls (pre-specified addendum): CUED first licks, 8 sensory / 8 motor areas, evoked. "
                       "A: ITPC with per-event movement covariates (3-D reach, peak speed, lick duration, protrusion, "
                       "lick counts, FaceRhythm factors, jaw amplitude, landing error) regressed out within session "
                       "(spout means kept) and with the covariate rows permuted within spout. B: within-animal "
                       "adjustment of the session ITPC for log response SNR or behavioral variability (filled square = "
                       "CI excludes 0 and >= 3/4 animals share the sign). C: the session covariates. PROVISIONAL, "
                       "n = 4 animals."), fontsize=9.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig5_itpc_controls.png", dpi=140)
    plt.close(fig)


def plot() -> None:
    plot_gate()
    plot_change()
    plot_example()
    plot_itpc()
    plot_ctrl()
    print("figures ->", FIGS)


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run", "gate", "summarise", "plot", "ctrl-run",
                                                     "ctrl-summarise"], default="plot")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--all", action="store_true", help="all curated sessions, only the families that passed the gate")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.plot_only or a.cmd == "plot":
        plot()
    elif a.cmd == "run":
        if a.all:
            fams = families(passed_measures())
            if not fams:
                sys.exit("no measure passed the gate: post sessions not fitted (pre-registered)")
            labs = a.labels or labels(pre_only=False)
        else:
            fams = {"coh", "cca"}
            labs = a.labels or labels(pre_only=True)
        run(labs, a.jobs, fams)
    elif a.cmd == "gate":
        gate()
    elif a.cmd == "summarise":
        summarise()
    elif a.cmd == "ctrl-run":
        ctrl_run(a.labels or labels(pre_only=False), a.jobs)
    elif a.cmd == "ctrl-summarise":
        ctrl_summarise()
    print(f"[{a.cmd}] wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
