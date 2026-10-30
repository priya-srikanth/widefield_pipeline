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
  * ITPC null (unspecified): the same subsamples with each event's phases randomised (uniform), mean of the draws.
  * "exceeds its null" = session value minus the mean of the null draws > 0 (95th-percentile version also tabled).
  * halves with < 30 events: both halves use N = min(30, n_half1, n_half2) events per coherence draw.

    python scripts/sm_coherence_cca.py run [--all] [--labels ...] [--jobs 4]
    python scripts/sm_coherence_cca.py gate
    python scripts/sm_coherence_cca.py summarise      # refuses unless a measure passed the gate
    python scripts/sm_coherence_cca.py plot | --plot-only
"""
from __future__ import annotations

import argparse
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
        sys.exit("no measure passed the pre-registered gate: summarise refused")
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
                       "sign; grey dots = animals. n = 4."), fontsize=9.5)
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


def plot() -> None:
    plot_gate()
    plot_change()
    plot_example()
    plot_itpc()
    print("figures ->", FIGS)


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run", "gate", "summarise", "plot"], default="plot")
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
    print(f"[{a.cmd}] wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
