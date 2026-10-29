"""BEFORE the lick: is the contralesional read-out / hemispheric imbalance of SPONTANEOUS ITI licks already present
before a self-initiated lick? (Priya 10-29: "look at a time window before spontaneous ITI licks".)

QUESTION. After the stroke the frozen pre-stroke lateral lick decoder reads ITI licks as more CONTRALESIONAL than where
they land (like-for-like offset, chronic for ITI licks; Fig 12.17: mostly a left / right DIFFERENCE, strongest in SSp-n),
and the post-lick hemispheric balance (0.1..0.5 s) tracks recovery. Motor command / plan, or sensory reafference /
post-movement state? A self-initiated lick (no cue, no lick for >= 2 s) whose contralesional reading and imbalance are
there BEFORE the lick points to a motor / preparatory origin; one that appears only after, to reafference.

LICKS. Exactly the primary ITI set: "ITI first after quiet" (lick_recovery_markers.QSET) = ITI licks of the KEY decoder
cache (lick_order_preds_areas_splitr8) whose onset follows the previous lick onset by >= 2 s (lrm._gaps); angle =
asin(side_k / reach_mm) (covref_covariates), spout = the previous trial's spout (pos). Subset "clean" (sensitivity):
also >= 2 s after the last response-window end and >= 0.6 s before the next spout arrival (iti_timing_audit), so no
trial event falls inside -1.5..+0.6 s.

TWO ALIGNMENTS (both from the session files, sessions_roi8fr):
  peak   iti_X, aligned on the DAQ lick PEAK (contact), lags -1.0..+1.0 s: every QSET lick; the primary balance's own
         traces (its post window reproduces lick_recovery_primary exactly -- printed check).
  onset  bout_peth, aligned on the DAQ lick ONSET, lags -1.5..+0.75 s: the QSET licks that are also extraction bout
         starts (no cue / strobe within -2..+0.75 s of the onset; ~70-80 % of them). Peak - onset ~30 ms (median).
  Traces are linearly interpolated between imaging frames, so a sample < 1 frame before the lick can carry a frame
  after it, and the tongue protrudes ~0.1-0.15 s before contact: treat -0.2..0 as POSSIBLY MOVEMENT.
WINDOWS (s from the lick; lower edge inclusive, upper exclusive, except the decoder's own 0..0.2 s, inclusive):
  pre_100_050 -1.0..-0.5, pre_050_020 -0.5..-0.2, pre_020_000 -0.2..0 (possibly movement), dec_000_020 0..0.2 (the
  frozen decoder's training window), post_010_050 0.1..0.5 (the primary balance window).
BASELINES. Decoders: RAW window level (the frozen decoder is trained on raw levels, no per-lick baseline). Balance:
  "raw" = window level; "bs" = window minus an earlier baseline: peak -1.0..-0.5 s (= il.BASE_RESP, the primary's; so
  bs is undefined for pre_100_050 there), onset -1.5..-1.0 s (= il.BASE; starts >= 0.5 s after the previous lick).

DECODE (lateral landing, mm, + = contralesional; z-scored Allen areas with the ALM / tjM1 split, il.area_features / the
frame-level SD):
  (a) frozen  THE 12.17 decoder: iti_lick_order.predictions' per-animal ITI ridge (all pre ITI licks, lags 0..0.2 s,
              pre sessions leave-one-session-out), captured through lo.CONTRIB_HOOK and applied to each window's features
              (check: on dec_000_020 / peak it reproduces the cached predictions).
  (b) window  StandardScaler + RidgeCV (lick_refit_decoders' ridge) trained on the window itself, on the animal's PRE
              QSET licks, pre sessions leave-one-session-out.
  Pre: held-out r of the decoded value with the lateral landing, with the lick angle, and BEYOND SPOUT (pre per-spout
  means removed from both). Post: the like-for-like offset per session = lick_offset_contrib_map.session_dh (Gaussian
  0.1 mm, reference = the animal's PRE QSET licks in the SAME window, pre sessions referenced to the other pre
  sessions), and its exact split beta . dh into per-area terms -> left / right DIFFERENCE / SHARED parts
  (lick_offset_terms_recovery.components' definition; SSp-n difference; oro + SSs difference w/o SSp-n; ...).
  Change vs pre: analysis_kit.boot_delta (paired nested animals > sessions).
  RULE 10: neither decoder goes through frozen_models.load_or_fit. (a) is the 12.17 decoder, itself a deterministic
  in-memory refit in iti_lick_order (not in the frozen store) -- reproduced bit-for-bit here; (b) is an exploratory
  per-window decoder. Freezing either would write into the shared frozen-model store (outside this script's write
  root) and read every pre session's U_atlas / SVTcorr for the signature -- not done; flagged in the report.
BALANCE. Primary-style: per lick, area x side value referenced to the animal's PRE line on lick angle (per animal x lick
  order, pre sessions leave-one-session-out; vectorised lrm._residuals); pool ALM, tjM1, SSp-m, SSp-n, SSs (contra =
  right hemisphere). ratio = log2(sum obs contra / sum predicted contra) - log2(same, ipsi) (bs only; NaN when a sum is
  <= 0 -- a near-zero pre-lick response makes it unstable, see n_ratio_nan), diff = mean over areas of the session-mean
  referenced (contra - ipsi) (raw and bs). By epoch (boot_delta) and recovery tracking within animals
  (lick_sspn_checks._track: per-animal r with ITI lick angle and the cued far-contra aim, MixedLM within-animal slope).
TIME COURSE. Sliding 0.1 s bins (2 samples, step 0.05 s) from -1.5 (onset; -1.0 peak) to +0.6 s: frozen offset, its
  SSp-n difference and all-pairs difference terms, balance diff raw / bs, change vs pre per epoch (boot_delta_columns,
  the same draw as boot_delta).
Outputs <OUT_W>/iti_prelick_licks.csv.gz (lick set), iti_prelick_sessions.csv (window x measure per session),
  iti_prelick_decode_r.csv (pre held-out r per animal), iti_prelick_tests.csv (change vs pre), iti_prelick_animals.csv
  (per-animal epoch change), iti_prelick_track.csv (recovery tracking), iti_prelick_timecourse_sessions.csv,
  iti_prelick_timecourse_tests.csv; figure figures/lick_iti_prelick.png.
Usage: python scripts/lick_iti_prelick.py [--plot-only]      (--plot-only redraws from the CSVs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_offset_contrib_map as ocm  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
ORO = set(POOL)
WINS = [("pre_100_050", -1.0, -0.5), ("pre_050_020", -0.5, -0.2), ("pre_020_000", -0.2, 0.0),
        ("dec_000_020", 0.0, 0.2), ("post_010_050", 0.1, 0.5)]
SRC = {"peak": (il.LAGS_D, il.BASE_RESP), "onset": (il.LAGS_P, il.BASE)}
SUBSETS = ("all", "clean")
SIGMA = ocm.SIGMA
MIN_CELL = 20                     # = lick_balance_regions.MIN_CELL (pre licks for a pre line)
MIN_PRE_LICKS = 10                # window decoder: pre licks per training session
TC_HI = 0.6
OUT = il.OUT_W
FIG = OUT / "figures"
COMP = ["total", "diff_all", "sum_all", "diff_SSp-n", "diff_oro_noSSpn", "sum_oro", "other", "unpaired"]


# ------------------------------------------------------------------------------------------------------- lick set
def lick_set() -> pd.DataFrame:
    P = pd.read_csv(lo.preds_path())
    P["k"] = P.groupby(["set", "label"]).cumcount()
    P = P[P.set == "iti"].copy()
    C = pd.read_csv(OUT / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "t", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), lrm._gaps)
    if fail:
        raise RuntimeError(f"gaps failed: {fail[:5]}")
    G = pd.concat([v for _i, v in res], ignore_index=True)
    P = P.merge(G, on=["set", "label", "k"], how="left", validate="one_to_one")
    acts = P.groupby("label").act.apply(lambda s: s.to_numpy(float))
    Q = P[P.gap >= il.GAP_MAIN].copy()
    Q["angle"] = np.degrees(np.arcsin(np.clip(Q.side_k / Q.reach_mm, -1, 1)))
    Q["ord_c"] = np.minimum(Q.order, lo.ORDER_CAP)
    A = pd.read_csv(il.OUT / "iti_timing_audit.csv", usecols=["label", "t", "since_stop", "to_strobe"])
    A = A.replace(np.inf, 1e9).drop_duplicates(["label", "t"])
    Q = Q.merge(A, on=["label", "t"], how="left")
    Q["clean"] = (Q.since_stop >= 2.0) & (Q.to_strobe >= TC_HI)
    Q = Q.sort_values(["label", "k"]).reset_index(drop=True)
    return Q, acts


def _one(item):
    """One session (module-level for spawn): item = (label, peak times of its QSET licks). Returns the peak-aligned
    traces (n, LAGS_D, cols) and the onset-aligned bout traces (n, LAGS_P, cols; NaN where the lick is not a bout start)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    label, tq = item
    tq = np.asarray(tq, float)
    d = il._load(label)
    it = d["iti_t"].astype(float)
    o = np.argsort(it)
    j = np.clip(np.searchsorted(it[o], tq), 0, len(it) - 1)
    rows = o[j]
    okp = np.abs(it[rows] - tq) < 1e-9
    Xp = np.full((len(tq), len(il.LAGS_D), len(d["cols"])), np.nan, np.float32)
    Xp[okp] = d["iti_X"][rows[okp]].astype(np.float32)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    A = A[np.isfinite(A.daq_on_s)].drop_duplicates("daq_peak_s")
    on = A.set_index("daq_peak_s").daq_on_s.reindex(tq).to_numpy(float)
    bt = d["bout_t"].astype(float)
    Xo = np.full((len(tq), len(il.LAGS_P), len(d["cols"])), np.nan, np.float32)
    oko = np.zeros(len(tq), bool)
    if len(bt):
        ob = np.argsort(bt)
        jb = np.clip(np.searchsorted(bt[ob], on), 0, len(bt) - 1)
        rb = ob[jb]
        oko = np.isfinite(on) & (np.abs(bt[rb] - on) < 1e-9)
        Xo[oko] = d["bout_peth"][rb[oko]].astype(np.float32)
    return {"Xp": Xp, "Xo": Xo, "okp": okp, "oko": oko, "on": on, "cols": [str(c) for c in d["cols"]]}


# ------------------------------------------------------------------------------------------------------- helpers
def wmask(lags: np.ndarray, lo_: float, hi_: float, incl: bool = False) -> np.ndarray:
    return (lags >= lo_ - 1e-9) & ((lags <= hi_ + 1e-9) if incl or hi_ >= lags.max() - 1e-9 else (lags < hi_ - 1e-9))


def frozen_fits() -> dict:
    """{animal: (S keys, fits)} of the 12.17 ITI decoder (lo.predictions through CONTRIB_HOOK)."""
    F = {}

    def hook(st, a, S, fits):
        if st == "iti":
            F[a] = (set(S), fits)
    lo.CONTRIB_HOOK = hook
    Pf = lo.predictions()
    lo.CONTRIB_HOOK = None
    return F, Pf


def fit_window(Zpre: dict, ypre: dict) -> dict:
    """{held-out pre label | None: (scaler, ridge)} trained on the window features of pre licks."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    labs = [x for x in sorted(Zpre) if len(ypre[x]) >= MIN_PRE_LICKS]
    out = {}
    if len(labs) < 3:
        return out
    for held in [*labs, None]:
        trn = [q for q in labs if q != held]
        Xt, yt = np.vstack([Zpre[q] for q in trn]), np.concatenate([ypre[q] for q in trn])
        sc = StandardScaler().fit(Xt)
        out[held] = (sc, RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), yt))
    return out


def split_terms(beta: np.ndarray, dh: np.ndarray, names: list[str]) -> dict:
    """The exact components of beta . dh (lick_offset_terms_recovery.components)."""
    rec = dict.fromkeys(COMP, 0.0)
    rec["total"] = float(beta @ dh)
    base = {}
    for i, n in enumerate(names):
        ar, h = n.rsplit("_", 1)
        base.setdefault(ar, {})[h] = i
    for ar, hh in base.items():
        if {"left", "right"} <= set(hh):
            r_, l_ = hh["right"], hh["left"]
            d_ = 0.5 * (beta[r_] - beta[l_]) * (dh[r_] - dh[l_])
            s_ = 0.5 * (beta[r_] + beta[l_]) * (dh[r_] + dh[l_])
            rec["diff_all"] += d_
            rec["sum_all"] += s_
            if ar == "SSp-n":
                rec["diff_SSp-n"] += d_
            elif ar in ORO:
                rec["diff_oro_noSSpn"] += d_
            if ar in ORO:
                rec["sum_oro"] += s_
            else:
                rec["other"] += d_ + s_
        else:
            rec["unpaired"] += float(sum(beta[i] * dh[i] for i in hh.values()))
    return rec


def residualize(V: np.ndarray, M: pd.DataFrame) -> np.ndarray:
    """V (n, m) minus the animal's PRE line on angle per animal x lick order; pre licks by the line fitted without their
    own session (vectorised lrm._residuals; rows with any NaN value or NaN angle are not fitted / get NaN)."""
    R = np.full(V.shape, np.nan)
    ok = np.isfinite(V).all(1) & np.isfinite(M.angle.to_numpy(float))
    ang = M.angle.to_numpy(float)
    lab = M.label.to_numpy()
    pre = (M.epoch == "pre").to_numpy()
    for _k, g in M.groupby(["animal", "ord_c"]).indices.items():
        g = np.asarray(g)
        pk = g[pre[g] & ok[g]]

        def fit(ix):
            if len(ix) < MIN_CELL or ang[ix].std() == 0:
                return None
            X = np.column_stack([np.ones(len(ix)), ang[ix]])
            return np.linalg.lstsq(X, V[ix], rcond=None)[0]
        B = fit(pk)
        post = g[~pre[g]]
        if B is not None and len(post):
            R[post] = V[post] - np.column_stack([np.ones(len(post)), ang[post]]) @ B
        for lb in np.unique(lab[pk]):
            Bq = fit(pk[lab[pk] != lb])
            sel = g[pre[g] & (lab[g] == lb)]
            if Bq is not None:
                R[sel] = V[sel] - np.column_stack([np.ones(len(sel)), ang[sel]]) @ Bq
    R[~ok] = np.nan
    return R


def balance_sessions(M: pd.DataFrame, vals: dict, kind: str) -> pd.DataFrame:
    """vals {area: (contra (n,), ipsi (n,))} -> per session diff (+ ratio for 'bs')."""
    V = np.column_stack([v for a in POOL for v in vals[a]])
    R = residualize(V, M)
    out = []
    for lab, ix in M.groupby("label").indices.items():
        o, r = V[ix], R[ix]
        okk = np.isfinite(r)
        rec = {"label": lab, "n_bal": int(okk.all(1).sum())}
        d = [np.nanmean(r[:, 2 * i] - r[:, 2 * i + 1]) if okk[:, 2 * i].any() else np.nan for i in range(len(POOL))]
        rec["bal_diff"] = float(np.mean(d)) if np.isfinite(d).all() else np.nan
        if kind == "bs":
            g = []
            for s in (0, 1):
                cs = [2 * i + s for i in range(len(POOL))]
                ob = np.where(okk[:, cs], o[:, cs], 0).sum()
                pr = np.where(okk[:, cs], o[:, cs] - r[:, cs], 0).sum()
                g.append(np.log2(ob / pr) if ob > 0 and pr > 0 else np.nan)
            rec["bal_ratio"] = g[0] - g[1]
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------------- compute
def compute() -> None:
    import iti_where_decoder as wd
    Q, acts = lick_set()
    items = [(lab, tuple(g.t.to_numpy(float))) for lab, g in Q.groupby("label", sort=True)]
    res, fail = ak.fan_sessions(items, _one, key=lambda it: it[0])
    if fail:
        raise RuntimeError(f"session load failed: {[f[0] if isinstance(f, tuple) else f for f in fail][:5]}")
    T = {it[0]: v for it, v in res}
    COLS = T[items[0][0]]["cols"]
    Q["ok_peak"] = np.concatenate([T[lab]["okp"] for lab, _ in items])
    Q["ok_onset"] = np.concatenate([T[lab]["oko"] for lab, _ in items])
    Q["t_onset"] = np.concatenate([T[lab]["on"] for lab, _ in items])
    print(f"== QSET licks: {len(Q)}; peak-matched {Q.ok_peak.mean():.3f}; onset (bout-start) matched {Q.ok_onset.mean():.3f}; "
          f"clean {Q.clean.mean():.3f}; peak - onset ms median {np.nanmedian((Q.t - Q.t_onset) * 1000):.1f}", flush=True)
    print(Q.groupby(["animal", "epoch"]).agg(n=("k", "size"), onset=("ok_onset", "mean"), clean=("clean", "mean"))
          .round(2).unstack("epoch").to_string(), flush=True)
    Q.drop(columns=[c for c in ("gap",) if c in Q]).to_csv(OUT / "iti_prelick_licks.csv.gz", index=False)
    names, SD = wd.area_meta()
    F, Pf = frozen_fits()
    # z-scored decoder features over all lags, per source x label
    Z, X = {}, {}
    for src in SRC:
        key = "Xp" if src == "peak" else "Xo"
        X[src] = {lab: T[lab][key] for lab, _ in items}
        Z[src] = {lab: (il.area_features(X[src][lab].astype(float), COLS, lab) /
                        SD.loc[lab, names].to_numpy(float)[None, None, :]).astype(np.float32) for lab, _ in items}
    # CHECK: the frozen decoder on dec_000_020 / peak reproduces the cached predictions
    m = wmask(il.LAGS_D, 0.0, 0.2, incl=True)
    d1, d0 = [], []
    for lab, g in Q.groupby("label"):
        a = g.animal.iloc[0]
        if a not in F or lab not in F[a][0]:
            continue
        sc, mdl = F[a][1][lab if lab in F[a][1] else None]
        d1.append(mdl.predict(sc.transform(Z["peak"][lab][:, m].mean(1))))
        d0.append(g.dec.to_numpy(float))
    d1, d0 = np.concatenate(d1), np.concatenate(d0)
    print(f"== CHECK frozen decoder reproduction (dec_000_020, peak): max |diff| {np.nanmax(np.abs(d1 - d0)):.2e} over "
          f"{len(d1)} licks", flush=True)
    ci = {c: [COLS.index(f"{a}_right") for a in POOL] if c == "c" else [COLS.index(f"{a}_left") for a in POOL]
          for c in ("c", "i")}

    SESS, RR, TC = [], [], []
    for src, (lags, base) in SRC.items():
        okcol = "ok_peak" if src == "peak" else "ok_onset"
        bm = wmask(lags, *base)
        bins = [(round(b, 3), round(b + 0.1, 3)) for b in np.arange(lags[0], TC_HI - 0.1 + 1e-9, 0.05)]
        for sub in SUBSETS:
            sel = Q[okcol].to_numpy(bool) & (Q.clean.to_numpy(bool) if sub == "clean" else True)
            pos_in, meta = {}, {}                  # row positions inside the per-label arrays (Q order), lick metadata
            for lab, ix in Q.groupby("label").indices.items():
                if sel[ix].any():
                    pos_in[lab] = np.flatnonzero(sel[ix])
                    meta[lab] = Q.iloc[ix[sel[ix]]]
            labs = sorted(pos_in)
            M = Q[sel]

            def windows_of(lo_, hi_, incl=False, lags=lags, src=src, pos_in=pos_in, labs=labs):
                w = wmask(lags, lo_, hi_, incl)
                return ({lab: Z[src][lab][pos_in[lab]][:, w].mean(1) for lab in labs},
                        {lab: X[src][lab][pos_in[lab]][:, w].mean(1) for lab in labs})
            Xb = {lab: X[src][lab][pos_in[lab]][:, bm].mean(1) for lab in labs}
            specs = [(n, lo_, hi_, n == "dec_000_020", "win") for n, lo_, hi_ in WINS] + \
                    [(f"tc_{b0:+.2f}", b0, b1, False, "tc") for b0, b1 in bins]
            for wname, lo_, hi_, incl, kind in specs:
                if lo_ < lags[0] - 1e-9:
                    continue
                Zw, Xw = windows_of(lo_, hi_, incl)
                rows = {lab: {"source": src, "subset": sub, "window": wname, "w_lo": lo_, "w_hi": hi_, "label": lab,
                              "animal": meta[lab].animal.iloc[0], "epoch": meta[lab].epoch.iloc[0], "n": len(meta[lab])}
                        for lab in labs}
                # ---- decoders
                for dname in (("frozen", "window") if kind == "win" else ("frozen",)):
                    for a in AN:
                        la = [lab for lab in labs if rows[lab]["animal"] == a]
                        if dname == "frozen":
                            if a not in F:
                                continue
                            la = [lab for lab in la if lab in F[a][0]]
                            fits = F[a][1]
                        else:
                            pre = [lab for lab in la if rows[lab]["epoch"] == "pre"]
                            fits = fit_window({x: Zw[x] for x in pre}, {x: meta[x].act.to_numpy(float) for x in pre})
                            if not fits:
                                continue
                            la = [lab for lab in la if rows[lab]["epoch"] != "pre" or lab in fits]
                        if not any(rows[x]["epoch"] == "pre" for x in la):
                            continue
                        sc0, m0 = fits[None]
                        beta = m0.coef_ / sc0.scale_

                        def dec(x, Xs, fits=fits):
                            s_, m_ = fits[x if x in fits else None]
                            return m_.predict(s_.transform(Xs))
                        S = {x: (rows[x]["epoch"], Zw[x], meta[x].act.to_numpy(float)) for x in la}
                        sess = ocm.session_dh(S, dec, SIGMA)
                        for x, (_e, dh, ro) in sess.items():
                            sp = split_terms(beta, dh, names)
                            rows[x].update({f"{dname}|readout": ro, **{f"{dname}|{k}": v for k, v in sp.items()}})
                        if kind == "win":                          # pre held-out r
                            pre = [x for x in la if rows[x]["epoch"] == "pre"]
                            dv = np.concatenate([dec(x, Zw[x]) for x in pre])
                            mm = pd.concat([meta[x] for x in pre])
                            yv, av, pv = mm.act.to_numpy(float), mm.angle.to_numpy(float), mm.pos.to_numpy()
                            dvb = dv - pd.Series(dv).groupby(pv).transform("mean").to_numpy()
                            yvb = yv - pd.Series(yv).groupby(pv).transform("mean").to_numpy()
                            oka = np.isfinite(av)
                            RR.append({"source": src, "subset": sub, "window": wname, "decoder": dname, "animal": a,
                                       "n_pre": len(dv), "n_pre_sess": len(pre), "r_lateral": np.corrcoef(dv, yv)[0, 1],
                                       "r_angle": np.corrcoef(dv[oka], av[oka])[0, 1],
                                       "r_beyond_spout": np.corrcoef(dvb, yvb)[0, 1]})
                # ---- balance
                Mw = pd.concat([meta[lab] for lab in labs], ignore_index=True)
                for bk in ("raw", "bs"):
                    if bk == "bs" and lo_ < base[1] - 1e-9:        # window overlaps / precedes the baseline
                        continue
                    vals = {}
                    for i, a in enumerate(POOL):
                        vc = np.concatenate([Xw[lab][:, ci["c"][i]] - (Xb[lab][:, ci["c"][i]] if bk == "bs" else 0)
                                             for lab in labs])
                        vi = np.concatenate([Xw[lab][:, ci["i"][i]] - (Xb[lab][:, ci["i"][i]] if bk == "bs" else 0)
                                             for lab in labs])
                        vals[a] = (vc, vi)
                    B = balance_sessions(Mw, vals, bk)
                    for r in B.itertuples():
                        rows[r.label][f"bal_{bk}|diff"] = r.bal_diff
                        if bk == "bs":
                            rows[r.label]["bal_bs|ratio"] = r.bal_ratio
                        rows[r.label]["n_bal"] = r.n_bal
                (SESS if kind == "win" else TC).extend(rows.values())
            print(f"   {src} / {sub}: {len(labs)} sessions, {len(M)} licks", flush=True)
    S = pd.DataFrame(SESS)
    TCs = pd.DataFrame(TC)
    S.to_csv(OUT / "iti_prelick_sessions.csv", index=False)
    TCs.to_csv(OUT / "iti_prelick_timecourse_sessions.csv", index=False)
    pd.DataFrame(RR).to_csv(OUT / "iti_prelick_decode_r.csv", index=False)
    # CHECK vs the primary marker (post_010_050 bs ratio, peak, all) -- lick_recovery_primary.marker on balsub_sessions
    try:
        import lick_recovery_primary as lrp
        Bs = pd.read_csv(OUT / "balsub_sessions.csv")
        Bs = Bs[Bs.set == lrm.QSET]
        mk = pd.Series(lrp.marker(Bs, "ratio_010_050").to_numpy(), index=Bs.label)
        mine = S[(S.source == "peak") & (S.subset == "all") & (S.window == "post_010_050")].set_index("label")["bal_bs|ratio"]
        j = pd.concat([mk.rename("primary"), mine.rename("here")], axis=1).dropna()
        print(f"== CHECK vs primary marker (post 0.1-0.5, peak, all): n {len(j)}, r {np.corrcoef(j.primary, j.here)[0, 1]:.6f}, "
              f"max |diff| {np.abs(j.primary - j.here).max():.2e}", flush=True)
    except Exception as ex:                                        # noqa: BLE001
        print(f"== CHECK vs primary marker failed: {ex}")
    tests(S, TCs)


def measures(S: pd.DataFrame) -> list[str]:
    return [c for c in S.columns if "|" in c]


def tests(S: pd.DataFrame, TCs: pd.DataFrame) -> None:
    import lick_sspn_checks as ssc
    rng = np.random.default_rng(0)
    out, anim = [], []
    for (src, sub, w), g in S.groupby(["source", "subset", "window"], sort=False):
        for c in measures(g):
            h = g.dropna(subset=[c])
            prd = {a: q[c].to_numpy() for a, q in h[h.epoch == "pre"].groupby("animal")}
            pm = h[h.epoch == "pre"].groupby("animal")[c].mean()
            for ep in EP:
                post = {a: q[c].to_numpy() for a, q in h[h.epoch == ep].groupby("animal") if a in prd}
                am = (h[h.epoch == ep].groupby("animal")[c].mean() - pm).dropna()
                for a, v in am.items():
                    anim.append({"source": src, "subset": sub, "window": w, "measure": c, "epoch": ep, "animal": a,
                                 "change": v})
                if not post:
                    continue
                iv = ak.boot_delta(post, prd, rng)
                if iv is None:
                    continue
                out.append({"source": src, "subset": sub, "window": w, "measure": c, "epoch": ep, "point": iv.point,
                            "lo": iv.lo, "hi": iv.hi, "n_animals": iv.n_animals,
                            "n_dir": int((np.sign(am) == np.sign(iv.point)).sum()), "pre_level": float(pm.mean()),
                            "n_ratio_nan": int(g[c].isna().sum()) if c.endswith("ratio") else 0})
    Tt, A = pd.DataFrame(out), pd.DataFrame(anim)
    Tt.to_csv(OUT / "iti_prelick_tests.csv", index=False)
    A.to_csv(OUT / "iti_prelick_animals.csv", index=False)
    # recovery tracking (within animal), per source x subset x window
    trk = []
    for (src, sub, w), g in S.groupby(["source", "subset", "window"], sort=False):
        cols = [c for c in measures(g) if c.split("|")[1] in ("readout", "total", "diff_all", "diff_SSp-n", "diff", "ratio")]
        t = ssc._track(g.assign(set="iti"), cols, f"{src}|{sub}|{w}")
        trk.append(t.assign(source=src, subset=sub, window=w))
    TR = pd.concat(trk, ignore_index=True)
    TR.to_csv(OUT / "iti_prelick_track.csv", index=False)
    # time course: change vs pre per bin (boot_delta_columns; sessions with a NaN bin dropped per measure)
    rows = []
    for (src, sub), g in TCs.groupby(["source", "subset"], sort=False):
        wins = sorted(g.window.unique(), key=lambda s: float(s[3:]))
        for c in measures(g):
            Wd = g.pivot_table(index=["label", "animal", "epoch"], columns="window", values=c).reindex(columns=wins)
            Wd = Wd.dropna(axis=1, how="all")
            wins_c = list(Wd.columns)
            Wd = Wd.dropna(axis=0, how="any").reset_index()
            if Wd.empty:
                continue
            pre = {a: q[wins_c].to_numpy(float) for a, q in Wd[Wd.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: q[wins_c].to_numpy(float) for a, q in Wd[Wd.epoch == ep].groupby("animal") if a in pre}
                if not post:
                    continue
                r = ak.boot_delta_columns(post, pre, np.random.default_rng(0))
                if r is None:
                    continue
                pt, lo_, hi_, _dr, na = r
                am = np.array([post[a].mean(0) - pre[a].mean(0) for a in sorted(post)])
                for k, w in enumerate(wins_c):
                    rows.append({"source": src, "subset": sub, "measure": c, "epoch": ep, "bin_lo": float(w[3:]),
                                 "bin_mid": float(w[3:]) + 0.025, "point": pt[k], "lo": lo_[k], "hi": hi_[k],
                                 "n_animals": na, "n_dir": int((np.sign(am[:, k]) == np.sign(pt[k])).sum()),
                                 **{f"chg_{a}": v for a, v in zip(sorted(post), am[:, k])}})
    TT = pd.DataFrame(rows)
    TT.to_csv(OUT / "iti_prelick_timecourse_tests.csv", index=False)
    report()


# ------------------------------------------------------------------------------------------------------- report + plot
def report() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    R = pd.read_csv(OUT / "iti_prelick_decode_r.csv")
    A = pd.read_csv(OUT / "iti_prelick_animals.csv")
    Tt = pd.read_csv(OUT / "iti_prelick_tests.csv")
    TR = pd.read_csv(OUT / "iti_prelick_track.csv")
    TT = pd.read_csv(OUT / "iti_prelick_timecourse_tests.csv")
    print("== PRE held-out r per animal (rule 8)")
    print(R.pivot_table(index=["source", "subset", "decoder", "window"], columns="animal",
                        values=["r_lateral", "r_beyond_spout"], sort=False).round(2).to_string())
    key = ["frozen|readout", "frozen|diff_all", "frozen|diff_SSp-n", "window|readout", "bal_raw|diff", "bal_bs|diff",
           "bal_bs|ratio"]
    for sub in SUBSETS:
        for src in SRC:
            a = A[(A.source == src) & (A.subset == sub) & A.measure.isin(key)]
            print(f"== per-animal change vs pre ({src}, {sub})")
            print(a.pivot_table(index=["measure", "window"], columns=["epoch", "animal"], values="change", sort=False)
                  .reindex(columns=EP, level=0).round(3).to_string())
            t = Tt[(Tt.source == src) & (Tt.subset == sub) & Tt.measure.isin(key)]
            print(t[["measure", "window", "epoch", "point", "lo", "hi", "n_dir", "n_animals", "pre_level", "n_ratio_nan"]]
                  .round(3).to_string(index=False))
    print("== recovery tracking (per-animal r; p_within = MixedLM within-animal slope)")
    print(TR[TR.measure.isin(key)].round(3).to_string(index=False))
    print("== time course: bins whose CI excludes 0 (frozen readout, SSp-n diff, balance diff)")
    s = TT[((TT.lo > 0) | (TT.hi < 0)) & TT.measure.isin(["frozen|readout", "frozen|diff_SSp-n", "bal_raw|diff",
                                                          "bal_bs|diff"])]
    print(s.groupby(["source", "subset", "measure", "epoch"]).bin_lo.apply(lambda v: list(np.round(v, 2))).to_string())


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    TT = pd.read_csv(OUT / "iti_prelick_timecourse_tests.csv")
    Tt = pd.read_csv(OUT / "iti_prelick_tests.csv")
    R = pd.read_csv(OUT / "iti_prelick_decode_r.csv")
    tcm = [("frozen|readout", "frozen decoder: like-for-like offset (mm)"),
           ("frozen|diff_SSp-n", "  of which SSp-n left/right DIFFERENCE (mm)"),
           ("bal_raw|diff", "balance, raw level: contra - ipsi vs pre line"),
           ("bal_bs|diff", "balance, minus baseline: contra - ipsi vs pre line")]
    fig, axs = plt.subplots(3, 4, figsize=(21, 13.5))
    for i, src in enumerate(("onset", "peak")):
        for j, (m, lab) in enumerate(tcm):
            ax = axs[i, j]
            for ep in EP:
                t = TT[(TT.source == src) & (TT.subset == "all") & (TT.measure == m) & (TT.epoch == ep)]
                if t.empty:
                    continue
                ax.plot(t.bin_mid, t.point, color=EPOCH_LINE[ep], lw=2, label=ep)
                ax.fill_between(t.bin_mid, t.lo, t.hi, color=EPOCH_LINE[ep], alpha=0.15, lw=0)
                sig = (t.lo > 0) | (t.hi < 0)
                ax.plot(t.bin_mid[sig], t.point[sig], "o", color=EPOCH_LINE[ep], ms=4)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.axvline(0, color="k", lw=0.8)
            ax.axvspan(-0.15, 0, color="0.85", alpha=0.6, lw=0)
            base = SRC[src][1]
            if m == "bal_bs|diff":
                ax.axvspan(*base, color="#cfe3f0", alpha=0.5, lw=0)
            ax.set_title(f"{src.upper()}-aligned: {lab}\nchange vs pre, 95% paired nested bootstrap (dots: CI excl. 0)",
                         fontsize=8.5)
            ax.set_xlabel(f"time from lick {src} (s; 0.1 s bins at their sample mean)", fontsize=8)
            if j == 0:
                ax.legend(fontsize=8, frameon=False)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    # row 3: windows
    wins = [w for w, _a, _b in WINS]
    ax = axs[2, 0]
    for k, (dname, mk) in enumerate((("frozen", "o"), ("window", "s"))):
        r = R[(R.source == "onset") & (R.subset == "all") & (R.decoder == dname)]
        for a, h in r.groupby("animal"):
            h = h.set_index("window").reindex(wins)
            ax.plot(np.arange(len(wins)) + (k - 0.5) * 0.2, h.r_lateral, mk, color="#7b3294" if dname == "frozen"
                    else "#2c7fb8", mfc="none" if a in ("PS93", "PS95") else None, ms=6)
        mu = r.groupby("window").r_lateral.mean().reindex(wins)
        ax.plot(np.arange(len(wins)) + (k - 0.5) * 0.2, mu, "-", color="#7b3294" if dname == "frozen" else "#2c7fb8",
                lw=1.5, label=f"{dname} decoder (mean of animals)")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(range(len(wins)), [w.replace("_", "\n") for w in wins], fontsize=7)
    ax.set_title("PRE held-out r (decoded vs lateral landing), onset-aligned\none marker per animal", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    for j, (m, lab) in enumerate((("frozen|readout", "frozen offset (mm)"), ("frozen|diff_SSp-n", "SSp-n difference (mm)"),
                                  ("bal_bs|ratio", "balance gain ratio (log2; bs)"))):
        ax = axs[2, j + 1]
        for k, ep in enumerate(EP):
            t = Tt[(Tt.source == "onset") & (Tt.subset == "all") & (Tt.measure == m) & (Tt.epoch == ep)].set_index(
                "window").reindex(wins)
            x = np.arange(len(wins)) + (k - 1) * 0.22
            ax.errorbar(x, t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o", color=EPOCH_LINE[ep], capsize=2,
                        label=ep)
            for xx, (_w, r) in zip(x, t.iterrows()):
                if np.isfinite(r.point):
                    ax.text(xx, r.hi, f"{int(r.n_dir)}/{int(r.n_animals)}", ha="center", va="bottom", fontsize=6)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xticks(range(len(wins)), [w.replace("_", "\n") for w in wins], fontsize=7)
        ax.set_title(f"onset-aligned windows: {lab}, change vs pre\n(k/n = animals with the sign of the mean)", fontsize=8.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("BEFORE the lick: frozen pre-stroke lateral decoder (12.17) and hemispheric balance (ALM, tjM1, SSp-m, SSp-n, "
                 "SSs; contra = right) around SPONTANEOUS ITI licks (first lick after >= 2 s quiet). + = contralesional.\n"
                 "Rows 1-2: sliding 0.1 s bins (onset-aligned: bout-start subset, baseline -1.5..-1.0 s, blue; "
                 "peak-aligned: all licks, baseline -1.0..-0.5 s); grey = last 0.15 s (tongue already moving). "
                 "Row 3: fixed windows. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "lick_iti_prelick.png", dpi=90)
    plt.close(fig)


def main() -> int:
    if "--plot-only" not in sys.argv:
        compute()
    else:
        report()
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
