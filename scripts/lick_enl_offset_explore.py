"""Follow-ups of lick_enl_angle_decode.py (ENL area features -> angle of the first post-cue lick; Priya 10-29).

TASK A (--history): is the WITHIN-SPOUT ENL signal real beyond trial history and face movement?
  One common PRE trial set per animal (W = 1.0 s ENL window; lick-free rule "slide" primary -- the imaging ENL decoders'
  sliding lick-free window, Priya 11-04 ~20:00 -- and "2s" (the pre-11-05 fixed window, no lick in the last 2 s) as the
  robustness row) -- the trials of lick_enl_angle_decode.select. The A2 face windows (-1.0..-0.1 / -2.0..-1.0 s) stay
  FIXED on the FaceRhythm grid (covariates, not the ENL decoder's window). Target = the first-lick angle deviation within session x spout. Every regressor is
  residualised within session x spout too (the target has no between-cell variance), then ridge (StandardScaler +
  RidgeCV, as the decoder), leave-one-pre-session-out, held-out pooled over trials:
    hist   = trial-history regressors: previous trial's spout x current spout (one-hot), previous minus current lateral /
             distance step, same-spout flag, previous trial hit (any lick cue..stop) and engaged flags, previous trial's
             first-lick angle deviation (within session x spout, from ALL matched trials; 0 + a missing flag when the
             previous trial has none), log cue gap, time in session (z, z^2). Previous = the preceding row of the full
             DAQ trial table (il._behaviour), not the preceding first-lick trial.
    full   = hist + the ENL area features.
  full is TWO-STAGE (primary): inside each training fold the ENL ridge is fitted to the history ridge's residual, so the
  two keep separate penalties (a joint ridge on [hist, ENL] shares one alpha; reported as dR2_joint, no null).
  Score: held-out R2 (1 - SSE / SS of the within-cell target), delta R2 = full - hist, r of (full - hist) with
  (y - hist). Null: ENL rows permuted within session x spout (history and target fixed), the full model refit,
  nshuf times; p per animal, Fisher-combined over the 4 animals.
  A2: + pre-cue FACE MOVEMENT: the 5 FaceRhythm master factors (fr_lick, fr_right, fr_left, fr_close, fr_far;
  il.fr_unblanked, the frozen encoder's set, UNBLANKED: 25 Hz inside the -2..+4 s trial window) averaged over the ENL
  window -1.0..-0.1 s AND over -2.0..-1.0 s before the cue, added to hist (sessions with a FaceRhythm projection).
  A3: which areas (two-stage, no null): per left/right area pair, (drop) delta R2 of the full model minus the full model
  without the pair; (single) delta R2 of hist + that pair over hist.
TASK B (--offset): the acute contralesional like-for-like ENL offset (frozen all-pre raw decoder, W 1.0, lf slide;
  frozen ids enl_w1_lfslide[_restw|_smean], new 11-05).
  B0 per-trial offsets rebuilt (lfl.offset, method nw) from the FROZEN decoders (lick_enl_angle_decode.frozen_fits) and
     checked against enl_angle_decode_sessions.csv.
  B1 per session vs recovery (recmark_sessions.csv aim / iti_ang; lick_sspn_checks._track: per-animal r on post
     sessions + MixedLM y ~ w + b + logday, groups = animal).
  B2 by target spout (orofacial_cohort.POS_ORDER): per animal epoch - pre of session x spout means + paired bootstrap.
  B3 exact area decomposition (lick_offset_contrib_map.session_dh, Gaussian nw rule, sigma from likeforlike.choose):
     offset = sum_k beta_k dh_k, beta = ridge coef / scaler scale; each left / right pair = DIFFERENCE
     0.5 (b_r - b_l)(dh_r - dh_l) + SHARED 0.5 (b_r + b_l)(dh_r + dh_l).
  B4 baseline shift or aim information? (a) features minus the session's restw (precue_restw_sessions.csv, / area SD),
     (b) features minus the session's equal-weight mean over spouts of the per-spout ENL means; each with its own
     FROZEN pre decoder (kinds enl_angle, variants _restw / _smean) and the same offset; (c) session offset vs the
     session's pre-cue contra - ipsi level above restw (ALM, tjM1, SSp-m, SSp-n; all trials and far-contra trials).
  B5 imaging control: session F0 asymmetry (balimg_primary_sessions.csv f0_470 = mean log2 F0 right / left, 5 pool
     areas) vs the offset; the offset with the within-animal F0 slope partialled.
  B6 vs the primary post-lick balance marker (recprim_sessions.csv `marker`, 3 panels): within-animal r + MixedLM.
  B7 target or execution: offset = (decoded - pre mean decoded at the SAME SPOUT) + execution term (pre decoded at the
     spout - the like-for-like reference at the actual angle); a decoder that reads the target gives a + offset whenever
     the executed angle shifts ipsilesionally at an unchanged target.
Outputs (il.OUT_W): enl_hist_cov.csv.gz (per-trial history + face covariates, the extraction cache), enl_hist_test.csv,
enl_hist_areas.csv, enl_offx_trials.csv.gz, enl_offx_sessions.csv, enl_offx_track.csv, enl_offx_spout.csv,
enl_offx_terms.csv, enl_offx_pairs.csv, enl_offx_variants.csv, enl_offx_links.csv, enl_offx_spout_shift.csv (B7); figures/lick_enl_history_test.png,
figures/lick_enl_offset_explore.png.
Usage: python scripts/lick_enl_offset_explore.py [--history] [--offset] [--nshuf 200] [--jobs 8] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_enl_angle_decode as ead  # noqa: E402
import likeforlike as lfl  # noqa: E402

OUT = il.OUT_W
FIG = OUT / "figures"
COV = OUT / "enl_hist_cov.csv.gz"
AN = ead.AN
EP = ead.EP
W = 1.0
LF = ead.LF_PRIMARY            # "slide" since 11-05 (was "2s")
SEED = 0
FR_LAGS = np.round(np.arange(-2.0, -0.0999, 0.04), 3)
FR_NAMES = list(il.FR_KEEP)
LAT = {"ipsi": -1.0, "mid": 0.0, "contra": 1.0}
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
KEY = ["ALM", "tjM1", "SSp-m", "SSp-n"]


# ------------------------------------------------------------------------------------------------ EXTRACT (history + FR)
def history_session(item: tuple) -> dict:
    """One session (module-level for spawn), item = (label, cue times of its trial-table rows in file order): per row the
    previous DAQ trial's spout / hit / engaged / gap / row, and the FaceRhythm factors averaged over the pre-cue windows.
    Rows are matched to the DAQ trial table by CUE TIME (<= 0.02 s; the trial table stores 6 significant digits), not by
    re-deriving the first-lick rows: the behavior lick table has changed since the extraction for a few sessions
    (PS92_0818: 351 vs 362 first-lick rows now), so a re-derivation would not line up."""
    label, cue = item[0], np.asarray(item[1], float)
    os.environ.setdefault("COMBINED_ENCODING_FR", "master_pre_v1")          # the frozen encoder's FR set (frozenenv.sh)
    os.environ.setdefault("COMBINED_ENCODING_FR_GROUPS", "master_v1:A")
    from wfield_local import config
    B = il._behaviour(label)
    Tt = B.trials.reset_index(drop=True)
    tc = Tt.cue_s.to_numpy(float)
    stop = Tt.stop_s.to_numpy(float)
    stop = np.where(np.isfinite(stop), stop, tc + 3.5)
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    hit = np.where(np.isfinite(tc), np.searchsorted(bo, stop, "right") - np.searchsorted(bo, tc) > 0, False)
    tcf = np.where(np.isfinite(tc), tc, -1e9)
    j = np.abs(tcf[None, :] - cue[:, None]).argmin(1) if len(cue) else np.zeros(0, int)
    if len(cue) and np.abs(tcf[j] - cue).max() > 0.02:
        raise RuntimeError(f"{label}: first-lick cue not found in the trial table")
    cue_pos = Tt.pos.astype(str).to_numpy()[j]
    pj = j - 1
    has = (pj >= 0) & np.isfinite(tc[np.maximum(pj, 0)])
    pp = np.where(has, Tt.pos.astype(str).to_numpy()[np.maximum(pj, 0)], "none")
    row_of = {int(jj): i for i, jj in enumerate(j)}
    prow = np.array([row_of.get(int(p), -1) if h else -1 for p, h in zip(pj, has)], int)
    # FaceRhythm, unblanked (il.fr_unblanked: 25 Hz inside the trial windows), pre-cue
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    F = np.full((len(cue), len(FR_LAGS), len(FR_NAMES)), np.nan)
    try:
        tt = (cue[:, None] + FR_LAGS[None, :]).ravel()
        F = il.fr_unblanked(an, date, tt).reshape(len(cue), len(FR_LAGS), len(FR_NAMES))
    except FileNotFoundError:
        pass
    m1, m2 = FR_LAGS >= -1.0 - 1e-9, FR_LAGS < -1.0 - 1e-9
    with np.errstate(all="ignore"):
        fr1, fr2 = np.nanmean(F[:, m1], 1), np.nanmean(F[:, m2], 1)
    return {"label": label, "cue": tc[j], "cue_pos": cue_pos, "prev_pos": pp, "prev_hit": np.where(has, hit[np.maximum(pj, 0)], False),
            "prev_eng": np.where(has, Tt.engaged.to_numpy(bool)[np.maximum(pj, 0)], False),
            "prev_gap": np.where(has, cue - tc[np.maximum(pj, 0)], np.nan), "prev_row": prow, "has_prev": has,
            "fr1": fr1, "fr2": fr2}


def extract_cov(jobs: int) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(COV)
    T, _n = ead.prepare(pd.read_csv(ead.TRIALS))
    items = [(lab, tuple(T.loc[T.label == lab, "cue"].to_numpy(float))) for lab in sorted(T.label.unique())]
    res, fails = ak.fan_sessions(items, history_session, jobs=jobs, key=lambda it: it[0])
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    rows = []
    for (lab, _c), v in res:
        n = int((T.label == lab).sum())
        bad = (v["cue_pos"] != T.loc[T.label == lab, "pos"].to_numpy(str)).sum()
        if bad:
            raise RuntimeError(f"{lab}: {bad} rows with a different spout in the DAQ trial table")
        if len(v["cue"]) != n:
            raise RuntimeError(f"{lab}: {len(v['cue'])} first-lick rows vs {n} in the trial table")
        tc = T.loc[T.label == lab, "cue"].to_numpy(float)
        if np.abs(tc - v["cue"]).max() > 0.01:
            raise RuntimeError(f"{lab}: cue times differ from the trial table")
        D = pd.DataFrame({"label": lab, "row": np.arange(n), "prev_pos": v["prev_pos"], "prev_hit": v["prev_hit"],
                          "prev_eng": v["prev_eng"], "prev_gap": v["prev_gap"], "prev_row": v["prev_row"],
                          "has_prev": v["has_prev"]})
        for k, nm in enumerate(FR_NAMES):
            D[f"{nm}_w1"] = v["fr1"][:, k]
            D[f"{nm}_w2"] = v["fr2"][:, k]
        rows.append(D)
    C = pd.concat(rows, ignore_index=True)
    C.to_csv(COV, index=False, float_format="%.6g")
    print(f"== {len(C)} trials; previous trial found {C.has_prev.mean():.3f}; previous is a first-lick row "
          f"{(C.prev_row >= 0).mean():.3f}; FaceRhythm finite {C[f'{FR_NAMES[0]}_w1'].notna().mean():.3f}", flush=True)
    return C


def with_history(T: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """Trial table (file order) + history / FR columns + prev_aw (previous first-lick row's within session x spout angle
    deviation from ALL matched trials) + time in session."""
    T = T.copy()
    T["row"] = T.groupby("label").cumcount()
    T = T.merge(C, on=["label", "row"], how="left", validate="one_to_one")
    ok = np.isfinite(T.angle)
    cell0 = (T.label + "|" + T.pos).to_numpy(str)
    aw = np.full(len(T), np.nan)
    aw[ok] = ead._resid(T.angle[ok].to_numpy(float), cell0[ok])
    T["aw_all"] = aw
    key = T.set_index(["label", "row"]).aw_all
    T["prev_aw"] = [key.get((lab, int(r)), np.nan) if r >= 0 else np.nan for lab, r in zip(T.label, T.prev_row)]
    T["tz"] = T.groupby("label").cue.transform(lambda c: (c - c.mean()) / c.std())
    return T


def history_matrix(D: pd.DataFrame) -> tuple[np.ndarray, list]:
    """The trial-history design (module docstring)."""
    cur = D.pos.to_numpy(str)
    prv = D.prev_pos.fillna("none").to_numpy(str)
    cols, X = [], []
    for c in POS_ORDER:
        for p in (*POS_ORDER, "none"):
            v = ((cur == c) & (prv == p)).astype(float)
            if v.any():
                cols.append(f"prev={p}|cur={c}"); X.append(v)

    def lat(s):
        return np.array([LAT.get(x.split("_")[-1], np.nan) for x in s])

    def far(s):
        return np.array([1.0 if x.startswith("far") else 0.0 if x.startswith("close") else np.nan for x in s])
    hp = prv != "none"
    X += [np.where(hp, lat(prv) - lat(cur), 0), np.where(hp, far(prv) - far(cur), 0), (prv == cur).astype(float),
          D.prev_hit.fillna(False).astype(float).to_numpy(), D.prev_eng.fillna(False).astype(float).to_numpy()]
    cols += ["d_lat", "d_far", "same", "prev_hit", "prev_eng"]
    pa = D.prev_aw.to_numpy(float)
    X += [np.nan_to_num(pa), (~np.isfinite(pa)).astype(float)]
    cols += ["prev_aw", "prev_aw_missing"]
    g = np.log(np.clip(D.prev_gap.to_numpy(float), 1, None))
    X += [np.where(np.isfinite(g), g, np.nanmedian(g)), D.tz.to_numpy(float), D.tz.to_numpy(float) ** 2]
    cols += ["log_gap", "tz", "tz2"]
    return np.column_stack(X), cols


# ------------------------------------------------------------------------------------------------ TASK A
def _loso_pred(X: np.ndarray, y: np.ndarray, sess: np.ndarray) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for s in np.unique(sess):
        te = sess == s
        p[te] = ead._pred(ead._fit(X[~te], y[~te]), X[te])
    return p


def _r2(p, y) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def _hist_folds(H, y, sess) -> tuple[np.ndarray, list]:
    """Stage 1 per held-out session: the history ridge fitted on the training sessions -> (held-out history prediction,
    [(test mask, training residual y - in-sample history fit)])."""
    ph, folds = np.full(len(y), np.nan), []
    for s in np.unique(sess):
        te = sess == s
        m = ead._fit(H[~te], y[~te])
        ph[te] = ead._pred(m, H[te])
        folds.append((te, y[~te] - ead._pred(m, H[~te])))
    return ph, folds


def _add_stage(X, ph, folds) -> np.ndarray:
    """Stage 2: a ridge on X fitted to the training sessions' history residual; full prediction = history + it."""
    pf = ph.copy()
    for te, res in folds:
        pf[te] += ead._pred(ead._fit(X[~te], res), X[te])
    return pf


def _perm_rows(cell, rng) -> np.ndarray:
    idx = np.arange(len(cell))
    for q in np.unique(cell):
        i = np.flatnonzero(cell == q)
        idx[i] = i[rng.permutation(len(i))]
    return idx


def _nested(H, X, y, sess, cell, nshuf, rng) -> dict:
    """hist vs hist + X (all within-cell residualised by the caller), held-out. PRIMARY = two-stage (X fitted to the
    history residual inside each training fold, so X and history keep their own ridge penalties); joint = one ridge on
    [history, X] (one shared penalty; secondary, no null). Null: X rows permuted within session x spout, stage 2 refit."""
    ph, folds = _hist_folds(H, y, sess)
    pf = _add_stage(X, ph, folds)
    pj = _loso_pred(np.column_stack([H, X]), y, sess)
    R2h = _r2(ph, y)
    out = {"r_hist": ead._r(ph, y), "r_full": ead._r(pf, y), "R2_hist": R2h, "R2_full": _r2(pf, y),
           "dR2": _r2(pf, y) - R2h, "r_add": ead._r(pf - ph, y - ph), "dR2_joint": _r2(pj, y) - R2h}
    null = np.array([_r2(_add_stage(X[_perm_rows(cell, rng)], ph, folds), y) - R2h for _ in range(nshuf)])
    if nshuf:
        out.update(dR2_null95=float(np.percentile(null, 95)), p=float((1 + np.sum(null >= out["dR2"])) / (1 + nshuf)))
    return out


def task_a(T: pd.DataFrame, names: list, nshuf: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy import stats
    frc = [f"{n}_{w}" for n in FR_NAMES for w in ("w1", "w2")]
    rows, arows = [], []
    rng = np.random.default_rng(SEED)
    for lf in ead.LF_RULES:
        fc = ead.fcols(W, lf, names)
        S = ead.select(T, W, lf)
        for a in AN:
            Pr = S[(S.animal == a) & (S.epoch == "pre")].reset_index(drop=True)
            sess = Pr.label.to_numpy(str)
            cell = np.char.add(np.char.add(sess, "|"), Pr.pos.to_numpy(str))
            y = ead._resid(Pr.angle.to_numpy(float), cell)
            H, _hc = history_matrix(Pr)
            H = ead._resid(H, cell)
            X = ead._resid(Pr[fc].to_numpy(float), cell)
            base = {"animal": a, "lickfree": lf, "n": len(y), "n_sessions": len(np.unique(sess)),
                    "frac_prev_aw": float(np.isfinite(Pr.prev_aw).mean())}
            r_enl = ead._r(_loso_pred(X, y, sess), y)
            rows.append({**base, "model": "A1 history -> + ENL", "r_enl_only": r_enl,
                         **_nested(H, X, y, sess, cell, nshuf, rng)})
            print(rows[-1], flush=True)
            # A2: + pre-cue face movement, on the trials with FaceRhythm
            fm = Pr[frc].notna().all(1).to_numpy()
            if fm.sum() > 200 and len(np.unique(sess[fm])) >= 3:
                c2, s2, y2 = cell[fm], sess[fm], ead._resid(Pr.angle.to_numpy(float)[fm], cell[fm])
                H2 = ead._resid(history_matrix(Pr[fm].reset_index(drop=True))[0], c2)
                F2 = ead._resid(Pr.loc[fm, frc].to_numpy(float), c2)
                X2 = ead._resid(Pr.loc[fm, fc].to_numpy(float), c2)
                b2 = {**base, "n": int(fm.sum()), "n_sessions": len(np.unique(s2))}
                if not fm.all():                         # A1 on the face subset (the same trials when FR covers all)
                    rows.append({**b2, "model": "A2 same subset: history -> + ENL",
                                 **_nested(H2, X2, y2, s2, c2, nshuf, rng)})
                rows.append({**b2, "model": "A2 history -> + face", **_nested(H2, F2, y2, s2, c2, nshuf, rng)})
                HF = np.column_stack([H2, F2])
                rows.append({**b2, "model": "A2 history + face -> + ENL", **_nested(HF, X2, y2, s2, c2, nshuf, rng)})
                print(rows[-1], flush=True)
            # A3 (primary lick-free rule only): per area pair, drop-one and single-area
            if lf == LF:
                ph, folds = _hist_folds(H, y, sess)
                pf = _add_stage(X, ph, folds)
                R2h, R2f = _r2(ph, y), _r2(pf, y)
                for ar in sorted({n.rsplit("_", 1)[0] for n in names}):
                    k = np.array([n.rsplit("_", 1)[0] == ar for n in names])
                    pd_ = _add_stage(X[:, ~k], ph, folds)
                    ps_ = _add_stage(X[:, k], ph, folds)
                    arows.append({"animal": a, "lickfree": lf, "area": ar, "n_features": int(k.sum()),
                                  "dR2_drop": R2f - _r2(pd_, y), "dR2_single": _r2(ps_, y) - R2h,
                                  "r_add_single": ead._r(ps_ - ph, y - ph), "dR2_full": R2f - R2h})
                print(f"[A3 {a}] done", flush=True)
    R = pd.DataFrame(rows)
    comb = []
    for (lf, mdl), g in R.groupby(["lickfree", "model"], sort=False):
        if "p" in g and g.p.notna().all():
            chi = -2 * np.log(g.p.to_numpy(float)).sum()
            comb.append({"animal": "Fisher", "lickfree": lf, "model": mdl, "n": int(g.n.sum()), "dR2": float(g.dR2.mean()),
                         "p": float(stats.chi2.sf(chi, 2 * len(g))), "n_animals_p05": int((g.p <= 0.05).sum()),
                         "n_animals_pos": int((g.dR2 > 0).sum())})
    R = pd.concat([R, pd.DataFrame(comb)], ignore_index=True)
    R.to_csv(OUT / "enl_hist_test.csv", index=False)
    A = pd.DataFrame(arows)
    A.to_csv(OUT / "enl_hist_areas.csv", index=False)
    return R, A


# ------------------------------------------------------------------------------------------------ TASK B
def _within(D: pd.DataFrame, x: str, y: str, post_only: bool = True, logday: bool = True) -> dict:
    """per-animal r of x with y + MixedLM y ~ w + b (+ logday), groups = animal (lick_sspn_checks._track's model)."""
    import statsmodels.formula.api as smf
    g = D.dropna(subset=[x, y]).copy()
    if post_only:
        g = g[(g.epoch != "pre") & (g.day > 0)]
    rr = {a: float(np.corrcoef(h[x], h[y])[0, 1]) if len(h) > 2 else np.nan for a, h in g.groupby("animal")}
    g["w"] = g[x] - g.groupby("animal")[x].transform("mean")
    g["b"] = g.groupby("animal")[x].transform("mean")
    g["yy"] = g[y]
    f = "yy ~ w + b" + (" + np.log(day)" if logday and post_only else "")
    try:
        m = smf.mixedlm(f, g, groups=g["animal"]).fit(reml=False)
        slope, p = float(m.params["w"]), float(m.pvalues["w"])
    except Exception:                                          # noqa: BLE001
        slope, p = np.nan, np.nan
    return {"x": x, "y": y, "post_only": post_only, "n": len(g), "slope_within": slope, "p_within": p,
            "n_pos": int(sum(v > 0 for v in rr.values() if np.isfinite(v))),
            **{f"r_{a}": rr.get(a, np.nan) for a in AN}}


def _epoch_change(SS: pd.DataFrame, val: str, tag: str, rng) -> list:
    """Per animal (rule 8) epoch - pre of the session means + paired nested bootstrap (analysis_kit.boot_delta)."""
    from wfield_local import analysis_kit as ak
    out = []
    pre = {a: h[val].dropna().to_numpy() for a, h in SS[SS.epoch == "pre"].groupby("animal")}
    for ep in EP[1:]:
        post = {a: h[val].dropna().to_numpy() for a, h in SS[SS.epoch == ep].groupby("animal")}
        post = {a: v for a, v in post.items() if len(v)}
        pre_ = {a: v for a, v in pre.items() if len(v)}
        ci, dr = ak.boot_delta(post, pre_, rng, draws=True)
        per = {a: float(np.mean(post[a]) - np.mean(pre_[a])) for a in sorted(set(post) & set(pre_))}
        out.append({"variant": tag, "epoch": ep, "delta": np.nan if ci is None else ci.point,
                    "lo": np.nan if ci is None else ci.lo, "hi": np.nan if ci is None else ci.hi,
                    "p": ak.boot_p(dr) if ci is not None else np.nan, "n_pos": int(sum(v > 0 for v in per.values())),
                    **{f"d_{a}": per.get(a, np.nan) for a in AN}})
    return out


def offsets(S: pd.DataFrame, fc: list, variant: str = "") -> tuple[pd.DataFrame, dict]:
    """Per-trial predictions (pre = frozen LOSO, post = frozen all-pre) + like-for-like offset, as run_variant."""
    rows, fz = [], {}
    for a in AN:
        A = S[S.animal == a].reset_index(drop=True)
        Pr = A[A.epoch == "pre"].reset_index(drop=True)
        X, y = Pr[fc].to_numpy(float), Pr.angle.to_numpy(float)
        sess = Pr.label.to_numpy(str)
        cell = np.char.add(np.char.add(sess, "|"), Pr.pos.to_numpy(str))
        FZ = ead.frozen_fits(a, W, LF, X, y, sess, cell, variant=variant)
        fz[a] = FZ
        P = ead.loso(X, y, sess, cell, Pr.pos.to_numpy(str), fits=FZ["loso"])
        Po = A[A.epoch != "pre"].reset_index(drop=True)
        B = pd.concat([Pr.assign(pred=P["raw"]), Po.assign(pred=ead._pred(FZ["raw"], Po[fc].to_numpy(float)))],
                      ignore_index=True)
        rows.append(B)
    R = pd.concat(rows, ignore_index=True)
    R["set"] = "enl"
    R["offset"], _o = lfl.offset(R, "angle", "pred", lfl.SIGMA_DEG, what=f"ENL angle offset_explore{variant}", method="nw")
    return R, fz


def session_table(R: pd.DataFrame) -> pd.DataFrame:
    return R.groupby(["label", "animal", "epoch"]).agg(n=("offset", "size"), offset=("offset", "mean"),
                                                      offset_n=("offset", lambda s: int(s.notna().sum()))).reset_index()


def task_b(T: pd.DataFrame, names: list) -> dict:
    import lick_offset_contrib_map as ocm
    fc = ead.fcols(W, LF, names)
    rng = np.random.default_rng(SEED)
    S = ead.select(T, W, LF)
    R, fz = offsets(S, fc)
    R[["label", "animal", "epoch", "pos", "cue", "angle", "pred", "offset"]].to_csv(OUT / "enl_offx_trials.csv.gz",
                                                                                     index=False, float_format="%.6g")
    SS = session_table(R)
    SS = SS[SS.n >= ead.MIN_SESS]
    old = pd.read_csv(OUT / "enl_angle_decode_sessions.csv")
    old = old[(old.window_s == W) & (old.lickfree == LF)].set_index("label").offset
    chk = float(np.nanmax(np.abs(SS.set_index("label").offset - old.reindex(SS.label).to_numpy())))
    print(f"== B0: session offsets vs enl_angle_decode_sessions.csv, max |diff| = {chk:.2e}", flush=True)
    rec = pd.read_csv(OUT / "recmark_sessions.csv", usecols=["label", "day", "aim", "iti_ang"]).drop_duplicates("label")
    SS = SS.merge(rec, on="label", how="left")
    # ---- B4: variants (restw-centred, session-mean-centred features), each with its own frozen pre decoder
    sd = il.area_sd().loc[:, names]
    RW = pd.read_csv(OUT / "precue_restw_sessions.csv")
    RW = RW[RW.restw_ok.astype(bool)]
    rw = RW.pivot_table(index="label", columns="area", values="restw")
    miss = [n for n in names if n not in rw.columns]
    if miss:
        raise RuntimeError(f"restw missing areas {miss}")
    Sr = S[S.label.isin(rw.index)].reset_index(drop=True).copy()
    Sr[fc] = Sr[fc].to_numpy(float) - (rw.loc[Sr.label, names].to_numpy(float) / sd.loc[Sr.label, names].to_numpy(float))
    Sm = S.copy()
    pm = Sm.groupby(["label", "pos"])[fc].mean().groupby("label").mean()
    Sm[fc] = Sm[fc].to_numpy(float) - pm.loc[Sm.label, fc].to_numpy(float)
    var_rows, vsess = [], {"base": SS}
    var_rows += _epoch_change(SS, "offset", "base (as published)", rng)
    for tag, D in (("_restw", Sr), ("_smean", Sm)):
        Rv, _f = offsets(D, fc, variant=tag)
        Sv = session_table(Rv)
        Sv = Sv[Sv.n >= ead.MIN_SESS]
        vsess[tag] = Sv
        var_rows += _epoch_change(Sv, "offset", {"_restw": "features - session restw",
                                                  "_smean": "features - session mean (equal-weight spouts)"}[tag], rng)
        SS = SS.merge(Sv[["label", "offset"]].rename(columns={"offset": f"offset{tag}"}), on="label", how="left")
    SSb = SS[SS.label.isin(Sr.label.unique())]
    var_rows += _epoch_change(SSb, "offset", "base, restw sessions only", rng)
    # restw-shift-only prediction: beta . (restw / sd - pre mean of restw / sd)
    shift = {}
    for a in AN:
        beta = fz[a]["raw"][1].coef_ / fz[a]["raw"][0].scale_
        labs = sorted(set(S.label[S.animal == a]) & set(rw.index))
        v = {x: rw.loc[x, names].to_numpy(float) / sd.loc[x, names].to_numpy(float) for x in labs}
        pre = [x for x in labs if x in set(S.label[(S.animal == a) & (S.epoch == "pre")])]
        mu = np.mean([v[x] for x in pre], 0)
        shift.update({x: float(beta @ (mu - v[x])) for x in labs})     # features minus restw: + restw reads as - offset
    SS["restw_term"] = -SS.label.map(shift)                            # what the raw restw level adds to the offset
    var_rows += _epoch_change(SS, "restw_term", "restw level x beta (part of the offset from the session restw)", rng)
    # pre-cue contra - ipsi level above restw (KEY areas), all trials and far-contra trials
    for fn, col in (("precue_restw_sessions.csv", "pc_ci"), ("precue_restw_sessions_farcontra.csv", "pc_ci_fc")):
        Q = pd.read_csv(OUT / fn)
        Q = Q[Q.restw_ok.astype(bool)]
        Q["base"] = Q.area.str.rsplit("_", n=1).str[0]
        Q["hemi"] = Q.area.str.rsplit("_", n=1).str[1]
        Q = Q[Q.base.isin(KEY)].pivot_table(index="label", columns="hemi", values="precue", aggfunc="mean")
        SS[col] = SS.label.map(Q["right"] - Q["left"])
    # ---- B5 F0
    F0 = pd.read_csv(OUT / "balimg_primary_sessions.csv", usecols=["label", "f0_470", "f0_415"]).drop_duplicates("label")
    SS = SS.merge(F0, on="label", how="left")
    SS["offset_f0part"] = np.nan
    for _a, g in SS.dropna(subset=["f0_470", "offset"]).groupby("animal"):
        b = np.polyfit(g.f0_470, g.offset, 1)[0]
        SS.loc[g.index, "offset_f0part"] = g.offset - b * (g.f0_470 - g.f0_470.mean())
    var_rows += _epoch_change(SS, "offset_f0part", "offset, within-animal F0 slope partialled", rng)
    var_rows += _epoch_change(SS, "f0_470", "F0 asymmetry log2 R/L (470)", rng)
    # ---- B6 primary balance marker
    M = pd.read_csv(OUT / "recprim_sessions.csv", usecols=["panel", "label", "marker"])
    for pn, g in M.groupby("panel"):
        k = "marker:" + pn.split(" ->")[0].replace(" ", "_") + ":" + pn.split("-> ")[-1]
        SS = SS.merge(g[["label", "marker"]].rename(columns={"marker": k}), on="label", how="left")
    SS.to_csv(OUT / "enl_offx_sessions.csv", index=False)
    V = pd.DataFrame(var_rows)
    V.to_csv(OUT / "enl_offx_variants.csv", index=False)
    # ---- B1 / B4c / B5 / B6 links
    import lick_sspn_checks as ssc
    S2 = pd.concat([SS.assign(set="iti"), SS.assign(set="trial")], ignore_index=True).drop(columns=["day", "aim", "iti_ang"])
    trk = ssc._track(S2, ["offset", "offset_restw", "offset_smean", "offset_f0part"], "ENL offset")
    trk = trk.drop_duplicates(["y", "measure"]).drop(columns="set")
    trk.to_csv(OUT / "enl_offx_track.csv", index=False)
    links = []
    for x in ("pc_ci", "pc_ci_fc", "f0_470", "restw_term"):
        links.append(_within(SS, x, "offset", post_only=False, logday=False))
        links.append(_within(SS, x, "offset", post_only=True))
    for k in [c for c in SS.columns if c.startswith("marker:")]:
        links.append(_within(SS, "offset", k, post_only=True))
        links.append(_within(SS, "offset", k, post_only=False, logday=False))
    L = pd.DataFrame(links)
    L.to_csv(OUT / "enl_offx_links.csv", index=False)
    # ---- B2 by spout
    sp = R.groupby(["label", "animal", "epoch", "pos"]).offset.agg(["mean", "count"]).reset_index()
    sp = sp[sp["count"] >= 3]
    sprows = []
    for pos in POS_ORDER:
        g = sp[sp.pos == pos].rename(columns={"mean": "offset"})
        for r_ in _epoch_change(g, "offset", pos, rng):
            sprows.append({**r_, "pos": pos, "n_trials_acute": int(g[g.epoch == "acute"]["count"].sum())})
    SP = pd.DataFrame(sprows)
    SP.to_csv(OUT / "enl_offx_spout.csv", index=False)
    # ---- B7 TARGET or EXECUTION? (added: a decoder that mostly reads the target spout gives a + like-for-like offset
    # whenever the executed angle shifts ipsilesionally at an unchanged target.) Per trial: tgt_off = decoded - the
    # animal's mean pre (LOSO) decoded value at the SAME SPOUT (target-matched, no angle matching); ang_shift = actual
    # angle - the pre mean angle at that spout (the behavioural aim change).
    pre = R[R.epoch == "pre"].groupby(["animal", "pos"])[["pred", "angle"]].mean()
    ix = pd.MultiIndex.from_arrays([R.animal, R.pos])
    R["tgt_off"] = R.pred.to_numpy() - pre.pred.reindex(ix).to_numpy()
    R["ang_shift"] = R.angle.to_numpy() - pre.angle.reindex(ix).to_numpy()
    R["exec_term"] = R.offset - R.tgt_off            # exact: offset = target-matched change + execution term
    S7 = R.groupby(["label", "animal", "epoch"])[["tgt_off", "ang_shift", "exec_term"]].mean().reset_index()
    S7 = S7[S7.label.isin(SS.label)]
    t7 = _epoch_change(S7, "tgt_off", "B7 target-matched decoded change (decoded - pre at same spout)", rng)
    t7 += _epoch_change(S7, "ang_shift", "B7 behavior: actual angle - pre at same spout", rng)
    t7 += _epoch_change(S7, "exec_term", "B7 execution term: offset - target-matched change (trials with an offset)", rng)
    V = pd.concat([V, pd.DataFrame(t7)], ignore_index=True)
    V.to_csv(OUT / "enl_offx_variants.csv", index=False)
    SS = SS.merge(S7[["label", "tgt_off", "ang_shift"]], on="label", how="left")
    SS.to_csv(OUT / "enl_offx_sessions.csv", index=False)
    L = pd.concat([L, pd.DataFrame([_within(SS, "ang_shift", "offset", post_only=False, logday=False),
                                    _within(SS, "ang_shift", "offset", post_only=True)])], ignore_index=True)
    L.to_csv(OUT / "enl_offx_links.csv", index=False)
    # per animal x spout x epoch: like-for-like offset vs behavioural shift (session x spout means, then epoch - pre)
    g7 = R.groupby(["animal", "epoch", "pos"])[["offset", "ang_shift", "tgt_off"]].mean().reset_index()
    g7 = g7.merge(g7[g7.epoch == "pre"][["animal", "pos", "offset"]].rename(columns={"offset": "off_pre"}),
                  on=["animal", "pos"])
    g7["d_offset"] = g7.offset - g7.off_pre
    g7 = g7[g7.epoch != "pre"]
    g7.to_csv(OUT / "enl_offx_spout_shift.csv", index=False)
    # ---- B3 exact decomposition
    Q = R.dropna(subset=["angle", "pred"]).reset_index(drop=True)
    sigma, _cv = lfl.choose(Q, "angle", "pred", lfl.SIGMA_DEG, "nw")
    trows, prow = [], []
    for a in AN:
        FZ = fz[a]
        sc, m = FZ["raw"]
        beta = m.coef_ / sc.scale_
        A = R[R.animal == a]
        Sd = {x: (g.epoch.iloc[0], g[fc].to_numpy(float), g.angle.to_numpy(float)) for x, g in A.groupby("label")}

        def dec(x, X, FZ=FZ):
            s_, m_ = FZ["loso"][x][0] if x in FZ["loso"] else FZ["raw"]
            return m_.predict(s_.transform(X))
        sess = ocm.session_dh(Sd, dec, sigma, nw=True)
        by = {ep: np.mean([v for e, v, _f in sess.values() if e == ep], 0) for ep in EP if any(e == ep for e, _v, _f in sess.values())}
        full = {ep: np.mean([f_ for e, _v, f_ in sess.values() if e == ep]) for ep in by}
        for ep in EP[1:]:
            if ep not in by:
                continue
            dh = by[ep] - by["pre"]
            trows += [{"animal": a, "epoch": ep, "feature": n, "beta": float(b_), "dh": float(d_), "term": float(b_ * d_),
                       "offset": float(beta @ dh), "readout_change": float(full[ep] - full["pre"])}
                      for n, b_, d_ in zip(names, beta, dh)]
            for ar in sorted({n.rsplit("_", 1)[0] for n in names}):
                if f"{ar}_right" not in names or f"{ar}_left" not in names:
                    continue
                ir, il_ = names.index(f"{ar}_right"), names.index(f"{ar}_left")
                prow.append({"animal": a, "epoch": ep, "area": ar,
                             "difference": 0.5 * (beta[ir] - beta[il_]) * (dh[ir] - dh[il_]),
                             "shared": 0.5 * (beta[ir] + beta[il_]) * (dh[ir] + dh[il_]),
                             "dh_diff": float(dh[ir] - dh[il_]), "dh_sum": float(dh[ir] + dh[il_]),
                             "b_diff": float(beta[ir] - beta[il_]), "b_sum": float(beta[ir] + beta[il_])})
    TT, PP = pd.DataFrame(trows), pd.DataFrame(prow)
    TT["sigma"] = sigma
    TT.to_csv(OUT / "enl_offx_terms.csv", index=False)
    PP.to_csv(OUT / "enl_offx_pairs.csv", index=False)
    return {"SS": SS, "V": V, "trk": trk, "L": L, "SP": SP, "TT": TT, "PP": PP, "chk": chk, "G7": g7}


# ------------------------------------------------------------------------------------------------ PRINT + FIGURES
def report_a(R: pd.DataFrame, A: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    c = ["animal", "lickfree", "model", "n", "n_sessions", "r_hist", "r_full", "R2_hist", "R2_full", "dR2", "dR2_null95",
         "r_add", "p", "n_animals_p05"]
    print("== TASK A (per animal, then Fisher)\n" + R[[x for x in c if x in R]].round(4).to_string(index=False))
    if len(A):
        P = A.pivot_table(index="area", columns="animal", values="dR2_drop")
        P["mean"] = P.mean(1)
        P["n_pos"] = (P[AN] > 0).sum(1)
        print("== A3 drop-one-area dR2\n" + P.sort_values("mean", ascending=False).round(5).head(12).to_string())
        Q = A.pivot_table(index="area", columns="animal", values="dR2_single")
        Q["mean"] = Q.mean(1)
        Q["n_pos"] = (Q[AN] > 0).sum(1)
        print("== A3 single-area dR2 over history\n" + Q.sort_values("mean", ascending=False).round(5).head(12).to_string())


def figure_a(R: pd.DataFrame, A: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from wfield_local import config
    col = {a: config.animal_color().get(a) for a in AN}
    mods = [m for m in ("A1 history -> + ENL", "A2 same subset: history -> + ENL", "A2 history -> + face",
                        "A2 history + face -> + ENL") if m in set(R.model)]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1.4]})
    g = R[(R.lickfree == LF) & (R.animal != "Fisher")]
    for i, a in enumerate(AN):
        for j, mdl in enumerate(mods):
            r = g[(g.animal == a) & (g.model == mdl)]
            if not len(r):
                continue
            x = j + (i - 1.5) * 0.18
            ax[0].bar(x, r.dR2.iloc[0] * 100, 0.17, color=col[a], label=a if j == 0 else None)
            if np.isfinite(r.dR2_null95.iloc[0]):
                ax[0].plot([x - 0.08, x + 0.08], [r.dR2_null95.iloc[0] * 100] * 2, color="k", lw=1.2)
    fz = R[(R.lickfree == LF) & (R.animal == "Fisher")].set_index("model")
    for j, mdl in enumerate(mods):
        if mdl in fz.index:
            ax[0].text(j, 1.0, f"Fisher p={fz.loc[mdl, 'p']:.2g}\n{int(fz.loc[mdl, 'n_animals_p05'])}/4 p<=.05", ha="center",
                       va="top", transform=ax[0].get_xaxis_transform(), fontsize=7)
    ax[0].set_xticks(range(len(mods)), [m.replace("A1 ", "").replace("A2 ", "").replace(" -> ", "\n-> ") for m in mods],
                     fontsize=7)
    ax[0].axhline(0, color="0.5", lw=0.6)
    ax[0].set_ylabel("held-out delta R2 (%) of the within-spout angle deviation")
    ax[0].set_title("1. does ENL add beyond trial history (and pre-cue face)?\n(black tick = 95th pct, ENL rows permuted "
                    "within session x spout)", fontsize=9)
    lo_, hi_ = ax[0].get_ylim()
    ax[0].set_ylim(lo_, hi_ + 0.2 * (hi_ - lo_))
    for a in AN:
        r = g[(g.animal == a) & (g.model == mods[0])]
        if len(r):
            ax[1].scatter([0, 1], [r.R2_hist.iloc[0] * 100, r.R2_full.iloc[0] * 100], color=col[a])
            ax[1].plot([0, 1], [r.R2_hist.iloc[0] * 100, r.R2_full.iloc[0] * 100], color=col[a], label=a)
    ax[1].set_xticks([0, 1], ["history", "history + ENL"])
    ax[1].set_ylabel("held-out R2 (%)")
    ax[1].axhline(0, color="0.5", lw=0.6)
    ax[1].set_title("2. A1 held-out R2 per animal (pre, leave-one-session-out)", fontsize=9)
    if len(A):
        P = A.pivot_table(index="area", columns="animal", values="dR2_drop") * 100
        S_ = A.pivot_table(index="area", columns="animal", values="dR2_single") * 100
        order = P.mean(1).sort_values(ascending=False).index[:14]
        y = np.arange(len(order))
        ax[2].barh(y + 0.2, P.loc[order].mean(1), 0.4, color="#4C72B0", label="drop pair from full (loss)")
        ax[2].barh(y - 0.2, S_.loc[order].mean(1), 0.4, color="#DD8452", label="pair alone over history (gain)")
        for a in AN:
            ax[2].scatter(P.loc[order, a], y + 0.2, s=8, color=col[a])
            ax[2].scatter(S_.loc[order, a], y - 0.2, s=8, color=col[a], marker="x")
        ax[2].set_yticks(y, order, fontsize=7)
        ax[2].invert_yaxis()
        ax[2].axvline(0, color="0.5", lw=0.6)
        ax[2].set_xlabel("held-out delta R2 (%), bar = mean of animals, dots = animals")
        ax[2].set_title("3. which areas (left + right pair): top 14 by drop-one loss", fontsize=9)
    fig.suptitle("NEURAL / BEHAVIOR: is the within-spout ENL -> first-lick angle signal more than trial history and "
                 "face movement? Pre-stroke, ENL -1..-0.1 s, no lick in the last 2 s; per animal. PROVISIONAL", fontsize=10)
    from matplotlib.lines import Line2D
    hd = {}
    for a_ in (ax[0], ax[2]):
        for h_, l_ in zip(*a_.get_legend_handles_labels()):
            hd.setdefault(l_, h_)
    hd["panel 3: o = drop, x = alone (animals)"] = Line2D([], [], ls="", marker="x", color="0.3")
    fig.legend(list(hd.values()), list(hd.keys()), loc="lower center", ncol=len(hd), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, FIG / "lick_enl_history_test.png", dpi=100)
    plt.close(fig)


def _rs(r) -> str:
    return ", ".join(f"{r['r_' + a]:+.2f}" for a in AN)


def figure_b(SS, V, trk, L, SP, PP, G7=None) -> None:
    import matplotlib.pyplot as plt
    from wfield_local import config
    col = {a: config.animal_color().get(a) for a in AN}
    fig, ax = plt.subplots(2, 4, figsize=(25, 10.5))
    # 1. session offset vs day
    a_ = ax[0, 0]
    for a in AN:
        g = SS[SS.animal == a].sort_values("day")
        a_.plot(g.day, g.offset, "o-", color=col[a], ms=3, lw=0.8, label=a)
    a_.axhline(0, color="0.5", lw=0.6)
    a_.axvline(0, color="k", lw=0.6, ls=":")
    a_.set_xlabel("day from stroke")
    a_.set_ylabel("like-for-like ENL offset (deg, + = contralesional)")
    t = trk[trk.measure == "offset"]
    a_.set_title("1. session offset; within-animal r with recovery:\n" + "; ".join(
        f"{r.y}: {_rs(r)} (p {r.p_within:.2g})" for _k, r in t.iterrows()), fontsize=8)
    # 2. by spout
    a_ = ax[0, 1]
    s = SP[SP.epoch == "acute"].set_index("pos").reindex(POS_ORDER)
    x = np.arange(len(POS_ORDER))
    a_.bar(x, s.delta, color="0.75", yerr=[s.delta - s.lo, s.hi - s.delta], capsize=3)
    for i, a in enumerate(AN):
        a_.scatter(x + (i - 1.5) * 0.08, s[f"d_{a}"], s=14, color=col[a], label=a)
    a_.set_xticks(x, [p.replace("_", "\n") for p in POS_ORDER], fontsize=8)
    a_.axhline(0, color="0.5", lw=0.6)
    a_.set_title("2. acute - pre offset by target spout (deg)\nbar = paired nested bootstrap, dots = animals", fontsize=9)
    # 3. pairs
    a_ = ax[0, 2]
    p = PP[PP.epoch == "acute"].groupby("area")[["difference", "shared"]].mean()
    p = p.loc[(p.difference.abs() + p.shared.abs()).sort_values(ascending=False).index[:12]]
    y = np.arange(len(p))
    a_.barh(y + 0.2, p.difference, 0.4, color="#C44E52", label="left/right DIFFERENCE")
    a_.barh(y - 0.2, p.shared, 0.4, color="#4C72B0", label="SHARED (sum)")
    for a in AN:
        q = PP[(PP.epoch == "acute") & (PP.animal == a)].set_index("area").reindex(p.index)
        a_.scatter(q.difference, y + 0.2, s=8, color=col[a])
        a_.scatter(q.shared, y - 0.2, s=8, color=col[a], marker="x")
    a_.set_yticks(y, p.index, fontsize=8)
    a_.invert_yaxis()
    a_.axvline(0, color="0.5", lw=0.6)
    tot = PP[PP.epoch == "acute"].groupby("animal")[["difference", "shared"]].sum()
    a_.set_title(f"3. acute offset by area pair (deg; sum = offset)\nall pairs: difference {tot.difference.mean():+.2f} "
                 f"[{', '.join(f'{v:+.2f}' for v in tot.difference)}], shared {tot.shared.mean():+.2f}", fontsize=8)
    # 4. variants
    a_ = ax[1, 0]
    v = V[V.epoch == "acute"].reset_index(drop=True)
    a_.barh(np.arange(len(v)), v.delta, xerr=[v.delta - v.lo, v.hi - v.delta], color="0.75", capsize=3)
    for a in AN:
        a_.scatter(v[f"d_{a}"], np.arange(len(v)), s=12, color=col[a])
    a_.set_yticks(range(len(v)), [f"{t_} (p {p_:.2g})" for t_, p_ in zip(v.variant, v.p)], fontsize=7)
    a_.invert_yaxis()
    a_.axvline(0, color="0.5", lw=0.6)
    a_.set_title("4. acute - pre: baseline-shift and imaging controls (deg)", fontsize=9)
    # 5. offset vs precue contra - ipsi (within animal, all sessions)
    a_ = ax[1, 1]
    for a in AN:
        g = SS[SS.animal == a]
        a_.scatter(g.pc_ci - g.pc_ci.mean(), g.offset - g.offset.mean(), s=12, color=col[a],
                   marker="o", label=a)
    lk = L[(L.x == "pc_ci") & (~L.post_only)]
    a_.set_xlabel("pre-cue contra - ipsi level above restw (ALM, tjM1, SSp-m, SSp-n), animal-centered")
    a_.set_ylabel("ENL offset, animal-centered (deg)")
    if len(lk):
        r = lk.iloc[0]
        a_.set_title(f"5. offset vs the pre-cue hemispheric level (all sessions)\nper-animal r "
                     f"{_rs(r)}; within slope p {r.p_within:.2g}", fontsize=8)
    # 6. vs primary marker
    a_ = ax[1, 2]
    mk = [c for c in SS.columns if c.startswith("marker:")]
    k = next((c for c in mk if "cued" in c), mk[0] if mk else None)
    if k:
        for a in AN:
            g = SS[(SS.animal == a) & (SS.epoch != "pre")]
            a_.scatter(g[k], g.offset, s=12, color=col[a], label=a)
        lk = L[(L.x == "offset") & (L.y.str.startswith("marker:")) & (L.post_only)]
        a_.set_xlabel(k)
        a_.set_ylabel("ENL offset (deg)")
        a_.set_title("6. vs the primary post-lick balance marker (post sessions)\n" + "\n".join(
            f"{r.y.split(':')[1][:22]}: r {_rs(r)}; p {r.p_within:.2g}" for _k, r in lk.iterrows()), fontsize=7)
    # 7. target vs execution
    a_ = ax[0, 3]
    v7 = V[V.variant.str.startswith("B7")]
    for k_, (vn, c_) in enumerate((("B7 target", "#4C72B0"), ("B7 behavior", "#DD8452"))):
        q = v7[v7.variant.str.startswith(vn)].set_index("epoch").reindex(EP[1:])
        xx = np.arange(3) + (k_ - 0.5) * 0.35
        a_.bar(xx, q.delta, 0.33, yerr=[q.delta - q.lo, q.hi - q.delta], color=c_, capsize=3,
               label="decoded - pre at same spout" if "target" in vn else "executed angle - pre at same spout")
        for a in AN:
            a_.scatter(xx, q[f"d_{a}"], s=10, color=col[a])
    base = V[V.variant == "base (as published)"].set_index("epoch").reindex(EP[1:])
    a_.plot(np.arange(3), base.delta, "k_", ms=25, mew=2, label="like-for-like offset (published)")
    a_.set_xticks(range(3), EP[1:])
    a_.axhline(0, color="0.5", lw=0.6)
    a_.legend(fontsize=7)
    a_.set_title("7. TARGET or EXECUTION? epoch - pre (deg): decoded angle at the same TARGET\nspout vs executed angle "
                 "at the same spout (bars = paired bootstrap, dots = animals)", fontsize=8)
    # 8. per spout: offset vs behavioural shift (acute)
    a_ = ax[1, 3]
    if G7 is not None and len(G7):
        g = G7[G7.epoch == "acute"]
        for a in AN:
            h = g[g.animal == a]
            a_.scatter(h.ang_shift, h.d_offset, color=col[a], s=18, label=a)
        rr = [np.corrcoef(h.d_offset, h.ang_shift)[0, 1] for _a, h in g.groupby("animal")]
        a_.axhline(0, color="0.5", lw=0.6)
        a_.axvline(0, color="0.5", lw=0.6)
        a_.set_xlabel("acute executed-angle shift at the spout (deg, + = contralesional)")
        a_.set_ylabel("acute - pre like-for-like offset at the spout (deg)")
        a_.set_title("8. per animal x spout, acute: offset vs aim shift\nr per animal "
                     + ", ".join(f"{v:+.2f}" for v in rr), fontsize=8)
    fig.suptitle("NEURAL: the acute contralesional ENL angle offset (frozen all-pre decoder, ENL -1..-0.1 s, no lick in "
                 "the last 2 s): recovery, spout, areas, baseline shift, imaging. + = contralesional. PROVISIONAL", fontsize=10)
    hd = {}
    for a_ in (ax[0, 0], ax[0, 2]):
        for h_, l_ in zip(*a_.get_legend_handles_labels()):
            hd.setdefault(l_, h_)
    fig.legend(list(hd.values()), [k if k in AN else f"panel 3: {k}" for k in hd],
               loc="lower center", ncol=len(hd), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=3.0)
    FIG.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, FIG / "lick_enl_offset_explore.png", dpi=100)
    plt.close(fig)


def report_b(o: dict) -> None:
    pd.set_option("display.width", 250)
    print(f"== B0 max |session offset - published| = {o['chk']:.2e}")
    print("== B1 track\n" + o["trk"].round(3).to_string(index=False))
    print("== B2 by spout\n" + o["SP"].round(3).to_string(index=False))
    P = o["PP"]
    print("== B3 pairs, acute (per animal)\n" + P[P.epoch == "acute"].pivot_table(
        index="area", columns="animal", values=["difference", "shared"]).round(3).to_string())
    print("== B3 totals\n" + P.groupby(["epoch", "animal"])[["difference", "shared"]].sum().round(3).to_string())
    print("== B4/B5 variants\n" + o["V"].round(3).to_string(index=False))
    print("== B4c/B5/B6/B7 links\n" + o["L"].round(3).to_string(index=False))
    T = o["TT"].groupby(["animal", "epoch"])[["offset", "readout_change"]].first()
    print("== B3 check: beta . dh vs like-for-like readout change (exact Gaussian kernel)\n" + T.round(3).to_string())
    if "G7" in o:
        G = o["G7"]
        print("== B7 per animal x spout (epoch - pre like-for-like offset vs behavioral angle shift at that spout)\n"
              + G.round(2).to_string(index=False))
        for ep, g in G.groupby("epoch"):
            rr = {a: np.corrcoef(h.d_offset, h.ang_shift)[0, 1] for a, h in g.groupby("animal")}
            print(f"   {ep}: r(d_offset, ang_shift) across spouts per animal "
                  + ", ".join(f"{a} {v:+.2f}" for a, v in rr.items()))


def main() -> int:
    from wfield_local import writeguard
    writeguard.assert_writable(OUT / "enl_hist_test.csv")
    il.split_only("lick_enl_offset_explore.py")
    jobs = min(8, int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8)
    nshuf = int(sys.argv[sys.argv.index("--nshuf") + 1]) if "--nshuf" in sys.argv else 200
    if "--plot-only" in sys.argv:
        if (OUT / "enl_hist_test.csv").exists():
            R, A = pd.read_csv(OUT / "enl_hist_test.csv"), pd.read_csv(OUT / "enl_hist_areas.csv")
            report_a(R, A)
            figure_a(R, A)
        if (OUT / "enl_offx_sessions.csv").exists():
            o = {k: pd.read_csv(OUT / f"enl_offx_{f}.csv") for k, f in (("SS", "sessions"), ("V", "variants"),
                                                                       ("trk", "track"), ("L", "links"), ("SP", "spout"),
                                                                       ("PP", "pairs"))}
            g7 = OUT / "enl_offx_spout_shift.csv"
            figure_b(o["SS"], o["V"], o["trk"], o["L"], o["SP"], o["PP"], pd.read_csv(g7) if g7.exists() else None)
        return 0
    T, names = ead.prepare(pd.read_csv(ead.TRIALS))
    if "--history" in sys.argv:
        C = extract_cov(jobs) if (not COV.exists() or "--extract" in sys.argv) else pd.read_csv(COV)
        Th = with_history(T, C)
        R, A = task_a(Th, names, nshuf)
        report_a(R, A)
        figure_a(R, A)
    if "--offset" in sys.argv:
        o = task_b(T, names)
        report_b(o)
        figure_b(o["SS"], o["V"], o["trk"], o["L"], o["SP"], o["PP"], o["G7"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
