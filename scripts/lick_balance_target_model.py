"""TARGET vs EXECUTED-ANGLE model of the primary-pool lick response (10-28/29 helper task).

QUESTION. Is the post-lick hemispheric balance (and the frozen lick decoder's like-for-like offset) the sum of a TARGET
(spout location / intended direction) component and an EXECUTED-angle component? Pre-stroke the two coincide; post-stroke
the landing error (landed - target) would expose the target component as an apparent "contralesional offset", so the
angle-referenced balance would be roughly proportional to the landing error (partly a re-description of the behaviour).
Is there a STATE SHIFT beyond it, and does that residual still track recovery?

SIGN CONVENTIONS (every table / figure):
  angle     lick angle = degrees(arcsin(side_k / reach_mm)), + = CONTRALESIONAL (sign forced so the pre far-contra mean
            is > 0)
  c / i     post-lick response pooled over ALM, tjM1, SSp-m, SSp-n, SSs (mean 0.1..0.5 s after the lick minus
            -1.0..-0.5 s), c = CONTRALESIONAL hemisphere ('right' features), i = IPSILESIONAL ('left')
  d         c - i per lick (difference balance); + = contralesional hemisphere relatively stronger
  balance   log2(sum c / sum pred c) - log2(sum i / sum pred i) over a session's licks (the primary gain ratio);
            + = contralesional hemisphere relatively stronger than the reference predicts
  tgt       target angle = the animal's PRE mean angle at the lick's spout (cued: the trial's spout; ITI: the spout
            position during the ITI = the PREVIOUS trial's spout)
  lerr      landing error = angle - tgt (deg); - = landed IPSILESIONAL of where pre licks to that spout landed
  w         target weight: pre fit y = k + a * tgt + b * angle, w = a / (a + b) (0 = executed angle only, 1 = target only)
  DIR       direction units for the balance decomposition: DIR = -1 x balance change, divided by the animal's pre
            across-spout slope of -balance on angle, i.e. degrees of "apparent lick direction"; + = activity looks like a
            MORE CONTRALESIONAL lick
  offset    frozen lateral decoder like-for-like read-out (offset_contrib terms_licks 'readout', mm); + = decoded more
            CONTRALESIONAL than pre licks landing at the same place; lerr_mm = act - pre mean act at the spout (mm)

METHOD (pre = AUGUST pre sessions; per-animal tables first, rule 8):
 1 Pre model per animal x set x response (c, i, d): angle line, angle cubic, target (6 levels), angle + target, mix
   (a * tgt + b * angle); cvR2 leave-one-pre-session-out; w with a session bootstrap CI.
 2 Angle-matched across targets: per animal x set x epoch, y ~ C(5-deg angle bin) + tgt (OLS) -> g = effect of 1 deg of
   target at matched landed angle, w_matched = g / pre angle-only slope; acute far-contra-target licks vs other-target
   licks in the same bins (difference, observed and the pre mix model's prediction).
 3 Every lick predicted from the PRE model (pre licks leave-one-session-out) with reference angle / both / mix; session
   (and session x spout) balance per reference; change vs pre, paired nested bootstrap (analysis_kit.boot_delta).
 4 Recovery tracking of the session balances: per-animal r, within-animal MixedLM (y ~ within + between + log day),
   cvR2 via lick_recovery_markers.models; the recprim marker on the same sessions for comparison.
 5 Frozen decoder offset: pre within-session x spout slope of readout on lerr_mm (k_pre, w_dec = -k_pre); session offset
   minus k_pre * lerr (residual state) by epoch; per animal post-session OLS offset ~ lerr.
 6 ITI first lick after >= 2 s quiet with the spout = the ITI spout position (previous trial's) as 'target', all of the
   above.
 7 (coordinator addition) 13.8-style per-session per-spout decomposition of the BALANCE: dDIR_s = SHIFT + GAIN x_s +
   TRACK dLand_s (x_s = spout's pre angle - pre mean angle, dLand_s = landing change at s vs pre, deg); compared with
   iti_decoded_decomposition (decomposition_tests.csv) and lick_balance_angle_lmm (balance_angle_lmm.csv).
OUTPUTS (<OUT_W>): baltarget_licks.csv.gz, baltarget_premodel.csv, baltarget_weights.csv, baltarget_matched.csv,
baltarget_sessions.csv, baltarget_epochspout.csv, baltarget_boot.csv, baltarget_track.csv, baltarget_cv.csv,
baltarget_decoder.csv, baltarget_decoder_sessions.csv, baltarget_decomp_sessions.csv, baltarget_decomp_tests.csv,
baltarget_compare.csv; figure figures/lick_balance_target_model.png. --plot-only redraws from the CSVs;
--reuse-licks reuses baltarget_licks.csv.gz instead of rebuilding the per-lick table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
import lick_recovery_primary as rp  # noqa: E402

OUT = il.OUT_W
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
SETS = {"trial": "cued first post-cue lick", "iti": "ITI first lick after >=2 s quiet"}
YS = ("c", "i", "d")
MODELS = ("angle", "angle3", "target", "both", "mix")
REFS = ("angle", "both", "mix")
BIN = 5.0
MIN_SP = 8
N_WBOOT = 300


def _wm(lo_, hi_):
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & (L <= hi_ + 1e-9) if hi_ <= 0 else (L >= lo_ - 1e-9) & (L < hi_ - 1e-9)


def _session(label: str) -> pd.DataFrame:
    """Per lick (both sets) the pooled post-lick response per hemisphere (as lick_balance_trial._session)."""
    acts = lbr.cv._acts(label)
    d = il._load(label)
    cols = list(d["cols"])
    base_m = _wm(*il.BASE_RESP)
    win = _wm(0.1, 0.5)
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        ok, mm = m >= 0, np.maximum(m, 0)
        X = d[xk].astype(float)
        R = {"set": st, "label": label, "k": np.arange(len(act))}
        for side, h in (("c", "right"), ("i", "left")):
            tr = np.sum([X[:, :, cols.index(f"{a}_{h}")] for a in rp.POOL], 0)
            v = tr[:, win].mean(1) - tr[:, base_m].mean(1)
            R[side] = np.where(ok, v[mm], np.nan)
        out.append(pd.DataFrame(R))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _is_pre(df: pd.DataFrame) -> pd.Series:
    from wfield_local import config
    return (df.epoch == "pre") & (~df.label.map(config.early_pre))


def licks() -> pd.DataFrame:
    import lick_recovery_markers as lrm
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path())
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(OUT / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    labels = sorted(P.label.unique())
    res, fail = ak.fan_sessions(labels, _session, jobs=8)
    if fail:
        raise RuntimeError(f"response failed: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    gres, gfail = ak.fan_sessions(labels, lrm._gaps, jobs=8)
    if gfail:
        raise RuntimeError(f"gaps failed: {gfail[:5]}")
    P = P.merge(pd.concat([v for _i, v in gres], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    sgn = float(np.sign(P[(P.set == "trial") & _is_pre(P) & (P.pos == "far_contra")].angle.mean()))
    P["angle"] *= sgn
    print(f"angle sign factor applied: {sgn:+.0f} (+ = contralesional)")
    L = P[((P.set == "trial") & (P.order == 0)) | ((P.set == "iti") & (P.gap >= il.GAP_MAIN))]
    L = L.dropna(subset=["angle", "c", "i", "pos"]).copy()
    R = pd.read_csv(OUT / "recmark_sessions.csv", usecols=["label", "day", "aim", "iti_ang"]).drop_duplicates("label")
    L = L.merge(R, on="label", how="left")
    L["d"] = L.c - L.i
    L["pre"] = _is_pre(L)
    keep = ["set", "animal", "label", "epoch", "day", "aim", "iti_ang", "pre", "pos", "order", "k", "act", "angle",
            "c", "i", "d"]
    return L[keep].reset_index(drop=True)


# ------------------------------------------------------------------ 1 + 3: pre models
def _X(df: pd.DataFrame, model: str, tmap: pd.Series) -> np.ndarray:
    a = df.angle.to_numpy(float) / 10
    one = np.ones(len(df))
    if model == "angle":
        return np.c_[one, a]
    if model == "angle3":
        return np.c_[one, a, a ** 2, a ** 3]
    D = (df.pos.to_numpy()[:, None] == np.array(POS[1:])[None, :]).astype(float)
    if model == "target":
        return np.c_[one, D]
    if model == "both":
        return np.c_[one, a, D]
    return np.c_[one, df.pos.map(tmap).to_numpy(float) / 10, a]


def _fit(tr: pd.DataFrame, y: str, model: str):
    tmap = tr.groupby("pos").angle.mean()
    return np.linalg.lstsq(_X(tr, model, tmap), tr[y].to_numpy(float), rcond=None)[0], tmap


def _pred(tr: pd.DataFrame, te: pd.DataFrame, y: str, model: str) -> np.ndarray:
    b, tmap = _fit(tr, y, model)
    return _X(te, model, tmap) @ b


def premodels(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    cvr, wr = [], []
    for (st, an), g in L.groupby(["set", "animal"]):
        pre = g[g.pre].reset_index(drop=True)
        labs = sorted(pre.label.unique())
        for y in YS:
            yy = pre[y].to_numpy(float)
            for model in MODELS:
                pr = np.full(len(pre), np.nan)
                for lab in labs:
                    te = (pre.label == lab).to_numpy()
                    pr[te] = _pred(pre[~te], pre[te], y, model)
                cvr.append({"set": st, "animal": an, "y": y, "model": model, "n_licks": len(pre), "n_sess": len(labs),
                            "cvR2": 1 - np.sum((yy - pr) ** 2) / np.sum((yy - yy.mean()) ** 2)})
            b, _ = _fit(pre, y, "mix")
            bang, _ = _fit(pre, y, "angle")
            bboth, _ = _fit(pre, y, "both")
            ws = []
            for _ in range(N_WBOOT):
                pick = rng.choice(labs, len(labs))
                bb, _t = _fit(pd.concat([pre[pre.label == x] for x in pick]), y, "mix")
                ws.append(bb[1] / (bb[1] + bb[2]))
            wr.append({"set": st, "animal": an, "y": y, "a_target_per10deg": b[1], "b_angle_per10deg": b[2],
                       "w": b[1] / (b[1] + b[2]), "w_lo": np.nanpercentile(ws, 2.5), "w_hi": np.nanpercentile(ws, 97.5),
                       "slope_angle_only_per10deg": bang[1], "slope_within_target_per10deg": bboth[1],
                       "within_over_total": bboth[1] / bang[1], "n_sess": len(labs)})
    return pd.DataFrame(cvr), pd.DataFrame(wr)


def predict_all(L: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (_st, _an), g in L.groupby(["set", "animal"]):
        g = g.copy()
        pre = g[g.pre]
        tmap = pre.groupby("pos").angle.mean()
        g["tgt"] = g.pos.map(tmap)
        g["lerr"] = g.angle - g.tgt
        for ref in REFS:
            for y in YS:
                col = f"p_{ref}_{y}"
                g[col] = _pred(pre, g, y, ref)
                for lab in pre.label.unique():
                    te = (g.label == lab).to_numpy()
                    g.loc[te, col] = _pred(pre[pre.label != lab], g[te], y, ref)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _bal(g: pd.DataFrame, ref: str) -> float:
    o_c, o_i, p_c, p_i = g.c.sum(), g.i.sum(), g[f"p_{ref}_c"].sum(), g[f"p_{ref}_i"].sum()
    if min(o_c, o_i, p_c, p_i) <= 0:
        return np.nan
    return float(np.log2(o_c / p_c) - np.log2(o_i / p_i))


def _sess_row(g: pd.DataFrame) -> dict:
    r = {"n": len(g), "angle": g.angle.mean(), "lerr": g.lerr.mean()}
    for ref in REFS:
        r[f"bal_{ref}"] = _bal(g, ref)
        r[f"dres_{ref}"] = float((g.d - g[f"p_{ref}_d"]).mean())
    return r


def sessions(Lp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    S, E = [], []
    for (st, an, lab, ep), g in Lp.groupby(["set", "animal", "label", "epoch"]):
        if ep == "pre" and not g.pre.iloc[0]:
            continue
        base = {"set": st, "animal": an, "label": lab, "epoch": ep, "day": g.day.iloc[0], "aim": g.aim.iloc[0],
                "iti_ang": g.iti_ang.iloc[0]}
        S.append({**base, "pos": "all", **_sess_row(g)})
        for p, h in g.groupby("pos"):
            if len(h) >= MIN_SP:
                S.append({**base, "pos": p, **_sess_row(h)})
    for (st, an, ep, p), g in Lp[Lp.pre | (Lp.epoch != "pre")].groupby(["set", "animal", "epoch", "pos"]):
        E.append({"set": st, "animal": an, "epoch": ep, "pos": p, **_sess_row(g)})
    return pd.DataFrame(S), pd.DataFrame(E)


def boots(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rows = []
    cols = [f"bal_{r}" for r in REFS] + [f"dres_{r}" for r in REFS] + ["lerr"]
    for (st, p), g in S[S.pos.isin(["all", "far_contra"])].groupby(["set", "pos"]):
        for c in cols:
            pre = {a: h[c].dropna().to_numpy() for a, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
            pre = {a: v for a, v in pre.items() if len(v)}
            for ep in EPS[1:]:
                post = {a: h[c].dropna().to_numpy() for a, h in sorted(g[g.epoch == ep].groupby("animal"))}
                post = {a: v for a, v in post.items() if len(v) and a in pre}
                if not post:
                    continue
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                am = {a: post[a].mean() - pre[a].mean() for a in post}
                rows.append({"set": st, "pos": p, "measure": c, "epoch": ep, "point": ci.point, "lo": ci.lo,
                             "hi": ci.hi, "n_neg": sum(v < 0 for v in am.values()), "n": len(am),
                             **{f"d_{a}": v for a, v in am.items()}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2: angle-matched across targets
def matched(Lp: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (st, an), g in Lp.groupby(["set", "animal"]):
        pre = g[g.pre]
        for y in YS:
            slope = W[(W.set == st) & (W.animal == an) & (W.y == y)].slope_angle_only_per10deg.iloc[0] / 10
            for ep in EPS:
                h = (pre if ep == "pre" else g[g.epoch == ep]).copy()
                h["bin"] = np.floor(h.angle / BIN).astype(int)
                cnt = h.groupby(["bin", "pos"]).size()
                okb = cnt[cnt >= 3].reset_index().groupby("bin").pos.nunique()
                h = h[h.bin.isin(okb[okb >= 2].index)]
                h = h[h.groupby(["bin", "pos"]).angle.transform("size") >= 3]
                r = {"set": st, "animal": an, "y": y, "epoch": ep, "n_matched": len(h), "n_bins": h.bin.nunique()}
                if len(h) >= 30 and h.bin.nunique() >= 2:
                    B = pd.get_dummies(h.bin).to_numpy(float)
                    X = np.c_[B, h.tgt.to_numpy(float), h.angle.to_numpy(float) - h.bin.to_numpy(float) * BIN]
                    b = np.linalg.lstsq(X, h[y].to_numpy(float), rcond=None)[0]
                    r.update(g_target_per_deg=b[-2], w_matched=b[-2] / slope)
                if ep != "pre":                      # far-contra-target licks vs other targets landing in the same bins
                    fc = h[h.pos == "far_contra"]
                    ot = h[h.pos != "far_contra"]
                    diffs, wts, pdf = [], [], []
                    for bn, f in fc.groupby("bin"):
                        o = ot[ot.bin == bn]
                        if len(f) >= 3 and len(o) >= 3:
                            wgt = 1 / (1 / len(f) + 1 / len(o))
                            diffs.append(f[y].mean() - o[y].mean())
                            pdf.append(f[f"p_mix_{y}"].mean() - o[f"p_mix_{y}"].mean())
                            wts.append(wgt)
                    if wts:
                        r.update(fc_minus_other_obs=np.average(diffs, weights=wts),
                                 fc_minus_other_mixpred=np.average(pdf, weights=wts), n_fc=int(fc.bin.isin(
                                     ot.bin).sum()), fc_bins=len(wts))
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4: recovery tracking
def track(S: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf

    import lick_recovery_markers as lrm
    lrm.N_SHUF = 300
    RP = pd.read_csv(OUT / "recprim_sessions.csv")
    rpm = {"trial": RP[RP.panel == "cued first lick -> aim"][["label", "marker"]],
           "iti": RP[RP.panel == "ITI first after quiet -> iti_ang"][["label", "marker"]]}
    preds = ["bal_angle", "bal_both", "bal_mix", "dres_angle", "dres_mix", "lerr", "recprim"]
    T, CV = [], []
    for st, yvs in (("trial", ["aim"]), ("iti", ["iti_ang", "aim"])):
        G0 = S[(S.set == st) & (S.pos == "all")].merge(rpm[st].rename(columns={"marker": "recprim"}), on="label",
                                                         how="left")
        for yv in yvs:
            G = G0[(G0.epoch != "pre") & (G0.day > 0)].dropna(subset=[yv]).copy()
            G["logday"] = np.log(G.day)
            for c in preds:
                g = G.dropna(subset=[c]).copy()
                rr = {a: np.corrcoef(h[c], h[yv])[0, 1] for a, h in g.groupby("animal")}
                g["w"] = g[c] - g.groupby("animal")[c].transform("mean")
                g["b"] = g.groupby("animal")[c].transform("mean")
                g["yy"] = g[yv]
                try:
                    m = smf.mixedlm("yy ~ w + b + logday", g, groups=g["animal"]).fit(reml=False)
                    sl, p = m.fe_params["w"], m.pvalues["w"]
                except Exception:                                          # noqa: BLE001
                    sl, p = np.nan, np.nan
                T.append({"set": st, "y": yv, "measure": c, "n": len(g), "slope_within": sl, "p_within": p,
                          **{f"r_{a}": v for a, v in rr.items()}})
            F = lrm.models(G0[["set", "label", "animal", "epoch", "day"] + preds].assign(aim=G0[yv]))
            CV.append(F["CV"].assign(set=st, y=yv))
    return pd.DataFrame(T), pd.concat(CV, ignore_index=True)


# ------------------------------------------------------------------ 5: frozen decoder offset
def decoder(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    T = pd.read_csv(OUT / "offset_contrib" / "terms_licks_areas_splitr8.csv.gz",
                    usecols=["set", "label", "act", "order", "pos", "readout"])
    T["set"] = T.set.replace({"cued": "trial"})
    key = ["set", "label", "order", "pos", "act"]
    for df in (T,):
        df["act"] = df.act.round(3)
    Lk = L[key[:-1] + ["act", "k", "angle", "epoch", "animal", "day", "aim", "iti_ang", "pre"]].copy()
    Lk["act"] = Lk.act.round(3)
    for df in (T, Lk):
        df["dup"] = df.groupby(key).cumcount()
    D = Lk.merge(T, on=key + ["dup"], how="inner")
    print(f"decoder: {len(D)} of {len(L)} licks matched to offset_contrib readouts")
    rows, srows = [], []
    for (st, an), g in D.groupby(["set", "animal"]):
        g = g.copy()
        pre = g[g.pre]
        g["lerr_mm"] = g.act - g.pos.map(pre.groupby("pos").act.mean())
        pre = g[g.pre]
        xw = pre.lerr_mm - pre.groupby(["label", "pos"]).lerr_mm.transform("mean")
        yw = pre.readout - pre.groupby(["label", "pos"]).readout.transform("mean")
        k = float((xw * yw).sum() / (xw ** 2).sum())
        sg = g.groupby(["label", "epoch"]).agg(day=("day", "first"), aim=("aim", "first"), iti_ang=("iti_ang", "first"),
                                               pre=("pre", "first"), off=("readout", "mean"),
                                               lerr_mm=("lerr_mm", "mean"), n=("readout", "size")).reset_index()
        sg = sg[(sg.epoch != "pre") | sg.pre]
        sg["off_resid"] = sg.off - k * sg.lerr_mm
        sg["set"], sg["animal"] = st, an
        srows.append(sg)
        post = sg[sg.epoch != "pre"]
        b = np.polyfit(post.lerr_mm, post.off, 1) if len(post) > 3 else (np.nan, np.nan)
        r = {"set": st, "animal": an, "k_pre_within": k, "w_dec": -k, "post_slope_off_on_lerr": b[0],
             "post_intercept_off_at_lerr0": b[1], "n_post_sess": len(post)}
        for ep in EPS:
            h = sg[sg.epoch == ep]
            r.update({f"off_{ep}": h.off.mean(), f"lerr_{ep}": h.lerr_mm.mean(), f"resid_{ep}": h.off_resid.mean()})
        rows.append(r)
    Sd = pd.concat(srows, ignore_index=True)
    R = pd.DataFrame(rows)
    brow = []
    for st, g in Sd.groupby("set"):
        for c in ("off", "off_resid", "lerr_mm"):
            pre = {a: h[c].to_numpy() for a, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
            for ep in EPS[1:]:
                post = {a: h[c].to_numpy() for a, h in sorted(g[g.epoch == ep].groupby("animal")) if a in pre}
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                brow.append({"set": st, "animal": "BOOT", "measure": c, "epoch": ep, "point": ci.point, "lo": ci.lo,
                             "hi": ci.hi})
    return pd.concat([R, pd.DataFrame(brow)], ignore_index=True), Sd


# ------------------------------------------------------------------ 7: 13.8-style decomposition of the balance
def decomp(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    rows = []
    for (st, an), g in L.groupby(["set", "animal"]):
        pre = g[g.pre]
        labs = sorted(pre.label.unique())

        def ref(excl, pre=pre):
            q = pre[pre.label != excl] if excl else pre
            return q.groupby("pos").agg(c=("c", "mean"), i=("i", "mean"), ang=("angle", "mean"))
        R0 = ref(None)
        dirb = -np.log2(R0.c.clip(lower=1e-12) / R0.i.clip(lower=1e-12))
        sp = float(np.polyfit(R0.ang, dirb, 1)[0])          # pre across-spout slope of -balance per deg
        x = R0.ang - pre.angle.mean()
        for (lab, ep), h in g.groupby(["label", "epoch"]):
            if ep == "pre" and lab not in labs:
                continue
            Rf = ref(lab if lab in labs else None)
            m = h.groupby("pos").agg(n=("c", "size"), c=("c", "mean"), i=("i", "mean"), ang=("angle", "mean"))
            m = m[(m.n >= MIN_SP) & m.index.isin(Rf.index)]
            ok = (m.c > 0) & (m.i > 0) & (Rf.c.reindex(m.index) > 0) & (Rf.i.reindex(m.index) > 0)
            m = m[ok]
            if len(m) < 4:
                continue
            dbal = np.log2(m.c / Rf.c.reindex(m.index)) - np.log2(m.i / Rf.i.reindex(m.index))
            ddir = -dbal / sp
            dl = m.ang - Rf.ang.reindex(m.index)
            X = np.c_[np.ones(len(m)), x.reindex(m.index), dl]
            c3 = np.linalg.lstsq(X, ddir.to_numpy(), rcond=None)[0]
            c2 = np.linalg.lstsq(X[:, :2], ddir.to_numpy(), rcond=None)[0]
            rows.append({"set": st, "animal": an, "label": lab, "epoch": ep, "n_spouts": len(m), "pre_slope_dir": sp,
                         "shift_dir_deg": c3[0], "gain": c3[1], "track": c3[2], "shift_raw_neg_bal": c3[0] * sp,
                         "shift_notrack_deg": c2[0], "gain_notrack": c2[1], "mean_dDIR_deg": float(ddir.mean()),
                         "mean_dLand_deg": float(dl.mean()),
                         **{f"dDIR_{p}": v for p, v in ddir.items()}, **{f"dLand_{p}": v for p, v in dl.items()}})
    S = pd.DataFrame(rows)
    out = []
    for st, g in S.groupby("set"):
        for col in ("shift_dir_deg", "gain", "track", "shift_notrack_deg", "gain_notrack", "mean_dDIR_deg",
                    "mean_dLand_deg"):
            prd = {a: h[col].dropna().to_numpy() for a, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
            pm = g[g.epoch == "pre"].groupby("animal")[col].mean()
            out.append({"set": st, "measure": col, "epoch": "pre", "level": pm.mean(),
                        **{f"a_{a}": v for a, v in pm.items()}})
            for ep in EPS[1:]:
                post = {a: h[col].dropna().to_numpy() for a, h in sorted(g[g.epoch == ep].groupby("animal")) if a in prd}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = g[g.epoch == ep].groupby("animal")[col].mean()
                out.append({"set": st, "measure": col, "epoch": ep, "level": am.mean(), "point": ci.point,
                            "lo": ci.lo, "hi": ci.hi, "n_pos": int(((am - pm).dropna() > 0).sum()),
                            **{f"a_{a}": v for a, v in am.items()}})
    return S, pd.DataFrame(out)


def compare(W: pd.DataFrame, DT: pd.DataFrame, DecR: pd.DataFrame) -> pd.DataFrame:
    """w (mix model, d) vs 1 - TRACK (13.8 decoder decomposition, pre sessions; this balance decomposition) vs the slope
    ratio post/pre of lick_balance_angle_lmm (resp, orofacial pair)."""
    rows = []
    dt = pd.read_csv(OUT / f"decomposition_tests{lo.SRC_TAG}.csv")
    lm = pd.read_csv(OUT / "balance_angle_lmm.csv")
    for st, nm in (("trial", "cued"), ("iti", "iti")):
        w = W[(W.set == st) & (W.y == "d")]
        r = {"set": st, "w_mix_d_mean": w.w.mean(), **{f"w_d_{a}": v for a, v in zip(w.animal, w.w)},
             "w_dec_mean": DecR[(DecR.set == st) & DecR.animal.isin(AN)].w_dec.mean()}
        for ep in EPS:
            t = dt[(dt.set == st) & (dt.measure == "track") & (dt.epoch == ep)]
            r[f"decoder13.8_1-TRACK_{ep}"] = 1 - t.level.iloc[0] if len(t) else np.nan
            b = DT[(DT.set == st) & (DT.measure == "track") & (DT.epoch == ep)]
            r[f"balance_1-TRACK_{ep}"] = 1 - b.level.iloc[0] if len(b) else np.nan
            if ep != "pre":
                q = lm[(lm.set == nm) & (lm.measure == "resp") & (lm.pair == "orofacial") & (lm.model == ep)]
                if len(q):
                    r[f"angle_lmm_slope_ratio_{ep}"] = 1 + q.slope_change_post.iloc[0] / q.slope_pre.iloc[0]
                    r[f"angle_lmm_shift_same_angle_{ep}"] = q.shift_post_same_angle.iloc[0]
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plot (from CSVs)
def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    col = {"pre": "0.4", **EPOCH_LINE}
    CVt = pd.read_csv(OUT / "baltarget_premodel.csv")
    W = pd.read_csv(OUT / "baltarget_weights.csv")
    M = pd.read_csv(OUT / "baltarget_matched.csv")
    S = pd.read_csv(OUT / "baltarget_sessions.csv")
    Dd = pd.read_csv(OUT / "baltarget_decoder_sessions.csv")
    DS = pd.read_csv(OUT / "baltarget_decomp_sessions.csv")
    fig, axs = plt.subplots(2, 6, figsize=(30, 10))
    for i, st in enumerate(SETS):
        nm = "CUED" if st == "trial" else "ITI"
        ax = axs[i, 0]
        t = CVt[(CVt.set == st) & (CVt.y == "d")]
        for a in AN:
            u = t[t.animal == a].set_index("model").reindex(MODELS)
            ax.plot(range(len(MODELS)), u.cvR2, "-" + ANIMAL_MARKER[a], color="k", ms=5, lw=0.7, label=a)
        ax.set_xticks(range(len(MODELS)), MODELS, fontsize=8)
        ax.set_ylabel("pre cvR2 (leave-one-pre-session-out)")
        ax.set_title(f"{nm} 1: pre model of d = c - i per lick\n(c = contralesional hemi, + = contra stronger)", fontsize=9)
        # legends: one figure legend below the panels (they sat on the data)
        ax = axs[i, 1]
        for j, y in enumerate(YS):
            u = W[(W.set == st) & (W.y == y)]
            for _, r in u.iterrows():
                xx = j + (AN.index(r.animal) - 1.5) * 0.12
                ax.errorbar(xx, r.w, yerr=[[r.w - r.w_lo], [r.w_hi - r.w]], fmt=ANIMAL_MARKER[r.animal], color="k",
                            ms=5, lw=0.8)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axhline(1, color="0.7", lw=0.6, ls=":")
        ax.set_xticks(range(3), ["c (contra hemi)", "i (ipsi hemi)", "d = c - i"], fontsize=8)
        ax.set_ylim(-0.6, 1.6)
        ax.set_ylabel("target weight w = a/(a+b)")
        ax.set_title(f"{nm} 1: y = k + a*target angle + b*landed angle (pre)\nw=0 executed only, w=1 target only; "
                     "95% session bootstrap", fontsize=9)
        ax = axs[i, 2]
        t = M[(M.set == st) & (M.y == "d")]
        for a in AN:
            u = t[t.animal == a].set_index("epoch").reindex(EPS)
            ax.plot(range(4), u.w_matched, "-" + ANIMAL_MARKER[a], color="k", ms=5, lw=0.7)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(range(4), EPS)
        ax.set_ylabel("target effect at matched landed angle /\npre angle slope (d)")
        ax.set_title(f"{nm} 2: does the TARGET matter at the same landed angle?\n(5-deg bins; 0 = no, 1 = as much as "
                     "angle)", fontsize=9)
        ax = axs[i, 3]
        g = S[(S.set == st) & (S.pos == "all")]
        for k, ref in enumerate(REFS):
            lv = g.groupby(["epoch", "animal"])[f"bal_{ref}"].mean()
            for a in AN:
                v = [lv.get((ep, a), np.nan) for ep in EPS]
                ax.plot(np.arange(4) + (k - 1) * 0.2, v, ANIMAL_MARKER[a], color=["#C44E52", "#4C72B0", "#55A868"][k],
                        ms=5, alpha=0.8, label=f"ref = {ref}" if a == "PS92" else None)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(range(4), EPS)
        ax.set_ylabel("session balance log2 gain (c - i)\n+ = contralesional hemi stronger than ref")
        ax.set_title(f"{nm} 3: balance vs PRE prediction; red = angle only (primary-like),\nblue = angle + target "
                     "levels, green = a*target + b*angle", fontsize=9)
        # (legend below the figure)
        ax = axs[i, 4]
        h = Dd[Dd.set == st]
        for _, r in h.iterrows():
            ax.plot(r.lerr_mm, r.off, ANIMAL_MARKER[r.animal], color=col[r.epoch], ms=4, mec="k", mew=0.2)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("session mean landing error (mm; - = landed ipsilesional of target)")
        ax.set_ylabel("decoder like-for-like offset (mm; + = read contralesional)")
        ax.set_title(f"{nm} 5: frozen decoder offset vs landing error, sessions", fontsize=9)
        ax = axs[i, 5]
        h = DS[DS.set == st]
        for k, term in enumerate(("shift_dir_deg", "track")):
            lv = h.groupby(["epoch", "animal"])[term].mean()
            for a in AN:
                v = [lv.get((ep, a), np.nan) for ep in EPS]
                ax.plot(np.arange(4) + (k - 0.5) * 0.25, np.array(v) * (1 if k == 0 else 10), ANIMAL_MARKER[a],
                        color=["#8172B2", "#CCB974"][k], ms=5, label=("SHIFT (deg)" if k == 0 else "TRACK x 10")
                        if a == "PS92" else None)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(range(4), EPS)
        ax.set_title(f"{nm} 7: 13.8-style balance decomposition per session\nover spouts; dDIR = SHIFT + GAIN x + "
                     "TRACK dLand;\nDIR = -balance in deg, + = looks more contralesional", fontsize=9)
        # (legend below the figure)
    for ax in axs.ravel():
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("TARGET vs EXECUTED angle in the primary-pool lick response (ALM, tjM1, SSp-m, SSp-n, SSs; 0.1-0.5 s after "
                 "the lick minus -1..-0.5 s). Angle + = CONTRALESIONAL; c = contralesional hemisphere, i = ipsilesional; "
                 "balance + = contralesional hemisphere stronger; pre = August pre sessions. Symbols = animals. "
                 "PROVISIONAL", fontsize=10)
    from matplotlib.lines import Line2D
    hd = ([Line2D([], [], marker=ANIMAL_MARKER[a], ls="", color="k", ms=6) for a in AN]
          + [Line2D([], [], marker="o", ls="", color=c, ms=6) for c in ("#C44E52", "#4C72B0", "#55A868")]
          + [Line2D([], [], marker="o", ls="", color=c, ms=6) for c in ("#8172B2", "#CCB974")]
          + [Line2D([], [], marker="o", ls="", color=col[ep], ms=6) for ep in EPS])
    lb = (list(AN) + [f"panel 3: ref = {r}" for r in REFS] + ["panel 7: SHIFT (deg)", "panel 7: TRACK x 10"]
          + [f"panel 5: {ep}" for ep in EPS])
    fig.legend(hd, lb, loc="lower center", ncol=len(lb), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "lick_balance_target_model.png", dpi=85)
    plt.close(fig)


def main() -> int:
    if "--plot-only" in sys.argv:                 # redraw (and re-derive the comparison table) from the saved CSVs
        compare(pd.read_csv(OUT / "baltarget_weights.csv"), pd.read_csv(OUT / "baltarget_decomp_tests.csv"),
                pd.read_csv(OUT / "baltarget_decoder.csv")).to_csv(OUT / "baltarget_compare.csv", index=False)
        plot()
        return 0
    pd.set_option("display.width", 250)
    f = OUT / "baltarget_licks.csv.gz"
    if "--reuse-licks" in sys.argv and f.exists():
        L = pd.read_csv(f)
    else:
        L = licks()
        L.to_csv(f, index=False)
    print(L.groupby(["set", "epoch"]).size().unstack().to_string())
    print("== pre mean angle by spout (deg, + = contralesional)\n" + L[L.pre].pivot_table(
        index=["set", "animal"], columns="pos", values="angle").reindex(columns=POS).round(1).to_string())
    CVt, W = premodels(L)
    CVt.to_csv(OUT / "baltarget_premodel.csv", index=False)
    W.to_csv(OUT / "baltarget_weights.csv", index=False)
    print("== 1 pre cvR2 per animal\n" + CVt.pivot_table(index=["set", "y", "animal"], columns="model",
                                                         values="cvR2").reindex(columns=MODELS).round(3).to_string())
    print("== 1 target weight w\n" + W.round(3).to_string(index=False))
    Lp = predict_all(L)
    M = matched(Lp, W)
    M.to_csv(OUT / "baltarget_matched.csv", index=False)
    print("== 2 matched-angle target effect\n" + M.round(3).to_string(index=False))
    S, E = sessions(Lp)
    S.to_csv(OUT / "baltarget_sessions.csv", index=False)
    E.to_csv(OUT / "baltarget_epochspout.csv", index=False)
    for st in SETS:
        for c in ("bal_angle", "bal_both", "bal_mix", "angle", "lerr"):
            print(f"== 3 {st} {c} per animal x epoch x spout\n" + E[E.set == st].pivot_table(
                index=["animal", "pos"], columns="epoch", values=c).reindex(columns=EPS).reindex(POS, level=1)
                .round(2).to_string())
    B = boots(S)
    B.to_csv(OUT / "baltarget_boot.csv", index=False)
    print("== 3 boot (change vs pre)\n" + B.round(3).to_string(index=False))
    Tr, CV = track(S)
    Tr.to_csv(OUT / "baltarget_track.csv", index=False)
    CV.to_csv(OUT / "baltarget_cv.csv", index=False)
    print("== 4 tracking\n" + Tr.round(3).to_string(index=False))
    print(CV[["set", "y", "pred", "n", "loao", "p_shuffle", "loblk", "loblk_plus_days", "loso_known"]].round(3)
          .to_string(index=False))
    DecR, Dd = decoder(L)
    DecR.to_csv(OUT / "baltarget_decoder.csv", index=False)
    Dd.to_csv(OUT / "baltarget_decoder_sessions.csv", index=False)
    print("== 5 decoder\n" + DecR.round(3).to_string(index=False))
    DS, DT = decomp(L)
    DS.to_csv(OUT / "baltarget_decomp_sessions.csv", index=False)
    DT.to_csv(OUT / "baltarget_decomp_tests.csv", index=False)
    print("== 7 balance decomposition\n" + DT.round(3).to_string(index=False))
    C = compare(W, DT, DecR)
    C.to_csv(OUT / "baltarget_compare.csv", index=False)
    print("== compare\n" + C.round(3).T.to_string())
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
