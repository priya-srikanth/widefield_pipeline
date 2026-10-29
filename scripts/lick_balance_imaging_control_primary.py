"""Is the PRIMARY hemispheric balance an IMAGING artefact?  (10-28; redo of lick_balance_imaging_control on the primary)

lick_balance_imaging_control.py (page Fig 14.19) checked the RAW balance (X_right - X_left AUC). The primary recovery
marker is now (lick_balance_subsets.py + lick_recovery_primary.py): the angle-matched GAIN RATIO
    log2(sum_areas obs contra / sum_areas pred contra) - log2(same, ipsi),   areas ALM, tjM1, SSp-m, SSp-n, SSs,
response = mean 0.1..0.5 s after the lick minus mean -1.0..-0.5 s (il.BASE_RESP), pred = the animal's PRE straight line
on lick angle per set x lick order (pre sessions leave-one-pre-session-out; lrm._residuals), sets: cued = first post-cue
lick (trial_1), ITI = first ITI lick after >= 2 s quiet (iti_q). Session values = recprim_sessions.csv `marker`.

Same confounds as the raw-balance control, applied to THAT quantity:
 (1) F0 = mean of frames_average_atlas (raw counts) over each ROI's pixels, 470 (functional slot, per session) and 415;
     asymmetry f0_X = mean over the 5 pool areas of log2(F0_right / F0_left). dF/F = dF / F0, so a one-sided F0 change
     scales that side's obs and NOT its pre-line prediction: pure artefact => d marker = - d f0_470. Checks: F0 asymmetry
     by epoch; within-animal regression marker ~ f0 (MixedLM within + between, per-animal r); the marker after undoing
     the F0 scaling (marker_dF = marker + (f0_470 - animal pre mean)) and after partialling the within-animal F0 slope
     (marker_part); the recovery link (per-animal r, LOAO cvR2, concurrent MixedLM y ~ w + b + log(day)) before / after.
 (2) 415 nm: the SAME estimator on the raw 415 half of SVT (and raw 470 for reference) projected on the same ROIs,
     slow drift (>= 600 s cosines) removed, x1e3 -- exactly as the raw-balance control / the session files. The 415
     lick response is small and sign-variable, so its gain ratio is often undefined; it is ALSO reported as the
     referenced difference rdiff_n = session mean over licks of (sum_areas res_contra - sum_areas res_ipsi) divided by
     the animal's PRE mean neural pooled response (sum_areas (obs_c + obs_i) / 2), i.e. in units of the neural response
     it could contaminate; the neural balance is given in the same units (neu_rdiff_n) for comparison.
Validation: the adopted signal re-sampled here reproduces the session files' stored traces (float16), and the neural
gain ratio rebuilt here reproduces recprim_sessions.csv `marker`.
Outputs (il.OUT_W): balimg_primary_licks.csv.gz (per-lick values, the extraction cache), balimg_primary_sessions.csv,
balimg_primary_f0.csv, balimg_primary_change.csv, balimg_primary_within.csv, balimg_primary_models.csv,
balimg_primary_link.csv; figure figures/balimg_primary_control.png.
Usage: python scripts/lick_balance_imaging_control_primary.py [--jobs 6] [--summarise]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_nested as bn  # noqa: E402
import lick_balance_regions as lbg  # noqa: E402
import lick_covariate_reference as cv  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402

POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
HEMI = {"c": "right", "i": "left"}                 # right = contralesional, left = ipsilesional
SIGS = ("neu", "adp", "r470", "h415")              # neu = stored session-file traces (the primary); adp = re-sampled
WIN = (0.1, 0.5)
EP = il.EP4
POST = EP[1:]
LICKS = il.OUT_W / "balimg_primary_licks.csv.gz"
F0F = il.OUT_W / "balimg_primary_f0.csv"
SETMAP = {"trial_1": lrm.C1SET, "iti_q": lrm.QSET}


def _wm(lo_, hi_):
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & ((L < hi_ - 1e-9) if hi_ < L.max() else (L <= hi_ + 1e-9))


def work(label: str) -> dict:
    """One session (module-level for spawn): per decoder-prediction lick, the primary response (0.1..0.5 s minus
    -1..-0.5 s) per pool area x hemisphere for the stored adopted traces, re-sampled adopted, raw 470, raw 415; F0."""
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.hemo_variants import functional_channel
    d = il._load(label)
    acts = cv._acts(label)
    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    n = len(ft)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    if list(M) != list(d["cols"]):
        raise RuntimeError(f"{label}: ROI columns differ from the session file")
    fc = functional_channel(label)
    man = json.loads((res / "hemo_meegkit_hpfit" / "manifest.json").read_text())
    if int(man.get("functional_channel", fc)) != fc:
        raise RuntimeError(f"{label}: functional slot {fc} vs manifest {man.get('functional_channel')}")
    keys = [f"{a}_{HEMI[s_]}" for a in POOL for s_ in ("c", "i")]
    fa = np.load(res / "allen_aligned_affine8v1" / "frames_average_atlas.npy").astype(np.float64).reshape(2, -1)[:, flat]
    f0 = {"label": label, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "func": fc}
    for k in keys:
        f0[f"f0_470|{k}"] = float(np.nanmean(fa[fc][M[k]]))
        f0[f"f0_415|{k}"] = float(np.nanmean(fa[(fc + 1) % 2][M[k]]))
        f0[f"npix|{k}"] = int(M[k].sum())
    Ra = np.stack([Ub[M[k]].mean(0) for k in keys], 1).astype(np.float64)        # (K, 10)
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    if svt.shape[1] // 2 < n:
        raise RuntimeError(f"{label}: raw SVT {svt.shape[1] // 2} pairs < {n} frames")
    cols = list(d["cols"])
    ci = [cols.index(k) for k in keys]
    bm, wm = _wm(*il.BASE_RESP), _wm(*WIN)
    lt = {"trial": d["tr_t"].astype(float), "iti": d["iti_t"].astype(float)}
    vals = {st: {} for st in lt}
    for st, xk in (("trial", "tr_X"), ("iti", "iti_X")):
        X = d[xk].astype(float)[:, :, ci]
        vals[st]["neu"] = X[:, wm].mean(1) - X[:, bm].mean(1)
    maxdiff = np.nan
    for sig in ("adp", "r470", "h415"):
        if sig == "adp":
            R = V[:n].astype(np.float64) @ Ra
        else:
            slot = fc if sig == "r470" else (fc + 1) % 2
            S = np.asarray(svt[:, slot::2][:, :n], np.float32)
            R = (Ra.T.astype(np.float32) @ S).T.astype(np.float64)
            del S
        R -= Q @ (Q.T @ R)
        R *= 1e3
        for st, t in lt.items():
            X = il._sample(R, ft, t, il.LAGS_D).astype(np.float64)
            if sig == "adp" and st == "trial" and len(t):
                ref = d["tr_X"].astype(np.float64)[:, :, ci]
                maxdiff = float(np.nanmax(np.abs(X - ref)) / max(np.nanpercentile(np.abs(ref), 99), 1e-9))
            vals[st][sig] = X[:, wm].mean(1) - X[:, bm].mean(1)
        del R
    out = []
    for st, sk in (("iti", "iti_side"), ("trial", "tr_side")):
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        ok, mm = m >= 0, np.maximum(m, 0)
        Rr = {"set": st, "label": label, "k": np.arange(len(act))}
        for sig in SIGS:
            for j, k in enumerate(keys):
                a, h = k.rsplit("_", 1)
                s_ = "c" if h == "right" else "i"
                Rr[f"{sig}|{a}|{s_}"] = np.where(ok, vals[st][sig][mm, j], np.nan)
        out.append(pd.DataFrame(Rr))
    return {"licks": pd.concat(out, ignore_index=True), "f0": f0, "maxdiff": maxdiff}


def extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    labs = sorted(pd.read_csv(lo.preds_path(), usecols=["label"]).label.unique())
    res, fails = ak.fan_sessions(labs, work, jobs=jobs)
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    md = pd.Series({x: v["maxdiff"] for x, v in res})
    print(f"== validation: max |adopted re-sampled - stored| / 99th pct |stored| = {md.max():.4f} (float16 store; "
          f"median {md.median():.4f})", flush=True)
    pd.concat([v["licks"] for _x, v in res], ignore_index=True).to_csv(LICKS, index=False, float_format="%.7g")
    pd.DataFrame([v["f0"] for _x, v in res]).to_csv(F0F, index=False, float_format="%.8g")


# ------------------------------------------------------------------------------------------------------------ SESSIONS
def _ratio(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((o > 0) & (p > 0), np.log2(np.clip(o, 1e-12, None) / np.clip(p, 1e-12, None)), np.nan)


def sessions() -> pd.DataFrame:
    """Per set x session: gain-ratio marker per signal (pooled exactly as lick_recovery_primary.marker), referenced
    difference in pre-neural units, F0 asymmetry; joined to recprim_sessions (day, recovery axes)."""
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    P = P.merge(pd.read_csv(LICKS), on=["set", "label", "k"], how="left", validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P["ord_c"] = np.minimum(P.order, lbg.CAP)
    vs = [f"{sig}|{a}|{s}" for sig in SIGS for a in POOL for s in ("c", "i")]
    gres, gfail = ak.fan_sessions(sorted(P[P.set == "iti"].label.unique()), lrm._gaps, jobs=6)
    if gfail:
        raise RuntimeError(f"gaps failed: {gfail[:5]}")
    Q = P[P.set == "iti"].merge(pd.concat([v for _i, v in gres], ignore_index=True), on=["set", "label", "k"],
                                how="left", validate="one_to_one")
    Q = Q[Q.gap >= il.GAP_MAIN].assign(set="iti_q").drop(columns="gap")
    Q1 = P[(P.set == "trial") & (P.order == 0)].assign(set="trial_1")
    P = lrm._residuals(pd.concat([Q, Q1], ignore_index=True), vs)
    rows = []
    for (st, lab, an, ep), g in P.groupby(["set", "label", "animal", "epoch"]):
        rec = {"set": SETMAP[st], "label": lab, "animal": an, "epoch": ep, "n_licks": len(g)}
        for sig in SIGS:
            ob = {s: g[[f"{sig}|{a}|{s}" for a in POOL]] for s in ("c", "i")}
            rs = {s: g[[f"res_{sig}|{a}|{s}" for a in POOL]] for s in ("c", "i")}
            # as lick_balance_subsets: per area, licks with obs and residual both finite, then summed over areas
            for s in ("c", "i"):
                o_, p_ = 0.0, 0.0
                for a in POOL:
                    ok = ob[s][f"{sig}|{a}|{s}"].notna() & rs[s][f"res_{sig}|{a}|{s}"].notna()
                    o_ += ob[s][f"{sig}|{a}|{s}"][ok].sum()
                    p_ += (ob[s][f"{sig}|{a}|{s}"][ok] - rs[s][f"res_{sig}|{a}|{s}"][ok]).sum()
                rec[f"{sig}_obs_{s}"], rec[f"{sig}_pred_{s}"] = o_, p_
            rec[f"{sig}_marker"] = float(_ratio(rec[f"{sig}_obs_c"], rec[f"{sig}_pred_c"])
                                         - _ratio(rec[f"{sig}_obs_i"], rec[f"{sig}_pred_i"]))
            rd = rs["c"].sum(1, min_count=len(POOL)) - rs["i"].sum(1, min_count=len(POOL))
            rec[f"{sig}_rdiff"] = float(rd.mean())
            rec[f"{sig}_lvl"] = float(((ob["c"].sum(1, min_count=len(POOL)) + ob["i"].sum(1, min_count=len(POOL))) / 2).mean())
        rows.append(rec)
    S = pd.DataFrame(rows)
    pre_resp = S[S.epoch == "pre"].groupby(["set", "animal"]).neu_lvl.mean().rename("pre_resp")
    S = S.join(pre_resp, on=["set", "animal"])
    for sig in SIGS:
        S[f"{sig}_rdiff_n"] = S[f"{sig}_rdiff"] / S.pre_resp
    F = pd.read_csv(F0F)
    for ch in ("470", "415"):
        F[f"f0_{ch}"] = np.mean([np.log2(F[f"f0_{ch}|{a}_right"] / F[f"f0_{ch}|{a}_left"]) for a in POOL], 0)
    S = S.merge(F[["label", "f0_470", "f0_415"]], on="label", how="left", validate="many_to_one")
    R = pd.read_csv(il.OUT_W / "recprim_sessions.csv")
    chk = S.merge(R[["set", "label", "marker"]].drop_duplicates(["set", "label"]), on=["set", "label"])
    print(f"== reproduction of recprim_sessions.csv marker: {len(chk)} rows, max |diff| "
          f"{np.nanmax(np.abs(chk.neu_marker - chk.marker)):.2e}; NaN mismatch "
          f"{int((chk.neu_marker.isna() != chk.marker.isna()).sum())}", flush=True)
    S = S.merge(R[["set", "label", "day"]].drop_duplicates(["set", "label"]), on=["set", "label"], how="left")
    ys = R.pivot_table(index="label", columns="yname", values="y", aggfunc="first")
    S = S.join(ys, on="label")
    # F0-undone and within-animal-partialled markers
    S["marker"] = S.neu_marker
    S["d_f0_470"] = S.f0_470 - S.groupby(["set", "animal"]).f0_470.transform(
        lambda v: v[S.loc[v.index, "epoch"] == "pre"].mean())
    S["marker_dF"] = S.marker + S.d_f0_470
    return S


# ------------------------------------------------------------------------------------------------------------ SUMMARY
def _boot(g: pd.DataFrame, v: str, ep: str):
    from wfield_local import analysis_kit as ak
    ref = {a: g[(g.animal == a) & (g.epoch == "pre")][v].dropna().to_numpy() for a in lrm.AN}
    post = {a: g[(g.animal == a) & (g.epoch == ep)][v].dropna().to_numpy() for a in lrm.AN}
    com = [a for a in lrm.AN if len(ref[a]) and len(post[a])]
    if len(com) < 3:
        return None, {}
    iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
    return iv, {a: post[a].mean() - ref[a].mean() for a in com}


MEAS = ["f0_470", "f0_415", "marker", "marker_dF", "marker_part", "neu_rdiff_n", "h415_rdiff_n", "r470_rdiff_n",
        "h415_marker", "r470_marker"]
PANELS = [(lrm.C1SET, "aim"), (lrm.QSET, "iti_ang"), (lrm.QSET, "aim")]


def summarise() -> None:
    import statsmodels.formula.api as smf
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    S = sessions()
    # ---- within-animal F0 regression (all sessions) and the partialled marker
    W, Mo = [], []
    parts = []
    for st, g in S.groupby("set"):
        g = g.dropna(subset=["marker", "f0_470"]).copy()
        g["f0_w"] = g.f0_470 - g.groupby("animal").f0_470.transform("mean")
        g["f0_b"] = g.groupby("animal").f0_470.transform("mean")
        for an, h in g.groupby("animal"):
            for sub, hh in (("all", h), ("pre", h[h.epoch == "pre"]), ("post", h[h.epoch != "pre"])):
                if len(hh) > 3 and hh.f0_470.std() > 0:
                    W.append({"set": st, "animal": an, "sessions": sub, "n": len(hh),
                              "r": float(np.corrcoef(hh.f0_470, hh.marker)[0, 1]),
                              "slope": float(np.polyfit(hh.f0_470, hh.marker, 1)[0]),
                              "f0_470_sd": float(hh.f0_470.std())})
        f = smf.mixedlm("marker ~ f0_w + f0_b", g, groups=g["animal"]).fit(reml=True)
        bw = float(f.params["f0_w"])
        Mo.append({"set": st, "model": "marker ~ f0_w + f0_b + (1|animal), all sessions", "term": "f0_w", "coef": bw,
                   "se": f.bse["f0_w"], "p": f.pvalues["f0_w"], "n": len(g)})
        gp = g[g.epoch == "pre"]
        fp = smf.mixedlm("marker ~ f0_w", gp, groups=gp["animal"]).fit(reml=True)
        Mo.append({"set": st, "model": "marker ~ f0_w + (1|animal), PRE sessions", "term": "f0_w",
                   "coef": fp.params["f0_w"], "se": fp.bse["f0_w"], "p": fp.pvalues["f0_w"], "n": len(gp)})
        g["marker_part"] = g.marker - bw * g.f0_w
        parts.append(g[["set", "label", "marker_part"]])
        for cov in ("", " + f0_470", " + f0_415", " + h415_rdiff_n"):
            h = g.dropna(subset=["h415_rdiff_n", "f0_415"])
            try:
                f = smf.mixedlm(f"marker ~ C(epoch, Treatment('pre')){cov}", h, groups=h["animal"]).fit(reml=True)
                for ep in POST:
                    t = f"C(epoch, Treatment('pre'))[T.{ep}]"
                    Mo.append({"set": st, "model": f"marker ~ epoch{cov} + (1|animal)", "term": ep, "coef": f.params[t],
                               "se": f.bse[t], "p": f.pvalues[t], "n": len(h)})
            except Exception as ex:  # noqa: BLE001
                Mo.append({"set": st, "model": f"epoch{cov}: FAILED {ex}"})
    S = S.merge(pd.concat(parts), on=["set", "label"], how="left")
    S.to_csv(il.OUT_W / "balimg_primary_sessions.csv", index=False, float_format="%.7g")
    W, Mo = pd.DataFrame(W), pd.DataFrame(Mo)
    W.to_csv(il.OUT_W / "balimg_primary_within.csv", index=False, float_format="%.6g")
    Mo.to_csv(il.OUT_W / "balimg_primary_models.csv", index=False, float_format="%.6g")
    print("== fraction of sessions with an undefined gain ratio: "
          + ", ".join(f"{st[:5]} {sig} {S[S.set == st][f'{sig}_marker'].isna().mean():.2f}"
                      for st in (lrm.C1SET, lrm.QSET) for sig in ("neu", "r470", "h415")))
    # ---- per-animal tables first (rule 8)
    for st in (lrm.C1SET, lrm.QSET):
        print(f"\n-- per animal x epoch means: {st}")
        print(S[S.set == st].groupby(["animal", "epoch"])[MEAS].mean().unstack("epoch")
              .reorder_levels([1, 0], axis=1).reindex(columns=EP, level=0).T.round(3).to_string())
    # ---- change vs pre (boot_delta)
    rows = []
    for st, g in S.groupby("set"):
        for v in MEAS:
            for ep in POST:
                iv, per = _boot(g, v, ep)
                if iv is None:
                    continue
                rows.append({"set": st, "measure": v, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                             "sig": bool(iv.lo > 0 or iv.hi < 0), "n_animals": iv.n_animals,
                             **{f"d_{an}": x for an, x in per.items()}})
    C = pd.DataFrame(rows)
    C.to_csv(il.OUT_W / "balimg_primary_change.csv", index=False, float_format="%.6g")
    C["s"] = [f"{d:+.3f}{'*' if s else ''}" for d, s in zip(C.delta, C.sig)]
    for st in (lrm.C1SET, lrm.QSET):
        print(f"\n== change vs pre (boot_delta animals>sessions; * CI excludes 0): {st}")
        t = C[C.set == st]
        print(t.pivot_table(index="measure", columns="epoch", values="s", aggfunc="first")
              .reindex(index=MEAS, columns=POST).to_string())
        print("   per animal:\n" + t.set_index(["measure", "epoch"])[[f"d_{a}" for a in lrm.AN]]
              .reindex(MEAS, level=0).round(3).to_string())
    print("\n== within-animal r / slope (marker ~ f0_470 asymmetry); artefact predicts slope -1\n"
          + W.pivot_table(index=["set", "sessions"], columns="animal", values=["r", "slope"]).round(2).to_string())
    print("\n== models (sessions as observations: anti-conservative)\n" + Mo.round(4).to_string(index=False))
    # ---- recovery link before / after
    L = []
    for st, yv in PANELS:
        G = S[(S.set == st) & (S.epoch != "pre") & (S.day > 0)].copy()
        G["logday"] = np.log(G.day)
        for x in ("marker", "marker_dF", "marker_part", "f0_470", "h415_rdiff_n", "neu_rdiff_n"):
            h = G.dropna(subset=[x, yv]).copy()
            rec = {"panel": f"{st} -> {yv}", "x": x, "n": len(h),
                   "loao": bn._loao_r2(h[x].to_numpy(float), h[yv].to_numpy(float), h.animal.to_numpy())}
            for an, q in h.groupby("animal"):
                rec[f"r_{an}"] = float(np.corrcoef(q[x], q[yv])[0, 1])
            h["w"] = h[x] - h.groupby("animal")[x].transform("mean")
            h["b"] = h.groupby("animal")[x].transform("mean")
            h["y"] = h[yv]
            try:
                m = smf.mixedlm("y ~ w + b + logday", h, groups=h["animal"]).fit(reml=False)
                rec.update({"slope_w": m.fe_params["w"], "p_w": m.pvalues["w"]})
            except Exception:  # noqa: BLE001
                pass
            L.append(rec)
    L = pd.DataFrame(L)
    L.to_csv(il.OUT_W / "balimg_primary_link.csv", index=False, float_format="%.6g")
    print("\n== recovery link (post sessions): LOAO cvR2, per-animal r, concurrent MixedLM y ~ w + b + log(day)\n"
          + L.round(3).to_string(index=False))
    figure(S, C, W, L)


def figure(S: pd.DataFrame, C: pd.DataFrame, W: pd.DataFrame, L: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    ec = {"pre": "0.6", **{e: EPOCH_LINE[e] for e in POST}}
    sets = [(lrm.C1SET, "cued first lick"), (lrm.QSET, "ITI first after quiet")]
    fig, axs = plt.subplots(3, 4, figsize=(22, 15))

    def bars(ax, st, meas, labels, title, ylab=None):
        w = 0.8 / len(meas)
        x = np.arange(len(POST))
        for i, (v, lab) in enumerate(zip(meas, labels)):
            t = C[(C.set == st) & (C.measure == v)].set_index("epoch").reindex(POST)
            xx = x + (i - (len(meas) - 1) / 2) * w
            ax.bar(xx, t.delta, width=w * 0.9, color=f"C{i}", alpha=0.45, label=lab)
            ax.errorbar(xx, t.delta, yerr=[t.delta - t.lo, t.hi - t.delta], fmt="none", ecolor="k", capsize=2, lw=0.8)
            for j, an in enumerate(lrm.AN):
                if f"d_{an}" in t:
                    ax.scatter(xx + (j - 1.5) * w * 0.15, t[f"d_{an}"], marker=ANIMAL_MARKER.get(an, "o"),
                               facecolor="none", edgecolor="k", s=18, lw=0.6, zorder=3)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(x, POST)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        if ylab:
            ax.set_ylabel(ylab, fontsize=8.5)

    # row 1: F0 asymmetry by epoch (one per set: the same sessions, F0 is per session) + marker vs F0 within animal
    bars(axs[0, 0], lrm.C1SET, ["f0_470", "f0_415"], ["F0 470", "F0 415"],
         "(a) F0 asymmetry, mean over 5 areas of log2(F0 right / F0 left)\nchange vs pre", "change vs pre\n(bar = boot_delta CI; symbols = animals)")
    ax = axs[0, 1]
    for an in lrm.AN:
        for ep in EP:
            h = S[(S.set == lrm.C1SET) & (S.animal == an) & (S.epoch == ep)]
            ax.plot(h.day, h.f0_470, ANIMAL_MARKER.get(an, "o"), color=ec[ep], ms=5, mec="k", mew=0.3)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.set_xlabel("day since stroke", fontsize=8.5)
    ax.set_ylabel("F0 470 log2(R/L)", fontsize=8.5)
    ax.set_title("(b) F0 470 asymmetry per session", fontsize=9)
    for c, (st, nm) in enumerate(sets):
        ax = axs[0, 2 + c]
        g = S[S.set == st].copy()
        g["d_marker"] = g.marker - g.animal.map(g[g.epoch == "pre"].groupby("animal").marker.mean())
        for an in lrm.AN:
            for ep in EP:
                h = g[(g.animal == an) & (g.epoch == ep)]
                ax.plot(h.d_f0_470, h.d_marker, ANIMAL_MARKER.get(an, "o"), color=ec[ep], ms=5, mec="k", mew=0.3)
        lim = np.nanmax(np.abs(g.d_f0_470)) * 1.1
        ax.plot([-lim, lim], [lim, -lim], "k:", lw=1, label="pure F0 artefact (slope -1)")
        w = W[(W.set == st) & (W.sessions == "all")].set_index("animal")
        ax.set_title(f"({'cd'[c]}) {nm}: balance vs F0 asymmetry, change from animal's pre mean\nwithin-animal r "
                     + " / ".join(f"{w.r.get(a, np.nan):+.2f}" for a in lrm.AN), fontsize=9)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("d F0 470 log2(R/L)", fontsize=8.5)
        ax.set_ylabel("d balance gain ratio (log2)", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False)
    # row 2: marker change vs pre, raw / F0-undone / partialled; recovery link before / after
    for c, (st, nm) in enumerate(sets):
        bars(axs[1, c], st, ["marker", "marker_dF", "marker_part"],
             ["primary balance", "F0 scaling undone", "within-animal F0 slope partialled"],
             f"({'ef'[c]}) {nm}: primary balance, change vs pre", "change vs pre (log2)")
    for c, (st, yv) in enumerate([PANELS[0], PANELS[1]]):
        ax = axs[1, 2 + c]
        t = L[L.panel == f"{st} -> {yv}"].set_index("x").reindex(["marker", "marker_dF", "marker_part", "f0_470",
                                                                  "h415_rdiff_n"])
        x = np.arange(len(t))
        ax.bar(x, t.loao, color=["#C44E52", "#DD8452", "#937860", "0.6", "#4C72B0"])
        for j, an in enumerate(lrm.AN):
            ax.scatter(x + (j - 1.5) * 0.12, t[f"r_{an}"], marker=ANIMAL_MARKER.get(an, "o"), facecolor="none",
                       edgecolor="k", s=22, zorder=3, label=f"{an} r" if j < 4 and c == 0 else None)
        for xi, (_x, r) in enumerate(t.iterrows()):
            if np.isfinite(r.get("p_w", np.nan)):
                ax.text(xi, -0.95, f"p_w {r.p_w:.2g}", ha="center", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xticks(x, ["balance", "F0 undone", "F0 partialled", "F0 asym\nalone", "415 balance"], fontsize=7.5)
        ax.set_title(f"({'gh'[c]}) recovery link, post sessions: {st} -> {yv}\nbar = LOAO cvR², symbols = per-animal r; "
                     "p_w = MixedLM within slope (+ log day)", fontsize=9)
        if c == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    # row 3: 415 nm
    for c, (st, nm) in enumerate(sets):
        bars(axs[2, c], st, ["neu_rdiff_n", "r470_rdiff_n", "h415_rdiff_n"],
             ["neural (adopted)", "raw 470", "415 nm"],
             f"({'ij'[c]}) {nm}: referenced balance (sum res contra - ipsi)\n/ animal's pre neural response, change vs pre",
             "change vs pre (pre-response units)")
    for c, (st, nm) in enumerate(sets):
        ax = axs[2, 2 + c]
        g = S[S.set == st]
        for an in lrm.AN:
            for ep in EP:
                h = g[(g.animal == an) & (g.epoch == ep)]
                ax.plot(h.neu_rdiff_n, h.h415_rdiff_n, ANIMAL_MARKER.get(an, "o"), color=ec[ep], ms=5, mec="k", mew=0.3)
        lim = np.nanmax(np.abs(g[["neu_rdiff_n", "h415_rdiff_n"]].to_numpy())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="0.7", ls=":", lw=1)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("neural referenced balance (pre-response units)", fontsize=8.5)
        ax.set_ylabel("415 referenced balance (same units)", fontsize=8.5)
        ax.set_title(f"({'kl'[c]}) {nm}: 415 vs neural balance per session", fontsize=9)
    for a_ in axs.ravel():
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    h = [Line2D([], [], color=ec[e], marker="o", lw=0, label=e) for e in EP]
    h += [Line2D([], [], color="0.3", marker=ANIMAL_MARKER.get(a, "o"), lw=0, label=a) for a in lrm.AN]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8.5, frameon=False)
    fig.suptitle("Is the PRIMARY hemispheric balance (angle-matched gain ratio, ALM + tjM1 + SSp-m + SSp-n + SSs, 0.1-0.5 s "
                 "after the first lick) an imaging artefact? right = contralesional\nF0 = mean raw fluorescence per ROI "
                 "(470 / 415); pure F0 artefact predicts d balance = - d F0 asymmetry; 415 = isosbestic channel through "
                 "the same estimator; boot_delta animals > sessions", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = il.OUT_W / "figures" / "balimg_primary_control.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"== wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args(argv)
    if not a.summarise or not LICKS.exists():
        extract(min(a.jobs, 6))
    summarise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
