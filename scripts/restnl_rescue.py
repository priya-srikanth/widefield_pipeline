"""RESTnl RESCUE SWEEP (Priya 10-29: "this is not ok -- how can we rescue, can we make the timing less stringent").
PROVISIONAL, n = 4.

PROBLEM. RESTnl (scripts/restnl.py = restw_strict_check `strict_w50`: restdock05 rest minus every cam4 lick and DAQ lick
[0.5 s before, 2.0 s after], equal-weight mean of the 6 per-position medians, >= 50 frames at EVERY position) leaves 10
sessions out, PS93 pre 7 -> 2.

SWEEP (per session; nothing on disk is overwritten -- own cache <restw_check>/rescue/sessions/, VERSION below):
  MASK   s20 / s15 / s10 : restdock05 minus every DAQ lick and EVERY cam4 lick (contact or not), buffer [0.5, X] s,
                           X = 2.0 / 1.5 / 1.0 (s20 == strict, checked against the restw_check cache)
         r20 / r15 / r10 : RESIDUAL-lick variant: DAQ licks [0.5, X] + only the cam4 licks of bouts (gap > 1 s splits)
                           with NO contact anywhere in the bout (no cam4 contact, no DAQ onset within the bout span);
                           cam4 non-contact licks inside contact bouts are NOT excluded beyond the DAQ buffer.
         The mask's min_rest_s floor is re-applied after each intersection (as restw_strict_check).
  FLOOR  per-position frame floor 50 / 20 / 10
  WEIGHT w6   : equal-weight mean of the 6 per-position medians, all 6 >= floor (current RESTnl rule)
         wr   : equal-weight mean over the positions that pass the floor (renormalized), >= 4 of 6 required
         flat : unweighted median over all rest frames, total >= 6 x floor
  Variant name <mask>_f<floor>_<weight>; the reference is s20_f50_w6 (= RESTnl).
LEVEL / frames = exactly restw_strict_check.session_worker (ROI v8 split traces, `rest_frames_by_position(docked=False,
engaged_only=True)` with the variant mask substituted in-process).
HEADLINES by baseline swap on the CURRENT engaged gate (precue_restnl p10, cue_to_lick/sessions_restnl):
  H1 chronic contra pre-cue rise, all spouts (precue = mean p10 over engaged trials - level, >= 10 trials)
  H2 acute far-contra pre-cue raise (far_contra trials only, >= 5)
  H3 chronic cue-to-lick mean level (mean_rw, raw + latency-matched, all spouts; lick_cue_to_lick per_session / matched /
     deltas, rng default_rng(0) in restw_strict_check.swap_c2l's order)
  every change = analysis_kit.boot_delta (iti_area_maps.changes for H1/H2); per-animal table printed first (rule 8).
LICK-TRIGGERED CHECK (how long does a lick stay in the trace?): per session, ITI cam4 licks next to rest:
  iso      single licks (no other cam4 / DAQ lick within [-3, +5] s), restdock05 rest at -1.5 and -0.6 s;
           response = trace - mean(-1.0..-0.5 s)
  boutend  last lick of a >= 2-lick bout, no lick for the next 5 s, restdock05 rest at +2.5 and +4.5 s;
           response = trace - the session's s20_f50_w6 (RESTnl) level  (offline)
Outputs: <iti_licks>/restw_check/rescue/ (sessions/<label>.npz, *.csv, restnl_rescue.png, figdata/)
Usage: python scripts/restnl_rescue.py [--jobs 4] [--labels A B ...] [--summarise]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import restnl  # noqa: E402
import restw_strict_check as rsc  # noqa: E402

OUTD = il.OUT / "restw_check" / "rescue"            # the ONLY write root
SESS = OUTD / "sessions"
VERSION = 1
PRE_BUF = 0.5
POSTS = {"20": 2.0, "15": 1.5, "10": 1.0}
MASKS = [f"{k}{p}" for k in ("s", "r") for p in POSTS]
FLOORS = [50, 20, 10]
WEIGHTS = ["w6", "wr", "flat"]
MIN_POS_WR = 4
REF = "s20_f50_w6"
BOUT_GAP = 1.0
LAGS_T = np.round(np.arange(-2.0, 5.0001, 0.05), 3)
AN, EP = rsc.AN, rsc.EP
KEY = ["tjM1_right", "ALM_right", "SSp-m_right", "SSp-n_right", "RSPd_right", "VISp_right"]
HKEY = ["tjM1_right", "ALM_right", "SSp-m_right", "SSp-n_right"]


# ------------------------------------------------------------------------------------------------ per session
def _bouts(t: np.ndarray, gap: float) -> np.ndarray:
    """Bout id per sorted event time (a gap > `gap` s starts a new bout)."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap)]) if len(t) else np.array([], int)


def session_worker(label: str) -> dict:
    f = SESS / f"{label}.npz"
    if f.exists() and int(np.load(f)["version"]) == VERSION:
        return {"label": label, "cached": True}
    import lick_cue_to_lick as c2l
    import lick_cue_to_lick_extract as cx
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import rest_by_position as rbp
    from wfield_local.paths import PathResolver
    s = next(x for x in config.load_sessions() if x["label"] == label)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    ref = np.load(il.OUT / "restw_check" / "sessions" / f"{label}.npz")
    # == restw_strict_check.session_worker's trace construction
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")
    if cs != [str(c) for c in ref["cols"]]:
        raise ValueError(f"{label}: columns differ from the restw_check cache")
    nT = Rs.shape[0]
    del V, Ub
    rest0, cs_s, codes, ts, fs_samp, sync = rbp._session_daq(s)
    fs = ak.daq_rate(s)
    eng = rbp._engaged_trials(s, cs_s, codes, fs)
    n = rest0.shape[0]
    minrun = int(float(config.defaults()["segmentation"]["rest"]["min_rest_s"]) * fs)
    daq_on = rsc._daq_lick_onsets(s, fs)
    sd = oo.session_dir(PathResolver(), oc.RUN, an, date)
    A = pd.read_csv(sd / "licks_all.csv", usecols=["context", "contact", "daq_on_s"])
    A = A[np.isfinite(A.daq_on_s.to_numpy(float))].sort_values("daq_on_s")
    c4 = A.daq_on_s.to_numpy(float)
    c4_con = A.contact.astype(str).str.lower().eq("true").to_numpy()
    c4_ctx = A.context.astype(str).to_numpy()
    # residual licks: cam4 licks in bouts with no contact (cam4 contact or a DAQ onset within the bout span +- 0.1 s)
    b = _bouts(c4, BOUT_GAP)
    daq_s = daq_on / fs
    resid = np.zeros(len(c4), bool)
    for k in np.unique(b):
        m = b == k
        t0, t1 = c4[m].min() - 0.1, c4[m].max() + 0.1
        has_daq = np.searchsorted(daq_s, t1, "right") > np.searchsorted(daq_s, t0, "left")
        resid[m] = not (c4_con[m].any() or has_daq)

    def smp(t_s):
        return np.round(np.asarray(t_s, float) * fs).astype(np.int64)
    nb = int(PRE_BUF * fs)
    masks = {}
    for k, post in POSTS.items():
        na = int(post * fs)
        m_daq = rsc.floor_runs(rest0 & ~rsc.widen(daq_on, n, nb, na), minrun)
        masks[f"s{k}"] = rsc.floor_runs(m_daq & ~rsc.widen(smp(c4), n, nb, na), minrun)
        masks[f"r{k}"] = rsc.floor_runs(m_daq & ~rsc.widen(smp(c4[resid]), n, nb, na), minrun)
    orig_daq, orig_eng = rbp._session_daq, rbp._engaged_trials
    out = {"version": VERSION, "cols": np.array(cs), "animal": str(ref["animal"]), "epoch": str(ref["epoch"]),
           "n_cam4": len(c4), "n_resid": int(resid.sum()), "n_daq": len(daq_on)}
    try:
        rbp._engaged_trials = lambda *_a, **_k: eng
        for v, m in masks.items():
            rbp._session_daq = lambda _s, _m=m: (_m, cs_s, codes, ts, fs_samp, sync)
            per, info = rbp.rest_frames_by_position(s, nT, docked=False, engaged_only=True)
            if info.get("error"):
                raise RuntimeError(f"{label} {v}: {info['error']}")
            fr = [np.asarray(per.get(c, []), int) for c in range(6)]
            fr = [x[x < nT] for x in fr]
            out[f"{v}__nfr_pos"] = np.array([len(x) for x in fr], int)
            out[f"{v}__posmed"] = np.stack([np.median(Rs[x], 0) if len(x) else np.full(len(cs), np.nan) for x in fr], 0)
            allf = np.concatenate(fr)
            out[f"{v}__flat"] = np.median(Rs[allf], 0) if allf.size else np.full(len(cs), np.nan)
            out[f"{v}__nsmp"] = int(m.sum())
    finally:
        rbp._session_daq, rbp._engaged_trials = orig_daq, orig_eng
    out["s20_check_maxdiff"] = float(np.nanmax(np.abs(out["s20__posmed"] - ref["strict__posmed"])))
    out["s20_check_nfr"] = int(np.abs(out["s20__nfr_pos"] - ref["strict__nfr_pos"]).sum())
    # ---- lick-triggered traces (ITI licks next to restdock05 rest)
    allick = np.sort(np.concatenate([c4, daq_s]))

    def rest_at(t_s):
        j = smp(t_s)
        ok = (j >= 0) & (j < n)
        r = np.zeros(len(j), bool)
        r[ok] = rest0[j[ok]]
        return r

    iti = c4_ctx == "iti"
    t_iso = c4[iti]
    if len(t_iso):
        lo = np.searchsorted(allick, t_iso - 3.0, "left")
        hi = np.searchsorted(allick, t_iso + 5.0, "right")
        near = np.array([np.sum(np.abs(allick[a:bb] - x) > 0.15) for a, bb, x in zip(lo, hi, t_iso)])
        t_iso = t_iso[(near == 0) & rest_at(t_iso - 1.5) & rest_at(t_iso - 0.6)]
    # bout ends over the union of cam4 + DAQ licks
    bu = _bouts(allick, BOUT_GAP)
    ends = []
    for k in np.unique(bu):
        tk = allick[bu == k]
        if len(tk) < 2:
            continue
        te = tk.max()
        nxt = allick[allick > te + 1e-6]
        if (len(nxt) == 0 or nxt[0] - te > 5.0):
            ends.append(te)
    t_end = np.asarray(ends, float)
    # the bout must contain an ITI cam4 lick (iti context) and the post-lick period must be rest
    if len(t_end):
        has_iti = np.array([np.any(iti & (np.abs(c4 - x) < 3.0) & (c4 <= x + 1e-6)) for x in t_end])
        t_end = t_end[has_iti & rest_at(t_end + 2.5) & rest_at(t_end + 4.5)]
    for nm, tt in (("iso", t_iso), ("boutend", t_end)):
        tt = tt[(tt + LAGS_T[0] > ft[0]) & (tt + LAGS_T[-1] < ft[-1])]
        out[f"lt_{nm}_n"] = len(tt)
        X = il._sample(Rs, ft, tt, LAGS_T).astype(np.float64) if len(tt) else np.full((0, len(LAGS_T), len(cs)), np.nan)
        out[f"lt_{nm}_mean"] = X.mean(0) if len(tt) else np.full((len(LAGS_T), len(cs)), np.nan)
    # ---- cue-to-lick per-trial mean (raw: restw set to 0) on the current engaged gate (cue_to_lick/sessions_restnl)
    zf = cx.SESS_NL / f"{label}.npz"
    if zf.exists():
        z = dict(np.load(zf))
        out["c2l_restnl_lev"] = np.asarray(z["restw"], float)
        z["restw"] = np.zeros_like(np.asarray(z["restw"], float))
        z["restw_ok"] = np.array(True)
        orig_load = np.load
        try:
            np.load = lambda *_a, **_k: z
            r = c2l._session_trials(label, cx.SESS_NL)
        finally:
            np.load = orig_load
        T = r["T"]
        T = T[T.tau == 0]
        piv = T.pivot_table(index="trial", columns="col", values="mean_rw").reindex(columns=cx.COLS)
        meta = T[T.col == cx.COLS[0]].set_index("trial").loc[piv.index]
        out["c2l_trial"] = piv.index.to_numpy(int)
        out["c2l_pos"] = meta.pos.astype(str).to_numpy().astype(str)
        out["c2l_lat"] = meta.lat.to_numpy(float)
        out["c2l_raw"] = piv.to_numpy(float)
    SESS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return {"label": label, "cached": False}


# ------------------------------------------------------------------------------------------------ levels per variant
def variant_level(z, mask: str, floor: int, weight: str) -> tuple[np.ndarray, bool, int]:
    """(level per col, ok, n positions used)."""
    npos = z[f"{mask}__nfr_pos"]
    pm = z[f"{mask}__posmed"]
    if weight == "w6":
        return pm.mean(0), bool((npos >= floor).all()), 6
    if weight == "wr":
        use = npos >= floor
        lev = pm[use].mean(0) if use.any() else np.full(pm.shape[1], np.nan)
        return lev, bool(use.sum() >= MIN_POS_WR), int(use.sum())
    return z[f"{mask}__flat"], bool(npos.sum() >= 6 * floor), int((npos > 0).sum())


def vnames() -> list[str]:
    return [f"{m}_f{f}_{w}" for m in MASKS for f in FLOORS for w in WEIGHTS]


def load_levels(labels: list) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(long level table label x variant x area, session x variant table, raw npz dict)."""
    rows, srows, Z = [], [], {}
    for x in labels:
        z = dict(np.load(SESS / f"{x}.npz", allow_pickle=True))  # own cache (c2l_pos object array)
        Z[x] = z
        cs = [str(c) for c in z["cols"]]
        for v in vnames():
            m, f, w = v.split("_")
            lev, ok, npu = variant_level(z, m, int(f[1:]), w)
            srows.append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "variant": v, "ok": ok,
                          "npos_used": npu, "nfr": int(z[f"{m}__nfr_pos"].sum()),
                          "nfr_min_pos": int(z[f"{m}__nfr_pos"].min())})
            for j, c in enumerate(cs):
                rows.append({"label": x, "variant": v, "area": c, "lev": float(lev[j]), "ok": ok})
    return pd.DataFrame(rows), pd.DataFrame(srows), Z


# ------------------------------------------------------------------------------------------------ headlines
def precue_tables(labels: list) -> dict:
    """{None | 'far_contra': long table label x area: raw mean p10 (no reference), n} from precue_restnl."""
    out = {None: [], "far_contra": []}
    for x in labels:
        z = np.load(il.OUT_W / restnl.cache("precue_restnl") / f"{x}.npz")   # where lick_precue_restw writes (D9 11-08)
        cs = [str(c) for c in z["cols"]]
        pos = z["pos"].astype(str)
        for sp, mn in ((None, 10), ("far_contra", 5)):
            m = np.ones(len(pos), bool) if sp is None else pos == sp
            if m.sum() < mn:
                continue
            a10 = np.nanmean(z["p10"][m], 0)
            for j, c in enumerate(cs):
                if c in HKEY:
                    out[sp].append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "area": c,
                                    "raw": a10[j], "restnl": float(z["restw"][j]), "restnl_ok": bool(z["restw_ok"])})
    return {k: pd.DataFrame(v) for k, v in out.items()}


def c2l_table(Z: dict) -> pd.DataFrame:
    import lick_cue_to_lick as c2l
    import lick_cue_to_lick_extract as cx
    rows = []
    for x, z in Z.items():
        if "c2l_raw" not in z:
            continue
        for j, c in enumerate(cx.COLS):
            if c not in HKEY:
                continue
            rows.append(pd.DataFrame({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]),
                                      "trial": z["c2l_trial"], "pos": z["c2l_pos"], "lat": z["c2l_lat"], "col": c,
                                      "raw": z["c2l_raw"][:, j], "restnl": float(z["c2l_restnl_lev"][j])}))
    T = pd.concat(rows, ignore_index=True)
    T["bin"] = np.clip(np.digitize(T.lat, c2l.EDGES) - 1, 0, len(c2l.EDGES) - 2)
    return T


def headlines(v: str, L: pd.DataFrame, PC: dict, C2: pd.DataFrame) -> tuple[list, list]:
    """(rows: headline x area -> delta/lo/hi/n_same_sign, per-animal rows)."""
    import iti_area_maps as iam
    import lick_cue_to_lick as c2l
    lv = L[(L.variant == v) & L.ok].set_index(["label", "area"]).lev
    rows, pa = [], []
    for h, sp, ep in (("H1 chronic pre-cue all", None, "chronic"), ("H2 acute far-contra pre-cue", "far_contra", "acute")):
        t = PC[sp].copy()
        t["precue"] = t.raw - lv.reindex(pd.MultiIndex.from_frame(t[["label", "area"]])).to_numpy()
        t = t[np.isfinite(t.precue)]
        C = iam.changes(t, "precue")
        for a in HKEY:
            q = C[(C.area == a) & (C.epoch == ep)]
            g = t[t.area == a].groupby(["animal", "epoch"]).precue.mean().unstack().reindex(columns=EP)
            per = (g[ep] - g["pre"]).reindex(AN)
            r = {"variant": v, "headline": h, "area": a, "delta": np.nan, "lo": np.nan, "hi": np.nan}
            if len(q):
                r.update(delta=q.delta.iloc[0], lo=q.lo.iloc[0], hi=q.hi.iloc[0])
            r["n_same_sign"] = int(np.nansum(np.sign(per) == np.sign(r["delta"])))
            rows.append(r)
            pa.append({"variant": v, "headline": h, "area": a, **{f"d_{k}": per[k] for k in AN},
                       **{f"nsess_pre_{k}": int(((t.area == a) & (t.animal == k) & (t.epoch == "pre")).sum()) for k in AN}})
    t = C2.copy()
    t["mean_rw"] = t.raw - lv.reindex(pd.MultiIndex.from_arrays([t.label, t.col])).to_numpy()
    t = t[np.isfinite(t.mean_rw)]
    rng = np.random.default_rng(0)
    S = c2l.per_session(t, "mean_rw")
    S["cmp"] = ""
    R = c2l.deltas(S, {"match": "raw"}, rng)
    R += c2l.deltas(c2l.matched(t, "mean_rw"), {"match": "latency-matched"}, rng, cmp_col=True)
    for r in R:
        if r["pos"] == "all" and r["epoch"] == "chronic" and r["col"] in HKEY:
            rows.append({"variant": v, "headline": f"H3 chronic cue-to-lick ({r['match']})", "area": r["col"],
                         "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "n_same_sign": r["n_same_sign"]})
            pa.append({"variant": v, "headline": f"H3 chronic cue-to-lick ({r['match']})", "area": r["col"],
                       **{f"d_{k}": r.get(f"d_{k}", np.nan) for k in AN}})
    return rows, pa


# ------------------------------------------------------------------------------------------------ summary
def summarise(labels: list) -> None:
    from figdata import save_fd
    pd.set_option("display.width", 250)
    L, SV, Z = load_levels(labels)
    chk = pd.DataFrame([{"label": x, "s20_maxdiff": float(z["s20_check_maxdiff"]), "s20_nfr": int(z["s20_check_nfr"])}
                        for x, z in Z.items()])
    print(f"== CHECK s20 vs restw_check strict: max |posmed diff| {np.nanmax(chk.s20_maxdiff):.2e}, "
          f"frame-count mismatches {int(chk.s20_nfr.sum())} (expect 0 / 0)")
    # ---- 1. sessions kept per animal x epoch per variant
    K = SV[SV.ok].groupby(["variant", "animal", "epoch"]).size().unstack(["animal", "epoch"], fill_value=0)
    cols_ae = [(a, e) for a in AN for e in EP]
    K = K.reindex(columns=pd.MultiIndex.from_tuples(cols_ae), fill_value=0).reindex(vnames())
    K.columns = [f"{a}_{e[:3]}" for a, e in cols_ae]
    K["min_cell"] = K.min(1)
    K["n_sess"] = K[[c for c in K.columns if c[:2] == "PS"]].sum(1)
    fr = SV[SV.ok].groupby("variant").nfr.agg(["median", "sum"]).reindex(vnames())
    K["rest_fr_median"] = fr["median"]
    K["rest_min_total"] = fr["sum"] / 31.25 / 60
    K.to_csv(OUTD / "rescue_sessions_kept.csv")
    print("== 1. sessions kept per animal x epoch (pre/acu/sub/chr), all variants\n" + K.round(1).to_string())
    # rest frames kept per animal x epoch (median per session, mask level, no floor) for the masks
    RF = SV[SV.variant.str.endswith("_f50_w6")].assign(mask=lambda d: d.variant.str[:3])
    RF = RF.groupby(["mask", "animal", "epoch"]).nfr.median().unstack(["animal", "epoch"]).reindex(
        columns=pd.MultiIndex.from_tuples(cols_ae))
    RF.columns = [f"{a}_{e[:3]}" for a, e in cols_ae]
    RF.to_csv(OUTD / "rescue_rest_frames_median.csv")
    print("== rest frames per session (median; engaged, all positions) per mask\n" + RF.round(0).to_string())
    # ---- 2. agreement with the reference on sessions where both exist
    W = L.pivot_table(index=["label", "area"], columns="variant", values="lev")
    OK = L.pivot_table(index=["label", "area"], columns="variant", values="ok")
    ag = []
    for v in vnames():
        for a in KEY:
            sel = (W.index.get_level_values("area") == a) & OK[v].astype(bool).to_numpy() & OK[REF].astype(bool).to_numpy()
            x, y = W.loc[sel, REF].to_numpy(), W.loc[sel, v].to_numpy()
            ag.append({"variant": v, "area": a, "n": int(sel.sum()), "r": float(np.corrcoef(x, y)[0, 1]) if sel.sum() > 2
                       else np.nan, "mean_diff": float(np.mean(y - x)), "sd_diff": float(np.std(y - x)),
                       "max_abs_diff": float(np.max(np.abs(y - x)))})
    AG = pd.DataFrame(ag)
    AG.to_csv(OUTD / "rescue_agreement.csv", index=False)
    print("== 2. agreement with RESTnl (s20_f50_w6) on shared sessions: r / mean diff (variant - RESTnl), dF/F x1e3\n"
          + AG.pivot_table(index="variant", columns="area", values=["r", "mean_diff"]).reindex(vnames()).round(3)
          .to_string())
    # drop-one-position bias on reference-OK sessions (what renormalising can cost when a position is missing)
    db = []
    for x, z in Z.items():
        lev, ok, _ = variant_level(z, "s20", 50, "w6")
        if not ok:
            continue
        cs = [str(c) for c in z["cols"]]
        for k in range(6):
            l5 = np.delete(z["s20__posmed"], k, 0).mean(0)
            for a in KEY:
                db.append({"label": x, "drop": k, "area": a, "d": l5[cs.index(a)] - lev[cs.index(a)]})
    DB = pd.DataFrame(db)
    DB.to_csv(OUTD / "rescue_dropone_bias.csv", index=False)
    print("== drop-one-position bias on RESTnl-OK sessions (5-pos mean minus 6-pos mean): |d| median / 95th pct\n"
          + DB.groupby("area").d.agg(lambda s: f"{np.median(np.abs(s)):.2f} / {np.percentile(np.abs(s), 95):.2f}")
          .to_string())
    # ---- 3. which positions miss in the dropped sessions
    miss = []
    for x, z in Z.items():
        for m in ("s20", "s15", "s10", "r20"):
            miss.append({"label": x, "mask": m, **{f"p{k}": int(z[f"{m}__nfr_pos"][k]) for k in range(6)}})
    MS = pd.DataFrame(miss)
    MS.to_csv(OUTD / "rescue_frames_per_position.csv", index=False)
    dropped = sorted(SV[(SV.variant == REF) & ~SV.ok].label)
    print("== frames per position in the RESTnl-dropped sessions\n"
          + MS[MS.label.isin(dropped)].pivot_table(index="label", columns="mask",
                                                   values=[f"p{k}" for k in range(6)]).astype(int).to_string())
    # ---- 4. lick-triggered check
    lt = lick_triggered(Z, L)
    # ---- 5. headlines
    PC = precue_tables(labels)
    C2 = c2l_table(Z)
    # check the swap reproduces the published RESTnl numbers
    pub1 = pd.read_csv(il.OUT / "precue_restw_changes_restnl.csv")
    pub2 = pd.read_csv(il.OUT / "precue_restw_changes_farcontra_restnl.csv")
    pub3 = pd.read_csv(il.OUT / "cue_to_lick" / "c2l_A_changes_restnl.csv")
    pick = [v for v in vnames() if v.endswith(("_w6", "_wr", "_flat"))]
    good = K[(K.min_cell >= 1)].index
    H, PA = [], []
    for v in pick:
        if v not in good:
            continue
        h, p = headlines(v, L, PC, C2)
        H += h
        PA += p
    H = pd.DataFrame(H)
    PA = pd.DataFrame(PA)
    H.to_csv(OUTD / "rescue_headlines.csv", index=False)
    PA.to_csv(OUTD / "rescue_headlines_per_animal.csv", index=False)
    r = H[H.variant == REF].set_index(["headline", "area"]).delta
    p1 = pub1[(pub1.value == "precue") & (pub1.epoch == "chronic")].set_index("area").delta
    p2 = pub2[(pub2.value == "precue") & (pub2.epoch == "acute")].set_index("area").delta
    p3 = pub3[(pub3.measure == "mean_rw") & (pub3.pos == "all") & (pub3.epoch == "chronic")].set_index(["match", "col"]).delta
    print("== CHECK reference swap vs published RESTnl (max |diff|): "
          f"H1 {max(abs(r[('H1 chronic pre-cue all', a)] - p1[a]) for a in HKEY):.2e}  "
          f"H2 {max(abs(r[('H2 acute far-contra pre-cue', a)] - p2[a]) for a in HKEY):.2e}  "
          f"H3 {max(abs(r[('H3 chronic cue-to-lick (raw)', a)] - p3[('raw', a)]) for a in HKEY):.2e}")
    save_fd("restnl_rescue", {"K": K, "AG": AG, "H": H, "PA": PA, "lt": lt, "DB": DB}, root=OUTD / "figdata")
    report(K, H, PA)
    draw(K, H, lt)


def lick_triggered(Z: dict, L: pd.DataFrame) -> pd.DataFrame:
    """Per animal x epoch x kind x area: mean response at +1.0 / +1.5 / 2.0 / 3.0 s and the peak."""
    lv = L[(L.variant == REF)].set_index(["label", "area"]).lev
    tr = []
    for x, z in Z.items():
        cs = [str(c) for c in z["cols"]]
        for kind in ("iso", "boutend"):
            nn = int(z[f"lt_{kind}_n"])
            if nn == 0:
                continue
            X = z[f"lt_{kind}_mean"]
            for a in KEY[:4]:
                y = X[:, cs.index(a)]
                if kind == "iso":
                    y = y - y[(LAGS_T >= -1.0) & (LAGS_T <= -0.5)].mean()
                else:
                    y = y - lv[(x, a)]
                tr.append({"label": x, "animal": str(z["animal"]), "epoch": str(z["epoch"]), "kind": kind, "area": a,
                           "n": nn, **{f"t{lag:+.2f}": y[i] for i, lag in enumerate(LAGS_T)}})
    TR = pd.DataFrame(tr)
    TR.to_csv(OUTD / "rescue_lick_triggered_sessions.csv.gz", index=False)
    tc = [c for c in TR.columns if c.startswith("t")]
    # per animal (session mean, n-weighted not), then animal mean
    G = TR.groupby(["kind", "area", "animal"])[tc].mean().groupby(["kind", "area"]).mean()
    pk = G[[c for c in tc if 0 <= float(c[1:]) <= 1.0]].max(1)
    for lag in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        G[f"at_{lag}"] = G[f"t{lag:+.2f}"]
    S = G[[f"at_{lag}" for lag in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]].copy()
    S.insert(0, "peak_0_1s", pk)
    for lag in (1.0, 1.5, 2.0):
        S[f"frac_peak_{lag}"] = S[f"at_{lag}"] / S.peak_0_1s
    n = TR.groupby(["kind", "animal"]).n.sum().unstack()
    print("== 4. lick-triggered (ITI licks next to restdock05 rest), animal-mean, dF/F x1e3; iso = re pre-lick -1..-0.5 s, "
          "boutend = re RESTnl level\n   licks per animal:\n" + n.to_string() + "\n" + S.round(2).to_string())
    pe = TR.groupby(["kind", "area", "animal", "epoch"])[[f"t{lag:+.2f}" for lag in (1.0, 1.5, 2.0, 3.0)]].mean()
    print("   per animal x epoch (pre only), contra tjM1\n" + pe.xs("tjM1_right", level="area").xs("pre", level="epoch")
          .round(2).to_string())
    S.to_csv(OUTD / "rescue_lick_triggered_summary.csv")
    return G[tc].reset_index()


def report(K: pd.DataFrame, H: pd.DataFrame, PA: pd.DataFrame) -> None:
    sh = shortlist(K)
    print("== 5. SHORTLIST (sessions kept)\n" + K.loc[sh].to_string())
    for v in sh:
        print(f"-- per animal (rule 8), {v}\n" + PA[PA.variant == v].drop(columns="variant").round(2).to_string(index=False))
    P = H[H.variant.isin(sh)].copy()
    P["txt"] = [f"{d:+.2f} [{lo:+.2f},{hi:+.2f}] {n}/4{'*' if (lo > 0 or hi < 0) else ''}"
                for d, lo, hi, n in zip(P.delta, P.lo, P.hi, P.n_same_sign)]
    print("== headlines by variant (boot_delta; * = CI excludes 0)\n"
          + P.pivot_table(index=["headline", "area"], columns="variant", values="txt", aggfunc="first")
          .reindex(columns=sh).to_string())


def shortlist(K: pd.DataFrame) -> list[str]:
    want = [REF, "s15_f50_w6", "s10_f50_w6", "s20_f20_w6", "s20_f10_w6", "s15_f20_w6", "s20_f50_wr", "s20_f20_wr",
            "s15_f20_wr", "s20_f50_flat", "s15_f50_flat", "r20_f50_w6"]
    return [v for v in want if v in K.index]


def draw(K: pd.DataFrame, H: pd.DataFrame, lt: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sh = [v for v in shortlist(K) if v in set(H.variant)]
    acol = {"PS92": "#1f77b4", "PS93": "#d62728", "PS94": "#2ca02c", "PS95": "#9467bd"}
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, :2])
    for j, a in enumerate(AN):
        for i, ep in enumerate(EP):
            y = K.loc[sh, f"{a}_{ep[:3]}"].to_numpy()
            ax.plot(np.arange(len(sh)) + (j * 4 + i - 7.5) * 0.045, y, "o", color=acol[a], ms=3.5,
                    alpha=0.4 + 0.2 * i, label=a if i == 0 else None)
    ax.axhline(3, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(range(len(sh)), sh, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("sessions kept per animal x epoch\n(dots per animal: pre, acute, subacute, chronic, darker = later)")
    ax.set_title("Fig 1. Sessions kept per variant (dashed = 3)", fontsize=9)
    ax.legend(fontsize=7, ncol=4)
    ps = [K.loc[v, "PS93_pre"] for v in sh]
    for i, p in enumerate(ps):
        ax.annotate(f"{p}", (i, -0.2), ha="center", fontsize=7, color=acol["PS93"], annotation_clip=False)
    ax2 = fig.add_subplot(gs[0, 2:])
    for kind, ls in (("iso", "-"), ("boutend", "--")):
        for a, c in (("tjM1_right", "k"), ("ALM_right", "#888"), ("SSp-m_right", "#d62728"), ("SSp-n_right", "#ff7f0e")):
            q = lt[(lt.kind == kind) & (lt.area == a)]
            if not len(q):
                continue
            tc = [c_ for c_ in q.columns if c_.startswith("t")]
            ax2.plot([float(t[1:]) for t in tc], q[tc].to_numpy()[0], ls=ls, color=c, lw=1.2,
                     label=f"{a.replace('_right', ' contra')} {kind}")
    for x in (1.0, 1.5, 2.0):
        ax2.axvline(x, color="0.7", lw=0.8, ls=":")
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_xlabel("time from lick (s); bout end = last lick of the bout")
    ax2.set_ylabel("dF/F x1e3 (iso: re -1..-0.5 s; bout end: re RESTnl)")
    ax2.set_title("Fig 2. Lick-triggered ITI licks next to rest (animal mean)", fontsize=9)
    ax2.legend(fontsize=6, ncol=2)
    hs = [("H1 chronic pre-cue all", "Fig 3. H1 chronic contra pre-cue rise (all spouts)"),
          ("H2 acute far-contra pre-cue", "Fig 4. H2 acute far-contra pre-cue raise"),
          ("H3 chronic cue-to-lick (raw)", "Fig 5. H3 chronic cue-to-lick level (raw)"),
          ("H3 chronic cue-to-lick (latency-matched)", "Fig 6. H3 chronic cue-to-lick (latency-matched)")]
    vc = plt.get_cmap("tab20")(np.linspace(0, 1, len(sh)))
    for k, (h, ttl) in enumerate(hs):
        ax = fig.add_subplot(gs[1, k])
        for i, a in enumerate(HKEY):
            for j, v in enumerate(sh):
                q = H[(H.variant == v) & (H.headline == h) & (H.area == a)]
                if not len(q) or not np.isfinite(q.delta.iloc[0]):
                    continue
                x = i + (j - (len(sh) - 1) / 2) * 0.065
                d = q.delta.iloc[0]
                ax.errorbar(x, d, yerr=[[d - q.lo.iloc[0]], [q.hi.iloc[0] - d]], fmt="o", ms=3, color=vc[j], capsize=0,
                            lw=0.8, label=v if (i == 0 and k == 0) else None)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(HKEY)), [a.replace("_right", "\ncontra") for a in HKEY], fontsize=7)
        ax.set_title(ttl, fontsize=8)
        if k == 0:
            ax.set_ylabel("change vs pre (dF/F x1e3), 95% CI")
            ax.legend(fontsize=5.5, ncol=2, loc="upper left")
    fig.suptitle("NEURAL. RESTnl rescue sweep: sessions kept, lick-triggered decay, and the three headline changes per "
                 "rest-baseline variant (baseline swap; boot_delta, nested animals > sessions). n = 4 mice, PROVISIONAL.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTD / "restnl_rescue.png", dpi=130)
    plt.close(fig)
    print("== figure", OUTD / "restnl_rescue.png")


def main() -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import parallel
    parallel.pin_blas()
    OUTD.mkdir(parents=True, exist_ok=True)
    labels = ([x for x in sys.argv[sys.argv.index("--labels") + 1:] if not x.startswith("--")]
              if "--labels" in sys.argv else il.labels())
    if "--summarise" not in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
        _res, fails = ak.fan_sessions(labels, session_worker, jobs=min(jobs, 4))
        if fails:
            print(f"!! {len(fails)} sessions failed: {fails}")
    done = sorted(p.stem for p in SESS.glob("*.npz") if p.stem in set(labels))
    if len(done) < len(labels):
        print(f"!! only {len(done)}/{len(labels)} sessions extracted")
    summarise(done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
