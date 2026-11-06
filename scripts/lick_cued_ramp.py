"""CUED versions of the ITI lick-response figures 14.1 / 14.3 / 14.4 (Priya 10-28: "yes add the cued fig 14 versions";
"can we show 2 s before to 3 s after?", "for cue-aligned also -2 to +3", "show a supplemental version per spout position").

The FIRST post-cue lick of each engaged trial, from the same session files, ROIs, baselines, tests and drawing helpers as the
ITI figures (`iti_licks.key_figures` / `ramp_compute`). The stored traces (`first_peth` -1.5..+0.75 s around the lick,
`first_cue` -1..+2.5 s around the cue) are too short for a -2..+3 s view, so `--extract` RE-SAMPLES them per session into
<OUT_W>/cued_wide/<label>.npz with exactly the steps of `iti_licks.extract` (ROI v8 masks, area means of U, drift basis
removed, x1e3; times = the stored first licks, rebuilt by `iti_licks._first_lick_times` and matched to first_rt) on LAGS_W
-2..+3 s, lick- AND cue-aligned, with each trial's spout. Every session is VALIDATED: the re-sampled trace at the stored
lags must equal the stored `first_peth` (float16 tolerance) -- the max |difference| is saved and printed.
Sessions with >= MIN_N first licks; per-spout cells >= MIN_SPOUT licks.
  CAVEAT (two alignments): the first cued lick comes ~0.2-0.5 s after the cue, so the lick-aligned pre-lick window
  (-0.4..-0.05 s) overlaps the CUE response -- not a self-paced build-up. Baselines: lick-aligned -1.5..-1.0 s before the
  lick (before the cue for reaction times < 1 s, the large majority); cue-aligned -1.0..-0.5 s before the cue (the spout has
  been in place ~2-3 s).
Also (Priya 10-28): row 3 of each PETH figure = difference from the animal's own pre curve (/ its pre peak); a masked
RATIO-to-pre supplement; a response-WINDOW sensitivity (0..0.3 / 0..0.5 / 0..1.0 / 0..1.5 s, peak) for cued and ITI licks;
the ITI bout starts on the same -2..+3 s window (lick_iti_ramp_wide.png, validated against the stored bout_peth).
Figures (figdata + --plot-only):
  lick_cued_ramp_sessions.png         lick-aligned per-session PETH -2..+3 s, mean +- SEM over sessions; row 2 / the
                                      animal's PRE response AUC; per-bin tests (sig bars, -2..+3 s)
  lick_cued_ramp_cue_sessions.png     the same cue-aligned (row 2 / the animal's pre cue-response AUC)
  lick_cued_ramp_summary.png          per-session AUCs: lick response 0..0.5 s, pre-lick window, cue response 0..0.5 s
  lick_cued_ramp_perspout.png         SUPPLEMENT: lick-aligned, per spout (columns) x key ROI (rows), / the animal's
                                      pre response AUC (all spouts), mean +- SEM over sessions
  lick_cued_ramp_cue_perspout.png     the same cue-aligned
  lick_iti_ramp_perspout.png          ITI bout starts by the PREVIOUS trial's spout
  (per-spout figures: per-bin sig bars as 14.1, tested against the animal's pre curve at that spout; 10-28)
Outputs <E>/iti_licks/cued_ramp_auc_sessions.csv, cued_ramp_session_tests.csv, sig_bins_cued_ramp.csv,
sig_bins_perspout.csv, perspout_auc_tests.csv (per-spout AUC 0..0.5 s / pre at that spout, change vs pre).
Usage: python scripts/lick_cued_ramp.py --extract [--jobs N]; python scripts/lick_cued_ramp.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

MIN_N = 10
MIN_SPOUT = 5
LAGS_W = np.round(np.arange(-2.0, 3.0001, 0.05), 3)
BASE_L = il.BASE                      # lick-aligned baseline (s from lick), as the ITI figures
BASE_C = (-1.0, -0.5)                 # cue-aligned baseline (s from cue)
BASE_SPOUT = (-1.0, -0.5)             # per-spout lick-aligned figures, cued AND ITI (Priya 10-28: "use -1 to -0.5 s")
# per-spout AUC windows (Priya 10-28: "a longer AUC might make a difference ... the far contra acute peak decreases
# earlier"; "what if we do 0-1s"): 0..0.5 (as before), 0..1.0, and the late part 0.5..1.0 alone
SPOUT_WINDOWS = ((0.0, 0.5), (0.0, 1.0), (0.5, 1.0))
BASE_RESP = il.BASE_RESP              # response-SIZE AUCs (14.10, 14.11); PETH curves + build-up keep BASE_L
ROIS = il.KEY_ROIS
SPOUT_ROIS = ["ALM_right", "tjM1_right", "SSp-m_right", "cortex"]
WIDE = "cued_wide"
# READ root for the derived caches (cued_wide/, bout_prev_pos.csv): OUT_W as before, but il.OUT in a SUBSET run (dock,
# ...; 10-29), whose OUT_W is the subset's own output folder (the caches are not subset-specific)
RIN = il.OUT if il.ITI_SUBSET else il.OUT_W
RESP_W = (0.0, 1.0)                   # response-size AUC window (Priya 10-28 evening: "0-1 s should be default now")
SPEC = (("resp_auc", "lick response AUC 0..1.0 s\n(dF/F x1e3 * s; baseline -1.0..-0.5 s)"),
        ("resp_auc_n", "lick response AUC 0..1.0 s / animal's pre\n(baseline -1.0..-0.5 s)"),
        ("pre_auc", "pre-lick window AUC -0.4..-0.05 s\n(CUE-contaminated; baseline -1.5..-1.0 s)"),
        ("cue_auc", "cue response AUC 0..1.0 s after cue"),
        ("cue_auc_n", "cue response AUC 0..1.0 s\n/ animal's pre cue AUC"))
SPEC_ITI = (("resp_auc", "response AUC 0..1.0 s (dF/F x1e3 * s)\nbaseline -1.0..-0.5 s"),
            ("resp_auc_n", "response AUC 0..1.0 s / animal's pre\n(baseline -1.0..-0.5 s)"),
            ("build_auc", "build-up AUC -0.4..-0.05 s\n(baseline -1.5..-1.0 s)"),
            ("build_auc_n", "build-up AUC / animal's pre\n0..0.5 s response AUC"))


# ------------------------------------------------------------------------------------------------------------ EXTRACT
def wide_session(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/cued_wide/<label>.npz; returns the validation numbers."""
    f = il.OUT_W / WIDE / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if "iti" in z.files:                              # files from before the ITI arrays (10-28) are rebuilt
            return {"label": label, "cached": True, "n": int(len(z["pos"])), "maxdiff": float(z["maxdiff"]),
                    "maxdiff_iti": float(z["maxdiff_iti"])}
    from wfield_local import combined_encoding as ce
    d = il._load(label)
    cue, fl = il._first_lick_times(label, d["first_rt"])
    # == iti_licks.extract's trace construction (keep in step with it; validated below against the stored first_peth)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise ValueError(f"{label}: ROI columns differ from the session file")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    ok = np.isfinite(fl) & np.isfinite(cue)
    chk = il._sample(R, ft, fl[ok], il.LAGS_P)
    ref = np.asarray(d["first_peth"], float)[ok]
    maxdiff = float(np.nanmax(np.abs(chk - ref))) if ok.any() else np.nan
    scale = float(np.nanmax(np.abs(ref))) if ok.any() else np.nan

    def samp(t):
        X = il._sample(R, ft, t, LAGS_W)
        out = (t[:, None] + LAGS_W[None, :] < ft[0]) | (t[:, None] + LAGS_W[None, :] > ft[-1])
        X[out] = np.nan                                   # no clamped values at the recording edges
        return X.astype(np.float16)

    # ITI bout starts (Priya 10-28: the longer response windows for the ITI figures too): every stored bout start, with
    # its gap and following-lick count so the figure applies the same selection as 14.1 (gap >= GAP_MAIN)
    bt = np.asarray(d["bout_t"], float)
    bchk = il._sample(R, ft, bt, il.LAGS_P)
    maxdiff_iti = float(np.nanmax(np.abs(bchk - np.asarray(d["bout_peth"], float)))) if len(bt) else np.nan
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, lick=samp(fl[ok]), cue=samp(cue[ok]), pos=np.asarray(d["first_pos"], str)[ok],
                        iti=samp(bt), iti_gap=np.asarray(d["bout_gap"], float), iti_nfol=np.asarray(d["bout_nfol"]),
                        cols=np.array(cols), lags=LAGS_W, maxdiff=maxdiff, maxdiff_iti=maxdiff_iti, scale=scale,
                        n_unmatched=int((~ok).sum()))
    return {"label": label, "cached": False, "n": int(ok.sum()), "maxdiff": maxdiff, "maxdiff_iti": maxdiff_iti,
            "scale": scale, "n_unmatched": int((~ok).sum())}


def extract(jobs: int | None) -> None:
    from wfield_local import analysis_kit as ak
    labs = il.labels()
    res, fails = ak.fan_sessions(labs, wide_session, jobs=jobs)
    V = pd.DataFrame([v for _x, v in res])
    print(V.to_string(index=False))
    print(f"== validation ITI: max |re-sampled - stored bout_peth| = {V.maxdiff_iti.max():.4f}")
    print(f"== validation: max |re-sampled - stored first_peth| = {V.maxdiff.max():.4f} (dF/F x1e3; traces span up to "
          f"{V.get('scale', pd.Series([np.nan])).max():.1f}); unmatched first licks {int(V.get('n_unmatched', pd.Series([0])).sum())}")
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")


# ------------------------------------------------------------------------------------------------------------ COMPUTE
# three trace sets: cued first lick (lick-aligned), the same trials cue-aligned, ITI bout starts (gap >= GAP_MAIN, as 14.1)
TSETS = {"cued": ("lick", BASE_L, "time from first cued lick onset (s)"),
         "cue": ("cue", BASE_C, "time from cue (s)"),
         "iti": ("iti", BASE_L, "time from ITI bout-start lick onset (s)")}
# response-window sensitivity (Priya 10-28: "add a longer 0-1.5 also"); peak = max of the session PETH in 0..1.5 s
WINDOWS = ((0.0, 0.3), (0.0, 0.5), (0.0, 1.0), (0.0, 1.5))
RATIO_MIN = 0.2       # masked ratio: shown only where the pre curve is >= 20 % of its peak


def _auc(v: np.ndarray, lags: np.ndarray, t0: float, t1: float) -> float:
    m = (lags >= t0 - 1e-9) & (lags <= t1 + 1e-9)
    return float(np.trapezoid(v[m], lags[m]))


_LT = None


PREV_CACHE = "bout_prev_pos.csv"     # derived: label, bout_t, prev_pos (built once by --prev-cache, 8 workers)
_PC = None


def bout_prev_pos(label: str, bout_t: np.ndarray) -> np.ndarray:
    """PREVIOUS trial's spout of each stored ITI bout start, from the derived cache (PREV_CACHE) when it covers the
    session -- the rebuild loads the session's behaviour off the share (~46 s per session; 10-28: three jobs x 107
    sessions spent ~80 min each on it). Falls back to the rebuild."""
    global _PC
    if _PC is None:
        f = RIN / PREV_CACHE
        _PC = {x: dict(zip(g.bout_t.to_numpy(float), g.prev_pos.fillna("").astype(str)))
               for x, g in pd.read_csv(f, float_precision="round_trip").groupby("label")} if f.exists() else {}
    if label in _PC:
        m = _PC[label]
        return np.array([m.get(float(t), "") for t in bout_t], dtype=object)
    return _bout_prev_pos_raw(label, bout_t)


def _prev_worker(label: str) -> dict:
    t = np.asarray(il._load(label)["bout_t"], float)
    return {"label": label, "bout_t": t, "prev_pos": _bout_prev_pos_raw(label, t)}


def prev_cache(jobs: int) -> None:
    """Build PREV_CACHE for every session in parallel (rule 6). bout_t is stored with repr precision (exact match)."""
    from wfield_local import analysis_kit as ak
    res, fails = ak.fan_sessions(il.labels(), _prev_worker, jobs=jobs)
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    C = pd.concat([pd.DataFrame({"label": v["label"], "bout_t": v["bout_t"], "prev_pos": v["prev_pos"]})
                   for _x, v in res], ignore_index=True)
    C.to_csv(il.OUT_W / PREV_CACHE, index=False, float_format="%.17g")
    print(f"== {PREV_CACHE}: {len(C)} bout starts, {C.label.nunique()} sessions, "
          f"unlabelled {(C.prev_pos.fillna('') == '').mean():.3f}")


def _bout_prev_pos_raw(label: str, bout_t: np.ndarray) -> np.ndarray:
    """PREVIOUS trial's spout of each stored ITI bout start (the stored `bout_pos` is the trial-WINDOW label, = the
    UPCOMING trial for 'lead' licks). Rebuilt exactly as `iti_licks.extract` builds prev_pos (lick table joined to the
    session's licks_all.csv by lick_uid; tail -> trial_id, lead -> trial_id - 1), matched on the exact onset time."""
    global _LT
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    if _LT is None:
        _LT = il.lick_table()
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    K = _LT[_LT.session == f"{an}_{date}"].copy()
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["context", "segment", "daq_on_s"])
    uid = K.lick_uid.to_numpy(int)
    K["daq_on_s"] = A.daq_on_s.to_numpy(float)[uid]
    T = il._behaviour(label).trials
    seg = A.segment.astype(str).to_numpy()[uid]
    pot = dict(zip(T.trial_id.to_numpy(), T.pos.astype(str).to_numpy()))
    prev = np.where(seg == "lead", K.trial_id.to_numpy() - 1, K.trial_id.to_numpy())
    K["prev_pos"] = [pot.get(t_, "") for t_ in prev]
    m = dict(zip(K.daq_on_s.to_numpy(float), K.prev_pos))
    return np.array([m.get(float(t), "") for t in bout_t], dtype=object)


def _sessions() -> dict:
    D = {}
    for x in il.labels():
        f = RIN / WIDE / f"{x}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}: run --extract first")
        z = np.load(f)
        if "iti" not in z.files:
            raise SystemExit(f"{f} predates the ITI arrays: re-run --extract")
        d = il._load(x)
        keep = z["iti_gap"] >= il.GAP_MAIN
        if il.ITI_SUBSET:                               # docked-spout etc. subset of the bout starts (10-29)
            keep &= il.bout_subset_keep(x, d["bout_t"])
        ipos = bout_prev_pos(x, np.asarray(d["bout_t"], float))
        D[x] = {"lick": z["lick"].astype(float), "cue": z["cue"].astype(float), "pos": z["pos"],
                "iti": z["iti"].astype(float)[keep], "iti_pos": ipos[keep], "cols": list(z["cols"]),
                "iti_iso": z["iti"].astype(float)[keep & (np.asarray(z["iti_nfol"]) == 0)],
                "animal": str(d["animal"]), "epoch": str(d["epoch"])}
    n = sum(len(d["iti_pos"]) for d in D.values())
    print(f"== ITI bout starts with a previous-trial spout: "
          f"{sum((d['iti_pos'] != '').sum() for d in D.values())} / {n}")
    return D


def _pe(D: dict, key: str, base: tuple, rois: list, pos: str | None = None, min_n: int = MIN_N) -> pd.DataFrame:
    rows = []
    for x, d in D.items():
        P = d[key] if pos is None else d[key][d["iti_pos" if key == "iti" else "pos"] == pos]
        if len(P) < min_n:
            continue
        P = P - np.nanmean(P[:, (LAGS_W >= base[0]) & (LAGS_W <= base[1]), :], 1, keepdims=True)
        m = np.nanmean(P, 0)
        for roi in rois:
            if roi in d["cols"]:
                rows.append({"label": x, "animal": d["animal"], "epoch": d["epoch"], "roi": roi,
                             "peth": m[:, d["cols"].index(roi)], "n": len(P)})
    return pd.DataFrame(rows)


def _pre_curve(PE: pd.DataFrame) -> dict:
    """{(roi, animal): the animal's mean PRE session curve} (raw, baseline-subtracted)."""
    pre = PE[PE.epoch == "pre"]
    return {k: np.nanmean(np.stack(g.peth.to_numpy()), 0) for k, g in pre.groupby(["roi", "animal"])}


def _curves(PE: pd.DataFrame, den: dict, rois: list, test: bool = True) -> tuple[dict, list]:
    """Rows: 0 raw; 1 / the animal's pre response AUC (0..0.5 s); 2 DIFFERENCE from the animal's own pre curve, in units
    of that pre curve's peak (0..1.5 s) -- the per-time change the significance bars test (Priya 10-28)."""
    pc = _pre_curve(PE) if len(PE) else {}
    pk = {k: np.nanmax(v[(LAGS_W >= 0) & (LAGS_W <= 1.5)]) for k, v in pc.items()}
    panels, recs = {}, []
    for j, roi in enumerate(rois):
        for i in range(3):
            long, curves = [], []
            for ep in il.EP4:
                t = PE[(PE.roi == roi) & (PE.epoch == ep)] if len(PE) else PE
                if t.empty:
                    continue
                M = np.stack(t.peth.to_numpy())
                if i == 1:
                    M = M / np.array([den.get((roi, a), np.nan) for a in t.animal])[:, None]
                elif i == 2:
                    M = np.stack([(m_ - pc.get((roi, a), np.nan)) / pk.get((roi, a), np.nan)
                                  for m_, a in zip(M, t.animal)])
                mu = np.nanmean(M, 0)
                se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.isfinite(M[:, 0]).sum())
                curves.append((ep, mu, se, len(t)))
                long += [pd.DataFrame({"label": x, "animal": a, "epoch": ep, "lag": LAGS_W, "v": m_})
                         for x, a, m_ in zip(t.label, t.animal, M)]
            S = il.sig_compute(pd.concat(long, ignore_index=True).dropna(), value="v",
                               tag={"roi": roi, "row": i}) if (long and test and i < 2) else None
            if S is not None:
                recs += S["recs"]
            panels[(i, j)] = {"roi": roi, "row": i, "curves": curves, "sig": S}
        # row 2 carries row 1's bins (same per-bin test: a change vs pre)
        panels[(2, j)]["sig"] = panels[(1, j)]["sig"]
    return panels, recs


def _ratio(PE: pd.DataFrame, rois: list) -> dict:
    """Masked RATIO to pre per time (supplement): mean over animals of (epoch curve / pre curve), each animal's curves its
    session means; shown only where that animal's pre curve >= RATIO_MIN x its peak (near-zero pre would blow up)."""
    pc = _pre_curve(PE)
    out = {}
    for roi in rois:
        for ep in il.EP4[1:]:
            R = []
            for a, g in PE[(PE.roi == roi) & (PE.epoch == ep)].groupby("animal"):
                if (roi, a) not in pc:
                    continue
                pre = pc[(roi, a)]
                pk = np.nanmax(pre[(LAGS_W >= 0) & (LAGS_W <= 1.5)])
                r = np.nanmean(np.stack(g.peth.to_numpy()), 0) / pre
                R.append(np.where(pre >= RATIO_MIN * pk, r, np.nan))
            if R:
                R = np.asarray(R)
                n = np.isfinite(R).sum(0)
                mu = np.where(n == len(R), np.nanmean(R, 0), np.nan)   # every animal (no composition steps)
                se = np.where(n == len(R), np.nanstd(R, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
                out[(roi, ep)] = (mu, se, len(R))
    return out


def _windows(PE: pd.DataFrame, rng) -> tuple[dict, list]:
    """Response-window sensitivity: per session AUC over each window (and the 0..1.5 s peak), / the animal's pre mean of
    the same measure; nested-bootstrap change vs pre (il.dots_compute)."""
    rows = []
    for _, r in PE.iterrows():
        v = r.peth
        rec = {"label": r.label, "animal": r.animal, "epoch": r.epoch, "roi": r.roi}
        for t0, t1 in WINDOWS:
            rec[f"auc_{t0:g}_{t1:g}"] = _auc(v, LAGS_W, t0, t1)
        rec["peak"] = float(np.nanmax(v[(LAGS_W >= 0) & (LAGS_W <= 1.5)]))
        rows.append(rec)
    T0 = pd.DataFrame(rows)
    vals = [f"auc_{t0:g}_{t1:g}" for t0, t1 in WINDOWS] + ["peak"]
    for v in vals:
        pre = T0[T0.epoch == "pre"].groupby(["roi", "animal"])[v].mean()
        T0[v + "_n"] = T0[v] / np.array([pre.get((r_, a_), np.nan) for r_, a_ in zip(T0.roi, T0.animal)])
    panels, tests = {}, []
    for j, roi in enumerate(ROIS):
        T = T0[T0.roi == roi]
        for i, v in enumerate(vals):
            S = il.dots_compute(T, v + "_n", rng)
            panels[(i, j)] = S
            tests += [{"roi": roi, **r_} for r_ in S["tests"]]
    return {"panels": panels, "vals": vals}, tests


def compute() -> dict:
    import iti_lick_order as lo
    D = _sessions()
    PE = {k: _pe(D, key, base, ROIS) for k, (key, base, _x) in TSETS.items()}
    PL, PC = PE["cued"], PE["cue"]
    rows = []
    br = (LAGS_W >= BASE_RESP[0]) & (LAGS_W <= BASE_RESP[1])
    for (_i, a), (_j, c) in zip(PL.iterrows(), PC.iterrows()):
        assert a.label == c.label and a.roi == c.roi
        # response size on BASE_RESP (Priya 10-28); the pre-lick window stays on BASE_L (build-up rule)
        # response / cue AUC over RESP_W (0..1.0 s since 10-28 evening, Priya: "0-1 s should be default now"); the
        # 0..0.5 s values are kept beside them (*_05) and still scale the PETH curves (14.5 / 14.6 row 2)
        rows.append({"label": a.label, "animal": a.animal, "epoch": a.epoch, "roi": a.roi, "n": a.n,
                     "resp_auc": _auc(a.peth - np.nanmean(a.peth[br]), LAGS_W, *RESP_W),
                     "resp_auc_05": _auc(a.peth - np.nanmean(a.peth[br]), LAGS_W, 0.0, 0.5),
                     "resp_auc_b15": _auc(a.peth, LAGS_W, 0.0, 0.5), "pre_auc": _auc(a.peth, LAGS_W, -0.4, -0.05),
                     "cue_auc": _auc(c.peth, LAGS_W, *RESP_W), "cue_auc_05": _auc(c.peth, LAGS_W, 0.0, 0.5)})
    T0 = pd.DataFrame(rows)
    # the PETH curves (14.5 / 14.6 row 2) keep BASE_L and the 0..0.5 s scale
    dr = T0[T0.epoch == "pre"].groupby(["roi", "animal"]).resp_auc_b15.mean()
    drr = T0[T0.epoch == "pre"].groupby(["roi", "animal"]).resp_auc.mean()
    dc = T0[T0.epoch == "pre"].groupby(["roi", "animal"]).cue_auc_05.mean()
    dcw = T0[T0.epoch == "pre"].groupby(["roi", "animal"]).cue_auc.mean()
    T0["resp_auc_n"] = T0.resp_auc / np.array([drr.get((r_, a_), np.nan) for r_, a_ in zip(T0.roi, T0.animal)])
    T0["cue_auc_n"] = T0.cue_auc / np.array([dcw.get((r_, a_), np.nan) for r_, a_ in zip(T0.roi, T0.animal)])
    T0.to_csv(il.OUT_W / "cued_ramp_auc_sessions.csv", index=False)
    # ITI summary on the wide traces (the stored ITI PETHs end at +0.75 s, too short for 0..1.0 s): response over RESP_W
    # (baseline BASE_RESP, / the animal's pre response over RESP_W) and the build-up exactly as 14.10 (baseline BASE_L,
    # AUC -0.4..-0.05 s, / the pre 0..0.5 s response AUC under BASE_L)
    ri = []
    for _i, a in PE["iti"].iterrows():
        ri.append({"label": a.label, "animal": a.animal, "epoch": a.epoch, "roi": a.roi,
                   "resp_auc": _auc(a.peth - np.nanmean(a.peth[br]), LAGS_W, *RESP_W),
                   "resp_auc_b15": _auc(a.peth, LAGS_W, 0.0, 0.5), "build_auc": _auc(a.peth, LAGS_W, -0.4, -0.05)})
    TI = pd.DataFrame(ri)
    pre_i = TI[TI.epoch == "pre"].groupby(["roi", "animal"])
    ki = list(zip(TI.roi, TI.animal))
    TI["resp_auc_n"] = TI.resp_auc / np.array([pre_i.resp_auc.mean().get(k, np.nan) for k in ki])
    TI["build_auc_n"] = TI.build_auc / np.array([pre_i.resp_auc_b15.mean().get(k, np.nan) for k in ki])
    TI.to_csv(il.OUT_W / "iti_ramp_auc_sessions_wide.csv", index=False)
    rng_i = np.random.default_rng(3)
    tests_i, sp_i = [], {}
    for j, roi in enumerate(ROIS):
        for i, (val, _yl) in enumerate(SPEC_ITI):
            S_ = il.dots_compute(TI[TI.roi == roi], val, rng_i)
            sp_i[(i, j)] = S_
            tests_i += [{"roi": roi, **r_} for r_ in S_["tests"]]
    pd.DataFrame(tests_i).to_csv(il.OUT_W / "iti_ramp_session_tests_wide.csv", index=False)
    print("== ITI summary (response 0..1.0 s), change vs pre\n" + pd.DataFrame(tests_i).round(3).to_string(index=False))
    pi = PE["iti"]
    di = {(r_, a_): _auc(np.nanmean(np.stack(g.peth.to_numpy()), 0), LAGS_W, 0.0, 0.5)
          for (r_, a_), g in pi[pi.epoch == "pre"].groupby(["roi", "animal"])}
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): cued lick response AUC / pre, contralesional tjM1 + SSp-m")
    print(T0[T0.roi.isin(["tjM1_right", "SSp-m_right"])].pivot_table(index=["roi", "animal"], columns="epoch",
                                                                     values="resp_auc_n").reindex(columns=il.EP4)
          .round(2).to_string())
    rng = np.random.default_rng(1)
    tests, sp_ = [], {}
    for j, roi in enumerate(ROIS):
        T = T0[T0.roi == roi]
        for i, (val, _ylab) in enumerate(SPEC):
            S = il.dots_compute(T, val, rng)
            sp_[(i, j)] = S
            tests += [{"roi": roi, **r_} for r_ in S["tests"]]
    TS = pd.DataFrame(tests)
    TS.to_csv(il.OUT_W / "cued_ramp_session_tests.csv", index=False)
    print("== change vs pre (nested bootstrap)\n" + TS.round(3).to_string(index=False))
    sess, recs = {}, []
    for k, den in (("cued", dr.to_dict()), ("cue", dc.to_dict()), ("iti", di)):
        sess[k], r_ = _curves(PE[k], den, ROIS)
        recs += [{"set": k, **x} for x in r_]
    pd.DataFrame(recs).to_csv(il.OUT_W / "sig_bins_cued_ramp.csv", index=False)
    ratio = {k: _ratio(PE[k], ROIS) for k in TSETS}
    win, wt = {}, []
    for k in ("cued", "iti"):
        # response-size windows on BASE_RESP (10-28)
        win[k], t_ = _windows(PE[k].assign(peth=[v - np.nanmean(v[br]) for v in PE[k].peth]), np.random.default_rng(2))
        wt += [{"set": k, **x} for x in t_]
    WT = pd.DataFrame(wt)
    WT.to_csv(il.OUT_W / "ramp_window_tests.csv", index=False)
    print("== response-window sensitivity (change vs pre of value / animal's pre), contralesional tjM1\n"
          + WT[WT.roi == "tjM1_right"].round(3).to_string(index=False))
    spout, srecs, auc_t = {}, [], []
    for key, base in (("lick", BASE_SPOUT), ("cue", BASE_C), ("iti", BASE_SPOUT)):
        # normaliser = the animal's pre response AUC over ALL spouts under the SAME baseline
        pa = _pe(D, key, base, SPOUT_ROIS)
        den = {(r_, a_): _auc(np.nanmean(np.stack(g.peth.to_numpy()), 0), LAGS_W, 0.0, 0.5)
               for (r_, a_), g in pa[pa.epoch == "pre"].groupby(["roi", "animal"])}
        for p in lo.POS:
            P_ = _pe(D, key, base, SPOUT_ROIS, pos=p, min_n=MIN_SPOUT)
            # per-bin tests as 14.1 (Priya 10-28: "significance for the per-spout figures")
            pan, r_ = _curves(P_, den, SPOUT_ROIS)
            srecs += [{"set": key, "spout": p, **x} for x in r_ if x.get("row") == 1]
            spout[(key, p)] = {j: pan[(1, j)] for j in range(len(SPOUT_ROIS))}
            auc_t += _spout_auc_tests(P_, key, p)
    pd.DataFrame(srecs).to_csv(il.OUT_W / "sig_bins_perspout.csv", index=False)
    AT = pd.DataFrame(auc_t)
    AT.to_csv(il.OUT_W / "perspout_auc_tests.csv", index=False)
    print("== per-spout response AUC 0..0.5 s / the animal's pre AUC AT THAT SPOUT, change vs pre (* = CI excludes 0)\n"
          + AT.pivot_table(index=["set", "roi", "spout", "window"], columns="epoch", values="cell", aggfunc="first")
          .reindex(columns=il.EP4[1:]).to_string())
    return dict(sess=sess, ratio=ratio, win=win, summary=sp_, spout=spout, pos=list(lo.POS), auc_tests=AT,
                summary_iti=sp_i)


def _draw_spout_windows(F: dict, key: str, what: str, path: Path) -> None:
    """Per-spout AUC change vs pre at the spout (AUC / the animal's pre AUC at that spout), rows = SPOUT_WINDOWS,
    columns = SPOUT_ROIS, x = spout, one marker per epoch with its 95% CI; hollow = CI includes 0 (Priya 10-28: "where is
    the longer window AUC")."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    AT, pos = F["auc_tests"], F["pos"]
    wins = [f"{w0:g}-{w1:g}" for w0, w1 in SPOUT_WINDOWS]
    fig, axs = plt.subplots(len(wins), len(SPOUT_ROIS), figsize=(20, 11), sharey="row", squeeze=False)
    for r, w in enumerate(wins):
        for c, roi in enumerate(SPOUT_ROIS):
            ax = axs[r, c]
            t = AT[(AT.set == key) & (AT.roi == roi) & (AT.window == w)]
            for k, ep in enumerate(il.EP4[1:]):
                q = t[t.epoch == ep].set_index("spout").reindex(pos)
                x = np.arange(len(pos)) + (k - 1) * 0.2
                ax.errorbar(x, q.delta, yerr=[q.delta - q.lo, q.hi - q.delta], fmt="none", ecolor=colors[ep], capsize=2)
                sg = q.sig.fillna(False).astype(bool).to_numpy()
                ax.plot(x[sg], q.delta[sg], "o", color=colors[ep], ms=5.5, zorder=3)
                ax.plot(x[~sg], q.delta[~sg], "o", mec=colors[ep], mfc="white", ms=5.5, zorder=3)
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(len(pos)), [p.replace("_", "\n") for p in pos], fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(roi, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"AUC {w.replace('-', '..')} s\nchange vs pre at the spout\n(/ pre AUC at that spout; 0 = no "
                              "change)", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], color=colors[e], marker="o", lw=0, label=e) for e in il.EP4[1:]]
               + [Line2D([], [], color="0.4", marker="o", mfc="white", lw=0, label="hollow = CI includes 0")],
               loc="lower center", ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle(f"PER-SPOUT response size by WINDOW, {what} (baseline -1.0..-0.5 s): per session AUC / the animal's mean "
                 "PRE AUC at the same spout, change vs pre, mean of animals with 95% paired nested bootstrap CI.\nRows: "
                 "0..0.5 s, 0..1.0 s, and the late part 0.5..1.0 s alone. _right = contralesional. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_png_svg(fig, path, dpi=100)
    plt.close(fig)


def _spout_auc_tests(PE: pd.DataFrame, key: str, p: str) -> list[dict]:
    """Per spout: response AUC (0..0.5 s after the lick / cue) / the animal's mean PRE AUC AT THAT SPOUT; change vs pre per
    epoch, `analysis_kit.boot_delta` (paired animals > sessions), animals with both pre and that epoch, >= 3 required.
    Prints the per-animal table first (rule 8)."""
    from wfield_local import analysis_kit as ak
    if PE.empty:
        return []
    out = []
    for (w0, w1), roi in [(w, r) for w in SPOUT_WINDOWS for r in SPOUT_ROIS]:
        g = PE[PE.roi == roi].assign(auc=lambda t, w0=w0, w1=w1: [_auc(v, LAGS_W, w0, w1) for v in t.peth])
        pre = g[g.epoch == "pre"].groupby("animal").auc.mean()
        g = g[g.animal.isin(pre.index)].copy()
        g["r"] = g.auc / g.animal.map(pre)
        if roi == "tjM1_right":
            print(f"-- {key} {p} {roi} AUC {w0:g}..{w1:g} s: per animal AUC / pre at this spout\n"
                  + g.pivot_table(index="animal", columns="epoch", values="r").reindex(columns=il.EP4).round(2).to_string())
        ref = {a: t.r.to_numpy() for a, t in g[g.epoch == "pre"].groupby("animal")}
        for ep in il.EP4[1:]:
            post = {a: t.r.to_numpy() for a, t in g[g.epoch == ep].groupby("animal")}
            com = sorted(set(post) & set(ref))
            if len(com) < 3:
                continue
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            sig = bool(iv.lo > 0 or iv.hi < 0)
            out.append({"set": key, "spout": p, "roi": roi, "window": f"{w0:g}-{w1:g}", "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                        "sig": sig, "n_animals": len(com),
                        "cell": f"{iv.point:+.2f}{'*' if sig else ''} [{iv.lo:+.2f},{iv.hi:+.2f}] ({len(com)})"})
    return out


# -------------------------------------------------------------------------------------------------------------- PLOTS
ROWLAB = {0: "dF/F x1e3 (baseline-subtracted)", 2: "minus the animal's PRE curve\n(/ its pre peak)"}


def _draw_sessions(F: dict, xlab: str, title: str, path: Path, nlab: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(3, len(ROIS), figsize=(23, 11), sharex=True)
    for j, _roi in enumerate(ROIS):
        for i in (0, 1, 2):
            ax = axs[i, j]
            p = F[(i, j)]
            for ep, mu, se, n in p["curves"]:
                if i == 2 and ep == "pre":
                    continue
                ax.plot(LAGS_W, mu, color=colors[ep], lw=2, label=f"{ep} ({n})")
                ax.fill_between(LAGS_W, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
            if p["sig"] is not None:
                il.sig_draw(ax, p["sig"], colors)
                il._headroom([ax])
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.5" if i == 2 else "0.8", lw=0.8 if i == 2 else 0.6, ls="--" if i == 2 else "-")
            ax.set_xlim(-2, 3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(p["roi"], fontsize=9)
            if j == 0:
                ax.set_ylabel(ROWLAB.get(i, nlab), fontsize=8.5)
            if i == 2:
                ax.set_xlabel(xlab)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title + "\nRow 3 = DIFFERENCE from each animal's own pre curve, in units of that pre curve's peak (0..1.5 s); "
                 "0 = no change. " + il.SIG_NOTE.replace("\n", " "), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, path, dpi=105)
    plt.close(fig)


def _draw_spout(F: dict, key: str, xlab: str, title: str, path: Path, nlab: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    pos = F["pos"]
    fig, axs = plt.subplots(len(SPOUT_ROIS), len(pos), figsize=(22, 11), sharex=True, sharey="row")
    for c, p in enumerate(pos):
        for r, roi in enumerate(SPOUT_ROIS):
            ax = axs[r, c]
            for ep, mu, se, n in F["spout"][(key, p)][r]["curves"]:
                ax.plot(LAGS_W, mu, color=colors[ep], lw=1.8, label=f"{ep} ({n})")
                ax.fill_between(LAGS_W, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
            S = F["spout"][(key, p)][r].get("sig")
            if S is not None:
                il.sig_draw(ax, S, colors)
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_xlim(-2, 3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:   # session counts per epoch in the column title (they differ by spout), not a legend on the data
                ns = " / ".join(str(n) for _e, _m, _s, n in F["spout"][(key, p)][r]["curves"])
                ax.set_title(f"{p.replace('_', ' ')}\nsessions {ns}", fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{roi}\n{nlab}", fontsize=8)
            if r == len(SPOUT_ROIS) - 1:
                ax.set_xlabel(xlab, fontsize=8)
    il._headroom(axs[:, 0])   # rows share y: raise each row once
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=colors[e], lw=2.2, label=e) for e in il.EP4], loc="lower center", ncol=4,
               fontsize=9, frameon=False, title="epoch (column titles: sessions pre / acute / subacute / chronic)",
               title_fontsize=8.5)
    fig.suptitle(title + "\n" + il.SIG_NOTE.replace("\n", " ") + " Tested per spout against the animal's pre curve at "
                 "that spout.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save_png_svg(fig, path, dpi=100)
    plt.close(fig)


def _draw_ratio(F: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(len(TSETS), len(ROIS), figsize=(23, 10), sharex=True)
    for i, k in enumerate(TSETS):
        for j, roi in enumerate(ROIS):
            ax = axs[i, j]
            for ep in il.EP4[1:]:
                if (roi, ep) in F["ratio"][k]:
                    mu, se, n = F["ratio"][k][(roi, ep)]
                    ax.plot(LAGS_W, mu, color=colors[ep], lw=2, label=f"{ep} ({n} animals)")
                    ax.fill_between(LAGS_W, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
            ax.axhline(1, color="0.5", lw=0.8, ls="--")
            ax.axvline(0, color="0.6", lw=0.6)
            ax.set_xlim(-2, 3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(roi, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{k.upper()}: epoch / pre", fontsize=9)
            if i == len(TSETS) - 1:
                ax.set_xlabel("time (s)", fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("SUPPLEMENT, NEURAL: RATIO to pre per time point (mean over animals of epoch curve / that animal's pre curve), "
                 f"shown only where EVERY animal's pre curve is >= {RATIO_MIN:.0%} of its peak (elsewhere undefined). x = time from the first lick (cued, ITI) or the cue.\n"
                 "Flat = the whole response scaled by one factor; a slope or bump = the SHAPE changed. Rows: first cued "
                 "lick, cue-aligned, ITI bout start. Descriptive.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_svg(fig, path, dpi=100)
    plt.close(fig)


def _draw_windows(F: dict, k: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    W = F["win"][k]
    fig, axs = plt.subplots(len(W["vals"]), len(ROIS), figsize=(21, 15))
    for j, roi in enumerate(ROIS):
        for i, v in enumerate(W["vals"]):
            ax = axs[i, j]
            ax.axhline(1, color="0.7", lw=0.8, ls="--")
            il.dots_draw(ax, W["panels"][(i, j)], colors)
            if i == 0:
                ax.set_title(roi, fontsize=9)
            if j == 0:
                lab = "peak 0..1.5 s" if v == "peak" else f"AUC {v.split('_')[1]}..{v.split('_')[2]} s"
                ax.set_ylabel(lab + "\n/ animal's pre", fontsize=8.5)
    what = "FIRST CUED lick (lick-aligned; windows after the lick overlap the cue response)" if k == "cued" else \
        "ITI bout-start lick (as 14.1)"
    fig.suptitle(f"RESPONSE-WINDOW SENSITIVITY, {what}: response AUC over 0..0.3 / 0..0.5 (the published window) / 0..1.0 / "
                 "0..1.5 s and the peak (0..1.5 s), baseline -1.0..-0.5 s before the lick (10-28), each / the animal's pre "
                 "mean of the same measure;\none dot per session, "
                 "black = mean +- SEM, * = nested-bootstrap change from pre (95% CI excludes 0).", fontsize=9.5)
    fig.tight_layout(rect=(0.01, 0, 0.99, 0.94))
    save_png_svg(fig, path, dpi=100)
    plt.close(fig)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    out = il.OUT_W / "figures"
    _draw_sessions(F["sess"]["cued"], TSETS["cued"][2],
                   "NEURAL, NO DECODER: dF/F around the FIRST CUED lick of each engaged trial (cued version of 14.1 / 14.3), "
                   "lick-ONSET aligned, -2..+3 s, baseline -1.5..-1.0 s; per-session PETH, line = mean over sessions, band = "
                   "SEM.\nCAUTION: the cue comes ~0.2-0.5 s before this lick, so activity just before 0 is mostly the CUE "
                   "response. Row 2 / the animal's PRE response AUC (0..0.5 s).",
                   out / "lick_cued_ramp_sessions.png", "/ animal's pre response AUC")
    _draw_sessions(F["sess"]["cue"], TSETS["cue"][2],
                   "NEURAL, NO DECODER: dF/F around the CUE of engaged trials with a first lick (cue-aligned, -2..+3 s; "
                   "baseline -1.0..-0.5 s before the cue); the first lick follows ~0.2-0.5 s after the cue.\nPer-session "
                   "PETH, line = mean over sessions, band = SEM. Row 2 / the animal's PRE cue-response AUC (0..0.5 s).",
                   out / "lick_cued_ramp_cue_sessions.png", "/ animal's pre cue-response AUC")
    _draw_sessions(F["sess"]["iti"], TSETS["iti"][2],
                   "NEURAL, NO DECODER: dF/F around SELF-INITIATED ITI licks (bout start, no lick for >= 2 s; as 14.1) on the "
                   "longer -2..+3 s window, baseline -1.5..-1.0 s; per-session PETH, line = mean over sessions, band = SEM.\n"
                   "Row 2 / the animal's PRE response AUC (0..0.5 s). Later licks of the bout fall inside the window (14.5 = "
                   "isolated licks).", out / "lick_iti_ramp_wide.png", "/ animal's pre response AUC")
    _draw_ratio(F, out / "lick_ramp_ratio.png")
    for k in ("cued", "iti"):
        _draw_windows(F, k, out / f"lick_ramp_windows_{k}.png")
    _draw_spout(F, "lick", "time from first cued lick (s)",
                "SUPPLEMENT, NEURAL: first cued lick PER SPOUT (target), lick-onset aligned (-2..+3 s, baseline -1.0..-0.5 s), "
                "/ the animal's PRE response AUC over all spouts (one scale across spouts); line = mean over sessions "
                f"(>= {MIN_SPOUT} licks of that spout), band = SEM; _right = contralesional.",
                out / "lick_cued_ramp_perspout.png", "/ pre response AUC")
    _draw_spout(F, "cue", "time from cue (s)",
                "SUPPLEMENT, NEURAL: CUE-aligned per spout (-2..+3 s, baseline -1.0..-0.5 s), / the animal's PRE "
                "cue-response AUC over all spouts; line = mean over sessions, band = SEM.",
                out / "lick_cued_ramp_cue_perspout.png", "/ pre cue AUC")
    _draw_spout(F, "iti", "time from ITI bout-start lick onset (s)",
                "SUPPLEMENT, NEURAL: ITI bout-start licks (no lick for >= 2 s) by the PREVIOUS trial's spout (the spout is "
                "retracted in the ITI), lick-onset aligned (-2..+3 s, baseline -1.0..-0.5 s), / the animal's PRE ITI response "
                f"AUC over all spouts; line = mean over sessions (>= {MIN_SPOUT} licks), band = SEM.",
                out / "lick_iti_ramp_perspout.png", "/ pre response AUC")
    if "auc_tests" in F:     # figdata from before 10-28 evening lacks the per-spout window tests
        _draw_spout_windows(F, "lick", "first cued lick by TARGET spout", out / "lick_cued_perspout_windows.png")
        _draw_spout_windows(F, "cue", "cue-aligned by target spout", out / "lick_cue_perspout_windows.png")
    fig, axs = plt.subplots(len(SPEC), len(ROIS), figsize=(21, 15))
    for j, roi in enumerate(ROIS):
        for i, (_val, ylab) in enumerate(SPEC):
            ax = axs[i, j]
            ax.axhline(0, color="0.8", lw=0.6)
            il.dots_draw(ax, F["summary"][(i, j)], colors)
            if i == 0:
                ax.set_title(roi, fontsize=9)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle("NEURAL, NO DECODER: FIRST CUED lick, per-session dF/F AUCs (cued version of 14.4; one dot per session, all "
                 "animals pooled; black = mean +- SEM over sessions). Lick response = AUC 0..1.0 s after lick onset (default since 10-28);\npre-lick "
                 "window = AUC -0.4..-0.05 s before it (overlaps the CUE response); cue response = AUC 0..1.0 s after the cue. "
                 "Baselines (10-28): lick response -1.0..-0.5 s, pre-lick window -1.5..-1.0 s before the lick, cue -1.0..-0.5 s before the cue. "
                 "* = change from pre with 95% nested-bootstrap CI (animals > sessions) excluding 0.", fontsize=9.5)
    fig.tight_layout(rect=(0.01, 0, 0.99, 0.94))
    save_png_svg(fig, out / "lick_cued_ramp_summary.png", dpi=110)
    plt.close(fig)
    if "summary_iti" in F:      # ITI AUC summary on the wide traces (14.10 with the 0..1.0 s response window)
        fig, axs = plt.subplots(len(SPEC_ITI), len(ROIS), figsize=(21, 12.5))
        for j, roi in enumerate(ROIS):
            for i, (_val, ylab) in enumerate(SPEC_ITI):
                ax = axs[i, j]
                ax.axhline(0, color="0.8", lw=0.6)
                il.dots_draw(ax, F["summary_iti"][(i, j)], colors)
                if i == 0:
                    ax.set_title(roi, fontsize=9)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8)
        fig.suptitle("NEURAL, NO DECODER: self-initiated ITI licks (bout start, no lick for >= 2 s), per-session dF/F AUCs (one "
                     "dot per session, all animals pooled; black = mean +- SEM over sessions). Response = AUC 0..1.0 s after lick "
                     "onset (baseline -1.0..-0.5 s; default since 10-28);\nbuild-up = AUC -0.4..-0.05 s before it (baseline "
                     "-1.5..-1.0 s), / the animal's pre 0..0.5 s response AUC (as before). * = change from pre with 95% "
                     "nested-bootstrap CI (animals > sessions) excluding 0.", fontsize=9.5)
        fig.tight_layout(rect=(0.01, 0, 0.99, 0.93))
        save_png_svg(fig, out / "lick_iti_ramp_summary_wide.png", dpi=110)
        plt.close(fig)


BASE_ALT = (-1.0, -0.5)
SENS_ROIS = ["ALM_right", "tjM1_right", "SSp-m_right", "cortex"]
SENS_VALS = (("resp05", 0.0, 0.5, "response AUC 0..0.5 s"), ("resp15", 0.0, 1.5, "response AUC 0..1.5 s"),
             ("build", -0.4, -0.05, "build-up AUC -0.4..-0.05 s"))


def baseline_sens() -> None:
    """BASELINE SENSITIVITY (Priya 10-28: keep the current baselines, add -1..-0.5 s as a check): lick-aligned ITI bout
    starts and first cued licks, baseline -1.5..-1.0 (build-up) vs -1.0..-0.5 s (response sizes since 10-28) before the lick; per session AUCs /
    the animal's pre mean of the same measure under the same baseline; change vs pre, paired nested bootstrap.
    Caveats drawn in the figure: -1.0..-0.5 can eat into a slow ITI build-up, and for cued licks it overlaps the cue when
    the reaction time is > 0.5 s. Outputs ramp_baseline_sens.csv, figures/lick_ramp_baseline_sens.png (+ figdata)."""
    from wfield_local import analysis_kit as ak
    D = _sessions()
    rows = []
    for k, key in (("iti", "iti"), ("iti_iso", "iti_iso"), ("cued", "lick")):   # iti_iso = no lick 0.05..0.75 s after
        for bname, base in (("-1.5..-1.0 (build-up baseline)", BASE_L), ("-1.0..-0.5 (response baseline)", BASE_ALT)):
            PE = _pe(D, key, base, SENS_ROIS)
            for v, t0, t1, _lab in SENS_VALS:
                PE[v] = [_auc(p, LAGS_W, t0, t1) for p in PE.peth]
            for roi, g in PE.groupby("roi"):
                for v, _t0, _t1, _lab in SENS_VALS:
                    pre = g[g.epoch == "pre"].groupby("animal")[v].mean()
                    ref = {a: g[(g.animal == a) & (g.epoch == "pre")][v].to_numpy() / pre[a] for a in pre.index}
                    for ep in il.EP4[1:]:
                        post = {a: g[(g.animal == a) & (g.epoch == ep)][v].to_numpy() / pre[a] for a in pre.index}
                        com = [a for a in pre.index if len(post[a]) and len(ref[a])]
                        if len(com) < 3:
                            continue
                        iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                        am = {a: float(post[a].mean() - ref[a].mean()) for a in com}
                        rows.append(dict(set=k, baseline=bname, roi=roi, value=v, epoch=ep, delta=iv.point, lo=iv.lo,
                                         hi=iv.hi, n_dir=sum(np.sign(x) == np.sign(iv.point) for x in am.values()),
                                         n=len(com)))
    T = pd.DataFrame(rows)
    T.to_csv(il.OUT_W / "ramp_baseline_sens.csv", index=False)
    pd.set_option("display.width", 220)
    print(T[T.roi == "tjM1_right"].round(3).to_string(index=False))
    save_fd("lick_ramp_baseline_sens", {"T": T})
    draw_baseline_sens({"T": T})


def draw_baseline_sens(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    T = F["T"]
    sets = ("iti", "iti_iso", "cued")
    fig, axs = plt.subplots(len(sets) * len(SENS_VALS), len(SENS_ROIS), figsize=(16, 25), squeeze=False)
    bl = list(dict.fromkeys(T.baseline))
    for si, st in enumerate(sets):
        for vi, (v, _t0, _t1, lab) in enumerate(SENS_VALS):
            for j, roi in enumerate(SENS_ROIS):
                ax = axs[si * len(SENS_VALS) + vi, j]
                for e, ep in enumerate(il.EP4[1:]):
                    for b, bn in enumerate(bl):
                        r = T[(T.set == st) & (T.value == v) & (T.roi == roi) & (T.epoch == ep) & (T.baseline == bn)]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        x = e + (b - 0.5) * 0.3
                        sig = r.lo > 0 or r.hi < 0
                        ax.errorbar(x, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="o" if b == 0 else "s",
                                    color=colors[ep], mfc=colors[ep] if sig else "white", ms=6, capsize=2)
                ax.axhline(0, color="0.6", lw=0.7, ls="--")
                ax.set_xticks(range(3), ["acute", "sub", "chronic"], fontsize=8)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                if si == 0 and vi == 0:
                    ax.set_title(roi, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{st.upper()}: {lab}\nchange vs pre (/ pre)", fontsize=8)
    fig.suptitle("BASELINE SENSITIVITY: change vs pre (each value / the animal's pre mean under the same baseline), 95% paired "
                 "nested-bootstrap CI; circle = baseline -1.5..-1.0 s before the lick (used for build-up), square = -1.0..-0.5 s (used for response sizes since 10-28); "
                 "filled = CI excludes 0.\nCaveats: -1.0..-0.5 s can subtract part of a slow ITI build-up; for cued licks it "
                 "overlaps the cue when the reaction time is > 0.5 s.\nLick-aligned: ITI bout starts (>= 2 s gap), ISOLATED "
                 "ITI licks (iti_iso: no lick 0.05..0.75 s after; later licks can still fall in 0.75..1.5 s), first cued "
                 "licks.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_ramp_baseline_sens.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if "--baseline-sens" in sys.argv:
        if plot_only():
            draw_baseline_sens(load_fd("lick_ramp_baseline_sens"))
        else:
            baseline_sens()
        return 0
    if "--prev-cache" in sys.argv:
        prev_cache(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8)
        return 0
    if "--extract" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
        extract(jobs)
        return 0
    if plot_only():
        plot(load_fd("lick_cued_ramp"))
        return 0
    F = compute()
    save_fd("lick_cued_ramp", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
