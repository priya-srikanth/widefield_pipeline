"""IS THE RESTW BASELINE CONTAMINATED IN AN EPOCH-DEPENDENT WAY? (Priya 10-29.) PROVISIONAL, n = 4.

CONCERNS. restdock05 rest (quiet_periods.rest_mask via behavior_events) excludes only DAQ (CONTACT) lick onsets, buffer
[0.5 before, 1.0 after] s: (a) cam4 licks WITHOUT spout contact stay inside rest, and contact rates fall post-stroke;
(b) 1.0 s after a lick is short for GCaMP6s.

WHAT THIS DOES (per session, nothing on disk is overwritten -- the restdock05 sample mask is READ, then intersected in
memory with extra exclusions):
  restw      the restdock05 mask unchanged (recomputed here; must equal the precue_restw cache restw -> checked)
  daq2       + DAQ lick onsets (the mask's own detector, config criteria) with buffer [0.5, 2.0] s        -> concern (b)
  strict     daq2 + EVERY cam4 lick of licks_all.csv (any context, contact or not; daq_on_s) [0.5, 2.0] s -> (a)+(b)
  strict_c1  strict + every cam1 2-D lick event (session_poses/cam1_licks/<a>_<d>_events_all.csv, daq_on_s)
  strict_jaw strict + frames of cam4 cleaned jaw y (clean.npz, px from the jaw rest point) above a per-session robust
             threshold (median + 5 x 1.4826 MAD over all video frames; hidden jaw = not counted) [0.5, 2.0] s
  After each intersection the mask's own min_rest_s floor (0.5 s) is re-applied, as rest_mask does last.
LEVEL = `rest_by_position.restw_from_frames` on the whole-session area series of `lick_precue_restw.session_worker`
(ROI v8, ALM / tjM1 / MOs_rest / MOp_rest split, slow drift removed, x1e3), rest frames from
`rest_frames_by_position(docked=False, engaged_only=True)` -- the module's own functions, with its mask loader fed the
variant mask (in-process substitution inside the worker, the module file untouched). Same 6-position / 200-frame floors.
Video coverage: cam4 / cam1 / jaw exclusions exist only inside the video windows (licks_all_coverage.json ~99 %).
RESULTS UNDER STRICT: per-session baseline swap, value_strict = value_restw + restw - strict (exact: every one of these
values is a trace minus the session restw), then the scripts' own statistics (iti_area_maps.changes for the pre-cue
maps; lick_cue_to_lick.per_session / matched / deltas with rng default_rng(0) in the published order for mean_rw).
ENGAGED-COUNT AUDIT: precue_restw cache trials (behaviour table at extraction time) vs the behaviour table now vs
enl_trials.csv.gz `engaged` (the gate of arrival_to_cue / stopped traces) vs the DAQ gate the restw rest frames use;
pre-cue level (-1..0 s) recomputed on the enl_trials engaged set.
Outputs: <iti_licks>/restw_check/ (sessions/<label>.npz, *.csv, figures/, figdata/).
Usage: python scripts/restw_strict_check.py [--jobs 4] [--labels A B ...] [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

SRC = il.OUT                                    # read-only inputs
OUTD = il.OUT / "restw_check"                   # the ONLY write root
SESS = OUTD / "sessions"
FIGS = OUTD / "figures"
FDR = OUTD / "figdata"
VERSION = 3
BUF = (0.5, 2.0)
VARIANTS = ["restw", "daq2", "strict", "strict_c1", "strict_jaw"]          # the masks
MIN_FR_RELAXED = 50
# the variants reported (d_<v> = v - restw; *_flat - restw_flat). PRIMARY = strict_w50 (all sessions kept)
SHOW = ["daq2", "strict", "strict_w50", "strict_c1_w50", "strict_jaw_w50", "strict_flat"]
VLAB = {"daq2": "DAQ licks [0.5, 2.0]", "strict": "STRICT (+ all cam4 licks), 200-frame floor",
        "strict_w50": "STRICT, 50-frame floor (primary)", "strict_c1_w50": "strict + cam1 licks, 50-frame floor",
        "strict_jaw_w50": "strict + jaw motion, 50-frame floor", "strict_flat": "STRICT unweighted median vs restw unweighted"}
LAGS = np.round(np.arange(-2.0, 3.0001, 0.05), 3)
WIN = (-1.0, 0.0)
KEYC = ["ALM_right", "tjM1_right", "ALM_left", "tjM1_left", "SSp-m_right", "SSp-n_right", "SSp-m_left", "SSp-n_left"]
EP = il.EP4
AN = ["PS92", "PS93", "PS94", "PS95"]


# ------------------------------------------------------------------------------------------------ mask helpers
def widen(idx: np.ndarray, n: int, nb: int, na: int) -> np.ndarray:
    """Boolean of length n, True within [i - nb, i + na] of every event sample i (= quiet_periods.widen_bool_sparse)."""
    idx = np.asarray(idx, np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    d = np.zeros(n + 1, np.int32)
    np.add.at(d, np.maximum(idx - nb, 0), 1)
    np.add.at(d, np.minimum(idx + na + 1, n), -1)
    return np.cumsum(d[:n]) > 0


def floor_runs(b: np.ndarray, n: int) -> np.ndarray:
    from wfield_local.quiet_periods import set_short_bool_to_low
    return set_short_bool_to_low(b, n)


def _daq_lick_onsets(s: dict, fs: float) -> np.ndarray:
    """The rest mask's own DAQ lick detector (behavior_events / rest_by_position._engaged_trials: config criteria)."""
    import h5py

    from wfield_local import config
    from wfield_local.lick_detection import detect_licks
    lk = config.defaults()["lick_detection"]
    with h5py.File(s["h5"], "r") as f:
        nm = [x.decode() for x in f["analog/channel_names"][:]]
        i = nm.index(lk.get("channel", "lick_analog"))
        if "samples_int16" in f["analog"]:
            sc = float(f["analog/int16_scale_volts_per_count"][i])
            of = float(f["analog/int16_offset_volts"][i])
            lv = f["analog/samples_int16"][:, i].astype(np.float32) * sc + of
        else:
            lv = np.asarray(f["analog/samples"][:, i], np.float32)
    det = detect_licks(lv, fs, thresh_upper=lk["thresh_upper"], thresh_lower=lk["thresh_lower"],
                       lockout_s=tuple(lk["lockout_falling_edge_s"]), min_ili_s=lk.get("min_ili_ms", 0) / 1000.0)
    return np.sort(np.asarray(det["lick_onsets"], np.int64))


def _jaw_moving_s(sd: Path) -> tuple[np.ndarray, dict]:
    """DAQ times (s) of cam4 frames whose cleaned jaw y (px from rest) exceeds the per-session robust threshold."""
    c = np.load(sd / "clean.npz")
    w = np.load(sd / "windows.npz")
    t = w["ix_daq_s"].astype(float)
    disp = c["jaw_y"].astype(float)               # cleaned jaw y = px from the jaw rest point (opening axis); hidden = NaN
    ok = np.isfinite(disp) & np.isfinite(t)
    med = float(np.median(disp[ok]))
    mad = float(np.median(np.abs(disp[ok] - med))) * 1.4826
    thr = med + 5 * mad
    mv = ok & (disp > thr)
    return t[mv], {"jaw_thr_px": thr, "jaw_med_px": med, "jaw_frac_moving": float(mv.sum() / max(ok.sum(), 1)),
                   "jaw_frac_visible": float(ok.mean())}


# ------------------------------------------------------------------------------------------------ per session
def session_worker(item) -> dict:
    """One session (module-level for spawn) -> SESS/<label>.npz. item = (label, enl-engaged cue times)."""
    label, enl_cue = item
    f = SESS / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            return {"label": label, "cached": True}
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import rest_by_position as rbp
    from wfield_local.paths import PathResolver
    src = np.load(SRC / "precue_restw" / f"{label}.npz")
    s = next(x for x in config.load_sessions() if x["label"] == label)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    # == lick_precue_restw.session_worker's trace construction
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")
    if cs != [str(c) for c in src["cols"]]:
        raise ValueError(f"{label}: columns differ from the precue_restw cache")
    nT = Rs.shape[0]
    del V, Ub
    # ---- the restdock05 mask + DAQ pieces, read ONCE (the module's own loader), and the DAQ engagement gate
    rest0, cs_s, codes, ts, fs_samp, sync = rbp._session_daq(s)
    fs = ak.daq_rate(s)
    eng = rbp._engaged_trials(s, cs_s, codes, fs)
    n = rest0.shape[0]
    nb, na = int(BUF[0] * fs), int(BUF[1] * fs)
    from wfield_local.quiet_periods import widen_bool_sparse  # noqa: F401  (definition mirrored by `widen`)
    minrun = int(float(config.defaults()["segmentation"]["rest"]["min_rest_s"]) * fs)
    daq_on = _daq_lick_onsets(s, fs)
    sd = oo.session_dir(PathResolver(), oc.RUN, an, date)
    A = pd.read_csv(sd / "licks_all.csv", usecols=["context", "contact", "daq_on_s"])
    A = A[np.isfinite(A.daq_on_s.to_numpy(float))]
    cam4_on = A.daq_on_s.to_numpy(float)
    cam4_contact = A.contact.astype(str).str.lower().eq("true").to_numpy()
    c1f = Path(PathResolver().root("dlc")) / "session_poses" / "cam1_licks" / f"{an}_{date}_events_all.csv"
    c1 = pd.read_csv(c1f, usecols=["daq_on_s", "cam4_matched"]) if c1f.exists() else None
    notes = {}
    try:
        jaw_t, jn = _jaw_moving_s(sd)
        notes.update(jn)
    except Exception as ex:                                            # noqa: BLE001
        jaw_t = None
        notes["jaw_err"] = f"{type(ex).__name__} {str(ex)[:60]}"

    def smp(t_s):
        return np.round(np.asarray(t_s, float) * fs).astype(np.int64)
    # sanity: the mask already excludes DAQ licks at [0.5, 1.0] -> re-applying must change nothing
    chk = floor_runs(rest0 & ~widen(daq_on, n, int(0.5 * fs), int(1.0 * fs)), minrun)
    n_chk = int((rest0 & ~chk).sum())
    m_daq2 = floor_runs(rest0 & ~widen(daq_on, n, nb, na), minrun)
    m_strict = floor_runs(m_daq2 & ~widen(smp(cam4_on), n, nb, na), minrun)
    masks = {"restw": rest0, "daq2": m_daq2, "strict": m_strict}
    if c1 is not None:
        masks["strict_c1"] = floor_runs(m_strict & ~widen(smp(c1.daq_on_s.dropna()), n, nb, na), minrun)
    if jaw_t is not None:
        masks["strict_jaw"] = floor_runs(m_strict & ~widen(smp(jaw_t), n, nb, na), minrun)

    # ---- the module's own frame selection on each variant mask (loader substituted in this worker process only)
    orig_daq, orig_eng = rbp._session_daq, rbp._engaged_trials
    out = {"version": VERSION, "cols": np.array(cs), "animal": str(src["animal"]), "epoch": str(src["epoch"]),
           "restw_cache": np.asarray(src["restw"], float), "restw_ok_cache": bool(src["restw_ok"])}
    per0 = None
    try:
        rbp._engaged_trials = lambda *_a, **_k: eng
        for v, m in masks.items():
            rbp._session_daq = lambda _s, _m=m: (_m, cs_s, codes, ts, fs_samp, sync)
            per, info = rbp.rest_frames_by_position(s, nT, docked=False, engaged_only=True)
            if info.get("error"):
                raise RuntimeError(f"{label} {v}: {info['error']}")
            base, used = rbp.restw_from_frames(Rs.T, per, label=f"{label} {v}", verbose=False)
            out[f"{v}__lev"] = base[:, 0] if base is not None else np.full(len(cs), np.nan)
            out[f"{v}__ok"] = base is not None
            out[f"{v}__nfr"] = int(sum(len(x) for x in per.values()))
            out[f"{v}__nfr_pos"] = np.array([len(per.get(c, [])) for c in range(6)], int)
            out[f"{v}__nsmp"] = int(m.sum())
            # per-position medians + the flat median, so any frame floor can be re-applied offline (summary)
            out[f"{v}__posmed"] = np.stack([np.median(Rs[np.asarray(per[c])[np.asarray(per[c]) < nT]], 0)
                                            if len(per.get(c, [])) else np.full(len(cs), np.nan) for c in range(6)], 0)
            allf = np.concatenate([np.asarray(x) for x in per.values()]) if per else np.array([], int)
            allf = allf[allf < nT]
            out[f"{v}__flat"] = np.median(Rs[allf], 0) if allf.size else np.full(len(cs), np.nan)
            if v == "restw":
                per0 = per
    finally:
        rbp._session_daq, rbp._engaged_trials = orig_daq, orig_eng
    # ---- licks inside the restw rest frames (the frames feeding restw), per minute of that rest
    fr_t = fs_samp[:nT] / fs
    fps = 1.0 / float(np.median(np.diff(ft)))
    rf = np.zeros(nT, bool)
    for x in per0.values():
        rf[np.asarray(x)] = True
    rest_min = rf.sum() / fps / 60.0

    def in_rest(t_s):
        t_s = np.asarray(t_s, float)
        t_s = t_s[np.isfinite(t_s)]
        j = np.clip(np.searchsorted(fr_t, t_s), 1, nT - 1)
        j = np.where(np.abs(fr_t[j] - t_s) < np.abs(fr_t[j - 1] - t_s), j, j - 1)
        near = np.abs(fr_t[j] - t_s) <= 1.0 / fps
        return near & rf[j]
    ir = in_rest(cam4_on)
    notes.update({"rest_min": rest_min, "fps": fps, "n_chk_daq_buffer": n_chk,
                  "ft_vs_daq_maxdiff_s": float(np.nanmax(np.abs(ft - fr_t))),
                  "cam4_n": len(cam4_on), "cam4_noncontact_n": int((~cam4_contact).sum()),
                  "cam4_noncontact_in_rest": int((ir & ~cam4_contact).sum()),
                  "cam4_contact_in_rest": int((ir & cam4_contact).sum()),
                  "daq_in_rest": int(in_rest(daq_on / fs).sum())})
    if c1 is not None:
        notes["cam1_n"] = len(c1)
        notes["cam1_in_rest"] = int(in_rest(c1.daq_on_s).sum())
        notes["cam1_only_in_rest"] = int(in_rest(c1.daq_on_s[~c1.cam4_matched.astype(str).str.lower().eq("true")]).sum())
    # ---- engaged-count audit
    def p10(cue):
        cue = np.asarray(cue, float)
        cue = cue[np.isfinite(cue)]
        X = il._sample(Rs, ft, cue, LAGS).astype(np.float64)
        bad = (cue[:, None] + LAGS[None, :] < ft[0]) | (cue[:, None] + LAGS[None, :] > ft[-1])
        X[bad] = np.nan
        w = (LAGS >= WIN[0]) & (LAGS <= WIN[1])
        return np.nanmean(np.nanmean(X[:, w], 1), 0)
    try:
        B = il._behaviour(label)
        T = B.trials
        notes["n_beh_now"] = int((T.engaged.astype(bool) & np.isfinite(T.cue_s.to_numpy(float))).sum())
    except Exception as ex:                                            # noqa: BLE001
        notes["n_beh_now"] = -1
        notes["beh_err"] = f"{type(ex).__name__} {str(ex)[:60]}"
    notes.update({"n_cache": len(src["pos"]), "n_enl": int(np.isfinite(np.asarray(enl_cue, float)).sum()),
                  "n_restgate": int(np.asarray(eng).sum()), "n_trials_daq": len(eng)})
    out["precue_cache_mean"] = np.nanmean(np.asarray(src["p10"], float), 0)
    out["precue_enl_mean"] = p10(enl_cue) if len(enl_cue) else np.full(len(cs), np.nan)
    for k, v in notes.items():
        out[f"note__{k}"] = v
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return {"label": label, "cached": False}


# ------------------------------------------------------------------------------------------------ summaries
def load_sessions(labels: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(long table label x area x variant level, per-session notes)."""
    rows, notes = [], []
    for x in labels:
        z = np.load(SESS / f"{x}.npz")
        cs = [str(c) for c in z["cols"]]
        an, ep = str(z["animal"]), str(z["epoch"])
        nt = {"label": x, "animal": an, "epoch": ep, "restw_ok_cache": bool(z["restw_ok_cache"]),
              "restw_maxdiff_vs_cache": float(np.nanmax(np.abs(z["restw__lev"] - z["restw_cache"])))}
        for v in VARIANTS:
            if f"{v}__lev" not in z.files:
                continue
            nt[f"nfr_{v}"], nt[f"ok_{v}"] = int(z[f"{v}__nfr"]), bool(z[f"{v}__ok"])
            nt[f"nsmp_{v}"] = int(z[f"{v}__nsmp"])
            nt[f"minpos_{v}"] = int(z[f"{v}__nfr_pos"].min())
            npos = z[f"{v}__nfr_pos"]
            levs = {v: (z[f"{v}__lev"], bool(z[f"{v}__ok"]))}
            if f"{v}__posmed" in z.files:
                # W50: the same equal-position weighting with the frame floor relaxed 200 -> 50 (all 6 positions still
                # required) -- keeps the sessions a 200-frame floor loses once strict rest is short. FLAT: unweighted.
                levs[f"{v}_w50"] = (z[f"{v}__posmed"].mean(0), bool((npos >= MIN_FR_RELAXED).all()))
                levs[f"{v}_flat"] = (z[f"{v}__flat"], bool(npos.sum() >= MIN_FR_RELAXED))
                nt[f"ok_{v}_w50"] = levs[f"{v}_w50"][1]
            for vv, (lev, ok) in levs.items():
                for j, c in enumerate(cs):
                    rows.append({"label": x, "animal": an, "epoch": ep, "area": c, "variant": vv,
                                 "lev": float(lev[j]), "ok": ok})
        for k in z.files:
            if k.startswith("note__"):
                val = z[k]
                nt[k[6:]] = val.item() if val.shape == () else val
        pc = z["precue_enl_mean"] - z["precue_cache_mean"]
        for j, c in enumerate(cs):
            rows.append({"label": x, "animal": an, "epoch": ep, "area": c, "variant": "engnow_shift",
                         "lev": float(pc[j]), "ok": True})
        for c in ("ALM_right", "tjM1_right"):
            nt[f"dprecue_enl_{c}"] = float(pc[cs.index(c)])
        nt["dprecue_enl_maxabs"] = float(np.nanmax(np.abs(pc)))
        notes.append(nt)
    return pd.DataFrame(rows), pd.DataFrame(notes)


def diff_table(L: pd.DataFrame) -> pd.DataFrame:
    """label x area: variant level minus restw (only sessions where restw AND the variant are OK)."""
    W = L.pivot_table(index=["label", "animal", "epoch", "area"], columns="variant", values="lev").reset_index()
    OK = L.pivot_table(index=["label", "area"], columns="variant", values="ok").reset_index()
    out = W[["label", "animal", "epoch", "area"]].copy()
    out["engnow_shift"] = W["engnow_shift"]
    for v in W.columns:
        if v in ("label", "animal", "epoch", "area", "restw", "engnow_shift"):
            continue
        ref = "restw_flat" if v.endswith("_flat") else "restw"
        if v == ref:
            continue
        good = (OK[ref].astype(bool) & OK[v].fillna(False).astype(bool)).to_numpy()
        out[f"d_{v}"] = np.where(good, W[v] - W[ref], np.nan)
    out["restw"] = W["restw"]
    return out


def per_animal(D: pd.DataFrame, val: str, areas: list) -> pd.DataFrame:
    P = D[D.area.isin(areas)].pivot_table(index=["area", "animal"], columns="epoch", values=val, aggfunc="mean")
    P = P.reindex(columns=EP)
    for ep in EP[1:]:
        P[f"chg_{ep}"] = P[ep] - P["pre"]
    return P


def swap_precue(D: pd.DataFrame, tag: str, area_set: list, rng_note: str = "") -> pd.DataFrame:
    """Pre-cue maps result under each variant: precue + restw - variant; iti_area_maps.changes on each."""
    import iti_area_maps as iam
    T = pd.read_csv(SRC / f"precue_restw_sessions{tag}.csv")
    T = T[T.restw_ok]
    out = []
    for v in ["restw"] + [x for x in SHOW if not x.endswith("_flat")]:
        t = T.copy()
        if v != "restw":
            if f"d_{v}" not in D:
                continue
            dd = D.set_index(["label", "area"])[f"d_{v}"]
            t["precue"] = t.precue - dd.reindex(pd.MultiIndex.from_frame(t[["label", "area"]])).to_numpy()
        t = t[t.area.isin(area_set)]
        if v == "restw" and tag == "":
            # ENGAGED-COUNT AUDIT: the same result on the CURRENT engaged-trial set (enl_trials), restw unchanged
            te = t.copy()
            es = D.set_index(["label", "area"])["engnow_shift"]
            te["precue"] = te.precue + es.reindex(pd.MultiIndex.from_frame(te[["label", "area"]])).to_numpy()
            Ce = iam.changes(te, "precue")
            Ce["variant"] = "restw_engaged_now"
            out.append(Ce.merge(te.groupby(["area", "epoch"]).precue.mean().rename("level_sessionmean").reset_index(),
                                on=["area", "epoch"], how="left"))
        C = iam.changes(t, "precue")
        C["variant"] = v
        lv = t.groupby(["area", "epoch"]).precue.mean().rename("level_sessionmean").reset_index()
        out.append(C.merge(lv, on=["area", "epoch"], how="left"))
    return pd.concat(out, ignore_index=True)


def swap_c2l(D: pd.DataFrame) -> pd.DataFrame:
    """Cue-to-lick mean_rw changes (raw + latency-matched) under each variant, published rng order."""
    import lick_cue_to_lick as c2l
    T0 = pd.read_csv(SRC / "cue_to_lick" / "c2l_trials.csv.gz",
                     usecols=["label", "animal", "epoch", "pos", "lat", "tau", "col", "mean_rw", "bin"])
    T0 = T0[T0.tau == 0]
    out = []
    for v in ["restw"] + [x for x in SHOW if not x.endswith("_flat")]:
        t = T0.copy()
        if v != "restw":
            if f"d_{v}" not in D:
                continue
            dd = D.set_index(["label", "area"])[f"d_{v}"]
            t["mean_rw"] = t.mean_rw - dd.reindex(pd.MultiIndex.from_arrays([t.label, t.col])).to_numpy()
            t = t[np.isfinite(t.mean_rw)]
        rng = np.random.default_rng(0)
        S = c2l.per_session(t, "mean_rw")
        S["cmp"] = ""
        rows = c2l.deltas(S, {"measure": "mean_rw", "match": "raw"}, rng)
        Mt = c2l.matched(t, "mean_rw")
        rows += c2l.deltas(Mt, {"measure": "mean_rw", "match": "latency-matched"}, rng, cmp_col=True)
        R = pd.DataFrame(rows)
        R["variant"] = v
        out.append(R)
    return pd.concat(out, ignore_index=True)


def summarise(labels: list) -> None:
    from wfield_local import analysis_kit as ak
    L, N = load_sessions(labels)
    N.to_csv(OUTD / "restw_check_session_notes.csv", index=False)
    print(f"== {len(N)} sessions; recomputed restw vs precue_restw cache: max |diff| = "
          f"{N.restw_maxdiff_vs_cache.max():.2e}; DAQ [0.5,1.0] re-application removed "
          f"{int(N.n_chk_daq_buffer.sum())} samples (expect 0)")
    print("== restw_ok per variant:", {c[3:]: int(N[c].sum()) for c in N.columns if c.startswith("ok_")})
    # ---- 2. rest time removed + licks in rest, per animal x epoch
    rem = []
    for (an, ep), g in N.groupby(["animal", "epoch"]):
        r = {"animal": an, "epoch": ep, "sessions": len(g), "rest_min_restw": g.rest_min.sum()}
        for v in VARIANTS[1:]:
            if f"nfr_{v}" in g:
                r[f"removed_{v}_%"] = 100 * (1 - g[f"nfr_{v}"].sum() / g.nfr_restw.sum())
        r["cam4_noncontact_in_rest_per_min"] = g.cam4_noncontact_in_rest.sum() / g.rest_min.sum()
        r["cam4_contact_in_rest_per_min"] = g.cam4_contact_in_rest.sum() / g.rest_min.sum()
        if "cam1_only_in_rest" in g:
            r["cam1_only_in_rest_per_min"] = g.cam1_only_in_rest.sum() / g.rest_min.sum()
        r["cam4_noncontact_frac_all"] = g.cam4_noncontact_n.sum() / g.cam4_n.sum()
        r["jaw_frac_moving"] = g.jaw_frac_moving.mean() if "jaw_frac_moving" in g else np.nan
        rem.append(r)
    REM = pd.DataFrame(rem)
    REM["epoch"] = pd.Categorical(REM.epoch, EP, ordered=True)
    REM = REM.sort_values(["animal", "epoch"])
    REM.to_csv(OUTD / "restw_check_rest_removed.csv", index=False)
    print("== 2. rest removed (% of restw rest frames) and licks inside restw rest, per animal x epoch\n"
          + REM.round(3).to_string(index=False))
    # ---- 3. strict - restw per area x animal x epoch
    D = diff_table(L)
    D.to_csv(OUTD / "restw_check_diff_sessions.csv", index=False)
    areas = sorted(D.area.unique())
    keep_areas = [a for a in areas if a != "cortex" and a.rsplit("_", 1)[0] not in ("MOs", "MOp")]
    PA = {v: per_animal(D, f"d_{v}", keep_areas) for v in SHOW if f"d_{v}" in D}
    for v, P in PA.items():
        P.to_csv(OUTD / f"restw_check_per_animal_{v}.csv")
    for v in ("strict_w50", "strict"):
        print(f"== 3. per animal (rule 8): {v} - RESTW (dF/F x1e3), key areas\n"
              + PA[v].loc[[a for a in KEYC if a in PA[v].index.get_level_values(0)]].round(2).to_string())
    print("== per animal: DAQ2 - RESTW, key areas\n"
          + PA["daq2"].loc[[a for a in KEYC if a in PA["daq2"].index.get_level_values(0)]].round(2).to_string())
    rows = []
    for v in SHOW:
        if f"d_{v}" not in D:
            continue
        for area, t in D.groupby("area"):
            if area not in keep_areas:
                continue
            ref = {a: t[(t.animal == a) & (t.epoch == "pre")][f"d_{v}"].dropna().to_numpy() for a in AN}
            lev = ak.boot_ci({a: ref[a] for a in AN if len(ref[a])}, np.random.default_rng(0))
            if lev is not None:
                rows.append({"variant": v, "area": area, "epoch": "pre", "kind": "level", "est": lev.point,
                             "lo": lev.lo, "hi": lev.hi})
            for ep in EP[1:]:
                post = {a: t[(t.animal == a) & (t.epoch == ep)][f"d_{v}"].dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(ref[a]) and len(post[a])]
                lv = ak.boot_ci({a: post[a] for a in post if len(post[a])}, np.random.default_rng(0))
                if lv is not None:
                    rows.append({"variant": v, "area": area, "epoch": ep, "kind": "level", "est": lv.point,
                                 "lo": lv.lo, "hi": lv.hi})
                if len(com) < 3:
                    continue
                iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                if iv is None:
                    continue
                pa = [float(np.mean(post[a]) - np.mean(ref[a])) for a in com]
                rows.append({"variant": v, "area": area, "epoch": ep, "kind": "change_vs_pre", "est": iv.point,
                             "lo": iv.lo, "hi": iv.hi, "sig": bool(iv.lo > 0 or iv.hi < 0), "n_animals": len(com),
                             "n_same_sign": int(sum(np.sign(p) == np.sign(iv.point) for p in pa))})
    S3 = pd.DataFrame(rows)
    S3.to_csv(OUTD / "restw_check_diff_stats.csv", index=False)
    print("== strict - restw: level per epoch (boot_ci) and change vs pre (boot_delta), key areas\n"
          + S3[S3.variant.isin(["strict_w50", "strict_flat"]) & S3.area.isin(KEYC)].round(3).to_string(index=False))
    print("== variants: significant changes vs pre of (variant - restw), any area\n"
          + S3[(S3.kind == "change_vs_pre") & S3.sig.fillna(False).astype(bool)].round(3).to_string(index=False))
    # ---- results moved
    P1 = swap_precue(D, "", keep_areas)
    P2 = swap_precue(D, "_farcontra", keep_areas)
    P3 = swap_c2l(D)
    P1.to_csv(OUTD / "restw_check_precue_all.csv", index=False)
    P2.to_csv(OUTD / "restw_check_precue_farcontra.csv", index=False)
    P3.to_csv(OUTD / "restw_check_c2l_mean.csv", index=False)
    kk = ["ALM_right", "tjM1_right"]
    print("== R1 pre-cue level change vs pre (all spouts), contra ALM / tjM1\n"
          + P1[P1.area.isin(kk)].pivot_table(index=["area", "epoch"], columns="variant", values="delta").round(2)
          .to_string())
    print("== R2 pre-cue level change vs pre (far-contra target)\n"
          + P2[P2.area.isin(kk)].pivot_table(index=["area", "epoch"], columns="variant", values="delta").round(2)
          .to_string())
    print("== R3 cue->lick mean level change vs pre (mean_rw), contra ALM / tjM1, all + far_contra\n"
          + P3[P3.col.isin(kk) & P3.pos.isin(["all", "far_contra"])]
          .pivot_table(index=["match", "col", "pos", "epoch"], columns="variant", values="delta").round(2).to_string())
    # ---- 4. engaged-count audit
    au = N[["label", "animal", "epoch", "n_cache", "n_beh_now", "n_enl", "n_restgate", "n_trials_daq",
            "dprecue_enl_ALM_right", "dprecue_enl_tjM1_right", "dprecue_enl_maxabs"]].copy()
    au["cache_minus_enl"] = au.n_cache - au.n_enl
    au.to_csv(OUTD / "restw_check_engaged_counts.csv", index=False)
    mm = au[au.cache_minus_enl != 0]
    print(f"== 4. engaged counts: {len(mm)}/{len(au)} sessions where cache != enl_trials engaged\n"
          + mm.round(3).to_string(index=False))
    print(f"   n_cache == n_beh_now in {(au.n_cache == au.n_beh_now).sum()}/{len(au)}")
    save_fd("restw_check", {"D": D, "S3": S3, "REM": REM, "P1": P1, "P2": P2, "P3": P3, "N": N}, root=FDR)
    draw()


# ------------------------------------------------------------------------------------------------ figures
def draw() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = load_fd("restw_check", root=FDR)
    D, S3, P1, P2, P3 = F["D"], F["S3"], F["P1"], F["P2"], F["P3"]
    FIGS.mkdir(parents=True, exist_ok=True)
    acol = {"PS92": "#1f77b4", "PS93": "#d62728", "PS94": "#2ca02c", "PS95": "#9467bd"}
    # Fig 1: (variant - restw) per area x epoch, per-animal dots + cohort CI
    areas = KEYC + ["MOs_rest_right", "MOp_rest_right", "SSp-bfd_right", "VISp_right"]
    areas = [a for a in areas if a in set(D.area)]
    for v in [x for x in SHOW if f"d_{x}" in D]:
        fig, axs = plt.subplots(2, 6, figsize=(17, 6.5), sharey=True)
        for ax, a in zip(axs.ravel(), areas):
            t = D[D.area == a]
            for k, ep in enumerate(EP):
                st = S3[(S3.variant == v) & (S3.area == a) & (S3.epoch == ep) & (S3.kind == "level")]
                if len(st):
                    ax.errorbar(k, st.est.iloc[0], yerr=[[st.est.iloc[0] - st.lo.iloc[0]], [st.hi.iloc[0] - st.est.iloc[0]]],
                                fmt="s", color="k", ms=5, capsize=3, zorder=3)
                for j, an in enumerate(AN):
                    y = t[(t.animal == an) & (t.epoch == ep)][f"d_{v}"].mean()
                    ax.plot(k + (j - 1.5) * 0.08, y, "o", color=acol[an], ms=4, alpha=0.85)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(4), EP, fontsize=7)
            ch = S3[(S3.variant == v) & (S3.area == a) & (S3.kind == "change_vs_pre")]
            star = " ".join(f"{e[:3]}*" for e in ch[ch.sig.fillna(False).astype(bool)].epoch)
            ax.set_title(f"{a.replace('_right', ' contra').replace('_left', ' ipsi')}" + (f"\nchg: {star}" if star else ""),
                         fontsize=8)
        for ax in axs.ravel()[len(areas):]:
            ax.axis("off")
        axs[0, 0].set_ylabel(f"{v} - restw\n(dF/F x1e3)")
        axs[1, 0].set_ylabel(f"{v} - restw\n(dF/F x1e3)")
        for an in AN:
            axs[0, -1].plot([], [], "o", color=acol[an], label=an)
        axs[0, -1].plot([], [], "s", color="k", label="cohort (boot_ci)")
        axs[0, -1].legend(fontsize=7, loc="best")
        fig.suptitle(f"NEURAL. Rest baseline check: {VLAB[v]} minus RESTW (restdock05), per area x epoch. Dots = per-animal "
                     "session mean; squares = cohort level with 95% CI; '*' = change vs pre CI excludes 0 (boot_delta). "
                     "n = 4 mice, PROVISIONAL.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(FIGS / f"restw_check_diff_{v}.png", dpi=130)
        plt.close(fig)
    # Fig 2: before / after for the three results
    specs = [("R1 pre-cue level, all spouts (Fig 14.20)", P1, "area", None, "delta"),
             ("R2 pre-cue level, far-contra target", P2, "area", None, "delta"),
             ("R3 cue->lick mean level (mean_rw, raw), all spouts", P3[(P3.match == "raw") & (P3.pos == "all")], "col",
              None, "delta"),
             ("R3 cue->lick mean level, latency-matched, all", P3[(P3.match == "latency-matched") & (P3.pos == "all")],
              "col", None, "delta")]
    vs = [x for x in ["restw", "daq2", "strict", "strict_w50", "strict_c1_w50", "strict_jaw_w50"] if x in set(P1.variant)]
    vcol = {"restw": "k", "daq2": "#ff7f0e", "strict": "#f4a3a3", "strict_w50": "#d62728", "strict_c1_w50": "#8c564b",
            "strict_jaw_w50": "#17becf"}
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharey="col")
    for c, (ttl, P, key, _x, val) in enumerate(specs):
        for r, a in enumerate(["ALM_right", "tjM1_right"]):
            ax = axs[r, c]
            t = P[P[key] == a]
            for k, ep in enumerate(EP[1:]):
                for j, v in enumerate(vs):
                    q = t[(t.variant == v) & (t.epoch == ep)]
                    if not len(q):
                        continue
                    x = k + (j - (len(vs) - 1) / 2) * 0.13
                    ax.errorbar(x, q[val].iloc[0], yerr=[[q[val].iloc[0] - q.lo.iloc[0]], [q.hi.iloc[0] - q[val].iloc[0]]],
                                fmt="o", color=vcol[v], ms=4, capsize=2, label=v if (k == 0 and r == 0 and c == 0) else None)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(3), EP[1:], fontsize=8)
            ax.set_title(f"{ttl}\n{a.replace('_right', ' contra')}", fontsize=8)
            if c == 0:
                ax.set_ylabel("change vs pre (dF/F x1e3)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("NEURAL. The three restw-referenced results under each rest baseline (per-session baseline swap: "
                 "value + restw - variant). Point + 95% CI (boot_delta, nested animals > sessions). n = 4, PROVISIONAL.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "restw_check_results_moved.png", dpi=130)
    plt.close(fig)
    print("== figures in", FIGS)


def main() -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import parallel
    parallel.pin_blas()
    if plot_only():
        draw()
        return 0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    if "--labels" in sys.argv:
        labels = [x for x in sys.argv[sys.argv.index("--labels") + 1:] if not x.startswith("--")]
    else:
        labels = il.labels()
    E = pd.read_csv(SRC / "enl_licks" / "enl_trials.csv.gz", usecols=["label", "cue_s", "engaged"])
    E = E[E.engaged.astype(str).str.lower().eq("true")]
    items = [(x, np.sort(E[E.label == x].cue_s.to_numpy(float))) for x in labels]
    res, fails = ak.fan_sessions(items, session_worker, jobs=min(jobs, 4), key=lambda it: it[0])
    if fails:
        print(f"!! {len(fails)} sessions failed: {fails}")
    done = sorted(p.stem for p in SESS.glob("*.npz") if p.stem in set(labels))
    summarise(done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
