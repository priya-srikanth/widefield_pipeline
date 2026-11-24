"""S10A onsets (caption review 11-23, ccB helper): what sets the "first movement" of the six-position latency
decomposition, and how robust each candidate signal is. Reads the four-camera rev-up per-trial table (cam3-on run,
`revup_behaviour.py`; first lick = THE fused four-camera lick train), this run's qc/jaw_onsets.csv and the per-session
caches of `s10a_onsets_extract.py`; no re-extraction.

SIGNALS (named categories, Priya 11-23):
  jaw3d    the four-camera 3-D jaw onset at EVERY spout (`trial_classes.jaw_call(mode="3d")` via
           `revup_behaviour.jaw_onsets`: jaw3d_onset_ms where jaw3d_source == "4cam"; a 3-D movement already under way at
           the cue = unknown). PRIMARY kinematic signal.
  jawcls   the trial-class jaw rule (mode "classes": 2-D per-camera onset, 3-D at the close-center spout).
  jaw2d    the per-camera 2-D onset (mode "2d" = production S10A until 11-23; no jaw at the close-center spout).
  mouth    "in-mouth lick (cam1)": the first kept cam1 2-D event (droplets / far-from-mouth out) NOT matched to a lick of
           the fused four-camera train (11-23; production matched against the cam4 2-D licks), cue .. first lick.
  cam1any  production cam1 signal: the first kept cam1 event of any kind (cam4-matched or not), cue .. first lick.
  fr       FaceRhythm lick vigor (2-D cam4 video; kept as its own category): z vs the session's pre-cue (-0.5 .. 0 s,
           all trials, median / 1.4826 MAD); onset = first lag >= 40 ms with z > Z for K consecutive 40-ms samples,
           before the first lick (fused train). Z = 3, K = 3 (production) unless stated.
  frpool   the same, z vs the animal's POOLED pre-stroke pre-cue samples (all pre sessions).
first movement (a variant = a set of signals) = the earliest candidate <= the first-lick latency; a measured trial with
no candidate before the lick has first movement = the lick (revup 0), as `revup_behaviour._revup`.
Trials: canonical (camera working period) and licked; cells = per animal x spout x epoch medians, then the mean over
animals (S10A bars). Ties between signals go to the first in the variant's order (jaw, mouth, fr).

    python s10a_onsets_analysis.py <cache_dir> <out_dir>
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ENC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding")
REVUP = ENC / "revup_fourcam_cam3on" / "revup_trials.csv.gz"
#: the per-session rev-up pickles of that run (full precision; revup_trials.csv.gz keeps ~5 significant digits, so its
#: cue_s / first_lick_s are rounded to 0.1-1 s and cannot be used for cue-relative times)
REVUP_SESS = ENC / "revup_fourcam_cam3on" / "behaviour_sessions_c1far_fused_52c775bb"


def load_trials() -> pd.DataFrame:
    """The rev-up trial table at full precision: per-session pickles + canonical / day / cls from revup_trials."""
    G = pd.read_csv(REVUP, usecols=["label", "trial_id", "canonical", "day", "cls"])
    G["canonical"] = G.canonical.astype(str).str.lower().eq("true")
    Ts = [pickle.loads((REVUP_SESS / f"{x}.pkl").read_bytes())["T"] for x in sorted(G.label.unique())]
    T = pd.concat(Ts, ignore_index=True).merge(G, on=["label", "trial_id"], how="left")
    assert T.canonical.notna().all() and len(T) == len(G)
    for c in ("jaw_judged", "has_cam1", "has_fr"):
        T[c] = T[c].astype(bool)
    return T
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
EP = ["pre", "acute", "subacute", "chronic"]
MOUSE = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}
ZS, KS = (2.0, 3.0, 4.0, 5.0), (2, 3, 4)
CTRL_SHIFT_S = 1.5        # no-cue control: pseudo-cue = cue - 1.5 s (spout docked, before the cue, no licks)
CTRL_MAX_S = 1.44         # control search lags 0.04 .. min(latency, 1.44 s) (ends 60 ms before the real cue)
MATCH_W = (40.0, 60.0, 80.0)   # cam1 event <-> 3-D jaw movement windows (ms); 60 = primary
RISE_UM = 200.0           # 3-D jaw rise counted as an opening = the 3-D onset floor (0.20 mm, jaw_onsets.OVERRIDE_3D)
#: variants: name -> ordered signal columns
VARIANTS = {
    "prod_2d3":       ["jaw2d", "cam1any", "fr"],          # production S10A (reproduce check)
    "cls_3sig":       ["jawcls", "cam1any", "fr"],         # task 1: three signals, 4-cam jaw at the near spout
    "cls_jaw":        ["jawcls"],                          # task 1: jaw only (classes rule)
    "fr_only":        ["fr"],
    "cls_3sig_pool":  ["jawcls", "cam1any", "frpool"],     # task 2b
    "j3_only":        ["jaw3d"],                           # 3-D jaw only (primary kinematic)
    "j3_mouth":       ["jaw3d", "mouth"],
    "j3_fr":          ["jaw3d", "fr"],
    "j3_all":         ["jaw3d", "mouth", "fr"],
    "j3_all_pool":    ["jaw3d", "mouth", "frpool"],
}
AVAIL = {"jaw2d": "j2_judged", "jawcls": "jc_judged", "jaw3d": "j3_judged", "cam1any": "has_cam1",
         "mouth": "has_cam1", "fr": "has_fr", "frpool": "has_fr"}


def lags():
    import revup_behaviour as rb
    return rb.LAGS_CUE


def onset(Z: np.ndarray, lim_s: np.ndarray, z: float, k: int, lag: np.ndarray, lo: float = 0.04) -> np.ndarray:
    """`revup_behaviour._onset` with explicit (z, k) and search end ``lim_s`` (s, per row)."""
    out = np.full(len(Z), np.nan)
    a = (Z > z).astype(int)
    run = np.zeros_like(a)
    for j in range(k):
        run[:, :a.shape[1] - j] += a[:, j:]
    ok = (run >= k) & (lag[None, :] >= lo)
    ok &= lag[None, :] <= np.where(np.isfinite(lim_s), lim_s, lag[-1])[:, None]
    has = ok.any(1)
    out[has] = lag[ok[has].argmax(1)]
    return out


def robust(bv: np.ndarray) -> tuple[float, float]:
    bv = bv[np.isfinite(bv)]
    med = float(np.median(bv))
    s = 1.4826 * float(np.median(np.abs(bv - med))) or float(np.std(bv)) or 1.0
    return med, s


def interp(t: np.ndarray, v: np.ndarray, ft: np.ndarray) -> np.ndarray:
    from wfield_local import face_rhythm_outputs as fro
    shp = ft.shape
    f = ft.ravel()
    out = np.full(f.size, np.nan)
    ok = np.isfinite(f)
    o = np.argsort(f[ok])
    vals = fro.to_frames(t, v[:, None].astype(float), f[ok][o], 0.5)[:, 0]
    idx = np.flatnonzero(ok)[o]
    out[idx] = vals
    return out.reshape(shp)


def first_in(times: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    import revup_behaviour as rb
    return rb._first_at(np.sort(times), lo, hi)


# ------------------------------------------------------------------------------------------------ per session
def per_session(label: str, g: pd.DataFrame, cache: Path) -> dict:
    import revup_behaviour as rb
    LAG = lags()
    d = pickle.loads((cache / f"{label}.pkl").read_bytes())
    an = g.animal.iloc[0]
    sess = f"{an}_2026{label.split('_')[1]}"
    ids = g.trial_id.to_numpy()
    cue = g.cue_s.to_numpy(float)
    fl = g.first_lick_s.to_numpy(float)
    R = pd.DataFrame({"label": label, "trial_id": ids})
    for m, tag, nm in (("2d", "j2", "jaw2d"), ("classes", "jc", "jawcls"), ("3d", "j3", "jaw3d")):
        ms, jj = rb.jaw_onsets(sess, ids, m)
        R[f"{nm}_ms"], R[f"{tag}_judged"] = ms, jj
    # cam1: in-mouth (not matched to the fused train) and production "any" (matched to cam4 or not), cue .. first lick
    C = d["cam1"]
    upto = np.where(np.isfinite(fl), fl, g.bound_s.to_numpy(float))
    ev = {}
    if C is not None:
        e_any = C.loc[C.keep_old, "daq_on_s"].to_numpy(float)
        e_mouth = C.loc[C.keep_new & ~C.fused_matched, "daq_on_s"].to_numpy(float)
        e_small_old = C.loc[C.keep_old & ~C.cam4_matched, "daq_on_s"].to_numpy(float)
        R["cam1any_ms"] = (first_in(e_any, cue, upto + 1e-9) - cue) * 1000
        R["mouth_ms"] = (first_in(e_mouth, cue, upto + 1e-9) - cue) * 1000
        R["n_mouth_pre"] = rb._count(np.sort(e_mouth), cue, upto)
        R["n_small_old_pre"] = rb._count(np.sort(e_small_old), cue, upto)
        flip = C.keep_old & (C.cam4_matched != C.fused_matched)
        ev = {"n_events": int(len(C)), "n_kept_old": int(C.keep_old.sum()), "n_kept_new": int(C.keep_new.sum()),
              "n_unmatched_cam4": int((C.keep_old & ~C.cam4_matched).sum()),
              "n_unmatched_fused": int((C.keep_new & ~C.fused_matched).sum()),
              "n_cam4unm_to_fusedm": int((flip & ~C.cam4_matched).sum()),
              "n_cam4m_to_fusedunm": int((flip & C.cam4_matched).sum())}
        mouth_t = e_mouth
    else:
        R["cam1any_ms"] = R["mouth_ms"] = np.nan
        R["n_mouth_pre"] = R["n_small_old_pre"] = np.nan
        mouth_t = np.array([])
    R["has_cam1"] = C is not None
    # FR vigor on the production grid
    F = d["fr"]
    out = {"R": R, "ev": ev, "label": label, "animal": an}
    if F is None:
        R["has_fr"] = False
        return out
    V = interp(F["t"], F["v"], cue[:, None] + LAG[None, :]).astype(np.float32)
    base = LAG < 0
    med, s = robust(V[:, base].ravel())
    R["has_fr"] = True
    out.update({"V": V, "base_med": med, "base_s": s, "base_vals": V[:, base].ravel()})
    # no-cue control grid: pseudo-cue = cue - 1.5 s, lags 0.04 .. 1.44 s
    Vc = interp(F["t"], F["v"], (cue - CTRL_SHIFT_S)[:, None] + LAG[None, :]).astype(np.float32)
    L = d["licks"]
    lk = np.sort(np.r_[L.daq_on_s.to_numpy(float), mouth_t])
    nlk = rb._count(lk[np.isfinite(lk)], cue - CTRL_SHIFT_S - 0.5, cue)
    docked = (cue - g.strobe_s.to_numpy(float)) >= CTRL_SHIFT_S + 0.5
    R["ctrl_ok"] = docked & (nlk == 0)
    out["Vc"] = Vc
    # 3-D jaw rise around each in-mouth event before the first lick (the match test) + per-trial traces
    J = d["jaw3d"]
    if J is not None and len(mouth_t):
        tr = {}
        tid = J["trial_id"]
        brk = np.flatnonzero(np.diff(tid)) + 1
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(tid)]):
            tr[int(tid[a])] = (J["t_daq_ms"][a:b], J["d_um"][a:b].astype(float))
        rows = []
        for i, (t_id, c0, u) in enumerate(zip(ids, cue, upto)):
            es = mouth_t[(mouth_t >= c0) & (mouth_t <= u + 1e-9)]
            for k, te in enumerate(es):
                te_ms = (te - c0) * 1000
                r = {"label": label, "trial_id": int(t_id), "k": k, "t_ms": te_ms, "row": i}
                if int(t_id) in tr:
                    tt, dd = tr[int(t_id)]
                    for w in MATCH_W:
                        win = (tt >= te_ms - w) & (tt <= te_ms + w)
                        pre = (tt >= te_ms - w - 80) & (tt <= te_ms)
                        r[f"rise_{int(w)}"] = (np.nanmax(dd[win]) - np.nanmin(dd[pre])
                                               if np.isfinite(dd[win]).any() and np.isfinite(dd[pre]).any() else np.nan)
                rows.append(r)
        out["M"] = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------------------------------------------ helpers
def move(T: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lat = T.lat_ms.to_numpy(float)
    cand = np.column_stack([T[f"{c}_ms"].to_numpy(float) for c in cols])
    cand = np.where(cand <= lat[:, None] + 1e-6, cand, np.nan)
    meas = np.zeros(len(T), bool)
    for c in cols:
        meas |= T[AVAIL[c]].to_numpy(bool)
    mv = np.nanmin(np.where(np.isfinite(cand), cand, np.inf), 1)
    src = np.full(len(T), "none", object)
    fin = np.isfinite(mv)
    for j in range(len(cols) - 1, -1, -1):
        src = np.where(fin & np.isfinite(cand[:, j]) & (np.abs(cand[:, j] - mv) < 1e-6), cols[j], src)
    mv = np.where(fin, mv, np.where(np.isfinite(lat) & meas, lat, np.nan))
    src = np.where(meas, src, "unmeasured")
    return mv, src


def cells(T: pd.DataFrame, mv: np.ndarray, src: np.ndarray, cols: list[str]) -> pd.DataFrame:
    X = T[["animal", "pos", "epoch", "lat_ms"]].copy()
    X["mv"], X["src"] = mv, src
    rows = []
    for (a, p, e), g in X.groupby(["animal", "pos", "epoch"]):
        g = g[np.isfinite(g.mv)]
        if not len(g):
            continue
        r = {"animal": a, "pos": p, "epoch": e, "n_licked": len(g), "lat_med_ms": np.median(g.lat_ms),
             "wait_med_ms": np.median(g.mv), "revup_med_ms": np.median(g.lat_ms - g.mv)}
        for c in [*cols, "none"]:
            r[f"share_{c}"] = float((g.src == c).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def mean_animals(C: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in C.columns if c not in ("animal", "pos", "epoch")]
    S = C.groupby(["pos", "epoch"])[num].mean().reset_index()
    S["pos"] = pd.Categorical(S.pos, POS, ordered=True)
    S["epoch"] = pd.Categorical(S.epoch, EP, ordered=True)
    return S.sort_values(["pos", "epoch"])


def desc(x: pd.Series) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "median": np.median(x), "q25": np.percentile(x, 25), "q75": np.percentile(x, 75),
            "frac_lt0": float((x < 0).mean()), "frac_lt_m100": float((x < -100).mean()),
            "frac_abs_le40": float((np.abs(x) <= 40).mean()), "frac_gt100": float((x > 100).mean())}


def by(T: pd.DataFrame, col: str, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in T.groupby(keys):
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **desc(g[col])})
    return pd.DataFrame(rows)


def animal_mean(D: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """mean over animals of a per-animal table (rule 8: per-animal first, then this)."""
    num = [c for c in D.columns if c not in ("animal", *keys) and D[c].dtype.kind in "fi"]
    return D.groupby(keys)[num].mean().reset_index()


# ------------------------------------------------------------------------------------------------ main
def main(cache: str, out: str) -> None:
    cache, out = Path(cache), Path(out)
    tab = out / "tables_ccB1123"
    tab.mkdir(parents=True, exist_ok=True)
    LAG = lags()
    T0 = load_trials()
    f_res = cache / "_analysis_per_session_ccB1123.pkl"     # per-session results (re-runs of the summaries reuse it)
    if f_res.exists():
        res = pickle.loads(f_res.read_bytes())
    else:
        res = []
        for lab, g in T0.groupby("label", sort=True):
            res.append(per_session(lab, g.reset_index(drop=True), cache))
            print(lab, flush=True)
        f_res.write_bytes(pickle.dumps(res))
    # ---- assemble per-trial
    R = pd.concat([r["R"] for r in res], ignore_index=True)
    T = T0.merge(R.drop(columns=["has_cam1", "has_fr"]), on=["label", "trial_id"], how="left")
    assert len(T) == len(T0)
    # FR z (own baseline, pooled pre baseline) and onsets for every (z, k) on all trials
    pre_pool = {}
    for an in sorted(T.animal.unique()):
        vals = np.concatenate([r["base_vals"] for r in res if r["animal"] == an and "V" in r
                               and T0.loc[T0.label == r["label"], "epoch"].iloc[0] == "pre"])
        pre_pool[an] = robust(vals)
    base_rows, V_own, V_pool, C_own, C_pool = [], [], [], [], []
    for r in res:
        n = len(r["R"])
        if "V" not in r:
            nanb = np.full((n, len(LAG)), np.nan, np.float32)
            V_own.append(nanb); V_pool.append(nanb); C_own.append(nanb); C_pool.append(nanb)
            continue
        m, s = r["base_med"], r["base_s"]
        pm, ps = pre_pool[r["animal"]]
        V_own.append((r["V"] - m) / s); V_pool.append((r["V"] - pm) / ps)
        C_own.append((r["Vc"] - m) / s); C_pool.append((r["Vc"] - pm) / ps)
        base_rows.append({"label": r["label"], "animal": r["animal"], "base_med": m, "base_mad_s": s,
                          "pool_med": pm, "pool_mad_s": ps})
    Zo, Zp, Co, Cp = (np.concatenate(x, 0) for x in (V_own, V_pool, C_own, C_pool))
    lim = T.lat_ms.to_numpy(float) / 1000
    T["fr_ms"] = onset(Zo, lim, 3.0, 3, LAG) * 1000
    T["frpool_ms"] = onset(Zp, lim, 3.0, 3, LAG) * 1000
    chk = np.nanmax(np.abs(T.fr_ms - T.fr_onset_ms))
    agree = np.mean(np.isclose(T.fr_ms, T.fr_onset_ms, atol=1, equal_nan=True))
    print(f"FR onset recomputed vs production: max |diff| {chk:.3g} ms, agree {agree:.4f}")
    B = pd.DataFrame(base_rows).merge(T0.groupby("label").epoch.first().reset_index(), on="label")
    B.to_csv(tab / "fr_baseline_by_session.csv", index=False, float_format="%.5g")
    # ---- trial set: canonical + licked
    W = T[T.canonical & np.isfinite(T.lat_ms)].copy()
    rowsW = W.index.to_numpy()
    lines = [f"FR onset recomputed vs production fr_onset_ms: max |diff| = {chk:.3g} ms; agree within 1 ms: "
             f"{agree:.4f} of trials"]
    # ---- variants
    allcells, summ = {}, []
    for name, cols in VARIANTS.items():
        mv, src = move(W, cols)
        W[f"mv_{name}"], W[f"src_{name}"] = mv, src
        C = cells(W, mv, src, cols)
        C.to_csv(tab / f"cells_{name}.csv", index=False, float_format="%.4g")
        allcells[name] = C
        S = mean_animals(C)
        S.insert(0, "variant", name)
        summ.append(S)
    if "move_ms" in W:
        d0 = np.nanmax(np.abs(W.mv_prod_2d3 - W.move_ms))
        lines.append(f"production move_ms reproduced by variant prod_2d3: max |diff| = {d0:.3g} ms")
    pd.concat(summ).to_csv(tab / "summary_mean_of_animals_all_variants.csv", index=False, float_format="%.4g")
    # shares by epoch (per animal, then mean)
    sh = []
    for name in VARIANTS:
        X = W.groupby(["animal", "epoch"])[f"src_{name}"].value_counts(normalize=True).unstack(fill_value=0)
        X = X.groupby("epoch").mean().reindex(EP)
        X.insert(0, "variant", name)
        sh.append(X.reset_index())
    pd.concat(sh).to_csv(tab / "shares_by_epoch.csv", index=False, float_format="%.4g")
    sh_an = []
    for name in ("prod_2d3", "j3_all", "j3_all_pool"):
        X = W.groupby(["animal", "epoch"])[f"src_{name}"].value_counts(normalize=True).unstack(fill_value=0)
        X.insert(0, "variant", name)
        sh_an.append(X.reset_index())
    pd.concat(sh_an).to_csv(tab / "shares_by_animal_epoch.csv", index=False, float_format="%.4g")
    # ---- 3-D jaw availability per spout x epoch (all canonical trials, licked or not)
    J = pd.read_csv(ENC / "qc" / "jaw_onsets.csv", usecols=["session", "trial_id", "jaw3d_moved", "jaw3d_at_cue",
                                                             "jaw3d_source", "jaw3d_onset_ms"], low_memory=False)
    J["label"] = J.session.str.slice(0, 5) + J.session.str.slice(9)
    A = T[T.canonical].merge(J, on=["label", "trial_id"], how="left")
    A["call"] = A.jaw3d_source.eq("4cam") & A.jaw3d_moved.notna()
    A["at_cue"] = A.jaw3d_at_cue.astype(str).str.lower().eq("true")
    av = A.groupby(["animal", "pos", "epoch"]).agg(n=("call", "size"), call=("call", "mean"),
                                                   judged=("j3_judged", "mean"),
                                                   moved=("jaw3d_moved", lambda x: np.nanmean(x == 1)),
                                                   at_cue=("at_cue", "mean")).reset_index()
    av.to_csv(tab / "jaw3d_availability_animal.csv", index=False, float_format="%.4g")
    animal_mean(av, ["pos", "epoch"]).to_csv(tab / "jaw3d_availability.csv", index=False, float_format="%.4g")
    # onset before the lick, per signal (licked canonical)
    for c in ("jaw2d", "jawcls", "jaw3d", "mouth", "cam1any", "fr", "frpool"):
        W[f"pre_{c}"] = np.isfinite(W[f"{c}_ms"]) & (W[f"{c}_ms"] <= W.lat_ms + 1e-6)
    pb = W.groupby(["animal", "pos", "epoch"])[[f"pre_{c}" for c in ("jaw2d", "jawcls", "jaw3d", "mouth", "cam1any",
                                                                      "fr", "frpool")]].mean().reset_index()
    animal_mean(pb, ["pos", "epoch"]).to_csv(tab / "signal_before_lick.csv", index=False, float_format="%.4g")
    animal_mean(pb, ["epoch"]).to_csv(tab / "signal_before_lick_by_epoch.csv", index=False, float_format="%.4g")
    # ---- pairwise onset differences (both onsets before the lick)
    pairs = {"fr_minus_jaw3d": ("fr", "jaw3d"), "fr_minus_jaw2d": ("fr", "jaw2d"), "frpool_minus_jaw3d":
             ("frpool", "jaw3d"), "mouth_minus_jaw3d": ("mouth", "jaw3d"), "jaw3d_minus_jaw2d": ("jaw3d", "jaw2d")}
    for nm, (a, b) in pairs.items():
        both = W[f"pre_{a}"] & W[f"pre_{b}"] if nm != "jaw3d_minus_jaw2d" else (np.isfinite(W.jaw3d_ms)
                                                                                 & np.isfinite(W.jaw2d_ms))
        W[nm] = np.where(both, W[f"{a}_ms"] - W[f"{b}_ms"], np.nan)
        by(W, nm, ["animal", "epoch"]).to_csv(tab / f"diff_{nm}_animal_epoch.csv", index=False, float_format="%.4g")
        by(W, nm, ["animal", "pos", "epoch"]).to_csv(tab / f"diff_{nm}_animal_pos_epoch.csv", index=False,
                                                     float_format="%.4g")
        animal_mean(by(W, nm, ["animal", "epoch"]), ["epoch"]).to_csv(tab / f"diff_{nm}_epoch.csv", index=False,
                                                                       float_format="%.4g")
        animal_mean(by(W, nm, ["animal", "pos", "epoch"]), ["pos", "epoch"]).to_csv(
            tab / f"diff_{nm}_pos_epoch.csv", index=False, float_format="%.4g")
    # ---- cam1 event matching vs the 3-D jaw (in-mouth events before the first lick, licked canonical trials)
    M = pd.concat([r["M"] for r in res if "M" in r], ignore_index=True)
    M = M.merge(T[["label", "trial_id", "animal", "epoch", "pos", "canonical", "lat_ms", "jaw3d_ms"]],
                on=["label", "trial_id"])
    M = M[M.canonical & np.isfinite(M.lat_ms)]
    for w in MATCH_W:
        on_ok = np.abs(M.jaw3d_ms - M.t_ms) <= w
        M[f"match_{int(w)}"] = on_ok | (M[f"rise_{int(w)}"] >= RISE_UM)
        M[f"match_onset_{int(w)}"] = on_ok
    M.to_csv(tab / "mouth_events_jaw3d_match.csv.gz", index=False, float_format="%.5g")
    mcols = [f"match_{int(w)}" for w in MATCH_W] + [f"match_onset_{int(w)}" for w in MATCH_W]
    mm = M.groupby(["animal", "pos", "epoch"])[mcols].mean().reset_index()
    mm["n_events"] = M.groupby(["animal", "pos", "epoch"]).size().to_numpy()
    mm.to_csv(tab / "mouth_match_animal_pos_epoch.csv", index=False, float_format="%.4g")
    me = M.groupby(["animal", "epoch"])[mcols].mean().reset_index()
    me["n_events"] = M.groupby(["animal", "epoch"]).size().to_numpy()
    me.to_csv(tab / "mouth_match_animal_epoch.csv", index=False, float_format="%.4g")
    animal_mean(me, ["epoch"]).to_csv(tab / "mouth_match_epoch.csv", index=False, float_format="%.4g")
    animal_mean(mm, ["pos", "epoch"]).to_csv(tab / "mouth_match_pos_epoch.csv", index=False, float_format="%.4g")
    # QC flag: in-mouth sets the first movement in j3_all but the 3-D jaw shows nothing (no onset <= lick, no rise)
    first_ev = M[M.k == 0].set_index(["label", "trial_id"])
    key = pd.MultiIndex.from_frame(W[["label", "trial_id"]])
    W["mouth_jaw_match60"] = first_ev["match_60"].reindex(key).to_numpy()
    W["qc_mouth_no_jaw"] = (W.src_j3_all == "mouth") & ~W.pre_jaw3d & (W.mouth_jaw_match60 == False)  # noqa: E712
    q = W.groupby(["animal", "epoch"]).agg(n=("qc_mouth_no_jaw", "size"), qc=("qc_mouth_no_jaw", "mean"),
                                           mouth_sets=("src_j3_all", lambda s: np.mean(s == "mouth"))).reset_index()
    q.to_csv(tab / "qc_mouth_without_jaw3d.csv", index=False, float_format="%.4g")
    # cam1 match change (fused vs cam4 2-D)
    EV = pd.DataFrame([{"label": r["label"], "animal": r["animal"], **r["ev"]} for r in res if r["ev"]])
    EV.to_csv(tab / "cam1_match_change_by_session.csv", index=False)
    W["d_npre"] = W.n_mouth_pre - W.n_small_old_pre
    lines.append("cam1 events, all sessions: " + ", ".join(f"{c} {int(EV[c].sum())}" for c in EV.columns[2:]))
    lines.append(f"cam1 unmatched events before the first lick (licked canonical trials): cam4 rule "
                 f"{int(W.n_small_old_pre.sum())}, fused rule {int(W.n_mouth_pre.sum())}; trials changed "
                 f"{int((W.d_npre != 0).sum())} of {len(W)}")
    # ---- (2a) FR baseline spread by animal x epoch
    Bs = B.groupby(["animal", "epoch"])[["base_med", "base_mad_s"]].agg(["mean", "median", "count"])
    Bs.columns = ["_".join(c) for c in Bs.columns]
    Bs.reset_index().to_csv(tab / "fr_baseline_by_animal_epoch.csv", index=False, float_format="%.5g")
    # ---- (2c) z x K sweep: false onsets in the no-cue control, real onsets, FR share, far-contra decomposition
    sw = []
    ctrl = W.ctrl_ok.fillna(False).to_numpy(bool)
    lim_w = np.minimum(W.lat_ms.to_numpy(float) / 1000, CTRL_MAX_S)
    for base_nm, Zm, Cm in (("own", Zo, Co), ("pooled_pre", Zp, Cp)):
        Zw, Cw = Zm[rowsW], Cm[rowsW]
        for z in ZS:
            for k in KS:
                real = onset(Zw, lim_w, z, k, LAG)
                fake = onset(Cw, lim_w, z, k, LAG)
                Wt = W[["animal", "pos", "epoch", "lat_ms"]].copy()
                Wt["real"], Wt["fake"], Wt["ctrl"] = np.isfinite(real), np.isfinite(fake), ctrl
                Wt["fr_ms"] = onset(Zw, W.lat_ms.to_numpy(float) / 1000, z, k, LAG) * 1000
                Wt["jaw3d_ms"], Wt["mouth_ms"] = W.jaw3d_ms, W.mouth_ms
                Wt["j3_judged"], Wt["has_cam1"], Wt["has_fr"] = W.j3_judged, W.has_cam1, W.has_fr
                mv, src = move(Wt, ["jaw3d", "mouth", "fr"])
                Wt["src"], Wt["mv"] = src, mv
                for (a, e), g in Wt.groupby(["animal", "epoch"]):
                    gc = g[g.ctrl]
                    fc = g[g.pos == "far_contra"]
                    sw.append({"baseline": base_nm, "z": z, "k": k, "animal": a, "epoch": e,
                               "n_ctrl": len(gc), "false_rate": gc.fake.mean() if len(gc) else np.nan,
                               "real_rate_ctrltrials": gc.real.mean() if len(gc) else np.nan,
                               "real_rate": g.real.mean(), "fr_share": np.mean(g.src == "fr"),
                               "fc_wait_ms": np.nanmedian(fc.mv), "fc_revup_ms": np.nanmedian(fc.lat_ms - fc.mv)})
    SW = pd.DataFrame(sw)
    SW.to_csv(tab / "fr_threshold_sweep_animal_epoch.csv", index=False, float_format="%.4g")
    animal_mean(SW, ["baseline", "z", "k", "epoch"]).to_csv(tab / "fr_threshold_sweep.csv", index=False,
                                                            float_format="%.4g")
    # ---- per-trial table + z traces for the clips
    keep = ["label", "animal", "epoch", "day", "trial_id", "pos", "cue_s", "strobe_s", "first_lick_s", "lat_ms",
            "jaw2d_ms", "jawcls_ms", "jaw3d_ms", "j3_judged", "mouth_ms", "cam1any_ms", "n_mouth_pre", "fr_ms",
            "frpool_ms", "fr_onset_ms", "move_ms", "ctrl_ok", "mouth_jaw_match60", "qc_mouth_no_jaw",
            *pairs.keys(), *[f"mv_{n}" for n in VARIANTS], *[f"src_{n}" for n in VARIANTS]]
    W[keep].to_csv(out / "s10a_trials_ccB1123.csv.gz", index=False, float_format="%.10g")
    np.savez_compressed(out / "s10a_frz_ccB1123.npz", z_own=Zo[rowsW].astype(np.float32),
                        z_pool=Zp[rowsW].astype(np.float32), lags=LAG, label=W.label.to_numpy(str),
                        trial_id=W.trial_id.to_numpy())
    (tab / "checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
