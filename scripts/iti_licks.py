"""SELF-INITIATED (ITI) 3-D LICKS: pre-lick ramp timing + target-free execution coding (Priya 10-25 night: "let's also do
the inter-trial 3d licks, pre-lick ramp timing"; proposed 10-25 as analyses 3 and 5).

WHY. Every trial lick follows a cue and has a target, so the trial readouts mix the plan / target with the movement and
with the cue response. ITI licks (whole-session 3-D table `pose3d/kinematics_all/licks.csv.gz`, context iti) have NO
trial, NO cue and NO reward schedule; the spout is still in place (ITI licks carry its position), so they are
target-free in the TASK sense but can still be aimed at the spout present. Two questions:
 (A) RAMP TIMING: how early does ALM / tjM1 activity rise before a self-initiated lick, and does that change after the
     stroke? (Bout-start ITI licks: no lick in the preceding 2 s -- 1 s as a secondary set, kind iti_bout_gap1 --, no cue /
     spout move within [-2, +0.75] s; ONSET-aligned, baseline -1.5..-1.0 s.) Trial first
     licks are a secondary set (a cue precedes them by the reaction time; read within RT terciles).
 (B) EXECUTION CODING WITHOUT A TRIAL: cross-validated decoding of each ITI lick's 3-D lateral direction (side_ml, mm)
     from area activity, per lag around the lick peak (-0.5 .. +0.5 s), vs n-matched in-trial (post-cue) licks of the
     same session; and the WITHIN-SPOUT direction (side_ml minus its spout's session mean: pure execution variation,
     no spout identity). Plus a FROZEN version: decoder trained on each animal's August pre ITI licks (pre sessions
     leave-one-session-out), applied to every post session. If ITI execution coding is intact while the trial
     far-contra code is lost, plan / target and execution dissociate.
ROIs (Priya 10-25: "ALM and tjM1 ... are the anterolateral 1/3 to 1/2 of MOs and MOp"): per hemisphere, the ROI_FRAC most
anterolateral pixels of MOs (ALM) / MOp (tjM1) on the eroded stat mask, ranked by z(anterior) + z(lateral distance from
the midline) within the area (Allen grid shared by all sessions; figure iti_licks_rois.png to check placement). _right =
contralesional (image right), _left = ipsilesional.
Signals: Allen-area / ROI means of the adopted hemo-corrected SVD (`combined_encoding.load_imaging`, the encoder's
targets), slow drift removed (the encoder's slow basis), dF/F x1e3. Lick times on the DAQ clock from the session's
licks_all.csv via lick_uid (NOT the kinematics table's daq_peak_s, which is rounded to 4 significant digits = up to
0.5 s off in a 1 h session): ramp PETHs are aligned to lick ONSET, direction decoding to the 3-D PEAK; the clock match to
the behaviour licks is checked per session and recorded.
Calcium indicator lag delays every rise; the pre -> post CHANGE is the readout, not the absolute lead.
Usage: python scripts/iti_licks.py [--jobs 3] [--labels ...]   (extract + summarise)
       python scripts/iti_licks.py --summarise                   (from saved sessions)
Output: <E>/iti_licks/ (sessions/<label>.npz, tables, figures)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

OUT = E / "iti_licks"
ROI_FRAC = 0.4
LAGS_P = np.round(np.arange(-1.5, 0.7501, 0.05), 3)          # PETH lags (s from lick peak)
LAGS_C = np.round(np.arange(-1.0, 2.5001, 0.05), 3)          # cue-aligned lags (trial first licks)
LAGS_D = np.round(np.arange(-0.5, 0.5001, 0.05), 3)          # decoding lags
BASE = (-1.5, -1.0)
BOUT_GAP = 1.0                     # extraction: ITI onsets with no lick for >= 1 s (each lick's gap is saved)
GAP_MAIN = 2.0                     # summaries: >= 2 s, so the -1.5..-1.0 s baseline starts >= 0.5 s after the last lick
#                                    (with 1 s the acute baseline still carried the previous licks' decay)
CUE_EXCL = (-2.0, 0.75)
N_TRIAL_MAX = 1500
RAMP_ROIS = ["ALM_right", "ALM_left", "tjM1_right", "tjM1_left", "MOs_right", "MOs_left", "MOp_right", "MOp_left",
             "SSp-m_right", "SSp-m_left", "cortex"]
EP4 = ["pre", "acute", "subacute", "chronic"]


def lick_table() -> pd.DataFrame:
    from wfield_local.paths import PathResolver
    p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
    return pd.read_csv(p, usecols=["session", "context", "lick_uid", "side_ml", "out_ap", "position", "trial_id"])


def roi_masks(parea, pstat, names, flat, shape, frac: float = ROI_FRAC) -> dict:
    """{name: boolean over brain pixels}: every Allen area (stat mask) + ALM / tjM1 per hemisphere (module docstring)."""
    H, W = (int(x) for x in shape)
    r, c = flat // W, flat % W
    m = {names.get(int(a), str(a)): (parea == a) & pstat for a in sorted(np.unique(parea[pstat])) if a != 0}
    left = np.zeros(len(flat), bool)
    right = np.zeros(len(flat), bool)
    for n, mk in m.items():
        left |= mk & n.endswith("_left")
        right |= mk & n.endswith("_right")
    mid = (np.median(c[left]) + np.median(c[right])) / 2.0 if left.any() and right.any() else W / 2.0
    for src, dst in (("MOs", "ALM"), ("MOp", "tjM1")):
        for h in ("left", "right"):
            mk = m.get(f"{src}_{h}")
            if mk is None or not mk.any():
                continue
            ant = -r[mk].astype(float)                      # anterior = top of the image
            lat = np.abs(c[mk] - mid).astype(float)
            u = (ant - ant.mean()) / (ant.std() or 1) + (lat - lat.mean()) / (lat.std() or 1)
            sel = np.zeros(len(flat), bool)
            sel[np.flatnonzero(mk)[u >= np.quantile(u, 1 - frac)]] = True
            m[f"{dst}_{h}"] = sel
    m["cortex"] = pstat.copy()
    return m


def _behaviour(label: str):
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    try:
        return ce.load_behaviour(an, date, PathResolver())
    except FileNotFoundError as ex:                      # no cam1 licks: cam4 timing (as frozen_encoder.session_design)
        if "cam1_licks" not in str(ex) or not os.environ.get(ce.LICKS_ENV):
            raise
        saved = os.environ.pop(ce.LICKS_ENV)
        try:
            return ce.load_behaviour(an, date, PathResolver())
        finally:
            os.environ[ce.LICKS_ENV] = saved


def _sample(R: np.ndarray, ft: np.ndarray, t: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """(n events, n lags, n columns) of R interpolated at t + lags."""
    tt = (t[:, None] + lags[None, :]).ravel()
    out = np.empty((len(tt), R.shape[1]), np.float32)
    for j in range(R.shape[1]):
        out[:, j] = np.interp(tt, ft, R[:, j])
    return out.reshape(len(t), len(lags), R.shape[1])


def extract(item) -> dict:
    """One session -> <out>/sessions/<label>.npz (module-level for spawn). item = (label, lick table path or None)."""
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    label, out = item
    f = Path(out) / "sessions" / f"{label}.npz"
    if f.exists():
        return {"label": label, "cached": True}
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    B = _behaviour(label)
    T = B.trials
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    K = lick_table()
    K = K[K.session == f"{an}_{date}"].copy()
    # FULL-PRECISION TIMES: the kinematics_all table stores daq_peak_s to 4 significant digits (0.1 s resolution after
    # 100 s, 1 s after 1000 s); lick_uid = the row of the session's licks_all.csv, which has the exact DAQ onset / peak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["context", "daq_on_s", "daq_peak_s"])
    uid = K.lick_uid.to_numpy(int)
    if not (A.context.to_numpy()[uid] == K.context.to_numpy()).all():
        raise RuntimeError(f"{label}: lick_uid join to licks_all.csv does not match on context")
    K["daq_peak_s"] = A.daq_peak_s.to_numpy(float)[uid]
    K["daq_on_s"] = A.daq_on_s.to_numpy(float)[uid]
    K = K.sort_values("daq_peak_s").reset_index(drop=True)
    # every lick ONSET (all cam4 licks of the windows + the behaviour licks) for the bout rule
    t_all = np.sort(np.concatenate([A.daq_on_s.to_numpy(float), np.asarray(B.lick_on_s, float)]))
    t_all = t_all[np.isfinite(t_all)]
    pc = K[K.context == "post_cue"].daq_peak_s.to_numpy(float)
    bp = np.sort(np.asarray(B.lick_peak_s, float)[np.isfinite(B.lick_peak_s)])
    if len(pc) and len(bp):
        j = np.clip(np.searchsorted(bp, pc), 1, len(bp) - 1)
        clock = float(np.median(np.minimum(np.abs(pc - bp[j - 1]), np.abs(pc - bp[j]))))
    else:
        clock = np.nan
    ev = np.sort(np.concatenate([T.cue_s.to_numpy(float), T.strobe_s.to_numpy(float)]))
    ev = ev[np.isfinite(ev)]
    lo, hi = ft[0] - LAGS_P[0], ft[-1] - LAGS_P[-1]

    def near_event(t, a, b):
        j = np.searchsorted(ev, t + a)
        return (j < len(ev)) & (ev[np.minimum(j, len(ev) - 1)] <= t + b)
    # (A) bout-start ITI licks
    iti = K[(K.context == "iti") & np.isfinite(K.side_ml) & np.isfinite(K.daq_on_s)].copy()
    ti = iti.daq_peak_s.to_numpy(float)
    to = iti.daq_on_s.to_numpy(float)                    # ramp PETHs are ONSET-aligned
    k = np.searchsorted(t_all, to - 0.05)                # onsets within 50 ms = the same lick (cam4 vs behaviour copy)
    gap = to - np.where(k > 0, t_all[np.maximum(k - 1, 0)], -np.inf)
    bout = (gap >= BOUT_GAP) & ~np.array([near_event(t, *CUE_EXCL) for t in to], bool) & (to > lo) & (to < hi)
    # trial first licks (engaged trials), ONSET-aligned (behaviour lick onsets)
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    Eg = T[T.engaged & np.isfinite(T.cue_s)]
    fl, rt, fpos = [], [], []
    for c, s, p in zip(Eg.cue_s.to_numpy(float), Eg.stop_s.to_numpy(float), Eg.pos.to_numpy()):
        s = s if np.isfinite(s) else c + 3.5
        j = np.searchsorted(bo, c)
        if j < len(bo) and bo[j] <= s and lo < bo[j] < hi and c + LAGS_C[0] > ft[0] and c + LAGS_C[-1] < ft[-1]:
            fl.append(bo[j]); rt.append(bo[j] - c); fpos.append(p)
    fl, rt = np.asarray(fl), np.asarray(rt)
    # (B) decoding sets: all ITI licks with 3-D direction; post-cue licks (random subset, seeded)
    dlo, dhi = ft[0] - LAGS_D[0], ft[-1] - LAGS_D[-1]
    di = iti[(ti > dlo) & (ti < dhi)]
    tr = K[(K.context == "post_cue") & np.isfinite(K.side_ml) & (K.daq_peak_s > dlo) & (K.daq_peak_s < dhi)]
    if len(tr) > N_TRIAL_MAX:
        tr = tr.iloc[np.sort(np.random.default_rng(0).choice(len(tr), N_TRIAL_MAX, replace=False))]
    f16 = np.float16
    np.savez_compressed(
        f, cols=np.array(cols), epoch=epochs.epoch_of(label), animal=an, clock_med_s=clock,
        bout_peth=_sample(R, ft, to[bout], LAGS_P).astype(f16), bout_side=iti.side_ml.to_numpy(float)[bout], bout_gap=gap[bout],
        bout_pos=iti.position.to_numpy(dtype=str)[bout],
        first_peth=_sample(R, ft, fl, LAGS_P).astype(f16) if len(fl) else np.zeros((0, len(LAGS_P), len(cols)), f16),
        first_cue=_sample(R, ft, fl - rt, LAGS_C).astype(f16) if len(fl) else np.zeros((0, len(LAGS_C), len(cols)), f16),
        first_rt=rt, first_pos=np.asarray(fpos, str),
        iti_X=_sample(R, ft, di.daq_peak_s.to_numpy(float), LAGS_D).astype(f16), iti_side=di.side_ml.to_numpy(float),
        iti_pos=di.position.to_numpy(dtype=str), iti_t=di.daq_peak_s.to_numpy(float),
        tr_X=_sample(R, ft, tr.daq_peak_s.to_numpy(float), LAGS_D).astype(f16), tr_side=tr.side_ml.to_numpy(float),
        tr_pos=tr.position.to_numpy(dtype=str), tr_t=tr.daq_peak_s.to_numpy(float))
    return {"label": label, "n_bout": int(bout.sum()), "n_iti": len(di), "n_first": len(fl), "clock_med_s": clock}


# ------------------------------------------------------------------------------------------------ summaries


def _load(label: str) -> dict:
    z = np.load(OUT / "sessions" / f"{label}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def _bout(d: dict, gap: float) -> np.ndarray:
    """Bout-start ITI PETHs whose preceding no-lick gap is >= gap s."""
    return d["bout_peth"][d["bout_gap"] >= gap]


def ramp_metrics(P: np.ndarray, lags: np.ndarray) -> tuple[float, float, float]:
    """(amplitude at the lick, t50, t20) of a baseline-subtracted PETH: t_x = the first lag after the LAST time (in
    [-1.0, 0]) the PETH was below x * amplitude (sign-aware); NaN if the amplitude is not positive-going."""
    A = float(P[(lags >= -0.1) & (lags <= 0.1)].mean())
    out = [A]
    for x in (0.5, 0.2):
        w = (lags >= -1.0) & (lags <= 0.0)
        v, L = P[w] * np.sign(A), lags[w]
        below = np.flatnonzero(v < x * abs(A))
        out.append(float(L[below[-1] + 1]) if len(below) and below[-1] + 1 < len(L) else (float(L[0]) if not len(below)
                                                                                                else np.nan))
    return out[0], out[1], out[2]


def _cv_r(X: np.ndarray, y: np.ndarray, t: np.ndarray, k: int = 5) -> float:
    """Ridge (alpha by inner CV), 5 contiguous time-block folds; r(prediction, y)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    o = np.argsort(t)
    X, y = X[o], y[o]
    folds = np.array_split(np.arange(len(y)), k)
    pred = np.full(len(y), np.nan)
    for te in folds:
        trn = np.setdiff1d(np.arange(len(y)), te)
        sc = StandardScaler().fit(X[trn])
        m = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(X[trn]), y[trn])
        pred[te] = m.predict(sc.transform(X[te]))
    return float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else np.nan


def _within(side: np.ndarray, pos: np.ndarray) -> np.ndarray:
    w = side.astype(float).copy()
    for q in np.unique(pos):
        m = pos == q
        w[m] -= w[m].mean()
    return w


def decode_session(d: dict, rng) -> list[dict]:
    rows = []
    ni, nt = len(d["iti_side"]), len(d["tr_side"])
    if ni < 50:
        return rows
    sub = np.sort(rng.choice(nt, ni, replace=False)) if nt >= ni else np.arange(nt)
    sets = {"iti": (d["iti_X"].astype(float), d["iti_side"], d["iti_pos"], d["iti_t"]),
            "trial": (d["tr_X"].astype(float)[sub], d["tr_side"][sub], d["tr_pos"][sub], d["tr_t"][sub])}
    feat = [i for i, c in enumerate(d["cols"]) if c not in ("cortex",) and not c.startswith(("ALM", "tjM1"))]
    for sname, (X, side, pos, t) in sets.items():
        for target, y in (("direction", side), ("within_spout", _within(side, pos))):
            for li, lag in enumerate(LAGS_D):
                rows.append({"set": sname, "target": target, "lag": float(lag), "n": len(y),
                             "r": _cv_r(X[:, li, feat], y, t)})
    return rows


def frozen_decode(D: dict) -> pd.DataFrame:
    """FROZEN ITI direction decoder per animal: ridge on the animal's August pre ITI licks (features = area activity
    averaged over lags 0..0.2 s), pre sessions scored leave-one-session-out, post sessions with the full pre fit."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    w = (LAGS_D >= 0.0) & (LAGS_D <= 0.2)
    rows = []
    for a in AN:
        S = {x: d for x, d in D.items() if str(d["animal"]) == a and len(d["iti_side"]) >= 20}
        pre = [x for x, d in S.items() if str(d["epoch"]) == "pre"]
        if len(pre) < 3:
            continue
        feat = [i for i, c in enumerate(next(iter(S.values()))["cols"]) if c != "cortex" and not c.startswith(("ALM", "tjM1"))]

        def xy(x, tg, S=S, feat=feat):
            d = S[x]
            y = d["iti_side"] if tg == "direction" else _within(d["iti_side"], d["iti_pos"])
            return d["iti_X"].astype(float)[:, w][:, :, feat].mean(1), y
        for tg in ("direction", "within_spout"):
            for x in S:
                trn = [p_ for p_ in pre if p_ != x]
                Xt = np.vstack([xy(p_, tg)[0] for p_ in trn])
                yt = np.concatenate([xy(p_, tg)[1] for p_ in trn])
                sc = StandardScaler().fit(Xt)
                m = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(Xt), yt)
                X, y = xy(x, tg)
                pr = m.predict(sc.transform(X))
                rows.append({"label": x, "animal": a, "epoch": str(S[x]["epoch"]), "target": tg, "n": len(y),
                             "r": float(np.corrcoef(pr, y)[0, 1]) if np.std(pr) > 0 else np.nan})
    return pd.DataFrame(rows)


def summarise(labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import analysis_kit as ak
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    D = {x: _load(x) for x in labels if (OUT / "sessions" / f"{x}.npz").exists()}
    meta = pd.DataFrame([{"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "clock_med_s": float(d["clock_med_s"]),
                          "n_bout": len(d["bout_side"]), "n_iti": len(d["iti_side"]), "n_first": len(d["first_rt"])}
                         for x, d in D.items()])
    meta.to_csv(OUT / "sessions_meta.csv", index=False)
    print("== sessions per animal x epoch (n bout-start ITI licks, summed)\n"
          + meta.pivot_table(index="animal", columns="epoch", values="n_bout", aggfunc="sum").reindex(columns=EP4).to_string())
    print(f"clock check (median |3-D post-cue peak - behaviour lick peak|, s): max over sessions {meta.clock_med_s.max():.3f}")
    # (A) ramp metrics per session x ROI (bout-start ITI licks, >= 10 per session) and trial first licks
    rr = []
    for x, d in D.items():
        cols = list(d["cols"])
        for kind, P in (("iti_bout", _bout(d, GAP_MAIN)), ("iti_bout_gap1", _bout(d, BOUT_GAP)),
                        ("trial_first", d["first_peth"])):
            if len(P) < 10:
                continue
            P = P.astype(float)
            b = P[:, (LAGS_P >= BASE[0]) & (LAGS_P <= BASE[1]), :].mean(1, keepdims=True)
            m = (P - b).mean(0)
            for roi in RAMP_ROIS:
                if roi in cols:
                    A, t50, t20 = ramp_metrics(m[:, cols.index(roi)], LAGS_P)
                    rr.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "kind": kind, "roi": roi,
                               "n": len(P), "amp": A, "t50": t50, "t20": t20})
    RR = pd.DataFrame(rr)
    RR.to_csv(OUT / "ramp_sessions.csv", index=False)
    print("\n== per animal (rule 8): t50 (s from lick ONSET), bout-start ITI licks\n"
          + RR[RR.kind == "iti_bout"].pivot_table(index=["roi", "animal"], columns="epoch", values="t50")
          .reindex(columns=EP4).round(2).to_string())
    bt = []
    for (kind, roi), t in RR.groupby(["kind", "roi"]):
        for val in ("t50", "t20", "amp"):
            ref = {a: t[(t.animal == a) & (t.epoch == "pre")][val].dropna().to_numpy() for a in AN}
            for ep in EP4[1:]:
                post = {a: t[(t.animal == a) & (t.epoch == ep)][val].dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(ref[a]) and len(post[a])]
                if len(com) < 3:
                    continue
                iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                bt.append({"kind": kind, "roi": roi, "value": val, "epoch": ep, "n_animals": len(com),
                           "pre": float(np.mean([ref[a].mean() for a in com])), "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                           "n_later": sum(post[a].mean() < ref[a].mean() for a in com)})
    BT = pd.DataFrame(bt, columns=["kind", "roi", "value", "epoch", "n_animals", "pre", "delta", "lo", "hi", "n_later"])
    BT.to_csv(OUT / "ramp_change.csv", index=False)
    print("\n== ramp change vs pre (nested bootstrap; t: negative = EARLIER rise)\n"
          + BT[BT.value != "amp"].round(3).to_string(index=False))
    # ramp figure: pooled PETH per epoch (mean of animals of session means, each animal's PETHs scaled by its pre amplitude)
    fig, axs = plt.subplots(2, len(RAMP_ROIS) // 2 + 1, figsize=(19, 7.5), squeeze=False)
    colors = {"pre": "k", "acute": "#d7301f", "subacute": "#fc8d59", "chronic": "#2b8cbe"}
    for i, roi in enumerate(RAMP_ROIS):
        ax = axs.flat[i]
        for ep in EP4:
            an_curves = []
            for a in AN:
                cs = []
                for d in D.values():
                    if str(d["animal"]) != a or str(d["epoch"]) != ep or len(_bout(d, GAP_MAIN)) < 10 or roi not in list(d["cols"]):
                        continue
                    P = _bout(d, GAP_MAIN).astype(float)[:, :, list(d["cols"]).index(roi)]
                    P = P - P[:, (LAGS_P >= BASE[0]) & (LAGS_P <= BASE[1])].mean(1, keepdims=True)
                    cs.append(P.mean(0))
                if cs:
                    an_curves.append((a, np.mean(cs, 0)))
            if an_curves:
                curves = []
                for a, cv in an_curves:
                    pre_a = RR[(RR.kind == "iti_bout") & (RR.roi == roi) & (RR.animal == a) & (RR.epoch == "pre")].amp.mean()
                    curves.append(cv / pre_a if np.isfinite(pre_a) and abs(pre_a) > 0 else cv * np.nan)
                ax.plot(LAGS_P, np.nanmean(curves, 0), color=colors[ep], lw=2, label=f"{ep} ({len(curves)})")
        ax.axvline(0, color="0.6", lw=0.6)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_title(roi, fontsize=9)
        ax.set_xlim(-1.5, 0.75)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axs.flat[len(RAMP_ROIS):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7, frameon=False)
    axs.flat[0].set_ylabel("activity / animal's pre amplitude at the lick")
    fig.suptitle("NEURAL: activity before SELF-INITIATED licks (bout-start ITI 3-D licks: no lick for 2 s, no cue / spout "
                 "move within -2..+0.75 s), lick-ONSET aligned, baseline -1.5..-1.0 s; mean of animals, each scaled by its pre "
                 "amplitude. ALM / tjM1 = anterolateral 40 % of MOs / MOp; _right = contralesional.", fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "figures" / "iti_ramp_peth.png", dpi=110)
    plt.close(fig)
    # (B) decoding: within session, per lag
    rng = np.random.default_rng(0)
    dr = []
    for x, d in D.items():
        for r_ in decode_session(d, rng):
            dr.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), **r_})
    DR = pd.DataFrame(dr)
    DR.to_csv(OUT / "decode_sessions.csv", index=False)
    if DR.empty:
        print("no session with >= 50 ITI licks")
        return
    win = DR[(DR.lag >= 0.0) & (DR.lag <= 0.2)].groupby(["label", "animal", "epoch", "set", "target"]).r.mean().reset_index()
    print("\n== per animal (rule 8): decoding r of lateral direction, lags 0..0.2 s, mean over sessions\n"
          + win.pivot_table(index=["target", "set", "animal"], columns="epoch", values="r").reindex(columns=EP4).round(2)
          .to_string())
    db = []
    for (s_, tg), t in win.groupby(["set", "target"]):
        ref = {a: t[(t.animal == a) & (t.epoch == "pre")].r.dropna().to_numpy() for a in AN}
        for ep in EP4[1:]:
            post = {a: t[(t.animal == a) & (t.epoch == ep)].r.dropna().to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            if len(com) < 3:
                continue
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            db.append({"set": s_, "target": tg, "epoch": ep, "n_animals": len(com),
                       "pre": float(np.mean([ref[a].mean() for a in com])), "delta": iv.point, "lo": iv.lo, "hi": iv.hi})
    DB = pd.DataFrame(db)
    DB.to_csv(OUT / "decode_change.csv", index=False)
    print("\n== decoding change vs pre (nested bootstrap)\n" + DB.round(3).to_string(index=False))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
    for i, tg in enumerate(("direction", "within_spout")):
        for j, s_ in enumerate(("iti", "trial")):
            ax = axs[i, j]
            t = DR[(DR.target == tg) & (DR.set == s_)]
            for ep in EP4:
                c = t[t.epoch == ep].groupby(["animal", "lag"]).r.mean().groupby("lag").mean()
                if len(c):
                    ax.plot(c.index, c.values, color=colors[ep], lw=2, label=ep)
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_title(f"{'ITI (self-initiated)' if s_ == 'iti' else 'in-trial post-cue (n-matched)'}: "
                         f"{'3-D lateral direction' if tg == 'direction' else 'within-spout direction'}", fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axs[0, 0].legend(fontsize=8, frameon=False)
    axs[1, 0].set_xlabel("lag from lick peak (s)")
    axs[1, 1].set_xlabel("lag from lick peak (s)")
    axs[0, 0].set_ylabel("CV r (decoded vs actual)")
    axs[1, 0].set_ylabel("CV r (decoded vs actual)")
    fig.suptitle("NEURAL: decoding each lick's 3-D lateral direction from cortical area activity, within session (ridge, "
                 "5 time-block folds), per lag; mean of animals. Within-spout = direction minus its spout's mean (execution "
                 "variation without spout identity).", fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "figures" / "iti_direction_decoding.png", dpi=110)
    plt.close(fig)
    FZ = frozen_decode(D)
    FZ.to_csv(OUT / "decode_frozen_sessions.csv", index=False)
    if not FZ.empty:
        print("\n== per animal (rule 8): FROZEN pre-ITI decoder, r per epoch (pre = leave-one-session-out)\n"
              + FZ.pivot_table(index=["target", "animal"], columns="epoch", values="r").reindex(columns=EP4).round(2)
              .to_string())
        fb = []
        for tg, t in FZ.groupby("target"):
            ref = {a: t[(t.animal == a) & (t.epoch == "pre")].r.dropna().to_numpy() for a in AN}
            for ep in EP4[1:]:
                post = {a: t[(t.animal == a) & (t.epoch == ep)].r.dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(ref[a]) and len(post[a])]
                if len(com) >= 3:
                    iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                    fb.append({"target": tg, "epoch": ep, "n_animals": len(com), "delta": iv.point, "lo": iv.lo,
                               "hi": iv.hi})
        pd.DataFrame(fb).to_csv(OUT / "decode_frozen_change.csv", index=False)
        print("\n== frozen decoder change vs pre\n" + pd.DataFrame(fb).round(3).to_string(index=False))
    json.dump({"n_sessions": len(D)}, open(OUT / "summary.json", "w"))
    print("->", OUT)


def roi_figure(label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import area_image, outline

    from wfield_local import combined_encoding as ce
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = roi_masks(parea, pstat, names, flat, shape)
    H, W = (int(x) for x in shape)
    img = np.zeros(H * W)
    for v, n in enumerate(("ALM_left", "ALM_right", "tjM1_left", "tjM1_right"), 1):
        g = np.zeros(H * W, bool)
        g[flat[M[n]]] = True
        img[g] = v
    ai, _nm = area_image(label)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(np.where(img.reshape(H, W) > 0, img.reshape(H, W), np.nan), cmap="Set1", vmin=1, vmax=9)
    ax.imshow(outline(ai), cmap="Greys", vmin=0, vmax=1.2, alpha=0.6)
    ax.set_title(f"ROIs: ALM (MOs) / tjM1 (MOp) anterolateral {ROI_FRAC:.0%}\nimage left = ipsilesional", fontsize=9)
    ax.axis("off")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "iti_licks_rois.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--rois-only", action="store_true")
    a = ap.parse_args(argv)
    assert_writable(OUT)
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    run = E / "frozen_encoder_gate-attempted_v7g3_t3d_frA" / "joint" / "sessions"
    labels = sorted(a.labels or (p.stem for p in run.glob("*.json")))
    labels = [x for x in labels if x[-4:] >= "0801"]
    if a.rois_only:
        roi_figure(labels[0])
        return 0
    if not a.summarise:
        roi_figure(labels[0])
        todo = [(x, str(OUT)) for x in labels if not (OUT / "sessions" / f"{x}.npz").exists()]
        print(f"[iti_licks] {len(labels)} sessions, {len(todo)} to extract -> {OUT}", flush=True)
        got, fail = ak.fan_sessions(todo, extract, jobs=a.jobs, label="extract")
        for it, r in got:
            if not r.get("cached"):
                print(f"  {it[0]}: bout {r['n_bout']}, iti {r['n_iti']}, first {r['n_first']}, clock {r['clock_med_s']:.3f} s",
                      flush=True)
        for it, err in fail:
            print(f"  !! {it[0]}: {err}", flush=True)
    summarise(labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
