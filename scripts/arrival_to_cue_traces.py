"""ARRIVAL -> CUE REGIONAL TRACES (Priya 10-29). DESCRIPTIVE, PROVISIONAL, n = 4 animals.

The pre-cue period runs from the spout-arrival strobe to the cue, and its length VARIES: in the enforced-no-lick (ENL)
window a lick that CONTACTS the spout restarts the 2-3 s pre-cue timer (non-contact licks do not), so the wait grows with
the ENL-violation (ENLv) licks. One fixed alignment cannot show the whole period, so three anchors are used.

TRIALS: ENGAGED trials (enl_trials.csv.gz `engaged` = the behaviour-table engagement gate, the precue_restw trial set;
the gate of the CURRENT behaviour table; the precue_restw cache can hold an older count, only its RESTW level is used) with a finite strobe and cue, in sessions of il.labels() (pre = August only)
whose precue_restw cache has restw_ok. ENLv licks = rows of enl_licks/enl_licks.csv.gz (cam4 licks with strobe <= onset
< cue; `contact` = DAQ contact matched to the lick). wait = cue - strobe.

TRACE: per area (ROI v8; ALM / tjM1 split from MOs / MOp, SSp-m, SSp-n, MOs_rest / MOp_rest references), the
construction of `lick_precue_restw_traces.session_worker` (area means of U_atlas on the adopted SVT, slow drift basis
removed, x1e3, no per-trial baseline) MINUS the session's RESTW level (<OUT_W>/precue_restw/<label>.npz). Sampled on a
0.05 s grid in three segments, each MASKED to the samples that belong to it:
  arr   -0.5 .. +1.0 s from arrival (strobe); masked at / after the first ENLv lick, or the cue if there is none
  last  -0.3 .. +1.0 s from the LAST ENLv lick (any, contact or not); masked before the previous ENLv lick (before
        arrival if it is the only one) and at / after the cue
  cue   -1.5 .. +0.5 s from the cue; masked before the last ENLv lick (before arrival for trials with no ENLv lick)
The last and cue segments can share samples when cue - last lick < 2.5 s (the same data shown on two anchors).
Trials with NO ENLv lick have no last-lick segment and are drawn separately.
Heatmaps: the same trace from -0.5 s to min(wait + 0.5, 12) s after arrival, trials sorted by wait.

AGGREGATION (figure A): per animal x epoch, mean over that animal's trials (pooled over sessions) per sample, kept where
>= MIN_N trials contribute (5 pooled over spouts, 3 per spout); then mean over animals, shade = SEM over animals; solid
where all 4 animals contribute, dotted where fewer.

Outputs under <il.OUT>/arrival_to_cue/: sessions/<label>.npz, counts_per_animal.csv, figdata/*.pkl, figures/*.png.
Usage: python scripts/arrival_to_cue_traces.py [--jobs 2] [--plot-only] [--only L1,L2] [--rest-variant restnl]
RESTnl VARIANT (`--rest-variant restnl`, default restw unchanged; no re-extraction, the trial set is already the current
gate): every stored trace (trace - RESTW) is shifted by + RESTW - RESTnl per session x area (exact), RESTnl = the LICK-FREE
rest level of scripts/restnl.py; sessions without a RESTnl level are left out; every output name gets the suffix _restnl.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as _fd  # noqa: E402
import iti_licks as il  # noqa: E402
import restnl  # noqa: E402
from figdata import plot_only  # noqa: E402

SFX = ""                        # "_restnl" under --rest-variant restnl (set in main)


def save_fd(stem, obj, root=None):
    return _fd.save_fd(stem + SFX, obj, root)


def load_fd(stem, root=None):
    return _fd.load_fd(stem + SFX, root)


def _o(name: str) -> str:
    stem, dot, ext = name.partition(".")
    return f"{stem}{SFX}{dot}{ext}"


def _rl(txt: str) -> str:
    return restnl.relabel(txt) if SFX else txt

OUTD = il.OUT / "arrival_to_cue"
SESS = OUTD / "sessions"
FIGD = OUTD / "figdata"
FIGS = OUTD / "figures"
VERSION = 2
DT = 0.05


def _grid(a, b):
    return np.round(np.arange(a, b + 1e-6, DT), 3)


SEGS = {"arr": _grid(-0.5, 1.0), "last": _grid(-0.3, 1.0), "cue": _grid(-1.5, 0.5)}
SEG_TICKS = {"arr": [-0.5, 0, 0.5], "last": [0, 0.5], "cue": [-1.0, -0.5, 0, 0.5]}
SEG_TXT = {"arr": "from arrival (s)", "last": "from last ENLv lick (s)", "cue": "from cue (s)"}
HM_MAX = 12.0
LAGS_H = _grid(-0.5, HM_MAX)
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "MOs_rest", "MOp_rest"]
REF = {"MOs_rest", "MOp_rest"}
HEMI = (("right", "contra (R)"), ("left", "ipsi (L)"))
COLS = [f"{a}_{h}" for a in AREAS for h, _ in HEMI]
HM_COLS = ["SSp-n_right", "tjM1_right", "SSp-n_left"]
POS6 = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
MIN_N = 5
MIN_N_POS = 3
N_AN = 4
HM_ROWS = 1200
BASE_TXT = "minus the session RESTW level (precue_restw cache)"
TRIAL_TXT = "ENGAGED trials (standard engagement gate), restw_ok sessions, pre = August only"
BASE_NL = f"minus the session's lick-free rest level ({restnl.DEF})"
TRIAL_NL = "ENGAGED trials (current engagement gate), sessions with a RESTnl level, pre = August only"


# ------------------------------------------------------------------------------------------------ per session
def session_worker(item) -> dict:
    """One session (module-level for spawn) -> SESS/<label>.npz."""
    label, T, L = item
    f = SESS / f"{label}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if int(z["version"]) == VERSION:
            return {"label": label, "cached": True}
    src = np.load(il.OUT_W / "precue_restw" / f"{label}.npz")
    restw_ok, animal, epoch = bool(src["restw_ok"]), str(src["animal"]), str(src["epoch"])
    T = T.sort_values("cue_s").reset_index(drop=True)
    E = T[T.engaged.astype(str).str.lower().eq("true") & np.isfinite(T.cue_s.to_numpy(float))].reset_index(drop=True)
    # the precue_restw cache can hold an older engaged count (gate updated since); only its session RESTW level is used
    n_eng_src, n_eng = len(src["pos"]), len(E)
    E = E[np.isfinite(E.strobe_s.to_numpy(float))].reset_index(drop=True)
    n = len(E)
    st, cue = E.strobe_s.to_numpy(float), E.cue_s.to_numpy(float)
    # ENLv licks of these trials (strobe <= onset < cue)
    L = L[L.trial_id.isin(E.trial_id)]
    L = L[(L.t_from_strobe >= 0) & (L.t_to_cue > 0)].sort_values("daq_on_s")
    row = pd.Series(np.arange(n), index=E.trial_id.to_numpy())
    lk_tr = row.loc[L.trial_id.to_numpy()].to_numpy(int)
    lk_t = L.daq_on_s.to_numpy(float) - st[lk_tr]
    lk_c = L.contact.astype(str).str.lower().eq("true").to_numpy()
    first, last, prev = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    n_enl, n_con = np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        t = lk_t[lk_tr == i]
        if t.size:
            first[i], last[i] = t[0], t[-1]
            prev[i] = t[-2] if t.size > 1 else np.nan
            n_enl[i], n_con[i] = t.size, int(lk_c[lk_tr == i].sum())
    wait = cue - st
    meta = dict(version=VERSION, restw_ok=restw_ok, animal=animal, epoch=epoch, cols=np.array(COLS),
                n_eng=n_eng, n_eng_src=n_eng_src,
                t_trial_id=E.trial_id.to_numpy(int), t_pos=np.asarray(E.pos.astype(str).to_numpy(), str), t_wait=wait,
                t_n_enl=n_enl, t_n_contact=n_con, t_first=first, t_last=last, t_prev=prev,
                lk_tr=lk_tr, lk_t=lk_t, lk_contact=lk_c)
    SESS.mkdir(parents=True, exist_ok=True)
    if not restw_ok:
        np.savez_compressed(f, **meta)
        return {"label": label, "cached": False, "n": n, "restw_ok": False}
    from wfield_local import combined_encoding as ce
    # imaging trace (== lick_precue_restw_traces.session_worker)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    Mk = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(Mk)
    Ra = np.stack([Ub[Mk[c]].mean(0) for c in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")
    sc = [str(c) for c in src["cols"]]
    if cs != sc:
        raise ValueError(f"{label}: columns differ from the precue_restw cache")
    R = Rs[:, [cs.index(c) for c in COLS]] - np.asarray(src["restw"], float)[[sc.index(c) for c in COLS]][None, :]

    def samp(t, lags, Rm=R):
        ok = np.isfinite(t)
        X = il._sample(Rm, ft, np.where(ok, t, ft[0]), lags).astype(np.float32)
        tt = t[:, None] + lags[None, :]
        X[(tt < ft[0]) | (tt > ft[-1]) | ~np.isfinite(tt)] = np.nan
        X[~ok] = np.nan
        return X

    has = n_enl > 0
    la, ll, lc = SEGS["arr"], SEGS["last"], SEGS["cue"]
    Xa = samp(st, la)
    Xa[la[None, :] >= (np.where(has, first, wait))[:, None]] = np.nan
    lt = st + last                                   # NaN for trials without ENLv licks
    Xl = samp(lt, ll)
    lo = np.where(np.isfinite(prev), prev, 0.0) - last
    Xl[ll[None, :] < lo[:, None]] = np.nan
    Xl[ll[None, :] >= (wait - last)[:, None]] = np.nan
    Xc = samp(cue, lc)
    Xc[lc[None, :] < (np.where(has, last, 0.0) - wait)[:, None]] = np.nan
    Xh = samp(st, LAGS_H, R[:, [COLS.index(c) for c in HM_COLS]])
    Xh[LAGS_H[None, :] > (wait + 0.5)[:, None]] = np.nan
    np.savez_compressed(f, **meta, Xa=Xa.astype(np.float16), Xl=Xl.astype(np.float16), Xc=Xc.astype(np.float16),
                        Xh=Xh.astype(np.float16))
    return {"label": label, "cached": False, "n": n, "restw_ok": True}


# ------------------------------------------------------------------------------------------------ loading
def load_all(labels: list) -> dict:
    metas, X = [], {k: [] for k in ("Xa", "Xl", "Xc", "Xh")}
    licks = []
    off = 0
    for x in labels:
        z = np.load(SESS / f"{x}.npz")
        if not bool(z["restw_ok"]):
            continue
        shift = None
        if SFX:   # stored = trace - RESTW -> trace - RESTnl
            nl, ok = restnl.level(x, COLS)
            if not ok:
                continue
            src = np.load(il.OUT_W / "precue_restw" / f"{x}.npz")
            sc = [str(c) for c in src["cols"]]
            shift = np.asarray(src["restw"], float)[[sc.index(c) for c in COLS]] - nl
        D = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("t_")})
        D["label"], D["animal"], D["epoch"] = x, str(z["animal"]), str(z["epoch"])
        metas.append(D)
        licks.append(pd.DataFrame({"tr": z["lk_tr"] + off, "t": z["lk_t"], "contact": z["lk_contact"]}))
        for k in X:
            v = z[k].astype(np.float32)
            if shift is not None:
                v = v + (shift[[COLS.index(c) for c in HM_COLS]] if k == "Xh" else shift)[None, None, :].astype(np.float32)
            X[k].append(v)
        off += len(D)
    D = pd.concat(metas, ignore_index=True)
    D["enl"] = D.n_enl > 0
    out = {k: np.concatenate(v).astype(np.float32) for k, v in X.items()}
    out["D"], out["licks"] = D, pd.concat(licks, ignore_index=True)
    return out


# ------------------------------------------------------------------------------------------------ tables
def counts(A: dict) -> pd.DataFrame:
    D = A["D"]
    seg_has = {k: np.isfinite(A[f"X{k[0]}"][:, :, 0]).any(1) for k in ("arr", "last", "cue")}
    rec = []
    for sp in ("all", "far_contra"):
        for (an, ep), g in D.groupby(["animal", "epoch"]):
            g = g if sp == "all" else g[g.pos == sp]
            if not len(g):
                continue
            ix = g.index.to_numpy()
            w = g.wait.to_numpy()
            e = g.enl.to_numpy()
            rec.append({"spouts": sp, "animal": an, "epoch": ep, "sessions": g.label.nunique(), "trials": len(g),
                        "with_enlv": int(e.sum()), "no_enlv": int((~e).sum()), "frac_enlv": round(e.mean(), 2),
                        "n_seg_arr": int(seg_has["arr"][ix].sum()), "n_seg_last": int(seg_has["last"][ix].sum()),
                        "n_seg_cue": int(seg_has["cue"][ix].sum()),
                        "enlv_licks_med": float(np.median(g.n_enl[e])) if e.any() else np.nan,
                        "contact_frac": round(g.n_contact.sum() / max(g.n_enl.sum(), 1), 2),
                        "wait_med": round(np.median(w), 2), "wait_q25": round(np.quantile(w, .25), 2),
                        "wait_q75": round(np.quantile(w, .75), 2), "wait_p90": round(np.quantile(w, .9), 2),
                        "wait_max": round(w.max(), 1), "frac_wait_gt12": round((w > HM_MAX - 0.5).mean(), 3),
                        "wait_med_noenlv": round(np.median(w[~e]), 2) if (~e).any() else np.nan})
    C = pd.DataFrame(rec)
    C["epoch"] = pd.Categorical(C.epoch, il.EP4, ordered=True)
    return C.sort_values(["spouts", "animal", "epoch"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ aggregation
def agg(A: dict, sel: np.ndarray, min_n: int) -> dict:
    """{seg: {epoch: (mu, se, n_an, n_trials_seg, n_animals_seg)}} with mu etc. (lags, cols)."""
    D = A["D"]
    out = {}
    for seg in ("arr", "last", "cue"):
        X = A[f"X{seg[0]}"]
        res = {}
        for ep in il.EP4:
            per, ntr = [], 0
            for an in sorted(D.animal.unique()):
                m = sel & (D.animal == an).to_numpy() & (D.epoch == ep).to_numpy()
                if not m.any():
                    continue
                Y = X[m]
                k = np.isfinite(Y).sum(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mu = np.nanmean(Y, 0)
                mu[k < min_n] = np.nan
                if np.isfinite(mu).any():
                    per.append(mu)
                    ntr += int(np.isfinite(Y[:, :, 0]).any(1).sum())
            if not per:
                continue
            P = np.stack(per)
            na = np.isfinite(P).sum(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu = np.nanmean(P, 0)
                sd = np.nanstd(P, 0, ddof=1) if len(per) > 1 else np.full(mu.shape, np.nan)
            res[ep] = (mu, sd / np.sqrt(np.maximum(na, 1)), na, ntr, len(per))
        out[seg] = res
    return out


def heat_data(A: dict, an: str | None, sp: str | None) -> dict:
    D, lk = A["D"], A["licks"]
    out = {}
    for ep in il.EP4:
        m = (D.epoch == ep).to_numpy()
        if an:
            m &= (D.animal == an).to_numpy()
        if sp:
            m &= (D.pos == sp).to_numpy()
        ix = np.flatnonzero(m)
        if not ix.size:
            continue
        ix = ix[np.argsort(D.wait.to_numpy()[ix], kind="stable")]
        n_all = ix.size
        if ix.size > HM_ROWS:                           # evenly spaced in wait order (keeps the distribution)
            ix = ix[np.unique(np.round(np.linspace(0, ix.size - 1, HM_ROWS)).astype(int))]
        rowof = pd.Series(np.arange(ix.size), index=ix)
        L = lk[lk.tr.isin(ix)]
        out[ep] = {"H": A["Xh"][ix].astype(np.float16), "wait": D.wait.to_numpy()[ix], "n_all": n_all,
                   "n_an": D.animal.iloc[ix].nunique(), "lk_row": rowof.loc[L.tr.to_numpy()].to_numpy(),
                   "lk_t": L.t.to_numpy(), "lk_c": L.contact.to_numpy(bool)}
    return out


def compute(labels: list) -> None:
    A = load_all(labels)
    D = A["D"]
    C = counts(A)
    OUTD.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUTD / _o("counts_per_animal.csv"), index=False)
    pd.set_option("display.width", 250)
    print("== per animal x epoch counts (rule 8), ENGAGED trials, restw_ok sessions\n" + C.to_string(index=False))
    allm = np.ones(len(D), bool)
    enl = D.enl.to_numpy()
    pooled = {"enl": agg(A, allm & enl, MIN_N), "none": agg(A, allm & ~enl, MIN_N)}
    save_fd("arrival_to_cue_pooled", {"pan": pooled, "n_trials": len(D), "n_sess": D.label.nunique()}, FIGD)
    draw_pooled("arrival_to_cue_pooled")
    for grp in ("enl", "none"):
        g = enl if grp == "enl" else ~enl
        per = {p: agg(A, g & (D.pos == p).to_numpy(), MIN_N_POS) for p in POS6}
        stem = f"arrival_to_cue_per_spout_{grp}"
        save_fd(stem, {"pan": per, "grp": grp}, FIGD)
        draw_spout(stem)
    for an in ("PS92", "PS93", None):
        for sp in (None, "far_contra"):
            stem = f"arrival_to_cue_heatmap_{an or 'pooled'}_{sp or 'all'}"
            save_fd(stem, {"pan": heat_data(A, an, sp), "an": an, "sp": sp}, FIGD)
            draw_heat(stem)
    # descriptive numbers for the report: mean level in windows, per animal x epoch, contra SSp-n / tjM1
    rec = []
    for seg, win in (("arr", (0.0, 0.5)), ("last", (0.0, 0.5)), ("cue", (-0.5, 0.0)), ("cue", (0.0, 0.5))):
        lags = SEGS[seg]
        w = (lags >= win[0] - 1e-9) & (lags <= win[1] + 1e-9)
        X = A[f"X{seg[0]}"]
        for grp, g in (("enl", enl), ("none", ~enl)):
            if seg == "last" and grp == "none":
                continue
            for (an, ep), gg in D[g].groupby(["animal", "epoch"]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    v = np.nanmean(np.nanmean(X[gg.index.to_numpy()][:, w], 1), 0)
                for c in ("SSp-n_right", "tjM1_right", "SSp-n_left", "ALM_right"):
                    rec.append({"seg": f"{seg}{win}", "grp": grp, "animal": an, "epoch": ep, "area": c,
                                "v": v[COLS.index(c)]})
    W = pd.DataFrame(rec).pivot_table(index=["area", "seg", "grp", "animal"], columns="epoch", values="v")
    W = W.reindex(columns=[e for e in il.EP4 if e in W.columns]).round(2)
    W.to_csv(OUTD / _o("window_levels_per_animal.csv"))
    print("== window levels (dF/F x1e3 above restw) per animal\n" + W.to_string())


# ------------------------------------------------------------------------------------------------ figures
def _break_marks(axl, axr):
    d = 0.015
    kw = dict(color="0.2", lw=0.9, clip_on=False)
    for ax, x in ((axl, 1.0), (axr, 0.0)):
        for y in (0.0, 1.0):
            ax.plot([x - d * 2, x + d * 2], [y - d * 3, y + d * 3], transform=ax.transAxes, **kw)


def _block(fig, spec, segs, ratios):
    sub = spec.subgridspec(1, len(segs), width_ratios=ratios, wspace=0.07)
    axs = [fig.add_subplot(sub[0, k]) for k in range(len(segs))]
    for k, ax in enumerate(axs):
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        if k > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
        if k < len(axs) - 1:
            ax.spines["right"].set_visible(False)
        else:
            ax.spines["right"].set_visible(False)
        if k > 0:
            _break_marks(axs[k - 1], ax)
    return axs


def _seg_curves(ax, seg, res, ci, show_n=False):
    from wfield_local.orofacial_cohort import EPOCH_LINE
    lags = SEGS[seg]
    ax.axvline(0, color="0.35", lw=0.7)
    ax.axhline(0, color="0.6", lw=0.5)
    ns = []
    for ep, (mu, se, na, ntr, nan_) in res.items():
        c = EPOCH_LINE[ep]
        m, s, a = mu[:, ci], se[:, ci], na[:, ci]
        ax.fill_between(lags, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.plot(lags, m, color=c, lw=0.9, ls=":")
        ax.plot(lags, np.where(a >= N_AN, m, np.nan), color=c, lw=1.3, label=ep)
        ns.append(str(ntr) + ("" if nan_ >= N_AN else f"({nan_}an)"))
    ax.set_xlim(lags[0], lags[-1])
    ax.set_xticks(SEG_TICKS[seg])
    if show_n:
        ax.set_title("n " + "/".join(ns), fontsize=5, color="0.25", pad=2)
    return "/".join(ns)


def _head(fig, title: str, fs: float, handles=None, labels=None, ncol: int = 4) -> float:
    """Suptitle pre-wrapped to the figure width + an optional legend row under it, both pinned to the top; returns the
    figure fraction where the panels may start. (A `wrap=True` suptitle reports a bogus bounding box that made
    bbox_inches='tight' keep a large blank band; pre-wrapping fixes that and the gap under the title.)"""
    import textwrap
    W, H = fig.get_size_inches()
    if SFX:
        title = _rl(title.replace(BASE_TXT, BASE_NL).replace(TRIAL_TXT, TRIAL_NL))
    txt = textwrap.fill(title, max(40, int(W * 72 / (0.6 * fs))))
    pad = 0.08 / H
    fig.suptitle(txt, fontsize=fs, y=1 - pad, va="top")
    y = 1 - pad - (txt.count("\n") + 1) * fs * 1.3 / 72 / H
    if handles:
        fig.legend(handles, labels, fontsize=9, frameon=False, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, y))
        y -= 0.32 / H
    return y


def _ratio(segs):
    return [SEGS[s][-1] - SEGS[s][0] for s in segs]


def draw_pooled(stem: str) -> None:
    import matplotlib.pyplot as plt
    F = load_fd(stem, FIGD)
    P = F["pan"]
    fig = plt.figure(figsize=(22, 2.25 * len(AREAS) + 1.6))
    outer = fig.add_gridspec(len(AREAS), 2, wspace=0.12, hspace=0.55, bottom=0.6 / fig.get_size_inches()[1],
                             left=0.04, right=0.99)
    seg3, seg2 = ("arr", "last", "cue"), ("arr", "cue")
    for i, a in enumerate(AREAS):
        row_axes = []
        for j, (h, hl) in enumerate(HEMI):
            ci = COLS.index(f"{a}_{h}")
            inner = outer[i, j].subgridspec(1, 2, width_ratios=[sum(_ratio(seg3)), sum(_ratio(seg2))], wspace=0.12)
            A3 = _block(fig, inner[0, 0], seg3, _ratio(seg3))
            A2 = _block(fig, inner[0, 1], seg2, _ratio(seg2))
            for ax, s in zip(A3, seg3):
                _seg_curves(ax, s, P["enl"][s], ci, show_n=True)
            for ax, s in zip(A2, seg2):
                _seg_curves(ax, s, P["none"][s], ci, show_n=True)
            A3[0].set_ylabel(f"{a} {hl}" + (" [ref]" if a in REF else "") + _rl("\ndF/F x1e3 - restw"), fontsize=7)
            A3[1].text(0.5, 1.22, f"{a} {hl}: trials WITH ENLv licks", transform=A3[1].transAxes, ha="center",
                       fontsize=7.5, weight="bold")
            A2[0].text(1.05, 1.22, "trials with NO ENLv lick", transform=A2[0].transAxes, ha="center", fontsize=7.5,
                       weight="bold")
            if i == len(AREAS) - 1:
                for ax, s in zip(A3 + A2, seg3 + seg2):
                    ax.set_xlabel(SEG_TXT[s], fontsize=7)
            row_axes += A3 + A2
        lo = min(ax.get_ylim()[0] for ax in row_axes)
        hi = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(lo, hi)
    h, lb = fig.axes[0].get_legend_handles_labels()
    top = _head(fig,
        f"NEURAL: arrival -> cue area traces, {BASE_TXT}; {TRIAL_TXT}; all spouts pooled ({F['n_trials']} trials, "
        f"{F['n_sess']} sessions). Three anchors (axis breaks //): arrival (masked from the first ENLv lick / cue), last "
        f"ENLv lick (masked before the previous ENLv lick and from the cue), cue (masked before the last ENLv lick, or "
        f"before arrival). Per animal x epoch mean over trials (>= {MIN_N} trials per sample), then mean over animals; "
        f"shade = SEM over animals; solid = 4 animals, dotted = fewer. 'n a/b/c/d' above each segment = trials "
        f"per epoch (pre, acute, subacute, chronic; '(k an)' = only k animals). GCaMP6s: decay ~1 s, so a segment carries the previous event's tail.",
        9, h, lb)
    outer.update(top=top - 0.45 / fig.get_size_inches()[1])   # room for the bold block headers above row 0
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_spout(stem: str) -> None:
    import matplotlib.pyplot as plt
    F = load_fd(stem, FIGD)
    P, grp = F["pan"], F["grp"]
    segs = ("arr", "last", "cue") if grp == "enl" else ("arr", "cue")
    areas = ["ALM", "tjM1", "SSp-m", "SSp-n"]
    fig = plt.figure(figsize=(26 if grp == "enl" else 20, 2.4 * len(areas) + 1.4))
    outer = fig.add_gridspec(len(areas), len(POS6), wspace=0.14, hspace=0.62, bottom=0.6 / fig.get_size_inches()[1],
                             left=0.04, right=0.99)
    for i, a in enumerate(areas):
        ci = COLS.index(f"{a}_right")
        row_axes = []
        for j, p in enumerate(POS6):
            axs = _block(fig, outer[i, j], segs, _ratio(segs))
            ns = [_seg_curves(ax, s, P[p][s], ci) for ax, s in zip(axs, segs)]
            # trial counts once per block (the per-segment labels collided on the narrow segments); per segment
            # only where the segments' counts differ
            ntxt = "n " + ns[0]
            if len(set(ns)) > 1:
                try:
                    dv = max(abs(int(x) - int(y)) for o in ns[1:] for x, y in zip(ns[0].split("/"), o.split("/")))
                    ntxt += f" (other segments within {dv} trials)"
                except ValueError:   # '(k an)' suffixes: fall back to listing every segment
                    ntxt = "  ".join(f"{s} n {n}" for s, n in zip(segs, ns))
            axs[len(axs) // 2].text(0.5 if len(axs) == 3 else 1.0, 1.2, f"{p.replace('_', '-')}: {a} contra (R)",
                                    transform=axs[len(axs) // 2].transAxes, ha="center", va="bottom", fontsize=7.5,
                                    weight="bold")
            axs[len(axs) // 2].text(0.5 if len(axs) == 3 else 1.0, 1.07, ntxt, transform=axs[len(axs) // 2].transAxes,
                                    ha="center", va="bottom", fontsize=6, color="0.25")
            if j == 0:
                axs[0].set_ylabel(_rl("dF/F x1e3 - restw"), fontsize=7)
            if i == len(areas) - 1:
                for ax, s in zip(axs, segs):
                    ax.set_xlabel(SEG_TXT[s], fontsize=6.5)
            row_axes += axs
        lo = min(ax.get_ylim()[0] for ax in row_axes)
        hi = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(lo, hi)
    h, lb = fig.axes[0].get_legend_handles_labels()
    what = "trials WITH ENLv licks" if grp == "enl" else "trials with NO ENLv lick (no last-lick segment)"
    top = _head(fig,
        f"NEURAL: arrival -> cue traces PER TARGET SPOUT, contralesional (right) orofacial areas, {what}; {BASE_TXT}; "
        f"{TRIAL_TXT}. Segments and masks as the pooled figure; per animal x epoch mean over trials (>= {MIN_N_POS} "
        f"trials per sample), mean over animals, shade = SEM over animals; solid = 4 animals, dotted = fewer. "
        f"'n a/b/c/d' = trials per epoch (pre, acute, subacute, chronic; '(k an)' = only k animals).", 9, h, lb)
    outer.update(top=top - 0.4 / fig.get_size_inches()[1])
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_heat(stem: str) -> None:
    import matplotlib.pyplot as plt
    F = load_fd(stem, FIGD)
    P = F["pan"]
    eps = [e for e in il.EP4 if e in P]
    fig, axs = plt.subplots(len(HM_COLS), len(eps), figsize=(4.2 * len(eps) + 1.0, 3.6 * len(HM_COLS) + 1.2),
                            squeeze=False)
    fig.subplots_adjust(hspace=0.38, bottom=0.75 / fig.get_size_inches()[1])   # 2-line titles clear the row above
    # title + top margin BEFORE the colorbars: a later subplots_adjust moves the panels under the colorbars
    who = F["an"] or "all 4 animals pooled (trials concatenated)"
    sp = "far-contra target only" if F["sp"] else "all spouts"
    top = _head(fig, f"NEURAL: single-trial traces from arrival to cue, {who}, {sp}; {BASE_TXT}; {TRIAL_TXT}. Rows sorted by "
                 f"wait (cue - arrival); dots = ENLv licks (black = contact, restarts the timer; orange = no contact); "
                 f"green tick = cue; grey = after cue + 0.5 s or outside the recording. Colour scale per area row "
                 f"(98th pct of |value|). Panels with > {HM_ROWS} trials show {HM_ROWS} rows evenly spaced in wait order.", 9)
    fig.subplots_adjust(top=top - 0.45 / fig.get_size_inches()[1])
    for i, c in enumerate(HM_COLS):
        allv = np.concatenate([P[e]["H"][:, :, i].astype(np.float32).ravel() for e in eps])
        vm = float(np.nanpercentile(np.abs(allv), 98)) if np.isfinite(allv).any() else 1.0
        for j, ep in enumerate(eps):
            ax = axs[i, j]
            d = P[ep]
            H = d["H"][:, :, i].astype(np.float32)
            nr = H.shape[0]
            im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest",
                           extent=(LAGS_H[0] - DT / 2, LAGS_H[-1] + DT / 2, nr - 0.5, -0.5))
            ax.set_facecolor("0.85")
            ax.axvline(0, color="k", lw=0.6)
            ms = 1.2 if nr > 300 else 2.5
            con = d["lk_c"]
            ax.scatter(d["lk_t"][~con], d["lk_row"][~con], s=ms, c="#ff7f00", marker="o", lw=0, label="ENLv no contact")
            ax.scatter(d["lk_t"][con], d["lk_row"][con], s=ms, c="k", marker="o", lw=0, label="ENLv contact")
            w = d["wait"]
            k = w <= HM_MAX
            ax.scatter(w[k], np.flatnonzero(k), s=ms * 6, c="#00c000", marker="|", lw=1.2, label="cue")
            ax.set_xlim(LAGS_H[0], LAGS_H[-1])
            ax.set_ylim(nr - 0.5, -0.5)
            sub = f" ({nr} shown)" if d["n_all"] > nr else ""
            ax.set_title(f"{c.replace('_right', ' contra (R)').replace('_left', ' ipsi (L)')}, {ep}: "
                         f"{d['n_all']} trials{sub}, {d['n_an']} an.\n>{HM_MAX - 0.5:g} s wait: "
                         f"{(w > HM_MAX - 0.5).mean():.0%} (cue off-axis)", fontsize=7)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel("trials sorted by wait (arrival -> cue)", fontsize=7)
            if i == len(HM_COLS) - 1:
                ax.set_xlabel("time from spout arrival (s)", fontsize=7)
        fig.colorbar(im, ax=list(axs[i]), fraction=0.012, pad=0.01).set_label(_rl("dF/F x1e3 - restw"), fontsize=7)
    h, lb = axs[0, 0].get_legend_handles_labels()   # one figure legend under the panels (was over PS92 pre data)
    fig.legend(h, lb, fontsize=8, markerscale=4, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def stems() -> list:
    out = [("pooled", "arrival_to_cue_pooled"), ("spout", "arrival_to_cue_per_spout_enl"),
           ("spout", "arrival_to_cue_per_spout_none")]
    out += [("heat", f"arrival_to_cue_heatmap_{a}_{s}") for a in ("PS92", "PS93", "pooled") for s in ("all", "far_contra")]
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    global SFX
    from wfield_local import analysis_kit as ak
    SFX = restnl.SFX if restnl.variant() == "restnl" else ""
    if SFX and not plot_only():   # no re-extraction: shift the stored restw-referenced traces
        compute(il.labels())
        return 0
    if plot_only():
        for kind, stem in stems():
            {"pooled": draw_pooled, "spout": draw_spout, "heat": draw_heat}[kind](stem)
        return 0
    jobs = min(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 2, 2)
    labs = il.labels()
    if "--only" in sys.argv:
        labs = [x for x in labs if x in sys.argv[sys.argv.index("--only") + 1].split(",")]
    T = pd.read_csv(il.OUT / "enl_licks" / "enl_trials.csv.gz", usecols=["label", "trial_id", "pos", "strobe_s", "cue_s",
                                                                        "engaged"])
    L = pd.read_csv(il.OUT / "enl_licks" / "enl_licks.csv.gz", usecols=["label", "trial_id", "daq_on_s", "t_from_strobe",
                                                                        "t_to_cue", "contact"])
    items = [(x, T[T.label == x].reset_index(drop=True), L[L.label == x].reset_index(drop=True)) for x in labs]
    _res, fails = ak.fan_sessions(items, session_worker, jobs=jobs, key=lambda it: it[0], label="arrival->cue")
    if fails:
        for it, err in fails:
            print(f"  !! {it[0]}: {err}", flush=True)
        raise SystemExit(f"{len(fails)} sessions failed")
    if "--extract-only" in sys.argv:
        return 0
    compute(labs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
