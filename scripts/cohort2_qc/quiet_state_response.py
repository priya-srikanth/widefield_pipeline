"""T21: is the acute ~2x larger lick-evoked cortical response a consequence of a SYNCHRONISED cortical state? (cohort1, 11-01)

WHY. Acutely the lick-evoked orofacial response is ~2x larger and quiet orofacial cortex is more synchronised (more 1-4 Hz
power, T20). In barrel cortex, passive sensory responses are larger in the synchronised quiet-wakefulness state
(Crochet & Petersen 2006). If the same holds here, licks that start from a high-slow-power state should evoke larger
responses WITHIN pre-stroke sessions, and the acute increase in pre-lick slow power should predict part of the acute
response increase.

EVENTS: imported from quiet_lick_lfo.py (T16): cued first lick of each trial and bout-initiating ITI licks (no lick in the
preceding 2 s, outside trial response windows, no cue / strobe within [-2, +0.75] s), ENGAGED part of the session.
Excluded here: another DAQ lick onset in [-1.0, 0) s (a superset of the requested -1.0 .. -0.1 window, so the
-0.5 .. -0.1 baseline is lick-free too), any running bout overlapping [-1.0, -0.1] s, any swapped-pair frame (+/-0.5 s,
quiet_spikes_v2.busy_frames) in [-1.0, +0.5] s.
TRACES: `quiet_area_spikes.session_traces(detrend=False)` (dF/F %), columns orofacial (bilateral), orofacial_left (IPSI,
lesion left), orofacial_right (CONTRA), cortex.
(a) PRE-LICK SLOW POWER: swapped frames interpolated (quiet_lick_lfo.interp_bad), zero-phase 1-4 Hz band-pass on the whole
    session (quiet_lick_lfo.bandpass: Butterworth 3, sosfiltfilt), Hilbert envelope^2, averaged over -1.0 .. -0.1 s.
    LEAK CONTROL: the zero-phase filter + Hilbert transform are acausal, so the post-lick response leaks backwards into
    the pre-lick window and could create a power-response correlation by itself; `pw_causal` = the same band, forward-only
    (sosfilt) band-passed trace squared, averaged over the same window (no future samples enter).
(b) RESPONSE: mean dF/F 0 .. 0.5 s minus mean -0.5 .. -0.1 s (raw trace, not band-passed).
WITHIN SESSION (>= MIN_EV licks per kind): OLS response ~ z(power) (z within session x kind) -> slope in % dF/F per SD of
pre-lick power; variants: causal power, log10 power. Per animal x epoch first (rule 8), then `mixed_models.lmm_table`
epoch + time models on the per-session slopes (pre_level = mean pre slope, its z-test = is the pre slope non-zero).
PREDICTION (per animal, kind, column): pre slope in RAW power units = within-session pooled slope over the animal's pre
licks (response and power demeaned per session); predicted increase = that slope x (mean post-epoch pre-lick power -
mean pre pre-lick power, session means averaged); observed = the same difference of mean responses. Linear power (as
requested) and log10 power (the linear extrapolation runs far outside the pre range when power triples).
MATCHED-POWER CHECK: response by pre-lick power quintile (animal's pooled pre+post bins), per epoch.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_state_response.py [--jobs 4] [--summarise] [--only PS92_0811 ...]
"""
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EP, EVENTS, columns, session_traces  # noqa: E402
from quiet_lick_lfo import BAND, KIND_LAB, KINDS, bandpass, interp_bad, lick_events  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames  # noqa: E402
from wfield_local import config  # noqa: E402

SR = V2 / "state_response"                                  # follows QS_LICK_SOURCE
COLS = ["orofacial", "orofacial_left", "orofacial_right", "cortex"]
CLAB = {"orofacial": "orofacial (bilateral)", "orofacial_left": "orofacial IPSI", "orofacial_right": "orofacial CONTRA",
        "cortex": "whole cortex"}
PRE_WIN, BASE_WIN, RESP_WIN = (-1.0, -0.1), (-0.5, -0.1), (0.0, 0.5)
LICK_EXCL = (-1.0, 0.0)
MIN_EV = 15
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
SLOPES = {"slope_z": "pw", "slope_causal": "pwc", "slope_log": "lpw"}
KIND_SHORT = {"cued": "cued", "iti": "ITI"}
SLAB = {"slope_z": "Hilbert power (as specified)", "slope_causal": "causal power (leak control)",
        "slope_log": "log10 Hilbert power"}


def extract(label):
    """One session -> state_response/sessions/<label>.npz of per-lick pre-lick power and response (module-level)."""
    from scipy.signal import butter, hilbert, sosfilt
    f = SR / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    Xr, ft, fs, rest, engaged, quit_s = session_traces(label, detrend=False)
    allc = columns()
    X = Xr[[allc.index(c) for c in COLS]]
    busy, badw = busy_frames(label, ft, fs)
    Xi = interp_bad(X, badw)
    BP = bandpass(Xi, fs)
    P = np.abs(hilbert(BP, axis=-1)) ** 2
    sos = butter(3, BAND, btype="band", fs=fs, output="sos")
    Pc = sosfilt(sos, Xi - Xi[:, :1], axis=-1) ** 2                 # forward only; start at 0 to limit the onset transient
    until = quit_s if np.isfinite(quit_s) else np.inf
    cued, iti, _ = lick_events(label, ft, until)
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fsd = float(z["fs"])
    lk = np.sort(z["lick_onsets"] / fsd)
    run = np.c_[z["running_starts"], z["running_stops"]] / fsd
    bad_cs = np.r_[0, np.cumsum(badw)]
    ss = lambda t: np.searchsorted(ft, t)                             # noqa: E731
    out = {}
    for kind, tt in (("cued", cued), ("iti", iti)):
        rec, reason = [], dict(edge=0, lick=0, run=0, swap=0)
        for t in tt:
            a, b = ss(t + PRE_WIN[0]), ss(t + PRE_WIN[1])
            b0, b1, r0, r1 = ss(t + BASE_WIN[0]), ss(t + BASE_WIN[1]), ss(t + RESP_WIN[0]), ss(t + RESP_WIN[1])
            if a < 2 * int(fs) or r1 >= len(ft) - 1 or b - a < 5:
                reason["edge"] += 1
                continue
            nl = np.searchsorted(lk, t + LICK_EXCL[1] - 1e-4) - np.searchsorted(lk, t + LICK_EXCL[0])
            if nl > 0:
                reason["lick"] += 1
                continue
            if len(run) and np.any((run[:, 0] < t + PRE_WIN[1]) & (run[:, 1] > t + PRE_WIN[0])):
                reason["run"] += 1
                continue
            if bad_cs[r1] - bad_cs[a] > 0:
                reason["swap"] += 1
                continue
            base = X[:, b0:b1].mean(1)
            rec.append(np.r_[t, P[:, a:b].mean(1), Pc[:, a:b].mean(1), X[:, r0:r1].mean(1) - base, base])
        k = len(COLS)
        R = np.asarray(rec, float).reshape(-1, 1 + 4 * k)
        out[f"{kind}_t"] = R[:, 0]
        out[f"{kind}_pw"], out[f"{kind}_pwc"] = R[:, 1:1 + k], R[:, 1 + k:1 + 2 * k]
        out[f"{kind}_resp"], out[f"{kind}_base"] = R[:, 1 + 2 * k:1 + 3 * k], R[:, 1 + 3 * k:]
        out[f"{kind}_n_in"] = len(tt)
        out[f"{kind}_excl"] = np.array([reason[x] for x in ("edge", "lick", "run", "swap")])
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, cols=np.array(COLS), fs=fs, quit_s=quit_s, **out)
    return label


# ------------------------------------------------------------------------------------------------------- summary
def lick_table():
    """Long per-lick table (August pre + post sessions only)."""
    import pandas as pd
    from wfield_local import epochs
    rows = []
    for f in sorted((SR / "sessions").glob("*.npz")):
        lab = f.stem
        ep = epochs.epoch_of(lab)
        if ep is None or (ep == "pre" and config.early_pre(lab)):
            continue
        d = np.load(f)
        for kind in KINDS:
            n = len(d[f"{kind}_t"])
            if n == 0:
                continue
            for ci, c in enumerate(COLS):
                rows.append(pd.DataFrame(dict(label=lab, animal=lab[:4], epoch=ep, day=epochs.days_since_stroke(lab),
                                              kind=kind, column=c, t=d[f"{kind}_t"], pw=d[f"{kind}_pw"][:, ci],
                                              pwc=d[f"{kind}_pwc"][:, ci], resp=d[f"{kind}_resp"][:, ci],
                                              base=d[f"{kind}_base"][:, ci])))
    L = pd.concat(rows, ignore_index=True)
    L["lpw"] = np.log10(L.pw)
    return L


def _slope(x, y):
    """OLS slope of y on z(x) (and on raw x), r."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < MIN_EV or np.std(x) == 0:
        return np.nan, np.nan, np.nan
    zx = (x - x.mean()) / x.std()
    b = np.polyfit(zx, y, 1)[0]
    return float(b), float(b / x.std()), float(np.corrcoef(x, y)[0, 1])


def session_table(L):
    import pandas as pd
    rows = []
    for (lab, kind, c), g in L.groupby(["label", "kind", "column"], sort=True):
        r = dict(label=lab, animal=lab[:4], epoch=g.epoch.iloc[0], day=g.day.iloc[0], kind=kind, column=c, n=len(g),
                 pw_mean=g.pw.mean(), pwc_mean=g.pwc.mean(), lpw_mean=g.lpw.mean(), resp_mean=g.resp.mean(),
                 pw_sd=g.pw.std(ddof=0))
        for s, v in SLOPES.items():
            r[s], r[s + "_raw"], r[s.replace("slope", "r")] = _slope(g[v].to_numpy(float), g.resp.to_numpy(float))
        r["slope_z_basecov"] = np.nan                       # slope with the pre-lick baseline level as a covariate
        if len(g) >= MIN_EV:
            zx = (g.pw - g.pw.mean()) / g.pw.std(ddof=0)
            Xd = np.column_stack([np.ones(len(g)), zx, g.base - g.base.mean()])
            r["slope_z_basecov"] = float(np.linalg.lstsq(Xd, g.resp.to_numpy(float), rcond=None)[0][1])
        rows.append(r)
    S = pd.DataFrame(rows)
    return S.sort_values(["column", "kind", "label"]).reset_index(drop=True)


def per_animal(S):
    """Rule 8: per animal x epoch mean of the per-session slopes (+ n sessions), and mean response / power."""
    import pandas as pd
    out = []
    G = S[S.n >= MIN_EV]
    for m in ["slope_z", "slope_causal", "slope_log", "slope_z_basecov", "r_z", "resp_mean", "pw_mean", "n"]:
        t = G.groupby(["column", "kind", "animal", "epoch"])[m].mean().unstack().reindex(columns=EP)
        t.insert(0, "measure", m)
        out.append(t.reset_index())
    ns = G.groupby(["column", "kind", "animal", "epoch"]).size().unstack().reindex(columns=EP)
    ns.insert(0, "measure", "n_sessions")
    out.append(ns.reset_index())
    return pd.concat(out, ignore_index=True)


def _pooled_within(g, v):
    """Within-session pooled slope (raw units) of resp on power variable v, + its OLS SE."""
    x = g[v] - g.groupby("label")[v].transform("mean")
    y = g.resp - g.groupby("label").resp.transform("mean")
    x, y = x.to_numpy(float), y.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    b = float((x * y).sum() / (x * x).sum())
    dof = len(x) - g.label.nunique() - 1
    se = float(np.sqrt(((y - b * x) ** 2).sum() / max(dof, 1) / (x * x).sum()))
    return b, se


def prediction(L, S):
    """Per animal x kind x column x post epoch: predicted (pre slope x change in mean pre-lick power) vs observed."""
    import pandas as pd
    rows = []
    Sv = S[S.n >= MIN_EV]
    Lv = L[L.label.isin(set(Sv.label))]
    for (c, kind, an), g in Lv.groupby(["column", "kind", "animal"]):
        pre = g[g.epoch == "pre"]
        sm = Sv[(Sv.column == c) & (Sv.kind == kind) & (Sv.animal == an)].groupby("epoch")[
            ["pw_mean", "lpw_mean", "pwc_mean", "resp_mean"]].mean()
        if pre.label.nunique() < 1 or "pre" not in sm.index:
            continue
        for v, tag in (("pw", "linear"), ("lpw", "log10"), ("pwc", "causal")):
            b, se = _pooled_within(pre, v)
            for ep in EP[1:]:
                if ep not in sm.index:
                    continue
                post = g[g.epoch == ep]
                b_post, _ = _pooled_within(post, v)
                dpw = sm.loc[ep, f"{v}_mean"] - sm.loc["pre", f"{v}_mean"]
                obs = sm.loc[ep, "resp_mean"] - sm.loc["pre", "resp_mean"]
                rows.append(dict(column=c, kind=kind, animal=an, epoch=ep, power=tag, pre_slope_raw=b,
                                 pre_slope_se=se, post_slope_raw=b_post, pre_power=sm.loc["pre", f"{v}_mean"],
                                 d_power=dpw, power_ratio=(sm.loc[ep, f"{v}_mean"] / sm.loc["pre", f"{v}_mean"]
                                                           if tag != "log10" else 10 ** dpw),
                                 pre_resp=sm.loc["pre", "resp_mean"], observed=obs, predicted=b * dpw,
                                 predicted_lo=(b - 1.96 * se) * dpw, predicted_hi=(b + 1.96 * se) * dpw,
                                 frac=b * dpw / obs if obs != 0 else np.nan))
    return pd.DataFrame(rows)


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    G = S[S.n >= MIN_EV]
    Ls = []
    for m in ("slope_z", "slope_causal", "slope_log", "slope_z_basecov", "resp_mean", "lpw_mean"):
        Ls.append(G[["label", "kind", "column"]].assign(measure=m, y=G[m].to_numpy(float)))
    Lm = mm.add_time_covariates(pd.concat(Ls), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(Lm, ["measure", "kind", "column"], models=("epoch", "time"), print_animals=False,
                     fdr_by=("measure",))
    T.to_csv(SR / "lmm_state_response.csv", index=False)
    return T


def lick_lmm(L, S):
    """Lick-level: within each epoch, resp ~ z(pw) with random intercept + slope per SESSION, animal fixed;
    pooled across epochs, the z(pw) x epoch interaction (does the slope change post-stroke)."""
    import warnings

    import pandas as pd
    import statsmodels.formula.api as smf
    rows = []
    keep = set(S[S.n >= MIN_EV].label)
    for (c, kind), g in L[L.label.isin(keep)].groupby(["column", "kind"]):
        g = g.assign(pz=g.groupby("label").pw.transform(lambda x: (x - x.mean()) / x.std(ddof=0)),
                     epoch=pd.Categorical(g.epoch, EP)).dropna(subset=["pz", "resp"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for re_f in ("~pz", "~1"):
                try:
                    r = smf.mixedlm("resp ~ C(animal) + C(epoch) * pz", g, groups=g["label"], re_formula=re_f).fit()
                    if np.all(np.isfinite(r.bse[[t for t in r.params.index if "pz" in t]])):
                        break
                except Exception:                                     # noqa: BLE001
                    continue
        for t in r.params.index:
            if "pz" in t:
                rows.append(dict(column=c, kind=kind, term=t, coef=r.params[t], se=r.bse[t], z=r.tvalues[t],
                                 p=r.pvalues[t], re=re_f, n_licks=len(g), n_sessions=g.label.nunique()))
    T = pd.DataFrame(rows)
    T.to_csv(SR / "lmm_lick_level.csv", index=False)
    return T


# ------------------------------------------------------------------------------------------------------- figures
def _wrap(s, n=46):
    import textwrap
    return "\n".join(textwrap.wrap(s, n))


def fig_slopes(S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    meas = ["slope_z", "slope_causal"]
    fig, ax = plt.subplots(4, 4, figsize=(15, 13.5), sharey="row")
    rng = np.random.default_rng(0)
    G0 = S[S.n >= MIN_EV]
    for i, (kind, m) in enumerate([(k, m) for k in KINDS for m in meas]):
        for j, c in enumerate(COLS):
            a = ax[i, j]
            G = G0[(G0.kind == kind) & (G0.column == c)]
            for x, ep in enumerate(EP):
                B = G[G.epoch == ep]
                for an, A in B.groupby("animal"):
                    a.scatter(x + rng.uniform(-0.18, 0.18, len(A)), A[m], marker=ANIMAL_MARKER[an], facecolors="none",
                              edgecolors=EPOCH_LINE[ep], s=18, lw=0.8, alpha=0.8)
                ci = boot_ci({an: list(A[m].dropna()) for an, A in B.groupby("animal") if A[m].notna().any()}, rng)
                if ci is not None:
                    a.errorbar(x + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="o",
                               color=EPOCH_LINE[ep], mfc=EPOCH_LINE[ep], ms=6, capsize=3, lw=1.6, zorder=5)
                t = T[(T.model == "epoch") & (T.measure == m) & (T.kind == kind) & (T.column == c)]
                term = "pre_level" if x == 0 else f"C(epoch)[T.{ep}]"
                t = t[t.term == term]
                if len(t) and t.q.iloc[0] < 0.05:
                    q = t.q.iloc[0]
                    sig_marks.star_top(a, x, "***" if q < 0.001 else "**" if q < 0.01 else "*", color=EPOCH_LINE[ep])
            a.axhline(0, color="0.6", lw=0.8, ls="--")
            a.set_xticks(range(len(EP)), EP, fontsize=8)
            a.set_title(_wrap(f"{KIND_LAB[kind]}, {CLAB[c]}: {SLAB[m]}", 40), fontsize=8.5)
            if j == 0:
                a.set_ylabel("slope: % dF/F response\nper SD of pre-lick 1-4 Hz power", fontsize=8)
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    ax[0, -1].legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
                     title="animal", title_fontsize=7)
    sig_marks.reserve_headroom(fig)
    fig.suptitle(_wrap("T21: within-session slope of the lick-evoked response (mean dF/F 0..0.5 s minus -0.5..-0.1 s) on "
                       "pre-lick 1-4 Hz power (-1.0..-0.1 s, z-scored within session). Open markers = sessions (shape = "
                       "animal, colour = epoch); filled = pooled mean, nested animal -> session bootstrap 95% CI. Stars: "
                       "at PRE = mean pre slope differs from 0 (mixed model pre_level); post epochs = slope differs from "
                       "pre (epoch model); FDR q within measure. Causal power = forward-only filter, no post-lick leak.",
                       190), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.94, 0.955), h_pad=1.6, w_pad=1.2)
    save_png_svg(fig, SR / "state_response_slopes.png", dpi=110)
    plt.close(fig)
    print("wrote", SR / "state_response_slopes.png")


def fig_prediction(Pd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    fig, ax = plt.subplots(3, 4, figsize=(15, 11.5))
    for i, tag in enumerate(("linear", "log10", "causal")):
        for j, c in enumerate(COLS):
            a = ax[i, j]
            G = Pd[(Pd.power == tag) & (Pd.column == c)]
            for _, r in G.iterrows():
                fc = EPOCH_LINE[r.epoch] if r.kind == "cued" else "none"
                a.errorbar(r.observed, r.predicted, yerr=[[r.predicted - min(r.predicted_lo, r.predicted_hi)],
                                                          [max(r.predicted_lo, r.predicted_hi) - r.predicted]],
                           fmt="none", ecolor=EPOCH_LINE[r.epoch], lw=0.7, alpha=0.6)
                a.scatter(r.observed, r.predicted, marker=ANIMAL_MARKER[r.animal], facecolors=fc,
                          edgecolors=EPOCH_LINE[r.epoch], s=40, lw=1.1, zorder=4)
            v = np.r_[G.observed, G.predicted_lo, G.predicted_hi, 0.0]
            lo, hi = np.nanmin(v), np.nanmax(v)
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            a.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
            a.axhline(0, color="0.8", lw=0.6)
            a.axvline(0, color="0.8", lw=0.6)
            a.set_xlim(lo, hi)
            a.set_ylim(lo, hi)
            a.set_aspect("equal")
            a.set_xlabel("observed change in response vs pre (% dF/F)", fontsize=8)
            if j == 0:
                a.set_ylabel(_wrap(f"predicted from pre slope x change in pre-lick power ({tag} power)", 34),
                             fontsize=8)
            ac = G[G.epoch == "acute"]
            txt = "; ".join(f"{KIND_SHORT[k]} {g.predicted.sum():.2f} / {g.observed.sum():.2f} = "
                            f"{g.predicted.sum() / g.observed.sum():.2f}" for k, g in ac.groupby("kind"))
            a.set_title(_wrap(f"{CLAB[c]}, {tag} power. ACUTE, summed over animals, predicted / observed: {txt}", 38),
                        fontsize=8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    h += [Line2D([], [], marker="o", ls="", mfc=EPOCH_LINE[e], mec=EPOCH_LINE[e], label=e) for e in EP[1:]]
    h += [Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="filled = cued first lick"),
          Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="open = ITI bout start")]
    ax[0, -1].legend(handles=h, fontsize=7, loc="upper left", bbox_to_anchor=(1.04, 1), frameon=False)
    fig.suptitle(_wrap("T21: how much of each animal's post-stroke change in the lick-evoked response is predicted by its "
                       "change in pre-lick 1-4 Hz power, using the animal's PRE-stroke within-session slope (licks pooled "
                       "over pre sessions, session-demeaned)? Dashed = unity (fully predicted). Error bars = 95% CI of the "
                       "pre slope only. Per-animal ratios are unstable where the observed change is ~0 (PS95 ITI), so titles give "
                       "the ratio of sums over animals.", 190), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 0.94), h_pad=1.5)
    save_png_svg(fig, SR / "state_response_prediction.png", dpi=110)
    plt.close(fig)
    print("wrote", SR / "state_response_prediction.png")


def fig_matched(L, S):
    """Response by pre-lick power quintile (animal's pooled bins), per epoch, orofacial bilateral."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    keep = set(S[S.n >= MIN_EV].label)
    G0 = L[(L.column == "orofacial") & L.label.isin(keep)]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.8), sharey="row")
    for i, kind in enumerate(KINDS):
        for j, an in enumerate(ANIMALS):
            a = ax[i, j]
            g = G0[(G0.kind == kind) & (G0.animal == an)]
            if g.empty:
                continue
            edges = np.quantile(g.pw, np.linspace(0, 1, 6))
            for ep in EP:
                e = g[g.epoch == ep]
                if len(e) < 25:
                    continue
                b = np.clip(np.searchsorted(edges, e.pw, side="right") - 1, 0, 4)
                xm = [np.median(e.pw[b == k]) if (b == k).sum() >= 5 else np.nan for k in range(5)]
                ym = [e.resp[b == k].mean() if (b == k).sum() >= 5 else np.nan for k in range(5)]
                se = [e.resp[b == k].std() / np.sqrt((b == k).sum()) if (b == k).sum() >= 5 else np.nan
                      for k in range(5)]
                a.errorbar(xm, ym, yerr=se, color=EPOCH_LINE[ep], marker=ANIMAL_MARKER[an], mfc="none", lw=1.2,
                           ms=5, capsize=2, label=f"{ep} ({len(e)})")
            a.set_xscale("log")
            a.set_title(_wrap(f"{an}: {KIND_LAB[kind]}, orofacial (bilateral)", 40), fontsize=8.5)
            a.set_xlabel("pre-lick 1-4 Hz power (quintile median, log)", fontsize=8)
            a.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False, ncol=2)
        ax[i, 0].set_ylabel("lick-evoked response\n(% dF/F, mean +/- SEM)", fontsize=8)
    fig.suptitle(_wrap("T21 matched-power check: is the post-stroke response still larger at the SAME pre-lick 1-4 Hz power? "
                       "Quintile bins of the animal's licks pooled over epochs; each epoch's licks in each bin (bins with "
                       ">= 5 licks). Colour = epoch; legend = licks per epoch.", 190), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3)
    save_png_svg(fig, SR / "state_response_matched_power.png", dpi=110)
    plt.close(fig)
    print("wrote", SR / "state_response_matched_power.png")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    L = lick_table()
    S = session_table(L)
    S.to_csv(SR / "state_response_sessions.csv", index=False)
    print(f"{S.label.nunique()} sessions, {L[L.column == 'orofacial'].shape[0]} licks")
    A = per_animal(S)
    A.to_csv(SR / "state_response_per_animal.csv", index=False)
    print("\n== RULE 8: per animal x epoch (mean over sessions with >= %d licks)" % MIN_EV)
    for m in ("n_sessions", "n", "slope_z", "slope_causal", "slope_log", "slope_z_basecov", "resp_mean", "pw_mean"):
        print(f"\n-- {m}\n" + A[(A.measure == m) & A.column.isin(COLS)].drop(columns="measure").round(3)
              .to_string(index=False))
    Pd = prediction(L, S)
    Pd.to_csv(SR / "state_response_prediction.csv", index=False)
    print("\n== PREDICTION: pre slope (raw units) x change in mean pre-lick power vs observed change in response")
    print(Pd[Pd.epoch == "acute"][["column", "kind", "animal", "power", "pre_slope_raw", "pre_slope_se",
                                   "post_slope_raw", "power_ratio", "pre_resp", "observed", "predicted", "frac"]]
          .round(4).to_string(index=False))
    agg = (Pd.groupby(["epoch", "column", "kind", "power"])[["predicted", "observed"]].sum()
           .assign(ratio_of_sums=lambda d: d.predicted / d.observed).reset_index())
    agg.to_csv(SR / "state_response_prediction_summed.csv", index=False)
    print("\n== summed over animals (ratio of sums)\n" + agg.round(3).to_string(index=False))
    T = stats(S)
    show = T[(T.model == "epoch")]
    print("\nLMM epoch model (per-session values; pre_level = mean pre value):")
    print(show[["measure", "kind", "column", "term", "coef", "se", "p", "q", "re", "n"]].round(4).to_string(index=False))
    tm = T[(T.model == "time") & (T.term != "pre_level")]
    print("\nLMM time model:")
    print(tm[["measure", "kind", "column", "term", "coef", "se", "p", "q"]].round(4).to_string(index=False))
    TL = lick_lmm(L, S)
    print("\nLick-level mixed model (resp ~ animal + epoch * z(power), random per session):")
    print(TL.round(4).to_string(index=False))
    fig_slopes(S, T)
    fig_prediction(Pd)
    fig_matched(L, S)
    (SR / "README.txt").write_text(__doc__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.jobs > 4:
        raise SystemExit("--jobs 4 max on this shared box")
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = a.only or [k for k in sorted(curated_labels()) if not config.early_pre(k) and k[:4] in ANIMALS]
        res, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if len(fail) > 3:
            raise SystemExit("too many failures")
        if a.only:
            return
    summarise()


if __name__ == "__main__":
    main()
