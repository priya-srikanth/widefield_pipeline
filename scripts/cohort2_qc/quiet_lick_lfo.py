"""T16: lick-locked low-frequency (1-4 Hz) activity vs the quiet-period 1-4 Hz increase, same sessions (cohort1, 11-01).

WHY. Quiet-period orofacial fluctuations RISE after the VLS stroke, most at 2-3 Hz (quiet events handoff 3.1-3.7).
Ramanathan et al. 2018 (Nat Med) found movement-locked low-frequency oscillations (LFOs, ~1-4 Hz) in perilesional
motor cortex during reaching FALL after stroke and their return tracks recovery. T16 asks, in the SAME sessions,
whether task-locked 1-4 Hz activity around licks falls while quiet 1-4 Hz power rises.

EVENTS (DAQ clock; canonical behaviour events `behavior_summary/events/<animal>/<date>.npz` lick_onsets, trials from
`trial_windows.trial_bounds`, quit from `analysis_kit.session_behavior`), ENGAGED part of the session only:
  cued   = first DAQ lick onset in [cue, stop_s] (stop_s missing -> cue + 3.5 s) of each trial.
  iti    = bout-initiating ITI lick: DAQ lick onset with no lick in the preceding 2 s, outside every trial's
           [cue, stop_s], and no cue / spout strobe within [-2, +0.75] s (as iti_licks.py's bout rule).
  Epochs whose window touches a swapped imaging pair (+/- 0.5 s) are dropped.
TRACES: `quiet_area_spikes.session_traces(detrend=False)` (encoder inputs, dF/F %), columns = quiet_fluct.TRACE_COLS;
swapped-pair frames (+/-0.5 s) linearly interpolated, then a zero-phase 1-4 Hz band-pass (Butterworth order 3,
sosfiltfilt) and the Hilbert transform on the WHOLE session before epoching; window -0.5 .. +1.5 s around lick onset.
MEASURES per session x event type x column (>= MIN_EV events):
  (a) evoked amplitude = peak-to-trough of the trial-averaged band-passed response over the window;
  (b) ITPC = |mean over trials of exp(i phase)|, averaged over 0 .. 0.5 s; PPC = the n-unbiased version
      ((n R^2 - 1) / (n - 1)), because ITPC and (a) both carry a 1/sqrt(n) noise floor and n changes post-stroke;
  (c) induced power = mean single-trial power of (trial - trial average) over the window; total = without removal;
  null: (a) and (b) for n-matched random onsets in the engaged session (median of N_NULL draws).
QUIET 1-4 Hz power: `v2_quiet/fluct/sessions/<label>.npz` psd integrated over 1-4 Hz (Welch on engaged-quiet runs >=
4 s), plus a check from this script's band-passed trace (variance over engaged quiet frames, quiet_spikes_v2 rule).
Every measure is divided by the animal's AUGUST-pre median (pre = MMDD >= 0806).
STATS: per-animal x epoch table first (rule 8), then `mixed_models.lmm_table` epoch + time models (log10 ratio for
amplitudes / powers, raw ITPC / PPC), FDR within measure.
RECOVERY LINK (within animal, post sessions): Pearson r of lick-locked ITPC (and the other measures) vs far-contra hit
rate (engaged trials, `session_behavior` trials, code 5 = far_R = far contra), raw and with log(day) regressed out of
both; mean r over animals; within-animal permutation (N_PERM).

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_lick_lfo.py [--jobs 3] [--summarise] [--only PS92_0811 ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EP, EVENTS, OUT, columns, session_traces  # noqa: E402
from quiet_fluct import FL, TRACE_COLS  # noqa: E402
from quiet_spikes_v2 import busy_frames  # noqa: E402

LL = OUT / "v2_quiet" / "lick_lfo"
BAND = (1.0, 4.0)
WIN = (-0.5, 1.5)
ITPC_WIN = (0.0, 0.5)
GAP_S, CUE_EXCL, RESP_FALLBACK = 2.0, (-2.0, 0.75), 3.5
MIN_EV, N_NULL, N_PERM = 15, 200, 5000
FAR_CONTRA = 5                                     # POSITION_NAMES[5] = far_R = far contra (left lesion)
KINDS = ("cued", "iti")
KIND_LAB = {"cued": "cued first lick", "iti": "bout-initiating ITI lick"}
SHOW = ["orofacial", "orofacial_left", "orofacial_right", "limb", "cortex"]
MEAS = ["evoked", "itpc", "ppc", "induced", "total", "evoked_null", "itpc_null", "quiet_psd", "quiet_bp"]
MLAB = {"evoked": "lick-locked evoked 1-4 Hz\namplitude / pre", "itpc": "lick-locked ITPC 0-0.5 s / pre",
        "ppc": "PPC 0-0.5 s", "induced": "induced 1-4 Hz power / pre", "total": "total 1-4 Hz power / pre",
        "quiet_psd": "quiet 1-4 Hz power / pre"}


def bandpass(X, fs):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(3, BAND, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def interp_bad(X, bad):
    """Linear interpolation over frames `bad` (and any non-finite frame), row by row."""
    X = X.copy()
    t = np.arange(X.shape[1])
    for i in range(len(X)):
        b = bad | ~np.isfinite(X[i])
        if b.all():
            continue
        if b.any():
            X[i, b] = np.interp(t[b], t[~b], X[i, ~b])
    return X


def lick_events(label, ft, engaged_until):
    """(cued first-lick onsets, ITI bout-start onsets) in s on the DAQ clock, engaged part only."""
    from wfield_local import trial_windows as TW
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    lk = np.sort(z["lick_onsets"] / float(z["fs"]))
    tb = TW.trial_bounds(animal, f"2026{mmdd}")
    cue = tb.cue_s.to_numpy(float)
    stop = tb.stop_s.to_numpy(float)
    stop = np.where(np.isfinite(stop), stop, cue + RESP_FALLBACK)
    cued = []
    for c, s in zip(cue, stop):
        if not np.isfinite(c):
            continue
        j = np.searchsorted(lk, c)
        if j < len(lk) and lk[j] <= s:
            cued.append(lk[j])
    cued = np.asarray(cued)
    gap = np.diff(np.r_[-np.inf, lk])
    inwin = np.zeros(len(lk), bool)
    for c, s in zip(cue, stop):
        if np.isfinite(c):
            inwin[np.searchsorted(lk, c):np.searchsorted(lk, s, side="right")] = True
    ev = np.sort(np.r_[cue, tb.strobe_s.to_numpy(float)])
    ev = ev[np.isfinite(ev)]
    near = (np.searchsorted(ev, lk + CUE_EXCL[1], side="right") - np.searchsorted(ev, lk + CUE_EXCL[0])) > 0
    iti = lk[(gap >= GAP_S) & ~inwin & ~near]
    ok = lambda t: t[(t < engaged_until) & (t + WIN[0] > ft[0] + 1) & (t + WIN[1] < ft[-1] - 1)]   # noqa: E731
    return ok(cued), ok(iti), tb


def far_contra_hit(bh):
    """(far-contra hit rate, all-position hit rate, n far-contra) over engaged trials (before the quit)."""
    if bh is None:
        return np.nan, np.nan, 0
    tr = bh.trials
    if not bh.quit["censored"]:
        tr = [r for r in tr if float(r["elapsed_s"]) < float(bh.quit["quit_elapsed_s"])]
    fc = [r["hit"] for r in tr if r["pos"] == FAR_CONTRA]
    return (float(np.mean(fc)) if fc else np.nan), float(np.mean([r["hit"] for r in tr])), len(fc)


def extract(label):
    """One session -> lick_lfo/sessions/<label>.npz (module-level for spawn)."""
    from scipy.signal import hilbert
    from wfield_local.analysis_kit import session_behavior
    f = LL / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    Xr, ft, fs, rest, engaged, quit_s = session_traces(label, detrend=False)
    cols = columns()
    X = Xr[[cols.index(c) for c in TRACE_COLS]]
    busy, badw = busy_frames(label, ft, fs)
    Xi = interp_bad(X, badw)
    BP = bandpass(Xi, fs)
    AN = hilbert(BP, axis=-1)
    PH = np.angle(AN)
    qe = ~busy & ~badw & np.isfinite(X).all(0) & engaged
    quiet_bp = np.nanvar(BP[:, qe], axis=1) if qe.sum() > 10 * fs else np.full(len(TRACE_COLS), np.nan)
    until = quit_s if np.isfinite(quit_s) else np.inf
    cued, iti, _ = lick_events(label, ft, until)
    lags = np.arange(int(np.floor(WIN[0] * fs)), int(np.ceil(WIN[1] * fs)) + 1)
    bad_cs = np.r_[0, np.cumsum(badw)]

    def frames(t):
        i0 = np.searchsorted(ft, t)                     # first frame at / after the onset
        lo, hi = i0 + lags[0], i0 + lags[-1] + 1
        okk = (lo >= 0) & (hi <= len(ft))
        okk[okk] &= (bad_cs[hi[okk]] - bad_cs[lo[okk]]) == 0
        return i0[okk], t[okk]
    out = {}
    for kind, t in (("cued", cued), ("iti", iti)):
        i0, tk = frames(t)
        idx = i0[:, None] + lags[None, :]
        out[f"{kind}_bp"] = BP[:, idx].transpose(1, 2, 0).astype(np.float32)        # n x lags x cols
        out[f"{kind}_ph"] = PH[:, idx].transpose(1, 2, 0).astype(np.float16)
        out[f"{kind}_t"] = tk
    # n-matched null: random onsets in the engaged, swap-free session (same window rules)
    rng = np.random.default_rng(int.from_bytes(label.encode(), "little") % (2 ** 32))
    eng_fr = np.flatnonzero(engaged & ~badw)
    eng_fr = eng_fr[(eng_fr + lags[0] >= 0) & (eng_fr + lags[-1] < len(ft))]
    tw = (lags / fs >= ITPC_WIN[0]) & (lags / fs <= ITPC_WIN[1])
    for kind in KINDS:
        n = len(out[f"{kind}_t"])
        ev_null, it_null = np.full((N_NULL, len(TRACE_COLS)), np.nan), np.full((N_NULL, len(TRACE_COLS)), np.nan)
        if n >= MIN_EV and len(eng_fr) > n:
            for b in range(N_NULL):
                r0 = rng.choice(eng_fr, n, replace=False)
                idx = r0[:, None] + lags[None, :]
                m = BP[:, idx].mean(1)                                               # cols x lags
                ev_null[b] = m.max(1) - m.min(1)
                it_null[b] = np.abs(np.exp(1j * PH[:, idx][:, :, tw]).mean(1)).mean(1)
        out[f"{kind}_evoked_null"] = np.nanmedian(ev_null, 0) if np.isfinite(ev_null).any() else ev_null[0]
        out[f"{kind}_itpc_null"] = np.nanmedian(it_null, 0) if np.isfinite(it_null).any() else it_null[0]
    fch, allh, nfc = far_contra_hit(session_behavior(label))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, cols=np.array(TRACE_COLS), lags_s=lags / fs, fs=fs, quiet_bp=quiet_bp,
                        quiet_min=qe.sum() / fs / 60, quit_s=quit_s, farcontra_hit=fch, all_hit=allh, n_farcontra=nfc,
                        **out)
    return label


# ------------------------------------------------------------------------------------------------------- summary
def measures(d, kind, ci):
    """Measures (a)-(c) for one session x kind x column index; None if too few events."""
    x = d[f"{kind}_bp"][:, :, ci].astype(float)
    n = len(x)
    if n < MIN_EV:
        return None
    lags = d["lags_s"]
    tw = (lags >= ITPC_WIN[0]) & (lags <= ITPC_WIN[1])
    m = x.mean(0)
    z = np.exp(1j * d[f"{kind}_ph"][:, :, ci].astype(float))[:, tw]
    R = np.abs(z.mean(0))
    return dict(n=n, evoked=float(m.max() - m.min()), itpc=float(R.mean()),
                ppc=float(((n * R ** 2 - 1) / (n - 1)).mean()), induced=float(((x - m) ** 2).mean()),
                total=float((x ** 2).mean()), evoked_null=float(d[f"{kind}_evoked_null"][ci]),
                itpc_null=float(d[f"{kind}_itpc_null"][ci]))


def quiet_psd_power(label, ci):
    f = FL / "sessions" / f"{label}.npz"
    if not f.exists():
        return np.nan
    z = np.load(f)
    fr, p = z["freqs"], z["psd"][ci]
    s = (fr >= BAND[0]) & (fr <= BAND[1])
    return float(np.trapezoid(p[s], fr[s])) if np.isfinite(p[s]).all() else np.nan


def session_table():
    import pandas as pd
    from wfield_local import epochs
    rows = []
    for f in sorted((LL / "sessions").glob("*.npz")):
        lab = f.stem
        ep = epochs.epoch_of(lab)
        if ep is None or (ep == "pre" and lab[-4:] < "0806"):
            continue
        d = dict(np.load(f))
        for c in SHOW:
            ci = TRACE_COLS.index(c)
            base = dict(label=lab, animal=lab[:4], epoch=ep, day=epochs.days_since_stroke(lab), column=c,
                        farcontra_hit=float(d["farcontra_hit"]), all_hit=float(d["all_hit"]),
                        quiet_psd=quiet_psd_power(lab, ci), quiet_bp=float(d["quiet_bp"][ci]),
                        quiet_min=float(d["quiet_min"]))
            for kind in KINDS:
                r = measures(d, kind, ci)
                rows.append({**base, "kind": kind, **(r or {"n": len(d[f"{kind}_t"])})})
    S = pd.DataFrame(rows)
    for m in MEAS:
        if m not in S:
            S[m] = np.nan
    # relative to the animal's AUGUST-pre median (per column x kind)
    for m in MEAS:
        pre = S[S.epoch == "pre"].groupby(["animal", "column", "kind"])[m].median()
        key = pd.MultiIndex.from_frame(S[["animal", "column", "kind"]])
        S[f"{m}_rel"] = S[m].to_numpy() / pre.reindex(key).to_numpy()
    return S.sort_values(["column", "kind", "label"]).reset_index(drop=True)


def per_animal(S, cols=("orofacial",)):
    """Rule 8: per animal x epoch MEAN of the per-session ratios to pre (post/pre)."""
    import pandas as pd
    out = []
    for c in cols:
        for kind in KINDS:
            G = S[(S.column == c) & (S.kind == kind)]
            for m in ("evoked", "itpc", "ppc", "induced", "total", "quiet_psd", "quiet_bp"):
                t = G.groupby(["animal", "epoch"])[f"{m}_rel" if m != "ppc" else m].mean().unstack()
                t = t.reindex(columns=EP)
                t.insert(0, "measure", m)
                t.insert(0, "kind", kind)
                t.insert(0, "column", c)
                out.append(t.reset_index())
    return pd.concat(out, ignore_index=True)


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = []
    for (c, kind), G in S[S.column.isin(SHOW)].groupby(["column", "kind"]):
        for m in ("evoked", "itpc", "ppc", "induced", "total", "quiet_psd"):
            if m == "quiet_psd" and kind != "cued":
                continue                                   # quiet power is per session, not per lick type
            y = G[m].to_numpy(float) if m in ("itpc", "ppc") else np.log10(G[f"{m}_rel"].to_numpy(float))
            L.append(G[["label"]].assign(column=c, kind=kind if m != "quiet_psd" else "quiet", measure=m, y=y))
    L = mm.add_time_covariates(pd.concat(L), august_pre=True)
    L = L[np.isfinite(L.y)]
    T = mm.lmm_table(L, ["measure", "kind", "column"], models=("epoch", "time"), print_animals=False,
                     fdr_by=("measure",))
    T.to_csv(LL / "lmm_lick_lfo.csv", index=False)
    return T


def _resid(y, d):
    Xd = np.column_stack([np.ones(len(y)), d])
    return y - Xd @ np.linalg.lstsq(Xd, y, rcond=None)[0]


def recovery_link(S, rng):
    """Within animal, post sessions: r(measure, far-contra hit rate), raw and log(day)-detrended; permutation."""
    import pandas as pd
    rows, keep = [], {}
    meas = [("cued", "itpc"), ("cued", "ppc"), ("cued", "evoked_rel"), ("cued", "induced_rel"), ("iti", "itpc"),
            ("iti", "ppc"), ("iti", "evoked_rel"), ("cued", "quiet_psd_rel")]
    G0 = S[(S.column == "orofacial") & (S.epoch != "pre") & (S.day > 0)]
    for kind, m in meas:
        for detr in (False, True):
            per, data = {}, {}
            for an, g in G0[G0.kind == kind].groupby("animal"):
                g = g.dropna(subset=[m, "farcontra_hit"])
                if len(g) < 5:
                    continue
                b, y = g.farcontra_hit.to_numpy(float), g[m].to_numpy(float)
                if detr:
                    d = np.log(g.day.to_numpy(float))
                    b, y = _resid(b, d), _resid(y, d)
                per[an] = float(np.corrcoef(b, y)[0, 1])
                data[an] = (b, y)
            if not per:
                continue
            obs = np.mean(list(per.values()))
            null = np.array([np.mean([np.corrcoef(rng.permutation(b), y)[0, 1] for b, y in data.values()])
                             for _ in range(N_PERM)])
            p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))
            rows.append(dict(kind=kind, measure=m, detrended_logday=detr, mean_r=obs, p_perm=p,
                             n_pos=sum(v > 0 for v in per.values()), n_animals=len(per),
                             **{f"r_{a}": v for a, v in per.items()}, **{f"n_{a}": len(data[a][0]) for a in data}))
            if (kind, m, detr) == ("cued", "itpc", True):
                keep = data
    R = pd.DataFrame(rows)
    R.to_csv(LL / "recovery_link.csv", index=False)
    return R, keep


# ------------------------------------------------------------------------------------------------------- figures
def fig_epochs(S, T):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    panels = ["evoked", "itpc", "induced", "quiet_psd"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6), sharey=False)
    rng = np.random.default_rng(0)
    G0 = S[S.column == "orofacial"]
    for i, kind in enumerate(KINDS):
        G = G0[G0.kind == kind]
        for j, m in enumerate(panels):
            a, col = ax[i, j], f"{m}_rel"
            for x, ep in enumerate(EP):
                B = G[G.epoch == ep]
                for an, A in B.groupby("animal"):
                    a.scatter(x + rng.uniform(-0.18, 0.18, len(A)), A[col], marker=ANIMAL_MARKER[an],
                              facecolors="none", edgecolors=EPOCH_LINE[ep], s=18, lw=0.8, alpha=0.8)
                ci = boot_ci({an: list(A[col].dropna()) for an, A in B.groupby("animal") if A[col].notna().any()},
                             rng)
                if ci is not None:
                    a.errorbar(x + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="o",
                               color=EPOCH_LINE[ep], mfc=EPOCH_LINE[ep], ms=6, capsize=3, lw=1.6, zorder=5)
                kk = "quiet" if m == "quiet_psd" else kind
                t = T[(T.model == "epoch") & (T.measure == m) & (T.kind == kk) & (T.column == "orofacial")
                      & (T.term == f"C(epoch)[T.{ep}]")]
                if x > 0 and len(t) and t.q.iloc[0] < 0.05:
                    q = t.q.iloc[0]
                    sig_marks.star_top(a, x, "***" if q < 0.001 else "**" if q < 0.01 else "*", color=EPOCH_LINE[ep])
            a.axhline(1, color="0.6", lw=0.8, ls="--")
            a.set_yscale("log")
            a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
            a.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
            a.tick_params(axis="y", which="both", labelsize=7)
            a.set_xticks(range(len(EP)), EP, fontsize=8)
            a.set_title(("QUIET (no lick / running within 1.5 s)\n" if m == "quiet_psd" else f"{KIND_LAB[kind]}\n")
                        + MLAB[m].replace("\n", " "), fontsize=8.5)
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    ax[0, -1].legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
                     title="animal", title_fontsize=7)
    sig_marks.reserve_headroom(fig)
    fig.suptitle("T16: lick-locked 1-4 Hz activity vs quiet-period 1-4 Hz power, orofacial cortex (bilateral), same "
                 "sessions. Each value / the animal's August-pre median (log axis; dashed = pre).\nBand-pass 1-4 Hz "
                 "zero-phase on the whole session; window -0.5..+1.5 s around DAQ lick onset; ITPC = Hilbert phase "
                 "consistency over 0-0.5 s; induced = single-trial power after removing the trial average. Open "
                 "markers = sessions (shape = animal, colour = epoch); filled = pooled mean, nested animal -> session "
                 "bootstrap 95% CI; stars = mixed model, epoch vs August pre, FDR q.", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    fig.savefig(LL / "lick_lfo_epochs.png", dpi=110)
    plt.close(fig)
    print("wrote", LL / "lick_lfo_epochs.png")


def fig_traces():
    """Per animal: epoch-mean lick-locked band-passed orofacial response and ITPC time course."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import epochs
    from wfield_local.orofacial_cohort import EPOCH_LINE
    ci = TRACE_COLS.index("orofacial")
    acc = {}
    lags = None
    for f in sorted((LL / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or (ep == "pre" and f.stem[-4:] < "0806"):
            continue
        d = np.load(f)
        lags = d["lags_s"] if lags is None else lags
        for kind in KINDS:
            x = d[f"{kind}_bp"][:, :, ci].astype(float)
            if len(x) < MIN_EV or x.shape[1] != len(lags):
                continue
            z = np.exp(1j * d[f"{kind}_ph"][:, :, ci].astype(float))
            acc.setdefault((kind, f.stem[:4], ep), []).append((x.mean(0), np.abs(z.mean(0))))
    ans = ["PS92", "PS93", "PS94", "PS95"]
    fig, ax = plt.subplots(4, 4, figsize=(15, 11), sharex=True, sharey="row")
    for i, kind in enumerate(KINDS):
        for j, an in enumerate(ans):
            ns = []
            for ep in EP:
                L = acc.get((kind, an, ep))
                if not L:
                    continue
                ns.append(f"{ep[:3]} {len(L)}")
                ax[2 * i, j].plot(lags, np.mean([a for a, _ in L], 0), color=EPOCH_LINE[ep], lw=1.3, label=ep)
                ax[2 * i + 1, j].plot(lags, np.mean([b for _, b in L], 0), color=EPOCH_LINE[ep], lw=1.3)
            for r in (2 * i, 2 * i + 1):
                ax[r, j].axvline(0, color="0.6", lw=0.8)
            ax[2 * i + 1, j].axvspan(*ITPC_WIN, color="0.92", zorder=0)
            ax[2 * i, j].set_title(f"{an}: {KIND_LAB[kind]}\nsessions: " + ", ".join(ns), fontsize=8.5)
        ax[2 * i, 0].set_ylabel(f"{kind}: 1-4 Hz band-passed\ndF/F % (trial mean)", fontsize=8)
        ax[2 * i + 1, 0].set_ylabel(f"{kind}: ITPC", fontsize=8)
    for a in ax[-1]:
        a.set_xlabel("s from DAQ lick onset")
    ax[0, -1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="epoch")
    fig.suptitle("Lick-locked 1-4 Hz activity in orofacial cortex (bilateral) per animal: mean over sessions of each "
                 "session's trial-averaged band-passed response (rows 1, 3) and inter-trial phase coherence (rows 2, "
                 "4; grey = the 0-0.5 s window scored). Colour = epoch (pre = August pre).", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.2)
    fig.savefig(LL / "lick_lfo_traces.png", dpi=110)
    plt.close(fig)
    print("wrote", LL / "lick_lfo_traces.png")


def fig_recovery(S, R, keep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    G0 = S[(S.column == "orofacial") & (S.epoch != "pre") & (S.day > 0)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    colr = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
    for an, g in G0[G0.kind == "cued"].groupby("animal"):
        ax[0].plot(g.day, g.itpc, marker=ANIMAL_MARKER[an], mfc="none", color=colr[an], lw=0.8, label=an)
        ax[1].plot(g.day, g.farcontra_hit, marker=ANIMAL_MARKER[an], mfc="none", color=colr[an], lw=0.8)
    for an, (b, y) in keep.items():
        ax[2].scatter(b, y, marker=ANIMAL_MARKER[an], facecolors="none", edgecolors=colr[an], s=22)
    for a in ax[:2]:
        a.set_xscale("log")
        a.set_xlabel("days since stroke (log)")
    ax[0].set_ylabel("cued-lick ITPC 0-0.5 s (orofacial)")
    ax[1].set_ylabel("far-contra hit rate (engaged trials)")
    ax[1].legend(*ax[0].get_legend_handles_labels(), fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1),
                 frameon=False)
    ax[2].set_xlabel("far-contra hit rate, log(day) removed")
    ax[2].set_ylabel("cued-lick ITPC, log(day) removed")
    r = R[(R.kind == "cued") & (R.measure == "itpc") & R.detrended_logday]
    if len(r):
        r = r.iloc[0]
        txt = "\n" + ", ".join(f"{a} {r[f'r_{a}']:+.2f}" for a in ["PS92", "PS93", "PS94", "PS95"] if f"r_{a}" in r and
                        np.isfinite(r[f"r_{a}"]))
        ax[2].set_title(f"within animal r: {txt}\nmean {r.mean_r:+.2f}, permutation p = {r.p_perm:.3f}", fontsize=8.5)
    ax[0].set_title("lick-locked ITPC across post-stroke sessions", fontsize=9)
    ax[1].set_title("far-contra recovery", fontsize=9)
    fig.suptitle("Does lick-locked 1-4 Hz phase locking track far-contra recovery within animal (post sessions)? "
                 "Ramanathan 2018 predicts r > 0 after removing the shared log(day) recovery trend.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=3)
    fig.savefig(LL / "lick_lfo_recovery.png", dpi=110)
    plt.close(fig)
    print("wrote", LL / "lick_lfo_recovery.png")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    S = session_table()
    S.to_csv(LL / "lick_lfo_sessions.csv", index=False)
    print(f"{S.label.nunique()} sessions")
    print("\n== events per session (orofacial), median by animal x epoch\n"
          + S[S.column == "orofacial"].pivot_table(index=["kind", "animal"], columns="epoch", values="n",
                                                    aggfunc="median").reindex(columns=EP).to_string())
    A = per_animal(S, ("orofacial", "orofacial_left", "orofacial_right"))
    A.to_csv(LL / "lick_lfo_per_animal.csv", index=False)
    print("\n== RULE 8: per animal x epoch, mean of per-session values / animal's August-pre median (PPC raw)")
    print(A[A.column == "orofacial"].round(3).to_string(index=False))
    o = S[(S.column == "orofacial")]
    print("\n== null levels (orofacial, median over sessions): evoked/null, ITPC vs null\n"
          + o.assign(ev_over_null=o.evoked / o.evoked_null, itpc_minus_null=o.itpc - o.itpc_null)
          .groupby(["kind", "epoch"])[["ev_over_null", "itpc_minus_null", "itpc", "itpc_null"]].median()
          .reindex(EP, level=1).round(3).to_string())
    print("\n== quiet-power check: Welch psd vs band-passed variance, Spearman over sessions")
    q = o[o.kind == "cued"]
    print(q[["quiet_psd", "quiet_bp"]].corr(method="spearman").round(3).to_string())
    T = stats(S)
    show = T[(T.model == "epoch") & (T.term != "pre_level") & (T.column == "orofacial")]
    print("\nLMM epoch model vs August pre (log10 ratio for amplitude / power; raw ITPC / PPC), orofacial:")
    print(show[["measure", "kind", "term", "coef", "se", "p", "q", "re", "n"]].round(4).to_string(index=False))
    tm = T[(T.model == "time") & (T.term != "pre_level") & (T.column == "orofacial")]
    print("\nLMM time model (post + log day), orofacial:")
    print(tm[["measure", "kind", "term", "coef", "se", "p", "q"]].round(4).to_string(index=False))
    R, keep = recovery_link(S, np.random.default_rng(0))
    print("\n== recovery link (within animal, post sessions, orofacial)\n" + R.round(3).to_string(index=False))
    fig_epochs(S, T)
    fig_traces()
    fig_recovery(S, R, keep)
    (LL / "README.txt").write_text(__doc__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.jobs > 5:
        raise SystemExit("--jobs 5 max on this shared box (coordinator 11-01)")
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels
        from wfield_local.parallel import fan_out
        labels = a.only or [k for k in sorted(curated_labels()) if k[-4:] >= "0806" and k[:4] in
                            ("PS92", "PS93", "PS94", "PS95")]
        res, fail = fan_out(labels, extract, jobs=a.jobs, label="session")
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
