"""T10 (quiet-events handoff 11-01): are the post-stroke quiet-period orofacial events / 2-3 Hz power CALCIUM, or blood /
light? The same measures on the corrected 470 (adopted), the UNCORRECTED 470 and the 415 nm channel (cohort1, PS92-95).

QUESTION. After the VLS stroke, brief quiet-period events in orofacial cortex rise ~5x acutely and quiet power rises at
~2-3 Hz (quiet_spikes_v2.py, quiet_fluct.py). The analysed signal is the hemo-corrected 470 (hemo_meegkit_hpfit SVTcorr,
`combined_encoding.load_imaging`). If the effect were haemodynamic (or a light / motion artefact common to both
wavelengths), the 415 nm channel -- calcium-insensitive (isosbestic, plus a small NEGATIVE calcium term) -- would show it
too, and uncorrected 470 would show it larger than corrected. If it is calcium, 415 shows no increase and no event-locked
response.

CHANNELS. `corr470` = adopted SVTcorr (identical traces to quiet_area_spikes.session_traces, asserted per session);
`raw470` = functional half of raw SVT.npy; `raw415` = the other half. RAW SVT is (K, 2T), the two channels interleaved;
the functional slot is `hemo_variants.functional_channel(label)` (0 for PS92_0828) and is checked against the session's
hemo manifest -- exactly the loader logic of scripts/iti_hemo_check.py. Both halves are dF/F on the same U basis
(divide_by_average per channel), so the same ROI weights (ROI mean of U_atlas, stat-mask pixels, quiet_area_spikes ROI
logic) project all three. No extra drift removal: every measure works on 2-s-median-detrended traces or 4-s linearly
detrended Welch segments.
FRAMES: QUIET ENGAGED (quiet_spikes_v2.busy_frames: no DAQ lick / running bout within 1.5 s, no swapped pair within
0.5 s; engaged = before the quit), identical for all channels. August sessions; pre = August pre (not config.early_pre, cohort1 cut 0806).
MEASURES per session x channel x ROI (orofacial_left / _right / orofacial / limb_left / _right / limb / cortex):
  (a) sd     robust SD (1.4826 x MAD) of the 2-s-median-detrended trace over quiet engaged frames
  (b) p14 / p0510  quiet band power 1-4 Hz / 0.5-1 Hz ((dF/F %)^2), Welch 4-s Hann segments, linear detrend, on quiet
             engaged runs >= 4 s (quiet_fluct definition), band power = sum PSD x df
  (c) rate   v2 detector (quiet_spikes_v2.detect) on THAT channel's detrended trace, both polarities, events / min of
             engaged quiet, threshold k = 4 x the animal's August-pre robust SD OF THAT CHANNEL (and ROI)
  (d) eta    at the CORRECTED-470 v2 events (v2_quiet/sessions npz, engaged, height > 4 x animal pre SD of corr470; the
             v2 threshold), the event-locked amplitude in each channel = detrended trace at the event frame minus the
             median of the preceding 1.0-0.2 s (the detector's height definition), sign-flipped for negative events so
             "amplitude in the event's direction"; pooled over both polarities. eta_ratio = 415 (or raw470) / corr470
             per session. NULL: the same at 400 random quiet engaged frames.
Per-animal x epoch table first (CLAUDE.md rule 8): ratio = median over the epoch's sessions / median over the animal's
pre sessions. Stats: mixed_models.lmm_table, epoch + time models vs August pre, on log10(sd), log10(power),
log10(rate + 0.2) and the linear eta amplitudes / ratios; FDR within measure.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_hemo_check.py [--jobs 3] [--summarise]
Output: M:/.../cohort1_quiet_spikes/v2_quiet/hemo415/ (sessions/<label>.npz cache, csv, quiet_hemo_check.png)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EP, GROUPS, columns, mad, session_traces  # noqa: E402
from quiet_spikes_v2 import BASE_S, V2, busy_frames, detect, detrend  # noqa: E402
from wfield_local import config  # noqa: E402

HO = V2 / "hemo415"
CACHE_VERSION = 1
CH = ("corr470", "raw470", "raw415")
CHLAB = {"corr470": "corrected 470 (analysed)", "raw470": "uncorrected 470", "raw415": "415 nm (calcium-insensitive)"}
ROIS = ["orofacial_left", "orofacial_right", "orofacial", "limb_left", "limb_right", "limb", "cortex"]
K_EVT, SEG_S, N_NULL = 4.0, 4.0, 400
BANDS = {"p14": (1.0, 4.0), "p0510": (0.5, 1.0)}
SHOW_ROI = "orofacial"


def _session(label):
    from wfield_local import config
    return next(x for x in config.load_sessions() if x["label"] == label)


def _roi_weights(Ub, flat, parea, names, shape, pstat):
    """ROI-mean rows of U_atlas in columns() order -- the quiet_area_spikes.session_traces logic."""
    from iti_licks import roi_masks
    m = roi_masks(parea, pstat, names, flat, shape)
    W = []
    for c in columns():
        if c == "cortex":
            mk = m["cortex"]
        elif c in GROUPS:
            mk = np.any([m[f"{a}_{h}"] for a in GROUPS[c] for h in ("left", "right") if f"{a}_{h}" in m], 0)
        elif c.rsplit("_", 1)[0] in GROUPS:
            g, h = c.rsplit("_", 1)
            mk = np.any([m[f"{a}_{h}"] for a in GROUPS[g] if f"{a}_{h}" in m], 0)
        else:
            mk = m.get(c, np.zeros(len(flat), bool))
        W.append(Ub[mk].mean(0) if mk.any() else np.full(Ub.shape[1], np.nan, np.float32))
    return np.asarray(W, np.float64)


def _animal_pre_sd_v2():
    """{animal: corr470 robust SD per column}, median over August-pre v2 sessions (the v2 threshold base)."""
    from wfield_local import epochs
    out = {}
    for f in sorted((V2 / "sessions").glob("*.npz")):
        if epochs.epoch_of(f.stem) == "pre" and not config.early_pre(f.stem):
            out.setdefault(f.stem[:4], []).append(np.load(f)["sd_q"])
    return {a: np.nanmedian(v, 0) for a, v in out.items()}


def work(label):
    """One session -> HO/sessions/<label>.npz. Module-level for spawn."""
    from scipy.signal import welch
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.hemo_variants import functional_channel
    f = HO / "sessions" / f"{label}.npz"
    if f.exists() and int(np.load(f)["cache_version"]) == CACHE_VERSION:
        return {"label": label, "cached": True}
    Xc, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)            # corrected, cols x T, dF/F %
    V, ft2, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    n = Xc.shape[1]
    W = _roi_weights(Ub, flat, parea, names, shape, pstat)
    chk = 100 * (W @ V[:min(n, 2000)].T.astype(np.float64))
    if not np.allclose(chk, Xc[:, :chk.shape[1]], rtol=1e-4, atol=1e-6, equal_nan=True):
        raise RuntimeError(f"{label}: ROI weights do not reproduce session_traces")
    res = Path(_session(label)["mc"]) / "wfield_local_results"
    fc = functional_channel(label)
    man = json.loads(Path(config.hemo_manifest_path(res)).read_text())   # selected variant, call time (review L2)
    mf = int(man.get("functional_channel", -1))
    if mf != fc:
        raise RuntimeError(f"{label}: functional slot {fc} vs hemo manifest {mf}")
    svt = np.load(res / "SVT.npy", mmap_mode="r")                                   # (K, 2T), read-only
    if abs(svt.shape[1] // 2 - n) > 2 and abs(svt.shape[1] // 2 - V.shape[0]) > 2:
        raise RuntimeError(f"{label}: raw SVT {svt.shape[1]} samples vs corrected {V.shape[0]} frames")
    if svt.shape[0] != W.shape[1]:
        raise RuntimeError(f"{label}: SVT K {svt.shape[0]} vs U K {W.shape[1]}")
    Wn = np.nan_to_num(W)
    Y = np.zeros((len(W), svt.shape[1]))
    step = 20000
    for a in range(0, svt.shape[1], step):                                          # chunked: memory-light
        Y[:, a:a + step] = Wn @ np.asarray(svt[:, a:a + step], np.float64)
    Y *= 100
    Y[~np.isfinite(W).all(1)] = np.nan
    X = {"corr470": Xc}
    for v, slot in (("raw470", fc), ("raw415", (fc + 1) % 2)):
        Z = Y[:, slot::2][:, :n]
        if Z.shape[1] < n:
            raise RuntimeError(f"{label} {v}: {Z.shape[1]} frames < {n}")
        X[v] = Z
    del Y
    busy, badw = busy_frames(label, ft, fs)
    quiet = ~busy & ~badw & np.isfinite(Xc).all(0) & np.isfinite(X["raw415"]).all(0)
    qe = quiet & engaged
    cols = columns()
    ri = [cols.index(c) for c in ROIS]
    nseg = int(round(SEG_S * fs))
    e = np.flatnonzero(np.diff(np.r_[0, qe.astype(int), 0]))
    runs = [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= nseg]
    b0, b1 = int(round(BASE_S[0] * fs)), int(round(BASE_S[1] * fs))
    # corrected-470 v2 events (the published ones) for (d); engaged only; threshold applied in the summary
    v2 = np.load(V2 / "sessions" / f"{label}.npz")
    out = dict(cache_version=CACHE_VERSION, label=label, fs=fs, func_slot=fc, n_frames=n, svt_pairs=svt.shape[1] // 2,
               n_quiet_eng=int(qe.sum()), rois=np.array(ROIS), n_runs=len(runs))
    rng = np.random.default_rng(int(label[2:4]) * 10000 + int(label[-4:]))
    cand = np.flatnonzero(qe)
    cand = cand[cand >= b0]
    null_fr = np.sort(rng.choice(cand, min(N_NULL, len(cand)), replace=False)) if len(cand) else np.zeros(0, int)
    for v in CH:
        Xd = detrend(X[v][ri], fs)
        out[f"sd_{v}"] = mad(Xd[:, qe]) if qe.sum() > 100 else np.full(len(ri), np.nan)
        # (b) Welch on quiet engaged runs >= 4 s
        P, wsum = None, 0
        for a, b in runs:
            fr, p = welch(X[v][ri, a:b], fs=fs, nperseg=nseg, noverlap=nseg // 2, detrend="linear", axis=-1)
            w = 1 + (b - a - nseg) // (nseg // 2)
            P = p * w if P is None else P + p * w
            wsum += w
        if not wsum:
            fr = np.fft.rfftfreq(nseg, 1 / fs)
        psd = P / wsum if wsum else np.full((len(ri), len(fr)), np.nan)
        out["freqs"], out[f"psd_{v}"] = fr, psd.astype(np.float32)
        df = fr[1] - fr[0]
        for bn, (lo, hi) in BANDS.items():
            out[f"{bn}_{v}"] = psd[:, (fr >= lo) & (fr < hi)].sum(1) * df
        # (c) v2 detector on this channel, both polarities, all candidate heights stored (k applied in summary)
        ev = []
        for j in range(len(ri)):
            if ROIS[j] == "cortex" or not np.isfinite(Xd[j]).all():
                continue
            for sign in (1, -1):
                pk, h = detect(sign * Xd[j], quiet, fs)
                ev += [(j, int(p), float(hh), int(engaged[p]), sign) for p, hh in zip(pk, h)]
        out[f"ev_{v}"] = np.array(ev, float).reshape(-1, 5)

        # (d) event-locked amplitude at corr470 v2 events: height in the event's direction, every channel
        def amp(frames, j, Xd=Xd):
            frames = frames[frames >= b0]
            if not len(frames):
                return np.zeros(0)
            return np.array([Xd[j, p] - np.median(Xd[j, p - b0:p - b1]) for p in frames])
        for j, c in enumerate(ROIS):
            if c == "cortex":
                continue
            gi = cols.index(c)
            for kind, sign in (("pos", 1), ("neg", -1)):
                a_ = v2[kind]
                sel = (a_[:, 0] == gi) & (a_[:, 3] == 1)
                fr_ev, h_ev = a_[sel, 1].astype(int), a_[sel, 2]
                ok = fr_ev >= b0
                out[f"evh_{c}_{kind}"] = h_ev[ok]                         # corr470 v2 height (threshold in summary)
                out[f"amp_{v}_{c}_{kind}"] = sign * amp(fr_ev[ok], j)
            out[f"null_{v}_{c}"] = amp(null_fr, j)
    HO.joinpath("sessions").mkdir(parents=True, exist_ok=True)
    tmp = HO / "sessions" / f"{label}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(f)
    return {"label": label, "func": fc, "n_quiet_min": round(qe.sum() / fs / 60, 1)}


# --------------------------------------------------------------------------------------------------------- summary
def load():
    from wfield_local import epochs
    D = {}
    for f in sorted((HO / "sessions").glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        ep = epochs.epoch_of(f.stem)
        if ep is None or (ep == "pre" and config.early_pre(f.stem)) or not f.stem.startswith(("PS92", "PS93", "PS94", "PS95")):
            continue
        D[f.stem] = dict(np.load(f), epoch=ep, day=epochs.days_since_stroke(f.stem))
    return D


def session_table(D):
    import pandas as pd
    pre_sd = {}                                    # per animal, channel: median August-pre SD per ROI
    for an in sorted({k[:4] for k in D}):
        for v in CH:
            pre_sd[an, v] = np.nanmedian([d[f"sd_{v}"] for k, d in D.items() if k[:4] == an and d["epoch"] == "pre"], 0)
    v2sd = _animal_pre_sd_v2()
    cols = columns()
    rows = []
    for k_, d in D.items():
        an, fs = k_[:4], float(d["fs"])
        mins = d["n_quiet_eng"] / fs / 60
        if mins < 0.25:
            continue
        r = dict(label=k_, animal=an, epoch=d["epoch"], day=d["day"], quiet_min=mins)
        for v in CH:
            ev = d[f"ev_{v}"]
            for j, c in enumerate(ROIS):
                r[f"{v}|{c}|sd"] = d[f"sd_{v}"][j]
                for bn in BANDS:
                    r[f"{v}|{c}|{bn}"] = d[f"{bn}_{v}"][j]
                if c == "cortex":
                    continue
                thr = K_EVT * pre_sd[an, v][j]
                sel = (ev[:, 0] == j) & (ev[:, 3] == 1) & (ev[:, 2] > thr)
                r[f"{v}|{c}|rate"] = sel.sum() / mins
                # (d) event-locked amplitudes at corr470 v2 events above the v2 threshold
                thr2 = K_EVT * v2sd[an][cols.index(c)]
                A = np.concatenate([d[f"amp_{v}_{c}_{kind}"][d[f"evh_{c}_{kind}"] > thr2] for kind in ("pos", "neg")])
                r[f"{v}|{c}|eta"] = float(np.mean(A)) if len(A) else np.nan
                r[f"{v}|{c}|null"] = float(np.mean(d[f"null_{v}_{c}"])) if len(d[f"null_{v}_{c}"]) else np.nan
                r[f"{c}|n_evt"] = len(A)
        for c in ROIS:
            if c == "cortex":
                continue
            r[f"raw415|{c}|eta_ratio"] = r[f"raw415|{c}|eta"] / r[f"corr470|{c}|eta"]
            r[f"raw470|{c}|eta_ratio"] = r[f"raw470|{c}|eta"] / r[f"corr470|{c}|eta"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True), pre_sd


def animal_table(S, measures, rois=ROIS):
    """Per animal x epoch: median over the epoch's sessions / median over pre sessions (rate: (x + 0.2) ratio)."""
    import pandas as pd
    rows = []
    for an, A in S.groupby("animal"):
        for v in CH:
            for c in rois:
                for m in measures:
                    col = f"{v}|{c}|{m}"
                    if col not in S:
                        continue
                    off = 0.2 if m == "rate" else 0.0
                    pre = A[A.epoch == "pre"][col].median() + off
                    for ep in EP:
                        E = A[A.epoch == ep][col]
                        if E.empty:
                            continue
                        rows.append(dict(animal=an, channel=v, roi=c, measure=m, epoch=ep, n=len(E),
                                         median=E.median(), ratio=(E.median() + off) / pre))
    return pd.DataFrame(rows)


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = []
    for v in CH:
        for c in ROIS:
            for m in ("sd", "p14", "p0510", "rate", "eta", "eta_ratio"):
                col = f"{v}|{c}|{m}"
                if col not in S or (m == "eta_ratio" and v == "corr470"):
                    continue
                y = S[col].astype(float)
                y = (np.log10(y + 0.2) if m == "rate" else np.log10(y) if m in ("sd", "p14", "p0510") else y)
                L.append(S[["label"]].assign(column=f"{v}|{c}", measure=m, y=y.to_numpy()))
    L = mm.add_time_covariates(pd.concat(L), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch", "time"), print_animals=False, fdr_by=("measure",))
    T.to_csv(HO / "lmm_hemo415.csv", index=False)
    return T


def figure(S, T, roi=SHOW_ROI):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    meas = [("sd", "quiet robust SD\n(x animal's pre)"), ("p14", "quiet 1-4 Hz power\n(x animal's pre)"),
            ("p0510", "quiet 0.5-1 Hz power\n(x animal's pre)"), ("rate", "event rate, own-channel k = 4\n((rate+0.2) x animal's pre)"),
            ("eta", "amplitude at corrected-470 events\n(dF/F %, event direction)")]
    fig, ax = plt.subplots(len(CH), len(meas), figsize=(3.4 * len(meas), 2.9 * len(CH) + 1.2), squeeze=False)
    rng = np.random.default_rng(0)
    for i, v in enumerate(CH):
        for j, (m, ylab) in enumerate(meas):
            a, col = ax[i, j], f"{v}|{roi}|{m}"
            Y = S[["animal", "epoch", col]].copy()
            if m != "eta":
                off = 0.2 if m == "rate" else 0.0
                pre = Y[Y.epoch == "pre"].groupby("animal")[col].median()
                Y[col] = (Y[col] + off) / (Y.animal.map(pre) + off)
                a.set_yscale("log")
                a.axhline(1, color="0.6", lw=0.8)
            else:
                a.axhline(0, color="0.6", lw=0.8)
                nul = S[f"{v}|{roi}|null"]
                a.axhspan(nul.quantile(0.05), nul.quantile(0.95), color="0.9", zorder=0)
            for x, ep in enumerate(EP):
                B = Y[Y.epoch == ep]
                for an, A in B.groupby("animal"):
                    a.scatter(x + rng.uniform(-0.18, 0.18, len(A)), A[col], marker=ANIMAL_MARKER[an], facecolors="none",
                              edgecolors=EPOCH_LINE[ep], s=16, lw=0.8, alpha=0.8)
                ci = boot_ci({an: list(A[col].dropna()) for an, A in B.groupby("animal")}, rng)
                if ci is not None:
                    a.errorbar(x + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="o",
                               color=EPOCH_LINE[ep], ms=5, capsize=3, lw=1.5, zorder=5)
                t = T[(T.model == "epoch") & (T.measure == m) & (T.column == f"{v}|{roi}") & (T.term == f"C(epoch)[T.{ep}]")]
                if x > 0 and len(t) and t.q.iloc[0] < 0.05:
                    q = t.q.iloc[0]
                    sig_marks.star_top(a, x, "***" if q < 0.001 else "**" if q < 0.01 else "*", color=EPOCH_LINE[ep])
            if m != "eta":
                yv = Y[col].replace([np.inf, -np.inf], np.nan).dropna()
                yv = yv[yv > 0]
                a.set_ylim(min(yv.min(), 1) / 1.3, max(yv.max(), 1) * 1.3)
                fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                a.yaxis.set_major_formatter(fmt)
                a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
            a.set_xticks(range(len(EP)), EP, fontsize=7)
            if i == 0:
                a.set_title(ylab, fontsize=8)
            if j == 0:
                a.set_ylabel(CHLAB[v], fontsize=8)
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    ax[0, -1].legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
                     title="animal", title_fontsize=7)
    sig_marks.reserve_headroom(fig)
    fig.suptitle("Is the post-stroke quiet-period change in orofacial cortex calcium or haemodynamic? The same measures on "
                 "the corrected 470 (analysed), uncorrected 470 and 415 nm channels -- bilateral orofacial ROI, engaged quiet "
                 "frames, August pre. Open markers = sessions (shape = animal, colour = epoch), filled = pooled mean with "
                 "nested bootstrap 95% CI; stars = LMM epoch vs pre (FDR within measure). Last column: grey band = 5-95% "
                 "of the random-quiet-frame null.", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 0.94, 0.90), w_pad=2.0, h_pad=1.5)
    fig.savefig(HO / "quiet_hemo_check.png", dpi=110)


def summarise():
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    D = load()
    print(f"{len(D)} sessions; 470 slots: {sorted({int(d['func_slot']) for d in D.values()})}; "
          f"slot 0 = {[k for k, d in D.items() if int(d['func_slot']) == 0]}")
    S, pre_sd = session_table(D)
    S.to_csv(HO / "quiet_hemo_sessions.csv", index=False)
    print("\nanimal August-pre robust SD (dF/F %), orofacial / limb:")
    for (an, v), x in pre_sd.items():
        print(f"  {an} {v:8s} " + " ".join(f"{c}={x[j]:.3f}" for j, c in enumerate(ROIS)))
    A = animal_table(S, ("sd", "p14", "p0510", "rate", "eta", "eta_ratio"))
    A.to_csv(HO / "quiet_hemo_animal_epoch.csv", index=False)
    for m in ("sd", "p14", "p0510", "rate"):
        for c in ("orofacial", "orofacial_left", "orofacial_right", "limb", "cortex"):
            G = A[(A.measure == m) & (A.roi == c)]
            if G.empty:
                continue
            print(f"\n== {m} post/pre ratio, {c} ==")
            print(G.pivot_table(index="animal", columns=["channel", "epoch"], values="ratio")
                  .reindex(columns=pd.MultiIndex.from_product([CH, EP])).round(2).to_string())
    for c in ("orofacial", "orofacial_left", "orofacial_right", "limb"):
        print(f"\n== event-locked amplitude at corrected-470 events (dF/F %), {c}: median per animal x epoch ==")
        G = A[(A.measure == "eta") & (A.roi == c)]
        print(G.pivot_table(index="animal", columns=["channel", "epoch"], values="median")
              .reindex(columns=pd.MultiIndex.from_product([CH, EP])).round(3).to_string())
        G = A[(A.measure == "eta_ratio") & (A.roi == c) & (A.channel != "corr470")]
        print(f"   amplitude ratio channel / corr470, {c}:")
        print(G.pivot_table(index="animal", columns=["channel", "epoch"], values="median")
              .reindex(columns=pd.MultiIndex.from_product([CH[1:], EP])).round(3).to_string())
        nul = S.groupby(["animal", "epoch"])[[f"{v}|{c}|null" for v in CH]].median().round(3)
        print("   null (random quiet frames), median:\n", nul.to_string())
    T = stats(S)
    if "model" not in T:
        print("no LMM terms (too few animals / epochs)")
        return
    sh = T[(T.model == "epoch") & (T.term != "pre_level") & T.column.str.contains(r"\|(orofacial|limb|cortex)$")]
    print("\nLMM epoch model vs August pre:\n",
          sh[["measure", "column", "term", "coef", "se", "p", "q", "n"]].round(4).to_string(index=False))
    figure(S, T)
    (HO / "README.txt").write_text(__doc__)
    print("wrote", HO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--plot-only", action="store_true", help="re-render the figure from the saved csv (no LMM refit)")
    a = ap.parse_args()
    if a.plot_only:
        import pandas as pd
        figure(pd.read_csv(HO / "quiet_hemo_sessions.csv"), pd.read_csv(HO / "lmm_hemo415.csv"))
        return
    if not a.summarise:
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = a.only or sorted(f.stem for f in (V2 / "sessions").glob("PS9[2-5]_*.npz"))
        res, fail = fan_sessions(labels, work, jobs=min(a.jobs, 3), label="session")
        for _, r in sorted(res, key=lambda x: x[0]):
            print(r)
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if fail:
            raise SystemExit("failures: no summary")
    summarise()


if __name__ == "__main__":
    main()
