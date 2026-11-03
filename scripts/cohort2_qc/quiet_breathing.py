"""T9 (breathing artefact): is the post-stroke rise of quiet-period orofacial 1-4 Hz power (peak ~2-3 Hz) breathing?
(cohort1 PS92-95 only, 2026-11-01; handoff STATUS_2026-11-01_QUIET_EVENTS_HANDOFF section 4, T9.)

2-3 Hz is the resting respiration rate of a mouse, so brain motion / blood with each breath could put power there.
FRAMES: exactly quiet_fluct's spectrum frames -- QUIET (quiet_spikes_v2.busy_frames: no DAQ contact lick / running bout
within 1.5 s before or after, no swapped imaging pair within 0.5 s), ENGAGED (before the quit), finite cortex traces,
AND finite face points (cam4 FaceRhythm own-session basis, QS_FR_BASIS=own). August sessions (pre = MMDD >= 0806).
SEGMENTS: every quiet run >= 4 s cut into 4-s Hann segments with 50 % overlap, linear detrend per segment (= the Welch
loop of quiet_fluct; each run contributes its segment count, i.e. the welch weighting); spectra / cross-spectra are
averaged over ALL segments of the session, so the coherence is the segment-pooled magnitude-squared coherence
(what scipy.signal.coherence gives on one long run), not an average of per-run coherences.

RESPIRATION PROXY. No FaceRhythm factor is a resting-breathing factor: the factors are spectrogram AMPLITUDES at 5 Hz
(no waveform at 2-3 Hz to be coherent with), and the nose / sniff factors in docs/FACE_RHYTHM_FINDINGS.md are task-locked
(sniff_prelick 4.1 Hz, nose_prelick_slow 1.3 Hz, whisk_run 14.7 Hz). So the proxy = the face-point PC (of the session's
own 20 point PCs, 25 Hz, interpolated to imaging frames) with the most prominent INTERIOR 1-5 Hz peak of its quiet-frame
displacement spectrum above the 1/f trend (choose_proxy), chosen per session WITHOUT looking at cortex. (A first rule,
max of the velocity spectrum in 1-5 Hz, landed on the 5 Hz band edge in most sessions -- not a peak -- and was
dropped.) Its peak frequency, prominence and peak-band displacement power (+-0.5 Hz) are the breathing read-outs (b).
Because a single PC may miss respiration, coh_max_pc = the best of all 20 PCs (upward-biased by the max). (A
frequency-domain CV regression on all 20 PCs at once was tried and dropped: unregularised, it overfit -- CV explained
fractions down to -3.6.)

MEASURES per session and trace (orofacial_left = ipsilesional, orofacial_right = contralesional, orofacial, limb, cortex):
 (a) coherence with the proxy vs frequency; NULL = the same with face segments randomly re-paired to cortex segments
     (mean of 5 permutations) = the bias floor, which depends on the segment count; coh_corr = coh - null.
 (c) band mean 1.5-4 Hz of coh_corr; coh_resp = the same AT the proxy's own peak (+-0.25 Hz); coh_max_pc.
 (d) pass 2 (extract_td): raw vs respiration-proxy-regressed 1-4 Hz power AND k = 4 event rate; the epoch effect with
     and without = how much of the increase survives.
 (e) frequency tracking across sessions within animal vs the proxy peak: the orofacial interior 1-5 Hz spectral peak
     (1/f removed) and ratio_peak_freq = the frequency of the largest gain over the animal's pre median spectrum.
Stats: mixed_models.lmm_table epoch model vs August pre (add_time_covariates(august_pre=True)), FDR within measure.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_breathing.py [--jobs 3] [--summarise]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QS_FR_BASIS", "own")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EP, columns, session_traces  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames  # noqa: E402

OUTB = V2 / "breathing"
COLS = ["orofacial_left", "orofacial_right", "orofacial", "limb", "cortex"]
SEG_S, N_PERM, BLOCK = 4.0, 5, 10
PEAK_BAND, COH_BAND, POW_BAND = (1.0, 5.0), (1.5, 4.0), (1.0, 4.0)
SHOWN = {"orofacial_left": "orofacial ipsilesional", "orofacial_right": "orofacial contralesional"}


def segments(m, nseg):
    """Start frames of 4-s, 50 %-overlap segments inside every run of m >= nseg (welch's segment placement)."""
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    st = []
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= nseg:
            st += list(range(a, b - nseg + 1, nseg // 2))
    return np.array(st, int)


def seg_fft(Y, starts, nseg):
    """(segments, channels, freqs) Hann-windowed rfft of linearly detrended segments of Y (channels x T)."""
    from scipy.signal import detrend
    from scipy.signal.windows import hann
    win = hann(nseg, sym=False)
    S = Y[:, starts[:, None] + np.arange(nseg)[None, :]].transpose(1, 0, 2)          # seg x ch x nseg
    S = detrend(S, axis=-1, type="linear") * win
    return np.fft.rfft(S, axis=-1), (win ** 2).sum()


def choose_proxy(fr, Pyy):
    """Respiration proxy = (point PC, peak Hz, prominence): the most prominent INTERIOR local maximum in 1-5 Hz of any
    point PC's displacement PSD after removing its 1/f trend (log-log line over 0.5-10 Hz; 3-bin smoothing; peaks
    searched over 0.75-6 Hz so a 1-5 Hz band edge cannot win). Cortex is not used."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    m, w = (fr >= 0.5) & (fr <= 10), (fr >= 0.75) & (fr <= 6)
    best = (0, np.nan, 0.0)
    for j, P in enumerate(Pyy):
        c = np.polyfit(np.log(fr[m]), np.log(P[m]), 1)
        flat = uniform_filter1d(np.log(P[w]) - np.polyval(c, np.log(fr[w])), 3)
        pk, pr = find_peaks(flat, prominence=0)
        f = fr[w][pk]
        keep = (f >= PEAK_BAND[0]) & (f <= PEAK_BAND[1])
        if keep.any():
            k = int(np.argmax(pr["prominences"][keep]))
            if pr["prominences"][keep][k] > best[2]:
                best = (j, float(f[keep][k]), float(pr["prominences"][keep][k]))
    return best


def extract(label):
    """One session -> spectra / coherence npz in OUTB/sessions (module-level for spawn)."""
    f = OUTB / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    from quiet_spikes_fr import own_series
    from wfield_local import face_rhythm_outputs as fro
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    busy, badw = busy_frames(label, ft, fs)
    cols = columns()
    X = Xr[[cols.index(c) for c in COLS]]
    _, _, tp, P = own_series(label)
    ok = np.isfinite(tp) & np.isfinite(P).all(1)
    tp, P = tp[ok].astype(float), P[ok].astype(float)
    o = np.argsort(tp, kind="stable")
    Pf = fro.to_frames(tp[o], P[o], ft, max_gap_s=0.2).T                          # 20 x T point PCs on imaging frames
    qe = ~busy & ~badw & engaged & np.isfinite(X).all(0) & np.isfinite(Pf).all(0)
    nseg = int(round(SEG_S * fs))
    st = segments(qe, nseg)
    if len(st) < 2 * BLOCK:
        raise RuntimeError(f"{label}: only {len(st)} quiet segments")
    FX, wss = seg_fft(X, st, nseg)
    FY, _ = seg_fft(Pf, st, nseg)
    fr = np.fft.rfftfreq(nseg, 1 / fs)
    scale = 2.0 / (fs * wss)                                                       # one-sided PSD density (welch)
    Pxx = (np.abs(FX) ** 2).mean(0) * scale
    Pyy = (np.abs(FY) ** 2).mean(0) * scale
    Sxy = np.einsum("sci,spi->cpi", FX, FY.conj()) / len(st) * scale
    coh_pc = np.abs(Sxy) ** 2 / (Pxx[:, None] * Pyy[None])                         # c x 20 x f
    rng = np.random.default_rng(int.from_bytes(label.encode(), "little") % (2 ** 32))
    null_pc = np.zeros_like(coh_pc)
    for _ in range(N_PERM):
        Sn = np.einsum("sci,spi->cpi", FX, FY[rng.permutation(len(st))].conj()) / len(st) * scale
        null_pc += np.abs(Sn) ** 2 / (Pxx[:, None] * Pyy[None]) / N_PERM
    # respiration proxy (cortex not used); summary recomputes it from Pyy with the same rule
    vel = Pyy * (2 * np.pi * fr) ** 2
    pk = (fr >= PEAK_BAND[0]) & (fr <= PEAK_BAND[1])
    j, fpk, prom = choose_proxy(fr, Pyy)
    pxx_cv = Pxx
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, freqs=fr, cols=np.array(COLS), Pxx=Pxx, Pyy=Pyy, coh_pc=coh_pc.astype(np.float32),
                        null_pc=null_pc.astype(np.float32), proxy=j, proxy_freq=fpk, proxy_prom=prom, vel=vel,
                        face_vel_1_5=float(vel[:, pk].sum(0).mean()),
                        pxx_cv=pxx_cv, n_seg=len(st), quiet_min=qe.sum() / fs / 60, fs=fs)
    return label


TD_COLS = ["orofacial_left", "orofacial_right", "orofacial", "limb"]
TD_LAGS_S = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
CLIP_Z = 5.0


def extract_td(label):
    """Pass 2 (Priya 11-01 priority update): TIME-DOMAIN correction. Regress the session's respiration proxy (the
    point PC chosen in pass 1) out of the 2-s-median-detrended traces: regressors = proxy displacement, its velocity and
    its square (2nd harmonic), each 2-s high-passed and robust-z clipped at +-5, at lags -0.3 .. +0.3 s (21 columns); out-of-fold ridge
    (quiet_spikes_fr.ridge_oof, 10 contiguous blocks, fitted on quiet engaged frames). Then on QUIET ENGAGED frames WITH
    face data: 1-4 Hz Welch power (4-s segments) of raw vs residual, robust SD of raw vs residual, and v2 events
    (quiet_spikes_v2.detect, positive and negative) on raw and residual; thresholds applied in the summary."""
    f = OUTB / "sessions_td" / f"{label}.npz"
    if f.exists():
        return label
    from quiet_area_spikes import mad
    from quiet_spikes_fr import _hp, own_series, ridge_oof
    from quiet_spikes_v2 import detect, detrend
    from wfield_local import face_rhythm_outputs as fro
    z1 = np.load(OUTB / "sessions" / f"{label}.npz")
    j = choose_proxy(z1["freqs"], z1["Pyy"])[0]
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    busy, badw = busy_frames(label, ft, fs)
    cols = columns()
    X = detrend(Xr[[cols.index(c) for c in TD_COLS]], fs)
    _, _, tp, P = own_series(label)
    ok = np.isfinite(tp) & np.isfinite(P).all(1)
    tp, P = tp[ok].astype(float), P[ok][:, j].astype(float)
    o = np.argsort(tp, kind="stable")
    p = fro.to_frames(tp[o], P[o][:, None], ft, max_gap_s=0.2)[:, 0]
    w = int(round(2.0 * fs)) | 1
    v = np.r_[np.nan, np.diff(p)] * fs
    def rclip(x):
        """Robust z-score clipped to +-CLIP_Z: face-point tracking glitches otherwise dominate the out-of-fold fit
        (first run: CV R2 down to -7 in 22 / 107 sessions, residual power ABOVE raw)."""
        med = np.nanmedian(x)
        s = 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-12
        return np.clip((x - med) / s, -CLIP_Z, CLIP_Z)
    ph = rclip(_hp(p, w))
    base = np.column_stack([ph, rclip(_hp(v, w)), rclip(_hp(ph ** 2, w))])
    cols_l = []
    for lag in TD_LAGS_S:
        k = int(round(lag * fs))
        Y = np.full_like(base, np.nan)
        if k >= 0:
            Y[:len(base) - k] = base[k:]
        else:
            Y[-k:] = base[:k]
        cols_l.append(Y)
    R = np.hstack(cols_l)
    okR = np.isfinite(R).all(1)
    quiet = ~busy & ~badw & np.isfinite(X).all(0) & okR
    qe = quiet & engaged
    T = X.shape[1]
    pred, _ = ridge_oof(R, X.T, qe, np.arange(T) * 10 // T)
    Res = X - np.nan_to_num(pred.T)
    r2 = 1 - ((X[:, qe] - pred.T[:, qe]) ** 2).sum(1) / ((X[:, qe] - X[:, qe].mean(1, keepdims=True)) ** 2).sum(1)
    nseg = int(round(SEG_S * fs))
    st = segments(qe, nseg)
    fr = np.fft.rfftfreq(nseg, 1 / fs)
    out = dict(fs=fs, cols=np.array(TD_COLS), r2=r2, n_qe=int(qe.sum()), n_seg=len(st), freqs=fr)
    for kind, Y in (("raw", X), ("res", Res)):
        F, wss = seg_fft(Y, st, nseg)
        out[f"psd_{kind}"] = (np.abs(F) ** 2).mean(0) * 2.0 / (fs * wss)
        out[f"sd_{kind}"] = mad(Y[:, qe])
        for sign, nm in ((1, "pos"), (-1, "neg")):
            L = []
            for i in range(len(TD_COLS)):
                pk, h = detect(sign * Y[i], quiet, fs)
                L += [(i, int(q), float(hh)) for q, hh in zip(pk, h) if engaged[q]]
            out[f"ev_{kind}_{nm}"] = np.array(L, float).reshape(-1, 3)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return label


def td_table():
    """Per session: 1-4 Hz power and k = 4 event rate (pos + neg, engaged quiet with face), raw vs proxy-regressed;
    threshold = 4 x the animal's August-pre robust SD of the SAME trace kind (raw or residual) on these frames."""
    import pandas as pd
    from wfield_local import epochs
    Z = {}
    for f in sorted((OUTB / "sessions_td").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or f.stem[:4] not in ("PS92", "PS93", "PS94", "PS95") or (ep == "pre" and f.stem[-4:] < "0806"):
            continue
        Z[f.stem] = (ep, dict(np.load(f)))
    sd = {}
    for lab, (ep, z) in Z.items():
        if ep == "pre":
            for kind in ("raw", "res"):
                sd.setdefault((lab[:4], kind), []).append(z[f"sd_{kind}"])
    sd = {k: np.median(v, 0) for k, v in sd.items()}
    rows = []
    for lab, (ep, z) in Z.items():
        fr = z["freqs"]
        r = dict(label=lab, animal=lab[:4], epoch=ep, qe_min=z["n_qe"] / z["fs"] / 60)
        for i, c in enumerate(TD_COLS):
            r[f"{c}|r2_proxy"] = float(z["r2"][i])
            for kind in ("raw", "res"):
                r[f"{c}|td_logP_{kind}"] = float(np.log10(band(fr, z[f"psd_{kind}"][i], POW_BAND)))
                thr = 4.0 * sd[(lab[:4], kind)][i]
                n = sum(int(((z[f"ev_{kind}_{nm}"][:, 0] == i) & (z[f"ev_{kind}_{nm}"][:, 2] > thr)).sum())
                        for nm in ("pos", "neg"))
                r[f"{c}|rate_{kind}"] = n / r["qe_min"]
                r[f"{c}|lograte_{kind}"] = float(np.log10(r[f"{c}|rate_{kind}"] + 0.2))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def cortical_peak(fr, P):
    """(Hz, prominence) of the most prominent INTERIOR 1-5 Hz local maximum of a PSD above its 1/f trend (same rule as
    choose_proxy; NaN if none). A plain argmax landed on 1.25 Hz in almost every session (a low-frequency bump the
    log-log line leaves), which says nothing about a 2-3 Hz component."""
    j, f, pr = choose_proxy(fr, P[None])
    return f, pr


def ratio_peak(fr, P, Pref):
    """Peak Hz (1-5 Hz) of log10(P / Pref), 3-bin smoothed: where this session's power GAINED most relative to the
    animal's pre-stroke median spectrum (leave-one-out for pre sessions) -- the frequency of the post-stroke rise."""
    from scipy.ndimage import uniform_filter1d
    r = uniform_filter1d(np.log10(P / Pref), 3)
    k = (fr >= PEAK_BAND[0]) & (fr <= PEAK_BAND[1])
    return float(fr[k][np.argmax(r[k])])


# ------------------------------------------------------------------------------------------------------- summary
def band(fr, y, b):
    m = (fr >= b[0]) & (fr <= b[1])
    return np.nanmean(y[..., m], -1)


def load():
    from wfield_local import epochs
    D = {}
    for f in sorted((OUTB / "sessions").glob("*.npz")):
        lab = f.stem
        ep = epochs.epoch_of(lab)
        if ep is None or lab[:4] not in ("PS92", "PS93", "PS94", "PS95") or (ep == "pre" and lab[-4:] < "0806"):
            continue
        z = dict(np.load(f))
        j, z["proxy_freq"], z["proxy_prom"] = choose_proxy(z["freqs"], z["Pyy"])   # recomputed: one rule for all
        z["proxy"] = j
        z["proxy_pow"] = float(z["Pyy"][j][np.abs(z["freqs"] - z["proxy_freq"]) <= 0.5].mean())
        z["coh"], z["null"] = z["coh_pc"][:, j], z["null_pc"][:, j]
        z["epoch"] = ep
        D[lab] = z
    return D


def session_table(D):
    import pandas as pd
    rows = []
    for lab, z in D.items():
        fr = z["freqs"]
        r = dict(label=lab, animal=lab[:4], epoch=z["epoch"], n_seg=int(z["n_seg"]), quiet_min=float(z["quiet_min"]),
                 proxy_pc=int(z["proxy"]), proxy_freq=float(z["proxy_freq"]), proxy_prom=float(z["proxy_prom"]),
                 log_proxy_pow=float(np.log10(z["proxy_pow"])), log_face_vel=float(np.log10(z["face_vel_1_5"])))
        for i, c in enumerate(COLS):
            r[f"{c}|coh"] = float(band(fr, z["coh"][i], COH_BAND))
            r[f"{c}|null"] = float(band(fr, z["null"][i], COH_BAND))
            r[f"{c}|coh_corr"] = r[f"{c}|coh"] - r[f"{c}|null"]
            r[f"{c}|coh_max_pc"] = float(band(fr, z["coh_pc"][i] - z["null_pc"][i], COH_BAND).max())
            r[f"{c}|logP"] = float(np.log10(band(fr, z["pxx_cv"][i], POW_BAND)))
            r[f"{c}|peak_freq"], r[f"{c}|peak_prom"] = cortical_peak(fr, z["Pxx"][i])
            ref = [y["Pxx"][i] for k2, y in D.items() if k2[:4] == lab[:4] and y["epoch"] == "pre" and k2 != lab]
            r[f"{c}|ratio_peak_freq"] = ratio_peak(fr, z["Pxx"][i], np.median(ref, 0)) if ref else np.nan
            near = np.abs(fr - r["proxy_freq"]) <= 0.25
            r[f"{c}|coh_resp"] = float(np.mean(z["coh"][i][near] - z["null"][i][near]))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


MEASURES = {"proxy_freq": "respiration-proxy peak (Hz)", "proxy_prom": "proxy peak prominence",
            "log_proxy_pow": "log10 proxy peak power", "log_face_vel": "log10 face velocity power 1-5 Hz"}
COL_MEASURES = {"coh_resp": "coherence with proxy at its peak frequency +-0.25 Hz (minus null)",
                "peak_freq": "cortical spectral peak 1-5 Hz (Hz, 1/f removed)", "coh_corr": "coherence with proxy 1.5-4 Hz (minus null)",
                "coh_max_pc": "max over 20 face PCs of coherence 1.5-4 Hz (minus null)",
                "ratio_peak_freq": "Hz of largest gain vs the animal's pre spectrum (1-5 Hz)",
                "logP": "log10 power 1-4 Hz"}
TD_MEASURES = {"td_logP_raw": "log10 1-4 Hz power (detrended, face frames)", "td_logP_res": "same, proxy regressed out",
               "lograte_raw": "log10(k=4 events/min + 0.2), raw", "lograte_res": "same, proxy regressed out",
               "r2_proxy": "CV R2 of the proxy model"}


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = [S[["label"]].assign(column="face", measure=m, y=S[m].to_numpy()) for m in MEASURES]
    L += [S[["label"]].assign(column=c, measure=m, y=S[f"{c}|{m}"].to_numpy()) for c in COLS for m in COL_MEASURES]
    L += [S[["label"]].assign(column=c, measure=m, y=S[f"{c}|{m}"].to_numpy()) for c in TD_COLS for m in TD_MEASURES
          if f"{c}|{m}" in S]
    L = mm.add_time_covariates(pd.concat(L), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch",), print_animals=False, fdr_by=("measure",))
    T.to_csv(OUTB / "lmm_breathing.csv", index=False)
    return T


def per_animal_tables(S):
    keys = ["proxy_freq", "proxy_prom", "log_proxy_pow", "log_face_vel"]
    keys += [f"orofacial|{m}" for m in COL_MEASURES] + ["orofacial|coh", "orofacial|null"]
    keys += [f"orofacial|{m}" for m in TD_MEASURES] + ["orofacial|rate_raw", "orofacial|rate_res"]
    keys += [f"{c}|{m}" for c in ("orofacial_left", "orofacial_right") for m in ("lograte_raw", "lograte_res")]
    keys += [f"{c}|{m}" for c in ("orofacial_left", "orofacial_right", "limb", "cortex")
             for m in ("coh_corr", "coh_resp", "coh_max_pc")]
    out = []
    for k in keys:
        T = S.pivot_table(index="animal", columns="epoch", values=k, aggfunc="mean").reindex(columns=EP)
        n = S.pivot_table(index="animal", columns="epoch", values=k, aggfunc="count").reindex(columns=EP)
        print(f"\n{k}  (mean over sessions; n sessions {n.fillna(0).astype(int).values.tolist()})\n", T.round(3).to_string())
        out.append(T.assign(measure=k).reset_index())
    import pandas as pd
    A = pd.concat(out)
    A.to_csv(OUTB / "breathing_animal_epoch.csv", index=False)
    # (d) how much of the acute (and later) increase in orofacial 1-4 Hz power remains after removal, per animal
    print("\n(d) orofacial post - pre per animal (log10 units): 1-4 Hz power and k=4 event rate, raw vs respiration "
          "proxy regressed out; frac_left = residual change / raw change")
    rows = []
    for an, G in S.groupby("animal"):
        pre = G[G.epoch == "pre"]
        for ep in EP[1:]:
            P = G[G.epoch == ep]
            r = dict(animal=an, epoch=ep)
            for m in ("td_logP_raw", "td_logP_res", "lograte_raw", "lograte_res"):
                r[m] = P[f"orofacial|{m}"].mean() - pre[f"orofacial|{m}"].mean()
            r["rate_ratio_raw"] = P["orofacial|rate_raw"].mean() / pre["orofacial|rate_raw"].mean()
            r["rate_ratio_res"] = P["orofacial|rate_res"].mean() / pre["orofacial|rate_res"].mean()
            r["frac_left_td_power"] = r["td_logP_res"] / r["td_logP_raw"]
            r["frac_left_td_rate"] = r["lograte_res"] / r["lograte_raw"]
            rows.append(r)
    R = pd.DataFrame(rows)
    print(R.round(3).to_string(index=False))
    R.to_csv(OUTB / "breathing_power_change_by_animal.csv", index=False)
    return A, R


def star(T, m, c, ep):
    t = T[(T.measure == m) & (T.column == c) & (T.term == f"C(epoch)[T.{ep}]")]
    if not len(t) or not t.q.iloc[0] < 0.05:
        return None
    q = t.q.iloc[0]
    return "***" if q < 0.001 else "**" if q < 0.01 else "*"


def tracking(S):
    """Does the cortical 1-5 Hz peak follow the respiration-proxy peak ACROSS SESSIONS WITHIN ANIMAL?"""
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr
    rows = []
    for c in ("orofacial", "orofacial_left", "orofacial_right"):
        for meas in ("peak_freq", "ratio_peak_freq"):
            col = f"{c}|{meas}"
            for an, G in S.groupby("animal"):
                G = G.dropna(subset=["proxy_freq", col])
                rho, p = spearmanr(G.proxy_freq, G[col])
                rows.append(dict(column=c, cortical=meas, animal=an, n=len(G), spearman_rho=rho, p=p,
                                 mean_abs_diff_hz=float(np.mean(np.abs(G.proxy_freq - G[col])))))
            D = S.rename(columns={col: "cpk"}).dropna(subset=["cpk", "proxy_freq"])
            try:
                r = smf.mixedlm("cpk ~ proxy_freq + C(epoch, Treatment('pre'))", D, groups=D.animal).fit(reml=True)
                rows.append(dict(column=c, cortical=meas, animal="LMM (animal intercept, + epoch)", n=len(D),
                                 slope_hz_per_hz=float(r.params["proxy_freq"]), se=float(r.bse["proxy_freq"]),
                                 p=float(r.pvalues["proxy_freq"])))
            except Exception as e:  # noqa: BLE001
                print("tracking LMM failed", c, meas, e)
    R = pd.DataFrame(rows)
    print("\nFREQUENCY TRACKING: cortical peak (interior 1-5 Hz peak above 1/f; or Hz of largest gain over the animal's "
          "pre) vs respiration-proxy peak, across sessions within animal\n", R.round(3).to_string(index=False))
    R.to_csv(OUTB / "breathing_frequency_tracking.csv", index=False)
    return R


def figure(D, S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    import textwrap

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    fig, ax = plt.subplots(3, 4, figsize=(18, 15.5))

    def wrap(t):
        return "\n".join(textwrap.fill(x, 42) for x in t.split("\n"))

    animals = sorted({k[:4] for k in D})
    fr = next(iter(D.values()))["freqs"]
    oi = COLS.index("orofacial")
    sel = (fr >= 0.5) & (fr <= 8)

    def spectra(a, key, ylabel, title, null=False):
        for ep in ("pre", "acute"):
            per = []
            for an in animals:
                Z = [z for k, z in D.items() if k.startswith(an) and z["epoch"] == ep]
                if not Z:
                    continue
                if key == "vel":
                    y = np.mean([z["vel"][z["proxy"]] / np.median(z["vel"][z["proxy"]][(fr >= 0.5) & (fr <= 10)])
                                 for z in Z], 0)
                else:
                    y = np.mean([z[key][oi] for z in Z], 0)
                per.append(y)
                a.plot(fr[sel], y[sel], color=EPOCH_LINE[ep], lw=0.7, alpha=0.6, marker=ANIMAL_MARKER[an],
                       markevery=4, ms=3.5, mfc="none")
            a.plot(fr[sel], np.mean(per, 0)[sel], color=EPOCH_LINE[ep], lw=2.4, label=f"{ep} (mean of animals)")
            if null:
                nl = np.mean([np.mean([z["null"][oi] for k, z in D.items() if k.startswith(an) and z["epoch"] == ep], 0)
                              for an in animals], 0)
                a.plot(fr[sel], nl[sel], color=EPOCH_LINE[ep], lw=1.2, ls="--", label=f"{ep} null (re-paired)")
        a.axvspan(*COH_BAND, color="0.92", zorder=0)
        a.set_xlabel("Hz")
        a.set_ylabel(ylabel, fontsize=8)
        a.set_title(wrap(title), fontsize=9)
        a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)

    spectra(ax[0, 0], "coh", "magnitude-squared coherence", "(a) orofacial vs respiration proxy: coherence", null=True)
    spectra(ax[0, 1], "vel", "velocity PSD / median 0.5-10 Hz", "(b) respiration-proxy velocity spectrum (chosen face-point PC, normalised)")
    spectra(ax[0, 2], "Pxx", "PSD (dF/F %)^2 / Hz", "orofacial quiet power spectrum (reference)")
    ax[0, 2].set_yscale("log")
    a = ax[0, 3]
    jit = np.random.default_rng(1)
    for (an, ep), G in S.groupby(["animal", "epoch"]):
        a.scatter(G.proxy_freq + jit.uniform(-0.06, 0.06, len(G)),
                  G["orofacial|ratio_peak_freq"] + jit.uniform(-0.06, 0.06, len(G)),
                  marker=ANIMAL_MARKER[an], facecolors="none", edgecolors=EPOCH_LINE[ep], s=22, lw=0.9)
    a.plot([1, 5], [1, 5], color="0.6", ls=":", lw=1)
    a.set_xlabel("respiration-proxy peak (Hz)")
    a.set_ylabel("orofacial: Hz of largest power gain vs the animal's pre (1-5 Hz)", fontsize=8)
    a.set_title(wrap("(e) frequency tracking: does the frequency of the cortical gain follow breathing? (sessions, "
                     "jittered; dotted = identity)"), fontsize=9)

    def markers(a, col, title, mm, cc):
        rng = np.random.default_rng(0)
        for x, ep in enumerate(EP):
            B = S[S.epoch == ep]
            for an, A in B.groupby("animal"):
                a.scatter(x + rng.uniform(-0.18, 0.18, len(A)), A[col], marker=ANIMAL_MARKER[an], facecolors="none",
                          edgecolors=EPOCH_LINE[ep], s=18, lw=0.8, alpha=0.8)
            ci = boot_ci({an: list(A[col].dropna()) for an, A in B.groupby("animal") if A[col].notna().any()},
                         np.random.default_rng(0))
            if ci is not None:
                a.errorbar(x + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="o",
                           color=EPOCH_LINE[ep], ms=6, capsize=3, lw=1.6, zorder=5)
            s = star(T, mm, cc, ep) if x > 0 else None
            if s:
                sig_marks.star_top(a, x, s, color=EPOCH_LINE[ep])
        a.set_xticks(range(len(EP)), EP, fontsize=8)
        a.set_title(wrap(title.replace("\n", " ")), fontsize=9)

    markers(ax[1, 0], "proxy_freq", "(b) respiration-proxy peak frequency (Hz)", "proxy_freq", "face")
    markers(ax[1, 1], "orofacial|coh_resp", "(c) orofacial coherence AT the respiration peak\n(+-0.25 Hz, minus null)",
            "coh_resp", "orofacial")
    markers(ax[1, 2], "orofacial|coh_corr", "(c) orofacial coherence with proxy, 1.5-4 Hz\n(minus null)",
            "coh_corr", "orofacial")
    markers(ax[1, 3], "orofacial|coh_max_pc", "(c) BEST of the 20 face-point PCs: orofacial coherence 1.5-4 Hz "
            "(minus null; upward-biased by the max)", "coh_max_pc", "orofacial")
    for a in ax[1, 1:]:
        a.axhline(0, color="0.5", lw=0.8)
    markers(ax[2, 0], "orofacial|td_logP_raw", "(d) orofacial log10 1-4 Hz power, raw", "td_logP_raw", "orofacial")
    markers(ax[2, 1], "orofacial|td_logP_res", "(d) orofacial log10 1-4 Hz power,\nrespiration proxy regressed out",
            "td_logP_res", "orofacial")
    markers(ax[2, 2], "orofacial|lograte_raw", "(d) orofacial k=4 events, log10(/min + 0.2), raw", "lograte_raw",
            "orofacial")
    markers(ax[2, 3], "orofacial|lograte_res", "(d) orofacial k=4 events, log10(/min + 0.2),\nproxy regressed out",
            "lograte_res", "orofacial")
    for pair in ((ax[2, 0], ax[2, 1]), (ax[2, 2], ax[2, 3])):
        lo, hi = min(p.get_ylim()[0] for p in pair), max(p.get_ylim()[1] for p in pair)
        for p in pair:
            p.set_ylim(lo, hi)
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=4, frameon=False, title="animal (marker shape)",
               title_fontsize=8, bbox_to_anchor=(0.5, 0.0))
    sig_marks.reserve_headroom(fig)

    def coef(m, c, ep="acute"):
        t = T[(T.measure == m) & (T.column == c) & (T.term == f"C(epoch)[T.{ep}]")]
        return float(t.coef.iloc[0]) if len(t) else np.nan

    pr, pres = coef("td_logP_raw", "orofacial"), coef("td_logP_res", "orofacial")
    rr, rres = coef("lograte_raw", "orofacial"), coef("lograte_res", "orofacial")
    paras = [
        "T9 breathing test: is the post-stroke rise in quiet orofacial 1-4 Hz activity breathing? (cohort1 PS92-95; "
        "quiet engaged frames with face video, 4-s segments)",
        f"Acute (LMM vs August pre): 1-4 Hz power {pr:+.2f} log10 raw vs {pres:+.2f} with the respiration proxy "
        f"regressed out; k=4 event rate {rr:+.2f} vs {rres:+.2f} log10. Coherence with the proxy is low (< 0.1) and "
        "does not rise after the stroke. Proxy = per-session face-point PC with the most prominent interior 1-5 Hz "
        "spectral peak above 1/f (no FaceRhythm factor is a resting-breathing factor).",
        "Top: thin = animals (shape), thick = mean of animals, grey band = 1.5-4 Hz; dashed = coherence floor from "
        "randomly re-paired segments. Lower rows: open markers = sessions (shape = animal, colour = epoch); filled = "
        "pooled mean, nested animal -> session bootstrap 95% CI; stars = LMM epoch vs August pre, FDR q.",
    ]
    fig.suptitle("\n".join(textwrap.fill(t, 190) for t in paras), fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.935), w_pad=2.5, h_pad=2.0)
    fig.savefig(OUTB / "quiet_breathing.png", dpi=110)
    print("wrote", OUTB / "quiet_breathing.png")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 240)
    D = load()
    S = session_table(D)
    S = S.merge(td_table().drop(columns=["animal", "epoch"]), on="label", how="left")
    S.to_csv(OUTB / "breathing_sessions.csv", index=False)
    print(f"{len(S)} sessions; segments median {S.n_seg.median():.0f} (min {S.n_seg.min()}); "
          f"time-domain pass {S['orofacial|rate_raw'].notna().sum()}")
    print("proxy PC index chosen (count):", S.proxy_pc.value_counts().sort_index().to_dict())
    per_animal_tables(S)                       # rule 8: animals before any pooled test
    tracking(S)
    T = stats(S)
    show = T[(T.term != "pre_level")]
    print("\nLMM epoch model vs August pre:")
    print(show[["measure", "column", "term", "coef", "se", "p", "q", "re", "n"]].round(4).to_string(index=False))
    figure(D, S, T)
    (OUTB / "README.txt").write_text(__doc__ + "\n" + (extract_td.__doc__ or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = a.only or sorted(f.stem for f in (V2 / "sessions").glob("*.npz")
                                  if f.stem[:4] in ("PS92", "PS93", "PS94", "PS95") and f.stem[-4:] >= "0806")
        for worker in (extract, extract_td):           # pass 2 needs pass 1's proxy choice
            res, fail = fan_sessions(labels, worker, jobs=min(a.jobs, 5), label="session")
            print(f"{worker.__name__}: {len(res)} done, {len(fail)} failed")
            for x in fail:
                print("FAILED", x)
        if a.only:
            return
    summarise()


if __name__ == "__main__":
    main()
