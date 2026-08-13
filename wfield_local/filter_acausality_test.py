"""Is the pre-cue position code partly a ZERO-PHASE FILTER ECHO of the post-cue response?

wfield.hemodynamic_correction high-passes both channels at 0.1 Hz with scipy filtfilt -- forward AND
backward -- and the high-passed 470 channel IS what becomes SVTcorr. filtfilt's impulse response is
symmetric in time (measured: -0.496 before an impulse, -0.496 after), so a position-specific post-cue
response casts a scaled, SIGN-FLIPPED shadow over the seconds BEFORE the cue. A linear decoder is
indifferent to sign, so that shadow is decodable position information in a window where, biologically,
none of it should have come from the future.

THE TEST. Rebuild SVTcorr from the SAME session with three high-pass variants and re-run the SAME
decoder on each:

    zerophase   HP applied to both channels with filtfilt          <- the current pipeline
    causal      HP applied with lfilter (forward only)             <- the pre-cue window cannot
                                                                      contain post-cue information
    fitonly     no HP on the output at all                         <- the correction applied to
                                                                      unfiltered data

Only the FILTER differs: the hemodynamic transform T is loaded from disk and reused for all three, so
the coefficient fit is held fixed and the comparison isolates the filter. (T is a spatial mixing
matrix; refitting it per variant would confound two changes at once.)

HOW TO READ IT. POST-CUE is the control -- the real response is inside the window there, so it should
survive every variant. PRE-CUE is the test:
  * pre-cue holds up under `causal`  -> the maintained code is real; the echo is not what carries it.
  * pre-cue collapses under `causal` -> what we have been calling a maintained plan is substantially
    the post-cue response smeared backwards by the preprocessing, and the pre-stroke headline claim
    needs re-deriving from causally-filtered data.

Anything in between is a mixture, and the causal number is then the honest size of the real effect.
"""
from __future__ import annotations

import glob
import sys
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

FS, HP, LP, FUNC = 31.23, 0.1, 14.0, 1        # configs/defaults.yaml svd.*


def _hp(X, mode):
    if mode in ("fitonly", "quietdetrend", "taskdetrend"):
        return X
    b, a = butter(2, HP / (FS / 2), btype="highpass")
    return filtfilt(b, a, X, padlen=50) if mode == "zerophase" else lfilter(b, a, X)


# ---------------------------------------------------------------- quiet-masked detrending
TRIAL_PRE_S, TRIAL_POST_S, DETREND_WIN_S = 1.0, 4.0, 60.0


def fit_mask(s, n_frames, csmp, cue, pre_s=TRIAL_PRE_S, post_s=TRIAL_POST_S,
             from_strobe=True, require_quiet=True):
    """Frames usable for estimating slow drift: BEHAVIOURALLY QUIET **and** outside every trial.

    Both halves are necessary and neither alone is enough.

    * TRIAL-MASKED (van Driel et al. 2021): if the drift estimate sees the evoked responses, it
      partially tracks them, and subtracting it redistributes response energy into neighbouring time --
      the same displacement the zero-phase filter causes, just via a different route.
    * QUIET-ONLY (Priya, 2026-08-13): cortex-wide activity is dominated by movement (Musall et al.
      2019), so drift fitted across running or licking epochs is fitting movement, not drift. The quiet
      mask from ``behavior_events`` is already "not running AND not licking AND not peri-reward".

    The trap this avoids: the ENL is an ENFORCED no-lick period, so quiet bouts overlap the PRE-CUE
    window heavily. Masking on quiet alone would fit the drift through exactly the signal we are trying
    to measure. Hence quiet AND task-free.

    Returns (mask, diagnostics).
    """
    from wfield_local import behavior_events, config

    animal, mmdd = s["label"].split("_")
    ev = behavior_events.get_or_compute(config.resolver(), animal, f"2026{mmdd}")
    mask = np.ones(n_frames, bool)

    # 1. drop every trial window: strobe - pre_s  ->  cue + post_s
    cs, ss = np.asarray(cue["cue_samples"]), np.asarray(cue["strobe_samples"])
    sr = float(cue["sample_rate_hz"])
    j = np.searchsorted(ss, cs, side="right") - 1
    lead = np.where(j >= 0, (cs - ss[np.clip(j, 0, len(ss) - 1)]) / sr, 0.0)
    cue_f = np.searchsorted(csmp, cs)
    for f, ld in zip(cue_f, lead):
        a = int(f - ((ld if from_strobe else 0.0) + pre_s) * FS)
        b = int(f + post_s * FS)
        mask[max(0, a):min(n_frames, b)] = False
    after_trials = float(mask.mean())

    # 2. keep only behaviourally QUIET bouts
    if require_quiet and ev is not None and len(np.asarray(ev.get("quiet_starts", []))):
        q = np.zeros(n_frames, bool)
        fs_daq = float(ev["fs"])
        for a, b in zip(np.asarray(ev["quiet_starts"]), np.asarray(ev["quiet_stops"])):
            fa, fb = np.searchsorted(csmp, a / fs_daq * sr), np.searchsorted(csmp, b / fs_daq * sr)
            q[max(0, int(fa)):min(n_frames, int(fb))] = True
        mask &= q
        quiet_available = True
    else:
        quiet_available = not require_quiet
    return mask, {"frac_after_trial_mask": after_trials, "frac_final": float(mask.mean()),
                  "quiet_available": quiet_available}


def detrend_masked(X, mask, win_s=DETREND_WIN_S, min_frac=0.05):
    """Subtract a slow trend estimated ONLY from ``mask`` samples. X is (K, T).

    Windowed MEDIAN (robust to any transient that slips through the mask) over eligible samples, then
    linear interpolation across windows that had too few. No convolution over the data, so unlike a
    filter it has no impulse response and cannot displace signal in time -- which is the entire point.
    ``win_s`` sets how fast a drift can be tracked (~1/(2*win_s) Hz); 60 s is far slower than any task
    event but far faster than real bleaching drift.
    """
    K, T = X.shape
    w = max(1, int(round(win_s * FS)))
    cs, vs = [], []
    for a in range(0, T, w):
        b = min(T, a + w)
        m = mask[a:b]
        if m.sum() >= max(10, min_frac * (b - a)):
            cs.append(0.5 * (a + b))
            vs.append(np.median(X[:, a:b][:, m], axis=1))
    if len(cs) < 2:
        return X, 0.0
    C = np.asarray(cs, float)
    V = np.stack(vs, 1)
    t = np.arange(T, dtype=float)
    trend = np.stack([np.interp(t, C, V[k]) for k in range(K)])
    return X - trend, len(cs) / max(1, int(np.ceil(T / w)))


def _lp(X):
    b, a = butter(2, LP / (FS / 2), btype="lowpass")
    return filtfilt(b, a, X, padlen=50)


def svtcorr(svt, T, mode, mask=None, win_s=DETREND_WIN_S):
    """Rebuild SVTcorr with the chosen drift removal, reusing the saved transform T."""
    a = _hp(svt[:, FUNC::2].astype(np.float64), mode)
    b = _hp(svt[:, (FUNC + 1) % 2::2].astype(np.float64), mode)
    if mode in ("quietdetrend", "taskdetrend"):
        if mask is None:
            raise ValueError("quietdetrend needs a fit mask (see fit_mask)")
        ma = mask[:a.shape[1]] if mask.size >= a.shape[1] else np.pad(mask, (0, a.shape[1] - mask.size))
        mb = mask[:b.shape[1]] if mask.size >= b.shape[1] else np.pad(mask, (0, b.shape[1] - mask.size))
        a, _ = detrend_masked(a, ma, win_s)
        b, _ = detrend_masked(b, mb, win_s)
    if LP < FS / 2:
        b = _lp(b)
    a = (a.T - np.nanmean(a, 1)).T
    b = (b.T - np.nanmean(b, 1)).T
    c = a - T @ b
    return (c.T - np.nanmean(c, 1)).T.astype(np.float32)


def roi_signal(allen_dir, SVTc):
    """Allen-ROI features from a given SVTcorr -- the same construction as _build_signal(source='roi'),
    contracted per ROI so the (npix, T) movie is never materialized."""
    U = np.load(f"{allen_dir}/U_atlas.npy")
    atlas = np.load(f"{allen_dir}/allen_area_atlas_native_grid.npy").reshape(-1)
    mask = np.load(f"{allen_dir}/allen_brain_mask_native_grid.npy").astype(bool).reshape(-1)
    Uf = U.reshape(-1, U.shape[2])
    rois, regs = [], []
    for l in np.unique(atlas):
        if l == 0:
            continue
        pix = (atlas == l) & mask
        if pix.sum() < 20:
            continue
        rois.append(np.nanmean(Uf[pix], 0) @ SVTc)
        regs.append(int(l))
    return np.asarray(rois), np.asarray(regs)


def decode(s, sig, regs, align):
    args = SimpleNamespace(source="roi", align=align, baseline="none",
                           pre_s=1.0, post_s=2.0, fs=FS, max_rt=2.0)
    X, y, g, _, _, _ = _trial_features(s, args, signal=sig, feat_region=regs)
    ng = min(5, int(np.unique(g).size))
    if ng < 2 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    return float(accuracy_score(y, cross_val_predict(clf, X, y, cv=GroupKFold(ng), groups=g)))


def main():
    labs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["PS95_0810"]
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["zerophase", "causal", "fitonly"]
    out = {}
    for lab in labs:
        s = next((x for x in SESSIONS if x["label"] == lab), None)
        if s is None:
            print(f"  !! {lab}: not registered", flush=True)
            continue
        res = f"{s['mc']}/wfield_local_results"
        ad = glob.glob(f"{res}/allen_aligned_affine8v1")
        if not ad:
            print(f"  !! {lab}: no allen dir", flush=True)
            continue
        try:
            svt = np.load(f"{res}/SVT.npy")
            T = np.load(f"{res}/T.npy").astype(np.float64)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue
        print(f"\n=== {lab}  SVT{svt.shape}  T{T.shape}", flush=True)
        mask = None
        if "quietdetrend" in modes or "taskdetrend" in modes:
            from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
            from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
            from wfield_local.locanmf_crossanimal_dff import _frames as _fr
            cue = _lc(s["h5"])
            lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
            _cf, _lf, csmp = _fr(s, cue, lk)
            if csmp is None:
                print("  !! regime A (no corrected-frame map) -> skipping quietdetrend", flush=True)
                modes = [m for m in modes if m != "quietdetrend"]
            else:
                nfr = svt[:, FUNC::2].shape[1]
                quiet_req = "quietdetrend" in modes
                mask, diag = fit_mask(s, nfr, csmp, cue, pre_s=(1.0 if quiet_req else 0.5),
                                      from_strobe=quiet_req, require_quiet=quiet_req)
                print(f"  fit mask: {100*diag['frac_after_trial_mask']:.1f}% task-free -> "
                      f"{100*diag['frac_final']:.1f}% also quiet"
                      f"{'' if diag['quiet_available'] else '  (NO quiet events file -- task-mask only)'}",
                      flush=True)
        out[lab] = {}
        for mode in modes:
            sig, regs = roi_signal(ad[0], svtcorr(svt, T, mode, mask=mask))
            pre = decode(s, sig, regs, "precue")
            post = decode(s, sig, regs, "cue")
            out[lab][mode] = (pre, post)
            print(f"  {mode:12s}  PRE-CUE {pre:.3f}    post-cue {post:.3f}", flush=True)
            del sig

    if len(out) > 1:
        print("\n=== mean over sessions (chance 0.167) ===")
        for mode in modes:
            pre = np.nanmean([out[l][mode][0] for l in out])
            post = np.nanmean([out[l][mode][1] for l in out])
            print(f"  {mode:10s}  PRE-CUE {pre:.3f}    post-cue {post:.3f}")
        base = np.nanmean([out[l]["zerophase"][0] for l in out])
        for mode in [m for m in modes if m != "zerophase"]:
            v = np.nanmean([out[l][mode][0] for l in out])
            print(f"  pre-cue {mode} - zerophase: {v - base:+.3f}")


if __name__ == "__main__":
    main()
