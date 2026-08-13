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
    if mode == "fitonly":
        return X
    b, a = butter(2, HP / (FS / 2), btype="highpass")
    return filtfilt(b, a, X, padlen=50) if mode == "zerophase" else lfilter(b, a, X)


def _lp(X):
    b, a = butter(2, LP / (FS / 2), btype="lowpass")
    return filtfilt(b, a, X, padlen=50)


def svtcorr(svt, T, mode):
    """Rebuild SVTcorr with the chosen high-pass, reusing the saved transform T."""
    a = _hp(svt[:, FUNC::2].astype(np.float64), mode)
    b = _hp(svt[:, (FUNC + 1) % 2::2].astype(np.float64), mode)
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
        out[lab] = {}
        for mode in ("zerophase", "causal", "fitonly"):
            sig, regs = roi_signal(ad[0], svtcorr(svt, T, mode))
            pre = decode(s, sig, regs, "precue")
            post = decode(s, sig, regs, "cue")
            out[lab][mode] = (pre, post)
            print(f"  {mode:10s}  PRE-CUE {pre:.3f}    post-cue {post:.3f}", flush=True)
            del sig

    if len(out) > 1:
        print("\n=== mean over sessions (chance 0.167) ===")
        for mode in ("zerophase", "causal", "fitonly"):
            pre = np.nanmean([out[l][mode][0] for l in out])
            post = np.nanmean([out[l][mode][1] for l in out])
            print(f"  {mode:10s}  PRE-CUE {pre:.3f}    post-cue {post:.3f}")
        base = np.nanmean([out[l]["zerophase"][0] for l in out])
        for mode in ("causal", "fitonly"):
            v = np.nanmean([out[l][mode][0] for l in out])
            print(f"  pre-cue {mode} - zerophase: {v - base:+.3f}")


if __name__ == "__main__":
    main()
