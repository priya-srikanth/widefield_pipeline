"""Hemodynamic correction under ALTERNATIVE drift-removal, written to LABELLED side-by-side outputs.

WHY THIS EXISTS. The pipeline's `SVTcorr.npy` is produced by `wfield.hemodynamic_correction`, which
removes slow drift with a ZERO-PHASE (acausal) 0.1 Hz Butterworth applied to both channels -- and the
filtered blue channel is what becomes the output. Measured over all 36 curated sessions, that inflates
PRE-CUE position decoding by ~0.21 (0.486 -> 0.247-0.273, lower in 35-36 of 36) while leaving POST-CUE
unchanged or better, because a zero-phase filter smears each post-cue response BACKWARDS in time. See
DECISIONS.md and :mod:`wfield_local.filter_acausality_test`.

NOTHING HERE OVERWRITES ANYTHING. The original `SVTcorr.npy` / `T.npy` / `rcoeffs.npy` stay exactly as
they are, and every variant is written to its OWN subdirectory beside them:

    <session>/motion_corrected/wfield_local_results/
        SVTcorr.npy                     <- ORIGINAL, untouched (variant "zerophase")
        T.npy  rcoeffs.npy              <- ORIGINAL, untouched
        hemo_strobedetrend_refitT/      <- one directory per variant, NEVER a bare file
            SVTcorr.npy  T.npy  rcoeffs.npy
            manifest.json               <- variant, params, refit_t, source signatures, timestamp

NAMING RULE (also stated in CLAUDE.md): a directory is `hemo_<variant>` , plus `_refitT` when the
hemodynamic coefficients were refitted on the drift-removed traces rather than reusing the saved `T`.
So `hemo_strobedetrend_refitT` and `hemo_strobedetrend` are different products and cannot be confused,
and anything reading a bare `SVTcorr.npy` is reading the original pipeline output by construction.
Every downstream result carries the variant string, so a figure can never silently mix two of them.

WHY refit_T MATTERS. `rcoeffs` are fitted from the 470-vs-415 regression AFTER drift removal, so
changing the drift removal changes what the fit sees. Reusing the saved `T` -- which the evaluation
harness does deliberately, to isolate one variable -- is right for a COMPARISON and wrong for a
PRODUCT: it applies a high-pass-derived transform to detrended traces. `refit_T` recomputes them
properly.

    python -m wfield_local.hemo_variants --sessions PS95_0810 --variant strobedetrend --write
    python -m wfield_local.hemo_variants --list
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

FS, HP, LP, FUNC = 31.23, 0.1, 14.0, 1          # configs/defaults.yaml svd.*

# Drift removal per variant. `mask` names the fit_mask spec in filter_acausality_test (None = no mask).
VARIANTS = {
    "zerophase":     dict(drift="filtfilt", mask=None,
                          note="the CURRENT pipeline: acausal 0.1 Hz high-pass applied to the output"),
    "causal":        dict(drift="lfilter", mask=None,
                          note="forward-only high-pass; no backward leakage but distorts phase"),
    "fitonly":       dict(drift="none", mask=None,
                          note="high-pass used for the coefficient fit only; drift left in the output"),
    "taskdetrend":   dict(drift="median", mask="taskdetrend",
                          note="masked detrend, evoked window only -- LEAVES 1.5 s of the pre-cue "
                               "window inside the drift fit, so it shaves real signal"),
    "strobedetrend": dict(drift="median", mask="strobedetrend",
                          note="masked detrend, whole trial masked -- the fit never sees the measured "
                               "window. Best measured variant"),
    "meegkit":       dict(drift="meegkit", mask="strobedetrend",
                          note="same mask, but de Cheveigne robust polynomial detrending via "
                               "meegkit.detrend (published implementation) instead of our median"),
    # HYBRIDS. The 415-residual check (hemo_residual_check) showed the 0.1 Hz high-pass is not only
    # doing harm: because rcoeffs is ONE scalar per pixel it can only be optimal for whichever band
    # dominates the fit, and the high-pass makes that the 0.1-14 Hz hemodynamic band. Detrending
    # leaves large slow variance in, so the coefficient gets tuned for the wrong band and vasomotion /
    # breathing are corrected WORSE (0.30/0.46 vs zerophase's 0.15/0.22). The fix is to separate the
    # two jobs the high-pass was doing: keep it for the FIT, where it belongs, and detrend the OUTPUT,
    # which is the only part that must not be smeared acausally.
    "detrend_hpfit": dict(drift="median", mask="strobedetrend", fit_drift="filtfilt",
                          note="HYBRID: rcoeffs fitted on 0.1 Hz high-passed traces (hemodynamic-band "
                               "optimal), correction applied to masked-median-detrended traces"),
    "meegkit_hpfit": dict(drift="meegkit", mask="strobedetrend", fit_drift="filtfilt",
                          note="HYBRID: rcoeffs fitted on 0.1 Hz high-passed traces, correction "
                               "applied to meegkit robust-polynomial-detrended traces"),
}

# The masked-median window. 60 s was the ORIGINAL default and is the single worst value available:
# measured, its 50% cutoff is ~110 s, sitting inside the 57-121 s position-block range, so it removed
# roughly half the amplitude of exactly the structure the blocks create. Only used by the median-based
# variants; kept at 600 s (50% cutoff ~19 min) for anything that still calls them.
DETREND_WIN_S = 600.0

# ADOPTED 2026-08-13 (Priya): order 10 for the shipped variant. Measured frequency response on a
# 150 min session -- amplitude RETAINED after detrending a pure sinusoid:
#
#   period       30s   60s  120s  300s  600s 1200s 3000s 9000s
#   order  5    1.00  1.00  1.00  1.00  0.99  0.98  0.91  0.01     50% cutoff ~90 min
#   order 10    1.00  1.00  1.00  0.99  0.98  0.94  0.18  0.00     50% cutoff ~43 min
#   order 40    1.00  1.00  1.00  0.99  0.96  0.87  0.22  0.00
#
# Order 10 removes more drift than order 5 at no measurable cost (pre-cue 0.378 vs 0.377, post-cue
# 0.777 vs 0.761 over 4 sessions) and still leaves the 1-2 min block band untouched (0.98-1.00). NB
# polynomials are GLOBAL basis functions, so raising the order adds oscillations across the whole
# record rather than local flexibility -- which is why order 40 is barely more aggressive than order 10,
# and why order is a much safer knob than a window length.
MEEGKIT_ORDER = 10


def _butter(mode):
    return butter(2, HP / (FS / 2), btype="highpass")


def remove_drift(X, variant, mask=None, win_s=DETREND_WIN_S, order=None):
    """Apply the variant's drift removal to (K, T) traces. Returns the cleaned array.

    ``order`` defaults to MEEGKIT_ORDER read AT CALL TIME, not bound as a default argument -- a sweep
    that set the module attribute between calls would otherwise silently keep using the value captured
    when this function was defined, and every "order" in the sweep would secretly be the same fit.
    """
    order = MEEGKIT_ORDER if order is None else int(order)
    d = VARIANTS[variant]["drift"]
    if d == "none":
        return X
    if d in ("filtfilt", "lfilter"):
        b, a = _butter(d)
        return filtfilt(b, a, X, padlen=50) if d == "filtfilt" else lfilter(b, a, X)
    if mask is None:
        raise ValueError(f"variant {variant!r} needs a fit mask")
    m = mask[:X.shape[1]] if mask.size >= X.shape[1] else np.pad(mask, (0, X.shape[1] - mask.size))
    if d == "median":
        from wfield_local.filter_acausality_test import detrend_masked
        return detrend_masked(X, m, win_s)[0]
    if d == "meegkit":
        from meegkit.detrend import detrend
        # meegkit works on (n_samples, n_channels) and wants weights of the SAME shape; masked
        # samples get weight 0 so the polynomial is fitted only where the mask allows.
        w = np.repeat(m.astype(float)[:, None], X.shape[0], axis=1)
        y, _w, _r = detrend(X.T.astype(np.float64), order=order, w=w)
        return np.asarray(y).T
    raise ValueError(f"unknown drift removal {d!r}")


def refit_T(U, a, b, chunk=20000):
    """Refit the hemodynamic coefficients on the DRIFT-REMOVED traces. Returns (T, rcoeffs).

    Same estimator as ``wfield.hemodynamic_correction`` -- per pixel, rcoeff = sum(a_i*b_i)/sum(b_i*b_i)
    where a_i = U_i @ SVTa and b_i = U_i @ SVTb -- but computed WITHOUT ever forming the (npix, T)
    movie, which for these sessions would be ~276 GB. Reassociating,

        sum_t (U_i . a[:,t]) (U_i . b[:,t])  ==  U_i @ (a @ b.T) @ U_i.T

    so the time axis contracts into two (K, K) Gram matrices first and the per-pixel step is O(npix*K^2)
    instead of O(npix*K*T). Exact, not an approximation -- the same reassociation used in
    ``joint_locanmf._operator`` and ``project_C_fixed_A``.
    """
    Uf = np.nan_to_num(np.asarray(U).reshape(-1, np.asarray(U).shape[-1])).astype(np.float64)
    Mab = np.asarray(a @ b.T, dtype=np.float64)
    Mbb = np.asarray(b @ b.T, dtype=np.float64)
    npix = Uf.shape[0]
    rc = np.empty(npix, dtype=np.float64)
    for i in range(0, npix, chunk):
        Ui = Uf[i:i + chunk]
        num = np.einsum("ik,ik->i", Ui @ Mab, Ui)
        den = np.einsum("ik,ik->i", Ui @ Mbb, Ui)
        rc[i:i + chunk] = np.where(den > 1e-30, num / np.maximum(den, 1e-30), 1e-10)
    rc[~np.isfinite(rc)] = 1e-10
    # T = pinv(U) @ (U * rc), via the (K,K) Gram so pinv(U) (a K x npix matrix) is never formed
    UtU = Uf.T @ Uf
    UtUrc = np.zeros_like(UtU)
    for i in range(0, npix, chunk):
        Ui = Uf[i:i + chunk]
        UtUrc += Ui.T @ (Ui * rc[i:i + chunk, None])
    return np.linalg.pinv(UtU) @ UtUrc, rc


def compute(session, variant, refit_t=True, win_s=DETREND_WIN_S, verbose=True, order=None):
    """(SVTcorr, T, rcoeffs, meta) for one session under one variant. Reads only; writes nothing."""
    from wfield_local.filter_acausality_test import MASK_SPEC, fit_mask
    from wfield_local.locanmf_crossanimal_dff import _frames as _fr
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc

    res = Path(session["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    a = svt[:, FUNC::2].astype(np.float64)
    b = svt[:, (FUNC + 1) % 2::2].astype(np.float64)

    spec = VARIANTS[variant]
    mask, mask_frac = None, None
    if spec["mask"]:
        cue = _lc(session["h5"])
        lk = _ll(session["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        _c, _l, csmp = _fr(session, cue, lk)
        if csmp is None:
            raise ValueError(f"{session['label']}: regime A has no corrected-frame map; "
                             f"variant {variant!r} needs one to build its mask")
        mask, d = fit_mask(session, a.shape[1], csmp, cue, **MASK_SPEC[spec["mask"]])
        mask_frac = d["frac_final"]

    def _finish(x, y):
        """violet lowpass + zero-mean, exactly as wfield.hemodynamic_correction does."""
        if LP < FS / 2:
            bb, aa = butter(2, LP / (FS / 2), btype="lowpass")
            y = filtfilt(bb, aa, y, padlen=50)
        return (x.T - np.nanmean(x, 1)).T, (y.T - np.nanmean(y, 1)).T

    fit_drift = spec.get("fit_drift")
    a_fit = b_fit = None
    if fit_drift:
        # Separate traces for the COEFFICIENT FIT. `remove_drift` keys off the variant, so build these
        # with the high-pass directly rather than inventing a pseudo-variant.
        bh, ah = butter(2, HP / (FS / 2), btype="highpass")
        a_fit, b_fit = _finish(filtfilt(bh, ah, a, padlen=50), filtfilt(bh, ah, b, padlen=50))

    a = remove_drift(a, variant, mask, win_s, order)
    b = remove_drift(b, variant, mask, win_s, order)
    a, b = _finish(a, b)

    if refit_t:
        U = np.load(res / "U.npy")
        # fit on the high-passed traces when the variant asks for it, else on the output traces
        T, rc = refit_T(U, a_fit if a_fit is not None else a, b_fit if b_fit is not None else b)
        del U
    elif fit_drift == "filtfilt":
        # The saved T IS the high-pass-fitted T: the original pipeline fitted rcoeffs on 0.1 Hz
        # high-passed traces, which is exactly what fit_drift="filtfilt" asks for. Verified -- refitting
        # on high-passed traces reproduces the saved rcoeffs and T to max abs diff 1.5e-6 -- and
        # local_wfield_summary.json confirms detrend_order=0, so there is no extra step to account for.
        # So reusing it is EQUIVALENT, not a shortcut, and it saves an 88 MB U.npy load plus the Gram
        # computation on every session. Production uses this path.
        T, rc = np.load(res / "T.npy").astype(np.float64), np.load(res / "rcoeffs.npy")
    elif fit_drift:
        raise ValueError(f"variant {variant!r} defines fit_drift={fit_drift!r}, which is NOT the "
                         f"pipeline's own high-pass, so the saved T does not correspond to it -- "
                         f"refit_t=True is required")
    else:
        T, rc = np.load(res / "T.npy").astype(np.float64), np.load(res / "rcoeffs.npy")

    c = a - T @ b
    c = (c.T - np.nanmean(c, 1)).T.astype(np.float32)
    meta = {"variant": variant, "refit_t": bool(refit_t), "label": session["label"],
            "drift": spec["drift"], "fit_drift": spec.get("fit_drift"),
            "mask": spec["mask"], "mask_frac": mask_frac,
            "win_s": win_s if spec["drift"] == "median" else None,
            "meegkit_order": (MEEGKIT_ORDER if order is None else int(order))
                             if spec["drift"] == "meegkit" else None,
            "fs": FS, "freq_highpass": HP, "freq_lowpass": LP, "functional_channel": FUNC,
            "note": spec["note"], "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if verbose:
        print(f"  {session['label']:12s} {variant}{'_refitT' if refit_t else ''}: SVTcorr{c.shape}"
              + (f"  mask {100*mask_frac:.1f}%" if mask_frac is not None else ""), flush=True)
    return c, T, rc, meta


def out_dir(session, variant, refit_t=True) -> Path:
    """`hemo_<variant>[_refitT]` beside the originals. NEVER a bare SVTcorr.npy."""
    return (Path(session["mc"]) / "wfield_local_results"
            / f"hemo_{variant}{'_refitT' if refit_t else ''}")


def write(session, variant, refit_t=True, win_s=DETREND_WIN_S, verbose=True) -> Path:
    """Compute and persist to the labelled subdirectory. Refuses to touch the original files."""
    from wfield_local import writeguard

    d = out_dir(session, variant, refit_t)
    writeguard.assert_writable(d)
    c, T, rc, meta = compute(session, variant, refit_t, win_s, verbose)
    tmp = d.with_name(d.name + f".{os.getpid()}.tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "SVTcorr.npy", c)
    np.save(tmp / "T.npy", T.astype(np.float32))
    np.save(tmp / "rcoeffs.npy", rc.astype(np.float32))
    (tmp / "manifest.json").write_text(json.dumps(meta, indent=2, default=float))
    if d.exists():                       # replacing a PREVIOUS run of the SAME variant, never the original
        import shutil
        shutil.rmtree(d)
    os.replace(tmp, d)
    if verbose:
        print(f"    wrote {d}", flush=True)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the variants and exit")
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--variant", default="strobedetrend", choices=sorted(VARIANTS))
    ap.add_argument("--no-refit-t", action="store_true",
                    help="reuse the saved T instead of refitting on the drift-removed traces "
                         "(correct for a controlled COMPARISON, wrong for a product)")
    ap.add_argument("--write", action="store_true", help="persist to the labelled subdirectory")
    args = ap.parse_args(argv)

    if args.list:
        for k, v in VARIANTS.items():
            print(f"  {k:15s} drift={v['drift']:9s} mask={str(v['mask']):14s} {v['note']}")
        return 0

    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    labs = args.sessions or []
    if not labs:
        ap.error("--sessions is required (or use --list)")
    rc = 0
    for lab in labs:
        s = next((x for x in SESSIONS if x["label"] == lab), None)
        if s is None:
            print(f"  !! {lab}: not registered", flush=True)
            rc = 1
            continue
        try:
            if args.write:
                write(s, args.variant, not args.no_refit_t)
            else:
                compute(s, args.variant, not args.no_refit_t)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
