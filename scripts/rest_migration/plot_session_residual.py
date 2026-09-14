"""Plot the whole-session signal BEFORE and AFTER drift correction, brain-masked spatial mean.

WHY. The order-10 robust polynomial is tuned for PHOTOBLEACHING -- a smooth, monotonic, session-scale
decay. A session in which the animal STOPS WORKING partway through poses a different shape: a
step-like change in the last stretch. Polynomials are GLOBAL basis functions, so a late step cannot be
absorbed locally; the fit either fails to follow it or rings across the whole record reaching for it.
This plots what actually happens, rather than reasoning about it.

Shows, on one time axis (session minutes):
  A  RAW 470 and 415 spatial-mean fluorescence, with the FITTED polynomial trend for each overlaid
  B  the DETRENDED traces -- what the drift removal leaves
  C  the corrected output actually analysed (meegkit_hpfit) against the stock zerophase product
  D  a zoom on the final stretch, where the animal stops
Rest frames and the last ENGAGED trial are marked throughout, so "where it stopped" is drawn from the
behaviour and not eyeballed from the trace.

    python -m scripts.rest_migration.plot_session_residual --sessions PS94_0903 PS92_0813
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import config  # noqa: E402
from wfield_local.hemo_variants import FS, FUNC, VARIANTS, remove_drift  # noqa: E402

VARIANT = "meegkit_hpfit"


def _brain_mean_op(mc):
    """``(u_mean, n_pixels)`` -- the operator that turns (K, T) into the brain-masked spatial mean."""
    import glob as _g

    res = Path(mc) / "wfield_local_results"
    # U_atlas, NOT U: the brain mask is on the Allen-REGISTERED grid (540x640), while `U.npy` is the
    # native camera grid (460x480). Indexing one with the other is an IndexError here -- but the same
    # confusion with two same-sized grids would silently average the wrong pixels. The decoders read
    # U_atlas for the same reason.
    hits = _g.glob(f"{res}/allen_aligned_affine8v1/allen_brain_mask_native_grid.npy")
    if not hits:
        raise FileNotFoundError("no Allen brain mask")
    m = np.load(hits[0]).astype(bool)
    U = np.load(f"{res}/allen_aligned_affine8v1/U_atlas.npy")
    if U.shape[:2] != m.shape:
        raise ValueError(f"U_atlas {U.shape[:2]} does not match brain mask {m.shape}")
    return np.nan_to_num(U)[m].mean(0).astype(np.float64), int(m.sum())


def _fit_mask_for(s, n_frames):
    """The SAME trial mask the production build uses, or None with a reason."""
    from wfield_local.filter_acausality_test import MASK_SPEC, fit_mask
    from wfield_local.locanmf_crossanimal_dff import _frames as _fr
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc

    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    _c, _l, csmp = _fr(s, cue, lk)
    if csmp is None:
        return None, None, "no corrected-frame map"
    mask, d = fit_mask(s, n_frames, csmp, cue, **MASK_SPEC[VARIANTS[VARIANT]["mask"]])
    return mask, d, None


def _behaviour_marks(s, n_frames):
    """``(rest_frames, last_engaged_min, trial_min)`` on the FRAME axis, or (None, None, None)."""
    from wfield_local.rest_by_position import _engaged_trials, _session_daq

    try:
        rest, cs, codes, ts, fs_samp, sync = _session_daq(s)
    except Exception as ex:                                          # noqa: BLE001
        print(f"    (no behaviour marks: {type(ex).__name__} {str(ex)[:60]})", flush=True)
        return None, None, None
    f_of = np.asarray(fs_samp)[:n_frames]
    rf = np.flatnonzero(rest[np.clip(f_of, 0, rest.size - 1)])
    eng = _engaged_trials(s, cs, codes)
    last = None
    if eng is not None and np.any(eng):
        last_sample = cs[np.flatnonzero(eng)[-1]]
        last = float(np.searchsorted(f_of, last_sample)) / FS / 60.0
    tmin = np.searchsorted(f_of, cs) / FS / 60.0
    return rf, last, tmin


def _panel_marks(ax, last_min, rest_frames, t, show_rest=True):
    if show_rest and rest_frames is not None and rest_frames.size:
        ax.plot(t[rest_frames], np.full(rest_frames.size, ax.get_ylim()[0]), ".",
                ms=0.4, color="0.55", zorder=0)
    if last_min is not None:
        ax.axvline(last_min, color="crimson", lw=1.2, ls="--", zorder=5)


def plot_session(label, outdir):
    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    print(f"=== {label} ===", flush=True)

    u_mean, npix = _brain_mean_op(s["mc"])
    svt = np.load(res / "SVT.npy")
    a = svt[:, FUNC::2].astype(np.float64)
    b = svt[:, (FUNC + 1) % 2::2].astype(np.float64)
    n = a.shape[1]
    t = np.arange(n) / FS / 60.0
    print(f"  {n} frames ({t[-1]:.1f} min), brain mask {npix} px", flush=True)

    mask, d, err = _fit_mask_for(s, n)
    if err:
        print(f"  !! {err}", flush=True)
        return None
    print(f"  drift-fit mask keeps {100*d['frac_final']:.1f}% of frames", flush=True)

    print("  detrending (meegkit order 10) ...", flush=True)
    a_det = remove_drift(a, VARIANT, mask, order=None)
    b_det = remove_drift(b, VARIANT, mask, order=None)

    raw470, raw415 = u_mean @ a, u_mean @ b
    det470, det415 = u_mean @ a_det, u_mean @ b_det
    trend470, trend415 = raw470 - det470, raw415 - det415

    corr = np.load(config.svtcorr_path(s["mc"]), mmap_mode="r")
    out_new = u_mean @ np.asarray(corr[:, :n], np.float64)
    bare = res / "SVTcorr.npy"
    out_old = (u_mean @ np.asarray(np.load(bare, mmap_mode="r")[:, :n], np.float64)
               if bare.exists() else None)

    rest_frames, last_min, trial_min = _behaviour_marks(s, n)
    stopped = last_min is not None and last_min < 0.95 * t[-1]
    if last_min is not None:
        print(f"  last ENGAGED trial at {last_min:.1f} min of {t[-1]:.1f} "
              f"({'STOPPED chunk present' if stopped else 'worked to the end'})", flush=True)

    fig, axes = plt.subplots(4, 1, figsize=(15, 13))

    ax = axes[0]
    ax.plot(t, raw470, lw=0.3, color="tab:blue", label="raw 470")
    ax.plot(t, raw415, lw=0.3, color="tab:purple", alpha=0.7, label="raw 415")
    ax.plot(t, trend470, lw=2.0, color="black", label="fitted polynomial (470)")
    ax.plot(t, trend415, lw=2.0, color="tab:orange", ls="--", label="fitted polynomial (415)")
    ax.set_ylabel("spatial-mean SVD units")
    ax.set_title(f"{label}  A. RAW fluorescence and the order-10 polynomial the drift removal "
                 f"subtracts  (mask keeps {100*d['frac_final']:.1f}% of frames)", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    _panel_marks(ax, last_min, rest_frames, t)

    ax = axes[1]
    ax.plot(t, det470, lw=0.3, color="tab:blue", label="470 detrended")
    ax.plot(t, det415, lw=0.3, color="tab:purple", alpha=0.7, label="415 detrended")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("detrended")
    ax.set_title("B. AFTER drift removal, before the hemodynamic subtraction — what the polynomial "
                 "leaves in each channel", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    _panel_marks(ax, last_min, rest_frames, t)

    ax = axes[2]
    ax.plot(t, out_new, lw=0.3, color="tab:green", label=f"ANALYSED ({VARIANT})")
    if out_old is not None:
        ax.plot(t, out_old, lw=0.3, color="tab:red", alpha=0.55, label="stock zerophase")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("corrected")
    ax.set_title("C. The CORRECTED signal every analysis reads (470 − T·415), against the stock "
                 "zero-phase product", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    _panel_marks(ax, last_min, rest_frames, t)

    # D. the final stretch, where the animal stops
    ax = axes[3]
    z0 = max(0.0, (last_min - 10.0) if last_min is not None else 0.75 * t[-1])
    sl = t >= z0
    ax.plot(t[sl], raw470[sl], lw=0.5, color="tab:blue", label="raw 470")
    ax.plot(t[sl], trend470[sl], lw=2.0, color="black", label="fitted polynomial")
    ax2 = ax.twinx()
    ax2.plot(t[sl], out_new[sl], lw=0.4, color="tab:green", label="corrected")
    ax2.set_ylabel("corrected", color="tab:green")
    ax.set_ylabel("raw 470")
    ax.set_xlabel("session time (min)")
    ax.set_title(f"D. ZOOM on the final stretch from {z0:.0f} min — does the polynomial follow the "
                 f"step, or ring reaching for it?", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    _panel_marks(ax, last_min, None, t, show_rest=False)

    if last_min is not None:
        axes[0].annotate("last ENGAGED trial", (last_min, axes[0].get_ylim()[1]),
                         xytext=(-4, -10), textcoords="offset points", ha="right",
                         fontsize=8, color="crimson")

    fig.tight_layout()
    p = Path(outdir) / f"session_residual_{label}.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print(f"  -> {p}", flush=True)

    # numbers, so the figure is not the only record
    def _tail_stats(x, name):
        mid = x[n // 4: 3 * n // 4]
        tail = x[int(0.90 * n):]
        print(f"    {name:22s} mid median {np.median(mid):+10.4g}   tail median "
              f"{np.median(tail):+10.4g}   tail-mid {np.median(tail)-np.median(mid):+10.4g} "
              f"({(np.median(tail)-np.median(mid))/(np.std(mid) or 1e-12):+.2f} SD)", flush=True)

    for arr, nm in ((raw470, "raw 470"), (trend470, "fitted trend 470"),
                    (det470, "detrended 470"), (out_new, "corrected (analysed)")):
        _tail_stats(arr, nm)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out) if a.out else Path.cwd() / "residual_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    for lab in a.sessions:
        try:
            plot_session(lab, outdir)
        except Exception as ex:                                       # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {ex}", flush=True)


if __name__ == "__main__":
    main()
