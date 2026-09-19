"""What each DRIFT ESTIMATOR does, end to end: trend, pre-subtraction residual, and final signal.

The head-to-head reports numbers; this shows the curves behind them, on the brain-masked spatial mean.

  A  RAW 470 with the trend each estimator SUBTRACTS drawn on it
  B  THE TRENDS ALONE, mean-removed -- shapes compared rather than offsets
  C  BEFORE the hemodynamic subtraction: detrended 470, i.e. what the drift removal leaves
  D  AFTER the hemodynamic subtraction: 470 - T*415, the signal every analysis actually reads
  E  ZOOM on the final stretch, around and after the animal stops

C AND D ARE DIFFERENT QUESTIONS, which is why both are drawn. C asks what the drift estimator left in
the functional channel. D asks what survives once the isosbestic is regressed out -- and because BOTH
channels are detrended with the same operator, an estimator can look poor in C and fine in D if what
it left behind was common to the two channels and the regression removed it anyway. Reading C alone
would overstate the differences between estimators.

Each trend is computed the way production would -- detrend the full (K, T) array, then project --
rather than detrending the already-projected trace. meegkit reweights per channel, so the two are not
the same operation and the cheap one would flatter the polynomial. `T` is the stock high-pass-fitted
transform, which for these hybrid variants IS the refit-T answer (equivalent to 1.5e-6).

    python -m scripts.rest_migration.plot_drift_estimators --sessions PS94_0819 PS94_0810 --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import config  # noqa: E402
from wfield_local.hemo_variants import FS, functional_channel, remove_drift  # noqa: E402

#: (label, variant, window, colour). `__rolling__`/`__linear__` dispatch to the local estimators.
#: 60 s is deliberately absent -- measured as the worst available, its ~110 s cutoff sitting inside
#: the 57-121 s position-block band.
ARMS = (("meegkit order 10 (production)", "meegkit_hpfit", None, "black"),
        ("WIN300 (kinked median)", "detrend_hpfit", 300.0, "tab:green"),
        ("ROLL300 (rolling median)", "__rolling__", 300.0, "tab:red"),
        ("LIN300 (rolling local-linear)", "__linear__", 300.0, "tab:blue"))


def _detrend(X, mask, variant, win):
    if variant in ("__rolling__", "__linear__"):
        from scripts.rest_migration.rolling_detrend import local_linear_detrend, rolling_detrend
        f = rolling_detrend if variant == "__rolling__" else local_linear_detrend
        return f(X, mask, win)
    kw = {"win_s": win} if win is not None else {}
    return remove_drift(np.array(X, copy=True), variant, mask, **kw)


def plot(label, outdir):
    from scripts.rest_migration.plot_session_residual import _behaviour_marks, _brain_mean_op
    from scripts.rest_migration.worktrunc_result_impact import _mask_for
    from wfield_local.filter_acausality_test import LP, _lp

    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    a0 = svt[:, functional_channel(s)::2].astype(np.float64)
    b0 = svt[:, (functional_channel(s) + 1) % 2::2].astype(np.float64)
    n = a0.shape[1]
    t = np.arange(n) / FS / 60.0
    mask = _mask_for(s, n)
    if mask is None:
        print(f"  !! {label}: no fit mask", flush=True)
        return
    T = np.load(res / "T.npy").astype(np.float64)
    u_mean, npix = _brain_mean_op(s["mc"])
    raw = u_mean @ a0
    # THE ISOSBESTIC, PROJECTED THE SAME WAY. It was loaded for the correction and never drawn,
    # so panel A showed the functional channel alone -- and the whole premise of the correction is
    # that the two channels share the slow component, which a reader cannot check without seeing
    # both. Same operator as the 470 so the comparison is like-for-like.
    raw_415 = u_mean @ b0
    _rf, last_min, _tm = _behaviour_marks(s, n)
    print(f"=== {label} ===  {t[-1]:.1f} min, mask keeps {100*mask.mean():.1f}%"
          + (f", last engaged {last_min:.1f} min" if last_min else ""), flush=True)

    trends, before, after = {}, {}, {}
    for name, variant, win, _c in ARMS:
        print(f"  {name} ...", flush=True)
        ad = _detrend(a0, mask, variant, win)
        bd = _detrend(b0, mask, variant, win)
        before[name] = u_mean @ ad
        trends[name] = raw - before[name]
        if LP < FS / 2:                       # violet lowpass, exactly as production
            bd = _lp(bd)
        az = (ad.T - np.nanmean(ad, 1)).T
        bz = (bd.T - np.nanmean(bd, 1)).T
        c = az - T @ bz
        after[name] = u_mean @ (c.T - np.nanmean(c, 1)).T

    fig, axes = plt.subplots(5, 1, figsize=(15, 16))

    def marks(ax):
        if last_min:
            ax.axvline(last_min, color="crimson", lw=1.2, ls="--", zorder=5)

    ax = axes[0]
    ax.plot(t, raw, lw=0.25, color="0.6", alpha=0.8, label="raw 470 (functional)")
    # BOTH CHANNELS ON ONE AXIS, not a twin. They are in the same units through the same operator,
    # and the point of drawing them together is that the SLOW COMPONENT IS SHARED -- which is the
    # premise the hemodynamic regression rests on. A twin axis would rescale one of them and make
    # a shared drift look like two unrelated curves.
    ax.plot(t, raw_415, lw=0.25, color="tab:purple", alpha=0.7, label="raw 415 (isosbestic)")
    for name, _v, _w, col in ARMS:
        ax.plot(t, trends[name], lw=2.0, color=col, label=name)
    ax.set_ylabel("spatial-mean SVD units")
    ax.set_title(f"{label}  A. RAW 470 + 415 and the trend each estimator SUBTRACTS  "
                 f"(fit mask keeps {100*mask.mean():.1f}% of frames, {npix} px)", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    marks(ax)
    print(f"  raw SD: 470 {np.std(raw):.5f}   415 {np.std(raw_415):.5f}   "
          f"corr(470,415) {np.corrcoef(raw, raw_415)[0, 1]:+.3f}", flush=True)

    ax = axes[1]
    for name, _v, _w, col in ARMS:
        ax.plot(t, trends[name] - trends[name].mean(), lw=1.8, color=col, label=name)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("trend (mean removed)")
    ax.set_title("B. THE TRENDS ALONE — shapes rather than offsets. Local estimators follow structure "
                 "the global polynomial arcs through", fontsize=10)
    marks(ax)

    ax = axes[2]
    for name, _v, _w, col in ARMS:
        ax.plot(t, before[name], lw=0.25, color=col, alpha=0.8, label=name)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("detrended 470")
    ax.set_title("C. BEFORE the hemodynamic subtraction — what the drift removal leaves in the "
                 "functional channel", fontsize=10)
    marks(ax)

    ax = axes[3]
    for name, _v, _w, col in ARMS:
        ax.plot(t, after[name], lw=0.25, color=col, alpha=0.8, label=name)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("corrected  (470 − T·415)")
    ax.set_title("D. AFTER the hemodynamic subtraction — THE SIGNAL EVERY ANALYSIS READS. Both "
                 "channels are detrended, so anything common to them is removed here regardless",
                 fontsize=10)
    marks(ax)

    ax = axes[4]
    z0 = max(0.0, (last_min - 12.0) if last_min else 0.7 * t[-1])
    sl = t >= z0
    ax.plot(t[sl], raw[sl], lw=0.4, color="0.6", alpha=0.7, label="raw 470")
    for name, _v, _w, col in ARMS:
        ax.plot(t[sl], trends[name][sl], lw=2.0, color=col, label=name)
    ax.set_xlabel("session time (min)")
    ax.set_ylabel("raw 470 + trends")
    ax.set_title(f"E. ZOOM from {z0:.0f} min — around and after the animal stops", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    marks(ax)

    fig.tight_layout()
    tag = "-".join(x[0].split()[0] for x in ARMS)
    p = Path(outdir) / f"drift_full_{label}__{tag}.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print(f"  -> {p}", flush=True)

    print(f"    {'arm':32s} {'trend SD':>10s} {'before SD':>10s} {'AFTER SD':>10s}", flush=True)
    for name, _v, _w, _c in ARMS:
        print(f"    {name:32s} {np.std(trends[name]):10.5f} {np.std(before[name]):10.5f} "
              f"{np.std(after[name]):10.5f}", flush=True)


def main():
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["PS94_0819", "PS94_0810"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=None, metavar="ARM",
                    choices=[x[0].split()[0] for x in ARMS],
                    help="which estimators to draw, by first word (meegkit WIN300 ROLL300 "
                         "LIN300); default all four. `--arms meegkit` gives the PRODUCTION chain "
                         "alone. That is a different question from the comparison: the four-arm "
                         "figure asks whether the estimator CHOICE matters, this one asks what "
                         "the pipeline actually does to a session, and the overlaid trends make "
                         "the second harder to read. The filename carries the arm set, so the "
                         "two can never overwrite each other.")
    a = ap.parse_args()
    if a.arms:
        keep = set(a.arms)
        ARMS = tuple(x for x in ARMS if x[0].split()[0] in keep)
        print(f"arms: {', '.join(x[0] for x in ARMS)}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for lab in a.sessions:
        try:
            plot(lab, a.out)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {ex}", flush=True)


if __name__ == "__main__":
    main()
