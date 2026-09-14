"""Overlay what each DRIFT ESTIMATOR actually subtracts: meegkit order 10 vs WIN600 vs WIN300.

The head-to-head reports numbers; this shows the curves behind them. For each session, the
brain-masked spatial mean of the raw 470 with all three fitted trends drawn on it, then the three
detrended outputs, then the trends alone (mean-removed, so their SHAPES can be compared rather than
their offsets), then a zoom on the final stretch.

Each trend is computed the way production would -- detrend the full (K, T) array, then project --
rather than detrending the already-projected trace. meegkit reweights per channel, so the two are not
the same operation and the cheap one would flatter the polynomial.

    python -m scripts.rest_migration.plot_drift_estimators --sessions PS94_0819 PS94_0810
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wfield_local import config  # noqa: E402
from wfield_local.hemo_variants import FS, FUNC, remove_drift  # noqa: E402

#: (label, variant, window, colour). Order 10 is production; the two windows bracket the ~8-12 min
#: "yellow line" timescale. 60 s is deliberately absent -- measured as the worst available, since its
#: ~110 s cutoff sits inside the 57-121 s position-block band.
ARMS = (("meegkit order 10", "meegkit_hpfit", None, "black"),
        ("WIN600 (median, ~19 min)", "detrend_hpfit", 600.0, "tab:orange"),
        ("WIN300 (median, ~9 min)", "detrend_hpfit", 300.0, "tab:green"))


def plot(label, outdir):
    from scripts.rest_migration.plot_session_residual import _behaviour_marks, _brain_mean_op
    from scripts.rest_migration.worktrunc_result_impact import _mask_for

    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    a = svt[:, FUNC::2].astype(np.float64)
    n = a.shape[1]
    t = np.arange(n) / FS / 60.0
    mask = _mask_for(s, n)
    if mask is None:
        print(f"  !! {label}: no fit mask", flush=True)
        return
    u_mean, npix = _brain_mean_op(s["mc"])
    raw = u_mean @ a
    _rf, last_min, _tm = _behaviour_marks(s, n)
    print(f"=== {label} ===  {t[-1]:.1f} min, mask keeps {100*mask.mean():.1f}%"
          + (f", last engaged {last_min:.1f} min" if last_min else ""), flush=True)

    trends, dets = {}, {}
    for name, variant, win, _c in ARMS:
        print(f"  {name} ...", flush=True)
        kw = {"win_s": win} if win is not None else {}
        det = remove_drift(a.copy(), variant, mask, **kw)
        dets[name] = u_mean @ det
        trends[name] = raw - dets[name]

    fig, axes = plt.subplots(4, 1, figsize=(15, 13))

    ax = axes[0]
    ax.plot(t, raw, lw=0.25, color="tab:blue", alpha=0.7, label="raw 470")
    for name, _v, _w, col in ARMS:
        ax.plot(t, trends[name], lw=2.0, color=col, label=name)
    ax.set_ylabel("spatial-mean SVD units")
    ax.set_title(f"{label}  A. RAW 470 and the trend each estimator SUBTRACTS "
                 f"(mask keeps {100*mask.mean():.1f}% of frames, {npix} px)", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    if last_min:
        ax.axvline(last_min, color="crimson", lw=1.2, ls="--")

    ax = axes[1]
    for name, _v, _w, col in ARMS:
        ax.plot(t, trends[name] - trends[name].mean(), lw=1.8, color=col, label=name)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("trend (mean removed)")
    ax.set_title("B. THE TRENDS ALONE, mean-removed — shapes compared rather than offsets. "
                 "A windowed median tracks local structure the global polynomial cannot reach",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    if last_min:
        ax.axvline(last_min, color="crimson", lw=1.2, ls="--")

    ax = axes[2]
    for name, _v, _w, col in ARMS:
        ax.plot(t, dets[name], lw=0.25, color=col, alpha=0.8, label=name)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("detrended 470")
    ax.set_title("C. WHAT IS LEFT after each — before the hemodynamic subtraction", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    if last_min:
        ax.axvline(last_min, color="crimson", lw=1.2, ls="--")

    ax = axes[3]
    z0 = max(0.0, (last_min - 12.0) if last_min else 0.7 * t[-1])
    sl = t >= z0
    ax.plot(t[sl], raw[sl], lw=0.4, color="tab:blue", alpha=0.6, label="raw 470")
    for name, _v, _w, col in ARMS:
        ax.plot(t[sl], trends[name][sl], lw=2.0, color=col, label=name)
    ax.set_xlabel("session time (min)")
    ax.set_ylabel("raw 470")
    ax.set_title(f"D. ZOOM from {z0:.0f} min — the stretch around and after the animal stops",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    if last_min:
        ax.axvline(last_min, color="crimson", lw=1.2, ls="--")

    fig.tight_layout()
    p = Path(outdir) / f"drift_estimators_{label}.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print(f"  -> {p}", flush=True)

    print("  trend SD (how much each removes) and pairwise shape difference:", flush=True)
    for name, _v, _w, _c in ARMS:
        print(f"    {name:26s} trend SD {np.std(trends[name]):.5f}   "
              f"residual SD {np.std(dets[name]):.5f}", flush=True)
    names = [x[0] for x in ARMS]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = trends[names[i]] - trends[names[j]]
            print(f"    {names[i][:12]:12s} vs {names[j][:12]:12s}  RMS diff {np.std(d):.5f}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["PS94_0819", "PS94_0810"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for lab in a.sessions:
        try:
            plot(lab, a.out)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {ex}", flush=True)


if __name__ == "__main__":
    main()
