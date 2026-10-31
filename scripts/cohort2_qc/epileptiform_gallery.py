"""Are the epileptiform_scan candidates PS122-like discharges or ordinary lick responses? (2026-10-31)

epileptiform_scan.py's strict set (amp >= 5%, half-width 60-250 ms, 415 flat) is dominated by cohort1 PS93 after
the stroke (20-36 per session from 08-20) plus PS92 08-26 / 08-28, while GCaMP6s lick responses can also reach
5-8% in the cortex mean. Per session this shows the three things that separate them:
  * waveform: mean 470 / 415 around the peak (PS122: one-pair rise, ~130 ms, then a ~2-s TWO-channel undershoot);
  * peak map: global and bilateral (PS122) vs a lick / orofacial map;
  * lick locking: share of events with a DAQ lick in the 0.5 s BEFORE the peak, against the same share at random
    times in the session (a lick response is lick-locked; PS122's events came after licking had stopped).
Plus three single events with licks marked.

    conda activate locanmf
    python scripts/cohort2_qc/epileptiform_gallery.py [LABEL ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

LABCAMS = Path("M:/MICROSCOPE/Priya/Widefield/labcams")
SCAN = LABCAMS / "cohort2" / "_tmp_qc_figures_20261031" / "epileptiform_scan"
DAQ2 = Path("M:/MICROSCOPE/Priya/Widefield/DAQ_recorder_output")
FS = 31.23
DEFAULT = ["PS122_1029", "PS93_0826", "PS93_0820", "PS93_0828", "PS92_0828", "PS93_0904", "PS94_0825", "PS95_0827"]
H5_C2 = {"PS122_1029": "20261029/PS122_20261029_192406.h5", "PS118_1029": "20261029/PS118_20261029_114909.h5",
         "PS120_1029": "20261029/PS120_20261029_135728.h5", "PS121_1029": "20261029/PS121_20261029_163740.h5"}


def strict(r):
    a, w = np.array(r["amp"]), np.array(r["halfwidth_ms"])
    return np.array(r["pairs"], int)[(a >= 5) & (w >= 60) & (w <= 250)]


def licks_for(r):
    from wfield_local import config
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    if r["cohort"] == "cohort1":
        s = next(s for s in config.load_sessions(machine="analysis_desktop") if s["label"] == r["label"])
    else:
        if r["label"] not in H5_C2:
            return np.array([], int)
        s = dict(label=r["label"], mc=str(Path(r["res"]).parent), h5=str(DAQ2 / H5_C2[r["label"]]), regime="B",
                 fmdir=None)
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    _, lf, _ = _frames(s, cue, lk)
    return np.sort(np.asarray(lf, int))


def one(r):
    res = Path(r["res"])
    func = r["func"]
    fa = np.load(res / ("frames_average_raw.npy" if (res / "frames_average_raw.npy").exists() else "frames_average.npy"))
    br = brain_mask(fa, func, res=res)
    U = np.load(res / "U.npy", mmap_mode="r")
    Ub = np.nan_to_num(np.asarray(U).reshape(-1, U.shape[-1])[br.reshape(-1)]).astype(np.float32)
    del U
    um = Ub.mean(0).astype(np.float64)
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    S = np.asarray(SVT[:, func::2], np.float32)
    x = 100 * (um @ S)
    y = 100 * (um @ np.asarray(SVT[:, 1 - func::2], np.float64))
    n = min(len(x), len(y))
    ev = strict(r)
    ev = ev[(ev > 70) & (ev < n - 70)]
    lags = np.arange(-31, 63)
    b = lambda v, e: v[e[:, None] + lags] - v[e[:, None] + np.arange(-8, -2)].mean(1, keepdims=True)  # noqa: E731
    m = Ub @ (S[:, ev].mean(1) - S[:, ev[:, None] + np.arange(-8, -2)].mean((1, 2)))
    img = np.full(br.size, np.nan, np.float32)
    img[br.reshape(-1)] = 100 * m
    lk = licks_for(r)
    rng = np.random.default_rng(0)
    rand = rng.integers(70, n - 70, 2000)

    def lick_before(e):
        if not len(lk):
            return np.nan
        k = np.searchsorted(lk, e)
        prev = np.where(k > 0, lk[np.maximum(k - 1, 0)], -10 ** 9)
        return float(np.mean((e - prev) <= int(0.5 * FS)))
    ex = ev[np.linspace(0, len(ev) - 1, min(3, len(ev))).astype(int)] if len(ev) else ev
    return dict(label=r["label"], n=len(ev), w470=b(x, ev).mean(0), w415=b(y, ev).mean(0), map=img.reshape(br.shape),
                lick_ev=lick_before(ev), lick_rand=lick_before(rand), n_licks=len(lk),
                ex=[(int(e), x[e - 94:e + 94] - x[e - 8:e - 2].mean(), y[e - 94:e + 94] - y[e - 8:e - 2].mean(),
                     lk[(lk >= e - 94) & (lk < e + 94)] - e) for e in ex])


def main():
    labels = sys.argv[1:] or DEFAULT
    R = {r["label"]: r for r in json.loads((SCAN / "epileptiform_scan.json").read_text())}
    from wfield_local.parallel import fan_out
    res, fail = fan_out([R[k] for k in labels], one, jobs=4, label="session")
    if fail:
        print("FAILED", fail)
    G = {g["label"]: g for _, g in res}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = [k for k in labels if k in G]
    fig, ax = plt.subplots(len(L), 5, figsize=(21, 3.4 * len(L)), gridspec_kw=dict(width_ratios=[1.3, 1, 1.3, 1.3, 1.3]))
    t = np.arange(-31, 63) / FS * 1000
    for i, k in enumerate(L):
        g = G[k]
        ax[i, 0].plot(t, g["w470"], color="tab:green", label="470")
        ax[i, 0].plot(t, g["w415"], color="tab:purple", label="415")
        ax[i, 0].axhline(0, color="k", lw=0.4)
        ax[i, 0].axvline(0, color="k", lw=0.4)
        ax[i, 0].set_title(f"{k}: {g['n']} events, mean around the peak", fontsize=9)
        ax[i, 0].set_xlabel("ms")
        ax[i, 0].legend(fontsize=7)
        lim = np.nanpercentile(np.abs(g["map"]), 99)
        ax[i, 1].imshow(g["map"], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[i, 1].set_title(f"peak map ±{lim:.1f}%\nlick in 0.5 s before: events {g['lick_ev']:.2f} vs random "
                           f"{g['lick_rand']:.2f}", fontsize=8)
        ax[i, 1].axis("off")
        for j in range(3):
            a_ = ax[i, 2 + j]
            if j < len(g["ex"]):
                e, xx, yy, ll = g["ex"][j]
                tt = np.arange(-94, 94) / FS
                a_.plot(tt, xx, color="tab:green", lw=0.8)
                a_.plot(tt, yy, color="tab:purple", lw=0.8)
                for f in ll:
                    a_.axvline(f / FS, ymin=0, ymax=0.08, color="k", lw=0.8)
                a_.set_title(f"event at {e / FS / 60:.2f} min (ticks = licks)", fontsize=8)
                a_.set_xlabel("s")
            else:
                a_.axis("off")
    fig.suptitle("epileptiform candidates (amp >= 5%, half-width 60-250 ms, 415 flat): PS122 10-29 (top) vs the "
                 "cohort1 sessions with most candidates", fontsize=12)
    fig.tight_layout()
    out = SCAN / "epileptiform_gallery.png"
    fig.savefig(out, dpi=75)
    print("wrote", out)
    for k in L:
        g = G[k]
        print(f"{k:12s} n {g['n']:3d}  lick-before events {g['lick_ev']:.2f} random {g['lick_rand']:.2f}  "
              f"470 peak {g['w470'][31]:.1f}  415 at peak {g['w415'][31]:+.2f}  470 at +1.3 s {g['w470'][31 + 40]:+.1f}"
              f"  415 at +1.3 s {g['w415'][31 + 40]:+.2f}")


if __name__ == "__main__":
    main()
