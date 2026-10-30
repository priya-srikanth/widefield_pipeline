"""Figure for the 8s-vs-6s comparison (QC doc §0j): first-lick traces, maps, spectra, spatial correlation.

Rows: (1) first-lick cortex-mean trace, dF/F %; (2) the same normalised to its own peak (kinetics); (3) hotspot
trace (top-2% pixels of the 0-0.4 s map), normalised; (4) power spectrum of the corrected cortex mean, normalised
to its 0.1-0.3 Hz level; (5) spontaneous pixel-pair correlation vs distance (from indicator_compare.json, PS120
recomputed here). Then one first-lick map (0-0.4 s) per session. PS120 is drawn as the blood-flow-only reference
(no GCaMP signal, §0h). Same signal / mask / events as indicator_compare.py.

    conda activate locanmf
    python scripts/cohort2_qc/indicator_figure.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from indicator_compare import OUT, VAR, analyse, first_licks  # noqa: E402
from swapmask_1029 import DAQ, ROOT, SESS  # noqa: E402

FS = 31.23
STYLE = {"PS118_1029": ("tab:blue", "-", "PS118 Thy1-6s"), "PS92_0811": ("tab:blue", "--", "PS92_0811 Thy1-6s (c1)"),
         "PS93_0814": ("tab:blue", ":", "PS93_0814 Thy1-6s (c1)"), "PS121_1029": ("tab:red", "-", "PS121 RiboL1-8s"),
         "PS122_1029": ("tab:red", "--", "PS122 RiboL1-8s"), "PS120_1029": ("0.5", "-", "PS120 (no GCaMP: blood flow only)")}


def traces(s, func):
    from wfield_local import hemo_variants as hv
    res = Path(s["mc"]) / "wfield_local_results"
    fa = np.load(res / "frames_average.npy")
    H, W = fa.shape[1:]
    br = brain_mask(fa, func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    del U
    C = np.load(res / VAR / "SVTcorr.npy").astype(np.float64)
    n = C.shape[1]
    bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
    C = hv._interp_pairs(C, bad)
    ff = first_licks(s)
    ff = np.array([f for f in ff if 40 < f < n - 60 and not bad[f - 31:f + 47].any()])
    lags = np.arange(-31, 47)
    E = C[:, ff[:, None] + lags].mean(1)
    E = E - E[:, :31].mean(1, keepdims=True)
    m = Ub @ C[:, ff[:, None] + np.arange(0, 13)].mean((1, 2)) - Ub @ C[:, ff[:, None] + np.arange(-31, 0)].mean((1, 2))
    hot = m >= np.percentile(m, 98)
    um = Ub.mean(0)
    f, P = welch(um @ C, fs=FS, nperseg=2048)
    img = np.full(H * W, np.nan)
    img[br.reshape(-1)] = 100 * m
    return dict(t=lags / FS, cortex=100 * (um @ E), hot=100 * (Ub[hot].mean(0) @ E), f=f, P=P, img=img.reshape(H, W),
                n=len(ff))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    c1 = {s["label"]: s for s in config.load_sessions(machine="analysis_desktop")}
    jobs = []
    for animal in ("PS118", "PS120", "PS121", "PS122"):
        sdir, h5 = SESS[animal]
        jobs.append((f"{animal}_1029", dict(label=f"{animal}_1029", mc=str(ROOT / sdir / "motion_corrected"),
                                            h5=str(DAQ / h5), regime="B", fmdir=None), 1))
    for lab in ("PS92_0811", "PS93_0814"):
        jobs.append((lab, c1[lab], hv.functional_channel(c1[lab])))
    stats = {r["session"]: r for r in json.loads((OUT / "indicator_compare.json").read_text())}
    if "PS120_1029" not in stats:
        stats["PS120_1029"] = analyse("PS120_1029", jobs[1][1], 1)
    T = {lab: traces(s, func) for lab, s, func in jobs}
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.05])
    axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
           fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    for lab, d in T.items():
        col, ls, name = STYLE[lab]
        kw = dict(color=col, ls=ls, lw=1.4, label=f"{name} (n {d['n']})")
        axs[0].plot(d["t"], d["cortex"], **kw)
        axs[1].plot(d["t"], d["cortex"] / max(d["cortex"][31:60].max(), 1e-9), **kw)
        axs[2].plot(d["t"], d["hot"] / max(d["hot"][31:60].max(), 1e-9), **kw)
        lo = d["P"][(d["f"] >= 0.1) & (d["f"] <= 0.3)].mean()
        axs[3].loglog(d["f"][1:], d["P"][1:] / lo, **kw)
        cv = stats[lab]["corr_vs_dist_px"]
        axs[4].plot([int(k) + 5 for k in cv], list(cv.values()), **kw)
    for a, ttl, yl in ((axs[0], "first lick: cortex mean (dF/F %)", "dF/F %"),
                       (axs[1], "first lick: cortex mean, normalised to peak (kinetics)", "fraction of peak"),
                       (axs[2], "first lick: hotspot (top 2% pixels), normalised to peak", "fraction of peak")):
        a.axvline(0, color="k", lw=0.5)
        a.axhline(0, color="k", lw=0.5)
        a.set_title(ttl, fontsize=10)
        a.set_xlabel("s from first lick (DAQ)")
        a.set_ylabel(yl)
    axs[1].set_ylim(-1.2, 1.2)
    axs[2].set_ylim(-1.2, 1.2)
    axs[0].legend(fontsize=7.5)
    axs[3].set_title("corrected cortex-mean power, normalised at 0.1-0.3 Hz (8s keeps more fast power)", fontsize=10)
    axs[3].set_xlabel("Hz")
    axs[4].set_title("spontaneous (0.3-5 Hz) pixel-pair correlation vs distance", fontsize=10)
    axs[4].set_xlabel("distance (px, same rig)")
    axs[4].set_ylabel("mean r")
    axs[4].axhline(0.5, color="k", lw=0.5)
    for i, (lab, d) in enumerate(T.items()):
        a = fig.add_subplot(gs[2, i])
        lim = np.nanpercentile(np.abs(d["img"]), 99)
        a.imshow(d["img"], cmap="RdBu_r", vmin=-lim, vmax=lim)
        r = stats[lab]
        a.set_title(f"{STYLE[lab][2]}\nfirst lick 0-0.4 s ±{lim:.2f}%\nd' {r['singletrial_dprime']:.2f}  "
                    f"t_peak {r['cortex_t_peak_s']:.2f}s", fontsize=8.5)
        a.axis("off")
    fig.suptitle("jGCaMP8s (RiboL1, red) vs GCaMP6s (Thy1, blue; dashed/dotted = cohort1) -- lag + swap-masked "
                 "correction, DAQ timing; grey = PS120 (blood flow only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "cohort2_indicator_compare.png", dpi=100)
    print("wrote", OUT / "cohort2_indicator_compare.png", json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                                                     for k, v in stats["PS120_1029"].items()
                                                                     if k != "corr_vs_dist_px"}))


if __name__ == "__main__":
    main()
