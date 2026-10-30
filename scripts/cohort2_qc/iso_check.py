"""Is 415 isosbestic for RiboL1-jGCaMP8s? And is the 415 still usable at the end of the session? cohort2 10-29.

Priya 2026-10-30: "is 415 signal at the end of the session ok? ... It may be true that 415 isn't truly isosbestic
for 8s -- let's explore." Thy1-GCaMP6s (PS118, PS120) is the reference: its 415 is the textbook isosbestic.

Four checks, all on DAQ-correct channels (`SVT` slots 0 = 415, 1 = 470, as dF/F; `frames_average`):
  1. END OF SESSION: 415 level above dark and the shot-noise floor in the last 10 min (photobleach JSON 'end'),
     plus the 415 fast-band amplitude and 470/415 coupling in session thirds (does 415 lose its content?).
  2. HEARTBEAT: 470 / 415 coherence and spectra. The cardiac line (~8-13 Hz awake mouse) is purely hemodynamic,
     so an isosbestic 415 shows it with the same dF/F sign as 470.
  3. EVENT-LOCKED: cue- and first-lick-locked cortex-mean dF/F of 470 and 415 (per-trial baseline -1..0 s).
     Hemodynamic 415 = slow (peak >~1 s), mostly a dip; calcium in 415 = fast, same time course as 470.
  4. MAPS: per-pixel 0-0.5 s first-lick response for 470 and 415 and their spatial correlation; calcium
     maps are focal and alike, hemodynamic 415 maps follow vessels.
Event clock: the behaviour board's device_t_ms mapped to PC time by a linear fit to the GUI timestamps (removes
serial jitter and board-clock drift), then PC time minus the DAQ file's start (its name, 1 s resolution).
A CONSTANT offset of up to ~1-2 s therefore remains: absolute latencies are approximate, but 415 vs 470 relative
timing and sign are exact (same frames). Frame times come from the clean-pairs frame map (original frame / 2 / FS).

    conda activate locanmf
    python scripts/cohort2_qc/iso_check.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.signal import butter, coherence, filtfilt, welch

FS = 31.23
GAIN = 0.46
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029")
BEH = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield")
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
SESS = [("PS118 Thy1 (bright)", "PS118_20261029_112630"), ("PS120 Thy1 (dim)", "PS120_20261029_133200"),
        ("PS121 RiboL1", "PS121_20261029_160848"), ("PS122 RiboL1", "PS122_20261029_192217")]
PRE, POST = 2.0, 6.0


def load(s):
    res = ROOT / s / "motion_corrected" / "wfield_local_results"
    fa = np.load(res / "frames_average.npy")
    brain = binary_erosion(fa[1] > 0.45 * fa[1].max(), iterations=4)
    U = np.load(res / "U.npy")
    K = U.shape[-1]
    Ub = np.nan_to_num(U.reshape(-1, K)[brain.reshape(-1)])
    del U
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    S470, S415 = np.asarray(svt[:, 1::2], np.float64), np.asarray(svt[:, 0::2], np.float64)
    fm = np.load(next((ROOT / s / "motion_corrected").glob("*frame_map.npz")))
    t = fm["original_frame_index_ch1"] / (2 * FS)
    return dict(fa=fa, brain=brain, Ub=Ub, S470=S470, S415=S415, t=t[:S470.shape[1]])


def events(s):
    animal = s.split("_")[0]
    pb = json.loads((ROOT / "photobleach" / f"photobleach_{animal}_1029.json").read_text())
    t0 = datetime.strptime(Path(pb["daq"]).stem.split("_", 1)[1], "%Y%m%d_%H%M%S").timestamp()
    d = next(BEH.glob(f"{animal}_20261029_*"))
    e = pd.read_csv(d / "events.csv", usecols=["gui_timestamp_unix", "device_t_ms", "event_name"])
    p = np.polyfit(e.device_t_ms.to_numpy(float), e.gui_timestamp_unix.to_numpy(float), 1)
    tt = np.polyval(p, e.device_t_ms.to_numpy(float)) - t0
    cue = tt[e.event_name.to_numpy() == "cue"]
    lk = np.sort(tt[e.event_name.to_numpy() == "lick_on"])
    first = []
    for c in cue:
        j = np.searchsorted(lk, c)
        if j < len(lk) and lk[j] - c < 1.5:
            first.append(lk[j])
    return dict(cue=cue, lick=np.array(first), pb=pb)


def hp(x, f=0.1):
    b, a = butter(2, f / (FS / 2), btype="highpass")
    return filtfilt(b, a, x, axis=-1)


def locked(y, t, ev):
    """(n_events, n_lags) baseline-subtracted epochs of y (1-D or (K,T) -> mean over events (K, n_lags))."""
    lags = np.arange(-int(PRE * FS), int(POST * FS))
    idx = np.searchsorted(t, ev)
    ok = (idx + lags[0] >= 0) & (idx + lags[-1] < len(t))
    ii = idx[ok][:, None] + lags[None, :]
    base = slice(int((PRE - 1.0) * FS), int(PRE * FS))
    if y.ndim == 1:
        ep = y[ii]
        return lags / FS, ep - ep[:, base].mean(1, keepdims=True)
    ep = y[:, ii]                                     # (K, n, L)
    ep = ep - ep[:, :, base].mean(2, keepdims=True)
    return lags / FS, ep.mean(1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 4, figsize=(17, 18), squeeze=False)
    summ = []
    for j, (name, s) in enumerate(SESS):
        D, E = load(s), events(s)
        um = D["Ub"].mean(0)
        a, b = hp(um @ D["S470"]), hp(um @ D["S415"])
        t = D["t"]
        r = dict(session=name)

        # 1. end of session
        fa = D["fa"]
        dark415 = min(fa[0][:5, :5].mean(), fa[0][:5, -5:].mean(), fa[0][-5:, :5].mean(), fa[0][-5:, -5:].mean())
        c = E["pb"]["channels"]
        for k, nm in (("start", "start"), ("end", "end")):
            sig = c["415"][k] - dark415
            r[f"415_{nm}_over_dark"] = sig / dark415
            r[f"415_{nm}_shot_pct"] = 100 / np.sqrt(sig * GAIN)
        thirds = np.array_split(np.arange(len(a)), 3)
        bb, aa = butter(2, [0.1 / (FS / 2), 1.0 / (FS / 2)], btype="band")
        al, bl = filtfilt(bb, aa, a), filtfilt(bb, aa, b)
        bh, ah = butter(2, [1.0 / (FS / 2), 14.0 / (FS / 2)], btype="band")
        bf = filtfilt(bh, ah, b)
        for i, th in enumerate(thirds):
            r[f"T{i+1}_415_fastSD_pct"] = 100 * bf[th].std()
            r[f"T{i+1}_slope_0.1-1Hz"] = al[th] @ bl[th] / (bl[th] @ bl[th])
            r[f"T{i+1}_r_0.1-1Hz"] = np.corrcoef(al[th], bl[th])[0, 1]

        # 2. heartbeat
        f, Cxy = coherence(a, b, fs=FS, nperseg=1024)
        _, Pa = welch(a, fs=FS, nperseg=1024)
        _, Pb = welch(b, fs=FS, nperseg=1024)
        band = (f > 6) & (f < 14)
        rel = Pb / np.median(Pb[(f > 3) & (f < 15)])
        fh = f[band][np.argmax(rel[band])]
        r["heart_Hz_415"] = fh
        r["heart_peak_415_rel"] = rel[band].max()
        r["coh_at_heart"] = Cxy[np.argmin(abs(f - fh))]
        # sign of the shared cardiac component: correlation of narrow-band traces
        bc, ac = butter(2, [(fh - 0.7) / (FS / 2), (fh + 0.7) / (FS / 2)], btype="band")
        r["heart_r_470_415"] = np.corrcoef(filtfilt(bc, ac, a), filtfilt(bc, ac, b))[0, 1]
        ax[0, j].semilogy(f, Pa, color="royalblue", label="470")
        ax[0, j].semilogy(f, Pb, color="violet", label="415")
        ax2 = ax[0, j].twinx()
        ax2.plot(f, Cxy, color="k", lw=0.8)
        ax2.set_ylim(0, 1)
        ax[0, j].set_title(f"{name}\nspectra; black = coherence (heart {fh:.1f} Hz, coh {r['coh_at_heart']:.2f}, "
                           f"r {r['heart_r_470_415']:+.2f})", fontsize=9)
        ax[0, j].legend(fontsize=8)

        # 3. event-locked
        for row, (ev, lab) in enumerate(((E["cue"], "cue"), (E["lick"], "first lick"))):
            lg, ea = locked(a, t, ev)
            _, eb = locked(b, t, ev)
            ma, mb = 100 * ea.mean(0), 100 * eb.mean(0)
            sa, sb = 100 * ea.std(0) / np.sqrt(len(ea)), 100 * eb.std(0) / np.sqrt(len(eb))
            A = ax[1 + row, j]
            for m, se, col, nm in ((ma, sa, "royalblue", "470"), (mb, sb, "violet", "415")):
                A.plot(lg, m, color=col, label=nm)
                A.fill_between(lg, m - se, m + se, color=col, alpha=0.25)
            A.axvline(0, color="k", lw=0.5)
            A.axhline(0, color="k", lw=0.5)
            w = (lg >= 0) & (lg < 0.5)
            w2 = (lg >= 1.0) & (lg < 3.0)
            r[f"{lab}_n"] = len(ea)
            r[f"{lab}_470_0-0.5s"], r[f"{lab}_415_0-0.5s"] = ma[w].mean(), mb[w].mean()
            r[f"{lab}_470_1-3s"], r[f"{lab}_415_1-3s"] = ma[w2].mean(), mb[w2].mean()
            r[f"{lab}_415_t_peakabs"] = lg[lg >= 0][np.argmax(abs(mb[lg >= 0]))]
            r[f"{lab}_470_t_peakabs"] = lg[lg >= 0][np.argmax(abs(ma[lg >= 0]))]
            A.set_title(f"{lab}-locked dF/F % (n {len(ea)})", fontsize=9)
            A.legend(fontsize=8)
            A.set_xlabel("s (constant clock offset <~2 s)")

        # 4. maps: first-lick 0-0.5 s and 1-3 s response per pixel (brain pixels)
        lg, Ka = locked(hp(D["S470"]), t, E["lick"])
        _, Kb = locked(hp(D["S415"]), t, E["lick"])
        H, W = D["brain"].shape
        for col, (w, wl) in enumerate((((lg >= 0) & (lg < 0.5), "0-0.5 s"), ((lg >= 1) & (lg < 3), "1-3 s"))):
            ma, mb = D["Ub"] @ Ka[:, w].mean(1), D["Ub"] @ Kb[:, w].mean(1)
            r[f"map_r_470_415_{wl}"] = np.corrcoef(ma, mb)[0, 1]
            if col == 0:
                img = np.full(H * W, np.nan)
                img[D["brain"].reshape(-1)] = 100 * ma
                lim = np.nanpercentile(abs(img), 99)
                ax[3, j].imshow(img.reshape(H, W), cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax[3, j].set_title(f"470 first-lick 0-0.5 s (±{lim:.2f}%)", fontsize=9)
                img[D["brain"].reshape(-1)] = 100 * mb
                lim = np.nanpercentile(abs(img), 99)
                ax[4, j].imshow(img.reshape(H, W), cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax[4, j].set_title(f"415 first-lick 0-0.5 s (±{lim:.2f}%); map r {r['map_r_470_415_0-0.5 s']:+.2f}",
                                   fontsize=9)
        for k in (3, 4):
            ax[k, j].axis("off")
        summ.append(r)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "session"},
              flush=True)
    fig.suptitle("cohort2 10-29: is 415 isosbestic? (spectra/coherence | cue | first lick | 470 map | 415 map)")
    fig.tight_layout()
    fig.savefig(OUT / "cohort2_iso_check.png", dpi=110)
    pd.DataFrame(summ).to_csv(OUT / "cohort2_iso_check.csv", index=False)
    print("wrote", OUT / "cohort2_iso_check.png")


if __name__ == "__main__":
    main()
