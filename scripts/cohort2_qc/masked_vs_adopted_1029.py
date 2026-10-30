"""10-29: the swap-masked hemo correction vs the adopted one -- does excluding the other-LED frames fix the RiboL1
correction, and what is left of the slow drift? (Priya 2026-10-30: "resume the 10/29 exploration".)

Per session, ADOPTED = hemo_meegkit_hpfit, MASKED = hemo_meegkit_hpfit_refitT_swapmask (swapmask_1029.py):
  * rcoeff brain median;
  * residual HEMODYNAMICS in the corrected cortex mean: coherence with 415 at the heartbeat, and the share of
    corrected 0.1-14 Hz variance explained by 415 (0 = all removed);
  * the slow band: corrected >300 / >600 s share, split into the 470 part and the T.415 part (channel_drift.py
    method, with the bad pairs interpolated in SVT so it matches the masked build);
  * the cue- and first-lick-locked corrected response (iso_check.py events), adopted vs masked.
Figure: cohort2_masked_vs_adopted.png (rows: rcoeff maps adopted | masked, slow band split, first-lick response).

    conda activate locanmf
    python scripts/cohort2_qc/masked_vs_adopted_1029.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.signal import butter, coherence, filtfilt

sys.path.insert(0, str(Path(__file__).parent))
from channel_drift import bins, cos_basis, detrend10, slow  # noqa: E402
from iso_check import events, locked  # noqa: E402

FS = 31.23
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029")
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
SESS = [("PS118 Thy1 (bright)", "PS118_20261029_112630"), ("PS120 Thy1 (dim)", "PS120_20261029_133200"),
        ("PS121 RiboL1", "PS121_20261029_160848"), ("PS122 RiboL1", "PS122_20261029_192217")]
VAR = {"adopted": "hemo_meegkit_hpfit", "masked": "hemo_meegkit_hpfit_refitT_swapmask"}


def interp(x, bad):
    i = np.arange(len(x))
    x = x.copy()
    x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return x


def bp(x, lo, hi):
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 4, figsize=(17, 15), squeeze=False)
    allr = []
    for j, (name, s) in enumerate(SESS):
        res = ROOT / s / "motion_corrected" / "wfield_local_results"
        fa = np.load(res / "frames_average.npy")
        brain = binary_erosion(fa[1] > 0.45 * fa[1].max(), iterations=4)
        U = np.load(res / "U.npy")
        K = U.shape[-1]
        um = np.nan_to_num(U.reshape(-1, K)[brain.reshape(-1)]).mean(0)
        del U
        bad = np.load(res / "swapped_pairs.npz")["bad"]
        svt = np.load(res / "SVT.npy", mmap_mode="r")
        n = svt.shape[1] // 2
        bad = bad[:n]
        a = interp(um @ np.asarray(svt[:, 1::2], np.float64), bad)
        S415 = np.asarray(svt[:, 0::2], np.float64)
        b = interp(um @ S415, bad)
        fm = np.load(next((res.parent).glob("*frame_map.npz")))
        t = fm["original_frame_index_ch1"][:n] / (2 * FS)
        E = events(s)
        r = dict(session=name, bad_pairs=int(bad.sum()))
        hz = None
        for k, (v, d) in enumerate(VAR.items()):
            c = um @ np.load(res / d / "SVTcorr.npy").astype(np.float64)
            c = interp(c, bad[:len(c)])                        # compare like with like: spikes out of both
            c = c - c.mean()
            rc = np.load(res / d / "rcoeffs.npy").reshape(brain.shape)     # refitT saves it flat
            r[f"{v}_rcoeff_med"] = float(np.median(rc[brain]))
            img = np.where(brain, rc, np.nan)
            im = ax[k, j].imshow(img, vmin=0, vmax=3, cmap="viridis")
            # residual hemodynamics: coherence of corrected with 415 at the heartbeat; 415-explained share 0.1-14 Hz
            f, C = coherence(c, b, fs=FS, nperseg=1024)
            if hz is None:
                f2, C2 = coherence(a, b, fs=FS, nperseg=1024)
                band = (f2 > 8) & (f2 < 13.5)
                hz = f2[band][np.argmax(C2[band])]
                r["heart_Hz"] = float(hz)
                r["raw_470_415_heart_coh"] = float(C2[band].max())
            r[f"{v}_corr_415_heart_coh"] = float(C[np.argmin(np.abs(f - hz))])
            cf, bf = bp(c, 0.1, 14), bp(b, 0.1, 14)
            r[f"{v}_415_R2_fast"] = float(np.corrcoef(cf, bf)[0, 1] ** 2)
            # slow band split with this variant's T (bad pairs interpolated in the 415 term too)
            Tm = np.load(res / d / "T.npy").astype(np.float64)
            bT = interp((um @ Tm) @ S415, bad)
            a_d, bT_d = detrend10(a), detrend10(bT)
            for p in (600, 300):
                Q = cos_basis(len(c), FS, p)
                sc, sa, sb = slow(c, Q), slow(a_d - a_d.mean(), Q), slow(bT_d - bT_d.mean(), Q)
                r[f"{v}_slow{p}_corrected"] = float(sc.var() / c.var())
                r[f"{v}_slow{p}_470part"] = float(sa.var() / c.var())
                r[f"{v}_slow{p}_T415part"] = float(sb.var() / c.var())
                if p == 300 and v == "masked":
                    sd = c.std()
                    tm = np.arange(len(bins(c))) * 2 / 60
                    ax[2, j].plot(tm, bins(c) / sd, color="0.8", lw=0.5, label="corrected (masked)")
                    ax[2, j].plot(tm, bins(sc) / sd, "k", lw=1.3, label=f"> 300 s ({100 * r['masked_slow300_corrected']:.0f}%)")
                    ax[2, j].plot(tm, bins(sa) / sd, color="royalblue", lw=0.9, label=f"470 part ({100 * r['masked_slow300_470part']:.0f}%)")
                    ax[2, j].plot(tm, bins(-sb) / sd, color="violet", lw=0.9, label=f"-T.415 part ({100 * r['masked_slow300_T415part']:.0f}%)")
                    ax[2, j].legend(fontsize=7)
                    ax[2, j].set_xlabel("min")
            # first-lick response of the corrected cortex mean
            lg, ep = locked(c, t[:len(c)], E["lick"])
            mresp = 100 * ep.mean(0)
            ax[3, j].plot(lg, mresp, color=["0.5", "crimson"][k], label=v)
            w = (lg >= 0) & (lg < 0.5)
            r[f"{v}_firstlick_0-0.5s_pct"] = float(mresp[w].mean())
        ax[3, j].axvline(0, color="k", lw=0.5)
        ax[3, j].axhline(0, color="k", lw=0.5)
        ax[3, j].legend(fontsize=8)
        ax[3, j].set_title("first-lick corrected cortex mean dF/F %", fontsize=9)
        ax[0, j].set_title(f"{name}\nrcoeff ADOPTED (median {r['adopted_rcoeff_med']:.2f})", fontsize=9)
        ax[1, j].set_title(f"rcoeff MASKED (median {r['masked_rcoeff_med']:.2f})", fontsize=9)
        for k in (0, 1):
            ax[k, j].axis("off")
        allr.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    fig.colorbar(im, ax=ax[0:2, :].ravel().tolist(), shrink=0.5, label="rcoeff (415 -> 470)")
    fig.suptitle("10-29: adopted hemo vs swap-masked (other-LED frames excluded)")
    fig.savefig(OUT / "cohort2_masked_vs_adopted.png", dpi=105)
    (OUT / "cohort2_masked_vs_adopted.json").write_text(json.dumps(allr, indent=1))
    print("wrote", OUT / "cohort2_masked_vs_adopted.png")


if __name__ == "__main__":
    main()
