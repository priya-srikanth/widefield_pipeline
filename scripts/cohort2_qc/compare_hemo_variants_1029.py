"""Compare hemo variants on the 10-29 sessions (QC doc §0g): adopted, swap-masked, and the lag-fix / fit-band candidates.

Per session x variant (cortex mean and maps on the dark-relative brain mask, bad pairs interpolated everywhere,
events from the DAQ):
  rcoeff      brain-median per-pixel 415->470 coefficient
  hemoR2_1-8  share of corrected 1-8 Hz variance explained by the (lag-fixed) 415 -> 0 if hemodynamics are removed
  heart_left  corrected / raw-470 power at the session's cardiac peak -> 0 if the heartbeat is removed
  lick_pk     corrected first-lick response 0-0.4 s (%), lick_dip 0.8-2.0 s (%) -> dip should shrink toward 0
  vessel_r    spatial r of the corrected 0-0.2 s lick map with the 415 0-0.2 s map (a vascular template) -> lower
  slow300     corrected >300 s variance share
Figure `cohort2_hemo_variants.png`: lick traces per variant (top), corrected 0-0.4 s lick map per variant (rows).

    conda activate locanmf
    python scripts/cohort2_qc/compare_hemo_variants_1029.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from channel_drift import cos_basis, slow  # noqa: E402
from swapmask_1029 import DAQ, ROOT, SESS  # noqa: E402

FS = 31.23
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
VARS = [("adopted", "hemo_meegkit_hpfit"), ("swapmask", "hemo_meegkit_hpfit_refitT_swapmask"),
        ("lag", "hemo_meegkit_hpfit_lag_refitT_swapmask"), ("lag fit 1-8 Hz", "hemo_meegkit_lag_fit1to8_refitT_swapmask"),
        ("lag fit heart", "hemo_meegkit_lag_fitheart_refitT_swapmask")]
COL = ["0.6", "k", "tab:blue", "tab:green", "tab:red"]


def first_lick_frames(s):
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    _cf, _lf, csmp = _frames(s, cue, lk)
    ls = np.sort(lk["lick_samples"])
    j = np.searchsorted(ls, cue["cue_samples"])
    has = (j < len(ls)) & ((ls[np.minimum(j, len(ls) - 1)] - cue["cue_samples"]) / cue["sample_rate_hz"] < 1.5)
    return np.searchsorted(csmp, ls[j[has]])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import hemo_variants as hv
    fig, ax = plt.subplots(1 + len(VARS), 4, figsize=(17, 4 + 3.2 * len(VARS)), squeeze=False)
    rows = []
    for j, (animal, (sdir, h5)) in enumerate(SESS.items()):
        mc = ROOT / sdir / "motion_corrected"
        res = mc / "wfield_local_results"
        s = dict(label=f"{animal}_1029", mc=str(mc), h5=str(DAQ / h5), regime="B", fmdir=None)
        ff = first_lick_frames(s)
        fa = np.load(res / "frames_average.npy")
        H, W = fa.shape[1:]
        br = brain_mask(fa, res=res)
        U = np.nan_to_num(np.load(res / "U.npy"))
        Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)]
        del U
        um = Ub.mean(0)
        svt = np.load(res / "SVT.npy", mmap_mode="r")
        n = svt.shape[1] // 2
        bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
        A = hv._interp_pairs(np.asarray(svt[:, 1::2], np.float64), bad)
        B = hv._advance_half_sample(hv._interp_pairs(np.asarray(svt[:, 0::2], np.float64), bad))
        a, b = um @ A, um @ B
        keep = np.array([f for f in ff if 70 < f < n - 130 and not bad[f - 62:f + 125].any()])
        lags = np.arange(-int(2 * FS), int(4 * FS))
        t = lags / FS

        def lmap(X, lo, hi, keep=keep, Ub=Ub):
            base = X[:, keep[:, None] + np.arange(-16, -3)].mean((1, 2))
            r = X[:, keep[:, None] + np.arange(int(lo * FS), int(hi * FS))].mean((1, 2))
            return Ub @ (r - base)

        tmpl = lmap(B, 0, 0.2)
        f, Pa = welch(a, fs=FS, nperseg=2048)
        hb = (f > 8) & (f < 13.5)
        fh = f[hb][np.argmax(welch(b, fs=FS, nperseg=2048)[1][hb])]
        b18, a18 = butter(2, [1 / (FS / 2), 8 / (FS / 2)], btype="band")
        bf = filtfilt(b18, a18, b)
        Q = cos_basis(n, FS, 300)
        for k, ((vn, vd), col) in enumerate(zip(VARS, COL)):
            if not (res / vd / "SVTcorr.npy").is_file():
                print(f"{animal} {vn}: not built", flush=True)
                continue
            C = hv._interp_pairs(np.load(res / vd / "SVTcorr.npy").astype(np.float64), bad)
            c = um @ C
            c = c - c.mean()
            rc = np.load(res / vd / "rcoeffs.npy").reshape(H, W)
            cf = filtfilt(b18, a18, c)
            _, Pc = welch(c, fs=FS, nperseg=2048)
            ep = c[keep[:, None] + lags]
            ep = 100 * (ep - ep[:, 31:62].mean(1, keepdims=True)).mean(0)
            m04 = lmap(C, 0, 0.4)
            r = dict(session=animal, variant=vn, rcoeff=float(np.median(rc[br])),
                     hemoR2_1_8=float(np.corrcoef(cf, bf)[0, 1] ** 2),
                     heart_left=float(Pc[np.argmin(np.abs(f - fh))] / Pa[np.argmin(np.abs(f - fh))]),
                     lick_pk=float(ep[(t >= 0) & (t < 0.4)].mean()), lick_dip=float(ep[(t >= 0.8) & (t < 2.0)].mean()),
                     vessel_r=float(np.corrcoef(lmap(C, 0, 0.2), tmpl)[0, 1]),
                     slow300=float(slow(c, Q).var() / c.var()))
            rows.append(r)
            print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()}), flush=True)
            ax[0, j].plot(t, ep, color=col, lw=1.6 if k == 1 else 1, label=vn)
            z = np.full(H * W, np.nan)
            z[br.reshape(-1)] = 100 * m04
            z = z.reshape(H, W)
            lim = np.nanpercentile(np.abs(z), 99)
            ax[1 + k, j].imshow(z, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[1 + k, j].set_title(f"{animal} {vn}: lick 0-0.4 s ±{lim:.2f}%\nrc {r['rcoeff']:.2f}  vessel r "
                                   f"{r['vessel_r']:+.2f}  dip {r['lick_dip']:+.2f}%", fontsize=8.5)
            ax[1 + k, j].axis("off")
        ax[0, j].axvline(0, color="k", lw=0.5)
        ax[0, j].axhline(0, color="k", lw=0.5)
        ax[0, j].set_title(f"{animal}: corrected first-lick, cortex mean (n {len(keep)})", fontsize=9)
        ax[0, j].legend(fontsize=7)
        ax[0, j].set_xlabel("s from first lick (DAQ)")
    fig.tight_layout()
    fig.savefig(OUT / "cohort2_hemo_variants.png", dpi=95)
    (OUT / "cohort2_hemo_variants.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / "cohort2_hemo_variants.png")


if __name__ == "__main__":
    main()
