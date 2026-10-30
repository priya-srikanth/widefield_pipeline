"""Where does cohort2's surviving slow drift come from -- 470 itself, or 415 bleaching leaked in through the
hemo coefficient? DAQ-correct channels only (2026-10-30).

WHY THIS SCRIPT REPLACES THE RAW-.dat PARTS OF THE 10-29 QC. The earlier scripts (iso_snr, stage_traces,
processing_stages, slow_is_bleach, usability_qc) read the raw .dat as fixed even/odd pairs, choosing the
"functional" slot as the brighter of frames 0..40. On 10-29 the LED parity flips 223-305 times per session
(about every 25 s; ~50/50 split; `*_cleanpairs_frame_map.npz`), so every raw-sampled "channel" was a near
equal MIX of 415 and 470. Here channel identity comes only from products built on the DAQ frame map:
`frames_average.npy` (2,H,W) and `SVT.npy` (K, 2T), both in `channel_order` 415-470 (slot 0 = 415,
slot 1 = 470 = `functional_channel`), plus `photobleach/*.json` (DAQ-labelled raw ROI medians).

The adopted correction is c = a_d - T b_d (hemo_variants.compute, meegkit_hpfit: a = 470, b = 415, _d =
order-10 meegkit detrend, T fitted on the 0.1 Hz high-passed traces). On the cortex mean,
um.c = um.a_d - (um T).b_d, so the corrected slow band splits EXACTLY into a 470 term and a 415 term. The
1-D detrend here is unweighted (no trial mask) -- an approximation, checked against the saved SVTcorr
(`recon_r`).

    conda activate locanmf
    python scripts/cohort2_qc/channel_drift.py [--root M:/MICROSCOPE/Priya/Widefield/labcams/20261029]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.signal import butter, filtfilt

FS = 31.23
GAIN = 0.46                      # PCO edge ~0.46 e-/count (approx)
SESS = [("PS118 Thy1 (bright)", "PS118_20261029_112630"),
        ("PS120 Thy1 (dim)", "PS120_20261029_133200"),
        ("PS121 RiboL1", "PS121_20261029_160848"),
        ("PS122 RiboL1", "PS122_20261029_192217")]
ISO, FUNC = 0, 1                 # channel_order 415-470 (every 10-29 summary json)


def cos_basis(n_t, fs, min_period_s):
    t = np.arange(n_t) / fs
    T = max(t[-1], 1e-9)
    n = int(np.floor(2 * T / min_period_s))
    return np.linalg.qr(np.column_stack([np.cos(np.pi * k * t / T) for k in range(n + 1)]))[0]


def slow(y, Q):
    return Q @ (Q.T @ y)


def detrend10(y):
    from meegkit.detrend import detrend
    return np.asarray(detrend(y[:, None].astype(np.float64), order=10)[0])[:, 0]


def bins(y, s=2.0):
    n = int(s * FS)
    m = len(y) // n
    return y[:m * n].reshape(m, n).mean(1)


def one(root: Path, name: str, s: str) -> dict:
    res = root / s / "motion_corrected" / "wfield_local_results"
    man = json.loads((res / "hemo_meegkit_hpfit" / "manifest.json").read_text())
    assert man["functional_channel"] == FUNC, man
    fa = np.load(res / "frames_average.npy")                 # (2,H,W), DAQ-split
    brain = binary_erosion(fa[FUNC] > 0.45 * fa[FUNC].max(), iterations=4)
    cor = lambda im: min(im[:5, :5].mean(), im[:5, -5:].mean(), im[-5:, :5].mean(), im[-5:, -5:].mean())
    lev = {}
    for ch, nm in ((FUNC, "470"), (ISO, "415")):
        F, dark = float(fa[ch][brain].mean()), float(cor(fa[ch]))
        lev[nm] = dict(F=F, dark=dark, shot_dff_pct=100 / np.sqrt(max(F - dark, 1) * GAIN))

    U = np.load(res / "U.npy")
    K = U.shape[-1]
    um = np.nan_to_num(U.reshape(-1, K)[brain.reshape(-1)]).mean(0)
    del U
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    a = um @ np.asarray(svt[:, FUNC::2], np.float64)           # cortex-mean 470 dF/F (fraction)
    b = um @ np.asarray(svt[:, ISO::2], np.float64)            # cortex-mean 415 dF/F
    Tm = np.load(res / "T.npy").astype(np.float64)
    bT = (um @ Tm) @ np.asarray(svt[:, ISO::2], np.float64)    # the 415 term as it enters the cortex mean
    del svt
    c = um @ np.load(res / "hemo_meegkit_hpfit" / "SVTcorr.npy").astype(np.float64)

    a_d, bT_d = detrend10(a), detrend10(bT)
    bl, al = butter(2, 14.0 / (FS / 2), btype="lowpass")
    bT_d = filtfilt(bl, al, bT_d, padlen=50)
    rec = a_d - bT_d
    rec, c = rec - rec.mean(), c - c.mean()
    recon_r = float(np.corrcoef(rec, c)[0, 1])

    # effective cortex-mean gain of the 415 term on the fast (hemodynamic) band
    bh, ah = butter(2, 0.1 / (FS / 2), btype="highpass")
    g = float(np.std(filtfilt(bh, ah, bT)) / np.std(filtfilt(bh, ah, b)))

    out = dict(name=name, session=s, minutes=len(c) / FS / 60, levels=lev, recon_r=recon_r, gain_fast=g,
               raw_pct={"470": 100 * (bins(a, 120)[-1] - bins(a, 120)[0]) / (1 + bins(a, 120)[0]),
                        "415": 100 * (bins(b, 120)[-1] - bins(b, 120)[0]) / (1 + bins(b, 120)[0])},
               sd_c=float(c.std()), bands={})
    for p in (900, 600, 300, 121):
        Q = cos_basis(len(c), FS, p)
        sc, sa, sb = slow(c, Q), slow(a_d - a_d.mean(), Q), slow(bT_d - bT_d.mean(), Q)
        out["bands"][p] = dict(c=float(sc.var() / c.var()), a=float(sa.var() / c.var()),
                               b=float(sb.var() / c.var()),
                               corr_ab=float(np.corrcoef(sa, sb)[0, 1]))
        if p == 300:
            traces = dict(a=bins(a), b=bins(b), c=bins(c), sc=bins(sc), sa=bins(sa), sb=bins(-sb))
    out["_traces"] = traces
    return out


def figure(rs, root: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, len(rs), figsize=(4.4 * len(rs), 9.6), squeeze=False)
    for j, r in enumerate(rs):
        lab = r["session"].split("_")[0] + "_1029"
        pb = json.loads((root / "photobleach" / f"photobleach_{lab}.json").read_text())
        for ch, col in (("470", "royalblue"), ("415", "violet")):
            t, v = pb["_norm"][ch]
            ax[0, j].plot(np.asarray(t) / 60, 100 * np.asarray(v), color=col,
                          label=f"{ch}  {pb['channels'][ch]['pct']:+.0f}%  (F {pb['channels'][ch]['median']:.0f})")
        ax[0, j].set_title(r["name"], fontsize=11)
        ax[0, j].legend(fontsize=8)
        tr = r["_traces"]
        tm = np.arange(len(tr["a"])) * 2 / 60
        ax[1, j].plot(tm, 100 * tr["a"], color="royalblue", lw=0.6, label="470")
        ax[1, j].plot(tm, 100 * tr["b"], color="violet", lw=0.6, label="415")
        ax[1, j].legend(fontsize=8)
        sd = r["sd_c"]
        ax[2, j].plot(tm, tr["c"] / sd, color="0.75", lw=0.5, label="corrected")
        ax[2, j].plot(tm, tr["sc"] / sd, color="k", lw=1.4, label=f"corrected >300 s ({100*r['bands'][300]['c']:.0f}%)")
        ax[2, j].plot(tm, tr["sa"] / sd, color="royalblue", lw=1, label=f"470 part ({100*r['bands'][300]['a']:.0f}%)")
        ax[2, j].plot(tm, tr["sb"] / sd, color="violet", lw=1, label=f"-T.415 part ({100*r['bands'][300]['b']:.0f}%)")
        ax[2, j].legend(fontsize=7.5)
        ax[2, j].set_xlabel("min")
    ax[0, 0].set_ylabel("RAW brain median, % of start\n(DAQ-labelled, photobleach)")
    ax[1, 0].set_ylabel("cortex-mean dF/F % (SVT)\nbefore detrend / correction")
    ax[2, 0].set_ylabel("corrected cortex mean (SD)\n2 s bins; slow parts >300 s")
    fig.suptitle("cohort2 10-29: per-channel bleaching (DAQ-correct) and what makes the corrected slow drift",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029"))
    ap.add_argument("--out", type=Path,
                    default=Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029"))
    a = ap.parse_args()
    rs = []
    for name, s in SESS:
        r = one(a.root, name, s)
        rs.append(r)
        L = r["levels"]
        print(f"{name:20s} {r['minutes']:5.0f} min | 470 F {L['470']['F']:7.0f} dark {L['470']['dark']:5.0f} "
              f"shot {L['470']['shot_dff_pct']:.2f}% | 415 F {L['415']['F']:7.0f} dark {L['415']['dark']:5.0f} "
              f"shot {L['415']['shot_dff_pct']:.2f}% | SVT end-start 470 {r['raw_pct']['470']:+.0f}% "
              f"415 {r['raw_pct']['415']:+.0f}% | recon r {r['recon_r']:.3f} fast gain {r['gain_fast']:.2f}",
              flush=True)
        for p, d in r["bands"].items():
            print(f"    >{p:4d} s: corrected {100*d['c']:5.1f}%  470 part {100*d['a']:5.1f}%  "
                  f"T.415 part {100*d['b']:5.1f}%  corr(470,T.415) {d['corr_ab']:+.2f}")
    a.out.mkdir(parents=True, exist_ok=True)
    figure(rs, a.root, a.out / "cohort2_channel_drift.png")
    (a.out / "cohort2_channel_drift.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "_traces"} for r in rs], indent=1))
    print("wrote", a.out / "cohort2_channel_drift.png")


if __name__ == "__main__":
    main()
