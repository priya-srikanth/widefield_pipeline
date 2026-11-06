"""490 nm vs 470 nm (PS118 GCaMP6s, PS121/PS122 RiboL1): response amplitude, trial-to-trial SD, and
single-trial response SNR = mean / across-trial SD of the baseline-subtracted hotspot response. This is a
standardized effect size (one-sample Cohen's d), NOT a signal-detection d' (no two-class discrimination,
and the denominator is the response's own SD, not an independent noise estimate).

Priya 11-05: "how is 490 vs 470 for the ribol1 mice" / "add 490 vs 470 figures to the deck". The imaging box's
`led_490_vs_470.py` "SNR" is the per-pixel SD of the WHOLE corrected signal over the shot floor, so any variance
(noise, residual blood signal, PS122's discharges) counts as signal. Here the signal is the task response:

  per trial: cue 0-1 s and first lick 0-0.5 s (first DAQ lick within the 3.5-s response window), both minus the
  pre-cue 1 s, adopted hemo_meegkit_hpfit, native pixels, brain mask minus clipped pixels;
  hotspot = top 2% of brain pixels of the ODD-trial mean; scored on EVEN trials (no selection bias);
  single-trial response SNR = mean / SD over even trials of the hotspot response.
Trials with a strict discharge within -1.5..+3.5 s of the cue are dropped (seizure_events_video.detect_events), and
PS122 10-29 after 110 min (QC doc §0o). The functional LED comes from the folder tag (PS12x_490_<date> = 490 nm).

    python scripts/cohort2_qc/led_490_vs_470_dprime.py      # -> $WORK/cohort2_490_vs_470_dprime.png + .json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from seizure_events_video import MICRO, detect_events  # noqa: E402

LABCAMS = MICRO / "Widefield" / "labcams"
DAQ = MICRO / "Widefield" / "DAQ_recorder_output"
FS = 31.23
VAR = "hemo_meegkit_hpfit"
EXCLUDE_AFTER_MIN = {("PS122", "20261029"): 110.0}
ANIMALS = ("PS118", "PS121", "PS122")   # PS118 = Thy1-GCaMP6s (490 from 11-05); PS121/PS122 = RiboL1-jGCaMP8s (490 from 11-04)


def one(sd):
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    sd = Path(sd)
    animal, date = sd.name.split("_")[0], sd.parent.name
    h5 = sorted((DAQ / date).glob(f"{animal}_{date}_*.h5"))[0]
    mc = sd / "motion_corrected"
    res = mc / "wfield_local_results"
    s = dict(label=f"{animal}_{date[4:]}", mc=str(mc), h5=str(h5), regime="B", fmdir=None)
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog")              # THE DAQ lick detector (config.lick_detector; REBUILD_CODE 8.53)
    cf, lf, _ = _frames(s, cue, lk)
    cf, lf = np.asarray(cf), np.sort(np.asarray(lf)[np.asarray(lf) >= 0])
    C = np.asarray(np.load(res / VAR / "SVTcorr.npy"), np.float64)
    n = C.shape[1]
    fa = np.load(res / "frames_average.npy")
    br = brain_mask(fa, 1, res=res)
    U = np.nan_to_num(np.load(res / "U.npy")).reshape(-1, C.shape[0])[br.reshape(-1)].astype(np.float64)
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    um = U.mean(0)
    x = 100 * (um @ np.asarray(SVT[:, 1::2], np.float64))
    y = 100 * (um @ np.asarray(SVT[:, 0::2], np.float64))
    ev = detect_events(x, y)
    ok = (cf > 40) & (cf < n - 40)
    cut = EXCLUDE_AFTER_MIN.get((animal, date))
    if cut is not None:
        ok &= cf < int(cut * 60 * FS)
    for e in ev:
        ok &= ~((e - cf >= -1.5 * FS) & (e - cf <= 3.5 * FS))
    cf = cf[ok]
    k = np.searchsorted(lf, cf)
    fl = np.where(k < len(lf), lf[np.minimum(k, len(lf) - 1)], -1)
    hit = (fl >= cf) & (fl - cf <= int(3.5 * FS))
    base = C[:, cf[:, None] + np.arange(-31, 0)].mean(2)
    out = dict(session=s["label"], animal=animal, date=date, led="490" if "_490_" in sd.name else "470",
               F0=float(fa[1][br].mean()))
    for kind, R in (("cue", C[:, cf[:, None] + np.arange(0, 31)].mean(2) - base),
                    ("lick", C[:, fl[hit][:, None] + np.arange(0, 16)].mean(2) - base[:, hit])):
        P = 100 * (U @ R)                                    # pix x trials, dF/F %
        hot = np.argsort(P[:, 0::2].mean(1))[-int(0.02 * len(U)):]
        tr = P[:, 1::2][hot].mean(0)
        out[kind] = dict(n=int(R.shape[1]), amp=float(tr.mean()), sd=float(tr.std(ddof=1)),
                         snr=float(tr.mean() / tr.std(ddof=1)))
    return out


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    marker = {"PS118": "^", "PS121": "o", "PS122": "s"}
    cols = (("amp", "hotspot response (ΔF/F %)"), ("sd", "trial-to-trial SD (ΔF/F %)"),
            ("snr", "single-trial response SNR\n(mean / across-trial SD)"))
    # cumulative x-positions: one slot per session, a gap between animals, so unequal group sizes never collide
    xp = {}; centre = {}; x = 0.0
    for an in ANIMALS:
        rr = [r for r in R if r["animal"] == an]
        x0 = x
        for r in rr:
            xp[(r["animal"], r["date"])] = x; x += 1.0
        centre[an] = (x0 + x - 1.0) / 2.0
        x += 1.5
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.8))
    for i, kind in enumerate(("cue", "lick")):
        for j, (key, lab) in enumerate(cols):
            a = ax[i, j]
            for an in ANIMALS:
                for r in (r for r in R if r["animal"] == an):
                    is490 = r["led"] == "490"
                    a.plot(xp[(r["animal"], r["date"])], r[kind][key], marker[an], ms=8,
                           mfc="tab:red" if is490 else "none", mec="tab:red" if is490 else "0.35", mew=1.4)
                    if i == 1:
                        a.text(xp[(r["animal"], r["date"])], -0.04, f"{r['date'][4:6]}-{r['date'][6:]} {r['led']}",
                               ha="right", va="top", rotation=90, fontsize=6.5,
                               color="tab:red" if is490 else "0.3", transform=a.get_xaxis_transform())
                a.text(centre[an], 1.01, an, ha="center", va="bottom", fontsize=8.5, transform=a.get_xaxis_transform())
            a.set_xticks([]); a.set_xlim(-0.8, x - 0.9)
            a.set_ylim(0, 1.18 * max(r[kind][key] for r in R))
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
            a.set_ylabel(lab, fontsize=9)
            if j == 0:
                a.set_title(("cue 0-1 s" if kind == "cue" else "first lick 0-0.5 s") + "  (minus pre-cue 1 s)",
                            fontsize=10, loc="left", pad=20)
    fig.suptitle("Single-trial response SNR = mean / across-trial SD of the baseline-subtracted hotspot response "
                 "(a standardized effect size, NOT a signal-detection d').\nPS118 GCaMP6s (triangles), RiboL1 PS121 "
                 "(circles) / PS122 (squares): 490 nm (red filled) vs 470 nm (grey open), one marker per session. "
                 "490 raises ΔF/F and trial SD about equally, so SNR ~ each mouse's best 470 day.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    from wfield_local.analysis_kit import fan_sessions
    items = [str(p) for an in ANIMALS for p in sorted(LABCAMS.glob(f"2026*/{an}_*"))
             if (p / "motion_corrected" / "wfield_local_results" / VAR / "SVTcorr.npy").exists()]
    res, fail = fan_sessions(items, one, jobs=6, label="session")
    if fail:
        print("FAILED", fail)
    R = sorted((r for _, r in res), key=lambda r: (r["animal"], r["date"]))
    work = Path(os.environ.get("WORK", "E:/labcams_data"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "cohort2_490_vs_470_dprime.json").write_text(json.dumps(R, indent=1))
    for r in R:
        print(f"{r['session']:11s} {r['led']}  cue respSNR {r['cue']['snr']:.2f}  lick respSNR {r['lick']['snr']:.2f}")
    figure(R, work / "cohort2_490_vs_470_dprime.png")


if __name__ == "__main__":
    main()
