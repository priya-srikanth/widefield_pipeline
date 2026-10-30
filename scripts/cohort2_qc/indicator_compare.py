"""jGCaMP8s (RiboL1: PS121, PS122) vs GCaMP6s (Thy1: PS118; cohort1 pre-stroke sessions as a same-rig reference)
-- metrics that do NOT depend on raw dF/F amplitude (Priya 2026-10-30: "how to best analyze 8s vs 6s"). PS120 is
excluded (no GCaMP signal, QC doc §0h).

Signal: the best correction available (10-29: hemo_meegkit_hpfit_lag_refitT_swapmask; cohort1: the same variant
where built), swap pairs interpolated, dark-relative brain mask (brainmask.py), DAQ events, native pixels.

  KINETICS (first lick, frame resolution; cortex mean and the session's hotspot = top-2% pixels of the 0-0.4 s map):
    onset (first frame >= 20% of peak, after -0.3 s), time to peak, decay to 50% of peak after it
  TEMPORAL: autocorrelation of the corrected cortex mean at 100 / 300 ms; share of 0.1-8 Hz power above 2 Hz
  USABLE SIGNAL: split-half (odd/even trials) r of the first-lick map; single-trial d' at the hotspot
    (mean response / SD across trials, response = 0-0.4 s minus -1-0 s)
  SPECIFICITY (spontaneous, 0.3-5 Hz, first 30 min): pixel-pair correlation vs distance -> distance where it falls
    to 0.5 (px; same rig and magnification for all sessions); participation ratio of the brain-pixel covariance;
    first-lick map sharpness = fraction of brain pixels above half the hotspot response

    conda activate locanmf
    python scripts/cohort2_qc/indicator_compare.py [--cohort1 PS92_0811 PS93_0814 ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from swapmask_1029 import DAQ, ROOT, SESS  # noqa: E402

FS = 31.23
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
VAR = "hemo_meegkit_hpfit_lag_refitT_swapmask"


def first_licks(s):
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


def kinetics(tc, t):
    post = t >= -0.3
    pk = np.argmax(np.where(post & (t < 1.5), tc, -np.inf))
    peak = tc[pk]
    on = np.flatnonzero(post & (tc >= 0.2 * peak))
    onset = t[on[0]] if len(on) else np.nan
    after = np.flatnonzero((np.arange(len(tc)) > pk) & (tc <= 0.5 * peak))
    decay = (t[after[0]] - t[pk]) if len(after) else np.nan
    return dict(onset_s=float(onset), t_peak_s=float(t[pk]), decay50_s=float(decay), peak_pct=float(100 * peak))


def analyse(label, s, func):
    from wfield_local import hemo_variants as hv
    res = Path(s["mc"]) / "wfield_local_results"
    fa = np.load(res / "frames_average.npy")
    H, W = fa.shape[1:]
    br = brain_mask(fa, func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    del U
    yy, xx = np.nonzero(br)
    C = np.load(res / VAR / "SVTcorr.npy").astype(np.float64)
    n = C.shape[1]
    bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
    C = hv._interp_pairs(C, bad)
    um = Ub.mean(0)
    out = dict(session=label)
    ff = first_licks(s)
    ff = ff[(ff > 40) & (ff < n - 60)]
    ff = np.array([f for f in ff if not bad[f - 31:f + 47].any()])
    lags = np.arange(-31, 47)
    t = lags / FS
    E = C[:, ff[:, None] + lags]                                  # K x trials x lags
    E = E - E[:, :, :31].mean(2, keepdims=True)
    resp = E[:, :, 31:44].mean(2)                                 # 0-0.4 s, K x trials
    m = Ub @ resp.mean(1)
    hot = m >= np.percentile(m, 98)
    uh = Ub[hot].mean(0)
    for nm, w in (("cortex", um), ("hotspot", uh)):
        k = kinetics(w @ E.mean(1), t)
        out.update({f"{nm}_{kk}": v for kk, v in k.items()})
    m_odd, m_even = Ub @ resp[:, 0::2].mean(1), Ub @ resp[:, 1::2].mean(1)
    out["splithalf_r"] = float(np.corrcoef(m_odd, m_even)[0, 1])
    rt = uh @ resp
    out["singletrial_dprime"] = float(rt.mean() / rt.std())
    out["n_trials"] = int(len(ff))
    out["sharpness_frac_above_half"] = float(np.mean(m >= 0.5 * m[hot].mean()))
    # temporal
    c = um @ C
    c = c - c.mean()
    ac = np.correlate(c[:60000] - c[:60000].mean(), c[:60000] - c[:60000].mean(), "full")[59999:59999 + 12]
    ac = ac / ac[0]
    out["autocorr_100ms"], out["autocorr_300ms"] = float(ac[3]), float(ac[9])
    f, P = welch(c, fs=FS, nperseg=2048)
    band = (f >= 0.1) & (f <= 8)
    out["power_frac_above_2Hz"] = float(P[band & (f > 2)].sum() / P[band].sum())
    # specificity: spontaneous 0.3-5 Hz covariance
    b, a = butter(2, [0.3 / (FS / 2), 5 / (FS / 2)], btype="band")
    X = filtfilt(b, a, C[:, :min(n, 56000)], axis=1)
    G = X @ X.T / X.shape[1]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Ub), min(2500, len(Ub)), replace=False)
    Us = Ub[idx]
    cov = Us @ G @ Us.T
    sd = np.sqrt(np.diag(cov))
    R = cov / np.outer(sd, sd)
    D = np.hypot(yy[idx][:, None] - yy[idx][None, :], xx[idx][:, None] - xx[idx][None, :])
    edges = np.arange(0, 301, 10)
    prof = [float(np.nanmean(R[(D >= lo) & (D < hi)])) for lo, hi in zip(edges[:-1], edges[1:])]
    out["corr_vs_dist_px"] = dict(zip([int(e) for e in edges[:-1]], prof))
    below = [e for e, p in zip(edges[:-1], prof) if p < 0.5]
    out["corr_halfdist_px"] = int(below[0]) if below else None
    w = np.sqrt(np.clip(np.linalg.eigvalsh(G), 0, None))
    M = (w[:, None] * np.linalg.eigh(G)[1].T) @ (Ub.T @ Ub) @ (np.linalg.eigh(G)[1] * w[None, :])
    lam = np.clip(np.linalg.eigvalsh(M), 0, None)
    out["participation_ratio"] = float(lam.sum() ** 2 / (lam ** 2).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort1", nargs="*", default=["PS92_0811", "PS93_0814"])
    a = ap.parse_args()
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    jobs = []
    for animal in ("PS118", "PS121", "PS122"):
        sdir, h5 = SESS[animal]
        jobs.append((f"{animal}_1029", dict(label=f"{animal}_1029", mc=str(ROOT / sdir / "motion_corrected"),
                                            h5=str(DAQ / h5), regime="B", fmdir=None), 1))
    c1 = {s["label"]: s for s in config.load_sessions(machine="analysis_desktop")}
    for lab in a.cohort1:
        if (Path(c1[lab]["mc"]) / "wfield_local_results" / VAR / "SVTcorr.npy").is_file():
            jobs.append((lab, c1[lab], hv.functional_channel(c1[lab])))
        else:
            print(f"{lab}: {VAR} not built yet, skipped", flush=True)
    rows = []
    for lab, s, func in jobs:
        r = analyse(lab, s, func)
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "corr_vs_dist_px"}),
              flush=True)
    (OUT / "indicator_compare.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / "indicator_compare.json")


if __name__ == "__main__":
    main()
