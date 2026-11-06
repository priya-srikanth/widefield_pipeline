"""Orofacial vs limb cortex alternation during QUIET periods, pre vs post stroke (cohort1, 2026-10-31).

Priya 10-31, watching the PS93 08-20 / 08-21 event videos: "rapid mode switching of activation of orofacial cortex
and activation of limb (sensory) cortex, but while the animal is just sitting still and not licking"; "do you see
what i mean by the orofacial sensorimotor and limb sensory activation post-stroke while quiet".

ROIs are taken from each session's OWN task maps (no atlas for cohort1 native pixels):
  limb      = top 2% of the running-onset map ([0, 2 s] minus [-2, 0 s], bouts >= 2 s after >= 3 s still)
  orofacial = top 2% of the first-lick map ([0, 0.5 s] minus pre-cue 1 s), limb pixels excluded
QUIET frames = no running bout, no lick and no cue within 2 s; kept in runs >= 5 s. Signal = corrected
(hemo_meegkit_hpfit) minus its 2-s running median (fast fluctuations only); pairs flagged swapped (+/-3) dropped.
Per session: quiet minutes; SD of each ROI in quiet; r(oro, limb); the same r after regressing out the cortex
mean (global); SWITCHES/min = sign changes of the 0.5-s-smoothed (z_oro - z_limb) that cross |1| each side.
Alternation predicts a NEGATIVE partial r and more switches post-stroke.

    python scripts/cohort2_qc/quiet_modes.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

FS = 31.23
EVENTS = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield/behavior_summary/events")
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261031/seizure")
VAR = "hemo_meegkit_hpfit"
SESS = {"PS92": (["0811", "0814"], ["0820", "0821", "0828"]),
        "PS93": (["0811", "0814"], ["0820", "0821", "0822"]),
        "PS94": (["0811", "0814"], ["0823", "0825"])}


def one(label):
    from scipy.ndimage import median_filter, uniform_filter1d
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    s = next(s for s in config.load_sessions(machine="analysis_desktop") if s["label"] == label)
    res, func = Path(s["mc"]) / "wfield_local_results", int(hv.functional_channel(s))
    cf, lf, csmp = _frames(s, _lc(s["h5"]), _ll(s["h5"], "lick_analog"))
    cf, csmp = np.asarray(cf), np.asarray(csmp)
    lf = np.sort(np.asarray(lf)[np.asarray(lf) >= 0])
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fs = float(z["fs"])
    st, sp = z["running_starts"], z["running_stops"]
    C = np.asarray(np.load(res / VAR / "SVTcorr.npy", mmap_mode="r"), np.float64)
    n = C.shape[1]
    br = brain_mask(np.load(res / "frames_average.npy"), func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    bad = np.zeros(n, bool)
    if (res / "swapped_pairs.npz").exists():
        b = np.load(res / "swapped_pairs.npz")["bad"][:n]
        bad[:len(b)] = np.convolve(b.astype(float), np.ones(7), "same")[:n] > 0

    def trig(t, post, pre):
        t = t[(t > 70) & (t < n - 70)]
        return Ub @ (C[:, t[:, None] + post].mean((1, 2)) - C[:, t[:, None] + pre].mean((1, 2)))
    keep = ((sp - st) >= 2 * fs) & (np.r_[np.inf, st[1:] - sp[:-1]] >= 3 * fs)
    run_map = trig(np.searchsorted(csmp, st[keep]), np.arange(0, 63), np.arange(-63, 0))
    k = np.searchsorted(lf, cf)
    fl = np.where(k < len(lf), lf[np.minimum(k, len(lf) - 1)], -1)
    ok = (cf >= 0) & (fl >= cf) & (fl - cf <= int(3.5 * FS))
    cfo, flo = cf[ok], fl[ok]
    sel = (cfo > 70) & (flo < n - 70)
    cfo, flo = cfo[sel], flo[sel]
    lick_map = Ub @ (C[:, flo[:, None] + np.arange(0, 16)].mean((1, 2)) - C[:, cfo[:, None] + np.arange(-31, 0)].mean((1, 2)))
    limb = run_map >= np.percentile(run_map, 98)
    lm = np.where(limb, -np.inf, lick_map)
    oro = lm >= np.percentile(lick_map[~limb], 98)
    busy = bad.copy()
    pad = int(2 * FS)
    for t in np.r_[lf, cf[cf >= 0]]:
        busy[max(0, t - pad):t + pad] = True
    for a, b in np.searchsorted(csmp, np.c_[st, sp]):
        busy[max(0, a - pad):b + pad] = True
    quiet = ~busy
    edges = np.flatnonzero(np.diff(np.r_[0, quiet.astype(int), 0]))
    keepq = np.zeros(n, bool)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 5 * FS:
            keepq[a:b] = True
    out = dict(session=label, animal=animal, quiet_min=float(keepq.sum() / FS / 60),
               n_limb=int(limb.sum()), n_oro=int(oro.sum()))
    if keepq.sum() < 30 * FS:
        return out
    tr = 100 * np.vstack([Ub[oro].mean(0), Ub[limb].mean(0), Ub.mean(0)]) @ C          # 3 x n
    tr = tr - median_filter(tr, (1, 63), mode="nearest")
    q = tr[:, keepq]
    o, li, g = q
    resid = [x - np.polyval(np.polyfit(g, x, 1), g) for x in (o, li)]
    zo, zl = ((x - x.mean()) / x.std() for x in (o, li))
    dd = uniform_filter1d(zo - zl, 16)
    state = np.where(dd > 1, 1, np.where(dd < -1, -1, 0))
    st_ = state[state != 0]
    out.update(sd_oro=float(o.std()), sd_limb=float(li.std()), r=float(np.corrcoef(o, li)[0, 1]),
               r_partial=float(np.corrcoef(*resid)[0, 1]),
               switches_per_min=float((np.diff(st_) != 0).sum() / (keepq.sum() / FS / 60)))
    return out


def main():
    from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
    labels = [f"{a}_{d}" for a, (pre, post) in SESS.items() for d in pre + post]
    res, fail = fan_sessions(labels, one, jobs=5, label="session")
    if fail:
        print("FAILED", fail)
    R = sorted((r for _, r in res), key=lambda r: r["session"])
    for r in R:
        r["epoch"] = "pre" if r["session"][-4:] < "0816" else "post"
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    (OUT / "quiet_modes.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
