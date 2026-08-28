"""Is the pre-cue position code partly a ZERO-PHASE FILTER ECHO of the post-cue response?

wfield.hemodynamic_correction high-passes both channels at 0.1 Hz with scipy filtfilt -- forward AND
backward -- and the high-passed 470 channel IS what becomes SVTcorr. filtfilt's impulse response is
symmetric in time (measured: -0.496 before an impulse, -0.496 after), so a position-specific post-cue
response casts a scaled, SIGN-FLIPPED shadow over the seconds BEFORE the cue. A linear decoder is
indifferent to sign, so that shadow is decodable position information in a window where, biologically,
none of it should have come from the future.

THE TEST. Rebuild SVTcorr from the SAME session with three high-pass variants and re-run the SAME
decoder on each:

    zerophase   HP applied to both channels with filtfilt          <- the current pipeline
    causal      HP applied with lfilter (forward only)             <- the pre-cue window cannot
                                                                      contain post-cue information
    fitonly     no HP on the output at all                         <- the correction applied to
                                                                      unfiltered data

Only the FILTER differs: the hemodynamic transform T is loaded from disk and reused for all three, so
the coefficient fit is held fixed and the comparison isolates the filter. (T is a spatial mixing
matrix; refitting it per variant would confound two changes at once.)

HOW TO READ IT. POST-CUE is the control -- the real response is inside the window there, so it should
survive every variant. PRE-CUE is the test:
  * pre-cue holds up under `causal`  -> the maintained code is real; the echo is not what carries it.
  * pre-cue collapses under `causal` -> what we have been calling a maintained plan is substantially
    the post-cue response smeared backwards by the preprocessing, and the pre-stroke headline claim
    needs re-deriving from causally-filtered data.

Anything in between is a mixture, and the causal number is then the honest size of the real effect.

THE ENGAGED CUT COMES FROM CONFIG (`decode.max_rt_s`), not from a literal here. It was hardcoded to
2.0 s while the config moved to 3.5 s on 2026-08-21 -- the task's REAL response window, read per
session from `gui_config.json` -- so this module was filing a lick at 2.5 s as "no lick" when it is a
REWARDED HIT the task scored, and "engaged" meant one thing here and another in the analysis this is a
diagnostic FOR. A sweep is internally consistent at either cut, which is exactly why it could drift
unnoticed; what it cannot survive is being quoted against a headline computed at the other one.

EVERY NUMBER RECORDED FOR THIS MODULE IN DECISIONS WAS MEASURED AT 2.0 s and is pre-change until
re-measured. The cut actually used is printed at run time, so a result can never be read without it.
"""
from __future__ import annotations

import glob
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

FS, HP, LP, FUNC = 31.23, 0.1, 14.0, 1        # configs/defaults.yaml svd.*



_ANNOUNCED = False


def _max_rt() -> float:
    """The engaged cut, from `decode.max_rt_s`, announced once so a result carries its own boundary.

    Announced rather than merely read: this module's recorded numbers were measured at 2.0 s, and a
    silent change of boundary is how a diagnostic and the headline it is a diagnostic FOR come to
    disagree without either looking wrong.
    """
    global _ANNOUNCED
    v = float(config.defaults()["decode"]["max_rt_s"])
    if not _ANNOUNCED:
        print(f"[{__name__.rsplit('.', 1)[-1]}] engaged cut = {v:g}s (decode.max_rt_s). Numbers "
              f"recorded in DECISIONS for this module were measured at 2.0s.", flush=True)
        _ANNOUNCED = True
    return v


def _hp(X, mode):
    if mode in ("fitonly", "quietdetrend", "taskdetrend", "strobedetrend"):
        return X
    b, a = butter(2, HP / (FS / 2), btype="highpass")
    return filtfilt(b, a, X, padlen=50) if mode == "zerophase" else lfilter(b, a, X)


# ---------------------------------------------------------------- quiet-masked detrending
TRIAL_PRE_S, TRIAL_POST_S, DETREND_WIN_S = 1.0, 4.0, 60.0


def fit_mask(s, n_frames, csmp, cue, pre_s=TRIAL_PRE_S, post_s=TRIAL_POST_S,
             from_strobe=True, require_quiet=True):
    """Frames usable for estimating slow drift: BEHAVIOURALLY QUIET **and** outside every trial.

    Both halves are necessary and neither alone is enough.

    * TRIAL-MASKED (van Driel et al. 2021): if the drift estimate sees the evoked responses, it
      partially tracks them, and subtracting it redistributes response energy into neighbouring time --
      the same displacement the zero-phase filter causes, just via a different route.
    * QUIET-ONLY (Priya, 2026-08-13): cortex-wide activity is dominated by movement (Musall et al.
      2019), so drift fitted across running or licking epochs is fitting movement, not drift. The quiet
      mask from ``behavior_events`` is already "not running AND not licking AND not peri-reward".

    The trap this avoids: the ENL is an ENFORCED no-lick period, so quiet bouts overlap the PRE-CUE
    window heavily. Masking on quiet alone would fit the drift through exactly the signal we are trying
    to measure. Hence quiet AND task-free.

    Returns (mask, diagnostics).
    """
    from wfield_local import behavior_events, config

    animal, mmdd = s["label"].split("_")
    ev = behavior_events.get_or_compute(config.resolver(), animal, f"2026{mmdd}")
    mask = np.ones(n_frames, bool)

    # 1. drop every trial window: strobe - pre_s  ->  cue + post_s
    cs, ss = np.asarray(cue["cue_samples"]), np.asarray(cue["strobe_samples"])
    sr = float(cue["sample_rate_hz"])
    j = np.searchsorted(ss, cs, side="right") - 1
    lead = np.where(j >= 0, (cs - ss[np.clip(j, 0, len(ss) - 1)]) / sr, 0.0)
    cue_f = np.searchsorted(csmp, cs)
    for f, ld in zip(cue_f, lead):
        a = int(f - ((ld if from_strobe else 0.0) + pre_s) * FS)
        b = int(f + post_s * FS)
        mask[max(0, a):min(n_frames, b)] = False
    after_trials = float(mask.mean())

    # 2. keep only behaviourally QUIET bouts
    if require_quiet and ev is not None and len(np.asarray(ev.get("quiet_starts", []))):
        q = np.zeros(n_frames, bool)
        fs_daq = float(ev["fs"])
        for a, b in zip(np.asarray(ev["quiet_starts"]), np.asarray(ev["quiet_stops"])):
            fa, fb = np.searchsorted(csmp, a / fs_daq * sr), np.searchsorted(csmp, b / fs_daq * sr)
            q[max(0, int(fa)):min(n_frames, int(fb))] = True
        mask &= q
        quiet_available = True
    else:
        quiet_available = not require_quiet
    return mask, {"frac_after_trial_mask": after_trials, "frac_final": float(mask.mean()),
                  "quiet_available": quiet_available}


# What each detrending variant masks OUT before estimating the drift. Declared here rather than
# buried in a call so the bounds are auditable -- they are a CHOICE, and an earlier version of this
# module hardcoded one of them while claiming the drift fit stayed clear of the measured window.
#
#   taskdetrend    cue-0.5s -> cue+4s     evoked response only. Leaves 1.5 s of the 2 s pre-cue window
#                                         INSIDE the drift fit, so it may shave real pre-cue signal.
#   strobedetrend  strobe-0.25s -> cue+4s the WHOLE trial including the ENL, so the fit never sees the
#                                         measured window. Costs eligible data: the strobe->cue lead is
#                                         a median 3 s but reaches a p90 of 18 s, so this masks a lot.
#   quietdetrend   as strobedetrend, AND only behaviourally quiet frames. NOT ESTIMABLE in this task --
#                                         quiet is 0.3-10.5% of a session (PS92: 30 s in 156 min).
MASK_SPEC = {
    "taskdetrend":   dict(pre_s=0.5,  from_strobe=False, require_quiet=False),
    "strobedetrend": dict(pre_s=0.25, from_strobe=True,  require_quiet=False),
    "quietdetrend":  dict(pre_s=1.0,  from_strobe=True,  require_quiet=True),
}


def detrend_masked(X, mask, win_s=DETREND_WIN_S, min_frac=0.05):
    """Subtract a slow trend estimated ONLY from ``mask`` samples. X is (K, T).

    Windowed MEDIAN (robust to any transient that slips through the mask) over eligible samples, then
    linear interpolation across windows that had too few. No convolution over the data, so unlike a
    filter it has no impulse response and cannot displace signal in time -- which is the entire point.
    ``win_s`` sets how fast a drift can be tracked (~1/(2*win_s) Hz); 60 s is far slower than any task
    event but far faster than real bleaching drift.
    """
    K, T = X.shape
    w = max(1, int(round(win_s * FS)))
    cs, vs = [], []
    for a in range(0, T, w):
        b = min(T, a + w)
        m = mask[a:b]
        if m.sum() >= max(10, min_frac * (b - a)):
            cs.append(0.5 * (a + b))
            vs.append(np.median(X[:, a:b][:, m], axis=1))
    if len(cs) < 2:
        return X, 0.0
    C = np.asarray(cs, float)
    V = np.stack(vs, 1)
    t = np.arange(T, dtype=float)
    trend = np.stack([np.interp(t, C, V[k]) for k in range(K)])
    return X - trend, len(cs) / max(1, int(np.ceil(T / w)))


def _lp(X):
    b, a = butter(2, LP / (FS / 2), btype="lowpass")
    return filtfilt(b, a, X, padlen=50)


def svtcorr(svt, T, mode, mask=None, win_s=DETREND_WIN_S):
    """Rebuild SVTcorr with the chosen drift removal, reusing the saved transform T."""
    a = _hp(svt[:, FUNC::2].astype(np.float64), mode)
    b = _hp(svt[:, (FUNC + 1) % 2::2].astype(np.float64), mode)
    if mode in ("quietdetrend", "taskdetrend", "strobedetrend"):
        if mask is None:
            raise ValueError("quietdetrend needs a fit mask (see fit_mask)")
        ma = mask[:a.shape[1]] if mask.size >= a.shape[1] else np.pad(mask, (0, a.shape[1] - mask.size))
        mb = mask[:b.shape[1]] if mask.size >= b.shape[1] else np.pad(mask, (0, b.shape[1] - mask.size))
        a, _ = detrend_masked(a, ma, win_s)
        b, _ = detrend_masked(b, mb, win_s)
    if LP < FS / 2:
        b = _lp(b)
    a = (a.T - np.nanmean(a, 1)).T
    b = (b.T - np.nanmean(b, 1)).T
    c = a - T @ b
    return (c.T - np.nanmean(c, 1)).T.astype(np.float32)


def roi_signal(allen_dir, SVTc):
    """Allen-ROI features from a given SVTcorr -- the same construction as _build_signal(source='roi'),
    contracted per ROI so the (npix, T) movie is never materialized."""
    U = np.load(f"{allen_dir}/U_atlas.npy")
    atlas = np.load(f"{allen_dir}/allen_area_atlas_native_grid.npy").reshape(-1)
    mask = np.load(f"{allen_dir}/allen_brain_mask_native_grid.npy").astype(bool).reshape(-1)
    Uf = U.reshape(-1, U.shape[2])
    rois, regs = [], []
    for l in np.unique(atlas):
        if l == 0:
            continue
        pix = (atlas == l) & mask
        if pix.sum() < 20:
            continue
        rois.append(np.nanmean(Uf[pix], 0) @ SVTc)
        regs.append(int(l))
    return np.asarray(rois), np.asarray(regs)


def decode(s, sig, regs, align):
    args = SimpleNamespace(source="roi", align=align, baseline="none",
                           pre_s=1.0, post_s=2.0, fs=FS, max_rt=_max_rt())
    X, y, g, _, _, _ = _trial_features(s, args, signal=sig, feat_region=regs)
    ng = min(5, int(np.unique(g).size))
    if ng < 2 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    return float(accuracy_score(y, cross_val_predict(clf, X, y, cv=GroupKFold(ng), groups=g)))


def patterns(s, sig):
    """Position patterns + level vs a far-from-cue baseline, for the SIGN test.

    The shadow prediction is sharper than "pre-cue decodes above chance": the pre-cue position pattern
    should be the NEGATIVE of the post-cue one, and pre-cue activity should sit BELOW the surrounding
    baseline. A maintained plan has no reason to be an inverted copy of the movement response.
    """
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_crossanimal_dff import _frames as _fr
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc

    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, _l, _c = _fr(s, cue, lk)
    codes = classify_cues_with_backup(s, cue, verbose=False)
    n, T = int(round(2.0 * FS)), sig.shape[1]
    far = np.ones(T, bool)
    w = int(round(5.0 * FS))
    for c in cue_f.astype(int):
        far[max(0, c - w):min(T, c + w)] = False
    base = sig[:, far].mean(1) if far.any() else sig.mean(1)
    pre, post, y = [], [], []
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        c = int(cue_f[k])
        if c - n < 0 or c + n > T:
            continue
        pre.append(sig[:, c - n:c].mean(1) - base)
        post.append(sig[:, c:c + n].mean(1) - base)
        y.append(int(codes[k]))
    pre, post, y = np.asarray(pre), np.asarray(post), np.asarray(y)
    if not len(y) or len(np.unique(y)) < len(DISPLAY_ORDER):
        return float("nan"), float("nan")
    P = np.stack([pre[y == p].mean(0) for p in DISPLAY_ORDER], 1)
    Q = np.stack([post[y == p].mean(0) for p in DISPLAY_ORDER], 1)
    P = P - P.mean(1, keepdims=True)
    Q = Q - Q.mean(1, keepdims=True)
    return float(np.corrcoef(P.ravel(), Q.ravel())[0, 1]), float(pre.mean())


def analyse_session(lab, modes, win_s=DETREND_WIN_S, refit_t=False):
    """Every metric for one session, in ONE pass so SVT is loaded once."""
    from wfield_local.locanmf_crossanimal_dff import _frames as _fr
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc

    s = next((x for x in SESSIONS if x["label"] == lab), None)
    if s is None:
        return None
    s_sess = s
    res = s["mc"] + "/wfield_local_results"
    ad = glob.glob(res + "/allen_aligned_affine8v1")
    if not ad:
        return None
    try:
        svt = np.load(res + "/SVT.npy")
        T = np.load(res + "/T.npy").astype(np.float64)
    except Exception as ex:                                          # noqa: BLE001
        print("  !! " + lab + ": " + type(ex).__name__ + " " + str(ex)[:60], flush=True)
        return None

    from wfield_local import hemo_variants as hv

    masks, fracs = {}, {}
    need = [m for m in modes if m in MASK_SPEC]
    if need:
        cue = _lc(s["h5"])
        lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        _cf, _lf, csmp = _fr(s, cue, lk)
        if csmp is None:      # regime A has no corrected-frame map -> cannot build a mask
            modes = [m for m in modes if m not in MASK_SPEC]
        else:
            for m in need:
                masks[m], d = fit_mask(s, svt[:, FUNC::2].shape[1], csmp, cue, **MASK_SPEC[m])
                fracs[m] = d["frac_final"]
            print("  {:12s} mask eligible: ".format(lab)
                  + "  ".join("{} {:.1f}%".format(m, 100 * fracs[m]) for m in need), flush=True)

    out = {"label": lab, "mask_frac": fracs, "refit_t": bool(refit_t)}
    for mode in modes:
        if refit_t:
            # PRODUCT path: coefficients refitted on the drift-removed traces. Reusing the saved T
            # (below) is right for isolating the filter and wrong for anything we would adopt.
            svtc, _T, _rc, _meta = hv.compute(s_sess, mode, refit_t=True, win_s=win_s, verbose=False)
        else:
            svtc = svtcorr(svt, T, mode, mask=masks.get(mode), win_s=win_s)
        sig, regs = roi_signal(ad[0], svtc)
        r, lvl = patterns(s, sig)
        out[mode] = {"precue": decode(s, sig, regs, "precue"),
                     "postcue": decode(s, sig, regs, "cue"),
                     "corr_pre_post": r, "pre_level": lvl}
        del sig
        d = out[mode]
        print("  {:12s} {:12s} PRE {:.3f}  post {:.3f}  corr(pre,post) {:+.3f}".format(
            lab, mode, d["precue"], d["postcue"], d["corr_pre_post"]), flush=True)
    return out


def _summary(rows, modes):
    """Paired comparison against the zerophase baseline, with a Wilcoxon signed-rank test.

    PAIRED because every mode saw the identical session, trials and folds -- the only thing that
    differs is the drift removal, so the per-session difference is the estimate and the spread across
    animals is not noise to be averaged away but the thing being tested.
    """
    from scipy.stats import wilcoxon
    print("\n=== {} sessions (chance {:.3f}) ===".format(len(rows), 1 / 6))
    print("{:13s} {:>9s} {:>9s} {:>8s}   vs zerophase (pre-cue)".format(
        "mode", "PRE-CUE", "post-cue", "corr"))
    base = np.array([r["zerophase"]["precue"] for r in rows], float) if "zerophase" in modes else None
    for m in modes:
        pre = np.array([r[m]["precue"] for r in rows], float)
        post = np.array([r[m]["postcue"] for r in rows], float)
        cor = np.array([r[m]["corr_pre_post"] for r in rows], float)
        extra = ""
        if base is not None and m != "zerophase":
            d = pre - base
            ok = np.isfinite(d)
            try:
                pv = wilcoxon(d[ok]).pvalue
            except Exception:                                        # noqa: BLE001
                pv = float("nan")
            extra = "   {:+.3f}   lower in {}/{}   p={:.2g}".format(
                np.nanmean(d), int((d[ok] < 0).sum()), int(ok.sum()), pv)
        print("{:13s} {:9.3f} {:9.3f} {:+8.3f}{}".format(
            m, np.nanmean(pre), np.nanmean(post), np.nanmean(cor), extra))

    if "zerophase" in modes:
        neg = np.array([r["zerophase"]["corr_pre_post"] for r in rows], float)
        print("\nSIGN TEST: corr(pre,post) NEGATIVE under zerophase in {}/{} sessions".format(
            int((neg < 0).sum()), int(np.isfinite(neg).sum())))
        for m in modes:
            if m == "zerophase":
                continue
            c = np.array([r[m]["corr_pre_post"] for r in rows], float)
            print("           under {:12s} negative in {}/{}   mean {:+.3f}".format(
                m, int((c < 0).sum()), int(np.isfinite(c).sum()), np.nanmean(c)))

    # per animal, because the cohort is NOT uniform and an average would hide that
    print("\n=== per animal, pre-cue (chance 0.167) ===")
    animals = sorted({r["label"][:4] for r in rows})
    print("{:6s} {:>4s}".format("", "n") + "".join("{:>13s}".format(m) for m in modes))
    for a in animals:
        rr = [r for r in rows if r["label"].startswith(a)]
        print("{:6s} {:4d}".format(a, len(rr))
              + "".join("{:13.3f}".format(np.nanmean([x[m]["precue"] for x in rr])) for m in modes))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="explicit labels (default: --from crossed with --animals)")
    ap.add_argument("--from", dest="from_dates", default=None, help="date spec (default: curated set)")
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--modes", default="zerophase,fitonly,taskdetrend")
    ap.add_argument("--win-s", type=float, default=DETREND_WIN_S)
    ap.add_argument("--refit-t", action="store_true",
                    help="refit the hemodynamic coefficients on the drift-removed traces "
                         "(the PRODUCT path). Default reuses the saved T, which isolates the filter "
                         "and is the correct choice for a controlled comparison.")
    ap.add_argument("--output", default=None, help="write the per-session results JSON here")
    args = ap.parse_args(argv)
    modes = args.modes.split(",")
    if args.sessions:
        labs = list(args.sessions)
    else:
        dates = (set(config.expand_dates(args.from_dates, width=4)) if args.from_dates
                 else set(config.curated_dates()))
        only = config.normalize_animals(args.animals) or sorted({x["label"][:4] for x in SESSIONS})
        labs = [x["label"] for x in SESSIONS
                if x["label"][:4] in set(only) and x["label"][-4:] in dates]
    print("[filter_test] {} sessions x {} modes: {}".format(len(labs), len(modes), modes), flush=True)

    rows = []
    for lab in labs:
        try:
            r = analyse_session(lab, list(modes), win_s=args.win_s, refit_t=args.refit_t)
        except Exception as ex:                                      # noqa: BLE001
            print("  !! " + lab + ": " + type(ex).__name__ + " " + str(ex)[:80], flush=True)
            continue
        if r and all(m in r for m in modes):
            rows.append(r)
    if not rows:
        print("no sessions analysed")
        return 1
    _summary(rows, modes)
    if args.output:
        import json
        from pathlib import Path
        Path(args.output).write_text(json.dumps(rows, indent=2, default=float))
        print("\nwrote " + args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
