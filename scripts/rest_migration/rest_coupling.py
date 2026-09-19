"""415/470 coupling DURING REST, by hemisphere, with a LAG. The three extensions at once.

Priya, 2026-09-19: *"sure let's try rest and L/R"* and *"and latency"*. They share all their data
loading, so they are one module.

WHY REST. Everything measured so far is task-evoked, which inherits the engagement gate, the
uneven-sampling-in-time correction, and a far-contralateral acute cell holding **62 trials across
12 sessions**. Rest inherits none of that and is abundant.

**BUT NOT FOR THE REASON I FIRST GAVE, and the correction matters for how much to expect.** I said
the task analysis was trial-limited at ~100 trials per position. It is not -- those are POOLED, and
pre-stroke runs ~3,650 trials per position per arm. **THE BINDING CONSTRAINT IS FOUR ANIMALS**
(three in chronic; PS94 has none), because the bootstrap resamples animals first. More samples per
session therefore do NOT narrow an animal-limited CI much. Rest buys cleaner cells and removes the
behavioural confounds; it does not buy power against between-animal variance.

**WHICH IS WHY THE L/R CONTRAST IS THE STRONGER OF THE TWO.** It is WITHIN animal and WITHIN
session, so each animal is its own control and the n=4 bottleneck does not apply to it. Every animal
in this cohort is lesioned on the LEFT (`epoch_figures.anatomical_labels`, which refuses to guess
and would raise on a mixed cohort), so left = IPSILESIONAL. A coupling change caused by a focal
lesion should be larger ipsilesionally; a change in both hemispheres equally is systemic or
instrumental.

THE THREE MEASURES, all computed on REST frames only, per hemisphere:

    r(415, 470)          TEMPORAL co-fluctuation -- the natural rest analogue of the spatial
                         correlation the task version used
    SD(415) / SD(470)    amplitude ratio, the rest analogue of the coupling gain
    LAG at peak x-corr   **and the FIRST VERSION OF THIS MEASURE WAS CONCEPTUALLY WRONG.** It
                         cross-correlated 470 against 415 and returned lags of -0.06 to 0.00 s --
                         zero to two frames -- across every session. That is not a failed
                         measurement, it is the correct answer to the wrong question: **415 and 470
                         BOTH SEE THE SAME BLOOD AT THE SAME INSTANT.** The haemoglobin absorbing
                         them is one physical quantity, so its contribution is SIMULTANEOUS in the
                         two channels and their cross-correlation must peak at zero.

                         The haemodynamic lag is between NEURAL ACTIVITY and BLOOD, not between two
                         optical channels that both report blood. So the pair has to be
                         `SVTcorr` (calcium, hemo removed) against 415 (haemodynamic), and a
                         positive peak lag then means blood FOLLOWS calcium, which is the HRF.
                         `SVTcorr = 470 - T*415` so the two are not independent -- but T is fitted
                         to null the haemodynamic term, which suppresses the zero-lag component
                         specifically and leaves the LAGGED relationship visible.

TWO THINGS THE LAG NEEDS TO BE MEANINGFUL. The traces are BANDPASSED 0.05-2 Hz with `filtfilt`
first, because a shared slow drift otherwise dominates the cross-correlation and pins the peak at
lag 0 with an enormous width -- and `filtfilt` is ZERO-PHASE, so it cannot introduce a differential
lag between the channels. And the ABSOLUTE lag is not interpretable: `470 = C + H` contains the
haemodynamic term too, which smears the peak toward zero. **A CHANGE in lag across epochs is
interpretable even when the absolute value is not**, which is the same argument that licenses every
other biased-but-stable measure in this analysis.

    python -m scripts.rest_migration.rest_coupling [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

BAND = (0.05, 2.0)          # Hz, zero-phase; isolates the haemodynamic band from drift and noise
MAX_LAG_S = 4.0
MIN_REST_S = 60.0
N_BOOT = 4000


def _bandpass(x, fs):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, x, padlen=min(3 * max(len(b), len(a)), x.size - 1))


def _xcorr_lag(a, b, fs, max_lag_s=MAX_LAG_S):
    """``(lag_s at peak, peak r)`` for `b` relative to `a`. POSITIVE = b LAGS a.

    Normalised so the peak is a correlation rather than a covariance, and searched only within
    +/- `max_lag_s` -- an unbounded search on a bandpassed pair finds a side lobe as often as the
    true peak.
    """
    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)
    n = a.size
    m = int(round(max_lag_s * fs))
    lags = np.arange(-m, m + 1)
    out = np.empty(lags.size)
    for i, L in enumerate(lags):
        if L < 0:
            out[i] = float(np.dot(a[-L:], b[:n + L])) / (n - abs(L))
        elif L > 0:
            out[i] = float(np.dot(a[:n - L], b[L:])) / (n - abs(L))
        else:
            out[i] = float(np.dot(a, b)) / n
    k = int(np.argmax(out))
    return float(lags[k] / fs), float(out[k])


def session_rest_coupling(s):
    """One row per hemisphere for this session, or ``None`` if it has too little rest."""
    from wfield_local import config
    from wfield_local.hemo_variants import FS, functional_channel
    from wfield_local.hemispheric_intensity import hemisphere_masks
    from wfield_local.rest_by_position import _session_daq

    res = Path(s["mc"]) / "wfield_local_results"
    allen = res / "allen_aligned_affine8v1"
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    fc = functional_channel(s)
    d470 = np.asarray(svt[:, fc::2], np.float64)
    d415 = np.asarray(svt[:, (fc + 1) % 2::2], np.float64)
    try:
        dcorr = np.asarray(np.load(config.svtcorr_in(res), mmap_mode="r"), np.float64)
    except Exception:                                                # noqa: BLE001
        dcorr = None

    U = np.load(allen / "U_atlas.npy", mmap_mode="r")
    left, right, groups = hemisphere_masks(allen)
    sm = groups.get("SM")
    rest_mask, _cs, _c, _ts, fs_samp, _sy = _session_daq(s)

    # REST FRAMES: the sample-resolution rest mask, reduced to frames by the frame->sample map.
    f_of = np.asarray(fs_samp)
    n = min(d470.shape[1], d415.shape[1], f_of.size)
    idx = np.clip(f_of[:n].astype(np.int64), 0, len(rest_mask) - 1)
    rest = np.asarray(rest_mask, bool)[idx]
    if rest.sum() < MIN_REST_S * FS:
        return None

    out = []
    for side, mask in (("ipsi", left), ("contra", right),
                       ("ipsi_SM", sm[0] if sm else None),
                       ("contra_SM", sm[1] if sm else None)):
        if mask is None or mask.sum() < 200:
            continue
        op = np.asarray(U, np.float64)[mask].mean(0)          # brain-mean operator for this ROI
        a = _bandpass(op @ d470[:, :n], FS)[rest]
        b = _bandpass(op @ d415[:, :n], FS)[rest]
        if a.size < MIN_REST_S * FS or a.std() == 0 or b.std() == 0:
            continue
        # THE LAG PAIRS CALCIUM AGAINST HAEMODYNAMICS, NOT 470 AGAINST 415. Both optical channels
        # see the SAME blood at the SAME instant, so r(470, 415) must peak at zero -- measured at
        # -0.06 to 0.00 s in every smoke-test session, which is the right answer to the wrong
        # question. `SVTcorr` has the haemodynamic term nulled, so a positive peak lag against 415
        # means blood FOLLOWS calcium: the HRF.
        c = _bandpass(op @ dcorr[:, :n], FS)[rest] if dcorr is not None else None
        lag, peak = (_xcorr_lag(c, b, FS) if c is not None and c.std() > 0
                     else (float("nan"), float("nan")))
        lag470, _p470 = _xcorr_lag(a, b, FS)     # kept as the ZERO-LAG SANITY CHECK, not a result
        out.append(dict(label=s["label"], side=side, n_rest_s=float(a.size / FS),
                        r_zero_lag=float(np.corrcoef(a, b)[0, 1]),
                        amp_ratio=float(b.std() / (a.std() or np.nan)),
                        lag_s=lag, r_at_peak=peak, lag_470_vs_415=lag470))
    return out or None


def _boot(by_animal, rng, n_boot=N_BOOT):
    animals = sorted(by_animal)
    if not animals:
        return None
    flat = [v for a in animals for v in by_animal[a]]
    o = []
    for _ in range(n_boot):
        vals = []
        for a in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            sa = by_animal[a]
            vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        if vals:
            o.append(float(np.mean(vals)))
    if len(o) < n_boot // 4:
        return None
    o = np.asarray(o)
    return float(np.mean(flat)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    rows = []
    for s in config.load_sessions():
        lab = s["label"]
        if lab not in want or (a.animals and config.animal_of(lab) not in a.animals):
            continue
        ep = epochs.epoch_of(lab)
        if not ep:
            continue
        try:
            got = session_rest_coupling(s)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if not got:
            print(f"  .. {lab}: too little rest -- skipped", flush=True)
            continue
        for r in got:
            r.update(animal=config.animal_of(lab), epoch=ep)
        rows += got
        w = {r["side"]: r for r in got}
        print(f"   {lab:14s} {ep:9s} rest {got[0]['n_rest_s'] / 60:.0f} min   "
              + "  ".join(f"{k} r={v['r_zero_lag']:+.2f} lag={v['lag_s']:+.2f}s"
                          for k, v in w.items() if k in ("ipsi", "contra")), flush=True)

    if not rows:
        print("no sessions -- a failed run, not a result")
        return 1
    q = out_dir / "epoch_19_rest_coupling.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {q}")

    rng = np.random.default_rng(a.seed)
    eps = [e for e in ("pre", "acute", "subacute", "chronic") if any(r["epoch"] == e for r in rows)]
    sides = [s_ for s_ in ("ipsi", "contra", "ipsi_SM", "contra_SM")
             if any(r["side"] == s_ for r in rows)]
    bar = "=" * 96

    for key, title in (("r_zero_lag", "TEMPORAL r(415, 470) DURING REST"),
                       ("amp_ratio", "AMPLITUDE RATIO SD(415)/SD(470) DURING REST"),
                       ("lag_s", "LAG OF PEAK CROSS-CORRELATION (s; POSITIVE = 415 LAGS 470)")):
        print(f"\n{bar}\n{title}\n{bar}")
        print(f"  {'epoch':<10}" + "".join(f"{s_:>22}" for s_ in sides))
        for e in eps:
            line = f"  {e:<10}"
            for s_ in sides:
                d = defaultdict(list)
                for r in rows:
                    if r["epoch"] == e and r["side"] == s_ and np.isfinite(r[key]):
                        d[r["animal"]].append(float(r[key]))
                g = _boot(d, rng)
                line += f"{g[0]:>11.3f} [{g[1]:+.2f},{g[2]:+.2f}]" if g else f"{'--':>22}"
            print(line)

    # THE WITHIN-ANIMAL, WITHIN-SESSION CONTRAST -- the one that escapes the n=4 bottleneck.
    print(f"\n{bar}\nIPSILESIONAL minus CONTRALESIONAL, paired within session\n{bar}")
    print(f"  {'epoch':<10}{'d r(415,470)':>24}{'d amp ratio':>24}{'d lag (s)':>24}")
    for e in eps:
        line = f"  {e:<10}"
        for key in ("r_zero_lag", "amp_ratio", "lag_s"):
            d = defaultdict(list)
            for lab in {r["label"] for r in rows if r["epoch"] == e}:
                got = {r["side"]: r for r in rows if r["label"] == lab}
                if "ipsi" in got and "contra" in got:
                    v = float(got["ipsi"][key]) - float(got["contra"][key])
                    if np.isfinite(v):
                        d[got["ipsi"]["animal"]].append(v)
            g = _boot(d, rng)
            star = " *" if g and (g[1] > 0 or g[2] < 0) else "  "
            line += (f"{g[0]:>+11.3f} [{g[1]:+.2f},{g[2]:+.2f}]{star}" if g else f"{'--':>24}")
        print(line)
    print("\n  * = 95% CI excludes zero. Every animal is lesioned on the LEFT, so ipsi = left.")
    print("  A focal lesion should move the IPSI side more. Both sides moving together is")
    print("  systemic or instrumental, not a consequence of the infarct.")
    print("\n  AND THE ABSOLUTE LAG IS NOT INTERPRETABLE -- 470 contains the haemodynamic term too,")
    print("  which smears the peak toward zero. A CHANGE in lag across epochs is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
