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
    LAG                  **THIS MEASURE WAS WRONG TWICE, AND BOTH MISTAKES ARE WORTH KEEPING.**

                         v1 cross-correlated 470 against 415 and returned -0.06 to 0.00 s in every
                         session. Not a failed measurement -- the correct answer to the wrong
                         question: **415 and 470 BOTH SEE THE SAME BLOOD AT THE SAME INSTANT**, so
                         the haemoglobin term is simultaneous in the two channels. The
                         haemodynamic lag is between NEURAL ACTIVITY and BLOOD, not between two
                         optical channels that both report blood. So the pair became `SVTcorr`
                         (calcium, hemo nulled) against 415.

                         v2 kept `argmax` and returned -0.26 to -0.48 s in every session: 415
                         LEADING calcium, which no haemodynamic response can do. **PLOTTING THE
                         WHOLE CURVE INSTEAD OF ITS ARGMAX SETTLED IT IN ONE LOOK.** The function
                         is BIPHASIC -- a positive lobe near -0.45 s and a deeper negative lobe
                         near +0.48 s -- and `argmax` was reporting the wrong one. Blood ABSORBS,
                         so it enters a dF/F fluorescence trace with a NEGATIVE sign, and the HRF
                         is therefore the TROUGH. A common sign flip on both channels is invisible
                         to any zero-lag correlation, which is why nothing earlier caught it.

                         The trough is also the STABLE lobe: +0.45 to +0.58 s across PS92, PS93
                         and PS94 and across both hemispheres, while the peak `argmax` was reading
                         wandered from -0.19 to -0.45 s over the same six measurements.

    RAW-PAIR TILT        `r(470, 415)` at +0.5 s minus the same at -0.5 s, which needs no `T` at
                         all -- see `_asym`. -0.20 to -0.34 in all six, same physics, no free
                         parameter and no regression-induced notch.

THREE THINGS THE LAG NEEDS TO BE MEANINGFUL. The traces are BANDPASSED 0.05-2 Hz with `filtfilt`
first, because a shared slow drift otherwise dominates the cross-correlation and pins the peak at
lag 0 with an enormous width -- and `filtfilt` is ZERO-PHASE, so it cannot introduce a differential
lag between the channels. The EXTREMUM MUST BE CHOSEN BY PHYSICS, not by `argmax`, for the reason
above. And the ABSOLUTE lag is not interpretable: `SVTcorr` is orthogonal to 415 at lag zero BY
CONSTRUCTION, so the regression itself notches the curve and pushes both lobes outward, and
`470 = C - H` still contains the haemodynamic term. **A CHANGE in lag across epochs is interpretable
even when the absolute value is not**, which is the same argument that licenses every other
biased-but-stable measure in this analysis.

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
ASYM_S = 0.5              # s; where the raw-pair asymmetry is read, near the measured trough
MIN_REST_S = 60.0
N_BOOT = 4000


def _bandpass(x, fs):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, x, padlen=min(3 * max(len(b), len(a)), x.size - 1))


def _xcorr_curve(a, b, fs, max_lag_s=MAX_LAG_S):
    """``(lags_s, r)`` for `b` relative to `a`. POSITIVE lag = b LAGS a.

    Normalised so the values are correlations rather than covariances, and evaluated only within
    +/- `max_lag_s`. **RETURN THE WHOLE CURVE, NOT AN EXTREMUM** -- the curve of interest here is
    BIPHASIC, and any single-extremum summary of a biphasic function is a choice that has to be
    made explicitly rather than by `argmax`.
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
    return lags / fs, out


def _hrf_lag(calcium, iso, fs, max_lag_s=MAX_LAG_S):
    """``(lag_s at the TROUGH, its depth as a positive number)``. POSITIVE lag = blood FOLLOWS
    calcium, which is the HRF.

    **THE TROUGH, NOT THE PEAK, AND THE REASON IS PHOTOMETRIC RATHER THAN STATISTICAL.** More
    haemoglobin absorbs more light, so blood volume enters a dF/F fluorescence trace with a
    NEGATIVE sign -- in BOTH channels, which is why the correction works at all, and which the
    zero-lag correlation cannot reveal because a common sign flip on both traces cancels. So the
    haemodynamic response to a calcium transient appears in `r(SVTcorr, 415)` as a NEGATIVE lobe
    at POSITIVE lag. `argmax` reads the other lobe and reports the HRF with its sign reversed.
    """
    lags, o = _xcorr_curve(calcium, -np.asarray(iso, np.float64), fs, max_lag_s)
    k = int(np.argmax(o))
    return float(lags[k]), float(o[k])


def _asym(f470, iso, fs, at_s=ASYM_S):
    """``r(+at_s) - r(-at_s)`` on the RAW 470/415 pair. NEGATIVE = blood follows calcium.

    **THE VALUE OF THIS ONE IS THAT IT DOES NOT TOUCH `T`.** `SVTcorr` is orthogonal to 415 at lag
    zero by construction, so the notch that splits its cross-correlation into two lobes is put
    there by the regression, and both lobe POSITIONS are therefore part signal and part fitting
    artefact. The raw pair has no notch. With `470 = C - H` and `415 = -kH`, the calcium term
    enters `r(470, 415)` with a MINUS sign at the haemodynamic delay, so an HRF tilts the raw
    cross-correlation to the LEFT and this difference goes negative. Same physics, no free
    parameter.
    """
    lags, o = _xcorr_curve(f470, iso, fs, at_s + 0.5)
    jp = int(np.argmin(np.abs(lags - at_s)))
    jm = int(np.argmin(np.abs(lags + at_s)))
    return float(o[jp] - o[jm])


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
        # THE LAG PAIRS CALCIUM AGAINST HAEMODYNAMICS, NOT 470 AGAINST 415, and it reads the
        # TROUGH -- see `_hrf_lag`. Blood absorbs, so it enters both dF/F traces NEGATIVELY, and
        # the HRF is therefore a negative lobe at positive lag. `_asym` measures the same tilt on
        # the raw pair, where no regression has put a notch at zero.
        c = _bandpass(op @ dcorr[:, :n], FS)[rest] if dcorr is not None else None
        lag, depth = (_hrf_lag(c, b, FS) if c is not None and c.std() > 0
                      else (float("nan"), float("nan")))
        out.append(dict(label=s["label"], side=side, n_rest_s=float(a.size / FS),
                        r_zero_lag=float(np.corrcoef(a, b)[0, 1]),
                        amp_ratio=float(b.std() / (a.std() or np.nan)),
                        lag_s=lag, r_at_trough=depth, asym_470_415=_asym(a, b, FS)))
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
                       ("lag_s", "HRF LAG, TROUGH OF r(SVTcorr, 415) (s; POSITIVE = blood FOLLOWS"
                        " calcium)"),
                       ("asym_470_415", "RAW-PAIR TILT r(+0.5s) - r(-0.5s) (NEGATIVE = blood"
                        " follows calcium; NO `T`)")):
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
    print(f"  {'epoch':<10}{'d r(415,470)':>24}{'d amp ratio':>24}{'d lag (s)':>24}"
          f"{'d raw tilt':>24}")
    for e in eps:
        line = f"  {e:<10}"
        for key in ("r_zero_lag", "amp_ratio", "lag_s", "asym_470_415"):
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
    print("\n  AND THE ABSOLUTE LAG IS NOT INTERPRETABLE. `SVTcorr` is orthogonal to 415 at lag")
    print("  zero BY CONSTRUCTION, so the regression itself notches the curve and shifts both")
    print("  lobes outward; 470 also still contains the haemodynamic term. A CHANGE across")
    print("  epochs is interpretable, and `asym_470_415` is the version with no `T` in it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
