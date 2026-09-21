"""Cue-evoked RAW 415 and 470, at frame resolution -- task-evoked haemodynamics, and a test of
whether 415 is isosbestic for THIS cohort.

TWO QUESTIONS, ONE MEASUREMENT.

1. TASK-EVOKED COUPLING (Priya, 2026-09-19): is there less cue- or lick-evoked haemodynamic
   response after the stroke? The existing `hemo_map_control` answer is a SPATIAL correlation
   between maps, which is attenuation-prone and says nothing about amplitude. This is the
   amplitude, per epoch.

2. IS 415 ISOSBESTIC HERE? The literature disagrees on where GCaMP's neutral/anionic crossing
   sits. THREE ESTIMATES, and only the folk one puts it at or below our 415:

       conventional photometry practice     405-415
       Simpson et al. 2024, Neuron primer   420-430 for GCaMP6   (PMC10939905, Table 2)
       Barnett/Drobizhev 2017, GCaMP6m      440-450, and NO true isosbestic point

   Below the crossing calcium DECREASES fluorescence; above it increases. All three agree on
   that much, and two of the three put our 415 BELOW it.

   SIMPSON ET AL. ALSO NAME THE OBSERVABLE, from a 405 nm control against GCaMP6f: "the
   isosbestic control signal has significant negative bleed-through of the GCaMP signal, due to
   405 nm excitation not exactly matching the isosbestic point for GCaMP6f. This is evident as a
   NEGATIVE PEAK IN THE EVENT-ALIGNED AVERAGE." That is this module's statistic, described by
   someone who observed it -- and at 405 rather than 415, so it bounds the leak from above.

   **TIMING SETTLES IT AND AMPLITUDE CANNOT.** Calcium is fast (hundreds of ms); haemodynamics is
   slow (peaks 1-2 s, lasts seconds). So:

       415 below the crossing  ->  EARLY NEGATIVE deflection, then the slow positive one
       415 flat (isosbestic)   ->  no early dip, only the slow rise

   The +0.54% to +1.96% cue-evoked 415 rises already in DECISIONS cannot settle this: they are
   WINDOW AVERAGES over the whole post-cue period, which average an early dip away completely.
   That is precisely why they looked like a clean positive haemodynamic response.

THE NORMALISATION TRAP THIS MODULE EXISTS TO AVOID, and it has already been paid once here
(DECISIONS, 2026-08-18: a check reported +692% and -2881% evoked responses). `U @ SVT`
reconstructs the DEVIATION from each channel's mean -- the reconstructed means are ZERO -- so
dividing by them is division by ~0.

AND THERE IS AN OVER-CORRECTION FOR IT, WHICH THIS MODULE ALSO PAID. `U @ SVT` in this pipeline
is ALREADY FRACTIONAL, so the fix is not "divide by a different mean" -- it is DO NOT DIVIDE.
Dividing by `frames_average.npy` (raw camera counts, ~1.2e4) was the first version here and it is
wrong by four orders of magnitude; the output was all zeros to three decimals. See `session_traces`.

CHANNEL IDENTITY IS DERIVED, NOT ASSUMED, for the same reason: `SVTcorr` is the corrected BLUE
channel, so whichever half of `SVT` it correlates with IS blue. A docstring is not evidence.

PER-TRIAL BASELINE, so slow drift never enters. Each trial is expressed against its own pre-cue
window, which is what makes the raw (undetrended) channels usable here -- and the drift variants
are irrelevant to an event-triggered average on that baseline.

    python -m scripts.rest_migration.nvc_evoked [--animals PS92 ...] [--align cue|lick]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

#: THE BOOTSTRAP LIVES IN ONE PLACE NOW. `analysis_kit` holds the nested animals->sessions
#: draw this module used to define for itself, bit-for-bit -- `tests/test_analysis_kit.py`
#: pins it against the pre-extraction source. Read that module before touching a draw: the
#: point-estimate convention DIFFERS between `boot_ci` (flat pool, for LEVELS) and
#: `boot_delta` (animal-weighted, for CHANGES), and the difference has retracted a result.
from wfield_local import analysis_kit as ak

PRE_S, POST_S = 1.0, 4.0          # window around the event
BASE_S = (-1.0, -0.2)             # per-trial baseline, ending before any response
EARLY_S = (0.0, 0.4)              # calcium timescale
LATE_S = (1.0, 3.0)               # haemodynamic timescale


MIN_TRIALS = 8                    # below this a session mean is noise, not a small sample
LICK_RESP_S = 2.0                 # a lick this soon after the cue makes the trial a LICK trial


def session_traces(s):
    """``(t, pct470, pct415, cue_frames, lick_frames)`` -- brain-mean % traces and event frames."""
    from wfield_local.hemo_variants import FS, functional_channel
    from scripts.rest_migration.plot_session_residual import _brain_mean_op

    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    # PER-SESSION, NOT THE COHORT CONSTANT: PS92_0828 is 0, and it is in the curated post set.
    #
    # AND NOT THE CORRELATION HEURISTIC THIS MODULE USED UNTIL 2026-09-19, which picked blue as
    # whichever half `SVTcorr` component 0 tracked. That is unsound, not merely noisy:
    # `SVTcorr = blue - T @ other`, so where the haemodynamic term dominates component 0 the
    # corrected trace genuinely resembles the CONTROL half more. Measured margins ran from 0.030
    # (a coin flip, PS93_0606) to a confident 0.071-vs-0.370 disagreement (PS92_0824), and in
    # `channel_position_maps` it swapped a session's channels and produced a figure that looked
    # like a finding rather than a fault.
    fc = functional_channel(s)
    a0 = svt[:, fc::2].astype(np.float64)                # functional (470)
    b0 = svt[:, (fc + 1) % 2::2].astype(np.float64)      # the 415 control

    u_mean, _npix = _brain_mean_op(s["mc"])
    d470 = u_mean @ a0
    d415 = u_mean @ b0
    # NO DENOMINATOR. `U @ SVT` is ALREADY FRACTIONAL in this pipeline -- `plot_drift_estimators`
    # reports these traces with SD ~0.028 and the 2026-08-18 cue-triggered check reported +3.69%
    # for the same quantity, so x100 IS the percentage. Dividing by `frames_average` (raw camera
    # counts, ~1.2e4) was my first version and it is wrong by four orders of magnitude -- an
    # OVER-correction for the documented trap, which was dividing by the RECONSTRUCTED mean
    # (identically zero), not by any mean at all.
    pct470 = 100.0 * d470
    pct415 = 100.0 * d415

    from wfield_local.rest_by_position import _session_daq
    _rest, cs, _codes, _ts, fs_samp, _sync = _session_daq(s)
    n = d470.size
    f_of = np.asarray(fs_samp)[:n]
    cue_frames = np.searchsorted(f_of, np.asarray(cs))
    cue_frames = cue_frames[(cue_frames > int(PRE_S * FS) + 1)
                            & (cue_frames < n - int(POST_S * FS) - 1)]
    # LICKS ON THE SAME FRAME CLOCK, by the same searchsorted, with the pipeline's canonical
    # detector parameters (`nolick_decoder` calls it with exactly these).
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    lick_frames = np.searchsorted(f_of, np.asarray(lk["lick_samples"]))
    t = (np.arange(-int(PRE_S * FS), int(POST_S * FS)) / FS)
    return t, pct470, pct415, cue_frames, lick_frames


def split_by_licking(cue_frames, lick_frames):
    """``(lick_trials, nolick_trials)`` cue frames.

    THE TWO CLASSES ARE NOT COMPLEMENTS, DELIBERATELY. `nolick` requires ZERO licks anywhere in
    the WHOLE analysis window [-1, +4] s, not merely none in the response window, because the
    point of the class is to remove the movement-locked component from the trace -- a lick at
    +3 s contaminates the late window just as effectively as one at +0.3 s contaminates the
    early one. Trials that lick only late are therefore in NEITHER class, and that is correct:
    they are neither clean nor comparable.

    `nolick` also excludes trials with a PRE-CUE lick, since the baseline window is [-1, -0.2].
    """
    from wfield_local.hemo_variants import FS

    lf = np.sort(np.asarray(lick_frames, np.int64))
    lo, hi = int(PRE_S * FS), int(POST_S * FS)
    resp = int(LICK_RESP_S * FS)
    licked, clean = [], []
    for f in cue_frames:
        if np.searchsorted(lf, f + resp) - np.searchsorted(lf, f) > 0:
            licked.append(f)
        elif np.searchsorted(lf, f + hi) - np.searchsorted(lf, f - lo) == 0:
            clean.append(f)
    return np.asarray(licked, np.int64), np.asarray(clean, np.int64)


def evoked(trace, cue_frames, t):
    """Per-trial-baselined event-triggered average of `trace`."""
    from wfield_local.hemo_variants import FS

    lo, hi = -int(PRE_S * FS), int(POST_S * FS)
    seg = []
    for f in cue_frames:
        w = trace[f + lo:f + hi]
        if w.size != (hi - lo):
            continue
        b = w[(t >= BASE_S[0]) & (t < BASE_S[1])]
        if b.size:
            seg.append(w - b.mean())
    return (np.mean(seg, axis=0), len(seg)) if seg else (None, 0)


def session_row(item):
    """One session's evoked table row and its two epoch-pooled curves. Module level, so
    `parallel.fan_out` can pickle it by name (CLAUDE.md ground rule 6).

    **RETURNS ``(row, curves, tvec, message)``; IT NEITHER ACCUMULATES NOR PRINTS.** The parent
    extends `rows` and `curves` and prints in `input_order`, so both the CSV and the pooled curves
    are in exactly the order the serial loop produced them -- and the bootstrap below draws over
    those lists, so any other order moves the CIs while leaving the point estimates exact.
    """
    lab = item
    from wfield_local import config, epochs

    s = next((x for x in config.load_sessions() if x["label"] == lab), None)
    if s is None:
        return None, [], None, None
    ep = epochs.epoch_of(lab)
    try:
        t, p470, p415, cf, lf = session_traces(s)
    except Exception as ex:                                          # noqa: BLE001
        return None, [], None, f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}"
    licked, clean = split_by_licking(cf, lf)
    an = config.animal_of(lab)
    w = lambda e, lo, hi: float(np.mean(e[(t >= lo) & (t < hi)]))   # noqa: E731
    r = dict(label=lab, animal=an, epoch=ep, n_lick=int(licked.size), n_nolick=int(clean.size))
    got = []
    for cls, frames in (("lick", licked), ("nolick", clean)):
        e470, n1 = evoked(p470, frames, t)
        e415, _n2 = evoked(p415, frames, t)
        if e470 is None or e415 is None or n1 < MIN_TRIALS:
            for k in ("470_early", "470_late", "415_early", "415_late"):
                r[f"{cls}_{k}"] = ""
            continue
        got.append(((cls, ep), (an, e470, e415)))
        r[f"{cls}_470_early"] = round(w(e470, *EARLY_S), 4)
        r[f"{cls}_470_late"] = round(w(e470, *LATE_S), 4)
        r[f"{cls}_415_early"] = round(w(e415, *EARLY_S), 4)
        r[f"{cls}_415_late"] = round(w(e415, *LATE_S), 4)
    return r, got, t, (f"   {lab:14s} {ep:9s} lick n={r['n_lick']:4d}  "
                       f"nolick n={r['n_nolick']:4d}   "
                       f"nolick 415 early {r.get('nolick_415_early', '--')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=None,
                    help="sessions in parallel (default cores-2 capped at 8; 1 for serial)")
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args(argv)

    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    animals = a.animals or ["PS92", "PS93", "PS94", "PS95"]
    rows, curves, tvec = [], defaultdict(list), None
    # `analysis_kit.curated_labels` is the session filter, once; it preserves `load_sessions`
    # order, which is NOT sorted.
    labels = ak.curated_labels(animals)

    # **`input_order`, NOT ALPHABETICAL** (CLAUDE.md ground rule 9). `rows` and `curves` are
    # iterated into a seeded RNG below, so collecting the fan-out in sorted order would move every
    # CI while leaving the point estimates exact. Restoring the input order makes this conversion
    # diff-identical to the serial run.
    res, fail = ak.fan_sessions(labels, session_row, jobs=a.jobs,
                                key=ak.input_order(labels))
    if fail:
        print(f"  !! {len(fail)} session(s) failed: "
              + ", ".join(f"{x[0]} ({x[1][:40]})" for x in fail[:4]), flush=True)
    # THE PARENT ACCUMULATES AND THE PARENT PRINTS, in input order -- see `session_row`.
    for _lab, (r, got, tv, msg) in res:
        if msg:
            print(msg, flush=True)
        if r is None:
            continue
        tvec = tv
        for key, entry in got:
            curves[key].append(entry)
        rows.append(r)

    if not rows:
        print("no sessions -- a failed run, not a result")
        return 1
    q = out_dir / "epoch_16_nvc_evoked.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {q}")

    # THE TEST IS THE NO-LICK EARLY 415 ALONE. On LICK trials a fast movement-locked term rides on
    # both channels and can bury a smaller negative calcium term; removing those trials is the
    # whole point, so the lick rows are shown for contrast and are not the statistic.
    rng = np.random.default_rng(a.seed)
    print(f"\n{'=' * 92}\nIS 415 ISOSBESTIC HERE? -- nested animals->sessions CI, "
          f"{EARLY_S[0]:.1f}-{EARLY_S[1]:.1f} s\n{'=' * 92}")
    print(f"  {'class':<8}{'epoch':<10}{'n':>4}{'470 early':>27}{'415 early':>27}{'ratio':>8}")
    stats = []
    for cls in ("lick", "nolick"):
        for ep in ("pre", "acute", "subacute", "chronic"):
            v = [r for r in rows if r["epoch"] == ep and r.get(f"{cls}_415_early", "") != ""]
            if not v:
                continue

            def grp(k, v=v, cls=cls):
                d = defaultdict(list)
                for r in v:
                    d[r["animal"]].append(float(r[f"{cls}_{k}"]))
                return ak.boot_ci(d, rng)

            c470, c415 = grp("470_early"), grp("415_early")
            if c470 is None or c415 is None:
                continue
            ratio = c415[0] / c470[0] if abs(c470[0]) > 1e-9 else float("nan")
            print(f"  {cls:<8}{ep:<10}{len(v):>4}"
                  f"{c470[0]:>+12.3f} [{c470[1]:+.3f},{c470[2]:+.3f}]"
                  f"{c415[0]:>+12.3f} [{c415[1]:+.3f},{c415[2]:+.3f}]{ratio:>8.2f}")
            stats.append(dict(cls=cls, epoch=ep, n_sessions=len(v),
                              e470=round(c470[0], 4), e470_lo=round(c470[1], 4),
                              e470_hi=round(c470[2], 4), e415=round(c415[0], 4),
                              e415_lo=round(c415[1], 4), e415_hi=round(c415[2], 4),
                              ratio=round(ratio, 4),
                              e415_ci_excludes_zero=bool(c415[1] > 0 or c415[2] < 0)))
    if stats:
        qs = out_dir / "epoch_16_nvc_evoked_stats.csv"
        with open(qs, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(stats[0]))
            wr.writeheader()
            wr.writerows(stats)
        print(f"\nwrote {qs}")

    print("\nHOW TO READ THE NO-LICK ROWS, AND ONLY THOSE:")
    print("  415 early CI BELOW 0   ->  415 sits BELOW the crossing: it carries -Ca. A dip.")
    print("  415 early CI ABOVE 0   ->  a positive fast term remains; NOT a demonstration that")
    print("      415 is isosbestic, because a neutral (scattering/focus) term also lands here.")
    print("      THE RATIO DISCRIMINATES: ~1.0 is spectrally neutral, well under 1 is not.")
    print("  415 early CI SPANS 0   ->  no fast 415 term resolvable. Read the trace AND the trial")
    print("      counts -- this is the outcome low power also produces.")
    print("\nAND GCaMP6s KINETICS BLUR THE WINDOWS (Priya, 2026-09-19): rise 200-500 ms,")
    print("  half-decay 1-2 s, so a calcium dip is NOT confined to [0, 0.4] and the haemodynamic")
    print("  rise is not confined to [1, 3]. The figure's TRACE is the primary object here; the")
    print("  windows summarise it, they do not replace it.")

    fig = _figure(curves, tvec, out_dir)
    if fig is not None:
        print(f"\nwrote {fig}")
    return 0


def _figure(curves, t, out_dir):
    """The TRACES, because a pair of windows cannot show a dip and a trace can.

    Simpson et al. 2024 name the observable as "a negative peak in the EVENT-ALIGNED AVERAGE" --
    a SHAPE, not a number. Two window averages are exactly what hid it here for a day.
    """
    if t is None or not curves:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [e for e in ("pre", "acute", "subacute", "chronic") if any(k[1] == e for k in curves)]
    if not eps:
        return None
    fig, axes = plt.subplots(2, len(eps), figsize=(3.3 * len(eps) + 0.6, 6.8),
                             squeeze=False, sharex=True, sharey="row")
    fig.subplots_adjust(top=0.74, bottom=0.09, hspace=0.22)
    style = {"lick": ("0.55", "--", "licked"), "nolick": ("#1b5e9c", "-", "NO lick in [-1, +4] s")}
    for j, ep in enumerate(eps):
        for chan in (0, 1):
            ax = axes[chan][j]
            for cls, (col, ls, lab) in style.items():
                v = curves.get((cls, ep)) or []
                if not v:
                    continue
                arr = np.asarray([x[1 + chan] for x in v])
                m = arr.mean(axis=0)
                ax.plot(t, m, ls, color=col, lw=1.8, label=f"{lab} (n={len(v)} sess)")
                # SEM ACROSS SESSIONS, not trials -- sessions are the unit the CI above resamples,
                # and a trial-wise band would be several times narrower for no added truth.
                if len(v) > 1:
                    se = arr.std(axis=0, ddof=1) / np.sqrt(len(v))
                    ax.fill_between(t, m - se, m + se, color=col, alpha=0.18, lw=0)
            ax.axhline(0, color="k", lw=0.7)
            ax.axvline(0, color="k", lw=0.7, ls=":")
            ax.axvspan(*EARLY_S, color="#c8102e", alpha=0.07, lw=0)
            ax.spines[["top", "right"]].set_visible(False)
            if chan == 0:
                ax.set_title(ep, fontsize=12, fontweight="bold")
            else:
                ax.set_xlabel("time from cue (s)", fontsize=9)
            if j == 0:
                ax.set_ylabel(("470 nm (GCaMP)" if chan == 0 else "415 nm (control)")
                              + "\n% dF/F, per-trial baselined", fontsize=9)
            if chan == 1 and j == len(eps) - 1:
                ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.text(0.5, 0.995, "epoch_16 -- is 415 nm isosbestic HERE? The cue-evoked trace, "
             "with and without licking", ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.958,
             "THE QUESTION. Below GCaMP`s neutral/anionic crossing calcium DECREASES fluorescence; "
             "above it, increases. Three published estimates of where that crossing sits -- 405-415 "
             "(folk practice), 420-430 (Simpson 2024), 440-450 (Barnett 2017) -- and two of the "
             "three put our 415 BELOW it.\n"
             "TIMING SETTLES IT AND AMPLITUDE CANNOT. A negative calcium term appears as an EARLY "
             "DIP (shaded) before the slow haemodynamic rise. A window average over the whole "
             "post-cue period averages a dip away completely, which is why the +0.5 to +2.0% "
             "figures already on record could not answer this.\n"
             "WHY THE NO-LICK CLASS. A FAST component in 415 cannot be haemodynamic (latency "
             ">= 300-500 ms), so on licking trials a movement-locked term could bury a smaller "
             "calcium dip -- and motion correction does not remove it, being z-motion, tilt and "
             "focus rather than in-plane translation. No-lick trials have ZERO licks anywhere in "
             "[-1, +4] s.\n"
             "READ THE 415 ROW ON THE SOLID TRACE. A dip means 415 carries -Ca, so the correction "
             "has a GAIN rather than a contamination. NO dip is not proof of isosbesticity: "
             "GCaMP6s kinetics (rise 200-500 ms, half-decay 1-2 s) smear the separation, and a "
             "spectrally neutral term (ratio ~1.0 against 470) also lands early.",
             ha="center", va="top", fontsize=8.2)
    out = out_dir / "epoch_16_nvc_evoked.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
