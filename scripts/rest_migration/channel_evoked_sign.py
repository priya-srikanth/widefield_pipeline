"""WHICH WAY DOES 415 MOVE AFTER A CUE, AND WHEN? The sign of the haemodynamic term, measured.

Priya, 2026-09-19: *"we're not measuring the Hgb absorbance, we're measuring the Ca-independent
(ish) fluorescence of GCaMP ... which will increase with increased blood"*, against my claim that
blood enters both dF/F traces NEGATIVELY because haemoglobin absorbs.

**THE WHOLE TROUGH-VERSUS-PEAK CHOICE IN `rest_coupling` RESTS ON THIS SIGN**, so it is worth one
measurement rather than an argument from spectra. A cue-aligned brain-mean average settles it,
because the two candidate contributions to 415 separate IN TIME: calcium bleed-through is
simultaneous with the 470 transient, and functional hyperaemia follows it by a second or more.

WHAT IT SHOWS, and it is neither of the two positions above -- **NOR IS IT THE SAME IN EVERY
ANIMAL, WHICH IS THE POINT THAT MATTERS.**

Every animal has a large FAST POSITIVE 415 component riding on the 470 calcium transient, 0.28 to
0.42 of it. At frame resolution it peaks EARLIER than 470 (0.26-0.29 s against 0.38 s), which is
far too early for a vascular response, so it is calcium bleed-through -- 415 sits blue of GCaMP6's
true isosbestic near 420-430 nm (the Neuron 2023 point). **0.25 s bins cannot resolve that offset**;
it comes from the frame-resolution measurement.

The LATE deflection is where the animals disagree. PS93 goes clearly NEGATIVE (-0.57% at 1.1-1.4 s
while 470 is still +0.79%), which is the absorption signature of hyperaemia and the thing that
would license reading a trough. PS95 barely dips. **PS92 NEVER GOES NEGATIVE, AND PS94 HOLDS
+0.9% OUT TO 3 s** with a late 415/470 ratio near 0.8 -- roughly double its own early ratio, so
there is an extra 415-heavy component late that is NOT scaled calcium bleed-through and is not
negative.

**SO THE EVOKED DATA DO NOT ESTABLISH A SINGLE SIGN ACROSS ANIMALS.** What survives for
`rest_coupling` is weaker and should be stated as such: at REST the cross-correlation is biphasic
with the trough at positive lag in every animal tested, and the T-free `asym_470_415` is negative
in every animal and epoch. Those are consistent; this evoked measurement is not, and PS94 in
particular is closer to Priya's reading than to mine.

A caveat on the late window specifically: cues recur every few seconds, so a pre-cue baseline
carries the previous trial's tail, and PS94's 470 does not return to baseline within 3 s either.
The late plateau may be partly trial-structure rather than physiology.

    python -m scripts.rest_migration.channel_evoked_sign [--animals PS92 ...] [--per-animal 3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PRE, POST = 3.0, 4.0
BIN_S = 0.25                 # Priya's request; wide enough to read the late trough off the figure
BASE = (-1.0, -0.2)          # pre-cue, and clear of the 415 fast component


def session_segments(s):
    """``(t, {channel: (n_cue, n_t) dF/F segments})`` cue-aligned, baseline-subtracted."""
    from wfield_local import config
    from wfield_local.hemo_variants import FS, functional_channel
    from wfield_local.hemispheric_intensity import hemisphere_masks
    from wfield_local.rest_by_position import _session_daq

    res = Path(s["mc"]) / "wfield_local_results"
    allen = res / "allen_aligned_affine8v1"
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    fc = functional_channel(s)
    d = {"470": np.asarray(svt[:, fc::2], np.float64),
         "415": np.asarray(svt[:, (fc + 1) % 2::2], np.float64),
         "SVTcorr": np.asarray(np.load(config.svtcorr_in(res), mmap_mode="r"), np.float64)}
    U = np.load(allen / "U_atlas.npy", mmap_mode="r")
    left, right, _g = hemisphere_masks(allen)
    op = np.asarray(U, np.float64)[left | right].mean(0)          # whole-brain mean
    _rest, cs, _codes, _ts, fs_samp, _sy = _session_daq(s)

    f_of = np.asarray(fs_samp)
    n = min(min(v.shape[1] for v in d.values()), f_of.size)
    a, b = int(round(PRE * FS)), int(round(POST * FS))
    cue_f = np.searchsorted(f_of[:n], np.asarray(cs, np.int64))
    ok = cue_f[(cue_f > a) & (cue_f < n - b)]
    if ok.size == 0:
        return None, None
    t = np.arange(-a, b) / FS
    jb = (t >= BASE[0]) & (t <= BASE[1])
    out = {}
    for k, v in d.items():
        x = op @ v[:, :n]
        seg = np.stack([x[f - a:f + b] for f in ok]) * 100.0      # per cent dF/F
        out[k] = seg - seg[:, jb].mean(1, keepdims=True)
    return t, out


def _bin(t, seg, bin_s=BIN_S):
    """Bin to `bin_s`, returning ``(centres, mean, sem)`` with the SEM taken ACROSS CUES."""
    edges = np.arange(-PRE, POST + 1e-9, bin_s)
    idx = np.digitize(t, edges) - 1
    keep = [k for k in range(edges.size - 1) if (idx == k).any()]
    c = np.array([(edges[k] + edges[k + 1]) / 2 for k in keep])
    m = np.array([seg[:, idx == k].mean() for k in keep])
    # SEM over CUES, not over the samples inside a bin -- samples within a bin are not independent
    # observations of the response, they are one response sampled repeatedly.
    e = np.array([seg[:, idx == k].mean(1).std(ddof=1) / np.sqrt(seg.shape[0]) for k in keep])
    return c, m, e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--per-animal", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    by_animal: dict[str, list] = {}
    for s in config.load_sessions():
        lab = s["label"]
        if lab not in want or epochs.epoch_of(lab) != "pre":
            continue
        an = config.animal_of(lab)
        if a.animals and an not in a.animals:
            continue
        by_animal.setdefault(an, []).append(s)

    pooled: dict[str, tuple] = {}
    for an in sorted(by_animal):
        acc, tt, ncue = {}, None, 0
        for s in by_animal[an][:a.per_animal]:
            try:
                t, seg = session_segments(s)
            except Exception as ex:                                  # noqa: BLE001
                print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
                continue
            if t is None:
                continue
            tt = t
            ncue += next(iter(seg.values())).shape[0]
            for k, v in seg.items():
                acc.setdefault(k, []).append(v)
            print(f"   {s['label']:14s} {next(iter(seg.values())).shape[0]:4d} cues", flush=True)
        if acc:
            pooled[an] = (tt, {k: np.concatenate(v) for k, v in acc.items()}, ncue)

    if not pooled:
        print("no sessions -- a failed run, not a result")
        return 1

    col = {"470": "#1f77b4", "415": "#d62728", "SVTcorr": "#2ca02c"}
    fig, axes = plt.subplots(1, len(pooled), figsize=(4.2 * len(pooled), 4.2), sharex=True,
                             squeeze=False)
    for ax, an in zip(axes[0], sorted(pooled)):
        t, seg, ncue = pooled[an]
        for k in ("470", "415", "SVTcorr"):
            c, m, e = _bin(t, seg[k])
            ax.plot(c, m, "-o", ms=2.6, lw=1.5, color=col[k], label=k)
            ax.fill_between(c, m - e, m + e, color=col[k], alpha=0.20, lw=0)
        c, m, _e = _bin(t, seg["415"])
        late = (c > 0.7) & (c < 3.0)
        if late.any():
            j = np.where(late)[0][int(np.argmin(m[late]))]
            # LABELLED AS A MINIMUM, NOT A TROUGH. In PS92 and PS94 the minimum over this window
            # is POSITIVE -- there is no trough to point at, and calling it one would draw a
            # conclusion the figure is supposed to be testing.
            ax.annotate(f"415 min {m[j]:+.2f}% at {c[j]:+.2f}s", (c[j], m[j]),
                        textcoords="offset points", xytext=(14, -24), fontsize=7.5,
                        color=col["415"],
                        arrowprops=dict(arrowstyle="->", color=col["415"], lw=0.8))
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axvline(0, color="0.4", lw=0.8, ls="--")
        ax.set_title(f"{an}  ({ncue} cues, pre-stroke)", fontsize=10)
        ax.set_xlabel("time from cue (s)")
        ax.set_xlim(-PRE, POST)
    axes[0][0].set_ylabel("brain-mean dF/F (%)")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("415 carries a LARGE FAST component tracking the 470 calcium transient in every "
                 "animal (0.28-0.42 of it).\nThe LATE deflection DISAGREES ACROSS ANIMALS: clearly "
                 "NEGATIVE in PS93 (absorption), absent in PS92, and PS94 holds +0.9% to 3 s.\n"
                 f"So the evoked sign is NOT established cohort-wide.   {BIN_S:.2f} s bins, SEM "
                 "across cues, 3 pre-stroke sessions pooled per animal.", fontsize=9.0)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    q = out_dir / "epoch_20_channel_evoked_sign.png"
    fig.savefig(q, dpi=180)
    print(f"\nwrote {q}")

    bar = "=" * 92
    print(f"\n{bar}\nCUE-ALIGNED BRAIN-MEAN dF/F (%), {BIN_S:.2f} s bins\n{bar}")
    for an in sorted(pooled):
        t, seg, ncue = pooled[an]
        c, _m, _e = _bin(t, seg["470"])
        js = [i for i, x in enumerate(c) if -1.0 <= x <= 3.0]
        print(f"\n  {an}  ({ncue} cues)")
        print("      t (s) " + "".join(f"{c[j]:>7.2f}" for j in js))
        for k in ("470", "415", "SVTcorr"):
            _c, m, _e = _bin(t, seg[k])
            print(f"   {k:>8s} " + "".join(f"{m[j]:>7.2f}" for j in js))
    print("\n  THE FAST POSITIVE IS CALCIUM BLEED-THROUGH -- at frame resolution it peaks EARLIER")
    print("  than 470 (0.26-0.29 s against 0.38 s), too early for a vascular response. 0.25 s")
    print("  bins cannot resolve that offset.")
    print("\n  THE LATE DEFLECTION DISAGREES ACROSS ANIMALS: PS93 clearly negative, PS95 barely,")
    print("  PS92 never, PS94 held near +0.9% out to 3 s. **The evoked sign is NOT established**")
    print("  cohort-wide, so it does NOT by itself license the trough reading in `rest_coupling`;")
    print("  that rests on the rest-period curve shape and on the T-free `asym_470_415`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
