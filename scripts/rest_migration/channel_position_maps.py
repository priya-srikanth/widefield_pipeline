"""415 nm vs RAW 470 nm vs CORRECTED 470 nm, as spout-position maps. Pre-stroke sessions.

Priya, 2026-09-19: *"compare 415 alone vs 470 alone vs corrected SVD maps for lick-aligned signal
at all 6 positions (similar to the maps in the preprocessing deck) for a handful of pre-stroke
sessions"*. The figures at `labcams/channel_comparison` are the ancestor of this -- built
2026-07-08 by `_compare_415_470_corr.py`, a repo-root one-off RETIRED on 2026-08-08 (45b9ef3) as
dead scratch, for ONE hardcoded session.

THREE THINGS ARE FIXED RELATIVE TO THAT SCRIPT, and two of them were real bugs.

1. **CHANNEL IDENTITY COMES FROM THE SESSION'S OWN RECORD, CHECKED.** The old script hardcoded
   `SVT[:, 0::2]` as 415 on the strength of a comment. THIS IS THE EXACT PLACE A SWAP IS INVISIBLE:
   both halves produce a plausible cortical map, so a transposed pair reads as "415 carries a lot
   of structure" rather than as a labelling error. My first attempt at a fix -- DERIVING the
   identity by correlating each half with `SVTcorr` -- was worse than useless and is documented in
   `_signals`: it is a coin flip on some sessions and it flipped the wrong way on PS93. The module
   now reads `functional_channel` from `local_wfield_summary.json` and keeps the correlation only
   as a cross-check that prints on disagreement.

2. **THE REPAIRED POSITION CLASSIFIER.** The old script called raw `_classify_cues`. A dead
   `spout_bit1` on the August sessions reads that bit low and COLLAPSES 6 positions onto 4
   (2->0, 3->1, 6->4, 7->5) -- and `PS92_0806` onward are pre-stroke sessions, so this is not a
   post-stroke-only concern. `classify_cues_with_backup` detects the collapse and repairs it from
   the behaviour log. On the June example session the two agree, which is why the original figure
   looked right.

3. **ANATOMICAL POSITION LABELS**, near/far x contra/ipsi, not the rig's `close_L` / `far_R`.

WHAT THE FIGURE IS FOR, AND WHAT IT CANNOT DO -- the second part corrected by Priya on the day it
was built. It shows how large 415 is against 470 at each position and how much the correction
removes. It does NOT adjudicate whether 415 carries calcium: *"a spatial pattern of correlation is
what we would expect with neurovascular coupling"*, and that is right, so a 415 map resembling the
470 map is the EXPECTED result under a correction that is working perfectly. See `_quantify`.

THE ONE THING IT DOES ADJUDICATE is whether the correction ran correctly at all: r(415, corrected)
must fall below r(415, raw). That check is what exposed the PS93 channel swap described above.

SHARED COLOUR SCALE ACROSS THE THREE COLUMNS OF A ROW, so relative magnitude is honest. This is the
whole point of the layout and it is why the columns are not individually normalised: the question
includes "how big is 415 compared to 470", and per-panel scaling would answer a different one.

    python -m scripts.rest_migration.channel_position_maps [--sessions PS92_0608 ...] [--n 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FS = 31.23
CUE_PRE_S, CUE_POST_S = 2.0, 2.0
LICK_POST_S = 0.15                 # matches the preprocessing deck's lick-aligned maps
DEFAULT_N = 4


def _signals(res: Path, allen: Path):
    """``[(name, SVT_half), ...]`` with 415/470 from the SESSION'S OWN RECORD, plus a cross-check.

    THE CORRELATION HEURISTIC WAS NOT GOOD ENOUGH, AND IT FAILED ON ITS FIRST REAL RUN
    (2026-09-19). The first version picked blue as whichever half `SVTcorr` component 0 correlated
    with more. Measured margins over five pre-stroke sessions:

        PS92_0606  0.134 / 0.575   margin 0.441   odd
        PS94_0606  0.146 / 0.419   margin 0.273   odd
        PS95_0606  0.036 / 0.284   margin 0.248   odd
        PS92_0608  0.140 / 0.272   margin 0.132   odd
        PS93_0606  0.276 / 0.246   margin 0.030   EVEN   <- a coin flip, and it came up wrong

    PS93 got the channels SWAPPED, and the resulting figure did not look broken -- it looked like a
    finding. Its stats were the outliers of the whole run (r(415, corrected) = +0.76 where every
    other animal gave a negative, amplitude ratio 2.33 where the rest gave 0.4-1.0), and I was one
    step from reporting "PS93's correction is misbehaving" as biology. **A weak discriminator that
    is usually right is worse than no discriminator, because it is only wrong where it matters.**

    The fix is that the answer was never in the data to begin with. `local_wfield_summary.json`
    RECORDS `functional_channel` per session; the correlation is kept as a CROSS-CHECK that prints
    when it disagrees, which is the right job for a heuristic of this strength.
    """
    import json

    from wfield_local import config
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    corr = np.load(config.svtcorr_in(res), mmap_mode="r")
    with open(res / "local_wfield_summary.json", encoding="utf-8") as fh:
        func = int(json.load(fh)["functional_channel"])
    blue = np.asarray(svt[:, func::2])
    other = np.asarray(svt[:, (func + 1) % 2::2])

    n = min(corr.shape[1], blue.shape[1], other.shape[1])
    c0 = np.asarray(corr[0, :n])
    r_blue = abs(np.corrcoef(c0, blue[0, :n])[0, 1])
    r_other = abs(np.corrcoef(c0, other[0, :n])[0, 1])
    ev = dict(functional_channel=func, r_blue=round(float(r_blue), 3),
              r_other=round(float(r_other), 3), crosscheck_agrees=bool(r_blue >= r_other))
    if not ev["crosscheck_agrees"]:
        print(f"      .. cross-check DISAGREES with functional_channel={func}: "
              f"r(SVTcorr, blue)={r_blue:.3f} < r(SVTcorr, other)={r_other:.3f}. Using the "
              f"RECORD. If this fires widely, the record is what to doubt.", flush=True)
    return [("415 nm (control)", other), ("470 nm (raw)", blue),
            ("470 nm (hemo-corrected)", np.asarray(corr))], ev


def _epoch_of(label):
    from wfield_local import epochs
    return epochs.epoch_of(label) or ""


def _boot_ci(by_animal, rng, n_boot=4000):
    """Nested animals -> sessions bootstrap CI of the mean, as `epoch_figures` does it."""
    animals = sorted(by_animal)
    if not animals:
        return None
    flat = [v for an in animals for v in by_animal[an]]
    out = []
    for _ in range(n_boot):
        vals = []
        for an in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            sa = by_animal[an]
            vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        if vals:
            out.append(float(np.mean(vals)))
    if len(out) < n_boot // 4:
        return None
    o = np.asarray(out)
    return float(np.mean(flat)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def epoch_summary(stats, out_dir, seed):
    """IS NEUROVASCULAR COUPLING WEAKER AFTER THE STROKE? The epoch contrast, per arm.

    Priya, 2026-09-19: *"we could just look at the r(415, raw) or r(415, 470) - is there a
    difference pre vs post-stroke to see differences in coupling, no?"* -- and this is a better
    question than the one the module was built for, because **IT DOES NOT REQUIRE RESOLVING WHERE
    GCaMP'S CROSSING SITS.** Whatever 415 is made of, if it is dominated by haemodynamics then
    `||415|| / ||470raw||` is haemodynamic response per unit neural response -- an NVC GAIN -- and
    `r(415, 470raw)` is how well the vascular map tracks the neural one. Both should FALL if
    coupling is lost.

    THE TWO STATISTICS ARE BIASED IN OPPOSITE, KNOWN DIRECTIONS, AND THAT IS WHY BOTH ARE HERE.
    `raw = C + H` contains H, so `r(415, raw)` is inflated by shared haemodynamics; `corr` had
    `T*415` subtracted, so `r(415, corr)` is deflated by construction (measured -0.19 to -0.36
    pre-stroke, and that negative is arithmetic, not biology). **Neither is a clean NVC
    coefficient.** They are usable HERE only because the bias is a property of the estimator and
    not of the epoch, so an epoch CONTRAST differences it away -- the same argument that lets `15j`
    read a biased cosine against its own null.

    WHAT WOULD BREAK THAT ARGUMENT, stated because it is the thing to check first if the result
    looks large: if the stroke changes the ratio of C to H in the raw channel, the inflation on
    `r(415, raw)` changes with it, and part of any epoch difference is the bias moving rather than
    the coupling. The amplitude ratio is the more robust of the two for exactly this reason -- it
    has a numerator and a denominator that are separately measured.

    AND AN INFARCT LOSES BOTH SIGNALS AT ONCE. A ratio whose denominator is collapsing is not a
    coupling measurement, so the per-position amplitude floor matters; acute far-contra in
    particular has few events and little of either channel.
    """
    import csv
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    eps = [e for e in ("pre", "acute", "subacute", "chronic")
           if any(x["epoch"] == e for x in stats)]
    rows = []
    bar = "=" * 92
    print(f"\n{bar}\nIS NEUROVASCULAR COUPLING WEAKER AFTER THE STROKE?\n{bar}")
    print(f"  {'arm':<7}{'epoch':<10}{'n':>4}{'||415||/||470raw||':>30}{'r(415, 470raw)':>30}")
    for arm in ("cue", "lick"):
        for e in eps:
            v = [x for x in stats if x["arm"] == arm and x["epoch"] == e]
            if not v:
                continue
            got = {}
            for key in ("amp_415_over_raw", "r_415_raw"):
                d = defaultdict(list)
                for x in v:
                    if np.isfinite(x[key]):
                        d[x["animal"]].append(float(x[key]))
                got[key] = _boot_ci(d, rng)
            if any(g is None for g in got.values()):
                continue
            a_, r_ = got["amp_415_over_raw"], got["r_415_raw"]
            print(f"  {arm:<7}{e:<10}{len(v):>4}"
                  f"{a_[0]:>16.3f} [{a_[1]:.3f},{a_[2]:.3f}]"
                  f"{r_[0]:>16.3f} [{r_[1]:.3f},{r_[2]:.3f}]")
            rows.append(dict(arm=arm, epoch=e, n_cells=len(v),
                             amp_ratio=round(a_[0], 4), amp_lo=round(a_[1], 4),
                             amp_hi=round(a_[2], 4), r_415_raw=round(r_[0], 4),
                             r_lo=round(r_[1], 4), r_hi=round(r_[2], 4)))
    if rows:
        q = out_dir / "channel_position_maps_by_epoch.csv"
        with open(q, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {q}")
    print("\n  BOTH FALLING after the stroke -> less haemodynamic response per unit neural")
    print("      response AND a worse spatial match: weaker coupling on two counts.")
    print("  AMPLITUDE FALLS, r HOLDS -> the vasculature still responds where it should, less.")
    print("  r FALLS, AMPLITUDE HOLDS -> the response is there but misdirected -- the pattern")
    print("      a damaged vascular bed would give.")
    print("  NEITHER MOVES -> no detectable change in coupling at this resolution. Check the")
    print("      event counts before believing it; acute far-contra is thin in both channels.")
    return rows


def _win_avg(S, frames, a, b):
    acc = np.zeros(S.shape[0], np.float64)
    for fr in frames:
        acc += np.asarray(S[:, fr + a:fr + b]).mean(1)
    return acc / max(len(frames), 1)


def _quantify(maps, mask):
    """``dict`` -- the numbers the eye cannot read off three panels on a shared scale.

    Priya, 2026-09-19: *"it seems like the correction is getting rid of some of the signal
    (amplitude seems blunted in the corrected map)"*. It is blunted, visibly -- 0.74 (cue) and
    0.83 (lick) of the raw norm, cohort median. **AMPLITUDE ALONE CANNOT SAY WHETHER THAT IS A
    LOSS**: with `raw = C + H`, a PERFECT correction returns `C`, which is smaller than `C + H`.
    Blunting is what success looks like too.

    **AND SPATIAL CORRELATION DOES NOT SEPARATE THEM EITHER -- this module's first version claimed
    it did, and Priya corrected it the same day:** *"a spatial pattern of correlation is what we
    would expect with neurovascular coupling"*. Exactly so. Blood flow follows neural activity, so
    the haemodynamic map RESEMBLES the neural map by construction. The measured `r_415_raw` of
    +0.56 to +0.59 is the NVC signature, not evidence of calcium in the control channel.

        415 carries scaled CALCIUM              -> positive r, ratio invariant to amplitude
        415 carries HAEMODYNAMICS that scale    -> positive r, ratio invariant to amplitude
            with calcium (i.e. NVC)

    The two are indistinguishable in every static quantity, because NVC is itself
    signal-proportional. **ONLY LATENCY SEPARATES THEM** -- zero lag for an optical leak, 0.5-2 s
    for NVC -- which is a different measurement from this one.

    SO WHAT ARE THESE NUMBERS FOR? `r_415_corr` is the check `hemo_map_control` already runs: the
    corrected map must be LESS like 415 than the raw one was, or the correction did not do its
    named job. That is a pass/fail on the correction, not an answer about the wavelength -- and it
    is the check that caught the PS93 channel swap, since a swapped session fails it loudly.
    """
    m = np.asarray(mask, bool)
    v = [np.asarray(x)[m].astype(np.float64) for x in maps]
    v = [x - x.mean() for x in v]

    def r(i, j):
        d = np.linalg.norm(v[i]) * np.linalg.norm(v[j])
        return float(v[i] @ v[j] / d) if d > 0 else float("nan")

    n415, n470, ncorr = (float(np.linalg.norm(x)) for x in v)
    return dict(r_415_raw=round(r(0, 1), 4), r_415_corr=round(r(0, 2), 4),
                r_raw_corr=round(r(1, 2), 4),
                amp_415_over_raw=round(n415 / n470, 4) if n470 > 0 else float("nan"),
                amp_corr_over_raw=round(ncorr / n470, 4) if n470 > 0 else float("nan"),
                # slope of 470 on 415 across PIXELS: how much of the raw map a scaled 415 accounts
                # for. Paired with r_415_raw because a slope without a fit quality means nothing.
                beta_415_on_raw=round(float(v[0] @ v[1] / (v[0] @ v[0])), 4)
                if v[0] @ v[0] > 0 else float("nan"))


def _limit(maps, mask):
    """99th percentile of |map| OVER THE BRAIN MASK ONLY.

    THE FIRST VERSION TOOK THE PERCENTILE OVER THE WHOLE 540x640 FRAME AND THE FIGURE WAS WRONG
    (2026-09-19, caught by Priya: *"is there really so little lick aligned activity in this
    session? ... Does this session just have an extraordinarily high response to lick near_ipsi?"*).
    No -- `U_atlas` carries a band of extreme values at the registered edge, outside cortex, and it
    was setting the colour limit. PS93_0606, lick arm, all-pixel limit against in-mask limit:

        near ipsi     0.0393 / 0.0394   x1.00      <- the ONE row that was not inflated
        near middle   0.1705 / 0.0365   x4.67
        near contra   0.1586 / 0.0331   x4.79
        far ipsi      0.1581 / 0.0489   x3.24
        far middle    0.1069 / 0.0685   x1.56
        far contra    0.0989 / 0.0459   x2.16

    So five rows were washed out by a factor of up to 4.8 and the sixth was not, which READS as one
    position with a huge response. In-mask the six limits span 0.033-0.069 -- under 2x, no outlier.
    **A per-row scale computed from pixels nobody is looking at turned an edge artefact into a
    result.** The maps are also NaN-ed outside the mask now, so the artefact cannot be drawn.
    """
    v = np.concatenate([np.asarray(m)[mask].ravel() for m in maps])
    return max(float(np.nanpercentile(np.abs(v), 99.0)), 1e-6)


def _row(axrow, maps, titles, edges, lim, mask):
    """Draw one position's three panels. NO COLOURBAR -- the figure carries a single shared one.

    Six identical colourbars was what a per-row limit left behind: once the limit is global they
    all read the same number, and stacked vertically their labels overlap into noise.
    """
    from wfield_local.atlas_overlay import overlay_regions
    im = None
    for ax, m, t in zip(axrow, maps, titles):
        ax.set_axis_off()
        shown = np.where(mask, np.asarray(m, np.float64), np.nan)
        im = ax.imshow(shown, cmap="RdBu_r", vmin=-lim, vmax=lim)
        overlay_regions(ax, edges)
        ax.set_title(t, fontsize=10)
    return im


def session_figure(s, out_dir, arm, ev_frames_by_code, order, labels, sig, edges, U, note, mask,
                   stats, figures=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.plot_lick_aligned_averages import _weighted_map

    pre_n, post_n = int(round(CUE_PRE_S * FS)), int(round(CUE_POST_S * FS))
    lpost = max(1, int(round(LICK_POST_S * FS)))
    lbl = "post - pre" if arm == "cue" else f"{LICK_POST_S * 1000:.0f} ms post-lick"
    im = None

    # MAPS FIRST, DRAW SECOND, so one limit can cover the whole figure. A PER-ROW limit makes the
    # three CHANNELS comparable and the six POSITIONS not, which is half the comparison the layout
    # exists for -- and "does this position respond more than that one" is a question a reader will
    # ask of a 6-row figure whether or not it was designed to answer it.
    built = {}
    for code in order:
        fr = ev_frames_by_code.get(code, np.array([], int))
        if fr.size == 0:
            continue
        if arm == "cue":
            built[code] = [_weighted_map(U, (_win_avg(S, fr, 0, post_n)
                                             - _win_avg(S, fr, -pre_n, 0)).astype(np.float32))
                           for _n, S in sig]
        else:
            built[code] = [_weighted_map(U, _win_avg(S, fr, 0, lpost).astype(np.float32))
                           for _n, S in sig]
    lim = _limit([m for ms in built.values() for m in ms], mask) if built else 1e-6

    fig, axes = plt.subplots(len(order), 3, figsize=(13, 3.55 * len(order) + 1.4),
                             squeeze=False, constrained_layout=True)
    for r, code in enumerate(order):
        fr = ev_frames_by_code.get(code, np.array([], int))
        if code not in built:
            # LEAVE THE CELL EMPTY rather than drawing a zero map -- a position with no events is
            # missing data, and a blue-white panel reads as "measured, and flat".
            for ax in axes[r]:
                ax.set_axis_off()
            axes[r][0].set_title(f"{labels[r]}: no {arm} events", fontsize=10, loc="left")
            continue
        maps = built[code]
        q = _quantify(maps, mask)
        stats.append(dict(label=s["label"], animal=s["label"].split("_")[0],
                          epoch=_epoch_of(s["label"]), arm=arm,
                          position=labels[r], n_events=int(fr.size), **q))
        if not figures:
            continue
        im = _row(axes[r], maps, [f"{n}\n{labels[r]}  n={fr.size}" for n, _S in sig], edges,
                  lim, mask)
        axes[r][2].set_xlabel(
            f"r(415, raw) = {q['r_415_raw']:+.2f}   r(415, corr) = {q['r_415_corr']:+.2f}\n"
            f"||415||/||raw|| = {q['amp_415_over_raw']:.2f}   "
            f"||corr||/||raw|| = {q['amp_corr_over_raw']:.2f}",
            fontsize=8.5, labelpad=4)
        axes[r][2].set_axis_on()
        axes[r][2].set_xticks([])
        axes[r][2].set_yticks([])
        for sp in axes[r][2].spines.values():
            sp.set_visible(False)
    if not figures:
        plt.close(fig)
        return None
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, pad=0.012, aspect=40)
        cb.set_label(f"{lbl}   (+/-{lim:.4g}, ONE SCALE FOR THE WHOLE FIGURE -- so the six "
                     f"positions are comparable to each other, not only the three channels "
                     f"within a row)", fontsize=9)
    import textwrap
    fig.suptitle(f"{s['label']} -- {arm}-aligned by spout position: "
                 f"415 vs RAW 470 vs CORRECTED 470\n"
                 + "\n".join(textwrap.wrap(note, 130)), fontsize=12)
    out = out_dir / f"{s['label']}_{arm}_415_vs_470_vs_corr_by_position.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_session(s, out_dir, stats, figures=True):
    """Both arms for one session. Returns the paths written; appends per-position rows to `stats`."""
    from wfield_local import epoch_figures as ef
    from wfield_local.atlas_overlay import region_edges
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES
    from wfield_local.rest_by_position import _session_daq

    mc = Path(s["mc"])
    res = mc / "wfield_local_results"
    allen = res / "allen_aligned_affine8v1"
    U = np.load(allen / "U_atlas.npy", mmap_mode="r")
    edges = region_edges(np.load(allen / "allen_area_atlas_native_grid.npy"))
    # THE SAME BRAIN MASK THE REST OF THE PIPELINE SCORES ON -- a correlation over the full
    # 540x640 frame is dominated by the empty surround, where both maps are ~0 and agree perfectly.
    mask = np.load(allen / "allen_brain_mask_native_grid.npy").astype(bool)
    sig, ev = _signals(res, allen)
    T = sig[-1][1].shape[1]

    _rest, cs, _c, _ts, fs_samp, _sync = _session_daq(s)
    f_of = np.asarray(fs_samp)
    cue = _load_cue_events(s["h5"])
    codes = np.asarray(classify_cues_with_backup(s, cue, verbose=False))
    cue_f = np.searchsorted(f_of, np.asarray(cs))

    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    lick_f = np.searchsorted(f_of, np.asarray(lk["lick_samples"]))
    # A LICK INHERITS THE POSITION OF THE CUE IT FOLLOWS. Anything before the first cue has none.
    j = np.searchsorted(np.asarray(cs), np.asarray(lk["lick_samples"]), side="right") - 1
    lick_codes = np.where(j >= 0, codes[np.clip(j, 0, None)], -1)

    pre_n, post_n = int(round(CUE_PRE_S * FS)), int(round(CUE_POST_S * FS))
    lpost = max(1, int(round(LICK_POST_S * FS)))
    cue_ok = (codes >= 0) & (cue_f >= pre_n) & (cue_f + post_n <= T)
    lick_ok = (lick_codes >= 0) & (lick_f >= 0) & (lick_f + lpost <= T)

    order = DISPLAY_ORDER
    raw = [POSITION_NAMES[c] for c in order]
    labels = [x.title() for x in ef.anatomical_labels(raw, short=False)] \
        if set(raw) <= set(CONF_LABELS) else raw
    note = (f"470 = SVT[:, {ev['functional_channel']}::2], from this session's recorded "
            f"functional_channel. Cross-check r(SVTcorr, blue) {ev['r_blue']} vs "
            f"r(SVTcorr, 415) {ev['r_other']}"
            + ("" if ev["crosscheck_agrees"] else "  -- CROSS-CHECK DISAGREES, record used")
            + ". Positions from the REPAIRED classifier.")

    out = []
    for arm, frames, valid, cds in (("cue", cue_f, cue_ok, codes),
                                    ("lick", lick_f, lick_ok, lick_codes)):
        by = {c: frames[valid & (cds == c)] for c in order}
        p = session_figure(s, out_dir, arm, by, order, labels, sig, edges, U, note, mask, stats,
                           figures=figures)
        if p is not None:
            out.append(p)
        print(f"   {s['label']:14s} {arm:4s} "
              + "  ".join(f"{labels[i][:12]}={by[c].size}" for i, c in enumerate(order)), flush=True)
    return out, ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="session labels; default: the first --n PRE-stroke sessions, one per animal")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--epochs", action="store_true",
                    help="ACROSS EPOCHS, which is the neurovascular-coupling question rather than "
                         "the channel-identity one. Takes --per-epoch sessions per animal per "
                         "epoch and reports ||415||/||470|| and r(415, 470) by epoch.")
    ap.add_argument("--per-epoch", type=int, default=2)
    ap.add_argument("--no-figures", action="store_true",
                    help="stats only; the per-session PNGs are the slow part and the epoch "
                         "comparison does not read them")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import config
    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "channel_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = {x["label"]: x for x in config.load_sessions()}
    if a.sessions:
        pick = [sessions[lab] for lab in a.sessions if lab in sessions]
        missing = [lab for lab in a.sessions if lab not in sessions]
        if missing:
            print(f"!! unknown session label(s): {missing}")
    elif a.epochs:
        # BALANCED OVER ANIMAL x EPOCH. Taking the first k of a date-sorted list gives all pre and
        # no chronic, and the whole question here is the epoch contrast.
        from wfield_local import epochs as ep_mod
        want = set(config.phase_labels("pre") + config.phase_labels("post"))
        seen, pick = {}, []
        for x in config.load_sessions():
            if x["label"] not in want:
                continue
            e = ep_mod.epoch_of(x["label"])
            if not e:
                continue
            k = (config.animal_of(x["label"]), e)
            if seen.get(k, 0) < a.per_epoch:
                seen[k] = seen.get(k, 0) + 1
                pick.append(x)
    else:
        # ONE PER ANIMAL, not the first n overall -- n=4 taken off the front would be four PS92
        # sessions, which answers a question about PS92 rather than about the correction.
        pre = [x for x in config.load_sessions() if x["label"] in set(config.phase_labels("pre"))]
        seen, pick = set(), []
        for x in pre:
            an = config.animal_of(x["label"])
            if an not in seen:
                seen.add(an)
                pick.append(x)
        pick = pick[: a.n]
    if not pick:
        print("no sessions -- a failed run, not a result")
        return 1
    print(f"[channel maps] {len(pick)} session(s) -> {out_dir}")
    wrote, stats = 0, []
    for s in pick:
        try:
            paths, _ev = run_session(s, out_dir, stats, figures=not a.no_figures)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            continue
        for p in paths:
            print(f"      wrote {p.name}", flush=True)
        wrote += len(paths)
    if not wrote and not a.no_figures:
        print("wrote nothing -- a failed run, not a result")
        return 1
    print(f"\n[channel maps] {wrote} figure(s) in {out_dir}")
    if not stats:
        return 0
    import csv
    q = out_dir / "channel_position_maps_stats.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    print(f"[channel maps] wrote {q}")

    # THE QUESTION THE AMPLITUDE CANNOT ANSWER, answered on the console. Printed rather than left
    # in the CSV, because "the corrected map looks blunted" is the observation that prompted this
    # and it needs its discriminator beside it, not one file away.
    bar = "=" * 86
    print(f"\n{bar}\nIS THE CORRECTION REMOVING SIGNAL, OR REMOVING BLOOD?\n{bar}")
    print(f"  {'arm':<9}{'r(415,raw)':>12}{'r(415,corr)':>14}"
          f"{'||415||/||raw||':>18}{'||corr||/||raw||':>19}{'n':>6}")
    for arm in ("cue", "lick"):
        v = [x for x in stats if x["arm"] == arm]
        if not v:
            continue
        med = {k: float(np.median([x[k] for x in v]))
               for k in ("r_415_raw", "r_415_corr", "amp_415_over_raw", "amp_corr_over_raw")}
        print(f"  {arm:<9}{med['r_415_raw']:>+12.3f}{med['r_415_corr']:>+14.3f}"
              f"{med['amp_415_over_raw']:>18.3f}{med['amp_corr_over_raw']:>19.3f}{len(v):>6}")
    print("\n  READ r(415, raw) AS NEUROVASCULAR COUPLING, NOT AS BLEED-THROUGH. Priya, 2026-09-19:")
    print("  *\"a spatial pattern of correlation is what we would expect with neurovascular")
    print("  coupling\"* -- and that is right. Blood flow follows neural activity, so the")
    print("  haemodynamic map SHOULD resemble the neural map. A high r is the NVC signature. It is")
    print("  NOT a discriminator, and this module's first version claimed it was.")
    print("\n  WHICH LEAVES THE REAL QUESTION OPEN, AND NOTHING HERE CLOSES IT. Both accounts --")
    print("  415 carrying scaled CALCIUM, and 415 carrying HAEMODYNAMICS THAT SCALE WITH calcium")
    print("  -- predict a positive spatial correlation AND an amplitude-invariant ratio, because")
    print("  NVC is itself signal-proportional. No amplitude and no map shape separates them.")
    print("\n  ONLY LATENCY DOES. Direct optical calcium leak has ZERO lag; NVC has 0.5-2 s. The")
    print("  test is the lag of peak cross-correlation between the channels, not their overlap.")
    print("\n  WHAT THIS TABLE IS STILL GOOD FOR:")
    print("    r(415, corr) must fall BELOW r(415, raw). If it does not, the correction did not")
    print("        do the thing it is named for -- that is `hemo_map_control`'s test, and every")
    print("        animal passes it here only AFTER the channel-assignment fix of 2026-09-19.")
    print("    ||corr||/||raw|| is how much amplitude went, NOT how much of what went was signal.")
    print("        A correct correction blunts the map too, because `raw = C + H` and H is real.")
    if a.epochs:
        epoch_summary(stats, out_dir, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
