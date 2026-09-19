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
LICK_POST_S = 2.0                   # 2 s from the first post-cue in-trial lick, matching the cue arm
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


def _quit_trials(s, cue_s, codes, lick_s, resp_s=2.0):
    """Bool per cue: True = inside the TERMINAL QUIT PERIOD, i.e. not engaged.

    Wraps `precue_engagement_states.engagement_gate`, the gate `beta_maps._quit_mask` uses, so this
    module excludes the same trials the rest of the deck does rather than inventing a second
    definition. Non-recovery is what that gate requires -- a mid-session dip the animal comes back
    from is not disengagement, and PS94_0817 is the session that makes the distinction concrete.

    Returns all-False (nothing excluded) if the gate cannot be built, and SAYS SO -- silently
    falling back to "everything is engaged" is how a gate stops existing without anyone noticing.
    """
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.precue_engagement_states import engagement_gate

    cue_s = np.asarray(cue_s, float)
    n = cue_s.size
    try:
        lk = np.asarray(lick_s, float)
        lo = np.searchsorted(lk, cue_s, side="left")
        hi = np.searchsorted(lk, cue_s + resp_s * _daq_rate(s), side="right")
        responded = (hi - lo) > 0
        order = np.arange(n)
        pos = np.array([POSITION_NAMES.get(int(c), str(c)) for c in np.asarray(codes)])
        ne = np.asarray(engagement_gate(order, responded, pos), bool)
        if ne.shape != (n,):
            raise ValueError(f"gate returned {ne.shape}, expected {(n,)}")
        if ne.any():
            print(f"      {s['label']}: quit period excludes {int(ne.sum())}/{n} trials",
                  flush=True)
        return ne
    except Exception as ex:                                          # noqa: BLE001
        print(f"      !! {s['label']}: engagement gate unavailable ({type(ex).__name__} "
              f"{str(ex)[:60]}) -- NOTHING excluded, epoch composition is unguarded", flush=True)
        return np.zeros(n, bool)


def _daq_rate(s):
    """DAQ sample rate, so a seconds-valued response window can be compared against samples."""
    import h5py
    with h5py.File(s["h5"], "r") as f:
        return float(f.attrs["sample_rate_hz"])


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


#: Canonical position order, near->far x ipsi/middle/contra, matching `DISPLAY_ORDER` under
#: `anatomical_labels`. Written out rather than derived so the figure does not silently reorder if
#: a caller passes positions in another order.
_POSITION_ORDER = ("Near Ipsi", "Near Middle", "Near Contra",
                   "Far Ipsi", "Far Middle", "Far Contra")

#: Row order. AMPLITUDE TERMS FIRST AND SPLIT INTO THEIR PARTS, because a ratio is only readable
#: once you can see which term moved -- and because the SD-only version silently measured the
#: structured quarter of a signal that is ~75% global.
#: THE TWO HEADLINE ROWS ARE 3 AND 4 (Priya, 2026-09-19: *"I'm not sure I care about the position
#: specificity of the neurovascular coupling. increased / decreased coupling at all positions would
#: be of interest if it bears out"*). So the coupling GAIN and the spatial MATCH come first, with
#: their two inputs above them; the specificity and globalness decomposition is demoted to the
#: bottom as supporting material rather than the answer.
_PANELS = (("rms_raw", "470 raw AMPLITUDE\nRMS = sqrt(mean^2 + SD^2)"),
           ("rms_415", "415 AMPLITUDE\nRMS"),
           ("amp_415_over_raw_rms", "*** COUPLING GAIN ***\nRMS ratio, includes the global term"),
           ("r_415_raw", "*** SPATIAL MATCH ***\nr(415, 470raw)"),
           ("mean_raw", "470 GLOBAL term\nspatial mean"),
           ("sd_raw", "470 STRUCTURE term\nspatial SD"),
           ("amp_415_over_raw", "gain, SD only\n(structure, ignores global)"),
           ("r_415_specific", "position-specific\nr(diag) - r(off-diag)"),
           ("within_470_offdiag", "neural globalness\nmean r(470_i, 470_j)"),
           ("within_415_offdiag", "vascular globalness\nmean r(415_i, 415_j)"),
           ("excess_415_globalness", "excess vascular\nwithin415 - within470"))


def position_epoch_figure(stats, out_dir, arm):
    """Per POSITION and epoch: session dots coloured by animal, with mean +/- SEM.

    Priya, 2026-09-19: *"can we do this per-position and plot the results with mean +- SEM and dots
    per session, across animals?"*

    **THE DOTS ARE THE POINT, NOT DECORATION.** Four animals contribute unequal numbers of sessions
    to each epoch, and one animal can carry a cell on its own -- acute far-contra is thin in event
    count for every animal and absent for some. A mean with an error bar hides that; a mean with
    the sessions drawn under it does not, and the reader can see immediately whether a "difference"
    is four animals agreeing or one animal with six sessions.

    **SEM ACROSS SESSIONS, STATED PLAINLY AS THE WRONG ERROR BAR FOR A COHORT CLAIM.** Sessions
    within an animal are not independent, so this SEM is narrower than an animals->sessions
    bootstrap CI and must not be read as one. The bootstrap is what `epoch_summary` prints and what
    any claim should cite; this figure exists to show the DISTRIBUTION, and an SEM is the
    conventional companion to a dot plot. Both are on the page so neither can be mistaken.

    **READ THE ROWS IN ORDER.** Row 3 is the coupling gain and it is the one people will look at
    first; rows 1 and 2 are there because a ratio can rise by losing its denominator, which after a
    stroke is exactly what one would expect the 470 response to do.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config

    rows = [x for x in stats if x["arm"] == arm]
    if not rows:
        return None
    eps = [e for e in ("pre", "acute", "subacute", "chronic")
           if any(x["epoch"] == e for x in rows)]
    poss = [p for p in _POSITION_ORDER if any(x["position"] == p for x in rows)]
    if not eps or not poss:
        return None
    colors = config.animal_color()
    animals = sorted({x["animal"] for x in rows})
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(len(_PANELS), len(poss),
                             figsize=(2.55 * len(poss) + 1.6, 2.9 * len(_PANELS) + 1.9),
                             squeeze=False, sharex=True, sharey="row")
    fig.subplots_adjust(top=0.80, bottom=0.07, hspace=0.30, wspace=0.16)
    for ci, pos in enumerate(poss):
        for ri, (key, ylab) in enumerate(_PANELS):
            ax = axes[ri][ci]
            for xi, e in enumerate(eps):
                v = [x for x in rows if x["position"] == pos and x["epoch"] == e
                     and np.isfinite(x[key])]
                if not v:
                    continue
                y = np.array([float(x[key]) for x in v])
                # JITTER IS SEEDED so the same session lands in the same place in every panel and
                # a reader can follow it down the column.
                jx = xi + (rng.random(y.size) - 0.5) * 0.30
                for an in animals:
                    k = [i for i, x in enumerate(v) if x["animal"] == an]
                    if k:
                        ax.scatter(jx[k], y[k], s=13, alpha=0.55, linewidths=0,
                                   color=colors.get(an, "0.5"),
                                   label=an if (ri == 0 and ci == 0 and xi == 0) else None)
                m = float(np.mean(y))
                se = float(np.std(y, ddof=1) / np.sqrt(y.size)) if y.size > 1 else 0.0
                ax.errorbar(xi, m, yerr=se, fmt="_", color="k", ms=22, lw=1.8,
                            capsize=5, zorder=5)
                ax.text(xi, ax.get_ylim()[0], f"{y.size}", ha="center", va="bottom",
                        fontsize=6.5, color="0.45")
            ax.set_xticks(range(len(eps)))
            ax.set_xticklabels(eps, fontsize=8, rotation=30, ha="right")
            ax.spines[["top", "right"]].set_visible(False)
            if key.startswith("r_415"):
                ax.axhline(0, color="0.6", lw=0.7)
            if ri == 0:
                ax.set_title(pos, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=8.5)
    h, lab = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="upper right", frameon=False, fontsize=9, ncol=len(h),
                   bbox_to_anchor=(0.995, 0.845))
    fig.text(0.5, 0.995, f"415 vs 470 by SPOUT POSITION and EPOCH -- {arm}-aligned. "
             f"Dots = sessions, coloured by animal; bar = mean +/- SEM; small number = n sessions.",
             ha="center", va="top", fontsize=13, fontweight="bold")
    import textwrap
    cap = ("ROW 3 IS THE COUPLING MEASURE AND ROWS 1-2 ARE WHY IT CANNOT BE READ ALONE. "
             "||415||/||470raw|| is haemodynamic response per unit neural response -- but a ratio "
             "RISES WHEN ITS DENOMINATOR FALLS, and a stroke is expected to lower the 470 response. "
             "If row 3 goes up while row 1 goes down and row 2 holds, that is a shrinking "
             "denominator, not better coupling.\n"
             "ROWS 1 AND 2 CARRY THE CROSS-DAY SCALING CONFOUND (expression, bleaching, window "
             "clarity) that the ratio exists to cancel -- `crossday_intensity` owns it -- so read "
             "them as diagnostics for row 3, never as amplitudes in their own right.\n"
             "THE SEM IS ACROSS SESSIONS AND IS THE WRONG ERROR BAR FOR A COHORT CLAIM: sessions "
             "within an animal are not independent, so it runs narrower than the "
             "animals->sessions bootstrap CI that `epoch_summary` prints. Cite the bootstrap; read "
             "the dots for whether four animals agree or one animal carries the cell.\n"
             "ROW 5 IS THE CONTROL FOR ROW 4. r(415, 470) rising acutely looks like stronger "
             "coupling, but acute maps are also more GLOBAL (15k puts the cue acute panel at ~75% "
             "global) and two broad blobs correlate for reasons unrelated to coupling. Row 5 "
             "subtracts r(415 here, 470 at the OTHER positions): if the rise survives it is "
             "position-specific; if row 5 is flat at zero the rise was globalness.")
    # WRAPPED TO THE FIGURE, not left to run off it. At six position columns the caption ran off
    # both edges at a size nobody can read.
    wrapped = "\n".join("\n".join(textwrap.wrap(p, 155)) for p in cap.split("\n"))
    fig.text(0.5, 0.962, wrapped, ha="center", va="top", fontsize=8.6)
    out = out_dir / f"channel_position_epoch_{arm}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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
            for key in ("amp_415_over_raw_rms", "r_415_raw"):
                d = defaultdict(list)
                for x in v:
                    if np.isfinite(x[key]):
                        d[x["animal"]].append(float(x[key]))
                got[key] = _boot_ci(d, rng)
            if any(g is None for g in got.values()):
                continue
            a_, r_ = got["amp_415_over_raw_rms"], got["r_415_raw"]
            print(f"  {arm:<7}{e:<10}{len(v):>4}"
                  f"{a_[0]:>16.3f} [{a_[1]:.3f},{a_[2]:.3f}]"
                  f"{r_[0]:>16.3f} [{r_[1]:.3f},{r_[2]:.3f}]")
            rows.append(dict(arm=arm, epoch=e, n_cells=len(v),
                             amp_ratio=round(a_[0], 4), amp_lo=round(a_[1], 4),
                             amp_hi=round(a_[2], 4), r_415_raw=round(r_[0], 4),
                             r_lo=round(r_[1], 4), r_hi=round(r_[2], 4)))
    # THE CONTRAST, WHICH IS WHAT "DID COUPLING CHANGE" ACTUALLY ASKS. Per-epoch CIs that overlap
    # are not a test, and reading two overlapping intervals as "no difference" is a known way to be
    # wrong. Priya, 2026-09-19: *"increased / decreased coupling at all positions would be of
    # interest if it bears out"* -- "bears out" is a contrast with an interval on it.
    #
    # PAIRED WITHIN ANIMAL, resampling animals and then sessions within animal on BOTH sides of the
    # difference together, so an animal that contributes many sessions to one epoch and few to the
    # other cannot drive the contrast through its own mean.
    print(f"\n{bar}\nCHANGE FROM PRE -- nested animals->sessions bootstrap of the DIFFERENCE\n{bar}")
    print(f"  {'arm':<7}{'epoch':<10}{'d coupling gain':>28}{'d spatial match':>28}")
    for arm in ("cue", "lick"):
        for e in [x for x in eps if x != "pre"]:
            line = f"  {arm:<7}{e:<10}"
            for key in ("amp_415_over_raw_rms", "r_415_raw"):
                a_pre, a_ep = defaultdict(list), defaultdict(list)
                for x in stats:
                    if x["arm"] != arm or not np.isfinite(x[key]):
                        continue
                    if x["epoch"] == "pre":
                        a_pre[x["animal"]].append(float(x[key]))
                    elif x["epoch"] == e:
                        a_ep[x["animal"]].append(float(x[key]))
                shared = sorted(set(a_pre) & set(a_ep))
                if not shared:
                    line += f"{'--':>28}"
                    continue
                obs = float(np.mean([np.mean(a_ep[an]) - np.mean(a_pre[an]) for an in shared]))
                draws = []
                for _ in range(4000):
                    pick = [shared[i] for i in rng.integers(0, len(shared), len(shared))]
                    d = []
                    for an in pick:
                        p, q_ = a_pre[an], a_ep[an]
                        d.append(np.mean([q_[i] for i in rng.integers(0, len(q_), len(q_))])
                                 - np.mean([p[i] for i in rng.integers(0, len(p), len(p))]))
                    draws.append(float(np.mean(d)))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                star = " *" if (lo > 0 or hi < 0) else "  "
                line += f"{obs:>+14.3f} [{lo:+.3f},{hi:+.3f}]{star}"
            print(line)
    print("\n  * = 95% CI of the DIFFERENCE excludes zero. Paired within animal; only animals")
    print("    contributing to BOTH epochs enter the contrast.")

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


def _win_avg_base(S, ev, bl, w, pre_n):
    """Per-event ``mean(ev -> ev+w) - mean(bl-pre_n -> bl)``, averaged over events.

    **THE BASELINE IS THE PRE-CUE WINDOW OF THE SAME TRIAL, FOR BOTH ARMS**, and it went back in
    on 2026-09-19 after being dropped earlier the same day. The reasoning for dropping it was that
    the maps are ALREADY dF/F -- `approximate_svd` divides by `frames_average` -- so the session
    mean image is the baseline and a second one is redundant. That is right about the CONSTANT and
    wrong about the DRIFT away from it.

    WHY THE DRIFT BITES, and it is not simply "there is drift". Because F0 is the session mean, the
    mean of dF/F over the WHOLE session is ~0 by construction, so a trial set spread evenly over
    the session picks up no offset at all. **THE EXPOSURE IS UNEVEN SAMPLING IN TIME.** Engagement
    declines within a session, so the trials that survive gating sit EARLY, where a decaying trace
    is still above its session mean -- a positive offset. Measured over 113 sessions, 415 falls
    -11.6% within a session against 470's -4.8% (415 falls further in 111/113), so the offset is
    **2.4x larger in 415 than in 470**. And engagement collapses sooner after the stroke, so the
    bias is EPOCH-DEPENDENT: exactly the shape that manufactures an epoch effect from nothing.

    A pre-cue window cancels it because the drift is an exponential with a tens-of-minutes time
    constant and the trial window is ~4 s, over which it is flat.

    THE LICK ARM BASELINES TO ITS TRIAL'S PRE-CUE WINDOW, NOT TO PRE-LICK. Two seconds before a
    first post-cue lick sits inside the cue response, so it is not a baseline. Using the ITI window
    also makes the two arms share a baseline and therefore become comparable, which they were not
    when cue was a difference and lick an absolute window.

    ONLY THE AMPLITUDE TERMS NEED THIS. The correlations spatially mean-remove before correlating,
    so a uniform offset already cancels there -- `_quantify`'s r columns are unaffected either way.
    """
    acc = np.zeros(S.shape[0], np.float64)
    n = 0
    for e, b in zip(ev, bl):
        if b - pre_n < 0 or e + w > S.shape[1]:
            continue
        acc += np.asarray(S[:, e:e + w]).mean(1) - np.asarray(S[:, b - pre_n:b]).mean(1)
        n += 1
    return acc / max(n, 1)


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
    raw_v = [np.asarray(x)[m].astype(np.float64) for x in maps]
    v = [x - x.mean() for x in raw_v]

    def r(i, j):
        d = np.linalg.norm(v[i]) * np.linalg.norm(v[j])
        return float(v[i] @ v[j] / d) if d > 0 else float("nan")

    # SPATIAL SD AND SPATIAL MEAN, SEPARATELY -- and the naming matters (Priya, 2026-09-19: *"is rms
    # the sd?"*). Yes: `||v||/sqrt(n)` on a MEAN-REMOVED vector IS the population SD, and the column
    # was called `rms_*` while measuring only the structured part. True RMS is `mean^2 + sd^2`, and
    # **THE MEAN TERM IS ~75% OF IT** (measured PS94_0817: 470 raw mean 0.0228 against SD 0.0130).
    #
    # That is not a naming quibble for the coupling question. The haemodynamic response is largely
    # GLOBAL -- within-channel globalness runs 0.65-0.92 -- so a gain ratio built on SD alone is
    # computed on the quarter of the signal where haemodynamics is least represented, and a stroke
    # that changed the uniform response would leave it flat. All three are now reported.
    mu = [float(x.mean()) for x in raw_v]
    n415, n470, ncorr = (float(np.linalg.norm(x)) / np.sqrt(max(x.size, 1)) for x in v)
    rms = [float(np.sqrt(m * m + sd * sd)) for m, sd in zip(mu, (n415, n470, ncorr))]
    return dict(r_415_raw=round(r(0, 1), 4), r_415_corr=round(r(0, 2), 4),
                r_raw_corr=round(r(1, 2), 4),
                # THE NUMERATOR AND DENOMINATOR SEPARATELY, because a RATIO CAN RISE BY LOSING ITS
                # DENOMINATOR. After the stroke the 470 response falls, so `amp_415_over_raw` going
                # up is equally consistent with "more vascular response per unit neural response"
                # and with "the same vascular response against less neural response" -- opposite
                # readings of the same number. Only these two columns separate them.
                #
                # THEY CARRY THE CROSS-DAY SCALING CONFOUND THE RATIO WAS BUILT TO CANCEL
                # (expression, bleaching, window clarity -- `crossday_intensity` owns it), so they
                # are DIAGNOSTIC for reading the ratio, not a measurement in their own right.
                sd_415=round(n415, 6), sd_raw=round(n470, 6), sd_corr=round(ncorr, 6),
                mean_415=round(mu[0], 6), mean_raw=round(mu[1], 6), mean_corr=round(mu[2], 6),
                rms_415=round(rms[0], 6), rms_raw=round(rms[1], 6), rms_corr=round(rms[2], 6),
                amp_415_over_raw_rms=(round(rms[0] / rms[1], 4) if rms[1] > 0 else float("nan")),
                amp_415_over_raw=round(n415 / n470, 4) if n470 > 0 else float("nan"),
                amp_corr_over_raw=round(ncorr / n470, 4) if n470 > 0 else float("nan"),
                # slope of 470 on 415 across PIXELS: how much of the raw map a scaled 415 accounts
                # for. Paired with r_415_raw because a slope without a fit quality means nothing.
                beta_415_on_raw=round(float(v[0] @ v[1] / (v[0] @ v[0])), 4)
                if v[0] @ v[0] > 0 else float("nan"))


def _offdiag_r(built, mask):
    """``{code: mean r(415 at this position, 470 at the OTHER positions)}`` -- the globalness null.

    **THE CONTROL FOR THE READING I COULD NOT OTHERWISE EXCLUDE.** `r(415, 470)` rises acutely at
    several positions, which looks like stronger neurovascular coupling. But acutely the maps also
    become more GLOBAL -- `15k` measures the cue acute panel at ~75% global -- and two broad, smooth
    maps correlate highly for reasons that have nothing to do with coupling. Mean-centring removes a
    DC offset; it does not remove a shared low-spatial-frequency pattern.

    THE MISMATCHED PAIR IS THE NULL. If 415 and 470 agree because the vasculature tracks activity
    AT THAT POSITION, the diagonal `r(415_i, 470_i)` beats the off-diagonal `r(415_i, 470_j)`. If
    they agree because both maps are the same broad blob, the two are EQUAL and the difference is
    zero. So `r_415_raw - r_415_raw_offdiag` is position-specific coupling with globalness
    differenced out, and it costs one extra correlation per pair rather than a new analysis.

    Cheap because every position's maps are already built for this session.
    """
    m = np.asarray(mask, bool)
    vec = {}
    for code, maps in built.items():
        a = np.asarray(maps[0])[m].astype(np.float64)
        b = np.asarray(maps[1])[m].astype(np.float64)
        vec[code] = (a - a.mean(), b - b.mean())
    def _mean_r(u, others):
        nu = np.linalg.norm(u)
        rs = []
        for w in others:
            d = nu * np.linalg.norm(w)
            if d > 0:
                rs.append(float(u @ w / d))
        return float(np.mean(rs)) if rs else float("nan")

    out = {}
    for code in built:
        a415, a470 = vec[code]
        oth = [c for c in built if c != code]
        out[code] = dict(
            # CROSS-channel off-diagonal: the globalness null for r(415, 470).
            cross=_mean_r(a415, [vec[c][1] for c in oth]),
            # WITHIN-channel off-diagonals: how alike this channel's own six position maps are.
            # THESE EXIST BECAUSE THE CROSS TERM RISING IS NOT SELF-EXPLANATORY (Priya, 2026-09-19:
            # *"wouldn't increased non-position-specific coupling after stroke be interesting
            # too?"* -- yes, and treating it only as a nuisance assumed the answer).
            #
            #     within_470 rises      the ACTIVITY became more global. 15k measures the cue
            #                           acute panel at ~75% global, so this is expected.
            #     within_415 rises MORE the VASCULAR response lost specificity beyond what the
            #                           neural change explains -- a damaged neurovascular unit,
            #                           and the interesting outcome.
            #     both rise together    the vasculature is still faithfully tracking a signal that
            #                           itself became diffuse. No NVC change.
            within_470=_mean_r(a470, [vec[c][1] for c in oth]),
            within_415=_mean_r(a415, [vec[c][0] for c in oth]))
    return out


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
                   stats, figures=True, base_by_code=None):
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
        # BOTH ARMS: 2 s from their own event, MINUS that trial's pre-cue window. See
        # `_win_avg_base` for why the baseline went back in after being dropped earlier today --
        # the short version is that dF/F's F0 is the session mean, so only UNEVEN SAMPLING IN TIME
        # produces an offset, and the gating makes the sampling uneven in an epoch-dependent way.
        w = post_n if arm == "cue" else lpost
        bl = (base_by_code or {}).get(code)
        if bl is None or len(bl) != len(fr):
            raise ValueError(f"{s[chr(39)+chr(108)+chr(97)+chr(98)+chr(101)+chr(108)+chr(39)]}: "
                             f"baseline frames missing for {code}")
        built[code] = [_weighted_map(U, _win_avg_base(S, fr, bl, w, pre_n).astype(np.float32))
                       for _n, S in sig]
    lim = _limit([m for ms in built.values() for m in ms], mask) if built else 1e-6
    offd = _offdiag_r(built, mask) if len(built) > 1 else {}

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
        o = offd.get(code) or {}
        od = o.get("cross", float("nan"))
        q["r_415_raw_offdiag"] = round(od, 4) if np.isfinite(od) else float("nan")
        # POSITION-SPECIFIC COUPLING: the diagonal minus the globalness null. See `_offdiag_r`.
        q["r_415_specific"] = (round(q["r_415_raw"] - od, 4)
                               if np.isfinite(od) and np.isfinite(q["r_415_raw"])
                               else float("nan"))
        w470, w415 = o.get("within_470", float("nan")), o.get("within_415", float("nan"))
        q["within_470_offdiag"] = round(w470, 4) if np.isfinite(w470) else float("nan")
        q["within_415_offdiag"] = round(w415, 4) if np.isfinite(w415) else float("nan")
        # EXCESS VASCULAR GLOBALNESS: how much more alike the 415 maps are across positions than
        # the 470 maps are. > 0 and RISING after the stroke is loss of vascular spatial
        # specificity that the neural change does not account for.
        q["excess_415_globalness"] = (round(w415 - w470, 4)
                                      if np.isfinite(w415) and np.isfinite(w470)
                                      else float("nan"))
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

    from wfield_local.framemap_event_maps import in_trial_mask, trial_end_samples
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    lick_s = np.asarray(lk["lick_samples"])
    lick_f = np.searchsorted(f_of, lick_s)
    # A LICK INHERITS THE POSITION OF THE CUE IT FOLLOWS. Anything before the first cue has none.
    j = np.searchsorted(np.asarray(cs), lick_s, side="right") - 1
    lick_codes = np.where(j >= 0, codes[np.clip(j, 0, None)], -1)

    # IN-TRIAL LICKS ONLY, and this module shipped WITHOUT it for a few hours (2026-09-19). Priya:
    # *"what is this run gated on? engaged-only trials?"* -- it was gated on nothing, which
    # reproduced exactly the artefact `framemap_event_maps.in_trial_mask` exists to remove and whose
    # severity this repo has already measured:
    #
    #     PS94 8/17 far_center and far_R had ZERO responses yet contributed 93 and 83 licks to
    #     their maps, at a median of 7.2 and 7.6 s after the cue -- by which time the NEXT spout
    #     had moved into place. 18% of licks pre-stroke at every position, against
    #     28/47/50/100/100% post-stroke.
    #
    # **THE CONTAMINATION IS GRADED BY SEVERITY, SO IT TRACKS THE VERY DEFICIT IT WOULD BE READ AS
    # EVIDENCE FOR** -- which makes it disqualifying for the `--epochs` arm specifically, not merely
    # untidy. The deck's own lick maps are titled "in-trial licks only" for this reason.
    #
    # Reward-consumption licks are KEPT: `trial_end` lands after the response window.
    te = trial_end_samples(s["h5"])
    n_all = int(lick_codes.size)
    if te is None or not len(te):
        print(f"      !! {s['label']}: no trial_end channel -- lick arm would be UNGATED, skipping "
              f"it rather than shipping the contaminated version", flush=True)
        in_trial = np.zeros(lick_s.shape, bool)
    else:
        in_trial = in_trial_mask(lick_s, np.asarray(cs), np.asarray(te))
    kept = int((in_trial & (lick_codes >= 0)).sum())
    print(f"      {s['label']}: in-trial licks {kept}/{n_all} "
          f"({100.0 * kept / max(n_all, 1):.0f}%)", flush=True)

    # THE TERMINAL QUIT PERIOD IS EXCLUDED FROM BOTH ARMS, using the same
    # `precue_engagement_states.engagement_gate` the rest of the deck uses via
    # `beta_maps._quit_mask`. A cue presented after the animal has stopped working is a real
    # stimulus, so this is not a labelling error the way an out-of-trial lick is -- but the quit
    # period is 3.1% of frames pre-stroke and 18.7% acutely (`docs/REST_ENGAGEMENT_AUDIT.md`), so an
    # UNGATED epoch comparison has its composition track the independent variable. That is the one
    # thing an epoch contrast cannot tolerate.
    not_engaged = _quit_trials(s, cs, codes, lick_s)

    pre_n, post_n = int(round(CUE_PRE_S * FS)), int(round(CUE_POST_S * FS))
    lpost = max(1, int(round(LICK_POST_S * FS)))
    cue_ok = (codes >= 0) & (cue_f >= pre_n) & (cue_f + post_n <= T) & ~not_engaged

    # ONE EVENT PER TRIAL: THE FIRST POST-CUE IN-TRIAL LICK (Priya, 2026-09-19). Averaging over
    # EVERY lick in a bout weights a trial by how much the animal licked, which is itself the
    # dependent variable -- a trial with 20 licks counted 20 times and a single-lick trial once.
    ok_l = in_trial & (lick_codes >= 0) & (lick_f >= 0) & (lick_f + lpost <= T)
    first_f, first_code, first_trial = [], [], []
    seen = set()
    for idx in np.flatnonzero(ok_l):                 # lick_s is sorted, so the first hit wins
        t = int(j[idx])
        if t in seen or not_engaged[t]:
            continue
        seen.add(t)
        first_f.append(int(lick_f[idx]))
        first_code.append(int(lick_codes[idx]))
        first_trial.append(t)
    first_f = np.asarray(first_f, np.int64)
    first_code = np.asarray(first_code, np.int64)
    first_trial = np.asarray(first_trial, np.int64)


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
    # EACH EVENT'S BASELINE IS ITS OWN TRIAL'S PRE-CUE WINDOW. For the cue arm the event IS the
    # cue; for the lick arm the event is the first post-cue lick and the baseline is still that
    # trial's pre-cue window -- pre-LICK would sit inside the cue response. See `_win_avg_base`.
    first_cue_f = cue_f[first_trial]
    for arm, frames, valid, cds, base in (
            ("cue", cue_f, cue_ok, codes, cue_f),
            ("lick", first_f, np.ones(first_f.shape, bool), first_code, first_cue_f)):
        by = {c: frames[valid & (cds == c)] for c in order}
        bb = {c: base[valid & (cds == c)] for c in order}
        p = session_figure(s, out_dir, arm, by, order, labels, sig, edges, U, note, mask, stats,
                           figures=figures, base_by_code=bb)
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
        # PRE TAKES THE **LAST** SESSIONS, EVERY OTHER EPOCH THE FIRST, and that asymmetry is the
        # whole point. Taking the first of each put PRE at 6-7 JUNE against ACUTE at 17-19 AUGUST,
        # a ten-week gap -- while the pre-stroke set runs through 14 AUGUST, three days before the
        # acute sessions. The first run showed the 470 map RMS TRIPLING from pre to acute, which
        # would be a startling result and is almost certainly the CROSS-DAY MULTIPLICATIVE SCALING
        # this project already documents (expression, bleaching, window clarity;
        # `crossday_intensity` owns it) reading as physiology across a ten-week baseline gap.
        #
        # Every epoch should sit as close to the lesion as its definition allows: pre from the
        # end of its window, post epochs from the start of theirs.
        from wfield_local import epochs as ep_mod
        want = set(config.phase_labels("pre") + config.phase_labels("post"))
        by_key = {}
        for x in config.load_sessions():
            if x["label"] not in want:
                continue
            e = ep_mod.epoch_of(x["label"])
            if not e:
                continue
            by_key.setdefault((config.animal_of(x["label"]), e), []).append(x)
        pick = []
        for (_an, e), xs in by_key.items():
            xs = sorted(xs, key=lambda y: y["label"])
            pick += xs[-a.per_epoch:] if e == "pre" else xs[: a.per_epoch]
        pick.sort(key=lambda y: y["label"])
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
        for arm in ("cue", "lick"):
            p = position_epoch_figure(stats, out_dir, arm)
            if p is not None:
                print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
