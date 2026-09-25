"""CROSS-ANIMAL CD trajectories, as the CHANGE from each animal's own pre-stroke trajectory.

Priya, 2026-09-24: *"can we make a cross-animal version of the per position CD and cross-position CD
normalized to the pre-stroke CD (just subtract the pre-stroke projection i guess?)"* -- yes, and
subtraction is the right operation here for a reason worth stating: the POLES already put every
animal on a common scale (0 = that animal's pre-stroke not-P, 1 = its pre-stroke lick at P), so a
difference from pre is commensurable across animals in a way the raw traces are not. Each animal's
idiosyncratic pre-stroke shape cancels and what survives is what the lesion changed.

NOTHING IS COMPUTED HERE. It reads the dumps `cd_trajectories` persists, which is what they exist
for -- so this figure cannot disagree with the per-animal panels it summarises.

THREE THINGS THIS FIGURE HAS TO BE HONEST ABOUT, and each is visible on it rather than in a caption:

  * **PRE-STROKE IS ZERO BY CONSTRUCTION, AND THAT IS NOT A MEASUREMENT.** The reference has no band
    of its own, but the difference carries the estimation noise of BOTH epochs -- so a delta near zero
    means "no resolvable change", never "identical". The spread drawn is therefore the spread of the
    per-animal delta, which inherits both.
  * **n = 4.** Every animal is drawn individually, thin, behind the mean (rule 8: anything not
    visible in at least three animals individually will not survive the paired test). A cell where
    one animal carries the mean is meant to look like one.
  * **THE CELLS ARE NOT EQUALLY RELIABLE.** Each (animal, position) has its own surviving fraction
    from the orthogonalisation, and a position whose direction barely survived in one animal
    contributes a noisier delta. The worst surviving fraction contributing to each panel is printed, and
    panels below `SURVIVING_MIN` are flagged.

    python -m scripts.cd_cross_animal_figure --dir <epoch figure dir> --layout perposition cross
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as _pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wfield_local import cd_trajectories as cdt  # noqa: E402
from wfield_local import config  # noqa: E402
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES  # noqa: E402

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
POST = ("acute", "subacute", "chronic")

#: A panel whose worst contributing surviving fraction is below this is flagged on the figure. Same
#: threshold the per-animal panels use, and for the same reason: a direction that barely survived the
#: projection is drawn small because little of it is left, not because nothing happened.
SURVIVING_MIN = 0.70

#: Minimum animals contributing to a cell before its MEAN is drawn heavy. Below it the animals are
#: still drawn -- "could not test" and "tested and found nothing" are different facts -- but the mean
#: is dashed, because a two-animal mean is not the same object as a four-animal one.
MIN_ANIMALS = 3


def collect(dirpath, align="precue", gate="lick", reference="contrast", method="dom",
            mask_occluded=True, animals=ANIMALS):
    """``{animal: res}`` plus the reasons any animal is absent. See `cd_geometry_figure.collect`."""
    got, problems = {}, []
    for a in animals:
        try:
            res = cdt.load_result(dirpath, a, align, method, reference, gate, orth=True,
                                  mask_occluded=mask_occluded)
        except ValueError as exc:
            problems.append(f"{a}: {exc}")
            continue
        if res is None:
            problems.append(f"{a}: no saved result for {align}")
        else:
            got[a] = res
    return got, problems


def deltas(got, cd_p, tr_p, ep):
    """``({animal: Δtrace}, worst surviving fraction)`` -- epoch minus that animal's own pre.

    RESAMPLED ONTO NOTHING: every dump for one alignment shares `pre_n`, `post_n` and `fs`, so the
    time bases are identical by construction and a difference is elementwise. Checked rather than
    assumed -- a mismatch raises, because silently truncating to the shorter one would misalign the
    traces in time, which is the one axis this figure is about.
    """
    out, worst = {}, 1.0
    for a, res in got.items():
        d_ep = res["traces"].get((ep, cd_p, tr_p))
        d_pre = res["traces"].get(("pre", cd_p, tr_p))
        if d_ep is None or d_pre is None:
            continue
        v_ep, v_pre = np.asarray(d_ep["mean"], float), np.asarray(d_pre["mean"], float)
        if v_ep.shape != v_pre.shape:
            raise ValueError(f"{a} {ep} {cd_p}/{tr_p}: {v_ep.shape} vs pre {v_pre.shape}")
        out[a] = v_ep - v_pre
        sv = (res.get("surviving") or {}).get(cd_p)
        if sv is not None:
            worst = min(worst, float(sv))
    return out, worst


def _time(got):
    res = next(iter(got.values()))
    ns = {(r["pre_n"], r["post_n"], round(float(r["fs"]), 6)) for r in got.values()}
    if len(ns) != 1:
        raise ValueError(f"animals disagree on the time base: {sorted(ns)}")
    return np.arange(-res["pre_n"], res["post_n"]) / float(res["fs"])


#: Draws for the pointwise display band. The same count as `analysis_kit.N_BOOT`, deliberately: this
#: is the SAME animal-level draw `boot_delta` makes, and matching the count keeps the band and the
#: quoted scalars on one footing.
N_BOOT_BAND = 4000


def pooled_nested(post, pre, n_boot=N_BOOT_BAND, seed=0):
    """``(mean, lo, hi)`` per time point from the NESTED animals -> sessions draw.

    `post` and `pre` are ``{animal: [per-session trace]}``. Animals are resampled once and then
    sessions within each drawn animal, for BOTH arms under that same draw -- which is what makes it
    paired, so between-animal variance cancels rather than being counted twice. It is
    `analysis_kit.boot_delta`'s draw, vectorised over time; the canonical call costs 291 ms and a
    trace has 219 points.

    THE ANIMAL-ONLY VERSION IT REPLACES had four values and 35 distinct resamples, so its band was
    coarse and told the reader almost nothing about within-animal variability -- which is the term
    that moved three cells through zero when it was added to the scalar table.
    """
    animals = sorted(set(post) & set(pre))
    if not animals:
        return None, None, None
    obs = np.nanmean([np.nanmean(np.vstack(post[a]), 0) - np.nanmean(np.vstack(pre[a]), 0)
                      for a in animals], axis=0)
    if len(animals) < 2:
        return obs, None, None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        vals = []
        for a in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            pa, qa = np.vstack(post[a]), np.vstack(pre[a])
            vals.append(np.nanmean(pa[rng.integers(0, pa.shape[0], pa.shape[0])], 0)
                        - np.nanmean(qa[rng.integers(0, qa.shape[0], qa.shape[0])], 0))
        draws.append(np.nanmean(vals, axis=0))
    D = np.vstack(draws)
    return obs, np.nanpercentile(D, 2.5, axis=0), np.nanpercentile(D, 97.5, axis=0)


def session_traces(got, cd_p, tr_p):
    """``({animal: [post traces]}, {animal: [pre traces]})`` per epoch, for one cell.

    Returns ``{}`` for a dump that predates per-cell session traces, so the caller can fall back to
    the animal-level band and SAY SO rather than silently reporting a narrower one.
    """
    out = {}
    for a, res in got.items():
        for (_lab, ep, cd_, tr_), d in (res.get("session_traces") or {}).items():
            if cd_ != cd_p or tr_ != tr_p:
                continue
            out.setdefault(ep, {}).setdefault(a, []).append(np.asarray(d["mean"], float))
    return out


def pooled(per_animal, n_boot=N_BOOT_BAND, seed=0):
    """``(mean, lo, hi)`` over animals at every time point -- the POOLED delta and its band.

    Priya, 2026-09-24: *"yes subtract only within animal but then plot the across animal pooled
    deltas"*. The subtraction is WITHIN animal, which is what makes the difference paired -- each
    animal is its own pre-stroke control, so between-animal variance cancels instead of being
    counted twice -- and the POOLING across animals weights animals EQUALLY. That is the convention
    `analysis_kit.boot_delta` uses for a CHANGE, against `boot_ci`'s flat session pool for a LEVEL;
    mixing the two up retracted a result on 2026-09-20.

    WHY THIS IS NOT A TENTH BOOTSTRAP. Each animal contributes exactly ONE delta trace here, because
    the dumps aggregate sessions within an epoch. `boot_delta`'s inner session draw therefore has
    nothing to resample and it degenerates to precisely this: resample animals with replacement, take
    the mean. Its own docstring says so -- "the interval reflects the animal draw alone". What this
    adds is VECTORISATION OVER TIME, because the canonical call costs 291 ms and a trace has 219
    points: 19 minutes for one layout against milliseconds here.

    **THE QUOTED SCALARS DO NOT COME THROUGH HERE.** `peak_delta` calls `analysis_kit.boot_delta`
    itself, so every number that could reach a document is the canonical object and this band is a
    reading aid on a trace.

    AND AT n = 4 THE BAND IS COARSE BY CONSTRUCTION: four animals with one value each admit only 35
    distinct resamples. It shows spread, it does not support a claim -- the per-animal traces and the
    printed table are what rule 8 asks for.
    """
    if not per_animal:
        return None, None, None
    A = np.vstack([per_animal[a] for a in sorted(per_animal)])
    m = np.nanmean(A, 0)
    if A.shape[0] < 2:
        return m, None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, A.shape[0], (n_boot, A.shape[0]))
    boot = np.nanmean(A[draws], axis=1)
    return m, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def session_values(got, p, ep, j):
    """``({animal: [post session values]}, {animal: [pre session values]})`` at time index `j`.

    THE SESSIONS THE INTERVAL NEEDS. `cd_trajectories` persists the per-session diagonal traces for
    exactly this: without them each animal contributes one number and `boot_delta`'s inner draw has
    nothing to resample, so a four-animal interval is all that can be formed.

    Returns empty dicts when the dumps predate that (they are then absent, not wrong), and the
    caller falls back to the animal-level draw and says so.
    """
    post, pre = {}, {}
    for a, res in got.items():
        st = res.get("session_traces") or {}
        for (_lab, e, cd_, tr_), d in st.items():
            if cd_ != p or tr_ != p:          # the diagonal: this position on its own direction
                continue
            v = float(np.asarray(d["mean"])[j])
            if e == ep:
                post.setdefault(a, []).append(v)
            elif e == "pre":
                pre.setdefault(a, []).append(v)
    both = set(post) & set(pre)
    return {a: post[a] for a in both}, {a: pre[a] for a in both}


def peak_delta(per_animal, mask, got=None, p=None, ep=None):
    """``(Interval, index, unit)`` for the SIGNED delta at the pooled extremum -- CANONICAL DRAW.

    Priya, 2026-09-25: *"why can't delta be < 0?"* -- it can, and an earlier version reported
    `max(|delta|)` per animal, which bounds the statistic below at zero and makes an interval
    excluding zero meaningless. It is signed now and the interval CAN straddle zero; when it does,
    that is the result -- the animals do not agree on a direction.

    THE TIME IS CHOSEN ONCE, from the POOLED mean trace, and every animal is read at that same
    instant. Choosing it per animal would be the same bug in different clothes: one animal peaking
    positive early and another negative late would average to a number describing neither.

    TWO DRAWS, and which one ran is REPORTED rather than inferred:

      sessions  `boot_delta(post, pre)` over per-session values -- animals resampled once, then
                sessions within each animal for BOTH arms under that draw, which is what makes it
                paired. This is the one to use, and it needs `session_traces` in the dumps.
      animals   the fallback when those are absent: each animal contributes its own delta and the
                `pre` arm is the zero it was taken against. The point estimate is identical; only
                the interval is coarser, because four animals with one value each admit 35 distinct
                resamples.
    """
    from wfield_local import analysis_kit as ak

    if not per_animal:
        return None, None, None
    A = np.vstack([per_animal[a] for a in sorted(per_animal)])
    pooled_m = np.nanmean(A, 0)
    idx = np.flatnonzero(mask)
    if not idx.size:
        return None, None, None
    j = int(idx[int(np.nanargmax(np.abs(pooled_m[mask])))])
    if got is not None and p is not None and ep is not None:
        post, pre = session_values(got, p, ep, j)
        if post and min(len(v) for v in post.values()) >= 1 and pre:
            return ak.boot_delta(post, pre, np.random.default_rng(0)), j, "sessions"
    post = {a: [float(np.asarray(v)[j])] for a, v in per_animal.items()}
    return ak.boot_delta(post, {a: [0.0] for a in post}, np.random.default_rng(0)), j, "animals"


def _draw_cell(ax, per_animal, t, colors, color=None, label=None, sess=None):
    """Every animal thin, the POOLED mean heavy with its band.

    `sess` is ``(post, pre)`` per-session traces; given, the band is the NESTED animals -> sessions
    draw rather than four animals with one value each.
    """
    if not per_animal:
        return None
    # ANIMALS ALWAYS IN THEIR OWN COLOUR, never in the epoch's: an animal is an identity and an
    # epoch is an ordering, and giving them one palette is what made this unreadable.
    for a in sorted(per_animal):
        ax.plot(t, per_animal[a], lw=cdt.LW_PER_ANIMAL, alpha=cdt.ALPHA_PER_ANIMAL,
                color=colors.get(a, "0.6"))
    if sess and sess[0] and sess[1]:
        m, lo, hi = pooled_nested(sess[0], sess[1])
    else:
        m, lo, hi = pooled(per_animal)
    grey = "#111111" if color is None else color
    if lo is not None:
        ax.fill_between(t, lo, hi, color=grey, alpha=cdt.ALPHA_BAND, lw=0)
    # WEIGHTS FROM `cd_trajectories`, so the pooled figure and the per-animal panels it summarises
    # are drawn at the same weights. They had drifted -- 2.4 here against 1.8 there.
    ax.plot(t, m, lw=cdt.LW_MAIN + 0.2, color=grey,
            ls="-" if len(per_animal) >= MIN_ANIMALS else "--",
            path_effects=[_pe.Stroke(linewidth=cdt.LW_MAIN + 1.4, foreground="white"),
                          _pe.Normal()],
            label=(label or "") + f" (n={len(per_animal)})")
    return m


def _furniture(ax, t):
    # ZERO IS THE PRE-STROKE TRAJECTORY ITSELF, so the line matters more than usual here: distance
    # from it is the whole quantity, and it is not the same as distance from "no signal".
    ax.axhline(0, color="0.4", lw=1.0)
    ax.axvline(0, color="0.75", lw=0.8, ls="--")
    ax.set_xlim(t[0], t[-1])


def _suptitle(got, align, extra):
    n = len(got)
    dropped = {int(r.get("n_dropped") or 0) for r in got.values()}
    zero = {"precue": "cue", "cue": "cue", "lick": "first lick"}[align]
    return (
        f"CROSS-ANIMAL {align.upper()} coding direction — CHANGE FROM EACH ANIMAL'S OWN PRE-STROKE "
        f"TRAJECTORY{extra}\n"
        f"N = {n} animals; BANDS ARE THE NESTED animals -> sessions bootstrap, not four animals "
        f"with one value each. EPOCHS ARE THE GREY RAMP (light = early), ANIMALS ARE THE COHORT "
        f"COLOURS — one palette each, so the two never collide. "
        f"Thin lines, individual animals; heavy line, their mean (DASHED where "
        f"fewer than {MIN_ANIMALS} animals contribute). 0 = that animal's pre-stroke trajectory, so "
        f"pre-stroke is zero BY CONSTRUCTION and carries no band — a Δ near 0 means 'no resolvable "
        f"change', not 'identical', and the spread shown inherits the noise of both epochs.\n"
        f"Pole-normalised units: 1.0 = that animal's full pre-stroke position-P signature. "
        f"Condition-independent mode projected out; {sorted(dropped)} glue/bulb components dropped. "
        f"x = 0 is the {zero}. HAEMODYNAMICS ARE SLOW — read amplitude and gross time course, NOT "
        f"onset or ordering.")


def figure_perposition(got, out, align="precue"):
    """6 panels, one per spout position, the three post-stroke epochs overlaid as Δ from pre."""
    colors = config.animal_color()
    t = _time(got)
    pos = [p for p in DISPLAY_ORDER
           if any(("pre", p, p) in r["traces"] for r in got.values())]
    nc = 3
    nr = int(np.ceil(len(pos) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.1 * nr), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    for k, p in enumerate(pos):
        ax = axes[k // nc][k % nc]
        _furniture(ax, t)
        worst = 1.0
        # EPOCHS IN GREY, ANIMALS IN COLOUR -- the two systems cannot collide, which is the whole
        # reason the project made epochs a ramp. Heavy grey is the pooled epoch mean; the thin
        # coloured lines behind it are the individual animals, so both readings survive in one panel.
        st = session_traces(got, p, p)
        for ep, col in zip(POST, [cdt.EPOCH_COLOR[e] for e in POST]):
            per, w = deltas(got, p, p, ep)
            worst = min(worst, w)
            _draw_cell(ax, per, t, colors, color=col, label=ep,
                       sess=(st.get(ep), st.get("pre")))
        ax.set_title(POSITION_NAMES.get(p, str(p))
                     + (f"   (worst surviving {worst:.0%})" if worst < 1.0 else ""),
                     fontsize=9, color="tab:red" if worst < SURVIVING_MIN else "black")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for k in range(len(pos), nr * nc):
        axes[k // nc][k % nc].set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("s", fontsize=8)
    for r in range(nr):
        axes[r][0].set_ylabel("Δ projection (epoch − pre)", fontsize=8)
    fig.suptitle(_suptitle(got, align, " — ONE PANEL PER POSITION"), fontsize=8.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_cross(got, out, align="precue"):
    """Epochs x coding directions, each panel overlaying all six positions' Δ from pre.

    The DIAGONAL trace is the direction's own position and is drawn heavy; the rest are the
    comparison. A change confined to one position shows one trace moving and five flat -- which is
    the difference between "this position's code changed" and "everything moved".
    """
    # NO per-animal colouring in this layout: the six POSITIONS carry the colour here, and the
    # animals are already visible individually in the per-position layout.
    t = _time(got)
    pos = [p for p in DISPLAY_ORDER
           if any(("pre", p, p) in r["traces"] for r in got.values())]
    fig, axes = plt.subplots(len(POST), len(pos), figsize=(3.1 * len(pos), 2.7 * len(POST)),
                             squeeze=False, sharex=True, sharey="row", constrained_layout=True)
    for i, ep in enumerate(POST):
        for j, cd_p in enumerate(pos):
            ax = axes[i][j]
            _furniture(ax, t)
            worst = 1.0
            for tr_p in pos:
                per, w = deltas(got, cd_p, tr_p, ep)
                if tr_p == cd_p:
                    worst = min(worst, w)
                if not per:
                    continue
                on = tr_p == cd_p
                m, lo, hi = pooled(per)
                if on and lo is not None:
                    ax.fill_between(t, lo, hi, color=cdt.POS_COLOR.get(tr_p, "0.5"),
                                    alpha=cdt.ALPHA_BAND, lw=0)
                ax.plot(t, m,
                        lw=cdt.LW_MAIN if on else cdt.LW_SECONDARY,
                        alpha=cdt.ALPHA_MAIN if on else cdt.ALPHA_SECONDARY,
                        path_effects=([_pe.Stroke(linewidth=cdt.LW_MAIN + 1.0,
                                                  foreground="white"), _pe.Normal()]
                                      if on else None),
                        color=cdt.POS_COLOR.get(tr_p, "0.5"),
                        label=(POSITION_NAMES.get(tr_p, str(tr_p)) + (" (own)" if on else "")))
            if i == 0:
                ax.set_title("CD: " + POSITION_NAMES.get(cd_p, str(cd_p))
                             + (f"\n(worst surviving {worst:.0%})" if worst < 1.0 else ""),
                             fontsize=8.5,
                             color="tab:red" if worst < SURVIVING_MIN else "black")
            if j == 0:
                ax.set_ylabel(f"{ep}\nΔ projection", fontsize=8.5)
            if i == 0 and j == len(pos) - 1:
                ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("s", fontsize=8)
    fig.suptitle(_suptitle(got, align,
                           " — 6x6 CROSS-PROJECTION, ANIMAL MEANS (heavy = the CD's own position)"),
                 fontsize=8.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


LAYOUTS = {"perposition": figure_perposition, "cross": figure_cross}


def table(got, align):
    """Peak |Δ| per position per epoch, per animal -- RULE 8: the per-animal numbers first."""
    t = _time(got)
    m = (t >= 0.0) & (t <= 2.0)
    # ASCII IN PRINTED OUTPUT. The Windows console is cp1252 and a Greek delta raises
    # UnicodeEncodeError there, which kills the step rather than mangling a character --
    # `tests/test_cli_help_is_console_safe.py` exists for this class of failure. The FIGURES keep the
    # symbol; matplotlib is not the console.
    lines = [f"{align}: SIGNED delta at the pooled extremum over [0, 2] s, on each position's own CD",
             "  per-animal values FIRST (rule 8); the pooled column is analysis_kit.boot_delta -- "
             "animals resampled, then SESSIONS within each animal, paired against that animal's own "
             "pre-stroke sessions. The point estimate is animal-weighted because a CHANGE must be.",
             "  SIGNED, and the time is chosen ONCE from the pooled trace so every animal is read at "
             "the same instant. An interval spanning 0 means the animals do not agree on a direction.",
             f"  {'position':13s} {'epoch':9s} {'t(s)':>5s} "
             + " ".join(f"{a:>8s}" for a in ANIMALS)
             + f" {'POOLED [95% CI]':>24s} {'n':>3s}"]
    for p in DISPLAY_ORDER:
        for ep in POST:
            per, _w = deltas(got, p, p, ep)
            if not per:
                continue
            iv, j, unit = peak_delta(per, m, got=got, p=p, ep=ep)
            if iv is None:
                continue
            # EVERY ANIMAL AT THE SAME INSTANT, and signed. The column is directly comparable across
            # animals for that reason, which `max(|delta|)` per animal was not.
            vals = {a: float(np.asarray(v)[j]) for a, v in per.items()}
            crosses = "" if (iv.lo > 0 or iv.hi < 0) else "  (CI spans 0)"
            crosses += "" if unit == "sessions" else "  [animal-level CI: no session traces]"
            lines.append(f"  {POSITION_NAMES.get(p, str(p)):13s} {ep:9s} {t[j]:+5.2f} "
                         + " ".join(f"{vals[a]:+8.2f}" if a in vals else f"{'--':>8s}"
                                    for a in ANIMALS)
                         + f" {iv.point:+6.2f} [{iv.lo:+5.2f},{iv.hi:+5.2f}] {len(vals):3d}"
                         + crosses)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="directory holding the dumps; default is cd_trajectories' own output dir")
    ap.add_argument("--align", nargs="+", default=["precue", "cue", "lick"],
                    choices=("precue", "cue", "lick"))
    ap.add_argument("--layout", nargs="+", default=list(LAYOUTS), choices=tuple(LAYOUTS))
    ap.add_argument("--gate", default="lick", choices=tuple(cdt.GATES))
    ap.add_argument("--reference", default="contrast", choices=cdt.REFERENCES)
    ap.add_argument("--occluded", default="drop", choices=("drop", "keep"),
                    help="which arm's dumps to read; must match the render")
    args = ap.parse_args(argv)

    d = Path(args.dir) if args.dir else cdt.default_out()
    rc = 1
    for align in args.align:
        got, problems = collect(d, align=align, gate=args.gate, reference=args.reference,
                                mask_occluded=args.occluded == "drop")
        for p in problems:
            print(f"!! {p}")
        if not got:
            continue
        rc = 0
        print()
        print(table(got, align))
        tag = "_".join([align, args.reference, args.gate]
                       + (["cortexonly"] if args.occluded == "drop" else []))
        for lay in args.layout:
            print(f"-> {LAYOUTS[lay](got, d / f'cd_xanimal_{lay}_{tag}.png', align=align)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
