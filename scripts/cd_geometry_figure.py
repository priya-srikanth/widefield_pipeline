"""The condition-independent mode across epochs: does it ROTATE, does it GROW, and what leaks.

Priya, 2026-09-24, asked for the overlap number on its own figure rather than only in every CD
title. This draws it, and next to it the thing the overlap CANNOT see -- Priya: *"the cosine will
not read out amplitude changes though, right"*.

NOTHING IS COMPUTED HERE. It reads the results `cd_trajectories` persists under `<dir>/results/`, so
the figure and the CD panels are views of ONE arithmetic and cannot drift apart. Run the renders
first; a missing dump is reported by name, never silently skipped.

THREE ROWS, AND THE THIRD IS THE ONLY ONE THAT LICENSES A CLAIM ABOUT A PANEL.

  A  SUBSPACE OVERLAP with the pre-stroke condition-independent subspace, per epoch. **Drawn
     against its chance level of K/n**, which is ~0.02 on a 95-component basis -- so an overlap of
     0.61 is ~30x chance and the mode is strongly CONSERVED. Reporting it as "a third of the mode
     rotated away" was wrong, and it is the reason the chance band is on the axis rather than in a
     caption.
  B  MAGNITUDE, as a ratio to pre-stroke. Overlap is SCALE-INVARIANT: a response that keeps its
     orientation exactly and halves in size scores 1.00. For a lesion study that is a blind spot on
     the most likely effect, so orientation and magnitude are drawn one above the other.
  C  THE UNREMOVED SHARED AMPLITUDE, `sqrt(1 - overlap) x scale`. This is what `--orth` fails to
     project out of a post-stroke panel, in units of the pre-stroke shared response. The residual
     table in the CD handoff used `sqrt(1 - overlap)` alone, i.e. assumed row B was flat at 1.0 --
     so row C is the correction, and the dashed line marks the value that table assumed.

READ A AND B TOGETHER OR NEITHER:

    overlap ~1, scale ~1     nothing moved
    overlap ~1, scale < 1    same geometry, weaker drive
    overlap < 1, scale ~1    REORGANISATION -- the code went somewhere else
    both < 1                 mixed, and neither number alone would have said so

A decoder score conflates all four, because every one of them lowers accuracy. That is what the
subspace view adds over "can position still be read out".

    python -m scripts.cd_geometry_figure --dir "<the --out dir of cd_trajectories>"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wfield_local import cd_trajectories as cdt  # noqa: E402
from wfield_local import config  # noqa: E402

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
ALIGNS = ("precue", "cue", "lick")

#: Only the post-stroke epochs have a value to draw: both quantities are defined AGAINST pre, so
#: pre's own entries are 1.000 by construction and would flatten every axis they shared.
POST = ("acute", "subacute", "chronic")


def collect(dirpath, animals=ANIMALS, aligns=ALIGNS, gate="lick", reference="contrast",
            method="dom", mask_occluded=True):
    """`{(animal, align): res}` for every dump that exists, plus the reasons the others do not.

    A STALE DUMP RAISES inside `load_result` and is recorded as a problem rather than drawn -- see
    `cd_trajectories.RESULT_GUARD`. Missing and stale are different facts and both are reported,
    because "this animal has no CIM" and "this animal's CIM was measured under other constants"
    lead to opposite next actions.

    `mask_occluded` DEFAULTS TO TRUE because the render's does (Priya, 2026-09-24: *"i think for the
    CD analyses we should drop the masked components"*), and the tag differs between the two arms --
    so a default of False here would silently find nothing and report every animal as missing. It
    did, once: the first run after the default flipped reported "no saved result" for all four.
    """
    got, problems = {}, []
    for a in animals:
        for al in aligns:
            try:
                res = cdt.load_result(dirpath, a, al, method, reference, gate, orth=True,
                                      mask_occluded=mask_occluded)
            except ValueError as exc:
                problems.append(f"{a} {al}: {exc}")
                continue
            if res is None:
                problems.append(f"{a} {al}: no saved result")
            elif not res.get("cim_cos"):
                problems.append(f"{a} {al}: saved, but no cim_cos (was it rendered --orth on?)")
            else:
                got[(a, al)] = res
    return got, problems


def load_nulls(dirpath, aligns=ALIGNS, gate="lick", animals=ANIMALS):
    """``{(animal, align): {epoch: {...}}}`` from `cd_overlap_null`'s dumps, or an empty dict.

    ABSENT IS NOT AN ERROR HERE: the null is a separate, slower step, and a geometry figure without
    the ceiling is still the figure it was before -- it just cannot answer "is this more than
    estimation noise". Every panel says which case it is in, rather than looking the same either way.
    """
    from wfield_local import results_store as rs

    out = {}
    for a in animals:
        for al in aligns:
            body = rs.load(dirpath, "cd_overlap_null", f"{a}_{al}_{gate}")
            if body and body.get("epochs"):
                out[(a, al)] = body["epochs"]
    return out


def ceiling(nulls, animal, align, epoch, key):
    """The pre-to-pre value for one cell: ``(median, lo, hi)`` or ``(None, None, None)``.

    `key` is "cos" or "scale". The MEDIAN over draws, matching how `cd_overlap_null.report` prints
    it, so the figure and the table cannot disagree about what the ceiling is.
    """
    import numpy as _np

    d = (nulls.get((animal, align)) or {}).get(epoch)
    if not d:
        return None, None, None
    # `or []` WOULD EVALUATE THE ARRAY'S TRUTH VALUE and raise -- the dumps restore these fields as
    # ndarrays, not lists, so the falsy-default idiom does not survive the round trip.
    raw = d.get(f"null_{key}")
    v = _np.asarray([] if raw is None else raw, float)
    v = v[_np.isfinite(v)]
    if not v.size:
        return None, None, None
    return (float(_np.median(v)), float(_np.percentile(v, 2.5)), float(_np.percentile(v, 97.5)))


def leak(cos, scale):
    """`sqrt(1 - overlap) x scale` -- the shared amplitude `--orth` leaves in a post-stroke panel.

    Overlap is a POWER fraction, so the amplitude that survives the projection is its square root;
    the second factor is the magnitude the cosine is blind to. With `scale` absent this reduces to
    what the first residual estimate used, which is only right if the shared response did not
    change size -- and it grew.
    """
    if cos is None:
        return float("nan")
    return float(np.sqrt(max(1.0 - float(cos), 0.0)) * (1.0 if scale is None else float(scale)))


def table(got, nulls=None):
    """The printed report. THE TABLE IS THE RESULT; the figure is how it is read at a glance.

    THREE REFERENCE POINTS PER ROW, and they answer different questions: `chance` is K/n, what two
    unrelated subspaces score; `ceil` is the trial-matched PRE-TO-PRE value, what these quantities
    read when nothing changed; and `/ceil` is the observation as a fraction of it, where 1.00 means
    indistinguishable from no change. Quoting the observation alone was the original error here.
    """
    nulls = nulls or {}
    lines = [f"{'animal':7s} {'align':7s} {'K':>2s} {'chance':>7s} {'epoch':10s} "
             f"{'overlap':>8s} {'ceil':>6s} {'/ceil':>6s} "
             f"{'scale':>7s} {'ceil':>6s} {'/ceil':>6s} {'leak':>7s}"]
    for (a, al), res in sorted(got.items()):
        cos, scale = res.get("cim_cos") or {}, res.get("cim_scale") or {}
        for ep in POST:
            c, sc = cos.get(ep), scale.get(ep)
            if c is None:
                continue
            cc = ceiling(nulls, a, al, ep, "cos")[0]
            cs = ceiling(nulls, a, al, ep, "scale")[0]
            lines.append(
                f"{a:7s} {al:7s} {res.get('cim_k') or 0:2d} "
                f"{res.get('cim_chance', float('nan')):7.3f} {ep:10s} "
                f"{c:8.3f} {('   --' if cc is None else f'{cc:6.3f}')} "
                f"{('   --' if not cc else f'{c / cc:6.2f}')} "
                f"{('    n/a' if sc is None else f'{sc:7.3f}')} "
                f"{('   --' if cs is None else f'{cs:6.3f}')} "
                f"{('   --' if not (cs and sc) else f'{sc / cs:6.2f}')} "
                f"{leak(c, sc):7.3f}")
    if not nulls:
        lines.append("  NO CEILING AVAILABLE -- run `python -m wfield_local.cd_overlap_null` first. "
                     "Without it an overlap has no scale beyond K/n.")
    return "\n".join(lines)


#: Fraction of the figure width the suptitle may occupy. The rest is margin, and 0.94 is enough that
#: a line ending in a long word does not touch the edge.
TITLE_WIDTH_FRAC = 0.94


def _wrap(text, fig, fontsize=8.5):
    """Wrap each paragraph to the figure's width, using the renderer's OWN measurement.

    The suptitle is the same length whether one alignment was requested or three, while the figure is
    4.5 inches wide in the first case and 12 in the second -- so a fixed column count runs off the
    edge of the narrow one (Priya, 2026-09-25: *"itile is cut off"*).

    MEASURED, NOT ESTIMATED, and that distinction cost a second round: the first fix guessed 17
    characters per inch, where the font is 14.3 for 'x' and 15.2 for real prose at 8.5 pt, so lines
    came out ~10% too wide and the title still overflowed. `get_window_extent` knows; nothing here has
    to.
    """
    import textwrap

    avail = float(fig.get_size_inches()[0]) * TITLE_WIDTH_FRAC
    try:
        probe = fig.text(0.5, 0.5, "n" * 60, fontsize=fontsize)
        per_inch = 60.0 / (probe.get_window_extent(
            renderer=fig.canvas.get_renderer()).width / fig.dpi)
        probe.remove()
    except Exception:                                                  # noqa: BLE001
        per_inch = 14.3          # the measured value for this font, as a floor if the probe fails
    cols = max(30, int(avail * per_inch))
    return "\n".join(textwrap.fill(par, cols) for par in text.split("\n"))


def _legend(ax, normalise, nulls):
    """The animal legend, plus ONE entry explaining the dashed line -- in the legend, not a caption.

    A reader should not have to remember what a line style meant two panels ago, and they certainly
    should not have to discover that it meant something different there.
    """
    from matplotlib.lines import Line2D

    h, lab = ax.get_legend_handles_labels()
    if nulls and not normalise:
        h = [*h, Line2D([], [], color="0.35", ls="--", lw=1.0)]
        lab = [*lab, "that animal's pre-to-pre ceiling\n(what this reads at NO CHANGE)"]
    ax.legend(h, lab, fontsize=6.5, frameon=False, ncol=2, loc="lower left")


def _norm(vals, ceils):
    """`vals / ceils` elementwise, NaN where the ceiling is missing rather than silently 1.0."""
    out = []
    for v, c in zip(vals, ceils):
        out.append(float(v) / float(c) if (v is not None and c) else np.nan)
    return out


def _series(got, align, key, animals):
    out = {}
    for a in animals:
        res = got.get((a, align))
        if res is None:
            continue
        d = res.get(key) or {}
        out[a] = [d.get(ep) for ep in POST]
    return out


def figure(got, out, nulls=None, normalise=False):
    """`normalise=True` divides every quantity by its pre-to-pre ceiling (1.0 = no change)."""
    nulls = nulls or {}
    colors = config.animal_color()
    aligns = [al for al in ALIGNS if any(k[1] == al for k in got)]
    animals = [a for a in ANIMALS if any(k[0] == a for k in got)]
    if not aligns:
        raise SystemExit("nothing to draw: no saved result carried a cim_cos")
    fig, axes = plt.subplots(3, len(aligns), figsize=(3.7 * len(aligns) + 0.8, 8.6),
                             squeeze=False, sharex=True, constrained_layout=True)
    x = np.arange(len(POST))
    for j, al in enumerate(aligns):
        cos = _series(got, al, "cim_cos", animals)
        scale = _series(got, al, "cim_scale", animals)
        ch = [got[(a, al)].get("cim_chance") for a in animals if (a, al) in got]
        ch = max([c for c in ch if c is not None] or [float("nan")])

        ax = axes[0][j]
        if not normalise:
            # THE CHANCE BAND, not a caption. Two unrelated K-dim subspaces of an n-dim space overlap
            # at K/n, so the distance from the BAND is the finding and the distance from 1.0 is not.
            # This figure exists partly because that was briefly read the other way round.
            ax.axhspan(0, ch, color="0.85", lw=0, zorder=0)
            ax.text(len(POST) - 0.55, ch, f" chance K/n = {ch:.3f}", fontsize=6.5,
                    va="bottom", ha="right", color="0.35")
        for a, v in cos.items():
            cl = [ceiling(nulls, a, al, ep, "cos")[0] for ep in POST]
            vv = _norm(v, cl) if normalise else v
            ax.plot(x, vv, "o-", color=colors.get(a, "0.4"), lw=1.2, ms=4, alpha=0.85,
                    label=a)
            # THE CEILING ITSELF, dashed in the animal's own colour: the pre-to-pre value this
            # observation has to be read against. Drawn per animal because it is per animal -- it
            # depends on that animal's session count and trial totals.
            if not normalise and any(c is not None for c in cl):
                ax.plot(x, [np.nan if c is None else c for c in cl], ls="--", lw=1.0,
                        color=colors.get(a, "0.4"), alpha=0.75)
        ax.set_ylim(0, 1.25 if normalise else 1.05)
        ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
        ks = "/".join(str(got[(a, al)].get("cim_k")) for a in animals if (a, al) in got)
        ax.set_title(f"{al.upper()}   (K = {ks})", fontsize=9)
        if j == 0:
            ax.set_ylabel("A  overlap / pre-to-pre CEILING\n(1 = no resolvable rotation)"
                          if normalise else
                          "A  subspace overlap with PRE", fontsize=8)
            _legend(ax, normalise, nulls)

        ax = axes[1][j]
        ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
        for a, v in scale.items():
            cl = [ceiling(nulls, a, al, ep, "scale")[0] for ep in POST]
            ax.plot(x, _norm(v, cl) if normalise else v, "s-", color=colors.get(a, "0.4"),
                    lw=1.2, ms=4, alpha=0.85)
            if not normalise and any(c is not None for c in cl):
                ax.plot(x, [np.nan if c is None else c for c in cl], ls="--", lw=1.0,
                        color=colors.get(a, "0.4"), alpha=0.75)
        if j == 0:
            ax.set_ylabel("B  magnitude / pre-to-pre CEILING\n(1 = no resolvable growth)"
                          if normalise else "B  magnitude / pre", fontsize=8)

        ax = axes[2][j]
        for a in animals:
            c = cos.get(a)
            if c is None:
                continue
            sc = scale.get(a) or [None] * len(POST)
            obs_leak = [leak(ci, si) for ci, si in zip(c, sc)]
            if normalise:
                # C'S OWN CEILING: the unremoved shared amplitude two PRE-STROKE halves leave behind.
                # Rows A and B are divided by their ceilings, so leaving C raw under a heading saying
                # "normalised to the pre-to-pre ceiling" would make one panel mean something
                # different from the two above it.
                cl = [leak(ceiling(nulls, a, al, ep, "cos")[0],
                           ceiling(nulls, a, al, ep, "scale")[0]) for ep in POST]
                ax.plot(x, _norm(obs_leak, cl), "^-", color=colors.get(a, "0.4"), lw=1.2, ms=4,
                        alpha=0.85)
                continue
            ax.plot(x, obs_leak, "^-", color=colors.get(a, "0.4"), lw=1.2, ms=4, alpha=0.85)
            # DASHED IS THE CEILING HERE TOO. It used to be `sqrt(1 - overlap)` with the scale
            # assumed 1.0 -- the residual estimate this figure exists to replace -- which meant the
            # same dashes meant one thing in rows A and B and another here. That belongs in
            # DECISIONS.md; an axis can carry one referent.
            cl = [leak(ceiling(nulls, a, al, ep, "cos")[0],
                       ceiling(nulls, a, al, ep, "scale")[0]) for ep in POST]
            if any(np.isfinite(v) for v in cl):
                ax.plot(x, cl, ls="--", color=colors.get(a, "0.4"), lw=0.9, alpha=0.6)
        if normalise:
            ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
        if j == 0:
            ax.set_ylabel("C  unremoved amplitude / CEILING\n(1 = no worse than no change)"
                          if normalise else
                          "C  UNREMOVED shared amplitude\nsqrt(1-overlap) x scale", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(POST, fontsize=8)
    have = "with" if nulls else "WITHOUT"
    fig.suptitle(_wrap(
        ("The CONDITION-INDEPENDENT MODE after stroke, NORMALISED TO THE PRE-TO-PRE CEILING\n"
         "Every quantity divided by its trial-matched pre-stroke-vs-pre-stroke value, so 1.0 means "
         "INDISTINGUISHABLE FROM NO CHANGE and the dotted line is the null, not perfection.\n"
         "A ratio hides whether a small value came from a low numerator or a high denominator, so "
         "read this beside the unnormalised figure, never instead of it."
         if normalise else
         "The CONDITION-INDEPENDENT MODE after stroke - orientation, magnitude, and what `--orth` "
         "leaves behind\n"
         "fitted on PRE-STROKE sessions and applied to every epoch (rule 10: a per-epoch mode would "
         "subtract away the change being measured)\n"
         f"DASHED = the TRIAL-MATCHED PRE-TO-PRE CEILING ({have} it): two subspaces from different "
         "sessions do not fully overlap even when nothing changed, and a mean over fewer trials has "
         "a larger norm, so the ceiling is what these read at NO CHANGE and it is not 1.0")
        + "\nA alone says the mode is CONSERVED (far above chance); B says it is LARGER; only C "
          "bounds the artefact left in a post-stroke CD panel", fig),
        fontsize=8.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="the --out directory a cd_trajectories run wrote (its results/ is read)")
    ap.add_argument("--gate", default="lick", choices=tuple(cdt.GATES))
    ap.add_argument("--reference", default="contrast", choices=cdt.REFERENCES)
    ap.add_argument("--align", nargs="+", default=list(ALIGNS), choices=ALIGNS)
    ap.add_argument("--occluded", default="drop", choices=("drop", "keep"),
                    help="which arm's dumps to read -- must MATCH the render, since the two are "
                         "stored under different tags. Default `drop`, as the render's is.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    d = Path(args.dir)
    got, problems = collect(d, aligns=tuple(args.align), gate=args.gate,
                            reference=args.reference,
                            mask_occluded=args.occluded == "drop")
    nulls = load_nulls(d, aligns=tuple(args.align), gate=args.gate)
    print(table(got, nulls))
    for p in problems:
        print(f"!! {p}")
    if not got:
        return 1
    # THE ARM IS IN THE FILENAME, for the same reason it is in the dump tag: two arms, two figures.
    suffix = "_cortexonly" if args.occluded == "drop" else ""
    out = Path(args.out) if args.out else d / f"cd_cim_geometry_{args.gate}{suffix}.png"
    print(f"-> {figure(got, out, nulls)}")
    # THE NORMALISED VIEW IS A SECOND FIGURE, not a replacement -- see its own subtitle. Only drawn
    # when there is a ceiling to normalise BY; otherwise it would be a figure of ones and NaNs.
    if nulls:
        out2 = d / f"cd_cim_geometry_{args.gate}{suffix}_vs_ceiling.png"
        print(f"-> {figure(got, out2, nulls, normalise=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
