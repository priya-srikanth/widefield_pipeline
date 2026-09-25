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
            method="dom"):
    """`{(animal, align): res}` for every dump that exists, plus the reasons the others do not.

    A STALE DUMP RAISES inside `load_result` and is recorded as a problem rather than drawn -- see
    `cd_trajectories.RESULT_GUARD`. Missing and stale are different facts and both are reported,
    because "this animal has no CIM" and "this animal's CIM was measured under other constants"
    lead to opposite next actions.
    """
    got, problems = {}, []
    for a in animals:
        for al in aligns:
            try:
                res = cdt.load_result(dirpath, a, al, method, reference, gate, orth=True)
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


def table(got):
    """The printed report. THE TABLE IS THE RESULT; the figure is how it is read at a glance."""
    lines = [f"{'animal':7s} {'align':7s} {'K':>2s} {'chance':>7s} {'epoch':10s} "
             f"{'overlap':>8s} {'scale':>7s} {'leak':>7s}"]
    for (a, al), res in sorted(got.items()):
        cos, scale = res.get("cim_cos") or {}, res.get("cim_scale") or {}
        for ep in POST:
            c, sc = cos.get(ep), scale.get(ep)
            if c is None:
                continue
            sc_s = "    n/a" if sc is None else f"{sc:7.3f}"
            lines.append(f"{a:7s} {al:7s} {res.get('cim_k') or 0:2d} "
                         f"{res.get('cim_chance', float('nan')):7.3f} {ep:10s} "
                         f"{c:8.3f} {sc_s} {leak(c, sc):7.3f}")
    return "\n".join(lines)


def _series(got, align, key, animals):
    out = {}
    for a in animals:
        res = got.get((a, align))
        if res is None:
            continue
        d = res.get(key) or {}
        out[a] = [d.get(ep) for ep in POST]
    return out


def figure(got, out):
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
        # THE CHANCE BAND, not a caption. Two unrelated K-dim subspaces of an n-dim space overlap at
        # K/n, so the distance from the BAND is the finding and the distance from 1.0 is not. This
        # figure exists partly because that was briefly read the other way round.
        ax.axhspan(0, ch, color="0.85", lw=0, zorder=0)
        ax.text(len(POST) - 0.55, ch, f" chance K/n = {ch:.3f}", fontsize=6.5,
                va="bottom", ha="right", color="0.35")
        for a, v in cos.items():
            ax.plot(x, v, "o-", color=colors.get(a, "0.4"), lw=1.6, ms=5, label=a)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
        ks = "/".join(str(got[(a, al)].get("cim_k")) for a in animals if (a, al) in got)
        ax.set_title(f"{al.upper()}   (K = {ks})", fontsize=9)
        if j == 0:
            ax.set_ylabel("A  subspace overlap with PRE\n(1 = the same subspace)", fontsize=8)
            ax.legend(fontsize=7, frameon=False, ncol=2)

        ax = axes[1][j]
        ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
        for a, v in scale.items():
            ax.plot(x, v, "s-", color=colors.get(a, "0.4"), lw=1.6, ms=5)
        if j == 0:
            ax.set_ylabel("B  magnitude / pre\n(the cosine CANNOT see this)", fontsize=8)

        ax = axes[2][j]
        for a in animals:
            c = cos.get(a)
            if c is None:
                continue
            sc = scale.get(a) or [None] * len(POST)
            ax.plot(x, [leak(ci, si) for ci, si in zip(c, sc)], "^-",
                    color=colors.get(a, "0.4"), lw=1.6, ms=5)
            # What the handoff's residual table assumed: sqrt(1 - overlap) with the scale at 1.0.
            # Drawn so the size of that assumption is visible rather than asserted in prose.
            ax.plot(x, [leak(ci, None) for ci in c], ls="--", color=colors.get(a, "0.4"),
                    lw=0.9, alpha=0.55)
        if j == 0:
            ax.set_ylabel("C  UNREMOVED shared amplitude\nsqrt(1-overlap) x scale", fontsize=8)
            ax.text(0.02, 0.04, "dashed = the scale assumed 1.0", transform=ax.transAxes,
                    fontsize=6.5, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels(POST, fontsize=8)
    fig.suptitle(
        "The CONDITION-INDEPENDENT MODE after stroke - orientation, magnitude, and what `--orth` "
        "leaves behind\n"
        "fitted on PRE-STROKE sessions and applied to every epoch (rule 10: a per-epoch mode would "
        "subtract away the change being measured)\n"
        "A alone says the mode is CONSERVED (far above chance); B says it is LARGER; only C bounds "
        "the artefact left in a post-stroke CD panel", fontsize=8.5)
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    got, problems = collect(Path(args.dir), aligns=tuple(args.align), gate=args.gate,
                            reference=args.reference)
    print(table(got))
    for p in problems:
        print(f"!! {p}")
    if not got:
        return 1
    out = Path(args.out) if args.out else Path(args.dir) / f"cd_cim_geometry_{args.gate}.png"
    print(f"-> {figure(got, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
