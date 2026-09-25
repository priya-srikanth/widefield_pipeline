"""WHERE DID A POSITION'S TRAJECTORY MOVE TO? The 6x6 similarity matrix and its post-stroke change.

Priya, 2026-09-25: *"is there a clean way to look at where a lick's trajectory MOVED to another
trajectory (eg did post-stroke far R end up looking more like far center than far center does)?"*

THE OBJECT. ``M[i, j]`` = position i's trials projected onto position j's coding direction, averaged
over the response window. The POLE NORMALISATION is what makes this readable: every CD is scaled so 0
is that animal's pre-stroke not-P and 1 is its pre-stroke lick at P, so an entry of 1.0 means "looks
exactly like pre-stroke position j" **in the same units in every column**. A row is therefore a
profile of what those trials resemble, and the question above becomes a comparison of two numbers:
``M_post[far_R, far_center]`` against ``M_post[far_center, far_center]``.

It costs nothing to compute. `cd_trajectories` already stores every (epoch, cd_position,
trial_position) trace, because the courses exist for all six directions over the whole session and
slicing any position's trials out of any course is free. This reads those dumps.

TWO THINGS THAT WOULD MAKE IT LIE, AND WHAT IS DONE ABOUT THEM.

  * **THE SIX DIRECTIONS ARE NOT ORTHOGONAL TO EACH OTHER, AND `--orth on` DOES NOT MAKE THEM SO.**
    That flag removes the CONDITION-INDEPENDENT MODE from each direction; it says nothing about the
    directions' relationship to one another. MEASURED after the CIM removal, pre-cue:

        animal   mean |cos| off-diagonal   max    ||sum of the six unit vectors||
        PS92          0.427               0.778            0.54
        PS95          0.332               0.618            0.72

    and for PS92 the CIM removal barely moved it at all (0.421 -> 0.427). Pairs run as high as
    close_R/far_L at -0.78 and close_center/far_center at -0.73, and 6.0 is what the sum would be if
    the six were aligned -- so the near-cancellation that motivates the whole orthogonalisation step
    is still fully present in the INTER-DIRECTION geometry.

    The structure is anatomically sensible, which is precisely why it is a trap: close positions
    correlate positively with each other, far positions likewise, and close-vs-far pairs run
    negative. That is the ring, so off-diagonal mass in M is partly "these two were always similar".
    **Hence the readout is ΔM = M_post − M_pre, within animal** -- pre-stroke M carries the geometry
    and subtracting each animal's own removes it.

    Mutually orthogonalising the six WOULD make the columns independent and the migration claim
    cleaner, and it is not free: Gram-Schmidt is order-dependent (whoever goes first keeps their
    whole direction), so it would need a symmetric whitening, and each axis would then mean "what is
    UNIQUE to P once the others are removed" rather than "the position-P coding direction". That is a
    different and arguably sharper question -- did far_R acquire far_center's SPECIFIC signature,
    rather than move into territory they share -- and it is not what this module currently asks.
  * **A ROW CAN MOVE BECAUSE EVERYTHING MOVED.** The condition-independent mode is projected out
    before any of this, and it grows post-stroke; what survives that is still shared to the extent
    the projection is imperfect. So a row that rises uniformly is reported as such (`row mean`)
    rather than being read as six separate migrations.

WHAT "USURPED" MEANS in the table: position i's trials score HIGHER on j's direction than j's own
trials do. That is the strong form of Priya's question and it is much more than "i drifted toward j" --
it says the identity of the trajectory at j is better matched by the wrong position's trials.

    python -m scripts.cd_migration --align precue
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wfield_local import cd_trajectories as cdt  # noqa: E402
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES  # noqa: E402

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPOCHS = ("pre", "acute", "subacute", "chronic")
POST = ("acute", "subacute", "chronic")

#: Window the projection is averaged over, in seconds from the alignment event. The response window
#: the directions were fitted on, so a diagonal entry is the quantity the poles define and should read
#: ~1.0 pre-stroke -- which is the anchor check, and it is printed so a broken matrix is visible.
WINDOW = (0.0, 2.0)

#: M is a signed LEVEL, ΔM is a signed CHANGE, and the project keeps the two maps distinct so that
#: "more" and "different" never look alike (`transfer_matrix.CMAP_CHANGE` / `CMAP_LEVEL`). The change
#: gets the canonical RdBu_r; the level gets the other diverging map. These were swapped.
CMAP_M = "PuOr_r"
CMAP_DELTA = "RdBu_r"


def matrices(res, window=WINDOW):
    """``{epoch: (M, present)}`` -- M[i, j], rows = trial position, cols = the CD projected onto."""
    t = np.arange(-res["pre_n"], res["post_n"]) / float(res["fs"])
    m = (t >= window[0]) & (t < window[1])
    pos = [p for p in DISPLAY_ORDER if p in res["positions"]]
    out = {}
    for ep in EPOCHS:
        M = np.full((len(pos), len(pos)), np.nan)
        for i, tr_p in enumerate(pos):
            for j, cd_p in enumerate(pos):
                d = res["traces"].get((ep, cd_p, tr_p))
                if d is not None:
                    M[i, j] = float(np.nanmean(np.asarray(d["mean"])[m]))
        if np.isfinite(M).any():
            out[ep] = (M, pos)
    return out


def pooled_matrices(got, window=WINDOW):
    """``{epoch: (M_mean, delta_mean, n, pos)}`` across animals, deltas taken WITHIN animal first."""
    per, deltas, pos_ref = {}, {}, None
    for _a, res in got.items():
        mats = matrices(res, window)
        if "pre" not in mats:
            continue
        pre, pos = mats["pre"]
        pos_ref = pos_ref or pos
        if pos != pos_ref:
            continue
        for ep, (M, _p) in mats.items():
            per.setdefault(ep, []).append(M)
            if ep != "pre":
                deltas.setdefault(ep, []).append(M - pre)
    out = {}
    for ep, mats_ in per.items():
        A = np.stack(mats_)
        D = np.stack(deltas[ep]) if ep in deltas else None
        out[ep] = (np.nanmean(A, 0), (np.nanmean(D, 0) if D is not None else None),
                   A.shape[0], pos_ref)
    return out


#: An incumbent scoring below this fraction of its own PRE-STROKE diagonal has VACATED its direction
#: rather than been out-competed on it. 0.5 is blunt and the two cases are reported separately rather
#: than thresholded into one number, so the cut only decides which list an entry appears in.
VACATED_FRAC = 0.5


def usurpation(M, M_pre, pos):
    """``(usurped, vacated)`` -- position i outscoring j on j's OWN direction, split by cause.

    THE STRONG FORM of Priya's question is "the trajectory at j is better matched by i's trials than
    by j's own", which is a claim about identity rather than distance. But `M[i, j] > M[j, j]` is
    satisfied two ways, and only one of them is that claim:

      USURPED  j still holds its direction (its diagonal is intact) and i outscores it anyway.
      VACATED  j's own diagonal has COLLAPSED, so almost anything outscores it. Measured on the first
               real run: five of six acute entries were "X beats far_R on far_R's direction", where
               far_R's own score had fallen from +1.23 to +0.07. That is far_R losing its identity,
               not five positions moving into it -- and ranked by margin those vacancies sat at the
               top of the list looking like the result.
    """
    usurped, vacated = [], []
    n = len(pos)
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(M[i, j]) or not np.isfinite(M[j, j]):
                continue
            if M[i, j] <= M[j, j]:
                continue
            own_pre = float(M_pre[j, j]) if M_pre is not None else float("nan")
            row = (i, j, float(M[i, j]), float(M[j, j]), own_pre)
            collapsed = np.isfinite(own_pre) and own_pre > 0 and M[j, j] < VACATED_FRAC * own_pre
            (vacated if collapsed else usurped).append(row)
    key = lambda r: r[2] - r[3]                                        # noqa: E731
    return sorted(usurped, key=key, reverse=True), sorted(vacated, key=key, reverse=True)


def per_animal_support(got, i_code, j_code, ep, window=WINDOW):
    """``(n_usurped, n_animals, rows)`` -- does each animal INDIVIDUALLY show i beating j on j's CD?

    RULE 8 MADE STRUCTURAL. A pooled matrix can satisfy `M[i, j] > M[j, j]` while no single animal
    does, because a mean over four can cross a threshold none of them crosses -- and "does i beat j"
    is a threshold statement. Measured on the first real run: the pooled acute `far_R -> far_center`
    hit was ONE animal, PS95, whose acute epoch is a single session; and `close_L -> close_center`
    was two, one of them with a margin of +0.05 and one that was actually the vacated case.

    So no claim is printed without this beside it.
    """
    rows, n_yes = [], 0
    for a in sorted(got):
        mats = matrices(got[a], window)
        if ep not in mats or "pre" not in mats:
            continue
        M, pos = mats[ep]
        if i_code not in pos or j_code not in pos:
            continue
        ii, jj = pos.index(i_code), pos.index(j_code)
        val, own, own_pre = float(M[ii, jj]), float(M[jj, jj]), float(mats["pre"][0][jj, jj])
        intact = own >= VACATED_FRAC * own_pre if own_pre > 0 else False
        yes = val > own and intact
        n_yes += bool(yes)
        rows.append((a, val, own, own_pre,
                     "USURPED" if yes else ("vacated" if val > own else "no")))
    return n_yes, len(rows), rows


#: Animals that must show a claim INDIVIDUALLY before it is presented as a cohort result (rule 8).
MIN_ANIMALS_SUPPORT = 3


def report(got, window=WINDOW):
    pooled = pooled_matrices(got, window)
    L = []
    pre = pooled.get("pre")
    if pre is not None:
        M, _d, n, pos = pre
        L.append(f"PRE-STROKE M (n={n} animals). Diagonal is the anchor and should read ~1.0; "
                 f"off-diagonal is the resemblance that ALREADY EXISTS between positions.")
        L.append("  rows = trials of, cols = projected onto")
        L.append("  " + " " * 14 + " ".join(f"{POSITION_NAMES.get(p, p)[:8]:>9s}" for p in pos))
        for i, p in enumerate(pos):
            L.append(f"  {POSITION_NAMES.get(p, p):13s} "
                     + " ".join(f"{M[i, j]:9.2f}" for j in range(len(pos))))
    for ep in POST:
        if ep not in pooled:
            continue
        M, D, n, pos = pooled[ep]
        L.append("")
        # ASCII: the Windows console is cp1252 and an em-dash does not survive it.
        L.append(f"{ep.upper()} -- CHANGE from pre, within animal then averaged (n={n})")
        L.append("  " + " " * 14 + " ".join(f"{POSITION_NAMES.get(p, p)[:8]:>9s}" for p in pos)
                 + "   row mean")
        for i, p in enumerate(pos):
            L.append(f"  {POSITION_NAMES.get(p, p):13s} "
                     + " ".join(f"{D[i, j]:+9.2f}" for j in range(len(pos)))
                     + f"  {np.nanmean(D[i]):+9.2f}")
        # WHERE EACH ROW NOW POINTS: the column it most resembles, and whether that is its own.
        L.append("  nearest identity per row (argmax over columns of M, not of the change):")
        for i, p in enumerate(pos):
            j = int(np.nanargmax(M[i]))
            flag = "" if j == i else f"   <-- moved toward {POSITION_NAMES.get(pos[j], pos[j])}"
            L.append(f"    {POSITION_NAMES.get(p, p):13s} best = "
                     f"{POSITION_NAMES.get(pos[j], pos[j]):13s} ({M[i, j]:+.2f}), "
                     f"own = {M[i, i]:+.2f}{flag}")
        M_pre = pooled["pre"][0] if "pre" in pooled else None
        us, vac = usurpation(M, M_pre, pos)
        if us:
            L.append("  USURPED -- the incumbent STILL HOLDS its direction and is outscored on it "
                     "anyway. This is the strong claim, and it is NOT a cohort result unless the "
                     f"per-animal count below reaches {MIN_ANIMALS_SUPPORT}/4 (rule 8):")
            for i, j, a, b, pre_own in us[:6]:
                n_yes, n_tot, rows = per_animal_support(got, pos[i], pos[j], ep, window)
                verdict = ("SUPPORTED" if n_yes >= MIN_ANIMALS_SUPPORT
                           else f"NOT SUPPORTED -- {n_yes} of {n_tot} animals")
                L.append(f"    {POSITION_NAMES.get(pos[i], pos[i]):13s} on "
                         f"{POSITION_NAMES.get(pos[j], pos[j]):13s} CD: {a:+.2f} vs its own "
                         f"{b:+.2f} (was {pre_own:+.2f})  margin {a - b:+.2f}"
                         f"   [{n_yes}/{n_tot} animals: {verdict}]")
                for an, val, own, _op, v in rows:
                    L.append(f"        {an:6s} {val:+6.2f} vs own {own:+6.2f}   {v}")
        else:
            L.append("  USURPED: none, once vacated directions are separated out.")
        if vac:
            L.append(f"  VACATED -- the incumbent's own score fell below {VACATED_FRAC:.0%} of its "
                     "pre-stroke value, so being outscored on it says little about the usurper:")
            for i, j, a, b, pre_own in vac[:4]:
                L.append(f"    {POSITION_NAMES.get(pos[j], pos[j]):13s} CD now holds only "
                         f"{b:+.2f} of its own trials (was {pre_own:+.2f}); e.g. "
                         f"{POSITION_NAMES.get(pos[i], pos[i])} scores {a:+.2f} on it")
    return "\n".join(L)


def figure(got, out, align, window=WINDOW):
    pooled = pooled_matrices(got, window)
    eps = [e for e in EPOCHS if e in pooled]
    if not eps:
        raise SystemExit("nothing to draw")
    pos = pooled[eps[0]][3]
    names = [POSITION_NAMES.get(p, str(p)) for p in pos]
    fig, axes = plt.subplots(2, len(eps), figsize=(3.0 * len(eps) + 1.6, 7.2), squeeze=False,
                             constrained_layout=True)
    Ms = [pooled[e][0] for e in eps]
    vmax = float(np.nanmax(np.abs(np.stack(Ms))))
    Ds = [pooled[e][1] for e in eps if pooled[e][1] is not None]
    dmax = float(np.nanmax(np.abs(np.stack(Ds)))) if Ds else 1.0
    for k, ep in enumerate(eps):
        M, D, n, pos = pooled[ep]
        ax = axes[0][k]
        # LEVEL vs CHANGE GET DIFFERENT MAPS, and they were the wrong way round: the project
        # reserves `transfer_matrix.CMAP_CHANGE` (RdBu_r) for signed CHANGE so that "more" and
        # "different" never look alike, and M is a level.
        im = ax.imshow(M, cmap=CMAP_M, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{ep}  (n={n})", fontsize=9)
        if k == 0:
            ax.set_ylabel("M — trials OF (row)\nprojected ONTO (col)", fontsize=8)
        _ticks(ax, names, k == 0)
        _annot(ax, M)
        if ep != "pre":
            _mark(ax, M, pooled["pre"][0] if "pre" in pooled else None, pos,
                  support=(lambda ic, jc, _e=ep: per_animal_support(got, ic, jc, _e, window)[0]))
        if k == len(eps) - 1:
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax2 = axes[1][k]
        if D is None:
            ax2.set_axis_off()
            continue
        im2 = ax2.imshow(D, cmap=CMAP_DELTA, vmin=-dmax, vmax=dmax)
        if k == 0:
            ax2.set_ylabel("ΔM — change from pre\n(within animal, then pooled)", fontsize=8)
        _ticks(ax2, names, k == 0)
        _annot(ax2, D, sign=True)
        if k == len(eps) - 1:
            fig.colorbar(im2, ax=ax2, fraction=0.046)
    from matplotlib.patches import Patch

    axes[0][0].legend(
        handles=[Patch(facecolor="none", edgecolor="#111111", lw=2.0,
                       label="USURPED: beats that column's own diagonal,\nwhich is still intact"),
                 Patch(facecolor="none", edgecolor="#111111", lw=2.0, linestyle=(0, (2, 1.4)),
                       label="dashed box: POOLED ONLY — fewer than\n"
                             f"{MIN_ANIMALS_SUPPORT}/4 animals show it individually (rule 8)"),
                 Patch(facecolor="none", edgecolor="0.25", hatch="////",
                       label="VACATED: this diagonal lost >50% of its\npre-stroke value")],
        fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.32))
    fig.suptitle(
        f"WHERE A POSITION'S TRAJECTORY MOVED — {align.upper()}, pooled over {pooled[eps[0]][2]} "
        f"animals\nM[i, j] = position i's trials projected onto position j's coding direction, "
        f"averaged over [{window[0]:g}, {window[1]:g}] s. Pole-normalised, so 1.0 means 'looks like "
        f"PRE-STROKE position j' and COLUMNS ARE COMPARABLE.\nThe diagonal is the anchor (~1.0 "
        f"pre-stroke). THE DIRECTIONS ARE NOT ORTHOGONAL — neighbouring positions resemble each "
        f"other already — so read the BOTTOM row, the change from pre, for migration.\n"
        f"BOLD = THE DIAGONAL, a position's own trials on its own direction — the anchor, ~1.0 "
        f"pre-stroke, and its fall is that position losing its own identity.\n"
        f"A BOXED cell is the strong claim: those trials score higher on the column's direction than "
        f"that column's OWN trials do — a comparison between two different rows, which is why it is "
        f"drawn rather than left to the eye. Top row is the LEVEL, bottom is the CHANGE, and they "
        f"use different colour maps on purpose.",
        fontsize=8)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _ticks(ax, names, ylab):
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names if ylab else [""] * len(names), fontsize=6.5)


def _mark(ax, M, M_pre, pos, support=None):
    """Box the USURPED cells and hatch the VACATED diagonals -- the two states, drawn.

    The figure used to leave the reader to compare an off-diagonal cell against a diagonal one in a
    DIFFERENT ROW, which is the comparison the whole analysis turns on and the one a heatmap is
    worst at. Definitions come from `usurpation`, so this cannot drift from the printed report.
    """
    from matplotlib.patches import Rectangle

    usurped, vacated = usurpation(M, M_pre, pos)
    for i, j, _a, _b, _pre in usurped:
        # THE BOX IS SOLID ONLY WHEN THE ANIMALS INDIVIDUALLY AGREE. A dashed box is a pooled-only
        # hit, which a mean over four can produce with no animal crossing the threshold -- exactly
        # what happened to `far_R -> far_center` (1 of 4, and that one a single-session epoch).
        n_yes = None
        if support is not None:
            n_yes = support(pos[i], pos[j])
        solid = n_yes is None or n_yes >= MIN_ANIMALS_SUPPORT
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.0,
                               linestyle="-" if solid else (0, (2, 1.4)),
                               edgecolor="#111111", zorder=5))
        if n_yes is not None:
            ax.text(j + 0.44, i - 0.40, f"{n_yes}/4", ha="right", va="top", fontsize=5.2,
                    color="#111111", zorder=6)
    for _i, j, _a, _b, _pre in vacated:
        ax.add_patch(Rectangle((j - 0.5, j - 0.5), 1, 1, fill=False, lw=1.4, hatch="////",
                               edgecolor="0.25", zorder=4))
    return usurped, vacated


def _annot(ax, M, sign=False):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:+.2f}" if sign else f"{M[i, j]:.2f}",
                    ha="center", va="center", fontsize=5.8,
                    color="white" if abs(M[i, j]) > 0.6 * np.nanmax(np.abs(M)) else "black",
                    fontweight="bold" if i == j else "normal")


def main(argv=None) -> int:
    from scripts.cd_cross_animal_figure import collect

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--align", nargs="+", default=["precue", "cue", "lick"],
                    choices=("precue", "cue", "lick"))
    ap.add_argument("--gate", default="lick", choices=tuple(cdt.GATES))
    ap.add_argument("--reference", default="contrast", choices=cdt.REFERENCES)
    ap.add_argument("--occluded", default="drop", choices=("drop", "keep"))
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
        print(f"\n===== {align.upper()} =====")
        print(report(got))
        tag = "_".join([align, args.reference, args.gate]
                       + (["cortexonly"] if args.occluded == "drop" else []))
        print(f"-> {figure(got, d / f'cd_migration_{tag}.png', align)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
