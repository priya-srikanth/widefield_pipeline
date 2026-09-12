"""Does recovery RETRACE the pre-stroke code, or migrate away and come back? A per-session view.

The epoch family (`epoch_acc_by_position`, `epoch_5rgapdelta_*`) answers what the CHRONIC state is:
a decoder frozen on pre-stroke activity returns to baseline at five of six positions, and the
advantage a within-session refit held acutely is gone by chronic under both training-set conventions.
That is an ENDPOINT comparison across three coarse bins, and the hypothesis it is used to support --
*"recovery proceeds by re-establishing pre-stroke activity patterns rather than by building new
ones"* -- is a claim about the ROUTE. Three points cannot describe a route.

This module re-bins the SAME records by session instead of by epoch. It adds no new analysis and
cannot disagree with the epoch figures: `grant_figures._collect_5c(mode="paired")` already returns
``{animal: (pre_record, {day: record})}`` keyed by days since that animal's own lesion, and the
epoch figures merely group those days with `epoch_figures.epoch_of_day`. Here they are left apart.

**THE TWO ACCOUNTS MAKE DIFFERENT PREDICTIONS ABOUT THE SHAPE, not just the endpoint.** Write
``F`` for the frozen decoder's accuracy deficit (how far the pre-stroke readout is from baseline) and
``G`` for the refit-minus-frozen gap above its pre-stroke value (how much information the session
carries that the pre-stroke readout cannot reach -- a reorganisation index):

    RETRACE      the code weakens and re-strengthens along its original axis. G stays near zero
                 throughout, or peaks early and decays monotonically to zero. F and G recover
                 together, and the (F, G) trajectory is a LINE traversed out and back.

    MIGRATE      the code moves to a new arrangement and the animal recovers by using it. G RISES
                 as F recovers -- information returns to a place the old readout cannot see -- and
                 the trajectory ENDS at the upper left rather than at the origin.

So the discriminating measurements are (a) whether G's maximum precedes F's recovery, (b) the sign
of G's trend AFTER its acute level, and (c) whether any late session exceeds that acute level.

**THE READING IS THE ENDPOINT, NOT THE SHAPE, and an earlier version of this docstring said
otherwise.** It described migrate as a LOOP with hysteresis and drew a panel titled that way.
Separating an outward path from a return path requires them to differ by more than the noise, and
with ~12 sessions per animal scattering by +/-0.05 in G there is no such separation to find here. The
claim the data CAN carry is where the trajectory finishes: at the origin, or above it.

**WHAT THIS CANNOT DO, stated here because the figure invites it.** Four animals, ~46 post-stroke
sessions. A cohort-level p-value on a slope over n=4 animals is not worth printing, and this module
does not print one: it reports each animal's trajectory and its own slope, and treats CONSISTENCY
ACROSS ANIMALS as the evidence -- the same standard the rest of the deck applies. A per-session
accuracy is also noisy (a few hundred trials), which is why the summary leans on the sign of a trend
rather than on any single session.

**IT WRITES ITS NUMBERS.** The epoch family emits PNG and SVG and nothing else, which is how
`PRELIM_DATA_VLS_STROKE.md` came to disagree with its own figures for two days without anything able
to flag it. Every value plotted here is also written to ``recovery_trajectory_<align>_<variant>.csv``
beside the figure.

CLI::

    python -m wfield_local.recovery_trajectory                  # post-cue, working
    python -m wfield_local.recovery_trajectory --align precue
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from wfield_local.writeguard import assert_writable

#: Sessions an animal must contribute AFTER its peak before its post-peak slope is reported. Two
#: points define a slope and say nothing; below this the animal is drawn and left out of the summary.
MIN_POST_PEAK = 3


def _pre_pair(pre_entry):
    """``(frozen, refit)`` pooled over an animal's pre-stroke sessions, from a PAIRED pre entry.

    In ``paired`` mode the pre entry is a LIST -- one record per pre-stroke session, because the
    refit arm's unit is a session -- not the single concatenated leave-one-session-out record the
    ``frozen`` mode returns. Pooling here rather than averaging per-session accuracies keeps the
    baseline on the same footing as a post-stroke session's accuracy, which is also a pooled count.
    """
    from wfield_local.grant_figures import REFIT_UNAVAILABLE

    recs = pre_entry if isinstance(pre_entry, list) else [pre_entry]
    ys, ps = [], []
    for r in recs:
        if r is None:
            continue
        y, p, _b = r
        p = np.asarray(p)
        if p.ndim != 2:
            continue
        ys.append(np.asarray(y))
        ps.append(p)
    if not ys:
        return None, None
    y, p = np.concatenate(ys), np.concatenate(ps)
    m = p[:, 1] != REFIT_UNAVAILABLE            # drop classes the refit arm was never asked about
    if m.sum() < 20:
        return None, None
    return float((p[m, 0] == y[m]).mean()), float((p[m, 1] == y[m]).mean())


def _session_pair(record):
    """``(frozen, refit, n_trials)`` for one session's paired record, on identical trials."""
    from wfield_local.grant_figures import REFIT_UNAVAILABLE

    if record is None:
        return None
    y, p, _b = record
    y, p = np.asarray(y), np.asarray(p)
    if p.ndim != 2:
        return None
    m = p[:, 1] != REFIT_UNAVAILABLE
    if m.sum() < 20:
        return None
    return float((p[m, 0] == y[m]).mean()), float((p[m, 1] == y[m]).mean()), int(m.sum())


def series(align: str = "cue", variant: str = "working", matched: bool = False) -> list[dict]:
    """One row per (animal, POST-stroke session): frozen, refit, gap, and both pre-subtracted.

    **Pre-stroke sessions are deliberately absent from the rows.** Both reported axes are differences
    FROM the pre-stroke baseline, so pre-stroke is the origin rather than a plotted point; including
    it would draw a cluster at (0, 0) that is the reference scattered by its own noise. The
    per-animal figure marks the origin so this is visible rather than assumed.

    ``F`` and ``G`` are the two axes the accounts differ on -- see the module docstring. Both are
    signed so that ZERO IS THE PRE-STROKE STATE and the lesion moves them positive:

        F = pre_frozen - frozen      the frozen readout's DEFICIT  (recovery drives it to 0)
        G = gap - pre_gap            reorganisation ABOVE baseline (retrace keeps it at 0)
    """
    from wfield_local import epoch_figures as ef
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant,
                                      "paired_matched" if matched else "paired")
    rows = []
    for an, got in sorted(per_animal.items()):
        if not got:
            continue
        pre_f, pre_r = _pre_pair(got[0])
        if pre_f is None:
            continue
        for day in sorted(got[1]):
            if int(day) < 0:
                continue
            sp = _session_pair(got[1][day])
            if sp is None:
                continue
            fz, rf, n = sp
            rows.append({
                "animal": an, "day": int(day),
                "matched": int(bool(matched)),
                "epoch": ef.epoch_of_day(an, int(day)) or "",
                "n_trials": n,
                "frozen": fz, "refit": rf, "gap": rf - fz,
                "pre_frozen": pre_f, "pre_refit": pre_r, "pre_gap": pre_r - pre_f,
                "F_deficit": pre_f - fz,
                "G_reorg": (rf - fz) - (pre_r - pre_f),
                # F - G, algebraically. Carried explicitly because it is the interpretable half:
                # what NO decoder recovers. See `figure_by_animal`'s unity line.
                "refit_deficit": pre_r - rf,
            })
    return rows


def pre_points(align: str = "cue", variant: str = "working", matched: bool = False) -> dict:
    """``{animal: [(F, G), ...]}`` for that animal's PRE-STROKE sessions, on the same two axes.

    **The origin is a point estimate and pre-stroke sessions scatter around it.** Without that
    scatter drawn, "has the trajectory returned to the origin?" has no scale: a star 0.05 from the
    origin is either a full return or a residual deficit depending on how far pre-stroke sessions sit
    from their own mean. This supplies the yardstick.

    The baseline these are measured against is the POOLED pre-stroke value, i.e. the mean of the very
    sessions plotted, so the cloud is centred near the origin by construction. That is the point --
    it is a dispersion, not an effect, and it is why the mean dot is drawn with SEM rather than with
    a confidence interval on a difference.
    """
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant,
                                      "paired_matched" if matched else "paired")
    out = {}
    for an, got in sorted(per_animal.items()):
        if not got:
            continue
        pre_f, pre_r = _pre_pair(got[0])
        if pre_f is None:
            continue
        entry = got[0] if isinstance(got[0], list) else [got[0]]
        pts = []
        for rec in entry:
            sp = _session_pair(rec)
            if sp is None:
                continue
            fz, rf, _n = sp
            pts.append((pre_f - fz, (rf - fz) - (pre_r - pre_f)))
        if pts:
            out[an] = pts
    return out


def summarise(rows: list[dict]) -> dict:
    """Per animal: the ACUTE level of reorganisation, and what happened to it afterwards.

    **THE REFERENCE IS THE ACUTE PEAK, NOT THE GLOBAL ONE, and getting that wrong inverts the
    result.** A first version took ``argmax`` over all post-stroke sessions. For PS93 the maximum
    sits on day 25 -- the LAST session -- so it was labelled "the peak", nothing could come after
    it, and the late-resurgence flag reported False. That is precisely the MIGRATE signature being
    hidden by the statistic meant to detect it: reorganisation still climbing at the end of
    recovery. The acute window is the lesion's immediate effect and is the only defensible
    reference for "did this grow afterwards".

    Reported per animal:

      ``acute_G``            reorganisation during the acute window (max over acute sessions)
      ``post_slope``         slope of G per day over the POST-acute sessions; negative = unwinding
      ``post_corr_F_G``      do the frozen deficit and reorganisation fall together? retrace says yes
      ``late_exceeds_acute`` any post-acute session above the acute level -- the migrate signature
    """
    out, animals = {}, sorted({r["animal"] for r in rows})
    for an in animals:
        rs = sorted((r for r in rows if r["animal"] == an), key=lambda r: r["day"])
        if len(rs) < 2:
            continue
        acute = [r for r in rs if r["epoch"] == "acute"]
        post = [r for r in rs if r["epoch"] in ("subacute", "chronic")]
        if not acute:
            # No labelled acute session: fall back to the earliest third, and SAY SO in the record
            # rather than silently using a different reference from the other animals.
            k = max(1, len(rs) // 3)
            acute, post, fallback = rs[:k], rs[k:], True
        else:
            fallback = False
        g_ac = max(r["G_reorg"] for r in acute)
        d = np.array([r["day"] for r in post], float)
        g = np.array([r["G_reorg"] for r in post])
        f = np.array([r["F_deficit"] for r in post])
        ok = len(post) >= MIN_POST_PEAK
        slope = float(np.polyfit(d, g, 1)[0]) if ok else float("nan")
        rho = (float(np.corrcoef(f, g)[0, 1])
               if ok and np.std(g) > 0 and np.std(f) > 0 else float("nan"))
        out[an] = {
            "n_sessions": len(rs), "n_acute": len(acute), "n_post": len(post),
            "acute_reference_is_fallback": fallback,
            "acute_G": float(g_ac),
            "acute_day": int([r["day"] for r in acute][
                int(np.argmax([r["G_reorg"] for r in acute]))]),
            "post_slope": slope, "post_corr_F_G": rho,
            "late_exceeds_acute": bool(len(post) and g.max() > g_ac + 1e-12),
            "max_post_G": float(g.max()) if len(post) else float("nan"),
            "final_day": int(rs[-1]["day"]), "final_G": float(rs[-1]["G_reorg"]),
            "final_F": float(rs[-1]["F_deficit"]),
        }
    scored = [v for v in out.values() if np.isfinite(v["post_slope"])]
    out["_cohort"] = {
        "animals": len(out),
        "animals_scored": len(scored),
        "negative_post_slope": sum(v["post_slope"] < 0 for v in scored),
        "late_exceeds_acute": sum(v["late_exceeds_acute"] for v in out.values()
                                  if isinstance(v, dict) and "late_exceeds_acute" in v),
    }
    return out


def write_pre_csv(pre, dest: Path) -> Path:
    """The pre-stroke cloud, so `--from-csv` can redraw it without rebuilding the bundles."""
    assert_writable(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["animal", "F_deficit", "G_reorg"])
        for an, pts in sorted((pre or {}).items()):
            for f, g in pts:
                w.writerow([an, f, g])
    return dest


def load_pre_csv(path) -> dict:
    out = {}
    pth = Path(path)
    if not pth.exists():
        return out
    with open(pth, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["animal"], []).append(
                (float(row["F_deficit"]), float(row["G_reorg"])))
    return out


def load_csv(path) -> list[dict]:
    """Re-read a written series, so a summary can be revised without rebuilding the bundles.

    This is the sidecar earning its keep on the first day it exists: correcting the reference in
    `summarise` would otherwise have cost a full twenty-minute re-derivation of numbers that had
    not changed.
    """
    num = {"day", "matched", "n_trials", "frozen", "refit", "gap", "pre_frozen", "pre_refit", "pre_gap",
           "F_deficit", "G_reorg", "refit_deficit"}
    def cast(k, v):
        if k not in num:
            return v
        return float(v) if v not in ("", None) else None

    with open(path, newline="", encoding="utf-8") as fh:
        rows = [{k: cast(k, v) for k, v in row.items()} for row in csv.DictReader(fh)]
    for r in rows:
        # SELF-HEALING rather than merely tolerant: refit_deficit is F - G by construction, so a
        # series written before the column existed can be completed on the way in instead of
        # carrying a blank that the next writer would propagate.
        if r.get("refit_deficit") is None and r.get("F_deficit") is not None                 and r.get("G_reorg") is not None:
            r["refit_deficit"] = r["F_deficit"] - r["G_reorg"]
    return rows


def write_csv(rows, dest: Path) -> Path:
    """Every plotted value, because a figure whose numbers exist nowhere drifts from its own text."""
    assert_writable(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["animal", "matched", "day", "epoch", "n_trials", "frozen", "refit", "gap",
            "pre_frozen", "pre_refit", "pre_gap", "F_deficit", "G_reorg", "refit_deficit"]
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        # `.get`, not `[...]`: a series reloaded from a CSV written before a column
        # existed must round-trip rather than KeyError on the way back out.
        w.writerows({c: r.get(c, '') for c in cols} for r in rows)
    return dest


def _animal_colours():
    from wfield_local.grant_figures import ANIMALS
    return {a: c for a, c in zip(ANIMALS, ("#4C72B0", "#DD8452", "#55A868", "#C44E52"))}


def _by_animal(rows):
    out = {}
    for r in rows:
        out.setdefault(r["animal"], []).append(r)
    return {a: sorted(v, key=lambda r: r["day"]) for a, v in sorted(out.items())}


def figure(rows, summary, out_dir, align, variant, tag="unmatched") -> list[Path]:
    """TWO panels: F over days and G over days. The grant figure.

    Time is on the x axis, which is where it belongs for a claim about a route. An earlier version
    put F against G in a third panel and titled it "retrace = a line, migrate = a loop"; that
    over-promised. Separating an outward path from a return path needs them further apart than the
    noise, and with ~12 sessions per animal scattering by +/-0.05 in G there is no such separation
    to find. The per-animal trajectory view is drawn SEPARATELY by `figure_by_animal`, where it can
    be read one animal at a time instead of four crossing paths in one axes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = _animal_colours()
    by = _by_animal(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for an, rs in by.items():
        d = [r["day"] for r in rs]
        axes[0].plot(d, [r["F_deficit"] for r in rs], "o-", color=col.get(an), label=an, ms=4.5,
                     lw=1.6)
        axes[1].plot(d, [r["G_reorg"] for r in rs], "o-", color=col.get(an), label=an, ms=4.5,
                     lw=1.6)
    for a, lab, ttl in ((axes[0], "frozen-readout deficit  (pre - frozen)",
                         "A   the pre-stroke readout recovers"),
                        (axes[1], "reorganisation  (refit gap - pre gap)",
                         "B   what it cannot reach does not")):
        a.axhline(0, color="0.35", lw=1.0, ls="--")
        a.set_xlabel("days from lesion")
        a.set_ylabel(lab)
        a.set_title(ttl, fontsize=11)
        a.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8.5, frameon=False, ncol=2)
    c = summary["_cohort"]
    fig.suptitle(f"Recovery trajectory ({tag.upper()} training sets), {align}-aligned / {variant}\n"
                 f"{c['animals']} animals, "
                 f"{len(rows)} post-stroke sessions", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _write(fig, out_dir, f"recovery_trajectory_{tag}_{align}_{variant}")


def figure_by_animal(rows, summary, out_dir, align, variant, pre=None,
                     tag="unmatched") -> list[Path]:
    """One (F, G) trajectory per animal, so four paths are not asked to share one axes.

    **The reading is the ENDPOINT, not the shape.** Time runs from the acute state -- high F, high G,
    upper right -- leftwards as the frozen readout recovers. Where the path finishes vertically is
    the whole question:

        ends at the ORIGIN            the code came back and nothing is left the old readout misses
        ends at the UPPER LEFT        the readout recovered and something persists beyond it

    Start is an open square, end a filled star, and the connecting segments carry arrowheads, because
    without a direction marker a scatter of joined points can be read in either temporal order -- and
    the two accounts differ precisely in direction.

    Axes are SHARED across animals. Per-panel autoscaling would make a mouse whose G never left 0.02
    look like one whose G sat at 0.2.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = _animal_colours()
    by = _by_animal(rows)
    if not by:
        return []
    fx = [r["F_deficit"] for r in rows] + [q[0] for v in (pre or {}).values() for q in v]
    gy = [r["G_reorg"] for r in rows] + [q[1] for v in (pre or {}).values() for q in v]
    xlim = (min(fx) - 0.04, max(fx) + 0.04)
    ylim = (min(gy) - 0.03, max(gy) + 0.04)
    days = [r["day"] for r in rows]

    n = len(by)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.9), squeeze=False)
    sc = None
    for ax, (an, rs) in zip(axes[0], by.items()):
        x = np.array([r["F_deficit"] for r in rs])
        y = np.array([r["G_reorg"] for r in rs])
        d = np.array([r["day"] for r in rs])
        ax.axhline(0, color="0.55", lw=0.9, ls="--")
        ax.axvline(0, color="0.55", lw=0.9, ls="--")
        # THE UNITY LINE F = G, WHICH IS "THE REFIT DECODER IS BACK TO BASELINE".
        # Algebraically F - G = pre_refit - refit, so the vertical distance from a point DOWN to
        # this line is the refit arm's own deficit -- the information NO decoder recovers. On the
        # line, everything is back and the whole frozen deficit is readout mismatch; below it, some
        # of the code is genuinely gone; above it, the session decodes better than it did before the
        # lesion. It is the line that separates "displaced" from "lost", which is the distinction
        # this whole family exists to draw.
        lim = [max(xlim[0], ylim[0]), min(xlim[1], ylim[1])]
        if lim[1] > lim[0]:
            ax.plot(lim, lim, ls=(0, (6, 4)), color="0.45", lw=1.1, zorder=0)
        # WHAT "BACK TO PRE-STROKE" LOOKS LIKE WHEN NOTHING HAPPENED. Pre-stroke sessions are
        # plotted faintly with their mean and +/-1 SEM, because the origin alone gives the return
        # no scale: a star 0.05 away is a full return or a residual deficit depending entirely on
        # how far pre-stroke sessions sit from their own mean.
        #
        # NEITHER AXIS IS ZERO BECAUSE EITHER ARM IS PERFECT. Pre-stroke the frozen decoder runs
        # 0.79-0.94, not 1.0, and refitting is WORSE than frozen (gap -0.056 to -0.101, the
        # training-set-size handicap). Both are absorbed into the baseline by construction; the
        # cloud is the dispersion that survives that subtraction.
        pp = (pre or {}).get(an) or []
        if pp:
            pf = np.array([q[0] for q in pp])
            pg = np.array([q[1] for q in pp])
            ax.scatter(pf, pg, s=16, c="0.72", edgecolor="none", zorder=0)
            n = max(1, len(pf))
            sf, sg = pf.std(ddof=1) / np.sqrt(n), pg.std(ddof=1) / np.sqrt(n)
            ax.errorbar(pf.mean(), pg.mean(), xerr=sf, yerr=sg, fmt="o", ms=6.5,
                        color="0.35", ecolor="0.35", elinewidth=1.6, capsize=3, zorder=6)

        # THE LESION STEP, origin -> first post-stroke session. Without it the square floats
        # unexplained: it sits far from the origin BECAUSE the lesion put it there, and a reader
        # seeing only the post-stroke path cannot tell whether the journey began at the plus or at
        # the square. Drawn dotted and grey so it reads as the event between two states rather than
        # as another measured transition -- nothing was recorded between pre-stroke and day 1.
        ax.annotate("", xy=(x[0], y[0]), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": "0.45", "ls": ":", "lw": 1.1,
                                "shrinkA": 5, "shrinkB": 6}, zorder=1)
        ax.plot(x, y, "-", color=col.get(an), alpha=0.45, lw=1.3, zorder=1)
        for i in range(len(x) - 1):
            ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                        arrowprops={"arrowstyle": "-|>", "color": col.get(an), "alpha": 0.55,
                                    "lw": 0.9, "shrinkA": 3.5, "shrinkB": 3.5}, zorder=2)
        sc = ax.scatter(x, y, c=d, cmap="viridis", vmin=min(days), vmax=max(days), s=38,
                        zorder=3, edgecolor="0.25", linewidth=0.6)
        ax.scatter([x[0]], [y[0]], marker="s", s=95, facecolor="none", edgecolor="0.15",
                   linewidth=1.4, zorder=4)
        ax.scatter([x[-1]], [y[-1]], marker="*", s=210, color=col.get(an), edgecolor="0.15",
                   linewidth=0.8, zorder=5)
        v = summary.get(an, {})
        ax.set_title(f"{an}   final F={v.get('final_F', float('nan')):+.3f}  "
                     f"G={v.get('final_G', float('nan')):+.3f}", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("frozen-readout deficit  F")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("reorganisation  G")
    fig.suptitle(
        "Per-animal recovery trajectory. THE ORIGIN IS THAT ANIMAL'S OWN PRE-STROKE STATE -- both "
        "axes are differences from its baseline.\n"
        "grey dots = pre-stroke sessions, dark cross = their mean +/-1 SEM (the spread with NO "
        "lesion);  dotted arrow = the lesion;  square = first post-stroke session, star = last\n"
        "DASHED DIAGONAL F=G: the refit decoder is back to baseline. Height above zero = the "
        "DISPLACED part;  distance below the diagonal = the part NO decoder recovers.",
        fontsize=9.5)
    # SUBPLOTS_ADJUST FIRST, THEN THE COLOURBAR IN ITS OWN AXES. Calling `fig.colorbar(ax=...)`
    # reserves space by shrinking the axes it is given, and a later `subplots_adjust` silently undoes
    # that -- which put the bar on top of the last panel's title and clipped it.
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.06, right=0.90)
    if sc is not None:
        cax = fig.add_axes((0.925, 0.15, 0.012, 0.63))
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("days from lesion", fontsize=9)
    return _write(fig, out_dir, f"recovery_trajectory_byanimal_{tag}_{align}_{variant}")


def _write(fig, out_dir, stem) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    assert_writable(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for ext in ("png", "svg"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=180)
        made.append(p)
    plt.close(fig)
    return made



def run(align="cue", variant="working", out_dir=None, from_csv=None) -> dict:
    """Both families, always. See the module docstring: the raw gap is positive pre-stroke under
    matching and negative under the unmatched design, so neither alone is about the lesion, and a
    reader given one family cannot tell which side of the bracket they are on."""
    from wfield_local.paths import PathResolver

    out_dir = Path(out_dir) if out_dir else (
        Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    made, summaries = [], {}
    for matched in (False, True):
        tag = "matched" if matched else "unmatched"
        stem = f"recovery_trajectory_{tag}_{align}_{variant}"
        rows = (load_csv(Path(from_csv).with_name(stem + ".csv")) if from_csv
                else series(align, variant, matched=matched))
        if not rows:
            print(f"[recovery_trajectory] {tag}: no paired records", flush=True)
            continue
        summary = summarise(rows)
        summaries[tag] = summary
        pre_path = out_dir / f"{stem}_pre.csv"
        pre = (load_pre_csv(Path(from_csv).with_name(pre_path.name)) if from_csv
               else pre_points(align, variant, matched=matched))
        if not from_csv:
            write_pre_csv(pre, pre_path)
        made += figure(rows, summary, out_dir, align, variant, tag=tag)
        made += figure_by_animal(rows, summary, out_dir, align, variant, pre=pre, tag=tag)
        csvp = write_csv(rows, out_dir / f"{stem}.csv")

        print(f"\n[recovery_trajectory] {align}/{variant} {tag.upper()}: "
              f"{len(rows)} sessions", flush=True)
        print(f"{'animal':8s} {'n':>3s} {'acuteG':>7s} {'post':>5s} {'slope/day':>10s} "
              f"{'late>acute':>11s} {'finalF':>7s} {'finalG':>7s}", flush=True)
        for an, v in summary.items():
            if an == "_cohort":
                continue
            print(f"{an:8s} {v['n_sessions']:3d} {v['acute_G']:7.3f} {v['n_post']:5d} "
                  f"{v['post_slope']:10.4f} {v['late_exceeds_acute']!s:>11s} "
                  f"{v['final_F']:7.3f} {v['final_G']:7.3f}", flush=True)
        c = summary["_cohort"]
        print(f"  cohort: post-acute slope negative {c['negative_post_slope']}/"
              f"{c['animals_scored']}; exceeds acute later {c['late_exceeds_acute']}/"
              f"{c['animals']}", flush=True)
        print(f"[recovery_trajectory] -> {csvp}", flush=True)
    return summaries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", default="cue", choices=("cue", "precue", "lick"))
    ap.add_argument("--variant", default="working", choices=("working", "lick"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--from-csv", default=None,
                    help="re-summarise a written series without rebuilding the bundles")
    args = ap.parse_args(argv)
    run(args.align, args.variant, args.output, args.from_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
