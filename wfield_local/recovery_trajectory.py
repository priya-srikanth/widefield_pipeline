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


def series(align: str = "cue", variant: str = "working") -> list[dict]:
    """One row per (animal, post-stroke session): frozen, refit, gap, and both pre-subtracted.

    ``F`` and ``G`` are the two axes the accounts differ on -- see the module docstring. Both are
    signed so that ZERO IS THE PRE-STROKE STATE and the lesion moves them positive:

        F = pre_frozen - frozen      the frozen readout's DEFICIT  (recovery drives it to 0)
        G = gap - pre_gap            reorganisation ABOVE baseline (retrace keeps it at 0)
    """
    from wfield_local import epoch_figures as ef
    from wfield_local import grant_figures as G

    per_animal, _days = G._collect_5c(align, variant, "paired")
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
                "epoch": ef.epoch_of_day(an, int(day)) or "",
                "n_trials": n,
                "frozen": fz, "refit": rf, "gap": rf - fz,
                "pre_frozen": pre_f, "pre_refit": pre_r, "pre_gap": pre_r - pre_f,
                "F_deficit": pre_f - fz,
                "G_reorg": (rf - fz) - (pre_r - pre_f),
            })
    return rows


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


def load_csv(path) -> list[dict]:
    """Re-read a written series, so a summary can be revised without rebuilding the bundles.

    This is the sidecar earning its keep on the first day it exists: correcting the reference in
    `summarise` would otherwise have cost a full twenty-minute re-derivation of numbers that had
    not changed.
    """
    num = {"day", "n_trials", "frozen", "refit", "gap", "pre_frozen", "pre_refit", "pre_gap",
           "F_deficit", "G_reorg"}
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (float(v) if k in num else v) for k, v in row.items()}
                for row in csv.DictReader(fh)]


def write_csv(rows, dest: Path) -> Path:
    """Every plotted value, because a figure whose numbers exist nowhere drifts from its own text."""
    assert_writable(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["animal", "day", "epoch", "n_trials", "frozen", "refit", "gap",
            "pre_frozen", "pre_refit", "pre_gap", "F_deficit", "G_reorg"]
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)
    return dest


def _animal_colours():
    from wfield_local.grant_figures import ANIMALS
    return {a: c for a, c in zip(ANIMALS, ("#4C72B0", "#DD8452", "#55A868", "#C44E52"))}


def _by_animal(rows):
    out = {}
    for r in rows:
        out.setdefault(r["animal"], []).append(r)
    return {a: sorted(v, key=lambda r: r["day"]) for a, v in sorted(out.items())}


def figure(rows, summary, out_dir, align, variant) -> list[Path]:
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
    fig.suptitle(f"Recovery trajectory, {align}-aligned / {variant} -- {c['animals']} animals, "
                 f"{len(rows)} post-stroke sessions", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _write(fig, out_dir, f"recovery_trajectory_{align}_{variant}")


def figure_by_animal(rows, summary, out_dir, align, variant) -> list[Path]:
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
    fx = [r["F_deficit"] for r in rows]
    gy = [r["G_reorg"] for r in rows]
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
    fig.suptitle("Per-animal recovery trajectory -- square = first session, star = last, "
                 "arrows = time\n"
                 "ending at the ORIGIN = the pre-stroke code returned; "
                 "ending UPPER LEFT = the readout recovered but something it cannot reach persists",
                 fontsize=10.5)
    # SUBPLOTS_ADJUST FIRST, THEN THE COLOURBAR IN ITS OWN AXES. Calling `fig.colorbar(ax=...)`
    # reserves space by shrinking the axes it is given, and a later `subplots_adjust` silently undoes
    # that -- which put the bar on top of the last panel's title and clipped it.
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.06, right=0.90)
    if sc is not None:
        cax = fig.add_axes((0.925, 0.15, 0.012, 0.63))
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("days from lesion", fontsize=9)
    return _write(fig, out_dir, f"recovery_trajectory_byanimal_{align}_{variant}")


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
    from wfield_local.paths import PathResolver

    rows = load_csv(from_csv) if from_csv else series(align, variant)
    if not rows:
        print("[recovery_trajectory] no paired records", flush=True)
        return {}
    summary = summarise(rows)
    # Same destination as the epoch family it extends (`epoch_grant_figures.main`).
    out_dir = Path(out_dir) if out_dir else (
        Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    made = figure(rows, summary, out_dir, align, variant)
    made += figure_by_animal(rows, summary, out_dir, align, variant)
    csvp = write_csv(rows, out_dir / f"recovery_trajectory_{align}_{variant}.csv")

    print(f"\n[recovery_trajectory] {align}/{variant}: {len(rows)} post-stroke sessions",
          flush=True)
    print(f"{'animal':8s} {'n':>3s} {'acuteG':>7s} {'day':>4s} {'post':>5s} {'maxPostG':>9s} "
          f"{'slope/day':>10s} {'corr(F,G)':>10s} {'late>acute':>11s} {'finalF':>7s} "
          f"{'finalG':>7s}", flush=True)
    for an, v in summary.items():
        if an == "_cohort":
            continue
        print(f"{an:8s} {v['n_sessions']:3d} {v['acute_G']:7.3f} {v['acute_day']:4d} "
              f"{v['n_post']:5d} {v['max_post_G']:9.3f} {v['post_slope']:10.4f} "
              f"{v['post_corr_F_G']:10.3f} {v['late_exceeds_acute']!s:>11s} "
              f"{v['final_F']:7.3f} {v['final_G']:7.3f}", flush=True)
    c = summary["_cohort"]
    print(f"\nCOHORT: post-acute slope negative in {c['negative_post_slope']}/"
          f"{c['animals_scored']} scored; reorganisation EXCEEDS its acute level later in "
          f"{c['late_exceeds_acute']}/{c['animals']} animals", flush=True)
    print(f"[recovery_trajectory] -> {made[0]}\n[recovery_trajectory] -> {csvp}", flush=True)
    return summary


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
