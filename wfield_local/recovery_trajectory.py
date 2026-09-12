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
                 the (F, G) trajectory is a LOOP with hysteresis: the return path sits above the
                 outward one.

So the discriminating measurements are (a) whether G's maximum precedes F's recovery, (b) the sign
of G's trend AFTER its peak, and (c) whether any late session exceeds the acute peak of G.

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


def figure(rows, summary, out_dir, align, variant) -> list[Path]:
    """Three panels: the two quantities over time, and the trajectory that distinguishes the accounts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.grant_figures import ANIMALS

    col = {a: c for a, c in zip(ANIMALS, ("#4C72B0", "#DD8452", "#55A868", "#C44E52"))}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    for an in sorted({r["animal"] for r in rows}):
        rs = sorted((r for r in rows if r["animal"] == an), key=lambda r: r["day"])
        d = [r["day"] for r in rs]
        axes[0].plot(d, [r["F_deficit"] for r in rs], "o-", color=col.get(an), label=an, ms=4)
        axes[1].plot(d, [r["G_reorg"] for r in rs], "o-", color=col.get(an), label=an, ms=4)
        # The trajectory, arrowed in time: a LINE traversed out and back is retrace; a LOOP is not.
        axes[2].plot([r["F_deficit"] for r in rs], [r["G_reorg"] for r in rs], "-",
                     color=col.get(an), alpha=0.55, lw=1.2)
        axes[2].scatter([r["F_deficit"] for r in rs], [r["G_reorg"] for r in rs],
                        c=d, cmap="viridis", s=26, zorder=3, edgecolor=col.get(an), linewidth=0.7)

    axes[0].set_ylabel("frozen-readout deficit  (pre − frozen)")
    axes[1].set_ylabel("reorganisation  (gap − pre gap)")
    for a in axes[:2]:
        a.axhline(0, color="0.4", lw=0.9, ls="--")
        a.set_xlabel("days from lesion")
        a.legend(fontsize=8, frameon=False)
    axes[0].set_title("A  does the pre-stroke readout recover?")
    axes[1].set_title("B  does information sit where it cannot see?")
    axes[2].axhline(0, color="0.4", lw=0.9, ls="--")
    axes[2].axvline(0, color="0.4", lw=0.9, ls="--")
    axes[2].set_xlabel("frozen-readout deficit")
    axes[2].set_ylabel("reorganisation")
    axes[2].set_title("C  RETRACE = out and back along a line\nMIGRATE = a loop (colour = day)")

    c = summary["_cohort"]
    fig.suptitle(
        f"Recovery trajectory, {align}-aligned / {variant} — {c['animals']} animals, {len(rows)} "
        f"post-stroke sessions\n"
        f"post-acute reorganisation slope negative in {c['negative_post_slope']}/"
        f"{c['animals_scored']} scored; reorganisation exceeds its ACUTE level later in "
        f"{c['late_exceeds_acute']}/{c['animals']} animals",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_dir = Path(out_dir)
    assert_writable(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for ext in ("png", "svg"):
        p = out_dir / f"recovery_trajectory_{align}_{variant}.{ext}"
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
