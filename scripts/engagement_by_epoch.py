"""`epoch_1f`: engagement by EPOCH, with no position axis -- because there is no position effect.

WHY THIS REPLACES THE PER-POSITION PANEL. `flag_engagement` / `reference_engagement` judge
working-vs-stopped ONLY at the spared REFERENCE positions (near ipsi, near middle), deliberately,
so that a run of misses at an impaired position counts as the deficit rather than as
disengagement. Engagement is therefore a SESSION-level property applied to every trial in that
session. Positions are interleaved in ~6-trial blocks, so a terminal stopped tail truncates all
six roughly equally and the per-position panel is flat BY CONSTRUCTION -- measured, the acute
spread across the six is 0.031 against an epoch effect of 0.18. Six bars invite a reader to look
for a spatial pattern that the design cannot produce. This draws the quantity that exists.

    pre 0.96 -> acute 0.78 -> subacute 0.81 -> chronic 0.93

THE SOURCE IS THE BEHAVIOUR PIPELINE'S OWN SESSION TABLE, `cohort_session_metrics.csv`, which
carries `n_engaged` and `n_disengaged` per session. That is the same gate every analysis figure
and the epoch rule already use, so this figure cannot disagree with them about what "working"
means. It also makes the figure auditable: the per-position version on the share is produced by
code that is not in this repository.

VERIFIED AGAINST THAT VERSION before replacing it -- pre 0.962 / acute 0.782 / subacute 0.825 /
chronic 0.945 here against 0.959 / 0.782 / 0.814 / 0.931 there, over 96 sessions. Acute agrees to
three decimals. The small differences are weighting: that figure is trial-weighted within
position, this one is the session mean.

ALSO DRAWS THE PER-ANIMAL TIMECOURSE (`epoch_1g`), for the same reason the bar figure exists:
the deck's `epoch_1e` is produced by code that is not in this repository, so it can be neither
audited nor regenerated here. The epoch BINS are animal-specific -- acute is a fraction of each
animal's post-stroke days and chronic starts when THAT animal's hit rate flattens -- so a
pooled bar averages bins that do not mean the same thing across animals. This panel shows what
was averaged.

    python -m scripts.engagement_by_epoch [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

KEY = "all positions"
N_BOOT = 2000


def sessions_by_epoch():
    """``(values, points, counts, n)`` -- engaged fraction per session, grouped by epoch."""
    import pandas as pd

    from wfield_local import config, epochs
    from wfield_local import epoch_audit as ea

    df = pd.read_csv(ea._cohort_path())
    # SAME DEDUPE RULE AS `load_far_position_tables`: two sessions were recorded on one date for
    # PS94, and a label is a date. Keep the one with the most engaged trials rather than an
    # arbitrary pick that could change between runs.
    df = df.sort_values("n_engaged").drop_duplicates(["animal", "date"], keep="last")

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    values, points, counts = {}, {}, {}
    n = 0
    for _i, r in df.iterrows():
        lab = f"{r['animal']}_{str(int(r['date']))[4:]}"
        if lab not in want:
            continue
        tot = float(r["n_engaged"]) + float(r["n_disengaged"])
        if not np.isfinite(tot) or tot <= 0:
            continue
        try:
            ep = epochs.epoch_of(lab)
        except Exception:                                                # noqa: BLE001
            ep = None
        if not ep:
            continue
        points.setdefault(ep, {}).setdefault(KEY, []).append((r["animal"],
                                                              float(r["n_engaged"]) / tot))
        n += 1
    for ep, by in points.items():
        vals = [v for _a, v in by[KEY]]
        values[ep] = {KEY: float(np.mean(vals))}
        counts[ep] = {KEY: len(vals)}
    return values, points, counts, n


def timecourse(out_dir):
    """`epoch_1g`: engaged fraction against days since that animal's OWN lesion, one line each.

    THE POOLED BARS AVERAGE ANIMAL-SPECIFIC BINS. Acute is a fraction of each animal's post-stroke
    days and chronic starts when that animal's hit rate flattens, so PS94's subacute runs to day 29
    while PS92's is days 7-9. A pooled number is therefore an average over bins that do not mean
    the same thing across animals; this shows the sessions behind it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config, epochs

    _v, points, _c, _n = sessions_by_epoch()
    per = {}
    for ep, by in points.items():
        for an, frac in by[KEY]:
            per.setdefault(an, []).append((ep, frac))
    # RE-DERIVE THE DAY from the label rather than the epoch, because the epoch is the thing this
    # figure exists to unpack.
    import pandas as pd

    from wfield_local import epoch_audit as ea
    df = pd.read_csv(ea._cohort_path())
    df = df.sort_values("n_engaged").drop_duplicates(["animal", "date"], keep="last")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    rows = []
    for _i, r in df.iterrows():
        lab = f"{r['animal']}_{str(int(r['date']))[4:]}"
        if lab not in want:
            continue
        tot = float(r["n_engaged"]) + float(r["n_disengaged"])
        if not np.isfinite(tot) or tot <= 0:
            continue
        d = epochs.days_since_stroke(lab)
        rows.append((r["animal"], -1 if d is None else int(d), float(r["n_engaged"]) / tot))

    colors = config.animal_color()
    animals = sorted({a for a, _d, _f in rows})
    fig, axes = plt.subplots(1, len(animals), figsize=(4.0 * len(animals), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, an in zip(axes, animals):
        pre = sorted(f for a, d, f in rows if a == an and d < 0)
        post = sorted((d, f) for a, d, f in rows if a == an and d >= 0)
        c = colors.get(an, "0.4")
        if pre:
            # PRE AS ONE POINT AT x=0 PLUS ITS SPREAD -- a pre-stroke session has no meaningful
            # "day since lesion", and scattering them over negative days would invent an axis.
            ax.errorbar([0], [float(np.mean(pre))],
                        yerr=[[float(np.mean(pre) - min(pre))], [float(max(pre) - np.mean(pre))]],
                        fmt="s", color=c, ms=7, capsize=3, lw=1.2, label="pre (mean, range)")
        if post:
            ax.plot([d for d, _f in post], [f for _d, f in post], "o-", color=c, lw=1.6, ms=5,
                    label="post-stroke sessions")
        ax.axhline(1.0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(an, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("days since that animal`s lesion", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("fraction of trials working", fontsize=11)
    axes[0].set_ylim(0.0, 1.06)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Engagement over days from lesion, per animal -- the sessions behind the pooled "
                 "bars", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    q = Path(out_dir) / "epoch_1g_engagement_timecourse.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import epoch_figures as ef
    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    values, points, counts, n = sessions_by_epoch()
    if not values:
        print("!! no sessions resolved -- is cohort_session_metrics.csv present?")
        return 1
    post = [e for e in ("acute", "subacute", "chronic") if e in values]
    print(f"engagement by epoch, {n} sessions")
    for e in ("pre", *post):
        print(f"   {e:<10} {values[e][KEY]:.3f}   n={counts[e][KEY]}")

    # EACH POST EPOCH AGAINST PRE, nested animals -> sessions, Bonferroni over the three.
    rng = np.random.default_rng(7)
    marks = collections.defaultdict(dict)
    for e in post:
        got = ef.scalar_contrast_draws(points, e, "pre", KEY, rng=rng, n_boot=N_BOOT)
        if got is None:
            continue
        _point, draws = got
        marks[e][KEY] = ef.contrast_marks(draws, n_comparisons=len(post))

    sub = ef.stats_line(
        counts, n_boot=N_BOOT,
        notes=[("bootstrap: animals -> sessions (a session is one number here, so there is no "
                "block level)"),
               ("WORKING is judged ONLY at the spared reference positions, so a run of misses at "
                "an impaired position counts as the deficit rather than as disengagement"),
               ("NO POSITION AXIS: the gate is a SESSION-level property, so the per-position "
                "panel it replaces was flat by construction (acute spread 0.031 across six "
                "positions against an epoch effect of 0.18)")])
    made = ef.bar_row(
        values, out_dir, name="epoch_1f_engagement_by_epoch",
        title="Engagement: fraction of trials the animal was still working, by epoch "
              "(one dot per session)",
        subtitle=sub, counts={e: counts[e][KEY] for e in counts}, marks=marks, points=points,
        ylabel="fraction of trials working", positions=[KEY], ylim=(0.0, 1.06))
    print(f"[1f] wrote {made}")
    print(f"[1g] wrote {timecourse(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
