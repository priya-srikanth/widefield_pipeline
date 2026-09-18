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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
