"""Does the `restw` baseline on `restdock05` DRIFT ACROSS EPOCHS? The narrow question §0b leaves open.

`uniform_shift_check` measured how much this baseline moves when the DEFINITION changes -- relaxing
the lick buffer from [1.0, 2.0] to [0.5, 1.0] -- and put it at a median 3.1% of the evoked signal.
That retired a QUIET-era figure (+0.0026 against a ~0.005 signal, about HALF the signal) which had
been used to argue `_RESTref_` cannot carry an across-epoch amplitude claim.

BUT 3.1% IS NOT THE QUANTITY AN ACROSS-EPOCH CLAIM RESTS ON. Sensitivity-to-a-definitional-change
and drift-between-pre-and-chronic are different numbers, and only the second one bears on reading
`map - restw` amplitudes across epochs. `docs/STATUS_2026-09-17.md` §0b says so explicitly and calls
the epoch-wise version cheap and unmeasured. This is it.

WHAT IS MEASURED, and it is deliberately THE EXACT SUBTRAHEND THE MAPS USE -- `session_restw_svt`,
the position-WEIGHTED flat rest baseline, on the configured variant, engagement-gated (which
`rest_frames_by_position` does by default, and which `docs/REST_ENGAGEMENT_AUDIT.md` requires: the
quit period is 3.1% of rest frames pre-stroke and 18.7% acute, so an ungated baseline has its
composition track the independent variable). Measuring a reimplementation would answer a question
about the reimplementation.

    b_s     the session's restw baseline, projected to ATLAS PIXELS and masked
    e_s     that session's evoked map: mean post-cue working-trial activity MINUS b_s.
            This is exactly a `_RESTref_` map averaged over positions, which is the point --
            the denominator is the thing whose amplitude an across-epoch claim would compare.
    k_s     ||e_s||, the session's own signal magnitude

`b_s / k_s` is the baseline IN UNITS OF THAT SESSION'S EVOKED SIGNAL, which is what removes the
cross-day MULTIPLICATIVE scaling (expression, bleaching, window clarity) that no subtraction touches
and that `crossday_intensity` owns. Without it this measures how bright the window was.

    d_E   = mean_{s in E}(b_s/k_s) - mean_{s in pre}(b_s/k_s)
    ratio = ||d_E||          already a FRACTION of the evoked signal -- no further division
    cos   = cos(d_E, mean_{s in pre}(e_s/k_s))

THE NULL IS THE WHOLE POINT, AND WITHOUT IT THE RATIO MEANS NOTHING. Baselines differ session to
session for reasons that have nothing to do with a lesion -- depth of rest, window clarity, how much
of the session the animal worked. So `d_E` is nonzero for any two groups of sessions and reading it
against ZERO would report drift in a cohort that has none. The null splits the PRE sessions into a
group of size `n_E` and the remainder and computes the identical statistic, so the epoch shift is
read against the ordinary session-to-session spread.

That null is CONSERVATIVE by construction: its second group holds `n_pre - n_E` sessions against the
observed statistic's `n_pre`, so its mean is noisier and its differences run slightly LARGE. It can
therefore hide a real shift; it cannot manufacture one.

ALIGNMENT IS REPORTED BECAUSE MAGNITUDE ALONE MISSES HALF OF IT, the same reasoning as
`uniform_shift_check`. A `d` ORTHOGONAL to the evoked pattern is estimation noise and harmless at
small magnitude. A `d` ALIGNED with it subtracts signal systematically at every position, which is
the failure `restw` exists to prevent, arriving by the across-epoch route.

BUT THE RAW COSINE IS BIASED POSITIVE AND MUST NEVER BE READ AGAINST ZERO. `d` carries the term
`-mean_pre(bn)`, and `e_ref = mean_pre(post/k) - mean_pre(bn)` carries the SAME vector with the SAME
sign. That shared additive component correlates the two before any drift exists, so a large positive
cosine is the null expectation, not a finding. Found on the first PS92 run, where the raw chronic
cosine of +0.77 looked like a clean alignment result and was partly construction.

NO VALID NULL FOR IT EXISTS IN THIS DESIGN. `epoch_row` records the two that were tried and the
algebra that defeats each: any group resampled WITHIN pre has the shared term partly cancel, while
the observed epoch is DISJOINT from pre and keeps it in full. Attaching such a null's p to the
observed cosine would be ANTI-CONSERVATIVE -- worse than reporting none. The cosine is therefore
reported DESCRIPTIVELY as `cos_with_evoked_biased` and carries NO p; removing the bias at source
needs `e_ref` built from pre sessions held OUT of `d`, which this cohort's pre counts cannot afford.
The MAGNITUDE never had this problem (it is a difference of two group means either way) and is the
column the claim rests on.

    python -m scripts.rest_migration.rest_baseline_epoch_drift [--animals PS92 ...] [--perm 2000]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

EPOCHS = ("acute", "subacute", "chronic")


def _session_vectors(s, align, post_s, variant):
    """``(b, e)`` -- the masked atlas-pixel restw baseline and evoked map, or ``None``.

    Both come back in the SAME space and from the SAME session load, so `e` is literally a
    `_RESTref_` map: no second basis, no second mask, nothing to fall out of step.
    """
    from wfield_local import joint_basis
    from wfield_local.beta_maps import MAP_SHAPE, stat_mask
    from wfield_local.position_reference_maps import _working_xy, session_restw_svt

    u, v = joint_basis._load_session(s["mc"])
    basew, positions = session_restw_svt(s, v)
    if basew is None:
        return None, "no restw baseline"
    base = np.asarray(basew).ravel()

    X, y = _working_xy(s, align, post_s, variant, "none", v)
    if not len(y):
        return None, "no working trials"
    K = u.shape[1]
    if X.shape[1] % K:
        return None, f"{X.shape[1]} features not a multiple of {K}"
    n_bins = X.shape[1] // K
    # BINS AVERAGED then trials averaged, exactly as `session_raw_maps._map` does. The per-bin maps
    # are the response's trajectory inside the window; their mean is the window's summary, and they
    # share a spatial basis so averaging is defined.
    post = np.asarray(X).reshape(len(y), n_bins, K).mean(axis=1).mean(axis=0)

    m = stat_mask().reshape(-1)
    b = (u @ base).reshape(MAP_SHAPE).reshape(-1)[m]
    e = (u @ (post - base)).reshape(MAP_SHAPE).reshape(-1)[m]
    got = normalise(b, e)
    if got is None:
        return None, "zero evoked norm"
    return (got[0], got[1], len(y), len(positions)), None


def normalise(b, e):
    """``(b/k, e/k)`` where ``k = ||e||`` -- or ``None`` if the session carries no signal.

    THE ONE LINE THAT MAKES SESSIONS COMPARABLE ACROSS DAYS, and the design's central claim: the
    result is invariant to multiplying that session's data by any positive constant. Expression,
    bleaching and window clarity all act as exactly such a constant, and NO SUBTRACTION REMOVES A
    GAIN TERM -- `crossday_intensity` owns that confound and this is the local defence against it.
    Without this the measurement would largely report how bright the window was on each day.

    Pure and separate from the session load so the invariance can be pinned by a test rather than
    asserted in a docstring.
    """
    k = float(np.linalg.norm(e))
    if not np.isfinite(k) or k <= 0:
        return None
    return np.asarray(b) / k, np.asarray(e) / k


def _stat(bn, group_a, group_b, e_ref):
    """``(ratio, cos)`` for two index groups. The ONE place the statistic is defined."""
    d = bn[group_a].mean(axis=0) - bn[group_b].mean(axis=0)
    nd = float(np.linalg.norm(d))
    ne = float(np.linalg.norm(e_ref))
    cos = float(d @ e_ref / (nd * ne)) if nd > 0 and ne > 0 else np.nan
    return nd, cos


def epoch_row(bn, e_ref, idx, pre, n_perm, rng):
    """``dict`` of the observed statistic and its pre-vs-pre null. Pure -- no session data.

    Split out of `run` so the SCIENCE is testable without a share mount: the positive control
    (an "epoch" drawn from pre must not come out significant), the closed-form zero case, and the
    cosine's construction bias all exercise this function directly.
    """
    ratio, cos = _stat(bn, idx, pre, e_ref)

    # THE MAGNITUDE NULL: two DISJOINT groups of pre sessions, the first the epoch's size. This
    # matches the observed statistic's structure and is conservative -- `gb` holds `n_pre - n_E`
    # sessions against the observed `n_pre`, so its mean is noisier and its differences run large.
    #
    # THE COSINE GETS NO p, AND THAT IS A RESULT ABOUT THE METHOD, NOT AN OVERSIGHT.
    # `d_obs = mean_E(bn) - mean_pre(bn)` and `e_ref = mean_pre(post/k) - mean_pre(bn)` share the
    # vector `-mean_pre(bn)` with the same sign, so the raw cosine is biased POSITIVE before any
    # drift exists. Two nulls were tried and NEITHER reproduces that bias:
    #
    #   `mean_ga - mean_gb` over disjoint halves     -- no `-mean_pre` term at all
    #   `mean_ga - mean_pre` with ga a subset of pre -- the term CANCELS. With |ga| = 4 of 20,
    #       `mean_pre = 0.2*mean_ga + 0.8*mean_gb`, so `d = 0.8*(mean_ga - mean_gb)` and the
    #       shared component is gone. Measured `cos_null_median` 0.009 on exchangeable data.
    #
    # The reason is structural: the epoch is DISJOINT from pre, so `-mean_pre` survives in full,
    # while any group resampled WITHIN pre has it partly cancel. A null drawn from pre therefore
    # cannot carry this bias, and attaching its p to the observed cosine would be
    # ANTI-CONSERVATIVE -- worse than reporting no p.
    #
    # THE FIX NEEDS DATA WE DO NOT HAVE: build `e_ref` from pre sessions held OUT of `d`, so
    # nothing is shared. PS92 has 11 pre sessions and acute has 6, which leaves no room to split
    # pre into a reference half and a comparison half and still match the epoch's size. So the
    # cosine is reported DESCRIPTIVELY, its bias is stated, and the magnitude carries the claim.
    draws = []
    for _ in range(n_perm):
        perm = rng.permutation(pre)
        ga, gb = perm[: len(idx)], perm[len(idx):]
        if gb.size == 0:
            continue
        draws.append(_stat(bn, ga, gb, e_ref)[0])
    draws = np.asarray(draws)
    p = float(np.mean(draws >= ratio)) if draws.size else np.nan
    return {"n_sessions": len(idx), "n_pre": len(pre),
            "shift_frac_of_evoked": round(ratio, 4),
            "null_median": round(float(np.median(draws)), 4) if draws.size else None,
            "null_p95": round(float(np.percentile(draws, 95)), 4) if draws.size else None,
            "p": round(p, 4) if np.isfinite(p) else None,
            # DESCRIPTIVE ONLY -- biased positive, no valid null available. See above.
            "cos_with_evoked_biased": round(cos, 4) if np.isfinite(cos) else None}


def run(animal, align, post_s, variant, n_perm, rng, log=print):
    from wfield_local import config, epochs

    rows = []
    bn, en, eps, labs = [], [], [], []
    for s in config.load_sessions():
        if not s["label"].startswith(f"{animal}_"):
            continue
        ep = epochs.epoch_of(s["label"])
        if ep not in ("pre",) + EPOCHS:
            continue
        try:
            got, err = _session_vectors(s, align, post_s, variant)
        except Exception as ex:                                        # noqa: BLE001
            log(f"   !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}")
            continue
        if got is None:
            log(f"   .. {s['label']}: {err} -- skipped")
            continue
        b, e, ntr, npos = got
        bn.append(b)
        en.append(e)
        eps.append(ep)
        labs.append(s["label"])
        log(f"   {s['label']:14s} {ep:9s} trials {ntr:4d}  restw positions {npos}/6")

    if not bn:
        return rows
    bn = np.asarray(bn)
    en = np.asarray(en)
    eps = np.asarray(eps)
    pre = np.flatnonzero(eps == "pre")
    if pre.size < 4:
        log(f"   !! {animal}: {pre.size} pre sessions -- cannot form a null, skipped")
        return rows
    e_ref = en[pre].mean(axis=0)

    for ep in EPOCHS:
        idx = np.flatnonzero(eps == ep)
        if idx.size == 0:
            continue
        r = epoch_row(bn, e_ref, idx, pre, n_perm, rng)
        rows.append({"animal": animal, "epoch": ep, **r})
        log(f"   {animal} {ep:9s} shift {r['shift_frac_of_evoked']:6.3f} of evoked   "
            f"null median {r['null_median']:6.3f} p95 {r['null_p95']:6.3f}   "
            f"p {r['p']:.3f}   cos {r['cos_with_evoked_biased']:+.3f} (biased +, no p)")
    return rows


def main() -> int:
    from wfield_local import config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--align", default="cue")
    ap.add_argument("--post", type=float, default=2.0)
    ap.add_argument("--variant", default="working")
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replot", action="store_true",
                    help="REDRAW from the existing CSV without recomputing. The permutation pass "
                         "is the whole cost here and it buys nothing when only the drawing "
                         "changes. Refuses if the CSV is absent rather than silently recomputing.")
    a = ap.parse_args()

    if a.replot:
        from pathlib import Path

        from wfield_local.paths import PathResolver
        p = Path(a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
                           / "epoch_15j_rest_baseline_epoch_drift.csv"))
        if not p.exists():
            print(f"!! no table at {p} -- run once WITHOUT --replot first.")
            return 1
        with open(p, newline="", encoding="utf-8") as fh:
            rows = [{k: (None if v == "" else v) for k, v in r.items()}
                    for r in csv.DictReader(fh)]
        print(f"REPLOT from {p.name}: {len(rows)} animal-epoch cells")
        print(f"wrote {_figure(rows, p.parent)}")
        return 0

    animals = a.animals or sorted({x["label"].split("_")[0] for x in config.load_sessions()})
    rng = np.random.default_rng(a.seed)
    print(f"REST BASELINE DRIFT ACROSS EPOCHS -- restw on the configured rest variant\n"
          f"align={a.align} post={a.post}s variant={a.variant} perm={a.perm}\n"
          f"{'=' * 78}", flush=True)

    rows = []
    for an in animals:
        print(f"\n{an}", flush=True)
        rows += run(an, a.align, a.post, a.variant, a.perm, rng,
                    log=lambda m: print(m, flush=True))

    if not rows:
        print("\nno rows -- a failed run, not a result")
        return 1

    from pathlib import Path

    from wfield_local.paths import PathResolver
    out = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
                    / "epoch_15j_rest_baseline_epoch_drift.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}", flush=True)

    print(f"\n{'=' * 78}\nSUMMARY -- the baseline shift as a FRACTION of the evoked signal\n"
          f"{'=' * 78}")
    print(f"{'epoch':<10}{'n':>4}{'shift':>9}{'null':>8}{'p<.05':>8}{'cos (biased +)':>17}")
    for ep in EPOCHS:
        v = [r for r in rows if r["epoch"] == ep]
        if not v:
            continue
        sh = np.array([r["shift_frac_of_evoked"] for r in v])
        nu = np.array([r["null_median"] for r in v], dtype=float)
        co = np.array([r["cos_with_evoked_biased"] for r in v], dtype=float)
        sig = sum(1 for r in v if r["p"] is not None and r["p"] < 0.05)
        print(f"{ep:<10}{len(v):>4}{np.median(sh):>9.3f}{np.nanmedian(nu):>8.3f}"
              f"{sig:>5}/{len(v):<2}{np.nanmedian(co):>+17.3f}")

    print("\nHOW TO READ IT -- THE SHIFT AGAINST ITS NULL, NEVER AGAINST ZERO:")
    print("  shift <= null, p large        ->  the baseline does NOT drift beyond ordinary")
    print("      session-to-session spread. `_RESTref_` carries an across-epoch amplitude claim.")
    print("  shift > null                  ->  the baseline MOVES more than ordinary session")
    print("      spread, and across-epoch `map - restw` amplitudes carry that movement.")
    print("\n  THE COSINE IS BIASED POSITIVE AND CARRIES NO p. `d` holds `-mean_pre(bn)` and")
    print("  `e_ref` is `mean_pre(post/k) - mean_pre(bn)`, so both carry that vector with the same")
    print("  sign. No null resampled WITHIN pre reproduces it -- the term cancels there, while the")
    print("  observed epoch is DISJOINT from pre and keeps it in full -- so it is DESCRIPTIVE")
    print("  only. Read the SHIFT column for the claim; use the cosine's SIGN at most.")

    fig = _figure(rows, out.parent)
    if fig is not None:
        print(f"\nwrote {fig}", flush=True)
    return 0


def _figure(rows, out_dir):
    """The drift as a figure, so it can go ON A SLIDE instead of staying in a CSV.

    IT EXISTED ONLY AS A TABLE UNTIL 2026-09-17 EVENING, which is how a result stays unread: the
    deck places FIGURES, and a family with no PNG is invisible to the deck's completeness check as
    well -- that check reports figures it EXPECTS and is silent about ones it was never told about.

    EACH SHIFT IS DRAWN AGAINST ITS OWN NULL, never against zero. The null is the ordinary
    session-to-session spread of this animal's pre-stroke baselines, so a bar shorter than its
    marker means the epoch moved no more than pre-stroke days move among themselves.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config

    colors = config.animal_color()
    animals = sorted({r["animal"] for r in rows})
    fig, axes = plt.subplots(1, len(EPOCHS), figsize=(3.4 * len(EPOCHS) + 0.6, 4.1),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(top=0.70, bottom=0.14)
    for j, ep in enumerate(EPOCHS):
        ax = axes[0][j]
        xs, seen = [], []
        for i, an in enumerate(animals):
            v = [r for r in rows if r["animal"] == an and r["epoch"] == ep]
            if not v:
                continue
            r = v[0]
            sh = float(r["shift_frac_of_evoked"])
            p = r["p"]
            sig = p is not None and float(p) < 0.05
            ax.bar(i, sh, width=0.62, color=colors.get(an, "0.5"),
                   edgecolor="k" if sig else "none", linewidth=1.6 if sig else 0,
                   alpha=1.0 if sig else 0.55)
            # THE NULL, AS A MARKER ON THE BAR. Drawn per animal because it is per animal: it is
            # that animal's own pre-stroke session-to-session spread, not a shared threshold.
            if r["null_p95"] is not None:
                ax.plot([i - 0.38, i + 0.38], [float(r["null_p95"])] * 2, "-", color="k", lw=1.4)
            if r["null_median"] is not None:
                ax.plot([i - 0.30, i + 0.30], [float(r["null_median"])] * 2, ":", color="0.35",
                        lw=1.2)
            xs.append(i)
            seen.append(an)
        ax.set_xticks(xs)
        ax.set_xticklabels(seen, fontsize=9)
        ax.set_title(ep, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("baseline shift, fraction of the evoked signal", fontsize=9)
    fig.text(0.5, 0.985, "epoch_15j -- does the REST BASELINE itself move across epochs?",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925,
             "Bar: || mean_epoch(b/k) - mean_pre(b/k) ||, each session's restw baseline divided by "
             "its OWN evoked norm k, which is what cancels cross-day multiplicative scaling.\n"
             "SOLID line = that animal's null 95th percentile, dotted = its null median -- the "
             "spread of its own pre-stroke baselines. A bar below the solid line moved no more "
             "than pre-stroke days move among themselves.\n"
             "Outlined + opaque = p < 0.05 against that null. IT IS AN UPPER BOUND on the "
             "amplitude bias, attained only if the shift aligns with the evoked pattern; an "
             "orthogonal shift costs ~r^2/2, so 0.15 means between ~1% and ~15%.",
             ha="center", va="top", fontsize=8.5)
    out = out_dir / "epoch_15j_rest_baseline_epoch_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
