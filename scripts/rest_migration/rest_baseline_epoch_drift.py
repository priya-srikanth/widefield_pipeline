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

FIXED 2026-09-18 -- THE COSINE NOW CARRIES A p, and the fix is not the one this file predicted.
`e_ref` is built LEAVE-ONE-ANIMAL-OUT, from the other three animals' pre sessions. That is possible
because `b_s` and `e_s` are atlas PIXELS on the shared grid, so another animal's pre is the same
space; it gives 36 sessions rather than the thin half that splitting one animal's own 11 would, and
it was the per-animal split -- not the idea -- that the counts could not afford.

WHAT IT FIXES: the null becomes STRUCTURALLY MATCHED. Self-referenced, the observed shared
`-mean_pre(bn)` with `e_ref` in FULL while a within-pre null shared it only partly, so the null
understated the bias and its p was anti-conservative. Held out, neither shares anything with
`e_ref`, so the same disjoint-halves null is matched and the p is legitimate.

WHAT IT DOES NOT FIX, MEASURED: it does not centre the null cosine at zero. |null median| is 0.212
held out against 0.233 self-referenced -- barely moved. The nonzero centre was never mainly the
shared term: under the null `d` is a difference between two random groups of pre BASELINES, and
those are not isotropic -- they occupy a low-dimensional, cortically structured subspace that
overlaps the evoked pattern. That is a property of the data and survives any reference. It is also
exactly why the cosine must be read against ITS OWN NULL and never against zero, which is what the
p now does.

THE RESULT: 1 of 11 cells at p < 0.05 against ~0.6 expected by chance. There is NO EVIDENCE the
baseline drift is preferentially aligned with the evoked pattern. That matters for the magnitude's
caveat -- the shift is an UPPER bound on amplitude bias, attained only under alignment, with
orthogonal shifts costing ~r^2/2 -- so the realistic bias sits toward the LOW end of the stated
range. State it as absence of evidence: 11 cells, one test each, no correction.

The MAGNITUDE never had this problem (it is a difference of two group means either way) and is
still the column the claim rests on.

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
    # SUPERSEDED 2026-09-18 -- KEPT BECAUSE THE REASONING IS STILL CORRECT AND THE CONCLUSION IS
    # NOT. Read it as the record of WHY a self-referenced cosine cannot carry a p; the block that
    # follows it supplies the held-out reference that fixes exactly this.
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
    # cosine WAS reported DESCRIPTIVELY, its bias stated, and the magnitude carried the claim.
    # THAT LAST SENTENCE IS THE PART THAT WAS WRONG: the split does not have to be within ONE
    # animal. `b_s` and `e_s` are atlas PIXELS on a shared grid, so the other three animals' pre
    # sessions are the same space and there are 36 of them.
    #
    # THE COSINE NOW GETS A NULL TOO, and only because `e_ref` is held out. With a self-referenced
    # `e_ref` the observed shares `-mean_pre(bn)` with it in FULL while a within-pre null shares it
    # only partly, so the null understated the bias and its p was anti-conservative. A held-out
    # reference shares nothing with EITHER, so the same disjoint-halves null is now structurally
    # matched to the observed and the p is legitimate.
    draws, cos_draws = [], []
    for _ in range(n_perm):
        perm = rng.permutation(pre)
        ga, gb = perm[: len(idx)], perm[len(idx):]
        if gb.size == 0:
            continue
        dr, cr = _stat(bn, ga, gb, e_ref)
        draws.append(dr)
        cos_draws.append(cr)
    draws = np.asarray(draws)
    cos_draws = np.asarray([c for c in cos_draws if np.isfinite(c)])
    p = float(np.mean(draws >= ratio)) if draws.size else np.nan
    return {"n_sessions": len(idx), "n_pre": len(pre),
            "shift_frac_of_evoked": round(ratio, 4),
            "null_median": round(float(np.median(draws)), 4) if draws.size else None,
            "null_p95": round(float(np.percentile(draws, 95)), 4) if draws.size else None,
            "p": round(p, 4) if np.isfinite(p) else None,
            "cos_with_evoked": round(cos, 4) if np.isfinite(cos) else None,
            "cos_null_median": (round(float(np.median(cos_draws)), 4)
                                if cos_draws.size else None),
            # TWO-SIDED on |cos|: an anti-aligned drift biases amplitude just as much as an
            # aligned one, only downward, so a one-sided test would miss half the failure mode.
            "cos_p": (round((1 + int((np.abs(cos_draws) >= abs(cos)).sum()))
                            / (1 + cos_draws.size), 4)
                      if cos_draws.size and np.isfinite(cos) else None)}


def collect(animal, align, post_s, variant, log=print):
    """``(bn, en, eps)`` for one animal -- the loading half of what `run` used to do.

    SPLIT OUT so `main` can build each animal`s evoked reference from the OTHER animals` pre
    sessions without paying the session load twice.
    """
    from wfield_local import config, epochs

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
        return None
    return np.asarray(bn), np.asarray(en), np.asarray(eps)


def run(animal, bn, en, eps, n_perm, rng, log=print, e_held=None):
    rows = []
    pre = np.flatnonzero(eps == "pre")
    if pre.size < 4:
        # THE `return` HERE WAS LOST IN THE collect()/run() SPLIT, so this logged
        # "skipped" and then carried straight on into `en[pre].mean()`. A message that
        # describes control flow the code does not take is worse than no message.
        log(f"   !! {animal}: {pre.size} pre sessions -- cannot form a null, skipped")
        return rows
    e_self = en[pre].mean(axis=0)
    # HELD-OUT REFERENCE, AND IT HAD TO COME FROM OTHER ANIMALS. `b_s` and `e_s` are ATLAS PIXELS
    # on the shared grid, so a reference built from a DIFFERENT animal's pre sessions is in the
    # same space and shares not one session with this animal's `d`. Splitting THIS animal's own
    # pre was the fix first proposed and its counts do not allow it -- PS92 has 11 pre against 6
    # acute. Leave-one-animal-out has 36 pre sessions behind it instead of a thin half.
    #
    # WHAT IT FIXES, AND WHAT IT DOES NOT -- measured 2026-09-18, because the prediction I made
    # for it was wrong and the correction is the useful part.
    #
    #   IT DOES NOT centre the null cosine at zero. Predicted it would; it does not.
    #   |null median| is 0.212 held-out against 0.233 self-referenced -- barely moved. The
    #   nonzero centre was never mainly the shared `-mean_pre(bn)` term: under the null `d` is a
    #   difference between two random groups of pre BASELINES, and those are not isotropic --
    #   they live in a low-dimensional, cortically structured subspace that overlaps the evoked
    #   pattern. That geometry is a property of the data and survives any choice of reference.
    #
    #   IT DOES make the null STRUCTURALLY MATCHED to the observed, which was the actual defect.
    #   Self-referenced, the observed shared `-mean_pre(bn)` with `e_ref` in FULL while a
    #   within-pre null shared it only partly, so the null understated the observed's bias and
    #   its p was ANTI-CONSERVATIVE. Held out, neither shares anything with `e_ref`, so the same
    #   disjoint-halves null is matched and the p is legitimate. That is what was blocking a p,
    #   and it is what is now unblocked.
    e_ref = e_self if e_held is None else e_held
    if e_held is None:
        log(f"   !! {animal}: no held-out reference -- cosine is the BIASED self-referenced one")

    for ep in EPOCHS:
        idx = np.flatnonzero(eps == ep)
        if idx.size == 0:
            continue
        r = epoch_row(bn, e_ref, idx, pre, n_perm, rng)
        # THE BIASED ONE IS KEPT BESIDE IT, not replaced, because the size of the bias is itself
        # the evidence that the held-out reference was necessary.
        r_self = epoch_row(bn, e_self, idx, pre, n_perm, rng)
        r["cos_with_evoked_selfref_biased"] = r_self["cos_with_evoked"]
        r["cos_null_median_selfref"] = r_self["cos_null_median"]
        r["eref"] = "held_out_animals" if e_held is not None else "self_BIASED"
        rows.append({"animal": animal, "epoch": ep, **r})
        log(f"   {animal} {ep:9s} shift {r['shift_frac_of_evoked']:6.3f} of evoked   "
            f"null median {r['null_median']:6.3f} p95 {r['null_p95']:6.3f}   "
            f"p {r['p']:.3f}   cos {r['cos_with_evoked']:+.3f} "
            f"(null {r['cos_null_median']:+.3f}, p {r['cos_p']:.3f})   "
            f"selfref {r['cos_with_evoked_selfref_biased']:+.3f} "
            f"(null {r['cos_null_median_selfref']:+.3f})")
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

    # PASS 1 -- load every animal once.
    got = {}
    for an in animals:
        print(f"\n{an}", flush=True)
        c = collect(an, a.align, a.post, a.variant, log=lambda m: print(m, flush=True))
        if c is not None:
            got[an] = c

    # PASS 2 -- each animal's evoked reference comes from the OTHER animals' PRE sessions.
    # `b_s` and `e_s` are atlas pixels on the shared grid, so this is the same space; it shares no
    # session with this animal's `d`, which kills the positive bias at source and makes the
    # cosine's null legitimate. Splitting the animal's OWN pre was the fix first proposed, and its
    # counts cannot afford it -- PS92 has 11 pre against 6 acute.
    # HOW COMPARABLE ARE THE ANIMALS? The held-out reference is only legitimate to the extent
    # that one animal's mean pre evoked pattern stands in for another's. If they diverge, the
    # cosine is ATTENUATED -- it scores the drift against a partly-wrong target -- and a null
    # result becomes partly a statement about power. Measured and printed, never assumed.
    pre_mean = {an: g[1][g[2] == "pre"].mean(axis=0) for an, g in got.items()
                if (g[2] == "pre").any()}
    names = sorted(pre_mean)
    if len(names) > 1:
        print("\nCROSS-ANIMAL COMPARABILITY of the mean PRE evoked pattern (unit-normalised "
              "per session before averaging, so this is SHAPE, not amplitude):", flush=True)
        rs = []
        for i, a1 in enumerate(names):
            for a2 in names[i + 1:]:
                v1, v2 = pre_mean[a1], pre_mean[a2]
                r = float(np.corrcoef(v1, v2)[0, 1])
                rs.append(r)
                print(f"   r({a1}, {a2}) = {r:+.3f}", flush=True)
        print(f"   mean pairwise r = {np.mean(rs):+.3f}   "
              f"-> attenuation of a held-out cosine is roughly this factor", flush=True)
        for an in names:
            others = [pre_mean[o] for o in names if o != an]
            if others:
                r = float(np.corrcoef(pre_mean[an], np.mean(others, axis=0))[0, 1])
                print(f"   r({an}, mean of OTHERS) = {r:+.3f}   "
                      f"<- the substitution this animal actually pays", flush=True)

    rows = []
    for an, (bn, en, eps) in got.items():
        pool = [g[1][g[2] == "pre"] for g_an, g in got.items() if g_an != an]
        pool = [x for x in pool if len(x)]
        e_held = np.concatenate(pool, axis=0).mean(axis=0) if pool else None
        n_held = sum(len(x) for x in pool)
        print(f"\n{an}: held-out reference from {n_held} pre sessions of "
              f"{len(pool)} other animals", flush=True)
        rows += run(an, bn, en, eps, a.perm, rng,
                    log=lambda m: print(m, flush=True), e_held=e_held)

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
    print(f"{'epoch':<10}{'n':>4}{'shift':>9}{'null':>8}{'p<.05':>8}{'cos':>9}"
          f"{'cosnull':>9}{'cos p<.05':>11}")
    for ep in EPOCHS:
        v = [r for r in rows if r["epoch"] == ep]
        if not v:
            continue
        sh = np.array([r["shift_frac_of_evoked"] for r in v])
        nu = np.array([r["null_median"] for r in v], dtype=float)
        co = np.array([r["cos_with_evoked"] for r in v], dtype=float)
        cn = np.array([r["cos_null_median"] for r in v], dtype=float)
        sig = sum(1 for r in v if r["p"] is not None and r["p"] < 0.05)
        csig = sum(1 for r in v if r.get("cos_p") is not None and float(r["cos_p"]) < 0.05)
        print(f"{ep:<10}{len(v):>4}{np.median(sh):>9.3f}{np.nanmedian(nu):>8.3f}"
              f"{sig:>5}/{len(v):<2}{np.nanmedian(co):>+9.3f}{np.nanmedian(cn):>+9.3f}"
              f"{csig:>8}/{len(v):<2}")

    print("\nHOW TO READ IT -- THE SHIFT AGAINST ITS NULL, NEVER AGAINST ZERO:")
    print("  shift <= null, p large        ->  the baseline does NOT drift beyond ordinary")
    print("      session-to-session spread. `_RESTref_` carries an across-epoch amplitude claim.")
    print("  shift > null                  ->  the baseline MOVES more than ordinary session")
    print("      spread, and across-epoch `map - restw` amplitudes carry that movement.")
    print("\n  THE COSINE HAS ITS OWN p SINCE 2026-09-18, and the two arms are DIFFERENT QUESTIONS:")
    print("      SHIFT column  -- does the baseline MOVE?   6/11 cells: yes, it does.")
    print("      COS column    -- does the movement MATTER? Only the component ALONG the evoked")
    print("          pattern biases a `map - restw` amplitude; an orthogonal shift costs ~r^2/2.")
    print("          1/11 at p < 0.05 against ~0.6 expected -> no evidence of alignment, so the")
    print("          SHIFT column is an upper bound that is rarely attained.")
    print("  READ THE COSINE AGAINST ITS OWN NULL (~0.21), NEVER AGAINST ZERO: pre baselines are")
    print("  not isotropic, so +0.4 is BELOW chance. The held-out reference is what makes the p")
    print("  legitimate -- self-referenced it ran 0.160 at alpha 0.05, held out it runs 0.060.")
    print("  SIGN: cos > 0 SHRINKS the measured amplitude, cos < 0 INFLATES it.")

    fig = _figure(rows, out.parent)
    if fig is not None:
        print(f"\nwrote {fig}", flush=True)
    return 0


def _figure(rows, out_dir):
    """The drift as a figure, so it can go ON A SLIDE instead of staying in a CSV.

    IT EXISTED ONLY AS A TABLE UNTIL 2026-09-17 EVENING, which is how a result stays unread: the
    deck places FIGURES, and a family with no PNG is invisible to the deck's completeness check as
    well -- that check reports figures it EXPECTS and is silent about ones it was never told about.

    TWO ROWS, BECAUSE THE TWO ARMS ANSWER DIFFERENT QUESTIONS AND ONE ROW INVITED THEM TO BE READ
    AS ONE (2026-09-19, Priya: *"so the cosine analysis suggests that there is NOT significant
    epoch-related changes in baseline?"*). It does not, and the single-row figure was why that
    reading was available -- the cosine's p landed in the CSV on 09-18 and never reached the PNG,
    so the only arm on a slide was the magnitude one and its caption still carried the pre-p
    caveat.

        ROW 1  DOES THE BASELINE MOVE?      || d ||, against that animal's own pre-stroke spread.
                                            6 of 11 cells significant. The answer is YES.
        ROW 2  DOES THE MOVEMENT MATTER?    cos(d, evoked), against its own null. 1 of 11 against
                                            ~0.6 expected. No evidence of alignment.

    Only the component of `d` ALONG the evoked pattern biases a `map - restw` amplitude; the
    orthogonal part is estimation noise and costs ~r^2/2. So row 1 is an UPPER BOUND and row 2 is
    the test of whether it is attained.

    THE SIGN IN ROW 2 IS NOT DECORATION. `map = post - (b_pre + d)`, so a shift ALIGNED with the
    evoked pattern (cos > 0) SHRINKS the measured amplitude and an ANTI-aligned one (cos < 0)
    INFLATES it. The one cell that clears p < 0.05 -- PS94 subacute, cos = -0.913 -- is in the
    inflating direction, and is also the largest shift in the table.

    EACH ARM IS DRAWN AGAINST ITS OWN NULL, never against zero, and for the cosine that is
    load-bearing: its null median is 0.212 and NOT 0 (pre baselines are not isotropic), so a bar
    of +0.4 read against zero would look like alignment when it is below chance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config

    def _f(v):
        return None if v in (None, "") else float(v)

    colors = config.animal_color()
    animals = sorted({r["animal"] for r in rows})
    fig, axes = plt.subplots(2, len(EPOCHS), figsize=(3.4 * len(EPOCHS) + 0.6, 7.4),
                            squeeze=False, sharey="row")
    fig.subplots_adjust(top=0.78, bottom=0.10, hspace=0.42)
    for j, ep in enumerate(EPOCHS):
        top, bot = axes[0][j], axes[1][j]
        xs, seen = [], []
        for i, an in enumerate(animals):
            v = [r for r in rows if r["animal"] == an and r["epoch"] == ep]
            if not v:
                continue
            r = v[0]
            c = colors.get(an, "0.5")
            sh, p = _f(r["shift_frac_of_evoked"]), _f(r.get("p"))
            sig = p is not None and p < 0.05
            top.bar(i, sh, width=0.62, color=c, edgecolor="k" if sig else "none",
                    linewidth=1.6 if sig else 0, alpha=1.0 if sig else 0.55)
            # THE NULL, AS A MARKER ON THE BAR. Drawn per animal because it is per animal: it is
            # that animal's own pre-stroke session-to-session spread, not a shared threshold.
            if _f(r.get("null_p95")) is not None:
                top.plot([i - 0.38, i + 0.38], [_f(r["null_p95"])] * 2, "-", color="k", lw=1.4)
            if _f(r.get("null_median")) is not None:
                top.plot([i - 0.30, i + 0.30], [_f(r["null_median"])] * 2, ":", color="0.35",
                         lw=1.2)

            co, cp, cn = _f(r.get("cos_with_evoked")), _f(r.get("cos_p")), \
                _f(r.get("cos_null_median"))
            if co is not None:
                csig = cp is not None and cp < 0.05
                bot.bar(i, co, width=0.62, color=c, edgecolor="k" if csig else "none",
                        linewidth=1.8 if csig else 0, alpha=1.0 if csig else 0.45)
                if cn is not None:
                    bot.plot([i - 0.30, i + 0.30], [cn] * 2, ":", color="0.35", lw=1.2)
                # The DERIVED quantity -- shift x cos is the signed component of the drift along
                # the evoked pattern, i.e. the actual amplitude bias, and it is what the two rows
                # are for. Labelled rather than plotted: the CALIBRATED p belongs to the cosine.
                if sh is not None:
                    # PINNED TO THE FLOOR OF THE PANEL, not beside the bar. Placed next to the
                    # bar it landed on the dotted null marker for any cell whose cosine was near
                    # zero -- exactly the cells where the number matters most.
                    bot.text(i, -1.28, f"{sh * co:+.2f}", ha="center", va="bottom",
                             fontsize=7.5, color="0.25")
            xs.append(i)
            seen.append(an)
        for ax in (top, bot):
            ax.set_xticks(xs)
            ax.set_xticklabels(seen, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        top.set_title(ep, fontsize=12, fontweight="bold")
        bot.axhline(0, color="k", lw=0.8)
        bot.set_ylim(-1.32, 1.05)
        bot.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        if j == 0:
            top.set_ylabel("MOVEMENT\n|| shift ||, fraction of evoked signal", fontsize=9)
            bot.set_ylabel("ALIGNMENT\ncos(shift, evoked pattern)", fontsize=9)
    fig.text(0.5, 0.995, "epoch_15j -- does the REST BASELINE move across epochs, and does it matter?",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.955,
             "TOP -- DOES IT MOVE? Bar: || mean_epoch(b/k) - mean_pre(b/k) ||, each session's restw "
             "baseline divided by its OWN evoked norm k, which cancels cross-day multiplicative "
             "scaling.\nSOLID line = that animal's null 95th percentile, dotted = its null median: "
             "the spread of its own pre-stroke baselines. Outlined + opaque = p < 0.05. "
             "6 of 11 cells: THE BASELINE DOES MOVE.\n"
             "BOTTOM -- DOES IT MATTER? Only the component ALONG the evoked pattern biases a "
             "map - restw amplitude; an orthogonal shift costs ~r^2/2. Dotted = the cosine's OWN "
             "null median, which is ~0.2 and NOT 0, so never read this against zero.\n"
             "1 of 11 cells at p < 0.05 against ~0.6 expected: NO EVIDENCE OF ALIGNMENT, so the top "
             "row is an upper bound that is rarely attained. Held-out reference; false-positive "
             "rate 0.060 at alpha 0.05.\n"
             "Small number = shift x cos, the SIGNED bias. cos > 0 SHRINKS the measured amplitude, "
             "cos < 0 INFLATES it. The one significant cell (PS94 subacute, -0.91) is inflating.",
             ha="center", va="top", fontsize=8.2)
    out = out_dir / "epoch_15j_rest_baseline_epoch_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
