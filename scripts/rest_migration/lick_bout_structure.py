"""IS THE WITHIN-SESSION DECLINE FATIGUE AFTER ALL? Three tests the median ILI cannot do.

Priya, 2026-09-20: *"number of licks per trial decline could also be fatigue right? the ILI is more
gated by a CPG, so decreased effort might just manifest as decreasing licks per trial rather than
change in ILI"*.

**THAT IS CORRECT AND IT RETIRES AN OVERSTATEMENT.** `quit_prodrome` found the median inter-lick
interval flat across every session and I wrote "no within-session motor fatigue" into DECISIONS.
**ILI MEASURES THE PERIOD OF THE RHYTHM, NOT THE DURATION OF ITS EXPRESSION.** The licking CPG in
the brainstem reticular formation produces a stereotyped ~7 Hz cycle once engaged, and peripheral
muscular fatigue or a central effort signal can terminate BOUTS earlier without perturbing the
cycle period at all -- a runner who tires usually holds cadence and stops sooner. So flat ILI
licenses only "the rhythm generator's output rate is intact", not "no fatigue".

THE THREE TESTS, none of which the per-trial median can do:

    1  WITHIN-BOUT ILI SLOPE.  The MEDIAN is exactly the statistic that hides end-of-bout slowing,
       which is the characteristic signature of peripheral fatigue. Compare the FIRST THIRD of a
       bout's intervals against its LAST THIRD, and ask whether that slope steepens across the
       session. A tongue that tires should lengthen its last intervals first.

    2  LICKS OR TIME?  Does the decline track ACCUMULATED LICKS (effort) or ELAPSED TIME (a clock)?
       Within a session the two are collinear at r = 0.965-0.993, which is what killed the naive
       version -- so the instrument is BETWEEN-session variation in early lick rate, exactly as in
       the quit-point test. Measured at the CROSSING POINT: where licks/trial first falls below
       `CROSS_FRAC` of the session's own early level.

           fatigue (fixed lick budget)  ->  crossing LICKS flat (slope 0), TIME ~ 1/rate (-1)
           clock   (fixed time)         ->  crossing TIME  flat (slope 0), LICKS ~ rate   (+1)

    3  BOUTS PER TRIAL vs LICKS PER BOUT.  Only their PRODUCT has been measured so far. Fatigue
       should SHORTEN bouts; disengagement should reduce how many are INITIATED. The decomposition
       separates them and neither is visible in licks-per-trial alone.

**WHAT THE EXISTING EVIDENCE CAN AND CANNOT CARRY.** The quit-point regression (+0.72 [+0.36, +1.06]
for licks on rate, where a fixed lick budget requires exactly 0) argues against effort accumulation
-- but it is about what TERMINATES THE SESSION, not about what shortens bouts within it, and it
should not be made to carry both claims.

    python -m scripts.rest_migration.lick_bout_structure [--gate] [--horizon-min 90]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

#: THE BOOTSTRAP LIVES IN ONE PLACE NOW. `analysis_kit` holds the nested animals->sessions
#: draw this module used to define for itself, bit-for-bit -- `tests/test_analysis_kit.py`
#: pins it against the pre-extraction source. Read that module before touching a draw: the
#: point-estimate convention DIFFERS between `boot_ci` (flat pool, for LEVELS) and
#: `boot_delta` (animal-weighted, for CHANGES), and the difference has retracted a result.
from wfield_local import analysis_kit as ak

BOUT_GAP_S = 0.5            # longer than this starts a new bout, matching `quit_prodrome`
MIN_BOUT_LICKS = 6          # to split a bout into thirds and still have >= 2 intervals each side
CROSS_FRAC = 0.70           # the crossing point: licks/trial below this fraction of the early level
EARLY_FRAC = 0.20           # "early level" = the first this-much of the session, by trial order
NQ = 5
EPS = ("pre", "acute", "subacute", "chronic")


def _bouts(times_s):
    """Split one trial's lick times (seconds) into bouts on gaps longer than `BOUT_GAP_S`."""
    if times_s.size == 0:
        return []
    cut = np.flatnonzero(np.diff(times_s) > BOUT_GAP_S) + 1
    return np.split(times_s, cut)


def session_bouts(item):
    """Per-trial bout structure for one session, or ``None``.

    MODULE-LEVEL for `parallel.fan_out` (ground rule 6); options travel in the item because spawn
    re-imports this module and never sees a runtime global.
    """
    lab, gate, horizon_min = item
    from wfield_local import config, epochs

    # SAME LOADER AS `quit_prodrome.session_row` -- this block was character-identical in both
    # modules, down to the 60-trial and 30-gated-trial floors. See `analysis_kit.session_behavior`.
    bh = ak.session_behavior(lab, gate=gate, horizon_min=horizon_min)
    if bh is None:
        return None
    tt = bh.gated
    cs, lick_s, sr, near = bh.cue_samples, bh.lick_samples, bh.sample_rate, bh.near

    rows, n = [], max(len(tt) - 1, 1)
    for i, r in enumerate(tt):
        k = int(r["order"])
        if k + 1 >= cs.size:
            continue
        seg = lick_s[(lick_s >= cs[k]) & (lick_s < cs[k + 1])]
        if seg.size == 0:
            continue
        ts = seg / sr
        bl = _bouts(ts)
        first, last = [], []
        for b in bl:
            if b.size < MIN_BOUT_LICKS:
                continue
            iv = np.diff(b)
            third = max(2, iv.size // 3)
            first.append(float(np.median(iv[:third])))
            last.append(float(np.median(iv[-third:])))
        rows.append(dict(
            frac=i / n, elapsed_s=float(r["elapsed_s"]),
            grp="near" if int(r["pos"]) in near else "far",
            n_licks=int(seg.size), n_bouts=len(bl),
            licks_per_bout=float(seg.size / max(len(bl), 1)),
            ili_first=(float(np.mean(first)) * 1000.0 if first else float("nan")),
            ili_last=(float(np.mean(last)) * 1000.0 if last else float("nan"))))
    if len(rows) < 30:
        return None

    # CUMULATIVE LICKS over the RETAINED trials, so it matches the window everything else uses.
    cum = np.cumsum([r["n_licks"] for r in rows])
    for j, r in enumerate(rows):
        r["cum_licks"] = float(cum[j])

    # THE CROSSING POINT -- where licks/trial first falls below CROSS_FRAC of the early level.
    # Reported in BOTH currencies so the regression can ask which one is conserved.
    early = [r["n_licks"] for r in rows if r["frac"] < EARLY_FRAC]
    base = float(np.mean(early)) if early else float("nan")
    cross_t = cross_l = float("nan")
    if np.isfinite(base) and base > 0:
        w = max(5, len(rows) // 20)
        sm = np.convolve([r["n_licks"] for r in rows], np.ones(w) / w, mode="same")
        hit = np.flatnonzero((sm < CROSS_FRAC * base)
                             & (np.arange(len(rows)) > len(rows) * EARLY_FRAC))
        if hit.size:
            cross_t = rows[int(hit[0])]["elapsed_s"]
            cross_l = rows[int(hit[0])]["cum_licks"]
    t0 = rows[0]["elapsed_s"]
    early_lpm = (float(np.sum([r["n_licks"] for r in rows if r["elapsed_s"] - t0 <= 600.0]))
                 / 10.0)
    return dict(label=lab, animal=config.animal_of(lab), epoch=epochs.epoch_of(lab),
                rows=rows, early_lpm=early_lpm, base_licks=base,
                cross_time_s=cross_t, cross_licks=cross_l)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--horizon-min", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    # `analysis_kit.curated_sessions` is this filter, once. IT PRESERVES `load_sessions`
    # ORDER on purpose -- that list is NOT sorted, and the pools below are iterated into a
    # seeded RNG, so quietly sorting here would move published CIs.
    labels = ak.curated_labels(a.animals)
    items = [(lab, a.gate, a.horizon_min) for lab in labels]
    # `fan_sessions` IS `fan_out` PLUS THE SORT -- see `analysis_kit`. `fan_out`
    # returns COMPLETION order; every bootstrap pool below is built by iterating this
    # collection, and a seeded RNG over a differently-ordered list gives different
    # draws (measured on `quit_prodrome`: the CI moved from [-22.9,-13.7] to
    # [-23.1,-13.6] while the point estimate stayed exact). No longer forgettable.
    # Sorting by the ITEM is sorting by the label, since the label is the tuple's first element.
    res, fail = ak.fan_sessions(items, session_bouts, jobs=a.jobs)
    sess = {r["label"]: r for _it, r in res if r is not None}
    if fail:
        print(f"  !! {len(fail)} failed: " + ", ".join(f"{x[0]}" for x in fail[:4]), flush=True)
    if not sess:
        print("no sessions -- a failed run, not a result")
        return 1
    print(f"  {len(sess)} sessions")

    rng = np.random.default_rng(a.seed)
    bar = "=" * 96
    POST = [e for e in EPS if e != "pre"]

    def quint(key, grp, b, e, an_only=True):
        d = defaultdict(list)
        for v in sess.values():
            if v["epoch"] != e:
                continue
            vals = [r[key] for r in v["rows"]
                    if r["grp"] == grp and b / NQ <= r["frac"] < (b + 1) / NQ
                    and np.isfinite(r[key])]
            if len(vals) >= 5:
                d[v["animal"]].append(float(np.mean(vals)))
        return d

    def per_session_q(key, grp, b, e):
        """``{animal: [per-session mean of `key` in quintile `b`]}`` for epoch `e`."""
        d = defaultdict(list)
        for v in sess.values():
            if v["epoch"] != e:
                continue
            vals = [r[key] for r in v["rows"]
                    if r["grp"] == grp and b / NQ <= r["frac"] < (b + 1) / NQ
                    and np.isfinite(r[key])]
            if len(vals) >= 5:
                d[v["animal"]].append(float(np.mean(vals)))
        return d

    def delta_pairs(key, grp, b, e):
        """``{animal: (epoch values, that animal's own pre values)}`` for one quintile."""
        ep, pr = per_session_q(key, grp, b, e), per_session_q(key, grp, 0 + b, "pre")
        return {an: (ep[an], pr[an]) for an in sorted(set(ep) & set(pr))}

    def gap_pairs(key, grp, e):
        """``{animal: (epoch Q5-Q1 per session, pre Q5-Q1 per session)}``.

        **Q1 AND Q5 COME FROM THE SAME SESSION**, so the within-session pairing survives; taking
        the difference of two independently-averaged endpoints would silently mix session sets,
        since a session can clear the 5-trial floor in one quintile and not the other.
        """
        out = {}
        for tag in (e, "pre"):
            d = defaultdict(list)
            for v in sess.values():
                if v["epoch"] != tag:
                    continue
                q = []
                for b in (0, NQ - 1):
                    vals = [r[key] for r in v["rows"]
                            if r["grp"] == grp and b / NQ <= r["frac"] < (b + 1) / NQ
                            and np.isfinite(r[key])]
                    q.append(float(np.mean(vals)) if len(vals) >= 5 else None)
                if q[0] is not None and q[1] is not None:
                    d[v["animal"]].append(q[1] - q[0])
            out[tag] = d
        return {an: (out[e][an], out["pre"][an])
                for an in sorted(set(out[e]) & set(out["pre"]))}

    # ---- TEST 3 first, because it frames the other two --------------------------------------
    print(f"\n{bar}\nTEST 3 -- BOUTS PER TRIAL vs LICKS PER BOUT, by quintile\n{bar}")
    print("  fatigue should SHORTEN bouts; disengagement should reduce how many are INITIATED.")
    for key, nm in (("n_bouts", "bouts per trial"), ("licks_per_bout", "licks per bout")):
        print(f"\n  {nm}")
        print(f"    {'epoch':<10}{'spouts':<7}" + "".join(f"{'Q' + str(i + 1):>9}" for i in
                                                          range(NQ)) + f"{'Q5-Q1 vs pre':>22}")
        for e in EPS:
            for grp in ("near", "far"):
                line = f"    {e:<10}{grp:<7}"
                for b in range(NQ):
                    g = ak.boot_ci(quint(key, grp, b, e), rng)
                    line += f"{g[0]:>9.2f}" if g else f"{'--':>9}"
                if e != "pre":
                    g = ak.boot_delta_pairs(gap_pairs(key, grp, e), rng)
                    if g:
                        star = " *" if (g[1] > 0 or g[2] < 0) else "  "
                        line += f"{g[0]:>+10.2f} [{g[1]:+.2f},{g[2]:+.2f}]{star}"
                print(line)

    # ---- TEST 1 -------------------------------------------------------------------------------
    print(f"\n{bar}\nTEST 1 -- WITHIN-BOUT ILI SLOPE (last third minus first third, ms)\n{bar}")
    print("  the per-trial MEDIAN hides end-of-bout slowing, which is what a tiring tongue does.")
    print(f"    {'epoch':<10}{'spouts':<7}" + "".join(f"{'Q' + str(i + 1):>10}" for i in range(NQ)))
    for v in sess.values():
        for r in v["rows"]:
            r["ili_slope"] = r["ili_last"] - r["ili_first"]
    for e in EPS:
        for grp in ("near", "far"):
            line = f"    {e:<10}{grp:<7}"
            for b in range(NQ):
                g = ak.boot_ci(quint("ili_slope", grp, b, e), rng)
                line += f"{g[0]:>10.1f}" if g else f"{'--':>10}"
            print(line)
    print("\n  POSITIVE and GROWING across quintiles = the tongue slows at the end of a bout, and")
    print("  more so as the session wears on. That is peripheral fatigue and the median misses it.")
    print("\n  WITHIN-ANIMAL DELTA FROM PRE (ms), and the Q5-Q1 gap change -- the LEVELS above are")
    print("  cohort means over different animal sets and are not comparable across epochs.")
    print(f"    {'epoch':<10}{'spouts':<7}"
          + "".join(f"{'Q' + str(i + 1):>17}" for i in range(NQ)) + f"{'Q5-Q1 vs pre':>22}")
    for e in POST:
        for grp in ("near", "far"):
            line = f"    {e:<10}{grp:<7}"
            for b in range(NQ):
                g = ak.boot_delta_pairs(delta_pairs("ili_slope", grp, b, e), rng)
                if g:
                    star = "*" if (g[1] > 0 or g[2] < 0) else " "
                    line += f"{g[0]:>+12.1f}{star}({g[3]})"
                else:
                    line += f"{'--':>17}"
            g = ak.boot_delta_pairs(gap_pairs("ili_slope", grp, e), rng)
            if g:
                star = " *" if (g[1] > 0 or g[2] < 0) else "  "
                line += f"{g[0]:>+10.1f} [{g[1]:+.1f},{g[2]:+.1f}]{star}"
            print(line)

    # ---- TEST 2 -------------------------------------------------------------------------------
    print(f"\n{bar}\nTEST 2 -- IS THE DECLINE PACED BY LICKS OR BY TIME?\n{bar}")
    ok = [v for v in sess.values()
          if np.isfinite(v["cross_time_s"]) and np.isfinite(v["cross_licks"])
          and v["cross_licks"] > 0 and v["early_lpm"] > 0]
    print(f"  {len(ok)} of {len(sess)} sessions reach the {CROSS_FRAC:.0%} crossing point")
    if len(ok) >= 12:
        R = np.log(np.array([v["early_lpm"] for v in ok]))
        T = np.log(np.array([v["cross_time_s"] for v in ok]))
        L = np.log(np.array([v["cross_licks"] for v in ok]))
        an = np.array([v["animal"] for v in ok])

        def dm(x):
            o = np.asarray(x, float).copy()
            for k in set(an):
                m = an == k
                o[m] -= o[m].mean()
            return o

        def slope(Y):
            b = float(np.polyfit(dm(R), dm(Y), 1)[0])
            o = []
            A = sorted(set(an))
            for _ in range(2000):
                idx = np.concatenate([np.flatnonzero(an == x)[
                    rng.integers(0, int((an == x).sum()), int((an == x).sum()))] for x in A])
                try:
                    rr, yy = R[idx], Y[idx]
                    oo = np.asarray(yy, float).copy()
                    rr2 = np.asarray(rr, float).copy()
                    for k in set(an[idx]):
                        m = an[idx] == k
                        oo[m] -= oo[m].mean()
                        rr2[m] -= rr2[m].mean()
                    o.append(float(np.polyfit(rr2, oo, 1)[0]))
                except Exception:                                    # noqa: BLE001
                    pass
            o = np.asarray(o)
            return b, float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))

        lam = slope(L - T)
        print(f"  slope log(overall rate) on log(early rate) = {lam[0]:+.2f}  <- attenuation lambda")
        for nm, Y, fat, clk in (("crossing TIME ", T, -abs(lam[0]), 0.0),
                                ("crossing LICKS", L, 0.0, abs(lam[0]))):
            b, lo, hi = slope(Y)
            print(f"  {nm} on log(early rate): {b:+.2f} [{lo:+.2f}, {hi:+.2f}]   "
                  f"fatigue predicts {fat:+.2f}, clock predicts {clk:+.2f}")
        print("\n  A FIXED LICK BUDGET REQUIRES THE LICKS SLOPE TO BE EXACTLY ZERO, and measurement")
        print("  noise in the predictor can only shrink slopes TOWARD zero, never away from it.")
    else:
        print("  too few sessions reach the crossing -- not run")

    # ---- FIGURES: three views of one per-trial measure ----------------------------------------
    # RAW / DELTA-FROM-PRE / GAP-CHANGE, because each answers a different question and the raw
    # panel alone misleads: the levels are cohort means over DIFFERENT ANIMAL SETS, and chronic
    # near sits at 3.36 bouts against far's 8.41 where no other epoch splits that way -- animal
    # composition, not biology. The delta panels remove it by construction.
    #
    # FACTORED rather than written twice: the within-bout deceleration was table-only in the first
    # version, which left the headline finding of this module with no figure at all.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def three_view(key, ylab, fname, title):
        fig, ax = plt.subplots(2, 4, figsize=(18.5, 8.4))
        for j, e in enumerate(EPS):
            axx = ax[0][j]
            for grp, cc in (("near", "#2ca02c"), ("far", "#9467bd")):
                xs, ys, lo_, hi_ = [], [], [], []
                for b in range(NQ):
                    g = ak.boot_ci(per_session_q(key, grp, b, e), rng)
                    if g:
                        xs.append(b + 1)
                        ys.append(g[0])
                        lo_.append(g[1])
                        hi_.append(g[2])
                if len(xs) >= 3:
                    axx.plot(xs, ys, "-o", ms=5, lw=1.8, color=cc, label=grp)
                    axx.fill_between(xs, lo_, hi_, color=cc, alpha=0.15, lw=0)
            axx.set_xticks(range(1, NQ + 1))
            axx.set_xticklabels([f"Q{i}" for i in range(1, NQ + 1)])
            axx.set_title(f"{e} -- RAW", fontsize=10)
            if j == 0:
                axx.set_ylabel(ylab)
                axx.legend(fontsize=8, frameon=False)
        for j, e in enumerate(POST):
            axx = ax[1][j]
            for grp, cc in (("near", "#2ca02c"), ("far", "#9467bd")):
                xs, ys, lo_, hi_ = [], [], [], []
                for b in range(NQ):
                    g = ak.boot_delta_pairs(delta_pairs(key, grp, b, e), rng)
                    if g:
                        xs.append(b + 1)
                        ys.append(g[0])
                        lo_.append(g[1])
                        hi_.append(g[2])
                if len(xs) >= 3:
                    axx.plot(xs, ys, "-o", ms=5, lw=1.8, color=cc, label=grp)
                    axx.fill_between(xs, lo_, hi_, color=cc, alpha=0.15, lw=0)
            axx.axhline(0, color="0.35", lw=1.0, ls="--")
            axx.set_xticks(range(1, NQ + 1))
            axx.set_xticklabels([f"Q{i}" for i in range(1, NQ + 1)])
            axx.set_title(f"{e} - pre (within animal)", fontsize=10)
            if j == 0:
                axx.set_ylabel(f"delta {ylab}")
        gx = ax[1][3]
        for grp, cc, off in (("near", "#2ca02c", -0.09), ("far", "#9467bd", +0.09)):
            xs, ys, lo_, hi_ = [], [], [], []
            for j, e in enumerate(POST):
                g = ak.boot_delta_pairs(gap_pairs(key, grp, e), rng)
                if g:
                    xs.append(j + off)
                    ys.append(g[0])
                    lo_.append(g[1])
                    hi_.append(g[2])
            if xs:
                gx.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo_),
                                          np.array(hi_) - np.array(ys)],
                            fmt="o", ms=6, lw=1.8, capsize=4, color=cc, label=grp)
        gx.axhline(0, color="0.35", lw=1.0, ls="--")
        gx.set_xticks(range(len(POST)))
        gx.set_xticklabels(POST, fontsize=8)
        gx.set_xlim(-0.5, len(POST) - 0.5)
        gx.set_title("change in the Q5-Q1 GAP", fontsize=10)
        gx.set_ylabel("(Q5-Q1) minus pre's (Q5-Q1)", fontsize=8)
        gx.legend(fontsize=8, frameon=False)
        for row, cols in ((ax[0], range(4)), (ax[1], range(3))):
            used = [row[i] for i in cols if row[i].has_data()]
            if len(used) > 1:
                lo = min(x.get_ylim()[0] for x in used)
                hi = max(x.get_ylim()[1] for x in used)
                for x in used:
                    x.set_ylim(lo, hi)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        pf = out_dir / fname
        fig.savefig(pf, dpi=170)
        plt.close(fig)
        print(f"  wrote {pf}")

    three_view("n_bouts", "bouts per trial", "epoch_26_bouts_per_trial.png",
               "BOUTS PER TRIAL across session quintiles. TOP raw (cohort means over different "
               "animal sets -- read with care).\nBOTTOM within-animal delta from pre, and the "
               "change in the Q5-Q1 gap. Y shared within each row; the gap panel has its own.")
    three_view("licks_per_bout", "licks per bout", "epoch_26_licks_per_bout.png",
               "LICKS PER BOUT across session quintiles -- does a bout get CUT SHORT as the "
               "session wears on?\nTOP raw, BOTTOM within-animal delta from pre and the "
               "Q5-Q1 gap change.")
    three_view("ili_slope", "within-bout deceleration (ms)",
               "epoch_26_within_bout_deceleration.png",
               "WITHIN-BOUT DECELERATION: the LAST THIRD of a bout's intervals minus its FIRST "
               "THIRD.\nThe per-trial MEDIAN ILI averages this away, which is why it looked "
               "like there was no motor change. POSITIVE = the tongue slows as the bout runs on.")

    q = out_dir / "epoch_26_lick_bout_structure.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "animal", "epoch", "early_lpm", "base_licks",
                                           "cross_time_s", "cross_licks", "n_trials"])
        w.writeheader()
        for lab, v in sess.items():
            w.writerow(dict(label=lab, animal=v["animal"], epoch=v["epoch"],
                            early_lpm=round(v["early_lpm"], 2),
                            base_licks=round(v["base_licks"], 2),
                            cross_time_s=v["cross_time_s"], cross_licks=v["cross_licks"],
                            n_trials=len(v["rows"])))
    print(f"\n  wrote {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
