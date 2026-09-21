"""WHY does engagement fall after the stroke? Behaviour only -- no imaging.

Priya, 2026-09-19: *"can we brainstorm how to think about the change in engagement post-stroke? it
could be framed as the animal being tired, feeling unwell or uninterested, or orofacial fatigue due
to the stroke"*.

THOSE THREE SIT ON TWO SEPARABLE AXES, and this module measures both.

    UNIFORM vs POSITION-SPECIFIC   malaise, satiety and general fatigue hit every spout equally.
                                   An orofacial deficit, learned avoidance and neglect do not.

    WON'T vs CAN'T                 **the sharper one.** `n_licks_pre` is ANTICIPATORY licking in
                                   the ENL window [trial_start, cue). An animal that is motivated
                                   but unable keeps anticipating and then fails; an animal that is
                                   unmotivated stops doing both.

**THE ANTICIPATORY MEASURE IS POSITION-RESOLVED, and that is not obvious.** The task controller
calls `emitPositionCode(currentTrialPos)` and pulses the strobe BEFORE the cue
(`Behavior_MobileSpouts_Zaber_Arduino_v36.ino`), so the spout is already at its position during the
pre-cue window. Anticipatory licking therefore reports willingness to try THAT spout, not a
position-blind arousal level -- which is what makes the won't/can't split work per position.

THE HEADLINE STATISTIC IS ANTICIPATION ON TRIALS THAT WENT ON TO MISS. High = the animal wanted the
water and did not get it (motor, or a reach that never made contact). Low = it was not trying.

WHAT THIS CANNOT DO, AND IT IS THE CRUX FOR THE OROFACIAL READING. The lick detector is CONTACT
based, so a reach that misses the spout registers as NOTHING -- "did not try" and "tried and missed"
are the same observation. The repo already records this failure for a specific animal
(`nolick_decoder`: *"PS93 reaches far_L poorly... licks occur but frequently do not reach contact,
so a short lick registers as nothing"*). Separating them needs the DLC tongue tracking that is still
pending; until then a low anticipation rate is NOT evidence of low motivation.

AND NOTE WHAT THE DECK'S ENGAGEMENT GATE ALREADY ASSUMES. `precue_engagement_states.engagement_gate`
requires a NON-RECOVERING TERMINAL COLLAPSE -- it is built to catch satiety-like quitting. A
position-specific effort cost would not trip it at all; it would show as an elevated miss rate at
particular spouts THROUGHOUT the session. So "engagement fell" as the deck measures it is already
committed to the global reading, and is a different quantity from "the animal attempted fewer
far-contralateral spouts".

**NO ENGAGEMENT GATE, AND THAT IS DELIBERATE -- DO NOT ADD ONE FOR CONSISTENCY.** Every other
module today gates the terminal quit period out, because there an epoch contrast cannot tolerate
composition tracking the independent variable. Here the quit period IS THE DEPENDENT VARIABLE.
Gating it would remove the phenomenon and leave a tautology: the animal is engaged on the trials
selected for being engaged. Every scorable trial enters, in order, whole session.

For the same reason there is no in-trial lick mask: a lick that arrives outside the response window
is informative about WHEN the animal tried, which is part of the question rather than contamination.

    python -m scripts.rest_migration.engagement_decomposition [--animals PS92 ...]
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

DEFAULT_RESP_S = 2.0


def session_trials(s, resp_s):
    """Per-trial rows for ONE session: position, hit, latency, anticipatory licks, trial order.

    Built here rather than via `daq_trials.build_trials` for one reason: positions come from
    `classify_cues_with_backup`, the REPAIRED classifier. The raw strobe collapses 6 positions onto
    4 on the August sessions with a dead `spout_bit1`, and those are pre-stroke days -- scoring a
    position-specific question on merged labels would answer a different question.
    """
    from wfield_local import daq_io
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    from wfield_local.rest_by_position import _session_daq

    cue = _load_cue_events(s["h5"])
    cs = np.asarray(cue["cue_samples"], float)
    codes = np.asarray(classify_cues_with_backup(s, cue, verbose=False))
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    licks = np.sort(np.asarray(lk["lick_samples"], float))
    with daq_io.open_daq(s["h5"]) as f:
        sr, _ = daq_io.session_attrs(f)
    _r, _cs2, _c, ts, _fs, _sy = _session_daq(s)
    ts = np.sort(np.asarray(ts, float))

    # THE ENL WINDOW OPENS AT THE LAST trial_start AT OR BEFORE THE CUE. Falling back to the cue
    # itself when there is none gives an EMPTY pre window, which scores as "no anticipation" -- so
    # those trials are dropped rather than counted as unmotivated.
    j = np.searchsorted(ts, cs, side="right") - 1
    pre_open = np.where(j >= 0, ts[np.clip(j, 0, None)], np.nan)

    nxt = np.append(cs[1:], np.inf)
    n = cs.size
    rows = []
    for k in range(n):
        if codes[k] < 0 or not np.isfinite(pre_open[k]) or pre_open[k] >= cs[k]:
            continue
        end = min(cs[k] + resp_s * sr, nxt[k])
        post = licks[(licks >= cs[k]) & (licks <= end)]
        npre = int(((licks >= pre_open[k]) & (licks < cs[k])).sum())
        rows.append(dict(pos=int(codes[k]), order=k, frac=k / max(n - 1, 1),
                         hit=int(post.size > 0), n_licks_pre=npre,
                         anticipated=int(npre > 0),
                         # EFFORT AND TIME, SEPARATELY. See `licks_vs_time`.
                         cum_licks=float(np.searchsorted(licks, cs[k])),
                         elapsed_s=float(cs[k] - cs[0]) / sr,
                         latency_s=(float(post[0] - cs[k]) / sr if post.size else np.nan)))
    return rows


def near_codes():
    """Position codes for the NEAR spouts -- derived from the names, not written down as {0,1,2}."""
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    return {c for c, nm in POSITION_NAMES.items() if nm.startswith("close")}


def licks_vs_time(rows, only=None):
    """``(partial r with LICKS, partial r with TIME, collinearity)`` for one session's decline.

    **RUN THIS ON THE NEAR SPOUTS (Priya, 2026-09-19: *"should we drop the affected spout
    locations? ie maybe just look at near"*), and the reason is not only noise.**

    THE NEAR SPOUTS ARE THE INTERNAL CONTROL. The animal demonstrably CAN reach them -- PS94 acute
    near-ipsi latency is 0.152 s against 0.136 s pre-stroke, essentially normal -- so any decline
    there is a change of STATE, not of CAPACITY. That is exactly the global component this test is
    trying to characterise, measured where capacity is not the limiting factor.

    INCLUDING THE FAR SPOUTS BREAKS IT THREE WAYS: their acute hit rate sits on the floor (PS94
    far-contra 0.017), so there is no variance for a correlation to work on; a floored series cannot
    show a within-session decline whatever the mechanism; and pooling them mixes the POSITION-
    SPECIFIC deficit into a question about GLOBAL state, which is the one confusion this whole
    decomposition exists to avoid.

    Priya, 2026-09-19: *"we could also think about looking at decline with number of licks vs
    decline with time - motor fatigue should be more related to the former right?"* Yes, and it is
    the one dissociation on the table that does not route through a lick-vs-non-lick distinction
    the contact detector cannot make:

        MOTOR FATIGUE     scales with EFFORT EXPENDED   -> cumulative licks
        MALAISE / BOREDOM scales with TIME ON TASK      -> elapsed seconds

    Both are cumulative and therefore collinear, so the SINGLE correlation with either is
    uninformative -- a partial correlation is the whole design. **THE COLLINEARITY IS RETURNED WITH
    THE RESULT** because it bounds what the test can say: if the animal licks at a perfectly
    constant rate the two predictors are identical and neither partial is estimable. What gives the
    test power is that lick rate is BURSTY -- bouts and pauses -- so cumulative licks runs ahead of
    and behind the clock within a session.

    Same shape as the 415-drift test earlier the same day (r(ratio, lick | time) against
    r(ratio, time | lick)), and read the same way: whichever predictor survives the other's removal
    is the one doing the work.
    """
    if only is not None:
        rows = [r for r in rows if r["pos"] in only]
    if len(rows) < 40:
        return None
    y = np.array([r["hit"] for r in rows], float)
    L = np.array([r["cum_licks"] for r in rows], float)
    T = np.array([r["elapsed_s"] for r in rows], float)
    if y.std() == 0 or L.std() == 0 or T.std() == 0:
        return None

    def resid(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    collin = float(np.corrcoef(L, T)[0, 1])
    if abs(collin) > 0.9995:                 # identical predictors -> nothing to partial out
        return None
    r_lick = float(np.corrcoef(resid(y, T), resid(L, T))[0, 1])
    r_time = float(np.corrcoef(resid(y, L), resid(T, L))[0, 1])
    return r_lick, r_time, collin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--resp-s", type=float, default=DEFAULT_RESP_S)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import config, epoch_figures as ef, epochs
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.paths import PathResolver
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    animals = a.animals
    raw = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    lab_of = dict(zip(DISPLAY_ORDER,
                      [x.title() for x in ef.anatomical_labels(raw, short=False)]
                      if set(raw) <= set(CONF_LABELS) else raw))

    per = []                     # one row per session x position
    # `analysis_kit.curated_sessions` is this filter, once. IT PRESERVES `load_sessions`
    # ORDER on purpose -- that list is NOT sorted, and the pools below are iterated into a
    # seeded RNG, so quietly sorting here would move published CIs.
    for s in ak.curated_sessions(animals):
        lab = s["label"]
        ep = epochs.epoch_of(lab)
        try:
            rows = session_trials(s, a.resp_s)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if len(rows) < 30:
            print(f"  .. {lab}: {len(rows)} scorable trials -- skipped", flush=True)
            continue
        lvt = licks_vs_time(rows, only=near_codes())        # NEAR ONLY -- see `licks_vs_time`
        lvt_all = licks_vs_time(rows)
        for c in DISPLAY_ORDER:
            v = [r for r in rows if r["pos"] == c]
            if len(v) < 10:
                continue
            miss = [r for r in v if not r["hit"]]
            # FIVE BINS, NOT THREE (Priya, 2026-09-19). Thirds give one number -- a first-to-last
            # difference -- which cannot tell a STEADY decline from a LATE COLLAPSE, and those are
            # different hypotheses: effort accumulating smoothly against an animal that works
            # normally and then quits. Five quintiles show the shape, and the endpoints still give
            # the same difference, so nothing is lost.
            q = [[r for r in v if i / 5 <= r["frac"] < (i + 1) / 5 or
                  (i == 4 and r["frac"] >= 1.0)] for i in range(5)]
            qhit = [float(np.mean([r["hit"] for r in b])) if b else float("nan") for b in q]
            early, late = q[0], q[4]
            per.append(dict(
                label=lab, animal=config.animal_of(lab), epoch=ep, position=lab_of[c],
                n_trials=len(v),
                hit_rate=float(np.mean([r["hit"] for r in v])),
                # THE HEADLINE: did the animal anticipate on trials it then MISSED?
                antic_on_miss=(float(np.mean([r["anticipated"] for r in miss]))
                               if miss else float("nan")),
                antic_rate=float(np.mean([r["anticipated"] for r in v])),
                mean_licks_pre=float(np.mean([r["n_licks_pre"] for r in v])),
                median_latency_s=float(np.nanmedian([r["latency_s"] for r in v])),
                hit_early=(float(np.mean([r["hit"] for r in early])) if early else float("nan")),
                hit_late=(float(np.mean([r["hit"] for r in late])) if late else float("nan")),
                # FLOOR-GUARDED: a cell whose EARLY hit rate is already near zero cannot drop, so
                # its within_drop reads ~0 and looks like preservation. PS94 acute far-contra:
                # hit 0.017, within_drop 0.015 -- not "spared", unmeasurable.
                within_drop_valid=bool(early and late
                                       and float(np.mean([r["hit"] for r in early])) >= 0.15),
                within_drop=((float(np.mean([r["hit"] for r in early]))
                              - float(np.mean([r["hit"] for r in late])))
                             if early and late else float("nan")),
                q1=qhit[0], q2=qhit[1], q3=qhit[2], q4=qhit[3], q5=qhit[4],
                n_miss=len(miss),
                r_hit_licks=(lvt[0] if lvt else float("nan")),
                r_hit_time=(lvt[1] if lvt else float("nan")),
                collinearity=(lvt[2] if lvt else float("nan")),
                r_hit_licks_allpos=(lvt_all[0] if lvt_all else float("nan")),
                r_hit_time_allpos=(lvt_all[1] if lvt_all else float("nan"))))
        print(f"   {lab:14s} {ep:9s} {len(rows)} trials", flush=True)

    if not per:
        print("no sessions -- a failed run, not a result")
        return 1
    q = out_dir / "epoch_17_engagement_decomposition.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    print(f"\nwrote {q}")

    rng = np.random.default_rng(a.seed)
    eps = [e for e in ("pre", "acute", "subacute", "chronic")
           if any(x["epoch"] == e for x in per)]
    poss = [lab_of[c] for c in DISPLAY_ORDER
            if any(x["position"] == lab_of[c] for x in per)]

    def cell(key, ep, pos=None):
        d = defaultdict(list)
        for x in per:
            if x["epoch"] != ep or (pos is not None and x["position"] != pos):
                continue
            if np.isfinite(x[key]):
                d[x["animal"]].append(float(x[key]))
        return ak.boot_ci(d, rng)

    bar = "=" * 92
    for key, title, why in (
            ("hit_rate", "HIT RATE", "the thing to be explained"),
            ("antic_on_miss", "ANTICIPATION ON MISSED TRIALS",
             "HIGH = wanted it and could not get it. LOW = was not trying."),
            ("within_drop", "WITHIN-SESSION DROP (early third - late third hit rate)",
             "builds with effort -> fatigue. flat -> malaise. uniform -> global."),
            ("median_latency_s", "MEDIAN FIRST-LICK LATENCY", "vigor")):
        print(f"\n{bar}\n{title}  --  {why}\n{bar}")
        print(f"  {'epoch':<10}{'ALL POSITIONS':>22}   " + "".join(f"{p[:11]:>13}" for p in poss))
        for e in eps:
            g = cell(key, e)
            line = f"  {e:<10}" + (f"{g[0]:>10.3f} [{g[1]:.2f},{g[2]:.2f}]" if g else f"{'--':>22}")
            line += "   "
            for p in poss:
                c = cell(key, e, p)
                line += f"{c[0]:>13.3f}" if c else f"{'--':>13}"
            print(line)

    # THE SHAPE OF THE WITHIN-SESSION DECLINE, NEAR SPOUTS ONLY -- five bins rather than a single
    # first-minus-last difference, which cannot tell a STEADY decline from a LATE COLLAPSE.
    near_lab = {lab_of[c] for c in near_codes() if c in lab_of}
    print(f"\n{bar}\nWITHIN-SESSION SHAPE, NEAR SPOUTS ONLY (hit rate by session quintile)\n{bar}")
    print(f"  {'epoch':<10}" + "".join(f"{'Q' + str(i + 1):>10}" for i in range(5))
          + f"{'Q1-Q5':>10}")
    for e in eps:
        line, vals = f"  {e:<10}", []
        for k in ("q1", "q2", "q3", "q4", "q5"):
            x = [float(r[k]) for r in per
                 if r["epoch"] == e and r["position"] in near_lab and np.isfinite(float(r[k]))]
            vals.append(float(np.mean(x)) if x else float("nan"))
            line += f"{vals[-1]:>10.3f}" if x else f"{'--':>10}"
        line += (f"{vals[0] - vals[-1]:>10.3f}"
                 if np.isfinite(vals[0] + vals[-1]) else f"{'--':>10}")
        print(line)
    print("\n  STEADY slope -> effort or time accumulating. LATE COLLAPSE -> the animal quits.")
    print("  NEAR spouts only: a floored far-spout series cannot show a decline at all.")

    # LICKS vs TIME -- the dissociation that avoids the contact-detector problem entirely.
    print(f"\n{bar}\nDOES THE DECLINE TRACK EFFORT (licks) OR THE CLOCK (time)?\n{bar}")
    print(f"  {'epoch':<10}{'partial r(hit, LICKS | time)':>32}"
          f"{'partial r(hit, TIME | licks)':>32}{'collinearity':>15}")
    for e in eps:
        g1, g2, cc = cell("r_hit_licks", e), cell("r_hit_time", e), cell("collinearity", e)
        line = f"  {e:<10}"
        for g in (g1, g2):
            line += f"{g[0]:>18.3f} [{g[1]:+.2f},{g[2]:+.2f}]" if g else f"{'--':>32}"
        line += f"{cc[0]:>15.3f}" if cc else f"{'--':>15}"
        print(line)
    print("\n  LICKS survives, TIME does not  -> MOTOR FATIGUE (decline tracks effort expended)")
    print("  TIME survives, LICKS does not  -> malaise / boredom / time on task")
    print("  NEITHER survives               -> collinearity too high to separate. Read that column:")
    print("      both predictors are cumulative, so what gives the test power is BURSTY licking,")
    print("      which lets cumulative licks run ahead of and behind the clock within a session.")
    print("  The SINGLE correlation with either predictor is meaningless here; only the partials.")

    print(f"\n{bar}\nHOW TO READ IT\n{bar}")
    print("  UNIFORM fall across positions           -> malaise / satiety / general fatigue")
    print("  POSITION-SPECIFIC fall                  -> orofacial deficit, avoidance, or neglect")
    print("  ANTICIPATION ON MISSES STAYS HIGH       -> motivated but unable (CAN'T)")
    print("  ANTICIPATION FALLS WITH HIT RATE        -> not trying (WON'T)")
    print("\n  AND THE CAVEAT THAT LIMITS ALL OF IT: lick detection is CONTACT-based, so a reach")
    print("  that misses registers as nothing. A LOW anticipation rate is therefore NOT evidence")
    print("  of low motivation -- it is equally consistent with reaching and missing. Separating")
    print("  them needs the DLC tongue tracking that is still pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
