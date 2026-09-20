"""LICK RATE ACROSS THE SESSION, and whether a quit has a PRODROME.

Priya, 2026-09-19: *"if we plot lick rate over the course of the session, is it slower before a
quit than at the end of sessions without quitting (or at the same time in other sessions that
hadn't quit by that time)"*, then *"can you plot the lick rate vs session time to see if there is
a drop off?"*.

**THE TWO PLOTS ANSWER DIFFERENT QUESTIONS AND THE SECOND ONE IS THE HONEST ONE.**

Panel A is the literal request: lick rate against absolute session time, by epoch. It is the right
plot for "does the animal slow down as the session goes on", and it is the WRONG plot for "is the
quit an accumulation or a step" -- **averaging step functions whose steps fall at different times
manufactures a smooth ramp** whatever the underlying shape (`DECISIONS`, 2026-09-19). Panel A is
therefore drawn twice: over all sessions, and over CENSORED sessions only.

**AND THE CENSORED PANEL IS NOT THE CLEAN CONTROL THIS DOCSTRING FIRST CLAIMED IT WAS.** It said
"no step can be hiding in the average", which confuses NO QUIT with NO DETECTED QUIT. Priya spotted
the difference on the figure (2026-09-20: *"in 23 quit prodrome gated, top row acute 'sessions with
NO quit' still looks like there's a step off in lick number"*). `engagement_gate` needs a sustained
non-recovering run to confirm a quit, so an animal that stops near the END of a session leaves too
little tail and the session is labelled censored -- and post-stroke sessions run to a fixed ~120
min, so a quit at 110 min has ten minutes to prove itself. The TERMINAL DROP table measures how
often that happens instead of assuming it does not.

**MEASURED, AND THE ACUTE CENSORED PANEL IS NOT USABLE.** Final 10 min against the session's own
mid-session rate: pre 0.77 (29 sessions, 21% below half), chronic 0.86 (18, 6%), subacute 0.67
(4, 25%) -- and **ACUTE 0.48 from FOUR sessions, THREE of which fall below half**. So the acute
"no quit" curve is about one clean session plus three quits the gate could not confirm, which is
exactly the step Priya saw in it. Read the pre and chronic censored curves; do not read acute.

Panel B is the artefact-free version. Each quitter is aligned to ITS OWN quit, and compared with
sessions from the SAME ANIMAL that were still engaged at that same absolute session time, read in
the same bins. Lick rate declines in every session, so a quit-aligned average alone always shows a
ramp into the quit and proves nothing; the time-matched control is what makes the comparison mean
something.

    PRODROME     rate falls BELOW the matched control in the minutes before the quit. The animal
                 was flagging first -- the quit is the end of a process.
    STEP         rate tracks the control up to the quit and then collapses. The quit is a state
                 change, which is Priya's by-eye impression (*"it looks mostly binary (suddenly
                 quits)"*) and the thing a fatigue account has to explain.

**AND IT IS SPLIT NEAR VERSUS FAR, WHICH FIXES A CONFOUND RATHER THAN ADDING A CUT** (Priya:
*"should we split it by spout position?"*). A pooled licks-per-MINUTE curve mixes all six spouts,
and after the stroke a far-contralateral trial yields almost no licks at all -- hit rate 0.033
acutely. So a post-stroke session sits lower on the pooled curve partly BECAUSE a third of its
trials are positions the animal cannot do, which is a composition effect and not a slowing. The
within-session SHAPE is fairly safe, since positions are interleaved and the mix is roughly
constant through a session; the BETWEEN-EPOCH LEVEL is not.

The position-split panels therefore use LICKS PER TRIAL rather than licks per minute, because the
trial is the unit a position attaches to. Near versus far rather than all six: six positions times
time bins times four epochs is too sparse to read, and `near_codes()` is the split the rest of the
deck already uses.

**AND LICKS PER TRIAL IS STILL NOT A RATE** (Priya: *"licks per trial is different than lick rate
though. can we also plot an inter-lick interval per trial?"*). It confounds HOW FAST the animal
licks with HOW LONG it chose to keep licking, and only the first is a motor measure. The
INTER-LICK INTERVAL separates them, and it is the best fatigue measure in this dataset:

    licks per trial   engagement + motor.  Falls if the animal gives up OR if it slows down.
    INTER-LICK INTERVAL   MOTOR ONLY.  A mouse licks at ~7 Hz; orofacial fatigue or a tongue
                 deficit should LENGTHEN the interval. A decision to stop cannot.

ILI is computed WITHIN BOUTS -- consecutive licks more than `BOUT_MAX_S` apart belong to different
bouts, and the gap between bouts is a pause, not a slow lick. The per-trial statistic is the MEDIAN
of the within-bout intervals, over trials with at least `MIN_LICKS` licks, so a trial with one
isolated lick contributes nothing rather than a spurious interval.

**THIS IS THE ONE MEASURE THAT COULD STILL RESCUE THE FATIGUE ACCOUNT.** The quit-point regression
showed the session ends on a clock rather than a lick budget, but that is about WHEN the animal
stops; it says nothing about whether the tongue slows while it is working. If ILI lengthens across
a session -- and more so after the stroke -- there is motor fatigue even though it is not what
terminates the session.

    python -m scripts.rest_migration.quit_prodrome [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

BOUT_MAX_S = 0.5                    # s; a longer gap is a new bout, not a slow lick
MIN_LICKS = 4                       # per trial, so the median is over >= 3 intervals
BIN_S = 150.0                       # 2.5 min bins, quit-aligned
SESS_BIN_S = 300.0                  # 5 min bins, absolute session time
SESS_MAX_S = 7200.0
WIN = (-1800.0, 600.0)              # -30 to +10 min around the quit
PRE_WIN = (-600.0, 0.0)             # the "just before the quit" window the headline number uses
N_BOOT = 4000
EPS = ("pre", "acute", "subacute", "chronic")


def _cum(rows):
    """``(elapsed_s, cum_licks)`` as monotone arrays for interpolation."""
    t = np.asarray([float(r["elapsed_s"]) for r in rows], float)
    c = np.asarray([float(r["cum_licks"]) for r in rows], float)
    o = np.argsort(t)
    return t[o], np.maximum.accumulate(c[o])


def rate_in(t, c, t0, t1):
    """Licks per minute between `t0` and `t1`, or NaN if the session does not span it."""
    if t1 <= t0 or t0 < t[0] or t1 > t[-1]:
        return float("nan")
    return float((np.interp(t1, t, c) - np.interp(t0, t, c)) / ((t1 - t0) / 60.0))


def _boot(by, rng, n_boot=N_BOOT):
    A = sorted(by)
    if not A:
        return None
    flat = [x for k in A for x in by[k]]
    o = []
    for _ in range(n_boot):
        vals = []
        for k in (A[i] for i in rng.integers(0, len(A), len(A))):
            sa = by[k]
            vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        o.append(float(np.mean(vals)))
    o = np.asarray(o)
    return float(np.mean(flat)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def session_row(item):
    """All per-session work for one session, or ``None`` if it cannot contribute.

    **THE OPTIONS TRAVEL IN THE ITEM, NOT IN A MODULE GLOBAL.** `fan_out` uses the SPAWN start
    method, so a child re-imports this module fresh and never sees a global the parent assigned at
    runtime -- a gate set that way would silently do nothing in every worker while appearing to
    work in a serial run.

    **MODULE-LEVEL BECAUSE `parallel.fan_out` SPAWNS AND SPAWN PICKLES BY NAME** (CLAUDE.md ground
    rule 6). Measured on six curated sessions: 52.3 s serial against 16.2 s over six workers, a
    x3.2 speed-up with byte-identical results. The loop is ~100% I/O -- `session_trials` and the
    lick/cue reads come off MICROSCOPE -- so the gain is from overlapping waits, not from cores,
    and it will not scale linearly however many workers are thrown at it.
    """
    import numpy as np
    from wfield_local import config, epochs
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    from scripts.rest_migration.channel_position_maps import _daq_rate
    from scripts.rest_migration.engagement_decomposition import near_codes, session_trials
    from scripts.rest_migration.quit_point import session_quit

    lab, gate, horizon_min = item
    s = next((x for x in config.load_sessions() if x["label"] == lab), None)
    if s is None:
        return None
    tr = session_trials(s, 2.0)
    if len(tr) < 60:
        return None
    q = session_quit(s, tr)
    if q is None:
        return None
    t, c = _cum(tr)

    # PER-TRIAL LICKS BY POSITION. `cum_licks` is the running count at each cue, so the difference
    # between consecutive cues is the licks belonging to that trial, and the position is the
    # EARLIER trial's. The last trial has no successor and is dropped.
    tt = sorted(tr, key=lambda r: float(r["elapsed_s"]))

    # ENGAGEMENT GATE AND COMMON HORIZON, both applied to the PER-TRIAL series only -- the
    # licks-per-minute curves and the quit alignment deliberately keep every trial, because the
    # quit itself is what they are measuring.
    #
    # WHY THE GATE MATTERS AND THE FIGURE WAS WRONG WITHOUT IT: trials inside the terminal quit
    # period have ~zero licks, and they are 0.040 of pre-stroke trials against 0.217 acutely --
    # a 5x epoch-dependent difference concentrated in the late bins. An ungated licks-per-trial
    # curve therefore has its composition track the independent variable, which is the one thing
    # an epoch contrast cannot tolerate (see `channel_position_maps`).
    if gate and not q["censored"]:
        tt = [r for r in tt if float(r["elapsed_s"]) < float(q["quit_elapsed_s"])]
    if horizon_min:
        tt = [r for r in tt if float(r["elapsed_s"]) <= horizon_min * 60.0]
    if len(tt) < 30:
        return None
    near = near_codes()
    # THE THIRD ELEMENT IS THE TRIAL-ORDER FRACTION, appended rather than inserted so every
    # existing consumer of (time, value) keeps working. Quartiles are defined on TRIAL ORDER, not
    # on wall-clock, to match the session-quintile convention in `engagement_decomposition` and
    # because trial count is what a session is actually made of.
    per_pos = {"near": [], "far": []}
    nt = max(len(tt) - 2, 1)
    for i in range(len(tt) - 1):
        dl = float(tt[i + 1]["cum_licks"]) - float(tt[i]["cum_licks"])
        if dl < 0:
            continue
        per_pos["near" if int(tt[i]["pos"]) in near else "far"].append(
            (float(tt[i]["elapsed_s"]), dl, i / nt))

    # PER-TRIAL MEDIAN INTER-LICK INTERVAL, the motor-only measure. Needs the actual lick TIMES,
    # which `session_trials` does not return -- it stores a running COUNT.
    per_ili = {"near": [], "far": []}
    cs = np.asarray(_load_cue_events(s["h5"])["cue_samples"], np.int64)
    lick_s = np.asarray(_load_daq_events(s["h5"], "lick_analog", 2.5, 1.0,
                                         (0.001, 0.020), 0.10)["lick_samples"], np.int64)
    sr = float(_daq_rate(s))
    for i_r, r in enumerate(tt):
        k = int(r["order"])
        if k + 1 >= cs.size:
            continue
        seg = lick_s[(lick_s >= cs[k]) & (lick_s < cs[k + 1])]
        if seg.size < MIN_LICKS:
            continue
        iv = np.diff(seg) / sr
        iv = iv[iv <= BOUT_MAX_S]                   # within-bout only
        if iv.size < MIN_LICKS - 1:
            continue
        per_ili["near" if int(r["pos"]) in near else "far"].append(
            (float(r["elapsed_s"]), float(np.median(iv)) * 1000.0,
             i_r / max(len(tt) - 1, 1)))

    return dict(label=lab, animal=config.animal_of(lab), epoch=epochs.epoch_of(lab),
                t=t, c=c, per_pos=per_pos, per_ili=per_ili,
                quit_s=(float("nan") if q["censored"] else float(q["quit_elapsed_s"])),
                censored=bool(q["censored"]), end_s=float(t[-1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--gate", action="store_true",
                    help="drop trials inside the terminal quit period from the PER-TRIAL panels "
                         "(licks/trial and ILI). They are 0.040 of pre trials against 0.217 "
                         "acutely, so ungated those panels track the recording, not the animal.")
    ap.add_argument("--horizon-min", type=float, default=None, metavar="T",
                    help="also drop per-trial data after T minutes. All epochs are complete to "
                         "60 min; pre falls to 33/44 by 90 and everything thins past 100.")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes; default `parallel.default_jobs()` (cores-2, cap 8)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import config, epochs, parallel
    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))

    labels = [s["label"] for s in config.load_sessions()
              if s["label"] in want and epochs.epoch_of(s["label"])
              and not (a.animals and config.animal_of(s["label"]) not in a.animals)]
    items = [(lab, a.gate, a.horizon_min) for lab in labels]
    if a.gate or a.horizon_min:
        print(f"PER-TRIAL PANELS: gate={a.gate} horizon={a.horizon_min} "
              f"(licks/min and quit-alignment panels are UNAFFECTED by design)")
    res, fail = parallel.fan_out(items, session_row, jobs=a.jobs, label="session")
    # **SORTED, NOT COMPLETION ORDER.** Keying by label is not enough: dict insertion order is
    # completion order, every bootstrap pool is built by iterating `sess`, and a seeded RNG drawing
    # indices over a differently-ordered list gives different draws. The first parallel run
    # reproduced the point estimate exactly (-18.7) and moved the CI ([-22.9,-13.7] -> [-23.1,
    # -13.6]) -- small, but a CI that changes run to run is not reproducible.
    sess = {lab: r for lab, r in sorted(((r["label"], r) for _l, r in res if r is not None),
                                        key=lambda kv: kv[0])}
    for lab in sorted(sess):
        v = sess[lab]
        print(f"   {lab:14s} {v['epoch']:9s} "
              + ("CENSORED" if v["censored"] else f"quit {v['quit_s'] / 60:.0f} min")
              + f"   ends {v['end_s'] / 60:.0f} min", flush=True)
    if fail:
        print(f"  !! {len(fail)} session(s) failed: "
              + ", ".join(f"{x[0]} ({x[1][:40]})" for x in fail[:4]), flush=True)

    if not sess:
        print("no sessions -- a failed run, not a result")
        return 1
    rng = np.random.default_rng(a.seed)
    bar = "=" * 96
    tag = ("_gated" if a.gate else "") + (f"_h{int(a.horizon_min)}" if a.horizon_min else "")

    # ---- PANEL A: lick rate against ABSOLUTE session time -------------------------------------
    sedges = np.arange(0.0, SESS_MAX_S + 1e-9, SESS_BIN_S)
    sctr = (sedges[:-1] + sedges[1:]) / 2
    sess_curves = {}
    for subset in ("all", "censored"):
        cur = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for v in sess.values():
            if subset == "censored" and not v["censored"]:
                continue
            for i in range(sctr.size):
                r = rate_in(v["t"], v["c"], sedges[i], sedges[i + 1])
                if np.isfinite(r):
                    cur[v["epoch"]][i][v["animal"]].append(r)
        sess_curves[subset] = cur

    print(f"\n{bar}\nLICK RATE (licks/min) AGAINST ABSOLUTE SESSION TIME -- ALL SESSIONS\n{bar}")
    show = [i for i in range(sctr.size) if sctr[i] <= 6600]
    print(f"  {'epoch':<10}" + "".join(f"{sctr[i] / 60:>8.0f}" for i in show))
    for e in EPS:
        g = [_boot(sess_curves["all"][e][i], rng) for i in show]
        n = max((len(sess_curves["all"][e][i]) for i in show), default=0)
        if not any(g):
            continue
        print(f"  {e:<10}" + "".join(f"{x[0]:>8.0f}" if x else f"{'--':>8}" for x in g)
              + f"   ({sum(1 for v in sess.values() if v['epoch'] == e)} sessions, {n} animals)")
    print("\n  MINUTES along the top. This is the plot that answers 'does the animal slow down',")
    print("  and it CANNOT answer 'is the quit a step' -- unaligned steps average into a ramp.")

    # ---- PANEL B: quit-aligned, against TIME-MATCHED controls ---------------------------------
    quitters = [k for k, v in sess.items() if not v["censored"]]
    edges = np.arange(WIN[0], WIN[1] + 1e-9, BIN_S)
    ctrs = (edges[:-1] + edges[1:]) / 2
    curves = {"quit": defaultdict(lambda: defaultdict(list)),
              "ctrl": defaultdict(lambda: defaultdict(list))}
    n_pairs, rows_out = 0, []
    for k in quitters:
        v = sess[k]
        tq = v["quit_s"]
        # MATCHED CONTROLS: same animal, still engaged at this quitter's quit time. A session that
        # had already quit by tq is not a control, it is another quitter.
        ctrl = [w for j, w in sess.items()
                if j != k and w["animal"] == v["animal"]
                and (w["censored"] or w["quit_s"] > tq) and w["end_s"] > tq]
        n_pairs += len(ctrl)
        for i in range(ctrs.size):
            r = rate_in(v["t"], v["c"], tq + edges[i], tq + edges[i + 1])
            if np.isfinite(r):
                curves["quit"][i][v["animal"]].append(r)
            for w in ctrl:
                rc = rate_in(w["t"], w["c"], tq + edges[i], tq + edges[i + 1])
                if np.isfinite(rc):
                    curves["ctrl"][i][w["animal"]].append(rc)
        rows_out.append(dict(
            label=k, animal=v["animal"], epoch=v["epoch"], quit_min=round(tq / 60, 1),
            n_matched_controls=len(ctrl),
            rate_pre_quit=rate_in(v["t"], v["c"], tq + PRE_WIN[0], tq + PRE_WIN[1]),
            rate_ctrl=(float(np.nanmean([rate_in(w["t"], w["c"], tq + PRE_WIN[0], tq + PRE_WIN[1])
                                         for w in ctrl])) if ctrl else float("nan")),
            # ILI GOES IN THE CSV so the per-animal check is a file read next time, not an
            # hour of re-loading. It was not here the first time and that cost a whole re-run.
            ili_near_ms=(float(np.median([x[1] for x in v["per_ili"]["near"]]))
                         if v["per_ili"]["near"] else float("nan")),
            ili_far_ms=(float(np.median([x[1] for x in v["per_ili"]["far"]]))
                        if v["per_ili"]["far"] else float("nan"))))

    print(f"\n{bar}\nQUIT-ALIGNED, against TIME-MATCHED controls "
          f"({len(quitters)} quitters, {n_pairs} pairings)\n{bar}")
    print(f"  {'t from quit (min)':<20}{'QUITTERS':>20}{'MATCHED CONTROLS':>22}{'difference':>14}")
    qm, cm = [], []
    for i in range(ctrs.size):
        g, h = _boot(curves["quit"][i], rng), _boot(curves["ctrl"][i], rng)
        qm.append(g)
        cm.append(h)
        if g and h:
            print(f"  {ctrs[i] / 60:>+8.1f}          {g[0]:>9.1f} [{g[1]:.0f},{g[2]:.0f}]"
                  f"{h[0]:>11.1f} [{h[1]:.0f},{h[2]:.0f}]{g[0] - h[0]:>+14.1f}")

    d = defaultdict(list)
    for r in rows_out:
        if np.isfinite(r["rate_pre_quit"]) and np.isfinite(r["rate_ctrl"]):
            d[r["animal"]].append(r["rate_pre_quit"] - r["rate_ctrl"])
    g = _boot(d, rng)
    print("\n  10 MIN BEFORE THE QUIT, quitter minus its OWN animal's time-matched controls:")
    print(f"    {g[0]:+.1f} licks/min [{g[1]:+.1f}, {g[2]:+.1f}]   ({len(d)} animals, "
          f"{sum(len(v) for v in d.values())} quitters)" if g else "    --")
    print("    NEGATIVE with a CI excluding zero = PRODROME. Spanning zero = STEP.")

    if rows_out:
        q = out_dir / f"epoch_23_quit_prodrome{tag}.csv"
        with open(q, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            wr.writeheader()
            wr.writerows(rows_out)
        print(f"\n  wrote {q}")

    # ---- PANEL C: LICKS PER TRIAL by NEAR/FAR against absolute session time --------------------
    pos_curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for v in sess.values():
        for grp, pts in v["per_pos"].items():
            if not pts:
                continue
            at = np.asarray([x[0] for x in pts], float)
            al = np.asarray([x[1] for x in pts], float)
            for i in range(sctr.size):
                m = (at >= sedges[i]) & (at < sedges[i + 1])
                if m.sum() >= 3:
                    pos_curves[(v["epoch"], grp)][i][v["animal"]].append(float(al[m].mean()))

    print(f"\n{bar}\nLICKS PER TRIAL by NEAR / FAR against session time "
          f"(composition-safe; see docstring)\n{bar}")
    print(f"  {'epoch':<10}{'spouts':<7}" + "".join(f"{sctr[i] / 60:>8.0f}" for i in show))
    for e in EPS:
        for grp in ("near", "far"):
            gg = [_boot(pos_curves[(e, grp)][i], rng) for i in show]
            if not any(gg):
                continue
            print(f"  {e:<10}{grp:<7}"
                  + "".join(f"{x[0]:>8.1f}" if x else f"{'--':>8}" for x in gg))
    print("\n  MINUTES along the top. LICKS PER TRIAL, so a position the animal cannot reach")
    print("  lowers its OWN curve instead of dragging down a pooled licks-per-minute rate.")

    # ---- PANEL D: median INTER-LICK INTERVAL against session time, by NEAR/FAR ----------------
    ili_curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for v in sess.values():
        for grp, pts in v.get("per_ili", {}).items():
            if not pts:
                continue
            at = np.asarray([x[0] for x in pts], float)
            al = np.asarray([x[1] for x in pts], float)
            for i in range(sctr.size):
                m = (at >= sedges[i]) & (at < sedges[i + 1])
                if m.sum() >= 3:
                    ili_curves[(v["epoch"], grp)][i][v["animal"]].append(float(np.median(al[m])))

    n_ili = sum(1 for v in sess.values() if any(v.get("per_ili", {}).values()))
    print(f"\n{bar}\nMEDIAN INTER-LICK INTERVAL (ms) against session time -- THE MOTOR-ONLY "
          f"MEASURE\n{bar}")
    print(f"  {n_ili} of {len(sess)} sessions have usable ILIs (within-bout, "
          f"gap <= {BOUT_MAX_S}s, >= {MIN_LICKS} licks/trial)")
    print(f"  {'epoch':<10}{'spouts':<7}" + "".join(f"{sctr[i] / 60:>8.0f}" for i in show))
    for e in EPS:
        for grp in ("near", "far"):
            gg = [_boot(ili_curves[(e, grp)][i], rng) for i in show]
            if not any(gg):
                continue
            print(f"  {e:<10}{grp:<7}"
                  + "".join(f"{x[0]:>8.0f}" if x else f"{'--':>8}" for x in gg))
    print("\n  A mouse licks at ~7 Hz, so ~140 ms is normal. ILI RISING across a session = the")
    print("  tongue slowing = MOTOR FATIGUE. ILI FLAT while licks-per-trial falls = the animal")
    print("  is choosing to stop, not losing the ability. This is the measure that separates them.")

    # ---- PER-ANIMAL ILI, because a cohort mean at n=4 is not a result (ground rule 8) ---------
    print(f"\n{bar}\nMEDIAN ILI (ms) PER ANIMAL, NEAR SPOUTS -- the check the cohort table "
          f"cannot do\n{bar}")
    print(f"  {'animal':<8}" + "".join(f"{e:>22}" for e in EPS))
    ili_by = defaultdict(lambda: defaultdict(list))
    for v in sess.values():
        vals = [x[1] for x in v.get("per_ili", {}).get("near", [])]
        if vals:
            ili_by[v["animal"]][v["epoch"]].append(float(np.median(vals)))
    for an in sorted(ili_by):
        line = f"  {an:<8}"
        for e in EPS:
            vv = ili_by[an].get(e, [])
            line += (f"{np.mean(vv):>14.0f} (n={len(vv):>2d})" if vv else f"{'--':>22}")
        print(line)
    print("\n  A cohort-level acute increase carried by fewer than three animals is not a result.")

    # PAIRED WITHIN ANIMAL, the test the level table cannot do -- pre baselines run 151 to 173 ms
    # across animals, so between-animal variance swamps a ~10 ms epoch effect unless it cancels.
    print(f"\n  CHANGE FROM PRE in median near-spout ILI (ms), paired within animal")
    for e in [x for x in EPS if x != "pre"]:
        P = {k: v for k, v in ((an, ili_by[an].get(e, [])) for an in ili_by) if v}
        Q = {k: v for k, v in ((an, ili_by[an].get("pre", [])) for an in ili_by) if v}
        shared = sorted(set(P) & set(Q))
        if not shared:
            continue
        obs = float(np.mean([np.mean(P[x]) - np.mean(Q[x]) for x in shared]))
        draws = []
        for _ in range(N_BOOT):
            dd = []
            for x in (shared[i] for i in rng.integers(0, len(shared), len(shared))):
                pa, qa = P[x], Q[x]
                dd.append(np.mean([pa[i] for i in rng.integers(0, len(pa), len(pa))])
                          - np.mean([qa[i] for i in rng.integers(0, len(qa), len(qa))]))
            draws.append(float(np.mean(dd)))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        star = " *" if (lo > 0 or hi < 0) else "  "
        print(f"    {e:<10} {len(shared)} animals  {obs:>+7.1f} ms [{lo:+.1f}, {hi:+.1f}]{star}")
    print("    A LONGER interval is SLOWER licking. This is a motor deficit if it holds.")

    # ALL-SESSION ILI TO CSV. The per-quitter file below covers only 45 of 96 sessions, so the
    # per-animal check could not be redone from it -- which is what forced a whole re-run.
    qa = out_dir / f"epoch_23_session_ili{tag}.csv"
    with open(qa, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "animal", "epoch", "censored",
                                           "ili_near_ms", "ili_far_ms", "n_near", "n_far"])
        w.writeheader()
        for lab in sorted(sess):
            v = sess[lab]
            nn = [x[1] for x in v["per_ili"]["near"]]
            ff = [x[1] for x in v["per_ili"]["far"]]
            w.writerow(dict(label=lab, animal=v["animal"], epoch=v["epoch"],
                            censored=v["censored"],
                            ili_near_ms=(float(np.median(nn)) if nn else ""),
                            ili_far_ms=(float(np.median(ff)) if ff else ""),
                            n_near=len(nn), n_far=len(ff)))
    print(f"\n  wrote {qa}")

    # ---- ARE THERE UNDETECTED QUITS IN THE "CENSORED" SESSIONS? ---------------------------
    # A censored session is one the DETECTOR found no quit in, which is not the same as one
    # the animal did not quit. Compare each censored session's final 10 min against its own
    # mid-session rate: a deep terminal drop with NO detected quit is a MISSED quit, and it
    # would put a step into the very panel that exists to be step-free.
    print(f"\n{bar}\nTERMINAL DROP IN CENSORED SESSIONS -- missed late quits?\n{bar}")
    print(f"  {'epoch':<10}{'censored n':>12}{'median last10/mid':>20}"
          f"{'frac < 0.5':>13}{'frac < 0.25':>14}")
    for e in EPS:
        rat = []
        for v in sess.values():
            if v["epoch"] != e or not v["censored"]:
                continue
            end = v["end_s"]
            mid = rate_in(v["t"], v["c"], 0.40 * end, 0.60 * end)
            last = rate_in(v["t"], v["c"], end - 600.0, end)
            if np.isfinite(mid) and np.isfinite(last) and mid > 1.0:
                rat.append(last / mid)
        if not rat:
            continue
        rat = np.asarray(rat)
        print(f"  {e:<10}{len(rat):>12}{np.median(rat):>20.2f}"
              f"{float((rat < 0.5).mean()):>13.2f}{float((rat < 0.25).mean()):>14.2f}")
    print("\n  Near 1 = the session really did run to the end engaged. Well below"
          " 0.5 with NO detected quit is a quit the gate could not confirm for want of a"
          " tail, and those sessions are what put a step into the 'no quit' panel.")

    # ---- SESSION QUINTILES (Priya, 2026-09-20: *"would quintile be better than quartile?"*) ----
    # **YES, AND THE REASON IS COMPARABILITY RATHER THAN RESOLUTION.** `engagement_decomposition`
    # already reports acute near-spout HIT RATE by session quintile as
    # 0.972 / 0.937 / 0.792 / 0.457 / 0.282. Putting licks-per-trial and ILI on the SAME bins makes
    # one sentence possible -- "across the quintiles where hit rate collapses from 0.97 to 0.28,
    # ILI does not move" -- which on a different binning would be hand-waving.
    #
    # Two lesser reasons: Q1 and Q5 are further apart than the first and last QUARTER, so a
    # monotonic decline shows a larger difference; and five bins separate a STEADY decline from a
    # LATE COLLAPSE, which no two-point summary can.
    NQ = 5
    fl = {}
    for key, src in (("rate", "per_pos"), ("ili", "per_ili")):
        for grp in ("near", "far"):
            for b in range(NQ):
                lo_f, hi_f = b / NQ, (b + 1) / NQ
                d = defaultdict(list)
                for v in sess.values():
                    vals = [x[1] for x in v.get(src, {}).get(grp, [])
                            if len(x) > 2 and lo_f <= x[2] < hi_f + (1e-9 if b == NQ - 1 else 0)]
                    if len(vals) >= 5:
                        d[(v["epoch"], v["animal"])].append(float(np.median(vals)))
                fl[(key, grp, b)] = d

    def _by_animal(key, grp, b, e):
        out = defaultdict(list)
        for (ee, an), vv in fl[(key, grp, b)].items():
            if ee == e:
                out[an] += vv
        return out

    for key, unit, title in (("rate", "licks/trial", "LICKS PER TRIAL"),
                             ("ili", "ms", "MEDIAN INTER-LICK INTERVAL")):
        print(f"\n{bar}\nSESSION QUINTILES -- {title} ({unit})\n{bar}")
        print(f"  {'epoch':<10}{'spouts':<7}"
              + "".join(f"{'Q' + str(i + 1):>11}" for i in range(NQ))
              + f"{'Q5 - Q1 (paired)':>28}")
        for e in EPS:
            for grp in ("near", "far"):
                line = f"  {e:<10}{grp:<7}"
                for b in range(NQ):
                    g = _boot(_by_animal(key, grp, b, e), rng)
                    line += f"{g[0]:>11.1f}" if g else f"{'--':>11}"
                a1, a5 = _by_animal(key, grp, 0, e), _by_animal(key, grp, NQ - 1, e)
                shared = sorted(set(a1) & set(a5))
                if shared:
                    obs = float(np.mean([np.mean(a5[x]) - np.mean(a1[x]) for x in shared]))
                    draws = []
                    for _ in range(N_BOOT):
                        dd = []
                        for x in (shared[i] for i in rng.integers(0, len(shared), len(shared))):
                            pa, qa = a5[x], a1[x]
                            dd.append(np.mean([pa[i] for i in rng.integers(0, len(pa), len(pa))])
                                      - np.mean([qa[i] for i in rng.integers(0, len(qa),
                                                                             len(qa))]))
                        draws.append(float(np.mean(dd)))
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    star = " *" if (lo > 0 or hi < 0) else "  "
                    line += f"{obs:>+14.1f} [{lo:+.1f},{hi:+.1f}]{star}({len(shared)})"
                print(line)
        print("  paired = within animal, only animals with BOTH Q1 and Q5. * = CI excludes zero.")
    print(f"\n  COMPARE against the hit-rate quintiles in `engagement_decomposition`:")
    print("    acute NEAR hit rate 0.972 / 0.937 / 0.792 / 0.457 / 0.282 over these same bins.")

    fig2, ax2 = plt.subplots(2, 4, figsize=(17.5, 8.0))
    for i_k, (key, ylab) in enumerate((("rate", "licks per trial"),
                                       ("ili", "inter-lick interval (ms)"))):
        for j, e in enumerate(EPS):
            axx = ax2[i_k][j]
            for grp, cc in (("near", "#2ca02c"), ("far", "#9467bd")):
                xs, ys, los, his = [], [], [], []
                for b in range(NQ):
                    g = _boot(_by_animal(key, grp, b, e), rng)
                    if g:
                        xs.append(b + 1)
                        ys.append(g[0])
                        los.append(g[1])
                        his.append(g[2])
                if len(xs) >= 3:
                    axx.plot(xs, ys, "-o", ms=5, lw=1.8, color=cc, label=grp)
                    axx.fill_between(xs, los, his, color=cc, alpha=0.15, lw=0)
            axx.set_xticks(range(1, NQ + 1))
            axx.set_xticklabels([f"Q{i}" for i in range(1, NQ + 1)])
            axx.set_title(e, fontsize=10)
            axx.set_xlabel("session quintile")
            if j == 0:
                axx.set_ylabel(ylab)
                axx.legend(fontsize=8, frameon=False)
    for row in ax2:
        used = [x for x in row if x.has_data()]
        if len(used) > 1:
            lo = min(x.get_ylim()[0] for x in used)
            hi = max(x.get_ylim()[1] for x in used)
            for x in used:
                x.set_ylim(lo, hi)
    fig2.suptitle("SESSION QUINTILES by epoch and spout group -- the same bins as the hit-rate "
                  "quintiles in `engagement_decomposition`.\n"
                  "TOP licks per trial = engagement + motor. BOTTOM inter-lick interval = MOTOR "
                  "ONLY. Y axes shared within each row.", fontsize=10)
    fig2.tight_layout(rect=(0, 0, 1, 0.91))
    p2 = out_dir / f"epoch_24_session_quintiles{tag}.png"
    fig2.savefig(p2, dpi=170)
    print(f"\n  wrote {p2}")

    # ---- WITHIN-ANIMAL DELTA FROM PRE (Priya, 2026-09-20) --------------------------------------
    # **THE LEVELS ARE NOT COMPARABLE ACROSS ANIMALS AND THE DIFFERENCES ARE.**
    # Pre-stroke near-spout
    # ILI runs 151 ms (PS95) to 173 ms (PS93), and pre licks-per-trial differs by more than the
    # epoch effect being looked for -- so a cohort mean of LEVELS is dominated by which animals
    # happen to be in each cell, which at n=4 with unequal session counts is most of the variance.
    # Subtracting each animal's OWN pre profile, quintile by quintile, removes it exactly.
    #
    # The bootstrap resamples ANIMALS ONCE and uses the same draw for both arms, so the pairing is
    # preserved; resampling the two arms independently would put the between-animal variance back.
    def _delta(key, grp, b, e):
        """Per-animal (epoch minus that animal's own pre) for one quintile."""
        pre_by, ep_by = defaultdict(list), defaultdict(list)
        for (ee, an), vv in fl[(key, grp, b)].items():
            if ee == "pre":
                pre_by[an] += vv
            elif ee == e:
                ep_by[an] += vv
        return {an: (ep_by[an], pre_by[an]) for an in sorted(set(ep_by) & set(pre_by))}

    def _boot_delta(pairs, n_boot=N_BOOT):
        ans = sorted(pairs)
        if not ans:
            return None
        obs = float(np.mean([np.mean(pairs[a][0]) - np.mean(pairs[a][1]) for a in ans]))
        o = []
        for _ in range(n_boot):
            dd = []
            for a in (ans[i] for i in rng.integers(0, len(ans), len(ans))):
                pa, qa = pairs[a]
                dd.append(np.mean([pa[i] for i in rng.integers(0, len(pa), len(pa))])
                          - np.mean([qa[i] for i in rng.integers(0, len(qa), len(qa))]))
            o.append(float(np.mean(dd)))
        o = np.asarray(o)
        return obs, float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)), len(ans)

    POST = [e for e in EPS if e != "pre"]
    for key, unit, title in (("rate", "licks/trial", "LICKS PER TRIAL"),
                             ("ili", "ms", "MEDIAN INTER-LICK INTERVAL")):
        print(f"\n{bar}")
        print(f"WITHIN-ANIMAL DELTA FROM PRE, BY QUINTILE -- {title} ({unit})")
        print(f"{bar}")
        print(f"  {'epoch':<10}{'spouts':<7}"
              + "".join(f"{'Q' + str(i + 1):>16}" for i in range(NQ)))
        for e in POST:
            for grp in ("near", "far"):
                line = f"  {e:<10}{grp:<7}"
                for b in range(NQ):
                    g = _boot_delta(_delta(key, grp, b, e))
                    if g:
                        star = "*" if (g[1] > 0 or g[2] < 0) else " "
                        line += f"{g[0]:>+11.1f}{star}({g[3]})"
                    else:
                        line += f"{'--':>16}"
                print(line)
        print("  each animal minus its OWN pre profile. * = 95% CI excludes zero. (n) = animals.")

    fig3, ax3 = plt.subplots(2, len(POST), figsize=(4.6 * len(POST), 8.0), squeeze=False)
    for i_k, (key, ylab) in enumerate((("rate", "delta licks per trial"),
                                       ("ili", "delta inter-lick interval (ms)"))):
        for j, e in enumerate(POST):
            axx = ax3[i_k][j]
            for grp, cc in (("near", "#2ca02c"), ("far", "#9467bd")):
                xs, ys, los, his = [], [], [], []
                for b in range(NQ):
                    g = _boot_delta(_delta(key, grp, b, e))
                    if g:
                        xs.append(b + 1)
                        ys.append(g[0])
                        los.append(g[1])
                        his.append(g[2])
                if len(xs) >= 3:
                    axx.plot(xs, ys, "-o", ms=5, lw=1.8, color=cc, label=grp)
                    axx.fill_between(xs, los, his, color=cc, alpha=0.15, lw=0)
            axx.axhline(0, color="0.35", lw=1.0, ls="--")
            axx.set_xticks(range(1, NQ + 1))
            axx.set_xticklabels([f"Q{i}" for i in range(1, NQ + 1)])
            axx.set_title(f"{e} - pre", fontsize=10)
            axx.set_xlabel("session quintile")
            if j == 0:
                axx.set_ylabel(ylab)
                axx.legend(fontsize=8, frameon=False)
    for row in ax3:
        used = [x for x in row if x.has_data()]
        if len(used) > 1:
            lo = min(x.get_ylim()[0] for x in used)
            hi = max(x.get_ylim()[1] for x in used)
            for x in used:
                x.set_ylim(lo, hi)
    fig3.suptitle("WITHIN-ANIMAL DELTA FROM PRE, by session quintile. Each animal minus its own "
                  "pre-stroke profile,\nso between-animal baseline spread (near ILI runs "
                  "151-173 ms across animals) cannot drive it. Dashed line = no change.",
                  fontsize=10)
    fig3.tight_layout(rect=(0, 0, 1, 0.90))
    p3 = out_dir / f"epoch_25_quintiles_delta_from_pre{tag}.png"
    fig3.savefig(p3, dpi=170)
    print(f"\n  wrote {p3}")

    # ---- FIGURE --------------------------------------------------------------------------------
    col = {"pre": "#4c72b0", "acute": "#c44e52", "subacute": "#dd8452", "chronic": "#55a868"}
    fig, axes = plt.subplots(3, 4, figsize=(19.5, 13.2))
    ax = axes[0]
    axes[0][3].axis("off")
    for j, (subset, ttl) in enumerate((("all", "ALL sessions"),
                                       ("censored", "sessions with NO quit (no step to hide)"))):
        for e in EPS:
            gg = [_boot(sess_curves[subset][e][i], rng) for i in range(sctr.size)]
            ok = [i for i in range(sctr.size) if gg[i] and len(sess_curves[subset][e][i]) >= 2]
            if len(ok) < 3:
                continue
            x = sctr[ok] / 60
            m = np.array([gg[i][0] for i in ok])
            lo = np.array([gg[i][1] for i in ok])
            hi = np.array([gg[i][2] for i in ok])
            ax[j].plot(x, m, "-", lw=1.8, color=col[e], label=e)
            ax[j].fill_between(x, lo, hi, color=col[e], alpha=0.15, lw=0)
        ax[j].set_xlabel("session time (min)")
        ax[j].set_ylabel("lick rate (licks/min)")
        ax[j].set_title(ttl, fontsize=10)
        ax[j].legend(fontsize=8, frameon=False)
    for name, arr, c in (("quitters", qm, "#d62728"),
                         ("time-matched controls", cm, "#1f77b4")):
        ok = [i for i in range(ctrs.size) if arr[i]]
        if not ok:
            continue
        x = ctrs[ok] / 60
        m = np.array([arr[i][0] for i in ok])
        lo = np.array([arr[i][1] for i in ok])
        hi = np.array([arr[i][2] for i in ok])
        ax[2].plot(x, m, "-o", ms=3.5, lw=1.7, color=c, label=name)
        ax[2].fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)
    ax[2].axvline(0, color="0.3", ls="--", lw=1.0)
    ax[2].set_xlabel("time from the quit (min)")
    ax[2].set_ylabel("lick rate (licks/min)")
    ax[2].set_title("QUIT-ALIGNED (controls read on the matched\nquitter's clock)", fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)
    # ROW 2: licks per trial, near vs far, one panel per epoch.
    for j, e in enumerate(EPS):
        axx = axes[1][j]
        for grp, c, ls in (("near", "#2ca02c", "-"), ("far", "#9467bd", "--")):
            gg = [_boot(pos_curves[(e, grp)][i], rng) for i in range(sctr.size)]
            ok = [i for i in range(sctr.size)
                  if gg[i] and len(pos_curves[(e, grp)][i]) >= 2]
            if len(ok) < 3:
                continue
            x = sctr[ok] / 60
            m = np.array([gg[i][0] for i in ok])
            lo = np.array([gg[i][1] for i in ok])
            hi = np.array([gg[i][2] for i in ok])
            axx.plot(x, m, ls, lw=1.8, color=c, label=f"{grp} spouts")
            axx.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
        axx.set_title(f"{e} -- licks per TRIAL", fontsize=10)
        axx.set_xlabel("session time (min)")
        if j == 0:
            axx.set_ylabel("licks per trial")
            axx.legend(fontsize=8, frameon=False)

    # ROW 3: median inter-lick interval, near vs far, one panel per epoch.
    for j, e in enumerate(EPS):
        axx = axes[2][j]
        for grp, c, ls in (("near", "#2ca02c", "-"), ("far", "#9467bd", "--")):
            gg = [_boot(ili_curves[(e, grp)][i], rng) for i in range(sctr.size)]
            ok = [i for i in range(sctr.size)
                  if gg[i] and len(ili_curves[(e, grp)][i]) >= 2]
            if len(ok) < 3:
                continue
            x = sctr[ok] / 60
            m = np.array([gg[i][0] for i in ok])
            lo = np.array([gg[i][1] for i in ok])
            hi = np.array([gg[i][2] for i in ok])
            axx.plot(x, m, ls, lw=1.8, color=c, label=f"{grp} spouts")
            axx.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
        axx.set_title(f"{e} -- median INTER-LICK INTERVAL", fontsize=10)
        axx.set_xlabel("session time (min)")
        if j == 0:
            axx.set_ylabel("inter-lick interval (ms)")
            axx.legend(fontsize=8, frameon=False)

    # SHARED Y WITHIN EACH FAMILY (Priya, 2026-09-20). **PER-PANEL AUTOSCALING MAKES FOUR EPOCHS
    # LOOK DIFFERENT WHEN THEY ARE NOT** -- matplotlib fits each axis to its own data, so a flat
    # ILI series spanning 157-177 ms and one spanning 168-190 ms both fill the panel and read as
    # the same picture. It is the colour-scale bug (DECISIONS, 2026-09-19) one dimension down, and
    # it bites hardest exactly where the answer is "nothing changes across the session", because a
    # rescaled flat line looks like structure.
    #
    # ROW-WISE rather than figure-wise: licks/min, licks/trial and milliseconds are three different
    # quantities and forcing one scale across them would be the opposite error.
    for row in axes:
        used = [x for x in row if x.has_data()]
        if len(used) < 2:
            continue
        lo = min(x.get_ylim()[0] for x in used)
        hi = max(x.get_ylim()[1] for x in used)
        for x in used:
            x.set_ylim(lo, hi)

    fig.suptitle("Lick rate across the session. LEFT and MIDDLE answer 'is there a drop-off'; "
                 "only the RIGHT panel can say whether a quit is a step or an accumulation, "
                 "because unaligned steps average into a ramp.\n"
                 "ROW 2 licks per trial = engagement + motor. ROW 3 inter-lick interval = MOTOR "
                 "ONLY: rising ILI is a slowing tongue, flat ILI with falling licks is a choice.\n"
                 "Y AXES ARE SHARED WITHIN EACH ROW, so panels in a row are directly comparable.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_dir / f"epoch_23_quit_prodrome{tag}.png"
    fig.savefig(p, dpi=170)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
