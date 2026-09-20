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
therefore drawn twice: over all sessions, and over CENSORED sessions only (those with no detected
quit at all), where no step can be hiding in the average.

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    from scripts.rest_migration.channel_position_maps import _daq_rate
    from scripts.rest_migration.engagement_decomposition import near_codes, session_trials
    from scripts.rest_migration.quit_point import session_quit

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))

    sess = {}
    for s in config.load_sessions():
        lab = s["label"]
        if lab not in want or (a.animals and config.animal_of(lab) not in a.animals):
            continue
        ep = epochs.epoch_of(lab)
        if not ep:
            continue
        try:
            tr = session_trials(s, 2.0)
            if len(tr) < 60:
                continue
            q = session_quit(s, tr)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if q is None:
            continue
        t, c = _cum(tr)
        # PER-TRIAL LICKS BY POSITION. `cum_licks` is the running count at each cue, so the
        # difference between consecutive cues is the licks belonging to that trial, and the
        # position is the EARLIER trial's. The last trial has no successor and is dropped.
        tt = sorted(tr, key=lambda r: float(r["elapsed_s"]))
        near = near_codes()
        per_pos = {"near": [], "far": []}
        for i in range(len(tt) - 1):
            dl = float(tt[i + 1]["cum_licks"]) - float(tt[i]["cum_licks"])
            if dl < 0:
                continue
            per_pos["near" if int(tt[i]["pos"]) in near else "far"].append(
                (float(tt[i]["elapsed_s"]), dl))

        # PER-TRIAL MEDIAN INTER-LICK INTERVAL, the motor-only measure. Needs the actual lick
        # TIMES, which `session_trials` does not return -- it stores a running COUNT.
        per_ili = {"near": [], "far": []}
        try:
            cs = np.asarray(_load_cue_events(s["h5"])["cue_samples"], np.int64)
            lick_s = np.asarray(_load_daq_events(s["h5"], "lick_analog", 2.5, 1.0,
                                                 (0.001, 0.020), 0.10)["lick_samples"], np.int64)
            sr = float(_daq_rate(s))
            for r in tt:
                k = int(r["order"])
                if k + 1 >= cs.size:
                    continue
                seg = lick_s[(lick_s >= cs[k]) & (lick_s < cs[k + 1])]
                if seg.size < MIN_LICKS:
                    continue
                iv = np.diff(seg) / sr
                iv = iv[iv <= BOUT_MAX_S]           # within-bout only
                if iv.size < MIN_LICKS - 1:
                    continue
                per_ili["near" if int(r["pos"]) in near else "far"].append(
                    (float(r["elapsed_s"]), float(np.median(iv)) * 1000.0))
        except Exception as ex:                                      # noqa: BLE001
            print(f"      .. {lab}: no ILI ({type(ex).__name__})", flush=True)
        sess[lab] = dict(animal=config.animal_of(lab), epoch=ep, t=t, c=c, per_pos=per_pos,
                         per_ili=per_ili,
                         quit_s=(float("nan") if q["censored"] else float(q["quit_elapsed_s"])),
                         censored=bool(q["censored"]), end_s=float(t[-1]))
        print(f"   {lab:14s} {ep:9s} "
              + ("CENSORED" if q["censored"] else f"quit {q['quit_elapsed_s'] / 60:.0f} min")
              + f"   ends {t[-1] / 60:.0f} min", flush=True)

    if not sess:
        print("no sessions -- a failed run, not a result")
        return 1
    rng = np.random.default_rng(a.seed)
    bar = "=" * 96

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
                                         for w in ctrl])) if ctrl else float("nan"))))

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
        q = out_dir / "epoch_23_quit_prodrome.csv"
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

    fig.suptitle("Lick rate across the session. LEFT and MIDDLE answer 'is there a drop-off'; "
                 "only the RIGHT panel can say whether a quit is a step or an accumulation, "
                 "because unaligned steps average into a ramp.\n"
                 "ROW 2 licks per trial = engagement + motor. ROW 3 inter-lick interval = MOTOR "
                 "ONLY: rising ILI is a slowing tongue, flat ILI with falling licks is a choice.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_dir / "epoch_23_quit_prodrome.png"
    fig.savefig(p, dpi=170)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
