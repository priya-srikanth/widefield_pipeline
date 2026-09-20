"""IS THERE A PRODROME? Lick rate before a quit, against TIME-MATCHED sessions that did not quit.

Priya, 2026-09-19: *"if we plot lick rate over the course of the session, is it slower before a
quit than at the end of sessions without quitting (or at the same time in other sessions that
hadn't quit by that time)"*.

**THE CONTROL IS THE WHOLE POINT, AND IT IS THE RIGHT ONE.** Lick rate declines within every
session, so a quit-aligned average on its own will always show a ramp down into the quit and prove
nothing. The question is whether the decline before a quit is STEEPER than the decline any session
shows at the same point in time. So each quitter is matched to sessions from the SAME ANIMAL that
were still engaged at that same absolute session time, and both are read in the same bins.

WHAT THE TWO OUTCOMES MEAN:

    PRODROME     rate falls below the time-matched control in the minutes before the quit. The
                 animal was flagging first -- consistent with an effort or fatigue threshold being
                 approached, and with the quit being the end of a process.
    STEP         rate tracks the control right up to the quit and then collapses. The quit is a
                 state change, not an accumulation -- Priya's by-eye impression (*"it looks mostly
                 binary (suddenly quits)"*) and the thing a fatigue account has to explain.

**ALIGNMENT IS TO EACH SESSION'S OWN QUIT, NEVER TO A SESSION FRACTION.** Averaging step functions
with variable step times manufactures a ramp -- a pitfall this repo has already hit once
(`DECISIONS`, 2026-09-19). The control curve inherits the same alignment by construction, since it
is read at the matched quitter's clock.

    python -m scripts.rest_migration.quit_prodrome [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

BIN_S = 150.0                       # 2.5 min bins
WIN = (-1800.0, 600.0)              # -30 to +10 min around the quit
PRE_WIN = (-600.0, 0.0)             # the "just before the quit" window the headline number uses
N_BOOT = 4000


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
    from scripts.rest_migration.engagement_decomposition import session_trials
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
        sess[lab] = dict(animal=config.animal_of(lab), epoch=ep, t=t, c=c,
                         quit_s=(float("nan") if q["censored"] else float(q["quit_elapsed_s"])),
                         censored=bool(q["censored"]), end_s=float(t[-1]))
        print(f"   {lab:14s} {ep:9s} "
              + ("CENSORED" if q["censored"] else f"quit {q['quit_elapsed_s'] / 60:.0f} min")
              + f"   ends {t[-1] / 60:.0f} min", flush=True)

    quitters = [k for k, v in sess.items() if not v["censored"]]
    if not quitters:
        print("no quitters -- a failed run, not a result")
        return 1

    edges = np.arange(WIN[0], WIN[1] + 1e-9, BIN_S)
    ctrs = (edges[:-1] + edges[1:]) / 2
    curves = {"quit": defaultdict(lambda: defaultdict(list)),
              "ctrl": defaultdict(lambda: defaultdict(list))}
    n_ctrl_pairs, rows_out = 0, []

    for k in quitters:
        v = sess[k]
        tq = v["quit_s"]
        # THE MATCHED CONTROLS: same animal, and STILL ENGAGED at this quitter's quit time --
        # either censored (never quit) or quit later. A session that had already quit by tq is
        # not a control, it is another quitter.
        ctrl = [w for j, w in sess.items()
                if j != k and w["animal"] == v["animal"]
                and (w["censored"] or w["quit_s"] > tq) and w["end_s"] > tq]
        n_ctrl_pairs += len(ctrl)
        for i in range(ctrs.size):
            r = rate_in(v["t"], v["c"], tq + edges[i], tq + edges[i + 1])
            if np.isfinite(r):
                curves["quit"][i][v["animal"]].append(r)
            for w in ctrl:
                rc = rate_in(w["t"], w["c"], tq + edges[i], tq + edges[i + 1])
                if np.isfinite(rc):
                    curves["ctrl"][i][w["animal"]].append(rc)
        rows_out.append(dict(label=k, animal=v["animal"], epoch=v["epoch"],
                             quit_min=round(tq / 60, 1), n_matched_controls=len(ctrl),
                             rate_pre_quit=rate_in(v["t"], v["c"], tq + PRE_WIN[0], tq + PRE_WIN[1]),
                             rate_ctrl=float(np.nanmean([rate_in(w["t"], w["c"],
                                                                 tq + PRE_WIN[0], tq + PRE_WIN[1])
                                                         for w in ctrl])) if ctrl else float("nan")))

    print(f"\n  {len(quitters)} quitters, {n_ctrl_pairs} quitter-control pairings "
          f"(same animal, still engaged at the matched quit time)")

    rng = np.random.default_rng(a.seed)

    def boot(by):
        A = sorted(by)
        if not A:
            return None
        flat = [x for k_ in A for x in by[k_]]
        o = []
        for _ in range(N_BOOT):
            vals = []
            for k_ in (A[i] for i in rng.integers(0, len(A), len(A))):
                sa = by[k_]
                vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
            o.append(float(np.mean(vals)))
        o = np.asarray(o)
        return (float(np.mean(flat)), float(np.percentile(o, 2.5)),
                float(np.percentile(o, 97.5)))

    bar = "=" * 92
    print(f"\n{bar}\nLICK RATE (licks/min) ALIGNED TO THE QUIT, against TIME-MATCHED controls\n{bar}")
    print(f"  {'t from quit (min)':<20}{'QUITTERS':>22}{'MATCHED CONTROLS':>24}{'difference':>14}")
    qm, cm = [], []
    for i in range(ctrs.size):
        g, h = boot(curves["quit"][i]), boot(curves["ctrl"][i])
        qm.append(g)
        cm.append(h)
        if g and h:
            print(f"  {ctrs[i] / 60:>+8.1f}            {g[0]:>9.1f} [{g[1]:.0f},{g[2]:.0f}]"
                  f"{h[0]:>13.1f} [{h[1]:.0f},{h[2]:.0f}]{g[0] - h[0]:>+14.1f}")

    # THE HEADLINE: the paired difference in the 10 min BEFORE the quit, within animal.
    d = defaultdict(list)
    for r in rows_out:
        if np.isfinite(r["rate_pre_quit"]) and np.isfinite(r["rate_ctrl"]):
            d[r["animal"]].append(r["rate_pre_quit"] - r["rate_ctrl"])
    g = boot(d)
    print(f"\n  10 MIN BEFORE THE QUIT, quitter minus its own animal's time-matched controls:")
    print(f"    {g[0]:+.1f} licks/min [{g[1]:+.1f}, {g[2]:+.1f}]" if g else "    --")
    print("    CI excluding zero and NEGATIVE = a PRODROME. Spanning zero = a STEP.")

    q = out_dir / "epoch_23_quit_prodrome.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        wr.writeheader()
        wr.writerows(rows_out)
    print(f"\n  wrote {q}")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for name, arr, col in (("quitters", qm, "#d62728"),
                           ("time-matched controls (same animal, still engaged)", cm, "#1f77b4")):
        ok = [i for i in range(ctrs.size) if arr[i]]
        x = ctrs[ok] / 60
        m = np.array([arr[i][0] for i in ok])
        lo = np.array([arr[i][1] for i in ok])
        hi = np.array([arr[i][2] for i in ok])
        ax.plot(x, m, "-o", ms=3.5, lw=1.7, color=col, label=name)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
    ax.axvline(0, color="0.3", ls="--", lw=1.0)
    ax.set_xlabel("time from the quit (min)")
    ax.set_ylabel("lick rate (licks/min)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Is the quit an accumulation or a step?\nControls are read on the MATCHED "
                 "quitter's clock, so both curves share an alignment.", fontsize=10)
    fig.tight_layout()
    p = out_dir / "epoch_23_quit_prodrome.png"
    fig.savefig(p, dpi=180)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
