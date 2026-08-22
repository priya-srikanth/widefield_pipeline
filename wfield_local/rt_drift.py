"""First-lick latency across the COURSE of a session, per position, per phase.

    python -m wfield_local.rt_drift [--animals PS94 ...] [--output <dir>]

WHY THIS IS ITS OWN MODULE, AND NOT IN THE NIGHTLY. It answers one specific question and needs a
DAQ pass to do it -- `position_coding_directions` carries every other diagnostic for free because
the feature matrices are already in memory, but reaction time is not in the features. Ten minutes of
event loading for a control that has now been run and is not expected to move makes a poor nightly
step, so it is a one-off (Priya, 2026-08-22: "even if x1 not nightly").

WHAT IT CONTROLS FOR. Two separate things, and they are easy to confuse:

  1. THE WOULD-BE-LICK OFFSET. A no-lick trial's lick-aligned window starts at the cue plus that
     position's median RT -- ONE number for the whole session. If the animal slowed as the session
     ran, late trials would be placed progressively too early, and their features would drift away
     from the lick pattern for a purely mechanical reason. That is the exact shape of a within-
     session decline, so it has to be excluded before any such decline is believed. Measured: RT is
     flat (PS94 0.200/0.200/0.200/0.200 on 0817), a drift of <=0.05 s against a 2 s window.

  2. SLOWED VERSUS SKIPPED. A collapse in response rate late in a session could be an animal getting
     slower or an animal stopping. Flat RT with falling response rate is the SATED TAIL -- the licks
     that still happen are as fast as ever, there are just fewer of them. That distinction is what
     licenses reading the response-rate collapse as disengagement rather than fatigue.

Post-stroke far positions are the exception and legitimately slow (PS93 far_R reaches 1.50 s), which
is the deficit and not a within-session effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local import config
from wfield_local.behavior_position import classify_cues_with_backup
from wfield_local.locanmf_crossanimal_dff import _frames
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import is_engaged
from wfield_local.paths import PathResolver
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events
from wfield_local.position_coding_directions import BY_SEVERITY, QLABELS, QUARTILES, _pos_color

MIN_Q = 5           # a median over fewer than this is not a median


def session_rt(s):
    """(codes, rt_seconds, engaged, within_session_fraction) for one session, or None."""
    fs = float(config.defaults()["decode"].get("fs", 30.0))
    max_rt = float(config.defaults()["decode"].get("max_rt_s", 3.5))
    cue = _load_cue_events(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, _ = _frames(s, cue, lk)
    if not len(cue_f):
        return None
    codes = np.asarray(classify_cues_with_backup(s, cue))
    ls = np.sort(np.asarray(lick_f))
    j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1)
    rt = first - cue_f
    maxrt_n = round(max_rt * fs)
    eng = np.array([is_engaged(first[k], rt[k], maxrt_n) for k in range(cue_f.size)])
    n = cue_f.size
    frac = np.arange(n) / max(n - 1, 1)
    return codes, rt / fs, eng, frac


def collect(animal):
    """Median first-lick latency per position per within-session quartile, pre and post stroke."""
    out = {}
    for phase in ("pre", "post"):
        acc = {p: [[] for _ in QUARTILES] for p in BY_SEVERITY}
        for lab in [x for x in config.phase_labels(phase) if x.startswith(animal)]:
            s = next((x for x in SESSIONS if x["label"] == lab), None)
            if s is None:
                continue
            got = session_rt(s)
            if got is None:
                continue
            codes, rt, eng, frac = got
            for c, name in POSITION_NAMES.items():
                if name not in acc:
                    continue
                for qi, (lo, hi) in enumerate(QUARTILES):
                    m = eng & (codes == c) & (frac >= lo) & (frac < hi)
                    acc[name][qi].extend(rt[m].tolist())
        out[phase] = {p: [{"median": (float(np.median(v)) if len(v) >= MIN_Q else None),
                           "n": len(v)} for v in rows] for p, rows in acc.items()}
    return out


def figure(data, out_dir):
    """One row per phase, one column per animal."""
    animals = sorted(data)
    fig, axes = plt.subplots(2, len(animals), figsize=(3.5 * len(animals), 7),
                             squeeze=False, sharey="row")
    for j, an in enumerate(animals):
        for i, phase in enumerate(("pre", "post")):
            ax = axes[i][j]
            rows = (data[an] or {}).get(phase) or {}
            drew = False
            for P in BY_SEVERITY:
                cells = rows.get(P) or []
                xs = [k for k, c in enumerate(cells) if c.get("median") is not None]
                if len(xs) < 2:
                    continue
                drew = True
                ax.plot(xs, [cells[k]["median"] for k in xs], marker="o", ms=4.5, lw=1.5,
                        color=_pos_color(P), label=(P if (i == 0 and j == 0) else None))
            ax.set_xticks(range(4)); ax.set_xticklabels(QLABELS, fontsize=7, rotation=30)
            ax.grid(alpha=0.3)
            if not drew:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                        fontsize=8)
            if i == 0:
                ax.set_title(an, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{'PRE' if phase == 'pre' else 'POST'}-stroke\n"
                              f"median first-lick latency (s)", fontsize=8.5)
    axes[0][0].legend(fontsize=6.5, ncol=2, frameon=False)
    fig.suptitle(
        "First-lick latency across the COURSE of a session (ENGAGED trials only).\n"
        "FLAT here means late trials are SKIPPED, not slowed -- so a falling response rate is "
        "disengagement, not fatigue; and a session-constant would-be-lick offset cannot manufacture "
        "a within-session gradient.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    q = Path(out_dir) / "coding_rtdrift.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for an in (config.normalize_animals(args.animals) or list(config.animals())):
        try:
            data[an] = collect(an)
            print(f"  {an}: collected", flush=True)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  !! {an}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
    if not data:
        return 1
    (out / "rt_drift.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {figure(data, out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
