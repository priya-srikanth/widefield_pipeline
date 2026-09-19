"""WHERE does the animal quit, and what has accumulated by then? The step, not the slope.

Priya, 2026-09-19: *"i can tell you anecdotally it looks mostly binary (suddenly quits) but does
that tell us anything about what causes it?"* -- followed by *"typically quits for the session"*.

**YES, AND IT CHANGES WHICH ANALYSIS IS THE RIGHT ONE.**

WHAT A STEP RULES OUT. Peripheral orofacial fatigue accumulates -- muscle does not fail abruptly --
so an abrupt quit is a poor fit for the muscular reading.

WHAT IT DOES NOT RULE OUT, and this is the trap. **Behaviour is a THRESHOLDED readout.** A smoothly
rising satiety signal crossing a decision boundary produces a step in the output. So a step rules
out a gradual behavioural MAPPING, not a gradual internal VARIABLE.

WHAT IT MAKES POSSIBLE. If the decline is a step you can LOCATE it and ask what is CONSERVED at it:

    cumulative LICKS conserved      an effort / consumption threshold
    elapsed TIME conserved          a clock -- boredom, time on task
    cumulative REWARDS conserved    satiety
    NONE conserved, all smaller     **the threshold itself moved** -- malaise, lowered motivation,
        after the stroke            which is the reading "feeling unwell or uninterested" predicts

**AND IT EXPOSES AN ARTEFACT IN THE AVERAGED CURVE.** Averaging step functions whose steps fall at
different times produces a SMOOTH-LOOKING DECLINE. A quintile plot of session-normalised hit rate
will therefore look like gradual accumulation even when every single session is a clean step -- and
it will look like evidence for the mechanism the step argues against. `engagement_decomposition`'s
within-session table has exactly that exposure; this module is the corrective, because aligning each
session to its OWN quit point before averaging is what lets the step survive the mean.

THE DETECTOR IS THE DECK'S OWN GATE, not a new definition. `precue_engagement_states.engagement_gate`
requires a NON-RECOVERING collapse, which Priya confirms matches the phenomenon -- so for once the
gate's assumption and the biology agree, and the quit point is the first trial of its terminal run.

    python -m scripts.rest_migration.quit_point [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HALF_WIN = 60          # trials either side of the quit point for the aligned average
N_BOOT = 4000


def session_quit(s, rows, resp_s=2.0):
    """``dict`` describing this session's terminal disengagement, or ``None`` if it never quits.

    Returns what had ACCUMULATED at the quit trial -- time, licks, rewards -- because those are the
    candidate threshold variables, and which of them is conserved across epochs is the question.
    """
    from wfield_local import daq_io
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.precue_engagement_states import engagement_gate

    order = np.array([r["order"] for r in rows])
    responded = np.array([bool(r["hit"]) for r in rows])
    pos = np.array([POSITION_NAMES.get(r["pos"], str(r["pos"])) for r in rows])
    ne = np.asarray(engagement_gate(order, responded, pos), bool)
    if not ne.any():
        return None
    # THE QUIT POINT IS THE FIRST TRIAL OF THE TERMINAL RUN. `engagement_gate` already requires
    # non-recovery, so scanning back from the end to the first False is the start of that run.
    k = len(ne)
    while k > 0 and ne[k - 1]:
        k -= 1
    if k == 0 or k >= len(rows):          # quit at trial 0 is a failed session, not a quit
        return None

    # rewards, for the satiety branch -- the only one of the three that needs a channel we do not
    # otherwise read here
    n_rew = None
    try:
        with daq_io.open_daq(s["h5"]) as f:
            sr, _ = daq_io.session_attrs(f)
            rv = daq_io.analog_channel(f, "reward_ttl", required=False)
        if rv is not None:
            edges = daq_io.rising_edges((np.asarray(rv) > 2.5).astype(np.int8))
            n_rew = int((np.asarray(edges, float) / sr <= rows[k]["elapsed_s"]).sum())
    except Exception:                                                # noqa: BLE001
        pass

    return dict(quit_trial=int(k), n_trials=len(rows),
                quit_frac=float(k / max(len(rows) - 1, 1)),
                quit_elapsed_s=float(rows[k]["elapsed_s"]),
                quit_cum_licks=float(rows[k]["cum_licks"]),
                quit_cum_rewards=(float(n_rew) if n_rew is not None else float("nan")),
                hit_before=float(np.mean([r["hit"] for r in rows[:k]])),
                hit_after=float(np.mean([r["hit"] for r in rows[k:]])),
                # STEP SHARPNESS: how much of the total drop happens in the 20 trials around the
                # quit. Near 1 = a genuine step. Near the fraction of trials spanned = a ramp.
                step_sharpness=_sharpness(rows, k))


def _sharpness(rows, k, w=10):
    h = np.array([r["hit"] for r in rows], float)
    pre, post = h[:k], h[k:]
    if pre.size < w or post.size < w:
        return float("nan")
    total = pre.mean() - post.mean()
    if abs(total) < 1e-9:
        return float("nan")
    local = h[max(0, k - w):k].mean() - h[k:k + w].mean()
    return float(local / total)


def _boot(by_animal, rng, n_boot=N_BOOT):
    animals = sorted(by_animal)
    if not animals:
        return None
    flat = [v for a in animals for v in by_animal[a]]
    out = []
    for _ in range(n_boot):
        vals = []
        for a in (animals[i] for i in rng.integers(0, len(animals), len(animals))):
            sa = by_animal[a]
            vals += [sa[i] for i in rng.integers(0, len(sa), len(sa))]
        if vals:
            out.append(float(np.mean(vals)))
    if len(out) < n_boot // 4:
        return None
    o = np.asarray(out)
    return float(np.mean(flat)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    from scripts.rest_migration.engagement_decomposition import near_codes, session_trials

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    near = near_codes()

    rows_out, aligned = [], defaultdict(list)
    n_seen, n_quit = defaultdict(int), defaultdict(int)
    for s in config.load_sessions():
        lab = s["label"]
        if lab not in want:
            continue
        if a.animals and config.animal_of(lab) not in a.animals:
            continue
        ep = epochs.epoch_of(lab)
        if not ep:
            continue
        try:
            tr = session_trials(s, 2.0)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if len(tr) < 60:
            continue
        n_seen[ep] += 1
        q = session_quit(s, tr)
        if q is None:
            print(f"   {lab:14s} {ep:9s} no terminal quit", flush=True)
            continue
        n_quit[ep] += 1
        q.update(label=lab, animal=config.animal_of(lab), epoch=ep)
        rows_out.append(q)
        # QUIT-ALIGNED hit rate, NEAR spouts only -- the alignment is the whole point, since
        # averaging unaligned steps manufactures a ramp.
        k = q["quit_trial"]
        nr = [r for r in tr if r["pos"] in near]
        idx = {id(r): i for i, r in enumerate(tr)}
        for r in nr:
            d = idx[id(r)] - k
            if -HALF_WIN <= d < HALF_WIN:
                aligned[(ep, d // 10)].append(r["hit"])
        print(f"   {lab:14s} {ep:9s} quit at trial {k}/{len(tr)} "
              f"({q['quit_frac']:.2f}), {q['quit_elapsed_s'] / 60:.0f} min, "
              f"{q['quit_cum_licks']:.0f} licks", flush=True)

    if not rows_out:
        print("no quits detected -- a failed run, not a result")
        return 1
    p = out_dir / "epoch_18_quit_point.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {p}")

    rng = np.random.default_rng(a.seed)
    eps = [e for e in ("pre", "acute", "subacute", "chronic") if n_quit.get(e)]
    bar = "=" * 96

    print(f"\n{bar}\nDOES THE ANIMAL QUIT AT ALL?\n{bar}")
    print(f"  {'epoch':<10}{'sessions':>10}{'with a terminal quit':>24}{'fraction':>12}")
    for e in eps:
        print(f"  {e:<10}{n_seen[e]:>10}{n_quit[e]:>24}{n_quit[e] / max(n_seen[e], 1):>12.2f}")

    print(f"\n{bar}\nWHAT HAS ACCUMULATED AT THE QUIT POINT\n{bar}")
    print(f"  {'epoch':<10}{'elapsed min':>22}{'cumulative licks':>26}{'cumulative rewards':>26}")
    for e in eps:
        line = f"  {e:<10}"
        for key, scale in (("quit_elapsed_s", 1 / 60.0), ("quit_cum_licks", 1.0),
                           ("quit_cum_rewards", 1.0)):
            d = defaultdict(list)
            for r in rows_out:
                if r["epoch"] == e and np.isfinite(r[key]):
                    d[r["animal"]].append(float(r[key]) * scale)
            g = _boot(d, rng)
            wdt = 22 if key == "quit_elapsed_s" else 26
            line += (f"{g[0]:>{wdt - 14}.0f} [{g[1]:.0f},{g[2]:.0f}]".rjust(wdt)
                     if g else "--".rjust(wdt))
        print(line)
    print("\n  CONSERVED across epochs -> that variable is the threshold.")
    print("  ALL SMALLER after the stroke -> the THRESHOLD ITSELF MOVED (malaise / motivation),")
    print("      which is a different answer from reaching the same threshold sooner.")

    print(f"\n{bar}\nIS IT REALLY A STEP?\n{bar}")
    print(f"  {'epoch':<10}{'hit before':>20}{'hit after':>20}{'step sharpness':>22}")
    for e in eps:
        line = f"  {e:<10}"
        for key in ("hit_before", "hit_after", "step_sharpness"):
            d = defaultdict(list)
            for r in rows_out:
                if r["epoch"] == e and np.isfinite(r[key]):
                    d[r["animal"]].append(float(r[key]))
            g = _boot(d, rng)
            wdt = 20 if key != "step_sharpness" else 22
            line += (f"{g[0]:>{wdt - 13}.2f} [{g[1]:.2f},{g[2]:.2f}]".rjust(wdt)
                     if g else "--".rjust(wdt))
        print(line)
    print("\n  step sharpness = drop across the 10 trials either side, over the whole-session drop.")
    print("  ~1.0 = the entire decline happens at the quit. Much less = a ramp with a kink.")

    print(f"\n{bar}\nQUIT-ALIGNED HIT RATE, NEAR SPOUTS (trial 0 = the quit)\n{bar}")
    ds = sorted({d for (_e, d) in aligned})
    print(f"  {'epoch':<10}" + "".join(f"{d * 10:>8}" for d in ds))
    for e in eps:
        line = f"  {e:<10}"
        for d in ds:
            v = aligned.get((e, d)) or []
            line += f"{np.mean(v):>8.2f}" if len(v) >= 10 else f"{'--':>8}"
        print(line)
    print("\n  ALIGNED, so a step survives the average. The same curve plotted against")
    print("  session-normalised time would look like a smooth ramp whatever the true shape --")
    print("  that is the artefact this module exists to avoid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
