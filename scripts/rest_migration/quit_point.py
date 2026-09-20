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

THE QUIT RATE IS NOT COMPARABLE ACROSS EPOCHS WITHOUT A COMMON HORIZON, and `--horizon-min` is the
fix (Priya, 2026-09-19: *"is there a time we can select that will include most sessions, eg 80
minutes, and compare quitting only within that window (to avoid the variable length issue
pre-stroke)"*).

Pre-stroke sessions ran "up to 2 hrs or until they stopped licking" and span 74-167 min; post-
stroke ran a strict 120 min / 100 trials per position and span 95-125. A longer session has more
opportunity to contain a quit AND more tail in which to confirm one, so the raw detection rate
tracks the RECORDING POLICY rather than the animal. It measured 0.34 pre against 0.75 acute, which
is why that number was retired.

Administrative censoring removes it by construction: observe EVERY session for exactly T minutes,
drop those shorter than T, and re-run the detector on the truncated record so a late quit is
equally unconfirmable everywhere. **T = 80 keeps 92 of 96 sessions** (pre 40, acute 16, subacute
17, chronic 19, all four animals), losing only four pre-stroke sessions.

**AND THIS ALSO REPAIRS EVERYTHING MEASURED *AT* THE QUIT POINT**, which inherited the same
selection. Licks-at-quit looked like a result (-1270 acutely) until the per-animal table showed
PS92 and PS93 each had exactly ONE pre-stroke session with a detected quit, so the bootstrap had
nothing to resample within them and their uncertainty never entered the CI.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HALF_WIN = 60          # trials either side of the quit point for the aligned average
N_BOOT = 4000


EARLY_S = 600.0          # s; the fixed early window the independent lick rate is measured over


def _early_rate(rows, early_s=EARLY_S):
    """``(licks/min over the first `early_s`, licks/trial over those trials, n_trials)``.

    **THE POINT OF THIS COLUMN IS THAT IT IS NOT DERIVED FROM THE QUIT POINT**, which makes it the
    only non-circular predictor available for the fatigue question.

    `quit_cum_licks / quit_elapsed_s` looks like a lick rate and must never be used as one: it is
    `L / T`, so regressing it on `T` is guaranteed negative and on `L` guaranteed positive, for
    arithmetic rather than biological reasons. A FIXED early window is independent of when the quit
    occurs, so the two competing predictions become testable and OPPOSITE:

        MOTOR FATIGUE, quit at a fixed LICK COUNT  ->  quit TIME  ~ 1/rate, log-log slope -1
                                                       quit LICKS ~ const,  log-log slope  0
        TIME-DRIVEN, quit at a fixed TIME          ->  quit TIME  ~ const,  log-log slope  0
                                                       quit LICKS ~ rate,   log-log slope +1
    """
    early = [r for r in rows if float(r["elapsed_s"]) <= early_s]
    if len(early) < 5:
        return float("nan"), float("nan"), len(early)
    span = max(float(early[-1]["elapsed_s"]), 1.0) / 60.0
    licks = float(early[-1]["cum_licks"]) - float(early[0]["cum_licks"])
    return licks / span, licks / max(len(early) - 1, 1), len(early)


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
    dur = float(rows[-1]["elapsed_s"])
    er, el, en = _early_rate(rows)

    # RIGHT-CENSORED, NOT ABSENT (Priya, 2026-09-19: *"count the full session time if there is no
    # disengaged tail"*). **THE QUIT-DETECTION RATE IS CONFOUNDED BY SESSION-TERMINATION POLICY**:
    # pre-stroke Priya ran "up to 2 hrs or until they stopped licking", so a pre-stroke session
    # often ENDED AT THE QUIT and leaves no tail for a gate that requires a sustained non-recovering
    # run. Post-stroke was a strict 120 min / 100 trials per position, so the tail is always there.
    #
    # THE POLICY CHANGE IS VISIBLE IN THE DURATIONS: pre spans 74-167 min (IQR 89-143) against
    # acute's 100-126 (IQR 104-121). Comparing DETECTION RATES across that is comparing how long
    # the experimenter kept recording, and the 0.34-pre against 0.75-acute this module first
    # reported is mostly that.
    #
    # SO THE OUTCOME IS TIME ENGAGED, WITH CENSORING. No detected quit -> time engaged is the WHOLE
    # SESSION and the observation is RIGHT-CENSORED: **IF THE ANIMAL NEVER QUIT, it would have gone
    # on longer, so the recorded value is a LOWER BOUND on true engagement.**
    #
    # THIS COMMENT FIRST SAID UPPER BOUND, WHICH WAS BACKWARDS (Priya: *"isn't the pre-time engaged
    # a LOWER bound... if the animal never quit"*). Right-censoring means the event has not happened
    # yet, so the observation bounds the truth from BELOW. The upper-bound reasoning applied only to
    # the OTHER subpopulation -- sessions the animal DID quit, stopped so promptly that no tail
    # survives for the gate, where the recorded duration is approximately the true quit time.
    # Pre-stroke "no detected quit" is a MIXTURE of the two, and both are <= the truth.
    #
    # THE CONSERVATISM SURVIVES BY THE OPPOSITE ARGUMENT, and it is what licenses any claim here:
    # pre is ~66% censored (29/44) against acute's ~25% (4/16), so measured pre UNDERSTATES true pre
    # engagement more than measured acute understates acute. A measured pre > acute therefore
    # UNDERSTATES the true gap. A post-stroke SHORTFALL is trustworthy; a post-stroke EXCESS could be
    # differential censoring alone and means nothing.
    if not ne.any():
        return dict(quit_trial=-1, n_trials=len(rows), quit_frac=float("nan"),
                    quit_elapsed_s=float("nan"), quit_cum_licks=float("nan"),
                    quit_cum_rewards=float("nan"), hit_before=float("nan"),
                    hit_after=float("nan"), step_sharpness=float("nan"),
                    session_dur_s=dur, time_engaged_s=dur, censored=True,
                    frac_engaged=1.0, early_lpm=er, early_lpt=el, early_n=en)
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
                step_sharpness=_sharpness(rows, k),
                early_lpm=er, early_lpt=el, early_n=en,
                session_dur_s=dur, time_engaged_s=float(rows[k]["elapsed_s"]),
                censored=False, frac_engaged=float(rows[k]["elapsed_s"] / dur) if dur else 1.0)


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
    ap.add_argument("--horizon-min", type=float, default=None, metavar="T",
                    help="ADMINISTRATIVE CENSORING: observe every session for exactly T minutes "
                         "and drop sessions shorter than T. Makes the quit RATE comparable across "
                         "epochs, which it is not otherwise -- see the module docstring. T=80 "
                         "keeps 92 of 96 sessions (pre 40, acute 16, subacute 17, chronic 19).")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    from scripts.rest_migration.engagement_decomposition import near_codes, session_trials

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    near = near_codes()
    if a.horizon_min is not None:
        print(f"COMMON HORIZON {a.horizon_min:.0f} min: sessions shorter than this are DROPPED "
              f"and the rest are TRUNCATED, so every session contributes equal observation time.")

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
        if a.horizon_min is not None:
            # EVERY SESSION IS OBSERVED FOR EXACTLY THE SAME LENGTH OF TIME, and the detector is
            # re-run on the truncated record rather than the quit being looked up from the full
            # one. A quit at 78 min is then UNCONFIRMABLE under an 80 min horizon for every
            # session equally, which is what makes the rate comparable.
            h = a.horizon_min * 60.0
            if float(tr[-1]["elapsed_s"]) < h:
                print(f"   .. {lab:14s} shorter than the {a.horizon_min:.0f} min horizon "
                      f"({float(tr[-1]['elapsed_s']) / 60:.0f} min) -- dropped", flush=True)
                continue
            tr = [r for r in tr if float(r["elapsed_s"]) <= h]
            if len(tr) < 60:
                continue
        n_seen[ep] += 1
        q = session_quit(s, tr)
        if q is None:
            continue
        q.update(label=lab, animal=config.animal_of(lab), epoch=ep)
        rows_out.append(q)
        if q["censored"]:
            print(f"   {lab:14s} {ep:9s} still engaged at session end -- CENSORED at "
                  f"{q['session_dur_s'] / 60:.0f} min", flush=True)
            continue
        n_quit[ep] += 1
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
        print("no sessions scored -- a failed run, not a result")
        return 1
    p = out_dir / (f"epoch_18_quit_point"
                   f"{'_h%d' % a.horizon_min if a.horizon_min else ''}.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {p}")

    rng = np.random.default_rng(a.seed)
    eps = [e for e in ("pre", "acute", "subacute", "chronic") if n_quit.get(e)]
    bar = "=" * 96

    # TIME ENGAGED, THE POLICY-ROBUST OUTCOME. The detection RATE below it is reported only so the
    # censoring is visible -- it is NOT a result, because pre-stroke sessions were stopped when the
    # animal stopped and therefore cannot show a tail.
    print(f"\n{bar}\nTIME ENGAGED (min) -- censored sessions counted as engaged THROUGHOUT\n{bar}")
    print(f"  {'epoch':<10}{'n':>4}{'time engaged':>24}{'session duration':>24}"
          f"{'frac of session':>18}{'censored':>11}")
    for e in eps:
        v = [r for r in rows_out if r["epoch"] == e]
        if not v:
            continue
        line = f"  {e:<10}{len(v):>4}"
        for key, scale in (("time_engaged_s", 1 / 60.0), ("session_dur_s", 1 / 60.0),
                           ("frac_engaged", 1.0)):
            d = defaultdict(list)
            for r in v:
                if np.isfinite(r[key]):
                    d[r["animal"]].append(float(r[key]) * scale)
            g = _boot(d, rng)
            wdt = 24 if key.endswith("_s") else 18
            fmt = ".0f" if key.endswith("_s") else ".2f"
            line += (f"{g[0]:{fmt}} [{g[1]:{fmt}},{g[2]:{fmt}}]".rjust(wdt) if g
                     else "--".rjust(wdt))
        line += f"{np.mean([r['censored'] for r in v]):>11.2f}"
        print(line)
    print("\n  CENSORED = no detected quit, so time engaged is the WHOLE session. If the animal")
    print("  NEVER QUIT it would have gone on longer, so that is a LOWER BOUND on true engagement.")
    print("  PRE-STROKE IS THE MOST CENSORED, so measured pre understates true pre engagement more")
    print("  than measured acute understates acute -- a measured pre > acute UNDERSTATES the gap.")
    print("  A post-stroke SHORTFALL is trustworthy; a post-stroke EXCESS could be differential")
    print("  censoring alone and means nothing.")

    print(f"\n{bar}\nDETECTION RATE -- NOT A RESULT, shown so the censoring above is visible\n{bar}")
    print(f"  {'epoch':<10}{'sessions':>10}{'with a detected quit':>24}{'fraction':>12}")
    for e in eps:
        print(f"  {e:<10}{n_seen[e]:>10}{n_quit[e]:>24}{n_quit[e] / max(n_seen[e], 1):>12.2f}")
    print("\n  PRE-STROKE SESSIONS RAN 'UP TO 2 HRS OR UNTIL THEY STOPPED LICKING' (Priya) -- so a")
    print("  pre session often ENDED AT THE QUIT and leaves no tail for a gate that needs a")
    print("  sustained non-recovering run. Post-stroke ran to a fixed 120 min / 100 trials. The")
    print("  durations show it: pre spans 74-167 min, acute 100-126. Comparing these fractions")
    print("  compares recording policy, not the animal.")

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
