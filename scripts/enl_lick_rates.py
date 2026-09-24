"""How much licking is actually near the ENL decoding window? Measure before changing anything.

Priya, 2026-09-24: *"how are the ENL numbers not lick free? from the few sessions before we fixed
the GUI when we let some stray licks through?"*

Not a GUI artefact -- the baseline rate of stray licks inside the ENL. `precue_window_start` records
the measurement that prompted the 2026-08-17 lick-free gate on the HEADLINE pre-cue number: the task
contingency keeps the window quiet **90.8-99.5%** of the time per session, and **PS93 8/9 falls to
76%**. This script re-measures that on the curated set and adds the number nobody has: the rate for
the second BEFORE the window.

WHY THE LEAD WINDOW IS A SEPARATE QUESTION. The ENL contingency protects the ENL. It says nothing
about the interval just before it, which sits right after the spout strobe where licking is not
suppressed the same way -- so the LEAD rate is not bounded by the IN rate and could be far larger.
Widefield hemodynamics are slow enough for a lick there to reach into the window.

WHY THIS RUNS BEFORE THE SPLIT. `enl_decode` reaches its features through
`nolick_decoder.session_features`, which builds a FIXED ``[cue - post_s, cue]`` window and applies
no lick gate at all -- a different definition of "the pre-cue window" from
`locanmf_position_decoder._trial_features`, which slides the window to a clean gap and DROPS a trial
with no clean gap anywhere. That divergence is a rule 9 problem and Priya has asked for it to be
closed, but the size of the effect decides how much the closure moves: this measures it first.

THREE RATES, per session and per animal x epoch:

    in_window   a lick inside the fixed [cue - post_s, cue] window -- what the 2026-08-17 gate
                catches, and what `session_features` currently keeps
    lead        a lick in the LEAD_S before that window -- unmeasured until now, and the thing
                Priya's contamination control is about
    no_clean    no lick-free window exists anywhere between the spout strobe and the cue, so the
                strict builder would DROP the trial. This is the only rate that costs trials;
                the other two only relabel them.

    python -m scripts.enl_lick_rates [--jobs N] [--cache CSV] [--lead-s 1.0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wfield_local import analysis_kit as ak
from wfield_local import config, epochs

#: Seconds before the decoding window to inspect. Priya, 2026-09-24. Named here rather than inlined
#: for the reason `locanmf_position_decoder` gives about `precue_baseline_s`: a load-bearing interval
#: buried in an expression is invisible to anyone reading for what the analysis assumes.
LEAD_S = 1.0

#: The ENL window width. Matches `decode.precue_post_s`; read from config rather than assumed.
def _post_s() -> float:
    return float(config.defaults()["decode"].get("precue_post_s", 2.0))


def _worker(item):
    """One session -> per-trial rows. Module level so spawn can pickle it by name (rule 6)."""
    import numpy as np

    from wfield_local.nolick_decoder import FS
    from wfield_local.locanmf_position_decoder import (
        _frames,
        _load_cue_events,
        _load_daq_events,
        classify_cues_with_backup,
        lickfree_window,
    )

    lab = item["label"]
    s = next((x for x in config.load_sessions() if x["label"] == lab), None)
    if s is None:
        return []
    try:
        cue = _load_cue_events(s["h5"])
        lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        cue_f, lick_f, _csmp = _frames(s, cue, lk)
        codes = classify_cues_with_backup(s, cue)
    except Exception as exc:
        return [{"label": lab, "error": f"{type(exc).__name__}: {exc}"[:80]}]

    post_n = int(round(item["post_s"] * FS))
    lead_n = int(round(item["lead_s"] * FS))
    ls = np.sort(np.asarray(lick_f))

    # the per-trial spout-strobe frame, exactly as `_trial_features` derives it -- the strict
    # builder will not slide a window earlier than the strobe, so `no_clean` depends on it
    cs = np.asarray(cue["cue_samples"]); ss = np.asarray(cue["strobe_samples"])
    sr = float(cue["sample_rate_hz"])
    jj = np.searchsorted(ss, cs, side="right") - 1
    lead_to_strobe = np.where(jj >= 0, (cs - ss[np.clip(jj, 0, len(ss) - 1)]) / sr, np.nan)
    strobe_f = cue_f - lead_to_strobe * FS

    rows = []
    for k in range(cue_f.size):
        if codes[k] < 0 or int(cue_f[k]) < 0:
            continue                                   # unclassified, or outside imaging coverage
        c0 = int(cue_f[k])
        fixed = c0 - post_n
        if fixed < 0:
            continue
        in_win = bool(np.any((ls >= fixed) & (ls < c0)))
        in_lead = bool(np.any((ls >= fixed - lead_n) & (ls < fixed)))
        no_clean = False
        if in_win:
            no_clean = lickfree_window(c0, strobe_f[k], ls, post_n) is None
        rows.append({"label": lab, "animal": lab.split("_")[0], "epoch": epochs.epoch_of(lab),
                     "in_window": in_win, "lead": in_lead, "no_clean": no_clean, "error": None})
    return rows


def summarise(df):
    """Per animal x epoch and per session. Printed, not plotted -- this decides whether to plot."""
    ok = df[df.error.isna()] if "error" in df.columns else df
    out = (ok.groupby(["animal", "epoch"])
             .agg(trials=("in_window", "size"),
                  in_window=("in_window", "mean"),
                  lead=("lead", "mean"),
                  no_clean=("no_clean", "mean")))
    for c in ("in_window", "lead", "no_clean"):
        out[c] = (100 * out[c]).round(2)
    return out


def worst_sessions(df, key="lead", n=10):
    ok = df[df.error.isna()] if "error" in df.columns else df
    per = (ok.groupby("label").agg(trials=(key, "size"), rate=(key, "mean")))
    per["rate"] = (100 * per["rate"]).round(2)
    return per.sort_values("rate", ascending=False).head(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--lead-s", type=float, default=LEAD_S)
    args = ap.parse_args(argv)

    cache = Path(args.cache) if args.cache else None
    if cache and cache.is_file() and not args.refresh:
        print(f"[lick_rates] reading {cache}", flush=True)
        df = pd.read_csv(cache)
    else:
        post_s = _post_s()
        items = [{"label": s["label"], "post_s": post_s, "lead_s": args.lead_s}
                 for s in ak.curated_sessions()]
        print(f"[lick_rates] {len(items)} curated sessions, "
              f"window {post_s:g}s, lead {args.lead_s:g}s", flush=True)
        # `fan_sessions` sorts by the ITEM; a dict is not orderable, so the key is required -- and
        # the sort is not optional, completion order would reorder every downstream table (rule 6).
        res, fail = ak.fan_sessions(items, _worker, jobs=args.jobs, key=lambda it: it["label"])
        for f in fail[:5]:
            print(f"    FAILED {f}", flush=True)
        df = pd.DataFrame([r for _i, chunk in res for r in chunk])
        if cache:
            df.to_csv(cache, index=False)
            print(f"[lick_rates] cached -> {cache}", flush=True)

    bad = df[df.error.notna()] if "error" in df.columns else df.iloc[:0]
    for _i, r in bad.head(5).iterrows():
        print(f"    unusable {r['label']}: {r['error']}", flush=True)

    print(f"\n=== % of trials with a lick, per animal x epoch (lead = {args.lead_s:g}s) ===")
    print(summarise(df).to_string())
    print("\n  in_window  a lick INSIDE the fixed window -- the 2026-08-17 gate catches these")
    print("  lead       a lick in the second BEFORE it -- unguarded anywhere, Priya's control")
    print("  no_clean   no lick-free window exists at all -- the only rate that COSTS trials")
    print("\n=== worst sessions by LEAD rate ===")
    print(worst_sessions(df, "lead").to_string())
    print("\n=== worst sessions by IN-WINDOW rate ===")
    print(worst_sessions(df, "in_window").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
