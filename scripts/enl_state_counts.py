"""How many WORKING and STOPPED trials actually exist, per animal x epoch x position?

Feasibility before implementation (Priya, 2026-09-24): the adjacent-window contrast in
`enl_states` needs a terminal collapse to exist in the first place, and pre-stroke -- where the
inference is actually valid -- a well-trained animal may simply work through the whole session.
If there are no pre-stroke stopped trials there is no pre-stroke contrast, and that is worth
knowing before the feature and decode layers are written rather than after.

BEHAVIOUR-SIDE COUNTS, AND THEY ARE AN UPPER BOUND ON THE IMAGING SET. Trials come from
`session_trials` (the repaired-position classifier), not from `_trial_features`, because this only
needs to count and the imaging path costs minutes per session. The two universes disagree slightly:
`hit` here uses the session's real response window while the decoder's `success` requires a lick
within `decode.max_rt_s`, so a slow-but-rewarded trial counts as a hit here and as a no-lick trial
there. That difference moves trials from `success` into `working`/`stopped`, never out of the
session, so **a cell that is empty here is empty there too** -- which is the question being asked.

    python -m scripts.enl_state_counts            # per animal x epoch
    python -m scripts.enl_state_counts --by-position
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak
from wfield_local import config, enl_states
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES


def _worker(item):
    """One session -> per-trial state rows. Module-level: spawn pickles by name (rule 6)."""
    from scripts.rest_migration.engagement_decomposition import session_trials
    from wfield_local import epochs
    from wfield_local.nolick_decoder import response_window_for
    from wfield_local.precue_engagement_states import engagement_gate

    lab = item["label"]
    s = next((x for x in config.load_sessions() if x["label"] == lab), None)
    if s is None:
        return []
    try:
        tr = session_trials(s, response_window_for(s))
    except Exception as exc:                       # a session without a rest mask, etc.
        return [{"label": lab, "error": str(exc)[:60]}]
    if len(tr) < 60:
        return [{"label": lab, "error": f"only {len(tr)} trials"}]

    order = np.array([t["order"] for t in tr])
    hit = np.array([bool(t["hit"]) for t in tr])
    lat = np.array([t["latency_s"] if t["latency_s"] is not None else np.nan for t in tr], float)
    pos = np.array([POSITION_NAMES.get(int(t["pos"]), str(t["pos"])) for t in tr])

    max_rt = float(config.defaults()["decode"].get("max_rt_s", 2.0))
    # A HIT IS A LICK ANYWHERE IN THE TRIAL WINDOW (Priya, 2026-09-24). Fixed here; it was the whole
    # of the behaviour-vs-decode gap.
    #
    # `session_trials` was called with `analysis_kit.RESP_S` = **2.0 s**, while `decode.max_rt_s` is
    # **3.5 s** and the task's own response window (`gui_config.json`) is 3.5 s. So `hit` was capped
    # at 2.0, `success = hit & (lat <= 3.5)` collapsed to just `hit`, and **a lick at 2.5 s was not
    # a hit at all here** -- it fell through to `stopped` or `working`. The decode calls that same
    # trial `success`, because `categorize` uses max_rt = 3.5.
    #
    # MEASURED, and this is what it cost: PS94 327 vs 326 and PS95 509 vs 495 -- agreeing per
    # position -- while PS93 read 67 against the decode's 40 and PS92 10 against 6. The two animals
    # that diverged are the two with the slowest licks, i.e. the most trials landing in the
    # 2.0-3.5 s band this cap discarded. The handoff had recorded that gap as a behaviour-vs-imaging
    # UNIVERSE difference; it was not, it was this.
    #
    # The window comes from `response_window_for`, the same per-session `gui_config.json` value
    # `nolick_decoder.categorize` uses, so the two cannot drift apart again (rule 9). `RESP_S` is
    # left alone: `analysis_kit.session_behavior` defaults to it and that is a separate question.
    success = hit & np.isfinite(lat) & (lat <= max_rt)
    not_eng = engagement_gate(order, hit, pos)
    state = np.where(success, "success", np.where(not_eng, "stopped", "working"))

    keep = enl_states.adjacent_window(state, order)
    an = lab.split("_")[0]
    ep = epochs.epoch_of(lab)
    return [{"label": lab, "animal": an, "epoch": ep, "pos": p, "state": st,
             "adjacent": bool(k), "error": None}
            for p, st, k in zip(pos, state, keep)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by-position", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--cache", default=None,
                    help="CSV to read/write the per-trial rows, so a re-run of the "
                         "summary does not re-read every DAQ file")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    cache = Path(args.cache) if args.cache else None
    if cache and cache.is_file() and not args.refresh:
        print(f"[enl_counts] reading {cache}", flush=True)
        df_all = pd.read_csv(cache)
        rows = df_all.to_dict("records")
    else:
        sess = [{"label": s["label"]} for s in ak.curated_sessions()]
        print(f"[enl_counts] {len(sess)} curated sessions", flush=True)
        # `fan_sessions` sorts by the ITEM and returns (results, failures); a dict item is not
        # orderable, so the key has to be given. The sort is not optional -- completion order
        # would make any downstream bootstrap draw differently between runs (rule 6).
        res, fail = ak.fan_sessions(sess, _worker, jobs=args.jobs, key=lambda it: it["label"])
        if fail:
            print(f"[enl_counts] {len(fail)} session(s) FAILED outright:", flush=True)
            for f in fail[:5]:
                print(f"    {f}", flush=True)
        rows = [r for _item, chunk in res for r in chunk]
        if cache:
            pd.DataFrame(rows).to_csv(cache, index=False)
            print(f"[enl_counts] cached -> {cache}", flush=True)

    bad = [r for r in rows if r.get("error") and str(r.get("error")) != "nan"]
    if bad:
        print(f"\n[enl_counts] {len({b['label'] for b in bad})} session(s) unusable:", flush=True)
        for b in bad[:8]:
            print(f"    {b['label']}: {b['error']}", flush=True)
    df = pd.DataFrame([r for r in rows
                       if not (r.get("error") and str(r.get("error")) != "nan")])
    if df.empty:
        print("[enl_counts] nothing usable"); return 1

    print(f"\n=== trials by state x epoch (all positions, {df.label.nunique()} sessions) ===")
    print(pd.crosstab(df.epoch, df.state).to_string())

    print("\n=== SESSIONS THAT CONTAIN ANY stopped TRIAL (the contrast needs one) ===")
    has = (df[df.state == "stopped"].groupby(["animal", "epoch"]).label.nunique()
           .rename("sessions_with_stopped"))
    tot = df.groupby(["animal", "epoch"]).label.nunique().rename("sessions")
    print(pd.concat([tot, has], axis=1).fillna(0).astype(int).to_string())

    print("\n=== ADJACENT-WINDOW pairs per animal x epoch (what the contrast actually gets) ===")
    adj = df[df.adjacent]
    piv = (adj[adj.state.isin(enl_states.DEFAULT_CONTRAST)]
           .groupby(["animal", "epoch", "state"]).size().unstack("state").fillna(0).astype(int))
    print(piv.to_string())

    if args.by_position:
        print("\n=== ADJACENT-WINDOW pairs per animal x epoch x POSITION ===")
        pp = (adj[adj.state.isin(enl_states.DEFAULT_CONTRAST)]
              .groupby(["animal", "epoch", "pos", "state"]).size()
              .unstack("state").fillna(0).astype(int))
        print(pp.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
