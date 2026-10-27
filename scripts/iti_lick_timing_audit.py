"""AUDIT: how isolated are the ITI licks the decoders use? (Priya 10-27: "verifying that first-lick ITI (and all ITI)
licks require 2 s no licking, and are not close to trial start or stop?")

What the code does (iti_licks.extract):
  ramp / bout-start set (bout_*): onset gap >= 1 s saved, summaries use >= 2 s (GAP_MAIN), and NO cue or spout strobe
      within [-2, +0.75] s of the onset (CUE_EXCL)                                  -> isolated by construction
  DECODER ITI set (iti_X / iti_t; every where-decoder, lick-order and same-place read-out): EVERY lick of context "iti"
      (outside the response windows: 'tail' after a trial's window, 'lead' before the next spout arrives) with a 3-D
      direction -- NO gap rule and NO distance-from-event rule
  'first ITI lick' (iti_lick_order._bout_idx): bout index 0 with BOUT_GAP = 1 s over ALL licks -- 1 s, not 2 s, and no
      event rule
This script measures, per decoder ITI lick: gap to the previous lick onset (any context), time since the last
response-window end (trial stop), time since the last cue, time to the next spout strobe (arrival) and to the next cue.
Output <E>/iti_licks/iti_timing_audit.csv (per lick) + printed summaries by epoch.
Usage: python scripts/iti_lick_timing_audit.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402


def one(item) -> dict:
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    label = item[0]
    z = il._load(label)
    t = z["iti_t"].astype(float)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    B = il._behaviour(label)
    T = B.trials
    on_all = np.sort(np.concatenate([A.daq_on_s.to_numpy(float), np.asarray(B.lick_on_s, float)]))
    on_all = on_all[np.isfinite(on_all)]
    pk = A.daq_peak_s.to_numpy(float)
    on = A.daq_on_s.to_numpy(float)
    m = pd.Series(on, index=pk)
    m = m[~m.index.duplicated()]
    to = m.reindex(t).to_numpy(float)                      # onset of each decoder ITI lick (exact peak join)
    k = np.searchsorted(on_all, to - 0.05)
    gap = to - np.where(k > 0, on_all[np.maximum(k - 1, 0)], -np.inf)

    def since(ev, x):
        ev = np.sort(ev[np.isfinite(ev)])
        j = np.searchsorted(ev, x, side="right") - 1
        return np.where(j >= 0, x - ev[np.maximum(j, 0)], np.inf)

    def until(ev, x):
        ev = np.sort(ev[np.isfinite(ev)])
        j = np.searchsorted(ev, x)
        return np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)] - x, np.inf)
    stop = T.stop_s.to_numpy(float) if "stop_s" in T else T.cue_s.to_numpy(float) + 3.5
    stop = np.where(np.isfinite(stop), stop, T.cue_s.to_numpy(float) + 3.5)
    df = pd.DataFrame(dict(label=label, animal=an, epoch=str(epochs.epoch_of(label)), t=t, gap=gap,
                           since_stop=since(stop, to), since_cue=since(T.cue_s.to_numpy(float), to),
                           to_strobe=until(T.strobe_s.to_numpy(float), to), to_cue=until(T.cue_s.to_numpy(float), to)))
    return {"label": label, "df": df}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    from wfield_local import analysis_kit as ak
    labels = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))
    got, fail = ak.fan_sessions([(x,) for x in labels], one, jobs=a.jobs, label="audit", log=lambda m: None)
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    D = pd.concat([r["df"] for _it, r in got], ignore_index=True)
    D.to_csv(il.OUT / "iti_timing_audit.csv", index=False)
    D["first1"] = D.gap >= 1.0
    D["first2"] = D.gap >= 2.0
    near = (D.since_stop < 1.0) | (D.to_strobe < 1.0)
    D["near1"] = near
    D["clean"] = (D.since_stop >= 2.0) & (D.to_strobe >= 2.0) & (D.since_cue >= 2.0)
    pd.set_option("display.width", 200)
    print(f"{len(D)} decoder ITI licks, {D.label.nunique()} sessions")
    print("quantiles (s):")
    print(D[["gap", "since_stop", "to_strobe", "since_cue", "to_cue"]].replace(np.inf, np.nan)
          .quantile([0.05, 0.1, 0.25, 0.5]).round(2).to_string())
    g = D.groupby("epoch")
    print("fractions by epoch:")
    print(pd.DataFrame({"n": g.size(), "gap>=1s (first, 1 s rule)": g.first1.mean(), "gap>=2s": g.first2.mean(),
                        "<1 s after window end or <1 s before spout arrival": g.near1.mean(),
                        ">=2 s from window end, cue and spout arrival": g.clean.mean()}).round(3).to_string())
    f = D[D.first1]
    print("FIRST licks (1 s rule): fraction also gap>=2 s:", round(float(f.first2.mean()), 3),
          "; >=2 s from events:", round(float(f.clean.mean()), 3))
    print(D.groupby(["animal", "epoch"]).agg(n=("gap", "size"), clean=("clean", "mean"), first2=("first2", "mean"))
          .round(2).unstack("epoch").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
