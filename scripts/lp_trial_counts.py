"""LP_licks vs DAQ: per-trial categories under both lick sources, per animal x position x epoch (rule 8 table before any
LP figure; handoff 10-21 §2c step 3).

For every curated session, `nolick_decoder.categorize` (the categories the imaging pipeline uses: engaged = first lick
within max_rt; late_rewarded; undetected) is run twice -- WIDEFIELD_LICK_SOURCE=daq (spout contact) and =lp (video
attempt licks, `lp_licks`) -- on the same cues. Reported: engaged trials per animal x position x epoch under each source,
and the number of sessions that clear the 20-trial-per-position floor of the lick-aligned maps (MIN_TRIALS_PER_CLASS),
which is what decides whether a lick-aligned cell exists at all (acute far contra: 0 sessions under DAQ).

Output: <run>/imaging_encoding/qc/lp_trial_counts.csv (per trial) and lp_trial_counts_summary.csv.
Usage: PYTHONPATH=. python scripts/lp_trial_counts.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def one(label):
    from wfield_local import config, epochs, lp_licks
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.nolick_decoder import categorize
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    s = next(x for x in config.load_sessions() if x["label"] == label)
    args = _args("locanmf", "lick")
    out = {}
    for src in ("daq", "lp"):
        os.environ[lp_licks.LICK_SOURCE_ENV] = src
        try:
            codes, cat, _blk, rt_s, _cue_f = categorize(s, args)[:5]
        except lp_licks.LPUnavailable:
            os.environ.pop(lp_licks.LICK_SOURCE_ENV, None)
            return []
        finally:
            os.environ.pop(lp_licks.LICK_SOURCE_ENV, None)
        out[src] = (np.asarray(codes), np.asarray(cat, dtype=object), np.asarray(rt_s, float))
    codes = out["daq"][0]
    ep = epochs.epoch_of(label)
    return [{"label": label, "trial": k, "epoch": ep, "pos": POSITION_NAMES.get(int(c), str(c)),
             "cat_daq": out["daq"][1][k], "cat_lp": out["lp"][1][k],
             "rt_daq": out["daq"][2][k], "rt_lp": out["lp"][2][k]}
            for k, c in enumerate(codes) if c >= 0]


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.beta_maps import MIN_TRIALS_PER_CLASS
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    labels = sorted(x for x in ak.curated_labels() if x[-4:] >= "0801")          # August pre + post
    got, fail = ak.fan_sessions(labels, one, jobs=a.jobs)
    for it, err in fail:
        print("FAILED", it, err)
    T = pd.DataFrame([r for _, rr in got for r in rr])
    T["animal"] = T.label.str.split("_").str[0]
    for s in ("daq", "lp"):
        T[f"eng_{s}"] = T[f"cat_{s}"] == "engaged"
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc"
    assert_writable(out)
    T.to_csv(out / "lp_trial_counts.csv", index=False, float_format="%.4g")
    per_sess = T.groupby(["animal", "epoch", "pos", "label"])[["eng_daq", "eng_lp"]].sum().reset_index()
    for s in ("daq", "lp"):
        per_sess[f"ok_{s}"] = per_sess[f"eng_{s}"] >= MIN_TRIALS_PER_CLASS
    S = per_sess.groupby(["animal", "pos", "epoch"]).agg(
        sessions=("label", "nunique"), eng_daq=("eng_daq", "sum"), eng_lp=("eng_lp", "sum"),
        maps_daq=("ok_daq", "sum"), maps_lp=("ok_lp", "sum")).reset_index()
    S.to_csv(out / "lp_trial_counts_summary.csv", index=False)
    order = ["pre", "acute", "subacute", "chronic"]
    pos_order = ["far_L", "close_L", "far_center", "close_center", "close_R", "far_R"]
    pd.set_option("display.width", 240)
    print(f"=== engaged trials (lick within max_rt) and sessions clearing the {MIN_TRIALS_PER_CLASS}-trial lick-map floor,"
          " DAQ contact vs LP attempts (August pre + post)")
    tab = S.set_index(["pos", "animal", "epoch"])[["sessions", "eng_daq", "eng_lp", "maps_daq", "maps_lp"]]
    tab = tab.unstack("epoch").reindex(columns=order, level=1)
    tab = tab.reindex(pos_order, level=0)
    print(tab.to_string())
    print("\n=== category agreement (all trials): rows DAQ, columns LP")
    print(pd.crosstab(T.cat_daq, T.cat_lp).to_string())
    print(f"\n-> {out / 'lp_trial_counts_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
