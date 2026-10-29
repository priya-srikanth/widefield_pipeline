"""ENL lick-free census, DAQ contact vs LP licks (Priya 10-21: "re-evaluate our ENL gate using LP_licks, to see how we
need to redefine 'lick-free' ENL where there are no LP licks detected (... we'll have to decrease the time window but
let's see)").

The pre-cue (ENL) window today is lick-free by DAQ CONTACT only: `precue_window_start` keeps the window ending at the
cue if no lick falls in it, else slides it to the latest clean gap after the spout strobe, else drops the trial. The
ENL restarts on contact, so contact-free is nearly guaranteed -- but LP shows NON-contact licks there. This applies THE
SAME RULE (the function itself, in ms) to the DAQ lick train and to the LP train (cam4 `licks_all` every context + cam1
events), for window lengths 2.0 / 1.5 / 1.0 / 0.5 s, per trial:
    fixed   the window ending at the cue is clean (the common case)
    kept    a clean window exists (fixed or slid back, bounded by the strobe) -> the trial stays in the ENL arm
NB cam1 events exist only post-cue, so the LP pre-cue train is cam4.

Output: <run>/imaging_encoding/qc/enl_lp_census.csv (per trial) and _summary.csv (animal x position x epoch x source x
window). Prints the per-animal table (rule 8).
Usage: PYTHONPATH=. python scripts/enl_lp_census.py [--jobs 4]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

WINDOWS_MS = (2000, 1500, 1000, 500)


def one(label):
    from wfield_local import config, epochs, lp_licks
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_position_decoder import precue_window_start
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _load_licks
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue
    s = next(x for x in config.load_sessions() if x["label"] == label)
    try:
        lp_t, _ = lp_licks.lp_lick_times(label)
    except lp_licks.LPUnavailable:
        return []
    cue = _load_cue(s["h5"])
    sr = float(cue["sample_rate_hz"])
    lk = _load_licks(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    if lk.get("lick_source") != "daq":
        raise RuntimeError("census must read DAQ contact: unset WIDEFIELD_LICK_SOURCE")
    c_ms = np.round(np.asarray(cue["cue_samples"]) / sr * 1e3).astype(np.int64)
    ss = np.asarray(cue["strobe_samples"])
    j = np.searchsorted(ss, cue["cue_samples"], side="right") - 1
    st_ms = np.where(j >= 0, np.round(ss[np.clip(j, 0, None)] / sr * 1e3), np.nan)
    trains = {"daq": np.sort(np.round(np.asarray(lk["lick_samples"]) / sr * 1e3).astype(np.int64)),
              "lp": np.sort(np.round(lp_t * 1e3).astype(np.int64))}
    codes = classify_cues_with_backup(s, cue)
    ep = epochs.epoch_of(label)
    rows = []
    for k, (c0, st, code) in enumerate(zip(c_ms, st_ms, codes)):
        if code < 0:
            continue
        r = {"label": label, "trial": k, "pos": POSITION_NAMES.get(int(code), str(code)), "epoch": ep,
             "strobe_to_cue_ms": float(c0 - st) if np.isfinite(st) else np.nan}
        for src, L in trains.items():
            for w in WINDOWS_MS:
                fixed = not np.any((L >= c0 - w) & (L < c0))
                r[f"{src}_fixed_{w}"] = fixed
                r[f"{src}_kept_{w}"] = fixed or precue_window_start(int(c0), st, L, w) is not None
        rows.append(r)
    return rows


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    labels = sorted(ak.curated_labels())
    got, fail = ak.fan_sessions(labels, one, jobs=a.jobs)
    for it, err in fail:
        print("FAILED", it, err)
    T = pd.DataFrame([r for _, rr in got for r in rr])
    T["animal"] = T.label.str.split("_").str[0]
    T["baseline"] = np.where(T.epoch != "pre", "post", np.where(T.label.str[-4:] >= "0801", "pre_aug", "pre_june"))
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc"
    assert_writable(out)
    T.to_csv(out / "enl_lp_census.csv", index=False)
    cols = [c for c in T.columns if c.startswith(("daq_", "lp_"))]
    S = T.groupby(["animal", "pos", "epoch"])[cols].mean().reset_index()
    S.to_csv(out / "enl_lp_census_summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 240)
    order = ["pre", "acute", "subacute", "chronic"]
    Tp = T[T.baseline != "pre_june"]
    for what in ("kept", "fixed"):
        print(f"\n=== share of trials whose ENL window is LICK-FREE ({what}), August pre + post, animal x epoch")
        tab = Tp.groupby(["animal", "epoch"])[[f"{s}_{what}_{w}" for s in ("daq", "lp") for w in WINDOWS_MS]].mean()
        print(tab.reindex(order, level=1).round(3).to_string())
    print("\n=== LP kept, by position (August pre / acute / chronic), 2.0 s and 1.0 s")
    tab = Tp[Tp.epoch.isin(["pre", "acute", "chronic"])].groupby(["pos", "animal", "epoch"])[
        ["daq_kept_2000", "lp_kept_2000", "lp_kept_1000", "lp_kept_500"]].mean().unstack("epoch")
    print(tab.round(2).to_string())
    print(f"\n-> {out / 'enl_lp_census_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
