"""QC tables of the camera trial gate (Priya 11-03 decisions 1 and 6, and the FINAL no-lick window decision).
Read-only on the data; writes four CSVs into <run>/imaging_encoding/qc/.

1. camera_coverage.csv -- VIDEO COVERAGE per session (N6): cues matched to a video trial, covered fraction, partial
   coverage, uncovered tail (then no end-at-last-lick cutoff; stop runs counted per covered segment:
   `trial_classes.classify_session`). Partially covered sessions (PS92_0908, 82.5 %) STAY in the camera-gated
   analyses for their covered trials; `lp_licks.stamp` names them on every camera-gate figure from this table.
2. camera_contact_nolick.csv -- per animal x epoch: DAQ no-lick trials (the DAQ detector's "undetected",
   `nolick_decoder.categorize` under the DAQ rule) whose cue_stop camera class is "contact". Since the FINAL decision
   (one definition: cue -> trial stop, contacts from cue + 70 ms, the contact table) such a trial is a CONTACT trial in
   the no-lick analysis too, so this table is the CHECK that the two DAQ lick paths agree; every disagreement is
   listed with its cause (camera_contact_nolick_trials.csv):
     late_after_response_window   the contact falls after the categorisation's response window, before the stop
     cue_frame                    a detected lick sits IN the cue's imaging frame (`searchsorted(..., "right")`
                                  starts after it)
     refractory_merge             a detected lick lies within the detector's minimum gap before the cue (the
                                  contact continues a pre-cue onset, so no new onset after the cue). The gap is
                                  `REFRACTORY_S`: 40 ms since the DAQ detector switch (REBUILD_CODE 8.53; 0.1 s before)
     definition_mismatch          none of the above: detect_licks (thresholds, min-ILI) vs the contact table
   None of these is treated as a sensor miss: "contact" IS the DAQ spout sensor.
3. camera_stop_lengths.csv -- per animal x epoch: trials, trials with a stop > 3.6 s, their median stop (ms), the
   median stop of all trials (a trial-length difference between epochs matters for licks-per-trial readouts).

    python scripts/camera_coverage_tables.py [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

def _min_gap_s() -> float:
    """The categorisation's minimum gap between kept DAQ onsets: max(refractory, 40 ms floor) of THE DAQ lick detector
    (`config.lick_detector`, REBUILD_CODE 8.53; it was the old imaging 0.10 s refractory)."""
    from wfield_local import config
    d = config.lick_detector()
    return max(d["refractory_s"], d["min_ili_s"])


REFRACTORY_S = _min_gap_s()


def coverage_path() -> Path:
    from wfield_local import lp_licks
    return lp_licks._enc_dir() / "qc" / "camera_coverage.csv"


def contact_path() -> Path:
    return coverage_path().with_name("camera_contact_nolick.csv")


def disagreement_causes(cat, cls, cue_f, ls, fs, contact_ms, rw_s) -> np.ndarray:
    """Cause per cue of a DAQ-undetected / camera-contact disagreement ("" where they agree). See the module docstring."""
    cat, cls = np.asarray(cat, object), np.asarray(cls, object)
    ls = np.sort(np.asarray(ls, float))
    out = np.full(len(cat), "", dtype=object)
    ref = int(np.ceil(REFRACTORY_S * fs))
    for k in np.flatnonzero((cat == "undetected") & (cls == "contact")):
        c = float(cue_f[k])
        cm = float(contact_ms[k]) if k < len(contact_ms) else np.nan
        if np.isfinite(cm) and cm > 1000.0 * rw_s:
            out[k] = "late_after_response_window"
        elif np.any(ls == c):
            out[k] = "cue_frame"
        elif np.any((ls >= c - ref) & (ls < c)):
            out[k] = "refractory_merge"
        else:
            out[k] = "definition_mismatch"
    return out


def stop_lengths(T: pd.DataFrame) -> pd.DataFrame:
    """Per animal x epoch: n trials, n with stop > 3.6 s, their median stop, the median of all (ms)."""
    g = T.groupby(["animal", "epoch"])
    out = pd.DataFrame({"n_trials": g.size(),
                        "n_stop_gt_3600": g.stop_ms.apply(lambda s: int((s > 3600).sum())),
                        "median_stop_gt_3600_ms": g.stop_ms.apply(lambda s: float(s[s > 3600].median())
                                                                   if (s > 3600).any() else np.nan),
                        "median_stop_ms": g.stop_ms.median()})
    return out.reset_index()


def main(argv=None) -> int:
    from wfield_local import lp_licks as _lpl
    from wfield_local import rebuild_root as _rr
    from wfield_local import run_mode as _rm
    # coordinator 11-04: under the render profile this RUN-INDEPENDENT input (video coverage; the DAQ gate is forced
    # below; no hemo / projection / pre dependence) is written INSIDE the rebuild root, at the untagged path
    # `trial_classes.coverage_note` reads (the former seed destination). A tagged run OUTSIDE a rebuild root still refuses.
    _tagged = bool(_lpl.output_tag())
    if _tagged and not _rr.active():
        _rm.require_default("camera_coverage_tables")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--labels", nargs="+", default=None, help="only these sessions (a quick check)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default <run>/imaging_encoding/qc)")
    a = ap.parse_args(argv)
    os.environ["WIDEFIELD_TRIAL_GATE"] = "daq"        # the DAQ categories as the detector makes them (the "before")
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    from wfield_local import nolick_decoder as nd
    from wfield_local import trial_classes as TC
    from wfield_local.writeguard import assert_writable
    cov, per, causes, stops = [], [], [], []
    for s in ak.curated_sessions(a.animals):
        lab = s["label"]
        if a.labels and lab not in a.labels:
            continue
        an, ep = config.animal_of(lab), str(epochs.epoch_of(lab))
        cs = TC.daq_cue_seconds(lab)
        cov.append(TC.session_coverage(lab, cs))
        T = TC.session_trials(lab)
        if T is None:
            continue
        args = nd._args(align="cue")
        codes, cat, _blk, _rt, cue_f, _eng, ls, _st = nd.categorize(s, args, with_licks=True)
        cls = TC.cue_classes(T, cs)                                        # raw cue_stop camera class per cue
        tid = TC.cue_classes(T.assign(cls=T.trial_id), cs)
        date8 = config.date8(lab.split("_")[1])
        ct = TC.trial_contact_times(an, date8, T)
        contact_ms = np.array([ct.get(int(x), np.nan) if x != "unknown" else np.nan for x in tid], float)
        rw = float(nd.response_window_for(s))
        k = min(len(cat), len(cls))
        why = disagreement_causes(cat[:k], cls[:k], cue_f[:k], ls, args.fs, contact_ms[:k], rw)
        per.append(pd.DataFrame({"animal": an, "epoch": ep, "nolick": np.asarray(cat[:k]) == "undetected",
                                 "cls": cls[:k]}))
        causes.append(pd.DataFrame({"label": lab, "animal": an, "epoch": ep, "cue": np.arange(k), "cause": why,
                                    "contact_ms": contact_ms[:k], "response_window_s": rw})[why != ""])
        stops.append(T.assign(animal=an, epoch=ep)[["animal", "epoch", "stop_ms"]])
    qc = a.out or coverage_path().parent
    if _tagged:
        # the fixed-name tables of this run-independent input are a registered data store inside the rebuild root
        # (writeguard's tag rule exempts exactly these files)
        for n in (coverage_path().name, contact_path().name, "camera_contact_nolick_trials.csv",
                  "camera_stop_lengths.csv"):
            _rm.register_data_store(qc / n)
    assert_writable(qc / coverage_path().name)
    qc.mkdir(parents=True, exist_ok=True)
    C = pd.DataFrame(cov)
    C.to_csv(qc / coverage_path().name, index=False)
    P = pd.concat(per, ignore_index=True) if per else pd.DataFrame(columns=["animal", "epoch", "nolick", "cls"])
    N = TC.contact_nolick_counts(P.cls, P.nolick, P.animal, P.epoch)
    W = pd.concat(causes, ignore_index=True) if causes else pd.DataFrame(columns=["animal", "epoch", "cause"])
    if len(W):
        N = N.merge(W.pivot_table(index=["animal", "epoch"], columns="cause", values="cue", aggfunc="count",
                                  fill_value=0).reset_index(), on=["animal", "epoch"], how="left").fillna(0)
    N.to_csv(qc / contact_path().name, index=False)
    W.to_csv(qc / "camera_contact_nolick_trials.csv", index=False)
    S = stop_lengths(pd.concat(stops, ignore_index=True)) if stops else pd.DataFrame()
    S.to_csv(qc / "camera_stop_lengths.csv", index=False)
    print("response windows used by the no-lick categorisation (s):",
          sorted(set(W.response_window_s)) if len(W) else "(no disagreement rows)")
    print(C[C.partial].to_string(index=False) if C.partial.any() else "no partially covered session")
    print(N.to_string(index=False))
    print(S.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
