"""SNAPSHOT of the shared trial classes per DAQ cue for the imaging section 1 table (and any imaging decoder that runs on
another branch): one row per DAQ cue of every curated imaging session (from 08-06).

    python -m scripts.s1_trial_class_snapshot [--window cue_stop] [--jobs 8]

Why a snapshot: the classes live on main (`wfield_local.trial_classes`, A4 28f2a632; `jaw_only_incomplete_figs`
session_classes / cue_classes / GATES), while the split-basis decoder runs on branch analysis/attempt-coding (its v8 ROIs
are not on main). Writing the classes once, already matched to DAQ cue indices and with the gate flags set, means the
decoders on either branch only INDEX this table -- no second copy of the classification or the matching -- and every
number is tied to the commit recorded beside it.
Window: "cue_stop" for BOTH the ENL / pre-cue and the cue columns (Priya 11-03: working = responded to THIS trial's cue;
the ENL window is lick-free by its own rule).
Output -> <run>/imaging_encoding/qc/s1_trial_classes_<window>.csv.gz (label, cue_idx, cue_s, cls, working, camera_lick,
working_prelast, has_cam1; cls "unknown" + both gates False where the video has no trial for the cue) + a .json sidecar (commit, window,
coverage). A session with no video (PS93_0813) is written with cls "no_video": camera rows drop it, DAQ rows keep it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def session_job(job) -> list[dict]:
    """Module-level worker: (label, window) -> rows per DAQ cue."""
    label, window = job
    from scripts import jaw_only_incomplete_figs as J
    from wfield_local import filter_acausality_test as F
    from wfield_local.locanmf_position_decoder import _load_cue_events
    s = next(x for x in F.SESSIONS if x["label"] == label)
    cue = _load_cue_events(s["h5"])
    cs = np.asarray(cue["cue_samples"], float) / float(cue["sample_rate_hz"])
    T = J.session_classes(label, window=window)
    if T is None:
        return [dict(label=label, cue_idx=i, cue_s=t, cls="no_video", cls_final="no_video", position="", working=False, working_raw=False,
                     stopped=False, working_pl_lick=False, working_pl_contact=False, working_pl_close=False,
                     camera_lick=False,
                     has_cam1=False) for i, t in enumerate(cs)]
    from wfield_local import trial_classes as tc
    c = J.cue_classes(T, cs)
    pos = J.cue_classes(T.assign(cls=T.position), cs)           # the same time matcher, carrying the spout position
    matched = J.cue_classes(T.assign(cls="matched"), cs) == "matched"
    # ONE CLASSIFIER (T9, third review): the session-level rules -- end at the last camera lick within the covered span,
    # sustained stop (>= 6 no_response / jaw_only starting >= 5 min after the first cue, Priya 11-04; earlier runs
    # early_run),
    # coverage -- come from trial_classes.classify_session WITH the cue times (the 5-min rule), never re-derived here.
    # `working` = the PRIMARY (end "lick"); working_pl_<anchor> = the robustness anchors.
    # stopped-time licks (Priya 11-04): the guarded rules get the cue train's span lick counts
    span = (tc.cue_span_licks(label, T, cs) if tc.stopped_rule() in tc.GUARDED_RULES
            and (tc.stop_lick_rule(), tc.stop_tail_start()) != ("none", "lick_trial") else None)
    fin = {e: tc.classify_session(c, matched, pos, end=e, cue_s=cs, span_licks=span) for e in ("lick", "contact", "close")}
    w, lk = tc.in_gate(fin["lick"], "working"), tc.in_gate(c, "camera_lick")
    wpl = {e: tc.in_gate(f, "working") for e, f in fin.items()}
    stop = np.isin(fin["lick"], list(tc.STOPPED_CLASSES))
    h = bool(T.has_cam1.iloc[0]) if len(T) else False
    return [dict(label=label, cue_idx=i, cue_s=t, cls=c[i], cls_final=fin["lick"][i], position=pos[i],
                 working=bool(w[i]), working_raw=bool(J.in_gate(c[i:i + 1], "working")[0]), stopped=bool(stop[i]),
                 camera_lick=bool(lk[i]), has_cam1=h,
                 **{f"working_pl_{e}": bool(v[i]) for e, v in wpl.items()}) for i, t in enumerate(cs)]


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("s1_trial_class_snapshot")   # unrouted share writer: refuses in a non-default run mode
    from wfield_local import analysis_kit as ak
    from wfield_local import filter_acausality_test as F
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="cue_stop", choices=("cue_stop", "trial"))
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    out = oo.encoding_root(oc.RUN, PathResolver()) / "qc"
    writeguard.assert_writable(out)
    cur = set(ak.curated_labels())
    labels = [s["label"] for s in F.SESSIONS if s["label"][-4:] >= "0806" and s["label"] in cur]
    got, fail = ak.fan_sessions([(lab, a.window) for lab in labels], session_job, jobs=a.jobs, key=lambda j: j[0])
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    D = pd.DataFrame([r for _j, rows in got for r in rows])
    # END-OF-SESSION CUTOFFS on top of `working`. PRIMARY = working_pl_lick (Priya 11-03: "last camera lick cutoff sounds
    # good"); the other two are robustness rows. They drop the late,
    # smaller jaw-only trials of the sated tail (e3, QUIT_TAIL_JAW_ONLY_2026-11-03: 0.9 / 3.8 / 1.9 / 1.1 % of working
    # pre / acute / subacute / chronic). THREE ANCHORS, because Priya 11-03: "if we drop after the last lick, acutely if
    # the last blocks are far contra or far mid we may drop real data" -- acute far-contra contact is ~5 %, so a CONTACT
    # anchor can cut a final far block of genuine attempts:
    #   working_pl_contact  at or before the session's last CONTACT cue
    #   working_pl_lick     at or before the last CAMERA LICK (contact or incomplete) -- far-spout attempts survive
    #   working_pl_close    at or before the last camera lick at the REFERENCE spouts (close ipsi / close mid), the
    #                       quit-tail idea: those spouts are barely affected, so the anchor marks when the animal
    #                       stopped, not where the last block happened to be
    # The rule lives in `wfield_local.trial_classes.in_working` (e3, b506c1e3); this snapshot only records it.
    # working_pl_<anchor> come from trial_classes.classify_session in session_job (one classifier, T9); no-video rows
    for name in ("contact", "lick", "close"):
        D[f"working_pl_{name}"] = D[f"working_pl_{name}"].fillna(False).astype(bool) if f"working_pl_{name}" in D \
            else False
    f = out / f"s1_trial_classes_{a.window}.csv.gz"
    D.to_csv(f, index=False, float_format="%.6f")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "wfield_local/trial_classes.py",
                                 "scripts/jaw_only_incomplete_figs.py"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1]).stdout.strip())
    cov = D.groupby("label").agg(n=("cls", "size"), unknown=("cls", lambda c: float((c == "unknown").mean())),
                                 working=("working", "mean"), camera_lick=("camera_lick", "mean"))
    meta = {"window": a.window, "commit": commit, "class_code_dirty": dirty, "n_sessions": int(D.label.nunique()),
            "no_video": sorted(D.loc[D.cls == "no_video", "label"].unique().tolist()),
            "low_match": cov[cov.unknown > 0.05].unknown.round(3).to_dict()}
    Path(str(f).replace(".csv.gz", ".json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    from wfield_local import epochs
    D["epoch"] = D.label.map(epochs.epoch_of)
    P = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
    for name in ("contact", "lick", "close"):
        drop = D.working_raw.fillna(False).astype(bool) & ~D[f"working_pl_{name}"]
        t = D[drop].groupby(["epoch", "position"]).size().unstack().reindex(columns=P).fillna(0).astype(int)
        t["all"] = t.sum(1)
        t["share_of_working"] = (t["all"] / D[D.working].groupby("epoch").size().reindex(t.index)).round(3)
        n_inc = int((drop & (D.cls == "incomplete")).sum())
        print(f"\n== cutoff at the last {name}: working trials dropped (incomplete among them: {n_inc})")
        print(t.reindex(["pre", "acute", "subacute", "chronic"]).to_string())
    print(D[D.cls != "no_video"].cls.value_counts(normalize=True).round(4).to_string())
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
