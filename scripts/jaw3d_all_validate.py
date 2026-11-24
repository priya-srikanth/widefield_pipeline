"""VALIDATION of the four-camera 3-D jaw call at ALL SIX spouts before the trial classes switch to it (CC-B 11-23; Priya's
11-23 rule: every six-position analysis on the four-camera 3-D kinematics; audit violation 1).

    python scripts/jaw3d_all_validate.py build   [--jobs 5]   # per-trial table, both jaw modes, + 3-D / 2-D evidence
    python scripts/jaw3d_all_validate.py summary              # agreement / confusion / deflection / floor tables

For every s1 session (the curated imaging set of `qc/s1_trial_classes_cue_stop.csv.gz`), `trial_classes.session_trials`
is run twice: jaw="classes" (production: cam4 4 px OR cam1 15 px, 3-D only at close_center) and jaw="3d" (the 3-D
call at all six spouts). Joined per trial with the raw columns of jaw_onsets.csv: the per-camera calls before masking,
the 3-D 95th-percentile displacement over the window (``jaw3d_disp_mm``) and over the pre-cue 500 ms (``jaw3d_rest_mm``,
the trial's own resting noise), cam4 / cam1 opening in px.

Agreement is reported on the trials where the jaw call DECIDES the class: no spout contact, no lick, no tongue seen,
not frame-gated (camera-mode class jaw_only from a jaw call, no_response or unknown). Positions are renamed by the
lesion (all four lesions LEFT: L = ipsi, R = contra); manuscript animal numbers PS92-95 = #9-#12.
Output: <review>/jaw3d_qc/ (writes only there).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

RUN = "lp_r4_heatmap_20261013__clean_v37_sync60"
QC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis") / RUN / "imaging_encoding" / "qc"
OUT = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23/jaw3d_qc")
POS = {"close_L": "near ipsi", "close_center": "near middle", "close_R": "near contra",
       "far_L": "far ipsi", "far_center": "far middle", "far_R": "far contra"}
POS_ORDER = list(POS.values())
EPOCHS = ["pre", "acute", "subacute", "chronic"]
ANIMAL_NO = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}
JCOLS = ["cam4_moved", "cam1_moved", "cam4_open_px", "cam1_open_px", "jaw3d_disp_mm", "jaw3d_rest_mm",
         "jaw3d_down_mm", "jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue", "jaw3d_source", "epoch"]


_RAW: dict = {}


def _raw_jaw() -> pd.DataFrame:
    """jaw_onsets.csv with the evidence columns (`trial_classes.jaw_table` reads only the class columns)."""
    if "J" not in _RAW:
        J = pd.read_csv(QC / "jaw_onsets.csv", low_memory=False, usecols=["session", "trial_id"] + JCOLS)
        _RAW["J"] = J.set_index(["session", "trial_id"]).sort_index()
    return _RAW["J"]


def session_job(label: str) -> pd.DataFrame | None:
    from wfield_local import trial_classes as tc
    A = tc.session_trials(label, jaw="classes")
    if A is None:
        return None
    B = tc.session_trials(label, jaw="3d")
    keep = ["trial_id", "position", "cls", "cls_detail", "n_contacts", "n_licks", "frame_gated", "jaw_judged",
            "jaw_onset_ms", "jaw_at_cue"]
    M = A[keep].merge(B[["trial_id", "cls", "cls_detail", "jaw_judged", "jaw_onset_ms", "jaw_at_cue"]],
                      on="trial_id", suffixes=("_2d", "_3d"))
    an, d = label.split("_")
    J = _raw_jaw()
    key = f"{an}_2026{d}" if len(d) == 4 else label
    j = J.loc[key].reindex(M.trial_id.to_numpy()) if key in J.index.get_level_values(0) else None
    for c in JCOLS:
        M[c] = j[c].to_numpy() if j is not None and c in j else np.nan
    M.insert(0, "label", label)
    M.insert(1, "animal", an)
    return M


def build(jobs: int) -> Path:
    from wfield_local import analysis_kit as ak
    labels = sorted(pd.read_csv(QC / "s1_trial_classes_cue_stop.csv.gz", usecols=["label"]).label.unique())
    res, fails = ak.fan_sessions(labels, session_job, jobs=jobs)
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.concat([r for _, r in res if r is not None], ignore_index=True)
    T["pos"] = T.position.map(POS)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "jaw3d_vs_2d_trials.csv.gz"
    T.to_csv(f, index=False, float_format="%.4g")
    print(f"{len(T)} trials, {T.label.nunique()} sessions -> {f}")
    return f


def _decides(T: pd.DataFrame) -> pd.Series:
    """Trials whose class the jaw call decides: no contact, no lick / tongue, not frame-gated (either mode)."""
    nolick = T.cls_detail_2d.isin(["jaw_only", "no_response", "unknown"]) & ~T.frame_gated.astype(bool)
    return nolick & T.cls_detail_3d.isin(["jaw_only", "no_response", "unknown"])


def summary() -> None:
    T = pd.read_csv(OUT / "jaw3d_vs_2d_trials.csv.gz", low_memory=False)
    T = T[T.epoch.isin(EPOCHS)]
    D = T[_decides(T)].copy()
    lines = []
    pd.set_option("display.width", 250)
    # 2-D raw call (before the close_center mask): either camera moved / both still / not judged
    c4, c1 = D.cam4_moved, D.cam1_moved
    D["call2d"] = np.where((c4 == 1) | (c1 == 1), "moved", np.where(c4.notna() | c1.notna(), "still", "none"))
    D["call3d"] = np.where(D.jaw3d_moved == 1, np.where(D.jaw3d_at_cue.astype(str).str.lower() == "true",
                                                       "at_cue", "moved"),
                           np.where(D.jaw3d_moved == 0, "still", "none"))
    # 1. class confusion per spout x epoch
    conf = D.groupby(["pos", "epoch", "cls_2d", "cls_3d"]).size().rename("n").reset_index()
    conf.to_csv(OUT / "confusion_class_pos_epoch.csv", index=False)
    tab = pd.crosstab([D.pos, D.cls_2d], D.cls_3d).reindex(POS_ORDER, level=0)
    lines += ["## class confusion (camera mode rows x 3d_all mode columns), jaw-decided trials, all epochs",
              tab.to_string(), ""]
    # agreement on the jaw-only call, judged in both
    both = D[D.cls_2d.isin(["jaw_only", "no_response"]) & D.cls_3d.isin(["jaw_only", "no_response"])
             & (D.cls_detail_2d != "tongue_at_lips")]
    rows = []
    for (p, e), g in both.groupby(["pos", "epoch"]):
        a, b = g.cls_2d == "jaw_only", g.cls_3d == "jaw_only"
        n = len(g)
        po = float((a == b).mean())
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        rows.append({"pos": p, "epoch": e, "n_judged_both": n, "jaw_only_2d": int(a.sum()), "jaw_only_3d": int(b.sum()),
                     "both": int((a & b).sum()), "2d_only": int((a & ~b).sum()), "3d_only": int((~a & b).sum()),
                     "agree": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else np.nan})
    A = pd.DataFrame(rows)
    A["pos"] = pd.Categorical(A.pos, POS_ORDER)
    A["epoch"] = pd.Categorical(A.epoch, EPOCHS)
    A = A.sort_values(["pos", "epoch"])
    A.to_csv(OUT / "agreement_pos_epoch.csv", index=False)
    lines += ["## agreement on jaw-only vs no-response (trials judged in both modes)", A.to_string(index=False), ""]
    # raw call agreement (2-D raw call before masking vs 3-D call)
    raw = pd.crosstab([D.pos, D.call2d], D.call3d).reindex(POS_ORDER, level=0)
    raw.to_csv(OUT / "confusion_rawcall_pos.csv")
    lines += ["## raw calls: 2-D (cam4 OR cam1, unmasked) x 3-D", raw.to_string(), ""]
    # 2. 3-D deflection on 2-D still vs moved
    q = (D[D.call2d != "none"].groupby(["pos", "call2d"]).jaw3d_disp_mm
         .describe(percentiles=[.1, .25, .5, .75, .9]).round(3))
    q.to_csv(OUT / "jaw3d_disp_by_2dcall.csv")
    lines += ["## 3-D jaw displacement (95th pct over cue..stop, mm) by 2-D call", q.to_string(), ""]
    # 3. pre-cue noise per spout (all trials with a 3-D value, all classes)
    R = T[T.jaw3d_rest_mm.notna()]
    rq = R.groupby("pos").jaw3d_rest_mm.describe(percentiles=[.5, .9, .95, .99]).round(3).reindex(POS_ORDER)
    rq.to_csv(OUT / "jaw3d_rest_noise_by_pos.csv")
    lines += ["## 3-D jaw pre-cue noise (95th pct of displacement over -500..0 ms, mm), all trials", rq.to_string(), ""]
    # still-trial false-positive proxy: share of 2-D-still trials whose 3-D displacement exceeds a floor
    fl = []
    for floor in (0.15, 0.20, 0.25, 0.30, 0.40):
        for p, g in D[D.call2d != "none"].groupby("pos"):
            st, mv = g[g.call2d == "still"], g[g.call2d == "moved"]
            fl.append({"floor_mm": floor, "pos": p, "still_above": round(float((st.jaw3d_disp_mm > floor).mean()), 3),
                       "moved_above": round(float((mv.jaw3d_disp_mm > floor).mean()), 3),
                       "rest_p95_above": round(float((R[R.pos == p].jaw3d_rest_mm > floor).mean()), 3)})
    F = pd.DataFrame(fl)
    F.to_csv(OUT / "floor_scan.csv", index=False)
    lines += ["## floor scan: share above the floor", F.pivot(index="pos", columns="floor_mm",
                                                              values=["still_above", "moved_above", "rest_p95_above"])
              .reindex(POS_ORDER).to_string(), ""]
    # coverage: decided trials without a 3-D call
    cov = D.groupby("pos").apply(lambda g: pd.Series({"n": len(g), "no_3d_call": int((g.call3d == "none").sum()),
                                                      "at_cue_3d": int((g.call3d == "at_cue").sum()),
                                                      "no_2d_call": int((g.call2d == "none").sum())}),
                                 include_groups=False).reindex(POS_ORDER)
    lines += ["## coverage of the decided trials", cov.to_string(), ""]
    (OUT / "validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("jaw3d_all_validate")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("build", "summary"))
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args(argv)
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    build(a.jobs) if a.step == "build" else summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
