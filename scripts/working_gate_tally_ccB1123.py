"""Camera working period vs the behavior-pipeline engagement flag, six-position cohort (CC-B 11-23, read-only).

Priya 11-23: every six-position analysis uses the camera WORKING PERIOD (final class of
`imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz` not in `trial_classes.OUT_OF_WORKING_PERIOD`, trials without
video dropped). Some outputs still use the behavior pipeline's per-trial `engaged` column
(`spout_behavior.session_metrics` -> `reference_engagement`, judged at the reference spouts close_L / close_center).

Writes, for every session of the class table:
  1. ``gate_tally_trials.csv.gz``: one row per cued trial: label, animal, epoch, position, cls (four-camera class),
     cls_detail, cls_final, in_wp (camera working period), in_flag (behavior `engaged`), on_grid (FaceRhythm bundle
     trial grid = the FR "engaged trials"), fr_own8 (drawn for the per-session own8 fit), fr_master (drawn for the
     frozen master pre fit).
  2. ``gate_tally_by_class.csv``: counts per epoch x animal x set (flag_not_wp, wp_not_flag, both) x four-camera class.
  3. ``fr_fit_rows_vs_wp.csv``: per FR task (key), the drawn fit trials (own8 and master) and how many fall outside the
     working period, by final class; plus between-trial 5 Hz rows that fall in a stretch whose latest cue is outside
     the working period.

The FR fit rows are READ from the plan the fits were run from (`plan.json`: `session_rows[key]` for `fitsession`
/ own8, `fit[animal]` for the master fit; `validate.cmd_fitsession` and `fit.build` read exactly these), so no draw
is recomputed.

    PYTHONPATH=. python scripts/working_gate_tally_ccB1123.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASS_TABLE = ("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/"
               "qc/s1_trial_classes_cue_stop.csv.gz")
NOT_WP = ("stopped", "stopped_licking", "early_run", "after_end", "unknown", "no_video")
MANU = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}
MATCH_S = 0.25


def nearest(x: np.ndarray, ref: np.ndarray, tol: float = MATCH_S) -> np.ndarray:
    """Index into sorted ``ref`` of the nearest value within ``tol`` of each ``x``, -1 otherwise. Pure."""
    if len(ref) == 0:
        return np.full(len(x), -1)
    hi = np.clip(np.searchsorted(ref, x), 0, len(ref) - 1)
    lo = np.clip(hi - 1, 0, None)
    j = np.where(np.abs(ref[lo] - x) <= np.abs(ref[hi] - x), lo, hi)
    return np.where(np.abs(ref[j] - x) <= tol, j, -1)


def flag_cues(animal: str, date: str) -> tuple[np.ndarray | None, str]:
    """DAQ cue times of the trials the behavior pipeline flagged engaged (`orofacial_cohort._reference_keep`, the
    reader the FR bundle and the orofacial gates use), or None."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local.paths import PathResolver
    sd = Path(PathResolver().root("orofacial")) / oc.RUN / "sessions" / animal / date
    if not (sd / "windows.npz").exists():
        return None, "no windows.npz"
    keep, how = oc._reference_keep(animal, date, sd)
    if keep is None:
        return None, how
    cue = oc._cue_table(sd)
    return np.sort(cue.loc[cue.trial_id.isin(keep), "cue_daq"].to_numpy(float)), how


def main(argv=None) -> int:
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.face_rhythm.factors.files import out_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = fro.factor_dir()
    od = out_dir(sd)
    pl = json.loads((sd / "plan.json").read_text())
    tasks = {t["key"]: t for t in pl["tasks"]}
    C = pd.read_csv(CLASS_TABLE, usecols=["label", "cue_s", "cls", "cls_detail", "cls_final", "position"])
    C["in_wp"] = ~C.cls_final.astype(str).isin(NOT_WP)
    for c in ("in_flag", "on_grid", "fr_own8", "fr_master"):
        C[c] = False
    C["flag_how"] = ""
    C["epoch"] = ""
    # epochs: FR plan tasks, else the cohort epoch table
    from wfield_local import config
    ep = {}
    for t in tasks.values():
        ep[f"{t['animal']}_{t['date'][4:]}"] = t["epoch"]
    labels = sorted(C.label.unique())
    for lab in labels:
        if lab not in ep:
            try:
                ep[lab] = config.epoch_of(lab)
            except Exception:  # noqa: BLE001
                ep[lab] = "?"
    C["epoch"] = C.label.map(ep)
    C["animal"] = C.label.str[:4]
    year = "2026"
    rows_fr = []
    for lab in labels:
        m = (C.label == lab).to_numpy()
        cue_all = C.loc[m, "cue_s"].to_numpy(float)
        o = np.argsort(cue_all)
        idx_sorted = np.flatnonzero(m)[o]
        cs = cue_all[o]
        animal = lab[:4]
        date = year + lab[5:]
        fc, how = flag_cues(animal, date)
        if fc is not None:
            j = nearest(fc, cs)
            C.loc[idx_sorted[j[j >= 0]], "in_flag"] = True
        C.loc[m, "flag_how"] = how
        for key in [k for k in tasks if k.startswith(f"{animal}_{date}_")]:
            f = od / f"{key}_trials.npz"
            if not f.exists():
                continue
            with np.load(f, mmap_mode="r") as z:
                gc = np.asarray(z["cue_daq_s"], float)
            jg = nearest(gc, cs)
            C.loc[idx_sorted[jg[jg >= 0]], "on_grid"] = True
            own = pl["session_rows"].get(key, {})
            own_tr = np.asarray(own.get("trials", {}).get(key, []), int)
            mas_tr = np.asarray(pl["fit"].get(animal, {}).get("trials", {}).get(key, []), int)
            rec = {"key": key, "label": lab, "animal": animal, "mouse": MANU[animal], "epoch": tasks[key]["epoch"],
                   "n_grid": len(gc), "n_grid_unmatched": int((jg < 0).sum())}
            for nm, tr in (("own8", own_tr), ("master", mas_tr)):
                jj = jg[tr] if len(tr) else np.array([], int)
                rec[f"{nm}_trials"] = len(tr)
                rec[f"{nm}_unmatched"] = int((jj < 0).sum())
                ok = jj[jj >= 0]
                C.loc[idx_sorted[ok], f"fr_{nm}"] = True
                fin = C.loc[idx_sorted[ok], "cls_final"].astype(str)
                rec[f"{nm}_out_wp"] = int(fin.isin(NOT_WP).sum())
                for c in NOT_WP:
                    rec[f"{nm}_out_{c}"] = int((fin == c).sum())
            # between-trial (and arrival) 5 Hz rows: in a stretch whose latest cue is outside the working period?
            sf = od / f"{key}_session.npz"
            with np.load(sf, mmap_mode="r") as z:
                t5 = np.asarray(z["t_daq_s"], float)
            wp_sorted = C.loc[idx_sorted, "in_wp"].to_numpy(bool)
            fl_sorted = C.loc[idx_sorted, "in_flag"].to_numpy(bool)
            last_wp = cs[wp_sorted].max() if wp_sorted.any() else -np.inf
            for nm, picks in (("own8", own), ("master", pl["fit"].get(animal, {}))):
                for kind in ("between", "arrival"):
                    b = np.asarray(picks.get(kind, {}).get(key, []), int)
                    b = b[b < len(t5)]
                    tt = t5[b]
                    k = np.searchsorted(cs, tt, side="right") - 1
                    prev_out = (k >= 0) & ~wp_sorted[np.clip(k, 0, None)]
                    rec[f"{nm}_{kind}_rows"] = len(b)
                    rec[f"{nm}_{kind}_in_stop"] = int(prev_out.sum())
                    rec[f"{nm}_{kind}_in_flagstop"] = int(((k >= 0) & ~fl_sorted[np.clip(k, 0, None)]).sum())
                    rec[f"{nm}_{kind}_after_last_wp_cue"] = int((tt > last_wp + 6.0).sum())
            rows_fr.append(rec)
        print(lab, how, flush=True)
    C["mouse"] = C.animal.map(MANU)
    C.to_csv(out / "gate_tally_trials.csv.gz", index=False)
    pd.DataFrame(rows_fr).to_csv(out / "fr_fit_rows_vs_wp.csv", index=False)
    V = C[C.cls_final.astype(str) != "no_video"]
    V = V.assign(set=np.select([V.in_flag & ~V.in_wp, V.in_wp & ~V.in_flag, V.in_wp & V.in_flag],
                               ["flag_not_wp", "wp_not_flag", "both"], "neither"))
    T = V.groupby(["epoch", "mouse", "set", "cls"]).size().rename("n").reset_index()
    T.to_csv(out / "gate_tally_by_class.csv", index=False)
    print(V.groupby(["epoch", "set"]).size().unstack(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
