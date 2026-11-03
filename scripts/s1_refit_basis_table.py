"""Imaging page section 1 per-animal table on the SPLIT Allen basis (Priya 11-03: "re-run on split basis").

    PYTHONPATH=. python scripts/s1_refit_basis_table.py [--jobs 6] [labels ...]

The section-1 table (within-session refit decoder, pre-cue / post-cue, per animal x epoch) was computed on PLAIN Allen
areas (`scripts/slowband_decode_check.py --all`, 10-20: `filter_acausality_test.roi_signal` + `decode`). The 10-27 late
decision (DECISIONS; `iti_licks` AREA BASIS) made Allen areas WITH the ALM / tjM1 split the primary decoder basis, plain
areas a robustness row. This re-runs the same decoder (`filter_acausality_test.decode`: Allen-area features, block CV,
C = 0.5, chance 1/6; same sessions, trials and windows) on three feature sets built from ONE load of the adopted SVTcorr:
  plain_orig   `roi_signal` exactly as the table (whole native brain mask, >= 20 px per area): reproduces the table
  plain_v8     the split basis's masks (eroded stat mask + the v8 lateral thin rim, `iti_licks.roi_masks`), every
               area but cortex, whole MOs / MOp, ALM / tjM1 out (`iti_licks.area_index(.., "plain")`)
  split        same masks; whole MOs / MOp out, ALM + tjM1 + MOs_rest / MOp_rest in (`area_index(.., "split")`); the rest
               areas are the parent's pixels outside the sub-ROI (= `iti_licks.with_rest`'s (parent - f sub) / (1 - f))
plain_v8 vs split differs ONLY in which areas enter, so it isolates the split; plain_orig vs plain_v8 is the mask change.
ROI version = `iti_licks.ROI_VERSION` (8, the final ROIs). Needs this branch (the v8 ROIs are not on main).
TRIALS (Priya 11-03: "use working trials both pre- and post-stroke"): every set is scored on WORKING trials (licking +
no-lick trials outside the terminal quit period; `decode_lick_working`, the rule of the LocaNMF refit) and, for
comparison, on LICKING trials only (`F.decode`, the table's rule until 11-03, which needs all 6 positions among licking
trials -- so the acute sessions with no far-contra lick (PS92_0818, PS94_0817, PS94_0820) had no value).
CAMERA LICK GATE (Priya 11-03: ENL / cue windows on working trials AND with a camera lick gate): run with
WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_ENL_WINDOW_S=2.0 -- the LP video lick train replaces DAQ contact everywhere
(the `lick` column = trials with a camera lick attempt within max_rt, hit or incomplete; the lick-free pre-cue window
and the quit-period call read video licks too); ENL pinned at the table's 2 s (LP default 1.0 s). Output tagged.
Sessions: curated, from 08-06 (as `slowband_decode_check --all`). Output -> <run>/imaging_encoding/qc/s1_refit_basis.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

EPS = ("pre", "acute", "subacute", "chronic")
SETS = ("plain_orig", "plain_v8", "split")


def basis_signals(label: str, SVTc: np.ndarray) -> dict:
    """{set: (signal (n_feat, T), names)} for the v8-mask feature sets of one session (SVTc: (K, T) adopted)."""
    import iti_licks as il

    from wfield_local import combined_encoding as ce
    _V, _ft, Ub, flat, parea, names, shape, pstat = ce._load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    for src, dst, h in il._rest_list():                       # rest = parent pixels outside the sub-ROI
        par, sub = M.get(f"{src}_{h}"), M.get(f"{dst}_{h}")
        if par is not None and sub is not None:
            M[f"{src}_rest_{h}"] = par & ~sub
    out = {}
    for basis in ("plain", "split"):
        cols = [n for n in M if M[n].any()]
        keep = [cols[i] for i in il.area_index(cols, basis)]
        Ra = np.stack([Ub[M[n]].mean(0) for n in keep], 0).astype(np.float64)       # (n_feat, K)
        out[f"{basis}_v8"] = (Ra @ SVTc, keep)
    out["split"] = out.pop("split_v8")
    return out


def _cv_acc(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """`filter_acausality_test.decode`'s scoring exactly: all 6 positions required, standardised logistic C = 0.5,
    GroupKFold(min(5, n blocks)) by position block, accuracy of the cross-validated predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local.filter_acausality_test import DISPLAY_ORDER
    ng = min(5, int(np.unique(g).size))
    if ng < 2 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    return float(accuracy_score(y, cross_val_predict(clf, X, y, cv=GroupKFold(ng), groups=g)))


def decode_lick_working(s: dict, sig: np.ndarray, regs, align: str) -> dict:
    """{lick, working, n_lick, n_work}: the table's decoder on LICKING trials (= `F.decode`, engaged = a lick within
    max_rt) and on WORKING trials (Priya 11-03: working both pre and post) = licking + no-lick trials outside the
    terminal quit period -- the `grant_kit._class_select("working")` rule of the LocaNMF refit (Figs 1.1 / 1.2): quit
    period = `precue_engagement_states.engagement_gate` on this session's trial order / responded / position, no-lick
    block ids = `grant_kit._runs_to_blocks` (runs of one position; negative ids, never colliding with the real ones)."""
    from types import SimpleNamespace

    from wfield_local import filter_acausality_test as F
    from wfield_local.grant_kit import _runs_to_blocks
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.locanmf_position_decoder import _trial_features
    from wfield_local.precue_engagement_states import engagement_gate
    args = SimpleNamespace(source="roi", align=align, baseline="none", pre_s=1.0, post_s=2.0, fs=F.FS,
                           max_rt=F._max_rt())
    X, y, g, Xn, yn, _r, ie, inl = _trial_features(s, args, signal=sig, feat_region=regs, with_indices=True)
    out = {"lick": _cv_acc(X, y, g), "n_lick": len(y)}
    name = np.array([POSITION_NAMES.get(int(v), str(v)) for v in np.concatenate([y, yn])])
    order = np.concatenate([ie, inl])
    o = np.argsort(order, kind="stable")
    quit_ = np.zeros(len(order), bool)
    from wfield_local import trial_classes as _tc
    # the "working" column here is the LABELLED quit-tail robustness row (the camera columns follow from the snapshot)
    with _tc.daq_quit_rule("s1_refit_basis_table: labelled quit-tail robustness row"):
        quit_[o] = engagement_gate(order[o], np.r_[np.ones(len(ie), bool), np.zeros(len(inl), bool)][o], name[o])
    keep = ~quit_[len(ie):]
    if len(yn) and keep.any():
        bn = _runs_to_blocks(np.zeros(len(yn), int), name[len(ie):])
        Xw, yw, gw = np.vstack([X, Xn[keep]]), np.r_[y, yn[keep]], np.r_[np.asarray(g, np.int64), bn[keep]]
    else:
        Xw, yw, gw = X, y, g
    out.update(working=_cv_acc(Xw, yw, gw), n_work=len(yw))
    # CAMERA-DEFINED gates (Priya 11-03, the new default): the shared A4 classes (window cue_stop for BOTH columns), read
    # from the snapshot per DAQ cue; every row (licking or not) is kept iff its cue is in the gate
    snap = class_snapshot(s["label"])                     # per DAQ cue: working / camera_lick flags (main, A4)
    rows = np.r_[ie, inl].astype(int)
    Xa, ya = np.vstack([X, Xn]) if len(yn) else X, np.r_[y, yn]
    ga = np.r_[np.asarray(g, np.int64), _runs_to_blocks(np.zeros(len(yn), int), name[len(ie):])]
    for gate in ("working", "camera_lick", "working_pl_contact", "working_pl_lick", "working_pl_close"):
        m = snap[gate].reindex(rows).fillna(False).to_numpy(bool) if snap is not None else np.zeros(len(rows), bool)
        out[f"cam_{gate}"] = _cv_acc(Xa[m], ya[m], ga[m]) if m.any() else np.nan
        out[f"n_cam_{gate}"] = int(m.sum())
    return out


def class_snapshot(label: str) -> pd.DataFrame | None:
    """This session's rows of the shared class SNAPSHOT (`scripts/s1_trial_class_snapshot.py` on main: the A4 classes,
    window cue_stop, matched to DAQ cue indices, gate flags set; commit in the .json beside it), indexed by cue_idx.
    Path from WFIELD_S1_CLASSES (default <run>/imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz). None = no rows
    (camera arms then empty). The snapshot is read, never re-derived, so this branch carries no copy of the rules."""
    import os

    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    f = os.environ.get("WFIELD_S1_CLASSES") or str(oo.encoding_root(oc.RUN, PathResolver()) / "qc"
                                                     / "s1_trial_classes_cue_stop.csv.gz")
    if f not in _SNAP:
        _SNAP[f] = pd.read_csv(f)
    d = _SNAP[f]
    d = d[d.label == label]
    return d.set_index("cue_idx") if len(d) else None


_SNAP: dict = {}


def session_job(label: str) -> dict:
    """Module-level worker: refit decode, pre-cue + post-cue, licking and working trials, for each feature set."""
    import glob

    from wfield_local import config
    from wfield_local import filter_acausality_test as F
    s = next(x for x in F.SESSIONS if x["label"] == label)
    res = s["mc"] + "/wfield_local_results"
    ad = glob.glob(res + "/allen_aligned_affine8v1")[0]
    # the ONE hemo switch (config.hemo_variant; rebuild 11-03): was hemo_{hv.ADOPTED}, the same path by default
    c = np.load(config.svtcorr_path(s["mc"])).astype(np.float64)
    sig, regs = F.roi_signal(ad, c.astype(np.float32))
    feats = {"plain_orig": (sig, regs), **{k: (v, np.arange(len(n))) for k, (v, n) in basis_signals(label, c).items()}}
    out = {"label": label}
    for k, (x, r) in feats.items():
        out[f"{k}_nfeat"] = int(x.shape[0])
        for align, w in (("precue", "pre"), ("cue", "post")):
            d = decode_lick_working(s, x, r, align)
            out[f"{k}_{w}"] = d["lick"]                  # licking trials (= the table's rule until 11-03)
            out[f"{k}_work_{w}"] = d["working"]          # working trials (primary from 11-03)
            out[f"n_lick_{w}"], out[f"n_work_{w}"] = d["n_lick"], d["n_work"]
            out[f"{k}_camwork_{w}"] = d["cam_working"]       # camera-defined working (DEFAULT from 11-03)
            out[f"{k}_camlick_{w}"] = d["cam_camera_lick"]   # camera lick gate (contact + incomplete)
            out[f"n_camwork_{w}"], out[f"n_camlick_{w}"] = d["n_cam_working"], d["n_cam_camera_lick"]
            for pl in ("contact", "lick", "close"):            # + end-of-session cutoff (snapshot docstring)
                out[f"{k}_campl{pl}_{w}"] = d[f"cam_working_pl_{pl}"]
                out[f"n_campl{pl}_{w}"] = d[f"n_cam_working_pl_{pl}"]
    print("  {:10s} ".format(label) + "  ".join(f"{k} {v:.3f}" for k, v in out.items()
                                                  if k.endswith(("_pre", "_post")) and not k.startswith("n_")),
          flush=True)
    return out


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import filter_acausality_test as F
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    out = oo.encoding_root(oc.RUN, PathResolver()) / "qc"
    from wfield_local import lp_licks
    out = lp_licks.output_dir(out)       # camera-lick (LP) runs write under <parent>/<tag>/qc (writeguard rule)
    writeguard.assert_writable(out)
    Path(out).mkdir(parents=True, exist_ok=True)
    cur = set(ak.curated_labels())
    labels = a.labels or [s["label"] for s in F.SESSIONS if s["label"][-4:] >= "0806" and s["label"] in cur]
    got, fail = ak.fan_sessions(labels, session_job, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    D = pd.DataFrame([r for _lab, r in got]).sort_values("label")
    D["animal"] = D.label.str.split("_").str[0]
    D["epoch"] = D.label.map(epochs.epoch_of)
    f = Path(out) / ("s1_refit_basis" + ("_subset" if a.labels else "") + ".csv")   # the run tag is in the directory
    D.to_csv(f, index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    for trials, tag in (("CAMERA WORKING (jaw or any camera lick)", "camwork_"),
                        ("CAMERA WORKING, to the last CONTACT", "campl" + "contact_"),
                        ("CAMERA WORKING, to the last CAMERA LICK", "campl" + "lick_"),
                        ("CAMERA WORKING, to the last lick at the CLOSE ipsi / mid spouts", "campl" + "close_"),
                        ("CAMERA LICK GATE (contact + incomplete)", "camlick_"),
                        ("QUIT-TAIL WORKING (lick source of this run)", "work_"),
                        ("LICKING (lick source of this run)", "")):
        for w in ("pre", "post"):
            cols = [f"{k}_{tag}{w}" for k in SETS]
            t = D.groupby(["animal", "epoch"])[cols].mean().unstack("epoch")
            print(f"\n== {trials} trials, {w}-cue, per animal x epoch (session means), chance 0.167")
            print(t.reindex(columns=list(EPS), level=1).round(3).to_string())
    print("\nsessions with a CAMERA-WORKING split post-cue value per animal x epoch:\n",
          D[D.split_camwork_post.notna()].groupby(["animal", "epoch"]).size().unstack().reindex(columns=list(EPS)))
    print("\nsessions per animal x epoch:\n", D.groupby(["animal", "epoch"]).size().unstack().reindex(columns=list(EPS)))
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
