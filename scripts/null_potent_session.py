"""Movement-null / movement-potent subspaces (Hasnain et al. 2025) per session, and where the spout-position code
lives: ENL (pre-cue) and post-cue position decoding from activity reconstructed in each subspace.

    python -m scripts.null_potent_session me PS93_0814                 # stage 1: motion energy + thresholds figure
    python -m scripts.null_potent_session analyze PS93_0814            # stage 2: labels, subspaces, decoding
    python -m scripts.null_potent_session all                          # every regime-B session, both stages

Priya, 2026-10-07 (supplying the Methods of Hasnain, Birnbaum, ..., Chandrasekaran & Economo 2025, Nat Neurosci):
stationarity from MOTION ENERGY (`wfield_local/motion_state.py`), subspaces by their joint objective
(`wfield_local/null_potent.py`). Differences from theirs, all documented there: 250 fps (3-frame median windows),
automatic valley thresholds with a figure to check them (override: configs `motion_state.thresholds`), DAQ licks
and running also count as moving, a 0.3 s post-movement buffer excluded (GCaMP decay), LocaNMF components z-scored
on an inter-trial baseline (0.7 .. 0.2 s before the spout strobe), d = min(N / 2, 20).

STAGE 1 (`me`): per camera, per-frame motion energy (99th-percentile pixel |median next 3 - median previous 3|,
ds 4) over the WHOLE video, parallel frame ranges; cached at <labcams>/null_potent/<label>/me_<cam>.npz; then the
imaging-clock max, the automatic threshold, and me_thresholds.png (histogram of log ME per camera, threshold line).
STAGE 2 (`analyze`): moving / stationary labels; subspaces fit on ALL labelled frames of the session (label-free,
so not circular for position decoding -- the paper fits on all trials); normVE; parallel-analysis dimensionality
(reported); two-stage-PCA control; then the project's position decoder (`locanmf_position_decoder._trial_features`
-- ENL = precue lick-free 2 s; post-cue = cue-aligned 2 s; engaged trials; block CV; logistic C 0.5; label-shuffle
null) on raw (z-scored) activity and on its reconstruction from each subspace (joint and two-stage).
Output: <labcams>/null_potent/null_potent_summary.csv (one row per session x window x arm) + per-session files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAMS = ("cam1", "cam2", "cam3", "cam4")


def _out(rv, label=None) -> Path:
    d = Path(rv.root("labcams")) / "null_potent"
    if label:
        d = d / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session(label):
    from wfield_local import config
    return next(s for s in config.load_sessions() if s["label"] == label)


def _animal_date(s):
    return s["label"].split("_")[0], Path(s["h5"]).parent.name


def _spout_still(animal, date, rv, ft, buffer_s=0.3) -> tuple[np.ndarray, str]:
    """``(mask, source)``: imaging frames in which the MOTORISED spout is NOT moving. Spout travel is apparatus
    motion, not mouse, so it is left out of the thresholds and labelled excluded, as is ``buffer_s`` after each move
    (the same GCaMP-decay buffer as after mouse movement: the moving spout can evoke activity). Priya, 2026-10-08:
    exclude only the spout-moving frames -- the logged ``dock_start -> dock`` and ``trial_start -> position``
    (`trial_windows.spout_motion_spans`, source 'log'). If the log's clock will not align, fall back to the
    coarser strobe .. trial end rule of the video regressors (source 'strobe..trial_end')."""
    from wfield_local import trial_windows as TW
    sp = TW.spout_motion_spans(animal, date, rv)
    if sp is not None:
        m = np.ones(len(ft), bool)
        for a0, a1 in zip(*sp):
            m &= ~((ft >= a0) & (ft < a1 + buffer_s))
        return m, "log"
    b = TW.trial_bounds(animal, date, rv)
    start = np.where(np.isfinite(b.strobe_s), b.strobe_s, b.cue_s - 0.5).astype(float)
    m = np.zeros(len(ft), bool)
    for a0, a1 in zip(start, b.stop_s.to_numpy(float)):
        if np.isfinite(a0) and np.isfinite(a1):
            m |= (ft >= a0) & (ft < a1)
    return m, "strobe..trial_end"


def stage_me(label, rv, workers=6, ds=4, win=3, log=print) -> dict:
    """Per-camera motion energy on the imaging clock (cached) + automatic thresholds + figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scripts.movement_encoding_session as MSE
    from wfield_local import config
    from wfield_local import motion_state as MST
    from wfield_local.movement_inputs import cam_frames_to_daq_s
    s = _session(label)
    animal, date = _animal_date(s)
    _, _, ft = MSE.imaging(label)
    out = _out(rv, label)
    me = {}
    for cam in CAMS:
        f = out / f"me_{cam}.npz"
        if f.exists():
            me[cam] = np.load(f)["me"]
            continue
        vids = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{cam}_*.avi"))
        tp = Path(rv.root("alignment_templates")) / cam / animal / f"{date}.npz"
        if not vids or not tp.exists():
            log(f"  {label} {cam}: no video / template -- skipped")
            continue
        tpl = dict(np.load(tp, allow_pickle=True))
        n = int(tpl["n_cam_frames"])
        v = MST.session_frame_scalar(vids[0], n, ds=ds, win=win, workers=workers)
        me[cam] = MST.to_imaging(v, cam_frames_to_daq_s(tpl, np.arange(len(v))), ft)
        np.savez_compressed(f, me=me[cam], frame_times_s=ft, ds=ds, win=win, video=vids[0].name)
        log(f"  {label} {cam}: motion energy done ({np.isfinite(me[cam]).mean():.1%} of imaging frames)")
    still, still_src = _spout_still(animal, date, rv, ft)
    me_in = {c: np.where(still, v, np.nan) for c, v in me.items()}
    over = ((config.defaults().get("motion_state", {}) or {}).get("thresholds", {}) or {}).get(label, {})
    thr = {c: float(over.get(c, MST.bimodal_threshold(v))) for c, v in me_in.items()}
    (out / "me_thresholds.json").write_text(json.dumps({"thresholds": thr, "override": over, "frames": f"spout still ({still_src})", "frac_frames_kept": float(still.mean())}, indent=1))
    fig, axs = plt.subplots(1, len(me), figsize=(4.2 * len(me), 3.2), squeeze=False)
    for ax, (c, v) in zip(axs[0], me_in.items()):
        lv = np.log10(v[np.isfinite(v) & (v > 0)])
        ax.hist(lv, bins=120, color="0.4")
        ax.axvline(np.log10(thr[c]), color="r")
        ax.set_title(f"{c}: threshold {thr[c]:.2f} ({(v[np.isfinite(v)] > thr[c]).mean():.0%} of spout-still frames moving)"
                     + (" [override]" if c in over else ""), fontsize=8)
        ax.set_xlabel("log10 motion energy (99th pct pixel)")
    fig.suptitle(f"{label}: per-frame motion energy and the moving threshold (valley between the modes); spout-moving frames excluded", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "me_thresholds.png", dpi=110)
    plt.close(fig)
    return {"me": me, "thr": thr, "ft": ft, "still": still, "still_src": still_src}


def stage_analyze(label, rv, n_perm=1000, n_rand=5, log=print) -> list[dict]:
    import glob

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import scripts.movement_encoding_session as MSE
    from wfield_local import config, dlc_frames, epochs
    from wfield_local import locanmf_position_decoder as LPD
    from wfield_local import motion_state as MST
    from wfield_local import nolick_analysis as na
    from wfield_local import null_potent as NP
    from wfield_local import trial_windows as TW
    from wfield_local.behavior_events import session_speed
    s = _session(label)
    animal, date = _animal_date(s)
    r1 = stage_me(label, rv, log=log)
    me, thr, ft = r1["me"], r1["thr"], r1["ft"]
    Y, reg, _ = MSE.imaging(label)
    b = TW.trial_bounds(animal, date, rv)
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    licks = dlc_frames.lick_onsets(animal, date, sid, rv)
    ts, sp = session_speed(Path(s["h5"]))
    run = np.interp(ft, ts, sp)
    rn = config.defaults()["segmentation"]["running"]
    lab = MST.label_frames(me, thr, ft, lick_s=licks, running=run, run_thresh=float(rn["thresh_speed_mm_s"]))
    lab[~r1["still"]] = -1                        # motorised spout travelling (+ buffer): apparatus motion
    # z-score on an inter-trial baseline (0.7 .. 0.2 s before each strobe)
    base = np.zeros(len(ft), bool)
    for st in b.strobe_s[np.isfinite(b.strobe_s)]:
        base |= (ft >= st - 0.7) & (ft < st - 0.2)
    mu, sd = Y[base].mean(0), Y[base].std(0)
    Z = (Y - mu) / np.where(sd > 0, sd, 1.0)
    dp = config.defaults()["decode"]
    names = {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
    reg = np.asarray(reg)
    # SUBSPACES ARE FIT WITHIN EACH AREA GROUP (Hasnain fit per region). Fitting on all components and then keeping
    # a group's columns of the reconstruction leaks other areas' activity into the group (2026-10-08: "MO" null
    # decoded above MO raw).
    groups = {"all": np.arange(Z.shape[1]),
              "SSp": np.flatnonzero([names.get(int(r), "").startswith("SSp") for r in reg]),
              "MO": np.flatnonzero([names.get(int(r), "").startswith(("MOp", "MOs")) for r in reg])}
    sub_out, rows = {}, []
    rng = np.random.default_rng(0)
    for gname, idx in groups.items():
        Zg = Z[:, idx]
        Xs, Xm = Zg[lab == 0], Zg[lab == 1]
        fit = NP.fit_subspaces(Xs, Xm)
        ts_ = NP.two_stage_pca(Xs, Zg[lab >= 0], k_null=min(5, len(idx) // 4), k_pot=min(5, len(idx) // 4))
        pa = NP.parallel_analysis(Zg[lab >= 0][:: max(1, int((lab >= 0).sum() // 20000))], n_shuffle=100)
        ve = fit["normVE"]
        dn, d2 = fit["Q_null"].shape[1], ts_["Q_null"].shape[1]
        log(f"  {label} {gname} (N {len(idx)}): stationary {np.mean(lab == 0):.0%}, moving {np.mean(lab == 1):.0%}, "
            f"excluded {np.mean(lab == -1):.0%}; d = {dn} each (parallel analysis: {pa}); normVE "
            f"null|stat {ve['null_stat']:.2f} null|mov {ve['null_mov']:.2f} pot|mov {ve['pot_mov']:.2f} "
            f"pot|stat {ve['pot_stat']:.2f}")
        sub_out.update({f"{gname}_idx": idx, f"{gname}_Q_null": fit["Q_null"], f"{gname}_Q_pot": fit["Q_pot"],
                        f"{gname}_Q_null_2s": ts_["Q_null"], f"{gname}_Q_pot_2s": ts_["Q_pot"]})
        c = Zg.mean(0)
        # Arm names avoid the bare word "null" (pandas reads it back as NaN). SIZE-MATCHED BASELINES: a d-dim
        # reconstruction keeps much of any linear code, so each subspace is read against random orthonormal
        # subspaces of the same size (``n_rand`` draws, fewer permutations) and against the REST (the dims
        # orthogonal to both).
        Qj = np.hstack([fit["Q_null"], fit["Q_pot"]])
        U, _, _ = np.linalg.svd(np.eye(len(idx)) - Qj @ Qj.T)
        signals = {"raw": Zg, "null_joint": NP.reconstruct(Zg, fit["Q_null"], c),
                   "potent_joint": NP.reconstruct(Zg, fit["Q_pot"], c),
                   "null_2stage": NP.reconstruct(Zg, ts_["Q_null"], c),
                   "potent_2stage": NP.reconstruct(Zg, ts_["Q_pot"], c)}
        if len(idx) > Qj.shape[1]:
            signals["rest_joint"] = NP.reconstruct(Zg, U[:, : len(idx) - Qj.shape[1]], c)
        for dd in sorted({dn, d2}):
            for k in range(n_rand):
                Qr, _ = np.linalg.qr(rng.standard_normal((len(idx), dd)))
                signals[f"random{dd}_{k}"] = NP.reconstruct(Zg, Qr, c)
        for win, align, post in [("ENL", "precue", dp["precue_post_s"]), ("postcue", "cue", dp["cue_post_s"])]:
            args = argparse.Namespace(align=align, baseline="none", fs=31.23, pre_s=1.0, post_s=post,
                                      max_rt=dp["max_rt_s"], source="locanmf", bins=None, cv="block")
            for nm, sig in signals.items():
                X, y, g, *_r = LPD._trial_features(s, args, signal=sig.T, feat_region=reg[idx])
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
                pred = cross_val_predict(clf, X, y, cv=GroupKFold(min(5, np.unique(g).size)), groups=g)
                nu = na.permutation_null(y, pred, n_perm=n_perm if not nm.startswith("random") else 100,
                                         labels=LPD.DISPLAY_ORDER)
                rows.append({"label": label, "animal": animal, "epoch": epochs.epoch_of(label),
                             "day": epochs.days_since_stroke(label), "window": win, "arm": nm, "areas": gname,
                             "n_comp": len(idx), "n_trials": len(y), "acc": float((pred == y).mean()),
                             "null_mean": nu["raw_null_mean"], "p": nu["raw_p"], "d": dn, "d_2stage": d2,
                             "parallel_dims": pa, "spout_mask": r1["still_src"],
                             "frac_stationary": float(np.mean(lab == 0)), "frac_moving": float(np.mean(lab == 1)),
                             **{f"normVE_{k}": v for k, v in ve.items()}})
            log(f"  {label} {gname} {win}: " + "  ".join(
                f"{r['arm']} {r['acc']:.2f}" for r in rows if r["window"] == win and r["areas"] == gname
                and not r["arm"].startswith("random")))
    np.savez_compressed(_out(rv, label) / "subspaces.npz", labels=lab, mu=mu, sd=sd, **sub_out)
    return rows


def main(argv=None) -> int:
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["me", "analyze", "all"])
    ap.add_argument("labels", nargs="*", help="session labels (default: every regime-B session with an epoch)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args(argv)
    rv = PathResolver()
    labels = a.labels or [s["label"] for s in config.load_sessions()
                          if s["regime"] == "B" and epochs.epoch_of(s["label"]) is not None]
    summ = _out(rv) / "null_potent_summary.csv"
    done = set(pd.read_csv(summ).label) if summ.exists() and a.stage != "me" else set()
    for label in labels:
        try:
            if a.stage == "me":
                stage_me(label, rv, workers=a.workers)
                continue
            if label in done:
                continue
            rows = stage_analyze(label, rv, n_perm=a.n_perm)
            pd.DataFrame(rows).to_csv(summ, mode="a", header=not summ.exists(), index=False)
        except Exception as e:                                           # noqa: BLE001
            print(f"!! {label}: {type(e).__name__}: {e}", flush=True)
    print("->", summ if a.stage != "me" else _out(rv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
