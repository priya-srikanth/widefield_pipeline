"""PHASE A VALIDATION GATE of a four-camera 3-D re-run against the adopted two-camera (cam1 + cam4) 3-D, PRE-STROKE
sessions only (VLS-paper refresh, `vls_stroke_paper/PROMPT_c2_4cam_3d_refresh.md` Phase A step 3; 2026-11-21).

    python -m scripts.pose3d_phaseA_gate [--tag _3net_20261121] [--jobs 6] [--only coverage agreement decode noise]

Two-camera = ``pose3d.pose3d_dir(source="2cam")`` (session_poses/pose3d); four-camera = ``pose3d_dir(source="4cam")``
with the run tag (``WIDEFIELD_POSE3D_TAG``, here --tag). Pre-stroke = ``epochs.epoch_of(label) == "pre"`` on the sessions
both runs have; "August baseline" = not ``config.early_pre`` (the June sessions + PS92 0805 are the early pre block).
Every output goes to ``<four-camera folder>/qc_phaseA/`` (refuses to overwrite an existing file unless --redo); nothing
is written into either run's tables.

GATES (definitions are the repo's; nothing new is measured):
 1. COVERAGE per session x run, from the per-frame tables: share of frames with a 3-D tongue / jaw point (all frames, and
    the response window cue .. trial stop), the share from >= 2 views vs a one-view fill, per-view use (2cam: src 1 / 2 /
    3; 4cam: ``{part}_rep_<cam>`` present), the views histogram, and the median reprojection error per view on
    multi-view points; plus on the COMMON frames (2cam both-camera point AND 4cam >= 2 views using cam1 + cam4) the cam1 /
    cam4 reprojection of both runs and the 3-D distance between the two tongue points.
 2. PER-LICK AGREEMENT, two lick identities:
    a. the runs' own 3-D licks (`pose3d.lick_peaks` tables ``<s>_licks.csv``; the four-camera run with the camera split +
       40 ms floor) matched one-to-one within a trial by extension-peak time, |dt| <= 30 ms (greedy, nearest first);
    b. the cam4 v36 lick windows (``licks.csv``) measured on each run's frames with `pose3d_kinematics.session_licks`
       (identical licks by construction; the kinematics definition the manuscript tables use).
    Metrics at the 3-D extension peak (farthest frame from the mouth zero): lateral displacement ``ml`` / ``side_ml`` (mm,
    + = mouse right), extension-peak MIDLINE angle asin(ml / reach) (deg; `pose3d_kinematics` ``midline``), ``reach_mm``.
    Per animal: Pearson r, mean offset (4cam - 2cam), median |difference|.
 3. SIX-WAY SPOUT DECODING from the lick endpoint: `pose3d_vs_cam4.decode` (multinomial logistic regression on standardized
    side_ml / out_ap / down_dv, leave-one-SESSION-out over the animal's curated August pre sessions, balanced accuracy,
    chance 1/6) on the 2b licks of camera-working-period trials (`trial_classes.gate_kin_licks`), identically on both runs;
    also on the production two-camera ``kinematics/licks.csv.gz`` (the 0.67-0.86 reference) and on the runs' own 3-D
    licks (2a tables: ml / ap / dv, working-period trials).
 4. SUBMOVEMENT NOISE CHECK: `within_lick_correction` per-lick extraction (cam4 v36 licks_all windows on the selected 3-D
    frames) for the curated August pre sessions, written to qc_phaseA/within_lick_correction/licks (NOT the module's
    routed folder, so a later Phase B run with other lick windows cannot silently reuse these pickles), its
    pre-registered threshold calibration (`calibrate`), `prepare` + `add_csm` + `sessions_table`. The two-camera side is
    the cached production extraction (imaging_encoding/within_lick_correction, read only). Pass: real CSM share > noise
    surrogate (csm_false) in every animal; far lateral (far_ipsi + far_contra) > close lateral (close_ipsi + close_contra).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

V36 = "lp_r4_heatmap_20261013__clean_v36"
AN = ["PS92", "PS93", "PS94", "PS95"]
RUNS = ("2cam", "4cam")
CAMS = {"2cam": ("cam1", "cam4"), "4cam": ("cam1", "cam4", "cam2", "cam3")}
PARTS = ("tongue", "jaw")
MATCH_MS = 30.0
POS6 = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")


def _env(tag: str) -> None:
    from wfield_local import pose3d
    os.environ[pose3d.TAG_ENV] = tag
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def run_dirs(tag: str) -> dict:
    from wfield_local import pose3d
    _env(tag)
    return {"2cam": pose3d.pose3d_dir(source="2cam"), "4cam": pose3d.pose3d_dir(source="4cam")}


def pre_sessions(dirs: dict) -> pd.DataFrame:
    """Pre-stroke sessions present in both runs: sid, animal, date, label, august (not early_pre), curated."""
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs, pose3d
    bad = {r: pose3d.failed_sessions(d) for r, d in dirs.items()}
    have = [{f.name[:13] for f in d.glob("PS*_frames.csv.gz")} for d in dirs.values()]
    cur = ak.curated_ids()
    rows = []
    for s in sorted(set.intersection(*have)):
        lab = f"{s[:4]}_{s[-4:]}"
        if s[:4] in AN and epochs.epoch_of(lab) == "pre":
            rows.append({"sid": s, "animal": s[:4], "date": s[5:], "label": lab, "august": not config.early_pre(s),
                         "curated": s in cur, **{f"failed_{r}": bad[r].get(s, "") for r in bad}})
    return pd.DataFrame(rows)


# ================================================================================================ per-session worker
def _cols(run: str) -> list[str]:
    c = ["trial_id", "position", "t_ms", "stop_ms"]
    for p in PARTS:
        c += [f"{p}_ml", f"{p}_ap", f"{p}_dv", f"{p}_src"]
        c += [f"{p}_rep_{k}" for k in CAMS["4cam"]] + [f"{p}_views"] if run == "4cam" else [f"{p}_rep1", f"{p}_rep4"]
    return c


def _used(S: pd.DataFrame, run: str, part: str) -> dict:
    """Per view: frames where that camera contributed to the part's 3-D point."""
    if run == "2cam":
        src = S[f"{part}_src"].to_numpy()
        return {"cam1": np.isin(src, (1, 2)), "cam4": np.isin(src, (1, 3))}
    return {k: S[f"{part}_rep_{k}"].notna().to_numpy() for k in CAMS["4cam"]}


def _rep(S: pd.DataFrame, run: str, part: str, cam: str) -> np.ndarray:
    if run == "2cam":
        return S[f"{part}_rep{cam[-1]}"].to_numpy(float)
    return S[f"{part}_rep_{cam}"].to_numpy(float)


def coverage(S: pd.DataFrame, run: str) -> dict:
    t, stop = S.t_ms.to_numpy(float), S.stop_ms.to_numpy(float)
    resp = (t >= 0) & (t <= stop)
    o = {"n_frames": len(S), "n_resp": int(resp.sum())}
    for p in PARTS:
        X = S[[f"{p}_ml", f"{p}_ap", f"{p}_dv"]].to_numpy(float)
        have = np.isfinite(X).all(1)
        src = S[f"{p}_src"].to_numpy()
        multi = have & (src == 1)
        o[f"{p}_frac"] = float(have.mean())
        o[f"{p}_frac_resp"] = float(have[resp].mean()) if resp.any() else np.nan
        o[f"{p}_multi_frac"] = float(multi.mean())
        o[f"{p}_multi_frac_resp"] = float(multi[resp].mean()) if resp.any() else np.nan
        o[f"{p}_fill_frac"] = float((have & (src > 1)).mean())
        used = _used(S, run, p)
        for cam, u in used.items():
            o[f"{p}_use_{cam}"] = float((u & have).mean())
            r = _rep(S, run, p, cam)[multi & u]
            r = r[np.isfinite(r)]
            o[f"{p}_rep_{cam}_med"] = float(np.median(r)) if len(r) else np.nan
            o[f"{p}_rep_{cam}_p90"] = float(np.percentile(r, 90)) if len(r) else np.nan
        if run == "4cam":
            v = S[f"{p}_views"].to_numpy()
            for k in (2, 3, 4):
                o[f"{p}_views{k}_frac"] = float((v == k).mean())
    return o


def common_frames(S2: pd.DataFrame, S4: pd.DataFrame) -> dict:
    """cam1 / cam4 reprojection of both runs and the 2cam-4cam tongue / jaw distance on frames both triangulate
    from >= 2 views with cam1 AND cam4 among them (like-for-like reprojection)."""
    M = S2.merge(S4, on=["trial_id", "t_ms"], suffixes=("_2", "_4"))
    o = {"n_joined": len(M)}
    for p in PARTS:
        k = ((M[f"{p}_src_2"] == 1) & (M[f"{p}_src_4"] == 1) & M[f"{p}_rep_cam1"].notna()
             & M[f"{p}_rep_cam4"].notna()).to_numpy()
        o[f"{p}_common_n"] = int(k.sum())
        for cam in ("cam1", "cam4"):
            o[f"{p}_common_rep_{cam}_2cam"] = float(np.nanmedian(M[f"{p}_rep{cam[-1]}"].to_numpy(float)[k])) if k.any() else np.nan
            o[f"{p}_common_rep_{cam}_4cam"] = float(np.nanmedian(M[f"{p}_rep_{cam}"].to_numpy(float)[k])) if k.any() else np.nan
        A = M[[f"{p}_ml_2", f"{p}_ap_2", f"{p}_dv_2"]].to_numpy(float)[k]
        B = M[[f"{p}_ml_4", f"{p}_ap_4", f"{p}_dv_4"]].to_numpy(float)[k]
        d = np.linalg.norm(A - B, axis=1)
        o[f"{p}_common_dist_med_mm"] = float(np.median(d)) if len(d) else np.nan
        o[f"{p}_common_dist_p90_mm"] = float(np.percentile(d, 90)) if len(d) else np.nan
        # tongue frames only one run has (any source)
        h2 = np.isfinite(M[[f"{p}_ml_2"]].to_numpy(float)).ravel()
        h4 = np.isfinite(M[[f"{p}_ml_4"]].to_numpy(float)).ravel()
        o[f"{p}_only2cam_n"], o[f"{p}_only4cam_n"], o[f"{p}_both_n"] = int((h2 & ~h4).sum()), int((h4 & ~h2).sum()), int((h2 & h4).sum())
    return o


def match_peaks(L2: pd.DataFrame, L4: pd.DataFrame, tol: float = MATCH_MS) -> pd.DataFrame:
    """One-to-one matches of two 3-D lick tables within each trial by extension-peak time (nearest first, |dt| <= tol)."""
    out = []
    g4 = {k: v for k, v in L4.groupby("trial_id")}
    for tr, a in L2.groupby("trial_id"):
        b = g4.get(tr)
        if b is None:
            continue
        ta, tb = a.t_peak_ms.to_numpy(float), b.t_peak_ms.to_numpy(float)
        D = np.abs(ta[:, None] - tb[None, :])
        pairs = np.argwhere(D <= tol)
        pairs = pairs[np.argsort(D[pairs[:, 0], pairs[:, 1]], kind="stable")]
        ua, ub = set(), set()
        for i, j in pairs:
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            out.append((a.index[i], b.index[j]))
    if not out:
        return pd.DataFrame()
    ia, ib = zip(*out)
    A = L2.loc[list(ia)].reset_index(drop=True).add_suffix("_2cam")
    B = L4.loc[list(ib)].reset_index(drop=True).add_suffix("_4cam")
    return pd.concat([A, B], axis=1)


def _midline(ml, reach):
    return np.degrees(np.arcsin(np.clip(np.asarray(ml, float) / np.asarray(reach, float), -1, 1)))


def session_worker(job) -> dict:
    """One pre-stroke session: coverage of both runs, common-frame stats, 3-D lick matching (2a) and the cam4-window
    kinematics on both runs (2b). Returns tables; the parent writes them."""
    sid, tag = job
    _env(tag)
    import scripts.pose3d_kinematics as PK
    from scripts.apparatus_mouth import _rig
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    dirs = run_dirs(tag)
    an, date = sid.split("_")
    S = {}
    for run in RUNS:
        X = pd.read_csv(dirs[run] / f"{sid}_frames.csv.gz", usecols=_cols(run))
        S[run] = X.drop_duplicates(["trial_id", "t_ms"]).sort_values(["trial_id", "t_ms"]).reset_index(drop=True)
    cov = [{"sid": sid, "run": run, **coverage(S[run], run)} for run in RUNS]
    com = {"sid": sid, **common_frames(S["2cam"], S["4cam"])}
    # 2a: the runs' own 3-D lick tables
    L = {run: pd.read_csv(dirs[run] / f"{sid}_licks.csv") for run in RUNS}
    for run in RUNS:
        L[run]["midline"] = _midline(L[run].ml, L[run].reach_mm)
    M = match_peaks(L["2cam"], L["4cam"])
    if len(M):
        M.insert(0, "sid", sid)
    lk = {"sid": sid, "licks_2cam": len(L["2cam"]), "licks_4cam": len(L["4cam"]), "matched": len(M)}
    # 2b: identical licks (cam4 v36 windows) measured on each run's frames (pose3d_kinematics definitions)
    lf = Path(rv.root("orofacial")) / V36 / "sessions" / an / date / "licks.csv"
    K = {}
    for run in RUNS if lf.exists() else ():                     # no v36 licks: skipped, as pose3d_kinematics does
        C = pd.read_csv(lf)
        geo = _rig(rv, an, date)["geometry"]
        k = PK.session_licks(S[run], C, geo, np.zeros(3))
        k["animal"], k["date"], k["session"], k["epoch"], k["run"] = an, date, sid, "pre", run
        K[run] = k
    own = pd.concat([L[r].assign(run=r, sid=sid, session=sid, animal=an, date=date) for r in RUNS], ignore_index=True)
    kin = pd.concat(K.values(), ignore_index=True) if K else pd.DataFrame()
    lk["v36_licks"] = lf.exists()
    return {"cov": cov, "com": com, "match": M, "lk": lk, "kin": kin, "own": own}


# ================================================================================================ summaries
def _agree(M: pd.DataFrame, metrics: dict, by: str = "animal") -> pd.DataFrame:
    rows = []
    groups = [("all", M)] + list(M.groupby(by, sort=True))
    for g, h in groups:
        for name, (c2, c4) in metrics.items():
            a, b = h[c2].to_numpy(float), h[c4].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            rows.append({by: g, "metric": name, "n": int(ok.sum()),
                         "r": float(np.corrcoef(a, b)[0, 1]) if ok.sum() > 2 else np.nan,
                         "offset_4m2": float(np.mean(b - a)) if ok.any() else np.nan,
                         "med_abs_diff": float(np.median(np.abs(b - a))) if ok.any() else np.nan,
                         "sd_2cam": float(np.std(a)) if ok.any() else np.nan, "sd_4cam": float(np.std(b)) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def _write(df_or_text, path: Path, redo: bool) -> None:
    if path.exists() and not redo:
        raise SystemExit(f"{path} exists (refusing to overwrite; --redo to replace)")
    if isinstance(df_or_text, pd.DataFrame):
        df_or_text.to_csv(path, index=False, float_format="%.5g")
    else:
        path.write_text(df_or_text)


def gates_1_2(P: pd.DataFrame, tag: str, jobs: int, out: Path, redo: bool) -> dict:
    from wfield_local import analysis_kit as ak
    res, fail = ak.fan_sessions([(s, tag) for s in P.sid], session_worker, jobs=jobs, key=lambda j: j[0])
    if fail:
        raise SystemExit(f"failed: {[(j[0], str(e)[:200]) for j, e in fail]}")
    vals = [v for _k, v in res]
    meta = P.set_index("sid")[["animal", "date", "august", "curated"]]
    COV = pd.DataFrame([c for v in vals for c in v["cov"]]).join(meta, on="sid")
    COM = pd.DataFrame([v["com"] for v in vals]).join(meta, on="sid")
    LK = pd.DataFrame([v["lk"] for v in vals]).join(meta, on="sid")
    LK["match_rate_of_2cam"], LK["match_rate_of_4cam"] = LK.matched / LK.licks_2cam, LK.matched / LK.licks_4cam
    M = pd.concat([v["match"] for v in vals if len(v["match"])], ignore_index=True).join(meta, on="sid")
    KIN = pd.concat([v["kin"] for v in vals], ignore_index=True)
    OWN = pd.concat([v["own"] for v in vals], ignore_index=True)
    # coverage, wide per session (2cam vs 4cam side by side)
    W = COV.pivot_table(index=["sid", "animal", "date", "august", "curated"], columns="run", aggfunc="first")
    W.columns = [f"{m}__{r}" for m, r in W.columns]
    W = W.dropna(axis=1, how="all").reset_index().merge(COM.drop(columns=["animal", "date", "august", "curated"]), on="sid")
    side = pd.concat([pd.read_csv(f) for f in run_dirs(tag)["4cam"].glob("summary_production*.csv")])
    W = W.merge(side[["session", "side_dropped"]].rename(columns={"session": "sid"}), on="sid", how="left")
    for p in PARTS:
        W[f"{p}_frac_ratio"] = W[f"{p}_frac__4cam"] / W[f"{p}_frac__2cam"]
        W[f"{p}_4cam_ge_2cam"] = W[f"{p}_frac__4cam"] >= W[f"{p}_frac__2cam"]
    _write(W, out / "gate1_coverage_sessions.csv", redo)
    keys = (["tongue_frac", "tongue_frac_resp", "tongue_multi_frac", "jaw_frac", "jaw_multi_frac"]
            + [f"{p}_rep_{c}_med" for p in PARTS for c in ("cam1", "cam4")])
    rows = []
    for (an, aug), h in W.groupby(["animal", "august"], sort=True):
        r = {"animal": an, "block": "August" if aug else "June/early", "n_sessions": len(h),
             "tongue_4cam_ge_2cam": int(h.tongue_4cam_ge_2cam.sum()), "jaw_4cam_ge_2cam": int(h.jaw_4cam_ge_2cam.sum())}
        for k in keys:
            for run in RUNS:
                c = f"{k}__{run}"
                if c in h:
                    r[f"{k}_{run}_med"] = float(h[c].median())
        for c in ("cam2", "cam3"):
            for p in PARTS:
                r[f"{p}_rep_{c}_med_4cam_med"] = float(h[f"{p}_rep_{c}_med__4cam"].median())
        for p in PARTS:
            for c in ("cam1", "cam4"):
                for run in RUNS:
                    r[f"{p}_common_rep_{c}_{run}_med"] = float(h[f"{p}_common_rep_{c}_{run}"].median())
            r[f"{p}_common_dist_med_mm"] = float(h[f"{p}_common_dist_med_mm"].median())
        rows.append(r)
    _write(pd.DataFrame(rows), out / "gate1_coverage_per_animal.csv", redo)
    # gate 2
    _write(LK, out / "gate2_lick_match_sessions.csv", redo)
    M.to_csv(out / "gate2a_matched_licks.csv.gz", index=False, float_format="%.5g")
    met_a = {"lateral ml (mm)": ("ml_2cam", "ml_4cam"), "midline angle (deg)": ("midline_2cam", "midline_4cam"),
             "reach (mm)": ("reach_mm_2cam", "reach_mm_4cam")}
    A = _agree(M, met_a)
    A.insert(0, "licks", "own 3-D licks, peaks within 30 ms")
    K2, K4 = (KIN[KIN.run == r].drop(columns="run") for r in RUNS)
    KM = K2.merge(K4, on=["session", "trial_id", "lick_idx"], suffixes=("_2cam", "_4cam"))
    KM["animal"] = KM.animal_2cam
    met_b = {"lateral side_ml (mm)": ("side_ml_2cam", "side_ml_4cam"), "midline angle (deg)": ("midline_2cam", "midline_4cam"),
             "reach (mm)": ("reach_mm_2cam", "reach_mm_4cam"), "frontal angle (deg)": ("frontal_2cam", "frontal_4cam"),
             "t peak 3-D (ms)": ("t_peak_3d_2cam", "t_peak_3d_4cam")}
    B = _agree(KM, met_b)
    B.insert(0, "licks", "cam4 v36 lick windows (identical licks)")
    _write(pd.concat([A, B], ignore_index=True), out / "gate2_agreement_per_animal.csv", redo)
    kn = KIN.groupby(["run", "animal"]).size().unstack(0)
    kn["both"] = KM.groupby("animal").size()
    _write(kn.reset_index(), out / "gate2b_kin_counts.csv", redo)
    KIN.to_csv(out / "gate2b_kin_licks.csv.gz", index=False, float_format="%.5g")
    OWN.to_csv(out / "gate3_own_licks.csv.gz", index=False, float_format="%.5g")
    return {"W": W, "LK": LK, "A": A, "B": B, "KIN": KIN, "OWN": OWN}


def gate_3(P: pd.DataFrame, KIN: pd.DataFrame, OWN: pd.DataFrame, dirs: dict, out: Path, redo: bool) -> pd.DataFrame:
    import scripts.pose3d_vs_cam4 as PV
    from wfield_local import trial_classes as tc
    aug = set(P[P.august & P.curated].sid)
    rows = []

    def run_set(L, name, feats):
        L = tc.gate_kin_licks(L[L.session.isin(aug)])
        for an, h in L.groupby("animal", sort=True):
            r = {"animal": an, "set": name, "n_sessions": h.session.nunique(), "n_licks": len(h),
                 "6 positions": PV.decode(h, feats, POS6)}
            for pn, pr in PV.PAIRS.items():
                r[pn] = PV.decode(h, feats, pr)
            rows.append(r)

    for run in RUNS:
        run_set(KIN[KIN.run == run], f"{run}: cam4 v36 lick windows", PV.FEATS["3-D"])
    prod = pd.read_csv(dirs["2cam"] / "kinematics" / "licks.csv.gz", dtype={"date": str})
    run_set(prod, "2cam production kinematics/licks.csv.gz (reference)", PV.FEATS["3-D"])
    for run in RUNS:
        run_set(OWN[OWN.run == run], f"{run}: own 3-D licks (lick_peaks ml/ap/dv)", ["ml", "ap", "dv"])
    D = pd.DataFrame(rows)
    _write(D, out / "gate3_decoding.csv", redo)
    return D


# ------------------------------------------------------------------------------------------------ gate 4
def wlc_worker(job) -> dict:
    """`within_lick_correction.session_licks` for one session on ``source``'s 3-D frames, pickles into ``licks_dir``."""
    sid, source, tag, licks_dir = job
    from wfield_local import pose3d
    os.environ[pose3d.SOURCE_ENV] = source
    _env(tag)
    sys.path.insert(0, str(REPO / "scripts"))
    import within_lick_correction as W
    W.LICKS = Path(licks_dir)
    assert pose3d.pose3d_dir().name.startswith("pose3d_4cam") == (source == "4cam")
    return W.session_licks(sid)


def gate_4(P: pd.DataFrame, tag: str, jobs: int, out: Path, redo: bool) -> dict:
    from wfield_local import analysis_kit as ak
    sys.path.insert(0, str(REPO / "scripts"))
    import within_lick_correction as W
    aug = sorted(P[P.august & P.curated].sid)
    ld = out / "within_lick_correction" / "licks"
    ld.mkdir(parents=True, exist_ok=True)
    res, fail = ak.fan_sessions([(s, "4cam", tag, str(ld)) for s in aug], wlc_worker, jobs=jobs, key=lambda j: j[0])
    if fail:
        raise SystemExit(f"within-lick extraction failed: {[(j[0], str(e)[:200]) for j, e in fail]}")
    from wfield_local import run_mode
    os.environ.pop("WIDEFIELD_POSE3D_SOURCE", None)                 # the two-camera (adopted) tree, read only
    two = run_mode.encoding_out_root(V36) / "within_lick_correction"
    assert two.name == "within_lick_correction" and two.parent.name == "imaging_encoding", two
    cal2 = json.loads((two / "csm_threshold.json").read_text())
    tabs, cals, LL = {}, {"2cam": cal2}, {}
    for run, d in (("2cam", two / "licks"), ("4cam", ld)):
        W.LICKS = d
        LL[run] = W.load_licks(aug)
    cals["4cam"] = W.calibrate(LL["4cam"])
    cals["2cam_recalibrated_here"] = W.calibrate(LL["2cam"])     # same rule on this session set (check vs the cached)
    _write(json.dumps(cals, indent=1), out / "within_lick_correction" / "csm_threshold_2cam_4cam.json", redo)
    for run in RUNS:
        L = W.prepare(LL[run])
        for thr_from in RUNS:
            th, thm = cals[thr_from]["theta_sp"], cals[thr_from]["theta_ml"]
            if th is None:
                continue
            Lc = W.add_csm(L.copy(), th, thm if thm is not None else th)
            S = W.sessions_table(Lc)
            S["run"], S["theta_from"], S["theta_sp"] = run, thr_from, th
            tabs[(run, thr_from)] = S
    S = pd.concat(tabs.values(), ignore_index=True)
    S = S[(S.order == "any") & (S.outcome == "any")]
    _write(S, out / "gate4_csm_sessions.csv", redo)
    c = S[S.context == "cued"]
    pa = c.groupby(["run", "theta_from", "animal", "group"])[["csm_share", "csm_false", "csm_excess", "full_cov",
                                                               "n_csm"]].mean().reset_index()
    _write(pa, out / "gate4_csm_per_animal.csv", redo)
    # far lateral vs close lateral, per animal (mean of the two spouts' session-mean shares)
    rows = []
    for (run, tf, an), h in c.groupby(["run", "theta_from", "animal"], sort=True):
        m = h.groupby("group").csm_share.mean()
        f = h.groupby("group").csm_false.mean()
        rows.append({"run": run, "theta_from": tf, "animal": an,
                     "csm_all": m.get("all"), "false_all": f.get("all"),
                     "far_lateral": np.nanmean([m.get("far_ipsi"), m.get("far_contra")]),
                     "close_lateral": np.nanmean([m.get("close_ipsi"), m.get("close_contra")]),
                     "far_mid": m.get("far_mid"), "close_mid": m.get("close_mid")})
    FN = pd.DataFrame(rows)
    FN["real_gt_noise"] = FN.csm_all > FN.false_all
    FN["far_gt_close"] = FN.far_lateral > FN.close_lateral
    _write(FN, out / "gate4_far_vs_close.csv", redo)
    return {"S": S, "FN": FN, "cal": cals}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="_3net_20261121", help="four-camera run tag (WIDEFIELD_POSE3D_TAG)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=["coverage", "agreement", "decode", "noise"],
                    help="coverage / agreement / decode share one per-session pass; noise = gate 4")
    ap.add_argument("--redo", action="store_true", help="replace existing gate tables")
    ap.add_argument("--sessions", nargs="*", default=None, help="restrict to these PSxx_YYYYMMDD (smoke test)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default <four-camera folder>/qc_phaseA)")
    a = ap.parse_args(argv)
    dirs = run_dirs(a.tag)
    out = a.out or dirs["4cam"] / "qc_phaseA"
    out.mkdir(exist_ok=True, parents=True)
    P = pre_sessions(dirs)
    if a.sessions:
        P = P[P.sid.isin(a.sessions)].reset_index(drop=True)
    _write(P, out / "sessions_pre.csv", a.redo)
    print(f"{len(P)} pre-stroke sessions ({int((P.august & P.curated).sum())} curated August); 2cam {dirs['2cam']}; "
          f"4cam {dirs['4cam']}; -> {out}", flush=True)
    pd.set_option("display.width", 250)
    if {"coverage", "agreement", "decode"} & set(a.only):
        R = gates_1_2(P, a.tag, a.jobs, out, a.redo)
        print(R["A"].round(3).to_string(index=False), "\n", R["B"].round(3).to_string(index=False), flush=True)
        if "decode" in a.only:
            print(gate_3(P, R["KIN"], R["OWN"], dirs, out, a.redo).round(3).to_string(index=False), flush=True)
    if "noise" in a.only:
        G = gate_4(P, a.tag, a.jobs, out, a.redo)
        print(G["FN"].round(3).to_string(index=False), flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
