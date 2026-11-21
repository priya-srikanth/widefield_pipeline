"""CORRECTIVE-SUBMOVEMENT (CSM) DIAGNOSTIC, two-camera v36 3-D vs the four-camera 3-D (Phase A gate 4 follow-up,
2026-11-21). READ ONLY on every existing output; writes only to ``<4cam>/qc_phaseA/csm_examples/``.

Question: the four-camera track finds more CSMs at CLOSE spouts (close_ipsi 0.08 -> 0.20, close_mid 0.06 -> 0.18) and
fewer at FAR spouts. Real or a tracking artifact?
  H1 artifact: the set of views behind each four-camera frame changes frame to frame -> step jumps -> speed peaks (the
     iid-noise surrogate of `within_lick_correction` cannot see a step).
  H2 coverage: close licks were short / broken in two-camera; four-camera completes them (and fills far-spout gaps).
  H3 threshold: the lower calibrated threshold (40 vs 50 mm/s) alone raises the share.

Method (nothing new is defined; the CSM measure is `within_lick_correction`'s): for each curated August pre session and
each cam4-v36 ``licks_all`` lick in BOTH runs' gate-4 pickles (same ``lick_uid`` = identical lick window), the run's
track is rebuilt with `within_lick_correction._track` (carrying the frame index through, so each tracked frame keeps its
per-frame VIEW SET), speed = |SG(5,2) velocity|, extension peak k = farthest frame from the mouth zero, speed-peak
prominences before / after k exactly as `_peaks` (checked against the pickled ``pk_sp``). A CSM peak = a qualifying peak
(prominence >= theta) that is NOT the largest-prominence peak of its phase (so n CSM peaks = `csm_count`).
View set per frame: four-camera = the cameras whose ``tongue_rep_<cam>`` is finite (>= 2 views) or the one camera of a
one-view fill (``tongue_views`` -1, ``tongue_src`` 2 / 3 / 6 / 7); two-camera = ``tongue_src`` (1 both, 2 cam1, 3 cam4);
a linearly filled one-frame gap (src 9) is its own code in both. "Near a change": a view-set change on one of the four
frame boundaries within 1.5 frames of the peak frame (i.e. the peak frame or a neighbor sits on the change); the
baseline is the same quantity over ALL tracked frames of the same licks.

    python -m scripts.pose3d_csm_examples extract [--jobs 6]   per-lick / per-peak tables (csm_examples/tables)
    python -m scripts.pose3d_csm_examples summary              numbers per spout (csm_examples/summary_*.csv)
    python -m scripts.pose3d_csm_examples traces               example speed / lateral traces (PNG)
    python -m scripts.pose3d_csm_examples frames               video frames around clear close-spout gains (PNG)
    python -m scripts.pose3d_csm_examples refuse [--jobs 5]    production fusion re-run in memory + every camera subset
                                                               re-triangulated (csm_examples/fused; ~10 min / session)
    python -m scripts.pose3d_csm_examples fixes                jump sizes, fixes, switch-aware noise control, smoother
                                                               validation (`pose3d_csm_fixes`; csm_examples/fixes)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

TAG = "_3net_20261121"
SIDE_BUNDLE = "lp_side_r2b_prod_20261118"
V36 = "lp_r4_heatmap_20261013__clean_v36"
THS = (40.0, 50.0)
DT = 0.004
FOCUS = ("close_ipsi", "close_mid", "far_ipsi", "far_contra")
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
CAMBIT = {"cam1": 1, "cam4": 2, "cam2": 4, "cam3": 8}
LONE = {2: "cam1", 3: "cam4", 6: "cam2", 7: "cam3"}
INTERP_CODE = 999
SEED = 0


def _env() -> None:
    from wfield_local import pose3d
    os.environ[pose3d.TAG_ENV] = TAG
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def dirs() -> dict:
    from wfield_local import pose3d, run_mode
    _env()
    d4 = pose3d.pose3d_dir(source="4cam")
    os.environ.pop(pose3d.SOURCE_ENV, None)
    two = run_mode.encoding_out_root(V36) / "within_lick_correction"
    assert two.name == "within_lick_correction" and two.parent.name == "imaging_encoding", two
    qa = d4 / "qc_phaseA"
    return {"2cam_frames": pose3d.pose3d_dir(source="2cam"), "4cam_frames": d4,
            "2cam_licks": two / "licks", "4cam_licks": qa / "within_lick_correction" / "licks",
            "qa": qa, "out": qa / "csm_examples"}


def sessions(D: dict) -> list[str]:
    import json
    cal = json.loads((D["qa"] / "within_lick_correction" / "csm_threshold_2cam_4cam.json").read_text())
    return sorted(cal["4cam"]["sessions"] if "sessions" in cal["4cam"] else [])


# ------------------------------------------------------------------------------------------------ per-lick machinery
def view_codes(S: pd.DataFrame, run: str) -> np.ndarray:
    src = S.tongue_src.to_numpy(int)
    if run == "2cam":
        return src.copy()
    code = np.zeros(len(S), int)
    for c, b in CAMBIT.items():
        code |= np.where(np.isfinite(S[f"tongue_rep_{c}"].to_numpy(float)), b, 0)
    lone = S.tongue_views.to_numpy(int) == -1
    code[lone] = 100 + src[lone]                     # one-view fill: its own code per camera
    return code


def load_frames(D: dict, run: str, sid: str) -> pd.DataFrame:
    cols = ["trial_id", "position", "t_ms", "tongue_src", "tongue_ml", "tongue_ap", "tongue_dv"]
    if run == "4cam":
        cols += ["tongue_views"] + [f"tongue_rep_{c}" for c in CAMBIT]
    cols += ["src1", "src4"]
    S = pd.read_csv(D[f"{run}_frames"] / f"{sid}_frames.csv.gz", usecols=cols)
    S = S.drop_duplicates(["trial_id", "t_ms"]).sort_values(["trial_id", "t_ms"]).reset_index(drop=True)
    S["vcode"] = view_codes(S, run)
    return S


def peaks_idx(x: np.ndarray, k: int):
    """`within_lick_correction._peaks` with the frame index kept: [(frame, prominence, phase)]; phase 0 before k, 1 after."""
    from scipy.signal import find_peaks, peak_prominences
    xp = np.r_[0.0, x, 0.0]
    pk, _ = find_peaks(xp)
    if not len(pk):
        return []
    pr = peak_prominences(xp, pk)[0]
    out = []
    for i, p in zip(pk - 1, pr):
        if i == k:
            continue
        out.append((int(i), float(np.round(p, 2)), 0 if i < k else 1))
    return out


def csm_peaks(pk: list, th: float) -> list:
    """Frames of the CSM peaks: qualifying peaks minus the largest one per phase."""
    out = []
    for ph in (0, 1):
        q = sorted([p for p in pk if p[2] == ph and p[1] >= th], key=lambda p: -p[1])
        out += [p[0] for p in q[1:]]
    return sorted(out)


def lick_track(W, fr: tuple, on_ms: float, off_ms: float):
    """(t, P, rows or -1 for an interpolated frame) for the lick window, or None (as `session_licks`)."""
    t, P, rows = fr
    sel = (t >= on_ms - 0.5) & (t <= off_ms + 0.5)
    if sel.sum() < 3:
        return None
    tr = W._track(t[sel], P[sel], rows[sel] + 1000)
    if tr is None:
        return None
    tt, PP, ss = tr
    return tt, PP, np.where(ss == 9, -1, ss - 1000)


def speed(P: np.ndarray) -> np.ndarray:
    from scipy.signal import savgol_filter
    return np.linalg.norm(savgol_filter(P, 5, 2, deriv=1, axis=0) / DT, axis=1)


def near(flags: np.ndarray, f: int, lo: int, hi: int) -> bool:
    a, b = max(0, f + lo), min(len(flags), f + hi + 1)
    return bool(flags[a:b].any())


def session_worker(sid: str) -> dict:
    _env()
    import within_lick_correction as W
    from wfield_local.paths import PathResolver
    D = dirs()
    rv = PathResolver()
    an, date = sid.split("_")
    A = pd.read_csv(Path(rv.root("orofacial")) / V36 / "sessions" / an / date / "licks_all.csv",
                    usecols=["trial_id", "position", "lick_idx", "on_ms", "off_ms", "context", "contact"])
    A["lick_uid"] = np.arange(len(A))
    Lk = {}
    for run in ("2cam", "4cam"):
        d = pd.read_pickle(D[f"{run}_licks"] / f"{sid}.pkl")
        m = W.working_mask(sid, d["trial_id"].to_numpy())
        if m is None:
            return {"sid": sid, "skip": "no camera classes"}
        Lk[run] = d[m & (d.context == "post_cue").to_numpy()].set_index("lick_uid")
    uids = sorted(set(Lk["2cam"].index) | set(Lk["4cam"].index))
    A = A.set_index("lick_uid").loc[uids]
    lick_rows, peak_rows, step_rows = [], [], []
    for run in ("2cam", "4cam"):
        S = load_frames(D, run, sid)
        g = {k: (v.t_ms.to_numpy(float), v[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float), v.index.to_numpy())
             for k, v in S.groupby("trial_id")}
        vc = S.vcode.to_numpy()
        Lr = Lk[run]
        for uid, r in A.iterrows():
            if uid not in Lr.index:
                continue
            row = Lr.loc[uid]
            base = {"sid": sid, "animal": an, "lick_uid": int(uid), "run": run, "pos": row.pos,
                    "contact": bool(row.contact), "full_ok": bool(row.full_ok), "n_run": int(row.n_run),
                    "n_win": int(row.n_win)}
            fr = g.get(int(r.trial_id))
            tr = None if fr is None else lick_track(W, fr, r.on_ms, r.off_ms)
            if tr is None or not row.full_ok:
                lick_rows.append(base)
                continue
            t, P, rows = tr
            n = len(P)
            k = int(np.argmax(np.linalg.norm(P, axis=1)))
            sp = speed(P)
            pk = peaks_idx(sp, k)
            pre_ = tuple(p[1] for p in pk if p[2] == 0)
            post_ = tuple(p[1] for p in pk if p[2] == 1)
            same = (pre_, post_) == tuple(row.pk_sp) if isinstance(row.pk_sp, tuple) else False
            code = np.where(rows < 0, INTERP_CODE, vc[np.clip(rows, 0, None)])
            chg = np.r_[False, code[1:] != code[:-1]]           # chg[i]: boundary between i-1 and i
            interp = rows < 0
            # near a change: boundary index c in {f-1, f, f+1, f+2} (within 1.5 frames of frame f)
            nc_all = np.array([near(chg, f, -1, 2) for f in range(n)])
            ni_all = np.array([near(interp, f, -2, 2) for f in range(n)])
            base.update({"recomp_match": same, "k": k, "n_chg": int(chg.sum()), "n_interp": int(interp.sum()),
                         "frac_near_chg": float(nc_all.mean()), "frac_near_interp": float(ni_all.mean()),
                         "n_sets": int(len(set(code.tolist()))),
                         "share_side": float(np.mean([(c != INTERP_CODE and c < 100 and (c & 12) > 0) or c in (106, 107)
                                                      for c in code])) if run == "4cam" else np.nan,
                         "share_lone": float(np.mean((code >= 100) & (code != INTERP_CODE))) if run == "4cam" else
                         float(np.mean(np.isin(code, (2, 3))))})
            for th in THS:
                cp = csm_peaks(pk, th)
                base[f"csm{int(th)}"] = int(len(cp))
                for f in cp:
                    peak_rows.append({"sid": sid, "animal": an, "lick_uid": int(uid), "run": run, "pos": row.pos,
                                      "th": th, "frame": f, "kind": "csm", "near_chg": bool(nc_all[f]),
                                      "near_interp": bool(ni_all[f]), "code": int(code[f]),
                                      "lick_frac_near_chg": float(nc_all.mean()),
                                      "lick_frac_near_interp": float(ni_all.mean())})
            # main speed peaks (largest per phase, qualifying at 40) as a within-lick reference
            for ph in (0, 1):
                q = sorted([p for p in pk if p[2] == ph], key=lambda p: -p[1])
                if q:
                    f = q[0][0]
                    peak_rows.append({"sid": sid, "animal": an, "lick_uid": int(uid), "run": run, "pos": row.pos,
                                      "th": np.nan, "frame": f, "kind": "main", "near_chg": bool(nc_all[f]),
                                      "near_interp": bool(ni_all[f]), "code": int(code[f]),
                                      "lick_frac_near_chg": float(nc_all.mean()),
                                      "lick_frac_near_interp": float(ni_all.mean())})
            # step size: |second difference| of the raw track at change vs no-change boundaries (interior)
            if n >= 3:
                d2 = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=1)       # centred on frame 1..n-2
                ch_c = chg[1:-1] | chg[2:]                                       # a change next to the centre frame
                for v, c_ in zip(d2, ch_c):
                    step_rows.append((run, row.pos, bool(c_), float(v)))
            lick_rows.append(base)
    return {"sid": sid, "licks": pd.DataFrame(lick_rows), "peaks": pd.DataFrame(peak_rows),
            "steps": pd.DataFrame(step_rows, columns=["run", "pos", "at_chg", "d2_mm"])}


def extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    D = dirs()
    sids = sessions(D)
    assert len(sids) == 31, len(sids)
    res, fail = ak.fan_sessions(sids, session_worker, jobs=jobs)
    if fail:
        raise SystemExit(f"failed: {[(j, str(e)[:300]) for j, e in fail]}")
    out = D["out"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    skip = [v["sid"] for _, v in res if "skip" in v]
    print("skipped (no camera classes):", skip)
    for key in ("licks", "peaks", "steps"):
        T = pd.concat([v[key] for _, v in res if key in v], ignore_index=True)
        T.to_pickle(out / f"{key}.pkl")
        print(key, len(T))


# ------------------------------------------------------------------------------------------------ summary
def summary() -> None:
    D = dirs()
    T = D["out"] / "tables"
    L = pd.read_pickle(T / "licks.pkl")
    PK = pd.read_pickle(T / "peaks.pkl")
    ST = pd.read_pickle(T / "steps.pkl")
    ok = L[L.full_ok]
    print("recomputed peaks == pickled pk_sp:", ok.assign(m=ok.recomp_match.astype(float)).groupby("run").m.mean().round(4).to_dict())
    W = L.pivot_table(index=["sid", "animal", "lick_uid", "pos"], columns="run",
                      values=["full_ok", "n_run", "csm40", "csm50"], aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    for c in ("full_ok_2cam", "full_ok_4cam"):
        W[c] = W[c].fillna(False).astype(bool)
    rows = []
    for pos in POS_ORDER:
        w = W[W.pos == pos]
        both = w[w.full_ok_2cam & w.full_ok_4cam]
        o4 = w[~w.full_ok_2cam & w.full_ok_4cam]
        o2 = w[w.full_ok_2cam & ~w.full_ok_4cam]
        r = {"pos": pos, "n_licks": len(w), "full_ok_2cam": w.full_ok_2cam.mean(), "full_ok_4cam": w.full_ok_4cam.mean(),
             "n_run_med_2cam": w.n_run_2cam.median(), "n_run_med_4cam": w.n_run_4cam.median(),
             "n_both": len(both), "n_4cam_only": len(o4), "n_2cam_only": len(o2)}
        for th in (40, 50):
            for run in ("2cam", "4cam"):
                a = w[w[f"full_ok_{run}"]]
                r[f"csm{th}_{run}_all"] = (a[f"csm{th}_{run}"] >= 1).mean()
                r[f"csm{th}_{run}_both"] = (both[f"csm{th}_{run}"] >= 1).mean()
            r[f"csm{th}_4cam_4only"] = (o4[f"csm{th}_4cam"] >= 1).mean() if len(o4) else np.nan
            r[f"csm{th}_2cam_2only"] = (o2[f"csm{th}_2cam"] >= 1).mean() if len(o2) else np.nan
            # same-lick transitions on matched licks
            r[f"gain{th}"] = ((both[f"csm{th}_2cam"] == 0) & (both[f"csm{th}_4cam"] >= 1)).mean()
            r[f"lose{th}"] = ((both[f"csm{th}_2cam"] >= 1) & (both[f"csm{th}_4cam"] == 0)).mean()
        rows.append(r)
    S1 = pd.DataFrame(rows)
    S1.to_csv(D["out"] / "summary_coverage_threshold.csv", index=False, float_format="%.4f")
    # per animal, matched licks, CSM share at 40 in both runs (rule 8: visible in >= 3 animals?)
    pa = []
    for (an, pos), w in W[W.full_ok_2cam & W.full_ok_4cam].groupby(["animal", "pos"]):
        pa.append({"animal": an, "pos": pos, "n": len(w), **{f"csm{th}_{run}": (w[f"csm{th}_{run}"] >= 1).mean()
                                                            for th in (40, 50) for run in ("2cam", "4cam")}})
    pd.DataFrame(pa).to_csv(D["out"] / "summary_per_animal_matched.csv", index=False, float_format="%.4f")
    # H1: CSM peaks near a view-set change vs all frames of the same licks
    rows = []
    for (run, pos), p in PK.groupby(["run", "pos"]):
        for kind, th in (("csm", 40.0), ("csm", 50.0), ("main", None)):
            q = p[(p.kind == kind) & ((p.th == th) if th else p.th.isna())]
            if not len(q):
                continue
            rows.append({"run": run, "pos": pos, "kind": kind if th is None else f"csm{int(th)}", "n_peaks": len(q),
                         "near_chg": q.near_chg.mean(), "base_near_chg": q.lick_frac_near_chg.mean(),
                         "near_interp": q.near_interp.mean(), "base_near_interp": q.lick_frac_near_interp.mean()})
    S2 = pd.DataFrame(rows)
    S2["pos"] = pd.Categorical(S2.pos, POS_ORDER)
    S2 = S2.sort_values(["run", "kind", "pos"])
    S2.to_csv(D["out"] / "summary_peaks_vs_view_changes.csv", index=False, float_format="%.4f")
    # H1b: CSM share in licks whose view set never changes vs licks with changes (4cam, threshold 40)
    rows = []
    for run in ("2cam", "4cam"):
        o = ok[ok.run == run]
        for pos in POS_ORDER:
            q = o[o.pos == pos]
            const = q[q.n_chg == 0]
            chg = q[q.n_chg > 0]
            rows.append({"run": run, "pos": pos, "n": len(q), "share_const": len(const) / max(len(q), 1),
                         "csm40_const": (const.csm40 >= 1).mean(), "csm40_chg": (chg.csm40 >= 1).mean(),
                         "csm50_const": (const.csm50 >= 1).mean(), "csm50_chg": (chg.csm50 >= 1).mean(),
                         "n_chg_mean": q.n_chg.mean(), "share_side_frames": q.share_side.mean(),
                         "share_one_view": q.share_lone.mean()})
    S3 = pd.DataFrame(rows)
    S3.to_csv(D["out"] / "summary_constant_viewset.csv", index=False, float_format="%.4f")
    # step size at change vs no-change boundaries
    S4 = ST.groupby(["run", "pos", "at_chg"]).d2_mm.agg(["count", "median", lambda x: np.percentile(x, 90)])
    S4.columns = ["n", "median_mm", "p90_mm"]
    S4 = S4.reset_index()
    S4.to_csv(D["out"] / "summary_step_size.csv", index=False, float_format="%.4f")
    # where the CSM peaks sit: the run's LAST frame (the zero-padded edge of `_peaks`) and one-view-fill frames
    Q = PK.merge(L[["sid", "lick_uid", "run", "n_run"]], on=["sid", "lick_uid", "run"])
    Q["last"] = Q.frame == Q.n_run - 1
    Q["fill"] = (Q.code >= 100) & (Q.code != INTERP_CODE)
    Q.loc[Q.run == "2cam", "fill"] = Q.loc[Q.run == "2cam", "code"].isin([2, 3])
    rows = []
    for (run, pos), q in Q.groupby(["run", "pos"]):
        c = q[(q.kind == "csm") & (q.th == 40.0)]
        m = q[q.kind == "main"]
        o = ok[(ok.run == run) & (ok.pos == pos)]
        nl = c[~c["last"]].groupby(["sid", "lick_uid"]).size()
        has = o.set_index(["sid", "lick_uid"]).index.isin(nl.index)
        rows.append({"run": run, "pos": pos, "csm40_peaks": len(c), "csm_on_last_frame": c["last"].mean(),
                     "csm_on_fill_frame": c.fill.mean(), "main_on_last_frame": m["last"].mean(),
                     "csm40_share": (o.csm40 >= 1).mean(), "csm40_share_no_last_frame_peak": has.mean()})
    S5 = pd.DataFrame(rows)
    S5["pos"] = pd.Categorical(S5.pos, POS_ORDER)
    S5 = S5.sort_values(["run", "pos"])
    S5.to_csv(D["out"] / "summary_edge_and_fill_peaks.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(S1.round(3).to_string())
        print(S2.round(3).to_string())
        print(S3.round(3).to_string())
        print(S4.round(3).to_string())
        print(S5.round(3).to_string())
        print(pd.DataFrame(pa).round(3).to_string())


# ------------------------------------------------------------------------------------------------ example traces
def pick_examples(n_change: int = 6, n_same: int = 2) -> pd.DataFrame:
    D = dirs()
    L = pd.read_pickle(D["out"] / "tables" / "licks.pkl")
    W = L.pivot_table(index=["sid", "lick_uid", "pos"], columns="run", values=["full_ok", "csm40", "csm50"],
                      aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    W = W[W.full_ok_2cam.fillna(False).astype(bool) & W.full_ok_4cam.fillna(False).astype(bool)]
    rng = np.random.default_rng(SEED)
    out = []
    for pos in FOCUS:
        w = W[W.pos == pos]
        if pos.startswith("close"):    # GAIN: none at 50 in 2cam, >= 1 at 40 in 4cam
            ch = w[(w.csm50_2cam == 0) & (w.csm40_4cam >= 1)]
            lab = "gain"
        else:                          # LOSE: >= 1 at 50 in 2cam, none at 40 in 4cam
            ch = w[(w.csm50_2cam >= 1) & (w.csm40_4cam == 0)]
            lab = "lose"
        un = w[(w.csm50_2cam >= 1) == (w.csm40_4cam >= 1)]
        a = ch.iloc[rng.choice(len(ch), min(n_change, len(ch)), replace=False)].assign(kind=lab)
        b = un.iloc[rng.choice(len(un), min(n_same, len(un)), replace=False)].assign(kind="unchanged")
        out += [a, b]
    return pd.concat(out, ignore_index=True)


def lick_detail(sid: str, uids: list[int]) -> dict:
    """{(run, uid): dict(t, P, code, sp, k, pk, rows, S)} for plotting."""
    _env()
    import within_lick_correction as W
    from wfield_local.paths import PathResolver
    D = dirs()
    rv = PathResolver()
    an, date = sid.split("_")
    A = pd.read_csv(Path(rv.root("orofacial")) / V36 / "sessions" / an / date / "licks_all.csv",
                    usecols=["trial_id", "position", "on_ms", "off_ms"])
    out = {}
    for run in ("2cam", "4cam"):
        S = load_frames(D, run, sid)
        vc = S.vcode.to_numpy()
        for uid in uids:
            r = A.iloc[uid]
            v = S[S.trial_id == r.trial_id]
            fr = (v.t_ms.to_numpy(float), v[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float), v.index.to_numpy())
            tr = lick_track(W, fr, r.on_ms, r.off_ms)
            t, P, rows = tr
            k = int(np.argmax(np.linalg.norm(P, axis=1)))
            sp = speed(P)
            code = np.where(rows < 0, INTERP_CODE, vc[np.clip(rows, 0, None)])
            out[(run, uid)] = {"t": t, "P": P, "rows": rows, "code": code, "sp": sp, "k": k, "pk": peaks_idx(sp, k),
                               "trial_id": int(r.trial_id), "on": r.on_ms, "off": r.off_ms, "S": S}
    return out


def code_label(c: int, run: str) -> str:
    if c == INTERP_CODE:
        return "interp"
    if run == "2cam":
        return {1: "c1+c4", 2: "c1 only", 3: "c4 only"}.get(c, str(c))
    if c >= 100:
        return LONE.get(c - 100, "?").replace("cam", "c") + " fill"
    return "+".join(k.replace("cam", "c") for k, b in CAMBIT.items() if c & b)


def traces() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = dirs()
    E = pick_examples()
    E.to_csv(D["out"] / "examples_picked.csv", index=False)
    det = {}
    for sid, g in E.groupby("sid"):
        det.update({(sid, *k): v for k, v in lick_detail(sid, g.lick_uid.astype(int).tolist()).items()})
    # colors per view set (4cam) -- stable palette across figures
    pal = plt.get_cmap("tab20").colors
    allc = sorted({int(c) for (s, run, u), v in det.items() if run == "4cam" for c in v["code"]})
    cmap = {c: pal[i % 20] for i, c in enumerate(allc)}
    for pos in FOCUS:
        e = E[E.pos == pos].reset_index(drop=True)
        n = len(e)
        fig, axs = plt.subplots(2, n, figsize=(2.6 * n, 5.6), gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
        for j, r in e.iterrows():
            ax, bx = axs[0, j], axs[1, j]
            d2, d4 = det[(r.sid, "2cam", int(r.lick_uid))], det[(r.sid, "4cam", int(r.lick_uid))]
            t0 = d2["on"]
            # shading: 4cam view set per frame (top band), 2cam interpolated / one-camera frames (lower band)
            ymax = max(d2["sp"].max(), d4["sp"].max()) * 1.18
            for tt, c in zip(d4["t"], d4["code"]):
                ax.axvspan(tt - t0 - 2, tt - t0 + 2, ymin=0.93, ymax=1.0, color=cmap[int(c)], lw=0)
            for tt, c in zip(d2["t"], d2["code"]):
                col = {1: "0.85", 2: "#c9a0dc", 3: "#f4b183", INTERP_CODE: "k"}.get(int(c), "w")
                ax.axvspan(tt - t0 - 2, tt - t0 + 2, ymin=0.86, ymax=0.92, color=col, lw=0)
            for d, run, col, th in ((d2, "2cam", "#4c72b0", 50.0), (d4, "4cam", "#dd5132", 40.0)):
                x = d["t"] - t0
                ax.plot(x, d["sp"], color=col, lw=1.3, label=f"{run}")
                ax.axvline(x[d["k"]], color=col, lw=0.8, ls=":")
                cp = csm_peaks(d["pk"], th)
                mp = [p[0] for p in d["pk"] if p[1] >= th and p[0] not in cp]
                ax.plot(x[mp], d["sp"][mp], "o", mfc="none", mec=col, ms=5)
                ax.plot(x[cp], d["sp"][cp], "*", color=col, ms=11, mec="k", mew=0.4)
                bx.plot(x, d["P"][:, 0], color=col, lw=1.2)
                bx.plot(x[cp], d["P"][cp, 0], "*", color=col, ms=9, mec="k", mew=0.4)
            ax.set_ylim(0, ymax)
            ax.set_title(f"{r.sid[:4]} {r.sid[-4:]} uid {int(r.lick_uid)}\n{r.kind}: 2cam {int(r.csm50_2cam)}@50, "
                         f"4cam {int(r.csm40_4cam)}@40", fontsize=7.5)
            ax.tick_params(labelsize=7)
            bx.tick_params(labelsize=7)
            bx.set_xlabel("ms from lick onset", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("tip speed (mm/s)", fontsize=8)
                bx.set_ylabel("lateral ml (mm, + right)", fontsize=8)
            for a in (ax, bx):
                a.spines["top"].set_visible(False)
                a.spines["right"].set_visible(False)
        # legends
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        used = sorted({int(c) for (s, run, u), v in det.items() if run == "4cam" and s in set(e.sid)
                       and u in set(e.lick_uid.astype(int)) for c in v["code"]})
        h = [Line2D([], [], color="#4c72b0", label="two-camera (CSM threshold 50)"),
             Line2D([], [], color="#dd5132", label="four-camera (CSM threshold 40)"),
             Line2D([], [], ls="none", marker="*", color="0.4", ms=9, label="CSM peak"),
             Line2D([], [], ls="none", marker="o", mfc="none", mec="0.4", label="main qualifying peak"),
             Line2D([], [], ls=":", color="0.4", label="extension peak")]
        h2 = [Patch(color=cmap[c], label="4cam " + code_label(c, "4cam")) for c in used]
        h2 += [Patch(color="0.85", label="2cam c1+c4"), Patch(color="#c9a0dc", label="2cam c1 only"),
               Patch(color="#f4b183", label="2cam c4 only"), Patch(color="k", label="interpolated (one-frame gap)")]
        fig.legend(handles=h + h2, loc="lower center", ncol=6, fontsize=6.5, frameon=False)
        fig.suptitle(f"{pos.replace('_', ' ')}: same lick, two-camera vs four-camera 3-D tongue tip (August pre, "
                     f"post-cue licks, camera working period). Top band = four-camera view set per frame; "
                     f"lower band = two-camera views", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
        f = D["out"] / f"traces_{pos}.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        print("wrote", f)


# ------------------------------------------------------------------------------------------------ video frames
def frames(n_licks: int = 3, half: int = 4) -> None:
    """For the clearest close-spout gains (largest CSM prominence at 40 in 4cam, none in 2cam at 50, from
    examples_picked + the peaks table): cam4 / cam1 / cam2 / cam3 crops for frames peak-half .. peak+half, cleaned 2-D
    keypoints (filled = used, open = rejected) and the fused 3-D tongue point reprojected (cross)."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import o2_inference as oi
    from wfield_local import pose3d_side as PS
    from wfield_local.paths import PathResolver
    from scripts.pose3d import _june_factor
    D = dirs()
    rv = PathResolver()
    L = pd.read_pickle(D["out"] / "tables" / "licks.pkl")
    W = L.pivot_table(index=["sid", "lick_uid", "pos"], columns="run", values=["full_ok", "csm50", "csm40", "n_run"],
                      aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    G = W[W.pos.isin(["close_ipsi", "close_mid"]) & (W.csm50_2cam == 0) & (W.csm40_4cam >= 1)]
    PK = pd.read_pickle(D["out"] / "tables" / "peaks.pkl")
    pk = PK[(PK.run == "4cam") & (PK.kind == "csm") & (PK.th == 40.0)].merge(G[["sid", "lick_uid", "n_run_4cam"]])
    pk["last"] = pk.frame == pk.n_run_4cam - 1
    pk["cat"] = np.where(pk["last"], "last_frame", np.where(pk.near_chg, "interior_near_switch", "interior_no_switch"))
    pk = pk.sort_values(["sid", "lick_uid", "frame"]).drop_duplicates(["sid", "lick_uid"])
    # one session holding all three kinds of close-spout gain (one fusion re-run), first lick of each kind
    full = pk.groupby("sid").cat.nunique()
    sid0 = sorted(full[full == 3].index)[0]
    pick = pk[pk.sid == sid0].groupby("cat").head(1).head(n_licks)
    for sid, g in pick.groupby("sid"):
        an, date = sid.split("_")
        assert not _june_factor(date), "August only"
        det = lick_detail(sid, g.lick_uid.astype(int).tolist())
        Z = PS.fuse_session(rv, an, date, side_bundle=SIDE_BUNDLE)
        F, order, pts, side = Z["F"], Z["order"], Z["pts"], Z["side"]
        X = Z["parts"]["tongue"]["X"]
        cams = {c.get_name(): c for c in Z["parts"]["tongue"]["g"].cameras}
        readers = {c: cv2.VideoCapture(str(oi.videos_for(rv, an, date, c)[0])) for c in order}
        src = {"cam1": F.src1.to_numpy(), "cam4": F.src4.to_numpy(),
               **{c: np.where(F[f"i_{c}"].to_numpy() >= 0,
                              side[c][0].src_frame.to_numpy()[np.clip(F[f"i_{c}"].to_numpy(), 0, None)], -1)
                  for c in PS.SIDE}}
        mouth = {c: PS.project(cams[c], np.zeros((1, 3)))[0] for c in order}
        for r in g.itertuples():
            d4 = det[("4cam", int(r.lick_uid))]
            f0 = int(r.frame)
            n4 = len(d4["t"])
            idx = list(range(f0 - half, f0 + half + 3))            # may run past the tracked run (shows the tail)
            ts = d4["t"][f0] + 4.0 * (np.array(idx) - f0)
            ft = F.t_ms.to_numpy(float)
            cand = np.flatnonzero(F.trial_id.to_numpy() == d4["trial_id"])
            iF = []
            for tt in ts:
                j = cand[np.argmin(np.abs(ft[cand] - tt))] if len(cand) else -1
                iF.append(int(j) if j >= 0 and abs(ft[j] - tt) < 2.0 else -1)
            # check: the re-fused point equals the frames table
            dd = [np.linalg.norm(X[i] - d4["P"][j]) for i, j in zip(iF, idx)
                  if i >= 0 and 0 <= j < n4 and d4["rows"][j] >= 0]
            fig, axs = plt.subplots(len(order), len(idx), figsize=(1.55 * len(idx), 1.6 * len(order) + 0.8))
            for jc, c in enumerate(order):
                cap = readers[c]
                for jj, (i, fi) in enumerate(zip(iF, idx)):
                    ax = axs[jc, jj]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for s_ in ax.spines.values():
                        s_.set_visible(False)
                    vf = src[c][i] if i >= 0 else -1
                    im = None
                    if vf >= 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vf))
                        ok_, im = cap.read()
                    if im is None:
                        ax.text(0.5, 0.5, "no frame", ha="center", fontsize=6, transform=ax.transAxes)
                        continue
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    cs = 220 if c in ("cam1", "cam4") else 160
                    cx, cy = mouth[c]
                    x0 = int(np.clip(cx - cs / 2, 0, im.shape[1] - cs))
                    y0 = int(np.clip(cy - cs / 2, 0, im.shape[0] - cs))
                    ax.imshow(im[y0:y0 + cs, x0:x0 + cs])
                    q = pts["tongue"][jc, i]
                    if np.isfinite(q).all():
                        used = bool(Z["parts"]["tongue"]["used"][jc, i])
                        ax.plot(q[0] - x0, q[1] - y0, "o", ms=4, mfc="#ff3c3c" if used else "none", mec="#ff3c3c",
                                mew=1)
                    if np.isfinite(X[i]).all():
                        rp = PS.project(cams[c], X[i][None])[0]
                        ax.plot(rp[0] - x0, rp[1] - y0, "x", color="cyan", ms=5, mew=1.2)
                    if jc == 0:
                        if 0 <= fi < n4:
                            lab = code_label(int(d4["code"][fi]), "4cam")
                            ax.set_title(f"{ts[jj] - d4['on']:.0f} ms{' *' if fi == f0 else ''}\n{lab}\n"
                                         f"{d4['sp'][fi]:.0f} mm/s", fontsize=6)
                        else:
                            vv = int(Z["parts"]["tongue"]["views"][i]) if i >= 0 else 0
                            ax.set_title(f"{ts[jj] - d4['on']:.0f} ms\noutside run\nviews {vv}", fontsize=6)
                    if jj == 0:
                        ax.set_ylabel(c, fontsize=7)
            fig.suptitle(f"{sid} uid {int(r.lick_uid)} ({r.pos.replace('_', ' ')}): four-camera CSM peak at "
                         f"{d4['t'][f0] - d4['on']:.0f} ms (*), kind: {r.cat.replace('_', ' ')}. "
                         f"Red dot = cleaned 2-D tongue label (filled = used in 3-D, open = rejected); cyan x = fused "
                         f"3-D point reprojected. Re-fused vs table max {max(dd) if dd else np.nan:.3f} mm",
                         fontsize=7)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            f = D["out"] / f"frames_{sid}_uid{int(r.lick_uid)}.png"
            fig.savefig(f, dpi=140)
            plt.close(fig)
            print("wrote", f)


# ------------------------------------------------------------------------------------------------ subset re-fusion
#: every camera subset with >= 2 views, as CAMBIT bitmasks (cam1 1, cam4 2, cam2 4, cam3 8)
SUBSETS = [m for m in range(1, 16) if bin(m).count("1") >= 2]


def fuse_worker(sid: str) -> dict:
    """Re-run the production four-camera fusion (`pose3d_side.fuse_session`, the run's side bundle) for one session, in
    memory, and for every frame of the session's post-cue lick windows (4cam gate-4 pickle, working period) triangulate
    the tongue from EVERY >= 2-camera subset of the views production USED at that frame, with production's shifted
    cameras (per-session side-tongue offsets applied). Writes csm_examples/fused/<sid>.pkl; nothing else."""
    _env()
    import within_lick_correction as W
    from wfield_local import pose3d_side as PS
    from wfield_local.paths import PathResolver
    D = dirs()
    rv = PathResolver()
    an, date = sid.split("_")
    f_out = D["out"] / "fused" / f"{sid}.pkl"
    if f_out.exists():
        return {"sid": sid, "cached": True}
    Z = PS.fuse_session(rv, an, date, side_bundle=SIDE_BUNDLE)
    F, order = Z["F"], Z["order"]
    P = Z["parts"]["tongue"]
    gs, used, X, nv = P["g"], P["used"], P["X"], P["views"]
    pts = Z["pts"]["tongue"]
    T = PS.frames_table(Z)
    S = pd.read_csv(D["4cam_frames"] / f"{sid}_frames.csv.gz", usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap",
                                                                         "tongue_dv"])
    same_len = len(S) == len(T)
    dmax = float(np.nanmax(np.abs(S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float) -
                                  T[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)))) if same_len else np.nan
    bit = np.array([CAMBIT[c] for c in order])
    umask = (used * bit[:, None]).sum(0)
    # frames inside the session's post-cue lick windows (4cam pickle rows, working period)
    d = pd.read_pickle(D["4cam_licks"] / f"{sid}.pkl")
    m = W.working_mask(sid, d["trial_id"].to_numpy())
    d = d[m & (d.context == "post_cue").to_numpy()]
    A = pd.read_csv(Path(rv.root("orofacial")) / V36 / "sessions" / an / date / "licks_all.csv",
                    usecols=["trial_id", "on_ms", "off_ms"])
    A["lick_uid"] = np.arange(len(A))
    A = A.set_index("lick_uid").loc[d.lick_uid.to_numpy()]
    tr, tt = F.trial_id.to_numpy(), F.t_ms.to_numpy(float)
    order_idx = np.lexsort((tt, tr))
    rows, uidc = [], []
    by_trial = pd.Series(order_idx).groupby(tr[order_idx]).apply(np.asarray).to_dict()
    for uid, r in A.iterrows():
        ii = by_trial.get(int(r.trial_id))
        if ii is None:
            continue
        ii = ii[(tt[ii] >= r.on_ms - 0.5) & (tt[ii] <= r.off_ms + 0.5)]
        rows.append(ii)
        uidc.append(np.full(len(ii), uid))
    ii = np.concatenate(rows)
    out = pd.DataFrame({"lick_uid": np.concatenate(uidc), "iF": ii, "trial_id": tr[ii], "t_ms": tt[ii],
                        "nv": nv[ii], "used": umask[ii]})
    for k, ax in enumerate(("ml", "ap", "dv")):
        out[f"X_{ax}"] = X[ii, k]
    for c in order:
        out[f"rep_{c}"] = T[f"tongue_rep_{c}"].to_numpy(float)[ii]
    sub = pts[:, ii]
    U = umask[ii]
    for s in SUBSETS:
        sel = (U & s) == s
        Xs = np.full((len(ii), 3), np.nan)
        if sel.any():
            keep = np.array([(s & b) > 0 for b in bit])
            q = np.where(keep[:, None, None], sub[:, sel], np.nan)
            Xs[sel] = PS.triangulate_n(gs, q)[0]
        for k, ax in enumerate(("ml", "ap", "dv")):
            out[f"S{s}_{ax}"] = Xs[:, k]
    out.attrs = {"tongue_offsets": Z["stats"]["tongue_offsets"], "order": order, "table_match_maxabs_mm": dmax,
                 "table_same_len": same_len}
    f_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(f_out)
    return {"sid": sid, "n": len(out), "dmax": dmax}


def refuse(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    D = dirs()
    res, fail = ak.fan_sessions(sessions(D), fuse_worker, jobs=jobs)
    for _, v in res:
        print(v)
    if fail:
        raise SystemExit(f"failed: {[(j, str(e)[:300]) for j, e in fail]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "summary", "traces", "frames", "refuse", "fixes"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.step == "extract":
        extract(a.jobs)
    elif a.step == "summary":
        summary()
    elif a.step == "traces":
        traces()
    elif a.step == "refuse":
        if a.only:
            for s in a.only:
                print(fuse_worker(s))
        else:
            refuse(a.jobs)
    elif a.step == "fixes":
        import pose3d_csm_fixes as FX
        FX.run()
    else:
        frames()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
