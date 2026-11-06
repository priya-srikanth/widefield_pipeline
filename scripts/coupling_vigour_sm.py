"""LICK VIGOUR / KINEMATICS CONTROL OF THE LICK-STATE COUPLING RISE + SENSORY -> MOTOR COUPLING KINETICS (10-30;
PROVISIONAL, n = 4). Follow-up of scripts/interhemi_crossarea.py (FINDINGS "CROSS-AREA CORRELATION STRUCTURE").

TASK 1 -- VIGOUR CONTROL. interhemi_crossarea found that during licks (ITI docked, cued first licks) the orofacial /
  frontal-lateral network covaries MORE after the stroke (raw ITI-dock oro_all +0.25 every epoch 4/4; cued +0.18 / +0.09 /
  +0.19). Its residual model (lick_interhemispheric._design: spout dummies + 3-D lick angle) does not remove trial-to-trial
  lick VIGOUR, which drives exactly those areas. Here the SAME events / windows / area features / correlation code
  (functions imported from interhemi_crossarea and lick_interhemispheric) are re-run with per-event MOVEMENT COVARIATES
  added to the residual design (every lag x area regressed on [1, spout dummies, angle, movement]):
    reach_mm, speed_peak      3-D tongue reach at the peak (mm) and peak speed -- pose3d kinematics_all/licks.csv.gz
                              joined on lick_uid -> exact DAQ peak (licks_all.csv), as iti_licks.extract
    dur_ms, protr_px          cam4 lick duration (off - on) and peak protrusion (licks_all.csv; px, within session)
    n_lick_w1 / n_lick_w2     number of lick onsets (all cam4 licks, licks_all.csv) within +-0.5 s / +-1 s of the peak
                              (lick count / local rate; includes the lick itself)
    fr_lick .. fr_far         FaceRhythm master_pre_v1 mode-A factors (iti_licks.fr_unblanked, UNBLANKED, mean over
                              -0.1..+0.2 s around the peak = lick_covariate_reference.FR_WIN; a face / jaw movement
                              read-out available for ITI licks too; in the ITI the projection is 5 Hz -- caveat)
    jaw_amp (cued only)       per-TRIAL cam4 jaw peak |deflection| (jaw_trials.csv, jaw_pass_qc), joined on trial_id --
                              the jaw is NOT in any per-lick table, so this is trial-level jaw vigour
    miss_mm, ang3d_err (cued) 3-D landing error vs the target spout (undefined for ITI licks: no target)
  A covariate is used in a session x set when >= 50 % of its events have it; missing values are filled with the session
  mean plus a missingness indicator; covariates are z-scored within session. Designs:
    old   [1, spout, angle]                                   (= the published model; reproduces xa_tests)
    mov   old + movement covariates                           (PRIMARY)
    perm  old + the SAME covariate columns with rows permuted (controls for the lost degrees of freedom)
  Readouts (same code as interhemi_crossarea TASK 2, w1 = -0.5..+0.5 s): area x area noise correlation of the window mean
  ('slow') and within-window correlation ('fast'), raw and global-controlled (leave-pair-out partial | mean of other 43
  areas), category means (oro_all, nonoro_all, all_pairs, within_ipsi / contra, between_homo / hetero, oro_minus_nonoro)
  + the homotopic orofacial-pool noise correlation (raw / leave-pair-out global partial, ihf 'oro'). Also: coverage per
  animal x epoch; MOVEMENT VARIABILITY (per session SD of each covariate, raw and within-spout) and its change vs pre per
  animal; the share of orofacial-pool window-mean variance the movement covariates explain (R2 over the old design).

TASK 2 -- SENSORY -> MOTOR COUPLING KINETICS (Priya 10-30). Sensory S = SSp-n, SSp-m, SSs, SSp-ul; motor M = tjM1, ALM,
  MOp_rest, MOs_rest (split basis, disjoint). Pair groups (contra = _right, ipsi = _left):
    Sc->Mc, Si->Mi (intrahemispheric), Sc->Mi, Si->Mc (interhemispheric); 16 area pairs each, metrics averaged over pairs
  Per pair, on residual traces (lick sets: mov design PRIMARY, old design reference; rest: intercept only), optionally
  leave-pair-out global removed (one beta per area per window, ica._rm_global) as sensitivity; each event de-meaned,
  deconvolved s = x + tau dx/dt (tau 0.6 s primary, 0 raw), then the lick_interhemispheric metric code with the pair stacked
  [S, M]: cc(L) = corr(S(t), M(t + L)), + L = SENSORY LEADS; asym = mean cc(+0.05..+0.30) - mean cc(-0.30..-0.05);
  gc_dir = (gain S->M - null) - (gain M->S - null) (3 past samples, event-shuffle null). Windows: lick sets w2 (-1..+1 s,
  lags +-0.5 s), rest w1 (-0.5..+0.5 s, lags +-0.3 s). Derived groups: intra = mean of the two within-hemisphere groups,
  inter = mean of the two across, intra_minus_inter, contra_minus_ipsi (Sc->Mc - Si->Mi).
  GCaMP6s: rise 0.1-0.2 s, decay 0.5-1 s; a lag / asymmetry is SHAPE, not neural timing; sub-frame lags are not
  interpretable; nothing here is causal.
Statistics: per-animal tables printed first (rule 8); analysis_kit.boot_delta (paired nested) of each post epoch vs AUGUST
  pre (interhemi_crossarea._tests).
Outputs: <OUT_W>/coupling_vigour/ (cv_*), <OUT_W>/sensorimotor_kinetics/ (sm_*), each with figdata/ + figures/.
Usage: python scripts/coupling_vigour_sm.py [--jobs 4] [--labels ...] [--summarise] [--plot-only]
  (run under the frozen env, rr2/frozenenv.sh, for the FaceRhythm factors)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interhemi_crossarea as ica  # noqa: E402
import iti_licks as il  # noqa: E402
import lick_interhemispheric as lih  # noqa: E402

OUT_W = lih.OUT_W
O1, O2 = OUT_W / "coupling_vigour", OUT_W / "sensorimotor_kinetics"
IN = il.OUT
EPS, AN = lih.EPS, lih.AN
MIN_N = 30
MIN_COV = 0.5
LICK_SETS = ["cued", "iti_dock"]
KIN = ["reach_mm", "speed_peak", "dur_ms", "protr_px", "n_lick_w1", "n_lick_w2", "fr_lick", "fr_right", "fr_left",
       "fr_close", "fr_far", "jaw_amp", "miss_mm", "ang3d_err"]
KIN_SHOW = ["reach_mm", "speed_peak", "dur_ms", "protr_px", "n_lick_w1", "fr_lick", "jaw_amp", "miss_mm"]
FR_WIN = np.round(np.arange(-0.1, 0.2001, 0.02), 3)          # = lick_covariate_reference.FR_WIN
DESIGNS = ["old", "mov", "perm"]   # NOT "null": pandas reads the string "null" back as NaN
CATS = ["oro_all", "nonoro_all", "all_pairs", "within_ipsi", "within_contra", "between_homo", "between_hetero",
        "oro_minus_nonoro", "oro_ihnc"]
SENS = ["SSp-n", "SSp-m", "SSs", "SSp-ul"]
MOT = ["tjM1", "ALM", "MOp_rest", "MOs_rest"]
GROUPS = {"Sc->Mc": ("right", "right"), "Si->Mi": ("left", "left"), "Sc->Mi": ("right", "left"),
          "Si->Mc": ("left", "right")}
DGROUPS = ["Sc->Mc", "Si->Mi", "Sc->Mi", "Si->Mc", "intra", "inter", "intra_minus_inter", "contra_minus_ipsi"]
SM_METRICS = ["asym", "gc_dir", "cc0", "lag_peak", "gc_sm", "gc_ms"]
TAUS = (0.6, 0.0)


# ------------------------------------------------------------------------------------------------ per-event kinematics
_KC: dict = {}


def _ktab() -> pd.DataFrame:
    if "K" not in _KC:
        from wfield_local.paths import PathResolver
        p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
        _KC["K"] = pd.read_csv(p, usecols=["session", "lick_uid", "trial_id", "reach_mm", "speed_peak", "ang3d_err",
                                           "miss_mm"])
    return _KC["K"]


def session_kin(label: str, times: dict) -> dict:
    """{set: DataFrame (n events, KIN)} for exact DAQ peak times (NaN where absent)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    sd = oo.session_dir(rv, oc.RUN, an, date)
    A = pd.read_csv(sd / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s", "on_ms", "off_ms", "protrusion_max_px"])
    K = _ktab()
    K = K[K.session == f"{an}_{date}"].drop_duplicates("lick_uid").set_index("lick_uid")
    row = pd.Series(np.arange(len(A)), index=A.daq_peak_s.to_numpy(float))
    row = row[~row.index.duplicated()]
    on = np.sort(A.daq_on_s.to_numpy(float)[np.isfinite(A.daq_on_s.to_numpy(float))])
    J = None
    if (sd / "jaw_trials.csv").exists():
        J = pd.read_csv(sd / "jaw_trials.csv", usecols=["trial_id", "jaw_peak_abs_deflection", "jaw_pass_qc"])
        J = J[J.jaw_pass_qc.astype(bool)].drop_duplicates("trial_id").set_index("trial_id").jaw_peak_abs_deflection
    out = {}
    for st, t in times.items():
        t = np.asarray(t, float)
        u = row.reindex(t).to_numpy()
        ok = np.isfinite(u)
        ui = u[ok].astype(int)
        D = pd.DataFrame(np.nan, index=range(len(t)), columns=KIN)
        kk = K.reindex(ui)
        for c in ("reach_mm", "speed_peak", "miss_mm", "ang3d_err"):
            D.loc[ok, c] = kk[c].to_numpy(float)
        D.loc[ok, "dur_ms"] = (A.off_ms.to_numpy(float) - A.on_ms.to_numpy(float))[ui]
        D.loc[ok, "protr_px"] = A.protrusion_max_px.to_numpy(float)[ui]
        D["n_lick_w1"] = np.searchsorted(on, t + 0.5, side="right") - np.searchsorted(on, t - 0.5)
        D["n_lick_w2"] = np.searchsorted(on, t + 1.0, side="right") - np.searchsorted(on, t - 1.0)
        if st == "cued" and J is not None:
            D.loc[ok, "jaw_amp"] = J.reindex(kk.trial_id.to_numpy()).to_numpy(float)
        if st != "cued":
            D[["miss_mm", "ang3d_err"]] = np.nan        # no target for ITI licks
        try:
            grid = (t[:, None] + FR_WIN[None, :]).ravel()
            FU = il.fr_unblanked(an, date, grid, rv)
            for j, v in enumerate(il.FR_KEEP):
                V = FU[:, j].reshape(len(t), len(FR_WIN))
                fin = np.isfinite(V)
                D[v] = np.where(fin.sum(1) >= len(FR_WIN) / 2, np.nanmean(np.where(fin, V, np.nan), 1), np.nan)
        except FileNotFoundError:
            pass
        D["matched"] = ok
        out[st] = D
    return out


def mov_columns(Kd: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """z-scored movement covariates (+ missingness indicators) of one session x set; columns with < MIN_COV coverage or
    no variance are dropped."""
    Z, used = [], []
    for c in KIN:
        v = Kd[c].to_numpy(float)
        f = np.isfinite(v)
        if f.mean() < MIN_COV or np.nanstd(v) == 0:
            continue
        v = np.where(f, v, np.nanmean(v))
        Z.append((v - v.mean()) / v.std())
        used.append(c)
        if (~f).any():
            ind = (~f).astype(float)
            Z.append(ind - ind.mean())
    return (np.column_stack(Z) if Z else np.zeros((len(Kd), 0))), used


def _event_times(label: str, meta: pd.DataFrame, dock_t: np.ndarray) -> dict:
    """Exact DAQ peak times of the cued / iti_dock events, with the SAME masks as interhemi_crossarea._sets."""
    fl = np.load(IN / "firstlick_full" / f"{label}.npz", allow_pickle=False)
    ang = lih._angle(fl["side_ml"], fl["reach_mm"])
    out = {"cued": fl["t"].astype(float)[np.isfinite(ang)]}
    z = il._load(label)
    t = np.round(z["iti_t"].astype(float), 4)
    ang = lih._angle(z["iti_side"], meta.set_index("tk").reindex(t).reach_mm.to_numpy(float))
    ok = np.isfinite(ang) & np.isin(t, dock_t)
    out["iti_dock"] = z["iti_t"].astype(float)[ok]
    return out


class Item(tuple):
    def __repr__(self):
        return str(self[0])


# ------------------------------------------------------------------------------------------------ per session
def _cats(Z: np.ndarray, cm: dict) -> dict:
    r = {c: float(np.nanmean(Z[m])) for c, m in cm.items() if c in CATS}
    r["oro_minus_nonoro"] = r["oro_all"] - r["nonoro_all"]
    return r


def _sm_pairs(names: list[str]) -> list[tuple[str, int, int]]:
    out = []
    for g, (hs, hm) in GROUPS.items():
        for s in SENS:
            for m in MOT:
                a, b = f"{s}_{hs}", f"{m}_{hm}"
                if a in names and b in names:
                    out.append((g, names.index(a), names.index(b)))
    return out


def session_work(item):
    label, meta, dock_t = item
    from wfield_local import config, epochs
    seed = int(label[2:4]) * 10000 + int(label[-4:])
    an, ep = config.animal_of(label), epochs.epoch_of(label)
    base = {"label": label, "animal": an, "epoch": ep}
    S = ica._sets(label, meta, dock_t)
    times = _event_times(label, meta, dock_t)
    KD = session_kin(label, times)
    cov_rows, var_rows, cat_rows, sm_rows, curves = [], [], [], [], {}
    k_rng = 0
    for st in ("cued", "iti_dock", "rest"):
        if st not in S:
            continue
        X, pos, ang, cols = S[st]
        n = len(X)
        if n < MIN_N:
            continue
        X = X.astype(np.float64)
        D0 = lih._design(pos, ang, n)
        designs = {"old": D0}
        A = il.area_features(X, cols, label)                     # (n, T, 45)
        names = il.area_names(cols)
        lick = st in LICK_SETS
        if lick:
            if len(times[st]) != n:
                raise RuntimeError(f"{label} {st}: {len(times[st])} event times vs {n} events")
            Kd = KD[st]
            cov_rows.append({**base, "set": st, "n": n, "matched": float(Kd.matched.mean()),
                             **{f"cov_{c}": float(np.isfinite(Kd[c]).mean()) for c in KIN}})
            # movement variability: raw SD and within-spout SD (spout dummies removed) per covariate
            Dp = lih._design(pos, None, n)
            vr = {**base, "set": st, "n": n}
            for c in KIN:
                v = Kd[c].to_numpy(float)
                f = np.isfinite(v)
                if f.sum() < 10:
                    continue
                vr[f"mean_{c}"] = float(v[f].mean())
                vr[f"sd_{c}"] = float(v[f].std(ddof=1))
                rr = v[f] - Dp[f] @ np.linalg.lstsq(Dp[f], v[f], rcond=None)[0]
                vr[f"wsd_{c}"] = float(np.sqrt((rr ** 2).sum() / max(f.sum() - Dp.shape[1], 1)))
            var_rows.append(vr)
            Zm, used = mov_columns(Kd)
            if Zm.shape[1] and n >= max(MIN_N, 3 * (D0.shape[1] + Zm.shape[1])):
                designs["mov"] = np.c_[D0, Zm]
                perm = np.random.default_rng([seed, 7]).permutation(n)
                designs["perm"] = np.c_[D0, Zm[perm]]
            cov_rows[-1].update({"used": ",".join(used), "n_mov_cols": Zm.shape[1], "mov_ok": "mov" in designs})
        order = ica.area_order(names)
        names_o = [names[i] for i in order]
        cm = ica._cat_masks(names_o)
        oro_ix = [names.index(f"{a}_{h}") for a in ica.POOL for h in ("left", "right")]
        nonoro_ix = [j for j, nm in enumerate(names) if nm.rsplit("_", 1)[0] not in ica.POOL]
        PT = lih._pair_traces(X, cols, label)["oro"]               # (n, T, 2) [contra, ipsi] pooled orofacial
        sm_pairs = _sm_pairs(names)
        r2 = {}
        for dname, Dd in designs.items():
            Ar = lih._resid(A, Dd)
            A1 = Ar[:, lih.IDX_W1] if lick else Ar
            W = A1.mean(1)
            Fd = A1 - A1.mean(1, keepdims=True)
            # -------- TASK 1: cross-area categories (lick sets, all designs; rest old only as reference)
            for unit, Z in (("slow", W), ("fast", Fd.reshape(-1, Fd.shape[-1]))):
                C = np.cov(Z, rowvar=False)[np.ix_(order, order)]
                for var, M in (("raw", ica._cov2corr(C)), ("gs", ica._pcorr_lpo(C))):
                    zz = np.arctanh(np.clip(M, -0.999, 0.999))
                    cat_rows.append({**base, "set": st, "unit": unit, "variant": var, "design": dname, "n": n,
                                     **_cats(zz, cm)})
            # homotopic oro-pool noise correlation (raw / leave-pair-out global partial), w1 window mean
            E = lih._resid(PT, Dd)
            E1 = E[:, lih.IDX_W1] if lick else E
            G = Ar[..., [j for j in range(len(names)) if j not in oro_ix]].mean(-1)
            G1 = G[:, lih.IDX_W1] if lick else G
            ce_, ie_, ge_ = E1[..., 0].mean(1), E1[..., 1].mean(1), G1.mean(1)
            for row in cat_rows[-4:]:
                if row["unit"] == "slow":
                    row["oro_ihnc"] = (float(np.arctanh(np.clip(lih._corr(ce_, ie_), -0.999, 0.999)))
                                       if row["variant"] == "raw" else ica._partial_nc(ce_, ie_, ge_))
            if lick:
                r2[dname] = (float((W[:, oro_ix].sum(1) ** 2).sum()), float((W[:, nonoro_ix].mean(1) ** 2).sum()))
            # -------- TASK 2: S -> M kinetics (lick sets: old + mov; rest: old)
            if dname == "perm":
                continue
            Ew_all = Ar                                            # lick sets: w2 = the full -1..+1 s window; rest: w1
            maxl = lih.MAXLW["w2" if lick else "w1"]
            for gsv in ("raw", "gsr"):
                for tau in TAUS:
                    per = {g: [] for g in GROUPS}
                    cur = {g: [] for g in GROUPS}
                    for g, si, mi in sm_pairs:
                        Ew = Ew_all[..., [si, mi]]
                        if gsv == "gsr":
                            Gp = Ew_all[..., [j for j in range(len(names)) if j not in (si, mi)]].mean(-1)
                            Ew = ica._rm_global(Ew, Gp)
                        Ed = Ew - Ew.mean(1, keepdims=True)
                        Et = lih._deconv(Ed, tau)
                        Et = Et - Et.mean(1, keepdims=True)
                        cc = lih.xcorr(Et, maxl)
                        k_rng += 1
                        gc = lih.granger(Et, np.random.default_rng([seed, k_rng]))
                        per[g].append({**lih.cc_summary(cc), "gc_dir": gc["gc_dir"],
                                       "gc_sm": gc["gc_ci"] - gc["gc_ci_null"], "gc_ms": gc["gc_ic"] - gc["gc_ic_null"]})
                        cur[g].append(cc)
                    gm = {}
                    for g in GROUPS:
                        if per[g]:
                            gm[g] = pd.DataFrame(per[g]).mean().to_dict()
                            curves[(st, dname, gsv, tau, g)] = np.mean(cur[g], 0)
                    if len(gm) == 4:
                        gm["intra"] = {m: 0.5 * (gm["Sc->Mc"][m] + gm["Si->Mi"][m]) for m in SM_METRICS}
                        gm["inter"] = {m: 0.5 * (gm["Sc->Mi"][m] + gm["Si->Mc"][m]) for m in SM_METRICS}
                        gm["intra_minus_inter"] = {m: gm["intra"][m] - gm["inter"][m] for m in SM_METRICS}
                        gm["contra_minus_ipsi"] = {m: gm["Sc->Mc"][m] - gm["Si->Mi"][m] for m in SM_METRICS}
                    for g, v in gm.items():
                        sm_rows.append({**base, "set": st, "design": dname, "gs": gsv, "tau": tau, "group": g, "n": n,
                                        **{m: v[m] for m in SM_METRICS}})
        if lick and "mov" in r2:
            var_rows[-1].update({"r2mov_oro": 1 - r2["mov"][0] / r2["old"][0], "r2null_oro": 1 - r2["perm"][0] / r2["old"][0],
                                 "r2mov_nonoro": 1 - r2["mov"][1] / r2["old"][1],
                                 "r2null_nonoro": 1 - r2["perm"][1] / r2["old"][1]})
    return (pd.DataFrame(cov_rows), pd.DataFrame(var_rows), pd.DataFrame(cat_rows), pd.DataFrame(sm_rows), curves)


# ------------------------------------------------------------------------------------------------ run
def run(labels, jobs) -> None:
    from wfield_local import analysis_kit as ak
    M = lih._meta()
    Aud = pd.read_csv(IN / "iti_dock_audit.csv", usecols=["label", "t", "dock"])
    Aud = Aud[Aud.dock]
    dk = {x: np.round(g.t.to_numpy(float), 4) for x, g in Aud.groupby("label")}
    empty = pd.DataFrame(columns=["tk", "reach_mm", "gap", "since_stop", "to_strobe"])
    items = [Item((x, M.get(x, empty), dk.get(x, np.zeros(0)))) for x in labels]
    t0 = time.time()
    res, fail = ak.fan_sessions(items, session_work, jobs=min(jobs, 4), key=lambda it: it[0])
    print(f"sessions done {time.time() - t0:.0f} s; failed {len(fail)}", flush=True)
    if fail:
        print("FAILED:", [str(f[0]) for f in fail][:10], flush=True)
    tag = "_test" if len(labels) < 100 else ""
    if fail and not tag:
        raise SystemExit("refusing to write a partial table")
    for d in (O1, O2):
        (d / "figdata").mkdir(parents=True, exist_ok=True)
    parts = [pd.concat([v[i] for _k, v in res], ignore_index=True) for i in range(4)]
    parts[0].to_csv(O1 / f"cv_coverage_sessions{tag}.csv", index=False)
    parts[1].to_csv(O1 / f"cv_variability_sessions{tag}.csv", index=False)
    parts[2].to_csv(O1 / f"cv_category_sessions{tag}.csv.gz", index=False)
    parts[3].to_csv(O2 / f"sm_sessions{tag}.csv.gz", index=False)
    curves = {(str(k[0]), *kk): c for k, v in res for kk, c in v[4].items()}
    with open(O2 / "figdata" / f"sm_curves{tag}.pkl", "wb") as fh:
        pickle.dump(curves, fh)


# ------------------------------------------------------------------------------------------------ summaries
def _print_animal(S, keys, m, title, filt):
    q = S.query(filt)
    P = q.groupby(keys + ["animal", "epoch"])[m].mean().unstack("epoch").reindex(columns=EPS)
    print(f"\n=== PER-ANIMAL ({title}) : {m}\n" + P.round(3).to_string(), flush=True)


def summarise1() -> None:
    Cv = ica._aug(pd.read_csv(O1 / "cv_coverage_sessions.csv"))
    ccols = ["matched"] + [f"cov_{c}" for c in KIN]
    CA = Cv.groupby(["set", "animal", "epoch"])[ccols + ["n", "mov_ok"]].mean().round(3).reset_index()
    CA.to_csv(O1 / "cv_coverage_animal.csv", index=False)
    print("\n=== COVERAGE (fraction of events with each covariate), animal x epoch\n"
          + CA[["set", "animal", "epoch", "n", "mov_ok", "matched", "cov_reach_mm", "cov_speed_peak", "cov_dur_ms",
                "cov_fr_lick", "cov_jaw_amp", "cov_miss_mm"]].to_string(), flush=True)
    V = ica._aug(pd.read_csv(O1 / "cv_variability_sessions.csv"))
    vm = [c for c in V.columns if c.startswith(("sd_", "wsd_", "mean_", "r2"))]
    for st in LICK_SETS:
        for c in [f"wsd_{k}" for k in KIN_SHOW] + ["r2mov_oro", "r2null_oro", "r2mov_nonoro"]:
            if c in V and V[V.set == st][c].notna().any():
                _print_animal(V, ["set"], c, f"{st} movement variability", f"set == '{st}'")
    TV = ica._tests(V, ["set"], vm, np.random.default_rng(0))
    TV.to_csv(O1 / "cv_variability_tests.csv", index=False)
    S = ica._aug(pd.read_csv(O1 / "cv_category_sessions.csv.gz", keep_default_na=False, na_values=[""]))
    S["design"] = S.design.replace({"null": "perm"})        # session tables of the first full run (10-30) said "null"
    for st in LICK_SETS:
        for var in ("raw", "gs"):
            for m in ("oro_all", "oro_minus_nonoro"):
                _print_animal(S, ["design"], m, f"{st} slow {var}",
                              f"set == '{st}' and unit == 'slow' and variant == '{var}'")
    T = ica._tests(S, ["set", "unit", "variant", "design"], CATS, np.random.default_rng(0))
    T.to_csv(O1 / "cv_category_tests.csv", index=False)
    # side-by-side: old vs mov vs null, slow unit, key categories
    sb = []
    for (st, var, m, ep), g in T[(T.unit == "slow")].groupby(["set", "variant", "metric", "epoch"]):
        r = {"set": st, "variant": var, "metric": m, "epoch": ep}
        for _i, x in g.iterrows():
            r[f"{x.design}"] = f"{x.delta:+.3f} [{x.lo:+.3f},{x.hi:+.3f}] {x.n_same_sign}/{x.n_animals}"
            r[f"{x.design}_d"] = x.delta
            for a in AN:
                r[f"{x.design}_{a}"] = x.get(f"d_{a}", np.nan)
        sb.append(r)
    SB = pd.DataFrame(sb)
    SB.to_csv(O1 / "cv_side_by_side.csv", index=False)
    print("\n=== SIDE BY SIDE (slow noise corr, change vs pre)\n"
          + SB[SB.metric.isin(["oro_all", "nonoro_all", "all_pairs", "oro_minus_nonoro", "oro_ihnc", "between_homo"])]
          [["set", "variant", "metric", "epoch", "old", "mov", "perm"]].to_string(), flush=True)
    # published reference (interhemi_crossarea xa_tests)
    try:
        X = pd.read_csv(ica.O2 / "xa_tests.csv")
        X = X[X.set.isin(LICK_SETS) & (X.unit == "slow") & X.metric.isin(["oro_all", "all_pairs"])]
        print("\n=== PUBLISHED xa_tests (for the 'old' check)\n"
              + X[["set", "variant", "metric", "epoch", "delta", "lo", "hi", "n_same_sign"]].round(3).to_string())
    except FileNotFoundError:
        pass
    with open(O1 / "figdata" / "cv_tables.pkl", "wb") as fh:
        pickle.dump({"T": T, "TV": TV, "CA": CA, "SB": SB}, fh)


def summarise2() -> None:
    S = ica._aug(pd.read_csv(O2 / "sm_sessions.csv.gz"))
    for st, dsg in (("cued", "mov"), ("iti_dock", "mov"), ("rest", "old")):
        for m in ("asym", "gc_dir"):
            _print_animal(S, ["group"], m, f"{st} {dsg} raw tau 0.6",
                          f"set == '{st}' and design == '{dsg}' and gs == 'raw' and tau == 0.6")
    T = ica._tests(S, ["set", "design", "gs", "tau", "group"], SM_METRICS, np.random.default_rng(0))
    T.to_csv(O2 / "sm_tests.csv", index=False)
    # pre levels per animal (does S lead at baseline?)
    P = S[S.epoch == "pre"].groupby(["set", "design", "gs", "tau", "group", "animal"])[SM_METRICS].mean().reset_index()
    P.to_csv(O2 / "sm_pre_levels_animal.csv", index=False)
    with open(O2 / "figdata" / "sm_curves.pkl", "rb") as fh:
        C = pickle.load(fh)
    meta = S.drop_duplicates("label")[["label", "animal", "epoch"]]
    CG = {}
    for (st, dsg, gsv, tau, g) in {k[1:] for k in C}:
        for ep in EPS:
            per = []
            for a in AN:
                labs = meta[(meta.animal == a) & (meta.epoch == ep)].label
                cs = [C[(lab, st, dsg, gsv, tau, g)] for lab in labs if (lab, st, dsg, gsv, tau, g) in C]
                if cs:
                    per.append(np.mean(cs, 0))
            if per:
                CG[(st, dsg, gsv, tau, g, ep)] = (np.mean(per, 0), np.stack(per))
    with open(O2 / "figdata" / "sm_tables.pkl", "wb") as fh:
        pickle.dump({"T": T, "P": P, "CG": CG}, fh)


# ------------------------------------------------------------------------------------------------ figures
def _wrap(s, w=160):
    return "\n".join(textwrap.wrap(s, w))


SET_LAB = {"cued": "CUED first licks", "iti_dock": "ITI licks, spout docked", "rest": "DOCKED QUIET REST"}


def _pts(ax, r, xx, ep, show_animals=True, filled=True, ms=5):
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    if show_animals:
        for a in AN:
            v = r.get(f"d_{a}", np.nan)
            if np.isfinite(v):
                ax.plot(xx - 0.06, v, ANIMAL_MARKER[a], mfc="none", mec=EPOCH_LINE[ep], ms=3.2, alpha=0.7)
    ax.errorbar(xx + 0.04, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="o", color=EPOCH_LINE[ep], ms=ms,
                lw=1.4, mfc=EPOCH_LINE[ep] if filled else "white", mec="k" if (r.excl0 and r.n_same_sign >= 3)
                else EPOCH_LINE[ep])


def _legend(fig, extra=()):
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    hd = [Line2D([], [], marker="o", ls="-", color=EPOCH_LINE[ep], ms=5) for ep in EPS[1:]]
    hd += [Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", color="0.4", ms=5) for a in AN]
    lb = [f"{ep} - pre (pooled 95% CI; black edge = excl. 0 and >= 3/4)" if i == 0 else f"{ep} - pre"
          for i, ep in enumerate(EPS[1:])] + AN
    for h, l_ in extra:
        hd.append(h)
        lb.append(l_)
    fig.legend(hd, lb, loc="outside lower center", ncol=5 if extra else min(len(hd), 9), fontsize=8, frameon=False)


def plot1() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with open(O1 / "figdata" / "cv_tables.pkl", "rb") as fh:
        D = pickle.load(fh)
    T, TV = D["T"], D["TV"]
    (O1 / "figures").mkdir(parents=True, exist_ok=True)
    # FIG 1: old vs mov vs null for the key categories, slow unit; rows = set, cols = raw / gs
    cats = [("oro_all", "orofacial\nnetwork"), ("nonoro_all", "other\nbilateral"), ("oro_minus_nonoro", "oro minus\nother"),
            ("oro_ihnc", "oro pool\nhomotopic")]
    dl = [("old", "publ.", False), ("mov", "+mov", True), ("perm", "+perm", False)]
    fig, axs = plt.subplots(2, 2, figsize=(17, 9.5), sharey=True, layout="constrained")
    for i, st in enumerate(LICK_SETS):
        for j, var in enumerate(("raw", "gs")):
            ax = axs[i, j]
            xt, xl = [], []
            for ci, (c, clab) in enumerate(cats):
                for di, (dsg, dlab, filled) in enumerate(dl):
                    x0 = ci * 3.6 + di * 1.1
                    xt.append(x0)
                    xl.append(dlab)
                    for k, ep in enumerate(EPS[1:]):
                        r = T[(T.set == st) & (T.unit == "slow") & (T.variant == var) & (T.design == dsg)
                              & (T.metric == c) & (T.epoch == ep)]
                        if r.empty:
                            continue
                        _pts(ax, r.iloc[0], x0 + (k - 1) * 0.3, ep, show_animals=(dsg == "mov"), filled=filled,
                             ms=4.5)
                ax.text(ci * 3.6 + 1.1, 1.0, clab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=8.5)
                if ci:
                    ax.axvline(ci * 3.6 - 0.75, color="0.85", lw=0.8)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(xt, xl, fontsize=6.5)
            ax.set_title(f"{SET_LAB[st]}: {'RAW' if var == 'raw' else 'GLOBAL-CONTROLLED'}", fontsize=9.5, pad=22)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_ylabel("change vs pre, mean noise corr\n(Fisher z, 1-s window)", fontsize=8.5)
    _legend(fig, [(Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=5), "open: publ. = published model, +perm = permuted movement"),
                  (Line2D([], [], marker="o", ls="", color="0.3", ms=5), "filled: +mov = movement partialled (per-animal dots)")])
    fig.suptitle(_wrap("Does the lick-state coupling rise survive partialling LICK VIGOUR? Cross-area noise correlation "
                       "(residual 1-s window mean) with the published residual model (spout + angle), with per-event "
                       "movement covariates added (3-D reach, peak speed, lick duration, protrusion, lick count +-0.5 / "
                       "+-1 s, FaceRhythm face factors; cued: + trial jaw amplitude, landing error), and with the same "
                       "covariates row-permuted (degrees-of-freedom control). PROVISIONAL n = 4"), fontsize=9.5)
    fig.savefig(O1 / "figures" / "cv_coupling_vs_movement.png", dpi=120)
    plt.close(fig)
    # FIG 2: movement variability change vs pre per animal (within-spout SD) + movement R2 of the oro pool
    mets = [(f"wsd_{k}", k) for k in KIN_SHOW] + [("r2mov_oro", "R2 movement\n(oro pool)"),
                                                  ("r2mov_nonoro", "R2 movement\n(other areas)")]
    fig, axs = plt.subplots(2, 1, figsize=(17, 8.5), layout="constrained")
    for ax, st in zip(axs, LICK_SETS):
        xs, xl = [], []
        for xi, (m, lab) in enumerate(mets):
            g = TV[(TV.set == st) & (TV.metric == m)]
            if g.empty:
                continue
            pre = g.pre_level.iloc[0]
            sc = abs(pre) if (m.startswith("wsd_") and np.isfinite(pre) and pre != 0) else 1.0
            for k, ep in enumerate(EPS[1:]):
                r = g[g.epoch == ep]
                if r.empty:
                    continue
                r = r.iloc[0].copy()
                for f_ in ["delta", "lo", "hi"] + [f"d_{a}" for a in AN]:
                    r[f_] = r[f_] / sc
                _pts(ax, r, xi + (k - 1) * 0.27, ep)
            xs.append(xi)
            xl.append(f"{lab}\n(pre {pre:.3g})" if m.startswith("wsd_") else lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(xs, xl, fontsize=7.5)
        ax.set_ylabel("within-spout SD: change / pre SD\nR2: change (absolute)", fontsize=8.5)
        ax.set_title(f"{SET_LAB[st]}: does MOVEMENT VARIABILITY rise after the stroke?", fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    _legend(fig)
    fig.suptitle(_wrap("Movement variability per session (SD of each covariate after removing the spout means), change vs "
                       "August pre as a fraction of the pre SD; right: share of orofacial-pool / other-area residual "
                       "window-mean variance explained by the movement covariates. PROVISIONAL n = 4"), fontsize=9.5)
    fig.savefig(O1 / "figures" / "cv_movement_variability.png", dpi=120)
    plt.close(fig)


GCOL = {"Sc->Mc": "#01665e", "Si->Mi": "#8c510a", "Sc->Mi": "#5ab4ac", "Si->Mc": "#d8b365"}


def plot2() -> None:
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    with open(O2 / "figdata" / "sm_tables.pkl", "rb") as fh:
        D = pickle.load(fh)
    T, CG = D["T"], D["CG"]
    (O2 / "figures").mkdir(parents=True, exist_ok=True)
    sets = [("cued", "mov"), ("iti_dock", "mov"), ("rest", "old")]
    # FIG 1: S->M cross-correlation curves (tau 0.6, raw), per group, epochs overlaid
    fig, axs = plt.subplots(3, 4, figsize=(17, 10), sharex="row", layout="constrained")
    for i, (st, dsg) in enumerate(sets):
        for j, g in enumerate(GROUPS):
            ax = axs[i, j]
            for ep in EPS:
                v = CG.get((st, dsg, "raw", 0.6, g, ep))
                if v is None:
                    continue
                L = (np.arange(len(v[0])) - (len(v[0]) - 1) // 2) * lih.DT
                ax.plot(L, v[0], color=EPOCH_LINE[ep], lw=1.6, label=ep)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"{SET_LAB[st]}: {g}", fontsize=9)
            if j == 0:
                ax.set_ylabel("corr(S(t), M(t+L))\n(deconvolved tau 0.6)", fontsize=8.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_xlabel("lag L (s); + = sensory leads", fontsize=8)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(_wrap("SENSORY -> MOTOR cross-correlation of residual, deconvolved (tau 0.6 s) traces, mean of the 16 area "
                       "pairs per group (S = SSp-n, SSp-m, SSs, SSp-ul; M = tjM1, ALM, MOp, MOs; c = contralesional, "
                       "i = ipsilesional), mean over animals of session means. Lick sets: movement-partialled residuals. "
                       "GCaMP6s: shape only, not neural lags. PROVISIONAL n = 4"), fontsize=9.5)
    fig.savefig(O2 / "figures" / "sm_xcorr_curves.png", dpi=120)
    plt.close(fig)
    # FIG 2: change vs pre in asym and gc_dir per group (primary variant), plus pre level text
    grp = ["Sc->Mc", "Si->Mi", "Sc->Mi", "Si->Mc", "intra_minus_inter", "contra_minus_ipsi"]
    glab = ["Sc->Mc", "Si->Mi", "Sc->Mi", "Si->Mc", "intra -\ninter", "Sc->Mc -\nSi->Mi"]
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.5), sharey="row", layout="constrained")
    for j, (st, dsg) in enumerate(sets):
        for i, m in enumerate(("asym", "gc_dir")):
            ax = axs[i, j]
            for xi, g in enumerate(grp):
                for k, ep in enumerate(EPS[1:]):
                    r = T[(T.set == st) & (T.design == dsg) & (T.gs == "raw") & (T.tau == 0.6) & (T.group == g)
                          & (T.metric == m) & (T.epoch == ep)]
                    if not r.empty:
                        _pts(ax, r.iloc[0], xi + (k - 1) * 0.27, ep)
                r = T[(T.set == st) & (T.design == dsg) & (T.gs == "raw") & (T.tau == 0.6) & (T.group == g)
                      & (T.metric == m)]
                if not r.empty:
                    ax.text(xi, 1.0, f"pre {r.pre_level.iloc[0]:+.3f}", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=7)
            ax.axvspan(3.5, 5.5, color="0.95", zorder=0)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(len(grp)), glab, fontsize=8)
            ax.set_title(f"{SET_LAB[st]}: {'xcorr asymmetry (+ = S leads)' if m == 'asym' else 'Granger S->M minus M->S'}",
                         fontsize=9.5, pad=14)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    axs[0, 0].set_ylabel("change vs pre, xcorr asymmetry", fontsize=8.5)
    axs[1, 0].set_ylabel("change vs pre, Granger direction", fontsize=8.5)
    _legend(fig)
    fig.suptitle(_wrap("Sensory -> motor DIRECTION over recovery (tau 0.6 s; lick sets movement-partialled, rest intercept "
                       "only): change vs August pre per pair group, per-animal dots + pooled paired bootstrap; grey = "
                       "contrasts (intra- minus interhemispheric; contralesional minus ipsilesional hemisphere). Text = "
                       "pooled pre level. PROVISIONAL n = 4"), fontsize=9.5)
    fig.savefig(O2 / "figures" / "sm_direction_change.png", dpi=120)
    plt.close(fig)
    # FIG 3: sensitivity -- intra / inter group means under each variant (old vs mov; raw vs gsr; tau 0 vs 0.6)
    vlist = [("mov", "raw", 0.6, "mov raw\ntau .6"), ("mov", "gsr", 0.6, "mov GSR\ntau .6"),
             ("old", "raw", 0.6, "published\nmodel tau .6"), ("mov", "raw", 0.0, "mov raw\ntau 0")]
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.5), sharey="row", layout="constrained")
    for j, (st, _d) in enumerate(sets):
        vv = vlist if st != "rest" else [("old", "raw", 0.6, "raw\ntau .6"), ("old", "gsr", 0.6, "GSR\ntau .6"),
                                         ("old", "raw", 0.0, "raw\ntau 0")]
        for i, m in enumerate(("asym", "gc_dir")):
            ax = axs[i, j]
            xt, xl = [], []
            for gi, g in enumerate(("intra", "inter")):
                for vi, (dsg, gsv, tau, lab) in enumerate(vv):
                    x0 = gi * (len(vv) + 1) + vi
                    xt.append(x0)
                    xl.append(lab)
                    for k, ep in enumerate(EPS[1:]):
                        r = T[(T.set == st) & (T.design == dsg) & (T.gs == gsv) & (T.tau == tau) & (T.group == g)
                              & (T.metric == m) & (T.epoch == ep)]
                        if not r.empty:
                            _pts(ax, r.iloc[0], x0 + (k - 1) * 0.27, ep, show_animals=False)
                ax.text(gi * (len(vv) + 1) + (len(vv) - 1) / 2, 1.0, "INTRAhemispheric" if g == "intra" else
                        "INTERhemispheric", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(xt, xl, fontsize=7)
            ax.set_title(f"{SET_LAB[st]}: {'xcorr asymmetry' if m == 'asym' else 'Granger S->M minus M->S'}",
                         fontsize=9.5, pad=16)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    axs[0, 0].set_ylabel("change vs pre, xcorr asymmetry", fontsize=8.5)
    axs[1, 0].set_ylabel("change vs pre, Granger direction", fontsize=8.5)
    _legend(fig)
    fig.suptitle(_wrap("Sensitivity of the sensory -> motor direction change: movement-partialled vs published residual "
                       "model, global signal removed (leave-pair-out), raw (tau 0) vs deconvolved traces; pooled 95% CI "
                       "only. PROVISIONAL n = 4"), fontsize=9.5)
    fig.savefig(O2 / "figures" / "sm_sensitivity.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot1()
        plot2()
        return
    if not a.summarise:
        if "COMBINED_ENCODING_FR" not in os.environ:
            raise SystemExit("run under the frozen env (rr2/frozenenv.sh): COMBINED_ENCODING_FR unset")
        from wfield_local import writeguard
        for d in (O1, O2):
            writeguard.assert_writable(d)
        run(a.labels or il.labels(), a.jobs)
        if a.labels:
            print(pd.read_csv(O1 / "cv_coverage_sessions_test.csv").round(3).T.to_string())
            S = pd.read_csv(O1 / "cv_category_sessions_test.csv.gz")
            print(S[S.unit == "slow"].groupby(["set", "variant", "design"])[["oro_all", "nonoro_all", "oro_ihnc"]]
                  .mean().round(3).to_string())
            S2 = pd.read_csv(O2 / "sm_sessions_test.csv.gz")
            print(S2[S2.tau == 0.6].groupby(["set", "design", "gs", "group"])[SM_METRICS].mean().round(3).to_string())
            return
    summarise1()
    summarise2()
    plot1()
    plot2()


if __name__ == "__main__":
    main()
