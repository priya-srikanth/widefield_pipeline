"""LEARNED, CONTEXT-SPECIFIC COMPENSATION -- behaviour-only tests (Priya 10-30; RECOVERY_MECHANISMS_LIT_20261030.md 1a).

DESIGN (fixed before any post-stroke number): docs/status/LEARNED_COMPENSATION_PREREG_20261030.md -- read it first.
  T1 learning signature: far-contra cued FIRST-lick launch error e_L (and engaged hit rate) within a session (late third
     - early third, W) and overnight (early third of day d+1 - late third of day d, O), post days 1-14 vs August pre.
  T2 carry-over: ITI launch angle relative to the August-pre ITI licks after the SAME previous spout in the SAME ITI phase
     (retract / docked / outbound; iti_dock_audit), far contra vs the other five; C2 = [d(FC,ret) - d(FC,dock)] -
     [d(O5,ret) - d(O5,dock)].
  T3 docked ITI landing angle relative to the previous trial's spout: D_off (vs pre at the same previous spout), D_fc,
     gain on the previous spout angle, GAP = D_off - cued first-lick endpoint error (capacity vs performance).

SIGNS: every angle + = CONTRALESIONAL (mouse right; licks_scored is already sign-forced). e_L / e_E - = ipsilesional of
the pre-stroke hit pattern. W + = later trials more contralesional; O - = overnight loss.

Inputs (read-only): <v36>/imaging_encoding/within_lick_correction/licks_scored.csv.gz, qc/trial_attempts.csv,
iti_licks/iti_dock_audit.csv, <v36>/sessions/<an>/<date>/{licks_all.csv, windows.npz}, the session gui_config (spout
geometry).
Usage (bash env per the repo):
  python scripts/learned_compensation.py run [--jobs 3]   per-session timing extraction + trial / ITI tables
  python scripts/learned_compensation.py summarise        per-animal tables, boot_delta, mixed models, figdata
  python scripts/learned_compensation.py plot             figures from figdata only (also: any command --plot-only)
Outputs: <v36>/imaging_encoding/learned_compensation/
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date as _date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

V36 = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36")
E = V36 / "imaging_encoding"
OUT = E / "learned_compensation"
FIG = OUT / "figures"
FD = OUT / "figdata"
TIM = OUT / "timing"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
PHASES = ["retract", "docked", "outbound"]
WIN = (1, 14)                 # post days, inclusive (T1 primary window)
MIN_SESS_T1 = 24              # scored far-contra trials per session (>= 8 per third)
MIN_CELL = 10                 # licks per ITI cell
BOUT_GAP = 0.5                # s: bout-start ITI licks
TBINS = np.arange(0.0, 4.01, 0.5)
SEED = 0


def _b(s: pd.Series) -> pd.Series:
    """Boolean column robust to CSV round trips (True / "True" / 1)."""
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def _date_of(label: str) -> _date:
    return _date(2026, int(label[-4:-2]), int(label[-2:]))


# ================================================================================================ per-session timing
def session_timing(item) -> dict:
    """(sid, label) -> per-trial cue / stop on the DAQ clock (windows.npz), per-lick DAQ onset + gap (licks_all), spout
    angles of the session's commanded positions. Module-level for spawn."""
    sid, label = item
    f = TIM / f"{sid}.pkl"
    if f.exists() and "--redo" not in sys.argv:
        return {"sid": sid, "cached": True}
    from scripts.apparatus_mouth import _rig
    from wfield_local import orofacial_cohort as oc
    from wfield_local import spout_world as SW
    from wfield_local.paths import PathResolver
    an, d8 = sid.split("_")
    sd = V36 / "sessions" / an / d8
    w = np.load(sd / "windows.npz", allow_pickle=False)
    ix = pd.DataFrame({"trial_id": w["ix_trial_id"], "t_ms": w["ix_t_ms"].astype(float),
                       "daq_s": w["ix_daq_s"].astype(float), "stop_ms": w["ix_stop_ms"].astype(float),
                       "position": w["ix_position"].astype(str)})
    ix = ix[np.isfinite(ix.daq_s)]
    ix["cue"] = ix.daq_s - ix.t_ms / 1000.0
    Tr = ix.groupby("trial_id").agg(cue_daq=("cue", "median"), stop_ms=("stop_ms", "first"),
                                    position=("position", "first")).reset_index()
    Tr["stop_daq"] = Tr.cue_daq + Tr.stop_ms / 1000.0
    A = pd.read_csv(sd / "licks_all.csv", usecols=["context", "daq_on_s"])
    A["lick_uid"] = np.arange(len(A))
    o = A.dropna(subset=["daq_on_s"]).sort_values("daq_on_s")
    gap = pd.Series(np.r_[np.inf, np.diff(o.daq_on_s.to_numpy(float))], index=o.index)
    A["gap"] = gap.reindex(A.index)
    geo = _rig(PathResolver(), an, d8)["geometry"]
    phi = {}
    for p in SW.POSITIONS:
        T = np.asarray(SW.commanded_world(geo, p), float)
        phi[oc.pos_key(p, "L")] = float(np.degrees(np.arcsin(np.clip(T[0] / np.linalg.norm(T), -1, 1))))
    out = {"sid": sid, "label": label, "trials": Tr, "licks": A[["lick_uid", "context", "daq_on_s", "gap"]], "phi": phi}
    TIM.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    tmp.replace(f)
    return {"sid": sid, "n_trials": len(Tr)}


def _load_timing(sid: str) -> dict:
    with open(TIM / f"{sid}.pkl", "rb") as fh:
        return pickle.load(fh)


# ================================================================================================ run
LS_COLS = ["sid", "label", "animal", "epoch", "day", "lick_uid", "trial_id", "lick_idx", "context", "pos", "contact",
           "onset_ok", "theta_L", "theta_E", "e_L", "e_E"]


def cmd_run(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    for d in (OUT, FIG, FD, TIM):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    LS = pd.read_csv(E / "within_lick_correction" / "licks_scored.csv.gz", usecols=LS_COLS, low_memory=False)
    LS = LS[LS.animal.isin(AN) & LS.epoch.isin(EPS)]
    LS = LS[(LS.epoch != "pre") | (LS.label.str[-4:-2] == "08")].copy()      # August-only pre (already so; asserted)
    LS["onset_ok"] = _b(LS.onset_ok)
    sids = sorted(LS.sid.unique())
    print(f"{len(sids)} sessions, {len(LS)} licks (load {time.time() - t0:.0f} s)", flush=True)
    items = [(s, f"{s[:4]}_{s[-4:]}") for s in sids]
    _res, fail = ak.fan_sessions(items, session_timing, jobs=jobs, key=lambda x: x[0], label="timing",
                                 log=lambda m: None)
    if fail:
        for it, err in fail:
            print(f"  !! {it}: {err}", flush=True)
        raise SystemExit("timing extraction failed")
    print(f"timing extracted {(time.time() - t0) / 60:.1f} min", flush=True)
    TA = pd.read_csv(E / "qc" / "trial_attempts.csv", low_memory=False,
                     usecols=["label", "trial_id", "pos", "cls", "canonical"])
    TA = TA[TA.label.isin(LS.label.unique())].copy()
    TA["canonical"] = _b(TA.canonical)
    TA["hit"] = TA.cls.eq("hit").astype(float)
    # ---------------------------------------------------------------- T1 trial table (all trials, block structure)
    TA = TA.sort_values(["label", "trial_id"]).reset_index(drop=True)
    TA["first_in_block"] = (TA.groupby("label").pos.shift() != TA.pos).astype(int)
    TA["block_id"] = TA.groupby("label").first_in_block.cumsum()
    TA["block_k"] = TA.groupby(["label", "block_id"]).cumcount()
    cue = LS[(LS.context == "post_cue")].sort_values(["label", "trial_id", "lick_idx"])
    f0 = cue[(cue.lick_idx == 0) & cue.onset_ok & np.isfinite(cue.e_L)]
    f0 = f0[["label", "trial_id", "pos", "e_L", "e_E"]].rename(columns={"pos": "pos_l", "e_L": "eL1", "e_E": "eE1"})
    sc = cue[cue.onset_ok & np.isfinite(cue.e_L)]
    m3 = sc.groupby(["label", "trial_id"]).head(3).groupby(["label", "trial_id"]).e_L.mean().rename("eL3").reset_index()
    T = TA.merge(f0, on=["label", "trial_id"], how="left").merge(m3, on=["label", "trial_id"], how="left")
    bad = T.pos_l.notna() & (T.pos_l != T.pos)
    if bad.any():
        raise SystemExit(f"join check failed: lick pos != trial pos on {int(bad.sum())} trials")
    meta = LS.groupby("label").agg(animal=("animal", "first"), epoch=("epoch", "first"), day=("day", "first"),
                                   sid=("sid", "first")).reset_index()
    T = T.drop(columns="pos_l").merge(meta, on="label", how="left")
    T.to_csv(OUT / "trials_t1.csv.gz", index=False, float_format="%.5g")
    print(f"T1 trials: {len(T)} ({int(T.canonical.sum())} engaged; first-lick e_L on {int(T.eL1.notna().sum())})",
          flush=True)
    # ---------------------------------------------------------------- ITI table
    D = pd.read_csv(E / "iti_licks" / "iti_dock_audit.csv", usecols=["label", "onset", "phase", "since_stop", "src"])
    I = LS[LS.context == "iti"].copy()
    rows, diag = [], []
    for sid, g in I.groupby("sid", sort=True):
        tm = _load_timing(sid)
        lab = tm["label"]
        g = g.merge(tm["licks"][["lick_uid", "daq_on_s", "gap"]], on="lick_uid", how="left")
        Tr = tm["trials"].sort_values("stop_daq").reset_index(drop=True)
        x = g.daq_on_s.to_numpy(float)
        st = Tr.stop_daq.to_numpy(float)
        j = np.searchsorted(st, x, side="right") - 1
        ok = (j >= 0) & np.isfinite(x)
        jj = np.maximum(j, 0)
        g["prev_tid"] = np.where(ok, Tr.trial_id.to_numpy()[jj], -1)
        g["since_stop_own"] = np.where(ok, x - st[jj], np.nan)
        nxt = np.minimum(jj + 1, len(Tr) - 1)
        g["next_tid"] = np.where(ok & (jj + 1 < len(Tr)), Tr.trial_id.to_numpy()[nxt], -1)
        # dock audit join on onset (|dt| < 2 ms)
        d = D[D.label == lab].sort_values("onset")
        on = d.onset.to_numpy(float)
        if len(on):
            k = np.clip(np.searchsorted(on, x), 1, max(len(on) - 1, 1))
            km = np.where(np.abs(on[np.minimum(k, len(on) - 1)] - x) <= np.abs(on[k - 1] - x), np.minimum(k, len(on) - 1),
                          k - 1)
            dt = np.abs(on[km] - x)
            hit = dt < 0.002
            g["phase_raw"] = np.where(hit, d.phase.to_numpy()[km], "unmatched")
            g["since_stop"] = np.where(hit, d.since_stop.to_numpy(float)[km], np.nan)
            g["dock_src"] = np.where(hit, d.src.to_numpy()[km], "")
        else:
            g["phase_raw"], g["since_stop"], g["dock_src"] = "unmatched", np.nan, ""
        g["phi_map"] = [tm["phi"]] * len(g)
        diag.append({"label": lab, "n_iti": len(g), "matched": float((g.phase_raw != "unmatched").mean()),
                     "since_stop_absdiff_med": float(np.nanmedian(np.abs(g.since_stop - g.since_stop_own)))
                     if g.since_stop.notna().any() else np.nan})
        rows.append(g)
    I = pd.concat(rows, ignore_index=True)
    ph = {"pre_retract": "retract", "retract": "retract", "docked": "docked", "outbound": "outbound"}
    I["phase"] = I.phase_raw.map(ph).fillna("other")
    TT = TA.set_index(["label", "trial_id"])
    pk = pd.MultiIndex.from_arrays([I.label, I.prev_tid])
    I["prev_pos"] = TT.pos.reindex(pk).to_numpy()
    I["prev_canonical"] = TT.canonical.reindex(pk).fillna(False).to_numpy(bool)
    I["prev_hit"] = TT.hit.reindex(pk).to_numpy(float)
    I["next_pos"] = TT.pos.reindex(pd.MultiIndex.from_arrays([I.label, I.next_tid])).to_numpy()
    I["phi_prev"] = [m.get(p, np.nan) if isinstance(p, str) else np.nan for m, p in zip(I.phi_map, I.prev_pos)]
    I = I.drop(columns="phi_map")
    I.to_csv(OUT / "iti_licks.csv.gz", index=False, float_format="%.6g")
    DG = pd.DataFrame(diag)
    DG.to_csv(OUT / "iti_join_diag.csv", index=False)
    phi = pd.DataFrame([{"sid": s, **_load_timing(s)["phi"]} for s in sids])
    phi.to_csv(OUT / "spout_angles.csv", index=False)
    print(f"ITI licks {len(I)}: matched to the dock audit {(I.phase_raw != 'unmatched').mean():.3f}; phases "
          f"{I.phase.value_counts().to_dict()}; prev engaged {I.prev_canonical.mean():.3f}; my since_stop vs audit "
          f"median |diff| {DG.since_stop_absdiff_med.median() * 1e3:.1f} ms", flush=True)
    print(f"run done {(time.time() - t0) / 60:.1f} min", flush=True)


# ================================================================================================ stats helpers
def _change_row(post: dict, pre: dict, rng) -> dict:
    from wfield_local import analysis_kit as ak
    iv, dr = ak.boot_delta(post, pre, rng, draws=True)
    common = sorted(set(post) & set(pre))
    dif = {a: float(np.mean(post[a]) - np.mean(pre[a])) for a in common}
    if iv is None:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n_animals": len(common), "n_same": 0,
                "change": False, **{f"d_{a}": dif.get(a, np.nan) for a in AN}}
    same = sum(1 for v in dif.values() if np.sign(v) == np.sign(iv.point) and v != 0)
    return {"delta": iv.point, "lo": iv.lo, "hi": iv.hi, "p": ak.boot_p(dr), "n_animals": len(common), "n_same": same,
            "change": bool((iv.lo > 0 or iv.hi < 0) and same >= 3), **{f"d_{a}": dif.get(a, np.nan) for a in AN}}


def _by_animal(df: pd.DataFrame, col: str) -> dict:
    g = df.dropna(subset=[col])
    return {a: list(h[col].astype(float)) for a, h in g.groupby("animal", sort=True)}


def _level(df: pd.DataFrame, col: str, rng) -> tuple:
    from wfield_local import analysis_kit as ak
    iv = ak.boot_ci(_by_animal(df, col), rng)
    return (iv.point, iv.lo, iv.hi) if iv else (np.nan, np.nan, np.nan)


def _mixed(formula: str, data: pd.DataFrame, vc: str, terms: list[str]) -> list[dict]:
    import statsmodels.formula.api as smf
    out = []
    try:
        m = smf.mixedlm(formula, data, groups=data["animal"], vc_formula={vc: f"0 + C({vc})"}, re_formula="1")
        r = m.fit(reml=True, method=["lbfgs", "powell"])
        ci = r.conf_int()
        for t in terms:
            out.append({"formula": formula, "term": t, "coef": float(r.params[t]), "lo": float(ci.loc[t, 0]),
                        "hi": float(ci.loc[t, 1]), "p": float(r.pvalues[t]), "n": len(data),
                        "n_groups": int(data[vc].nunique()), "converged": bool(r.converged)})
    except Exception as ex:  # noqa: BLE001
        out.append({"formula": formula, "term": "ERROR", "coef": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan,
                    "n": len(data), "n_groups": 0, "converged": False, "error": str(ex)[:200]})
    return out


def _period(r) -> str:
    if r.epoch == "pre":
        return "pre"
    return "window" if WIN[0] <= r.day <= WIN[1] else "late"


# ================================================================================================ T1
def _thirds(v: np.ndarray) -> tuple:
    parts = np.array_split(np.asarray(v, float), 3)
    return tuple(float(np.mean(p)) for p in parts)


def t1_sessions(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lab, an, ep, day), g in T[T.canonical].groupby(["label", "animal", "epoch", "day"], sort=True):
        r = {"label": lab, "animal": an, "epoch": ep, "day": int(day)}
        fc = g[g.pos == "far_contra"].sort_values("trial_id")
        for nm, col, sub in (("eL", "eL1", fc), ("eE", "eE1", fc), ("eL3", "eL3", fc), ("hit", "hit", fc),
                             ("eLnf", "eL1", fc[fc.first_in_block == 0])):
            v = sub[col].dropna().to_numpy(float)
            r[f"n_{nm}"] = len(v)
            if len(v) >= MIN_SESS_T1:
                e, m_, l_ = _thirds(v)
                x = np.linspace(0, 1, len(v))
                r.update({f"early_{nm}": e, f"mid_{nm}": m_, f"late_{nm}": l_, f"W_{nm}": l_ - e, f"M_{nm}": float(v.mean()),
                          f"b_{nm}": float(np.polyfit(x, v, 1)[0])})
        # other five spouts: first-lick e_L thirds per spout, averaged (each spout its own order)
        ws = []
        for p in POS[:5]:
            v = g[g.pos == p].sort_values("trial_id").eL1.dropna().to_numpy(float)
            if len(v) >= MIN_SESS_T1:
                e, _m, l_ = _thirds(v)
                ws.append(l_ - e)
        if len(ws) >= 3:
            r["W_o5"] = float(np.mean(ws))
        if "W_eL" in r and "W_o5" in r:
            r["W_fc_minus_o5"] = r["W_eL"] - r["W_o5"]
        rows.append(r)
    S = pd.DataFrame(rows)
    S["period"] = [_period(r) for r in S.itertuples()]
    return S


def t1_pairs(S: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    rows = []
    for an, g in S.sort_values("label").groupby("animal", sort=True):
        g = g.sort_values("day").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            gap = (_date_of(b.label) - _date_of(a.label)).days
            if gap > max_gap or a.period != b.period or a.period == "late":
                continue
            r = {"animal": an, "label_d": a.label, "label_d1": b.label, "day_d": a.day, "gap_days": gap,
                 "period": a.period, "epoch_d": a.epoch, "epoch_d1": b.epoch}
            for nm in ("eL", "hit", "eE", "eL3"):
                if np.isfinite(a.get(f"late_{nm}", np.nan)) and np.isfinite(b.get(f"early_{nm}", np.nan)):
                    r[f"O_{nm}"] = b[f"early_{nm}"] - a[f"late_{nm}"]
                    r[f"S_{nm}"] = b[f"M_{nm}"] - a[f"M_{nm}"]
                    r[f"Wd_{nm}"] = a[f"W_{nm}"]
            rows.append(r)
    return pd.DataFrame(rows)


def t1_mixed(T: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    per = S.set_index("label").period
    X = T[T.canonical & (T.pos == "far_contra")].copy()
    X["period"] = X.label.map(per)
    X = X[X.period.isin(["pre", "window"])].sort_values(["label", "trial_id"])
    X["post"] = (X.period == "window").astype(int)
    out = []
    for col, nm in (("eL1", "e_L first lick"), ("hit", "hit (linear probability)")):
        d = X.dropna(subset=[col]).copy()
        n = d.groupby("label")[col].transform("size")
        d = d[n >= MIN_SESS_T1].copy()
        d["x"] = d.groupby("label").cumcount() / (d.groupby("label")[col].transform("size") - 1)
        d["y"] = d[col].astype(float)
        d["fib"] = d.first_in_block
        d["session"] = d.label
        for r in _mixed("y ~ x * post + fib", d, "session", ["x", "x:post", "post", "fib"]):
            out.append({"test": "T1 within-session slope", "readout": nm, **r})
    # overnight (secondary): late third of d vs early third of d+1, per pair
    P = t1_pairs(S, 1)
    rows = []
    for k, pr in enumerate(P.itertuples()):
        for lab, part, sec in ((pr.label_d, 2, 0), (pr.label_d1, 0, 1)):
            v = X[X.label == lab].eL1.dropna().to_numpy(float)
            if len(v) < MIN_SESS_T1:
                continue
            seg = np.array_split(v, 3)[part]
            rows += [{"animal": pr.animal, "pair": k, "second": sec, "post": int(pr.period == "window"), "y": y}
                     for y in seg]
    if rows:
        Q = pd.DataFrame(rows)
        for r in _mixed("y ~ second * post", Q, "pair", ["second", "second:post"]):
            out.append({"test": "T1 overnight (trial-level pairs)", "readout": "e_L first lick", **r})
    return pd.DataFrame(out)


# ================================================================================================ T2 / T3 references
def _refs(I: pd.DataFrame, val: str, keys: list[str]) -> pd.Series:
    """Per lick: the animal's August-pre mean of `val` at the same keys, leaving the lick's own session out (pre)."""
    pre = I[(I.epoch == "pre") & np.isfinite(I[val])]
    tot = pre.groupby(["animal", *keys])[val].agg(["sum", "count"])
    own = pre.groupby(["animal", *keys, "label"])[val].agg(["sum", "count"])
    idx_t = pd.MultiIndex.from_frame(I[["animal", *keys]])
    s_t = tot["sum"].reindex(idx_t).to_numpy(float)
    n_t = tot["count"].reindex(idx_t).to_numpy(float)
    idx_o = pd.MultiIndex.from_frame(I[["animal", *keys, "label"]])
    s_o = np.nan_to_num(own["sum"].reindex(idx_o).to_numpy(float))
    n_o = np.nan_to_num(own["count"].reindex(idx_o).to_numpy(float))
    is_pre = (I.epoch == "pre").to_numpy()
    s = np.where(is_pre, s_t - s_o, s_t)
    n = np.where(is_pre, n_t - n_o, n_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(n >= 5, s / n, np.nan)
    return pd.Series(ref, index=I.index)


def _iti_base(I: pd.DataFrame) -> pd.DataFrame:
    I = I[I.prev_canonical & I.prev_pos.isin(POS)].copy()
    I["grp"] = np.where(I.prev_pos == "far_contra", "FC", "O5")
    I["onset_ok"] = _b(I.onset_ok)
    return I


def t2_cells(I: pd.DataFrame, val: str, key: str, sel: np.ndarray | None = None) -> pd.DataFrame:
    """Session x prev group x key cells of d = val - ref(animal, prev_pos, key) (LOSO for pre)."""
    J = I if sel is None else I[sel]
    J = J[np.isfinite(J[val]) & J[key].notna()].copy()
    J["d"] = J[val] - _refs(J, val, ["prev_pos", key])
    J = J[np.isfinite(J.d)]
    C = J.groupby(["label", "animal", "epoch", "day", "grp", key]).d.agg(["mean", "count"]).reset_index()
    return C[C["count"] >= MIN_CELL]


def t2_contrast(C: pd.DataFrame, key: str = "phase", a: str = "retract", b: str = "docked") -> pd.DataFrame:
    W = C.pivot_table(index=["label", "animal", "epoch", "day"], columns=["grp", key], values="mean")
    W.columns = [f"{g}_{k}" for g, k in W.columns]
    W = W.reset_index()
    need = [f"FC_{a}", f"FC_{b}", f"O5_{a}", f"O5_{b}"]
    for c in need:
        if c not in W:
            W[c] = np.nan
    W["C2"] = (W[f"FC_{a}"] - W[f"FC_{b}"]) - (W[f"O5_{a}"] - W[f"O5_{b}"])
    W["FC_decay"] = W[f"FC_{a}"] - W[f"FC_{b}"]
    return W


# ================================================================================================ summarise
def cmd_summarise() -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    log = []

    def P(s=""):
        print(s, flush=True)
        log.append(str(s))

    T = pd.read_csv(OUT / "trials_t1.csv.gz", low_memory=False)
    T["canonical"] = _b(T.canonical)
    I = pd.read_csv(OUT / "iti_licks.csv.gz", low_memory=False)
    I["prev_canonical"] = _b(I.prev_canonical)
    assert not set(T.animal.dropna()) - set(AN) and not set(I.animal) - set(AN), "cohort2 leaked in"
    assert (T[T.epoch == "pre"].label.str[-4:-2] == "08").all(), "June pre leaked in"
    fd = {}
    # ------------------------------------------------------------------------------------------------ T1
    S = t1_sessions(T)
    S.to_csv(OUT / "t1_sessions.csv", index=False)
    P1, P2 = t1_pairs(S, 1), t1_pairs(S, 2)
    P1.to_csv(OUT / "t1_pairs_gap1.csv", index=False)
    P2.to_csv(OUT / "t1_pairs_gap2.csv", index=False)
    P("== T1 sessions per animal x period (n with a far-contra e_L cell / hit cell)")
    P(S.groupby(["animal", "period"]).agg(n=("label", "size"), n_eL=("W_eL", "count"), n_hit=("W_hit", "count"))
      .unstack().to_string())
    P("== T1 PER-ANIMAL (rule 8): session means by period")
    cols = ["M_eL", "W_eL", "b_eL", "M_hit", "W_hit", "W_o5", "W_fc_minus_o5", "W_eLnf"]
    PA = S.groupby(["animal", "period"])[cols].mean()
    P(PA.round(3).to_string())
    PB = pd.concat([P1.groupby(["animal", "period"])[["O_eL", "S_eL", "O_hit", "S_hit"]].mean(),
                    P1.groupby(["animal", "period"]).size().rename("n_pairs")], axis=1)
    P("== T1 PER-ANIMAL overnight pairs (gap 1 d)")
    P(PB.round(3).to_string())
    PA.reset_index().to_csv(OUT / "t1_per_animal_sessions.csv", index=False)
    PB.reset_index().to_csv(OUT / "t1_per_animal_pairs.csv", index=False)
    rng = np.random.default_rng(SEED)
    B1 = []
    pre_s, win_s = S[S.period == "pre"], S[S.period == "window"]
    for m, primary in (("W_eL", True), ("W_hit", True), ("b_eL", False), ("W_o5", False), ("W_fc_minus_o5", False),
                       ("W_eE", False), ("W_eL3", False), ("W_eLnf", False), ("M_eL", False), ("M_hit", False)):
        B1.append({"test": "T1", "metric": m, "set": "window d1-14 vs pre", "primary": primary,
                   **_change_row(_by_animal(win_s, m), _by_animal(pre_s, m), rng)})
    for nm, PP in (("gap1", P1), ("gap<=2", P2)):
        pre_p, win_p = PP[PP.period == "pre"], PP[PP.period == "window"]
        for m in ("O_eL", "O_hit", "S_eL", "S_hit", "O_eE", "O_eL3"):
            if m not in PP:
                continue
            B1.append({"test": "T1", "metric": m, "set": f"window pairs vs pre pairs ({nm})",
                       "primary": nm == "gap1" and m in ("O_eL", "O_hit"),
                       **_change_row(_by_animal(win_p, m), _by_animal(pre_p, m), rng)})
    for ep in EPS[1:]:
        for m in ("W_eL", "W_hit", "W_o5"):
            B1.append({"test": "T1", "metric": m, "set": f"{ep} vs pre", "primary": False,
                       **_change_row(_by_animal(S[S.epoch == ep], m), _by_animal(pre_s, m), rng)})
    B1 = pd.DataFrame(B1)
    pm = B1.primary.to_numpy(bool)
    B1["p_holm"] = np.nan
    B1.loc[pm, "p_holm"] = ak.holm(B1.loc[pm, "p"].to_numpy())
    # pooled levels (boot_ci) for the figure
    lev = []
    for m in ("W_eL", "W_hit", "W_o5", "W_fc_minus_o5", "M_eL"):
        for per, g in S.groupby("period"):
            lev.append({"metric": m, "period": per, **dict(zip(("level", "lo", "hi"), _level(g, m, rng)))})
    for m in ("O_eL", "S_eL", "O_hit", "S_hit"):
        for per, g in P1.groupby("period"):
            lev.append({"metric": m, "period": per, **dict(zip(("level", "lo", "hi"), _level(g, m, rng)))})
    L1 = pd.DataFrame(lev)
    P("== T1 boot_delta (95 %, paired animals -> sessions; change = CI excl. 0 AND >= 3/4 same sign)")
    P(B1[["metric", "set", "primary", "delta", "lo", "hi", "n_same", "n_animals", "change", "p_holm",
          *[f"d_{a}" for a in AN]]].round(3).to_string(index=False))
    M1 = t1_mixed(T, S)
    P("== T1 mixed models (groups animal, session / pair variance component)")
    P(M1.drop(columns=[c for c in ("formula", "error") if c in M1]).round(4).to_string(index=False))
    B1.to_csv(OUT / "t1_boot.csv", index=False)
    M1.to_csv(OUT / "t1_mixed.csv", index=False)
    L1.to_csv(OUT / "t1_levels.csv", index=False)
    fd["t1"] = {"S": S, "P1": P1, "B": B1, "L": L1, "M": M1}
    # ------------------------------------------------------------------------------------------------ T2
    J = _iti_base(I)
    P(f"\n== ITI licks after engaged trials: {len(J)} of {len(I)}; phases {J.phase.value_counts().to_dict()}")
    cen = J.groupby(["animal", "epoch", "grp", "phase"]).size().unstack(["grp", "phase"], fill_value=0)
    P(cen.to_string())
    Jl = J[J.onset_ok]
    C = t2_cells(Jl, "theta_L", "phase", Jl.phase.isin(PHASES).to_numpy())
    W2 = t2_contrast(C)
    cuedfc = S.set_index("label").M_eL
    W2["cued_fc_eL"] = W2.label.map(cuedfc)
    W2.to_csv(OUT / "t2_sessions.csv", index=False)
    C.to_csv(OUT / "t2_cells.csv", index=False)
    P("== T2 PER-ANIMAL (rule 8): session means of d (deg vs pre ITI at the same previous spout + phase) and C2")
    show = ["FC_retract", "FC_docked", "FC_outbound", "O5_retract", "O5_docked", "O5_outbound", "C2", "cued_fc_eL"]
    PA2 = W2.groupby(["animal", "epoch"])[[c for c in show if c in W2]].mean()
    PA2["n_C2"] = W2.groupby(["animal", "epoch"]).C2.count()
    P(PA2.round(2).to_string())
    PA2.reset_index().to_csv(OUT / "t2_per_animal.csv", index=False)
    rng = np.random.default_rng(SEED + 1)
    B2 = []
    pre2 = W2[W2.epoch == "pre"]
    for ep in EPS[1:]:
        cur = W2[W2.epoch == ep]
        for m in ("C2", "FC_decay", "FC_retract", "FC_docked", "FC_outbound", "O5_retract", "O5_docked", "cued_fc_eL"):
            if m in W2:
                B2.append({"test": "T2", "variant": "primary (theta_L, all ITI)", "metric": m, "epoch": ep,
                           "primary": m == "C2", **_change_row(_by_animal(cur, m), _by_animal(pre2, m), rng)})
    # secondary variants of C2
    variants = {
        "theta_E": (J, "theta_E", J.phase.isin(PHASES).to_numpy()),
        "bout-start (gap >= 0.5 s)": (Jl, "theta_L", (Jl.phase.isin(PHASES) & (Jl.gap >= BOUT_GAP)).to_numpy()),
        "previous trial a hit": (Jl, "theta_L", (Jl.phase.isin(PHASES) & (Jl.prev_hit == 1)).to_numpy()),
    }
    sec = {}
    for vn, (src, val, sel) in variants.items():
        Cv = t2_cells(src, val, "phase", sel)
        Wv = t2_contrast(Cv)
        sec[vn] = Wv
        pv = Wv[Wv.epoch == "pre"]
        for ep in EPS[1:]:
            for m in ("C2", "FC_retract", "FC_docked"):
                B2.append({"test": "T2", "variant": vn, "metric": m, "epoch": ep, "primary": False,
                           **_change_row(_by_animal(Wv[Wv.epoch == ep], m), _by_animal(pv, m), rng)})
    # outbound split by next spout same / different
    Jo = Jl[Jl.phase == "outbound"].copy()
    Jo["nxt"] = np.where(Jo.next_pos == Jo.prev_pos, "same", "diff")
    Co = t2_cells(Jo, "theta_L", "nxt")
    for nx in ("same", "diff"):
        Wn = Co[Co.nxt == nx].pivot_table(index=["label", "animal", "epoch", "day"], columns="grp", values="mean")\
            .reset_index()
        for ep in EPS[1:]:
            for g in ("FC", "O5"):
                if g in Wn:
                    B2.append({"test": "T2", "variant": f"outbound, next spout {nx}", "metric": f"{g}_outbound",
                               "epoch": ep, "primary": False,
                               **_change_row(_by_animal(Wn[Wn.epoch == ep], g), _by_animal(Wn[Wn.epoch == "pre"], g),
                                             rng)})
    B2 = pd.DataFrame(B2)
    pm = B2.primary.to_numpy(bool)
    B2["p_holm"] = np.nan
    B2.loc[pm, "p_holm"] = ak.holm(B2.loc[pm, "p"].to_numpy())
    P("== T2 boot_delta vs pre")
    P(B2[["variant", "metric", "epoch", "delta", "lo", "hi", "n_same", "change", "p_holm", *[f"d_{a}" for a in AN]]]
      .round(2).to_string(index=False))
    # since-stop bins (descriptive)
    Jb = Jl[np.isfinite(Jl.since_stop) & (Jl.since_stop >= 0) & (Jl.since_stop < TBINS[-1])].copy()
    Jb["tbin"] = np.digitize(Jb.since_stop, TBINS) - 1
    Cb = t2_cells(Jb, "theta_L", "tbin")
    Cb.to_csv(OUT / "t2_since_stop_cells.csv", index=False)
    lvb = []
    rng_l = np.random.default_rng(SEED + 2)
    for (ep, g, tb), h in Cb.groupby(["epoch", "grp", "tbin"]):
        h = h.rename(columns={"mean": "v"})
        lvb.append({"epoch": ep, "grp": g, "tbin": tb, "t": TBINS[tb] + 0.25, "n_sessions": len(h),
                    **dict(zip(("level", "lo", "hi"), _level(h, "v", rng_l)))})
    LB = pd.DataFrame(lvb)
    # pooled phase levels for the figure
    lv2 = []
    for ep in EPS:
        cur = W2[W2.epoch == ep]
        for c in ["FC_retract", "FC_docked", "FC_outbound", "O5_retract", "O5_docked", "O5_outbound", "C2", "cued_fc_eL",
                  "FC_decay"]:
            if c in cur:
                lv2.append({"epoch": ep, "metric": c, **dict(zip(("level", "lo", "hi"), _level(cur, c, rng_l)))})
    L2 = pd.DataFrame(lv2)
    # mixed model (post sessions, retract + docked, lick level)
    Jm = Jl[Jl.phase.isin(["retract", "docked"]) & (Jl.epoch != "pre") & np.isfinite(Jl.since_stop)].copy()
    Jm["d"] = Jm.theta_L - _refs(Jl, "theta_L", ["prev_pos", "phase"]).reindex(Jm.index)
    Jm = Jm[np.isfinite(Jm.d)]
    Jm["FC"] = (Jm.grp == "FC").astype(int)
    Jm["session"] = Jm.label
    M2 = pd.DataFrame(_mixed("d ~ FC * since_stop + C(epoch)", Jm, "session", ["FC", "since_stop", "FC:since_stop"]))
    P("== T2 mixed model (post, retract + docked)")
    P(M2.drop(columns=[c for c in ("formula", "error") if c in M2]).round(4).to_string(index=False))
    B2.to_csv(OUT / "t2_boot.csv", index=False)
    M2.to_csv(OUT / "t2_mixed.csv", index=False)
    L2.to_csv(OUT / "t2_levels.csv", index=False)
    LB.to_csv(OUT / "t2_since_stop_levels.csv", index=False)
    fd["t2"] = {"W": W2, "B": B2, "L": L2, "LB": LB, "Cb": Cb, "M": M2}
    # ------------------------------------------------------------------------------------------------ T3
    cued = T[T.canonical & T.eE1.notna()].groupby(["label", "pos"]).eE1.agg(["mean", "count"]).reset_index()
    cued = cued[cued["count"] >= MIN_CELL].set_index(["label", "pos"])["mean"]
    rng = np.random.default_rng(SEED + 3)
    B3, S3all = [], {}
    for vn, sel, val in (("docked, theta_E (primary)", J.phase == "docked", "theta_E"),
                         ("docked, theta_L", (J.phase == "docked") & J.onset_ok, "theta_L"),
                         ("all ITI phases, theta_E", J.phase.isin(PHASES), "theta_E")):
        K = J[sel.to_numpy(bool) & np.isfinite(J[val])].copy()
        K["dv"] = K[val] - _refs(K, val, ["prev_pos"])
        cell = K.groupby(["label", "animal", "epoch", "day", "prev_pos"]).agg(
            m=(val, "mean"), d=("dv", "mean"), n=(val, "size"), phi=("phi_prev", "mean")).reset_index()
        cell = cell[cell.n >= MIN_CELL]
        cell["cued_eE"] = cued.reindex(pd.MultiIndex.from_frame(cell[["label", "prev_pos"]])).to_numpy()
        rows = []
        for (lab, an, ep, day), g in cell.groupby(["label", "animal", "epoch", "day"], sort=True):
            r = {"label": lab, "animal": an, "epoch": ep, "day": day, "n_spouts": len(g)}
            gd = g.dropna(subset=["d"])
            if len(gd) >= 4:
                r["D_off"] = float(gd.d.mean())
                gc = gd.dropna(subset=["cued_eE"])
                if len(gc) >= 4:
                    r["cued_mean"] = float(gc.cued_eE.mean())
                    r["GAP"] = float(gc.d.mean() - gc.cued_eE.mean())
            if len(g) >= 4 and g.phi.nunique() >= 3:
                r["gain"] = float(np.polyfit(g.phi, g.m, 1)[0])
            fc = g[g.prev_pos == "far_contra"]
            if len(fc) and np.isfinite(fc.d.iloc[0]):
                r["D_fc"] = float(fc.d.iloc[0])
                r["cued_fc_eE"] = float(fc.cued_eE.iloc[0]) if np.isfinite(fc.cued_eE.iloc[0]) else np.nan
            for p in POS:
                gp = g[g.prev_pos == p]
                r[f"D_{p}"] = float(gp.d.iloc[0]) if len(gp) else np.nan
                r[f"m_{p}"] = float(gp.m.iloc[0]) if len(gp) else np.nan
                r[f"rel_{p}"] = float(gp.m.iloc[0] - gp.phi.iloc[0]) if len(gp) else np.nan
            rows.append(r)
        S3 = pd.DataFrame(rows)
        S3all[vn] = S3
        if vn.endswith("(primary)"):
            P(f"\n== T3 [{vn}] PER-ANIMAL (rule 8): session means")
            PA3 = S3.groupby(["animal", "epoch"])[["D_off", "D_fc", "gain", "cued_mean", "GAP"]].mean()
            PA3["n"] = S3.groupby(["animal", "epoch"]).D_off.count()
            P(PA3.round(2).to_string())
            PA3.reset_index().to_csv(OUT / "t3_per_animal.csv", index=False)
            S3.to_csv(OUT / "t3_sessions.csv", index=False)
        pre3 = S3[S3.epoch == "pre"]
        for ep in EPS[1:]:
            for m in ("D_off", "GAP", "D_fc", "gain", "cued_mean", *[f"D_{p}" for p in POS]):
                if m in S3:
                    B3.append({"test": "T3", "variant": vn, "metric": m, "epoch": ep,
                               "primary": vn.endswith("(primary)") and m in ("D_off", "GAP"),
                               **_change_row(_by_animal(S3[S3.epoch == ep], m), _by_animal(pre3, m), rng)})
        if vn.endswith("(primary)"):
            Km = K[(K.epoch != "pre") & np.isfinite(K.dv) & np.isfinite(K.phi_prev)].copy()
            Km["session"] = Km.label
            M3 = pd.DataFrame(_mixed("dv ~ phi_prev + C(epoch)", Km, "session", ["phi_prev"]))
    B3 = pd.DataFrame(B3)
    pm = B3.primary.to_numpy(bool)
    B3["p_holm"] = np.nan
    B3.loc[pm, "p_holm"] = ak.holm(B3.loc[pm, "p"].to_numpy())
    P("== T3 boot_delta vs pre")
    P(B3[["variant", "metric", "epoch", "delta", "lo", "hi", "n_same", "change", "p_holm", *[f"d_{a}" for a in AN]]]
      .round(2).to_string(index=False))
    P("== T3 mixed model (post, docked, theta_E - pre ref ~ previous spout angle)")
    P(M3.drop(columns=[c for c in ("formula", "error") if c in M3]).round(4).to_string(index=False))
    rng_l = np.random.default_rng(SEED + 4)
    S3p = S3all["docked, theta_E (primary)"]
    lv3 = []
    for ep in EPS:
        cur = S3p[S3p.epoch == ep]
        for m in ("D_off", "GAP", "D_fc", "gain", "cued_mean", *[f"D_{p}" for p in POS], *[f"m_{p}" for p in POS],
                  *[f"rel_{p}" for p in POS]):
            lv3.append({"epoch": ep, "metric": m, **dict(zip(("level", "lo", "hi"), _level(cur, m, rng_l)))})
    L3 = pd.DataFrame(lv3)
    B3.to_csv(OUT / "t3_boot.csv", index=False)
    M3.to_csv(OUT / "t3_mixed.csv", index=False)
    L3.to_csv(OUT / "t3_levels.csv", index=False)
    phi = pd.read_csv(OUT / "spout_angles.csv")
    fd["t3"] = {"S": S3p, "B": B3, "L": L3, "M": M3, "phi": phi}
    FD.mkdir(parents=True, exist_ok=True)
    with open(FD / "learned_compensation.pkl", "wb") as fh:
        pickle.dump(fd, fh)
    (OUT / "summarise_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("figdata written", flush=True)


# ================================================================================================ plots
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})


def _top_band(ax, frac: float = 0.16):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def _star(ax, x: float, row) -> None:
    """Star in the reserved top band (axes y 0.93) when the pre-registered change criterion holds."""
    import matplotlib.transforms as mt
    if row is None or not bool(row.get("change", False)):
        return
    tr = mt.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x, 0.955, "*", transform=tr, ha="center", va="center", fontsize=13, fontweight="bold")


def _jit(an: str) -> float:
    return {"PS92": -0.15, "PS93": -0.05, "PS94": 0.05, "PS95": 0.15}[an]


def _legend_animals_epochs(fig, epochs=EPS, y=0.005):
    from matplotlib.lines import Line2D

    from wfield_local import orofacial_cohort as oc
    h = [Line2D([], [], ls="", marker=oc.ANIMAL_MARKER[a], mfc="none", mec="0.3", label=a) for a in AN]
    h += [Line2D([], [], ls="", marker="s", color=oc.EPOCH_LINE[e], label=e) for e in epochs]
    h += [Line2D([], [], ls="-", lw=2.5, marker="o", color="0.2", label="pooled (95 % CI)")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=7, bbox_to_anchor=(0.5, y))


def fig1_daycourse(fd: dict) -> None:
    """T1 sawtooth: per animal, early -> late third of each session (solid), late(d) -> early(d+1) (dashed)."""
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    S, P1 = fd["t1"]["S"], fd["t1"]["P1"]
    S = S[S.period.isin(["pre", "window"])]
    # Priya 10-30: SESSION ORDER on x (each session a block: early | mid | late third, 1 unit wide, gaps between blocks)
    # so within-session and between-session segments are both readable; EVERY consecutive session pair is joined
    # (dashed = next day, dotted = > 1 day apart). _ = P1 kept for the figdata contract.
    _ = P1
    fig, axs = plt.subplots(2, 4, figsize=(17, 6.6))
    W3, GAP = (0.0, 0.5, 1.0), 0.8
    for row, (nm, yl) in enumerate((("eL", "far-contra first-lick launch error e_L (deg)"),
                                    ("hit", "far-contra hit rate (engaged)"))):
        for c, an in enumerate(AN):
            ax = axs[row, c]
            g = S[(S.animal == an)].dropna(subset=[f"early_{nm}"]).sort_values("day").reset_index(drop=True)
            prev, ticks, tl, x0_post = None, [], [], None
            for k, r in enumerate(g.itertuples()):
                col = oc.EPOCH_LINE[r.epoch]
                x0 = k * (1.0 + GAP)
                if x0_post is None and r.day > 0:
                    x0_post = x0 - GAP / 2
                xs = [x0 + w for w in W3]
                ys = [getattr(r, f"early_{nm}"), getattr(r, f"mid_{nm}"), getattr(r, f"late_{nm}")]
                ax.plot(xs, ys, "-", color=col, lw=1.3)
                ax.plot(xs, ys, ls="", marker=oc.ANIMAL_MARKER[an], mfc="none", mec=col, ms=4.5, mew=1.0)
                if prev is not None:
                    gap_d = r.day - prev[2]
                    ax.plot([prev[0], xs[0]], [prev[1], ys[0]], "--" if gap_d == 1 else ":", color="0.55",
                            lw=0.9 if gap_d == 1 else 0.8)
                prev = (xs[-1], ys[-1], r.day)
                ticks.append(x0 + 0.5)
                tl.append(f"{int(r.day)}")
            if nm == "eL":
                ax.axhline(0, color="0.7", lw=0.6)
            if x0_post is not None:
                ax.axvline(x0_post, color="0.6", lw=0.8)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tl, fontsize=6.5, rotation=90)
            if row == 0:
                ax.set_title(an)
            if c == 0:
                ax.set_ylabel(yl)
            if row == 1:
                ax.set_xlabel("sessions in order (label = day from stroke; grey line = stroke)", fontsize=8)
    fig.suptitle("T1 within-session change and change between sessions, far contra (August pre + post days 1-14): each "
                 "session = early | mid | late third (solid); dashed = to the next day, dotted = to a session > 1 day "
                 "later", fontsize=9)
    _legend_animals_epochs(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save(fig, "lc_fig1_t1_daycourse")


def _summary_panel(ax, df: pd.DataFrame, m: str, lev: pd.DataFrame, cats: list, catcol: str, B: pd.DataFrame,
                   bsel: dict, title: str, ylab: str = "deg"):
    from wfield_local import orofacial_cohort as oc
    for i, c in enumerate(cats):
        g = df[(df[catcol] == c)].dropna(subset=[m])
        for r in g.itertuples():
            ep = getattr(r, "epoch", None) or getattr(r, "epoch_d1", "pre")
            ax.plot(i + _jit(r.animal) + np.random.default_rng(hash(r.Index) % 2**32).uniform(-0.04, 0.04),
                    getattr(r, m), ls="", marker=oc.ANIMAL_MARKER[r.animal], mfc="none",
                    mec=oc.EPOCH_LINE.get(ep, "0.4"), ms=4, mew=0.9, alpha=0.9)
        lv = lev[(lev.metric == m) & (lev[catcol if catcol in lev else "period"] == c)]
        if len(lv) and np.isfinite(lv.level.iloc[0]):
            col = oc.EPOCH_LINE.get(c, "0.15") if c in oc.EPOCH_LINE else "0.15"
            ax.errorbar(i + 0.33, lv.level.iloc[0], yerr=[[lv.level.iloc[0] - lv.lo.iloc[0]], [lv.hi.iloc[0] - lv.level.iloc[0]]],
                         fmt="o", color=col, mfc=col, ms=6, lw=2.5, capsize=0, zorder=5)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(range(len(cats)), cats)
    ax.set_xlim(-0.5, len(cats) - 0.4)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel(ylab)
    _top_band(ax)
    for i, c in enumerate(cats):
        sel = B
        for k, v in bsel.items():
            sel = sel[sel[k] == v]
        rr = sel[sel["_cat"] == c] if "_cat" in sel else sel.iloc[0:0]
        if len(rr):
            _star(ax, i, rr.iloc[0].to_dict())


def fig2_t1_summary(fd: dict) -> None:
    import matplotlib.pyplot as plt
    S, P1, B, L = fd["t1"]["S"], fd["t1"]["P1"], fd["t1"]["B"].copy(), fd["t1"]["L"]
    S = S[S.period.isin(["pre", "window"])]
    P1 = P1.rename(columns={"epoch_d1": "epoch"})
    B["_cat"] = "window"
    fig, axs = plt.subplots(1, 6, figsize=(15, 3.9))
    specs = [(S, "W_eL", "window d1-14 vs pre", "W: late - early third, e_L"),
             (P1, "O_eL", "window pairs vs pre pairs (gap1)", "O: early(d+1) - late(d), e_L"),
             (P1, "S_eL", "window pairs vs pre pairs (gap1)", "S: day step, e_L"),
             (S, "W_o5", "window d1-14 vs pre", "W at the other 5 spouts, e_L"),
             (S, "W_hit", "window d1-14 vs pre", "W: hit rate"),
             (P1, "O_hit", "window pairs vs pre pairs (gap1)", "O: hit rate")]
    for ax, (df, m, st, ttl) in zip(axs, specs):
        _summary_panel(ax, df, m, L, ["pre", "window"], "period", B, {"metric": m, "set": st}, ttl,
                       "hit-rate units" if "hit" in m else "deg")
    fig.suptitle("T1 learning signature, far contra cued first lick: within-session W, overnight O, day step S "
                 "(points = sessions / overnight pairs; * = boot_delta change vs pre, CI excl. 0 and >= 3/4)",
                 fontsize=9)
    _legend_animals_epochs(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save(fig, "lc_fig2_t1_summary")


def fig3_t2_phases(fd: dict) -> None:
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    W, L = fd["t2"]["W"], fd["t2"]["L"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    x = np.arange(3)
    for ax, ep in zip(axs, EPS):
        g = W[W.epoch == ep]
        for grp, col, dx in (("FC", oc.EPOCH_LINE[ep], 0.08), ("O5", "0.6", -0.08)):
            pa = g.reindex(columns=["animal", *[f"{grp}_{p}" for p in PHASES]]).groupby("animal").mean()
            for an, r in pa.iterrows():
                ax.plot(x + dx + _jit(an) * 0.3, r.to_numpy(float), ls="", marker=oc.ANIMAL_MARKER[an], mfc="none",
                        mec=col, ms=4.5, mew=0.9)
            lv = L[(L.epoch == ep)].set_index("metric")
            y = np.array([lv.level.get(f"{grp}_{p}", np.nan) for p in PHASES])
            lo = np.array([lv.lo.get(f"{grp}_{p}", np.nan) for p in PHASES])
            hi = np.array([lv.hi.get(f"{grp}_{p}", np.nan) for p in PHASES])
            ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], fmt="-o", color=col, mfc=col, lw=2.2, ms=5, capsize=0,
                        label=f"after {'far contra' if grp == 'FC' else 'other 5 spouts'}")
        cu = L[(L.epoch == ep) & (L.metric == "cued_fc_eL")]
        if len(cu) and np.isfinite(cu.level.iloc[0]):
            ax.axhline(cu.level.iloc[0], color=oc.EPOCH_LINE[ep], ls="--", lw=1)
            ax.text(2.42, cu.level.iloc[0], "cued FC", fontsize=6, va="bottom", color=oc.EPOCH_LINE[ep])
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xticks(x, ["retract", "docked", "outbound"])
        ax.set_xlim(-0.4, 2.9)
        ax.set_title(ep)
        ax.legend(fontsize=6, loc="lower left", frameon=False)
    axs[0].relim()
    axs[0].autoscale(axis="y")
    axs[0].set_ylabel("ITI launch angle - pre ITI (same prev spout, phase), deg")
    fig.suptitle("T2 ITI launch angle by cycle phase after a far-contra (colour) vs other-spout (grey) trial; "
                 "open = animal means, filled = pooled sessions (95 % CI); dashed = cued far-contra e_L", fontsize=9)
    _legend_animals_epochs(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    _save(fig, "lc_fig3_t2_phases")


def fig4_t2_summary(fd: dict) -> None:
    import matplotlib.pyplot as plt
    W, B, L = fd["t2"]["W"], fd["t2"]["B"].copy(), fd["t2"]["L"].copy()
    B = B[B.variant == "primary (theta_L, all ITI)"].copy()
    B["_cat"] = B.epoch
    L = L.rename(columns={"epoch": "period"})
    W = W.assign(period=W.epoch)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (m, ttl) in zip(axs, (("C2", "C2 carry-over contrast"), ("FC_retract", "d after FC, retract"),
                                  ("FC_docked", "d after FC, docked"), ("cued_fc_eL", "cued far-contra e_L"))):
        _summary_panel(ax, W, m, L, EPS, "period", B, {"metric": m}, ttl)
    fig.suptitle("T2 carry-over: C2 = [d(FC,retract) - d(FC,docked)] - [d(O5,retract) - d(O5,docked)] (+ = "
                 "early-ITI re-aim after far-contra trials); * = change vs pre", fontsize=9)
    _legend_animals_epochs(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    _save(fig, "lc_fig4_t2_summary")


def fig5_t2_time(fd: dict) -> None:
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    LB, Cb = fd["t2"]["LB"], fd["t2"]["Cb"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, ep in zip(axs, EPS):
        for grp, col in (("FC", oc.EPOCH_LINE[ep]), ("O5", "0.6")):
            g = LB[(LB.epoch == ep) & (LB.grp == grp)].sort_values("t")
            ax.fill_between(g.t, g.lo, g.hi, color=col, alpha=0.18, lw=0)
            ax.plot(g.t, g.level, "-o", color=col, ms=3.5, lw=2, label=f"after {'far contra' if grp == 'FC' else 'other 5'}")
            pa = Cb[(Cb.epoch == ep) & (Cb.grp == grp)].groupby(["animal", "tbin"])["mean"].mean().reset_index()
            for r in pa.itertuples():
                ax.plot(TBINS[r.tbin] + 0.25 + _jit(r.animal) * 0.4, r[3], ls="", marker=oc.ANIMAL_MARKER[r.animal],
                        mfc="none", mec=col, ms=3.2, mew=0.7, alpha=0.8)
        for t, nm in ((0.45, "retract"), (1.6, "docked"), (2.9, "outbound")):
            ax.text(t, 1.0, nm, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color="0.4")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_title(ep, pad=10)
        ax.set_xlabel("time since the previous trial's stop (s)")
        ax.legend(fontsize=6, loc="lower right", frameon=False)
    axs[0].set_ylabel("ITI launch angle - pre ITI (same prev spout, bin), deg")
    fig.suptitle("T2 (descriptive) ITI launch angle vs time since the trial stop, 0.5 s bins", fontsize=9)
    _legend_animals_epochs(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    _save(fig, "lc_fig5_t2_time")


def fig6_t3(fd: dict) -> None:
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    S, B, L, phi = fd["t3"]["S"], fd["t3"]["B"].copy(), fd["t3"]["L"].copy(), fd["t3"]["phi"]
    B = B[B.variant == "docked, theta_E (primary)"].copy()
    B["_cat"] = B.epoch
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1])
    # A: ITI landing angle vs the previous spout's angle (pooled per spout), pre vs each epoch
    ax = fig.add_subplot(gs[0, 0:2])
    ph = phi[POS].mean()
    # close and far spouts share an angle, so they are drawn as two series (Priya 10-30: "two pink dots"):
    # FAR = filled markers / solid line, CLOSE = open markers / dashed line, small x offset
    for ep in EPS:
        lv = L[L.epoch == ep].set_index("metric")
        for ring, dx, ls_, mf in (("close", -1.2, "--", "none"), ("far", 1.2, "-", None)):
            ps = [p for p in POS if p.startswith(ring)]
            xx = ph[ps].to_numpy() + dx
            y = np.array([lv.level.get(f"m_{p}", np.nan) for p in ps])
            lo = np.array([lv.lo.get(f"m_{p}", np.nan) for p in ps])
            hi = np.array([lv.hi.get(f"m_{p}", np.nan) for p in ps])
            o = np.argsort(xx)
            ax.errorbar(xx[o], y[o], yerr=[(y - lo)[o], (hi - y)[o]], fmt="o", ls=ls_, color=oc.EPOCH_LINE[ep],
                        mfc=mf if mf else oc.EPOCH_LINE[ep], lw=1.8, ms=5, capsize=0,
                        label=f"{ep}, {ring} spouts")
    lim = [ph.min() - 5, ph.max() + 5]
    ax.plot(lim, lim, ":", color="0.6", lw=0.8)
    ax.text(lim[1], lim[1], " identity", fontsize=6, color="0.5", va="bottom")
    for side in ("ipsi", "mid", "contra"):
        xa = float(np.mean([ph[f"close_{side}"], ph[f"far_{side}"]]))
        ax.annotate(f"close / far {side}", (xa, 1.0), xycoords=("data", "axes fraction"), fontsize=6, ha="center",
                    va="top", color="0.4")
    ax.set_xlabel("previous trial's spout angle (deg, + = contralesional)")
    ax.set_ylabel("docked ITI landing angle theta_E (deg)")
    ax.set_title("A. docked ITI licks vs where the last spout was (pooled sessions, 95 % CI)", fontsize=8, pad=12)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
    # B: D per previous spout, canonical order
    ax = fig.add_subplot(gs[0, 2:4])
    x = np.arange(len(POS))
    for k, ep in enumerate(EPS[1:]):
        lv = L[L.epoch == ep].set_index("metric")
        y = np.array([lv.level.get(f"D_{p}", np.nan) for p in POS])
        lo = np.array([lv.lo.get(f"D_{p}", np.nan) for p in POS])
        hi = np.array([lv.hi.get(f"D_{p}", np.nan) for p in POS])
        dx = (k - 1) * 0.12
        ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], fmt="-o", color=oc.EPOCH_LINE[ep], lw=2, ms=5, capsize=0, label=ep)
        pa = S[S.epoch == ep].groupby("animal")[[f"D_{p}" for p in POS]].mean()
        for an, r in pa.iterrows():
            ax.plot(x + dx + _jit(an) * 0.25, r.to_numpy(float), ls="", marker=oc.ANIMAL_MARKER[an], mfc="none",
                    mec=oc.EPOCH_LINE[ep], ms=3.5, mew=0.7)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(x, [p.replace("_", "\n") for p in POS])
    ax.set_ylabel("docked ITI theta_E - pre at same prev spout (deg)")
    ax.set_title("B. by previous spout (open = animal means)", fontsize=8)
    _top_band(ax)
    for i, p in enumerate(POS):
        rr = B[(B.metric == f"D_{p}") & (B.epoch == "chronic")]
        if len(rr):
            _star(ax, i, rr.iloc[0].to_dict())
    ax.legend(fontsize=7, frameon=False, loc="lower left", title="* = chronic change", title_fontsize=6)
    L2 = L.rename(columns={"epoch": "period"})
    S2 = S.assign(period=S.epoch)
    for j, (m, ttl) in enumerate((("D_off", "D_off (mean over prev spouts)"), ("D_fc", "D_fc (after far contra)"),
                                  ("GAP", "GAP = D_off - cued e_E"), ("gain", "gain on prev-spout angle"))):
        ax = fig.add_subplot(gs[1, j])
        _summary_panel(ax, S2, m, L2, EPS, "period", B, {"metric": m}, ttl, "deg / deg" if m == "gain" else "deg")
    fig.suptitle("T3 docked ITI aim relative to the previous trial's spout (capacity vs performance); "
                 "* = boot_delta change vs pre (CI excl. 0 and >= 3/4)", fontsize=9)
    _legend_animals_epochs(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "lc_fig6_t3")


def _save(fig, name: str) -> None:
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


def cmd_plot() -> None:
    from wfield_local import writeguard
    writeguard.assert_writable(FIG)
    _style()
    with open(FD / "learned_compensation.pkl", "rb") as fh:
        fd = pickle.load(fh)
    for f in (fig1_daycourse, fig2_t1_summary, fig3_t2_phases, fig4_t2_summary, fig5_t2_time, fig6_t3):
        f(fd)


def main() -> int:
    a = sys.argv[1:]
    jobs = int(a[a.index("--jobs") + 1]) if "--jobs" in a else 3
    jobs = min(jobs, 3)
    cmd = a[0] if a and not a[0].startswith("--") else "all"
    if "--plot-only" in a or cmd == "plot":
        cmd_plot()
        return 0
    if cmd in ("run", "all"):
        cmd_run(jobs)
    if cmd in ("summarise", "all"):
        cmd_summarise()
    if cmd in ("all", "summarise"):
        cmd_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
