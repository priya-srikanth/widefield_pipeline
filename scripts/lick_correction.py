"""CORRECTION LICKS after a miss (Priya item 4, 10-29): does cortex use error feedback to correct the next lick, and does
that change after the stroke (more contralesional-hemisphere involvement)?

SIGN CONVENTIONS (every table / figure):
  angle     lick angle = degrees(arcsin(side_ml / reach_mm)), + = CONTRALESIONAL (sign forced so the August-pre
            far-contra mean is > 0; as lick_balance_target_model)
  tgt       target angle = the animal's August-PRE mean angle of CONTACT post-cue licks at that spout
  e_k       landing error of lick k = angle_k - tgt (deg); - = landed IPSILESIONAL of the target
  dang      correction = angle_{k+1} - angle_k (deg); toward-target correction = -e_k direction
  miss_k    lick k did NOT contact the spout (licks_all `contact`: a DAQ lick-sensor onset inside the cam4 lick's rise
            start .. fall end +- 8 ms; electrical contact measured, the pose only assigns it to a lick)
  HEMISPHERE: c = contralesional = '_right' features (image right), i = ipsilesional ('_left'); c - i + = contralesional
            hemisphere larger.

BEHAVIOR (all curated sessions with 3-D licks; kinematics_all licks.csv.gz, context post_cue; lick order = lick_idx
within the trial's post-cue licks; pairs k -> k+1 within a trial with both 3-D tracked, peak-to-peak interval <= ILI_MAX):
  per session x spout group (all / far contra / other five) x order (k = 0 first lick / k >= 1 later licks):
    dang = a + g_hit * (-e_k) + d_g * miss_k * (-e_k) + step * miss_k * sign(-e_k) + spout FE
  g_hit  = correction gain after a HIT (1 = the next lick lands back on the target line, 0 = it repeats lick k).
           REGRESSION TO THE MEAN: if licks were independent noise around the target, g = 1 with no feedback at all;
           if they drift as a random walk, g = 0. So g alone is not an error-correction measure.
  g_miss = g_hit + d_g; the FEEDBACK-specific quantities are the MISS-minus-HIT contrasts at the SAME landing error:
           d_g (extra gain) and step (extra deg toward the target after a miss). Also the matched-error binned contrast
           xcorr = mean over 5-deg e_k bins (miss-weighted) of [sign(-e) dang | miss] - [same | hit].
  epochs vs pre: paired nested bootstrap animals > sessions (analysis_kit.boot_delta); per-animal tables first (rule 8).
  recovery: post sessions, MixedLM aim ~ within + between + log(day) + (1 | animal) (lick_recovery_markers.models
  form; aim = this script's far-contra first-lick angle - pre, deg, - = ipsilesional) and measure ~ log(day).

NEURAL (imaging sessions; own extraction: EVERY post-cue lick, no 1500 cap; same area traces as iti_licks.extract:
  ROI v8 masks, drift removed, dF/F x1e3; lags LAGS_X around each lick's exact DAQ peak):
  N1 ERROR SIGNAL (hit vs miss at matched angle): per session, area x hemisphere x window W after lick k's peak,
     y_W = b . [1, y_pre, angle_k, angle_k^2, angle_k+1, reach_k, reach_k+1, contact_k+1, ILI, first, spout FE]
           + gamma * miss_k
     y_pre = level -0.2..-0.1 s before lick k's peak. BOTH licks' angles (and k+1's contact) are in the model because the
     GCaMP6s response to lick k overlaps lick k+1's (ILI ~150-200 ms); LEVELS in fixed windows, not onsets. gamma +
     = larger activity after a miss. Raw dF/F and deconvolved (revup_neural.deconv, tau 0.6 s).
  N2 CORRECTION PREDICTION: per session, 5-fold CV grouped by trial; target dang; baseline OLS on
     [-e_k, miss_k, miss_k * -e_k, first, reach_k, spout FE]; + neural = per area x hemisphere change (W minus y_pre)
     residualised on [angle_k, angle_k^2, reach_k, spout FE] inside each training fold (lick k's angle-predicted response
     removed), ridge (RidgeCV). dR2 = cvR2(baseline + neural) - cvR2(baseline), on all pairs and within miss-k / hit-k
     pairs (SSE reduction / subset SST). Feature sets: both (6 areas x 2), contra (6), ipsi (6), control (RSPd x 2 +
     VISp_right; VISp_left / VISl_left are absent from the session columns). NOTE: activity in any window after lick k
     already contains lick k+1's motor command (calcium lag), so dR2 on HIT pairs = next-command information; the
     error-feedback-specific part is miss minus hit.
  3 POST-STROKE: session values -> per animal x epoch (rule 8) -> boot_delta vs pre; hemisphere difference c - i;
     recovery: MixedLM value ~ aim + log(day) + (1 | animal) on post sessions, per-animal r.

Areas: SSp-n, SSp-m, tjM1, ALM, SSs, MOp (whole MOp, includes tjM1); controls RSPd (both), VISp (right only).
Usage: python scripts/lick_correction.py --extract [--jobs 4]     (per-session neural extraction, cached)
       python scripts/lick_correction.py                          (behavior + neural analyses + figures)
       python scripts/lick_correction.py --plot-only              (redraw from figdata)
Outputs <OUT_W>/lick_correction/ (sessions/<label>.npz, CSVs), figures <OUT_W>/figures/lick_correction_*.png,
figdata <OUT_W>/figdata/lick_correction_*.pkl.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import mask_keys  # noqa: E402

ROOT = il.OUT_W
OUT = ROOT / "lick_correction"
SESS = OUT / "sessions"
FIGS = ROOT / "figures"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
AREAS = ["SSp-n", "SSp-m", "tjM1", "ALM", "SSs", "MOp"]
POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]           # primary orofacial pool (lick_recovery_primary.POOL)
CTRL_COLS = ["RSPd_right", "RSPd_left", "VISp_right"]
LAGS_X = np.round(np.arange(-0.5, 0.8001, 0.05), 3)
ILI_MAX = 0.35
PRE_W = (-0.2, -0.1)
WINS = {"w0_0.0-0.1": (0.0, 0.1), "w1_0.1-0.2": (0.1, 0.2), "w2_0.2-0.35": (0.2, 0.35), "w3_0.35-0.6": (0.35, 0.6)}
PRED_WINS = {"pre_-0.1-0.0": (-0.1, 0.0), "w0_0.0-0.1": (0.0, 0.1), "w1_0.1-0.2": (0.1, 0.2)}
TAUS = (0.0, 0.6)
MIN_CLASS = 10            # misses AND hits needed per session cell
MIN_PAIRS = 40
BIN = 5.0
SEED = 0


# ------------------------------------------------------------------------------------------------ extraction
def kin_table() -> pd.DataFrame:
    from wfield_local.paths import PathResolver
    p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
    K = pd.read_csv(p, usecols=["session", "context", "trial_id", "lick_idx", "lick_uid", "position", "contact",
                                "side_ml", "reach_mm", "out_ap", "down_dv", "miss_mm", "lat_err_mm", "speed_peak",
                                "epoch"])
    K = K[K.context == "post_cue"].copy()
    K["label"] = K.session.str[:4] + "_" + K.session.str[-4:]
    return K


def extract(label: str) -> dict:
    """One imaging session -> SESS/<label>.npz: area traces at LAGS_X around EVERY post-cue lick's exact DAQ peak."""
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    f = SESS / f"{label}.npz"
    if f.exists():
        mask_keys.check_npz(f)                    # stale-read guard (11-09)
        return {"label": label, "cached": True}
    t0 = time.time()
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    from wfield_local.paths import PathResolver as _PR
    p = Path(_PR().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
    K = pd.read_csv(p, usecols=["session", "context", "trial_id", "lick_idx", "lick_uid", "position", "contact",
                                "side_ml", "reach_mm", "out_ap", "down_dv", "miss_mm", "lat_err_mm", "speed_peak"])
    K = K[(K.session == f"{an}_{date}") & (K.context == "post_cue")].copy()
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["context", "daq_on_s", "daq_peak_s", "contact"])
    uid = K.lick_uid.to_numpy(int)
    if not (A.context.to_numpy()[uid] == K.context.to_numpy()).all():
        raise RuntimeError(f"{label}: lick_uid join to licks_all.csv does not match on context")
    if not (A.contact.astype(bool).to_numpy()[uid] == K.contact.astype(bool).to_numpy()).all():
        raise RuntimeError(f"{label}: contact flag differs between licks_all and kinematics_all")
    K["daq_peak_s"] = A.daq_peak_s.to_numpy(float)[uid]
    K["daq_on_s"] = A.daq_on_s.to_numpy(float)[uid]
    lo, hi = ft[0] - LAGS_X[0], ft[-1] - LAGS_X[-1]
    K = K[np.isfinite(K.daq_peak_s) & (K.daq_peak_s > lo) & (K.daq_peak_s < hi)].sort_values("daq_peak_s")
    t = K.daq_peak_s.to_numpy(float)
    X = il._sample(R, ft, t, LAGS_X).astype(np.float16)
    SESS.mkdir(parents=True, exist_ok=True)
    meta = {c: K[c].to_numpy() for c in ["trial_id", "lick_idx", "lick_uid", "side_ml", "reach_mm", "out_ap",
                                          "down_dv", "miss_mm", "lat_err_mm", "speed_peak", "daq_on_s"]}
    np.savez_compressed(f, cols=np.array(cols), X=X, t=t, lags=LAGS_X, contact=K.contact.astype(bool).to_numpy(),
                        position=K.position.to_numpy(dtype=str), epoch=epochs.epoch_of(label), animal=an, **meta,
                        **mask_keys.stamp())
    return {"label": label, "n": len(t), "s": round(time.time() - t0, 1)}


def imaging_labels() -> list[str]:
    import iti_lick_order as lo
    return sorted(pd.read_csv(lo.preds_path(), usecols=["label"]).label.unique())


def run_extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(SESS)
    labels = imaging_labels()
    t0 = time.time()
    res, fail = ak.fan_sessions(labels, extract, jobs=jobs, label="session")
    S = pd.DataFrame([v for _k, v in res])
    S.to_csv(OUT / "extract_summary.csv", index=False)
    print(f"extracted {len(res)} sessions in {(time.time() - t0) / 60:.1f} min; failed {fail[:5]}", flush=True)
    if fail:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ behaviour
def _is_aug_pre(label: str, epoch: str) -> bool:
    from wfield_local import config
    return epoch == "pre" and not config.early_pre(label)


def _exact(label: str) -> pd.DataFrame:
    """lick_uid -> exact DAQ peak time (licks_all.csv; the kinematics table rounds daq_peak_s to 4 significant digits)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_peak_s"])
    return pd.DataFrame({"label": label, "lick_uid": np.arange(len(A)), "t": A.daq_peak_s.to_numpy(float)})


def beh_licks() -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Every post-cue lick of the curated sessions with angle, target, error; plus the sign factor and target table."""
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    K = kin_table()
    labs = ak.curated_labels()
    K = K[K.label.isin(labs)].copy()
    ep = {lab: epochs.epoch_of(lab) for lab in sorted(K.label.unique())}
    K["epoch"] = K.label.map(ep)
    K = K[K.epoch.isin(EPS)].copy()
    K["animal"] = K.label.str[:4]
    K["pos"] = [oc.pos_key(p, "L") for p in K.position]          # all four lesions are LEFT (orofacial_cohort)
    K["aug_pre"] = [_is_aug_pre(lab, e) for lab, e in zip(K.label, K.epoch)]
    res, fail = ak.fan_sessions(sorted(K.label.unique()), _exact, jobs=4)
    if fail:
        raise RuntimeError(f"exact times failed: {fail[:5]}")
    T = pd.concat([v for _k, v in res], ignore_index=True)
    K = K.merge(T, on=["label", "lick_uid"], how="left", validate="many_to_one")
    K["angle"] = np.degrees(np.arcsin(np.clip(K.side_ml / K.reach_mm, -1, 1)))
    sgn = float(np.sign(K[K.aug_pre & (K.pos == "far_contra")].angle.mean()))
    K["angle"] *= sgn
    K["contact"] = K.contact.astype(bool)
    TG = (K[K.aug_pre & K.contact].groupby(["animal", "pos"]).angle.agg(["mean", "size"]).reset_index()
          .rename(columns={"mean": "tgt", "size": "n_tgt"}))
    K = K.merge(TG[["animal", "pos", "tgt"]], on=["animal", "pos"], how="left")
    K["e"] = K.angle - K.tgt
    days = {lab: epochs.days_since_stroke(lab) for lab in sorted(K.label.unique())}
    K["day"] = K.label.map(days)
    return K.sort_values(["label", "trial_id", "lick_idx"]).reset_index(drop=True), sgn, TG


def pairs_of(K: pd.DataFrame) -> pd.DataFrame:
    """Consecutive post-cue licks k -> k+1 of the same trial, both with an angle, peak-to-peak <= ILI_MAX."""
    K = K.sort_values(["label", "trial_id", "lick_idx"]).reset_index(drop=True)
    nx = K.shift(-1)
    ok = ((nx.label == K.label) & (nx.trial_id == K.trial_id) & (nx.lick_idx == K.lick_idx + 1)
          & np.isfinite(K.angle) & np.isfinite(nx.angle.astype(float)))
    P = K[ok].copy()
    N = nx[ok]
    P["angle1"] = N.angle.astype(float).to_numpy()
    P["e1"] = N.e.astype(float).to_numpy()
    P["contact1"] = N.contact.astype(bool).to_numpy()
    P["reach1"] = N.reach_mm.astype(float).to_numpy()
    P["ili"] = N.t.astype(float).to_numpy() - P.t.to_numpy(float)
    if "row" in K:
        P["row1"] = N.row.astype(float).to_numpy()
    # previous lick's error (same trial, consecutive) -- a trial-level aim-bias covariate for k >= 1
    pv = K.shift(1)
    okp = (pv.label == K.label) & (pv.trial_id == K.trial_id) & (pv.lick_idx == K.lick_idx - 1)
    P["e_prev"] = np.where(okp[ok], pv.e[ok].astype(float), np.nan)
    P = P[(P.ili > 0) & (P.ili <= ILI_MAX) & np.isfinite(P.e)].copy()
    P["dang"] = P.angle1 - P.angle
    P["miss"] = (~P.contact).astype(float)
    P["first"] = (P.lick_idx == 0).astype(float)
    P["toward"] = np.sign(-P.e) * P.dang                  # + = moved toward the target line
    return P.reset_index(drop=True)


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and SEs (classical)."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(r @ r) / dof
    try:
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return b, se


def _spout_fe(pos: np.ndarray) -> np.ndarray:
    lv = [p for p in POS if (pos == p).any()]
    return np.column_stack([(pos == p).astype(float) for p in lv[1:]]) if len(lv) > 1 else np.zeros((len(pos), 0))


def corr_fit(g: pd.DataFrame) -> dict:
    """Correction model on one cell of pairs (module docstring)."""
    out = {"n": len(g), "n_miss": int(g.miss.sum()), "miss_rate": float(g.miss.mean()) if len(g) else np.nan}
    nm, nh = int(g.miss.sum()), int((1 - g.miss).sum())
    ne = -g.e.to_numpy(float)
    y = g.dang.to_numpy(float)
    m = g.miss.to_numpy(float)
    F = _spout_fe(g.pos.to_numpy())
    if nh >= MIN_CLASS:
        h = m == 0
        b, _ = _ols(np.column_stack([np.ones(h.sum()), ne[h], F[h]]), y[h])
        out["g_hit"] = b[1]
        out["hit1_after_hit"] = float(g.contact1[h].mean())
    if nm >= MIN_CLASS:
        k = m == 1
        b, _ = _ols(np.column_stack([np.ones(k.sum()), ne[k], F[k]]), y[k])
        out["g_miss"] = b[1]
        out["toward_miss"] = float(g.toward[k].mean())
        out["hit1_after_miss"] = float(g.contact1[k].mean())
        out["e_miss"] = float(g.e[k].mean())
    if nm >= MIN_CLASS and nh >= MIN_CLASS:
        X = np.column_stack([np.ones(len(g)), ne, m * ne, m * np.sign(ne), F])
        b, se = _ols(X, y)
        out.update({"d_g": b[2], "step": b[3], "step_se": se[3]})
        # TRIAL-BIAS CONTROL (k >= 1): a miss selects licks from trials aimed off target, so the next lick repeats
        # the bias; conditioning on the previous lick's error (e_k-1) removes a persistent trial-level aim component
        pe = g.e_prev.to_numpy(float)
        q = np.isfinite(pe)
        if (m[q] == 1).sum() >= MIN_CLASS and (m[q] == 0).sum() >= MIN_CLASS:
            Xq = np.column_stack([np.ones(q.sum()), ne[q], -pe[q], m[q] * ne[q], m[q] * np.sign(ne[q]), F[q]])
            b, se = _ols(Xq, y[q])
            out.update({"g_hit_b": b[1], "prev_w": b[2], "d_g_b": b[3], "step_b": b[4], "step_b_se": se[4]})
        # matched-error binned contrast
        bins = np.floor(g.e.to_numpy(float) / BIN)
        d, w = [], []
        for bb in np.unique(bins):
            s = bins == bb
            if (s & (m == 1)).sum() >= 2 and (s & (m == 0)).sum() >= 2:
                tw = g.toward.to_numpy(float)
                d.append(tw[s & (m == 1)].mean() - tw[s & (m == 0)].mean())
                w.append((s & (m == 1)).sum())
        out["xcorr"] = float(np.average(d, weights=w)) if d else np.nan
        out["xcorr_cover"] = float(np.sum(w) / nm) if d else 0.0
    return out


GROUPS = {"all": None, "far_contra": ["far_contra"], "other5": [p for p in POS if p != "far_contra"]}
ORDERS = {"first": lambda P: P.lick_idx == 0, "later": lambda P: P.lick_idx >= 1, "any": lambda P: P.lick_idx >= 0}
BMET = ["g_hit", "g_miss", "d_g", "step", "xcorr", "step_b", "d_g_b", "g_hit_b", "prev_w", "toward_miss", "miss_rate",
        "hit1_after_miss", "hit1_after_hit"]


def beh_sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lab, an, ep, aug, day), g in P.groupby(["label", "animal", "epoch", "aug_pre", "day"], sort=True,
                                                dropna=False):
        if ep == "pre" and not aug:
            continue
        for gn, ps in list(GROUPS.items()) + [(p, [p]) for p in POS if p not in GROUPS]:
            gg = g if ps is None else g[g.pos.isin(ps)]
            for on, f in ORDERS.items():
                c = gg[f(gg)]
                if len(c) < MIN_CLASS:
                    continue
                rows.append({"label": lab, "animal": an, "epoch": ep, "day": day, "group": gn, "order": on,
                             **corr_fit(c)})
    return pd.DataFrame(rows)


def session_aim(K: pd.DataFrame) -> pd.Series:
    """Far-contra FIRST post-cue lick angle minus the animal's August-pre mean of the same (deg, - = ipsilesional)."""
    F = K[(K.pos == "far_contra") & (K.lick_idx == 0) & np.isfinite(K.angle)]
    ref = F[F.aug_pre].groupby("animal").angle.mean()
    s = F.groupby(["label", "animal"]).angle.mean().reset_index()
    s["aim"] = s.angle - s.animal.map(ref)
    return s.set_index("label").aim


def per_animal(S: pd.DataFrame, keys: list[str], mets: list[str]) -> pd.DataFrame:
    """Rule 8: per animal x epoch mean over sessions (and n sessions)."""
    A = S.groupby(keys + ["animal", "epoch"])[mets].agg(["mean", "count"])
    A.columns = [f"{m}_{s}" for m, s in A.columns]
    return A.reset_index()


def boot_table(S: pd.DataFrame, keys: list[str], mets: list[str], seed: int = SEED) -> pd.DataFrame:
    """Per cell: pooled pre level (boot_ci) and epoch - pre change (boot_delta), n animals same sign."""
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(seed)
    rows = []
    for kv, g in S.groupby(keys, sort=True):
        kv = kv if isinstance(kv, tuple) else (kv,)
        for m in mets:
            gm = g.dropna(subset=[m])
            pre = {a: list(h[m]) for a, h in gm[gm.epoch == "pre"].groupby("animal", sort=True)}
            for ep in EPS:
                cur = {a: list(h[m]) for a, h in gm[gm.epoch == ep].groupby("animal", sort=True)}
                lev = ak.boot_ci(cur, rng) if cur else None
                r = {**dict(zip(keys, kv)), "metric": m, "epoch": ep,
                     "level": lev.point if lev else np.nan, "level_lo": lev.lo if lev else np.nan,
                     "level_hi": lev.hi if lev else np.nan, "n_animals": len(cur)}
                if ep != "pre":
                    dl = ak.boot_delta(cur, pre, rng)
                    common = sorted(set(cur) & set(pre))
                    da = [np.mean(cur[a]) - np.mean(pre[a]) for a in common]
                    r.update({"delta": dl.point if dl else np.nan, "delta_lo": dl.lo if dl else np.nan,
                              "delta_hi": dl.hi if dl else np.nan, "n_pos": int(np.sum(np.array(da) > 0)),
                              "n_paired": len(common),
                              "per_animal_delta": " ".join(f"{a}:{x:+.3g}" for a, x in zip(common, da))})
                rows.append(r)
    return pd.DataFrame(rows)


def track(S: pd.DataFrame, keys: list[str], mets: list[str], aim: pd.Series) -> pd.DataFrame:
    """Post-stroke sessions: per-animal r (vs aim, vs log day) and MixedLM aim ~ within + between + log(day) and
    value ~ log(day) (+ (1 | animal))."""
    import statsmodels.formula.api as smf
    rows = []
    S = S.copy()
    S["aim"] = S.label.map(aim)
    for kv, g in S[(S.epoch != "pre") & (S.day > 0)].groupby(keys, sort=True):
        kv = kv if isinstance(kv, tuple) else (kv,)
        for m in mets:
            h = g.dropna(subset=[m, "aim"]).copy()
            if h.animal.nunique() < 3 or len(h) < 15:
                continue
            h["logday"] = np.log(h.day.astype(float))
            h["v"] = h[m].astype(float)
            r = {**dict(zip(keys, kv)), "metric": m, "n": len(h)}
            for a in AN:
                ha = h[h.animal == a]
                r[f"r_aim_{a}"] = np.corrcoef(ha.v, ha.aim)[0, 1] if len(ha) >= 4 and ha.v.std() > 0 else np.nan
                r[f"r_logday_{a}"] = (np.corrcoef(ha.v, ha.logday)[0, 1] if len(ha) >= 4 and ha.v.std() > 0
                                      else np.nan)
            h["w"] = h.v - h.groupby("animal").v.transform("mean")
            h["b"] = h.groupby("animal").v.transform("mean")
            try:
                mm = smf.mixedlm("aim ~ w + b + logday", h, groups=h["animal"]).fit(reml=False)
                r.update({"aim_slope_w": mm.fe_params["w"], "aim_lo": mm.conf_int().loc["w", 0],
                          "aim_hi": mm.conf_int().loc["w", 1], "aim_p": mm.pvalues["w"]})
            except Exception as ex:                                            # noqa: BLE001
                r["note"] = f"aim model failed: {ex}"
            try:
                mm = smf.mixedlm("v ~ logday", h, groups=h["animal"]).fit(reml=False)
                r.update({"logday_slope": mm.fe_params["logday"], "logday_lo": mm.conf_int().loc["logday", 0],
                          "logday_hi": mm.conf_int().loc["logday", 1], "logday_p": mm.pvalues["logday"]})
            except Exception as ex:                                            # noqa: BLE001
                r["note2"] = f"logday model failed: {ex}"
            rows.append(r)
    return pd.DataFrame(rows)


def binned_curves(P: pd.DataFrame) -> pd.DataFrame:
    """For the figure: mean dang by 5-deg bin of -e_k, hit vs miss k, per epoch x group x animal (August pre only)."""
    Q = P[(P.epoch != "pre") | P.aug_pre].copy()
    Q["xb"] = (np.floor(-Q.e / BIN) + 0.5) * BIN
    rows = []
    for gn, ps in GROUPS.items():
        g = Q if ps is None else Q[Q.pos.isin(ps)]
        for (ep, an, ms, xb), h in g.groupby(["epoch", "animal", "miss", "xb"]):
            rows.append({"group": gn, "epoch": ep, "animal": an, "miss": ms, "x": xb, "y": h.dang.mean(), "n": len(h)})
    return pd.DataFrame(rows)


def behaviour() -> dict:
    K, sgn, TG = beh_licks()
    print(f"angle sign factor {sgn:+.0f}; {len(K)} post-cue licks, {K.label.nunique()} sessions", flush=True)
    TG.to_csv(OUT / "corr_targets.csv", index=False)
    P = pairs_of(K)
    print(f"{len(P)} pairs (ILI <= {ILI_MAX} s); miss rate {P.miss.mean():.3f}", flush=True)
    P.to_csv(OUT / "corr_pairs.csv.gz", index=False)
    aim = session_aim(K)
    aim.rename("aim").to_csv(OUT / "corr_session_aim.csv")
    S = beh_sessions(P)
    S.to_csv(OUT / "corr_beh_sessions.csv", index=False)
    PA = per_animal(S[S.group.isin(["all", "far_contra", "other5"])], ["group", "order"], BMET)
    PA.to_csv(OUT / "corr_beh_per_animal.csv", index=False)
    print("\n== PER-ANIMAL (rule 8): behavior, order any ==")
    show = PA[PA.order.isin(["first", "later"])]
    print(show[["group", "order", "animal", "epoch", "g_hit_mean", "g_miss_mean", "step_mean", "xcorr_mean",
                "miss_rate_mean", "step_count"]].round(3).to_string(index=False), flush=True)
    B = boot_table(S, ["group", "order"], BMET)
    B.to_csv(OUT / "corr_beh_boot.csv", index=False)
    TR = track(S[S.group.isin(["all", "far_contra", "other5"])], ["group", "order"],
               ["g_hit", "g_miss", "d_g", "step", "xcorr", "step_b", "toward_miss"], aim)
    TR.to_csv(OUT / "corr_beh_track.csv", index=False)
    return {"S": S, "B": B, "TR": TR, "C": binned_curves(P), "sgn": sgn}


# ------------------------------------------------------------------------------------------------ neural
def _deconv(X: np.ndarray, tau: float) -> np.ndarray:
    """revup_neural.deconv along the lag axis (axis 1) of (licks, lags, cols)."""
    import revup_neural as rn
    if not tau:
        return X
    return np.moveaxis(rn.deconv(np.moveaxis(X, 1, 0), tau), 0, 1)


def _wmean(X: np.ndarray, w: tuple) -> np.ndarray:
    s = (LAGS_X >= w[0] - 1e-9) & (LAGS_X < w[1] - 1e-9)
    return X[:, s].mean(1)


def neural_session(item) -> dict:
    """One session: N1 gamma rows and N2 dR2 rows (module docstring). item = (label, sgn, {(animal, pos): tgt})."""
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.model_selection import GroupKFold

    from wfield_local import orofacial_cohort as oc
    label, sgn, tgt = item
    f = SESS / f"{label}.npz"
    z = np.load(f, allow_pickle=True)
    mask_keys.check_obj(z, f)                     # stale-read guard (11-09)
    an, ep = str(z["animal"]), str(z["epoch"])
    cols = list(z["cols"])
    L = pd.DataFrame({k: z[k] for k in ["trial_id", "lick_idx", "side_ml", "reach_mm", "t"]})
    L["row"] = np.arange(len(L))
    L["contact"] = z["contact"].astype(bool)
    L["pos"] = [oc.pos_key(p, "L") for p in z["position"]]
    L["angle"] = sgn * np.degrees(np.arcsin(np.clip(L.side_ml / L.reach_mm, -1, 1)))
    L["tgt"] = [tgt.get((an, p), np.nan) for p in L.pos]
    L["e"] = L.angle - L.tgt
    L["label"], L["aug_pre"] = label, _is_aug_pre(label, ep)
    P = pairs_of(L)
    P = P.dropna(subset=["row1", "angle1", "reach1", "reach_mm"]).reset_index(drop=True)
    G, D = [], []
    base = {"label": label, "animal": an, "epoch": ep}
    nm, nh = int(P.miss.sum()), int((1 - P.miss).sum())
    if len(P) < MIN_PAIRS:
        return {"G": G, "D": D, "n_pairs": len(P), "n_miss": nm}
    feats = {f"{a}|{h}": cols.index(f"{a}_{'right' if h == 'c' else 'left'}") for a in AREAS for h in ("c", "i")}
    feats.update({c.replace("_right", "|c").replace("_left", "|i"): cols.index(c) for c in CTRL_COLS if c in cols})
    rows = P.row.to_numpy(int)
    F = _spout_fe(P.pos.to_numpy())
    ang, ang1 = P.angle.to_numpy(float), P.angle1.to_numpy(float)
    for tau in TAUS:
        Xall = _deconv(z["X"][rows].astype(float), tau)
        lv = {k: {w: _wmean(Xall[:, :, j], ws) for w, ws in {**WINS, **PRED_WINS, "pre": PRE_W}.items()}
              for k, j in feats.items()}
        for h in ("c", "i"):                                          # primary pool (sum of 5 orofacial areas)
            lv[f"POOL|{h}"] = {w: sum(lv[f"{a}|{h}"][w] for a in POOL) for w in lv[f"ALM|{h}"]}
        # N1 gamma
        if nm >= MIN_CLASS and nh >= MIN_CLASS:
            for k in lv:
                for w in WINS:
                    X = np.column_stack([np.ones(len(P)), lv[k]["pre"], ang, ang ** 2 / 10, ang1,
                                         P.reach_mm.to_numpy(float), P.reach1.to_numpy(float),
                                         P.contact1.to_numpy(float), P.ili.to_numpy(float), P["first"].to_numpy(float),
                                         F, P.miss.to_numpy(float)])
                    b, se = _ols(X, lv[k][w])
                    hitmean = float((lv[k][w] - lv[k]["pre"])[P.miss.to_numpy() == 0].mean())
                    G.append({**base, "tau": tau, "feat": k, "win": w, "gamma": b[-1], "gamma_se": se[-1],
                              "resp_hit": hitmean, "n": len(P), "n_miss": nm})
        # N2 correction prediction
        if nm < MIN_CLASS or nh < MIN_CLASS:
            continue
        y = P.dang.to_numpy(float)
        ne = -P.e.to_numpy(float)
        m = P.miss.to_numpy(float)
        XB = np.column_stack([ne, m, m * ne, P["first"].to_numpy(float), P.reach_mm.to_numpy(float), F])
        XR = np.column_stack([ang, ang ** 2 / 10, P.reach_mm.to_numpy(float), F])     # residualise neural on these
        grp = P.trial_id.to_numpy()
        nf = min(5, len(np.unique(grp)))
        sets = {"both": [f"{a}|{h}" for a in AREAS for h in ("c", "i")], "contra": [f"{a}|c" for a in AREAS],
                "ipsi": [f"{a}|i" for a in AREAS], "control": [k for k in feats if k.split("|")[0] in ("RSPd", "VISp")]}
        sets.update({f"area_{a}": [f"{a}|c", f"{a}|i"] for a in AREAS})
        for w in PRED_WINS:
            pred_b = np.zeros(len(y))
            preds = {s: np.zeros(len(y)) for s in sets}
            for tr, te in GroupKFold(nf).split(XB, y, grp):
                lb = LinearRegression().fit(XB[tr], y[tr])
                pred_b[te] = lb.predict(XB[te])
                for s, ks in sets.items():
                    N = np.column_stack([lv[k][w] - lv[k]["pre"] for k in ks])
                    rr = LinearRegression().fit(XR[tr], N[tr])
                    Nr = N - rr.predict(XR)
                    mu, sd = Nr[tr].mean(0), Nr[tr].std(0) + 1e-9
                    Z = np.column_stack([XB, (Nr - mu) / sd])
                    mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Z[tr], y[tr])
                    preds[s][te] = mdl.predict(Z[te])
            for sub, sm in (("all", np.ones(len(y), bool)), ("miss", m == 1), ("hit", m == 0)):
                sst = float(((y[sm] - y[sm].mean()) ** 2).sum())
                sse_b = float(((y[sm] - pred_b[sm]) ** 2).sum())
                for s in sets:
                    sse = float(((y[sm] - preds[s][sm]) ** 2).sum())
                    D.append({**base, "tau": tau, "win": w, "set": s, "subset": sub, "n": int(sm.sum()),
                              "r2_base": 1 - sse_b / sst, "dR2": (sse_b - sse) / sst})
    return {"G": G, "D": D, "n_pairs": len(P), "n_miss": nm}


def neural(sgn: float) -> dict:
    from wfield_local import config
    from wfield_local import analysis_kit as ak
    TG = pd.read_csv(OUT / "corr_targets.csv")
    tgt = {(a, p): float(t) for a, p, t in zip(TG.animal, TG.pos, TG.tgt)}
    labs = sorted(p.stem for p in SESS.glob("*.npz"))
    res, fail = ak.fan_sessions([(lab, sgn, tgt) for lab in labs], neural_session, jobs=4, key=lambda x: x[0])
    if fail:
        raise RuntimeError(f"neural failed: {[f[0] if isinstance(f, tuple) else f for f in fail][:5]}")
    G = pd.DataFrame([r for _k, v in res for r in v["G"]])
    D = pd.DataFrame([r for _k, v in res for r in v["D"]])
    N = pd.DataFrame([{"label": k[0], "n_pairs": v["n_pairs"], "n_miss": v["n_miss"]} for k, v in res])
    N.to_csv(OUT / "corr_neural_counts.csv", index=False)
    # August-pre only as the pre epoch
    aug = {lab for lab in labs if not config.early_pre(lab)}
    G = G[(G.epoch != "pre") | G.label.isin(aug)]
    D = D[(D.epoch != "pre") | D.label.isin(aug)]
    G.to_csv(OUT / "corr_gamma_sessions.csv", index=False)
    D.to_csv(OUT / "corr_pred_sessions.csv", index=False)
    # hemisphere difference c - i per session
    GW = G.pivot_table(index=["label", "animal", "epoch", "tau", "win", "n_miss"], columns="feat", values="gamma"
                       ).reset_index()
    for a in AREAS + ["POOL", "RSPd"]:
        if f"{a}|c" in GW and f"{a}|i" in GW:
            GW[f"{a}|c-i"] = GW[f"{a}|c"] - GW[f"{a}|i"]
    GL = GW.melt(id_vars=["label", "animal", "epoch", "tau", "win", "n_miss"], var_name="feat", value_name="gamma")
    DW = D.pivot_table(index=["label", "animal", "epoch", "tau", "win", "subset"], columns="set", values="dR2"
                       ).reset_index()
    DW["contra-ipsi"] = DW["contra"] - DW["ipsi"]
    DL = DW.melt(id_vars=["label", "animal", "epoch", "tau", "win", "subset"], var_name="set", value_name="dR2")
    DM = DL.pivot_table(index=["label", "animal", "epoch", "tau", "win", "set"], columns="subset", values="dR2"
                        ).reset_index()
    DM["miss-hit"] = DM["miss"] - DM["hit"]
    DL2 = DM.melt(id_vars=["label", "animal", "epoch", "tau", "win", "set"], var_name="subset", value_name="dR2")
    PAG = per_animal(GL, ["tau", "win", "feat"], ["gamma"])
    PAG.to_csv(OUT / "corr_gamma_per_animal.csv", index=False)
    PAD = per_animal(DL2, ["tau", "win", "set", "subset"], ["dR2"])
    PAD.to_csv(OUT / "corr_pred_per_animal.csv", index=False)
    print("\n== PER-ANIMAL (rule 8): gamma (miss - hit, matched angles), raw, POOL ==")
    print(PAG[(PAG.tau == 0) & PAG.feat.str.startswith("POOL")].pivot_table(
        index=["win", "feat", "animal"], columns="epoch", values="gamma_mean")[EPS].round(2).to_string(), flush=True)
    print("\n== PER-ANIMAL (rule 8): dR2 correction prediction, raw, all/miss/hit ==")
    print(PAD[(PAD.tau == 0) & PAD.set.isin(["both", "contra", "ipsi", "control"])].pivot_table(
        index=["win", "set", "subset", "animal"], columns="epoch", values="dR2_mean")[EPS].round(3).to_string(),
        flush=True)
    BG = boot_table(GL.rename(columns={"gamma": "v"}), ["tau", "win", "feat"], ["v"])
    BG.to_csv(OUT / "corr_gamma_boot.csv", index=False)
    BD = boot_table(DL2.rename(columns={"dR2": "v"}), ["tau", "win", "set", "subset"], ["v"])
    BD.to_csv(OUT / "corr_pred_boot.csv", index=False)
    aim = pd.read_csv(OUT / "corr_session_aim.csv").set_index("label").aim
    from wfield_local import epochs
    GLd = GL.assign(day=GL.label.map({lab: epochs.days_since_stroke(lab) for lab in labs}))
    DLd = DL2.assign(day=DL2.label.map({lab: epochs.days_since_stroke(lab) for lab in labs}))
    keyf = [f"{a}|{h}" for a in ["POOL", "SSp-n", "SSp-m", "tjM1", "ALM", "SSs", "MOp"] for h in ("c", "i", "c-i")]
    TG_ = track(GLd[(GLd.tau == 0) & GLd.feat.isin(keyf)].rename(columns={"gamma": "v_gamma"}), ["win", "feat"],
                ["v_gamma"], aim)
    TD_ = track(DLd[(DLd.tau == 0) & DLd.set.isin(["both", "contra", "ipsi", "contra-ipsi"])].rename(
        columns={"dR2": "v_dR2"}), ["win", "set", "subset"], ["v_dR2"], aim)
    TG_.to_csv(OUT / "corr_gamma_track.csv", index=False)
    TD_.to_csv(OUT / "corr_pred_track.csv", index=False)
    return {"BG": BG, "BD": BD, "PAG": PAG, "PAD": PAD, "TG": TG_, "TD": TD_, "GL": GL, "DL": DL2}


# ------------------------------------------------------------------------------------------------ figures
def _wrap(s: str, n: int = 110) -> str:
    import textwrap
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))


def _epoch_panel(ax, B: pd.DataFrame, S: pd.DataFrame | None, val: str, sel: dict, lines: dict, title: str,
                 ylabel: str) -> None:
    """Pooled level +- CI per epoch (one line per `lines` entry) + per-animal session means (markers)."""
    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    x = np.arange(len(EPS))
    for j, (name, (selb, col)) in enumerate(lines.items()):
        b = B
        for k, v in {**sel, **selb}.items():
            b = b[b[k] == v]
        b = b.set_index("epoch").reindex(EPS)
        off = (j - (len(lines) - 1) / 2) * 0.12
        ax.errorbar(x + off, b.level, yerr=[b.level - b.level_lo, b.level_hi - b.level], color=col, marker="_",
                    ms=14, lw=1.6, capsize=2, label=name)
        for i, ep in enumerate(EPS):
            if ep != "pre" and "delta" in b and np.isfinite(b.loc[ep, "delta"]):
                if b.loc[ep, "delta_lo"] > 0 or b.loc[ep, "delta_hi"] < 0:
                    ax.text(i + off, b.loc[ep, "level_hi"], "*", color=col, ha="center", va="bottom", fontsize=11)
        if S is not None:
            s = S
            for k, v in {**sel, **selb}.items():
                if k in s:
                    s = s[s[k] == v]
            for an, h in s.groupby("animal"):
                mu = h.groupby("epoch")[val].mean().reindex(EPS)
                ax.plot(x + off + 0.04, mu, ls="none", marker=ANIMAL_MARKER.get(an, "o"), mfc="none", mec=col,
                        ms=4, alpha=0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(x, EPS, fontsize=7)
    ax.set_title(_wrap(title, 48), fontsize=8)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=7)


XLAB = ("lick k landed this far IPSILESIONAL of the target (deg)\n(target - landed; + = ipsilesional miss, "
        "- = contralesional miss)")
YLAB = "next lick's shift, angle_k+1 - angle_k\n(deg; + = moved contralesional)"


def _curve_panel(ax, C: pd.DataFrame, gn: str, ep: str, eps: list, marker: dict, ecol: dict) -> None:
    """Binned next-lick shift vs lick k's landing error (hit solid, miss dashed); per-animal thin gray, mean thick."""
    for e in eps:
        g = C[(C.group == gn) & (C.epoch == e) & (C.n >= 3)]
        for ms, ls, lab in ((0, "-", "lick k HIT"), (1, "--", "lick k MISS")):
            h = g[g.miss == ms]
            for an, ha in h.groupby("animal"):
                ax.plot(ha.x, ha.y, ls=ls, color="0.75", lw=0.6, marker=marker[an], ms=2.5)
            mu = h.groupby("x").y.mean()
            mu = mu[h.groupby("x").animal.nunique() >= 3]
            ax.plot(mu.index, mu.values, ls=ls, color=ecol[e], lw=2.2, label=f"{e}: {lab} (mean of animals)")
    ax.plot([-40, 40], [-40, 40], color="0.8", lw=1.0, zorder=0)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-30, 40)
    ax.text(-38, 38, "gray diagonal, slope 1 = next lick returns\nfully to the target (also what independent\n"
            "licks scattered around the target give);\nslope 0 = next lick repeats lick k", fontsize=6.5,
            va="top", ha="left", color="0.3")
    ax.text(38, -28, "e.g. x = +10: lick k landed 10 deg IPSI of\ntarget; y = +10: next lick moved 10 deg\n"
            "CONTRA, i.e. back onto the target", fontsize=6.5, ha="right", va="bottom", color="0.3")
    ax.set_title(f"{ep.upper()}, {'all spouts' if gn == 'all' else 'far contra spout'}", fontsize=9)
    ax.set_xlabel(XLAB, fontsize=7)
    ax.set_ylabel(YLAB, fontsize=7)
    ax.tick_params(labelsize=7)


def plot(FB: dict, FN: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    FIGS.mkdir(parents=True, exist_ok=True)
    S, B, C = FB["S"], FB["B"], FB["C"]
    GCOL = {"far contra": ("far_contra", "#b2182b"), "other 5 spouts": ("other5", "#2166ac")}
    # ---------- figure 1: behaviour
    fig, axs = plt.subplots(4, 4, figsize=(15, 15.6), layout="constrained")
    for r, gn in enumerate(["all", "far_contra"]):
        for c, ep in enumerate(EPS):
            _curve_panel(axs[r, c], C, gn, ep, [ep], ANIMAL_MARKER, EPOCH_LINE)
    lines = {k: ({"group": v[0]}, v[1]) for k, v in GCOL.items()}
    for c, (m, yl, tt) in enumerate([("g_hit", "gain (1 = back on target line)", "1-3. correction GAIN after a HIT"),
                                     ("g_miss", "gain", "1-4. correction GAIN after a MISS"),
                                     ("step", "deg toward target", "1-5. EXTRA step toward target after a MISS vs a "
                                                                   "HIT at the same error (regression)"),
                                     ("xcorr", "deg toward target", "1-6. EXTRA correction after a MISS, 5-deg "
                                                                    "error-matched bins")]):
        _epoch_panel(axs[2, c], B[(B.metric == m) & (B.order == "any")], S[S.order == "any"], m, {}, lines,
                     f"{tt}; all lick orders", yl)
    for c, (m, on) in enumerate([("step", "first"), ("step", "later"), ("xcorr", "first"), ("xcorr", "later")]):
        _epoch_panel(axs[3, c], B[(B.metric == m) & (B.order == on)], S[S.order == on], m, {}, lines,
                     f"1-{7 + c}. {m} after a miss, lick k = {'FIRST lick' if on == 'first' else 'LATER licks'}",
                     "deg toward target")
    fig.suptitle(_wrap("Fig 1. BEHAVIOR -- correction of the next lick after a miss (cued post-cue licks, pairs k -> "
                       "k+1 within a trial, ILI <= 0.35 s). Bars = pooled session level, 95% nested bootstrap CI; "
                       "* = epoch differs from pre (paired boot_delta); open markers = animals (o PS92, s PS93, ^ "
                       "PS94, D PS95). Gain alone includes regression to the mean; the MISS-minus-HIT contrasts "
                       "(1-5..1-10) are the feedback-specific measures.", 170), fontsize=9)
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[2, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, ["rows 1-2: " + x.split(": ", 1)[1].replace(" (mean of animals)", "") + " (mean of animals; color = epoch)"
                         for x in l1] + [f"rows 3-4: {x}" for x in l2], loc="outside lower center", ncol=4, fontsize=8,
               frameon=False)
    save_png_svg(fig, FIGS / "lick_correction_behaviour.png", dpi=110)
    plt.close(fig)
    # ---------- figure 1b: COMPACT curves (Priya 10-29): pre vs chronic only, all spouts + far contra
    fig, axs = plt.subplots(1, 4, figsize=(17, 5.4), layout="constrained")
    for c, (gn, ep) in enumerate([("all", "pre"), ("all", "chronic"), ("far_contra", "pre"), ("far_contra", "chronic")]):
        _curve_panel(axs[c], C, gn, ep, [ep], ANIMAL_MARKER, EPOCH_LINE)
    h1, l1 = axs[0].get_legend_handles_labels()
    h3, l3 = axs[1].get_legend_handles_labels()
    fig.legend(h1 + h3, l1 + l3,
               loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(_wrap("Next-lick correction, cued post-cue licks (pairs k -> k+1 within a trial, ILI <= 0.35 s), PRE vs "
                       "CHRONIC. x = how far lick k landed IPSILESIONAL of the target (+) or contralesional (-); y = how "
                       "far the next lick moved (+ = contralesional). Points on the gray diagonal (slope 1) = the next "
                       "lick returns fully to the target, which is also what independent licks scattered around the "
                       "target give; slope 0 = the next lick repeats lick k. Solid = lick k contacted the spout (HIT), "
                       "dashed = MISS; thick = mean of animals (bins with >= 3 animals), thin gray = animals. Acute / "
                       "subacute panels: lick_correction_behaviour.png. PROVISIONAL (n = 4).", 190), fontsize=9.5)
    save_png_svg(fig, FIGS / "lick_correction_behaviour_compact.png", dpi=110)
    plt.close(fig)
    # ---------- figure 2: per spout x epoch
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    for ax, m, tt in zip(axs.ravel(), ["miss_rate", "g_miss", "step", "xcorr"],
                         ["2-1. miss rate of lick k", "2-2. gain after a MISS", "2-3. extra step after a miss (deg)",
                          "2-4. extra correction, error-matched (deg)"]):
        b = B[(B.metric == m) & (B.order == "any") & B.group.isin(POS)]
        M = b.pivot_table(index="group", columns="epoch", values="level").reindex(index=POS, columns=EPS)
        Dl = b.pivot_table(index="group", columns="epoch", values="delta").reindex(index=POS, columns=EPS)
        sig = b.assign(s=(b.delta_lo > 0) | (b.delta_hi < 0)).pivot_table(index="group", columns="epoch", values="s"
                                                                          ).reindex(index=POS, columns=EPS)
        npos = b.pivot_table(index="group", columns="epoch", values="n_pos").reindex(index=POS, columns=EPS)
        vmax = np.nanmax(np.abs(M.to_numpy())) if np.isfinite(M.to_numpy()).any() else 1
        im = ax.imshow(M.to_numpy(float), cmap="RdBu_r" if m != "miss_rate" else "viridis",
                       vmin=-vmax if m != "miss_rate" else 0, vmax=vmax, aspect="auto")
        for i in range(len(POS)):
            for j in range(len(EPS)):
                v = M.iloc[i, j]
                if np.isfinite(v):
                    t = f"{v:.2f}"
                    if j > 0 and np.isfinite(Dl.iloc[i, j]):
                        t += f"\nd{Dl.iloc[i, j]:+.2f}{'*' if sig.iloc[i, j] == 1 else ''} ({int(npos.iloc[i, j])}+)"
                    ax.text(j, i, t, ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(EPS)), EPS, fontsize=9)
        ax.set_yticks(range(len(POS)), [p.replace("_", " ") for p in POS], fontsize=9)
        ax.set_title(_wrap(tt, 60), fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle(_wrap("Fig 2. Correction by SPOUT x EPOCH (all lick orders; pooled session level; d = change vs pre, "
                       "* CI excludes 0, (n+) animals with a positive change)", 120), fontsize=10)
    save_png_svg(fig, FIGS / "lick_correction_spouts.png", dpi=110)
    plt.close(fig)
    # ---------- figure 3: error signal
    BG = FN["BG"].rename(columns={})
    GL = FN["GL"]
    fig, axs = plt.subplots(3, 4, figsize=(17, 12.5), layout="constrained")
    hl = {"contra hemisphere": ("|c", "#b2182b"), "ipsi hemisphere": ("|i", "#2166ac")}
    for r, tau in enumerate(TAUS):
        for c, w in enumerate(WINS):
            ln = {k: ({"feat": f"POOL{v[0]}"}, v[1])
                  for k, v in {**hl, "contra - ipsi (+ = contra larger)": ("|c-i", "k")}.items()}
            _epoch_panel(axs[r, c], BG[(BG.tau == tau) & (BG.win == w)], GL[(GL.tau == tau) & (GL.win == w)]
                         .rename(columns={"gamma": "v"}), "v", {}, ln,
                         f"3-{r * 4 + c + 1}. {'raw dF/F' if not tau else f'deconvolved tau {tau}'}: MISS - HIT "
                         f"after lick k, window {w.split('_')[1]} s, orofacial pool", "gamma (dF/F x1e3)")
    w = "w1_0.1-0.2"
    for c, ep in enumerate(EPS):
        ax = axs[2, c]
        b = BG[(BG.tau == 0) & (BG.win == w) & (BG.epoch == ep)]
        names = AREAS + ["RSPd", "VISp"]
        for j, (hn, (sfx, col)) in enumerate(hl.items()):
            bb = b.set_index("feat").reindex([f"{a}{sfx}" for a in names])
            xx = np.arange(len(names)) + (j - 0.5) * 0.35
            ax.bar(xx, bb.level, width=0.33, color=col, alpha=0.7, label=hn)
            ax.errorbar(xx, bb.level, yerr=[bb.level - bb.level_lo, bb.level_hi - bb.level], fmt="none", ecolor=col,
                        lw=0.8, capsize=1.5)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks(range(len(names)), names, fontsize=8, rotation=45)
        ax.grid(axis="y", color="0.92", lw=0.5)
        ax.set_title(_wrap(f"3-{9 + c}. {ep}: MISS - HIT per area, window 0.1-0.2 s, raw", 48), fontsize=8)
        ax.tick_params(labelsize=7)
    hh, ll = axs[0, 0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="outside lower center", ncol=3, fontsize=8, frameon=False,
               title="rows 1-2 lines; row 3 bars use the same hemisphere colors", title_fontsize=8)
    fig.suptitle(_wrap("Fig 3. ERROR SIGNAL -- activity after a MISSED lick k minus a HIT at matched angles of BOTH "
                       "lick k and k+1 (plus reach, k+1 contact, ILI, order, spout, pre-lick level; per-session OLS "
                       "coefficient). Pool = ALM + tjM1 + SSp-m + SSp-n + SSs per hemisphere. Bars = pooled session "
                       "level with 95% nested bootstrap CI; * = differs from pre.", 170), fontsize=9)
    save_png_svg(fig, FIGS / "lick_correction_error_signal.png", dpi=110)
    plt.close(fig)
    # ---------- figure 4: prediction
    BD, DL = FN["BD"], FN["DL"]
    fig, axs = plt.subplots(3, 4, figsize=(18, 13), layout="constrained")
    scol = {"both hemispheres": ("both", "k"), "contra": ("contra", "#b2182b"), "ipsi": ("ipsi", "#2166ac"),
            "control RSPd/VISp": ("control", "0.6")}
    for r, w in enumerate(PRED_WINS):
        for c, sub in enumerate(["all", "miss", "hit", "miss-hit"]):
            ln = {k: ({"set": v[0]}, v[1]) for k, v in scol.items()}
            _epoch_panel(axs[r, c], BD[(BD.tau == 0) & (BD.win == w) & (BD.subset == sub)],
                         DL[(DL.tau == 0) & (DL.win == w) & (DL.subset == sub)].rename(columns={"dR2": "v"}), "v",
                         {}, ln, f"4-{r * 4 + c + 1}. window {w.split('_')[1]} s rel. lick k: dcvR2 of the "
                                 f"correction, {sub} pairs", "dcvR2 (neural added)")
    hh, ll = axs[0, 0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(_wrap("Fig 4. Does activity around lick k predict the next lick's correction beyond lick k's angle, "
                       "error and contact? Cross-validated within session (5-fold by trial); neural = area level "
                       "change residualised on lick k's angle / reach / spout; dcvR2 = gain over the behavioral "
                       "baseline. Windows after lick k already contain lick k+1's command (calcium lag), so HIT pairs "
                       "measure next-command information; MISS - HIT is the error-specific part.", 170), fontsize=9)
    save_png_svg(fig, FIGS / "lick_correction_prediction.png", dpi=110)
    plt.close(fig)
    print("figures written", flush=True)


def main() -> int:
    if "--extract" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
        run_extract(min(jobs, 4))
        return 0
    from figdata import load_fd, plot_only, save_fd
    if plot_only():
        FB, FN = load_fd("lick_correction_behaviour"), load_fd("lick_correction_neural")
    else:
        from wfield_local import writeguard
        writeguard.assert_writable(OUT)
        t0 = time.time()
        if "--neural-only" in sys.argv:
            FB = load_fd("lick_correction_behaviour")
        else:
            FB = behaviour()
            save_fd("lick_correction_behaviour", FB)
        print(f"behavior done {(time.time() - t0) / 60:.1f} min", flush=True)
        if "--behaviour-only" in sys.argv:
            FN = load_fd("lick_correction_neural")
        else:
            FN = neural(FB["sgn"])
            save_fd("lick_correction_neural", FN)
        print(f"neural done {(time.time() - t0) / 60:.1f} min", flush=True)
    plot(FB, FN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
