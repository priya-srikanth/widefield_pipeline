"""KINEMATIC NORMALITY -- restitution vs compensation (analysis 2.2, docs/RECOVERY_MECHANISMS_LIT_20261030.md; SRRR /
Levin, Kleim & Wolf 2009: recovery = the pre-stroke movement pattern returns; compensation = the task is solved again with
a different pattern). PRE-REGISTRATION (fixed before any post-stroke number): docs/status/KINEMATIC_NORMALITY_PREREG_20261030.md.

Per engaged cued trial (PS92-95, curated sessions, August-only pre), the FIRST post-cue lick's 3-D kinematics + trial
timing + jaw are scored as a squared Mahalanobis distance d2 from the SAME animal's pre-stroke trials at the SAME spout
(features z-scored on pre, Ledoit-Wolf covariance; pre trials scored leave-one-SESSION-out so pre has an honest level).
NORMALITY = fraction of trials inside the pre 95 % contour (d2 <= chi2_0.95(k)). Primary trial set = HITS.
Secondary: attempted trials (hit + incomplete), the trial's lick-SEQUENCE vector, a leave-one-session-out pre-vs-post
logistic classifier (normality = mean P(pre)).

Per session series: engaged hit rate (all / far contra), far-contra aim (read from aim_by_spout_sessions.csv), normality,
and EXISTING cortical readouts read from their saved session tables (never refitted here):
  C1 bal      cued first-lick orofacial balance, PRIMARY recovery marker (page 14.17): iti_licks/recprim_sessions.csv
              panel 'cued first lick -> aim', column marker (+ = contralesional hemisphere larger)
  C2 iti_off  ITI decoder offset, same-place like-for-like, docked-spout ITI: iti_licks/itidock_main/matched_gauss_sessions.csv
              (all spouts, cls all, set iti, resid mm; + = read contralesional of the landing)
  C3 ret_hit  frozen-decoder far-contra retention on HITS, lick window: attempt_coding/decoder_retention_sessions.csv
  C4 tjm1     first cued lick response AUC 0-1 s, contralesional tjM1: iti_licks/area_auc_sessions_split_cued.csv (resp)
Each series per animal on a FRACTION-OF-ACUTE-DEFICIT scale f = (x - acute) / (pre - acute) (pre = 1, acute = 0);
exponential fit y = a - (a - b) exp(-d / tau) (as recovery_timing.exp_fit) -> half-recovery day tau ln2 and asymptote a,
95 % CIs by resampling post sessions (1000 draws, the SAME draws for every series -> paired differences).
Comparison: within-animal r of each cortical series with hit rate vs with normality, log(day) partialled (+ partials on
each other), pooled Fisher z, within-animal permutation p; epoch changes by analysis_kit.boot_delta.

SIGNS: lick angle + = contralesional (mouse right); aim + = contralesional, - = ipsilesional deficit; jaw lateral
lr_wide_mm + = ipsilesional (mouse left); spouts L = ipsi, R = contra (all lesions LEFT VLS).

Usage:  python scripts/kinematic_normality.py            (run + summarise + plot)
        python scripts/kinematic_normality.py --plot-only (redraw from figdata)
Outputs: <E>/kinematic_normality/{trials.csv.gz, sessions.csv, coverage.csv, per_animal_epoch.csv, epoch_tests.csv,
         fits.csv, fit_diffs.csv, corr_animals.csv, corr_pooled.csv}, figures/*.png, figdata/*.pkl
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
from wfield_local import run_mode as _rm  # noqa: E402
OUT = _rm.subdir(E / "kinematic_normality", lp_family=True)   # mode-routed (LP family; default LP folder kept)
FIG = OUT / "figures"
FD = OUT / "figdata"
P3 = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics_all")
AN = ["PS92", "PS93", "PS94", "PS95"]
EP4 = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]     # canonical order
P3MAP = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra", "far_L": "far_ipsi",
         "far_center": "far_mid", "far_R": "far_contra"}
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}

# --- pre-registered feature sets (section 3) ---
CORE = ["angle", "sagittal", "reach_mm", "log_speed", "lat_err_mm", "log_miss", "log_lat"]
FEATS = {p: CORE + (["jaw_lead"] if p != "close_mid" else []) + (["lr_wide_mm"] if p not in ("close_mid", "far_mid")
                                                                   else []) for p in POS}
SEQ = ["log_contact", "n_before_contact", "log_nlicks", "angle3", "reach3"]
MIN_PRE_TRIALS, MIN_PRE_SESS, MIN_CELL = 30, 3, 5
N_FIT_BOOT, N_PERM, N_DIFF_BOOT = 1000, 5000, 2000

# --- per-session series (section 5) ---
BEH = [("hit", "hit rate (engaged, all spouts)"), ("hit_fc", "hit rate, far contra"),
       ("aim_fc", "far-contra aim (deg vs pre)"), ("norm", "kinematic normality (hits, all spouts)"),
       ("norm_fc", "kinematic normality (hits, far contra)")]
SEC = [("d2", "median d2/k (hits)"), ("norm_att", "normality, attempted trials"),
       ("norm_seq", "sequence normality (hits)"), ("norm_clf", "classifier P(pre) (hits)")]
CORT = [("bal", "C1 cued orofacial balance (14.17)"), ("iti_off", "C2 ITI decoder offset (docked)"),
        ("ret_hit", "C3 frozen far-contra code on hits"), ("tjm1", "C4 first-cued-lick response, contra tjM1")]
SRC = {"aim_fc": "iti_licks/aim_by_spout_sessions.csv [pos far_contra, chg]",
       "bal": "iti_licks/recprim_sessions.csv [panel 'cued first lick -> aim', marker]",
       "iti_off": "iti_licks/itidock_main/matched_gauss_sessions.csv [all spouts / all / iti, resid]",
       "ret_hit": "attempt_coding/decoder_retention_sessions.csv [window lick, cls2 hit]",
       "tjm1": "iti_licks/area_auc_sessions_split_cued.csv [area tjM1_right, resp]"}
ALL_SERIES = BEH + SEC + CORT
LAB = dict(ALL_SERIES)


def _w(path: Path) -> Path:
    from wfield_local.writeguard import assert_writable
    assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================================ data
def sessions() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    rows = []
    for lab in ak.curated_labels(order="sorted"):
        an = lab[:4]
        if an not in AN:                                   # never cohort 2
            continue
        ep = epochs.epoch_of(lab)
        if ep is None or (ep == "pre" and lab[-4:] < "0801"):     # August-only pre
            continue
        rows.append({"label": lab, "animal": an, "epoch": ep, "day": epochs.days_since_stroke(lab)})
    return pd.DataFrame(rows)


def load_trials(S: pd.DataFrame) -> pd.DataFrame:
    """One row per engaged (canonical) trial with first-lick + sequence + jaw features."""
    T = pd.read_csv(E / "qc" / "trial_attempts.csv", low_memory=False)
    T = T[T.label.isin(S.label) & T.canonical.astype(bool)].copy()
    T = T.drop(columns=["epoch", "day"]).merge(S, on=["label", "animal"], how="left")
    K = pd.read_csv(P3 / "licks.csv.gz", low_memory=False,
                    usecols=["trial_id", "lick_idx", "position", "context", "contact", "reach_mm", "side_ml",
                             "sagittal", "lat_err_mm", "miss_mm", "speed_peak", "session"])
    K = K[K.context == "post_cue"].copy()
    K["label"] = K.session.str[:5] + K.session.str[-4:]
    K = K[K.label.isin(S.label) & np.isfinite(K.reach_mm)].sort_values(["label", "trial_id", "lick_idx"])
    K["angle"] = np.degrees(np.arcsin(np.clip(K.side_ml / K.reach_mm, -1, 1)))     # + = contralesional
    F = K.groupby(["label", "trial_id"], sort=False).first().reset_index()
    F["pos3"] = F.position.map(P3MAP)
    K["k"] = K.groupby(["label", "trial_id"]).cumcount()
    K3 = K[K.k < 3].groupby(["label", "trial_id"])[["angle", "reach_mm"]].mean().rename(
        columns={"angle": "angle3", "reach_mm": "reach3"})
    cont = K.assign(c=K.contact.astype(str).str.lower().eq("true")).groupby(["label", "trial_id"])
    nb = cont.apply(lambda g: float(np.argmax(g.c.to_numpy())) if g.c.any() else np.nan, include_groups=False)
    F = F.merge(K3.reset_index(), on=["label", "trial_id"], how="left").merge(
        nb.rename("n_before_contact").reset_index(), on=["label", "trial_id"], how="left")
    T = T.merge(F[["label", "trial_id", "pos3", "lick_idx", "angle", "sagittal", "reach_mm", "speed_peak", "lat_err_mm",
                   "miss_mm", "angle3", "reach3", "n_before_contact"]], on=["label", "trial_id"], how="left")
    bad = T.pos3.notna() & (T.pos3 != T.pos)
    if bad.any():
        raise SystemExit(f"join check failed: 3-D position != trial pos on {int(bad.sum())} trials")
    J = pd.read_csv(P3 / "jaw_trials.csv.gz")
    J["label"] = J.session.str[:5] + J.session.str[-4:]
    T = T.merge(J[["label", "trial_id", "lr_wide_mm"]], on=["label", "trial_id"], how="left")
    T["log_speed"] = np.log(T.speed_peak.where(T.speed_peak > 0))
    T["log_miss"] = np.log(T.miss_mm + 0.05)
    T["log_lat"] = np.log(T.cam4.where(T.cam4 > 0))
    T["jaw_lead"] = T.cam4 - T.jaw_onset_ms
    T["log_contact"] = np.log(T.contact.where(T.contact > 0))
    T["log_nlicks"] = np.log(T.n_licks.where(T.n_licks > 0))
    T["is_hit"] = T.cls.eq("hit")
    T["is_att"] = T.cls.isin(["hit", "incomplete"])
    return T.sort_values(["animal", "label", "trial_id"]).reset_index(drop=True)


# ============================================================================================ scoring
def _fit_ref(X: np.ndarray):
    from sklearn.covariance import LedoitWolf
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    lw = LedoitWolf().fit((X - mu) / sd)
    return mu, sd, lw.precision_


def _d2(X, ref):
    mu, sd, P = ref
    Z = (X - mu) / sd
    return np.einsum("ij,jk,ik->i", Z, P, Z)


def mahal(T: pd.DataFrame, trialset: str, feats_of) -> pd.Series:
    """d2 / k per trial (NaN = not scored) for trial set 'hit' | 'att', features feats_of(pos)."""
    from scipy.stats import chi2
    sel = T.is_hit if trialset == "hit" else T.is_att
    d2 = pd.Series(np.nan, index=T.index)
    inside = pd.Series(np.nan, index=T.index)
    for (_an, pos), g in T[sel].groupby(["animal", "pos"]):
        f = feats_of(pos)
        g = g.dropna(subset=f)
        pre = g[g.epoch == "pre"]
        if len(pre) < MIN_PRE_TRIALS or pre.label.nunique() < MIN_PRE_SESS:
            continue
        thr = chi2.ppf(0.95, len(f))
        Xp = pre[f].to_numpy(float)
        for lab in pre.label.unique():                      # LOSO for pre
            m = (pre.label == lab).to_numpy()
            v = _d2(Xp[m], _fit_ref(Xp[~m]))
            d2.loc[pre.index[m]], inside.loc[pre.index[m]] = v / len(f), (v <= thr).astype(float)
        post = g[g.epoch != "pre"]
        if len(post):
            v = _d2(post[f].to_numpy(float), _fit_ref(Xp))
            d2.loc[post.index], inside.loc[post.index] = v / len(f), (v <= thr).astype(float)
    return d2, inside


def classifier(T: pd.DataFrame) -> pd.Series:
    """Out-of-fold P(pre) per hit trial, leave-one-session-out over the animal's pre + post sessions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = pd.Series(np.nan, index=T.index)
    for (_an, pos), g in T[T.is_hit].groupby(["animal", "pos"]):
        f = FEATS[pos]
        g = g.dropna(subset=f)
        if (g.epoch == "pre").sum() < MIN_PRE_TRIALS or (g.epoch != "pre").sum() < MIN_PRE_TRIALS:
            continue
        X, y = g[f].to_numpy(float), (g.epoch != "pre").to_numpy(int)
        for lab in g.label.unique():
            m = (g.label == lab).to_numpy()
            if len(np.unique(y[~m])) < 2:
                continue
            sc = StandardScaler().fit(X[~m])
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(X[~m]), y[~m])
            out.loc[g.index[m]] = clf.predict_proba(sc.transform(X[m]))[:, 0]
    return out


def score(T: pd.DataFrame) -> pd.DataFrame:
    T = T.copy()
    T["d2k"], T["inside"] = mahal(T, "hit", lambda p: FEATS[p])
    T["d2k_att"], T["inside_att"] = mahal(T, "att", lambda p: FEATS[p])
    T["d2k_seq"], T["inside_seq"] = mahal(T, "hit", lambda p: SEQ)
    T["p_pre"] = classifier(T)
    return T


def coverage(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (an, ep, pos), g in T.groupby(["animal", "epoch", "pos"]):
        h = g[g.is_hit]
        a = g[g.is_att]
        rows.append({"animal": an, "epoch": ep, "pos": pos, "n_engaged": len(g), "n_hits": len(h),
                     "cov_hits_first": h[FEATS[pos]].notna().all(1).mean() if len(h) else np.nan,
                     "scored_hits": h.inside.notna().mean() if len(h) else np.nan,
                     "scored_att": a.inside_att.notna().mean() if len(a) else np.nan,
                     "scored_seq": h.inside_seq.notna().mean() if len(h) else np.nan,
                     **{f"cov_{f}": h[f].notna().mean() if len(h) else np.nan for f in CORE + ["jaw_lead", "lr_wide_mm"]}})
    C = pd.DataFrame(rows)
    C["pos"] = pd.Categorical(C.pos, POS, ordered=True)
    C["epoch"] = pd.Categorical(C.epoch, EP4, ordered=True)
    return C.sort_values(["animal", "pos", "epoch"])


# ============================================================================================ session series
def _frac(g: pd.DataFrame, col: str, pos=None) -> float:
    g = g[g[col].notna()]
    if pos is not None:
        g = g[g.pos == pos]
        return float(g[col].mean()) if len(g) >= MIN_CELL else np.nan
    n = g.groupby("pos")[col].transform("size")
    g = g[n >= MIN_CELL]
    return float(g[col].mean()) if len(g) else np.nan


def session_table(T: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, g in T.groupby("label"):
        fc = g[g.pos == "far_contra"]
        h = g[g.is_hit]
        d2 = h[h.d2k.notna()]
        d2 = d2[d2.groupby("pos").d2k.transform("size") >= MIN_CELL]
        rows.append({"label": lab, "hit": g.is_hit.mean(), "hit_fc": fc.is_hit.mean() if len(fc) else np.nan,
                     "norm": _frac(g, "inside"), "norm_fc": _frac(g, "inside", "far_contra"),
                     "d2": float(d2.d2k.median()) if len(d2) else np.nan, "norm_att": _frac(g, "inside_att"),
                     "norm_seq": _frac(g, "inside_seq"), "norm_clf": _frac(g, "p_pre"),
                     "n_engaged": len(g), "n_hits_scored": int(g.inside.notna().sum()),
                     "n_fc_hits_scored": int(fc.inside.notna().sum())})
    D = S.merge(pd.DataFrame(rows), on="label", how="left")
    # existing tables (read, never refitted)
    import iti_licks as _il                                  # the WRITER's root (lick_cued_aim_by_spout -> il.OUT; T6)
    G = pd.read_csv(_il.OUT / "aim_by_spout_sessions.csv")
    D = D.merge(G[G.pos == "far_contra"][["label", "chg"]].rename(columns={"chg": "aim_fc"}), on="label", how="left")
    R = pd.read_csv(E / "iti_licks" / "recprim_sessions.csv")
    D = D.merge(R[R.panel == "cued first lick -> aim"][["label", "marker"]].rename(columns={"marker": "bal"}),
                on="label", how="left")
    M = pd.read_csv(E / "iti_licks" / "itidock_main" / "matched_gauss_sessions.csv")
    M = M[(M.scope == "all spouts") & (M.cls == "all") & (M.set == "iti")]
    D = D.merge(M[["label", "resid"]].rename(columns={"resid": "iti_off"}), on="label", how="left")
    from wfield_local import attempt_coding as _ac
    Rt = pd.read_csv(_ac.out_dir() / "decoder_retention_sessions.csv")   # the writer's folder (review N1)
    Rt = Rt[(Rt.window == "lick") & (Rt.cls2 == "hit")]
    D = D.merge(Rt[["label", "ret"]].rename(columns={"ret": "ret_hit"}), on="label", how="left")
    A = pd.read_csv(E / "iti_licks" / "area_auc_sessions_split_cued.csv")
    D = D.merge(A[A.area == "tjM1_right"][["label", "resp"]].rename(columns={"resp": "tjm1"}), on="label", how="left")
    return D.sort_values(["animal", "day"]).reset_index(drop=True)


def normalise(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """f = (x - acute) / (pre - acute) per animal x series; flags."""
    N = D[["label", "animal", "epoch", "day"]].copy()
    ref = []
    for key, _l in ALL_SERIES:
        N[key] = np.nan
        for an in AN:
            g = D[D.animal == an]
            pre = g[g.epoch == "pre"][key].dropna()
            ac = g[g.epoch == "acute"][key].dropna()
            ac_src = "acute epoch"
            if not len(ac):
                post = g[(g.epoch != "pre")].dropna(subset=[key])
                ac = post[key].iloc[:1]
                ac_src = f"earliest post (day {post.day.iloc[0]:.0f})" if len(post) else "none"
            if not len(pre) or not len(ac):
                ref.append({"animal": an, "series": key, "pre": np.nan, "acute": np.nan, "pre_sd": np.nan,
                            "acute_src": ac_src, "deficit": False})
                continue
            p, a, sd = pre.mean(), ac.mean(), pre.std(ddof=1)
            ok = abs(p - a) >= (sd if np.isfinite(sd) else 0)
            ref.append({"animal": an, "series": key, "pre": p, "acute": a, "pre_sd": sd, "acute_src": ac_src,
                        "deficit": bool(ok)})
            m = D.animal == an
            N.loc[m, key] = (D.loc[m, key] - a) / (p - a) if p != a else np.nan
    return N, pd.DataFrame(ref)


# ============================================================================================ fits
def _expf(x, a, b, tau):
    return a - (a - b) * np.exp(-x / tau)


def exp_fit(d, y):
    """As recovery_timing.exp_fit (same bounds / start), returning (t_half, a)."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(y)
    d, y = d[ok], y[ok]
    if len(d) < 4:
        return np.nan, np.nan
    try:
        (a, b, tau), _ = curve_fit(_expf, d, y, p0=(np.nanmax(y), np.nanmin(y), 5.0),
                                   bounds=([-2, -2, 0.3], [3, 3, 200]), maxfev=20000)
        return tau * np.log(2), a
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def fit_all(N: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    keys = [k for k, _ in ALL_SERIES if k != "aim_fc"] + ["aim_fc"]
    rng = np.random.default_rng(0)
    rows, drows, curves = [], [], {}
    for an in AN:
        g = N[(N.animal == an) & (N.epoch != "pre") & (N.day > 0)].sort_values("day").reset_index(drop=True)
        d = g.day.to_numpy(float)
        Y = {k: g[k].to_numpy(float) for k in keys}
        pt = {k: exp_fit(d, Y[k]) for k in keys}
        bs = {k: [] for k in keys}
        for _ in range(N_FIT_BOOT):
            i = rng.integers(0, len(g), len(g))
            for k in keys:
                bs[k].append(exp_fit(d[i], Y[k][i]))
        bs = {k: np.array(v) for k, v in bs.items()}

        def q(x):
            return np.nanpercentile(x, [2.5, 97.5]) if np.isfinite(x).sum() > 50 else (np.nan, np.nan)
        for k in keys:
            r = ref[(ref.animal == an) & (ref.series == k)]
            th, a = pt[k]
            rows.append({"animal": an, "series": k, "deficit": bool(r.deficit.iloc[0]) if len(r) else False,
                         "acute_src": r.acute_src.iloc[0] if len(r) else "", "n_post": int(np.isfinite(Y[k]).sum()),
                         "t_half": th, "t_lo": q(bs[k][:, 0])[0], "t_hi": q(bs[k][:, 0])[1],
                         "asym": a, "a_lo": q(bs[k][:, 1])[0], "a_hi": q(bs[k][:, 1])[1],
                         "fit_ok_share": float(np.isfinite(bs[k][:, 0]).mean())})
            if np.isfinite(th):
                from scipy.optimize import curve_fit
                ok = np.isfinite(Y[k])
                try:
                    par, _ = curve_fit(_expf, d[ok], Y[k][ok], p0=(np.nanmax(Y[k]), np.nanmin(Y[k]), 5.0),
                                       bounds=([-2, -2, 0.3], [3, 3, 200]), maxfev=20000)
                    curves[(an, k)] = par
                except (RuntimeError, ValueError):
                    pass
        for k in keys:
            for base in ("hit", "norm", "hit_fc", "norm_fc"):
                if k == base:
                    continue
                diff = bs[k][:, 0] - bs[base][:, 0]
                ok = np.isfinite(diff)
                lo, hi = (np.percentile(diff[ok], [2.5, 97.5]) if ok.sum() > 50 else (np.nan, np.nan))
                drows.append({"animal": an, "series": k, "minus": base, "diff": pt[k][0] - pt[base][0], "lo": lo,
                              "hi": hi, "share_later": float((diff[ok] > 0).mean()) if ok.any() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(drows), curves


# ============================================================================================ correlations
def _res(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b


def _r(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _corr_set(c, h, n, ld):
    """r_hit, r_norm (log day partialled) and the two partials, from raw arrays."""
    cr, hr, nr = _res(c, [ld]), _res(h, [ld]), _res(n, [ld])
    return (_r(cr, hr), _r(cr, nr), _r(_res(cr, [nr]), _res(hr, [nr])), _r(_res(cr, [hr]), _res(nr, [hr])))


def correlations(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    A, P = [], []
    pairs = [("primary", "hit", "norm"), ("secondary", "hit_fc", "norm_fc")]
    for pair, hk, nk in pairs:
        for ck, _l in CORT:
            per = {}
            for an in AN:
                g = D[(D.animal == an) & (D.epoch != "pre") & (D.day > 0)].dropna(subset=[ck, hk, nk])
                if len(g) < 5:
                    continue
                c, h, n, ld = (g[ck].to_numpy(float), g[hk].to_numpy(float), g[nk].to_numpy(float),
                               np.log(g.day.to_numpy(float)))
                per[an] = (c, h, n, ld)
                rh, rn, rh_n, rn_h = _corr_set(c, h, n, ld)
                A.append({"pair": pair, "cortex": ck, "animal": an, "n": len(g), "r_hit": rh, "r_norm": rn,
                          "r_hit|norm": rh_n, "r_norm|hit": rn_h,
                          "hit_wins": abs(rh_n) > abs(rn_h) if np.isfinite(rh_n) and np.isfinite(rn_h) else np.nan})
            if not per:
                continue
            ans = sorted(per)

            def pooled(st, ans=ans):
                z = np.array([np.arctanh(np.clip(st[a], -0.999, 0.999)) for a in ans])
                return np.nanmean(z, 0)
            obs = pooled({a: np.array(_corr_set(*per[a])) for a in ans})
            null = []
            for _ in range(N_PERM):
                st = {}
                for a in ans:
                    c, h, n, ld = per[a]
                    st[a] = np.array(_corr_set(rng.permutation(_res(c, [ld])), h, n, ld))
                null.append(pooled(st))
            null = np.array(null)
            pv = [(np.abs(null[:, j]) >= abs(obs[j]) - 1e-12).mean() for j in range(4)]
            bd = []
            for _ in range(N_DIFF_BOOT):
                st = {}
                for a in ans:
                    c, h, n, ld = per[a]
                    i = rng.integers(0, len(c), len(c))
                    st[a] = np.array(_corr_set(c[i], h[i], n[i], ld[i]))
                z = pooled(st)
                bd.append((z[0] - z[1], abs(z[2]) - abs(z[3])))
            bd = np.array(bd)
            sub = pd.DataFrame([x for x in A if x["pair"] == pair and x["cortex"] == ck])
            P.append({"pair": pair, "cortex": ck, "n_animals": len(ans),
                      "r_hit": np.tanh(obs[0]), "p_hit": pv[0], "r_norm": np.tanh(obs[1]), "p_norm": pv[1],
                      "r_hit|norm": np.tanh(obs[2]), "p_hit|norm": pv[2], "r_norm|hit": np.tanh(obs[3]),
                      "p_norm|hit": pv[3],
                      "zdiff_hit_minus_norm": obs[0] - obs[1],
                      "zdiff_lo": np.nanpercentile(bd[:, 0], 2.5), "zdiff_hi": np.nanpercentile(bd[:, 0], 97.5),
                      "abs_partial_diff": abs(obs[2]) - abs(obs[3]),
                      "apd_lo": np.nanpercentile(bd[:, 1], 2.5), "apd_hi": np.nanpercentile(bd[:, 1], 97.5),
                      "animals_hit_wins": int(sub.hit_wins.eq(True).sum()),
                      "animals_norm_wins": int(sub.hit_wins.eq(False).sum())})
    Pd = pd.DataFrame(P)

    def verdict(r):
        if r.animals_hit_wins >= 3 and r.p_hit < 0.05:
            return "follows SUCCESS"
        if r.animals_norm_wins >= 3 and r.p_norm < 0.05:
            return "follows NORMALITY"
        return "undetermined"
    if len(Pd):
        Pd["verdict"] = Pd.apply(verdict, axis=1)
    return pd.DataFrame(A), Pd


# ============================================================================================ epoch stats
def epoch_stats(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    keys = ["norm", "norm_fc", "d2", "norm_att", "norm_seq", "norm_clf", "hit", "hit_fc"]
    PA = D.groupby(["animal", "epoch"])[keys].mean().reset_index()
    PA["epoch"] = pd.Categorical(PA.epoch, EP4, ordered=True)
    PA = PA.sort_values(["animal", "epoch"])
    rows = []
    for k in keys:
        rng = np.random.default_rng(0)
        pre = {an: D[(D.animal == an) & (D.epoch == "pre")][k].dropna().tolist() for an in AN}
        pre = {a: v for a, v in pre.items() if v}
        for ep in EP4[1:]:
            post = {an: D[(D.animal == an) & (D.epoch == ep)][k].dropna().tolist() for an in AN}
            post = {a: v for a, v in post.items() if v}
            iv = ak.boot_delta(post, pre, rng)
            com = sorted(set(post) & set(pre))
            dif = {a: np.mean(post[a]) - np.mean(pre[a]) for a in com}
            same = max(sum(v > 0 for v in dif.values()), sum(v < 0 for v in dif.values()))
            rows.append({"measure": k, "epoch": ep, "delta": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan,
                         "hi": iv.hi if iv else np.nan, "n_animals": len(com), "same_sign": same,
                         **{f"d_{a}": dif.get(a, np.nan) for a in AN},
                         "change": bool(iv is not None and iv.excludes_zero and same >= 3)})
    return PA, pd.DataFrame(rows)


# ============================================================================================ figures
def _star_top(ax, x, text="*", row=0, step=0.07):
    try:
        from wfield_local.sig_marks import star_top
        return star_top(ax, x, text=text, row=row)
    except ImportError:
        import matplotlib.patheffects as pe
        ax._wf_mark_rows = max(getattr(ax, "_wf_mark_rows", 0), row + 1)
        return ax.text(x, 0.99 - step * row, text, ha="center", va="top", fontsize=12, fontweight="bold",
                       transform=ax.get_xaxis_transform(), clip_on=False, zorder=6,
                       path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])


def _reserve(fig, step=0.07, pad=0.03):
    try:
        from wfield_local.sig_marks import reserve_headroom
        return reserve_headroom(fig)
    except ImportError:
        for ax in fig.get_axes():
            n = getattr(ax, "_wf_mark_rows", 0)
            if n:
                lo, hi = ax.get_ylim()
                frac = step * n + pad
                ax.set_ylim(lo, hi + (hi - lo) * frac / (1 - frac))


def _clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


BCOL = {"hit": "0.15", "hit_fc": "0.55", "aim_fc": "#2471a3", "norm": "#c0392b", "norm_fc": "#e67e22"}
CCOL = {"bal": "#8e44ad", "iti_off": "#16a085", "ret_hit": "#b03a2e", "tjm1": "#7f8c8d"}


def draw(F: dict) -> None:
    from wfield_local.orofacial_cohort import EPOCH_LINE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _w(FIG / "x")
    N, fits, curves = F["N"], F["fits"], F["curves"]
    YL = (-1.6, 2.4)

    # ---------- Fig 1: per animal day courses, normalised (2 rows: behaviour, cortex) ----------
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.4), sharex=True, sharey=True)
    for j, an in enumerate(AN):
        g = N[N.animal == an].sort_values("day")
        post = g[g.epoch != "pre"]
        for row, ser in ((0, BEH), (1, CORT)):
            ax = axs[row, j]
            ax.axhline(1, color="0.6", ls="--", lw=0.8)
            ax.axhline(0, color="0.8", ls=":", lw=0.8)
            ax.axvspan(-14, 0, color="0.94", lw=0)
            pre = g[g.epoch == "pre"]
            if row == 1:            # reference fits of hit and normality in grey
                for k, ls in (("hit", "-"), ("norm", "--")):
                    if (an, k) in curves:
                        xx = np.linspace(1, post.day.max(), 200)
                        ax.plot(xx, np.clip(_expf(xx, *curves[(an, k)]), *YL), color="0.35", ls=ls, lw=1.6, zorder=2)
            for k, _l in ser:
                col = BCOL.get(k) or CCOL.get(k)
                fr = fits[(fits.animal == an) & (fits.series == k)]
                nodef = len(fr) and not bool(fr.deficit.iloc[0])
                ax.plot(post.day, post[k].clip(*YL), "o", color=col, ms=3.6, alpha=0.45 if nodef else 0.85,
                        mfc="none" if nodef else col, zorder=3)
                ax.plot(np.full(len(pre), -6) + np.linspace(-3, 3, len(pre)), pre[k].clip(*YL), ".", color=col,
                        ms=3, alpha=0.5)
                if (an, k) in curves and not nodef:
                    xx = np.linspace(1, post.day.max(), 200)
                    ax.plot(xx, np.clip(_expf(xx, *curves[(an, k)]), *YL), color=col, lw=1.8, zorder=4)
            ax.set_ylim(*YL)
            _clean(ax)
            if row == 0:
                ax.set_title(an, fontsize=11)
            if row == 1:
                ax.set_xlabel("days from stroke (pre sessions at x = -6)")
            if j == 0:
                ax.set_ylabel(("BEHAVIOUR" if row == 0 else "CORTEX") + "\nfraction of acute deficit recovered\n"
                              "(pre = 1, acute = 0)")
    h0 = [Line2D([], [], color=BCOL[k], marker="o", lw=1.8, label=l) for k, l in BEH]
    h1 = ([Line2D([], [], color=CCOL[k], marker="o", lw=1.8, label=l) for k, l in CORT]
          + [Line2D([], [], color="0.35", lw=1.6, label="fit: hit rate (all spouts)"),
             Line2D([], [], color="0.35", lw=1.6, ls="--", label="fit: kinematic normality"),
             Line2D([], [], color="0.4", marker="o", mfc="none", ls="", label="open = no acute deficit (not fitted)")])
    axs[0, 3].legend(handles=h0, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axs[1, 3].legend(handles=h1, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Recovery per animal on a common scale (fraction of the acute deficit recovered): task success, "
                 "kinematic normality (hits; pre-stroke 95 % contour) and existing cortical readouts.\nLines = "
                 "exponential fits y = a - (a - b) exp(-d / tau); values clipped to the axis. n = 4, PROVISIONAL",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 0.86, 0.93))
    fig.savefig(_w(FIG / "kn_fig1_daycourse_per_animal.png"), dpi=120)
    plt.close(fig)

    # ---------- Fig 2: normality by epoch ----------
    D, ET = F["D"], F["epoch_tests"]
    panels = [("norm", "normality, HITS, all spouts\n(fraction inside the pre 95 % contour)"),
              ("norm_fc", "normality, HITS, far contra"), ("norm_att", "normality, ATTEMPTED trials"),
              ("norm_clf", "classifier P(pre), hits\n(leave-one-session-out)")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8))
    for k_, (m, lab) in enumerate(panels):
        ax = axs[k_]
        for i, ep in enumerate(EP4):
            g = D[D.epoch == ep]
            means = []
            for j, an in enumerate(AN):
                v = g[g.animal == an][m].dropna().to_numpy()
                if len(v):                       # Priya 10-30: shape = animal, colour = EPOCH (open markers)
                    xs = i + (j - 1.5) * 0.09 + np.random.default_rng(10 * i + j).uniform(-0.025, 0.025, len(v))
                    ax.scatter(xs, v, marker=MK[an], s=18, facecolors="none", edgecolors=EPOCH_LINE[ep], alpha=0.75,
                               lw=0.9, zorder=4)
                    means.append(v.mean())
            if means:
                ax.plot([i - 0.32, i + 0.32], [np.mean(means)] * 2, color=EPOCH_LINE[ep], lw=2.6, zorder=5)
            r = ET[(ET.measure == m) & (ET.epoch == ep)]
            if len(r) and bool(r.change.iloc[0]):
                _star_top(ax, i)
        ax.set_xticks(range(4))
        ax.set_xticklabels(EP4)
        ax.set_title(lab, fontsize=9.5)
        _clean(ax)
    axs[0].set_ylabel("session value")
    _reserve(fig)
    fig.legend(handles=[Line2D([], [], marker=MK[a], mfc="none", mec="0.3", ls="", label=a) for a in AN]
               + [Line2D([], [], color=EPOCH_LINE[e], lw=2.6, label=f"{e} (mean of animals)") for e in EP4],
               loc="lower center", ncol=8, fontsize=8.5, frameon=False)
    fig.suptitle("Kinematic normality by epoch (one dot per session; pre = leave-one-session-out). * (top band) = "
                 "boot_delta vs August pre, 95 % CI excludes 0 AND >= 3/4 animals same sign", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(_w(FIG / "kn_fig2_normality_by_epoch.png"), dpi=120)
    plt.close(fig)

    # ---------- Fig 3: half-times and asymptotes (dot + CI per animal, rows = series) ----------
    order = [k for k, _ in BEH] + [k for k, _ in CORT]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    for ax, (v, lo, hi, xl, xlim) in zip(axs, (("t_half", "t_lo", "t_hi", "half-recovery day (tau ln 2)", (0, 30)),
                                               ("asym", "a_lo", "a_hi", "asymptote a (fraction of acute deficit "
                                                "recovered)", (-1, 2.2)))):
        for i, k in enumerate(order):
            for j, an in enumerate(AN):
                r = fits[(fits.animal == an) & (fits.series == k)]
                if not len(r) or not np.isfinite(r[v].iloc[0]):
                    continue
                r = r.iloc[0]
                y = i + (j - 1.5) * 0.16
                ok = bool(r.deficit)
                ax.plot([np.clip(r[lo], *xlim), np.clip(r[hi], *xlim)], [y, y], color=ACOL[an], lw=1.2,
                        alpha=0.8 if ok else 0.3)
                ax.plot(np.clip(r[v], *xlim), y, MK[an], color=ACOL[an], mfc=ACOL[an] if ok else "none", ms=6)
        if v == "asym":
            ax.axvline(1, color="0.6", ls="--", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xl)
        _clean(ax)
        ax.axhline(len(BEH) - 0.5, color="0.75", lw=0.8)
    axs[0].set_yticks(range(len(order)))
    axs[0].set_yticklabels([LAB[k] for k in order], fontsize=8.5)
    axs[0].invert_yaxis()
    axs[1].legend(handles=[Line2D([], [], marker=MK[a], color=ACOL[a], label=a) for a in AN]
                  + [Line2D([], [], marker="o", color="0.4", mfc="none", ls="", label="no acute deficit (flag)")],
                  fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Per-animal exponential fits: half-recovery day and asymptote, 95 % CI by resampling post "
                 "sessions (values beyond the axis are clipped)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 0.9, 0.94))
    fig.savefig(_w(FIG / "kn_fig3_halftimes_asymptotes.png"), dpi=120)
    plt.close(fig)

    # ---------- Fig 4: success vs normality partial correlations ----------
    CA, CP = F["corr_animals"], F["corr_pooled"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, pair in zip(axs, ("primary", "secondary")):
        for i, (ck, _l) in enumerate(CORT):
            for j, an in enumerate(AN):
                r = CA[(CA.pair == pair) & (CA.cortex == ck) & (CA.animal == an)]
                if not len(r):
                    continue
                y = i + (j - 1.5) * 0.12
                ax.plot(r["r_hit|norm"], y, MK[an], color="0.15", ms=6, mfc="0.15")
                ax.plot(r["r_norm|hit"], y, MK[an], color="#c0392b", ms=6, mfc="none")
            p = CP[(CP.pair == pair) & (CP.cortex == ck)]
            if len(p):
                p = p.iloc[0]
                ax.text(1.05, i, f"{p.verdict}\nhit|norm {p['r_hit|norm']:+.2f} (p {p['p_hit|norm']:.3f})\n"
                                 f"norm|hit {p['r_norm|hit']:+.2f} (p {p['p_norm|hit']:.3f})",
                        fontsize=7.5, va="center", transform=ax.get_yaxis_transform())
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title({"primary": "PRIMARY: all-spout hit rate vs all-spout normality",
                      "secondary": "SECONDARY: far-contra hit rate vs far-contra normality"}[pair], fontsize=9.5)
        ax.set_xlabel("partial r with the cortical series (log day partialled)")
        _clean(ax)
    axs[0].set_yticks(range(len(CORT)))
    axs[0].set_yticklabels([l for _k, l in CORT], fontsize=8.5)
    axs[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], marker="o", color="0.15", ls="", label="r(cortex, hit rate | normality)"),
                        Line2D([], [], marker="o", color="#c0392b", mfc="none", ls="",
                               label="r(cortex, normality | hit rate)")]
               + [Line2D([], [], marker=MK[a], color="0.5", ls="", label=a) for a in AN],
               loc="lower center", ncol=6, fontsize=8.5, frameon=False)
    fig.suptitle("Does each cortical readout track task SUCCESS or kinematic NORMALITY?\nWithin-animal partial "
                 "correlations over post sessions; pooled = mean Fisher z, within-animal permutation p", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 0.86, 0.93))
    fig.savefig(_w(FIG / "kn_fig4_success_vs_normality.png"), dpi=120)
    plt.close(fig)
    print("-> figures in", FIG)


# ============================================================================================ main
def run() -> dict:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    S = sessions()
    print(f"sessions: {len(S)}  ", S.groupby(["animal", "epoch"]).size().unstack().reindex(columns=EP4).to_string())
    T = load_trials(S)
    print(f"engaged trials {len(T)}; hits {int(T.is_hit.sum())}; attempted {int(T.is_att.sum())}")
    T = score(T)
    C = coverage(T)
    D = session_table(T, S)
    N, ref = normalise(D)
    print("== per animal x epoch session means (rule 8: per-animal table first)")
    PA, ET = epoch_stats(D)
    print(PA.round(3).to_string(index=False))
    print("== boot_delta vs August pre (change = CI excludes 0 AND >= 3/4 same sign)")
    print(ET.round(3).to_string(index=False))
    print("== normalisation references (pre / acute session means, raw units)")
    print(ref.round(3).to_string(index=False))
    fits, fdiff, curves = fit_all(N, ref)
    print("== exponential fits on the fraction-of-acute-deficit scale (post sessions; CI = 1000 session resamples)")
    print(fits.round(2).to_string(index=False))
    CA, CP = correlations(D)
    print("== within-animal correlations (log day partialled)")
    print(CA.round(3).to_string(index=False))
    print("== pooled (mean Fisher z; permutation p)")
    print(CP.round(3).to_string(index=False))
    for name, df in (("trials.csv.gz", T), ("sessions.csv", D), ("sessions_normalised.csv", N), ("coverage.csv", C),
                     ("per_animal_epoch.csv", PA), ("epoch_tests.csv", ET), ("norm_reference.csv", ref),
                     ("fits.csv", fits), ("fit_diffs.csv", fdiff), ("corr_animals.csv", CA),
                     ("corr_pooled.csv", CP)):
        df.to_csv(_w(OUT / name), index=False)
    pd.DataFrame([{"series": k, "label": LAB[k], "source": SRC.get(k, "this analysis")} for k, _ in ALL_SERIES]).to_csv(
        _w(OUT / "series_sources.csv"), index=False)
    F = {"N": N, "D": D, "fits": fits, "curves": curves, "epoch_tests": ET, "corr_animals": CA, "corr_pooled": CP}
    from figdata import save_fd
    _w(FD / "x")
    save_fd("kinematic_normality", F, FD)
    return F


def main() -> int:
    from figdata import load_fd
    if "--plot-only" in sys.argv:
        F = load_fd("kinematic_normality", FD)
    else:
        F = run()
    draw(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
