"""COVARIATE REFERENCES for the like-for-like (same-place) offset (overnight queue 5a, approved backlog; Priya 10-27 late:
"Four COVARIATE references for the like-for-like offset ... Gaussian kernel on landing + linear covariate terms fit on
pre licks"). Does the contralesional read-out offset survive once the lick's KINEMATICS and the FACE (FaceRhythm) around
it are matched as well as its landing place?

Per lick of the frozen KEY-row lateral decoder (split basis, per-lick predictions `iti_lick_order.preds_path()`), four
references, all built from the animal's PRE licks of the same set (any spout; pre licks leave-one-pre-session-out):
  landing    the current Gaussian same-place reference (likeforlike method "nw": Gaussian-weighted MEAN of the pre
             decoded values on landing, NaN beyond the pre landing range / weight < 3; sigma by held-out pre error)
  kin        landing + 3-D kinematics: reach (reach_mm, 3-D tip distance from the mouth at the peak), peak speed
             (speed_peak), elevation (below the axial plane, atan2(down, hypot(lateral, forward)), deg) -- the pose3d lick
             table (`iti_licks.lick_table`), joined on lick_uid -> exact DAQ peak time as iti_licks.extract does
  fr         landing + the five FaceRhythm master_pre_v1 mode-A factors of the frozen encoder (iti_licks.FR_KEEP: lick,
             right, left, close, far; grouping COMBINED_ENCODING_FR_GROUPS=master_v1:A, absolute expression), each the
             mean over FR_WIN around the lick peak (spans the decoder's 0..0.2 s feature window)
  fr_kin     landing + FR + kinematics
MODEL = PARTIAL-LINEAR (Robinson double residual): decoded = m(landing) + beta . Z + e, m non-parametric. Each of decoded
and the covariates Z is residualised on landing with the SAME Gaussian kernel as the landing-only reference (Nadaraya-
Watson from the pre licks, same sigma, same range / weight rules); beta = OLS (no intercept) of the decoded residual on
the covariate residuals over PRE licks; reference(lick) = m_dec(x) + beta . (Z - m_Z(x)), so
    resid = (dec - m_dec(x)) - beta . (Z - m_Z(x)).
Why this one (vs a covariate fit inside each kernel neighbourhood): with Z empty it IS the current Gaussian reference
(reference 1 reproduces likeforlike "nw" to float precision -- checked at run time), so the four references differ ONLY
by the covariate terms; beta is one pre-stroke slope per set x animal (thousands of pre licks per coefficient, no
per-neighbourhood fits that go unstable at the range edges where neighbours are few); the landing dependence of Z is
removed before beta is fitted, so beta is the covariate effect AT A FIXED LANDING PLACE and cannot re-absorb the landing
effect. Pre licks: m from the other pre sessions and beta fitted on the other pre sessions' (leave-one-out) residuals,
so the pre sessions stay out of sample (~0 by construction, their spread = the noise level); post licks: m and beta from
all pre licks. COMMON LICK SET: all four references are scored on the licks with landing, decoded and all eight
covariates finite (so differences are the covariates, not the lick set); coverage per set x epoch and the landing-only
offset on ALL licks (the published number) are reported beside it.
FACE RHYTHM IN THE ITI: the frozen encoder BLANKS FaceRhythm outside position strobe .. trial end (face_rhythm_outputs.
keep_mask, the video mask), which removes ~96 % of ITI lick peaks. Here the factors are read UNBLANKED (blank=False):
inside the trial windows the values are identical to the frozen encoder's (25 Hz trial samples); outside them the FR
projection's 5 Hz session samples are used (a slower, smoother face read-out in the ITI -- caveat). `fr_in_mask` records
which licks the frozen mask would have kept. (NB the stored `bout_fr` of the ITI session files interpolates ACROSS the
blanked ITI stretches -- iti_licks.extract fr_at uses only finite frames -- so it is not an ITI face measurement either;
since 10-28 the loader here is the shared `iti_licks.fr_unblanked`, which the ITI controls now use too.)
How much of the offset each covariate set explains: explained = landing-only resid - covariate resid (= beta . Z
residual, per lick), its change vs pre with the same session test, and share = 1 - change(ref) / change(landing).
Session tests as lick_matched_readout: session means (>= MIN licks), change vs pre = paired nested bootstrap
(analysis_kit.boot_delta), per-animal table first (rule 8), mixed model second opinion (iti_lick_order._lmm).
Outputs <OUT_W>/covref_{covariates.csv.gz,betas.csv,coverage.csv,sessions.csv,tests.csv,explained.csv},
figures/lick_covref_offset<SRC_TAG>.png; figdata/lick_covref_offset<SRC_TAG>.pkl (--plot-only redraws from it).
Usage: python scripts/lick_covariate_reference.py [--plot-only] [--animals PS94 ...] [--jobs 8]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import likeforlike as lfl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP = ["acute", "subacute", "chronic"]
EPS = ["pre", *EP]
SETS = {"iti": "ITI", "trial": "cued"}
STEM = f"lick_covref_offset{lo.SRC_TAG}"       # canonical split basis: untagged (iti_lick_order NAMING)
MIN = 8
KIN = ["reach_mm", "speed_peak", "elev_deg"]
FR = list(il.FR_KEEP)
FR_WIN = np.round(np.arange(-0.1, 0.2001, 0.02), 3)     # s from the lick peak; mean of the samples (>= half finite)
REFS = {"landing": [], "kin": KIN, "fr": FR, "fr_kin": FR + KIN}
RLAB = {"landing": "landing only\n(current)", "kin": "+ kinematics", "fr": "+ FaceRhythm", "fr_kin": "+ FR\n+ kinematics"}
RCOL = {"landing": "0.25", "kin": "#2c7fb8", "fr": "#1b9e77", "fr_kin": "#7b3294"}
ALLCOV = FR + KIN


# ------------------------------------------------------------------------------------------------ per-lick covariates
_KC: dict = {}


def _kin_table() -> pd.DataFrame:
    if "K" not in _KC:
        _KC["K"] = il.lick_table()
    return _KC["K"]


def _match(side: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Row of each prediction in the session file: the predictions are the session's licks in file order with some
    dropped (iti_lick_order.predictions `ok`), so an ordered subsequence match on the exact landing value recovers them.
    -1 = unmatched. (= iti_licks.subseq_match, the shared definition.)"""
    return il.subseq_match(side, act)


def _acts(label: str) -> dict:
    """{set: landing values of the label's prediction rows, in prediction order} (the prediction cache, read once per
    process -- the fan-out item stays a bare label, so the progress log stays readable)."""
    if "P" not in _KC:
        _KC["P"] = pd.read_csv(lo.preds_path(), usecols=["set", "label", "act"])
    P = _KC["P"]
    return {st: g.act.to_numpy(float) for st, g in P[P.label == label].groupby("set")}


def _session_cov(label: str) -> pd.DataFrame:
    """One session (module-level for spawn) -> per prediction row: set, label, k (row within label x set), t (exact DAQ
    peak), the kinematics, the FR factors, fr_in_mask."""
    from wfield_local import config
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    acts = _acts(label)
    rv = PathResolver()
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    d = il._load(label)
    A = pd.read_csv(oo.session_dir(rv, oc.RUN, an, date) / "licks_all.csv", usecols=["daq_peak_s"])
    K = _kin_table()
    K = K[K.session == f"{an}_{date}"].drop_duplicates("lick_uid")
    kt = pd.DataFrame({"reach_mm": K.reach_mm.to_numpy(float), "speed_peak": K.speed_peak.to_numpy(float),
                       "side_k": K.side_ml.to_numpy(float),
                       "elev_deg": np.degrees(np.arctan2(K.down_dv.to_numpy(float),
                                                         np.hypot(K.side_ml.to_numpy(float), K.out_ap.to_numpy(float))))},
                      index=A.daq_peak_s.to_numpy(float)[K.lick_uid.to_numpy(int)])
    kt = kt[~kt.index.duplicated()]
    out = []
    for st, (tk, sk) in {"iti": ("iti_t", "iti_side"), "trial": ("tr_t", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = _match(d[sk].astype(float), act)
        t = np.where(m >= 0, d[tk].astype(float)[np.maximum(m, 0)], np.nan)
        R = pd.DataFrame({"set": st, "label": label, "k": np.arange(len(act)), "t": t})
        R[["reach_mm", "speed_peak", "elev_deg", "side_k"]] = kt.reindex(t)[["reach_mm", "speed_peak", "elev_deg",
                                                                             "side_k"]].to_numpy()
        try:
            grid = (np.nan_to_num(t)[:, None] + FR_WIN[None, :]).ravel()
            FU = il.fr_unblanked(an, date, grid, rv)     # the shared UNBLANKED loader (rule 9; 10-28)
            for j, v in enumerate(FR):
                V = FU[:, j].reshape(len(t), len(FR_WIN))
                ok = np.isfinite(V).sum(1) >= len(FR_WIN) / 2
                R[v] = np.where(ok & np.isfinite(t), np.nanmean(np.where(np.isfinite(V), V, np.nan), 1), np.nan)
            R["fr_in_mask"] = fro.keep_mask(an, date, rv, np.nan_to_num(t))
        except FileNotFoundError as ex:                  # no FaceRhythm for this session: NaN (reported in coverage)
            print(f"  {label}: no FaceRhythm ({ex})", flush=True)
            R[FR] = np.nan
            R["fr_in_mask"] = False
        out.append(R)
    return pd.concat(out, ignore_index=True)


def covariates(P: pd.DataFrame, jobs: int | None) -> pd.DataFrame:
    """Per-lick covariates for every prediction row (cached <OUT_W>/covref_covariates.csv.gz, keyed by set x label x k)."""
    f = il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz"
    labs = sorted(P.label.unique())
    if f.exists():
        C = pd.read_csv(f)
        if set(labs) <= set(C.label):
            return C
    res, fail = ak.fan_sessions(labs, _session_cov, jobs=jobs)
    if fail:
        raise RuntimeError(f"covariates failed for {len(fail)} sessions: {fail[:5]}")
    C = pd.concat([v for _it, v in res], ignore_index=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    C.to_csv(f, index=False)
    return C


# ------------------------------------------------------------------------------------------------ references
def _nw_multi(xr: np.ndarray, Yr: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-weighted MEAN of each column of Yr at x -- likeforlike.reference(method "nw") for many columns at once:
    grid step sigma / 5 over the pre range [Q_EDGE percentiles], NaN where the raw weight is < 3 and beyond the range."""
    lo_, hi = np.percentile(xr, lfl.Q_EDGE)
    g = np.arange(lo_, hi + sigma / 5, sigma / 5)
    w = np.exp(-0.5 * ((g[:, None] - xr[None, :]) / sigma) ** 2)
    sw = w.sum(1)
    V = (w @ Yr) / np.maximum(sw, 1e-12)[:, None]
    V[sw < 3] = np.nan
    xc = np.clip(x, lo_, hi)
    out = np.column_stack([np.interp(xc, g, V[:, j]) for j in range(Yr.shape[1])])
    out[(x < lo_) | (x > hi)] = np.nan
    return out


def _ols(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(Z).all(1)
    return np.linalg.lstsq(Z[ok], y[ok], rcond=None)[0] if ok.sum() > Z.shape[1] + 10 else np.full(Z.shape[1], np.nan)


def references(Q: pd.DataFrame, sigma: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """resid_<ref> and expl_<ref> (= resid_landing - resid_<ref>) per lick of Q (the common lick set); beta table."""
    cols = ["dec", *ALLCOV]
    out = pd.DataFrame(np.nan, index=Q.index, columns=[f"resid_{r}" for r in REFS] + [f"expl_{r}" for r in REFS])
    betas = []
    for (st, a), g in Q.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        if len(pre) < 20 or pre.label.nunique() < 2:
            continue
        M = np.full((len(g), len(cols)), np.nan)
        ip = g.epoch.to_numpy() == "pre"
        post = g[~ip]
        if len(post):
            M[~ip] = _nw_multi(pre.act.to_numpy(float), pre[cols].to_numpy(float), post.act.to_numpy(float), sigma)
        labs = g.label.to_numpy()
        for lab in pre.label.unique():
            o = pre[pre.label != lab]
            i = labs == lab
            M[i] = _nw_multi(o.act.to_numpy(float), o[cols].to_numpy(float), g.act.to_numpy(float)[i], sigma)
        Rs = g[cols].to_numpy(float) - M
        ry = Rs[:, 0]
        for ref, cov in REFS.items():
            if not cov:
                r = ry
            else:
                Z = Rs[:, [1 + ALLCOV.index(c) for c in cov]]
                r = np.full(len(g), np.nan)
                b_all = _ols(Z[ip], ry[ip])
                r[~ip] = ry[~ip] - Z[~ip] @ b_all
                for lab in pre.label.unique():           # pre: beta from the OTHER pre sessions (out of sample)
                    i = labs == lab
                    b = _ols(Z[ip & ~i], ry[ip & ~i])
                    r[i] = ry[i] - Z[i] @ b
                zsd = np.nanstd(Z[ip], 0)
                okp = ip & np.isfinite(ry)
                mse0 = float(np.nanmean(ry[okp] ** 2))
                betas.append(dict(set=st, animal=a, ref=ref, n_pre=int(okp.sum()),
                                  pre_mse_landing=mse0, pre_mse_ref=float(np.nanmean(r[okp] ** 2)),
                                  **{f"b_{c}": float(v) for c, v in zip(cov, b_all)},
                                  **{f"bz_{c}": float(v * s) for c, v, s in zip(cov, b_all, zsd)}))
            out.loc[g.index, f"resid_{ref}"] = r
    for ref in REFS:
        out[f"expl_{ref}"] = out["resid_landing"] - out[f"resid_{ref}"]
    return out, pd.DataFrame(betas)


# ------------------------------------------------------------------------------------------------ sessions + tests
VALS = [f"resid_{r}" for r in REFS] + [f"expl_{r}" for r in REFS if r != "landing"]


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, g0 in (("all", P), ("first", P[P.order == 0])):
        for (st, a, lab, ep), h in g0.groupby(["set", "animal", "label", "epoch"]):
            r = dict(scope=f"all spouts|{cls}", cls=cls, set=st, animal=a, label=lab, epoch=ep, n=len(h))
            for v in [*VALS, "resid_landing_all"]:
                r[v] = h[v].mean() if h[v].notna().sum() >= MIN else np.nan
            r["n_common"] = int(h.common.sum())
            rows.append(r)
    return pd.DataFrame(rows)


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (scope, st), g in S.groupby(["scope", "set"]):
        for val in [*VALS, "resid_landing_all"]:
            prd = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            prd = {a: v for a, v in prd.items() if len(v)}
            pm = g[g.epoch == "pre"].groupby("animal")[val].mean()
            for ep in EP:
                post = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                post = {a: v for a, v in post.items() if len(v)}
                if not post:
                    continue
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                if ci is None:
                    continue
                am = (g[g.epoch == ep].groupby("animal")[val].mean() - pm).dropna()
                mx = lo._lmm(g, val, ep) if g.animal.nunique() > 1 else {"mixed_coef": np.nan, "mixed_p": np.nan}
                out.append(dict(scope=scope, set=st, measure=val, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am), **mx,
                                **{f"animal_{a}": v for a, v in am.items()}))
    return pd.DataFrame(out)


def explained(T: pd.DataFrame) -> pd.DataFrame:
    """Share of the landing-only offset change explained by each covariate set: 1 - change(ref) / change(landing),
    from the animal-weighted points, and per animal."""
    rows = []
    for (scope, st, ep), g in T.groupby(["scope", "set", "epoch"]):
        g = g.set_index("measure")
        if "resid_landing" not in g.index:
            continue
        base = g.loc["resid_landing"]
        for ref in REFS:
            if f"resid_{ref}" not in g.index:
                continue
            r = g.loc[f"resid_{ref}"]
            rows.append(dict(scope=scope, set=st, epoch=ep, ref=ref, change_landing=base.point, change_ref=r.point,
                             explained_mm=base.point - r.point,
                             share=1 - r.point / base.point if abs(base.point) > 1e-9 else np.nan,
                             **{f"share_{c[7:]}": 1 - r[c] / base[c] if abs(base[c]) > 1e-9 else np.nan
                                for c in g.columns if c.startswith("animal_")}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figure
def compute(animals: list[str] | None, jobs: int | None) -> dict:
    P = pd.read_csv(lo.preds_path())
    if animals:
        P = P[P.animal.isin(animals)]
    P = P.reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = covariates(P, jobs)
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    bad = np.nanmax(np.abs(P.side_k - P.act)) if P.side_k.notna().any() else np.nan
    print(f"join check: {int(P.t.isna().sum())} predictions unmatched; max |kinematics side_ml - act| = {bad:.2e} mm")
    # sigma as the published offset (likeforlike.offset "nw": held-out pre error on ALL licks); landing-only on ALL
    # licks = the published number, shown beside the common-set references
    ok = np.isfinite(P[["act", "dec"]].to_numpy(float)).all(1)
    sg, cv = lfl.choose(P[ok], "act", "dec", lfl.SIGMA_MM, "nw")
    print("held-out pre error per method x sigma:\n" + cv.round(4).to_string(index=False) + f"\n-> sigma = {sg} mm")
    P["resid_landing_all"] = np.nan
    P.loc[ok, "resid_landing_all"] = P.dec[ok] - lfl.ref_table(P[ok], "act", "dec", sg, "nw").ref
    P["common"] = np.isfinite(P[["act", "dec", *ALLCOV]].to_numpy(float)).all(1)
    Q = P[P.common]
    R, B = references(Q, sg)
    P = P.join(R)
    # reference 1 must reproduce likeforlike "nw" on the common set
    chk = lfl.ref_table(Q, "act", "dec", sg, "nw")
    d1 = np.nanmax(np.abs((Q.dec - chk.ref) - P.loc[Q.index, "resid_landing"]))
    print(f"landing-only reference vs likeforlike nw on the common set: max |diff| = {d1:.2e} (sigma {sg} mm)")
    cov = P.groupby(["set", "epoch"]).agg(n=("dec", "size"), common=("common", "mean"),
                                          fr_finite=("fr_lick", lambda s: s.notna().mean()),
                                          kin_finite=("reach_mm", lambda s: s.notna().mean()),
                                          fr_in_mask=("fr_in_mask", "mean"),
                                          landing_ref=("resid_landing_all", lambda s: s.notna().mean())).reset_index()
    S = sessions(P)
    T = tests(S)
    X = explained(T)
    return dict(sigma=sg, sigma_cv=cv, coverage=cov, betas=B, S=S, T=T, X=X,
                animals=sorted(P.animal.unique()), d1=d1)


def draw(D: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    T, X = D["T"], D["X"]
    refs = list(REFS)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), squeeze=False, sharey="row")
    for i, st in enumerate(SETS):
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            for k, ref in enumerate(refs):
                t = T[(T.scope == "all spouts|all") & (T.set == st) & (T.epoch == ep) & (T.measure == f"resid_{ref}")]
                if t.empty:
                    continue
                r = t.iloc[0]
                c = RCOL[ref]
                for q, a in enumerate(D["animals"]):
                    v = r.get(f"animal_{a}", np.nan)
                    if np.isfinite(v):
                        ax.plot(k - 0.15 + 0.1 * q, v, ANIMAL_MARKER.get(a, "o"), mfc="none", mec=c, ms=6, mew=1.1)
                ax.errorbar(k + 0.25, r.point, yerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o", color=c, ms=7,
                            capsize=3, lw=1.6)
                sig = r.lo > 0 or r.hi < 0
                x_ = X[(X.scope == "all spouts|all") & (X.set == st) & (X.epoch == ep) & (X.ref == ref)]
                # explained (mm) always; the % only where the landing-only change itself excludes 0 (a share of ~0 is noise)
                b_ = T[(T.scope == "all spouts|all") & (T.set == st) & (T.epoch == ep) & (T.measure == "resid_landing")]
                bsig = len(b_) and (b_.lo.iloc[0] > 0 or b_.hi.iloc[0] < 0)
                sh = "" if ref == "landing" or x_.empty else f"\nexpl {x_.explained_mm.iloc[0]:+.2f}" + (
                    f" ({100 * x_.share.iloc[0]:.0f}%)" if bsig else "")
                mp = "†" if np.isfinite(r.get("mixed_p", np.nan)) and r.mixed_p < 0.05 else ""
                ax.text(k, 1.01, f"{r.point:+.2f}{'*' if sig else ''}{mp}\n{int(r.n_dir)}/{int(r.n)}{sh}", ha="center",
                        va="bottom", fontsize=6.8, color=c, fontweight="bold" if sig else "normal",
                        transform=ax.get_xaxis_transform(), clip_on=False)
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(len(refs)), [RLAB[r] for r in refs], fontsize=7.5)
            ax.set_xlim(-0.6, len(refs) - 0.4)
            ax.set_title(f"{SETS[st]} licks, {ep} - pre", fontsize=9.5, pad=38, color=EPOCH_LINE[ep])
            if j == 0:
                ax.set_ylabel("change vs pre of the like-for-like offset\n(mm, + = read more contralesional)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    cov = D["coverage"]
    cv = ", ".join(f"{SETS[s]} {e} {100 * c:.0f}%" for s, e, c in cov[["set", "epoch", "common"]].itertuples(index=False)
                   if e != "pre")
    fig.suptitle("Does the like-for-like (same-place) offset survive matching the lick's KINEMATICS and FACE too? Frozen "
                 "KEY-row lateral decoder (z-scored " + lo.DEC_LAB + ").\nReference = the animal's PRE licks (any spout): "
                 f"Gaussian kernel on landing (sigma {D['sigma']} mm) + linear covariate terms fit on pre licks "
                 "(partial-linear; pre leave-one-session-out).\nKinematics = 3-D reach, peak speed, elevation; FaceRhythm "
                 "= 5 master_pre_v1 factors, mean -0.1..+0.2 s around the lick peak.\nOpen markers = animals (circle PS92, "
                 "square PS93, triangle PS94, diamond PS95), filled = mean of animals with 95% paired nested bootstrap CI; "
                 "* = CI excludes 0 (bold), † = mixed model p < 0.05, k/n animals;\nexpl = landing-only change minus this "
                 "reference's change (mm; % of the landing-only change where that excludes 0). Common lick set (all "
                 f"covariates finite):\n{cv}. PROVISIONAL", fontsize=9)
    lo._layout(fig, bottom=0.02)
    out = il.OUT_W / "figures" / f"{STEM}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, out, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--animals", nargs="*")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    if plot_only():
        draw(load_fd(STEM))
        return 0
    if "COMBINED_ENCODING_FR" not in os.environ or "COMBINED_ENCODING_FR_GROUPS" not in os.environ:
        raise SystemExit("run under the frozen env (rr2/frozenenv.sh): COMBINED_ENCODING_FR / _FR_GROUPS unset")
    D = compute(a.animals, a.jobs)
    pd.set_option("display.width", 250)
    print("== coverage (share of licks in the common set; FR / kinematics finite; FR inside the frozen video mask):")
    print(D["coverage"].round(3).to_string(index=False))
    print("== betas (per set x animal; bz = beta x covariate SD, mm per SD) and pre held-out MSE:")
    print(D["betas"].round(4).to_string(index=False))
    T = D["T"]
    cols = ["scope", "set", "measure", "epoch", "point", "lo", "hi", "n_dir", "n", "mixed_p"] + \
        [c for c in T.columns if c.startswith("animal_")]
    print("== per animal (rule 8) + nested bootstrap:")
    print(T[cols].round(3).to_string(index=False))
    print("== explained share of the landing-only offset change:")
    print(D["X"].round(3).to_string(index=False))
    tag = lo.SRC_TAG
    il.OUT_W.mkdir(parents=True, exist_ok=True)
    D["betas"].to_csv(il.OUT_W / f"covref_betas{tag}.csv", index=False)
    D["coverage"].to_csv(il.OUT_W / f"covref_coverage{tag}.csv", index=False)
    D["S"].to_csv(il.OUT_W / f"covref_sessions{tag}.csv", index=False)
    T.to_csv(il.OUT_W / f"covref_tests{tag}.csv", index=False)
    D["X"].to_csv(il.OUT_W / f"covref_explained{tag}.csv", index=False)
    save_fd(STEM, D)
    draw(D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
