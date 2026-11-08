"""PER-SESSION Haufe LEVEL maps with AMPLITUDE KEPT -- the per-session replacement for the held-out-epoch-decoder statistics
of `iti_decoder_pattern_stats.py` (lick_decoder_pattern_stats_lat_levels.png) as the Key map figure (Priya 10-27).

MAPS. Every session gets its own decoder, fit on THAT session's licks only (ridge, penalty by internal CV), and its own
Haufe activation pattern A = Cov(X, yhat) / Var(yhat) on the same licks, mapped to pixels with the animal's pre-stroke SVD
basis (W @ A) -- exactly `iti_session_patterns._pattern` / `_within` (features: 200 SVD components, lags 0..0.2 s from the
lick peak, unscaled). Unlike `iti_session_pattern_tests.load` the maps are NOT scaled to unit RMS: each is divided by ONE
number per animal x set x target, the mean RMS of the animal's PRE-session ALL-lick maps of that set and target, so a
cortex-wide change in response size stays in the map, and first-lick maps are on the SAME scale as all-lick maps.
Targets per lick set (ITI; cued = all post-cue licks of the extraction, `iti_licks.extract`):
  lat / sag   landing position (lateral + = contralesional; sagittal extension, + = longer)
  wlat        WITHIN-SPOUT lateral: landing minus the session's mean landing at the spout (cued: target spout; ITI:
              previous trial's spout), spouts with < MIN_SPOUT licks dropped
  sp_<spout>  one-vs-rest indicator of the spout (cued: target; ITI: previous trial's spout); ALL licks only
Lick subsets: ALL licks, and FIRST licks (ITI: no lick onset of any context in the preceding 2 s, gap >= 2.0 in
<E>/iti_licks/iti_timing_audit.csv joined on label + exact peak time, = `iti_lick_order.iti_keep` with
WFIELD_ITI_FILTER=gap2 -- Priya 10-27, replacing the 1 s bout rule; NOT also required to be far from events, since the
ITI is only ~3.1 s; the share of these >= 1 s from both the window end and the next spout arrival is reported; cued:
lick_idx 0 = the first post-cue lick, `iti_first_lick._lick_idx`). Later licks of a bout overlap the previous lick's activity at lags 0..0.2 s
(ILI ~150 ms) and are corrective. A session map needs >= MIN_LICKS (40) licks in the subset (>= MIN_SPOUT per spout).

LEVEL TEST per map x epoch (is the epoch's map != 0, per pixel): `iti_session_cluster_tests.test_level`. Statistic
`level_z`: per animal the mean of its session maps of the epoch / s_pooled, s_pooled = the within-animal session SD
pooled (df-weighted) over ALL animals' sessions of that epoch (floored at its 10th percentile over pixels), summed over
animals / sqrt(k) (animals weighted equally); TFCE (E 0.5, H 2); null = SIGN-FLIP of whole
SESSION maps, independently per session (within each animal), N_PERM x; FWER p = share of permutations whose max |TFCE|
over cortex reaches the pixel's. Why per session, not per animal: a self-fit Haufe pattern flips sign exactly with the
target (A(X, -y) = -A(X, y)), so if a session's licks carry no information about the target its map is symmetric about 0,
and sessions are independent fits on disjoint licks -> session sign-flips are exchangeable under H0 "no session of this
epoch codes the target". Flipping whole maps keeps the spatial correlation in the null. Flipping per ANIMAL (the random-
effects test) has 2^4 = 16 arrangements at n = 4 (min two-sided p 0.125): it cannot reach 0.05. So inference is over
sessions within THESE animals (fixed effects over animals); the per-animal table (rule 8) and the dashed outline (>= 3
animals share the sign) are the guard against one animal carrying a pixel.
CHANGE TEST epoch - pre: `iti_session_cluster_tests.test_one`, statistic `change_z`: per animal (mean of its epoch
sessions - mean of its pre sessions) / s_pooled, s_pooled = the within-animal session SD pooled over ALL animals and both
groups (df-weighted, floored), summed over animals / sqrt(k); null = session labels permuted between pre and the epoch
within animal (s_pooled recomputed per draw), max-|TFCE| FWER; on the amplitude-kept maps. The same two statistics serve
the shape-only figures of iti_session_cluster_tests.py (one definition, rule 9).
INCLUSION: EVERY animal with >= 1 session in the epoch (level) or >= 1 pre AND >= 1 epoch session (change) contributes --
an animal with one session in a group adds no df to s_pooled but its mean / difference still counts, and sign-flipping a
single session map is still valid. A cell is drawn whenever any animal has data (no minimum number of animals); the dashed
">= 3 animals share the sign" outline is the only animal-agreement marker. Session counts per animal per cell are
printed (rule 8).

PER-ANIMAL TABLE (rule 8): per animal x map x epoch: n sessions, licks per session, gain (mean session-map RMS / the pre
denominator), RMS of the animal's epoch-mean map and its noise-debiased version (mean^2 - var/n per pixel), r to its own
pre mean map (pre: r of the odd- vs even-session pre means = the reliability floor), GLOBAL SHARE = (mean over cortex)^2
/ mean square of the epoch-mean map (1 = a uniform cortex-wide component; debiased version too) and the RMS left after
removing the cortex mean (rms_nonglobal).
AMPLITUDE CAVEAT: A = Cov(X, yhat) / Var(yhat) scales as 1 / |decoder weights|, so a session whose ridge decoder is
shrunk toward 0 (little information about the target) gets a LARGE, noise-dominated pattern; the map's sign-symmetric
noise does not bias the level test, but amplitude in A is not coding strength. The table therefore also gives fit_med (in-
sample var(yhat) / var(y)) and the ENCODING SLOPE map B = Cov(X, y) / Var(y) per session (no decoder; activity per unit
target), normalized the same way: gain_enc, rms_enc_mean, global_share_enc, r_pre_enc.
EXCLUSION: a session map is dropped when its decoder's in-sample fit share var(yhat) / var(y) < MIN_FIT (0.05): the ridge
has shrunk to ~0 and A is pure noise blown up by 1 / |w| (measured: cued FIRST-lick within-spout, median fit 0.018, 45 of
107 sessions < 0.01, |A| / |B| up to ~2600x vs 1.3-2.2x for cued all-lick lateral). The fit share is invariant to y -> -y,
so selecting on it leaves the session sign-flip null valid. Dropped sessions are counted in session_levels/dropped.csv.
All map statistics use the drawn pixels (stat mask on every 2nd row / column, `figdata.imaging_grid`).
Outputs <E>/iti_licks/session_levels/{patterns.pkl (A vectors per session), tests.pkl, per_animal.csv, tests.csv,
first_lick_counts.csv (incl. far1), heldout_comparison.csv}; figures/lick_session_levels_{lat,sag,within_spout}.png,
lick_{cued,iti}_session_levels_spout.png, lick_session_change_amp_{lat,sag,within_spout,spout_cued,spout_iti}.png;
figure data figdata/<figure stem>.pkl. --plot-only redraws from the figure data alone; --reuse-tests rebuilds tables +
figures from tests.pkl.
Z-SCORED VARIANT (Priya 10-27 late "yes do both", OVERNIGHT 5c-ter: are the ITI maps' cortex-wide components real or a
scale effect of the unscaled SVD features?): env WFIELD_MAP_SCALE=frame divides each SVD component of every session by
its frame-wise SD in that session (<E>/iti_licks/basis_svdbasis/frame_sd.csv, `iti_basis_frame_sd.py`; 107 sessions x
200 components). Everything else is identical: the per-session decoder standardizes its features (StandardScaler), so
yhat, the fit share and the MIN_FIT drops are unchanged and the patterns transform EXACTLY as A_z = A / sd, B_z = B / sd
(per session; checked numerically). So the variant reuses the unscaled pattern cache (session_levels/patterns_gap2_enc.pkl,
read only) and rescales before mapping to pixels (W @ A_z), normalizing and testing as above. Outputs go to
<E>/iti_licks/session_levels_z/ and every figure / figure-data stem gets "_z" (nothing unscaled is overwritten).
The side-by-side comparison is `lick_session_levels_scale_compare.py`.
Usage: [WFIELD_MAP_SCALE=frame] python scripts/lick_session_level_maps.py [--plot-only | --reuse-tests] [--perm N]
       [--jobs N]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import os
import pickle
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import CHANGE_CMAP, PRE_CMAP, grid_fns, imaging_grid, load_fd, plot_only, save_fd, sym_lim  # noqa: E402
from iti_session_cluster_tests import print_counts, test_level, test_one  # noqa: E402
from iti_session_patterns import AN, MIN_LICKS, MIN_SPOUT, _pattern, _within  # noqa: E402

from wfield_local.orofacial_cohort import POS_ORDER  # noqa: E402

POS = list(POS_ORDER)
EP = ["pre", "acute", "subacute", "chronic"]
SETS = ("iti", "trial")
SUBS = ("all", "first")
N_PERM = 1000
MIN_FIT = 0.05                        # session map kept only if in-sample var(yhat) / var(y) >= this (see docstring)
DROPPED: list = []
GAP_FIRST = 2.0                       # ITI first lick: no lick onset in the preceding 2 s
SCALE = os.environ.get("WFIELD_MAP_SCALE", "")       # "" = unscaled SVD features (canonical) | "frame" (module docstring)
if SCALE not in ("", "frame"):
    raise ValueError(f"WFIELD_MAP_SCALE={SCALE!r}: expected frame or unset")
ZS = "_z" if SCALE else ""                           # output tag of the scaled variant
BASE_OUTD = il.OUT / "session_levels"                # the unscaled run (its pattern cache serves both)
OUTD = il.OUT / f"session_levels{ZS}"
FIG = il.OUT / "figures"
SETLAB = {"iti": "ITI", "trial": "cued"}
TGLAB = {"lat": "LATERAL position", "sag": "SAGITTAL extension", "wlat": "WITHIN-SPOUT lateral"}
SIGN = {"lat": "positive (orange / red) = more active for licks landing more CONTRALESIONAL",
        "sag": "positive (orange / red) = more active for LONGER (more extended) licks",
        "wlat": "positive (orange / red) = more active for licks landing more CONTRALESIONAL than the spout's mean",
        "sp": "positive (orange / red) = more active for licks to THAT spout (cued: target; ITI: previous trial's spout)"}
LEVEL_FIG = {"lat": "lick_session_levels_lat.png", "sag": "lick_session_levels_sag.png",
             "wlat": "lick_session_levels_within_spout.png", "sp_trial": "lick_cued_session_levels_spout.png",
             "sp_iti": "lick_iti_session_levels_spout.png"}
CHANGE_FIG = {"lat": "lick_session_change_amp_lat.png", "sag": "lick_session_change_amp_sag.png",
              "wlat": "lick_session_change_amp_within_spout.png", "sp_trial": "lick_session_change_amp_spout_cued.png",
              "sp_iti": "lick_session_change_amp_spout_iti.png"}


def ztag(name: str, zs: str = ZS) -> str:
    """Figure / figure-data name of the scaled variant: 'a.png' -> 'a_z.png' (unchanged for the unscaled run)."""
    p = Path(name)
    return f"{p.stem}{zs}{p.suffix}"


LEVEL_FIG0, CHANGE_FIG0 = dict(LEVEL_FIG), dict(CHANGE_FIG)      # untagged names (the comparison reads both variants)
LEVEL_FIG = {k: ztag(v) for k, v in LEVEL_FIG0.items()}
CHANGE_FIG = {k: ztag(v) for k, v in CHANGE_FIG0.items()}


def fam_keys(fam: str) -> list[str]:
    if fam.startswith("sp_"):
        st = fam[3:]
        return [f"{st}/all/sp_{p}" for p in POS]
    return [f"{st}/{sub}/{fam}" for st in SETS for sub in SUBS]


def denom_key(k: str) -> str:
    """The key whose pre-session RMS normalizes k: the ALL-lick map of the same set and target."""
    st, _sub, tg = k.split("/")
    return f"{st}/all/{tg}"


# ------------------------------------------------------------------------------------------------ per-session patterns


_AUD = {}


def _put(rec: dict, key: str, X: np.ndarray, y: np.ndarray) -> None:
    """Haufe pattern A (`iti_session_patterns._pattern`) + the ENCODING SLOPE B = Cov(X, y) / Var(y) (no decoder) + the
    in-sample fit share var(yhat) / var(y). A = Cov(X, yhat) / Var(yhat) scales as 1 / |decoder weights|, so a session
    whose decoder is shrunk toward 0 (little information) gets a LARGE, noisy A: amplitude in A is not coding strength.
    B is the amplitude cross-check (activity per unit target)."""
    A = _pattern(X, y)
    if A is None:
        return
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    yh = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(X), y).predict(sc.transform(X))
    rec["A"][key], rec["n"][key] = A, len(y)
    rec["B"][key] = ((X - X.mean(0)).T @ (y - y.mean())) / (len(y) - 1) / y.var(ddof=1)
    rec["fit"][key] = float(yh.var() / y.var())


def _audit() -> pd.DataFrame:
    if "A" not in _AUD:
        _AUD["A"] = (pd.read_csv(il.OUT / "iti_timing_audit.csv", usecols=["label", "t", "gap", "since_stop", "to_strobe"])
                     .replace(np.inf, 1e9).set_index(["label", "t"]).sort_index())
    return _AUD["A"]


def _session(x: str):
    """One session -> record {epoch, animal, n: {key: licks}, A: {key: (200,) pattern}}. Module-level (spawn)."""
    import iti_first_lick as fl
    import iti_where_decoder as wd
    from iti_basis_decoder import LAGS as LB

    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    _x, d = wd._load_one((x, "svdbasis", ""))
    if d is None:
        return None
    z = il._load(x)
    ft = wd._frame_range(x, ce)
    rec = {"epoch": d["epoch"], "animal": d["animal"], "n": {}, "A": {}, "B": {}, "fit": {}, "n_set": {}}
    for st, kt, kd in (("iti", "iti_t", "iti_side"), ("trial", "tr_t", "tr_side")):
        t, y = z[kt].astype(float), z[kd].astype(float)
        keep = (t > ft[0] - LB[0]) & (t < ft[-1] - LB[-1]) & np.isfinite(y)   # the svdbasis edge rule (wd._load_one)
        t = t[keep]
        X, C = d[st]
        if len(t) != len(X):
            raise ValueError(f"{x} {st}: {len(t)} lick times vs {len(X)} feature rows")
        if st == "iti":                                  # ITI first = gap >= 2 s since any lick onset (Priya 10-27)
            Ax = _audit().loc[x]                         # nearest-time join: the CSV round-trip moves ~1% of t
            ta = Ax.index.to_numpy(float)                 # in the last digits, so an exact reindex misses them
            j = np.clip(np.searchsorted(ta, t), 1, len(ta) - 1)
            j = np.where(np.abs(ta[j - 1] - t) <= np.abs(ta[j] - t), j - 1, j)
            Au = Ax.iloc[j].set_axis(t)
            Au = Au.where(np.repeat((np.abs(ta[j] - t) <= 1e-6)[:, None], Au.shape[1], 1))
            if Au.gap.isna().any():
                raise ValueError(f"{x}: {int(Au.gap.isna().sum())} ITI licks missing from iti_timing_audit.csv")
            idx = np.where(Au.gap.to_numpy(float) >= GAP_FIRST, 0.0, 1.0)
            far1 = ((Au.since_stop >= 1.0) & (Au.to_strobe >= 1.0)).to_numpy()
            rec["iti_first_far1"] = (int((far1 & (idx == 0)).sum()), int((idx == 0).sum()))
        else:
            idx = fl._lick_idx(x, t)
        pos = np.array([p if p in POS or "_" not in p else oc.pos_key(p, "L") for p in d[f"{st}_pos"]])
        for sub in SUBS:
            m = np.ones(len(t), bool) if sub == "all" else (idx == 0)
            rec["n_set"][f"{st}/{sub}"] = int(m.sum())
            Xs, Cs, ps = X[m], C[m], pos[m]
            ok = np.isfinite(Cs[:, 0])
            _put(rec, f"{st}/{sub}/lat", Xs[ok], Cs[ok, 0])
            ok3 = np.isfinite(Cs[:, 3])
            _put(rec, f"{st}/{sub}/sag", Xs[ok3], Cs[ok3, 3])
            yw, kw = _within(np.where(ok, Cs[:, 0], np.nan), np.where(ok, ps, ""))
            kw &= ok
            if kw.sum() >= MIN_LICKS:
                _put(rec, f"{st}/{sub}/wlat", Xs[kw], yw[kw])
            if sub == "all":
                okp = np.isin(ps, POS)
                for sp in POS:
                    yy = (ps[okp] == sp).astype(float)
                    if MIN_SPOUT <= yy.sum() < len(yy):
                        _put(rec, f"{st}/all/sp_{sp}", Xs[okp], yy)
    return rec


def patterns(jobs: int) -> dict:
    # ITI first = gap >= 2 s (the 1 s bout-rule version: patterns.pkl). The UNSCALED patterns serve both variants (the
    # scaled one rescales them in build_maps): read from the unscaled folder; a scaled run that has to compute them writes
    # its own copy (never into the unscaled folder)
    f = OUTD / "patterns_gap2_enc.pkl"
    for g in (BASE_OUTD / f.name, f):
        if g.exists():
            return pickle.loads(g.read_bytes())
    from wfield_local import analysis_kit as ak
    labels = il.labels()
    res, fails = ak.fan_sessions(labels, _session, jobs=jobs, label="session pattern", log=lambda m: print(m[:110],
                                                                                                         flush=True))
    if fails:
        raise SystemExit(f"session patterns failed: {[(x, str(e)[:200]) for x, e in fails[:5]]}")
    P = {x: r for x, r in res if r is not None and r["animal"] in AN}
    f.write_bytes(pickle.dumps(P, protocol=pickle.HIGHEST_PROTOCOL))
    return P


def frame_scale(labels) -> dict:
    """{label: 1 / frame SD per SVD component} for WFIELD_MAP_SCALE=frame (iti_basis_frame_sd.py); None unscaled.
    Fails loudly on a missing session or a non-positive SD."""
    if not SCALE:
        return {x: None for x in labels}
    F = pd.read_csv(il.OUT / "basis_svdbasis" / "frame_sd.csv", index_col=0)
    miss = sorted(set(labels) - set(F.index))
    if miss:
        raise SystemExit(f"frame_sd.csv lacks {len(miss)} session(s): {miss[:5]}")
    V = F.loc[list(labels)].to_numpy(float)
    if not (np.isfinite(V) & (V > 0)).all():
        raise SystemExit("frame_sd.csv: non-positive or missing SD")
    return {x: 1.0 / F.loc[x].to_numpy(float) for x in labels}


def build_maps(P: dict, sel: np.ndarray) -> dict:
    """{animal: {key: {label, epoch, day, n, M (sessions, drawn pixels) / animal's pre RMS of the all-lick map}}}.
    Scaled variant: each session's A and B divided by its frame SD per component first (module docstring)."""
    from iti_decoder_patterns import SPATIAL
    from recovery_timing import series
    s = series("lick")
    days = dict(zip(s.label, s.day))
    inv = frame_scale(sorted(P))

    def sc(x, v):
        return v if inv[x] is None else v * inv[x]
    out = {}
    for a in AN:
        labs = sorted(x for x in P if P[x]["animal"] == a)
        if not labs:
            continue
        W = np.asarray(pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], np.float32)[sel]
        keys = sorted({k for x in labs for k in P[x]["A"]})
        raw = {}
        for k in keys:
            ls = [x for x in labs if k in P[x]["A"] and P[x]["fit"][k] >= MIN_FIT]
            DROPPED.extend((a, k, x, P[x]["epoch"]) for x in labs if k in P[x]["A"] and P[x]["fit"][k] < MIN_FIT)
            if not ls:
                continue
            M = (W @ np.column_stack([sc(x, P[x]["A"][k]) for x in ls]).astype(np.float32)).T.astype(np.float64)
            E = (W @ np.column_stack([sc(x, P[x]["B"][k]) for x in ls]).astype(np.float32)).T.astype(np.float64)
            raw[k] = (ls, M, E)
        out[a] = {}
        for k in [k for k in keys if k in raw]:
            ls, M, E = raw[k]
            dk = raw.get(denom_key(k))
            pre = [i for i, x in enumerate(dk[0]) if P[x]["epoch"] == "pre"] if dk else []
            if not pre:
                continue
            den = float(np.sqrt((dk[1][pre] ** 2).mean(1)).mean())
            den_e = float(np.sqrt((dk[2][pre] ** 2).mean(1)).mean())
            out[a][k] = dict(label=ls, epoch=np.array([P[x]["epoch"] for x in ls]),
                             day=np.array([days.get(x, np.nan) for x in ls], float),
                             n=np.array([P[x]["n"][k] for x in ls]), M=M / den, den=den,
                             E=E / den_e, fit=np.array([P[x]["fit"][k] for x in ls]))
        del W
    return out


# ------------------------------------------------------------------------------------------------ tests


def _dispatch(item):
    kind, rest = item
    return test_level(rest) if kind == "level" else test_one(rest)


def run_tests(L: dict, idx_img: np.ndarray, n_perm: int, jobs: int) -> dict:
    from wfield_local import parallel
    keys = sorted({k for a in L for k in L[a]})
    items = []
    for k in keys:
        for ep in EP:
            lv = {a: L[a][k]["M"][L[a][k]["epoch"] == ep] for a in AN if a in L and k in L[a]}
            lv = {a: m for a, m in lv.items() if len(m)}           # every animal with >= 1 session in the epoch
            if lv:
                items.append(("level", (k, ep, lv, idx_img, n_perm)))
            if ep == "pre":
                continue
            ch = {}
            for a in AN:
                v = L.get(a, {}).get(k)
                if v is None:
                    continue
                pre, post = v["M"][v["epoch"] == "pre"], v["M"][v["epoch"] == ep]
                if len(pre) and len(post):                          # >= 1 session in each group
                    ch[a] = (pre, post)
            if ch:
                items.append(("change", (k, ep, ch, idx_img, n_perm)))
    print(f"{len(items)} tests x {n_perm} permutations on {jobs} workers", flush=True)
    res, fails = parallel.fan_out(items, _dispatch, jobs=jobs, label="test", log=lambda m: print(m[:110], flush=True))
    if fails:
        raise SystemExit(f"tests failed: {[(f[0][0], f[0][1][:2], str(f[1])[:200]) for f in fails[:5]]}")
    return {(r["key"], r["epoch"], r["kind"]): r for _it, r in sorted(res, key=lambda t: (t[1]["key"], t[1]["epoch"],
                                                                                           t[1]["kind"]))}


# ------------------------------------------------------------------------------------------------ tables


def gshare(m: np.ndarray) -> float:
    return float(m.mean() ** 2 / (m ** 2).mean()) if (m ** 2).mean() > 0 else np.nan


def per_animal(L: dict) -> pd.DataFrame:
    rows = []
    for a in AN:
        for k, v in L.get(a, {}).items():
            pre = v["M"][v["epoch"] == "pre"]
            pre_e = v["E"][v["epoch"] == "pre"].mean(0) if len(pre) else None
            if len(pre) >= 2:
                o = np.argsort(v["day"][v["epoch"] == "pre"])
                pre_mean = pre.mean(0)
                r_floor = float(np.corrcoef(pre[o[0::2]].mean(0), pre[o[1::2]].mean(0))[0, 1])
            else:
                pre_mean, r_floor = None, np.nan
            for ep in EP:
                M = v["M"][v["epoch"] == ep]
                Ee = v["E"][v["epoch"] == ep]
                if not len(M):
                    continue
                n = len(M)
                m = M.mean(0)
                ms = (m ** 2).mean()
                ms_db = ms - (M.var(0, ddof=1).mean() / n if n > 1 else np.nan)
                g = M.mean(1)                                        # cortex mean of each session map
                g2_db = g.mean() ** 2 - (g.var(ddof=1) / n if n > 1 else np.nan)
                rows.append(dict(
                    animal=a, key=k, set=k.split("/")[0], sub=k.split("/")[1], target=k.split("/")[2], epoch=ep,
                    n_sess=n, licks_med=float(np.median(v["n"][v["epoch"] == ep])),
                    gain=float(np.sqrt((M ** 2).mean(1)).mean()), rms_mean=float(np.sqrt(ms)),
                    rms_debiased=float(np.sqrt(max(ms_db, 0))) if np.isfinite(ms_db) else np.nan,
                    rms_nonglobal=float(np.sqrt(ms - m.mean() ** 2)),
                    r_pre=(r_floor if ep == "pre" else float(np.corrcoef(m, pre_mean)[0, 1]) if pre_mean is not None
                           else np.nan),
                    global_share=gshare(m), cortex_mean=float(m.mean()),
                    global_share_debiased=float(g2_db / ms_db) if np.isfinite(g2_db) and ms_db > 0 else np.nan,
                    fit_med=float(np.median(v["fit"][v["epoch"] == ep])),
                    gain_enc=float(np.sqrt((Ee ** 2).mean(1)).mean()), rms_enc_mean=float(np.sqrt((Ee.mean(0) ** 2).mean())),
                    global_share_enc=gshare(Ee.mean(0)),
                    r_pre_enc=(float(np.corrcoef(Ee.mean(0), pre_e)[0, 1]) if ep != "pre" and pre_e is not None
                               else np.nan)))
    return pd.DataFrame(rows)


def tests_table(R: dict) -> pd.DataFrame:
    rows = []
    for (k, ep, kind), r in R.items():
        cx = np.isfinite(r["p"])
        sig = (np.nan_to_num(r["p"], nan=1) < 0.05) & cx
        pt = r["point"][cx]
        rows.append(dict(key=k, epoch=ep, kind=kind, n_animals=r["n"], n_sessions=r["n_sess"],
                         frac_sig=float(sig.sum() / cx.sum()), frac_sig_pos=float((sig & (r["point"] > 0)).sum() / cx.sum()),
                         frac_sig_neg=float((sig & (r["point"] < 0)).sum() / cx.sum()),
                         frac_sig_agree3=float((sig & (np.nan_to_num(r["agree"]) >= 3)).sum() / cx.sum()),
                         min_p=float(np.nanmin(r["p"])), rms=float(np.sqrt((pt ** 2).mean())), global_share=gshare(pt),
                         cortex_mean=float(pt.mean())))
    return pd.DataFrame(rows).sort_values(["key", "kind", "epoch"])


def heldout_comparison(T: pd.DataFrame) -> pd.DataFrame:
    try:
        H = load_fd("lick_decoder_pattern_stats_lat")["T"]
    except SystemExit:
        return pd.DataFrame()
    H = H[(H.kind == "level") & (H.axis == "lat")]
    rows = []
    for st in SETS:
        for ep in EP:
            h = H[(H.set == st) & (H.epoch == ep)]
            # iti_decoder_pattern_stats since 10-27 late: frac_sig = permutation FWER max-stat, frac_sig_fdr = the old
            # bootstrap BH, frac_sig_maxboot = the old bootstrap max-stat (before: frac_sig = BH, frac_sig_max)
            new = "frac_sig_fdr" in h
            row = dict(set=SETLAB[st], epoch=ep,
                       heldout_n_animals=float(h.n_animals.iloc[0]) if len(h) else np.nan,
                       heldout_fwer_maxstat=float(h["frac_sig" if new else "frac_sig_max"].iloc[0]) if len(h) else np.nan,
                       heldout_bh=float(h["frac_sig_fdr" if new else "frac_sig"].iloc[0]) if len(h) else np.nan)
            for sub in SUBS:
                t = T[(T.key == f"{st}/{sub}/lat") & (T.epoch == ep) & (T.kind == "level")]
                row[f"session_{sub}_tfce"] = float(t.frac_sig.iloc[0]) if len(t) else np.nan
                row[f"session_{sub}_n_animals"] = int(t.n_animals.iloc[0]) if len(t) else 0
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures


def _rowlab(k: str) -> str:
    st, sub, tg = k.split("/")
    if tg.startswith("sp_"):
        return f"{SETLAB[st]} {tg[3:].replace('_', ' ')}"
    return f"{SETLAB[st]}, {sub} licks"


def _sig_contours(ax, r):
    sig = np.nan_to_num(r["p"], nan=1) < 0.05
    if sig.any():
        ax.contour(sig.astype(float), levels=[0.5], colors="k", linewidths=1.2)
        s3 = sig & (np.nan_to_num(r["agree"]) >= 3)
        if s3.any() and not (s3 == sig).all():
            ax.contour(s3.astype(float), levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
    cx = np.isfinite(r["p"])
    return 100 * sig.sum() / cx.sum()


def _head(fam: str, change: bool, n_perm: int) -> str:
    tg = "sp" if fam.startswith("sp_") else fam
    what = (f"{SETLAB[fam[3:]]} per-spout one-vs-rest maps (all licks), canonical spout order" if tg == "sp"
            else f"{TGLAB[tg]} maps; rows = lick set x subset (FIRST = ITI: no lick in the preceding 2 s; cued: first post-cue lick)")
    unit = "pattern / animal's mean pre-session RMS of the all-lick map (amplitude KEPT; one scale per row)"
    if change:
        test = (f"Col 1 = pre LEVEL (purple-orange, own scale), cols 2-4 = epoch - pre (red-blue). SOLID = TFCE FWER "
                f"0.05: per animal (mean epoch - mean pre) / within-animal session SD pooled over all animals and groups, "
                f"summed; every animal with >= 1 session per group counts; session labels permuted within animal ({n_perm}x); "
                "DASHED = also >= 3 animals share the sign.")
    else:
        test = (f"SOLID = TFCE FWER 0.05 vs 0: per animal mean map / within-animal session SD pooled over all animals, "
                f"summed; every animal with >= 1 session counts; null = sign-flip of each SESSION map ({n_perm}x, max |TFCE|); DASHED = also >= 3 animals share the sign. Title: n sessions / animals, % cortex "
                "significant, global share = (cortex mean)^2 / mean square.")
    scl = (" Z-SCORED FEATURES: each SVD component / its session frame-wise SD (iti_basis_frame_sd) before mapping."
           if SCALE else "")
    paras = [f"NEURAL: per-SESSION self-fit Haufe {'CHANGE' if change else 'LEVEL'} maps -- {what}. Mean of animals; "
             f"{unit}.{scl}", test,
             f"SIGN: {SIGN[tg]}. ipsi (lesion) = image LEFT. PROVISIONAL"]
    return "\n".join(textwrap.fill(q, 150) for q in paras)


def plot_levels(D: dict) -> None:
    R, fam, keys, ol, n_perm = D["R"], D["fam"], D["keys"], D["ol"], D["n_perm"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in keys if any((k, e, "level") in R for e in EP)]
    # within-spout FIRST-lick rows are not drawn (Priya 10-27): a single-session within-spout decoder on first licks
    # fits ~nothing (median fit share 0.06-0.18), most sessions are dropped and most cells are < 3 animals
    keys = [k for k in keys if not (fam == "wlat" and "/first/" in k)]
    fig, axs = plt.subplots(len(keys), 4, figsize=(13.5, 2.75 * len(keys) + 1.3), squeeze=False, layout="constrained")
    for i, k in enumerate(keys):
        lim = sym_lim([R[(k, e, "level")]["point"] for e in EP if (k, e, "level") in R])
        im = None
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            r = R.get((k, ep, "level"))
            if r is None:
                ax.set_title(f"{ep}: no data", fontsize=7.5)
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                continue
            im = ax.imshow(r["point"], cmap=PRE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            fs = _sig_contours(ax, r)
            pt = r["point"][np.isfinite(r["point"])]
            ax.set_title(f"{ep}: {r['n_sess']} sess / {r['n']} an\nsig {fs:.0f}%, global {gshare(pt):.2f}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(_rowlab(k), fontsize=8.5)
        if im is not None:
            cb = fig.colorbar(im, ax=axs[i, :].tolist(), fraction=0.02, pad=0.01)
            cb.set_label("LEVEL / pre RMS", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(_head(fam, False, n_perm), fontsize=8)
    save_png_svg(fig, FIG / LEVEL_FIG[fam], dpi=95)
    plt.close(fig)


def plot_change(D: dict) -> None:
    R, fam, keys, ol, n_perm = D["R"], D["fam"], D["keys"], D["ol"], D["n_perm"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in keys if (k, "pre", "level") in R]
    # within-spout FIRST-lick rows are not drawn (Priya 10-27): a single-session within-spout decoder on first licks
    # fits ~nothing (median fit share 0.06-0.18), most sessions are dropped and most cells are < 3 animals
    keys = [k for k in keys if not (fam == "wlat" and "/first/" in k)]
    fig, axs = plt.subplots(len(keys), 4, figsize=(13.5, 2.75 * len(keys) + 1.3), squeeze=False, layout="constrained")
    for i, k in enumerate(keys):
        lim_c = sym_lim([R[(k, e, "change")]["point"] for e in EP[1:] if (k, e, "change") in R])
        im_l = im_c = None
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            r = R.get((k, ep, "level" if ep == "pre" else "change"))
            if r is None:
                ax.set_title(f"{ep} - pre: no data", fontsize=7.5)
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                continue
            if ep == "pre":
                lim = sym_lim([r["point"]])
                im_l = ax.imshow(r["point"], cmap=PRE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
            else:
                im_c = ax.imshow(r["point"], cmap=CHANGE_CMAP, vmin=-lim_c, vmax=lim_c, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            fs = _sig_contours(ax, r)
            cx = np.isfinite(r["p"])
            sig = (np.nan_to_num(r["p"], nan=1) < 0.05)
            if ep == "pre":
                ax.set_title(f"pre (level): {r['n_sess']} sess / {r['n']} an\nsig {fs:.0f}%", fontsize=7.5)
            else:
                ax.set_title(f"{ep} - pre: {r['n_sess']} sess (pre + {ep}) / {r['n']} an\nsig {fs:.0f}% "
                             f"(+{100 * (sig & (r['point'] > 0)).sum() / cx.sum():.0f} / "
                             f"-{100 * (sig & (r['point'] < 0)).sum() / cx.sum():.0f})", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(_rowlab(k), fontsize=8.5)
        if im_l is not None:
            cb = fig.colorbar(im_l, ax=axs[i, 0], fraction=0.06, pad=0.02)
            cb.set_label("pre LEVEL / pre RMS", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
        if im_c is not None:
            cb = fig.colorbar(im_c, ax=axs[i, 1:].tolist(), fraction=0.025, pad=0.01)
            cb.set_label("CHANGE / pre RMS", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(_head(fam, True, n_perm), fontsize=8)
    save_png_svg(fig, FIG / CHANGE_FIG[fam], dpi=95)
    plt.close(fig)


FAMS = ["lat", "sag", "wlat", "sp_trial", "sp_iti"]


def draw(D_by_fam: dict) -> None:
    for fam in FAMS:
        plot_levels(D_by_fam[(fam, "level")])
        plot_change(D_by_fam[(fam, "change")])


# ------------------------------------------------------------------------------------------------ main


def main() -> int:
    if plot_only():
        draw({(fam, kd): load_fd((LEVEL_FIG if kd == "level" else CHANGE_FIG)[fam]) for fam in FAMS
              for kd in ("level", "change")})
        return 0
    import time
    n_perm = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else N_PERM
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 20
    OUTD.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = patterns(jobs)
    print(f"patterns: {len(P)} sessions, {time.time() - t0:.0f} s", flush=True)
    grid = imaging_grid()
    to_img, ol = grid_fns(grid)
    npix = len(grid["pix"])
    g = to_img(np.arange(npix, dtype=float))
    ok = np.isfinite(g)
    sel = g[ok].astype(int)
    idx_img = np.full(g.shape, -1)
    idx_img[ok] = np.arange(ok.sum())
    L = build_maps(P, sel)
    Dr = pd.DataFrame(DROPPED, columns=["animal", "key", "label", "epoch"])
    Dr.to_csv(OUTD / "dropped.csv", index=False)
    print(f"\n== session maps dropped (fit share < {MIN_FIT}), per key x epoch")
    print(Dr.groupby(["key", "epoch"]).size().unstack("epoch").reindex(columns=EP).fillna(0).astype(int).to_string()
          if len(Dr) else "none")

    # first-lick counts (licks per session in each subset) + sessions skipped
    C = []
    for x, r in sorted(P.items()):
        for st in SETS:
            for sub in SUBS:
                C.append(dict(label=x, animal=r["animal"], epoch=r["epoch"], set=SETLAB[st], sub=sub,
                              licks=r["n_set"].get(f"{st}/{sub}", 0), has_lat_map=f"{st}/{sub}/lat" in r["A"],
                              far1=r["iti_first_far1"][0] if st == "iti" and sub == "first" else np.nan))
    C = pd.DataFrame(C)
    C.to_csv(OUTD / "first_lick_counts.csv", index=False)
    pd.set_option("display.width", 240)
    print(f"\n== licks per session by subset (median [min-max]) and sessions with a map (>= {MIN_LICKS} licks)")
    print(C.groupby(["set", "sub", "animal", "epoch"]).agg(
        n=("licks", "size"), med=("licks", "median"), lo=("licks", "min"), hi=("licks", "max"),
        with_map=("has_lat_map", "sum")).unstack("epoch").reindex(columns=EP, level=1).to_string())
    F = C[(C.set == "ITI") & (C["sub"] == "first")].groupby("epoch")[["far1", "licks"]].sum().reindex(EP)
    F["share_far1"] = F.far1 / F.licks
    Fa = C[(C.set == "ITI") & (C["sub"] == "first")].groupby(["animal", "epoch"])[["far1", "licks"]].sum()
    Fa = (Fa.far1 / Fa.licks).unstack("epoch").reindex(columns=EP)
    print("\n== ITI FIRST licks (gap >= 2 s): share >= 1 s from both the response-window end and the next spout arrival")
    print(F.round(3).to_string())
    print(Fa.round(3).to_string())

    A = per_animal(L)
    A.to_csv(OUTD / "per_animal.csv", index=False)
    cols = ["n_sess", "licks_med", "gain", "rms_mean", "rms_debiased", "rms_nonglobal", "r_pre", "global_share",
            "global_share_debiased", "fit_med", "gain_enc", "rms_enc_mean", "global_share_enc", "r_pre_enc"]
    for tg in ("lat", "wlat"):
        print(f"\n== PER ANIMAL (rule 8), target {tg}: amplitude in units of the animal's pre all-lick session RMS")
        print(A[A.target == tg].set_index(["set", "sub", "animal", "epoch"])[cols].round(3).to_string())

    f = OUTD / "tests.pkl"
    t1 = time.time()
    if "--reuse-tests" in sys.argv and f.exists():
        R = pickle.loads(f.read_bytes())
    else:
        R = run_tests(L, idx_img, n_perm, jobs)
        f.write_bytes(pickle.dumps(R, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"tests: {time.time() - t1:.0f} s", flush=True)
    print_counts(R, "per-session LEVEL / CHANGE tests (amplitude kept)")
    T = tests_table(R)
    T.to_csv(OUTD / "tests.csv", index=False)
    print("\n== TFCE tests (fraction of cortex p_FWER < 0.05)")
    print(T.round(3).to_string(index=False))
    H = heldout_comparison(T)
    H.to_csv(OUTD / "heldout_comparison.csv", index=False)
    print("\n== LATERAL level maps: % cortex significant, per-session (TFCE) vs held-out (max-stat FWER / BH)")
    print(H.round(3).to_string(index=False))

    Dall = {}
    for fam in FAMS:
        keys = fam_keys(fam)
        Rf = {kk: v for kk, v in R.items() if kk[0] in keys}
        for kd, name in (("level", LEVEL_FIG), ("change", CHANGE_FIG)):
            D = dict(R=Rf, fam=fam, keys=keys, ol=ol, n_perm=n_perm)
            save_fd(name[fam], D)
            Dall[(fam, kd)] = D
    draw(Dall)
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
