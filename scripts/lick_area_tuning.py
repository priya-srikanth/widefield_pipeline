"""PER-AREA, PER-HEMISPHERE LICK-DIRECTION TUNING on MATCHED licks (Priya 10-29, item 1).

QUESTION. The cued within-spout balance (lick_balance_matched_error.py: post licks vs pre licks to the SAME spout landing at
the SAME angle) rises ABOVE pre over recovery (+0.13 subacute / +0.16 chronic, 4/4), and the ITI set keeps a lasting
shift. Is that a GAIN change (steeper tuning), an additive OFFSET (whole tuning curve up), or BROADER (flatter) tuning --
and in which area / hemisphere?

LICKS. <OUT_W>/baltarget_licks.csv.gz (lick_balance_target_model.py): cued = first post-cue lick; ITI = first lick after
>= 2 s quiet (pos = the spout in place = the PREVIOUS trial's spout). ITI_far1 = the ITI licks >= 1 s after the trial's
response-window end AND >= 1 s before the next spout arrival (iti_timing_audit.csv, the far1 rule of iti_lick_order; ~ the
spout docked), joined on the exact DAQ peak time (covref_covariates t). Pre = AUGUST pre sessions only (as baltarget).
RESPONSE per lick, area, hemisphere: mean over lags 0.1..0.5 s from the lick PEAK minus mean over -1.0..-0.5 s (the
session files' iti_X / tr_X, peak-aligned on iti_licks.LAGS_D -- exactly the primary balance's window; NB peak, not onset),
dF/F x1e3, Allen-area / ROI means (split basis: ALM, tjM1 = the v8 ROIs; MOp_rest = MOp minus tjM1). Areas: ALM, tjM1,
SSp-m, SSp-n, SSs, MOp_rest, controls VISpm (VISp has no left ROI in the session files), RSPd; POOL = ALM + tjM1 + SSp-m + SSp-n + SSs (the primary pool; validation:
its POOL rows must reproduce baltarget c / i and balmatch_sessions bal_spout).

SIGN CONVENTIONS. angle + = CONTRALESIONAL side (DIRECTION sign). hemisphere c = contralesional (image right, '_right'),
i = ipsilesional ('_left'); every 'c-i' quantity is + when the CONTRALESIONAL hemisphere is relatively larger
(HEMISPHERE sign).

NO EXTRAPOLATION. Per set x animal, a lick is kept only if its angle lies inside the 2.5-97.5 percentile range of the
animal's PRE licks to the same spout (applied to every epoch, pre included; as lick_balance_matched_error).

PER SESSION x area x hemisphere (kept licks, >= MIN_LICKS):
  pre-referenced (primary; matched target + landed angle). pred = the animal's PRE within-spout OLS line of the response
  on angle (pre sessions leave-one-session-out; < 10 pre licks at a spout -> spout mean):
    lratio   log2(sum y / sum pred)                      relative LEVEL vs pre at matched spout + angle (NaN if mean pred < MIN_PRED = 3)
    shift    mean(y - pred)                              additive level change (dF/F x1e3)
    rshift   shift / mean(pred)                          the same as a fraction of the pre-predicted response
    depth    g from OLS  y = A + g (pred - mean pred)    tuning DEPTH relative to pre (1 = unchanged; > 1 steeper /
                                                         gain; < 1 flatter / BROADER); minus 1 in the tables (depth - 1)
  -> a pure MULTIPLICATIVE GAIN moves rshift and depth-1 together (equal); a pure additive OFFSET moves rshift with
     depth-1 = 0; BROADER tuning = depth-1 < 0. The c-i lratio of POOL is the matched-error balance (bal_spout).
  raw tuning (descriptive). OLS of y on (angle - ref) / 10 over the session's kept licks, ref = the animal's pre mean
  angle of the set: slope (per 10 deg), off_ref (response at ref), amp (mean response); spout means (spout x session).
CHANGE VS PRE: per animal first (rule 8: difference of session means, n of animals with the pooled sign), then
analysis_kit.boot_delta (paired nested animals -> sessions; rng seed 0 per test). c-i = per-session c minus i.
OUTPUTS (<OUT_W>): areatune_licks.csv.gz (per lick area responses + flags), areatune_sessions.csv, areatune_spoutmeans.csv,
areatune_animals.csv (per animal x epoch level + change), areatune_tests.csv (pooled change, CI, n same sign, per-animal
columns), areatune_validation.csv; figures figures/lick_area_tuning_{curves_cued,curves_iti,curves_iti_far1,decomp,
pool}.png. --plot-only redraws from the CSVs; --reuse-licks reuses areatune_licks.csv.gz.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = Path(il.OUT_W)
if il.ITI_SUBSET:
    raise SystemExit("run without WFIELD_ITI_FILTER (this script defines its own far1 subset)")
FIG = OUT / "figures"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POST = EPS[1:]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs", "MOp_rest", "VISpm", "RSPd"]   # VISp has no left ROI (masked)
POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
ALLA = ["POOL", *AREAS]
SETS = {"trial": "CUED first post-cue lick", "iti": "ITI first lick after >= 2 s quiet",
        "iti_far1": "ITI first lick, far1 subset (>= 1 s from trial stop and next spout arrival)"}
HEMI = {"c": "right", "i": "left"}
Q = (2.5, 97.5)
MIN_LICKS = 10
MIN_PRED = 3.0          # dF/F x1e3: below this the ratio measures (lratio, rshift) are NaN (control areas in the ITI)
METRICS = ["lratio", "shift", "rshift", "depth1", "slope", "off_ref", "amp"]
COL = {"pre": "0.45", "acute": "#C44E52", "subacute": "#DD8452", "chronic": "#4C72B0"}
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


def _wm(lo_, hi_):
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & (L <= hi_ + 1e-9) if hi_ <= 0 else (L >= lo_ - 1e-9) & (L < hi_ - 1e-9)


# ------------------------------------------------------------------------------------------------ per-lick responses
def _session(label: str) -> pd.DataFrame:
    """One session (module-level for spawn): per prediction row (set, k) the response of every area x hemisphere
    (same lick match and window as lick_balance_target_model._session)."""
    import lick_balance_reference as lbr
    acts = lbr.cv._acts(label)
    d = il._load(label)
    base_m, win = _wm(*il.BASE_RESP), _wm(0.1, 0.5)
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        ok, mm = m >= 0, np.maximum(m, 0)
        X = d[xk].astype(float)
        v = X[:, win].mean(1) - X[:, base_m].mean(1)              # (licks, cols)
        v, cols = il.with_rest(v, d["cols"], label, basis="split")
        R = {"set": st, "label": label, "k": np.arange(len(act))}
        for a in AREAS:
            for hk, h in HEMI.items():
                j = cols.index(f"{a}_{h}") if f"{a}_{h}" in cols else None
                R[f"{a}|{hk}"] = np.where(ok, v[mm, j], np.nan) if j is not None else np.nan
        out.append(pd.DataFrame(R))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_licks() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    L = pd.read_csv(OUT / "baltarget_licks.csv.gz")
    labels = sorted(L.label.unique())
    res, fail = ak.fan_sessions(labels, _session, jobs=4)
    if fail:
        raise RuntimeError(f"area responses failed: {fail[:5]}")
    R = pd.concat([v for _i, v in res], ignore_index=True)
    L = L.merge(R, on=["set", "label", "k"], how="left", validate="one_to_one")
    C = pd.read_csv(OUT / "covref_covariates.csv.gz", usecols=["set", "label", "k", "t"])
    L = L.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    A = pd.read_csv(OUT / "iti_timing_audit.csv", usecols=["label", "t", "since_stop", "to_strobe"])
    A["tr"] = A.t.round(4)
    A = A.drop_duplicates(["label", "tr"]).drop(columns="t")
    L["tr"] = L.t.round(4)
    L = L.merge(A, on=["label", "tr"], how="left", validate="many_to_one").drop(columns="tr")
    L["far1"] = (L.since_stop >= 1.0) & (L.to_strobe >= 1.0)
    for hk in HEMI:
        L[f"POOL|{hk}"] = L[[f"{a}|{hk}" for a in POOL]].sum(1, min_count=len(POOL))
    return L


def validate(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hk in HEMI:
        e = (L[f"POOL|{hk}"] - L[hk]).abs()
        rows.append({"check": f"POOL|{hk} vs baltarget {hk}", "n": int(e.notna().sum()), "max_abs_err": e.max(),
                     "sd": L[hk].std()})
    it = L[L.set == "iti"]
    rows.append({"check": "ITI licks joined to timing audit", "n": len(it), "frac": it.since_stop.notna().mean()})
    for ep in EPS:
        rows.append({"check": f"ITI far1 fraction {ep}", "n": int((it.epoch == ep).sum()),
                     "frac": it[it.epoch == ep].far1.mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ per-session metrics
def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 10 or np.std(x) == 0:
        return 0.0, float(np.mean(y)) if len(y) else np.nan
    b1, b0 = np.polyfit(x, y, 1)
    return b1, b0


def _pred(g: pd.DataFrame, ycols: list[str]) -> pd.DataFrame:
    """Within-spout pre line per spout x response column; pre sessions leave-one-session-out."""
    P = pd.DataFrame(np.nan, index=g.index, columns=ycols)
    pre = g[g.epoch == "pre"]
    for p in g.pos.unique():
        sel = g.pos == p
        pr = pre[pre.pos == p]
        for lab in [None, *pr.label.unique()]:
            tr = pr[pr.label != lab] if lab else pr
            rows = sel & ((g.label == lab) if lab else (g.epoch != "pre"))
            if not rows.any():
                continue
            x = g.loc[rows, "angle"].to_numpy(float)
            for y in ycols:
                b1, b0 = _line(tr.angle.to_numpy(float), tr[y].to_numpy(float))
                P.loc[rows, y] = b0 + b1 * x
    return P


def _metrics(y: np.ndarray, p: np.ndarray, x: np.ndarray) -> dict:
    sy, sp = y.sum(), p.sum()
    # relative measures only where the pre-predicted response is a real response (mean >= MIN_PRED dF/F x1e3)
    ok = p.mean() >= MIN_PRED
    o = {"lratio": np.log2(sy / sp) if ok and sy > 0 else np.nan, "shift": float(np.mean(y - p))}
    pb = p.mean()
    o["rshift"] = o["shift"] / pb if ok else np.nan
    pc = p - pb
    o["depth1"] = float(pc @ (y - y.mean()) / (pc @ pc)) - 1 if pc @ pc > 0 else np.nan
    b1, b0 = np.polyfit(x, y, 1)
    o.update(slope=b1, off_ref=b0, amp=float(y.mean()))
    return o


def sessions(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ycols = [f"{a}|{h}" for a in ALLA for h in HEMI]
    srows, mrows, kept = [], [], []
    for st in SETS:
        base = L[L.set == ("trial" if st == "trial" else "iti")]
        if st == "iti_far1":
            base = base[base.far1]
        for an, g in base.groupby("animal"):
            g = g.dropna(subset=["angle", "pos", *ycols]).copy()
            pre = g[g.epoch == "pre"]
            lo = pre.groupby("pos").angle.quantile(Q[0] / 100)
            hi = pre.groupby("pos").angle.quantile(Q[1] / 100)
            g["inside"] = (g.angle >= g.pos.map(lo)) & (g.angle <= g.pos.map(hi))
            ref = float(pre.angle.mean())
            for ep, h in g.groupby("epoch"):
                kept.append({"set": st, "animal": an, "epoch": ep, "n": len(h), "frac_inside": h.inside.mean()})
            P = _pred(g, ycols)
            m = g[g.inside]
            for (lab, ep), h in m.groupby(["label", "epoch"]):
                if len(h) < MIN_LICKS:
                    continue
                x = (h.angle.to_numpy(float) - ref) / 10
                r = {"set": st, "animal": an, "label": lab, "epoch": ep, "day": h.day.iloc[0], "n": len(h)}
                for a in ALLA:
                    for hk in HEMI:
                        c = f"{a}|{hk}"
                        for k, v in _metrics(h[c].to_numpy(float), P.loc[h.index, c].to_numpy(float), x).items():
                            r[f"{k}|{a}|{hk}"] = v
                    for k in METRICS:
                        r[f"{k}|{a}|c-i"] = r[f"{k}|{a}|c"] - r[f"{k}|{a}|i"]
                srows.append(r)
                for pos, hp in h.groupby("pos"):
                    mr = {"set": st, "animal": an, "label": lab, "epoch": ep, "pos": pos, "n": len(hp),
                          "angle": hp.angle.mean(), "pre_angle": pre[pre.pos == pos].angle.mean()}
                    for c in ycols:
                        mr[c] = hp[c].mean()
                    mrows.append(mr)
        print(f"[{st}] sessions done", flush=True)
    return pd.DataFrame(srows), pd.DataFrame(mrows), pd.DataFrame(kept)


def tests(S: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    vcols = [c for c in S.columns if c.count("|") == 2]
    arows, trows = [], []
    for st, g in S.groupby("set"):
        for c in vcols:
            met, a, hk = c.split("|")
            pre = {an: h[c].dropna().to_numpy() for an, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
            pre = {an: v for an, v in pre.items() if len(v)}
            for an, v in pre.items():
                arows.append({"set": st, "metric": met, "area": a, "hemi": hk, "animal": an, "epoch": "pre",
                              "level": v.mean(), "change": 0.0, "n_sess": len(v)})
            for ep in POST:
                post = {an: h[c].dropna().to_numpy() for an, h in sorted(g[g.epoch == ep].groupby("animal"))}
                post = {an: v for an, v in post.items() if len(v) and an in pre}
                if not post:
                    continue
                am = {an: post[an].mean() - pre[an].mean() for an in post}
                for an, v in post.items():
                    arows.append({"set": st, "metric": met, "area": a, "hemi": hk, "animal": an, "epoch": ep,
                                  "level": v.mean(), "change": am[an], "n_sess": len(v)})
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                if ci is None:
                    continue
                sg = np.sign(ci.point)
                trows.append({"set": st, "metric": met, "area": a, "hemi": hk, "epoch": ep, "point": ci.point,
                              "lo": ci.lo, "hi": ci.hi, "n_same_sign": int(sum(np.sign(v) == sg for v in am.values())),
                              "n_animals": len(am), "excl0": bool(ci.lo > 0 or ci.hi < 0),
                              **{f"d_{an}": v for an, v in am.items()}})
        print(f"[{st}] tests done", flush=True)
    return pd.DataFrame(arows), pd.DataFrame(trows)


# ------------------------------------------------------------------------------------------------ figures
def _wrap(s: str, w: int = 60) -> str:
    return "\n".join(textwrap.fill(p, w) for p in s.split("\n"))


def _clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def plot_curves(M: pd.DataFrame, st: str) -> None:
    import matplotlib.pyplot as plt
    m = M[M.set == st]
    if m.empty:
        return
    fig, axs = plt.subplots(3, len(ALLA), figsize=(2.55 * len(ALLA), 8.4), squeeze=False)
    for j, a in enumerate(ALLA):
        for i, hk in enumerate(["c", "i", "c-i"]):
            ax = axs[i, j]
            for ep in EPS:
                e = m[m.epoch == ep]
                if e.empty:
                    continue
                y = e[f"{a}|c"] - e[f"{a}|i"] if hk == "c-i" else e[f"{a}|{hk}"]
                # animal mean of session means per spout, then mean over animals (x = animal's pre spout angle)
                t = (e.assign(y=y).groupby(["animal", "pos"]).agg(y=("y", "mean"), x=("pre_angle", "first"))
                     .groupby("pos").agg(y=("y", "mean"), x=("x", "mean"), n=("y", "size")).reindex(POS).dropna())
                ax.plot(t.x, t.y, "-o", color=COL[ep], ms=3.5, lw=1.4, label=ep)
            ax.axhline(0, color="k", lw=0.4)
            if i == 0:
                ax.set_title(f"{a}", fontsize=9)
            if j == 0:
                ax.set_ylabel({"c": "CONTRA hemisphere\nresponse (dF/F x1e3)", "i": "IPSI hemisphere\nresponse",
                               "c-i": "contra - ipsi\n(+ = contra larger)"}[hk], fontsize=8)
            if i == 2:
                ax.set_xlabel("pre mean angle of spout (deg,\n+ = contralesional)", fontsize=7)
            ax.tick_params(labelsize=7)
            _clean(ax)
    hd, lb = axs[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=len(lb), fontsize=8, frameon=False)
    fig.suptitle(_wrap(f"{SETS[st]}: post-lick response (0.1-0.5 s minus -1..-0.5 s) by spout, kept licks only (inside "
                       "the pre angle range for their spout); spouts placed at the animal's PRE mean angle; mean over "
                       "animals of animal means. Canonical spouts left to right = ipsilesional to contralesional. "
                       "PROVISIONAL", 170), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    save_png_svg(fig, FIG / f"lick_area_tuning_curves_{st}.png", dpi=90)
    plt.close(fig)


def _bars(ax, T: pd.DataFrame, st: str, met: str, hk: str, areas: list[str], big: bool = False) -> None:
    t = T[(T.set == st) & (T.metric == met) & (T.hemi == hk)]
    x = np.arange(len(areas))
    for k, ep in enumerate(POST):
        e = t[t.epoch == ep].set_index("area").reindex(areas)
        xx = x + (k - 1) * 0.25
        ax.errorbar(xx, e.point, yerr=[e.point - e.lo, e.hi - e.point], fmt="o", color=COL[ep],
                    ms=5 if big else 4, capsize=2, label=ep)
        for an in AN:
            if f"d_{an}" in e:
                ax.plot(xx + 0.06, e[f"d_{an}"], MK[an], mfc="none", color=COL[ep], ms=3.5 if big else 2.5,
                        alpha=0.6, lw=0)
        for xi, (_, r) in zip(xx, e.iterrows()):
            if np.isfinite(r.point):
                ax.text(xi, r.hi, f"{int(r.n_same_sign)}", ha="center", va="bottom",
                        fontsize=7.5 if big else 5.5, color=COL[ep])
    ax.axhline(0, color="k", lw=0.5)
    if big:
        for xb in x[:-1] + 0.5:
            ax.axvline(xb, color="0.9", lw=0.6, zorder=0)
    ax.set_xticks(x, areas, rotation=45 if not big else 30, ha="right", fontsize=9 if big else 7)
    ax.tick_params(axis="y", labelsize=8.5 if big else 7)
    _clean(ax)


DECOMP = [("lratio", "c-i", "SHARE: d log2(post/pre-predicted), contra - ipsi\n(POOL = matched-error balance; "
           "+ = contralesional hemisphere gains)"),
          ("rshift", "c", "LEVEL, contralesional hemisphere: d mean(y - pre pred) / mean pred\n(fraction; + = larger "
           "than pre at same spout + angle)"),
          ("rshift", "i", "LEVEL, ipsilesional hemisphere\n(same as left)"),
          ("depth1", "c", "TUNING DEPTH, contralesional: d (g - 1), y = A + g (pred - mean)\n(+ steeper / gain, "
           "- flatter / broader)"),
          ("depth1", "i", "TUNING DEPTH, ipsilesional\n(same as left)")]


def _fig_legend(fig, ax) -> None:
    """One legend for the figure, below the panels (never over data)."""
    from matplotlib.lines import Line2D
    hd, lb = ax.get_legend_handles_labels()
    hd = list(hd) + [Line2D([], [], marker=MK[a], ls="", mfc="none", color="0.4", ms=5) for a in AN]
    lb = list(lb) + [f"{a} (open)" for a in AN]
    fig.legend(hd, lb, loc="lower center", ncol=len(lb), fontsize=8.5, frameon=False)


# _simple variants: cued + ITI-after-quiet rows only (far1 subset dropped), the orofacial areas + POOL + one
# control (VISpm), bigger markers / fonts.  Plotting-only selection of the same test table.
SIMPLE_AREAS = ["POOL", "ALM", "tjM1", "SSp-m", "SSp-n", "SSs", "VISpm"]
SIMPLE_SETS = ["trial", "iti"]


def plot_decomp(T: pd.DataFrame, simple: bool = False) -> None:
    import matplotlib.pyplot as plt
    sets = [s for s in SETS if s in set(T.set) and (not simple or s in SIMPLE_SETS)]
    areas = [a for a in ALLA if a in SIMPLE_AREAS] if simple else ALLA
    w, h = (5.4, 4.6) if simple else (4.6, 3.9)
    fig, axs = plt.subplots(len(sets), len(DECOMP), figsize=(w * len(DECOMP), h * len(sets)), squeeze=False)
    for i, st in enumerate(sets):
        for j, (met, hk, ttl) in enumerate(DECOMP):
            ax = axs[i, j]
            _bars(ax, T, st, met, hk, areas, big=simple)
            ax.set_title(_wrap(f"{SETS[st]}\n{ttl}", 62), fontsize=8.5 if simple else 7.5)
    _fig_legend(fig, axs[0, 0])
    fig.suptitle(_wrap("Change vs pre of the matched-lick tuning per area (licks inside the pre angle range of their spout; "
                       "reference = the animal's PRE within-spout line, pre LOSO). Gain = LEVEL and DEPTH move together; "
                       "additive offset = LEVEL moves, DEPTH ~0; broader tuning = DEPTH < 0. Filled = pooled change, 95% "
                       "paired nested bootstrap; open = animals; number = animals with the pooled sign. Hemisphere: c = "
                       "contralesional (image right), i = ipsilesional. PROVISIONAL", 200), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04 if simple else 0.03, 1, 0.92 if simple else 0.94))
    save_png_svg(fig, FIG / f"lick_area_tuning_decomp{'_simple' if simple else ''}.png", dpi=90)
    plt.close(fig)


# _dots variant (Priya 10-29: decomp_simple too busy, colours off-standard): key areas only, three terms, cued + ITI
# rows, standard epoch colours (orofacial_cohort.EPOCH_LINE), per-animal dots + pooled 95% CI.  Plotting only.
DOT_AREAS = [("ALM", "c"), ("tjM1", "c"), ("SSp-m", "c"), ("SSp-n", "c"), ("VISpm", "c")]
DOT_TERMS = [("rshift", "c", "OFFSET / LEVEL (contra hemisphere)",
              "change in level vs pre at matched\nspout + angle (fraction of pre-predicted)"),
             ("lratio", "c-i", "CONTRA SHARE (contra - ipsi)",
              "change in log2(post / pre-predicted),\ncontra - ipsi (+ = contra gains)"),
             ("depth1", "c", "TUNING DEPTH (contra hemisphere)",
              "change in tuning depth (g - 1)\n(+ steeper / gain, - flatter / broader)")]


def plot_decomp_dots(T: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    sets = [s for s in SIMPLE_SETS if s in set(T.set)]
    fig, axs = plt.subplots(len(sets), len(DOT_TERMS), figsize=(4.6 * len(DOT_TERMS), 3.7 * len(sets) + 1.6),
                            squeeze=False, layout="constrained")
    x = np.arange(len(DOT_AREAS))
    for i, st in enumerate(sets):
        for j, (met, hk, ttl, yl) in enumerate(DOT_TERMS):
            ax = axs[i, j]
            ax.axvspan(len(DOT_AREAS) - 1.5, len(DOT_AREAS) - 0.5, color="0.93", zorder=0)
            for k, ep in enumerate(POST):
                for xi, (a, _h) in zip(x, DOT_AREAS):
                    r = T[(T.set == st) & (T.metric == met) & (T.area == a) & (T.hemi == hk) & (T.epoch == ep)]
                    if r.empty or not np.isfinite(r.point.iloc[0]):
                        continue
                    r = r.iloc[0]
                    xx = xi + (k - 1) * 0.26
                    for an in AN:
                        v = r.get(f"d_{an}", np.nan)
                        if np.isfinite(v):
                            ax.plot(xx - 0.07, v, ANIMAL_MARKER[an], mfc="none", mec=EPOCH_LINE[ep], ms=4, alpha=0.75)
                    ax.errorbar(xx + 0.05, r.point, yerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o",
                                color=EPOCH_LINE[ep], ms=5.5, lw=1.6, capsize=0)
                    if int(r.n_animals) < 4:
                        ax.text(xx + 0.05, r.hi, f"n={int(r.n_animals)}", fontsize=6, ha="center", va="bottom",
                                color=EPOCH_LINE[ep])
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x, [a if a != "VISpm" else "VISpm\n(control)" for a, _h in DOT_AREAS], fontsize=8.5)
            ax.set_xlim(-0.5, len(DOT_AREAS) - 0.5)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.set_title(f"{'CUED first lick' if st == 'trial' else 'ITI first lick after quiet'}: {ttl}",
                         fontsize=9)
            _clean(ax)
    hd = [Line2D([], [], marker="o", ls="-", color=EPOCH_LINE[ep], ms=5.5) for ep in POST]
    hd += [Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", color="0.4", ms=5) for a in AN]
    fig.legend(hd, [f"{ep} - pre (pooled, 95% CI)" for ep in POST] + [f"{a}" for a in AN],
               loc="outside lower center", ncol=7, fontsize=8.5, frameon=False)
    fig.suptitle(_wrap("Matched-lick tuning change vs pre per area (licks to the same spout landing at the same angle as "
                       "pre; contralesional hemisphere unless stated). A pure additive OFFSET moves the level with depth "
                       "~0; a GAIN moves level and depth together; BROADER tuning = depth < 0. Filled = pooled change "
                       "(paired nested bootstrap 95% CI); open = animals; 0 = pre. VISpm (gray) = non-orofacial control; "
                       "n = animals when < 4 (ratio terms undefined where the pre response < 3 dF/F x1e3). PROVISIONAL "
                       "(n = 4)", 170), fontsize=9.5)
    save_png_svg(fig, FIG / "lick_area_tuning_decomp_dots.png", dpi=100)
    plt.close(fig)


RAW = [("slope", "c", "raw SLOPE, contralesional (per 10 deg)"), ("slope", "i", "raw SLOPE, ipsilesional"),
       ("off_ref", "c", "raw OFFSET at the pre mean angle, contra"), ("off_ref", "i", "raw OFFSET, ipsi"),
       ("shift", "c-i", "matched additive SHIFT, contra - ipsi\n(dF/F x1e3)")]


def plot_raw(T: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    sets = [s for s in SETS if s in set(T.set)]
    fig, axs = plt.subplots(len(sets), len(RAW), figsize=(4.6 * len(RAW), 3.9 * len(sets)), squeeze=False)
    for i, st in enumerate(sets):
        for j, (met, hk, ttl) in enumerate(RAW):
            ax = axs[i, j]
            _bars(ax, T, st, met, hk, ALLA)
            ax.set_title(_wrap(f"{SETS[st]}\n{ttl}", 62), fontsize=7.5)
    _fig_legend(fig, axs[0, 0])
    fig.suptitle(_wrap("Raw tuning lines per session (OLS of the response on lick angle over the kept licks; angle + = "
                       "contralesional): change vs pre of the SLOPE (gain) and the OFFSET at the animal's pre mean angle "
                       "(shift), dF/F x1e3. Note: raw slope mixes target and landed angle. PROVISIONAL", 200),
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save_png_svg(fig, FIG / "lick_area_tuning_raw.png", dpi=90)
    plt.close(fig)


def plot_all() -> None:
    import matplotlib
    matplotlib.use("Agg")
    T = pd.read_csv(OUT / "areatune_tests.csv")
    M = pd.read_csv(OUT / "areatune_spoutmeans.csv")
    for st in SETS:
        plot_curves(M, st)
    plot_decomp(T)
    plot_decomp(T, simple=True)
    plot_decomp_dots(T)
    plot_raw(T)


def main() -> int:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    if "--dots-only" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        plot_decomp_dots(pd.read_csv(OUT / "areatune_tests.csv"))
        return 0
    if "--plot-only" not in sys.argv:
        if "--reuse-licks" in sys.argv:
            L = pd.read_csv(OUT / "areatune_licks.csv.gz")
        else:
            L = build_licks()
            L.to_csv(OUT / "areatune_licks.csv.gz", index=False)
        V = validate(L)
        V.to_csv(OUT / "areatune_validation.csv", index=False)
        print(V.to_string(index=False))
        S, M, K = sessions(L)
        # validation against lick_balance_matched_error (POOL c-i lratio == bal_spout)
        B = pd.read_csv(OUT / "balmatch_sessions.csv")
        chk = S[S.set.isin(["trial", "iti"])][["set", "label", "lratio|POOL|c-i"]].merge(B[["set", "label", "bal_spout"]],
                                                                                           on=["set", "label"])
        err = (chk["lratio|POOL|c-i"] - chk.bal_spout).abs()
        print(f"validation POOL c-i lratio vs balmatch bal_spout: n {len(chk)}, max |err| {err.max():.2e}")
        V = pd.concat([V, pd.DataFrame([{"check": "POOL c-i lratio vs balmatch_sessions bal_spout", "n": len(chk),
                                         "max_abs_err": err.max()}])], ignore_index=True)
        V.to_csv(OUT / "areatune_validation.csv", index=False)
        S.to_csv(OUT / "areatune_sessions.csv", index=False)
        M.to_csv(OUT / "areatune_spoutmeans.csv", index=False)
        K.to_csv(OUT / "areatune_kept.csv", index=False)
        A, T = tests(S)
        A.to_csv(OUT / "areatune_animals.csv", index=False)
        T.to_csv(OUT / "areatune_tests.csv", index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_rows", 500)
        key = A[(A.metric.isin(["lratio", "rshift", "depth1"])) & (A.epoch != "pre")]
        print("== per animal change vs pre (lratio / rshift / depth1)\n"
              + key.pivot_table(index=["set", "metric", "area", "hemi", "epoch"], columns="animal",
                                values="change").round(3).to_string())
        print("== pooled\n" + T[T.metric.isin(["lratio", "rshift", "depth1"])].round(3).to_string(index=False))
    plot_all()
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
