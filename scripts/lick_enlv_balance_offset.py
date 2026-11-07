"""ENLv LICKS: hemispheric BALANCE and frozen-decoder OFFSET (Priya 10-29), side by side with cued and ITI (docked /
moving). ENLv = ENL-violation lick (cam4 licks_all context ``pre_cue``: position strobe <= onset < cue; spout stationary at
the trial's position, no cue, no reward). NEVER call them ITI. Inputs: lick_enlv_extract.py (sessions/, sessions_reg/,
frozen readers).

EVENT DEFINITION (Priya 10-29). PRIMARY = ENLv licks with ONSET >= 1.0 s after spout arrival (strobe), all licks, whose
whole lick-aligned analysis window ENDS BEFORE THE MEASURED CUE (peak + 0.5 s < cue: the 0.1..0.45 s balance samples and
the 0..0.2 s decoder samples), raw signals. Timed from ARRIVAL, never from the cue. SENSITIVITY (table only): onset
0.5-1.5 s, 1.0-2.0 s, >= 1.5 s, >= 0 s, the FIRST ENLv lick of each trial (any time) vs LATER licks (also both >= 1 s),
bout starts (clean_start: >= 1 s since any lick) vs all, contact / non-contact, without the non-contact licks < 2 s before
the cue, a SHORT window (balance 0.1..0.25 s, cue rule peak + 0.3 s), and the cue rule off. ARRIVAL-CORRECTED (`_c`):
the per-session arrival + lick-onset ridge regression of lick_enlv_extract.regress (pre-cue frames only, no cue kernel),
arrival component removed; reported for every set, needed for >= 0 s and first licks.

MEASURES (definitions as the ITI / cued scripts; + signs: DIRECTION + = contralesional; HEMISPHERE + = contralesional
hemisphere larger; angle + = contralesional):
  c, i       pooled ALM + tjM1 + SSp-m + SSp-n + SSs response, 0.1..0.45 s after the lick peak minus -1.0..-0.5 s
             (c = right = contralesional hemisphere)
  bal_all    primary-style balance: log2(sum c / sum pred c) - log2(sum i / sum pred i), pred = the animal's PRE line on
             lick angle across spouts (pre sessions leave-one-session-out) -- lick_recovery_primary's gain ratio
  bal_spout  within-spout balance on licks INSIDE the pre angle range of their spout (lick_balance_matched_error)
  bal_angle_m  across-spout angle reference on the same inside-range licks (matched-error's bal_angle)
  d_raw_ci / d_bal_angle / lerr   per spout (lick_ci_spout_epoch: target frame / landed frame / landing error, deg)
  off_<r>    SAME-PLACE offset (lick_matched_readout --ref loclin): decoded - local-linear reference of the animal's PRE
             ENLv licks (same reader, pre LOSO) at the same landed lateral place (mm)
  offsp_<r>  MATCHED-ERROR offset: off_<r> referenced by the within-spout pre line on landing (inside-range licks)
  readers r: iti = frozen pre ITI decoder, trial = frozen pre cued decoder (TRANSFER; lick_reader kinds, verified
             bit-identical to lick_order_preds), enlv = frozen pre ENLv decoder trained on the PRIMARY pre ENLv licks
             (align enlv_on1; pre sessions scored by their held-out sibling). Decoder features are always the raw
             (uncorrected) 0..0.2 s features for `off`; `off_<r>_c` uses the arrival-corrected features.
  aim_err    behavior: angle - the animal's pre cued mean angle at the spout (enl_lick_audit)
BOUT STARTS (BOUTS = on1_clean) are reported next to the primary in enlv_compare: the balance on ALL licks has an unstable
baseline (-1..-0.5 s often holds the previous lick of the bout); the analog of the ITI first-after-quiet set is the bout
start. NB clean_start already implies >= 1 s after arrival, so on0_clean == on1_clean.
Change vs pre: per-animal table first (rule 8), then analysis_kit.boot_delta (paired nested animals -> sessions); flag
'<3' when fewer than 3 animals share the pooled sign.
Outputs <OUT_W>/enlv_licks/: enlv_table.csv.gz (per lick), enlv_sessions.csv, enlv_tests.csv (variant x measure x epoch),
enlv_animals.csv, enlv_spout_tests.csv, enlv_compare.csv (ENLv vs cued / ITI all / docked / moving), enlv_counts_*.csv,
enlv_arrival_*.csv, enlv_firstlick_latency.csv; figures/lick_enlv_*.png; figdata/lick_enlv_*.pkl.
Usage: python scripts/lick_enlv_balance_offset.py --table      (per-lick table: features, readers, ENLv decoder)
       python scripts/lick_enlv_balance_offset.py              (all tables + figures)
       python scripts/lick_enlv_balance_offset.py --plot-only  (redraw every figure from figdata)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402
import lick_enlv_extract as ex  # noqa: E402

OUT = ex.OUT
FIG = OUT / "figures"
FD = OUT / "figdata"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POST = EPS[1:]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]   # orofacial_cohort.POS_ORDER
READERS = ("iti", "trial", "enlv")
MIN_S = 8            # licks per session (all spouts)
MIN_SP = 5           # licks per session x spout (ci_spout_epoch MIN_LICKS)
MIN_M = 10           # inside-range licks per session (matched error MIN_LICKS)
Q = (2.5, 97.5)
SIGMAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
W_SHORT = ex._wm(0.1, 0.3)
PRIMARY = "on1"
BOUTS = "on1_clean"          # bout starts within the primary window (clean_start already implies >= 1 s after arrival)
# name: (onset window from strobe | "first" | "later", (lo, hi), clean, cue_end, signal, contact, drop non-contact near cue)
VARIANTS = {
    "on1": ("win", (1.0, np.inf), False, 0.5, "", None, False),
    "on1_clean": ("win", (1.0, np.inf), True, 0.5, "", None, False),
    "w05_15": ("win", (0.5, 1.5), False, 0.5, "", None, False),
    "w10_20": ("win", (1.0, 2.0), False, 0.5, "", None, False),
    "on15": ("win", (1.5, np.inf), False, 0.5, "", None, False),
    "on15_clean": ("win", (1.5, np.inf), True, 0.5, "", None, False),
    "on0": ("win", (0.0, np.inf), False, 0.5, "", None, False),
    "on0_clean": ("win", (0.0, np.inf), True, 0.5, "", None, False),
    "first": ("first", (0.0, np.inf), False, 0.5, "", None, False),
    "later": ("later", (0.0, np.inf), False, 0.5, "", None, False),
    "first_on1": ("first", (1.0, np.inf), False, 0.5, "", None, False),
    "later_on1": ("later", (1.0, np.inf), False, 0.5, "", None, False),
    "on1_corr": ("win", (1.0, np.inf), False, 0.5, "_c", None, False),
    "on0_corr": ("win", (0.0, np.inf), False, 0.5, "_c", None, False),
    "first_corr": ("first", (0.0, np.inf), False, 0.5, "_c", None, False),
    "later_corr": ("later", (0.0, np.inf), False, 0.5, "_c", None, False),
    "on1_contact": ("win", (1.0, np.inf), False, 0.5, "", True, False),
    "on1_noncontact": ("win", (1.0, np.inf), False, 0.5, "", False, False),
    "on1_nonearcue": ("win", (1.0, np.inf), False, 0.5, "", None, True),
    "on1_short": ("win", (1.0, np.inf), False, 0.3, "_s", None, False),
    "on1_nocuerule": ("win", (1.0, np.inf), False, None, "", None, False),
}
VLAB = {"on1": "PRIMARY onset >= 1.0 s", "on1_clean": ">= 1.0 s, bout starts", "w05_15": "0.5-1.5 s",
        "w10_20": "1.0-2.0 s", "on15": ">= 1.5 s", "on15_clean": ">= 1.5 s, bout starts", "on0": ">= 0 s (all)",
        "on0_clean": ">= 0 s, bout starts", "first": "FIRST ENLv lick / trial", "later": "later ENLv licks",
        "first_on1": "first lick, >= 1 s", "later_on1": "later licks, >= 1 s", "on1_corr": ">= 1 s, arrival-corrected",
        "on0_corr": ">= 0 s, arrival-corrected", "first_corr": "first lick, arrival-corrected",
        "later_corr": "later licks, arrival-corrected", "on1_contact": ">= 1 s, contact only",
        "on1_noncontact": ">= 1 s, non-contact only", "on1_nonearcue": ">= 1 s, minus non-contact < 2 s pre-cue",
        "on1_short": ">= 1 s, SHORT window 0.1-0.25 s", "on1_nocuerule": ">= 1 s, cue rule OFF"}
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


# ================================================================================================ per-lick table
def _day_map() -> dict:
    R = pd.read_csv(il.OUT / "recmark_sessions.csv", usecols=["label", "day"]).drop_duplicates("label")
    return dict(zip(R.label, R.day))


def table() -> pd.DataFrame:
    import iti_where_decoder as wd
    L = pd.read_csv(il.OUT / "enl_licks" / "enl_licks.csv.gz")
    T = pd.read_csv(il.OUT / "enl_licks" / "enl_trials.csv.gz", usecols=["label", "trial_id", "strobe_s", "cue_s"])
    L = L.merge(T, on=["label", "trial_id"], how="left", validate="many_to_one")
    K = il.lick_table()
    K = K[K.context == "pre_cue"][["session", "lick_uid", "side_ml", "reach_mm"]].drop_duplicates(["session", "lick_uid"])
    L["session"] = L.label.str[:5] + "2026" + L.label.str[-4:]
    L = L.merge(K, on=["session", "lick_uid"], how="left", validate="many_to_one")
    names, SD = wd.area_meta(basis="split")
    parts = []
    for lab, g in L.groupby("label", sort=True):
        f = ex.SESS / f"{lab}.npz"
        if not f.exists():
            continue
        z, zr = np.load(f), np.load(ex.SESS_REG / f"{lab}.npz")
        if not (z["lick_uid"] == g.lick_uid.to_numpy()).all():
            raise RuntimeError(f"{lab}: lick order mismatch")
        cols = [str(c) for c in z["cols"]]
        g = g.copy()
        g["peak"] = z["peak"]
        g["ok"] = z["ok"]
        sd = SD.loc[lab, names].to_numpy(float)[None, :]
        for sfx, kd, kp, kb, kc, ki in (("", "Xdec", "Xpost", "Xbase", "cT", "iT"),
                                        ("_c", "Xdec_c", "Xpost_c", "Xbase_c", "cT_c", "iT_c")):
            zz = z if sfx == "" else zr
            Xd = il.area_features(zz[kd].astype(float), cols, lab) / sd
            g[[f"f{j}{sfx}" for j in range(Xd.shape[1])]] = Xd
            for side, h in (("c", "right"), ("i", "left")):
                idx = [cols.index(f"{a}_{h}") for a in ex.POOL]
                g[f"{side}{sfx}"] = zz[kp][:, idx].sum(1) - zz[kb][:, idx].sum(1)
            for side, key in (("c", kc), ("i", ki)):
                tr = zz[key].astype(float)
                g[f"{side}_s{sfx}"] = tr[:, W_SHORT].mean(1) - tr[:, ex.W_BASE].mean(1)
        parts.append(g)
    L = pd.concat(parts, ignore_index=True)
    L = L[L.ok].reset_index(drop=True)
    L["day"] = L.label.map(_day_map())
    L["act"] = L.side_ml
    L["peak_from_strobe"] = L.peak - L.strobe_s
    L["peak_to_cue"] = L.cue_s - L.peak
    L = L.sort_values(["label", "trial_id", "daq_on_s"]).reset_index(drop=True)
    L["rank"] = L.groupby(["label", "trial_id"]).cumcount()
    L["first"] = L["rank"] == 0
    L["n_in_trial"] = L.groupby(["label", "trial_id"]).lick_uid.transform("size")
    nf = len(names)
    # frozen readers (ITI, cued): pre sessions by their held-out sibling, else the full model
    fits = ex.readers(4)
    for sfx in ("", "_c"):
        F = L[[f"f{j}{sfx}" for j in range(nf)]].to_numpy(float)
        for st in ("iti", "trial"):
            out = np.full(len(L), np.nan)
            for a in AN:
                fa = fits[(a, st)]
                for lab in L.label[L.animal == a].unique():
                    m = (L.label == lab).to_numpy()
                    sc, mdl = fa[lab] if lab in fa else fa[None]
                    out[m] = mdl.predict(sc.transform(F[m]))
            L[f"dec_{st}{sfx}"] = out
    # frozen ENLv decoder: PRIMARY pre licks (onset >= 1 s, window clear of the cue, 3-D landing known)
    tr = select(L, "on1") & (L.epoch == "pre") & np.isfinite(L.act) & L.pos.isin(POS)
    F = L[[f"f{j}" for j in range(nf)]].to_numpy(float)
    for sfx in ("", "_c"):
        L[f"dec_enlv{sfx}"] = np.nan
    for a in AN:
        S = {lab: (F[(tr & (L.label == lab)).to_numpy()], L.act[tr & (L.label == lab)].to_numpy(float))
             for lab in sorted(L.label[tr & (L.animal == a)].unique())}
        S = {k: v for k, v in S.items() if len(v[1]) >= 20}
        fa = ex.freeze(a, "enlv_on1", S, "lick_enlv_balance_offset.py")
        for lab in L.label[L.animal == a].unique():
            m = (L.label == lab).to_numpy()
            sc, mdl = fa[lab] if lab in fa else fa[None]
            for sfx in ("", "_c"):
                L.loc[m, f"dec_enlv{sfx}"] = mdl.predict(sc.transform(L.loc[m, [f"f{j}{sfx}" for j in range(nf)]]
                                                                         .to_numpy(float)))
    keep = [c for c in L.columns if not (c.startswith("f") and c[1:].split("_")[0].isdigit())]
    L = L[keep].drop(columns=["session", "ok"])
    L.to_csv(OUT / "enlv_table.csv.gz", index=False)
    print(f"table: {len(L)} ENLv licks, {L.label.nunique()} sessions", flush=True)
    return L


def select(L: pd.DataFrame, v: str) -> pd.Series:
    kind, (lo, hi), clean, cue_end, _sig, contact, nonear = VARIANTS[v]
    m = (L.t_from_strobe >= lo) & (L.t_from_strobe < hi)
    if kind == "first":
        m &= L["first"]
    elif kind == "later":
        m &= ~L["first"]
    if clean:
        m &= L.clean_start
    if cue_end is not None:
        m &= L.peak_to_cue > cue_end
    if contact is not None:
        m &= L.contact == contact
    if nonear:
        m &= ~((~L.contact) & (L.t_to_cue < 2.0))
    return m


# ================================================================================================ measures
def _line(tr, x, y):
    if len(tr) < 10 or tr[x].std() == 0:
        return 0.0, float(tr[y].mean()) if len(tr) else np.nan
    b1, b0 = np.polyfit(tr[x], tr[y], 1)
    return b1, b0


def _predict(g: pd.DataFrame, by_spout: bool, pairs) -> pd.DataFrame:
    """lick_balance_matched_error._predict generalized to any (y, x, out) pairs; pre LOSO."""
    g = g.copy()
    for _y, _x, col in pairs:
        g[col] = np.nan
    pre = g[g.epoch == "pre"]
    groups = [(g.pos == p, pre[pre.pos == p]) for p in g.pos.unique()] if by_spout else [(g.pos == g.pos, pre)]
    for sel, pr in groups:
        for lab in [None, *pr.label.unique()]:
            tr = pr[pr.label != lab] if lab else pr
            rows = sel & ((g.label == lab) if lab else (g.epoch != "pre"))
            if not rows.any():
                continue
            for y, x, col in pairs:
                b1, b0 = _line(tr.dropna(subset=[y, x]), x, y)
                g.loc[rows, col] = b0 + b1 * g.loc[rows, x]
    return g


def _bal(c, pc, i, pi) -> float:
    sc, sp, si, sq = c.sum(), pc.sum(), i.sum(), pi.sum()
    if min(sc, sp, si, sq) <= 0:
        return np.nan
    return float(np.log2(sc / sp) - np.log2(si / sq))


SIGMA: dict = {}


def measures(L: pd.DataFrame, v: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session table (all spouts) and session x spout table for one variant."""
    import likeforlike as lfl
    sig = VARIANTS[v][4]
    csig = sig if sig != "_s" else "_s"
    g0 = L[select(L, v)].copy()
    g0["c"], g0["i"] = g0[f"c{csig}"], g0[f"i{csig}"]
    g0 = g0.dropna(subset=["angle", "c", "i", "act"])
    g0 = g0[g0.pos.isin(POS)].reset_index(drop=True)
    g0["set"] = "enlv"
    dsfx = "_c" if sig == "_c" else ""
    for r in READERS:
        y = f"dec_{r}{dsfx}"
        if r not in SIGMA:
            SIGMA[r] = lfl.choose(g0.assign(_y=g0[y]), "act", "_y", SIGMAS)[0]
            print(f"  same-place reference sigma, reader {r}: {SIGMA[r]} mm", flush=True)
        R = lfl.ref_table(g0.assign(_y=g0[y]), "act", "_y", SIGMA[r])
        g0[f"off_{r}"] = g0[y] - R.ref
    srows, sprows = [], []
    for an, g in g0.groupby("animal"):
        pre = g[g.epoch == "pre"]
        if pre.label.nunique() < 2:
            continue
        tgt = pre.groupby("pos").angle.mean()
        g = g.copy()
        g["lerr"] = g.angle - g.pos.map(tgt)
        g["inside"] = (g.angle >= g.pos.map(pre.groupby("pos").angle.quantile(Q[0] / 100))) & \
            (g.angle <= g.pos.map(pre.groupby("pos").angle.quantile(Q[1] / 100)))
        ga = _predict(g, False, [("c", "angle", "pa_c"), ("i", "angle", "pa_i")])
        pairs = [("c", "angle", "ps_c"), ("i", "angle", "ps_i")] + [(f"off_{r}", "act", f"ps_off_{r}") for r in READERS]
        gs = _predict(g, True, pairs)
        g["pa_c"], g["pa_i"] = ga.pa_c, ga.pa_i
        for col in [p[2] for p in pairs]:
            g[col] = gs[col]
        for (lab, ep), h in g.groupby(["label", "epoch"]):
            if len(h) < MIN_S:
                continue
            r = {"variant": v, "animal": an, "label": lab, "epoch": ep, "day": h.day.iloc[0], "n": len(h),
                 "bal_all": _bal(h.c, h.pa_c, h.i, h.pa_i), "raw_ci_all": np.log2(h.c.sum() / h.i.sum())
                 if h.c.sum() > 0 and h.i.sum() > 0 else np.nan,
                 "lerr": h.lerr.mean(), "aim_err": h.aim_err.mean(), "t_from_strobe": h.t_from_strobe.mean()}
            for rd in READERS:
                r[f"off_{rd}"] = h[f"off_{rd}"].mean()
            m = h[h.inside]
            r["n_inside"] = len(m)
            if len(m) >= MIN_M:
                r["bal_spout"] = _bal(m.c, m.ps_c, m.i, m.ps_i)
                r["bal_angle_m"] = _bal(m.c, m.pa_c, m.i, m.pa_i)
                for rd in READERS:
                    r[f"offsp_{rd}"] = float((m[f"off_{rd}"] - m[f"ps_off_{rd}"]).mean())
            srows.append(r)
            for p, hp in h.groupby("pos"):
                if len(hp) < MIN_SP:
                    continue
                q = {"variant": v, "animal": an, "label": lab, "epoch": ep, "pos": p, "n": len(hp),
                     "raw_ci": np.log2(hp.c.sum() / hp.i.sum()) if hp.c.sum() > 0 and hp.i.sum() > 0 else np.nan,
                     "bal_angle": _bal(hp.c, hp.pa_c, hp.i, hp.pa_i), "lerr": hp.lerr.mean(),
                     "aim_err": hp.aim_err.mean()}
                for rd in READERS:
                    q[f"off_{rd}"] = hp[f"off_{rd}"].mean()
                sprows.append(q)
    S, SP = pd.DataFrame(srows), pd.DataFrame(sprows)
    if len(SP):
        ref = SP[SP.epoch == "pre"].groupby(["animal", "pos"])[["raw_ci", "bal_angle"]].mean()
        SP = SP.join(ref.add_prefix("pre_"), on=["animal", "pos"])
        SP["d_raw_ci"], SP["d_bal_angle"] = SP.raw_ci - SP.pre_raw_ci, SP.bal_angle - SP.pre_bal_angle
        # 'all' / far_contra C/I rows (lick_ci_spout_epoch: mean over the session's spouts)
        agg = SP.groupby(["animal", "label", "epoch"])[["d_raw_ci", "d_bal_angle"]].mean().add_suffix("_spmean")
        S = S.join(agg, on=["animal", "label", "epoch"])
    return S, SP


MEAS = ["bal_all", "bal_spout", "bal_angle_m", "d_raw_ci_spmean", "d_bal_angle_spmean", "raw_ci_all",
        *[f"off_{r}" for r in READERS], *[f"offsp_{r}" for r in READERS], "lerr", "aim_err"]
SPMEAS = ["d_raw_ci", "d_bal_angle", "lerr", "aim_err", *[f"off_{r}" for r in READERS]]


def tests(S: pd.DataFrame, meas, keys) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    trows, arows = [], []
    for k, g in S.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        base = dict(zip(keys, k))
        for c in meas:
            if c not in g:
                continue
            by = {(a, ep): h[c].dropna().to_numpy() for (a, ep), h in g.groupby(["animal", "epoch"])}
            pre = {a: by[(a, "pre")] for a in AN if len(by.get((a, "pre"), [])) > 0}
            for ep in EPS:
                for a in AN:
                    x = by.get((a, ep), np.array([]))
                    if len(x):
                        arows.append({**base, "measure": c, "epoch": ep, "animal": a, "n_sess": len(x),
                                      "level": float(x.mean()),
                                      "delta": float(x.mean() - pre[a].mean()) if a in pre else np.nan})
            for ep in POST:
                post = {a: by[(a, ep)] for a in AN if len(by.get((a, ep), [])) > 0 and a in pre}
                r = {**base, "measure": c, "epoch": ep}
                ci = ak.boot_delta(post, pre, np.random.default_rng(0)) if post else None
                d = {a: float(post[a].mean() - pre[a].mean()) for a in post}
                if ci is None:
                    r.update(point=np.nan, lo=np.nan, hi=np.nan, n_animals=len(d), n_same=0, star="", flag="")
                else:
                    ns = int(sum(np.sign(x) == np.sign(ci.point) for x in d.values()))
                    r.update(point=ci.point, lo=ci.lo, hi=ci.hi, n_animals=len(d), n_same=ns,
                             star="*" if ci.excludes_zero else "", flag="<3" if ns < 3 else "")
                r["n_sess_post"] = sum(len(v) for v in post.values())
                r.update({a: d.get(a, np.nan) for a in AN})
                trows.append(r)
    return pd.DataFrame(trows), pd.DataFrame(arows)


# ================================================================================================ comparisons
def compare(T: pd.DataFrame, TS: pd.DataFrame) -> pd.DataFrame:
    """ENLv (primary, every reader) vs cued / ITI all / ITI docked / ITI moving on the same quantities."""
    from wfield_local import analysis_kit as ak
    root = il.OUT
    fold = {"ITI all": root, "ITI docked": root / "itifilter_dock", "ITI moving": root / "itifilter_notdock"}
    rows = []

    def add(q, src, ep, point, lo_, hi_, d: dict):
        vals = [x for x in d.values() if np.isfinite(x)]
        ns = int(sum(np.sign(x) == np.sign(point) for x in vals)) if np.isfinite(point) else 0
        rows.append({"quantity": q, "source": src, "epoch": ep, "point": point, "lo": lo_, "hi": hi_, "n_same": ns,
                     "n": len(vals), "flag": "<3" if ns < 3 else "", **{a: d.get(a, np.nan) for a in AN}})
    Q = [("same-place offset (mm)", "resid"), ("matched-error offset (mm)", "off_spout"),
         ("within-spout balance", "bal_spout"), ("primary balance (angle ref.)", "recprim"),
         ("C/I d_raw_ci, all spouts", ("all", "d_raw_ci")), ("C/I d_raw_ci, far contra", ("far_contra", "d_raw_ci")),
         ("C/I d_bal_angle, all spouts", ("all", "d_bal_angle")),
         ("C/I d_bal_angle, far contra", ("far_contra", "d_bal_angle"))]
    for src, st, d0 in [("cued", "trial", root)] + [(k, "iti", v) for k, v in fold.items()]:
        M = pd.read_csv(d0 / "matched_loclin_tests.csv")
        M = M[(M.scope == "all spouts") & (M.measure == "resid") & (M.set == st) & (M.cls == "all")]
        for _, r in M.iterrows():
            add(Q[0][0], src, r.epoch, r.point, r.lo, r.hi, {a: r.get(f"animal_{a}", np.nan) for a in AN})
        B = pd.read_csv(d0 / "balmatch_tests.csv")
        for q, mm in (Q[1], Q[2]):
            for _, r in B[(B.set == st) & (B.measure == mm)].iterrows():
                add(q, src, r.epoch, r.point, r.lo, r.hi, {a: r.get(f"d_{a}", np.nan) for a in AN})
        RP = pd.read_csv(d0 / "recprim_sessions.csv")
        RP = RP[RP.panel == ("cued first lick -> aim" if st == "trial" else "ITI first after quiet -> iti_ang")]
        pre = {a: h.marker.dropna().to_numpy() for a, h in sorted(RP[RP.epoch == "pre"].groupby("animal"))}
        for ep in POST:
            post = {a: h.marker.dropna().to_numpy() for a, h in sorted(RP[RP.epoch == ep].groupby("animal")) if a in pre}
            ci = ak.boot_delta(post, pre, np.random.default_rng(0))
            add(Q[3][0], src, ep, ci.point, ci.lo, ci.hi, {a: post[a].mean() - pre[a].mean() for a in post})
        C = pd.read_csv(d0 / "ci_spout_epoch_tests.csv")
        for q, (grp, var) in Q[4:]:
            for _, r in C[(C.set == st) & (C.grp == grp) & (C["var"] == var)].iterrows():
                add(q, src, r.epoch, r.est, r.lo, r.hi, {a: r.get(a, np.nan) for a in AN})
    # ENLv: PRIMARY (all licks >= 1 s) and its BOUT STARTS (clean_start, >= 1 s since any lick: the analogue of the ITI
    # 'first after quiet' set the ITI balance / matched-error numbers use)
    for vv, tag in ((PRIMARY, "ENLv"), (BOUTS, "ENLv bout starts")):
        P = T[T.variant == vv]
        PS = TS[(TS.variant == vv) & (TS.pos == "far_contra")]
        for q, f in (("same-place offset (mm)", "off_{}"), ("matched-error offset (mm)", "offsp_{}")):
            for rd in READERS:
                for _, r in P[P.measure == f.format(rd)].iterrows():
                    add(q, f"{tag} ({rd} reader)", r.epoch, r.point, r.lo, r.hi, {a: r[a] for a in AN})
        for q, m in (("within-spout balance", "bal_spout"), ("primary balance (angle ref.)", "bal_all"),
                     ("C/I d_raw_ci, all spouts", "d_raw_ci_spmean"),
                     ("C/I d_bal_angle, all spouts", "d_bal_angle_spmean")):
            for _, r in P[P.measure == m].iterrows():
                add(q, tag, r.epoch, r.point, r.lo, r.hi, {a: r[a] for a in AN})
        for q, m in (("C/I d_raw_ci, far contra", "d_raw_ci"), ("C/I d_bal_angle, far contra", "d_bal_angle")):
            for _, r in PS[PS.measure == m].iterrows():
                add(q, tag, r.epoch, r.point, r.lo, r.hi, {a: r[a] for a in AN})
    return pd.DataFrame(rows)


# ================================================================================================ diagnostics
def arrival(L: pd.DataFrame) -> dict:
    """Arrival kernels (regression, pre-strobe referenced) and lick-free-trial traces per animal x epoch; return time."""
    from wfield_local import epochs
    rows, kern, free, stab = [], {}, {}, []
    areas = None
    for f in sorted(ex.SESS_REG.glob("*.npz")):
        lab = f.stem
        zr, z = np.load(f), np.load(ex.SESS / f"{lab}.npz")
        cols = [str(c) for c in zr["cols"]]
        a, ep = lab[:4], str(epochs.epoch_of(lab))
        pool = {s: [cols.index(f"{x}_{h}") for x in ex.POOL] for s, h in (("c", "right"), ("i", "left"))}
        areas = areas or [c for c in cols if c.split("_")[0] in ex.POOL]
        kc = {**{c: zr["k_arr"][:, cols.index(c)] for c in areas}, **{f"pool_{s}": zr["k_arr"][:, ix].sum(1)
                                                                       for s, ix in pool.items()}}
        lk = {**{c: zr["k_lick"][:, cols.index(c)] for c in areas},
              **{f"pool_{s}": zr["k_lick"][:, ix].sum(1) for s, ix in pool.items()}}
        kern.setdefault((a, ep), []).append((kc, lk))
        am = z["arr_mean"][6]
        pre_m = (ex.LAGS_A >= -1.0) & (ex.LAGS_A <= -0.5)
        tr = {f"pool_{s}": am[:, ix].sum(1) - am[pre_m][:, ix].sum(1).mean() for s, ix in pool.items()}
        tr.update({c: am[:, cols.index(c)] - am[pre_m][:, cols.index(c)].mean() for c in areas})
        free.setdefault((a, ep), []).append((tr, int(z["arr_trials"]), z["arr_n"][6]))
        stab.append({"label": lab, "animal": a, "epoch": ep, "n_trials": int(zr["n_trials"]), "alpha": float(zr["alpha"]),
                     "cv_r2": float(zr["cv_r2"]), "splithalf_r_pool_median": float(np.nanmedian(
                         zr["splithalf_r"][pool["c"] + pool["i"]])), "n_lickfree_trials": int(z["arr_trials"])})
    lk_ = ex.LAGS_K
    K = {}
    for key, lst in kern.items():
        K[key] = ({c: np.mean([x[0][c] for x in lst], 0) for c in lst[0][0]},
                  {c: np.mean([x[1][c] for x in lst], 0) for c in lst[0][1]})
        kc, lkk = K[key]
        for c in kc:
            k = kc[c]
            post = lk_ >= 0
            ipk = np.argmax(np.abs(np.where(post, k, 0)))
            thr = 0.2 * abs(k[ipk])
            after = np.flatnonzero(post & (lk_ >= lk_[ipk]) & (np.abs(k) >= thr))
            t_ret = float(lk_[after[-1]] + 0.05) if len(after) else np.nan
            lpk = float(np.max(np.abs(lkk[c][(lk_ >= 0) & (lk_ <= 0.6)])))
            w = lambda lo_, hi_, k=k: float(np.mean(k[(lk_ >= lo_) & (lk_ < hi_)]))  # noqa: E731
            rows.append({"animal": key[0], "epoch": key[1], "area": c, "arr_peak": float(k[ipk]),
                         "arr_peak_t": float(lk_[ipk]), "t_below20pct_for_good": t_ret, "arr_1.0_1.5": w(1.0, 1.5),
                         "arr_1.5_2.0": w(1.5, 2.0), "arr_2.0_3.0": w(2.0, 3.0), "lick_kernel_peak": lpk,
                         "arr_1.0_1.5_over_lick": w(1.0, 1.5) / lpk if lpk > 0 else np.nan,
                         "arr_1.5_2.0_over_lick": w(1.5, 2.0) / lpk if lpk > 0 else np.nan})
    FR = {}
    for key, lst in free.items():
        n = np.sum([x[2] for x in lst], 0)
        FR[key] = ({c: np.nansum([np.nan_to_num(x[0][c]) * x[2] for x in lst], 0) / np.maximum(n, 1)
                    for c in lst[0][0]}, n, sum(x[1] for x in lst))
    # MODEL-FREE check on lick-free trials (no lick from 1 s before arrival to the cue): response remaining at
    # 1.0-1.5 / 1.5-2.0 s as a share of its 0-1 s peak (pre-strobe -1..-0.5 s = 0); lags with < 5 trials dropped
    frows = []
    la = ex.LAGS_A
    for key, (tr, n, nt) in FR.items():
        for c, v in tr.items():
            v = np.where(n >= 5, v, np.nan)
            ipk = np.nanargmax(np.abs(np.where((la >= 0) & (la <= 1.0), v, np.nan)))
            pk = v[ipk]
            m = lambda lo_, hi_, v=v: float(np.nanmean(v[(la >= lo_) & (la < hi_)]))  # noqa: E731
            frows.append({"animal": key[0], "epoch": key[1], "area": c, "n_trials": nt, "peak": float(pk),
                          "peak_t": float(la[ipk]), "m_1.0_1.5": m(1.0, 1.5), "m_1.5_2.0": m(1.5, 2.0),
                          "frac_1.0_1.5": m(1.0, 1.5) / pk, "frac_1.5_2.0": m(1.5, 2.0) / pk,
                          "n_trials_at_1.5": int(n[np.argmin(np.abs(la - 1.5))]),
                          "n_trials_at_2.0": int(n[np.argmin(np.abs(la - 2.0))])})
    return {"rows": pd.DataFrame(rows), "stab": pd.DataFrame(stab), "K": K, "FR": FR, "areas": areas,
            "free": pd.DataFrame(frows)}


def counts(L: pd.DataFrame) -> dict:
    out = {}
    rows = []
    for v in ("on1", "on0", "first", "on1_short"):
        base = VARIANTS[v]
        m0 = select(L.assign(peak_to_cue=np.inf), v)                  # before the cue rule
        m1 = select(L, v)
        g = L.assign(before=m0, kept=m1)
        for (a, ep, p), h in g[g.before].groupby(["animal", "epoch", "pos"]):
            rows.append({"variant": v, "cue_end_s": base[3], "animal": a, "epoch": ep, "pos": p, "n_before": len(h),
                         "n_kept": int(h.kept.sum()), "n_removed": int((~h.kept).sum()),
                         "share_removed": float((~h.kept).mean())})
    out["cue_rule"] = pd.DataFrame(rows)
    g = L[select(L.assign(peak_to_cue=np.inf), "on1")]
    out["contact"] = g.groupby(["animal", "epoch", "pos"]).agg(
        n=("contact", "size"), n_contact=("contact", "sum"),
        n_noncontact=("contact", lambda s: int((~s).sum())),
        n_lt2s_before_cue=("t_to_cue", lambda s: int((s < 2.0).sum())),
        n_noncontact_lt2s=("t_to_cue", lambda s: int(((s < 2.0) & ~g.loc[s.index, "contact"]).sum())),
        n_window_crosses_cue=("peak_to_cue", lambda s: int((s <= 0.5).sum()))).reset_index()
    F = L[L["first"]]
    out["first_latency"] = F.groupby(["animal", "epoch"]).t_from_strobe.agg(
        n_first="size", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
        share_lt05=lambda s: (s < 0.5).mean(), share_lt1=lambda s: (s < 1.0).mean(),
        share_ge15=lambda s: (s >= 1.5).mean()).reset_index()
    # first vs later in the SAME trials (>= 2 ENLv licks): behaviour
    G = L[L.n_in_trial >= 2]
    fl = G.groupby(["animal", "epoch", "label", "first"]).aim_err.mean().unstack("first")
    fl = (fl[True] - fl[False]).rename("aim_first_minus_later").reset_index()
    out["first_later_aim"] = fl.groupby(["animal", "epoch"]).aim_first_minus_later.agg(["mean", "size"]).reset_index()
    return out


# ================================================================================================ figures
def _style(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.axhline(0, color="0.7", lw=0.6, zorder=0)


def _header(fig, n_title: int = 2) -> None:
    """Place suptitle, the figure legend (if any) and the top of the panels in INCHES from the top edge, after
    tight_layout, so the header has no dead band (tight_layout's rect left 1-2 in of empty space). Plotting only."""
    H = fig.get_figheight()
    y = 0.08
    if fig._suptitle is not None:
        fig._suptitle.set_y(1 - y / H)
        fig._suptitle.set_va("top")
        y += 0.16 * (fig._suptitle.get_text().count("\n") + 1) + 0.06
    for lg in fig.legends:
        lg.set_bbox_to_anchor((0.5, 1 - y / H), transform=fig.transFigure)
        lg.set_loc("upper center")
        y += 0.2 * -(-len(lg.get_texts()) // max(lg._ncols, 1)) + 0.12
    hs = fig.subplotpars.hspace
    fig.subplots_adjust(top=1 - (y + 0.17 * n_title + 0.08) / H, hspace=max(hs, 0.38))


def fig_arrival(F: dict) -> None:
    import matplotlib.pyplot as plt
    K, FR, R, lk = F["K"], F["FR"], F["rows"], ex.LAGS_K
    fig, axs = plt.subplots(4, 4, figsize=(19, 15), squeeze=False)
    col = {"pre": "0.3", "acute": "#C44E52", "subacute": "#DD8452", "chronic": "#4C72B0"}
    for j, a in enumerate(AN):
        for ep in EPS:
            if (a, ep) not in K:
                continue
            kc, lkk = K[(a, ep)]
            axs[0, j].plot(lk, kc["pool_c"], color=col[ep], lw=1.4, label=f"{ep} contra pool")
            axs[0, j].plot(lk, kc["pool_i"], color=col[ep], lw=1.0, ls="--", label=f"{ep} ipsi pool")
            axs[1, j].plot(lk, lkk["pool_c"], color=col[ep], lw=1.4)
            axs[1, j].plot(lk, lkk["pool_i"], color=col[ep], lw=1.0, ls="--")
        ntr = []
        for ep in EPS:
            if (a, ep) in K and (a, ep) in FR:
                tr, n, nt = FR[(a, ep)]
                ok = n >= 5
                axs[2, j].plot(ex.LAGS_A[ok], tr["pool_c"][ok], color=col[ep], lw=1.4)
                axs[2, j].plot(ex.LAGS_A[ok], tr["pool_i"][ok], color=col[ep], lw=1.0, ls="--")
                ntr.append(f"{ep[:3]} {nt}")
        for ax in axs[:3, j]:
            for x in (1.0, 1.5):
                ax.axvline(x, color="k", lw=0.6, ls=":")
            _style(ax)
        axs[0, j].set_title(f"{a}: ARRIVAL kernel (regression, pre-strobe = 0)", fontsize=9)
        axs[1, j].set_title(f"{a}: LICK-ONSET kernel (same fit)", fontsize=9)
        axs[2, j].set_title(f"{a}: lick-free trials, strobe-aligned (after-cue masked)\nn trials: " + ", ".join(ntr),
                            fontsize=9)
        h = F["onsets"].get(a)
        if h is not None:
            axs[3, j].hist(np.clip(h["all"], 0, 6), bins=np.arange(0, 6.01, 0.1), color="0.6")
            ax2 = axs[3, j].twinx()
            ax2.hist(np.clip(h["first"], 0, 6), bins=np.arange(0, 6.01, 0.1), histtype="step", color="#C44E52",
                     lw=1.4)
            ax2.tick_params(axis="y", colors="#C44E52")
        axs[3, j].axvline(1.0, color="k", lw=0.8)
        axs[3, j].set_title(f"{a}: BEHAVIOR onset time from arrival (s; clipped at 6)", fontsize=9)
        axs[3, j].set_xlabel("time since spout arrival (strobe) / lick onset (s)")
    axs[0, 0].set_ylabel("dF/F x1e3 (summed pool)")
    fr = F["free"]
    pre = fr[(fr.epoch == "pre") & fr.area.str.startswith("pool")].groupby("area")[["frac_1.0_1.5", "frac_1.5_2.0"]]\
        .median()
    rr = R[(R.epoch == "pre") & R.area.str.startswith("pool")].groupby("area")[
        ["arr_1.0_1.5_over_lick", "arr_1.5_2.0_over_lick"]].median()
    txt = ("PRE, median of animals -- lick-free trials: " + "; ".join(
        f"{k} {r['frac_1.0_1.5']:+.0%} / {r['frac_1.5_2.0']:+.0%} of its 0-1 s peak at 1.0-1.5 / 1.5-2.0 s"
        for k, r in pre.iterrows()) + "; regression arrival kernel: " + "; ".join(
        f"{k} {r['arr_1.0_1.5_over_lick']:+.0%} / {r['arr_1.5_2.0_over_lick']:+.0%} of the lick-kernel peak"
        for k, r in rr.iterrows()))
    fig.suptitle(textwrap.fill("NEURAL + BEHAVIOR: does the spout-ARRIVAL response overlap ENLv licks? Rows 1-2: "
                               "per-session ridge regression (arrival kernel + lick-onset kernel, PRE-CUE frames only, "
                               "no cue kernel; the kernel's late level is weakly identified -- it trades off against the "
                               "intercept); row 3: model-free strobe-aligned mean on lick-free trials (thin pre-stroke "
                               "in PS92 / PS93). Pooled ALM + tjM1 + SSp-m + SSp-n + SSs per hemisphere, pre-strobe "
                               "-1..-0.5 s = 0; dotted = 1.0 / 1.5 s. " + txt + ". PROVISIONAL", 210), fontsize=9,
                 y=0.995)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hl = [Line2D([], [], color=col[ep], lw=2, label=ep) for ep in EPS]
    hl += [Line2D([], [], color="0.3", lw=1.4, label="solid = contra pool"),
           Line2D([], [], color="0.3", lw=1.0, ls="--", label="dashed = ipsi pool"),
           Patch(color="0.6", label="row 4: all ENLv licks (left axis)"),
           Line2D([], [], color="#C44E52", lw=1.4, label="row 4: first ENLv lick / trial (right axis)")]
    fig.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=len(hl), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _header(fig, 1)
    save_png_svg(fig, FIG / "lick_enlv_arrival.png", dpi=90)
    plt.close(fig)


SRC_COL = {"cued": "#555555", "ITI all": "#1f78b4", "ITI docked": "#1b9e77", "ITI moving": "#e7298a",
           "ENLv": "#d95f02", "ENLv (iti reader)": "#d95f02", "ENLv (trial reader)": "#a6761d",
           "ENLv (enlv reader)": "#7570b3", "ENLv bout starts": "#8c2d04", "ENLv bout starts (iti reader)": "#8c2d04",
           "ENLv bout starts (trial reader)": "#bf812d", "ENLv bout starts (enlv reader)": "#54278f"}


def fig_compare(C: pd.DataFrame, keep: tuple | None = None, name: str = "lick_enlv_compare.png") -> None:
    import matplotlib.pyplot as plt
    qs = list(dict.fromkeys(C.quantity))
    fig, axs = plt.subplots(2, 4, figsize=(20 if keep else 26, 10.5), squeeze=False)
    for ax, q in zip(axs.flat, qs):
        G = C[C.quantity == q]
        srcs = [s for s in SRC_COL if s in set(G.source) and (keep is None or s in keep)]
        w = 0.8 / len(srcs)
        for k, s in enumerate(srcs):
            g = G[G.source == s].set_index("epoch").reindex(POST)
            x = np.arange(3) + (k - (len(srcs) - 1) / 2) * w
            ax.errorbar(x, g.point, yerr=[g.point - g.lo, g.hi - g.point], fmt="o", color=SRC_COL[s], ms=5, capsize=2,
                        label=s)
            for e in range(3):
                for a in AN:
                    ax.plot(x[e] + 0.03, g[a].iloc[e], MK[a], mfc="none", mec=SRC_COL[s], ms=3.2, alpha=0.7)
                if np.isfinite(g.point.iloc[e]):
                    ax.text(x[e], g.hi.iloc[e], f"{int(g.n_same.iloc[e])}/{int(g.n.iloc[e])}", fontsize=5.5,
                            ha="center", va="bottom", color=SRC_COL[s])
        ax.set_xticks(range(3), POST)
        _style(ax)
        ax.set_title(f"{q}\nchange vs pre", fontsize=9)
    hd = {}
    for ax in axs.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            hd.setdefault(l_, h_)
    hd = {s: hd[s] for s in SRC_COL if s in hd}
    fig.legend(list(hd.values()), list(hd.keys()), loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=min(len(hd), 6), fontsize=9, frameon=False)
    fig.suptitle(textwrap.fill(
        ("SIMPLIFIED (cued, ITI all, ENLv on the ITI reader / for balance; all sources = lick_enlv_compare). "
         if keep else "") +
        "NEURAL: ENLv licks (PRIMARY: onset >= 1.0 s after spout arrival, window ends before the cue; spout stationary, "
        "no cue, no reward) vs cued first licks and ITI licks (all / docked spout / moving spout). Offsets: frozen "
        "pre-stroke decoders -- ITI reader (= the ITI numbers' decoder), cued reader, ENLv-trained; references = each "
        "set's own pre licks. Balance + = contralesional hemisphere larger; offset + = read contralesional. Points = "
        "change vs pre, 95% paired nested bootstrap; open markers = animals (o PS92, s PS93, ^ PS94, D PS95); k/n = "
        "animals with the pooled sign. ENLv per-lick sets differ from ITI (all licks, not first-after-quiet) for the "
        "balance -- see enlv_tests for bout starts. PROVISIONAL", 220), fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _header(fig, 2)
    save_png_svg(fig, FIG / name, dpi=90)
    plt.close(fig)


def fig_perspout(TS: pd.DataFrame, AS: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    ms = [("d_raw_ci", "C/I vs pre at the same spout (TARGET frame, log2)"),
          ("d_bal_angle", "balance vs pre angle model (LANDED frame, log2)"),
          ("off_iti", "same-place offset, ITI reader (mm)"), ("off_enlv", "same-place offset, ENLv decoder (mm)"),
          ("lerr", "BEHAVIOR landing error vs pre ENLv at spout (deg)")]
    col = {"acute": "#C44E52", "subacute": "#DD8452", "chronic": "#4C72B0"}
    off = {"acute": -0.22, "subacute": 0.0, "chronic": 0.22}
    fig, axs = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 5.2), squeeze=False)
    x = np.arange(len(POS))
    for ax, (m, lab) in zip(axs.flat, ms):
        for ep in POST:
            t = TS[(TS.variant == PRIMARY) & (TS.measure == m) & (TS.epoch == ep)].set_index("pos").reindex(POS)
            ax.errorbar(x + off[ep], t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o-", color=col[ep], ms=5,
                        lw=1.2, capsize=2.5, label=ep)
            h = AS[(AS.variant == PRIMARY) & (AS.measure == m) & (AS.epoch == ep)]
            for a in AN:
                ha = h[h.animal == a].set_index("pos").reindex(POS)
                ax.plot(x + off[ep] + 0.06, ha.delta, MK[a], color=col[ep], ms=3.2, alpha=0.6, mec="none")
            for k, p in enumerate(POS):
                if p in t.index and isinstance(t.loc[p, "flag"], str) and t.loc[p, "flag"]:
                    ax.text(k + off[ep], t.loc[p, "hi"], "<3", fontsize=6, color=col[ep], ha="center", va="bottom")
        ax.set_xticks(x, POS, rotation=25, fontsize=8)
        _style(ax)
        ax.set_title(lab + "\nchange vs pre", fontsize=9)
    h_, l_ = axs.flat[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(l_), fontsize=9, frameon=False)
    fig.suptitle("NEURAL / BEHAVIOR per spout, ENLv PRIMARY (onset >= 1.0 s after arrival, window before the cue). "
                 "Spout = the trial's spout in place. Lines = pooled change vs pre (95% nested bootstrap), small markers "
                 "= animals; '<3' = fewer than 3 animals share the sign. PROVISIONAL", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _header(fig, 2)
    save_png_svg(fig, FIG / "lick_enlv_perspout.png", dpi=90)
    plt.close(fig)


SWEEP_M = [("bal_all", "primary balance\n(angle ref.)"), ("bal_spout", "within-spout balance"),
           ("d_raw_ci_spmean", "raw C/I vs pre\n(mean over spouts)"), ("off_iti", "same-place offset\nITI reader (mm)"),
           ("offsp_iti", "matched-error offset\nITI reader (mm)"), ("off_enlv", "same-place offset\nENLv decoder (mm)"),
           ("lerr", "BEHAVIOR landing\nerror (deg)")]


SWEEP_SIMPLE_V = ["on1", "on1_clean", "on0", "on1_corr", "on1_contact", "on1_noncontact", "on1_nocuerule"]
SWEEP_SIMPLE_M = ["bal_all", "d_raw_ci_spmean", "off_iti", "lerr"]


def fig_sweep(T: pd.DataFrame, vs: list | None = None, mm: list | None = None,
              name: str = "lick_enlv_sweep.png") -> None:
    import matplotlib.pyplot as plt
    vs = list(VARIANTS) if vs is None else vs
    sm = SWEEP_M if mm is None else [x for x in SWEEP_M if x[0] in mm]
    fig, axs = plt.subplots(1, len(sm), figsize=(3.6 * len(sm) + (1.5 if mm else 0), 0.36 * len(vs) + 2.5),
                            sharey=True, squeeze=False)
    col = {"acute": "#C44E52", "subacute": "#DD8452", "chronic": "#4C72B0"}
    y = np.arange(len(vs))[::-1]
    for ax, (m, lab) in zip(axs.flat, sm):
        for k, ep in enumerate(POST):
            t = T[(T.measure == m) & (T.epoch == ep)].set_index("variant").reindex(vs)
            yy = y + (k - 1) * 0.25
            ax.errorbar(t.point, yy, xerr=[t.point - t.lo, t.hi - t.point], fmt="o", color=col[ep], ms=3.5,
                        capsize=1.5, lw=0.9, label=ep)
            for a in AN:
                ax.plot(t[a], yy, MK[a], color=col[ep], ms=2.2, alpha=0.5, mec="none")
        ax.axvline(0, color="0.6", lw=0.7)
        ax.axhspan(y[0] - 0.5, y[0] + 0.5, color="#ffe9a8", alpha=0.5, zorder=0)
        ax.set_title(lab, fontsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0, 0].set_yticks(y, [VLAB[v] for v in vs], fontsize=7.5)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    H = fig.get_size_inches()[1]
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.55 / H), ncol=len(l_), fontsize=9,
               frameon=False)
    sup = ("NEURAL / BEHAVIOR: ENLv SENSITIVITY table (change vs pre; 95% nested bootstrap; small markers = "
           "animals). Highlight = PRIMARY (onset >= 1.0 s after arrival). Raw signals unless 'arrival-corrected' "
           "(per-session arrival + lick ridge regression, pre-cue frames). PROVISIONAL")
    if mm:
        sup = "SIMPLIFIED (key variants / measures; full table = lick_enlv_sweep). " + sup
    fig.suptitle(textwrap.fill(sup, 150 if mm else 250), fontsize=9, y=1 - 0.08 / H)
    fig.tight_layout(rect=(0, 0, 1, 1 - (1.15 if mm else 0.9) / H))
    _header(fig, 2)
    save_png_svg(fig, FIG / name, dpi=90)
    plt.close(fig)


def fig_firstlater(T: pd.DataFrame, vs: list | None = None, name: str = "lick_enlv_first_later.png") -> None:
    import matplotlib.pyplot as plt
    vs = ["first", "later", "first_corr", "later_corr", "first_on1", "later_on1"] if vs is None else vs
    ms = [("aim_err", "BEHAVIOR aim error vs pre cued (deg)"), ("off_iti", "same-place offset, ITI reader (mm)"),
          ("off_enlv", "same-place offset, ENLv decoder (mm)"), ("bal_all", "primary balance (angle ref.)"),
          ("d_raw_ci_spmean", "raw C/I vs pre (mean over spouts)")]
    cmap = {"first": "#d95f02", "later": "#7570b3", "first_corr": "#fdae6b", "later_corr": "#bcbddc",
            "first_on1": "#a63603", "later_on1": "#3f007d"}
    fig, axs = plt.subplots(1, len(ms), figsize=(5.0 * len(ms), 5), squeeze=False)
    w = 0.8 / len(vs)
    for ax, (m, lab) in zip(axs.flat, ms):
        for k, v in enumerate(vs):
            t = T[(T.variant == v) & (T.measure == m)].set_index("epoch").reindex(POST)
            x = np.arange(3) + (k - (len(vs) - 1) / 2) * w
            ax.errorbar(x, t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o", color=cmap[v], ms=4, capsize=2,
                        label=VLAB[v])
            for a in AN:
                ax.plot(x + 0.02, t[a], MK[a], mfc="none", mec=cmap[v], ms=3, alpha=0.7)
        ax.set_xticks(range(3), POST)
        _style(ax)
        ax.set_title(lab + "\nchange vs pre", fontsize=9)
    h_, l_ = axs.flat[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(l_), fontsize=9, frameon=False)
    sup = ("NEURAL / BEHAVIOR: FIRST ENLv lick of the trial vs LATER ENLv licks (any time / arrival-corrected / "
           "both restricted to onset >= 1 s, which separates a first-lick effect from a time-since-arrival "
           "effect). Windows end before the cue. PROVISIONAL")
    if len(vs) < 6:
        sup = ("SIMPLIFIED: first vs later ENLv licks, both onset >= 1 s after arrival (separates a first-lick effect "
               "from a time-since-arrival effect); all six variants = lick_enlv_first_later. Windows end before the "
               "cue. PROVISIONAL")
    fig.suptitle(sup, fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _header(fig, 2)
    save_png_svg(fig, FIG / name, dpi=90)
    plt.close(fig)


def draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    FIG.mkdir(parents=True, exist_ok=True)
    fig_arrival(F["arrival"])
    fig_compare(F["compare"])
    fig_compare(F["compare"], ("cued", "ITI all", "ENLv", "ENLv (iti reader)"), "lick_enlv_compare_simple.png")
    fig_perspout(F["TS"], F["AS"])
    fig_sweep(F["T"])
    fig_sweep(F["T"], SWEEP_SIMPLE_V, SWEEP_SIMPLE_M, "lick_enlv_sweep_simple.png")
    fig_firstlater(F["T"])
    fig_firstlater(F["T"], ["first_on1", "later_on1"], "lick_enlv_first_later_simple.png")


# ================================================================================================ main
def run() -> dict:
    L = pd.read_csv(OUT / "enlv_table.csv.gz")
    for c in ("first", "clean_start", "contact"):
        L[c] = L[c].astype(bool)
    Ss, SPs = [], []
    for v in VARIANTS:
        S, SP = measures(L, v)
        Ss.append(S)
        SPs.append(SP)
        print(f"[{v}] sessions {len(S)}, licks {int(S.n.sum()) if len(S) else 0}", flush=True)
    S, SP = pd.concat(Ss, ignore_index=True), pd.concat(SPs, ignore_index=True)
    S.to_csv(OUT / "enlv_sessions.csv", index=False)
    SP.to_csv(OUT / "enlv_spout_sessions.csv", index=False)
    T, A = tests(S, MEAS, ["variant"])
    TS, AS = tests(SP, SPMEAS, ["variant", "pos"])
    T.to_csv(OUT / "enlv_tests.csv", index=False)
    A.to_csv(OUT / "enlv_animals.csv", index=False)
    TS.to_csv(OUT / "enlv_spout_tests.csv", index=False)
    AS.to_csv(OUT / "enlv_spout_animals.csv", index=False)
    C = compare(T, TS)
    C.to_csv(OUT / "enlv_compare.csv", index=False)
    N = counts(L)
    for k, v in N.items():
        v.to_csv(OUT / f"enlv_counts_{k}.csv", index=False)
    AR = arrival(L)
    AR["rows"].to_csv(OUT / "enlv_arrival_kernels.csv", index=False)
    AR["stab"].to_csv(OUT / "enlv_arrival_stability.csv", index=False)
    AR["free"].to_csv(OUT / "enlv_arrival_lickfree.csv", index=False)
    AR["onsets"] = {a: {"all": g.t_from_strobe.to_numpy(), "first": g[g["first"]].t_from_strobe.to_numpy()}
                    for a, g in L[L.epoch == "pre"].groupby("animal")}
    pd.DataFrame([{"reader": k, "sigma_mm": v} for k, v in SIGMA.items()]).to_csv(OUT / "enlv_sigma.csv", index=False)
    report(T, A, C, N, AR)
    return {"T": T, "TS": TS, "AS": AS, "compare": C, "arrival": AR}


def report(T, A, C, N, AR) -> None:
    pd.set_option("display.width", 260, "display.max_rows", 500, "display.max_columns", 40)
    show = ["bal_all", "bal_spout", "d_raw_ci_spmean", "off_iti", "off_trial", "off_enlv", "offsp_iti", "offsp_enlv",
            "lerr", "aim_err"]
    print("\n===== PRIMARY: PER-ANIMAL change vs pre (rule 8)")
    a = A[(A.variant == PRIMARY) & A.measure.isin(show) & (A.epoch != "pre")]
    print(a.pivot_table(index=["measure", "animal"], columns="epoch", values="delta").reindex(columns=POST)
          .round(3).to_string())
    print("\n===== n sessions / animal x epoch (primary)")
    print(A[(A.variant == PRIMARY) & (A.measure == "bal_all")].pivot_table(index="animal", columns="epoch",
                                                                           values="n_sess").to_string())
    print("\n===== POOLED (boot_delta) for every variant")
    t = T[T.measure.isin(show)]
    print(t[["variant", "measure", "epoch", "point", "lo", "hi", "n_same", "n_animals", "star", "flag"]]
          .round(3).to_string(index=False))
    print("\n===== COMPARE\n" + C[["quantity", "source", "epoch", "point", "lo", "hi", "n_same", "n", "flag"]]
          .round(3).to_string(index=False))
    print("\n===== cue-rule removals (sum over spouts)\n" + N["cue_rule"].groupby(["variant", "animal", "epoch"])[
        ["n_before", "n_removed"]].sum().reset_index().to_string(index=False))
    print("\n===== first-lick latency\n" + N["first_latency"].round(2).to_string(index=False))
    print("\n===== arrival kernels, pool, per animal x epoch\n" + AR["rows"][AR["rows"].area.str.startswith("pool")]
          .round(3).to_string(index=False))
    fr = AR["free"]
    print("\n===== lick-free trials (model-free arrival check), pool\n" + fr[fr.area.str.startswith("pool")].round(3)
          .to_string(index=False))
    print("\n===== kernel stability\n" + AR["stab"].groupby(["animal", "epoch"]).agg(
        n_sess=("label", "size"), n_trials=("n_trials", "sum"), cv_r2=("cv_r2", "median"),
        splithalf=("splithalf_r_pool_median", "median"), lickfree=("n_lickfree_trials", "sum")).round(2).to_string())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--table" in sys.argv:
        table()
        return 0
    if figdata.plot_only():
        F = figdata.load_fd("lick_enlv", FD)
    else:
        F = run()
        figdata.save_fd("lick_enlv", F, FD)
    draw(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
