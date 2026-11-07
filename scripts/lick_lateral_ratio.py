"""CONTRALESIONAL / IPSILESIONAL response RATIO per spout and epoch (Priya 10-28: "can we look at the ALM contra/ipsi, tjM1
contra/ipsi and SSp-m and n contra/ipsi ratio per-spout traces over epochs? I'm curious if the ratio (1) is different for
different spout targets and (2) changes post-stroke").

Same traces as 14.7 (`lick_cued_ramp` cued_wide files): first cued lick by TARGET spout (lick-aligned, baseline -1.0..-0.5
s), the same trials cue-aligned (baseline -1.0..-0.5 s before the cue), ITI bout starts by the PREVIOUS trial's spout
(baseline -1.0..-0.5 s). Pairs: ALM, tjM1, SSp-m, SSp-n, each X_right (contralesional) / X_left (ipsilesional), raw dF/F
(the same units in both hemispheres).
  TRACES: per animal x epoch x spout, the session-mean contra curve / the session-mean ipsi curve, shown only where the
          ipsi curve is >= RATIO_MIN of its 0..1.5 s peak (near-zero denominators blow up); mean over the animals defined at
          each lag (>= MIN_TRACE_ANIMALS; dotted where fewer than all). Descriptive.
  AUC RATIO: per session x spout (>= MIN_SPOUT licks), log2(contra AUC / ipsi AUC) over 0..0.5 s (and 0..1.0 s); NaN if
          either AUC <= 0 (counted). 0 = equal, +1 = contra twice ipsi.
  (1) spout effect: pre sessions, mixed model logr ~ C(spout) + (1|animal) vs logr ~ 1 + (1|animal), likelihood ratio
      (ML fits); per-animal pre table first (rule 8).
  (2) post-stroke change: per spout and epoch, `analysis_kit.boot_delta` (paired animals > sessions) of logr vs pre at that
      spout; and whether the SPOUT PATTERN changes: logr ~ C(spout) * C(epoch) + (1|animal) vs the additive model (LR).
Outputs <OUT_W>/lateral_ratio_sessions.csv, lateral_ratio_tests.csv, lateral_ratio_models.csv; figures
lick_lateral_ratio_traces_<set>.png, lick_lateral_ratio_auc.png (figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_cued_ramp as cr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

OROFACIAL = ["ALM", "tjM1", "SSp-m", "SSp-n"]
PAIRS = [*OROFACIAL, "orofacial"]   # + the four pooled (Priya 10-28: "pool orofacial sensorimotor -- less noisy?")
RINGS = ["ring_close", "ring_far"]                     # mean over the ring's three spouts per session (not lick-pooled)
SIDES = ["side_ipsi", "side_mid", "side_contra"]       # close + far of one side (spout names end in _ipsi/_mid/_contra)
SETS = {"cued": ("lick", cr.BASE_SPOUT, "time from first cued lick (s)", "first cued lick, by TARGET spout"),
        "cue": ("cue", cr.BASE_C, "time from cue (s)", "cue-aligned, by target spout"),
        "iti": ("iti", cr.BASE_SPOUT, "time from ITI bout-start lick (s)", "ITI bout start, by PREVIOUS trial's spout")}
WINDOWS = ((0.0, 1.0), (0.0, 0.5))      # first = primary figures (0..1.0 s default since 10-28 evening)
RATIO_MIN = 0.2
MIN_TRACE_ANIMALS = 2    # traces: mean over the animals defined at a lag (dotted where fewer than all)
EP = ["acute", "subacute", "chronic"]
L = cr.LAGS_W


def _session_curves(D: dict) -> pd.DataFrame:
    """One row per session x set x spout x pair: session-mean baseline-subtracted contra and ipsi curves (raw dF/F)."""
    import iti_lick_order as lo
    rows = []
    for x, d in D.items():
        for st, (key, base, _xl, _t) in SETS.items():
            X = d[key]
            pos = d["iti_pos"] if key == "iti" else d["pos"]
            # "pooled" = every lick of the set, spouts not separated (Priya 10-28: "pooled across-spout ratio traces")
            # SIDE groups (Priya 10-28: "pool ipsi / mid / contra"): close + far of one side together
            for p in [*lo.POS, *SIDES, "pooled"]:
                if p == "pooled":
                    P = X
                elif p in SIDES:
                    P = X[np.isin(pos, [q for q in lo.POS if q.endswith(p.split("_")[1])])]
                else:
                    P = X[pos == p]
                if len(P) < (cr.MIN_SPOUT if p in lo.POS else cr.MIN_N):
                    continue
                P = P - np.nanmean(P[:, (L >= base[0]) & (L <= base[1]), :], 1, keepdims=True)
                m = np.nanmean(P, 0)
                for a in PAIRS:
                    if a == "orofacial":   # pooled orofacial sensorimotor: mean of the four areas' curves per hemisphere
                        c = [d["cols"].index(f"{q}_right") for q in OROFACIAL]
                        i = [d["cols"].index(f"{q}_left") for q in OROFACIAL]
                        mc, mi = m[:, c].mean(1), m[:, i].mean(1)
                    else:
                        mc, mi = m[:, d["cols"].index(f"{a}_right")], m[:, d["cols"].index(f"{a}_left")]
                    rows.append({"label": x, "animal": d["animal"], "epoch": d["epoch"], "set": st, "spout": p,
                                 "pair": a, "n": len(P), "contra": mc, "ipsi": mi})
    return pd.DataFrame(rows)


def _traces(C: pd.DataFrame) -> dict:
    out = {}
    for (st, a, p, ep), g in C.groupby(["set", "pair", "spout", "epoch"]):
        R = []
        for _an, h in g.groupby("animal"):
            c = np.nanmean(np.stack(h.contra.to_numpy()), 0)
            i = np.nanmean(np.stack(h.ipsi.to_numpy()), 0)
            pk = np.nanmax(i[(L >= 0) & (L <= 1.5)])
            R.append(np.where(i >= RATIO_MIN * pk, c / i, np.nan))
        out[(st, a, p, ep)] = np.asarray(R)            # animals x lags (NaN where that animal's ipsi curve is masked)
    return out


def _trace_stats(R: np.ndarray, min_animals: int = MIN_TRACE_ANIMALS):
    """Mean over the animals defined at each lag (>= min_animals), SEM, and the count (Priya 10-28: "don't have to have
    every animal defined; dotted line where < 4")."""
    n = np.isfinite(R).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(n >= min_animals, np.nanmean(R, 0), np.nan)
        se = np.where(n >= 2, np.nanstd(R, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
    return mu, se, n


def _plot_trace(ax, R: np.ndarray, color) -> None:
    """Solid where every animal of the group is defined, dotted where fewer (but >= MIN_TRACE_ANIMALS)."""
    mu, se, n = _trace_stats(R)
    full = n == len(R)
    ax.plot(L, np.where(full, mu, np.nan), color=color, lw=1.9)
    ax.plot(L, np.where(~full, mu, np.nan), color=color, lw=1.4, ls=":")
    ax.fill_between(L, mu - se, mu + se, color=color, alpha=0.12, lw=0)


def _lr(full: str, red: str, df: pd.DataFrame) -> tuple[float, float, int]:
    import statsmodels.formula.api as smf
    from scipy import stats
    f1 = smf.mixedlm(full, df, groups=df["animal"]).fit(reml=False)
    f0 = smf.mixedlm(red, df, groups=df["animal"]).fit(reml=False)
    k = len(f1.fe_params) - len(f0.fe_params)
    lr = 2 * (f1.llf - f0.llf)
    return float(lr), float(stats.chi2.sf(lr, k)), int(k)


def compute() -> dict:
    from wfield_local import analysis_kit as ak
    D = cr._sessions()
    C = _session_curves(D)
    for w0, w1 in WINDOWS:
        w = (L >= w0) & (L <= w1)
        tag = f"{w0:g}-{w1:g}"
        C[f"c_{tag}"] = [np.trapezoid(v[w], L[w]) for v in C.contra]
        C[f"i_{tag}"] = [np.trapezoid(v[w], L[w]) for v in C.ipsi]
        ok = (C[f"c_{tag}"] > 0) & (C[f"i_{tag}"] > 0)
        C[f"logr_{tag}"] = np.where(ok, np.log2(C[f"c_{tag}"].clip(lower=1e-12) / C[f"i_{tag}"].clip(lower=1e-12)), np.nan)
    S = C.drop(columns=["contra", "ipsi"])
    S.to_csv(il.OUT_W / "lateral_ratio_sessions.csv", index=False)
    pd.set_option("display.width", 230)
    print("== rows with a non-positive AUC (ratio undefined), by set x window:")
    for w0, w1 in WINDOWS:
        tag = f"{w0:g}-{w1:g}"
        print(f"   {tag}: " + ", ".join(f"{st} {S[S.set == st][f'logr_{tag}'].isna().mean():.3f}" for st in SETS))
    import iti_lick_order as lo
    tests, models = [], []
    for (st, a), g in S.groupby(["set", "pair"]):
        for w0, w1 in WINDOWS:
            v = f"logr_{w0:g}-{w1:g}"
            if (w0, w1) == WINDOWS[0]:
                print(f"-- per animal (rule 8) PRE log2(contra/ipsi) {st} {a} {v}\n"
                      + g[g.epoch == "pre"].pivot_table(index="animal", columns="spout", values=v)
                      .reindex(columns=list(lo.POS)).round(2).to_string())
            Gall = g.dropna(subset=[v]).rename(columns={v: "y"})
            G = Gall[Gall.spout.isin(lo.POS)]                 # per-spout rows: models and the spout mean "all"
            GS = Gall[Gall.spout.isin(SIDES)]
            pre = G[G.epoch == "pre"]
            try:
                lr, p_, k = _lr("y ~ C(spout)", "y ~ 1", pre)
                models.append({"set": st, "pair": a, "window": v, "test": "pre: spout effect", "lr": lr, "p": p_, "df": k})
                lr, p_, k = _lr("y ~ C(spout) * C(epoch)", "y ~ C(spout) + C(epoch)", G)
                models.append({"set": st, "pair": a, "window": v, "test": "spout x epoch", "lr": lr, "p": p_, "df": k})
                lr, p_, k = _lr("y ~ C(spout) + C(epoch)", "y ~ C(spout)", G)
                models.append({"set": st, "pair": a, "window": v, "test": "epoch (all spouts)", "lr": lr, "p": p_, "df": k})
                lr, p_, k = _lr("y ~ C(spout)", "y ~ 1", GS[GS.epoch == "pre"])
                models.append({"set": st, "pair": a, "window": v, "test": "pre: side effect", "lr": lr, "p": p_, "df": k})
                lr, p_, k = _lr("y ~ C(spout) * C(epoch)", "y ~ C(spout) + C(epoch)", GS)
                models.append({"set": st, "pair": a, "window": v, "test": "side x epoch", "lr": lr, "p": p_, "df": k})
                # DISTANCE (Priya 10-28: "secondary close vs far pooled figure"): does the post-stroke change differ
                # between the far and the close ring, beyond each spout's own level? per-spout rows, spout fixed
                GD = G.assign(far=G.spout.str.startswith("far").astype(int))
                lr, p_, k = _lr("y ~ C(spout) + C(epoch) * far", "y ~ C(spout) + C(epoch)", GD)
                models.append({"set": st, "pair": a, "window": v, "test": "distance x epoch", "lr": lr, "p": p_, "df": k})
            except Exception as ex:                     # a fit that does not converge is reported, not hidden
                models.append({"set": st, "pair": a, "window": v, "test": f"FAILED {ex}", "lr": np.nan, "p": np.nan})
            for p in [*lo.POS, "all", *SIDES, *RINGS, "ring_diff", "pooled"]:
                if p in ("all", *RINGS):   # one value per session: mean over its spouts (of the ring)
                    h = G if p == "all" else G[G.spout.str.startswith(p.split("_")[1])]
                    h = h.groupby(["label", "animal", "epoch"]).y.mean().reset_index()
                elif p == "ring_diff":     # far ring minus close ring per session: does far shift MORE than close?
                    rg = G.assign(ring=np.where(G.spout.str.startswith("far"), "far", "close")) \
                        .groupby(["label", "animal", "epoch", "ring"]).y.mean().unstack("ring")
                    h = (rg["far"] - rg["close"]).rename("y").dropna().reset_index()
                else:
                    h = Gall[Gall.spout == p]
                ref = {an: t.y.to_numpy() for an, t in h[h.epoch == "pre"].groupby("animal")}
                lev = {an: t.y.mean() for an, t in h[h.epoch == "pre"].groupby("animal")}
                tests.append({"set": st, "pair": a, "window": v, "spout": p, "epoch": "pre",
                              "level": float(np.mean(list(lev.values()))) if lev else np.nan,
                              "level_se": float(np.std(list(lev.values()), ddof=1) / np.sqrt(len(lev))) if len(lev) > 1 else np.nan,
                              "n_animals": len(lev)})
                for ep in EP:
                    post = {an: t.y.to_numpy() for an, t in h[h.epoch == ep].groupby("animal")}
                    com = sorted(an for an in set(post) & set(ref) if len(post[an]) and len(ref[an]))
                    if len(com) < 3:
                        continue
                    iv = ak.boot_delta({an: post[an] for an in com}, {an: ref[an] for an in com}, np.random.default_rng(0))
                    lv = [post[an].mean() for an in com]
                    tests.append({"set": st, "pair": a, "window": v, "spout": p, "epoch": ep,
                                  "level": float(np.mean(lv)), "level_se": float(np.std(lv, ddof=1) / np.sqrt(len(lv))),
                                  "delta": iv.point, "lo": iv.lo, "hi": iv.hi, "sig": bool(iv.lo > 0 or iv.hi < 0),
                                  "n_animals": len(com),
                                  **{f"d_{an}": post[an].mean() - ref[an].mean() for an in com}})
    T, M = pd.DataFrame(tests), pd.DataFrame(models)
    T.to_csv(il.OUT_W / "lateral_ratio_tests.csv", index=False)
    M.to_csv(il.OUT_W / "lateral_ratio_models.csv", index=False)
    print("== mixed models (LR tests)\n" + M.round(4).to_string(index=False))
    t = T[(T.window == "logr_0-0.5") & (T.epoch != "pre")].copy()
    t["c"] = [f"{d:+.2f}{'*' if s else ''}" for d, s in zip(t.delta, t.sig)]
    print("== change vs pre of log2(contra/ipsi), 0..0.5 s\n"
          + t.pivot_table(index=["set", "pair", "spout"], columns="epoch", values="c", aggfunc="first")
          .reindex(columns=EP).to_string())
    return {"T": T, "M": M, "tr": _traces(C), "pos": list(lo.POS)}


def _sessions_table(F: dict):
    """The per-session table behind the AUC figures (for the session dots; A13): F["S"] when the figure data carry it,
    else <OUT_W>/lateral_ratio_sessions.csv written by `compute`; None if neither exists (dots then skipped)."""
    if F.get("S") is not None:
        return F["S"]
    f = il.OUT_W / "lateral_ratio_sessions.csv"
    return pd.read_csv(f) if f.exists() else None


def _session_values(S, st: str, a: str, v: str, p: str, pos) -> pd.DataFrame:
    """Per session (animal, epoch, y) for one AUC cell, built exactly like `compute`'s test rows: a spout row as is;
    "all" / a ring = the session's mean over its spouts (of the ring); empty when S is None."""
    if S is None:
        return pd.DataFrame(columns=["animal", "epoch", "y"])
    g = S[(S.set == st) & (S.pair == a)].dropna(subset=[v]).rename(columns={v: "y"})
    if p in ("all", *RINGS):
        h = g[g.spout.isin(pos)]
        if p != "all":
            h = h[h.spout.str.startswith(p.split("_")[1])]
        return h.groupby(["label", "animal", "epoch"]).y.mean().reset_index()
    return g[g.spout == p][["animal", "epoch", "y"]]


def _dots(ax, d: pd.DataFrame, x: float, ep: str, seed: int, sub=None) -> None:
    """Session dots at x: OPEN markers, shape = animal, colour = epoch (A13, Priya 11-07). `sub` = per-animal value
    subtracted (the rings' change vs pre)."""
    from wfield_local import cat_points as cp
    q = d[d.epoch == ep]
    if q.empty:
        return
    y = q.y.to_numpy(float) - (q.animal.map(sub).to_numpy(float) if sub is not None else 0.0)
    cp.session_dots(ax, x, y, q.animal, cp.epoch_color(ep), jitter=0.035, seed=seed, ms=3.0, alpha=0.5)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    pos, T, M = F["pos"], F["T"], F["M"]
    out = il.OUT_W / "figures"
    for st, (_k, base, xl, what) in SETS.items():
        fig, axs = plt.subplots(len(PAIRS), len(pos), figsize=(22, 14), sharex=True, sharey="row")
        for r, a in enumerate(PAIRS):
            for c, p in enumerate(pos):
                ax = axs[r, c]
                for ep in il.EP4:
                    if (st, a, p, ep) not in F["tr"]:
                        continue
                    _plot_trace(ax, F["tr"][(st, a, p, ep)], col[ep])
                ax.axhline(1, color="0.5", lw=0.8, ls="--")
                ax.axvline(0, color="0.6", lw=0.6)
                ax.set_xlim(-0.5, 2)
                ax.set_yscale("log", base=2)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if r == 0:
                    ax.set_title(p.replace("_", " "), fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{a}: contra / ipsi\n(log scale; 1 = equal)", fontsize=8.5)
                if r == len(PAIRS) - 1:
                    ax.set_xlabel(xl, fontsize=8)
        fig.legend(handles=[Line2D([], [], color=col[e], lw=2.2, label=e) for e in il.EP4], loc="lower center", ncol=4,
                   fontsize=9.5, frameon=False)
        fig.suptitle(f"NEURAL: CONTRALESIONAL / IPSILESIONAL dF/F ratio over time, {what}, per spout (baseline "
                     f"{base[0]:g}..{base[1]:g} s); per animal (mean contra curve) / (mean ipsi curve), shown only where the "
                     f"ipsi curve is >= {RATIO_MIN:.0%} of its 0..1.5 s peak;\nline = mean over the animals defined at each lag "
                     "(solid = all animals, dotted = 2-3 animals), band = SEM over animals. Descriptive; tests on the AUC ratio in lick_lateral_ratio_auc.png. "
                     "PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        save_png_svg(fig, out / f"lick_lateral_ratio_traces_{st}.png", dpi=95)
        plt.close(fig)
    # AUC summary: rows = sets, columns = pairs; x = spouts, one marker per epoch (+ the session dots, A13 11-07)
    from wfield_local import cat_points as cp
    from wfield_local.sig_marks import reserve_headroom, star_top
    S = _sessions_table(F)
    for w0, w1 in WINDOWS:
        v, wl = f"logr_{w0:g}-{w1:g}", f"{w0:g}..{w1:g} s"
        fig, axs = plt.subplots(len(SETS), len(PAIRS), figsize=(26, 12.5), squeeze=False)
        for r, st in enumerate(SETS):
            for c, a in enumerate(PAIRS):
                ax = axs[r, c]
                t = T[(T.set == st) & (T.pair == a) & (T.window == v)]
                xs = [*pos, "all"]
                for k, ep in enumerate(il.EP4):
                    q = t[t.epoch == ep].set_index("spout").reindex(xs)
                    x = np.arange(len(xs)) + (k - 1.5) * 0.17
                    for xx, p_ in zip(x, xs):
                        _dots(ax, _session_values(S, st, a, v, p_, pos), xx, ep, seed=k)
                    cp.summary_point(ax, x, q.level.to_numpy(float), q.level_se.to_numpy(float), col[ep], ms=4.5,
                                     capsize=1.5)
                    if ep != "pre":
                        for xx, (_s, row) in zip(x, q.iterrows()):
                            if row.get("sig") is True:                 # top band (Priya 10-30)
                                star_top(ax, xx, "*", col[ep], row=k - 1, fontsize=10)
                ax.axhline(0, color="0.6", lw=0.7, ls="--")
                ax.axvline(len(pos) - 0.5, color="0.85", lw=0.8)
                ax.set_xticks(range(len(xs)), [s.replace("_", "\n") for s in xs], fontsize=7.5)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                m = M[(M.set == st) & (M.pair == a) & (M.window == v)].set_index("test")
                pz = {k_: m.p.get(k_, np.nan) for k_ in ("pre: spout effect", "spout x epoch", "epoch (all spouts)")}
                ax.set_title(f"{a}, {st}\npre spout effect p = {pz['pre: spout effect']:.2g}; spout x epoch p = "
                             f"{pz['spout x epoch']:.2g}; epoch p = {pz['epoch (all spouts)']:.2g}", fontsize=8.5)
                if c == 0:
                    ax.set_ylabel(f"{SETS[st][3]}\nlog2(contra AUC / ipsi AUC), {wl}\n(0 = equal; +1 = contra 2x)", fontsize=8)
        fig.legend(handles=cp.legend_handles(sorted(S.animal.unique()) if S is not None else [], il.EP4,
                                             epoch_label="{e}: mean +- SEM over animals"), loc="lower center",
                   ncol=8, fontsize=9.5, frameon=False)
        reserve_headroom(fig)
        fig.suptitle(f"NEURAL: contralesional / ipsilesional response ratio per spout and epoch: log2 of the AUC ratio {wl} "
                     "(per session, spouts with >= 5 licks), mean +- SEM over animals; 'all' = each session's mean over spouts; "
                     "open = sessions (shape = animal, colour = epoch).\n"
                     "* (top, one row per post epoch) = change vs pre at that spout, 95% paired nested bootstrap CI excludes 0. Titles: mixed models (random "
                     "intercept per animal, ML, likelihood ratio): (1) spout effect in PRE sessions; (2) spout x epoch "
                     "interaction; epoch main effect. PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        save_png_svg(fig, out / ("lick_lateral_ratio_auc.png" if (w0, w1) == WINDOWS[0] else f"lick_lateral_ratio_auc_{w0:g}-{w1:g}.png"), dpi=100)
        plt.close(fig)


SIDE_LAB = {"side_ipsi": "ipsi spouts\n(close + far)", "side_mid": "mid spouts\n(close + far)",
            "side_contra": "contra spouts\n(close + far)"}


def plot_sides(F: dict) -> None:
    """Side groups (close + far of one side; Priya 10-28): traces per set (rows = pairs, columns = sides) and the AUC
    ratio per set x pair with x = side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    T, M = F["T"], F["M"]
    out = il.OUT_W / "figures"
    leg = [Line2D([], [], color=col[e], lw=2.2, label=e) for e in il.EP4]
    for st, (_k, base, xl, what) in SETS.items():
        fig, axs = plt.subplots(len(PAIRS), len(SIDES), figsize=(14, 14), sharex=True, sharey="row", squeeze=False)
        for r, a in enumerate(PAIRS):
            for c, sd in enumerate(SIDES):
                ax = axs[r, c]
                for ep in il.EP4:
                    if (st, a, sd, ep) in F["tr"]:
                        _plot_trace(ax, F["tr"][(st, a, sd, ep)], col[ep])
                ax.axhline(1, color="0.5", lw=0.8, ls="--")
                ax.axvline(0, color="0.6", lw=0.6)
                ax.set_xlim(-0.5, 2)
                ax.set_yscale("log", base=2)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if r == 0:
                    ax.set_title(SIDE_LAB[sd].replace("\n", " "), fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{a}: contra / ipsi\n(log; 1 = equal)", fontsize=9)
                if r == len(PAIRS) - 1:
                    ax.set_xlabel(xl, fontsize=8.5)
        fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
        fig.suptitle(f"NEURAL: CONTRALESIONAL / IPSILESIONAL dF/F ratio over time, {what}, spouts POOLED BY SIDE (close + far; "
                     f"baseline {base[0]:g}..{base[1]:g} s).\nPer animal (mean contra curve) / (mean ipsi curve), shown where the "
                     f"ipsi curve is >= {RATIO_MIN:.0%} of its peak; mean over the animals defined (solid = all, dotted = 2-3), "
                     "band = SEM over animals. Descriptive. PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        save_png_svg(fig, out / f"lick_lateral_ratio_side_traces_{st}.png", dpi=100)
        plt.close(fig)
    for w0, w1 in WINDOWS:
        v = f"logr_{w0:g}-{w1:g}"
        fig, axs = plt.subplots(len(SETS), len(PAIRS), figsize=(22, 11.5), sharey="row", squeeze=False)
        for r, st in enumerate(SETS):
            for c, a in enumerate(PAIRS):
                ax = axs[r, c]
                t = T[(T.set == st) & (T.pair == a) & (T.window == v)]
                for k, ep in enumerate(il.EP4):
                    q = t[t.epoch == ep].set_index("spout").reindex(SIDES)
                    x = np.arange(len(SIDES)) + (k - 1.5) * 0.15
                    ax.errorbar(x, q.level, yerr=q.level_se, fmt="o", color=col[ep], ms=5, capsize=1.5, lw=1)
                    if ep != "pre":
                        for xx, (_s, row) in zip(x, q.iterrows()):
                            if row.get("sig") is True:
                                ax.text(xx, row.level + (row.level_se if np.isfinite(row.level_se) else 0), "*",
                                        color=col[ep], ha="center", va="bottom", fontsize=12, fontweight="bold")
                ax.axhline(0, color="0.6", lw=0.7, ls="--")
                ax.set_xticks(range(len(SIDES)), [SIDE_LAB[s] for s in SIDES], fontsize=8)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                m = M[(M.set == st) & (M.pair == a) & (M.window == v)].set_index("test")
                ps, pi = m.p.get("pre: side effect", np.nan), m.p.get("side x epoch", np.nan)
                ax.set_title(f"{a}, {st}\npre side effect p = {ps:.2g}; side x epoch p = {pi:.2g}", fontsize=8.5)
                if c == 0:
                    ax.set_ylabel(f"{SETS[st][3]}\nlog2(contra / ipsi AUC), {w0:g}..{w1:g} s", fontsize=8)
        fig.legend(handles=[Line2D([], [], color=col[e], marker="o", lw=0, label=e) for e in il.EP4],
                   loc="lower center", ncol=4, fontsize=9.5, frameon=False)
        fig.suptitle(f"NEURAL: contralesional / ipsilesional response ratio, spouts POOLED BY SIDE (close + far): log2 of the "
                     f"per-session AUC ratio {w0:g}..{w1:g} s, mean +- SEM over animals; * = change vs pre, 95% paired nested "
                     "bootstrap CI excludes 0.\nTitles: mixed models with sessions as observations (anti-conservative at "
                     "n = 4 animals): side effect in PRE sessions; side x epoch interaction. PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        save_png_svg(fig, out / ("lick_lateral_ratio_side_auc.png" if (w0, w1) == WINDOWS[0]
                           else f"lick_lateral_ratio_side_auc_{w0:g}-{w1:g}.png"), dpi=100)
        plt.close(fig)


def plot_rings(F: dict) -> None:
    """Close vs far ring (each session's mean over the ring's three spouts): CHANGE vs pre per epoch with 95% CI (the ring
    LEVEL averages opposite-side ratios, so only the change is drawn); titles = distance x epoch LR test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE as col
    from wfield_local.sig_marks import star_top
    T, M = F["T"], F["M"]
    S = _sessions_table(F)
    pos = F["pos"]
    for w0, w1 in WINDOWS:
        v = f"logr_{w0:g}-{w1:g}"
        fig, axs = plt.subplots(len(SETS), len(PAIRS), figsize=(21, 11), sharey="row", squeeze=False)
        for r, st in enumerate(SETS):
            for c, a in enumerate(PAIRS):
                ax = axs[r, c]
                t = T[(T.set == st) & (T.pair == a) & (T.window == v)]
                for k, ep in enumerate(EP):
                    q = t[t.epoch == ep].set_index("spout").reindex(RINGS)
                    x = np.arange(len(RINGS)) + (k - 1) * 0.18
                    for xx, rg in zip(x, RINGS):              # session dots: change vs the animal's pre mean (A13)
                        dv = _session_values(S, st, a, v, rg, pos)
                        base = dv[dv.epoch == "pre"].groupby("animal").y.mean()
                        _dots(ax, dv, xx, ep, seed=k, sub=base)
                    ax.errorbar(x, q.delta, yerr=[q.delta - q.lo, q.hi - q.delta], fmt="none", ecolor=col[ep], capsize=2)
                    sg = q.sig.fillna(False).astype(bool).to_numpy()
                    ax.plot(x[sg], q.delta[sg], "o", color=col[ep], ms=6, zorder=3)
                    ax.plot(x[~sg], q.delta[~sg], "o", mec=col[ep], mfc="white", ms=6, zorder=3)
                    for xx, (_s, row) in zip(x, q.iterrows()):   # * = change vs pre, CI excludes 0 (top band)
                        if row.get("sig") is True:
                            star_top(ax, xx, "*", col[ep], row=1, fontsize=11)
                # far minus close per epoch (session difference, paired nested bootstrap vs pre)
                dd = t[(t.spout == "ring_diff")].set_index("epoch")
                txt = []
                for ep in EP:
                    if ep in dd.index and np.isfinite(dd.loc[ep, "delta"]):
                        r_ = dd.loc[ep]
                        s_ = f"{r_.delta:+.2f}"
                        txt.append(rf"$\mathbf{{{s_}}}$*" if r_.sig is True else s_)
                    else:
                        txt.append("n/a")
                ax.text(0.5, 0.97, "far - close, a / s / c: " + " / ".join(txt), transform=ax.transAxes, ha="center",
                        va="top", fontsize=8)
                ax.axhline(0, color="0.6", lw=0.7, ls="--")
                ax.set_xticks(range(len(RINGS)), ["close ring\n(3 spouts)", "far ring\n(3 spouts)"])
                ax.set_xlim(-0.6, 1.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                m = M[(M.set == st) & (M.pair == a) & (M.window == v)].set_index("test")
                pd_ = m.p.get("distance x epoch", np.nan)
                ps = "< 1e-10" if np.isfinite(pd_) and pd_ < 1e-10 else f"= {pd_:.2g}"
                ttl = f"distance x epoch p {ps}"
                ax.set_title(f"{a}, {st}: " + (rf"$\mathbf{{{ttl.replace(' ', '~')}}}$" if pd_ < 0.05 else ttl),
                             fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{SETS[st][3]}\nchange in log2(contra / ipsi), {w0:g}..{w1:g} s", fontsize=8)
            # ONCE per shared-y row (it was per axes, which compounded the headroom five times): top band = text + stars
            lo_, hi_ = axs[r, 0].get_ylim()
            axs[r, 0].set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
        from wfield_local import cat_points as cp
        fig.legend(handles=cp.animal_handles(sorted(S.animal.unique()) if S is not None else [])
                   + [Line2D([], [], color=col[e], marker="o", lw=0, label=e) for e in EP]
                   + [Line2D([], [], color="0.4", marker="o", mfc="white", lw=0, label="hollow = CI includes 0")],
                   loc="lower center", ncol=8, fontsize=9.5, frameon=False)
        fig.suptitle(f"Contra / ipsi ratio, CLOSE vs FAR ring: change vs pre of log2(contra / ipsi AUC {w0:g}..{w1:g} s), "
                     "each session's mean over the ring's three spouts (ipsi, mid, contra); mean of animals, 95% paired "
                     "nested bootstrap CI.\nTitles: mixed model on per-spout rows, y ~ spout + epoch * far vs y ~ spout + "
                     "epoch (sessions as observations; anti-conservative). PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.05, 1, 0.92))
        save_png_svg(fig, il.OUT_W / "figures" / ("lick_lateral_ratio_rings.png" if (w0, w1) == WINDOWS[0]
                                            else f"lick_lateral_ratio_rings_{w0:g}-{w1:g}.png"), dpi=100)
        plt.close(fig)


def plot_pooled(F: dict) -> None:
    """Pooled across spouts (every lick of the set): contra / ipsi traces per set x pair, and the AUC ratio per pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    T = F["T"]
    out = il.OUT_W / "figures"
    fig, axs = plt.subplots(len(SETS), len(PAIRS), figsize=(23, 11), sharey=True, squeeze=False)
    for r, (st, (_k, base, xl, what)) in enumerate(SETS.items()):
        for c, a in enumerate(PAIRS):
            ax = axs[r, c]
            for ep in il.EP4:
                if (st, a, "pooled", ep) in F["tr"]:
                    _plot_trace(ax, F["tr"][(st, a, "pooled", ep)], col[ep])
            ax.axhline(1, color="0.5", lw=0.8, ls="--")
            ax.axvline(0, color="0.6", lw=0.6)
            ax.set_xlim(-0.5, 2)
            ax.set_yscale("log", base=2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(f"{a}: contralesional / ipsilesional", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{what.split(',')[0]}\n(baseline {base[0]:g}..{base[1]:g} s)\ncontra / ipsi (log; 1 = equal)",
                              fontsize=8.5)
            ax.set_xlabel(xl, fontsize=8)
    fig.legend(handles=[Line2D([], [], color=col[e], lw=2.2, label=e) for e in il.EP4], loc="lower center", ncol=4,
               fontsize=9.5, frameon=False)
    fig.suptitle("NEURAL: CONTRALESIONAL / IPSILESIONAL dF/F ratio over time, POOLED over spouts (every lick of the set): rows "
                 "= first cued lick, cue, ITI bout start (no lick for >= 2 s).\nPer animal (mean contra curve) / (mean ipsi "
                 f"curve), shown only where the ipsi curve is >= {RATIO_MIN:.0%} of its 0..1.5 s peak; mean over the animals defined (solid = all, dotted = 2-3), band "
                 "= SEM over animals. Descriptive. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_png_svg(fig, out / "lick_lateral_ratio_pooled_traces.png", dpi=100)
    plt.close(fig)
    fig, axs = plt.subplots(len(WINDOWS), len(SETS), figsize=(17, 9), squeeze=False)
    for r, (w0, w1) in enumerate(WINDOWS):
        v = f"logr_{w0:g}-{w1:g}"
        for c, st in enumerate(SETS):
            ax = axs[r, c]
            t = T[(T.set == st) & (T.window == v) & (T.spout == "pooled")]
            for k, ep in enumerate(il.EP4):
                q = t[t.epoch == ep].set_index("pair").reindex(PAIRS)
                x = np.arange(len(PAIRS)) + (k - 1.5) * 0.17
                ax.errorbar(x, q.level, yerr=q.level_se, fmt="o", color=col[ep], ms=5, capsize=1.5, lw=1)
                if ep != "pre":
                    for xx, (_s, row) in zip(x, q.iterrows()):
                        if row.get("sig") is True:
                            ax.text(xx, row.level + (row.level_se if np.isfinite(row.level_se) else 0), "*",
                                    color=col[ep], ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(len(PAIRS)), PAIRS)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(SETS[st][3].split(",")[0], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"log2(contra AUC / ipsi AUC), {w0:g}..{w1:g} s\n(0 = equal; +1 = contra 2x)", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], color=col[e], marker="o", lw=0, label=e) for e in il.EP4], loc="lower center",
               ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle("NEURAL: contralesional / ipsilesional response ratio POOLED over spouts: log2 of the per-session AUC ratio, "
                 "mean +- SEM over animals; * = change vs pre, 95% paired nested bootstrap CI (animals > sessions) excludes 0."
                 "\nColumns: first cued lick, cue, ITI bout start (>= 2 s without licking). PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save_png_svg(fig, out / "lick_lateral_ratio_pooled_auc.png", dpi=100)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_lateral_ratio"))
        plot_pooled(load_fd("lick_lateral_ratio"))
        plot_sides(load_fd("lick_lateral_ratio"))
        plot_rings(load_fd("lick_lateral_ratio"))
        return 0
    F = compute()
    save_fd("lick_lateral_ratio", F)
    plot(F)
    plot_pooled(F)
    plot_sides(F)
    plot_rings(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
