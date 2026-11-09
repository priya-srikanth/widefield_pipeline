"""SAME-TARGET SAME-PLACE check of the like-for-like offset (overnight queue 5b, approved backlog; Priya 10-27: separate
"more misses" from "each lick read differently").

The like-for-like reference compares each lick with the animal's PRE licks (any spout) that landed at the same place.
Pre licks at a place come mostly from the spout that sits there, so a post-stroke MISS aimed at spout A that lands
where spout B's licks land is compared with B licks. If the decoder carries some TARGET information, such a miss reads
toward A for that reason alone, and a session with more misses shows a larger offset with no change in how any one
lick is read (COMPOSITION). Here the reference is restricted to pre licks with the SAME TARGET that landed at the same
place (Gaussian on landing within target spout, likeforlike method "nw"; pre licks leave-one-pre-session-out), so target
identity is matched as well as landing. Target: cued licks = the cued spout; ITI licks = the previous trial's spout (as
everywhere in the ITI family; the spout is retracted during the ITI).
Read-outs (session means >= MIN licks; change vs pre = paired nested bootstrap, per-animal table first, rule 8):
  any_all   any-spout same-place offset on ALL licks (the published number; sigma by held-out pre error)
  any_cov   the same, on the licks that HAVE a same-target same-place reference (coverage-matched)
  tt        same-target same-place offset on those licks
  tt_minus  tt - any_cov per lick (what matching the target removes)
any_all vs any_cov = what the coverage restriction does (acute large misses often have no same-target pre licks that
landed that far out); any_cov vs tt = composition. tt ~ any_cov and != 0 -> each lick is read differently; tt ~ 0 while
any_cov != 0 -> the offset is target composition. Coverage per set x epoch x spout and per animal is reported.
Same-target sigma is chosen by its own held-out pre error (pools are per spout, so sparser); the any-spout sigma is the
published one.
Outputs <OUT_W>/sametarget_{coverage,sigma_cv,sessions,tests}.csv, figures/lick_sametarget_place.png;
figdata/lick_sametarget_place.pkl (--plot-only redraws from it).
Usage: python scripts/lick_same_target_place.py [--plot-only] [--animals PS94 ...]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
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
POS = lo.POS
STEM = f"lick_sametarget_place{lo.SRC_TAG}"
MIN = 8
VALS = ("any_all", "any_cov", "tt", "tt_minus")
# panel titles say what "covered" means (Priya 10-28 asked): licks that HAVE a same-target same-place pre reference
VLAB = {"any_all": "1. vs pre licks of ANY spout, same place\n(all licks = published 12.1)",
        "any_cov": "2. vs ANY spout, same place, only licks\nwith a same-target reference ('covered')",
        "tt": "3. vs pre licks of the SAME TARGET,\nsame place (same licks as 2)",
        "tt_minus": "4. = 3 minus 2: what also matching the\ntarget removes (- = target composition)"}


def per_lick(animals: list[str] | None) -> tuple[pd.DataFrame, dict]:
    P = il.subset_preds(pd.read_csv(lo.preds_path()))
    if animals:
        P = P[P.animal.isin(animals)]
    P = P[np.isfinite(P[["act", "dec"]].to_numpy(float)).all(1)].reset_index(drop=True)
    sg, cv = lfl.choose(P, "act", "dec", lfl.SIGMA_MM, "nw")
    P["any_all"] = P.dec - lfl.ref_table(P, "act", "dec", sg, "nw").ref
    # same target: the reference groups are animal x target spout (ref_table groups by set x animal)
    Q = P.assign(animal=P.animal + "|" + P.pos)
    sg_t, cv_t = lfl.choose(Q, "act", "dec", lfl.SIGMA_MM, "nw")
    P["tt"] = P.dec - lfl.ref_table(Q, "act", "dec", sg_t, "nw").ref
    cov = P.tt.notna()
    P["any_cov"] = P.any_all.where(cov)
    P["tt_minus"] = P.tt - P.any_cov
    P["covered"] = cov
    print(f"sigma any spout {sg} mm, same target {sg_t} mm")
    return P, dict(sigma_any=sg, sigma_tt=sg_t, cv=pd.concat([cv.assign(ref="any spout"), cv_t.assign(ref="same target")]))


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cells = [("all spouts", "all", P), ("all spouts", "first", P[P.order == 0])] + [(p, "all", P[P.pos == p]) for p in POS]
    for scope, cls, g0 in cells:
        for (st, a, lab, ep), h in g0.groupby(["set", "animal", "label", "epoch"]):
            r = dict(scope=scope, cls=cls, set=st, animal=a, label=lab, epoch=ep, n=len(h), covered=float(h.covered.mean()))
            for v in VALS:
                r[v] = h[v].mean() if h[v].notna().sum() >= MIN else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (scope, cls, st), g in S.groupby(["scope", "cls", "set"]):
        g = g.assign(scope=f"{scope}|{cls}")
        for val in VALS:
            prd = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            prd = {a: v for a, v in prd.items() if len(v)}
            pm = g[g.epoch == "pre"].groupby("animal")[val].mean()
            for ep in EP:
                post = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                post = {a: v for a, v in post.items() if len(v)}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0)) if post else None
                if ci is None:
                    continue
                am = (g[g.epoch == ep].groupby("animal")[val].mean() - pm).dropna()
                mx = lo._lmm(g, val, ep) if g.animal.nunique() > 1 else {"mixed_coef": np.nan, "mixed_p": np.nan}
                out.append(dict(scope=scope, cls=cls, set=st, measure=val, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am), **mx,
                                **{f"animal_{a}": v for a, v in am.items()}))
    return pd.DataFrame(out)


def coverage(P: pd.DataFrame) -> pd.DataFrame:
    """Share of licks with a same-target same-place reference, per set x epoch x spout (and all spouts), per animal."""
    A = P.groupby(["set", "epoch", "pos", "animal"]).covered.mean().rename("covered").reset_index()
    B = P.groupby(["set", "epoch", "animal"]).covered.mean().rename("covered").reset_index().assign(pos="all spouts")
    C = pd.concat([A, B], ignore_index=True)
    M = C.groupby(["set", "epoch", "pos"]).covered.mean().rename("covered").reset_index().assign(animal="mean of animals")
    return pd.concat([C, M], ignore_index=True)


def compute(animals: list[str] | None) -> dict:
    P, info = per_lick(animals)
    S = sessions(P)
    T = tests(S)
    return dict(**info, S=S, T=T, coverage=coverage(P), animals=sorted(P.animal.unique()))


def draw(D: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import lick_matched_readout as lmr
    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    S, T, C = D["S"], D["T"], D["coverage"]
    fig, axs = plt.subplots(2, 5, figsize=(21, 8.6), squeeze=False)
    for i, st in enumerate(SETS):
        for j, val in enumerate(VALS):
            ax = axs[i, j]
            g = S[(S.set == st) & (S.scope == "all spouts") & (S.cls == "all")].rename(columns={val: "resid"})
            t = T[(T.set == st) & (T.scope == "all spouts") & (T.cls == "all") & (T.measure == val)].set_index("epoch")
            for k, ep in enumerate(EPS):
                lmr._dots(ax, g[g.epoch == ep].dropna(subset=["resid"]), k, col[ep])
                if ep in t.index:
                    lmr._stars(ax, k, t.loc[ep], col[ep])
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(4), EPS, fontsize=7.5)
            ax.set_title(f"{SETS[st]}: {VLAB[val]}", fontsize=8.5, pad=24)
            if j == 0:
                ax.set_ylabel("read-out vs pre licks at the same place\n(mm, + = contralesional)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        ax = axs[i, 4]
        c = C[(C.set == st) & (C.animal == "mean of animals")]
        for p in [*POS, "all spouts"]:
            q = c[c.pos == p].set_index("epoch").reindex(EPS).covered
            ax.plot(range(4), 100 * q, "o-" if p == "all spouts" else ".-", lw=2.2 if p == "all spouts" else 1,
                    color="k" if p == "all spouts" else None, label=p.replace("_", " "))
        ax.set_xticks(range(4), EPS, fontsize=7.5)
        ax.set_ylim(0, 102)
        ax.set_ylabel("% of licks with a same-target\nsame-place pre reference (mean of animals)", fontsize=8)
        ax.set_title(f"{SETS[st]}: COVERAGE by target spout", fontsize=8.5, pad=24)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("SAME-TARGET SAME-PLACE check: is the like-for-like offset 'more misses' (target composition) or 'each lick "
                 "read differently'? Frozen KEY-row lateral decoder (z-scored " + lo.DEC_LAB + ").\nReference = the "
                 "animal's PRE licks that landed at the same place (Gaussian on landing, pre leave-one-session-out): "
                 f"any spout (sigma {D['sigma_any']} mm) vs the SAME TARGET spout only (sigma {D['sigma_tt']} mm; cued = "
                 "cued spout, ITI = previous trial's spout).\nsmall filled = SESSIONS, open = animal means, bar = mean of "
                 "animals; change vs pre, 95% paired nested bootstrap, * = CI excludes 0 (bold), k/n animals. PROVISIONAL",
                 fontsize=9)
    lo._layout(fig, bottom=0.02)
    out = il.OUT_W / "figures" / f"{STEM}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, out, dpi=100)
    plt.close(fig)


def trial_set_sentence(animals=None) -> str:
    """THE lick set of the figure (Priya 11-09): KEY per-lick cache rows (run's ITI filter), finite landing and decoded
    value; the same-target rows score only licks with a same-target same-place pre reference (coverage)."""
    import lick_refit_decoders as rd
    extra = ["finite landing and decoded position",
             "same-target rows: only licks with a same-target same-place pre reference"]
    if animals:
        extra.append("animals " + ", ".join(animals) + " only")
    return rd.key_sets(filt=None, extra=extra)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--animals", nargs="*")
    a = ap.parse_args()
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence(None if a.plot_only else a.animals))
    if plot_only():
        draw(load_fd(STEM))
        return 0
    D = compute(a.animals)
    pd.set_option("display.width", 250)
    C = D["coverage"]
    print("== coverage (share of licks with a same-target same-place reference), all spouts:")
    print(C[C.pos == "all spouts"].pivot_table(index=["set", "animal"], columns="epoch", values="covered")
          .reindex(columns=EPS).round(3).to_string())
    T = D["T"]
    cols = ["scope", "cls", "set", "measure", "epoch", "point", "lo", "hi", "n_dir", "n", "mixed_p"] + \
        [c for c in T.columns if c.startswith("animal_")]
    print("== per animal (rule 8) + nested bootstrap (all spouts):")
    print(T[T.scope == "all spouts"][cols].round(3).to_string(index=False))
    tag = lo.SRC_TAG
    il.OUT_W.mkdir(parents=True, exist_ok=True)
    C.to_csv(il.OUT_W / f"sametarget_coverage{tag}.csv", index=False)
    D["cv"].to_csv(il.OUT_W / f"sametarget_sigma_cv{tag}.csv", index=False)
    D["S"].to_csv(il.OUT_W / f"sametarget_sessions{tag}.csv", index=False)
    T.to_csv(il.OUT_W / f"sametarget_tests{tag}.csv", index=False)
    save_fd(STEM, D)
    draw(D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
