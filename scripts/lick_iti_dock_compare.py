"""ITI offset HEADLINES on DOCKED-spout licks vs all ITI licks vs far1 (Priya 10-29: "ITI offset results restricted to
licks made while the spout is docked", i.e. not during its retraction or outbound move).

The ITI subsets (iti_licks.ITI_SUBSET_FILTERS; per-lick flags from iti_dock_audit.py) keep the FROZEN all-ITI decoders
(rule 10: the lick_order_preds cache and the offset_contrib terms are reused, nothing is refit) and drop ITI rows
downstream; each filter's outputs live in <OUT>/itifilter_<filter>/ under the primary file names:
  dock      onset in [dock, next trial_start]  (the strict docked window, wfield_local.docked_periods)
  dock01    the same with 0.1 s margins at both ends
  gap2dock  dock & >= 2 s since the previous lick onset (the all-ITI read-out (a) only; b-e already use gap >= 2)
  notdock   the complement: licks while the spout moves (retraction, outbound) -- interpretation check
  far1s     >= 1 s after the window end and before the next arrival, as a SUBSET (far1 of lick_iti_filter_compare
            REFIT the decoder; both are shown for (a))
Headlines (+ = contralesional, change vs pre, paired nested bootstrap analysis_kit.boot_delta, per-animal first):
  a  frozen-decoder same-place ITI offset, matched_loclin resid, all spouts, all ITI licks  (lick_matched_readout --ref loclin)
  b  matched-error ITI decoder offset off_spout and within-spout balance bal_spout (lick_balance_matched_error)
  c  primary ITI balance (ITI first after >= 2 s quiet, pool ALM+tjM1+SSp-m+SSp-n+SSs, ratio 0.1-0.5 s;
     lick_balance_subsets --sessions-only -> lick_recovery_primary): change vs pre + recovery tracking vs the ITI
     lick-angle axis (per-animal r, LOAO cvR2)
  d  ITI SSp-n left/right difference term of the decoder offset (offset_contrib terms_licks ssp_n_diff, session mean
     over the kept licks; reference = the frozen all-ITI same-place reference): change vs pre + tracking (iti_ang)
  e  ITI rows of lick_ci_spout_epoch (C/I balance by spout x epoch; d_bal_angle, all spouts + far contra)
Steps (each with WFIELD_ITI_FILTER=<filter> in the environment):
  --prep   copy the read-only inputs the downstream scripts read from OUT_W into the filter folder and write the
           filtered baltarget_licks.csv.gz there (the b / c / e lick table: ITI first after quiet)
  --sspn   headline d for the active filter (no filter = all ITI; writes itidock_sspn_*.csv to OUT_W)
  (none)   gather every filter + draw figures/lick_iti_dock_compare.png (figdata figdata/lick_iti_dock_compare.pkl;
           --plot-only redraws) and lick_iti_dock_compare_tables.csv
Usage: [WFIELD_ITI_FILTER=dock] python scripts/lick_iti_dock_compare.py [--prep | --sspn | --plot-only]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
STEM = "lick_iti_dock_compare"
TERMS = "offset_contrib/terms_licks_areas_splitr8.csv.gz"
COPY = ["covref_covariates.csv.gz", "recmark_sessions.csv", TERMS]
# sets shown: (key, label, folder or tag)
SETS = [("all", "all ITI licks (primary)"), ("far1", "far1, decoder REFIT (existing)"),
        ("far1s", "far1, frozen decoder subset"), ("dock", "DOCKED (dock -> trial_start)")]
EXTRA = [("dock01", "docked, 0.1 s margins"), ("gap2dock", "docked + 2 s quiet"),
         ("notdock", "COMPLEMENT: spout moving (retract / outbound)"),
         ("notdock_n", "complement, size-matched to docked")]
COL = {"all": "k", "far1": "#999999", "far1s": "#d95f02", "dock": "#1b9e77", "dock01": "#66a61e", "gap2dock": "#7570b3",
       "notdock": "#e7298a", "notdock_n": "#f4a6c8"}
QPANEL = "ITI first after quiet -> iti_ang"


def fdir(key: str) -> Path:
    return il.OUT if key in ("all", "far1") else il.OUT / f"itifilter_{key}"


# ------------------------------------------------------------------------------------------------------------- prep
def prep() -> int:
    if not il.ITI_SUBSET:
        raise SystemExit("--prep needs WFIELD_ITI_FILTER in " + ", ".join(il.ITI_SUBSET_FILTERS))
    dst = il.OUT_W
    (dst / "offset_contrib").mkdir(parents=True, exist_ok=True)
    (dst / "figures").mkdir(exist_ok=True)
    for f in COPY:
        if not (dst / f).exists():
            shutil.copy2(il.OUT / f, dst / f)
            print(f"  copied {f}")
    B = pd.read_csv(il.OUT / "baltarget_licks.csv.gz")
    keep = il.iti_subset_keep(B)
    print(f"baltarget_licks: ITI {int(keep[(B.set == 'iti').to_numpy()].sum())} of {int((B.set == 'iti').sum())} kept")
    B[keep].to_csv(dst / "baltarget_licks.csv.gz", index=False)
    return 0


# ------------------------------------------------------------------------------------------------------------- sspn
def _terms_with_k() -> pd.DataFrame:
    """terms_licks (frozen decoder, per lick) keyed to the prediction-cache index k (join on set, label, order, pos,
    act + duplicate counter, as lick_balance_matched_error._readouts; terms_licks drops a few rows)."""
    import iti_lick_order as lo
    P = pd.read_csv(lo.preds_path(), usecols=["set", "label", "order", "pos", "act"])
    P["k"] = P.groupby(["set", "label"]).cumcount()
    T = pd.read_csv(il.OUT / TERMS)
    T["set"] = T.set.replace({"cued": "trial"})
    key = ["set", "label", "order", "pos", "act"]
    for df in (T, P):
        df["act"] = df.act.round(3)
        df["dup"] = df.groupby(key).cumcount()
    T = T.merge(P, on=[*key, "dup"], how="left")
    if T.k.isna().any():
        raise RuntimeError(f"{int(T.k.isna().sum())} terms rows without a prediction row")
    T["k"] = T.k.astype(int)
    return T


def sspn() -> int:
    import lick_recovery_markers as lrm

    from wfield_local import analysis_kit as ak
    T = _terms_with_k()
    T = T[(T.set == "iti") & il.iti_subset_keep(T)]
    S = T.groupby(["set", "animal", "label", "epoch"]).agg(diff_sspn=("ssp_n_diff", "mean"), total=("total", "mean"),
                                                           n_licks=("total", "size")).reset_index()
    S = S[S.n_licks >= 8]
    R = pd.read_csv(il.OUT / "recmark_sessions.csv", usecols=["label", "day", "aim", "iti_ang"]).drop_duplicates("label")
    S = S.merge(R, on="label", how="left")
    S.to_csv(il.OUT_W / "itidock_sspn_sessions.csv", index=False)
    rows = []
    for c in ("diff_sspn", "total"):
        pre = {a: h[c].to_numpy() for a, h in sorted(S[S.epoch == "pre"].groupby("animal"))}
        for ep in EP:
            post = {a: h[c].to_numpy() for a, h in sorted(S[S.epoch == ep].groupby("animal")) if a in pre}
            ci = ak.boot_delta(post, pre, np.random.default_rng(0))
            am = {a: post[a].mean() - pre[a].mean() for a in post}
            rows.append({"measure": c, "epoch": ep, "point": ci.point, "lo": ci.lo, "hi": ci.hi,
                         "n_dir": sum(np.sign(v) == np.sign(ci.point) for v in am.values()), "n": len(am),
                         "n_licks": int(S[S.epoch == ep].n_licks.sum()), **am})
    TT = pd.DataFrame(rows)
    TT.to_csv(il.OUT_W / "itidock_sspn_tests.csv", index=False)
    lrm.N_SHUF = 1000
    out = []
    for yv in ("iti_ang", "aim"):
        F = lrm.models(S[["set", "label", "animal", "epoch", "day", "diff_sspn", "total"]].assign(aim=S[yv]))
        an = F["AN"].pivot_table(index="pred", columns="animal", values="r").reset_index()
        out.append(F["CV"].merge(an, on="pred").merge(F["M"][["pred", "slope", "p"]], on="pred", how="left").assign(y=yv))
    TR = pd.concat(out, ignore_index=True)
    TR.to_csv(il.OUT_W / "itidock_sspn_track.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"== SSp-n difference, filter {il.ITI_SUBSET or 'all'}: change vs pre (per animal, then bootstrap)\n"
          + TT.round(3).to_string(index=False))
    print("== tracking\n" + TR[["y", "pred", "n", "loao", "p_shuffle", "slope", "p", *AN]].round(3).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------------------------------------- compare
def _nlicks(A: pd.DataFrame, key: str, gap2: bool) -> pd.Series:
    # far1 (refit) uses the same rule as far1s: count its licks with the far1s flag
    sel = np.ones(len(A), bool) if key == "all" else A["far1s" if key == "far1" else key].to_numpy(bool)
    if gap2:
        sel &= (A.gap >= il.GAP_MAIN).to_numpy()
    return A[sel].groupby("epoch").size()


def _tests_row(q, key, ep, point, lo_, hi_, animals: dict, n_licks) -> dict:
    vals = [v for v in animals.values() if np.isfinite(v)]
    return {"quantity": q, "set": key, "epoch": ep, "point": point, "lo": lo_, "hi": hi_,
            "n_dir": int(sum(np.sign(v) == np.sign(point) for v in vals)), "n": len(vals), "n_licks": n_licks,
            **{a: animals.get(a, np.nan) for a in AN}}


def _track_rows(q, key, S: pd.DataFrame, xcol: str, ycol: str, loao: float, p_shuf: float) -> dict:
    post = S[(S.epoch != "pre") & (S.day > 0)].dropna(subset=[xcol, ycol])
    r = {a: float(np.corrcoef(h[xcol], h[ycol])[0, 1]) for a, h in post.groupby("animal") if len(h) > 2}
    return {"quantity": q, "set": key, "loao": loao, "p_shuffle": p_shuf, "mean_r": float(np.mean(list(r.values()))),
            **{a: r.get(a, np.nan) for a in AN}}


def compute() -> dict:
    from wfield_local import analysis_kit as ak
    A = pd.read_csv(il.OUT / "iti_dock_audit.csv")
    keys = [k for k, _ in SETS + EXTRA]
    rows, trk = [], []
    for key in keys:
        d = fdir(key)
        n_all, n_q = _nlicks(A, key, False), _nlicks(A, key, True)
        # (a) same-place offset
        fa = d / ("matched_loclin_far1_tests.csv" if key == "far1" else "matched_loclin_tests.csv")
        if fa.exists():
            T = pd.read_csv(fa)
            T = T[(T.scope == "all spouts") & (T.measure == "resid") & (T.set == "iti") & (T.cls == "all")]
            for _, r in T[T.epoch.isin(EP)].iterrows():
                rows.append(_tests_row("a: same-place offset (mm)", key, r.epoch, r.point, r.lo, r.hi,
                                       {a: r.get(f"animal_{a}", np.nan) for a in AN}, int(n_all.get(r.epoch, 0))))
        if key in ("far1", "gap2dock"):
            continue
        # (b) matched-error
        fb = d / "balmatch_tests.csv"
        if fb.exists():
            T = pd.read_csv(fb)
            for meas, q in (("off_spout", "b: matched-error offset (mm)"), ("bal_spout", "b: within-spout balance")):
                for _, r in T[(T.set == "iti") & (T.measure == meas)].iterrows():
                    rows.append(_tests_row(q, key, r.epoch, r.point, r.lo, r.hi, {a: r.get(f"d_{a}", np.nan) for a in AN},
                                           int(n_q.get(r.epoch, 0))))
        # (c) primary ITI balance: change vs pre (from recprim_sessions) + tracking
        fc = d / "recprim_sessions.csv"
        if fc.exists():
            S = pd.read_csv(fc)
            S = S[S.panel == QPANEL]
            pre = {a: h.marker.dropna().to_numpy() for a, h in sorted(S[S.epoch == "pre"].groupby("animal"))}
            for ep in EP:
                post = {a: h.marker.dropna().to_numpy() for a, h in sorted(S[S.epoch == ep].groupby("animal")) if a in pre}
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                rows.append(_tests_row("c: primary ITI balance (log2)", key, ep, ci.point, ci.lo, ci.hi,
                                       {a: post[a].mean() - pre[a].mean() for a in post}, int(n_q.get(ep, 0))))
            CV = pd.read_csv(d / "recprim_cv.csv")
            cv = CV[(CV.panel == QPANEL) & (CV.pred == "marker")].iloc[0]
            trk.append(_track_rows("c: primary ITI balance vs ITI angle axis", key, S, "marker", "y", cv.loao,
                                   cv.p_shuffle))
        # (d) SSp-n difference
        fd = d / "itidock_sspn_tests.csv"
        if fd.exists():
            T = pd.read_csv(fd)
            for _, r in T[T.measure == "diff_sspn"].iterrows():
                rows.append(_tests_row("d: SSp-n L/R difference term (mm)", key, r.epoch, r.point, r.lo, r.hi,
                                       {a: r.get(a, np.nan) for a in AN}, int(r.n_licks)))
            TR = pd.read_csv(d / "itidock_sspn_track.csv")
            S = pd.read_csv(d / "itidock_sspn_sessions.csv")
            t = TR[(TR.y == "iti_ang") & (TR.pred == "diff_sspn")].iloc[0]
            trk.append(_track_rows("d: SSp-n difference vs ITI angle axis", key, S, "diff_sspn", "iti_ang", t.loao,
                                   t.p_shuffle))
        # (e) C/I by spout x epoch, ITI rows
        fe = d / "ci_spout_epoch_tests.csv"
        if fe.exists():
            T = pd.read_csv(fe)
            for grp in ("all", "far_contra"):
                for _, r in T[(T.set == "iti") & (T.grp == grp) & (T["var"] == "d_bal_angle")].iterrows():
                    rows.append(_tests_row(f"e: C/I d_bal_angle, {grp}", key, r.epoch, r.est, r.lo, r.hi,
                                           {a: r.get(a, np.nan) for a in AN}, int(n_q.get(r.epoch, 0))))
    R, K = pd.DataFrame(rows), pd.DataFrame(trk)
    return {"R": R, "K": K}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    R, K = F["R"], F["K"]
    qs = list(dict.fromkeys(R.quantity))
    show = [k for k, _ in SETS] + ["notdock", "notdock_n"]
    lab = dict(SETS + EXTRA)
    nq = len(qs) + 1
    ncol = 3
    nrow = int(np.ceil(nq / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 4.8 * nrow), squeeze=False)
    for i, q in enumerate(qs):
        ax = axs.flat[i]
        G = R[(R.quantity == q) & R.set.isin(show)]
        ks = [k for k in show if k in set(G.set)]
        w = 0.8 / max(len(ks), 1)
        for j, k in enumerate(ks):
            g = G[G.set == k].set_index("epoch").reindex(EP)
            x = np.arange(len(EP)) + (j - (len(ks) - 1) / 2) * w
            ax.errorbar(x, g.point, yerr=[g.point - g.lo, g.hi - g.point], fmt="o", color=COL[k], ms=5, capsize=2.5,
                        zorder=3)
            for e, ep in enumerate(EP):
                if not np.isfinite(g.loc[ep, "point"]):
                    continue
                for a in AN:
                    ax.plot(x[e] + 0.04, g.loc[ep, a], ANIMAL_MARKER[a], mfc="none", mec=COL[k], ms=3.5, alpha=0.7)
                sig = "*" if g.loc[ep, "lo"] > 0 or g.loc[ep, "hi"] < 0 else ""
                ax.text(x[e], g.loc[ep, "hi"], f"{g.loc[ep, 'point']:+.2f}{sig}\n{int(g.loc[ep, 'n_dir'])}/"
                        f"{int(g.loc[ep, 'n'])}\nn={int(g.loc[ep, 'n_licks'])}", ha="center", va="bottom", fontsize=5.5,
                        color=COL[k])
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xticks(range(len(EP)), EP)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.25 * (hi_ - lo_))
        ax.set_title(f"{q}\nchange vs pre", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax = axs.flat[len(qs)]
    tq = list(dict.fromkeys(K.quantity))
    for i, q in enumerate(tq):
        G = K[(K.quantity == q) & K.set.isin(show)]
        ks = [k for k in show if k in set(G.set)]
        for j, k in enumerate(ks):
            g = G[G.set == k].iloc[0]
            x = i + (j - (len(ks) - 1) / 2) * 0.22
            for a in AN:
                ax.plot(x, g[a], ANIMAL_MARKER[a], mfc="none", mec=COL[k], ms=5)
            ax.plot([x - 0.08, x + 0.08], [g.mean_r] * 2, color=COL[k], lw=2)
            ax.text(x, 1.02, f"LOAO\n{g.loao:+.2f}", ha="center", va="bottom", fontsize=6, color=COL[k])
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(range(len(tq)), [q.split(":")[0] + ": " + q.split(":")[1].split(" vs")[0] for q in tq], fontsize=7)
    ax.set_ylim(-1.05, 1.35)
    ax.set_ylabel("within-animal r with the ITI lick-angle axis\n(post-stroke sessions; bar = mean)", fontsize=8)
    ax.set_title("recovery tracking (markers = animals; LOAO cvR2 above)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for ax in axs.flat[len(qs) + 1:]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[k], marker="o", lw=0, label=lab[k]) for k in show]
    h += [Line2D([], [], color="0.4", marker=ANIMAL_MARKER[a], mfc="none", lw=0, label=a) for a in AN]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle("NEURAL: ITI offset headlines on DOCKED-spout licks (onset between dock and the next trial_start) vs all "
                 "ITI licks vs far1. Frozen all-ITI decoders, rows subset (far1 refit = the earlier variant).\n"
                 "Point = cohort change vs pre, 95% paired nested bootstrap, * = CI excludes 0; open markers = animals; "
                 "k/n = animals with the cohort sign; n = ITI licks (a: all; b-e: first after >= 2 s quiet). "
                 "+ = contralesional. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(il.OUT_W / "figures" / f"{STEM}.png", dpi=100)
    plt.close(fig)


def main() -> int:
    if "--prep" in sys.argv:
        return prep()
    if "--sspn" in sys.argv:
        return sspn()
    if figdata.plot_only():
        F = figdata.load_fd(STEM)
    else:
        F = compute()
        figdata.save_fd(STEM, F)
        F["R"].to_csv(il.OUT_W / f"{STEM}_tables.csv", index=False)
        F["K"].to_csv(il.OUT_W / f"{STEM}_tracking.csv", index=False)
    pd.set_option("display.width", 250)
    print(F["R"].round(3).to_string(index=False))
    print(F["K"].round(3).to_string(index=False))
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
