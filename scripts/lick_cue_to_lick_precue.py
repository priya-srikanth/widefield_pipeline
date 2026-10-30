"""CHRONIC PRE-CUE 'DIP' DECOMPOSITION (Priya 10-29, task D). PROVISIONAL.

Priya: the chronic pre-cue dF/F looks DECREASED in the NON-restw-normalised cue-aligned regional traces, but not (in
orofacial areas) in the restw-referenced ones (Fig 14.21, lick_precue_restw_traces.py). Which component moved?

Per session x area (8 areas x 2 hemispheres; ENGAGED trials = the precue_restw set; per-trial raw traces of
lick_cue_to_lick_extract.py):
  raw      = the cue-aligned area trace with NO baseline (drift basis incl. the constant removed, so 0 = the SESSION MEAN
             of that area), mean over -1..0 s (also -2..-1, -0.5..0) before the cue = the non-restw pre-cue level
  restw    = the session's RESTW baseline in the same units (rest-frame median, position-weighted; Fig 14.21). Because the
             trace has zero session mean, restw = rest level MINUS session mean, i.e. -restw = how far the session's
             average sits ABOVE rest (driven by task / lick responses).
  rel      = raw - restw = the restw-referenced pre-cue level (Fig 14.21; checked equal to precue_restw p10 - restw)
  shape    = in-trial shape: mean(-2..-1.5) - mean(-1..-0.5) (what an in-trial baseline -1..-0.5 s, Fig 14.6 / 14.7,
             shows before the cue)
  F0       = raw 470 nm fluorescence of the area (balimg_f0.csv, mask 'full'; orofacial areas only), change in % of pre
Change vs pre per animal (rule 8 table first), then analysis_kit.boot_delta. Identity: d_raw = d_rel + d_restw.
Outputs (<OUT_W>/cue_to_lick/): c2l_D_sessions.csv, c2l_D_per_animal.csv, c2l_D_changes.csv; figures/
lick_cue_to_lick_D_raw_traces.png (the non-restw cue-aligned traces), lick_cue_to_lick_D_decomposition.png; figdata.
Usage: python scripts/lick_cue_to_lick_precue.py [--plot-only] [--rest-variant restnl]   (needs lick_cue_to_lick_extract.py's
cache)
RESTnl VARIANT (`--rest-variant restnl`, default restw unchanged): the per-trial cache sessions_restnl/ (CURRENT engaged
gate), 'restw' column = the LICK-FREE rest level (scripts/restnl.py), rel = raw - RESTnl; sessions without a RESTnl level
left out; every output name gets the suffix _restnl.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import figdata as _fd  # noqa: E402
import lick_cue_to_lick_extract as cx  # noqa: E402
import restnl  # noqa: E402
from figdata import plot_only  # noqa: E402

SFX = ""                        # "_restnl" under --rest-variant restnl (set in main)
VAR = "restw"


def save_fd(stem, obj, root=None):
    return _fd.save_fd(stem + SFX, obj, root)


def load_fd(stem, root=None):
    return _fd.load_fd(stem + SFX, root)


def _o(name: str) -> str:
    stem, dot, ext = name.partition(".")
    return f"{stem}{SFX}{dot}{ext}"


def _rl(txt: str) -> str:
    return restnl.relabel(txt) + " Trials = CURRENT engaged gate." if SFX else txt

OUT = cx.OUT
FIGS = OUT / "figures"
FDR = OUT / "figdata"
EP = il.EP4
POST = EP[1:]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
LC = cx.LAGS_C
WINS = {"raw_m2_m1": (-2.0, -1.0), "raw": (-1.0, 0.0), "raw_m05_0": (-0.5, 0.0)}
COMP = ["raw", "restw", "rel", "shape", "F0pct"]
CLAB = {"raw": "raw pre-cue -1..0 s\n(no baseline)", "restw": "RESTW level", "rel": "pre-cue minus RESTW\n(Fig 14.21)",
        "shape": "in-trial shape\n(-2..-1.5) - (-1..-0.5)", "F0pct": "F0 (470 nm)\n% change"}
TL = np.round(np.arange(-2.0, 3.0 + 1e-9, 0.025), 4)


def _w(a, b):
    return (LC >= a - 1e-9) & (LC <= b + 1e-9)


def sessions() -> tuple[pd.DataFrame, dict]:
    rows, tr = [], {}
    tsel = _w(TL[0], TL[-1])
    for x in il.labels():
        z = np.load(cx.sess_dir(VAR) / f"{x}.npz")
        if not bool(z["restw_ok"]):
            continue
        an, ep = str(z["animal"]), str(z["epoch"])
        Xc = z["Xc"].astype(np.float64)
        pos = z["pos"].astype(str)
        rw = z["restw"].astype(np.float64)
        for sp in ["all"] + POS:
            m = np.ones(len(pos), bool) if sp == "all" else pos == sp
            if m.sum() < (10 if sp == "all" else 5):
                continue
            X = Xc[m]
            v = {k: np.nanmean(np.nanmean(X[:, _w(*w)], 1), 0) for k, w in WINS.items()}
            shape = np.nanmean(np.nanmean(X[:, _w(-2.0, -1.5)], 1), 0) - np.nanmean(np.nanmean(X[:, _w(-1.0, -0.5)], 1), 0)
            for j, c in enumerate(cx.COLS):
                rows.append({"label": x, "animal": an, "epoch": ep, "pos": sp, "col": c, "n": int(m.sum()),
                             **{k: float(val[j]) for k, val in v.items()}, "restw": float(rw[j]),
                             "rel": float(v["raw"][j] - rw[j]), "shape": float(shape[j])})
            if sp == "all":
                tr[x] = (an, ep, np.nanmean(X[:, tsel], 0), rw)
    S = pd.DataFrame(rows)
    # F0 (orofacial areas only)
    f0 = pd.read_csv(il.OUT_W / "balimg_f0.csv")
    f0 = f0[f0["mask"] == "full"]
    F = pd.concat([f0[["label", "pair", "f0_470_right"]].rename(columns={"f0_470_right": "F0"}).assign(h="right"),
                   f0[["label", "pair", "f0_470_left"]].rename(columns={"f0_470_left": "F0"}).assign(h="left")])
    F["col"] = F.pair + "_" + F.h
    S = S.merge(F[["label", "col", "F0"]], on=["label", "col"], how="left")
    pre0 = S[S.epoch == "pre"].groupby(["animal", "col"]).F0.mean().rename("F0_pre")
    S = S.merge(pre0, left_on=["animal", "col"], right_index=True, how="left")
    S["F0pct"] = 100 * (S.F0 / S.F0_pre - 1)
    return S, tr


def changes(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(0)
    out = []
    for (sp, col), g in S.groupby(["pos", "col"], sort=True):
        for k in COMP + ["raw_m2_m1", "raw_m05_0"]:
            if g[k].isna().all():
                continue
            pre = {a: list(x[k].dropna()) for a, x in g[g.epoch == "pre"].groupby("animal", sort=True)}
            for E in POST:
                post = {a: list(x[k].dropna()) for a, x in g[g.epoch == E].groupby("animal", sort=True)}
                pre_ = {a: v for a, v in pre.items() if v}
                post = {a: v for a, v in post.items() if v}
                iv = ak.boot_delta(post, pre_, rng)
                if iv is None:
                    continue
                per = {a: float(np.mean(post[a]) - np.mean(pre_[a])) for a in sorted(set(pre_) & set(post))}
                same = sum(np.sign(v) == np.sign(iv.point) for v in per.values())
                out.append({"pos": sp, "col": col, "comp": k, "epoch": E, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                            "n_animals": iv.n_animals, "n_same_sign": same, "sig": iv.excludes_zero,
                            "flag_lt3": same < 3, **{f"d_{a}": v for a, v in per.items()}})
    return pd.DataFrame(out)


def _wrap(txt: str, fig_w: float, fs: float) -> str:
    """Pre-wrap a suptitle to the figure width so constrained layout reserves exactly its height."""
    import textwrap
    return textwrap.fill(txt, max(40, int(fig_w * 72 / (0.55 * fs))))


def draw_traces(stem: str) -> None:
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(stem, FDR)
    areas = cx.AREAS
    fig, axs = plt.subplots(len(areas), 2, figsize=(11, 2.2 * len(areas)), sharex=True, squeeze=False,
                            layout="constrained")
    for i, a in enumerate(areas):
        for j, (h, hl) in enumerate(cx.HEMI):
            ax = axs[i, j]
            c = f"{a}_{h}"
            if c not in F["cur"]:
                ax.set_visible(False)
                continue
            ax.axvline(0, color="0.3", lw=0.8)
            ax.axhline(0, color="0.6", lw=0.6)
            for ep, (mu, se, _n, rwm) in F["cur"][c].items():
                col = EPOCH_LINE[ep]
                ax.fill_between(TL, mu - se, mu + se, color=col, alpha=0.15, lw=0)
                ax.plot(TL, mu, color=col, lw=1.3, label=ep)
                ax.axhline(rwm, color=col, lw=0.9, ls="--")
            ax.axvspan(-1.0, 0.0, color="0.9", zorder=0)
            ax.set_title(f"{a} {hl}" + ("  [reference]" if a in cx.REF else ""), fontsize=9)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("dF/F x1e3\n(0 = session mean)", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("time from cue (s)")
    for j in range(axs.shape[1]):   # a hidden bottom panel leaves its column without x tick labels: label the last shown
        vis = [ax for ax in axs[:, j] if ax.get_visible()]
        if vis:
            vis[-1].tick_params(labelbottom=True)
            vis[-1].set_xlabel("time from cue (s)")
    h, lab = axs[0, 0].get_legend_handles_labels()   # one figure legend below the panels (was inside ALM contra)
    fig.legend(h, lab, loc="outside lower center", fontsize=8, frameon=False, ncol=4)
    fig.suptitle(_wrap(_rl("NEURAL, NO BASELINE (non-restw version of Fig 14.21): cue-aligned area traces of ENGAGED trials, all "
                 "spouts; the drift basis incl. the constant is removed, so 0 = the area's SESSION MEAN. Dashed = the "
                 "epoch's mean RESTW level in the same units (rest sits below the session mean). Per animal x epoch mean "
                 "over sessions, mean of animals, shade = SEM over animals. Grey = -1..0 s pre-cue window. "
                 f"n: {F['nlab']}. PROVISIONAL."), 11, 9), fontsize=9)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_decomp(stem: str) -> None:
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(stem, FDR)
    PA, D, acol = F["PA"], F["D"], F["acol"]
    cols = [c for c in F["cols"]]
    fig, axs = plt.subplots(len(cols), len(COMP), figsize=(2.2 * len(COMP), 1.45 * len(cols)), squeeze=False,
                            layout="constrained")
    for i, c in enumerate(cols):
        rowv = []
        for k in COMP[:4]:
            if (c, k) in PA.index:
                rowv.append(PA.loc[(c, k)].to_numpy(float))
        lim = np.nanmax(np.abs(np.concatenate([r.ravel() for r in rowv]))) * 1.15 if rowv else 1
        for j, k in enumerate(COMP):
            ax = axs[i, j]
            ax.axhline(0, color="0.6", lw=0.6)
            if (c, k) not in PA.index:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                t = PA.loc[(c, k)]
                for an, r in t.iterrows():
                    ax.plot(range(len(POST)), r.reindex(POST).to_numpy(float), "o", color=acol.get(an, "0.5"), ms=3.5)
                mu = t.reindex(columns=POST).mean(0)
                ax.scatter(range(len(POST)), mu, marker="_", s=180, c=[EPOCH_LINE[e] for e in POST], lw=2.2, zorder=4)
                for _q, r in D[(D.col == c) & (D.comp == k)].iterrows():
                    if r.sig:
                        ax.text(POST.index(r.epoch), 1.0, "*" if not r.flag_lt3 else "(*)", ha="center", va="bottom", fontsize=9,
                                transform=ax.get_xaxis_transform(), color=EPOCH_LINE[r.epoch], clip_on=False)
                if k != "F0pct":
                    ax.set_ylim(-lim, lim)
                ax.set_xticks(range(len(POST)), [e[:3] for e in POST], fontsize=6)
                ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(CLAB[k], fontsize=7, pad=12)
            if j == 0:
                ax.set_ylabel(c.replace("_right", "\ncontra").replace("_left", "\nipsi"), fontsize=7)
    hs = [plt.Line2D([], [], color=v, marker="o", ls="", ms=4, label=a) for a, v in sorted(acol.items())]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle(_wrap(_rl("NEURAL, change from PRE per animal (dots) of the pre-cue components, ENGAGED trials, all spouts "
                 "(dF/F x1e3; first four columns share one scale per row): raw pre-cue level with NO baseline (0 = "
                 "session mean) = (pre-cue minus RESTW) + RESTW; in-trial shape = what a -1..-0.5 s in-trial baseline "
                 "shows; F0 = raw 470 nm fluorescence (% of the animal's pre). Bar = mean of animals; * = paired nested "
                 "bootstrap CI excludes 0 (analysis_kit.boot_delta), (*) = fewer than 3 animals share the sign. "
                 "PROVISIONAL, n = 4."), 2.2 * len(COMP), 8), fontsize=8)
    fig.savefig(FIGS / f"{stem}{SFX}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


SIMPLE_ROWS = ["tjM1_right", "SSp-m_right", "SSp-n_right", "ALM_right", "SSp-bfd_right", "VISp_right"]
SIMPLE_COMP = ["raw", "restw", "rel"]


def draw_decomp_simple(src: str = "lick_cue_to_lick_D_decomposition", out: str = "lick_cue_to_lick_D_simple") -> None:
    """Page version of the decomposition (Priya 10-29, 'too many panels'): contralesional rows, three components
    (raw = rel + restw); saved per-animal changes and the saved boot_delta point / CI / stars; F0 quoted in the title."""
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(src, FDR)
    PA, D, acol = F["PA"], F["D"], F["acol"]
    rows = [c for c in SIMPLE_ROWS if c in F["cols"]]
    fig, axs = plt.subplots(len(rows), len(SIMPLE_COMP), figsize=(8.4, 1.75 * len(rows) + 0.6), squeeze=False,
                            sharey="row", layout="constrained")
    for i, c in enumerate(rows):
        for j, k in enumerate(SIMPLE_COMP):
            ax = axs[i, j]
            ax.axhline(0, color="0.6", lw=0.6)
            if (c, k) in PA.index:
                t = PA.loc[(c, k)].reindex(columns=POST)
                for a_i, (an, r) in enumerate(t.iterrows()):
                    ax.plot(np.arange(len(POST)) - 0.12 + 0.03 * (a_i - 1.5), r.to_numpy(float), "o", ms=3.2,
                            color=acol.get(an, "0.5"), alpha=0.9)
            dd = D[(D.col == c) & (D.comp == k)].set_index("epoch").reindex(POST)
            for x, (E, r) in enumerate(dd.iterrows()):
                if not np.isfinite(r.delta):
                    continue
                ax.errorbar(x + 0.12, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="s", ms=5.5, capsize=3,
                            color=EPOCH_LINE[E], lw=1.4, zorder=4)
                if r.sig:
                    ax.text(x + 0.12, 1.0, "*" if not r.flag_lt3 else "(*)", ha="center", va="bottom", fontsize=10,
                            color=EPOCH_LINE[E], transform=ax.get_xaxis_transform(), clip_on=False)
            ax.set_xticks(range(len(POST)), POST, fontsize=7)
            ax.set_xlim(-0.5, len(POST) - 0.5)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(_rl(CLAB[k]).replace(" Trials = CURRENT engaged gate.", ""), fontsize=8.5, pad=14)
            if j == 0:
                nm = c.replace("_right", " contra") + ("\n[reference]" if c.split("_")[0] in cx.REF else "")
                ax.set_ylabel(f"{nm}\nchange vs pre", fontsize=8)
    hs = [plt.Line2D([], [], color=v, marker="o", ls="", ms=4, label=a) for a, v in sorted(acol.items())]
    hs += [plt.Line2D([], [], color=EPOCH_LINE[e], marker="s", ls="-", ms=5, label=f"{e}: mean + 95% CI") for e in POST]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=7.5, frameon=False)
    f0 = D[(D.comp == "F0pct") & (D.epoch == "chronic") & D.col.isin(SIMPLE_ROWS)].set_index("col")
    f0s = ", ".join(f"{c.split('_')[0]} {f0.loc[c, 'delta']:+.1f}%{' *' if f0.loc[c, 'sig'] else ''}"
                    for c in SIMPLE_ROWS if c in f0.index)
    nsig = int(f0.sig.sum()) if len(f0) else 0
    head = F.get("head") or ("In the contralesional orofacial areas the pre-cue 'dip' seen without a baseline is the RESTW "
                             "level falling; pre-cue minus RESTW is unchanged or slightly UP chronically. The reference "
                             "areas SSp-bfd / VISp instead fall relative to RESTW. ")
    fig.suptitle(_wrap(head + _rl(
        "Change from PRE of the raw pre-cue level (-1..0 s, NO baseline, 0 = session mean) = "
        "(pre-cue minus RESTW, the Fig 14.21 quantity) + RESTW level, contralesional areas, ENGAGED trials, all spouts "
        "(dF/F x1e3; one scale per row). Raw F0 (470 nm) chronic change: " + f0s
        + (" (all n.s.)" if nsig == 0 else "") + ". NEURAL; dots = per-animal change (colour = animal), square + bar = "
        "paired nested bootstrap point and 95% CI (analysis_kit.boot_delta); * = CI excludes 0, (*) = fewer than 3 "
        "animals share the sign. PROVISIONAL, n = 4."), 8.4, 8.5), fontsize=8.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{out}{SFX}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global SFX, VAR
    VAR = restnl.variant()
    SFX = restnl.SFX if VAR == "restnl" else ""
    if plot_only():
        draw_traces("lick_cue_to_lick_D_raw_traces")
        draw_decomp("lick_cue_to_lick_D_decomposition")
        draw_decomp_simple()
        return 0
    from lick_cue_to_lick import _acol
    S, tr = sessions()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / _o("c2l_D_sessions.csv"), index=False)
    # sanity: rel vs the precue_restw session table (all spouts)
    C = pd.read_csv(il.OUT_W / f"precue_restw_sessions{SFX}.csv")
    J = C.merge(S[S.pos == "all"][["label", "col", "rel"]], left_on=["label", "area"], right_on=["label", "col"])
    print(f"== sanity: rel vs precue_restw_sessions.csv precue: max |diff| {(J.rel - J.precue).abs().max():.2e} "
          f"over {len(J)} rows")
    # per animal (rule 8): epoch means then change vs pre
    A = S.groupby(["pos", "col", "animal", "epoch"])[COMP + ["raw_m2_m1", "raw_m05_0", "F0"]].mean()
    A.to_csv(OUT / _o("c2l_D_per_animal.csv"))
    a = A.xs("all", level="pos")
    pre = a.xs("pre", level="epoch")
    rows = []
    for ep in POST:
        d = a.xs(ep, level="epoch") - pre.reindex(a.xs(ep, level="epoch").index)
        d["epoch"] = ep
        rows.append(d.reset_index())
    DA = pd.concat(rows, ignore_index=True)
    DA.to_csv(OUT / _o("c2l_D_per_animal_change_all.csv"), index=False)
    key = [f"{x}_{h}" for x in cx.AREAS for h in ("right", "left") if f"{x}_{h}" in cx.COLS]
    pd.set_option("display.width", 250)
    print("== per animal (rule 8): CHRONIC minus pre, all spouts (dF/F x1e3; F0pct in %)")
    s = DA[DA.epoch == "chronic"].set_index(["col", "animal"]).loc[key][["raw", "restw", "rel", "shape", "raw_m2_m1",
                                                                         "F0pct"]]
    print(s.round(2).to_string())
    D = changes(S)
    D.to_csv(OUT / _o("c2l_D_changes.csv"), index=False)
    print("== boot_delta, all spouts\n" + D[(D.pos == "all") & D.comp.isin(COMP)].pivot_table(
        index="col", columns=["epoch", "comp"], values="delta").reindex(columns=POST, level=0).reindex(key).round(2).to_string())
    print("== significant (CI excl 0), all spouts\n" + D[(D.pos == "all") & D.sig & D.comp.isin(COMP)][
        ["col", "comp", "epoch", "delta", "lo", "hi", "n_same_sign"]].round(2).to_string(index=False))
    # figures
    cur, nlab = {}, {}
    meta = pd.DataFrame([(x, v[0], v[1]) for x, v in tr.items()], columns=["label", "animal", "epoch"])
    for j, c in enumerate(cx.COLS):
        cur[c] = {}
        for ep in EP:
            m = meta[meta.epoch == ep]
            per = []
            rws = []
            for _an, g in m.groupby("animal"):
                per.append(np.nanmean([tr[x][2][:, j] for x in g.label], 0))
                rws.append(np.mean([tr[x][3][j] for x in g.label]))
            if not per:
                continue
            P = np.array(per)
            cur[c][ep] = (P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P)) if len(P) > 1 else np.zeros(P.shape[1]), len(P),
                          float(np.mean(rws)))
    for ep in EP:
        m = meta[meta.epoch == ep]
        nlab[ep] = f"{ep} {m.animal.nunique()}/{len(m)}"
    save_fd("lick_cue_to_lick_D_raw_traces", {"cur": cur, "nlab": "animals/sessions " + ", ".join(nlab.values())}, FDR)
    draw_traces("lick_cue_to_lick_D_raw_traces")
    PA = DA.set_index(["col", "animal", "epoch"])[COMP].stack().rename("v").reset_index().rename(columns={"level_3": "comp"})
    PA = PA.pivot_table(index=["col", "comp", "animal"], columns="epoch", values="v")
    save_fd("lick_cue_to_lick_D_decomposition", {"PA": PA, "D": D[D.pos == "all"], "acol": _acol(), "cols": key}, FDR)
    draw_decomp("lick_cue_to_lick_D_decomposition")
    draw_decomp_simple()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
