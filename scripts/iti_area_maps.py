"""Whole-cortex view of the self-initiated-lick response (Priya 10-26: "should we also do some non-orofacial per-area
activity maps to look at the whole cortex question? (supplemental figure with VIS areas)").

From the saved ITI sessions (`scripts/iti_licks.py`, every Allen area per bout-start ITI lick, >= 2 s without licking):
per session x area, response AUC (0..0.5 s after lick onset = first frame the tongue is visible) and build-up AUC
(-0.4..-0.05 s), baseline -1.5..-1.0 s.
  lick_iti_area_maps.png   change from pre per area in RAW dF/F AUC (acute / subacute / chronic) on the Allen atlas, one colour
                      scale per row; areas whose nested-bootstrap CI (animals > sessions) excludes 0 are outlined (black).
                      (10-26: per-area normalisation by the area's own pre response was dropped -- small caudal areas
                      with near-zero pre responses gave huge ratios of either sign and dominated the maps.)
  lick_iti_area_peth.png   per-session mean PETH +- SEM (all animals pooled) for non-orofacial areas, both hemispheres.
  lick_iti_area_groups.png orofacial (ALM, tjM1, MOs, MOp, SSp-m, SSp-n) vs visual + retrosplenial areas: one dot per
                      session; group-mean AUCs divided by the animal's pre GROUP-mean response AUC.
  lick_iti_area_compare.png orofacial ROIs next to control areas (`compare`).
Image left = ipsilesional (_left); _right = contralesional.
FIGURE DATA (Priya 10-27): each figure's drawn numbers are saved to <OUT>/figdata/<figure stem>.pkl (the area image +
names + outline for the maps); `--plot-only` redraws from those files alone (no session load, no bootstrap).
Usage: python scripts/iti_area_maps.py [--compare] [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402
from fr_musall import AN  # noqa: E402

PETH_AREAS = ["VISp", "VISl", "VISam", "RSPd", "SSp-bfd", "SSp-ul", "SSp-ll", "SSp-tr", "ACAd", "PL"]
ORO = ("ALM", "tjM1", "MOs", "MOp", "SSp-m", "SSp-n")
VISR = ("VIS", "RSP")
GROUP_ROWS = (("resp_n", "response AUC / pre"), ("build_n", "build-up AUC / pre response"))
GROUP_COLS = ("orofacial", "visual + RSP", "orofacial - visual + RSP")


def session_table(D: dict) -> tuple[pd.DataFrame, dict]:
    rows, peths = [], {}
    for x, d in D.items():
        m = d["bout_gap"] >= il.GAP_MAIN
        if m.sum() < 10:
            continue
        P = d["bout_peth"][m].astype(float)
        P = P - P[:, (il.LAGS_P >= il.BASE[0]) & (il.LAGS_P <= il.BASE[1]), :].mean(1, keepdims=True)
        mu = P.mean(0)                                    # (lags, areas)
        for j, c in enumerate(d["cols"]):
            if c == "cortex":
                continue
            rows.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "area": str(c),
                         "resp": il._auc(mu[:, j], 0.0, 0.5), "build": il._auc(mu[:, j], -0.4, -0.05)})
            peths[(x, str(c))] = mu[:, j]
    T = pd.DataFrame(rows)
    den = T[T.epoch == "pre"].groupby(["area", "animal"]).resp.mean()
    d_ = np.array([den.get((a_, n_), np.nan) for a_, n_ in zip(T.area, T.animal)])
    d_ = np.where(np.abs(d_) > 0, d_, np.nan)
    T["resp_n"], T["build_n"] = T.resp / d_, T.build / d_
    return T, peths


def changes(T: pd.DataFrame, val: str) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    out = []
    for area, t in T.groupby("area"):
        ref = {a: t[(t.animal == a) & (t.epoch == "pre")][val].dropna().to_numpy() for a in AN}
        for ep in il.EP4[1:]:
            post = {a: t[(t.animal == a) & (t.epoch == ep)][val].dropna().to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            if len(com) < 3:
                continue
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            out.append({"area": area, "value": val, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                        "sig": bool(iv.lo > 0 or iv.hi < 0), "n_animals": len(com)})
    return pd.DataFrame(out)


def main() -> int:
    """Compute every statistic, save the four figdata files, then draw (`draw_all` = the --plot-only path)."""
    from frozen_area_maps import area_image, outline
    labels = il.labels()
    D = {x: il._load(x) for x in labels}
    T, peths = session_table(D)
    T.to_csv(il.OUT / "area_auc_sessions.csv", index=False)
    C = pd.concat([changes(T, "resp"), changes(T, "build")])
    C.to_csv(il.OUT / "area_auc_changes.csv", index=False)
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    img, nm = area_image(labels[0])
    # ---- maps: per row the colour limit, per panel the area -> change dict + the significant areas
    rows = []
    for val, ttl in (("resp", "response AUC 0..0.5 s"), ("build", "build-up AUC -0.4..-0.05 s")):
        c = C[C.value == val]
        lim = np.nanpercentile(np.abs(c.delta), 98) or 1.0
        pans = []
        for ep in il.EP4[1:]:
            ce_ = c[c.epoch == ep].set_index("area")
            pans.append((ep, ce_.delta.to_dict(), list(ce_.index[ce_.sig]), int(ce_.sig.sum())))
        rows.append((ttl, lim, pans))
    save_fd("lick_iti_area_maps", {"img": img, "nm": nm, "ol": outline(img), "rows": rows})
    # ---- PETHs, non-orofacial areas
    pp = {}
    for j, base in enumerate(PETH_AREAS):
        for i, h in enumerate(("right", "left")):
            area = f"{base}_{h}"
            cur = []
            for ep in il.EP4:
                labs = [x for x in T[(T.area == area) & (T.epoch == ep)].label.unique() if (x, area) in peths]
                if not labs:
                    continue
                M = np.stack([peths[(x, area)] for x in labs])
                cur.append((ep, M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))))
            pp[(i, j)] = cur
    save_fd("lick_iti_area_peth", {"lags": il.LAGS_P, "panels": pp, "areas": list(PETH_AREAS)})
    # ---- group summary: orofacial vs visual + RSP
    T["group"] = np.where(T.area.str.rsplit("_", n=1).str[0].isin(ORO), "orofacial",
                          np.where(T.area.str.startswith(VISR), "visual + RSP", "other"))
    G = T[T.group != "other"].groupby(["label", "animal", "epoch", "group"])[["resp", "build"]].mean().reset_index()
    den = G[G.epoch == "pre"].groupby(["group", "animal"]).resp.mean()
    dg = np.array([den.get((g_, a_), np.nan) for g_, a_ in zip(G.group, G.animal)])
    G["resp_n"], G["build_n"] = G.resp / dg, G.build / dg
    rng = np.random.default_rng(4)
    tests, gp = [], {}
    for i, (val, _ttl) in enumerate(GROUP_ROWS):
        for j, g in enumerate(GROUP_COLS):
            if j < 2:
                sub = G[G.group == g]
            else:                                         # is the change orofacial-specific? session-paired difference
                Wd = G.pivot_table(index=["label", "animal", "epoch"], columns="group", values=val).dropna()
                sub = (Wd["orofacial"] - Wd["visual + RSP"]).rename(val).reset_index()
            S = il.dots_compute(sub, val, rng)
            gp[(i, j)] = S
            for r_ in S["tests"]:
                tests.append({"group": g, **r_})
    pd.DataFrame(tests).to_csv(il.OUT / "area_group_tests.csv", index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    save_fd("lick_iti_area_groups", {"panels": gp})
    compare(T, peths, draw=False)
    draw_all()
    return 0


def draw_all(only: tuple = ("maps", "peth", "groups", "compare")) -> None:
    """Draw the figures from figdata/<stem>.pkl alone (no session load, no bootstrap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import paint

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    if "maps" in only:
        F = load_fd("lick_iti_area_maps")
        img, nm, ol = F["img"], F["nm"], F["ol"]
        fig, axs = plt.subplots(2, 3, figsize=(14, 9))
        for i, (ttl, lim, pans) in enumerate(F["rows"]):
            for j, (ep, delta, sig_areas, n_sig) in enumerate(pans):
                ax = axs[i, j]
                M = paint(img, nm, delta)
                im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                sig = paint(img, nm, {a: 1.0 for a in sig_areas})
                sig_edge = np.zeros(img.shape, bool)
                ok = np.isfinite(sig)
                sig_edge[:-1] |= ok[:-1] != ok[1:]
                sig_edge[:, :-1] |= ok[:, :-1] != ok[:, 1:]
                ax.imshow(np.where(sig_edge, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"{ttl}\n{ep} - pre ({n_sig} areas significant)", fontsize=9)
                ax.axis("off")
            fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label("change in dF/F AUC (x1e3 s)")
        fig.suptitle("NEURAL, NO DECODER: self-initiated (ITI) licks across the whole cortex -- change from pre per Allen area in raw dF/F "
                     "AUC (x1e3 s); black outline = nested-bootstrap CI excludes 0. Image left = "
                     "ipsilesional.", fontsize=10, wrap=True)
        fig.savefig(il.OUT / "figures" / "lick_iti_area_maps.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    if "peth" in only:
        F = load_fd("lick_iti_area_peth")
        lags, areas = F["lags"], F["areas"]
        fig, axs = plt.subplots(2, len(areas), figsize=(2.3 * len(areas), 5.5), sharex=True)
        for j, base in enumerate(areas):
            for i, h in enumerate(("right", "left")):
                ax = axs[i, j]
                for ep, mu, se in F["panels"][(i, j)]:
                    ax.plot(lags, mu, color=colors[ep], lw=1.6, label=ep)
                    ax.fill_between(lags, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
                ax.axvline(0, color="0.6", lw=0.6)
                ax.axhline(0, color="0.8", lw=0.6)
                ax.set_xlim(-1.5, 0.75)
                ax.set_title(f"{base} {'contra' if h == 'right' else 'ipsi'}", fontsize=8)
                ax.tick_params(labelsize=7)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        axs[0, 0].legend(fontsize=6, frameon=False)
        axs[0, 0].set_ylabel("dF/F x1e3", fontsize=8)
        axs[1, 0].set_ylabel("dF/F x1e3", fontsize=8)
        fig.suptitle("NEURAL, NO DECODER: dF/F around self-initiated licks in NON-orofacial areas (per-session mean PETH, all animals "
                     "pooled; band = SEM over sessions; lick onset at 0).", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(il.OUT / "figures" / "lick_iti_area_peth.png", dpi=110)
        plt.close(fig)
    if "groups" in only:
        F = load_fd("lick_iti_area_groups")
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for i, (_val, ttl) in enumerate(GROUP_ROWS):
            for j, g in enumerate(GROUP_COLS):
                ax = axs[i, j]
                il.dots_draw(ax, F["panels"][(i, j)], colors)
                ax.set_title(f"{g}: {ttl}", fontsize=9)
        fig.suptitle("NEURAL, NO DECODER: self-initiated licks (dF/F), orofacial vs visual + retrosplenial areas (area means per session, divided by "
                     "the animal's pre group-mean response AUC; one dot per session; * = nested-bootstrap change from pre; right = "
                     "orofacial minus visual + RSP in the same session).", fontsize=9.5, wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(il.OUT / "figures" / "lick_iti_area_groups.png", dpi=110)
        plt.close(fig)
    if "compare" in only:
        compare_draw(load_fd("lick_iti_area_compare"))


#: orofacial ROIs next to control areas (Priya 10-26: "control area figures for activity traces, e.g. VIS to compare to
#: ALM"); contralesional (_right) hemisphere
COMPARE = ["ALM_right", "tjM1_right", "SSp-m_right", "VISp_right", "VISam_right", "RSPd_right", "SSp-bfd_right", "ACAd_right"]


def compare(T: pd.DataFrame, peths: dict, draw: bool = True) -> None:
    """Orofacial vs control areas side by side: per-session mean PETH (all animals pooled, band = SEM over sessions);
    top raw with ONE shared y-axis, bottom each session divided by its animal's pre-stroke peak (0..0.5 s) of that area.
    Saves figdata/lick_iti_area_compare.pkl (curves + per-bin tests), then draws it unless draw=False."""
    post = (il.LAGS_P >= 0.0) & (il.LAGS_P <= 0.5)
    pk = {}
    for (x, ar), v in peths.items():
        a_ = T[T.label == x].animal.iloc[0]
        if T[T.label == x].epoch.iloc[0] == "pre":
            pk.setdefault((a_, ar), []).append(v)
    pk = {k: float(np.mean(v, 0)[post].max()) for k, v in pk.items()}
    shown = (il.LAGS_P >= -1.5) & (il.LAGS_P <= 0.75)
    sig_recs, panels = [], {}
    for j, ar in enumerate(COMPARE):
        long = {False: [], True: []}
        curves = {0: [], 1: []}
        for ep in il.EP4:
            labs = sorted(x for x in T[(T.area == ar) & (T.epoch == ep)].label.unique() if (x, ar) in peths)
            if not labs:
                continue
            for i, norm in enumerate((False, True)):
                M = np.stack([peths[(x, ar)] / (pk.get((T[T.label == x].animal.iloc[0], ar), np.nan) if norm else 1.0)
                              for x in labs])
                mu, se = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(len(M))
                curves[i].append((ep, mu, se, len(M)))
                long[norm] += [pd.DataFrame({"label": x, "animal": T[T.label == x].animal.iloc[0], "epoch": ep,
                                             "lag": il.LAGS_P[shown], "v": m_[shown]}) for x, m_ in zip(labs, M)]
        for i, norm in enumerate((False, True)):
            S = None
            if long[norm]:
                S = il.sig_compute(pd.concat(long[norm], ignore_index=True), value="v", tag={"area": ar, "normalised": norm})
                sig_recs += S["recs"]
            panels[(i, j)] = {"curves": curves[i], "sig": S}
    pd.DataFrame(sig_recs).to_csv(il.OUT / "sig_bins_3_31.csv", index=False)
    F = {"lags": il.LAGS_P, "areas": list(COMPARE), "panels": panels}
    save_fd("lick_iti_area_compare", F)
    if draw:
        compare_draw(F)


def compare_draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    lags, areas = F["lags"], F["areas"]
    fig, axs = plt.subplots(2, len(areas), figsize=(2.3 * len(areas), 6.2), sharex=True, sharey="row")
    for j, ar in enumerate(areas):
        n_ep = max(len(F["panels"][(0, j)]["curves"]), len(F["panels"][(1, j)]["curves"]))
        for k in range(n_ep):                             # same drawing order as before: per epoch, raw then normalised
            for i in (0, 1):
                cur = F["panels"][(i, j)]["curves"]
                if k < len(cur):
                    ep, mu, se, n = cur[k]
                    axs[i, j].plot(lags, mu, color=colors[ep], lw=1.6, label=f"{ep} ({n})")
                    axs[i, j].fill_between(lags, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
        for i in (0, 1):
            if F["panels"][(i, j)]["sig"] is not None:
                il.sig_draw(axs[i, j], F["panels"][(i, j)]["sig"], colors)
        for i in range(2):
            ax = axs[i, j]
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            if i == 1:
                ax.axhline(1, color="0.8", lw=0.6, ls="--")
                ax.set_xlabel("s from lick onset", fontsize=8)
            ax.set_xlim(-1.5, 0.75)
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        grp = "OROFACIAL" if j < 3 else "control"
        axs[0, j].set_title(f"{ar.replace('_right', '')} contra\n({grp})", fontsize=9, fontweight="bold" if j < 3 else None)
    axs[0, 0].set_ylabel("dF/F x1e3 (shared axis)", fontsize=8)
    axs[1, 0].set_ylabel("/ animal's pre peak of that area", fontsize=8)
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("NEURAL, NO DECODER: self-initiated (ITI) licks (dF/F), orofacial ROIs next to control areas (contralesional; first ITI lick "
                 "after >= 2 s without licking; per-session mean PETH, all animals pooled, band = SEM over sessions; "
                 "bottom = each session / its animal's pre-stroke peak of that area).\n" + il.SIG_NOTE, fontsize=9.5,
                 wrap=True)
    il._headroom([axs[0, 0], axs[1, 0]])                 # y shared per row: raise each row once
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(il.OUT / "figures" / "lick_iti_area_compare.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if plot_only():                                      # redraw from figdata only
        draw_all(("compare",) if "--compare" in sys.argv else ("maps", "peth", "groups", "compare"))
        raise SystemExit(0)
    if "--compare" in sys.argv:                          # only the comparison figure
        labels_ = il.labels()
        compare(*session_table({x: il._load(x) for x in labels_}))
        raise SystemExit(0)
    raise SystemExit(main())
