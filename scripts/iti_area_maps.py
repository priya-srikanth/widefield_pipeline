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
Image left = ipsilesional (_left); _right = contralesional.
Usage: python scripts/iti_area_maps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from fr_musall import AN  # noqa: E402

PETH_AREAS = ["VISp", "VISl", "VISam", "RSPd", "SSp-bfd", "SSp-ul", "SSp-ll", "SSp-tr", "ACAd", "PL"]
ORO = ("ALM", "tjM1", "MOs", "MOp", "SSp-m", "SSp-n")
VISR = ("VIS", "RSP")


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import area_image, outline, paint

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    labels = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))
    D = {x: il._load(x) for x in labels}
    T, peths = session_table(D)
    T.to_csv(il.OUT / "area_auc_sessions.csv", index=False)
    C = pd.concat([changes(T, "resp"), changes(T, "build")])
    C.to_csv(il.OUT / "area_auc_changes.csv", index=False)
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    img, nm = area_image(labels[0])
    ol = outline(img)
    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    for i, (val, ttl) in enumerate((("resp", "response AUC 0..0.5 s"), ("build", "build-up AUC -0.4..-0.05 s"))):
        c = C[C.value == val]
        lim = np.nanpercentile(np.abs(c.delta), 98) or 1.0
        for j, ep in enumerate(il.EP4[1:]):
            ax = axs[i, j]
            ce_ = c[c.epoch == ep].set_index("area")
            M = paint(img, nm, ce_.delta.to_dict())
            im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            sig = paint(img, nm, {a: 1.0 for a in ce_.index[ce_.sig]})
            sig_edge = np.zeros(img.shape, bool)
            ok = np.isfinite(sig)
            sig_edge[:-1] |= ok[:-1] != ok[1:]
            sig_edge[:, :-1] |= ok[:, :-1] != ok[:, 1:]
            ax.imshow(np.where(sig_edge, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{ttl}\n{ep} - pre ({int(ce_.sig.sum())} areas significant)", fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label("change in dF/F AUC (x1e3 s)")
    fig.suptitle("NEURAL, NO DECODER: self-initiated (ITI) licks across the whole cortex -- change from pre per Allen area in raw dF/F "
                 "AUC (x1e3 s); black outline = nested-bootstrap CI excludes 0. Image left = "
                 "ipsilesional.", fontsize=10, wrap=True)
    fig.savefig(il.OUT / "figures" / "lick_iti_area_maps.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # PETHs, non-orofacial areas
    fig, axs = plt.subplots(2, len(PETH_AREAS), figsize=(2.3 * len(PETH_AREAS), 5.5), sharex=True)
    for j, base in enumerate(PETH_AREAS):
        for i, h in enumerate(("right", "left")):
            ax = axs[i, j]
            area = f"{base}_{h}"
            for ep in il.EP4:
                labs = [x for x in T[(T.area == area) & (T.epoch == ep)].label.unique() if (x, area) in peths]
                if not labs:
                    continue
                M = np.stack([peths[(x, area)] for x in labs])
                mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
                ax.plot(il.LAGS_P, mu, color=colors[ep], lw=1.6, label=ep)
                ax.fill_between(il.LAGS_P, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
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
    # group summary: orofacial vs visual + RSP
    T["group"] = np.where(T.area.str.rsplit("_", n=1).str[0].isin(ORO), "orofacial",
                          np.where(T.area.str.startswith(VISR), "visual + RSP", "other"))
    G = T[T.group != "other"].groupby(["label", "animal", "epoch", "group"])[["resp", "build"]].mean().reset_index()
    den = G[G.epoch == "pre"].groupby(["group", "animal"]).resp.mean()
    dg = np.array([den.get((g_, a_), np.nan) for g_, a_ in zip(G.group, G.animal)])
    G["resp_n"], G["build_n"] = G.resp / dg, G.build / dg
    rng = np.random.default_rng(4)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    tests = []
    for i, (val, ttl) in enumerate((("resp_n", "response AUC / pre"), ("build_n", "build-up AUC / pre response"))):
        for j, g in enumerate(("orofacial", "visual + RSP", "orofacial - visual + RSP")):
            ax = axs[i, j]
            if j < 2:
                sub = G[G.group == g]
            else:                                         # is the change orofacial-specific? session-paired difference
                Wd = G.pivot_table(index=["label", "animal", "epoch"], columns="group", values=val).dropna()
                sub = (Wd["orofacial"] - Wd["visual + RSP"]).rename(val).reset_index()
            for r_ in il.session_dots(ax, sub, val, colors, rng):
                tests.append({"group": g, **r_})
            ax.set_title(f"{g}: {ttl}", fontsize=9)
    pd.DataFrame(tests).to_csv(il.OUT / "area_group_tests.csv", index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    fig.suptitle("NEURAL, NO DECODER: self-initiated licks (dF/F), orofacial vs visual + retrosplenial areas (area means per session, divided by "
                 "the animal's pre group-mean response AUC; one dot per session; * = nested-bootstrap change from pre; right = "
                 "orofacial minus visual + RSP in the same session).", fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(il.OUT / "figures" / "lick_iti_area_groups.png", dpi=110)
    plt.close(fig)
    compare(T, peths)
    return 0


#: orofacial ROIs next to control areas (Priya 10-26: "control area figures for activity traces, e.g. VIS to compare to
#: ALM"); contralesional (_right) hemisphere
COMPARE = ["ALM_right", "tjM1_right", "SSp-m_right", "VISp_right", "VISam_right", "RSPd_right", "SSp-bfd_right", "ACAd_right"]


def compare(T: pd.DataFrame, peths: dict) -> None:
    """Orofacial vs control areas side by side: per-session mean PETH (all animals pooled, band = SEM over sessions);
    top raw with ONE shared y-axis, bottom each session divided by its animal's pre-stroke peak (0..0.5 s) of that area."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    post = (il.LAGS_P >= 0.0) & (il.LAGS_P <= 0.5)
    pk = {}
    for (x, ar), v in peths.items():
        a_ = T[T.label == x].animal.iloc[0]
        if T[T.label == x].epoch.iloc[0] == "pre":
            pk.setdefault((a_, ar), []).append(v)
    pk = {k: float(np.mean(v, 0)[post].max()) for k, v in pk.items()}
    fig, axs = plt.subplots(2, len(COMPARE), figsize=(2.3 * len(COMPARE), 6.2), sharex=True, sharey="row")
    shown = (il.LAGS_P >= -1.5) & (il.LAGS_P <= 0.75)
    sig_recs = []
    for j, ar in enumerate(COMPARE):
        long = {False: [], True: []}
        for ep in il.EP4:
            labs = sorted(x for x in T[(T.area == ar) & (T.epoch == ep)].label.unique() if (x, ar) in peths)
            if not labs:
                continue
            for i, norm in enumerate((False, True)):
                M = np.stack([peths[(x, ar)] / (pk.get((T[T.label == x].animal.iloc[0], ar), np.nan) if norm else 1.0)
                              for x in labs])
                mu, se = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(len(M))
                axs[i, j].plot(il.LAGS_P, mu, color=colors[ep], lw=1.6, label=f"{ep} ({len(M)})")
                axs[i, j].fill_between(il.LAGS_P, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
                long[norm] += [pd.DataFrame({"label": x, "animal": T[T.label == x].animal.iloc[0], "epoch": ep,
                                             "lag": il.LAGS_P[shown], "v": m_[shown]}) for x, m_ in zip(labs, M)]
        for i, norm in enumerate((False, True)):
            if long[norm]:
                sig_recs += il.sig_bars(axs[i, j], pd.concat(long[norm], ignore_index=True), colors, value="v",
                                        tag={"area": ar, "normalised": norm})
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
    pd.DataFrame(sig_recs).to_csv(il.OUT / "sig_bins_3_31.csv", index=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(il.OUT / "figures" / "lick_iti_area_compare.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if "--compare" in sys.argv:                          # only the comparison figure
        labels_ = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))
        compare(*session_table({x: il._load(x) for x in labels_}))
        raise SystemExit(0)
    raise SystemExit(main())
