"""Whole-cortex view of the self-initiated-lick response (Priya 10-26: "should we also do some non-orofacial per-area
activity maps to look at the whole cortex question? (supplemental figure with VIS areas)").

From the saved ITI sessions (`scripts/iti_licks.py`, every Allen area per bout-start ITI lick, >= 2 s without licking):
per session x area, response AUC (0..0.5 s after lick onset = first frame the tongue is visible) and build-up AUC
(-0.4..-0.05 s), baseline -1.5..-1.0 s, each divided by that area's PRE response AUC for the animal (pre response = 1).
  iti_area_maps.png   change from pre per area (acute / subacute / chronic) painted on the Allen atlas, one colour scale
                      per row; areas whose nested-bootstrap CI (animals > sessions) excludes 0 are outlined (black).
  iti_area_peth.png   per-session mean PETH +- SEM (all animals pooled) for non-orofacial areas, both hemispheres.
  iti_area_groups.png orofacial (ALM, tjM1, MOs, MOp, SSp-m, SSp-n) vs visual + retrosplenial areas: one dot per
                      session, normalised response and build-up.
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
    C = pd.concat([changes(T, "resp_n"), changes(T, "build_n")])
    C.to_csv(il.OUT / "area_auc_changes.csv", index=False)
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    img, nm = area_image(labels[0])
    ol = outline(img)
    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    for i, (val, ttl) in enumerate((("resp_n", "response AUC 0..0.5 s"), ("build_n", "build-up AUC -0.4..-0.05 s"))):
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
        fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label("change / animal's pre response AUC")
    fig.suptitle("NEURAL: self-initiated (ITI) licks across the whole cortex -- change from pre per Allen area, normalised to "
                 "each animal's pre response AUC of that area; black outline = nested-bootstrap CI excludes 0. Image left = "
                 "ipsilesional.", fontsize=10, wrap=True)
    fig.savefig(il.OUT / "figures" / "iti_area_maps.png", dpi=110, bbox_inches="tight")
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
    fig.suptitle("NEURAL: activity around self-initiated licks in NON-orofacial areas (per-session mean PETH, all animals "
                 "pooled; band = SEM over sessions; lick onset at 0).", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(il.OUT / "figures" / "iti_area_peth.png", dpi=110)
    plt.close(fig)
    # group summary: orofacial vs visual + RSP
    T["group"] = np.where(T.area.str.rsplit("_", n=1).str[0].isin(ORO), "orofacial",
                          np.where(T.area.str.startswith(VISR), "visual + RSP", "other"))
    G = T[T.group != "other"].groupby(["label", "animal", "epoch", "group"])[["resp_n", "build_n"]].mean().reset_index()
    rng = np.random.default_rng(4)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    tests = []
    for i, (val, ttl) in enumerate((("resp_n", "response AUC / pre"), ("build_n", "build-up AUC / pre response"))):
        for j, g in enumerate(("orofacial", "visual + RSP")):
            ax = axs[i, j]
            for r_ in il.session_dots(ax, G[G.group == g], val, colors, rng):
                tests.append({"group": g, **r_})
            ax.set_title(f"{g}: {ttl}", fontsize=9)
    pd.DataFrame(tests).to_csv(il.OUT / "area_group_tests.csv", index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    fig.suptitle("NEURAL: self-initiated licks, orofacial vs visual + retrosplenial areas (area means per session; one dot per "
                 "session; * = nested-bootstrap change from pre).", fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(il.OUT / "figures" / "iti_area_groups.png", dpi=110)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
