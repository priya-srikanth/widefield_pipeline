"""Checks on the per-session vs frozen REMAPPING maps (Priya 10-25: "this is striking, can we review" -> checks 1, 2):

  1. TIME COURSE: session mixed models (`wfield_local.mixed_models`, time y ~ post + post:lday and behaviour y ~ post +
     post:deficit) of the remapping gain (self - frozen, per session) per area group x regressor group. A stroke effect
     steps up at the stroke and / or tracks the deficit; slow drift rises steadily with log day and is larger in later,
     better-behaving sessions (the Layer 2 control-area signature, ENCODING_FINDINGS 8.11).
  2. PER ANIMAL (rule 8): per area, the number of animals whose own chronic (and acute) change is > 0, painted on the atlas.

Reads `session_vs_frozen/sessions/*.json` (self-fit R2, all areas) and the frozen run's session jsons (same windows).
Usage: python scripts/remap_checks.py [--tag gate-attempted_v7g3_t3d_frA]
Output: <run>/session_vs_frozen/{remap_lmm_<deficit>.csv, figures/remap_lmm_<window>_<model>.png, figures/remap_per_animal_<window>.png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402
from session_vs_frozen import DEFAULT_VARIANTS, frozen_r2, self_r2  # noqa: E402
from specificity_control import GROUPS  # noqa: E402

QLAB = {"joint": "full", "no_target": "target", "no_spout": "spout", "no_task": "task", "no_movement": "movement",
        "no_fr": "fr", "no_fr_dir": "fr_dir"}


def remap_table(run: Path) -> pd.DataFrame:
    """Per session x window x area x q: gain = self - frozen (full) or unique_self - unique_frozen (groups)."""
    out = run / "session_vs_frozen"
    variants = [v for v in DEFAULT_VARIANTS if (run / v / "sessions").is_dir()]
    S = self_r2(out / "sessions", areas="all")
    F = pd.concat([frozen_r2(run, v, areas="all") for v in variants])
    G = S.merge(F, on=["label", "window", "area", "variant"])
    key = ["label", "animal", "epoch", "window", "area"]
    W = G.pivot_table(index=key, columns="variant", values=["self", "frozen"]).reset_index()
    rows = []
    for v in variants:
        if v == "joint":
            val = W[("self", "joint")] - W[("frozen", "joint")]
        else:
            val = (W[("self", "joint")] - W[("self", v)]) - (W[("frozen", "joint")] - W[("frozen", v)])
        rows.append(pd.DataFrame({k: W[k] for k in key}).assign(q=QLAB[v], u=val.to_numpy()))
    U = pd.concat(rows, ignore_index=True)
    U.columns = [c if isinstance(c, str) else c[0] for c in U.columns]
    return U


def group_rows(U: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_an, w, q), g in U[U.area != "cortex"].groupby(["animal", "window", "q"]):
        nlab = g.label.nunique()
        have = set(g.groupby("area").label.nunique().loc[lambda s, n=nlab: s == n].index)
        for gname, areas in GROUPS.items():
            use = [a for a in areas if a in have]
            if use:
                rows.append(g[g.area.isin(use)].groupby("label").u.mean().reset_index().assign(area=gname, window=w, q=q))
    C = U[U.area == "cortex"][["label", "u", "window", "q"]].assign(area="cortex")
    return pd.concat([*rows, C], ignore_index=True)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from frozen_area_maps import area_image, outline, paint
    from wfield_local import mixed_models as mm
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    ap.add_argument("--deficits", nargs="+", default=["ttc", "contact"])
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = run / "session_vs_frozen"
    (out / "figures").mkdir(exist_ok=True)
    U = remap_table(run)
    pd.set_option("display.width", 230)

    # CHECK 1: time course
    Gr = group_rows(U)
    Gr = Gr[Gr.area.isin(["orofacial contra", "orofacial ipsi", "control", "cortex"])]
    L = Gr.assign(source="remap", measure=Gr.q + " | " + Gr.window + " | " + Gr.area, y=Gr.u)[["label", "source", "measure", "y"]]
    L = mm.add_time_covariates(L)
    for dname in a.deficits:
        Ld = mm.add_deficit(L, mm.deficit(E / "qc", dname))
        T = mm.lmm_table(Ld, ["source", "measure"], ("time", "behaviour"), fdr_by=("source",),
                         print_animals=(dname == a.deficits[0]))
        T.to_csv(out / f"remap_lmm_{dname}.csv", index=False, float_format="%.5g")
        for model in ("time", "behaviour"):
            if model == "time" and dname != a.deficits[0]:
                continue
            S_ = T[(T.model == model) & (T.term != "Intercept")]
            print(f"\n=== {model} (deficit {dname})\n" + S_.pivot_table(index="measure", columns="term",
                                                                     values=["coef", "z", "q"]).round(3).to_string())
            sfx = "" if model == "time" else f"_{dname}"
            for w in ("cue", "lick"):
                panels = [(f"{q} | {ar}", {"source": "remap", "measure": f"{q} | {w} | {ar}"})
                          for q in ("full", "target", "spout", "movement") for ar in ("orofacial contra", "control")]
                mm.plot_fits(Ld, T, panels, model, out / "figures" / f"remap_lmm_{w}_{model}{sfx}.png",
                             f"NEURAL: remapping gain (self - frozen R2) per session, {w} window", "self - frozen (R2)", ncols=4)

    # CHECK 2: per-animal counts per area
    A = U[U.area != "cortex"]
    pre = A[A.epoch == "pre"].groupby(["animal", "window", "area", "q"]).u.mean()
    img, nm = area_image(A.label.iloc[0])
    ol = outline(img)
    qs = [q for q in QLAB.values() if q in set(A.q)]
    for w in ("cue", "lick"):
        fig, axs = plt.subplots(2, len(qs), figsize=(2.6 * len(qs) + 0.6, 6.0), squeeze=False)
        for i, ep in enumerate(("acute", "chronic")):
            post = A[(A.epoch == ep) & (A.window == w)].groupby(["animal", "area", "q"]).u.mean()
            d = (post - pre.xs(w, level="window")).dropna().rename("d").reset_index()
            for j, q in enumerate(qs):
                dq = d[d.q == q]
                n_up = dq.assign(up=dq.d > 0).groupby("area").up.sum()
                n_an = dq.groupby("area").animal.nunique()
                net = (2 * n_up - n_an).to_dict()               # up minus down
                ax = axs[i, j]
                im = ax.imshow(paint(img, nm, net), cmap="RdBu_r", vmin=-4, vmax=4, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                sg = paint(img, nm, {k: 1.0 for k, v in n_up.items() if v >= 3 or n_an[k] - v >= 3})
                if np.isfinite(sg).any():
                    ax.contour(np.nan_to_num(sg), levels=[0.5], colors="k", linewidths=1.2)
                ax.text(0.02, 0.02, f">=3 up: {int((n_up >= 3).sum())}, >=3 down: {int(((n_an - n_up) >= 3).sum())}",
                        transform=ax.transAxes, fontsize=6.5, bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "lw": 0})
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(q if q == "full" else f"{q} unique", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"{ep} - pre", fontsize=8)
        cb = fig.colorbar(im, cax=fig.add_axes([0.25, 0.04, 0.5, 0.02]), orientation="horizontal", ticks=[-4, -2, 0, 2, 4])
        cb.set_label("animals up minus down (of 4)", fontsize=7)
        fig.suptitle(f"NEURAL: REMAPPING per animal (rule 8), {w} window ({a.tag}): per area, animals whose own epoch - pre "
                     "change in (self - frozen) is > 0 minus < 0; black = >= 3 of 4 agree. Image left = ipsilesional.",
                     fontsize=8.5)
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.11, top=0.88, wspace=0.06, hspace=0.12)
        fig.savefig(out / "figures" / f"remap_per_animal_{w}.png", dpi=110)
        plt.close(fig)
    print("->", out / "figures", "; animals:", AN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
