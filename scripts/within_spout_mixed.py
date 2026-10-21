"""Mixed-effects readout of the within-spout deviation check (Priya 10-20: "let's try a mixed model - allowing time post
stroke to be a variable"). Reads the per-session table of `scripts/within_spout_deviation.py` (lick-aligned by default).

Per outcome (area slope, crossed = left - right, bilateral = (left + right) / 2; within- and between-spout; dF per 10 deg
executed lateral, x1e3), sessions as observations:

  models (wfield_local/mixed_models.py -- the shared definition): `time` y ~ post + post:lday, `behaviour`
  y ~ post + post:deficit (far-contra contact deficit), `epoch` y ~ C(epoch); random intercept + step + slope per animal
  with the module's fallback ladder (incl. fixed animal intercepts); FDR within model x term x kind; per-animal epoch
  means printed first (rule 8). Unweighted (per-session SEs not used). Pixel maps: the same
time model with FIXED animal intercepts (OLS per pixel; with 4 animals fixed and random intercepts nearly coincide),
coefficient maps with |t| > 3 outlined -- descriptive (pixels are correlated; the area table is the test).

Usage: PYTHONPATH=. python scripts/within_spout_mixed.py [--folder within_spout_deviation_lick]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AREAS = ("MOs", "MOp", "SSp-m", "SSp-ul", "SSp-ll", "SSp-n", "SSp-bfd", "SSp-tr", "SSs", "ACAd", "PL", "RSPd",
         "RSPagl", "VISp", "VISa", "AUDd")
FOCUS = ("SSp-m", "MOs", "MOp", "cortex")


def outcomes(D: pd.DataFrame) -> pd.DataFrame:
    """Long table: label, animal, epoch, day, kind (within / between), measure (e.g. 'SSp-m crossed'), y."""
    D = D[(D.variant == "mean") & ((D.spouts == "all") if "spouts" in D else True)]
    rows = []
    for kind in ("within", "between"):
        W = D.pivot_table(index="label", columns="area", values=f"b_{kind}") * 10 * 1e3
        out = {"cortex bilateral": W["cortex"]} if "cortex" in W else {}
        for a in AREAS:
            if f"{a}_left" in W and f"{a}_right" in W:
                out[f"{a} crossed"] = W[f"{a}_left"] - W[f"{a}_right"]
                out[f"{a} bilateral"] = (W[f"{a}_left"] + W[f"{a}_right"]) / 2
        for m, s in out.items():
            rows.append(pd.DataFrame({"label": s.index, "kind": kind, "measure": m, "y": s.to_numpy()}))
    from wfield_local import mixed_models as mm
    return mm.add_time_covariates(pd.concat(rows, ignore_index=True).dropna(subset=["y"]))


def outcomes_dirmid(DM: pd.DataFrame) -> pd.DataFrame:
    """Same long table from dirmid.csv: kind 'dir' (toward the mouse's left) / 'mid' (toward the midline)."""
    rows = []
    for kind in ("dir", "mid"):
        D = DM.assign(variant="mean", spouts="all", **{f"b_{k}": DM[f"b_{kind}"] for k in ("within", "between")})
        L = outcomes(D)
        L = L[L.kind == "within"].assign(kind=kind)
        rows.append(L)
    return pd.concat(rows, ignore_index=True)



def area_table(L: pd.DataFrame, qc_dir: Path, deficit: str | None = None) -> pd.DataFrame:
    """Time / behaviour / epoch models per kind x measure (wfield_local/mixed_models.py)."""
    from wfield_local import mixed_models as mm
    L = mm.add_deficit(L, mm.deficit(qc_dir, deficit or mm.DEFICIT))
    return mm.lmm_table(L, ["kind", "measure"], ("time", "behaviour", "epoch"), fdr_by=("kind",),
                        print_animals=False)              # the rule-8 table is printed by main (focus measures)


def pixel_maps(folder: Path, labels: list, figs: Path) -> None:
    """Per-pixel OLS: map ~ animal intercepts + post + post:lday; coefficient maps, |t| > 3 outlined (descriptive)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import epochs
    for kind in ("within_mean", "between_mean", "dir_mean", "mid_mean"):
        Y, meta, flat, sh = [], [], None, None
        for lab in labels:
            f = folder / "sessions" / f"{lab}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            keys = list(z["keys"])
            if kind not in keys:
                continue
            Y.append(z["maps"][keys.index(kind)] * 10 * 1e3)
            flat, sh = z["flat"], tuple(z["shape"])
            d = epochs.days_since_stroke(lab)
            meta.append((lab[:4], float(d > 0), np.log(max(d, 1)) if d > 0 else 0.0, epochs.epoch_of(lab)))
        if not Y:
            continue
        Y = np.stack(Y)
        an = sorted({m[0] for m in meta})
        X = np.column_stack([[float(m[0] == a) for m in meta] for a in an] + [[m[1] for m in meta],
                                                                              [m[2] for m in meta]])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        dof = max(len(Y) - X.shape[1], 1)
        cov = np.linalg.pinv(X.T @ X)
        s2 = (res ** 2).sum(0) / dof
        terms = {"pre level (mean of animal intercepts)": (np.full(len(an), 1 / len(an)).tolist() + [0, 0]),
                 "post step at day 1": [0] * len(an) + [1, 0],
                 "post: per e-fold of days": [0] * len(an) + [0, 1],
                 # the model's change AVERAGED OVER THE ACTUAL CHRONIC SESSIONS (each animal's own days; Priya 10-20:
                 # "chronic is different days for each animal") -- not a single assumed day
                 "chronic sessions vs pre (model, own days)":
                     [0] * len(an) + [1, float(np.mean([m[2] for m in meta if m[3] == "chronic"] or [np.log(36)]))]}
        fig, ax = plt.subplots(1, len(terms), figsize=(3.2 * len(terms), 3.4), squeeze=False)
        for j, (name, c) in enumerate(terms.items()):
            c = np.asarray(c, float)
            est = c @ beta
            t = est / np.sqrt(s2 * (c @ cov @ c))
            img, tm = np.full(sh[0] * sh[1], np.nan), np.zeros(sh[0] * sh[1])
            img[flat], tm[flat] = est, np.abs(t) > 3
            lim = np.nanpercentile(np.abs(img), 98) or 1e-6
            ax[0, j].imshow(img.reshape(sh), cmap="RdBu_r", vmin=-lim, vmax=lim)
            if (np.abs(t) > 3).any():
                ax[0, j].contour(tm.reshape(sh), levels=[0.5], colors="k", linewidths=0.6)
            ax[0, j].set_title(f"{name}\n|lim| {lim:.2g} (x1e-3 dF / 10 deg); |t|>3: {100 * np.mean(np.abs(t) > 3):.0f} %",
                               fontsize=7)
            ax[0, j].axis("off")
        from scripts.within_spout_deviation import KIND_TITLE
        fig.suptitle(f"{KIND_TITLE.get(kind, kind)}: session ~ animal + post + post:log(day), {len(Y)} sessions; "
                     "ipsi hemisphere = image left; outlines descriptive", fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / f"mixed_time_maps_{kind.split('_')[0]}.png", dpi=150)
        plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--folder", default="within_spout_deviation_lick")
    ap.add_argument("--deficit", default=None, help="behavioural index (mixed_models.DEFICITS); non-default -> *_<name>")
    a = ap.parse_args(argv)
    folder = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc" / a.folder
    D = pd.read_csv(folder / "within_spout_deviation.csv")
    L = outcomes(D)
    if (folder / "dirmid.csv").exists():                     # direction / toward-midline coefficients
        L = pd.concat([L, outcomes_dirmid(pd.read_csv(folder / "dirmid.csv"))], ignore_index=True)
    from wfield_local import mixed_models as mm
    print(mm.animal_epoch_table(L[L.measure.str.startswith(FOCUS)], ["kind", "measure"]).round(3).to_string())  # rule 8
    T = area_table(L, folder.parent, a.deficit)
    sfx = "" if (a.deficit or mm.DEFICIT) == mm.DEFICIT else f"_{a.deficit}"
    T.to_csv(folder / f"mixed_model{sfx}.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 230)
    for kind in [k for k in ("within", "between", "dir", "mid") if k in set(T.kind)]:
        for model in ("time", "behaviour"):
            print(f"\n=== {kind}-spout slope (x1e-3 dF / 10 deg executed lateral), {model} model; q = FDR over measures")
            S = T[(T.kind == kind) & (T.model == model) & T.measure.str.startswith(FOCUS)]
            print(S.pivot_table(index="measure", columns="term", values=["coef", "z", "q"]).round(3).to_string())
            print("  random-effect structure:", S.groupby("measure").re.first().value_counts().to_dict())
        print(f"--- {kind}, epoch model (vs pre)")
        S = T[(T.kind == kind) & (T.model == "epoch") & T.measure.str.startswith(FOCUS)]
        print(S.pivot_table(index="measure", columns="term", values=["coef", "z"]).round(3).to_string())
    print("\n  per-animal deviation of the post step (BLUP), time model, within:")
    S = T[(T.kind == "within") & (T.model == "time") & (T.term == "post") & T.measure.str.startswith(FOCUS)]
    print(S.set_index("measure")[[c for c in S.columns if c.startswith("step_dev_")]].round(3).to_string())
    figs = folder.parents[1] / "figures" / "within_spout" / folder.name   # results, not qc (Priya 10-21)
    figs.mkdir(parents=True, exist_ok=True)
    pixel_maps(folder, sorted(D.label.unique()), figs)
    print(f"\n-> {folder / 'mixed_model.csv'}, {figs}/mixed_time_maps_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
