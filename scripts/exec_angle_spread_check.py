"""Is the post-stroke rise in EXECUTED-angle encoding stronger tracking or just more lick-angle spread? (Priya 2026-10-19,
on `position_dR2_maps_by_epoch.png`: the within-position executed-angle dR2 rises acutely / chronically, strongest in
the IPSILESIONAL (left = image-left) lateral cortex.)

Variance explained by one regressor scales with the regressor's variance (dR2 ~ beta^2 var(x) / var(y)), so a session
whose licks scatter more around their position's mean angle gives the same neural sensitivity more variance to
explain. Per session (main movement_first run, engaged trials, August-only pre):
  sd_dev   = SD of the lick's executed peak angle minus its position's mean (the exec_angle regressor; deg)
  dR2      = drop (order-free) and shuffle exec_angle dR2, lick window, per hemisphere (left = ipsilesional)
  sens     = dR2 / var(dev) * 100 (dR2 per 100 deg^2 of deviation variance: the sensitivity, spread removed)
If the epoch change survives in `sens`, cortex tracks executed direction more; if it vanishes, it was spread.

    python scripts/exec_angle_spread_check.py [--run lp_r4_heatmap_20261013__clean_v36]

Output -> <LP_analysis>/<run>/imaging_encoding/qc/exec_angle_spread.{csv,png}
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

AREAS = ("SSp-m", "MOs", "MOp", "SSp-ul", "SSp-n")
EPOCHS = ("pre", "acute", "subacute", "chronic")
HEMI = {"left": "ipsi (L)", "right": "contra (R)"}


def session_spread(rv, run: str, animal: str, date: str) -> dict:
    """SD / variance of the within-position executed-angle deviation over ENGAGED-trial licks (as the model builds it:
    position means over all cue-table licks, measured peaks with reach >= MIN_REACH_PX)."""
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    sd = oo.session_dir(rv, run, animal, date)
    keep, _ = oc.engagement_keep(animal, date, sd)
    cue_t = oc._cue_table(sd).set_index("trial_id")
    L = pd.read_csv(sd / "licks.csv")
    L = L[L.trial_id.isin(cue_t.index)]
    ok = (~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX) & np.isfinite(L.peak_angle_deg)
    L = L[ok].assign(pos=[oc.pos_key(p, "L") for p in L[ok].position])
    L = L.assign(dev=L.peak_angle_deg - L.groupby("pos").peak_angle_deg.transform("mean"))
    if keep is not None:
        L = L[L.trial_id.isin(keep)]
    out = {"n_licks": len(L), "sd_dev": float(L.dev.std()), "var_dev": float(L.dev.var())}
    for p, g in L.groupby("pos"):
        out[f"sd_dev_{p}"] = float(g.dev.std())
    return out


def main(argv=None) -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    a = ap.parse_args(argv)
    rv = PathResolver()
    base = oo.root(rv) / a.run / "imaging_encoding"
    out = base / "qc"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for jf in sorted((base / "movement_first" / "sessions").glob("PS*_*.json")):
        m = json.loads(jf.read_text(encoding="utf-8"))
        if m["epoch"] not in EPOCHS or (m["epoch"] == "pre" and m["date"][4:8] < oc.LATE_PRE_FROM):
            continue
        lab = m["label"]
        R = pd.read_csv(jf.with_name(f"{lab}_regions.csv"))
        R = R[R.window == "lick"].pivot_table(index="area", columns="model", values="r2")
        row = {"label": lab, "animal": m["animal"], "epoch": m["epoch"],
               **session_spread(rv, a.run, m["animal"], m["date"])}
        for ar in AREAS:
            for h in HEMI:
                k = f"{ar}_{h}"
                if k in R.index:
                    row[f"drop|{k}"] = R.loc[k, "full"] - R.loc[k, "drop:exec_angle"]
                    row[f"shuf|{k}"] = R.loc[k, "full"] - R.loc[k, "shuf:exec_angle"]
        rows.append(row)
        print(lab, m["epoch"], f"sd_dev {row['sd_dev']:.1f}", flush=True)
    D = pd.DataFrame(rows)
    for c in [c for c in D.columns if c.startswith(("drop|", "shuf|"))]:
        D["sens" + c[4:] if c.startswith("drop") else "senshuf" + c[4:]] = D[c] / D.var_dev * 100
    D.to_csv(out / "exec_angle_spread.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    show = ["sd_dev"] + [f"{k}|SSp-m_{h}" for k in ("drop", "shuf", "sens", "senshuf") for h in HEMI]
    for c in show:
        if c in D:
            t = D.groupby(["animal", "epoch"])[c].mean().unstack("epoch").reindex(columns=list(EPOCHS))
            print(f"\n{c} (per animal, mean over sessions)\n" + t.round(4).to_string())
    _figure(D, out / "exec_angle_spread.png", a.run)
    print(f"-> {out / 'exec_angle_spread.csv'}")
    return 0


def _figure(D: pd.DataFrame, path, run: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    from wfield_local.orofacial_cohort import save_png_svg
    acol = config.animal_color()
    eps = [e for e in EPOCHS if e in set(D.epoch)]
    animals = sorted(D.animal.unique())
    cols = [("sd_dev", "lick-angle spread\nSD of within-position deviation (deg)")]
    for ar in ("SSp-m", "MOs"):
        for h in HEMI:
            cols.append((f"drop|{ar}_{h}", f"{ar} {HEMI[h]}\nexec-angle drop dR$^2$ (lick window)"))
    for ar in ("SSp-m", "MOs"):
        for h in HEMI:
            cols.append((f"sens|{ar}_{h}", f"{ar} {HEMI[h]}\nSENSITIVITY = drop dR$^2$ / var(dev) x 100"))
    cols = [c for c in cols if c[0] in D]
    nc = 5
    nr = int(np.ceil(len(cols) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.6 * nr + 1.2), squeeze=False)
    for i, (c, lab) in enumerate(cols):
        ax = axs[i // nc][i % nc]
        means = []
        for ei, e in enumerate(eps):
            for ai, an in enumerate(animals):
                v = D[(D.epoch == e) & (D.animal == an)][c].dropna().to_numpy()
                if len(v):
                    xo = ei + (ai - (len(animals) - 1) / 2) * 0.16
                    ax.plot(np.full(len(v), xo), v, "o", ms=3, color=acol.get(an, "0.5"), alpha=0.45, mec="none")
                    ax.plot(xo, v.mean(), "D", ms=6, color=acol.get(an, "0.5"), mec="k", mew=0.6,
                            label=an if (i == 0 and ei == 0) else None)
            means.append(D[D.epoch == e].groupby("animal")[c].mean().mean())
        ax.plot(range(len(eps)), means, "-", color="k", lw=2, zorder=5)
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels(eps)
        ax.set_title(lab, fontsize=10)
    for j in range(len(cols), nr * nc):
        axs[j // nc][j % nc].set_axis_off()
    axs[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Executed-angle encoding: spread vs sensitivity -- {run}. Dots = sessions, diamonds = animal means, "
                 "line = mean of animals. Left hemisphere = IPSILESIONAL. If the rise survives as sensitivity, "
                 "cortex tracks executed direction more; if not, licks just scatter more.", fontsize=10.5, y=0.995,
                 va="top", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_png_svg(fig, path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
