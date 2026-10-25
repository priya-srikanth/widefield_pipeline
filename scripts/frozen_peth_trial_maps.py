"""Musall-style PETH vs TRIAL-BY-TRIAL maps in the FROZEN frame, at Allen-area resolution (Priya 10-25: "do we have PETH vs
trial by trial maps (as in the Musall paper)"). Musall et al. 2019 Fig. 4 maps cvR2 of the full model and each group's
unique contribution separately for the trial-averaged response (PETH) and single trials; here the same split per Allen
area from the frozen run's saved per-area R2 (`frozen_encoder.musall_r2`: "avg" = R2 of the per-spout trial averages,
"trial" = R2 after removing each spout's mean, i.e. trial-to-trial variability around the PETH).

Quantities: full model R2; unique R2 of g = R2[joint] - R2[no_g] for task, spout, movement (+ fr, fr_dir when the run has
FaceRhythm). Rows: pre level (PETH, trial), then acute and chronic minus pre (PETH, trial), within animal then mean of
animals; outline = area whose change is significant after Bonferroni over areas (nested bootstrap animals -> sessions).
Pixel resolution needs per-pixel PETH / window moments that the basis run does not save (queued as a separate pass).

Usage: python scripts/frozen_peth_trial_maps.py [--tag gate-attempted_v7g3_t3d_frA]
Output: <E>/frozen_encoder_<tag>/figures/peth_trial_maps_<window>.png, peth_trial_area.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

QS = [("full", None), ("task", "no_task"), ("spout", "no_spout"), ("movement", "no_movement"), ("FaceRhythm", "no_fr"),
      ("FR direction", "no_fr_dir")]
#: Musall et al. 2019 (PETH / trial-by-trial prediction panels): each variable group ALONE. instructed = our lick group
#: (lick timing / speed / opening deviations), uninstructed = "other" (nose, running, other pose; FaceRhythm not in it)
QS_ALONE = [("full", None), ("task", "task_only"), ("instructed (lick)", "lick_only"), ("uninstructed (other)", "other_only"),
            ("spout", "spout_only"), ("movement", "movement_only")]
MEASURES = [("avg", "PETH"), ("trial", "trial-by-trial")]


def table(run: Path, variant: str) -> pd.DataFrame:
    rows = []
    for js in sorted((run / variant / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["label"][-4:] < "0801":
            continue
        for w, d in m["r2_musall"].items():
            for meas, _ in MEASURES:
                for ai, ar in enumerate(m["areas"]):
                    if ar != "cortex":
                        rows.append((m["label"], m["animal"], m["epoch"], w, meas, ar, d[meas][ai]))
    return pd.DataFrame(rows, columns=["label", "animal", "epoch", "window", "measure", "area", "r2"])


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    ap.add_argument("--alone", action="store_true",
                    help="Musall et al. 2019 layout: each group ALONE (single-group model R2) instead of unique R2")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    key = ["label", "animal", "epoch", "window", "measure", "area"]
    J = table(run, "joint").set_index(key).r2
    parts = []
    qs = []
    sfx = "_alone" if a.alone else ""
    for q, v in (QS_ALONE if a.alone else QS):
        if v is not None and not (run / v / "sessions").is_dir():
            continue
        val = J if v is None else (table(run, v).set_index(key).r2 if a.alone else J - table(run, v).set_index(key).r2)
        parts.append(val.rename("u").reset_index().assign(q=q))
        qs.append(q)
    U = pd.concat(parts, ignore_index=True)
    B = pd.concat([ffc.boot(U[U.q == q], "u", epochs=["acute", "chronic"], keys=("window", "measure", "area")).assign(q=q)
                   for q in qs], ignore_index=True)
    B = B[B.epoch.isin(["acute", "chronic"])].copy()
    B["p_bonf"] = np.minimum(B.p * B.groupby(["q", "window", "measure", "epoch"]).area.transform("nunique"), 1.0)
    B.to_csv(run / f"peth_trial_area{sfx}.csv", index=False)
    pre = U[U.epoch == "pre"].groupby(["q", "window", "measure", "area", "animal"]).u.mean() \
        .groupby(["q", "window", "measure", "area"]).mean()
    pd.set_option("display.width", 220)
    s = B[B.area.isin(["SSp-m_right", "SSp-m_left"]) & (B.epoch == "chronic")]
    print("== chronic minus pre, SSp-m (per-animal values in columns)\n"
          + s[["q", "window", "measure", "area", "point", "lo", "hi", "n_up", "n", "p_bonf"]].round(3).to_string(index=False))
    img, nm = area_image(U.label.iloc[0])
    ol = outline(img)
    rows_ = [(f"pre, {ml}", "pre", mk) for mk, ml in MEASURES] + \
            [(f"{ep} - pre, {ml}", ep, mk) for ep in ("acute", "chronic") for mk, ml in MEASURES]
    for w in ("cue", "lick"):
        fig, axs = plt.subplots(len(rows_), len(qs), figsize=(2.6 * len(qs) + 0.6, 2.5 * len(rows_) + 0.9), squeeze=False)
        for j, q in enumerate(qs):
            pl = pre.loc[q, w]
            vmax = np.nanpercentile(pl.to_numpy(), 98)
            ch = B[(B.q == q) & (B.window == w)]
            lim = np.nanpercentile(np.abs(ch.point), 98) or 1e-3
            for i, (title, ep, mk) in enumerate(rows_):
                ax = axs[i, j]
                if ep == "pre":
                    im = ax.imshow(paint(img, nm, pl.loc[mk].to_dict()), cmap="viridis", vmin=0 if q == "full" else None,
                                   vmax=vmax, interpolation="nearest")
                else:
                    s_ = ch[(ch.epoch == ep) & (ch.measure == mk)].set_index("area")
                    im = ax.imshow(paint(img, nm, s_.point.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim,
                                   interpolation="nearest")
                    sg = paint(img, nm, {k: 1.0 for k, v in (s_.p_bonf < 0.05).items() if v})
                    if np.isfinite(sg).any():                 # white halo + black line: visible on dark and light fills
                        ax.contour(np.nan_to_num(sg), levels=[0.5], colors="w", linewidths=3.4)
                        ax.contour(np.nan_to_num(sg), levels=[0.5], colors="k", linewidths=1.6)
                    up = int(((s_.p_bonf < 0.05) & (s_.point > 0)).sum())
                    dn = int(((s_.p_bonf < 0.05) & (s_.point < 0)).sum())
                    ax.text(0.02, 0.02, f"sig {up} up / {dn} down", transform=ax.transAxes, fontsize=6.5,
                            bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "lw": 0})
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(q if q == "full" else (f"{q} alone" if sfx else f"{q} unique"), fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(title, fontsize=7.5)
                if i in (1, 3, 5):                            # one bar per row pair, in its own inset below the pair
                    cax = ax.inset_axes([0.1, -0.13, 0.8, 0.06])
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.ax.tick_params(labelsize=6)
        title = (f"NEURAL: Musall-style PETH vs TRIAL-BY-TRIAL R2 by Allen area, FROZEN frame ({a.tag}), {w} window. "
                 "Columns: full model, then each group's " + ("R2 ALONE (single-group model; Musall et al. 2019 layout)" if sfx else "unique R2 (joint - without it)") + ". Rows: pre level, then acute and "
                 "chronic minus pre; PETH = per-spout trial averages, trial-by-trial = variability around them. Outline = "
                 "area whose change is significant after Bonferroni over areas (nested bootstrap); corner = count. "
                 "Image left = ipsilesional.")
        fig.suptitle(textwrap.fill(title, width=int(42 * len(qs))), fontsize=8.5)   # wrapped to the figure width
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.02, top=0.9, wspace=0.06, hspace=0.3)
        fig.savefig(run / "figures" / f"peth_trial_maps_{w}{sfx}.png", dpi=105)
        plt.close(fig)
    print("->", run / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
