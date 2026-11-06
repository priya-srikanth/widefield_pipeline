"""Musall-style PETH vs TRIAL-BY-TRIAL maps in the FROZEN frame, at Allen-area resolution (Priya 10-25: "do we have PETH vs
trial by trial maps (as in the Musall paper)"). Musall et al. 2019 Fig. 4 maps cvR2 of the full model and each group's
unique contribution separately for the trial-averaged response (PETH) and single trials; here the same split per Allen
area from the frozen run's saved per-area R2 (`frozen_encoder.musall_r2`: "avg" = R2 of the per-spout trial averages,
"trial" = R2 after removing each spout's mean, i.e. trial-to-trial variability around the PETH).

Quantities: full model R2; unique R2 of g = R2[joint] - R2[no_g] for task, spout, movement (+ fr, fr_dir when the run has
FaceRhythm). Rows: pre level (PETH, trial), then acute and chronic minus pre (PETH, trial), within animal then mean of
animals; outline = area whose change is significant after Bonferroni over areas (nested bootstrap animals -> sessions).
Pixel resolution needs per-pixel PETH / window moments that the basis run does not save (queued as a separate pass).

ALM / tjM1 (Priya 11-06, deck slide 94): with --split, *_split companions of each figure paint MOs / MOp as ALM + MOs_rest
/ tjM1 + MOp_rest (ROI v8 on the stat mask, `encoder_split_rows.split_image`, as residual_roi / specificity_control).
The area run saved Allen-area means only, so the split values are the frozen Musall R2 of each ROI mean from the
FROZEN_PIX_MUSALL basis re-run with the area run's alphas (`frozen_encoder_<tag>_basis_pixmusall_union`,
`encoder_split_rows.musall_rows`); own bootstrap, Allen Bonferroni count, Allen colour scales. Allen outputs unchanged.

Usage: python scripts/frozen_peth_trial_maps.py [--tag gate-attempted_v7g3_t3d_frA] [--alone] [--split] [--out DIR]
Output: <E>/frozen_encoder_<tag>/figures/peth_trial_maps_<window>[_alone][_split].png, peth_trial_area[_alone][_split].csv
"""
from __future__ import annotations
from wfield_local.frozen_encoder import design_tag as _design_tag  # frame tag of the FROZEN_LICKS_ALL design (8.51)

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
#: every variant the two layouts read (the --predict list of the pixmusall union re-run must cover them)
VARIANTS = ["joint", *(v for _q, v in QS + QS_ALONE if v is not None)]


def table(run: Path, variant: str) -> pd.DataFrame:
    from wfield_local import config
    rows = []
    for js in sorted((run / variant / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if config.early_pre(m["label"]):
            continue
        for w, d in m["r2_musall"].items():
            for meas, _ in MEASURES:
                for ai, ar in enumerate(m["areas"]):
                    if ar != "cortex":
                        rows.append((m["label"], m["animal"], m["epoch"], w, meas, ar, d[meas][ai]))
    return pd.DataFrame(rows, columns=["label", "animal", "epoch", "window", "measure", "area", "r2"])


def split_table(pix: Path, fit_run: Path, variant: str) -> pd.DataFrame:
    """`table` counterpart for the ALM / tjM1 / MOs_rest / MOp_rest rows: frozen Musall R2 (avg = PETH, trial) of each
    split ROI mean from the FROZEN_PIX_MUSALL basis re-run ``pix`` (`encoder_split_rows.musall_rows`; early pre dropped
    as in `table`)."""
    import encoder_split_rows as esr
    from wfield_local import config
    t = pd.concat([esr.musall_rows(pix, variant, meas, tuple(esr.SPLIT_AREAS), fit_run=fit_run).assign(measure=meas)
                   for meas, _ in MEASURES], ignore_index=True)
    t = t[[not config.early_pre(x) for x in t.label]]
    return t[["label", "animal", "epoch", "window", "measure", "area", "r2"]].reset_index(drop=True)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import encoder_split_rows as esr
    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=_design_tag("gate-attempted_v7g3_t3d_frA"))
    ap.add_argument("--alone", action="store_true",
                    help="Musall et al. 2019 layout: each group ALONE (single-group model R2) instead of unique R2")
    ap.add_argument("--split", action="store_true",
                    help="also draw *_split companions: MOs / MOp painted as ALM + MOs_rest / tjM1 + MOp_rest, from the "
                         "FROZEN_PIX_MUSALL union re-run (encoder_split_rows.PIX_UNION)")
    ap.add_argument("--pix-run", default=None,
                    help=f"pixmusall run folder (default <E>/frozen_encoder_<tag>{esr.PIX_UNION}); implies --split")
    ap.add_argument("--fit-run", default=None, help="folder holding fit/spatial_<animal>.pkl (default <tag>_basis)")
    ap.add_argument("--out", default=None, help="output folder (default the run folder; figures in <out>/figures)")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = Path(a.out) if a.out else run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    split = bool(a.split or a.pix_run)
    pix = fit_run = None
    if split:
        pix0, fit0 = esr.pix_union_paths(E, a.tag)
        pix, fit_run = Path(a.pix_run or pix0), Path(a.fit_run or fit0)
        if not (pix / "joint" / "sessions").is_dir():
            raise SystemExit(f"--split: no pixmusall run at {pix} (render the union re-run first)")
    key = ["label", "animal", "epoch", "window", "measure", "area"]
    J = table(run, "joint").set_index(key).r2
    JS = split_table(pix, fit_run, "joint").set_index(key).r2 if split else None
    parts, sparts = [], []
    qs = []
    sfx = "_alone" if a.alone else ""
    for q, v in (QS_ALONE if a.alone else QS):
        if v is not None and not (run / v / "sessions").is_dir():
            continue
        val = J if v is None else (table(run, v).set_index(key).r2 if a.alone else J - table(run, v).set_index(key).r2)
        parts.append(val.rename("u").reset_index().assign(q=q))
        qs.append(q)
        if split:
            if v is not None and not (pix / v / "sessions").is_dir():
                print(f"  !! split rows: {pix.name} has no {v}: the '{q}' split ROIs are left blank", flush=True)
                continue
            sv = JS if v is None else split_table(pix, fit_run, v).set_index(key).r2
            sparts.append((sv if (v is None or a.alone) else JS - sv).rename("u").reset_index().assign(q=q))
    U = pd.concat(parts, ignore_index=True)
    B = pd.concat([ffc.boot(U[U.q == q], "u", epochs=["acute", "chronic"], keys=("window", "measure", "area")).assign(q=q)
                   for q in qs], ignore_index=True)
    B = B[B.epoch.isin(["acute", "chronic"])].copy()
    B["p_bonf"] = np.minimum(B.p * B.groupby(["q", "window", "measure", "epoch"]).area.transform("nunique"), 1.0)
    B.to_csv(out / f"peth_trial_area{sfx}.csv", index=False)
    pre = U[U.epoch == "pre"].groupby(["q", "window", "measure", "area", "animal"]).u.mean() \
        .groupby(["q", "window", "measure", "area"]).mean()
    pd.set_option("display.width", 220)
    s = B[B.area.isin(["SSp-m_right", "SSp-m_left"]) & (B.epoch == "chronic")]
    print("== chronic minus pre, SSp-m (per-animal values in columns)\n"
          + s[["q", "window", "measure", "area", "point", "lo", "hi", "n_up", "n", "p_bonf"]].round(3).to_string(index=False))
    img, nm = area_image(U.label.iloc[0])
    ol = outline(img)
    kinds = [("allen", img, nm, ol, pre, B)]
    if sparts:
        # split ROIs: their own bootstrap (each group draws from its own seed, so the Allen rows above are untouched),
        # tested with the SAME Bonferroni count as the Allen areas (encoder_split_rows.bonf_allen, as residual_roi.py)
        US = pd.concat(sparts, ignore_index=True)
        BS = pd.concat([ffc.boot(US[US.q == q], "u", epochs=["acute", "chronic"], keys=("window", "measure", "area"))
                        .assign(q=q) for q in US.q.unique()], ignore_index=True)
        BS = BS[BS.epoch.isin(["acute", "chronic"])].copy()
        BS["p_bonf"] = esr.bonf_allen(BS, ["q", "window", "measure", "epoch"], ref=B)
        BS.to_csv(out / f"peth_trial_area{sfx}_split.csv", index=False)
        preS = US[US.epoch == "pre"].groupby(["q", "window", "measure", "area", "animal"]).u.mean() \
            .groupby(["q", "window", "measure", "area"]).mean()
        s = BS[BS.epoch == "chronic"]
        print("== split ROIs, chronic minus pre\n"
              + s[["q", "window", "measure", "area", "point", "lo", "hi", "n_up", "n", "p_bonf"]].round(3).to_string(index=False))
        img_s, nm_s = esr.split_image(img, nm)           # MOs / MOp drawn as their split ROIs (residual_roi.py)
        kinds.append(("split", img_s, nm_s, outline(img_s), pd.concat([pre, preS]).sort_index(),
                      pd.concat([B, BS], ignore_index=True)))
    rows_ = [(f"pre, {ml}", "pre", mk) for mk, ml in MEASURES] + \
            [(f"{ep} - pre, {ml}", ep, mk) for ep in ("acute", "chronic") for mk, ml in MEASURES]
    for kind, img_, nm_, ol_, pre_, B_ in kinds:
        for w in ("cue", "lick"):
            fig, axs = plt.subplots(len(rows_), len(qs), figsize=(2.6 * len(qs) + 0.6, 2.5 * len(rows_) + 0.9),
                                    squeeze=False)
            for j, q in enumerate(qs):
                pl = pre.loc[q, w]                       # colour scales from the ALLEN values (the same in both kinds)
                vmax = np.nanpercentile(pl.to_numpy(), 98)
                lim = np.nanpercentile(np.abs(B[(B.q == q) & (B.window == w)].point), 98) or 1e-3
                pl_ = pre_.loc[q, w]
                ch = B_[(B_.q == q) & (B_.window == w)]
                for i, (title, ep, mk) in enumerate(rows_):
                    ax = axs[i, j]
                    if ep == "pre":
                        im = ax.imshow(paint(img_, nm_, pl_.loc[mk].to_dict()), cmap="viridis",
                                       vmin=0 if q == "full" else None, vmax=vmax, interpolation="nearest")
                    else:
                        s_ = ch[(ch.epoch == ep) & (ch.measure == mk)].set_index("area")
                        im = ax.imshow(paint(img_, nm_, s_.point.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim,
                                       interpolation="nearest")
                        sg = paint(img_, nm_, {k: 1.0 for k, v in (s_.p_bonf < 0.05).items() if v})
                        if np.isfinite(sg).any():         # white halo + black line: visible on dark and light fills
                            ax.contour(np.nan_to_num(sg), levels=[0.5], colors="w", linewidths=3.4)
                            ax.contour(np.nan_to_num(sg), levels=[0.5], colors="k", linewidths=1.6)
                        sa = s_[~s_.index.isin(esr.SPLIT_AREAS)]
                        up = int(((sa.p_bonf < 0.05) & (sa.point > 0)).sum())
                        dn = int(((sa.p_bonf < 0.05) & (sa.point < 0)).sum())
                        txt = f"sig {up} up / {dn} down"
                        if kind == "split":
                            ss_ = s_[s_.index.isin(esr.SPLIT_AREAS)]
                            txt += f"\nsplit ROIs sig {int((ss_.p_bonf < 0.05).sum())} / {len(ss_)}"
                        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6.5,
                                bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "lw": 0})
                    ax.imshow(ol_, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                    ax.set_xticks([]); ax.set_yticks([])
                    if i == 0:
                        ax.set_title(q if q == "full" else (f"{q} alone" if sfx else f"{q} unique"), fontsize=8.5)
                    if j == 0:
                        ax.set_ylabel(title, fontsize=7.5)
                    if i in (1, 3, 5):                    # one bar per row pair, in its own inset below the pair
                        cax = ax.inset_axes([0.1, -0.13, 0.8, 0.06])
                        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                        cb.ax.tick_params(labelsize=6)
            title = (f"NEURAL: Musall-style PETH vs TRIAL-BY-TRIAL R2 by Allen area, FROZEN frame ({a.tag}), {w} window. "
                     "Columns: full model, then each group's " + ("R2 ALONE (single-group model; Musall et al. 2019 layout)" if sfx else "unique R2 (joint - without it)") + ". Rows: pre level, then acute and "
                     "chronic minus pre; PETH = per-spout trial averages, trial-by-trial = variability around them. Outline = "
                     "area whose change is significant after Bonferroni over areas (nested bootstrap); corner = count. "
                     "Image left = ipsilesional.")
            if kind == "split":
                title += (" MOs / MOp drawn as ALM + MOs_rest / tjM1 + MOp_rest (ROI v8 on the stat mask; R2 of each ROI "
                          f"mean from the basis re-run {pix.name}, area-run alphas), tested with the Allen Bonferroni "
                          "count; colour scales as the Allen figure.")
            fig.suptitle(textwrap.fill(title, width=int(42 * len(qs))), fontsize=8.5)   # wrapped to the figure width
            fig.subplots_adjust(left=0.05, right=0.97, bottom=0.02, top=0.9 if kind == "allen" else 0.88, wspace=0.06,
                                hspace=0.3)
            fig.savefig(out / "figures" / f"peth_trial_maps_{w}{sfx}{'_split' if kind == 'split' else ''}.png", dpi=105)
            plt.close(fig)
    print("->", out / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
