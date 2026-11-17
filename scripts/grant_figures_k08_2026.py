#!/usr/bin/env python3
"""GRANT FIGURES ONLY — K08 resubmission, November 2026 deadline.

This script exists solely to render panels for the K08 Research Strategy figures.
It is NOT part of the analysis pipeline and nothing here feeds any result.
It recomputes nothing: every panel is drawn from the CSV companions written by the
epoch render set, exactly as `wf_grant_panels.py` does for the CAMS figure.

Output goes to ONE clearly named directory (see --out), default:

    <grant_figures root>/K08_2026_grant_panels/

Each panel is written as both .svg (editable text, for Illustrator) and .png (600 dpi).

Panel manifest lives in PANELS below and mirrors
`K08/Proposal/2026/K08_panel_manifest.md`.

Usage, on the imaging computer where the data lives:
    python scripts/grant_figures_k08_2026.py --data <.../grant_figures/epoch/data> \
                                             --out  <.../K08_2026_grant_panels>
    python scripts/grant_figures_k08_2026.py --list     # show the manifest, render nothing
"""
from __future__ import annotations
import argparse, csv, pathlib, sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# House style — identical to wf_grant_panels.py so the K08 and CAMS figures match
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "svg.fonttype": "none",            # keep text as text for Illustrator
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.2,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 2.0, "pdf.fonttype": 42, "ps.fonttype": 42,
})

POS        = ["nI", "nM", "nC", "fI", "fM", "fC"]
TICK_SHORT = ["Ipsi", "Mid", "Contra", "Ipsi", "Mid", "Contra"]
EPOCHS     = ["pre", "acute", "subacute", "chronic"]

# Deck colours, so grant panels match the Overview deck the PI already knows
EPOCH_COLOUR = {"pre": "#808080", "acute": "#e0218a",
                "subacute": "#d4a017", "chronic": "#1b7f79"}

# ----------------------------------------------------------------------------
# Manifest. source="csv" panels are rendered here as vector.
# source="deck"/"server" panels are raster-only and are placed in Illustrator.
# ----------------------------------------------------------------------------
PANELS = [
    # --- Figure 1: model, deficit, behaviour -------------------------------
    ("F1", "histology",            "reuse",  "existing grant figure"),
    ("F1", "fluoroscopy_frames",   "server", "Fluoroscopy/grant figures/swallow_frames.svg"),
    ("F1", "fluoroscopy_metrics",  "server", "per_swallow_distributions_panel_*.svg"),
    ("F1", "fluoroscopy_bolus",    "server", "bolus_density.svg + swallow_bolus.svg"),
    ("F1", "task_schematic",       "reuse",  "existing 6-spout schematic"),
    ("F1", "hit_rate_by_spout",    "csv",    "epoch_1c_behaviour_timecourse / deck slide 4"),
    ("F1", "cam1_cam4_frame",      "deck",   "slide 29 mp4, camera block only, no 3-D"),
    ("F1", "first_lick_aim",       "deck",   "slide 6, WEIGHTED panel cropped off"),
    ("F1", "paths_3d",             "deck",   "slide 61"),
    ("F1", "trial_outcomes",       "deck",   "slide 37"),
    # --- Figure 2: cortical representation ---------------------------------
    ("F2", "frozen_refit_gap",     "deck",   "slide 20"),
    ("F2", "acc_by_position_cue",  "csv",    "epoch_acc_by_position_cue_working"),
    ("F2", "acc_by_position_precue","csv",   "epoch_acc_by_position_precue_working"),
    ("F2", "lick_maps_2row",       "deck",   "slide 25 (already the 2-row subset)"),
    # --- Figure 3: FaceRhythm and encoder ----------------------------------
    ("F3", "fr_master_face",       "deck",   "slide 72"),
    ("F3", "fr_12_factors",        "deck",   "slide 74"),
    ("F3", "fr_lickR_lickL",       "deck",   "slide 77"),
    ("F3", "fr_loading_change",    "deck",   "slide 85"),
    ("F3", "encoder_residual",     "deck",   "slide 95 + pre-stroke level from encoding_goal_page_src"),
    ("F3", "encoder_roi_cued",     "server", "cued ROI row, tjM1 + SSp-m R/L only"),
    ("F3", "readout_cued_first",   "server", "like-for-like offset, cued FIRST lick panel"),
    ("F3", "area_unique_maps",     "server", "lick_area_unique_maps_* (ROI-parcellated)"),
]


def read_csv(data: pathlib.Path, name: str):
    with open(data / name, newline="") as fh:
        return list(csv.DictReader(fh))


def read_meta(data: pathlib.Path, name: str):
    out = {}
    with open(data / name, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) >= 3 and r[0] != "kind":
                out.setdefault(r[0], {})[r[1]] = r[2]
    return out


def save(fig, out: pathlib.Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.svg", dpi=600, transparent=True)
    fig.savefig(out / f"{stem}.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"  wrote {stem}.svg / .png")


# ----------------------------------------------------------------------------
# Panel renderers
# ----------------------------------------------------------------------------
def panel_acc_by_position(data, out, window: str, stem: str, width_in=3.5, height_in=1.9):
    """Per-position decoding accuracy by epoch. window = 'cue_working' | 'precue_working'."""
    base = f"epoch_acc_by_position_{window}"
    rows = read_csv(data, f"{base}.csv")
    meta = read_meta(data, f"{base}_meta.csv")
    chance = float(meta.get("chance", {}).get("", 0.1667))
    ylo = float(meta.get("ylim", {}).get("lo", 0.0))
    yhi = float(meta.get("ylim", {}).get("hi", 1.1))

    val = {(r["epoch"], r["position"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    w, gap = 0.19, 0.10
    for pi, p in enumerate(POS):
        x0 = pi * (1 + gap)
        for ei, ep in enumerate(EPOCHS):
            r = val.get((ep, p))
            if r is None:
                continue
            x = x0 + ei * w
            y = float(r["value"])
            ax.bar(x, y, width=w * 0.92, color=EPOCH_COLOUR[ep],
                   edgecolor="none", zorder=2)
            if r.get("lo") and r.get("hi"):
                ax.plot([x, x], [float(r["lo"]), float(r["hi"])], color="k",
                        lw=0.7, zorder=3)
            if r.get("mark"):
                ax.text(x, float(r["hi"] or y) + 0.02, r["mark"], ha="center",
                        va="bottom", fontsize=7, color=EPOCH_COLOUR[ep])
    ax.axhline(chance, ls="--", lw=0.7, color="0.4", zorder=1)
    ax.set_xticks([pi * (1 + gap) + 1.5 * w for pi in range(len(POS))])
    ax.set_xticklabels(TICK_SHORT)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel(f"accuracy (chance {chance:.2f})")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, out, stem)


RENDERERS = {
    "acc_by_position_cue":    lambda d, o: panel_acc_by_position(d, o, "cue_working",    "F2_acc_by_position_cue"),
    "acc_by_position_precue": lambda d, o: panel_acc_by_position(d, o, "precue_working", "F2_acc_by_position_precue"),
}


def main():
    ap = argparse.ArgumentParser(description="K08 2026 GRANT FIGURES ONLY")
    ap.add_argument("--data", type=pathlib.Path,
                    help="grant_figures/epoch/data directory")
    ap.add_argument("--out", type=pathlib.Path,
                    help="output directory (default: <data>/../../K08_2026_grant_panels)")
    ap.add_argument("--only", nargs="*", help="render only these panel keys")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    a = ap.parse_args()

    if a.list or not a.data:
        print("K08 2026 grant panel manifest\n")
        for fig, key, src, note in PANELS:
            flag = "RENDERED HERE" if key in RENDERERS else src.upper()
            print(f"  {fig}  {key:24s} {flag:14s} {note}")
        print("\n  csv    = vector, rendered by this script")
        print("  deck   = raster, extracted from Overview_2026-11-06.pptx")
        print("  server = file already on the server")
        print("  reuse  = panel already in the grant")
        return 0

    out = a.out or (a.data.parent.parent / "K08_2026_grant_panels")
    keys = a.only or list(RENDERERS)
    print(f"data: {a.data}\nout:  {out}\n")
    for k in keys:
        if k not in RENDERERS:
            print(f"  skip {k} (not rendered here; see manifest)")
            continue
        print(f"rendering {k}")
        RENDERERS[k](a.data, out)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
