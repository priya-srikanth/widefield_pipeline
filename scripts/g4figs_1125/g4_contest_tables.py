"""Per-animal tables (rule 8) for Figs 1.6 / 1.13-1.15: cortex unique dR2 of target bins and executed bins and the
contest (target - executed), per animal x epoch (mean over sessions), G4 vs v36, for widths 0.5 / 1 / 2."""
import sys
from pathlib import Path
import pandas as pd
G4 = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/figures_working/g4figs_ccB_20261125/imaging_encoding_view/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk")
V36 = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk")
BINS = "movement_first_subset-nokin_exec-polar_licks-combined_bins"
EP = ["pre", "acute", "subacute", "chronic"]
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
rows = []
for lin, R in (("G4", G4), ("v36", V36)):
    for f, bw in (("_bw0.5", 0.5), ("", 1), ("_bw2", 2)):
        p = R / f"{BINS}{f}" / "tables" / "region_measures.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); c = d[d.pos == "cortex"]
        for w in ("lick", "cue"):
            c2 = c.assign(t=c[f"nokin:u:tgtbin|{w}"], x=c[f"nokin:u:exbin|{w}"])
            c2 = c2.assign(d=c2.t - c2.x)
            A = c2.groupby(["animal", "epoch"])[["t", "x", "d"]].mean()
            for (an, ep), r in A.iterrows():
                rows.append(dict(lineage=lin, width=bw, window=w, animal=an, epoch=ep, target=r.t, executed=r.x, contest=r.d,
                                 n=int((c2.animal == an).mul(c2.epoch == ep).sum())))
T = pd.DataFrame(rows)
T.to_csv(OUT / "contest_per_animal.csv", index=False, float_format="%.5f")
pd.set_option("display.width", 220)
for (lin, bw, w), g in T.groupby(["lineage", "width", "window"]):
    P = g.pivot_table(index="animal", columns="epoch", values="contest")[EP]
    D = P[EP[1:]].sub(P["pre"], axis=0)
    print(f"\n{lin} width {bw} {w}: contest per animal (x1e3), then change vs own pre")
    print((P * 1e3).round(2).to_string()); print((D * 1e3).round(2).to_string())
