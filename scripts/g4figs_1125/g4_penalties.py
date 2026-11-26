"""Per-session ridge penalties the G4 per-session fits selected (read-only from the bundle's session JSONs).

Each session's full model picks one penalty per group (task / lick / orofacial / uninstructed) by a 2-pass coordinate
search over ALPHA_GRID (1..1e6) on held-out R2 (combined_encoding.py ~l.1994); every non-shuffle reduced model
(drop / alone) re-searches the groups it contains. Output: per-session long table + per animal x epoch summary
(median, and the share of sessions at each grid value), for the full model of each tag, and for the drop:<group> models
the movement readouts rest on. Pre = August-only (config.early_pre dropped).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wfield_local import config
B = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/o2/enc_fourcam3d_20261124/out")
V36 = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk")
OUT = Path(sys.argv[1])
SRC = {"G4": {"main": B / "main/sessions", "fr_A": B / "fr_A/sessions", "bins_w1": B / "bins_w1/sessions"},
       "v36": {"main": V36 / "movement_first_exec-polar_licks-combined/sessions",
               "fr_A": V36 / "movement_first_fr-master_pre_v1_frg-A_subset-fr_exec-polar_licks-combined/sessions",
               "bins_w1": V36 / "movement_first_subset-nokin_exec-polar_licks-combined_bins/sessions"}}
rows = []
for lin, d in SRC.items():
    for tag, sd in d.items():
        for jf in sorted(sd.glob("PS*_*.json")):
            m = json.loads(jf.read_text(encoding="utf-8"))
            if m["epoch"] == "pre" and config.early_pre(m["date"]):
                continue
            for mod, al in (m.get("model_alphas") or {"full": m["alphas"]}).items():
                if not (mod == "full" or mod.startswith(("drop:", "fr:full", "nokin:"))):
                    continue
                for g, a in al.items():
                    if g in ("video", "frpts") or (g == "fr" and "fr" not in mod):
                        continue
                    rows.append(dict(lineage=lin, tag=tag, label=m["label"], animal=m["animal"], epoch=m["epoch"],
                                     model=mod, group=g, alpha=a))
T = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
T.to_csv(OUT / "penalties_by_session.csv.gz", index=False)
T["log10a"] = np.log10(T.alpha)
S = (T.groupby(["lineage", "tag", "model", "group", "animal", "epoch"]).agg(
    n=("alpha", "size"), median_log10=("log10a", "median"), q25=("log10a", lambda x: x.quantile(.25)),
    q75=("log10a", lambda x: x.quantile(.75)), modal=("alpha", lambda x: x.mode().iloc[0])).reset_index())
S.to_csv(OUT / "penalties_by_animal_epoch.csv", index=False)
# epoch shift of the median log10 penalty vs the animal's own pre (per animal), full model
F = S[S.model == "full"].pivot_table(index=["lineage", "tag", "group", "animal"], columns="epoch", values="median_log10")
for e in ("acute", "subacute", "chronic"):
    if e in F:
        F[f"d_{e}"] = F[e] - F["pre"]
F.to_csv(OUT / "penalties_full_shift_vs_pre.csv")
pd.set_option("display.width", 250)
print(F.round(2).to_string())
