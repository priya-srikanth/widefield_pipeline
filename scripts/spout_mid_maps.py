"""Per-animal per-position DIRECTION maps for far_ipsi and close_contra (contact-controlled, cam4 lick-aligned; Priya 10-21
item 4). spout_<pos> session maps = within-spout lateral slope (dF per deg) x 10 = dF per 10 deg of deviation TOWARD THE
MOUSE'S LEFT (= ipsilesional; red = more activity for licks deviating left, blue = for licks deviating right /
contralesional). Columns: far_ipsi pre / chronic / chronic-pre, close_contra pre / acute / acute-pre; rows = animals;
per-column shared colour limit. ipsi = image LEFT.
PER-POSITION DIRECTION, NOT TOWARD-MIDLINE (Priya 10-21 afternoon: "i don't know that i wanted the toward midline
analysis, just the per-position analysis"). The 10-21 toward-midline version multiplied ipsi spouts by -1; its figures
stay as spout_mid_maps_per_animal*.png for comparison, this one writes spout_dir_maps_per_animal*.png.
--z (Priya 10-21 "show me the new standardized z-scored maps"): each animal x spout map x (its PRE-stroke median
within-spout lateral SD) / (its PRE-stroke median within-spout cortex response SD) -- the map analogue of
within_spout_mixed.add_z (pre-stroke SD of activity per pre-stroke SD of deviation) -- plus a POOLED row (animals
weighted equally)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "C:/Users/SabatiniLab/Github/widefield_pipeline")
from wfield_local import analysis_kit as ak  # noqa: E402
from wfield_local import epochs  # noqa: E402

E = "M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding"
args = [x for x in sys.argv[1:] if x != "--z"]
Z = "--z" in sys.argv
folder = args[0] if args else "within_spout_deviation_lick_contact"
src = f"{E}/qc/{folder}/sessions"
labels = [x for x in ak.curated_labels() if x[-4:] >= "0806"]
zs = {}
if Z:
    import pandas as pd
    T = pd.read_csv(f"{E}/qc/{folder}/within_spout_deviation.csv")
    T = T[(T.variant == "mean") & (T.area == "cortex") & (T.epoch == "pre")]
    for (an, sp), g in T.groupby(["animal", "spouts"]):
        zs[(an, sp)] = float(g.lat_sd.median() / g.resp_sd.median())
acc = {}
shape = None
for lab in labels:
    try:
        z = np.load(f"{src}/{lab}.npz")
    except FileNotFoundError:
        continue
    shape = tuple(z["shape"])
    m = dict(zip(z["keys"], z["maps"]))
    for pos, sg in (("far_ipsi", 1.0), ("close_contra", 1.0)):     # direction: no side flip (+ = mouse's left)
        k = f"spout_{pos}"
        if k in m:
            img = np.full(shape[0] * shape[1], np.nan, np.float32)
            img[z["flat"]] = (m[k] * sg * zs[(lab.split("_")[0], pos)] if Z     # z: pre-SD activity per pre-SD deviation
                              else m[k] * sg * 10 * 1e3)                # x1e-3 dF per 10 deg toward the mouse's left
            acc.setdefault((lab.split("_")[0], pos, epochs.epoch_of(lab)), []).append(img.reshape(shape))
mean = {k: np.nanmean(v, 0) for k, v in acc.items()}
animals = sorted({k[0] for k in mean})
if Z:                                                           # POOLED, animals weighted equally
    for pos in ("far_ipsi", "close_contra"):
        for ep in ("pre", "acute", "subacute", "chronic"):
            v = [mean[(an, pos, ep)] for an in animals if (an, pos, ep) in mean]
            if v:
                mean[("POOLED", pos, ep)] = np.nanmean(v, 0)
                acc[("POOLED", pos, ep)] = v
    animals = animals + ["POOLED"]
cols = [("far_ipsi", "pre", None), ("far_ipsi", "chronic", None), ("far_ipsi", "chronic", "pre"),
        ("close_contra", "pre", None), ("close_contra", "acute", None), ("close_contra", "acute", "pre")]
fig, ax = plt.subplots(len(animals), len(cols), figsize=(2.3 * len(cols), 2.2 * len(animals)), squeeze=False)
for j, (pos, ep, ref) in enumerate(cols):
    imgs = []
    for an in animals:
        a = mean.get((an, pos, ep))
        if a is not None and ref:
            r = mean.get((an, pos, ref))
            a = a - r if r is not None else None
        imgs.append(a)
    lim = np.nanpercentile(np.abs(np.stack([i for i in imgs if i is not None])), 98)
    for i, (an, img) in enumerate(zip(animals, imgs)):
        a_ = ax[i, j]
        a_.axis("off")
        if img is None:
            continue
        a_.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
        n = len(acc.get((an, pos, ep), []))
        a_.set_title(f"{an} {pos} {ep}{' - ' + ref if ref else ''}\nn={n}, |lim| {lim:.2g}", fontsize=7)
unit = "z: pre-stroke SD of activity per pre-stroke SD of lick deviation" if Z else "x1e-3 dF / 10 deg"
fig.suptitle(f"Per-position DIRECTION slope, {unit}\nred = more activity for licks deviating toward the mouse's LEFT "
             f"(ipsilesional), blue = toward its RIGHT (contralesional)\n{folder}; "
             "ipsi = image left" + ("; POOLED = animals weighted equally (n = animals)" if Z else ""), fontsize=9)
fig.tight_layout()
out = f"{E}/figures/within_spout/{folder}/spout_dir_maps_per_animal{'_z' if Z else ''}.png"
fig.savefig(out, dpi=130)
print(out)
