"""Per-animal toward-midline maps for far_ipsi and close_contra (contact-controlled, cam4 lick-aligned; Priya 10-21 item 4).
spout_<pos> session maps = within-spout lateral slope (dF per deg); x side sign (contra +1, ipsi -1) x 10 = dF per 10 deg
TOWARD THE MIDLINE (red = more activity for medial licks). Columns: far_ipsi pre / chronic / chronic-pre, close_contra
pre / acute / acute-pre; rows = animals; per-column shared colour limit. ipsi = image LEFT."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "C:/Users/SabatiniLab/Github/widefield_pipeline")
from wfield_local import analysis_kit as ak  # noqa: E402
from wfield_local import epochs  # noqa: E402

E = "M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding"
folder = sys.argv[1] if len(sys.argv) > 1 else "within_spout_deviation_lick_contact"
src = f"{E}/qc/{folder}/sessions"
labels = [x for x in ak.curated_labels() if x[-4:] >= "0806"]
acc = {}
shape = None
for lab in labels:
    try:
        z = np.load(f"{src}/{lab}.npz")
    except FileNotFoundError:
        continue
    shape = tuple(z["shape"])
    m = dict(zip(z["keys"], z["maps"]))
    for pos, sg in (("far_ipsi", -1.0), ("close_contra", 1.0)):
        k = f"spout_{pos}"
        if k in m:
            img = np.full(shape[0] * shape[1], np.nan, np.float32)
            img[z["flat"]] = m[k] * sg * 10 * 1e3              # x1e-3 dF per 10 deg toward the midline
            acc.setdefault((lab[:4], pos, epochs.epoch_of(lab)), []).append(img.reshape(shape))
mean = {k: np.nanmean(v, 0) for k, v in acc.items()}
animals = sorted({k[0] for k in mean})
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
fig.suptitle(f"Toward-midline slope per spout, x1e-3 dF / 10 deg (red = more activity for MEDIAL licks); {folder}; "
             "ipsi = image left", fontsize=9)
fig.tight_layout()
out = f"{E}/figures/within_spout/{folder}/spout_mid_maps_per_animal.png"
fig.savefig(out, dpi=130)
print(out)
