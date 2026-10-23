"""PRE-STROKE PER-SPOUT pixel maps of the frozen-encoder decomposition from a basis run (Priya 10-23: "why is the far
contra map not showing L hemisphere activation ...").

Signed evoked response (window minus baseline, as `frozen_pixel_maps`) of each spout's pre-stroke sessions, per model
part: data / spout (= target + spout-locked lick, the whole spout-specific response) / target / exec_locked / lick.
Rows = spouts (contra = the mouse's right), one figure per animal + the animal mean (rule 8). ``--contrast`` subtracts
the six-spout mean per part, removing the large response shared by all spouts: that is where laterality shows (contra
spouts -> left hemisphere, ipsi -> right, in all four animals) and where the target / exec_locked SPLIT is seen to be
unstable (close_mid: +13 vs -13, cancelling) while their SUM (spout) tracks the data -- target and the spout-locked lick
are the same variable on on-target licks; only strays separate them. dF/F x 1e3; ipsi (lesion LEFT) = image LEFT.

Usage: python scripts/frozen_spout_maps.py --tag gate-attempted_v6basis [--variant joint] [--align lick] [--contrast]
Output: <E>/frozen_encoder_<tag>/figures/pre_spout_{maps,contrast}_<variant>_<align>_<animal|mean>.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SPOUTS = ("far_contra", "close_contra", "close_mid", "far_mid", "close_ipsi", "far_ipsi")
PARTS = ("data", "spout", "target", "exec_locked", "lick")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frozen_area_maps import area_image, outline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v6basis")
    ap.add_argument("--variant", default="joint")
    ap.add_argument("--align", default="lick", choices=("lick", "cue"))
    ap.add_argument("--contrast", action="store_true", help="subtract the six-spout mean per part")
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    sess = sorted(js for js in (out / a.variant / "sessions").glob("*.json") if js.stem[-4:] >= "0801")
    fd = out / "figures"
    assert_writable(fd)
    fd.mkdir(parents=True, exist_ok=True)
    _V, _ft, _Ub, flat, _pa, _nm, shape, pstat = ce.load_imaging(sess[0].stem)
    H, W = (int(x) for x in shape)
    pix = flat[pstat]
    img, _ = area_image(sess[0].stem)                      # also checks ipsi = image left
    ol = outline(img)[::2, ::2]
    acc: dict = {}
    for js in sess:
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["epoch"] != "pre":
            continue
        z = np.load(js.with_suffix(".npz"))[f"evoked_{a.align}"][:, :, :-1].astype(np.float64)   # drop cortex mean
        for p in SPOUTS:
            if p not in m["pos_order"]:
                continue
            v = z[:, m["pos_order"].index(p)]
            if np.isfinite(v).all():
                acc.setdefault((m["animal"], p), []).append({q: v[m["evoked_parts"].index(q)] for q in PARTS})
    animals = sorted({an for an, _ in acc})
    maps: dict = {}
    for an in animals:
        Wb = np.asarray(pickle.loads((out / "fit" / f"spatial_{an}.pkl").read_bytes())["P"], np.float64)
        for p in SPOUTS:
            for q in PARTS:
                g = np.full(H * W, np.nan)
                g[pix] = Wb @ np.mean([r[q] for r in acc[(an, p)]], 0) * 1e3
                maps[(an, p, q)] = g.reshape(H, W)[::2, ::2]
        if a.contrast:
            for q in PARTS:
                mu = np.nanmean([maps[(an, p, q)] for p in SPOUTS], 0)
                for p in SPOUTS:
                    maps[(an, p, q)] = maps[(an, p, q)] - mu
    kind = "contrast" if a.contrast else "maps"
    for an in [*animals, "mean"]:
        M = {(p, q): (maps[(an, p, q)] if an != "mean" else np.nanmean([maps[(b, p, q)] for b in animals], 0))
             for p in SPOUTS for q in PARTS}
        lim = np.nanpercentile(np.abs(np.concatenate([x.ravel() for x in M.values()])), 99)
        fig, axs = plt.subplots(len(SPOUTS), len(PARTS), figsize=(1.8 * len(PARTS), 1.7 * len(SPOUTS) + 0.8),
                                squeeze=False)
        for i, p in enumerate(SPOUTS):
            for j, q in enumerate(PARTS):
                ax = axs[i][j]
                ax.imshow(M[(p, q)], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(q, fontsize=8)
                if j == 0:
                    ax.set_ylabel(p, fontsize=8)
        what = " MINUS the 6-spout mean" if a.contrast else ""
        fig.suptitle(f"{an}: PRE-STROKE evoked ({a.align}-aligned) per spout{what}, by model part ({a.variant}, basis "
                     f"targets); dF/F x 1e3, scale +/-{lim:.0f}. ipsi (lesion) = image LEFT; contra spouts = mouse's "
                     "right", fontsize=8)
        fig.savefig(fd / f"pre_spout_{kind}_{a.variant}_{a.align}_{an}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"-> {fd} (pre_spout_{kind}_*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
