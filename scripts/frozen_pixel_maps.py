"""PIXEL MAPS of the frozen-encoder decomposition from a SPATIAL-target run (`frozen_encoder --target basis | grid`;
Priya 10-22: "go ahead with the area map and then pixel maps").

Per session the run saved the EVOKED response (response window minus baseline; lick: 0 .. 1 s vs -0.3 .. -0.1 s from
the first post-cue lick, cue: 0.5 .. 1.5 s vs -0.5 .. 0 s) of the data and of each group's part, per spout x target.
Everything is linear, so per animal the epoch mean minus the pre mean is taken IN TARGET SPACE and reconstructed once:
basis -> pixels = W @ values (W = the animal's shared basis on the stat mask); grid -> each block painted with its
value. Then the mean over animals in pixel space (rule 8: animals first). Columns: data, full, each part, data - full.
dF/F x 1e3; ipsi (lesioned LEFT hemisphere) = image LEFT (orientation checked against the atlas names, as the area
maps); displayed at half resolution.

Usage: python scripts/frozen_pixel_maps.py --tag gate-attempted_v4basis [--variant joint] [--spout far_contra]
Output: <E>/frozen_encoder_<tag>/figures/pixel_maps_<variant>_<spout>_<epoch>_<align>.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

EPOCHS = ("acute", "subacute", "chronic")
SHOW = ("data", "full", "lick", "spout", "target", "exec_locked", "exec_within", "contact", "other", "timing",
        "residual")


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
    ap.add_argument("--tag", default="gate-attempted_v4basis")
    ap.add_argument("--variant", default="joint")
    ap.add_argument("--spout", default="far_contra")
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    sess = sorted(js for js in (out / a.variant / "sessions").glob("*.json") if js.stem[-4:] >= "0801")
    fd = out / "figures"
    assert_writable(fd)
    fd.mkdir(parents=True, exist_ok=True)
    _V, _ft, _Ub, flat, _parea, _names, shape, pstat = ce.load_imaging(sess[0].stem)
    H, W = (int(x) for x in shape)
    pix = flat[pstat]
    img, nm = area_image(sess[0].stem)                     # also checks ipsi = image left
    ol = outline(img)[::2, ::2]
    # per animal x epoch: mean evoked in TARGET space (parts x targets), per alignment
    acc: dict = {}
    parts_of: dict = {}
    for js in sess:
        m = json.loads(js.read_text(encoding="utf-8"))
        if a.spout not in m["pos_order"]:
            continue
        z = np.load(js.with_suffix(".npz"))
        pi = m["pos_order"].index(a.spout)
        parts_of[m["animal"]] = m["evoked_parts"]
        for al in ("lick", "cue"):
            ev = z[f"evoked_{al}"][:, pi, :-1].astype(np.float64)           # drop the cortex-mean target
            if np.isfinite(ev).all():
                acc.setdefault((m["animal"], m["epoch"], al), []).append(ev)
    spatial = {an: pickle.loads((out / "fit" / f"spatial_{an}.pkl").read_bytes()) for an in parts_of}

    def to_pixels(an, v):                                 # (targets,) -> (H, W) image, NaN off the stat mask
        sp = spatial[an]
        g = np.full(H * W, np.nan)
        if sp["kind"] == "basis":
            g[pix] = np.asarray(sp["P"], np.float64) @ v
        else:
            P = sp["P"].tocsc()
            blk_of_pix = np.asarray(P.argmax(1)).ravel()                     # each stat pixel's block column
            has = np.asarray(P.sum(1)).ravel() > 0
            g[pix[has]] = v[blk_of_pix[has]]
        return g.reshape(H, W)
    for al in ("lick", "cue"):
        for ep in EPOCHS:
            maps = {}
            for an in sorted(parts_of):
                if (an, "pre", al) not in acc or (an, ep, al) not in acc:
                    continue
                dlt = np.mean(acc[(an, ep, al)], 0) - np.mean(acc[(an, "pre", al)], 0)   # (parts, targets)
                names = parts_of[an]
                vals = {p: dlt[names.index(p)] for p in names}
                vals["residual"] = vals["data"] - vals["full"]
                maps[an] = {p: to_pixels(an, v * 1e3)[::2, ::2] for p, v in vals.items() if p in SHOW}
            if not maps:
                continue
            show = [p for p in SHOW if all(p in mm for mm in maps.values())]
            maps["mean"] = {p: np.nanmean([maps[an][p] for an in maps], 0) for p in show}
            lim = np.nanpercentile(np.abs(np.concatenate([maps[an][p].ravel() for an in maps for p in show])), 98)
            fig, axs = plt.subplots(len(maps), len(show), figsize=(1.75 * len(show), 1.75 * len(maps) + 0.9),
                                    squeeze=False)
            for i, an in enumerate(maps):
                for j, p in enumerate(show):
                    ax = axs[i][j]
                    ax.imshow(maps[an][p], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(p, fontsize=8)
                    if j == 0:
                        n_ep = len(acc.get((an, ep, al), [])) if an != "mean" else len(maps) - 1
                        ax.set_ylabel(f"{an}\n(n={n_ep})", fontsize=8)
            sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-lim, lim))
            fig.colorbar(sm, ax=axs, fraction=0.012, pad=0.01, label="change vs pre, dF/F x 1e3")
            kind = next(iter(spatial.values()))["kind"]
            fig.suptitle(f"Frozen encoder ({a.variant}, {kind} targets) PIXEL maps: {a.spout}, {ep} minus pre, "
                         f"{al}-aligned evoked response, per animal then mean. residual = data - full. ipsi (lesion) = "
                         f"image LEFT. PROVISIONAL", fontsize=8)
            fig.savefig(fd / f"pixel_maps_{a.variant}_{a.spout}_{ep}_{al}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    print(f"-> {fd} (pixel_maps_{a.variant}_{a.spout}_*.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
