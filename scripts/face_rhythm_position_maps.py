"""Face movement maps PER SPOUT POSITION, epoch vs pre (Priya 10-24: "show the spatial speed map per position to look
at asymmetry").

`proj/` stores, per session and spout position, each tracked mesh point's RMS speed in the pre-cue (cue -2..0 s) and
response (cue 0..+4 s) windows of engaged trials (``speed_precue`` / ``speed_response``, (6, P), with bin counts
``speed_n_bins``). Per animal x epoch x position: the bin-weighted mean of speed^2 over sessions, rooted; mapped as
log2(epoch / pre) per point on the master face, and the contra / ipsi asymmetry of the change.

    python scripts/face_rhythm_position_maps.py [--set master_pre_v1] [--window response|precue]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/position_maps/``: position_maps_<window>_<animal>.png (rows = positions,
cols = acute / subacute / chronic), position_maps_<window>_mean.png (mean of the animal maps), and
position_maps_<window>.csv (per animal x epoch x position: contra / ipsi mean log2 change, its asymmetry).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

POSITIONS = frc.POSITIONS                      # animal's L (ipsilesional) -> R (contralesional)
EPOCHS = frc.EPOCHS


def position_speed(fdir: Path, window: str) -> dict:
    """{(animal, epoch, position): (P,) RMS speed} -- bin-weighted mean of speed^2 over the epoch's sessions."""
    acc = {}
    for t in frc.sessions(fdir, "proj").itertuples():
        with np.load(t.file) as z:
            if "speed_n_bins" not in z.files:
                continue
            sp, n = z[f"speed_{window}"], z["speed_n_bins"]
            nb = n[1::2] if window == "precue" else n[2::2]
            pos = [str(x) for x in np.unique(z["position"])]                      # the stored order (`window_labels`)
        for k, p in enumerate(pos):
            if nb[k] > 0 and np.isfinite(sp[k]).all():
                s0, n0 = acc.get((t.animal, t.epoch, p), (0.0, 0))
                acc[(t.animal, t.epoch, p)] = (s0 + sp[k] ** 2 * nb[k], n0 + nb[k])
    return {k: np.sqrt(s / n) for k, (s, n) in acc.items() if n}


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--window", default="response", choices=("response", "precue"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "position_maps"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    M = np.load(fdir / "master.npz")
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    contra = home[:, 0] <= mid                                       # cam4: image LEFT = contralesional
    V = position_speed(fdir, a.window)
    animals = sorted({k[0] for k in V})
    rows, maps = [], {}
    for an in animals:
        for p in POSITIONS:
            pre = V.get((an, "pre", p))
            if pre is None:
                continue
            for ep in EPOCHS[1:]:
                v = V.get((an, ep, p))
                if v is None:
                    continue
                m = np.log2(np.maximum(v, 1e-9) / np.maximum(pre, 1e-9))
                maps[(an, ep, p)] = m
                c, i = float(m[contra].mean()), float(m[~contra].mean())
                rows.append({"animal": an, "epoch": ep, "position": p, "log2_change_contra": c,
                             "log2_change_ipsi": i, "contra_minus_ipsi": c - i,
                             "speed_contra_vs_pre": float(v[contra].mean() / pre[contra].mean()),
                             "speed_ipsi_vs_pre": float(v[~contra].mean() / pre[~contra].mean())})
    T = pd.DataFrame(rows)
    T.to_csv(figure_layout.sidecar_for(out, f"position_maps_{a.window}", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(f"{a.window}: contra minus ipsi of the log2(epoch / pre) speed change, per animal (rows = position)")
    print(T.pivot_table(index=["animal", "position"], columns="epoch", values="contra_minus_ipsi", observed=True)
          .reindex(columns=list(EPOCHS[1:])).round(2).to_string())

    def draw(get, title, fn):
        with frf.style():
            fig, axs = frf.subplots(len(POSITIONS), 3, figsize=(10.5, 3.1 * len(POSITIONS)), squeeze=False)
            sc = None
            for r_i, p in enumerate(POSITIONS):
                for c_i, ep in enumerate(EPOCHS[1:]):
                    ax = axs[r_i, c_i]
                    ax.axis("off")
                    m = get(ep, p)
                    if m is None:
                        continue
                    sc = ax.scatter(home[:, 0], home[:, 1], c=m, s=4, cmap="RdBu_r", vmin=-1, vmax=1)
                    ax.axvline(mid, color="k", lw=0.4)
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
                    ax.set_title(f"{p} -- {ep}", fontsize=9)
            if sc is not None:
                fig.colorbar(sc, ax=axs, shrink=0.3, label="log2(epoch / pre)")
            fig.suptitle(title)
            frf.save(fig, out / fn, f"""
Per mesh point on the master face (image LEFT = CONTRALESIONAL on cam4, all lesions left; vertical line = the face
midline, x = {mid:.0f}): log2(epoch / pre) of the point's RMS speed in the {a.window.upper()} window
({'cue -2..0 s' if a.window == 'precue' else 'cue 0..+4 s'}) of engaged trials at ONE spout position (rows; the animal's
L = ipsilesional, R = contralesional spouts), sessions pooled bin-weighted within the epoch. Red = more movement than
pre, blue = less; clipped at +-1 (2x). Contra / ipsi means of the change: data/position_maps_{a.window}.csv.""")

    for an in animals:
        draw(lambda ep, p, an=an: maps.get((an, ep, p)), f"{fdir.name}: {an}, {a.window} speed per spout position vs pre",
             f"position_maps_{a.window}_{an}.png")

    def mean_map(ep, p):
        ms = [maps[(an, ep, p)] for an in animals if (an, ep, p) in maps]
        return np.mean(ms, 0) if ms else None
    draw(mean_map, f"{fdir.name}: mean of {len(animals)} animals, {a.window} speed per spout position vs pre",
         f"position_maps_{a.window}_mean.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
