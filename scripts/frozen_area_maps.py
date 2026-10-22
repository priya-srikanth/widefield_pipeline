"""AREA MAPS of the frozen-encoder decomposition (Priya 10-22: "are there maps that go along with this most recent
analysis?" -> "go ahead with the area map and then pixel maps"). The frozen encoder predicts Allen-area activity, so
these are per-area values painted on the atlas -- the coarse version; pixel maps need the grid target (next).

Per session (`<E>/frozen_encoder_<tag>/<variant>/sessions/*.npz|json`, lick- or cue-aligned PETHs of data and each
model part, per spout x area): evoked = mean over the response window minus the baseline window, exactly
`scripts/lp_encoder_decompose.py` (lick: 0 .. 1 s vs -0.3 .. -0.1 s from the first post-cue lick; cue: 0.5 .. 1.5 s
vs -0.5 .. 0 s). Change = mean over the epoch's sessions minus mean over the pre sessions, PER ANIMAL (rule 8), then
the mean over animals. Columns: data, full prediction, each part, data - full (what the frozen pre-stroke mapping does
not predict from the behaviour actually made). Units dF/F x 1e3. ipsi (lesioned, LEFT hemisphere) = image LEFT
(checked against the atlas area names). Areas absent in an animal are blank.

Usage: python scripts/frozen_area_maps.py [--tag gate-attempted_v4] [--variant joint] [--spout far_contra]
Output: <E>/frozen_encoder_<tag>/figures/area_maps_<variant>_<spout>_<epoch>_<align>.png + area_maps_<variant>.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = {"lick": ((0.0, 1.0), (-0.3, -0.1)), "cue": ((0.5, 1.5), (-0.5, 0.0))}
EPOCHS = ("acute", "subacute", "chronic")
SHOW = ("data", "full", "lick", "spout", "target", "exec_locked", "exec_within", "contact", "other", "timing",
        "residual")


def session_rows(js: Path, align: str) -> list[dict]:
    m = json.loads(js.read_text(encoding="utf-8"))
    z = np.load(js.with_suffix(".npz"))
    peth = z["peth_lick" if align == "lick" else "peth"]
    off = z["peth_off_s"]
    (r0, r1), (b0, b1) = WINDOWS[align]
    w, b = (off >= r0) & (off < r1), (off >= b0) & (off <= b1)
    parts = m["peth_parts"]
    rows = []
    for pi, pos in enumerate(m["pos_order"]):
        ev = np.nanmean(peth[:, pi][:, :, w], -1) - np.nanmean(peth[:, pi][:, :, b], -1)   # (parts, areas)
        for ai, area in enumerate(m["areas"]):
            r = {"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "pos": pos, "area": area}
            # the model part "pos" (position regressors) must not overwrite the spout label (as lp_encoder_decompose)
            r.update({("pos_part" if p == "pos" else p): float(ev[ci, ai]) for ci, p in enumerate(parts)})
            if "data" in r and "full" in r:
                r["residual"] = r["data"] - r["full"]
            rows.append(r)
    return rows


def area_image(label: str):
    """(H, W) image of area NAMES' indices + the name list, from one session's atlas (the Allen grid is shared)."""
    from wfield_local import combined_encoding as ce
    _V, _ft, _Ub, flat, parea, names, shape, _pstat = ce.load_imaging(label)
    H, W = (int(x) for x in shape)
    img = np.full(H * W, -1)
    keys = sorted(names)
    lut = {k: i for i, k in enumerate(keys)}
    img[flat] = [lut.get(int(a), -1) for a in parea]
    img = img.reshape(H, W)
    nm = [names[k] for k in keys]
    # ORIENTATION CHECK (ipsi = lesioned LEFT hemisphere = image LEFT)
    cols = {h: np.mean([np.nonzero(img == i)[1].mean() for i, n in enumerate(nm) if n.endswith(h) and (img == i).any()])
            for h in ("_left", "_right")}
    if not cols["_left"] < W / 2 < cols["_right"]:
        raise RuntimeError(f"atlas orientation: mean column of _left areas {cols['_left']:.0f}, _right "
                           f"{cols['_right']:.0f}, width {W} -- left hemisphere is not on the image left")
    return img, nm


def paint(img, nm, values: dict) -> np.ndarray:
    out = np.full(img.shape, np.nan)
    for i, n in enumerate(nm):
        if n in values and np.isfinite(values[n]):
            out[img == i] = values[n]
    return out


def outline(img) -> np.ndarray:
    b = np.zeros(img.shape, bool)
    b[:-1] |= img[:-1] != img[1:]
    b[:, :-1] |= img[:, :-1] != img[:, 1:]
    return np.where(b & (img >= 0) | b & (np.roll(img, 1, 0) >= 0), 1.0, np.nan)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v4")
    ap.add_argument("--variant", default="joint")
    ap.add_argument("--spout", default="far_contra")
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    sess = sorted((out / a.variant / "sessions").glob("*.json"))
    sess = [js for js in sess if js.stem[-4:] >= "0801"]
    fd = out / "figures"
    assert_writable(fd)
    fd.mkdir(parents=True, exist_ok=True)
    img, nm = area_image(sess[0].stem)
    ol = outline(img)
    allrows = []
    for align in ("lick", "cue"):
        T = pd.DataFrame([r for js in sess for r in session_rows(js, align)])
        T["align"] = align
        allrows.append(T)
        T = T[T.pos == a.spout]
        parts = [p for p in SHOW if p in T.columns]
        T = T.assign(**{p: T[p] * 1e3 for p in parts})          # dF/F x 1e3, as the decomposition tables
        A = T.groupby(["animal", "epoch", "area"])[parts].mean()
        animals = sorted(T.animal.unique())
        for ep in EPOCHS:
            D = {}
            for an in animals:
                if (an, "pre") in A.index.droplevel(2) and (an, ep) in A.index.droplevel(2):
                    D[an] = A.loc[(an, ep)] - A.loc[(an, "pre")]
            if not D:
                continue
            mean = pd.concat(D.values()).groupby(level=0).mean()
            rowsets = [*D.items(), ("mean", mean)]
            lim = np.nanpercentile(np.abs(np.concatenate([d[parts].to_numpy().ravel() for _, d in rowsets])), 98)
            fig, axs = plt.subplots(len(rowsets), len(parts), figsize=(1.75 * len(parts), 1.75 * len(rowsets) + 0.9),
                                    squeeze=False)
            for i, (an, d) in enumerate(rowsets):
                for j, p in enumerate(parts):
                    ax = axs[i][j]
                    ax.imshow(paint(img, nm, d[p].to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim,
                              interpolation="nearest")
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(p, fontsize=8)
                    if j == 0:
                        n_ep = T[(T.animal == an) & (T.epoch == ep)].label.nunique() if an != "mean" else len(D)
                        ax.set_ylabel(f"{an}\n(n={n_ep})", fontsize=8)
            sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-lim, lim))
            fig.colorbar(sm, ax=axs, fraction=0.012, pad=0.01, label="change vs pre, dF/F x 1e3")
            fig.suptitle(f"Frozen encoder ({a.variant}) AREA maps: {a.spout}, {ep} minus pre, {align}-aligned evoked "
                         f"response, per animal then mean. residual = data - full. ipsi (lesion) = image LEFT. "
                         f"Coarse (42 Allen areas); PROVISIONAL", fontsize=8)
            fig.savefig(fd / f"area_maps_{a.variant}_{a.spout}_{ep}_{align}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    pd.concat(allrows).to_csv(fd / f"area_maps_{a.variant}.csv", index=False, float_format="%.5g")
    print(f"-> {fd} (area_maps_{a.variant}_{a.spout}_*.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
