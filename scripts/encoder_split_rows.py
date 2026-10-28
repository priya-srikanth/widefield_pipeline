"""ALM / tjM1 / MOs_rest / MOp_rest ROWS for the encoder and FaceRhythm AREA figures (Priya 10-27: "add ALM / tjM1 /
MOs-rest / MOp-rest area rows to the encoder and FaceRhythm area figures"). Plan + costing:
docs/status/ALM_TJM1_ROWS_PLAN_20261027.md.

ONE ROI DEFINITION: `iti_licks.roi_masks` (ROI v8, DECISIONS 10-27 "ROI MASK FINAL": ALM = anterolateral 40 % of MOs,
tjM1 = anterolateral 50 % of MOp), RESTRICTED TO THE ENCODER'S 16-px STAT MASK -- the mask every existing encoder area
value is computed on -- so the parts tile the existing rows exactly: ALM + MOs_rest = the encoder's MOs and tjM1 +
MOp_rest = its MOp, per hemisphere (SSE / SST add; evoked means add with pixel-share weights). The v8 thin lateral rim
and ALM front line (~11 % of the v8 ALM / tjM1 pixels) lie outside the stat mask and are therefore not in these rows.
Image side: `_left` = ipsilesional = image LEFT; `_right` = contralesional.

NO REFIT -- two per-pixel quantities already on disk are re-aggregated over the new masks:
  BLOCK  per-session encoder (`combined_encoding`): held-out SSE and SST summed over 2 x 2-px blocks per model x window
         (npz `maps`, `flat` = block ids). ROI SSE = sum over blocks of (ROI share of the block's brain pixels) x block
         SSE (exact for interior blocks; ~3-4 % of an ROI's pixels sit in partial blocks). Checked against the exact
         per-pixel region R2 (<label>_regions.csv), 42 areas x 3 models x 2 windows x 2 sessions: max |error| 0.0008,
         median 0.00002 (MOs / MOp / SSp-m <= 0.0001).
  BASIS  frozen encoder, basis-target run (`frozen_encoder_<tag>_basis`): evoked response (response minus baseline
         window) per spout of the data and of each model part, in the animal's pre-stroke spatial basis (stat pixels x
         K). ROI value = evoked @ (basis' ROI-mean weights). Same design (and for the noFR frame the same alphas) as the
         area-target run, so it is the area run's value on the basis-projected data. Checked against the area run
         (specificity_control sessions.csv, 7 areas x 2 windows x 4 sessions, one per animal): r = 1.000, residual
         max |difference| 0.15 x1e3 dF/F (median 0.02 noFR, 0.05 FR-A; FR-A's basis run picked lick alpha 1e4 vs 1e3).
Not covered (only area-mean traces were saved -> a refit is needed; see the plan doc): frozen Musall R2 (claims figures),
per-session trial / PETH R2 (`fr_trial_peth`), Layer 2 (`allsession_encoder`).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

#: the new rows, in figure order: per parent (MOs, then MOp) the sub-ROI then the rest, contra (right) before ipsi (left)
SPLIT_AREAS = [f"{n}_{h}" for src, dst in il.SPLIT_PAIRS for n in (dst, f"{src}_rest") for h in ("right", "left")]
#: the parents the rows tile, for the additivity checks
PARENT = {f"{n}_{h}": f"{src}_{h}" for src, dst in il.SPLIT_PAIRS for n in (dst, f"{src}_rest") for h in il.HEMIS}


def side_label(area: str) -> str:
    """'ALM_right' -> 'ALM contra', 'MOs_rest_left' -> 'MOs_rest ipsi' (lesion left: right = contralesional)."""
    return area.replace("_right", " contra").replace("_left", " ipsi")


@lru_cache(maxsize=1)
def grid() -> tuple:
    """(flat, parea, names, shape, pstat) of the shared Allen-aligned grid. The brain mask, area atlas and area names are
    byte-identical in all 114 August+ sessions (sha1, 10-27), so one session's imaging load defines it."""
    from wfield_local import combined_encoding as ce
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(il.run_labels()[0])
    return flat, parea, names, tuple(int(x) for x in shape), pstat


@lru_cache(maxsize=1)
def masks() -> dict:
    """{area: boolean over brain pixels (`grid` flat order)} for SPLIT_AREAS and their encoder parents (MOs / MOp per
    hemisphere on the stat mask, exactly the encoder's region masks)."""
    flat, parea, names, shape, pstat = grid()
    roi = il.roi_masks(parea, pstat, names, flat, shape)
    code = {v: k for k, v in names.items()}
    out = {}
    for src, dst in il.SPLIT_PAIRS:
        for h in il.HEMIS:
            par = (parea == code[f"{src}_{h}"]) & pstat
            sub = roi[f"{dst}_{h}"] & par                       # v8 ROI on the encoder's stat mask
            out[f"{src}_{h}"], out[f"{dst}_{h}"], out[f"{src}_rest_{h}"] = par, sub, par & ~sub
    return out


def pixel_share() -> dict:
    """{split area: its share of its parent's pixels} (for pixel-weighted checks / labels)."""
    m = masks()
    return {a: float(m[a].sum() / m[PARENT[a]].sum()) for a in SPLIT_AREAS}


# --------------------------------------------------------------------------------------------------------------- BLOCK
def block_weights(block_ids: np.ndarray, areas=SPLIT_AREAS) -> np.ndarray:
    """(len(areas), n_blocks): each block's share of brain pixels inside each area, on the per-session encoder's 2 x 2
    grid (block id = (row // 2) * ceil(W / 2) + col // 2, as `combined_encoding`; ``block_ids`` = its saved `flat`)."""
    flat, _pa, _nm, shape, _ps = grid()
    rr, cc = np.unravel_index(flat, shape)
    ublk, blk = np.unique((rr // 2) * ((shape[1] + 1) // 2) + (cc // 2), return_inverse=True)
    if not np.array_equal(ublk, np.asarray(block_ids)):
        raise ValueError("block grid of the encoder output differs from the brain-mask grid")
    n = np.bincount(blk, minlength=len(ublk)).astype(float)
    m = masks()
    return np.stack([np.bincount(blk, weights=m[a].astype(float), minlength=len(ublk)) / n for a in areas])


def block_r2(npz, models, windows, Wb: np.ndarray, areas=SPLIT_AREAS) -> dict:
    """{(area, window, model): held-out R2} = 1 - SSE / SST summed over each area's blocks (weights ``Wb`` from
    `block_weights`), from one session's encoder npz (`maps` rows 'model|window', 'TOTAL|window' = SST)."""
    k = {str(x): i for i, x in enumerate(npz["map_keys"])}
    M = npz["maps"]
    out = {}
    for w in windows:
        sst = Wb @ M[k[f"TOTAL|{w}"]].astype(np.float64)
        for mod in models:
            sse = Wb @ M[k[f"{mod}|{w}"]].astype(np.float64)
            for i, a in enumerate(areas):
                out[(a, w, mod)] = float(1.0 - sse[i] / sst[i])
    return out


# --------------------------------------------------------------------------------------------------------------- BASIS
@lru_cache(maxsize=8)
def basis_weights(basis_run: str, animal: str, areas=tuple(SPLIT_AREAS)) -> np.ndarray:
    """(K, len(areas)): ROI-mean weights in the animal's frozen basis (`<basis_run>/fit/spatial_<animal>.pkl`, P = stat
    pixels x K, orthonormal columns), for evoked arrays whose last target is the cortex mean."""
    import pickle
    P = np.asarray(pickle.loads((Path(basis_run) / "fit" / f"spatial_{animal}.pkl").read_bytes())["P"], np.float64)
    pstat = grid()[4]
    m = masks()
    w = np.stack([m[a][pstat] / m[a][pstat].sum() for a in areas], 1)          # (stat pixels, areas)
    return P.T @ w


def basis_evoked(basis_run: str, variant: str, label: str, animal: str) -> tuple:
    """One session of a frozen basis run -> (parts, pos_order, {window: (parts, spouts, len(SPLIT_AREAS)) evoked}),
    the spatial-target counterpart of the area run's evoked PETH means (`frozen_encoder._predict_spatial`)."""
    import json
    sess = Path(basis_run) / variant / "sessions"
    meta = json.loads((sess / f"{label}.json").read_text(encoding="utf-8"))
    z = np.load(sess / f"{label}.npz")
    wk = basis_weights(str(basis_run), animal)
    ev = {w: z[f"evoked_{w}"].astype(np.float64)[..., :-1] @ wk for w in ("cue", "lick")}   # last target = cortex
    return meta["evoked_parts"], meta["pos_order"], ev


def residual_rows(run: Path, labels, variant: str = "joint", by_spout: bool = False):
    """Split-row counterpart of `specificity_control.session_rows` (mean of the six spouts; columns data, full, residual,
    x1e3 dF/F) or, with ``by_spout``, of `residual_roi.spout_table` (one row per spout; column residual), for the
    sessions ``labels`` of the area-target run ``run``, read from its basis twin `<run>_basis` (module docstring)."""
    import json

    import pandas as pd

    from wfield_local import config
    basis = Path(f"{run}_basis")
    rows = []
    for lab in labels:
        an = config.animal_of(lab)
        parts, pos, ev = basis_evoked(str(basis), variant, lab, an)
        ep = json.loads((basis / variant / "sessions" / f"{lab}.json").read_text(encoding="utf-8"))["epoch"]
        for w, e in ev.items():
            d, f = e[parts.index("data")] * 1e3, e[parts.index("full")] * 1e3             # (spouts, areas)
            if by_spout:
                rows += [(lab, an, ep, w, sp, a, float(d[si, ai] - f[si, ai]))
                         for si, sp in enumerate(pos) for ai, a in enumerate(SPLIT_AREAS)]
            else:
                dm, fm = np.nanmean(d, 0), np.nanmean(f, 0)
                rows += [{"animal": an, "label": lab, "epoch": ep, "window": w, "area": a, "data": float(dm[ai]),
                          "full": float(fm[ai]), "residual": float(dm[ai] - fm[ai])} for ai, a in enumerate(SPLIT_AREAS)]
    if by_spout:
        return pd.DataFrame(rows, columns=["label", "animal", "epoch", "window", "spout", "area", "residual"])
    return pd.DataFrame(rows)


def split_image(img: np.ndarray, nm: list) -> tuple:
    """`frozen_area_maps.area_image` output with MOs / MOp painted as their split ROIs (display only: every pixel of the
    Allen parent goes to the sub-ROI if it is in the full v8 ROI, else to the rest). Returns (img, names)."""
    flat, parea, names, shape, pstat = grid()
    roi = il.roi_masks(parea, pstat, names, flat, shape)
    img2 = img.copy().ravel()
    nm2 = list(nm)
    for src, dst in il.SPLIT_PAIRS:
        for h in il.HEMIS:
            par = img2 == nm.index(f"{src}_{h}")
            g = np.zeros(img2.size, bool)
            g[flat[roi[f"{dst}_{h}"]]] = True
            nm2 += [f"{dst}_{h}", f"{src}_rest_{h}"]
            img2[par & g] = len(nm2) - 2
            img2[par & ~g] = len(nm2) - 1
    return img2.reshape(img.shape), nm2
