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
  MUSALL frozen Musall R2 (claims figures 0.3 / 0.4 / 0.9-0.11), from the basis runs RE-RUN with FROZEN_PIX_MUSALL=1
         (`<run>_basis_pixmusall`: per session x variant x window the target-space second moments of the frame / trial /
         avg segments, `frozen_encoder.musall_moments`): ROI R2 = 1 - w'Ew / w'Sw (`musall_rows`). Validation against
         the area run's MOs / MOp / SSp-m / cortex values: docs/status/ALM_TJM1_ROWS_PLAN_20261027.md.
  ROI TARGETS  Layer 2 (`allsession_encoder --roi-targets`): the ROI means as extra encoder targets (`roi_targets`).
Not covered (only area-mean traces were saved -> a refit is needed; see the plan doc): per-session trial / PETH R2
(`fr_trial_peth`).
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


def area_mask(area: str) -> np.ndarray:
    """Boolean over brain pixels for a split ROI / parent (`masks`) or any Allen area on the stat mask (the encoder's
    area target: `frozen_encoder.session_design` averages Ub over (parea == code) & pstat)."""
    m = masks()
    if area in m:
        return m[area]
    _flat, parea, names, _shape, pstat = grid()
    code = {v: k for k, v in names.items()}
    return (parea == code[area]) & pstat


def roi_targets(areas=tuple(SPLIT_AREAS)) -> dict:
    """{"names", "P"}: ROI-mean weights over the STAT pixels (columns sum to 1), the `spatial` / extra-target form
    `frozen_encoder.session_design` takes -- Ub[pstat].T @ P = each ROI's mean, i.e. the area-target definition."""
    pstat = grid()[4]
    return {"names": list(areas), "P": np.stack([area_mask(a)[pstat] / area_mask(a)[pstat].sum() for a in areas], 1)}


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
    return P.T @ roi_targets(areas)["P"]                                          # (K, areas)


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


def musall_rows(pix_run, variant: str, measure: str = "frame", areas=tuple(SPLIT_AREAS), fit_run=None):
    """Split-row counterpart of `frozen_fr_compare.session_r2` (label, animal, epoch, window, area, r2): the frozen Musall
    R2 (measure frame | trial | avg) of each ROI mean, from a FROZEN_PIX_MUSALL basis run ``pix_run`` (npz keys
    mus_<window>_<measure>_<E|S>, targets = K basis components + the cortex mean): R2 = 1 - w'Ew / w'Sw, w = the ROI's
    basis weights (`basis_weights`, cortex entry 0). An area named "cortex" uses the cortex target itself. ``fit_run`` =
    the folder holding fit/spatial_<animal>.pkl (default ``pix_run``; a --fit-from re-run keeps the basis in its source)."""
    import json

    import pandas as pd

    from frozen_fr_compare import WINDOWS
    sess = Path(pix_run) / variant / "sessions"
    fit_run = str(fit_run or pix_run)
    rows = []
    for js in sorted(sess.glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        z = np.load(js.with_suffix(".npz"))
        K = len(m["targets"]) - 1
        if m["targets"][-1] != "cortex":
            raise ValueError(f"{js}: last target is not the cortex mean")
        roi = [a for a in areas if a != "cortex"]
        W = np.zeros((K + 1, len(areas)))
        if roi:
            W[:K, [areas.index(a) for a in roi]] = basis_weights(fit_run, m["animal"], tuple(roi))
        if "cortex" in areas:
            W[K, areas.index("cortex")] = 1.0
        for w in WINDOWS:
            kE, kS = f"mus_{w}_{measure}_E", f"mus_{w}_{measure}_S"
            if kE not in z.files:
                raise KeyError(f"{js.with_suffix('.npz')}: no {kE} (not a FROZEN_PIX_MUSALL run with the {measure} moment)")
            Em, Sm = z[kE].astype(np.float64), z[kS].astype(np.float64)
            sse, sst = np.einsum("ka,kl,la->a", W, Em, W), np.einsum("ka,kl,la->a", W, Sm, W)
            rows += [{"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "window": w, "area": a,
                      "r2": float(1.0 - sse[i] / max(sst[i], 1e-12))} for i, a in enumerate(areas)]
    return pd.DataFrame(rows, columns=["label", "animal", "epoch", "window", "area", "r2"])


#: the FROZEN_PIX_MUSALL basis re-run with the UNION of the variants slides 94 (`frozen_peth_trial_maps.py`) and 139
#: (`session_vs_frozen.py --maps`) need (11-06): `frozen_encoder_<area-run tag><PIX_UNION>`, fit (pre statistics,
#: templates, basis) read from `frozen_encoder_<tag>_basis`, alphas held at the area run's. The 10-28 twin
#: (`..._basis_pixmusall`, frozen_claims' variants) is a different folder and is never overwritten.
PIX_UNION = "_basis_pixmusall_union"


#: the EXACT split-ROI frozen run (11-06, Priya: slide 139 / remap_checks need exact ROI R2): the area run re-scored with
#: the 8 ROI means as extra targets, penalties held (`frozen_encoder --roi-targets`): `frozen_encoder_<tag><ROI_EXACT>`
ROI_EXACT = "_roiexact"


def roi_frozen_source(enc_root, tag: str, mode: str = "auto", roi_run=None, pix_run=None, fit_run=None) -> tuple:
    """(run folder, basis fit run or None) of the frozen split-ROI R2 for the area run ``tag``: mode "exact" = the
    --roi-targets run (fit None), "pixmusall" = the basis route (`pix_union_paths`), "auto" = exact when present, else
    the pixmusall union run. SystemExit when the chosen run has no joint sessions."""
    ex = Path(roi_run) if roi_run else Path(enc_root) / f"frozen_encoder_{tag}{ROI_EXACT}"
    pix0, fit0 = pix_union_paths(enc_root, tag)
    pix = (Path(pix_run or pix0), Path(fit_run or fit0))
    ok = lambda r: (Path(r) / "joint" / "sessions").is_dir()   # noqa: E731
    if mode == "exact" or (mode == "auto" and ok(ex)):
        if not ok(ex):
            raise SystemExit(f"no exact ROI-target frozen run at {ex} (frozen_encoder --roi-targets)")
        return ex, None
    if not ok(pix[0]):
        raise SystemExit(f"no frozen split-ROI run: neither {ex} (exact) nor {pix[0]} (pixmusall union)")
    return pix


def pix_union_paths(enc_root, tag: str) -> tuple:
    """(pixmusall union run, basis run holding fit/spatial_<animal>.pkl) of the area run ``tag`` under ``enc_root``."""
    return Path(enc_root) / f"frozen_encoder_{tag}{PIX_UNION}", Path(enc_root) / f"frozen_encoder_{tag}_basis"


def bonf_allen(B, keys, ref=None) -> np.ndarray:
    """p_bonf = min(p x number of ALLEN areas in the row's ``keys`` group, 1): the split ROIs re-partition MOs / MOp, so
    they are tested with the Allen count and do not change it (as `residual_roi.py`). Rows of B: area, p, *keys; the
    count comes from ``ref`` (default B: e.g. the Allen table when B holds the split rows only)."""
    ref = B if ref is None else ref
    n = ref[~ref.area.isin(SPLIT_AREAS)].groupby(list(keys)).area.nunique()
    return np.minimum(B.p.to_numpy(float) * B.set_index(list(keys)).index.map(n).to_numpy(float), 1.0)


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
    # the edge rim lateral to ALM / tjM1 is in no ROI mean; blank that sliver (Priya 10-28: "fix the sliver in all maps")
    from iti_area_maps import trim_rest
    return trim_rest(img2.reshape(img.shape), nm2), nm2
