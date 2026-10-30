"""PIXEL MAPS of the matched-lick change (Priya 10-29, item 5): WHERE in cortex do the ITI offset and the slow rise of the
contralesional share live, on licks matched for TARGET and LANDED ANGLE?

WHY A NEW COMPUTATION. Existing pixel-level lick products do not hold this quantity: the iti_licks session files hold area
/ ROI means only; map_collect's cache holds trial-level position maps (15r / 14), not per lick; lick_offset_contrib_map
maps the frozen decoder's contribution with a target-blind Gaussian same-place reference in the 0..0.2 s window;
lick_balance_maps_svd holds per-lick SVD responses in 0..0.2 s with no reference. The matched-error reference needs per-
lick pixel responses in the primary window, so they are rebuilt here from the SVD exactly as those scripts do.

LICKS / SETS: <OUT_W>/areatune_licks.csv.gz (lick_area_tuning.py; = baltarget_licks + far1 flag): cued first post-cue
lick; ITI first lick after >= 2 s quiet; ITI far1 subset (>= 1 s from the trial stop and the next spout arrival). Pre =
AUGUST pre sessions.
PIXEL RESPONSE per lick: adopted hemo-corrected SVD (combined_encoding.load_imaging: V, U_atlas), slow drift removed
(ce._slow_basis, as iti_licks.extract), x1e3, sampled at the lick PEAK + iti_licks.LAGS_D (linear interpolation): mean over
0.1..0.5 s minus mean over -1.0..-0.5 s (the primary balance window) -> K-vector c per lick; pixel p = U_blk[p] . c on a
DS x DS block grid of the shared 540 x 640 Allen grid (block mean of U over the session's in-mask pixels, as
lick_balance_maps_svd._block_U; a block is kept if >= half of it is brain mask). VALIDATION: the session's ROI means
(iti_licks.roi_masks) of the same c reproduce baltarget c / i (POOL) per lick (pixval_*.csv).
MATCHED REFERENCE (lick_balance_matched_error.py logic, per pixel): per set x animal x spout, OLS line of the pixel
response on lick angle over the animal's PRE licks to that spout (pre sessions leave-one-session-out; < 10 pre licks ->
spout mean); a lick is used only if its angle lies inside the 2.5-97.5 % range of the pre licks to its spout (no
extrapolation). Session map R_s(p) = mean over its kept licks of (response - pre prediction) (>= MIN_LICKS kept licks).
CHANGE MAP: per animal, mean over its epoch sessions of R_s minus the mean over its pre sessions (pre R_s are LOSO, ~0);
POOLED = mean of the 4 animal maps (= analysis_kit.boot_delta's point). MIRRORED DIFFERENCE map at contralesional (image
RIGHT) pixel p: change(p) - change(mirror p) (mirror = reflection about the atlas midline, column 319.5); + = the
contralesional hemisphere gained more than the mirrored ipsilesional pixel (HEMISPHERE sign). Image LEFT = ipsilesional.
ONE colour scale per set across epochs and across both map types. Allen outlines.
STATISTICS: a priori AREAS (ALM, tjM1, SSp-m, SSp-n, SSs, MOp_rest, VISpm, RSPd, POOL; ROI pixel weights of each session
on the block grid): area value per session = ROI-weighted mean of R_s; change vs pre by analysis_kit.boot_delta (paired
nested animals -> sessions, rng seed 0), per-animal changes FIRST (rule 8); contra - ipsi per area. Pixel test: the
existing calibrated helper beta_maps.significance_contour (nested animals -> sessions bootstrap, max-statistic FWER, 8x8
bins, rim-artefact suppression; calibrated on pre-vs-pre nulls by map_test_calibration.py), normalise=False (the pre
maps are ~0 by construction) -- drawn as a contour, read as supporting only.
OUTPUTS (<OUT_W>/): pixmatch_area_sessions.csv, pixmatch_area_animals.csv, pixmatch_area_tests.csv, pixmatch_validation.csv,
pixmatch_sig.csv; figdata/lick_matched_pixel_maps.pkl (maps, per animal); figures/lick_matched_pixel_{cued,iti,iti_far1}
.png, lick_matched_pixel_animals_{cued,iti}.png, lick_matched_pixel_areas.png. Per-session c / U cache: LOCAL --cache
folder (default the session scratch dir), never on M:.
Usage: python scripts/lick_matched_pixel_maps.py [--cache DIR] [--jobs 4] | --plot-only
       --epochs [--plot-only]: per-epoch RAW maps + change maps -> lick_matched_pixel_{trial,iti}_epochs.png
       (figdata lick_matched_pixel_maps_epochs.pkl; rebuilt from the local cache, needs the main figdata)
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = Path(il.OUT_W)
if il.ITI_SUBSET:
    raise SystemExit("run without WFIELD_ITI_FILTER")
FIG = OUT / "figures"
STEM = "lick_matched_pixel_maps"
DS = 4
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POST = EPS[1:]
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs", "MOp_rest", "VISpm", "RSPd"]
POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
SETS = {"trial": "CUED first post-cue lick", "iti": "ITI first lick after >= 2 s quiet",
        "iti_far1": "ITI first lick, far1 subset (>= 1 s from trial stop and next spout arrival)"}
Q = (2.5, 97.5)
MIN_LICKS = 10
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
COL = {"acute": "#C44E52", "subacute": "#DD8452", "chronic": "#4C72B0"}
DEFAULT_CACHE = Path(r"C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab"
                     r"\f082ee0c-b33c-49a8-a537-dcee3bb7c84c\scratchpad\lmp_cache")


def _wm(lo_, hi_):
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & (L <= hi_ + 1e-9) if hi_ <= 0 else (L >= lo_ - 1e-9) & (L < hi_ - 1e-9)


def grid() -> dict:
    from wfield_local import beta_maps
    bm = beta_maps.brain_mask()
    H, W = bm.shape
    h, w = H // DS, W // DS
    rr, cc = np.divmod(np.arange(H * W), W)
    blk = (rr // DS) * w + (cc // DS)
    frac = np.bincount(blk, bm.ravel().astype(float), h * w) / DS ** 2
    good = np.flatnonzero(frac >= 0.5)
    br, bc = np.divmod(np.arange(h * w), w)
    mir = br * w + (w - 1 - bc)
    pos = np.full(h * w, -1)
    pos[good] = np.arange(len(good))
    right = np.flatnonzero((bc[good] >= w // 2) & (pos[mir[good]] >= 0))      # indices into good
    return {"H": H, "W": W, "h": h, "w": w, "blk": blk, "good": good, "right": right, "mirror": pos[mir[good[right]]]}


# ------------------------------------------------------------------------------------------------ per-session extract
def extract(item) -> dict:
    """(label, cache dir) -> <cache>/<label>.npz: U (good blocks x K), C_<set> (pred rows x K; NaN = unmatched),
    W (area-hemi ROI weights on the good blocks), pool c / i of the same c (validation). Module-level for spawn."""
    import lick_balance_reference as lbr

    from lick_balance_maps_svd import _block_U
    from wfield_local import combined_encoding as ce
    label, cache = item
    f = Path(cache) / f"{label}.npz"
    if f.exists():
        return {"label": label, "cached": True}
    G = grid()
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    if tuple(shape) != (G["H"], G["W"]):
        raise RuntimeError(f"{label}: grid {shape}")
    U = _block_U(Ub, flat, G)[G["good"]]
    U = np.nan_to_num(U).astype(np.float32)
    Q_ = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    Vd = V.astype(np.float64)
    Vd -= Q_ @ (Q_.T @ Vd)
    Vd *= 1e3
    win, base = _wm(0.1, 0.5), _wm(*il.BASE_RESP)
    lw, lb = il.LAGS_D[win], il.LAGS_D[base]
    M = il.roi_masks(parea, pstat, names, flat, shape)
    # ROI pixel-count weights per good block (area-hemi rows); rest areas built as in with_rest
    nb = G["h"] * G["w"]
    pos = np.full(nb, -1)
    pos[G["good"]] = np.arange(len(G["good"]))
    wrows, wnames = [], []
    for a in AREAS:
        for hk, h in (("c", "right"), ("i", "left")):
            if a == "MOp_rest":
                sel = M[f"MOp_{h}"] & ~M[f"tjM1_{h}"]          # boolean masks over `flat`
            else:
                sel = M[f"{a}_{h}"]
            b = pos[G["blk"][flat[sel]]]
            b = b[b >= 0]
            wrows.append(np.bincount(b, minlength=len(G["good"])) / max(len(b), 1))
            wnames.append(f"{a}|{hk}")
    acts = lbr.cv._acts(label)
    d = il._load(label)
    out = {"U": U, "W": np.asarray(wrows, np.float32), "wnames": np.array(wnames)}
    for st, (tk, sk) in {"iti": ("iti_t", "iti_side"), "trial": ("tr_t", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        C = np.full((len(act), V.shape[1]), np.nan, np.float32)
        pc = np.full((len(act), 2), np.nan)
        if len(act):
            m = il.subseq_match(d[sk].astype(float), act)
            ok = m >= 0
            t = d[tk].astype(float)[m[ok]]
            tw = (t[:, None] + lw[None, :]).ravel()
            tb = (t[:, None] + lb[None, :]).ravel()
            cw = np.empty((len(tw), Vd.shape[1]))
            cb = np.empty((len(tb), Vd.shape[1]))
            for j in range(Vd.shape[1]):
                cw[:, j] = np.interp(tw, ft, Vd[:, j])
                cb[:, j] = np.interp(tb, ft, Vd[:, j])
            c = cw.reshape(len(t), len(lw), -1).mean(1) - cb.reshape(len(t), len(lb), -1).mean(1)
            C[ok] = c
            for j, h in enumerate(("right", "left")):
                pc[ok, j] = sum(c @ Ub[M[f"{a}_{h}"]].astype(np.float64).mean(0) for a in POOL)
        out[f"C_{st}"] = C
        out[f"pool_{st}"] = pc
    Path(cache).mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(f)
    return {"label": label, "K": int(V.shape[1]), "P": int(U.shape[0])}


# ------------------------------------------------------------------------------------------------ maps
def _lick_table() -> pd.DataFrame:
    L = pd.read_csv(OUT / "areatune_licks.csv.gz", usecols=["set", "animal", "label", "epoch", "pos", "k", "angle",
                                                             "c", "i", "far1", "day"])
    L = L.dropna(subset=["angle", "pos", "c", "i"])
    F = L[(L.set == "iti") & L.far1.astype(bool)].assign(set="iti_far1")
    L = pd.concat([L, F], ignore_index=True)
    out = []
    for (_st, _an), g in L.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        lo = pre.groupby("pos").angle.quantile(Q[0] / 100)
        hi = pre.groupby("pos").angle.quantile(Q[1] / 100)
        out.append(g.assign(inside=(g.angle >= g.pos.map(lo)) & (g.angle <= g.pos.map(hi))))
    return pd.concat(out, ignore_index=True)


def _stats(L: pd.DataFrame, cache: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Per (set, label, pos): scalar sums + pixel sums SY, SaY over all licks and SY_in over kept licks."""
    ST, val, W = {}, [], {}
    for lab, gl in L.groupby("label"):
        z = np.load(cache / f"{lab}.npz")
        U = z["U"].astype(np.float64)
        W[lab] = (z["W"].astype(np.float64), list(z["wnames"]))
        for st, g in gl.groupby("set"):
            src = "trial" if st == "trial" else "iti"
            C = z[f"C_{src}"].astype(np.float64)[g.k.to_numpy(int)]
            if st != "iti_far1":
                pc = z[f"pool_{src}"][g.k.to_numpy(int)]
                for j, h in enumerate("ci"):
                    e = np.abs(pc[:, j] - g[h].to_numpy(float))
                    val.append({"label": lab, "set": st, "hemi": h, "n": len(e), "max_abs_err": np.nanmax(e),
                                "sd": g[h].std(), "n_nan": int(np.isnan(pc[:, j]).sum())})
            okc = np.isfinite(C).all(1)
            for p, gp in g[okc].groupby("pos"):
                Cp = C[okc][(g[okc].pos == p).to_numpy()]
                a = gp.angle.to_numpy(float)
                ins = gp.inside.to_numpy(bool)
                ST[(st, lab, p)] = {"n": len(a), "sa": a.sum(), "saa": (a * a).sum(),
                                    "SY": Cp.sum(0) @ U.T, "SaY": (a @ Cp) @ U.T,
                                    "n_in": int(ins.sum()), "sa_in": a[ins].sum(), "SY_in": Cp[ins].sum(0) @ U.T}
        print(f"  stats {lab}", flush=True)
    return ST, pd.DataFrame(val), W


def session_maps(L: pd.DataFrame, ST: dict) -> dict:
    """{(set, label): (R residual map, Pm pre-predicted mean map, Ym kept-lick mean map, n kept)}."""
    out = {}
    meta = L.drop_duplicates(["set", "label"]).set_index(["set", "label"])
    for (st, an), g in L.groupby(["set", "animal"]):
        labs = sorted(g.label.unique())
        pre_labs = sorted(g[g.epoch == "pre"].label.unique())
        spouts = sorted(g.pos.unique())
        for lab in labs:
            num = den = 0
            sr = sp = sy = 0.0
            for p in spouts:
                s = ST.get((st, lab, p))
                if s is None or s["n_in"] == 0:
                    continue
                tr = [ST[(st, q, p)] for q in pre_labs if q != lab and (st, q, p) in ST]
                n = sum(t["n"] for t in tr)
                if n == 0:
                    continue
                sa, saa = sum(t["sa"] for t in tr), sum(t["saa"] for t in tr)
                SY, SaY = sum(t["SY"] for t in tr), sum(t["SaY"] for t in tr)
                dd = n * saa - sa * sa
                if n < 10 or dd <= 0:
                    b1, b0 = 0.0, SY / n
                else:
                    b1 = (n * SaY - sa * SY) / dd
                    b0 = (SY - b1 * sa) / n
                pred = s["n_in"] * b0 + b1 * s["sa_in"]
                sr = sr + (s["SY_in"] - pred)
                sp = sp + pred
                sy = sy + s["SY_in"]
                num += s["n_in"]
                den += 1
            if num >= MIN_LICKS:
                out[(st, lab)] = (sr / num, sp / num, sy / num, num, meta.loc[(st, lab), "epoch"], an)
    return out


def _full(F: dict, v: np.ndarray) -> np.ndarray:
    img = np.full(F["h"] * F["w"], np.nan)
    img[F["good"]] = v
    return np.repeat(np.repeat(img.reshape(F["h"], F["w"]), DS, 0), DS, 1)


def _mirror_full(F: dict, v: np.ndarray) -> np.ndarray:
    d = np.full(len(F["good"]), np.nan)
    d[F["right"]] = v[F["right"]] - v[F["mirror"]]
    return _full(F, d)


def compute(cache: Path, jobs: int) -> dict:
    from wfield_local import analysis_kit as ak
    from wfield_local import beta_maps
    L = _lick_table()
    labels = sorted(L.label.unique())
    t0 = time.time()
    res, fail = ak.fan_sessions([(lab, str(cache)) for lab in labels], extract, jobs=jobs)
    if fail:
        raise RuntimeError(f"extract failed: {fail[:5]}")
    print(f"extract done {time.time() - t0:.0f} s", flush=True)
    ST, VAL, W = _stats(L, cache)
    SM = session_maps(L, ST)
    del ST
    G = grid()
    F = {k: G[k] for k in ("h", "w", "good", "right", "mirror")}
    # per-session area values (ROI-weighted mean of R) -> area tables
    arows = []
    for (st, lab), (R, Pm, _ym, n, ep, an) in SM.items():
        Wm, wn = W[lab]
        r = {"set": st, "label": lab, "animal": an, "epoch": ep, "n": n}
        for j, nm in enumerate(wn):
            r[f"shift|{nm}"] = float(Wm[j] @ R)
            r[f"pred|{nm}"] = float(Wm[j] @ Pm)
        for h in "ci":
            r[f"shift|POOL|{h}"] = sum(r[f"shift|{a}|{h}"] for a in POOL)
            r[f"pred|POOL|{h}"] = sum(r[f"pred|{a}|{h}"] for a in POOL)
        for a in [*AREAS, "POOL"]:
            r[f"shift|{a}|c-i"] = r[f"shift|{a}|c"] - r[f"shift|{a}|i"]
            for h in "ci":
                p_ = r[f"pred|{a}|{h}"]
                r[f"rshift|{a}|{h}"] = r[f"shift|{a}|{h}"] / p_ if p_ >= 3.0 else np.nan     # as lick_area_tuning.MIN_PRED
            r[f"rshift|{a}|c-i"] = r[f"rshift|{a}|c"] - r[f"rshift|{a}|i"]
        arows.append(r)
    AS = pd.DataFrame(arows).sort_values(["set", "animal", "label"]).reset_index(drop=True)
    AA, AT = [], []
    for st, g in AS.groupby("set"):
        for c in [c for c in AS.columns if c.startswith(("shift|", "rshift|"))]:
            met, a, h = c.split("|")
            pre = {x: v[c].dropna().to_numpy() for x, v in sorted(g[g.epoch == "pre"].groupby("animal"))}
            for ep in POST:
                post = {x: v[c].dropna().to_numpy() for x, v in sorted(g[g.epoch == ep].groupby("animal"))}
                post = {x: v for x, v in post.items() if len(v) and len(pre.get(x, []))}
                if not post:
                    continue
                am = {x: post[x].mean() - pre[x].mean() for x in post}
                for x, v in am.items():
                    AA.append({"set": st, "metric": met, "area": a, "hemi": h, "epoch": ep, "animal": x, "change": v})
                ci = ak.boot_delta(post, {x: pre[x] for x in post}, np.random.default_rng(0))
                sg = np.sign(ci.point)
                AT.append({"set": st, "metric": met, "area": a, "hemi": h, "epoch": ep, "point": ci.point, "lo": ci.lo,
                           "hi": ci.hi, "n_same_sign": int(sum(np.sign(v) == sg for v in am.values())),
                           "n_animals": len(am), "excl0": bool(ci.lo > 0 or ci.hi < 0),
                           **{f"d_{x}": v for x, v in am.items()}})
    # maps: per animal x epoch mean session map; change vs pre; pooled; significance contours
    maps, sig_rows, contours = {}, [], {}
    stat = beta_maps.stat_mask()
    cov = np.isfinite(_full(F, np.zeros(len(F["good"]))))
    m_all = stat & cov
    mir_cov = np.isfinite(_mirror_full(F, np.zeros(len(F["good"]))))
    m_mir = m_all & mir_cov & np.fliplr(m_all)
    for st in SETS:
        by = {}
        for (s2, lab), v in SM.items():
            if s2 == st:
                by.setdefault((v[5], v[4]), []).append((lab, v))
        for an in AN:
            if (an, "pre") not in by:
                continue
            prem = np.mean([v[0] for _l, v in by[(an, "pre")]], 0)
            maps[(st, an, "pre_resp")] = np.mean([v[2] for _l, v in by[(an, "pre")]], 0).astype(np.float32)
            for ep in POST:
                if (an, ep) in by:
                    maps[(st, an, ep)] = (np.mean([v[0] for _l, v in by[(an, ep)]], 0) - prem).astype(np.float32)
                    maps[(st, an, f"n_{ep}")] = len(by[(an, ep)])
        for ep in POST:
            pre_d = {an: [_full(F, v[0]) for _l, v in by[(an, "pre")]] for an in AN if (an, "pre") in by
                     and (an, ep) in by}
            post_d = {an: [_full(F, v[0]) for _l, v in by[(an, ep)]] for an in pre_d}
            for kind, mk in (("diff", m_all), ("mirror", m_mir)):
                if kind == "mirror":
                    pre_d = {an: [_mirror_full(F, v[0]) for _l, v in by[(an, "pre")]] for an in pre_d}
                    post_d = {an: [_mirror_full(F, v[0]) for _l, v in by[(an, ep)]] for an in pre_d}
                keep, lab_ = beta_maps.significance_contour(pre_d, post_d, method="nested", normalise=False, mask=mk)
                contours[(st, ep, kind)] = keep
                sig_rows.append({"set": st, "epoch": ep, "map": kind, "n_animals": len(pre_d),
                                 "sig_px": int(keep.sum()) if keep is not None else 0, "label": lab_})
                print(f"  sig {st} {ep} {kind}: {lab_}", flush=True)
            del pre_d, post_d
    from frozen_area_maps import area_image
    img, nm = area_image(labels[0])
    return {"F": F, "maps": maps, "contours": contours, "img": img, "names": nm, "area_sessions": AS,
            "area_animals": pd.DataFrame(AA), "area_tests": pd.DataFrame(AT), "validation": VAL,
            "sig": pd.DataFrame(sig_rows), "masks": {"all": m_all, "mirror": m_mir}}


# ------------------------------------------------------------------------------------------------ figures
def _wrap(s: str, w: int = 150) -> str:
    return "\n".join(textwrap.fill(p, w) for p in s.split("\n"))


def plot(D: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline
    F, maps, ol = D["F"], D["maps"], outline(D["img"])
    m_all = D["masks"]["all"]
    for st in SETS:
        pooled = {}
        for ep in POST:
            v = [maps[(st, an, ep)] for an in AN if (st, an, ep) in maps]
            if v:
                pooled[ep] = (np.mean(v, 0), len(v))
        if not pooled:
            continue
        diff = {ep: np.where(m_all, _full(F, v), np.nan) for ep, (v, _n) in pooled.items()}
        mirr = {ep: np.where(D["masks"]["mirror"], _mirror_full(F, v), np.nan) for ep, (v, _n) in pooled.items()}
        vmax = np.nanpercentile(np.abs(np.concatenate([x[np.isfinite(x)] for x in [*diff.values(), *mirr.values()]])),
                                99)
        pre = np.mean([maps[(st, an, "pre_resp")] for an in AN if (st, an, "pre_resp") in maps], 0)
        # dedicated colour-bar column (and a spacer before the PRE panel) so bars never sit on the panel titles
        fig_h = 4.2 * len(POST) + 1.3   # extra headroom so the wrapped suptitle clears the panel titles
        fig = plt.figure(figsize=(15, fig_h))
        gs = fig.add_gridspec(len(POST), 5, width_ratios=[1, 1, 0.045, 0.35, 1], wspace=0.08, hspace=0.35)
        axs = np.array([[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), fig.add_subplot(gs[i, 4])]
                        for i in range(len(POST))])
        for i, ep in enumerate(POST):
            if ep not in pooled:
                for ax in axs[i]:
                    ax.axis("off")
                continue
            for j, (img, kind, ttl) in enumerate(((diff[ep], "diff", "post - pre at matched spout + angle"),
                                                  (mirr[ep], "mirror", "MIRRORED: contra pixel - mirrored ipsi pixel"))):
                ax = axs[i, j]
                im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                k = D["contours"].get((st, ep, kind))
                if k is not None and k.any():
                    ax.contour(k.astype(float), levels=[0.5], colors="k", linewidths=1.0)
                srow = D["sig"][(D["sig"].set == st) & (D["sig"].epoch == ep) & (D["sig"]["map"] == kind)]
                ax.set_title(_wrap(f"{ep.upper()} (n = {pooled[ep][1]} animals): {ttl}\n"
                                   f"pixel test: {srow.label.iloc[0] if len(srow) else '-'}", 48), fontsize=7.5)
                ax.axis("off")
            cax = fig.add_subplot(gs[i, 2])
            fig.colorbar(im, cax=cax).set_label("dF/F x1e3 (+ = larger after)", fontsize=8)
            ax = axs[i, 2]
            if i == 0:
                imp = ax.imshow(np.where(m_all, _full(F, pre), np.nan), cmap="magma", interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                ax.set_title("PRE response, kept licks (mean of animals)\nfor scale; own colour scale", fontsize=7.5)
                pcax = ax.inset_axes([1.03, 0.1, 0.04, 0.8])
                fig.colorbar(imp, cax=pcax).set_label("dF/F x1e3", fontsize=8)
            ax.axis("off")
        fig.suptitle(_wrap(f"{SETS[st]}: PIXEL map of the post-lick response (0.1-0.5 s minus -1..-0.5 s from the lick peak) "
                           "of post-stroke licks minus PRE licks to the SAME spout landing at the SAME angle (pre within-"
                           "spout line per pixel, pre LOSO; only licks inside the pre angle range of their spout). Mean "
                           "of animal change maps; one colour scale across epochs and both map types. Image LEFT = "
                           "ipsilesional. Contour = calibrated nested-bootstrap max-stat test (supporting only; areas "
                           "are the a priori test). PROVISIONAL", 160), fontsize=9, y=0.995)
        fig.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=1 - 1.75 / fig_h)
        fig.savefig(FIG / f"lick_matched_pixel_{st}.png", dpi=90)
        plt.close(fig)
        # per animal (rule 8): mirrored map per animal x epoch, same scale
        fig, axs = plt.subplots(len(AN), 2 * len(POST), figsize=(3.0 * 2 * len(POST), 2.9 * len(AN)), squeeze=False)
        for i, an in enumerate(AN):
            for j, ep in enumerate(POST):
                for kk, kind in enumerate(("diff", "mirror")):
                    ax = axs[i, 2 * j + kk]
                    ax.axis("off")
                    if (st, an, ep) not in maps:
                        continue
                    v = maps[(st, an, ep)]
                    img = (np.where(m_all, _full(F, v), np.nan) if kind == "diff"
                           else np.where(D["masks"]["mirror"], _mirror_full(F, v), np.nan))
                    ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                    ax.set_title(f"{an} {ep} {'post-pre' if kind == 'diff' else 'mirrored c-i'}\n"
                                 f"({maps.get((st, an, 'n_' + ep), 0)} sessions)", fontsize=7)
        fig.suptitle(_wrap(f"{SETS[st]}: PER ANIMAL matched-lick change maps (same scale as the pooled figure, +-"
                           f"{vmax:.1f} dF/F x1e3); image LEFT = ipsilesional; mirrored = contra pixel minus mirrored "
                           "ipsi pixel (+ = contralesional hemisphere gained more). PROVISIONAL", 150), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(FIG / f"lick_matched_pixel_animals_{st}.png", dpi=80)
        plt.close(fig)
    # a priori area test figure
    T = D["area_tests"]
    areas = ["POOL", *AREAS]
    panels = [("rshift", "c", "contralesional, relative (fraction of pre-predicted)"),
              ("rshift", "i", "ipsilesional, relative"),
              ("rshift", "c-i", "contra - ipsi, relative (+ = contralesional share up)"),
              ("shift", "c-i", "contra - ipsi, additive (dF/F x1e3)")]
    sets = [s for s in SETS if s in set(T.set)]
    fig, axs = plt.subplots(len(sets), len(panels), figsize=(4.8 * len(panels), 3.8 * len(sets)), squeeze=False)
    for i, st in enumerate(sets):
        for j, (met, h, ttl) in enumerate(panels):
            ax = axs[i, j]
            t = T[(T.set == st) & (T.metric == met) & (T.hemi == h)]
            x = np.arange(len(areas))
            for k, ep in enumerate(POST):
                e = t[t.epoch == ep].set_index("area").reindex(areas)
                xx = x + (k - 1) * 0.25
                ax.errorbar(xx, e.point, yerr=[e.point - e.lo, e.hi - e.point], fmt="o", color=COL[ep], ms=4,
                            capsize=2, label=ep)
                for an in AN:
                    if f"d_{an}" in e:
                        ax.plot(xx + 0.06, e[f"d_{an}"], MK[an], mfc="none", color=COL[ep], ms=2.5, alpha=0.6, lw=0)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x, areas, rotation=45, ha="right", fontsize=7)
            ax.set_title(_wrap(f"{SETS[st]}\n{ttl}", 60), fontsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    from matplotlib.lines import Line2D
    hd, lb = axs[0, 0].get_legend_handles_labels()
    hd = list(hd) + [Line2D([], [], marker=MK[a], ls="", mfc="none", color="0.4", ms=5) for a in AN]
    lb = list(lb) + [f"{a} (open)" for a in AN]
    fig.legend(hd, lb, loc="lower center", ncol=len(lb), fontsize=8.5, frameon=False)
    fig.suptitle(_wrap("A priori AREA test on the matched pixel maps: ROI-weighted mean of the session map (post - pre at "
                       "matched spout + angle), change vs pre; filled = paired nested bootstrap 95% CI, open = animals. "
                       "Hemisphere: c = contralesional. PROVISIONAL", 170), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(FIG / "lick_matched_pixel_areas.png", dpi=90)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ per-epoch raw maps
# Priya 10-29: show the RAW per-epoch response maps (kept licks, as the PRE panel) next to the change maps. The main
# figdata holds only the PRE mean response, so the per-epoch means are rebuilt from the local SVD cache with the SAME
# kept-lick / spout / MIN_LICKS rules (_stats + session_maps), fanned per session, and saved under their own stem.
EP_STEM = "lick_matched_pixel_maps_epochs"
EP_SETS = ["trial", "iti"]


def _stats_one(item) -> tuple:
    """(label, lick rows of that label, cache) -> the _stats entries of one session (module-level for spawn)."""
    lab, g, cache = item
    ST, _val, _W = _stats(g, Path(cache))
    return ST


def epoch_maps(cache: Path, jobs: int) -> dict:
    from wfield_local import analysis_kit as ak
    L = _lick_table()
    L = L[L.set.isin(EP_SETS)]
    items = [(lab, g, str(cache)) for lab, g in L.groupby("label")]
    res, fail = ak.fan_sessions(items, _stats_one, jobs=jobs, key=lambda it: it[0])
    if fail:
        raise RuntimeError(f"stats failed: {fail[:3]}")
    ST = {}
    for _k, v in res:
        ST.update(v)
    SM = session_maps(L, ST)
    del ST
    out, nses = {}, {}
    for st in EP_SETS:
        for an in AN:
            for ep in EPS:
                v = [x for (s2, _l), x in SM.items() if s2 == st and x[5] == an and x[4] == ep]
                if v:
                    out[(st, an, ep)] = np.mean([x[2] for x in v], 0).astype(np.float32)    # kept-lick mean response
                    nses[(st, an, ep)] = len(v)
    return {"resp": out, "n_sess": nses}


def plot_epochs(D: dict, E: dict) -> None:
    """Per set: row 1 = RAW kept-lick response pre / acute / subacute / chronic (one magma scale); rows 2-3 = the change
    maps of the main figure (post - pre at matched spout + angle; mirrored contra - ipsi), one RdBu scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline

    from wfield_local.orofacial_cohort import EPOCH_LINE
    F, maps, ol = D["F"], D["maps"], outline(D["img"])
    m_all, m_mir = D["masks"]["all"], D["masks"]["mirror"]
    for st in EP_SETS:
        raw, nan_ = {}, {}
        for ep in EPS:
            v = [E["resp"][(st, an, ep)] for an in AN if (st, an, ep) in E["resp"]]
            if v:
                raw[ep] = np.where(m_all, _full(F, np.mean(v, 0)), np.nan)
                nan_[ep] = len(v)
        if not raw:
            continue
        pooled = {}
        for ep in POST:
            v = [maps[(st, an, ep)] for an in AN if (st, an, ep) in maps]
            if v:
                pooled[ep] = (np.mean(v, 0), len(v))
        diff = {ep: np.where(m_all, _full(F, v), np.nan) for ep, (v, _n) in pooled.items()}
        mirr = {ep: np.where(m_mir, _mirror_full(F, v), np.nan) for ep, (v, _n) in pooled.items()}
        rvals = np.concatenate([x[np.isfinite(x)] for x in raw.values()])
        rlo, rhi = np.nanpercentile(rvals, 1), np.nanpercentile(rvals, 99)
        vmax = np.nanpercentile(np.abs(np.concatenate([x[np.isfinite(x)] for x in [*diff.values(), *mirr.values()]])),
                                99)
        fig = plt.figure(figsize=(15.5, 12.4))
        gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.06, hspace=0.28)
        rows = [("raw", raw, "RAW response, kept licks"),
                ("diff", diff, "CHANGE: post - pre at matched\nspout + angle"),
                ("mirror", mirr, "MIRRORED change: contra pixel -\nmirrored ipsi pixel")]
        for i, (kind, dd, rt) in enumerate(rows):
            im = None
            for j, ep in enumerate(EPS):
                ax = fig.add_subplot(gs[i, j])
                ax.axis("off")
                if ep not in dd:
                    if kind != "raw" and ep == "pre":
                        ax.text(0.5, 0.5, "pre = reference\n(change maps are\npost - pre)", ha="center", va="center",
                                fontsize=9, color="0.4", transform=ax.transAxes)
                    continue
                if kind == "raw":
                    im = ax.imshow(dd[ep], cmap="magma", vmin=rlo, vmax=rhi, interpolation="nearest")
                    n = nan_[ep]
                else:
                    im = ax.imshow(dd[ep], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                    k = D["contours"].get((st, ep, kind))
                    if k is not None and k.any():
                        ax.contour(k.astype(float), levels=[0.5], colors="k", linewidths=0.9)
                    n = pooled[ep][1]
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                ax.set_title(f"{ep.upper()} (n = {n} animals)\n{rt}", fontsize=8.5, color=EPOCH_LINE[ep])
            cax = fig.add_subplot(gs[i, 4])
            fig.colorbar(im, cax=cax).set_label("dF/F x1e3" if kind == "raw" else "dF/F x1e3\n(+ = larger after)",
                                                fontsize=8)
        fig.suptitle(_wrap(f"{SETS[st]}: per-epoch RAW post-lick response maps (0.1-0.5 s minus -1..-0.5 s from the lick "
                           "peak; mean of animals of the session means over kept licks = licks inside the pre angle "
                           "range of their spout) on ONE colour scale (row 1), then the matched change maps (rows 2-3, "
                           "one scale; contour = calibrated nested-bootstrap max-stat test, supporting only). NB raw "
                           "epoch minus raw pre is not exactly row 2: row 2 subtracts the pre prediction at each lick's "
                           "own spout + landed angle. Image LEFT = ipsilesional. PROVISIONAL", 175), fontsize=9.5,
                     y=0.995)
        fig.subplots_adjust(left=0.01, right=0.95, bottom=0.01, top=0.9)
        fig.savefig(FIG / f"lick_matched_pixel_{st}_epochs.png", dpi=90)
        plt.close(fig)


def main() -> int:
    from figdata import load_fd, save_fd
    if "--epochs" in sys.argv:              # per-epoch raw maps (+ change maps), new files only
        ap = argparse.ArgumentParser()
        ap.add_argument("--epochs", action="store_true")
        ap.add_argument("--plot-only", action="store_true")
        ap.add_argument("--cache", default=str(DEFAULT_CACHE))
        ap.add_argument("--jobs", type=int, default=4)
        a = ap.parse_args()
        t0 = time.time()
        if a.plot_only:
            E = load_fd(EP_STEM)
        else:
            E = epoch_maps(Path(a.cache), min(a.jobs, 4))
            save_fd(EP_STEM, E)
        plot_epochs(load_fd(STEM), E)
        print(f"epochs done in {time.time() - t0:.0f} s")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        D = load_fd(STEM)
    else:
        D = compute(Path(a.cache), min(a.jobs, 4))
        save_fd(STEM, D)
        D["area_sessions"].to_csv(OUT / "pixmatch_area_sessions.csv", index=False)
        D["area_animals"].to_csv(OUT / "pixmatch_area_animals.csv", index=False)
        D["area_tests"].to_csv(OUT / "pixmatch_area_tests.csv", index=False)
        D["validation"].to_csv(OUT / "pixmatch_validation.csv", index=False)
        D["sig"].to_csv(OUT / "pixmatch_sig.csv", index=False)
        v = D["validation"]
        print(f"validation pool (ROI means of pixel c vs baltarget c/i): max |err| {v.max_abs_err.max():.3g}, "
              f"median sd {v.sd.median():.3g}")
        pd.set_option("display.width", 250)
        T = D["area_tests"]
        print(T[T.metric == "rshift"].round(3).to_string(index=False))
    plot(D)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
