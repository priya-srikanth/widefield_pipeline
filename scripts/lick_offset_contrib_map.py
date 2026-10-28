"""WHERE IN CORTEX the same-place read-out offset comes from, per decoder basis (Priya 10-27: "run the chronic offset
map"; why does the chronic offset persist in Allen areas / LocaNMF but not in the pre-stroke SVD basis?) -- and, since
10-27 late (Priya "yes do both", OVERNIGHT 5c-bis), on the ANGLE / 3-D REACH decoder too.

The frozen pre-stroke decoder is linear in its (scaled) features h: dec = b0 + beta . h, with beta = ridge coef /
StandardScaler scale (all-pre fit). Per lick, the like-for-like change is
  dh_i = h_i - sum_j K(act_i - act_j) h_j / sum_j K       (j = the animal's pre licks of the same set; Gaussian, sigma)
and beta . dh is the part of the offset a pixel map can show. The like-for-like read-out itself (readout_change) scores
pre licks with their HELD-OUT (leave-one-pre-session-out) decoder, post licks with the all-pre decoder;
readout_change = beta . dh + heldout_ref_term, the latter = the in-sample vs held-out gap of the pre reference (larger
for bases with more features).
Per basis the features are f = M x of pixel activity x, with a spatial matrix B such that M B = I:
  areas    M = area means (stat mask, iti_licks.roi_masks), B = area indicators (split basis: ALM, tjM1 and the disjoint
           MOs_rest / MOp_rest masks, il.area_names order)
  svd      M = P' (the animal's pre-stroke spatial basis, orthonormal), B = P
  locanmf  M = s * pinv(A) (joint pre-stroke LocaNMF footprints A, footprint scale s), B = A / s
(each further divided by its per-session scaling sd, folded into h). Pixel CONTRIBUTION MAP
  c(p) = w(p) * dx(p),  w = M' beta (decoder weight per pixel),  dx = B dh (activity change at matched landing)
sums over pixels to beta . dh = the offset. Split: GLOBAL part = sum_p w(p) * mean(dx) (a uniform cortex-wide change)
vs PATTERN part = the rest. Per animal: epoch mean of session means of dh, minus the same for pre. Per-FEATURE terms
beta_k * dh_k (terms_*.csv) sum to the offset exactly; in the area basis they are the per-area sums of c.

COORDINATES (--coord):
  lateral (default)  the KEY lateral decoder (iti_lick_order.predictions via CONTRIB_HOOK), any basis
                     (WFIELD_LICK_DECODER), sigma = SIGMA (0.1 mm), reference = every pre lick with weight > 1e-6
                     (unchanged since 10-27).
  angle / reach / elevation   the frozen POLAR decoder of iti_where_angle_reach.py (per animal x set ridge on the
                     z-scored Allen areas of il.AREA_BASIS, default the ALM / tjM1 split; one ridge per coordinate;
                     iti_where_angle_reach.CONTRIB_HOOK). Reference = the GAUSSIAN same-place mean of
                     lick_matched_angle_reach.py (likeforlike method "nw", the PRIMARY reference since 10-27 late):
                     sigma by held-out pre error (likeforlike.choose, printed; angle 2 deg on 10-27), and a lick gets
                     a reference only inside the pre range (0.5-99.5 %) with raw kernel weight >= 3 (likeforlike's
                     rule). The kernel is evaluated exactly per lick (likeforlike interpolates a grid), so
                     readout_change matches matched_axes_tests.csv closely, not to the last digit.
                     Several coordinates in one call share one decoder fit (--coord angle reach).
Outputs <E>/iti_licks/offset_contrib/
  lateral:  summary_<tag>.csv, maps_<tag>.npz, terms_<tag>.csv; figures/lick_{iti,cued}_offset_contrib_map.png
  polar:    summary_<coord>_<tag>.csv, maps_<coord>_<tag>.npz, terms_<coord>_<tag>.csv;
            figures/lick_offset_contrib_<coord>.png (maps + per-area terms) and, for angle,
            figures/lick_offset_contrib_angle_vs_lateral.png (needs the lateral run of the same tag);
            figure data figdata/lick_offset_contrib_<coord>.pkl (--plot-only redraws from it).
Usage: WFIELD_LICK_DECODER=<areas|svd|locanmf> [WFIELD_LOCANMF_MASK=1] python scripts/lick_offset_contrib_map.py
       python scripts/lick_offset_contrib_map.py --coord angle reach      (split basis)
       python scripts/lick_offset_contrib_map.py [--coord angle reach] --plot-only
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = il.OUT / "offset_contrib"
FIG = il.OUT / "figures"
SIGMA = 0.1                      # mm, the held-out-chosen Gaussian width of the LATERAL same-place reference
EP = ["acute", "subacute", "chronic"]
AN = ["PS92", "PS93", "PS94", "PS95"]
SETS = {"iti": "ITI", "trial": "cued"}
H, W = 540, 640
ACC: dict = {}
MIN_REF = 8                      # licks with a reference per session
# polar coordinate -> (ridge index in iti_where_angle_reach.predictions, actual col, decoded col, likeforlike sigmas, unit)
POLAR = {"angle": (0, "act_ang", "dec_ang", "SIGMA_DEG", "deg"),
         "reach": (1, "act_reach", "dec_reach", "SIGMA_MM", "mm"),
         "elevation": (2, "act_elev", "dec_elev", "SIGMA_DEG", "deg")}
PLUS = {"angle": "+ = read more CONTRALESIONAL", "reach": "+ = read LONGER", "elevation": "+ = read STEEPER DOWN",
        "lateral": "+ = read more CONTRALESIONAL"}


def session_dh(S: dict, dec, sigma: float, nw: bool = False) -> dict:
    """THE same-place feature change per session. S {label: (epoch, X scaled, act)}; dec(label, X) -> decoded values
    (held-out decoder for a pre label). Returns {label: (epoch, mean dh, mean like-for-like read-out)}. A pre session is
    referenced to the OTHER pre sessions. nw = likeforlike's Gaussian rule (inside the reference's 0.5-99.5 % range and
    raw weight >= 3); else every lick with weight > 1e-6 (the lateral rule)."""
    import likeforlike as lfl
    pre = [x for x, v in S.items() if v[0] == "pre"]
    Xp = np.vstack([S[x][1] for x in pre])
    ap = np.concatenate([S[x][2] for x in pre])
    dp = np.concatenate([dec(x, S[x][1]) for x in pre])           # pre licks, each by its held-out decoder
    lab = np.concatenate([np.full(len(S[x][2]), x, dtype=object) for x in pre])
    sess = {}
    for x, (ep, X, act, *_r) in S.items():
        ref = lab != x
        K = np.exp(-0.5 * ((act[:, None] - ap[None, ref]) / sigma) ** 2)
        ok = K.sum(1) > 1e-6
        if nw:
            lo, hi = np.percentile(ap[ref], lfl.Q_EDGE)
            ok = (K.sum(1) >= 3) & (act >= lo) & (act <= hi)
        if ok.sum() < MIN_REF:
            continue
        Kn = K[ok] / K[ok].sum(1, keepdims=True)
        dh = X[ok] - Kn @ Xp[ref]
        full = dec(x, X[ok]) - Kn @ dp[ref]                # = the like-for-like read-out (held-out pre reference)
        sess[x] = (ep, dh.mean(0), float(full.mean()))
    return sess


def _hook(st, a, S, fits):
    """iti_lick_order.CONTRIB_HOOK (lateral): S {label: (epoch, X scaled, act, order, pos)}, fits {held: (sc, ridge)}."""
    sc, m = fits[None]

    def dec(x, X):
        s_, m_ = fits[x if x in fits else None]
        return m_.predict(s_.transform(X))
    ACC[(st, a)] = (m.coef_ / sc.scale_, session_dh(S, dec, SIGMA))


def _polar_hook(st, a, S, fits):
    """iti_where_angle_reach.CONTRIB_HOOK: S {label: (epoch, X scaled, Y (ang, reach, elev, sag), pos)},
    fits {held: (sc, [ridge per coordinate])}. Kept raw: sigma is chosen after the predictions (likeforlike.choose)."""
    ACC[(st, a)] = (S, fits)


def _grid(a: str):
    """flat brain-pixel indices, stat mask over them, area masks over them (one pre session of the animal)."""
    from iti_licks import roi_masks
    from wfield_local import combined_encoding as ce
    lab = il.labels(f"{a}_*")[0]
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(lab)
    if tuple(int(s) for s in shape) != (H, W):
        raise ValueError(f"{a}: imaging grid {shape} != {(H, W)}")
    return flat, pstat, roi_masks(parea, pstat, names, flat, shape)


def feature_names(source: str, a: str) -> list[str]:
    """The decoder's feature names in the area bases (il.area_names order); component numbers otherwise."""
    if source in ("areas", "areas_split"):
        cols = [str(c) for c in il._load(il.labels(f"{a}_*")[0])["cols"]]
        return il.area_names(cols, "split" if source == "areas_split" else "plain")
    return []


def spatial(source: str, a: str, keep=None):
    """(B (npix, K), Mt (npix, K), pixel flat indices on the H x W grid) for one animal and basis."""
    if source in ("areas", "areas_split"):
        flat, _pstat, M = _grid(a)
        if source == "areas_split":
            for src, dst, h in il._rest_list():
                M[f"{src}_rest_{h}"] = M[f"{src}_{h}"] & ~M[f"{dst}_{h}"]
        names = feature_names(source, a)                              # the decoder's feature order (il.area_features)
        B = np.stack([M[n] for n in names], 1).astype(float)          # indicators over brain pixels
        Mt = B / B.sum(0, keepdims=True)
        return B, Mt, flat
    if source == "svd":
        from iti_basis_decoder import SPATIAL
        flat, pstat, _M = _grid(a)
        P = np.asarray(pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], float)
        return P, P, flat[pstat]
    from wfield_local import joint_locanmf as jl
    from wfield_local.joint_locanmf import _footprint_scale
    b = jl.load(a, warn_stale=False)
    Araw = np.asarray(b.A, float).reshape(-1, b.ncomp)
    inside = np.isfinite(Araw).any(1)
    A = np.nan_to_num(Araw[inside])
    s = _footprint_scale(np.asarray(b.A), b.ncomp)
    Mt = (A @ np.linalg.pinv(A.T @ A)) * s[None, :]                    # (s pinv(A))'
    B = A / s[None, :]
    if keep is not None:
        B, Mt = B[:, keep], Mt[:, keep]
    return B, Mt, np.flatnonzero(inside)


def decompose(st: str, a: str, beta: np.ndarray, sess: dict, B, Mt, pix, names: list[str], dec_change) -> tuple:
    """Per epoch: the offset beta . dh split into global / pattern parts + per-feature terms + the contribution image.
    dec_change(epoch) -> the LOSO decoded change of the predictions (sanity column). Returns (rows, maps, terms)."""
    if B.shape[1] != len(beta):
        raise ValueError(f"{a}: {B.shape[1]} spatial columns vs {len(beta)} features")
    w = Mt @ beta
    by = {ep: np.mean([v for e, v, _f in sess.values() if e == ep], 0) for ep in ["pre", *EP]
          if any(e == ep for e, _v, _f in sess.values())}
    full = {ep: np.mean([f_ for e, _v, f_ in sess.values() if e == ep]) for ep in by}
    rows, maps, terms = [], {}, []
    wimg = np.full(H * W, np.nan)
    wimg[pix] = w
    maps[(st, a, "w")] = wimg.reshape(H, W).astype(np.float32)
    names = names or [f"c{k}" for k in range(len(beta))]
    for ep in EP:
        if ep not in by or "pre" not in by:
            continue
        dh = by[ep] - by["pre"]
        dx = B @ dh
        c = w * dx
        glob = float(w.sum() * dx.mean())
        img = np.full(H * W, np.nan)
        img[pix] = c
        maps[(st, a, ep)] = img.reshape(H, W).astype(np.float32)
        rows.append(dict(set=st, animal=a, epoch=ep, offset=float(beta @ dh), check=float(c.sum()), global_part=glob,
                         pattern_part=float(c.sum()) - glob, readout_change=float(full[ep] - full["pre"]),
                         heldout_ref_term=float(full[ep] - full["pre"] - beta @ dh), w_sum=float(w.sum()),
                         dx_mean=float(dx.mean()), n_pre=sum(e == "pre" for e, _v, _f in sess.values()),
                         n_ep=sum(e == ep for e, _v, _f in sess.values()), loso_dec_change=dec_change(ep)))
        terms += [dict(set=st, animal=a, epoch=ep, feature=n, beta=float(b_), dh=float(d_), term=float(b_ * d_))
                  for n, b_, d_ in zip(names, beta, dh)]
    return rows, maps, terms


def _save(stem: str, rows, maps, terms) -> pd.DataFrame:
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"maps_{stem}.npz", **{"|".join(k): v for k, v in maps.items()})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / f"summary_{stem}.csv", index=False)       # one file per basis / coordinate: runs go in parallel
    pd.DataFrame(terms).to_csv(OUT / f"terms_{stem}.csv", index=False)
    return R


def compute():
    """LATERAL (the KEY decoder, any basis)."""
    import os

    import iti_lick_order as lo
    source = lo.SOURCE
    lo.CONTRIB_HOOK = _hook
    P = lo.predictions()                                # fills ACC; P = the LOSO predictions (sanity)
    tag = lo.KEY_TAG.lstrip("_") or "areas"          # basis key (KEY_TAG: "areas" = plain, "areas_splitr8" = split)
    rows, maps, terms = [], {}, []
    for (st, a), (beta, sess) in sorted(ACC.items()):
        keep = lo.locanmf_inmask(a) if lo.MASK_LOC else None
        B, Mt, pix = spatial(source, a, keep)
        g = P[(P.set == st) & (P.animal == a)]
        r_, m_, t_ = decompose(st, a, beta, sess, B, Mt, pix, feature_names(source, a),
                               lambda ep, g=g: float(g[g.epoch == ep].dec.mean() - g[g.epoch == "pre"].dec.mean()))
        rows += [dict(source=tag, **r) for r in r_]
        maps.update(m_)
        terms += t_
    R = _save(tag, rows, maps, terms)
    print(os.environ.get("WFIELD_LICK_DECODER"), tag)
    print(R.round(3).to_string())
    # FIRST-K COMPONENT MAPS (Priya 10-28, SVD looks patchy): WFIELD_CONTRIB_KMAX=K rebuilds the maps from the first K
    # spatial components only. EXACT for an orthonormal basis (SVD: B = P, Mt = P): sum_p (P_K b_K)(P_K dh_K) = b_K . dh_K,
    # the part of the SAME offset carried by components 1..K (no cross terms, no refit); summary column share_of_offset.
    kmax = int(os.environ.get("WFIELD_CONTRIB_KMAX", "0"))
    if kmax and source == "svd":
        rows_k, maps_k = [], {}
        for (st, a), (beta, sess) in sorted(ACC.items()):
            B, Mt, pix = spatial(source, a, None)
            K = min(kmax, len(beta))
            sk = {x: (e, v[:K], f) for x, (e, v, f) in sess.items()}
            g = P[(P.set == st) & (P.animal == a)]
            r_, m_, _t = decompose(st, a, beta[:K], sk, B[:, :K], Mt[:, :K], pix, None,
                                   lambda ep, g=g: float(g[g.epoch == ep].dec.mean() - g[g.epoch == "pre"].dec.mean()))
            full = {(r["set"], r["epoch"]): r["offset"] for r in rows if r["animal"] == a and r["set"] == st}
            rows_k += [dict(source=f"{tag}_k{kmax}", **r, share_of_offset=r["offset"] / full[(st, r["epoch"])]
                            if full.get((st, r["epoch"])) else np.nan) for r in r_]
            maps_k.update(m_)
        Rk = _save(f"{tag}_k{kmax}", rows_k, maps_k, [])
        print(f"== first {kmax} components: offset and share of the full offset")
        print(Rk.groupby(["set", "epoch"])[["offset", "share_of_offset"]].mean().round(3).to_string())
    return R


def polar_tag() -> str:
    """Basis key of the polar decoder's outputs (the Allen basis of il.AREA_BASIS + ROI version, = lo.KEY_TAG of the
    lateral run in the same basis: 'areas_splitr8')."""
    if il.AREA_BASIS != "split":
        return "areas"
    return "areas_split" + ("" if il.ROI_VERSION == "1" else f"r{il.ROI_VERSION}")


def compute_polar(coords: list[str]) -> dict:
    """ANGLE / REACH / ELEVATION offset contributions (module docstring). The polar decoder is FITTED here (the hook
    only fires on a fit; the cached prediction file of iti_where_angle_reach is never read, nor written)."""
    import iti_where_angle_reach as wa
    import likeforlike as lfl
    ACC.clear()
    wa.CONTRIB_HOOK = _polar_hook
    P = wa.predictions()
    wa.CONTRIB_HOOK = None
    tag, source = polar_tag(), "areas_split" if il.AREA_BASIS == "split" else "areas"
    sp = {a: spatial(source, a) + (feature_names(source, a),) for a in sorted({a for _s, a in ACC})}
    out = {}
    for coord in coords:
        k, xc, yc, sgs, unit = POLAR[coord]
        Q = P.dropna(subset=[xc, yc]).reset_index(drop=True)
        sigma, cv = lfl.choose(Q, xc, yc, getattr(lfl, sgs), "nw")
        print(f"[{coord}] Gaussian (nw) sigma by held-out pre error = {sigma:g} {unit}\n"
              + cv[cv.method == "nw"].round(4).to_string(index=False), flush=True)
        rows, maps, terms = [], {}, []
        for (st, a), (S, fits) in sorted(ACC.items()):
            sc, ms = fits[None]
            beta = ms[k].coef_ / sc.scale_

            def dec(x, X, fits=fits, k=k):
                s_, m_ = fits[x if x in fits else None]
                return m_[k].predict(s_.transform(X))
            Sk = {x: (v[0], v[1], v[2][:, k]) for x, v in S.items()}
            sess = session_dh(Sk, dec, sigma, nw=True)
            B, Mt, pix, names = sp[a]
            g = Q[(Q.set == st) & (Q.animal == a)]
            r_, m_, t_ = decompose(st, a, beta, sess, B, Mt, pix, names,
                                   lambda ep, g=g, yc=yc: float(g[g.epoch == ep][yc].mean() - g[g.epoch == "pre"][yc].mean()))
            rows += [dict(source=tag, coord=coord, unit=unit, sigma=sigma, **r) for r in r_]
            maps.update(m_)
            terms += [dict(coord=coord, **t) for t in t_]
        R = _save(f"{coord}_{tag}", rows, maps, terms)
        pd.set_option("display.width", 250)
        print(f"== {coord}: per animal (rule 8) -- offset = global + pattern; readout_change = offset + held-out term")
        print(R[["set", "animal", "epoch", "offset", "global_part", "pattern_part", "readout_change",
                 "heldout_ref_term", "n_pre", "n_ep"]].round(3).to_string(index=False))
        out[coord] = (R, maps, pd.DataFrame(terms))
    return {"P": P, "out": out, "areas": {a: (v[0], v[2], v[3]) for a, v in sp.items()}}


# ------------------------------------------------------------------------------------------------ figure data + figures


def _mean_maps(maps: dict, st: str) -> dict:
    """Mean over animals of the contribution maps per epoch (+ weight map 'w'), every 2nd pixel."""
    out = {}
    for ep in [*EP, "w"]:
        v = [maps[(st, a, ep)] for a in AN if (st, a, ep) in maps]
        if v:
            out[ep] = np.nanmean(np.stack(v), 0)[::2, ::2].astype(np.float32)
    return out


def _area_terms_from_map(img: np.ndarray, B: np.ndarray, pix: np.ndarray) -> np.ndarray:
    """Per-area sum of a full-resolution contribution image (areas basis: disjoint indicator columns of B)."""
    return B.T @ np.nan_to_num(img.ravel()[pix])


def figure_data(coord: str, R: pd.DataFrame, maps: dict, T: pd.DataFrame, areas: dict) -> dict:
    """Everything the coordinate's figure (and, for angle, the angle-vs-lateral figure) draws."""
    F = {"coord": coord, "unit": str(R.unit.iloc[0]), "sigma": float(R.sigma.iloc[0]), "summary": R, "terms": T,
         "maps": {st: _mean_maps(maps, st) for st in SETS}, "basis": il.basis_label()}
    tag = polar_tag()
    lat = OUT / f"maps_{tag}.npz"
    if coord == "angle" and lat.exists() and (OUT / f"summary_{tag}.csv").exists():
        Z = np.load(lat)
        Lm = {k: Z[k] for k in Z.files}
        rows, rmap = [], []
        for st in SETS:
            for a in AN:
                if a not in areas:
                    continue
                B, pix, names = areas[a]
                for ep in EP:
                    key = f"{st}|{a}|{ep}"
                    if key not in Lm or (st, a, ep) not in maps:
                        continue
                    la = _area_terms_from_map(Lm[key], B, pix)
                    an = _area_terms_from_map(maps[(st, a, ep)], B, pix)
                    rows += [dict(set=st, animal=a, epoch=ep, feature=n, lateral=float(x), angle=float(y))
                             for n, x, y in zip(names, la, an)]
                    m1, m2 = Lm[key].ravel()[pix], maps[(st, a, ep)].ravel()[pix]
                    ok = np.isfinite(m1) & np.isfinite(m2)
                    rmap.append(dict(set=st, animal=a, epoch=ep, r_map=float(np.corrcoef(m1[ok], m2[ok])[0, 1])))
        F["vs_lateral"] = dict(terms=pd.DataFrame(rows), r_map=pd.DataFrame(rmap),
                               lateral_summary=pd.read_csv(OUT / f"summary_{tag}.csv"),
                               lateral_maps={st: {ep: np.nanmean(np.stack([Lm[f"{st}|{a}|{ep}"] for a in AN
                                                                            if f"{st}|{a}|{ep}" in Lm]), 0)[::2, ::2]
                                                  for ep in EP} for st in SETS})
    return F


def _short(n: str) -> str:
    return n.replace("_left", " L").replace("_right", " R").replace("_rest", "-rest")


def plot_polar(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coord, unit, R, T = F["coord"], F["unit"], F["summary"], F["terms"]
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 8, width_ratios=[1, 1, 1, 0.06, 0.25, 1, 0.45, 1.1], wspace=0.08, hspace=0.25)
    for i, st in enumerate(SETS):
        mm = F["maps"][st]
        lim = float(np.nanpercentile(np.abs(np.stack([mm[e] for e in EP if e in mm])), 99.5)) if mm else 1.0
        im = None
        for j, ep in enumerate(EP):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            if ep not in mm:
                continue
            im = ax.imshow(mm[ep], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            r = R[(R.set == st) & (R.epoch == ep)]
            k = int((np.sign(r.offset) == np.sign(r.offset.mean())).sum())
            ax.set_title(f"{SETS[st]}: {ep} - pre\noffset {r.offset.mean():+.2f} {unit} ({k}/{len(r)} an.)\nglobal "
                         f"{r.global_part.mean():+.2f}, pattern {r.pattern_part.mean():+.2f}; read-out "
                         f"{r.readout_change.mean():+.2f}", fontsize=8)
        if im is not None:
            cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, 3]))
            cb.set_label(f"{unit} per pixel ({PLUS[coord]})", fontsize=7)
        ax = fig.add_subplot(gs[i, 5])
        ax.axis("off")
        if "w" in mm:
            wl = float(np.nanpercentile(np.abs(mm["w"]), 99.5))
            ax.imshow(mm["w"], cmap="PuOr_r", vmin=-wl, vmax=wl, interpolation="nearest")
            ax.set_title(f"{SETS[st]}: decoder weight per pixel\n(pre fit; {coord}, {unit} per z unit)", fontsize=8)
        ax = fig.add_subplot(gs[i, 7])
        t = T[T.set == st]
        if len(t):
            am = t.groupby(["feature", "epoch"]).term.mean().unstack("epoch").reindex(columns=EP)
            agree = t.groupby(["feature", "epoch"]).term.apply(lambda s: int((np.sign(s) == np.sign(s.mean())).sum()))
            agree = agree.unstack("epoch").reindex(columns=EP)
            order = am.abs().max(1).sort_values(ascending=False).index[:14]
            am, agree = am.loc[order], agree.loc[order]
            vl = float(np.nanmax(np.abs(am.to_numpy()))) or 1.0
            ax.imshow(am.to_numpy(), cmap="RdBu_r", vmin=-vl, vmax=vl, aspect="auto")
            for (y, x), v in np.ndenumerate(am.to_numpy()):
                if np.isfinite(v):
                    ax.text(x, y, f"{v:+.2f}\n{agree.iloc[y, x]}/{t.animal.nunique()}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(len(EP)), EP, fontsize=7)
            ax.set_yticks(range(len(order)), [_short(n) for n in order], fontsize=7)
            ax.set_title(f"{SETS[st]}: per-area term beta x dh ({unit}),\nmean of animals (k/n = animals with its sign); top 14 "
                         "|term|", fontsize=8)
    fig.suptitle(f"NEURAL: where the same-place {coord.upper()} read-out offset comes from -- frozen pre-stroke {coord} "
                 f"decoder (z-scored {F['basis']}). Contribution per pixel = decoder weight x activity change vs pre "
                 f"licks at the same {coord} (Gaussian, sigma {F['sigma']:g} {unit});\nsums to the offset. Global = a "
                 "uniform cortex-wide change x the summed weight; pattern = the rest; read-out = the like-for-like "
                 "change (offset + held-out pre-reference term). Mean of animals; ipsi (lesion) = image LEFT. "
                 "SUPPLEMENTARY, PROVISIONAL", fontsize=9)
    fig.savefig(FIG / f"lick_offset_contrib_{coord}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def plot_vs_lateral(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = F["vs_lateral"]
    T, Rm, LS, R = V["terms"], V["r_map"], V["lateral_summary"], F["summary"]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    cols = {"acute": "#d6604d", "subacute": "#b8860b", "chronic": "#1b7837"}
    for i, st in enumerate(SETS):
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            m = V["lateral_maps"][st][ep]
            lim = float(np.nanpercentile(np.abs(m), 99.5))
            ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.axis("off")
            ls = LS[(LS.set == st) & (LS.epoch == ep)]
            r = Rm[(Rm.set == st) & (Rm.epoch == ep)]
            ax.set_title(f"{SETS[st]} LATERAL {ep} - pre (mm)\noffset {ls.offset.mean():+.2f} (global "
                         f"{ls.global_part.mean():+.2f}); map r with angle\n{r.r_map.mean():+.2f} "
                         f"[{', '.join(f'{v:+.2f}' for v in r.r_map)}]", fontsize=8)
        ax = axs[i, 3]
        t = T[T.set == st].groupby(["feature", "epoch"])[["lateral", "angle"]].mean().reset_index()
        for ep in EP:
            q = t[t.epoch == ep]
            if len(q) > 2:
                rr = np.corrcoef(q.lateral, q.angle)[0, 1]
                ax.scatter(q.lateral, q.angle, s=14, color=cols[ep], label=f"{ep} (r {rr:+.2f})")
        top = t[t.epoch == "chronic"].assign(m=lambda d: d.angle.abs()).nlargest(6, "m")
        for q in top.itertuples():
            ax.annotate(_short(q.feature), (q.lateral, q.angle), fontsize=6)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("LATERAL per-area term (mm)", fontsize=8)
        ax.set_ylabel("ANGLE per-area term (deg)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        a = R[R.set == st].groupby("epoch")[["offset", "global_part"]].mean().reindex(EP)
        ax.set_title(f"{SETS[st]}: per-area terms, mean of animals\nangle offset (global) "
                     + ", ".join(f"{e[:3]} {a.loc[e, 'offset']:+.2f} ({a.loc[e, 'global_part']:+.2f})" for e in EP
                                 if e in a.index and np.isfinite(a.loc[e, 'offset'])), fontsize=8)
    fig.suptitle("ANGLE vs LATERAL offset contributions (same frozen-decoder logic, Allen areas + ALM / tjM1 split, "
                 "Gaussian same-place reference): left = the LATERAL contribution maps (mean of animals) with the "
                 "pixel-wise r to the ANGLE map\n(mean [per animal]); right = per-area terms (beta x dh), each dot one "
                 "area. + = contralesional in both. SUPPLEMENTARY, PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "lick_offset_contrib_angle_vs_lateral.png", dpi=90)
    plt.close(fig)


# unmasked LocaNMF ("locanmfF") dropped from the figure (Priya 10-28: superseded by the masked basis)
TAGS = [("areas", "Allen areas"), ("areas_splitr8", "Allen, ALM / tjM1 split"), ("locanmfFM", "LocaNMF, glue / MOB comps dropped"),
        ("svdF", "pre-stroke SVD"), ("svdF_k50", "pre-stroke SVD, first 50 components")]


def plot():
    global _MASK
    from wfield_local import beta_maps
    _MASK = np.asarray(beta_maps.brain_mask(), bool)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [(t, lab) for t, lab in TAGS if (OUT / f"maps_{t}.npz").exists()]
    R = pd.concat([pd.read_csv(OUT / f"summary_{t}.csv") for t, _lab in tags], ignore_index=True)   # lateral only
    global _IMG, _NM
    from iti_area_maps import split_image
    _IMG, _NM = split_image(il.labels()[0])                 # Allen areas with ALM / tjM1 / rest (ROI v8)
    for st, stlab in (("iti", "ITI licks"), ("trial", "cued licks")):
        wpix = {}
        # + a row with the SVD maps PROJECTED onto the Allen areas with ALM / tjM1 (Priya 10-28): per area, the sum of
        # the pixel contributions (same units as the area-basis rows' area totals) and of the pixel weights
        rows_ = list(tags) + [("svdF@areas", "pre-stroke SVD, summed per Allen area (ALM / tjM1 split)")]             if any(t == "svdF" for t, _l in tags) else list(tags)
        fig, axs = plt.subplots(len(rows_), 4, figsize=(15, 3.6 * len(rows_)), squeeze=False)
        for i, (t, lab) in enumerate(rows_):
            proj = t.endswith("@areas")
            t = t.replace("@areas", "")
            Z = np.load(OUT / f"maps_{t}.npz")
            ims = {ep: [Z[f"{st}|{a}|{ep}"] for a in AN if f"{st}|{a}|{ep}" in Z] for ep in [*EP, "w"]}
            mean = {ep: np.nanmean(np.stack(v), 0) for ep, v in ims.items() if v}
            # drawn maps masked to the brain mask (Priya 10-28: kept LocaNMF footprints extend past it, e.g. the bulbs;
            # 2-3 % of |contribution|) -- the offset / global / pattern numbers are unchanged (computed on all pixels)
            mean = {ep: np.where(_MASK, m_, np.nan) for ep, m_ in mean.items()}
            if proj:                                        # per-area SUM painted on the area (incl. the weights)
                pm = {}
                for ep, m_ in mean.items():
                    A_ = np.full(_IMG.shape, np.nan)
                    v_ = np.where(np.isfinite(m_), m_, 0)
                    for k_ in range(len(_NM)):
                        sel = _IMG == k_
                        if sel.any():
                            A_[sel] = v_[sel].sum()
                    pm[ep] = np.where(_MASK, A_, np.nan)
                mean = pm
            lim = np.nanpercentile(np.abs(np.stack([mean[e] for e in EP if e in mean])), 99.5)
            for j, ep in enumerate(EP):
                ax = axs[i, j]
                if ep in mean:
                    im = ax.imshow(mean[ep], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                    r = R[(R.source == t) & (R.set == st) & (R.epoch == ep)]
                    ax.set_title(f"{lab}: {ep} - pre\noffset {r.offset.mean():+.2f} mm (global {r.global_part.mean():+.2f}, "
                                 f"pattern {r.pattern_part.mean():+.2f})", fontsize=8)
                ax.axis("off")
            plt.colorbar(im, ax=axs[i, 2], fraction=0.04,
                         label=("mm per AREA (sum)" if proj else "mm per pixel") + " (+ = contralesional)")
            # 4th column: decoder WEIGHT PER AREA (Priya 10-28: "not weight per pixel; keep that supplemental") = the sum
            # of the pixel weights over each area (ALM / tjM1 / rest split, ROI v8) = the decoder's change for a uniform +1
            # change of that area (= the area's coefficient in the area basis); comparable across bases
            Wa = np.full(_IMG.shape, np.nan)
            for k_ in range(len(_NM)):
                sel = _IMG == k_
                if sel.any():
                    Wa[sel] = np.nansum(np.where(np.isfinite(mean["w"]), mean["w"], 0)[sel])
            Wa = np.where(_MASK, Wa, np.nan)
            if proj:                                        # already per-area sums in the projected row
                Wa = mean["w"]
            if not proj:
                wpix[t] = mean["w"]
            ax = axs[i, 3]
            wl = np.nanpercentile(np.abs(Wa), 99.5)
            im = ax.imshow(Wa, cmap="PuOr_r", vmin=-wl, vmax=wl, interpolation="nearest")
            ax.set_title(f"{lab}: decoder weight PER AREA (pre fit)", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.04, label="mm per unit area activity")
        fig.suptitle(f"NEURAL: where the same-place read-out offset comes from, {stlab}. Contribution per pixel = decoder "
                     "weight x activity change vs pre licks landing at the same place (Gaussian, 0.1 mm);\nsums to the "
                     "offset. Global = a uniform cortex-wide change x the summed weight; pattern = the rest. Mean of "
                     "animals; ipsi (lesion) = image LEFT; drawn inside the brain mask. Right column = decoder weight per AREA "
                     "(sum of its pixel weights).\nThe offset totals agree across bases; the weights share the broad lateral "
                     "structure but differ in detail (ridge weights are not unique). PROVISIONAL", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(FIG / f"lick_{'iti' if st == 'iti' else 'cued'}_offset_contrib_map.png", dpi=90)
        plt.close(fig)
        # SUPPLEMENT: the per-pixel weights (the former 4th column)
        fig, axs = plt.subplots(1, len(wpix), figsize=(4.2 * len(wpix), 4.2), squeeze=False)
        for j, (t, lab) in enumerate([x for x in tags if x[0] in wpix]):
            w_ = np.where(_MASK, wpix[t], np.nan)
            wl = np.nanpercentile(np.abs(w_), 99.5)
            im = axs[0, j].imshow(w_, cmap="PuOr_r", vmin=-wl, vmax=wl, interpolation="nearest")
            axs[0, j].set_title(f"{lab}:\ndecoder weight per PIXEL", fontsize=8)
            axs[0, j].axis("off")
            plt.colorbar(im, ax=axs[0, j], fraction=0.04)
        fig.suptitle(f"SUPPLEMENT: per-pixel decoder weights, {stlab} (mean of animals; a large area with a solid coefficient "
                     "looks pale because its weight is spread over many pixels -- see the per-area weights in the main figure)",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(FIG / f"lick_{'iti' if st == 'iti' else 'cued'}_offset_contrib_pixelweight.png", dpi=90)
        plt.close(fig)


def _coords() -> list[str]:
    if "--coord" not in sys.argv:
        return ["lateral"]
    i = sys.argv.index("--coord") + 1
    out = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        out.append(sys.argv[i])
        i += 1
    bad = [c for c in out if c != "lateral" and c not in POLAR]
    if bad or not out:
        raise SystemExit(f"--coord: expected lateral or {sorted(POLAR)}, got {out}")
    if "lateral" in out and len(out) > 1:
        raise SystemExit("--coord lateral runs alone (another decoder)")
    return out


def main() -> int:
    from figdata import load_fd, plot_only, save_fd
    coords = _coords()
    if coords == ["lateral"]:
        if not plot_only():
            compute()
        plot()
        return 0
    if plot_only():
        Fs = {c: load_fd(f"lick_offset_contrib_{c}") for c in coords}
    else:
        il.split_only("lick_offset_contrib_map --coord")          # polar outputs carry the split tag only
        res = compute_polar(coords)
        Fs = {}
        for c, (R, maps, T) in res["out"].items():
            Fs[c] = figure_data(c, R, maps, T, res["areas"])
            save_fd(f"lick_offset_contrib_{c}", Fs[c])
    for F in Fs.values():
        plot_polar(F)
        if "vs_lateral" in F:
            plot_vs_lateral(F)
            V = F["vs_lateral"]
            print("== angle vs lateral: pixel-wise r of the contribution maps, per animal")
            print(V["r_map"].pivot_table(index=["set", "animal"], columns="epoch", values="r_map")
                  .reindex(columns=EP).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
