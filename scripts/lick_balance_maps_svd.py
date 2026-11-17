"""PIXEL-level version of lick_balance_maps.py (Priya 10-28: "same with the SVD basis, not tied to Allen areas"): where on
the cortex does the per-lick hemispheric BALANCE carry TASK vs MOVEMENT information? Musall-style alone / unique cvR2.

Per lick (the frozen KEY-row decoder's lick set, `iti_lick_order.preds_path()`, matched to the session file's lick rows
by `il.subseq_match` exactly as in lick_balance_maps.py; cued and ITI licks separately):
  ACTIVITY of pixel p = drift-removed dF/F (x1e3) averaged over 0..0.2 s from the lick peak (`lick_balance_reference.W`
  on `il.LAGS_D`), built from the SVD exactly as `iti_licks.extract` builds the area features: V (adopted hemo-corrected
  SVTcorr) -> slow-drift projection removed (`ce._slow_basis`, DRIFT_MIN_PERIOD_S) -> linear interpolation at
  t_peak + lag -> window mean = a K-vector c per lick; pixel activity = U_atlas[p] . c.
  BALANCE_p = activity(p) - activity(mirror(p)) for p on the RIGHT (contralesional) hemisphere, mirror(p) = the pixel
  reflected across the brain midline. MIDLINE: the Allen area image's left / right hemisphere boundary sits at column
  319.5 in every row of the shared 540 x 640 grid (checked at run time; mirroring c -> 639 - c maps 98.6 % of
  left-area pixels onto the same right area, vs 95.8 % for allen_register.BREGMA_OFFSET's x = 320, i.e. c -> 640 - c).
  GRID: the shared Allen grid (U_atlas is atlas-registered; every session's brain pixels lie inside
  `beta_maps.brain_mask()`), downsampled by DS (default 2; block mean of U over the block's in-mask pixels -- linear, so
  a block's activity is the mean of its pixels' activity). With an even width the midline lies on a block edge, so
  mirror blocks are exact. A right block is kept when it and its mirror block hold >= half their pixels in the brain
  mask and every session has data in both.
Regressors (identical to lick_balance_maps.py): TASK = spout one-hot (cued: target; ITI: the PREVIOUS trial's spout);
MOVEMENT = lateral angle (asin(side / reach)), 3-D reach, peak tongue speed (covref covariates file), z-scored within the
animal x epoch x set group. Per animal x epoch x set (>= 2 sessions, >= 100 licks): OLS + intercept, leave-one-SESSION-
out cvR2 per pixel (pooled SSE / SST over held-out licks) of FULL, TASK alone, MOVEMENT alone; TASK unique = full -
movement alone; MOVEMENT unique = full - task alone. EXACT in c-space: Y_s = C_s D_s' (D_s = U_right - U_mirror of
session s), so X'Y, the held-out SSE and the SST are all formed from K x K / q x K session sums, never the lick x pixel
matrix. Maps = mean over animals; each right pixel's value is painted on BOTH the pixel and its mirror (a pair
property, as in the area version). Image left = ipsilesional.
Outputs <OUT_W>/balmap_svd_maps.npz (per animal maps), balmap_svd_area_summary.csv (per animal x epoch x set x area:
mean pixel cvR2 over the area's right pixels), balmap_svd_validation.csv; figures <OUT_W>/figures/
balmap_svd_{cued,iti}.png, balmap_svd_pre_move_alone_per_animal.png; figdata balmap_svd_maps (--plot-only).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402
from lick_balance_maps import MEASURES, SETS  # noqa: E402

DS = int(os.environ.get("BALMAP_SVD_DS", "2"))
JOBS = 6                                     # U + V per worker (~1.5 GB each)
LAGS_W = il.LAGS_D[lbr.W]                    # 0, 0.05, ..., 0.2 s
PREFIX = "balmap_svd_"


# ---------------------------------------------------------------------------------------------------------------- grid
def _grid() -> dict:
    """Downsampled shared grid: block of every pixel, the kept right blocks and their mirrors."""
    from wfield_local import beta_maps
    bm = beta_maps.brain_mask()
    H, W = bm.shape
    if H % DS or W % DS or (W // DS) % 2:
        raise RuntimeError(f"grid {bm.shape} not divisible by DS={DS} with an even block width")
    h, w = H // DS, W // DS
    rr, cc = np.divmod(np.arange(H * W), W)
    blk = (rr // DS) * w + (cc // DS)                   # block id of every full-res pixel
    frac = np.bincount(blk, bm.ravel().astype(float), h * w) / DS ** 2
    good = frac >= 0.5
    br, bc = np.divmod(np.arange(h * w), w)
    mir = br * w + (w - 1 - bc)                          # column c -> 639 - c  <=>  block column j -> w - 1 - j
    right = np.flatnonzero(good & (bc >= w // 2) & good[mir])
    return {"H": H, "W": W, "h": h, "w": w, "blk": blk, "right": right, "mirror": mir[right]}


def _check_midline(img: np.ndarray, nm: list) -> float:
    hemi = np.zeros(img.shape)
    for i, n in enumerate(nm):
        hemi[img == i] = -1 if n.endswith("_left") else (1 if n.endswith("_right") else 0)
    mids = [(np.flatnonzero(r < 0).max() + np.flatnonzero(r > 0).min()) / 2.0 for r in hemi
            if (r < 0).any() and (r > 0).any()]
    mid = float(np.median(mids))
    if abs(mid - (img.shape[1] - 1) / 2.0) > 0.01 or np.ptp(mids) > 0:
        raise RuntimeError(f"atlas midline at {mid} (range {np.ptp(mids)}), not column {(img.shape[1] - 1) / 2}")
    return mid


# ----------------------------------------------------------------------------------------------------------- per session
def _window_c(V: np.ndarray, ft: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(len(t), K) drift-removed temporal components (x1e3) averaged over the 0..0.2 s window after each t."""
    from wfield_local import combined_encoding as ce
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    Vd = V.astype(np.float64)
    Vd -= Q @ (Q.T @ Vd)
    Vd *= 1e3
    tt = (t[:, None] + LAGS_W[None, :]).ravel()
    out = np.empty((len(tt), Vd.shape[1]))
    for j in range(Vd.shape[1]):
        out[:, j] = np.interp(tt, ft, Vd[:, j])
    return out.reshape(len(t), len(LAGS_W), -1).mean(1)


def _block_U(Ub: np.ndarray, flat: np.ndarray, G: dict) -> np.ndarray:
    """(h*w, K) block mean of U over the session's in-mask pixels; NaN for blocks with none."""
    b = G["blk"][flat]
    n = np.bincount(b, minlength=G["h"] * G["w"]).astype(float)
    S = np.zeros((G["h"] * G["w"], Ub.shape[1]))
    np.add.at(S, b, Ub.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / n[:, None]


def _session(label: str) -> dict:
    """One session (module-level for spawn) -> per set: C (n preds, K; NaN = unmatched) + D (right blocks, K)."""
    from lick_covariate_reference import _acts

    from wfield_local import combined_encoding as ce
    G = _grid()
    V, ft, Ub, flat, _pa, _nm, shape, _ps = ce.load_imaging(label)
    if tuple(shape) != (G["H"], G["W"]):
        raise RuntimeError(f"{label}: grid {shape}")
    Ublk = _block_U(Ub, flat, G)
    D = (Ublk[G["right"]] - Ublk[G["mirror"]]).astype(np.float32)
    acts = _acts(label)
    d = il._load(label)
    out = {"D": D, "C": {}}
    for st, (tk, sk) in {"iti": ("iti_t", "iti_side"), "trial": ("tr_t", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        C = np.full((len(act), V.shape[1]), np.nan)
        if (m >= 0).any():
            C[m >= 0] = _window_c(V, ft, d[tk].astype(float)[m[m >= 0]])
        out["C"][st] = C
    return out


# ------------------------------------------------------------------------------------------------------------- the fit
def _cv_r2_csp(Xs: list, Cs: list, Ds: list, cols: list[np.ndarray]) -> list[np.ndarray]:
    """Leave-one-session-out cvR2 per pixel of Y_s = C_s D_s' for OLS on each design-column subset `cols` of the full
    design (X_s INCLUDING the intercept column 0); pooled SSE / SST over held-out licks (== lick_balance_maps._cv_r2
    on the materialized lick x pixel matrix)."""
    S = len(Xs)
    N = sum(len(x) for x in Xs)
    CtC = [c.T @ c for c in Cs]
    yy = [np.einsum("pk,kl,pl->p", D, M, D, optimize=True) for D, M in zip(Ds, CtC)]   # sum_i y_ip^2 per session
    ysum = sum(D @ c.sum(0) for D, c in zip(Ds, Cs))
    sst = sum(yy) - ysum ** 2 / N
    XtY = [(x.T @ c) @ D.T for x, c, D in zip(Xs, Cs, Ds)]      # (q, npix) per session
    XtX = [x.T @ x for x in Xs]
    XtC = [x.T @ c for x, c in zip(Xs, Cs)]
    XtY_all, XtX_all = sum(XtY), sum(XtX)
    out = []
    for sel in cols:
        ix = np.ix_(sel, sel)
        sse = np.zeros(len(sst))
        for t in range(S):
            B = np.linalg.lstsq((XtX_all - XtX[t])[ix], (XtY_all - XtY[t])[sel], rcond=None)[0]   # (q', npix)
            cross = np.einsum("pk,qk,qp->p", Ds[t], XtC[t][sel], B, optimize=True)
            quad = np.einsum("qp,qr,rp->p", B, XtX[t][ix], B, optimize=True)
            sse += yy[t] - 2 * cross + quad
        out.append(1 - sse / sst)
    return out


def _validate(label: str, G: dict) -> pd.DataFrame:
    """Pixel activity averaged over each Allen ROI (iti_licks.roi_masks, full resolution) vs the session file's area
    feature (iti_X window mean); and the DS-grid pixel balance averaged over an area's right blocks vs the area balance."""
    from frozen_area_maps import area_image

    from wfield_local import combined_encoding as ce
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    d = il._load(label)
    c = _window_c(V, ft, d["iti_t"].astype(float))
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(d["cols"])
    X = d["iti_X"].astype(float)[:, lbr.W, :].mean(1)
    rows = []
    for j, n in enumerate(cols):
        a = c @ Ub[M[n]].astype(np.float64).mean(0)
        e = a - X[:, j]
        rows.append({"label": label, "check": "area_mean", "area": n, "n": len(a), "max_abs_err": np.abs(e).max(),
                     "rms_err": np.sqrt(np.mean(e ** 2)), "sd_feature": X[:, j].std(),
                     "rel_rms": np.sqrt(np.mean(e ** 2)) / X[:, j].std(), "r": np.corrcoef(a, X[:, j])[0, 1]})
    Ublk = _block_U(Ub, flat, G)
    bal = c @ (Ublk[G["right"]] - Ublk[G["mirror"]]).T              # (licks, right blocks)
    img, nm = area_image(label)
    _check_midline(img, nm)
    for a in ("ALM", "tjM1", "SSp-m", "SSp-n", "SSp-bfd", "VISp", "RSPd"):
        if f"{a}_right" not in M or f"{a}_right" not in cols or f"{a}_left" not in cols:
            continue
        sel = np.zeros(G["h"] * G["w"], bool)                       # blocks holding the right ROI's pixels
        sel[np.unique(G["blk"][flat[M[f"{a}_right"]]])] = True
        in_area = sel[G["right"]] & np.isfinite(bal).all(0)
        pb = bal[:, in_area].mean(1)
        ab = X[:, cols.index(f"{a}_right")] - X[:, cols.index(f"{a}_left")]
        rows.append({"label": label, "check": f"ds{DS}_balance", "area": a, "n": len(pb), "n_blocks": int(in_area.sum()),
                     "r": np.corrcoef(pb, ab)[0, 1], "rms_err": np.sqrt(np.mean((pb - ab) ** 2)), "sd_feature": ab.std(),
                     "rel_rms": np.sqrt(np.mean((pb - ab) ** 2)) / ab.std()})
    return pd.DataFrame(rows)


def compute() -> dict:
    from iti_area_maps import split_image

    from wfield_local import analysis_kit as ak
    G = _grid()
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz",
                    usecols=["set", "label", "k", "reach_mm", "speed_peak", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    labels = sorted(P.label.unique())

    V = _validate(labels[0], G)
    V.to_csv(il.OUT_W / f"{PREFIX}validation.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"== VALIDATION ({labels[0]}): per-lick pixel activity averaged over each ROI vs the session-file feature")
    a = V[V.check == "area_mean"]
    print(f"   {len(a)} ROIs: median r {a.r.median():.6f}, min r {a.r.min():.6f}; rel. RMS err median "
          f"{a.rel_rms.median():.2e}, max {a.rel_rms.max():.2e}; max |err| {a.max_abs_err.max():.3e} (x1e3 dF/F; "
          f"features stored float16)")
    print(V[V.check != "area_mean"][["area", "n_blocks", "r", "rel_rms"]].round(4).to_string())

    res, fail = ak.fan_sessions(labels, _session, jobs=JOBS)
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    S = dict(res)
    ok = np.ones(len(G["right"]), bool)
    for v in S.values():
        ok &= np.isfinite(v["D"]).all(1) & (np.abs(v["D"]).sum(1) > 0)   # D == 0: U zero on both sides (outside FOV)
    print(f"right blocks: {len(G['right'])} with both blocks in the mask; {ok.sum()} with data (finite, non-zero "
          f"balance loading) in every session (DS={DS})", flush=True)
    keep = np.flatnonzero(ok)

    maps, rows = {}, []
    for (st, an, ep), g in P.groupby(["set", "animal", "epoch"]):
        g = g.dropna(subset=["angle", "reach_mm", "speed_peak"])
        cm = np.array([np.isfinite(S[lab]["C"][st][k]).all() for lab, k in zip(g.label, g.k)], bool)
        g = g[cm]
        if g.label.nunique() < 2 or len(g) < 100:
            continue
        task = pd.get_dummies(g.pos.astype(str), drop_first=True).to_numpy(float)
        mv = g[["angle", "reach_mm", "speed_peak"]].to_numpy(float)
        mv = (mv - mv.mean(0)) / mv.std(0)
        X = np.column_stack([np.ones(len(g)), task, mv])
        nt = task.shape[1]
        full = np.arange(X.shape[1])
        tsel = np.arange(1 + nt)
        msel = np.r_[0, np.arange(1 + nt, X.shape[1])]
        Xs, Cs, Ds = [], [], []
        for lab in sorted(g.label.unique()):
            sel = (g.label == lab).to_numpy()
            Xs.append(X[sel])
            Cs.append(S[lab]["C"][st][g.k.to_numpy()[sel]])
            Ds.append(S[lab]["D"][keep].astype(np.float64))
        r_full, r_task, r_move = _cv_r2_csp(Xs, Cs, Ds, [full, tsel, msel])
        mp = {"full": r_full, "task_alone": r_task, "move_alone": r_move,
              "task_unique": r_full - r_move, "move_unique": r_full - r_task}
        maps[(SETS[st], an, ep)] = {m: mp[m].astype(np.float32) for m, _t in MEASURES}
        rows.append({"set": SETS[st], "animal": an, "epoch": ep, "n": len(g), "n_sess": g.label.nunique(),
                     **{f"{m}_median": float(np.median(mp[m])) for m, _t in MEASURES},
                     **{f"{m}_p95": float(np.percentile(mp[m], 95)) for m, _t in MEASURES}})
        print(f"  {SETS[st]:5s} {an} {ep:9s} n={len(g):6d} sessions={g.label.nunique():2d}  median px cvR2: full "
              f"{np.nanmedian(r_full):.3f}  move {np.nanmedian(r_move):.3f}  task {np.nanmedian(r_task):.3f}", flush=True)
    R = pd.DataFrame(rows)

    img, nm = split_image(labels[0])
    _check_midline(img, nm)
    bimg = img[DS // 2::DS, DS // 2::DS].ravel()[G["right"][keep]]
    arows = []
    for (st, an, ep), mp in maps.items():
        for i, n in enumerate(nm):
            if not n.endswith("_right"):
                continue
            sel = bimg == i
            if sel.sum() < 3:
                continue
            arows.append({"set": st, "animal": an, "epoch": ep, "area": n[:-6], "n_blocks": int(sel.sum()),
                          **{m: float(np.mean(mp[m][sel])) for m, _t in MEASURES}})
    A = pd.DataFrame(arows)
    A.to_csv(il.OUT_W / f"{PREFIX}area_summary.csv", index=False)
    R.to_csv(il.OUT_W / f"{PREFIX}groups.csv", index=False)
    key = A[A.area.isin(["ALM", "tjM1", "SSp-m", "SSp-n", "SSp-ul", "SSp-bfd", "VISp", "RSPd", "ACAd"])]
    print("== per animal (rule 8), PRE: movement alone / task alone, mean pixel cvR2 over the area's right blocks")
    print(key[key.epoch == "pre"].pivot_table(index=["set", "area"], columns="animal",
                                              values=["move_alone", "task_alone"]).round(3).to_string())
    print("== mean of animals per epoch (area means of pixel cvR2)")
    print(A.groupby(["set", "area", "epoch"])[["move_alone", "task_alone", "full"]].mean().unstack("epoch")
          .loc[lambda x: x.index.get_level_values(1).isin(["ALM", "tjM1", "SSp-m", "SSp-n", "SSp-bfd", "VISp"])]
          .round(3).to_string())
    np.savez_compressed(il.OUT_W / f"{PREFIX}maps.npz", right_blocks=G["right"][keep], mirror_blocks=G["mirror"][keep],
                        grid=np.array([G["h"], G["w"], DS]),
                        keys=np.array(["__".join(k) for k in maps]),
                        **{f"{'__'.join(k)}__{m}": v[m] for k, v in maps.items() for m, _t in MEASURES})
    return {"maps": maps, "img": img, "nm": nm, "right": G["right"][keep], "mirror": G["mirror"][keep],
            "h": G["h"], "w": G["w"], "ds": DS, "validation": V}


# ------------------------------------------------------------------------------------------------------------- figures
def _paint(F: dict, v: np.ndarray) -> np.ndarray:
    out = np.full(F["h"] * F["w"], np.nan)
    out[F["right"]] = v
    out[F["mirror"]] = v
    return np.repeat(np.repeat(out.reshape(F["h"], F["w"]), F["ds"], 0), F["ds"], 1)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline

    maps, ol = F["maps"], outline(F["img"])
    va = F["validation"]
    vtxt = va[va.check == "area_mean"]
    vline = (f"Validation ({vtxt.label.iloc[0]}, ITI licks): pixel activity averaged over each of {len(vtxt)} ROIs "
             f"reproduces the area feature, median r = {vtxt.r.median():.5f}, rel. RMS error "
             f"{vtxt.rel_rms.median():.1e}.")
    animals = sorted({k[1] for k in maps})
    for st in ("cued", "ITI"):
        fig, axs = plt.subplots(len(MEASURES), 4, figsize=(15, 3.6 * len(MEASURES)), squeeze=False)
        for i, (m, ttl) in enumerate(MEASURES):
            avg = {}
            for ep in il.EP4:
                vs = [maps[(st, an, ep)][m] for an in animals if (st, an, ep) in maps]
                if vs:
                    avg[ep] = (np.mean(vs, 0), len(vs))
            allv = np.concatenate([a for a, _n in avg.values()]) if avg else np.array([0.0])
            vmax = max(np.nanpercentile(allv, 99), 0.02)
            im = None
            for j, ep in enumerate(il.EP4):
                ax = axs[i, j]
                ax.axis("off")
                if ep not in avg:
                    ax.set_title(f"{ttl}\n{ep}: no animal with >= 2 sessions", fontsize=8.5)
                    continue
                im = ax.imshow(_paint(F, avg[ep][0]), cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                ax.set_title(f"{ttl}\n{ep} (n = {avg[ep][1]} animals)", fontsize=8.5)
            if im is not None:
                fig.colorbar(im, ax=axs[i, :], fraction=0.015, pad=0.01).set_label("cvR2 of the pixel balance", fontsize=8)
        fig.suptitle(f"{st.upper()} licks, PIXEL level (SVD): how much of each pixel pair's hemispheric BALANCE "
                     f"(activity(p) - activity(mirror p), 0..0.2 s after the lick peak, x1e3 dF/F) is explained by TASK "
                     "(spout) vs MOVEMENT (lick angle, reach, speed); Musall-style alone / unique cvR2 "
                     "(leave-one-session-out, per animal x epoch; mean of animals).\nEach right-hemisphere pixel's "
                     f"value is painted on it AND its mirror (pair property); {F['ds']}x{F['ds']} blocks; mirror = "
                     "reflection about the atlas midline (column 319.5); image left = ipsilesional; Allen outlines. "
                     f"PROVISIONAL\n{vline}", fontsize=9.5)
        from wfield_local import trial_set_text as tst
        with tst.active(None, text=trial_set_sentence(st)):        # this figure's lick set only
            fig.savefig(il.OUT_W / "figures" / f"{PREFIX}{st.lower()}.png", dpi=95, bbox_inches="tight")
        plt.close(fig)

    fig, axs = plt.subplots(2, len(animals), figsize=(3.8 * len(animals), 7.4), squeeze=False)
    for i, st in enumerate(("cued", "ITI")):
        vs = [maps[(st, an, "pre")]["move_alone"] for an in animals if (st, an, "pre") in maps]
        vmax = max(np.nanpercentile(np.concatenate(vs), 99), 0.02) if vs else 0.1
        im = None
        for j, an in enumerate(animals):
            ax = axs[i, j]
            ax.axis("off")
            if (st, an, "pre") not in maps:
                ax.set_title(f"{an} {st}: pre not fitted (< 2 sessions)", fontsize=8.5)
                continue
            im = ax.imshow(_paint(F, maps[(st, an, "pre")]["move_alone"]), cmap="magma", vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.set_title(f"{an}, {st} licks, PRE", fontsize=9)
        if im is not None:
            fig.colorbar(im, ax=axs[i, :], fraction=0.015, pad=0.01).set_label("MOVEMENT alone cvR2", fontsize=8)
    fig.suptitle("Per animal (n = 4, before pooling): PRE-stroke MOVEMENT-alone cvR2 (angle, reach, speed) of the pixel "
                 "balance activity(p) - activity(mirror p), 0..0.2 s after the lick peak; leave-one-session-out.\n"
                 "Value painted on the pixel and its mirror; image left = ipsilesional. PROVISIONAL", fontsize=9.5)
    fig.savefig(il.OUT_W / "figures" / f"{PREFIX}pre_move_alone_per_animal.png", dpi=95, bbox_inches="tight")
    plt.close(fig)


def trial_set_sentence(st: str | None = None) -> str:
    """The lick set of the pixel balance maps (Priya 11-09): the KEY decoder's lick set matched to the session file rows,
    lick angle / reach / peak speed finite, animal x epoch x set groups with >= 2 sessions and >= 100 licks; ``st`` =
    "cued" / "ITI" (one figure) or None (both: the per-animal PRE figure)."""
    import lick_covariate_reference as cv
    keep = ["lick angle, reach and peak speed finite", "animal x epoch groups with >= 2 sessions and >= 100 licks"]
    sets = {"cued": ("trial",), "ITI": ("iti",), None: ("iti", "trial")}[st]
    return cv.key_lick_set(iti_extra=keep, cued_extra=keep, sets=sets)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    if plot_only():
        plot(load_fd(f"{PREFIX}maps"))
        return 0
    F = compute()
    save_fd(f"{PREFIX}maps", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
