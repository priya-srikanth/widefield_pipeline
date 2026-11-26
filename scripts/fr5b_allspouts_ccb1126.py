"""Fig 5B angle-matched across ALL SIX spouts (CC-B 2026-11-26; Priya 11-26). Same rule as the far-contra variant
(`fr5b_variants_ccb1126.py`), applied per animal AND per spout: 5-deg bins of the 3-D executed angle; a post-stroke
lick weighs p_pre(bin) / p_epoch(bin), with p_pre from that animal's pre licks AT THAT SPOUT; a bin needs >= 5 post
licks; an animal x spout x epoch is used only if its bins cover >= 50 % of that spout's pre mass. Pre licks weigh 1.
Post-cue licks only (pre-cue licks have no 3-D angle). Everything else as live 5B (window -0.1..+0.2 s, 108 sessions,
log2 vs own pre, 10,000 within-animal permutations via `face_rhythm_aligned_maps.py --perm-points`).

Input: the per-lick cache of `fr5b_variants_ccb1126.py licks --all-positions` (<cache>/perlick6/<key>.npz).

    python scripts/fr5b_allspouts_ccb1126.py coverage --cache <dir> --out <new dir>     # coverage table only
    python scripts/fr5b_allspouts_ccb1126.py maps --cache <dir> --out <same dir>        # maps npz (needs coverage)
    face_rhythm_aligned_maps.py --maps-npz <out>/data/fr5b_allspouts_maps_ccB1126.npz --out <out> --aligns cue_wp
        lick_4c_wp lick_post lick_post_ang lick_matched --align-names cue lick post post_ang matched --perm 10000
        --perm-points
    python scripts/fr5b_allspouts_ccb1126.py summary --out <dir>

Aligns of the maps npz: cue_wp, lick_4c_wp (copied from production: the permutation run then reproduces the
production rows), lick_post (post-cue licks, all; = production lick_4c_wpp, plumbing check), lick_post_ang (post-cue
licks with a 3-D angle, unweighted: the reference the weighting starts from), lick_matched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fr5b_variants_ccb1126 import (ALL_POS, EPOCHS, LICKS3D, PROD_NPZ, angle_bins,  # noqa: E402
                                           match_weights)

POS_NAME = dict(zip(ALL_POS, ("near ipsi", "near mid", "near contra", "far ipsi", "far mid", "far contra")))
NEW = ("lick_post", "lick_post_ang", "lick_matched")
TAG = "ccB1126"


def load_meta(cache: Path, keys) -> pd.DataFrame:
    rows = []
    for i, k in enumerate(keys):
        with np.load(cache / "perlick6" / f"{k}.npz") as z:
            rows.append(pd.DataFrame({"row": i, "key": k, "label": str(z["label"]), "animal": str(z["animal"]),
                                      "epoch": str(z["epoch"]), "date": str(z["date"]), "position": z["position"],
                                      "lick_key": z["lick_key"], "rel_s": z["rel_s"], "nf": z["nf"]}))
    T = pd.concat(rows, ignore_index=True)
    T["session"] = T.animal + "_" + T.date
    K = pd.read_csv(LICKS3D, usecols=["session", "lick_key", "position", "side_ml", "out_ap", "down_dv"])
    K = K.drop_duplicates(["session", "lick_key"]).rename(columns={"position": "position_3d"})
    K["angle_deg"] = np.degrees(np.arctan2(K.side_ml, np.hypot(K.out_ap, K.down_dv)))   # + = contralesional
    T = T.merge(K[["session", "lick_key", "position_3d", "angle_deg"]], on=["session", "lick_key"], how="left",
                validate="many_to_one")
    T["post"] = T.rel_s >= 0
    T["has_angle"] = T.angle_deg.notna()
    return T


def weights(T: pd.DataFrame, a) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = angle_bins(a.bin_deg)
    T = T.copy()
    T["w_match"] = 0.0
    covs = []
    for p in ALL_POS:
        TA = T[(T.position == p) & T.post & T.has_angle]
        w, cov = match_weights(TA, edges, a.min_bin, a.min_cover)
        T.loc[TA.index, "w_match"] = w
        covs.append(cov.assign(position=p, spout=POS_NAME[p]))
    C = pd.concat(covs, ignore_index=True)
    C["position"] = pd.Categorical(C.position, ALL_POS)
    C["epoch"] = pd.Categorical(C.epoch, EPOCHS)
    return T, C.sort_values(["position", "animal", "epoch"]).reset_index(drop=True)


GRID_STEP = 0.05                       # deg; KDE evaluated on this grid (histogram convolved with the kernel)
GRID = np.arange(-95.0, 95.0 + GRID_STEP / 2, GRID_STEP)


def kde_grid(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian KDE (fixed bandwidth ``sigma`` deg, unit mass) of the angles ``x`` on GRID: the histogram at
    GRID_STEP convolved with the normalized kernel (exact up to the 0.05-deg binning). Pure."""
    edges = np.r_[GRID - GRID_STEP / 2, GRID[-1] + GRID_STEP / 2]
    h = np.histogram(np.clip(x, GRID[0], GRID[-1]), edges)[0].astype(float)
    u = np.arange(-int(5 * sigma / GRID_STEP), int(5 * sigma / GRID_STEP) + 1) * GRID_STEP
    k = np.exp(-0.5 * (u / sigma) ** 2)
    k /= k.sum() * GRID_STEP
    return np.convolve(h / max(len(x), 1), k, mode="same")


def gauss_weights(T: pd.DataFrame, sigma: float, cap_q: float = 0.99, min_cover: float = 0.5,
                  min_neff: float = 20.0, min_near: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """Priya 11-26 Gaussian matching, per animal (call per spout): pre licks weigh 1; a post lick of epoch e weighs
    f_pre(theta) / f_e(theta) (fixed-bandwidth Gaussian KDEs, ``sigma`` deg, f_e from that epoch's own licks), capped at
    the ``cap_q`` quantile of the animal x epoch's weights, then scaled so the weights sum to the raw post count.
    Coverage = the share of the pre density lying within +- 2 sigma of >= ``min_near`` post licks; an animal x epoch is
    used when coverage >= ``min_cover`` AND n_eff = (sum w)^2 / sum w^2 >= ``min_neff`` (else all its weights 0). Pure."""
    w = pd.Series(0.0, index=T.index)
    rows = []
    for an, g in T.groupby("animal"):
        pre = g[g.epoch == "pre"]
        w[pre.index] = 1.0
        if not len(pre):
            continue
        fpre = kde_grid(pre.angle_deg.to_numpy(float), sigma)
        for ep in EPOCHS[1:]:
            q = g[g.epoch == ep]
            if not len(q):
                continue
            th = q.angle_deg.to_numpy(float)
            fe = kde_grid(th, sigma)
            ww = np.interp(th, GRID, fpre) / np.maximum(np.interp(th, GRID, fe), 1e-300)
            cap = float(np.quantile(ww, cap_q))
            ww = np.minimum(ww, cap)
            ww = ww * len(ww) / ww.sum() if ww.sum() > 0 else ww
            n_eff = float(ww.sum() ** 2 / (ww ** 2).sum()) if (ww > 0).any() else 0.0
            ths = np.sort(th)
            near = (np.searchsorted(ths, GRID + 2 * sigma, side="right") - np.searchsorted(ths, GRID - 2 * sigma))
            cover = float((fpre * (near >= min_near)).sum() * GRID_STEP)
            used = bool(cover >= min_cover and n_eff >= min_neff)
            w[q.index] = ww * float(used)
            rows.append({"animal": an, "epoch": ep, "sigma": sigma, "used": used, "n_pre": len(pre), "n_post": len(q),
                         "coverage": cover, "n_eff": n_eff, "n_eff_frac": n_eff / len(q), "cap": cap,
                         "max_weight": float(ww.max()), "pre_median_deg": float(pre.angle_deg.median()),
                         "post_median_deg": float(np.median(th)),
                         "matched_post_median_deg": float(_wmed(th, ww)), "n_sessions": q.label.nunique()})
    return w, pd.DataFrame(rows)


def _wmed(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)] if c[-1] > 0 else np.nan


def weights_gauss(T: pd.DataFrame, sigma: float, positions=ALL_POS) -> tuple[np.ndarray, pd.DataFrame]:
    W = np.zeros(len(T))
    covs = []
    for p in positions:
        TA = T[(T.position == p) & T.post & T.has_angle]
        w, cov = gauss_weights(TA, sigma)
        W[TA.index.to_numpy()] = w.to_numpy()
        covs.append(cov.assign(position=p, spout=POS_NAME[p]))
    C = pd.concat(covs, ignore_index=True)
    C["position"] = pd.Categorical(C.position, ALL_POS)
    C["epoch"] = pd.Categorical(C.epoch, EPOCHS)
    return W, C.sort_values(["sigma", "position", "animal", "epoch"]).reset_index(drop=True)


def cmd_gauss(a) -> int:
    """Coverage first (printed + written); then, unless --coverage-only, the maps npz with aligns cue_wp, lick_4c_wp
    (production copies), lick_post_ang (unweighted reference), lick_matched (hard bins, comparison), lick_gauss4 (primary)
    and the sigma sensitivities."""
    cache, out = Path(a.cache), Path(a.out)
    Z = np.load(PROD_NPZ)
    keys = list(Z["key"].astype(str))
    T = load_meta(cache, keys)
    cov_f = out / "data" / f"fr5b_gauss_coverage_{TAG}.csv"
    if a.coverage_only:
        if out.exists():
            raise SystemExit(f"refusing: {out} exists (a run writes only to its own new folder)")
        (out / "data").mkdir(parents=True)
    Ws, Cs = {}, []
    for s in a.sigmas:
        Ws[s], C = weights_gauss(T, s)
        Cs.append(C)
    C = pd.concat(Cs, ignore_index=True)
    if a.coverage_only:
        C.to_csv(cov_f, index=False, float_format="%.4f")
        pd.set_option("display.width", 220)
        show = C[C.sigma == a.sigmas[0]][["spout", "animal", "epoch", "used", "n_post", "coverage", "n_eff", "max_weight",
                                          "pre_median_deg", "post_median_deg", "matched_post_median_deg"]]
        print(show.round(3).to_string(index=False))
        for s in a.sigmas:
            print(f"sigma {s}: animals usable per spout x epoch\n",
                  C[C.sigma == s].groupby(["position", "epoch"], observed=True).used.sum().unstack().to_string())
        return 0
    npz = out / "data" / f"fr5b_gauss_maps_{TAG}.npz"
    if npz.exists():
        raise SystemExit(f"refusing to overwrite {npz}")
    Th, _ = weights(T, a)                                    # hard-bin rule (comparison), same as the hard-bin run
    sets = {"lick_post_ang": (T.post & T.has_angle).to_numpy(float), "lick_matched": Th.w_match.to_numpy(float)}
    for s in a.sigmas:
        sets[f"lick_gauss{s:g}"] = Ws[s]
    new = list(sets)
    positions = list(Z["positions"].astype(str))
    prod_aligns = list(Z["aligns"].astype(str))
    aligns = ["cue_wp", "lick_4c_wp", *new]
    nS, P = len(keys), Z["maps"].shape[-1]
    maps = np.full((nS, len(aligns), 6, P), np.nan, np.float32)
    counts = np.zeros((nS, len(aligns), 6), int)
    for j, al in enumerate(("cue_wp", "lick_4c_wp")):
        maps[:, j] = Z["maps"][:, prod_aligns.index(al)]
        counts[:, j] = Z["counts"][:, prod_aligns.index(al)]
    nf = T.nf.to_numpy(float)
    rowv, posv = T.row.to_numpy(), T.position.to_numpy()
    for i, k in enumerate(keys):
        with np.load(cache / "perlick6" / f"{k}.npz") as z:
            S1 = z["S1"].astype(np.float64)
        ii_s = np.flatnonzero(rowv == i)
        for pi, p in enumerate(positions):
            jj = ii_s[posv[ii_s] == p]
            for j, al in enumerate(new, start=2):
                w = sets[al][jj]
                m = w > 0
                counts[i, j, pi] = int(m.sum())
                den = (w[m] * nf[jj][m]).sum()
                if den > 0:
                    maps[i, j, pi] = (w[m, None] * S1[jj[m] - ii_s[0]]).sum(0) / den
    wins = np.array([(0.0, 2.0)] + [(-0.1, 0.2)] * (len(aligns) - 1))
    np.savez(npz, maps=maps, counts=counts, key=Z["key"], label=Z["label"], animal=Z["animal"], epoch=Z["epoch"],
             aligns=np.array(aligns), positions=Z["positions"], windows=wins, home=Z["home"],
             tmpl=Z["tmpl"], speed_response=Z["speed_response"])
    for s in a.sigmas:
        T[f"w_gauss{s:g}"] = Ws[s]
    T["w_hard"] = Th.w_match.to_numpy()
    T.drop(columns=["row"]).to_csv(out / "data" / f"fr5b_gauss_licks_{TAG}.csv.gz", index=False)
    print("aligns:", aligns, "->", npz)
    return 0


def pooled_weights(T: pd.DataFrame, sigma: float, cap_q: float = 0.99, min_cover: float = 0.5,
                   min_neff: float = 20.0, min_near: int = 5) -> tuple[dict, pd.DataFrame]:
    """Priya 11-26 (correction): POOLED pre reference, matched for angle but NOT for target spout. Per animal x epoch x
    target-spout row: the row's post licks (post-cue, with a 3-D angle) weigh 1; ALL of the animal's pre post-cue licks
    with an angle (every spout) weigh f_post,row(theta) / f_pre,all(theta) (fixed-bandwidth Gaussian KDEs), capped at the
    ``cap_q`` quantile and scaled to sum to the pre count. Coverage = share of the post row's density lying within
    +- 2 sigma of >= ``min_near`` pre licks; used when coverage >= ``min_cover`` AND n_eff (weighted pre set) >=
    ``min_neff``. -> ({(epoch, position): weight vector over T (pre licks; 0 elsewhere / unused)}, table). Pure."""
    elig = (T.post & T.has_angle).to_numpy()
    Wd = {(e, p): np.zeros(len(T)) for e in EPOCHS[1:] for p in ALL_POS}
    rows = []
    for an in sorted(T.animal.unique()):
        ia = np.flatnonzero((T.animal == an).to_numpy() & elig)
        pre_i = ia[(T.epoch.to_numpy()[ia] == "pre")]
        if not len(pre_i):
            continue
        thp = T.angle_deg.to_numpy(float)[pre_i]
        fpre = kde_grid(thp, sigma)
        ths = np.sort(thp)
        near = np.searchsorted(ths, GRID + 2 * sigma, side="right") - np.searchsorted(ths, GRID - 2 * sigma)
        for ep in EPOCHS[1:]:
            for p in ALL_POS:
                q_i = ia[(T.epoch.to_numpy()[ia] == ep) & (T.position.to_numpy()[ia] == p)]
                if not len(q_i):
                    continue
                thq = T.angle_deg.to_numpy(float)[q_i]
                fq = kde_grid(thq, sigma)
                ww = np.interp(thp, GRID, fq) / np.maximum(np.interp(thp, GRID, fpre), 1e-300)
                cap = float(np.quantile(ww, cap_q))
                ww = np.minimum(ww, cap)
                ww = ww * len(ww) / ww.sum() if ww.sum() > 0 else ww
                n_eff = float(ww.sum() ** 2 / (ww ** 2).sum()) if (ww > 0).any() else 0.0
                cover = float((fq * (near >= min_near)).sum() * GRID_STEP)
                used = bool(cover >= min_cover and n_eff >= min_neff)
                if used:
                    Wd[(ep, p)][pre_i] = ww
                same = T.position.to_numpy()[pre_i] == p
                rows.append({"sigma": sigma, "position": p, "spout": POS_NAME[p], "animal": an, "epoch": ep,
                             "used": used, "n_post_row": len(q_i), "n_pre_all": len(pre_i), "coverage": cover,
                             "n_eff_pre": n_eff, "n_eff_frac": n_eff / len(pre_i), "cap": cap,
                             "post_median_deg": float(np.median(thq)), "pre_all_median_deg": float(np.median(thp)),
                             "matched_pre_median_deg": float(_wmed(thp, ww)),
                             "pre_weight_share_same_spout": float(ww[same].sum() / ww.sum()) if ww.sum() > 0 else np.nan,
                             "n_sessions_post": len(set(T.label.to_numpy()[q_i]))})
    C = pd.DataFrame(rows)
    C["position"] = pd.Categorical(C.position, ALL_POS)
    C["epoch"] = pd.Categorical(C.epoch, EPOCHS)
    return Wd, C.sort_values(["position", "animal", "epoch"]).reset_index(drop=True)


def cmd_pooled(a) -> int:
    """Pooled-reference matching: coverage table first; then (unless --coverage-only) a maps npz. The pre reference
    depends on the epoch, so each (sigma, epoch) is its own align ``pr<sigma>_<epoch>``: pre sessions carry the
    weighted pooled-pre map of that row, sessions of that epoch carry the row's post licks (weight 1), all other
    sessions are empty. `face_rhythm_aligned_maps.py --perm` then shuffles SESSION labels within animal between the
    pre sessions (weighted reference maps, weights attached to their licks) and the epoch's sessions -- the live 5B
    session-level null with the weighting carried by each session's map. Also aligns lick_post_ang (unweighted),
    lick_matched (per-spout hard bins) and lick_gauss4 (per-spout Gaussian) for comparison."""
    cache, out = Path(a.cache), Path(a.out)
    Z = np.load(PROD_NPZ)
    keys = list(Z["key"].astype(str))
    T = load_meta(cache, keys)
    if a.coverage_only:
        if out.exists():
            raise SystemExit(f"refusing: {out} exists (a run writes only to its own new folder)")
        (out / "data").mkdir(parents=True)
    Wp, Cs = {}, []
    for s in a.sigmas:
        Wp[s], C = pooled_weights(T, s)
        Cs.append(C)
    C = pd.concat(Cs, ignore_index=True)
    if a.coverage_only:
        C.to_csv(out / "data" / f"fr5b_pooledref_coverage_{TAG}.csv", index=False, float_format="%.4f")
        pd.set_option("display.width", 230)
        print(C[C.sigma == a.sigmas[0]][["spout", "animal", "epoch", "used", "n_post_row", "coverage", "n_eff_pre",
                                         "post_median_deg", "matched_pre_median_deg", "pre_weight_share_same_spout"]]
              .round(3).to_string(index=False))
        for s in a.sigmas:
            print(f"sigma {s}: animals usable per spout x epoch\n",
                  C[C.sigma == s].groupby(["position", "epoch"], observed=True).used.sum().unstack().to_string())
        return 0
    npz = out / "data" / f"fr5b_pooledref_maps_{TAG}.npz"
    if npz.exists():
        raise SystemExit(f"refusing to overwrite {npz}")
    elig = (T.post & T.has_angle).to_numpy()
    Th, _ = weights(T, a)
    Wg, Cg = weights_gauss(T, a.sigmas[0])
    Cg.to_csv(out / "data" / f"fr5b_gauss_perspout_coverage_{TAG}.csv", index=False, float_format="%.4f")
    ep_v, pos_v, an_v = T.epoch.to_numpy(), T.position.to_numpy(), T.animal.to_numpy()
    # per-lick weight per (align, position): callables give the weight vector for one position
    simple = {"lick_post_ang": elig.astype(float), "lick_matched": Th.w_match.to_numpy(float), "lick_gauss4": Wg}
    used = {(s, r.epoch, r.position, r.animal): r.used for s, Cx in zip(a.sigmas, Cs) for r in Cx.itertuples()}
    aligns = ["cue_wp", "lick_4c_wp", *simple, *[f"pr{s:g}_{e}" for s in a.sigmas for e in EPOCHS[1:]]]
    positions = list(Z["positions"].astype(str))
    prod_aligns = list(Z["aligns"].astype(str))
    nS, P = len(keys), Z["maps"].shape[-1]
    maps = np.full((nS, len(aligns), 6, P), np.nan, np.float32)
    counts = np.zeros((nS, len(aligns), 6), int)
    for j, al in enumerate(("cue_wp", "lick_4c_wp")):
        maps[:, j] = Z["maps"][:, prod_aligns.index(al)]
        counts[:, j] = Z["counts"][:, prod_aligns.index(al)]

    def wvec(al, p):
        if al in simple:
            return np.where(pos_v == p, simple[al], 0.0)
        s, e = al[2:].split("_")
        s = float(s)
        post = (ep_v == e) & (pos_v == p) & elig
        ok = np.array([used.get((s, e, p, an), False) for an in an_v])   # animal x epoch x row usable
        return np.where(ep_v == "pre", Wp[s][(e, p)], 0.0) + (post & ok).astype(float)

    nf = T.nf.to_numpy(float)
    rowv = T.row.to_numpy()
    W = {(al, p): wvec(al, p) for al in aligns[2:] for p in positions}
    for i, k in enumerate(keys):
        with np.load(cache / "perlick6" / f"{k}.npz") as z:
            S1 = z["S1"].astype(np.float64)
        ii_s = np.flatnonzero(rowv == i)
        for (al, p), wv in W.items():
            j, pi = aligns.index(al), positions.index(p)
            w = wv[ii_s]
            m = w > 0
            counts[i, j, pi] = int(m.sum())
            den = (w[m] * nf[ii_s][m]).sum()
            if den > 0:
                maps[i, j, pi] = (w[m, None] * S1[m]).sum(0) / den
    wins = np.array([(0.0, 2.0)] + [(-0.1, 0.2)] * (len(aligns) - 1))
    np.savez(npz, maps=maps, counts=counts, key=Z["key"], label=Z["label"], animal=Z["animal"], epoch=Z["epoch"],
             aligns=np.array(aligns), positions=Z["positions"], windows=wins, home=Z["home"],
             tmpl=Z["tmpl"], speed_response=Z["speed_response"])
    print("aligns:", " ".join(aligns), "->", npz)
    return 0


def cmd_coverage(a) -> int:
    cache, out = Path(a.cache), Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists (a run writes only to its own new folder)")
    (out / "data").mkdir(parents=True)
    Z = np.load(PROD_NPZ)
    T = load_meta(cache, list(Z["key"].astype(str)))
    pos_ok = float((T.position_3d.dropna() == T.position[T.position_3d.notna()]).mean())
    T, C = weights(T, a)
    C.to_csv(out / "data" / f"fr5b_allspouts_coverage_{TAG}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    show = C[["spout", "animal", "epoch", "used", "n_post", "pre_mass_covered", "n_eff", "pre_median_deg",
              "post_median_deg"]]
    print(show.round(3).to_string(index=False))
    n_used = C.groupby(["position", "epoch"], observed=True).used.sum().unstack()
    print("animals usable per spout x epoch:\n", n_used.to_string())
    (out / "data" / f"fr5b_allspouts_checks_{TAG}.txt").write_text(
        f"post-cue licks with a 3-D angle: {int((T.post & T.has_angle).sum())} / {int(T.post.sum())}\n"
        f"spout of the fused-lick trial vs 3-D table position agreement: {pos_ok:.6f}\n"
        f"bins {a.bin_deg} deg; >= {a.min_bin} post licks per bin; animal x spout x epoch used when >= {a.min_cover} "
        f"of that spout's pre mass is covered\n")
    return 0


def cmd_maps(a) -> int:
    cache, out = Path(a.cache), Path(a.out)
    npz = out / "data" / f"fr5b_allspouts_maps_{TAG}.npz"
    if npz.exists():
        raise SystemExit(f"refusing to overwrite {npz}")
    Z = np.load(PROD_NPZ)
    keys = list(Z["key"].astype(str))
    positions = list(Z["positions"].astype(str))
    prod_aligns = list(Z["aligns"].astype(str))
    T = load_meta(cache, keys)
    T, _ = weights(T, a)
    sets = {"lick_post": T.post.to_numpy(float), "lick_post_ang": (T.post & T.has_angle).to_numpy(float),
            "lick_matched": T.w_match.to_numpy(float)}
    aligns = ["cue_wp", "lick_4c_wp", *NEW]
    nS, P = len(keys), Z["maps"].shape[-1]
    maps = np.full((nS, len(aligns), 6, P), np.nan, np.float32)
    counts = np.zeros((nS, len(aligns), 6), int)
    for j, al in enumerate(("cue_wp", "lick_4c_wp")):
        maps[:, j] = Z["maps"][:, prod_aligns.index(al)]
        counts[:, j] = Z["counts"][:, prod_aligns.index(al)]
    nf = T.nf.to_numpy(float)
    rowv, posv = T.row.to_numpy(), T.position.to_numpy()
    for i, k in enumerate(keys):
        with np.load(cache / "perlick6" / f"{k}.npz") as z:
            S1 = z["S1"].astype(np.float64)
        ii_s = np.flatnonzero(rowv == i)
        for pi, p in enumerate(positions):
            jj = ii_s[posv[ii_s] == p]
            for j, al in enumerate(NEW, start=2):
                w = sets[al][jj]
                m = w > 0
                counts[i, j, pi] = int(m.sum())
                den = (w[m] * nf[jj][m]).sum()
                if den > 0:
                    maps[i, j, pi] = (w[m, None] * S1[jj[m] - ii_s[0]]).sum(0) / den
    wpp = prod_aligns.index("lick_4c_wpp")
    ref = Z["maps"][:, wpp]
    rel = np.nanmax(np.abs(maps[:, 2] - ref) / np.maximum(np.abs(ref), 1e-12))
    cnt_ok = bool((counts[:, 2] == Z["counts"][:, wpp]).all())
    print(f"post-cue plumbing vs production lick_4c_wpp (all six spouts): max rel diff {rel:.3g}; counts equal {cnt_ok}")
    wins = np.array([(0.0, 2.0)] + [(-0.1, 0.2)] * (len(aligns) - 1))
    np.savez(npz, maps=maps, counts=counts, key=Z["key"], label=Z["label"], animal=Z["animal"], epoch=Z["epoch"],
             aligns=np.array(aligns), positions=Z["positions"], windows=wins, home=Z["home"],
             tmpl=Z["tmpl"], speed_response=Z["speed_response"])
    T.drop(columns=["row"]).to_csv(out / "data" / f"fr5b_allspouts_licks_{TAG}.csv.gz", index=False)
    with open(out / "data" / f"fr5b_allspouts_checks_{TAG}.txt", "a") as fh:
        fh.write(f"post-cue maps vs production lick_4c_wpp: max rel diff {rel:.3g}; counts equal {cnt_ok}\n")
    print("->", npz)
    return 0


def cmd_summary(a) -> int:
    """Per spout: per animal x epoch (rule 8 first) contra / ipsi half means of the log2 change vs own pre for live,
    post_ang (unweighted reference) and matched; pooled ring counts by sign and half; staged render table."""
    import scripts.face_rhythm_aligned_maps as AM
    from wfield_local import face_rhythm_readouts as frr
    out = Path(a.out)
    Z = np.load(out / "data" / a.npz)
    PP = pd.read_csv(out / "data" / "aligned_maps_perm_points.csv")
    maps, counts = Z["maps"].astype(float), Z["counts"]
    animal, epoch, label = Z["animal"].astype(str), Z["epoch"].astype(str), Z["label"].astype(str)
    aligns = list(Z["aligns"].astype(str))
    positions = list(Z["positions"].astype(str))
    names = dict(zip(aligns, a.names))
    contra = Z["home"][:, 0] <= float(frr.MASTER_MIDLINE_X)
    rows, pooled, stage = [], [], []
    for al in aligns[1:]:
        nm = names[al]
        for p in ALL_POS:
            lg, an_s, ep_s = AM.session_log2(maps, counts, label, animal, epoch, aligns.index(al), positions.index(p))
            pre_an = set(an_s[ep_s == "pre"])
            for ep in EPOCHS[1:]:
                ans = sorted(set(an_s[ep_s == ep]) & pre_an)
                for an in ans:
                    v = np.nanmean(lg[(an_s == an) & (ep_s == ep)], 0)
                    rows.append({"variant": nm, "position": p, "spout": POS_NAME[p], "epoch": ep, "animal": an,
                                 "n_sessions": int(((an_s == an) & (ep_s == ep)).sum()),
                                 "contra_half": np.nanmean(v[contra]), "ipsi_half": np.nanmean(v[~contra]),
                                 "whole_face": np.nanmean(v)})
                g = PP[(PP["align"] == nm) & (PP.position == p) & (PP.epoch == ep)].sort_values("point")
                rec = {"variant": nm, "position": p, "spout": POS_NAME[p], "epoch": ep, "n_animals": len(ans),
                       "animals": "+".join(ans), "tested": bool(len(g))}
                if len(g):
                    mv, sg = g.mean_log2.to_numpy(), g.sig_cluster.to_numpy(bool)
                    rec.update(contra_half=mv[contra].mean(), ipsi_half=mv[~contra].mean(),
                               ring_up=int((sg & (mv > 0)).sum()), ring_down=int((sg & (mv <= 0)).sum()),
                               ring_up_contra=int((sg & (mv > 0) & contra).sum()),
                               ring_down_contra=int((sg & (mv <= 0) & contra).sum()),
                               ring_up_ipsi=int((sg & (mv > 0) & ~contra).sum()),
                               ring_down_ipsi=int((sg & (mv <= 0) & ~contra).sum()))
                    stage.append(pd.DataFrame({"variant": nm, "position": p, "epoch": ep, "point": g.point.to_numpy(),
                                               "log2_ratio": mv, "ring": sg, "n_animals": len(ans)}))
                pooled.append(rec)
    R, Pd = pd.DataFrame(rows), pd.DataFrame(pooled)
    S = pd.concat(stage, ignore_index=True)

    def fold(D):   # pooled-reference aligns pr<sigma>_<epoch>: keep that epoch only, one variant pooled<sigma>
        m = D.variant.str.match(r"^pr[0-9.]+_")
        if not m.any():
            return D
        sg, ep = D.variant[m].str.extract(r"^pr([0-9.]+)_(\w+)$").T.values
        keep = ~m.copy()
        keep[m] = D.epoch[m].to_numpy() == ep
        D = D.copy()
        D.loc[m, "variant"] = "pooled" + sg
        return D[keep].reset_index(drop=True)

    R, Pd, S = fold(R), fold(Pd), fold(S)
    R.to_csv(out / "data" / f"fr5b_{a.tag}_per_animal_{TAG}.csv", index=False, float_format="%.4f")
    Pd.to_csv(out / "data" / f"fr5b_{a.tag}_pooled_{TAG}.csv", index=False, float_format="%.4f")
    for nm in dict.fromkeys(S.variant):
        S[S.variant == nm].drop(columns="variant").to_csv(
            out / "data" / f"fig5B_{a.tag}_maps_{nm}_{TAG}.csv", index=False, float_format="%.17g")
    pd.set_option("display.width", 250)
    print(Pd.round(3).to_string(index=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("coverage", "maps"):
        s = sp.add_parser(nm)
        s.add_argument("--cache", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--bin-deg", type=float, default=5.0)
        s.add_argument("--min-bin", type=int, default=5)
        s.add_argument("--min-cover", type=float, default=0.5)
    s = sp.add_parser("gauss")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--sigmas", type=float, nargs="+", default=[4.0, 2.0, 8.0], help="first = primary")
    s.add_argument("--coverage-only", action="store_true")
    s.add_argument("--bin-deg", type=float, default=5.0)              # hard-bin comparison
    s.add_argument("--min-bin", type=int, default=5)
    s.add_argument("--min-cover", type=float, default=0.5)
    s = sp.add_parser("pooled")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--sigmas", type=float, nargs="+", default=[4.0, 2.0, 8.0], help="first = primary")
    s.add_argument("--coverage-only", action="store_true")
    s.add_argument("--bin-deg", type=float, default=5.0)
    s.add_argument("--min-bin", type=int, default=5)
    s.add_argument("--min-cover", type=float, default=0.5)
    s = sp.add_parser("summary")
    s.add_argument("--out", required=True)
    s.add_argument("--npz", default=f"fr5b_allspouts_maps_{TAG}.npz")
    s.add_argument("--names", nargs="+", default=["cue", "lick", "post", "post_ang", "matched"])
    s.add_argument("--tag", default="allspouts")
    a = ap.parse_args(argv)
    return {"coverage": cmd_coverage, "maps": cmd_maps, "gauss": cmd_gauss, "pooled": cmd_pooled,
            "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
