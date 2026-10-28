"""WHERE the like-for-like lick read-out offset is carried, and the decoder's EFFECTIVE WEIGHT per Allen area for a 1-SD
fluctuation, in three decoder bases on ONE physical scale (Priya 10-28: "where carries the offset (ITI and cued)? Allen
ALM / tjM1, LocaNMF and SVD"; "the decoder per 1 SD map").

WHY A NEW SCALE. Every frozen lick decoder sees features h = M x / sd (x = pixel dF/F, M = the basis' read-out, sd = the
feature's per-session frame SD), so its coefficients are per feature-SD. `lick_offset_contrib_map` draws w = M' beta and
dx = B dh with beta and dh in those per-SD units: exact totals, and harmless in the Allen basis (each area is one
feature), but in SVD / LocaNMF every component carries a different SD, so its per-pixel weight and contribution maps mix
units. Here every basis is put on pixel dF/F with ONE reference SD per feature (sd_ref = mean over the animal's pre
sessions; the scaled features are unchanged, so the decoder and the offset are untouched):
  w_true  = M_true' (beta / sd_ref)          mm per unit dF/F per pixel
  dx_true = B_true (dh * sd_ref)             dF/F change at matched landing (M_true B_true = I)
  c_true  = w_true * dx_true, sums EXACTLY to beta . dh = the offset for ANY positive sd_ref (printed check)
  M_true: Allen 1e3 * area indicator / n (the area features are 1e3 x area means); SVD P (pre-stroke basis, P'x);
          LocaNMF s * pinv(A) (glue / bulb components dropped, 'FM').  Unit check: one pre session per animal rebuilt
          through M_true, SD vs the stored frame-SD tables (ratio printed; must be ~1).
OUTPUTS per Allen area (ALM / tjM1 split, ROI v8, each animal's own masks; pixels in no area = 'outside'):
  offset carried  C_a = sum_{p in a} c_true(p)   per set x animal x epoch; share = C_a / offset
  effective weight E_a = SD_a * sum_{p in a} w_true(p)   = the decoder's change (mm) for a UNIFORM +1 SD change of area
                   a's mean (SD_a = its pre-session frame SD); in the Allen basis = the area's coefficient exactly
  Neither is unique across bases (ridge weights on correlated features are not): read where the bases AGREE.
  'SVD first 50' = components 1..50 of the same SVD decoder (exact part of its offset, orthonormal basis).
Needs terms_<tag>.csv from lick_offset_contrib_map (per-feature beta, dh). Per-animal table first (rule 8).
Outputs <OUT>/offset_contrib/where_{areas,weights}.csv; figures lick_offset_where_{maps,bars}.png,
lick_decoder_weight_1sd.png; figdata lick_offset_where (--plot-only).
Usage: python scripts/lick_offset_where.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_offset_contrib_map as lom  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
BASES = [("areas_splitr8", "areas_split", "Allen areas, ALM / tjM1 split"),
         ("locanmfFM", "locanmf", "LocaNMF (glue / bulb comps dropped)"),
         ("svdF", "svd", "pre-stroke SVD (all comps)"),
         ("svdF_k50", "svd", "pre-stroke SVD, first 50 comps")]
SETLAB = {"iti": "ITI licks", "trial": "cued licks"}
FAMS = ["ALM", "tjM1", "MOs_rest", "MOp_rest", "SSp-m", "SSp-n", "SSp other", "VIS", "RSP", "other", "outside"]


def fam(name: str) -> tuple[str, str]:
    if name == "outside":
        return "outside", ""
    base, side = name.rsplit("_", 1)
    side = {"left": "ipsi", "right": "contra"}.get(side, side)
    if base in ("ALM", "tjM1", "MOs_rest", "MOp_rest", "SSp-m", "SSp-n"):
        return base, side
    if base.startswith("SSp"):
        return "SSp other", side
    if base.startswith("VIS"):
        return "VIS", side
    if base.startswith("RSP"):
        return "RSP", side
    return "other", side


def _pre_labels(a: str) -> list[str]:
    from wfield_local import epochs
    return [x for x in il.labels(f"{a}_*") if epochs.epoch_of(x) == "pre"]


def sd_ref(source: str, a: str, keep) -> np.ndarray:
    pre = _pre_labels(a)
    if source == "areas_split":
        names = lom.feature_names("areas_split", a)
        return il.area_sd().loc[pre, names].to_numpy(float).mean(0)
    F = pd.read_csv(il.OUT / f"basis_{'svdbasis' if source == 'svd' else 'locanmf'}" / "frame_sd.csv", index_col=0)
    v = np.vstack([F.loc[x].dropna().to_numpy(float) for x in pre if x in F.index]).mean(0)
    return v[keep] if keep is not None else v


def true_spatial(source: str, a: str, keep):
    """(B_true, M_true, pix) on pixel dF/F. lom.spatial gives Mt with Mt' B = I; the Allen features carry the 1e3."""
    B, Mt, pix = lom.spatial(source, a, keep)
    if source == "areas_split":
        return B / 1e3, Mt * 1e3, pix
    return B, Mt, pix


def area_labels(a: str) -> tuple[np.ndarray, list[str]]:
    """Full-grid (H*W) index into names (-1 = no area), the animal's own split masks (as lom.spatial)."""
    flat, _pstat, M = lom._grid(a)
    for src, dst, h in il._rest_list():
        M[f"{src}_rest_{h}"] = M[f"{src}_{h}"] & ~M[f"{dst}_{h}"]
    names = lom.feature_names("areas_split", a)
    lab = np.full(lom.H * lom.W, -1)
    for k, n in enumerate(names):
        lab[flat[M[n]]] = k
    return lab, names


def unit_check(source: str, a: str, keep) -> float:
    """One pre session's features REBUILT from the imaging through M_true (pixel dF/F = V Ub'), drift-removed as the
    decoder inputs, SD vs the stored frame SD (median ratio; ~1 = M_true is on the decoder's scale). LocaNMF footprint
    pixels outside the imaging grid are dropped (their share is small; a ratio well off 1 would flag it)."""
    from wfield_local import combined_encoding as ce
    x = _pre_labels(a)[0]
    _B, Mt, pix = true_spatial(source, a, keep)
    V, ft, Ub, flat, *_r = ce.load_imaging(x)                          # Ub rows = the `flat` pixels
    inv = np.full(lom.H * lom.W, -1)
    inv[flat] = np.arange(len(flat))
    idx = inv[pix]
    ok = idx >= 0
    R = V.astype(np.float64) @ (Mt[ok].T @ Ub[idx[ok]].astype(np.float64)).T
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    if source == "areas_split":
        ref = il.area_sd().loc[x, lom.feature_names("areas_split", a)].to_numpy(float)
    else:
        F = pd.read_csv(il.OUT / f"basis_{'svdbasis' if source == 'svd' else 'locanmf'}" / "frame_sd.csv", index_col=0)
        ref = F.loc[x].dropna().to_numpy(float)
        ref = ref[keep] if keep is not None else ref
    return float(np.median(R.std(0) / ref))


def compute() -> dict:
    import iti_lick_order as lo
    rows, wrows, checks = [], [], []
    SDA = il.area_sd()
    for tag, source, _lab in BASES:
        f = lom.OUT / f"terms_{tag.replace('_k50', '')}.csv"
        if not f.exists() or f.stat().st_size < 10:
            raise SystemExit(f"missing {f}: run lick_offset_contrib_map.py for this basis")
        T = pd.read_csv(f)
        kmax = 50 if tag.endswith("_k50") else None
        for a in AN:
            keep = lo.locanmf_inmask(a) if source == "locanmf" else None
            B, Mt, pix = true_spatial(source, a, keep)
            sdr = sd_ref(source, a, keep)
            lab, names = area_labels(a)
            plab = lab[pix]
            SDa = SDA.loc[_pre_labels(a), names].to_numpy(float).mean(0) / 1e3     # area-mean frame SD in dF/F
            if not kmax:
                checks.append(dict(basis=tag, animal=a, unit_ratio=unit_check(source, a, keep)))
            for st in ("iti", "trial"):
                for ep in EP:
                    t = T[(T.set == st) & (T.animal == a) & (T.epoch == ep)]
                    if t.empty:
                        continue
                    beta, dh = t.beta.to_numpy(float), t.dh.to_numpy(float)
                    if len(beta) != B.shape[1]:
                        raise ValueError(f"{tag} {a}: {len(beta)} terms vs {B.shape[1]} spatial columns")
                    K = kmax or len(beta)
                    w = Mt[:, :K] @ (beta[:K] / sdr[:K])
                    dx = B[:, :K] @ (dh[:K] * sdr[:K])
                    c = w * dx
                    off = float(beta[:K] @ dh[:K])
                    if abs(c.sum() - off) > 1e-6 * max(1.0, abs(off)):
                        raise AssertionError(f"{tag} {a} {st} {ep}: sum c {c.sum():.6f} != offset {off:.6f}")
                    for k, n in enumerate([*names, "outside"]):
                        sel = plab == (k if n != "outside" else -1)
                        rows.append(dict(basis=tag, set=st, animal=a, epoch=ep, area=n, carried=float(c[sel].sum()),
                                         offset=off))
                    if ep == "chronic":                                       # weights are per fit, not per epoch
                        for k, n in enumerate(names):
                            wrows.append(dict(basis=tag, set=st, animal=a, area=n,
                                              eff_w=float(SDa[k] * w[plab == k].sum())))
            print(f"[{tag} {a}] done", flush=True)
    R, Wt, C = pd.DataFrame(rows), pd.DataFrame(wrows), pd.DataFrame(checks)
    R.to_csv(lom.OUT / "where_areas.csv", index=False)
    Wt.to_csv(lom.OUT / "where_weights.csv", index=False)
    pd.set_option("display.width", 220)
    print("== unit check (rebuilt feature SD / stored frame SD; ~1 = M_true is on the decoder's scale)\n"
          + C.round(3).to_string(index=False))
    R[["fam", "side"]] = R.area.apply(lambda n: pd.Series(fam(n)))
    Wt[["fam", "side"]] = Wt.area.apply(lambda n: pd.Series(fam(n)))
    G = R.groupby(["basis", "set", "epoch", "animal", "fam", "side"]).carried.sum().reset_index()
    print("== per animal (rule 8): offset carried by ALM + tjM1 (both sides) and SSp-m + n, chronic and acute (mm)")
    for ep in ("acute", "chronic"):
        g = G[(G.epoch == ep) & G.fam.isin(["ALM", "tjM1", "SSp-m", "SSp-n"])].copy()
        g["grp"] = np.where(g.fam.isin(["ALM", "tjM1"]), "ALM+tjM1", "SSp-m+n")
        print(f"-- {ep}\n" + g.groupby(["set", "basis", "grp", "animal"]).carried.sum().unstack("animal").round(3)
              .to_string())
    tot = R.groupby(["basis", "set", "epoch", "animal"]).offset.first().groupby(["basis", "set", "epoch"]).mean()
    print("== offset (mean of animals; the decoder's beta . dh, = the 11.x totals)\n" + tot.unstack("epoch").round(3)
          .to_string())
    return {"R": R, "W": Wt, "C": C}


def _area_img():
    from iti_area_maps import split_image
    img, nm = split_image(il.labels()[0])
    return img, nm


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import beta_maps
    mask = np.asarray(beta_maps.brain_mask(), bool)
    img, nm = _area_img()
    R, Wt = F["R"], F["W"]

    def paint(vals: dict) -> np.ndarray:
        out = np.full(img.shape, np.nan)
        for k, n in enumerate(nm):
            if n in vals:
                out[img == k] = vals[n]
        return np.where(mask, out, np.nan)

    # 1) offset carried per AREA, painted (rows = basis, cols = epoch), per set
    for st in ("iti", "trial"):
        fig, axs = plt.subplots(len(BASES), 3, figsize=(12, 3.5 * len(BASES)), squeeze=False)
        r = R[R.set == st]
        A = r.groupby(["basis", "epoch", "area", "animal"]).carried.sum().groupby(["basis", "epoch", "area"]).mean()
        lim = np.nanpercentile(np.abs(A.drop(index="outside", level="area").to_numpy()), 99)
        for i, (tag, _s, lab) in enumerate(BASES):
            for j, ep in enumerate(EP):
                ax = axs[i, j]
                if (tag, ep) not in A.index.droplevel("area"):
                    ax.axis("off")
                    continue
                v = A.loc[(tag, ep)]
                im = ax.imshow(paint(v.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                off = r[(r.basis == tag) & (r.epoch == ep)].groupby("animal").offset.first().mean()
                ax.set_title(f"{lab}\n{ep} - pre: offset {off:+.2f} mm; outside areas {v.get('outside', 0):+.2f}",
                             fontsize=8)
                ax.axis("off")
            plt.colorbar(im, ax=axs[i, 2], fraction=0.04, label="mm (+ = contra)")
        fig.suptitle(f"NEURAL: WHERE the like-for-like read-out offset is carried, {SETLAB[st]}\nper Allen area (ALM / "
                     "tjM1 split): sum of decoder weight x activity change at matched landing, every basis on pixel dF/F"
                     "\nSums to the offset (+ 'outside areas'). Mean of animals; ipsi (lesion) = image LEFT. Not unique "
                     "across bases: read where they agree. PROVISIONAL", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(lom.FIG / f"lick_{'iti' if st == 'iti' else 'cued'}_offset_where_maps.png", dpi=90)
        plt.close(fig)

    # 2) bars by area family x side, per set x epoch: bases side by side, animal SEM
    G = R.groupby(["basis", "set", "epoch", "animal", "fam", "side"]).carried.sum().reset_index()
    cats = [(f_, s_) for f_ in FAMS for s_ in (("",) if f_ == "outside" else ("ipsi", "contra"))]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), squeeze=False, sharey=True)
    cols = ["#d95f02", "#1b9e77", "#7570b3", "#a6a6ff"]
    for i, st in enumerate(("iti", "trial")):
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            y = np.arange(len(cats))
            for b, (tag, _s, lab) in enumerate(BASES):
                g = G[(G.basis == tag) & (G.set == st) & (G.epoch == ep)]
                P = g.pivot_table(index=["fam", "side"], columns="animal", values="carried", aggfunc="sum")
                P = P.reindex(pd.MultiIndex.from_tuples(cats)).fillna(0)
                m, se = P.mean(1), P.std(1, ddof=1) / np.sqrt(P.notna().sum(1).clip(lower=1))
                ax.barh(y + (b - 1.5) * 0.2, m, 0.2, xerr=se, color=cols[b], label=lab, error_kw=dict(lw=0.6))
            ax.set_yticks(y, [f"{f_} {s_}".strip() for f_, s_ in cats], fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="k", lw=0.7)
            ax.set_title(f"{SETLAB[st]}, {ep} - pre", fontsize=9)
            ax.set_xlabel("offset carried (mm; + = contralesional)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("NEURAL: offset carried per area FAMILY and side (ipsi = lesioned), four decoder bases on pixel dF/F. "
                 "Mean of animals +- SEM across animals. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(lom.FIG / "lick_offset_where_bars.png", dpi=90)
    plt.close(fig)

    # 3) EFFECTIVE WEIGHT per area for a +1 SD uniform change of its mean (rows = set, cols = basis), one scale per row
    fig, axs = plt.subplots(2, len(BASES), figsize=(4.2 * len(BASES), 8), squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        A = Wt[Wt.set == st].groupby(["basis", "area"]).eff_w.mean()
        lim = np.nanpercentile(np.abs(A.to_numpy()), 99)
        for j, (tag, _s, lab) in enumerate(BASES):
            ax = axs[i, j]
            if tag in A.index.get_level_values(0):
                im = ax.imshow(paint(A.loc[tag].to_dict()), cmap="PuOr_r", vmin=-lim, vmax=lim, interpolation="nearest")
                a_ = A.loc[tag]
                top = ", ".join(f"{n.replace('_left', ' i').replace('_right', ' c')} {v:+.2f}"
                                for n, v in a_.reindex(a_.abs().sort_values(ascending=False).index)[:4].items())
                ax.set_title(f"{SETLAB[st]}: {lab}\nlargest: {top}", fontsize=7)
            ax.axis("off")
        plt.colorbar(im, ax=axs[i, -1], fraction=0.04, label="mm per +1 SD of the area mean")
    fig.suptitle("NEURAL: decoder EFFECTIVE WEIGHT per area = change of the frozen lateral read-out for a uniform +1 SD "
                 "change of that area's mean (pre-session frame SD), every basis on pixel dF/F; one colour scale per row.\n"
                 "In the Allen basis = the area coefficient. Mean of animals; ipsi (lesion) = image LEFT; i / c = ipsi / "
                 "contra. Weights are not importance (see the drop-area test). PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(lom.FIG / "lick_decoder_weight_1sd.png", dpi=90)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_offset_where"))
        return 0
    F = compute()
    save_fd("lick_offset_where", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
