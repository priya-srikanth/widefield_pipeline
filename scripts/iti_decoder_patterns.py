"""WHERE in cortex is lick position carried? Haufe activation patterns of the lick-position decoder, per epoch, as pixel
maps (Priya 10-26: "compare per-session or per-epoch decoder weights by SVD map (and map the decoder weights per
position across epochs)"; handoff 00f to-do).

Raw ridge weights are not maps (they cancel correlated noise, so a weight can be large where there is no signal), so
each decoder is turned into its ACTIVATION PATTERN (Haufe et al. 2014): A = Cov(X, yhat) / Var(yhat), the activity
change per unit of decoded position. Features = each animal's pre-stroke shared SVD basis (200 components, the encoder
pixel run's `spatial_<animal>.pkl`; same licks / lags 0..0.2 s as the 3.15 / 3.18 decoders, `iti_where_decoder.load`),
so a pattern maps back to pixels exactly: map = W @ A (W = the basis on the Allen-aligned stat mask).
Per animal x lick set (ITI / cued) x axis (lateral, sagittal):
  pre      ridge fit on the animal's PRE licks (all pre sessions pooled), pattern from the same licks
  epoch    ridge REFIT on that epoch's licks (pooled sessions): where position is carried in that epoch
Similarity of an epoch's pattern to pre = pixel correlation over the stat mask. Noise floor: sessions of each epoch are
split in two halves (alternating by date) and patterns refit per half; reliability = r(half 1, half 2);
disattenuated similarity = r(epoch, pre) across halves / sqrt(rel_pre * rel_epoch) (1 = same map up to noise).
Maps are divided by the RMS of the animal's pre pattern (animals equal), then averaged over animals; ipsi = image LEFT.
Features are NOT z-scored (SVD components share one dF/F scale); the z-scored check is per area (3.21 / KEY row).
Outputs <E>/iti_licks/decoder_patterns.csv (similarities), figures/lick_decoder_patterns_{lat,sag}.png,
figures/lick_decoder_patterns_animals_lat.png. Usage: python scripts/iti_decoder_patterns.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
AXES = {"lat": 0, "sag": 3}
SPATIAL = il.E / "frozen_encoder_gate-attempted_v7g3_t3d_frA_te_basis" / "fit"


def pattern(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Haufe pattern of a ridge decoder of y from X (fit and pattern on the same licks)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(X), y)
    yh = m.predict(sc.transform(X))
    Xc = X - X.mean(0)
    yc = yh - yh.mean()
    return (Xc.T @ yc) / (len(yc) - 1) / yc.var(ddof=1)


def _stack(S: dict, labs: list[str], k: int):
    X = np.vstack([S[x][1] for x in labs])
    y = np.concatenate([S[x][2][:, k] for x in labs])
    return X, y


def compute():
    D = wd.load("svdbasis")
    rows, maps = [], {}
    for a in AN:
        sp = pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())
        W = np.asarray(sp["P"], np.float64)                      # (stat pixels, K)
        for st in ("iti", "trial"):
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                ok = np.isfinite(C[:, [0, 3]]).all(1)
                if ok.sum() >= 20:
                    S[x] = (d["epoch"], X[ok], C[ok])
            by = {ep: sorted(x for x, v in S.items() if v[0] == ep) for ep in EP}
            if len(by["pre"]) < 3:
                continue
            if W.shape[1] != next(iter(S.values()))[1].shape[1]:
                raise ValueError(f"{a}: basis has {W.shape[1]} columns, features {next(iter(S.values()))[1].shape[1]}")
            for ax_, k in AXES.items():
                A = {ep: pattern(*_stack(S, by[ep], k)) for ep in EP if len(by[ep]) >= 2}
                pre_rms = float(np.sqrt(np.mean((W @ A["pre"]) ** 2)))
                for ep, v in A.items():
                    maps[(a, st, ax_, ep)] = (W @ v) / pre_rms
                halves = {ep: (by[ep][0::2], by[ep][1::2]) for ep in A}
                H = {ep: [W @ pattern(*_stack(S, h, k)) for h in hs] for ep, hs in halves.items()}

                def corr(u, v):
                    return float(np.corrcoef(u, v)[0, 1])
                rel_pre = corr(*H["pre"])
                for ep in A:
                    rel = corr(*H[ep])
                    cross = np.mean([corr(H[ep][i], H["pre"][j]) for i in (0, 1) for j in (0, 1)]) if ep != "pre" \
                        else rel_pre
                    rows.append(dict(animal=a, set=st, axis=ax_, epoch=ep, n_sessions=len(by[ep]),
                                     r_full=corr(maps[(a, st, ax_, ep)], maps[(a, st, ax_, "pre")]),
                                     rel=rel, r_cross_halves=cross,
                                     r_disatt=cross / np.sqrt(max(rel, 1e-6) * max(rel_pre, 1e-6))
                                     if rel > 0 and rel_pre > 0 else np.nan,
                                     amp=float(np.sqrt(np.mean(maps[(a, st, ax_, ep)] ** 2)))))
        del W
    return pd.DataFrame(rows), maps


def plot(T: pd.DataFrame, maps: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from frozen_area_maps import area_image, outline
    from wfield_local import combined_encoding as ce
    lab0 = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))[0]
    _V, _ft, _Ub, flat, _pa, _nm, shape, pstat = ce.load_imaging(lab0)
    Hh, Ww = (int(x) for x in shape)
    pix = flat[pstat]
    img, _ = area_image(lab0)
    ol = outline(img)[::2, ::2]

    def to_img(v):
        g = np.full(Hh * Ww, np.nan)
        g[pix] = v
        return g.reshape(Hh, Ww)[::2, ::2]
    axname = {"lat": "LATERAL position (+ = contralesional)", "sag": "SAGITTAL extension (+ = further out)"}
    setname = {"iti": "ITI (self-initiated) licks", "trial": "CUED in-trial licks"}
    for ax_ in AXES:
        cols = [*EP, "chronic - pre"]
        fig, axs = plt.subplots(2, len(cols), figsize=(2.9 * len(cols), 6.4), squeeze=False)
        for i, st in enumerate(("iti", "trial")):
            M = {}
            for ep in EP:
                have = [maps[(a, st, ax_, ep)] for a in AN if (a, st, ax_, ep) in maps]
                if have:
                    M[ep] = (np.nanmean(have, 0), len(have))
            both = [a for a in AN if (a, st, ax_, "chronic") in maps]
            M["chronic - pre"] = (np.nanmean([maps[(a, st, ax_, "chronic")] - maps[(a, st, ax_, "pre")]
                                              for a in both], 0), len(both))
            lim = np.nanpercentile(np.abs(np.concatenate([v.ravel() for v, _ in M.values()])), 99)
            for j, c in enumerate(cols):
                ax = axs[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                if c not in M:
                    ax.axis("off")
                    continue
                v, n = M[c]
                im = ax.imshow(to_img(v), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                g = T[(T.set == st) & (T.axis == ax_) & (T.epoch == (c if c in EP else "chronic"))]
                sub = "" if c == "pre" else (f"\nsimilarity to pre {g.r_disatt.mean():.2f} (noise-corrected)"
                                             if c in EP else "")
                ax.set_title(f"{c} (n={n} animals){sub}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(setname[st], fontsize=9)
            fig.colorbar(im, ax=axs[i, :].tolist(), fraction=0.012, pad=0.01,
                         label="pattern / animal's pre RMS")
        fig.suptitle(f"NEURAL: WHERE cortex carries {axname[ax_]} -- Haufe activation pattern of the lick-position "
                     "decoder (Cov(activity, decoded position) / Var), refit per epoch,\nin each animal's pre-stroke SVD "
                     "basis (200 components, lags 0..0.2 s from the lick peak), mapped to pixels, / the animal's pre "
                     "pattern RMS, mean of animals. Red = more activity for licks further +; ipsi (lesion) = image LEFT. "
                     "Unscaled features. PROVISIONAL", fontsize=9)
        fig.savefig(il.OUT / "figures" / f"lick_decoder_patterns_{ax_}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    # per animal, lateral
    fig, axs = plt.subplots(2 * len(AN), len(EP), figsize=(2.6 * len(EP), 2.3 * 2 * len(AN)), squeeze=False)
    for ai, a in enumerate(AN):
        for si, st in enumerate(("iti", "trial")):
            i = 2 * ai + si
            vs = [maps.get((a, st, "lat", ep)) for ep in EP]
            ok = [v for v in vs if v is not None]
            lim = np.nanpercentile(np.abs(np.concatenate([v.ravel() for v in ok])), 99) if ok else 1
            for j, (ep, v) in enumerate(zip(EP, vs)):
                ax = axs[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                if v is None:
                    ax.axis("off")
                    continue
                ax.imshow(to_img(v), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                g = T[(T.animal == a) & (T.set == st) & (T.axis == "lat") & (T.epoch == ep)]
                ax.set_title(f"{ep}" + (f" (sim {g.r_disatt.iloc[0]:.2f})" if ep != "pre" and len(g) else ""),
                             fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{a}\n{'ITI' if st == 'iti' else 'cued'}", fontsize=8)
    fig.suptitle("Per animal: Haufe pattern of the LATERAL lick-position decoder, refit per epoch (pre-stroke SVD basis), "
                 "/ pre RMS; sim = noise-corrected similarity to pre. ipsi = image LEFT. PROVISIONAL", fontsize=9)
    fig.savefig(il.OUT / "figures" / "lick_decoder_patterns_animals_lat.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    T, maps = compute()
    T.to_csv(il.OUT / "decoder_patterns.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): noise-corrected similarity of each epoch's pattern to pre (1 = same map)")
    print(T.pivot_table(index=["set", "axis", "animal"], columns="epoch", values="r_disatt")[EP[1:]].round(2).to_string())
    print("\n== split-half reliability of the pattern")
    print(T.pivot_table(index=["set", "axis", "animal"], columns="epoch", values="rel")[EP].round(2).to_string())
    print("\n== pattern amplitude (RMS / pre RMS)")
    print(T.pivot_table(index=["set", "axis", "animal"], columns="epoch", values="amp")[EP].round(2).to_string())
    with open(il.OUT / "decoder_patterns_maps.pkl", "wb") as fh:
        pickle.dump({k: v.astype(np.float32) for k, v in maps.items()}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    plot(T, maps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
