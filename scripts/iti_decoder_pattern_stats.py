"""Statistics for the Haufe decoder-pattern maps (3.33; Priya 10-27: "can 3.33 have stats?").

3.33 shows one pattern per animal x epoch (decoder refit on the epoch's pooled licks), so there is nothing to resample.
Here every SESSION gets its own pattern, from a decoder that never saw it: the epoch's ridge decoder (pre-stroke SVD
basis, lags 0..0.2 s, as `iti_decoder_patterns`) is refit leaving that session out, applied to the session's licks,
and the session pattern is A_s = Cov(X_s, yhat_s) / Var(yhat_s), mapped to pixels (W @ A_s) and divided by the
animal's pooled pre pattern RMS (as 3.33). Then per pixel, nested bootstrap (animals > sessions, `N_BOOT` draws):
  level    epoch map (mean of animals): 95% CI excludes 0 -> pixel carries position in that epoch
  change   epoch - pre (paired by animal, as `analysis_kit.boot_delta`): CI excludes 0 -> map changed there
Two-sided bootstrap p per pixel, Benjamini-Hochberg q = 0.05 over the stat mask; significant pixels are outlined.
Also per pixel: number of animals whose own mean has the sign of the pooled mean (n = 4; rule 8: a pixel driven by one
animal is flagged by the count, not by the CI). Similarity to pre: per-animal r(epoch session-mean map, pre map) with a
session-bootstrap 95% CI (within animal), printed per animal.
Outputs <E>/iti_licks/decoder_pattern_stats.csv (per epoch / axis / set: fraction of mask significant, per-animal
similarity CI), figures/iti_decoder_pattern_stats_{lat,sag}.png. Usage: python scripts/iti_decoder_pattern_stats.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_decoder_patterns as dp  # noqa: E402
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

AN, EP, AXES = dp.AN, dp.EP, dp.AXES
N_BOOT = 1000
Q = 0.05
AGREE_MIN = 3                                                # outline: >= 3 of 4 animals share the sign (Priya 10-27)


def _ridge(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return sc, RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(X), y)


def session_maps():
    """{(animal, set, axis): {epoch: (n_sessions, pixels) array}} of held-out session patterns / pre RMS."""
    D = wd.load("svdbasis")
    out = {}
    for a in AN:
        W = np.asarray(pickle.loads((dp.SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], np.float32)
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
            for ax_, k in AXES.items():
                pre_rms = float(np.sqrt(np.mean((W @ dp.pattern(*dp._stack(S, by["pre"], k))) ** 2)))
                res = {}
                for ep in EP:
                    if len(by[ep]) < 2:
                        continue
                    rows = []
                    for x in by[ep]:
                        sc, m = _ridge(*dp._stack(S, [q for q in by[ep] if q != x], k))
                        X = S[x][1]
                        yh = m.predict(sc.transform(X))
                        A = ((X - X.mean(0)).T @ (yh - yh.mean())) / (len(yh) - 1) / yh.var(ddof=1)
                        rows.append((W @ A.astype(np.float32)) / pre_rms)
                    res[ep] = np.vstack(rows)
                out[(a, st, ax_)] = res
                print(f"{a} {st} {ax_}: " + ", ".join(f"{e} {v.shape[0]}" for e, v in res.items()), flush=True)
    return out


def _boot(per_animal: dict, rng, paired_ref: dict | None = None) -> np.ndarray:
    """(N_BOOT, pixels) of the mean-of-animals map (minus the paired pre map if given), animals > sessions."""
    an = sorted(per_animal)
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(an, size=len(an), replace=True)
        acc = 0
        for a in pick:
            M = per_animal[a]
            v = M[rng.integers(0, len(M), len(M))].mean(0)
            if paired_ref is not None:
                R = paired_ref[a]
                v = v - R[rng.integers(0, len(R), len(R))].mean(0)
            acc = acc + v
        draws.append(acc / len(pick))
    return np.asarray(draws)


def _bh(p: np.ndarray, q: float = Q) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[: np.max(np.nonzero(ok)[0]) + 1]] = True
    return sig


def _maxstat(B: np.ndarray, point: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """FWER (max-statistic) significance, as the decoder page's Haufe maps: standardise each bootstrap draw per pixel
    (centred on the bootstrap mean, / bootstrap SD), take the max |t| over pixels per draw, threshold = its (1 - alpha)
    quantile; a pixel is significant if |point| / SD exceeds it."""
    sd = B.std(0)
    sd = np.where(sd > 0, sd, np.inf)
    tmax = np.abs((B - B.mean(0)) / sd).max(1)
    return np.abs(point) / sd > np.quantile(tmax, 1 - alpha)


def test(SM: dict):
    rng = np.random.default_rng(0)
    res, rows = {}, []
    for st in ("iti", "trial"):
        for ax_ in AXES:
            have = {a: SM[(a, st, ax_)] for a in AN if (a, st, ax_) in SM}
            pre = {a: v["pre"] for a, v in have.items()}
            for ep in EP:
                lvl = {a: v[ep] for a, v in have.items() if ep in v}
                for kind, ref in (("level", None), ("change", pre)):
                    if kind == "change" and ep == "pre":
                        continue
                    if kind == "change":
                        lvl_ = {a: m for a, m in lvl.items() if a in pre}
                    else:
                        lvl_ = lvl
                    am = np.vstack([lvl_[a].mean(0) - (ref[a].mean(0) if ref else 0) for a in sorted(lvl_)])
                    point = am.mean(0)
                    B = _boot(lvl_, rng, ref)
                    p = np.minimum(1.0, 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0)) + 1 / N_BOOT)
                    sig = _bh(p)
                    agree = (np.sign(am) == np.sign(point)[None, :]).sum(0)
                    smax = _maxstat(B, point)
                    res[(st, ax_, ep, kind)] = dict(point=point, sig=sig, agree=agree, n=len(am), sig_max=smax)
                    rows.append(dict(set=st, axis=ax_, epoch=ep, kind=kind, n_animals=len(am), frac_sig_max=float(smax.mean()),
                                     frac_sig=float(sig.mean()), frac_sig_agree=float((sig & (agree >= min(AGREE_MIN, len(am)))).mean())))
            for a, v in have.items():                              # within-animal similarity to pre, session bootstrap
                P = v["pre"].mean(0)
                for ep in EP[1:]:
                    if ep not in v:
                        continue
                    M = v[ep]
                    bs = [np.corrcoef(M[rng.integers(0, len(M), len(M))].mean(0), P)[0, 1] for _ in range(200)]
                    rows.append(dict(set=st, axis=ax_, epoch=ep, kind="similarity", animal=a,
                                     r=float(np.corrcoef(M.mean(0), P)[0, 1]), lo=float(np.percentile(bs, 2.5)),
                                     hi=float(np.percentile(bs, 97.5))))
    return res, pd.DataFrame(rows)


def plot(res: dict, T: pd.DataFrame) -> None:
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

    def to_img(v, fill=np.nan):
        g = np.full(Hh * Ww, fill, dtype=float)
        g[pix] = v
        return g.reshape(Hh, Ww)[::2, ::2]
    axname = {"lat": "LATERAL position (+ = contralesional)", "sag": "SAGITTAL extension (+ = further out)"}
    for ax_ in AXES:
        cols = [*EP, "acute - pre", "subacute - pre", "chronic - pre"]
        fig, axs = plt.subplots(2, len(cols), figsize=(2.7 * len(cols), 6.6), squeeze=False)
        for i, st in enumerate(("iti", "trial")):
            lim = np.nanpercentile(np.abs(np.concatenate([res[(st, ax_, e, "level")]["point"]
                                                          for e in EP if (st, ax_, e, "level") in res])), 99)
            for j, c in enumerate(cols):
                ax = axs[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                key = (st, ax_, c, "level") if c in EP else (st, ax_, c.split(" ")[0], "change")
                if key not in res:
                    ax.axis("off")
                    continue
                r = res[key]
                im = ax.imshow(to_img(r["point"]), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                s_img = to_img((r["sig"] & (r["agree"] >= min(AGREE_MIN, r["n"]))).astype(float), 0)
                ax.contour(s_img, levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
                if "sig_max" in r:
                    ax.contour(to_img(r["sig_max"].astype(float), 0), levels=[0.5], colors="k", linewidths=1.3)
                g = T[(T.set == st) & (T.axis == ax_) & (T.epoch == key[2]) & (T.kind == key[3])]
                fs = float(g.frac_sig.iloc[0]) if len(g) else np.nan
                fa = float(g.frac_sig_agree.iloc[0]) if len(g) else np.nan
                ax.set_title(f"{c} (n={r['n']})\nsig {100 * fs:.0f}% of cortex ({100 * fa:.0f}% also >=3/{r['n']} agree)",
                             fontsize=7.5)
                if j == 0:
                    ax.set_ylabel("ITI licks" if st == "iti" else "cued licks", fontsize=9)
            fig.colorbar(im, ax=axs[i, :].tolist(), fraction=0.01, pad=0.01, label="pattern / animal's pre RMS")
        fig.suptitle(f"STATS for the decoder pattern maps -- NEURAL: Haufe pattern of the {axname[ax_]} decoder, one held-out pattern per "
                     "SESSION (epoch decoder refit leaving the session out), / animal's pre RMS, mean of animals.\n"
                     f"SOLID outline = max-statistic FWER 0.05 over cortex (as the decoder page); DASHED = 95% nested "
                     f"bootstrap CI (animals > sessions, {N_BOOT} draws) excludes 0 after Benjamini-Hochberg q 0.05 AND >= 3 "
                     "animals (of 4) agree in sign. Change columns = "
                     "epoch - pre, paired by animal.\nipsi (lesion) = image LEFT. Unscaled features. PROVISIONAL",
                     fontsize=9)
        fig.savefig(il.OUT / "figures" / f"iti_decoder_pattern_stats_{ax_}.png", dpi=105, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    cache = il.OUT / "decoder_pattern_sessions.pkl"
    if cache.exists():
        SM = pickle.loads(cache.read_bytes())
    else:
        SM = session_maps()
        cache.write_bytes(pickle.dumps(SM, protocol=pickle.HIGHEST_PROTOCOL))
    res, T = test(SM)
    T.to_csv(il.OUT / "decoder_pattern_stats.csv", index=False)
    pd.set_option("display.width", 220)
    sim = T[T.kind == "similarity"]
    print("== per animal (rule 8): similarity of the session-mean epoch map to pre, within-animal 95% CI")
    print(sim.assign(s=sim.apply(lambda r: f"{r.r:.2f} [{r.lo:.2f},{r.hi:.2f}]", axis=1))
          .pivot_table(index=["set", "axis", "animal"], columns="epoch", values="s", aggfunc="first")[EP[1:]].to_string())
    print("\n== fraction of cortex significant (BH q 0.05), and also >= 3 animals same sign")
    print(T[T.kind != "similarity"][["set", "axis", "epoch", "kind", "n_animals", "frac_sig", "frac_sig_agree"]]
          .round(3).to_string(index=False))
    plot(res, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
