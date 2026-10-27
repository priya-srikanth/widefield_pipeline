"""Statistics for the PER-SPOUT Haufe maps (3.35; Priya 10-27: "can we add stats to 3.35 too").

Same construction as `iti_decoder_pattern_stats.py`, for the one-vs-rest spout decoders of `iti_spout_patterns.py`
(cued licks, the animal's pre-stroke SVD basis, lags 0..0.2 s): each SESSION gets a held-out pattern (the epoch's
one-vs-rest ridge decoder refit leaving that session out, applied to the session's licks, A_s = Cov(X_s, yhat_s) /
Var(yhat_s)), mapped to pixels and divided by the RMS of the animal's pooled pre map for that spout (as 3.35).
Per pixel, nested bootstrap (animals > sessions): level (epoch map) and change (epoch - pre, paired by animal), BH
q = 0.05 over cortex; outline = significant AND >= 3 animals (of 4) agree in sign. Per animal x spout: similarity of the
session-mean epoch map to the pre map with a within-animal session-bootstrap 95% CI.
--iti: the same for ITI (self-initiated) licks labelled by the PRECEDING trial's spout (`iti_pos`; Priya 10-27: "map
distinguishing ITI licks based on preceding trial's spout location").
Outputs <E>/iti_licks/spout_pattern_stats[_iti].csv, figures/lick_cued_spout_pattern_stats.png / lick_iti_spout_pattern_stats.png. Usage:
python scripts/iti_spout_pattern_stats.py [--iti]
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
from iti_decoder_pattern_stats import AGREE_MIN, N_BOOT, _bh, _boot, _maxstat, _ridge  # noqa: E402
from iti_decoder_patterns import SPATIAL, pattern  # noqa: E402
from iti_spout_patterns import AN, EP, POS  # noqa: E402


def session_maps(st: str = "trial"):
    """{(animal, spout): {epoch: (n_sessions, pixels)}} of held-out one-vs-rest session patterns / pre RMS."""
    from wfield_local import orofacial_cohort as oc
    D = wd.load("svdbasis")
    out = {}
    for a in AN:
        W = np.asarray(pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], np.float32)
        S = {}
        for x, d in sorted(D.items()):
            if d["animal"] != a:
                continue
            X, _C = d[st]
            P = np.array([p if p in POS or "_" not in p else oc.pos_key(p, "L") for p in d[f"{st}_pos"]])
            ok = np.isin(P, POS)
            if ok.sum() >= 30:
                S[x] = (d["epoch"], X[ok], P[ok])
        by = {ep: sorted(x for x, v in S.items() if v[0] == ep) for ep in EP}
        if len(by["pre"]) < 3:
            continue
        for sp in POS:
            Xp = np.vstack([S[x][1] for x in by["pre"]])
            yp = np.concatenate([(S[x][2] == sp).astype(float) for x in by["pre"]])
            if not 0 < yp.sum() < len(yp):
                continue
            rms = float(np.sqrt(np.mean((W @ pattern(Xp, yp).astype(np.float32)) ** 2)))
            res = {}
            for ep in EP:
                rows = []
                for x in by[ep]:
                    trn = [q for q in by[ep] if q != x]
                    if not trn:
                        continue
                    Xt = np.vstack([S[q][1] for q in trn])
                    yt = np.concatenate([(S[q][2] == sp).astype(float) for q in trn])
                    if not 0 < yt.sum() < len(yt):
                        continue
                    sc, m = _ridge(Xt, yt)
                    X = S[x][1]
                    yh = m.predict(sc.transform(X))
                    if yh.var() <= 0:
                        continue
                    A = ((X - X.mean(0)).T @ (yh - yh.mean())) / (len(yh) - 1) / yh.var(ddof=1)
                    rows.append((W @ A.astype(np.float32)) / rms)
                if len(rows) >= 2:
                    res[ep] = np.vstack(rows)
            out[(a, sp)] = res
            print(f"{a} {sp}: " + ", ".join(f"{e} {v.shape[0]}" for e, v in res.items()), flush=True)
    return out


def test(SM: dict):
    rng = np.random.default_rng(0)
    res, rows = {}, []
    for sp in POS:
        have = {a: SM[(a, sp)] for a in AN if (a, sp) in SM and "pre" in SM[(a, sp)]}
        pre = {a: v["pre"] for a, v in have.items()}
        for ep in EP:
            lvl = {a: v[ep] for a, v in have.items() if ep in v}
            for kind in ("level", "change"):
                if kind == "change" and ep == "pre":
                    continue
                ref = pre if kind == "change" else None
                am = np.vstack([lvl[a].mean(0) - (ref[a].mean(0) if ref else 0) for a in sorted(lvl)])
                point = am.mean(0)
                B = _boot(lvl, rng, ref)
                p = np.minimum(1.0, 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0)) + 1 / N_BOOT)
                sig = _bh(p)
                agree = (np.sign(am) == np.sign(point)[None, :]).sum(0)
                smax = _maxstat(B, point)
                res[(sp, ep, kind)] = dict(point=point, sig=sig, agree=agree, n=len(am), sig_max=smax)
                rows.append(dict(spout=sp, epoch=ep, kind=kind, n_animals=len(am), frac_sig=float(sig.mean()),
                                 frac_sig_max=float(smax.mean()),
                                 frac_sig_agree=float((sig & (agree >= min(AGREE_MIN, len(am)))).mean())))
        for a, v in have.items():
            P = v["pre"].mean(0)
            for ep in EP[1:]:
                if ep not in v:
                    continue
                M = v[ep]
                bs = [np.corrcoef(M[rng.integers(0, len(M), len(M))].mean(0), P)[0, 1] for _ in range(200)]
                rows.append(dict(spout=sp, epoch=ep, kind="similarity", animal=a, r=float(np.corrcoef(M.mean(0), P)[0, 1]),
                                 lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
    return res, pd.DataFrame(rows)


def plot(res: dict, T: pd.DataFrame, st: str = "trial") -> None:
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
    cols = [*EP, "acute - pre", "subacute - pre", "chronic - pre"]
    fig, axs = plt.subplots(len(POS), len(cols), figsize=(2.5 * len(cols), 2.45 * len(POS)), squeeze=False)
    lim = np.nanpercentile(np.abs(np.concatenate([r["point"] for k, r in res.items() if k[2] == "level"])), 99)
    for i, sp in enumerate(POS):
        for j, c in enumerate(cols):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            key = (sp, c, "level") if c in EP else (sp, c.split(" ")[0], "change")
            if key not in res:
                ax.axis("off")
                continue
            r = res[key]
            im = ax.imshow(to_img(r["point"]), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.contour(to_img((r["sig"] & (r["agree"] >= min(AGREE_MIN, r["n"]))).astype(float), 0), levels=[0.5], colors="k",
                       linewidths=0.7, linestyles="--")
            if "sig_max" in r:
                ax.contour(to_img(r["sig_max"].astype(float), 0), levels=[0.5], colors="k", linewidths=1.3)
            g = T[(T.spout == sp) & (T.epoch == key[1]) & (T.kind == key[2])]
            sim = T[(T.spout == sp) & (T.epoch == key[1]) & (T.kind == "similarity")]
            sub = f"\nsim to pre {sim.r.mean():.2f}" if c in EP[1:] and len(sim) else ""
            ax.set_title(f"{c} (n={r['n']}) sig {100 * float(g.frac_sig_agree.iloc[0]):.0f}%{sub}", fontsize=7)
            if j == 0:
                ax.set_ylabel(sp.replace("_", " "), fontsize=9)
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.008, pad=0.01, label="pattern / animal's pre RMS (spout)")
    what = ("ITI (self-initiated) licks labelled by the PRECEDING trial's spout" if st == "iti" else "cued licks")
    fig.suptitle(("ITI licks by PRECEDING spout" if st == "iti" else "STATS for the per-spout maps") + " -- NEURAL: per-spout one-vs-rest "
                 f"Haufe maps ({what}), one held-out pattern per "
                 "SESSION, / animal's pre RMS for the spout, mean of animals.\nSOLID outline = max-statistic FWER 0.05 "
                 "(as the decoder page); DASHED = 95% nested "
                 f"bootstrap CI (animals > sessions, {N_BOOT} draws) excludes 0 after BH q 0.05 AND >= 3 animals (of 4) agree in "
                 "sign; 'sig %' = the dashed fraction of cortex.\nChange columns = epoch - pre, paired "
                 "by animal; sim = r(session-mean epoch map, pre map), mean of animals. ipsi (lesion) = image LEFT. "
                 "PROVISIONAL", fontsize=9)
    fig.savefig(il.OUT / "figures" / ('lick_iti_spout_pattern_stats.png' if st == 'iti' else 'lick_cued_spout_pattern_stats.png'), dpi=95, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    st = "iti" if "--iti" in (argv if argv is not None else sys.argv[1:]) else "trial"
    sfx = "_iti" if st == "iti" else ""
    cache = il.OUT / f"spout_pattern_sessions{sfx}.pkl"
    if cache.exists():
        SM = pickle.loads(cache.read_bytes())
    else:
        SM = session_maps(st)
        cache.write_bytes(pickle.dumps(SM, protocol=pickle.HIGHEST_PROTOCOL))
    res, T = test(SM)
    T.to_csv(il.OUT / f"spout_pattern_stats{sfx}.csv", index=False)
    pd.set_option("display.width", 220)
    sim = T[T.kind == "similarity"]
    print("== per animal (rule 8): similarity of the session-mean epoch map to pre, within-animal 95% CI")
    print(sim.assign(s=sim.apply(lambda r: f"{r.r:.2f} [{r.lo:.2f},{r.hi:.2f}]", axis=1))
          .pivot_table(index=["spout", "animal"], columns="epoch", values="s", aggfunc="first")
          .reindex(columns=EP[1:]).to_string())
    print("\n== fraction of cortex significant (BH q 0.05), and also >= 3 animals same sign")
    print(T[T.kind != "similarity"][["spout", "epoch", "kind", "n_animals", "frac_sig", "frac_sig_agree"]]
          .round(3).to_string(index=False))
    plot(res, T, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
