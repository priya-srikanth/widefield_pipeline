"""Map-level permutation tests + AMPLITUDE-NORMALISED change maps on the fully per-session Haufe maps
(`iti_session_patterns.py`; Priya 10-27: "is there a way to measure significant DIFFERENCES between maps" -> "yes build
map-level permutation and amplitude-normalized change maps").

Every session map is first scaled to unit RMS over the stat mask (SHAPE only; its RMS / the animal's pre mean RMS is
kept separately as GAIN), so a cortex-wide change in response size cannot pass as a change in layout.
(1) MAP-LEVEL test, per animal x map (set / target) x epoch: distance d = 1 - r(mean of the epoch's session maps, mean of
    the pre session maps). Null: session labels permuted between pre and that epoch within the animal (N_PERM); per-animal
    p; cohort: mean of the animals' d against the joint null (independent permutations per animal), p, and the number of
    animals with p < 0.05 (rule 8). This works at n = 4 animals because the permutation is over SESSIONS (8 pre x 10-24
    post per animal), unlike a sign-flip over 4 animals (16 arrangements, min p 0.0625).
(2) SIMILARITY over days: r(session map, mean pre map), pre sessions against the mean of the OTHER pre sessions.
(3) AMPLITUDE-NORMALISED CHANGE MAPS: per pixel, epoch - pre on the unit-RMS maps, paired by animal, nested bootstrap
    (animals > sessions, `iti_decoder_pattern_stats._boot`), BH q 0.05 over cortex, outline = significant AND >= 3
    animals share the sign. Level maps (epoch mean) drawn alongside.
Outputs <E>/iti_licks/session_pattern_{maplevel,similarity,gain}.csv, figures/iti_session_pattern_maplevel.png,
figures/iti_session_pattern_similarity.png, figures/iti_session_pattern_change_<family>.png.
Usage: python scripts/iti_session_pattern_tests.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from iti_decoder_pattern_stats import AGREE_MIN, _bh, _boot  # noqa: E402
from iti_session_patterns import AN, OUT, POS  # noqa: E402

EP = ["pre", "acute", "subacute", "chronic"]
N_PERM = 5000
STRIDE = 8          # whole-map distance on every 8th pixel (maps are smooth: 200 SVD components)
FAMILIES = {
    "position": ["iti/lat", "trial/lat", "iti/sag", "trial/sag"],
    "within_spout": ["iti/wlat", "trial/wlat", "iti/wsag", "trial/wsag"],
    "spout_cued": [f"trial/sp_{p}" for p in POS],
    "spout_iti": [f"iti/sp_{p}" for p in POS],
}


def load() -> dict:
    """{animal: {map key: {"epoch": [...], "day": [...], "M": (n, pix) unit-RMS, "gain": (n,)}}}"""
    out = {}
    for a in AN:
        f = OUT / f"{a}.pkl"
        if not f.exists():
            continue
        S = pickle.loads(f.read_bytes())
        keys = sorted({k for r in S.values() for k in r["maps"]})
        out[a] = {}
        for k in keys:
            labs = [x for x in sorted(S) if k in S[x]["maps"]]
            M = np.vstack([S[x]["maps"][k] for x in labs]).astype(np.float64)
            rms = np.sqrt((M ** 2).mean(1))
            ep = np.array([S[x]["epoch"] for x in labs])
            pre_rms = rms[ep == "pre"].mean() if (ep == "pre").any() else np.nan
            out[a][k] = {"label": labs, "epoch": ep, "day": np.array([S[x]["day"] for x in labs], float),
                         "M": M / rms[:, None], "gain": rms / pre_rms}
    return out


def _dist(A: np.ndarray, B: np.ndarray) -> float:
    return 1.0 - float(np.corrcoef(A.mean(0), B.mean(0))[0, 1])


def maplevel(L: dict) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    keys = sorted({k for a in L for k in L[a]})
    for k in keys:
        for ep in EP[1:]:
            obs, null = {}, {}
            for a in AN:
                v = L.get(a, {}).get(k)
                if v is None:
                    continue
                pre, post = v["M"][v["epoch"] == "pre"], v["M"][v["epoch"] == ep]
                if len(pre) < 2 or len(post) < 2:
                    continue
                Z = np.vstack([pre, post])[:, ::STRIDE]
                n0, n = len(pre), len(pre) + len(post)
                obs[a] = _dist(Z[:n0], Z[n0:])
                # all permutations at once: indicator weights -> group means by one matrix product
                Ind = np.zeros((N_PERM, n))
                for i in range(N_PERM):
                    Ind[i, rng.permutation(n)[:n0]] = 1
                A_ = (Ind @ Z) / n0
                B_ = ((1 - Ind) @ Z) / (n - n0)
                A_ -= A_.mean(1, keepdims=True)
                B_ -= B_.mean(1, keepdims=True)
                null[a] = 1 - (A_ * B_).sum(1) / np.sqrt((A_ ** 2).sum(1) * (B_ ** 2).sum(1))
            if not obs:
                continue
            an = sorted(obs)
            D = np.mean([obs[a] for a in an])
            Dn = np.mean([null[a] for a in an], axis=0)
            row = dict(map=k, epoch=ep, n_animals=len(an), d_mean=D,
                       p_cohort=(np.sum(Dn >= D) + 1) / (N_PERM + 1))
            ps = {}
            for a in an:
                ps[a] = (np.sum(null[a] >= obs[a]) + 1) / (N_PERM + 1)
                row[f"d_{a}"] = obs[a]
                row[f"p_{a}"] = ps[a]
            row["n_sig_animals"] = int(sum(p < 0.05 for p in ps.values()))
            rows.append(row)
    T = pd.DataFrame(rows)
    T["sig_bh"] = _bh(T.p_cohort.to_numpy()) if len(T) else []
    return T


def similarity(L: dict) -> pd.DataFrame:
    rows = []
    for a, mk in L.items():
        for k, v in mk.items():
            pre = np.flatnonzero(v["epoch"] == "pre")
            if len(pre) < 2:
                continue
            for i in range(len(v["label"])):
                ref = v["M"][[j for j in pre if j != i]].mean(0)
                rows.append(dict(animal=a, map=k, label=v["label"][i], epoch=v["epoch"][i], day=v["day"][i],
                                 r_to_pre=float(np.corrcoef(v["M"][i], ref)[0, 1]), gain=float(v["gain"][i])))
    return pd.DataFrame(rows)


def change_maps(L: dict, keys: list) -> dict:
    rng = np.random.default_rng(0)
    res = {}
    for k in keys:
        have = {a: L[a][k] for a in AN if a in L and k in L[a]}
        pre = {a: v["M"][v["epoch"] == "pre"] for a, v in have.items() if (v["epoch"] == "pre").sum() >= 2}
        for ep in EP:
            lvl = {a: v["M"][v["epoch"] == ep] for a, v in have.items() if (v["epoch"] == ep).sum() >= 2 and a in pre}
            if not lvl:
                continue
            kinds = ("level",) if ep == "pre" else ("change",)
            for kind in kinds:
                ref = pre if kind == "change" else None
                am = np.vstack([lvl[a].mean(0) - (ref[a].mean(0) if ref else 0) for a in sorted(lvl)])
                point = am.mean(0)
                B = _boot(lvl, rng, ref)
                p = np.minimum(1.0, 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0)) + 1 / len(B))
                sig = _bh(p)
                agree = (np.sign(am) == np.sign(point)[None, :]).sum(0)
                res[(k, ep, kind)] = dict(point=point, sig=sig & (agree >= min(AGREE_MIN, len(am))), n=len(am),
                                          frac=float((sig & (agree >= min(AGREE_MIN, len(am)))).mean()))
    return res


def _imaging_grid():
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
    return to_img, ol


def plot_change(res: dict, keys: list, fam: str, T: pd.DataFrame, to_img, ol) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("pre", "level"), ("acute", "change"), ("subacute", "change"), ("chronic", "change")]
    fig, axs = plt.subplots(len(keys), 4, figsize=(11.5, 2.55 * len(keys) + 1.2), squeeze=False)
    for i, k in enumerate(keys):
        lv = res.get((k, "pre", "level"))
        lim_l = np.nanpercentile(np.abs(lv["point"]), 99) if lv else 1
        chg = [res[(k, e, "change")]["point"] for e, kd in cols[1:] if (k, e, "change") in res]
        lim_c = np.nanpercentile(np.abs(np.concatenate(chg)), 99) if chg else 1
        for j, (ep, kind) in enumerate(cols):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            r = res.get((k, ep, kind))
            if r is None:
                ax.axis("off")
                continue
            lim = lim_l if kind == "level" else lim_c
            ax.imshow(to_img(r["point"]), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.contour(to_img(r["sig"].astype(float), 0), levels=[0.5], colors="k", linewidths=0.8)
            sub = ""
            if kind == "change":
                t = T[(T.map == k) & (T.epoch == ep)]
                if len(t):
                    t = t.iloc[0]
                    sub = (f"\nmap differs: d {t.d_mean:.2f}, p {t.p_cohort:.3f}{'*' if t.sig_bh else ''}, "
                           f"{int(t.n_sig_animals)}/{int(t.n_animals)} animals")
            ax.set_title(f"{'pre (level)' if kind == 'level' else ep + ' - pre'} n={r['n']}, sig {100 * r['frac']:.0f}%"
                         + sub, fontsize=7)
            if j == 0:
                ax.set_ylabel(k.replace("trial/", "cued ").replace("iti/", "ITI ").replace("sp_", ""), fontsize=8.5)
    fig.suptitle(f"NEURAL: fully per-SESSION Haufe maps ({fam}), each session map scaled to unit RMS (SHAPE only: a "
                 "cortex-wide gain change cannot appear).\nColumn 1 = pre map; columns 2-4 = epoch - pre, mean of "
                 "animals; outline = per-pixel nested bootstrap CI excludes 0 after BH q 0.05 AND >= 3 animals share the "
                 "sign.\n'map differs' = whole-map permutation test: d = 1 - r(epoch mean map, pre mean map), session "
                 "labels permuted within animal; * = BH over all maps x epochs. ipsi (lesion) = image LEFT. PROVISIONAL",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_size_inches()[1]))
    fig.savefig(il.OUT / "figures" / f"iti_session_pattern_change_{fam}.png", dpi=95)
    plt.close(fig)


def plot_similarity(S: pd.DataFrame, keys: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(keys), 4, figsize=(15, 2.3 * len(keys) + 1), squeeze=False, sharex=True)
    for i, k in enumerate(keys):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = S[(S.map == k) & (S.animal == a)].sort_values("day")
            ax.axvspan(g.day.min() - 1 if len(g) else -12, 0, color="0.93", zorder=0)
            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.plot(g.day, g.r_to_pre, "o-", color="k", ms=3, lw=1)
            ax.set_ylim(-0.4, 1.0)
            ax.axhline(0, color="0.7", lw=0.5, ls=":")
            if len(g):
                ax.axhline(g[g.epoch == "pre"].r_to_pre.mean(), color="#2980b9", lw=0.8)
            ax.set_title(f"{a}: {k.replace('trial/', 'cued ').replace('iti/', 'ITI ')}", fontsize=8)
            if i == len(keys) - 1:
                ax.set_xlabel("day after stroke", fontsize=8)
            if j == 0:
                ax.set_ylabel("r(session map, pre map)", fontsize=7.5)
    fig.suptitle("NEURAL: per-session map SHAPE vs the animal's pre map over days (pre sessions vs the mean of the OTHER "
                 "pre sessions = the noise floor, blue line). Grey band = pre. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_size_inches()[1]))
    fig.savefig(il.OUT / "figures" / "iti_session_pattern_similarity.png", dpi=90)
    plt.close(fig)


def plot_maplevel(T: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    maps = [k for f in FAMILIES.values() for k in f if k in set(T.map)]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(maps) + 1.8))
    M = np.full((len(maps), 3), np.nan)
    for i, k in enumerate(maps):
        for j, ep in enumerate(EP[1:]):
            t = T[(T.map == k) & (T.epoch == ep)]
            if len(t):
                t = t.iloc[0]
                M[i, j] = t.d_mean
                ax.text(j, i, f"{t.d_mean:.2f}{'*' if t.sig_bh else ''} {int(t.n_sig_animals)}/{int(t.n_animals)}",
                        ha="center", va="center", fontsize=7, fontweight="bold" if t.sig_bh else "normal")
    ax.imshow(M, cmap="Reds", vmin=0, vmax=np.nanmax(M) if np.isfinite(M).any() else 1, aspect="auto")
    ax.set_xticks(range(3), EP[1:])
    ax.set_yticks(range(len(maps)), [k.replace("trial/", "cued ").replace("iti/", "ITI ") for k in maps], fontsize=7.5)
    ax.set_title("Whole-map SHAPE change vs pre: d = 1 - r(epoch map, pre map), unit-RMS session maps;\n* = session-"
                 "label permutation within animal, cohort p, BH q 0.05; k/n = animals with own p < 0.05", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(il.OUT / "figures" / "iti_session_pattern_maplevel.png", dpi=100)
    plt.close(fig)


def main() -> int:
    L = load()
    if not L:
        print("no session maps yet (run iti_session_patterns.py)")
        return 1
    T = maplevel(L)
    T.to_csv(il.OUT / "session_pattern_maplevel.csv", index=False)
    S = similarity(L)
    S.to_csv(il.OUT / "session_pattern_similarity.csv", index=False)
    G = S.groupby(["map", "animal", "epoch"]).gain.mean().unstack("epoch").reindex(columns=EP)
    G.to_csv(il.OUT / "session_pattern_gain.csv")
    pd.set_option("display.width", 230)
    print("== per animal (rule 8): whole-map shape distance d and permutation p, by epoch")
    print(T[["map", "epoch", "d_mean", "p_cohort", "sig_bh", "n_sig_animals", "n_animals"]
            + [c for c in T.columns if c.startswith(("d_PS", "p_PS"))]].round(3).to_string(index=False))
    print("\n== map GAIN (session map RMS / animal's pre RMS), per animal")
    print(G.round(2).to_string())
    to_img, ol = _imaging_grid()
    for fam, keys in FAMILIES.items():
        keys = [k for k in keys if any(k in L[a] for a in L)]
        if keys:
            plot_change(change_maps(L, keys), keys, fam, T, to_img, ol)
    plot_similarity(S, ["iti/lat", "trial/lat", "iti/wlat", "trial/wlat", "trial/sp_far_contra"])
    plot_maplevel(T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
