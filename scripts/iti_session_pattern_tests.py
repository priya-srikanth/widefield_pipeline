"""Map-level permutation tests + AMPLITUDE-NORMALIZED change maps on the fully per-session Haufe maps
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
(3) AMPLITUDE-NORMALIZED CHANGE MAPS: per pixel, epoch - pre on the unit-RMS maps, paired by animal, nested bootstrap
    (animals > sessions, `iti_decoder_pattern_stats._boot`), BH q 0.05 over cortex, outline = significant AND >= 3
    animals share the sign. Level maps (epoch mean) drawn alongside.
Outputs <E>/iti_licks/session_pattern_{maplevel,similarity,gain}.csv, figures/lick_session_pattern_maplevel.png,
figures/lick_session_pattern_similarity.png, figures/lick_session_pattern_change_{position,within_spout}.png,
figures/lick_{cued,iti}_session_pattern_change_spout.png.
Figure data (figdata.py convention): <E>/iti_licks/figdata/<figure stem>.pkl holds everything each figure draws (the
whole-map test table, the similarity table, and per change figure the change_maps results + the imaging grid), so
`--plot-only` redraws all six figures from those files alone -- no permutation, no bootstrap (the change maps took hours).
Usage: python scripts/iti_session_pattern_tests.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import CHANGE_CMAP, CMAP_NOTE, PRE_CMAP, grid_fns, imaging_grid, load_fd, plot_only, save_fd  # noqa: E402
from iti_decoder_pattern_stats import AGREE_MIN, _boot, _maxstat  # noqa: E402
from iti_session_patterns import AN, OUT, POS  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP = ["pre", "acute", "subacute", "chronic"]
N_PERM = 5000
STRIDE = 8          # whole-map distance on every 8th pixel (maps are smooth: 200 SVD components)
FAMILIES = {
    "position": ["iti/lat", "trial/lat", "iti/sag", "trial/sag"],
    "within_spout": ["iti/wlat", "trial/wlat", "iti/wsag", "trial/wsag"],
    "spout_cued": [f"trial/sp_{p}" for p in POS],
    "spout_iti": [f"iti/sp_{p}" for p in POS],
}
CHANGE_FIG = {"position": "lick_session_pattern_change_position.png",       # lick_ names since 10-27
              "within_spout": "lick_session_pattern_change_within_spout.png",
              "spout_cued": "lick_cued_session_pattern_change_spout.png",
              "spout_iti": "lick_iti_session_pattern_change_spout.png"}


def load(only: list | None = None) -> dict:
    """{animal: {map key: {"epoch": [...], "day": [...], "M": (n, pix) unit-RMS, "gain": (n,)}}}; `only` = keep these
    map keys (the change-map workers load just their family)."""
    out = {}
    for a in AN:
        f = OUT / f"{a}.pkl"
        if not f.exists():
            continue
        S = pickle.loads(f.read_bytes())
        keys = sorted({k for r in S.values() for k in r["maps"]})
        if only is not None:
            keys = [k for k in keys if k in set(only)]
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
            # DRIFT FLOOR: the label permutation also detects slow drift (pre sessions precede post ones), so read d
            # against the distance between each animal's EARLY and LATE pre sessions (split at the median day)
            dpp = []
            for a in an:
                v = L[a][k]
                o = np.argsort(np.where(v["epoch"] == "pre", v["day"], np.inf))[: int((v["epoch"] == "pre").sum())]
                if len(o) >= 4:
                    dpp.append(_dist(v["M"][o[: len(o) // 2], ::STRIDE], v["M"][o[len(o) // 2:], ::STRIDE]))
            row = dict(map=k, epoch=ep, n_animals=len(an), d_mean=D, d_pre_early_vs_late=float(np.mean(dpp)) if dpp
                       else np.nan, p_cohort=(np.sum(Dn >= D) + 1) / (N_PERM + 1))
            ps = {}
            for a in an:
                ps[a] = (np.sum(null[a] >= obs[a]) + 1) / (N_PERM + 1)
                row[f"d_{a}"] = obs[a]
                row[f"p_{a}"] = ps[a]
            row["n_sig_animals"] = int(sum(p < 0.05 for p in ps.values()))
            rows.append(row)
    T = pd.DataFrame(rows)
    T["sig_bh"] = ak.bh(T.p_cohort.to_numpy()) if len(T) else []
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
                p = ak.boot_p_columns(B)                  # shared p (decision 9, Priya 11-05)
                sig = ak.bh(p)
                agree = (np.sign(am) == np.sign(point)[None, :]).sum(0)
                res[(k, ep, kind)] = dict(point=point, sig=sig & (agree >= min(AGREE_MIN, len(am))), n=len(am),
                                          frac=float((sig & (agree >= min(AGREE_MIN, len(am)))).mean()),
                                          sig_max=_maxstat(B, point))
    return res


def _change_worker(item):
    """Module-level (spawn) worker: one map family's change maps. Each family's change_maps starts its own rng(0), as the
    serial loop did, so fanning the families out leaves every draw unchanged."""
    _fam, keys = item
    return change_maps(load(keys), list(keys))


def plot_change(D: dict) -> None:
    """D = figure data: res (change_maps output), keys, fam, T (whole-map test table), grid."""
    res, keys, fam, T = D["res"], D["keys"], D["fam"], D["T"]
    to_img, ol = grid_fns(D["grid"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("pre", "level"), ("acute", "change"), ("subacute", "change"), ("chronic", "change")]
    fig, axs = plt.subplots(len(keys), 4, figsize=(12.8, 2.6 * len(keys) + 1.4), squeeze=False, layout="constrained")
    for i, k in enumerate(keys):
        im_l = im_c = None
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
            if kind == "level":                     # pre LEVEL: purple-orange, own scale (Priya 10-27)
                im_l = ax.imshow(to_img(r["point"]), cmap=PRE_CMAP, vmin=-lim_l, vmax=lim_l, interpolation="nearest")
            else:
                im_c = ax.imshow(to_img(r["point"]), cmap=CHANGE_CMAP, vmin=-lim_c, vmax=lim_c, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.contour(to_img(r["sig"].astype(float), 0), levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
            ax.contour(to_img(r["sig_max"].astype(float), 0), levels=[0.5], colors="k", linewidths=1.3)
            sub = ""
            if kind == "change":
                t = T[(T["map"] == k) & (T.epoch == ep)]
                if len(t):
                    t = t.iloc[0]
                    sub = (f"\nd {t.d_mean:.2f} (pre early-late {t.d_pre_early_vs_late:.2f}), p {t.p_cohort:.3f}"
                           f"{'*' if t.sig_bh else ''}, "
                           f"{int(t.n_sig_animals)}/{int(t.n_animals)} animals")
            ax.set_title(f"{'pre (level)' if kind == 'level' else ep + ' - pre'} n={r['n']}, sig {100 * r['frac']:.0f}%"
                         + sub, fontsize=7)
            if j == 0:
                ax.set_ylabel(k.replace("trial/", "cued ").replace("iti/", "ITI ").replace("sp_", ""), fontsize=8.5)
        if im_l is not None:
            cb = fig.colorbar(im_l, ax=axs[i, 0], fraction=0.06, pad=0.02)
            cb.set_label("pre LEVEL (unit RMS)", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
        if im_c is not None:
            cb = fig.colorbar(im_c, ax=axs[i, 1:].tolist(), fraction=0.02, pad=0.01)
            cb.set_label("CHANGE epoch - pre", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
    import textwrap
    paras = [f"NEURAL: fully per-SESSION Haufe maps ({fam}), each session map scaled to unit RMS (SHAPE only: a "
             f"cortex-wide gain change cannot appear). {CMAP_NOTE[0].upper() + CMAP_NOTE[1:]}; one scale per row for each.",
             "Column 1 = pre map; columns 2-4 = epoch - pre, mean of animals; SOLID = max-statistic FWER 0.05; DASHED = "
             "per-pixel nested bootstrap CI excludes 0 after BH q 0.05 AND >= 3 animals share the sign.",
             "Titles: d = 1 - r(epoch mean map, pre mean map) (whole-map permutation test, session labels permuted within "
             "animal; * = BH over all maps x epochs). ipsi (lesion) = image LEFT. PROVISIONAL"]
    txt = "\n".join(textwrap.fill(q, 125) for q in paras)       # wrapped to the figure width (titles ran off the sides)
    fig.suptitle(txt, fontsize=8.5)
    save_png_svg(fig, il.OUT / "figures" / CHANGE_FIG[fam], dpi=95)
    plt.close(fig)


SIM_KEYS = ["iti/lat", "trial/lat", "iti/wlat", "trial/wlat", "trial/sp_far_contra"]


def plot_similarity(D: dict) -> None:
    S, keys = D["S"], D["keys"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(keys), 4, figsize=(15, 2.3 * len(keys) + 1), squeeze=False, sharex=True)
    for i, k in enumerate(keys):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = S[(S["map"] == k) & (S.animal == a)].sort_values("day")
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
                 "pre sessions = the noise floor, blue line). Gray band = pre. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_size_inches()[1]))
    save_png_svg(fig, il.OUT / "figures" / "lick_session_pattern_similarity.png", dpi=90)
    plt.close(fig)


def plot_maplevel(D: dict) -> None:
    T = D["T"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    maps = [k for f in FAMILIES.values() for k in f if k in set(T["map"])]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(maps) + 1.8))
    M = np.full((len(maps), 3), np.nan)
    for i, k in enumerate(maps):
        for j, ep in enumerate(EP[1:]):
            t = T[(T["map"] == k) & (T.epoch == ep)]
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
    save_png_svg(fig, il.OUT / "figures" / "lick_session_pattern_maplevel.png", dpi=100)
    plt.close(fig)


def trial_set_sentence(fam: str | None = None) -> str:
    """The licks of one map family (``fam``), or of every family (the whole-map test / similarity figures) (Priya 11-09)."""
    from iti_session_patterns import family_sentence
    return family_sentence(fam or "all")


def _plot_change(D: dict) -> None:
    from wfield_local import trial_set_text as tst
    with tst.active(None, text=trial_set_sentence(D["fam"])):
        plot_change(D)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if plot_only():                                               # redraw from figdata only: no permutation, no bootstrap
        plot_maplevel(load_fd("lick_session_pattern_maplevel"))
        plot_similarity(load_fd("lick_session_pattern_similarity"))
        for fam in FAMILIES:
            try:
                _plot_change(load_fd(CHANGE_FIG[fam]))
            except SystemExit as e:
                print(e)
        return 0
    L = load()
    if not L:
        print("no session maps yet (run iti_session_patterns.py)")
        return 1
    if "--reuse-maplevel" in sys.argv:                            # keep the saved whole-map permutation test
        T = pd.read_csv(il.OUT / "session_pattern_maplevel.csv")
    else:
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
    save_fd("lick_session_pattern_maplevel", dict(T=T))
    save_fd("lick_session_pattern_similarity", dict(S=S, keys=SIM_KEYS))
    plot_similarity(dict(S=S, keys=SIM_KEYS))
    plot_maplevel(dict(T=T))
    grid = imaging_grid()
    fams = [(fam, tuple(k for k in keys if any(k in L[a] for a in L))) for fam, keys in FAMILIES.items()]
    fams = [f for f in fams if f[1]]
    del L                                                         # the workers load their own family
    from wfield_local import parallel
    got, fail = parallel.fan_out(fams, _change_worker, jobs=len(fams), label="change-map family")
    if fail:
        raise SystemExit(f"change maps failed: {fail}")
    for (fam, keys), res in sorted(got):
        D = dict(res=res, keys=list(keys), fam=fam, T=T, grid=grid)
        save_fd(CHANGE_FIG[fam], D)
        _plot_change(D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
