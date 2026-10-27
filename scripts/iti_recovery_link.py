"""Do the frozen where-decoder's post-stroke changes (r loss, beyond-pull offsets, MAE) track behavioural recovery?
(Priya 10-26: "does PS94, who had the least complete recovery, show the most persistence of the decreased r and
increased error signal?")

Behaviour (--beh) per session, / the animal's pre mean: "hit" = far-contra hit rate (`recovery_timing.series`;
saturates chronically, all animals ~1.0); "angle" = mean tongue angle atan2(lateral, sagittal) (= encoder exec_angle)
of cued far-contra licks; "lateral" = their mean lateral protrusion (side_ml). The kinematic measures use the same
licks the decoder scores (not independent of r / offsets; read with that in mind).
Decoder measures per session and lick set (ITI / in-trial), from the where-decoder outputs:
  r_lat      decoded vs actual lateral r / the animal's pre mean (`where_decoder.csv`, Allen areas)
  bey_lat    mean decoded lateral position beyond the pre pull (as `iti_where_beyond_pull.py`), minus pre mean
  bey_<ax2>  same on the second axis (sagittal, or outward in the 10-26 outward backup)
  mae_lat, mae_2d   MAE / pre mean (`where_mae_sessions.csv`; includes pull + offsets, no direction)
Two levels:
  BETWEEN animals (n = 4, descriptive only): chronic means per animal, ranked against chronic behaviour.
  WITHIN animal across post sessions: Pearson r of measure vs behaviour, raw and with log(day) regressed out of both
  (the shared recovery trend); mean of animals, within-animal permutation of behaviour (N_PERM).
Usage: python scripts/iti_recovery_link.py [--src DIR]   (DIR default <E>/iti_licks; output written there)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from recovery_timing import series  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
N_PERM = 5000


def _files(src: Path, z: bool) -> tuple[Path, Path]:
    """(per-lick predictions, per-session decoder r) for unscaled or within-session z-scored features."""
    if z:
        return src / "zscore_check" / "where_preds_z.csv", src / "zscore_check" / "where_decoder_z.csv"
    return src / "where_preds.csv", src / "where_decoder.csv"


def session_measures(src: Path, z: bool = False) -> tuple[pd.DataFrame, str]:
    fp, fd = _files(src, z)
    p = pd.read_csv(fp)
    ax2 = "sag" if "act_sag" in p.columns else "out"
    p = p.dropna(subset=["act_lat", f"act_{ax2}", "dec_lat", f"dec_{ax2}"])
    parts = []
    for (_st, _a), g in p.groupby(["set", "animal"]):
        g = g.copy()
        for ax_ in ("lat", ax2):
            pre = g[g.epoch == "pre"]
            b, c = np.polyfit(pre[f"act_{ax_}"], pre[f"dec_{ax_}"], 1)
            g[f"bey_{ax_}"] = g[f"dec_{ax_}"] - (c + b * g[f"act_{ax_}"])
        g["mae_lat"] = (g.dec_lat - g.act_lat).abs()
        g["mae_2d"] = np.hypot(g.dec_lat - g.act_lat, g[f"dec_{ax2}"] - g[f"act_{ax2}"])
        parts.append(g)
    S = (pd.concat(parts).groupby(["set", "animal", "epoch", "label"])[["bey_lat", f"bey_{ax2}", "mae_lat", "mae_2d"]]
         .mean().reset_index())
    D = pd.read_csv(fd)
    D = D[(D.source == "areas") & (D.target == "lateral")][["set", "label", "r"]].rename(columns={"r": "r_lat"})
    S = S.merge(D, on=["set", "label"], how="left")
    meas = ["r_lat", "bey_lat", f"bey_{ax2}", "mae_lat", "mae_2d"]
    pre = S[S.epoch == "pre"].groupby(["set", "animal"])[meas].mean()
    for m in meas:
        ref = np.array([pre.loc[(s_, a_), m] if (s_, a_) in pre.index else np.nan for s_, a_ in zip(S.set, S.animal)])
        S[m] = S[m] / ref if m.startswith(("r_", "mae")) else S[m] - ref
    return S, ax2


def _resid(y, d):
    X = np.column_stack([np.ones(len(y)), d])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def within(G: pd.DataFrame, meas: list[str], beh: dict) -> dict:
    out = {}
    for st in ("trial", "iti"):
        for m in meas:
            for kind in ("raw", "detrended"):
                rs = []
                for a in AN:
                    g = G[(G.animal == a) & (G.set == st)].dropna(subset=[m])
                    if len(g) < 5:
                        rs.append(np.nan)
                        continue
                    b, y = beh[a][g.idx.to_numpy()], g[m].to_numpy(float)
                    if kind == "detrended":
                        d = np.log(g.day.to_numpy(float))
                        b, y = _resid(b, d), _resid(y, d)
                    rs.append(float(np.corrcoef(b, y)[0, 1]))
                out[(st, m, kind)] = rs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(il.OUT))
    ap.add_argument("--beh", choices=("hit", "angle", "lateral"), default="hit")
    ap.add_argument("--z", action="store_true", help="decoder on within-session Z-SCORED features")
    ap.add_argument("--plot-only", action="store_true", help="redraw the pooled + session-scatter figures from CSVs")
    args = ap.parse_args(argv)
    if args.plot_only:
        tag = args.beh + ("_z" if args.z else "")
        GA = pd.read_csv(Path(args.src) / f"recovery_link_sessions_withpre_{tag}.csv")
        plot_pooled(GA, tag, args.z, Path(args.src))
        plot_session_scatter(GA, tag, args.z, Path(args.src))
        return 0
    src, tag = Path(args.src), args.beh + ("_z" if args.z else "")
    S, ax2 = session_measures(src, args.z)
    meas = ["r_lat", "bey_lat", f"bey_{ax2}", "mae_lat", "mae_2d"]
    B = series("lick")[["label", "day", "beh"]]
    if args.beh != "hit":
        p = pd.read_csv(_files(src, args.z)[0])
        p = p[(p.set == "trial") & (p.pos == "far_contra")].dropna(subset=["act_lat", "act_sag"])
        p["k"] = np.degrees(np.arctan2(p.act_lat, p.act_sag)) if args.beh == "angle" else p.act_lat
        k = p.groupby(["animal", "epoch", "label"]).k.mean().reset_index()
        k["beh"] = k.k / k.animal.map(k[k.epoch == "pre"].groupby("animal").k.mean())
        B = B[["label", "day"]].merge(k[["label", "beh"]], on="label")
    GA = S.merge(B, on="label", how="inner").dropna(subset=["beh", "day"])   # pre sessions kept for the figures
    G = GA[(GA.epoch != "pre") & (GA.day > 0)].copy()
    pd.set_option("display.width", 220)
    # BETWEEN animals (descriptive)
    bb = G.drop_duplicates("label").groupby(["animal", "epoch"]).beh.mean().unstack()[EP]
    print("== far-contra hit rate / pre (behaviour), per animal\n" + bb.round(2).to_string())
    for st in ("trial", "iti"):
        T = G[G.set == st].groupby(["animal", "epoch"])[meas].mean().unstack("epoch")
        T = T.reindex(columns=EP, level=1)
        print(f"\n== {st}: per animal (r, MAE = / pre; bey = change vs pre, mm)\n" + T.round(2).to_string())
    # WITHIN animal
    sess = {a: sorted(G[G.animal == a].label.unique()) for a in AN}
    beh = {a: G.drop_duplicates("label").set_index("label").beh.reindex(sess[a]).to_numpy(float) for a in AN}
    G["idx"] = [sess[a].index(x) for a, x in zip(G.animal, G.label)]
    obs = within(G, meas, beh)
    rng = np.random.default_rng(0)
    null = {k: [] for k in obs}
    for _ in range(N_PERM):
        s_ = within(G, meas, {a: rng.permutation(v) for a, v in beh.items()})
        for k, v in s_.items():
            null[k].append(np.nanmean(v))
    rows = []
    for (st, m, kind), rs in obs.items():
        mu = np.nanmean(rs)
        p = (np.sum(np.abs(null[(st, m, kind)]) >= abs(mu)) + 1) / (N_PERM + 1)
        rows.append({"set": st, "measure": m, "kind": kind, **dict(zip(AN, np.round(rs, 2))), "mean_r": round(mu, 3),
                     "p_perm": round(p, 4)})
    R = pd.DataFrame(rows)
    print("\n== within-animal r(measure, behaviour / pre) over post sessions\n" + R.to_string(index=False))
    R.to_csv(src / f"recovery_link_within_{tag}.csv", index=False)
    G.to_csv(src / f"recovery_link_sessions_{tag}.csv", index=False)
    GA.to_csv(src / f"recovery_link_sessions_withpre_{tag}.csv", index=False)
    plot_course(GA, R, tag, ax2, args.z, src)
    plot_pooled(GA, tag, args.z, src)
    plot_session_scatter(GA, tag, args.z, src)
    return 0


def plot_session_scatter(GA: pd.DataFrame, tag: str, z: bool, src: Path) -> None:
    """POOLED over animals, one dot per SESSION (pre included): decoder lateral offset vs far-contra aim, colour =
    animal, marker = epoch; a line per animal = its least-squares fit over post sessions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blab = {"hit": "far-contra hit rate / pre", "angle": "far-contra cued lick angle / pre",
            "lateral": "far-contra lateral protrusion / pre"}[tag.replace("_z", "")]
    cols = plt.get_cmap("tab10")
    mk = {"pre": "o", "acute": "^", "subacute": "s", "chronic": "*"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        ax = axs[0, i]
        g = GA[GA.set == st]
        for k, a in enumerate(AN):
            h = g[g.animal == a]
            for ep, m_ in mk.items():
                q = h[h.epoch == ep]
                ax.scatter(q.beh, q.bey_lat, marker=m_, s=40 if m_ == "*" else 22, color=cols(k), alpha=0.8,
                           edgecolor="k" if ep == "pre" else "none", lw=0.4, label=a if ep == "acute" else None)
            post = h[h.epoch != "pre"].dropna(subset=["beh", "bey_lat"])
            if len(post) >= 5:
                b, c = np.polyfit(post.beh, post.bey_lat, 1)
                xx = np.linspace(post.beh.min(), post.beh.max(), 20)
                ax.plot(xx, c + b * xx, color=cols(k), lw=1.2)
        post = g[g.epoch != "pre"].dropna(subset=["beh", "bey_lat"])
        r_all = float(np.corrcoef(post.beh, post.bey_lat)[0, 1])
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.axvline(1, color="0.7", lw=0.6, ls=":")
        for ep, m_ in mk.items():
            ax.scatter([], [], marker=m_, color="0.5", label=ep)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.set_xlabel(blab, fontsize=9)
        ax.set_ylabel("lateral offset beyond pre pull, mm (change vs pre;\n+ = read contralesional)", fontsize=9)
        ax.set_title(f"{'ITI' if st == 'iti' else 'cued'} licks: all post sessions pooled r {r_all:+.2f} "
                     "(between + within animal)", fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle(("Z-SCORED FEATURES. " if z else "") + "POOLED, one dot per SESSION: FROZEN decoder's lateral offset "
                 "beyond its pre pull vs far-contra aim. Colour = animal, marker = epoch (outlined = pre-stroke);\n"
                 "lines = per-animal fit over post sessions. The within-animal test with the shared time trend removed "
                 "is in 3.37 (this pooled r mixes animals and includes the time trend). PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(src / "figures" / f"iti_recovery_link_sessions_{tag}.png", dpi=100)
    plt.close(fig)


def plot_pooled(GA: pd.DataFrame, tag: str, z: bool, src: Path) -> None:
    """Per epoch (pre included): behaviour and lateral offset, animal means + nested-bootstrap CI of the change vs pre,
    and the per-animal epoch trajectory in (behaviour, offset) space. Per-animal table printed first (rule 8)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import analysis_kit as ak
    eps = ["pre", *EP]
    blab = {"hit": "far-contra hit rate / pre", "angle": "far-contra lick angle / pre",
            "lateral": "far-contra lateral protrusion / pre"}[tag.replace("_z", "")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), squeeze=False)
    cols = plt.get_cmap("tab10")
    rows = []
    for i, st in enumerate(("iti", "trial")):
        g = GA[GA.set == st]
        am = g.groupby(["animal", "epoch"])[["beh", "bey_lat"]].mean()
        print(f"\n== POOLED {st}: per animal x epoch (beh / pre; bey_lat change vs pre, mm)\n"
              + am.unstack("epoch").reindex(columns=eps, level=1).round(3).to_string())
        for j, (val, lab) in enumerate((("beh", blab), ("bey_lat", "lateral offset beyond pre pull, mm\n"
                                                                   "(+ = read contralesional; change vs pre)"))):
            ax = axs[i, j]
            prd = {a: h[val].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for k, a in enumerate(AN):
                y = [am[val].get((a, e), np.nan) for e in eps]
                ax.plot(range(4), y, "o-", color=cols(k), ms=5, lw=1, alpha=0.75, label=a)
            mu = [am[val].xs(e, level="epoch").mean() for e in eps]
            ax.plot(range(4), mu, "-", color="k", lw=2.5, zorder=5, label="mean of animals")
            for x, e in enumerate(EP, start=1):
                post = {a: h[val].to_numpy() for a, h in g[g.epoch == e].groupby("animal")}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                ref = mu[0]
                ax.errorbar(x + 0.12, ref + ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="D",
                            color="k", ms=6, capsize=3, zorder=6)
                sig = ci.lo > 0 or ci.hi < 0
                ax.text(x + 0.12, ref + ci.hi, f"{ci.point:+.2f}" + (" *" if sig else ""), ha="center",
                        va="bottom", fontsize=8)
                rows.append(dict(set=st, measure=val, epoch=e, point=ci.point, lo=ci.lo, hi=ci.hi,
                                 n_dir=int(((am[val].xs(e, level="epoch") - am[val].xs("pre", level="epoch")) *
                                            np.sign(ci.point) > 0).sum())))
            ax.axhline(1 if val == "beh" else 0, color="0.7", lw=0.7, ls=":")
            ax.set_xticks(range(4), eps)
            ax.set_ylabel(lab, fontsize=8.5)
            ax.set_title(f"{'ITI' if st == 'iti' else 'cued'} licks: {'behaviour' if val == 'beh' else 'decoder'}",
                         fontsize=9.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5, frameon=False)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        ax = axs[i, 2]
        for k, a in enumerate(AN):
            x = [am["beh"].get((a, e), np.nan) for e in eps]
            y = [am["bey_lat"].get((a, e), np.nan) for e in eps]
            ax.plot(x, y, "-", color=cols(k), lw=1, alpha=0.7)
            for _e, xx, yy, mk in zip(eps, x, y, ("o", "^", "s", "*")):
                ax.plot(xx, yy, mk, color=cols(k), ms=7 if mk != "*" else 11, mec="k", mew=0.4)
        ax.plot([], [], "o", color="0.5", label="pre")
        ax.plot([], [], "^", color="0.5", label="acute")
        ax.plot([], [], "s", color="0.5", label="subacute")
        ax.plot([], [], "*", color="0.5", ms=10, label="chronic")
        ax.legend(fontsize=7.5, frameon=False)
        ax.axhline(0, color="0.7", lw=0.7, ls=":")
        ax.axvline(1, color="0.7", lw=0.7, ls=":")
        ax.set_xlabel(blab, fontsize=8.5)
        ax.set_ylabel("lateral offset (mm, change vs pre)", fontsize=8.5)
        ax.set_title("per animal, epoch means: aim vs decoder offset", fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    pd.DataFrame(rows).to_csv(src / f"recovery_link_pooled_{tag}.csv", index=False)
    fig.suptitle(("Z-SCORED FEATURES. " if z else "") + "POOLED per epoch (pre included): far-contra aim and the "
                 "FROZEN pre-stroke decoder's lateral offset beyond its pre pull.\nThin lines = animals (session "
                 "means per epoch); black line = mean of animals; diamonds = mean of animals' pre + change vs pre "
                 "with 95% nested bootstrap CI\n(animals > sessions; * = CI excludes 0). TOP = ITI licks, BOTTOM = "
                 "cued licks. Behaviour is the same in both rows (cued far-contra licks).", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(src / "figures" / f"iti_recovery_link_pooled_{tag}.png", dpi=100)
    plt.close(fig)


def plot_course(G: pd.DataFrame, R: pd.DataFrame, tag: str, ax2: str, z: bool, src: Path) -> None:
    """Per animal over days, pre-stroke sessions included (left of the stroke line): behaviour (/ pre) and the
    decoder's lateral offset beyond the pull. Pre values scatter around 1 / 0 by construction (the reference)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blab = {"hit": "far-contra hit rate / pre", "angle": "far-contra cued lick ANGLE / pre",
            "lateral": "far-contra cued lick lateral protrusion / pre"}[tag.replace("_z", "")]
    zl = " [decoder on Z-SCORED features]" if z else ""
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex="col", squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = G[(G.animal == a) & (G.set == st)].sort_values("day")
            ax.plot(g.day, g.beh, "o-", color="k", ms=4, lw=1.2, label=blab)
            ax.axhline(1, color="0.8", lw=0.6)
            ax.set_ylabel(blab if j == 0 else "", fontsize=8.5)
            ax2_ = ax.twinx()
            ax2_.plot(g.day, g.bey_lat, "s-", color="#c0392b", ms=4, lw=1.2, alpha=0.85,
                      label="decoded lateral offset beyond pre pull (mm, change vs pre)")
            ax2_.axhline(0, color="#c0392b", lw=0.5, ls=":")
            ax.axvspan(g.day.min() - 1, 0, color="0.93", zorder=0)
            ax.axvline(0, color="k", lw=0.8, ls="--")
            if i == 0 and j == 0:
                ax.text(0, 1.0, " stroke", transform=ax.get_xaxis_transform(), fontsize=7.5, va="top")
            if j == 3:
                ax2_.set_ylabel("lateral offset, mm\n(+ = read contralesional)" + ("\n" + zl.strip() if z else ""),
                                fontsize=8.5, color="#c0392b")
            r = R[(R.set == st) & (R.measure == "bey_lat")].set_index("kind")[a]
            ax.set_title(f"{a}, {'ITI' if st == 'iti' else 'cued'} licks: r {r['raw']:+.2f} "
                         f"(time trend removed {r['detrended']:+.2f})", fontsize=9)
            if i == 1:
                ax.set_xlabel("day after stroke", fontsize=9)
            for s_ in ("top",):
                ax.spines[s_].set_visible(False)
    rr = R[(R.measure == "bey_lat")].set_index(["set", "kind"])
    fig.suptitle(("Z-SCORED FEATURES. " if z else "") + "NEURAL vs BEHAVIOUR, within animal: as far-contra aim "
                 f"recovers, does the decoder's contralesional reading fade?\nBlack = {blab} (one dot per session; grey = PRE-STROKE "
                 "sessions, the reference); red = FROZEN pre-stroke decoder's lateral offset beyond its pre pull (Allen areas, lags "
                 "0..0.2 s;\n+ = cortex reads the lick more contralesional than it landed)" + zl + ". TOP = ITI licks "
                 "(different licks from those defining recovery), BOTTOM = cued licks.\nMean within-animal r: ITI "
                 f"{rr.loc[('iti', 'raw'), 'mean_r']:+.2f} (time trend removed {rr.loc[('iti', 'detrended'), 'mean_r']:+.2f},"
                 f" p {rr.loc[('iti', 'detrended'), 'p_perm']:.4f}); cued {rr.loc[('trial', 'raw'), 'mean_r']:+.2f} "
                 f"({rr.loc[('trial', 'detrended'), 'mean_r']:+.2f}, p {rr.loc[('trial', 'detrended'), 'p_perm']:.4f}); "
                 "p = within-animal permutation of behaviour.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(src / "figures" / f"iti_recovery_link_{tag}.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
