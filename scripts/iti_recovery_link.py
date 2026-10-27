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
    args = ap.parse_args(argv)
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
    G = S[S.epoch != "pre"].merge(B, on="label", how="inner").dropna(subset=["beh", "day"])
    G = G[G.day > 0].copy()
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
    plot_course(G, R, tag, ax2, args.z, src)
    return 0


def plot_course(G: pd.DataFrame, R: pd.DataFrame, tag: str, ax2: str, z: bool, src: Path) -> None:
    """Per animal over post-stroke days: behaviour (/ pre) and the decoder's lateral offset beyond the pull."""
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
                 f"recovers, does the decoder's contralesional reading fade?\nBlack = {blab} (one dot per post-stroke "
                 "session); red = FROZEN pre-stroke decoder's lateral offset beyond its pre pull (Allen areas, lags "
                 "0..0.2 s;\n+ = cortex reads the lick more contralesional than it landed)" + zl + ". TOP = ITI licks "
                 "(different licks from those defining recovery), BOTTOM = cued licks.\nMean within-animal r: ITI "
                 f"{rr.loc[('iti', 'raw'), 'mean_r']:+.2f} (time trend removed {rr.loc[('iti', 'detrended'), 'mean_r']:+.2f},"
                 f" p {rr.loc[('iti', 'detrended'), 'p_perm']:.4f}); cued {rr.loc[('trial', 'raw'), 'mean_r']:+.2f} "
                 f"({rr.loc[('trial', 'detrended'), 'mean_r']:+.2f}, p {rr.loc[('trial', 'detrended'), 'p_perm']:.4f}); "
                 "p = within-animal permutation of behaviour.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(src / "figures" / f"iti_recovery_link_{tag}.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
