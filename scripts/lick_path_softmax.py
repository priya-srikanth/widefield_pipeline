"""TRAJECTORY softmax confusion matrices, the path analogue of the landing matrices (Priya 10-24: "can you make the same
softmax_trajectory_change confusion matrices as you did for landing?"). LICK DATA only.

  1. per lick: the 3-D tongue path on a fixed 4 ms grid from -PRE to +POST ms around the lick's 3-D peak (the TRACKED part
     of a lick: before ~-36 ms the tongue is not seen and earlier frames belong to the previous lick), from the per-frame
     tracks `session_poses/pose3d/<session>_frames.csv.gz`; cached per session under trajectories/per_lick/.
  2. templates: per animal x spout, the mean pre-stroke path and a per-timepoint, per-axis SD pooled within spout
     (pre sessions use leave-one-session-out templates, so a pre lick is never scored against a template it built).
  3. distance of a lick to spout p = mean over its tracked time points of the squared SD-unit distance summed over the
     three axes (>= MIN_PTS points); weights w_p ~ exp(-0.5 d_p^2 / width^2), widths 0.5 and 1 -- the landing rule,
     applied to the path instead of the peak point.
  4. matrices per animal x epoch (row = target spout, column = "path as" spout); epoch minus pre within animal, mean of
     animals, nested bootstrap (animals -> sessions), Bonferroni over 36 cells; second line = animals with the same sign.

Usage: python scripts/lick_path_softmax.py [--jobs 10]
Output: <dlc>/session_poses/pose3d/trajectories/path_softmax_*.csv (+ per-lick path_softmax_licks.csv.gz), figures/path_softmax_*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRE_MS, POST_MS, DT = 32, 32, 4
GRID = np.arange(-PRE_MS, POST_MS + DT, DT)
MIN_PTS = 8
SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
LAB = ["cI", "cM", "cC", "fI", "fM", "fC"]
MOUSE = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
         "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
AN = ["PS92", "PS93", "PS94", "PS95"]
I = Path("C:/Users/SabatiniLab/wfield_cache/pose3d_inputs")


def _dlc() -> Path:
    from wfield_local.paths import PathResolver
    return Path(PathResolver().root("dlc"))


def session_job(item):
    """(session, out dir, licks as (trial_id, t_peak_3d) array) -> cached per-lick paths (n x T x 3, float32)."""
    session, out, L = item
    f = Path(out) / f"{session}.npz"
    if f.exists():
        return str(f)
    F = pd.read_csv(_dlc() / "session_poses" / "pose3d" / f"{session}_frames.csv.gz",
                    usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"]).sort_values(["trial_id", "t_ms"])
    by = {t: g for t, g in F.groupby("trial_id")}
    P = np.full((len(L), len(GRID), 3), np.nan, np.float32)
    for i, (tid, tp) in enumerate(L):
        g = by.get(tid)
        if g is None:
            continue
        t = g.t_ms.to_numpy(float) - tp
        for a, c in enumerate(("tongue_ml", "tongue_ap", "tongue_dv")):
            v = g[c].to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            x = np.interp(GRID, t[ok], v[ok], left=np.nan, right=np.nan)
            # no bridging across tracking gaps longer than 2 frames
            near = np.min(np.abs(t[ok][None, :] - GRID[:, None]), 1) <= 2 * DT
            x[~near] = np.nan
            P[i, :, a] = x
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, paths=P, trial_id=L[:, 0], t_peak=L[:, 1])
    tmp.replace(f)
    return str(f)


def boot_change(S: dict, ep: str, nb=4000, seed=0):
    rng = np.random.default_rng(seed)
    ans = [a for a in AN if (a, ep) in S and (a, "pre") in S]
    obs = np.nanmean([np.nanmean(S[(a, ep)], 0) - np.nanmean(S[(a, "pre")], 0) for a in ans], 0)
    B = np.empty((nb, 6, 6))
    for b in range(nb):
        d = []
        for a in rng.choice(ans, len(ans), replace=True):
            e_, p_ = S[(a, ep)], S[(a, "pre")]
            d.append(np.nanmean([e_[i] for i in rng.integers(0, len(e_), len(e_))], 0)
                     - np.nanmean([p_[i] for i in rng.integers(0, len(p_), len(p_))], 0))
        B[b] = np.nanmean(d, 0)
    al = 0.05 / 36
    lo, hi = np.nanquantile(B, al / 2, 0), np.nanquantile(B, 1 - al / 2, 0)
    agree = np.sum([np.sign(np.nanmean(S[(a, ep)], 0) - np.nanmean(S[(a, "pre")], 0)) == np.sign(obs) for a in ans], 0)
    return obs, (lo > 0) | (hi < 0), agree, len(ans)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args(argv)
    K = pd.read_csv(I / "pose3d_kin_licks_snapshot_20261023.csv.gz",
                    usecols=["animal", "session", "epoch", "trial_id", "position", "t_peak_3d", "date"])
    K = K[(K.date.astype(str).str[4:6] != "06") & K.t_peak_3d.notna()].copy()       # August-on (encoding rule)
    K["target"] = K.position.map(MOUSE)
    K = K.dropna(subset=["target"]).reset_index(drop=True)
    out = _dlc() / "session_poses" / "pose3d" / "trajectories"
    cache = out / "per_lick"
    cache.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    items = [(s, str(cache), g[["trial_id", "t_peak_3d"]].to_numpy(np.int64)) for s, g in K.groupby("session")]
    print(f"[path softmax] {len(items)} sessions, {len(K)} licks", flush=True)
    _res, fail = ak.fan_sessions(items, session_job, jobs=a.jobs, key=lambda it: it[0], label="session")
    if fail:
        for it, err in fail:
            print(f"  !! {it[0]}: {err}", flush=True)
        raise SystemExit("refusing partial results")
    paths = np.full((len(K), len(GRID), 3), np.nan, np.float32)
    for s, g in K.groupby("session"):
        paths[g.index.to_numpy()] = np.load(cache / f"{s}.npz")["paths"]
    rows, mats, per_lick = [], {}, []
    for an in AN:
        ka = K[K.animal == an]
        Pa = paths[ka.index.to_numpy()].astype(np.float64)
        pre = (ka.epoch == "pre").to_numpy()
        sess = ka.session.to_numpy()
        tg = ka.target.to_numpy()
        # per-session sums for LOSO templates
        S1, S2, N = {}, {}, {}
        for s in np.unique(sess[pre]):
            for p in SP:
                m = (sess == s) & (tg == p) & pre
                X = Pa[m]
                S1[(s, p)] = np.nansum(X, 0); S2[(s, p)] = np.nansum(X ** 2, 0); N[(s, p)] = np.isfinite(X).sum(0)
        def template(excl, S1=S1, S2=S2, N=N):
            mu, var, cnt = [], np.zeros((len(GRID), 3)), np.zeros((len(GRID), 3))
            for p in SP:
                s1 = sum(v for (s, q), v in S1.items() if q == p and s != excl)
                s2 = sum(v for (s, q), v in S2.items() if q == p and s != excl)
                n = sum(v for (s, q), v in N.items() if q == p and s != excl)
                m_ = s1 / np.maximum(n, 1)
                ok = n >= 20
                var += np.where(ok, s2 - n * m_ ** 2, 0.0)
                cnt += np.where(ok, n - 1, 0)
                m_[~ok] = np.nan
                mu.append(m_)
            sd = np.sqrt(np.nanmax([var / np.maximum(cnt, 1), np.full_like(var, 1e-6)], 0))
            return np.stack(mu), sd
        cache_t = {}
        D2 = np.full((len(ka), 6), np.nan)
        for s in np.unique(sess):
            key = s if s in set(sess[pre]) else None
            if key not in cache_t:
                cache_t[key] = template(key)
            mu, sd = cache_t[key]
            m = sess == s
            X = Pa[m]
            z = ((X[:, None] - mu[None]) / sd[None, None]) ** 2            # licks x spouts x T x 3
            ok = np.isfinite(z).all(3)
            npts = ok.sum(2)
            d2 = np.where(ok[..., None], z, 0).sum((2, 3)) / np.maximum(npts, 1)
            d2[npts < MIN_PTS] = np.nan
            D2[m] = d2
        for w in (0.5, 1.0):
            good = np.isfinite(D2).all(1)
            e = np.exp(-0.5 * (D2 - np.nanmin(D2, 1, keepdims=True)) / w ** 2)
            W = e / e.sum(1, keepdims=True)
            df = pd.DataFrame(W[good], columns=SP)
            df["epoch"] = ka.epoch.to_numpy()[good]; df["session"] = sess[good]; df["target"] = tg[good]
            df["trial_id"] = ka.trial_id.to_numpy()[good]; df["t_peak_3d"] = ka.t_peak_3d.to_numpy()[good]
            per_lick.append(df.assign(width=w, animal=an))
            print(f"  {an} width {w}: {good.sum()} of {len(ka)} licks scored", flush=True)
            for (ep, s), g in df.groupby(["epoch", "session"]):
                M = np.full((6, 6), np.nan)
                for r, t in enumerate(SP):
                    x = g[g.target == t]
                    if len(x) >= 10:
                        M[r] = x[SP].mean().to_numpy(float)
                mats.setdefault((w, an, ep), []).append(M)
                for r, t in enumerate(SP):
                    if np.isfinite(M[r]).all():
                        rows.append({"width": w, "animal": an, "epoch": ep, "session": s, "target": t,
                                     **{f"as_{q}": M[r, j] for j, q in enumerate(SP)}})
    pd.concat(per_lick).to_csv(out / "path_softmax_licks.csv.gz", index=False)   # per-lick weights (neural comparison)
    R = pd.DataFrame(rows)
    R.to_csv(out / "path_softmax_sessions.csv", index=False)
    stats = []
    for w in (0.5, 1.0):
        S = {(an, ep): v for (ww, an, ep), v in mats.items() if ww == w}
        # per-animal matrices
        fig, axs = plt.subplots(4, 4, figsize=(13, 13))
        for i, an in enumerate(AN):
            for j, ep in enumerate(("pre", "acute", "subacute", "chronic")):
                ax = axs[i, j]
                if (an, ep) in S:
                    M = np.nanmean(S[(an, ep)], 0)
                    ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
                    for r in range(6):
                        for c in range(6):
                            if np.isfinite(M[r, c]) and M[r, c] >= 0.1:
                                ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=6,
                                        color="k" if M[r, c] > 0.6 else "w")
                ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=7)
                ax.set_yticks(range(6)); ax.set_yticklabels(LAB, fontsize=7)
                ax.set_title(f"{an} {ep}", fontsize=9)
        fig.suptitle(f"LICK DATA: trajectory softmax, width {w:g}: rows = target spout, columns = the spout whose pre-stroke mean "
                     f"PATH the lick follows ({-PRE_MS}..+{POST_MS} ms around the peak); rows sum to 1.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / "figures" / f"path_softmax_matrices_w{w:g}.png", dpi=120)
        plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
        for k, ep in enumerate(("acute", "subacute", "chronic")):
            obs, sig, agree, n = boot_change(S, ep)
            ax = axs[k]
            ax.imshow(obs, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            for r in range(6):
                for c in range(6):
                    if sig[r, c]:
                        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec="k", lw=2))
                    if abs(obs[r, c]) >= 0.06 or sig[r, c]:
                        ax.text(c, r, f"{obs[r, c]:+.2f}\n{agree[r, c]}/{n}", ha="center", va="center", fontsize=6.5,
                                color="w" if abs(obs[r, c]) > 0.4 else "k")
                    stats.append({"width": w, "epoch": ep, "target": SP[r], "as": SP[c], "change": obs[r, c],
                                  "significant": bool(sig[r, c]), "same_sign": int(agree[r, c]), "n_animals": n})
            ax.set_xticks(range(6)); ax.set_xticklabels(LAB); ax.set_yticks(range(6)); ax.set_yticklabels(LAB)
            ax.set_title(f"{ep} minus pre", fontsize=10)
            if k == 0:
                ax.set_ylabel("target spout")
            ax.set_xlabel("path follows spout")
        fig.suptitle(f"LICK DATA: trajectory softmax (width {w:g}), epoch minus pre within animal, mean of 4 animals. Boxed = nested "
                     f"bootstrap excludes 0 (Bonferroni over 36); second line = animals with the same sign.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "figures" / f"path_softmax_change_stats_w{w:g}.png", dpi=130)
        plt.close(fig)
    T = pd.DataFrame(stats)
    T.to_csv(out / "path_softmax_change_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(T[T.target == "far_contra"].round(3).to_string(index=False))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
