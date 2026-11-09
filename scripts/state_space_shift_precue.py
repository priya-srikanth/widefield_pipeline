"""Is the pre-lick orofacial rise the same as the PRE-CUE rise against rest? (Priya 11-09 "great keep going" on the open
question: "whether it is the same increase as the pre-cue rise against rest (Fig 9.8)").

Per trial, cohort SVD basis, minus the session's RESTnl level, units of the animal's pre-template pixel RMS (as Figs
10.21-10.23), region means (orofacial = tjM1 + SSp-m + SSp-n; posterior-medial = RSPd + SSp-tr + SSp-ll; contra / ipsi):
  pre-cue    the trial's own lick-free (>= 1 s) pre-cue ENL window (`state_space_enl_baseline.py`, Fig 9.8's window)
  pre-lick   -1.0..-0.1 s before the first lick (first-lick-aligned extract; mostly after the cue)
  post-lick  0..+0.7 s after the first lick
and pre-lick - pre-cue (what the cue -> first-lick period adds on top of the pre-cue state). Session means, change vs
own pre, animal-weighted nested bootstrap (`boot_delta`, 2,000); * = 95 % CI excludes 0, k/4. Plus, across post-stroke
sessions, the within-animal correlation of the pre-cue change with the pre-lick and post-lick changes (animal-demeaned;
per-animal r printed first, rule 8; within-animal shuffle, 2,000).
Outputs: <state_space_recovery>/shift_precue{tag}/{sessions,changes,coupling}.csv, page_figs/ss_shift_precue{tag}.png.
EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_recovery_trials as R  # noqa: E402
import state_space_shift_behavior as SB  # noqa: E402
import state_space_sim as S  # noqa: E402
import state_space_trajectories as TR  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE  # noqa: E402

EPS = SB.EPS
OUT = S.ROOT / f"shift_precue{S.tag()}"
FIGS = S.ROOT / "page_figs"
WINS = {"pre-lick": (-1.0, -0.1), "post-lick": (0.0, 0.7)}
READS = ["pre-cue", "pre-lick", "post-lick", "pre-lick - pre-cue"]


def main():
    if S.BASIS != "svd":
        sys.exit("state_space_shift_precue: needs the cohort SVD basis (WF_SS_BASIS=svd)")
    from wfield_local import area_basis, beta_maps
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(atlas.shape)
    flat = np.flatnonzero(mask.ravel())
    split = area_basis.basis_masks(area_basis.roi_masks(atlas.ravel()[flat], beta_maps.stat_mask().ravel()[flat],
                                                        dict(S._atlas_names()), flat, mask.shape))["split"]
    W = {}
    for reg, areas in SB.REGIONS.items():
        for side, hemi in SB.SIDES.items():
            px = np.logical_or.reduce([np.asarray(split[f"{a}_{hemi}"], bool) for a in areas if f"{a}_{hemi}" in split])
            W[f"{reg} {side}"] = Uc[px].mean(0)
    t, _ = TR.times()
    wix = {w: np.flatnonzero((t >= a - 1e-9) & (t < b - 1e-9)) for w, (a, b) in WINS.items()}
    data, _ = S.load_sessions()
    rows = []
    for an, sess in sorted(data.items()):
        Tpre = R.template([s for s in sess if s[1] == "pre"])
        ok = np.isfinite(Tpre).all(axis=(1, 2))
        B0 = Tpre[ok] - S.flat(Tpre[ok]).mean(0)
        unit = float(np.sqrt((B0 ** 2).sum() / B0.size * B0.shape[-1] / Uc.shape[0]))
        for lab, ep, day, Y, pos in sess:
            if ep not in ("pre", *EPS):
                continue
            c = C._rest(lab, an) if (C.REST / f"{lab}.npz").exists() else None
            f = C.ENL / f"{lab}.npz"
            if c is None or not f.exists():
                continue
            E = np.load(f)["b"].astype(float)
            if len(E) != len(pos):
                continue
            keep = np.isfinite(E).all(1)
            Z = (Y[keep] - c) / unit
            Ez = (E[keep] - c) / unit
            r = {"animal": an, "label": lab, "epoch": ep, "day": day}
            for reg, w in W.items():
                v = Z @ w
                r[f"{reg} | pre-cue"] = float((Ez @ w).mean())
                for wn, ix in wix.items():
                    r[f"{reg} | {wn}"] = float(v[:, ix].mean())
                r[f"{reg} | pre-lick - pre-cue"] = r[f"{reg} | pre-lick"] - r[f"{reg} | pre-cue"]
            rows.append(r)
        print(f"  {an} done", flush=True)
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "sessions.csv", index=False)
    rng = np.random.default_rng(0)

    def change(col, ep):
        pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in D.groupby("animal")}
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in D.groupby("animal")}
        post = {a: v for a, v in post.items() if v and pre.get(a)}
        iv = kit.boot_delta(post, pre, rng) if post else None
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post)
        return (iv.point, iv.lo, iv.hi, up, len(post)) if iv is not None else (np.nan,) * 3 + (0, 0)

    regs = list(W)
    fig, axs = plt.subplots(2, len(regs), figsize=(4.4 * len(regs), 8), squeeze=False)
    recs, coup = [], []
    for j, reg in enumerate(regs):
        ax = axs[0, j]
        for k, rd in enumerate(READS):
            for e, ep in enumerate(EPS):
                pt, lo, hi, up, n = change(f"{reg} | {rd}", ep)
                x = k + (e - 1) * 0.22
                ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=EPOCH_LINE[ep], ms=5, lw=1.4, capsize=0)
                if lo > 0 or hi < 0:
                    sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=8)
                recs.append({"region": reg, "readout": rd, "epoch": ep, "change": pt, "lo": lo, "hi": hi, "k_up": up,
                             "n": n})
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(READS)), READS, rotation=20, fontsize=8.5)
        ax.set_title(reg, fontsize=10)
        if j == 0:
            ax.set_ylabel("change vs own pre (minus RESTnl;\npre pixel RMS)")
        # coupling across post sessions: pre-cue change vs pre-lick / post-lick change (animal-demeaned)
        ax = axs[1, j]
        post = D[D.epoch != "pre"].copy()
        for rd in ("pre-cue", "pre-lick", "post-lick"):
            post[rd] = post[f"{reg} | {rd}"] - post.groupby("animal")[f"{reg} | {rd}"].transform("mean")
        for rd, mk in (("pre-lick", "o"), ("post-lick", "^")):
            r_an = {an: round(float(np.corrcoef(g["pre-cue"], g[rd])[0, 1]), 2) for an, g in post.groupby("animal")}
            b = float((post["pre-cue"] * post[rd]).sum() / (post["pre-cue"] ** 2).sum())
            null = []
            for _ in range(2000):
                sh = post.groupby("animal")[rd].transform(lambda v: rng.permutation(v.to_numpy()))
                null.append(float((post["pre-cue"] * sh).sum() / (post["pre-cue"] ** 2).sum()))
            p = float((np.abs(null) >= abs(b)).mean())
            coup.append({"region": reg, "vs": rd, "r_by_animal": r_an, "within_slope": b, "shuffle_p": p})
            for an, g in post.groupby("animal"):
                ax.scatter(g["pre-cue"], g[rd], marker=ANIMAL_MARKER[an] if rd == "pre-lick" else mk,
                           facecolors="none" if rd == "pre-lick" else "0.6",
                           edgecolors=[EPOCH_LINE[e] for e in g.epoch], s=16, lw=0.7, alpha=0.8)
            print(f"  {reg}: pre-cue vs {rd}: r {r_an} slope {b:+.2f} p={p:.3f}", flush=True)
        q = [c for c in coup if c["region"] == reg]
        ax.set_title(f"post sessions, animal-demeaned\npre-lick slope {q[0]['within_slope']:+.2f} (p={q[0]['shuffle_p']:.3f}); "
                     f"post-lick {q[1]['within_slope']:+.2f} (p={q[1]['shuffle_p']:.3f})", fontsize=8.5)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("pre-cue level (session, minus animal mean)")
        if j == 0:
            ax.set_ylabel("pre-lick (open) / post-lick (filled)\n(session, minus animal mean)")
    pd.DataFrame(recs).to_csv(OUT / "changes.csv", index=False)
    pd.DataFrame(coup).to_csv(OUT / "coupling.csv", index=False)
    fig.suptitle(C._wrap(
        "Is the pre-lick rise the pre-cue rise? Region means minus the session's RESTnl level (cohort SVD; units of the "
        "animal's pre-template pixel RMS): the trial's lick-free pre-cue ENL window (Fig 9.8's window), -1.0..-0.1 s "
        "before the first lick, 0..+0.7 s after it, and pre-lick minus pre-cue. Top: change vs own pre (nested bootstrap "
        "animals -> sessions, 2,000; * = 95 % CI excludes 0). Bottom: post-stroke sessions, animal-demeaned, pre-cue "
        "level vs pre-lick (open, shape = animal) and post-lick (filled triangles); within-animal slope with a "
        "within-animal shuffle (2,000)."), y=1.04, fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_shift_precue{S.tag()}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(pd.DataFrame(recs).round(2).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
