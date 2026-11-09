"""Is the TONIC orofacial elevation (pre-cue minus RESTnl, Fig 10.24) partly REST MOVING? (Priya 11-09: "should we check
with raw (instead of RESTnl) to see if this is partly rest moving?")

The per-trial signal has no absolute level: the slow drift is projected out per session INCLUDING the constant
(`state_space_rest_baseline.py`), so "raw" = relative to the session's whole-recording mean. Any level read "minus
RESTnl" is the difference of two raw levels, so the tonic change splits exactly:
    (pre-cue - RESTnl) change  =  raw pre-cue change  -  raw RESTnl change
Per session, region means (orofacial = tjM1 + SSp-m + SSp-n; posterior-medial = RSPd + SSp-tr + SSp-ll; contra / ipsi;
split-basis areas, ROI v8), cohort SVD basis, units of the animal's pre-template pixel RMS (as Figs 10.21-10.25), RAW:
  RESTnl      the session's RESTnl level (rest frames minus licks, engaged period; `rest_baseline_svd`)
  pre-cue     the trial's own lick-free (>= 1 s) pre-cue ENL window, mean over trials
  pre-lick    -1.0..-0.1 s before the first lick;  post-lick  0..+0.7 s after it (first-lick-aligned extract)
and the same three task levels minus RESTnl. Change vs own pre, animal-weighted nested bootstrap (`boot_delta`, 2,000),
* = 95 % CI excludes 0; per-animal values printed first (rule 8). Plus, across post-stroke sessions (animal-demeaned), how
much of the vs-RESTnl tonic level is carried by the raw pre-cue level vs by the raw rest level (within-animal slopes,
within-animal shuffle, 2,000).
Caveat: the session mean itself contains the task (licking time, engagement), so a raw level is relative to a reference
that can move too; neither reference is absolute. The test is whether the two references AGREE.
Outputs: <state_space_recovery>/shift_rawref{tag}/{sessions,changes,coupling}.csv, page_figs/ss_shift_rawref{tag}.png.
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
OUT = S.ROOT / f"shift_rawref{S.tag()}"
FIGS = S.ROOT / "page_figs"
WINS = {"pre-lick": (-1.0, -0.1), "post-lick": (0.0, 0.7)}
RAW = ["RESTnl", "pre-cue", "pre-lick", "post-lick"]
REL = ["pre-cue", "pre-lick", "post-lick"]


def main():
    if S.BASIS != "svd":
        sys.exit("state_space_shift_rawref: needs the cohort SVD basis (WF_SS_BASIS=svd)")
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
            if ep not in ("pre", *EPS) or not (C.REST / f"{lab}.npz").exists():
                continue
            c = C._rest(lab, an)
            f = C.ENL / f"{lab}.npz"
            if c is None or not f.exists():
                continue
            E = np.load(f)["b"].astype(float)
            if len(E) != len(pos):
                continue
            keep = np.isfinite(E).all(1)
            r = {"animal": an, "label": lab, "epoch": ep, "day": day}
            for reg, w in W.items():
                rest = float(c @ w) / unit
                v = (Y[keep] @ w) / unit
                lv = {"RESTnl": rest, "pre-cue": float((E[keep] @ w).mean()) / unit,
                      **{wn: float(v[:, ix].mean()) for wn, ix in wix.items()}}
                for k in RAW:
                    r[f"{reg} | raw {k}"] = lv[k]
                for k in REL:
                    r[f"{reg} | {k} - RESTnl"] = lv[k] - rest
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
        per = {a: round(float(np.mean(post[a]) - np.mean(pre[a])), 2) for a in post}
        iv = kit.boot_delta(post, pre, rng)
        return iv.point, iv.lo, iv.hi, sum(v > 0 for v in per.values()), len(per), per

    reads = [f"raw {k}" for k in RAW] + [f"{k} - RESTnl" for k in REL]
    regs = list(W)
    fig, axs = plt.subplots(2, len(regs), figsize=(4.8 * len(regs), 8.4), squeeze=False)
    recs, coup = [], []
    for j, reg in enumerate(regs):
        ax = axs[0, j]
        for k, rd in enumerate(reads):
            for e, ep in enumerate(EPS):
                pt, lo, hi, up, n, per = change(f"{reg} | {rd}", ep)
                print(f"  {reg} | {rd} | {ep}: {per} -> {pt:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
                x = k + (e - 1) * 0.22
                ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="o" if rd.startswith("raw") else "D",
                            color=EPOCH_LINE[ep], ms=5, lw=1.4, capsize=0,
                            mfc=EPOCH_LINE[ep] if rd.startswith("raw") else "white")
                if lo > 0 or hi < 0:
                    sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=8)
                recs.append({"region": reg, "readout": rd, "epoch": ep, "change": pt, "lo": lo, "hi": hi, "k_up": up,
                             "n": n, "per_animal": per})
        ax.axvline(len(RAW) - 0.5, color="0.75", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(reads)), [rd.replace(" - RESTnl", "\n- RESTnl").replace("raw ", "raw\n")
                                          for rd in reads], fontsize=8)
        ax.set_title(f"{reg}\nfilled = raw (vs session mean); open = minus RESTnl", fontsize=9.5)
        if j == 0:
            ax.set_ylabel("change vs own pre\n(pre pixel RMS)")
        # post sessions, animal-demeaned: is the vs-RESTnl tonic level carried by raw pre-cue or by raw rest?
        ax = axs[1, j]
        post = D[D.epoch != "pre"].copy()
        y = f"{reg} | pre-cue - RESTnl"
        for col in (y, f"{reg} | raw pre-cue", f"{reg} | raw RESTnl"):
            post[col + " dm"] = post[col] - post.groupby("animal")[col].transform("mean")
        for src, mk, sgn in (("raw pre-cue", "o", 1), ("raw RESTnl", "s", -1)):
            xcol = f"{reg} | {src} dm"
            x_ = sgn * post[xcol]
            b = float((x_ * post[y + " dm"]).sum() / (post[y + " dm"] ** 2).sum())   # share of tonic variance: x on y
            null = []
            for _ in range(2000):
                sh = post.groupby("animal")[xcol].transform(lambda v: rng.permutation(v.to_numpy()))
                null.append(float((sgn * sh * post[y + " dm"]).sum() / (post[y + " dm"] ** 2).sum()))
            p = float((np.abs(null) >= abs(b)).mean())
            r_an = {an: round(float(np.corrcoef(g[y + " dm"], sgn * g[xcol])[0, 1]), 2) for an, g in post.groupby("animal")}
            coup.append({"region": reg, "source": ("+" if sgn > 0 else "-") + src, "share_of_tonic": b, "shuffle_p": p,
                         "r_by_animal": r_an})
            print(f"  {reg}: tonic carried by {'+' if sgn > 0 else '-'}{src}: share {b:+.2f} p={p:.3f} r {r_an}",
                  flush=True)
            for an, g in post.groupby("animal"):
                ax.scatter(g[y + " dm"], sgn * g[xcol], marker=ANIMAL_MARKER[an] if src == "raw pre-cue" else mk,
                           facecolors="none" if src == "raw pre-cue" else "0.6",
                           edgecolors=[EPOCH_LINE[e] for e in g.epoch], s=16, lw=0.7, alpha=0.8)
        q = [c for c in coup if c["region"] == reg]
        ax.set_title(f"post sessions, animal-demeaned\nshare of tonic carried by raw pre-cue {q[0]['share_of_tonic']:+.2f} "
                     f"(p={q[0]['shuffle_p']:.3f}), by -raw rest {q[1]['share_of_tonic']:+.2f} "
                     f"(p={q[1]['shuffle_p']:.3f})", fontsize=8)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("pre-cue - RESTnl (session, minus animal mean)")
        if j == 0:
            ax.set_ylabel("raw pre-cue (open, shape = animal) /\n- raw RESTnl (filled squares)")
    pd.DataFrame(recs).to_csv(OUT / "changes.csv", index=False)
    pd.DataFrame(coup).to_csv(OUT / "coupling.csv", index=False)
    fig.suptitle(C._wrap(
        "Is the tonic orofacial elevation rest moving? Region means (cohort SVD; units of the animal's pre-template pixel "
        "RMS) RAW, i.e. relative to the session's whole-recording mean (the drift removal takes out the constant): the "
        "RESTnl level, the trial's lick-free pre-cue ENL window, -1.0..-0.1 s before and 0..+0.7 s after the first lick "
        "(filled), and the three task levels minus RESTnl (open; = raw level - raw RESTnl). Top: change vs own pre "
        "(nested bootstrap animals -> sessions, 2,000; * = 95 % CI excludes 0). Bottom: post-stroke sessions, "
        "animal-demeaned; the share of the vs-RESTnl pre-cue level carried by the raw pre-cue level and by minus the raw "
        "rest level (the two shares sum to 1), within-animal shuffle (2,000)."), y=1.04, fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_shift_rawref{S.tag()}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
