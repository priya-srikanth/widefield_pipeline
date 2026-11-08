"""2-D state-space trajectories (Priya 11-07: "is it possible to represent this as 2d trajectories in state space?").

Alignment from WF_SS_ALIGN (state_space_sim: cue / cue25 / lick / splice). First-lick trials (contact + incomplete).

A. ACROSS DAYS (how it gets there), per animal, within spout, in the fixed per-animal pre space (top 8):
   x = along the pre template (0) -> chronic template (1) line (the trial-cloud axis);
   y = the ACUTE departure orthogonal to it: (acute template - pre template) with its x part removed, scaled by the
       same pre -> chronic distance.
   Each spout has its own frame; a session's point = mean over its spouts (>= 5 trials) of the spout means; trial clouds
   = 1-SD ellipses of the trial coordinates per epoch. Templates use all sessions of the epoch (x is circular for
   chronic, y for acute -- they define the axes).
B. WITHIN THE TRIAL (what kind of change): every animal's pre spout x time templates, centered on that animal's own pre
   mean, stacked -> one cohort PCA over the 62 Allen areas; the top 2 PCs. Each animal x epoch template (mean of its
   sessions' spout means, centered on the animal's PRE mean, so a shared shift stays visible) is projected; pooled = mean
   over animals. Lines = time within the trial (time-continuous); dot = window start, ring = the event (cue / lick).

Outputs: Widefield/state_space_recovery/trajectories{tag}/ (csv) and page_figs/ss_traj_{days,loops,loops_animals}_<align>.png.
Exploratory.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import epochs  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE  # noqa: E402

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
OUT = ROOT / f"trajectories{S.tag()}"
FIGS = ROOT / "page_figs"
EPOCHS = R.EPOCHS
LABEL = {"cue": "cue-aligned -0.5..2.0 s", "cue25": "cue-aligned -0.5..2.5 s", "lick": "first-lick-aligned -1.0..0.7 s",
         "splice": "cue -0.5..0.5 s + first lick -0.5..0.7 s"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def load_all():
    data = {}
    for f in sorted(glob.glob(S.ITI + "/PS9*.npz")):
        Y, pos, cols, ep, an = S.load(f)
        lab = Path(f).stem
        data.setdefault(an, []).append((lab, ep, epochs.days_since_stroke(lab), Y, pos))
    return {a: sorted(v, key=lambda s: s[2]) for a, v in data.items()}, cols


def times():
    """Bin centers and the segment break (splice) for the current alignment."""
    t, brk = [], []
    for k, (_, _, bins) in enumerate(S.SEGMENTS[S.ALIGN]):
        if k:
            brk.append(len(t))
        t += [a + 0.05 for a, _ in bins]
    return np.array(t), brk


# ---------------------------------------------------------------- A. across days

def days_coords(sess):
    pre = [s for s in sess if s[1] == "pre"]
    P, mu = R.fixed_space(R.template(pre))
    tmpl = {ep: R.template([s for s in sess if s[1] == ep]) for ep in ("pre", "acute", "chronic")}
    z = {ep: ((T - mu) @ P) for ep, T in tmpl.items()}
    frames = {}
    for si, p in enumerate(S.POS):
        zp, zc, za = (z[e][si].ravel() for e in ("pre", "chronic", "acute"))
        if not (np.isfinite(zp).all() and np.isfinite(zc).all() and np.isfinite(za).all()):
            continue
        d = zc - zp
        L = np.linalg.norm(d)
        u = d / L
        a = za - zp
        ap = a - (a @ u) * u
        frames[p] = (zp, u, ap / np.linalg.norm(ap), L, float(np.linalg.norm(ap) / np.linalg.norm(a)))
    srows, trows = [], []
    for lab, ep, day, Y, pos in sess:
        zt = (Y - mu) @ P
        xs, ys = [], []
        for p, (zp, u, v, L, _) in frames.items():
            m = pos == p
            if m.sum() < R.MIN_TRIALS:
                continue
            q = zt[m].reshape(m.sum(), -1) - zp
            x, y = q @ u / L, q @ v / L
            xs.append(x.mean())
            ys.append(y.mean())
            trows += [{"label": lab, "epoch": ep, "day": day, "spout": p, "x": a, "y": b} for a, b in zip(x, y)]
        if xs:
            srows.append({"label": lab, "epoch": ep, "day": day, "x": float(np.mean(xs)), "y": float(np.mean(ys))})
    orth = float(np.mean([f[4] for f in frames.values()]))
    return pd.DataFrame(srows), pd.DataFrame(trows), orth


def ellipse(ax, x, y, color, lw=2.0, ls="-"):
    if len(x) < 5:
        return
    C = np.cov(x, y)
    w, V = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fill=False,
                         edgecolor=color, lw=lw, ls=ls, zorder=3))


def fig_days(D):
    an_list = sorted(D)
    fig, axs = plt.subplots(1, 5, figsize=(19, 4.3), sharex=True, sharey=True)
    for ax, key in zip(axs, ["all animals"] + an_list):
        sel = an_list if key == "all animals" else [key]
        for ep in EPOCHS:
            tr = pd.concat([D[a][1][D[a][1].epoch == ep] for a in sel])
            ellipse(ax, tr.x, tr.y, EPOCH_LINE[ep])
        for a in sel:
            s = D[a][0]
            for ep, g in s.groupby("epoch"):
                g = g.sort_values("day")
                if key != "all animals":
                    ax.plot(g.x, g.y, color=EPOCH_LINE[ep], lw=0.7, alpha=0.6, zorder=2)   # within epoch only
                ax.scatter(g.x, g.y, marker=ANIMAL_MARKER[a], facecolors="none", edgecolors=EPOCH_LINE[ep], s=26,
                           lw=1, zorder=4)
        for xx in (0, 1):
            ax.axvline(xx, color="0.75", lw=0.8, zorder=0)
        ax.axhline(0, color="0.75", lw=0.8, zorder=0)
        ttl = key if key == "all animals" else f"{key}  (acute departure {100 * D[key][2]:.0f} % orthogonal)"
        ax.set_title(ttl)
        ax.set_xlabel("along pre (0) -> chronic (1)")
    axs[0].set_ylabel("acute departure, orthogonal\n(same units: pre -> chronic distance)")
    hs = [plt.Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", mec="k", label=a) for a in an_list]
    hs += [plt.Line2D([], [], color=EPOCH_LINE[e], lw=2, label=f"{e} (1-SD trial cloud)") for e in EPOCHS]
    fig.legend(handles=hs, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(f"Sessions and trial clouds in the pre -> chronic / acute-departure plane, within spout ({LABEL[S.ALIGN]})",
                 y=1.13)
    fig.tight_layout()
    sig_marks.reserve_headroom(fig)
    fig.savefig(FIGS / f"ss_traj_days_{S.ALIGN}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- B. within the trial

def loops(data):
    centered, mus = {}, {}
    for an, sess in data.items():
        pre = [s for s in sess if s[1] == "pre"]
        T = R.template(pre)
        mu = S.flat(T[np.isfinite(T).all(axis=(1, 2))]).mean(0)
        mus[an] = mu
        centered[an] = {ep: R.template([s for s in sess if s[1] == ep]) - mu for ep in EPOCHS
                        if any(s[1] == ep for s in sess)}
    F = np.concatenate([S.flat(c["pre"][np.isfinite(c["pre"]).all(axis=(1, 2))]) for c in centered.values()])
    _, sv, Vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
    W = Vt[:2].T
    ve = (sv[:2] ** 2).sum() / (sv ** 2).sum()
    proj = {an: {ep: c @ W for ep, c in d.items()} for an, d in centered.items()}       # 6 x bins x 2
    pooled = {ep: np.nanmean(np.stack([proj[a][ep] for a in proj if ep in proj[a]]), axis=0) for ep in EPOCHS}
    return proj, pooled, ve


def draw_loops(ax, Z, t, brk, color, lw):
    segs = np.split(np.arange(len(t)), brk) if brk else [np.arange(len(t))]
    for seg in segs:
        ax.plot(Z[seg, 0], Z[seg, 1], color=color, lw=lw)
    ax.plot(Z[0, 0], Z[0, 1], "o", color=color, ms=4)
    ev = [i for i in range(len(t)) if abs(t[i] - 0.05) < 1e-6]                         # bin starting at the event
    for i in ev:
        ax.plot(Z[i, 0], Z[i, 1], "o", mfc="none", mec=color, ms=8, mew=1.5)


def fig_loops(proj, pooled, ve):
    t, brk = times()
    fig, axs = plt.subplots(1, 6, figsize=(19, 3.9), sharex=True, sharey=True)
    for si, (ax, p) in enumerate(zip(axs, S.POS)):
        for ep in EPOCHS:
            if np.isfinite(pooled[ep][si]).all():
                draw_loops(ax, pooled[ep][si], t, brk, EPOCH_LINE[ep], 2.0)
        ax.set_title(p.replace("_", " "))
        ax.set_xlabel("PC1")
    axs[0].set_ylabel("PC2")
    hs = [plt.Line2D([], [], color=EPOCH_LINE[e], lw=2, label=e) for e in EPOCHS]
    hs += [plt.Line2D([], [], marker="o", ls="", color="k", ms=4, label="window start"),
           plt.Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=8, label="event (0 s)")]
    fig.legend(handles=hs, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Mean trial trajectories, cohort pre-stroke top-2 plane ({100 * ve:.0f} % of pre variance), mean of 4 "
                 f"animals ({LABEL[S.ALIGN]})", y=1.15)
    fig.tight_layout()
    sig_marks.reserve_headroom(fig)
    fig.savefig(FIGS / f"ss_traj_loops_{S.ALIGN}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(4, 6, figsize=(19, 12), sharex=True, sharey=True)
    for r, an in enumerate(sorted(proj)):
        for si, p in enumerate(S.POS):
            ax = axs[r, si]
            for ep in EPOCHS:
                Z = proj[an].get(ep)
                if Z is not None and np.isfinite(Z[si]).all():
                    draw_loops(ax, Z[si], t, brk, EPOCH_LINE[ep], 1.5)
            if r == 0:
                ax.set_title(p.replace("_", " "))
            if si == 0:
                ax.set_ylabel(f"{an}\nPC2")
    fig.suptitle(f"Per animal, same plane ({LABEL[S.ALIGN]})", y=1.0)
    fig.tight_layout()
    sig_marks.reserve_headroom(fig)
    fig.savefig(FIGS / f"ss_traj_loops_animals_{S.ALIGN}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    data, _ = load_all()
    D = {an: days_coords(sess) for an, sess in sorted(data.items())}
    pd.concat([d[0].assign(animal=a) for a, d in D.items()]).to_csv(OUT / "days_sessions.csv", index=False)
    pd.concat([d[1].assign(animal=a) for a, d in D.items()]).to_csv(OUT / "days_trials.csv", index=False)
    fig_days(D)
    proj, pooled, ve = loops(data)
    fig_loops(proj, pooled, ve)
    with pd.option_context("display.width", 200, "display.precision", 2):
        s = pd.concat([d[0].assign(animal=a) for a, d in D.items()])
        print(s.groupby(["animal", "epoch"])[["x", "y"]].mean().unstack().reindex(columns=EPOCHS, level=1))
        print("acute departure orthogonal share:", {a: round(d[2], 2) for a, d in D.items()}, "top-2 var", round(ve, 2))


if __name__ == "__main__":
    main()
