"""Per spout position: does the "changed representation" track recovery the same way at every spout? (Priya 11-07: "the
acute departure axis and the pre vs chronic axis seem like good supporting evidence of a changed post-stroke motor
representation. is it worth looking at this per-position (might the relationship be different for different spout
positions)").

Per session x spout (first-lick-aligned unless WF_SS_ALIGN says otherwise; from state_space_pertrial.py's per-trial
table): mean x (pre (0) -> chronic (1) line of that spout) and y (acute-departure axis of that spout), against
  (a) RECOVERY: far-contra first-cued-lick aim vs own pre (deg, aim_deficit_sessions.csv, as state_space_step_vs_gradual);
  (b) the spout's OWN executed landing: mean 3-D lateral landing (side_ml) of its first licks minus the animal's pre mean
      at that spout (mm; landing join).
Post-stroke sessions; per-animal r (rule 8), within-animal slope with a within-animal shuffle (aim / landing permuted
within animal, 2000 draws). Output: Widefield/state_space_recovery/per_position{tag}/, page_figs/ss_perpos_<suffix>.png.
Exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import state_space_page_figs as F  # noqa: E402
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402
import state_space_step_vs_gradual as SG  # noqa: E402

from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402  (PNG + svg/ twin, Priya 11-06)

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
OUT = ROOT / f"per_position{S.tag()}"
SUF = S.tag().lstrip("_") or "cue"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False})


def load():
    T = pd.read_csv(ROOT / f"pertrial{S.tag()}" / "pertrial.csv.gz")
    L = pd.read_csv(R.LANDING)[["label", "row", "side_ml"]].rename(columns={"row": "trial"})
    T = T.merge(L, on=["label", "trial"], how="left")
    s = T.groupby(["animal", "label", "epoch", "day", "spout"])[["x", "y", "side_ml"]].mean().reset_index()
    pre = s[s.epoch == "pre"].groupby(["animal", "spout"]).side_ml.mean()
    s["land"] = s.side_ml - s.set_index(["animal", "spout"]).index.map(pre).to_numpy()
    aim = pd.read_csv(SG.AIM)[["label", "farcontra_first"]].rename(columns={"farcontra_first": "aim"})
    return s.merge(aim, on="label", how="left")


def wslope(g, c, v, rng, n=2000):
    g = g.dropna(subset=[c, v])
    if g.animal.nunique() < 2 or len(g) < 10:
        return np.nan, np.nan
    def b(h):
        x = h[v] - h.groupby("animal")[v].transform("mean")
        y = h[c] - h.groupby("animal")[c].transform("mean")
        return float((x * y).sum() / (x ** 2).sum())
    b0 = b(g)
    null = []
    for _ in range(n):
        h = g.copy()
        h[v] = h.groupby("animal")[v].transform(lambda s: rng.permutation(s.to_numpy()))
        null.append(b(h))
    return b0, float((np.abs(np.asarray(null)) >= abs(b0)).mean())


def main():
    rng = np.random.default_rng(0)
    s = load()
    post = s[s.epoch != "pre"]
    rows = []
    for p in S.POS:
        g = post[post.spout == p]
        for c in ("x", "y"):
            for v in ("aim", "land"):
                r_an = {an: float(h[[c, v]].dropna().corr().iloc[0, 1]) for an, h in g.groupby("animal")
                        if h[[c, v]].dropna().shape[0] > 3}
                b, pv = wslope(g, c, v, rng)
                rows.append({"spout": p, "readout": c, "vs": v, "slope": b, "shuffle_p": pv,
                             **{f"r_{an}": round(r, 2) for an, r in r_an.items()}})
    A = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "per_position_sessions.csv", index=False)
    A.to_csv(OUT / "per_position_fits.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3, "display.max_rows", 100):
        print(A.to_string(index=False))
    figure(s, A)


def figure(s, A):
    post = s[s.epoch != "pre"]
    prs = s[s.epoch == "pre"]
    specs = [("x", "aim", "position pre (0) -> chronic (1)", "far-contra aim vs own pre (deg)"),
             ("y", "aim", "acute-departure axis", "far-contra aim vs own pre (deg)"),
             ("x", "land", "position pre (0) -> chronic (1)", "own-spout landing vs own pre (mm, + = contralesional)"),
             ("y", "land", "acute-departure axis", "own-spout landing vs own pre (mm, + = contralesional)")]
    fig, axs = plt.subplots(4, 6, figsize=(19, 13), sharey="row")
    for i, (c, v, ct, vt) in enumerate(specs):
        for j, p in enumerate(S.POS):
            ax = axs[i, j]
            g = post[post.spout == p]
            q = prs[prs.spout == p]
            ax.scatter(q[v], q[c], color="0.78", s=9, zorder=0)
            for an, h in g.groupby("animal"):
                ax.scatter(h[v], h[c], marker=ANIMAL_MARKER[an], facecolors="none",
                           edgecolors=[EPOCH_LINE[e] for e in h.epoch], s=16, lw=0.8)
                hh = h.dropna(subset=[c, v])
                if len(hh) > 3:
                    b = np.polyfit(hh[v], hh[c], 1)
                    xx = np.array([hh[v].min(), hh[v].max()])
                    ax.plot(xx, np.polyval(b, xx), color="0.35", lw=0.8)
            a = A[(A.spout == p) & (A.readout == c) & (A.vs == v)].iloc[0]
            rs = " / ".join(f"{a.get(f'r_{an}', np.nan):.2f}" for an in sorted(ANIMAL_MARKER))
            star = "*" if a.shuffle_p < 0.05 else ""
            ax.set_title(f"{p.replace('_', ' ') if i == 0 else ''}\nr {rs}{star}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{ct}\n(vs {v})")
            if i in (1, 3):
                ax.set_xlabel(vt, fontsize=8)
    hs = [plt.Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", mec="k", label=a) for a in sorted(ANIMAL_MARKER)]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EPOCH_LINE[e], label=e) for e in ("acute", "subacute",
                                                                                                  "chronic")]
    hs += [plt.Line2D([], [], marker="o", ls="", color="0.78", label="pre sessions"),
           plt.Line2D([], [], color="0.35", lw=0.8, label="line per animal")]
    fig.legend(handles=hs, loc="upper center", ncol=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Per spout position: the changed representation against recovery (rows 1-2) and against that spout's own "
                 f"landing (rows 3-4) ({F.ALIGN_LABEL[S.ALIGN]}); r = within-animal PS92 / 93 / 94 / 95, * = within-animal "
                 f"shuffle p < .05", y=1.045)
    fig.tight_layout()
    save_png_svg(fig, ROOT / "page_figs" / f"ss_perpos_{SUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
