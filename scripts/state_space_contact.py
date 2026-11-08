"""Contact vs incomplete first licks, within session (Priya 11-07: "contact + incomplete as primary, contact-only as
robustness, incomplete-only as descriptive, and direct within-session contrast"). This script = the CONTRAST; the
contact-only / incomplete-only runs are `state_space_recovery_trials.py --condition contact | incomplete`.

Per animal, the fixed pre space (top 8 of the pre template, ALL first-lick trials) and the per-spout frames of
state_space_trajectories (x = pre (0) -> chronic (1) line, y = acute departure orthogonal to it, in pre -> chronic
units). Per session x spout with >= MIN_TRIALS contact AND >= MIN_TRIALS incomplete first licks (3-D first lick touched /
did not touch the spout; landing csv `contact`), for each class:
  x, y         mean trial position in the frame
  size         full-space response norm over 62 areas (state_space_sim.BAL_BINS response minus baseline), log2
  r_pre        correlation of the class mean (bins x areas, centered per area) with the spout's pre template
Contrast = incomplete - contact, averaged over the session's spouts. Pre sessions give the pre-stroke contrast; post
contrasts are shown as levels and as change vs own pre (boot_delta). Alignment from WF_SS_ALIGN.
Output: Widefield/state_space_recovery/contact{tag}/contact_{classes,contrast}.csv, page_figs/ss_contact_<align>.png.
Exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402  (PNG + svg/ twin, Priya 11-06)

ROOT = S.ROOT
OUT = ROOT / f"contact{S.tag()}"
READ = [("x", "position pre (0) -> chronic (1)\n(fraction of pre -> chronic distance)"), ("y", "acute-departure axis\n(same units)"), ("size", "response size\n(log2 ratio)"),
        ("r_pre", "pattern r with the pre template\n(correlation)")]


def main():
    base, resp = S.BAL_BINS[S.ALIGN]
    Lc = pd.read_csv(R.LANDING)
    Lc = {lab: g for lab, g in Lc[Lc.contact.notna()].groupby("label")}

    def with_contact(lab, Y, pos):
        c = np.full(len(pos), np.nan)
        g = Lc.get(lab)
        if g is not None:
            c[g.row.to_numpy()] = g.contact.to_numpy(float)
        return Y, pos, c
    data, _ = S.load_sessions(with_contact)
    rows = []
    for an, sess in sorted(data.items()):
        P, mu, fr = R.frames([s[:5] for s in sess])
        for lab, ep, day, Y, pos, c in sess:
            for p, f in fr.items():
                m1, m0 = (pos == p) & (c == 1), (pos == p) & (c == 0)
                if m1.sum() < R.MIN_TRIALS or m0.sum() < R.MIN_TRIALS:
                    continue
                Tc = f["Tpre"] - f["Tpre"].mean(0)
                for cls, m in (("contact", m1), ("incomplete", m0)):
                    M = Y[m].mean(0)
                    x, y = R.xy(((M - mu) @ P).ravel() - f["zp"], f)
                    Mc = M - M.mean(0)
                    rows.append({"animal": an, "label": lab, "epoch": ep, "day": day, "spout": p, "cls": cls,
                                 "n": int(m.sum()), "x": float(x), "y": float(y),
                                 "size": float(np.log2(np.linalg.norm(M[resp].mean(0) - M[base].mean(0)))),
                                 "r_pre": float((Mc * Tc).sum() / np.linalg.norm(Mc) / np.linalg.norm(Tc))})
    C = pd.DataFrame(rows)
    W = C.pivot_table(index=["animal", "label", "epoch", "day", "spout"], columns="cls", values=[k for k, _ in READ])
    D = pd.DataFrame({k: W[(k, "incomplete")] - W[(k, "contact")] for k, _ in READ}).reset_index()
    Dsess = D.groupby(["animal", "label", "epoch", "day"])[[k for k, _ in READ]].mean().reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / "contact_classes.csv", index=False)
    D.to_csv(OUT / "contact_contrast_spouts.csv", index=False)
    Dsess.to_csv(OUT / "contact_contrast.csv", index=False)
    rng = np.random.default_rng(0)
    with pd.option_context("display.width", 220, "display.precision", 2):
        print("sessions x spouts with both classes:", D.groupby("epoch").size().to_dict())
        print(D.groupby(["epoch", "spout"]).size().unstack().reindex(R.EPOCHS))
        print(Dsess.groupby(["animal", "epoch"])[[k for k, _ in READ]].mean().unstack().reindex(columns=R.EPOCHS, level=1))
        for k, _ in READ:
            lv = []
            for ep in R.EPOCHS:
                iv = kit.boot_ci({a: g[k].tolist() for a, g in Dsess[Dsess.epoch == ep].groupby("animal")}, rng)
                lv.append(f"{ep[:3]} {iv.point:+.2f} [{iv.lo:+.2f},{iv.hi:+.2f}]" if iv else f"{ep[:3]} -")
            print(f"  incomplete - contact, {k:6s}:", " | ".join(lv))
    figure(Dsess)


def figure(Dsess):
    import matplotlib.pyplot as plt

    import state_space_page_figs as F
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (k, t) in zip(axs, READ):
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        for i, ep in enumerate(R.EPOCHS):
            g = Dsess[Dsess.epoch == ep].assign(d=lambda x, k=k: x[k])
            F.scatter_sessions(ax, g, i)
            iv = kit.boot_ci({a: h[k].tolist() for a, h in g.groupby("animal")}, rng)
            if iv is None:
                continue
            ax.errorbar(i + 0.3, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], fmt="o", color=F.EPOCH_LINE[ep],
                        ms=6, lw=2)
            if iv.lo > 0 or iv.hi < 0:
                F.sig_marks.star_top(ax, i + 0.3, "*", color=F.EPOCH_LINE[ep])
        ax.set_xticks(range(4), R.EPOCHS)
        ax.set_xlim(-0.5, 3.6)
        ax.set_title(t)
    axs[0].set_ylabel("incomplete - contact (same session, same spout)")
    F.legend_animals_epochs(fig, y=1.08)
    fig.suptitle(f"Incomplete vs contact first licks within session ({F.ALIGN_LABEL[S.ALIGN]})", y=1.15)
    fig.tight_layout()
    F.sig_marks.reserve_headroom(fig)
    save_png_svg(fig, ROOT / "page_figs" / f"ss_contact_{S.ALIGN}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
