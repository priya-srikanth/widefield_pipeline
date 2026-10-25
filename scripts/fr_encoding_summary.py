"""FaceRhythm in the per-session encoder (O2 batch enc_v36_full_20261024, runs fr_A / fr_B / fr_S / fr_D): does face
movement add variance, and does it absorb the spout-position code? (handoff: "does FaceRhythm absorb the chronic residual
or the spout coding?")

Per session x area x window, from the region tables (held-out R2, combined_encoding):
  FR gain           fr:full - full                       (what FaceRhythm adds to the full behavioural model)
  position | FR     fr:full - fr:drop:position           (unique spout-position R2 with FR in the model)
  position          full - drop:position (main run)      (unique spout-position R2 without FR; same base model)
  retained          position | FR / position             (1 = FR takes none of the position code)
Modes (wfield_local/fr_groups.py): A = lick amount + side + distance + other; B = lick amount + other (directionless);
S = + side; D = + distance. Per animal x epoch means (rule 8), pooled change from pre with the nested bootstrap.

Usage: python scripts/fr_encoding_summary.py [--areas cortex SSp-m_left ...]
Output: <E>/fr_encoding_summary/{fr_session_table.csv, fr_summary_by_animal.csv, figures/*.png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
MAIN = E / "movement_first_exec-polar_licks-combined" / "sessions"
FR = {m: E / f"movement_first_fr-master_pre_v1_subset-fr_exec-polar_licks-combined_frg-{m}" / "sessions" for m in "ABSD"}
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
MCOL = {"A": "#1b9e77", "B": "#7570b3", "S": "#d95f02", "D": "#e7298a"}


def session_table(areas, windows) -> pd.DataFrame:
    from wfield_local import config, epochs
    rows = []
    for f in sorted(MAIN.glob("*_regions.csv")):
        lab = f.stem.replace("_regions", "")
        if lab[-4:] < "0801":
            continue
        m = pd.read_csv(f)
        m = m[m.area.isin(areas) & m.window.isin(windows)]
        base = m.pivot_table(index=["area", "window"], columns="model", values="r2")
        if "drop:position" not in base:
            continue
        for mode, d in FR.items():
            g = d / f.name
            if not g.exists():
                continue
            r = pd.read_csv(g)
            r = r[r.area.isin(areas) & r.window.isin(windows)].pivot_table(index=["area", "window"], columns="model", values="r2")
            for (ar, w), x in r.iterrows():
                b = base.loc[(ar, w)]
                rows.append({"label": lab, "animal": config.animal_of(lab), "epoch": epochs.epoch_of(lab), "mode": mode,
                             "area": ar, "window": w, "full_main": b["full"], "full_frrun": x["full"],
                             "fr_gain": x["fr:full"] - x["full"],
                             "fr_alone": x["fr:alone"],
                             "pos": b["full"] - b["drop:position"],
                             "pos_fr": x["fr:full"] - x["fr:drop:position"]})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.analysis_kit import boot_delta
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areas", nargs="+", default=["cortex", "SSp-m_left", "SSp-m_right", "MOs_left", "MOs_right"])
    ap.add_argument("--windows", nargs="+", default=["enl", "cue", "lick"])
    a = ap.parse_args(argv)
    out = E / "fr_encoding_summary"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T = session_table(a.areas, a.windows)
    T.to_csv(out / "fr_session_table.csv", index=False)
    dfull = (T.full_main - T.full_frrun).abs().max()
    print(f"{T.label.nunique()} sessions; |full (main run) - full (FR run)| max {dfull:.2e} (same base model)", flush=True)
    A = T.groupby(["mode", "area", "window", "animal", "epoch"])[["fr_gain", "fr_alone", "pos", "pos_fr"]].mean().reset_index()
    A["retained"] = A.pos_fr / A.pos.where(A.pos.abs() > 0.5)
    A.to_csv(out / "fr_summary_by_animal.csv", index=False)
    pd.set_option("display.width", 220)
    show = A[(A.area == "cortex")].pivot_table(index=["window", "mode", "animal"], columns="epoch",
                                               values=["fr_gain", "pos", "pos_fr"]).round(3)
    print(show.to_string())
    rng = lambda: np.random.default_rng(0)  # noqa: E731
    lines = []
    for area in a.areas:
        fig, axs = plt.subplots(len(a.windows), 3, figsize=(15, 3.6 * len(a.windows)), squeeze=False)
        for r, w in enumerate(a.windows):
            t = T[(T.area == area) & (T.window == w)]
            # (1) FR gain per epoch and mode (level, animal means + pooled mean)
            ax = axs[r, 0]
            for k, m in enumerate("ABSD"):
                for i, ep in enumerate(EP):
                    v = t[(t["mode"] == m) & (t.epoch == ep)].groupby("animal").fr_gain.mean()
                    if not len(v):
                        continue
                    x = i + (k - 1.5) * 0.2
                    ax.bar(x, v.mean(), width=0.18, color=MCOL[m], alpha=0.85, label=f"mode {m}" if i == 0 else None)
                    for an, y in v.items():
                        ax.scatter(x, y, marker=MK[an], s=12, facecolor="w", edgecolor="k", lw=0.6, zorder=5)
            ax.set_xticks(range(4)); ax.set_xticklabels(EP); ax.axhline(0, color="k", lw=0.6)
            ax.set_ylabel(f"{w} window\nR2 (fraction)"); ax.set_title("what FaceRhythm adds (fr:full - full)", fontsize=9)
            if r == 0:
                ax.legend(fontsize=7, frameon=False, ncol=4)
            # (2) unique position R2 without FR vs with FR (mode B and A)
            ax = axs[r, 1]
            for k, (col, lab, cc) in enumerate((("pos", "no FR", "0.3"), ("pos_fr_B", "with FR mode B", MCOL["B"]),
                                                 ("pos_fr_A", "with FR mode A", MCOL["A"]))):
                for i, ep in enumerate(EP):
                    if col == "pos":
                        v = t[(t["mode"] == "B") & (t.epoch == ep)].groupby("animal").pos.mean()
                    else:
                        v = t[(t["mode"] == col[-1]) & (t.epoch == ep)].groupby("animal").pos_fr.mean()
                    if not len(v):
                        continue
                    x = i + (k - 1) * 0.26
                    ax.bar(x, v.mean(), width=0.24, color=cc, alpha=0.85, label=lab if i == 0 else None)
                    for an, y in v.items():
                        ax.scatter(x, y, marker=MK[an], s=12, facecolor="w", edgecolor="k", lw=0.6, zorder=5)
            ax.set_xticks(range(4)); ax.set_xticklabels(EP); ax.axhline(0, color="k", lw=0.6)
            ax.set_title("unique spout-position R2 (fraction)", fontsize=9)
            if r == 0:
                ax.legend(fontsize=7, frameon=False)
            # (3) pooled change from pre (nested bootstrap): position unique with FR (A), without FR, and FR gain
            ax = axs[r, 2]
            for k, (col, mode, lab, cc) in enumerate((("pos", "B", "position, no FR", "0.3"),
                                                       ("pos_fr", "A", "position | FR (A)", MCOL["A"]),
                                                       ("fr_gain", "A", "FR gain (A)", "#66a61e"))):
                for i, ep in enumerate(EP[1:]):
                    s = t[t["mode"] == mode]
                    post = {an: s[(s.animal == an) & (s.epoch == ep)][col].to_numpy() for an in AN
                            if len(s[(s.animal == an) & (s.epoch == ep)])}
                    pre = {an: s[(s.animal == an) & (s.epoch == "pre")][col].to_numpy() for an in AN
                           if len(s[(s.animal == an) & (s.epoch == "pre")])}
                    iv = boot_delta(post, pre, rng())
                    if iv is None:
                        continue
                    x = i + (k - 1) * 0.26
                    ax.bar(x, iv.point, width=0.24, color=cc, alpha=0.85, label=lab if i == 0 else None)
                    ax.errorbar(x, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], color="k", capsize=2, lw=1)
                    if iv.lo > 0 or iv.hi < 0:
                        ax.text(x, iv.hi, "*", ha="center", va="bottom")
                    lines.append(f"{area} {w} {lab} {ep}-pre: {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}]")
            ax.set_xticks(range(3)); ax.set_xticklabels(EP[1:]); ax.axhline(0, color="k", lw=0.6)
            ax.set_title("change from pre (pooled, 95% nested CI; * excludes 0)", fontsize=9)
            if r == 0:
                ax.legend(fontsize=7, frameon=False)
        fig.suptitle(f"NEURAL: FaceRhythm (master_pre_v1, rank 12) in the per-session encoder, {area}. Held-out R2 (fraction); bars = mean of "
                     "animals, markers = animals (o PS92, s PS93, ^ PS94, D PS95).\nMode A = FR lick amount + side + distance + other; B = "
                     "directionless (lick amount + other); S = + side; D = + distance.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / "figures" / f"fr_summary_{area}.png", dpi=115)
        plt.close(fig)
    (out / "pooled_changes.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
