"""Goal page Figs 9 / 9b / 10 (`fig9_softmax_landing.png`, `fig9b_softmax_landing_change.png`, `fig10_softmax_width.png`):
softmax executed bins per target spout, its change from pre, and the softmax width sweep of the contest.

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from scratchpad_scripts_backup_20261104/S1_cdd1e016/
  fig_softmax.py. Computation, layout and styling unchanged (the original body, wrapped in draw()). Changes: inputs
  are THIS run's binned-contest encoder runs (`run_mode.encoding_out_root()/<BINS>{,_bw0.5,_bw2}`; was a hard-coded
  adopted-run M: root), --out / --plot-only / figdata (`_common`; figdata = the session JSONs + region tables).

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_softmax.py [--out DIR] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

BINS = "movement_first_subset-nokin_exec-polar_licks-combined_bins"


def session_jsons() -> list:
    """THIS run's binned-contest encoder sessions (`run_mode.encoding_out_root()/<BINS>/sessions/*.json`), sorted."""
    from wfield_local import run_mode
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((run_mode.encoding_out_root() / BINS / "sessions").glob("*.json"))]

NAME = "softmax"


def compute() -> dict:
    from wfield_local import run_mode
    E = run_mode.encoding_out_root()
    return {"sessions": session_jsons(),
            "bins": {f: pd.read_csv(E / f"{BINS}{f}" / "tables" / "region_measures.csv") for f in ("_bw0.5", "", "_bw2")}}


def draw(fd: dict, OUT: str) -> None:
    SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
    LAB = ["cI", "cM", "cC", "fI", "fM", "fC"]
    AN = ["PS92", "PS93", "PS94", "PS95"]; EP = ["pre", "acute", "subacute", "chronic"]
    acc = {}
    for j in fd["sessions"]:
        b = j.get("bins") or {}
        sw = b.get("soft_weight") or {}
        conf = b.get("confusion") or {}
        for t in SP:
            if t not in sw:
                continue
            n = sum(conf.get(t, {}).values()) or 1          # licks at that target (for weighting sessions)
            row = np.array([sw[t].get(e, 0.0) for e in SP]) * n
            k = (j["animal"], j["epoch"], t)
            a = acc.setdefault(k, [np.zeros(6), 0]); a[0] += row; a[1] += n
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i, an in enumerate(AN):
        for k, ep in enumerate(EP):
            M = np.full((6, 6), np.nan)
            for r, t in enumerate(SP):
                if (an, ep, t) in acc:
                    s, n = acc[(an, ep, t)]; M[r] = s / n
            ax = axs[i, k]
            ax.imshow(M, cmap="viridis", vmin=0, vmax=1)          # level matrices: viridis 0-1 on every page figure
            for r in range(6):
                for c in range(6):
                    if np.isfinite(M[r, c]) and M[r, c] >= 0.1:
                        ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=6.5, color="k" if M[r, c] > 0.6 else "w")
            ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=7); ax.set_yticks(range(6)); ax.set_yticklabels(LAB, fontsize=7)
            if i == 0: ax.set_title(ep)
            if k == 0: ax.set_ylabel(f"{an}\ntarget spout (row)")
            if i == 3: ax.set_xlabel("executed soft bin (where licks landed)")
    fig.suptitle("Softmax executed bins (width 1): for each TARGET spout (row), the mean soft weight of its licks on each "
                 "executed bin (column); rows sum to 1.\nDiagonal = landed like the target's pre-stroke licks.\n"
                 "Acute far-contra (fC row) licks land as far-mid / far-ipsi; c = close, f = far; I / M / C = ipsi / mid / contra.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_svg(fig, OUT + "/fig9_softmax_landing.png", dpi=130); plt.close(fig)

    # width sweep
    rows = []
    for f, w in [("_bw0.5", 0.5), ("", 1), ("_bw2", 2)]:
        d = fd["bins"][f].copy()
        c = d[d.pos == "cortex"]
        for win in ("cue", "lick"):
            g = c.groupby(["animal", "epoch"])[[f"nokin:u:tgtbin|{win}", f"nokin:u:exbin|{win}"]].mean()
            for (an, ep), r in g.iterrows():
                rows.append(dict(width=w, window=win, animal=an, epoch=ep, target=r.iloc[0], executed=r.iloc[1]))
    T = pd.DataFrame(rows)
    fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharey="row")
    for r, win in enumerate(("lick", "cue")):
        for k, an in enumerate(AN):
            ax = axs[r, k]
            for j, ep in enumerate(EP):
                x = T[(T.window == win) & (T.animal == an) & (T.epoch == ep)].sort_values("width")
                off = j * 4
                ax.plot(np.arange(3) + off, x.target, "-o", color="#6a3d9a", label="target" if (j == 0 and k == 0) else None)
                ax.plot(np.arange(3) + off, x.executed, "-s", color="#e66101", label="executed" if (j == 0 and k == 0) else None)
            ax.axhline(0, color="k", lw=0.6)
            ax.spines["top"].set_visible(False)              # no top frame line (Priya 11-03)
            ax.set_xticks([j * 4 + 1 for j in range(4)]); ax.set_xticklabels(EP, fontsize=8)
            if r == 0: ax.set_title(an)
            if k == 0: ax.set_ylabel(f"{win} window\nunique dR2 (R2 fraction)")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Softmax width sweep: within each epoch the three points are widths 0.5 / 1 / 2 (left to right).\nChronic target "
                 "leads executed at every width in all four animals, except PS94 at width 2 in the lick window (executed slightly "
                 "ahead); the executed set gains as bins widen.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_png_svg(fig, OUT + "/fig10_softmax_width.png", dpi=130)
    print("ok")

    # epoch minus pre, within animal, then the cross-animal mean (Priya 10-24)
    def mat(an, ep):
        M = np.full((6, 6), np.nan)
        for r, t in enumerate(SP):
            if (an, ep, t) in acc:
                s, n = acc[(an, ep, t)]; M[r] = s / n
        return M
    EPP = ["acute", "subacute", "chronic"]
    D = {(an, ep): mat(an, ep) - mat(an, "pre") for an in AN for ep in EPP}
    fig, axs = plt.subplots(5, 3, figsize=(9.5, 15))
    for i, an in enumerate(AN + ["mean"]):
        for k, ep in enumerate(EPP):
            M = np.nanmean([D[(a, ep)] for a in AN], 0) if an == "mean" else D[(an, ep)]
            ax = axs[i, k]
            ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            for r in range(6):
                for c in range(6):
                    if np.isfinite(M[r, c]) and abs(M[r, c]) >= 0.1:
                        ax.text(c, r, f"{M[r, c]:+.2f}", ha="center", va="center", fontsize=6.5, color="w" if abs(M[r, c]) > 0.4 else "k")
            ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=7); ax.set_yticks(range(6)); ax.set_yticklabels(LAB, fontsize=7)
            if i == 0: ax.set_title(f"{ep} minus pre")
            if k == 0: ax.set_ylabel(("MEAN of 4 animals" if an == "mean" else an) + "\ntarget spout (row)")
            if i == 4: ax.set_xlabel("executed soft bin")
            if an == "mean":
                for s in ax.spines.values(): s.set_linewidth(2)
    fig.suptitle("LICK DATA (behaviour, no neural signal): change in where licks land, epoch minus pre within each animal, "
                 "then the mean of animals (bottom row).\nBlue on the diagonal = fewer licks landing like their own target; "
                 "red off-diagonal = landing like another spout. Softmax width 1.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_svg(fig, OUT + "/fig9b_softmax_landing_change.png", dpi=130); plt.close(fig)
    print("ok 9b")

def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    fd = _common.load_figdata(a.out, NAME) if a.plot_only else compute()
    if not a.plot_only:
        _common.save_figdata(a.out, NAME, fd)
    draw(fd, str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
