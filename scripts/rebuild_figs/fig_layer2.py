"""Goal page Layer-2 Figs 19.x: `fig5_layer2_change.png` (each epoch's kernel CHANGE vs the drift null) and
`fig6_layer2_groups.png` (per kernel group).

PROVENANCE: recreated 2026-11-04 (cohort-1 rebuild, RECONCILIATION section 7) from the scratchpad-only
  scratchpad_scripts_backup_20261104/S1_cdd1e016/fig_layer2.py. Computation, layout, seeds and styling unchanged.
  Changes: the input folder is THIS run's allsession encoder (`run_mode.encoding_out_root()/allsession_encoder_<tag>`,
  default v7g3_t3d_ref; was a hard-coded adopted-run M: path), --out / --plot-only / figdata (`_common`).
INPUT: <encoding>/allsession_encoder_<tag>/allsession_changes_by_session.csv + matched_n_control.csv (written by
  `wfield_local.allsession_encoder` and `scripts/layer2_matched_n.py`).

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_layer2.py [--allsession-tag v7g3_t3d_ref] [--out DIR] [--plot-only]
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
import _common  # noqa: E402

NAME = "layer2"
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre_late", "acute", "subacute", "chronic"]
COL = {"pre_late": "0.6", "acute": "#d95f02", "subacute": "#e6ab02", "chronic": "#1b9e77"}
G = ["lick", "contact", "target", "exec_within", "exec_locked", "timing", "other"]


def allsession_dir(tag: str) -> Path:
    """THIS run's allsession encoder folder (the writer's routed root)."""
    from wfield_local import run_mode
    return run_mode.encoding_out_root() / f"allsession_encoder_{tag}"


def compute(d: Path) -> dict:
    W = pd.read_csv(d / "allsession_changes_by_session.csv", keep_default_na=False, na_values=[""])
    _c = [c for c in W.columns if c.startswith("change")]
    W[_c] = W[_c] / 1e3                     # stored x1e3 -> R2 fraction
    M = pd.read_csv(d / "matched_n_control.csv")
    M["change_ref"] = M["change_ref"] / 1e3
    return {"W": W, "M": M, "src": str(d)}


def draw(fd: dict, out: Path) -> None:
    W, M = fd["W"], fd["M"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharey="row")
    for r, area in enumerate(["cortex", "SSp-m_left"]):
        C = W[(W.fit == "combined") & (W.area == area) & (W.epoch != "pre")]
        for k, an in enumerate(AN):
            ax = axes[r, k]
            a = C[C.animal == an]
            ns = []
            for i, e in enumerate(EP):
                v = a[a.epoch == e].change_ref.to_numpy()
                ns.append(len(v))
                if len(v):
                    ax.bar(i, v.mean(), color=COL[e], width=0.75)
                    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=9,
                               color="k", alpha=0.5, lw=0)
            if area == "cortex":
                m = M[M.animal == an]
                for i, (sub, mk) in enumerate((("first", "v"), ("last", "^"))):  # noqa: B007 (the original's loop, unchanged)
                    v = m[(m.chronic_subset == sub) & (m.epoch == "chronic")].change_ref.mean()
                    ax.scatter(3.45, v, marker=mk, s=40, color="k", zorder=5,
                               label=f"chronic, {sub} 4 sessions" if k == 0 else None)
            nul = a[a.epoch == "pre_late"].change_ref.mean()
            ax.axhline(nul, color="0.4", ls="--", lw=1, label="drift null (pre_late mean)" if k == 0 and r == 0 else None)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"{t}\nn={n}" for t, n in zip(["pre late (null)", "acute", "subacute", "chronic"], ns)],
                               fontsize=8)
            if r == 0:
                ax.set_title(an)
            if k == 0:
                ax.set_ylabel(f"{area}\nchange from early-pre kernels\nheld-out R2 gain (fraction)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Layer 2 (redesigned): how much each epoch's kernel CHANGE improves held-out prediction, measured from the "
                 "early pre-stroke kernels in one fit\nbars = epoch means, dots = sessions; triangles = chronic refit with "
                 "only its first / last 4 sessions (matched to the null's 4)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "fig5_layer2_change.png", dpi=140)
    plt.close(fig)

    # per-group unique change, cortex
    C = W[(W.fit == "combined") & (W.area == "cortex") & (W.epoch.isin(["pre_late", "chronic", "acute"]))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for k, an in enumerate(AN):
        ax = axes[k]
        a = C[C.animal == an].groupby("epoch")[[f"change_{g}" for g in G]].mean()
        for j, e in enumerate(["pre_late", "acute", "chronic"]):
            if e in a.index:
                ax.bar(np.arange(len(G)) + (j - 1) * 0.27, a.loc[e].to_numpy(), width=0.27, color=COL[e],
                       label={"pre_late": "pre late (drift null)", "acute": "acute", "chronic": "chronic"}[e]
                       if k == 0 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(G)))
        ax.set_xticklabels(G, rotation=40, ha="right", fontsize=8)
        ax.set_title(an)
    axes[0].set_ylabel("unique change of one group\nheld-out R2 (fraction), cortex")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Layer 2 per kernel group: no single group (lick = motor timing, contact = touch, target, direction ...) "
                 "changes more chronically than the drift null", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig6_layer2_groups.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--allsession-tag", default="v7g3_t3d_ref", help="allsession_encoder_<tag> folder of this run")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    fd = _common.load_figdata(a.out, NAME) if a.plot_only else compute(allsession_dir(a.allsession_tag))
    if not a.plot_only:
        _common.save_figdata(a.out, NAME, fd)
    draw(fd, a.out)
    print(f"-> {a.out}/fig5_layer2_change.png, fig6_layer2_groups.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
