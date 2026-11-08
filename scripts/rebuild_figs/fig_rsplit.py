"""Goal page Fig 5b (Layer-2 Figs 19.x): `fig5b_layer2_random_splits.png` -- Layer 2 random-split null per animal (20
random pre pseudo-epochs) vs each post epoch.

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from scratchpad_scripts_backup_20261104/S1_cdd1e016/
  fig_rsplit.py. Computation, layout, seeds and styling unchanged. Changes: the input is THIS run's allsession encoder
  folder (`run_mode.encoding_out_root()/allsession_encoder_<tag>`; was a hard-coded adopted-run M: path); the summary
  CSV goes to --out (the original wrote it INTO the input folder); --out / --plot-only / figdata (`_common`).
INPUT: <encoding>/allsession_encoder_<tag>/random_splits_<animal>.csv (`scripts/layer2_random_splits.py`).

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_rsplit.py [--allsession-tag v7g3_t3d_ref] [--out DIR] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
from wfield_local.frozen_encoder import design_tag as _design_tag  # frame tag of the FROZEN_LICKS_ALL design (8.51)

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from fig_layer2 import allsession_dir  # noqa: E402

NAME = "layer2_random_splits"
AN = ["PS92", "PS93", "PS94", "PS95"]
AR = ["cortex", "SSp-m_left", "SSp-m_right"]
EP = ["pseudo", "acute", "subacute", "chronic"]
COL = {"pseudo": "0.55", "acute": "#d95f02", "subacute": "#e6ab02", "chronic": "#1b9e77"}


def compute(d: Path) -> dict:
    T = {}
    for an in AN:
        t = pd.read_csv(d / f"random_splits_{an}.csv")
        t["change_ref"] = t["change_ref"] / 1e3   # stored x1e3
        T[an] = t
    return {"T": T, "src": str(d)}


def draw(fd: dict, out: Path) -> pd.DataFrame:
    fig, axs = plt.subplots(3, 4, figsize=(14, 10), sharey="row")
    rows = []
    for j, an in enumerate(AN):
        T = fd["T"][an]
        M = T.groupby(["split", "area", "epoch"]).change_ref.mean().unstack("epoch")
        for i, ar in enumerate(AR):
            ax = axs[i, j]
            m = M.xs(ar, level="area")
            labs = []
            for k, ep in enumerate(EP):
                if ep not in m:
                    labs.append(f"{ep}\n(none)")
                    continue
                v = m[ep].dropna().to_numpy()
                ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=12,
                           color=COL[ep], alpha=0.7, lw=0)
                ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color="k", lw=1.5)
                if ep == "pseudo":
                    labs.append("random pre\nsplit (null)")
                    ax.axhspan(np.quantile(v, 0.05), np.quantile(v, 0.95), color="0.85", zorder=0)
                else:
                    p = float((m["pseudo"] >= m[ep]).mean())
                    labs.append(f"{ep}\nP={p:.2f}")
                    rows.append(dict(animal=an, area=ar, epoch=ep, median=np.median(v),
                                     null_median=m["pseudo"].median(), p=p))
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(len(EP)))
            ax.set_xticklabels(labs, fontsize=8)
            if i == 0:
                ax.set_title(an)
            if j == 0:
                ax.set_ylabel(f"{ar}\nchange_ref, held-out R2 gain (fraction)")
    fig.suptitle("NEURAL: Layer 2 random-split null. Each dot = one of 20 fits in which a random set of August pre sessions "
                 "plays a\npseudo epoch (gray; shaded = its 5-95% range) alongside the real post epochs; line = median over "
                 "splits.\nP = share of splits where the pseudo epoch gains at least as much as that epoch (0.00 = none of "
                 "20).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_svg(fig, out / "fig5b_layer2_random_splits.png", dpi=125)
    plt.close(fig)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--allsession-tag", default=_design_tag("v7g3_t3d_ref"), help="allsession_encoder_<tag> folder of this run")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    fd = _common.load_figdata(a.out, NAME) if a.plot_only else compute(allsession_dir(a.allsession_tag))
    if not a.plot_only:
        _common.save_figdata(a.out, NAME, fd)
    S = draw(fd, a.out)
    S.to_csv(a.out / "random_splits_summary.csv", index=False)
    if len(S):
        print(S.pivot_table(index=["area", "epoch"], columns="animal", values="p"))
    print(f"-> {a.out}/fig5b_layer2_random_splits.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
