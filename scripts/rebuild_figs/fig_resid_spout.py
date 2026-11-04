"""Goal page Fig 7 (`fig7_residual_<spout>_<F>px.png`): pooled frozen-encoder residual map of ONE spout (or "all") at F
px bins, lick and cue windows, with significant-bin contours.

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from scratchpad_scripts_backup_20261104/S1_cdd1e016/
  fig_resid_spout.py. Computation, layout and styling unchanged (it shares `fig_resid16.panel`, the same function
  the original duplicated). Changes: THIS run's encoder run (`frozen_encoder.out_root()`), positional args -> --bin /
  --out / --spout, --plot-only / figdata (`_common`).

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_resid_spout.py --spout far_contra [--bin 16] [--encoder-tag gate-attempted_v7g3_t3d_basis] [--out DIR] [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from fig_resid16 import EP, load_maps, panel  # noqa: E402
from fig_resid_per_animal import encoder_run  # noqa: E402


def draw(fd: dict, OUT: Path, SPT: str) -> None:
    F = fd["F"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 8.4))
    for i, al in enumerate(("lick", "cue")):
        lim = max(np.nanpercentile(np.abs(fd["npz"][(SPT, al, e)][0]), 98) for e in EP)
        for j, ep in enumerate(EP):
            r, _ = panel(fd, axs[i, j], SPT, al, ep, lim)
            axs[i, j].set_title(f"{al} window, {ep} minus pre\n{r.sig}/{r.n_bins} bins ({r.up} up, {r.down} down)",
                                fontsize=9)
    fig.suptitle(f"NEURAL: frozen-encoder RESIDUAL, {SPT.upper()}, epoch minus all-LOSO-pre, pre-normalised, mean of 4 "
                 f"animals.\nBlack = significant {F} px bins (nested bootstrap animals -> sessions, Bonferroni). Image left = "
                 f"ipsilesional.", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.86, wspace=0.05, hspace=0.32)
    for i in range(2):
        fig.colorbar(axs[i, 0].images[0], ax=list(axs[i]), fraction=0.025, pad=0.02,
                     label="residual change\n(/ animal's pre-stroke scale)")
    fig.savefig(OUT / f"fig7_residual_{SPT}_{F}px.png", dpi=125)
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--spout", required=True, help="a spout name or 'all'")
    ap.add_argument("--bin", type=int, default=16, help="F: the significance bin size, px")
    ap.add_argument("--encoder-tag", default="gate-attempted_v7g3_t3d_basis", help="frozen_encoder_<tag> run of this run")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"resid_{a.spout}_{a.bin}"
    fd = _common.load_figdata(a.out, name) if a.plot_only else load_maps(encoder_run(a.encoder_tag), a.bin, [a.spout])
    if not a.plot_only:
        _common.save_figdata(a.out, name, fd)
    draw(fd, a.out, a.spout)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
