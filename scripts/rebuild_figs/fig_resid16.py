"""Goal page Figs 7c / 7d (`fig7c_residual_allspouts_<F>px.png`, `fig7d_residual_perspout_{lick,cue}_<F>px.png`):
pooled frozen-encoder residual maps at F px bins with significant-bin contours (all spouts pooled; per spout).

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from scratchpad_scripts_backup_20261104/S1_cdd1e016/
  fig_resid16.py. Computation, layout and styling unchanged. Changes: the encoder run is THIS run's
  (`frozen_encoder.out_root()/frozen_encoder_<tag>`; was a hard-coded adopted-run M: path), positional args -> --bin /
  --out, --plot-only / figdata (`_common`: the residual npz arrays, the significance table, the area outline).
INPUT: <run>/residual_sig_<F>px.csv, <run>/resid<F>_<spout|all>_<window>_<epoch>.npz (frozen_residual_significance).

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_resid16.py [--bin 16] [--encoder-tag gate-attempted_v7g3_t3d_basis] [--out DIR] [--plot-only]
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402
from fig_resid_per_animal import encoder_run  # noqa: E402

EP = ["acute", "subacute", "chronic"]
SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]


def load_maps(O: Path, F: int, spouts) -> dict:
    """{"R": significance table, "outline": area outline, "npz": {(spout, window, epoch): (mean, sig, H, W)}}."""
    from frozen_area_maps import area_image, outline
    lab = sorted((O / "joint" / "sessions").glob("PS92_*.json"))[0].stem
    out = {"R": pd.read_csv(O / f"residual_sig_{F}px.csv"), "outline": outline(area_image(lab)[0]), "npz": {}, "F": F,
           "src": str(O)}
    for sp in spouts:
        for al in ("lick", "cue"):
            for ep in EP:
                z = np.load(O / f"resid{F}_{sp}_{al}_{ep}.npz")
                out["npz"][(sp, al, ep)] = (z["mean"], z["sig"], int(z["H"]), int(z["W"]))
    return out


def panel(fd, ax, sp, al, ep, lim=None):
    F, R, ol = fd["F"], fd["R"], fd["outline"]
    mm, sig, H, W = fd["npz"][(sp, al, ep)]
    lim = lim or (np.nanpercentile(np.abs(mm), 98) or 1)
    im = ax.imshow(mm, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    panel.im = im
    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.4, interpolation="nearest")
    if sig.size > 1 and sig.any():
        big = np.zeros((H, W))
        big[:sig.shape[0] * F, :sig.shape[1] * F] = np.kron(sig, np.ones((F, F)))
        ax.contour(big, levels=[0.5], colors="k", linewidths=1.0)
    r = R[(R.spout == sp) & (R.window == al) & (R.epoch == ep)].iloc[0]
    ax.set_xticks([])
    ax.set_yticks([])
    return r, lim


def draw(fd: dict, OUT: Path) -> None:
    F = fd["F"]
    # all spouts pooled
    fig, axs = plt.subplots(2, 3, figsize=(11, 8.4))
    for i, al in enumerate(("lick", "cue")):
        lim = max(np.nanpercentile(np.abs(fd["npz"][("all", al, e)][0]), 98) for e in EP)
        for j, ep in enumerate(EP):
            r, _ = panel(fd, axs[i, j], "all", al, ep, lim)
            axs[i, j].set_title(f"{al} window, {ep} minus pre\n{r.sig}/{r.n_bins} bins ({r.up} up, {r.down} down)",
                                fontsize=9)
    fig.suptitle(f"NEURAL: frozen-encoder RESIDUAL, ALL 6 SPOUTS POOLED, epoch minus all-LOSO-pre, pre-normalized, mean of 4 "
                 f"animals.\nBlack = significant {F} px bins (nested bootstrap animals -> sessions, Bonferroni). Image left = "
                 f"ipsilesional.", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.86, wspace=0.05, hspace=0.32)
    for i in range(2):
        fig.colorbar(axs[i, 0].images[0], ax=list(axs[i]), fraction=0.025, pad=0.02,
                     label="residual change\n(/ animal's pre-stroke scale)")
    save_png_svg(fig, OUT / f"fig7c_residual_allspouts_{F}px.png", dpi=125)
    plt.close(fig)
    # per spout
    for al in ("lick", "cue"):
        fig, axs = plt.subplots(3, 6, figsize=(17, 10))
        lim = max(np.nanpercentile(np.abs(fd["npz"][(s, al, e)][0]), 98) for s in SP for e in EP)
        for j, sp in enumerate(SP):
            for i, ep in enumerate(EP):
                r, _ = panel(fd, axs[i, j], sp, al, ep, lim)
                axs[i, j].set_title(f"{sp} {ep}\n{r.sig}/{r.n_bins} ({r.up} up, {r.down} down)", fontsize=8)
        fig.suptitle(f"NEURAL: frozen-encoder residual PER SPOUT, {al} window, epoch minus pre (same scale in all panels); "
                     f"black = significant {F} px bins. Image left = ipsilesional.", fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.92, bottom=0.01, top=0.9, wspace=0.05, hspace=0.35)
        fig.colorbar(axs[0, 0].images[0], ax=list(axs.ravel()), fraction=0.015, pad=0.01,
                     label="residual change\n(/ animal's pre-stroke scale)")
        save_png_svg(fig, OUT / f"fig7d_residual_perspout_{al}_{F}px.png", dpi=110)
        plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--bin", type=int, default=16, help="F: the significance bin size, px")
    ap.add_argument("--encoder-tag", default=_design_tag("gate-attempted_v7g3_t3d_basis"), help="frozen_encoder_<tag> run of this run")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"resid{a.bin}"
    fd = (_common.load_figdata(a.out, name) if a.plot_only
          else load_maps(encoder_run(a.encoder_tag), a.bin, ["all", *SP]))
    if not a.plot_only:
        _common.save_figdata(a.out, name, fd)
    draw(fd, a.out)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
