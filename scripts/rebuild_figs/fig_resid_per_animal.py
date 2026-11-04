"""Goal page Fig 8 (`fig8_residual_per_animal_{lick,cue}.png`): frozen-encoder RESIDUAL change per animal (epoch - all
LOSO pre), far contra, each animal normalised by its own pre-stroke scale; contour = within-animal test.

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from scratchpad_scripts_backup_20261104/S1_cdd1e016/
  fig_resid_per_animal.py. Computation, layout and styling unchanged. Changes: the encoder run is THIS run's
  (`frozen_encoder.out_root()/frozen_encoder_<tag>`, default gate-attempted_v7g3_t3d_basis; was a hard-coded adopted-run
  M: path), --out / --plot-only / figdata (`_common`: the panels + the area outline).

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_resid_per_animal.py [--encoder-tag gate-attempted_v7g3_t3d_basis] [--out DIR] [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

NAME = "resid_per_animal"
EP = ("acute", "subacute", "chronic")


def encoder_run(tag: str) -> Path:
    """THIS run's frozen-encoder run folder (the encoder's tagged root)."""
    from wfield_local import frozen_encoder as fe
    return fe.out_root() / f"frozen_encoder_{tag}"


def compute(out: Path) -> dict:
    from frozen_area_maps import area_image, outline
    from frozen_residual_significance import run_tests, session_residual_maps

    from wfield_local import beta_maps as bm
    res = {}
    for al in ("lick", "cue"):
        by, H, W = session_residual_maps(out, "joint", "far_contra", al)
        img, _ = area_image(next(iter(by.values()))[0][0])
        ol = outline(img)
        ans = sorted(by)
        panels = {}
        for an in ans:
            pre = [s[3] for s in by[an] if s[1] == "pre"]
            sc = bm.pre_scale({an: pre})[an]
            for ep in EP:
                post = [s[3] for s in by[an] if s[1] == ep]
                if not post:
                    continue
                d = (np.nanmean(post, 0) - np.nanmean(pre, 0)) / sc
                mk, n = None, 0
                if len(post) >= 2:
                    r = run_tests({an: pre}, {an: post}, methods=("musall",))
                    mk, n = r["musall"]["mask"], r["musall"]["sig_bins"]
                panels[(an, ep)] = (d, mk, n, len(post), len(pre))
        res[al] = {"ans": ans, "panels": panels, "outline": ol}
    return res


def draw(fd: dict, OUT: Path) -> None:
    for al in ("lick", "cue"):
        ans, panels, ol = fd[al]["ans"], fd[al]["panels"], fd[al]["outline"]
        lim = np.nanpercentile(np.abs(np.concatenate([p[0][np.isfinite(p[0])] for p in panels.values()])), 98)
        fig, axs = plt.subplots(len(ans), 3, figsize=(10.5, 3.0 * len(ans)), squeeze=False)
        for i, an in enumerate(ans):
            for j, ep in enumerate(EP):
                ax = axs[i][j]
                ax.set_xticks([])
                ax.set_yticks([])
                if (an, ep) not in panels:
                    ax.set_facecolor("0.93")
                    ax.set_title(f"{an} {ep}: no sessions", fontsize=8)
                    continue
                d, mk, n, npost, npre = panels[(an, ep)]
                ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.4, interpolation="nearest")
                if mk is not None and mk.any():
                    ax.contour(mk.astype(float), levels=[0.5], colors="k", linewidths=0.9)
                ttl = f"{an} {ep} (n {npost} vs {npre} pre)\n" + (f"{n}/2022 bins" if npost >= 2 else "1 session: not testable")
                ax.set_title(ttl, fontsize=8)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-lim, lim))
        fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.01, label="residual change / animal's pre-stroke scale")
        fig.suptitle(f"Frozen-encoder RESIDUAL change per animal (epoch - all LOSO pre), far contra, {al} window, each animal "
                     f"normalised by its own pre-stroke scale, shared colour scale.\nContour = within-animal test, sessions as "
                     f"unit (UNCALIBRATED: see random-split null). Image left = ipsilesional.", fontsize=8)
        fig.savefig(OUT / f"fig8_residual_per_animal_{al}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--encoder-tag", default="gate-attempted_v7g3_t3d_basis", help="frozen_encoder_<tag> run of this run")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    fd = _common.load_figdata(a.out, NAME) if a.plot_only else compute(encoder_run(a.encoder_tag))
    if not a.plot_only:
        _common.save_figdata(a.out, NAME, fd)
    draw(fd, a.out)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
