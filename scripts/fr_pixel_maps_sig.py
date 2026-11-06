"""Musall-style FaceRhythm PIXEL maps WITH their significance on the same panels (Priya 10-25: "what happened with stats on the
FR neural maps eg 16g" -- the tests were drawn only in the separate significance figure).

Reads the per-animal maps saved by `fr_musall.py` (fr_musall_pixel_maps_mode<M>.npz, keys "epoch|animal|quantity|window")
and the tests saved by `fr_pixel_significance.py` (pixel_significance/sig_<quantity>_<window>_<epoch>_<16|32>px.npz: nested
bootstrap animals -> sessions, Bonferroni over bins; calibrated, 5 % of random pre half-splits give >= 1 false bin).
Rows: pre level, then acute / subacute / chronic minus pre (within animal, then mean of animals); contours = significant
bins (black solid 16 px, purple dashed 32 px); "not tested" where no test exists (FaceRhythm alone). Image left = ipsilesional.

Usage: python scripts/fr_pixel_maps_sig.py [--mode A] [--windows cue lick]
Output: <E>/fr_encoding_summary/figures/fr_pixel_maps_sig_<window>_mode<M>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E, PIX  # noqa: E402

FACTORS = {16: 8, 32: 16}
EPS = ["acute", "subacute", "chronic"]


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from combined_encoding_figures import _outline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="A")
    ap.add_argument("--windows", nargs="+", default=["cue", "lick"])        # cue before lick (Priya 10-25)
    a = ap.parse_args(argv)
    S = E / "fr_encoding_summary"
    z = np.load(S / f"fr_musall_pixel_maps_mode{a.mode}.npz")
    A = {tuple(str(k).split("|")): m for k, m in zip(z["keys"], z["maps"])}       # (epoch, animal, q, w) -> map
    sigdir = S / "pixel_significance"
    labs = sorted({lab for lab in pd.read_csv(S / f"fr_musall_by_animal_mode{a.mode}.csv").get("label", [])}) \
        if (S / f"fr_musall_by_animal_mode{a.mode}.csv").exists() else []
    ol = _outline(pd.DataFrame({"label": labs})) if labs else None
    for w in a.windows:
        rows_ = [("pre (level)", None)] + [(f"{ep} minus pre", ep) for ep in EPS]
        fig, axs = plt.subplots(len(rows_), len(PIX), figsize=(2.3 * len(PIX), 2.45 * len(rows_) + 0.9), squeeze=False)
        for j, (q, qlab) in enumerate(PIX):
            ans = [an for an in AN if ("pre", an, q, w) in A]
            dl = {ep: np.nanmean([A[(ep, an, q, w)] - A[("pre", an, q, w)] for an in ans if (ep, an, q, w) in A], 0)
                  for ep in EPS}
            lim = max(np.nanpercentile(np.abs(d), 99) for d in dl.values())        # one change scale per column
            for r, (title, ep) in enumerate(rows_):
                ax = axs[r, j]
                if ep is None:
                    M = np.nanmean([A[("pre", an, q, w)] for an in ans], 0)
                    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=np.nanpercentile(M, 99), interpolation="nearest")
                else:
                    M = dl[ep]
                    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                if ol is not None and ol.shape == M.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                if ep is not None:
                    txt = []
                    for F, col, ls in ((16, "k", "solid"), (32, "#7a0177", "dashed")):
                        fp = sigdir / f"sig_{q}_{w}_{ep}_{F}px.npz"
                        if not fp.exists():
                            continue
                        sig = np.load(fp)["sig"]
                        f = FACTORS[F]
                        if sig.any():
                            big = np.zeros(M.shape)
                            hh, ww = min(sig.shape[0] * f, M.shape[0]), min(sig.shape[1] * f, M.shape[1])
                            big[:hh, :ww] = np.kron(sig, np.ones((f, f)))[:hh, :ww]
                            ax.contour(big, levels=[0.5], colors=col, linewidths=0.9, linestyles=ls)
                        txt.append(f"{F}px {int(sig.sum())}")
                    ax.text(0.02, 0.02, "  ".join(txt) if txt else "not tested", transform=ax.transAxes, fontsize=6.5,
                            va="bottom", bbox={"facecolor": "white", "alpha": 0.7, "pad": 1, "lw": 0})
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if r == 0:
                    ax.set_title(qlab, fontsize=8)
                if j == 0:
                    ax.set_ylabel(title, fontsize=9)
                if r in (0, len(rows_) - 1):
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation="horizontal")
                    cb.ax.tick_params(labelsize=6)
        fig.suptitle(f"NEURAL: Musall-style PIXEL maps WITH significance, per-session encoder with FaceRhythm (mode {a.mode}), "
                     f"{w} window; held-out R2 (fraction, 0-1) per 2x2-px block, within animal then mean of 4.\nRow 1 pre level "
                     "(viridis); rows 2-4 epoch minus pre (RdBu, one scale per column). Contours = significant bins: black 16 px, "
                     "purple dashed 32 px (nested bootstrap animals > sessions, Bonferroni over bins; corner = number of "
                     "significant bins). Image left = ipsilesional.", fontsize=9)
        fig.subplots_adjust(left=0.03, right=0.995, top=0.9, bottom=0.03, wspace=0.08, hspace=0.14)
        p = S / "figures" / f"fr_pixel_maps_sig_{w}_mode{a.mode}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print("->", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
