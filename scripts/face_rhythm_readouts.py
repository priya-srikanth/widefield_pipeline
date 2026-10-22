"""FaceRhythm session readouts: tables + figures for a frozen factor set (computation in
`wfield_local/face_rhythm_readouts.py`; pipeline: docs/FACE_RHYTHM_PIPELINE.md).

    python scripts/face_rhythm_readouts.py [--set master_pre_v1] [--bundle fr_cam4_20261021] [--band all] [--out DIR]

Needs ``<set>/master.npz`` + ``proj/`` (and ``own/`` for the EVR gap / morphing). Writes to
``<dlc>/face_rhythm/cam4/readouts/<set>/`` (default; CSVs in its ``data/`` subfolder, `figure_layout.sidecar_for`):
  components.csv           per frozen component: peak Hz, face-region weights, lick-rate correlation, response /
                           session ratio, far-L vs far-R selectivity, SUGGESTED name; the suggestions also go to
                           <set>/labels_suggested.json -- copy to <set>/labels.json once checked against the gallery
                           (labels.json names the loader's columns; it is Priya's call, never written here)
  session_table.csv        one row per session: amplitudes per component x window, reconstruction error, EVR own /
                           frozen / gap, matched cosines, contra / ipsi movement power + asymmetry; animal / epoch / day
  position_amplitudes.csv  response-window amplitude per session x position x component
  epoch_means.csv / epoch_deltas.csv   per-animal epoch means; paired animal-weighted change vs pre (boot_delta)
  amplitudes_by_epoch.png  per component: response / between amplitude per animal across epochs
  recovery_by_day.png      reconstruction error, EVR gap, spatial / spectral morphing vs days since stroke
  movement_maps_<measure>_<window>.png   log2(epoch / pre) per point on the master face, per animal + animal mean
                           (speed = absolute movement; share = normalised spectral distribution)
  asymmetry_by_epoch.png   (contra - ipsi) / (contra + ipsi) per window, speed and share (cam4 ipsi = image right)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EPOCHS = ("pre", "acute", "subacute", "chronic")


def _epoch_lines(ax, S, col, title):
    for a, g in S.groupby("animal"):
        m = g.groupby("epoch")[col].mean().reindex(EPOCHS)
        ax.plot(range(4), m.to_numpy(), "o-", ms=3, lw=1, label=a)
    ax.set_xticks(range(4), [e[:5] for e in EPOCHS], fontsize=7)
    ax.set_title(title, fontsize=8)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--band", default="all", choices=tuple(frr.BANDS), help="frequency band of the SHARE measure")
    ap.add_argument("--measure", default="speed", choices=("speed", "share"),
                    help="movement maps: absolute speed of the points, or the normalised spectral share")
    ap.add_argument("--midline-x", type=float, default=None, help="image x of the face midline (default: "
                                                                   "frr.MASTER_MIDLINE_X, the master's LP nose)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fdir", type=Path, default=None, help="factor-set folder (default: <bundle>/factors/<set>)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    M = np.load(fdir / "master.npz")
    names, home, freqs = fro.labels(fdir), M["home"], M["freqs"]

    C = frr.describe_components(fdir)
    C.to_csv(figure_layout.sidecar_for(out, "components", ".csv"), index=False, float_format="%.4g")
    (fdir / "labels_suggested.json").write_text(json.dumps({str(k): n for k, n in zip(C.comp, C.suggested)}, indent=1))
    print(C[["comp", "suggested", "peak_hz", "top_region", "w_rim", "lick_rate_corr", "farL_vs_farR"]]
          .round(3).to_string(index=False))
    S = frr.session_table(fdir, names, home, freqs)
    S.to_csv(figure_layout.sidecar_for(out, "session_table", ".csv"), index=False, float_format="%.5g")
    _, Pos = frr.amplitudes(fdir, names)
    Pos.to_csv(figure_layout.sidecar_for(out, "position_amplitudes", ".csv"), index=False, float_format="%.5g")
    metrics = [c for c in S.columns if c.startswith(("amp_", "recon_err", "evr_", "cos_", "asym_"))]
    A, D = frr.epoch_stats(S, metrics)
    A.to_csv(figure_layout.sidecar_for(out, "epoch_means", ".csv"), index=False, float_format="%.5g")
    D.to_csv(figure_layout.sidecar_for(out, "epoch_deltas", ".csv"), index=False, float_format="%.5g")
    key = [c for c in metrics if c.startswith(("recon_err_trials", "evr_gap", "asym_speed_response"))]
    print("PER-ANIMAL epoch means (rule 8: read these before the pooled deltas)")
    print(A[["animal", "epoch", "n_sessions", *key]].round(3).to_string(index=False))

    # amplitudes by epoch
    fig, axs = plt.subplots(len(names), 3, figsize=(10, 1.9 * len(names)), squeeze=False)
    for k, nm in enumerate(names):
        for j, w in enumerate(("precue", "response", "between")):
            col = f"amp_{nm}_{w}"
            if col in S:
                _epoch_lines(axs[k, j], S, col, f"{nm} {w}")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "amplitudes_by_epoch.png", dpi=110)
    plt.close(fig)

    # recovery by day
    cols = [c for c in ("recon_err_trials", "recon_err_session", "evr_gap", "cos_spatial_mean", "cos_spectral_mean")
            if c in S]
    fig, axs = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3), squeeze=False)
    for ax, c in zip(axs[0], cols):
        for an, g in S.groupby("animal"):
            ax.plot(g.day, g[c], "o-", ms=3, lw=0.8, label=an)
        ax.axvline(0.5, color="k", lw=0.5)
        ax.set(xlabel="days since stroke", title=c)
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "recovery_by_day.png", dpi=110)
    plt.close(fig)

    # movement maps + asymmetry
    _, maps = frr.movement(fdir, home, freqs, a.midline_x, a.band, a.measure)
    animals = sorted({k[0] for k in maps if isinstance(k, tuple)})
    for w in ("between", "precue", "response"):
        fig, axs = plt.subplots(len(animals) + 1, 3, figsize=(8, 2.6 * (len(animals) + 1)), squeeze=False)
        for j, ep in enumerate(EPOCHS[1:]):
            mm = []
            for i, an in enumerate(animals):
                ax = axs[i, j]
                ax.axis("off")
                v = maps.get((an, ep, w))
                if v is None:
                    continue
                mm.append(v)
                ax.scatter(home[:, 0], home[:, 1], c=v, s=3, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.axvline(maps["midline_x"], color="k", lw=0.4)
                ax.set_title(f"{an} {ep}", fontsize=8)
                ax.invert_yaxis()
                ax.set_aspect("equal")
            ax = axs[-1, j]
            ax.axis("off")
            if mm:
                sc = ax.scatter(home[:, 0], home[:, 1], c=np.mean(mm, 0), s=3, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_title(f"animal mean {ep} (n={len(mm)})", fontsize=8)
                ax.invert_yaxis()
                ax.set_aspect("equal")
        what = "speed (absolute)" if a.measure == "speed" else f"spectral share ({a.band} band; normalised per frame)"
        fig.suptitle(f"{w} movement {what}: log2(epoch / pre) per point; image right = ipsilesional", fontsize=9)
        fig.colorbar(sc, ax=axs, shrink=0.4) if animals else None
        fig.savefig(out / f"movement_maps_{a.measure}_{w}.png", dpi=110)
        plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6))
    for i, m in enumerate(("speed", "share")):
        for ax, w in zip(axs[i], ("between", "precue", "response")):
            _epoch_lines(ax, S, f"asym_{m}_{w}", f"{m}: (contra - ipsi) / sum, {w}")
            ax.axhline(0, color="k", lw=0.5)
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "asymmetry_by_epoch.png", dpi=110)
    plt.close(fig)
    print(f"{len(S)} sessions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
