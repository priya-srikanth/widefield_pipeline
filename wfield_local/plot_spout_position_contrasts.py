"""Plot atlas-overlaid mean images and pairwise spout-position contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


POSITION_NAMES = ["close_center", "close_L", "close_R", "far_center", "far_L", "far_R"]
PAIRWISE = [
    ("close_L", "close_R"),
    ("close_center", "close_R"),
    ("close_center", "close_L"),
    ("far_L", "far_R"),
    ("far_center", "far_R"),
    ("far_center", "far_L"),
    ("close_L", "far_L"),
    ("close_R", "far_R"),
    ("close_center", "far_center"),
]


# region_edges is centralized in wfield_local.atlas_overlay (symmetric edge fix)
from wfield_local.atlas_overlay import region_edges as _region_edges


def _overlay_edges(ax, edges: np.ndarray, alpha: float = 0.7) -> None:
    rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
    rgba[edges] = (0, 0, 0, alpha)
    ax.imshow(rgba, interpolation="nearest")


def _robust_lim(arrays, pct=99.0) -> float:
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(vals), pct)), 1e-9)


def _functional_slot(label: str) -> int:
    """Which slot of ``frames_average`` holds the 470 (functional) frames, for THIS session.

    Normally 1. PS92 2026-08-28 is 0: labcams saved a normal alternating session mislabelled
    single-channel, and the rescue relabel locked exposure offset 1, which swaps the slots. The
    override records that as `preprocess.svd.functional_channel`, and `preprocess.py` already reads
    it per session when cross-registering -- "a swapped session must still register against the
    reference's OWN 470 channel". This figure did not, so it printed "415 nm mean" over the 470
    image and vice versa (Priya, 2026-09-07, preprocessing deck).

    THE LABEL ARRIVES SUFFIXED. Callers pass the pipeline label -- `PS92_0828_affine8v1`, not
    `PS92_0828` -- and `config.defaults(session=...)` keys on the bare session. Looking up the
    suffixed form found no override and fell back to 1, which is the WRONG answer for the only
    session that has one, so the first regeneration after this fix reproduced the same mislabelled
    figure. The session id is extracted rather than assumed.

    Falls back to 1 only when the label does not contain a session id at all: an ad-hoc render
    should still work, and 1 is what every session but one uses. A label that DOES name a session
    resolves through config, so a missing override there is a real answer and not a guess.
    """
    import re

    from wfield_local import config

    m = re.match(r"(PS\d+_\d{4})", str(label))
    try:
        return int(config.defaults(session=m.group(1) if m else label)
                   ["preprocess"]["svd"]["functional_channel"])
    except Exception:                                                  # noqa: BLE001
        return 1


def _plot_mean_overlay(label: str, frames_average: np.ndarray, edges: np.ndarray, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    # LABELLED BY WAVELENGTH, INDEXED BY SLOT. The two coincide for every session except a swapped
    # one, where assuming slot 0 == 415 mislabels both panels -- the image is right and the caption
    # is wrong, which is the harder error to notice.
    func = _functional_slot(label)
    iso = 1 - func
    titles = ["415 nm mean (isosbestic)", "470 nm mean (functional)",
              "470 nm mean + Allen outlines"]
    arrays = [frames_average[iso], frames_average[func], frames_average[func]]
    for ax, title, arr in zip(axes, titles, arrays):
        ax.imshow(arr, cmap="gray", vmin=np.percentile(arr, 1), vmax=np.percentile(arr, 99.5))
        if "Allen" in title:
            _overlay_edges(ax, edges, alpha=0.8)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"{label} mean motion-corrected frames")
    out = outdir / f"{label}_mean_415_470_with_allen_overlay.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_pairwise(label: str, maps: dict[str, np.ndarray], edges: np.ndarray, outdir: Path) -> Path:
    contrasts = {}
    for a, b in PAIRWISE:
        if a in maps and b in maps:
            contrasts[f"{a} - {b}"] = maps[a] - maps[b]

    lim = _robust_lim(contrasts.values(), pct=99.0)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    im = None
    for ax, (name, arr) in zip(axes.ravel(), contrasts.items()):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim)
        _overlay_edges(ax, edges)
        ax.set_title(name)
        ax.set_axis_off()
    for ax in axes.ravel()[len(contrasts) :]:
        ax.set_axis_off()
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Difference in post-pre map")
    fig.suptitle(f"{label} pairwise spout-position contrasts from post-pre maps")
    out = outdir / f"{label}_pairwise_spout_position_delta_contrasts_allen_overlay.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot extra spout-position atlas overlays.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--trial-maps", type=Path, required=True)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    trial = np.load(args.trial_maps)
    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    frames_average = np.load(args.allen_dir / "frames_average_atlas.npy")
    edges = _region_edges(atlas)

    delta_maps = {
        pos: trial[f"{pos}_delta"]
        for pos in POSITION_NAMES
        if f"{pos}_delta" in trial.files
    }
    mean_png = _plot_mean_overlay(args.label, frames_average, edges, args.output)
    pair_png = _plot_pairwise(args.label, delta_maps, edges, args.output)

    payload = {}
    for a, b in PAIRWISE:
        if a in delta_maps and b in delta_maps:
            payload[f"{a}_minus_{b}"] = delta_maps[a] - delta_maps[b]
    np.savez_compressed(
        args.output / f"{args.label}_pairwise_spout_position_delta_contrasts.npz",
        **payload,
    )
    summary = {
        "label": args.label,
        "trial_maps": str(args.trial_maps),
        "allen_dir": str(args.allen_dir),
        "outputs": [str(mean_png), str(pair_png)],
        "pairwise_definition": "Each contrast is first condition's post-pre map minus second condition's post-pre map.",
        # DERIVED, not asserted. This string said "Channel 0 shown as 415 nm" unconditionally --
        # the same slot-is-wavelength assumption the figure itself carried, restated as provenance.
        "mean_image_note": (
            f"Channel {1 - _functional_slot(args.label)} shown as 415 nm (isosbestic), "
            f"channel {_functional_slot(args.label)} as 470 nm (functional), from this session's "
            f"preprocess.svd.functional_channel."),
    }
    (args.output / f"{args.label}_extra_spout_position_overlay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
