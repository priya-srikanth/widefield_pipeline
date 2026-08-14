"""Widefield SVD activity maps for QUIET vs RUNNING behavioral states.

Reconstructs hemodynamic-corrected activity (SVTcorr @ U) averaged over the animal's quiet periods
and running bouts, and their contrast. The quiet/running periods come from the CANONICAL behavior
events (:mod:`wfield_local.behavior_events`) — the same licks/reward/running/quiet identity the whole
pipeline shares — so this map is consistent with the behavior figures rather than re-detecting movement.

Corrected imaging frames are mapped to DAQ samples via the cleanpairs ``frame_map`` (same mapping the
cue/lick maps use); a frame is "quiet"/"running" if its DAQ sample falls in a quiet/running bout. Emits
a 3-panel figure (quiet, running, running−quiet) + maps ``.npz`` + summary, for the preprocessing deck.

    python -m wfield_local.plot_running_activity_maps --label PS92_0806_affine8v1 \
        --wfield-results <mc>/wfield_local_results --allen-dir <...>/allen_aligned_affine8v1 \
        --events <behavior_summary>/events/PS92/20260806.npz --daq-h5 <...>.h5 \
        --frame-map <...>_cleanpairs_frame_map.npz --cleanpairs-summary <...>_summary.json \
        --output <mc>/running_activity_affine8v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local import config

from wfield_local.atlas_overlay import region_edges as _region_edges
from wfield_local.behavior_events import load_events
from wfield_local.framemap_event_maps import _corrected_frame_samples, _offset_from_summary


def _decode_analog_channel(f, channel_name: str) -> np.ndarray:
    names = [name.decode() for name in f["analog/channel_names"][:]]
    idx = names.index(channel_name)
    if "samples_int16" in f["analog"]:
        raw = f["analog/samples_int16"][:, idx]
        return raw.astype(np.float32) * float(f["analog/int16_scale_volts_per_count"][idx]) \
            + float(f["analog/int16_offset_volts"][idx])
    return np.asarray(f["analog/samples"][:, idx], dtype=np.float32)


def _pco_samples(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        names = [n.decode() for n in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
        bits = np.unpackbits(packed[:, None], axis=1, bitorder="little")[:, : len(names)]
    return np.flatnonzero(np.diff(bits[:, names.index("pco_exposure")].astype(np.int8), prepend=0) == 1)


def _overlay_regions(ax, edges: np.ndarray) -> None:
    overlay = np.zeros((*edges.shape, 4), dtype=np.float32)
    overlay[edges] = (0, 0, 0, 0.65)
    ax.imshow(overlay, interpolation="nearest")


def _weighted_map(U: np.ndarray, svt_mean: np.ndarray) -> np.ndarray:
    return np.tensordot(U, svt_mean, axes=([2], [0])).astype(np.float32)


def _mask_from_edges(starts, stops, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in zip(np.asarray(starts, np.int64), np.asarray(stops, np.int64)):
        m[s:e] = True
    return m


def _display_limit(arrays, percentile: float) -> float:
    vals = np.concatenate([a.ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    return max(float(np.nanpercentile(np.abs(vals), percentile)), 1e-6) if vals.size else 1e-6


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Quiet vs running widefield SVD activity maps.")
    p.add_argument("--label", required=True)
    p.add_argument("--events", type=Path, required=True, help="behavior_events <animal>/<date>.npz")
    p.add_argument("--wfield-results", type=Path, required=True)
    p.add_argument("--allen-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--daq-h5", type=Path, required=True, help="for pco_exposure -> corrected-frame samples")
    p.add_argument("--frame-map", type=Path, default=None, help="cleanpairs frame_map (regime B); "
                   "without it, raw//2 pairing is used (regime A)")
    p.add_argument("--cleanpairs-summary", type=Path, default=None, help="for the exposure offset")
    p.add_argument("--offset", type=int, default=None)
    p.add_argument("--percentile", type=float, default=99.0)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    ev = load_events(args.events)
    if ev is None:
        raise SystemExit(f"[running_activity] no behavior events at {args.events} "
                         f"(run behavior_events first)")
    n = int(ev["n_samples"])
    fs = float(ev["fs"])
    running = _mask_from_edges(ev["running_starts"], ev["running_stops"], n)
    quiet = _mask_from_edges(ev["quiet_starts"], ev["quiet_stops"], n)

    U = np.load(args.allen_dir / "U_atlas.npy", mmap_mode="r")
    SVTcorr = np.load(config.svtcorr_in(args.wfield_results), mmap_mode="r")
    edges = _region_edges(np.load(args.allen_dir / "allen_area_atlas_native_grid.npy"))
    T = SVTcorr.shape[1]

    pco = _pco_samples(args.daq_h5)
    if args.frame_map is not None:
        offset = args.offset if args.offset is not None else _offset_from_summary(args.cleanpairs_summary)
        csample = _corrected_frame_samples(args.frame_map, pco, offset)   # DAQ sample per corrected frame
        regime = "B(frame-map)"
    else:
        csample = pco[np.clip(2 * np.arange(T), 0, pco.size - 1)]
        offset, regime = 0, "A(raw//2)"
    csample = csample[:T] if csample.size >= T else np.pad(csample, (0, T - csample.size))
    cs = np.clip(csample, 0, n - 1)
    frame_running = running[cs]
    frame_quiet = quiet[cs]

    def _state_map(mask):
        return _weighted_map(U, np.asarray(SVTcorr[:, mask]).mean(axis=1)) if mask.sum() else None

    quiet_map = _state_map(frame_quiet)
    running_map = _state_map(frame_running)
    contrast = (running_map - quiet_map) if (quiet_map is not None and running_map is not None) else None

    summary = {
        "label": args.label, "events": str(args.events), "wfield_results": str(args.wfield_results),
        "svtcorr": str(config.svtcorr_in(args.wfield_results)),
        "allen_dir": str(args.allen_dir), "frame_map": str(args.frame_map), "regime": regime,
        "offset": int(offset), "fs": fs, "svt_frames": int(T),
        "quiet_frames": int(frame_quiet.sum()), "running_frames": int(frame_running.sum()),
        "n_running_bouts": int(np.asarray(ev["running_starts"]).size),
        "n_quiet_periods": int(np.asarray(ev["quiet_starts"]).size),
    }
    if quiet_map is None or running_map is None:
        summary["skipped"] = True
        summary["skip_reason"] = f"need quiet AND running frames; got quiet={int(frame_quiet.sum())}, " \
                                 f"running={int(frame_running.sum())}"
        (args.output / f"{args.label}_quiet_running_activity_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(f"[running_activity] {args.label} SKIPPED: {summary['skip_reason']}", flush=True)
        return 0

    lim = _display_limit([quiet_map, running_map, contrast], args.percentile)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    panels = [("quiet", quiet_map, int(frame_quiet.sum()), lim, "RdBu_r"),
              ("running", running_map, int(frame_running.sum()), lim, "RdBu_r"),
              ("running − quiet", contrast, int(frame_running.sum()), lim, "RdBu_r")]
    im = None
    for ax, (title, arr, ncnt, vlim, cmap) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=-vlim, vmax=vlim)
        _overlay_regions(ax, edges)
        ax.set_axis_off()
        ax.set_title(f"{title}\nn={ncnt} frames")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01, label=f"ΔF/F (±{lim:.4g})")
    fig.suptitle(f"{args.label}: SVD activity by behavioral state (quiet vs running)", fontsize=14)
    png = args.output / f"{args.label}_quiet_running_activity_maps.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output / f"{args.label}_quiet_running_activity_maps.npz",
                        quiet_map=quiet_map, running_map=running_map, running_minus_quiet=contrast,
                        frame_quiet=frame_quiet, frame_running=frame_running)
    summary["display_limit"] = lim
    summary["outputs"] = {"png": str(png)}
    (args.output / f"{args.label}_quiet_running_activity_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[running_activity] wrote {png.name} "
          f"(quiet={int(frame_quiet.sum())}, running={int(frame_running.sum())} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
