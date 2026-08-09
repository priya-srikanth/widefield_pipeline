"""Blackfly behavior-camera <-> DAQ temporal alignment templates (B5).

The behavior Arduino emits one irregular sync pulse train (bounded-random ITI ~0.25-0.67 s, each pulse
~100 ms = ~25 cam frames / ~500 DAQ samples wide) that lands BOTH on the DAQ digital ``sync`` line
(``port0/line0``, bit0 of ``digital/packed_samples``, 5000 Hz) AND on every Blackfly's GPIO (LSB of the
Bonsai CSV's 3rd column). So each free-running camera aligns to the DAQ session clock by matching its
GPIO rising-edge train to the DAQ sync rising-edge train; the per-cam-per-date template then maps camera
TIME <-> DAQ time (for post-stroke multi-angle DLC and behavior<->imaging alignment). The PCO imaging
camera needs no template (already on the DAQ clock via ``pco_exposure``).

Matcher: the proven bounded-window ITI-fingerprint from :func:`wfield_local.frame_sync.align_edge_sequences`
(faithful port of stroke_orofacial; O(N·window), NOT the O(N^2) all-pairs form). Mapping is built from the
camera's absolute per-frame TIMESTAMPS (not row indices), so a dropped frame in an ITI just removes an
anchor without shifting the time axis — the map rides through it. Detection of drops is separate and
reported here: ``n_frame_drops`` (gaps in the monotonic ``frame_id``, as in :mod:`wfield_local.dropframe_qc`),
the DAQ/cam edge-count delta, matched-anchor count, and the fit residual. Templates are COMPACT (affine +
matched edges, not a dense 25M-sample lookup); map on demand via :func:`cam_seconds_to_daq_seconds`.

CLI::

    python -m wfield_local.camera_sync 20260807                 # all animals/cams for the date
    python -m wfield_local.camera_sync 20260807 --only PS94     # one animal
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from wfield_local import writeguard
from wfield_local.frame_sync import align_edge_sequences, _norm_to_01

CAM_RE = re.compile(r"(cam\d+)_(.+)\.csv$", re.I)
ANIMAL_RE = re.compile(r"(PS\d+)")

# Quality gate (an improvement over orofacial, which has no residual check): a good alignment is
# sub-frame here (~1-2 ms); a genuine misalignment jumps to ~ITI scale (100s of ms).
RESID_MS_MAX = 10.0
MIN_MATCH_FRAC = 0.5


# --------------------------------------------------------------------------- edge extraction
def daq_sync_edges(h5_path, sync_name: str = "sync") -> tuple[np.ndarray, float, int]:
    """Rising-edge sample indices of the DAQ digital ``sync`` line, its rate, and total sample count."""
    with h5py.File(h5_path, "r") as h:
        names = [n.decode() for n in h["digital/channel_names"][:]]
        if sync_name not in names:
            raise ValueError(f"{h5_path}: no digital channel {sync_name!r} (have {names})")
        bit = names.index(sync_name)
        packed = h["digital/packed_samples"][:, 0]
        fs = float(h.attrs["sample_rate_hz"])
    line = ((packed >> bit) & 1).astype(np.int8)
    edges = np.flatnonzero(np.diff(line) > 0) + 1
    return edges, fs, int(packed.shape[0])


def _read_cam(csv_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame_id, timestamp_ns, gpio) columns of a Bonsai cam CSV (headerless, 3 int columns)."""
    a = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2], dtype="int64").to_numpy()
    return a[:, 0], a[:, 1], a[:, 2]


# --------------------------------------------------------------------------- template
def build_template(h5_path, csv_path, sync_name: str = "sync", sync_bit: int = 0,
                   window: int = 20, p: float = 0.1, min_matched: int = 5) -> dict:
    """Build a compact cam<->DAQ alignment template from a DAQ .h5 + a Bonsai cam CSV."""
    m = CAM_RE.search(Path(csv_path).name)
    cam, recording = (m.group(1).lower(), m.group(2)) if m else ("?", "?")

    daq_edge, fs, n_daq = daq_sync_edges(h5_path, sync_name)
    fid, ts_ns, gpio = _read_cam(csv_path)
    n_cam = int(fid.size)
    n_frame_drops = int((fid[-1] - fid[0] + 1) - n_cam)                 # frame_id contiguity (see dropframe_qc)
    cam_edge = np.flatnonzero(np.diff(((gpio >> sync_bit) & 1).astype(np.int8)) > 0) + 1
    cam_edge_t = ts_ns[cam_edge].astype(np.float64) / 1e9              # absolute camera clock (s)
    daq_edge_t = daq_edge.astype(np.float64) / fs

    # proven bounded-window ITI match on [0,1]-normalized edge TIMES (cam=s1, daq=s2)
    ci, di, dist = align_edge_sequences(_norm_to_01(cam_edge_t), _norm_to_01(daq_edge_t), window, p)
    if ci.size < min_matched:
        raise ValueError(f"{cam} {recording}: only {ci.size} matched sync edges (daq={daq_edge.size} "
                         f"cam={cam_edge.size}) — cannot align")
    mcf, mds = cam_edge[ci], daq_edge[di]                              # matched cam-frame / daq-sample idx
    mct, mdt = cam_edge_t[ci], daq_edge_t[di]                          # matched cam-time / daq-time (s)

    slope_t, inter_t = np.polyfit(mct, mdt, 1)                          # cam seconds -> daq seconds
    slope_s, inter_s = np.polyfit(mcf.astype(np.float64), mds.astype(np.float64), 1)   # cam frame -> daq sample
    resid_ms = (mdt - (slope_t * mct + inter_t)) * 1e3
    rms = float(np.sqrt((resid_ms ** 2).mean()))
    quality_ok = bool(rms < RESID_MS_MAX and ci.size >= MIN_MATCH_FRAC * min(cam_edge.size, daq_edge.size))

    return dict(
        cam=cam, recording=recording, daq_h5=Path(h5_path).name,
        fs_daq=fs, fps_cam=float(fs / slope_s) if slope_s else np.nan,
        n_cam_frames=n_cam, n_daq_samples=n_daq, n_frame_drops=n_frame_drops,
        n_daq_edges=int(daq_edge.size), n_cam_edges=int(cam_edge.size), n_matched=int(ci.size),
        slope_daqSample_per_camFrame=float(slope_s), intercept_daqSample=float(inter_s),
        slope_daqSec_per_camSec=float(slope_t), intercept_daqSec=float(inter_t),
        resid_ms_max=float(np.abs(resid_ms).max()), resid_ms_rms=rms, quality_ok=quality_ok,
        matched_cam_edge_frame=mcf.astype(np.int64), matched_daq_edge_sample=mds.astype(np.int64),
        matched_cam_edge_sec=mct, matched_daq_edge_sec=mdt,
    )


def cam_seconds_to_daq_seconds(template: dict, cam_seconds) -> np.ndarray:
    """Map camera timestamps (s) -> DAQ time (s) via the fitted affine.

    Time-based (not frame-index) on purpose: a dropped frame removes an anchor but never shifts the
    absolute-timestamp axis, so this stays correct across gaps (a frame-index map would renumber).
    The affine (vs piecewise interp over the matched edges) holds globally — the clocks share a rate
    to ~µs/s (``resid_ms_rms`` ~1-2 ms), and it extrapolates cleanly to the recording ends where the
    matcher's ±window margin leaves the first/last ~``window`` edges unmatched. ``quality_ok`` gates
    whether the affine is trustworthy (a clock glitch would inflate the residual and clear the flag).
    """
    return (template["slope_daqSec_per_camSec"] * np.asarray(cam_seconds, dtype=np.float64)
            + template["intercept_daqSec"])


def save_template(template: dict, path) -> Path:
    """Save a template as ``.npz`` (writeguard-checked)."""
    path = Path(path)
    writeguard.assert_writable(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **template)
    return path


def template_path(csv_path) -> Path:
    """Default location: next to the cam CSV, ``<cam>_<recording>_daq_alignment.npz``."""
    p = Path(csv_path)
    return p.with_name(p.stem + "_daq_alignment.npz")


# --------------------------------------------------------------------------- CLI / batch
def _match_daq(daq_root, animal, date) -> str | None:
    hits = sorted(h for h in glob.glob(f"{daq_root}/**/*.h5", recursive=True)
                  if animal in Path(h).name and date in Path(h).name)
    return hits[0] if hits else None


def run(date, rv, animals=None, verbose=True) -> list[dict]:
    """Build + save a template for every ``<PSxx>/cam*.csv`` on ``date`` (the PCO cam needs none)."""
    cam_root = rv.resolve("behavior_cameras", date)
    daq_root = rv.root("daq_recorder_output")
    made = []
    for animal_dir in sorted(p for p in Path(cam_root).iterdir() if p.is_dir() and ANIMAL_RE.fullmatch(p.name)):
        if animals and animal_dir.name not in set(animals):
            continue
        daq = _match_daq(daq_root, animal_dir.name, date)
        if not daq:
            print(f"[camera_sync] {animal_dir.name}: no DAQ .h5 for {date} -> skip", flush=True)
            continue
        for csv in sorted(animal_dir.glob("cam*.csv")):
            try:
                t = build_template(daq, csv)
                out = save_template(t, template_path(csv))
                made.append(t)
                flag = "" if t["quality_ok"] else "  <<< QUALITY CHECK FAILED"
                if verbose:
                    print(f"[camera_sync] {animal_dir.name} {t['cam']}: matched {t['n_matched']}/"
                          f"{t['n_cam_edges']} edges, resid rms={t['resid_ms_rms']:.2f}ms, "
                          f"frame_drops={t['n_frame_drops']} -> {out.name}{flag}", flush=True)
            except Exception as e:
                print(f"[camera_sync] {animal_dir.name} {csv.name}: FAILED {type(e).__name__}: {e}", flush=True)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="YYYYMMDD")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals, or 'all'")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    from wfield_local import config
    from wfield_local.paths import PathResolver
    run(args.date, PathResolver(machine=args.machine), animals=config.normalize_animals(args.only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
