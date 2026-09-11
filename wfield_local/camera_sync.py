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


def _drop_trim_params() -> tuple[float, float, float]:
    from wfield_local import config
    d = ((config.defaults().get("sync") or {}).get("drop_trim")) or {}
    return (float(d.get("bin_s", 10.0)), float(d.get("max_drop_pct", 1.0)),
            float(d.get("min_clean_frac", 0.5)))


def clean_edge_mask(fid, ts_ns, edge_frame_idx, bin_s=None, max_drop_pct=None,
                    min_clean_frac=None) -> tuple[np.ndarray, float]:
    """Which sync edges sit in LOW-DROP parts of the recording. Returns ``(mask, clean_frac)``.

    **A camera that drops frames also drops GPIO sync EDGES**, and a missed edge does not merely
    remove an anchor -- it makes the bounded-window ITI matcher pair the wrong two pulses, because
    the fingerprint it matches on is the sequence of intervals. Those bad anchors are then fitted
    with the good ones, and one affine has to split the difference.

    Measured on PS92 2026-09-08: clean for 104 of 122 minutes, and still 6.4 / 7.7 / 9.7 / 13.6 ms
    residual across the four cameras against the 1.15 ms its untroubled sibling PS93 managed the same
    morning. cam4 failed ``quality_ok`` outright. Fitting on the clean region is not a repair of the
    broken tail -- the frames there are genuinely gone -- it stops the tail from corrupting the 86% of
    the session that is perfect.

    Drops are localised by binning ``frame_id`` discontinuities in TIME rather than by row, so the
    bins mean the same thing whatever the drop rate. **Below ``min_clean_frac`` nothing is trimmed**:
    a short lever arm yields a confidently wrong slope, and a mostly-broken recording should fail the
    quality gate rather than be rescued into looking fine.
    """
    b, maxpct, minfrac = _drop_trim_params()
    bin_s = b if bin_s is None else bin_s
    max_drop_pct = maxpct if max_drop_pct is None else max_drop_pct
    min_clean_frac = minfrac if min_clean_frac is None else min_clean_frac

    fid = np.asarray(fid, dtype=np.int64)
    t = (np.asarray(ts_ns, dtype=np.float64) - float(ts_ns[0])) / 1e9
    lost = np.diff(fid) - 1                      # frames missing before each retained frame
    lost = np.clip(lost, 0, None).astype(np.float64)
    b_idx = (t[1:] / bin_s).astype(np.int64)
    n_bins = int(b_idx.max()) + 1 if b_idx.size else 1
    lost_per = np.bincount(b_idx, weights=lost, minlength=n_bins)
    kept_per = np.bincount(b_idx, minlength=n_bins).astype(np.float64)
    pct = 100.0 * lost_per / np.maximum(lost_per + kept_per, 1.0)
    bad_bin = pct > max_drop_pct

    edge_bin = np.clip((t[np.asarray(edge_frame_idx, dtype=np.int64)] / bin_s).astype(np.int64),
                       0, n_bins - 1)
    mask = ~bad_bin[edge_bin]
    frac = float(mask.mean()) if mask.size else 1.0
    if frac < min_clean_frac:
        # Not enough clean anchors to trust a trimmed fit. Keep everything and let quality_ok speak.
        return np.ones_like(mask, dtype=bool), frac
    return mask, frac


def robust_fit_mask(mct, mdt, seed_mask, outlier_ms=None, max_iter=8,
                    min_clean_frac=None) -> np.ndarray:
    """Drop anchors whose residual says they are mis-PAIRED, wherever in the recording they sit.

    The drop-rate mask above catches anchors inside a frame-dropping stretch. It is not sufficient,
    because **a missing GPIO edge re-pairs the matcher globally**: the ITI fingerprint is a sequence,
    so losing one pulse can shift the pairing for edges far from the damage. On a synthetic 50%
    mid-session dropout every matched anchor sits in a clean bin and the fit still lands at 214 ms.

    So: fit on the drop-clean anchors, discard anchors more than ``outlier_ms`` from that line, refit,
    repeat. Two clocks sharing a rate to ~µs/s put every honest anchor within ~1-2 ms of the line, and
    a mis-pairing lands an ITI away -- hundreds of ms. The separation is not subtle, which is why a
    fixed threshold is safe here and a percentile would not be.

    **Only ever called when frames were dropped**, so a clean recording's fit is untouched.
    """
    _b, _m, minfrac = _drop_trim_params()
    min_clean_frac = minfrac if min_clean_frac is None else min_clean_frac
    if outlier_ms is None:
        from wfield_local import config
        outlier_ms = float((((config.defaults().get("sync") or {}).get("drop_trim")) or {})
                           .get("resid_outlier_ms", 3.0))

    keep = np.asarray(seed_mask, dtype=bool).copy()
    n0 = int(keep.sum())
    for _ in range(max_iter):
        if keep.sum() < 5:
            break
        slope, inter = np.polyfit(mct[keep], mdt[keep], 1)
        resid_ms = np.abs(mdt - (slope * mct + inter)) * 1e3
        # THE THRESHOLD HAS TO SURVIVE THE CONTAMINATION IT IS MEASURING. A least-squares line pulled
        # by one gross outlier leaves EVERY honest anchor above a fixed 3 ms bar -- 40 anchors and one
        # of them an ITI out puts the rest at ~11 ms -- so a fixed cut would reject the whole set and
        # the guard below would give up. Scaling by the median absolute residual makes the cut loose
        # while the fit is bad and tighten to the fixed floor once it is clean.
        cut = max(outlier_ms, 5.0 * float(np.median(resid_ms[keep])))
        nxt = keep & (resid_ms <= cut)
        if nxt.sum() < max(5, min_clean_frac * n0):
            break                       # refusing to whittle the fit down to a short lever arm
        if nxt.sum() == keep.sum():
            break
        keep = nxt
    return keep


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

    # Anchors from frame-dropping stretches are excluded from the FIT but kept in the file: a missed
    # GPIO edge makes the ITI matcher pair the wrong pulses, and those pairs would otherwise drag the
    # single affine across the whole recording. No-op when nothing was dropped.
    fit = np.ones(ci.size, dtype=bool)
    clean_frac = 1.0
    if n_frame_drops:
        fit, clean_frac = clean_edge_mask(fid, ts_ns, mcf)
        fit = robust_fit_mask(mct, mdt, fit)
    if fit.sum() < min_matched:                    # never trim below what the matcher itself demands
        fit = np.ones(ci.size, dtype=bool)

    slope_t, inter_t = np.polyfit(mct[fit], mdt[fit], 1)                # cam seconds -> daq seconds
    slope_s, inter_s = np.polyfit(mcf[fit].astype(np.float64),          # cam frame -> daq sample
                                  mds[fit].astype(np.float64), 1)
    resid_ms = (mdt - (slope_t * mct + inter_t)) * 1e3                  # over EVERY matched edge
    rms = float(np.sqrt((resid_ms[fit] ** 2).mean()))                   # ... but scored on the fit set
    quality_ok = bool(rms < RESID_MS_MAX and ci.size >= MIN_MATCH_FRAC * min(cam_edge.size, daq_edge.size))

    return dict(
        cam=cam, recording=recording, daq_h5=Path(h5_path).name,
        fs_daq=fs, fps_cam=float(fs / slope_s) if slope_s else np.nan,
        n_cam_frames=n_cam, n_daq_samples=n_daq, n_frame_drops=n_frame_drops,
        n_daq_edges=int(daq_edge.size), n_cam_edges=int(cam_edge.size), n_matched=int(ci.size),
        slope_daqSample_per_camFrame=float(slope_s), intercept_daqSample=float(inter_s),
        slope_daqSec_per_camSec=float(slope_t), intercept_daqSec=float(inter_t),
        resid_ms_max=float(np.abs(resid_ms[fit]).max()), resid_ms_rms=rms, quality_ok=quality_ok,
        # PROVENANCE of the trim, so a reader can tell a clean recording from a rescued one.
        n_fit_edges=int(fit.sum()), drop_trim_applied=bool(not fit.all()),
        clean_edge_frac=float(clean_frac),
        resid_ms_rms_all=float(np.sqrt((resid_ms ** 2).mean())),
        fit_cam_sec_lo=float(mct[fit].min()), fit_cam_sec_hi=float(mct[fit].max()),
        matched_in_fit=fit,
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


def template_path(rv, date: str, animal: str, cam: str) -> Path:
    """Dedicated tree (keeps the raw camera dirs clean; mirrors stroke_orofacial's convention):
    ``<alignment_templates>/<cam>/<PSxx>/<YYYYMMDD>.npz``. The recording timestamp is kept inside
    the template (``recording`` key), so one file per cam per animal per date is unambiguous."""
    return Path(rv.root("alignment_templates")) / cam / animal / f"{date}.npz"


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
                out = save_template(t, template_path(rv, date, animal_dir.name, t["cam"]))
                made.append(t)
                flag = "" if t["quality_ok"] else "  <<< QUALITY CHECK FAILED"
                if verbose:
                    print(f"[camera_sync] {animal_dir.name} {t['cam']}: matched {t['n_matched']}/"
                          f"{t['n_cam_edges']} edges, resid rms={t['resid_ms_rms']:.2f}ms, "
                          f"frame_drops={t['n_frame_drops']}"
                          + (f", TRIMMED to {t['n_fit_edges']}/{t['n_matched']} clean anchors "
                             f"(was {t['resid_ms_rms_all']:.2f}ms over all)"
                             if t.get("drop_trim_applied") else "")
                          + f" -> {out.name}{flag}", flush=True)
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
