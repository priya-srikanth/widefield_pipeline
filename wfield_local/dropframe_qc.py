"""Dropped-frame QC for the Blackfly/Bonsai behavior-camera CSVs (folds in the local
``dropframe_check_all.py``, now in-repo).

Each Bonsai CSV is three unlabeled columns — ``frame_id, timestamp_ns, gpio`` — one row per frame
(4 cameras per session, ~250 fps ⇒ ~4.003 ms/frame). A dropped frame shows up as a gap in the
monotonic ``frame_id`` sequence (and/or an over-long timestamp delta). For each cam recording this
reports rows, id-span, dropped count/%, gap events, and the timestamp-delta stats, and writes
``dropped_frames_summary_<DATE>.csv`` (machine-readable, one row per cam) + ``.txt`` (human table)
next to the data.

Layout (both machines): ``<behavior_cameras>/<YYYYMMDD>/<PSxx>/cam{1..4}_<ISO>.csv``. On the analysis
box the same tree also lives at ``D:\\camera`` before upload, so ``--root`` can point there directly.

CLI::

    python -m wfield_local.dropframe_qc 20260807               # MICROSCOPE behavior_cameras/<date>
    python -m wfield_local.dropframe_qc 20260807 --root D:/camera   # pre-upload staging (flat <PSxx>/ dirs)
    python -m wfield_local.dropframe_qc 20260807 --output <dir>     # write the summary elsewhere
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import writeguard

CAM_RE = re.compile(r"(cam\d+)_(.+)\.csv$", re.I)
ANIMAL_RE = re.compile(r"^PS\d+$")

# Machine-readable summary columns (kept byte-compatible with the prior dropframe_check_all output).
CSV_COLUMNS = ["date", "session", "cam", "recording", "rows", "id_span", "dropped", "drop_pct",
               "gap_events", "max_gap_frames", "mean_dt_ms", "max_dt_ms", "ts_gaps_gt_1p5x",
               "dur_s", "fps"]


def _read_id_ts(path) -> np.ndarray:
    """(N, 2) int64 array of (frame_id, timestamp_ns) — the gpio column is not needed for drop QC."""
    return pd.read_csv(path, header=None, usecols=[0, 1], dtype="int64").to_numpy()


def analyze_csv(path) -> dict:
    """Per-recording drop stats for one Bonsai cam CSV (cam/recording parsed from the filename)."""
    name = Path(path).name
    m = CAM_RE.search(name)
    cam = m.group(1).lower() if m else "?"
    recording = m.group(2) if m else "?"
    arr = _read_id_ts(path)
    ids, ts = arr[:, 0], arr[:, 1]
    rows = int(len(ids))
    id_span = int(ids[-1] - ids[0] + 1) if rows else 0
    did = np.diff(ids)
    gap_events = int((did > 1).sum())
    max_gap_frames = int((did - 1).max()) if rows > 1 and gap_events else 0
    dt_ms = np.diff(ts) / 1e6
    med = float(np.median(dt_ms)) if dt_ms.size else 0.0
    dur_s = round(float((ts[-1] - ts[0]) / 1e9), 1) if rows > 1 else 0.0
    return dict(
        cam=cam, recording=recording, rows=rows, id_span=id_span,
        dropped=id_span - rows,
        drop_pct=round(100 * (id_span - rows) / id_span, 1) if id_span else 0.0,
        gap_events=gap_events, max_gap_frames=max_gap_frames,
        mean_dt_ms=round(float(dt_ms.mean()), 4) if dt_ms.size else 0.0,
        max_dt_ms=round(float(dt_ms.max()), 3) if dt_ms.size else 0.0,
        ts_gaps_gt_1p5x=int((dt_ms > 1.5 * med).sum()) if med else 0,
        dur_s=dur_s,
        fps=round(rows / dur_s, 2) if dur_s else 0.0,
    )


def scan(session_root, date) -> list[dict]:
    """Analyze every ``<PSxx>/cam*.csv`` under ``session_root``; rows ordered by (animal, cam)."""
    root = Path(session_root)
    out = []
    for animal_dir in sorted(p for p in root.iterdir() if p.is_dir() and ANIMAL_RE.match(p.name)):
        for csv in sorted(animal_dir.glob("cam*.csv")):
            out.append(dict(date=date, session=animal_dir.name, **analyze_csv(csv)))
    return out


def _csv_text(rows) -> str:
    lines = [",".join(CSV_COLUMNS)]
    lines += [",".join(str(r[c]) for c in CSV_COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def _txt_text(rows, date) -> str:
    hdr = (f"{'session':<8}  {'cam':<4}  {'recording':<24} {'rows':>9} {'dropped':>8} {'drop%':>7} "
           f"{'gaps':>5} {'maxgap':>7} {'mean_ms':>8} {'max_ms':>8} {'dur_s':>9} {'fps':>7}")
    out = [f"Dropped-frame QC - {date}  ({len(rows)} cam recordings)", hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['session']:<8}  {r['cam']:<4}  {r['recording']:<24} {r['rows']:>9} "
                   f"{r['dropped']:>8} {r['drop_pct']:>6}% {r['gap_events']:>5} {r['max_gap_frames']:>7} "
                   f"{r['mean_dt_ms']:>8} {r['max_dt_ms']:>8} {r['dur_s']:>9} {r['fps']:>7}")
    total_dropped = sum(r["dropped"] for r in rows)
    bad = [r for r in rows if r["dropped"]]
    if total_dropped == 0:
        out += ["", "RESULT: No dropped frames (all frame-id sequences contiguous)."]
    else:
        out += ["", f"RESULT: {total_dropped} dropped frames across {len(bad)} recording(s) "
                    f"(rows with dropped>0)."]
    return "\n".join(out) + "\n"


def write_summary(rows, out_dir, date) -> tuple[Path, Path]:
    """Write ``dropped_frames_summary_<date>.{csv,txt}`` into ``out_dir`` (guarded)."""
    out = Path(out_dir)
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"dropped_frames_summary_{date}.csv"
    txt_path = out / f"dropped_frames_summary_{date}.txt"
    csv_path.write_text(_csv_text(rows), encoding="utf-8")
    txt_path.write_text(_txt_text(rows, date), encoding="utf-8")
    return csv_path, txt_path


def run(session_root, date, out_dir=None) -> list[dict]:
    """Scan + write the summary; returns the per-cam rows. ``out_dir`` defaults to ``session_root``."""
    rows = scan(session_root, date)
    if not rows:
        print(f"[dropframe_qc] no <PSxx>/cam*.csv found under {session_root}", flush=True)
        return rows
    csv_path, txt_path = write_summary(rows, out_dir or session_root, date)
    total = sum(r["dropped"] for r in rows)
    print(f"[dropframe_qc] {len(rows)} cam recordings, {total} dropped frame(s) -> {csv_path}", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="YYYYMMDD", help="recording date")
    ap.add_argument("--root", default=None,
                    help="dir holding the <PSxx>/cam*.csv session dirs (default: behavior_cameras/<date>)")
    ap.add_argument("--output", default=None, help="summary output dir (default: the session root)")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    if args.root:
        session_root = args.root
    else:
        from wfield_local.paths import PathResolver
        session_root = PathResolver(machine=args.machine).resolve("behavior_cameras", args.date)
    run(session_root, args.date, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
