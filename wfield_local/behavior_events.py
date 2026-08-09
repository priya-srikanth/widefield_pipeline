"""Canonical DAQ behavior-event segmentation — identify events ONCE, reuse everywhere.

Every analysis that needs licks / rewards / running / quiet / grooming should LOAD the arrays this
module produces, rather than re-detecting from the raw DAQ. One session -> one ``*_events.npz`` on the
DAQ sample clock (5000 Hz), so behavior figures and imaging analyses share an identical event identity.

What it computes (all on the DAQ clock, from ``DAQ_recorder_output/<date>/<animal>_<date>_*.h5``):

* **licks** — :func:`wfield_local.lick_detection.detect_licks` on ``lick_analog`` with the config
  ``lick_detection`` criteria (thresholds + lockout + the ``min_ili_ms`` physiological floor). This is
  the SAME call the imaging lick-aligned maps make, so behavior and imaging identify the same licks.
* **rewards** — rising edges of the ``reward_ttl`` line.
* **running bouts** — :func:`wfield_local.treadmill.find_running_bouts` on the calibrated treadmill.
* **quiet periods** — slow-treadmill AND not-near-lick/reward, buffered (the F0-baseline mask, identical
  logic to ``quiet_periods.py``).
* **grooming** — single-spout long-contact proxy (contact longer than a lick). EXPERIMENTAL and OFF by
  default: this rig has one spout, so there is no bilateral conjunction to confirm grooming, and
  pre-stroke sessions show ~0 long contacts (durations are unimodal ~65-95 ms). Enable post-stroke once
  validated.

Params live in ``configs/defaults.yaml`` (``segmentation`` + ``lick_detection``) — single source, shared
with ``quiet_periods.py``.

    python -m wfield_local.behavior_events 20260806               # all animals for a date
    python -m wfield_local.behavior_events 20260806 --only PS92   # one animal
    python -m wfield_local.behavior_events 20260806 --force       # recompute even if the .npz exists
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.lick_detection import detect_licks
from wfield_local.quiet_periods import (
    _rising,
    _runs_at_least,
    idx2bool,
    set_short_bool_to_low,
    widen_bool_sparse,
)
from wfield_local.treadmill import bout_edges, calibrate_treadmill, find_running_bouts, smooth_treadmill

SCHEMA_VERSION = 1


def _read_analog(f, name: str) -> np.ndarray:
    names = [s.decode() for s in f["analog/channel_names"][:]]
    i = names.index(name)
    if "samples_int16" in f["analog"]:
        sc = float(f["analog/int16_scale_volts_per_count"][i])
        of = float(f["analog/int16_offset_volts"][i])
        return f["analog/samples_int16"][:, i].astype(np.float32) * sc + of
    return np.asarray(f["analog/samples"][:, i], dtype=np.float32)


def compute_events(h5_path: Path, seg: dict | None = None, lick: dict | None = None) -> dict:
    """Detect all behavior events from one DAQ ``.h5``. Returns a dict of sample-indexed arrays + meta."""
    import h5py

    seg = seg if seg is not None else config.defaults()["segmentation"]
    lick = lick if lick is not None else config.defaults()["lick_detection"]
    with h5py.File(h5_path, "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        lick_v = _read_analog(f, lick.get("channel", "lick_analog"))
        tread_v = _read_analog(f, seg["treadmill"]["channel"])
        reward_v = _read_analog(f, seg["reward"]["channel"])
    n = int(lick_v.size)

    # licks (config criteria incl. the min_ili physiological floor) — identical to imaging's call
    det = detect_licks(lick_v, fs, thresh_upper=lick["thresh_upper"], thresh_lower=lick["thresh_lower"],
                       lockout_s=tuple(lick["lockout_falling_edge_s"]),
                       min_ili_s=lick.get("min_ili_ms", 0) / 1000.0)
    lick_onsets = np.asarray(det["lick_onsets"], dtype=np.int64)
    reward_samples = np.asarray(_rising(reward_v, seg["reward"]["thresh_v"]), dtype=np.int64)

    # running bouts (clearly moving)
    tw = seg["treadmill"]
    speed = smooth_treadmill(
        calibrate_treadmill(tread_v, tw["offset_v"], tw["volt_sec_per_rot"], tw["mm_per_rot"]),
        fs, tw["smoothing_sigma_s"])
    rn = seg["running"]
    running = find_running_bouts(speed, fs, rn["thresh_speed_mm_s"], rn["max_gap_s"], rn["min_duration_s"])
    run_starts, run_stops = bout_edges(running)

    # quiet = slow AND not-near-(running/lick/reward), buffered  (same as quiet_periods.py)
    q = seg["quiet"]

    def wid(b, buf):
        return widen_bool_sparse(b, int(buf[0] * fs), int(buf[1] * fs))

    slow = speed < q["speed_mm_s"]
    quiet = (~wid(~slow, q["treadmill_buffer_s"])
             & ~wid(idx2bool(lick_onsets, n), q["lick_buffer_s"])
             & ~wid(idx2bool(reward_samples, n), q["reward_buffer_s"]))

    # grooming (single-spout long contact) — experimental, off by default
    gr = seg["grooming"]
    groom_starts = groom_stops = np.empty(0, dtype=np.int64)
    if gr.get("enabled", False):
        contact = lick_v < gr["contact_thresh_v"]
        groom_bool = _runs_at_least(contact, int(gr["max_contact_s"] * fs))
        groom_starts, groom_stops = bout_edges(groom_bool)
        quiet = quiet & ~wid(groom_bool, gr["buffer_s"])
    quiet = set_short_bool_to_low(quiet, int(q["min_quiet_s"] * fs))
    quiet_starts, quiet_stops = bout_edges(quiet)

    return {
        "schema_version": SCHEMA_VERSION, "daq_h5": h5_path.name, "fs": fs, "n_samples": n,
        "lick_onsets": lick_onsets, "reward_samples": reward_samples,
        "running_starts": run_starts, "running_stops": run_stops,
        "quiet_starts": quiet_starts, "quiet_stops": quiet_stops,
        "grooming_starts": groom_starts, "grooming_stops": groom_stops,
        "params": json.dumps({"segmentation": seg, "lick_detection": lick}),
    }


def events_path(rv, animal: str, date: str) -> Path:
    """Canonical per-session events file: ``behavior_out/events/<animal>/<date>.npz``."""
    return Path(rv.root("behavior_out")) / "events" / animal / f"{date}.npz"


def save_events(events: dict, path: Path) -> Path:
    from wfield_local import writeguard
    writeguard.assert_writable(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **events)
    return path


def load_events(path: Path) -> dict | None:
    """Load an events ``.npz`` as a plain dict (arrays + scalars). None if missing."""
    if not Path(path).exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {}
        for k in z.files:
            v = z[k]
            out[k] = v.item() if v.ndim == 0 else v
        return out


def get_or_compute(rv, animal: str, date: str, force: bool = False) -> dict | None:
    """Load the canonical events for a session, computing + saving them if absent. None if no DAQ .h5.

    THIS is the entry point every consumer should call to get canonical licks/rewards/quiet/etc."""
    from wfield_local.spout_behavior import _daq_h5_for
    p = events_path(rv, animal, date)
    if not force:
        cached = load_events(p)
        if cached is not None:
            return cached
    h5 = _daq_h5_for(rv, animal, date)
    if h5 is None:
        return None
    ev = compute_events(h5)
    save_events(ev, p)
    return ev


def lick_onsets_s(events: dict) -> np.ndarray:
    """Lick onset times in seconds from an events dict."""
    return np.asarray(events["lick_onsets"], dtype=float) / float(events["fs"])


def run(date, rv, animals=None, force=False) -> int:
    """Produce (and save) canonical event arrays for every animal with a DAQ ``.h5`` on ``date``."""
    from wfield_local.spout_behavior import _daq_h5_for
    animals = animals or [a for a in config.animals()]
    made = 0
    for a in animals:
        h5 = _daq_h5_for(rv, a, date)
        if h5 is None:
            continue
        ev = get_or_compute(rv, a, date, force=force)
        if ev is not None:
            p = events_path(rv, a, date)
            print(f"[behavior_events] {a} {date}: {ev['lick_onsets'].size} licks, "
                  f"{ev['reward_samples'].size} rewards, {ev['running_starts'].size} running bouts, "
                  f"{ev['quiet_starts'].size} quiet periods -> {p.name}", flush=True)
            made += 1
    print(f"[behavior_events] {made} session(s) written for {date}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="YYYYMMDD")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals, or 'all'")
    ap.add_argument("--force", action="store_true", help="recompute even if the .npz exists")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    from wfield_local.paths import PathResolver
    return run(args.date, PathResolver(machine=args.machine),
               animals=config.normalize_animals(args.only), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
