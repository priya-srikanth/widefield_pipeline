"""Relabel/trim a labcams DAT to DAQ-confirmed illuminated 415/470 frame pairs.

labcams writes camera frames to the .dat in raw arrival order; the 415/470
"channels" are only an interpretation imposed by the ``_N_H_W_`` filename at read
time (frame i -> channel i % N). When acquisition is trial-gated, the LED phase
and the running frame counter can drift apart between trials (a trial with an odd
frame count flips which wavelength lands in "channel 0"), so a naive reshape mixes
415 and 470 across trials. Continuously-saved-but-LED-gated sessions additionally
contain dark inter-trial frames.

This module uses the DAQ ``pco_exposure`` / ``led415_ttl`` / ``led470_ttl``
channels as ground truth to label every saved physical frame, drop dark frames,
keep clean adjacent 415/470 pairs, and write a standard two-channel labcams DAT
(channel 0 = 415, channel 1 = 470) plus a frame map and summary. This is the
canonical step to run BEFORE motion correction and SVD on trial-gated data.

Modes:
  rescue          - continuously-saved sessions (expects many dark inter-trial
                    frames; drops them). Default.
  acquire-enable  - PCO Acquire-Enable gated sessions (expects ~no dark frames;
                    warns if many are found, which would indicate a gating fault).

WHEN THE DAT HAS MORE FRAMES THAN THE DAQ HAS EXPOSURE PULSES (2026-09-28). The two recorders are
started and stopped by hand, so the camera can be running before the DAQ starts (frames with no
pulse at the HEAD) or after it stops (at the TAIL). Which end the excess sits at cannot be told
from the counts, and it decides the mapping of EVERY frame: 1a68c7c (2026-09-22) assumed the tail
and was wrong on the first session it ran on -- PS92_0922, where the DAQ was started 2.466 s after
the camera. The result was every trial window reading 154 frames too early and, because 154 is even
but the LED alternation has ~150 hiccups per session, 4.6% of frames landing in the wrong channel
with the wrong channel's mean subtracted. That looked like a whole-cortex anti-phase vascular
oscillation and took a day to diagnose (docs/EXPERIMENT_ERRORS.md, 2026-09-22).

The camlog labcams writes beside the DAT records a per-frame LED id (`#LED:<id>,...` lines) that
is independent of the DAQ and shares the DAQ's alternation hiccups. Aligning the two sequences
gives the head offset EXACTLY and doubles as a check on every session (agreement must be ~1.0).
So: a DAT longer than the DAQ record is placed by the camlog, or by an explicit `--head-offset`,
and is otherwise REFUSED -- the pre-1a68c7c behaviour, restored with a way out. Guessing is the
one thing this step must never do again.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np


DAT_RE = re.compile(r"_(?P<nchan>\d+)_(?P<h>\d+)_(?P<w>\d+)_(?P<dtype>uint16|int16|uint8|float32|float64)\.dat$")


def parse_labcams_dat_name(path: Path) -> tuple[int, int, int, np.dtype]:
    match = DAT_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse labcams DAT shape from filename: {path.name}")
    nchan = int(match.group("nchan"))
    height = int(match.group("h"))
    width = int(match.group("w"))
    dtype = np.dtype(match.group("dtype"))
    return nchan, height, width, dtype


def rising_falling_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    rises = np.flatnonzero((~mask[:-1]) & mask[1:]) + 1
    falls = np.flatnonzero(mask[:-1] & (~mask[1:])) + 1
    if mask[0]:
        rises = np.r_[0, rises]
    if mask[-1]:
        falls = np.r_[falls, len(mask)]
    if len(falls) and len(rises) and falls[0] < rises[0]:
        falls = falls[1:]
    if len(rises) and len(falls) and rises[-1] > falls[-1]:
        rises = rises[:-1]
    n = min(len(rises), len(falls))
    return rises[:n], falls[:n]


def analog_ttl_mask(volts: np.ndarray, fixed_threshold: float | None) -> tuple[np.ndarray, float, float, float]:
    lo, hi = np.percentile(volts, [1, 99.9])
    if fixed_threshold is None:
        threshold = lo + 0.5 * (hi - lo)
        if hi - lo < 0.5:
            threshold = 2.5
    else:
        threshold = fixed_threshold
    return volts > threshold, float(threshold), float(lo), float(hi)


def camlog_path_for(dat: Path) -> Path | None:
    """The labcams camlog beside a DAT: ``<stem without _N_H_W_dtype>.camlog``, or None if absent."""
    dat = Path(dat)
    m = DAT_RE.search(dat.name)
    stem = dat.name[:m.start()] if m else dat.stem
    p = dat.parent / f"{stem}.camlog"
    return p if p.is_file() else None


def camlog_led_ids(camlog: Path) -> np.ndarray:
    """Per-frame LED id from the camlog's ``#LED:<id>,<n>,<t_ms>`` lines, in write order.

    labcams emits one such line per frame; the ids are the two LED channels (5 and 6 on this rig).
    Which id is 415 and which 470 is NOT assumed -- the alignment below is polarity-free.
    """
    out = []
    with open(camlog, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#LED:"):
                out.append(int(line[5:].split(",", 1)[0]))
    return np.asarray(out, dtype=np.int64)


def camlog_agreement(cam_ids: np.ndarray, daq_labels: np.ndarray, head: int, n_compare: int = 150_000) -> float:
    """Fraction of the first ``n_compare`` DAQ pulses whose 415/470 label matches the camera's LED id
    when DAT frame ``j`` is taken to be pulse ``j - head``. Polarity-free (which LED id is 415 is not
    assumed); unlabelled pulses (dark / both) are ignored. ~1.0 at the right offset on every session."""
    cam = np.asarray(cam_ids)
    if cam.size == 0:
        raise ValueError("camlog carries no #LED lines")
    cam_bin = (cam == cam[0]).astype(np.int8)
    daq_bin = np.where(daq_labels == 415, 0, np.where(daq_labels == 470, 1, -1)).astype(np.int8)
    a = cam_bin[head:head + n_compare]
    b = daq_bin[:len(a)]
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    a, b = a[:m], b[:m]
    ok = b >= 0
    if not ok.any():
        return 0.0
    g = float((a[ok] == b[ok]).mean())
    return max(g, 1.0 - g)


def find_dat_head_offset(cam_ids: np.ndarray, daq_labels: np.ndarray, max_offset: int,
                         n_compare: int = 150_000) -> tuple[int, float, float]:
    """Head offset ``h`` such that DAT frame ``j`` was exposed on DAQ pulse ``j - h``.

    Returns ``(h, agreement_at_h, agreement_at_0)`` -- see `camlog_agreement`. The PS92_0922
    signature was 0.954 at 0 and 1.0000 at 154, with 0.9997 at 153 and 155: the ±1 offsets mislabel
    exactly the ~150 alternation-hiccup frames, so the discrimination is real but thin, and a
    mid-session camlog defect inside the comparison window (PS93_0904 has one) can defeat it. The
    caller therefore only trusts this where the DAT is longer than the DAQ record.
    """
    scores = [camlog_agreement(cam_ids, daq_labels, h, n_compare) for h in range(0, int(max_offset) + 1)]
    best = int(np.argmax(scores))
    return best, scores[best], scores[0]


#: Below this the camlog and the DAQ disagree on which LED lit which frame, and the mapping is wrong.
CAMLOG_AGREEMENT_MIN = 0.999


def load_daq_labels(
    h5_path: Path,
    physical_frame_count: int,
    offset: int | None,
    led_threshold: float | None,
    camlog: Path | None = None,
    head_offset: int | None = None,
) -> tuple[np.ndarray, dict]:
    """One label (415 / 470 / 3=both / 0=dark or unmonitored) per DAT PHYSICAL FRAME.

    ``offset`` is the pulse-side offset (a stray leading DAQ pulse; searched over 0/1 when None).
    ``head_offset`` is the frame-side offset: DAT frames written BEFORE the DAQ began recording.
    When the DAT is longer than the DAQ record it is taken from ``head_offset`` if given, else read
    off the camlog (see `find_dat_head_offset`), else the call REFUSES. Frames before the head and
    past the DAQ's coverage are labelled 0 and drop out of the pairing; their counts are reported in
    ``meta['dat_head_dropped_before_daq']`` / ``['dat_tail_dropped_beyond_daq']``.
    """
    with h5py.File(h5_path, "r") as h5:
        fs = float(h5.attrs["sample_rate_hz"])
        digital_names = [x.decode() for x in h5["digital/channel_names"][()]]
        analog_names = [x.decode() for x in h5["analog/channel_names"][()]]
        packed = h5["digital/packed_samples"][()][:, 0]
        pco_bit = digital_names.index("pco_exposure")
        pco = ((packed >> pco_bit) & 1).astype(bool)

        samples_i16 = h5["analog/samples_int16"][()].astype(np.float32)
        scale = h5["analog/int16_scale_volts_per_count"][()]
        zero = h5["analog/int16_offset_volts"][()]
        volts = samples_i16 * scale + zero
        led415 = volts[:, analog_names.index("led415_ttl")]
        led470 = volts[:, analog_names.index("led470_ttl")]

    pco_rise, pco_fall = rising_falling_edges(pco)
    b415, thr415, lo415, hi415 = analog_ttl_mask(led415, led_threshold)
    b470, thr470, lo470, hi470 = analog_ttl_mask(led470, led_threshold)

    labels_all = np.zeros(len(pco_rise), dtype=np.int16)
    for i, (start, stop) in enumerate(zip(pco_rise, pco_fall)):
        window = slice(max(0, start - 1), min(len(pco), stop + 1))
        has415 = bool(b415[window].any())
        has470 = bool(b470[window].any())
        if has415 and has470:
            labels_all[i] = 3
        elif has415:
            labels_all[i] = 415
        elif has470:
            labels_all[i] = 470

    # Pulse-side offset: a stray leading DAQ pulse with no frame. Scored on the label sequence
    # itself, which is why it CANNOT see a frame-side (head) offset -- that is handled below.
    P = int(physical_frame_count)
    candidate_offsets = [offset] if offset is not None else [0, 1]
    best = None
    for off in candidate_offsets:
        if off < 0 or off >= len(labels_all):
            continue
        cov = labels_all[off:off + P]
        illum = cov[cov != 0]
        same_adjacent = int(np.sum(illum[1:] == illum[:-1])) if len(illum) > 1 else 0
        both = int(np.sum(illum == 3))
        score = (len(illum), -same_adjacent, -both)
        if best is None or score > best[0]:
            best = (score, off)
    if best is None:
        raise ValueError("No valid DAQ exposure-label offset for DAT frame count")
    chosen_offset = best[1]
    avail = len(labels_all) - chosen_offset            # pulses the DAQ can offer from this offset
    excess = P - avail                                 # DAT frames the DAQ never saw (>0), or spare pulses (<0)

    # Frame-side offset: DAT frames written before the DAQ began. Placed by the camlog or by hand;
    # never guessed. See the module docstring for the session that made this rule.
    cam_agree_h = cam_agree_0 = None
    fmt = (lambda v: "n/a" if v is None else f"{v:.4f}")
    if head_offset is not None:
        head = int(head_offset)
        if camlog is not None:                          # report, do not decide: the operator decided
            ids = camlog_led_ids(camlog)
            cam_agree_h = camlog_agreement(ids, labels_all[chosen_offset:], head)
            cam_agree_0 = camlog_agreement(ids, labels_all[chosen_offset:], 0)
    elif camlog is not None:
        head, cam_agree_h, cam_agree_0 = find_dat_head_offset(camlog_led_ids(camlog), labels_all[chosen_offset:], max(excess, 0) + 2)
        if excess <= 0:
            # The DAQ covers every frame, so the DAQ labels are authoritative as they always were and the
            # camlog is only a CHECK. It is not a reliable one on every session -- measured 2026-09-28:
            # 10 of 120 camlogs agree with the DAQ at chance (the trial-gated early-June sessions, the
            # lesion day, and PS93_0904), presumably labcams not writing one #LED line per frame -- so
            # disagreement here WARNS and never refuses.
            if head != 0 or cam_agree_0 < CAMLOG_AGREEMENT_MIN:
                print(f"[relabel] WARNING: camlog LED record agrees with the DAQ labels on {100*cam_agree_0:.2f}% of "
                      f"frames at offset 0 (best {head}: {100*cam_agree_h:.2f}%). The DAQ covers the whole DAT so the "
                      f"mapping stands at 0; the camlog is unreliable on this session.", flush=True)
            head = 0
        elif cam_agree_h < CAMLOG_AGREEMENT_MIN:
            raise ValueError(f"DAT has {excess} more frame(s) than the DAQ recorded, and the camlog agrees with the DAQ "
                             f"labels on only {100*cam_agree_h:.2f}% of frames at its best head offset ({head}) -- it "
                             f"cannot place them. Pass --head-offset once you have established it another way.")
    elif excess > 0:
        raise ValueError(f"DAT has {excess} more frame(s) than the DAQ recorded exposure pulses for, and there is no "
                         f"camlog to say whether they precede or follow the DAQ record. Guessing the tail put PS92_0922 "
                         f"2.466 s out of register (docs/EXPERIMENT_ERRORS.md). Pass --camlog or --head-offset.")
    else:
        head = 0

    n_cov = max(0, min(avail, P - head))
    labels = np.zeros(P, dtype=np.int16)               # per DAT physical frame; 0 = never illuminated / unmonitored
    labels[head:head + n_cov] = labels_all[chosen_offset:chosen_offset + n_cov]
    dat_tail_dropped = int(P - head - n_cov)
    covered = labels[head:head + n_cov]

    if head > 0:
        print(f"[relabel] WARNING: DAT has {head} frame(s) ({head / 62.5:.2f} s at 62.5 Hz) written BEFORE the DAQ "
              f"recorded its first exposure pulse -- the DAQ was started after the camera. Frame j <-> pulse "
              f"j - {head}. Camlog agreement at this offset {fmt(cam_agree_h)} (at 0: {fmt(cam_agree_0)}).",
              flush=True)
    if dat_tail_dropped > 0:
        print(f"[relabel] DAT has {dat_tail_dropped} frame(s) beyond the DAQ exposure record "
              f"(camera kept writing after the DAQ stopped); dropping the unmonitored tail. "
              f"Normal sessions have 0.", flush=True)

    meta = {
        "sample_rate_hz": fs,
        "daq_pco_exposure_count": int(len(labels_all)),
        "dat_physical_frame_count": P,
        "dat_head_dropped_before_daq": int(head),
        "dat_tail_dropped_beyond_daq": dat_tail_dropped,
        "camlog_used": camlog is not None,
        "camlog_agreement_at_head": (None if cam_agree_h is None else float(cam_agree_h)),
        "camlog_agreement_at_zero": (None if cam_agree_0 is None else float(cam_agree_0)),
        "chosen_exposure_offset": int(chosen_offset),
        "led415_threshold_v": thr415,
        "led470_threshold_v": thr470,
        "led415_p1_v": lo415,
        "led415_p999_v": hi415,
        "led470_p1_v": lo470,
        "led470_p999_v": hi470,
        # counted over the DAQ-COVERED frames only, so head/tail frames do not read as 'dark'
        "labels_415": int(np.sum(covered == 415)),
        "labels_470": int(np.sum(covered == 470)),
        "labels_both": int(np.sum(covered == 3)),
        "labels_dark": int(np.sum(covered == 0)),
    }
    return labels, meta


def make_clean_pairs(labels: np.ndarray, order: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    illuminated = np.flatnonzero(labels != 0)
    pairs: list[tuple[int, int]] = []
    pair_labels: list[tuple[int, int]] = []
    skipped: list[int] = []
    i = 0
    while i + 1 < len(illuminated):
        a = int(illuminated[i])
        b = int(illuminated[i + 1])
        la = int(labels[a])
        lb = int(labels[b])
        if (la, lb) in ((415, 470), (470, 415)):
            if order == "415-470":
                pair = (a, b) if (la, lb) == (415, 470) else (b, a)
                lab = (415, 470)
            elif order == "470-415":
                pair = (a, b) if (la, lb) == (470, 415) else (b, a)
                lab = (470, 415)
            else:
                pair = (a, b)
                lab = (la, lb)
            pairs.append(pair)
            pair_labels.append(lab)
            i += 2
        else:
            skipped.append(a)
            i += 1
    if i < len(illuminated):
        skipped.append(int(illuminated[i]))
    return np.asarray(pairs, dtype=np.int64), np.asarray(pair_labels, dtype=np.int16), np.asarray(skipped, dtype=np.int64)


def write_trimmed_dat(
    source: Path,
    output: Path,
    pairs: np.ndarray,
    height: int,
    width: int,
    dtype: np.dtype,
    chunk_pairs: int,
) -> None:
    src = np.memmap(source, mode="r", dtype=dtype, shape=(source.stat().st_size // (height * width * dtype.itemsize), height, width))
    out = np.memmap(output, mode="w+", dtype=dtype, shape=(len(pairs), 2, height, width))
    for start in range(0, len(pairs), chunk_pairs):
        stop = min(start + chunk_pairs, len(pairs))
        idx = pairs[start:stop]
        out[start:stop, 0] = src[idx[:, 0]]
        out[start:stop, 1] = src[idx[:, 1]]
        out.flush()
        print(f"wrote pairs {stop}/{len(pairs)}", flush=True)
    del out
    del src


def relabel_dat_from_daq(
    dat: Path,
    daq_h5: Path,
    output_dir: Path,
    label: str = "illuminated",
    offset: int | None = None,
    led_threshold: float | None = None,
    order: str = "415-470",
    chunk_pairs: int = 256,
    mode: str = "rescue",
    map_only: bool = False,
    camlog: Path | None = None,
    head_offset: int | None = None,
) -> dict:
    """Relabel a labcams DAT to DAQ-confirmed 415/470 pairs and write a new DAT.

    Returns a summary dict; ``summary['output_dat']`` is the relabeled .dat path,
    suitable as input to motion correction. Importable for pipeline use.

    ``map_only=True`` computes the pairs + writes the frame_map (.npz/.csv) and summary but
    SKIPS materializing the (raw-sized) cleanpairs .dat — the caller applies the mapping on the
    fly during motion correction (see run_wfield_motion.RelabeledDat). ``output_dat`` is then None.

    ``camlog`` defaults to the labcams camlog beside the DAT when one exists; it places any frames
    the DAQ never saw (see `load_daq_labels`). ``head_offset`` overrides it.
    """
    dat = Path(dat)
    daq_h5 = Path(daq_h5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nchan, height, width, dtype = parse_labcams_dat_name(dat)
    if nchan != 2:
        raise ValueError(f"Expected a two-channel labcams DAT filename, got nchan={nchan}")
    frame_bytes = height * width * dtype.itemsize
    physical_frames = dat.stat().st_size // frame_bytes
    if physical_frames * frame_bytes != dat.stat().st_size:
        raise ValueError("DAT size is not an integer number of physical frames")

    if camlog is None:
        camlog = camlog_path_for(dat)
    labels, meta = load_daq_labels(daq_h5, int(physical_frames), offset, led_threshold,
                                   camlog=camlog, head_offset=head_offset)
    head = int(meta["dat_head_dropped_before_daq"])
    dark_frac = meta["labels_dark"] / max(int(physical_frames), 1)
    if mode == "acquire-enable" and dark_frac > 0.05:
        print(f"WARNING [acquire-enable mode]: {meta['labels_dark']} dark frames "
              f"({100*dark_frac:.1f}%) found; acquire-enable gating should produce ~none. "
              f"Check that the camera only acquired during trials.", flush=True)

    pairs, pair_labels, skipped = make_clean_pairs(labels, order)
    out_dat = output_dir / f"{dat.stem}_{label}_cleanpairs_2_{height}_{width}_{dtype.name}.dat"
    map_npz = output_dir / f"{dat.stem}_{label}_cleanpairs_frame_map.npz"
    map_csv = output_dir / f"{dat.stem}_{label}_cleanpairs_frame_map.csv"
    summary_path = output_dir / f"{dat.stem}_{label}_cleanpairs_summary.json"

    print(f"[relabel:{mode}] source physical frames: {physical_frames:,}", flush=True)
    print(f"[relabel:{mode}] clean pairs: {len(pairs):,}; skipped singleton/problem frames: {len(skipped):,}; "
          f"dark dropped: {meta['labels_dark']:,}", flush=True)
    if map_only:
        print(f"[relabel:{mode}] map-only: skipping cleanpairs .dat write "
              f"(applied on the fly during motion correction)", flush=True)
    else:
        print(f"[relabel:{mode}] output: {out_dat}", flush=True)
        write_trimmed_dat(dat, out_dat, pairs, height, width, dtype, chunk_pairs)

    # `original_frame_index_*` are DAT frame indices (what the frame GATHER needs).
    # `daq_pulse_index_*` are the DAQ exposure pulses those frames were exposed on: frame - head +
    # pulse-offset. They differ only on a session with a head offset, and consumers that turn a
    # frame into a DAQ time must use the pulse index (framemap_event_maps._corrected_frame_samples).
    np.savez_compressed(
        map_npz,
        pair_index=np.arange(len(pairs), dtype=np.int64),
        original_frame_index_ch0=pairs[:, 0],
        original_frame_index_ch1=pairs[:, 1],
        daq_pulse_index_ch0=pairs[:, 0] - head + int(meta["chosen_exposure_offset"]),
        daq_pulse_index_ch1=pairs[:, 1] - head + int(meta["chosen_exposure_offset"]),
        dat_head_offset=np.int64(head),
        channel_label_ch0=pair_labels[:, 0],
        channel_label_ch1=pair_labels[:, 1],
        labels_per_original_frame=labels,
        skipped_original_frame_index=skipped,
    )
    with map_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_index", "original_frame_index_ch0", "channel_label_ch0", "original_frame_index_ch1", "channel_label_ch1"])
        for i, ((a, b), (la, lb)) in enumerate(zip(pairs, pair_labels)):
            writer.writerow([i, int(a), int(la), int(b), int(lb)])

    summary = {
        **meta,
        "mode": mode,
        "source_dat": str(dat),
        "daq_h5": str(daq_h5),
        "output_dat": (None if map_only else str(out_dat)),
        "frame_map_npz": str(map_npz),
        "frame_map_csv": str(map_csv),
        "output_shape": [int(len(pairs)), 2, int(height), int(width)],
        "output_dtype": dtype.name,
        "channel_order": order,
        "clean_pairs": int(len(pairs)),
        "skipped_illuminated_frames": int(len(skipped)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel labcams DAT to DAQ-confirmed 415/470 pairs.")
    parser.add_argument("dat", type=Path)
    parser.add_argument("daq_h5", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="illuminated")
    parser.add_argument("--mode", choices=("rescue", "acquire-enable"), default="rescue",
                        help="rescue: drop dark inter-trial frames (continuous saving). "
                             "acquire-enable: warn if dark frames found (gated acquisition).")
    parser.add_argument("--offset", type=int, default=None, help="DAQ pco_exposure offset relative to DAT frames. Default chooses 0/1 automatically.")
    parser.add_argument("--led-threshold", type=float, default=None)
    parser.add_argument("--order", choices=("415-470", "470-415", "as-acquired"), default="415-470")
    parser.add_argument("--chunk-pairs", type=int, default=256)
    parser.add_argument("--map-only", action="store_true",
                        help="compute pairs + write frame_map/summary but SKIP the raw-sized "
                             "cleanpairs .dat (applied on the fly during motion correction).")
    parser.add_argument("--camlog", type=Path, default=None,
                        help="labcams camlog with per-frame #LED lines (default: the one beside the DAT). "
                             "Places DAT frames the DAQ never recorded; a DAT longer than the DAQ record "
                             "is REFUSED without it or --head-offset.")
    parser.add_argument("--head-offset", type=int, default=None,
                        help="DAT frames written before the DAQ began (frame j <-> pulse j - head). "
                             "Overrides the camlog. PS92_0922 needs 154.")
    args = parser.parse_args()
    relabel_dat_from_daq(
        args.dat, args.daq_h5, args.output_dir,
        label=args.label, offset=args.offset, led_threshold=args.led_threshold,
        order=args.order, chunk_pairs=args.chunk_pairs, mode=args.mode, map_only=args.map_only,
        camlog=args.camlog, head_offset=args.head_offset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
