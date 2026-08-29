"""Config-driven nightly widefield preprocessing (IMAGING box).

Replaces the retired per-date ``_nightly_<DATE>.py`` / ``_mc_svd_*`` / ``_maps_*`` /
``_photobleach_*`` one-off drivers. For a given ``YYYYMMDD`` it:

  1. discovers that date's raw sessions on the imaging box's local drive (``raw_labcams`` =
     ``E:/labcams_data/<date>``, ``raw_daq`` = ``E:/DAQ_recorder_output``) — no per-date
     hard-coding of session dir names, DAQ filenames, or frame dims,
  2. per session, in animal order: sign-fixed motion correction -> SVD -> cross-register to
     that animal's reference (``preprocess.reference_date``, default 6/6; reference session
     derived from ``sessions.yaml`` + per-animal ``reference_landmarks``) -> push the LocaNMF
     inputs to MICROSCOPE (``labcams`` = ``N:`` on the imaging box) FIRST so the GPU box can
     start LocaNMF on early sessions,
  3. runs photobleaching QC (:mod:`wfield_local.photobleach`) over the same sessions.

Run in the ``wfield`` env on the imaging box (the interpreter running this module is reused
for every sub-step, so no hard-coded python path)::

    python -m wfield_local.preprocess 20260808
    python -m wfield_local.preprocess 0808                       # MMDD also accepted (cohort year assumed)
    python -m wfield_local.preprocess 0806 0807                  # a list (comma or space)
    python -m wfield_local.preprocess 0806-0808                  # an inclusive range
    python -m wfield_local.preprocess all                        # every date-dir on the raw drive
    python -m wfield_local.preprocess 20260808 --dry-run         # print the plan only
    python -m wfield_local.preprocess 20260808 --only PS94 PS95  # subset of animals (or --only all)
    python -m wfield_local.preprocess 20260807 --skip-preprocess --only PS92
                                                                 # re-run only downstream steps
                                                                 # (maps/xall) on already-processed
                                                                 # sessions (raw archived off E:)

The date grammar (list / range / ``all`` / either width) and ``--only`` are shared verbatim with the
analysis CLI (:mod:`wfield_local.nightly_figs`); see :func:`wfield_local.config.expand_dates`.

All heavy steps run as ``python -m wfield_local.<engine>`` subprocesses (run_wfield_motion,
run_wfield_local, cross_day_align), exactly as the retired drivers did — this module imports
no wfield code, so it loads in any env.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from wfield_local import config
from wfield_local import photobleach
from wfield_local import writeguard
from wfield_local.paths import PathResolver

REPO = Path(__file__).resolve().parents[1]
DAT_RE = re.compile(r"pco_edge_run\d+_\d+_(2_\d+_\d+)_uint16\.dat$", re.I)
# Same name shape but ANY channel count, so a raw that labcams mis-saved as single-channel (1_H_W) --
# a normal alternating 415/470 session whose movie header wrote the wrong channel count -- is WARNED
# about, not silently dropped by discovery (PS92 8/28). Groups: (channels, H, W).
DAT_ANY_RE = re.compile(r"pco_edge_run\d+_\d+_(\d+)_(\d+)_(\d+)_uint16\.dat$", re.I)
ANIMAL_RE = re.compile(r"(PS9\d)")


# --------------------------------------------------------------------------- discovery
def _session_dir_of(dat: Path, yyyymmdd: str) -> Path:
    """Nearest ancestor of a raw .dat that names the session (``<ANIMAL>_<YYYYMMDD>_*``)."""
    for p in dat.parents:
        if yyyymmdd in p.name and ANIMAL_RE.search(p.name):
            return p
    for p in dat.parents:                      # fallback: parent of raw_widefield_data/
        if p.name.lower() == "raw_widefield_data":
            return p.parent
    return dat.parent


def _match_daq(h5s: list[Path], animal: str | None, yyyymmdd: str) -> Path | None:
    """DAQ .h5 whose NAME contains the animal + the 8-digit date (robust to date-dir typos).

    A ``*_concat.h5`` WINS when one exists. When an acquisition crashes mid-session the day is
    recorded as two segments plus a joined one, so three .h5 match the animal and date, and plain
    sorting returns the FIRST SEGMENT -- for PS92 8/12 that is 24 trials of a 225-trial day. The joined
    file is the authoritative record of such a day (see the sessions.yaml note on PS92 "0812"), and
    picking it here keeps the discovery path consistent with what sessions.yaml already declares.
    """
    if not animal:
        return None
    cands = sorted(h for h in h5s if animal in h.name and yyyymmdd in h.name)
    concat = [h for h in cands if h.stem.endswith("_concat")]
    return (concat or cands)[0] if cands else None


def _discover(yyyymmdd: str, raw_root: str, daq_root: str) -> list[dict]:
    """Core discovery (roots passed explicitly so it is unit-testable off the imaging box)."""
    datedir = Path(raw_root) / yyyymmdd
    if not datedir.exists():
        return []
    all_dats = [Path(p) for p in glob.glob(str(datedir / "**" / "*_uint16.dat"), recursive=True)]
    dats = [p for p in all_dats if DAT_RE.search(p.name)]
    for p in all_dats:                        # never silently drop a raw we can see (2026-08-28)
        if DAT_RE.search(p.name):
            continue
        m = DAT_ANY_RE.search(p.name)
        if m and m.group(1) != "2":
            ch, h, w = m.groups()
            print(f"[discover] WARNING: {p.name} is labelled {ch}-channel, not 2, and was SKIPPED. "
                  f"A normal 415/470 session that labcams mis-saved as single-channel looks exactly "
                  f"like this; if its DAQ shows both LEDs alternating, rename the '_{ch}_{h}_{w}_' "
                  f"segment to '_2_{h}_{w}_' to rescue it. [{p}]", flush=True)
        else:
            print(f"[discover] WARNING: unrecognized raw name {p.name} was SKIPPED. [{p}]", flush=True)
    h5s = [Path(p) for p in glob.glob(str(Path(daq_root) / "**" / "*.h5"), recursive=True)]

    by_sess: dict[Path, list[Path]] = {}
    for dat in dats:
        by_sess.setdefault(_session_dir_of(dat, yyyymmdd), []).append(dat)

    out = []
    for sd, group in by_sess.items():
        dat = max(group, key=lambda p: p.stat().st_size)     # largest run = the main movie
        dims = DAT_RE.search(dat.name).group(1)
        am = ANIMAL_RE.search(sd.name) or ANIMAL_RE.search(str(dat))
        animal = am.group(1) if am else None
        daq = _match_daq(h5s, animal, yyyymmdd)
        out.append(dict(animal=animal, mmdd=yyyymmdd[4:], sess=sd.name, sess_dir=str(sd),
                        raw_dat=str(dat), daq_h5=(str(daq) if daq else None),
                        dims=dims, n_dats=len(group)))
    out.sort(key=lambda s: (s["animal"] or "", s["sess"]))
    return out


def _blank_frame(f, i: int, single: int) -> bool:
    """Is frame ``i`` all zeros? ``count(0)`` so the scan happens in C, not a Python loop."""
    f.seek(i * single)
    b = f.read(single)
    return len(b) == single and b.count(0) == single


def raw_integrity(dat_path, dims: str, settle_s: float = 4.0) -> dict:
    """Is this raw ``.dat`` whole, and has it stopped growing?

    WHY, and what NOT to check. On 2026-08-20 I called PS94's upload truncated because the file size
    was not a multiple of a 415/470 PAIR. It was wrong twice over. The ``.dat`` is a sequence of
    SINGLE ``h x w`` uint16 frames -- the ``2`` in ``2_460_480`` is the channel pairing, not the
    file's atomic unit -- and a recording that stops on an odd frame leaves a half pair, which is
    normal. The file was also still being written, so the size I measured was a partial flush 256 B
    past a frame boundary. Both mistakes point the same way: measure against the right block, and
    make sure the thing has stopped moving before judging it.

    HARD failure: the size is not a whole number of single frames, OR the last frame is all zeros.
    Neither can happen to a complete file, so refusing costs nothing and saves a multi-hour run on a
    partial upload.

    WHY THE LAST FRAME IS CHECKED AT ALL. The size test alone is not enough, and on 2026-08-20 it
    was not: PS92 and PS95 arrived with byte counts that divided exactly, matched their camlogs to
    the frame, and had stopped growing hours earlier -- and were 53% and 93% unwritten zeros. The
    transfer preallocates the destination to its full size, so a copy that dies leaves a file whose
    every measurable property says "whole". Both preprocessed to completion and produced maps whose
    brightness was simply scaled by the fraction that made it (PS92 frames_average mean 6295 against
    ~17000; PS95 994), which no downstream step flagged.

    An all-zero frame is unambiguous: an sCMOS writes its dark offset on every pixel even with the
    LED off, so 220,800 exact zeros is unwritten file, never a dark recording. The last frame is the
    whole test -- a complete recording cannot end on one. The interior probes below only locate the
    boundary, so the message can say how much survived.

    SOFT warning: the frame count disagrees with the camlog. That invariant holds on every session
    still on the share (4 of 4 -- older raw is archived off, so the evidence is thin), which is not
    enough to justify blocking a night's processing on it.
    """
    import time

    p = Path(dat_path)
    if not p.exists():
        return {"ok": False, "reason": "missing", "frames": 0}
    nch, h, w = (int(x) for x in dims.split("_"))
    single = h * w * 2                       # ONE frame, uint16 -- not the pair
    n0 = p.stat().st_size
    if settle_s:
        time.sleep(settle_s)
    n = p.stat().st_size
    frames, rem = divmod(n, single)
    cam = p.parent / "pco_edge_run000_00000000.camlog"
    rows = None
    if cam.exists():
        with open(cam, errors="ignore") as f:
            rows = sum(1 for ln in f if ln.strip() and not ln.startswith("#"))
    out = {"bytes": n, "grew_during_check": n != n0, "frames": frames, "remainder": rem,
           "camlog_rows": rows, "pairs": frames // 2, "odd_last_pair": bool(frames % 2),
           "single_frame_bytes": single}
    out.update(last_frame_blank=None, first_blank_frame=None, written_fraction=None,
               interior_blank=[], locked=False)
    if frames > 0 and rem == 0 and not out["grew_during_check"]:
        try:
            with open(p, "rb") as f:
                out["last_frame_blank"] = _blank_frame(f, frames - 1, single)
                if out["last_frame_blank"]:
                    lo, hi = -1, 0
                    if not _blank_frame(f, 0, single):
                        lo, hi = 0, frames - 1
                        while hi - lo > 1:                       # first blank frame, in ~19 seeks
                            mid = (lo + hi) // 2
                            if _blank_frame(f, mid, single):
                                hi = mid
                            else:
                                lo = mid
                    out["first_blank_frame"] = hi
                    out["written_fraction"] = hi / frames
                else:
                    # A whole tail does not rule out a hole. One zero frame mid-recording is odd but
                    # survivable, so this warns and does not block -- unlike a blank tail, it has never
                    # been seen and is not worth refusing a night's processing over.
                    out["interior_blank"] = [i for i in
                                             (int(frames * fr) for fr in (0.1, 0.25, 0.5, 0.75, 0.9))
                                             if _blank_frame(f, i, single)]
        except OSError as ex:
            # A file that EXISTS and cannot be OPENED is an upload in progress, not a broken file.
            # Windows hands the writer an exclusive handle, so the destination of a running copy stats
            # fine (size and mtime both readable) and raises PermissionError on open. On 2026-08-21 that
            # killed the PS95 poller outright the moment its upload started -- and preprocess calls this
            # too, so a night whose raw was still arriving would have taken the whole run down with it
            # rather than skipping one session. Not readable yet is a WAIT, never a verdict.
            out["locked"] = True
            out["reason"] = (f"cannot be opened yet ({type(ex).__name__}) -- another process still "
                             f"holds it, i.e. the upload is in progress")
            out["ok"] = False
            return out
    out["ok"] = (rem == 0) and not out["grew_during_check"] and not out["last_frame_blank"]
    out["reason"] = (
        "still uploading (grew during the check)" if out["grew_during_check"]
        else f"{rem} B past a frame boundary -- partial upload" if rem
        else (f"unwritten from frame {out['first_blank_frame']} of {frames} -- only "
              f"{100 * (out['written_fraction'] or 0.0):.1f}% of the file was actually copied; "
              f"the SIZE is complete, the CONTENT is not. Re-upload from the acquisition box."
              ) if out["last_frame_blank"]
        else "")
    return out


def discover_raw_sessions(yyyymmdd: str, rv: PathResolver,
                          raw_root=None, daq_root=None) -> list[dict]:
    """Sessions for a date, from this machine's raw roots unless they are overridden.

    THE OVERRIDE EXISTS BECAUSE RAW DOES NOT ALWAYS LAND LOCALLY. The imaging box's normal flow is
    acquire to ``E:/labcams_data`` -> preprocess -> upload results to MICROSCOPE, and `raw_labcams`
    encodes that. On 2026-08-20 the night's raw arrived on MICROSCOPE with nothing on E: at all, so
    discovery found zero sessions and said so correctly -- the data was simply somewhere the pipeline
    had never been told to look. Pointing it at the shared tree is a path question, not a pipeline
    change: outputs still land in ``<session>/motion_corrected`` beside the raw, which is exactly
    where every previous night's already sit.
    """
    return _discover(yyyymmdd,
                     raw_root or rv.root("raw_labcams"),
                     daq_root or rv.root("raw_daq"))


def list_raw_dates(rv: PathResolver, raw_root=None) -> list[str]:
    """All YYYYMMDD acquisition date-dirs present on this machine's raw_labcams root (for `all`/ranges).

    Empty when the raw drive isn't mounted (e.g. a dry-run off the imaging box) — explicit date
    tokens still work in that case; only `all` / ranges need the disk to enumerate against."""
    root = Path(raw_root or rv.root("raw_labcams"))
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name))


def discover_processed_sessions(yyyymmdd: str, rv: PathResolver) -> list[dict]:
    """Discover ALREADY-PROCESSED sessions from the MICROSCOPE tree (``labcams`` = N:).

    For re-running downstream steps (maps/xall/deck) on a date whose raw ``.dat`` has been
    archived off the imaging box's E:. A session counts as processed if its ``motion_corrected``
    holds a ``*cleanpairs_frame_map.npz`` (the maps step's key input). DAQ ``.h5`` is matched
    from the MICROSCOPE DAQ root (``daq_recorder_output`` = N:), so this works after E: is cleaned.
    ``raw_dat`` is ``None`` (photobleach needs raw and is skipped for such sessions).
    """
    datedir = Path(rv.root("labcams")) / yyyymmdd
    if not datedir.exists():
        return []
    h5s = [Path(p) for p in glob.glob(str(Path(rv.root("daq_recorder_output")) / "**" / "*.h5"),
                                      recursive=True)]
    out = []
    for sd in sorted(datedir.glob(f"*{yyyymmdd}*")):
        if not (sd.is_dir() and ANIMAL_RE.search(sd.name)):
            continue
        fms = sorted((sd / "motion_corrected").glob("*cleanpairs_frame_map.npz"))
        if not fms:
            continue                                   # not processed -> skip
        m = re.search(r"_(2_\d+_\d+)_uint16", fms[0].name)
        animal = ANIMAL_RE.search(sd.name).group(1)
        daq = _match_daq(h5s, animal, yyyymmdd)
        out.append(dict(animal=animal, mmdd=yyyymmdd[4:], sess=sd.name, sess_dir=str(sd),
                        raw_dat=None, daq_h5=(str(daq) if daq else None),
                        dims=(m.group(1) if m else "?"), n_dats=0))
    out.sort(key=lambda s: (s["animal"] or "", s["sess"]))
    return out


# --------------------------------------------------------------------------- reference
def reference_for(animal: str, params: dict, rv: PathResolver) -> tuple[str, str, str]:
    """(ref_date, ref_results_dir, ref_landmarks_json) for an animal's cross-reg reference.

    The reference session directory is derived from ``sessions.yaml``'s ``reference_date``
    (6/6) entry; the landmark version comes from ``animals.yaml reference_landmarks``.
    """
    ref_date = str(params["reference_date"])
    raw = config._load("sessions.yaml")["sessions"]
    entry = (raw.get(animal) or {}).get(ref_date)
    if not entry:
        raise SystemExit(f"[preprocess] no reference {ref_date} session for {animal} in sessions.yaml")
    ref_reldir = str(entry["mc"]).rsplit("/motion_corrected", 1)[0]      # "20260606/PS92_..."
    lm = (config.animals().get(animal) or {}).get("reference_landmarks")
    if not lm:
        raise SystemExit(f"[preprocess] no reference_landmarks for {animal} in animals.yaml")
    results = rv.resolve("labcams", f"{ref_reldir}/motion_corrected/wfield_local_results")
    landmarks = rv.resolve("labcams", f"{ref_reldir}/raw_widefield_data/dorsal_cortex_landmarks_{lm}.json")
    return ref_date, results, landmarks


# --------------------------------------------------------------------------- xall (all-days QC)
def _sessions_registered_and_on_disk(rv: PathResolver) -> list[dict]:
    """Registered sessions (sessions.yaml) UNION every ``<YYYYMMDD>/<ANIMAL>_<date>_*/motion_corrected``
    on the labcams tree (N:), as ``{label, mc}``. Lets the all-days xall overlay include the just-
    processed night before the analysis box has registered it (registration lags a night behind)."""
    reg = config.load_sessions(machine=rv.machine)
    seen = {s["label"] for s in reg}
    root = rv.root("labcams")
    for dd in sorted(glob.glob(f"{root}/20*")):
        d = os.path.basename(dd)
        if not re.fullmatch(r"\d{8}", d):
            continue
        for sd in sorted(glob.glob(f"{dd}/*")):
            m = re.match(r"(PS\d+)_" + d, os.path.basename(sd))
            label = f"{m.group(1)}_{d[4:]}" if m else None
            if label and label not in seen and Path(f"{sd}/motion_corrected").is_dir():
                reg.append({"label": label, "mc": f"{sd}/motion_corrected".replace("\\", "/")})
                seen.add(label)
    return reg


def refresh_xall(animals: list[str], params: dict, rv: PathResolver, dry_run: bool) -> None:
    """Refresh each animal's all-days cross-day vasculature QC overlay (``xday/<animal>_xall``).

    QC-ONLY (``warp_u=False``): registers every registered task-era session (>= ``xall_min_date``)
    to the animal's reference and writes the multi-day `<animal>_cross_day_alignment_qc.png` +
    `<animal>_<date>_mean470_in_ref/ccf.npy`. Does NOT touch any session's allen_aligned dir
    (GPU LocaNMF inputs untouched). Folds in the retired `_xall_refresh.py`.
    """
    min_mmdd = str(params.get("xall_min_date", "0605"))
    all_sessions = _sessions_registered_and_on_disk(rv)
    for animal in animals:
        ref_date, ref_results, ref_landmarks = reference_for(animal, params, rv)
        sess = {}
        for s in all_sessions:
            a, mmdd = s["label"].split("_")
            if a != animal or mmdd < min_mmdd:
                continue
            results_dir = f"{s['mc']}/wfield_local_results"
            # SKIP sessions with no preprocessed outputs. Cross-day QC spans every registered date, but
            # an offloaded/in-flight session (raw staged to MICROSCOPE for the desktop, not yet
            # preprocessed) has no frames_average.npy and would crash cross_day_align mid-run, taking the
            # whole nightly's maps step down with it. (2026-08-22)
            if not Path(f"{results_dir}/frames_average.npy").exists():
                print(f"[xall] {animal}_{mmdd}: no frames_average.npy (not preprocessed yet) -> skip",
                      flush=True)
                continue
            sess[f"{animal}_{mmdd}"] = {"results": results_dir}
        if f"{animal}_{ref_date}" not in sess:
            print(f"[xall] {animal}: no reference {ref_date} in task-era set -> skip", flush=True)
            continue
        sess[f"{animal}_{ref_date}"]["landmarks"] = ref_landmarks   # reference carries landmarks
        cfg = {"animal": animal, "mode": "reference-native", "func_channel": 1,
               "reference": f"{animal}_{ref_date}", "warp_u": False,
               "output": rv.resolve("xday_qc", f"{animal}_xall"), "sessions": sess}
        cfg_path = f"{rv.root('xday_qc')}/_xall_config_{animal}.json"
        print(f"[xall] {animal}: {len(sess)} days {sorted(sess)} -> {cfg['output']}", flush=True)
        if not dry_run:
            Path(rv.root("xday_qc")).mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
        _run(["wfield_local.cross_day_align", cfg_path], dry_run)


# --------------------------------------------------------------------------- activity maps
def _registered_behavior_trials(animal, mmdd):
    """`behavior_trials` registered for this session in sessions.yaml, or None.

    Discovery builds its session dicts from the FILESYSTEM, so they never carried the config override
    and `_behavior_trials_args` always fell through to globbing the behavior_logs root. That is wrong
    for exactly the session the override exists for: PS93 8/5's GUI log is EMPTY (its positions were
    recovered from cam1), so the glob handed the maps step an unusable trials.csv and it died with
    "could not align trials.csv to DAQ cues".
    """
    try:
        for s in config.load_sessions():
            if s["label"] == f"{animal}_{mmdd}" and s.get("behavior_trials"):
                return str(s["behavior_trials"])
    except Exception:                                    # noqa: BLE001 - config must never break discovery
        return None
    return None


def _behavior_trials_args(session: dict, animal: str, yyyymmdd: str, rv: PathResolver) -> list[str]:
    """``["--behavior-trials", trials.csv]`` when a recovered CSV is discoverable, else ``[]``.

    Preference order (no per-date special-casing — ``classify_cues_with_backup`` guards this, so
    passing trials.csv on a good session is safe; it only overrides when the DAQ is broken):
      1. an explicit ``session["behavior_trials"]`` (a config-recovered CSV, e.g. PS93 8/5 cam1),
      2. glob the behavior_logs root for ``<animal>_<yyyymmdd>_*/trials.csv``.
    """
    bt = session.get("behavior_trials") or _registered_behavior_trials(animal, yyyymmdd[4:])
    if bt:
        return ["--behavior-trials", str(bt)]
    hits = sorted(glob.glob(f"{rv.root('behavior_logs')}/{animal}_{yyyymmdd}_*/trials.csv"))
    if hits:
        return ["--behavior-trials", hits[0]]
    return []


def _maps_commands(session: dict, params: dict, rv: PathResolver,
                   allow_missing: bool = False) -> list[list[str]]:
    """The 8-command cue/lick/quiet MAPS chain for one session (pure; glob/os only).

    Reproduces the retired ``_maps_behavior_run.py`` per-session chain, config-driven. All I/O
    paths are on MICROSCOPE (``labcams`` root = N: on the imaging box) because the deck reads
    figures from there. The ``*cleanpairs_frame_map.npz`` is globbed from the session's
    ``motion_corrected`` dir; if absent and ``allow_missing`` (dry-run off the imaging box), a
    ``<cleanpairs_frame_map.npz>`` placeholder stands in so a representative chain still prints —
    otherwise a :class:`SystemExit` is raised (regime B / no frame_map is unexpected on a real run).
    """
    mp = params["maps"]
    tag = mp["tag"]
    animal, mmdd, sess = session["animal"], session["mmdd"], session["sess"]
    yyyymmdd = f"2026{mmdd}"
    lab = f"{animal}_{mmdd}_{tag}"

    mc = rv.resolve("labcams", f"{yyyymmdd}/{sess}/motion_corrected")
    res = f"{mc}/wfield_local_results"
    allen = f"{res}/allen_aligned_{tag}"
    cue = f"{mc}/spout_trial_averages_{tag}"
    lick = f"{mc}/lick_aligned_{tag}"
    quiet = f"{mc}/quiet_{tag}"
    running_act = f"{mc}/running_activity_{tag}"
    events_npz = f"{rv.root('behavior_out')}/events/{animal}/{yyyymmdd}.npz"

    matches = sorted(glob.glob(f"{mc}/*cleanpairs_frame_map.npz"))
    if matches:
        fm = matches[0]
    elif allow_missing:
        fm = f"{mc}/<cleanpairs_frame_map.npz>"     # dry-run placeholder (no N: access on this box)
    else:
        raise SystemExit(f"[preprocess] {lab}: no *cleanpairs_frame_map.npz under {mc}")
    summ = fm.replace("_frame_map.npz", "_summary.json")

    cnpz = f"{cue}/{lab}_spout_positions_1s_pre_post_delta_maps.npz"
    csum = cnpz.replace("_maps.npz", "_summary.json")
    lnpz = f"{lick}/{lab}_lick_aligned_150ms_post_by_spout_maps.npz"
    lsum = lnpz.replace("_maps.npz", "_summary.json")
    qf = f"{quiet}/{lab}_quiet_frame.npy"
    daq = session["daq_h5"]

    bt = _behavior_trials_args(session, animal, yyyymmdd, rv)
    cue_pre_s, cue_post_s = str(mp["cue_pre_s"]), str(mp["cue_post_s"])
    lick_post_s = str(mp["lick_post_s"])

    return [
        ["wfield_local.framemap_event_maps", "--what", "cue", "--daq-h5", daq,
         "--wfield-results", res, "--allen-dir", allen, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", cue, "--label", lab,
         "--pre-s", cue_pre_s, "--post-s", cue_post_s] + bt,
        ["wfield_local.plot_spout_trial_averages_shared_scale", "--label", lab,
         "--trial-maps", cnpz, "--allen-dir", allen, "--output", cue, "--summary", csum],
        ["wfield_local.plot_spout_position_contrasts", "--label", lab,
         "--trial-maps", cnpz, "--allen-dir", allen, "--output", cue],
        ["wfield_local.framemap_event_maps", "--what", "lick", "--daq-h5", daq,
         "--wfield-results", res, "--allen-dir", allen, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", lick, "--label", lab,
         "--post-s", lick_post_s] + bt,
        ["wfield_local.plot_lick_position_contrasts", "--label", lab,
         "--lick-maps", lnpz, "--allen-dir", allen, "--output", lick],
        ["wfield_local.plot_lick_vs_cue_spout_maps", "--label", lab,
         "--cue-maps", cnpz, "--lick-maps", lnpz, "--allen-dir", allen, "--output", lick,
         "--cue-summary", csum, "--lick-summary", lsum],
        ["wfield_local.quiet_periods", "--daq-h5", daq, "--label", lab,
         "--output", quiet, "--frame-map", fm, "--cleanpairs-summary", summ],
        ["wfield_local.framemap_event_maps", "--what", "lick", "--daq-h5", daq,
         "--wfield-results", res, "--allen-dir", allen, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", lick, "--label", lab,
         "--post-s", lick_post_s, "--quiet-frame", qf] + bt,
        # canonical behavior events (shared licks/reward/running/quiet), then quiet-vs-running SVD maps
        ["wfield_local.behavior_events", yyyymmdd, "--only", animal],
        ["wfield_local.plot_running_activity_maps", "--label", lab, "--events", events_npz,
         "--wfield-results", res, "--allen-dir", allen, "--daq-h5", daq, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", running_act],
    ]


def generate_maps(session: dict, params: dict, rv: PathResolver, dry_run: bool) -> None:
    """Run the 8-command cue/lick/quiet MAPS chain for one session (frame_map globbed at run time)."""
    animal, sess = session["animal"], session["sess"]
    print(f"\n################ {animal} {sess} maps ################", flush=True)
    for cmd in _maps_commands(session, params, rv, allow_missing=dry_run):
        _run(cmd, dry_run)


# --------------------------------------------------------------------------- run steps
def _run(args: list, dry_run: bool) -> None:
    cmd = [sys.executable, "-m"] + [str(a) for a in args]
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    if subprocess.run(cmd, cwd=str(REPO)).returncode:
        raise SystemExit(f"[preprocess] FAILED: {args[0]}")


def preprocess_session(s: dict, params: dict, rv: PathResolver, dry_run: bool,
                       redo: bool = False) -> None:
    """motion(fixed) -> SVD -> cross-register to reference -> push LocaNMF inputs to MICROSCOPE.

    ``redo`` recomputes the three steps that are otherwise skipped whenever their output
    already exists. WHY IT HAS TO EXIST: on 2026-08-20 PS92 and PS95 were preprocessed from
    uploads that were 53% and 93% unwritten zeros. Recovering them means re-running after a
    good upload -- and without this flag that run prints "[skip] motion-corrected bin exists"
    and "[skip] SVTcorr exists" and reuses the garbage, fast enough to look like success. The
    only alternative was moving files out from under the pipeline by hand, which is a worse
    thing to have to get right at midnight.
    """
    animal, mmdd, sess, dims = s["animal"], s["mmdd"], s["sess"], s["dims"]
    yyyymmdd = f"2026{mmdd}"  # imaging cohort is 2026; keep in sync with the date arg
    mc = f"{s['sess_dir']}/motion_corrected"
    binp = f"{mc}/motioncorrect_{dims}_uint16.bin"
    results = f"{mc}/wfield_local_results"
    svd = params["svd"]
    print(f"\n################ {animal} {sess} (dims {dims}) ################", flush=True)
    if redo:
        print("[redo] recomputing steps whose outputs already exist", flush=True)
    if s["daq_h5"] is None:
        raise SystemExit(f"[preprocess] {animal} {sess}: no matching DAQ .h5 found")

    # 1 motion correction (sign-fixed, 2d)
    #
    # A REPAIRED session (wfield_local.repair_single_channel, e.g. PS95 2026-08-13) is ALREADY clean
    # 415/470 pairs, and its frame k is no longer DAQ exposure k. Re-running the TTL relabel on it
    # would not error -- `load_daq_labels` only checks that the frame count FITS inside the exposure
    # list, so offset 0 passes and it would pair frames from the dropped single-channel prefix,
    # silently producing a wrong movie. So the relabel is skipped and the frame map written by the
    # repair (which indexes ORIGINAL exposures, as downstream timing expects) is used instead.
    repaired = Path(s["raw_dat"]).parent / "repair_manifest.json"
    if repaired.exists():
        fms = sorted(Path(mc).glob("*cleanpairs_frame_map.npz"))
        if not fms and not dry_run:
            raise SystemExit(
                f"[preprocess] {animal} {sess}: repaired session has no *cleanpairs_frame_map.npz "
                f"under {mc} — run repair_single_channel.write_frame_map first")
        print(f"[repaired] {repaired.name} present: skipping TTL relabel, using "
              f"{fms[0].name if fms else '<frame map MISSING>'}", flush=True)
    if Path(binp).exists() and not dry_run and not redo:
        print("[skip] motion-corrected bin exists", flush=True)
    elif repaired.exists():
        _run(["wfield_local.run_wfield_motion", s["raw_dat"], "--output", mc,
              "--mode", params["motion_mode"]], dry_run)
    else:
        _run(["wfield_local.run_wfield_motion", s["raw_dat"], "--output", mc,
              "--daq-h5", s["daq_h5"], "--relabel-mode", params["relabel_mode"],
              "--mode", params["motion_mode"], "--relabel-on-the-fly"], dry_run)

    # 2 SVD (functional channel = 470)
    if Path(f"{results}/SVTcorr.npy").exists() and not dry_run and not redo:
        print("[skip] SVTcorr exists", flush=True)
    else:
        _run(["wfield_local.run_wfield_local", binp, "--output", results,
              "-k", svd["k"], "--functional-channel", svd["functional_channel"],
              "--fs", svd["fs"], "--freq-highpass", svd["freq_highpass"],
              "--freq-lowpass", svd["freq_lowpass"]], dry_run)

    # 2b motion-correction QC figure (shift stats + raw/corrected mean + residual std) -> mc/motion_qc.
    #    Pushed to N: in step 4 so the deck's per-date motion-QC slide shows this night. On-the-fly runs
    #    have no cleanpairs .dat, so pass the raw + corrected .bin explicitly.
    mqc = f"{mc}/motion_qc"
    qc_cmd = ["wfield_local.qc_motion_correction", "--motion-dir", mc,
              "--label", f"{animal}_{mmdd}_{params['maps']['tag']}", "--output", mqc,
              "--cor-bin", binp, "--fs", svd["fs"]]
    if s.get("raw_dat"):
        qc_cmd += ["--raw-dat", s["raw_dat"]]
    _run(qc_cmd, dry_run)

    # 3 cross-register to the animal's reference (6/6) -> emit allen_aligned_affine8v1
    ref_date, ref_results, ref_landmarks = reference_for(animal, params, rv)
    cfg = {"animal": animal, "mode": "reference-native", "func_channel": svd["functional_channel"],
           "reference": f"{animal}_{ref_date}", "warp_u": True,
           "output": rv.resolve("xday_qc", f"{animal}_{mmdd}"),
           "sessions": {f"{animal}_{ref_date}": {"results": ref_results, "landmarks": ref_landmarks},
                        f"{animal}_{mmdd}": {"results": results}}}
    cfg_path = f"{mc}/xday_config_{animal}_{mmdd}.json"
    print(f"[xreg] config -> {cfg_path}", flush=True)
    if not dry_run:
        Path(mc).mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
    _run(["wfield_local.cross_day_align", cfg_path], dry_run)

    # 3b build the ADOPTED hemodynamic/drift-removal variant beside the originals.
    #
    # The SVD writes only the zerophase SVTcorr (that IS what `wfield.hemodynamic_correction`
    # produces). The adopted variant is a separate product, and until now it had to be built by hand
    # per session afterwards -- a step that would eventually be forgotten, and whose absence is SILENT:
    # the session simply has no variant directory and drops out of the variant-based analyses.
    # Built BEFORE the push so it ships to MICROSCOPE with the rest of the results dir.
    hv = (params.get("hemo") or {})
    build_variant = hv.get("build_variant", "meegkit_hpfit")
    if build_variant and build_variant not in ("none", "zerophase"):
        vdir = f"{results}/hemo_{build_variant}"
        if Path(vdir).exists() and not dry_run and not redo:
            print(f"[skip] {Path(vdir).name} exists", flush=True)
        else:
            # NON-FATAL: the hemo variant is an optional add-on for variant-based analyses; the core
            # LocaNMF inputs (SVTcorr + Allen) must still push even if it fails. A session with a
            # coverage gap / single-channel prefix can make meegkit divide by all-zero weights and
            # raise -- that must not abort the whole night's preprocessing of the other animals.
            try:
                _run(["wfield_local.hemo_variants", "--variant", build_variant, "--write",
                      "--no-refit-t",            # hybrid: the saved T IS the high-pass-fitted T (3.5e-5)
                      "--label", f"{animal}_{mmdd}", "--mc", mc, "--h5", s["daq_h5"]], dry_run)
            except SystemExit as ex:
                print(f"[warn] hemo_variants FAILED for {animal}_{mmdd} ({ex}); continuing WITHOUT the "
                      f"{build_variant} variant (LocaNMF inputs still pushed). Session likely has a "
                      f"coverage gap / single-channel prefix -- rebuild the variant once that is fixed.",
                      flush=True)

    # 4 push LocaNMF inputs to MICROSCOPE FIRST (results dir + frame_map + summary; NOT the .bin)
    ndst = rv.resolve("labcams", f"{yyyymmdd}/{sess}/motion_corrected")
    nres = f"{ndst}/wfield_local_results"
    # ALREADY IN PLACE? Then there is nothing to push, and pushing would DESTROY the results.
    #
    # This step is rmtree(destination) followed by copytree(source -> destination). That is correct
    # when the raw was staged locally and the outputs live on E:. It is catastrophic when the raw is
    # discovered ON MICROSCOPE (--raw-root), because `results` and `nres` are then the SAME
    # directory: the rmtree deletes the results this run just spent hours computing, and the
    # copytree then fails with the source gone. Found by dry-running 2026-08-20, where the night's
    # raw arrived on MICROSCOPE rather than on E:.
    _same = (Path(results).resolve() == Path(nres).resolve())
    print(f"[push] {results} -> {nres}" + ("   SKIPPED: already in place" if _same else ""),
          flush=True)
    if not dry_run and not _same:
        writeguard.assert_writable(ndst)   # never rmtree/copy outside MICROSCOPE/Priya (rule 1)
        Path(ndst).mkdir(parents=True, exist_ok=True)
        if Path(nres).exists():
            shutil.rmtree(nres)
        shutil.copytree(results, nres)   # SVT/SVTcorr/U/T/rcoeffs/frames_average/summary + allen dir
        for pat in params["push_frame_map_globs"]:
            for f in glob.glob(f"{mc}/{pat}"):
                if Path(f).resolve() != Path(ndst, os.path.basename(f)).resolve():
                    shutil.copy2(f, os.path.join(ndst, os.path.basename(f)))
        # motion-correction QC dir (deck reads motion_corrected/motion_qc/*_motion_qc.png from N:)
        if Path(mqc).exists():
            nmqc = f"{ndst}/motion_qc"
            if Path(nmqc).exists():
                shutil.rmtree(nmqc)
            shutil.copytree(mqc, nmqc)

    # 5 drop the local relabel/cleanpairs intermediate: a transient hand-off from the TTL relabel
    # (separate h5py process) to motion correction — regenerable from raw, NOT archived (standby
    # keeps raw + motioncorrect.bin) and read by nothing downstream (SVD used the .bin; maps use the
    # pushed frame_map). Frees ~raw-sized local scratch. Guarded on the .bin so it only fires once
    # motion correction has actually produced its output.
    if not dry_run and Path(binp).exists():
        for f in glob.glob(f"{mc}/*cleanpairs*_uint16.dat"):
            try:
                os.remove(f)
                print(f"[cleanup] removed relabel intermediate {os.path.basename(f)}", flush=True)
            except OSError as e:
                print(f"[cleanup] could not remove {f}: {e}", flush=True)
    print(f"################ {animal} {sess} DONE ################", flush=True)


# --------------------------------------------------------------------------- main
def _process_date(date: str, args, rv: PathResolver, params: dict) -> set:
    """Discover + motion/SVD/xreg/push + maps + photobleach for ONE date. Returns animals processed."""
    sessions = discover_raw_sessions(date, rv, raw_root=getattr(args, "raw_root", None),
                                     daq_root=getattr(args, "daq_root", None))
    # CHECK THE RAW BEFORE COMMITTING HOURS TO IT (2026-08-20)
    usable = []
    for s in sessions:
        integ = raw_integrity(s["raw_dat"], s["dims"])
        s["raw_integrity"] = integ
        if not integ["ok"]:
            print(f"  !! {s['animal']} {s['sess']}: RAW NOT USABLE -- {integ['reason']} "
                  f"({integ['bytes']} B). Skipping; re-upload and re-run.", flush=True)
            if not getattr(args, "allow_incomplete_raw", False):
                continue
            print("     --allow-incomplete-raw given: processing anyway", flush=True)
        if integ.get("camlog_rows") is not None and integ["camlog_rows"] != integ["frames"]:
            print(f"  !! {s['animal']} {s['sess']}: .dat has {integ['frames']} frames but the "
                  f"camlog has {integ['camlog_rows']} -- proceeding, but the frame map may be off.",
                  flush=True)
        elif integ.get("interior_blank"):
            print(f"  !! {s['animal']} {s['sess']}: all-zero frame(s) at "
                  f"{integ['interior_blank']} but the tail is written -- proceeding, but look at "
                  f"the raw before trusting this session.", flush=True)
        elif integ["odd_last_pair"]:
            print(f"     {s['animal']}: {integ['frames']} frames (odd) -- last 415/470 pair is "
                  f"incomplete, which is where the recording stopped. Normal.", flush=True)
        usable.append(s)
    sessions = usable
    if args.skip_preprocess:
        # MERGE, do not fall back only when raw discovery is empty. Ownership of a night can be SPLIT
        # across the two boxes (8/12: this box preprocessed PS92+PS93, the imaging box PS94+PS95), so
        # exactly one animal's raw sat on E: here -- and a fallback gated on "found nothing" then
        # refreshed xall/maps for that ONE animal and silently skipped the other three. Downstream
        # steps should run on everything PROCESSED for the date, wherever the raw happens to live.
        have = {s["animal"] for s in sessions}
        extra = [s for s in discover_processed_sessions(date, rv) if s["animal"] not in have]
        if extra:
            print(f"[preprocess] {date}: +{len(extra)} PROCESSED session(s) from MICROSCOPE whose raw "
                  f"is not on {rv.root('raw_labcams')} ({', '.join(sorted(s['animal'] for s in extra))})"
                  f" -> included in the downstream steps")
            sessions = sorted(sessions + extra, key=lambda s: (s["animal"] or "", s["sess"]))
    if args.only:
        requested = set(args.only)
        sessions = [s for s in sessions if s["animal"] in requested]
        missing = sorted(requested - {s["animal"] for s in sessions})
        if missing:
            print(f"[preprocess] WARNING: --only asked for {', '.join(missing)} but no session was "
                  f"discovered for {'them' if len(missing) > 1 else 'it'} on {date} -- see any "
                  f"[discover] warnings above (e.g. a single-channel-mislabelled or missing raw).",
                  flush=True)
    if not sessions:
        where = "raw or processed" if args.skip_preprocess else "raw"
        print(f"[preprocess] no {where} sessions discovered for {date} under {rv.root('raw_labcams')}")
        return set()

    print(f"\n[preprocess] {date}: {len(sessions)} session(s) on machine={rv.machine}")
    for s in sessions:
        flag = "" if s["daq_h5"] else "  <<< NO DAQ MATCH"
        print(f"  {s['animal']}  {s['sess']}  dims={s['dims']}  daq={os.path.basename(s['daq_h5'] or '?')}{flag}")

    if args.skip_preprocess:
        print("[preprocess] --skip-preprocess: skipping motion/SVD/cross-register/push", flush=True)
    else:
        for s in sessions:
            # per-session params so configs/session_overrides.yaml actually applies (e.g. PS92 8/28
            # needs svd.functional_channel=0: labcams saved that 2-ch session mislabeled 1_460_480,
            # the rescue relabel locked offset 1 and swapped the .bin channel slots -- motion is fine
            # but SVD/hemo must read slot 0 as the 470 functional channel). Was passing global params,
            # so overrides never reached the per-session steps. Non-overridden sessions are unchanged.
            sp = config.defaults(session=f"{s['animal']}_{s['mmdd']}")["preprocess"]
            preprocess_session(s, sp, rv, args.dry_run,
                               redo=getattr(args, 'redo', False))

    # cue/lick/quiet activity maps (AFTER the push loop so the GPU-box LocaNMF push stays first)
    if not args.skip_maps:
        print("\n################ activity maps ################", flush=True)
        for s in sessions:
            sp = config.defaults(session=f"{s['animal']}_{s['mmdd']}")["preprocess"]
            generate_maps(s, sp, rv, args.dry_run)

    if not args.skip_photobleach:
        print("\n################ photobleach QC ################", flush=True)
        out_dir = rv.resolve("labcams", f"{date}/photobleach")
        triples = [(f"{s['animal']}_{s['mmdd']}", s["raw_dat"], s["daq_h5"])
                   for s in sessions if s["daq_h5"] and s["raw_dat"]]
        if args.dry_run:
            print(f"[dry-run] photobleach.run({len(triples)} sessions) -> {out_dir}")
        else:
            photobleach.run(triples, out_dir)

    return {s["animal"] for s in sessions if s["animal"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Config-driven nightly widefield preprocessing (imaging box).")
    ap.add_argument("dates", nargs="+", metavar="DATE",
                    help="session dates — same grammar as the analysis CLI: MMDD or YYYYMMDD, a range "
                         "(0806-0808), 'all' (every date-dir on the raw drive), or a comma/space list")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL",
                    help="restrict to these animals (e.g. PS94 PS95), or 'all' (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the discovery + planned commands only")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="skip motion/SVD/cross-register/push; re-run only downstream steps "
                         "(maps/xall) on already-processed sessions. Falls back to discovering "
                         "processed sessions from MICROSCOPE when the raw .dat is archived off E:.")
    ap.add_argument("--skip-photobleach", action="store_true")
    ap.add_argument("--skip-maps", action="store_true", help="skip the cue/lick/quiet activity-maps pass")
    ap.add_argument("--skip-xall", action="store_true", help="skip the all-days cross-day QC refresh")
    ap.add_argument("--skip-crossday-intensity", action="store_true",
                    help="skip the cross-day raw ROI fluorescence-intensity trend")
    ap.add_argument("--machine", default=None, help="override machine (default: auto-detect)")
    ap.add_argument("--redo", action="store_true",
                    help="recompute motion correction, SVD and the hemo variant even when their outputs already exist (default: skip). Needed to recover a session preprocessed from a bad upload.")
    ap.add_argument("--allow-incomplete-raw", action="store_true",
                    help="process a .dat that is not a whole number of frames (default: skip it)")
    ap.add_argument("--raw-root", default=None,
                    help="discover raw sessions HERE instead of this machine's raw_labcams root "
                         "(e.g. the MICROSCOPE labcams tree when the night's raw was uploaded "
                         "rather than staged locally). Outputs still land beside the raw.")
    ap.add_argument("--daq-root", default=None,
                    help="likewise for the DAQ .h5 tree (default: this machine's raw_daq root)")
    args = ap.parse_args(argv)

    rv = PathResolver(machine=args.machine)
    params = config.defaults()["preprocess"]
    try:
        dates = config.expand_dates(args.dates, width=8,
                                    available=list_raw_dates(rv, args.raw_root))
    except ValueError as e:
        ap.error(str(e))
    if not dates:
        print(f"[preprocess] no dates resolved from {args.dates} "
              f"(under {args.raw_root or rv.root('raw_labcams')})")
        return 1
    args.only = config.normalize_animals(args.only)   # None => all animals

    all_animals = set()
    for date in dates:
        all_animals |= _process_date(date, args, rv, params)
    if not all_animals:
        print(f"[preprocess] no raw sessions discovered for any of {dates}")
        return 1

    # all-days cross-day QC overlay (xall): ONCE after all dates, for every animal processed (the
    # per-date cross-day QC is emitted per session in preprocess_session step 3; this is the rollup)
    if not args.skip_xall:
        print("\n################ cross-day all-days QC (xall) ################", flush=True)
        refresh_xall(sorted(all_animals), params, rv, args.dry_run)

    # cross-day raw ROI fluorescence intensity trend: ONCE after all dates (reads every session's
    # frames_average on N:, so it picks up the days just pushed). Folds in the retired _crossday_intensity.py.
    if not args.skip_crossday_intensity:
        print("\n################ cross-day raw ROI intensity ################", flush=True)
        if args.dry_run:
            print(f"[dry-run] crossday_intensity.run(labcams={rv.root('labcams')}) -> {rv.root('xday_qc')}")
        else:
            from wfield_local import crossday_intensity   # lazy (pulls scipy/matplotlib)
            crossday_intensity.run(rv.root("labcams"), rv.root("xday_qc"))

    print(f"\nPREPROCESS {' '.join(dates)} motion->SVD->xreg->push ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
