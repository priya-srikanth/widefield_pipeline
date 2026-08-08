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
from wfield_local.paths import PathResolver

REPO = Path(__file__).resolve().parents[1]
DAT_RE = re.compile(r"pco_edge_run\d+_\d+_(2_\d+_\d+)_uint16\.dat$", re.I)
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
    """DAQ .h5 whose NAME contains the animal + the 8-digit date (robust to date-dir typos)."""
    if not animal:
        return None
    cands = sorted(h for h in h5s if animal in h.name and yyyymmdd in h.name)
    return cands[0] if cands else None


def _discover(yyyymmdd: str, raw_root: str, daq_root: str) -> list[dict]:
    """Core discovery (roots passed explicitly so it is unit-testable off the imaging box)."""
    datedir = Path(raw_root) / yyyymmdd
    if not datedir.exists():
        return []
    dats = [Path(p) for p in glob.glob(str(datedir / "**" / "*_uint16.dat"), recursive=True)
            if DAT_RE.search(Path(p).name)]
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


def discover_raw_sessions(yyyymmdd: str, rv: PathResolver) -> list[dict]:
    return _discover(yyyymmdd, rv.root("raw_labcams"), rv.root("raw_daq"))


def list_raw_dates(rv: PathResolver) -> list[str]:
    """All YYYYMMDD acquisition date-dirs present on this machine's raw_labcams root (for `all`/ranges).

    Empty when the raw drive isn't mounted (e.g. a dry-run off the imaging box) — explicit date
    tokens still work in that case; only `all` / ranges need the disk to enumerate against."""
    root = Path(rv.root("raw_labcams"))
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
def refresh_xall(animals: list[str], params: dict, rv: PathResolver, dry_run: bool) -> None:
    """Refresh each animal's all-days cross-day vasculature QC overlay (``xday/<animal>_xall``).

    QC-ONLY (``warp_u=False``): registers every registered task-era session (>= ``xall_min_date``)
    to the animal's reference and writes the multi-day `<animal>_cross_day_alignment_qc.png` +
    `<animal>_<date>_mean470_in_ref/ccf.npy`. Does NOT touch any session's allen_aligned dir
    (GPU LocaNMF inputs untouched). Folds in the retired `_xall_refresh.py`.
    """
    min_mmdd = str(params.get("xall_min_date", "0605"))
    all_sessions = config.load_sessions(machine=rv.machine)
    for animal in animals:
        ref_date, ref_results, ref_landmarks = reference_for(animal, params, rv)
        sess = {}
        for s in all_sessions:
            a, mmdd = s["label"].split("_")
            if a != animal or mmdd < min_mmdd:
                continue
            sess[f"{animal}_{mmdd}"] = {"results": f"{s['mc']}/wfield_local_results"}
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
def _behavior_trials_args(session: dict, animal: str, yyyymmdd: str, rv: PathResolver) -> list[str]:
    """``["--behavior-trials", trials.csv]`` when a recovered CSV is discoverable, else ``[]``.

    Preference order (no per-date special-casing — ``classify_cues_with_backup`` guards this, so
    passing trials.csv on a good session is safe; it only overrides when the DAQ is broken):
      1. an explicit ``session["behavior_trials"]`` (a config-recovered CSV, e.g. PS93 8/5 cam1),
      2. glob the behavior_logs root for ``<animal>_<yyyymmdd>_*/trials.csv``.
    """
    bt = session.get("behavior_trials")
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


def preprocess_session(s: dict, params: dict, rv: PathResolver, dry_run: bool) -> None:
    """motion(fixed) -> SVD -> cross-register to reference -> push LocaNMF inputs to MICROSCOPE."""
    animal, mmdd, sess, dims = s["animal"], s["mmdd"], s["sess"], s["dims"]
    yyyymmdd = f"2026{mmdd}"  # imaging cohort is 2026; keep in sync with the date arg
    mc = f"{s['sess_dir']}/motion_corrected"
    binp = f"{mc}/motioncorrect_{dims}_uint16.bin"
    results = f"{mc}/wfield_local_results"
    svd = params["svd"]
    print(f"\n################ {animal} {sess} (dims {dims}) ################", flush=True)
    if s["daq_h5"] is None:
        raise SystemExit(f"[preprocess] {animal} {sess}: no matching DAQ .h5 found")

    # 1 motion correction (sign-fixed, 2d)
    if Path(binp).exists() and not dry_run:
        print("[skip] motion-corrected bin exists", flush=True)
    else:
        _run(["wfield_local.run_wfield_motion", s["raw_dat"], "--output", mc,
              "--daq-h5", s["daq_h5"], "--relabel-mode", params["relabel_mode"],
              "--mode", params["motion_mode"]], dry_run)

    # 2 SVD (functional channel = 470)
    if Path(f"{results}/SVTcorr.npy").exists() and not dry_run:
        print("[skip] SVTcorr exists", flush=True)
    else:
        _run(["wfield_local.run_wfield_local", binp, "--output", results,
              "-k", svd["k"], "--functional-channel", svd["functional_channel"],
              "--fs", svd["fs"], "--freq-highpass", svd["freq_highpass"],
              "--freq-lowpass", svd["freq_lowpass"]], dry_run)

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

    # 4 push LocaNMF inputs to MICROSCOPE FIRST (results dir + frame_map + summary; NOT the .bin)
    ndst = rv.resolve("labcams", f"{yyyymmdd}/{sess}/motion_corrected")
    nres = f"{ndst}/wfield_local_results"
    print(f"[push] {results} -> {nres}", flush=True)
    if not dry_run:
        Path(ndst).mkdir(parents=True, exist_ok=True)
        if Path(nres).exists():
            shutil.rmtree(nres)
        shutil.copytree(results, nres)   # SVT/SVTcorr/U/T/rcoeffs/frames_average/summary + allen dir
        for pat in params["push_frame_map_globs"]:
            for f in glob.glob(f"{mc}/{pat}"):
                shutil.copy2(f, os.path.join(ndst, os.path.basename(f)))
    print(f"################ {animal} {sess} DONE ################", flush=True)


# --------------------------------------------------------------------------- main
def _process_date(date: str, args, rv: PathResolver, params: dict) -> set:
    """Discover + motion/SVD/xreg/push + maps + photobleach for ONE date. Returns animals processed."""
    sessions = discover_raw_sessions(date, rv)
    if not sessions and args.skip_preprocess:
        sessions = discover_processed_sessions(date, rv)
        if sessions:
            print(f"[preprocess] {date}: raw not on {rv.root('raw_labcams')} (archived); "
                  f"discovered {len(sessions)} PROCESSED session(s) from MICROSCOPE for downstream steps")
    if args.only:
        sessions = [s for s in sessions if s["animal"] in set(args.only)]
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
            preprocess_session(s, params, rv, args.dry_run)

    # cue/lick/quiet activity maps (AFTER the push loop so the GPU-box LocaNMF push stays first)
    if not args.skip_maps:
        print("\n################ activity maps ################", flush=True)
        for s in sessions:
            generate_maps(s, params, rv, args.dry_run)

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
    args = ap.parse_args(argv)

    rv = PathResolver(machine=args.machine)
    params = config.defaults()["preprocess"]
    try:
        dates = config.expand_dates(args.dates, width=8, available=list_raw_dates(rv))
    except ValueError as e:
        ap.error(str(e))
    if not dates:
        print(f"[preprocess] no dates resolved from {args.dates} (under {rv.root('raw_labcams')})")
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
