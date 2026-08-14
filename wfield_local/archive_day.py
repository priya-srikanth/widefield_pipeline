"""End-of-day archival + cleanup for a widefield recording day (reusable).

Daily policy this implements:
  * RAW camera movies            -> M: standby   (cold, immutable originals)
  * MOTION-CORRECTED video (.bin) -> M: standby  (huge; kept OFF MICROSCOPE to
    save space, under <date>\\<animal>\\motion_corrected\\). LocaNMF doesn't need
    it (it uses SVTcorr + the atlas), so this doesn't affect the GPU.
  * ALL other outputs + DAQ      -> N: MICROSCOPE (SVD: U/SVT/SVTcorr, Allen
    transform, event/lick/quiet maps, motion QC, DAQ h5)
  * Once every copy is size-verified, the copied E: files PLUS the reproducible
    E:-only intermediates (cleanpairs ``*.dat`` and any ``*_concat`` raw) may be
    deleted from E: to reclaim space.

Nothing is hardcoded to a session/date/dimension: pass --date YYYYMMDD and the
tool walks E:\\labcams_data\\<date>, mirrors the tree to M:/N:, and (on clean)
re-verifies each destination before removing the E: copy.

Classification of every file under E:\\labcams_data\\<date>:
  - ``*_uint16.dat`` whose name contains ``cleanpairs``            -> E-only intermediate
  - ``*_uint16.dat`` inside a ``*concat*`` folder                  -> E-only intermediate (concat raw)
  - any other ``*_uint16.dat``                                     -> RAW  -> M: (mirrors tree)
  - everything else (npy/npz/png/json/csv/camlog/tif/bin/...)      -> OUTPUT -> N: (mirrors tree)
DAQ h5 files containing <date> anywhere under E:\\DAQ_recorder_output -> N: DAQ
under a ``<date>\\`` folder (canonical server layout: ``DAQ_recorder_output\\<date>\\
<animal>_<date>_<time>.h5``), regardless of how they are foldered on E:.

Verification is size-based by default. ``--hash`` upgrades it to byte-level
(SHA-256): the source hash is computed *during* the copy (free) and stored in the
N: manifest, and the small N: outputs (+ DAQ) are read back and byte-compared. The
huge raw/.bin on M: stay size-matched + fingerprinted-on-copy, because a full M:
read-back is prohibitively slow (~15-150 MB/s); ``--hash-raw`` opts into that deep
read-back for an on-demand check (e.g. weekly, or when M: is idle).

Subcommands
  archive   copy E: -> M:/N: (idempotent; LocaNMF inputs first so a GPU can start
            immediately; M: raw copied last). Writes a manifest (with per-file
            sha256) on N:. --hash byte-verifies outputs as they land.
  verify    report whether every E: file has a confirmed copy on M:/N: (size, or
            SHA-256 for outputs with --hash / everything with --hash-raw).
  clean     delete confirmed-copied E: files + the reproducible intermediates.
            DRY-RUN by default; pass --execute to delete. Each delete re-verifies
            the destination at delete time (size, or bytes with --hash); an
            intermediate is removed only once its regeneration sources are confirmed
            archived (the session's raw on M: and a DAQ h5 for the date on N:).
            Empty dirs are pruned.

Examples
  python -m wfield_local.archive_day archive --date 20260604 --hash
  python -m wfield_local.archive_day verify  --date 20260604 --hash
  python -m wfield_local.archive_day clean   --date 20260604 --hash            # dry-run
  python -m wfield_local.archive_day clean   --date 20260604 --hash --execute
  python -m wfield_local.archive_day verify  --date 20260604 --hash-raw        # deep M: read-back

Drive roots default to this rig's mounts; override with --e-lab/--e-daq/--m-raw/
--n-lab/--n-daq if they differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from wfield_local import writeguard

BUF = 64 * 1024 * 1024

def _defaults() -> dict:
    """Drive roots for THIS machine, from configs/paths.yaml.

    These were hardcoded to the imaging box's mounts. The helper box added 2026-08-11 maps STANDBY AT A
    DIFFERENT DEPTH (``M:\\Widefield\\labcams``, the share already rooted at Priya), so the old
    ``M:\\collaborations\\Priya\\...`` default pointed at a directory that does not exist there — a
    tool that copies and then deletes, aimed at nothing. Resolved per machine now; a root that is
    unavailable stays None and the CLI reports it rather than silently writing to a wrong place.
    """
    from wfield_local import config
    rv = config.resolver()

    def _r(name):
        try:
            return rv.root(name)
        except Exception:                      # noqa: BLE001 - root not mounted on this machine
            return None

    return dict(e_lab=_r("raw_labcams"), e_daq=_r("raw_daq"), m_raw=_r("standby_labcams"),
                n_lab=_r("labcams"), n_daq=_r("daq_recorder_output"))


DEFAULTS = _defaults()


def _size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def _sha256(p):
    """Full SHA-256 of a file (streamed). Returns None if unreadable."""
    h = hashlib.sha256()
    try:
        with open(p, "rb", buffering=0) as f:
            while True:
                b = f.read(BUF)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def _is_big(job):
    """Huge cold files that go to M: standby (raw movie + motion-corrected .bin).

    Byte-verifying these means reading them back over the (slow) M: link, so by
    default they are size-matched + fingerprinted-on-copy rather than fully
    read-back-hashed. ``--hash-raw`` opts into the full read-back."""
    return job.get("kind") in ("raw", "mcbin")


def discover(cfg, date):
    """Return (jobs, intermediates, daq_jobs) for one recording day.

    jobs:          dicts(src,dst,kind in {raw,output},session) -- copied E->M/N
    intermediates: dicts(src,session,kind) -- reproducible, E-only (never copied)
    daq_jobs:      dicts(src,dst,kind=daq)
    """
    e_date = os.path.join(cfg["e_lab"], date)
    if not os.path.isdir(e_date):
        raise SystemExit(f"no labcams data for {date}: {e_date}")
    jobs, inter = [], []
    for root, _dirs, files in os.walk(e_date):
        parent = os.path.basename(root)
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, e_date)
            session = rel.split(os.sep)[0]
            if f.endswith("_uint16.dat"):
                manifest = os.path.join(root, "repair_manifest.json")
                if "cleanpairs" in f:
                    inter.append(dict(src=src, session=session, kind="cleanpairs"))
                elif "concat" in parent.lower():
                    inter.append(dict(src=src, session=session, kind="concat_raw"))
                elif os.path.isfile(manifest):
                    # A REPAIRED .dat (repair_single_channel) is not an acquisition -- it is a
                    # re-paired rebuild of one, sitting between the raw and the .bin exactly like a
                    # cleanpairs file. Archiving it would put a derivative in the raw archive under a
                    # name that looks like an original. It is deterministic, so it is an intermediate.
                    inter.append(dict(src=src, session=session, kind="repaired_raw",
                                      manifest=manifest))
                else:
                    jobs.append(dict(src=src, dst=os.path.join(cfg["m_raw"], date, rel),
                                     kind="raw", session=session))
            elif f.startswith("motioncorrect_") and f.endswith(".bin"):
                # mirror into the session folder alongside raw_widefield_data on M:
                jobs.append(dict(src=src, kind="mcbin", session=session,
                                 dst=os.path.join(cfg["m_raw"], date, rel)))
            else:
                jobs.append(dict(src=src, dst=os.path.join(cfg["n_lab"], date, rel),
                                 kind="output", session=session))
                if f.endswith(".camlog"):
                    # The camlog is an ACQUISITION RECORD, not an output: per-frame write log plus the
                    # LED controller's own state, which is what independently verified PS95 8/13's
                    # frame alignment when the DAQ alone could not settle it. It was landing only on
                    # MICROSCOPE -- a single copy of irreplaceable data. ~30 MB, so mirror it.
                    jobs.append(dict(src=src, dst=os.path.join(cfg["m_raw"], date, rel),
                                     kind="camlog_standby", session=session))
    return jobs, inter, discover_daq(cfg, date)


def discover_daq(cfg, date):
    """DAQ ``.h5`` files containing ``date`` under E: -> N: (independent of labcams; safe to run early).

    The destination is always ``<n_daq>/<date>/<file>.h5`` — the canonical server layout — no matter
    how the file is foldered on E:. Deriving it from ``date`` (not from the E: parent dir) is what
    keeps a loose ``E:\\DAQ_recorder_output\\PSxx_<date>_*.h5`` from landing in a nested
    ``DAQ_recorder_output\\DAQ_recorder_output\\`` on N:, and normalizes typo'd E: date dirs too.
    """
    daq = []
    if os.path.isdir(cfg["e_daq"]):
        for root, _dirs, files in os.walk(cfg["e_daq"]):
            for f in files:
                if f.endswith(".h5") and date in f:
                    daq.append(dict(src=os.path.join(root, f),
                                    dst=os.path.join(cfg["n_daq"], date, f),
                                    kind="daq", session=None))
    return daq


def _priority(job):
    """Copy order: LocaNMF inputs first, DAQ, other N: outputs, huge M: files last."""
    if job["kind"] in ("raw", "mcbin"):
        return 4   # huge cold files -> M: standby, least urgent
    if job["kind"] == "daq":
        return 2
    d = job["dst"].lower()
    if "wfield_local_results" in d or "allen_aligned" in d:
        return 0   # SVD (SVTcorr) + Allen transform = LocaNMF inputs
    return 3       # other N: outputs (maps, QC, shifts, summaries, frame_map, ...)


def _copy_one(src, dst, verify=False, big=False):
    """Copy src->dst (size-checked). Returns (status, src_sha256|None).

    The source SHA-256 is computed *during* the copy (piggybacks on the read we
    already do, so it is free) and returned for the manifest. When ``verify`` and
    the file is not ``big``, the destination is read back and hashed to confirm a
    bit-for-bit copy; ``big`` files (raw/.bin -> slow M:) are size-checked only."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ssz = _size(src)
    if os.path.exists(dst) and _size(dst) == ssz:
        if verify and not big:                       # already there: byte-check anyway
            sh = _sha256(src)
            if sh is None or sh != _sha256(dst):
                return "FAIL", sh
            return "skip", sh
        return "skip", None
    tmp = dst + ".part"
    h = hashlib.sha256()
    with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
        while True:
            b = fi.read(BUF)
            if not b:
                break
            fo.write(b)
            h.update(b)
    os.replace(tmp, dst)
    src_hash = h.hexdigest()
    if _size(dst) != ssz:
        return "FAIL", src_hash
    if verify and not big and _sha256(dst) != src_hash:   # read-back byte verify (cheap files)
        return "FAIL", src_hash
    return "ok", src_hash


def cmd_archive(cfg, date, verify=False, hash_raw=False):
    jobs, inter, daq = discover(cfg, date)
    allj = sorted(jobs + daq, key=lambda j: (_priority(j), -_size(j["src"])))
    total = sum(_size(j["src"]) for j in allj)
    vmsg = ("byte-verify outputs" + (" + raw/bin" if hash_raw else "; raw/bin fingerprint-on-copy")
            if verify else "size-verify only")
    print(f"[archive {date}] {len(allj)} files, {total/1e9:.1f} GB "
          f"(LocaNMF inputs first, raw->M last; {vmsg})\n", flush=True)
    res = {"ok": 0, "skip": 0, "FAIL": 0}
    fails, done = [], 0
    for i, j in enumerate(allj, 1):
        sz = _size(j["src"])
        big = _is_big(j) and not hash_raw
        try:
            st, sh = _copy_one(j["src"], j["dst"], verify=verify, big=big)
        except Exception as ex:
            st, sh = "FAIL", None
            print(f"   EXCEPTION {j['src']}: {ex}", flush=True)
        j["sha256"] = sh                             # recorded in manifest (None if not computed)
        res[st] = res.get(st, 0) + 1
        if st == "FAIL":
            fails.append(j)
        done += sz
        print(f"[{i}/{len(allj)}] {sz/1e9:6.2f} GB {st:4}  {j['dst']}  ({done/1e9:.1f}/{total/1e9:.1f})",
              flush=True)
    print(f"\n[archive {date}] ok={res['ok']} skip={res['skip']} FAIL={res['FAIL']}", flush=True)
    for j in fails:
        print(f"  FAILED: {j['src']} -> {j['dst']}", flush=True)
    if inter:
        gb = sum(_size(j["src"]) for j in inter) / 1e9
        print(f"[archive {date}] {len(inter)} reproducible intermediates left on E "
              f"({gb:.1f} GB) -- removed by `clean`, not copied.", flush=True)
    # manifest on N
    man = dict(date=date, n_files=len(allj), total_gb=round(total / 1e9, 2),
               result=res, intermediates=[j["src"] for j in inter],
               jobs=[{"src": j["src"], "dst": j["dst"], "kind": j["kind"],
                      "sha256": j.get("sha256")} for j in allj])
    mandir = os.path.join(cfg["n_lab"], date)
    try:
        os.makedirs(mandir, exist_ok=True)
        with open(os.path.join(mandir, f"_archive_manifest_{date}.json"), "w") as fh:
            json.dump(man, fh, indent=2)
    except OSError as ex:
        print(f"  (manifest not written: {ex})", flush=True)
    return 1 if fails else 0


def _verify(cfg, date, use_hash=False, hash_raw=False):
    """Classify each job as ok / missing / mismatch.

    ``use_hash``  -> also SHA-256-compare the small (N:) output files, not just size.
    ``hash_raw``  -> extend the byte compare to the huge raw/.bin on M: (slow read-back).
    A size match with a failed hash lands in ``mismatch`` with dest size ``-2``."""
    jobs, inter, daq = discover(cfg, date)
    allj = jobs + daq
    ok, missing, mismatch = [], [], []
    for j in allj:
        s, d = _size(j["src"]), _size(j["dst"])
        if d < 0:
            missing.append(j)
        elif d != s:
            mismatch.append((j, s, d))
        elif use_hash and not (_is_big(j) and not hash_raw):
            if _sha256(j["src"]) != _sha256(j["dst"]):
                mismatch.append((j, s, -2))          # size ok but bytes differ
            else:
                ok.append(j)
        else:
            ok.append(j)
    return allj, ok, missing, mismatch, inter, daq


def cmd_upload_daq(cfg, date, verify=False):
    """Copy ONLY the day's DAQ ``.h5`` to N: (size-verified, or byte-verified with --hash). Run FIRST in
    the imaging nightly so the analysis box can start the behavior pipeline (behavior_events) while imaging
    is still doing SVD."""
    daq = discover_daq(cfg, date)
    if not daq:
        print(f"[upload-daq {date}] no DAQ .h5 found under {cfg['e_daq']} (nothing to do)", flush=True)
        return 0
    res = {"ok": 0, "skip": 0, "FAIL": 0}
    fails = []
    for j in daq:
        try:
            st, _sh = _copy_one(j["src"], j["dst"], verify=verify, big=False)
        except Exception as ex:
            st = "FAIL"
            print(f"   EXCEPTION {j['src']}: {ex}", flush=True)
        res[st] = res.get(st, 0) + 1
        if st == "FAIL":
            fails.append(j)
        print(f"[upload-daq {date}] {_size(j['src'])/1e9:.2f} GB {st:4}  {j['dst']}", flush=True)
    print(f"[upload-daq {date}] ok={res['ok']} skip={res['skip']} FAIL={res['FAIL']} "
          f"(E: untouched)", flush=True)
    return 1 if fails else 0


def cmd_verify(cfg, date, use_hash=False, hash_raw=False):
    allj, ok, missing, mismatch, inter, _ = _verify(cfg, date, use_hash, hash_raw)
    how = ("byte-verified (outputs" + (" + raw/bin" if hash_raw else "; raw/bin size-only") + ")"
           if use_hash else "size-matched")
    print(f"[verify {date}] {len(ok)}/{len(allj)} confirmed copied ({how}) on M:/N:")
    for j in missing:
        print(f"  MISSING on dest: {j['src']} -> {j['dst']}")
    for j, s, d in mismatch:
        print(f"  {'BYTE MISMATCH' if d == -2 else 'SIZE MISMATCH'}: {j['src']} E={s}"
              f"{'' if d == -2 else f' dest={d}'}")
    print(f"[verify {date}] reproducible E-only intermediates (removed by clean): {len(inter)}")
    for j in inter:
        print(f"  {j['kind']}: {j['src']} ({_size(j['src'])/1e9:.1f} GB)")
    return 0 if not missing and not mismatch else 1


def _original_raw_archived(cfg, date, j) -> bool:
    """Is the ORIGINAL acquisition a repaired .dat was rebuilt from safely on standby?

    The normal regeneration test ("a raw for this session was confirmed copied from E:") cannot work
    here: the original single-channel file was never staged locally -- the repair read it straight off
    MICROSCOPE. So confirm the original itself, by the name recorded in ``repair_manifest.json``, at
    the expected byte size derived from the manifest's own frame count. Size rather than a hash because
    this only decides whether a REGENERABLE intermediate may go; the original is not being deleted.
    """
    import re

    try:
        with open(j["manifest"], "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        name = os.path.basename(meta["source_dat"])
        n_frames = int(meta["n_frames_in"])
    except (OSError, KeyError, ValueError, TypeError):
        return False
    # Geometry comes from the labcams filename (`_<nchan>_<h>_<w>_<dtype>.dat`), which is the
    # convention the whole pipeline already parses -- the repair manifest does not record it.
    m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16\.dat$", name)
    if not m:
        return False
    h, w = int(m.group(2)), int(m.group(3))
    expect = n_frames * h * w * 2                            # uint16, single-channel flat frames
    return _size(os.path.join(cfg["m_raw"], date, j["session"], "raw_widefield_data", name)) == expect


def _session_processed(cfg, date, session) -> bool:
    """Has this session actually been preprocessed, or is its raw merely STAGED?

    ``clean`` was written for acquire -> preprocess -> archive -> clean, where raw on E: is always
    already processed. Re-staging raw FROM the archive to reprocess it (as the 8/13 re-run does)
    breaks that assumption: the staged copy is archived by definition, so it looks deletable, and
    deleting it silently throws away a 40-minute network copy of data that has not been used yet.
    A session counts as processed only once its SVD output exists.
    """
    for root in (os.path.join(cfg["e_lab"], date, session, "motion_corrected"),
                 os.path.join(cfg["n_lab"], date, session, "motion_corrected")):
        if _size(os.path.join(root, "wfield_local_results", "SVTcorr.npy")) > 0:
            return True
    return False


def cmd_clean(cfg, date, execute, use_hash=False, hash_raw=False):
    allj, ok_jobs, missing, mismatch, inter, daq = _verify(cfg, date, use_hash, hash_raw)
    # confirmed regeneration sources per session
    raw_ok = set()
    for j in ok_jobs:
        if j["kind"] == "raw":
            raw_ok.add(j["session"])
    daq_ok = any(_size(j["dst"]) == _size(j["src"]) for j in daq)

    # Never delete the raw of a session that has not been preprocessed: it is STAGED, not spent.
    staged = [j for j in ok_jobs
              if j["kind"] == "raw" and not _session_processed(cfg, date, j["session"])]
    staged_sessions = {j["session"] for j in staged}
    # ...and not that session's DAQ .h5 either: preprocessing reads it from E:, so removing it fails
    # the very run the raw was staged for. DAQ files are named <animal>_<date>_<time>.h5, which shares
    # only the animal+date prefix with the labcams session folder.
    staged_prefix = tuple("_".join(s.split("_")[:2]) for s in staged_sessions)

    def _keep(j):
        if j in staged:
            return False
        if j["session"] in staged_sessions and j["kind"] in ("output", "raw"):
            return False
        if j["kind"] == "daq" and staged_prefix and \
                os.path.basename(j["src"]).startswith(staged_prefix):
            return False
        return True

    to_delete = [j for j in ok_jobs if _keep(j)]
    inter_del, inter_skip = [], []
    for j in inter:
        if j["kind"] == "repaired_raw":
            ok = _original_raw_archived(cfg, date, j) and daq_ok
        else:
            ok = j["session"] in raw_ok and daq_ok
        if ok:
            inter_del.append(j)
        else:
            inter_skip.append(j)

    freed = sum(_size(j["src"]) for j in to_delete) + sum(_size(j["src"]) for j in inter_del)
    print(f"{'EXECUTE' if execute else 'DRY-RUN'} clean {date}: "
          f"delete {len(to_delete)} copied + {len(inter_del)} intermediates "
          f"= {freed/1e9:.1f} GB | not-yet-copied (kept): {len(missing)+len(mismatch)} | "
          f"intermediates kept (regen unconfirmed): {len(inter_skip)}", flush=True)
    for j in staged:
        print(f"  KEEP (STAGED, session not preprocessed yet): {j['src']}")
    for j in missing:
        print(f"  KEEP (dest missing): {j['src']}")
    for j, s, d in mismatch:
        print(f"  KEEP (size mismatch): {j['src']}")
    for j in inter_skip:
        print(f"  KEEP (regen source not confirmed): {j['src']}")

    if not execute:
        print("\n(dry-run; pass --execute to delete)")
        return 0

    deleted = 0
    seen_src = set()
    for j in to_delete:
        if j["src"] in seen_src:                     # camlog has TWO destinations; delete once
            continue
        seen_src.add(j["src"])
        if _size(j["dst"]) != _size(j["src"]):          # re-verify size at delete time
            print(f"  SKIP at-delete (dest size changed): {j['src']}")
            continue
        if use_hash and not (_is_big(j) and not hash_raw):   # re-verify bytes at delete time
            if _sha256(j["src"]) != _sha256(j["dst"]):
                print(f"  SKIP at-delete (dest bytes differ): {j['src']}")
                continue
        # HARD RULE: acquired data only goes when a server copy is confirmed. The size/byte
        # re-verification just above IS that confirmation, so pass the destination as the verified
        # copy rather than asserting the rule is satisfied somewhere upstream.
        writeguard.assert_deletable(j["src"], verified_copies=[j["dst"]])
        os.remove(j["src"])
        deleted += 1
    for j in inter_del:
        # intermediates are reproducible from inputs whose archival was confirmed above
        writeguard.assert_deletable(j["src"], derived=True)
        os.remove(j["src"])
        deleted += 1
    # prune empty dirs under the day's E tree + any emptied DAQ parent dirs
    pruned = 0
    roots = [os.path.join(cfg["e_lab"], date)] + sorted({os.path.dirname(j["src"]) for j in daq})
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dp, _dirs, _files in os.walk(r, topdown=False):
            try:
                if not os.listdir(dp):
                    os.rmdir(dp)
                    pruned += 1
            except OSError:
                pass
    print(f"\ndeleted {deleted} files; pruned {pruned} empty dirs; freed ~{freed/1e9:.1f} GB")
    return 0


def main():
    ap = argparse.ArgumentParser(description="End-of-day widefield archival + cleanup.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("archive", "verify", "clean", "upload-daq"))
    ap.add_argument("--date", required=True, help="recording day, YYYYMMDD (labcams folder name)")
    ap.add_argument("--execute", action="store_true", help="(clean) actually delete; default dry-run")
    ap.add_argument("--hash", action="store_true",
                    help="byte-verify (SHA-256) the small N: outputs, not just size; "
                         "raw/.bin on M: stay size-matched + fingerprinted-on-copy")
    ap.add_argument("--hash-raw", action="store_true",
                    help="extend --hash to full read-back of the huge raw/.bin on M: (slow)")
    ap.add_argument("--machine", default=None, help="accepted + ignored (drive roots come from --e-*/--n-*)")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    use_hash = args.hash or args.hash_raw
    if args.command == "archive":
        return cmd_archive(cfg, args.date, verify=use_hash, hash_raw=args.hash_raw)
    if args.command == "verify":
        return cmd_verify(cfg, args.date, use_hash=use_hash, hash_raw=args.hash_raw)
    if args.command == "upload-daq":
        return cmd_upload_daq(cfg, args.date, verify=use_hash)
    return cmd_clean(cfg, args.date, args.execute, use_hash=use_hash, hash_raw=args.hash_raw)


if __name__ == "__main__":
    sys.exit(main())
