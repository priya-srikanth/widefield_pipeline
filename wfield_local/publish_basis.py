"""Publish the joint-LocaNMF bases to MICROSCOPE, byte-verified.

WHY (Priya, 2026-08-26). The joint bases live in a LOCAL directory (`E:/joint_bases` on the helper
box). That has bitten twice in two days:

  THE DECK COULD NOT BE BUILT.  The behavior box ran the 8/24 and 8/25 analysis but could not produce
                                any joint-basis figure, because the basis is not on a share it can
                                reach -- so its deck hit the completeness gate and refused.
  THE BASIS CANNOT BE REBUILT IDENTICALLY.  `joint_locanmf` refuses to silently refit precisely
                                because "a refit over a grown session set is a DIFFERENT reference
                                frame". So a basis is not a cache that can be regenerated; it is a
                                reference, and losing the disk it sits on loses the frame that every
                                cross-day joint result was computed in.

VERIFICATION IS BY CONTENT, NOT SIZE. Rule 0: a copy whose existence and size you have checked is not
a verified copy, and a preallocated or interrupted destination can match on size while differing in
bytes. Each file is compared by SHA-256 -- read back from the DESTINATION after the copy, not from
the source buffer -- and the manifest records the digest so a later run can re-verify without
re-copying.

NEVER OVERWRITES A VERIFIED-DIFFERENT FILE. A basis directory is immutable by construction: its
`basis_id` is a hash of its own inputs, so two directories with the same id must have the same
contents. A digest mismatch under an unchanged id means corruption somewhere, and the honest response
is to report it rather than pick a side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from wfield_local import config, joint_locanmf, writeguard

CHUNK = 8 << 20


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def server_root():
    return Path(config.resolver().root("labcams")) / "joint_bases"


def local_bases():
    """[(animal, basis_dir)] for every basis on this box, newest last."""
    root = Path(joint_locanmf.BASIS_DIR)
    out = []
    if not root.exists():
        return out
    for an in sorted(root.iterdir()):
        if not an.is_dir():
            continue
        for b in sorted(an.iterdir(), key=lambda x: x.stat().st_mtime):
            if b.is_dir() and (b / "manifest.json").exists():
                out.append((an.name, b))
    return out


def publish_one(animal, basis_dir: Path, dry_run=False, log=print) -> dict:
    dst = server_root() / animal / basis_dir.name
    res = {"animal": animal, "basis": basis_dir.name, "copied": 0, "verified": 0,
           "already": 0, "MISMATCH": []}
    files = sorted(p for p in basis_dir.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(basis_dir)
        d = dst / rel
        want = sha256(p)
        if d.exists():
            # Re-verify rather than trust: this is the whole point of publishing a reference.
            if sha256(d) == want:
                res["already"] += 1
                continue
            res["MISMATCH"].append(str(rel))
            log(f"  !! {animal}/{basis_dir.name}/{rel}: published copy DIFFERS from local and a "
                f"basis id is a hash of its own inputs, so this is corruption, not an update. "
                f"NOT overwritten.")
            continue
        if dry_run:
            res["copied"] += 1
            continue
        writeguard.assert_writable(d.parent)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)
        if sha256(d) != want:                      # read back from the DESTINATION
            res["MISMATCH"].append(str(rel))
            log(f"  !! {animal}/{basis_dir.name}/{rel}: copied but the destination digest does not "
                f"match. Left in place for inspection; do NOT treat this basis as published.")
            continue
        res["copied"] += 1
        res["verified"] += 1
    if not dry_run and not res["MISMATCH"]:
        (dst / "PUBLISH_MANIFEST.json").write_text(json.dumps(
            {"animal": animal, "basis_id": basis_dir.name, "n_files": len(files),
             "sha256": {str(p.relative_to(basis_dir)): sha256(p) for p in files}}, indent=2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--animals", nargs="+", default=None)
    a = ap.parse_args(argv)
    bases = [(an, b) for an, b in local_bases() if not a.animals or an in set(a.animals)]
    if not bases:
        print("[publish_basis] no local basis found", flush=True)
        return 1
    bad = 0
    for an, b in bases:
        r = publish_one(an, b, dry_run=a.dry_run)
        flag = "  !! MISMATCH" if r["MISMATCH"] else ""
        print(f"  {an}/{b.name}: copied {r['copied']}, verified {r['verified']}, "
              f"already current {r['already']}{flag}", flush=True)
        bad += len(r["MISMATCH"])
    print(f"[publish_basis] {'DRY RUN' if a.dry_run else 'done'} -> {server_root()}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
