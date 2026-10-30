"""Cross-registration REFERENCE status per animal, and how to click its Allen landmarks (2026-10-29).

Each animal's sessions are registered to ONE reference session of that animal; the reference carries the hand-
clicked dorsal-cortex landmarks that place everything in Allen CCF. cohort1 used 6/6; cohort2 uses the cohort's
MASTER ALIGNMENT SESSION (configs/cohorts.yaml `reference_date`, overridable per animal). This prints, for the
active cohort, each animal's reference, its frame size, where the raw .dat is, the GUI command, and what is still
missing. With --sessions it lists every session with its frame size, to choose a master session on the large ROI.

    python -m wfield_local.alignment_reference                    # active cohort (WIDEFIELD_COHORT; default cohort1)
    python -m wfield_local.alignment_reference --cohort cohort2
    python -m wfield_local.alignment_reference --cohort cohort2 --sessions

The GUI (`python -m wfield_local.allen_register <dat>`) reads the frame size from the .dat name, so any ROI works.
Type the cohort's `reference_landmarks` label (e.g. v1) in its label box and Save: it writes
`dorsal_cortex_landmarks_<label>.json` beside the .dat. Preprocess copies that file to MICROSCOPE, builds the
reference's Allen dir, and registers every other session (a different ROI is placed at its offset, not stretched).
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from wfield_local import config
from wfield_local.paths import PathResolver

DAT_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_uint16\.dat$")


def _dats(rv: PathResolver, reldir: str) -> list[str]:
    """Raw .dat files for a session, wherever they still are (imaging-box E:, then standby)."""
    out = []
    for root in ("raw_labcams", "standby_labcams"):
        try:
            base = rv.root(root)
        except (RuntimeError, FileNotFoundError, KeyError):
            continue
        out += sorted(glob.glob(f"{base}/{reldir}/**/*_uint16.dat", recursive=True))
    return out


def _frame(path: str) -> str:
    m = DAT_RE.search(Path(path).name)
    return f"{m.group(2)}x{m.group(3)}" if m else "?"


def _session_dirs(rv: PathResolver, animal: str) -> list[tuple[str, str]]:
    """(MMDD, labcams-relative session dir) for every processed session of the animal on MICROSCOPE."""
    out = []
    for p in sorted(Path(rv.root("labcams")).glob(f"20*/{animal}_20*_*/motion_corrected")):
        d8 = p.parent.parent.name
        if re.fullmatch(r"\d{8}", d8):
            out.append((d8[4:], f"{d8}/{p.parent.name}"))
    return sorted(out, key=lambda x: config.date_key(x[0]))


def _shape_on_share(rv: PathResolver, reldir: str) -> str:
    try:
        import numpy as np
        sh = np.load(Path(rv.root("labcams"), reldir, "motion_corrected/wfield_local_results/frames_average.npy"),
                     mmap_mode="r").shape
        return f"{sh[-2]}x{sh[-1]}"
    except Exception:
        return "?"


def report(rv: PathResolver, list_sessions: bool = False) -> None:
    from wfield_local import preprocess
    params = config.defaults()["preprocess"]
    allen = str(config.defaults()["locanmf"]["allen_dir_name"])
    print(f"cohort {config.cohort()}: {', '.join(config.animals())}\n")
    for animal in config.animals():
        ref, lm, explicit = preprocess.reference_spec(animal, params)
        src = ("animal override" if (config.animals()[animal] or {}).get("reference_date")
               else "cohort master" if explicit else "cohort1 default")
        print(f"== {animal}  reference {ref or '(not chosen)'} [{src}]  landmarks label {lm or '(not set)'}")
        if list_sessions:
            for mmdd, rel in _session_dirs(rv, animal):
                print(f"     {mmdd}  {_shape_on_share(rv, rel):>9}  {rel}")
        if not ref:
            print("   -> set reference_date in configs/cohorts.yaml (master) or this animal's animals.yaml entry\n")
            continue
        rel = preprocess._reference_reldir(animal, ref, rv)
        if not rel:
            print(f"   -> no {animal} session dated {config.date8(ref)} on MICROSCOPE yet\n")
            continue
        res = Path(rv.root("labcams"), rel, "motion_corrected/wfield_local_results")
        name = f"dorsal_cortex_landmarks_{lm}.json" if lm else None
        on_share = bool(name) and Path(rv.root("labcams"), rel, "raw_widefield_data", name).exists()
        dats = _dats(rv, rel)
        print(f"   session  {rel}  frame {_shape_on_share(rv, rel)}")
        print(f"   landmarks on MICROSCOPE: {'yes' if on_share else 'NO'}   reference Allen dir: "
              f"{'yes' if (res / allen / 'U_atlas.npy').exists() else 'not built yet'}")
        if not on_share:
            if dats:
                big = max(dats, key=lambda p: Path(p).stat().st_size)
                print(f"   -> click landmarks:  python -m wfield_local.allen_register \"{big}\"   "
                      f"(frame {_frame(big)}; label box: {lm or 'v1'})")
            else:
                print("   -> raw .dat not found on this machine (imaging-box E: or standby); run this on the imaging "
                      "box, or load the .dat from within `python -m wfield_local.allen_register`")
        print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", help="cohort name (configs/cohorts.yaml); default: WIDEFIELD_COHORT or cohort1")
    ap.add_argument("--sessions", action="store_true", help="also list every session with its frame size")
    ap.add_argument("--machine", default=None)
    a = ap.parse_args(argv)
    if a.cohort:
        config.use_cohort_for_dates([], explicit=a.cohort)
    report(PathResolver(machine=a.machine), a.sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
