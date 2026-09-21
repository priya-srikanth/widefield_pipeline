"""MOVE THE PUBLISHED OUTPUT TREE INTO ITS 2026-09-21 LAYOUT. Dry-run by default.

    PYTHONPATH=$(pwd) python scripts/restructure_output_tree.py            # plan only, moves nothing
    PYTHONPATH=$(pwd) python scripts/restructure_output_tree.py --apply

WHY THIS EXISTS AS A SCRIPT AND NOT A SESSION OF `mv`. It touches published artefacts on
MICROSCOPE, so what moved and why has to be readable afterwards by someone who was not here. It is
also IDEMPOTENT -- every step checks whether it has already run -- because the alternative is a
half-applied layout nobody can reason about, and because the nightly may land between a plan and
an apply.

NOTHING IS DELETED (ground rules 0/1). Every operation is a MOVE, within the same filesystem, so
it is a rename: fast, atomic per file, and reversible by reading this file backwards. Files that
are retired go to a `retired/<reason>/` directory, never away.

WHAT CHANGES, AND WHY EACH ONE

  1. `grant_figures/{epoch/,}svg/`  -- 98 + 352 vector copies out of the figure directories.
     They are a deliverable for pasting into the grant document; the deck never reads one. Mixed
     in, they double the apparent size of a directory whose whole job is "what figures exist".

  2. `grant_figures/epoch/data/`    -- 894 CSV + 42 JSON sidecars. 54% of that directory was data,
     not figures. `deck_values.Resolver` reads them by stem and now searches `data/` too.

  3. `grant_figures/epoch/retired/` -- the four existing `retired_<reason>/` directories, which
     already carried READMEs, consolidated under one parent with an index. Plus `QC_scratch/` for
     the eight `_QC_*` / `_mask_*` images: ad-hoc pictures dropped into the epoch directory, named
     in `docs/status/STATUS_2026-09-21_ENGINEERING.md` as "not deck material", and the oldest files there.

  4. `labcams/analysis_figures/`    -- was `labcams/locanmf_lick_pooled/cue_analysis`. THE NAME WAS
     THE PROBLEM: this is the live nightly mirror of every analysis figure, 3,378 files, and it sat
     inside a directory named after an abandoned June 2026 pooling experiment. Its `analysis_json/`
     becomes `json/`. One root in `configs/paths.yaml` moves with it.

  5. `labcams/retired/locanmf_lick_pooled_202606/` -- the actual June one-off: ten top-level
     figures, `component_cards/`, and the two `LEGACY_*.pptx`.

  6. `labcams/retired/mirror_pre_august/` -- 210 mirror figures from June and July that no current
     renderer rewrites. `_publish_figs` never deletes, deliberately ("mirror cleanup stays a manual
     step"), so this IS that manual step.

  7. `channel_comparison/retired/2026-07-08_oneoff/` -- the SVGs and PNGs left by
     `_compare_415_470_corr.py`, retired 2026-08-08, now fully superseded.

WHAT DOES NOT MOVE. `xday` (30 G), `deck_history` (9.4 G), `joint_bases` (8.2 G), `frozen_models`,
`snapshots` and the per-date preprocessing trees: their names already say what they are, and moving
tens of gigabytes to rename nothing is a bad trade. `Widefield/quarantine` is raw-data quarantine
with its own README and a separate decision attached to it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

#: Figures from before this date in the nightly mirror that nothing re-renders. Chosen as the
#: boundary because the August restructure of the epoch families renamed most of what precedes it.
MIRROR_CUTOFF = "2026-08-01"

#: The eight ad-hoc images that were dropped into the epoch figure directory.
QC_SCRATCH = ("_QC_glue_BRIGHT", "_QC_glue_candidates", "_QC_posterior_left_LABELLED",
              "_mask_FRP_and_glue_candidate", "_mask_FRP_vs_MOB", "_mask_check_FIXED",
              "_mask_glue_measured", "_mask_glue_per_animal")


class Plan:
    """Collects moves, prints them, and applies them only when asked."""

    def __init__(self, apply: bool):
        self.apply = apply
        self.moves: list[tuple[Path, Path]] = []
        self.notes: list[str] = []
        self._claimed: set[Path] = set()

    def move(self, src: Path, dst: Path):
        """Queue one move. FIRST CLAIM WINS, and that is not a detail.

        The sweeps overlap on purpose -- `*.svg` collects vectors while the QC sweep collects
        `_QC_*.*` -- so a `_QC_*.svg` matches both. Both would pass the `exists()` check at PLAN
        time, because nothing has moved yet, and the second would then fail at APPLY time against
        a file that is no longer there. A half-applied layout is the one outcome this script must
        not produce, so a source that is already spoken for is dropped here instead.
        """
        if not src.exists() or src in self._claimed:
            return
        if dst.exists():
            self.notes.append(f"already in place, skipped: {dst}")
            return
        self._claimed.add(src)
        self.moves.append((src, dst))

    def note(self, s):
        self.notes.append(s)

    def run(self):
        by_dest = {}
        for src, dst in self.moves:
            by_dest.setdefault(dst.parent, []).append((src, dst))
        for parent, items in sorted(by_dest.items(), key=lambda kv: str(kv[0])):
            print(f"\n  {parent}   <- {len(items)} item(s)")
            for src, dst in items[:4]:
                print(f"      {src.name}")
            if len(items) > 4:
                print(f"      ... and {len(items) - 4} more")
        for n in self.notes:
            print(f"  note: {n}")
        print(f"\n  TOTAL {len(self.moves)} move(s)")
        if not self.apply:
            print("  DRY RUN -- nothing moved. Re-run with --apply.")
            return 0
        done = 0
        for src, dst in self.moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
                done += 1
            except OSError as ex:
                print(f"  !! {src} -> {dst}: {type(ex).__name__} {ex}")
        print(f"  moved {done}/{len(self.moves)}")
        return 0 if done == len(self.moves) else 1


README = """# Retired figures — what is in here and why

Everything under this directory was PUBLISHED output that has been superseded. Nothing here was
deleted; MICROSCOPE keeps originals (ground rules 0/1) and these are kept so a figure that turns
up in an old document can still be traced to the run that made it.

Moved here by `scripts/restructure_output_tree.py` on 2026-09-21. Each subdirectory carries the
reason in its name, and the four that predate this consolidation kept their own READMEs.

| directory | what it holds |
|---|---|
| `QUIETref/` | the QUIET-reference family of `epoch_15r`, retired 2026-09-12 |
| `INVALID_PRECUEref_on_precue_arm/` | PRECUE-referenced figures on the PRE-CUE arm, which normalises a window to itself |
| `ORPHANED_pre_rename_20260913/` | figures whose names changed on 2026-09-13; the files are the pre-rename copies |
| `superseded/` | `epoch_14` beta maps replaced by the reference-family versions |
| `QC_scratch/` | ad-hoc `_QC_*` / `_mask_*` images dropped into the epoch directory; never deck material |
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually move (default: plan only)")
    ap.add_argument("--root", type=Path, default=None, help="labcams root (default: resolved)")
    a = ap.parse_args(argv)

    if a.root is None:
        from wfield_local.paths import PathResolver
        a.root = Path(PathResolver().root("labcams"))
    lab = a.root
    if not lab.exists():
        sys.exit(f"labcams root does not exist: {lab}")
    print(f"labcams root: {lab}\n{'=' * 96}")

    p = Plan(a.apply)
    grant, epoch = lab / "grant_figures", lab / "grant_figures" / "epoch"
    retired = epoch / "retired"

    # 3 FIRST, because `Plan.move` is first-claim-wins and the sweeps overlap: a `_QC_*.svg` is
    # both a QC scratch image and a vector, and it belongs with its PNG in QC_scratch rather than
    # alone in svg/. Retiring before sorting also keeps retired files out of the new subdirectories
    # entirely, which is what makes `ls epoch/*.png` mean "the current figures".
    for d in sorted(epoch.glob("retired_*")) if epoch.exists() else []:
        if d.is_dir():
            p.move(d, retired / d.name[len("retired_"):])
    for stem in QC_SCRATCH:
        for f in sorted(epoch.glob(stem + ".*")) if epoch.exists() else []:
            p.move(f, retired / "QC_scratch" / f.name)

    # 1/2. vectors and sidecars out of the figure directories
    for d in (grant, epoch):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.svg")):
            p.move(f, d / "svg" / f.name)
    if epoch.exists():
        # `.npz` TOO. The `epoch_14` beta-map bundles are sidecars like any other, and leaving
        # 52 of them at the top while the writer had already moved to `data/` would have split one
        # family across two layouts -- the reader finds the new path first, so the old files would
        # simply have stopped being read with nothing to say so.
        for ext in ("*.csv", "*.json", "*.npz"):
            for f in sorted(epoch.glob(ext)):
                p.move(f, epoch / "data" / f.name)
        for ext in ("*.csv", "*.json", "*.npz"):
            for f in sorted(grant.glob(ext)):
                p.move(f, grant / "data" / f.name)
    # Windows thumbnail caches. Not data; they reappear whenever the folder is browsed, which is
    # why they are swept rather than fought.
    for d in (grant, epoch, lab / "analysis_figures"):
        for f in sorted(d.glob("Thumbs.db")) if d.exists() else []:
            p.move(f, d / "retired" / "Thumbs.db")

    # 4/5/6. the mirror, and the June one-off it was hiding inside
    pooled = lab / "locanmf_lick_pooled"
    mirror_old, mirror_new = pooled / "cue_analysis", lab / "analysis_figures"
    lab_retired = lab / "retired"
    if mirror_old.exists() and not mirror_new.exists():
        import datetime as _dt
        cut = _dt.datetime.fromisoformat(MIRROR_CUTOFF).timestamp()
        for f in sorted(mirror_old.glob("*.png")) + sorted(mirror_old.glob("*.svg")):
            if f.stat().st_mtime < cut:
                p.move(f, lab_retired / "mirror_pre_august" / f.name)
        for f in sorted(mirror_old.glob("LEGACY_*.pptx")):
            p.move(f, lab_retired / "locanmf_lick_pooled_202606" / f.name)
        p.note(f"then the whole mirror moves: {mirror_old} -> {mirror_new} "
               f"(with analysis_json/ -> json/)")
    if pooled.exists():
        for f in sorted(pooled.glob("*")):
            if f.name in ("cue_analysis",):
                continue
            p.move(f, lab_retired / "locanmf_lick_pooled_202606" / f.name)

    # 6b. diagnostic output from a probe that is now archived. `plot_drift_estimators` moved to
    # `scripts/rest_migration/archive/` on 2026-09-21; its nine figures were sitting inside the
    # GRANT deliverable directory, which is for the summary set and nothing else.
    de = grant / "drift_estimators"
    if de.exists():
        p.move(de, lab_retired / "drift_estimators")

    # 7. the retired 415-vs-470 one-off
    chan = lab / "channel_comparison"
    if chan.exists():
        for f in sorted(chan.glob("*.svg")):
            p.move(f, chan / "retired" / "2026-07-08_oneoff" / f.name)
        for f in sorted(chan.glob("*_415_vs_470_vs_corr.png")):
            p.move(f, chan / "retired" / "2026-07-08_oneoff" / f.name)
        for f in sorted(chan.glob("Thumbs.db")):
            p.move(f, chan / "retired" / f.name)

    rc = p.run()

    # The mirror directory itself moves last, once its retirees are out, so the rename is a single
    # cheap operation rather than 3,000 individual ones.
    if a.apply and mirror_old.exists() and not mirror_new.exists():
        shutil.move(str(mirror_old), str(mirror_new))
        print(f"  moved mirror -> {mirror_new}")
        oldj, newj = mirror_new / "analysis_json", mirror_new / "json"
        if oldj.exists() and not newj.exists():
            shutil.move(str(oldj), str(newj))
            print(f"  moved {oldj.name} -> {newj.name}")
    if a.apply and retired.exists():
        (retired / "README.md").write_text(README, encoding="utf-8")
        print(f"  wrote {retired / 'README.md'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
