"""Swap the cam3-re-included 3-D of PS95 0818 / 0820 into the adopted four-camera run (Priya 11-23: "can we add back cam3
then?"; DECISIONS 2026-11-23).

    python scripts/swap_cam3_on_20261123.py --variant b_pnp4 [--dry-run]

Source = `scripts/qc_cam3_drop.py run ... --variant <variant>` (``<P3D>/qc_cam3_drop_20261122/cam3_on/<variant>/``);
target = the adopted run P3D = ``session_poses/pose3d_4cam_3net_20261121_sync60``. Our own derived outputs only (Priya
11-05: moving them within M: is fine); every replaced file is MOVED to ``<P3D>/_superseded_cam3drop_20261123/`` (same
relative path) first and listed in its MANIFEST.md. Steps:
  1. per session: ``<s>_frames.csv.gz``, ``<s>_licks.csv`` -> superseded; the variant's files in place (same names).
  2. ``summary_production_<side>.csv`` -> superseded; a NEW file under the same name with the two rows replaced by the
     variant's (its ``side_dropped`` keeps the REAL gate result, + ``side_forced_on`` / ``pnp_*``). Same name because
     readers glob ``summary_production*.csv`` / ``summary_*.csv`` (two files would double every row).
  3. per-session result caches of the two sessions (``_cache/<kind>/<s>.pkl``) -> superseded (they would miss anyway:
     the key holds the frames file's size + mtime).
  4. ``kinematics_fused/``: licks.csv.gz, licks_gated.csv.gz, jaw_trials.csv.gz, sessions.csv -> superseded; new files
     = the old text with the two sessions' rows replaced by the variant's (`qc_cam3_drop.kin`: the production code path,
     `pose3d_kinematics._session_result` + frame-coverage gate; a control reproduced production row for row), every
     other row byte-identical (read as text); by_animal.csv rebuilt from the new sessions.csv (`by_animal`).
  5. ``kinematics_fused_postcue`` + ``_allviews``: every file -> superseded, then `scripts/pose3d_kin_postcue.py --force`
     regenerates both from the new kinematics_fused (no 3-D recompute); rows of every other session checked identical.
The fused lick train, frame coverage and side events stay production (the lick SET does not change; cam3 is still
listed in ``side_dropped``, so `frame_coverage.fusion_cams` is unchanged).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

WT = Path(__file__).resolve().parents[1]
PY = "C:/Users/SabatiniLab/miniconda3/envs/dlc/python.exe"
CREATE_NO_WINDOW = 0x08000000
SIDE = "lp_side_r2b_prod_20261118"
P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
SUP = P3D / "_superseded_cam3drop_20261123"
SESSIONS = ["PS95_20260818", "PS95_20260820"]
KF_SPLICE = ["licks.csv.gz", "licks_gated.csv.gz", "jaw_trials.csv.gz", "sessions.csv"]
POSTCUE = ["kinematics_fused_postcue", "kinematics_fused_postcue_allviews"]


def splice_text(old: pd.DataFrame, new: pd.DataFrame, sessions, key: str = "session") -> pd.DataFrame:
    """``old`` (read as text) with the rows of ``sessions`` replaced by ``new``'s (columns aligned to ``old``; a column
    ``new`` lacks is blank, one ``old`` lacks is dropped), placed where the first replaced row was (else appended), in
    ``new``'s row order. Pure."""
    keep = ~old[key].isin(sessions)
    new = new[new[key].isin(sessions)].reindex(columns=old.columns).fillna("")
    first = int((~keep).to_numpy().argmax()) if (~keep).any() else len(old)
    before = old.iloc[:first][keep.iloc[:first].to_numpy()]
    after = old.iloc[first:][keep.iloc[first:].to_numpy()]
    return pd.concat([before, new.astype(str), after], ignore_index=True)


def _txt(f: Path) -> pd.DataFrame:
    return pd.read_csv(f, dtype=str, keep_default_na=False)


class Swap:
    def __init__(self, src: Path, dry: bool):
        self.src, self.dry, self.log = src, dry, []

    def move(self, f: Path, why: str) -> Path:
        dst = SUP / f.relative_to(P3D)
        self.log.append(("moved", f, dst, why))
        if not self.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise SystemExit(f"{dst} already exists (swap already run?)")
            shutil.move(str(f), str(dst))
        return dst

    def put(self, f: Path, write, why: str) -> None:
        self.log.append(("written", f, None, why))
        if not self.dry:
            write(f)


def run(variant: str, dry: bool) -> int:
    src = P3D / "qc_cam3_drop_20261122" / "cam3_on" / variant
    sw = Swap(src, dry)
    for s in SESSIONS:                                   # inputs present before anything moves
        for f in (src / f"{s}_frames.csv.gz", src / f"{s}_licks.csv", src / "kinematics_fused" / f"{s}_licks.csv.gz",
                  src / "kinematics_fused" / f"{s}_jaw_trials.csv.gz", src / "kinematics_fused" / f"{s}_session_row.csv"):
            if not f.exists():
                raise SystemExit(f"missing {f}")
    # 1. per-session 3-D files
    for s in SESSIONS:
        for name in (f"{s}_frames.csv.gz", f"{s}_licks.csv"):
            sw.move(P3D / name, "per-session 3-D (cam3 gate-dropped)")
            sw.put(P3D / name, lambda f, n=name: shutil.copy2(src / n, f), f"{variant} (cam3 forced on)")
    # 2. summary
    sname = f"summary_production_{SIDE}.csv"
    old = _txt(P3D / sname)
    new_rows = _txt(src / sname)
    cols = list(old.columns) + [c for c in new_rows.columns if c not in old.columns]
    S = splice_text(old.reindex(columns=cols).fillna(""), new_rows.reindex(columns=cols), SESSIONS)
    sw.move(P3D / sname, "summary with the cam3-dropped rows")
    sw.put(P3D / sname, lambda f: S.to_csv(f, index=False), f"two rows replaced by {variant}'s (+ forced / pnp columns)")
    # 3. per-session caches
    for d in sorted((P3D / "_cache").iterdir()):
        for s in SESSIONS:
            if (d / f"{s}.pkl").exists():
                sw.move(d / f"{s}.pkl", "per-session result cache (keyed on the old frames file)")
    # 4. kinematics_fused splice
    kf, ks = P3D / "kinematics_fused", src / "kinematics_fused"
    newp = {"licks.csv.gz": [ks / f"{s}_licks.csv.gz" for s in SESSIONS],
            "licks_gated.csv.gz": [ks / f"{s}_licks_gated.csv.gz" for s in SESSIONS],
            "jaw_trials.csv.gz": [ks / f"{s}_jaw_trials.csv.gz" for s in SESSIONS],
            "sessions.csv": [ks / f"{s}_session_row.csv" for s in SESSIONS]}
    built = {}
    for name in KF_SPLICE:
        o = _txt(kf / name)
        parts = [_txt(f) for f in newp[name] if f.exists()]
        n = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=o.columns)
        built[name] = splice_text(o, n, SESSIONS)
        print(f"kinematics_fused/{name}: {len(o)} -> {len(built[name])} rows "
              f"({int(o.session.isin(SESSIONS).sum())} -> {len(n)} for the two sessions)")
    for name in KF_SPLICE:
        sw.move(kf / name, "combined table with the cam3-dropped rows")
        sw.put(kf / name, lambda f, n=name: built[n].to_csv(f, index=False), "two sessions' rows replaced (others "
               "byte-identical)")
    from scripts.pose3d_kinematics import by_animal
    sw.move(kf / "by_animal.csv", "rule-8 table from the old sessions.csv")
    sw.put(kf / "by_animal.csv", lambda f: by_animal(pd.read_csv(kf / "sessions.csv", dtype={"date": str})).to_csv(
        f, index=False), "rebuilt from the new sessions.csv (`pose3d_kinematics.by_animal`)")
    # 5. post-cue folders, regenerated
    for d in POSTCUE:
        for f in sorted((P3D / d).iterdir()):
            if f.is_file() and f.name in ("licks.csv.gz", "sessions.csv", "by_animal.csv", "jaw_trials.csv.gz",
                                          "manifest.json"):
                sw.move(f, "post-cue view of the old kinematics_fused")
    if not dry:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_"))}
        env.update({"MKL_THREADING_LAYER": "SEQUENTIAL", "PYTHONPATH": str(WT), "PYTHONUNBUFFERED": "1"})
        log = SUP / "pose3d_kin_postcue.log"
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.Popen([PY, "scripts/pose3d_kin_postcue.py", "--force", "--jobs", "4"], cwd=str(WT), env=env,
                                 stdout=fh, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
            print(f"pose3d_kin_postcue pid {p.pid} -> {log}", flush=True)
            rc = p.wait()
        print(f"pose3d_kin_postcue rc {rc} in {(time.time() - t0) / 60:.1f} min")
        if rc:
            raise SystemExit(f"pose3d_kin_postcue failed (rc {rc}); see {log}")
    for d in POSTCUE:
        for name in ("licks.csv.gz", "sessions.csv", "by_animal.csv", "jaw_trials.csv.gz", "manifest.json"):
            sw.log.append(("written", P3D / d / name, None, "regenerated by scripts/pose3d_kin_postcue.py --force"))
    if not dry:
        sw.log += fixup()
    _manifest(sw, variant)
    return 0


def splice_measures(old: pd.DataFrame, new: pd.DataFrame, sessions, keep=("lick_idx", "lick_idx_fused")) -> pd.DataFrame:
    """``old`` (text) with, for the rows of ``sessions`` whose ``lick_key`` is in ``new``, every column but ``keep`` taken
    from ``new``; the rows' order, their ``keep`` columns and any row ``new`` lacks stay as in ``old``. Pure."""
    out = old.copy()
    m = out.session.isin(sessions)
    n = new[new.session.isin(sessions)].astype(str).set_index(["session", "lick_key"])
    idx = pd.MultiIndex.from_frame(out.loc[m, ["session", "lick_key"]])
    hit = idx.isin(n.index)
    rows = out.index[m][hit]
    cols = [c for c in out.columns if c not in keep and c not in ("session", "lick_key") and c in n.columns]
    out.loc[rows, cols] = n.loc[idx[hit], cols].to_numpy()
    return out


def fixup() -> list:
    """After `run`: every OTHER session's rows back to the superseded text (11-23 check: regenerating the post-cue view
    changed only ``lick_idx_fused`` -- the on-disk kinematics_fused was renumbered after the 11-22 post-cue build -- and
    dropped the 28,917 cam1-only rows of ``_allviews``, which the current kinematics_fused no longer holds; by_animal
    differed by float noise, 1e-14). The two sessions: their regenerated rows with the OLD ``lick_idx`` /
    ``lick_idx_fused`` (lick sets are identical); ``_allviews`` cam1-only rows of the two sessions keep their old
    (cam3-less) measures; sessions / by_animal rows of the two sessions recomputed from the spliced licks."""
    from scripts import pose3d_kin_postcue as PC
    from scripts.pose3d_kinematics import by_animal
    log = []

    o, n = _txt(SUP / "kinematics_fused" / "by_animal.csv"), _txt(P3D / "kinematics_fused" / "by_animal.csv")
    splice_text(o, n, SESSIONS).to_csv(P3D / "kinematics_fused" / "by_animal.csv", index=False)
    log.append(("written", P3D / "kinematics_fused" / "by_animal.csv", None, "fix-up: other rows = old text"))
    S_fused = pd.read_csv(P3D / "kinematics_fused" / "sessions.csv", dtype={"date": str})
    for d in POSTCUE:
        lo, ln = _txt(SUP / d / "licks.csv.gz"), _txt(P3D / d / "licks.csv.gz")
        L = splice_measures(lo, ln, SESSIONS)
        assert (L[~L.session.isin(SESSIONS)].reset_index(drop=True)
                .equals(lo[~lo.session.isin(SESSIONS)].reset_index(drop=True)))
        L.to_csv(P3D / d / "licks.csv.gz", index=False)
        P2 = pd.read_csv(P3D / d / "licks.csv.gz", dtype={"date": str, "lick_key": str})
        P2 = P2[P2.session.isin(SESSIONS)]
        S2 = PC.session_table(S_fused[S_fused.session.isin(SESSIONS)], P2)
        so = _txt(SUP / d / "sessions.csv")
        S_new = splice_text(so, S2.astype(object).where(S2.notna(), "").astype(str), SESSIONS)
        S_new.to_csv(P3D / d / "sessions.csv", index=False)
        B = by_animal(pd.read_csv(P3D / d / "sessions.csv", dtype={"date": str}))
        splice_text(_txt(SUP / d / "by_animal.csv"), B.astype(object).where(B.notna(), "").astype(str),
                    SESSIONS).to_csv(P3D / d / "by_animal.csv", index=False)
        for name in ("licks.csv.gz", "sessions.csv", "by_animal.csv"):
            log.append(("written", P3D / d / name, None, "fix-up: other sessions = old text; the two sessions' rows "
                        "regenerated (old lick_idx / lick_idx_fused" + ("; their cam1-only rows keep old measures)"
                                                                         if "allviews" in d else ")")))
        mf = json.loads((SUP / d / "manifest.json").read_text())
        mf["cam3_reincluded_20261123"] = "PS95_20260818 / PS95_20260820 rows: cam3 forced on (see " \
            "_superseded_cam3drop_20261123/MANIFEST.md)"
        (P3D / d / "manifest.json").write_text(json.dumps(mf, indent=1))
        log.append(("written", P3D / d / "manifest.json", None, "old manifest + cam3_reincluded_20261123 note"))
    return log


def _manifest(sw: Swap, variant: str) -> None:
    lines = ["# Superseded by the PS95 0818 / 0820 cam3 re-inclusion (2026-11-23)", "",
             "Priya 11-23: \"can we add back cam3 then?\" -- PS95 (mouse 12) 0818 / 0820 cam3 had failed the side-camera "
             "spout gate from chin-fur bias on the close-spout tip labels. Replacement 3-D: "
             f"`qc_cam3_drop_20261122/cam3_on/{variant}/` (see `qc_cam3_drop_20261122/cam3_on/validation.csv`). "
             "Swapped by `scripts/swap_cam3_on_20261123.py` (pilot branch `pilot/side4cam-wholesession`); DECISIONS "
             "2026-11-23.", "", "Fused lick train, frame coverage and side events are unchanged (production).", "",
             "| action | original path | superseded copy | note |", "|---|---|---|---|"]
    for act, f, dst, why in sw.log:
        lines.append(f"| {act} | `{f.relative_to(P3D).as_posix()}` | "
                     f"{'`' + dst.relative_to(P3D).as_posix() + '`' if dst else ''} | {why} |")
    lines += ["", f"P3D = `{P3D.as_posix()}`"]
    txt = "\n".join(lines) + "\n"
    if sw.dry:
        print(txt)
    else:
        SUP.mkdir(parents=True, exist_ok=True)
        (SUP / "MANIFEST.md").write_text(txt, encoding="utf-8")
        (SUP / "manifest.json").write_text(json.dumps([[a, str(f), str(d) if d else "", w] for a, f, d, w in sw.log],
                                                      indent=1))
        print("->", SUP / "MANIFEST.md")


def _append_manifest(log: list) -> None:
    lines = ["", "## Fix-up (same night, `--fixup-only`)", "", "| action | original path | superseded copy | note |",
             "|---|---|---|---|"]
    lines += [f"| {a} | `{f.relative_to(P3D).as_posix()}` |  | {w} |" for a, f, _d, w in log]
    with open(SUP / "MANIFEST.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("->", SUP / "MANIFEST.md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", required=True, choices=["a_forced", "b_pnp4"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fixup-only", action="store_true", help="re-run only the fix-up step (after a completed swap)")
    a = ap.parse_args(argv)
    if a.fixup_only:
        _append_manifest(fixup())
        return 0
    return run(a.variant, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
