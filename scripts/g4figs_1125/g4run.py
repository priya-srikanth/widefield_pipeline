"""Run an encoding-family READER on the collected G4 per-session fits WITHOUT collecting them (CC-B helper, 11-25).

G4 = O2 bundle enc_fourcam3d_20261124 (20 tags x 107 sessions, on the share under <bundle>/out/<TAG>/sessions). The
readers expect the collected layout `<encoding_out_root>/<folder>/sessions`. `collect` would copy ~33 GB per tag into
the protected routed encoding folder; instead this driver gives the reader a VIEW root inside this run's own new
output folder:

  * `<VIEW>/<folder>/sessions[/...]`  -> READ from `<bundle>/out/<TAG>/sessions[/...]` (load-only; never written)
  * `<VIEW>/<folder>/<anything else>` -> the real path inside VIEW (tables / figures the reader writes)

`run_mode.encoding_out_root`, `frozen_encoder.out_root` and `fr_groups.run_folder` are patched to return the view, so
module-level constants (`E = ...`) pick it up. The reader's folder names still say `licks-combined` (v36 naming); both
that and the four-camera name (`licks-fourcam`, what `combined_encoding.out_dir` gives on this lineage) map to the tag.

Rule 11: the render profile is applied (WIDEFIELD_REBUILD_ROOT active), WIDEFIELD_FROZEN_LOAD_ONLY=1, WFIELD_RUN_OUT =
this run's output folder; the driver additionally REFUSES any file opened for writing outside OUT (audit hook).

    python g4run.py [--v36] <script.py> [script args...]
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path, WindowsPath

OUT = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/figures_working/g4figs_ccB_20261125")
BUNDLE = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/o2/enc_fourcam3d_20261124")
V36 = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/"
           r"imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-"
           r"cls-contact-incomplete-stopA2jexrt3500c70_daqlk")
REPO = Path(__file__).resolve().parents[2]
REAL_ROOT = Path(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk")

#: reader folder (v36 naming, `licks-combined`) -> G4 tag
FOLDERS = {
    "movement_first_exec-polar_licks-combined": "main",
    "movement_first_exec-polar_licks-combined_nolicksall": "main_nolicksall",
    "movement_first_subset-nokin_exec-polar_licks-combined_bins": "bins_w1",
    "movement_first_subset-nokin_exec-polar_licks-combined_bins_bw0.5": "bins_w05",
    "movement_first_subset-nokin_exec-polar_licks-combined_bins_bw2": "bins_w2",
    "movement_first_subset-nokin_exec-xyz2_licks-combined_bins": "bins_w1_xyz2",
    "movement_first_subset-drop_exec-polar_licks-combined": "drop",
    "movement_first_subset-droppos-_exec-polar_licks-combined": "droppos",
    "movement_first_subset-shufpos-_exec-polar_licks-combined": "shufpos",
    "movement_first_subset-nodir_exec-polar_licks-combined": "nodir",
    "movement_first_subset-trmodels_exec-polar_licks-combined": "trmodels",
    "musall_subset-shuf_exec-polar_licks-combined": "musall_shuf",
    **{f"movement_first_fr-master_pre_v1_frg-{m}_subset-fr_exec-polar_licks-combined": f"fr_{m}" for m in "ABSD"},
    # running-model sensitivity rows (own names; same folder name as main under the code)
    "movement_first_exec-polar_licks-combined__main_spline": "main_spline",
    "movement_first_exec-polar_licks-combined__main_thr_pushhinge": "main_thr_pushhinge",
}
FOLDERS.update({k.replace("licks-combined", "licks-fourcam"): v for k, v in list(FOLDERS.items())})


def _sessions_dir(tag: str) -> Path:
    if V36_MODE:
        # v36 comparison: the collected v36 run of the same folder (read-only)
        name = next(k for k, v in FOLDERS.items() if v == tag and "licks-combined" in k)
        return V36 / name / "sessions"
    return BUNDLE / "out" / tag / "sessions"


V36_MODE = False
TAG = ("hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-"
       "stopA2jexrt3500c70_daqlk")
#: the run tag as the last folder (writeguard: a tagged run writes only under a folder named by its tag)
VIEW = OUT / "imaging_encoding_view" / TAG


class VP(WindowsPath):
    """A path under VIEW whose `<folder>/sessions` children route to the bundle (read side)."""

    def _make_child(self, args):
        return _route(super()._make_child(args))


def _route(p):
    try:
        rel = WindowsPath(str(p)).relative_to(VIEW)
    except ValueError:
        return p
    parts = rel.parts
    if parts and parts[0].startswith("frozen_encoder_"):
        # frozen frames (read-only here; the guard refuses writes outside OUT): the real four-camera encoding root
        return WindowsPath(REAL_ROOT, *parts)
    if len(parts) >= 2 and parts[0] in FOLDERS and parts[1].lower() == "sessions":
        return WindowsPath(_sessions_dir(FOLDERS[parts[0]]), *parts[2:])
    return VP(str(p))


def _guard(event, args):
    """Refuse a file opened for WRITING (or a mkdir / rename / remove) outside OUT."""
    if event == "open":
        path, mode = args[0], args[1]
        if not isinstance(path, (str, bytes, os.PathLike)) or isinstance(path, int):
            return
        if not isinstance(mode, str) or not any(c in mode for c in "wax+"):
            return
    elif event in ("os.mkdir", "os.remove", "os.rename", "os.replace", "os.rmdir", "shutil.rmtree"):
        path = args[0]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
    else:
        return
    p = os.path.abspath(os.fsdecode(path)).lower().replace("\\", "/")
    ok = [str(OUT).lower().replace("\\", "/"), os.environ.get("TEMP", "~~").lower().replace("\\", "/"),
          os.environ.get("TMP", "~~").lower().replace("\\", "/")]
    if p.startswith("nul") or any(p.startswith(o) for o in ok if o):
        return
    if "matplotlib" in p or "fontlist" in p or "__pycache__" in p or p.endswith(".pyc"):
        return
    raise PermissionError(f"g4run guard: write outside {OUT} refused: {p} ({event})")


def main() -> int:
    global V36_MODE, VIEW
    argv = sys.argv[1:]
    if argv and argv[0] == "--v36":
        V36_MODE, argv = True, argv[1:]
        VIEW = OUT / "imaging_encoding_view_v36" / TAG
    script, args = argv[0], argv[1:]
    sys.path[:0] = [str(REPO), str(REPO / "scripts")]
    from wfield_local import render_profile
    render_profile.apply("cohort1_rebuild_20261123_fourcam")
    os.environ["WIDEFIELD_FROZEN_LOAD_ONLY"] = "1"
    os.environ["WFIELD_RUN_OUT"] = str(OUT)
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    for k, v in json_env().items():                      # the bundle's encoder settings (as collect applies them)
        os.environ.setdefault(k, v)
    import wfield_local
    assert Path(wfield_local.__file__).resolve().is_relative_to(REPO), wfield_local.__file__
    from wfield_local import frozen_encoder as fe
    from wfield_local import fr_groups as fg
    from wfield_local import run_mode
    VIEW.mkdir(parents=True, exist_ok=True)
    run_mode.encoding_out_root = lambda run=None: VP(str(VIEW))
    fe.out_root = lambda: VP(str(VIEW))
    _orig_rf = fg.run_folder
    fg.run_folder = lambda root, mode, *a, **k: _route(_orig_rf(root, mode, *a, **k))   # Path(root) inside drops VP
    for name in FOLDERS:                                 # empty folder stubs so exists() checks pass
        if "licks-combined" in name and "__" not in name:
            (VIEW / name).mkdir(exist_ok=True)
    sys.addaudithook(_guard)
    print(f"g4run: wfield_local = {wfield_local.__file__}; view = {VIEW} ({'v36' if V36_MODE else 'G4'} sessions)",
          flush=True)
    sys.argv = [script, *args]
    runpy.run_path(script, run_name="__main__")
    return 0


def json_env() -> dict:
    import json
    env = json.loads((BUNDLE / "env.json").read_text())
    return {k: (str(BUNDLE / v[1:]) if isinstance(v, str) and v.startswith("@") else v) for k, v in env.items()}


if __name__ == "__main__":
    raise SystemExit(main())
