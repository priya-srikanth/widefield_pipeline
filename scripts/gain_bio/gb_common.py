"""Shared paths for the GAIN_BIO chain (`scripts/gain_bio/`; deck slide 102, `fig4_gain_sessions.png`).

PROVENANCE: added 2026-11-05 with the port of M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/gain_bio_1104/
  (gain_bio.py -> summarise_gain_bio.py -> gain_stats.py -> fig_sessions.py; docs/status/GAIN_BIO_2026-11-04.md). The
  originals wrote beside themselves (HERE / "sessions", "tables", "figs"), read the adopted quiet-events sessions and the
  SD_RISE session list from sibling folders, and hard-coded the repo path. Here every one of those is ONE function:

  out_root()       GAIN_BIO_OUT, else lp_licks.output_dir(<cohort_out>/gain_bio): the default run -> <cohort_out>/gain_bio;
                   a tagged run (the render profile) -> <rebuild root>/labcams/<run tag>/gain_bio. Never the original
                   decoder_mix_checks/gain_bio_1104 folder.
  quiet_dir()      GAIN_BIO_QUIET_DIR, else THIS run's LP quiet-events v2 folder (the routed output of
                   scripts/cohort2_qc/quiet_spikes_v2.py run with QS_LICK_SOURCE=daq+lp:
                   quiet_area_spikes._out_root() / "v2_quiet_lp"; Priya 11-06 (2), was the DAQ "v2_quiet").
  session_table()  GAIN_BIO_SESSIONS (a CSV with label, animal, epoch), else <out>/sessions.csv when gain_bio.py has
                   written it, else default_sessions().

The settings travel by env var because `analysis_kit.fan_sessions` workers re-import the modules (spawn).

PRIYA'S 11-06 ANSWERS to the port's choices (docs/status/GAIN_BIO_2026-11-04.md section 8):
  (2) quiet events = the LP version (QUIET_SUBDIR / QUIET_LICK_SOURCE below), not the DAQ-contact one;
  (3) TASK = cue .. cue + 2 s of the CAMERA WORKING PERIOD cues only (`task_cues`, TASK_ROWS "period"), not every cue;
  (4) the 415 comparison is like with like: 415 (raw, 0.1 Hz HP) against raw 470 with the SAME high-pass (NONCA / CA_REF);
      the corrected signal is compared with raw 470 only as the correction check (CORR_CHECK), both high-passed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPS = ("pre", "acute", "subacute", "chronic")
#: the 107 SD_RISE / GAIN_BIO sessions = cohort-1 curated sessions from 08-06 on (August pre) minus PS93_0813, which has
#: no behaviour video (no camera working period: under the camera trial gate it has no engaged frame at all)
FIRST_MMDD = "0806"
NO_VIDEO = ("PS93_0813",)


def add_paths() -> None:
    """repo, scripts/ and scripts/cohort2_qc on sys.path (the original's `_paths`, without a machine path)."""
    for p in (str(REPO), str(SCRIPTS), str(SCRIPTS / "cohort2_qc")):
        if p not in sys.path:
            sys.path.insert(0, p)


def default_out() -> Path:
    from wfield_local import config, lp_licks
    return lp_licks.output_dir(Path(config.resolver().root("cohort_out")) / "gain_bio")


def out_root() -> Path:
    return Path(os.environ["GAIN_BIO_OUT"]) if os.environ.get("GAIN_BIO_OUT") else default_out()


#: quiet events (Priya 11-06 (2)): the LP version -- quiet_spikes_v2 with QS_LICK_SOURCE=daq+lp, so DAQ contact licks
#: AND cam4 / cam1 Lightning-Pose video licks are excluded from quiet frames; it writes <quiet_spikes root>/v2_quiet_lp
QUIET_LICK_SOURCE = "daq+lp"
QUIET_SUBDIR = "v2_quiet_lp"

#: TASK frame set (Priya 11-06 (3)): which cues open a cue .. cue + TASK_S window.
#:   "period"  the camera WORKING PERIOD (trial_classes.camera_period_trials: first cue -> last camera lick, minus the
#:             stopped runs; every class inside it, incl. no_response / jaw_only) -- the default
#:   "working" the camera working ROWS only (trial_classes.in_sel(cls, "working"): contact | incomplete under in_working)
#:   "all"     every cue (the 11-04 run; the engaged-frame gate then cut at the quit)
TASK_ROWS = "period"
TASK_ROWS_CHOICES = ("period", "working", "all")

#: the 415 comparison, like with like (Priya 11-06 (4)): the non-calcium channel NONCA against the calcium channel with
#: identical processing CA_REF (both halves of raw SVT.npy, same 0.1 Hz high-pass). CORR_CHECK = the hemo-correction
#: check (production corrected vs raw 470, both high-passed). The unfiltered corrected 'corr' is never set against 415.
NONCA = "r415_hp"
CA_REF = "r470_hp"
CORR_CHECK = ("corr_hp", "r470_hp")
GAIN_SIGS = ("corr_hp", "r470_hp", "r415_hp")        # fig 4 panel A / gain_stats A_gain, in this order


def default_quiet_dir() -> Path:
    add_paths()
    from quiet_area_spikes import _out_root
    return _out_root() / QUIET_SUBDIR


def quiet_dir() -> Path:
    return Path(os.environ["GAIN_BIO_QUIET_DIR"]) if os.environ.get("GAIN_BIO_QUIET_DIR") else default_quiet_dir()


def default_sessions():
    """DataFrame label, animal, epoch: curated cohort-1 sessions from FIRST_MMDD on with an epoch, minus NO_VIDEO."""
    import pandas as pd

    from wfield_local import epochs
    from wfield_local.analysis_kit import curated_labels
    rows = []
    for lab in sorted(curated_labels(list(ANIMALS))):
        ep = epochs.epoch_of(lab)
        if lab[:4] in ANIMALS and lab[-4:] >= FIRST_MMDD and ep in EPS and lab not in NO_VIDEO:
            rows.append(dict(label=lab, animal=lab[:4], epoch=ep))
    return pd.DataFrame(rows, columns=["label", "animal", "epoch"])


def task_rows() -> str:
    """GAIN_BIO_TASK_ROWS (--task-rows), else TASK_ROWS."""
    v = os.environ.get("GAIN_BIO_TASK_ROWS") or TASK_ROWS
    if v not in TASK_ROWS_CHOICES:
        raise ValueError(f"GAIN_BIO_TASK_ROWS={v!r}; expected one of {TASK_ROWS_CHOICES}")
    return v


def task_cues(label: str, cue_s, rows: str | None = None):
    """``(keep, note)``: which of the session's DAQ cues (``cue_s``, s, the FULL train) open a TASK window, under
    `task_rows` (see TASK_ROWS). A session without camera classes (no video) keeps NO cue under "period" / "working"
    -- never a DAQ fallback."""
    import numpy as np
    import pandas as pd

    from wfield_local import trial_classes as tc
    rows = rows or task_rows()
    if rows not in TASK_ROWS_CHOICES:
        raise ValueError(f"task rows {rows!r}; expected one of {TASK_ROWS_CHOICES}")
    cue_s = np.asarray(cue_s, float)
    if rows == "all":
        return np.ones(len(cue_s), bool), "all cues"
    if rows == "period":
        keep, _contact, has = tc.camera_period_trials(label, cue_s)
    else:
        full = tc.daq_cue_seconds(label)
        cls = tc.cue_classes_for(label, full)
        has = bool(np.any(cls != "unknown"))
        keep = tc.in_sel(tc.cue_classes(pd.DataFrame({"cue_daq": full, "cls": cls}), cue_s), "working")
    keep = np.asarray(keep, bool) if has else np.zeros(len(cue_s), bool)
    return keep, (f"camera {rows}: {int(keep.sum())}/{len(cue_s)} cues" if has else f"camera {rows}: no camera classes")


def session_table():
    import pandas as pd
    if os.environ.get("GAIN_BIO_SESSIONS"):
        return pd.read_csv(os.environ["GAIN_BIO_SESSIONS"])[["label", "animal", "epoch"]]
    f = out_root() / "sessions.csv"
    if f.exists():
        return pd.read_csv(f)[["label", "animal", "epoch"]]
    return default_sessions()


def guard(path: Path) -> Path:
    """mkdir + `writeguard.assert_writable` (refuses a share write of a tagged run outside its tag folder)."""
    from wfield_local.writeguard import assert_writable
    path = Path(path)
    assert_writable(path / "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


def common_args(ap) -> None:
    """--out / --quiet-dir / --sessions, exported to the env so spawned workers see them."""
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default: lp_licks.output_dir(<cohort_out>/gain_bio), run-tagged under a profile)")
    ap.add_argument("--quiet-dir", type=Path, default=None,
                    help="quiet-events v2 folder with sessions/*.npz and quiet_events_sessions.csv (default: this "
                         "run's LP quiet_spikes_v2 output, quiet_area_spikes._out_root()/v2_quiet_lp)")
    ap.add_argument("--sessions", type=Path, default=None,
                    help="CSV with label, animal, epoch (default: the 107-session rule in gb_common.default_sessions)")
    ap.add_argument("--task-rows", choices=TASK_ROWS_CHOICES, default=None,
                    help=f"which cues open a TASK window (default {TASK_ROWS!r}: the camera working period)")


def apply_args(a) -> None:
    for k, v in (("GAIN_BIO_OUT", a.out), ("GAIN_BIO_QUIET_DIR", getattr(a, "quiet_dir", None)),
                 ("GAIN_BIO_SESSIONS", getattr(a, "sessions", None)),
                 ("GAIN_BIO_TASK_ROWS", getattr(a, "task_rows", None))):
        if v is not None:
            os.environ[k] = str(v)
