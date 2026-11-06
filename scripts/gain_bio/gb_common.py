"""Shared paths for the GAIN_BIO chain (`scripts/gain_bio/`; deck slide 102, `fig4_gain_sessions.png`).

PROVENANCE: added 2026-11-05 with the port of M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/gain_bio_1104/
  (gain_bio.py -> summarise_gain_bio.py -> gain_stats.py -> fig_sessions.py; docs/status/GAIN_BIO_2026-11-04.md). The
  originals wrote beside themselves (HERE / "sessions", "tables", "figs"), read the adopted quiet-events sessions and the
  SD_RISE session list from sibling folders, and hard-coded the repo path. Here every one of those is ONE function:

  out_root()       GAIN_BIO_OUT, else lp_licks.output_dir(<cohort_out>/gain_bio): the default run -> <cohort_out>/gain_bio;
                   a tagged run (the render profile) -> <rebuild root>/labcams/<run tag>/gain_bio. Never the original
                   decoder_mix_checks/gain_bio_1104 folder.
  quiet_dir()      GAIN_BIO_QUIET_DIR, else THIS run's quiet-events v2 folder (the routed output of
                   scripts/cohort2_qc/quiet_spikes_v2.py: quiet_area_spikes._out_root() / "v2_quiet").
  session_table()  GAIN_BIO_SESSIONS (a CSV with label, animal, epoch), else <out>/sessions.csv when gain_bio.py has
                   written it, else default_sessions().

The settings travel by env var because `analysis_kit.fan_sessions` workers re-import the modules (spawn).
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


def default_quiet_dir() -> Path:
    add_paths()
    from quiet_area_spikes import _out_root
    return _out_root() / "v2_quiet"


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
                         "run's quiet_spikes_v2 output, quiet_area_spikes._out_root()/v2_quiet)")
    ap.add_argument("--sessions", type=Path, default=None,
                    help="CSV with label, animal, epoch (default: the 107-session rule in gb_common.default_sessions)")


def apply_args(a) -> None:
    for k, v in (("GAIN_BIO_OUT", a.out), ("GAIN_BIO_QUIET_DIR", getattr(a, "quiet_dir", None)),
                 ("GAIN_BIO_SESSIONS", getattr(a, "sessions", None))):
        if v is not None:
            os.environ[k] = str(v)
