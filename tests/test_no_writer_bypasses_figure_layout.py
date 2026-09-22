"""No live module may build a FIGURE SIDECAR path by hand.

THE CHECK THIS COMPLEMENTS, AND WHY NEITHER IS ENOUGH ALONE.
`scripts/check_figure_layout.py` looks at the output tree, which is the strongest evidence there
is -- what is on disk cannot be evaded by phrasing. But **it can only see a writer that has RUN.**
On 2026-09-22 it found six bypassing writers and they were fixed; the same afternoon
`recovery_trajectory` ran for the first time since the migration and put eight more files flat,
and a scan then turned up three more in `reference_family_roi` that had not run at all. A tree
audit is blind to code that has not executed yet, and that blindness is invisible: the tool
reports a clean tree and is telling the truth about the tree.

So this is the static half. It goes blind the moment somebody spells the join a new way -- that
is why it is not the primary guard -- but it sees writers before they run, which the tree audit
never will.

DELIBERATELY NARROW. The rule is "a sidecar extension joined onto a directory, where the filename
follows the epoch/figure naming convention". A broader rule matched 69 lines, almost all of them
session artefacts in session directories (`U_atlas.npy`, `events.csv`, `gui_config.json`) -- a
guard that cries wolf is one people learn to silence.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A path literal ending in a sidecar extension, joined onto something that reads as an out dir.
JOIN = re.compile(r"""(?:out_dir|out|a\.out)\s*/\s*f?["'][^"']*\.(?:csv|svg|npz|json)["']""")
#: ...and named like a figure this repo registers, rather than like session data.
FIGURE_ISH = re.compile(r"\{stem\}|epoch_\d|epoch_[a-z]|recovery_trajectory|channel_position_maps")

#: Directories whose contents are frozen evidence, not live writers.
SKIP_PARTS = {"archive"}

#: `path:line` sites that match the pattern but do NOT write into a figure directory, each with
#: the reason. An entry is only valid while its `out` is genuinely not a figure dir — same policy
#: as `check_figure_layout.NOT_SIDECARS`, and asserted below rather than trusted.
#:
#: dlc_calibration — `out = Path(out_dir or cal_dir)` is the CAMERA CALIBRATION RECORDING
#:   directory. `charuco_survey_<rec>.csv/.txt` belong beside the recording they describe; there
#:   is no figure anywhere for them to be a sidecar of.
EXEMPT_SITES = {"wfield_local/dlc_calibration.py": "writes into the calibration recording dir"}

#: Lines that legitimately name the flat path.
def _exempt(line: str) -> bool:
    t = line.strip()
    return (
        t.startswith("#")
        # prose in a docstring, e.g. figure_layout's own explanation of the bug
        or "``" in t
        # the module that DEFINES the layout, and the tool that audits it
        or "figure_layout" in t
        # already routed: `fl.`, `fl2.`, `_fl.`
        or re.search(r"\b_?fl\d?\.", t) is not None
        # the fallback arm of `find_sidecar_for(...) or <flat path>`, which is the point of it
        or t.startswith("or ")
    )


def _live_sources():
    for root in (ROOT / "wfield_local", ROOT / "scripts"):
        for p in sorted(root.rglob("*.py")):
            if SKIP_PARTS & set(p.parts):
                continue
            yield p


def test_no_live_module_builds_a_figure_sidecar_path_by_hand():
    bad = []
    for p in _live_sources():
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if _exempt(line):
                continue
            rel = p.relative_to(ROOT).as_posix()
            if rel in EXEMPT_SITES:
                continue
            if JOIN.search(line) and FIGURE_ISH.search(line):
                bad.append(f"{rel}:{i}  {line.strip()[:90]}")
    assert not bad, (
        "these write a figure sidecar to the FLAT directory, where `find_sidecar` will not "
        "prefer it:\n  " + "\n  ".join(bad)
        + "\n\nUse `figure_layout.sidecar_for(out_dir, stem, suffix)`, and move the READER with "
          "the writer -- the read side already prefers `data/`, so changing only one of them "
          "makes the pipeline serve stale numbers with both files present and nothing erroring."
    )


def test_the_rule_actually_matches_the_pattern_it_is_meant_to_catch():
    """Guard the guard: if the regex stops matching, the test above passes on everything."""
    caught = "    q = out_dir / f\"epoch_99_something{tag}.csv\""
    assert JOIN.search(caught) and FIGURE_ISH.search(caught) and not _exempt(caught)
    # and a session artefact in a session directory must NOT be caught
    for ok in ('    U = np.load(ad / "U_atlas.npy")',
               '    ev = pd.read_csv(d / "events.csv")',
               '    q = fl.sidecar_for(out_dir, "epoch_17_x", ".csv")'):
        assert not (JOIN.search(ok) and FIGURE_ISH.search(ok) and not _exempt(ok)), ok


def test_every_exemption_names_a_file_that_still_exists():
    """An exemption for a deleted or renamed file silently widens the rule."""
    assert EXEMPT_SITES, "an empty exemption map would make this test vacuous"
    for rel in EXEMPT_SITES:
        assert (ROOT / rel).is_file(), f"{rel} is exempted but does not exist"
