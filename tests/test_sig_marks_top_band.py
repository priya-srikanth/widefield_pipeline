"""Every significance star sits in the TOP BAND (Priya 10-30), never on the data; the top spine is hidden (11-03).

The 11-08 settings audit found `sig_marks.sig_star` still drawing the star beside the whiskers / animal dots, and the
frozen bar figures (partial_refit, session_vs_frozen, specificity_control, frozen_claims via `frozen_fr_compare.bars`)
on it. `sig_star` now routes to `star_top`; these tests pin that and that every figure module placing a star through
`sig_marks` also reserves the band (`reserve_headroom`, or `orofacial_cohort._tidy`). A static scan also fails any NEW
`ax.text` / `ax.annotate` of a bare star in data coordinates; `LEGACY_ON_DATA` lists the scripts that predate the rule
(not rendered for a page since; convert one and remove it from the list when it is next touched).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from wfield_local import sig_marks as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIRS = ("scripts", "wfield_local")

#: scripts with a bare star drawn in data coordinates, from before the 10-30 rule (11-08 scan). A matrix cell mark
#: (iti_pattern_area_stats: the star IS the cell annotation) is not an on-data star and stays allowed.
LEGACY_ON_DATA = {
    "scripts/fr_direction.py",
    "scripts/fr_direction_mechanism.py",
    "scripts/fr_encoding_summary.py",
    "scripts/iti_pattern_area_stats.py",
    "scripts/lick_balance_regions.py",
    "scripts/lick_correction.py",
    "scripts/lick_interhemispheric.py",
    "scripts/lick_lateral_ratio.py",
    "scripts/lick_recovery_markers.py",
    "scripts/lick_sspn_checks.py",
    "scripts/movement_null_plan_maps.py",
}


@pytest.fixture
def fig():
    f = plt.figure()
    yield f
    plt.close(f)


def test_sig_star_goes_to_the_top_band(fig):
    ax = fig.subplots()
    ax.plot([0, 1], [0, 5])
    t = sm.sig_star(ax, 0.5, 1.0, 4.0, [2.0, 5.0])
    assert t is not None
    assert t.get_transform() == ax.get_xaxis_transform()      # y in axes fraction, not data units
    assert t.get_position() == pytest.approx((0.5, 0.99))
    assert ax._wf_mark_rows == 1


def test_sig_star_decrease_also_top_and_none_when_ci_spans_zero(fig):
    ax = fig.subplots()
    t = sm.sig_star(ax, 1.0, -4.0, -1.0, [-5.0])
    assert t.get_position()[1] == pytest.approx(0.99)         # a decrease is marked at the top too
    assert sm.sig_star(ax, 2.0, -1.0, 1.0) is None


def test_reserve_headroom_is_idempotent(fig):
    ax = fig.subplots()
    ax.set_ylim(0, 1)
    sm.sig_star(ax, 0.0, 0.1, 0.9)
    sm.reserve_headroom(fig)
    once = ax.get_ylim()
    sm.reserve_headroom(fig)
    assert ax.get_ylim() == once
    assert not ax.spines["top"].get_visible()


def _sources():
    for d in FIG_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            yield p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8", errors="replace")


def test_every_star_user_reserves_the_band():
    """A module that places stars through sig_marks (or `frozen_fr_compare.bars`, which does) must reserve the band."""
    missing = []
    for rel, src in _sources():
        if rel == "wfield_local/sig_marks.py":
            continue
        uses = re.search(r"\b(sig_star|star_top)\(", src) or re.search(r"\bffc\.bars\(", src) \
            or ("from frozen_fr_compare import" in src and re.search(r"\bbars\(", src))
        if uses and not re.search(r"\breserve_headroom\b|\b_tidy\(", src):     # the name, not a call: some alias it
            missing.append(rel)
    assert not missing, f"stars placed without reserve_headroom (top band / no top spine): {missing}"


_STAR = re.compile(r"^(\*{1,4}|n\.?s\.?)$")


def _is_star(n) -> bool:
    if isinstance(n, ast.Constant) and isinstance(n.value, str):
        return bool(_STAR.match(n.value.strip()))
    if isinstance(n, ast.IfExp):
        return _is_star(n.body) or _is_star(n.orelse)
    if isinstance(n, ast.BoolOp):
        return any(_is_star(v) for v in n.values)
    return False


def _in_axes_coords(call: ast.Call) -> bool:
    for k in call.keywords:
        if k.arg in ("transform", "xycoords") and ast.unparse(k.value) not in ("ax.transData", "'data'"):
            return True
    return False


def test_no_new_star_drawn_on_the_data():
    found = set()
    for rel, src in _sources():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("text", "annotate")):
                continue
            s = (n.args[2] if len(n.args) >= 3 else None) if n.func.attr == "text" else (n.args[0] if n.args else None)
            if s is not None and _is_star(s) and not _in_axes_coords(n):
                found.add(rel)
    new = sorted(found - LEGACY_ON_DATA)
    assert not new, f"star drawn in data coordinates (use sig_marks.star_top + reserve_headroom): {new}"
