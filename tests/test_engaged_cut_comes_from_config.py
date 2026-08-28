"""No module fixes the engaged cut with a literal. `decode.max_rt_s` is the single definition.

WHY IT MATTERS. `decode.max_rt_s` moved 2.0 -> 3.5 s on 2026-08-21, the task's REAL response window
read per session from `gui_config.json`. A module still cutting at 2.0 s files a lick at 2.5 s as
"no lick" when the task scored it as a REWARDED HIT -- so "engaged" means one thing there and another
in the analysis the module is a diagnostic FOR.

THE FAILURE IS SILENT BY CONSTRUCTION. A parameter sweep is internally consistent at either cut, so
nothing about its own output looks wrong; the disagreement only appears when its number is quoted
against a headline. `decoder_c_sweep`, `encoder_bins_test` and `filter_acausality_test` sat that way
for a week. The same literal was found and removed from `locanmf_frozen_decoder._args` on 2026-08-22.

ONE MODULE IS ALLOWED TO DIFFER, and only because it says why: `nolick_decoder` keeps 2.0 s
deliberately, since its whole subject is the trials between 2.0 s and the response window -- moving
its cut to 3.5 s would empty the `late_rewarded` category it exists to describe. Its docstring is the
model: it explains why 2.0 s is correct THERE rather than merely recording that it is used.
"""
import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "wfield_local"

#: Modules that may name a cut in code, each with the reason it is not the config value.
EXEMPT = {
    # the three-arm split (engaged / late_rewarded / undetected) is DEFINED by this boundary
    "nolick_decoder.py",
    "nolick_analysis.py",
    # the loader and the config module itself
    "config.py",
}


def _literal_max_rt(path):
    """`max_rt=<number>` / `max_rt_s=<number>` keyword literals at any call site in the file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:                                  # pragma: no cover
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg in ("max_rt", "max_rt_s") and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, (int, float)):
                    out.append((kw.arg, kw.value.value, kw.value.lineno))
    return out


@pytest.mark.parametrize("path", sorted(p for p in PKG.glob("*.py") if p.name not in EXEMPT),
                         ids=lambda p: p.name)
def test_the_engaged_cut_is_not_a_literal(path):
    bad = _literal_max_rt(path)
    assert not bad, (
        f"{path.name} hardcodes the engaged cut {bad}; read it from "
        f"config.defaults()['decode']['max_rt_s'] instead. If this module genuinely needs a "
        f"different boundary, say WHY in its docstring the way nolick_decoder._args does and add it "
        f"to EXEMPT -- the exemption is the documentation, not a way around the check.")


def test_the_sweeps_announce_the_cut_they_used():
    """Reading the config is not enough on its own: every number recorded for these modules was
    measured at 2.0 s, so a result has to carry its own boundary or it can be misquoted."""
    for name in ("decoder_c_sweep", "encoder_bins_test", "filter_acausality_test",
                 # found by the AST walk above, not by the by-hand survey that listed the other three
                 "postcue_window_test"):
        mod = __import__(f"wfield_local.{name}", fromlist=["_max_rt"])
        assert hasattr(mod, "_max_rt"), f"{name} has no _max_rt()"
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert "decode.max_rt_s" in src, f"{name} does not name the config key it reads"
        assert "measured at 2.0" in src, (
            f"{name} does not record that its published numbers predate the 3.5 s cut")


def test_the_exemption_carries_its_reason():
    """An exemption that does not explain itself is indistinguishable from the bug."""
    src = (PKG / "nolick_decoder.py").read_text(encoding="utf-8")
    assert "late_rewarded" in src and "2.0" in src, (
        "nolick_decoder is exempt because its three-arm split is defined by the 2.0 s boundary; "
        "if that is no longer true it should not be exempt")
