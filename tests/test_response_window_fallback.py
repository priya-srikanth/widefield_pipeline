"""ONE fallback response window (code review 11-04 item 6, 2026-11-05).

Until 11-05 behavior (`spout_behavior`, via the 2.0 s `behavior.licking.response_window_s`) and imaging
(`nolick_decoder` 3.5, `quiet_periods` decode.max_rt_s 3.5) fell back to DIFFERENT windows for a session without a
gui_config, so the same word, "responded", could mean two things. Now every resolver goes through
`daq_trials.response_window_s`, whose fallback is `behavior.response_window_fallback_s`, and no production call passes its
own. On 11-05 every behavior-log folder (264) had its own timing.response_window, so the change moved no number.
"""
import ast
import json
from pathlib import Path

from wfield_local import config, daq_trials, nolick_decoder, quiet_periods

REPO = Path(__file__).resolve().parents[1]


def test_the_fallback_is_configured_and_is_the_task_window():
    assert config.defaults()["behavior"]["response_window_fallback_s"] == 3.5
    assert daq_trials.fallback_response_window_s() == 3.5


def test_every_resolver_falls_back_to_the_same_window(tmp_path):
    fb = daq_trials.fallback_response_window_s()
    assert daq_trials.response_window_s(tmp_path) == (fb, "default")                 # no gui_config at all
    (tmp_path / "gui_config.json").write_text(json.dumps({"timing": {"precue_min": "2000"}}))
    assert daq_trials.response_window_s(tmp_path) == (fb, "default")                 # gui_config without the key
    assert quiet_periods.response_window_s(tmp_path) == (fb, "default")
    assert quiet_periods.response_window_s(None) == (fb, "default")
    assert nolick_decoder.response_window_for({"label": "PS99_0101"}) == fb          # unresolvable session


def test_a_session_value_still_wins(tmp_path):
    (tmp_path / "gui_config.json").write_text(json.dumps({"timing": {"response_window": "3000"}}))
    assert daq_trials.response_window_s(tmp_path) == (3.0, "gui_config.json:timing")
    assert quiet_periods.response_window_s(tmp_path) == (3.0, "gui_config.json:timing")


def test_no_production_call_passes_its_own_fallback():
    bad = []
    for base in ("wfield_local", "scripts"):
        for p in (REPO / base).rglob("*.py"):
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                is_daq = name == "response_window_s" and (
                    (isinstance(f, ast.Attribute) and getattr(f.value, "id", "") in ("daq_trials", "dt"))
                    or p.name == "daq_trials.py")
                if (is_daq and (len(n.args) > 1 or any(k.arg == "default" for k in n.keywords))) or \
                        (name == "response_window_for" and (len(n.args) > 1 or n.keywords)):
                    bad.append(f"{p.relative_to(REPO).as_posix()}:{n.lineno}")
    assert not bad, bad
