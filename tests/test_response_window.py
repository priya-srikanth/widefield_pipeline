"""Consolidation A1 (Priya 11-03): no analysis hard-codes the response window; each session's own window is read from its
behavior log through `analysis_kit.response_window_s`."""
from wfield_local import analysis_kit as ak


def test_no_constant_window():
    assert ak.RESP_S is None          # was 2.0, never the task's window


def test_reader_is_the_session_log(monkeypatch):
    import wfield_local.nolick_decoder as nd
    monkeypatch.setattr(nd, "response_window_for", lambda s: {"PS92_0818": 3.5, "PS92_0512": 3.0}[s["label"]])
    assert ak.response_window_s({"label": "PS92_0818"}) == 3.5
    assert ak.response_window_s({"label": "PS92_0512"}) == 3.0


def test_rest_engagement_uses_the_one_reader(monkeypatch):
    from wfield_local import rest_engagement as re_
    monkeypatch.setattr(ak, "response_window_s", lambda s: 3.5)
    assert re_._response_window_s({"label": "PS92_0818"}) == 3.5
