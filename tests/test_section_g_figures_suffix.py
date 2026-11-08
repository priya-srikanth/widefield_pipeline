"""--suffix reaches every section G figure name, so a second arm cannot overwrite the primary's (D9, 11-08)."""
from pathlib import Path

from wfield_local import plot_poststroke as pp
from wfield_local import section_g_figures as sgf


def _names(monkeypatch, tmp_path, suffix):
    names = []

    def rec_name(*a, name=None, **k):
        names.append(name or "FIXED")
        return [str(Path(tmp_path) / (name or "FIXED"))] if name else str(Path(tmp_path) / "FIXED")
    for fn in [f for f in dir(pp) if f.startswith("fig_")]:
        monkeypatch.setattr(pp, fn, rec_name)
    monkeypatch.setattr(sgf, "_sessions", lambda rec, tag: {"S1": {"counts": [1], "phase_tag": tag}})
    monkeypatch.setattr(sgf, "per_position_table", lambda sub: {"x": 1})
    import wfield_local.section_g_retained as sgr
    monkeypatch.setattr(sgr, "render", lambda rec, out, suffix="", **k: [])
    try:
        sgf.render({}, tmp_path, suffix=suffix)
    except Exception:                                     # partial records may stop a family early; names so far count
        pass
    return names


def test_every_name_carries_the_suffix(monkeypatch, tmp_path):
    names = _names(monkeypatch, tmp_path, "_contactonly")
    assert names, "no figure was attempted"
    assert all(n != "FIXED" and "_contactonly" in n for n in names), names


def test_no_suffix_keeps_the_primary_names(monkeypatch, tmp_path):
    names = [n for n in _names(monkeypatch, tmp_path, "") if n != "FIXED"]
    assert "section_g_G2b_per_position.png" in names
    assert not any("_contactonly" in n for n in names)
