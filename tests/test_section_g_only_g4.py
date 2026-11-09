"""`poststroke_section_g --only g4` (Priya 11-09, "only rerun what needs to be rerun"): recompute only the G4 camera-class
keys (cue / precue pools only) and merge them into the run's EXISTING section_g.json -- identical session labels
required, exactly `ONLY_KEYS["g4"]` replaced, a `g4_rerun` provenance key added, the old file backed up beside it
(never over an existing backup), written atomically. Without ``--only`` the runner is unchanged.
"""
import json

import pytest

from wfield_local import parallel
from wfield_local import poststroke_compare as pc
from wfield_local import poststroke_section_g as sg
from wfield_local import run_mode

OLD = {
    "PS92_0818": {"animal": "PS92", "label": "PS92_0818", "phase_tag": "post", "arms": {"all": {"x": 1}},
                  "looks_like_which": {"v": 0.3}, "g4_primary": "camera",
                  "g4_camclass_precue": {"frame": "camclass/z-contact/pos-contact+incomplete/neg-none", "v": 1},
                  "g4_camclass_cue": {"frame": "old", "v": 2}, "impaired_nolick": {"cue": 0.5}},
    "PS93_0818": {"animal": "PS93", "label": "PS93_0818", "phase_tag": "post", "arms": {"all": {"x": 2}},
                  "g4_primary": "camera", "g4_camclass_cue": {"frame": "old", "v": 3}},
}
NEW = {
    "PS92_0818": {"animal": "PS92", "label": "PS92_0818", "phase_tag": "post", "g4_primary": "camera",
                  "g4_camclass_precue": {"frame": "neg-no_response", "v": 10},
                  "g4_camclass_cue": {"frame": "neg-no_response", "v": 20}},
    # this session's cue G4 produced nothing this time: the old key must go, not linger
    "PS93_0818": {"animal": "PS93", "label": "PS93_0818", "phase_tag": "post", "g4_primary": "camera",
                  "g4_camclass_precue": {"frame": "neg-no_response", "v": 30}},
}


class _Now:
    def __init__(self, s):
        self.s = s

    def strftime(self, fmt):
        import datetime as dt
        return dt.datetime.strptime(self.s, "%Y%m%d-%H%M").strftime(fmt)


def _write(tmp_path, rec=OLD):
    p = tmp_path / "section_g__TAG.json"
    p.write_text(json.dumps(rec, indent=1))
    return p


def test_merge_replaces_exactly_the_g4_keys(tmp_path):
    p = _write(tmp_path)
    before = p.read_bytes()
    bak = sg.merge_only(p, json.loads(json.dumps(NEW)), "g4", provenance={"code_sha": "abc", "run_tag": "TAG"},
                        now=_Now("20261109-1230"))
    got = json.loads(p.read_text())
    assert bak.name == "section_g__TAG.bak_pre_g4_20261109-1230.json" and bak.read_bytes() == before
    assert set(got) == set(OLD)
    for lab, old in OLD.items():
        rest = {k: v for k, v in got[lab].items() if k not in sg.ONLY_KEYS["g4"] + ("g4_rerun",)}
        assert rest == {k: v for k, v in old.items() if k not in sg.ONLY_KEYS["g4"]}       # every other key equal
        for k in sg.ONLY_KEYS["g4"]:
            assert got[lab].get(k) == NEW[lab].get(k)
        pr = got[lab]["g4_rerun"]
        assert pr["code_sha"] == "abc" and pr["run_tag"] == "TAG" and pr["date"] == "2026-11-09 12:30"
        assert pr["keys"] == list(sg.ONLY_KEYS["g4"])
    assert "g4_camclass_cue" not in got["PS93_0818"]                     # dropped, not left stale
    assert not list(tmp_path.glob("*.tmp_*"))                            # atomic write left no temp file


def test_merge_refuses_a_label_mismatch(tmp_path):
    p = _write(tmp_path)
    before = p.read_bytes()
    new = {k: v for k, v in NEW.items() if k != "PS93_0818"}
    with pytest.raises(ValueError, match="session labels differ"):
        sg.merge_only(p, new, "g4", now=_Now("20261109-1230"))
    assert p.read_bytes() == before and not list(tmp_path.glob("*.bak_*"))


def test_merge_never_overwrites_a_backup(tmp_path):
    p = _write(tmp_path)
    bak = tmp_path / "section_g__TAG.bak_pre_g4_20261109-1230.json"
    bak.write_text("keep me")
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        sg.merge_only(p, NEW, "g4", now=_Now("20261109-1230"))
    assert bak.read_text() == "keep me" and p.read_bytes() == before


def test_merge_needs_an_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sg.merge_only(tmp_path / "nope.json", NEW, "g4")


@pytest.fixture
def mainenv(tmp_path, monkeypatch):
    from wfield_local import trial_classes as tc
    monkeypatch.setattr(run_mode, "routed_output", lambda p, **k: tmp_path)
    monkeypatch.setattr(run_mode, "keyed_name", lambda n, **k: n)
    monkeypatch.setattr(tc, "camera_gate_on", lambda: False)             # no design sidecar
    calls = []

    def _collect(animals=None, include_excluded=True, jobs=None, **kw):
        calls.append(kw)
        return json.loads(json.dumps(NEW if kw.get("only") else OLD))

    monkeypatch.setattr(sg, "collect", _collect)
    return tmp_path, calls


def test_default_main_is_unchanged(mainenv):
    out, calls = mainenv
    assert sg.main(["--output", str(out)]) == 0
    assert calls == [{}]                                                 # collect called exactly as before
    p = out / "section_g.json"
    assert p.read_text() == json.dumps(OLD, indent=1, default=float)     # byte-identical dump, no merge
    assert sorted(x.name for x in out.iterdir()) == ["section_g.json"]   # no backup, no temp file


def test_main_only_g4_merges(mainenv):
    out, calls = mainenv
    (out / "section_g.json").write_text(json.dumps(OLD, indent=1))
    assert sg.main(["--output", str(out), "--only", "g4"]) == 0
    assert calls == [{"only": "g4"}]
    got = json.loads((out / "section_g.json").read_text())
    assert got["PS92_0818"]["g4_camclass_cue"]["v"] == 20 and got["PS92_0818"]["arms"] == OLD["PS92_0818"]["arms"]
    pr = got["PS92_0818"]["g4_rerun"]
    assert pr["train_classes"] == {"positive": ["contact", "incomplete"], "negative": ["no_response"]}
    assert pr["g4_frame"].endswith("neg-no_response") and pr["run_tag"] == "section_g"
    assert len(list(out.glob("section_g.bak_pre_g4_*.json"))) == 1


def test_main_only_g4_refuses_before_work_without_a_file(mainenv):
    out, calls = mainenv
    with pytest.raises(SystemExit):
        sg.main(["--output", str(out), "--only", "g4"])
    assert calls == []


def _fake_fan_out(seen):
    def fan_out(items, worker, *, jobs=None, label="unit", log=print):
        items = list(items)
        seen.append((label, items))
        if label == "pool":
            return [((an, al, inc), {"post": (["A_0", "A_1"], [1], None)} if al == "cue" else {})
                    for an, al, inc in items], []
        return [(u, None) for u in items], []
    return fan_out


def test_collect_units(monkeypatch):
    seen = []
    monkeypatch.setattr(parallel, "fan_out", _fake_fan_out(seen))
    sg.collect(["PS92", "PS93"], jobs=1)
    (_, warm), (_, units) = seen
    assert sorted({al for _an, al, _i in warm}) == ["cue", "lick", "precue"] and len(warm) == 6
    assert units == [("PS92", "post", None, 1), ("PS93", "post", None, 1)]      # 4-tuples, as before
    seen.clear()
    sg.collect(["PS92", "PS93"], jobs=1, only="g4")
    (_, warm), (_, units) = seen
    assert sorted({al for _an, al, _i in warm}) == ["cue", "precue"] and len(warm) == 4
    assert units == [("PS92", "post", None, 1, "g4"), ("PS93", "post", None, 1, "g4")]


@pytest.fixture
def readouts(monkeypatch):
    monkeypatch.setattr(pc, "looks_like_which", lambda d, keep: {"llw": 1})
    monkeypatch.setattr(pc, "fits_engaged_distribution", lambda d, keep: {"fe": 1})
    monkeypatch.setattr(pc, "g4_disc", lambda: "camera")
    monkeypatch.setattr(pc, "g4_camclass", lambda d, keep: {"g4": sorted(d["post_i"])})
    monkeypatch.setattr(pc, "impaired_nolick_readout", lambda d, keep, alignment=None: {"imp": alignment})
    monkeypatch.setattr(pc, "preserved_detail", lambda d, i: {"keep": [0, 1], "rule": "r", "counts": {},
                                                               "min_post": 5})
    return {a: {"post_i": {0, 1}, "kept": ["A_0", "A_1"]} for a in ("cue", "precue", "lick")}


def test_nolick_readouts_keep_their_key_order(readouts):
    out = sg._nolick_readouts(readouts, 1, [0, 1])
    assert list(out) == ["looks_like_which", "fits_engaged_precue", "fits_engaged_cue", "g4_primary",
                         "g4_camclass_precue", "g4_camclass_cue", "impaired_nolick"]
    assert out["g4_camclass_cue"] == {"g4": [1]}


def test_run_session_only_g4(readouts):
    rec = sg.run_session(readouts, "A", 1, "A_1", "post", only="g4")
    assert rec == {"animal": "A", "label": "A_1", "phase_tag": "post", "g4_primary": "camera",
                   "g4_camclass_precue": {"g4": [1]}, "g4_camclass_cue": {"g4": [1]}}
