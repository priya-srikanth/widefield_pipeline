"""`--rest-variant` of the interhemispheric / cross-area / coupling family (Priya 11-09, RESTNL_ROLLOUT_PLAN category (d)).

PRIMARY = restnl: the docked restdock05 rest frames INTERSECTED with the RESTnl lick-free mask (restnl_mask.lickfree_mask);
SUPPLEMENT = restdock05 (the default, every path unchanged); RAW = the lick-set metrics with no rest reference.
Pinned here: (1) the default resolves to the original paths / names; (2) restnl outputs never collide with them;
(3) a frame next to a CAMERA-ONLY lick is kept under restdock05 and dropped under restnl (synthetic masks, the real
accessor `restnl_mask.lickfree_mask` with its file reads stubbed); (4) the raw outputs do not depend on the rest rows.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
MODS = ("lick_interhemispheric", "interhemi_crossarea", "coupling_vigour_sm")


def _load(monkeypatch, variant: str | None):
    argv = ["x"] + ([] if variant is None else ["--rest-variant", variant])
    monkeypatch.setattr(sys, "argv", argv)
    out = []
    for m in MODS:
        out.append(importlib.reload(importlib.import_module(m)) if m in sys.modules else importlib.import_module(m))
    return out


@pytest.fixture
def default_after(monkeypatch):
    """Leave the modules imported in their DEFAULT state for whatever test runs next."""
    yield
    monkeypatch.setattr(sys, "argv", ["x"])
    for m in MODS:
        if m in sys.modules:
            importlib.reload(sys.modules[m])


def _names(src: str) -> list[str]:
    return re.findall(r'_o\(f?"([^"]+)"\)', src)


def test_default_is_restdock05_with_unchanged_paths(monkeypatch, default_after):
    lih, ica, cv = _load(monkeypatch, None)
    assert lih.REST_VARIANT == "restdock05" and lih.VSFX == ""
    assert lih.REST_DIR == lih.OUT_W / "interhemi_rest_w1"
    for n in _names((REPO / "scripts" / "lick_interhemispheric.py").read_text(encoding="utf-8")):
        assert lih._o(n) == n
    assert (ica.O1.name, ica.O2.name) == ("interhemi_followup", "cross_area")
    assert (cv.O1.name, cv.O2.name) == ("coupling_vigour", "sensorimotor_kinetics")
    assert "SUPPLEMENT" in lih.REST_TXT and "restdock05" in lih.REST_TXT and "DAQ licks excluded" in lih.REST_TXT


def test_restnl_outputs_are_separate(monkeypatch, default_after):
    names = _names((REPO / "scripts" / "lick_interhemispheric.py").read_text(encoding="utf-8"))
    assert len(names) >= 20
    lih, ica, cv = _load(monkeypatch, "restnl")
    assert lih.REST_VARIANT == "restnl"
    assert lih.REST_DIR == lih.OUT_W / "interhemi_rest_w1_restnl"
    tagged = {lih._o(n) for n in names}
    assert not tagged & set(names), "a restnl output name equals a default one"
    assert lih._o("lick_interhemi_lead_tau0.6.png") == "lick_interhemi_lead_tau0.6_restnl.png"
    assert lih._o("interhemi_sessions.csv.gz") == "interhemi_sessions_restnl.csv.gz"
    assert (ica.O1.name, ica.O2.name) == ("interhemi_followup_restnl", "cross_area_restnl")
    assert (cv.O1.name, cv.O2.name) == ("coupling_vigour_restnl", "sensorimotor_kinetics_restnl")
    assert lih.REST_TXT == "rest = docked quiet rest, lick-free (RESTnl: no camera or DAQ licks)"
    # the raw outputs are variant-free on purpose and never collide with a tagged / default name
    assert not set(lih.RAW_NAMES.values()) & (tagged | set(names))


def test_bad_variant_refused():
    import lick_interhemispheric as lih
    with pytest.raises(SystemExit):
        lih.rest_variant(["x", "--rest-variant", "restw"])
    assert lih.rest_variant(["x"]) == "restdock05"


def test_every_writer_takes_the_flag():
    for m in MODS:
        src = (REPO / "scripts" / f"{m}.py").read_text(encoding="utf-8")
        assert '"--rest-variant"' in src, m
    for m in MODS[1:]:
        src = (REPO / "scripts" / f"{m}.py").read_text(encoding="utf-8")
        assert 'lih._o("' in src and "CAP_REST" in src, m


# ------------------------------------------------------------------------------------------------ the intersection
FS = 100.0
N = 2000                               # 20 s of samples; sample index == frame index in the fake frame map
CAM_LICK_S = 10.0                      # a camera-only lick (no DAQ contact)
DAQ_LICK_S = 15.0


def _stub(monkeypatch, tmp_path):
    from wfield_local import analysis_kit as ak
    from wfield_local import rest_by_position as rbp
    from wfield_local import restnl_mask as rnm
    rest0 = np.ones(N, bool)           # the restdock05 rest sample mask (already DAQ-contact clean around nothing here)
    eng = np.array([True, True])
    parts = (rest0, np.array([0, 1000]), np.array([1, 2]), np.arange(N) / FS, np.arange(N) * 50, None)
    seen = {}

    def fake_frames(s, n_frames, *, docked=False, engaged_only=True):
        rest, cs, codes, ts, fs_samp, sync = rbp._session_daq(s)
        seen["docked"], seen["eng"] = docked, rbp._engaged_trials(s, cs, codes)
        return {0: np.flatnonzero(rest)}, {"n_periods": 1, "reconstructed": False}

    la = tmp_path / "licks_all.csv"
    pd.DataFrame({"daq_on_s": [CAM_LICK_S]}).to_csv(la, index=False)
    monkeypatch.setattr(rbp, "_session_daq", lambda s: parts)
    monkeypatch.setattr(rbp, "_engaged_trials", lambda *a, **k: eng)
    monkeypatch.setattr(rbp, "rest_frames_by_position", fake_frames)
    monkeypatch.setattr(ak, "daq_rate", lambda s: FS)
    monkeypatch.setattr(rnm, "daq_lick_onsets", lambda s, fs: np.array([int(DAQ_LICK_S * FS)]))
    monkeypatch.setattr(rnm, "licks_all_path", lambda label: la)
    return rbp, seen, eng


def test_camera_only_lick_dropped_under_restnl_kept_under_restdock05(monkeypatch, tmp_path):
    import lick_interhemispheric as lih
    rbp, seen, eng = _stub(monkeypatch, tmp_path)
    s = {"label": "PS92_0901"}
    per0, _ = lih.docked_rest_frames(s, rbp._session_daq(s))                 # restdock05 (default path)
    assert seen["docked"] is True
    got, info = lih._restnl_daq(s)
    per1, _ = lih.docked_rest_frames(s, got, eng=info.pop("_eng"))
    assert seen["docked"] is True and seen["eng"] is eng
    f0, f1 = set(per0[0].tolist()), set(per1[0].tolist())
    cam, daq = int(CAM_LICK_S * FS), int(DAQ_LICK_S * FS)
    assert cam in f0 and cam not in f1                                         # camera-only lick: the point of RESTnl
    assert daq not in f1
    assert cam - int(0.5 * FS) - 1 in f1 and cam + int(2.0 * FS) + 1 in f1     # the [0.5, 2.0] s buffer, nothing more
    assert f1 < f0                                                             # an intersection: a strict subset
    assert info["nl_n_cam4_licks"] == 1 and info["nl_n_daq_licks"] == 1


def test_restnl_without_licks_all_is_excluded_not_a_daq_fallback(monkeypatch, tmp_path):
    import lick_interhemispheric as lih
    from wfield_local import restnl_mask as rnm
    _stub(monkeypatch, tmp_path)
    monkeypatch.setattr(rnm, "licks_all_path", lambda label: tmp_path / "absent.csv")
    got, why = lih._restnl_daq({"label": "PS93_0813"})
    assert got is None and "licks_all" in why


def test_default_path_never_touches_the_restnl_accessor():
    import lick_interhemispheric as lih
    assert lih.REST_VARIANT == "restdock05"
    src = (REPO / "scripts" / "lick_interhemispheric.py").read_text(encoding="utf-8")
    body = src.split("def rest_extract(", 1)[1].split("\ndef ", 1)[0]
    assert 'if REST_VARIANT == "restnl":' in body and "_restnl_daq(s)" in body


# ------------------------------------------------------------------------------------------------ raw (no rest reference)
def _sessions(rest_shift: float) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for a in ("PS92", "PS93", "PS94", "PS95"):
        for k, ep in enumerate(["pre"] * 3 + ["acute"] * 2 + ["subacute"] * 2 + ["chronic"] * 3):
            lab = f"{a}_{(805 + k * 3):04d}" if ep == "pre" else f"{a}_{(900 + k * 3):04d}"
            for st in ("cued", "iti_all", "rest"):
                v = rng.normal(size=10) + (rest_shift if st == "rest" else 0.0)
                rows.append({"label": lab, "animal": a, "epoch": ep, "set": st, "win": "w1", "pair": "oro", "tau": 0.0,
                             "n": 40, **dict(zip(["nc_post", "nc_pre", "nc_win", "cc0", "cc_peak", "lag_peak", "asym",
                                                  "gc_ci", "gc_ic", "gc_dir"], v))})
    return pd.DataFrame(rows)


def test_raw_outputs_do_not_depend_on_the_rest_rows(monkeypatch, tmp_path):
    import lick_interhemispheric as lih
    S = _sessions(0.0)
    labs = sorted(set(S.label))
    pd.DataFrame({"label": labs, "day": [int(x[-3:]) - 890 for x in labs],
                  "aim": np.linspace(-1, 1, len(labs))}).to_csv(tmp_path / "recmark_sessions.csv", index=False)
    pd.DataFrame({"label": labs, "set": "trial", "bal_spout": np.linspace(0, 1, len(labs))}).to_csv(
        tmp_path / "balmatch_sessions.csv", index=False)
    monkeypatch.setattr(lih, "IN", tmp_path)
    monkeypatch.setattr(lih, "N_BOOT", 200)
    out = {}
    for shift in (0.0, 3.0):
        d = tmp_path / f"o{shift}"
        d.mkdir()
        monkeypatch.setattr(lih, "OUT_W", d)
        lih.raw_outputs(_sessions(shift))
        out[shift] = {k: (d / lih.RAW_NAMES[k]).read_bytes() for k in ("tests", "recovery")}
    assert out[0.0] == out[3.0]
    T = pd.read_csv(tmp_path / "o0.0" / lih.RAW_NAMES["tests"])
    assert set(T.set) == {"cued", "iti_all"} and (T.rest_ref == "raw (no rest reference)").all() and len(T)
