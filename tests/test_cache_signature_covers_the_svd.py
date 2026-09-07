"""A preprocessing fix must invalidate the caches built on it.

WHY (2026-09-07). PS92 2026-08-28 was recorded with its 415/470 excitation channels swapped. The
fix landed 08-29 as a per-session `functional_channel: 0` override and regenerated `SVTcorr.npy`
and `U.npy` -- and nothing recomputed. `session_signature` hashed `locanmf_C`, the `.h5` and the
behaviour trials; none of those moved, so every cached ROI and pixel feature kept serving
swapped-channel values. The G8d SVD maps and every ROI-sourced result carried them for nine days.

`locanmf_C` IS NOT A PROXY for the SVD: it is downstream of `SVTcorr`, so it is stale whenever the
SVD is, but it is only REWRITTEN when someone re-runs LocaNMF -- which a preprocessing fix does not
do. One input changed and its hash was not in the key.
"""
import inspect

from wfield_local import session_cache


def _sig(tmp_path, monkeypatch, svt_bytes=b"a", u_bytes=b"a", c_bytes=b"a"):
    mc = tmp_path / "motion_corrected"
    (mc / "wfield_local_results" / "hemo_meegkit_hpfit").mkdir(parents=True, exist_ok=True)
    (mc / "loc").mkdir(parents=True, exist_ok=True)
    svt = mc / "wfield_local_results" / "hemo_meegkit_hpfit" / "SVTcorr.npy"
    u = mc / "wfield_local_results" / "U.npy"
    c = mc / "loc" / "L_locanmf_C.npy"
    svt.write_bytes(svt_bytes); u.write_bytes(u_bytes); c.write_bytes(c_bytes)
    monkeypatch.setattr("wfield_local.config.locanmf_dir", lambda m, variant=None: str(mc / "loc"))
    monkeypatch.setattr("wfield_local.config.svtcorr_path", lambda m, variant=None: str(svt))
    return session_cache.session_signature({"mc": str(mc), "label": "L", "h5": "", "behavior_trials": ""})


def test_a_changed_svtcorr_changes_the_signature(tmp_path, monkeypatch):
    a = _sig(tmp_path, monkeypatch, svt_bytes=b"before")
    b = _sig(tmp_path, monkeypatch, svt_bytes=b"after-the-channel-fix")
    assert a != b, ("regenerating SVTcorr did not move the signature, so every cached ROI and pixel "
                    "feature would go on serving pre-fix values")


def test_a_changed_U_changes_the_signature(tmp_path, monkeypatch):
    a = _sig(tmp_path, monkeypatch, u_bytes=b"before")
    b = _sig(tmp_path, monkeypatch, u_bytes=b"after")
    assert a != b, "U.npy is an SVD output and part of the identity"


def test_the_signature_names_the_svd_explicitly():
    src = inspect.getsource(session_cache.session_signature)
    assert "svtcorr_path" in src, "the SVD must be hashed via config.svtcorr_path, not reconstructed"
    assert "U.npy" in src
