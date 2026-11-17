"""THE REBUILD GLUE-MASK BUG (found 11-08 by the state-space arm, fixed 11-09).

Under the render profile ``cohort1_rebuild_20261103`` `paint_exclusion.mask_dir` followed the rebuild root, where the
hand-painted fiber-glue masks do not exist, and `beta_maps.painted_exclusion` swallowed that and returned an all-False
mask: every brain / stat mask kept the glue (stat mask 166,974 px instead of 129,770). These tests pin the fix:

* the masks resolve to the ORIGINAL grant_figures under the profile, through the fence (union = 40,663 px);
* a missing / unreadable / mis-shaped / empty mask RAISES; the only all-False answer is the explicit cohort opt-out;
* the encoder input cache key carries the masks' content (`beta_maps.mask_signature`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROFILE = "cohort1_rebuild_20261103"
UNION_PX = 40_663
STAT_PX = 129_770


def _share_masks_present() -> bool:
    try:
        from wfield_local.paths import PathResolver
        return (Path(PathResolver().input_root("cohort_out")) / "grant_figures"
                / "exclusion_mask_painted_union.npy").exists()
    except Exception:                                                  # noqa: BLE001
        return False


PROBE = r"""
import json, os
from wfield_local import render_profile
render_profile.apply("%s")
os.environ["WIDEFIELD_REBUILD_ROOT"] = %r                       # the profile's root, pointed at a scratch folder
from wfield_local import config, beta_maps, paint_exclusion
rv = config.resolver()
out = {"mask_dir": paint_exclusion.mask_dir().as_posix(), "input": rv.input_root("cohort_out"),
       "cohort_out": rv.root("cohort_out"),
       "union": int(beta_maps.painted_exclusion().sum()),
       "ps92": int(beta_maps.painted_exclusion("PS92").sum()),
       "stat": int(beta_maps.stat_mask().sum()), "brain": int(beta_maps.brain_mask().sum()),
       "sig": beta_maps.mask_signature()}
try:                                                             # the rest of grant_figures stays fenced
    open(rv.input_root("cohort_out") + "/grant_figures/epoch/data/zz_absent.csv")
    out["other"] = "READ"
except PermissionError:
    out["other"] = "refused"
except FileNotFoundError:
    out["other"] = "not fenced"
print("JSON" + json.dumps(out))
"""


def _run(code, tmp_path, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_"))}
    env.update(env_extra or {})
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path), capture_output=True, text=True,
                          timeout=300)


@pytest.mark.skipif(not _share_masks_present(), reason="needs the painted masks on the share")
def test_masks_resolve_to_the_original_grant_figures_under_the_rebuild_profile(tmp_path):
    r = _run(PROBE % (PROFILE, (tmp_path / "rebuild").as_posix()), tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.split("JSON", 1)[1])
    assert out["mask_dir"].lower() == (out["input"].replace("\\", "/") + "/grant_figures").lower()
    assert "rebuild" in out["cohort_out"].replace("\\", "/").lower()        # derived outputs still go to the root
    assert out["union"] == UNION_PX, out
    assert out["ps92"] > 0
    assert out["stat"] == STAT_PX, out                                    # 166,974 with the glue in
    assert out["other"] == "refused", out                                 # only the mask files are opened up
    # the same masks with no profile at all
    r0 = _run("import json; from wfield_local import beta_maps as b; print('JSON' + json.dumps("
              "[int(b.painted_exclusion().sum()), int(b.stat_mask().sum()), b.mask_signature()]))", tmp_path)
    assert r0.returncode == 0, r0.stderr[-3000:]
    union0, stat0, sig0 = json.loads(r0.stdout.split("JSON", 1)[1])
    assert (union0, stat0, sig0) == (out["union"], out["stat"], out["sig"])


def test_fence_allows_exactly_the_painted_mask_files(monkeypatch):
    from wfield_local import rebuild_root as rr
    lab = "m:/microscope/priya/widefield/labcams/"
    ok = rr._labcams_input(lab + "grant_figures/exclusion_mask_painted_union.npy", lab)
    assert ok
    assert rr._labcams_input(lab + "grant_figures/exclusion_mask_painted_ps92.npy", lab)
    assert not rr._labcams_input(lab + "grant_figures/epoch/data/x.csv", lab)
    assert not rr._labcams_input(lab + "grant_figures", lab)               # no listing of the folder
    assert not rr._labcams_input(lab + "frozen_models/x.pkl", lab)
    assert rr._labcams_input(lab + "20260806/x", lab)                      # session data still an input


def test_load_mask_raises_instead_of_returning_empty(tmp_path, monkeypatch):
    from wfield_local import paint_exclusion as pe
    monkeypatch.delenv("WIDEFIELD_COHORT", raising=False)
    shape = (8, 10)
    with pytest.raises(FileNotFoundError):
        pe.load_mask("union", shape, out_dir=tmp_path)                   # missing
    np.save(pe.mask_path("union", tmp_path), np.ones((4, 4), bool))
    with pytest.raises(ValueError, match="shape"):
        pe.load_mask("union", shape, out_dir=tmp_path)                   # wrong shape
    np.save(pe.mask_path("union", tmp_path), np.zeros(shape, bool))
    with pytest.raises(ValueError, match="empty"):
        pe.load_mask("union", shape, out_dir=tmp_path)                   # empty on disk
    pe.mask_path("union", tmp_path).write_bytes(b"not a numpy file")
    with pytest.raises(ValueError, match="unreadable"):
        pe.load_mask("union", shape, out_dir=tmp_path)                   # unreadable
    m = np.zeros(shape, bool)
    m[2:4, 3:6] = True
    np.save(pe.mask_path("union", tmp_path), m)
    assert np.array_equal(pe.load_mask("union", shape, out_dir=tmp_path), m)


def test_painted_exclusion_raises_when_the_mask_dir_is_wrong(tmp_path, monkeypatch):
    """The bug itself: a mask directory with nothing in it must STOP a reader, not hand it an empty mask."""
    from wfield_local import beta_maps as bm
    from wfield_local import paint_exclusion as pe
    monkeypatch.delenv("WIDEFIELD_COHORT", raising=False)
    monkeypatch.setattr(pe, "mask_dir", lambda out_dir=None: Path(out_dir) if out_dir else tmp_path)
    bm.painted_exclusion.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            bm.painted_exclusion()
        from wfield_local import component_exclusion as ce
        with pytest.raises(FileNotFoundError):
            ce.excluded_pixels("PS92")
    finally:
        bm.painted_exclusion.cache_clear()


def test_cohort_opt_out_is_explicit(tmp_path, monkeypatch):
    from wfield_local import paint_exclusion as pe
    monkeypatch.setenv("WIDEFIELD_COHORT", "cohort2")
    assert pe.painted_policy() == "none"
    assert not pe.load_mask("union", (8, 10), out_dir=tmp_path).any()   # no file needed, all-False on purpose
    monkeypatch.setenv("WIDEFIELD_COHORT", "cohort1")
    assert pe.painted_policy() == "required"


def test_encoder_input_cache_key_follows_the_mask_content(tmp_path, monkeypatch):
    from wfield_local import beta_maps as bm
    from wfield_local import combined_encoding as ce
    monkeypatch.setenv(ce.CACHE_ENV, str(tmp_path))
    builds = []
    monkeypatch.setattr(bm, "mask_signature", lambda n=12: "glue_in")
    ce._cached("imaging", "lbl", [], lambda: builds.append(1) or "A")
    assert ce._cached("imaging", "lbl", [], lambda: builds.append(1) or "B") == "A"    # same masks: a hit
    monkeypatch.setattr(bm, "mask_signature", lambda n=12: "glue_out")
    assert ce._cached("imaging", "lbl", [], lambda: builds.append(1) or "C") == "C"    # new masks: a miss
    assert len(builds) == 2 and len(list((tmp_path / "imaging").glob("lbl_*.pkl"))) == 2


def test_mask_signature_changes_with_the_painted_mask(monkeypatch):
    from wfield_local import beta_maps as bm
    if bm.allen_mask() is None:
        pytest.skip("no Allen mask on this machine")
    shape = bm.MAP_SHAPE
    glue = np.zeros(shape, bool)
    glue[200:260, 300:380] = True
    monkeypatch.setattr(bm, "painted_exclusion", lambda key="union": glue)
    a = bm.mask_signature()
    glue2 = glue.copy()
    glue2[100:120, 300:320] = True
    monkeypatch.setattr(bm, "painted_exclusion", lambda key="union": glue2)
    assert bm.mask_signature() != a


def test_iti_extract_cache_refuses_other_masks(tmp_path, monkeypatch):
    """`iti_licks.extract` is skip-if-exists on a path that does not carry the masks: the stored mask_sig must match."""
    sys.path.insert(0, str(REPO / "scripts"))
    import iti_licks as il
    from wfield_local import rebuild_root as rr
    f = tmp_path / "x.npz"
    monkeypatch.setattr(il, "_mask_sig", lambda: "glue_out")
    np.savez(f, mask_sig="glue_out")
    il.check_extract_masks(f)                                             # same masks: reused
    np.savez(f, mask_sig="glue_in")
    with pytest.raises(RuntimeError, match="masks"):
        il.check_extract_masks(f)                                         # other masks: refused
    np.savez(f, cols=np.array(["a"]))                                     # pre-11-09 file, no mask_sig
    monkeypatch.delenv(rr.ENV, raising=False)
    il.check_extract_masks(f)                                             # default root: built with the glue out
    monkeypatch.setattr(rr, "active", lambda: True)
    with pytest.raises(RuntimeError, match="before the 11-09"):
        il.check_extract_masks(f)                                         # rebuild root: glue in -> refused
