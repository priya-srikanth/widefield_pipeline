"""Test-suite guards.

THE SHARE WRITE GUARD. On 2026-08-22 a phantom session turned up in the cross-day intensity scan:

    N:/MICROSCOPE/Priya/Widefield/labcams/20260820/PS92_20260820_120000/
        motion_corrected/wfield_local_results/SVTcorr.npy      (1 byte)

One byte, because it was a test helper's ``_touch()``. The test patched the subprocess runner but
not the resolver, so ``preprocess_session``'s push step did real filesystem work against the live
MICROSCOPE root, re-creating the directory on every run of the suite. It was a LEGAL write -- inside
Priya's subtree, so ``writeguard`` allowed it, correctly -- which is why nothing caught it.

writeguard answers "may this code write here?". This answers a different question: "may a TEST write
anywhere real?" The answer is no, and it has to be enforced at the filesystem call, because the way
this happened was a test reaching production paths without ever naming one.

THE SESSION CACHE IS THE SAME QUESTION ON A LOCAL DISK. `session_cache.CACHE_DIR` resolves to
`E:/.widefield_session_cache` on this box -- shared lab state, but not a UNC share, so the guard
above never looked at it. On 2026-08-28 a NameError inside a bootstrap being refactored was
swallowed, and running the SUITE wrote twelve `None` entries into that production cache, which the
corrected code then read back and turned into empty figures. A `delta_test-delta` entry was sitting
in it too. The keys are content digests, so synthetic fixture data could never collide with a real
session -- but that is luck, not isolation. Tests now get their own cache directory.
"""
from __future__ import annotations

import builtins
import os
import pathlib
import shutil

import pytest

#: Prefixes that are live data, never fixtures. Checked on a normalised, upper-cased path.
SHARE_MARKS = ("N:/", "M:/", "//RESEARCH.FILES", "//STANDBY.FILES", "/MICROSCOPE/")

_WRITE_MODES = set("wxa+")


def _is_share(path) -> bool:
    try:
        p = os.fspath(path)
    except TypeError:
        return False
    u = str(p).replace("\\", "/").upper()
    return any(m in u for m in SHARE_MARKS)


def _refuse(path, how: str):
    raise AssertionError(
        f"a test tried to {how} a live data share:\n    {path}\n\n"
        f"Redirect PathResolver at tmp_path instead:\n"
        f"    monkeypatch.setattr(PathResolver, 'root', lambda self, n: str(tmp_path / n))\n"
        f"    monkeypatch.setattr(PathResolver, 'resolve', lambda self, n, r: str(tmp_path / n / r))\n"
        f"Patching the subprocess runner is NOT enough: the push step does its own filesystem work, "
        f"and a write inside Priya's subtree is legal, so writeguard will not stop it.")


@pytest.fixture(autouse=True)
def no_share_writes(monkeypatch):
    """Fail any test that opens-for-write, mkdirs, copies or removes on a real share.

    READS are left alone on purpose: a few tests legitimately read configs or check that a path
    resolver produces the right string. It is writing that had the side effect.
    """
    real_open = builtins.open
    real_mkdir = pathlib.Path.mkdir
    real_makedirs = os.makedirs
    real_copytree = shutil.copytree
    real_copy = shutil.copyfile
    real_rmtree = shutil.rmtree
    real_replace = pathlib.Path.replace

    def guarded_open(file, mode="r", *a, **k):
        if _WRITE_MODES & set(str(mode)) and _is_share(file):
            _refuse(file, "open for writing")
        return real_open(file, mode, *a, **k)

    def guarded_mkdir(self, *a, **k):
        if _is_share(self):
            _refuse(self, "mkdir on")
        return real_mkdir(self, *a, **k)

    def guarded_makedirs(name, *a, **k):
        if _is_share(name):
            _refuse(name, "makedirs on")
        return real_makedirs(name, *a, **k)

    def guarded_copytree(src, dst, *a, **k):
        if _is_share(dst):
            _refuse(dst, "copytree into")
        return real_copytree(src, dst, *a, **k)

    def guarded_copyfile(src, dst, *a, **k):
        if _is_share(dst):
            _refuse(dst, "copy into")
        return real_copy(src, dst, *a, **k)

    def guarded_rmtree(path, *a, **k):
        if _is_share(path):
            _refuse(path, "rmtree on")
        return real_rmtree(path, *a, **k)

    def guarded_replace(self, target):
        if _is_share(self) or _is_share(target):
            _refuse(target, "rename onto")
        return real_replace(self, target)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pathlib.Path, "mkdir", guarded_mkdir)
    monkeypatch.setattr(os, "makedirs", guarded_makedirs)
    monkeypatch.setattr(shutil, "copytree", guarded_copytree)
    monkeypatch.setattr(shutil, "copyfile", guarded_copyfile)
    monkeypatch.setattr(shutil, "rmtree", guarded_rmtree)
    monkeypatch.setattr(pathlib.Path, "replace", guarded_replace)
    yield

@pytest.fixture(autouse=True)
def no_real_cache_writes(monkeypatch, tmp_path_factory):
    """Point every cache read and write at a per-test directory.

    REDIRECTED RATHER THAN REFUSED, unlike the share guard above. Caching is meant to be
    transparent: a test exercising a bootstrap should not have to know one exists, so failing it
    for touching the cache would punish the wrong thing. Redirecting also gives each test a COLD
    cache, which is what makes a cold-vs-warm round-trip assertion mean anything.

    `CACHE_DIR` is a module global read at call time by `session_cache`, `grant_figures` and
    `joint_locanmf` alike, so patching the one attribute covers all three. Tests that set it
    themselves still win -- they run after this.
    """
    from wfield_local import session_cache
    monkeypatch.setattr(session_cache, "CACHE_DIR", tmp_path_factory.mktemp("session_cache"))
    yield
