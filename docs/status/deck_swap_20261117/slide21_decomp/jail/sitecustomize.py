"""Slide-21 decomposition WRITE JAIL (2026-11-17). Loaded by every Python process (incl. spawned pool workers) whose
PYTHONPATH holds this folder, but active only while SLIDE21_JAIL is set.

* Refuses every write (open for write, mkdir, rename/replace, remove, copy, ...) to a share path (M:/, N:/, any
  '/microscope/' path) unless it lies under SLIDE21_JAIL (the decomposition folder), and to the encoder input cache
  (WFIELD_INPUT_CACHE). Local temp / __pycache__ / matplotlib caches stay writable.
* Caps the default worker count: os.cpu_count() reports SLIDE21_CPUS (default 5) so parallel.default_jobs() = 3.
Every refusal is appended to <SLIDE21_JAIL>/jail_refusals.log before it raises.
"""
import os
import sys

_JAIL = os.environ.get("SLIDE21_JAIL", "").strip()

if _JAIL:
    _cpus = int(os.environ.get("SLIDE21_CPUS", "5"))
    os.cpu_count = lambda: _cpus  # noqa: E731

    def _n(p):
        try:
            s = os.fsdecode(p) if isinstance(p, bytes) else os.fspath(p)
        except TypeError:
            return None
        if not s:
            return None
        return os.path.abspath(s).replace("\\", "/").lower().rstrip("/") + "/"

    _ALLOW = _n(_JAIL)
    _IC = _n(os.environ["WFIELD_INPUT_CACHE"]) if os.environ.get("WFIELD_INPUT_CACHE") else None
    _LOG = os.path.join(_JAIL, "jail_refusals.log")
    _WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    _W = {"os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.truncate": (0,), "shutil.rmtree": (0,),
          "os.rename": (0, 1), "os.replace": (0, 1), "shutil.copyfile": (1,), "shutil.copytree": (1,),
          "shutil.move": (0, 1), "os.symlink": (1,), "os.link": (1,), "os.chmod": (0,), "os.utime": (0,)}
    _busy = [False]

    def _bad(c):
        if c is None or c.startswith(_ALLOW) or c.startswith("//./") or c.startswith("//?/nul"):
            return False                     # the null device (os.devnull = "nul" -> \\.\nul) is not a share
        if _IC and c.startswith(_IC):
            return True
        return c.startswith("m:/") or c.startswith("n:/") or "/microscope/" in c or c.startswith("//")

    def _refuse(p, ev):
        if _busy[0]:
            return
        c = _n(p)
        if not _bad(c):
            return
        _busy[0] = True
        try:
            with open(_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"pid {os.getpid()} {ev} {p}\n")
        except Exception:  # noqa: BLE001
            pass
        finally:
            _busy[0] = False
        raise PermissionError(f"SLIDE21 JAIL: refusing write ({ev}) outside {_JAIL}: {p}")

    def _hook(ev, args):
        if ev == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            if isinstance(mode, str):
                w = any(ch in mode for ch in "wax+")
            else:
                w = bool(isinstance(flags, int) and flags & _WFLAGS)
            if w and isinstance(path, (str, bytes, os.PathLike)):
                _refuse(path, "open")
            return
        idx = _W.get(ev)
        if idx:
            for i in idx:
                if i < len(args) and isinstance(args[i], (str, bytes, os.PathLike)):
                    _refuse(args[i], ev)

    sys.addaudithook(_hook)
