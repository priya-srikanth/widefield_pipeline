"""PROTECTED ROOTS (CLAUDE.md rule 11, Priya 2026-11-24): production, manuscript-lineage, patch, manuscript and
frozen-store folders are READ-ONLY during a run.

On 11-24 four overwrites in one day had the same shape: a write landed in a protected folder as a SIDE EFFECT (a
production default when an override was missing; a reader that froze decoders on a store miss; a recomputed cache
saved through a READ resolver into a patch folder). Each was a write the run never meant to make, so the guard is a
LOCATION rule, like `writeguard`'s rule-1 check, driven by one registry (`configs/protected_roots.yaml`).

A write to a path under a protected root is allowed only when:
  * the root is a frozen store and the run allows freezing (``WFIELD_ALLOW_FREEZE=1`` or ``--allow-freeze`` on the
    command line); nothing else opens a frozen store;
  * otherwise the path lies inside the run's declared NEW output root (``WFIELD_RUN_OUT``, ``os.pathsep``-separated;
    the active ``WIDEFIELD_REBUILD_ROOT`` counts as one), or
  * ``WFIELD_ALLOW_PRODUCTION_WRITE=<reason>`` is set: the write goes ahead and the reason is printed (and appended to
    ``WFIELD_PRODUCTION_WRITE_LOG`` when that names a file), once per protected root per process.

A ``WFIELD_RUN_OUT`` entry that IS a protected root (or contains one) grants nothing for that root: the declared
folder must be new.

Enforcement points:
  * `writeguard.assert_writable` (every existing call site) and `writeguard.assert_not_protected`;
  * `figure_layout.save_png_svg` (every page / deck figure);
  * a process-wide audit hook (`install`, added by ``wfield_local/__init__.py`` at import), which sees every
    Python-level open-for-write, mkdir, remove, rename, replace, rmtree, copy and utime, whatever module makes it
    (``DataFrame.to_csv``, ``np.save``, ``pickle.dump`` on an open file, ``joblib.dump``, ``Path.write_text`` ...).
    Not seen: writes made in C code that bypasses Python's ``open`` (h5py, cv2, pyarrow's own filesystem) and writes
    by other programs (rsync, robocopy, shell redirects).

``WFIELD_PROTECTED_GUARD=warn`` prints refusals instead of raising (a transition aid; the default is to refuse).
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys
import threading
from pathlib import Path

__all__ = ["ProtectedWriteError", "ALLOW_FREEZE_ENV", "RUN_OUT_ENV", "ALLOW_PRODUCTION_ENV", "allow_freeze",
           "match", "refusal", "check", "require_out", "out_from_env", "install", "registry", "roots_on_this_machine"]

RUN_OUT_ENV = "WFIELD_RUN_OUT"
ALLOW_PRODUCTION_ENV = "WFIELD_ALLOW_PRODUCTION_WRITE"
ALLOW_FREEZE_ENV = "WFIELD_ALLOW_FREEZE"
MODE_ENV = "WFIELD_PROTECTED_GUARD"
LOG_ENV = "WFIELD_PRODUCTION_WRITE_LOG"
REGISTRY_ENV = "WFIELD_PROTECTED_ROOTS_FILE"          # tests point this at a fixture registry
_TRUE = ("1", "true", "yes", "on")


class ProtectedWriteError(PermissionError):
    """A write under a protected root (CLAUDE.md rule 11). The message names the path, the root, its kind and the
    way to declare a run output folder."""


# ----------------------------------------------------------------------------------------------- registry
_REG: list | None = None
_REG_KEY = None


def _registry_file() -> Path:
    return Path(os.environ.get(REGISTRY_ENV) or Path(__file__).resolve().parents[1] / "configs" / "protected_roots.yaml")


def registry() -> list[dict]:
    """The registry entries: dicts with ``where`` ("share" | "local"), ``segs`` (lowercased pattern segments) or
    ``path`` (normalized local path), ``kind``, ``hint`` and the original ``pattern``. Dynamic entries: the local
    frozen store (`frozen_models.local_dir`) and a ``WIDEFIELD_FROZEN_MODEL_DIR`` store."""
    global _REG, _REG_KEY
    key = (str(_registry_file()), os.environ.get("WIDEFIELD_FROZEN_MODEL_DIR", ""))
    if _REG is not None and _REG_KEY == key:
        return _REG
    import yaml
    with open(_registry_file(), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out = []
    for e in raw.get("share") or []:
        pat = str(e["path"]).replace("\\", "/").strip("/").lower()
        out.append({"where": "share", "segs": pat.split("/"), "kind": e.get("kind", "production"),
                    "hint": e.get("hint", ""), "pattern": e["path"]})
    local = list(raw.get("local") or [])
    env_store = os.environ.get("WIDEFIELD_FROZEN_MODEL_DIR", "").strip()
    if env_store:
        local.append({"path": env_store, "kind": "frozen_store"})
    for e in local:
        p = str(e["path"])
        if "{frozen_local}" in p:
            try:
                from wfield_local import frozen_models
                p = p.replace("{frozen_local}", str(frozen_models.local_dir()))
            except Exception:                                          # noqa: BLE001 - no machine profile
                continue
        k = _key(p)
        ent = {"where": k[0], "kind": e.get("kind", "production"), "hint": e.get("hint", ""), "pattern": p}
        if k[0] == "share":
            ent["segs"] = k[1].split("/")
        else:
            ent["path"] = k[1]
        out.append(ent)
    _REG, _REG_KEY = out, key
    return out


def _norm(path) -> str:
    s = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
    s = os.path.normpath(os.path.abspath(s)) if s else ""
    return s.replace("\\", "/").lower().rstrip("/")


#: a MICROSCOPE mount: "<drive>:/MICROSCOPE/Priya/...", a UNC path "//<host>/.../MICROSCOPE/Priya/..." or the mac mount
#: "/Volumes/Priya/...". Anchored at the mount, so a test's fake tree under a temp folder (".../tmp/MICROSCOPE/Priya")
#: is a LOCAL path, never a protected share root.
_SHARE_RE = re.compile(r"^(?:[a-z]:/microscope/priya|//[^/]+/(?:[^/]+/)*?microscope/priya|(?:[a-z]:)?/volumes/priya)(?:/|$)")


def _key(path) -> tuple[str, str]:
    """("share", <path relative to the Priya root>) on any MICROSCOPE mount, else ("local", <normalized path>)."""
    c = _norm(path)
    if c.startswith("//?/unc/"):                           # Windows extended-length prefixes
        c = "//" + c[8:]
    elif c.startswith("//?/"):
        c = c[4:]
    m = _SHARE_RE.match(c)
    if m:
        return "share", c[m.end():].strip("/")
    return "local", c


def _within(k: tuple[str, str], root: tuple[str, str]) -> bool:
    return k[0] == root[0] and (k[1] == root[1] or k[1].startswith(root[1] + "/") or root[1] == "")


def match(path) -> list[tuple[dict, tuple[str, str]]]:
    """Every registry entry ``path`` falls under, with the concrete root it matched (as a `_key`)."""
    return _match_key(_key(path))


def _match_key(k: tuple[str, str]) -> list[tuple[dict, tuple[str, str]]]:
    hits = []
    for e in registry():
        if e["where"] != k[0]:
            continue
        if k[0] == "share":
            segs = k[1].split("/") if k[1] else []
            n = len(e["segs"])
            if len(segs) >= n and all(fnmatch.fnmatchcase(s, p) for s, p in zip(segs, e["segs"])):
                hits.append((e, ("share", "/".join(segs[:n]))))
        elif _within(k, ("local", e["path"])):
            hits.append((e, ("local", e["path"])))
    return hits


# ----------------------------------------------------------------------------------------------- allowances
def allow_freeze() -> bool:
    """True when this run may FIT and STORE new frozen references: ``WFIELD_ALLOW_FREEZE=1`` or ``--allow-freeze`` on
    the command line (the flag is copied into the env so spawned workers inherit it)."""
    if "--allow-freeze" in sys.argv:
        os.environ[ALLOW_FREEZE_ENV] = "1"
    return os.environ.get(ALLOW_FREEZE_ENV, "").strip().lower() in _TRUE


def _run_out_roots() -> list[tuple[str, str]]:
    out = [_key(x) for x in os.environ.get(RUN_OUT_ENV, "").split(os.pathsep) if x.strip()]
    rb = os.environ.get("WIDEFIELD_REBUILD_ROOT", "").strip()
    if rb:
        out.append(_key(rb))
    return out


def _is_protected_root_or_parent(r: tuple[str, str]) -> bool:
    """True when a declared run root IS a protected root, lies INSIDE one or contains one (it is then not a NEW
    folder of its own: a new subfolder of production is still production)."""
    if _match_key(r):
        return True
    for e in registry():
        if e["where"] != r[0]:
            continue
        if r[0] == "local":
            if _within(("local", e["path"]), r):
                return True
            continue
        segs = r[1].split("/") if r[1] else []
        if len(segs) <= len(e["segs"]) and all(fnmatch.fnmatchcase(s, p) for s, p in zip(segs, e["segs"])):
            return True
    return False


_LOGGED: set = set()
_LOCK = threading.Lock()


def _log_production_write(root: tuple[str, str], path, reason: str) -> None:
    with _LOCK:
        if root in _LOGGED:
            return
        _LOGGED.add(root)
    msg = (f"[protected] PRODUCTION WRITE under {root[1]} (first file {path}); {ALLOW_PRODUCTION_ENV}={reason!r}; "
           f"pid {os.getpid()}; argv {' '.join(sys.argv)[:300]}")
    print(msg, file=sys.stderr, flush=True)
    lf = os.environ.get(LOG_ENV, "").strip()
    if lf:
        try:
            import datetime as dt
            with open(lf, "a", encoding="utf-8") as fh:
                fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")
        except OSError:
            pass


def refusal(path) -> str | None:
    """Why a write to ``path`` is refused (None = allowed). Pure apart from the production-write log line."""
    hits = match(path)
    if not hits:
        return None
    frozen = [h for h in hits if h[0]["kind"] == "frozen_store"]
    if frozen:
        if allow_freeze():
            return None
        e, root = frozen[0]
        return (f"{root[1]} is a FROZEN STORE ({e['pattern']}); stores are load-only by default (CLAUDE.md rules 10-11). "
                f"Fitting / freezing needs Priya's approval and {ALLOW_FREEZE_ENV}=1 or --allow-freeze.")
    k = _key(path)
    for r in _run_out_roots():
        rb = os.environ.get("WIDEFIELD_REBUILD_ROOT", "").strip()
        is_rebuild = bool(rb) and r == _key(rb)
        if _within(k, r) and (is_rebuild or not _is_protected_root_or_parent(r)):
            return None
    reason = os.environ.get(ALLOW_PRODUCTION_ENV, "").strip()
    e, root = hits[0]
    if reason:
        _log_production_write(root, path, reason)
        return None
    bad_decl = [r for r in _run_out_roots() if _is_protected_root_or_parent(r)
                and not (os.environ.get("WIDEFIELD_REBUILD_ROOT") and r == _key(os.environ["WIDEFIELD_REBUILD_ROOT"]))]
    return (f"{root[1]} is a protected {e['kind'].upper()} folder ({e['pattern']}). A run writes only to its own NEW "
            f"output folder (CLAUDE.md rule 11): give the script an output folder"
            + (f" ({e['hint']})" if e.get("hint") else "")
            + f", or declare one with {RUN_OUT_ENV}=<new folder>. A production write needs Priya's logged decision: "
              f"{ALLOW_PRODUCTION_ENV}=<reason>."
            + (f" NOTE: {RUN_OUT_ENV} names a protected root itself ({[x[1] for x in bad_decl]}), which grants nothing."
               if bad_decl else ""))


def _mode() -> str:
    return os.environ.get(MODE_ENV, "enforce").strip().lower() or "enforce"


def check(path, what: str = "write") -> None:
    """Raise `ProtectedWriteError` when a write to ``path`` is refused (``WFIELD_PROTECTED_GUARD=warn``: print only).
    The refusal is printed to stderr too, so a caller that swallows exceptions still leaves a trace."""
    why = refusal(path)
    if not why:
        return
    msg = f"[protected] refusing to {what} {path}: {why}"
    print(msg, file=sys.stderr, flush=True)
    if _mode() == "warn":
        return
    raise ProtectedWriteError(msg)


def require_out(path, what: str = "output folder", hint: str = "") -> Path:
    """Call at a script's start with its resolved OUTPUT folder: SystemExit (a clear message, before any compute) when
    that folder is protected and not allowed (CLAUDE.md rule 11: "A script with no output folder given REFUSES; it
    never falls back to a production default"). Returns the path."""
    why = refusal(path)
    if why and _mode() != "warn":
        raise SystemExit(f"[protected] {what} {path} is not writable for this run: {why}"
                         + (f"\n  {hint}" if hint else ""))
    if why:
        print(f"[protected] WARN {what} {path}: {why}", file=sys.stderr, flush=True)
    return Path(path)


def out_from_env(env: str, default, what: str = "") -> Path:
    """THE OUTPUT FOLDER OF A SCRIPT WHOSE OVERRIDE IS AN ENV VAR (rule 11: no production default). ``env`` set -> that
    folder (checked with `require_out`); unset -> SystemExit naming ``env``, unless
    WFIELD_ALLOW_PRODUCTION_WRITE=<reason> is set, in which case ``default`` is used (and logged on its first write).
    Call it at the start of main(), not at import: importing a module for its helpers must stay free."""
    v = os.environ.get(env, "").strip()
    name = what or Path(sys.argv[0]).name or "this script"
    if v:
        return require_out(Path(v), f"{name} output folder ({env})")
    if os.environ.get(ALLOW_PRODUCTION_ENV, "").strip():
        print(f"[protected] {name}: {env} unset; writing the DEFAULT folder {default} under "
              f"{ALLOW_PRODUCTION_ENV}={os.environ[ALLOW_PRODUCTION_ENV]!r}", file=sys.stderr, flush=True)
        return Path(default)
    raise SystemExit(f"[protected] {name}: no output folder given. Set {env}=<new folder> (CLAUDE.md rule 11: a run "
                     f"never falls back to its default {default}); a production write needs "
                     f"{ALLOW_PRODUCTION_ENV}=<reason>.")


def roots_on_this_machine(share_root=None) -> list[tuple[str, Path]]:
    """Concrete (kind, folder) pairs of the registry on this machine, for the mtime audit: share globs expanded under
    the MICROSCOPE Priya root (``share_root``; default the resolver's ``microscope`` root)."""
    import glob
    if share_root is None:
        from wfield_local import config
        share_root = config.resolver().root("microscope")
    out, seen = [], set()
    for e in registry():
        if e["where"] == "share":
            pat = str(Path(share_root) / Path(*e["pattern"].replace("\\", "/").strip("/").split("/")))
            hits = sorted(glob.glob(pat))
        else:
            hits = [e["pattern"]] if os.path.exists(e["pattern"]) else []
        for h in hits:
            n = _norm(h)
            if n not in seen:
                seen.add(n)
                out.append((e["kind"], Path(h)))
    return out


# ----------------------------------------------------------------------------------------------- audit hook
_WRITE_EVENTS = {"os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.truncate": (0,), "shutil.rmtree": (0,),
                 "os.rename": (0, 1), "os.replace": (0, 1), "shutil.copyfile": (1,), "shutil.copytree": (1,),
                 "shutil.move": (0, 1), "os.symlink": (1,), "os.link": (1,), "os.utime": (0,)}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_BUSY = threading.local()
_installed = False


def _hook_check(path, event: str) -> None:
    if not isinstance(path, (str, bytes, os.PathLike)) or isinstance(path, bool) or getattr(_BUSY, "on", False):
        return
    _BUSY.on = True
    try:
        if not (os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)):
            return
        check(path, f"write ({event})")
    finally:
        _BUSY.on = False


def _hook(event: str, args) -> None:
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(mode, str):
            if not any(ch in mode for ch in "wax+"):
                return
        elif not (isinstance(flags, int) and flags & _WRITE_FLAGS):
            return
        _hook_check(path, "open")
        return
    idx = _WRITE_EVENTS.get(event)
    if idx is None:
        return
    for i in idx:
        if i < len(args):
            _hook_check(args[i], event)


def install() -> None:
    """Add the protected-roots audit hook to this process (idempotent). Audit hooks cannot be removed; the hook is
    cheap (a dict lookup for every non-write event)."""
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True
