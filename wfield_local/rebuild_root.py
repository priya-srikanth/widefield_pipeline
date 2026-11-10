"""THE REBUILD ROOT (coordinator 11-03, fourth review): one fresh folder that holds EVERY derived output of a render.

Routing writers one at a time did not converge (about 120 unrouted writers remained after three rounds), so a render
profile with ``rebuild_root`` makes it structural:

* RESOLVE. With ``WIDEFIELD_REBUILD_ROOT`` set, `paths.PathResolver` maps every derived logical root into it
  (``configs/paths.yaml`` ``rebuild.derived_roots``: figures_working, cue_analysis_out, cohort_out = the labcams-level
  outputs: grant_figures, frozen_models, ...; decks_out = the .pptx decks, since 2026-11-06) and the derived trees
  under an input root (``rebuild.derived_subtrees``: ``<LP_analysis>/<run>/imaging_encoding``, through `PathResolver.derived` /
  `orofacial_outputs.encoding_root`). The profile puts the frozen store and the session cache (and with it the RESTnl
  caches beside it) under the root too. Readers and writers use the SAME resolution, so a reader sees the rebuild's own
  output or nothing.
* FENCE. `install()` adds a process-wide audit hook (`sys.addaudithook`, so EVERY Python-level ``open`` / ``os.*`` /
  ``shutil`` call is seen, whatever module makes it) that, while the env var is set:
    - REFUSES a WRITE (open for write, mkdir, remove, rename, rmtree, ...) on a data tree outside the rebuild root: any
      MICROSCOPE / standby share path, and this box's default derived roots (figures_working, the session caches, the
      frozen / joint-basis stores beside it). Local paths that are not data trees (temp / scratch, __pycache__, user
      config) are untouched. The ONE exception is the hemo-variant build step (``WIDEFIELD_REBUILD_VARIANT_BUILD=1``):
      it may write ``hemo_*`` folders inside the per-session labcams data.
    - REFUSES a READ of a DEFAULT DERIVED product: the original figures_working / cue_analysis_out trees, the original
      ``*/imaging_encoding`` trees, every top-level labcams entry that is not session data or a listed input
      (``rebuild.labcams_inputs``), the default session caches and frozen stores. So an unrouted reader fails loudly
      instead of reading the pre-rebuild copy.
  `wfield_local/__init__.py` installs it at import when the env var is set (every pipeline module imports the package,
  and subprocesses inherit the env). Not seen: writes made in C code that bypass Python's ``open`` (h5py, cv2); none
  of the render's writers use them for outputs.
* SEED. Inputs that live INSIDE a derived tree (epoch boundaries; the camera / behavior QC tables in
  ``imaging_encoding/qc``) are copied in once by ``python -m wfield_local.rebuild_root init`` with a sha256 manifest
  (``rebuild.seed_inputs``); inputs that sit beside derived products (``rebuild.inputs_in_place``: the joint bases) are
  read in place through `PathResolver.input_root`.
"""
from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ENV = "WIDEFIELD_REBUILD_ROOT"
VARIANT_BUILD_ENV = "WIDEFIELD_REBUILD_VARIANT_BUILD"
MANIFEST = "REBUILD_ROOT.json"


class RebuildFenceError(PermissionError):
    """A write outside the rebuild root, or a read of a default derived product, during a rebuild-root run."""


def root() -> Path | None:
    """The active rebuild root (``WIDEFIELD_REBUILD_ROOT``), or None."""
    v = os.environ.get(ENV, "").strip()
    return Path(v.replace("\\", "/")) if v else None


def active() -> bool:
    return root() is not None


@functools.lru_cache(maxsize=1)
def spec() -> dict:
    """``configs/paths.yaml`` ``rebuild`` block."""
    import yaml
    from wfield_local.paths import CONFIG_DIR
    with open(CONFIG_DIR / "paths.yaml", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("rebuild") or {}


def _norm(p) -> str:
    """Absolute, forward-slash, lowercase, trailing-slash form (for prefix tests)."""
    s = os.path.abspath(os.fspath(p)).replace("\\", "/").lower()
    return s.rstrip("/") + "/"


def derived_root(name: str) -> str | None:
    """The rebuild-root location of WHOLE derived logical root ``name`` (None if it is not one, or no rebuild)."""
    r = root()
    sub = (spec().get("derived_roots") or {}).get(name)
    return None if r is None or sub is None else f"{r.as_posix().rstrip('/')}/{sub}"


def derived_subtree(name: str, rel: str) -> str | None:
    """The rebuild-root location of ``<root name>/<rel>`` when ``rel`` falls in a derived subtree of input root
    ``name`` (``rebuild.derived_subtrees``), else None."""
    r = root()
    d = (spec().get("derived_subtrees") or {}).get(name)
    if r is None or not d:
        return None
    rel = str(rel).replace("\\", "/").strip("/")
    segs = rel.split("/")
    for pat in d.get("patterns") or ():
        k = len(pat.split("/"))
        if len(segs) >= k and fnmatch.fnmatchcase("/".join(segs[:k]), pat):
            return f"{r.as_posix().rstrip('/')}/{d['dest']}/{rel}"
    return None


# ------------------------------------------------------------------------------------------------------- the fence
def _input_roots() -> dict:
    """{logical root: ORIGINAL (unremapped) mount} for the roots the fence needs; unknown roots are skipped."""
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    out = {}
    for n in ("figures_working", "cue_analysis_out", "cohort_out", "labcams", "orofacial"):
        try:
            out[n] = rv.input_root(n)
        except Exception:                                    # noqa: BLE001 - a root this machine lacks
            pass
    return out


@functools.lru_cache(maxsize=8)
def _fence(rkey: str) -> dict:
    """Normalized prefixes for the fence of rebuild root ``rkey`` (cached per root)."""
    roots = _input_roots()
    fw = roots.get("figures_working")
    local = []                                               # this box's default derived stores (siblings of fw)
    if fw:
        par = _norm(Path(fw).parent)
        local = [par + ".widefield_session_cache", par + "frozen_models", par + "joint_bases"]
    deny_read = [_norm(roots[n]) for n in ("figures_working", "cue_analysis_out") if n in roots]
    deny_read += [x for x in local if not x.endswith("joint_bases")]
    return {"root": _norm(rkey), "fw": _norm(fw) if fw else None, "local": local, "deny_read": deny_read,
            "labcams": _norm(roots["labcams"]) if "labcams" in roots else None,
            "orofacial": _norm(roots["orofacial"]) if "orofacial" in roots else None,
            "temp": _norm(__import__("tempfile").gettempdir())}


_SHARE_MARKS = ("/microscope/", "standby.files.med.harvard.edu", "/neurobio/sabatini/", "/volumes/priya/")


def _labcams_input(c: str, lab: str) -> bool:
    """True if ``c`` (under the original labcams tree ``lab``) is session data or a listed labcams input."""
    rest = c[len(lab):].strip("/")
    if not rest:
        return True                                          # the labcams folder itself (a listing of dates)
    top = rest.split("/", 1)[0]
    for pat in spec().get("labcams_inputs") or ():
        if "/" in pat and not pat.startswith("re:"):     # a relative-path glob: exact files (the painted glue masks)
            if fnmatch.fnmatchcase(rest, pat.lower()):
                return True
            continue
        if pat.startswith("re:"):
            if re.match(pat[3:], top):
                return True
        elif top == pat.lower():
            return True
    return False


def write_refusal(path) -> str | None:
    """Why a write to ``path`` is refused in the active rebuild (None = allowed)."""
    r = root()
    if r is None:
        return None
    c = _norm(path)
    f = _fence(str(r))
    if c.startswith(f["root"]) or c.startswith(f["temp"]):
        return None
    if os.environ.get(VARIANT_BUILD_ENV, "") not in ("", "0") and f["labcams"] and c.startswith(f["labcams"]) \
            and any(seg.startswith("hemo_") for seg in c[len(f["labcams"]):].split("/")):
        return None                                          # the hemo-variant build step (per-session hemo_* dirs)
    if any(m in c for m in _SHARE_MARKS):
        return "a share path outside the rebuild root"
    if f["fw"] and c.startswith(f["fw"]):
        return "this box's default figures_working"
    if any(c.startswith(x) for x in f["local"]):
        return "a default local store (session cache / frozen models / joint bases)"
    return None


#: EXACT source paths `init` may read while it seeds (coordinator 11-04): filled from `seed_plan()` for the duration of
#: `init` only (`_seeding`), so the seed copies can be made under an applied profile without opening the fence to any
#: other default derived product
_SEED_ALLOW: set = set()


class _seeding:
    """Context: allow reads of exactly these seed SOURCE paths (normalized) while `init` copies them."""

    def __init__(self, paths):
        self.paths = {_norm(p) for p in paths}

    def __enter__(self):
        _SEED_ALLOW.update(self.paths)
        return self

    def __exit__(self, *exc):
        _SEED_ALLOW.difference_update(self.paths)
        return False


def read_refusal(path) -> str | None:
    """Why a read of ``path`` is refused in the active rebuild (None = allowed): a DEFAULT derived product."""
    r = root()
    if r is None:
        return None
    c = _norm(path)
    if c in _SEED_ALLOW:                      # a planned seed SOURCE, while `init` runs (and only then)
        return None
    f = _fence(str(r))
    if c.startswith(f["root"]):
        return None
    if any(c.startswith(x) for x in f["deny_read"]):
        return "a default derived tree (figures_working / analysis_figures / session cache / frozen store)"
    if f["labcams"] and c.startswith(f["labcams"]) and not _labcams_input(c, f["labcams"]):
        return "a default labcams-level derived product (only session data and rebuild.labcams_inputs are inputs)"
    if f["orofacial"] and c.startswith(f["orofacial"]):
        rel = c[len(f["orofacial"]):].strip("/")
        segs = rel.split("/")
        if len(segs) >= 2 and segs[1] == "imaging_encoding":
            return "a default imaging_encoding tree"
    return None


_WRITE_EVENTS = {"os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.truncate": (0,), "shutil.rmtree": (0,),
                 "os.rename": (0, 1), "os.replace": (0, 1), "shutil.copyfile": (1,), "shutil.copytree": (1,),
                 "shutil.move": (0, 1), "os.symlink": (1,), "os.link": (1,), "os.chmod": (0,), "os.utime": (0,)}
_READ_EVENTS = {"os.listdir": (0,), "os.scandir": (0,), "shutil.copyfile": (0,), "shutil.copytree": (0,),
                "glob.glob": (0,)}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_installed = False


def _pathish(x) -> bool:
    return isinstance(x, (str, bytes, os.PathLike)) and not isinstance(x, bool)


_BUSY = __import__("threading").local()


def _check(path, write: bool, event: str) -> None:
    if not _pathish(path) or getattr(_BUSY, "on", False):
        return                                   # re-entrant call (the fence itself reading paths.yaml / tempdir)
    p = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
    if not p:
        return
    _BUSY.on = True
    try:
        why = write_refusal(p) if write else read_refusal(p)
    finally:
        _BUSY.on = False
    if why:
        raise RebuildFenceError(
            f"rebuild root {root()}: refusing to {'WRITE' if write else 'READ'} {p} ({event}): {why}. Route the "
            "path through config.resolver() / PathResolver.derived (orofacial_outputs.encoding_root), or seed the "
            "input with `python -m wfield_local.rebuild_root init`.")


def _hook(event: str, args) -> None:
    if event == "open":
        if os.environ.get(ENV) is None:
            return
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(mode, str):
            write = any(ch in mode for ch in "wax+")
        else:
            write = bool(isinstance(flags, int) and flags & _WRITE_FLAGS)
        _check(path, write, "open")
        return
    w = _WRITE_EVENTS.get(event)
    rd = _READ_EVENTS.get(event)
    if (w is None and rd is None) or os.environ.get(ENV) is None:
        return
    for i in (w or ()):
        if i < len(args):
            _check(args[i], True, event)
    for i in (rd or ()):
        if i < len(args):
            _check(args[i], False, event)


def install() -> None:
    """Add the fence audit hook to this process (idempotent; inert while WIDEFIELD_REBUILD_ROOT is unset)."""
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


def install_if_active() -> None:
    if os.environ.get(ENV, "").strip():
        install()


def require_seed(path) -> None:
    """Raise when ``path`` (an input a reader would otherwise silently skip) is missing INSIDE the active rebuild root:
    the root was not seeded (`init`). No-op outside a rebuild root or for paths outside it."""
    r = root()
    if r is None or not _norm(path).startswith(_norm(r)) or Path(path).exists():
        return
    raise FileNotFoundError(f"{path} is missing from rebuild root {r}: seed it first "
                            f"(python -m wfield_local.rebuild_root init --profile <profile>)")


# ---------------------------------------------------------------------------------------------------------- init
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_plan() -> list[tuple[str, Path, Path]]:
    """[(logical root/rel, source = the ORIGINAL input, destination under the rebuild root)] for `init`."""
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    out = []
    for name, rels in (spec().get("seed_inputs") or {}).items():
        for rel in rels:
            src = Path(rv.input_root(name)) / rel
            dst = Path(rv.derived(name, rel))
            out.append((f"{name}/{rel}", src, dst))
    return out


def init(dry_run: bool = False) -> dict:
    """Create the rebuild root and COPY the seed inputs into it (sha256-verified; an existing copy with the same hash
    is kept, a DIFFERENT one raises). Writes REBUILD_ROOT.json (sources, hashes, the derived-root map)."""
    r = root()
    if r is None:
        raise SystemExit(f"{ENV} is not set: apply the render profile first")
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    rec = {"rebuild_root": str(r), "derived_roots": {n: derived_root(n) for n in spec().get("derived_roots") or {}},
           "seeded": []}
    plan = seed_plan()
    # the fence (installed by render_profile.apply) refuses default derived products; the planned seed SOURCES -- and
    # nothing else -- are readable for the duration of the copy (coordinator 11-04)
    with _seeding([src for _k, src, _d in plan]):
        _seed_copy(plan, rec, dry_run)
    rec["cohort_out"] = rv.root("cohort_out")
    if not dry_run:
        r.mkdir(parents=True, exist_ok=True)
        (r / MANIFEST).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def _seed_copy(plan, rec, dry_run):
    for key, src, dst in plan:
        if not src.exists():
            rec["seeded"].append({"input": key, "source": str(src), "missing": True})
            continue
        hs = _sha256(src)
        if dst.exists():
            if _sha256(dst) != hs:
                raise SystemExit(f"seed {dst} exists and differs from its source {src}: refusing to overwrite")
        elif not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if _sha256(dst) != hs:
                raise SystemExit(f"seed copy {dst} does not match {src}")
        rec["seeded"].append({"input": key, "source": str(src), "dest": str(dst), "sha256": hs})


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("init", "show"))
    ap.add_argument("--profile", default=None, help="apply this render profile first")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.profile:
        from wfield_local import render_profile
        render_profile.apply(a.profile)
    if a.cmd == "show":
        print(json.dumps({"rebuild_root": str(root()), "derived_roots": {n: derived_root(n) for n in
                                                                          spec().get("derived_roots") or {}},
                          "seed": [(k, str(s), str(d)) for k, s, d in seed_plan()] if active() else []}, indent=1))
        return 0
    print(json.dumps(init(dry_run=a.dry_run), indent=1))
    return 0


if __name__ == "__main__":
    # `python -m wfield_local.rebuild_root` runs THIS file as __main__, a second copy of the module: the audit hook that
    # render_profile.apply installs is the IMPORTED module's and reads ITS state (_SEED_ALLOW, _BUSY). Run the CLI in
    # the imported module, so init's seed allow-list is the one the hook checks (coordinator 11-04).
    from wfield_local import rebuild_root as _rr
    raise SystemExit(_rr.main())
