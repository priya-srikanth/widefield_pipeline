"""Local widefield processing helpers for the DAQ recorder repository."""

# THE REBUILD-ROOT FENCE (coordinator 11-03): while WIDEFIELD_REBUILD_ROOT is set, refuse writes outside the rebuild root
# and reads of default derived products, in every process that imports the package (wfield_local/rebuild_root.py).
import os as _os

if _os.environ.get("WIDEFIELD_REBUILD_ROOT", "").strip():
    from wfield_local import rebuild_root as _rebuild_root

    _rebuild_root.install()

# PROTECTED ROOTS (CLAUDE.md rule 11, Priya 11-24): production / patch / manuscript folders and the frozen stores are
# read-only during a run; every Python-level write under them is refused unless the run declared it
# (wfield_local/protected_roots.py, configs/protected_roots.yaml). Always on; WFIELD_PROTECTED_GUARD=warn only prints.
from wfield_local import protected_roots as _protected_roots  # noqa: E402

_protected_roots.install()
