"""Local widefield processing helpers for the DAQ recorder repository."""

# THE REBUILD-ROOT FENCE (coordinator 11-03): while WIDEFIELD_REBUILD_ROOT is set, refuse writes outside the rebuild root
# and reads of default derived products, in every process that imports the package (wfield_local/rebuild_root.py).
import os as _os

if _os.environ.get("WIDEFIELD_REBUILD_ROOT", "").strip():
    from wfield_local import rebuild_root as _rebuild_root

    _rebuild_root.install()
