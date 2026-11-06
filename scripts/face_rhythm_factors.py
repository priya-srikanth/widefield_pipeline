"""FaceRhythm frozen pre-stroke factor set: the command line. The code lives in `wfield_local/face_rhythm/factors/`
(split from this script 10-25, every definition moved verbatim); stages, inputs and outputs: that package's docstring
(``--help``) and docs/FACE_RHYTHM_PIPELINE.md.

    desktop   python scripts/face_rhythm_factors.py plan|select|report ...
    O2        python <bundle>/factors/<set>/fr_factors.py build|fit|loso|splithalf|project|fitsession|fitepoch ...

On O2 there is no repo: `plan` (or `deploy_code`) copies this file to ``<set>/fr_factors.py`` and the package to
``<set>/fr_factors_lib/``. A copy next to this file is always preferred over the repo, so a set runs exactly the code
deployed into it. Old imports (``from scripts import face_rhythm_factors as f; f.cmd_project``, ``from
face_rhythm_factors import npz_memmap``) still resolve: attributes are forwarded to the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if (_here / "fr_factors_lib" / "__init__.py").exists():          # a deployed set (O2, or its copy on the share)
    sys.path.insert(0, str(_here))
    import fr_factors_lib as _pkg
else:                                                            # the repo
    try:
        from wfield_local.face_rhythm import factors as _pkg
    except ImportError:
        sys.path.insert(0, str(_here.parent))
        from wfield_local.face_rhythm import factors as _pkg


def __getattr__(name: str):
    try:
        return getattr(_pkg, name)
    except AttributeError:
        for m in ("config", "files", "cp", "sample", "fit", "project", "validate", "o2", "report", "cli"):
            if hasattr(getattr(_pkg, m), name):
                return getattr(getattr(_pkg, m), name)
        raise


if __name__ == "__main__":
    raise SystemExit(_pkg.cli.main())
