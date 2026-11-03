"""Render profiles: ONE file per render configuration (configs/render_profiles/<name>.yaml; coordinator 11-03).

A profile holds the env switches of a render (``env``), the per-block arguments that must NOT be global (``block_args``,
e.g. the decoders' ``--pre-from 0801``) and notes. Apply it BEFORE importing analysis modules: `hemo_variants.ADOPTED`
is resolved at import.

    python -m wfield_local.render_profile cohort1_rebuild_20261103 --shell powershell   # prints $env:... lines
    python -m wfield_local.render_profile cohort1_rebuild_20261103 --shell bash         # prints export lines
    python -m wfield_local.render_profile cohort1_rebuild_20261103 --check              # prints the run tag
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parents[1] / "configs" / "render_profiles"


def load(name: str) -> dict:
    import yaml
    p = PROFILE_DIR / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"render profile {name!r} not found in {PROFILE_DIR}")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    d["env"] = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
    return d


def apply(name: str, environ=None) -> dict:
    """Set the profile's env in ``environ`` (default os.environ); returns the profile."""
    d = load(name)
    env = os.environ if environ is None else environ
    env.update(d["env"])
    return d


def shell_lines(name: str, shell: str = "powershell") -> list[str]:
    d = load(name)
    if shell == "bash":
        return [f'export {k}="{v}"' for k, v in d["env"].items()]
    return [f'$env:{k} = "{v}"' for k, v in d["env"].items()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name")
    ap.add_argument("--shell", choices=("powershell", "bash"), default=None)
    ap.add_argument("--check", action="store_true", help="apply in-process and print the run tag")
    a = ap.parse_args(argv)
    if a.shell:
        print("\n".join(shell_lines(a.name, a.shell)))
    if a.check or not a.shell:
        d = apply(a.name)
        from wfield_local import run_mode
        print("run tag:", run_mode.tag())
        print("expected:", (d.get("notes") or {}).get("expected_run_tag"))
        print("block args:", d.get("block_args"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
