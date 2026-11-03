"""Render profiles: ONE file per render configuration (configs/render_profiles/<name>.yaml; coordinator 11-03).

A profile holds the env switches of a render (``env``), the per-block arguments that must NOT be global (``block_args``,
e.g. the decoders' ``--pre-from 0801``) and notes. Apply it BEFORE importing analysis modules: `hemo_variants.ADOPTED`
is resolved at import.

    python -m wfield_local.render_profile cohort1_rebuild_20261103 --shell powershell   # prints $env:... lines
    python -m wfield_local.render_profile cohort1_rebuild_20261103 --shell bash         # prints export lines
    python -m wfield_local.render_profile cohort1_rebuild_20261103 --check              # verifies the tags (exit 1)

Applying a profile CLEARS every other WIDEFIELD_* / WFIELD_* variable first (R5, fourth review), and a profile with
``rebuild_root`` sends every derived output into that one folder (wfield_local/rebuild_root.py; run
``python -m wfield_local.rebuild_root init --profile <name>`` once to seed it).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parents[1] / "configs" / "render_profiles"


#: env prefixes a profile OWNS: every variable with one of them that the profile does not set is CLEARED by `apply` (R5,
#: fourth review: a leftover GATE_SOURCE / ENL_WINDOW_S / FROZEN_OUT_ROOT in the shell silently re-routed outputs)
OWNED_PREFIXES = ("WIDEFIELD_", "WFIELD_")


def load(name: str) -> dict:
    import yaml
    p = PROFILE_DIR / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"render profile {name!r} not found in {PROFILE_DIR}")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    env = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
    rr = d.get("rebuild_root")
    if rr:
        # THE REBUILD ROOT (coordinator 11-03): every derived root resolves inside it (wfield_local/rebuild_root.py);
        # the content-keyed stores go there too, so a profile can never share a store or cache with another run
        from wfield_local import rebuild_root
        r = str(rr).replace("\\", "/").rstrip("/")
        stores = rebuild_root.spec().get("stores") or {}
        env[rebuild_root.ENV] = r
        env.setdefault("WIDEFIELD_FROZEN_MODEL_DIR", f"{r}/{stores.get('frozen_models', 'frozen_models')}")
        env.setdefault("WIDEFIELD_SESSION_CACHE", f"{r}/{stores.get('session_cache', 'session_cache')}")
    d["env"] = env
    return d


def leftovers(d: dict, environ=None) -> list[str]:
    """Owned-prefix variables set in ``environ`` that profile ``d`` does not set (what `apply` clears)."""
    env = os.environ if environ is None else environ
    return sorted(k for k in env if k.startswith(OWNED_PREFIXES) and k not in d["env"])


def apply(name: str, environ=None) -> dict:
    """CLEAR every leftover WIDEFIELD_* / WFIELD_* variable, then set the profile's env in ``environ`` (default
    os.environ); returns the profile with ``cleared`` (the names removed)."""
    d = load(name)
    env = os.environ if environ is None else environ
    d["cleared"] = leftovers(d, env)
    for k in d["cleared"]:
        env.pop(k, None)
    env.update(d["env"])
    if env is os.environ and d["env"].get("WIDEFIELD_REBUILD_ROOT"):
        from wfield_local import rebuild_root
        rebuild_root.install()
    return d


def shell_lines(name: str, shell: str = "powershell") -> list[str]:
    """Shell lines that first REMOVE every WIDEFIELD_* / WFIELD_* variable of that shell, then set the profile's."""
    d = load(name)
    if shell == "bash":
        clear = ["for v in $(compgen -e | grep -E '^(WIDEFIELD|WFIELD)_'); do unset \"$v\"; done"]
        return clear + [f'export {k}="{v}"' for k, v in d["env"].items()]
    clear = ["Get-ChildItem Env: | Where-Object { $_.Name -like 'WIDEFIELD_*' -or $_.Name -like 'WFIELD_*' } | "
             "ForEach-Object { Remove-Item \"Env:$($_.Name)\" }"]
    return clear + [f'$env:{k} = "{v}"' for k, v in d["env"].items()]


def check(name: str) -> tuple[bool, list[str]]:
    """Apply in-process and compare the FULL output tag (`lp_licks.output_tag`, LP / gate-LP / ENL included) and the
    run tag with the profile's ``notes.expected_run_tag``; report the rebuild root and the derived roots."""
    d = apply(name)
    from wfield_local import config, lp_licks, run_mode
    want = (d.get("notes") or {}).get("expected_run_tag")
    full, rt = lp_licks.output_tag(), run_mode.tag()
    lines = [f"cleared: {d['cleared'] or 'none'}", f"run tag: {rt}", f"output tag: {full}", f"expected: {want}"]
    rr = d["env"].get("WIDEFIELD_REBUILD_ROOT")
    lines.append(f"rebuild root: {rr}")
    if rr:
        rv = config.resolver()
        for n in ("figures_working", "cue_analysis_out", "cohort_out"):
            lines.append(f"  {n}: {rv.root(n)}")
        from wfield_local import orofacial_outputs as oo
        lines.append(f"  imaging_encoding: {oo.encoding_root()}")
    lines.append(f"block args: {d.get('block_args')}")
    ok = want is None or (full == want and rt == want)
    return ok, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name")
    ap.add_argument("--shell", choices=("powershell", "bash"), default=None)
    ap.add_argument("--check", action="store_true", help="apply in-process and print / verify the tags")
    a = ap.parse_args(argv)
    if a.shell:
        print("\n".join(shell_lines(a.name, a.shell)))
    if a.check or not a.shell:
        ok, lines = check(a.name)
        print("\n".join(lines))
        if not ok:
            print("MISMATCH: the output tag is not the profile's expected tag")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
