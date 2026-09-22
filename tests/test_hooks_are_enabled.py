"""The quality gates must be switched ON in this clone, and this is the only thing that says so.

`core.hooksPath` is LOCAL git config. It cannot be committed, so enabling the hooks in one clone
does nothing for the other, and a clone with them off looks exactly like a clone with them on:
commits succeed, pushes succeed, nothing is printed. They were off on the imaging box from August
until 2026-09-22 and 21 commits went through ungated before anyone noticed.

WHY THIS IS A FAILURE AND NOT A WARNING. A warning is what the pre-commit hook itself used to
print when it could not find ruff, and it went unread on the very first commit after the hooks
were enabled. The repo's own rule is that a check which cannot fail is decoration. The fix is one
command and it is in the message.

WHY IT CANNOT GATE A PUSH, WHICH IS THE LIMITATION TO KNOW. `pre-push` runs the suite -- so when
the hooks are ON this test runs and passes, and when they are OFF `pre-push` never runs at all and
this test never gets the chance to block anything. It only fires for somebody running `pytest` by
hand. That is enough to surface the state, and it is not enough to enforce it; nothing committable
can enforce local config.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _git(*args) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):                      # noqa: BLE001
        pytest.skip("git is not runnable here")
    return r.stdout.decode("utf-8", "replace").strip()


def test_the_repo_ships_the_hooks_and_their_installer():
    """Checked separately from whether they are ENABLED: this part is committable, that part is not."""
    for rel in (".githooks/pre-commit", ".githooks/pre-push", ".githooks/_find_python.sh",
                "scripts/setup-hooks.sh"):
        assert (ROOT / rel).is_file(), f"{rel} is missing -- the gate cannot be enabled without it"


def test_core_hookspath_points_at_the_repo_hooks():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout (worktree or export)")
    got = _git("config", "core.hooksPath")
    assert got == ".githooks", (
        f"git hooks are NOT enabled in this clone (core.hooksPath = {got or '<unset>'!s}).\n"
        f"  Nothing you commit or push here is being linted or tested.\n"
        f"  Fix, once per clone:   bash scripts/setup-hooks.sh\n"
        f"  It sets core.hooksPath = .githooks, which is local config and cannot be committed --\n"
        f"  which is exactly why this test exists rather than a line in a README."
    )
