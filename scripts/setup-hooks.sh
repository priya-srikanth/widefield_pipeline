#!/usr/bin/env bash
# One-command install of the repo's git hooks (.githooks/). Run once per clone, on each machine.
#   bash scripts/setup-hooks.sh
# core.hooksPath is LOCAL git config (never committed), so each clone opts in independently.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true   # no-op on Windows filesystems
echo "hooks enabled: core.hooksPath = .githooks (pre-commit ruff, pre-push pytest)."
