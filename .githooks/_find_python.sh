#!/usr/bin/env bash
# Locate a python that actually has the dev tools. Sourced by pre-commit and pre-push.
#
# WHY THIS IS SHARED AND WHY IT TRIES SO HARD. A bare `python` in Git Bash on this machine
# resolves to the Windows App Store SHIM, which has neither ruff nor pytest. Both hooks used to
# guess `$CONDA_PREFIX/python.exe` and fall back to `python`, and in a shell where CONDA_PREFIX
# is unset -- which is every shell an editor or a GUI git client spawns -- the fallback is the
# shim. On 2026-09-22 that made the pre-commit hook print "ruff not installed — skipping lint"
# and pass, on the very first commit after the hooks were enabled. A gate that cannot find its
# tool must not be a gate that reports success.
#
# Sets PY. Returns 1 if nothing usable was found; the caller decides whether that is fatal.
find_python() {                       # find_python <import-name-to-require>
  local need="$1" cand
  for cand in \
      "${CONDA_PREFIX:-}/python.exe" \
      "${CONDA_PREFIX:-}/bin/python" \
      "$(git rev-parse --show-toplevel)/.venv/Scripts/python.exe" \
      "$(git rev-parse --show-toplevel)/.venv/bin/python" \
      "$HOME/miniconda3/envs/locanmf/python.exe" \
      "$HOME/anaconda3/envs/locanmf/python.exe" \
      "$HOME/.conda/envs/locanmf/python.exe" \
      "$(command -v python3 2>/dev/null)" \
      "$(command -v python 2>/dev/null)"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    if "$cand" -c "import $need" >/dev/null 2>&1; then PY="$cand"; return 0; fi
  done
  return 1
}
