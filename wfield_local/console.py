"""Make this process's stdout/stderr able to carry the characters the figures already contain.

WHY THIS EXISTS (2026-08-27). Windows gives a Python process cp1252 stdout, on a console AND when
redirected to a file. Matplotlib renders a negative tick label with U+2212 MINUS SIGN, not
hyphen-minus, so `grant_figures._save` -- which reports a figure's own layout overlaps by NAMING the
offending tick labels -- printed a line cp1252 cannot encode and raised UnicodeEncodeError.

That is worse than a crash in a reporter, for three reasons:

  * `_save` prints its findings BEFORE calling savefig, so the figure was never written. The failure
    landed exactly on the figures that HAD a layout fault; clean ones were unaffected.
  * The previous render's PNG stayed on disk, so the output directory still looked complete and the
    deck placed a stale figure with nothing to say so.
  * `main()` caught it per-figure and logged one `!!` line, so a nightly exited 0 having silently
    dropped a figure. It cost figure 6 on the 2026-08-27 render before it was traced.

`errors="replace"` is deliberate belt-and-braces: with utf-8 it can never fire, but it means a future
change of encoding degrades a character rather than destroying a figure. A reporter must not be able
to fail harder than the thing it reports on.
"""
from __future__ import annotations

import sys


def use_utf8_stdout() -> bool:
    """Reconfigure stdout/stderr to UTF-8. Returns True if both are UTF-8 afterwards.

    Call at the top of a CLI entry point, never at import: a library import should not mutate the
    interpreter's streams out from under whatever embedded it.
    """
    ok = True
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Not a reconfigurable text stream (pytest capture, a plain pipe wrapper). Nothing to
            # do, and nothing worth failing a render over -- report it and let the caller decide.
            ok = ok and (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8"
    return ok
