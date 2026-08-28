"""The layout reporter must not be able to destroy the figure it is reporting on.

`grant_figures._save` prints the offending tick labels when it finds an overlap, and matplotlib
writes a negative tick as U+2212. On Windows stdout is cp1252 -- on a console AND redirected -- so
that print raised UnicodeEncodeError BEFORE savefig ran. The figure was never written, the previous
render's PNG stayed on disk looking current, and the nightly logged one `!!` line and exited 0.
"""
import subprocess
import sys

from wfield_local.console import use_utf8_stdout


def test_reconfigures_streams_to_utf8():
    assert use_utf8_stdout() or True          # never raises, whatever the stream is
    for s in (sys.stdout, sys.stderr):
        enc = (getattr(s, "encoding", "") or "").lower().replace("-", "")
        assert enc in ("utf8", "") or enc.startswith("utf"), enc


def test_a_minus_sign_survives_a_redirected_child_process():
    """The real failure: a child whose stdout is a PIPE (as under the nightly) printing U+2212.

    Without `use_utf8_stdout` this is exactly the UnicodeEncodeError that cost figure 6.
    """
    code = ("from wfield_local.console import use_utf8_stdout;"
            "use_utf8_stdout();"
            "print(\"[layout] ax11.xtick '\u22121.00' x crowds '\u22120.75'\")")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "\u22121.00".encode() in r.stdout, r.stdout


def test_without_the_fix_the_same_print_is_what_broke():
    """Pins WHY the helper exists: the bare print fails on a cp1252 stream, so this is a real bug
    being fixed and not a precaution. Skipped where the default encoding already handles it."""
    import pytest
    code = ("import sys;"
            "sys.stdout.reconfigure(encoding='cp1252');"
            "print('\u22121.00')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
    if r.returncode == 0:
        pytest.skip("this interpreter's cp1252 stream did not reject U+2212")
    assert b"UnicodeEncodeError" in r.stderr
