"""scripts/patch_splice.py: other sessions' rows stay byte-identical; headers must match."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_splice as ps  # noqa: E402

OLD = ["label,x\n", "A,1.000000001\n", "B,2\n", "B,3\n", "C,4\n"]
NEW = ["label,x\n", "B,20\n", "B,30\n", "B,31\n"]


def test_splice_replaces_only_the_labels_rows():
    out, rem, add = ps.splice_lines(OLD, NEW, {"B"})
    assert (rem, add) == (2, 3)
    assert out == ["label,x\n", "A,1.000000001\n", "B,20\n", "B,30\n", "B,31\n", "C,4\n"]


def test_header_mismatch_and_missing_rows_refuse():
    with pytest.raises(ValueError, match="header"):
        ps.splice_lines(OLD, ["label,y\n", "B,1\n"], {"B"})
    with pytest.raises(ValueError, match="no rows"):
        ps.splice_lines(OLD, ["label,x\n", "Z,1\n"], {"B"})
