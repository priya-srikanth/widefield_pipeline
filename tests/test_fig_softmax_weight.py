"""fig_softmax weights each session's soft-weight row by the licks AIMED at the target (bug B2, 11-09).

`combined_encoding` stores bins["confusion"] = pd.crosstab(target, exec).to_dict(), which is keyed {exec: {target: n}};
the old weight `sum(conf[t].values())` was the number of licks that LANDED in bin t.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rebuild_figs"))
import fig_softmax as fs  # noqa: E402

SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]


def _conf(tl, el):
    """Exactly the producer's expression (combined_encoding.py, res["bins"]["confusion"])."""
    return pd.crosstab(pd.Series(tl, name="target"), pd.Series(el, name="exec")).to_dict()


def test_crosstab_to_dict_is_keyed_by_exec():
    # 5 licks aimed at far_contra, 4 of them landing at far_mid; 1 lick aimed at far_mid landing there
    tl = ["far_contra"] * 5 + ["far_mid"]
    el = ["far_mid"] * 4 + ["far_contra", "far_mid"]
    conf = _conf(tl, el)
    assert conf["far_mid"] == {"far_contra": 4, "far_mid": 1}          # outer key = EXECUTED bin
    assert sum(conf["far_contra"].values()) == 1                      # the old "licks at target far_contra": wrong
    assert fs.target_lick_counts(conf) == {"far_contra": 5, "far_mid": 1}


def test_session_weight_uses_target_counts():
    """Two sessions, target far_contra: A has 9 aimed licks (all landing far_mid), B has 1 (landing on target). The
    pooled row must be the per-lick mean, 0.9 * A + 0.1 * B, not weighted by landings (A 0 -> weight 1, B 1)."""
    def sess(tl, el, sw):
        return {"animal": "PS92", "epoch": "pre", "bins": {"confusion": _conf(tl, el), "soft_weight": {"far_contra": sw}}}
    wa = dict(zip(SP, [0, 0, 0, 0, 1.0, 0]))
    wb = dict(zip(SP, [0, 0, 0, 0, 0, 1.0]))
    A = sess(["far_contra"] * 9 + ["far_mid"], ["far_mid"] * 10, wa)
    B = sess(["far_contra"], ["far_contra"], wb)
    got = {}
    orig_save, orig_imshow = fs.save_png_svg, None
    import matplotlib.axes
    orig_imshow = matplotlib.axes.Axes.imshow

    def spy(self, M, *a, **k):
        got.setdefault("M", np.array(M))
        return orig_imshow(self, M, *a, **k)

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop
    matplotlib.axes.Axes.imshow = spy
    fs.save_png_svg = stop
    try:
        fs.draw({"sessions": [A, B], "bins": {}}, "unused")
    except Stop:
        pass
    finally:
        matplotlib.axes.Axes.imshow = orig_imshow
        fs.save_png_svg = orig_save
    row = got["M"][SP.index("far_contra")]
    assert np.allclose(row[SP.index("far_mid")], 0.9) and np.allclose(row[SP.index("far_contra")], 0.1)
