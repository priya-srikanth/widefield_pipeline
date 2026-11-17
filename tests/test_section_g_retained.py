"""`section_g_retained`: the n-corrected retained above-chance fraction of Section G within-session decoding."""
from __future__ import annotations

import math

from wfield_local import section_g_retained as sgr


def _within(label, post, pre_mean, n):
    return {"n_positions": n, "chance": 1.0 / n, "post_arm": "engaged only",
            "within_pre_band": {"mean": pre_mean, "sd": 0.05, "min": pre_mean - 0.1, "max": pre_mean + 0.1, "n": 5},
            "post": {label: {"within_accuracy": post, "z": (post - pre_mean) / 0.05,
                             "inside_pre_range": abs(post - pre_mean) <= 0.1}}}


def _session(label, tag, n_lick, posts):
    """posts = {(arm, cond): (post, pre_mean)}; arm 'all' is 6-way, 'lickonly' is n_lick-way."""
    arms = {"all": {"chance": 1 / 6}, "lickonly": {"chance": 1 / n_lick}}
    for (arm, cond), (post, pre) in posts.items():
        arms[arm][f"{cond} within-session"] = _within(label, post, pre, 6 if arm == "all" else n_lick)
    return {"animal": label.split("_")[0], "label": label, "phase_tag": tag, "arms": arms}


def _rec():
    return {
        # 4-way LICK-ONLY session: post at the pre mean, so retained = 1
        "ZZ2_0102": _session("ZZ2_0102", "post", 4, {("lickonly", "post-cue"): (0.7, 0.7),
                                                     ("all", "post-cue"): (1 / 6, 0.6)}),
        # 6-way LICK-ONLY session: post at chance, retained = 0
        "AA1_0101": _session("AA1_0101", "post", 6, {("lickonly", "post-cue"): (1 / 6, 0.6),
                                                     ("lickonly", "post-lick"): (0.5, 0.6)}),
        "AA1_0100": _session("AA1_0100", "excluded", 6, {("lickonly", "pre-cue"): (0.4, 0.5)}),
        "AA1_0099": {"animal": "AA1", "label": "AA1_0099", "phase_tag": "pre", "arms": {}},
    }


def test_retained_one_at_pre_and_zero_at_chance():
    rows = {(d["label"], d["arm"], d["cond"]): d for d in sgr.retained_table(_rec())}
    assert math.isclose(rows[("ZZ2_0102", "lickonly", "post-cue")]["retained"], 1.0)
    assert math.isclose(rows[("AA1_0101", "lickonly", "post-cue")]["retained"], 0.0, abs_tol=1e-12)
    assert math.isclose(rows[("ZZ2_0102", "all", "post-cue")]["retained"], 0.0, abs_tol=1e-12)


def test_n_correction_4_vs_6_way():
    # same raw post and pre accuracies, different n: the ratio uses each session's own chance
    post, pre = 0.5, 0.7
    r4 = sgr.retained_fraction(post, pre, 1 / 4)
    r6 = sgr.retained_fraction(post, pre, 1 / 6)
    assert math.isclose(r4, (0.5 - 0.25) / (0.7 - 0.25))
    assert math.isclose(r6, (0.5 - 1 / 6) / (0.7 - 1 / 6))
    assert r4 < r6           # the 4-way session has less margin above chance left
    assert math.isclose(sgr.dnorm(post, pre, 1 / 4), (0.5 - 0.7) / 0.75)
    assert math.isclose(sgr.dnorm(post, pre, 1 / 6), (0.5 - 0.7) / (5 / 6))
    rows = {(d["label"], d["cond"]): d for d in sgr.retained_table(_rec()) if d["arm"] == "lickonly"}
    assert rows[("ZZ2_0102", "post-cue")]["n_positions"] == 4
    assert math.isclose(rows[("ZZ2_0102", "post-cue")]["chance"], 0.25)
    assert rows[("AA1_0101", "post-lick")]["n_positions"] == 6


def test_no_margin_is_nan():
    assert math.isnan(sgr.retained_fraction(0.3, 0.25, 0.25))


def test_rows_sorted_and_tagged():
    rows = sgr.retained_table(_rec())
    keys = [(d["arm"], d["cond"], d["animal"], d["label"]) for d in rows]
    arm_i = {"all": 0, "lickonly": 1}
    cond_i = {c: i for i, c in enumerate(sgr.CONDS)}
    assert keys == sorted(keys, key=lambda k: (arm_i[k[0]], cond_i[k[1]], k[2], k[3]))
    assert [(d["label"], d["arm"], d["cond"]) for d in rows] == [
        ("ZZ2_0102", "all", "post-cue"),
        ("AA1_0101", "lickonly", "post-cue"), ("ZZ2_0102", "lickonly", "post-cue"),
        ("AA1_0100", "lickonly", "pre-cue"),
        ("AA1_0101", "lickonly", "post-lick")]
    assert {d["label"] for d in rows if d["phase_tag"] == "excluded"} == {"AA1_0100"}
    assert all(d["epoch"] == "excluded" for d in rows if d["phase_tag"] == "excluded")
    assert "AA1_0099" not in {d["label"] for d in rows}          # pre sessions are not rows
    # deterministic regardless of input dict order
    assert sgr.retained_table(dict(reversed(list(_rec().items())))) == rows


def test_table_prints(capsys):
    sgr.print_animal_epoch_table(sgr.retained_table(_rec()))
    assert "LICK-ONLY | post-cue" in capsys.readouterr().out
