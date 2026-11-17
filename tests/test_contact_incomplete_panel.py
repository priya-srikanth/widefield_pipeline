"""The CONTACT vs INCOMPLETE panel (Priya 11-04) builds from synthetic data and respects the n floor."""
import numpy as np

from wfield_local import contact_incomplete_panel as cip


def _by_class(rng, n_contact, n_incomplete):
    out = {"post": {}}
    for cls, n in (("contact", n_contact), ("incomplete", n_incomplete)):
        M = rng.random((6, 6))
        M = M / M.sum(1, keepdims=True)
        out["post"][cls] = {"matrix": M.tolist(), "n_per_true_position": [n] * 6}
    return out


def _rows(rng):
    rows = []
    for an in ("PS92", "PS93", "PS94", "PS95"):
        for s in range(3):
            res = {"scored_centre": "working_rows", "confusion_by_class": {}}
            for lab in (f"{an}_080{s}",):
                C = {c: (np.diag(rng.integers(8, 20, 6)) + rng.integers(0, 3, (6, 6))).tolist()
                     for c in ("contact", "incomplete")}
                res["confusion_by_class"][lab] = C
            rows += cip.rows_from_section_d(res, animal=an, align="cue", epoch_of=lambda lab: "pre")
        for s, ep in enumerate(("acute", "subacute", "chronic")):
            # incomplete is THIN (5 rows per spout) in the acute session
            rows += cip.rows_from_by_class(_by_class(rng, 40, 5 if ep == "acute" else 30), animal=an,
                                           session=f"{an}_09{s}", epoch=ep, align="cue", centre="working_rows")
    return rows


def test_panel_builds_and_respects_the_n_floor(tmp_path):
    rng = np.random.default_rng(0)
    T = cip.panel_table(_rows(rng))
    assert set(T.cls) == {"contact", "incomplete"} and set(T.epoch) == {"pre", "acute", "subacute", "chronic"}
    assert (T.thin == (T.n < cip.MIN_N)).all()
    assert T[(T.epoch == "acute") & (T.cls == "incomplete")].thin.all()
    assert not T[(T.epoch == "acute") & (T.cls == "contact")].thin.any()
    ref = cip.pre_reference(T)
    assert all(kind == "same" for (_v, kind) in ref.values())       # both classes exist pre here
    fig = cip.draw(T, tmp_path / "p.png")
    assert (tmp_path / "p.png").exists()
    # thin cells are omitted from the dots: the acute incomplete x positions carry no scatter points
    ax = fig.axes[1]                                                  # cue | acute
    xs = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections if len(c.get_offsets())])
    inc = [j + cip.CLASS_OFFSET["incomplete"] for j in range(6)]
    assert not any(abs(x - i) < 0.06 for x in xs for i in inc)
    con = [j + cip.CLASS_OFFSET["contact"] for j in range(6)]
    assert sum(abs(x - c) < 0.06 for x in xs for c in con) == 4 * 6        # one dot per animal x spout (non-thin)
    assert not ax.spines["top"].get_visible()


def test_pre_reference_falls_back_to_pre_working():
    rng = np.random.default_rng(1)
    rows = [r for r in _rows(rng) if not (r["epoch"] == "pre" and r["cls"] == "incomplete")]
    T = cip.panel_table(rows)
    ref = cip.pre_reference(T)
    assert {kind for (al, cls, sp), (_v, kind) in ref.items() if cls == "incomplete"} == {"working"}
