"""The OOD control must describe the SAME model it is a control for, trained on pre-stroke only.

TWO BUGS IN ONE LINE, found 2026-08-28 while packaging the frozen models.

`ood_control` began `clf = _pipe().fit(XE, YE)` over EVERY pooled session, so once post-stroke nights
registered they entered its training set -- the identical contamination fixed in the main arm on
2026-08-26, still live in the control and called from three lines below the frozen load that fixed
it. The LOSO and permutation references ran over all sessions too, so the "no-information floor"
this arm exists to establish was itself partly computed on post-stroke data.

And it FIT ITS OWN MODEL. That defeats the purpose. This control's entire claim is "THIS decoder's
confidence is not evidence" -- a control that fits a second decoder is characterising something other
than the decoder being defended. Agreement between the two rested on both happening to call `_pipe()`
with the same rows, which is exactly the kind of coincidence the frozen store exists to replace with
an identity.
"""
import inspect

import numpy as np
import pytest

from wfield_local import locanmf_frozen_decoder as D


def test_it_accepts_the_frozen_models_and_the_prestroke_index():
    sig = inspect.signature(D.ood_control).parameters
    for p in ("models", "pre_i", "GU", "labels"):
        assert p in sig, f"ood_control cannot be told about {p}"


def test_the_caller_passes_them():
    """A control that CAN take the frozen models but is not given them is no better than before."""
    src = inspect.getsource(D.pooled_frozen_loso)
    call = [ln for ln in src.splitlines() if "ood_control(" in ln]
    assert call, "pooled_frozen_loso no longer calls ood_control"
    line = call[0]
    for arg in ("models=models", "pre_i=pre_i", "GU=GU", "labels=kept"):
        assert arg in line, f"ood_control is called without {arg}: {line.strip()}"


def test_no_fit_over_the_unmasked_matrix_remains():
    """The literal regression: `_pipe().fit(XE, YE)` with no mask is the bug.

    COMMENTS ARE STRIPPED FIRST, because the fix documents the old line verbatim and a naive
    substring search matches the explanation instead of the code -- a test that fails on its own
    changelog teaches the next person to delete the changelog.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in inspect.getsource(D.ood_control).splitlines())
    assert "_pipe().fit(XE, YE)" not in code, "still fits over every pooled session"
    assert "_pipe().fit(XE[m], YE[m])" in code, "the masked fallback fit is gone"


def _synthetic(n_pre=3, n_post=2, n=40, d=5, seed=0):
    """Pre-stroke sessions carry real position signal; post-stroke sessions are pure noise.

    Constructed so contamination is DETECTABLE: a model trained on both is measurably worse on the
    pre-stroke rows than one trained on pre alone. If the restriction were dropped the entropy
    references would move, which is what the test below asserts.
    """
    rng = np.random.RandomState(seed)
    XE, YE, GE = [], [], []
    for g in range(n_pre + n_post):
        y = rng.randint(0, 6, n)
        sig = np.eye(6)[y] @ rng.RandomState if False else np.eye(6)[y][:, :d]
        X = rng.randn(n, d) * 0.4 + (sig * 3.0 if g < n_pre else 0.0)
        XE.append(X); YE.append(y); GE.append(np.full(n, g))
    return (np.vstack(XE), np.concatenate(YE), np.concatenate(GE))


def test_post_stroke_sessions_do_not_enter_the_references():
    """END TO END. The engaged entropy must be identical whether or not post-stroke sessions are in
    the pool -- that is what "the reference is pre-stroke" means operationally."""
    XE, YE, GE = _synthetic(n_pre=3, n_post=2)
    pre_i = [0, 1, 2]
    XU, YU, GU = XE[:0], YE[:0], np.array([], int)

    with_post = D.ood_control(XE, YE, GE, XU, YU, GU=GU, pre_i=pre_i)
    m = np.isin(GE, pre_i)
    pre_only = D.ood_control(XE[m], YE[m], GE[m], XU, YU, GU=GU, pre_i=pre_i)

    assert with_post["engaged_H"] == pytest.approx(pre_only["engaged_H"])
    assert with_post["shuffle_H"] == pytest.approx(pre_only["shuffle_H"])


def test_the_nolick_arm_is_prestroke_too():
    """This function's own docstring calls the no-lick trials "the nearest available PRE-STROKE
    analogue of a failed post-stroke attempt". Post-stroke no-lick trials are what it is the
    reference FOR, so including them would make the reference partly the thing being measured."""
    XE, YE, GE = _synthetic(n_pre=3, n_post=2)
    pre_i = [0, 1, 2]
    rng = np.random.RandomState(1)
    XU = rng.randn(120, XE.shape[1]); YU = rng.randint(0, 6, 120)
    GU = np.repeat([0, 1, 2, 3, 4], 24)

    out = D.ood_control(XE, YE, GE, XU, YU, GU=GU, pre_i=pre_i)
    assert out["n_nolick"] == 72, f"post-stroke no-lick trials entered the reference: {out['n_nolick']}"


def test_it_uses_the_stored_model_when_given_one():
    """Given the frozen models it must not fit its own -- the fit callable here would be a second
    decoder, and then the control would not be about the decoder."""
    XE, YE, GE = _synthetic(n_pre=3, n_post=0)
    pre_i = [0, 1, 2]
    labels = ["S0", "S1", "S2"]
    fitted = D._pipe().fit(XE, YE)
    models = {"full": fitted, "loso": {lab: fitted for lab in labels}}
    out = D.ood_control(XE, YE, GE, XE[:0], YE[:0], GU=np.array([], int),
                        pre_i=pre_i, models=models, labels=labels)
    assert "engaged_H" in out and 0.0 <= out["engaged_H"] <= 1.0
