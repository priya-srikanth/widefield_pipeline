"""All-session encoder (shared kernels + per-epoch kernel changes): the augmented Gram assembled from per-session
statistics equals the explicit augmented design's, and a planted kernel change is recovered in the right group."""
import numpy as np

from wfield_local import allsession_encoder as ae


def test_augment_matches_explicit_design():
    rng = np.random.default_rng(0)
    n, p, r, n_ep = 40, 3, 2, 3
    for e in range(n_ep + 1):
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=(n, r))
        Z = np.zeros((n, (n_ep + 1) * p))
        Z[:, :p] = X
        if e:
            Z[:, e * p:(e + 1) * p] = X
        Ga, ba = ae.augment(X.T @ X, X.T @ Y, e, n_ep)
        assert np.allclose(Ga, Z.T @ Z) and np.allclose(ba, Z.T @ Y)


def _sessions(rng, change_group_cols, n=400, p=4, eps=("pre",) * 4 + ("acute",) * 4, delta=1.5, noise=0.5):
    beta = rng.normal(size=(p, 1))
    out = []
    for i, ep in enumerate(eps):
        X = rng.normal(size=(n, p))
        b = beta.copy()
        if ep == "acute":
            b[change_group_cols] += delta
        y = X @ b + noise * rng.normal(size=(n, 1))
        out.append({"label": f"s{i}", "epoch": ep, "G": X.T @ X, "b": X.T @ y, "yy": (y ** 2).sum(0)})
    return out


def test_planted_change_recovered_in_its_group():
    rng = np.random.default_rng(1)
    groups = np.array(["lick", "lick", "contact", "other"])
    pen = np.array(["a", "a", "a", "a"])
    ok = np.ones(4, bool)
    sess = _sessions(rng, change_group_cols=[2])                  # the CONTACT kernel changes acutely
    f = ae.fit_animal(sess, {"a": 1.0}, groups, pen, ok, ("acute",))
    acute = [s["label"] for s in sess if s["epoch"] == "acute"]
    yy = {s["label"]: s["yy"][0] for s in sess}

    def gain(m):
        return np.mean([(f["sse"][m][k][0] - f["sse"]["all"][k][0]) / yy[k] for k in acute])
    assert gain("shared") > 0.05                                  # the change matters held out
    assert gain("no_d_contact") > 0.05                            # ... and it is the contact change
    assert abs(gain("no_d_lick")) < 0.01 and abs(gain("no_d_other")) < 0.01
    assert f["dbeta"]["acute"][2, 0] > 1.0 and np.abs(f["dbeta"]["acute"][[0, 1, 3], 0]).max() < 0.3


def test_no_change_gives_no_gain():
    rng = np.random.default_rng(2)
    groups = np.array(["lick", "lick", "contact", "other"])
    sess = _sessions(rng, change_group_cols=[], delta=0.0)
    f = ae.fit_animal(sess, {"a": 1.0}, groups, np.array(["a"] * 4), np.ones(4, bool), ("acute",))
    acute = [s["label"] for s in sess if s["epoch"] == "acute"]
    g = np.mean([(f["sse"]["shared"][k][0] - f["sse"]["all"][k][0]) / s["yy"][0]
                 for k, s in zip(acute, [s for s in sess if s["epoch"] == "acute"])])
    assert g < 0.005                                              # held-out: no free gain from extra columns


def test_change_ref_is_measured_from_the_reference():
    """change_ref (10-24): reference sessions gain exactly 0 by construction; an epoch with a planted change gains; an
    unchanged epoch (the drift null's role) gains ~0 -- while the old `change` (vs a pooled refit) also credits the
    REFERENCE sessions once another epoch pulls the pooled kernels away (the flaw it replaces)."""
    rng = np.random.default_rng(3)
    groups = np.array(["lick", "lick", "contact", "other"])
    sess = _sessions(rng, change_group_cols=[2], eps=("pre",) * 4 + ("pre_late",) * 4 + ("acute",) * 4, delta=2.0)
    f = ae.fit_animal(sess, {"a": 1.0}, groups, np.array(["a"] * 4), np.ones(4, bool), ("pre_late", "acute"))
    yy = {s["label"]: s["yy"][0] for s in sess}

    def gain(m, ep, base="all"):
        ks = [s["label"] for s in sess if s["epoch"] == ep]
        return np.mean([(f["sse"][m][k][0] - f["sse"][base][k][0]) / yy[k] for k in ks])
    assert abs(gain("ref", "pre")) < 1e-9                         # reference: no change block -> all == ref
    assert gain("ref", "acute") > 0.05                            # planted change, measured from the reference
    assert abs(gain("ref", "pre_late")) < 0.01                    # unchanged epoch ~ 0 (the matched drift null)
    assert gain("shared", "pre") > 0.01                           # the old readout credits the reference too
