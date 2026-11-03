"""ONE stored helper for the ITI / cued-lick ridge read-outs (11-02 pass 2; Priya: "yes cache all frozen decoders").

Every ITI / cued-lick script (`scripts/iti_licks.py` and the ~11 `scripts/iti_*` / `lick_gain_prediction` copies,
`docs/status/FROZEN_ALL_20261102.md` section 2 row 9) fits the same pre-stroke reference: StandardScaler + RidgeCV on
the animal's PRE sessions, once per held-out pre session (the pre column, scored leave-one-session-out) and once on all
of them (what every post-stroke session is scored by). `pre_loso_ridge` is that construction, and it goes through
`frozen_models.load_or_fit` (kind `iti_readout`) unless WIDEFIELD_ITI_FROZEN=0 (false / no / off), which fits in
memory exactly as the scripts did.

IDENTITY. ``tag`` names the read-out (script, lick set, target, lag / window / bin, feature set); the spec also
carries the alphas, the number of targets and a DATA SIGNATURE -- sha1 of the training arrays in training order --
because these read-outs train on derived per-lick tables (`iti_licks/sessions*/*.npz`, LP licks, z-scoring choices)
that `frozen_models._session_sig` (U_atlas / SVTcorr stats) cannot see. A re-extraction therefore mints a new id
(and `siblings` reports SPEC-CHANGED) instead of serving a model fitted on other data.

LP-ONLY BY NATURE: the licks are LP video licks (the read-outs exist only in the LP run), so there is no DAQ-trained
version; the spec always carries lick_source=lp.
"""
from __future__ import annotations

import hashlib

import numpy as np

ITI_FROZEN_ENV = "WIDEFIELD_ITI_FROZEN"


def ridge_fit(Xt, yt, alphas, k=None):
    """(scaler, model): StandardScaler then RidgeCV(alphas); with ``k``, one RidgeCV per target column (a list)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xt)
    Z = sc.transform(Xt)
    if k is None:
        return sc, RidgeCV(alphas=alphas).fit(Z, yt)
    return sc, [RidgeCV(alphas=alphas).fit(Z, yt[:, j]) for j in range(k)]


def fit_all(pre, X, Y, alphas, k=None, loso=True):
    """{held: (scaler, model)} for held in [*pre, None]: training = ``pre`` minus ``held`` IN ``pre`` ORDER (None = all).
    ``loso=False``: the all-pre model only ({None: ...}), for read-outs that never score a pre session held out."""
    out = {}
    for held in ([*pre, None] if loso else [None]):
        trn = [q for q in pre if q != held]
        out[held] = ridge_fit(np.vstack([X[q] for q in trn]), np.concatenate([Y[q] for q in trn]), alphas, k)
    return out


def data_sig(pre, X, Y) -> str:
    h = hashlib.sha1()
    for q in pre:
        for a in (X[q], Y[q]):
            a = np.ascontiguousarray(a)
            h.update(f"{q}|{a.dtype}|{a.shape}|".encode())
            h.update(a.tobytes())
    return h.hexdigest()[:16]


def pre_loso_ridge(animal, tag, pre, X, Y, alphas, *, k=None, loso=True, log=print):
    """The stored {held: (scaler, model)} for one read-out (see the module docstring), or the in-memory fit when
    WIDEFIELD_ITI_FROZEN=0. ``X`` / ``Y``: {session label: features (n, p)} / {label: targets (n,) or (n, k)}."""
    from wfield_local import frozen_models
    pre = list(pre)
    alphas = np.asarray(alphas, float)
    if not frozen_models.family_on(ITI_FROZEN_ENV):
        return fit_all(pre, X, Y, alphas, k, loso)
    safe = "".join(c if (c.isalnum() or c in "._+-") else "-" for c in str(tag))   # the align names a directory
    spec = frozen_models.make_spec(animal, "iti_readout", align=safe, source="iti", train_labels=pre,
                                   post_s=0.0, n_features=int(np.asarray(X[pre[0]]).shape[1]))
    spec["tag"] = str(tag)
    spec["lick_source"] = "lp"                           # LP-only by nature (module docstring)
    spec["alphas"] = [float(a) for a in alphas]
    spec["n_targets"] = None if k is None else int(k)
    spec["loso"] = bool(loso)
    spec["train_order"] = pre
    spec["data_sig"] = data_sig(pre, X, Y)
    payload, status = frozen_models.load_or_fit(spec, lambda: fit_all(pre, X, Y, alphas, k, loso),
                                                meta={"n_train": int(sum(len(Y[q]) for q in pre))}, log=log)
    if status != frozen_models.STATUS_HIT:
        log(f"  [iti] {animal} {tag}: iti_readout {frozen_models._slug(spec)} {status}")
    return payload
