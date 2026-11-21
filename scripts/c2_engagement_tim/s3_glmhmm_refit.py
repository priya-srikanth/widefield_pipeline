"""Step 3 (item 10a): GLM-HMM refits with inputs a lateralized motor deficit does not drive; six-position cohort.

Production model (`glm_hmm`, K5, design v2, pooled over all animals and all August-onward sessions): per trial cycle
five categorical outputs -- OUTCOME (attempt = contact + incomplete / jaw-only / no response), pre-cue lick (y/n), ITI
lick (y/n), running in the pre-cue and in the ITI (still / some / running) -- each a state-specific multinomial
logistic function of [shared animal offsets | state intercept, 5 spout dummies, spout-moved]; sticky transitions.

Sequences are rebuilt from the production posterior table (it stores every trial's class, licks and running
fractions; no session reload) and the production fit is first REPRODUCED (``--check``) before any variant is fitted.
Variants (same class, K5, design v2, `glm_hmm.GLMHMM` EM; ONE run warm-started from the production parameters, as
`glm_hmm.run_frozen`'s LOSO refits, because 5 random restarts x up to 200 EM iterations at ~13 s per iteration would
take hours (timed 11-20); EM stops at the production tolerance or 150 iterations):
  ipsimid   OUTCOME observed only on ipsilesional + middle spouts (close/far L + center; all lesions left); on the
            contralesional spouts the outcome is unobserved (emission 1), so their state is inferred from the chain and
            from the non-spout outputs (licks, running) only.
  init      OUTCOME = response INITIATION: responded (contact, incomplete or jaw-only = the jaw opened) vs no response;
            reaching the spout (contact vs jaw-only) no longer enters.
Roles per refit from its own state profiles (`behavior_states.glmhmm_roles` rule: engaged = attempt rate >= 0.9 on
the ipsi/mid trials for 'ipsimid'; disengaged = the highest no-response rate of the rest).
Output: glmhmm_refit_<variant>_K5.csv.gz (label, trial_id, p_state*, role) next to this script.
Usage: WF_REPO=... python s3_glmhmm_refit.py <out_dir> [--time] [--check] [--variants ipsimid init]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from wfield_local import glm_hmm as gh  # noqa: E402

K = 5
CONTRA = {"close_R", "far_R"}


def sequences(variant="prod"):
    raw = pd.read_csv(Path(gh.out_dir()) / "states_pooled_all_K5_v2.csv.gz")
    labs = sorted(raw.label.unique())
    animals = sorted({lab.split("_")[0] for lab in labs})
    om = gh.OUTCOMES["3"]
    data = {}
    for lab in labs:
        S = raw[raw.label == lab].sort_values("trial_id").reset_index(drop=True)
        y = S.cls.map(om).fillna(-1).astype(int).to_numpy()
        if variant == "ipsimid":
            y = np.where(S.position.isin(CONTRA).to_numpy(), -1, y)
        elif variant == "init":
            y = np.where(S.cls.isin(["contact", "incomplete", "jaw_only"]), 0, np.where(S.cls.eq("no_response"), 1, -1))
        rl = lambda f: np.array([gh._run_level(v) for v in f])  # noqa: E731
        Y = np.column_stack([y, S.precue_lick.fillna(-1).astype(int), S.iti_lick.fillna(-1).astype(int),
                             rl(S.run_frac_precue.to_numpy(float)), rl(S.run_frac_iti.to_numpy(float))])
        an = lab.split("_")[0]
        A = np.array([[1.0 if an == a else 0.0 for a in animals]] * len(S))
        P = np.zeros((len(S), len(gh.POSITIONS) - 1))
        for t, i in enumerate(S.position.map({p: i for i, p in enumerate(gh.POSITIONS)}).fillna(-1).astype(int)):
            if i > 0:
                P[t, i - 1] = 1.0
        pos = S.position.to_numpy()
        moved = np.r_[0.0, (pos[1:] != pos[:-1]).astype(float)]
        X = np.hstack([A[:, 1:], np.ones((len(S), 1)), P, moved[:, None]])
        data[lab] = (X, Y, S[["label", "trial_id", "position", "cls"]])
    gh.N_SHARED = len(animals) - 1
    return data, raw, len(animals) - 1


def prod_model(ns):
    """The production K5 v2 fit (params json, table state order) as a `glm_hmm.GLMHMM`."""
    import json
    P = json.loads((Path(gh.out_dir()) / "params_pooled_all_K5_v2.json").read_text())
    m = gh.GLMHMM(K, n_cls=P["n_classes"], n_shared=ns)
    m.W = [np.asarray(w, float) for w in P["W"]]
    m.B = [np.asarray(b, float) for b in P["B"]]
    m.A, m.pi = np.asarray(P["A"], float), np.asarray(P["pi"], float)
    return m


def warm_init(variant, ns):
    """Production parameters as the EM start (one run, as `glm_hmm.run_frozen`'s LOSO refits). 'init': the outcome
    weights become the no-response-vs-rest logit of the production fit."""
    m = prod_model(ns)
    if variant == "init":
        m.W[0] = m.W[0][:, 1:2, :].copy()
        m.B[0] = m.B[0][1:2, :].copy()
        m.C = [2, 2, 2, 3, 3]
    return m


def fit(data, n_cls, ns, n_restarts=gh.N_RESTARTS, max_iter=None, init=None):
    if max_iter:
        gh.MAX_ITER = max_iter
    labs = sorted(data)
    m = gh.GLMHMM(K, n_cls=n_cls, n_shared=ns).fit([data[lab][:2] for lab in labs], n_restarts=n_restarts, init=init)
    order = m.engagement_order(np.vstack([data[lab][0] for lab in labs]).mean(0))
    post = m.posteriors([data[lab][:2] for lab in labs])
    rows = []
    for lab, g in zip(labs, post):
        T = data[lab][2].copy()
        g = g[:, order]
        for k in range(K):
            T[f"p_state{k}"] = g[:, k]
        rows.append(T)
    return pd.concat(rows, ignore_index=True), m


def roles(T, variant):
    att = T.cls.isin(["contact", "incomplete"]).to_numpy(float)
    nr = T.cls.eq("no_response").to_numpy(float)
    use = ~T.position.isin(CONTRA).to_numpy() if variant == "ipsimid" else np.ones(len(T), bool)
    pc = [c for c in T if c.startswith("p_state")]
    prof = {int(c[7:]): (np.average(att[use], weights=T[c].to_numpy()[use]), np.average(nr[use], weights=T[c].to_numpy()[use]))
            for c in pc}
    eng = [k for k, v in prof.items() if v[0] >= 0.9]
    rest = {k: v for k, v in prof.items() if k not in eng}
    dis = max(rest, key=lambda k: rest[k][1]) if rest else None
    R = {k: "engaged" if k in eng else "disengaged" if k == dis else "partial" for k in prof}
    return R, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["ipsimid", "init"])
    a = ap.parse_args()
    out = Path(a.out)
    if a.time:
        data, raw, ns = sequences("prod")
        t0 = time.time()
        fit(data, [2 + 1, 2, 2, 3, 3], ns, n_restarts=1, max_iter=5)
        print(f"5 EM iterations, 1 restart: {time.time() - t0:.0f} s (production = up to {gh.N_RESTARTS} x "
              f"{200} iterations)", flush=True)
        return
    if a.check:                     # rebuilt sequences + production parameters reproduce the production posteriors
        data, raw, ns = sequences("prod")
        m = prod_model(ns)
        labs = sorted(data)
        post = m.posteriors([data[lab][:2] for lab in labs])
        T = pd.concat([data[lab][2].assign(**{f"p_state{k}": g[:, k] for k in range(K)}) for lab, g in zip(labs, post)],
                      ignore_index=True).sort_values(["label", "trial_id"]).reset_index(drop=True)
        old = raw.sort_values(["label", "trial_id"]).reset_index(drop=True)
        d = np.abs(old[[f"p_state{k}" for k in range(K)]].to_numpy() - T[[f"p_state{k}" for k in range(K)]].to_numpy()).max()
        print(f"production reproduced from rebuilt sequences: max |posterior diff| = {d:.2e}", flush=True)
    for v in a.variants:
        f = out / f"glmhmm_refit_{v}_K5.csv.gz"
        if f.exists():
            print("exists, skipped:", f)
            continue
        data, raw, ns = sequences(v)
        n_cls = [2 if v == "init" else 3, 2, 2, 3, 3]
        t0 = time.time()
        T, m = fit(data, n_cls, ns, init=warm_init(v, ns), max_iter=150)
        R, prof = roles(T, v)
        pc = [c for c in T if c.startswith("p_state")]
        T["state"] = [int(pc[i][7:]) for i in T[pc].to_numpy().argmax(1)]
        T["role"] = T.state.map(R)
        T.to_csv(f, index=False)
        print(v, f"{time.time() - t0:.0f} s; roles", R, "profiles (attempt, no-response)",
              {k: tuple(round(x, 3) for x in p) for k, p in prof.items()}, flush=True)


if __name__ == "__main__":
    main()
