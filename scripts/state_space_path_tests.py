"""Path tests for the state-space recovery analysis (Priya 11-07; plan docs/STATE_SPACE_RECOVERY_PLAN.md).

(2) DRIFT DIRECTION. Pre-stroke sessions already move along the pre -> chronic line over the August days (alpha rises
    from day -11 to -3). Per animal, the pre drift direction d = least-squares slope of the pre sessions' spout-mean
    vectors on day (fixed pre space, all 6 spouts x 25 bins x k). Reported: cosine of d with (chronic - pre) and with
    (acute - pre), and the share of (chronic - pre) along d; plus how many days of pre-rate drift would cover
    |chronic - pre|. A split-half check (pre sessions odd vs even days) gives d's own reliability.
(3) REPERTOIRE BEFORE MEAN (Golub 2018 / Oby 2019 analog). Under a cloud that SLIDES along the pre -> chronic line,
    the share of chronic-like trials is set by how far the mean has moved. For each post session: take the animal's
    pre trials (spout by spout), shift them so their mean sits at the session's own alpha on the line, scale their
    spread to the session's spread, and score the share of trials nearer the chronic template. Observed - expected
    > 0 = chronic-like patterns are over-represented in the tails before the mean gets there.

Within spout, fixed per-animal pre space; first-lick trials (contact + incomplete). Exploratory.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import epochs  # noqa: E402

OUT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery/path_tests")


def flatvec(M):
    return M.reshape(-1)


def load_all():
    data = {}
    for f in sorted(glob.glob(S.ITI + "/PS9*.npz")):
        Y, pos, cols, ep, an = S.load(f)
        lab = Path(f).stem
        data.setdefault(an, []).append((lab, ep, epochs.days_since_stroke(lab), Y, pos))
    return {a: sorted(v, key=lambda s: s[2]) for a, v in data.items()}


def drift_direction(sess, P, mu):
    pre = [s for s in sess if s[1] == "pre"]
    V = {s[0]: (R.vec(s[3], s[4], P, mu)[0]) for s in sess}
    ok = np.all([np.isfinite(V[s[0]]).all(axis=(1, 2)) for s in sess], axis=0)   # spouts present everywhere
    vec = {k: flatvec(v[ok]) for k, v in V.items()}

    def slope(group):
        days = np.array([s[2] for s in group], float)
        X = np.stack([vec[s[0]] for s in group])
        dc = days - days.mean()
        return (dc[:, None] * (X - X.mean(0))).sum(0) / (dc ** 2).sum(), np.ptp(days)

    d, span = slope(pre)
    m = {ep: np.mean([vec[s[0]] for s in sess if s[1] == ep], axis=0) for ep in ("pre", "acute", "chronic")}
    dch, dac = m["chronic"] - m["pre"], m["acute"] - m["pre"]
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))  # noqa: E731
    odd, even = pre[0::2], pre[1::2]
    rel = cos(slope(odd)[0], slope(even)[0]) if len(odd) > 2 and len(even) > 2 else np.nan
    return {"cos_drift_chronic": cos(d, dch), "cos_drift_acute": cos(d, dac), "cos_acute_chronic": cos(dac, dch),
            "share_chronic_along_drift": float((dch @ d / np.linalg.norm(d)) ** 2 / (dch @ dch)),
            "days_of_pre_drift_to_chronic": float(np.linalg.norm(dch) / np.linalg.norm(d)),
            "pre_span_days": float(span), "drift_split_half_cos": rel, "n_spouts": int(ok.sum())}


def repertoire(sess, P, mu, rng):
    pre = [s for s in sess if s[1] == "pre"]
    chr_ = [s for s in sess if s[1] == "chronic"]
    Tp, Tc = R.template(pre), R.template(chr_)
    zp, zc = (Tp - mu) @ P, (Tc - mu) @ P
    pre_trials = {p: np.concatenate([((s[3][s[4] == p] - mu) @ P).reshape((s[4] == p).sum(), -1) for s in pre])
                  for p in S.POS}
    out = []
    for lab, ep, day, Y, pos in sess:
        if ep in ("pre", "chronic"):
            continue
        z = (Y - mu) @ P
        obs, exp_ = [], []
        for si, p in enumerate(S.POS):
            m = pos == p
            if m.sum() < R.MIN_TRIALS or not (np.isfinite(zp[si]).all() and np.isfinite(zc[si]).all()):
                continue
            a0, c0 = zp[si].ravel(), zc[si].ravel()
            dlt = c0 - a0
            zt = z[m].reshape(m.sum(), -1)
            alpha = float((zt.mean(0) - a0) @ dlt / (dlt @ dlt))
            near_c = lambda X, c0=c0, a0=a0: ((X - c0) ** 2).sum(1) < ((X - a0) ** 2).sum(1)  # noqa: E731
            obs.append(near_c(zt).mean())
            # the slid cloud: pre trials of this spout, re-centred at the session's own alpha, spread matched
            pt = pre_trials[p]
            pt = pt[rng.choice(len(pt), size=min(len(pt), 4 * m.sum()), replace=False)]
            dev = pt - pt.mean(0)
            sc = np.sqrt(((zt - zt.mean(0)) ** 2).sum(1).mean() / (dev ** 2).sum(1).mean())
            exp_.append(near_c(a0 + alpha * dlt + sc * dev).mean())
        if obs:
            out.append({"label": lab, "epoch": ep, "day": day, "observed": float(np.mean(obs)),
                        "expected_slid": float(np.mean(exp_)), "excess": float(np.mean(obs) - np.mean(exp_))})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = load_all()
    dr, rp = [], []
    for an, sess in sorted(data.items()):
        pre = [s for s in sess if s[1] == "pre"]
        P, mu = R.fixed_space(R.template(pre))
        dr.append({"animal": an, **drift_direction(sess, P, mu)})
        rp += [{"animal": an, **r} for r in repertoire(sess, P, mu, rng)]
    dr, rp = pd.DataFrame(dr), pd.DataFrame(rp)
    dr.to_csv(OUT / "drift_direction.csv", index=False)
    rp.to_csv(OUT / "repertoire_before_mean.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3, "display.max_columns", 20):
        print("=== (2) pre-stroke drift direction vs the post-stroke changes ===")
        print(dr.to_string(index=False))
        print("\n=== (3) chronic-like share: observed vs a cloud sliding to the session's own position ===")
        print(rp.groupby(["animal", "epoch"])[["observed", "expected_slid", "excess"]].mean()
              .reindex(["acute", "subacute"], level="epoch"))
        print("sessions with excess > 0, by epoch:", rp.assign(up=rp.excess > 0).groupby("epoch").up.agg(["sum", "count"]).to_dict())


if __name__ == "__main__":
    main()
