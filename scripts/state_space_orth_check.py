"""Is the acute change really "sideways"? Noise-corrected orthogonal share (Priya 11-08: "how are we defining orthogonal"
-> "yes run").

The reported share (`state_space_recovery_trials.frames`, "orth"; FINDINGS 0c item 2 "80-95 %") is, per spout in the
animal's fixed pre space (spout template = bins x top-8 PCs, flattened), |a - (a.u)u| / |a| with a = acute template - pre
template and u = the unit pre -> chronic direction, averaged over spouts. Two problems: it is a LENGTH share (squared:
variance share), and trial noise in the acute template (few trials) adds mostly orthogonal length, so the raw share is
biased toward 1 (a random direction in D dims has share ~sqrt(1 - 1/D), ~.996 at D = 136).

Cross-fitted estimate, per animal, N_SPLIT random trial splits (within session and spout): pre trials in thirds, acute
trials in halves, chronic whole. u from chronic - pre third 3; a1 = acute half 1 - pre third 1, a2 = acute half 2 - pre
third 2: independent noise, so a1.a2 estimates |a|^2 and (a1.u)(a2.u) estimates (a.u)^2 without the noise terms. Noise
in u (chronic + one pre third) only attenuates the along part a little (biases the share UP, conservative for "less
sideways than reported"). Per spout and pooled over spouts (sums of squares). Same fixed space as the main analysis.
Reading: corrected share ~0 = the acute change lies on the pre -> chronic line; ~1 = unrelated to it.
Output: <state_space_recovery>/orth_check{tag}/orth_check.csv (per animal x spout x split) + printed per-animal summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

N_SPLIT = 50
OUT = S.ROOT / f"orth_check{S.tag()}"


def split_template(sess, n_parts, rng):
    """[template of part k] for k < n_parts: each session's trials split at random within spout."""
    parts = [[] for _ in range(n_parts)]
    for lab, ep, day, Y, pos in sess:
        lab_k = np.empty(len(pos), int)
        for p in S.POS:
            m = np.flatnonzero(pos == p)
            lab_k[m] = rng.permutation(np.arange(len(m)) % n_parts)
        for k in range(n_parts):
            parts[k].append((lab, ep, day, Y[lab_k == k], pos[lab_k == k]))
    return [R.template(p) for p in parts]


def animal(an, sess, rng):
    by = {ep: [s for s in sess if s[1] == ep] for ep in ("pre", "acute", "chronic")}
    P, mu = R.fixed_space(R.template(by["pre"]))
    z = lambda T: (T - mu) @ P  # noqa: E731
    Tc = z(R.template(by["chronic"]))
    full = {ep: z(R.template(v)) for ep, v in by.items()}
    rows = []
    for k in range(N_SPLIT):
        p1, p2, p3 = (z(t) for t in split_template(by["pre"], 3, rng))
        a1, a2 = (z(t) for t in split_template(by["acute"], 2, rng))
        for si, sp in enumerate(S.POS):
            v = [x[si].ravel() for x in (p1, p2, p3, a1, a2, Tc, full["pre"], full["acute"], full["chronic"])]
            if not all(np.isfinite(x).all() for x in v):
                continue
            q1, q2, q3, b1, b2, c, fp, fa, fc = v
            u = (c - q3) / np.linalg.norm(c - q3)
            d1, d2 = b1 - q1, b2 - q2
            tot, along = float(d1 @ d2), float((d1 @ u) * (d2 @ u))
            sgn = float(np.sign((d1 + d2) @ u))                         # toward (+) or away from (-) chronic
            pos = float((d1 + d2) @ u / 2 / np.linalg.norm(c - q3))      # acute position along the line (1 = chronic)
            uf = (fc - fp) / np.linalg.norm(fc - fp)                  # the raw (reported) share, full data
            araw = fa - fp
            rows.append({"animal": an, "spout": sp, "split": k, "tot": tot, "along": along, "sign": sgn, "pos": pos,
                         "raw_orth": float(np.linalg.norm(araw - (araw @ uf) * uf) / np.linalg.norm(araw)),
                         "raw_tot": float(araw @ araw), "dim": len(u)})
    return rows


def main():
    rng = np.random.default_rng(0)
    data, _ = S.load_sessions()
    D = pd.DataFrame([r for an, sess in sorted(data.items()) for r in animal(an, sess, rng)])
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "orth_check.csv", index=False)
    D["orth2"] = D.tot - D.along
    D["share_spout"] = np.sqrt(np.clip(D.orth2, 0, None) / D.tot.where(D.tot > 0))
    out = []
    for an, g in D.groupby("animal"):
        g = g.assign(sal=g.sign * np.sqrt(np.clip(g.along, 0, None)))   # signed along length per spout (sign of the sum)
        per_split = g.groupby("split").agg(orth2=("orth2", "sum"), tot=("tot", "sum"), along=("along", "sum"),
                                           sal=("sal", "sum"), mean_spout=("share_spout", "mean"), pos=("pos", "mean"),
                                           n_away=("sign", lambda x: int((x < 0).sum())))
        pooled = np.sqrt(np.clip(per_split.orth2, 0, None) / per_split.tot)
        cosv = np.sign(per_split.sal) * np.sqrt(per_split.along.clip(lower=0) / per_split.tot)
        raw = g[g.split == 0]
        out.append({"animal": an, "raw_share (reported)": raw.raw_orth.mean(),
                    "noise_frac_of_raw_len2": 1 - per_split.tot.median() / raw.raw_tot.sum(),
                    "corr_share_pooled": pooled.median(), "lo5": pooled.quantile(.05), "hi95": pooled.quantile(.95),
                    "corr_share_mean_spout": per_split.mean_spout.median(), "corr_cos_along": cosv.median(), "acute_pos_on_line": per_split.pos.median(),
                    "spouts_away": per_split.n_away.median(),
                    "var_share_orth": (pooled ** 2).median(), "chance_share": np.sqrt(1 - 1 / raw.dim.iloc[0])})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "orth_check_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print(f"=== acute change orthogonal to pre -> chronic ({S.ALIGN_LABEL[S.ALIGN]}, basis {S.BASIS}, {N_SPLIT} splits) ===")
        print(res.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
