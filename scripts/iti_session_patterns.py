"""FULLY PER-SESSION Haufe pattern maps (Priya 10-27: "can we not do per-session haufe maps"; map-level permutation and
amplitude-normalized change maps; per-spout-position version of the position maps).

Every session gets its own decoder, fit on THAT session's licks only (ridge, penalty by internal CV), and its own
activation pattern A = Cov(X, yhat) / Var(yhat) (Haufe et al. 2014) on the same licks, mapped to pixels with the animal's
pre-stroke SVD basis (W @ A; features as `iti_decoder_patterns`: 200 components, lags 0..0.2 s from the lick peak).
Because no map depends on which sessions are labeled pre / post, session labels can be permuted (`iti_session_pattern_
tests.py`). Targets, per lick set (ITI; cued = all post-cue licks, `iti_licks.extract`):
  lat, sag            landing position (lateral + = contralesional; sagittal extension)
  wlat, wsag          WITHIN-SPOUT: landing minus the session's mean landing at that spout (cued: target spout; ITI:
                      preceding trial's spout); spouts with < MIN_SPOUT licks dropped -> the target is held fixed, so the
                      map is the EXECUTED direction only
  sp_<spout>          one-vs-rest indicator of the spout (cued: target; ITI: preceding), >= MIN_SPOUT licks of it
Output <E>/iti_licks/session_patterns/<animal>.pkl = {label: {"epoch", "day", "n": {target: licks},
"maps": {f"{set}/{target}": float32 pixels}}}. Cached per animal (delete the file to recompute).
Usage: python scripts/iti_session_patterns.py
"""
from __future__ import annotations

import pickle
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
from iti_decoder_patterns import SPATIAL  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
MIN_LICKS = 40
MIN_SPOUT = 10
OUT = il.OUT / "session_patterns"


def _pattern(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    if len(y) < MIN_LICKS or np.std(y) <= 0:
        return None
    sc = StandardScaler().fit(X)
    m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(X), y)
    yh = m.predict(sc.transform(X))
    if yh.var() <= 0:
        return None
    return ((X - X.mean(0)).T @ (yh - yh.mean())) / (len(yh) - 1) / yh.var(ddof=1)


def _within(y: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Landing minus the session's mean landing at the same spout; keep spouts with >= MIN_SPOUT licks."""
    keep = np.zeros(len(y), bool)
    out = np.full(len(y), np.nan)
    for p in np.unique(pos):
        k = pos == p
        if p in POS and k.sum() >= MIN_SPOUT:
            out[k] = y[k] - y[k].mean()
            keep |= k
    return out, keep


def family_sentence(fam: str) -> str:
    """The licks behind one map family of these per-session maps (Priya 11-09; read by iti_session_pattern_tests /
    iti_session_cluster_tests, which draw them). fam: position | within_spout | spout_cued | spout_iti (or a map key
    prefix "iti" / "trial")."""
    import iti_where_decoder as wd
    from wfield_local import trial_set_text as tst
    kw = dict(tracked="the map's target coordinate", min_n=MIN_LICKS, unit=" (per map)")
    if fam == "all":               # every family together (whole-map tests, similarity)
        kw.update(tracked="", unit=" per map (position maps: target coordinate tracked; spout maps: licks at the six "
                                   "spouts)")
        ex = [f"within-spout and one-vs-rest spout maps: spouts with >= {MIN_SPOUT} licks in the session"]
        return tst.union(wd.iti_caption("", extra=ex, **kw), wd.cued_caption(extra=ex, **kw))
    if fam in ("spout_cued", "spout_iti"):
        kw.update(tracked="", unit=" at the six spouts")
        sp = [f"one-vs-rest map per spout with >= {MIN_SPOUT} licks of it"]
        return (wd.cued_caption(extra=sp, **kw) if fam == "spout_cued"
                else wd.iti_caption("", extra=sp, spout="spout = the previous trial's spout", **kw))
    ex = ([f"within-spout target: spouts with >= {MIN_SPOUT} licks in the session (one of the six)"]
          if fam == "within_spout" else [])
    return tst.union(wd.iti_caption("", extra=ex, **kw), wd.cued_caption(extra=ex, **kw))


def main() -> int:
    from recovery_timing import series

    from wfield_local import orofacial_cohort as oc
    OUT.mkdir(parents=True, exist_ok=True)
    days = dict(zip(series("lick").label, series("lick").day))
    todo = [a for a in AN if not (OUT / f"{a}.pkl").exists()]
    if not todo:
        print("all cached")
        return 0
    D = wd.load("svdbasis")
    for a in todo:
        W = np.asarray(pickle.loads(il.te_basis_spatial(a, SPATIAL).read_bytes())["P"], np.float32)
        res = {}
        for x, d in sorted(D.items()):
            if d["animal"] != a:
                continue
            rec = {"epoch": d["epoch"], "day": days.get(x, np.nan), "n": {}, "maps": {}}
            for st in ("iti", "trial"):
                X, C = d[st]
                pos = np.array([p if p in POS or "_" not in p else oc.pos_key(p, "L") for p in d[f"{st}_pos"]])
                for tg, k in (("lat", 0), ("sag", 3)):
                    ok = np.isfinite(C[:, k])
                    A = _pattern(X[ok], C[ok, k])
                    if A is not None:
                        rec["maps"][f"{st}/{tg}"] = (W @ A.astype(np.float32))
                        rec["n"][f"{st}/{tg}"] = int(ok.sum())
                    yw, keep = _within(np.where(ok, C[:, k], np.nan), np.where(ok, pos, ""))
                    keep &= ok
                    A = _pattern(X[keep], yw[keep]) if keep.sum() >= MIN_LICKS else None
                    if A is not None:
                        rec["maps"][f"{st}/w{tg}"] = (W @ A.astype(np.float32))
                        rec["n"][f"{st}/w{tg}"] = int(keep.sum())
                okp = np.isin(pos, POS)
                for sp in POS:
                    y = (pos[okp] == sp).astype(float)
                    if MIN_SPOUT <= y.sum() < len(y):
                        A = _pattern(X[okp], y)
                        if A is not None:
                            rec["maps"][f"{st}/sp_{sp}"] = (W @ A.astype(np.float32))
                            rec["n"][f"{st}/sp_{sp}"] = int(y.sum())
            res[x] = rec
            print(f"{x} {rec['epoch']}: {len(rec['maps'])} maps", flush=True)
        (OUT / f"{a}.pkl").write_bytes(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
        del W
    return 0


if __name__ == "__main__":

    raise SystemExit(main())
