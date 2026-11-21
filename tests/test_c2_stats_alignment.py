"""Cohort-2 statistics aligned with cohort 1 (Priya 2026-11-19; DECISIONS.md "Cohort-2 statistics aligned").

Two opt-in options in `orofacial_stats`, both with the OLD behaviour as the default:
  * scalar BH family: `session_change_tests(fdr_scope="all")` / `apply_family_fdr` (one family across endpoints);
  * trajectory clusters: `cluster_permutation_p` (per-cluster permutation p, BH within measure by the caller).

Parity is pinned against FROZEN copies of the pre-change functions (CLAUDE.md rule 9: exact equality, no tolerance),
so a default output that moves fails here. The cluster numbers of stroke_orofacial_pipeline's
`tests/test_c2_aligned_stats.py::test_cluster_test_matches_cohort2` (its port of this function) are pinned too.
"""
import numpy as np
import pandas as pd
import pytest

from wfield_local import orofacial_stats as st


# ----------------------------------------------------------------------------------------------- frozen pre-change code
def _legacy_session_change_tests(P, metrics, epochs=("acute", "subacute", "chronic"), seed=0, n_boot=2000):
    """orofacial_stats.session_change_tests as of origin/main 36d943ef (before 2026-11-19)."""
    from scipy.stats import mannwhitneyu

    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        if m not in P:
            continue
        d = st._deltas(P, m)
        block = []
        for pos, gp in d.groupby("pos", sort=True):
            pre = gp[gp.epoch == "pre"]
            for e in epochs:
                post = gp[gp.epoch == e]
                x, y = post.delta.dropna().to_numpy(), pre.delta.dropna().to_numpy()
                r = {"metric": m, "pos": pos, "epoch": e, "n_sessions": len(x), "n_pre_sessions": len(y),
                     "n_animals": int(post.animal.nunique()), "median_change": float(np.median(x)) if len(x) else np.nan,
                     "mean_change": float(np.mean(x)) if len(x) else np.nan}
                if len(x) >= 2 and len(y) >= 2:
                    u, p = mannwhitneyu(x, y, alternative="two-sided")
                    r.update(mwu_U=float(u), p=float(p))
                else:
                    r.update(mwu_U=np.nan, p=np.nan)
                post_a = {a: g.delta.dropna().tolist() for a, g in post.groupby("animal", sort=True)}
                pre_a = {a: g.delta.dropna().tolist() for a, g in pre.groupby("animal", sort=True)}
                pre_a = {a: v for a, v in pre_a.items() if v and post_a.get(a)}
                post_a = {a: v for a, v in post_a.items() if v and a in pre_a}
                iv = ak.boot_delta(post_a, pre_a, rng, n_boot=n_boot) if len(post_a) >= 2 else None
                r.update(boot_change=iv.point if iv else np.nan, boot_lo=iv.lo if iv else np.nan,
                         boot_hi=iv.hi if iv else np.nan)
                block.append(r)
        q = st.bh_fdr([b["p"] for b in block])
        for b, qq in zip(block, q):
            b["q_fdr"] = float(qq)
            b["headline_sig"] = bool(np.isfinite(qq) and qq < st.ALPHA)
            b["boot_sig"] = bool(np.isfinite(b["boot_lo"]) and (b["boot_lo"] > 0 or b["boot_hi"] < 0))
            b["agree"] = ("both" if b["headline_sig"] and b["boot_sig"] else "MWU only" if b["headline_sig"]
                          else "bootstrap only" if b["boot_sig"] else "neither")
        rows += block
    return pd.DataFrame(rows)


def _legacy_cluster_permutation(curves, is_post, animal, n_perm=1000, seed=0, alpha=st.ALPHA):
    """orofacial_stats.cluster_permutation as of origin/main 36d943ef (before 2026-11-19)."""
    from scipy.stats import t as tdist
    curves = np.asarray(curves, float)
    is_post = np.asarray(is_post, bool)
    mask = np.zeros(curves.shape[1], bool)
    if is_post.sum() < 2 or (~is_post).sum() < 2:
        return mask
    thr = float(tdist.ppf(1 - alpha / 2, max(int(is_post.sum() + (~is_post).sum() - 2), 1)))
    obs = st._clusters(st._welch_t(curves[is_post], curves[~is_post]), thr)
    if not obs:
        return mask
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(animal == a) for a in np.unique(animal)]
    null = np.zeros(n_perm)
    lab = is_post.copy()
    for k in range(n_perm):
        for g in groups:
            lab[g] = rng.permutation(is_post[g])
        cl = st._clusters(st._welch_t(curves[lab], curves[~lab]), thr)
        null[k] = max((c[2] for c in cl), default=0.0)
    for s, e, mass in obs:
        if (np.sum(null >= mass) + 1) / (n_perm + 1) < alpha:
            mask[s:e] = True
    return mask


# --------------------------------------------------------------------------------------------------------- toy data
def _panel(seed=0, n_animals=4):
    """Two metrics, three positions, pre + three epochs; effects of different sizes so BH scopes differ."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_animals):
        off = rng.normal(0, 3)
        for ep, n, shift in (("pre", 6, 0.0), ("acute", 3, 1.6), ("subacute", 3, 0.7), ("chronic", 4, 0.2)):
            for k in range(n):
                for pos, w in (("far_contra", 1.0), ("far_mid", 0.5), ("close_ipsi", 0.0)):
                    rows.append({"animal": f"A{a}", "date": f"{ep}{k}", "epoch": ep, "pos": pos,
                                 "m1": off + w * shift + rng.normal(0, 1),
                                 "m2": 2 * off + 0.4 * w * shift + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def _toy_sessions(seed=1, bins=40):
    """stroke_orofacial_pipeline tests/test_c2_aligned_stats.py::_toy_sessions, verbatim data."""
    rng = np.random.default_rng(seed)
    recs, meta = [], []
    for a, (npre, npost) in {"A": (6, 5), "B": (4, 7), "C": (5, 3)}.items():
        for _ in range(npre):
            recs.append(rng.normal(0, 1, bins))
            meta.append((a, "pre"))
        for _ in range(npost):
            v = rng.normal(0, 1, bins)
            v[10:22] += 1.8
            v[30:34] -= 0.9
            recs.append(v)
            meta.append((a, "acute"))
    return np.vstack(recs), np.array([m[1] == "acute" for m in meta]), np.array([m[0] for m in meta])


# ------------------------------------------------------------------------------------------------------ scalar family
def test_default_session_change_tests_is_unchanged():
    P = _panel()
    new = st.session_change_tests(P, ["m1", "m2"], n_boot=300)
    old = _legacy_session_change_tests(P, ["m1", "m2"], n_boot=300)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    assert "q_fdr_metric" not in new and "fdr_family" not in new


def test_fdr_scope_all_is_one_family_and_keeps_the_draws():
    P = _panel()
    old = st.session_change_tests(P, ["m1", "m2"], n_boot=300)
    fam = st.session_change_tests(P, ["m1", "m2"], n_boot=300, fdr_scope="all")
    # same p and bootstrap, per-metric q kept, family q = BH over every row
    for c in ("p", "boot_change", "boot_lo", "boot_hi", "mean_change"):
        assert np.array_equal(fam[c].to_numpy(), old[c].to_numpy(), equal_nan=True)
    assert np.array_equal(fam.q_fdr_metric.to_numpy(), old.q_fdr.to_numpy(), equal_nan=True)
    assert np.allclose(fam.q_fdr, st.bh_fdr(old.p), equal_nan=True)
    assert (fam.fdr_family == "all").all()
    # stars follow the same rule
    exp = (fam.q_fdr < st.ALPHA) & ((fam.boot_lo > 0) | (fam.boot_hi < 0))
    assert ((fam.agree == "both") == exp).all()


def test_apply_family_fdr_pools_separate_calls_and_per_metric_reproduces_default():
    P = _panel()
    a = st.session_change_tests(P, ["m1"], n_boot=300)
    b = st.session_change_tests(P, ["m2"], n_boot=300, seed=1)
    fam = st.apply_family_fdr(pd.concat([a, b], ignore_index=True))
    assert np.allclose(fam.q_fdr, st.bh_fdr(np.r_[a.p, b.p]))
    per = st.apply_family_fdr(pd.concat([a, b], ignore_index=True), family_cols=["metric"])
    both = pd.concat([a, b], ignore_index=True)
    assert np.array_equal(per.q_fdr.to_numpy(), both.q_fdr.to_numpy(), equal_nan=True)
    assert (per.agree == both.agree).all() and (per.headline_sig == both.headline_sig).all()


def test_one_metric_family_all_equals_default_stars():
    P = _panel()
    old = st.session_change_tests(P, ["m1"], n_boot=300)
    fam = st.session_change_tests(P, ["m1"], n_boot=300, fdr_scope="all")
    assert np.array_equal(fam.q_fdr.to_numpy(), old.q_fdr.to_numpy(), equal_nan=True)
    assert (fam.agree == old.agree).all()


def test_deltas_given_matches_the_recentred_default():
    P = _panel()
    d = pd.concat([st._deltas(P, m).rename(columns={"delta": m}).set_index(["animal", "date", "epoch", "pos"])
                   for m in ("m1", "m2")], axis=1).reset_index()
    a = st.session_change_tests(P, ["m1", "m2"], n_boot=300)
    b = st.session_change_tests(d, ["m1", "m2"], n_boot=300, deltas_given=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_bad_fdr_scope_refused():
    with pytest.raises(ValueError, match="fdr_scope"):
        st.session_change_tests(_panel(), ["m1"], fdr_scope="metrics")


# ---------------------------------------------------------------------------------------------- trajectory clusters
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cluster_permutation_is_unchanged_and_p_reproduces_its_mask(seed):
    rng = np.random.default_rng(seed)
    n_pre, n_post, bins = 12, 10, 50
    curves = rng.normal(0, 1, (n_pre + n_post, bins))
    curves[n_pre:, 12:20] += 1.2 + 0.3 * seed
    curves[n_pre:, 35:38] -= 0.8
    is_post = np.r_[np.zeros(n_pre, bool), np.ones(n_post, bool)]
    animal = np.r_[np.repeat(["A", "B", "C"], 4), np.repeat(["A", "B"], 3), np.repeat(["C"], 4)]
    old = _legacy_cluster_permutation(curves, is_post, animal, n_perm=200)
    assert np.array_equal(st.cluster_permutation(curves, is_post, animal, n_perm=200), old)
    cl = st.cluster_permutation_p(curves, is_post, animal, n_perm=200)
    m = np.zeros(bins, bool)
    for c in cl:
        if c["p_raw"] < st.ALPHA:
            m[c["start"]:c["stop"]] = True
    assert np.array_equal(m, old)
    # with q = p_raw the mask helper gives the old mask too
    assert np.array_equal(st.cluster_mask_from_q(bins, cl, [c["p_raw"] for c in cl]), old)


def test_cluster_p_matches_stroke_orofacial_reference():
    """Same numbers as stroke_orofacial_pipeline test_cluster_test_matches_cohort2 (rng = default_rng(0))."""
    curves, is_post, animal = _toy_sessions()
    cl = st.cluster_permutation_p(curves, is_post, animal, n_perm=500, seed=0)
    assert cl[0]["t_threshold"] == pytest.approx(2.048, abs=1e-3)       # t(.975, df=28)
    got = [(c["start"], c["end_bin"], round(c["p"], 4)) for c in cl]
    assert got == [(10, 21, 0.002), (23, 23, 0.7585), (32, 32, 0.3253), (35, 35, 0.1417)]


def test_no_cluster_and_too_few_sessions_give_no_rows():
    rng = np.random.default_rng(5)
    is_post = np.r_[np.zeros(6, bool), np.ones(6, bool)]
    animal = np.repeat(["A", "B"], 6)
    flat = np.zeros((12, 20)) + rng.normal(0, 1e-9, (12, 20))
    assert st.cluster_permutation_p(flat * 0, is_post, animal, n_perm=50) == []
    assert st.cluster_permutation_p(rng.normal(0, 1, (3, 20)), np.array([True, False, False]),
                                    np.array(["A", "A", "A"]), n_perm=50) == []


def test_bh_within_measure_can_drop_an_uncorrected_cluster():
    """A weak cluster that passes p < .05 alone can fail BH once the measure's other cells are in the family."""
    p = np.array([0.03, 0.04, 0.5, 0.7])
    q = st.bh_fdr(p)
    assert (p[:2] < 0.05).all() and not (q[:2] < 0.05).any()
