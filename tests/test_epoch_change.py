"""`analysis_kit.epoch_change` reproduces EXACTLY the change-vs-pre copies it replaced in the state-space arm (11-13
cleanup; rule 9: pin a draw, do not eyeball it). The frozen sources below are the pre-consolidation code, verbatim
apart from being lifted out of their closures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local import analysis_kit as kit

POST = ["acute", "subacute", "chronic"]


def _table(seed=3, with_nan=True):
    rng = np.random.default_rng(seed)
    rows = []
    n = {"pre": (6, 1, 8, 3), "acute": (4, 2, 0, 1), "subacute": (2, 3, 5, 7), "chronic": (9, 4, 6, 2)}
    for i, an in enumerate(["PS92", "PS93", "PS94", "PS95"]):
        for ep, k in n.items():
            for s in range(k[i]):
                v = rng.normal({"pre": 0, "acute": -0.5, "subacute": 0.2, "chronic": 0.1}[ep], 0.3)
                rows.append(dict(animal=an, epoch=ep, label=f"{an}_{ep}{s}", v=np.nan if with_nan and s == 1 else v))
    rows.append(dict(animal="PS99", epoch="acute", label="PS99_a", v=1.0))      # an animal with no pre
    return pd.DataFrame(rows)


# --- frozen: state_space_page_figs.change (da9f52ae)
def _page_figs_change(df, col, rng):
    pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in df.groupby("animal")}
    out = {}
    for ep in POST:
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in df.groupby("animal")}
        post = {a: v for a, v in post.items() if v and pre.get(a)}
        iv = kit.boot_delta(post, pre, rng) if post else None
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post)
        out[ep] = (iv.point, iv.lo, iv.hi, up, len(post)) if iv is not None else (np.nan,) * 3 + (0, 0)
    return out


# --- frozen: state_space_recovery_trials.summarize inner loop (da9f52ae): no pre filter on post
def _recovery_change(df, m, rng):
    pre = {a: g[g.epoch == "pre"][m].dropna().tolist() for a, g in df.groupby("animal")}
    out = {}
    for ep in POST:
        post = {a: g[g.epoch == ep][m].dropna().tolist() for a, g in df.groupby("animal")}
        post = {a: v for a, v in post.items() if v}
        iv = kit.boot_delta(post, pre, rng)
        sign = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post if pre.get(a))
        out[ep] = (iv.point, iv.lo, iv.hi, sign)
    return out


def _ec(df, col, rng, epochs=POST):
    return kit.epoch_change(df, col, rng, epochs)


def test_matches_page_figs_change():
    df = _table()
    a = _page_figs_change(df, "v", np.random.default_rng(0))
    b = _ec(df, "v", np.random.default_rng(0))
    for ep in POST:
        e = b[ep]
        assert (e.point, e.lo, e.hi, e.n_up, e.n_animals) == a[ep] or (np.isnan(a[ep][0]) and np.isnan(e.point))


def test_matches_per_epoch_calls_with_shared_rng():
    """The shift_* closures call change(col, ep) once per epoch on one rng: same sequence as epochs=[ep]."""
    df = _table()
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)

    def change(col, ep):            # frozen: state_space_shift_behavior.main.change (da9f52ae), Sx -> df, rng -> r1
        pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in df.groupby("animal")}
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in df.groupby("animal")}
        post = {a: v for a, v in post.items() if v and pre.get(a)}
        iv = kit.boot_delta(post, pre, r1) if post else None
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post)
        return (iv.point, iv.lo, iv.hi, up, len(post)) if iv is not None else (np.nan,) * 3 + (0, 0)

    for col in ("v", "v"):
        for ep in POST:
            e = kit.epoch_change(df, col, r2, [ep])[ep]
            assert (e.point, e.lo, e.hi, e.n_up, e.n_animals) == change(col, ep)


def test_matches_recovery_trials_summarize():
    """PS99 (no pre) is dropped up front here; boot_delta drops it anyway, so the interval is identical."""
    df = _table()
    a = _recovery_change(df, "v", np.random.default_rng(0))
    b = _ec(df, "v", np.random.default_rng(0))
    for ep in POST:
        assert (b[ep].point, b[ep].lo, b[ep].hi, b[ep].n_up) == a[ep]


def test_per_animal_table_and_empty_epoch():
    df = _table(with_nan=False)
    r = kit.epoch_change(df, "v", np.random.default_rng(0), ["acute", "none"])
    assert set(r["acute"].per_animal) == {"PS92", "PS93", "PS95"}        # PS94 has no acute; PS99 has no pre
    pre = df[df.epoch == "pre"].groupby("animal").v.mean()
    post = df[df.epoch == "acute"].groupby("animal").v.mean()
    assert np.isclose(r["acute"].per_animal["PS92"], post["PS92"] - pre["PS92"], rtol=0, atol=1e-15)  # pandas vs np sum order
    assert np.isnan(r["none"].point) and r["none"].n_animals == 0


# ------------------------------------------------------------------ within-animal shuffles (11-13 cleanup)
def _post_table(seed=7):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"animal": np.repeat(["PS95", "PS92", "PS94", "PS93"], [5, 7, 4, 6])})
    d["x"] = rng.normal(size=len(d))
    d["y"] = 0.4 * d.x + rng.normal(size=len(d))
    return d.sample(frac=1, random_state=1).reset_index(drop=True)          # animals interleaved


def test_permute_within_is_the_groupby_transform_draw():
    d = _post_table()
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(5):
        a = d.groupby("animal").y.transform(lambda v: r1.permutation(v.to_numpy())).to_numpy()
        assert (kit.permute_within(d.y, d.animal, r2) == a).all()


def test_within_shuffle_p_matches_the_shift_precue_loop():
    """frozen: state_space_shift_precue coupling slope + shuffle p (da9f52ae)."""
    d = _post_table()
    r1, r2 = np.random.default_rng(0), np.random.default_rng(0)
    b = float((d.x * d.y).sum() / (d.x ** 2).sum())
    null = []
    for _ in range(300):
        sh = d.groupby("animal").y.transform(lambda v: r1.permutation(v.to_numpy()))
        null.append(float((d.x * sh).sum() / (d.x ** 2).sum()))
    p_old = float((np.abs(null) >= abs(b)).mean())
    x = d.x.to_numpy()
    obs, p_new = kit.within_shuffle_p(lambda y: float((x * y).sum() / (x ** 2).sum()), d.y.to_numpy(),
                                      d.animal.to_numpy(), r2, 300)
    assert p_new == p_old and np.isclose(obs, b, rtol=0, atol=1e-15)


def test_within_shuffle_p_matches_the_detour_loop():
    """frozen: state_space_detour_vs_balance.partial shuffle (np.unique order) (da9f52ae)."""
    d = _post_table()
    rp, rt = d.x, d.y.to_numpy()
    r1, r2 = np.random.default_rng(0), np.random.default_rng(0)
    r0 = float(np.corrcoef(rp, rt)[0, 1])
    null = np.empty(200)
    an = d.animal.to_numpy()
    for i in range(200):
        sh = rp.copy()
        for a in np.unique(an):
            m = an == a
            sh[m] = r1.permutation(sh[m].to_numpy())
        null[i] = np.corrcoef(sh, rt)[0, 1]
    p_old = float((np.abs(null) >= abs(r0)).mean())
    assert kit.within_shuffle_p(lambda a: np.corrcoef(a, rt)[0, 1], rp.to_numpy(), an, r2, 200)[1] == p_old
