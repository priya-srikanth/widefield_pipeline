"""Can spout position be decoded from FACE movement (FaceRhythm factors) in the lick-free pre-cue (ENL) window?

Priya 10-25: "can we also analyze whether we can decode spout position from fr factors in the no-lick ENL time periods
(1s or 2s windows)?" The imaging ENL decode (page section 1-2) reads spout position from cortex before the cue on
lick-free trials. If the face already differs by spout there (posture / orienting toward a spout that has been in
place ~3 s), part of that cortical code could be movement. This is the FR side of that question.

Inputs: the frozen cohort factors (`master_pre_v1`, rank 12) projected on every session (`<set>/proj/*.npz`; absolute
expression = trials * trials_D, 25 Hz bins on the DAQ clock, cue -2 to +4 s; ENGAGED trials only, the behaviour
gate `orofacial_cohort.engagement_keep`). Pre = August only (the FR bundle has no June).

Trial selection: a trial is LICK-FREE for window W if neither an LP video lick (`lp_licks.lp_lick_times`, cam4; catches
licks that miss the spout) nor a DAQ spout contact (`analysis_kit.lick_samples`) falls in [cue - W, cue). Fixed
window ending at the cue (the FR grid starts at cue - 2 s, so the imaging decoder's slide-back is not possible).

Features: the 12 factors' mean in 0.5 s sub-bins of the window (W = 1 s -> 24 features, 2 s -> 48). CAVEAT: the FR
spectrogram is acausal (VQT sigma ~0.3-1 s), so the last bins before the cue can carry post-cue licking. Each W is
therefore also run with the features stopping 0.5 s before the cue ("end -0.5 s"; the trial selection is unchanged).

Decoder = the imaging decoders' (`locanmf_frozen_decoder._pipe`: StandardScaler + multinomial logistic C 0.5), cross-
validated with GroupKFold over position blocks (`block_ids`, the session's own block_size_max), positions with >= 5
lick-free trials only; chance = 1 / n positions decoded. Null per session: labels permuted against the fixed
cross-validated predictions (`nolick_analysis`-style, 2,000x). Change vs pre: `analysis_kit.boot_delta` (paired
animals -> sessions), per-animal table printed first (CLAUDE.md rule 8).

    python scripts/fr_enl_decode.py [--jobs N] [--only PS92_0806 ...]
Output: <dlc>/face_rhythm/cam4/readouts/master_pre_v1/enl_decode/ (fr_enl_decode_sessions.csv, _epochs.csv, figure).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = [(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)]          # (window s, features end this long before cue)
SUB_S = 0.5
MIN_PER_POS = 5
N_PERM = 2000


def arm_name(w, end):
    return f"ENL {w:.0f} s" + (f", features end -{end:.1f} s" if end else "")


def _session_dicts():
    from wfield_local import analysis_kit as ak
    return {s["label"]: s for s in ak.curated_sessions(order="sorted")}


def one(task):
    """All arms for one session (a label and its proj files; PS92_0812 comes as two video halves)."""
    label, files = task
    from sklearn.model_selection import GroupKFold

    from wfield_local import analysis_kit as ak
    from wfield_local import block_ids as bi
    from wfield_local import lp_licks
    from wfield_local.locanmf_frozen_decoder import _pipe
    X, cue, pos, ep, an = [], [], [], None, None
    for f in files:
        with np.load(f) as z:
            X.append(z["trials"] * z["trials_D"][..., None])
            cue.append(z["cue_daq_s"].astype(float))
            pos.append(z["position"].astype(str))
            t = z["t_ms"] / 1000.0
            ep, an = str(z["epoch"]), str(z["animal"])
    X, cue, pos = np.concatenate(X), np.concatenate(cue), np.concatenate(pos)
    o = np.argsort(cue)
    X, cue, pos = X[o], cue[o], pos[o]
    s = _session_dicts().get(label)
    if s is None:
        return []
    lp, _ = lp_licks.lp_lick_times(label)
    daq = np.sort(ak.lick_samples(s) / ak.daq_rate(s))
    # blocks from the FULL engaged trial sequence, then subset (a dropped trial does not merge two blocks)
    codes = pd.Categorical(pos).codes
    blocks = bi.block_ids(codes, bi.block_size_max_for(s))
    rows = []
    rng = np.random.default_rng(0)
    for w, end in ARMS:
        def n_in(tr, w=w):
            return np.searchsorted(tr, cue) - np.searchsorted(tr, cue - w)
        free = (n_in(lp) == 0) & (n_in(daq) == 0)
        edges = np.arange(-w, -end - 1e-9, SUB_S)
        F = np.concatenate([np.nanmean(X[:, (t >= a) & (t < a + SUB_S), :], axis=1) for a in edges], axis=1)
        ok = free & np.isfinite(F).all(1)
        keep_pos = [p for p, n in pd.Series(pos[ok]).value_counts().items() if n >= MIN_PER_POS]
        m = ok & np.isin(pos, keep_pos)
        row = dict(label=label, animal=an, epoch=ep, arm=arm_name(w, end), window_s=w, end_s=end,
                   n_engaged=len(pos), n_lickfree=int(free.sum()), n_used=int(m.sum()), n_pos=len(keep_pos),
                   chance=1 / len(keep_pos) if keep_pos else np.nan)
        g = blocks[m]
        if len(keep_pos) >= 2 and len(np.unique(g)) >= 3:
            y = pos[m]
            pred = np.empty(len(y), dtype=object)
            for tr, te in GroupKFold(min(5, len(np.unique(g)))).split(F[m], y, g):
                if len(np.unique(y[tr])) < 2:
                    pred[te] = y[tr][0]
                    continue
                pred[te] = _pipe().fit(F[m][tr], y[tr]).predict(F[m][te])
            acc = float(np.mean(pred == y))
            null = np.array([np.mean(pred == rng.permutation(y)) for _ in range(N_PERM)])
            rec = {p: float(np.mean(pred[y == p] == p)) for p in keep_pos}
            row.update(accuracy=acc, above_chance=(acc - row["chance"]) / (1 - row["chance"]),
                       p_perm=float((1 + np.sum(null >= acc)) / (1 + N_PERM)),
                       null_mean=float(null.mean()),
                       **{f"recall_{p}": rec.get(p, np.nan) for p in
                          ("far_L", "close_L", "close_center", "far_center", "close_R", "far_R")})
        rows.append(row)
    return rows


def epoch_table(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    out = []
    for arm, d in S.groupby("arm", sort=False):
        d = d.dropna(subset=["above_chance"])
        pre = {a: g.above_chance.to_numpy() for a, g in d[d.epoch == "pre"].groupby("animal")}
        rng = np.random.default_rng(0)
        for e in ("pre", "acute", "subacute", "chronic"):
            g = d[d.epoch == e]
            per = g.groupby("animal").agg(acc=("accuracy", "mean"), abv=("above_chance", "mean"),
                                         sig=("p_perm", lambda p: int((p < .05).sum())), n=("p_perm", "size"))
            r = dict(arm=arm, epoch=e, accuracy=per.acc.mean(), above_chance=per.abv.mean(),
                     sessions_p05=f"{int(per.sig.sum())}/{int(per.n.sum())}",
                     **{f"{a}_acc": per.acc.get(a, np.nan) for a in ("PS92", "PS93", "PS94", "PS95")})
            if e != "pre":
                post = {a: gg.above_chance.to_numpy() for a, gg in g.groupby("animal")}
                iv = ak.boot_delta(post, pre, rng)
                if iv is not None:
                    r.update(delta=iv.point, lo=iv.lo, hi=iv.hi, excludes_0=bool(iv.lo > 0 or iv.hi < 0),
                             below_pre=sum(np.mean(post[a]) < np.mean(pre[a]) for a in set(pre) & set(post)))
            out.append(r)
    return pd.DataFrame(out)


def figure(S: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.face_rhythm import common
    col = common.animal_colors()
    EP = ["pre", "acute", "subacute", "chronic"]
    arms = list(dict.fromkeys(S.arm))
    fig, axs = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 3.8), sharey=True)
    for ax, arm in zip(np.atleast_1d(axs), arms):
        d = S[(S.arm == arm)].dropna(subset=["accuracy"])
        for k, e in enumerate(EP):
            g = d[d.epoch == e]
            for a, gg in g.groupby("animal"):
                jit = {"PS92": -.18, "PS93": -.06, "PS94": .06, "PS95": .18}.get(a, 0)
                ax.scatter(np.full(len(gg), k + jit), gg.accuracy, s=9, color=col.get(a, "0.5"), alpha=.5, lw=0)
                ax.plot([k + jit - .05, k + jit + .05], [gg.accuracy.mean()] * 2, color=col.get(a, "0.5"), lw=2.2)
        means = [d[d.epoch == e].groupby("animal").accuracy.mean().mean() for e in EP]
        ax.plot(range(4), means, "-s", color="k", lw=1.8, ms=5, label="mean of animals")
        ch = d.chance.median()
        ax.axhline(ch, color="0.5", ls=":", lw=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(EP, fontsize=8)
        ax.set_title(arm, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel("spout position decoded from FR factors\n(cross-validated accuracy)")
    for a in ("PS92", "PS93", "PS94", "PS95"):
        np.atleast_1d(axs)[-1].plot([], [], color=col.get(a), lw=2.2, label=a)
    np.atleast_1d(axs)[-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("PRE-CUE (ENL, lick-free) window -- spout position from face movement (FaceRhythm master_pre_v1, "
                 "12 factors, 0.5 s bins). Dots = sessions, bars = animal means, black = mean of animals, "
                 "dotted = chance (1 / positions decoded)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.face_rhythm import common
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    T = common.sessions(common.default_set_dir())
    tasks = [(lab, tuple(g.file)) for lab, g in T.groupby("label", sort=True)]
    if a.only:
        tasks = [t for t in tasks if t[0] in a.only]
    out = a.out or common.readouts_dir() / "enl_decode"
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    res, fails = ak.fan_sessions(tasks, one, jobs=a.jobs, key=lambda t: t[0])
    for f in fails:
        print("!!", f)
    S = pd.DataFrame([r for _, rows in res for r in rows])
    S.to_csv(out / "fr_enl_decode_sessions.csv", index=False, float_format="%.4g")
    E = epoch_table(S)
    E.to_csv(out / "fr_enl_decode_epochs.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(E.round(3).to_string(index=False))
        print(S.groupby(["arm", "epoch"])[["n_engaged", "n_lickfree", "n_used", "n_pos"]].median())
    figure(S, out / "fr_enl_decode.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
