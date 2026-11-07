"""PER-SESSION shape tests of the FaceRhythm factors -- the MAIN tests from 11-07 (Priya: "make the per-session test the
main test for shape, for every factor"; "everything should be delta per-animal pre wherever possible").

Every session's own fit (`fr_factors.py fitsession`, 8 starts, ``<set>/own8/``) is matched to the master with the current
rule (`face_rhythm_epoch_shape.MATCH_RULES`, default v2). Every readout is a per-session value minus the animal's OWN pre
mean (delta per animal), the statistic is the mean over animals of (epoch mean - pre mean), animals weighted equally, and
the null shuffles each animal's pre / epoch session labels WITHIN the animal (counts kept; no refit). p = (1 + #null
beyond obs) / (1 + n); Holm per epoch x measure over two families, the lick pair (lick_right, lick_left; pre-specified)
and the other units (exploratory), as the pooled-fit null.

  SCALARS (recovered sessions only)
    asym            loading asymmetry (contra - ipsi) / sum of the matched column; two-sided
    angle_spatial   angle (deg) between the session's matched spatial vector and its animal's PRE-mean vector
                    (leave-one-out for pre sessions, recomputed under every shuffle); one-sided (larger)
    angle_spectral  the same on the frequency profile; one-sided
    side_spec       lick pair only: mean r(loading, own-side spout templates) - mean r(other side), templates = the
                    animal's OWN pre per-spout speed minus its spout mean; two-sided
  LOSS              recovered (0 / 1) per session; delta = recovery rate epoch - pre; one-sided (lower)
  MAPS              loading fraction per mesh point, delta vs own pre; ``--scale``: each animal's delta divided per
                    point by its own pre session SD (a noisy animal cannot dominate); cluster mass + max-statistic
                    (`face_rhythm.stats.session_shuffle_test`)
  SENSITIVITY       (a) maps without AMBIGUOUS sessions (the unit's best spatial cosine beats the runner-up by < .05);
                    (b) lick_right: does a session's map change track its change in the share of RIGHT licks (both
                    delta vs own pre, then centred within animal)?
  FRAGILITY         per unit: the pooled-fit refit null's 95th-percentile acute spatial angle / the pooled pre-only fit
                    noise (preA - preB): large = the pooled test is inconclusive for that unit (lick_left, 11-07)
  DIRECTION         share of left / centre / right executed licks per session, delta vs own pre, same test

    python scripts/face_rhythm_session_shape_tests.py [--set master_pre_v1] [--match-rule v2] [--perm 2000]
-> ``<readouts>/<set>/session_shape_<rule>/``: session_scalars.png, session_loss.png, session_maps_scaled_lick.png,
session_maps_unambiguous_lick.png, lick_direction.png, data/*.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_rhythm_epoch_shape as es  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402

LICK = ("lick_right", "lick_left")
OWN_SIDE = {"lick_right": ("close_R", "far_R"), "lick_left": ("close_L", "far_L")}
AMBIGUOUS = 0.05
SCALARS = ("asym", "angle_spatial", "angle_spectral", "side_spec")
ONE_SIDED = ("angle_spatial", "angle_spectral")


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def load_sessions(fdir: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """One row per session x unit: recovered, margin, vec (unit spatial vector), load (fraction per point), spec
    (unit frequency profile), asym, side_spec (lick pair, own templates)."""
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_readouts as frr
    M, A_ref, B_ref, names = es.reference(fdir)
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    S = frc.sessions(fdir, "proj")
    tmpl = {}                                                   # animal -> (6, P) own pre spout-specific speed
    for an, g in S[S.epoch == "pre"].groupby("animal"):
        acc = []
        for t in g.itertuples():
            with np.load(t.file, allow_pickle=True) as z:
                have = list(z["power_positions"].astype(str))
                sp = np.full((6, z["speed_response"].shape[1]), np.nan)
                for i, p in enumerate(frc.POSITIONS):
                    if p in have:
                        sp[i] = z["speed_response"][have.index(p)]
                acc.append(sp)
        m = np.nanmean(acc, 0)
        tmpl[an] = m - np.nanmean(m, 0, keepdims=True)
    import face_rhythm_cohort_null as N
    rows = []
    for t in S.itertuples():
        f = fdir / "own8" / f"{t.key}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as z:
            A, B = z["A_own"].astype(float), z["B_own"].astype(float)
        m, cols, ok = es.pick(A_ref, B_ref, A, B)
        vec, mag = fm.spatial_view(A, "vector"), fm.spatial_view(A, "magnitude")
        asym = fm.loading_asymmetry(A, home, mid)
        for k, nm in enumerate(names):
            c = int(cols[k])
            srt = np.sort(m["cs_all"][k])[::-1]
            ld = mag[:, c] / mag[:, c].sum()
            rows.append(dict(key=t.key, animal=t.animal, date=str(t.date), epoch=t.epoch, unit=nm, recovered=bool(ok[k]),
                             margin=float(srt[0] - srt[1]), vec=unit(vec[:, c]), load=ld, spec=unit(np.abs(B[:, c])),
                             asym=float(asym[c]),
                             side_spec=N.side_spec(ld, tmpl[t.animal], OWN_SIDE[nm]) if nm in OWN_SIDE else np.nan))
    return pd.DataFrame(rows), names, home


def _perm_p(obs, null, side):
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return np.nan
    if side == "greater":
        return (1 + (null >= obs).sum()) / (1 + len(null))
    if side == "less":
        return (1 + (null <= obs).sum()) / (1 + len(null))
    return (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + len(null))


def scalar_stat(groups, kind):
    """groups: per animal (values (n,) or vectors (n, D), is_pre (n,) bool). Mean over animals of (epoch - pre)."""
    out = []
    for V, pre in groups:
        if pre.sum() < 1 or (~pre).sum() < 1:
            continue
        if kind == "value":
            out.append(np.nanmean(V[~pre]) - np.nanmean(V[pre]))
        else:                                                   # angle to the pre mean, leave-one-out for pre
            P = V[pre]
            tot = P.sum(0)
            pm = unit(tot)
            a_post = es.deg(V[~pre] @ pm)
            a_pre = [es.deg(float(P[i] @ unit(tot - P[i]))) for i in range(len(P))] if len(P) > 1 else [np.nan]
            out.append(np.nanmean(a_post) - np.nanmean(a_pre))
    return float(np.mean(out)) if out else np.nan


def shuffle_test(groups, kind, n_perm, rng, side):
    obs = scalar_stat(groups, kind)
    null = np.array([scalar_stat([(V, rng.permutation(pre)) for V, pre in groups], kind) for _ in range(n_perm)])
    return obs, _perm_p(obs, null, side), len(groups)


def scalars(T: pd.DataFrame, names: list[str], n_perm: int, rng, drift: bool = False) -> pd.DataFrame:
    """``drift``: the same tests on each animal's EARLY vs LATE pre sessions (date order), epoch "late_pre"."""
    from wfield_local import analysis_kit as ak
    rows = []
    for ep in (["late_pre"] if drift else frc.POST_EPOCHS):
        for m in SCALARS:
            for nm in names:
                if m == "side_spec" and nm not in OWN_SIDE:
                    continue
                d = T[(T.unit == nm) & T.recovered & T.epoch.isin(["pre", ep])]
                groups = []
                for _an, g in d.groupby("animal"):
                    if drift:                                   # early half of the pre sessions = the reference
                        g = g[g.epoch == "pre"].sort_values("date")
                        if len(g) < 2:
                            continue
                        pre = np.arange(len(g)) < len(g) // 2
                    else:
                        pre = (g.epoch == "pre").to_numpy()
                    if m in ONE_SIDED:
                        V = np.vstack(g["vec" if m == "angle_spatial" else "spec"].to_numpy())
                    else:
                        V = g[m].to_numpy(float)
                    groups.append((V, pre))
                obs, p, n_an = shuffle_test(groups, "angle" if m in ONE_SIDED else "value", n_perm, rng,
                                            "greater" if m in ONE_SIDED else "two")
                rows.append(dict(epoch=ep, measure=m, unit=nm, family="lick pair" if nm in LICK else "other",
                                 delta=obs, p=p, n_animals=n_an,
                                 n_sessions=int((d.epoch == "pre").sum() if drift else (d.epoch == ep).sum())))
    R = pd.DataFrame(rows)
    # Holm within a FAMILY across every epoch x measure (Priya 11-07): lick pair = 24 tests (pre-specified), other
    # units = the rest (exploratory); the narrower per-epoch x measure version kept as p_holm_epoch for reference
    for col, keys in (("p_holm", ["family"]), ("p_holm_epoch", ["epoch", "measure", "family"])):
        R[col] = np.nan
        for _, idx in R.groupby(keys).groups.items():
            ok = [i for i in idx if np.isfinite(R.at[i, "p"])]
            if ok:
                R.loc[ok, col] = ak.holm(R.loc[ok, "p"].to_numpy())
    return R


def loss(T: pd.DataFrame, names: list[str], n_perm: int, rng) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rows = []
    for ep in frc.POST_EPOCHS:
        for nm in names:
            d = T[(T.unit == nm) & T.epoch.isin(["pre", ep])]
            groups = [(g.recovered.to_numpy(float), (g.epoch == "pre").to_numpy()) for _, g in d.groupby("animal")]
            obs, p, n_an = shuffle_test(groups, "value", n_perm, rng, "less")
            rec = d.groupby("epoch").recovered.agg(["sum", "size"])
            rows.append(dict(epoch=ep, unit=nm, family="lick pair" if nm in LICK else "other", delta_recovery=obs,
                             p_loss=p, pre=f"{int(rec.loc['pre', 'sum'])}/{int(rec.loc['pre', 'size'])}",
                             post=f"{int(rec.loc[ep, 'sum'])}/{int(rec.loc[ep, 'size'])}" if ep in rec.index else "-",
                             n_animals=n_an))
    R = pd.DataFrame(rows)
    R["p_holm"] = ak.holm(R.p_loss.to_numpy())               # across every unit x epoch (Priya 11-07)
    R["p_holm_epoch"] = np.nan
    for _, idx in R.groupby(["epoch"]).groups.items():
        R.loc[idx, "p_holm_epoch"] = ak.holm(R.loc[idx, "p_loss"].to_numpy())
    return R


def map_groups(T, nm, ep, scale=False, drop_ambiguous=False):
    per = []
    d = T[(T.unit == nm) & T.recovered & T.epoch.isin(["pre", ep])]
    if drop_ambiguous:
        d = d[d.margin >= AMBIGUOUS]
    for _an, g in d.groupby("animal"):
        g = g.sort_values("epoch", key=lambda s: s != "pre")    # pre first
        X = np.vstack(g.load.to_numpy())
        n = int((g.epoch == "pre").sum())
        if n < 2 or n == len(X):
            continue
        if scale:
            sd = X[:n].std(0, ddof=1)
            sd = np.maximum(sd, np.nanpercentile(sd, 10))        # floor: points with ~0 pre variance
            X = X / sd[None, :]
        per.append((X, n))
    return per


def maps(T, units, home, n_perm, rng, **kw):
    from wfield_local.face_rhythm import stats as frs
    adj = frs.mesh_adjacency(home)
    res, rows = {}, []
    for nm in units:
        for ep in frc.POST_EPOCHS:
            r = frs.session_shuffle_test(map_groups(T, nm, ep, **kw), adj, n_perm, rng)
            if r is None:
                continue
            r["n_pre"] = r["n_ref"]
            res[(nm, ep)] = r
            rows.append(dict(unit=nm, epoch=ep, clusters=int((r["p_cluster"] < .05).sum()),
                             cluster_points=int(r["sig_cluster"].sum()), maxstat_points=int(r["sig_point"].sum()),
                             n_animals=r["n_animals"], n_sessions=r["n_sessions"], n_pre=r["n_ref"]))
    return res, pd.DataFrame(rows)


def right_share_check(T: pd.DataFrame, D: pd.DataFrame, res_maps: dict) -> pd.DataFrame:
    """lick_right, per post epoch: per session, the projection of its map delta (vs own pre) onto the epoch's group
    pattern, against its delta in the share of RIGHT licks (vs own pre); both centred within animal; Spearman."""
    from scipy.stats import spearmanr
    rows = []
    d = T[(T.unit == "lick_right") & T.recovered]
    for ep in frc.POST_EPOCHS:
        if ("lick_right", ep) not in res_maps:
            continue
        pat = unit(res_maps[("lick_right", ep)]["mean"])
        xs, ys = [], []
        for an, g in d[d.epoch.isin(["pre", ep])].groupby("animal"):
            pre = g[g.epoch == "pre"]
            post = g[g.epoch == ep]
            if pre.empty or post.empty:
                continue
            base = np.vstack(pre.load.to_numpy()).mean(0)
            sc = np.array([(v - base) @ pat for v in post.load])
            fr = D.set_index(["animal", "date"]).frac_right
            pre_fr = np.nanmean([fr.get((an, x), np.nan) for x in pre.date])
            dfr = np.array([fr.get((an, x), np.nan) - pre_fr for x in post.date])
            ok = np.isfinite(dfr)
            if ok.sum() >= 2:
                xs += list(dfr[ok] - dfr[ok].mean())
                ys += list(sc[ok] - sc[ok].mean())
        if len(xs) >= 5:
            rho, p = spearmanr(xs, ys)
            rows.append(dict(epoch=ep, n_sessions=len(xs), spearman_rho=rho, p=p))
    return pd.DataFrame(rows)


def fragility(set_name: str, names: list[str]) -> pd.DataFrame:
    from wfield_local import figure_layout as fl
    R = frc.readouts_dir(set_name) / f"epoch_shape_cohort_{es.RULE}" if es.RULE != "v1" else frc.readouts_dir(set_name) / "epoch_shape_cohort"
    nul = pd.read_csv(fl.sidecar_for(R, "cohort_null", ".csv", mkdir=False))
    dft = pd.read_csv(fl.sidecar_for(R, "drift_floor", ".csv", mkdir=False))
    rows = []
    for nm in names:
        a = nul[(nul.factor == nm) & (nul.epoch == "acute") & (nul.measure == "angle_spatial")]
        c = dft[(dft.tag == "COHORT") & (dft.name == nm) & (dft.epoch == "acute")]
        if a.empty or c.empty:
            continue
        hi, noise, drift = float(a.null_hi.iloc[0]), float(c.noise_spatial.iloc[0]), float(c.drift_spatial.iloc[0])
        rows.append(dict(unit=nm, null_95=hi, prefit_noise=noise, prefit_drift=drift,
                         ratio=hi / max(noise, drift, 1e-6)))
    F = pd.DataFrame(rows)
    if len(F):
        ref = F.ratio.median()
        F["pooled_test"] = np.where(F.ratio > 2 * ref, "inconclusive (fragile)", "usable")
    return F


def direction(D: pd.DataFrame, n_perm: int, rng) -> pd.DataFrame:
    rows = []
    for ep in frc.POST_EPOCHS:
        for col in ("frac_left", "frac_centre", "frac_right"):
            d = D[D.epoch.isin(["pre", ep])]
            groups = [(g[col].to_numpy(float), (g.epoch == "pre").to_numpy()) for _, g in d.groupby("animal")]
            obs, p, n_an = shuffle_test(groups, "value", n_perm, rng, "two")
            per = {an: float(g[g.epoch == ep][col].mean() - g[g.epoch == "pre"][col].mean())
                   for an, g in d.groupby("animal")}
            rows.append(dict(epoch=ep, share=col, delta=obs, p=p, **per))
    return pd.DataFrame(rows)


#: display order (Priya 11-07): the lick pair, then the other licking units, then the rest
ORDER = ("lick_right", "lick_left", "lick_ongoing_7Hz", "lick_fast_close", "lick_slow_late", "lick_far_nose",
         "prelick_nose", "nose_steps", "slow_jaw", "whisk_run")
YLAB = {"asym": "loading asymmetry\n(contra - ipsi), delta", "angle_spatial": "spatial angle to own\npre mean, delta (deg)",
        "angle_spectral": "spectral angle to own\npre mean, delta (deg)", "side_spec": "side specificity, delta",
        "loss": "recovered (0 / 1),\ndelta vs own pre rate"}


def session_values(T: pd.DataFrame, names: list[str], mode: str = "delta") -> pd.DataFrame:
    """One row per session x unit x measure: the session's value minus its animal's own pre mean (angles: the angle to
    the animal's pre-mean vector minus the animal's mean leave-one-out pre angle). Loss: recovered (0 / 1) minus the
    animal's pre recovery rate (all sessions). PRE sessions (Priya 11-07: show the pre-stroke variability) are scored
    LEAVE-ONE-OUT -- against the mean of the animal's OTHER pre sessions -- so their spread is not shrunk by
    including themselves. ``mode="raw"`` (Priya 11-07: raw and normalised as separate figures): the session's own value
    (asymmetry, side specificity) and the plain angle to the animal's pre-mean vector (pre sessions: leave-one-out
    angle), no baseline subtracted; no loss rows."""
    rows = []
    if mode == "raw":
        for nm in names:
            for an, g in T[(T.unit == nm) & T.recovered].groupby("animal"):
                pre = g[g.epoch == "pre"]
                if len(pre) < 2:
                    continue
                for m in ("asym", "side_spec"):
                    if m == "side_spec" and nm not in OWN_SIDE:
                        continue
                    for r in g.itertuples():
                        rows.append(dict(unit=nm, animal=an, epoch=r.epoch, measure=m, value=getattr(r, m)))
                for m, col in (("angle_spatial", "vec"), ("angle_spectral", "spec")):
                    P = np.vstack(pre[col].to_numpy())
                    tot = P.sum(0)
                    pm = unit(tot)
                    for i in range(len(P)):
                        rows.append(dict(unit=nm, animal=an, epoch="pre", measure=m,
                                         value=float(es.deg(float(P[i] @ unit(tot - P[i]))))))
                    for r in g[g.epoch != "pre"].itertuples():
                        rows.append(dict(unit=nm, animal=an, epoch=r.epoch, measure=m,
                                         value=float(es.deg(float(getattr(r, col) @ pm)))))
        return pd.DataFrame(rows)
    for nm in names:
        for an, g in T[T.unit == nm].groupby("animal"):
            pre_all = g[g.epoch == "pre"]
            post_all = g[g.epoch != "pre"]
            rate = pre_all.recovered.mean()
            for r in post_all.itertuples():
                rows.append(dict(unit=nm, animal=an, epoch=r.epoch, measure="loss", value=float(r.recovered) - rate))
            if len(pre_all) > 1:
                rec = pre_all.recovered.to_numpy(float)
                for i in range(len(rec)):
                    rows.append(dict(unit=nm, animal=an, epoch="pre", measure="loss",
                                     value=rec[i] - np.delete(rec, i).mean()))
            pre, post = pre_all[pre_all.recovered], post_all[post_all.recovered]
            if pre.empty or post.empty:
                continue
            for m in ("asym", "side_spec"):
                if m == "side_spec" and nm not in OWN_SIDE:
                    continue
                base = pre[m].mean()
                for r in post.itertuples():
                    rows.append(dict(unit=nm, animal=an, epoch=r.epoch, measure=m, value=getattr(r, m) - base))
                pv = pre[m].to_numpy(float)
                if len(pv) > 1:
                    for i in range(len(pv)):
                        rows.append(dict(unit=nm, animal=an, epoch="pre", measure=m, value=pv[i] - np.delete(pv, i).mean()))
            for m, col in (("angle_spatial", "vec"), ("angle_spectral", "spec")):
                P = np.vstack(pre[col].to_numpy())
                tot = P.sum(0)
                a_loo = np.array([es.deg(float(P[i] @ unit(tot - P[i]))) for i in range(len(P))]) if len(P) > 1 \
                    else np.array([np.nan])
                loo = np.nanmean(a_loo)
                pm = unit(tot)
                for r in post.itertuples():
                    rows.append(dict(unit=nm, animal=an, epoch=r.epoch, measure=m,
                                     value=float(es.deg(float(getattr(r, col) @ pm))) - loo))
                for i in range(len(a_loo) if len(P) > 1 else 0):   # pre: own LOO angle minus the OTHER pre sessions' mean
                    rows.append(dict(unit=nm, animal=an, epoch="pre", measure=m,
                                     value=float(a_loo[i] - np.delete(a_loo, i).mean()) if len(P) > 2 else np.nan))
    return pd.DataFrame(rows)


def _row(ax, V: pd.DataFrame, R: pd.DataFrame, units: list[str], pcol: str, zero: bool = True) -> None:
    """Per unit x epoch: faded open session markers (shape = animal, colour = epoch) + mean +- SEM over sessions; stars
    (Holm, animal-weighted within-animal shuffle test) in the top band."""
    from wfield_local import cat_points as cp
    from wfield_local import sig_marks
    ecol = frc.epoch_colors()
    eps = ["pre", *frc.POST_EPOCHS]                  # pre = leave-one-out pre sessions: the pre-stroke variability
    off = cp.epoch_offsets(len(eps), step=0.21)
    for u_i, u in enumerate(units):
        for k, ep in enumerate(eps):
            v = V[(V.unit == u) & (V.epoch == ep)]
            x = u_i + off[k]
            cp.session_dots(ax, x, v.value.to_numpy(float), v.animal, ecol[ep], jitter=0.07, seed=11 * u_i + k)
            if len(v):
                se = v.value.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else None
                cp.summary_point(ax, x, v.value.mean(), se, ecol[ep], ms=5)
            r = R[(R.unit == u) & (R.epoch == ep)]
            if ep != "pre" and len(r) and np.isfinite(r[pcol].iloc[0]) and r[pcol].iloc[0] < .05:
                sig_marks.star_top(ax, x, "**" if r[pcol].iloc[0] < .01 else "*", color=ecol[ep])
    if zero:
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_xticks(range(len(units)), units, rotation=25, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(units) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)


YLAB_RAW = {"asym": "loading asymmetry\n(contra - ipsi)", "angle_spatial": "spatial angle to own\npre mean (deg)",
            "angle_spectral": "spectral angle to own\npre mean (deg)", "side_spec": "side specificity"}


def scalar_figure(Rs, V, units, path, title, ylab, zero, caption):
    import matplotlib.pyplot as plt

    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import sig_marks
    ecol = frc.epoch_colors()
    with frf.style():
        fig = plt.figure(figsize=(13.5, 2.9 * len(SCALARS)))
        gs = fig.add_gridspec(len(SCALARS), len(units), hspace=0.75)
        for i, m in enumerate(SCALARS):
            us = [u for u in units if u in set(Rs[Rs.measure == m].unit)]
            ax = fig.add_subplot(gs[i, :len(us)])                 # side_spec (2 units) gets a 2-unit-wide panel
            _row(ax, V[V.measure == m], Rs[Rs.measure == m], us, "p_holm", zero=zero or m in ("asym", "side_spec"))
            if not zero and m.startswith("angle"):
                ax.set_ylim(bottom=0)
            ax.set_ylabel(ylab[m], fontsize=8)
            if i == 0:
                hs = [plt.Line2D([], [], color=ecol[e], marker="o", ls="none",
                                 label=e + (" (leave-one-out)" if e == "pre" else "")) for e in ("pre", *frc.POST_EPOCHS)]
                ax.legend(handles=hs + cp.animal_handles(sorted(V.animal.unique())), fontsize=7, frameon=False,
                          loc="upper left", bbox_to_anchor=(1.0, 1.0))
        sig_marks.reserve_headroom(fig)
        fig.suptitle(title, y=0.995)
        frf.save(fig, path, caption)


def figures(Rs, Rl, V, out, set_name, names, Vraw=None):
    import matplotlib.pyplot as plt

    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import sig_marks
    ecol = frc.epoch_colors()
    units = [u for u in ORDER if u in names] + [u for u in names if u not in ORDER]
    common = """ASYMMETRY = (contra - ipsi) / sum of the loading. SIDE SPECIFICITY (lateral lick pair only) = mean r of the
session's loading map with its animal's own pre spout-specific movement maps at the factor's own side (close + far) minus
the opposite side; a drop = the factor looks less like movement at its own side's spouts. Stars in the top band = the animal-weighted test (mean over animals of epoch - pre; within-animal session-label shuffles): * Holm < .05, ** < .01, Holm within a family ACROSS all epochs and measures -- the lick pair (24 tests, pre-specified) or the other units (exploratory)."""
    scalar_figure(Rs, V, units, out / "session_scalars.png",
                  f"{set_name}: per-session shape readouts, NORMALISED (delta vs each animal's own pre)", YLAB, True, """
Per-session FaceRhythm fits (own8) matched to the master (rule v2), recovered sessions only. BLACK = each PRE session
against the mean of its animal's OTHER pre sessions (leave-one-out): the pre-stroke session-to-session variability.
Coloured open markers = post sessions (shape = animal, colour = epoch): the readout minus the animal's own pre mean
(angles: the angle to the animal's pre-mean vector minus the animal's mean leave-one-out pre angle; NEGATIVE = closer to
the pre mean than a typical pre session, not a direction). Filled = mean +- SEM over sessions. """ + common)
    if Vraw is not None:
        scalar_figure(Rs, Vraw, units, out / "session_scalars_raw.png",
                      f"{set_name}: per-session shape readouts, RAW", YLAB_RAW, False, """
Same sessions and tests as session_scalars.png, values NOT baseline-subtracted: each session's own asymmetry and side
specificity, and its plain angle (>= 0 deg) to its animal's pre-mean vector; BLACK pre markers = each pre session's
angle to the mean of its animal's OTHER pre sessions (how far pre sessions sit from each other). Animals differ in their
raw baseline (asymmetry, side specificity), which the normalised figure removes. Filled = mean +- SEM over sessions. """
                      + common)
    with frf.style():
        fig, ax = frf.subplots(1, 1, figsize=(13.5, 3.8))
        _row(ax, V[V.measure == "loss"], Rl, units, "p_holm")
        ax.set_ylabel(YLAB["loss"], fontsize=8)
        hs = [plt.Line2D([], [], color=ecol[e], marker="o", ls="none", label=e + (" (leave-one-out)" if e == "pre" else ""))
              for e in ("pre", *frc.POST_EPOCHS)]
        ax.legend(handles=hs + cp.animal_handles(sorted(V.animal.unique())), fontsize=7, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
        sig_marks.reserve_headroom(fig)
        fig.suptitle(f"{set_name}: factor loss per session, delta vs each animal's own pre recovery rate")
        frf.save(fig, out / "session_loss.png", """
Each open marker = one post session: recovered (1) or not (0) -- its own fit's best match to the unit reaches spatial cos
>= .7 and spectral cos >= .5 after v2 matching -- minus its animal's pre recovery rate (so markers sit at two levels per
animal). Filled = mean +- SEM over sessions. Stars = animal-weighted test (mean over animals of the recovery-rate change;
within-animal session shuffles), one-sided (fewer recovered), Holm across every unit x epoch (30 tests). BLACK = each pre
session against its animal's OTHER pre sessions (leave-one-out).""")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--match-rule", choices=sorted(es.MATCH_RULES), default="v2")
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the scalar / loss figures from data/session_scalars.csv + session_loss.csv (no tests)")
    a = ap.parse_args(argv)
    es.RULE = a.match_rule
    fdir = a.fdir or frc.default_set_dir(set_name=a.set)
    out = frc.readouts_dir(a.set) / f"session_shape_{a.match_rule}"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    side = lambda s: figure_layout.sidecar_for(out, s, ".csv")  # noqa: E731
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    T, names, home = load_sessions(fdir)
    if a.plot_only:
        figures(pd.read_csv(side("session_scalars")), pd.read_csv(side("session_loss")), session_values(T, names), out,
                fdir.name, names, Vraw=session_values(T, names, "raw"))
        print("->", out)
        return 0
    D = pd.read_csv(frc.readouts_dir(a.set).parent.parent / "validation" / a.set / "lick_epochs" / "data" / "sessions.csv")
    D["date"] = D.date.astype(str)
    tot = D[["n_left", "n_centre", "n_right"]].sum(1)
    for c in ("left", "centre", "right"):
        D[f"frac_{c}"] = D[f"n_{c}"] / tot

    Rd = scalars(T, names, a.perm, rng, drift=True)
    Rd.to_csv(side("session_scalars_driftpre"), index=False, float_format="%.4g")
    Rd["cell"] = [f"{d:+.3g} p{p:.3f}{'*' if h < .05 else ''}" for d, p, h in zip(Rd.delta, Rd.p, Rd.p_holm)]
    print("SCALARS DRIFT CONTROL (late pre - early pre):\n" + Rd.pivot(index="unit", columns="measure", values="cell")
          .reindex(index=list(names)).to_string() + "\n")
    Rs = scalars(T, names, a.perm, rng)
    Rs.to_csv(side("session_scalars"), index=False, float_format="%.4g")
    print("SCALARS (delta vs own pre; * = Holm < .05):")
    Rs["cell"] = [f"{d:+.3g} p{p:.3f}{'*' if h < .05 else ''}" for d, p, h in zip(Rs.delta, Rs.p, Rs.p_holm)]
    for m in SCALARS:
        print(f"\n{m}\n" + Rs[Rs.measure == m].pivot(index="unit", columns="epoch", values="cell")
              .reindex(index=[u for u in names if u in set(Rs[Rs.measure == m].unit)]).to_string())
    Rl = loss(T, names, a.perm, rng)
    Rl.to_csv(side("session_loss"), index=False, float_format="%.4g")
    print("\nLOSS (recovery rate delta vs own pre; one-sided):\n" + Rl.round(3).to_string(index=False))

    res_u, Mu = maps(T, list(names), home, a.perm, rng)
    res_s, Ms = maps(T, list(names), home, a.perm, rng, scale=True)
    res_a, Ma = maps(T, list(names), home, a.perm, rng, drop_ambiguous=True)
    MM = Mu.merge(Ms, on=["unit", "epoch"], suffixes=("", "_scaled")).merge(Ma, on=["unit", "epoch"], how="left",
                                                                           suffixes=("", "_unambiguous"))
    MM.to_csv(side("session_maps_compare"), index=False)
    print("\nMAPS: clusters / cluster points -- unscaled vs scaled vs unambiguous sessions only:")
    print(MM[["unit", "epoch", "clusters", "cluster_points", "clusters_scaled", "cluster_points_scaled",
              "clusters_unambiguous", "cluster_points_unambiguous", "n_sessions", "n_sessions_unambiguous"]]
          .to_string(index=False))
    import face_rhythm_session_loading_perm as sl
    sl.figure(res_s, list(LICK), home, out, "session_maps_scaled_lick.png", fdir.name + " (scaled by own pre SD)")
    sl.figure(res_a, list(LICK), home, out, "session_maps_unambiguous_lick.png",
              fdir.name + f" (unambiguous sessions only, margin >= {AMBIGUOUS})")

    RS = right_share_check(T, D, res_u)
    RS.to_csv(side("lickright_vs_right_share"), index=False, float_format="%.4g")
    print("\nlick_right map change vs change in right-lick share (within animal):\n" + RS.round(3).to_string(index=False))
    F = fragility(a.set, names)
    F.to_csv(side("pooled_fragility"), index=False, float_format="%.4g")
    print("\nFRAGILITY of the pooled-fit test (acute):\n" + F.round(2).to_string(index=False))
    Dd = direction(D, a.perm, rng)
    Dd.to_csv(side("lick_direction"), index=False, float_format="%.4g")
    print("\nLICK DIRECTION shares, delta vs own pre:\n" + Dd.round(3).to_string(index=False))
    figures(Rs, Rl, session_values(T, names), out, fdir.name, names, Vraw=session_values(T, names, "raw"))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
