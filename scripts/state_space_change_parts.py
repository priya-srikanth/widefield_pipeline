"""What is the post-stroke change made of? An exact split of each epoch's change into gain, shared shift, rotation, other
on-manifold and off-manifold parts (Priya 11-08: "so rotation is 'on manifold' but shift / pattern change is off
manifold?" -> "yes please split into parts").

Per animal and epoch, Delta = epoch template - pre template (spout x bin x feature means of first-lick trials, all
spouts jointly, spouts present in pre / acute / chronic), in the full feature space (Allen areas, LocaNMF components or
cohort SVD coordinates; WF_SS_BASIS). Nested sets of patterns built from the PRE template B = T_pre - mu (mu = per-feature
mean, as `R.fixed_space`) and the animal's top-8 pre space P:
  1 global gain      B                                  (the pre response scaled)
  2 per-area gain    B restricted to each feature        (beyond the global gain)
  3 per-area offset  a constant level per feature, the same at every spout and bin (a baseline / reference change)
  4 per-spout offset a constant level per spout x feature, beyond 3 (Priya 11-08 "is there a per spout shift piece":
                     a FREE per-spout shift would be the whole spout-specific change; its static part is a change in
                     the spout-specific level, e.g. the pre-cue position code carried into the window, F12)
  5 shared shift     any spout-constant, TIME-VARYING bin x feature pattern beyond 3 (condition-independent)
  6 rotation         B P W P^T, W skew-symmetric (k(k-1)/2 patterns): an infinitesimal rotation of the pre response
                     WITHIN the pre space (linear in W, so exact for small angles; 30 deg is ~.5 rad), beyond 1-5
  7 other, on manifold    the rest, inside P (P P^T)
  8 off manifold          the rest, outside P
Parts are the increments of orthogonal projections onto the growing sets, so they are orthogonal and their energies
(squared norms) sum exactly to |Delta|^2. Order matters for the shared variance between sets (gain before shift before
rotation, as the BCI readouts: "beyond gain"); the order is fixed in advance.

NOISE: cross-fitted like `state_space_orth_check`. Per split, pre trials in thirds and epoch trials in halves (within
session and spout); the patterns come from pre third 3 alone, Delta1 = epoch half 1 - pre third 1, Delta2 = epoch half 2 -
pre third 2; part energy = <Pi_k Delta1, Delta2> (unbiased for the noise-free energy since the two noises are
independent). N_SPLIT splits, median. Floor: "pre" row = odd vs even pre sessions (between-day drift), same procedure.

Plane: each part's contribution to the session-mean position in the 2-D plane of section 7 (x along pre -> chronic, y
along the acute departure, units of the pre -> chronic length; `R.frames`), full data, averaged over spouts. The parts sum
to x and y exactly; the off-manifold part contributes 0 by construction (the plane lies in P).

Output: <state_space_recovery>/change_parts{tag}/parts.csv (animal x epoch x part: energy, share), plane.csv, and a
figure page_figs/ss_change_parts{align}{basis}.png. EXPLORATORY.

    WF_SS_BASIS=allen|svd|lnmf WF_SS_ALIGN=lick|cue python scripts/state_space_change_parts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_orth_check as O  # noqa: E402
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402

N_SPLIT = 20
N_CHANCE = 10
OUT = S.ROOT / f"change_parts{S.tag()}"
PARTS = ["global gain", "per-area gain", "per-area offset", "per-spout offset", "shared shift", "rotation",
         "other, on manifold", "off manifold"]
COLORS = ["#1b7837", "#7fbf7b", "#80cdc1", "#35978f", "#2166ac", "#b2182b", "#f4a582", "#8c510a"]
EPOCHS = ["acute", "subacute", "chronic"]


def bases(Tp, ok, P):
    """Cumulative pattern sets (N x m matrices, N = spouts x bins x features) from the pre template."""
    mu = S.flat(Tp[ok]).mean(0)
    B = Tp[ok] - mu                                                  # s x bins x F
    s, nb, F = B.shape
    g0 = B.reshape(-1, 1)
    ga = np.zeros((s * nb * F, F))
    for f in range(F):
        m = np.zeros_like(B)
        m[..., f] = B[..., f]
        ga[:, f] = m.ravel()
    off_a = np.tile(np.eye(F), (s * nb, 1))                           # per-area offset: constant over spouts and bins
    off_s = np.zeros((s * nb * F, s * F))                             # per-spout offset: constant over bins
    for i in range(s):
        off_s[i * nb * F:(i + 1) * nb * F, i * F:(i + 1) * F] = np.tile(np.eye(F), (nb, 1))
    sh = np.tile(np.eye(nb * F), (s, 1))                              # spout-constant patterns (time-varying)
    k = P.shape[1]
    rot = []
    BP = B @ P                                                        # s x bins x k
    for i in range(k):
        for j in range(i + 1, k):
            W = np.zeros((k, k))
            W[i, j], W[j, i] = 1, -1
            rot.append((BP @ W @ P.T).ravel())
    rot = np.stack(rot, 1)
    sets = [g0, np.hstack([g0, ga]), np.hstack([g0, ga, off_a]), np.hstack([g0, ga, off_a, off_s]),
            np.hstack([g0, ga, off_a, off_s, sh]), np.hstack([g0, ga, off_a, off_s, sh, rot])]
    out = []
    for M in sets:                       # drop dependent columns (the per-feature gains contain the global one)
        q, r = np.linalg.qr(M)
        d = np.abs(np.diag(r))
        out.append(q[:, d > 1e-9 * d.max()])
    return out


def energies(D1, D2, Q, P, shape):
    """Cross-fitted part energies (6,) of Delta1 / Delta2 (flattened) for cumulative orthonormal bases Q."""
    cum = [0.0] + [float((q.T @ D1) @ (q.T @ D2)) for q in Q]
    e = [cum[i + 1] - cum[i] for i in range(len(Q))]
    q = Q[-1]
    R1, R2 = D1 - q @ (q.T @ D1), D2 - q @ (q.T @ D2)
    r1, r2 = R1.reshape(shape) @ P, R2.reshape(shape) @ P
    ein = float((r1 * r2).sum())
    return np.array(e + [ein, float(R1 @ R2) - ein]), float(D1 @ D2)


def parts_full(D, Q, P, shape):
    """The six part vectors of one Delta (flattened), exactly summing to D."""
    out, prev = [], np.zeros_like(D)
    for q in Q:
        cur = q @ (q.T @ D)
        out.append(cur - prev)
        prev = cur
    R_ = (D - prev).reshape(shape)
    rin = R_ @ P @ P.T
    return out + [rin.ravel(), (R_ - rin).ravel()]


def animal(an, sess, rng):
    by = {ep: [s for s in sess if s[1] == ep] for ep in ["pre", *EPOCHS]}
    full = {ep: R.template(v) for ep, v in by.items() if v}
    ok = np.all([np.isfinite(full[e]).all(axis=(1, 2)) for e in ("pre", "acute", "chronic") if e in full], axis=0)
    rows = []
    for k in range(N_SPLIT):
        p1, p2, p3 = O.split_template(by["pre"], 3, rng)
        P = S.topk(p3[ok])
        Q = bases(p3, ok, P)
        shape = p3[ok].shape
        for ep in EPOCHS:
            if not by.get(ep):
                continue
            a1, a2 = O.split_template(by[ep], 2, rng)
            if not (np.isfinite(a1[ok]).all() and np.isfinite(a2[ok]).all()):
                continue
            e, tot = energies((a1[ok] - p1[ok]).ravel(), (a2[ok] - p2[ok]).ravel(), Q, P, shape)
            rows.append({"animal": an, "epoch": ep, "split": k, "total": tot, **dict(zip(PARTS, e))})
        if k == 0:                                   # CHANCE: where an isotropic random change would land
            for _ in range(N_CHANCE):
                g = rng.standard_normal(Q[0].shape[0])
                e, tot = energies(g, g, Q, P, shape)
                rows.append({"animal": an, "epoch": "chance (random change)", "split": k, "total": tot,
                             **dict(zip(PARTS, e))})
        odd, even = by["pre"][0::2], by["pre"][1::2]                    # floor: between-day pre differences
        if len(odd) >= 2 and len(even) >= 2:
            o1, o2 = O.split_template(odd, 2, rng)
            v1, v2 = O.split_template(even, 2, rng)
            if all(np.isfinite(x[ok]).all() for x in (o1, o2, v1, v2)):
                e, tot = energies((o1[ok] - v1[ok]).ravel(), (o2[ok] - v2[ok]).ravel(), Q, P, shape)
                rows.append({"animal": an, "epoch": "pre (odd - even days)", "split": k, "total": tot,
                             **dict(zip(PARTS, e))})
    # plane contributions (full data; the frame of section 7)
    P, mu, fr = R.frames(sess)
    Q = bases(full["pre"], ok, P)
    shape = full["pre"][ok].shape
    sp_ok = [p for i, p in enumerate(S.POS) if ok[i]]
    plane = []
    for ep in EPOCHS:
        if ep not in full or not np.isfinite(full[ep][ok]).all():
            continue
        pv = parts_full((full[ep][ok] - full["pre"][ok]).ravel(), Q, P, shape)
        for name, v in zip(PARTS, pv):
            z = v.reshape(shape) @ P                                      # spouts x bins x k (a difference: no mu)
            xs, ys = [], []
            for i, p in enumerate(sp_ok):
                if p in fr:
                    x, y = R.xy(z[i].ravel(), fr[p])
                    xs.append(x)
                    ys.append(y)
            plane.append({"animal": an, "epoch": ep, "part": name, "x": float(np.mean(xs)), "y": float(np.mean(ys))})
    return rows, plane


def figure(T, Pl, out):
    """Left: each part's energy (units of the animal's pre response energy), mean of animals, stacked (negative
    cross-fitted values, = noise, drawn as 0). Right: each part's contribution to the plane position, dots = animals."""
    eps = ["pre (odd - even days)", *EPOCHS]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axs[0]
    for i, ep in enumerate(eps):
        g = T[T.epoch == ep]
        if not len(g):
            continue
        bottom = 0.0
        for name, c in zip(PARTS, COLORS):
            v = float(np.mean(np.clip(g[name] / g.pre_unit, 0, None)))
            ax.bar(i, v, 0.6, bottom=bottom, color=c, label=name if i == 1 else None)
            bottom += v
    ax.set_xticks(range(len(eps)), ["pre\n(odd vs even days)", *EPOCHS])
    ax.set_ylabel("energy of the change\n(units of the animal's pre response energy)")
    ax.set_title("size of each part (cross-fitted, mean of 4 animals)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for ax, coord, ttl in ((axs[1], "x", "contribution to x: along pre (0) -> chronic (1)"),
                           (axs[2], "y", "contribution to y: acute departure (orthogonal)")):
        ax.axhline(0, color="0.6", lw=0.8)
        for i, ep in enumerate(EPOCHS):
            for pi, (name, c) in enumerate(zip(PARTS[:-1], COLORS[:-1])):
                g = Pl[(Pl.epoch == ep) & (Pl.part == name)]
                xx = i + (pi - 3) * 0.11
                ax.scatter(np.full(len(g), xx), g[coord], color=c, s=14, alpha=0.6, zorder=3)
                ax.plot([xx - 0.06, xx + 0.06], [g[coord].mean()] * 2, color=c, lw=2.2)
        ax.set_xticks(range(3), EPOCHS)
        ax.set_title(ttl)
        ax.set_ylabel("units of the pre -> chronic length")
    xf = {"": "", "gsr": "; cortex-wide mode removed"}.get(S.XFORM, f"; {S.XFORM}")
    fig.suptitle(f"What the change is made of ({S.ALIGN_LABEL[S.ALIGN]}; {S.BASIS_LABEL[S.BASIS] or 'Allen areas'}{xf}); "
                 "off manifold adds 0 to the plane by construction", y=1.02)
    fig.tight_layout()
    save_png_svg(fig, out, dpi=140, bbox_inches="tight")
    plt.close(fig)


PRE_UNIT: dict = {}
N_SPLIT_SESS = 8


def _pre_unit(pre_sess):
    T = R.template(pre_sess)
    ok = np.isfinite(T).all(axis=(1, 2))
    return float(((T[ok] - S.flat(T[ok]).mean(0)) ** 2).sum())


def session_rows(an, sess, rng):
    """PER SESSION (Priya 11-08 "per-session dots rather than per animal (normalized within animal)"): each session's
    template vs the animal's pre template, cross-fitted part energies (median of N_SPLIT_SESS splits) in units of the
    animal's pre response energy, + each part's x / y contribution in the section-7 plane (full data). Pre sessions are
    scored LEAVE-ONE-OUT against the other pre sessions: the null dots."""
    pre = [s for s in sess if s[1] == "pre"]
    unit = _pre_unit(pre)
    Pf, _mu, fr = R.frames(sess)
    cache = {}

    def basis(ref_sess, key, ok):
        k = (key, tuple(ok))
        if k not in cache:
            T = R.template(ref_sess)
            P = S.topk(T[ok])
            cache[k] = (P, bases(T, ok, P))
        return cache[k]
    out = []
    for lab, ep, day, Y, pos in sess:
        if ep not in ("pre", *EPOCHS):
            continue
        ref = [s for s in pre if s[0] != lab]                          # pre: leave this session out
        Tses = R.template([(lab, ep, day, Y, pos)])
        Tref = R.template(ref)
        ok = np.isfinite(Tses).all(axis=(1, 2)) & np.isfinite(Tref).all(axis=(1, 2))
        if ok.sum() < 3:
            continue
        es = []
        for _k in range(N_SPLIT_SESS):
            p1, p2, p3 = O.split_template(ref, 3, rng)
            t1, t2 = O.split_template([(lab, ep, day, Y, pos)], 2, rng)
            okk = ok & np.all([np.isfinite(x).all(axis=(1, 2)) for x in (p1, p2, p3, t1, t2)], axis=0)
            if okk.sum() < 3:
                continue
            P = S.topk(p3[okk])
            Q = bases(p3, okk, P)
            e, tot = energies((t1[okk] - p1[okk]).ravel(), (t2[okk] - p2[okk]).ravel(), Q, P, p3[okk].shape)
            es.append(np.r_[tot, e])
        if not es:
            continue
        med = np.median(es, 0) / unit
        row = {"animal": an, "label": lab, "epoch": ep, "day": day, "n_spouts": int(ok.sum()), "total": med[0],
               **{f"E {p}": v for p, v in zip(PARTS, med[1:])}}
        # plane contributions (full data; pre sessions vs the other pre sessions)
        P_, Q_ = basis(ref, "loo-" + lab if ep == "pre" else "pre", ok)
        pv = parts_full((Tses[ok] - Tref[ok]).ravel(), Q_, P_, Tses[ok].shape)
        sp = [p for i, p in enumerate(S.POS) if ok[i]]
        for name, v in zip(PARTS, pv):
            z = v.reshape(Tses[ok].shape) @ Pf
            xy = [R.xy(z[i].ravel(), fr[p]) for i, p in enumerate(sp) if p in fr]
            row[f"x {name}"] = float(np.mean([a for a, _ in xy])) if xy else np.nan
            row[f"y {name}"] = float(np.mean([b for _, b in xy])) if xy else np.nan
        out.append(row)
    return out


def fig_sessions(D, out):
    """3 rows (energy; x contribution; y contribution) x parts; sessions = open markers (shape = animal, color = epoch),
    minus the animal's own pre mean; pooled change + 95 % CI (boot_delta); * = CI excludes 0, dagger = LMM backup."""
    import state_space_page_figs as F
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(3, len(PARTS), figsize=(3.5 * len(PARTS), 10.5))
    for c, name in enumerate(PARTS):
        F.epoch_panel(axs[0, c], D, f"E {name}", f"{name}\nenergy (pre response energy = 1)", rng)
        for r, coord, lab in ((1, "x", "contribution to x (along pre -> chronic)"),
                              (2, "y", "contribution to y (acute departure)")):
            if name == "off manifold":
                axs[r, c].axis("off")
                continue
            F.epoch_panel(axs[r, c], D, f"{coord} {name}", f"{name}\n{lab}", rng)
    for ax in axs[:, 0]:
        ax.set_ylabel("change vs own pre")
    F.legend_animals_epochs(fig, y=1.0)
    xf = {"": "", "gsr": "; cortex-wide mode removed"}.get(S.XFORM, f"; {S.XFORM}")
    fig.suptitle(f"What the change is made of, per session ({S.ALIGN_LABEL[S.ALIGN]}; "
                 f"{S.BASIS_LABEL[S.BASIS] or 'Allen areas'}{xf})", y=1.035)
    fig.tight_layout()
    F.flush_marks(fig)
    F.sig_marks.reserve_headroom(fig)
    save_png_svg(fig, out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main_sessions():
    rng = np.random.default_rng(0)
    data, _ = S.load_sessions()
    rows = []
    for an, sess in sorted(data.items()):
        rows += session_rows(an, sess, rng)
        print(f"  {an} sessions done", flush=True)
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "sessions.csv", index=False)
    fig_sessions(D, S.ROOT / "page_figs" / f"ss_change_parts_sessions{S.tag()}.png")
    with pd.option_context("display.width", 250, "display.precision", 3):
        print(D.groupby(["epoch"])[[c for c in D if c.startswith("E ")]].median().to_string())
    print(f"wrote {OUT / 'sessions.csv'}")


def main():
    if "--sessions" in sys.argv:
        return main_sessions()
    rng = np.random.default_rng(0)
    data, _ = S.load_sessions()
    rows, plane = [], []
    for an, sess in sorted(data.items()):
        r, p = animal(an, sess, rng)
        rows += r
        plane += p
        pre = R.template([s for s in sess if s[1] == "pre"])
        okp = np.isfinite(pre).all(axis=(1, 2))
        PRE_UNIT[an] = float(((pre[okp] - S.flat(pre[okp]).mean(0)) ** 2).sum())
        print(f"  {an} done", flush=True)
    D = pd.DataFrame(rows)
    T = D.groupby(["animal", "epoch"])[["total", *PARTS]].median().reset_index()
    for name in PARTS:
        T[f"share {name}"] = T[name] / T["total"]
    T["pre_unit"] = T.animal.map(PRE_UNIT)
    Pl = pd.DataFrame(plane)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "parts_splits.csv", index=False)
    T.to_csv(OUT / "parts.csv", index=False)
    Pl.to_csv(OUT / "plane.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 3):
        print("\n=== share of the change's energy per part (cross-fitted, median of splits) ===")
        print(T[["animal", "epoch", "total", *[f"share {p}" for p in PARTS]]].to_string(index=False))
        print("\n=== contribution of each part to the plane position (units of pre -> chronic length) ===")
        print(Pl.pivot_table(index=["epoch", "part"], columns="animal", values=["x", "y"], sort=False).to_string())
        rng2 = np.random.default_rng(1)
        print("\n=== cohort: share of the change per part, mean of animals (boot over animals); chance = an "
              "isotropic random change ===")
        for ep in ["chance (random change)", "pre (odd - even days)", *EPOCHS]:
            for name in PARTS:
                v = {a: [g[f"share {name}"].iloc[0]] for a, g in T[T.epoch == ep].groupby("animal")}
                iv = kit.boot_ci(v, rng2) if v else None
                if iv is not None:
                    print(f"  {ep:24s} {name:20s} {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}]")
    figure(T, Pl, S.ROOT / "page_figs" / f"ss_change_parts{S.tag()}.png")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
