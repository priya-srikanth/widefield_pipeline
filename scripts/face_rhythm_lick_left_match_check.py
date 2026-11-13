"""Gut check of the lick_left MATCH (Priya 11-12: "look at lick left factors' spatial weighting, for the per-session and
pooled analyses ... to gut check that it's matching correctly"). Draws the spatial loading of the own-fit column that
rule v2 (`face_rhythm_epoch_shape.pick`: best spatial cosine; ties within .05 -> best spectral; recovered at spatial >= .7
and spectral >= .5) assigns to a master factor, in:

  pooled_<factor>.png        COHORT + LOAO fits (rows) x master / pre / acute / subacute / chronic; where the runner-up
                             column is within .05 of the pick, it is drawn underneath (second row block)
  session_mean_<factor>.png  per-session fits (own8), mean matched loading per animal x epoch (recovered sessions)
  sessions_<factor>_<animal>.png  every session's matched column, in date order

Every panel: loading magnitude per mesh point as a fraction of the column's total (shape only), image left =
contralesional, line = face midline; title = spatial / frequency cosine to the master, margin to the runner-up, and the
column's loading asymmetry (contra - ipsi) / sum. RED title = not recovered, margin < .05, or spatial cosine < .8.

    python scripts/face_rhythm_lick_left_match_check.py [--factor lick_left]
-> ``<readouts>/master_pre_v1/lick_left_match_check/``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_rhythm_epoch_shape as es  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402

TAGS = ("COHORT", "LOAO-PS92", "LOAO-PS93", "LOAO-PS94", "LOAO-PS95")
GROUPS = ("pre", "acute", "subacute", "chronic")
RED = "#b2182b"


def col_info(A_ref, B_ref, A, B, k, home, mid):
    """Matched column for master factor k under the current rule, its runner-up, and their readouts."""
    from wfield_local import face_rhythm_matching as fm
    m, cols, rec = es.pick(A_ref, B_ref, A, B)
    cs, cf = m["cs_all"][k], m["cf_all"][k]
    c = int(cols[k])
    others = [j for j in np.argsort(cs)[::-1] if j != c]
    r = int(others[0])
    mag = fm.spatial_view(A, "magnitude")
    asym = fm.loading_asymmetry(A, home, mid)

    def one(j):
        return dict(col=j, load=mag[:, j] / mag[:, j].sum(), cs=float(cs[j]), cf=float(cf[j]), asym=float(asym[j]))
    return dict(pick=one(c), runner=one(r), margin=float(cs[c] - cs[r]), rec=bool(rec[k]))


def draw(ax, home, load, hi, title, bad, mid):
    ax.scatter(home[:, 0], home[:, 1], c=load, s=3, cmap="magma_r", vmin=0, vmax=hi)
    ax.axvline(mid, color="0.4", lw=0.4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=6.5, color=RED if bad else "k")


def ttl(d, extra=""):
    p = d["pick"]
    return (f"{extra}own {p['col']}  cos {p['cs']:.2f}  freq {p['cf']:.2f}\nmargin {d['margin']:.2f}  "
            f"asym {p['asym']:+.2f}")


def is_bad(d):
    return (not d["rec"]) or d["margin"] < 0.05 or d["pick"]["cs"] < 0.8


def pooled_figure(fdir, nm, out):
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_readouts as frr
    M, A_ref, B_ref, names = es.reference(fdir)
    home, mid, k = M["home"], float(frr.MASTER_MIDLINE_X), names.index(nm)
    mref = fm.spatial_view(A_ref, "magnitude")[:, k]
    mref = mref / mref.sum()
    masym = float(fm.loading_asymmetry(A_ref, home, mid)[k])
    D = {}
    for t in TAGS:
        for g in GROUPS:
            with np.load(fdir / "own_epoch_cohort" / f"{t}_{g}.npz") as z:
                D[(t, g)] = col_info(A_ref, B_ref, z["A_own"].astype(float), z["B_own"].astype(float), k, home, mid)
    close = [(t, g) for (t, g), d in D.items() if d["margin"] < 0.05]
    nrow = len(TAGS) + (1 if close else 0)
    ncol = 1 + max(len(GROUPS), len(close))
    hi = float(np.percentile(np.concatenate([mref] + [d["pick"]["load"] for d in D.values()]), 99.5))
    with frf.style():
        fig, axs = frf.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow), squeeze=False)
        for a in axs.ravel():
            a.axis("off")
        for r, t in enumerate(TAGS):
            draw(axs[r, 0], home, mref, hi, f"{es.tag_label(t)}\nmaster  asym {masym:+.2f}", False, mid)
            for c, g in enumerate(GROUPS):
                d = D[(t, g)]
                draw(axs[r, c + 1], home, d["pick"]["load"], hi, ttl(d, f"{g}: "), is_bad(d), mid)
        if close:
            axs[-1, 0].text(0.5, 0.5, "RUNNER-UP columns\n(margin < .05)", ha="center", va="center", fontsize=8,
                            transform=axs[-1, 0].transAxes)
            for c, (t, g) in enumerate(close):
                rr = D[(t, g)]["runner"]
                draw(axs[-1, c + 1], home, rr["load"], hi, f"{es.tag_label(t)} {g} runner-up\nown {rr['col']}  cos "
                     f"{rr['cs']:.2f}  freq {rr['cf']:.2f}\nasym {rr['asym']:+.2f}", False, mid)
        fig.suptitle(f"{nm}: matched column in each pooled fit (rule v2); image left = contralesional")
        frf.save(fig, out / f"pooled_{nm}.png", f"""
{nm} in the all-animals pooled fits (COHORT) and the leave-one-animal-out fits (rows). Each panel = the own-fit column
rule v2 assigns to the master {nm} (best spatial cosine; columns within .05 of it -> best frequency cosine), drawn as
loading magnitude per mesh point / total (one color scale for the figure). Titles: spatial / frequency cosine with the
master, margin of the best spatial cosine over the runner-up column, loading asymmetry (contra - ipsi) / sum. RED =
margin < .05, cosine < .8 or not recovered. Bottom row (if any) = the runner-up column for every fit with margin < .05.""")
    return D


def session_figures(fdir, nm, out):
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_readouts as frr
    M, A_ref, B_ref, names = es.reference(fdir)
    home, mid, k = M["home"], float(frr.MASTER_MIDLINE_X), names.index(nm)
    mref = fm.spatial_view(A_ref, "magnitude")[:, k]
    mref = mref / mref.sum()
    S = frc.sessions(fdir, "proj")
    rows = []
    for t in S.itertuples():
        f = fdir / "own8" / f"{t.key}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            d = col_info(A_ref, B_ref, z["A_own"].astype(float), z["B_own"].astype(float), k, home, mid)
        rows.append(dict(key=t.key, animal=t.animal, date=str(t.date), epoch=t.epoch, **d))
    animals = sorted({r["animal"] for r in rows})
    acol = frc.epoch_colors()
    hi = float(np.percentile(np.concatenate([mref] + [r["pick"]["load"] for r in rows]), 99.5))
    with frf.style():
        fig, axs = frf.subplots(len(animals), 1 + len(GROUPS), figsize=(2.3 * (1 + len(GROUPS)), 2.5 * len(animals)),
                                squeeze=False)
        summ = []
        for r_i, an in enumerate(animals):
            draw(axs[r_i, 0], home, mref, hi, f"{an}\nmaster", False, mid)
            for c, g in enumerate(GROUPS):
                R = [r for r in rows if r["animal"] == an and r["epoch"] == g and r["rec"]]
                ax = axs[r_i, c + 1]
                if not R:
                    ax.axis("off")
                    continue
                load = np.mean([r["pick"]["load"] for r in R], 0)
                nb = sum(is_bad(r) for r in R)
                a = np.mean([r["pick"]["asym"] for r in R])
                summ.append((an, g, len(R), nb, a))
                draw(ax, home, load, hi, f"{g}: mean of {len(R)} sessions\nasym {a:+.2f}  ({nb} flagged)", False, mid)
        fig.suptitle(f"{nm}: matched column per session fit, mean per animal x epoch (recovered sessions, rule v2)")
        frf.save(fig, out / f"session_mean_{nm}.png", f"""
{nm} in the per-session fits (own8, 8 starts): per animal (rows) and epoch, the MEAN loading (magnitude per mesh point /
total) of the column rule v2 assigns to the master {nm}, over recovered sessions; one color scale. Titles: number of
sessions, mean loading asymmetry (contra - ipsi) / sum of the matched columns, and how many of them are flagged (margin
to the runner-up < .05 or spatial cosine < .8; see the per-animal session sheets).""")
        for an in animals:
            R = sorted([r for r in rows if r["animal"] == an], key=lambda r: r["date"])
            nc = 8
            nr = int(np.ceil((len(R) + 1) / nc))
            fig, axs = frf.subplots(nr, nc, figsize=(2.0 * nc, 2.3 * nr), squeeze=False)
            for a in axs.ravel():
                a.axis("off")
            draw(axs[0, 0], home, mref, hi, "master", False, mid)
            for i, r in enumerate(R, start=1):
                ax = axs[i // nc, i % nc]
                draw(ax, home, r["pick"]["load"], hi, f"{r['date']} {r['epoch']}\n" + ttl(r)
                     + ("" if r["rec"] else "\nNOT RECOVERED"), is_bad(r), mid)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.add_patch(__import__("matplotlib").patches.Rectangle((0, 1.0), 1, 0.03, transform=ax.transAxes,
                                                                        color=acol.get(r["epoch"], "0.5"),
                                                                        clip_on=False))
            fig.suptitle(f"{an}: {nm} matched column in every session fit (date order; bar = epoch color)")
            frf.save(fig, out / f"sessions_{nm}_{an}.png", f"""
{an}, every session's own fit (own8): the column rule v2 assigns to the master {nm} (first panel), loading magnitude per
mesh point / total, one color scale shared with session_mean_{nm}.png. Colored bar = epoch. Titles: own column index,
spatial / frequency cosine with the master, margin over the runner-up column, loading asymmetry (contra - ipsi) / sum.
RED = margin < .05, cosine < .8 or not recovered.""")
    return rows, summ


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import writeguard
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--factor", default="lick_left")
    a = ap.parse_args(argv)
    es.RULE = "v2"
    fdir = frc.default_set_dir(set_name=fro.SET)
    out = frc.readouts_dir(fro.SET) / "lick_left_match_check"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pooled_figure(fdir, a.factor, out)
    print("POOLED (tag, group): col, cos, freq, margin, asym")
    for (t, g), d in D.items():
        p = d["pick"]
        print(f"  {t:10s} {g:9s} own {p['col']:2d}  cos {p['cs']:.2f}  freq {p['cf']:.2f}  margin {d['margin']:.2f}  "
              f"asym {p['asym']:+.2f}" + ("  FLAG" if is_bad(d) else ""))
    rows, summ = session_figures(fdir, a.factor, out)
    print("PER SESSION (animal, epoch): n recovered, n flagged, mean asym")
    for s in summ:
        print(f"  {s[0]} {s[1]:9s} n {s[2]:2d}  flagged {s[3]:2d}  asym {s[4]:+.2f}")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
