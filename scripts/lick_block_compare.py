"""Nested blocks + Shapley split for the frozen lateral lick DECODER (Priya 10-28).

Cortex (24 decoder areas, left + right together) is cut into 6 DISJOINT groups:
  oro      ALM, tjM1, SSp-m, SSp-n
  bf+un    SSp-bfd, SSp-un
  MO-rest  MOs_rest, MOp_rest
  body     SSp-ul, SSp-ll, SSp-tr        (the Allen atlas has no MO trunk area: MOp / MOs are not divided by body part)
  SSs      secondary somatosensory
  other    ACAd, PL, RSP*, VIS*
and the decoder is refit on EVERY non-empty combination (63 per animal x lick set; same features, z-scoring, ridge,
leave-one-pre-session-out and like-for-like offset as lick_area_unique_maps, 11.13). Value of a combination: pre held-out
r (mean of pre sessions) and the acute / chronic offset change vs pre (mm); empty set = 0.
From that one lattice:
  1. NESTED BLOCKS (Priya's comparison): orofacial (oro) < oro+bf+un < sensorimotor (+ MO-rest + body) < +SSs (+ SSs);
     full vs WITHOUT the block vs block ALONE, unique to block = full - without, unique to rest = full - alone.
  2. LADDER: the same blocks read cumulatively (4 -> 6 -> 11 -> 12 -> 24 areas), and the reverse (removing them).
  3. SHAPLEY: each group's average marginal contribution over all orders of adding groups; the 6 shares sum EXACTLY
     to the full decoder's value, so shared information is split rather than double-counted (alone) or dropped (unique).
Per-animal tables printed first (rule 8). Outputs <OUT_W>/block_lattice_sessions.csv, block_compare_summary.csv,
block_shapley.csv; figures lick_block_compare.png, lick_block_ladder_shapley.png (figdata + --plot-only).
"""
from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
import lick_area_importance as lai  # noqa: E402
import lick_refit_decoders as rd  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

# VARIANT (env WFIELD_BLOCK_VARIANT, read at import so spawned workers see it): "" = 6 groups with SSs on its own (11.14);
# "oroSSs" = SSs folded into the orofacial group (Priya 10-28: "redo Shapley and include SSs in the orofacial
# sensorimotor group"; S2 has a mouth representation and carried no unique information), 5 groups.
VARIANT = os.environ.get("WFIELD_BLOCK_VARIANT", "")
if VARIANT == "oroSSs":
    GROUPS = {
        "oro+SSs": ("ALM", "tjM1", "SSp-m", "SSp-n", "SSs"),
        "bf+un": ("SSp-bfd", "SSp-un"),
        "MO-rest": ("MOs_rest", "MOp_rest"),
        "body": ("SSp-ul", "SSp-ll", "SSp-tr"),
        "other": None,
    }
    BLOCKS = {
        "oro+SSs": ("oro+SSs",),
        "+bf+un": ("oro+SSs", "bf+un"),
        "sensorimotor": ("oro+SSs", "bf+un", "MO-rest", "body"),
    }
else:
    GROUPS = {
        "oro": ("ALM", "tjM1", "SSp-m", "SSp-n"),
        "bf+un": ("SSp-bfd", "SSp-un"),
        "MO-rest": ("MOs_rest", "MOp_rest"),
        "body": ("SSp-ul", "SSp-ll", "SSp-tr"),
        "SSs": ("SSs",),
        "other": None,                     # every remaining area
    }
    BLOCKS = {                             # nested, as sets of groups
        "orofacial": ("oro",),
        "oro+bf+un": ("oro", "bf+un"),
        "sensorimotor": ("oro", "bf+un", "MO-rest", "body"),
        "+SSs": ("oro", "bf+un", "MO-rest", "body", "SSs"),
    }
GN = list(GROUPS)
SFX = f"_{VARIANT}" if VARIANT else ""
N_AREAS = 24
MEAS = [("pre_r", "pre held-out r"), ("off_acute", "acute offset change (mm)"), ("off_chronic", "chronic offset change (mm)")]
COLS = [("full", "full"), ("without", "without block"), ("alone", "block alone"),
        ("uniq_block", "full - without\n(unique to block)"), ("uniq_rest", "full - alone\n(unique to rest)")]


def _group_idx(names: list[str]) -> dict:
    base = [n.rsplit("_", 1)[0] for n in names]
    named = {a for g in GROUPS.values() if g for a in g}
    miss = named - set(base)
    assert not miss, f"areas not in the decoder: {miss}"
    return {g: np.array([i for i, b in enumerate(base) if (b in areas if areas else b not in named)])
            for g, areas in GROUPS.items()}


def _key(combo) -> str:
    return "+".join(g for g in GN if g in combo)


def lattice_unit(item) -> list[dict]:
    """All 63 group combinations for one (lick set, animal). Module-level for spawn."""
    st, a = item
    names = list(wd.area_meta()[0])
    gi = _group_idx(names)
    D = rd.load_z()
    S = {x: (d["epoch"], *d[st]) for x, d in sorted(D.items()) if d["animal"] == a and len(d[st][1]) >= rd.MIN_LICKS}
    pre = [x for x, v in S.items() if v[0] == "pre"]
    if len(pre) < 3:
        return []
    rows = []
    for k in range(1, len(GN) + 1):
        for combo in itertools.combinations(GN, k):
            keep = np.sort(np.concatenate([gi[g] for g in combo]))
            fits = {h: rd._fit(np.vstack([S[q][1][:, keep] for q in pre if q != h]),
                               np.vstack([S[q][2][:, :1] for q in pre if q != h])) for h in [*pre, None]}
            dec = {x: rd._pred(fits[x if v[0] == "pre" else None], v[1][:, keep])[:, 0] for x, v in S.items()}
            for x, (ep, _X, Y) in S.items():
                act = Y[:, 0]
                ref = [q for q in pre if q != x]
                off = lai._offset(np.concatenate([S[q][2][:, 0] for q in ref]), np.concatenate([dec[q] for q in ref]),
                                  act, dec[x])
                rows.append(dict(set=st, animal=a, label=x, epoch=ep, combo=_key(combo), n_groups=k, n_feat=len(keep),
                                 r=float(np.corrcoef(dec[x], act)[0, 1]), offset=float(np.nanmean(off))))
    return rows


def _values(R: pd.DataFrame) -> pd.DataFrame:
    """Per set x animal x combo: pre_r, off_acute, off_chronic (epoch change vs pre)."""
    A = R.groupby(["set", "animal", "combo", "epoch"])[["r", "offset"]].mean().reset_index()
    recs = []
    for (st, an, c), t in A.groupby(["set", "animal", "combo"]):
        t = t.set_index("epoch")
        rec = {"set": st, "animal": an, "combo": c, "pre_r": t.loc["pre", "r"]}
        for ep in ("acute", "chronic"):
            rec[f"off_{ep}"] = t.loc[ep, "offset"] - t.loc["pre", "offset"] if ep in t.index else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def compute() -> dict:
    from wfield_local import analysis_kit as ak
    items = [(st, a) for st in rd.SETS for a in lai.AN]
    res, fail = ak.fan_sessions(items, lattice_unit, jobs=8, label="set x animal")
    assert not fail, f"failed units: {fail}"
    R = pd.DataFrame([r for _item, rows in res for r in rows])
    R = R.sort_values(["set", "animal", "combo", "label"]).reset_index(drop=True)
    R.to_csv(il.OUT_W / f"block_lattice_sessions{SFX}.csv", index=False)
    V = _values(R)
    full = _key(GN)
    rest = {b: _key([g for g in GN if g not in gs]) for b, gs in BLOCKS.items()}
    out = []
    for (st, an), t in V.groupby(["set", "animal"]):
        t = t.set_index("combo")
        for b, gs in BLOCKS.items():
            for m, _ in MEAS:
                f, al, wo = t.loc[full, m], t.loc[_key(gs), m], t.loc[rest[b], m]
                out.append(dict(set=st, block=b, animal=an, measure=m, full=f, without=wo, alone=al, uniq_block=f - wo,
                                uniq_rest=f - al))
    Q = pd.DataFrame(out)
    Q.to_csv(il.OUT_W / f"block_compare_summary{SFX}.csv", index=False)
    # Shapley over the 6 groups
    n = len(GN)
    sh = []
    for (st, an), t in V.groupby(["set", "animal"]):
        t = t.set_index("combo")
        for m, _ in MEAS:
            def v(c, t=t, m=m):
                return 0.0 if not c else float(t.loc[_key(c), m])
            for g in GN:
                others = [h for h in GN if h != g]
                phi = sum(math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                          * (v((*c, g)) - v(c)) for k in range(n) for c in itertools.combinations(others, k))
                sh.append(dict(set=st, animal=an, measure=m, group=g, shapley=phi, full=v(GN)))
    SH = pd.DataFrame(sh)
    SH.to_csv(il.OUT_W / f"block_shapley{SFX}.csv", index=False)
    chk = SH.groupby(["set", "animal", "measure"]).agg(s=("shapley", "sum"), f=("full", "first"))
    assert np.allclose(chk.s, chk.f, atol=1e-9), "Shapley shares do not sum to the full value"
    pd.set_option("display.width", 230)
    print("== nested blocks, per animal (rule 8)\n" + Q.set_index(["set", "measure", "block", "animal"]).round(3).to_string())
    print("== nested blocks, mean of animals (n animals > 0 for the unique columns)")
    for (st, m, b), t in Q.groupby(["set", "measure", "block"], sort=False):
        print(f"{st:6s} {m:12s} {b:13s} " + "  ".join(f"{c}={t[c].mean():+.3f}" for c, _ in COLS)
              + f"   uniq_block>0 {int((t.uniq_block > 0).sum())}/{len(t)}  uniq_rest>0 {int((t.uniq_rest > 0).sum())}/{len(t)}")
    print("== Shapley per animal\n" + SH.pivot_table(index=["set", "measure", "group"], columns="animal", values="shapley")
          .reindex(GN, level=2).round(3).to_string())
    print("== Shapley mean of animals (share of full)")
    M = SH.groupby(["set", "measure", "group"]).shapley.mean().unstack("group")[GN]
    print(M.round(3).assign(full=M.sum(1).round(3)).to_string())
    return {"Q": Q, "V": V, "SH": SH}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Q, V, SH = F["Q"], F["V"], F["SH"]
    nsz = {b: sum(len(GROUPS[g]) for g in gs) for b, gs in BLOCKS.items()}
    # 1) nested-block comparison
    clr = ["0.25", "#999999", "#4C72B0", "#C44E52", "#55A868"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 9.5))
    nb, w = len(BLOCKS), 0.16
    for i, st in enumerate(rd.SETS):
        for j, (m, ttl) in enumerate(MEAS):
            ax = axs[i, j]
            t = Q[(Q["set"] == st) & (Q.measure == m)]
            for k, (c, lab) in enumerate(COLS):
                xs = np.arange(nb) + (k - 2) * w + (0.04 if k >= 3 else 0)
                ax.bar(xs, [t[t.block == b][c].mean() for b in BLOCKS], width=w * 0.92, color=clr[k],
                       label=lab if (i, j) == (0, 0) else None, hatch="//" if k >= 3 else None,
                       edgecolor="w" if k < 3 else "k", lw=0.4)
                for xi, b in enumerate(BLOCKS):
                    vv = t[t.block == b][c].to_numpy()
                    ax.plot(np.full(len(vv), xs[xi]), vv, "k.", ms=3.5, alpha=0.7)
                    if k >= 3:
                        ax.text(xs[xi], max(np.nanmax(vv), 0) + 0.01, f"{int((vv > 0).sum())}/{len(vv)}", ha="center",
                                va="bottom", fontsize=6.5)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(np.arange(nb), [f"{b}\n({nsz[b]} areas)" for b in BLOCKS], fontsize=8)
            ax.set_title(f"{rd.SETS[st]} licks: {ttl}", fontsize=10)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.legend(loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Nested blocks, frozen lateral lick decoder: full vs without vs alone; unique to block = full - without; "
                 "unique to rest of cortex = full - alone.\norofacial = ALM, tjM1, SSp-m, -n; oro+bf+un = + SSp-bfd, -un; "
                 "sensorimotor = + MOs-rest, MOp-rest, SSp-ul, -ll, -tr; +SSs = + SSs (full = 24 areas)."
                 + (" VARIANT oroSSs: SSs inside the orofacial group (oro+SSs; +bf+un; sensorimotor)." if VARIANT else "")
                 + " Bars = mean, "
                 "dots = animals, n/4 = animals > 0. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(il.OUT_W / "figures" / f"lick_block_compare{SFX}.png", dpi=95)
    plt.close(fig)
    # 2) ladder + Shapley
    steps = [*BLOCKS.items(), ("all cortex", tuple(GN))]
    n_named = sum(len(a) for a in GROUPS.values() if a)
    xl = [f"{s}\n({sum(len(GROUPS[g]) if GROUPS[g] else N_AREAS - n_named for g in gs)})" for s, gs in steps]
    pal = {"oro": "#C44E52", "oro+SSs": "#C44E52", "bf+un": "#DD8452", "MO-rest": "#8172B3", "body": "#937860",
           "SSs": "#DA8BC3", "other": "#8C8C8C"}
    gcol = {g: pal[g] for g in GN}
    fig, axs = plt.subplots(2, 4, figsize=(21, 9.5), gridspec_kw={"width_ratios": [1, 1, 1.1, 1.1]})
    for i, st in enumerate(rd.SETS):
        v = V[V["set"] == st]
        for j, (ms, ttl) in enumerate(((["pre_r"], "LADDER: pre held-out r"),
                                        (["off_acute", "off_chronic"], "LADDER: offset change (mm)"))):
            ax = axs[i, j]
            for m, ls in zip(ms, ("-", "--"), strict=False):
                for direction, mk, al in (("add", "o", 1.0), ("remove", "s", 0.55)):
                    ys = []
                    for _s, gs in steps:
                        c = _key(gs) if direction == "add" else _key([g for g in GN if g not in gs])
                        ys.append(v[v.combo == c].set_index("animal")[m] if c else pd.Series(0.0, index=v.animal.unique()))
                    Y = pd.concat(ys, axis=1)
                    for _an, row in Y.iterrows():
                        ax.plot(range(len(steps)), row.to_numpy(), ls, color="k" if direction == "add" else "tab:blue",
                                lw=0.5, alpha=0.25)
                    ax.plot(range(len(steps)), Y.mean().to_numpy(), ls, marker=mk, color="k" if direction == "add"
                            else "tab:blue", lw=1.8, alpha=al, label=f"{m.replace('off_', '')}: "
                            + ("block alone (adding)" if direction == "add" else "full without block (removing)"))
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(steps)), xl, fontsize=7.5)
            ax.set_title(f"{rd.SETS[st]} licks: {ttl}", fontsize=10)
            ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
        for j, (ms, ttl) in enumerate(((["pre_r"], "SHAPLEY share of pre r"),
                                        (["off_acute", "off_chronic"], "SHAPLEY share of offset (mm)"))):
            ax = axs[i, 2 + j]
            ww = 0.8 / len(ms)
            for k, m in enumerate(ms):
                s = SH[(SH["set"] == st) & (SH.measure == m)]
                xs = np.arange(len(GN)) + (k - (len(ms) - 1) / 2) * ww
                mu = [s[s.group == g].shapley.mean() for g in GN]
                ax.bar(xs, mu, width=ww * 0.9, color=[gcol[g] for g in GN], alpha=1.0 if k == 0 else 0.5,
                       hatch=None if k == 0 else "//", edgecolor="k", lw=0.3)
                for xi, g in enumerate(GN):
                    vv = s[s.group == g].shapley.to_numpy()
                    ax.plot(np.full(len(vv), xs[xi]), vv, "k.", ms=3.5, alpha=0.7)
                tot = s.groupby("animal").shapley.sum().mean()
                ax.text(0.98, 0.95 - 0.07 * k, f"{m.replace('off_', '') if 'off' in m else 'pre r'}: full = {tot:+.3f}"
                        + (" (solid)" if k == 0 and len(ms) > 1 else " (hatched)" if k else ""),
                        transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(GN)), [f"{g}\n({len(GROUPS[g]) if GROUPS[g] else 'rest'})" for g in GN], fontsize=7.5)
            ax.set_title(f"{rd.SETS[st]} licks: {ttl}", fontsize=10)
        for ax in axs[i]:
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle(f"Frozen lateral lick decoder refit on every combination of {len(GN)} disjoint area groups. LADDER: decoders built up "
                 "block by block (black) and the full decoder with the same blocks removed (blue); thin = animals.\n"
                 "SHAPLEY: each group's average gain over all orders of adding groups (shares sum exactly to the full "
                 "decoder; bars = mean, dots = animals). " + ("oro+SSs = ALM, tjM1, SSp-m, -n, SSs" if VARIANT else "oro = ALM, tjM1, SSp-m, -n") + "; bf+un = SSp-bfd, -un; MO-rest = "
                 "MOs-rest, MOp-rest; body = SSp-ul, -ll, -tr; other = ACAd, PL, RSP, VIS. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(il.OUT_W / "figures" / f"lick_block_ladder_shapley{SFX}.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd(f"lick_block_compare{SFX}"))
        return 0
    F = compute()
    save_fd(f"lick_block_compare{SFX}", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
