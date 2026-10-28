"""DROP-ONE-AREA importance of the frozen lateral lick decoder (Priya 10-28: "surprised ALM / tjM1 are weighted so little"
-> "yes" to the importance test).

Same features, z-scoring, ridge and pre-session leave-one-out as the KEY frozen decoder (`lick_refit_decoders`: Allen
areas with the ALM / tjM1 split, lags 0..0.2 s, each area / its session frame SD). For each area GROUP (left + right
together; VIS = all visual areas, RSP = all retrosplenial) the decoder is refit WITHOUT that group, per animal x lick set:
  pre r          held-out (leave-one-pre-session-out) r of decoded vs actual lateral position; importance = full - dropped
  post r / pre   frozen decoder r in each post epoch divided by the pre r of the SAME (dropped) decoder
  offset         like-for-like offset (decoded minus pre licks that landed at the same place, Gaussian sigma 0.1 mm, pre
                 scored leave-one-out), change vs pre, mean of animals
Weights are not importance: correlated areas share them (ridge), so a low weight can hide information another area
duplicates; dropping an area measures what it adds UNIQUELY. Per-animal table first (rule 8).
Outputs <OUT_W>/area_importance_{sessions,summary}.csv, figures/lick_area_importance.png; figdata (--plot-only).
Usage: python scripts/lick_area_importance.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
import lick_refit_decoders as rd  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
SIGMA = 0.1


COMBOS = {"ALM+tjM1": ("ALM", "tjM1"), "SSp-m+n": ("SSp-m", "SSp-n"), "ALM+tjM1+SSp-m+n": ("ALM", "tjM1", "SSp-m", "SSp-n"),
          "MOs_rest": ("MOs_rest",), "MOp_rest": ("MOp_rest",), "VIS": ("VIS",), "RSP": ("RSP",)}   # + per side (10-28)


def _fam(base: str) -> str:
    return "VIS" if base.startswith("VIS") else "RSP" if base.startswith("RSP") else base


def groups(names: list[str]) -> dict:
    """Single areas (left + right together; VIS / RSP pooled) + Priya's 10-28 combinations, each LEFT (ipsilesional),
    RIGHT (contralesional) and BILATERAL: ALM+tjM1, SSp-m+n, ALM+tjM1+SSp-m+n."""
    g = {}
    for i, n in enumerate(names):
        base = n.rsplit("_", 1)[0]
        base = "VIS" if base.startswith("VIS") else "RSP" if base.startswith("RSP") else base
        g.setdefault(base, []).append(i)
    for cname, parts in COMBOS.items():
        for side, sides in (("ipsi (left)", ("left",)), ("contra (right)", ("right",)), ("bilateral", ("left", "right"))):
            g[f"{cname} {side}"] = [i for i, n in enumerate(names) if _fam(n.rsplit("_", 1)[0]) in parts
                                    and n.rsplit("_", 1)[1] in sides]
    return {"(none)": [], **g}


def _offset(act_pre, dec_pre, act, dec) -> np.ndarray:
    """dec minus the Gaussian-weighted mean of pre decoded at the same landing (weight >= 3 licks, else NaN)."""
    w = np.exp(-0.5 * ((act[:, None] - act_pre[None, :]) / SIGMA) ** 2)
    sw = w.sum(1)
    return np.where(sw >= 3, dec - (w @ dec_pre) / np.maximum(sw, 1e-12), np.nan)


def compute() -> dict:
    names, _sd = wd.area_meta()
    G = groups(list(names))
    D = rd.load_z()
    rows = []
    for st in rd.SETS:
        for a in AN:
            S = {x: (d["epoch"], *d[st]) for x, d in sorted(D.items()) if d["animal"] == a and len(d[st][1]) >= rd.MIN_LICKS}
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            for gname, drop in G.items():
                keep = np.setdiff1d(np.arange(len(names)), drop)
                fits = {h: rd._fit(np.vstack([S[q][1][:, keep] for q in pre if q != h]),
                                   np.vstack([S[q][2][:, :1] for q in pre if q != h])) for h in [*pre, None]}
                dec = {x: rd._pred(fits[x if v[0] == "pre" else None], v[1][:, keep])[:, 0] for x, v in S.items()}
                for x, (ep, _X, Y) in S.items():
                    act = Y[:, 0]
                    ref = [q for q in pre if q != x]
                    off = _offset(np.concatenate([S[q][2][:, 0] for q in ref]), np.concatenate([dec[q] for q in ref]),
                                  act, dec[x])
                    rows.append(dict(set=st, animal=a, label=x, epoch=ep, group=gname, n_feat=len(keep),
                                     r=float(np.corrcoef(dec[x], act)[0, 1]), offset=float(np.nanmean(off))))
            print(f"[{st} {a}] {len(G)} groups done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT_W / "area_importance_sessions.csv", index=False)
    # per animal x epoch, then change vs pre; importance = full minus dropped
    A = R.groupby(["set", "group", "animal", "epoch"])[["r", "offset"]].mean().reset_index()
    out = []
    for (st, g), t in A.groupby(["set", "group"]):
        P = t.pivot_table(index="animal", columns="epoch", values=["r", "offset"])
        rec = {"set": st, "group": g, "pre_r": P["r"]["pre"].mean()}
        for ep in EP[1:]:
            if ep in P["r"]:
                rec[f"r_ratio_{ep}"] = (P["r"][ep] / P["r"]["pre"]).mean()
                rec[f"offset_{ep}"] = (P["offset"][ep] - P["offset"]["pre"]).mean()
        out.append(rec)
    T = pd.DataFrame(out)
    full = T[T.group == "(none)"].set_index("set")
    for c in [c for c in T.columns if c not in ("set", "group")]:
        T[f"d_{c}"] = T[c] - T.set.map(full[c])
    T.to_csv(il.OUT_W / "area_importance_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("== importance (dropped minus full; pre r drop = information the group adds uniquely)")
    print(T.sort_values(["set", "d_pre_r"])[["set", "group", "pre_r", "d_pre_r", "d_r_ratio_chronic", "d_offset_acute",
                                              "d_offset_chronic"]].round(3).to_string(index=False))
    # per animal (rule 8): pre r drop for the motor groups
    Pa = A[A.epoch == "pre"].pivot_table(index=["set", "animal"], columns="group", values="r")
    print("== per animal pre r, full and motor groups dropped\n"
          + Pa[[c for c in ("(none)", "ALM", "tjM1", "MOs_rest", "MOp_rest", "SSp-m", "SSp-n") if c in Pa]].round(3)
          .to_string())
    return {"T": T}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = F["T"][F["T"].group != "(none)"]
    cols = [("d_pre_r", "pre held-out r: dropped - full\n(more negative = more unique information)"),
            ("d_r_ratio_chronic", "chronic r / pre: dropped - full"),
            ("d_offset_acute", "acute like-for-like offset change:\ndropped - full (mm)"),
            ("d_offset_chronic", "chronic like-for-like offset change:\ndropped - full (mm)")]
    fig, axs = plt.subplots(2, len(cols), figsize=(21, 17), squeeze=False)
    for i, st in enumerate(rd.SETS):
        t = T[T.set == st].sort_values("d_pre_r")
        y = np.arange(len(t))
        for j, (c, lab) in enumerate(cols):
            ax = axs[i, j]
            col = ["#d95f02" if g.startswith(("ALM", "tjM1")) else "#1b9e77" if "+" in g else "#7570b3" if g in ("MOs_rest", "MOp_rest", "SSp-m", "SSp-n")
                   else "0.6" for g in t.group]
            ax.barh(y, t[c], color=col)
            ax.set_yticks(y, t.group, fontsize=8)
            ax.axvline(0, color="k", lw=0.7)
            ax.set_title(f"{'ITI' if st == 'iti' else 'cued'}: {lab}", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("DROP-ONE-AREA importance of the frozen lateral lick decoder (Allen areas with the ALM / tjM1 split; each group "
                 "= left + right; refit without it, same pre leave-one-session-out). Orange = ALM / tjM1 (incl. combinations), green = other combinations (each side and bilateral), purple = other "
                 "orofacial motor / sensory.\nMean of animals. Weights are not importance: correlated areas share ridge "
                 "weight, dropping an area measures what it adds uniquely. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(il.OUT_W / "figures" / "lick_area_importance.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_area_importance"))
        return 0
    F = compute()
    save_fd("lick_area_importance", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
