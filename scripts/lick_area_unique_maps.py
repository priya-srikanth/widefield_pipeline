"""Musall-style ALONE / UNIQUE maps for the frozen lateral lick DECODER (Priya 10-28: "I meant full decoder vs full decoder
without orofacial = orofacial unique").

Same features, z-scoring, ridge and leave-one-pre-session-out as the KEY frozen decoder (`lick_refit_decoders`; the fitting
code of `lick_area_importance`, 11.12). For each area (left + right together; ALM / tjM1 / MOs_rest / MOp_rest split;
visual and retrosplenial areas kept SEPARATE here) and for blocks (orofacial = ALM + tjM1 + SSp-m + SSp-n; orofacial+MO =
also MOs_rest + MOp_rest; non-orofacial = every other area), per animal x lick set, two refits:
  ALONE    decoder trained on that group only
  UNIQUE   full decoder minus the decoder WITHOUT that group
Measures: pre held-out r (leave-one-pre-session-out) and the like-for-like offset change vs pre (Gaussian sigma 0.1 mm,
11.12's rule) in the acute and chronic epochs. UNIQUE r = full - dropped (> 0 = information only this group carries);
UNIQUE offset = full offset - dropped offset (how much of the offset needs this group). ALONE offsets sit on that group's
own (often weak) read-out, so read them with the group's alone r. Mean of animals; maps paint each area's value on both
hemispheres. Image left = ipsilesional.
Outputs <OUT_W>/area_unique_{sessions,summary}.csv; figures lick_area_unique_maps_{iti,cued}.png (figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

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

ORO = ("ALM", "tjM1", "SSp-m", "SSp-n")
ORO_MO = (*ORO, "MOs_rest", "MOp_rest")
BLOCKS = ["orofacial", "orofacial+MO", "non-orofacial"]


def groups(names: list[str]) -> dict:
    base = [n.rsplit("_", 1)[0] for n in names]
    g = {}
    for i, b in enumerate(base):
        g.setdefault(b, []).append(i)
    g["orofacial"] = [i for i, b in enumerate(base) if b in ORO]
    g["orofacial+MO"] = [i for i, b in enumerate(base) if b in ORO_MO]
    g["non-orofacial"] = [i for i, b in enumerate(base) if b not in ORO_MO]
    return g


def compute() -> dict:
    names, _sd = wd.area_meta()
    names = list(names)
    G = groups(names)
    D = rd.load_z()
    rows = []
    allidx = np.arange(len(names))
    for st in rd.SETS:
        for a in lai.AN:
            S = {x: (d["epoch"], *d[st]) for x, d in sorted(D.items()) if d["animal"] == a and len(d[st][1]) >= rd.MIN_LICKS}
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            specs = [("full", "(full)", allidx)]
            for gname, idx in G.items():
                specs.append(("alone", gname, np.asarray(idx)))
                specs.append(("dropped", gname, np.setdiff1d(allidx, idx)))
            for mode, gname, keep in specs:
                if not len(keep):
                    continue
                fits = {h: rd._fit(np.vstack([S[q][1][:, keep] for q in pre if q != h]),
                                   np.vstack([S[q][2][:, :1] for q in pre if q != h])) for h in [*pre, None]}
                dec = {x: rd._pred(fits[x if v[0] == "pre" else None], v[1][:, keep])[:, 0] for x, v in S.items()}
                for x, (ep, _X, Y) in S.items():
                    act = Y[:, 0]
                    ref = [q for q in pre if q != x]
                    off = lai._offset(np.concatenate([S[q][2][:, 0] for q in ref]), np.concatenate([dec[q] for q in ref]),
                                      act, dec[x])
                    rows.append(dict(set=st, animal=a, label=x, epoch=ep, mode=mode, group=gname, n_feat=len(keep),
                                     r=float(np.corrcoef(dec[x], act)[0, 1]), offset=float(np.nanmean(off))))
            print(f"[{st} {a}] {len(specs)} fits done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT_W / "area_unique_sessions.csv", index=False)
    A = R.groupby(["set", "mode", "group", "animal", "epoch"])[["r", "offset"]].mean().reset_index()
    # per animal: pre r, offset change vs pre per epoch
    recs = []
    for (st, mode, g, an), t in A.groupby(["set", "mode", "group", "animal"]):
        t = t.set_index("epoch")
        if "pre" not in t.index:
            continue
        rec = {"set": st, "mode": mode, "group": g, "animal": an, "pre_r": t.loc["pre", "r"]}
        for ep in ("acute", "chronic"):
            rec[f"off_{ep}"] = t.loc[ep, "offset"] - t.loc["pre", "offset"] if ep in t.index else np.nan
        recs.append(rec)
    P = pd.DataFrame(recs)
    full = P[P["mode"] == "full"].set_index(["set", "animal"])
    out = []
    for _i, r in P[P["mode"] != "full"].iterrows():
        f = full.loc[(r["set"], r.animal)]
        d = {"set": r["set"], "group": r.group, "animal": r.animal}
        if r["mode"] == "alone":
            d.update(alone_pre_r=r.pre_r, alone_off_acute=r.off_acute, alone_off_chronic=r.off_chronic)
        else:
            d.update(unique_pre_r=f.pre_r - r.pre_r, unique_off_acute=f.off_acute - r.off_acute,
                     unique_off_chronic=f.off_chronic - r.off_chronic)
        out.append(d)
    Q = pd.DataFrame(out).groupby(["set", "group", "animal"]).first().reset_index()
    Q.to_csv(il.OUT_W / "area_unique_summary.csv", index=False)
    pd.set_option("display.width", 230)
    print("== full decoder per animal: pre r / offset acute / chronic\n" + full[["pre_r", "off_acute", "off_chronic"]]
          .round(3).to_string())
    print("== per animal (rule 8), blocks\n" + Q[Q.group.isin(BLOCKS)].set_index(["set", "group", "animal"]).round(3).to_string())
    M = Q.groupby(["set", "group"]).mean(numeric_only=True).reset_index()
    print("== mean of animals, top areas by unique pre r\n" + M.sort_values(["set", "unique_pre_r"], ascending=[True, False])
          .groupby("set").head(8).round(3).to_string(index=False))
    from iti_area_maps import split_image
    img, nm = split_image(il.labels()[0])
    return {"M": M, "Q": Q, "img": img, "nm": nm, "full": full.reset_index()}


PANELS = [("alone_pre_r", "ALONE: pre held-out r"), ("alone_off_acute", "ALONE: acute offset change (mm)"),
          ("alone_off_chronic", "ALONE: chronic offset change (mm)"),
          ("unique_pre_r", "UNIQUE: full - without (pre r)"), ("unique_off_acute", "UNIQUE: acute offset, full - without (mm)"),
          ("unique_off_chronic", "UNIQUE: chronic offset, full - without (mm)")]


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline, paint

    M, img, nm = F["M"], F["img"], F["nm"]
    ol = outline(img)
    for st in rd.SETS:
        m = M[(M["set"] == st) & ~M.group.isin(BLOCKS)].set_index("group")
        b = M[(M["set"] == st) & M.group.isin(BLOCKS)].set_index("group").reindex(BLOCKS)
        fig = plt.figure(figsize=(19, 10.5))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.9])
        for k, (c, ttl) in enumerate(PANELS):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            vals = {f"{g}_{h}": float(v) for g, v in m[c].items() for h in ("left", "right")}
            if "pre_r" in c:
                lim = max(np.nanmax(np.abs(m[c])), 1e-3)
                im = ax.imshow(paint(img, nm, vals), cmap="magma" if c.startswith("alone") else "RdBu_r",
                               vmin=0 if c.startswith("alone") else -lim, vmax=lim, interpolation="nearest")
            else:
                lim = max(np.nanpercentile(np.abs(m[c]), 98), 1e-3)
                im = ax.imshow(paint(img, nm, vals), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.set_title(ttl, fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        for k, (cols, ttl) in enumerate(((["alone_pre_r", "unique_pre_r"], "pre held-out r"),
                                         (["unique_off_acute", "unique_off_chronic"], "UNIQUE offset (mm)"))):
            ax = fig.add_subplot(gs[k, 3])
            x = np.arange(len(BLOCKS))
            for j, c in enumerate(cols):
                ax.bar(x + (j - 0.5) * 0.38, b[c], width=0.36, label=c.replace("_", " "))
                q = F["Q"][(F["Q"]["set"] == st) & F["Q"].group.isin(BLOCKS)]
                for xi, g in enumerate(BLOCKS):
                    ax.plot(np.full(4, xi + (j - 0.5) * 0.38), q[q.group == g][c], "k.", ms=4)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x, BLOCKS, fontsize=8)
            ax.set_title(f"BLOCKS: {ttl}\n(bars = mean, dots = animals)", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        fu = F["full"][F["full"]["set"] == st]
        fig.suptitle(f"{'ITI' if st == 'iti' else 'CUED'} licks: Musall-style ALONE / UNIQUE maps of the frozen lateral lick "
                     f"decoder (full decoder: pre r {fu.pre_r.mean():.2f}, offset acute {fu.off_acute.mean():+.2f} / chronic "
                     f"{fu.off_chronic.mean():+.2f} mm).\nALONE = decoder trained on that area only; UNIQUE = full minus the "
                     "decoder without that area (offset: how much of the offset needs it). Each area's value on both "
                     "hemispheres; ALM / tjM1 split; image left = ipsilesional. Mean of animals. PROVISIONAL", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_png_svg(fig, il.OUT_W / "figures" / f"lick_area_unique_maps_{'iti' if st == 'iti' else 'cued'}.png", dpi=95)
        plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_area_unique_maps"))
        return 0
    F = compute()
    save_fd("lick_area_unique_maps", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
