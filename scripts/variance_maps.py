"""TOTAL VARIANCE maps of the widefield signal, epoch vs pre (Priya 10-25: "is the increased variance post-stroke itself
significant?" -> "plot variance maps pooled and pooled per-animal normalized for delta vs pre").

Per session, per 2 x 2-px block (half-resolution grid of the per-session encoder's stored maps, main run
`movement_first_exec-polar_licks-combined`): variance = SST / n_frames of the window (`TOTAL|<window>` map = the summed
squared deviation of the held-out data from its mean over the window's frames; n frames from the session json). Data
only -- no model involved. Per animal x epoch: mean over sessions.
  POOLED              epoch - pre within animal, mean of the 4 animals (raw dF/F^2 units)
  PER-ANIMAL NORMED   log2(epoch / pre) within animal, mean of animals (each animal's own scale; 1 = doubling)
Significance (normalised version): `beta_maps.hierarchical_bootstrap_significance` on per-session log2 variance (animals
-> sessions, Bonferroni over bins), 16 and 32 full-res px (8 / 16 block px), stat mask = eroded brain mask; calibration:
random pre half-splits (share of splits with >= 1 false bin). Image left = ipsilesional.

Usage: python scripts/variance_maps.py [--windows lick cue] [--n-null 50]
Output: <E>/variance_maps/ (figures/variance_maps_<window>.png, sig.csv, null.csv, per_animal.npz)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E, MAIN  # noqa: E402
from fr_pixel_significance import FACTORS, block_mask, test  # noqa: E402

EPS = ["acute", "subacute", "chronic"]


def session_var(jf: Path, windows) -> dict | None:
    from wfield_local import config, epochs
    m = json.loads(jf.read_text())
    lab = m["label"]
    if lab[-4:] < "0801":
        return None
    z = np.load(jf.with_suffix(".npz"))
    k = {str(x): i for i, x in enumerate(z["map_keys"])}
    H, W = (int(x) for x in z["shape"])
    out = {"label": lab, "animal": config.animal_of(lab), "epoch": epochs.epoch_of(lab)}
    for w in windows:
        g = np.full(H * W, np.nan)
        g[z["flat"]] = z["maps"][k[f"TOTAL|{w}"]].astype(float) / max(m["window_frames"][w], 1)
        out[w] = g.reshape(H, W)
    return out


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed

    from combined_encoding_figures import _outline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", default=["lick", "cue"])
    ap.add_argument("--n-null", type=int, default=50)
    a = ap.parse_args(argv)
    out = E / "variance_maps"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    got = Parallel(n_jobs=8)(delayed(session_var)(jf, a.windows) for jf in sorted(MAIN.glob("PS*_*.json")))
    S = [g for g in got if g is not None]
    meta = pd.DataFrame([{k: s[k] for k in ("label", "animal", "epoch")} for s in S])
    mask = block_mask()
    ol = _outline(meta)
    print(f"{len(S)} sessions", flush=True)
    rows, nrows, save = [], [], {}
    for w in a.windows:
        by = {}
        for s in S:
            by.setdefault((s["epoch"], s["animal"]), []).append(np.where(mask, s[w], np.nan))
        A = {k: np.nanmean(v, 0) for k, v in by.items()}
        for (ep, an), M in A.items():
            save[f"{w}|{ep}|{an}"] = M.astype(np.float32)
        pre_lv = np.nanmean([A[("pre", an)] for an in AN if ("pre", an) in A], 0)
        fig, axs = plt.subplots(3, 3, figsize=(12, 11.5), squeeze=False)
        for i, ep in enumerate(EPS):
            com = [an for an in AN if (ep, an) in A and ("pre", an) in A]
            raw = np.nanmean([A[(ep, an)] - A[("pre", an)] for an in com], 0)
            nrm = np.nanmean([np.log2(A[(ep, an)] / A[("pre", an)]) for an in com], 0)
            pb = {an: [np.log2(x) for x in by[("pre", an)]] for an in com}
            qb = {an: [np.log2(x) for x in by[(ep, an)]] for an in com}
            sig = {}
            for F, f in FACTORS.items():
                sg, n = test(pb, qb, f, mask)
                k_ = 0 if sg is None else int(sg.sum())
                up = dn = 0
                if k_:
                    hh, ww = sg.shape
                    sm = np.nanmean(nrm[:hh * f, :ww * f].reshape(hh, f, ww, f), (1, 3))
                    up, dn = int((sg & (sm > 0)).sum()), int((sg & (sm < 0)).sum())
                sig[F] = (sg, up, dn, n)
                rows.append({"window": w, "epoch": ep, "px": F, "sig": k_, "up": up, "down": dn, "n_bins": n,
                             "n_animals": len(com)})
            panels = [("pre-stroke level, log10 variance" if i == 0 else None, np.log10(pre_lv), "viridis", None),
                      (f"{ep} - pre (pooled, dF/F$^2$)", raw, "RdBu_r", "sym"),
                      (f"{ep} vs pre, log2 ratio (per-animal normalised)", nrm, "RdBu_r", 1.0)]
            for j, (title, M, cmap, lim) in enumerate(panels):
                ax = axs[i, j]
                if j == 0 and i > 0:
                    ax.set_visible(False)
                    continue
                if lim is None:
                    im = ax.imshow(M, cmap=cmap, vmin=np.nanpercentile(M, 1), vmax=np.nanpercentile(M, 99), interpolation="nearest")
                else:
                    v = np.nanpercentile(np.abs(M), 99) if lim == "sym" else lim
                    im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
                if ol is not None and ol.shape == M.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                if j == 2:
                    for F, col, ls in ((16, "k", "solid"), (32, "#7a0177", "dashed")):
                        sg, f = sig[F][0], FACTORS[F]
                        if sg is not None and sg.any():
                            big = np.zeros(M.shape)
                            big[:sg.shape[0] * f, :sg.shape[1] * f] = np.kron(sg, np.ones((f, f)))
                            ax.contour(big, levels=[0.5], colors=col, linewidths=1.0, linestyles=ls)
                    title += (f"\n16px {sig[16][1]}+/{sig[16][2]}- of {sig[16][3]}; "
                              f"32px {sig[32][1]}+/{sig[32][2]}- of {sig[32][3]}")
                ax.set_title(title, fontsize=8.5)
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
        # null
        for i_ in range(a.n_null):
            rng = np.random.default_rng(2000 + i_)
            pb, qb = {}, {}
            for an in AN:
                p = by.get(("pre", an), [])
                if len(p) < 4:
                    continue
                idx = rng.permutation(len(p))
                h = len(p) // 2
                pb[an] = [np.log2(p[j]) for j in idx[:h]]
                qb[an] = [np.log2(p[j]) for j in idx[h:]]
            for F, f in FACTORS.items():
                sg = test(pb, qb, f, mask)[0]
                nrows.append({"window": w, "split": i_, "px": F, "false_bins": 0 if sg is None else int(sg.sum())})
        nr = pd.DataFrame(nrows)
        nr = nr[nr.window == w].groupby("px").false_bins.apply(lambda x: (x > 0).mean())
        fig.suptitle(f"NEURAL (data only): TOTAL VARIANCE of the widefield signal per 2x2-px block, {w} window "
                     "(SST / frames, per session; within animal, then mean of 4 animals).\nMiddle: epoch - pre, raw. "
                     "Right: log2(epoch / pre) per animal (1 = doubling);\ncontours = significant bins (black 16 px, purple "
                     f"dashed 32 px; nested bootstrap on log2 variance, Bonferroni). Null: >= 1 false bin in {nr.get(16, np.nan):.0%} "
                     f"(16 px) / {nr.get(32, np.nan):.0%} (32 px) of pre half-splits. Image left = ipsilesional.", fontsize=8.5)
        fig.subplots_adjust(left=0.02, right=0.97, top=0.87, bottom=0.02, wspace=0.12, hspace=0.25)
        fig.savefig(out / "figures" / f"variance_maps_{w}.png", dpi=115)
        plt.close(fig)
        print(w, [(r["epoch"], r["px"], r["up"], r["down"]) for r in rows if r["window"] == w], flush=True)
    pd.DataFrame(rows).to_csv(out / "sig.csv", index=False)
    N = pd.DataFrame(nrows)
    N.to_csv(out / "null.csv", index=False)
    print(N.groupby(["window", "px"]).false_bins.agg(mean="mean", any_rate=lambda x: (x > 0).mean()).round(3))
    np.savez_compressed(out / "per_animal.npz", **save)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
