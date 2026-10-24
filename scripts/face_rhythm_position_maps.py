"""Face movement maps PER SPOUT POSITION, epoch vs pre (Priya 10-24: "show the spatial speed map per position to look
at asymmetry").

`proj/` stores, per session and spout position, each tracked mesh point's RMS speed in the pre-cue (cue -2..0 s) and
response (cue 0..+4 s) windows of engaged trials (``speed_precue`` / ``speed_response``, (6, P), with bin counts
``speed_n_bins``). Per animal x epoch x position: the bin-weighted mean of speed^2 over sessions, rooted; mapped as
log2(epoch / pre) per point on the master face, and the contra / ipsi asymmetry of the change.

STATISTICS (Priya 10-24: "can we do these stats for the movement per spout position maps too"), on the MEAN figure, the
same as `face_rhythm_aligned_maps.py`: per SESSION (split videos joined) log2 speed minus the mean log2 of the animal's
pre sessions, per point; the mean map = mean over animals of each animal's epoch mean (so the map is what is tested).
  * per POINT: `analysis_kit.boot_delta_columns` (one paired animals -> sessions draw for all points); ringed when the
    95 % interval excludes 0, >= ``--min-agree`` (3) of 4 animals agree in sign, and Benjamini-Hochberg FDR < .05 over
    the face;
  * WHOLE FACE: mean log2 change over all points, mean of animals, and how many animals are below pre;
  * HALVES: contra-half minus ipsi-half change per session, `boot_delta`; * = interval excludes 0, ** = also Holm over
    the 6 positions of an epoch.

    python scripts/face_rhythm_position_maps.py [--set master_pre_v1] [--window response|precue] [--plot-only]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/position_maps/``: position_maps_<window>_<animal>.png (rows = positions,
cols = acute / subacute / chronic), position_maps_<window>_mean.png (mean of the animal maps), and
position_maps_<window>.csv (per animal x epoch x position: contra / ipsi mean log2 change, its asymmetry);
data/position_maps_<window>_points.csv (marked points per position x epoch), data/position_maps_<window>_halves.csv,
data/position_maps_<window>_cache.pkl (``--plot-only``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

POSITIONS = frc.POSITIONS                      # animal's L (ipsilesional) -> R (contralesional)
STATS_TEXT = """
MEAN figure: per SESSION log2 speed minus the mean log2 of the animal's pre sessions, mean over each animal's epoch
sessions, then over animals. Black RINGS: points whose change has a paired animals -> sessions bootstrap 95 % interval
excluding 0 (one draw for all points, `analysis_kit.boot_delta_columns`), >= 3 of 4 animals agreeing in sign, and
Benjamini-Hochberg FDR < .05 over the face. Titles: whole-face mean log2 change (animals below pre) and the
contra-half minus ipsi-half change (`boot_delta`; * = 95 % interval excludes 0, ** = also Holm over the 6 positions)."""
EPOCHS = frc.EPOCHS


def position_speed(fdir: Path, window: str) -> dict:
    """{(animal, epoch, position): (P,) RMS speed} -- bin-weighted mean of speed^2 over the epoch's sessions."""
    acc = {}
    for t in frc.sessions(fdir, "proj").itertuples():
        with np.load(t.file) as z:
            if "speed_n_bins" not in z.files:
                continue
            sp, n = z[f"speed_{window}"], z["speed_n_bins"]
            nb = n[1::2] if window == "precue" else n[2::2]
            pos = [str(x) for x in np.unique(z["position"])]                      # the stored order (`window_labels`)
        for k, p in enumerate(pos):
            if nb[k] > 0 and np.isfinite(sp[k]).all():
                s0, n0 = acc.get((t.animal, t.epoch, p), (0.0, 0))
                acc[(t.animal, t.epoch, p)] = (s0 + sp[k] ** 2 * nb[k], n0 + nb[k])
    return {k: np.sqrt(s / n) for k, (s, n) in acc.items() if n}


def session_speed(fdir: Path, window: str) -> dict:
    """{(animal, label, epoch, position): (P,) RMS speed} per SESSION (a split video's halves pooled bin-weighted)."""
    acc = {}
    for t in frc.sessions(fdir, "proj").itertuples():
        with np.load(t.file) as z:
            if "speed_n_bins" not in z.files:
                continue
            sp, n = z[f"speed_{window}"], z["speed_n_bins"]
            nb = n[1::2] if window == "precue" else n[2::2]
            pos = [str(x) for x in np.unique(z["position"])]
        for k, p in enumerate(pos):
            if nb[k] > 0 and np.isfinite(sp[k]).all():
                key = (t.animal, t.label, t.epoch, p)
                s0, n0 = acc.get(key, (0.0, 0))
                acc[key] = (s0 + sp[k] ** 2 * nb[k], n0 + nb[k])
    return {k: np.sqrt(s / n) for k, (s, n) in sorted(acc.items()) if n}


def statistics(fdir: Path, window: str, contra: np.ndarray, n_boot: int, fdr: float, min_agree: int) -> tuple:
    """(mean_maps {(epoch, position): (P,)}, marks {(epoch, position): bool (P,)}, points DataFrame, halves DataFrame)."""
    from wfield_local import analysis_kit as ak
    from wfield_local.face_rhythm import stats as frs
    V = session_speed(fdir, window)
    rng, rng_half = np.random.default_rng(0), np.random.default_rng(1)
    mean_maps, marks, prow, hrow = {}, {}, [], []
    for p in POSITIONS:
        keys = [k for k in V if k[3] == p]
        lg = {k: np.log2(np.maximum(V[k], 1e-9)) for k in keys}
        rel = {}
        for an in sorted({k[0] for k in keys}):
            pre = [lg[k] for k in keys if k[0] == an and k[2] == "pre"]
            if not pre:
                continue
            base = np.mean(pre, 0)
            for k in keys:
                if k[0] == an:
                    rel[k] = lg[k] - base
        by = lambda ep, rel=rel: {an: np.array([rel[k] for k in sorted(rel) if k[0] == an and k[2] == ep])  # noqa: E731
                         for an in sorted({k[0] for k in rel}) if any(k[0] == an and k[2] == ep for k in rel)}
        pre_d = by("pre")
        for ep in EPOCHS[1:]:
            post_d = by(ep)
            if not post_d:
                continue
            per = np.array([post_d[an].mean(0) for an in sorted(post_d)])          # (animals, P)
            mean_maps[(ep, p)] = per.mean(0)
            res = ak.boot_delta_columns(post_d, pre_d, rng, n_boot)
            if res is not None:
                obs, lo, hi, o, _n = res
                agree = np.where(obs < 0, (per < 0).sum(0), (per > 0).sum(0))
                sig = ((lo > 0) | (hi < 0)) & (agree >= min_agree) & ak.bh(ak.boot_p_columns(o), fdr)
                marks[(ep, p)] = sig
                prow.append({"position": p, "epoch": ep, "marked": int(sig.sum()), "marked_contra": int((sig & contra).sum()),
                             "marked_ipsi": int((sig & ~contra).sum()), "marked_down": int((sig & (obs < 0)).sum()),
                             "marked_up": int((sig & (obs > 0)).sum())})
            whole = per.mean(1)
            h_post = {an: list(v[:, contra].mean(1) - v[:, ~contra].mean(1)) for an, v in post_d.items()}
            h_pre = {an: list(v[:, contra].mean(1) - v[:, ~contra].mean(1)) for an, v in pre_d.items()}
            iv, draws = ak.boot_delta(h_post, h_pre, rng_half, draws=True)
            hrow.append({"position": p, "epoch": ep, **{an: float(np.mean(v)) for an, v in sorted(h_post.items())},
                         "contra_minus_ipsi": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                         "ci_hi": iv.hi if iv else np.nan, "excludes_0": bool(iv and (iv.lo > 0 or iv.hi < 0)),
                         "p_boot": ak.boot_p(draws) if iv else np.nan,
                         "n_contra_lower": sum(np.mean(v) < 0 for v in h_post.values()),
                         "whole_face_log2": float(whole.mean()), "n_whole_lower": int((whole < 0).sum()),
                         "n_animals": len(per)})
    H = frs.add_holm(pd.DataFrame(hrow), ["epoch"])
    return mean_maps, marks, pd.DataFrame(prow), H


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--window", default="response", choices=("response", "precue"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--min-agree", type=int, default=3, help="animals agreeing in sign (Priya 10-24: 3 of 4)")
    ap.add_argument("--plot-only", action="store_true", help="redraw the mean figure's statistics from the cache")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "position_maps"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    M = np.load(fdir / "master.npz")
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    contra = home[:, 0] <= mid                                       # cam4: image LEFT = contralesional
    V = position_speed(fdir, a.window)
    animals = sorted({k[0] for k in V})
    rows, maps = [], {}
    for an in animals:
        for p in POSITIONS:
            pre = V.get((an, "pre", p))
            if pre is None:
                continue
            for ep in EPOCHS[1:]:
                v = V.get((an, ep, p))
                if v is None:
                    continue
                m = np.log2(np.maximum(v, 1e-9) / np.maximum(pre, 1e-9))
                maps[(an, ep, p)] = m
                c, i = float(m[contra].mean()), float(m[~contra].mean())
                rows.append({"animal": an, "epoch": ep, "position": p, "log2_change_contra": c,
                             "log2_change_ipsi": i, "contra_minus_ipsi": c - i,
                             "speed_contra_vs_pre": float(v[contra].mean() / pre[contra].mean()),
                             "speed_ipsi_vs_pre": float(v[~contra].mean() / pre[~contra].mean())})
    T = pd.DataFrame(rows)
    T.to_csv(figure_layout.sidecar_for(out, f"position_maps_{a.window}", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(f"{a.window}: contra minus ipsi of the log2(epoch / pre) speed change, per animal (rows = position)")
    print(T.pivot_table(index=["animal", "position"], columns="epoch", values="contra_minus_ipsi", observed=True)
          .reindex(columns=list(EPOCHS[1:])).round(2).to_string())

    cache = figure_layout.sidecar_for(out, f"position_maps_{a.window}_cache", ".pkl")
    if a.plot_only:
        mean_maps, marks, PT, H = pd.read_pickle(cache)
    else:
        mean_maps, marks, PT, H = statistics(fdir, a.window, contra, a.n_boot, a.fdr, a.min_agree)
        pd.to_pickle((mean_maps, marks, PT, H), cache)
    PT.to_csv(figure_layout.sidecar_for(out, f"position_maps_{a.window}_points", ".csv"), index=False)
    H.to_csv(figure_layout.sidecar_for(out, f"position_maps_{a.window}_halves", ".csv"), index=False, float_format="%.4g")
    an_cols = [c for c in H.columns if c.startswith("PS")]
    print("\nHALVES (session level): contra - ipsi log2 change per animal FIRST (rule 8), then boot_delta + Holm")
    print(H[["position", "epoch", *an_cols, "contra_minus_ipsi", "ci_lo", "ci_hi", "p_holm", "mark", "whole_face_log2",
             "n_whole_lower"]].round(3).to_string(index=False))

    # what is mapped: the SPEED of each face point's movement (px / s, RMS) in the window -- not reaction time
    what = ("movement speed of each face point, cue 0 to +4 s" if a.window == "response"
            else "movement speed of each face point, cue -2 to 0 s")

    def draw(get, title, fn, stats=False):
        with frf.style():
            fig, axs = frf.subplots(len(POSITIONS), 3, figsize=(10.5, 3.1 * len(POSITIONS)), squeeze=False)
            sc = None
            for r_i, p in enumerate(frc.POSITIONS_DISPLAY):
                for c_i, ep in enumerate(EPOCHS[1:]):
                    ax = axs[r_i, c_i]
                    ax.axis("off")
                    m = get(ep, p)
                    if m is None:
                        continue
                    sc = ax.scatter(home[:, 0], home[:, 1], c=m, s=4, cmap="RdBu_r", vmin=-1, vmax=1)
                    ax.axvline(mid, color="k", lw=0.4)
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
                    title_ = f"{p} -- {ep}"
                    if stats:
                        sg = marks.get((ep, p))
                        if sg is not None:
                            ax.scatter(home[sg, 0], home[sg, 1], s=7, facecolors="none", edgecolors="k", linewidths=0.35)
                        h = H[(H.position == p) & (H.epoch == ep)]
                        if len(h):
                            h = h.iloc[0]
                            title_ += (f"\nwhole face {h.whole_face_log2:+.2f}, {int(h.n_whole_lower)}/{int(h.n_animals)} "
                                       f"lower\ncontra - ipsi {h.contra_minus_ipsi:+.2f}{h['mark']}")
                    ax.set_title(title_, fontsize=7 if stats else 9)
            if sc is not None:
                fig.colorbar(sc, ax=axs, shrink=0.3, label="log2(epoch / pre)")
            fig.suptitle(title.replace(", per spout", ",\nper spout"), fontsize=11)   # long: two lines
            frf.save(fig, out / fn, f"""
Per mesh point on the master face (image LEFT = CONTRALESIONAL on cam4, all lesions left; vertical line = the face
midline, x = {mid:.0f}): log2(epoch / pre) of the point's RMS speed in the {a.window.upper()} window
({'cue -2..0 s' if a.window == 'precue' else 'cue 0..+4 s'}) of engaged trials at ONE spout position (rows; the animal's
L = ipsilesional, R = contralesional spouts), sessions pooled bin-weighted within the epoch. Red = more movement than
pre, blue = less; clipped at +-1 (2x). Contra / ipsi means of the change: data/position_maps_{a.window}.csv.{STATS_TEXT if stats else ""}""")

    for an in animals:
        draw(lambda ep, p, an=an: maps.get((an, ep, p)), f"{fdir.name}: {an}, {what}, per spout position vs pre",
             f"position_maps_{a.window}_{an}.png")

    draw(lambda ep, p: mean_maps.get((ep, p)), f"{fdir.name}: mean of {len(animals)} animals, {what}, per spout "
         "position vs pre (rings = consistent points)", f"position_maps_{a.window}_mean.png", stats=True)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
