"""MOVEMENT-NULL PLAN ENCODING MAPS, Musall-style (Priya 10-30 approved; FINDINGS 0b item (5) "movement-null plan encoding
model DEFERRED"; RUNNING_LIST_2026-10-29 sec 1). Question: after every movement regressor is in the model, how much
variance does the SPOUT / PLAN term still explain, where on cortex, and does that change after the stroke?

NO NEW MODEL, NO NEW REGRESSORS, NO REFIT. The frozen pre-stroke encoder already holds the requested nested groups as
groups of one joint ridge fit (`wfield_local/frozen_encoder.py` FGROUP, grouping v3):
  spout / task            timing (cue, arrival) + TARGET (the six position blocks, pre- and post-cue)
  non-directional mvmt    LICK (deviation of lick onsets / tongue speed / jaw opening + speed from the spout template)
                          + OTHER (nose x / y, running, push)
  directional face        EXEC_WITHIN direction traces (tongue x / y / protrusion, jaw x: deviation from the spout's
                          pre-stroke path) + EXEC_LOCKED (the spout-locked template of every lick column) + in the FR-A
                          frame every FaceRhythm factor (incl. side / distance factors)
  upcoming deviation      EXEC_WITHIN exec_angle: the per-lick executed within-spout angle, a lick-locked kernel with
                          lags -0.5 .. +2 s, so the 0.5 s BEFORE each lick is covered
  lick history            NOT a regressor of the frozen encoder (no previous-trial / previous-lick term exists); NOT
                          added here (no invented regressors) -- stated as a limit
Musall UNIQUE dR2 of the target = (SSE[no_target] - SSE[joint]) / SST = the variance the target explains when it is
entered LAST, after all movement groups (the strictest ordering: shared target / movement variance goes to movement).
`spout` unique (target + exec_locked + exec_within dropped together) is shown beside it as the less strict term (it
includes the executed spout-locked and within-spout movement).

DATA: the FROZEN_PIX_MUSALL basis twins already on the share (`frozen_encoder_<frame>_basis_pixmusall`, 107 sessions;
target-space second moments of residual E and data S per variant x window, measure "frame" = every frame of every trial
in the MUSALL window: cue 0 .. 1.5 s after the cue, lick -0.1 .. 1.0 s around the first lick). Pixel value =
diag(W E W') / diag(W S W'), W = the animal's frozen spatial basis (`<frame>_basis/fit/spatial_<an>.pkl`), i.e. R2 of
the basis-projected pixel. Pre sessions are held out (leave-one-pre-session-out), post sessions use the all-pre betas.
Frames: FR-A (`..._frA`, FaceRhythm in the model) = PRIMARY (directional face included); noFR = sensitivity.
Display: per animal x epoch SSE and SST summed over sessions before the ratio, then the mean of animals (as
frozen_pixel_musall). Test (epoch - pre): per-session maps, nested bootstrap animals -> sessions on 16-px bins,
Bonferroni (`fr_pixel_significance.test`). Image LEFT = ipsilesional.

AREA DOTS: per-session unique R2 from `frozen_fr_compare/musall_sessions_v2.csv` (scripts/frozen_fr_musall_v2.py: Allen
areas from the area run, ALM / tjM1 from the basis twin); levels = animal means (dots) + mean of animals; changes from
musall_change_v2.csv (boot_delta, nested). Rule-8 per-animal table printed first.

Usage: python scripts/movement_null_plan_maps.py [--frames FR-A noFR] [--jobs 4] [--plot-only]
Output: <E>/movement_null_plan/{figdata_<frame>.npz, maps_table_<frame>.csv, maps_sig_<frame>.csv, area_levels.csv,
        figures/movement_null_maps_<frame>_<window>.png, figures/movement_null_area_dots_<frame>.png}
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
from wfield_local.frozen_encoder import design_tag as _design_tag  # frame tag of the FROZEN_LICKS_ALL design (8.51)

import argparse
import json
import pickle
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

FRAMES = {"FR-A": _design_tag("frozen_encoder_gate-attempted_v7g3_t3d_frA"), "noFR": _design_tag("frozen_encoder_gate-attempted_v7g3_t3d")}
QS = [("target", "no_target", "TARGET unique (plan after all movement)"),
      ("spout", "no_spout", "SPOUT unique (target + executed spout part)")]
WINDOWS = ("cue", "lick")
EP4 = ("pre", "acute", "subacute", "chronic")
EPOCH_COL = {"pre": "k", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
OUT = E / "movement_null_plan"
#: area-dot set: key areas + controls (VISp has no ipsilesional area on the stat mask)
DOT_AREAS = [("SSp-n", 2), ("SSp-m", 2), ("ALM", 2), ("tjM1", 2), ("MOp", 2), ("SSs", 2), ("VISp", 1), ("RSPd", 2)]


@lru_cache(maxsize=8)
def _spatial(fit_dir: str, animal: str) -> dict:
    return pickle.loads((Path(fit_dir) / f"spatial_{animal}.pkl").read_bytes())


def session_job(item):
    """One session -> {"label", "animal", "epoch", (window, "S"|variant): stat-pixel vector}. Module-level for spawn."""
    from frozen_r2_maps import pixel_sse
    label, pix_run, fit_dir, variants = item
    js = Path(pix_run) / "joint" / "sessions" / f"{label}.json"
    m = json.loads(js.read_text(encoding="utf-8"))
    sp = _spatial(fit_dir, m["animal"])
    keep = np.array([t != "cortex" for t in m["targets"]])
    out = {"label": label, "animal": m["animal"], "epoch": m["epoch"]}
    for v in variants:
        z = np.load(Path(pix_run) / v / "sessions" / f"{label}.npz")
        for w in WINDOWS:
            if v == "joint":
                out[(w, "S")] = pixel_sse(sp, z[f"mus_{w}_frame_S"], keep).astype(np.float32)
            out[(w, v)] = pixel_sse(sp, z[f"mus_{w}_frame_E"], keep).astype(np.float32)
    return out


def compute(frame: str, jobs: int) -> Path:
    from fr_pixel_significance import FACTORS, block_mask, test

    from wfield_local import config
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    run = E / f"{FRAMES[frame]}_basis_pixmusall"
    fit_dir = E / f"{FRAMES[frame]}_basis" / "fit"
    labels = sorted(js.stem for js in (run / "joint" / "sessions").glob("*.json") if not config.early_pre(js.stem))
    variants = ("joint", *[v for _q, v, _l in QS])
    print(f"[{frame}] {len(labels)} sessions from {run}", flush=True)
    res, fail = ak.fan_sessions([(x, str(run), str(fit_dir), variants) for x in labels], session_job, jobs=jobs,
                                label="pixmoments")
    if fail:
        raise RuntimeError(f"failed: {fail}")
    S = [r for _it, r in res]                                       # sorted by label (fan_sessions)
    _V, _ft, _Ub, flat, _pa, _nm, shape, pstat = ce.load_imaging(labels[0])
    H, W = int(shape[0]), int(shape[1])
    pix = flat[pstat]
    mask = block_mask()

    def paint(vec):
        g = np.full(H * W, np.nan)
        g[pix] = vec
        return np.where(mask, g.reshape(H, W)[::2, ::2], np.nan)

    meta = pd.DataFrame([{k: s[k] for k in ("label", "animal", "epoch")} for s in S])
    sess_maps, pooled, rows = {}, {}, []
    for w in WINDOWS:
        for q, v, _l in QS:
            for s in S:
                sst = np.where(s[(w, "S")] > 0, s[(w, "S")].astype(np.float64), np.nan)
                sess_maps[(s["label"], w, q)] = paint((s[(w, v)] - s[(w, "joint")]) / sst).astype(np.float32)
            for (an, ep), g in meta.groupby(["animal", "epoch"]):
                ss = [s for s in S if s["label"] in set(g.label)]
                sst = sum(s[(w, "S")].astype(np.float64) for s in ss)
                val = (sum(s[(w, v)].astype(np.float64) for s in ss) - sum(s[(w, "joint")].astype(np.float64)
                                                                          for s in ss)) / np.where(sst > 0, sst, np.nan)
                pooled[(an, ep, w, q)] = paint(val)
                rows.append({"frame": frame, "window": w, "q": q, "animal": an, "epoch": ep, "n_sessions": len(ss),
                             "median_px": float(np.nanmedian(val)), "p95_px": float(np.nanpercentile(val, 95)),
                             "mean_px": float(np.nanmean(val))})
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / f"maps_table_{frame}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print("== per animal (rule 8): pixel unique dR2, 95th percentile over brain pixels\n"
              + T.pivot_table(index=["window", "q", "animal"], columns="epoch", values="p95_px")
              .reindex(columns=list(EP4)).round(4).to_string(), flush=True)
    animals = sorted(meta.animal.unique())
    lev, chg, sig, srows = {}, {}, {}, []
    for w in WINDOWS:
        for q, _v, _l in QS:
            for ep in EP4:
                got = [pooled[(an, ep, w, q)] for an in animals if (an, ep, w, q) in pooled]
                lev[(w, q, ep)] = np.nanmean(got, 0)
            pre_by = {}
            for _, r in meta[meta.epoch == "pre"].iterrows():
                pre_by.setdefault(r.animal, []).append(sess_maps[(r.label, w, q)].astype(np.float64))
            for ep in EP4[1:]:
                com = [an for an in animals if (an, "pre", w, q) in pooled and (an, ep, w, q) in pooled]
                chg[(w, q, ep)] = np.nanmean([pooled[(an, ep, w, q)] - pooled[(an, "pre", w, q)] for an in com], 0)
                post_by = {}
                for _, r in meta[meta.epoch == ep].iterrows():
                    post_by.setdefault(r.animal, []).append(sess_maps[(r.label, w, q)].astype(np.float64))
                sg, n = test({a: pre_by[a] for a in com}, {a: post_by[a] for a in com}, FACTORS[16], mask)
                sig[(w, q, ep)] = sg
                f = FACTORS[16]
                up = dn = 0
                if sg is not None and sg.any():
                    hh, ww = sg.shape
                    sm = np.nanmean(chg[(w, q, ep)][:hh * f, :ww * f].reshape(hh, f, ww, f), (1, 3))
                    up, dn = int((sg & (sm > 0)).sum()), int((sg & (sm < 0)).sum())
                srows.append({"frame": frame, "window": w, "q": q, "epoch": ep, "bins_up": up, "bins_down": dn,
                              "n_bins": n, "animals": "/".join(com)})
                print(f"  {frame} {w} {q} {ep}: sig bins up {up} / down {dn} of {n}", flush=True)
    pd.DataFrame(srows).to_csv(OUT / f"maps_sig_{frame}.csv", index=False)
    fd = OUT / f"figdata_{frame}.npz"
    pay = {"labels": np.array(meta.label, dtype=str), "animals_of": np.array(meta.animal, dtype=str),
           "epochs_of": np.array(meta.epoch, dtype=str)}
    for (w, q, ep), M in lev.items():
        pay[f"lev|{w}|{q}|{ep}"] = M.astype(np.float32)
    for (w, q, ep), M in chg.items():
        pay[f"chg|{w}|{q}|{ep}"] = M.astype(np.float32)
        if sig[(w, q, ep)] is not None:
            pay[f"sig|{w}|{q}|{ep}"] = sig[(w, q, ep)]
    for (an, ep, w, q), M in pooled.items():
        pay[f"animal|{an}|{ep}|{w}|{q}"] = M.astype(np.float32)
    for (lab, w, q), M in sess_maps.items():
        pay[f"session|{lab}|{w}|{q}"] = M
    np.savez_compressed(fd, **pay)
    print("figdata ->", fd, flush=True)
    return fd


def plot_maps(plt, frame: str) -> list:
    import matplotlib
    from combined_encoding_figures import _outline
    from fr_pixel_significance import FACTORS
    z = np.load(OUT / f"figdata_{frame}.npz", allow_pickle=True)   # own file; older saves held object labels
    Sg = pd.read_csv(OUT / f"maps_sig_{frame}.csv")
    ol = _outline(pd.DataFrame({"label": list(z["labels"])}))
    out = []
    for w in WINDOWS:
        cols = [("lev", ep) for ep in EP4] + [("chg", ep) for ep in EP4[1:]]
        fig, axs = plt.subplots(len(QS), len(cols), figsize=(2.35 * len(cols) + 0.4, 2.75 * len(QS) + 1.6),
                                squeeze=False)
        for i, (q, _v, ql) in enumerate(QS):
            L = [z[f"lev|{w}|{q}|{ep}"] for ep in EP4]
            vmax = max(float(np.nanpercentile(np.stack(L), 99)), 1e-6)       # ONE level scale per row
            C = [z[f"chg|{w}|{q}|{ep}"] for ep in EP4[1:]]
            cl = max(float(np.nanpercentile(np.abs(np.stack(C)), 99)), 1e-6)  # ONE change scale per row
            for j, (kind, ep) in enumerate(cols):
                ax = axs[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                M = z[f"{kind}|{w}|{q}|{ep}"]
                if kind == "lev":
                    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
                else:
                    im = ax.imshow(M, cmap="RdBu_r", vmin=-cl, vmax=cl, interpolation="nearest")
                    k = f"sig|{w}|{q}|{ep}"
                    if k in z.files and z[k].any():
                        f = FACTORS[16]
                        big = np.zeros(M.shape)
                        sg = z[k]
                        hh, ww = min(sg.shape[0] * f, M.shape[0]), min(sg.shape[1] * f, M.shape[1])
                        big[:hh, :ww] = np.kron(sg, np.ones((f, f)))[:hh, :ww]
                        ax.contour(big, levels=[0.5], colors="w", linewidths=2.2)
                        ax.contour(big, levels=[0.5], colors="k", linewidths=0.9)
                    r = Sg[(Sg.window == w) & (Sg.q == q) & (Sg.epoch == ep)]
                    if len(r):
                        ax.text(0.02, 0.98, f"{r.iloc[0].bins_up}+/{r.iloc[0].bins_down}-", transform=ax.transAxes,
                                fontsize=6.5, va="top", bbox={"facecolor": "white", "alpha": 0.7, "pad": 1, "lw": 0})
                if ol is not None and ol.shape == M.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                if i == 0:
                    ax.set_title(ep if kind == "lev" else f"{ep} - pre", fontsize=9, color=EPOCH_COL[ep])
                if j == 0:
                    ax.set_ylabel(textwrap.fill(ql, 22), fontsize=8)
                if j in (len(EP4) - 1, len(cols) - 1):                 # one colour bar per scale, under the last map
                    cax = ax.inset_axes([0.15, -0.08, 0.7, 0.05])
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
                    cb.locator = matplotlib.ticker.MaxNLocator(3)
                    cb.update_ticks()
                    cb.set_label("unique dR²" if kind == "lev" else "change from pre (dR²)", fontsize=6.5, labelpad=1)
        frl = {"FR-A": "frozen + FaceRhythm (directional face IN the movement set; primary)",
               "noFR": "frozen, no FaceRhythm (sensitivity)"}[frame]
        ttl = (f"NEURAL: movement-null plan maps -- Musall UNIQUE dR² of the spout / plan terms entered AFTER every movement "
               f"group (non-directional licking, nose, running, spout-locked + within-spout direction, upcoming lick angle"
               f"{', FaceRhythm' if frame == 'FR-A' else ''}). {frl}; "
               f"{'cue window 0-1.5 s' if w == 'cue' else 'first-lick window -0.1-1.0 s'}, every frame. Left 4: level "
               f"per epoch (mean of animals; pre held out). Right 3: epoch - pre; outlines = significant 16-px bins "
               f"(nested bootstrap animals > sessions, Bonferroni), +/- = bins up / down. Image left = ipsilesional. "
               f"No lick-history regressor exists in this model. PROVISIONAL, n = 4.")
        fig.suptitle("\n".join(textwrap.wrap(ttl, 175)), fontsize=8.5)
        fig.subplots_adjust(left=0.06, right=0.995, top=1 - 1.25 / (2.75 * len(QS) + 1.6), bottom=0.09,
                            wspace=0.04, hspace=0.32)
        p = OUT / "figures" / f"movement_null_maps_{frame}_{w}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_png_svg(fig, p, dpi=110)
        plt.close(fig)
        out.append(p)
    return out


def area_levels() -> pd.DataFrame:
    U = pd.read_csv(E / "frozen_fr_compare" / "musall_sessions_v2.csv")
    keep = [f"{a}_right" for a, _n in DOT_AREAS] + [f"{a}_left" for a, n in DOT_AREAS if n == 2]
    U = U[U.area.isin(keep) & U.q.isin([q for q, _v, _l in QS])]
    L = U.groupby(["frame", "q", "window", "area", "animal", "epoch"]).u.mean().reset_index()
    L.to_csv(OUT / "area_levels.csv", index=False)
    return L


def plot_area_dots(plt, frame: str, L: pd.DataFrame) -> Path:
    from simple_key_figures import EPOCH_COL as EC
    B = pd.read_csv(E / "frozen_fr_compare" / "musall_change_v2.csv")
    items, x = [], 0.0
    for a, n in DOT_AREAS:
        for h in (("right", "left") if n == 2 else ("right",)):
            items.append((x, f"{a}_{h}", a, h))
            x += 1
        x += 0.5
    rows = [(w, q) for w in WINDOWS for q, _v, _l in QS]
    fig, axs = plt.subplots(len(rows), 2, figsize=(17, 2.9 * len(rows) + 1.4), squeeze=False,
                            gridspec_kw={"width_ratios": [1, 1]})
    off = {"pre": -0.27, "acute": -0.09, "subacute": 0.09, "chronic": 0.27}
    rng = np.random.default_rng(0)
    for i, (w, q) in enumerate(rows):
        l_ = L[(L.frame == frame) & (L.window == w) & (L.q == q)]
        b_ = B[(B.frame == frame) & (B.window == w) & (B.q == q)]
        for j, kind in enumerate(("level", "change")):
            ax = axs[i, j]
            for x0, area, _a, h in items:
                for ep in (EP4 if kind == "level" else EP4[1:]):
                    xx = x0 + off[ep]
                    col = EC[ep]
                    if kind == "level":
                        v = l_[(l_.area == area) & (l_.epoch == ep)].set_index("animal").u.reindex(AN).to_numpy(float)
                        pt = np.nanmean(v)
                    else:
                        r = b_[(b_.area == area) & (b_.epoch == ep)]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        v = np.array([r.get(an, np.nan) for an in AN], float)
                        pt = r.point
                        ax.plot([xx, xx], [r.lo, r.hi], color=col, lw=1.3, zorder=4)
                        if r.lo > 0 or r.hi < 0:
                            ax.text(xx, max(r.hi, np.nanmax(v)), "*", color=col, ha="center", va="bottom",
                                    fontsize=10, fontweight="bold")
                    ax.scatter(xx + rng.uniform(-0.03, 0.03, 4), v, s=7, c="0.6", lw=0, zorder=3)
                    ax.plot(xx, pt, "o", ms=5.5, color=col, mfc=col if h == "right" else "white", mew=1.3, zorder=5)
            ax.axhline(0, color="0.5", lw=0.7, zorder=1)
            ax.set_xticks([it[0] for it in items], [{"right": "c", "left": "i"}[it[3]] for it in items], fontsize=7.5)
            for x0, _area, a, h in items:
                if h == "right":
                    n = 2 if (a, 2) in DOT_AREAS else 1
                    ax.text(x0 + (0.5 if n == 2 else 0), -0.13, a, transform=ax.get_xaxis_transform(), ha="center",
                            va="top", fontsize=8)
            ax.set_xlim(-0.6, items[-1][0] + 0.6)
            ax.set_title(f"{'cue' if w == 'cue' else 'first-lick'} window, {q} unique: "
                         + ("LEVEL per epoch" if kind == "level" else "epoch - pre"), fontsize=9, pad=4)
            ax.set_ylabel("unique R²" if kind == "level" else "change from pre (R²)", fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    hs = [plt.Line2D([], [], marker="o", ls="none", color=EC[ep], label=ep) for ep in EP4]
    fig.legend(handles=hs, loc="upper right", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"NEURAL: movement-null plan term per area ({frame} frame) -- target / spout unique R² entered after "
                 f"every movement group. VISp / RSPd / SSs = controls.\nDots = animals (session means); marker = mean of "
                 f"animals (filled = contralesional c, open = ipsilesional i); change: bar = 95 % nested CI (boot_delta), "
                 f"* excludes 0. PROVISIONAL, n = 4.", fontsize=9.5, x=0.01, ha="left")
    fig.subplots_adjust(left=0.05, right=0.995, top=1 - 1.15 / (2.9 * len(rows) + 1.4), bottom=0.05, wspace=0.12,
                        hspace=0.62)
    p = OUT / "figures" / f"movement_null_area_dots_{frame}.png"
    save_png_svg(fig, p, dpi=110)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", nargs="+", default=list(FRAMES), choices=tuple(FRAMES))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true", help="redraw from the saved figdata / CSVs")
    a = ap.parse_args(argv)
    assert_writable(OUT)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    if not a.plot_only:
        for fr in a.frames:
            compute(fr, a.jobs)
    L = area_levels()
    per = L[L.area.isin(["SSp-n_right", "SSp-m_right", "ALM_right", "tjM1_right", "MOp_right", "VISp_right",
                         "RSPd_right"])].pivot_table(index=["frame", "window", "q", "area", "animal"], columns="epoch",
                                                     values="u").reindex(columns=list(EP4))
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print("== per animal (rule 8): area unique R2 level per epoch (contralesional)\n" + per.round(4).to_string())
    out = []
    for fr in a.frames:
        out += plot_maps(plt, fr)
        out.append(plot_area_dots(plt, fr, L))
    print("\n".join(str(p) for p in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
