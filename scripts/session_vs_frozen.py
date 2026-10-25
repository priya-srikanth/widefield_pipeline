"""PER-SESSION vs FROZEN "remapping" (Priya 10-25: "compare the per-session and frozen results too for remapping").

For each session: the SAME design as the frozen run (same regressors incl. the animal's pre-stroke spout templates, same
trial gate, same Allen-area targets), fit on THAT session only -- ridge, leave-one-trial-block-out (5 contiguous blocks),
penalties re-chosen within the session -- and scored with the same `musall_r2` windows (cue 0..1.5 s, lick -0.1..1.0 s)
on the held-out predictions. The frozen R2 of the same session comes from the frozen run (pre: leave-one-session-out).

  gap(session)       = R2[session self-fit] - R2[frozen]            (how much a session-specific mapping beats the pre one)
  remap(epoch)       = gap(epoch) - gap(pre)                         (within animal; then mean of animals, nested bootstrap)
  unique gap of g    = unique_self(g) - unique_frozen(g), unique = R2[joint] - R2[no_g] in each frame

The pre gap is NOT zero: the self-fit trains on 4/5 of one session, the frozen fit on the other pre sessions, so the two
differ in training size and in session-to-session variability. Only the CHANGE from pre is read (Layer 2 is the formal
test of a changed mapping; this is its per-session, model-comparison counterpart). Session size enters the self-fit
(fewer trials -> worse self-fit), so n_trials is saved and the per-animal table shows it.

Run with the SAME env as the frozen run it is compared against (run_frozen_fr.sh for --tag gate-attempted_v7g3_t3d_frA).
Usage: python scripts/session_vs_frozen.py --tag gate-attempted_v7g3_t3d_frA [--jobs 2] [--variants joint no_target ...]
       python scripts/session_vs_frozen.py --tag ... --plot-only
Output: <E>/frozen_encoder_<tag>/session_vs_frozen/{sessions/<label>.json, gap.csv, remap_boot.csv, figures/*.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left"]
DEFAULT_VARIANTS = ["joint", "no_target", "no_spout", "no_task", "no_movement", "no_fr", "no_fr_dir"]
NFOLD = 5


def self_fit_job(item):
    """One session: leave-one-trial-block-out ridge on its own frames, per variant -> musall R2 per window x area."""
    label, run_dir, variants = item
    from wfield_local import config
    from wfield_local import frozen_encoder as fe
    run = Path(run_dir)
    an = config.animal_of(label)
    d = fe.session_design(label, fe._templates_of(str(run / "fit" / "templates.pkl"), an))
    f0 = np.load(run / "fit" / f"{an}_joint.npz", allow_pickle=True)
    if len(d["col_var"]) != len(f0["col_var"]) or (d["col_var"] != f0["col_var"]).any():
        raise ValueError(f"{label}: design column layout differs from the frozen run (wrong env?)")
    X, Rd, tri = d["X"], d["Rd"], d["frame_trial"]
    ut = np.unique(tri)
    fold_of_trial = dict(zip(ut, np.minimum(np.arange(len(ut)) * NFOLD // len(ut), NFOLD - 1)))
    fold = np.array([fold_of_trial[t] for t in tri])
    stats = []
    for k in range(NFOLD):
        m = fold == k
        stats.append({"label": f"{label}#f{k}", "animal": an, "epoch": d["epoch"], "col_var": d["col_var"],
                      "area_names": d["area_names"], "n": int(m.sum()), "XtX": X[m].T @ X[m], "XtY": X[m].T @ Rd[m],
                      "YtY": (Rd[m] ** 2).sum(0), "sxx": (X[m] ** 2).sum(0), "pos_order": d["pos_order"],
                      "pos_counts": d["pos_counts"]})
    out = {"label": label, "animal": an, "epoch": d["epoch"], "day": d["day"], "n_trials": int(d["n_trials"]),
           "nfold": NFOLD, "r2_musall": {}, "alphas": {}}
    for v in variants:
        fit = fe.fit_animal(stats, variant=v)
        ix = [d["area_names"].index(a) for a in fit["areas"]]
        Zv = (X / fit["scale"]) @ fit["C"]
        P = np.zeros((len(X), len(ix)))
        for k in range(NFOLD):
            m = fold == k
            P[m] = Zv[m] @ fit["beta_loso"][f"{label}#f{k}"]
        R = Rd[:, ix]
        res = {}
        for al, (a0, a1) in fe.MUSALL_WINDOWS.items():
            fr_ = d["cue_f"] if al == "cue" else d["first_f"]
            keep = fr_ >= 0
            if keep.sum() < 2:
                continue
            offm = np.arange(int(round(a0 * d["fs"])), int(round(a1 * d["fs"])))
            mr = fe.musall_r2(fe._segments(R, fr_[keep], offm), fe._segments(P, fr_[keep], offm), d["pos"][keep])
            res[al] = {kk: [float(x) for x in vv] for kk, vv in mr.items()}
        out["r2_musall"][v] = res
        out["alphas"][v] = {str(g): float(a) for g, a in fit["alphas"].items()}
        out["areas"] = list(fit["areas"])
    return out


def frozen_r2(run: Path, variant: str, measure: str = "frame", areas=None) -> pd.DataFrame:
    rows = []
    for f in sorted((run / variant / "sessions").glob("*.json")):
        m = json.loads(f.read_text())
        for w, dd in m["r2_musall"].items():
            for a in (m["areas"] if areas == "all" else (areas or AREAS)):
                if a in m["areas"]:
                    rows.append({"label": m["label"], "window": w, "area": a, "variant": variant,
                                 "frozen": dd[measure][m["areas"].index(a)]})
    return pd.DataFrame(rows)


def self_r2(sdir: Path, measure: str = "frame", areas=None) -> pd.DataFrame:
    rows = []
    for f in sorted(sdir.glob("*.json")):
        m = json.loads(f.read_text())
        for v, ww in m["r2_musall"].items():
            for w, dd in ww.items():
                for a in (m["areas"] if areas == "all" else (areas or AREAS)):
                    if a in m["areas"]:
                        rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"],
                                     "n_trials": m["n_trials"], "window": w, "area": a, "variant": v,
                                     "self": dd[measure][m["areas"].index(a)]})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--maps", action="store_true", help="Allen-area maps of the remapping (all areas; after the fits)")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = run / "session_vs_frozen"
    sdir = out / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in a.variants if (run / v / "sessions").is_dir()]
    if not a.plot_only:
        from wfield_local import analysis_kit as ak
        labels = a.labels or sorted(p.stem for p in (run / "joint" / "sessions").glob("*.json"))
        todo = [lab for lab in labels if not (sdir / f"{lab}.json").exists()]
        print(f"{len(todo)} of {len(labels)} sessions to fit; variants {variants}", flush=True)
        res, fail = ak.fan_sessions([(lab, str(run), tuple(variants)) for lab in todo], self_fit_job, jobs=a.jobs,
                                    label="selffit")
        for _it, r in res:
            (sdir / f"{r['label']}.json").write_text(json.dumps(r))
        for it, err in fail:
            print(f"  !! {it[0]}: {err}", flush=True)
    if a.maps:
        area_maps(run, out, variants)
        return 0
    summarise(run, out, variants)
    return 0


def area_maps(run: Path, out: Path, variants) -> None:
    """Allen-area maps of the remapping: per area, (self - frozen) epoch - pre within animal, mean of animals, for the
    full model and each group's unique R2; outline = significant after Bonferroni over areas (nested bootstrap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    S = self_r2(out / "sessions", areas="all")
    F = pd.concat([frozen_r2(run, v, areas="all") for v in variants])
    G = S.merge(F, on=["label", "window", "area", "variant"])
    G = G[G.area != "cortex"]
    key = ["label", "animal", "epoch", "window", "area"]
    W = G.pivot_table(index=key, columns="variant", values=["self", "frozen"]).reset_index()
    qs, rows = [], []
    for v in variants:
        q = "full" if v == "joint" else v.replace("no_", "")
        val = (W[("self", "joint")] - W[("frozen", "joint")]) if v == "joint" else \
            (W[("self", "joint")] - W[("self", v)]) - (W[("frozen", "joint")] - W[("frozen", v)])
        rows.append(pd.DataFrame({k: W[k] for k in key}).assign(q=q, u=val.to_numpy()))
        qs.append(q)
    U = pd.concat(rows, ignore_index=True)
    U.columns = [c if isinstance(c, str) else c[0] for c in U.columns]
    B = pd.concat([ffc.boot(U[U.q == q], "u", epochs=["acute", "chronic"]).assign(q=q) for q in qs], ignore_index=True)
    B = B[B.epoch.isin(["acute", "chronic"])].copy()
    B["p_bonf"] = np.minimum(B.p * B.groupby(["q", "window", "epoch"]).area.transform("nunique"), 1.0)
    B.to_csv(out / "remap_area.csv", index=False)
    img, nm = area_image(U.label.iloc[0])
    ol = outline(img)
    (out / "figures").mkdir(exist_ok=True)
    for w in ("cue", "lick"):
        fig, axs = plt.subplots(2, len(qs), figsize=(2.6 * len(qs) + 0.6, 6.2), squeeze=False)
        for j, q in enumerate(qs):
            Bq = B[(B.q == q) & (B.window == w)]
            lim = np.nanpercentile(np.abs(Bq.point), 98) or 1e-3
            for i, ep in enumerate(("acute", "chronic")):
                ax = axs[i, j]
                s_ = Bq[Bq.epoch == ep].set_index("area")
                im = ax.imshow(paint(img, nm, s_.point.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                sg = paint(img, nm, {k: 1.0 for k, v in (s_.p_bonf < 0.05).items() if v})
                if np.isfinite(sg).any():
                    ax.contour(np.nan_to_num(sg), levels=[0.5], colors="w", linewidths=3.4)
                    ax.contour(np.nan_to_num(sg), levels=[0.5], colors="k", linewidths=1.6)
                up, dn = int(((s_.p_bonf < .05) & (s_.point > 0)).sum()), int(((s_.p_bonf < .05) & (s_.point < 0)).sum())
                ax.text(0.02, 0.02, f"sig {up} up / {dn} down", transform=ax.transAxes, fontsize=6.5,
                        bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "lw": 0})
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(q if q == "full" else f"{q} unique", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"{ep} - pre", fontsize=8)
            cax = axs[1, j].inset_axes([0.1, -0.13, 0.8, 0.06])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=6)
        fig.suptitle(f"NEURAL: REMAPPING by Allen area, {w} window ({run.name.replace('frozen_encoder_', '')}): each session refit "
                     "on itself (frozen design + windows) minus the frozen pre-stroke model, epoch - pre within animal, mean of 4."
                     "\n> 0 = a session-specific mapping explains more than the pre-stroke one, beyond the pre-stroke gap. "
                     "Outline = significant after Bonferroni over areas (nested bootstrap). Image left = ipsilesional.",
                     fontsize=8.5)
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.86, wspace=0.06, hspace=0.3)
        fig.savefig(out / "figures" / f"remap_area_maps_{w}.png", dpi=110)
        plt.close(fig)
    pd.set_option("display.width", 220)
    sig = B[B.p_bonf < .05]
    print(sig.groupby(["q", "window", "epoch"]).apply(lambda g: f"{(g.point > 0).sum()} up / {(g.point < 0).sum()} down",
                                                     include_groups=False).to_string())
    print("->", out / "figures")


def summarise(run: Path, out: Path, variants) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    S = self_r2(out / "sessions")
    if S.empty:
        print("no self-fit sessions yet")
        return
    F = pd.concat([frozen_r2(run, v) for v in variants])
    G = S.merge(F, on=["label", "window", "area", "variant"])
    G["gap"] = G["self"] - G["frozen"]
    G.to_csv(out / "gap.csv", index=False)
    key = ["label", "animal", "epoch", "n_trials", "window", "area"]
    W = G.pivot_table(index=key, columns="variant", values=["self", "frozen"]).reset_index()
    rows = []
    for v in variants:
        q = "full" if v == "joint" else v.replace("no_", "")
        if v == "joint":
            val = W[("self", "joint")] - W[("frozen", "joint")]
        else:
            val = (W[("self", "joint")] - W[("self", v)]) - (W[("frozen", "joint")] - W[("frozen", v)])
        rows.append(pd.DataFrame({k: W[k] for k in key}).assign(q=q, u=val.to_numpy()))
    U = pd.concat(rows, ignore_index=True)
    U.columns = [c if isinstance(c, str) else c[0] for c in U.columns]
    pd.set_option("display.width", 220)
    per = U[(U.q == "full") & U.area.isin(["cortex", "SSp-m_right", "SSp-m_left"])] \
        .groupby(["window", "area", "animal", "epoch"]).u.mean().unstack("epoch").reindex(columns=ffc.EP4)
    print("== per animal (rule 8): R2 gap self - frozen, full model\n" + per.round(3).to_string())
    nt = U.drop_duplicates("label").groupby(["animal", "epoch"]).n_trials.median().unstack("epoch").reindex(columns=ffc.EP4)
    print("== trials per session (median)\n" + nt.to_string())
    B = pd.concat([ffc.boot(U[U.q == q], "u").assign(q=q) for q in U.q.unique()])
    B.to_csv(out / "remap_boot.csv", index=False)
    if "area" not in B:
        print("too few sessions for the bootstrap")
        return
    print(B[B.area.isin(["cortex", "SSp-m_right", "SSp-m_left"])].round(4).to_string(index=False))
    qs = ["full", *[v.replace("no_", "") for v in variants if v != "joint"]]
    cols = dict(zip(qs, ["#333333", "#e41a1c", "#ff7f00", "#7570b3", "#1b9e77", "#e7298a", "#a6761d", "#666666"]))
    ffc.COL.update(cols)
    (out / "figures").mkdir(exist_ok=True)
    for w in ffc.WINDOWS:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4.8))
        ffc.bars(ax, B[(B.window == w) & B.q.isin(qs)].assign(frame=lambda d: d.q), qs, AREAS, "remap: (self - frozen) epoch - pre (R2)",
                 "black = full model; colours = each group's UNIQUE R2, self-fit minus frozen")
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"NEURAL: per-session vs FROZEN encoder, {w} window, {run.name.replace('frozen_encoder_', '')}. "
                     "Each session refit on itself (same design, leave-one-trial-block-out) minus the frozen pre-stroke "
                     "model,\nas a change from pre (within animal). > 0 = a session-specific mapping helps more than "
                     "before the stroke. Bars = mean of animals; whisker 95 % nested CI; * excludes 0; dots = animals; "
                     "grey band = pre split-half range.", fontsize=9)
        fig.subplots_adjust(left=0.05, right=0.88, top=0.84, bottom=0.12)
        fig.savefig(out / "figures" / f"session_vs_frozen_{w}.png", dpi=105)
        plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    raise SystemExit(main())
