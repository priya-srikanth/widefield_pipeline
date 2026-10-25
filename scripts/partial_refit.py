"""PARTIAL REFIT: refit only the TARGET vs only the EXECUTED-POSITION terms per session, everything else frozen (Priya
10-25: "let's consider models with refitting only target vs executed position terms"). Asks WHOSE mapping changed: the
self-fit (`session_vs_frozen.py`) refits everything at once and shows that a session-specific mapping fits better after
the stroke (and that target information reappears), not which part.

Per session (frozen design of the run, the animal's frozen pre-stroke weights; pre sessions use their leave-one-session-
out weights): the columns of ONE group set are refit to that session's residual (data - frozen prediction of all other
columns), ridge with the frozen penalty of each column's group, cross-validated over 5 contiguous trial blocks (held-out
predictions only). Variants:
  frozen       nothing refit (reference)
  target       target columns (pos_enl / pos_cue)
  exec         executed movement's position terms: exec_locked (pose spout-locked templates) + exec_within (within-spout
               direction, executed angle)
  exec_fr      exec + FaceRhythm side / distance (fr_dir)
  target_exec  target + exec
  target_exec_fr  target + exec + fr_dir (the whole position package)
  target_shuf  NULL: target columns rebuilt with the trial positions circularly shifted by half the session (block
               structure kept, alignment with the data broken), refit the same way
Scores: `frozen_encoder.musall_r2` (frame / trial / avg) per window x area, and frame R2 per spout (all areas).
Readout: gain = R2[variant] - R2[frozen], epoch - pre within animal (nested bootstrap), per-animal table first.

Run with the SAME env as the frozen run (run_frozen_fr.sh for the frA frame).
Usage: python scripts/partial_refit.py --tag gate-attempted_v7g3_t3d_frA [--jobs 2] [--labels ...] [--plot-only]
Output: <run>/partial_refit/{fits.pkl, sessions/<label>.json, gain.csv, boot.csv, figures/partial_refit_<window>.png}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left"]
NFOLD = 5
SETS = {"target": ("target",), "exec": ("exec_locked", "exec_within"), "exec_fr": ("exec_locked", "exec_within", "fr_dir"),
        "target_exec": ("target", "exec_locked", "exec_within"),
        "target_exec_fr": ("target", "exec_locked", "exec_within", "fr_dir")}
VARIANTS = ["target", "exec", "exec_fr", "target_exec", "target_exec_fr", "target_shuf"]


def frozen_fits(run: Path) -> Path:
    """Per animal frozen joint fit (scale, C, vcol, betas incl. leave-one-session-out, alphas), from the run's saved pre
    statistics; cached in partial_refit/fits.pkl."""
    from wfield_local import frozen_encoder as fe
    p = run / "partial_refit" / "fits.pkl"
    if p.exists():
        return p
    st = pickle.loads((run / "fit" / "pre_stats.pkl").read_bytes())
    fits = {}
    for an in sorted({s["animal"] for s in st}):
        f = fe.fit_animal(sorted((s for s in st if s["animal"] == an), key=lambda s: s["label"]), variant="joint")
        fits[an] = {k: f[k] for k in ("scale", "C", "vcol", "col_var", "areas", "alphas", "beta_all", "beta_loso")}
        print(f"  frozen fit {an}: {len(f['beta_loso'])} pre sessions", flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(fits, protocol=pickle.HIGHEST_PROTOCOL))
    return p


def _scores(fe, R, P, d, ix_pos):
    out = {}
    for al, (a0, a1) in fe.MUSALL_WINDOWS.items():
        fr_ = d["cue_f"] if al == "cue" else d["first_f"]
        keep = fr_ >= 0
        if keep.sum() < 2:
            continue
        offm = np.arange(int(round(a0 * d["fs"])), int(round(a1 * d["fs"])))
        Ds, Ps = fe._segments(R, fr_[keep], offm), fe._segments(P, fr_[keep], offm)
        pos = d["pos"][keep]
        mr = fe.musall_r2(Ds, Ps, pos)
        res = {k: [float(x) for x in v] for k, v in mr.items()}
        for q in ix_pos:
            m = pos == q
            if m.sum() >= 2:
                res[f"spout:{q}"] = [float(x) for x in fe.musall_r2(Ds[m], Ps[m], pos[m])["frame"]]
        out[al] = res
    return out


def _refit(Z, r, lam, fold):
    """Cross-validated ridge of r on Z (held-out predictions per trial-block fold)."""
    out = np.zeros_like(r)
    for k in range(NFOLD):
        te = fold == k
        A = Z[~te].T @ Z[~te] + np.diag(lam)
        out[te] = Z[te] @ np.linalg.solve(A, Z[~te].T @ r[~te])
    return out


def job(item):
    label, run_dir, fits_path = item
    from wfield_local import config
    from wfield_local import frozen_encoder as fe
    run = Path(run_dir)
    an = config.animal_of(label)
    F = pickle.loads(Path(fits_path).read_bytes())[an]
    tpl = fe._templates_of(str(run / "fit" / "templates.pkl"), an)
    d = fe.session_design(label, tpl)
    if len(d["col_var"]) != len(F["col_var"]) or (d["col_var"] != F["col_var"]).any():
        raise ValueError(f"{label}: design column layout differs from the frozen fit (wrong env?)")
    ix = [d["area_names"].index(a) for a in F["areas"]]
    R = d["Rd"][:, ix]
    Zv = (d["X"] / F["scale"]) @ F["C"]
    beta = F["beta_loso"].get(label, F["beta_all"])
    vcol = np.asarray(F["vcol"])
    grp = np.array([fe.FGROUP[v] for v in vcol])
    lam_all = np.array([F["alphas"][fe.FPEN[v]] for v in vcol])
    tri = d["frame_trial"]
    ut = np.unique(tri)
    fmap = dict(zip(ut, np.minimum(np.arange(len(ut)) * NFOLD // len(ut), NFOLD - 1)))
    fold = np.array([fmap[t] for t in tri])
    pos_order = sorted(set(d["pos"]))
    preds = {"frozen": Zv @ beta}
    for name, gs in SETS.items():
        m = np.isin(grp, gs)
        base = Zv[:, ~m] @ beta[~m]
        preds[name] = base + _refit(Zv[:, m], R - base, lam_all[m], fold)
    # NULL: target columns from positions circularly shifted by half the session (patched in this worker only)
    ce = fe._ce()
    orig = ce.raw_blocks

    def shifted(B, ft, pos_perm=None, drop_pos=None):
        p = B.trials.pos.to_numpy()
        sh = np.roll(p, len(p) // 2)
        return orig(B, ft, pos_perm={"pos_enl": sh, "pos_cue": sh}, drop_pos=drop_pos)
    ce.raw_blocks = shifted
    try:
        d2 = fe.session_design(label, tpl)
    finally:
        ce.raw_blocks = orig
    m = grp == "target"
    Z2 = (d2["X"] / F["scale"]) @ F["C"]
    base = Zv[:, ~m] @ beta[~m]
    preds["target_shuf"] = base + _refit(Z2[:, m], R - base, lam_all[m], fold)
    out = {"label": label, "animal": an, "epoch": d["epoch"], "day": d["day"], "n_trials": int(d["n_trials"]),
           "areas": list(F["areas"]), "scores": {k: _scores(fe, R, P, d, pos_order) for k, P in preds.items()}}
    return out


def table(sdir: Path, measure: str = "frame", areas=AREAS) -> pd.DataFrame:
    rows = []
    for f in sorted(sdir.glob("*.json")):
        m = json.loads(f.read_text())
        for v, ww in m["scores"].items():
            for w, dd in ww.items():
                for key, vals in dd.items():
                    if key not in (measure,) and not key.startswith("spout:"):
                        continue
                    for a in (m["areas"] if areas == "all" else areas):
                        if a in m["areas"]:
                            rows.append((m["label"], m["animal"], m["epoch"], v, w, key, a, vals[m["areas"].index(a)]))
    return pd.DataFrame(rows, columns=["label", "animal", "epoch", "variant", "window", "measure", "area", "r2"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = run / "partial_refit"
    sdir = out / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    if not a.plot_only:
        from wfield_local import analysis_kit as ak
        fp = frozen_fits(run)
        labels = a.labels or sorted(p.stem for p in (run / "joint" / "sessions").glob("*.json"))
        todo = [lab for lab in labels if not (sdir / f"{lab}.json").exists()]
        print(f"{len(todo)} of {len(labels)} sessions", flush=True)
        res, fail = ak.fan_sessions([(lab, str(run), str(fp)) for lab in todo], job, jobs=a.jobs, label="partial")
        for _it, r in res:
            (sdir / f"{r['label']}.json").write_text(json.dumps(r))
        for it, err in fail:
            print(f"  !! {it[0]}: {err}", flush=True)
    summarise(run, out)
    return 0


def summarise(run: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    T = table(out / "sessions")
    if T.label.nunique() < 8:
        print(T.groupby(["variant", "window", "area"]).r2.mean().unstack("variant").round(4).to_string())
        return
    key = ["label", "animal", "epoch", "window", "measure", "area"]
    W = T.pivot_table(index=key, columns="variant", values="r2")
    G = pd.concat([(W[v] - W["frozen"]).rename("u").reset_index().assign(q=v) for v in VARIANTS if v in W],
                  ignore_index=True)
    G.to_csv(out / "gain.csv", index=False)
    Gf = G[G.measure == "frame"]
    pd.set_option("display.width", 230)
    per = Gf[Gf.area.isin(["SSp-m_right", "SSp-m_left", "cortex"])].groupby(["q", "window", "area", "animal", "epoch"]) \
        .u.mean().unstack("epoch").reindex(columns=ffc.EP4)
    print("== per animal (rule 8): R2 gain over frozen, by refit set\n" + per.round(4).to_string())
    B = pd.concat([ffc.boot(Gf[Gf.q == q], "u").assign(q=q, frame=q) for q in VARIANTS], ignore_index=True)
    B.to_csv(out / "boot.csv", index=False)
    print(B[B.area.isin(["SSp-m_right", "SSp-m_left", "cortex"]) & B.epoch.isin(["acute", "chronic"])]
          [["q", "window", "area", "epoch", "ref", "point", "lo", "hi", "n_up", "n"]].round(4).to_string(index=False))
    Sp = G[G.measure.str.startswith("spout:")].assign(spout=lambda d: d.measure.str[6:])
    Bs = pd.concat([ffc.boot(Sp[Sp.q == q], "u", keys=("window", "area", "spout")).assign(q=q)
                    for q in ("target", "exec", "target_shuf")], ignore_index=True)
    Bs.to_csv(out / "boot_by_spout.csv", index=False)
    print("== by spout, SSp-m contra, chronic\n" + Bs[(Bs.area == "SSp-m_right") & (Bs.epoch == "chronic")]
          [["q", "window", "spout", "point", "lo", "hi", "n_up", "n"]].round(4).to_string(index=False))
    cols = dict(zip(VARIANTS, ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#999999"]))
    ffc.COL.update(cols)
    (out / "figures").mkdir(exist_ok=True)
    for w in ffc.WINDOWS:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4.8))
        ffc.bars(ax, B[B.window == w], VARIANTS, AREAS, "R2 gain over frozen, epoch - pre",
                 "refit ONLY: red = target, blue = executed position, green = executed + FR side / distance, purple = "
                 "target + executed, orange = target + executed + FR, grey = target with shuffled labels (null)")
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"NEURAL: PARTIAL REFIT, {w} window ({run.name.replace('frozen_encoder_', '')}): one group refit per session "
                     "(cross-validated), all else frozen;\ngain in held-out R2 over the frozen model, as a change from pre "
                     "(within animal). Bars = mean of animals; whisker 95 % nested CI; * excludes 0; dots = animals.",
                     fontsize=9)
        fig.subplots_adjust(left=0.05, right=0.86, top=0.84, bottom=0.12)
        fig.savefig(out / "figures" / f"partial_refit_{w}.png", dpi=105)
        plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    raise SystemExit(main())
