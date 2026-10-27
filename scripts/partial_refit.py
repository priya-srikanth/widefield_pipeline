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
  exec_shuf    NULL for exec: executed templates assigned by spout labels shifted by half the session (10-25; added to
               an existing run with --add exec_shuf). Read every set AGAINST its null: the nulls also gain post-stroke
Scores: `frozen_encoder.musall_r2` (frame / trial / avg) per window x area, and frame R2 per spout (all areas).
Readout: gain = R2[variant] - R2[frozen], epoch - pre within animal (nested bootstrap), per-animal table first.

FROZEN_DESIGN=te runs (DECISIONS 2026-10-26: target vs execution kernels) use their own sets and nulls (TE_SETS):
  target / target_spout (pos_enl, pos_cue) / target_lick (lick_onset@tgt:*) / execution / exec_land
  (lick_onset@land:*) / contact / target_land; nulls target_perm + target_lick_perm (ONE block permutation of the trial
  spouts fed to BOTH raw_blocks and te_lick_parts, so the arrival / cue kernels and the target-labelled lick kernels see
  the same shuffled labels), exec_land_perm + execution_perm (landing weights circularly shifted along the session's
  landed licks by a random 10-90 % of their number; kinematics / FR stay real -> conservative). N_PERM draws each.
Run with the SAME env as the frozen run (run_frozen_fr.sh for the frA frame; run_te.sh for the te frame).
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
        "target_exec_fr": ("target", "exec_locked", "exec_within", "fr_dir"),
        # FROZEN_LICKSPOUT runs: the per-spout lick kernels (executed, lick-locked spout code) -- the crisp contest
        "lickspout": ("lickspout",), "exec_ls": ("exec_locked", "exec_within", "lickspout"),
        "target_ls": ("target", "lickspout")}
VARIANTS = ["target", "exec", "exec_fr", "target_exec", "target_exec_fr", "lickspout", "exec_ls", "target_ls", "target_shuf",
            "exec_shuf"]
NULLS = ("target_shuf", "exec_shuf", "target_perm", "exec_perm")
VARIANTS += ["target_perm", "exec_perm"]
#: BLOCK-PERMUTATION nulls (Priya 10-25: "spout labels aren't totally random - a half session shift may not be
#: appropriate" -> "let's do block permutation"): spouts come in runs of ~6 trials (10-90 % 4-8); one circular shift is
#: a single noisy draw (label agreement 0.12-0.19 per animal x epoch vs chance 0.167, PS92_1006 0.34). Each draw permutes
#: the ORDER of the same-spout runs (run lengths travel with their runs, so block structure is kept and agreement with
#: the true labels is chance in expectation); N_PERM draws, each scored, scores AVERAGED (not the predictions).
N_PERM = 5
CONTRASTS = {"target - null": ("target", "target_perm"), "exec - null": ("exec", "exec_perm"),
             "target - shift null": ("target", "target_shuf"), "exec - shift null": ("exec", "exec_shuf"),
             "exec - target": ("exec", "target"),
             "lickspout - target": ("lickspout", "target")}
#: FROZEN_DESIGN=te sets: name -> (FGROUP groups, column-variable prefixes); a column is in the set if either matches
TE_SETS = {"target": (("target",), ()), "target_spout": ((), ("pos_enl", "pos_cue")),
           "target_lick": ((), ("lick_onset@tgt:",)), "execution": (("execution",), ()),
           "exec_land": ((), ("lick_onset@land:",)), "contact": (("contact",), ()),
           "target_land": (("target",), ("lick_onset@land:",))}
TE_NULLS = {"target_perm": "target", "target_lick_perm": "target_lick", "exec_land_perm": "exec_land",
            "execution_perm": "execution"}
TE_CONTRASTS = {"target - null": ("target", "target_perm"), "target lick - null": ("target_lick", "target_lick_perm"),
                "landing lick - null": ("exec_land", "exec_land_perm"),
                "execution - null": ("execution", "execution_perm"),
                "landing lick - target lick": ("exec_land", "target_lick"), "execution - target": ("execution", "target")}
TE = __import__("os").environ.get("FROZEN_DESIGN", "") == "te"
if TE:
    VARIANTS = [*TE_SETS, *TE_NULLS]
    CONTRASTS = TE_CONTRASTS


def model_label(run: Path) -> str:
    """Plain-words name of the frozen model a partial refit was run on (Priya 10-26: every figure must say which partial
    refit model the data come from); built from the run tag, e.g. gate-attempted_v7g3_t3d_frA_te."""
    tag = run.name.replace("frozen_encoder_", "")
    parts = ["frozen pre-stroke encoder (August pre, attempted trials, 3-D tongue)"]
    if "_frA" in tag:
        parts.append("+ FaceRhythm (mode A)")
    if tag.endswith("_te"):
        design = ("NEW target-vs-execution design: target = spout identity + target-labelled lick kernels, execution = "
                  "3-D tongue / jaw + FaceRhythm + landing-labelled lick kernels; ITI frames included")
    elif tag.endswith("_ls"):
        design = "OLD design + per-spout lick kernels (target vs executed spout-locked templates vs lick kernels)"
    else:
        design = "OLD design: target columns vs executed spout-locked templates + within-spout direction"
    return f"MODEL: {' '.join(parts)}  [run {tag}]\n{design}"


def _te_mask(name, vcol, grp):
    gs, pre = TE_SETS[name]
    return np.isin(grp, gs) | np.array([str(v).startswith(pre) for v in vcol]) if pre else np.isin(grp, gs)


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


def _block_perm(p, rng):
    """Labels with the order of their same-label runs permuted (run lengths kept)."""
    p = np.asarray(p, dtype=object)
    st = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
    ln = np.diff(np.r_[st, len(p)])
    o = rng.permutation(len(st))
    return np.repeat(p[st][o], ln[o])


def _avg_scores(ds):
    """Element-wise mean of a list of nested score dicts (window -> measure -> per-area list)."""
    out = {}
    for w in ds[0]:
        out[w] = {k: [float(np.mean([d[w][k][i] for d in ds if w in d and k in d[w]])) for i in range(len(v))]
                  for k, v in ds[0][w].items()}
    return out


def _perm_scores(fe, label, tpl, F, Zv, beta, grp, R, lam_all, fold, which, d, pos_order):
    """Block-permutation nulls: N_PERM draws per null, each scored; returns averaged score dicts."""
    import zlib
    ce = fe._ce()
    out = {}
    for name in which:
        sc = []
        for i in range(N_PERM):
            rng = np.random.default_rng(zlib.crc32(f"{label}|{name}|{i}".encode()))
            if name == "target_perm":
                orig = ce.raw_blocks

                def permuted(B, ft, pos_perm=None, drop_pos=None, _o=orig, _r=rng):
                    pp = _block_perm(B.trials.pos.to_numpy(), _r)
                    return _o(B, ft, pos_perm={"pos_enl": pp, "pos_cue": pp}, drop_pos=drop_pos)
                ce.raw_blocks = permuted
                try:
                    d2 = fe.session_design(label, tpl)
                finally:
                    ce.raw_blocks = orig
                m = grp == "target"
            else:
                orig_s = fe.spout_split

                def split_perm(x, pos_split, k_of, pos_order_, M, _o=orig_s, _r=rng):
                    return _o(x, _block_perm(pos_split, _r), k_of, pos_order_, M)
                fe.spout_split = split_perm
                try:
                    d2 = fe.session_design(label, tpl)
                finally:
                    fe.spout_split = orig_s
                m = np.isin(grp, SETS["exec"])
            base = Zv[:, ~m] @ beta[~m]
            P = base + _refit(((d2["X"] / F["scale"]) @ F["C"])[:, m], R - base, lam_all[m], fold)
            sc.append(_scores(fe, R, P, d, pos_order))
        out[name] = _avg_scores(sc)
    return out


def _te_perm_scores(fe, label, tpl, F, Zv, beta, vcol, grp, R, lam_all, fold, d, pos_order):
    """FROZEN_DESIGN=te nulls (module docstring): per draw one rebuilt design per null family, the family's sets refit
    from it; scores averaged over N_PERM draws."""
    import zlib
    ce = fe._ce()
    sc = {k: [] for k in TE_NULLS}
    for i in range(N_PERM):
        for fam in ("target", "land"):
            rng = np.random.default_rng(zlib.crc32(f"{label}|te_{fam}|{i}".encode()))
            orig_rb, orig_tl = ce.raw_blocks, fe.te_lick_parts
            held = {}
            if fam == "target":
                def rb(B, ft, pos_perm=None, drop_pos=None, _o=orig_rb, _r=rng, _h=held):
                    _h["pp"] = _block_perm(B.trials.pos.to_numpy(), _r)
                    return _o(B, ft, pos_perm={"pos_enl": _h["pp"], "pos_cue": _h["pp"]}, drop_pos=drop_pos)

                def tl(*args, _o=orig_tl, _h=held, **kw):
                    return _o(*args, tpos_perm=_h["pp"], **kw)
                ce.raw_blocks = rb
            else:
                frac = float(rng.uniform(0.1, 0.9))

                def tl(*args, _o=orig_tl, _f=frac, **kw):
                    return _o(*args, land_roll=_f, **kw)
            fe.te_lick_parts = tl
            try:
                d2 = fe.session_design(label, tpl)
            finally:
                ce.raw_blocks, fe.te_lick_parts = orig_rb, orig_tl
            Z2 = (d2["X"] / F["scale"]) @ F["C"]
            for null, base_set in TE_NULLS.items():
                if (fam == "target") != base_set.startswith("target"):
                    continue
                m = _te_mask(base_set, vcol, grp)
                base = Zv[:, ~m] @ beta[~m]
                sc[null].append(_scores(fe, R, base + _refit(Z2[:, m], R - base, lam_all[m], fold), d, pos_order))
    return {k: _avg_scores(v) for k, v in sc.items()}


def _null_preds(fe, label, tpl, F, Zv, beta, grp, R, lam_all, fold, which):
    """NULLS: the refit set's columns rebuilt with the spout labels circularly shifted by half the session (block
    structure kept, alignment with the data broken), refit the same way. target_shuf: target columns from shifted trial
    positions (raw_blocks patched). exec_shuf: executed spout-locked / within-spout columns from templates assigned by
    shifted per-frame spout labels (spout_split patched; frames rolled by half the record ~ trials shifted by half the
    session; exec_angle is label-free and stays real -> conservative). Patches live in this worker only."""
    out = {}
    if "target_shuf" in which:
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
        base = Zv[:, ~m] @ beta[~m]
        out["target_shuf"] = base + _refit(((d2["X"] / F["scale"]) @ F["C"])[:, m], R - base, lam_all[m], fold)
    if "exec_shuf" in which:
        orig_s = fe.spout_split

        def split_shifted(x, pos_split, k_of, pos_order, M):
            ps = np.asarray(pos_split, dtype=object)
            return orig_s(x, np.roll(ps, len(ps) // 2), k_of, pos_order, M)
        fe.spout_split = split_shifted
        try:
            d3 = fe.session_design(label, tpl)
        finally:
            fe.spout_split = orig_s
        m = np.isin(grp, SETS["exec"])
        base = Zv[:, ~m] @ beta[~m]
        out["exec_shuf"] = base + _refit(((d3["X"] / F["scale"]) @ F["C"])[:, m], R - base, lam_all[m], fold)
    return out


def job(item):
    label, run_dir, fits_path, *rest = item
    add = tuple(rest[0]) if rest and rest[0] else None     # add-only mode: compute these nulls, merge into the saved json
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
    sdir = Path(fits_path).parent / "sessions"
    if add:
        old = json.loads((sdir / f"{label}.json").read_text())
        new = _null_preds(fe, label, tpl, F, Zv, beta, grp, R, lam_all, fold, add)
        old["scores"].update({k: _scores(fe, R, P, d, pos_order) for k, P in new.items()})
        old["scores"].update(_perm_scores(fe, label, tpl, F, Zv, beta, grp, R, lam_all, fold,
                                          [n for n in add if n.endswith("_perm")], d, pos_order))
        (sdir / f"{label}.json").write_text(json.dumps(old))
        return old
    if TE:
        if not fe.TE:
            raise ValueError("partial_refit TE mode needs FROZEN_DESIGN=te in the frozen encoder too")
        for name in TE_SETS:
            m = _te_mask(name, vcol, grp)
            if m.any():
                base = Zv[:, ~m] @ beta[~m]
                preds[name] = base + _refit(Zv[:, m], R - base, lam_all[m], fold)
        perm = _te_perm_scores(fe, label, tpl, F, Zv, beta, vcol, grp, R, lam_all, fold, d, pos_order)
        out = {"label": label, "animal": an, "epoch": d["epoch"], "day": d["day"], "n_trials": int(d["n_trials"]),
               "design": "te", "areas": list(F["areas"]),
               "scores": {**{k: _scores(fe, R, P, d, pos_order) for k, P in preds.items()}, **perm}}
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / f"{label}.json").write_text(json.dumps(out))
        return out
    for name, gs in SETS.items():
        m = np.isin(grp, gs)
        if not set(gs) <= set(grp):                      # e.g. lickspout sets in a run without FROZEN_LICKSPOUT
            continue
        base = Zv[:, ~m] @ beta[~m]
        preds[name] = base + _refit(Zv[:, m], R - base, lam_all[m], fold)
    preds.update(_null_preds(fe, label, tpl, F, Zv, beta, grp, R, lam_all, fold, NULLS))
    perm = _perm_scores(fe, label, tpl, F, Zv, beta, grp, R, lam_all, fold, ("target_perm", "exec_perm"), d,
                        pos_order)
    out = {"label": label, "animal": an, "epoch": d["epoch"], "day": d["day"], "n_trials": int(d["n_trials"]),
           "areas": list(F["areas"]), "scores": {**{k: _scores(fe, R, P, d, pos_order) for k, P in preds.items()},
                                                 **perm}}
    sdir.mkdir(parents=True, exist_ok=True)                  # saved per session as it finishes -> a restart resumes
    (sdir / f"{label}.json").write_text(json.dumps(out))
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
    ap.add_argument("--add", nargs="+", choices=NULLS, help="compute only these nulls for sessions already scored")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = run / "partial_refit"
    sdir = out / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    if not a.plot_only:
        from wfield_local import analysis_kit as ak
        fp = frozen_fits(run)
        labels = a.labels or sorted(p.stem for p in (run / "joint" / "sessions").glob("*.json"))
        if a.add:
            todo = [lab for lab in labels if (sdir / f"{lab}.json").exists()
                    and not set(a.add) <= set(json.loads((sdir / f"{lab}.json").read_text())["scores"])]
        else:
            todo = [lab for lab in labels if not (sdir / f"{lab}.json").exists()]
        print(f"{len(todo)} of {len(labels)} sessions", flush=True)
        res, fail = ak.fan_sessions([(lab, str(run), str(fp), a.add) for lab in todo], job, jobs=a.jobs, label="partial")
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
    seen = []
    for v in [v for v in VARIANTS if v in W]:            # sets with an absent group (older runs) duplicate their base
        if any(np.allclose(W[v], W[u], equal_nan=True) for u in seen):
            W = W.drop(columns=v)
        else:
            seen.append(v)
    G = pd.concat([(W[v] - W["frozen"]).rename("u").reset_index().assign(q=v) for v in VARIANTS if v in W],
                  ignore_index=True)
    #: CONTRASTS (10-25): the shuffled-label null also gains post-stroke (any refit block-structured regressor absorbs
    #: some of the larger post residual), so target is read AGAINST it; executed vs target = whose refit helps more
    Wg = G.pivot_table(index=key, columns="q", values="u")
    for name, (x, y) in CONTRASTS.items():
        if x in Wg and y in Wg:
            G = pd.concat([G, (Wg[x] - Wg[y]).rename("u").reset_index().assign(q=name)], ignore_index=True)
    G.to_csv(out / "gain.csv", index=False)
    Gf = G[G.measure == "frame"]
    pd.set_option("display.width", 230)
    per = Gf[Gf.area.isin(["SSp-m_right", "SSp-m_left", "cortex"])].groupby(["q", "window", "area", "animal", "epoch"]) \
        .u.mean().unstack("epoch").reindex(columns=ffc.EP4)
    print("== per animal (rule 8): R2 gain over frozen, by refit set\n" + per.round(4).to_string())
    QQ = [*VARIANTS, *CONTRASTS]
    B = pd.concat([ffc.boot(Gf[Gf.q == q], "u").assign(q=q, frame=q) for q in QQ if (Gf.q == q).any()], ignore_index=True)
    B.to_csv(out / "boot.csv", index=False)
    print(B[B.area.isin(["SSp-m_right", "SSp-m_left", "cortex"]) & B.epoch.isin(["acute", "chronic"])]
          [["q", "window", "area", "epoch", "ref", "point", "lo", "hi", "n_up", "n"]].round(4).to_string(index=False))
    Sp = G[G.measure.str.startswith("spout:")].assign(spout=lambda d: d.measure.str[6:])
    Bs = pd.concat([ffc.boot(Sp[Sp.q == q], "u", keys=("window", "area", "spout")).assign(q=q)
                    for q in (("target", "target_lick", "exec_land", "target_perm") if TE else ("target", "exec", "target_shuf"))
                    if (Sp.q == q).any()], ignore_index=True)
    Bs.to_csv(out / "boot_by_spout.csv", index=False)
    print("== by spout, SSp-m contra, chronic\n" + Bs[(Bs.area == "SSp-m_right") & (Bs.epoch == "chronic")]
          [["q", "window", "spout", "point", "lo", "hi", "n_up", "n"]].round(4).to_string(index=False))
    cols = dict(zip(VARIANTS, ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#66c2a5", "#999999",
                               "#bdd7e7"]))
    cols.update({"target - null": "#b2182b", "exec - null": "#6baed6", "exec - target": "#2166ac",
                 "lickspout - target": "#8c510a"})
    if TE:
        cols = dict(zip(VARIANTS, ["#e41a1c", "#fb9a99", "#b2182b", "#377eb8", "#1f78b4", "#4daf4a", "#984ea3", "#999999",
                                   "#cccccc", "#9ecae1", "#c6dbef"]))
        cols.update({"target - null": "#b2182b", "target lick - null": "#e7298a", "landing lick - null": "#2166ac",
                     "execution - null": "#6baed6", "landing lick - target lick": "#8c510a",
                     "execution - target": "#01665e"})
    ffc.COL.update(cols)
    (out / "figures").mkdir(exist_ok=True)
    for w in ffc.WINDOWS:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 8.6))
        ffc.bars(ax, B[B.window == w], [v for v in VARIANTS if (B.q == v).any()], AREAS, "R2 gain over frozen, epoch - pre",
                 "refit ONLY one set (legend); greys / light blues = the label-shuffled nulls" if TE else
                 "refit ONLY: red = target, blue = executed position, green = executed + FR side / distance, purple = "
                 "target + executed, orange = target + executed + FR, grey = target with shuffled labels (null)")
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ffc.bars(ax2, B[B.window == w], [v for v in CONTRASTS if (B.q == v).any()], AREAS, "difference of gains, epoch - pre",
                 "CONTRASTS (legend): each set minus its label-shuffled null; landing vs target-labelled lick kernels; "
                 "execution vs target" if TE else
                 "CONTRASTS: dark red = target minus its shuffled-label null; light blue = executed minus its shuffled-"
                 "label null; dark blue = executed minus target (whose refit helps more)")
        ax2.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"NEURAL: PARTIAL REFIT, {w} window: one group refit per session (cross-validated), all else frozen;\n"
                     f"{model_label(run)}\ngain in held-out R2 over the frozen model, as a change from pre "
                     "(within animal). Bars = mean of animals; whisker 95 % nested CI; * excludes 0; dots = animals.",
                     fontsize=9)
        fig.subplots_adjust(left=0.05, right=0.83, top=0.86, bottom=0.07, hspace=0.35)
        fig.savefig(out / "figures" / f"partial_refit_bars_{w}.png", dpi=105)            # supplementary (all sets)
        plt.close(fig)
    # MAIN: epoch lines of the contrasts that carry the claim (Priya 10-25: bars too many to read)
    ser = [(q, lab, cols[q]) for q, lab in (("target - null", "target (vs its null)"),
                                            ("exec - null", "executed movement (vs its null)"),
                                            ("lickspout - target", "per-spout lick kernels minus target"),
                                            ("target lick - null", "target-labelled lick kernels (vs null)"),
                                            ("landing lick - null", "landing-labelled lick kernels (vs null)"),
                                            ("landing lick - target lick", "landing- minus target-labelled lick kernels"),
                                            ("execution - target", "execution minus target"))
           if (B.q == q).any()]
    areas = ["SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "cortex"]
    fig, axs = plt.subplots(len(ffc.WINDOWS), len(areas), figsize=(14, 3.2 * len(ffc.WINDOWS) + 1.2), sharey=True,
                            squeeze=False)
    for r, w in enumerate(ffc.WINDOWS):
        for c, a_ in enumerate(areas):
            ax = axs[r, c]
            ffc.epoch_lines(ax, B[B.window == w], ser, a_)
            if r == 0:
                ax.set_title(ffc.ALAB.get(a_, a_), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{w} window\nrefit gain beyond null\n(change from pre, R2)", fontsize=8.5)
    ffc.lines_legend(fig, ser)
    nulls = ("nulls: target sets = block permutation of the trial spouts;\nlanding / execution sets = landing weights "
             "circularly shifted along the landed licks; 5 draws each" if TE else
             "null = block permutation of same-spout runs (5 draws)")
    fig.suptitle(f"NEURAL: PARTIAL REFIT, which part of the mapping changed? Refit ONE part per session (rest frozen)\n"
                 f"{model_label(run)}\ngain in held-out R2 beyond its null ({nulls}), as a change from pre\n"
                 "(within animal; line = mean of animals, band = 95% nested CI animals > sessions). "
                 "All refit sets: partial_refit_bars_<window>.png.", fontsize=9.5)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.15, wspace=0.08, hspace=0.25)
    fig.savefig(out / "figures" / "partial_refit_lines.png", dpi=110)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    raise SystemExit(main())
