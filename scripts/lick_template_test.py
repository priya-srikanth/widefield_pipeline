"""Pre-stroke lick TEMPLATES: does a post-stroke lick evoke the pre-stroke response of the spout it was AIMED at, or of
WHERE IT LANDED? (Priya 10-20: "should we have kernels that match pre-stroke per-spout average licks?" -> "yes")

Why: pre-stroke target and executed nearly coincide, so free kernels cannot separate them; post-stroke they separate,
but free kernels fitted there can bend to either model. Templates learned PRE-stroke and held FIXED leave each
hypothesis no free shape after the stroke (one gain only).

Area level (Allen areas on the eroded stat mask, atlas space -> comparable across sessions), adopted SVTcorr with the
encoding fit's drift filter, trial frames only (strobe - 0.5 .. stop + 1 s).
  pass 1  per PRE session: area ~ shared kernels (cue, arrival, lick timing incl. contacts / cam1-only licks) +
          spout-specific lick kernels (deviation-coded, = `lick_bin_tgt`), ridge -> per-spout kernel coefficients.
          Templates = the animal's mean over its pre sessions (a pre session uses the mean WITHOUT itself).
  pass 2  every session: predictions from the fixed templates --
            target    lick_bin_tgt  (each lick -> its aimed spout's template)
            executed  lick_bin_exec (each lick -> the soft mix of landing-zone templates; calibrated zones,
                      combined_encoding BINS_CALIB_ENV)
          per area, 5-fold (contiguous) CV R2 on LICK-WINDOW frames (<= 0.2 s from a lick peak) of
            base = shared kernels;  base + g * target;  base + g * executed;  base + g1 * target + g2 * executed
          -> gain over base and UNIQUE shares (both - other).
          CEILING (Priya 10-21, "pre-stroke target vs pre-stroke execution vs neither"): the same spout-specific lick
          kernels fitted FREELY in this session (aimed-spout layout, landing-zone layout, both; ridge RIDGE, same CV)
          = what any spout-locked lick response explains here. neither = r2_free - r2_both: lick-locked spout structure
          that matches NO pre-stroke pattern (r2_free = the best of the free fits).
Needs the 3-D environment: COMBINED_ENCODING_EXEC (3-D set), _EXEC3D_TABLE, _TARGET3D_TABLE, _BINS_CALIB; uses
COMBINED_ENCODING_LICKS if set. Output: <run>/imaging_encoding/qc/lick_templates_<exec>/.

Usage: PYTHONPATH=. python scripts/lick_template_test.py --jobs 1 [labels]   (default: labels_3d list via --labels-file)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = 5
RIDGE = 1e-4                         # x mean diagonal of X'X (raw columns, so templates transfer across sessions)
FOCUS = ("cortex", "SSp-m_left", "SSp-m_right", "MOs_left", "MOs_right", "MOp_left", "MOp_right")
EPOCHS = ("pre", "acute", "subacute", "chronic")


def session_design(label: str) -> dict:
    """Area traces, trial / lick-window frame masks, shared + target / executed blocks for one session."""
    from wfield_local import combined_encoding as ce
    os.environ[ce.BINS_ENV] = "1"
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    B = ce.load_behaviour(config.animal_of(label), "2026" + label.split("_")[-1], PathResolver())
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0].astype(np.float32)
    V = V - Q @ (Q.T @ V)
    areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
    Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)
    anames = [names.get(a, str(a)) for a in areas] + ["cortex"]
    Y = (V @ Ra).astype(np.float64)
    T = B.trials
    E = T[T.engaged & np.isfinite(T.cue_s)]
    st = np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0)
    rows = np.zeros(len(ft), bool)
    for a, b in zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0):
        rows |= (ft >= a) & (ft <= b)
    pk = np.sort(B.lick_peak_s)
    j = np.searchsorted(pk, ft)
    d = np.minimum(np.abs(ft - pk[np.clip(j - 1, 0, len(pk) - 1)]), np.abs(ft - pk[np.clip(j, 0, len(pk) - 1)]))
    R = ce.raw_blocks(B, ft)
    tl, el, frame, Wt, P = ce.exec_bins(B, weights=True)
    if frame != "3d_calibrated":
        raise RuntimeError(f"{label}: executed bins frame {frame!r} (needs the calibrated 3-D environment)")
    shared = np.hstack([R["cue"], R["arrival"], R["licks"]])
    blocks = {"tgt": R["lick_bin_tgt"], "exec": R["lick_bin_exec"]}
    return {"label": label, "animal": label.split("_")[0], "epoch": epochs.epoch_of(label), "Y": Y[rows], "areas": anames,
            "shared": shared[rows], "cat": np.hstack([R["pos_enl"], R["pos_cue"]])[rows], "tgt": blocks["tgt"][rows], "exec": blocks["exec"][rows],
            "lickwin": (d <= 0.2)[rows], "agree": float(np.mean(tl[tl != ""] == el[tl != ""]))}


def _ridge(X, Y, lam=RIDGE):
    G = X.T @ X
    return np.linalg.solve(G + lam * np.mean(np.diag(G)) * np.eye(G.shape[0]), X.T @ Y)


def fit_templates(S: dict) -> np.ndarray:
    """Pass 1: spout-specific kernel coefficients (tgt columns x areas) from a PRE session."""
    X = np.column_stack([np.ones(len(S["Y"])), S["shared"], S["tgt"]])
    b = _ridge(X, S["Y"])
    return b[1 + S["shared"].shape[1]:]


def cv_r2(y, X, folds, mask, lam=1e-6):
    """Held-out R2 of (near-)OLS y ~ X on contiguous folds, scored on `mask` frames."""
    pred = np.zeros_like(y)
    for k in range(folds.max() + 1):
        tr, te = folds != k, folds == k
        b = _ridge(X[tr], y[tr], lam)
        pred[te] = X[te] @ b
    r = y[mask] - pred[mask]
    return 1 - (r ** 2).sum() / ((y[mask] - y[mask].mean()) ** 2).sum()


def evaluate(S: dict, tmpl: np.ndarray, tmpl_areas: list) -> list:
    """Pass 2: per area, CV R2 of base / +target / +executed / +both, templates fixed."""
    idx = [tmpl_areas.index(a) for a in S["areas"] if a in tmpl_areas]
    keep = [a for a in S["areas"] if a in tmpl_areas]
    Ys = S["Y"][:, [S["areas"].index(a) for a in keep]]
    Tm = tmpl[:, idx]
    pt, pe = S["tgt"] @ Tm, S["exec"] @ Tm                     # (frames, areas) predictions
    n = len(Ys)
    folds = np.minimum((np.arange(n) * FOLDS) // n, FOLDS - 1)
    one = np.ones((n, 1))
    rows = []
    for i, a in enumerate(keep):
        row = {"label": S["label"], "animal": S["animal"], "epoch": S["epoch"], "area": a, "agree": S["agree"]}
        # '' = base of shared kernels; '_cat' = STRICT: + the six free spout-position kernels (pos_enl / pos_cue), so
        # the templates compete only for LICK-LOCKED spout-specific variance beyond the sustained position code
        for sfx, base in (("", np.column_stack([one, S["shared"]])),
                          ("_cat", np.column_stack([one, S["shared"], S["cat"]]))):
            r = {m: cv_r2(Ys[:, i], np.column_stack([base] + c), folds, S["lickwin"])
                 for m, c in (("base", []), ("tgt", [pt[:, i]]), ("exec", [pe[:, i]]), ("both", [pt[:, i], pe[:, i]]))}
            row.update({**{f"r2_{m}{sfx}": v for m, v in r.items()},
                        f"gain_tgt{sfx}": r["tgt"] - r["base"], f"gain_exec{sfx}": r["exec"] - r["base"],
                        f"u_tgt{sfx}": r["both"] - r["exec"], f"u_exec{sfx}": r["both"] - r["tgt"]})
        base = np.column_stack([one, S["shared"]])
        fr = {m: cv_r2(Ys[:, i], np.column_stack([base] + c), folds, S["lickwin"], RIDGE)
              for m, c in (("tgt", [S["tgt"]]), ("exec", [S["exec"]]), ("both", [S["tgt"], S["exec"]]))}
        free = max(fr.values())
        row.update({**{f"r2_free_{m}": v for m, v in fr.items()}, "r2_free": free,
                    "neither": free - row["r2_both"], "ceiling_gain": free - row["r2_base"]})
        rows.append(row)
    return rows


OUT_ENV = "LICK_TEMPLATES_OUT"        # output folder, inherited by the session workers


def pass1_one(label: str) -> str:
    """Fit one PRE session's spout kernels; saved to <out>/sessions/<label>_tmpl.npz."""
    S = session_design(label)
    np.savez(Path(os.environ[OUT_ENV]) / "sessions" / f"{label}_tmpl.npz", coef=fit_templates(S),
             areas=np.array(S["areas"]))
    return label


def template_for(label: str, pre: list):
    """The animal's mean pre-stroke template, without `label` itself; areas aligned by atlas name."""
    d = Path(os.environ[OUT_ENV]) / "sessions"
    srcs = [k for k in pre if k[:4] == label.split('_')[0] and k != label and (d / f"{k}_tmpl.npz").exists()]
    if not srcs:
        return None, None
    Z = {k: np.load(d / f"{k}_tmpl.npz") for k in srcs}
    A = {k: list(Z[k]["areas"]) for k in srcs}
    common = [c for c in A[srcs[0]] if all(c in A[k] for k in srcs)]
    M = np.mean([Z[k]["coef"][:, [A[k].index(c) for c in common]] for k in srcs], 0)
    return M, common


def pass2_one(item) -> list:
    label, pre = item
    M, common = template_for(label, list(pre))
    if M is None:
        raise RuntimeError("no pre-stroke template for this animal")
    return evaluate(session_design(label), M, common)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--labels-file", default=None, help="whitespace-separated labels (e.g. the 3-D labels_3d.txt)")
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. the 3-D table tag)")
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    mode = os.environ.get(ce.EXEC_ENV, "")
    if not mode or not os.environ.get(ce.BINS_CALIB_ENV):
        raise SystemExit("needs the 3-D environment (COMBINED_ENCODING_EXEC + tables + COMBINED_ENCODING_BINS_CALIB)")
    labels = a.labels or Path(a.labels_file).read_text().split()
    cur = set(ak.curated_labels())
    if any(x not in cur for x in labels):
        raise SystemExit(f"not curated: {[x for x in labels if x not in cur]}")
    out = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc" / (f"lick_templates_{mode}" + (f"_{a.tag}" if a.tag else ""))
    writeguard.assert_writable(out.parent)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    os.environ[OUT_ENV] = str(out)
    pre = [x for x in labels if epochs.epoch_of(x) == "pre"]
    _got, fail = ak.fan_sessions([x for x in pre if not (out / "sessions" / f"{x}_tmpl.npz").exists()],
                                 pass1_one, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED (pass 1) {it}: {err}")
    got2, fail2 = ak.fan_sessions([(x, tuple(pre)) for x in labels], pass2_one, jobs=a.jobs)
    for it, err in fail2:
        print(f"  FAILED (pass 2) {it}: {err}")
    D = pd.DataFrame([r for _l, rs in got2 for r in rs])
    D.to_csv(out / "lick_templates.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    F = D[D.area.isin(FOCUS)]
    for c in ("gain_tgt", "gain_exec", "u_tgt", "u_exec", "u_tgt_cat", "u_exec_cat", "ceiling_gain", "neither"):
        t = F.groupby(["area", "animal", "epoch"])[c].mean().unstack("epoch").reindex(columns=list(EPOCHS)) * 100
        print(f"\n  {c} (x100 = % of lick-window variance; templates fixed from pre-stroke)\n" + t.round(2).to_string())
    print("\n  fraction of licks landing in their own zone (agree)\n"
          + D.groupby(["animal", "epoch"]).agree.mean().unstack().reindex(columns=list(EPOCHS)).round(2).to_string())
    C = F[F.area == "cortex"].groupby(["animal", "epoch"])[["gain_tgt", "gain_exec", "neither", "ceiling_gain"]].mean()
    C = C.assign(target_like=C.gain_tgt / C.ceiling_gain, exec_like=C.gain_exec / C.ceiling_gain,
                 neither_share=C.neither / C.ceiling_gain)
    print("\n  cortex, share of the session's spout-locked lick structure (ceiling = free kernels): target-like /"
          " execution-like (overlap, each alone) / NEITHER\n"
          + C[["target_like", "exec_like", "neither_share"]].unstack("animal").reindex(list(EPOCHS)).round(2).to_string())
    print(f"\n-> {out / 'lick_templates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
