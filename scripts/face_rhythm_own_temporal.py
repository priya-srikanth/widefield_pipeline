"""Trial-aligned temporal expression of the per-session own-fit columns matched to lick_left / lick_right, and of their
runner-ups, by spout position (Priya 11-16: "look at the expression of the identified lick left and lick right factors
during each spout position trial group ... and the same for runner ups ... absolute not normalized ... are the
identified vs runner-up lick-left factors present on lick-left trials").

The own8 fits store only spatial (A) and spectral (B) loadings, so the time course is recomputed: each session's trial
spectrograms (``out/<key>_trials.npz``, cue -2..+4 s, 25 Hz, engaged trials) are projected onto ALL 12 own columns with
A and B fixed (`cp.project_rows`, exact per-row NNLS, rows / the animal's RMS scale as in the fit), A and B columns
scaled to unit norm first so a column's weight is in data units and columns / sessions are comparable. ABSOLUTE
expression = weight x FR normalizer D (as `face_rhythm_peak.py`). Up to ``--per-pos`` trials per spout (evenly spaced
through the session; lossy bins dropped). Per-session trial means per position for all 12 columns are cached locally
(``<cache>/fr_own_temporal/<key>.npz``), so the pick / runner choice can change without re-reading the share.

Pick / runner-up = `face_rhythm_lick_left_match_check.col_info` (rule v2 pick; runner-up = best-spatial other column).

    python scripts/face_rhythm_own_temporal.py [--per-pos 12] [--jobs 6]
-> ``<readouts>/master_pre_v1/lick_left_match_check/``: own_temporal_lick.png, own_temporal_lick_<animal>.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = Path(os.environ.get("FR_OWN_TEMPORAL_CACHE", Path.home() / "wfield_cache" / "fr_own_temporal"))
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")     # canonical order (ipsi, mid, contra)
POS_LAB = {"close_L": "close ipsi", "close_center": "close mid", "close_R": "close contra", "far_L": "far ipsi",
           "far_center": "far mid", "far_R": "far contra"}
POS_STYLE = {p: dict(color={"L": "#2166ac", "c": "0.45", "R": "#b2182b"}[p.split("_")[1][0]],
                     ls="-" if p.startswith("close") else "--") for p in POS}
EPOCHS = ("pre", "acute", "subacute", "chronic")
UNITS = ("lick_left", "lick_right")


def _unit(M: np.ndarray) -> np.ndarray:
    return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)


def session_curves(key: str, per_pos: int = 12) -> str:
    """Worker: per-position trial-mean ABSOLUTE expression (151, 12) of every own8 column -> cache npz."""
    from wfield_local.face_rhythm import common as frc
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.face_rhythm.factors.cp import normaliser, project_rows, _scaled
    from wfield_local.face_rhythm.factors.files import load_master, load_plan, npz_memmap, npz_small, out_dir
    dst = CACHE / f"{key}_n{per_pos}.npz"
    if dst.exists():
        return str(dst)
    sd = frc.default_set_dir(set_name=fro.SET)
    _A, _B, scales, M = load_master(sd)
    with np.load(sd / "own8" / f"{key}.npz") as z:
        A, B, animal = _unit(z["A_own"].astype(np.float64)), _unit(z["B_own"].astype(np.float64)), str(z["animal"])
    s = float(scales[animal])
    pr = load_plan(sd)["params"]
    freqs = M["freqs"].astype(np.float64)
    dfn = lambda x: normaliser(x * s, freqs, float(pr.get("vqt_norm", 0.95)), float(pr.get("one_over_f", 0.5)))  # noqa: E731
    f = out_dir(sd) / f"{key}_trials.npz"
    S = npz_memmap(f, "spec")
    meta = npz_small(f, ("t_ms", "position"))
    with np.load(sd / "proj" / f"{key}.npz") as z:
        lossy = z["trials_lossy"].astype(bool) if "trials_lossy" in z.files else np.zeros(S.shape[:2], bool)
    pos = meta["position"].astype(str)
    out = {"t_ms": meta["t_ms"]}
    for p in POS:
        idx = np.nonzero(pos == p)[0]
        if len(idx) > per_pos:
            idx = idx[np.round(np.linspace(0, len(idx) - 1, per_pos)).astype(int)]
        cur = []
        for ti in idx:
            if lossy[ti].all():
                continue
            W, _E, _N2, D = project_rows(_scaled(np.asarray(S[ti:ti + 1][0]), s), A, B, row_fn=dfn)
            X = W.astype(np.float64) * D[:, None]
            X[lossy[ti]] = np.nan
            cur.append(X)
        out[f"{p}_mean"] = np.nanmean(cur, 0) if cur else np.full((S.shape[1], A.shape[1]), np.nan)
        out[f"{p}_n"] = np.array(len(cur))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.npz")
    np.savez(tmp, **out)
    os.replace(tmp, dst)
    return str(dst)


def _session_cols(fdir, keys):
    """(key -> {unit: (pick col, runner col, rec, flagged)}) under rule v2."""
    import face_rhythm_epoch_shape as es
    import face_rhythm_lick_left_match_check as mc
    from wfield_local import face_rhythm_readouts as frr
    M, A_ref, B_ref, names = es.reference(fdir)
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    out = {}
    for key in keys:
        with np.load(fdir / "own8" / f"{key}.npz") as z:
            A, B = z["A_own"].astype(float), z["B_own"].astype(float)
        out[key] = {u: (lambda d: (d["pick"]["col"], d["runner"]["col"], d["rec"], mc.is_bad(d)))(
            mc.col_info(A_ref, B_ref, A, B, names.index(u), home, mid)) for u in UNITS}
    return out


def figure(T, curves, t, title, out_png, methods):
    """Rows: lick_left pick / runner-up, lick_right pick / runner-up; columns: epochs; lines: spouts."""
    from wfield_local import face_rhythm_figures as frf
    rows = [(u, w) for u in UNITS for w in ("pick", "runner")]
    with frf.style():
        fig, axs = frf.subplots(len(rows), len(EPOCHS), figsize=(2.6 * len(EPOCHS), 1.9 * len(rows)), squeeze=False,
                                sharex=True)
        for r_i, (u, w) in enumerate(rows):
            ymax = 0.0
            for c_i, ep in enumerate(EPOCHS):
                ax = axs[r_i, c_i]
                sel = T[(T.epoch == ep) & T[f"{u}_rec"]]
                nses = len(sel)
                for p in POS:
                    # per animal: mean over its sessions; then mean over animals
                    per_an = []
                    for _an, g in sel.groupby("animal"):
                        cs = [curves[k][f"{p}_mean"][:, col] for k, col in zip(g.key, g[f"{u}_{w}"])
                              if int(curves[k][f"{p}_n"]) > 0]
                        if cs:
                            per_an.append(np.nanmean(cs, 0))
                    if per_an:
                        y = np.nanmean(per_an, 0)
                        ymax = max(ymax, float(np.nanmax(y)))
                        ax.plot(t, y, lw=1.0, label=POS_LAB[p], **POS_STYLE[p])
                ax.axvline(0, color="0.6", lw=0.5)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                if r_i == 0:
                    ax.set_title(ep, fontsize=9)
                ax.text(0.02, 0.97, f"{nses} sess", transform=ax.transAxes, fontsize=6.5, va="top", color="0.4")
                if c_i == 0:
                    ax.set_ylabel(f"{u}\n{'matched' if w == 'pick' else 'runner-up'}\n(abs.)", fontsize=7.5)
                if r_i == len(rows) - 1:
                    ax.set_xlabel("time from cue (s)")
            for ax in axs[r_i]:
                ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
        axs[0, -1].legend(fontsize=6, frameon=False, loc="upper right")
        fig.suptitle(title, fontsize=10)
        frf.save(fig, out_png, methods)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    import face_rhythm_epoch_shape as es
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import writeguard
    from wfield_local.face_rhythm import common as frc
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-pos", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    es.RULE = "v2"
    fdir = frc.default_set_dir(set_name=fro.SET)
    out = frc.readouts_dir(fro.SET) / "lick_left_match_check"
    writeguard.assert_writable(out)
    S = frc.sessions(fdir, "own8")
    keys = list(S.key)
    from functools import partial
    paths = ak.fan_sessions(keys, partial(session_curves, per_pos=a.per_pos), jobs=a.jobs) \
        if hasattr(ak, "fan_sessions") else [session_curves(k, a.per_pos) for k in keys]
    del paths
    curves = {k: dict(np.load(CACHE / f"{k}_n{a.per_pos}.npz")) for k in keys}
    t = curves[keys[0]]["t_ms"] / 1000.0
    cols = _session_cols(fdir, keys)
    T = S[["key", "animal", "date", "epoch"]].copy()
    for u in UNITS:
        T[f"{u}_pick"] = [cols[k][u][0] for k in keys]
        T[f"{u}_runner"] = [cols[k][u][1] for k in keys]
        T[f"{u}_rec"] = [cols[k][u][2] for k in keys]
        T[f"{u}_flag"] = [cols[k][u][3] for k in keys]
    methods = f"""
Trial-aligned ABSOLUTE expression (temporal weight x FR normalizer D, not normalized) of the per-session own-fit
(own8) column that rule v2 matches to the master lick_left / lick_right ("matched") and of its runner-up (the other
column with the highest spatial cosine to the master), by spout position (lines; blue = ipsilesional = left spouts,
gray = middle, red = contralesional = right spouts; solid = close, dashed = far). Each session's trial spectrograms
(engaged trials, cue -2..+4 s, 25 Hz) projected onto all 12 own columns with spatial and spectral loadings fixed and
scaled to unit norm (so amplitudes are comparable across columns and sessions); up to {a.per_pos} trials per spout per
session, evenly spaced; lossy bins dropped. Per animal: mean over its sessions where that unit is recovered (spatial
cos >= .7, spectral >= .5); drawn: mean over animals. y axis shared within a row. Descriptive; no test.
Script: scripts/face_rhythm_own_temporal.py."""
    figure(T, curves, t, "Per-session fits: matched vs runner-up lick columns, expression by spout (all animals)",
           out / "own_temporal_lick.png", methods)
    for an in sorted(T.animal.unique()):
        figure(T[T.animal == an], curves, t, f"{an}: matched vs runner-up lick columns, expression by spout",
               out / f"own_temporal_lick_{an}.png", methods.replace("Per animal: mean over its sessions", "Mean over "
                                                                    "this animal's sessions"))
    # numbers: response-window (cue 0..+2 s) mean, left spouts vs right spouts, per unit x column x epoch
    w = (t >= 0) & (t <= 2)
    rows = []
    for r in T.itertuples():
        c = curves[r.key]
        for u in UNITS:
            if not getattr(r, f"{u}_rec"):
                continue
            for wh in ("pick", "runner"):
                col = getattr(r, f"{u}_{wh}")
                for p in POS:
                    if int(c[f"{p}_n"]) > 0:
                        rows.append(dict(animal=r.animal, epoch=r.epoch, key=r.key, unit=u, which=wh, pos=p,
                                         flag=getattr(r, f"{u}_flag"), resp=float(np.nanmean(c[f"{p}_mean"][w, col])),
                                         pre=float(np.nanmean(c[f"{p}_mean"][t < 0, col]))))
    R = pd.DataFrame(rows)
    from wfield_local import figure_layout
    R.to_csv(figure_layout.sidecar_for(out, "own_temporal_lick", ".csv"), index=False, float_format="%.5g")
    R["side"] = R.pos.str.split("_").str[1].map({"L": "ipsi", "R": "contra", "center": "mid"})
    P = (R.groupby(["unit", "which", "epoch", "animal", "side"]).resp.mean().groupby(["unit", "which", "epoch", "side"])
         .mean().unstack("side")[["ipsi", "mid", "contra"]])
    P["ipsi/contra"] = P.ipsi / P.contra
    pd.set_option("display.width", 200)
    print(P.reindex(EPOCHS, level="epoch").round(3).to_string())
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
