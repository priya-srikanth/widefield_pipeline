"""Spout-in-mesh control for the FaceRhythm pre-cue (ENL) spout-position decode (FACE_RHYTHM_FINDINGS 7, FINDINGS 13).

Open since 10-25: the face (FR factors) decodes spout position before the cue on lick-free trials (2 s .43), and the
video motion-energy check masked the spout pixels (`enl_video_spout_check`) but FR had no such control. Could the FR
mesh points be tracking the SPOUT itself rather than the face?

Geometry (master frame, `register/master_overlay.npz`: every session's per-position spout tips pushed into the master):
the 1282 mesh points (`master.npz` ``home``) sit above the close-spout tips by design (Priya's ROI); 15 points lie within
30 px of any session's close tip, 63 within 60 px, NONE within 56 px of a far tip. So the test is whether those
lowest points carry the decode.

Method: re-project the ENL bins (cue -2 s to the cue) of every session's TRIALS spectrogram on the frozen basis with
the mesh points within R px of ANY spout tip (all sessions, all six positions -- one mask for every trial and session,
so the mask itself cannot carry position) removed from both the data rows and the spatial factors (A). The per-row
normaliser D is FR's own (recovered from the full row, as `cmd_project`), so only the projection changes. Then the
SAME decoder / trials / arms as `fr_enl_decode` (its ``one``). R = 0 (no mask) reproduces `proj/` (checked per session:
max |W - proj| reported) and is the reference arm.

    python scripts/fr_spout_mesh_control.py [--jobs 3] [--radii 30 60] [--only PS92_0806 ...]
    python scripts/fr_spout_mesh_control.py --summary          # tables + figure from the stored per-session csv
Output: <dlc>/face_rhythm/cam4/readouts/master_pre_v1/spout_mesh_control/ ; reduced projections cached locally in
<wfield_cache>/fr_spout_mesh_control/ (derived, re-creatable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fr_enl_decode as fed  # noqa: E402

RADII = (30, 60)
T0_MS, T1_MS = -2000.0, 0.0              # the ENL arms' widest window (2 s ending at the cue)
CHUNK_TR = 24
CACHE = Path("C:/Users/SabatiniLab/wfield_cache/fr_spout_mesh_control")


def point_masks(sd: Path, radii) -> dict:
    """{R: bool (P,) True = KEEP} -- points farther than R px from every spout tip of every session (master frame)."""
    from wfield_local.paths import PathResolver
    M = np.load(sd / "master.npz")
    home = M["home"].astype(float)
    O = np.load(Path(PathResolver().root("dlc")) / "face_rhythm" / "cam4" / "register" / "master_overlay.npz")
    tips = O["tip_xy"].reshape(-1, 2)
    tips = tips[np.isfinite(tips).all(1)]
    d = np.min(np.linalg.norm(home[:, None] - tips[None], axis=2), 1)
    return {r: d >= r for r in radii}, d


def reproject(job):
    """One task file -> {R: (trials (n_tr, nb, Rk) W, D (n_tr, nb))} on the ENL bins, + the check vs proj/."""
    key, pfile, sd, radii, out = job
    from wfield_local.face_rhythm.factors.cp import _scaled, normaliser, project_rows
    from wfield_local.face_rhythm.factors.files import load_master, load_plan
    from wfield_local.face_rhythm.factors.sample import spec_of
    dst = {r: out / f"R{r}" / f"{key}.npz" for r in (0, *radii)}
    if all(p.exists() for p in dst.values()):
        return key, "cached", np.nan
    A, B, scales, M = load_master(sd)
    keep, _ = point_masks(sd, radii)
    with np.load(pfile) as z:
        meta = {k: z[k] for k in ("t_ms", "trial_id", "position", "cue_daq_s", "epoch", "animal", "date")}
        Wref = z["trials"]
    t = meta["t_ms"].astype(float)
    b = np.flatnonzero((t >= T0_MS) & (t < T1_MS))
    b0, b1 = int(b[0]), int(b[-1]) + 1
    s = float(scales[str(meta["animal"])])
    pr = load_plan(sd)["params"]
    freqs = M["freqs"].astype(np.float64)
    fr_norm, fr_ex = float(pr.get("vqt_norm", 0.95)), float(pr.get("one_over_f", 0.5))
    St = spec_of(sd, key, "trials")
    n_tr, P = St.shape[0], St.shape[-2]
    nb = b1 - b0
    Rk = A.shape[1]
    Wout = {r: np.full((n_tr, nb, Rk), np.nan, np.float32) for r in dst}
    D = np.full((n_tr, nb), np.nan, np.float32)
    rowkeep = {0: np.ones(2 * P, bool), **{r: np.concatenate([keep[r], keep[r]]) for r in radii}}
    for i in range(0, n_tr, CHUNK_TR):
        blk = np.asarray(St[i:i + CHUNK_TR, b0:b1])                         # (c, nb, 2, P, F) float16
        c = blk.shape[0]
        X = _scaled(blk.reshape(c * nb, *blk.shape[2:]), s)[0:c * nb]      # (rows, 2P, F) float32 / s
        D[i:i + c] = normaliser(X * s, freqs, fr_norm, fr_ex).reshape(c, nb)
        for r, rk in rowkeep.items():
            W, _, _ = project_rows(X[:, rk], A[rk], B)
            Wout[r][i:i + c] = W.reshape(c, nb, Rk)
    bad = ~np.isfinite(Wref[:, b0:b1]).all(-1)                              # proj/'s lossy trials / bins
    for r in Wout:
        Wout[r][bad] = np.nan
    chk = float(np.nanmax(np.abs(Wout[0] - Wref[:, b0:b1])) / max(np.nanmax(np.abs(Wref[:, b0:b1])), 1e-9))
    for r, p in dst.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, trials=Wout[r], trials_D=D, **{**meta, "t_ms": meta["t_ms"][b0:b1]})
    return key, "done", chk


def decode_arm(job):
    label, files = job
    return fed.one((label, files))


def summary(out: Path, S: pd.DataFrame) -> None:
    rows = []
    for R, d in S.groupby("R", sort=True):
        E = fed.epoch_table(d.drop(columns="R"))
        E.insert(1, "R_px", R)
        rows.append(E)
    E = pd.concat(rows)
    E.to_csv(out / "spout_mesh_control_epochs.csv", index=False, float_format="%.4g")
    W = S.pivot_table(index=["label", "animal", "epoch", "arm"], columns="R", values="accuracy").reset_index()
    rad = sorted(c for c in W.columns if isinstance(c, (int, np.integer)) and c > 0)
    for r in rad:
        W[f"d_R{r}"] = W[r] - W[0]
    W.to_csv(out / "spout_mesh_control_paired.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(E.round(3).to_string(index=False))
        print("\nper-animal mean accuracy (R = 0 / masked), all epochs:")
        print(W.groupby(["arm", "animal"])[[0, *rad]].mean().round(3).to_string())
        print("\nmasked - unmasked accuracy, sessions (median, [min, max]):")
        for r in rad:
            print(W.groupby("arm")[f"d_R{r}"].describe()[["50%", "min", "max"]].round(3).to_string())
    figure(S, out / "spout_mesh_control.png")


def figure(S: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.face_rhythm import common
    ecol = common.epoch_colors()
    EP = ["pre", "acute", "subacute", "chronic"]
    arms = list(dict.fromkeys(S.arm))
    rad = sorted(S.R.unique())
    fig, axs = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 3.6), sharey=True, layout="constrained")
    for ax, arm in zip(np.atleast_1d(axs), arms):
        d = S[S.arm == arm].dropna(subset=["accuracy"])
        for k, e in enumerate(EP):
            for j, r in enumerate(rad):
                am = d[(d.epoch == e) & (d.R == r)].groupby("animal").accuracy.mean()
                x = k + (j - (len(rad) - 1) / 2) * 0.22
                ax.plot([x] * len(am), am, "o", mfc="none", mec=ecol[e], ms=4, alpha=.7)
                se = am.std(ddof=1) / np.sqrt(len(am)) if len(am) > 1 else 0
                ax.errorbar(x, am.mean(), se, fmt=["s", "D", "^"][j % 3], color=ecol[e], ms=5.5, capsize=2)
        ax.axhline(d.chance.median(), color="0.5", ls=":", lw=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(EP, fontsize=8)
        ax.set_title(arm, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel("spout position decoded from FR factors\n(cross-validated accuracy)")
    fig.suptitle("Spout-in-mesh control: mesh points within R px of any spout tip removed before projection. Markers "
                 "left to right: R = " + ", ".join(f"{r} px" if r else "none" for r in rad)
                 + ". Open = animal means; filled = mean of animals +- SEM; dotted = chance", fontsize=8.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.face_rhythm import common
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--radii", type=int, nargs="+", default=list(RADII))
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    sd = common.default_set_dir()
    out = common.readouts_dir() / "spout_mesh_control"
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.summary:
        summary(out, pd.read_csv(out / "spout_mesh_control_sessions.csv"))
        return 0
    keep, dist = point_masks(sd, a.radii)
    print("points kept:", {r: f"{int(k.sum())} / {len(k)}" for r, k in keep.items()})
    T = common.sessions(sd)
    if a.only:
        T = T[T.label.isin(a.only)]
    jobs = [(k, f, sd, tuple(a.radii), CACHE) for k, f in zip(T.key, T.file)]
    res, fails = ak.fan_sessions(jobs, reproject, jobs=a.jobs, key=lambda j: j[0])
    for f in fails:
        print("!! reproject", f)
    chk = pd.DataFrame([r for _, r in res], columns=["key", "status", "rel_maxdiff_vs_proj"])
    chk.to_csv(out / "reproject_check.csv", index=False)
    print("R = 0 vs proj/: max relative |dW|", np.nanmax(chk.rel_maxdiff_vs_proj.to_numpy(float), initial=0))
    rows = []
    for r in (0, *a.radii):
        dj = [(lab, tuple(CACHE / f"R{r}" / f"{k}.npz" for k in g.key)) for lab, g in T.groupby("label", sort=True)]
        dj = [j for j in dj if all(p.exists() for p in j[1])]
        res, fails = ak.fan_sessions(dj, decode_arm, jobs=a.jobs, key=lambda j: j[0])
        for f in fails:
            print("!! decode", r, f)
        rows += [{**x, "R": r} for _, rr in res for x in rr]
    S = pd.DataFrame(rows)
    S.to_csv(out / "spout_mesh_control_sessions.csv", index=False, float_format="%.4g")
    summary(out, S)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
