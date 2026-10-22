"""FaceRhythm factor identity vs independent behaviour: LP licks (executed direction, time in trial, lick frequency) and
the treadmill (Priya 10-22; computation `wfield_local/face_rhythm_validation.py`).

    python scripts/face_rhythm_validate.py --set pilot_pre_PS92_PS93 [--epoch pre] [--jobs 3]

-> ``<dlc>/face_rhythm/cam4/validation/<set>/``:
  factor_vs_licks_running.png   per factor: (1) lick-triggered expression by executed direction (left blue / centre
                                grey / right red) for licks EARLY (0-1 s after the cue, solid) vs LATE (1.5-3 s,
                                dashed); (2) expression around the lick (0..+0.2 s) vs instantaneous lick frequency,
                                by direction; (3) running-onset triggered expression (whole session).
  factor_correlations.png       factors x [lick rate left / centre / right, lick frequency, time from cue, speed in
                                trials, speed outside lick periods]: Pearson r over the 40 ms trial bins, session means.
  data/                         the tables (figure_layout.sidecar_for).
Session means first, then the mean over sessions (each session counts once). Absolute expression (weights x D).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIR_COL = {"left": "tab:blue", "centre": "0.4", "right": "tab:red"}
PHASES = {"early": (0.0, 1.0), "late": (1.5, 3.0)}
FREQ_BINS = [4, 5.5, 6.5, 7.5, 8.5, 11]
CORR_COLS = ["rate_left", "rate_centre", "rate_right", "lick_freq_hz", "t_from_cue", "speed_mm_s"]


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_validation as frv
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="pilot_pre_PS92_PS93", help="factor set (or <bundle>/<set>)")
    ap.add_argument("--epoch", default="pre", help="sessions of this epoch ('all' for every projected session)")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, set_name=a.set)
    sess = sorted({tuple(f.stem.split("_")[:2]) for f in (fdir / "proj").glob("*.npz")
                   if a.epoch == "all" or str(np.load(f)["epoch"]) == a.epoch})
    if not sess:
        sys.exit(f"no projected {a.epoch} sessions in {fdir}")
    res, fails = ak.fan_sessions([(str(fdir), an, d) for an, d in sess], frv.session, jobs=a.jobs,
                                 key=lambda j: (j[1], j[2]))
    for j, e in fails:
        print(f"  !! {j[1]}_{j[2]}: {e}")
    S = [v for _, v in res]
    R = S[0]["lta"].shape[-1]
    out = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "validation" / fro.set_tag(a.set)
    writeguard.assert_writable(out.parent)
    out.mkdir(parents=True, exist_ok=True)

    # (1) lick-triggered by direction x phase: session means, then mean over sessions
    lta = {}
    for s in S:
        L = s["licks"]
        for dname in DIR_COL:
            for ph, (lo, hi) in PHASES.items():
                m = ((L.direction == dname) & (L.t_rel >= lo) & (L.t_rel < hi)).to_numpy()
                if m.sum() >= 5:
                    lta.setdefault((dname, ph), []).append((s["lta"][m].mean(0), int(m.sum())))
    # (2) expression around the lick (0..+0.2 s) vs lick frequency, by direction
    w0 = (frv.LAGS >= 0) & (frv.LAGS <= 0.2)
    fq = {}
    for s in S:
        L = s["licks"]
        bins = pd.cut(L.freq_hz, FREQ_BINS)
        for dname in DIR_COL:
            for b in bins.cat.categories:
                m = ((L.direction == dname) & (bins == b)).to_numpy()
                if m.sum() >= 5:
                    fq.setdefault((dname, b.mid), []).append(s["lta"][m][:, w0].mean((0, 1)))
    # (3) running-onset triggered
    run = [s["run_ta"].mean(0) for s in S if len(s["run_ta"])]
    n_on = sum(s["run_onsets"] for s in S)

    fig, axs = plt.subplots(R, 3, figsize=(14, 2.3 * R), squeeze=False)
    for k in range(R):
        a0, a1, a2 = axs[k]
        for (dname, ph), xs in sorted(lta.items()):
            y = np.mean([x[0][:, k] for x in xs], 0)
            a0.plot(frv.LAGS, y, color=DIR_COL[dname], ls="-" if ph == "early" else "--", lw=1.1,
                    label=f"{dname} {ph} ({sum(x[1] for x in xs)})" if k == 0 else None)
        a0.axvline(0, color="k", lw=0.5)
        a0.set_title(f"factor {k}: around each lick (solid early 0-1 s, dashed late 1.5-3 s after cue)", fontsize=7)
        for dname in DIR_COL:
            pts = sorted((mid, np.mean([v[k] for v in vs])) for (dd, mid), vs in fq.items() if dd == dname)
            if pts:
                a1.plot(*zip(*pts), "o-", color=DIR_COL[dname], ms=3, lw=1)
        a1.set_title("expression at the lick (0..+0.2 s) vs lick frequency (Hz)", fontsize=7)
        if run:
            m = np.mean([r[:, k] for r in run], 0)
            a2.plot(frv.RUN_LAGS, m, color="tab:green")
        a2.axvline(0, color="k", lw=0.5)
        a2.set_title(f"around running-bout onsets ({n_on} onsets, {len(run)} sessions)", fontsize=7)
        if k == 0:
            a0.legend(fontsize=5, ncol=2)
    labs = ", ".join(f"{s['animal']}_{s['date'][4:]}" for s in S)
    fig.suptitle(f"{a.set}: factor expression (ABSOLUTE) vs LP licks and treadmill; {len(S)} {a.epoch} sessions "
                 f"({labs})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "factor_vs_licks_running.png", dpi=105)
    plt.close(fig)

    # correlation table: per session r over trial bins, then the mean over sessions; + speed outside licks
    C = []
    for s in S:
        B = s["bins"]
        row = {}
        for k in range(R):
            for c in CORR_COLS:
                m = B[[f"f{k}", c]].dropna()
                row[(k, c)] = np.corrcoef(m[f"f{k}"], m[c])[0, 1] if len(m) > 50 and m[c].std() > 0 else np.nan
            row[(k, "speed_lickfree_session")] = s["r_speed_lickfree"][k]
        C.append(row)
    cols = CORR_COLS + ["speed_lickfree_session"]
    M = np.array([[np.nanmean([c[(k, col)] for c in C]) for col in cols] for k in range(R)])
    T = pd.DataFrame(M, columns=cols, index=[f"f{k}" for k in range(R)])
    T.to_csv(figure_layout.sidecar_for(out, "factor_correlations", ".csv"), float_format="%.3f")
    fig, ax = plt.subplots(figsize=(8, 0.45 * R + 1.5))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(R):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(R), T.index)
    fig.colorbar(im, ax=ax, label="r (mean over sessions)")
    ax.set_title(f"{a.set}: factor vs licks / treadmill ({len(S)} {a.epoch} sessions; 40 ms trial bins; last column: "
                 "whole session outside licks)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "factor_correlations.png", dpi=110)
    plt.close(fig)
    print(T.round(2).to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
