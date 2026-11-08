"""June (06/06-06/08) as a REFERENCE epoch for the FaceRhythm peak readout (11-08, Priya: June as a reference).

Same quantity as `face_rhythm_peak.py` (FINDINGS 5d): per session and frozen factor of ``master_pre_v1``, the PEAK over
cue 0 .. +4 s of the trial-averaged ABSOLUTE expression (temporal weight x FR normalizer D, engaged trials), per spout
side. June sessions come from the separate projection set ``o2/fr_cam4_june_20261108/factors/master_pre_v1_june/proj``
(`face_rhythm_june_project.py`), production sessions from ``master_pre_v1/proj``; both on the SAME frozen basis.

SCALE: June image px are ~1.13x August's (the camera moved; `face_rhythm_register` JUNE). The absolute expression
recovers the raw VQT magnitude, which is linear in displacement, so each June session's peak is divided by its face
scale sqrt(sx * sy) of the June -> August-template affine (``register/qc_june_fields.csv``) before comparing.

Every value is log2(peak) minus the mean log2 peak of the animal's OWN August pre sessions (Priya 11-07); the change
per epoch (june, acute, subacute, chronic) is `analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted,
95 % interval), Holm over the 12 factors within each epoch x side. ``--unscaled``: skip the scale division (shows how
much the camera scale alone moves June).

    python scripts/face_rhythm_june_reference.py [--split side|none] [--unscaled]
-> ``<dlc>/face_rhythm/cam4/readouts/master_pre_v1/june_reference/``: june_peak_sessions[_sfx].csv,
   june_peak_stats[_sfx].csv, june_peak_reference[_sfx].png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import stats as frs  # noqa: E402

EPOCHS = ("pre", "june", "acute", "subacute", "chronic")
JUNE_BUNDLE = "fr_cam4_june_20261108"


def june_scales(rv) -> dict[str, float]:
    """{label PS9x_MMDD: face scale sqrt(sx * sy)} of the June -> August-template face affine."""
    from wfield_local import face_rhythm_outputs as fro
    Q = pd.read_csv(frc.readouts_dir(fro.SET, rv).parent.parent / "register" / "qc_june_fields.csv")
    return {frc.label(s.split("_")[0], s.split("_")[1]): float(np.sqrt(x * y))
            for s, x, y in zip(Q.session, Q.affine_sx, Q.affine_sy)}


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import face_rhythm_peak as fp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="side", choices=("none", "side"))
    ap.add_argument("--window", type=float, nargs=2, default=(0.0, 4.0))
    ap.add_argument("--unscaled", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, fro.BUNDLE, fro.SET)
    jdir = oi.bundle_dir(rv, JUNE_BUNDLE) / "factors" / f"{fro.SET}_june"
    out = frc.readouts_dir(fro.SET, rv) / "june_reference"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    sfx = ("" if a.split == "none" else f"_{a.split}") + ("_unscaled" if a.unscaled else "")
    names = frc.factor_names(fdir)

    # production + June sessions through the SAME peak code (sorted union, rule 9)
    orig = frc.sessions
    frc.sessions = lambda d, sub="proj": pd.concat([orig(fdir, sub), orig(jdir, sub)], ignore_index=True) \
        .sort_values("key", kind="stable").reset_index(drop=True)
    try:
        S = fp.peaks(fdir, a.split, tuple(a.window))
    finally:
        frc.sessions = orig
    scale = {} if a.unscaled else june_scales(rv)
    jm = S.epoch == "june"
    S.loc[jm, "peak"] = S.loc[jm, "peak"] / S.loc[jm, "label"].map(scale).fillna(1.0)
    if not a.unscaled:
        assert S.loc[jm, "label"].map(scale).notna().all(), "June session without a face scale"
    S["log2_vs_pre"] = frs.log2_vs_pre(S, "peak", ["animal", "group", "factor"])
    S["name"] = [names[k] for k in S.factor]
    S.to_csv(figure_layout.sidecar_for(out, f"june_peak_sessions{sfx}", ".csv"), index=False, float_format="%.5g")

    rng = np.random.default_rng(0)
    groups = fp.display_groups(S.group.unique())
    rows = []
    for g in groups:                                     # fixed loop order -> reproducible draws
        for k in range(len(names)):
            d = S[(S.group == g) & (S.factor == k)]
            for r in frs.epoch_change(d, "log2_vs_pre", rng, a.n_boot, epochs=EPOCHS[1:]):
                rows.append({"group": g, "factor": k, "name": names[k], **r})
    T = frs.add_holm(pd.DataFrame(rows), ["epoch", "group"])
    T.to_csv(figure_layout.sidecar_for(out, f"june_peak_stats{sfx}", ".csv"), index=False, float_format="%.4g")

    pd.set_option("display.width", 250)
    W = T.pivot_table(index=["group", "factor", "name"], columns="epoch", values="delta").reindex(columns=EPOCHS[1:])
    M = T.pivot_table(index=["group", "factor", "name"], columns="epoch", values="mark", aggfunc="first") \
        .reindex(columns=EPOCHS[1:])
    print(f"log2 peak change vs own August pre ({'unscaled' if a.unscaled else 'June / face scale'}); * = 95 % CI "
          "excludes 0, ** = also Holm p < 0.05")
    print((W.round(2).astype(str) + M.fillna("")).to_string())
    ratio = (W["june"].abs() / W["acute"].abs()).replace([np.inf], np.nan)
    print(f"median |june| / |acute| over factor x side: {ratio.median():.2f}")

    # figure: per side, factors as rows of dots -- june vs acute vs chronic deltas with intervals
    ec = {**frc.epoch_colors(), "june": "0.55"}
    side_txt = " of each spout side (contra = right spouts; lesions left)" if a.split == "side" else ""
    scale_txt = ("June peaks NOT scale-corrected." if a.unscaled else
                 "June peaks divided by the session face scale sqrt(sx sy) of the June -> August affine, since June "
                 "image px are larger.")
    with frf.style():
        fig, axs = frf.subplots(1, len(groups), figsize=(4.2 * len(groups) + 1.5, 0.42 * len(names) + 1.6),
                                squeeze=False, sharey=True)
        for j, g in enumerate(groups):
            ax = axs[0, j]
            for i, ep in enumerate(("june", "acute", "chronic")):
                d = T[(T.group == g) & (T.epoch == ep)].sort_values("factor")
                y = d.factor.to_numpy() + (i - 1) * 0.25
                ax.errorbar(d.delta, y, xerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], fmt="o", ms=3.5, lw=0.9,
                            color=ec[ep], label=ep, mfc="white" if ep == "june" else ec[ep])
                for yy, x, mk in zip(y, d.ci_hi, d.mark):
                    if mk:
                        ax.text(x + 0.03, yy, mk, va="center", fontsize=7, color=ec[ep])
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"{g} spouts" if g != "all" else "all trials", fontsize=10)
            ax.set_xlabel("log2 peak change vs own August pre")
        axs[0, 0].set_yticks(range(len(names)))
        axs[0, 0].set_yticklabels([f"{k} {n}" for k, n in enumerate(names)], fontsize=8)
        axs[0, 0].invert_yaxis()
        frf.legend_outside(axs[0, -1])
        fig.suptitle("FaceRhythm factor peaks: June (training stage) vs acute / chronic, each vs own August pre")
        frf.save(fig, out / f"june_peak_reference{sfx}.png", f"""
Frozen pre-stroke factor set master_pre_v1 (rank 12, fit on August pre sessions only); June sessions (06/06-06/08, 3 per
animal, never in the fit) projected on the same basis. Per session and factor: PEAK over cue {a.window[0]:g} .. +{a.window[1]:g} s
of the trial-averaged absolute expression (weight x FR normalizer D), engaged trials{side_txt}, cam4 FaceRhythm, 25 Hz grid.
{scale_txt}
Value = log2(peak) - mean log2 peak of the animal's OWN August pre sessions. Dot = change vs pre by the paired animals ->
sessions bootstrap (animal-weighted, {a.n_boot} draws), bar = 95 % interval; * = interval excludes 0, ** = also Holm
p < 0.05 over the 12 factors within each epoch x side. June = open gray; acute / chronic = epoch colors.""")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
